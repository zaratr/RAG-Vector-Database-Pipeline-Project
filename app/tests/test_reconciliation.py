"""Tests for deterministic ingestion reconciliation."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.services.reconciliation import reconcile_ingestion


class FakeEmbeddingProvider:
    async def embed_texts(self, texts):
        return [[float(len(text))] for text in texts]


class FakeVectorStore:
    def __init__(self):
        self.rows = {
            "chunk:staged": {},
            "chunk:failed": {},
            "uuid-aliasing-ready": {"metadata": {"chunk_id": 1}},
            "orphan": {"metadata": {"chunk_id": 999999}},
        }
        self.deleted = []

    async def list_ids(self):
        return list(self.rows)

    async def delete(self, ids):
        self.deleted.extend(ids)
        for item_id in ids:
            self.rows.pop(item_id, None)

    async def upsert_embeddings(self, embeddings, metadatas, ids, documents=None):
        for index, item_id in enumerate(ids):
            self.rows[item_id] = {
                "embedding": embeddings[index],
                "metadata": metadatas[index],
                "document": documents[index],
            }


def _add_document(session, title, status, vector_id):
    document = models.Document(title=title, ingestion_status=status)
    session.add(document)
    session.flush()
    chunk = models.Chunk(
        document_id=document.id,
        index=0,
        text=f"{title} text",
        start_offset=0,
        end_offset=len(title) + 5,
        vector_id=vector_id,
    )
    session.add(chunk)
    session.flush()
    return document, chunk


@pytest.mark.asyncio
async def test_reconciliation_hides_nonready_and_idempotently_repairs_ready_vectors():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    ready, ready_chunk = _add_document(session, "Ready", "ready", None)
    staged, staged_chunk = _add_document(
        session, "Staged", "staged", "chunk:staged"
    )
    failed, failed_chunk = _add_document(
        session, "Failed", "failed", "chunk:failed"
    )
    session.commit()
    store = FakeVectorStore()

    first = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        batch_size=1,
    )
    second = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        batch_size=10,
    )

    assert first == {
        "nonready_vectors_deleted": 2,
        "orphan_vectors_deleted": 2,
        "staged_documents_failed": 1,
        "pending_extractions_failed": 0,
        "ready_chunks_upserted": 1,
    }
    assert second["staged_documents_failed"] == 0
    assert second["nonready_vectors_deleted"] == 0
    assert second["orphan_vectors_deleted"] == 0
    assert staged.ingestion_status == "failed"
    assert staged.failure_code == "reconciled_incomplete"
    assert store.deleted == [
        "chunk:failed",
        "chunk:staged",
        "orphan",
        "uuid-aliasing-ready",
    ]
    assert list(store.rows) == [f"chunk:{ready_chunk.id}"]
    assert ready_chunk.vector_id == f"chunk:{ready_chunk.id}"
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_terminalizes_pending_extractions_with_completed_at():
    """DEFECT-8 regression: staged doc with pending extraction must be
    terminalized to failed WITH completed_at set, not crash with NameError."""
    from app.services.reconciliation import reconcile_ingestion

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    # Create a staged document with a chunk that has a pending extraction.
    document = models.Document(title="Staged with extraction", ingestion_status="staged")
    session.add(document)
    session.flush()
    chunk = models.Chunk(
        document_id=document.id,
        index=0,
        text="Some text.",
        start_offset=0,
        end_offset=10,
        media_type="text/plain",
        vector_id=f"chunk:{document.id}:0",
    )
    session.add(chunk)
    session.flush()
    extraction = models.GraphExtraction(
        chunk_id=chunk.id,
        provider="ollama",
        model="gemma4:latest",
        prompt_version="graph-v1",
        schema_version="graph-relations-v1",
        status="pending",
        input_sha256="a" * 64,
        attempt_count=1,
        is_identity_owner=True,
    )
    session.add(extraction)
    session.commit()

    store = FakeVectorStore()
    report = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        batch_size=10,
    )

    # The pending extraction must be terminalized.
    assert report["pending_extractions_failed"] == 1
    assert report["staged_documents_failed"] == 1

    # The extraction must be failed WITH completed_at set (lifecycle invariant).
    session.refresh(extraction)
    assert extraction.status == "failed"
    assert extraction.error_code == "reconciled_incomplete"
    assert extraction.completed_at is not None

    session.close()
