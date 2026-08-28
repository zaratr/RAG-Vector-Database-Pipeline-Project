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
    def __init__(self, ids=None):
        self.rows = {id_: {} for id_ in (ids or [])}
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
                "document": documents[index] if documents else None,
            }


def _add_document(session, title, status, vector_id, *, chunk_text=None):
    document = models.Document(title=title, ingestion_status=status)
    session.add(document)
    session.flush()
    text = chunk_text or f"{title} text"
    chunk = models.Chunk(
        document_id=document.id,
        index=0,
        text=text,
        start_offset=0,
        end_offset=len(text),
        vector_id=vector_id,
        media_type="text/plain",
    )
    session.add(chunk)
    session.flush()
    return document, chunk


def _add_extraction(session, chunk_id, status="succeeded"):
    # Fixture adaptation: the ck_graph_extractions_lifecycle
    # CHECK requires completed_at NOT NULL for terminal rows and
    # error_code NOT NULL for failed rows, so the helper fills those where the
    # status requires them. Every crash-matrix assertion is unaffected.
    from datetime import datetime, timezone

    completed_at = (
        None if status == "pending" else datetime.now(timezone.utc).replace(tzinfo=None)
    )
    error_code = "provider_failure" if status == "failed" else None
    extraction = models.GraphExtraction(
        chunk_id=chunk_id,
        provider="ollama",
        model="gemma4",
        prompt_version="graph-v1",
        schema_version="graph-relations-v1",
        status=status,
        input_sha256="a" * 64,
        attempt_count=1,
        is_identity_owner=1,
        completed_at=completed_at,
        error_code=error_code,
    )
    session.add(extraction)
    session.flush()
    return extraction


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
    store = FakeVectorStore(
        ids=["chunk:staged", "chunk:failed", "uuid-aliasing-ready", "orphan"]
    )

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


# ── Crash matrix tests ──


@pytest.mark.asyncio
async def test_reconciliation_crash_before_staged_commit_is_noop():
    """Matrix: before staged commit → no document → no-op."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    store = FakeVectorStore(ids=[])

    result = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result["staged_documents_failed"] == 0
    assert result["nonready_vectors_deleted"] == 0
    assert result["orphan_vectors_deleted"] == 0

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_crash_after_staged_commit_fails_document_and_pending_extractions():
    """Matrix: staged commit before external work → document failed, pending→failed."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Staged", "staged", "chunk:staged")
    _add_extraction(session, chunk.id, status="pending")
    session.commit()

    store = FakeVectorStore(ids=["chunk:staged"])

    result = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result["staged_documents_failed"] == 1
    assert result["pending_extractions_failed"] == 1

    session.refresh(doc)
    assert doc.ingestion_status == "failed"
    assert doc.failure_code == "reconciled_incomplete"

    extraction = session.query(models.GraphExtraction).filter_by(chunk_id=chunk.id).one()
    assert extraction.status == "failed"

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_crash_after_extraction_before_vector_preserves_evidence():
    """Matrix: extraction terminal, before vector write → doc failed, evidence preserved."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Extracted No Vector", "staged", "chunk:ext")
    _add_extraction(session, chunk.id, status="succeeded")
    session.commit()

    store = FakeVectorStore(ids=[])

    result = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result["staged_documents_failed"] == 1
    session.refresh(doc)
    assert doc.ingestion_status == "failed"

    # Evidence preserved
    extraction = session.query(models.GraphExtraction).filter_by(chunk_id=chunk.id).one()
    assert extraction.status == "succeeded"

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_crash_during_partial_vector_write_deletes_all_doc_ids():
    """Matrix: partial/all vector write before ready commit → delete all doc IDs, doc failed."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Partial Vector", "staged", "chunk:partial")
    _add_extraction(session, chunk.id, status="succeeded")
    session.commit()

    store = FakeVectorStore(ids=["chunk:partial"])

    result = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result["nonready_vectors_deleted"] == 1
    assert "chunk:partial" in store.deleted

    session.refresh(doc)
    assert doc.ingestion_status == "failed"

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_ready_document_idempotently_upserts_vectors():
    """Matrix: ready commit → preserve and idempotently upsert."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Ready", "ready", "chunk:ready")
    _add_extraction(session, chunk.id, status="succeeded")
    session.commit()

    store = FakeVectorStore(ids=[])

    result = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result["ready_chunks_upserted"] == 1
    assert result["staged_documents_failed"] == 0

    session.refresh(doc)
    assert doc.ingestion_status == "ready"

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_second_run_has_zero_mutation_counters():
    """After convergence, second run has all mutation counters at zero (except ready upsert)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Ready2", "ready", "chunk:ready2")
    _add_extraction(session, chunk.id, status="succeeded")
    session.commit()

    store = FakeVectorStore(ids=[])

    first = await reconcile_ingestion(
        session=session, embedding_provider=FakeEmbeddingProvider(), vector_store=store,
    )
    second = await reconcile_ingestion(
        session=session, embedding_provider=FakeEmbeddingProvider(), vector_store=store,
    )

    assert second["nonready_vectors_deleted"] == 0
    assert second["orphan_vectors_deleted"] == 0
    assert second["staged_documents_failed"] == 0
    assert second["pending_extractions_failed"] == 0
    assert second["ready_chunks_upserted"] == 1  # idempotent upsert always runs

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_never_promotes_staged_to_ready():
    """Reconciliation never promotes staged → ready."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Never Ready", "staged", "chunk:never")
    session.commit()

    store = FakeVectorStore(ids=["chunk:never"])

    await reconcile_ingestion(
        session=session, embedding_provider=FakeEmbeddingProvider(), vector_store=store,
    )

    session.refresh(doc)
    assert doc.ingestion_status == "failed"  # never promoted to ready

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_never_re_runs_graph_extraction():
    """Reconciliation must not re-run graph extraction on any chunk."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "No Re-extract", "staged", "chunk:nore")
    ext = _add_extraction(session, chunk.id, status="failed")
    ext.error_code = "original_failure"
    session.commit()

    store = FakeVectorStore(ids=[])

    await reconcile_ingestion(
        session=session, embedding_provider=FakeEmbeddingProvider(), vector_store=store,
    )

    # The extraction status and error_code should remain unchanged (not re-run)
    session.refresh(ext)
    assert ext.status == "failed"
    assert ext.error_code == "original_failure"

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_compensation_delete_failure_then_recovery():
    """Compensation delete fails, then second reconciliation completes cleanup."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Comp Fail", "staged", "chunk:comp")
    session.commit()

    class FailOnceVectorStore(FakeVectorStore):
        def __init__(self):
            super().__init__(ids=["chunk:comp"])
            self._delete_should_fail = True

        async def delete(self, ids):
            if self._delete_should_fail:
                self._delete_should_fail = False
                raise RuntimeError("delete failed")
            await super().delete(ids)

    store = FailOnceVectorStore()

    # First reconciliation: delete fails but staged→failed still happens
    with pytest.raises(RuntimeError, match="delete failed"):
        await reconcile_ingestion(
            session=session, embedding_provider=FakeEmbeddingProvider(), vector_store=store,
        )

    session.refresh(doc)
    assert doc.ingestion_status == "failed"

    # Second reconciliation: delete succeeds now
    result = await reconcile_ingestion(
        session=session, embedding_provider=FakeEmbeddingProvider(), vector_store=store,
    )
    assert result["nonready_vectors_deleted"] >= 1

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_report_includes_pending_extractions_failed_counter():
    """Report must include 'pending_extractions_failed' key."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc, chunk = _add_document(session, "Counter Test", "staged", "chunk:counter")
    _add_extraction(session, chunk.id, status="pending")
    session.commit()

    store = FakeVectorStore(ids=["chunk:counter"])

    result = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert "pending_extractions_failed" in result
    assert result["pending_extractions_failed"] == 1

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_reconciliation_ready_only_retrieval_exclusion():
    """Failed documents are excluded from ready_chunks query during reconciliation."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    ready_doc, ready_chunk = _add_document(session, "Ready", "ready", None)
    failed_doc, failed_chunk = _add_document(session, "Failed", "failed", "chunk:failed")
    session.commit()

    store = FakeVectorStore(ids=["chunk:failed"])

    result = await reconcile_ingestion(
        session=session,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    # Only the ready chunk is upserted; failed chunk's vector is deleted
    assert result["ready_chunks_upserted"] == 1
    assert result["nonready_vectors_deleted"] == 1
    assert "chunk:failed" in store.deleted

    session.close()
    engine.dispose()
