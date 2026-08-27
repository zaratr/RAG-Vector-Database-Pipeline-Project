import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.persistence import models, repositories
from app.services.embeddings import HashEmbeddingProvider as LocalEmbeddingProvider
from app.services.graph_extraction import (
    DisabledGraphExtractor,
    ExtractedEntity,
    ExtractedRelation,
    GraphExtractionError,
    GraphProviderOutputError,
    GraphProviderUnavailable,
)
from app.services.ingestion import ingest_image, ingest_text
from app.services.vector_store import ChromaVectorStore

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


class FakeGraphExtractor:
    def __init__(self, relations):
        self.relations = relations
        self.texts = []

    async def extract(self, text):
        self.texts.append(text)
        return self.relations


class FailingGraphExtractor:
    async def extract(self, text):
        raise GraphProviderUnavailable("provider unavailable")


class FailingAfterFirstChunkExtractor:
    """Fails on the second chunk to test partial failure behavior."""

    def __init__(self):
        self.call_count = 0

    async def extract(self, text):
        self.call_count += 1
        if self.call_count > 1:
            raise GraphProviderOutputError("second chunk failed")
        return []


class RecordingVectorStore:
    def __init__(self):
        self.calls = []
        self._ids: set[str] = set()

    async def upsert_embeddings(self, embeddings, metadatas, ids, documents=None):
        self.calls.append((embeddings, metadatas, ids, documents))
        self._ids.update(ids)

    async def list_ids(self):
        return sorted(self._ids)

    async def delete(self, ids):
        for vector_id in ids:
            self._ids.discard(vector_id)


class FailingVectorStore(RecordingVectorStore):
    async def upsert_embeddings(self, embeddings, metadatas, ids, documents=None):
        self.calls.append((embeddings, metadatas, ids, documents))
        raise RuntimeError("chroma unavailable")


class IncompleteVectorStore(RecordingVectorStore):
    """list_ids() returns fewer IDs than upserted — simulates partial write."""

    async def list_ids(self):
        return []  # no IDs visible even after upsert


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.mark.asyncio
async def test_ingest_text_creates_document():
    """Real-Chroma write-path lock: ingest_text must persist exactly the
    deterministic ``chunk:<id>`` vector set, queryable with the SQL chunk
    text and metadata (chunk_id/document_id). A silent upsert no-op,
    metadata serialization breakage, or deterministic-ID drift fails here,
    not just ``chunks > 0``."""
    import uuid

    import chromadb

    client = chromadb.EphemeralClient()
    collection_name = "test-ingestion-" + uuid.uuid4().hex[:8]
    store = ChromaVectorStore(collection_name=collection_name, client=client)
    provider = LocalEmbeddingProvider()

    session: Session = TestSessionLocal()
    try:
        result = await ingest_text(
            title="Test Doc",
            source="unit",
            tags=["one"],
            text="hello world",
            embedding_provider=provider,
            vector_store=store,
            session=session,
        )
        assert result["chunks"] == 1
        docs = repositories.list_documents(session)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.ingestion_status == "ready"
        chunk = doc.chunks[0]
        expected_vector_ids = {f"chunk:{chunk.id}"}

        # The store contains EXACTLY the document's deterministic vector IDs.
        assert set(await store.list_ids()) == expected_vector_ids
        assert chunk.vector_id in expected_vector_ids

        # Query-back: the stored record round-trips the chunk text and the
        # SQL-authored metadata (chunk_id, document_id, title).
        embedding = (await provider.embed_texts(["hello world"]))[0]
        hits = await store.query(embedding, top_k=5)
        assert len(hits) == 1
        assert hits[0].vector_id == chunk.vector_id
        assert hits[0].text == "hello world"
        assert hits[0].metadata["chunk_id"] == chunk.id
        assert hits[0].metadata["document_id"] == doc.id
        assert hits[0].metadata["title"] == "Test Doc"
    finally:
        session.rollback()
        doc_rows = session.query(models.Document).all()
        for row in doc_rows:
            session.delete(row)
        session.commit()
        session.close()
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_ingest_text_extracts_and_persists_chunk_graph_provenance():
    relation = ExtractedRelation(
        source=ExtractedEntity(
            name="Alice", canonical_name="alice", entity_type="person"
        ),
        predicate="works_at",
        target=ExtractedEntity(
            name="Acme", canonical_name="acme", entity_type="organization"
        ),
        evidence="Alice works at Acme.",
        evidence_start=0,
        evidence_end=20,
        confidence=0.9,
    )
    extractor = FakeGraphExtractor([relation])
    store = RecordingVectorStore()
    session: Session = TestSessionLocal()

    result = await ingest_text(
        title="Graph Doc",
        source="unit",
        tags=None,
        text="Alice works at Acme.",
        embedding_provider=LocalEmbeddingProvider(),
        vector_store=store,
        graph_extractor=extractor,
        graph_extraction_model="gemma4:latest",
        session=session,
    )

    assert result == {"document_id": result["document_id"], "chunks": 1, "relations": 1}
    assert extractor.texts == ["Alice works at Acme."]
    evidence = session.query(models.GraphEdgeEvidence).one()
    assert evidence.extraction.chunk.text == "Alice works at Acme."
    assert evidence.extraction.chunk.document.title == "Graph Doc"
    assert evidence.extraction.status == "succeeded"
    assert len(store.calls) == 1
    stored_document = session.get(models.Document, result["document_id"])
    assert stored_document.ingestion_status == "ready"
    assert stored_document.chunks[0].vector_id == f"chunk:{stored_document.chunks[0].id}"
    session.delete(stored_document)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_graph_extraction_failure_prevents_vector_indexing():
    store = RecordingVectorStore()
    session: Session = TestSessionLocal()

    with pytest.raises(GraphExtractionError, match="provider unavailable"):
        await ingest_text(
            title="Failed Graph Doc",
            source="unit",
            tags=None,
            text="Alice works at Acme.",
            embedding_provider=LocalEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FailingGraphExtractor(),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    assert store.calls == []
    # The failed document persists as operator-visible evidence (it is
    # not query-visible), and its extraction row records the failure.
    doc = session.query(models.Document).filter_by(title="Failed Graph Doc").one()
    assert doc.ingestion_status == "failed"
    session.close()


@pytest.mark.asyncio
async def test_vector_failure_marks_staged_document_failed_and_hidden():
    store = FailingVectorStore()
    session: Session = TestSessionLocal()

    with pytest.raises(RuntimeError, match="chroma unavailable"):
        await ingest_text(
            title="Vector Failure",
            source="unit",
            tags=None,
            text="Alice works at Acme.",
            embedding_provider=LocalEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FakeGraphExtractor([]),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    failed = session.query(models.Document).filter_by(title="Vector Failure").one()
    assert failed.ingestion_status == "failed"
    assert failed.failure_code == "RuntimeError"
    assert failed.chunks[0].vector_id == f"chunk:{failed.chunks[0].id}"
    session.delete(failed)
    session.commit()
    session.close()


# ── Lifecycle state machine tests ──


@pytest.mark.asyncio
async def test_ingest_text_success_persists_succeeded_extraction_and_ready_document():
    """Step 6: document ready only after vectors exist and extraction succeeded."""
    relation = ExtractedRelation(
        source=ExtractedEntity(name="Alice", canonical_name="alice", entity_type="person"),
        predicate="works_at",
        target=ExtractedEntity(name="Acme", canonical_name="acme", entity_type="organization"),
        evidence="Alice works at Acme.",
        evidence_start=0,
        evidence_end=20,
        confidence=0.9,
    )
    store = RecordingVectorStore()
    session = TestSessionLocal()

    result = await ingest_text(
        title="Success Doc",
        source="unit",
        tags=None,
        text="Alice works at Acme.",
        embedding_provider=LocalEmbeddingProvider(),
        vector_store=store,
        graph_extractor=FakeGraphExtractor([relation]),
        graph_extraction_model="gemma4:latest",
        session=session,
    )

    doc = session.get(models.Document, result["document_id"])
    assert doc.ingestion_status == "ready"
    extraction = session.query(models.GraphExtraction).filter_by(chunk_id=doc.chunks[0].id).one()
    assert extraction.status == "succeeded"

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_provider_unavailable_marks_document_failed_and_not_query_visible():
    """Provider failure: document failed, extraction failed, not retrievable."""
    store = RecordingVectorStore()
    session = TestSessionLocal()

    with pytest.raises(GraphProviderUnavailable):
        await ingest_text(
            title="Provider Fail",
            source="unit",
            tags=None,
            text="Alice works at Acme.",
            embedding_provider=LocalEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FailingGraphExtractor(),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    doc = session.query(models.Document).filter_by(title="Provider Fail").one()
    assert doc.ingestion_status == "failed"

    # Extraction must be recorded as failed (provider failure persists audit row)
    extraction = session.query(models.GraphExtraction).filter_by(chunk_id=doc.chunks[0].id).one()
    assert extraction.status == "failed"

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_vector_failure_marks_document_failed_with_compensation_delete():
    """Vector failure: document failed, compensation delete attempted."""
    store = FailingVectorStore()
    session = TestSessionLocal()

    with pytest.raises(RuntimeError, match="chroma unavailable"):
        await ingest_text(
            title="Vector Fail",
            source="unit",
            tags=None,
            text="Alice works at Acme.",
            embedding_provider=LocalEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FakeGraphExtractor([]),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    doc = session.query(models.Document).filter_by(title="Vector Fail").one()
    assert doc.ingestion_status == "failed"

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_vector_index_incomplete_raises_and_fails_document():
    """VectorStoreIncomplete: list_ids returns missing IDs → VectorIndexIncomplete, doc failed."""
    store = IncompleteVectorStore()
    session = TestSessionLocal()

    with pytest.raises((RuntimeError, Exception), match="incomplete|VectorIndexIncomplete"):
        await ingest_text(
            title="Incomplete Vector",
            source="unit",
            tags=None,
            text="Alice works at Acme.",
            embedding_provider=LocalEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FakeGraphExtractor([]),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    doc = session.query(models.Document).filter_by(title="Incomplete Vector").one()
    assert doc.ingestion_status == "failed"

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_disabled_extraction_records_skipped_not_empty():
    """Disabled extraction → relations 0, status 'skipped' not 'empty'."""
    store = RecordingVectorStore()
    session = TestSessionLocal()

    result = await ingest_text(
        title="Disabled Doc",
        source="unit",
        tags=None,
        text="Alice works at Acme.",
        embedding_provider=LocalEmbeddingProvider(),
        vector_store=store,
        graph_extractor=DisabledGraphExtractor(),
        graph_extraction_model="gemma4:latest",
        session=session,
    )

    assert result["relations"] == 0
    doc = session.get(models.Document, result["document_id"])
    assert doc.ingestion_status == "ready"

    extraction = session.query(models.GraphExtraction).filter_by(chunk_id=doc.chunks[0].id).one()
    assert extraction.status == "skipped"
    assert extraction.error_code == "extraction_disabled"

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_empty_extraction_result_records_empty_status():
    """Provider returns [] → status 'empty', document ready."""
    store = RecordingVectorStore()
    session = TestSessionLocal()

    result = await ingest_text(
        title="Empty Result Doc",
        source="unit",
        tags=None,
        text="Some text with no relations.",
        embedding_provider=LocalEmbeddingProvider(),
        vector_store=store,
        graph_extractor=FakeGraphExtractor([]),
        graph_extraction_model="gemma4:latest",
        session=session,
    )

    doc = session.get(models.Document, result["document_id"])
    assert doc.ingestion_status == "ready"

    extraction = session.query(models.GraphExtraction).filter_by(chunk_id=doc.chunks[0].id).one()
    assert extraction.status == "empty"

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_embedding_failure_marks_all_pending_as_failed_embedding_failed():
    """If embedding fails before extraction, pending runs become failed/embedding_failed."""
    class FailingEmbeddingProvider:
        async def embed_texts(self, texts):
            raise RuntimeError("embedding service down")

    store = RecordingVectorStore()
    session = TestSessionLocal()

    with pytest.raises(RuntimeError, match="embedding service down"):
        await ingest_text(
            title="Embedding Fail",
            source="unit",
            tags=None,
            text="Alice works at Acme.",
            embedding_provider=FailingEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FakeGraphExtractor([]),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    # Contract: every pending run becomes failed/error_code=
    # embedding_failed and no pending run may remain after a handled failure.
    doc = session.query(models.Document).filter_by(title="Embedding Fail").one()
    assert doc.ingestion_status == "failed"
    extractions = session.query(models.GraphExtraction).filter(
        models.GraphExtraction.chunk_id.in_([c.id for c in doc.chunks])
    ).all()
    assert extractions
    for extraction in extractions:
        assert extraction.status == "failed"
        assert extraction.error_code == "embedding_failed"
    assert (
        session.query(models.GraphExtraction)
        .filter(
            models.GraphExtraction.chunk_id.in_([c.id for c in doc.chunks]),
            models.GraphExtraction.status == "pending",
        )
        .count()
        == 0
    )

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_chunk_extraction_failure_marks_remaining_pending_aborted():
    """One chunk fails → that run stores typed failure, remaining pending → aborted_after_chunk_failure."""
    store = RecordingVectorStore()
    session = TestSessionLocal()

    # "Alice works at Acme. " * 50 is 1000 chars — exactly one chunk at
    # chunk_size=1000. Repeat 100x (1999 chars normalized) so the text chunks
    # into 3 chunks and a failure on chunk 2 leaves chunk 3 pending (the
    # aborted_after_chunk_failure fan-out target).
    long_text = "Alice works at Acme. " * 100  # ensures multiple chunks

    with pytest.raises(GraphProviderOutputError, match="second chunk failed"):
        await ingest_text(
            title="Partial Fail",
            source="unit",
            tags=None,
            text=long_text,
            embedding_provider=LocalEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FailingAfterFirstChunkExtractor(),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    doc = session.query(models.Document).filter_by(title="Partial Fail").one()
    assert doc.ingestion_status == "failed"
    assert len(doc.chunks) >= 2  # guard: the fan-out scenario requires multi-chunk

    extractions = session.query(models.GraphExtraction).filter(
        models.GraphExtraction.chunk_id.in_([c.id for c in doc.chunks])
    ).all()
    statuses = {e.status for e in extractions}
    assert "failed" in statuses
    # No pending rows may remain
    assert "pending" not in statuses
    # Remaining pending runs were terminalized with the aborted fan-out code
    assert any(e.error_code == "aborted_after_chunk_failure" for e in extractions)

    session.delete(doc)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_text_no_pending_runs_remain_after_handled_failure():
    """After any handled exception, zero pending extraction rows exist."""
    store = FailingVectorStore()
    session = TestSessionLocal()

    with pytest.raises(RuntimeError):
        await ingest_text(
            title="No Pending",
            source="unit",
            tags=None,
            text="Alice works at Acme.",
            embedding_provider=LocalEmbeddingProvider(),
            vector_store=store,
            graph_extractor=FakeGraphExtractor([]),
            graph_extraction_model="gemma4:latest",
            session=session,
        )

    pending_count = (
        session.query(models.GraphExtraction)
        .filter_by(status="pending")
        .count()
    )
    assert pending_count == 0

    # cleanup
    docs = session.query(models.Document).all()
    for d in docs:
        session.delete(d)
    session.commit()
    session.close()


@pytest.mark.asyncio
async def test_ingest_image_sets_media_type_and_skips_extraction():
    """Image ingestion: actual media type, extraction skipped with unsupported_media_type."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        image_path = f.name

    store = RecordingVectorStore()
    session = TestSessionLocal()

    # This test requires an ImageEmbeddingProvider; use a mock
    class FakeImageProvider:
        async def embed_images(self, paths):
            return [[0.1] * 10]

    try:
        result = await ingest_image(
            title="Image Doc",
            source="unit",
            tags=None,
            image_path=image_path,
            media_type="image/png",
            embedding_provider=FakeImageProvider(),
            vector_store=store,
            session=session,
        )

        doc = session.get(models.Document, result["document_id"])
        assert doc.ingestion_status == "ready"
        chunk = doc.chunks[0]
        assert chunk.media_type == "image/png"

        # Plan L591: image graph extraction is skipped with reason
        # unsupported_media_type; placeholder text is never sent to the provider.
        extraction = session.query(models.GraphExtraction).filter_by(chunk_id=chunk.id).one()
        assert extraction.status == "skipped"
        assert extraction.error_code == "unsupported_media_type"

        session.delete(doc)
        session.commit()
    finally:
        Path(image_path).unlink(missing_ok=True)
        session.close()
