"""Ingestion edge cases: duplicate ingestion, embedding-failure containment,
and unicode/boundary chunk-offset invariants.

Every test drives the production ``ingest_text`` (or ``chunk_text`` for the
pure offset arithmetic) against a real ephemeral Chroma collection and real
SQL, so each one fails if the ingestion pipeline regresses.
"""
from __future__ import annotations

import uuid

import chromadb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.persistence import models
from app.services.chunking import chunk_text
from app.services.embeddings import HashEmbeddingProvider
from app.services.ingestion import VectorIndexIncomplete, ingest_text
from app.services.vector_store import ChromaVectorStore

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


class _DisposableStore:
    """A real ephemeral ChromaVectorStore with a unique collection name."""

    def __enter__(self):
        self.client = chromadb.EphemeralClient()
        self.name = "test-ingestion-edges-" + uuid.uuid4().hex[:8]
        self.store = ChromaVectorStore(collection_name=self.name, client=self.client)
        return self.store

    def __exit__(self, *exc_info):
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        return False


def _purge_documents(session: Session) -> None:
    session.rollback()
    for document in session.query(models.Document).all():
        session.delete(document)
    session.commit()


# ---------------------------------------------------------------------------
# Duplicate ingestion: same text+title twice must not clobber either document
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_ingestion_creates_distinct_documents_and_vectors():
    with _DisposableStore() as store:
        provider = HashEmbeddingProvider()
        session: Session = TestSessionLocal()
        try:
            result_one = await ingest_text(
                title="Duplicate Doc", source="unit", tags=None,
                text="deterministic duplicate payload",
                embedding_provider=provider, vector_store=store, session=session,
            )
            result_two = await ingest_text(
                title="Duplicate Doc", source="unit", tags=None,
                text="deterministic duplicate payload",
                embedding_provider=provider, vector_store=store, session=session,
            )

            assert result_one["document_id"] != result_two["document_id"]

            doc_one = session.get(models.Document, result_one["document_id"])
            doc_two = session.get(models.Document, result_two["document_id"])
            assert doc_one.ingestion_status == "ready"
            assert doc_two.ingestion_status == "ready"

            ids_one = {chunk.vector_id for chunk in doc_one.chunks}
            ids_two = {chunk.vector_id for chunk in doc_two.chunks}
            # Deterministic chunk:<id> IDs are per-chunk: no shared vector ID,
            # no clobbering — and exactly the union is present in the store.
            assert ids_one and ids_two
            assert ids_one.isdisjoint(ids_two)
            assert all(vector_id.startswith("chunk:") for vector_id in ids_one | ids_two)
            assert set(await store.list_ids()) == ids_one | ids_two

            # Both documents' chunks are retrievable from the shared store.
            embedding = (
                await provider.embed_texts(["deterministic duplicate payload"])
            )[0]
            hits = await store.query(embedding, top_k=5)
            assert len(hits) == len(ids_one | ids_two)
            hit_chunk_ids = {hit.metadata["chunk_id"] for hit in hits}
            assert {chunk.id for chunk in doc_one.chunks} <= hit_chunk_ids
            assert {chunk.id for chunk in doc_two.chunks} <= hit_chunk_ids
        finally:
            _purge_documents(session)
            session.close()


# ---------------------------------------------------------------------------
# Embedding failure containment: no staged document or orphan vectors remain
# ---------------------------------------------------------------------------


class EmptyListEmbeddingProvider:
    """Provider that returns zero embeddings regardless of chunk count."""

    async def embed_texts(self, texts):
        return []


class MismatchedCountEmbeddingProvider:
    """Provider that returns one embedding regardless of chunk count."""

    async def embed_texts(self, texts):
        return [[0.1] * 8]


async def _ingest_and_capture_failure(store, provider, text):
    """Run ingest_text expecting it to raise; return the live session plus
    the captured exception so the caller can assert containment state."""
    session: Session = TestSessionLocal()
    try:
        with pytest.raises(Exception) as excinfo:
            await ingest_text(
                title="Embedding Failure", source="unit", tags=None, text=text,
                embedding_provider=provider, vector_store=store, session=session,
            )
        return session, excinfo
    except Exception:
        session.close()
        raise


def _assert_failure_containment(session, title):
    """After a handled ingestion failure the document is failed with a code
    (operator-visible, never query-visible) and no extraction row for it
    remains pending. Returns the failed document."""
    document = session.query(models.Document).filter_by(title=title).one()
    assert document.ingestion_status == "failed"
    assert document.failure_code
    for chunk in document.chunks:
        for extraction in chunk.graph_extractions:
            assert extraction.status != "pending"
    return document


@pytest.mark.asyncio
async def test_empty_embedding_list_fails_document_and_leaves_no_vectors():
    with _DisposableStore() as store:
        session, excinfo = await _ingest_and_capture_failure(
            store, EmptyListEmbeddingProvider(), "one chunk of text"
        )
        try:
            # An empty embedding list is rejected by the store write (Chroma
            # refuses empty upserts) or by the completeness check — either way
            # the failure is contained below.
            assert isinstance(excinfo.value, (ValueError, VectorIndexIncomplete))
            _assert_failure_containment(session, "Embedding Failure")
            # Containment: nothing entered the vector store.
            assert await store.list_ids() == []
        finally:
            _purge_documents(session)
            session.close()


@pytest.mark.asyncio
async def test_embedding_count_mismatch_fails_document_and_compensates():
    """Multiple chunks but one embedding vector: the store write fails, the
    document is marked failed, and compensation leaves no orphan vectors."""
    text = " ".join(f"paragraph {i} with plenty of unique content" for i in range(120))
    assert len(chunk_text(" ".join(text.split()))) > 1  # precondition: multi-chunk

    with _DisposableStore() as store:
        session, excinfo = await _ingest_and_capture_failure(
            store, MismatchedCountEmbeddingProvider(), text
        )
        try:
            assert not isinstance(excinfo.value, AssertionError)
            _assert_failure_containment(session, "Embedding Failure")
            assert await store.list_ids() == []
        finally:
            _purge_documents(session)
            session.close()


# ---------------------------------------------------------------------------
# Chunk-offset invariants: boundary arithmetic and unicode codepoint offsets
# ---------------------------------------------------------------------------


def test_chunk_text_exact_boundary_single_chunk():
    """len(text) == chunk_size produces exactly one chunk covering [0, end]."""
    text = "a" * 1000
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert (chunks[0]["start_offset"], chunks[0]["end_offset"]) == (0, 1000)
    assert text[chunks[0]["start_offset"]:chunks[0]["end_offset"]] == chunks[0]["text"]


def test_chunk_text_one_past_boundary_second_chunk_starts_at_overlap():
    """len(text) == chunk_size + 1: the second chunk starts at end-overlap,
    and every chunk's text is exactly text[start:end]."""
    text = "b" * 1001
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert chunks[0]["start_offset"] == 0
    assert chunks[0]["end_offset"] == 1000
    assert chunks[1]["start_offset"] == chunks[0]["end_offset"] - 200
    assert chunks[1]["end_offset"] == 1001
    for chunk in chunks:
        assert text[chunk["start_offset"]:chunk["end_offset"]] == chunk["text"]
        assert len(chunk["text"]) == chunk["end_offset"] - chunk["start_offset"]


def test_chunk_text_offsets_always_slice_to_chunk_text():
    """Across many lengths the offset invariants hold: contiguous progress,
    no infinite loop, text[start:end] == chunk text, overlap honored."""
    for size in (1, 199, 200, 201, 999, 1000, 1001, 1200, 2500):
        text = "x" * size
        chunks = chunk_text(text)
        assert chunks, size
        for previous, current in zip(chunks, chunks[1:]):
            assert current["start_offset"] == previous["end_offset"] - 200
        for index, chunk in enumerate(chunks):
            assert chunk["index"] == index
            assert text[chunk["start_offset"]:chunk["end_offset"]] == chunk["text"]
        assert chunks[-1]["end_offset"] == len(text)


def test_chunk_text_unicode_offsets_are_python_codepoints():
    """Offsets address Python codepoints: astral-plane emoji (2 UTF-16 units,
    4 UTF-8 bytes) count as one, and slicing reproduces chunk text exactly."""
    text = ("中文测试 🎉 héllo wörld " * 80).strip()
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert text[chunk["start_offset"]:chunk["end_offset"]] == chunk["text"]
    assert "🎉" in chunks[0]["text"]


@pytest.mark.asyncio
async def test_ingest_text_unicode_document_offsets_match_stored_text():
    """Service level: a multi-chunk unicode document's stored offsets satisfy
    normalized_text[start:end] == chunk.text for every persisted chunk."""
    sentence = "中文段落 🎉 héllo wörld — naïve façade. "
    text = (sentence * 40).strip()  # single-spaced, > 1000 chars
    normalized = " ".join(text.split())
    assert len(normalized) > 1000

    with _DisposableStore() as store:
        session: Session = TestSessionLocal()
        try:
            result = await ingest_text(
                title="Unicode Doc", source="unit", tags=None, text=text,
                embedding_provider=HashEmbeddingProvider(),
                vector_store=store, session=session,
            )
            assert result["chunks"] > 1

            document = session.get(models.Document, result["document_id"])
            assert document.ingestion_status == "ready"
            for chunk in document.chunks:
                assert (
                    normalized[chunk.start_offset:chunk.end_offset] == chunk.text
                ), f"chunk {chunk.index} offsets do not slice to its stored text"
            assert "🎉" in document.chunks[0].text
        finally:
            _purge_documents(session)
            session.close()
