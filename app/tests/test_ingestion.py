import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.persistence import models, repositories
from app.services.embeddings import HashEmbeddingProvider as LocalEmbeddingProvider
from app.services.graph_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    GraphExtractionError,
)
from app.services.ingestion import ingest_text
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
        raise GraphExtractionError("provider unavailable")


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


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.mark.asyncio
async def test_ingest_text_creates_document():
    provider = LocalEmbeddingProvider()
    store = ChromaVectorStore(collection_name="test-ingestion")

    session: Session = TestSessionLocal()
    result = await ingest_text(
        title="Test Doc",
        source="unit",
        tags=["one"],
        text="hello world",
        embedding_provider=provider,
        vector_store=store,
        session=session,
    )
    assert result["chunks"] > 0
    docs = repositories.list_documents(session)
    assert len(docs) == 1
    session.delete(docs[0])
    session.commit()
    session.close()


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
    # 10A.4: the failed document persists as operator-visible evidence (it is
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
