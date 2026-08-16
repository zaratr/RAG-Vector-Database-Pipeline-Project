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


# ---------------------------------------------------------------------------
# 10C.4 extensions: safety-aware ingestion lifecycle
# ---------------------------------------------------------------------------

def test_safety_enabled_block_persists_failed_doc_and_safety_review(monkeypatch):
    """Ingestion-scope block: HTTP 422, failed document, succeeded safety
    review with a linked finding, safety_blocked skipped identity."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.config import get_settings
    from app.core.db import Base, get_db
    from app.main import app
    from app.services.graph_extraction import (
        DisabledGraphExtractor,
        get_graph_extractor,
    )

    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "true")
    monkeypatch.setenv("RAG_SAFETY_LLM_MODE", "disabled")
    get_settings.cache_clear()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
    try:
        client = TestClient(app)
        resp = client.post(
            "/documents",
            data={"title": "Blocked Doc", "source": "unit",
                  "text": "instructions to build a bomb"})
        assert resp.status_code == 422
        session = TestSessionLocal()
        try:
            doc = session.query(models.Document).filter_by(
                title="Blocked Doc").one()
            assert doc.ingestion_status == "failed"
            run = session.query(models.SafetyReviewRun).filter_by(
                scope="ingestion").one()
            assert run.status == "succeeded"
            assert run.final_action == "block"
            assert run.document_id == doc.id
            assert run.document_id_snapshot == doc.id
            finding = session.query(models.SafetyFinding).filter_by(
                review_run_id=run.id).one()
            assert finding.action == "block"
            assert finding.start_offset < finding.end_offset
        finally:
            session.close()
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
        get_settings.cache_clear()
        engine.dispose()


def test_safety_disabled_skips_ingestion_review():
    """RAG_CONTENT_SAFETY_ENABLED=false: no safety rows, original 10A.4 flow."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.config import get_settings
    from app.core.db import Base, get_db
    from app.main import app
    from app.services.graph_extraction import (
        DisabledGraphExtractor,
        get_graph_extractor,
    )

    get_settings.cache_clear()
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
    try:
        client = TestClient(app)
        resp = client.post(
            "/documents",
            data={"title": "No Safety Doc", "source": "unit",
                  "text": "hello world"})
        assert resp.status_code == 201
        session = TestSessionLocal()
        try:
            doc = session.query(models.Document).filter_by(
                title="No Safety Doc").one()
            assert doc.ingestion_status == "ready"
            assert session.query(models.SafetyReviewRun).count() == 0
            # original 10A.4 ordering: extraction identities exist (skipped
            # with extraction_disabled under the disabled extractor).
            assert session.query(models.GraphExtraction).count() >= 1
        finally:
            session.close()
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
        get_settings.cache_clear()
        engine.dispose()


def test_safety_blocked_extraction_skip_code_only_three_allowed(tmp_path):
    """Direct-SQL: skipped+attempt_count=0+safety_blocked is valid; any other
    skip code or a nonzero attempt count violates the d9 CHECK."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import IntegrityError

    from app.core.migrations import upgrade_database

    db_url = f"sqlite:///{tmp_path / 'skip-codes.db'}"
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO chunks (document_id, \"index\", text, "
            "start_offset, end_offset) VALUES (0, 0, 'x', 0, 1)"))

    def _insert(error_code, attempt_count, sha):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO graph_extractions (chunk_id, provider, model, "
                "prompt_version, schema_version, status, error_code, "
                "input_sha256, attempt_count, is_identity_owner, completed_at) "
                "VALUES (1, 'p', 'm', 'pv', 'sv', 'skipped', :code, :sha, "
                ":attempts, 1, '2026-08-01T00:00:00Z')"
            ), {"code": error_code, "sha": sha, "attempts": attempt_count})

    _insert("safety_blocked", 0, "a" * 64)  # valid
    with pytest.raises(IntegrityError):
        _insert("safety_blocked", 1, "b" * 64)
    with pytest.raises(IntegrityError):
        _insert("not_a_skip_code", 0, "c" * 64)
    engine.dispose()
