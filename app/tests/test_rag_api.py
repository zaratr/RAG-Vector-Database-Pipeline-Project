import chromadb
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base, get_db
from app.main import app
from app.persistence import models
from app.services.graph_extraction import (
    DisabledGraphExtractor,
    GraphExtractionError,
    GraphProviderUnavailable,
    get_graph_extractor,
)
from app.services.llm import DummyLLMClient
from app.services.vector_store import ChromaVectorStore
from app.api import routes_documents, routes_query

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


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


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
client = TestClient(app)

# Captured at import time, before the module fixture swaps the factory, so
# the opt-in live-LLM lane can exercise the settings-configured client.
_REAL_GET_LLM_CLIENT = routes_query.get_llm_client


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(
        collection_name="test-rag-api", client=client
    )
    original_documents_store = routes_documents.get_vector_store
    original_query_store = routes_query.get_vector_store
    original_query_llm = routes_query.get_llm_client
    routes_documents.get_vector_store = lambda: vector_store
    routes_query.get_vector_store = lambda: vector_store
    # Hermetic answer lane: /query builds its LLM client from settings
    # (default provider "ollama"), which would make a live HTTP call for
    # every query test. Swap the factory for the echo dummy — the same
    # pattern as test_graph_api — so query tests prove the retrieval ->
    # generation wiring without a running Ollama.
    routes_query.get_llm_client = lambda *args, **kwargs: DummyLLMClient()
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    routes_documents.get_vector_store = original_documents_store
    routes_query.get_vector_store = original_query_store
    routes_query.get_llm_client = original_query_llm
    client.delete_collection("test-rag-api")


def test_ingest_and_query():
    ingest_response = client.post(
        "/documents",
        data={"title": "API Doc", "text": "FastAPI enables quick APIs", "source": "unit"},
    )
    assert ingest_response.status_code == 201
    doc_id = ingest_response.json()["document_id"]

    query_response = client.post("/query", json={"query": "FastAPI", "top_k": 1})
    assert query_response.status_code == 200
    payload = query_response.json()
    assert "answer" in payload
    assert len(payload["context"]) >= 1
    assert "vector" in payload["context"][0]["metadata"]["retrieval_sources"]
    # Grounding: the answer must be built from the retrieved context. The
    # dummy LLM echoes the exact context strings it receives, so a regression
    # that breaks retrieval-into-generation wiring (wrong/empty context list
    # reaching the LLM) fails this assertion instead of passing on HTTP 200.
    assert payload["context"][0]["text"] in payload["answer"]

    detail = client.get(f"/documents/{doc_id}")
    assert detail.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "test", "retrieval_mode": "unknown"},
        {"query": "test", "graph_max_hops": 0},
        {"query": "test", "graph_max_hops": 4},
        {"query": "test", "top_k": 0},
        {"query": ""},
    ],
)
def test_query_rejects_invalid_retrieval_controls(payload):
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_graph_extraction_provider_failure_returns_502_and_persists_failed_state():
    """10A.4: invalid provider output → HTTP 502; failed state persists (operator-visible)."""
    class FailingExtractor:
        async def extract(self, text):
            raise GraphExtractionError("unavailable")

    app.dependency_overrides[get_graph_extractor] = lambda: FailingExtractor()
    try:
        response = client.post(
            "/documents",
            data={"title": "Must Persist Failed", "text": "Alice works at Acme."},
        )
    finally:
        app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()

    assert response.status_code == 502
    session = TestSessionLocal()
    try:
        doc = session.query(models.Document).filter_by(title="Must Persist Failed").one()
        # Failed document remains operator-visible but is not query-visible.
        assert doc.ingestion_status == "failed"
    finally:
        session.close()


def test_graph_provider_unavailable_returns_503_and_persists_failed_state():
    """10A.4: provider unavailable → HTTP 503; failed doc operator-visible, not query-visible."""
    class UnavailableExtractor:
        async def extract(self, text):
            raise GraphProviderUnavailable("offline")

    app.dependency_overrides[get_graph_extractor] = lambda: UnavailableExtractor()
    try:
        response = client.post(
            "/documents",
            data={"title": "Unavailable Graph", "text": "Alice works at Acme."},
        )
    finally:
        app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()

    assert response.status_code == 503
    session = TestSessionLocal()
    try:
        doc = session.query(models.Document).filter_by(title="Unavailable Graph").one()
        assert doc.ingestion_status == "failed"
    finally:
        session.close()


# ── Markdown ingestion tests (F3) ───────────────────────────────────

MARKDOWN_CONTENT = "# RAG Overview\n\nRetrieval-augmented generation combines search with LLMs.\n"


def test_ingest_markdown_content_type():
    """Upload a .md file with Content-Type: text/markdown → 201 + chunks created."""
    response = client.post(
        "/documents",
        data={"title": "MD Content Test", "source": "unit"},
        files={"file": ("test.md", MARKDOWN_CONTENT, "text/markdown")},
    )
    assert response.status_code == 201
    assert response.json()["chunks"] > 0


def test_ingest_markdown_filename_fallback():
    """Upload a .md file with generic Content-Type → route falls back to filename .md → 201."""
    response = client.post(
        "/documents",
        data={"title": "MD Fallback Test", "source": "unit"},
        files={"file": ("readme.md", MARKDOWN_CONTENT, "application/octet-stream")},
    )
    assert response.status_code == 201
    assert response.json()["chunks"] > 0


def test_ingest_markdown_extension_variant():
    """Upload a .markdown file with generic content-type → filename fallback handles it → 201."""
    response = client.post(
        "/documents",
        data={"title": "MD Variant Test", "source": "unit"},
        files={"file": ("notes.markdown", MARKDOWN_CONTENT, "application/octet-stream")},
    )
    assert response.status_code == 201
    assert response.json()["chunks"] > 0


def test_ingest_txt_via_filename_fallback():
    """Upload a .txt file with generic content-type → filename fallback handles it → 201."""
    response = client.post(
        "/documents",
        data={"title": "TXT Fallback Test", "source": "unit"},
        files={"file": ("notes.txt", "Some text content", "application/octet-stream")},
    )
    assert response.status_code == 201
    assert response.json()["chunks"] > 0


def test_ingest_empty_file_returns_400():
    """Upload an empty file → 400 with detail message, not 500."""
    response = client.post(
        "/documents",
        data={"title": "Empty Test", "source": "unit"},
        files={"file": ("empty.md", "", "text/markdown")},
    )
    assert response.status_code == 400
    assert "No text content" in response.json()["detail"]


def test_ingest_invalid_utf8_returns_400():
    """Upload a file with invalid UTF-8 bytes → 400 with detail message, not 500."""
    response = client.post(
        "/documents",
        data={"title": "Bad UTF8 Test", "source": "unit"},
        files={"file": ("bad.md", b"\xff\xfe\x00\x01", "text/markdown")},
    )
    assert response.status_code == 400
    assert "not valid UTF-8" in response.json()["detail"]


def test_ingest_corrupt_pdf_returns_400():
    """Upload a corrupt/truncated PDF → 400 with detail message, not 500."""
    response = client.post(
        "/documents",
        data={"title": "Corrupt PDF Test", "source": "unit"},
        files={"file": ("corrupt.pdf", b"%PDF-1.4\nbroken content\n%%EOF", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Could not parse PDF" in response.json()["detail"]


def test_ingest_unsupported_file_type_returns_400():
    """Upload an unsupported file type (.dat) → 400."""
    response = client.post(
        "/documents",
        data={"title": "Unsupported Test", "source": "unit"},
        files={"file": ("data.dat", b"binary data", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_ingest_whitespace_only_returns_400():
    """Upload a file with only whitespace → 400 via the .strip() guard, not 500."""
    response = client.post(
        "/documents",
        data={"title": "Whitespace Test", "source": "unit"},
        files={"file": ("blank.md", "   \n\t  \n  ", "text/markdown")},
    )
    assert response.status_code == 400
    assert "No text content" in response.json()["detail"]


def test_ingest_valid_pdf_no_extractable_text_returns_400():
    """Upload a structurally-valid PDF that yields zero text → 400, not 500."""
    # Minimal valid PDF structure with no text streams
    minimal_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n206\n%%EOF"
    response = client.post(
        "/documents",
        data={"title": "Empty PDF Test", "source": "unit"},
        files={"file": ("empty_text.pdf", minimal_pdf, "application/pdf")},
    )
    assert response.status_code == 400
    assert "No text content" in response.json()["detail"]


# ── 10A.4 HTTP behavior tests (phase10-test-specifications appendix) ──
# Adapted to the implemented API surface: routes are mounted without an
# /api/v1 prefix, /documents takes Form fields, and the graph extractor is
# injected via app.dependency_overrides (the established pattern above).


def test_post_documents_provider_unavailable_returns_503():
    """Provider unavailable → HTTP 503; failed doc not query-visible."""
    from app.services.graph_extraction import GraphProviderUnavailable

    class UnavailableExtractor:
        async def extract(self, text):
            raise GraphProviderUnavailable("unavailable")

    app.dependency_overrides[get_graph_extractor] = lambda: UnavailableExtractor()
    try:
        response = client.post(
            "/documents",
            data={"title": "503 Test", "source": "test", "text": "Alice works at Acme."},
        )
    finally:
        app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()

    assert response.status_code == 503


def test_post_documents_invalid_provider_output_returns_502():
    """Invalid provider output → HTTP 502; failed state persists."""
    from app.services.graph_extraction import GraphProviderOutputError

    class InvalidOutputExtractor:
        async def extract(self, text):
            raise GraphProviderOutputError("bad output")

    app.dependency_overrides[get_graph_extractor] = lambda: InvalidOutputExtractor()
    try:
        response = client.post(
            "/documents",
            data={"title": "502 Test", "source": "test", "text": "Alice works at Acme."},
        )
    finally:
        app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()

    assert response.status_code == 502


def test_post_documents_vector_failure_returns_503_with_stable_detail(monkeypatch):
    """Vector failure → HTTP 503 with detail 'Vector index unavailable'."""
    class FailingStore:
        async def upsert_embeddings(self, *args, **kwargs):
            raise RuntimeError("chroma connection refused")

        async def list_ids(self):
            return []

        async def delete(self, ids):
            return None

    monkeypatch.setattr(
        routes_documents, "get_vector_store", lambda: FailingStore()
    )

    response = client.post(
        "/documents",
        data={"title": "Vector 503", "source": "test", "text": "Alice works at Acme."},
    )

    assert response.status_code == 503
    assert "Vector index unavailable" in response.json().get("detail", "")


def test_post_documents_success_returns_201_with_document_id_chunks_relations():
    """Success → HTTP 201 with document_id, chunks, relations; document ready."""
    response = client.post(
        "/documents",
        data={"title": "Success API", "source": "test", "text": "Alice works at Acme."},
    )

    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert "chunks" in data
    assert "relations" in data


def test_post_documents_disabled_extraction_returns_201_relations_zero_skipped():
    """Disabled extraction → 201, relations 0, extraction status skipped (not empty)."""
    # The module-level dependency override already installs DisabledGraphExtractor.
    response = client.post(
        "/documents",
        data={"title": "Disabled API", "source": "test", "text": "Alice works at Acme."},
    )

    assert response.status_code == 201
    assert response.json()["relations"] == 0

    session = TestSessionLocal()
    try:
        doc = session.query(models.Document).filter_by(title="Disabled API").one()
        assert doc.ingestion_status == "ready"
        extraction = (
            session.query(models.GraphExtraction)
            .filter_by(chunk_id=doc.chunks[0].id)
            .one()
        )
        assert extraction.status == "skipped"
        assert extraction.error_code == "extraction_disabled"
        session.delete(doc)
        session.commit()
    finally:
        session.close()


def test_query_excludes_non_ready_documents():
    """Non-ready documents are excluded from query results."""
    # Ingest a document, mark it failed, query should not return it
    response = client.post(
        "/documents",
        data={
            "title": "Will Fail",
            "source": "test",
            "text": "Unique search text xyz123.",
        },
    )
    assert response.status_code == 201
    doc_id = response.json()["document_id"]

    # Manually mark as failed via direct DB access (simulating crash)
    session = TestSessionLocal()
    try:
        doc = session.get(models.Document, doc_id)
        doc.ingestion_status = "failed"
        session.commit()
    finally:
        session.close()

    query_response = client.post(
        "/query", json={"query": "Unique search text xyz123.", "retrieval_mode": "vector"}
    )
    assert query_response.status_code == 200

    # The failed document's chunks must not appear in results
    results = query_response.json()
    for item in results.get("context", []):
        assert item.get("metadata", {}).get("document_id") != doc_id


# ── 10A.6 /query retrieval-mode tests (phase10-test-specifications appendix) ──


def test_query_hybrid_mode_returns_200_with_context():
    response = client.post("/query", json={
        "query": "FastAPI", "top_k": 3, "retrieval_mode": "hybrid",
        "graph_max_hops": 2,
    })
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert isinstance(payload["context"], list)
    if payload["context"]:
        assert payload["context"][0]["text"] in payload["answer"]


def test_query_graph_mode_returns_200_without_embedding_dependency():
    response = client.post("/query", json={
        "query": "FastAPI", "top_k": 3, "retrieval_mode": "graph",
        "graph_max_hops": 1,
    })
    # graph mode may return empty context if no graph data, but must be 200
    assert response.status_code == 200


def test_query_hybrid_with_unsupported_filter_returns_422():
    response = client.post("/query", json={
        "query": "FastAPI", "retrieval_mode": "hybrid",
        "filters": {"unknown_key": 1},
    })
    assert response.status_code == 422


def test_query_graph_with_unsupported_filter_returns_422():
    response = client.post("/query", json={
        "query": "FastAPI", "retrieval_mode": "graph",
        "filters": {"document_id": "not-an-int"},
    })
    assert response.status_code == 422


def test_query_vector_mode_accepts_filters_and_remains_default():
    response = client.post("/query", json={
        "query": "FastAPI", "filters": {"document_id": 1},
    })
    # default mode is vector; 200 even if no hits match the filter
    assert response.status_code == 200


def test_query_with_no_matching_context_returns_200_and_empty_context():
    """Grounded no-evidence contract over HTTP: a query whose filter matches
    nothing returns 200 with context == [] and still produces an answer."""
    response = client.post("/query", json={
        "query": "anything", "filters": {"document_id": 999999},
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["context"] == []
    assert payload["answer"].strip()


def test_query_hybrid_traversal_limit_returns_503(monkeypatch):
    """If retrieve_graph_paths raises GraphTraversalLimitError, /query
    must return 503 (mapped in routes_query.py)."""
    from app.services import rag
    from app.services.graph_retrieval import GraphTraversalLimitError

    # retrieve_graph_paths is synchronous in production; the appendix's
    # ``async def _boom(**kwargs)`` shape is adapted to the sync signature
    # (and ``session`` is passed positionally), preserving the patch point,
    # the raised exception, and the asserted 503.
    def _boom(*args, **kwargs):
        raise GraphTraversalLimitError("cap exceeded")
    # retrieval.py wires retrieve_graph_paths (not retrieve_graph_contexts)
    # into hybrid/graph candidate building, so that is the patch point.
    monkeypatch.setattr(rag.retrieval, "retrieve_graph_paths", _boom)
    response = client.post("/query", json={
        "query": "FastAPI", "retrieval_mode": "hybrid",
    })
    assert response.status_code == 503


def test_post_image_documents_vector_failure_returns_503_with_stable_detail(monkeypatch):
    """W1: image ingestion vector failure maps to the same stable 503 detail
    as the text route (10A.4 HTTP behavior contract)."""
    class FakeImageProvider:
        async def embed_images(self, paths):
            return [[0.1] * 10]

    class FailingStore:
        async def upsert_embeddings(self, *args, **kwargs):
            raise RuntimeError("chroma connection refused")

        async def list_ids(self):
            return []

        async def delete(self, ids):
            return None

    monkeypatch.setattr(
        routes_documents, "get_image_embedding_provider", lambda: FakeImageProvider()
    )
    monkeypatch.setattr(
        routes_documents, "get_vector_store", lambda: FailingStore()
    )

    response = client.post(
        "/documents",
        data={"title": "Image Vector 503", "source": "test"},
        files={"file": ("pic.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "image/png")},
    )

    assert response.status_code == 503
    assert "Vector index unavailable" in response.json().get("detail", "")


@pytest.mark.skipif(
    not os.environ.get("RAG_LIVE_LLM"),
    reason="opt-in live lane: set RAG_LIVE_LLM=1 with a reachable LLM at RAG_LLM_BASE_URL",
)
def test_query_answer_lane_with_live_llm_optin():
    """Opt-in live-LLM lane: exercises the settings-configured LLM client
    (real Ollama/Gemma) through POST /query. Skipped in the hermetic suite;
    enabled by setting RAG_LIVE_LLM=1 with a reachable RAG_LLM_BASE_URL."""
    ingest_response = client.post(
        "/documents",
        data={"title": "Live LLM Doc", "text": "FastAPI enables quick APIs", "source": "unit"},
    )
    assert ingest_response.status_code == 201

    module_llm_factory = routes_query.get_llm_client
    routes_query.get_llm_client = _REAL_GET_LLM_CLIENT
    try:
        response = client.post("/query", json={"query": "What does FastAPI do?", "top_k": 1})
    finally:
        routes_query.get_llm_client = module_llm_factory

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].strip()
    assert len(payload["context"]) >= 1
