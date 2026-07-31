import chromadb
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


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(
        collection_name="test-rag-api", client=client
    )
    original_documents_store = routes_documents.get_vector_store
    original_query_store = routes_query.get_vector_store
    routes_documents.get_vector_store = lambda: vector_store
    routes_query.get_vector_store = lambda: vector_store
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    routes_documents.get_vector_store = original_documents_store
    routes_query.get_vector_store = original_query_store
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


def test_graph_extraction_provider_failure_returns_502_and_rolls_back_document():
    class FailingExtractor:
        async def extract(self, text):
            raise GraphExtractionError("unavailable")

    app.dependency_overrides[get_graph_extractor] = lambda: FailingExtractor()
    try:
        response = client.post(
            "/documents",
            data={"title": "Must Roll Back", "text": "Alice works at Acme."},
        )
    finally:
        app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()

    assert response.status_code == 502
    assert response.json()["detail"] == "Graph extraction provider failed"
    session = TestSessionLocal()
    try:
        assert session.query(models.Document).filter_by(title="Must Roll Back").count() == 0
    finally:
        session.close()


def test_graph_provider_unavailable_returns_503_and_writes_nothing():
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
        assert session.query(models.Document).filter_by(title="Unavailable Graph").count() == 0
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
