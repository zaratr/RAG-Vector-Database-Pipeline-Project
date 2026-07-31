import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base, get_db
from app.main import app

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
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


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

    detail = client.get(f"/documents/{doc_id}")
    assert detail.status_code == 200


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
