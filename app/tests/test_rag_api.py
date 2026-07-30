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
