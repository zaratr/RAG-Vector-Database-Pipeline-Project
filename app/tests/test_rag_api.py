from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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
