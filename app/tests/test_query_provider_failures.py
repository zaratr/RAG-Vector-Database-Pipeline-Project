"""LLM provider failure contract for POST /query.

The answer LLM is an external provider invoked during /query. Transport
failures — connection refused, timeout, and 5xx responses — must surface as a
stable, typed HTTP 503 with detail ``LLM provider unavailable`` (mirroring the
graph-extractor and vector-store provider-failure conventions), and a provider
that answers unusable output (a non-JSON envelope) must surface as a typed 502
with detail ``LLM provider failed`` — never an unhandled 500 with a traceback.
All failure lanes are hermetic: the transport is either a raising stub or
httpx.MockTransport, so no Ollama is needed.
"""
from __future__ import annotations

import uuid

import chromadb
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base, get_db
from app.main import app
from app.services.graph_extraction import DisabledGraphExtractor, get_graph_extractor
from app.services.llm import DummyLLMClient, OllamaLLMClient
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


app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
client = TestClient(app)

_COLLECTION = "test-query-llm-failures-" + uuid.uuid4().hex[:8]


class ConnectionRefusedLLM:
    async def generate_answer(self, query, context):
        raise httpx.ConnectError("connection refused")


class TimedOutLLM:
    async def generate_answer(self, query, context):
        raise httpx.ReadTimeout("timed out")


def _llm_500_transport() -> httpx.MockTransport:
    """MockTransport whose /chat/completions answers HTTP 500."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")

    return httpx.MockTransport(handler)


def _llm_malformed_envelope_transport() -> httpx.MockTransport:
    """MockTransport whose /chat/completions answers 200 with a non-JSON
    body — a provider that answers garbage instead of failing to answer."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>proxy error page</body></html>")

    return httpx.MockTransport(handler)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Bind the app to this module's engine/store/dummy-LLM for the module's
    duration, restoring whatever the previously-run module left in place
    (same order-safe pattern as test_graph_api.graph_api_environment)."""
    ephemeral = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(collection_name=_COLLECTION, client=ephemeral)
    original_documents_store = routes_documents.get_vector_store
    original_query_store = routes_query.get_vector_store
    original_query_llm = routes_query.get_llm_client
    prior_db_override = app.dependency_overrides.get(get_db)
    routes_documents.get_vector_store = lambda: vector_store
    routes_query.get_vector_store = lambda: vector_store
    routes_query.get_llm_client = lambda *args, **kwargs: DummyLLMClient()
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    routes_documents.get_vector_store = original_documents_store
    routes_query.get_vector_store = original_query_store
    routes_query.get_llm_client = original_query_llm
    ephemeral.delete_collection(_COLLECTION)
    if prior_db_override is not None:
        app.dependency_overrides[get_db] = prior_db_override
    else:
        app.dependency_overrides.pop(get_db, None)
    test_engine.dispose()


def _ingest_one_document():
    response = client.post(
        "/documents",
        data={"title": "LLM Failure Doc", "text": "LLM failure lane evidence text", "source": "unit"},
    )
    assert response.status_code == 201
    return response.json()["document_id"]


def _use_llm_client(monkeypatch, llm):
    monkeypatch.setattr(routes_query, "get_llm_client", lambda *args, **kwargs: llm)


def test_query_connection_refused_returns_stable_503(monkeypatch):
    """Connection failure to the LLM provider -> 503 with the pinned detail,
    no traceback leakage. Retrieval itself succeeded (document ingested), so
    this pins exactly the provider-failure mapping."""
    _ingest_one_document()
    _use_llm_client(monkeypatch, ConnectionRefusedLLM())

    response = client.post("/query", json={"query": "evidence", "top_k": 3})

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider unavailable"
    assert "Traceback" not in response.text


def test_query_llm_timeout_returns_stable_503(monkeypatch):
    _ingest_one_document()
    _use_llm_client(monkeypatch, TimedOutLLM())

    response = client.post("/query", json={"query": "evidence", "top_k": 3})

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider unavailable"
    assert "Traceback" not in response.text


def test_query_llm_http_500_returns_stable_503(monkeypatch):
    """A production OllamaLLMClient whose endpoint answers 500 raises
    HTTPStatusError inside generate_answer; the route must map it to the
    same stable 503 contract."""
    _ingest_one_document()
    _use_llm_client(
        monkeypatch,
        OllamaLLMClient(
            base_url="http://llm-invalid.test/v1",
            model="gemma4:latest",
            transport=_llm_500_transport(),
        ),
    )

    response = client.post("/query", json={"query": "evidence", "top_k": 3})

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider unavailable"
    assert "Traceback" not in response.text


def test_query_llm_malformed_envelope_returns_stable_502(monkeypatch):
    """A production OllamaLLMClient whose endpoint answers 200 with a
    non-JSON body raises LLMProviderOutputError; the route maps it to a
    typed 502 (provider answered unusable output — distinct from the 503
    transport-unavailable lane), never an unhandled 500 with a traceback."""
    _ingest_one_document()
    _use_llm_client(
        monkeypatch,
        OllamaLLMClient(
            base_url="http://llm-invalid.test/v1",
            model="gemma4:latest",
            transport=_llm_malformed_envelope_transport(),
        ),
    )

    response = client.post("/query", json={"query": "evidence", "top_k": 3})

    assert response.status_code == 502
    assert response.json()["detail"] == "LLM provider failed"
    assert "Traceback" not in response.text


def test_query_happy_path_returns_grounded_answer():
    """Sanity lane for the failure tests above: with a working LLM the same
    fixture yields 200 with an answer grounded in the retrieved context, so
    the 503s above can only come from the provider-failure mapping."""
    _ingest_one_document()

    response = client.post("/query", json={"query": "evidence", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["context"], "ingested document must be retrievable"
    assert payload["context"][0]["text"] in payload["answer"]
