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
    # system_prompt: the merged /query lane passes 10B's immutable
    # context-security prompt through the LLMClient protocol.
    async def generate_answer(self, query, context, system_prompt=None):
        raise httpx.ConnectError("connection refused")


class TimedOutLLM:
    async def generate_answer(self, query, context, system_prompt=None):
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


def _install_permissive_retrieval_policy(monkeypatch):
    """Neutralize 10B's retrieval-security distance/cap controls in-process.

    This module pins the answer-provider failure contract (503 transport /
    502 malformed envelope) with deterministic hash embeddings whose l2
    distances exceed the production-calibrated max_distance; without the
    neutralization, retrieval would reject every candidate and the provider
    failure lanes would never be reached. Poisoning controls are pinned by
    test_poisoning.py / test_security_api.py under the real policy.
    """
    from app.services import retrieval as retrieval_module
    from app.services.retrieval_security import RetrievalSecurityPolicy

    monkeypatch.setattr(
        retrieval_module,
        "_POLICY_CACHE",
        RetrievalSecurityPolicy(
            version="retrieval-security-v1",
            metric="l2",
            max_distance=1e9,
            per_source_cap=1000,
            per_document_cap=1000,
            max_candidates=1000,
            near_duplicate_jaccard=1.0,
            calibration_fixture_sha256="deterministic-lane",
            calibration_clean_recall=1.0,
            calibration_poison_share=0.0,
            calibration_tool_version="calibrate-v1",
            calibration_embedding_model="jinaai/jina-clip-v1",
        ),
    )

@pytest.fixture(scope="module", autouse=True)
def setup_db(tmp_path_factory):
    """Bind the app to this module's engine/store/dummy-LLM for the module's
    duration, restoring whatever the previously-run module left in place
    (same order-safe pattern as test_graph_api.graph_api_environment).

    The ingestion rate limiter builds its own engine from
    settings.database_url (10B D-8/D-36), so env + settings cache must point
    at a disposable, fully-migrated database for /documents to succeed — the
    same isolation pattern as test_rag_api.setup_db. 10B's retrieval-security
    controls are neutralized for the module via the _POLICY_CACHE hook (the
    module pins the provider-failure contract, not poisoning controls)."""
    import os

    from app.config import get_settings
    from app.services import retrieval as retrieval_module
    from app.services.retrieval_security import RetrievalSecurityPolicy

    rate_db = tmp_path_factory.mktemp("qpf-rate") / "rate.db"
    original_db_url = os.environ.get("RAG_DATABASE_URL")
    os.environ["RAG_DATABASE_URL"] = f"sqlite:///{rate_db}"
    get_settings.cache_clear()
    rate_engine = create_engine(f"sqlite:///{rate_db}")
    Base.metadata.create_all(bind=rate_engine)
    rate_engine.dispose()
    original_policy_cache = retrieval_module._POLICY_CACHE
    retrieval_module._POLICY_CACHE = RetrievalSecurityPolicy(
        version="retrieval-security-v1",
        metric="l2",
        max_distance=1e9,
        per_source_cap=1000,
        per_document_cap=1000,
        max_candidates=1000,
        near_duplicate_jaccard=1.0,
        calibration_fixture_sha256="deterministic-lane",
        calibration_clean_recall=1.0,
        calibration_poison_share=0.0,
        calibration_tool_version="calibrate-v1",
        calibration_embedding_model="jinaai/jina-clip-v1",
    )
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
    retrieval_module._POLICY_CACHE = original_policy_cache
    if original_db_url is None:
        os.environ.pop("RAG_DATABASE_URL", None)
    else:
        os.environ["RAG_DATABASE_URL"] = original_db_url
    get_settings.cache_clear()


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
