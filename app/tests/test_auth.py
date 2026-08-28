"""Phase 10B.2 — operator authentication tests."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.auth import require_operator
from app.config import Settings


def test_operator_api_disabled_returns_404():
    settings = Settings(operator_api_enabled=False)
    with pytest.raises(HTTPException) as exc:
        require_operator(credentials=None, settings=settings)
    assert exc.value.status_code == 404


def test_operator_api_enabled_missing_bearer_returns_401():
    settings = Settings(operator_api_enabled=True, operator_token="x" * 32)
    with pytest.raises(HTTPException) as exc:
        require_operator(credentials=None, settings=settings)
    assert exc.value.status_code == 401


def test_operator_api_enabled_invalid_bearer_returns_401():
    settings = Settings(operator_api_enabled=True, operator_token="x" * 32)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-token")
    with pytest.raises(HTTPException) as exc:
        require_operator(credentials=creds, settings=settings)
    assert exc.value.status_code == 401


def test_operator_api_enabled_valid_bearer_returns_true():
    token = "x" * 32
    settings = Settings(operator_api_enabled=True, operator_token=token)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    result = require_operator(credentials=creds, settings=settings)
    assert result is True


# ---------------------------------------------------------------------------
# Startup validation (appendix 10B.2): enabling the operator API with a short
# credential must fail Settings construction.

def test_operator_token_must_be_32_chars_when_enabled_or_startup_fails(monkeypatch):
    monkeypatch.setenv("RAG_OPERATOR_API_ENABLED", "true")
    monkeypatch.setenv("RAG_OPERATOR_TOKEN", "short")
    from app.config import Settings
    with pytest.raises(Exception):
        Settings()


def test_operator_token_of_32_chars_is_accepted_when_enabled(monkeypatch):
    monkeypatch.setenv("RAG_OPERATOR_API_ENABLED", "true")
    monkeypatch.setenv("RAG_OPERATOR_TOKEN", "x" * 32)
    from app.config import Settings
    settings = Settings()
    assert settings.operator_api_enabled is True


# ---------------------------------------------------------------------------
# HTTP lane (appendix 10B.2): valid operator bearer on a protected trusted
# source ingests with the policy-assigned trusted state. This is the one
# public-ingestion matrix row whose equivalent evidence was policy-level only.

def test_public_ingestion_valid_bearer_protected_source_continues(tmp_path):
    import chromadb
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import app.api.routes_documents as routes_documents
    from app.config import get_settings
    from app.core.db import Base, get_db
    from app.main import app
    from app.services.graph_extraction import DisabledGraphExtractor, get_graph_extractor
    from app.services.provenance import load_source_trust_policy
    from app.services.vector_store import ChromaVectorStore

    token = "b" * 40

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    class _ConstantEmbeddingProvider:
        async def embed_texts(self, texts):
            return [[1.0] * 8 for _ in texts]

    rl_engine = create_engine(f"sqlite:///{tmp_path / 'auth-rate-limit.db'}")
    Base.metadata.create_all(bind=rl_engine)
    rl_engine.dispose()

    import os

    existing_enabled = os.environ.get("RAG_OPERATOR_API_ENABLED")
    existing_token = os.environ.get("RAG_OPERATOR_TOKEN")
    existing_db = os.environ.get("RAG_DATABASE_URL")
    existing_rate = os.environ.get("RAG_INGEST_RATE_LIMIT_REQUESTS")
    os.environ["RAG_OPERATOR_API_ENABLED"] = "true"
    os.environ["RAG_OPERATOR_TOKEN"] = token
    os.environ["RAG_DATABASE_URL"] = f"sqlite:///{tmp_path / 'auth-rate-limit.db'}"
    os.environ["RAG_INGEST_RATE_LIMIT_REQUESTS"] = "1000"
    get_settings.cache_clear()

    ephemeral = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test-auth-ingestion", client=ephemeral)
    app.state.source_trust_policy = load_source_trust_policy(
        get_settings().source_trust_policy_path
    )
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
    prev_documents_embeddings = routes_documents.get_embedding_provider
    prev_documents_store = routes_documents.get_vector_store
    routes_documents.get_embedding_provider = lambda: _ConstantEmbeddingProvider()
    routes_documents.get_vector_store = lambda: store

    try:
        client = TestClient(app)
        response = client.post(
            "/documents",
            data={"title": "Trusted", "source": "operator-curated", "text": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201, response.text
        body = response.json()

        # Server-assigned trusted state came from the policy rule, not the
        # client label alone.
        session = SessionLocal()
        try:
            from sqlalchemy import select

            from app.persistence import models

            doc = session.execute(
                select(models.Document).order_by(models.Document.id.desc())
            ).scalars().first()
            assert doc is not None
            assert doc.source == "operator-curated"
            assert doc.trust_tier == "trusted"
            assert doc.trust_score == 1.0
            assert doc.trust_policy_version == "source-trust-v1"
        finally:
            session.close()
        assert body.get("document_id") is not None
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)
        routes_documents.get_embedding_provider = prev_documents_embeddings
        routes_documents.get_vector_store = prev_documents_store
        for var, value in (
            ("RAG_OPERATOR_API_ENABLED", existing_enabled),
            ("RAG_OPERATOR_TOKEN", existing_token),
            ("RAG_DATABASE_URL", existing_db),
            ("RAG_INGEST_RATE_LIMIT_REQUESTS", existing_rate),
        ):
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        get_settings.cache_clear()
        try:
            ephemeral.delete_collection("test-auth-ingestion")
        except Exception:
            pass
        engine.dispose()
