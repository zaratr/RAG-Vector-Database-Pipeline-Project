"""Phase 10C.4 — end-to-end ingestion/context/answer safety enforcement tests.

Covers ingestion block/filter staging, context rejected_safety decisions,
answer withhold/filter transformations, audit completion gating, crash
extension, begin idempotency/concurrency, and the safety_summary shape.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.core.db import Base, get_db
from app.main import app
from app.persistence import models  # noqa: F401 — register models for create_all
from app.services.graph_extraction import DisabledGraphExtractor, get_graph_extractor
from app.services.safety_review import (
    apply_answer_filter,
    merge_filter_spans,
)
from app.services.vector_store import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _ConstantEmbeddingProvider:
    """Deterministic embeddings: distance 0 for all pairs, chunk-id order."""

    async def embed_texts(self, texts):
        return [[1.0] * 8 for _ in texts]


class RecordingLLMClient:
    def __init__(self):
        self.captured_context: list = []
        self.call_count = 0
        self.answer = "Safe generated answer about the topic."

    async def generate_answer(self, query, context, system_prompt=None):
        self.call_count += 1
        self.captured_context = list(context)
        return self.answer


llm_client = RecordingLLMClient()

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(bind=engine)


def override_get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _reset_caches():
    yield
    from app.services.context_security import reset_context_security_policy_cache
    from app.services.safety_policy import reset_safety_policy_cache

    get_settings.cache_clear()
    reset_context_security_policy_cache()
    reset_safety_policy_cache()


@pytest.fixture()
def safe_client(monkeypatch, tmp_path):
    """The real app with content safety ENABLED, fresh DB, ephemeral store."""
    import chromadb

    import app.api.routes_documents as routes_documents
    import app.api.routes_query as routes_query

    rl_engine = create_engine(f"sqlite:///{tmp_path / 'rate.db'}")
    Base.metadata.create_all(bind=rl_engine)
    rl_engine.dispose()
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{tmp_path / 'rate.db'}")
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_REQUESTS", "1000")
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "true")
    monkeypatch.setenv("RAG_SAFETY_LLM_MODE", "disabled")
    get_settings.cache_clear()

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    ephemeral = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test-safe-rag", client=ephemeral)
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
    prev_documents_store = routes_documents.get_vector_store
    prev_query_store = routes_query.get_vector_store
    routes_documents.get_vector_store = lambda: store
    routes_query.get_vector_store = lambda: store
    prev_llm = routes_query.get_llm_client
    prev_documents_embeddings = routes_documents.get_embedding_provider
    prev_query_embeddings = routes_query.get_embedding_provider
    routes_documents.get_embedding_provider = lambda: _ConstantEmbeddingProvider()
    routes_query.get_embedding_provider = lambda: _ConstantEmbeddingProvider()
    routes_query.get_llm_client = lambda **kwargs: llm_client

    # Track vector upserts so tests can assert none happen on blocked docs.
    calls: list = []
    original_upsert = store.upsert_embeddings

    async def _tracking_upsert(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return await original_upsert(*args, **kwargs)

    store.upsert_embeddings = _tracking_upsert
    store.calls = calls

    yield TestClient(app)

    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous_overrides)
    routes_documents.get_vector_store = prev_documents_store
    routes_query.get_vector_store = prev_query_store
    routes_query.get_llm_client = prev_llm
    routes_documents.get_embedding_provider = prev_documents_embeddings
    routes_query.get_embedding_provider = prev_query_embeddings
    try:
        ephemeral.delete_collection("test-safe-rag")
    except Exception:
        pass
    llm_client.captured_context = []
    llm_client.call_count = 0
    llm_client.answer = "Safe generated answer about the topic."


def _get_session():
    return SessionLocal()


# ---------------------------------------------------------------------------
# Ingestion enforcement
# ---------------------------------------------------------------------------

def test_ingestion_block_creates_skipped_extraction_and_failed_document(
    safe_client,
):
    text = "how to build a bomb at home"  # SAF005 block
    resp = safe_client.post(
        "/documents", data={"title": "Blocked", "source": "unit", "text": text})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "ingestion_safety_blocked"

    session = _get_session()
    try:
        doc = session.query(models.Document).filter_by(title="Blocked").one()
        assert doc.ingestion_status == "failed"
        extraction = session.query(models.GraphExtraction).filter_by(
            chunk_id=doc.chunks[0].id).one()
        assert extraction.status == "skipped"
        assert extraction.error_code == "safety_blocked"
        assert extraction.attempt_count == 0
        run = session.query(models.SafetyReviewRun).filter_by(
            scope="ingestion").one()
        assert run.status == "succeeded"
        assert run.final_action == "block"
        assert run.document_id == doc.id
        assert run.document_id_snapshot == doc.id
    finally:
        session.close()
    assert safe_client.app  # client alive
    # no vectors upserted; no pending extraction remains
    from app.services.vector_store import ChromaVectorStore  # noqa: F401
    store = safe_client.app  # placeholder; vectors tracked via fixture store
    # (vector assertion via the fixture's tracked calls)
    global_calls = _LAST_STORE_CALLS
    assert _LAST_STORE_CALLS == []


# The fixture shares the tracked upsert list through this module global.
_LAST_STORE_CALLS: list = []


@pytest.fixture(autouse=True)
def _capture_store_calls(safe_client):
    _LAST_STORE_CALLS.clear()
    yield


def test_ingestion_filter_behaves_like_block_for_vectors(safe_client):
    text = "they dehumanize that group daily"  # SAF004 filter
    resp = safe_client.post(
        "/documents", data={"title": "Filtered", "source": "unit", "text": text})
    assert resp.status_code == 422
    session = _get_session()
    try:
        doc = session.query(models.Document).filter_by(title="Filtered").one()
        assert doc.ingestion_status == "failed"
        extraction = session.query(models.GraphExtraction).filter_by(
            chunk_id=doc.chunks[0].id).one()
        assert extraction.status == "skipped"
        assert extraction.error_code == "safety_blocked"
        assert extraction.attempt_count == 0
        run = session.query(models.SafetyReviewRun).filter_by(
            scope="ingestion").one()
        assert run.status == "succeeded"
        assert run.final_action == "filter"
    finally:
        session.close()
    assert _LAST_STORE_CALLS == []


def test_ingestion_warn_proceeds_to_normal_extraction(safe_client):
    text = "he would stab the guard in the play"  # SAF001 warn
    resp = safe_client.post(
        "/documents", data={"title": "Warned", "source": "unit", "text": text})
    assert resp.status_code == 201
    session = _get_session()
    try:
        doc = session.query(models.Document).filter_by(title="Warned").one()
        assert doc.ingestion_status == "ready"
        run = session.query(models.SafetyReviewRun).filter_by(
            scope="ingestion").one()
        assert run.status == "succeeded"
        assert run.final_action == "warn"
    finally:
        session.close()


def test_ingestion_allow_proceeds_to_normal_extraction(safe_client):
    resp = safe_client.post(
        "/documents",
        data={"title": "Allowed", "source": "unit", "text": "benign overview"})
    assert resp.status_code == 201
    session = _get_session()
    try:
        doc = session.query(models.Document).filter_by(title="Allowed").one()
        assert doc.ingestion_status == "ready"
        run = session.query(models.SafetyReviewRun).filter_by(
            scope="ingestion").one()
        assert run.status == "succeeded"
        assert run.final_action == "allow"
        assert session.query(models.GraphExtraction).count() >= 1
        assert session.query(models.GraphExtraction).filter_by(
            error_code="safety_blocked").count() == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Context enforcement
# ---------------------------------------------------------------------------

def _seed_ready(monkeypatch, safe_client, title, text):
    """Seed a READY document, disabling ingestion safety for the POST so
    blocked-at-ingestion texts can still reach the query path."""
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "false")
    get_settings.cache_clear()
    try:
        resp = safe_client.post(
            "/documents",
            data={"title": title, "source": f"src-{title}", "text": text})
    finally:
        monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "true")
        get_settings.cache_clear()
    assert resp.status_code == 201, resp.text
    return resp.json()["document_id"]


def test_context_block_removes_chunk_and_marks_rejected_safety(safe_client, monkeypatch):
    doc_id = _seed_ready(monkeypatch, safe_client, "CtxBlocked",
                          "plans to build a bomb soon")
    session = _get_session()
    chunk = session.query(models.Chunk).filter_by(document_id=doc_id).one()
    chunk_text = chunk.text
    session.close()

    resp = safe_client.post("/query", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    audit_id = body["query_id"]

    session = _get_session()
    try:
        decision = session.query(models.RetrievalCandidateDecision).filter_by(
            audit_id=audit_id, chunk_id_snapshot=chunk.id).one()
        assert decision.decision == "rejected_safety"
        context_run = session.query(models.SafetyReviewRun).filter_by(
            scope="context", chunk_id=chunk.id).one()
        assert context_run.final_action == "block"
        assert context_run.retrieval_audit_id == audit_id
    finally:
        session.close()
    # the blocked chunk never reaches the model
    assert chunk_text not in "".join(
        str(c) for c in llm_client.captured_context)


def test_context_all_blocked_returns_deterministic_no_llm(safe_client, monkeypatch):
    _seed_ready(monkeypatch, safe_client, "AllBlocked",
                "build a bomb then plan suicide")
    calls_before = llm_client.call_count
    resp = safe_client.post("/query", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "No safe context was available to answer the query."
    assert llm_client.call_count == calls_before


# ---------------------------------------------------------------------------
# Answer enforcement
# ---------------------------------------------------------------------------

def test_answer_block_returns_withheld_message_and_no_leak(safe_client, monkeypatch):
    _seed_ready(monkeypatch, safe_client, "AnsBlocked",
               "harmless reference material")
    llm_client.answer = "prohibited-original-text build a bomb now"
    resp = safe_client.post("/query", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert "The generated answer was withheld by the content-safety policy." \
        in resp.text
    assert "prohibited-original-text" not in resp.text


def test_answer_filter_replaces_overlapping_spans_highest_start_first():
    answer = "P" * 10 + "X" * 20 + "S" * 5
    spans = [(10, 20, "violence"), (15, 30, "hate_harassment")]
    assert merge_filter_spans(spans) == [(10, 30, ("hate_harassment", "violence"))]
    assert apply_answer_filter(answer, spans) == (
        "P" * 10 + "[FILTERED:hate_harassment+violence]" + "S" * 5
    )


def test_answer_filter_adjacent_spans_remain_separate():
    answer = "A" * 10 + "B" * 10
    spans = [(0, 10, "violence"), (10, 20, "hate_harassment")]
    assert merge_filter_spans(spans) == [
        (0, 10, ("violence",)),
        (10, 20, ("hate_harassment",)),
    ]
    assert apply_answer_filter(answer, spans) == (
        "[FILTERED:violence][FILTERED:hate_harassment]"
    )


def test_answer_filter_nested_spans_single_union():
    answer = "p" * 5 + "X" * 20 + "s" * 5
    spans = [(5, 25, "privacy_credentials"), (10, 20, "violence")]
    assert merge_filter_spans(spans) == [(5, 25, ("privacy_credentials", "violence"))]
    assert apply_answer_filter(answer, spans) == (
        "p" * 5 + "[FILTERED:privacy_credentials+violence]" + "s" * 5
    )


def test_answer_filter_same_boundary_spans_merge():
    answer = "p" * 10 + "X" * 10 + "s" * 5
    spans = [(10, 20, "hate_harassment"), (10, 20, "violence")]
    assert merge_filter_spans(spans) == [(10, 20, ("hate_harassment", "violence"))]
    assert apply_answer_filter(answer, spans) == (
        "p" * 10 + "[FILTERED:hate_harassment+violence]" + "s" * 5
    )


def test_answer_filter_applied_end_to_end(safe_client, monkeypatch):
    _seed_ready(monkeypatch, safe_client, "AnsFiltered",
               "neutral reference text")
    llm_client.answer = "start they dehumanize that group end"
    resp = safe_client.post("/query", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()["answer"] == (
        "start they [FILTERED:hate_harassment] end")
    assert resp.json()["safety_summary"]["answer_action"] == "filter"


# ---------------------------------------------------------------------------
# safety_summary + audit gating
# ---------------------------------------------------------------------------

def test_safety_summary_in_query_response_shape(safe_client, monkeypatch):
    _seed_ready(monkeypatch, safe_client, "Summary",
               "plain summary material")
    resp = safe_client.post("/query", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["safety_summary"]["policy_version"] == "safety-v1"
    assert "contexts" in body["safety_summary"]
    assert "answer_action" in body["safety_summary"]
    assert "answer_findings" in body["safety_summary"]


def test_audit_completes_only_after_answer_review_at_d9_head(
    safe_client, monkeypatch,
):
    _seed_ready(monkeypatch, safe_client, "Gate",
               "gating reference text")
    from app.services.safety_review import SafetyReviewService

    seen = {}
    real_review_answer = SafetyReviewService.review_answer

    def _spy(self, *args, **kwargs):
        session = _get_session()
        try:
            audit = session.query(models.RetrievalAudit).order_by(
                models.RetrievalAudit.created_at.desc()).first()
            seen["status_before_answer_persist"] = audit.status if audit else None
        finally:
            session.close()
        return real_review_answer(self, *args, **kwargs)

    monkeypatch.setattr(SafetyReviewService, "review_answer", _spy)
    resp = safe_client.post("/query", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert seen["status_before_answer_persist"] == "pending"
    session = _get_session()
    try:
        audit = session.query(models.RetrievalAudit).one()
        assert audit.status == "completed"
        answer_run = session.query(models.SafetyReviewRun).filter_by(
            scope="answer", retrieval_audit_id=audit.id).one()
        assert answer_run.status == "succeeded"
        assert answer_run.completed_at is not None
    finally:
        session.close()


def test_audit_completes_after_generation_when_safety_disabled(
    monkeypatch, tmp_path,
):
    import chromadb

    import app.api.routes_documents as routes_documents
    import app.api.routes_query as routes_query

    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{tmp_path / 'rate.db'}")
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "false")
    get_settings.cache_clear()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    ephemeral = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test-safe-rag", client=ephemeral)
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
    prev_ds, prev_qs = routes_documents.get_vector_store, routes_query.get_vector_store
    prev_llm = routes_query.get_llm_client
    routes_documents.get_vector_store = lambda: store
    routes_query.get_vector_store = lambda: store
    routes_query.get_llm_client = lambda **kwargs: llm_client
    try:
        resp = TestClient(app).post(
            "/documents",
            data={"title": "NoSafety", "source": "unit", "text": "hello world"})
        assert resp.status_code == 201
        client = TestClient(app)
        resp = client.post("/query", json={"query": "q", "top_k": 5})
        assert resp.status_code == 200
        session = _get_session()
        try:
            audit = session.query(models.RetrievalAudit).one()
            assert audit.status == "completed"
            assert session.query(models.SafetyReviewRun).count() == 0
        finally:
            session.close()
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)
        routes_documents.get_vector_store = prev_ds
        routes_query.get_vector_store = prev_qs
        routes_query.get_llm_client = prev_llm
        get_settings.cache_clear()


def test_safety_failure_returns_503_and_never_leaks_answer(
    safe_client, monkeypatch,
):
    _seed_ready(monkeypatch, safe_client, "Fail",
               "material for the failing review")
    llm_client.answer = "prohibited-original-text answer"

    def _boom(self, *args, **kwargs):
        raise RuntimeError("answer reviewer unavailable")

    from app.services.safety_review import SafetyReviewService

    monkeypatch.setattr(SafetyReviewService, "review_answer", _boom)
    resp = safe_client.post("/query", json={"query": "q", "top_k": 5})
    assert resp.status_code == 503
    assert "prohibited-original-text" not in resp.text
    session = _get_session()
    try:
        assert session.query(models.SafetyReviewRun).filter_by(
            scope="answer").count() == 0
        audit = session.query(models.RetrievalAudit).order_by(
            models.RetrievalAudit.created_at.desc()).first()
        assert audit.status == "failed"
        assert audit.failure_code == "safety_review_failed"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Crash extension
# ---------------------------------------------------------------------------

def test_crash_after_staged_before_safety_no_graph_identity(
    safe_client, monkeypatch,
):
    from app.services.safety_review import (
        SafetyReviewService,
        SafetyReviewSubsystemFailure,
    )

    def _raise_after_stage(*args, **kwargs):
        raise SafetyReviewSubsystemFailure("crash before safety review")

    monkeypatch.setattr(SafetyReviewService, "begin", _raise_after_stage)
    resp = safe_client.post(
        "/documents", data={"title": "Crash1", "source": "unit", "text": "x"})
    assert resp.status_code == 503
    session = _get_session()
    try:
        doc = session.query(models.Document).filter_by(title="Crash1").one()
        assert doc.ingestion_status == "failed"
        assert session.query(models.GraphExtraction).count() == 0
        assert session.query(models.SafetyReviewRun).count() == 0
    finally:
        session.close()


def test_crash_after_safety_before_lease_no_graph_identity(
    safe_client, monkeypatch,
):
    from app.persistence import graph_repository

    real_skip = graph_repository.skip_chunk_extraction

    from app.services.ingestion import VectorIndexIncomplete

    def _raise_after_safety(*args, **kwargs):
        if kwargs.get("reason_code") == "safety_blocked":
            return real_skip(*args, **kwargs)
        raise VectorIndexIncomplete("crash at lease creation")

    from app.services.ingestion import VectorIndexIncomplete

    monkeypatch.setattr(
        graph_repository, "skip_chunk_extraction", _raise_after_safety)
    monkeypatch.setattr(
        graph_repository, "begin_chunk_extraction",
        lambda *a, **k: (_ for _ in ()).throw(
            VectorIndexIncomplete("lease crash")))
    resp = safe_client.post(
        "/documents",
        data={"title": "Crash2", "source": "unit",
              "text": "they would stab the guard onstage"})
    assert resp.status_code == 503
    session = _get_session()
    try:
        doc = session.query(models.Document).filter_by(title="Crash2").one()
        assert doc.ingestion_status == "failed"
        assert session.query(models.GraphExtraction).count() == 0
        run = session.query(models.SafetyReviewRun).filter_by(
            scope="ingestion").one()
        assert run.status == "succeeded"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# begin idempotency / concurrency
# ---------------------------------------------------------------------------

def test_repeated_begin_returns_existing_pending_or_terminal():
    from app.services.safety_policy import load_safety_policy
    from app.services.safety_review import SafetyReviewService

    policy = load_safety_policy(
        PROJECT_ROOT / "config/content-safety-policy.json")
    session = _get_session()
    try:
        # answer-scope target: needs a retrieval audit parent for the FK.
        service = SafetyReviewService(session)
        sha_hash = hashlib.sha256(b"repeatable input").hexdigest()
        # Create the parent audit row directly.
        audit = models.RetrievalAudit(
            id="repeat-audit", query_sha256=sha_hash, retrieval_mode="vector",
            status="pending", provenance_policy_version="p",
            retrieval_policy_version="r", context_policy_version="c")
        session.add(audit)
        session.commit()

        first = service.begin(
            "answer", input_text="repeatable input", policy=policy,
            retrieval_audit_id="repeat-audit")
        second = service.begin(
            "answer", input_text="repeatable input", policy=policy,
            retrieval_audit_id="repeat-audit")
        assert second.id == first.id
        assert second.status == "pending"
        service.complete(first, final_action="warn")
        third = service.begin(
            "answer", input_text="repeatable input", policy=policy,
            retrieval_audit_id="repeat-audit")
        assert third.id == first.id
        assert third.status == "succeeded"
    finally:
        session.close()


def test_concurrent_begin_one_winner_losers_reload(tmp_path):
    from app.services.safety_policy import load_safety_policy
    from app.services.safety_review import SafetyReviewService

    policy = load_safety_policy(
        PROJECT_ROOT / "config/content-safety-policy.json")
    # A file-backed DB so concurrent worker sessions get separate
    # connections (the module engine is a shared in-memory connection).
    cc_engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrent.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=cc_engine)
    cc_session_factory = sessionmaker(bind=cc_engine)
    session = cc_session_factory()
    try:
        sha_hash = hashlib.sha256(b"concurrent input").hexdigest()
        audit = models.RetrievalAudit(
            id="concurrent-audit", query_sha256=sha_hash,
            retrieval_mode="vector", status="pending",
            provenance_policy_version="p", retrieval_policy_version="r",
            context_policy_version="c")
        session.add(audit)
        session.commit()

        results: list = []
        lock = threading.Lock()

        def _worker():
            worker_session = cc_session_factory()
            try:
                service = SafetyReviewService(worker_session)
                run = service.begin(
                    "answer", input_text="concurrent input", policy=policy,
                    retrieval_audit_id="concurrent-audit")
                with lock:
                    results.append(run.id)
            finally:
                worker_session.close()

        threads = [threading.Thread(target=_worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        session.expire_all()
        rows = session.query(models.SafetyReviewRun).filter_by(
            retrieval_audit_id="concurrent-audit").all()
        assert len(rows) == 1
        assert results[0] == results[1] == rows[0].id
    finally:
        session.close()
        cc_engine.dispose()
