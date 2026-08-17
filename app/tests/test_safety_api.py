"""Phase 10C.5 — operator safety APIs and statistics tests.

Covers: `GET /safety/findings`, `GET /safety/findings/{id}`,
`GET /safety/stats`, auth matrix, redaction, `bounded_excerpt` NULL rules,
schema shapes.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTH = {"Authorization": "Bearer " + "s" * 32}

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
def safety_env(monkeypatch):
    monkeypatch.setenv("RAG_OPERATOR_API_ENABLED", "true")
    monkeypatch.setenv("RAG_OPERATOR_TOKEN", "s" * 32)
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "true")
    get_settings.cache_clear()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)
    get_settings.cache_clear()


def _seed_findings():
    """Seed three findings across scopes/categories/actions/sources."""
    session = SessionLocal()
    try:
        sha = "a" * 64
        # ingestion-scope run on a document (deterministic violence warn)
        doc = models.Document(title="f1", source="unit", ingestion_status="ready")
        session.add(doc)
        session.flush()
        run1 = models.SafetyReviewRun(
            scope="ingestion", status="succeeded", document_id=doc.id,
            document_id_snapshot=doc.id, input_sha256=sha,
            policy_version="safety-v1", detector_version="rules-v1",
            llm_status="skipped", final_action="warn",
            completed_at=datetime(2026, 8, 2, 10, 0, 0, tzinfo=timezone.utc))
        session.add(run1)
        session.flush()
        session.add(models.SafetyFinding(
            review_run_id=run1.id, category="violence", severity=3,
            action="warn", start_offset=0, end_offset=4,
            source_rule_ids='["SAF001_violence"]', excerpt_sha256=sha))

        # context-scope run (merged privacy filter)
        audit = models.RetrievalAudit(
            id="audit-1", query_sha256=sha, retrieval_mode="vector",
            status="completed", provenance_policy_version="p",
            retrieval_policy_version="r", context_policy_version="c",
            completed_at=datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc))
        session.add(audit)
        chunk = models.Chunk(
            document_id=doc.id, index=0, text="api key is here",
            start_offset=0, end_offset=15, vector_id="v1")
        session.add(chunk)
        session.flush()
        run2 = models.SafetyReviewRun(
            scope="context", status="succeeded", document_id=doc.id,
            chunk_id=chunk.id, document_id_snapshot=doc.id,
            chunk_id_snapshot=chunk.id, retrieval_audit_id="audit-1",
            input_sha256=sha, policy_version="safety-v1",
            detector_version="rules-v1", llm_status="skipped",
            final_action="filter", completed_at=datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc))
        session.add(run2)
        session.flush()
        session.add(models.SafetyFinding(
            review_run_id=run2.id, category="privacy_credentials",
            severity=4, action="filter", start_offset=0, end_offset=10,
            source_rule_ids='["LLM_privacy_credentials","SAF006_privacy_credentials"]',
            excerpt_sha256=sha))

        # answer-scope run (llm-only violence block)
        run3 = models.SafetyReviewRun(
            scope="answer", status="succeeded", retrieval_audit_id="audit-1",
            input_sha256=sha, policy_version="safety-v1",
            detector_version="rules-v1", llm_status="skipped",
            final_action="block", completed_at=datetime(2026, 8, 3, 10, 0, 0, tzinfo=timezone.utc))
        session.add(run3)
        session.flush()
        session.add(models.SafetyFinding(
            review_run_id=run3.id, category="violence", severity=4,
            action="block", start_offset=2, end_offset=6,
            source_rule_ids='["LLM_violence"]', excerpt_sha256=sha))
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def seeded_client(safety_env):
    _seed_findings()
    yield safety_env


# ---------------------------------------------------------------------------
# List endpoint shape, validation, ordering
# ---------------------------------------------------------------------------

def test_findings_list_empty_page_shape(safety_env):
    resp = safety_env.get("/safety/findings", headers=AUTH)
    assert resp.json() == {"items": [], "total": 0, "limit": 20,
                           "offset": 0, "from": None, "to": None}


def test_findings_list_default_pagination_shape(seeded_client):
    resp = seeded_client.get("/safety/findings", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"items", "total", "limit", "offset", "from", "to"}
    assert body["limit"] == 20 and body["offset"] == 0
    assert body["from"] is None and body["to"] is None
    assert body["total"] == 3


def test_findings_list_limit_range_1_to_100(seeded_client):
    for bad in (0, 101):
        r = seeded_client.get(f"/safety/findings?limit={bad}", headers=AUTH)
        assert r.status_code == 422


def test_findings_list_invalid_from_returns_422(seeded_client):
    r = seeded_client.get("/safety/findings?from=not-a-date", headers=AUTH)
    assert r.status_code == 422


def test_findings_list_naive_timestamp_returns_422(seeded_client):
    r = seeded_client.get("/safety/findings?from=2026-08-01T10:00:00",
                          headers=AUTH)
    assert r.status_code == 422


def test_findings_list_reversed_window_returns_422(seeded_client):
    r = seeded_client.get(
        "/safety/findings?from=2026-08-02T00:00:00Z&to=2026-08-01T00:00:00Z",
        headers=AUTH)
    assert r.status_code == 422


def test_findings_list_sorts_created_desc_then_finding_id(seeded_client):
    body = seeded_client.get("/safety/findings?limit=50", headers=AUTH).json()
    keys = [(i["created_at"], i["id"]) for i in body["items"]]
    assert keys == sorted(keys, reverse=True)


def test_findings_list_normalizes_from_to_to_utc_z(seeded_client):
    r = seeded_client.get("/safety/findings?from=2026-08-01T00:00:00Z",
                          headers=AUTH).json()
    assert r["from"].endswith("Z")


# ---------------------------------------------------------------------------
# Detail endpoint
# ---------------------------------------------------------------------------

def test_findings_detail_returns_finding_and_run(seeded_client):
    fid = seeded_client.get("/safety/findings", headers=AUTH).json()["items"][0]["id"]
    body = seeded_client.get(f"/safety/findings/{fid}", headers=AUTH).json()
    assert "finding" in body and "run" in body
    assert body["finding"]["id"] == fid
    assert body["run"]["status"] == "succeeded"
    assert body["run"]["final_action"] in ("warn", "filter", "block")


def test_findings_detail_unknown_id_returns_404_with_literal_detail(seeded_client):
    r = seeded_client.get("/safety/findings/999999", headers=AUTH)
    assert r.status_code == 404
    assert r.json() == {"detail": "Safety finding not found"}


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

def test_stats_aggregates_sum_to_total_findings(seeded_client):
    body = seeded_client.get("/safety/stats", headers=AUTH).json()
    total = body["total_findings"]
    assert total == 3
    for key in ("by_policy_version", "by_category", "by_action", "by_scope"):
        assert sum(row["count"] for row in body[key]) == total


def test_stats_zero_results_use_empty_arrays(seeded_client):
    body = seeded_client.get(
        "/safety/stats?policy_version=nonexistent", headers=AUTH).json()
    assert body["total_findings"] == 0
    for key in ("by_policy_version", "by_category", "by_action", "by_scope"):
        assert body[key] == []


def test_stats_unknown_policy_version_returns_zero_not_404(seeded_client):
    r = seeded_client.get("/safety/stats?policy_version=never", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["total_findings"] == 0


def test_stats_policy_version_arrays_sort_lexically(seeded_client):
    body = seeded_client.get("/safety/stats", headers=AUTH).json()
    versions = [row["policy_version"] for row in body["by_policy_version"]]
    assert versions == sorted(versions)


# ---------------------------------------------------------------------------
# Auth matrix
# ---------------------------------------------------------------------------

def test_route_unavailable_when_operator_api_disabled(
    seeded_client, monkeypatch,
):
    monkeypatch.setenv("RAG_OPERATOR_API_ENABLED", "false")
    get_settings.cache_clear()
    for path in ("/safety/findings", "/safety/stats"):
        assert seeded_client.get(path, headers=AUTH).status_code == 404


def test_route_unavailable_when_safety_disabled(seeded_client, monkeypatch):
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "false")
    get_settings.cache_clear()
    for path in ("/safety/findings", "/safety/stats"):
        assert seeded_client.get(path, headers=AUTH).status_code == 404


def test_safety_disabled_yields_404_before_bearer_parsing_missing_bearer(
        seeded_client, monkeypatch):
    """D-63 regression: the feature-flag gate must run before auth, so a
    missing bearer under safety-disabled is 404, never 401."""
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "false")
    get_settings.cache_clear()
    for path in ("/safety/findings", "/safety/findings/1", "/safety/stats"):
        assert seeded_client.get(path).status_code == 404


def test_safety_disabled_yields_404_before_bearer_parsing_invalid_bearer(
        seeded_client, monkeypatch):
    """D-63 regression: invalid bearer under safety-disabled is 404 too."""
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "false")
    get_settings.cache_clear()
    for path in ("/safety/findings", "/safety/stats"):
        r = seeded_client.get(path, headers={"Authorization": "Bearer bad"})
        assert r.status_code == 404


def test_missing_bearer_returns_401_with_challenge(seeded_client):
    r = seeded_client.get("/safety/findings")
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_invalid_bearer_returns_401_with_challenge(seeded_client):
    r = seeded_client.get("/safety/findings",
                          headers={"Authorization": "Bearer bad"})
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


# ---------------------------------------------------------------------------
# bounded_excerpt NULL rules
# ---------------------------------------------------------------------------

def test_bounded_excerpt_null_for_privacy_credentials(seeded_client):
    body = seeded_client.get("/safety/findings?category=privacy_credentials",
                             headers=AUTH).json()
    for item in body["items"]:
        assert item["bounded_excerpt"] is None


def test_bounded_excerpt_null_for_answer_scope(seeded_client):
    body = seeded_client.get("/safety/findings?scope=answer",
                             headers=AUTH).json()
    for item in body["items"]:
        assert item["bounded_excerpt"] is None


def test_bounded_excerpt_null_for_filter_or_block_action(seeded_client):
    body = seeded_client.get("/safety/findings?action=block",
                             headers=AUTH).json()
    for item in body["items"]:
        assert item["bounded_excerpt"] is None


def test_bounded_excerpt_round_trips_verbatim_when_legitimate(seeded_client):
    """Positive control: a persisted non-null excerpt on a warn ingestion
    finding is returned verbatim by list and detail, proving the NULL rules
    above are enforced per category/scope/action and not by vacuous seeding
    (the seeder never writes a non-null excerpt)."""
    session = SessionLocal()
    try:
        finding = (
            session.query(models.SafetyFinding)
            .join(models.SafetyReviewRun,
                  models.SafetyFinding.review_run_id == models.SafetyReviewRun.id)
            .filter(models.SafetyReviewRun.scope == "ingestion",
                    models.SafetyFinding.action == "warn")
            .one())
        finding.bounded_excerpt = "KEPT-EXCERPT-123"
        session.commit()
        finding_id = finding.id
    finally:
        session.close()
    detail = seeded_client.get(f"/safety/findings/{finding_id}",
                               headers=AUTH).json()
    assert detail["finding"]["bounded_excerpt"] == "KEPT-EXCERPT-123"
    page = seeded_client.get("/safety/findings?category=violence",
                             headers=AUTH).json()
    item = next(i for i in page["items"] if i["id"] == finding_id)
    assert item["bounded_excerpt"] == "KEPT-EXCERPT-123"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def test_no_secret_sentinels_in_any_response(seeded_client):
    SENTINEL = "AKIAIOSFODNN7EXAMPLE"
    session = SessionLocal()
    try:
        document = models.Document(title="sentinel-source", source="unit",
                                   ingestion_status="ready")
        session.add(document)
        session.flush()
        chunk = models.Chunk(
            document_id=document.id, index=0,
            text=f"Config uses {SENTINEL} for access.",
            start_offset=0, end_offset=30, media_type="text/plain",
            vector_id=f"chunk:{document.id}:0")
        session.add(chunk)
        session.flush()
        run = models.SafetyReviewRun(
            scope="context", status="succeeded", document_id=document.id,
            chunk_id=chunk.id, document_id_snapshot=document.id,
            chunk_id_snapshot=chunk.id, retrieval_audit_id="audit-1",
            input_sha256="b" * 64, policy_version="safety-v1",
            detector_version="rules-v1", llm_status="skipped",
            final_action="filter", completed_at=datetime(2026, 8, 4, 10, 0, 0, tzinfo=timezone.utc))
        session.add(run)
        session.flush()
        session.add(models.SafetyFinding(
            review_run_id=run.id, category="privacy_credentials",
            severity=4, action="filter", start_offset=11, end_offset=30,
            source_rule_ids='["SAF006_privacy_credentials"]',
            excerpt_sha256=hashlib.sha256(SENTINEL.encode()).hexdigest()))
        session.commit()
    finally:
        session.close()

    findings_resp = seeded_client.get("/safety/findings?limit=100", headers=AUTH)
    stats_resp = seeded_client.get("/safety/stats", headers=AUTH)
    first_id = findings_resp.json()["items"][0]["id"]
    detail_resp = seeded_client.get(f"/safety/findings/{first_id}", headers=AUTH)
    for resp_text in (findings_resp.text, stats_resp.text, detail_resp.text):
        assert SENTINEL not in resp_text


# ---------------------------------------------------------------------------
# Source derivation
# ---------------------------------------------------------------------------

def test_source_field_derivation_deterministic_llm_merged(seeded_client):
    from app.api.routes_safety import derive_source

    assert derive_source(["SAF001_violence"]) == "deterministic"
    assert derive_source(
        ["SAF001_violence", "SAF006_privacy_credentials"]) == "deterministic"
    assert derive_source(["LLM_violence"]) == "llm"
    assert derive_source(
        ["LLM_violence", "LLM_privacy_credentials"]) == "llm"
    assert derive_source(["SAF001_violence", "LLM_violence"]) == "merged"
    assert derive_source(
        ["LLM_privacy_credentials", "SAF006_privacy_credentials"]) == "merged"
    assert derive_source(["LLM_violence", "SAF001_violence"]) == "merged"
    with pytest.raises(ValueError):
        derive_source(["XYZ_violence"])
    with pytest.raises(ValueError):
        derive_source(["SAF001_violence", "BAD_rule"])
    with pytest.raises(ValueError):
        derive_source([])

    body = seeded_client.get("/safety/findings?limit=100", headers=AUTH).json()
    for item in body["items"]:
        ids = item["source_rule_ids"]
        expected = ("deterministic" if all(r.startswith("SAF") for r in ids)
                    else "llm" if all(r.startswith("LLM_") for r in ids)
                    else "merged")
        assert item["source"] == expected == derive_source(ids)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_findings_filters_category_action_scope_document(seeded_client):
    body = seeded_client.get(
        "/safety/findings?category=violence&action=block&scope=answer",
        headers=AUTH).json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["category"] == "violence"
    assert item["action"] == "block"
    assert item["scope"] == "answer"
    assert item["source"] == "llm"

    stats = seeded_client.get(
        "/safety/stats?category=privacy_credentials", headers=AUTH).json()
    assert stats["total_findings"] == 1
    assert stats["by_action"] == [{"action": "filter", "count": 1}]
