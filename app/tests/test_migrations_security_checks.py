"""Phase 10B remediation — direct-SQL CHECK behavioral set for the c8
security schema and the c9f5b3e7a1d8 decision-CHECK follow-up revision.

The constraints exist in the migration chain and the ORM; these tests prove
the database itself rejects violations (trust tier/score, audit lifecycle,
retrieval mode, decision uniqueness, provenance-score range, rate-bucket
composite PK) and that the decision CHECK revision upgrades/downgrades
losslessly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from app.core.migrations import upgrade_database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

SECURITY_REVISION = "c8a4e6b0d3f2"
SECURITY_DOWN_REVISION = "b7f3d5a9c2e1"
DECISION_CHECK_HEAD = "c9f5b3e7a1d8"
# Merged chain head (content-safety + rag-poisoning): d9 on top of c9.
CHAIN_HEAD = "d9b5f7c1e4a3"


def _db_url(path: Path) -> str:
    return f"sqlite:///{path}"


def _alembic_config(db_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option(
        "script_location", str(PROJECT_ROOT / "app" / "persistence" / "alembic")
    )
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["database_url_explicit"] = True
    return cfg


def _revision(engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def _insert_audit(conn, audit_id="a", status="pending", mode="hybrid",
                  candidates=0, selected=0, rejected=0,
                  completed=None, failure=None):
    conn.execute(
        text(
            "INSERT INTO retrieval_audits (id, query_sha256, retrieval_mode, "
            "status, provenance_policy_version, retrieval_policy_version, "
            "context_policy_version, candidate_count, selected_count, "
            "rejected_count, completed_at, failure_code) "
            "VALUES (:id, :h, :mode, :status, 'v', 'r', 'c', :cand, :sel, :rej, "
            ":completed, :failure)"
        ),
        {"id": audit_id, "h": "a" * 64, "mode": mode, "status": status,
         "cand": candidates, "sel": selected, "rej": rejected,
         "completed": completed, "failure": failure},
    )


def _insert_decision(conn, audit_id="a", doc=1, chunk=1, decision="selected",
                     score=0.5, sha="x"):
    conn.execute(
        text(
            "INSERT INTO retrieval_candidate_decisions (audit_id, "
            "document_id_snapshot, chunk_id_snapshot, decision, "
            "provenance_score, reason_codes, content_sha256) "
            "VALUES (:audit, :doc, :chunk, :decision, :score, '[]', :sha)"
        ),
        {"audit": audit_id, "doc": doc, "chunk": chunk, "decision": decision,
         "score": score, "sha": sha},
    )


# ---------------------------------------------------------------------------
# c8 direct-SQL CHECK behavioral set (appendix 10B.2).

def test_c8_documents_trust_tier_check_rejects_invalid_value(tmp_path):
    db_url = _db_url(tmp_path / "security-tier-check.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO documents (title, trust_tier) "
                              "VALUES ('bad', 'god-mode')"))
    engine.dispose()


def test_c8_documents_trust_score_check_rejects_out_of_range(tmp_path):
    db_url = _db_url(tmp_path / "security-score-check.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    for bad_score in (-0.01, 1.01):
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO documents (title, trust_score) "
                    f"VALUES ('bad', {bad_score})"))
    # Boundary values are legal.
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO documents (title, trust_score) "
                          "VALUES ('lo', 0.0)"))
        conn.execute(text("INSERT INTO documents (title, trust_score) "
                          "VALUES ('hi', 1.0)"))
    engine.dispose()


def test_c8_retrieval_audits_lifecycle_checks(tmp_path):
    """pending: no completion/failure; completed: completion/no failure;
    failed: completion + failure; all: candidate = selected + rejected."""
    db_url = _db_url(tmp_path / "audit-lifecycle.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)

    # pending must not carry completed_at or failure_code
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "p1", completed="2026-01-01 00:00:00")
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "p1f", failure="code")
    # completed requires completed_at and no failure code
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "c1", status="completed")
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "c2", status="completed",
                          completed="2026-01-01 00:00:00", failure="late_code")
    # failed requires completed_at AND a failure code
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "f1", status="failed",
                          completed="2026-01-01 00:00:00")
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "f2", status="failed", failure="no_completion")
    # unknown status itself rejected
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "s1", status="cancelled",
                          completed="2026-01-01 00:00:00", failure="x")
    # counter equality enforced in every state
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "p2", candidates=2, selected=1, rejected=0)
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "c3", status="completed",
                          completed="2026-01-01 00:00:00",
                          candidates=3, selected=1, rejected=1)
    # Legal rows in all three states are accepted.
    with engine.begin() as conn:
        _insert_audit(conn, "ok-p")
        _insert_audit(conn, "ok-c", status="completed",
                      completed="2026-01-01 00:00:00", candidates=1, selected=1)
        _insert_audit(conn, "ok-f", status="failed",
                      completed="2026-01-01 00:00:00", failure="provider",
                      candidates=1, rejected=1)
    engine.dispose()


def test_c8_retrieval_mode_check_rejects_invalid_value(tmp_path):
    db_url = _db_url(tmp_path / "audit-mode.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_audit(conn, "x", mode="semantic")
    engine.dispose()


def test_c8_candidate_decisions_unique_per_audit_and_chunk(tmp_path):
    db_url = _db_url(tmp_path / "decision-unique.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        _insert_audit(conn, "a", candidates=1, selected=1)
        _insert_decision(conn, "a", doc=1, chunk=1, decision="selected", sha="x")
    # Duplicate (audit_id, chunk_id_snapshot) rejected even with a different
    # decision and document snapshot.
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            _insert_decision(conn, "a", doc=2, chunk=1,
                             decision="rejected_duplicate", score=0.3, sha="y")
    # The same chunk under a different audit is legal.
    with engine.begin() as conn:
        _insert_audit(conn, "b", candidates=1, rejected=1)
        _insert_decision(conn, "b", doc=2, chunk=1,
                         decision="rejected_duplicate", sha="y")
    engine.dispose()


def test_c8_provenance_score_check_rejects_out_of_range(tmp_path):
    db_url = _db_url(tmp_path / "decision-score.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        _insert_audit(conn, "a", candidates=1, selected=1)
    for bad in (-0.01, 1.01):
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                _insert_decision(conn, "a", doc=1, chunk=1,
                                 decision="selected", score=bad, sha="z")
    # Boundary scores are legal.
    with engine.begin() as conn:
        _insert_decision(conn, "a", doc=2, chunk=2, score=0.0, sha="lo")
        _insert_decision(conn, "a", doc=3, chunk=3, score=1.0, sha="hi")
    engine.dispose()


def test_c8_ingestion_rate_buckets_composite_pk(tmp_path):
    db_url = _db_url(tmp_path / "rate-buckets.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO ingestion_rate_buckets (identity_sha256, "
            "window_start_epoch, request_count) VALUES ('id1', 1000, 1)"))
    # Duplicate composite PK rejected.
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO ingestion_rate_buckets (identity_sha256, "
                "window_start_epoch, request_count) VALUES ('id1', 1000, 2)"))
    # Same identity in a different window is a distinct bucket.
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO ingestion_rate_buckets (identity_sha256, "
            "window_start_epoch, request_count) VALUES ('id1', 2000, 1)"))
    # request_count must be > 0.
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO ingestion_rate_buckets (identity_sha256, "
                "window_start_epoch, request_count) VALUES ('id2', 2000, 0)"))
    engine.dispose()


def test_c8_downgrade_to_b7_refused_when_skipped_rows_absent_preserves_data(tmp_path):
    """Downgrade from the security head to b7 is allowed when no rows depend
    on the security-only tables; all rows/IDs survive; re-upgrade is
    lossless."""
    db_url = _db_url(tmp_path / "c8-downgrade.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO documents (id, title, trust_tier) "
                          "VALUES (1, 'Pre', 'untrusted')"))
    engine.dispose()

    command.downgrade(_alembic_config(db_url), SECURITY_DOWN_REVISION)
    engine = create_engine(db_url)
    assert _revision(engine) == SECURITY_DOWN_REVISION
    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT title FROM documents WHERE id=1")).scalar() == "Pre"
        # Security-only tables are gone after downgrade.
        assert conn.execute(text(
            "SELECT name FROM sqlite_master WHERE name IN "
            "('retrieval_audits','retrieval_candidate_decisions',"
            "'ingestion_rate_buckets')")).fetchall() == []
    engine.dispose()

    # Re-upgrade is lossless.
    upgrade_database(db_url)
    engine = create_engine(db_url)
    assert _revision(engine) == CHAIN_HEAD
    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT title FROM documents WHERE id=1")).scalar() == "Pre"
    engine.dispose()


# ---------------------------------------------------------------------------
# Decision CHECK follow-up revision (c9f5b3e7a1d8, plan L1160/L1205): the
# complete 10B.3 decision enum enforced at the database level.

def test_c9_decision_check_exists_at_head_but_not_at_c8(tmp_path):
    head_url = _db_url(tmp_path / "decision-check-head.db")
    upgrade_database(head_url)
    engine = create_engine(head_url)
    checks = {c["name"] for c in inspect(engine).get_check_constraints(
        "retrieval_candidate_decisions")}
    assert "ck_candidate_decisions_decision" in checks
    engine.dispose()

    c8_url = _db_url(tmp_path / "decision-check-c8.db")
    command.upgrade(_alembic_config(c8_url), SECURITY_REVISION)
    engine = create_engine(c8_url)
    checks = {c["name"] for c in inspect(engine).get_check_constraints(
        "retrieval_candidate_decisions")}
    assert "ck_candidate_decisions_decision" not in checks
    engine.dispose()


def test_c9_decision_check_rejects_invalid_value(tmp_path):
    db_url = _db_url(tmp_path / "decision-check-value.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        _insert_audit(conn, "a", candidates=1, selected=1)
    for bad in ("accepted", "rejected", "maybe"):
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                _insert_decision(conn, "a", doc=1, chunk=1, decision=bad, sha="bad")
    # Every documented 10B.3 decision value is accepted at the boundary, and
    # the merged d9 head additionally accepts 10C.4's rejected_safety (its
    # legality is pinned by test_migrations.py's d9 CHECK tests).
    legal = ("selected", "rejected_distance", "rejected_blocked_source",
             "rejected_source_cap", "rejected_document_cap",
             "rejected_duplicate", "rejected_injection", "rejected_safety")
    with engine.begin() as conn:
        for i, decision in enumerate(legal, start=1):
            _insert_decision(conn, "a", doc=i, chunk=i, decision=decision,
                             sha=f"{i:064d}")
    engine.dispose()


def test_c9_upgrade_from_c8_head_preserves_decision_rows(tmp_path):
    db_url = _db_url(tmp_path / "decision-preserve.db")
    command.upgrade(_alembic_config(db_url), SECURITY_REVISION)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        _insert_audit(conn, "a", candidates=3, selected=1, rejected=2)
        _insert_decision(conn, "a", doc=1, chunk=1, decision="selected",
                         score=0.5, sha="1" * 64)
        _insert_decision(conn, "a", doc=2, chunk=2, decision="rejected_duplicate",
                         score=0.3, sha="2" * 64)
        _insert_decision(conn, "a", doc=3, chunk=3, decision="rejected_injection",
                         score=0.0, sha="3" * 64)
    engine.dispose()

    upgrade_database(db_url)
    engine = create_engine(db_url)
    assert _revision(engine) == CHAIN_HEAD
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT chunk_id_snapshot, decision, provenance_score, "
            "content_sha256 FROM retrieval_candidate_decisions "
            "ORDER BY chunk_id_snapshot")).fetchall()
    assert [tuple(r) for r in rows] == [
        (1, "selected", 0.5, "1" * 64),
        (2, "rejected_duplicate", 0.3, "2" * 64),
        (3, "rejected_injection", 0.0, "3" * 64),
    ]
    engine.dispose()


def test_c9_orm_metadata_matches_physical_schema(tmp_path):
    db_url = _db_url(tmp_path / "decision-orm-match.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)

    from app.persistence.models import RetrievalCandidateDecision

    insp = inspect(engine)
    orm_columns = {c.name for c in RetrievalCandidateDecision.__table__.columns}
    physical_columns = {c["name"] for c in insp.get_columns(
        "retrieval_candidate_decisions")}
    assert orm_columns == physical_columns

    orm_checks = {
        c.name for c in RetrievalCandidateDecision.__table__.constraints
        if isinstance(c, sqlalchemy.CheckConstraint)
    }
    physical_checks = {
        c["name"] for c in insp.get_check_constraints(
            "retrieval_candidate_decisions")
    }
    assert orm_checks == physical_checks
    assert "ck_candidate_decisions_decision" in physical_checks
    engine.dispose()
