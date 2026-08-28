"""Phase 10B.2 — prune_security_audits CLI tests (subprocess-based)."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRUNE_SCRIPT = PROJECT_ROOT / "scripts" / "prune_security_audits.py"


def _make_disposable_db(tmp_path: Path, audit_count: int = 0, completed_days_ago: int = 0) -> Path:
    """Create a disposable migrated SQLite DB with audit rows."""
    run_id = secrets.token_hex(16)
    db_path = tmp_path / f"prune-security-audits-{run_id}.db"
    # Create the DB with the c8 schema using the app's models
    db_url = f"sqlite:///{db_path}"
    from sqlalchemy import create_engine
    from app.core.db import Base
    from app.core.migrations import upgrade_database
    from app.persistence import models
    from datetime import datetime, timedelta, timezone
    from sqlalchemy.orm import sessionmaker

    upgrade_database(db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    if audit_count > 0:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=completed_days_ago)
        for i in range(audit_count):
            audit = models.RetrievalAudit(
                id=f"test-audit-{run_id}-{i:04d}",
                query_sha256=hashlib.sha256(f"query-{i}".encode()).hexdigest(),
                retrieval_mode="vector",
                status="completed",
                provenance_policy_version="source-trust-v1",
                retrieval_policy_version="retrieval-v1",
                context_policy_version="context-v1",
                candidate_count=1,
                selected_count=1,
                rejected_count=0,
                completed_at=cutoff_time,
            )
            session.add(audit)
            decision = models.RetrievalCandidateDecision(
                audit_id=f"test-audit-{run_id}-{i:04d}",
                document_id_snapshot=i + 1,
                chunk_id_snapshot=i + 1,
                decision="selected",
                provenance_score=0.5,
                reason_codes="[]",
                content_sha256="a" * 64,
            )
            session.add(decision)

    # Add one pending audit that should NOT be pruned
    pending = models.RetrievalAudit(
        id=f"pending-{run_id}",
        query_sha256=hashlib.sha256(b"pending").hexdigest(),
        retrieval_mode="vector",
        status="pending",
        provenance_policy_version="source-trust-v1",
        retrieval_policy_version="retrieval-v1",
        context_policy_version="context-v1",
        candidate_count=0,
        selected_count=0,
        rejected_count=0,
    )
    session.add(pending)
    session.commit()
    session.close()
    engine.dispose()
    return db_path


def _run_prune(db_path: Path, before_days: int, extra_args: list = None) -> subprocess.CompletedProcess:
    env = {**os.environ}
    argv = [
        sys.executable, str(PRUNE_SCRIPT),
        "--before-days", str(before_days),
        "--database-url", f"sqlite:///{db_path}",
        "--allow-disposable-database",
    ]
    if extra_args:
        argv.extend(extra_args)
    return subprocess.run(argv, capture_output=True, text=True, check=False, cwd=str(PROJECT_ROOT), env=env)


def test_prune_empty_db_exits_zero(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=0)
    result = _run_prune(db_path, before_days=30)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["deleted_audits"] == 0
    assert data["eligible_audits"] == 0


def test_prune_deletes_old_completed_audits(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=5, completed_days_ago=60)
    result = _run_prune(db_path, before_days=30)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["deleted_audits"] == 5
    assert data["eligible_audits"] == 5
    assert data["deleted_candidate_decisions"] == 5


def test_prune_retains_pending_audits(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=5, completed_days_ago=60)
    result = _run_prune(db_path, before_days=30)
    assert result.returncode == 0
    # Verify pending audit still exists
    conn = sqlite3.connect(str(db_path))
    pending = conn.execute("SELECT COUNT(*) FROM retrieval_audits WHERE status='pending'").fetchone()[0]
    conn.close()
    assert pending == 1


def test_prune_dry_run_preserves_all_data(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=5, completed_days_ago=60)
    original_bytes = db_path.read_bytes()
    result = _run_prune(db_path, before_days=30, extra_args=["--dry-run"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["dry_run"] is True
    assert data["deleted_audits"] == 0
    assert data["eligible_audits"] == 5
    # Data unchanged
    after_bytes = db_path.read_bytes()
    assert len(original_bytes) == len(after_bytes)


def test_prune_idempotent_second_run(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=5, completed_days_ago=60)
    first = _run_prune(db_path, before_days=30)
    assert first.returncode == 0
    second = _run_prune(db_path, before_days=30)
    assert second.returncode == 0
    data = json.loads(second.stdout)
    assert data["deleted_audits"] == 0
    assert data["eligible_audits"] == 0


def test_prune_rejects_invalid_before_days(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=1, completed_days_ago=60)
    result = _run_prune(db_path, before_days=0)
    assert result.returncode == 2


def test_prune_rejects_database_url_without_allow_disposable(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=1, completed_days_ago=60)
    argv = [
        sys.executable, str(PRUNE_SCRIPT),
        "--before-days", "30",
        "--database-url", f"sqlite:///{db_path}",
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=str(PROJECT_ROOT))
    assert result.returncode == 2


def test_prune_output_has_closed_keys(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=2, completed_days_ago=60)
    result = _run_prune(db_path, before_days=30)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    required_keys = {
        "schema_version", "head", "as_of_utc", "cutoff_utc",
        "before_days", "dry_run", "batch_size",
        "planned_batches", "applied_batches",
        "eligible_audits", "eligible_candidate_decisions",
        "eligible_safety_reviews", "eligible_safety_findings",
        "deleted_audits", "deleted_candidate_decisions",
        "deleted_safety_reviews", "deleted_safety_findings",
    }
    assert required_keys.issubset(set(data.keys()))
    assert data["schema_version"] == "security-audit-prune-v1"
    assert data["head"] == "c9f5b3e7a1d8"
    assert data["batch_size"] == 1000


# ---------------------------------------------------------------------------
# Appendix 10B.2 subprocess entries: two-batch atomic deletion over 1001
# eligible audits and unknown-argument refusal.

def test_c8_disposable_subprocess_contract_1001_audits_two_batches(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=1001, completed_days_ago=60)
    try:
        result = _run_prune(db_path, before_days=30)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["eligible_audits"] == 1001
        assert data["eligible_candidate_decisions"] == 1001
        assert data["planned_batches"] == 2
        assert data["applied_batches"] == 2
        assert data["deleted_audits"] == 1001
        assert data["deleted_candidate_decisions"] == 1001
        conn = sqlite3.connect(str(db_path))
        remaining_audits = conn.execute(
            "SELECT COUNT(*) FROM retrieval_audits WHERE status != 'pending'"
        ).fetchone()[0]
        remaining_decisions = conn.execute(
            "SELECT COUNT(*) FROM retrieval_candidate_decisions"
        ).fetchone()[0]
        conn.close()
        assert remaining_audits == 0
        assert remaining_decisions == 0
        # Second run is idempotent (zero-count success).
        second = _run_prune(db_path, before_days=30)
        assert second.returncode == 0, second.stderr
        data2 = json.loads(second.stdout)
        assert data2["eligible_audits"] == 0
        assert data2["deleted_audits"] == 0
        assert data2["applied_batches"] == 0
    finally:
        for suffix in ("", "-wal", "-shm"):
            p = db_path.with_name(db_path.name + suffix)
            if p.exists():
                p.unlink()


def test_c8_disposable_subprocess_contract_rejects_unknown_args(tmp_path):
    db_path = _make_disposable_db(tmp_path, audit_count=1, completed_days_ago=60)
    try:
        argv = [
            sys.executable, str(PRUNE_SCRIPT),
            "--before-days", "30",
            "--database-url", f"sqlite:///{db_path}",
            "--allow-disposable-database",
            "--bogus",
        ]
        result = subprocess.run(argv, capture_output=True, text=True, check=False,
                                cwd=str(PROJECT_ROOT))
        assert result.returncode == 2
        # The refused invocation deleted nothing.
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM retrieval_audits").fetchone()[0]
        conn.close()
        assert count == 2  # one completed + one pending
    finally:
        for suffix in ("", "-wal", "-shm"):
            p = db_path.with_name(db_path.name + suffix)
            if p.exists():
                p.unlink()
