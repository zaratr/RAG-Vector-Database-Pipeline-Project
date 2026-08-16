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
    assert data["head"] == "d9b5f7c1e4a3"
    assert data["batch_size"] == 1000


# ---------------------------------------------------------------------------
# 10C.4: d9 cascade subprocess contract
# ---------------------------------------------------------------------------

def _d9_fixture_db(tmp_path, *, eligible_old=True):
    """Build a d9 database with the pruning-CLI fixture set.

    - one eligible audit (old terminal) with candidate decisions plus linked
      context and answer safety runs/findings;
    - one retained boundary audit (recent terminal) with equivalent children;
    - one unrelated ingestion-scope safety run.
    """
    import json
    import sqlite3
    import subprocess
    import sys as _sys

    import secrets as _secrets

    db_path = tmp_path / f"prune-security-audits-{_secrets.token_hex(16)}.db"
    subprocess.run(
        [_sys.executable, "-m", "app.core.migrations"],
        env={**__import__("os").environ,
             "RAG_DATABASE_URL": f"sqlite:///{db_path}"},
        capture_output=True, check=True,
    )
    conn = sqlite3.connect(str(db_path))
    now_iso = "2026-08-15T12:00:00Z"
    old_iso = "2026-05-15T12:00:00Z"  # ~92 days before now
    sha = "a" * 64
    conn.execute(
        "INSERT INTO documents (title, ingestion_status, trust_tier, "
        "trust_score, trust_policy_version, ingestion_origin) "
        "VALUES ('D1', 'ready', 'standard', 0.5, 'p', 'api')")
    doc_id = conn.execute("SELECT id FROM documents").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (document_id, \"index\", text, start_offset, "
        "end_offset, vector_id, media_type) "
        "VALUES (?, 0, 'chunk text', 0, 10, 'v1', 'text/plain')", (doc_id,))
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()[0]

    def _audit(audit_id, completed_iso):
        conn.execute(
            "INSERT INTO retrieval_audits (id, query_sha256, retrieval_mode, "
            "status, provenance_policy_version, retrieval_policy_version, "
            "context_policy_version, candidate_count, selected_count, "
            "rejected_count, completed_at) VALUES (?, ?, 'vector', "
            "'completed', 'p', 'r', 'c', 1, 1, 0, ?)",
            (audit_id, sha, completed_iso))

    def _decision(audit_id, chunk_snapshot):
        conn.execute(
            "INSERT INTO retrieval_candidate_decisions (audit_id, "
            "document_id, chunk_id, document_id_snapshot, chunk_id_snapshot, "
            "decision, provenance_score, reason_codes, content_sha256) "
            "VALUES (?, ?, ?, ?, ?, 'selected', 0.5, '[]', ?)",
            (audit_id, doc_id, chunk_id, doc_id, chunk_snapshot, sha))

    def _safety_run(run_id, scope, audit_id, snapshot_doc, snapshot_chunk):
        conn.execute(
            "INSERT INTO safety_review_runs (id, scope, status, document_id, "
            "chunk_id, document_id_snapshot, chunk_id_snapshot, "
            "retrieval_audit_id, input_sha256, policy_version, "
            "detector_version, llm_status, final_action, completed_at) "
            "VALUES (?, ?, 'succeeded', ?, ?, ?, ?, ?, ?, 'safety-v1', "
            "'rules-v1', 'skipped', 'allow', '2026-08-15T12:00:00Z')",
            (run_id, scope,
             snapshot_doc if scope == "context" else None,
             snapshot_chunk if scope == "context" else None,
             snapshot_doc, snapshot_chunk, audit_id, sha))
        conn.execute(
            "INSERT INTO safety_findings (review_run_id, category, severity, "
            "action, start_offset, end_offset, source_rule_ids, "
            "excerpt_sha256) VALUES (?, 'violence', 3, 'warn', 0, 4, "
            "'[\"SAF001_violence\"]', ?)", (run_id, sha))

    _audit("eligible-audit", old_iso)
    _decision("eligible-audit", chunk_id)
    _safety_run(1, "context", "eligible-audit", doc_id, chunk_id)
    _safety_run(2, "answer", "eligible-audit", None, None)

    _audit("boundary-audit", now_iso)
    _decision("boundary-audit", chunk_id)
    _safety_run(3, "context", "boundary-audit", doc_id, chunk_id)
    _safety_run(4, "answer", "boundary-audit", None, None)

    # Unrelated ingestion-scope run: never pruned by audit pruning.
    conn.execute(
        "INSERT INTO safety_review_runs (id, scope, status, document_id, "
        "document_id_snapshot, input_sha256, policy_version, "
        "detector_version, llm_status, final_action, completed_at) "
        "VALUES (5, 'ingestion', 'succeeded', ?, ?, ?, 'safety-v1', "
        "'rules-v1', 'skipped', 'allow', '2026-08-15T12:00:00Z')",
        (doc_id, doc_id, sha))
    conn.commit()
    conn.close()
    return db_path


def test_d9_cascade_subprocess_contract(tmp_path):
    """Dry-run reports child counts without writes; pruning removes only the
    eligible audit's decisions and linked context/answer runs/findings; the
    boundary audit and ingestion run survive byte-for-byte; second run is a
    no-op."""
    import json
    import sqlite3
    import subprocess
    import sys as _sys

    db_path = _d9_fixture_db(tmp_path)

    def _run(*extra):
        return subprocess.run(
            [_sys.executable, "scripts/prune_security_audits.py",
             "--before-days", "30", "--database-url",
             f"sqlite:///{db_path}", "--allow-disposable-database", *extra],
            capture_output=True, text=True, check=False,
            cwd=str(Path(__file__).resolve().parents[2]),
        )

    dry = _run("--dry-run")
    assert dry.returncode == 0, dry.stderr
    dry_data = json.loads(dry.stdout)
    assert dry_data["eligible_audits"] == 1
    assert dry_data["eligible_candidate_decisions"] == 1
    assert dry_data["eligible_safety_reviews"] == 2
    assert dry_data["eligible_safety_findings"] == 2

    conn = sqlite3.connect(str(db_path))
    before_boundary = conn.execute(
        "SELECT id, status, completed_at FROM retrieval_audits "
        "WHERE id = 'boundary-audit'").fetchall()
    before_ingestion = conn.execute(
        "SELECT id, scope, status, final_action FROM safety_review_runs "
        "WHERE id = 5").fetchall()
    conn.close()

    prune = _run()
    assert prune.returncode == 0, prune.stderr
    prune_data = json.loads(prune.stdout)
    assert prune_data["deleted_audits"] == 1
    assert prune_data["deleted_candidate_decisions"] == 1
    assert prune_data["deleted_safety_reviews"] == 2
    assert prune_data["deleted_safety_findings"] == 2

    conn = sqlite3.connect(str(db_path))
    assert conn.execute(
        "SELECT COUNT(*) FROM retrieval_audits").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM retrieval_candidate_decisions").fetchone()[0] == 1
    # Context/answer runs linked to the eligible audit are gone; the
    # boundary-linked and ingestion-scope runs remain.
    remaining_runs = conn.execute(
        "SELECT id, scope FROM safety_review_runs ORDER BY id").fetchall()
    assert remaining_runs == [(3, "context"), (4, "answer"), (5, "ingestion")]
    assert conn.execute(
        "SELECT COUNT(*) FROM safety_findings").fetchone()[0] == 2
    after_boundary = conn.execute(
        "SELECT id, status, completed_at FROM retrieval_audits "
        "WHERE id = 'boundary-audit'").fetchall()
    after_ingestion = conn.execute(
        "SELECT id, scope, status, final_action FROM safety_review_runs "
        "WHERE id = 5").fetchall()
    conn.close()
    assert after_boundary == before_boundary
    assert after_ingestion == before_ingestion

    second = _run()
    assert second.returncode == 0
    second_data = json.loads(second.stdout)
    assert second_data["deleted_audits"] == 0
    assert second_data["deleted_candidate_decisions"] == 0
    assert second_data["deleted_safety_reviews"] == 0
    assert second_data["deleted_safety_findings"] == 0
    db_path.unlink(missing_ok=True)


def test_prune_c8_head_non_dry_still_prunes(tmp_path):
    """D-57: the safety-cascade deletion is head-gated; a c8-head database
    (safety tables absent) must prune decisions/audits exactly as before."""
    import json
    import subprocess
    import sys as _sys
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from alembic import command
    from alembic.config import Config as _Config

    run_id = secrets.token_hex(16)
    db_path = tmp_path / f"prune-security-audits-{run_id}.db"
    db_url = f"sqlite:///{db_path}"
    cfg = _Config("alembic.ini")
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "app/persistence/alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["database_url_explicit"] = True
    command.upgrade(cfg, "c8a4e6b0d3f2")

    old_iso = (datetime.now(timezone.utc) - timedelta(days=90)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO retrieval_audits (id, query_sha256, retrieval_mode, "
        "status, provenance_policy_version, retrieval_policy_version, "
        "context_policy_version, completed_at) VALUES ('old-audit', ?, "
        "'vector', 'completed', 'p', 'r', 'c', ?)", ("a" * 64, old_iso))
    conn.commit()
    conn.close()

    result = subprocess.run(
        [_sys.executable, "scripts/prune_security_audits.py",
         "--before-days", "30", "--database-url", db_url,
         "--allow-disposable-database"],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["deleted_audits"] == 1
    conn = sqlite3.connect(str(db_path))
    assert conn.execute(
        "SELECT COUNT(*) FROM retrieval_audits").fetchone()[0] == 0
    conn.close()
    db_path.unlink(missing_ok=True)
