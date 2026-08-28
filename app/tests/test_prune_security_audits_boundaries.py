"""Phase 10B remediation — prune CLI boundary and guard contract tests.

Covers the cutoff semantics regression (plan L1189: eligibility is exactly
``completed_at < cutoff`` on comparable instants; equality retained, pending
never eligible), the plan's ``(completed_at ASC, id ASC)`` ordering guarantee,
and the disposable-URL guard (plan L1187: a basename matching the disposable
pattern must still be refused when the resolved file IS the configured
production database, including via symlink/junction aliasing).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRUNE_SCRIPT = PROJECT_ROOT / "scripts" / "prune_security_audits.py"

_AS_OF = "2026-03-02T12:00:00Z"   # fixed instant → deterministic cutoff
# _CUTOFF = _AS_OF - 30 days = 2026-01-31T12:00:00Z


def _disposable_path(tmp_path: Path) -> Path:
    return tmp_path / f"prune-security-audits-{secrets.token_hex(16)}.db"


def _migrated(db_path: Path) -> Path:
    from app.core.migrations import upgrade_database

    upgrade_database(f"sqlite:///{db_path}")
    return db_path


def _seed_audit_at(db_path: Path, audit_id: str, when: datetime) -> None:
    """Insert one completed audit through the ORM (production storage format:
    SQLAlchemy's ' '-separated SQLite datetime string)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.persistence import models

    engine = create_engine(f"sqlite:///{db_path}")
    session = sessionmaker(bind=engine)()
    session.add(models.RetrievalAudit(
        id=audit_id,
        query_sha256=hashlib.sha256(audit_id.encode()).hexdigest(),
        retrieval_mode="vector",
        status="completed",
        provenance_policy_version="source-trust-v1",
        retrieval_policy_version="retrieval-v1",
        context_policy_version="context-v1",
        candidate_count=0,
        selected_count=0,
        rejected_count=0,
        completed_at=when.replace(tzinfo=None),
    ))
    session.commit()
    session.close()
    engine.dispose()


def _run_prune(db_path: Path, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
    argv = [
        sys.executable, str(PRUNE_SCRIPT),
        "--before-days", "30",
        "--database-url", f"sqlite:///{db_path}",
        "--allow-disposable-database",
        *extra,
    ]
    return subprocess.run(
        argv, capture_output=True, text=True, check=False,
        cwd=str(PROJECT_ROOT), env=env or os.environ.copy(),
    )


def _cleanup(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = db_path.with_name(db_path.name + suffix)
        if p.exists():
            p.unlink()


def _surviving_ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    ids = {row[0] for row in conn.execute("SELECT id FROM retrieval_audits")}
    conn.close()
    return ids


def test_prune_retains_audit_completed_after_cutoff_on_same_date(tmp_path):
    """The reproduced gate defect: an audit completed 6h AFTER the cutoff but
    on the same calendar date must be retained. A raw string comparison
    (' '-separated stored value vs 'T'-separated cutoff, ' ' < 'T') wrongly
    made every same-date audit eligible."""
    db_path = _migrated(_disposable_path(tmp_path))
    try:
        _seed_audit_at(db_path, "after-same-date", datetime(2026, 1, 31, 18, 0, 0))
        result = _run_prune(db_path, "--as-of-utc", _AS_OF)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["eligible_audits"] == 0
        assert data["deleted_audits"] == 0
        assert "after-same-date" in _surviving_ids(db_path)
    finally:
        _cleanup(db_path)


def test_prune_retains_audit_completed_exactly_at_cutoff(tmp_path):
    """Equality at the boundary is retained (strict ``<``), while one
    microsecond earlier is eligible."""
    db_path = _migrated(_disposable_path(tmp_path))
    try:
        _seed_audit_at(db_path, "boundary-equal", datetime(2026, 1, 31, 12, 0, 0))
        _seed_audit_at(db_path, "one-microsecond-before",
                       datetime(2026, 1, 31, 11, 59, 59, 999999))
        result = _run_prune(db_path, "--as-of-utc", _AS_OF)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["eligible_audits"] == 1
        assert data["deleted_audits"] == 1
        survivors = _surviving_ids(db_path)
        assert "boundary-equal" in survivors
        assert "one-microsecond-before" not in survivors
    finally:
        _cleanup(db_path)


def test_prune_day_boundary_deletes_before_retains_after(tmp_path):
    db_path = _migrated(_disposable_path(tmp_path))
    try:
        _seed_audit_at(db_path, "previous-day", datetime(2026, 1, 30, 23, 0, 0))
        _seed_audit_at(db_path, "after-same-date", datetime(2026, 1, 31, 18, 0, 0))
        _seed_audit_at(db_path, "next-day", datetime(2026, 2, 1, 0, 0, 0))
        result = _run_prune(db_path, "--as-of-utc", _AS_OF)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["eligible_audits"] == 1
        assert data["deleted_audits"] == 1
        survivors = _surviving_ids(db_path)
        assert "previous-day" not in survivors
        assert {"after-same-date", "next-day"} <= survivors
    finally:
        _cleanup(db_path)


def test_prune_pending_never_eligible_at_boundary(tmp_path):
    """Pending audits are never eligible even when older than the cutoff."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.persistence import models

    db_path = _migrated(_disposable_path(tmp_path))
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        session = sessionmaker(bind=engine)()
        session.add(models.RetrievalAudit(
            id="old-pending",
            query_sha256=hashlib.sha256(b"old-pending").hexdigest(),
            retrieval_mode="vector",
            status="pending",
            provenance_policy_version="source-trust-v1",
            retrieval_policy_version="retrieval-v1",
            context_policy_version="context-v1",
            candidate_count=0,
            selected_count=0,
            rejected_count=0,
        ))
        session.commit()
        session.close()
        engine.dispose()

        result = _run_prune(db_path, "--as-of-utc", _AS_OF)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["eligible_audits"] == 0
        assert data["deleted_audits"] == 0
        assert "old-pending" in _surviving_ids(db_path)
    finally:
        _cleanup(db_path)


def test_prune_eligible_ids_ordered_by_completed_at_then_id(tmp_path):
    """Plan ordering guarantee: eligible IDs are selected in
    ``(completed_at ASC, id ASC)`` order, computed on parsed instants (not
    mixed-format strings), with the id tie-break at equal instants."""
    from datetime import timezone

    from scripts.prune_security_audits import _select_eligible

    db_path = tmp_path / "ordering.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE retrieval_audits (id TEXT, status TEXT, completed_at TEXT)")
    rows = [
        # Deliberately inserted against the expected consumption order.
        ("c", "2026-01-01 10:00:00.000000"),
        ("a", "2026-01-03 09:00:00.000000"),
        ("b", "2026-01-02 23:59:59.999999"),
        ("z", "2026-01-01 10:00:00"),          # equal instant to "c", later id
        ("iso-t", "2026-01-02T08:00:00.000Z"),  # ISO 'T'-separated stored form
        ("pending-1", None),
        ("newer", "2026-06-01 00:00:00.000000"),
    ]
    for audit_id, completed in rows:
        status = "pending" if completed is None else "completed"
        conn.execute(
            "INSERT INTO retrieval_audits VALUES (?, ?, ?)",
            (audit_id, status, completed),
        )
    conn.commit()

    cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
    eligible = _select_eligible(conn, cutoff)
    conn.close()

    assert eligible == ["c", "z", "iso-t", "b", "a"]


def test_prune_fails_closed_on_unparseable_terminal_completed_at(tmp_path):
    """A terminal audit whose completed_at cannot be parsed fails the
    invocation closed (exit 2) instead of silently misordering it."""
    db_path = _migrated(_disposable_path(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO retrieval_audits (id, query_sha256, retrieval_mode, "
            "status, provenance_policy_version, retrieval_policy_version, "
            "context_policy_version, candidate_count, selected_count, "
            "rejected_count, completed_at) "
            "VALUES ('garbage','h','vector','completed','v','r','c',0,0,0,'not-a-date')")
        conn.commit()
        conn.close()
        result = _run_prune(db_path, "--as-of-utc", _AS_OF)
        assert result.returncode == 2
        assert "eligibility" in result.stderr.lower()
        assert _surviving_ids(db_path) >= {"garbage"}
    finally:
        _cleanup(db_path)


# ---------------------------------------------------------------------------
# Disposable-URL guard (plan L1187).

def test_prune_refuses_disposable_url_equal_to_production_path(tmp_path):
    run_id = secrets.token_hex(16)
    name = f"prune-security-audits-{run_id}.db"
    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()
    prod_db = prod_dir / name
    prod_db.write_bytes(b"production database bytes")
    env = {**os.environ, "RAG_DATABASE_URL": f"sqlite:///{prod_db}"}
    result = _run_prune(prod_db, env=env)
    assert result.returncode == 2
    assert "production" in result.stderr.lower()
    # The production file is untouched and its path is never printed.
    assert prod_db.read_bytes() == b"production database bytes"
    assert str(prod_db) not in result.stderr


def test_prune_refuses_alias_resolving_to_production_path(tmp_path):
    """A symlink to production (or, on Windows without symlink privilege, a
    directory junction over the production directory) is refused."""
    import platform

    run_id = secrets.token_hex(16)
    name = f"prune-security-audits-{run_id}.db"
    prod_dir = tmp_path / "prod"
    prod_dir.mkdir()
    prod_db = prod_dir / name
    prod_db.write_bytes(b"production database bytes")
    alias_dir = tmp_path / "alias"

    alias_kind = None
    try:
        alias_dir.symlink_to(prod_dir)
        alias_kind = "symlink"
    except OSError:
        if platform.system() == "Windows":
            import _winapi

            _winapi.CreateJunction(str(prod_dir), str(alias_dir))
            alias_kind = "junction"
        else:
            pytest.skip("symlinks not supported")
    try:
        alias_db = alias_dir / name
        assert alias_db.exists()
        env = {**os.environ, "RAG_DATABASE_URL": f"sqlite:///{prod_db}"}
        result = _run_prune(alias_db, env=env)
        assert result.returncode == 2
        assert "production" in result.stderr.lower()
        assert prod_db.read_bytes() == b"production database bytes"
    finally:
        if alias_kind == "symlink":
            alias_dir.unlink()
        elif alias_kind == "junction":
            os.rmdir(alias_dir)


def test_prune_accepts_genuine_disposable_path_with_production_configured(tmp_path):
    db_path = _migrated(_disposable_path(tmp_path))
    elsewhere = tmp_path / "elsewhere" / "prod.db"
    env = {**os.environ, "RAG_DATABASE_URL": f"sqlite:///{elsewhere}"}
    try:
        result = _run_prune(db_path, env=env)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["eligible_audits"] == 0
        # The configured production path was never created or touched.
        assert not elsewhere.exists()
    finally:
        _cleanup(db_path)


def test_prune_rejects_non_sqlite_disposable_url():
    run_id = secrets.token_hex(16)
    bad_url = f"postgresql://localhost/prune-security-audits-{run_id}.db"
    argv = [
        sys.executable, str(PRUNE_SCRIPT),
        "--before-days", "30",
        "--database-url", bad_url,
        "--allow-disposable-database",
    ]
    result = subprocess.run(
        argv, capture_output=True, text=True, check=False, cwd=str(PROJECT_ROOT))
    assert result.returncode == 2
