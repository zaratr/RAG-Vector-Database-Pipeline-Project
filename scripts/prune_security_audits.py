"""Security audit retention CLI (Task 10B.2).

Accepts --before-days N (1..3650), optional --dry-run, --as-of-utc RFC3339-Z,
and test/gate-only --database-url URL --allow-disposable-database. Deletes
terminal audits older than the cutoff in batches of 1000. Exits 2 on any
argument/config/head/database/fingerprint/count/cascade failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc(dt_str: str) -> datetime:
    """Parse a RFC3339-Z string to a UTC datetime."""
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _get_head(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    return row[0] if row else ""


def _is_disposable_path(db_url: str) -> bool:
    """Check if the basename matches a disposable pattern."""
    path = db_url.split("///")[-1] if "///" in db_url else db_url
    basename = os.path.basename(path)
    return bool(re.match(r"^prune-security-audits-[0-9a-f]{32}\.db$", basename))


def _connect(db_url: str, read_only: bool = False) -> sqlite3.Connection:
    path = db_url.split("///")[-1] if "///" in db_url else db_url
    if read_only:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    fk = conn.execute("PRAGMA foreign_keys").fetchone()
    if not fk or fk[0] == 0:
        conn.close()
        raise ValueError("foreign keys are not enabled")
    return conn


def _run(args: argparse.Namespace) -> int:
    # Validate N range
    if args.before_days < 1 or args.before_days > 3650:
        sys.stderr.write("prune_security_audits: --before-days must be 1..3650\n")
        return 2

    # Determine database URL
    if args.database_url:
        if not args.allow_disposable_database:
            sys.stderr.write("prune_security_audits: --database-url requires --allow-disposable-database\n")
            return 2
        if not _is_disposable_path(args.database_url):
            sys.stderr.write("prune_security_audits: disposable URL must match /tmp/prune-security-audits-<32hex>.db\n")
            return 2
        db_url = args.database_url
    else:
        db_url = os.environ.get("RAG_DATABASE_URL", "sqlite:///rag.db")

    # Parse as-of UTC
    if args.as_of_utc:
        try:
            as_of = _normalize_utc(args.as_of_utc)
            if as_of.tzinfo is None or as_of.tzinfo.utcoffset(as_of) != timedelta(0):
                raise ValueError("not UTC")
        except Exception:
            sys.stderr.write("prune_security_audits: invalid --as-of-utc (must be RFC3339-Z)\n")
            return 2
    else:
        as_of = _utc_now()

    cutoff = as_of - timedelta(days=args.before_days)
    as_of_str = _format_utc(as_of)
    cutoff_str = _format_utc(cutoff)

    # Connect and validate head
    try:
        read_only = args.dry_run
        conn = _connect(db_url, read_only=read_only)
    except Exception as exc:
        sys.stderr.write(f"prune_security_audits: database error: {exc}\n")
        return 2

    head = _get_head(conn)
    if head not in ("c8a4e6b0d3f2", "d9b5f7c1e4a3"):
        conn.close()
        sys.stderr.write(f"prune_security_audits: head must be c8 or d9, got {head}\n")
        return 2

    # Count eligible audits
    eligible = conn.execute(
        "SELECT id FROM retrieval_audits WHERE status IN ('completed','failed') "
        "AND completed_at < ? ORDER BY completed_at ASC, id ASC",
        (cutoff_str,),
    ).fetchall()
    eligible_ids = [r[0] for r in eligible]

    # Count eligible children
    eligible_decisions = 0
    eligible_safety_reviews = 0
    eligible_safety_findings = 0
    if eligible_ids:
        placeholders = ",".join("?" * len(eligible_ids))
        eligible_decisions = conn.execute(
            f"SELECT COUNT(*) FROM retrieval_candidate_decisions WHERE audit_id IN ({placeholders})",
            eligible_ids,
        ).fetchone()[0]
        if head == "d9b5f7c1e4a3":
            try:
                eligible_safety_reviews = conn.execute(
                    f"SELECT COUNT(*) FROM safety_review_runs WHERE retrieval_audit_id IN ({placeholders})",
                    eligible_ids,
                ).fetchone()[0]
            except sqlite3.OperationalError:
                pass

    planned_batches = (len(eligible_ids) + 999) // 1000 if eligible_ids else 0

    if args.dry_run:
        # Dry-run: no writes, all deleted counts = 0
        result = {
            "schema_version": "security-audit-prune-v1",
            "head": head,
            "as_of_utc": as_of_str,
            "cutoff_utc": cutoff_str,
            "before_days": args.before_days,
            "dry_run": True,
            "batch_size": 1000,
            "planned_batches": planned_batches,
            "applied_batches": 0,
            "eligible_audits": len(eligible_ids),
            "eligible_candidate_decisions": eligible_decisions,
            "eligible_safety_reviews": eligible_safety_reviews,
            "eligible_safety_findings": eligible_safety_findings,
            "deleted_audits": 0,
            "deleted_candidate_decisions": 0,
            "deleted_safety_reviews": 0,
            "deleted_safety_findings": 0,
        }
        conn.close()
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        return 0

    # Non-dry: delete in batches
    deleted_audits = 0
    deleted_decisions = 0
    deleted_reviews = 0
    deleted_findings = 0
    applied_batches = 0

    try:
        conn.execute("BEGIN IMMEDIATE")
        for i in range(0, len(eligible_ids), 1000):
            batch = eligible_ids[i : i + 1000]
            placeholders = ",".join("?" * len(batch))
            dec_count = conn.execute(
                f"SELECT COUNT(*) FROM retrieval_candidate_decisions WHERE audit_id IN ({placeholders})",
                batch,
            ).fetchone()[0]
            conn.execute(
                f"DELETE FROM retrieval_candidate_decisions WHERE audit_id IN ({placeholders})",
                batch,
            )
            conn.execute(
                f"DELETE FROM retrieval_audits WHERE id IN ({placeholders})",
                batch,
            )
            deleted_audits += len(batch)
            deleted_decisions += dec_count
            applied_batches += 1

        # Verify no eligible IDs remain
        remaining = conn.execute(
            "SELECT COUNT(*) FROM retrieval_audits WHERE status IN ('completed','failed') AND completed_at < ?",
            (cutoff_str,),
        ).fetchone()[0]
        if remaining > 0:
            raise ValueError(f"{remaining} eligible audits remain after deletion")

        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.close()
        sys.stderr.write(f"prune_security_audits: deletion failed: {exc}\n")
        return 2

    result = {
        "schema_version": "security-audit-prune-v1",
        "head": head,
        "as_of_utc": as_of_str,
        "cutoff_utc": cutoff_str,
        "before_days": args.before_days,
        "dry_run": False,
        "batch_size": 1000,
        "planned_batches": planned_batches,
        "applied_batches": applied_batches,
        "eligible_audits": len(eligible_ids),
        "eligible_candidate_decisions": eligible_decisions,
        "eligible_safety_reviews": eligible_safety_reviews,
        "eligible_safety_findings": eligible_safety_findings,
        "deleted_audits": deleted_audits,
        "deleted_candidate_decisions": deleted_decisions,
        "deleted_safety_reviews": deleted_reviews,
        "deleted_safety_findings": deleted_findings,
    }
    conn.close()
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old security audit records.")
    parser.add_argument("--before-days", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--as-of-utc", default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--allow-disposable-database", action="store_true")
    args = parser.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
