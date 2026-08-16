"""Fingerprint production SQL/Chroma state (B-14 gate support).

Reads configured SQL/Chroma and emits sorted-key minified JSON + LF containing:
Alembic head, per-application-table row count and sorted primary-key tuples,
and sorted Chroma IDs. Never includes text, excerpts, environment, metadata, or
credentials. Performs no write. Exits 2 on any inventory failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fingerprint_sql(database_url: str) -> dict:
    """Fingerprint the SQL database (read-only)."""
    import sqlite3
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(database_url)
    insp = inspect(engine)
    tables = insp.get_table_names()

    result: dict = {}
    with engine.connect() as conn:
        # Alembic head
        if "alembic_version" in tables:
            row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
            result["alembic_head"] = row[0] if row else None
        else:
            result["alembic_head"] = None

        table_fingerprints = {}
        for table in sorted(tables):
            if table == "alembic_version":
                continue
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
            pk_cols = insp.get_pk_constraint(table).get("constrained_columns", [])
            pk_tuples = []
            if pk_cols and count > 0:
                cols = ", ".join(pk_cols)
                rows = conn.execute(text(f"SELECT {cols} FROM {table} ORDER BY {cols}")).fetchall()
                pk_tuples = [list(r) for r in rows]
            table_fingerprints[table] = {
                "row_count": count,
                "primary_keys": pk_tuples,
            }
        result["tables"] = table_fingerprints
    engine.dispose()
    return result


def _fingerprint_chroma(host: str | None, port: int, collection_name: str = "rag-collection") -> list[str]:
    """Fingerprint Chroma collection IDs (read-only).

    A missing production collection is an empty inventory; any other
    inventory failure is an error and must abort rather than silently
    masquerade as an empty collection (plan §10B named-volume restoration).
    """
    if not host:
        return []
    import chromadb
    client = chromadb.HttpClient(host=host, port=port)
    try:
        col = client.get_collection(collection_name)
    except Exception as exc:
        if "not found" in str(exc).lower():
            return []
        raise
    result = col.get(include=[])
    return sorted(result.get("ids", []))


def fingerprint(json_output: bool = False) -> dict:
    from app.config import get_settings
    settings = get_settings()

    sql_fp = _fingerprint_sql(settings.database_url)
    chroma_ids = _fingerprint_chroma(
        settings.chroma_host, settings.chroma_port
    )

    fingerprint_data = {
        **sql_fp,
        "chroma_ids": chroma_ids,
    }
    return fingerprint_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        fp = fingerprint()
    except Exception as exc:
        sys.stderr.write(f"fingerprint_production_state: {exc}\n")
        return 2

    payload = json.dumps(fp, sort_keys=True, separators=(",", ":")) + "\n"
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
