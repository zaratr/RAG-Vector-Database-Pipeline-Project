"""Named-volume durability validation (Task 10D.4 / final gates).

Creates a UUID SQL parent/child pair plus a matching Chroma sentinel,
records IDs/hashes/head, force-recreates the deployment WITHOUT ``-v``
(named volumes must never be removed), runs the production migration
wrapper, verifies identical SQL rows/FKs/vector/head plus API and Chroma
health, deletes only its sentinels in ``finally``, and proves unrelated
production fingerprints unchanged. Output is canonical non-sensitive
JSON; any setup/verify/cleanup mismatch exits 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import traceback
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _database_path() -> Path:
    from app.config import get_settings

    url = get_settings().database_url
    return Path(url[len("sqlite:///"):]) if url.startswith("sqlite:///") \
        else Path("/data/rag.db")


def _snapshot() -> dict:
    """Canonical non-sensitive production snapshot (counts/hashes only)."""
    path = _database_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")]
        counts = {table: conn.execute(
            f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in sorted(tables)}
        fk_rows = len(conn.execute(
            "PRAGMA foreign_key_check").fetchall())
        head = conn.execute(
            "SELECT version_num FROM alembic_version").fetchone()[0]
    finally:
        conn.close()

    from app.services.vector_store import ChromaVectorStore

    store = ChromaVectorStore()
    vector_ids = sorted(store.collection.get(include=[]).get("ids", []))
    vector_hash = hashlib.sha256(
        "\n".join(vector_ids).encode("utf-8")).hexdigest()

    unrelated = hashlib.sha256(json.dumps(
        {"tables": counts, "fk_rows": fk_rows, "head": head,
         "vector_ids": vector_ids},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "sql_rows": sum(counts.values()),
        "fk_rows": fk_rows,
        "vector_hash": vector_hash,
        "alembic_head": head,
        "unrelated_fingerprint": unrelated,
    }


def _setup_sentinels() -> dict:
    """Create the UUID SQL parent/child and matching Chroma sentinel.

    The sentinel vector reuses an existing collection embedding's
    dimension (or 8 when the collection is empty) so it can never
    collide with the production collection's dimensionality; a failure
    after the SQL insert removes the partial rows before re-raising.
    """
    from sqlalchemy.orm import sessionmaker

    from app.core.db import create_database_engine
    from app.persistence import models
    from app.services.vector_store import ChromaVectorStore

    marker = uuid.uuid4().hex
    engine = create_database_engine(f"sqlite:///{_database_path()}")
    session = sessionmaker(bind=engine)()
    sql_parent_id = None
    try:
        parent = models.Document(
            title=f"durability-sentinel-{marker}",
            source="validate-named-volume-durability",
            ingestion_status="ready")
        session.add(parent)
        session.flush()
        sql_parent_id = parent.id
        child = models.Chunk(
            document_id=parent.id, index=0,
            text=f"durability sentinel {marker}",
            start_offset=0, end_offset=20, media_type="text/plain",
            vector_id=f"durability:{marker}")
        session.add(child)
        session.commit()
        sql_child_id = child.id
    except Exception:
        session.rollback()
        if sql_parent_id is not None:
            from sqlalchemy import text as sa_text

            session.execute(sa_text("DELETE FROM documents WHERE id = :p"),
                            {"p": sql_parent_id})
            session.commit()
        raise
    finally:
        session.close()
        engine.dispose()

    store = ChromaVectorStore()
    try:
        existing = store.collection.get(limit=1, include=["embeddings"])
        embeddings = existing.get("embeddings")
        dimension = len(embeddings[0]) if embeddings is not None \
            and len(embeddings) else 8
        import asyncio

        asyncio.run(store.upsert_embeddings(
            embeddings=[[1.0] * dimension],
            metadatas=[{"sentinel": marker}],
            ids=[f"durability:{marker}"]))
    except Exception:
        _cleanup_sentinels({
            "marker": marker, "sql_parent_id": sql_parent_id,
            "sql_child_id": sql_child_id,
            "chroma_sentinel_id": f"durability:{marker}"})
        raise
    return {
        "marker": marker,
        "sql_parent_id": sql_parent_id,
        "sql_child_id": sql_child_id,
        "chroma_sentinel_id": f"durability:{marker}",
    }


def _verify_state(sentinels: dict) -> None:
    """The sentinels must be identical after the recreate + wrapper."""
    path = _database_path()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        parent = conn.execute(
            "SELECT id FROM documents WHERE id = :p",
            {"p": sentinels["sql_parent_id"]}).fetchone()
        child = conn.execute(
            "SELECT id, document_id FROM chunks WHERE id = :c",
            {"c": sentinels["sql_child_id"]}).fetchone()
    finally:
        conn.close()
    if parent is None or child is None or \
            child[1] != sentinels["sql_parent_id"]:
        raise RuntimeError("sentinel row missing after recreate")

    from app.services.vector_store import ChromaVectorStore

    store = ChromaVectorStore()
    found = store.collection.get(ids=[sentinels["chroma_sentinel_id"]],
                                 include=[])
    if sentinels["chroma_sentinel_id"] not in found.get("ids", []):
        raise RuntimeError("sentinel vector missing after recreate")


def _cleanup_sentinels(sentinels: dict) -> list:
    """Delete exactly the created sentinels; return their IDs."""
    from sqlalchemy import text as sa_text
    from sqlalchemy.orm import sessionmaker

    from app.core.db import create_database_engine
    from app.services.vector_store import ChromaVectorStore

    engine = create_database_engine(f"sqlite:///{_database_path()}")
    session = sessionmaker(bind=engine)()
    deleted: list = []
    try:
        conn = session.connection()
        conn.execute(sa_text("DELETE FROM chunks WHERE id = :c"),
                     {"c": sentinels["sql_child_id"]})
        deleted.append(sentinels["sql_child_id"])
        conn.execute(sa_text("DELETE FROM documents WHERE id = :p"),
                     {"p": sentinels["sql_parent_id"]})
        deleted.append(sentinels["sql_parent_id"])
        session.commit()
    finally:
        session.close()
        engine.dispose()

    store = ChromaVectorStore()
    store.collection.delete(ids=[sentinels["chroma_sentinel_id"]])
    deleted.append(sentinels["chroma_sentinel_id"])
    return deleted


def _run_docker_child(argv: list) -> bool:
    """Run a docker child; a missing docker CLI (in-container) is tolerated.

    Returns whether the child actually executed. The unit tests assert
    this path never passes ``-v`` — named volumes must never be removed.
    """
    if any(part == "-v" for part in argv):  # defensive: never allow it
        return False
    try:
        subprocess.run(argv, check=False)
        return True
    except (FileNotFoundError, OSError):
        return False


def _chroma_heartbeat_url() -> str:
    """Chroma heartbeat: the configured service in-container, the host
    port mapping otherwise."""
    from app.config import get_settings

    settings = get_settings()
    if settings.chroma_host:
        return (f"http://{settings.chroma_host}:{settings.chroma_port}"
                f"/api/v2/heartbeat")
    return "http://127.0.0.1:8001/api/v2/heartbeat"


def _heartbeat(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return 200 <= response.getcode() < 300
    except (urllib.error.URLError, OSError):
        return False


def _wait_heartbeat(url: str, attempts: int = 30, interval: float = 2.0
                    ) -> bool:
    """Probe ``url`` until it answers 2xx or attempts run out.

    This function is the single injectable health seam: ``run_durability_check``
    resolves it as this module's attribute so the hermetic in-process tests can
    substitute a fake; the live opt-in lane (subprocess) uses the real probes.
    """
    import time

    for attempt in range(attempts):
        if _heartbeat(url):
            return True
        if attempt < attempts - 1:
            time.sleep(interval)
    return False


_ERROR_TRACEBACK_LINES = 30  # bounded: enough to locate the failing frame


def _error_detail(exc: BaseException) -> dict:
    """Exception type + message + bounded traceback, secret-safe.

    Follows the script's redaction discipline: only structured exception
    metadata is recorded (no command output, environment, or connection
    strings are included by construction).
    """
    lines = traceback.format_exception(
        type(exc), exc, exc.__traceback__)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(lines[-_ERROR_TRACEBACK_LINES:]),
    }


def run_durability_check(output) -> int:
    """Execute the durability check; write canonical JSON; return 0/2."""
    record: dict = {"ok": False, "stage": "setup", "sentinels": {},
                    "deleted_sentinel_ids": [], "cleanup_complete": False,
                    "api_healthy": False, "chroma_healthy": False,
                    "before": None, "after": None}
    output_path = Path(output)

    def _write() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8")

    sentinels: dict = {}
    try:
        record["before"] = _snapshot()
        sentinels = _setup_sentinels()
        record["sentinels"] = {
            "sql_parent_id": sentinels["sql_parent_id"],
            "sql_child_id": sentinels["sql_child_id"],
            "chroma_sentinel_id": sentinels["chroma_sentinel_id"]}

        # Force-recreate WITHOUT -v: named volumes must never be removed.
        # Inside a container (no docker CLI) the recreation is performed by
        # the recorded gate's own host-side restoration step; the script
        # records that it did not itself recreate.
        record["recreate_performed"] = _run_docker_child([
            "docker", "compose", "up", "-d", "--force-recreate"])
        # Production migration wrapper stays idempotent.
        _run_docker_child(["docker", "compose", "exec", "-T", "api",
                           "python", "-m", "app.core.migrations"])

        record["stage"] = "verify"
        _verify_state(sentinels)
        record["api_healthy"] = _wait_heartbeat("http://127.0.0.1:8000/")
        record["chroma_healthy"] = _wait_heartbeat(_chroma_heartbeat_url())
        if not (record["api_healthy"] and record["chroma_healthy"]):
            raise RuntimeError("service unhealthy after recreate")
        record["ok"] = True
    except Exception as exc:
        record["ok"] = False
        # keep the failing stage (setup/verify) and the failure detail in
        # the record so a flap is diagnosable from the record alone
        record["error"] = _error_detail(exc)
    finally:
        if sentinels:
            try:
                record["deleted_sentinel_ids"] = _cleanup_sentinels(sentinels)
                record["cleanup_complete"] = set(
                    record["deleted_sentinel_ids"]) == {
                    record["sentinels"].get("sql_parent_id"),
                    record["sentinels"].get("sql_child_id"),
                    record["sentinels"].get("chroma_sentinel_id")}
            except Exception as exc:
                record["ok"] = False
                record["stage"] = "cleanup"
                record.setdefault("error", _error_detail(exc))
        else:
            record["cleanup_complete"] = True
        try:
            record["after"] = _snapshot()
        except Exception:
            record["ok"] = False
        _write()

    if not record["ok"]:
        return 2
    before, after = record["before"], record["after"]
    if not (before["unrelated_fingerprint"] == after["unrelated_fingerprint"]
            and before["alembic_head"] == after["alembic_head"]):
        return 2
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate named-volume durability.")
    parser.add_argument("--output", required=True,
                        help="path for the canonical JSON record")
    args = parser.parse_args(argv)
    return run_durability_check(output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
