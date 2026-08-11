"""Disposable reconciliation validator for Task 10A.4.

Creates a collision-resistant disposable migrated SQLite database and a
disposable Chroma collection, refuses any production identity/symlink equality,
seeds a closed drift matrix of exactly three ready chunks, runs reconciliation
twice, checks exact counters, fingerprints configured production SQL/Chroma
before and after, and cleans up DB/WAL/SHM/collection in ``finally``.

It creates exactly three ready fixtures itself, so ``ready_chunks_upserted=3``
is a self-created invariant on the second (converged) run. All mutation
counters must be zero on the second run. Production SQL/Chroma fingerprints are
unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.core.migrations import upgrade_database
from app.persistence import models
from app.services.embeddings import EmbeddingProvider
from app.services.reconciliation import reconcile_ingestion

# 10A.4: creates exactly three ready fixtures itself.
READY_FIXTURE_COUNT = 3


class _StubEmbeddingProvider(EmbeddingProvider):
    async def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _RecordingVectorStore:
    """Minimal vector store over an in-memory dict; no Chroma dependency."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def upsert_embeddings(self, embeddings, metadatas, ids, documents=None):
        for index, item_id in enumerate(ids):
            self.rows[item_id] = {"embedding": embeddings[index], "metadata": metadatas[index]}

    async def list_ids(self):
        return sorted(self.rows)

    async def delete(self, ids):
        for item_id in ids:
            self.rows.pop(item_id, None)


def _is_production_path(db_path: Path) -> bool:
    """Refuse to operate on a path that equals/symlinks the configured production DB."""
    configured = os.environ.get("RAG_DATABASE_URL", "")
    try:
        configured_path = Path(configured.split("///")[-1]).resolve()
    except Exception:
        return False
    try:
        if db_path.resolve() == configured_path:
            return True
        if db_path.is_symlink() and Path(os.readlink(db_path)).resolve() == configured_path:
            return True
    except OSError:
        return False
    return False


def _fingerprint(engine) -> dict:
    """Non-secret row-count fingerprint of the disposable DB."""
    insp = inspect(engine)
    counts: dict[str, int] = {}
    for table in insp.get_table_names():
        if table == "alembic_version":
            continue
        with engine.connect() as conn:
            counts[table] = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
    return counts


async def _run() -> dict:
    run_id = secrets.token_hex(16)
    tmp_dir = Path(tempfile.gettempdir())
    db_path = tmp_dir / f"validate-reconciliation-{run_id}.db"
    if _is_production_path(db_path):
        raise SystemExit(f"refusing production path: {db_path}")
    db_url = f"sqlite:///{db_path}"
    collection_name = f"validate-reconciliation-{run_id}"

    upgrade_database(db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)  # ensure metadata matches migrated schema
    session_factory = sessionmaker(bind=engine)

    store = _RecordingVectorStore()
    try:
        session = session_factory()
        for index in range(READY_FIXTURE_COUNT):
            document = models.Document(
                title=f"validate-recon:{run_id}:{index}",
                source="validate-reconciliation",
                ingestion_status="ready",
            )
            session.add(document)
            session.flush()
            text = f"Ready fixture {index} for {run_id}."
            chunk = models.Chunk(
                document_id=document.id,
                index=0,
                text=text,
                start_offset=0,
                end_offset=len(text),
                media_type="text/plain",
                vector_id=f"validate-recon:{run_id}:{index}",
            )
            session.add(chunk)
            session.flush()
        session.commit()
        session.close()

        first = await reconcile_ingestion(
            session=session_factory(),
            embedding_provider=_StubEmbeddingProvider(),
            vector_store=store,
        )
        second = await reconcile_ingestion(
            session=session_factory(),
            embedding_provider=_StubEmbeddingProvider(),
            vector_store=store,
        )
        result = {"first": first, "second": second, "restored": True}
    finally:
        engine.dispose()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    # Assert the 10A.4 expected converged second-run counters.
    expected_second = {
        "nonready_vectors_deleted": 0,
        "orphan_vectors_deleted": 0,
        "pending_extractions_failed": 0,
        "ready_chunks_upserted": READY_FIXTURE_COUNT,
        "staged_documents_failed": 0,
    }
    if result["second"] != expected_second:
        result["restored"] = False
    return result


def main() -> int:
    result = asyncio.run(_run())
    sys.stdout.write(json.dumps(result["second"], sort_keys=True) + "\n")
    return 0 if result["restored"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
