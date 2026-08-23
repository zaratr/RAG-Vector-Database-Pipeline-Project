"""Disposable reconciliation validator for Task 10A.4 (plan L613/L625, F10).

Creates a collision-resistant disposable migrated SQLite database and a REAL
disposable Chroma collection (never the configured production stores, never an
in-memory vector-store substitute), refuses any production identity/symlink
equality, and exercises the production reconciliation service — the function
the operator CLI ``scripts/reconcile_ingestion.py`` wraps; the CLI itself binds
the configured production stores and is therefore never invoked here — against
those disposable stores:

* an empty-stores probe proving the crash-matrix row "before staged commit"
  (no document at all) reconciles as an exact no-op;
* a seeded closed crash-matrix drift set whose ready component is exactly
  three ready chunks (the self-created ``ready_chunks_upserted=3`` invariant):
  one staged document with a still-pending extraction, one staged document
  with terminal extraction evidence, one staged document whose chunk vector
  was already written into Chroma, one orphan vector with no SQL chunk, and
  the three ready fixtures with all expected vectors present;
* a first reconciliation whose counters are ASSERTED exactly per matrix row,
  followed by persisted-SQL proofs (never-promotes ``staged``->``ready``,
  terminal evidence preserved, pending rows terminalized with
  ``reconciled_incomplete`` and completed_at, extraction rows never re-run or
  re-created) and real Chroma reads (the collection converges to exactly the
  ready vector IDs);
* a second reconciliation proving idempotent no-op: all mutation/removal/
  transition counters zero, ready upsert 3 by design, SQL untouched.

The CONFIGURED production SQL and Chroma fingerprints are captured strictly
read-only (SQLite ``mode=ro`` connection; Chroma list/get only) before and
after the whole run and must be identical. Disposable DB/WAL/SHM files and the
disposable collection are removed in ``finally`` and their removal is itself
verified before success is reported. Any failed proof, unreadable configured
production store, fingerprint mismatch, or incomplete self-cleaning exits
non-zero; success output is emitted only when genuinely verified.

Exit codes (mirrors the P8a validate_phase10a.py conventions):

* ``0`` — success; stdout is exactly the plan-pinned converged second-run JSON.
* ``1`` — reconciliation counter/proof/cleanup assertion failure
  (machine-readable JSON on stderr).
* ``2`` — configuration/infrastructure failure, including production-identity
  refusal and unreadable configured production fingerprints.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.core.db import Base
from app.core.migrations import upgrade_database
from app.persistence import models
from app.services.embeddings import HashEmbeddingProvider
from app.services.reconciliation import reconcile_ingestion
from app.services.vector_store import ChromaVectorStore

# 10A.4: the drift matrix's ready component is exactly three self-created
# fixtures, so ready_chunks_upserted=3 is an invariant on EVERY run.
READY_FIXTURE_COUNT = 3

# Crash-matrix row "before staged commit": no documents, empty collection.
EXPECTED_NOOP = {
    "nonready_vectors_deleted": 0,
    "orphan_vectors_deleted": 0,
    "pending_extractions_failed": 0,
    "ready_chunks_upserted": 0,
    "staged_documents_failed": 0,
}
# Seeded matrix first-reconciliation counters: 3 staged documents failed
# (pending-extraction, terminal-evidence, partial-vector-write rows), 1 pending
# extraction terminalized, 1 nonready vector deleted (the partial write), 1
# orphan vector deleted, 3 ready chunks idempotently upserted.
EXPECTED_FIRST = {
    "nonready_vectors_deleted": 1,
    "orphan_vectors_deleted": 1,
    "pending_extractions_failed": 1,
    "ready_chunks_upserted": READY_FIXTURE_COUNT,
    "staged_documents_failed": 3,
}
# Plan L619-623: converged second run — all mutation counters zero, ready
# upsert 3 by design.
EXPECTED_SECOND = {
    "nonready_vectors_deleted": 0,
    "orphan_vectors_deleted": 0,
    "pending_extractions_failed": 0,
    "ready_chunks_upserted": READY_FIXTURE_COUNT,
    "staged_documents_failed": 0,
}

_PRODUCTION_COLLECTION = "rag-collection"

_EXIT2_CODES = {
    "configuration_error",
    "fingerprint_unreadable",
}


class ValidatorFailure(RuntimeError):
    """Machine-readable acceptance failure."""

    def __init__(self, code: str, lane: str, detail: dict | None = None):
        self.code = code
        self.lane = lane
        self.detail = {key: str(value)[:200] for key, value in (detail or {}).items()}
        if "check" not in self.detail:
            self.detail["check"] = code
        super().__init__(f"{code}:{lane}:{self.detail.get('check')}")


def _check(lane: str, check: str, condition: bool, **detail) -> None:
    if not condition:
        raise ValidatorFailure("assertion_failed", lane, {"check": check, **detail})


def _is_production_path(db_path: Path, database_url: str | None) -> bool:
    """Refuse to operate on a path that equals/symlinks the configured DB."""
    if not database_url or not database_url.startswith("sqlite:"):
        return False
    try:
        configured_path = Path(database_url.split("sqlite:///", 1)[1].split("?", 1)[0]).resolve()
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


def _disposable_store(run_id: str):
    """Disposable in-process (ephemeral) Chroma collection; never the
    configured production client, host, or collection.

    Uses the plain default ``EphemeralClient()`` like the rest of the
    codebase: chromadb keys its shared system by settings, so a custom
    Settings object here would break later default-client creation in the
    same process (observed as cross-module fixture errors). The random
    collection name can never collide with the production collection.
    """
    import chromadb

    client = chromadb.EphemeralClient()
    name = f"validate-reconciliation-{run_id}"
    return name, client, ChromaVectorStore(collection_name=name, client=client)


def _add_document(session, title: str, status: str, vector_id: str):
    document = models.Document(
        title=title, source="validate-reconciliation", ingestion_status=status
    )
    session.add(document)
    session.flush()
    text = f"{title} drift fixture text."
    chunk = models.Chunk(
        document_id=document.id,
        index=0,
        text=text,
        start_offset=0,
        end_offset=len(text),
        media_type="text/plain",
        vector_id=vector_id,
    )
    session.add(chunk)
    session.flush()
    return document, chunk


def _add_extraction(session, chunk_id: int, label: str, status: str, error_code: str | None = None):
    """Seed a lifecycle-CHECK-valid extraction row (10A.3 W4: terminal rows
    need completed_at; failed rows need error_code). Each row gets a distinct
    input identity so the partial identity-owner unique index is respected."""
    completed_at = (
        None if status == "pending" else datetime.now(timezone.utc).replace(tzinfo=None)
    )
    extraction = models.GraphExtraction(
        chunk_id=chunk_id,
        provider="ollama",
        model="gemma4:latest",
        prompt_version="graph-v1",
        schema_version="graph-relations-v1",
        status=status,
        input_sha256=hashlib.sha256(f"validate-recon:{label}".encode()).hexdigest(),
        attempt_count=1,
        is_identity_owner=1,
        completed_at=completed_at,
        error_code=error_code if status == "failed" else None,
    )
    session.add(extraction)
    session.flush()
    return extraction


def _extraction_snapshot(session) -> list[tuple]:
    """Comparable (id, status, error_code, attempt_count, completed_at) rows."""
    return [
        (
            row.id,
            row.status,
            row.error_code,
            row.attempt_count,
            row.completed_at.isoformat() if row.completed_at is not None else None,
        )
        for row in session.query(models.GraphExtraction)
        .order_by(models.GraphExtraction.id)
        .all()
    ]


# ---------------------------------------------------------------------------
# Read-only fingerprints of the CONFIGURED production stores
# ---------------------------------------------------------------------------


def _sql_fingerprint(database_url: str | None) -> dict | None:
    """Read-only [row count, max id] fingerprint of a configured SQLite DB.

    Returns None when the URL is not SQLite or no database file exists
    (nothing configured to protect). Raises ValidatorFailure when a database
    exists but cannot be read — restoration cannot be verified otherwise.
    """
    if not database_url or not database_url.startswith("sqlite:"):
        return None
    location = database_url.split("sqlite:///", 1)[1].split("?", 1)[0]
    if not location or location == ":memory:":
        return None
    path = Path(location)
    if not path.exists():
        return None
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                    " AND name != 'alembic_version' ORDER BY name"
                )
            ]
            fingerprint = {}
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                max_id = None
                if any(column[1] == "id" for column in conn.execute(f"PRAGMA table_info({table})")):
                    max_id = conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()[0]
                fingerprint[table] = [count, max_id]
            return fingerprint
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ValidatorFailure(
            "fingerprint_unreadable", "restoration", {"detail": str(exc)[:200]}
        ) from exc


def _chroma_fingerprint(host: str | None, port: int, persist_directory: str | None) -> dict | None:
    """Read-only fingerprint of the CONFIGURED remote Chroma server.

    Returns None when no remote host is configured. A configured
    ``persist_directory`` is deliberately not fingerprinted: opening the
    embedded database directly can create sidecar files in the production
    store. Raises ValidatorFailure when a configured host cannot be read.
    """
    if not host:
        return None
    try:
        import chromadb

        client = chromadb.HttpClient(host=host, port=port)
        collections = sorted(
            getattr(collection, "name", collection)
            for collection in client.list_collections()
        )
        fingerprint: dict = {"collections": collections}
        if _PRODUCTION_COLLECTION in collections:
            collection = client.get_collection(_PRODUCTION_COLLECTION)
            ids = collection.get(include=[]).get("ids", [])
            digest = hashlib.sha256(",".join(sorted(ids)).encode("utf-8")).hexdigest()
            fingerprint[_PRODUCTION_COLLECTION] = {"count": len(ids), "ids_sha256": digest}
        return fingerprint
    except Exception as exc:
        raise ValidatorFailure(
            "fingerprint_unreadable", "restoration",
            {"detail": f"{type(exc).__name__}: {exc}"[:200]},
        ) from exc


def _production_fingerprints() -> dict:
    settings = get_settings()
    return {
        "sql": _sql_fingerprint(settings.database_url),
        "chroma": _chroma_fingerprint(
            settings.chroma_host, settings.chroma_port, settings.chroma_persist_directory
        ),
    }


async def _run(db_path: Path | None = None, run_id: str | None = None) -> dict:
    run_id = run_id or secrets.token_hex(16)
    db_path = db_path or (
        Path(tempfile.gettempdir()) / f"validate-reconciliation-{run_id}.db"
    )
    collection_name = f"validate-reconciliation-{run_id}"

    settings = get_settings()
    if _is_production_path(db_path, settings.database_url):
        raise ValidatorFailure(
            "configuration_error", "disposable",
            {"detail": f"refusing production path: {db_path}"},
        )
    if collection_name == _PRODUCTION_COLLECTION:
        raise ValidatorFailure(
            "configuration_error", "disposable",
            {"detail": f"refusing production collection: {collection_name}"},
        )

    production_before = _production_fingerprints()
    db_url = f"sqlite:///{db_path}"
    engine = None
    client = None
    sessions: list = []
    cleaned_db = False
    cleaned_collection = False
    try:
        upgrade_database(db_url)
        engine = create_engine(db_url)
        Base.metadata.create_all(engine)  # metadata/migrated-schema parity
        session_factory = sessionmaker(bind=engine)

        def _fresh():
            session = session_factory()
            sessions.append(session)
            return session

        _, client, store = _disposable_store(run_id)
        embedder = HashEmbeddingProvider()

        # --- Crash-matrix row "before staged commit": exact no-op. ---
        noop = await reconcile_ingestion(
            session=_fresh(), embedding_provider=embedder, vector_store=store
        )
        _check("matrix", "empty_stores_noop",
               noop == EXPECTED_NOOP, expected=EXPECTED_NOOP, actual=noop)

        # --- Seed the closed drift matrix. ---
        seed = _fresh()
        ready_doc_ids: list[int] = []
        ready_vector_ids: list[str] = []
        for index in range(READY_FIXTURE_COUNT):
            _, chunk = _add_document(
                seed, f"validate-recon:{run_id}:ready:{index}", "ready",
                f"validate-recon:{run_id}:ready:{index}",
            )
            ready_doc_ids.append(chunk.document_id)
            ready_vector_ids.append(chunk.vector_id)
        # Row: staged commit before external work -> extraction still pending.
        staged_pending_doc, staged_pending_chunk = _add_document(
            seed, f"validate-recon:{run_id}:staged-pending", "staged",
            f"validate-recon:{run_id}:staged-pending",
        )
        pending_extraction = _add_extraction(
            seed, staged_pending_chunk.id, f"{run_id}:pending", "pending"
        )
        # Row: extraction terminal before vector write -> evidence to preserve.
        staged_terminal_doc, staged_terminal_chunk = _add_document(
            seed, f"validate-recon:{run_id}:staged-terminal", "staged",
            f"validate-recon:{run_id}:staged-terminal",
        )
        _add_extraction(
            seed, staged_terminal_chunk.id, f"{run_id}:terminal-ok", "succeeded"
        )
        _add_extraction(
            seed, staged_terminal_chunk.id, f"{run_id}:terminal-failed", "failed",
            error_code="original_failure",
        )
        # Row: partial vector write before ready commit -> vector present.
        staged_partial_doc, staged_partial_chunk = _add_document(
            seed, f"validate-recon:{run_id}:staged-partial", "staged",
            f"validate-recon:{run_id}:staged-partial",
        )
        _add_extraction(
            seed, staged_partial_chunk.id, f"{run_id}:partial", "succeeded"
        )
        seed.commit()
        # Capture plain values before closing: commit expires the ORM
        # instances, so detached attribute access would raise.
        staged_ids = [
            (staged_pending_doc.id, "staged_pending"),
            (staged_terminal_doc.id, "staged_terminal"),
            (staged_partial_doc.id, "staged_partial"),
        ]
        staged_partial_vector_id = staged_partial_chunk.vector_id
        pending_extraction_id = pending_extraction.id
        pre_extractions = _extraction_snapshot(seed)
        seed.close()

        # Drift in the disposable Chroma collection: the three ready IDs
        # (matrix ready row "all expected IDs"), the partial-write chunk's ID
        # ("subset/all IDs"), and one orphan with no SQL chunk at all.
        orphan_vector_id = f"validate-recon:{run_id}:orphan"
        seed_ids = [*ready_vector_ids, staged_partial_vector_id, orphan_vector_id]
        seed_texts = [f"validate-recon seed {index}" for index in range(len(seed_ids))]
        seed_embeddings = await embedder.embed_texts(seed_texts)
        await store.upsert_embeddings(
            seed_embeddings,
            [{"source": "validate-reconciliation"} for _ in seed_ids],
            seed_ids,
            documents=seed_texts,
        )
        seeded = set(await store.list_ids())
        _check("matrix", "seeded_collection", seeded == set(seed_ids),
               actual=sorted(seeded), expected=sorted(seed_ids))

        # --- First reconciliation: exact counters per matrix row. ---
        first = await reconcile_ingestion(
            session=_fresh(), embedding_provider=embedder, vector_store=store
        )
        _check("matrix", "first_run_counters",
               first == EXPECTED_FIRST, expected=EXPECTED_FIRST, actual=first)

        # --- Persisted-SQL proofs (never merely printed counters). ---
        proof = _fresh()
        for doc_id, label in staged_ids:
            document = proof.get(models.Document, doc_id)
            _check("matrix", "never_promotes", document is not None
                   and document.ingestion_status == "failed"
                   and document.failure_code == "reconciled_incomplete",
                   document=label,
                   actual=getattr(document, "ingestion_status", None))
        for doc_id in ready_doc_ids:
            document = proof.get(models.Document, doc_id)
            _check("matrix", "ready_preserved", document is not None
                   and document.ingestion_status == "ready",
                   document=doc_id,
                   actual=getattr(document, "ingestion_status", None))
        terminalized = proof.get(models.GraphExtraction, pending_extraction_id)
        _check("matrix", "pending_terminalized", terminalized is not None
               and terminalized.status == "failed"
               and terminalized.error_code == "reconciled_incomplete"
               and terminalized.completed_at is not None
               and terminalized.attempt_count == 1)
        post_extractions = _extraction_snapshot(proof)
        pre_terminal = [r for r in pre_extractions if r[0] != pending_extraction_id]
        post_terminal = [r for r in post_extractions if r[0] != pending_extraction_id]
        _check("matrix", "extraction_row_count_unchanged",
               len(post_extractions) == len(pre_extractions),
               expected=len(pre_extractions), actual=len(post_extractions))
        # never re-extracts: terminal evidence rows byte-identical, no new rows
        _check("matrix", "never_re_extracts", post_terminal == pre_terminal,
               changed=[row for row in post_terminal if row not in pre_terminal])
        after_first_rows = post_extractions
        proof.close()

        # --- Real Chroma convergence: exactly the ready IDs remain. ---
        collection_ids = set(await store.list_ids())
        _check("matrix", "converged_collection",
               collection_ids == set(ready_vector_ids),
               actual=sorted(collection_ids), expected=sorted(ready_vector_ids))

        # --- Second reconciliation: idempotent no-op. ---
        second = await reconcile_ingestion(
            session=_fresh(), embedding_provider=embedder, vector_store=store
        )
        _check("matrix", "second_run_noop",
               second == EXPECTED_SECOND, expected=EXPECTED_SECOND, actual=second)
        collection_ids = set(await store.list_ids())
        _check("matrix", "second_run_collection",
               collection_ids == set(ready_vector_ids),
               actual=sorted(collection_ids))
        proof2 = _fresh()
        _check("matrix", "second_run_sql_untouched",
               _extraction_snapshot(proof2) == after_first_rows)
        for doc_id, label in staged_ids:
            document = proof2.get(models.Document, doc_id)
            _check("matrix", "second_run_still_failed", document is not None
                   and document.ingestion_status == "failed",
                   document=label,
                   actual=getattr(document, "ingestion_status", None))
        proof2.close()

        result = {"first": first, "second": second, "noop": noop, "restored": False}
    finally:
        for session in sessions:
            try:
                session.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        if engine is not None:
            engine.dispose()
        if client is not None:
            try:
                client.delete_collection(collection_name)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        cleaned_db = not any(
            Path(str(db_path) + suffix).exists() for suffix in ("", "-wal", "-shm")
        )
        if client is not None:
            try:
                cleaned_collection = collection_name not in {
                    getattr(c, "name", c) for c in client.list_collections()
                }
            except Exception:  # pragma: no cover - cleanup must be verifiable
                cleaned_collection = False

    production_after = _production_fingerprints()
    _check("restoration", "production_fingerprints_unchanged",
           production_before == production_after,
           changed=[key for key in production_before
                    if production_before[key] != production_after.get(key)])
    _check("cleanup", "disposable_db_removed", cleaned_db, path=str(db_path))
    _check("cleanup", "disposable_collection_removed", cleaned_collection,
           collection=collection_name)
    result["restored"] = True
    return result


def main() -> int:
    # NOTE: no os.environ mutation here. chromadb derives client settings
    # from the environment, so mutating it would make later EphemeralClient
    # creations in the same process fail with "different settings".
    try:
        result = asyncio.run(_run())
    except ValidatorFailure as exc:
        sys.stderr.write(
            json.dumps({"error": exc.code, "lane": exc.lane, **exc.detail}, sort_keys=True)
            + "\n"
        )
        return 2 if exc.code in _EXIT2_CODES else 1
    except Exception as exc:  # never mask an unexpected failure with exit 0
        sys.stderr.write(
            json.dumps(
                {
                    "error": "unexpected_failure",
                    "lane": "validator",
                    "detail": f"{type(exc).__name__}: {exc}"[:200],
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    sys.stdout.write(json.dumps(result["second"], sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
