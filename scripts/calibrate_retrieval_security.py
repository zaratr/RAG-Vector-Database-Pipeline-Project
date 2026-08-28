"""Retrieval security calibration tool (Task 10B.3).

Embeds/indexes the calibration corpus through the production FastEmbed/Chroma
adapters in a disposable collection backed by a disposable SQLite database,
runs queries, computes clean recall and poisoned-context share at each
candidate threshold, selects the largest threshold meeting both bounds, and
writes byte-deterministic policy JSON.

Disposable lifecycle: the process receives ``RAG_DATABASE_URL=
sqlite:////tmp/calibration-<32 hex>.db`` and a matching ``--run-id`` /
``--collection-name calibration-<32 hex>``. Production SQL is opened only
through ``file:...?mode=ro`` for fingerprinting; the production Chroma
collection (exactly ``rag-collection``) is fingerprinted from its sorted IDs.
Both fingerprints must be unchanged after cleanup, which runs in ``finally``.

CLI surface: --fixtures, --schema, --validate-only, --run-id, --collection-name,
--production-collection-name, --production-database-url, --stdout.

Exit codes: 0=pass, 1=no threshold found, 2=config/isolation/failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.retrieval_security import (  # noqa: E402
    CalibrationMetrics,
    compute_calibration_metrics as _per_selection_metrics,
    select_max_distance_threshold as _select_from_evaluations,
    validate_calibration_corpus,
)

# Backward-compatible alias for earlier callers.
validate_corpus = validate_calibration_corpus

EXPECTED_MIGRATION_HEAD = "c9f5b3e7a1d8"
PRODUCTION_COLLECTION_NAME = "rag-collection"
DISPOSABLE_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DISPOSABLE_DB_URL_RE = re.compile(r"^sqlite:////tmp/calibration-(?P<run_id>[0-9a-f]{32})\.db$")
SANITIZED_STDERR_LIMIT = 500


def _fail(message: str) -> None:
    """Exit 2 with a bounded, sanitized diagnostic (never payload bytes)."""
    sys.stderr.write(f"calibrate: {message}\n")
    raise SystemExit(2)


def _sqlite_path_from_url(url: str, *, what: str) -> Path:
    if not url.startswith("sqlite:///"):
        _fail(f"{what} must be a sqlite:/// URL, got a different scheme")
    raw = url[len("sqlite:///"):]
    if not raw.startswith("/"):
        _fail(f"{what} must be an absolute sqlite:////path URL")
    return Path(raw)


def fingerprint_sqlite(path: Path) -> str:
    """Non-sensitive fingerprint: alembic revision + per-table row counts and
    sorted primary-key tuples, hashed as canonical sorted JSON. Read-only."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        revision_row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        revision = revision_row[0] if revision_row else None
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
            )
        ]
        snapshot = {"alembic_revision": revision, "tables": []}
        for table in sorted(tables):
            pk_cols = [
                row[1]
                for row in sorted(conn.execute(f"PRAGMA table_info({table})"), key=lambda r: r[5])
                if row[5]
            ]
            if pk_cols:
                cols = ", ".join(pk_cols)
                pk_tuples = sorted(
                    tuple(row) for row in conn.execute(f"SELECT {cols} FROM {table}")
                )
            else:
                pk_tuples = []
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            snapshot["tables"].append(
                {
                    "name": table,
                    "row_count": count,
                    "primary_keys": [list(t) for t in pk_tuples],
                }
            )
        canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    finally:
        conn.close()


def fingerprint_chroma(collection_name: str) -> str:
    """Fingerprint a Chroma collection by its sorted vector IDs."""
    from app.services.vector_store import _create_client

    client = _create_client()
    try:
        collection = client.get_collection(collection_name)
        ids = sorted(collection.get(include=[]).get("ids", []))
    except Exception:
        ids = []
    canonical = json.dumps(
        {"collection": collection_name, "ids": ids}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_calibration_metrics(
    all_distances: dict[str, list[tuple[str, float]]],
    threshold: float,
    corpus: dict,
) -> CalibrationMetrics:
    """Micro-aggregated clean recall and poisoned-context share at ``threshold``.

    all_distances: query_id → list of (fixture_doc_id, l2_distance), already
    constrained to the corpus. Selection at a threshold is inclusive
    (``l2_distance <= threshold``) over the total pre-rank order
    ``(l2_distance, fixture_document_id)``.
    """
    label_by_id = {d["id"]: d["label"] for d in corpus["documents"]}
    total_required = 0
    total_required_selected = 0
    total_selected = 0
    total_poison_selected = 0
    off_query_poisons: set[str] = set()

    for q in corpus["queries"]:
        distances = all_distances.get(q["id"], [])
        ranked = sorted(distances, key=lambda x: (x[1], x[0]))
        selected_ids = [doc_id for doc_id, dist in ranked if dist <= threshold]
        metrics = _per_selection_metrics(
            selected_ids=selected_ids,
            required_clean_ids=q["required_clean_ids"],
            label_by_id=label_by_id,
            poisoned_ids=q["poisoned_ids"],
        )
        total_required += len(q["required_clean_ids"])
        total_required_selected += sum(1 for rid in q["required_clean_ids"] if rid in selected_ids)
        total_selected += len(selected_ids)
        total_poison_selected += sum(1 for sid in selected_ids if label_by_id.get(sid) == "poison")
        off_query_poisons.update(metrics.off_query_poison)

    clean_recall = total_required_selected / total_required if total_required > 0 else 0.0
    poison_share = total_poison_selected / total_selected if total_selected > 0 else 0.0
    return CalibrationMetrics(
        clean_recall=clean_recall,
        poisoned_context_share=poison_share,
        off_query_poison=sorted(off_query_poisons),
    )


def select_max_distance_threshold(
    all_distances: dict[str, list[tuple[str, float]]],
    corpus: dict,
) -> tuple[float, CalibrationMetrics] | None:
    """Select the largest threshold meeting recall ≥ 0.90 and poison ≤ 0.20.

    Candidate thresholds are the sorted distinct finite distances followed by
    ``math.nextafter(D[-1], inf)`` when that sentinel is finite and greater.
    Returns ``(threshold, metrics)`` or None when no threshold passes.
    """
    all_d = set()
    for dists in all_distances.values():
        for _, d in dists:
            if math.isfinite(d):
                all_d.add(d)
    if not all_d:
        return None
    D = sorted(all_d)
    sentinel = math.nextafter(D[-1], math.inf)
    if math.isfinite(sentinel) and sentinel > D[-1]:
        candidates = D + [sentinel]
    else:
        candidates = D

    evaluations: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = compute_calibration_metrics(all_distances, threshold, corpus)
        evaluations.append((threshold, metrics.clean_recall, metrics.poisoned_context_share))
    chosen = _select_from_evaluations(evaluations)
    if chosen is None:
        return None
    return chosen, compute_calibration_metrics(all_distances, chosen, corpus)


def run_validate_only(fixtures_path: Path, schema_path: Path | None) -> int:
    """Validate fixture and print counts without embedding."""
    corpus = json.loads(fixtures_path.read_text(encoding="utf-8"))
    schema = None
    if schema_path and schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_calibration_corpus(corpus, schema)
    output = {
        "documents": len(corpus["documents"]),
        "queries": len(corpus["queries"]),
        "schema_version": corpus["schema_version"],
        "status": "valid",
    }
    sys.stdout.write(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def _verify_chroma_space_l2(collection) -> None:
    """The Chroma collection configuration must resolve the distance space to l2."""
    config = getattr(collection, "configuration", None)
    text = str(config) if config is not None else ""
    match = re.search(r"space['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_]+)", text)
    if match:
        space = match.group(1).lower()
        if space != "l2":
            _fail(f"chroma distance space is {space!r}, expected l2")
        return
    _fail("chroma collection distance space could not be resolved to l2")


def _resolve_isolation(args, run_id: str) -> tuple[Path, Path, str, str]:
    """Validate identities/paths before any mutation.

    Returns (disposable_db_path, production_db_path, production_sql_fingerprint,
    production_chroma_fingerprint).
    """
    if not DISPOSABLE_RUN_ID_RE.match(run_id):
        _fail(f"run-id must be 32 lowercase hex characters: {run_id!r}")
    expected_collection = f"calibration-{run_id}"
    if args.collection_name != expected_collection:
        _fail(f"collection-name must equal {expected_collection!r}")
    if args.production_collection_name != PRODUCTION_COLLECTION_NAME:
        _fail(f"production-collection-name must be exactly {PRODUCTION_COLLECTION_NAME!r}")

    disposable_url = os.environ.get("RAG_DATABASE_URL", "")
    match = DISPOSABLE_DB_URL_RE.match(disposable_url)
    if not match:
        _fail("RAG_DATABASE_URL must be sqlite:////tmp/calibration-<run-id>.db")
    if match.group("run_id") != run_id:
        _fail("RAG_DATABASE_URL run id does not match --run-id")

    disposable_path = _sqlite_path_from_url(disposable_url, what="disposable database URL")
    for suffix in ("", "-wal", "-shm"):
        if disposable_path.parent.joinpath(disposable_path.name + suffix).exists():
            _fail(f"disposable database already exists: {disposable_path}{suffix}")

    production_path = _sqlite_path_from_url(
        args.production_database_url, what="production database URL"
    )
    if not production_path.is_file():
        _fail(f"production database missing or unreadable: {production_path}")
    if disposable_path == production_path:
        _fail("disposable database path must differ from the production path")
    if os.path.realpath(disposable_path) == os.path.realpath(production_path):
        _fail("disposable database path resolves to the production path")

    from app.services.vector_store import _create_client

    client = _create_client()
    existing = {c.name if hasattr(c, "name") else str(c) for c in client.list_collections()}
    if args.collection_name in existing:
        _fail(f"disposable collection already exists: {args.collection_name}")
    if args.collection_name == args.production_collection_name:
        _fail("disposable collection must differ from the production collection")

    production_sql_fp = fingerprint_sqlite(production_path)
    production_chroma_fp = fingerprint_chroma(args.production_collection_name)
    return disposable_path, production_path, production_sql_fp, production_chroma_fp


def _run_migrations_wrapper(disposable_url: str) -> None:
    """Run the supported migrations wrapper against the disposable database."""
    env = dict(os.environ)
    env["RAG_DATABASE_URL"] = disposable_url
    result = subprocess.run(
        [sys.executable, "-m", "app.core.migrations"],
        env=env,
        shell=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[-SANITIZED_STDERR_LIMIT:]
        _fail(f"migrations wrapper failed: {stderr!r}")


def _verify_disposable_head(disposable_path: Path) -> None:
    conn = sqlite3.connect(f"file:{disposable_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        head = row[0] if row else None
    finally:
        conn.close()
    if head != EXPECTED_MIGRATION_HEAD:
        _fail(f"disposable migration head is {head!r}, expected {EXPECTED_MIGRATION_HEAD!r}")


async def _insert_fixtures(disposable_url: str, corpus: dict, run_id: str) -> dict:
    """Insert exactly six documents and six chunks in one transaction.

    Returns the bijection ``fixture_id → {document_id, chunk_id, vector_id}``
    plus the reverse ``vector_id → fixture_id`` mapping.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.persistence import models

    engine = create_engine(disposable_url)
    Session = sessionmaker(bind=engine)
    bijection: dict[str, dict] = {}
    committed = False
    session = Session()
    try:
        for fixture in sorted(corpus["documents"], key=lambda d: d["id"]):
            doc = models.Document(
                title=f"calibration:{run_id}:{fixture['id']}",
                source=fixture["source"],
                tags="phase10-calibration",
                ingestion_status="ready",
                failure_code=None,
                trust_tier="standard",
                trust_score=0.5,
                trust_policy_version="source-trust-v1",
                ingestion_origin="calibration",
            )
            session.add(doc)
            session.flush()
            vector_id = f"calibration:{run_id}:{fixture['id']}"
            chunk = models.Chunk(
                document_id=doc.id,
                index=0,
                text=fixture["text"],
                start_offset=0,
                end_offset=len(fixture["text"]),
                vector_id=vector_id,
                media_type="text/plain",
            )
            session.add(chunk)
            session.flush()
            bijection[fixture["id"]] = {
                "document_id": doc.id,
                "chunk_id": chunk.id,
                "vector_id": vector_id,
            }
        session.commit()
        committed = True
    finally:
        session.close()
        engine.dispose()
    if not committed or len(bijection) != len(corpus["documents"]):
        _fail("fixture insert did not complete")
    return bijection


def _assert_fingerprint_sqlite(path: Path, expected: str) -> None:
    if fingerprint_sqlite(path) != expected:
        raise RuntimeError("production SQL fingerprint changed")


def _assert_fingerprint_chroma(collection_name: str, expected: str) -> None:
    if fingerprint_chroma(collection_name) != expected:
        raise RuntimeError("production Chroma fingerprint changed")


def _cleanup_fixtures(disposable_url: str, run_id: str) -> None:
    """Delete fixture rows by exact run prefix and verify zero matching rows."""
    from sqlalchemy import create_engine, text as sa_text

    engine = create_engine(disposable_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                sa_text("DELETE FROM chunks WHERE vector_id LIKE :prefix"),
                {"prefix": f"calibration:{run_id}:%"},
            )
            conn.execute(
                sa_text("DELETE FROM documents WHERE title LIKE :prefix"),
                {"prefix": f"calibration:{run_id}:%"},
            )
            remaining_docs = conn.execute(
                sa_text("SELECT COUNT(*) FROM documents WHERE title LIKE :prefix"),
                {"prefix": f"calibration:{run_id}:%"},
            ).scalar()
            remaining_chunks = conn.execute(
                sa_text("SELECT COUNT(*) FROM chunks WHERE vector_id LIKE :prefix"),
                {"prefix": f"calibration:{run_id}:%"},
            ).scalar()
        if remaining_docs or remaining_chunks:
            raise RuntimeError("fixture cleanup left rows behind")
    finally:
        engine.dispose()


def _unlink_disposable(disposable_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = disposable_path.parent.joinpath(disposable_path.name + suffix)
        try:
            if p.exists():
                p.unlink()
        except OSError as exc:
            # RuntimeError (not SystemExit) so the finally-step wrapper catches it.
            raise RuntimeError(f"failed to unlink disposable database {p}: {exc}") from exc


def run_full_calibration(
    fixtures_path: Path,
    schema_path: Path | None,
    run_id: str,
    collection_name: str,
    production_collection_name: str,
    production_database_url: str,
    stdout_output: bool,
) -> int:
    """Run full calibration with embedding, indexing, and threshold selection."""
    import asyncio

    corpus = json.loads(fixtures_path.read_text(encoding="utf-8"))
    schema = None
    if schema_path and schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_calibration_corpus(corpus, schema)

    fixture_sha = hashlib.sha256(fixtures_path.read_bytes()).hexdigest()

    class _Args:
        pass

    args = _Args()
    args.collection_name = collection_name
    args.production_collection_name = production_collection_name
    args.production_database_url = production_database_url

    disposable_path, production_path, prod_sql_fp, prod_chroma_fp = _resolve_isolation(args, run_id)
    disposable_url = os.environ["RAG_DATABASE_URL"]

    exit_code = 2
    store = None
    cleanup_error: str | None = None
    try:
        _run_migrations_wrapper(disposable_url)
        _verify_disposable_head(disposable_path)
        bijection = asyncio.run(_insert_fixtures(disposable_url, corpus, run_id))

        from app.services.embeddings import get_embedding_provider
        from app.services.vector_store import ChromaVectorStore, _create_client

        embedding_provider = get_embedding_provider()
        doc_texts = [d["text"] for d in corpus["documents"]]
        query_texts = [q["text"] for q in corpus["queries"]]
        doc_embeddings = asyncio.run(embedding_provider.embed_texts(doc_texts))
        query_embeddings = asyncio.run(embedding_provider.embed_texts(query_texts))

        client = _create_client()
        client.get_or_create_collection(
            name=collection_name, embedding_function=None, metadata={"hnsw:space": "l2"}
        )
        disposable_collection = client.get_collection(collection_name)
        _verify_chroma_space_l2(disposable_collection)
        store = ChromaVectorStore(collection_name=collection_name, client=client)
        _verify_chroma_space_l2(store.collection)

        vector_ids = [bijection[d["id"]]["vector_id"] for d in corpus["documents"]]
        asyncio.run(
            store.upsert_embeddings(
                embeddings=[list(e) for e in doc_embeddings],
                metadatas=[{"fixture_id": d["id"]} for d in corpus["documents"]],
                ids=vector_ids,
                documents=doc_texts,
            )
        )

        vector_to_fixture = {v["vector_id"]: fid for fid, v in bijection.items()}
        all_distances: dict[str, list[tuple[str, float]]] = {}
        for qi, query in enumerate(corpus["queries"]):
            results = asyncio.run(
                store.query(embedding=list(query_embeddings[qi]), top_k=corpus["top_k"])
            )
            seen_vector_ids: set[str] = set()
            dists: list[tuple[str, float]] = []
            for result in results:
                vector_id = getattr(result, "vector_id", None)
                if vector_id is None:
                    _fail("chroma candidate without an exact vector ID")
                if vector_id in seen_vector_ids:
                    _fail(f"duplicate/aliased vector ID returned: {vector_id}")
                seen_vector_ids.add(vector_id)
                fixture_id = vector_to_fixture.get(vector_id)
                if fixture_id is None or not vector_id.startswith(f"calibration:{run_id}:"):
                    _fail(f"chroma candidate does not map to this run: {vector_id}")
                distance = getattr(result, "score", None)
                if distance is None or not math.isfinite(float(distance)):
                    _fail("non-finite l2 distance returned")
                dists.append((fixture_id, float(distance)))
            fixture_ids = [fid for fid, _ in dists]
            expected_ids = sorted(d["id"] for d in corpus["documents"])
            if sorted(fixture_ids) != expected_ids:
                _fail(f"query {query['id']} did not return every fixture exactly once")
            if not any(pid in fixture_ids for pid in query["poisoned_ids"]):
                _fail(f"query {query['id']} retrieved no query-local poison before defenses")
            all_distances[query["id"]] = dists

        result = select_max_distance_threshold(all_distances, corpus)
        if result is None:
            sys.stderr.write("calibrate: no threshold meets both bounds\n")
            exit_code = 1
            return exit_code

        threshold, metrics = result
        policy = {
            "calibration": {
                "calibration_tool_version": "calibrate-v1",
                "clean_recall": round(metrics.clean_recall, 6),
                "embedding_model": corpus["embedding_model"],
                "fixture_sha256": fixture_sha,
                "poisoned_context_share": round(metrics.poisoned_context_share, 6),
            },
            "max_candidates": 50,
            "max_distance": round(threshold, 6),
            "metric": "l2",
            "near_duplicate_jaccard": 0.90,
            "per_document_cap": 2,
            "per_source_cap": 2,
            "version": "retrieval-security-v1",
        }
        payload = (
            json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
            + b"\n"
        )

        if stdout_output:
            sys.stdout.buffer.write(payload)
        else:
            output_path = _REPO_ROOT / "config" / "retrieval-security-policy.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(payload)
            sys.stdout.write(
                json.dumps(
                    {
                        "fixture_sha256": fixture_sha,
                        "max_distance": policy["max_distance"],
                        "clean_recall": policy["calibration"]["clean_recall"],
                        "poisoned_context_share": policy["calibration"][
                            "poisoned_context_share"
                        ],
                        "output": str(output_path),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        exit_code = 0
        return exit_code
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 2
        raise
    except Exception as exc:  # bounded sanitized failure
        sys.stderr.write(f"calibrate: {type(exc).__name__}\n")
        exit_code = 2
        raise SystemExit(2)
    finally:
        # Each cleanup step is independent: a failure in one must not skip the
        # others (D-21). Cleanup failure overrides any prior result with exit 2.
        cleanup_error: str | None = None

        def _step(message: str, action) -> None:
            nonlocal cleanup_error
            try:
                action()
            except Exception:
                cleanup_error = cleanup_error or message

        def _delete_collection() -> None:
            if store is not None:
                store.delete_collection(collection_name)
            else:
                try:
                    from app.services.vector_store import _create_client

                    _create_client().delete_collection(collection_name)
                except Exception:
                    pass  # never created or already absent

        _step("disposable collection cleanup failure", _delete_collection)
        _step("fixture cleanup failure", lambda: _cleanup_fixtures(disposable_url, run_id))
        _step("disposable database unlink failure", lambda: _unlink_disposable(disposable_path))
        _step(
            "production SQL fingerprint check failed",
            lambda: _assert_fingerprint_sqlite(production_path, prod_sql_fp),
        )
        _step(
            "production Chroma fingerprint check failed",
            lambda: _assert_fingerprint_chroma(production_collection_name, prod_chroma_fp),
        )
        if cleanup_error is not None:
            sys.stderr.write(f"calibrate: {cleanup_error}\n")
            raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate retrieval security policy.")
    parser.add_argument(
        "--fixtures",
        default=str(_REPO_ROOT / "app/tests/fixtures/retrieval_calibration.json"),
    )
    parser.add_argument(
        "--schema",
        default=str(_REPO_ROOT / "app/tests/fixtures/retrieval_calibration.schema.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--production-collection-name", default="rag-collection")
    parser.add_argument("--production-database-url", default="sqlite:////data/rag.db")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    fixtures_path = Path(args.fixtures)
    schema_path = Path(args.schema) if args.schema else None

    if not fixtures_path.is_file():
        sys.stderr.write(f"calibrate: fixture not found: {fixtures_path}\n")
        raise SystemExit(2)

    if args.validate_only:
        code = run_validate_only(fixtures_path, schema_path)
        if code != 0:
            raise SystemExit(code)
        return 0

    run_id = args.run_id or secrets.token_hex(16)
    collection_name = args.collection_name or f"calibration-{run_id}"

    code = run_full_calibration(
        fixtures_path=fixtures_path,
        schema_path=schema_path,
        run_id=run_id,
        collection_name=collection_name,
        production_collection_name=args.production_collection_name,
        production_database_url=args.production_database_url,
        stdout_output=args.stdout,
    )
    if code != 0:
        raise SystemExit(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
