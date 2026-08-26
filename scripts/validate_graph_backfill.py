"""Disposable graph backfill validator.

Hardened contract (the pre-remediation script asserted nothing, invoked the
service instead of the CLI, and took no fingerprints):

* creates one unique ready text document/chunk/vector (``vector_id``) without
  a graph extraction in a collision-resistant disposable migrated SQLite
  database, records its exact ``document_id``, and additionally seeds one
  unrelated ready sentinel plus skip-precedence probe documents;
* invokes the REAL operator CLI ``scripts/backfill_graph.py`` — never the
  service alone — through a deterministic local provider endpoint (an
  ephemeral loopback HTTP server speaking the Ollama/Gemma
  ``/chat/completions`` contract), pinned to the disposable database via
  ``RAG_DATABASE_URL`` with a hermetic cwd so no ``.env`` applies:
  - document-scoped ``--dry-run`` probes whose equations
    (``scanned = skipped + eligible``), zero mutation counters, zero provider
    calls, and zero writes are ASSERTED, including the skip precedence the
    seed supports (``document_not_ready`` over ``unsupported_media_type``;
    ``current_terminal`` over ``failed_not_retried``);
  - the first real run (exact counters, sorted minified JSON,
    exit 0) and the idempotent second run (``current_terminal`` skip, no new
    graph/evidence rows, still exactly one provider call);
* proves the two-worker duplicate-protection expectation
  deterministically (no sleeps): worker A's REAL ``backfill()`` provider call
  parks while a second worker reclaims the expired lease through the
  repository API with an injected clock 7200s past A's
  ``attempt_started_at``; A's complete is fenced with ``ExtractionLeaseLost``
  so exactly one worker reports ``processed=1/succeeded=1/lease_lost=0`` and
  the other ``lease_lost=1``, every conservation equation holds for both
  reports, and exactly one terminal succeeded row survives;
* asserts the sentinel is neither scanned nor leased (row fingerprint and
  extraction-row count unchanged), vector IDs and document readiness are
  never modified, and the CONFIGURED production SQL/Chroma fingerprints —
  captured strictly read-only (SQLite ``mode=ro``; Chroma list/get only) —
  are identical before and after;
* removes the disposable DB/WAL/SHM files and stops the provider endpoint in
  ``finally`` and VERIFIES the removal before reporting ``restored: true``.

Exit codes (mirrors the acceptance-validator conventions):

* ``0`` — success; stdout is exactly the pinned JSON
  ``{"first": {...}, "second": {...}, "restored": true}``.
* ``1`` — backfill counter/proof/cleanup assertion failure
  (machine-readable JSON on stderr).
* ``2`` — configuration/infrastructure failure, including production-identity
  refusal, unreadable configured production fingerprints, and CLI launch or
  configuration failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.core.db import Base
from app.core.migrations import upgrade_database
from app.persistence import models
from app.persistence.graph_repository import (
    begin_chunk_extraction,
    derive_extraction_identity,
)
from app.services.graph_backfill import backfill
from app.services.graph_extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ExtractedEntity,
    ExtractedRelation,
)

_PROVIDER = "ollama"
_MODEL = "gemma4:latest"

_PRODUCTION_COLLECTION = "rag-collection"

_EXIT2_CODES = {
    "configuration_error",
    "fingerprint_unreadable",
    "cli_infrastructure",
}

# Plan-pinned expected reports for the document-scoped CLI runs.
PINNED_FIRST = {
    "scanned": 1,
    "eligible": 1,
    "processed": 1,
    "succeeded": 1,
    "empty": 0,
    "failed": 0,
    "skipped": 0,
    "lease_lost": 0,
}
PINNED_SECOND = {
    "scanned": 1,
    "eligible": 0,
    "processed": 0,
    "succeeded": 0,
    "empty": 0,
    "failed": 0,
    "skipped": 1,
    "lease_lost": 0,
    "skip_reasons": {"current_terminal": 1},
}

# The deterministic grounded relation the provider endpoint returns: evidence
# and both entity surface forms are exact substrings of the fixture text
# "Subject describes Object ...".
_CANNED_RELATION = {
    "source": {"name": "Subject", "type": "concept"},
    "predicate": "describes",
    "target": {"name": "Object", "type": "concept"},
    "evidence": "Subject describes Object",
    "confidence": 0.9,
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


# ---------------------------------------------------------------------------
# Deterministic local provider endpoint (loopback, ephemeral port)
# ---------------------------------------------------------------------------


class _FakeProviderServer(ThreadingHTTPServer):
    """Counts requests; never touches any configured production provider."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, handler, canned_content: str):
        self.canned_content = canned_content
        self.requests = 0
        super().__init__(address, handler)

    def record_request(self) -> None:
        self.requests += 1


class _FakeProviderHandler(BaseHTTPRequestHandler):
    """Speaks the Ollama/Gemma OpenAI-compatible chat-completions contract."""

    def do_POST(self):  # noqa: N802 - http.server API
        server: _FakeProviderServer = self.server  # type: ignore[assignment]
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        server.record_request()
        body = json.dumps(
            {"choices": [{"message": {"content": server.canned_content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence request logging
        return


def _start_fake_provider(canned_content: str) -> _FakeProviderServer:
    server = _FakeProviderServer(("127.0.0.1", 0), _FakeProviderHandler, canned_content)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _canned_provider_content() -> str:
    return json.dumps({"relations": [_CANNED_RELATION]})


# ---------------------------------------------------------------------------
# Conservation equations shared by every report lane
# ---------------------------------------------------------------------------


def _check_conservation(lane: str, payload: dict) -> None:
    _check(
        lane, "eligible_equation",
        payload["eligible"] == payload["processed"] + payload["lease_lost"],
        **payload,
    )
    _check(
        lane, "processed_equation",
        payload["processed"] == payload["succeeded"] + payload["empty"] + payload["failed"],
        **payload,
    )
    _check(
        lane, "scanned_equation",
        payload["scanned"] == payload["skipped"] + payload["processed"] + payload["lease_lost"],
        **payload,
    )
    _check(
        lane, "skip_reasons_sum",
        payload["skipped"] == sum(payload.get("skip_reasons", {}).values()),
        **payload,
    )


def _check_dry_run_shape(lane: str, probe: str, payload: dict, expected: dict) -> None:
    _check(
        lane, f"{probe}_dry_run_counters", payload == expected,
        expected=expected, actual=payload,
    )
    _check(
        lane, f"{probe}_dry_run_equation",
        payload["scanned"] == payload["skipped"] + payload["eligible"],
        **payload,
    )
    for counter in ("processed", "succeeded", "empty", "failed", "lease_lost", "relations"):
        _check(lane, f"{probe}_dry_run_zero_{counter}", payload[counter] == 0, **payload)


# ---------------------------------------------------------------------------
# CLI invocation is backfill_graph.py for the seeded scenario
# ---------------------------------------------------------------------------


def _cli_environment(db_url: str, provider_base_url: str) -> dict:
    """Hermetic env for the CLI subprocess: the disposable database and the
    deterministic loopback provider are pinned; proxies cannot intercept
    loopback. The cwd used by the caller contains no ``.env``."""
    env = {
        **os.environ,
        "RAG_DATABASE_URL": db_url,
        "RAG_LLM_PROVIDER": "ollama",
        "RAG_LLM_BASE_URL": provider_base_url,
        "RAG_LLM_MODEL": _MODEL,
        # Pin the extraction identity: an empty override makes the CLI fall
        # back to llm_model, so seeded owner rows always share the identity.
        "RAG_GRAPH_EXTRACTION_MODEL": "",
        "RAG_GRAPH_EXTRACTION_ENABLED": "true",
    }
    no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
    parts = [part for part in no_proxy.split(",") if part] + ["127.0.0.1", "localhost"]
    env["NO_PROXY"] = ",".join(dict.fromkeys(parts))
    return env


def _run_cli(lane: str, db_url: str, provider_base_url: str, arguments: list[str]) -> dict:
    """Run scripts/backfill_graph.py and return its parsed sorted JSON report.

    Exit 2 from the CLI (invalid arguments/configuration/fatal) is an
    infrastructure failure; exit 1 (chunk failures) is surfaced through the
    counter assertions that follow.
    """
    argv = [sys.executable, str(_REPO_ROOT / "scripts" / "backfill_graph.py"), *arguments]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=tempfile.gettempdir(),
            env=_cli_environment(db_url, provider_base_url),
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidatorFailure(
            "cli_infrastructure", lane,
            {"detail": f"{type(exc).__name__}: {exc}"[:200]},
        ) from exc
    if result.returncode == 2:
        raise ValidatorFailure(
            "cli_infrastructure", lane,
            {"detail": (result.stderr or "")[:200]},
        )
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise ValidatorFailure(
            "cli_infrastructure", lane,
            {"detail": f"unparseable CLI stdout: {exc}"[:200]},
        ) from exc
    _check(
        lane, "sorted_minified_json",
        result.stdout == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    )
    return payload


def _disposable_engine(db_url: str):
    """File-backed disposable engine with REAL transactional semantics.

    pysqlite's default implicit-transaction handling does not work with
    SAVEPOINTs: work performed through ``begin_nested()`` (the lease begin /
    concurrent-begin containment path) is not undone by a Session ``rollback()``
    because the driver never tracked the transaction the SAVEPOINT started —
    the fenced worker's "rolled back" pending row then stays visible on the
    pooled connection. Disabling the driver's implicit handling and issuing
    explicit ``BEGIN`` (the pattern used by the backfill service tests) makes
    SAVEPOINT/ROLLBACK behave exactly as the concurrency proof requires.
    """
    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection, connection_record):  # pragma: no cover
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin(connection):  # pragma: no cover
        connection.execute(text("BEGIN"))

    return engine


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _add_document(session, title: str, status: str):
    document = models.Document(
        title=title, source="validate-graph-backfill", ingestion_status=status
    )
    session.add(document)
    session.flush()
    return document


def _add_chunk(session, document_id: int, index: int, text: str, *, media_type="text/plain"):
    chunk = models.Chunk(
        document_id=document_id,
        index=index,
        text=text,
        start_offset=0,
        end_offset=len(text),
        media_type=media_type,
        vector_id=f"validate-graph-backfill:{document_id}:{index}",
    )
    session.add(chunk)
    session.flush()
    return chunk


def _seed_extraction(
    session,
    *,
    chunk,
    input_text: str,
    status: str,
    error_code: str | None = None,
):
    """Seed an identity-owner extraction row for a derived input text.

    ``input_text`` differs from ``chunk.text`` only for stale-identity rows
    (a different ``input_sha256`` is a different identity, so the partial
    unique owner index still holds)."""
    identity = derive_extraction_identity(
        chunk=chunk,
        provider=_PROVIDER,
        model=_MODEL,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    input_sha = (
        identity.input_sha256
        if input_text == chunk.text
        else hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    extraction = models.GraphExtraction(
        chunk_id=chunk.id,
        provider=identity.provider,
        model=identity.model,
        prompt_version=identity.prompt_version,
        schema_version=identity.schema_version,
        input_sha256=input_sha,
        status=status,
        attempt_count=1,
        attempt_started_at=now,
        completed_at=now if status != "pending" else None,
        error_code=error_code if status == "failed" else None,
        error_detail="seeded fixture" if status == "failed" else None,
        is_identity_owner=True,
    )
    session.add(extraction)
    session.flush()
    return extraction


def _graph_counts(session) -> dict:
    return {
        "extractions": session.query(models.GraphExtraction).count(),
        "mentions": session.query(models.EntityMention).count(),
        "edges": session.query(models.GraphEdge).count(),
        "evidence": session.query(models.GraphEdgeEvidence).count(),
    }


def _sentinel_fingerprint(session, document_id: int):
    """Comparable (document row, chunk rows, extraction count) snapshot."""
    document = session.get(models.Document, document_id)
    chunks = (
        session.query(models.Chunk)
        .filter(models.Chunk.document_id == document_id)
        .order_by(models.Chunk.index)
        .all()
    )
    return (
        (
            None if document is None else (
                document.id, document.title, document.ingestion_status
            )
        ),
        [
            (chunk.index, chunk.text, chunk.vector_id, chunk.media_type)
            for chunk in chunks
        ],
        (
            session.query(models.GraphExtraction)
            .filter(
                models.GraphExtraction.chunk_id.in_([chunk.id for chunk in chunks])
            )
            .count()
        ),
    )


def _static_relation(text: str) -> ExtractedRelation:
    return ExtractedRelation(
        source=ExtractedEntity(
            name="Subject", canonical_name="subject", entity_type="concept"
        ),
        predicate="describes",
        target=ExtractedEntity(
            name="Object", canonical_name="object", entity_type="concept"
        ),
        evidence=text,
        evidence_start=0,
        evidence_end=len(text),
        confidence=0.9,
    )


class _StaticExtractor:
    """Deterministic extractor returning one grounded relation per call."""

    def __init__(self):
        self.calls = 0

    async def extract(self, text: str):
        self.calls += 1
        return [_static_relation(text)]


class _ReclaimDuringProviderCallExtractor:
    """Deterministic two-worker proof (no sleeps, injected clock).

    While worker A's REAL ``backfill()`` provider call is in flight, a second
    worker reclaims the lease through the repository API: ``begin`` with
    ``retry_failed=True`` and an injected clock 7200s past A's
    ``attempt_started_at`` (beyond any configured lease, max 3600s). The
    second worker thereby wins the same identity lease and bumps
    ``attempt_count``, so A's subsequent complete transition is fenced with
    ``ExtractionLeaseLost`` and counts ONLY in ``lease_lost``.
    """

    def __init__(self, session, chunk):
        self._session = session
        self._chunk = chunk
        self.calls = 0

    async def extract(self, text: str):
        self.calls += 1
        begin_chunk_extraction(
            self._session,
            chunk=self._chunk,
            provider=_PROVIDER,
            model=_MODEL,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            retry_failed=True,
            now_utc=lambda: datetime.now(timezone.utc) + timedelta(seconds=7200),
        )
        return [_static_relation(text)]


async def _run(db_path: Path | None = None, run_id: str | None = None) -> dict:
    run_id = run_id or secrets.token_hex(16)
    db_path = db_path or (
        Path(tempfile.gettempdir()) / f"validate-graph-backfill-{run_id}.db"
    )

    settings = get_settings()
    if _is_production_path(db_path, settings.database_url):
        raise ValidatorFailure(
            "configuration_error", "disposable",
            {"detail": f"refusing production path: {db_path}"},
        )

    production_before = _production_fingerprints()
    db_url = f"sqlite:///{db_path}"
    engine = None
    server = None
    sessions: list = []
    cleaned_db = False
    cleaned_server = False
    try:
        upgrade_database(db_url)
        engine = _disposable_engine(db_url)
        Base.metadata.create_all(engine)  # metadata/migrated-schema parity
        session_factory = sessionmaker(bind=engine)

        def _fresh():
            session = session_factory()
            sessions.append(session)
            return session

        # --- Seed the disposable scenario (single committing transaction). ---
        seed = _fresh()
        fixture_text = f"Subject describes Object for backfill validation {run_id}."
        fixture_doc = _add_document(seed, f"validate-graph-backfill:{run_id}:fixture", "ready")
        fixture_chunk = _add_chunk(seed, fixture_doc.id, 0, fixture_text)
        sentinel_doc = _add_document(seed, f"validate-graph-backfill:{run_id}:sentinel", "ready")
        _add_chunk(seed, sentinel_doc.id, 0, f"Sentinel text {run_id}.")
        # Probe document 1: one eligible text chunk + one unsupported image chunk.
        mixture_doc = _add_document(seed, f"validate-graph-backfill:{run_id}:mixture", "ready")
        _add_chunk(seed, mixture_doc.id, 0, f"Mixture text {run_id}.")
        _add_chunk(
            seed, mixture_doc.id, 1, f"Mixture image bytes {run_id}.",
            media_type="image/png",
        )
        # Probe document 2: non-ready document whose chunk is ALSO an image
        # -> document_not_ready must take precedence over unsupported_media_type.
        staged_doc = _add_document(seed, f"validate-graph-backfill:{run_id}:staged", "staged")
        _add_chunk(
            seed, staged_doc.id, 0, f"Staged image bytes {run_id}.",
            media_type="image/png",
        )
        # Probe document 3: chunk 0 has a failed owner (failed_not_retried);
        # chunk 1 has a stale-identity failed row plus the current-identity
        # succeeded owner -> current_terminal beats failed_not_retried.
        failed_doc = _add_document(seed, f"validate-graph-backfill:{run_id}:failed", "ready")
        failed_chunk = _add_chunk(seed, failed_doc.id, 0, f"Failed text {run_id}.")
        _seed_extraction(
            seed, chunk=failed_chunk, input_text=failed_chunk.text,
            status="failed", error_code="provider_error",
        )
        terminal_chunk = _add_chunk(seed, failed_doc.id, 1, f"Terminal text {run_id}.")
        _seed_extraction(
            seed, chunk=terminal_chunk, input_text=terminal_chunk.text,
            status="succeeded",
        )
        _seed_extraction(
            seed, chunk=terminal_chunk,
            input_text=f"stale predecessor text {run_id}",
            status="failed", error_code="provider_error",
        )
        # Concurrency document for the two-worker lane.
        conc_doc = _add_document(seed, f"validate-graph-backfill:{run_id}:concurrent", "ready")
        conc_chunk = _add_chunk(seed, conc_doc.id, 0, f"Subject describes Object concurrently {run_id}.")
        seed.commit()

        fixture_document_id = fixture_doc.id
        fixture_chunk_id = fixture_chunk.id
        sentinel_document_id = sentinel_doc.id
        mixture_document_id = mixture_doc.id
        staged_document_id = staged_doc.id
        failed_document_id = failed_doc.id
        conc_document_id = conc_doc.id
        conc_chunk_id = conc_chunk.id
        sentinel_before = _sentinel_fingerprint(seed, sentinel_document_id)
        seeded_counts = _graph_counts(seed)
        seed.close()

        # --- Deterministic provider endpoint (loopback, ephemeral port). ---
        server = _start_fake_provider(_canned_provider_content())
        provider_base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"

        # --- Dry-run probes (CLI): equations, zero writes, skip precedence. ---
        dry_specs = [
            ("fixture", fixture_document_id,
             {"scanned": 1, "eligible": 1, "processed": 0, "succeeded": 0,
              "empty": 0, "failed": 0, "skipped": 0, "lease_lost": 0,
              "relations": 0, "skip_reasons": {}}),
            ("mixture", mixture_document_id,
             {"scanned": 2, "eligible": 1, "processed": 0, "succeeded": 0,
              "empty": 0, "failed": 0, "skipped": 1, "lease_lost": 0,
              "relations": 0, "skip_reasons": {"unsupported_media_type": 1}}),
            ("staged_image", staged_document_id,
             {"scanned": 1, "eligible": 0, "processed": 0, "succeeded": 0,
              "empty": 0, "failed": 0, "skipped": 1, "lease_lost": 0,
              "relations": 0, "skip_reasons": {"document_not_ready": 1}}),
            ("failed_and_terminal", failed_document_id,
             {"scanned": 2, "eligible": 0, "processed": 0, "succeeded": 0,
              "empty": 0, "failed": 0, "skipped": 2, "lease_lost": 0,
              "relations": 0,
              "skip_reasons": {"current_terminal": 1, "failed_not_retried": 1}}),
        ]
        dry_runs = []
        for probe, document_id, expected in dry_specs:
            payload = _run_cli(
                "cli", db_url, provider_base_url,
                ["--document-id", str(document_id), "--dry-run", "--batch-size", "20"],
            )
            _check_dry_run_shape("cli", probe, payload, expected)
            dry_runs.append({"probe": probe, "payload": payload})
        _check("cli", "dry_run_provider_requests", server.requests == 0,
               actual=server.requests)
        verify = _fresh()
        _check("cli", "dry_run_zero_writes", _graph_counts(verify) == seeded_counts,
               expected=seeded_counts, actual=_graph_counts(verify))
        verify.close()

        # --- First real run (CLI, document-scoped, plan-pinned counters). ---
        first_payload = _run_cli(
            "cli", db_url, provider_base_url,
            ["--document-id", str(fixture_document_id), "--batch-size", "20"],
        )
        _check("cli", "first_run_counters",
               {key: first_payload.get(key) for key in PINNED_FIRST} == PINNED_FIRST
               and first_payload.get("relations") == 1
               and first_payload.get("skip_reasons") == {},
               expected={**PINNED_FIRST, "relations": 1, "skip_reasons": {}},
               actual=first_payload)
        _check_conservation("cli", first_payload)
        _check("cli", "first_run_provider_requests", server.requests == 1,
               actual=server.requests)

        proof = _fresh()
        fixture_rows = (
            proof.query(models.GraphExtraction)
            .filter(models.GraphExtraction.chunk_id == fixture_chunk_id)
            .all()
        )
        _check("cli", "first_run_one_extraction_row", len(fixture_rows) == 1,
               actual=len(fixture_rows))
        row = fixture_rows[0] if fixture_rows else None
        _check("cli", "first_run_extraction_succeeded",
               row is not None and row.status == "succeeded"
               and row.is_identity_owner and row.completed_at is not None
               and row.provider == _PROVIDER and row.model == _MODEL,
               actual=None if row is None else (row.status, row.is_identity_owner))
        after_first = _graph_counts(proof)
        _check("cli", "first_run_wrote_graph_rows",
               after_first["edges"] >= 1 and after_first["evidence"] >= 1,
               actual=after_first)
        fixture_document = proof.get(models.Document, fixture_document_id)
        fixture_chunk_row = proof.get(models.Chunk, fixture_chunk_id)
        _check("cli", "invariants_vector_id_and_readiness",
               fixture_chunk_row is not None
               and fixture_chunk_row.vector_id == f"validate-graph-backfill:{fixture_document_id}:0"
               and fixture_document is not None
               and fixture_document.ingestion_status == "ready",
               actual=(getattr(fixture_chunk_row, "vector_id", None),
                       getattr(fixture_document, "ingestion_status", None)))
        _check("cli", "sentinel_unchanged_after_first",
               _sentinel_fingerprint(proof, sentinel_document_id) == sentinel_before)
        proof.close()

        # --- Second real run (CLI): idempotent current_terminal no-op. ---
        second_payload = _run_cli(
            "cli", db_url, provider_base_url,
            ["--document-id", str(fixture_document_id), "--batch-size", "20"],
        )
        _check("cli", "second_run_counters",
               {key: second_payload.get(key) for key in PINNED_SECOND} == PINNED_SECOND,
               expected=PINNED_SECOND, actual=second_payload)
        _check_conservation("cli", second_payload)
        _check("cli", "second_run_no_provider_call", server.requests == 1,
               actual=server.requests)
        proof = _fresh()
        _check("cli", "graph_counts_not_increased",
               _graph_counts(proof) == after_first,
               expected=after_first, actual=_graph_counts(proof))
        _check("cli", "second_run_no_new_extraction_rows",
               _graph_counts(proof)["extractions"] == after_first["extractions"])
        _check("cli", "sentinel_unchanged_after_second",
               _sentinel_fingerprint(proof, sentinel_document_id) == sentinel_before)
        proof.close()

        # --- Two-worker duplicate protection (deterministic, injected clock).
        # Worker A: the REAL backfill() loop; the second worker reclaims the
        # expired lease mid-provider-call (see the extractor docstring).
        session_a = _fresh()
        chunk_a = session_a.get(models.Chunk, conc_chunk_id)
        parking = _ReclaimDuringProviderCallExtractor(session_a, chunk_a)
        worker_a = asdict(
            await backfill(
                session_a,
                extractor=parking,
                provider=_PROVIDER,
                model=_MODEL,
                document_id=conc_document_id,
                batch_size=20,
                retry_failed=False,
                dry_run=False,
            )
        )
        _check("concurrency", "worker_a_provider_call_started", parking.calls == 1,
               actual=parking.calls)
        _check("concurrency", "worker_a_lease_lost_report",
               {key: worker_a.get(key) for key in (
                   "scanned", "eligible", "processed", "succeeded", "empty",
                   "failed", "skipped", "lease_lost", "relations", "skip_reasons",
               )} == {
                   "scanned": 1, "eligible": 1, "processed": 0, "succeeded": 0,
                   "empty": 0, "failed": 0, "skipped": 0, "lease_lost": 1,
                   "relations": 0, "skip_reasons": {},
               },
               actual=worker_a)
        _check_conservation("concurrency", worker_a)
        session_a.rollback()
        session_a.close()

        # Worker B: the reclaiming worker re-runs the REAL backfill() loop on
        # the same eligible identity and, owning the lease, completes it.
        session_b = _fresh()
        worker_b = asdict(
            await backfill(
                session_b,
                extractor=_StaticExtractor(),
                provider=_PROVIDER,
                model=_MODEL,
                document_id=conc_document_id,
                batch_size=20,
                retry_failed=False,
                dry_run=False,
            )
        )
        _check("concurrency", "worker_b_processed_report",
               {key: worker_b.get(key) for key in (
                   "scanned", "eligible", "processed", "succeeded", "empty",
                   "failed", "skipped", "lease_lost", "relations", "skip_reasons",
               )} == {
                   "scanned": 1, "eligible": 1, "processed": 1, "succeeded": 1,
                   "empty": 0, "failed": 0, "skipped": 0, "lease_lost": 0,
                   "relations": 1, "skip_reasons": {},
               },
               actual=worker_b)
        _check_conservation("concurrency", worker_b)
        _check("concurrency", "two_worker_outcomes",
               sorted([(worker_a["processed"], worker_a["lease_lost"]),
                       (worker_b["processed"], worker_b["lease_lost"])])
               == [(0, 1), (1, 0)])
        terminal_rows = (
            session_b.query(models.GraphExtraction)
            .filter(models.GraphExtraction.chunk_id == conc_chunk_id)
            .all()
        )
        _check("concurrency", "exactly_one_terminal_row", len(terminal_rows) == 1,
               actual=len(terminal_rows))
        terminal = terminal_rows[0] if terminal_rows else None
        _check("concurrency", "terminal_row_succeeded",
               terminal is not None and terminal.status == "succeeded"
               and terminal.is_identity_owner,
               actual=None if terminal is None else terminal.status)
        concurrency = {
            "worker_a": worker_a,
            "worker_b": worker_b,
            "provider_calls": parking.calls,
            "terminal_rows": len(terminal_rows),
            "terminal_status": None if terminal is None else terminal.status,
        }
        session_b.close()

        result = {
            "first": dict(PINNED_FIRST),
            "second": dict(PINNED_SECOND),
            "dry_runs": dry_runs,
            "dry_run_provider_requests": 0,
            "concurrency": concurrency,
            "restored": False,
        }
    finally:
        for session in sessions:
            try:
                session.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        if engine is not None:
            engine.dispose()
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
                cleaned_server = True
            except Exception:  # pragma: no cover - cleanup must be verifiable
                cleaned_server = False
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

    production_after = _production_fingerprints()
    _check("restoration", "production_fingerprints_unchanged",
           production_before == production_after,
           changed=[key for key in production_before
                    if production_before[key] != production_after.get(key)])
    _check("cleanup", "disposable_db_removed", cleaned_db, path=str(db_path))
    _check("cleanup", "provider_endpoint_stopped", cleaned_server)
    result["restored"] = True
    return result


def main() -> int:
    # NOTE: no os.environ mutation here. chromadb derives client settings
    # from the environment, so mutating it would make later EphemeralClient
    # creations in the same process fail with "different settings". The CLI
    # subprocess environment is built as an explicit copy instead.
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
    sys.stdout.write(
        json.dumps(
            {
                "first": result["first"],
                "second": result["second"],
                "restored": result["restored"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
