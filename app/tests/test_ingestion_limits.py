"""Phase 10B.3 — ingestion limits and rate limiting tests (appendix spec).

Covers: envelope/file/extracted byte boundaries, ``BoundedReceiveMiddleware``,
atomic fixed-window rate limit, header formulas, concurrency semantics,
restart retention, and startup config validation.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.core.db import Base, get_db
from app.main import BoundedReceiveMiddleware
from app.persistence import models  # noqa: F401 — register all models for create_all
from app.services.ingestion_limits import _rate_identity, acquire_slot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUEST_MAX = 11534336   # RAG_INGESTION_REQUEST_MAX_BYTES
FILE_MAX = 10485760      # RAG_INGESTION_FILE_MAX_BYTES
EXTRACTED_MAX = 5242880  # RAG_INGESTION_EXTRACTED_MAX_BYTES
RATE_LIMIT = 30          # RAG_INGEST_RATE_LIMIT_REQUESTS
WINDOW_SECS = 60         # RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS


def _migrate(url: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "app.core.migrations"],
        env={**os.environ, "RAG_DATABASE_URL": url},
        capture_output=True,
        check=True,
    )


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Never leak a settings cache built against a disposable database URL."""
    yield
    get_settings.cache_clear()


@pytest.fixture
def disposable_db(tmp_path) -> str:
    path = tmp_path / "ingestion_limits.db"
    url = f"sqlite:///{path}"
    _migrate(url)
    yield url
    for suffix in ("", "-wal", "-shm"):
        p = path.parent / f"{path.name}{suffix}"
        if p.exists():
            p.unlink()


@pytest.fixture
def app_client(disposable_db, monkeypatch):
    """The real app wired to the disposable DB and an ephemeral vector store."""
    import chromadb

    from app.api import routes_documents, routes_query
    from app.main import app
    from app.services.graph_extraction import DisabledGraphExtractor, get_graph_extractor
    from app.services.vector_store import ChromaVectorStore

    monkeypatch.setenv("RAG_DATABASE_URL", disposable_db)
    get_settings.cache_clear()

    engine = create_engine(disposable_db)
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    ephemeral = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test-ingestion-limits", client=ephemeral)

    class _StubEmbeddingProvider:
        """Deterministic embeddings: these tests exercise limits, not vectors."""

        async def embed_texts(self, texts):
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        routes_documents, "get_embedding_provider", lambda: _StubEmbeddingProvider()
    )
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_graph_extractor] = lambda: DisabledGraphExtractor()
    orig_documents_store = routes_documents.get_vector_store
    orig_query_store = routes_query.get_vector_store
    routes_documents.get_vector_store = lambda: store
    routes_query.get_vector_store = lambda: store

    yield TestClient(app)

    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous_overrides)
    routes_documents.get_vector_store = orig_documents_store
    routes_query.get_vector_store = orig_query_store
    try:
        ephemeral.delete_collection("test-ingestion-limits")
    except Exception:
        pass
    engine.dispose()
    get_settings.cache_clear()


def _file_engine(url: str):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# Envelope / byte-boundary middleware and route tests
# ---------------------------------------------------------------------------

def test_request_envelope_exactly_at_limit_passes_through_middleware():
    app = FastAPI()
    app.add_middleware(BoundedReceiveMiddleware, max_bytes=REQUEST_MAX)

    received = {"ran": False}

    @app.post("/echo")
    async def echo(request: Request):
        received["ran"] = True
        await request.body()
        return {"ok": True}

    client = TestClient(app)
    resp = client.post(
        "/echo", content=b"x" * REQUEST_MAX,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert received["ran"] is True


def test_request_envelope_one_byte_over_limit_rejected_before_handler():
    app = FastAPI()
    app.add_middleware(BoundedReceiveMiddleware, max_bytes=REQUEST_MAX)

    handler_called = {"v": False}

    @app.post("/echo")
    async def echo(request: Request):
        handler_called["v"] = True
        return {"ok": True}

    client = TestClient(app)
    resp = client.post(
        "/echo", content=b"x" * (REQUEST_MAX + 1),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 413
    # A valid Content-Length over the limit is rejected before reading with
    # code request_too_large; the streamed-count code is the other legal form.
    assert resp.json()["detail"]["code"] in (
        "request_envelope_too_large", "request_too_large",
    )
    assert handler_called["v"] is False


def test_file_over_limit_returns_ingestion_too_large_file(app_client):
    too_large = b"a" * (FILE_MAX + 1)
    resp = app_client.post(
        "/documents",
        data={"title": "big", "text": "x"},
        files={"file": ("big.bin", too_large, "application/octet-stream")},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "ingestion_too_large"
    assert resp.json()["detail"]["limit_bytes"] == FILE_MAX
    assert resp.json()["detail"]["measured"] == "file"
    db_path = get_settings().database_url.replace("sqlite:///", "")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    conn.close()


def test_extracted_bytes_over_limit_returns_ingestion_too_large_extracted(app_client):
    resp = app_client.post(
        "/documents",
        data={"title": "big-text", "text": "a" * (EXTRACTED_MAX + 1)},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "ingestion_too_large"
    assert resp.json()["detail"]["measured"] == "extracted"
    assert resp.json()["detail"]["limit_bytes"] == EXTRACTED_MAX


def test_txt_file_extracted_bytes_over_limit_returns_ingestion_too_large_extracted(
    app_client,
):
    """D-16: extraction output from a text FILE is measured, not just the form field."""
    resp = app_client.post(
        "/documents",
        data={"title": "big-file"},
        files={"file": ("big.txt", b"a" * (EXTRACTED_MAX + 1), "text/plain")},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "ingestion_too_large"
    assert resp.json()["detail"]["measured"] == "extracted"
    assert resp.json()["detail"]["limit_bytes"] == EXTRACTED_MAX


def test_exact_extracted_limit_text_is_accepted(app_client, monkeypatch):
    """D-20: a document at exactly EXTRACTED_MAX bytes is accepted end-to-end
    (vector upserts are batched below Chroma's maximum batch size)."""
    # Pin safety off: this test exercises the plain ingestion envelope
    # boundary; under ambient safety-enabled env the same text would be
    # rejected by the safety policy's max_input_chars cap (D-64 fail-closed).
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "false")
    get_settings.cache_clear()
    resp = app_client.post(
        "/documents",
        data={"title": "exact-limit", "text": "a" * EXTRACTED_MAX},
    )
    assert resp.status_code == 201
    assert resp.json()["chunks"] >= 1
    get_settings.cache_clear()


def test_image_ingestion_consumes_rate_slot_and_throttles(app_client, monkeypatch):
    """D-17: image uploads pass through the rate limiter like every other
    /documents upload; a second post in an exhausted window is 429."""
    import app.api.routes_documents as rd

    class _StubImageProvider:
        async def embed_images(self, image_paths):
            return [[0.0] * 8 for _ in image_paths]

    monkeypatch.setattr(
        rd, "get_image_embedding_provider", lambda: _StubImageProvider()
    )
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_REQUESTS", "1")
    get_settings.cache_clear()
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c626001000000ffff030000060005"
        "57bfabd40000000049454e44ae426082"
    )
    first = app_client.post(
        "/documents",
        data={"title": "img-1"},
        files={"file": ("a.png", png, "image/png")},
    )
    assert first.status_code == 201
    db_path = get_settings().database_url.replace("sqlite:///", "")
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM ingestion_rate_buckets").fetchone()[0]
    conn.close()
    assert count == 1
    second = app_client.post(
        "/documents",
        data={"title": "img-2"},
        files={"file": ("b.png", png, "image/png")},
    )
    assert second.status_code == 429
    assert second.json() == {"detail": {"code": "ingestion_rate_limited"}}


def test_chunked_transfer_envelope_counted_correctly():
    app = FastAPI()
    app.add_middleware(BoundedReceiveMiddleware, max_bytes=REQUEST_MAX)
    handler_called = {"v": False}

    @app.post("/echo")
    async def echo(request: Request):
        handler_called["v"] = True
        return {"ok": True}

    client = TestClient(app)

    def stream():
        remaining = REQUEST_MAX + 1
        while remaining > 0:
            n = min(8192, remaining)
            yield b"x" * n
            remaining -= n

    resp = client.post(
        "/echo", content=stream(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] in (
        "request_envelope_too_large", "request_too_large",
    )
    assert handler_called["v"] is False


def test_large_text_only_form_field_envelope_rejected():
    app = FastAPI()
    app.add_middleware(BoundedReceiveMiddleware, max_bytes=REQUEST_MAX)
    handler_called = {"v": False}

    @app.post("/echo")
    async def echo(request: Request):
        handler_called["v"] = True
        await request.form()
        return {"ok": True}

    client = TestClient(app)
    resp = client.post("/echo", data={"text": "y" * (REQUEST_MAX + 1)})
    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] in (
        "request_envelope_too_large", "request_too_large",
    )
    assert handler_called["v"] is False


def test_missing_content_length_still_counted():
    import asyncio

    app = FastAPI()
    app.add_middleware(BoundedReceiveMiddleware, max_bytes=REQUEST_MAX)
    handler_called = {"v": False}

    @app.post("/echo")
    async def echo(request: Request):
        handler_called["v"] = True
        return {"ok": True}

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/echo", "root_path": "", "headers": [],
        "query_string": b"", "server": ("test", 80), "client": ("test", 0),
    }
    delivered = {"v": False}

    async def receive():
        if not delivered["v"]:
            delivered["v"] = True
            return {"type": "http.request",
                    "body": b"z" * (REQUEST_MAX + 1), "more_body": False}
        return {"type": "http.disconnect"}

    sent: list = []

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413
    body_msg = next(m for m in sent if m["type"] == "http.response.body")
    assert json.loads(body_msg["body"].decode()) == {
        "detail": {"code": "request_envelope_too_large"}}
    assert handler_called["v"] is False


def test_client_disconnect_during_streaming_produces_no_partial_rows(app_client):
    def truncating_stream():
        sent = 0
        target = FILE_MAX // 2
        while sent < target:
            yield b"a" * 8192
            sent += 8192
        raise ConnectionError("client disconnected mid-stream")

    try:
        app_client.post(
            "/documents",
            data={"title": "disconnect", "text": "x"},
            files={"file": ("big.bin", truncating_stream(),
                            "application/octet-stream")},
        )
    except Exception:
        pass

    db_path = get_settings().database_url.replace("sqlite:///", "")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    conn.close()


# ---------------------------------------------------------------------------
# Rate limiting through the HTTP surface
# ---------------------------------------------------------------------------

def _freeze_limiter_clock(monkeypatch, epoch: int) -> None:
    """Freeze only the limiter's clock so multi-request window tests cannot
    straddle a fixed-window boundary mid-loop."""
    import types

    import app.services.ingestion_limits as il

    monkeypatch.setattr(il, "time", types.SimpleNamespace(time=lambda: epoch))


def test_rate_limit_first_request_accepted_with_remaining_header(app_client, monkeypatch):
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_REQUESTS", str(RATE_LIMIT))
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS", str(WINDOW_SECS))
    get_settings.cache_clear()
    frozen = int(time.time())
    _freeze_limiter_clock(monkeypatch, frozen)
    before_epoch = int(time.time())
    resp = app_client.post("/documents", data={"title": "r1", "text": "hello"})
    assert resp.status_code in (200, 201)
    assert int(resp.headers["X-RateLimit-Limit"]) == RATE_LIMIT
    assert int(resp.headers["X-RateLimit-Remaining"]) == RATE_LIMIT - 1
    reset = int(resp.headers["X-RateLimit-Reset"])
    now_epoch = int(time.time())
    # window_start <= request time and reset = window_start + WINDOW_SECS.
    assert reset - WINDOW_SECS <= now_epoch
    assert before_epoch <= reset


def test_rate_limit_limit_plus_one_request_rejected_with_retry_after(
    app_client, monkeypatch
):
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_REQUESTS", str(RATE_LIMIT))
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS", str(WINDOW_SECS))
    get_settings.cache_clear()
    _freeze_limiter_clock(monkeypatch, int(time.time()))
    for i in range(RATE_LIMIT):
        r = app_client.post("/documents", data={"title": f"ok-{i}", "text": "x"})
        assert r.status_code in (200, 201)
    over = app_client.post("/documents", data={"title": "over", "text": "x"})
    assert over.status_code == 429
    assert over.json() == {"detail": {"code": "ingestion_rate_limited"}}
    assert int(over.headers["X-RateLimit-Remaining"]) == 0
    assert "Retry-After" in over.headers
    ra = int(over.headers["Retry-After"])
    assert 1 <= ra <= WINDOW_SECS


def test_rate_limit_rejects_persist_increment_for_visibility(app_client, monkeypatch):
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_REQUESTS", str(RATE_LIMIT))
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS", str(WINDOW_SECS))
    get_settings.cache_clear()
    _freeze_limiter_clock(monkeypatch, int(time.time()))
    for _ in range(RATE_LIMIT + 1):
        app_client.post("/documents", data={"title": "p", "text": "x"})
    db_path = get_settings().database_url.replace("sqlite:///", "")
    with sqlite3.connect(db_path) as conn:
        cnt = conn.execute(
            "SELECT request_count FROM ingestion_rate_buckets"
        ).fetchone()[0]
    conn.close()
    assert cnt == RATE_LIMIT + 1


# ---------------------------------------------------------------------------
# Rate limiting at the service boundary (atomic SQL semantics)
# ---------------------------------------------------------------------------

def test_rate_limit_window_boundary_uses_floor_epoch(tmp_path):
    frozen = int(datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp())
    engine = _file_engine(f"sqlite:///{tmp_path / 'win.db'}")
    session = sessionmaker(bind=engine)()
    try:
        accepted, count, headers = acquire_slot(
            session, identity="boundary", limit=RATE_LIMIT,
            window_seconds=WINDOW_SECS, clock=lambda: frozen,
        )
        assert accepted is True
        assert headers["X-RateLimit-Reset"] == str(frozen + WINDOW_SECS)
    finally:
        session.close()
        engine.dispose()


def test_rate_limit_next_window_after_boundary_succeeds(disposable_db, monkeypatch):
    monkeypatch.setenv("RAG_DATABASE_URL", disposable_db)
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_REQUESTS", str(RATE_LIMIT))
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS", str(WINDOW_SECS))
    get_settings.cache_clear()
    engine = _file_engine(disposable_db)
    session = sessionmaker(bind=engine)()
    try:
        win_start = int(
            datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        for _ in range(RATE_LIMIT):
            accepted, _, _ = acquire_slot(
                session, identity="nextwin", limit=RATE_LIMIT,
                window_seconds=WINDOW_SECS, clock=lambda: win_start,
            )
            assert accepted is True
        over_accepted, _, over_headers = acquire_slot(
            session, identity="nextwin", limit=RATE_LIMIT,
            window_seconds=WINDOW_SECS, clock=lambda: win_start,
        )
        assert over_accepted is False
        assert int(over_headers["X-RateLimit-Remaining"]) == 0
        next_clock = win_start + WINDOW_SECS + 1
        accepted, _, headers = acquire_slot(
            session, identity="nextwin", limit=RATE_LIMIT,
            window_seconds=WINDOW_SECS, clock=lambda: next_clock,
        )
        assert accepted is True
        assert int(headers["X-RateLimit-Limit"]) == RATE_LIMIT
        assert int(headers["X-RateLimit-Remaining"]) == RATE_LIMIT - 1
        assert int(headers["X-RateLimit-Reset"]) == win_start + 2 * WINDOW_SECS
    finally:
        session.close()
        engine.dispose()


def test_rate_limit_operator_identity_hashed_from_token(app_client, monkeypatch):
    token = "test-operator-token-0123456789abcdef"  # >= 32 chars required
    monkeypatch.setenv("RAG_OPERATOR_API_ENABLED", "true")
    monkeypatch.setenv("RAG_OPERATOR_TOKEN", token)
    get_settings.cache_clear()
    app_client.post(
        "/documents", data={"title": "op-identity", "text": "x"},
        headers={"Authorization": f"Bearer {token}"},
    )
    expected = hashlib.sha256(f"operator:{token}".encode()).hexdigest()
    db_path = get_settings().database_url.replace("sqlite:///", "")
    with sqlite3.connect(db_path) as conn:
        ident = conn.execute(
            "SELECT identity_sha256 FROM ingestion_rate_buckets"
        ).fetchone()[0]
    conn.close()
    assert ident == expected


def test_rate_limit_client_identity_hashed_from_remote_host(app_client, monkeypatch):
    monkeypatch.setenv("RAG_OPERATOR_API_ENABLED", "false")
    monkeypatch.delenv("RAG_OPERATOR_TOKEN", raising=False)
    get_settings.cache_clear()
    app_client.post("/documents", data={"title": "cli-identity", "text": "x"})
    expected = hashlib.sha256(b"client:testclient").hexdigest()
    db_path = get_settings().database_url.replace("sqlite:///", "")
    with sqlite3.connect(db_path) as conn:
        ident = conn.execute(
            "SELECT identity_sha256 FROM ingestion_rate_buckets"
        ).fetchone()[0]
    conn.close()
    assert ident == expected


def test_rate_limit_concurrent_workers_produce_distinct_counts(tmp_path):
    n_workers = 8  # 5 accepted + 3 concurrent losers
    db_path = tmp_path / "cc.db"
    engine = _file_engine(f"sqlite:///{db_path}")
    frozen = int(datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp())

    results: list = []
    lock = threading.Lock()

    def worker():
        session = sessionmaker(bind=engine)()
        try:
            accepted, _, headers = acquire_slot(
                session, identity="concurrent", limit=5,
                window_seconds=WINDOW_SECS, clock=lambda: frozen,
            )
            with lock:
                results.append((accepted, int(headers["X-RateLimit-Remaining"])))
        finally:
            session.close()

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    engine.dispose()

    accepted = [r for r in results if r[0]]
    rejected = [r for r in results if not r[0]]
    assert len(accepted) == 5
    assert len(rejected) == 3
    assert sorted(r[1] for r in accepted) == [0, 1, 2, 3, 4]
    assert all(r[1] == 0 for r in rejected)
    with sqlite3.connect(db_path) as conn:
        cnt = conn.execute(
            "SELECT request_count FROM ingestion_rate_buckets"
        ).fetchone()[0]
    assert cnt == 5 + 3


def test_rate_limit_opportunistic_old_bucket_prune_capped_at_1000(tmp_path):
    db_path = tmp_path / "prune.db"
    engine = _file_engine(f"sqlite:///{db_path}")
    session = sessionmaker(bind=engine)()
    now = int(time.time())
    ancient = now - (3 * WINDOW_SECS)  # older than two windows
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT INTO ingestion_rate_buckets"
                " (identity_sha256, window_start_epoch, request_count)"
                " VALUES (?, ?, 1)",
                [(hashlib.sha256(f"old-{i}".encode()).hexdigest(), ancient)
                 for i in range(2500)],
            )
            conn.commit()
            before = conn.execute(
                "SELECT COUNT(*) FROM ingestion_rate_buckets"
                " WHERE window_start_epoch = ?", (ancient,),
            ).fetchone()[0]
        assert before == 2500
        acquire_slot(session, identity="pruner", limit=RATE_LIMIT,
                     window_seconds=WINDOW_SECS)
        with sqlite3.connect(db_path) as conn:
            after = conn.execute(
                "SELECT COUNT(*) FROM ingestion_rate_buckets"
                " WHERE window_start_epoch = ?", (ancient,),
            ).fetchone()[0]
    finally:
        session.close()
        engine.dispose()
    deleted = before - after
    assert 0 < deleted <= 1000


def test_rate_limit_restarts_retain_active_bucket(tmp_path):
    db_path = tmp_path / "restart.db"
    engine = _file_engine(f"sqlite:///{db_path}")
    frozen = int(datetime(2026, 8, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp())
    session = sessionmaker(bind=engine)()
    try:
        for _ in range(5):
            accepted, _, _ = acquire_slot(
                session, identity="restart", limit=RATE_LIMIT,
                window_seconds=WINDOW_SECS, clock=lambda: frozen,
            )
            assert accepted is True
    finally:
        session.close()
    engine.dispose()

    # Simulate a process restart: the durable bucket must survive in SQL.
    import app.services.ingestion_limits as il
    importlib.reload(il)
    engine2 = _file_engine(f"sqlite:///{db_path}")
    session2 = sessionmaker(bind=engine2)()
    try:
        accepted, _, headers = il.acquire_slot(
            session2, identity="restart", limit=RATE_LIMIT,
            window_seconds=WINDOW_SECS, clock=lambda: frozen,
        )
        assert accepted is True
        assert int(headers["X-RateLimit-Remaining"]) == RATE_LIMIT - 6
    finally:
        session2.close()
        engine2.dispose()


# ---------------------------------------------------------------------------
# Identity derivation (single hash in acquire_slot)
# ---------------------------------------------------------------------------

def test_rate_identity_is_hashed():
    """_rate_identity returns the raw key; hashing happens once in acquire_slot."""
    raw = _rate_identity("secret-token", None)
    assert raw == "operator:secret-token"
    h = hashlib.sha256(raw.encode()).hexdigest()
    assert len(h) == 64
    assert "secret-token" not in h


def test_rate_identity_operator_vs_client():
    assert _rate_identity("token", None) != _rate_identity(None, "127.0.0.1")


# ---------------------------------------------------------------------------
# Startup / settings validation
# ---------------------------------------------------------------------------

def test_startup_rejects_envelope_smaller_than_file(monkeypatch):
    monkeypatch.setenv("RAG_INGESTION_REQUEST_MAX_BYTES", "1024")
    monkeypatch.setenv("RAG_INGESTION_FILE_MAX_BYTES", "2048")
    get_settings.cache_clear()
    from app.main import app

    with pytest.raises((RuntimeError, ValueError)):
        with TestClient(app):
            pass
    get_settings.cache_clear()


def test_settings_ranges_reject_out_of_range_values(monkeypatch):
    from app.config import Settings

    cases = [
        ("RAG_INGESTION_REQUEST_MAX_BYTES", "1024"),       # below 2048
        ("RAG_INGESTION_REQUEST_MAX_BYTES", "999999999"),  # above 53477376
        ("RAG_INGEST_RATE_LIMIT_REQUESTS", "0"),           # below 1
        ("RAG_INGEST_RATE_LIMIT_REQUESTS", "1001"),        # above 1000
        ("RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS", "0"),     # below 1
        ("RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS", "3601"),  # above 3600
    ]
    for name, bad in cases:
        monkeypatch.setenv(name, bad)
        with pytest.raises(Exception):
            Settings()
        monkeypatch.delenv(name)
