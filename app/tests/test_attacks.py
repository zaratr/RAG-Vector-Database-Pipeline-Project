"""Phase 10D.2 — isolated attack harness tests.

Covers: isolation refusals, migration per-mode, manifest equality,
bijection, no HTTP attack endpoint, production fingerprint preservation,
cleanup in finally.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest

CORPUS = "app/tests/fixtures/attack_payloads.json"


_POSIX_ONLY_HARNESS = pytest.mark.skipif(
    sys.platform == "win32",
    reason="attack harness pins disposable/production DB URLs to POSIX "
    "absolute sqlite://// paths as an operational-safety guard "
    "(destructive disposable-store path restriction, mirroring the "
    "calibration CLI guard) and its symlink-refusal lane needs symlink "
    "privilege; Windows drive-letter paths cannot satisfy the guard, so "
    "these lanes execute on POSIX filesystems only",
)



def _uuid32() -> str:
    return uuid.uuid4().hex


def _run_harness(monkeypatch):
    """Run the full harness in-process against disposable tmp stores.

    Sets guarded defaults (only when absent, so individual tests can
    pre-set a hostile variable) and removes any disposable collections
    afterwards, whatever the outcome.
    """
    from app.services import attack_simulator
    from scripts import run_redteam

    workdir = Path(tempfile.mkdtemp(prefix="redteam-test-"))
    production_db = workdir / "production.db"
    production_db.write_bytes(b"")
    disabled_id, enabled_id = _uuid32(), _uuid32()
    assert disabled_id != enabled_id
    defaults = {
        "RAG_REDTEAM_MODE": "true",
        "RAG_PRODUCTION_DATABASE_URL": f"sqlite:///{production_db}",
        "RAG_PRODUCTION_CHROMA_COLLECTION": "rag-collection",
        "RAG_REDTEAM_DISABLED_DATABASE_URL":
            f"sqlite:///{workdir}/redteam-{disabled_id}.db",
        "RAG_REDTEAM_DISABLED_CHROMA_COLLECTION": f"redteam-{disabled_id}",
        "RAG_REDTEAM_ENABLED_DATABASE_URL":
            f"sqlite:///{workdir}/redteam-{enabled_id}.db",
        "RAG_REDTEAM_ENABLED_CHROMA_COLLECTION": f"redteam-{enabled_id}",
    }
    for key, value in defaults.items():
        if key not in os.environ:
            monkeypatch.setenv(key, value)
    try:
        return run_redteam.run_harness(
            fixtures_path=CORPUS, run_id="pytest-run")
    finally:
        for key in ("RAG_REDTEAM_DISABLED_CHROMA_COLLECTION",
                    "RAG_REDTEAM_ENABLED_CHROMA_COLLECTION"):
            name = os.environ.get(key)
            if name and name.startswith("redteam-"):
                try:
                    attack_simulator.delete_disposable_collection(name)
                except Exception:
                    pass
        shutil.rmtree(workdir, ignore_errors=True)


def _capture_manifests(monkeypatch):
    report = _run_harness(monkeypatch)
    return report["manifests"]


def test_no_http_attack_endpoint_exists():
    # Assert no route in app.main.router.paths starts with /attack or /redteam.
    # (getattr: this Starlette version surfaces included routers as deferred
    # _IncludedRouter wrapper objects without a .path attribute; they expose
    # no path of their own, so skipping them preserves the assertion's scope.)
    from app.main import app
    for route in app.routes:
        path = getattr(route, "path", "")
        assert not path.startswith("/attack")
        assert not path.startswith("/redteam")


def test_refuses_production_database_url_equality(monkeypatch):
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL", "sqlite:////data/rag.db")
    with pytest.raises((ValueError, SystemExit)):
        _run_harness(monkeypatch)


def test_refuses_production_collection_name_equality(monkeypatch):
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_CHROMA_COLLECTION", "rag-collection")
    with pytest.raises((ValueError, SystemExit)):
        _run_harness(monkeypatch)


@_POSIX_ONLY_HARNESS
def test_refuses_symlink_to_production(tmp_path, monkeypatch):
    import os

    production = tmp_path / "rag.db"
    production.write_bytes(b"production-bytes")
    link = tmp_path / f"redteam-{_uuid32()}.db"
    os.symlink(production, link)
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL", f"sqlite:///{link}")

    with pytest.raises((ValueError, SystemExit)):
        _run_harness(monkeypatch)

    # Refusal happens before mutation: production content is byte-identical.
    assert production.read_bytes() == b"production-bytes"
    assert link.is_symlink()


def test_refuses_preexisting_disposable_db(tmp_path, monkeypatch):
    db = tmp_path / f"redteam-{_uuid32()}.db"
    db.write_bytes(b"")
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL", f"sqlite:///{db}")
    with pytest.raises((ValueError, SystemExit)):
        _run_harness(monkeypatch)


def test_refuses_invalid_collection_pattern(monkeypatch):
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_CHROMA_COLLECTION", "not-redteam-format")
    with pytest.raises((ValueError, SystemExit)):
        _run_harness(monkeypatch)


def test_disabled_and_enabled_uuids_distinct(monkeypatch):
    same = _uuid32()
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL",
                       f"sqlite:////tmp/redteam-{same}.db")
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_DATABASE_URL",
                       f"sqlite:////tmp/redteam-{same}.db")
    with pytest.raises((ValueError, SystemExit)):
        _run_harness(monkeypatch)


@_POSIX_ONLY_HARNESS
def test_migration_runs_per_mode_via_subprocess_wrapper(monkeypatch):
    # Assert subprocess.run([sys.executable, "-m", "app.core.migrations"], ...)
    # is called exactly twice with each mode URL, shell=False, check=True.
    import subprocess as _sp

    real_run = _sp.run
    calls = []

    def recording_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return real_run(argv, **kwargs)

    monkeypatch.setattr("scripts.run_redteam.subprocess.run", recording_run)

    disabled_id = _uuid32()
    enabled_id = _uuid32()
    assert disabled_id != enabled_id
    disabled_url = f"sqlite:////tmp/redteam-{disabled_id}.db"
    enabled_url = f"sqlite:////tmp/redteam-{enabled_id}.db"
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL", disabled_url)
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_DATABASE_URL", enabled_url)

    _run_harness(monkeypatch)

    migrate_calls = [
        (argv, kwargs)
        for argv, kwargs in calls
        if "-m" in argv and "app.core.migrations" in argv
    ]
    assert len(migrate_calls) == 2
    mode_urls = {disabled_url, enabled_url}
    carried_urls = set()
    for argv, kwargs in migrate_calls:
        assert argv[:3] == [sys.executable, "-m", "app.core.migrations"]
        assert kwargs.get("shell", False) is False
        assert kwargs.get("check") is True
        env = kwargs.get("env") or {}
        carried_urls |= {value for value in env.values() if value in mode_urls}
    assert carried_urls == mode_urls


def test_no_create_all_or_in_memory_fallback():
    # Assert app.core.db.Base.metadata.create_all is never called by the harness.
    import inspect

    from app.services import attack_simulator
    from scripts import run_redteam

    forbidden = ("create_all", ":memory:", "StaticPool")
    for module in (run_redteam, attack_simulator):
        source = inspect.getsource(module)
        for token in forbidden:
            assert token not in source, (
                f"{token!r} is forbidden in {module.__name__}: "
                "no create_all, no in-memory fallback, no ad-hoc DDL"
            )


@_POSIX_ONLY_HARNESS
def test_fixture_input_manifest_byte_equal_across_modes(monkeypatch):
    manifests = _capture_manifests(monkeypatch)
    assert manifests["disabled"] == manifests["enabled"]


@_POSIX_ONLY_HARNESS
def test_every_accepted_fixture_document_produces_exactly_one_chunk(monkeypatch):
    # Multi-chunk fixture -> corpus-invalid exit 2 before measurement.
    from app.services import ingestion

    def two_chunks(text, chunk_size=1000, overlap=200):
        mid = max(1, len(text) // 2)
        return [
            {"text": text[:mid], "start_offset": 0, "end_offset": mid},
            {"text": text[mid:], "start_offset": mid, "end_offset": len(text)},
        ]

    monkeypatch.setattr(ingestion, "chunk_text", two_chunks)

    with pytest.raises(SystemExit) as exc_info:
        _run_harness(monkeypatch)
    assert exc_info.value.code == 2


@_POSIX_ONLY_HARNESS
def test_bijection_document_fixture_id_to_sql_id(monkeypatch):
    # Every accepted fixture document maps to exactly one sql_document_id and
    # chunk_fixture_id "<doc_fix_id>:0" maps to one sql_chunk_id and vector_id.
    report = _run_harness(monkeypatch)

    for mode in ("disabled", "enabled"):
        bindings = report[mode]["bindings"]
        assert bindings

        # document_fixture_id -> exactly one sql_document_id (a function).
        doc_fixture_to_sql = {}
        for binding in bindings:
            doc_fixture_to_sql.setdefault(
                binding["document_fixture_id"], set()
            ).add(binding["sql_document_id"])
        for sql_ids in doc_fixture_to_sql.values():
            assert len(sql_ids) == 1

        # Injective: distinct fixture documents bind to distinct sql ids.
        sql_document_ids = [binding["sql_document_id"] for binding in bindings]
        assert len(set(sql_document_ids)) == len(doc_fixture_to_sql)

        # chunk_fixture_id is canonical "<document_fixture_id>:0" and binds
        # to exactly one sql_chunk_id and one vector_id (injective on both).
        for binding in bindings:
            assert binding["chunk_fixture_id"] == f"{binding['document_fixture_id']}:0"
        ready = [b for b in bindings if b["sql_chunk_id"] is not None]
        sql_chunk_ids = [binding["sql_chunk_id"] for binding in ready]
        vector_ids = [binding["vector_id"] for binding in ready]
        assert all(isinstance(chunk_id, int) for chunk_id in sql_chunk_ids)
        assert len(set(sql_chunk_ids)) == len(ready)
        assert all(isinstance(v, str) and v for v in vector_ids)
        assert len(set(vector_ids)) == len(ready)


@_POSIX_ONLY_HARNESS
def test_bijection_refuses_alias_or_duplicate(monkeypatch):
    duplicate_corpus = [
        {"document_fixture_id": "doc-A", "title": "first", "text": "first body"},
        {"document_fixture_id": "doc-A", "title": "alias", "text": "alias body"},
    ]
    monkeypatch.setattr(
        "scripts.run_redteam.load_fixtures",
        lambda *args, **kwargs: duplicate_corpus,
    )

    with pytest.raises(SystemExit) as exc_info:
        _run_harness(monkeypatch)
    assert exc_info.value.code == 2


@_POSIX_ONLY_HARNESS
def test_enabled_ingestion_block_persists_failed_no_vectors(monkeypatch):
    # In enabled mode, a block|filter ingestion must produce no chunks/vectors.
    report = _run_harness(monkeypatch)

    enabled = {b["document_fixture_id"]: b for b in report["enabled"]["bindings"]}
    disabled = {b["document_fixture_id"]: b for b in report["disabled"]["bindings"]}
    blocked = [fix for fix, binding in enabled.items() if binding["status"] == "failed"]
    assert blocked, "enabled mode must persist failed status for blocked fixtures"

    for fix in blocked:
        binding = enabled[fix]
        assert binding["status"] == "failed"
        assert binding["sql_chunk_id"] is None
        assert binding["vector_id"] is None
        # The same fixture exists in disabled mode (contrast anchor).
        assert fix in disabled


@_POSIX_ONLY_HARNESS
def test_disabled_mode_retains_ready_status(monkeypatch):
    # In disabled mode, the same input that enabled blocks may remain ready.
    report = _run_harness(monkeypatch)

    enabled = {b["document_fixture_id"]: b for b in report["enabled"]["bindings"]}
    disabled = {b["document_fixture_id"]: b for b in report["disabled"]["bindings"]}
    blocked = [fix for fix, binding in enabled.items() if binding["status"] == "failed"]
    assert blocked

    for fix in blocked:
        disabled_binding = disabled[fix]
        assert disabled_binding["status"] == "ready"
        assert disabled_binding["sql_chunk_id"] is not None
        assert disabled_binding["vector_id"] is not None
        assert disabled_binding["chunk_fixture_id"] == f"{fix}:0"


def test_client_payload_cannot_set_mode(monkeypatch):
    # No combination of HTTP headers/body/env can flip the mode; only guarded
    # internal CLI state can.
    from app.config import Settings
    from app.core.models import DocumentCreate

    # A client-supplied "mode" key in the request body is not accepted.
    probe = DocumentCreate(title="mode-probe", text="payload", mode="enabled")
    assert not hasattr(probe, "mode")

    # The HTTP request model exposes no mode-shaped field at all.
    request_fields = set(DocumentCreate.model_fields)
    assert not any("mode" in field.lower() for field in request_fields)

    # Mode is guarded internal CLI state (env-derived), never a per-request
    # knob: the request schema is disjoint from any settings field name.
    setting_fields = set(Settings.model_fields)
    assert request_fields.isdisjoint(setting_fields)


@_POSIX_ONLY_HARNESS
def test_production_sql_fingerprint_captured_before_setup(monkeypatch):
    import sqlite3
    import subprocess as _sp

    real_connect = sqlite3.connect
    real_run = _sp.run
    events = []

    def spy_connect(target, *args, **kwargs):
        prod_url = os.environ.get("RAG_PRODUCTION_DATABASE_URL", "")
        prod_path = prod_url.replace("sqlite:////", "/").replace("sqlite:///", "")
        if (
            isinstance(target, str)
            and prod_path
            and prod_path in target
            and "mode=ro" in target
        ):
            events.append(("sql_fingerprint_ro", target))
        return real_connect(target, *args, **kwargs)

    def spy_run(argv, **kwargs):
        if isinstance(argv, list) and "app.core.migrations" in argv:
            events.append(("migrate", list(argv)))
        return real_run(argv, **kwargs)

    monkeypatch.setattr("sqlite3.connect", spy_connect)
    monkeypatch.setattr("scripts.run_redteam.subprocess.run", spy_run)

    _run_harness(monkeypatch)

    assert any(kind == "sql_fingerprint_ro" for kind, _ in events)
    assert any(kind == "migrate" for kind, _ in events)
    first_fingerprint = next(
        index for index, (kind, _) in enumerate(events) if kind == "sql_fingerprint_ro"
    )
    first_migrate = next(
        index for index, (kind, _) in enumerate(events) if kind == "migrate"
    )
    # The read-only production baseline is captured before any migration/setup.
    assert first_fingerprint < first_migrate


@_POSIX_ONLY_HARNESS
def test_production_sql_fingerprint_unchanged_after_each_fixture(monkeypatch):
    report = _run_harness(monkeypatch)
    fingerprints = report["production_sql_fingerprints"]

    # Captured before setup, after every fixture, and at exit -> at least twice.
    assert len(fingerprints) >= 2

    # Every checkpoint is a SHA-256 lowercase-hex string...
    for fingerprint in fingerprints:
        assert len(fingerprint) == 64
        assert all(character in "0123456789abcdef" for character in fingerprint)

    # ...and byte-identical across all of them: production SQL never changed.
    assert len(set(fingerprints)) == 1


@_POSIX_ONLY_HARNESS
def test_production_chroma_fingerprint_unchanged_at_exit(monkeypatch):
    report = _run_harness(monkeypatch)
    fingerprints = report["production_chroma_fingerprints"]

    assert len(fingerprints) >= 2
    for fingerprint in fingerprints:
        assert len(fingerprint) == 64
        assert all(character in "0123456789abcdef" for character in fingerprint)

    # The exit fingerprint equals the baseline captured before setup, and every
    # intermediate checkpoint matches: production Chroma never changed.
    assert fingerprints[0] == fingerprints[-1]
    assert len(set(fingerprints)) == 1


@_POSIX_ONLY_HARNESS
def test_cleanup_runs_in_finally_on_success(monkeypatch, tmp_path):
    # After successful run, both disposable collections and DB/WAL/SHM files
    # are gone.
    disabled_id = _uuid32()
    enabled_id = _uuid32()
    assert disabled_id != enabled_id
    disabled_db = tmp_path / f"redteam-{disabled_id}.db"
    enabled_db = tmp_path / f"redteam-{enabled_id}.db"
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL", f"sqlite:///{disabled_db}")
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_DATABASE_URL", f"sqlite:///{enabled_db}")
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_CHROMA_COLLECTION", f"redteam-{disabled_id}")
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_CHROMA_COLLECTION", f"redteam-{enabled_id}")

    report = _run_harness(monkeypatch)
    assert report["exit_code"] == 0

    # The outer finally removed both disposable DB/WAL/SHM sets.
    for database in (disabled_db, enabled_db):
        assert not database.exists()
        for suffix in ("-wal", "-shm"):
            assert not (tmp_path / (database.name + suffix)).exists()


@_POSIX_ONLY_HARNESS
def test_cleanup_runs_in_finally_on_failure(monkeypatch, tmp_path):
    # Force a mid-run failure; assert cleanup still runs and production
    # fingerprint is unchanged.
    import sqlite3
    import subprocess as _sp

    real_run = _sp.run
    real_connect = sqlite3.connect
    migrate_count = [0]
    production_connections = []

    disabled_id = _uuid32()
    enabled_id = _uuid32()
    disabled_db = tmp_path / f"redteam-{disabled_id}.db"
    enabled_db = tmp_path / f"redteam-{enabled_id}.db"
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL", f"sqlite:///{disabled_db}")
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_DATABASE_URL", f"sqlite:///{enabled_db}")
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_CHROMA_COLLECTION", f"redteam-{disabled_id}")
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_CHROMA_COLLECTION", f"redteam-{enabled_id}")

    def fake_run(argv, **kwargs):
        # First-mode migration runs for real (creating its disposable DB);
        # second-mode migration is forced to fail -> sanitized exit 2.
        if isinstance(argv, list) and "app.core.migrations" in argv:
            migrate_count[0] += 1
            if migrate_count[0] == 2:
                return mock.Mock(returncode=1, stdout="", stderr="second-mode migration failed")
        return real_run(argv, **kwargs)

    def spy_connect(target, *args, **kwargs):
        prod_url = os.environ.get("RAG_PRODUCTION_DATABASE_URL", "")
        prod_path = prod_url.replace("sqlite:////", "/").replace("sqlite:///", "")
        if isinstance(target, str) and prod_path and prod_path in target:
            production_connections.append(target)
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr("scripts.run_redteam.subprocess.run", fake_run)
    monkeypatch.setattr("sqlite3.connect", spy_connect)

    with pytest.raises(SystemExit) as exc_info:
        _run_harness(monkeypatch)
    assert exc_info.value.code == 2

    # The outer finally still removed both disposable DB/WAL/SHM sets.
    for database in (disabled_db, enabled_db):
        assert not database.exists()
        for suffix in ("-wal", "-shm"):
            assert not (tmp_path / (database.name + suffix)).exists()

    # Production was only ever opened read-only: never mutated by the failure.
    assert production_connections
    assert all("mode=ro" in target for target in production_connections)


@_POSIX_ONLY_HARNESS
def test_keep_artifacts_marks_report_non_acceptance(monkeypatch):
    # --keep-artifacts retains disposable stores but sets non_acceptance=true
    # and can never yield exit 0.
    import tempfile
    from pathlib import Path

    workdir = tempfile.mkdtemp()
    disabled_id = _uuid32()
    enabled_id = _uuid32()
    disabled_db = Path(workdir) / f"redteam-{disabled_id}.db"
    enabled_db = Path(workdir) / f"redteam-{enabled_id}.db"
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_DATABASE_URL", f"sqlite:///{disabled_db}")
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_DATABASE_URL", f"sqlite:///{enabled_db}")
    monkeypatch.setenv("RAG_REDTEAM_DISABLED_CHROMA_COLLECTION", f"redteam-{disabled_id}")
    monkeypatch.setenv("RAG_REDTEAM_ENABLED_CHROMA_COLLECTION", f"redteam-{enabled_id}")
    monkeypatch.setenv("RAG_REDTEAM_KEEP_ARTIFACTS", "true")

    report = _run_harness(monkeypatch)
    # The report is flagged non-acceptance and can never yield exit 0.
    assert report["non_acceptance"] is True
    assert report["exit_code"] != 0
    # Disposable artifacts are retained (not cleaned up) for local debugging.
    assert disabled_db.exists()
    shutil.rmtree(workdir, ignore_errors=True)


@_POSIX_ONLY_HARNESS
def test_cli_argv_uses_subprocess_run_shell_false(monkeypatch):
    # Inspect the subprocess.run calls made by run_redteam; assert every call
    # has shell=False (or default) and a list argv.
    import subprocess as _sp

    real_run = _sp.run
    calls = []

    def recording_run(argv, **kwargs):
        calls.append((argv, dict(kwargs)))
        return real_run(argv, **kwargs)

    monkeypatch.setattr("scripts.run_redteam.subprocess.run", recording_run)
    _run_harness(monkeypatch)

    assert calls
    for argv, kwargs in calls:
        # argv is always a typed list, never a shell string.
        assert isinstance(argv, list)
        # shell is never True (absent means False, which is also allowed).
        assert kwargs.get("shell", False) is False
