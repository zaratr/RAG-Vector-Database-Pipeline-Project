"""Hermetic contract tests for ``scripts/validate_phase10c.py`` (Task 10C.5).

``validate_phase10c`` is the LIVE end-to-end validator: its exit-0 happy path
by design drives a running, safety-enabled API (ingestion block/warn, query
with safety summary, operator finding/stats/auth surfaces). These tests prove
everything DETERMINISTIC about its contract hermetically, mirroring the
``test_validate_reconciliation`` / ``test_validate_graph_backfill`` patterns:

* every refusal lane exits 2 with the exact machine message and no stdout
  (content safety disabled, missing operator token, non-sqlite database URL,
  non-absolute sqlite URL, invalid arguments);
* against a disposable fully-migrated DB with an unreachable API, the
  failure lane still runs its ``finally`` cleanup and leaves the store's
  fingerprint — and the file bytes — untouched (self-cleaning under failure,
  never a partial fixture);
* an unrelated-SQL fingerprint change between capture and cleanup is caught
  and withholds success with exit 2, while the unrelated rows themselves are
  left in place (reported, not silently destroyed);
* the SQLite fingerprint helper is read-only, deterministic, and excludes
  ``alembic_version`` / ``ingestion_rate_buckets`` exactly as pinned.

The full exit-0 happy path needs a live safety-enabled API and is therefore
an EXPLICIT OPT-IN lane (``RAG_PHASE10C_LIVE_BASE_URL``): never silently
skipped — without the variable the test reports itself as skipped with that
reason; with it, the run must exit 0 and print the closed success JSON
contract over a disposable store.

The script has no ``--database-url``/``--allow-disposable-database`` flags of
its own: it operates on the CONFIGURED ``RAG_DATABASE_URL`` (and refuses
non-absolute sqlite URLs), so these tests follow that convention — every lane
runs as a subprocess with a hermetic environment (no ``RAG_*`` inheritance,
``cwd`` outside the repo so no ``.env`` applies) against disposable stores.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "validate_phase10c.py"
_OPERATOR_TOKEN = "t" * 48  # validator requires a configured token


def _slash_form_url(db_path: Path) -> str:
    """sqlite URL that satisfies the script's absolute-path contract.

    The script requires the part after ``sqlite:///`` to start with ``/`` and
    uses it as a POSIX-flavored absolute path. On Windows the driveless form
    (``sqlite:////Users/...``) resolves against the current drive, so the test
    keeps ``cwd`` on the same drive as the disposable DB.
    """
    return "sqlite:///" + db_path.resolve().as_posix().split(":", 1)[1]


def _migrated_db(tmp_path: Path) -> Path:
    """Disposable DB migrated to the current head (graph + safety tables)."""
    db_path = tmp_path / f"validate-phase10c-{secrets.token_hex(16)}.db"
    subprocess.run(
        [sys.executable, "-m", "app.core.migrations"],
        env={**os.environ, "RAG_DATABASE_URL": f"sqlite:///{db_path}"},
        capture_output=True,
        check=True,
        cwd=str(_PROJECT_ROOT),
    )
    return db_path


def _hermetic_env() -> dict:
    """Environment with no inherited RAG_* configuration; safety enabled."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
    env.update({
        "RAG_DATABASE_URL": "sqlite:///absent.db",
        "RAG_CONTENT_SAFETY_ENABLED": "true",
        "RAG_OPERATOR_TOKEN": _OPERATOR_TOKEN,
    })
    return env


def _run_script(env: dict, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),  # outside the repo: no .env, no stray stores
        env=env,
    )


# ---------------------------------------------------------------------------
# Refusal lanes: exit 2, exact machine message, no stdout
# ---------------------------------------------------------------------------


def test_invalid_argument_exits_2(tmp_path):
    result = _run_script(_hermetic_env(), tmp_path, "--bogus")
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert result.stdout == ""


def test_refuses_when_content_safety_disabled(tmp_path):
    env = _hermetic_env()
    del env["RAG_CONTENT_SAFETY_ENABLED"]
    result = _run_script(env, tmp_path)
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "validate_phase10c: content safety must be enabled for this validator"
    )
    assert result.stdout == ""


def test_refuses_without_operator_token(tmp_path):
    env = _hermetic_env()
    del env["RAG_OPERATOR_TOKEN"]
    result = _run_script(env, tmp_path)
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "validate_phase10c: operator API must be enabled with a configured token"
    )
    assert result.stdout == ""


def test_refuses_non_sqlite_database_url(tmp_path):
    env = _hermetic_env()
    env["RAG_DATABASE_URL"] = "postgresql://user:secret@localhost:5432/rag"
    result = _run_script(env, tmp_path)
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "validate_phase10c: RAG_DATABASE_URL must be sqlite for this validator"
    )
    assert result.stdout == ""


def test_refuses_relative_sqlite_url(tmp_path):
    env = _hermetic_env()
    env["RAG_DATABASE_URL"] = "sqlite:///relative/rag.db"
    result = _run_script(env, tmp_path)
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "validate_phase10c: RAG_DATABASE_URL must be an absolute "
        "sqlite:////path URL"
    )
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# Failure lane over a disposable migrated store: finally-cleanup runs and the
# fingerprint guard holds when nothing changed
# ---------------------------------------------------------------------------


def test_unreachable_api_failure_lane_leaves_store_byte_intact(tmp_path):
    """With valid settings over a disposable fully-migrated DB and an
    unreachable API, the validator fails nonzero (uncaught transport error),
    emits no success JSON, and its finally-cleanup leaves the store's
    fingerprint and file bytes untouched."""
    from scripts import validate_phase10c as validator

    db_path = _migrated_db(tmp_path)
    env = _hermetic_env()
    env["RAG_DATABASE_URL"] = _slash_form_url(db_path)

    fingerprint_before = validator._fingerprint_sqlite(db_path)
    sha_before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    result = _run_script(env, tmp_path, "--base-url", "http://127.0.0.1:1")

    assert result.returncode != 0
    assert result.stdout == ""
    assert "ConnectError" in result.stderr

    # Self-cleaning under failure: fingerprint (row-count snapshot) and file
    # bytes unchanged, no journal sidecars left behind.
    assert validator._fingerprint_sqlite(db_path) == fingerprint_before
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == sha_before
    for suffix in ("-journal", "-wal", "-shm"):
        assert not db_path.with_name(db_path.name + suffix).exists()


def test_unrelated_fingerprint_change_withholds_success(tmp_path, monkeypatch, capsys):
    """A change to unrelated SQL state between the before-fingerprint and the
    finally-cleanup must withhold success: exit code 2 with the mismatch
    surfaced, and the unrelated rows themselves preserved (reported, never
    silently destroyed). Injected via the first HTTP call so the script's own
    fingerprint comparison and cleanup path are what run."""
    import httpx
    from pydantic import SecretStr

    from scripts import validate_phase10c as validator

    db_path = _migrated_db(tmp_path)
    url = _slash_form_url(db_path)

    settings = SimpleNamespace(
        content_safety_enabled=True,
        operator_token=SecretStr(_OPERATOR_TOKEN),
        database_url=url,
    )
    # The script imports get_settings from app.config at call time inside
    # main(), so patching the module attribute redirects it deterministically.
    monkeypatch.setattr("app.config.get_settings", lambda: settings)

    class _MutatingClient:
        """Stands in for httpx.Client: the first POST writes an unrelated
        document row (fingerprinted table, not run-tag cleanup scope) and then
        fails like an unreachable API."""

        def post(self, path, **kwargs):
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "INSERT INTO documents (title, ingestion_status, trust_tier, "
                "trust_score, trust_policy_version, ingestion_origin) "
                "VALUES ('unrelated-doc', 'ready', 'standard', 0.5, 'p', 'api')"
            )
            conn.commit()
            conn.close()
            raise httpx.ConnectError("injected: API unreachable")

        def get(self, *args, **kwargs):  # pragma: no cover - unreachable
            raise AssertionError("script must fail before any GET")

    monkeypatch.setattr(
        validator, "httpx",
        SimpleNamespace(Client=lambda **kwargs: _MutatingClient()),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["validate_phase10c.py", "--base-url", "http://127.0.0.1:1"],
    )

    with pytest.raises(SystemExit) as excinfo:
        validator.main()
    assert excinfo.value.code == 2

    captured = capsys.readouterr()
    assert captured.out == ""  # no success JSON on withheld success
    assert "unrelated SQL state changed" in captured.err
    assert "SQL fingerprint mismatch" in captured.err

    # The unrelated row survives: cleanup removed only run-tag scope.
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT title FROM documents ORDER BY id").fetchall()
    conn.close()
    assert rows == [("unrelated-doc",)]


# ---------------------------------------------------------------------------
# Fingerprint helper contract
# ---------------------------------------------------------------------------


def test_fingerprint_sqlite_read_only_deterministic_and_scoped(tmp_path):
    from scripts import validate_phase10c as validator

    db_path = tmp_path / "fingerprint-probe.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT)")
    conn.execute("CREATE TABLE alembic_version (version_num TEXT)")
    conn.execute("CREATE TABLE ingestion_rate_buckets (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO documents (id, title) VALUES (1, 'a'), (2, 'b')")
    conn.execute("INSERT INTO alembic_version VALUES ('d9b5f7c1e4a3')")
    conn.execute("INSERT INTO ingestion_rate_buckets VALUES (7)")
    conn.commit()
    conn.close()
    before_stat = db_path.stat()

    expected = hashlib.sha256(json.dumps(
        {"tables": [{"name": "documents", "row_count": 2}]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    first = validator._fingerprint_sqlite(db_path)
    second = validator._fingerprint_sqlite(db_path)

    assert first == expected  # exact canonical snapshot, exclusions pinned
    assert first == second  # deterministic across calls
    after_stat = db_path.stat()
    assert (before_stat.st_mtime_ns, before_stat.st_size) == (
        after_stat.st_mtime_ns, after_stat.st_size)  # read-only probe

    # Writes to an EXCLUDED table do not move the fingerprint...
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO ingestion_rate_buckets VALUES (8)")
    conn.commit()
    conn.close()
    assert validator._fingerprint_sqlite(db_path) == first

    # ...while a counted-table write does.
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO documents (id, title) VALUES (3, 'c')")
    conn.commit()
    conn.close()
    assert validator._fingerprint_sqlite(db_path) != first


# ---------------------------------------------------------------------------
# Live happy path: explicit opt-in, never a silent skip
# ---------------------------------------------------------------------------

_LIVE_BASE_URL = os.environ.get("RAG_PHASE10C_LIVE_BASE_URL")


@pytest.mark.skipif(
    not _LIVE_BASE_URL,
    reason="live-API lane: set RAG_PHASE10C_LIVE_BASE_URL to a running "
           "safety-enabled API (with disposable stores) to exercise the "
           "exit-0 happy path and success JSON contract",
)
def test_live_api_opt_in_happy_path_json_contract(tmp_path):
    """Opt-in: against a running safety-enabled API over a disposable
    migrated store the validator must exit 0 and print exactly the closed
    success JSON contract, leaving the store fingerprint restored."""
    from scripts import validate_phase10c as validator

    db_path = _migrated_db(tmp_path)
    env = _hermetic_env()
    env["RAG_DATABASE_URL"] = _slash_form_url(db_path)
    fingerprint_before = validator._fingerprint_sqlite(db_path)

    result = _run_script(env, tmp_path, "--base-url", _LIVE_BASE_URL)

    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert set(data) == {
        "audits", "documents_created", "findings_listed", "run_tag", "status",
    }
    assert data["status"] == "pass"
    assert data["documents_created"] >= 1
    assert data["audits"] >= 1
    assert data["run_tag"].startswith("phase10c-")
    # Self-cleaning: every fixture removed, unrelated state restored.
    assert validator._fingerprint_sqlite(db_path) == fingerprint_before
