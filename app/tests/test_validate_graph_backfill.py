"""Tests for the hardened disposable graph backfill validator (F11 remediation).

``scripts/validate_graph_backfill.py`` is the 10A.8 live validator. The F11
defect: it asserted NOTHING (always exited 0 with ``restored:true``), invoked
the service directly instead of the ``backfill_graph.py`` CLI, and took no
fingerprints. These tests prove the hardened contract:

* happy path: exit 0 with exactly the plan-pinned ``first``/``second``/
  ``restored`` JSON, produced by actually invoking the CLI;
* a lying backfill report (CLI subprocess and in-process lanes) fails
  non-zero with a machine-readable error (counters are ASSERTED, never
  merely printed);
* a configured-production fingerprint mismatch fails non-zero;
* the two-worker duplicate-protection expectation (plan 10A.8: exactly one
  worker ``processed=1/lease_lost=0``, the other ``lease_lost=1``) is proved
  deterministically (injected clock, no sleeps) with the conservation
  equations holding for both workers;
* disposable DB files are removed afterwards (self-cleaning verified).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_graph_backfill.py"
)
_CLI_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "backfill_graph.py"
)

# Plan 10A.8 pins the validator's expected JSON exactly.
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
PINNED_OUTPUT = {"first": PINNED_FIRST, "second": PINNED_SECOND, "restored": True}


# ---------------------------------------------------------------------------
# Subprocess lane: proves exit codes, sys.path bootstrap, and pinned JSON
# ---------------------------------------------------------------------------

# Startup shim loaded via PYTHONPATH inside the subprocess. It patches the
# source module at the moment it finishes importing, BEFORE any script binds
# its names, so BOTH the validator process and every CLI subprocess it spawns
# (PYTHONPATH is inherited through the validator's env) exercise a LYING
# backfill report. Dry-run reports are passed through untouched so the lie is
# proven against the pinned first real run's counters.
_SITECUSTOMIZE = textwrap.dedent(
    '''
    import builtins
    import dataclasses
    import os
    import sys

    _mode = os.environ.get("BACKFILL_TEST_MODE")
    _real_import = builtins.__import__
    _patched = set()

    def _patch(name, module):
        if name == "app.services.graph_backfill" and _mode == "wrong_counters":
            real = module.backfill

            async def lying_backfill(*args, **kwargs):
                report = await real(*args, **kwargs)
                if kwargs.get("dry_run"):
                    return report
                return dataclasses.replace(report, succeeded=report.succeeded + 1)

            module.backfill = lying_backfill

    def _patching_import(name, *args, **kwargs):
        module = _real_import(name, *args, **kwargs)
        if not _mode:
            return module
        if name in _patched:
            return module
        target = sys.modules.get(name)
        if target is None:
            return module
        if name == "app.services.graph_backfill" and not hasattr(
            target, "backfill"
        ):
            return module
        _patched.add(name)
        _patch(name, target)
        return module

    builtins.__import__ = _patching_import
    '''
)


def _run_validator_script(monkeypatch, tmp_path, mode=""):
    """Run the validator as a real subprocess with a hermetic environment.

    Production stores are made unconfigured (missing sqlite path, no Chroma
    host) so the configured-production fingerprint lane is exercised in its
    "nothing configured to protect" form; cwd is tmp_path so no .env applies.
    """
    (tmp_path / "sitecustomize.py").write_text(
        _SITECUSTOMIZE, encoding="utf-8", newline="\n"
    )
    if mode:
        monkeypatch.setenv("BACKFILL_TEST_MODE", mode)
    else:
        monkeypatch.delenv("BACKFILL_TEST_MODE", raising=False)
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{tmp_path}/absent.db")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local")
    monkeypatch.delenv("RAG_CHROMA_HOST", raising=False)
    monkeypatch.delenv("RAG_CHROMA_PERSIST_DIRECTORY", raising=False)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )


def test_validator_happy_path_exits_0_with_pinned_json(monkeypatch, tmp_path):
    """An intact backfill scenario exits 0 and prints exactly the plan-pinned
    first/second/restored JSON."""
    result = _run_validator_script(monkeypatch, tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == PINNED_OUTPUT
    assert "Traceback" not in result.stderr


def test_validator_counter_lie_exits_nonzero(monkeypatch, tmp_path):
    """A backfill report that lies about its counters must FAIL with a
    non-zero exit and a machine-readable error, never the success JSON."""
    result = _run_validator_script(monkeypatch, tmp_path, "wrong_counters")

    assert result.returncode != 0, result.stdout
    assert result.returncode == 1
    assert result.stdout.strip() == ""
    error = json.loads(result.stderr.strip().splitlines()[-1])
    assert "lane" in error
    assert "check" in error
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# In-process lane: direct access to the hardened internals
# ---------------------------------------------------------------------------


def _stable_fingerprints():
    return {"sql": None, "chroma": None}


@pytest.mark.asyncio
async def test_run_invokes_cli_asserts_lanes_and_cleans_up(monkeypatch, tmp_path):
    """One in-process run invokes the REAL backfill_graph.py CLI (never the
    service alone) for the pinned document-scoped runs, asserts the dry-run
    equations and skip reasons, and removes the disposable DB files."""
    from scripts import validate_graph_backfill as validator

    monkeypatch.setattr(
        validator, "_production_fingerprints", _stable_fingerprints
    )
    cli_argv: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(argv, *args, **kwargs):
        cli_argv.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(
        validator, "subprocess", SimpleNamespace(run=recording_run)
    )

    db_path = tmp_path / "validate-graph-backfill-test.db"
    result = await validator._run(db_path=db_path, run_id="deadbeef" * 4)

    # The CLI was really invoked (F11: the old validator called the service).
    assert cli_argv, "validator never invoked the backfill_graph.py CLI"
    cli_calls = [argv for argv in cli_argv if str(argv[1]).endswith("backfill_graph.py")]
    assert len(cli_calls) >= 6  # 4 dry-run probes + first + second real runs
    assert all("--document-id" in argv for argv in cli_calls)
    assert any("--dry-run" in argv for argv in cli_calls)

    # Pinned document-scoped runs.
    assert result["first"] == PINNED_FIRST
    assert result["second"] == PINNED_SECOND

    # Dry-run lanes: equations and skip precedence, asserted on CLI payloads.
    dry = result["dry_runs"]
    by_name = {entry["probe"]: entry["payload"] for entry in dry}
    fixture = by_name["fixture"]
    assert (fixture["scanned"], fixture["eligible"], fixture["skipped"]) == (1, 1, 0)
    assert fixture["processed"] == fixture["succeeded"] == fixture["empty"] == 0
    assert fixture["failed"] == fixture["lease_lost"] == fixture["relations"] == 0
    assert fixture["scanned"] == fixture["skipped"] + fixture["eligible"]
    mixture = by_name["mixture"]
    assert mixture["scanned"] == mixture["skipped"] + mixture["eligible"] == 2
    assert mixture["skip_reasons"] == {"unsupported_media_type": 1}
    staged = by_name["staged_image"]
    # precedence: document_not_ready beats unsupported_media_type
    assert staged["skip_reasons"] == {"document_not_ready": 1}
    failed = by_name["failed_and_terminal"]
    assert failed["scanned"] == failed["skipped"] + failed["eligible"] == 2
    # precedence: current_terminal beats failed_not_retried
    assert failed["skip_reasons"] == {
        "current_terminal": 1,
        "failed_not_retried": 1,
    }
    # Zero writes across every dry run (no provider call ever started).
    assert result["dry_run_provider_requests"] == 0

    assert result["restored"] is True

    # Self-cleaning proven: DB and sidecar files gone.
    for suffix in ("", "-wal", "-shm"):
        assert not Path(str(db_path) + suffix).exists(), suffix


@pytest.mark.asyncio
async def test_run_proves_two_worker_duplicate_protection(monkeypatch, tmp_path):
    """The validator's concurrency lane proves the plan's two-worker
    expectation deterministically: exactly one worker reports
    processed=1/succeeded=1/lease_lost=0, the other reports lease_lost=1;
    every conservation equation holds for both and exactly one terminal
    succeeded row survives."""
    from scripts import validate_graph_backfill as validator

    monkeypatch.setattr(
        validator, "_production_fingerprints", _stable_fingerprints
    )

    db_path = tmp_path / "validate-graph-backfill-two-worker.db"
    result = await validator._run(db_path=db_path, run_id="cafebabe" * 4)

    concurrency = result["concurrency"]
    reports = [concurrency["worker_a"], concurrency["worker_b"]]
    # The parking worker's provider call really started (no fake skip).
    assert concurrency["provider_calls"] == 1

    loser, winner = reports  # worker A is fenced by construction
    assert winner["processed"] == 1
    assert winner["succeeded"] == 1
    assert winner["lease_lost"] == 0
    assert winner["failed"] == 0
    assert loser["eligible"] == 1
    assert loser["processed"] == 0
    assert loser["succeeded"] == 0
    assert loser["skipped"] == 0
    assert loser["lease_lost"] == 1
    # failed is never double-counted for a fenced-out worker.
    assert loser["failed"] == 0

    for report in reports:
        assert report["eligible"] == report["processed"] + report["lease_lost"]
        assert report["processed"] == (
            report["succeeded"] + report["empty"] + report["failed"]
        )
        assert report["scanned"] == (
            report["skipped"] + report["processed"] + report["lease_lost"]
        )
        assert report["skipped"] == sum(report["skip_reasons"].values())

    # Exactly one terminal succeeded row for the contended identity.
    assert concurrency["terminal_rows"] == 1
    assert concurrency["terminal_status"] == "succeeded"

    for suffix in ("", "-wal", "-shm"):
        assert not Path(str(db_path) + suffix).exists(), suffix


@pytest.mark.asyncio
async def test_run_fails_when_report_lies(monkeypatch, tmp_path):
    """A lying backfill report must raise a machine-readable failure instead
    of being echoed as success, and cleanup must still run."""
    from scripts import validate_graph_backfill as validator
    import dataclasses

    real = validator.backfill

    async def lying_backfill(*args, **kwargs):
        report = await real(*args, **kwargs)
        if kwargs.get("dry_run"):
            return report
        return dataclasses.replace(report, succeeded=report.succeeded + 1)

    monkeypatch.setattr(validator, "backfill", lying_backfill)
    monkeypatch.setattr(
        validator, "_production_fingerprints", _stable_fingerprints
    )

    db_path = tmp_path / "validate-graph-backfill-lie.db"
    with pytest.raises(validator.ValidatorFailure) as excinfo:
        await validator._run(db_path=db_path, run_id="abcd1234" * 4)
    assert excinfo.value.lane in {"cli", "concurrency"}
    assert excinfo.value.detail["check"]
    # self-cleaning still removed the disposable DB
    assert not db_path.exists()


def test_main_fails_nonzero_when_production_fingerprint_changes(
    monkeypatch, tmp_path, capsys
):
    """Success must be withheld when the configured production fingerprints do
    not restore exactly; exit code non-zero with a machine-readable error."""
    from scripts import validate_graph_backfill as validator

    fingerprints = iter(
        [
            {"sql": {"documents": [1, 1]}, "chroma": None},
            {"sql": {"documents": [2, 1]}, "chroma": None},
        ]
    )
    monkeypatch.setattr(
        validator, "_production_fingerprints", lambda: next(fingerprints)
    )

    exit_code = validator.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.strip() == ""
    error = json.loads(captured.err.strip().splitlines()[-1])
    assert error["lane"] == "restoration"
    assert "check" in error


@pytest.mark.asyncio
async def test_run_refuses_production_db_identity(monkeypatch, tmp_path):
    """The validator must refuse a disposable DB path that equals the
    configured production database (plan: production strictly read-only)."""
    from scripts import validate_graph_backfill as validator

    prod = tmp_path / "prod.db"
    prod.write_bytes(b"")
    settings = SimpleNamespace(
        database_url=f"sqlite:///{prod}",
        chroma_host=None,
        chroma_port=8000,
        chroma_persist_directory=None,
    )
    monkeypatch.setattr(validator, "get_settings", lambda: settings)

    with pytest.raises(validator.ValidatorFailure) as excinfo:
        await validator._run(db_path=prod, run_id="a" * 32)
    assert excinfo.value.code == "configuration_error"
    # nothing was created or mutated by the refused run
    assert prod.read_bytes() == b""


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def test_sql_fingerprint_missing_database_returns_none(tmp_path):
    from scripts import validate_graph_backfill as validator

    assert validator._sql_fingerprint(f"sqlite:///{tmp_path}/missing.db") is None


def test_sql_fingerprint_reads_counts_and_max_ids_read_only(tmp_path):
    import sqlite3

    from scripts import validate_graph_backfill as validator

    db_path = tmp_path / "prod-like.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT);"
        "CREATE TABLE alembic_version (version_num TEXT);"
        "INSERT INTO documents (id, title) VALUES (5, 'a'), (9, 'b');"
        "INSERT INTO alembic_version VALUES ('x');"
    )
    conn.commit()
    conn.close()
    before_stat = db_path.stat()

    fingerprint = validator._sql_fingerprint(f"sqlite:///{db_path}")

    assert fingerprint == {"documents": [2, 9]}
    assert "alembic_version" not in fingerprint
    after_stat = db_path.stat()
    assert (before_stat.st_mtime_ns, before_stat.st_size) == (
        after_stat.st_mtime_ns,
        after_stat.st_size,
    )


def test_chroma_fingerprint_without_configured_host_returns_none():
    from scripts import validate_graph_backfill as validator

    assert validator._chroma_fingerprint(None, 8000, None) is None
