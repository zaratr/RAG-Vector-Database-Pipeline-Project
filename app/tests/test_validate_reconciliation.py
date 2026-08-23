"""Tests for the hardened disposable reconciliation validator (F10 remediation).

``scripts/validate_reconciliation.py`` is the 10A.4 live-idempotence validator.
The F10 defect: it defined ``_fingerprint`` but never called it (no production
SQL/Chroma fingerprints despite its own docstring) and used an in-memory
vector-store substitute instead of a disposable Chroma collection. These tests
prove the hardened contract:

* happy path: exit 0 with exactly the plan-pinned converged second-run JSON;
* a seeded reconciliation counter lie fails non-zero with a machine-readable
  error (counters are ASSERTED, never merely printed);
* a configured-production fingerprint mismatch fails non-zero;
* the crash-matrix proofs run against a REAL disposable Chroma collection
  (delete/upsert genuinely executed through ``ChromaVectorStore``) and a
  disposable migrated SQLite DB, and both are removed afterwards
  (self-cleaning verified, not assumed).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_reconciliation.py"
)

# Plan L619-623 pins the converged second-run JSON exactly.
PINNED_SECOND = {
    "nonready_vectors_deleted": 0,
    "orphan_vectors_deleted": 0,
    "pending_extractions_failed": 0,
    "ready_chunks_upserted": 3,
    "staged_documents_failed": 0,
}
# Crash-matrix row "before staged commit": no documents at all -> exact no-op.
EXPECTED_NOOP = {
    "nonready_vectors_deleted": 0,
    "orphan_vectors_deleted": 0,
    "pending_extractions_failed": 0,
    "ready_chunks_upserted": 0,
    "staged_documents_failed": 0,
}
# Seeded matrix: 3 staged documents (pending-extraction, terminal-evidence,
# partial-vector-write), 1 nonready vector present, 1 orphan vector, and the
# 3 self-created ready fixtures (plan: ready_chunks_upserted=3 invariant).
EXPECTED_FIRST = {
    "nonready_vectors_deleted": 1,
    "orphan_vectors_deleted": 1,
    "pending_extractions_failed": 1,
    "ready_chunks_upserted": 3,
    "staged_documents_failed": 3,
}


# ---------------------------------------------------------------------------
# Subprocess lane: proves exit codes, sys.path bootstrap, and pinned JSON
# ---------------------------------------------------------------------------

# Startup shim loaded via PYTHONPATH inside the subprocess. It patches the
# source module at the moment it finishes importing, BEFORE the script binds
# its names, so the validator exercises a LYING production reconciliation
# report and must catch it via its own counter assertions.
_SITECUSTOMIZE = textwrap.dedent(
    '''
    import builtins
    import os
    import sys

    _mode = os.environ.get("RECON_TEST_MODE")
    _real_import = builtins.__import__
    _patched = set()

    def _patch(name, module):
        if name == "app.services.reconciliation" and _mode == "wrong_counters":
            real = module.reconcile_ingestion

            async def lying_reconcile(**kwargs):
                report = dict(await real(**kwargs))
                report["ready_chunks_upserted"] += 1
                return report

            module.reconcile_ingestion = lying_reconcile

    def _patching_import(name, *args, **kwargs):
        module = _real_import(name, *args, **kwargs)
        if not _mode:
            return module
        if name in _patched:
            return module
        target = sys.modules.get(name)
        if target is None:
            return module
        if name == "app.services.reconciliation" and not hasattr(
            target, "reconcile_ingestion"
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
    (tmp_path / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8", newline="\n")
    if mode:
        monkeypatch.setenv("RECON_TEST_MODE", mode)
    else:
        monkeypatch.delenv("RECON_TEST_MODE", raising=False)
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
    """An intact reconciliation run exits 0 and prints exactly the plan-pinned
    converged second-run JSON."""
    result = _run_validator_script(monkeypatch, tmp_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == PINNED_SECOND
    assert "Traceback" not in result.stderr


def test_validator_seeded_counter_lie_exits_nonzero(monkeypatch, tmp_path):
    """A reconciliation report that lies about its counters must FAIL with a
    non-zero exit and a machine-readable error, never the success JSON."""
    result = _run_validator_script(monkeypatch, tmp_path, "wrong_counters")

    assert result.returncode != 0, result.stdout
    assert result.returncode == 1
    assert result.stdout.strip() == ""
    error = json.loads(result.stderr.strip().splitlines()[-1])
    assert error["lane"] == "matrix"
    assert "check" in error
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# In-process lane: direct access to the hardened internals
# ---------------------------------------------------------------------------


def _stable_fingerprints():
    return {"sql": None, "chroma": None}


@pytest.mark.asyncio
async def test_run_proves_matrix_counters_and_cleans_up(monkeypatch, tmp_path):
    """One in-process run returns the exact no-op/first/second counter proofs,
    executes real Chroma deletions/upserts through ``ChromaVectorStore``, and
    removes the disposable DB files and collection."""
    from scripts import validate_reconciliation as validator

    monkeypatch.setattr(
        validator, "_production_fingerprints", _stable_fingerprints
    )
    spy_delete: list[list[str]] = []
    spy_upsert: list[list[str]] = []
    real_delete = validator.ChromaVectorStore.delete
    real_upsert = validator.ChromaVectorStore.upsert_embeddings

    async def recording_delete(self, ids):
        spy_delete.append(list(ids))
        return await real_delete(self, ids)

    async def recording_upsert(self, embeddings, metadatas, ids, documents=None):
        spy_upsert.append(list(ids))
        return await real_upsert(
            self, embeddings, metadatas, ids, documents=documents
        )

    monkeypatch.setattr(validator.ChromaVectorStore, "delete", recording_delete)
    monkeypatch.setattr(
        validator.ChromaVectorStore, "upsert_embeddings", recording_upsert
    )

    run_id = "deadbeef" * 4
    db_path = tmp_path / "validate-reconciliation-test.db"
    result = await validator._run(db_path=db_path, run_id=run_id)

    assert result["noop"] == EXPECTED_NOOP
    assert result["first"] == EXPECTED_FIRST
    assert result["second"] == PINNED_SECOND
    assert result["restored"] is True

    # Real Chroma vector paths were exercised: the seed upsert (3 ready + 1
    # partial-write + 1 orphan), the first-run deletion of exactly the 2
    # stale IDs, the convergence upserts of the 3 ready IDs on each run, and
    # empty deletions on the no-op probe and the converged second run.
    assert [len(ids) for ids in spy_upsert] == [5, 3, 3]
    assert set(spy_upsert[1]) == set(spy_upsert[2])
    assert len(set(spy_upsert[1])) == 3
    assert len(spy_delete) == 3
    assert spy_delete[0] == []
    assert len(spy_delete[1]) == 2
    assert set(spy_delete[1]).isdisjoint(set(spy_upsert[1]))
    assert spy_delete[2] == []

    # Self-cleaning proven: DB and sidecar files gone, collection gone.
    for suffix in ("", "-wal", "-shm"):
        assert not Path(str(db_path) + suffix).exists(), suffix
    import chromadb

    names = [
        getattr(c, "name", c)
        for c in chromadb.EphemeralClient().list_collections()
    ]
    assert f"validate-reconciliation-{run_id}" not in names


@pytest.mark.asyncio
async def test_run_fails_when_reconciliation_report_lies(monkeypatch, tmp_path):
    """A lying reconciliation report must raise a machine-readable failure
    instead of being echoed as success, and cleanup must still run."""
    from scripts import validate_reconciliation as validator

    real = validator.reconcile_ingestion

    async def lying_reconcile(**kwargs):
        report = dict(await real(**kwargs))
        report["staged_documents_failed"] += 1
        return report

    monkeypatch.setattr(validator, "reconcile_ingestion", lying_reconcile)
    monkeypatch.setattr(
        validator, "_production_fingerprints", _stable_fingerprints
    )

    db_path = tmp_path / "validate-reconciliation-lie.db"
    with pytest.raises(validator.ValidatorFailure) as excinfo:
        await validator._run(db_path=db_path, run_id="cafebabe" * 4)
    assert excinfo.value.lane == "matrix"
    assert excinfo.value.detail["check"]
    # self-cleaning still removed the disposable DB
    assert not db_path.exists()


def test_main_fails_nonzero_when_production_fingerprint_changes(
    monkeypatch, tmp_path, capsys
):
    """Success must be withheld when the configured production fingerprints do
    not restore exactly; exit code non-zero with a machine-readable error."""
    from scripts import validate_reconciliation as validator

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
    configured production database (plan: refuses production identity)."""
    from scripts import validate_reconciliation as validator

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
    from scripts import validate_reconciliation as validator

    assert validator._sql_fingerprint(f"sqlite:///{tmp_path}/missing.db") is None


def test_sql_fingerprint_reads_counts_and_max_ids_read_only(tmp_path):
    from scripts import validate_reconciliation as validator

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
    # read-only: file mtime/size unchanged
    after_stat = db_path.stat()
    assert (before_stat.st_mtime_ns, before_stat.st_size) == (
        after_stat.st_mtime_ns,
        after_stat.st_size,
    )


def test_chroma_fingerprint_without_configured_host_returns_none():
    from scripts import validate_reconciliation as validator

    assert validator._chroma_fingerprint(None, 8000, None) is None
