"""Phase 10D.4 — named-volume durability tests.

Covers: sentinel creation, force-recreate without ``-v``, migration
wrapper idempotency, fingerprint equality, cleanup.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_stores(monkeypatch, tmp_path):
    """Run the in-process durability checks against isolated stores.

    The pinned tests exercise the REAL setup/verify/cleanup helpers; on
    the production /data volume a mocked-away cleanup would leak real
    sentinels (D-81). Every in-process test therefore gets a migrated
    disposable SQLite database and an ephemeral Chroma client. The
    real-subprocess argv test is unaffected (it runs the script in a
    fresh interpreter against the real deployment and cleans up after
    itself).
    """
    import scripts.validate_named_volume_durability as mod
    from app.services import vector_store

    db = tmp_path / "durability-isolated.db"
    db.write_bytes(b"")
    env = dict(os.environ)
    env["RAG_DATABASE_URL"] = f"sqlite:///{db}"
    subprocess.run([sys.executable, "-m", "app.core.migrations"],
                   env=env, check=True, capture_output=True)
    monkeypatch.setattr(mod, "_database_path", lambda: db)

    import chromadb

    monkeypatch.setattr(vector_store, "_create_client",
                        lambda: chromadb.EphemeralClient())
    from app.services import attack_simulator

    monkeypatch.setattr(attack_simulator, "_chroma_client",
                        lambda: chromadb.EphemeralClient())


_API_HEARTBEAT_URL = "http://127.0.0.1:8000/"


def _fake_heartbeats(monkeypatch, healthy=True):
    """Replace the module heartbeat seam with a recording fake.

    Structurally guarantees the in-process lanes never touch the network
    (D: heartbeat stage previously depended on the ambient Docker stack)
    and records the probed URLs so the seam cannot silently drift.
    """
    import scripts.validate_named_volume_durability as mod

    calls: list = []

    def fake(url, attempts=30, interval=2.0):
        calls.append(url)
        return healthy

    monkeypatch.setattr(mod, "_wait_heartbeat", fake)
    return calls


def _assert_heartbeat_urls(calls):
    import scripts.validate_named_volume_durability as mod

    assert calls == [_API_HEARTBEAT_URL, mod._chroma_heartbeat_url()]


def _run_validator(monkeypatch, out, calls_out=None):
    """Run the durability check with docker children mocked (no -v ever)
    and the heartbeat health stage faked healthy (no network)."""
    from scripts.validate_named_volume_durability import run_durability_check

    def fake_run(args, **kw):
        if args[0:3] == ["docker", "compose", "up"]:
            assert "-d" in args and "--force-recreate" in args
            assert "-v" not in args  # named volumes must NEVER be removed
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    calls = _fake_heartbeats(monkeypatch, healthy=True)
    try:
        return run_durability_check(output=out)
    finally:
        if calls_out is not None:
            calls_out.extend(calls)


def test_sentinel_survives_force_recreate_without_volume_flag(monkeypatch, tmp_path):
    # 1. create UUID SQL parent/child + Chroma sentinel
    # 2. record IDs/hashes/head
    # 3. docker compose up -d --force-recreate (NO -v)
    # 4. run production migration wrapper
    # 5. assert identical SQL rows/FKs/vector IDs/head
    # 6. assert API + Chroma heartbeat healthy
    from scripts.validate_named_volume_durability import run_durability_check
    recompose = []

    def fake_run(args, **kw):
        recompose.append(args)
        if args[0:3] == ["docker", "compose", "up"]:
            assert "-d" in args and "--force-recreate" in args
            assert "-v" not in args  # named volumes must NEVER be removed
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    calls = _fake_heartbeats(monkeypatch, healthy=True)
    out = tmp_path / "durability.json"
    rc = run_durability_check(output=out)
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["before"]["sql_rows"] == rec["after"]["sql_rows"]
    assert rec["before"]["fk_rows"] == rec["after"]["fk_rows"]
    assert rec["before"]["vector_hash"] == rec["after"]["vector_hash"]
    assert rec["before"]["alembic_head"] == rec["after"]["alembic_head"]
    assert rec["api_healthy"] is True and rec["chroma_healthy"] is True
    assert any(a[0:3] == ["docker", "compose", "up"] for a in recompose)
    _assert_heartbeat_urls(calls)


def test_unrelated_fingerprints_unchanged(monkeypatch, tmp_path):
    # The durability validator must fingerprint unrelated state before/after
    # and assert equality.
    out = tmp_path / "durability.json"
    calls: list = []
    rc = _run_validator(monkeypatch, out, calls_out=calls)
    assert rc == 0
    rec = json.loads(out.read_text())
    # non-sentinel production state is byte-identical before vs after the run
    assert rec["before"]["unrelated_fingerprint"] == rec["after"]["unrelated_fingerprint"]
    _assert_heartbeat_urls(calls)


def test_cleanup_deletes_only_sentinels_in_finally(monkeypatch, tmp_path):
    # After the run, only the created sentinel rows/IDs are deleted; unrelated
    # state is untouched.
    out = tmp_path / "durability.json"
    calls: list = []
    rc = _run_validator(monkeypatch, out, calls_out=calls)
    assert rc == 0
    rec = json.loads(out.read_text())
    # cleanup ran in the finally block and removed exactly the created sentinels
    assert rec["cleanup_complete"] is True
    created = rec["sentinels"]
    assert set(rec["deleted_sentinel_ids"]) == {
        created["sql_parent_id"], created["sql_child_id"],
        created["chroma_sentinel_id"]}
    # unrelated production state is byte-identical before vs after
    assert rec["before"]["unrelated_fingerprint"] == rec["after"]["unrelated_fingerprint"]
    _assert_heartbeat_urls(calls)


def test_unhealthy_heartbeat_exits_two_without_network(monkeypatch, tmp_path):
    # When the health stage reports unhealthy (fake; no network), the script
    # must exit 2, record api/chroma_healthy False at the verify stage, and
    # still run sentinel cleanup to completion.
    import scripts.validate_named_volume_durability as mod

    monkeypatch.setattr(subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    calls = _fake_heartbeats(monkeypatch, healthy=False)
    out = tmp_path / "durability.json"
    rc = mod.run_durability_check(output=out)
    assert rc == 2
    rec = json.loads(out.read_text())
    assert rec["ok"] is False
    assert rec["stage"] == "verify"
    assert rec["api_healthy"] is False and rec["chroma_healthy"] is False
    assert rec["cleanup_complete"] is True
    _assert_heartbeat_urls(calls)


def test_setup_mismatch_exits_two(monkeypatch, tmp_path):
    # If sentinel creation fails, the script exits 2 and emits canonical JSON.
    import scripts.validate_named_volume_durability as mod

    def _boom(*args, **kwargs):
        raise RuntimeError("sentinel creation failed")

    monkeypatch.setattr(mod, "_setup_sentinels", _boom)
    monkeypatch.setattr(subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    out = tmp_path / "durability.json"
    rc = mod.run_durability_check(output=out)
    assert rc == 2
    rec = json.loads(out.read_text())
    assert rec["ok"] is False and rec["stage"] == "setup"


def test_verify_mismatch_exits_two(monkeypatch, tmp_path):
    # If post-recreate verification fails (e.g. missing row), exit 2.
    import scripts.validate_named_volume_durability as mod

    def _boom(*args, **kwargs):
        raise RuntimeError("sentinel row missing after recreate")

    monkeypatch.setattr(mod, "_verify_state", _boom)
    monkeypatch.setattr(subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    out = tmp_path / "durability.json"
    rc = mod.run_durability_check(output=out)
    assert rc == 2
    rec = json.loads(out.read_text())
    assert rec["ok"] is False and rec["stage"] == "verify"


def test_cleanup_mismatch_exits_two(monkeypatch, tmp_path):
    # If cleanup cannot delete a sentinel, exit 2.
    import scripts.validate_named_volume_durability as mod

    def _boom(*args, **kwargs):
        raise RuntimeError("sentinel row could not be deleted")

    monkeypatch.setattr(mod, "_cleanup_sentinels", _boom)
    _fake_heartbeats(monkeypatch, healthy=True)  # reaches the health stage
    monkeypatch.setattr(subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    out = tmp_path / "durability.json"
    rc = mod.run_durability_check(output=out)
    assert rc == 2
    rec = json.loads(out.read_text())
    assert rec["ok"] is False and rec["stage"] == "cleanup"


def test_output_is_canonical_non_sensitive_json(monkeypatch, tmp_path):
    out = tmp_path / "durability.json"
    _run_validator(monkeypatch, out)
    obj = json.loads(out.read_text())
    # No text/secrets/credentials in output
    blob = json.dumps(obj)
    for forbidden in ["token", "secret", "password", "credential"]:
        assert forbidden not in blob.lower()


@pytest.mark.skipif(
    not os.environ.get("RAG_LIVE_DURABILITY"),
    reason="opt-in live lane: runs scripts/validate_named_volume_durability.py "
    "as a real subprocess against a live Docker Compose deployment "
    "(docker compose up --force-recreate + exec into the running api); set "
    "RAG_LIVE_DURABILITY=1 with the deployment up to execute",
)
def test_validate_named_volume_durability_argv_exact(monkeypatch, tmp_path):
    argv = [sys.executable, "scripts/validate_named_volume_durability.py",
            "--output", str(tmp_path / "durability.json")]
    result = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0
