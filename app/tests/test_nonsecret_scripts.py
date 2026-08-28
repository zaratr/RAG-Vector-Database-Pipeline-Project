"""Phase 10B plan-body script tests: non-secret snapshots and the production
state fingerprint (plan L1116/L1118 deliverables).

``scripts/snapshot_nonsecret_deployment.py`` and
``scripts/snapshot_nonsecret_settings.py`` are exercised as real subprocesses
over disposable output trees with sentinel secrets injected into the
environment; ``scripts/fingerprint_production_state.py`` is exercised against
a disposable migrated database. The assertions pin exit codes, canonical LF
bytes, determinism where promised, and the absence of every secret marker.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DEPLOYMENT = PROJECT_ROOT / "scripts" / "snapshot_nonsecret_deployment.py"
SNAPSHOT_SETTINGS = PROJECT_ROOT / "scripts" / "snapshot_nonsecret_settings.py"
FINGERPRINT = PROJECT_ROOT / "scripts" / "fingerprint_production_state.py"

SENTINEL_TOKEN = "SENTINEL-operator-token-do-not-emit-0123456789"
SENTINEL_API_KEY = "SENTINEL-sk-api-key-do-not-emit"
SENTINEL_DB_URL = "sqlite:///SENTINEL-db-with-secret-do-not-emit.db"
SECRET_MARKERS = (SENTINEL_TOKEN, SENTINEL_API_KEY, "SENTINEL-db-with-secret")

FORBIDDEN_KEY = "(?i)(token|secret|password|credential|api[_-]?key|database.*url)"


def _script_env(tmp_path: Path, **extra: str) -> dict:
    """Clean subprocess environment: project importable via PYTHONPATH, cwd
    outside the repository (no .env pickup), sentinel secrets injected."""
    import re

    env = {"PYTHONPATH": str(PROJECT_ROOT), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    for key, value in os.environ.items():
        if key.upper().startswith(("PYTHON", "PATH", "SYSTEMROOT", "TEMP", "TMP")) \
                and key not in env:
            env[key] = value
    if "PATH" not in env:
        env["PATH"] = os.environ.get("PATH", "")
    env.update({
        "RAG_OPERATOR_TOKEN": SENTINEL_TOKEN,
        "OPENAI_API_KEY": SENTINEL_API_KEY,
        "RAG_DATABASE_URL": SENTINEL_DB_URL,
        "RAG_SECURITY_AUDIT_RETENTION_DAYS": "45",
        **extra,
    })
    return env


def _run(script: Path, *args: str, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, check=False, cwd=str(cwd), env=env,
    )


# ---------------------------------------------------------------------------
# snapshot_nonsecret_settings.py (running-settings projection, stdout lane).

def test_snapshot_settings_stdout_is_canonical_and_secret_free(tmp_path):
    result = _run(SNAPSHOT_SETTINGS, cwd=tmp_path, env=_script_env(tmp_path))
    assert result.returncode == 0, result.stderr
    raw = result.stdout
    assert raw.endswith("\n") and "\r" not in raw
    for marker in SECRET_MARKERS:
        assert marker not in raw
    payload = json.loads(raw)
    assert list(payload.keys()) == sorted(payload.keys())
    # Forbidden keys never appear; allowlisted keys do.
    import re

    assert not any(re.search(FORBIDDEN_KEY, key) for key in payload)
    assert payload["security_audit_retention_days"] == 45
    assert payload["operator_api_enabled"] is False
    # Sorted-key minified canonical form reproduces the bytes exactly.
    assert raw == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def test_snapshot_settings_is_deterministic(tmp_path):
    env = _script_env(tmp_path)
    first = _run(SNAPSHOT_SETTINGS, cwd=tmp_path, env=env)
    second = _run(SNAPSHOT_SETTINGS, cwd=tmp_path, env=env)
    assert first.returncode == 0 and second.returncode == 0
    assert first.stdout == second.stdout


def test_snapshot_settings_file_output_matches_stdout(tmp_path):
    out = tmp_path / "nested" / "settings.json"
    result = _run(SNAPSHOT_SETTINGS, "--settings-output", str(out),
                  cwd=tmp_path, env=_script_env(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    written = out.read_bytes()
    assert written.endswith(b"\n") and b"\r" not in written
    stdout_lane = _run(SNAPSHOT_SETTINGS, cwd=tmp_path, env=_script_env(tmp_path))
    assert written.decode("utf-8") == stdout_lane.stdout
    # No partial temp file remains next to the atomic write.
    assert list(out.parent.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# snapshot_nonsecret_deployment.py.

def test_snapshot_deployment_settings_lane_is_secret_free_and_atomic(tmp_path):
    out = tmp_path / "settings-file.json"
    result = _run(SNAPSHOT_DEPLOYMENT, "--settings-output", str(out),
                  cwd=tmp_path, env=_script_env(tmp_path))
    assert result.returncode == 0, result.stderr
    raw = out.read_text(encoding="utf-8")
    for marker in SECRET_MARKERS:
        assert marker not in raw
    payload = json.loads(raw)
    assert list(payload.keys()) == sorted(payload.keys())
    import re

    assert not any(re.search(FORBIDDEN_KEY, key) for key in payload)


def test_snapshot_deployment_filters_secret_environment_keys(tmp_path, monkeypatch):
    """The compose-config filter strips every forbidden service environment
    key while preserving allowlisted ones, canonically and deterministically
    (docker boundary stubbed; filtering/canonicalization logic is real)."""
    import scripts.snapshot_nonsecret_deployment as snap

    compose_config = {
        "services": {
            "api": {
                "environment": {
                    "RAG_DATABASE_URL": "sqlite:////data/rag.db",
                    "RAG_OPERATOR_TOKEN": SENTINEL_TOKEN,
                    "RAG_OPENAI_API_KEY": SENTINEL_API_KEY,
                    "RAG_OPERATOR_API_ENABLED": "false",
                },
            },
            "migrate": {
                "environment": {"RAG_DATABASE_URL": "sqlite:////data/rag.db"},
            },
        }
    }

    class _Result:
        returncode = 0
        stdout = json.dumps(compose_config)
        stderr = ""

    def _fake_run(*args, **kwargs):
        return _Result()

    monkeypatch.setattr(snap.subprocess, "run", _fake_run)
    out = tmp_path / "deployment.json"
    payload = snap.snapshot_deployment(str(out))

    raw = out.read_text(encoding="utf-8")
    for marker in (SENTINEL_TOKEN, SENTINEL_API_KEY):
        assert marker not in raw
    assert set(payload["services"]["api"]["environment"]) == {
        "RAG_OPERATOR_API_ENABLED"}
    # Database URLs are excluded from the deployment snapshot as well (the
    # plan's allowlist excludes every database URL), so migrate — whose only
    # environment key is RAG_DATABASE_URL — projects an empty environment.
    assert payload["services"]["migrate"]["environment"] == {}
    assert raw == json.dumps(payload, sort_keys=True,
                             separators=(",", ":")) + "\n"
    # Deterministic: a second snapshot is byte-identical.
    out2 = tmp_path / "deployment-2.json"
    snap.snapshot_deployment(str(out2))
    assert out2.read_bytes() == out.read_bytes()


def test_snapshot_deployment_refuses_malformed_compose_output(tmp_path, monkeypatch):
    import scripts.snapshot_nonsecret_deployment as snap

    class _Result:
        returncode = 0
        stdout = "{not-json"
        stderr = ""

    monkeypatch.setattr(snap.subprocess, "run", lambda *a, **k: _Result())
    with pytest.raises(ValueError):
        snap.snapshot_deployment(str(tmp_path / "never.json"))
    assert not (tmp_path / "never.json").exists()


def test_snapshot_deployment_cli_exits_two_on_malformed_compose(tmp_path):
    """The CLI contract (plan §10B.2): failure exits 2 with no partial file.

    Docker is absent from the hermetic environment, so `docker compose
    config` itself fails — the same failure class the CLI must map to 2."""
    out = tmp_path / "never-written.json"
    result = _run(SNAPSHOT_DEPLOYMENT, "--deployment-output", str(out),
                  cwd=tmp_path, env=_script_env(tmp_path))
    assert result.returncode == 2
    assert result.stdout == ""
    assert not out.exists()


def test_snapshot_deployment_surfaces_docker_failure(tmp_path, monkeypatch):
    import scripts.snapshot_nonsecret_deployment as snap

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "compose exploded"

    monkeypatch.setattr(snap.subprocess, "run", lambda *a, **k: _Result())
    with pytest.raises(RuntimeError):
        snap.snapshot_deployment(str(tmp_path / "never.json"))


# ---------------------------------------------------------------------------
# fingerprint_production_state.py.

def _fingerprint_db(tmp_path: Path) -> Path:
    from sqlalchemy import create_engine, text

    from app.core.migrations import upgrade_database

    db_path = tmp_path / "fingerprint.db"
    db_url = f"sqlite:///{db_path}"
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO documents (id, title, source, tags, ingestion_status, "
            "trust_tier, trust_score, trust_policy_version, ingestion_origin) "
            "VALUES (1, 'SENTRY-TITLE-never-emit', 'wiki', NULL, 'ready', "
            "'untrusted', 0.2, 'source-trust-v1', 'api')"))
        conn.execute(text(
            'INSERT INTO chunks (id, document_id, "index", text, start_offset, '
            "end_offset, vector_id, media_type) VALUES (1, 1, 0, "
            "'SENTRY-CHUNK-TEXT-never-emit', 0, 5, 'vec-1', 'text/plain')"))
    engine.dispose()
    return db_path


def test_fingerprint_reports_heads_counts_and_pks_without_content(tmp_path):
    db_path = _fingerprint_db(tmp_path)
    env = _script_env(tmp_path, RAG_DATABASE_URL=f"sqlite:///{db_path}")
    env.pop("RAG_CHROMA_HOST", None)
    result = _run(FINGERPRINT, "--json", cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    raw = result.stdout
    assert raw.endswith("\n") and "\r" not in raw
    # No document/chunk content or secret leaks into the fingerprint.
    for marker in (*SECRET_MARKERS, "SENTRY-TITLE-never-emit",
                   "SENTRY-CHUNK-TEXT-never-emit"):
        assert marker not in raw
    payload = json.loads(raw)
    assert payload["alembic_head"] == "d9b5f7c1e4a3"
    assert payload["chroma_ids"] == []
    assert payload["tables"]["documents"]["row_count"] == 1
    assert payload["tables"]["documents"]["primary_keys"] == [[1]]
    assert payload["tables"]["chunks"]["row_count"] == 1
    assert "text" not in json.dumps(payload["tables"]["chunks"])


def test_fingerprint_is_deterministic_and_read_only(tmp_path):
    db_path = _fingerprint_db(tmp_path)
    env = _script_env(tmp_path, RAG_DATABASE_URL=f"sqlite:///{db_path}")
    env.pop("RAG_CHROMA_HOST", None)
    before_bytes = db_path.read_bytes()
    first = _run(FINGERPRINT, "--json", cwd=tmp_path, env=env)
    second = _run(FINGERPRINT, "--json", cwd=tmp_path, env=env)
    assert first.returncode == 0 and second.returncode == 0
    assert first.stdout == second.stdout
    assert db_path.read_bytes() == before_bytes


def test_fingerprint_exits_two_on_inventory_failure(tmp_path):
    env = _script_env(
        tmp_path,
        RAG_DATABASE_URL=f"sqlite:///{tmp_path / 'no-such-dir' / 'missing.db'}",
    )
    env.pop("RAG_CHROMA_HOST", None)
    result = _run(FINGERPRINT, "--json", cwd=tmp_path, env=env)
    assert result.returncode == 2
    assert result.stdout == ""
