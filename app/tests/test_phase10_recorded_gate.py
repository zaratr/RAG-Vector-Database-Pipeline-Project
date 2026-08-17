import json
from pathlib import Path
from unittest import mock

import pytest


GATE_REGISTRY = {"phase10a", "phase10b", "phase10c", "phase10d", "documentation", "final"}


def test_gate_registry_contains_exactly_six_gates():
    from scripts.run_recorded_gate import GATES

    assert set(GATES.keys()) == GATE_REGISTRY


def test_gate_steps_are_typed_argv_arrays_not_shell_strings():
    from scripts.run_recorded_gate import GATES

    for gate_id, steps in GATES.items():
        for step in steps:
            if step.get("kind") == "subprocess":
                argv = step["argv"]
                assert isinstance(argv, list), f"{gate_id} step argv must be list"
                assert all(isinstance(arg, str) for arg in argv)
                assert step.get("shell", False) is False


def test_run_recorded_gate_stops_at_first_primary_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    call_log = []

    def fake_run(argv, **kwargs):
        call_log.append(list(argv))
        if "pytest" in argv:
            return mock.Mock(returncode=1, stdout="FAILED\n", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_recorded_gate.subprocess.run", fake_run)

    from scripts.run_recorded_gate import run_gate

    run_gate(gate="phase10a", plan="plan.md", reports_dir=str(reports_dir))

    # After pytest fails, subsequent primary steps must not execute
    post_fail = False
    for argv in call_log:
        if "pytest" in argv:
            post_fail = True
        if post_fail and "validate_phase10a.py" in argv:
            pytest.fail("Primary execution continued after failure instead of stopping")


def test_run_recorded_gate_always_executes_restoration_in_finally(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    restoration_called = []

    def fake_run(argv, **kwargs):
        if "pytest" in argv:
            return mock.Mock(returncode=1, stdout="", stderr="")
        if "force-recreate" in " ".join(argv):
            restoration_called.append(list(argv))
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_recorded_gate.subprocess.run", fake_run)

    from scripts.run_recorded_gate import run_gate

    run_gate(gate="phase10a", plan="plan.md", reports_dir=str(reports_dir))

    assert restoration_called  # restoration runs even after primary failure


def test_run_recorded_gate_restoration_failure_exits_2(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    def fake_run(argv, **kwargs):
        if "force-recreate" in " ".join(argv):
            return mock.Mock(returncode=1, stdout="", stderr="restoration failed")
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_recorded_gate.subprocess.run", fake_run)

    from scripts.run_recorded_gate import run_gate

    exit_code = run_gate(gate="phase10a", plan="plan.md", reports_dir=str(reports_dir))
    assert exit_code == 2


def test_command_ledger_has_deterministic_order_after_timestamp_normalization(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.run_recorded_gate.subprocess.run",
        lambda argv, **kw: mock.Mock(returncode=0, stdout="", stderr=""),
    )

    from scripts.run_recorded_gate import run_gate

    run_gate(gate="phase10a", plan="plan.md", reports_dir=str(reports_dir))

    ledger_path = reports_dir / "phase10a-command-ledger.json"
    assert ledger_path.exists()
    ledger = json.loads(ledger_path.read_text())

    ordinals = [row["ordinal"] for row in ledger["steps"]]
    assert ordinals == sorted(ordinals)
    assert ordinals[0] == 0


def test_command_ledger_has_secret_scan_passed_on_every_row(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.run_recorded_gate.subprocess.run",
        lambda argv, **kw: mock.Mock(returncode=0, stdout="output", stderr=""),
    )

    from scripts.run_recorded_gate import run_gate

    run_gate(gate="phase10a", plan="plan.md", reports_dir=str(reports_dir))

    ledger = json.loads((reports_dir / "phase10a-command-ledger.json").read_text())
    for row in ledger["steps"]:
        assert row["secret_scan_passed"] is True


def test_command_ledger_rejects_reordered_steps(monkeypatch, tmp_path):
    """A tampered ledger with reordered ordinals must fail schema validation."""
    from jsonschema import validate as jsonschema_validate
    from jsonschema import ValidationError

    ledger_path = tmp_path / "phase10a-command-ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tampered = {
        "schema": "phase10-command-ledger-v1",
        "steps": [
            {"ordinal": 1, "kind": "subprocess", "argv": ["echo", "second"]},
            {"ordinal": 0, "kind": "subprocess", "argv": ["echo", "first"]},
        ],
    }
    ledger_path.write_text(json.dumps(tampered))

    from scripts.run_recorded_gate import validate_command_ledger

    with pytest.raises(ValidationError, match="ordinal"):
        validate_command_ledger(json.loads(ledger_path.read_text()))


def test_phase10b_registry_kinds_produce_schema_valid_ledger():
    """D-45: every step kind PRIMARY_STEPS['phase10b'] can emit must produce a
    ledger row the closed command-ledger schema accepts. Built from the real
    registry, not a patched shape."""
    from scripts.run_recorded_gate import (
        PRIMARY_STEPS,
        _record_step,
        validate_command_ledger,
    )

    ledger = []
    for ordinal, step in enumerate(PRIMARY_STEPS["phase10b"]):
        # Do not execute the step; only exercise the recording path with
        # plausible empty outputs (the recorded shape is what must validate).
        _record_step(ledger, ordinal, step, 0, "", "", phase="primary")
    payload = {"schema": "phase10-command-ledger-v1", "gate_id": "phase10b",
               "steps": ledger}
    validate_command_ledger(payload)  # raises ValidationError on any bad kind
    kinds = {row["kind"] for row in ledger}
    assert kinds <= {"subprocess", "assertion", "internal"}
    assert "internal" in kinds  # the rich non-subprocess kinds are recorded


def test_host_python_steps_resolve_to_sys_executable(monkeypatch, tmp_path):
    """D-46: host-side python registry steps must invoke sys.executable so a
    venv-resident runner cannot resolve a dependency-less interpreter."""
    import sys as _sys
    from unittest import mock

    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    seen = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.run_recorded_gate.subprocess.run", fake_run)

    from scripts.run_recorded_gate import _run_step

    _run_step(["python", "scripts/source_manifest.py", "--output", "x.json"])
    assert seen[-1][0] == _sys.executable
    assert seen[-1][1:] == ["scripts/source_manifest.py", "--output", "x.json"]
    # Non-python argv pass through untouched.
    _run_step(["docker", "compose", "config", "--quiet"])
    assert seen[-1] == ["docker", "compose", "config", "--quiet"]


def test_resolve_image_id_survives_stale_container_records(monkeypatch):
    """D-47/D-49: after a no-cache rebuild, `docker compose images -q` can fail
    because running containers reference a removed image record; the binding
    must resolve the freshly built image via the compose project/service
    labels even for build-only services (no `image` key in compose config)."""
    import json as _json
    from unittest import mock

    import scripts.create_phase10_source_binding as binding

    calls = []

    # REAL compose config shape for this project: build-only services carry
    # no `image` key (config emits None), so the fallback must go through the
    # compose project/service labels on the freshly built image.
    real_config_shape = {
        "name": "rag-vector-database-pipeline-project",
        "services": {
            "api": {
                "build": {"context": ".", "dockerfile": "Dockerfile"},
                "image": None,
            }
        },
    }

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:4] == ["docker", "compose", "images", "-q"]:
            return mock.Mock(returncode=1, stdout="", stderr="No such image: sha256:dead")
        if argv[:3] == ["docker", "compose", "config"]:
            return mock.Mock(returncode=0, stdout=_json.dumps(real_config_shape), stderr="")
        if argv[:3] == ["docker", "image", "ls"]:
            return mock.Mock(
                returncode=0,
                stdout="sha256:staleold\t2026-08-14 09:00:00\nsha256:freshlybuilt\t2026-08-15 15:18:09\n",
                stderr="",
            )
        raise AssertionError(f"unexpected argv {argv}")

    monkeypatch.setattr(binding.subprocess, "run", fake_run)
    assert binding._resolve_image_id("api") == "sha256:freshlybuilt"
    assert any(
        "--filter" in argv and "com.docker.compose.project=" in " ".join(argv)
        for argv in calls
    )

    # Happy path: current containers still win.
    def fake_run_current(argv, **kwargs):
        if argv[:4] == ["docker", "compose", "images", "-q"]:
            return mock.Mock(returncode=0, stdout="sha256:current\n", stderr="")
        raise AssertionError(f"unexpected argv {argv}")

    monkeypatch.setattr(binding.subprocess, "run", fake_run_current)
    assert binding._resolve_image_id("api") == "sha256:current"


def test_heartbeat_waits_for_readiness_and_times_out(monkeypatch):
    """D-48: the heartbeat polls until the service answers 2xx and fails only
    after the bounded attempt budget."""
    from scripts.run_recorded_gate import _heartbeat_step

    state = {"attempts": 0}

    class _Resp:
        def __init__(self, code):
            self._code = code

        def getcode(self):
            return self._code

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=5):
        state["attempts"] += 1
        if state["attempts"] < 3:
            raise ConnectionError("not listening yet")
        return _Resp(200)

    sleeps: list = []

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    rc, _, err = _heartbeat_step(
        {"url": "http://127.0.0.1:8000/", "attempts": 10, "interval_seconds": 0.01}
    )
    assert rc == 0
    assert state["attempts"] == 3
    assert len(sleeps) == 2  # backed off between failed probes

    def fake_never(url, timeout=5):
        raise ConnectionError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_never)
    rc, _, err = _heartbeat_step(
        {"url": "http://127.0.0.1:8000/", "attempts": 4, "interval_seconds": 0.01}
    )
    assert rc == 1
    assert "4 attempts" in err


def test_phase10b_restored_snapshots_compared_only_after_unscope():
    """D-50: every restored-snapshot/cmp step must execute after the scoped
    operator env is destroyed — the plan places these comparisons outside the
    scoped subshell, so the base-vs-restored equality can actually hold."""
    from scripts.run_recorded_gate import PHASE10B_RESTORATION_STEPS, PRIMARY_STEPS

    full_sequence = list(PRIMARY_STEPS["phase10b"]) + list(PHASE10B_RESTORATION_STEPS)
    unscope_positions = [
        i for i, step in enumerate(full_sequence)
        if step.get("kind") == "phase10b_unscoped_env"
    ]
    assert unscope_positions, "phase10b must destroy the scoped env"
    unscope_at = unscope_positions[0]

    restored_markers = (
        "phase10b-restored-deployment.json",
        "phase10b-restored-expected-settings.json",
        "phase10b-restored-running-settings.json",
    )
    for index, step in enumerate(full_sequence):
        argv = " ".join(step.get("argv", []))
        touches_restored = any(marker in argv for marker in restored_markers)
        compares_restored = argv.startswith("cmp ") and "restored" in argv
        if touches_restored or compares_restored:
            assert index > unscope_at, (
                f"restored snapshot/cmp at index {index} runs before unscope "
                f"at {unscope_at}"
            )

    # And the scoped env is never destroyed mid-primary.
    assert not any(
        step.get("kind") == "phase10b_unscoped_env"
        for step in PRIMARY_STEPS["phase10b"]
    )


def test_phase10c_registry_kinds_produce_schema_valid_ledger():
    """D-45: every step kind PRIMARY_STEPS['phase10c'] can emit must produce a
    ledger row the closed command-ledger schema accepts. Built from the real
    registry, not a patched shape."""
    from scripts.run_recorded_gate import (
        PRIMARY_STEPS,
        _record_step,
        validate_command_ledger,
    )

    ledger = []
    for ordinal, step in enumerate(PRIMARY_STEPS["phase10c"]):
        _record_step(ledger, ordinal, step, 0, "", "", phase="primary")
    payload = {"schema": "phase10-command-ledger-v1", "gate_id": "phase10c",
               "steps": ledger}
    validate_command_ledger(payload)  # raises ValidationError on any bad kind
    kinds = {row["kind"] for row in ledger}
    assert kinds <= {"subprocess", "assertion", "internal"}
    assert "internal" in kinds  # the rich non-subprocess kinds are recorded


def test_phase10c_restored_snapshots_compared_only_after_unscope():
    """D-50: every restored-snapshot/cmp step must execute after the scoped
    operator/content-safety env is destroyed -- the plan places these
    comparisons outside the scoped subshell, so the base-vs-restored equality
    can actually hold."""
    from scripts.run_recorded_gate import PHASE10C_RESTORATION_STEPS, PRIMARY_STEPS

    full_sequence = list(PRIMARY_STEPS["phase10c"]) + list(PHASE10C_RESTORATION_STEPS)
    unscope_positions = [
        i for i, step in enumerate(full_sequence)
        if step.get("kind") == "phase10c_unscoped_env"
    ]
    assert unscope_positions, "phase10c must destroy the scoped env"
    unscope_at = unscope_positions[0]

    restored_markers = (
        "phase10c-restored-deployment.json",
        "phase10c-restored-expected-settings.json",
        "phase10c-restored-running-settings.json",
    )
    for index, step in enumerate(full_sequence):
        argv = " ".join(step.get("argv", []))
        touches_restored = any(marker in argv for marker in restored_markers)
        compares_restored = argv.startswith("cmp ") and "restored" in argv
        if touches_restored or compares_restored:
            assert index > unscope_at, (
                f"restored snapshot/cmp at index {index} runs before unscope "
                f"at {unscope_at}"
            )

    # And the scoped env is never destroyed mid-primary.
    assert not any(
        step.get("kind") == "phase10c_unscoped_env"
        for step in PRIMARY_STEPS["phase10c"]
    )


def test_phase10c_scoped_env_sets_all_eight_keys(monkeypatch, tmp_path):
    """phase10c_scoped_env must set five RAG_*/OPERATOR_* keys plus three
    SOURCE_* keys from the manifest, with a fresh bearer token."""
    import os
    from unittest import mock

    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    # Plant a manifest with known source fields.
    manifest = {
        "commit_sha": "abc123def456",
        "image_context_sha256": "sha256:feedbeef",
        "dirty": "false",
    }
    (reports_dir / "phase10c-source-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    # Clear any pre-existing scoped keys so assertions are deterministic.
    from scripts.run_recorded_gate import _PHASE10C_SCOPED_KEYS

    for key in _PHASE10C_SCOPED_KEYS:
        os.environ.pop(key, None)

    from scripts.run_recorded_gate import _phase10c_scoped_env

    rc, out, err = _phase10c_scoped_env(reports_dir)

    assert rc == 0
    assert out == ""
    assert err == ""

    # Five operator / content-safety keys.
    assert os.environ["RAG_OPERATOR_API_ENABLED"] == "true"
    assert os.environ["RAG_OPERATOR_TOKEN"] == os.environ["OPERATOR_BEARER"]
    bearer = os.environ["OPERATOR_BEARER"]
    assert len(bearer) >= 32  # token_urlsafe(32) is well above this
    # RAG_CONTENT_SAFETY_ENABLED and RAG_SAFETY_LLM_MODE are 10C-specific.
    assert os.environ["RAG_CONTENT_SAFETY_ENABLED"] == "true"
    assert os.environ["RAG_SAFETY_LLM_MODE"] == "rules_only"

    # Three SOURCE_* keys derived from the manifest.
    assert os.environ["SOURCE_REVISION"] == "abc123def456"
    assert os.environ["SOURCE_CONTEXT_SHA256"] == "sha256:feedbeef"
    assert os.environ["SOURCE_DIRTY"] == "false"

    # Cleanup: restore process env.
    from scripts.run_recorded_gate import _phase10c_unscoped_env

    _phase10c_unscoped_env()


def test_phase10c_unscoped_env_removes_all_scoped_keys(monkeypatch, tmp_path):
    """phase10c_unscoped_env must remove exactly the eight scoped keys, leaving
    the process environment identical to the pre-scope state."""
    import os

    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    # Plant a minimal manifest so scoped_env can read it.
    (reports_dir / "phase10c-source-manifest.json").write_text(
        json.dumps({}), encoding="utf-8"
    )

    from scripts.run_recorded_gate import (
        _PHASE10C_SCOPED_KEYS,
        _phase10c_scoped_env,
        _phase10c_unscoped_env,
    )

    # Snapshot pre-existing env state for the scoped keys.
    pre_env = {key: os.environ.get(key) for key in _PHASE10C_SCOPED_KEYS}

    # Apply scoped env.
    _phase10c_scoped_env(reports_dir)
    # Verify at least some keys are now set.
    assert os.environ["RAG_OPERATOR_API_ENABLED"] == "true"
    assert os.environ["RAG_CONTENT_SAFETY_ENABLED"] == "true"

    # Unscope.
    _phase10c_unscoped_env()

    # Every scoped key must be absent (restored to pre-scope state).
    for key in _PHASE10C_SCOPED_KEYS:
        assert os.environ.get(key) == pre_env.get(key), (
            f"{key} not restored: got {os.environ.get(key)!r}, expected {pre_env.get(key)!r}"
        )
