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
    reports_dir = tmp_path / "reports"
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
    reports_dir = tmp_path / "reports"
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
    reports_dir = tmp_path / "reports"
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
    reports_dir = tmp_path / "reports"
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
    reports_dir = tmp_path / "reports"
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
