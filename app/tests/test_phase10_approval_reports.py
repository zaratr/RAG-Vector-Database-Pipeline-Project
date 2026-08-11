import json
from pathlib import Path

import pytest


def test_review_input_schema_rejects_actionable_warn_with_approved():
    from jsonschema import validate as jsonschema_validate
    from jsonschema import ValidationError

    review_input = {
        "gate_id": "phase10a",
        "owner": "test",
        "terminal_verdict": "APPROVED",
        "findings": [
            {"severity": "WARN", "actionable": True, "message": "must fix this"}
        ],
        "utc_timestamp": "2026-08-09T00:00:00Z",
        "markdown_path": "draft.md",
        "markdown_sha256": "abc123",
    }

    schema_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "phase10-gate-review-input.schema.json"
    )
    schema = json.loads(schema_path.read_text())

    with pytest.raises(ValidationError, match="actionable"):
        jsonschema_validate(review_input, schema)


def test_write_gate_report_refuses_existing_approved_pair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    approvals_dir = tmp_path / ".hermes" / "reports" / "approvals"
    approvals_dir.mkdir(parents=True)
    (approvals_dir / "phase10a.json").write_text(
        json.dumps({"terminal_verdict": "APPROVED"}), encoding="utf-8"
    )
    (approvals_dir / "phase10a.md").write_text("# APPROVED\n", encoding="utf-8")

    from scripts.write_phase10_gate_report import write_report

    with pytest.raises((RuntimeError, ValueError), match="immutable"):
        write_report(gate="phase10a", plan="plan.md")


def test_write_gate_report_archives_existing_not_approved_pair_before_retry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    approvals_dir = tmp_path / ".hermes" / "reports" / "approvals"
    approvals_dir.mkdir(parents=True)
    (approvals_dir / "phase10a.json").write_text(
        json.dumps({"terminal_verdict": "NOT APPROVED"}), encoding="utf-8"
    )
    (approvals_dir / "phase10a.md").write_text("# NOT APPROVED\n", encoding="utf-8")

    from scripts.write_phase10_gate_report import write_report

    write_report(gate="phase10a", plan="plan.md")

    archive_dir = tmp_path / ".hermes" / "reports" / "approval-attempts" / "phase10a"
    assert archive_dir.exists()
    archived_files = list(archive_dir.iterdir())
    assert len(archived_files) >= 3  # report.json, report.md, source-manifest.json


def test_write_gate_report_validates_command_ledger_completeness(tmp_path, monkeypatch):
    """A command ledger with missing steps must fail writer validation."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hermes" / "reports").mkdir(parents=True)

    from scripts.write_phase10_gate_report import validate_command_ledger_for_gate

    incomplete_ledger = {"steps": [{"ordinal": 0, "kind": "subprocess", "argv": ["echo"]}]}
    with pytest.raises((ValueError, RuntimeError), match="incomplete|missing"):
        validate_command_ledger_for_gate(incomplete_ledger, gate="phase10a")


def test_write_gate_report_validates_source_manifest_hash_matches_build_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / ".hermes" / "reports"
    reports_dir.mkdir(parents=True)

    build_manifest = reports_dir / "phase10a-source-manifest.json"
    build_manifest.write_text(json.dumps({"delivery_tree_sha256": "aaa"}), encoding="utf-8")

    from scripts.write_phase10_gate_report import verify_manifest_match

    with pytest.raises((ValueError, RuntimeError), match="manifest.*mismatch"):
        verify_manifest_match(
            gate="phase10a",
            build_manifest_path=str(build_manifest),
            approval_manifest_hash="bbb",
        )


def test_verify_phase10_approvals_validates_exact_required_gate_set(tmp_path, monkeypatch):
    """Missing any required gate must fail verification."""
    monkeypatch.chdir(tmp_path)
    approvals_dir = tmp_path / ".hermes" / "reports" / "approvals"
    approvals_dir.mkdir(parents=True)
    # Only create one gate, missing the rest
    (approvals_dir / "phase10a.json").write_text(
        json.dumps({"gate_id": "phase10a", "terminal_verdict": "APPROVED"}),
        encoding="utf-8",
    )

    from scripts.verify_phase10_approvals import verify_approvals

    result = verify_approvals(approvals_dir=str(approvals_dir))

    assert result["valid"] is False
    assert "phase10b" in result["missing_gates"]


def test_verify_phase10_approvals_rejects_any_fail_finding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    approvals_dir = tmp_path / ".hermes" / "reports" / "approvals"
    approvals_dir.mkdir(parents=True)
    for gate in ["phase10a", "phase10b", "phase10c", "phase10d", "documentation", "final"]:
        verdict = "NOT APPROVED" if gate == "phase10a" else "APPROVED"
        (approvals_dir / f"{gate}.json").write_text(
            json.dumps({"gate_id": gate, "terminal_verdict": verdict}),
            encoding="utf-8",
        )
        (approvals_dir / f"{gate}.md").write_text(f"# {verdict}\n", encoding="utf-8")

    from scripts.verify_phase10_approvals import verify_approvals

    result = verify_approvals(approvals_dir=str(approvals_dir))

    assert result["valid"] is False
    assert any(g["gate_id"] == "phase10a" and g["has_fail"] for g in result["gates"])


def test_verify_phase10_approvals_excludes_raw_compose_env_or_secret_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    approvals_dir = tmp_path / ".hermes" / "reports" / "approvals"
    approvals_dir.mkdir(parents=True)
    for gate in ["phase10a", "phase10b", "phase10c", "phase10d", "documentation", "final"]:
        content = json.dumps({
            "gate_id": gate,
            "terminal_verdict": "APPROVED",
            "raw_env": "RAG_TOKEN=secret123",
        })
        (approvals_dir / f"{gate}.json").write_text(content, encoding="utf-8")

    from scripts.verify_phase10_approvals import scan_for_secrets

    for gate in ["phase10a", "phase10b", "phase10c", "phase10d", "documentation", "final"]:
        report_path = approvals_dir / f"{gate}.json"
        assert scan_for_secrets(str(report_path)) is True  # leak detected
