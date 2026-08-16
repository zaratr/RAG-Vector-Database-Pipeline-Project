import json
import os
from pathlib import Path

import pytest

_PLAN_FILENAME = "2026-08-01_094008-phase-10-contract-reassessment-and-implementation.md"
# Plan may live in-project (.hermes) or in an external archive.
_candidate_paths = [
    Path(__file__).resolve().parents[2] / ".hermes" / "plans" / _PLAN_FILENAME,
    Path(os.environ.get("RAG_PLAN_PATH", "/c/Users/zarat/Projects/random_archive/.hermes/plans")) / _PLAN_FILENAME,
]
PLAN_PATH = next((p for p in _candidate_paths if p.exists()), _candidate_paths[0])


def test_validate_plan_payloads_locates_all_json_fence_payloads():
    from scripts.validate_phase10_plan_payloads import locate_payloads

    payloads = locate_payloads(str(PLAN_PATH))

    labels = {p["label"] for p in payloads}
    assert "content-safety-policy" in labels
    assert "content-safety-fixture" in labels
    assert "context-security-policy" in labels
    assert "attack-corpus" in labels


def test_validate_plan_payloads_asserts_declared_byte_count_for_content_safety_policy():
    from scripts.validate_phase10_plan_payloads import locate_payloads, reserialize_payload

    payloads = locate_payloads(str(PLAN_PATH))
    policy = next(p for p in payloads if p["label"] == "content-safety-policy")
    serialized = reserialize_payload(policy)

    assert len(serialized) == 1785


def test_validate_plan_payloads_asserts_declared_sha256_for_content_safety_policy():
    import hashlib

    from scripts.validate_phase10_plan_payloads import locate_payloads, reserialize_payload

    payloads = locate_payloads(str(PLAN_PATH))
    policy = next(p for p in payloads if p["label"] == "content-safety-policy")
    serialized = reserialize_payload(policy)

    assert hashlib.sha256(serialized).hexdigest() == "2a9c9c5d4d44cce8ecb02bbf2b8586f6dd86dc410e474b93552e22180637d4f1"


def test_validate_plan_payloads_asserts_declared_byte_count_for_context_security_policy():
    from scripts.validate_phase10_plan_payloads import locate_payloads, reserialize_payload

    payloads = locate_payloads(str(PLAN_PATH))
    policy = next(p for p in payloads if p["label"] == "context-security-policy")
    serialized = reserialize_payload(policy)

    assert len(serialized) == 2207


def test_validate_plan_payloads_locates_inline_text_source_trust_policy():
    from scripts.validate_phase10_plan_payloads import locate_payloads

    payloads = locate_payloads(str(PLAN_PATH))
    trust = next((p for p in payloads if p["label"] == "source-trust-policy"), None)

    assert trust is not None
    assert trust["payload_type"] == "inline-text"
    assert len(trust["serialized"]) == 288


def test_validate_plan_payloads_exits_2_on_missing_payload(tmp_path):
    """A plan missing a required labeled payload must exit 2."""
    bogus_plan = tmp_path / "bogus.md"
    bogus_plan.write_text("# No payloads here\n", encoding="utf-8")

    from scripts.validate_phase10_plan_payloads import validate_plan

    with pytest.raises(SystemExit) as exc_info:
        validate_plan(str(bogus_plan))

    assert exc_info.value.code == 2


def test_validate_plan_payloads_exits_2_on_byte_count_mismatch(tmp_path, monkeypatch):
    """A payload whose declared byte count doesn't match serialized bytes exits 2."""
    plan = tmp_path / "mismatch.md"
    plan.write_text(
        '# Plan\n\n```json\n{"a":1}\n```\n<!-- payload: test-payload bytes=999 sha256=abc -->\n',
        encoding="utf-8",
    )

    from scripts.validate_phase10_plan_payloads import validate_plan

    with pytest.raises(SystemExit) as exc_info:
        validate_plan(str(plan))

    assert exc_info.value.code == 2


def test_validate_plan_payloads_does_not_print_payload_bytes(tmp_path, capsys):
    """On failure, raw payload content must never be printed."""
    plan = tmp_path / "fail.md"
    plan.write_text(
        '# Plan\n\n```json\n{"secret":"value"}\n```\n<!-- payload: leak-payload bytes=1 -->\n',
        encoding="utf-8",
    )

    from scripts.validate_phase10_plan_payloads import validate_plan

    with pytest.raises(SystemExit):
        validate_plan(str(plan))

    captured = capsys.readouterr()
    assert "secret" not in captured.out
    assert "secret" not in captured.err
