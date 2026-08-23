import json
from pathlib import Path

import pytest

PLAN_PATH = Path(__file__).resolve().parents[2] / ".hermes" / "plans" / (
    "2026-08-01_094008-phase-10-contract-reassessment-and-implementation.md"
)


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


def test_validate_plan_payloads_locates_prose_declared_attack_corpus():
    """The 17,757-byte corpus fence is prose-declared; the full-text fence scan
    must parse it (a payload larger than the marker fence window)."""
    import hashlib

    from scripts.validate_phase10_plan_payloads import locate_payloads, reserialize_payload

    payloads = locate_payloads(str(PLAN_PATH))
    corpus = next(p for p in payloads if p["label"] == "attack-corpus")

    assert corpus["payload_type"] == "json-fence"
    serialized = reserialize_payload(corpus)
    assert len(serialized) == 17757
    assert (
        hashlib.sha256(serialized).hexdigest()
        == "bca0c8eed73c02a346b92658b9545620aa4565e5decb8befc6a74f541f8d03ce"
    )


def test_validate_plan_payloads_locates_text_fence_owasp_excerpt():
    """The OWASP excerpt is a raw-text payload bound by a leading prose
    declaration; it serializes as raw UTF-8 bytes plus a final LF."""
    import hashlib

    from scripts.validate_phase10_plan_payloads import locate_payloads, reserialize_payload

    payloads = locate_payloads(str(PLAN_PATH))
    excerpt = next(p for p in payloads if p["label"] == "owasp-excerpt")

    assert excerpt["payload_type"] == "text-fence"
    serialized = reserialize_payload(excerpt)
    assert len(serialized) == 436
    assert (
        hashlib.sha256(serialized).hexdigest()
        == "e024da7f5a562882e3ba8c8eae62d74db29d9aa86ec20b5512ad6be58c0c6200"
    )


def test_validate_plan_payloads_locates_every_manifest_label():
    """All seven B-16 manifest labels must resolve to exactly one payload each."""
    from scripts.validate_phase10_plan_payloads import KNOWN_LABELS, locate_payloads

    payloads = locate_payloads(str(PLAN_PATH))

    assert {p["label"] for p in payloads} == KNOWN_LABELS


def test_validate_plan_passes_on_the_committed_plan():
    """validate_plan must fully pass (any defect raises SystemExit(2))."""
    from scripts.validate_phase10_plan_payloads import validate_plan

    validate_plan(str(PLAN_PATH))


def test_prose_trailing_declaration_locates_preceding_fence(tmp_path):
    """A "`path`: N bytes ... SHA-256 `hex`" line binds the fence above it."""
    import hashlib

    from scripts.validate_phase10_plan_payloads import locate_payloads, reserialize_payload

    serialized = json.dumps({"a": 1}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    plan = tmp_path / "trailing.md"
    plan.write_text(
        '# Plan\n\n```json\n{"a":1}\n```\n\n'
        "`app/tests/fixtures/content_safety.json`: "
        + str(len(serialized))
        + " bytes including final LF, SHA-256 `"
        + hashlib.sha256(serialized).hexdigest()
        + "`\n",
        encoding="utf-8",
    )

    payloads = locate_payloads(str(plan))
    fixture = next(p for p in payloads if p["label"] == "content-safety-fixture")

    assert fixture["payload_type"] == "json-fence"
    assert reserialize_payload(fixture) == serialized


def test_prose_leading_declaration_binds_text_fence(tmp_path):
    """A "`path` is exactly these N ... SHA-256 `hex`" lead binds the ```` ````text
    fence below it for .txt artifacts and serializes raw text plus final LF."""
    import hashlib

    from scripts.validate_phase10_plan_payloads import locate_payloads

    body = "first line\nsecond line"
    raw = (body + "\n").encode("utf-8")
    plan = tmp_path / "leading.md"
    plan.write_text(
        "# Plan\n\n`docs/references/owasp-llm-top10-2025-excerpt.txt` is exactly these "
        + str(len(raw))
        + " UTF-8/LF bytes, with SHA-256 `"
        + hashlib.sha256(raw).hexdigest()
        + "`:\n\n```text\n"
        + body
        + "\n```\n",
        encoding="utf-8",
    )

    payloads = locate_payloads(str(plan))
    excerpt = next(p for p in payloads if p["label"] == "owasp-excerpt")

    assert excerpt["payload_type"] == "text-fence"
    assert excerpt["serialized"] == raw


def test_marker_takes_precedence_over_prose_declaration(tmp_path):
    """When a label carries both an HTML marker and a prose declaration, the
    marker's declaration wins and exactly one payload is recorded."""
    import hashlib

    from scripts.validate_phase10_plan_payloads import locate_payloads

    serialized = json.dumps({"a": 1}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    plan = tmp_path / "precedence.md"
    plan.write_text(
        "# Plan\n\n<!-- payload: content-safety-fixture bytes="
        + str(len(serialized))
        + " sha256="
        + hashlib.sha256(serialized).hexdigest()
        + " type=fence -->\n```json\n{\"a\":1}\n```\n\n"
        "`app/tests/fixtures/content_safety.json`: 999 bytes, SHA-256 `"
        + "0" * 64
        + "`\n",
        encoding="utf-8",
    )

    payloads = locate_payloads(str(plan))
    fixture = [p for p in payloads if p["label"] == "content-safety-fixture"]

    assert len(fixture) == 1
    assert fixture[0]["declared_bytes"] == len(serialized)
    assert fixture[0]["serialized"] == serialized


def test_large_fence_beyond_marker_window_parses_via_prose_declaration(tmp_path):
    """A fenced payload larger than the marker fence window (8192 chars) must
    still parse when prose-declared, because prose declarations scan the full
    text for the fence."""
    import hashlib

    from scripts.validate_phase10_plan_payloads import locate_payloads, reserialize_payload

    obj = {"fixtures": [{"id": f"fixture-{i:04d}", "text": "x" * 40} for i in range(200)]}
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    assert len(serialized) > 8192
    plan = tmp_path / "large.md"
    plan.write_text(
        "# Plan\n\nThe committed attack file is exactly the following "
        + str(len(serialized))
        + "-byte payload with SHA-256 `"
        + hashlib.sha256(serialized).hexdigest()
        + "`:\n\n```json\n"
        + serialized.decode("utf-8").rstrip("\n")
        + "\n```\n",
        encoding="utf-8",
    )

    payloads = locate_payloads(str(plan))
    corpus = next(p for p in payloads if p["label"] == "attack-corpus")

    assert reserialize_payload(corpus) == serialized
    assert corpus["declared_bytes"] == len(serialized)
