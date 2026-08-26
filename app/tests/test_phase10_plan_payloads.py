"""Self-contained tests for ``scripts/validate_phase10_plan_payloads``.

Every test builds a synthetic plan document under ``tmp_path``; the suite must
pass from a clean checkout with no external plan document present. The CLI
validator remains available to operators via an explicit ``--plan`` path.

Earlier revisions pinned some of these tests to a plan document stored outside
the repository. Per the self-containment directive those document-pinned
assertions were removed; what remains (and what the converted tests cover) is
the validator machinery: marker and prose declaration parsing, inline-text
extraction, byte-count/SHA-256 verification on both the success and failure
paths, and the no-payload-leak guarantee.
"""

import json

import pytest


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


def test_validate_plan_payloads_locates_inline_text_payload_via_type_inline_marker(tmp_path):
    """A ``type=inline`` marker binds the backtick-wrapped JSON that follows it.

    Converted from a document-pinned test: the machinery under assertion is the
    inline-text extraction path (payload_type, canonical serialization), not any
    particular plan document's content.
    """
    from scripts.validate_phase10_plan_payloads import locate_payloads

    obj = {"source": "operator-supplied", "trust": "high"}
    serialized = (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    plan = tmp_path / "inline.md"
    plan.write_text(
        "# Plan\n\n<!-- payload: source-trust-policy bytes="
        + str(len(serialized))
        + " type=inline -->\n`"
        + serialized.decode("utf-8")
        + "`\n",
        encoding="utf-8",
    )

    payloads = locate_payloads(str(plan))
    trust = next(p for p in payloads if p["label"] == "source-trust-policy")

    assert trust["payload_type"] == "inline-text"
    assert trust["serialized"] == serialized


def test_validate_plan_passes_when_every_declared_payload_matches(tmp_path):
    """validate_plan must fully pass when all required payloads are present and
    their declared byte counts and SHA-256 digests match (any defect raises
    SystemExit(2)).

    Converted from a document-pinned test: the assertion logic (full success
    path of the validator, including SHA-256 verification and the inline-text
    required payload) is preserved against a synthetic plan document.
    """
    import hashlib

    from scripts.validate_phase10_plan_payloads import validate_plan

    def fence(label: str, obj: dict) -> str:
        body = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        raw = body.encode("utf-8") + b"\n"
        return (
            f"<!-- payload: {label} bytes={len(raw)} "
            f"sha256={hashlib.sha256(raw).hexdigest()} -->\n```json\n{body}\n```\n"
        )

    inline_obj = {"source": "operator-supplied"}
    inline_raw = (
        json.dumps(inline_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    plan = tmp_path / "complete.md"
    plan.write_text(
        "# Plan\n\n"
        + fence("content-safety-policy", {"policy": "block"})
        + "\n"
        + fence("content-safety-fixture", {"fixture": True})
        + "\n"
        + fence("context-security-policy", {"context": "strict"})
        + "\n"
        + fence("attack-corpus", {"attacks": []})
        + "\n"
        + f"<!-- payload: source-trust-policy bytes={len(inline_raw)} type=inline -->\n"
        + "`"
        + inline_raw.decode("utf-8")
        + "`\n",
        encoding="utf-8",
    )

    validate_plan(str(plan))


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
