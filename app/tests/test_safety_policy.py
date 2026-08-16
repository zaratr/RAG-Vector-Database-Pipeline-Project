"""Phase 10C.1 — versioned safety policy tests.

Covers: pinned policy/fixture bytes & SHA-256, category/action/scope/
precedence validation, every fixture mapping, startup refusal, immutability.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "config/content-safety-policy.json"
FIXTURE_PATH = PROJECT_ROOT / "app/tests/fixtures/content_safety.json"
POLICY_BYTES_EXPECTED = 1785
POLICY_SHA256_EXPECTED = (
    "2a9c9c5d4d44cce8ecb02bbf2b8586f6dd86dc410e474b93552e22180637d4f1"
)
FIXTURE_BYTES_EXPECTED = 4627
FIXTURE_SHA256_EXPECTED = (
    "2560db13b33f42c986229d522702540a50a1a842f3fbb71bd8e9f25a94aa0758"
)
EXPECTED_CATEGORIES = {
    "violence", "self_harm", "sexual_content", "hate_harassment",
    "illegal_activity", "privacy_credentials",
}
EXPECTED_ACTIONS = {"allow", "warn", "filter", "block"}
EXPECTED_PRECEDENCE = ["allow", "warn", "filter", "block"]
EXPECTED_RULE_IDS = {
    "SAF001_violence", "SAF002_self_harm", "SAF003_sexual_content",
    "SAF004_hate_harassment", "SAF005_illegal_activity", "SAF006_privacy_credentials",
}


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Never leak a settings cache built against a patched env."""
    yield
    from app.config import get_settings

    get_settings.cache_clear()


def _mutate(mutation: str) -> tuple[dict, dict]:
    """Produce corrupted copies of the policy/fixture for each mutation."""
    policy = json.loads(POLICY_PATH.read_text())
    fixture = json.loads(FIXTURE_PATH.read_text())

    if mutation == "duplicate_rule_id":
        policy["rules"][1]["rule_id"] = policy["rules"][0]["rule_id"]
    elif mutation == "unknown_category":
        policy["rules"][0]["category"] = "explosions"
    elif mutation == "unknown_action":
        policy["rules"][0]["action"] = "silence"
    elif mutation == "invalid_severity_high":
        policy["rules"][0]["severity"] = 5
    elif mutation == "invalid_severity_negative":
        policy["rules"][0]["severity"] = -1
    elif mutation == "unmapped_fixture":
        # A benign case starts matching a rule but still expects allow with
        # no findings.
        case = next(c for c in fixture["cases"] if c["id"] == "violence-benign")
        case["text"] = f"{case['text']} stab the guard"
    elif mutation == "duplicate_fixture_id":
        fixture["cases"][1]["id"] = fixture["cases"][0]["id"]
    elif mutation == "empty_corpus":
        fixture["cases"] = []
    elif mutation == "unknown_top_level_key":
        policy["unknown_key"] = True
    elif mutation == "fixture_reference_absent_rule":
        finding = next(
            f for c in fixture["cases"] for f in c["expected_findings"]
        )
        finding["source_rule_ids"] = ["SAF999_nonexistent"]
    elif mutation == "offset_mismatch":
        finding = next(
            f for c in fixture["cases"] for f in c["expected_findings"]
        )
        finding["end"] = finding["end"] + 3
    elif mutation == "real_credential_token_present":
        case = fixture["cases"][0]
        case["text"] = f"{case['text']} sk-abcdefghijklmnopqrst"
    else:
        raise AssertionError(f"unknown mutation {mutation!r}")
    return policy, fixture


# ---------------------------------------------------------------------------
# Pinned bytes
# ---------------------------------------------------------------------------

def test_policy_byte_count_and_sha256_immutable():
    raw = POLICY_PATH.read_bytes()
    assert len(raw) == POLICY_BYTES_EXPECTED
    assert raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == POLICY_SHA256_EXPECTED


def test_fixture_byte_count_and_sha256_immutable():
    raw = FIXTURE_PATH.read_bytes()
    assert len(raw) == FIXTURE_BYTES_EXPECTED
    assert raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256_EXPECTED


# ---------------------------------------------------------------------------
# Policy shape
# ---------------------------------------------------------------------------

def test_policy_categories_match_expected_set():
    obj = json.loads(POLICY_PATH.read_text())
    assert {c["id"] for c in obj["categories"]} == EXPECTED_CATEGORIES


def test_policy_precedence_is_block_filter_warn_allow():
    obj = json.loads(POLICY_PATH.read_text())
    assert obj["precedence"] == EXPECTED_PRECEDENCE


def test_policy_rule_ids_and_actions_match_expected():
    obj = json.loads(POLICY_PATH.read_text())
    rule_ids = {r["rule_id"] for r in obj["rules"]}
    actions = {r["action"] for r in obj["rules"]}
    assert rule_ids == EXPECTED_RULE_IDS
    assert actions.issubset(EXPECTED_ACTIONS)


def test_every_rule_scope_is_ingestion_context_answer():
    obj = json.loads(POLICY_PATH.read_text())
    for rule in obj["rules"]:
        assert set(rule["scopes"]) == {"ingestion", "context", "answer"}


# ---------------------------------------------------------------------------
# Loader contract
# ---------------------------------------------------------------------------

def test_load_safety_policy_returns_immutable_object():
    from app.services.safety_policy import SafetyPolicy, load_safety_policy

    policy = load_safety_policy(POLICY_PATH)
    assert isinstance(policy, SafetyPolicy)
    assert policy.version == "safety-v1"
    with pytest.raises((TypeError, AttributeError)):
        policy.version = "tampered"


def test_every_fixture_mapping_loads_and_matches_policy():
    """The committed corpus validates against the policy and every case's
    expected action/findings agree with the pinned rules."""
    from app.services.safety_policy import load_safety_policy

    policy = load_safety_policy(POLICY_PATH, fixture_path=FIXTURE_PATH)
    cases = json.loads(FIXTURE_PATH.read_text())["cases"]
    rule_by_id = {r.rule_id: r for r in policy.rules}
    for case in cases:
        for finding in case["expected_findings"]:
            for rule_id in finding["source_rule_ids"]:
                rule = rule_by_id[rule_id]
                assert finding["category"] == rule.category
                assert finding["action"] == rule.action
                assert finding["severity"] == rule.severity
        # The strongest expected finding action equals the case action.
        if case["expected_findings"]:
            strongest = case["expected_findings"][0]["action"]
            from app.services.safety_policy import action_precedence

            for finding in case["expected_findings"][1:]:
                strongest = action_precedence(strongest, finding["action"])
            assert strongest == case["expected_action"]
        else:
            assert case["expected_action"] == "allow"


@pytest.mark.parametrize("mutation", [
    "duplicate_rule_id", "unknown_category", "unknown_action",
    "invalid_severity_high", "invalid_severity_negative",
    "unmapped_fixture", "duplicate_fixture_id", "empty_corpus",
    "unknown_top_level_key", "fixture_reference_absent_rule",
    "offset_mismatch", "real_credential_token_present",
])
def test_policy_or_fixture_rejects_each_invalid_mutation(mutation, tmp_path):
    bad_policy, bad_fixture = _mutate(mutation)
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(bad_policy))
    f = tmp_path / "fixture.json"
    f.write_text(json.dumps(bad_fixture))
    from app.services.safety_policy import load_safety_policy

    with pytest.raises(ValueError):
        load_safety_policy(p, fixture_path=f)


def test_severity_never_overrides_action_precedence():
    # A severity=1 block must outrank a severity=4 warn at the same span.
    from app.services.safety_policy import action_precedence

    assert action_precedence("block", "warn") == "block"
    assert action_precedence("warn", "filter") == "filter"


# ---------------------------------------------------------------------------
# Startup and settings wiring
# ---------------------------------------------------------------------------

def test_startup_aborts_when_safety_enabled_and_policy_missing(monkeypatch):
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "true")
    monkeypatch.setenv("RAG_CONTENT_SAFETY_POLICY_PATH", "/nonexistent.json")
    from app.config import get_settings

    get_settings.cache_clear()
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        from app.main import app

        with TestClient(app):
            pass


def test_startup_aborts_when_safety_enabled_and_policy_invalid(monkeypatch, tmp_path):
    bad = json.loads(POLICY_PATH.read_text())
    bad["version"] = "wrong"
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(bad))
    monkeypatch.setenv("RAG_CONTENT_SAFETY_ENABLED", "true")
    monkeypatch.setenv("RAG_CONTENT_SAFETY_POLICY_PATH", str(p))
    from app.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(ValueError):
        from app.main import app

        with TestClient(app):
            pass


def test_settings_render_default_safety_disabled(monkeypatch):
    monkeypatch.delenv("RAG_CONTENT_SAFETY_ENABLED", raising=False)
    monkeypatch.delenv("RAG_CONTENT_SAFETY_POLICY_PATH", raising=False)
    monkeypatch.delenv("RAG_SAFETY_LLM_MODE", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.content_safety_enabled is False
    assert s.safety_llm_mode == "disabled"
    assert str(s.content_safety_policy_path) == "/app/config/content-safety-policy.json"


def test_settings_reject_unknown_safety_llm_mode(monkeypatch):
    monkeypatch.setenv("RAG_SAFETY_LLM_MODE", "sometimes")
    from app.config import Settings

    with pytest.raises(Exception):
        Settings()


def test_no_request_can_replace_policy_version():
    # Calling load_safety_policy twice returns the same immutable object via
    # app.main lifespan; a second load with a different path during a request
    # must be a no-op (the injected policy object is unchanged).
    from app.services.safety_policy import load_safety_policy

    startup = load_safety_policy(POLICY_PATH)
    pinned_version = startup.version
    pinned_rule_ids = [r.rule_id for r in startup.rules]
    # The injected policy is frozen: request code cannot reassign version/rules.
    with pytest.raises((TypeError, AttributeError)):
        startup.version = "safety-v2"
    with pytest.raises((TypeError, AttributeError)):
        startup.rules = []
    # A second load of the authoritative path yields an equal object with the
    # same pinned version; it never mutates the startup reference.
    reloaded = load_safety_policy(POLICY_PATH)
    assert startup.version == pinned_version == "safety-v1"
    assert [r.rule_id for r in startup.rules] == pinned_rule_ids
    assert reloaded.version == startup.version
    assert {r.rule_id for r in startup.rules} == EXPECTED_RULE_IDS
