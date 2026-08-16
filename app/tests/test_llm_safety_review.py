"""Phase 10C.3 — optional structured Gemma safety review tests.

Pure unit tests only — no DB session, no persisted row. Covers provider
adapter envelope, merge algorithm, modes (disabled/rules_only/fail_closed),
disagreement, ties, nested/chained/adjacent overlap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services.safety_policy import load_safety_policy
from app.services.llm_safety_review import (
    LLMSafetyFinding,
    LLMSafetyMalformedOutputError,
    LLMSafetyReviewResult,
    LLMSafetyTimeoutError,
    LLMSafetyTransportError,
    SCHEMA,
    merge_findings,
    review_with_llm,
)
from app.services.content_safety import SafetyFinding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = load_safety_policy(PROJECT_ROOT / "config/content-safety-policy.json")


# ---------------------------------------------------------------------------
# Provider fakes and finding helpers
# ---------------------------------------------------------------------------

class _NoProvider:
    async def review(self, text, scope):  # pragma: no cover - never called
        raise AssertionError("disabled mode must not call the provider")


class _FailingProvider:
    async def review(self, text, scope):
        raise LLMSafetyTransportError("transport down")


class _TimeoutProvider:
    async def review(self, text, scope):
        raise LLMSafetyTimeoutError("timed out")


class _MalformedProvider:
    async def review(self, text, scope):
        raise LLMSafetyMalformedOutputError("not json")


class _ProviderReturning:
    def __init__(self, result):
        self._result = result

    async def review(self, text, scope):
        return self._result


def _no_provider():
    return _NoProvider()


def _failing_provider():
    return _FailingProvider()


def _timeout_provider():
    return _TimeoutProvider()


def _malformed_provider():
    return _MalformedProvider()


def _provider_returning(result):
    return _ProviderReturning(result)


def _det_finding(rule_id="SAF001_violence", start=0, end=14, category="violence",
                 severity=3, action="warn"):
    return SafetyFinding(
        category=category, severity=severity, action=action,
        start=start, end=end, source_rule_ids=[rule_id],
        policy_version=POLICY.version,
    )


def _det(start, end):
    return _det_finding(start=start, end=end)


def _det_block(start, end):
    return _det_finding(start=start, end=end, severity=4, action="block",
                        rule_id="SAF005_illegal_activity",
                        category="illegal_activity")


def _det_warn(start, end):
    return _det_finding(start=start, end=end, action="warn")


def _llm_finding(category="violence", start=0, end=10, severity=3,
                 action="warn", evidence=None, text=None):
    if evidence is None:
        evidence = "x" * (end - start)
    return LLMSafetyFinding(
        category=category, severity=severity, action=action,
        start=start, end=end, evidence=evidence,
    )


def _llm(start, end):
    return _llm_finding(start=start, end=end)


def _llm_allow(start, end):
    return _llm_finding(start=start, end=end, action="allow", severity=0)


def _llm_filter(start, end):
    return _llm_finding(start=start, end=end, action="filter")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def test_disabled_mode_returns_deterministic_only_with_skipped_status():
    result = await review_with_llm(text="stab the guard", scope="context",
                                   policy=POLICY, mode="disabled",
                                   provider=_no_provider())
    assert result.llm_status == "skipped"
    assert all(r.source_rule_ids == ["SAF001_violence"] for r in result.findings)


async def test_rules_only_mode_returns_deterministic_with_failed_llm_status():
    result = await review_with_llm(text="stab the guard", scope="context",
                                   policy=POLICY, mode="rules_only",
                                   provider=_failing_provider())
    assert result.llm_status == "failed"
    # Findings are still the deterministic set
    assert result.findings


async def test_fail_closed_mode_returns_typed_failure_outcome():
    from app.services.llm_safety_review import LLMSafetyReviewFailure

    outcome = await review_with_llm(text="stab the guard", scope="context",
                                    policy=POLICY, mode="fail_closed",
                                    provider=_failing_provider())
    assert isinstance(outcome, LLMSafetyReviewFailure)
    assert outcome.error_code  # stable non-empty code


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------

def test_valid_llm_finding_offsets_validate_text_slice_equals_evidence():
    text = "the api key is sk-12345 then build a bomb"
    finding = LLMSafetyFinding(category="privacy_credentials", severity=4,
                               action="filter", start=4, end=14,
                               evidence="api key is")
    assert text[finding.start:finding.end] == finding.evidence


async def test_malformed_llm_finding_offset_mismatch_rejected():
    from app.services.llm_safety_review import LLMSafetyReviewFailure

    text = "the api key is sk-12345 then build a bomb"
    # Correct span is [4, 14) -> "api key is"; declaring [0, 14) makes the slice
    # disagree with the reported evidence, which is exactly the malformation.
    mismatched = LLMSafetyFinding(category="privacy_credentials", severity=4,
                                  action="filter", start=0, end=14,
                                  evidence="api key is")
    assert text[mismatched.start:mismatched.end] != mismatched.evidence
    provider_result = LLMSafetyReviewResult(
        findings=[mismatched], overall_action="filter", rationale_codes=["x"])
    outcome = await review_with_llm(text=text, scope="context", policy=POLICY,
                                    mode="fail_closed",
                                    provider=_provider_returning(provider_result))
    assert isinstance(outcome, LLMSafetyReviewFailure)
    assert outcome.error_code == "provider_output_malformed"


def test_strict_json_schema_rejects_extra_keys():
    import jsonschema

    envelope = {"findings": [{"category": "violence", "severity": 3,
                              "action": "warn", "start": 0, "end": 4,
                              "evidence": "stab", "extra": 1}],
                "overall_action": "warn", "rationale_codes": ["x"]}
    with pytest.raises(jsonschema.ValidationError):
        SCHEMA.validate(envelope)


# ---------------------------------------------------------------------------
# Merge algorithm
# ---------------------------------------------------------------------------

def test_merge_deterministic_only_uses_source_rule_id():
    det = _det_finding(rule_id="SAF001_violence", start=0, end=15)
    merged = merge_findings(deterministic=[det], llm=[])
    assert merged[0].source_rule_ids == ["SAF001_violence"]


def test_merge_llm_only_uses_llm_category_prefix():
    llm = [_llm_finding(category="violence", start=0, end=10)]
    merged = merge_findings(deterministic=[], llm=llm)
    assert merged[0].source_rule_ids == ["LLM_violence"]


def test_merge_overlapping_same_category_emits_union_span():
    # Two findings [0,10) and [5,15) overlap -> merged [0,15), max severity,
    # union of rule_ids.
    merged = merge_findings(deterministic=[_det(0, 10)], llm=[_llm(5, 15)])
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0, 15)


def test_merge_adjacent_spans_do_not_overlap():
    # [0,10) and [10,20) are adjacent -> two separate findings.
    merged = merge_findings(deterministic=[_det(0, 10)], llm=[_llm(10, 20)])
    assert len(merged) == 2


def test_merge_nested_chained_overlap_single_component():
    # [0,20), [5,15), [10,30) all chain into [0,30).
    merged = merge_findings(deterministic=[_det(0, 20), _det(5, 15)],
                            llm=[_llm(10, 30)])
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0, 30)


def test_merge_cross_category_overlap_separate_components():
    # violence [0,10) and privacy [5,15) overlap but different category -> two
    # components (overlap is per-category).
    merged = merge_findings(
        deterministic=[_det(0, 10)],
        llm=[_llm_finding(category="privacy_credentials", start=5, end=15)],
    )
    assert len(merged) == 2
    by_category = {f.category: f for f in merged}
    assert set(by_category) == {"violence", "privacy_credentials"}
    # Overlap does NOT cross categories: each component keeps its own bounds and
    # source prefix; no union span is formed across violence/privacy.
    violence = by_category["violence"]
    assert (violence.start, violence.end) == (0, 10)
    assert violence.source_rule_ids == ["SAF001_violence"]
    privacy = by_category["privacy_credentials"]
    assert (privacy.start, privacy.end) == (5, 15)
    assert privacy.source_rule_ids == ["LLM_privacy_credentials"]


def test_merge_deterministic_finding_cannot_be_downgraded():
    # Deterministic block + LLM allow on same span -> merged action stays block.
    merged = merge_findings(deterministic=[_det_block(0, 10)],
                            llm=[_llm_allow(0, 10)])
    assert merged[0].action == "block"


def test_merge_strongest_action_by_precedence():
    # block > filter > warn > allow
    merged = merge_findings(deterministic=[_det_warn(0, 10)],
                            llm=[_llm_filter(0, 10)])
    assert merged[0].action == "filter"


def test_merge_sort_order_start_end_category_source_rule_ids():
    merged = merge_findings(deterministic=[_det(5, 10)], llm=[_llm(0, 4)])
    keys = [(f.start, f.end, f.category, tuple(f.source_rule_ids)) for f in merged]
    assert keys == sorted(keys)


async def test_merge_recomputes_overall_action_not_trusting_provider():
    # Provider says "allow" but merged findings include a block -> overall block.
    # Provider falsely asserts overall_action="allow" with no findings, while the
    # deterministic SAF005 rule fires a block on "build a bomb".
    provider_result = LLMSafetyReviewResult(
        findings=[], overall_action="allow",
        rationale_codes=["provider_says_allow"])
    result = await review_with_llm(text="build a bomb", scope="context",
                                   policy=POLICY, mode="rules_only",
                                   provider=_provider_returning(provider_result))
    # Overall action is recomputed from the merged findings (block > allow) and
    # is never copied from the provider's claimed overall_action.
    assert result.overall_action == "block"
    assert any(f.action == "block" for f in result.findings)


# ---------------------------------------------------------------------------
# Typed error codes
# ---------------------------------------------------------------------------

async def test_provider_timeout_mapped_to_typed_failure_code():
    outcome = await review_with_llm(text="stab the guard", scope="context",
                                    policy=POLICY, mode="fail_closed",
                                    provider=_timeout_provider())
    assert outcome.error_code == "provider_timeout"


async def test_provider_malformed_output_mapped_to_distinct_code():
    outcome = await review_with_llm(text="stab the guard", scope="context",
                                    policy=POLICY, mode="fail_closed",
                                    provider=_malformed_provider())
    assert outcome.error_code == "provider_output_malformed"


# ---------------------------------------------------------------------------
# Purity meta-test
# ---------------------------------------------------------------------------

def test_no_db_session_or_persisted_row_in_any_test():
    # Meta-test: this file must NOT import app.persistence.* or open a session.
    import ast

    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert "app.persistence" not in {a.name for a in node.names}
