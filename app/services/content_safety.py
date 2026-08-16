"""Deterministic first-pass content-safety classification (Task 10C.2).

Pure function over the immutable 10C.1 policy: NFKC + casefold normalization
with an index map back to the original string, original-string half-open
offsets, scope-filtered rule application, dedup/sort by
``(start, end, source_rule_ids)``, and the fixed action precedence. No
database, provider, time, or random access.
"""
from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import BaseModel

from app.services.safety_policy import SafetyPolicy, action_precedence

SafetyScope = Literal["ingestion", "context", "answer"]


class SafetyInputLimitError(ValueError):
    """Input exceeds the policy's max_input_chars (Unicode code points).

    Raised before classification; input is never silently truncated.
    """


class SafetyFinding(BaseModel):
    category: str
    severity: int            # 0..4
    action: Literal["allow", "warn", "filter", "block"]
    start: int
    end: int
    source_rule_ids: list[str]  # sorted unique, e.g. SAF... and/or LLM_...
    policy_version: str


class SafetyAssessment(BaseModel):
    scope: SafetyScope
    action: Literal["allow", "warn", "filter", "block"]
    findings: list[SafetyFinding]
    policy_version: str


def _build_index_map(text: str) -> tuple[str, list[int]]:
    """NFKC + casefold normalize text, building an index map.

    Returns ``(normalized_text, index_map)`` where ``index_map[norm_pos]``
    gives the original-string position for each normalized character.
    """
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for orig_pos, char in enumerate(text):
        nfkc = unicodedata.normalize("NFKC", char)
        folded = nfkc.casefold()
        for fc in folded:
            normalized_chars.append(fc)
            index_map.append(orig_pos)
    return "".join(normalized_chars), index_map


def _translate_span(norm_start: int, norm_end: int,
                    index_map: list[int]) -> tuple[int, int]:
    """Translate normalized-text offsets to original-string offsets."""
    orig_start = index_map[norm_start]
    orig_end = index_map[norm_end - 1] + 1
    return orig_start, orig_end


def classify_content(text: str, *, scope: SafetyScope,
                     policy: SafetyPolicy) -> SafetyAssessment:
    """Classify ``text`` under ``scope`` against the immutable policy.

    Applies only rules whose scopes contain the requested scope. Findings use
    original-string half-open offsets, are deduplicated by
    ``(start, end, source_rule_ids)`` and sorted by
    ``(start, end, source_rule_ids)``. The overall action is the strongest
    finding action or ``allow`` when there are none. Pure: no database,
    provider, time, or random access. Input over ``max_input_chars`` Unicode
    code points is rejected with :class:`SafetyInputLimitError` before any
    classification.
    """
    if len(text) > policy.max_input_chars:
        raise SafetyInputLimitError(
            f"input of {len(text)} code points exceeds max_input_chars="
            f"{policy.max_input_chars}"
        )

    normalized_text, index_map = _build_index_map(text)

    raw_findings: list[tuple[int, int, tuple[str, ...], str, int, str]] = []
    for rule in policy.rules:
        if scope not in rule.scopes:
            continue
        pattern = unicodedata.normalize("NFKC", rule.pattern).casefold()
        if not pattern:
            continue
        search_from = 0
        while True:
            pos = normalized_text.find(pattern, search_from)
            if pos == -1:
                break
            norm_end = pos + len(pattern)
            start, end = _translate_span(pos, norm_end, index_map)
            raw_findings.append(
                (start, end, (rule.rule_id,), rule.category,
                 rule.severity, rule.action)
            )
            search_from = norm_end

    # Deduplicate by (start, end, source_rule_ids); sort by the same key.
    seen: set[tuple[int, int, tuple[str, ...]]] = set()
    findings: list[SafetyFinding] = []
    for start, end, rule_ids, category, severity, action in sorted(raw_findings):
        key = (start, end, rule_ids)
        if key in seen:
            continue
        seen.add(key)
        findings.append(SafetyFinding(
            category=category,
            severity=severity,
            action=action,  # type: ignore[arg-type]
            start=start,
            end=end,
            source_rule_ids=sorted(set(rule_ids)),
            policy_version=policy.version,
        ))

    overall = "allow"
    for finding in findings:
        overall = action_precedence(overall, finding.action)

    return SafetyAssessment(
        scope=scope,
        action=overall,  # type: ignore[arg-type]
        findings=findings,
        policy_version=policy.version,
    )
