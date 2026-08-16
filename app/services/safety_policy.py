"""Versioned content-safety policy (Task 10C.1).

Loads and strictly validates the pinned content-safety policy and (optionally)
its fixture corpus. The policy is an immutable object: stable category IDs,
severity 0-4, actions allow|warn|filter|block, scopes ingestion|context|
answer, action precedence allow < warn < filter < block (severity never
overrides action), and benign educational content as mandatory negative
controls in the fixture corpus.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

POLICY_VERSION = "safety-v1"
CATEGORIES = (
    "violence", "self_harm", "sexual_content", "hate_harassment",
    "illegal_activity", "privacy_credentials",
)
ACTIONS = ("allow", "warn", "filter", "block")
SCOPES = ("ingestion", "context", "answer")
LLM_FALLBACK_MODES = ("disabled", "rules_only", "fail_closed")

_TOP_LEVEL_KEYS = frozenset({
    "version", "categories", "rules", "llm_fallback", "max_input_chars",
    "precedence", "retention_days",
})
_CATEGORY_KEYS = frozenset({"id", "description"})
_RULE_KEYS = frozenset({
    "rule_id", "category", "pattern", "pattern_type", "scopes", "severity",
    "action",
})
_FIXTURE_TOP_KEYS = frozenset({"schema_version", "policy_version", "cases"})
_CASE_KEYS = frozenset({"id", "scope", "text", "expected_action", "expected_findings"})
_FINDING_KEYS = frozenset({
    "start", "end", "severity", "action", "category", "source_rule_ids",
})

# Credential-like tokens that must never appear in policy or fixture bytes.
_CREDENTIAL_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{32,}"),
)


def action_precedence(action_a: str, action_b: str) -> str:
    """Return the stronger action by the fixed precedence
    allow < warn < filter < block. Severity never participates."""
    rank = {action: index for index, action in enumerate(ACTIONS)}
    return action_a if rank[action_a] >= rank[action_b] else action_b


@dataclass(frozen=True)
class SafetyRule:
    rule_id: str
    category: str
    pattern: str
    pattern_type: str
    scopes: tuple[str, ...]
    severity: int
    action: str


@dataclass(frozen=True)
class SafetyPolicy:
    version: str
    categories: tuple[tuple[str, str], ...]  # (id, description)
    rules: tuple[SafetyRule, ...]
    llm_fallback: str
    max_input_chars: int
    precedence: tuple[str, ...]
    retention_days: int

    def rule_by_id(self, rule_id: str) -> SafetyRule | None:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None


def _reject(message: str) -> None:
    raise ValueError(f"safety policy invalid: {message}")


def _check_credentials(text: str) -> None:
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            _reject("credential-like token present")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        _reject(f"file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _reject(f"not valid JSON ({path}): {exc}")
    if not isinstance(raw, dict):
        _reject(f"top level must be an object ({path})")
    return raw


def _validate_policy(raw: dict) -> dict:
    if set(raw) != _TOP_LEVEL_KEYS:
        _reject(
            f"top-level key set mismatch (unknown={sorted(set(raw) - _TOP_LEVEL_KEYS)}, "
            f"missing={sorted(_TOP_LEVEL_KEYS - set(raw))})"
        )
    if raw["version"] != POLICY_VERSION:
        _reject(f"unsupported version: {raw['version']!r}")
    if raw["llm_fallback"] not in LLM_FALLBACK_MODES:
        _reject(f"invalid llm_fallback: {raw['llm_fallback']!r}")
    if not isinstance(raw["max_input_chars"], int) or raw["max_input_chars"] <= 0:
        _reject("max_input_chars must be a positive integer")
    if list(raw["precedence"]) != list(ACTIONS):
        _reject(f"precedence must be exactly {list(ACTIONS)}")
    if not isinstance(raw["retention_days"], int) or raw["retention_days"] < 0:
        _reject("retention_days must be a non-negative integer")

    category_ids: list[str] = []
    for category in raw["categories"]:
        if not isinstance(category, dict) or set(category) != _CATEGORY_KEYS:
            _reject("category key set mismatch")
        if category["id"] not in CATEGORIES:
            _reject(f"unknown category id: {category['id']!r}")
        if category["id"] in category_ids:
            _reject(f"duplicate category id: {category['id']!r}")
        if not isinstance(category["description"], str) or not category["description"]:
            _reject(f"category description must be non-empty: {category['id']!r}")
        category_ids.append(category["id"])
    if set(category_ids) != set(CATEGORIES):
        _reject("category set must cover exactly the six stable categories")

    rule_ids: list[str] = []
    for rule in raw["rules"]:
        if not isinstance(rule, dict) or set(rule) != _RULE_KEYS:
            _reject("rule key set mismatch")
        if not isinstance(rule["rule_id"], str) or not rule["rule_id"]:
            _reject("rule_id must be a non-empty string")
        if rule["rule_id"] in rule_ids:
            _reject(f"duplicate rule id: {rule['rule_id']!r}")
        if rule["category"] not in category_ids:
            _reject(f"unknown rule category: {rule['category']!r}")
        if not isinstance(rule["pattern"], str) or not rule["pattern"]:
            _reject(f"empty pattern in {rule['rule_id']!r}")
        if rule["pattern_type"] != "literal":
            _reject(f"pattern_type must be literal in v1: {rule['pattern_type']!r}")
        if not isinstance(rule["scopes"], list) or not rule["scopes"]:
            _reject(f"rule scopes must be a non-empty list: {rule['rule_id']!r}")
        for scope in rule["scopes"]:
            if scope not in SCOPES:
                _reject(f"unknown scope {scope!r} in {rule['rule_id']!r}")
        if not isinstance(rule["severity"], int) or not 0 <= rule["severity"] <= 4:
            _reject(f"severity must be an integer in 0..4: {rule['rule_id']!r}")
        if rule["action"] not in ACTIONS:
            _reject(f"unknown action: {rule['action']!r}")
        rule_ids.append(rule["rule_id"])
    if not rule_ids:
        _reject("rules must be a non-empty list")
    return raw


def _validate_fixture(raw: dict, policy_raw: dict) -> dict:
    if set(raw) != _FIXTURE_TOP_KEYS:
        _reject(
            f"fixture key set mismatch (unknown={sorted(set(raw) - _FIXTURE_TOP_KEYS)}, "
            f"missing={sorted(_FIXTURE_TOP_KEYS - set(raw))})"
        )
    if raw["schema_version"] != "content-safety-fixtures-v1":
        _reject(f"unsupported fixture schema_version: {raw['schema_version']!r}")
    if raw["policy_version"] != policy_raw["version"]:
        _reject("fixture policy_version does not match the policy")
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases:
        _reject("fixture corpus must be a non-empty list")

    policy_rules = {rule["rule_id"]: rule for rule in policy_raw["rules"]}
    referenced_rules: set[str] = set()
    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != _CASE_KEYS:
            _reject("fixture case key set mismatch")
        if not isinstance(case["id"], str) or not case["id"]:
            _reject("fixture case id must be a non-empty string")
        if case["id"] in case_ids:
            _reject(f"duplicate fixture case id: {case['id']!r}")
        case_ids.append(case["id"])
        if case["scope"] not in SCOPES:
            _reject(f"unknown fixture scope: {case['scope']!r}")
        if not isinstance(case["text"], str) or not case["text"]:
            _reject(f"empty fixture text: {case['id']!r}")
        if case["expected_action"] not in ACTIONS:
            _reject(f"unknown expected_action: {case['expected_action']!r}")
        normalized_text = _normalize(case["text"])
        matched_rules: set[str] = set()
        for finding in case["expected_findings"]:
            if not isinstance(finding, dict) or set(finding) != _FINDING_KEYS:
                _reject("fixture finding key set mismatch")
            start, end = finding["start"], finding["end"]
            if not isinstance(start, int) or not isinstance(end, int):
                _reject("fixture finding offsets must be integers")
            if not 0 <= start < end <= len(case["text"]):
                _reject(f"fixture finding span out of range: {case['id']!r}")
            if finding["category"] not in CATEGORIES:
                _reject(f"unknown finding category: {finding['category']!r}")
            if finding["action"] not in ACTIONS:
                _reject(f"unknown finding action: {finding['action']!r}")
            if not isinstance(finding["severity"], int) or not 0 <= finding["severity"] <= 4:
                _reject(f"finding severity must be 0..4: {case['id']!r}")
            if not isinstance(finding["source_rule_ids"], list) or not finding["source_rule_ids"]:
                _reject(f"finding source_rule_ids must be a non-empty list: {case['id']!r}")
            for rule_id in finding["source_rule_ids"]:
                if rule_id not in policy_rules:
                    _reject(f"fixture references absent rule: {rule_id!r}")
                rule = policy_rules[rule_id]
                if finding["category"] != rule["category"]:
                    _reject(
                        f"fixture finding category disagrees with rule "
                        f"{rule_id!r}: {case['id']!r}"
                    )
                if finding["action"] != rule["action"]:
                    _reject(
                        f"fixture finding action disagrees with rule "
                        f"{rule_id!r}: {case['id']!r}"
                    )
                if finding["severity"] != rule["severity"]:
                    _reject(
                        f"fixture finding severity disagrees with rule "
                        f"{rule_id!r}: {case['id']!r}"
                    )
                # Offsets must point at the rule's literal pattern in the
                # case text (normalized comparison; v1 fixtures pin spans at
                # the pattern boundary).
                span_text = _normalize(case["text"][start:end])
                if span_text != _normalize(rule["pattern"]):
                    _reject(
                        f"fixture finding span does not match rule pattern "
                        f"{rule_id!r}: {case['id']!r}"
                    )
                referenced_rules.add(rule_id)
                matched_rules.add(rule_id)
        # Every rule whose pattern occurs in the case text must be mapped by
        # the fixture (fixtures are exhaustive mappings, not selections).
        for rule_id, rule in policy_rules.items():
            if _normalize(rule["pattern"]) in normalized_text:
                if rule_id not in matched_rules and case["expected_action"] == "allow":
                    _reject(
                        f"unmapped fixture: case {case['id']!r} matches rule "
                        f"{rule_id!r} but expects allow with no findings"
                    )
    if case_ids != sorted(case_ids):
        _reject("fixture cases must be lexical by id")
    unreferenced = set(policy_rules) - referenced_rules
    if unreferenced:
        _reject(f"unreferenced rules: {sorted(unreferenced)}")
    return raw


def load_safety_policy(path, fixture_path=None) -> SafetyPolicy:
    """Load and strictly validate the content-safety policy.

    Unknown/missing keys, duplicate IDs, unknown categories/actions/scopes,
    invalid severity, credential-like tokens, and (with ``fixture_path``) any
    fixture corruption — absent rule references, offset mismatches, duplicate
    case IDs, empty corpus, unmapped matches, unreferenced rules — raise
    ValueError. Returns an immutable :class:`SafetyPolicy`.
    """
    policy_path = Path(path)
    raw = _load_json(policy_path)
    _check_credentials(policy_path.read_text(encoding="utf-8"))
    _validate_policy(raw)

    if fixture_path is not None:
        fixture = Path(fixture_path)
        fixture_raw = _load_json(fixture)
        _check_credentials(fixture.read_text(encoding="utf-8"))
        _validate_fixture(fixture_raw, raw)

    return SafetyPolicy(
        version=raw["version"],
        categories=tuple(
            (category["id"], category["description"]) for category in raw["categories"]
        ),
        rules=tuple(
            SafetyRule(
                rule_id=rule["rule_id"],
                category=rule["category"],
                pattern=rule["pattern"],
                pattern_type=rule["pattern_type"],
                scopes=tuple(rule["scopes"]),
                severity=rule["severity"],
                action=rule["action"],
            )
            for rule in raw["rules"]
        ),
        llm_fallback=raw["llm_fallback"],
        max_input_chars=raw["max_input_chars"],
        precedence=tuple(raw["precedence"]),
        retention_days=raw["retention_days"],
    )
