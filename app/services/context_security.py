"""Context-security detector for retrieved prompt injection (Task 10B.4).

Treats retrieved text as untrusted data and prevents it from changing system
behavior. Applies a deterministic literal-pattern policy with NFKC/casefold
normalization, encoded-marker detection, and span dedup/merge.
"""
from __future__ import annotations

import base64
import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ContextAction = Literal["allow", "quarantine", "block"]

POLICY_VERSION = "context-security-v1"

# Deterministic rule IDs are fixed (plan §10B.4): the policy must contain
# exactly these six and no others.
FIXED_RULE_IDS = frozenset({
    "CTX001_instruction_override",
    "CTX002_system_prompt_request",
    "CTX003_tool_or_credential_request",
    "CTX004_role_impersonation",
    "CTX005_encoded_instruction_marker",
    "CTX006_cross_context_command",
})

_RULE_KEYS = frozenset({
    "rule_id", "action", "pattern", "pattern_type", "normalization",
    "case_sensitive", "word_boundary", "encoded_marker",
})
_ENCODED_MARKER_KEYS = frozenset({"encoding", "marker_prefix", "max_encoded_length"})
_NEGATIVE_CONTROL_KEYS = frozenset({"id", "text", "expected_action", "rationale"})
_SPAN_SELECTION_KEYS = frozenset({"dedup", "overlap_strategy", "sort_order"})
_TOP_LEVEL_KEYS = frozenset({"version", "rules", "negative_controls", "span_selection"})


class ContextSecurityResult(BaseModel):
    chunk_id: int
    action: ContextAction
    rule_ids: list[str]
    matched_spans: list[tuple[int, int]]
    policy_version: str


class ContextSecurityPolicy:
    """Validated, immutable context-security policy."""

    def __init__(self, raw: dict):
        self.version = raw["version"]
        self.rules = raw["rules"]
        self.negative_controls = raw.get("negative_controls", [])
        self.span_selection = raw.get("span_selection", {})

    def normalize_pattern(self, pattern: str) -> str:
        return unicodedata.normalize("NFKC", pattern).casefold()


def load_context_security_policy(path: str) -> ContextSecurityPolicy:
    """Load and strictly validate the context-security policy.

    Unknown or missing keys at any object level, unknown rule IDs or actions,
    non-literal pattern types, wrong normalization/flags, or wrong version
    raise ValueError (startup aborts).
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"context-security policy not found: {path}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"context-security policy is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_LEVEL_KEYS:
        raise ValueError("context-security policy top-level key set mismatch")
    if raw["version"] != POLICY_VERSION:
        raise ValueError(f"unexpected policy version: {raw['version']}")
    if not isinstance(raw["rules"], list) or not raw["rules"]:
        raise ValueError("policy rules must be a non-empty array")
    rule_ids = set()
    for rule in raw["rules"]:
        if not isinstance(rule, dict) or not set(rule) <= _RULE_KEYS:
            raise ValueError(f"unknown rule key in {rule.get('rule_id')!r}")
        required = _RULE_KEYS - {"encoded_marker"}
        if not required <= set(rule):
            raise ValueError(f"rule missing keys: {sorted(required - set(rule))}")
        if rule["rule_id"] not in FIXED_RULE_IDS or rule["rule_id"] in rule_ids:
            raise ValueError(f"unknown or duplicate rule id: {rule['rule_id']!r}")
        rule_ids.add(rule["rule_id"])
        if rule["action"] not in ("quarantine", "block"):
            raise ValueError(f"invalid action: {rule['action']!r}")
        if not isinstance(rule["pattern"], str) or not rule["pattern"]:
            # An empty pattern would match at every position; reject at load.
            raise ValueError(f"pattern must be a non-empty string in {rule['rule_id']}")
        if rule["pattern_type"] != "literal":
            raise ValueError(f"pattern_type must be literal in v1: {rule['pattern_type']!r}")
        if rule["normalization"] != "nfkc_casefold":
            raise ValueError(f"normalization must be nfkc_casefold: {rule['normalization']!r}")
        if rule["case_sensitive"] is not False:
            raise ValueError("case_sensitive must be false")
        if rule["word_boundary"] is not False:
            raise ValueError("word_boundary must be false")
        marker = rule.get("encoded_marker")
        if marker is not None:
            if not isinstance(marker, dict) or set(marker) != _ENCODED_MARKER_KEYS:
                raise ValueError(f"encoded_marker key set mismatch in {rule['rule_id']}")
            if marker["encoding"] not in ("base64", "base64_and_hex", "hex"):
                raise ValueError(f"invalid encoded_marker encoding: {marker['encoding']!r}")
            if not isinstance(marker["marker_prefix"], str) or len(marker["marker_prefix"]) > 200:
                raise ValueError("marker_prefix must be a string of at most 200 chars")
            if marker["max_encoded_length"] != 200:
                raise ValueError("max_encoded_length must be 200")
    if rule_ids != FIXED_RULE_IDS:
        raise ValueError(f"rule id set mismatch: {sorted(rule_ids)}")
    for control in raw["negative_controls"]:
        if not isinstance(control, dict) or set(control) != _NEGATIVE_CONTROL_KEYS:
            raise ValueError("negative control key set mismatch")
        if control["expected_action"] not in ("allow", "quarantine", "block"):
            raise ValueError(f"invalid expected_action: {control['expected_action']!r}")
    span = raw["span_selection"]
    if not isinstance(span, dict) or set(span) != _SPAN_SELECTION_KEYS:
        raise ValueError("span_selection key set mismatch")
    if span["dedup"] != "exact_overlap" or span["overlap_strategy"] != "merge" \
            or span["sort_order"] != "(start, end, rule_id)":
        raise ValueError("span_selection values mismatch")
    return ContextSecurityPolicy(raw)


# Immutable process-wide policy: loaded once, fail-closed, no request reload.
_POLICY_CACHE: ContextSecurityPolicy | None = None


def get_context_security_policy() -> ContextSecurityPolicy:
    """Return the immutable context-security policy, loading it exactly once."""
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        from app.config import get_settings

        _POLICY_CACHE = load_context_security_policy(
            get_settings().context_security_policy_path
        )
    return _POLICY_CACHE


def reset_context_security_policy_cache() -> None:
    """Test hook: forget the cached policy so the next call reloads strictly."""
    global _POLICY_CACHE
    _POLICY_CACHE = None


def _build_index_map(text: str) -> tuple[str, list[int]]:
    """NFKC + casefold normalize text, building an index map.

    Returns ``(normalized_text, index_map)`` where ``index_map[norm_pos]``
    gives the original-text position for each normalized character.
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


def _find_all(haystack: str, needle: str) -> list[int]:
    """Find all non-overlapping occurrences of needle in haystack."""
    positions = []
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + len(needle)
    return positions


def _translate_span(norm_start: int, norm_end: int, index_map: list[int]) -> tuple[int, int]:
    """Translate normalized-text offsets to original-text offsets."""
    if norm_start >= len(index_map):
        return (0, 0)
    orig_start = index_map[norm_start]
    if norm_end <= len(index_map):
        orig_end = index_map[norm_end - 1] + 1
    else:
        orig_end = index_map[-1] + 1
    return (orig_start, orig_end)


def _check_encoded_marker(
    rule: dict,
    original_text: str,
) -> list[tuple[int, int]]:
    """Check for encoded markers (base64/hex) in the ORIGINAL text.

    Encoded content is case-sensitive (base64/hex), so this scans the original
    text rather than the normalized version. The detector never executes
    decoded content; it only checks for pattern presence.
    """
    marker = rule.get("encoded_marker")
    if not marker:
        return []
    pattern = unicodedata.normalize("NFKC", rule["pattern"]).casefold()
    max_len = marker.get("max_encoded_length", 200)
    prefix = marker.get("marker_prefix", "")
    encoding = marker.get("encoding", "base64")
    spans = []

    # Build regex for encoded content on original text.
    prefix_escaped = re.escape(prefix)
    if encoding == "hex":
        char_class = r"[0-9a-fA-F]"
    else:
        char_class = r"[A-Za-z0-9+/=]"
    regex = re.compile(prefix_escaped + f"({char_class}{{2,{max_len}}})")

    for match in regex.finditer(original_text):
        encoded_str = match.group(1)
        decoded_variants: list[str] = []
        if encoding in ("base64", "base64_and_hex"):
            try:
                decoded_variants.append(
                    base64.b64decode(encoded_str).decode("utf-8", errors="ignore")
                )
            except Exception:
                pass
        if encoding in ("hex", "base64_and_hex"):
            try:
                decoded_variants.append(
                    bytes.fromhex(encoded_str).decode("utf-8", errors="ignore")
                )
            except Exception:
                pass
        for decoded in decoded_variants:
            if pattern in unicodedata.normalize("NFKC", decoded).casefold():
                spans.append((match.start(), match.end()))
                break
    return spans


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, list[str]]]:
    """Deduplicate by exact overlap, merge overlapping spans, sort by (start, end, rule_id)."""
    # Deduplicate exact (start, end, rule_id)
    seen = set()
    unique = []
    for start, end, rule_id in spans:
        key = (start, end, rule_id)
        if key not in seen:
            seen.add(key)
            unique.append((start, end, rule_id))

    # Sort by (start, end, rule_id)
    unique.sort()

    # Merge overlapping spans
    if not unique:
        return []
    merged = [(unique[0][0], unique[0][1], [unique[0][2]])]
    for start, end, rule_id in unique[1:]:
        last_start, last_end, last_rules = merged[-1]
        if start < last_end:  # overlap
            merged[-1] = (last_start, max(last_end, end), last_rules + [rule_id])
        else:
            merged.append((start, end, [rule_id]))
    return merged


def detect_context_injection(
    text: str,
    policy: ContextSecurityPolicy,
    chunk_id: int = 0,
) -> list[ContextSecurityResult]:
    """Detect prompt injection patterns in text.

    Returns one ``ContextSecurityResult`` per merged finding, sorted by
    ``(start, end, rule_id)``; an empty list means the text is allowed. Each
    finding carries its span's sorted unique rule IDs, the strictest action of
    those rules (block > quarantine > allow), and spans as original-text
    half-open offsets.
    """
    normalized_text, index_map = _build_index_map(text)

    all_spans: list[tuple[int, int, str, str]] = []

    for rule in policy.rules:
        pattern = policy.normalize_pattern(rule["pattern"])
        positions = _find_all(normalized_text, pattern)
        for pos in positions:
            norm_end = pos + len(pattern)
            orig_start, orig_end = _translate_span(pos, norm_end, index_map)
            all_spans.append((orig_start, orig_end, rule["rule_id"], rule["action"]))

        # Check encoded markers (on original text, not normalized).
        enc_spans = _check_encoded_marker(rule, text)
        for span in enc_spans:
            all_spans.append((span[0], span[1], rule["rule_id"], rule["action"]))

    if not all_spans:
        return []

    merged = _merge_spans([(s, e, rid) for s, e, rid, _ in all_spans])
    rule_action = {rid: action for _, _, rid, action in all_spans}

    results: list[ContextSecurityResult] = []
    for start, end, rule_ids in merged:
        action = "allow"
        if any(rule_action.get(rid) == "block" for rid in rule_ids):
            action = "block"
        elif any(rule_action.get(rid) == "quarantine" for rid in rule_ids):
            action = "quarantine"
        results.append(ContextSecurityResult(
            chunk_id=chunk_id,
            action=action,
            rule_ids=sorted(set(rule_ids)),
            matched_spans=[(start, end)],
            policy_version=policy.version,
        ))
    return results
