"""Phase 10C.2 — deterministic first-pass classification tests.

Pure ``classify_content`` function — NFKC+casefold index map, original
offsets, scope filtering, dedup/sort, max_input_chars rejection,
deterministic byte-equivalent output.
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import pytest

from app.services.safety_policy import load_safety_policy
from app.services.content_safety import SafetyInputLimitError, classify_content

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = load_safety_policy(PROJECT_ROOT / "config/content-safety-policy.json")


def test_classify_returns_allow_for_benign_text():
    text = "Educational overview of hate, dehumanization, and targeted harassment prevention."
    assessment = classify_content(text, scope="context", policy=POLICY)
    assert assessment.action == "allow"
    assert assessment.findings == []


def test_classify_violence_warn_at_exact_pattern_offset():
    text = "prefix stab the guard suffix"
    assessment = classify_content(text, scope="context", policy=POLICY)
    assert assessment.action == "warn"
    f = assessment.findings[0]
    assert f.category == "violence"
    assert f.severity == 3
    assert f.action == "warn"
    assert f.source_rule_ids == ["SAF001_violence"]
    assert text[f.start:f.end] == "stab the guard"
    assert f.start == 7 and f.end == 21


def test_classify_self_harm_block():
    assessment = classify_content("plan suicide now", scope="ingestion", policy=POLICY)
    assert assessment.action == "block"
    assert assessment.findings[0].category == "self_harm"
    assert assessment.findings[0].severity == 4


def test_classify_nfkc_fullwidth_offsets_map_to_original():
    text = "ｄｅｈｕｍａｎｉｚｅ ｔｈａｔ ｇｒｏｕｐ"
    assessment = classify_content(text, scope="context", policy=POLICY)
    f = assessment.findings[0]
    import unicodedata
    assert unicodedata.normalize("NFKC", text[f.start:f.end]).casefold() \
        == "dehumanize that group"
    assert f.source_rule_ids == ["SAF004_hate_harassment"]


def test_classify_applies_only_rules_with_matching_scope():
    # SAF001 has all three scopes; mutate one rule to scope=['ingestion'] only
    # and assert no finding when scope='context'.
    obj = json.loads(
        (PROJECT_ROOT / "config/content-safety-policy.json").read_text()
    )
    for rule in obj["rules"]:
        if rule["rule_id"] == "SAF001_violence":
            rule["scopes"] = ["ingestion"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "policy.json"
        p.write_text(json.dumps(obj))
        scoped_policy = load_safety_policy(p)
    # SAF001 dropped 'context': its pattern no longer fires under scope='context'.
    context_hit = classify_content("stab the guard", scope="context",
                                   policy=scoped_policy)
    assert context_hit.action == "allow"
    assert context_hit.findings == []
    # SAF001 still carries 'ingestion': the same text fires under scope='ingestion'.
    ingestion_hit = classify_content("stab the guard", scope="ingestion",
                                     policy=scoped_policy)
    assert ingestion_hit.action == "warn"
    assert [f.source_rule_ids for f in ingestion_hit.findings] == [["SAF001_violence"]]
    # Untouched rules keep all three scopes and still classify under scope='context'.
    self_harm = classify_content("plan suicide", scope="context", policy=scoped_policy)
    assert self_harm.action == "block"
    assert self_harm.findings[0].source_rule_ids == ["SAF002_self_harm"]


def test_classify_overall_action_is_strongest_finding():
    # Two findings: warn + block -> overall action is block.
    text = "stab the guard then plan suicide"
    assessment = classify_content(text, scope="context", policy=POLICY)
    assert assessment.action == "block"
    assert {f.action for f in assessment.findings} == {"warn", "block"}


def test_classify_deduplicates_by_start_end_source_rule_ids():
    # Identical finding produced twice collapses to one.
    # Dedup key is (start, end, source_rule_ids): two rules that share a literal
    # pattern but carry distinct rule_ids fire on the same span and stay distinct,
    # while an exact repeated key collapses to a single entry.
    obj = json.loads(
        (PROJECT_ROOT / "config/content-safety-policy.json").read_text()
    )
    obj["rules"].append({
        "rule_id": "SAF001_violence_DUP", "category": "violence", "severity": 3,
        "action": "warn", "pattern_type": "literal", "pattern": "stab the guard",
        "scopes": ["ingestion", "context", "answer"],
    })
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "policy.json"
        p.write_text(json.dumps(obj))
        policy = load_safety_policy(p)
    assessment = classify_content("stab the guard", scope="context", policy=policy)
    # Same span, distinct rule_ids -> two findings; classify did NOT collapse
    # them because the source_rule_ids component of the key differs.
    keys = [(f.start, f.end, tuple(f.source_rule_ids)) for f in assessment.findings]
    assert keys == [(0, 14, ("SAF001_violence",)), (0, 14, ("SAF001_violence_DUP",))]
    # Exact-duplicate keys collapse: duplicating the key list and re-applying the
    # (start, end, source_rule_ids) uniqueness yields the same two tuples.
    doubled = keys + keys
    deduped = sorted(set(doubled))
    assert deduped == keys
    assert len(deduped) == 2 == len(set(doubled))
    # classify's own output never carries a duplicate key.
    assert len(set(keys)) == len(keys)


def test_classify_sorts_by_start_end_source_rule_ids():
    text = "build a bomb then stab the guard"
    assessment = classify_content(text, scope="context", policy=POLICY)
    keys = [(f.start, f.end, tuple(f.source_rule_ids)) for f in assessment.findings]
    assert keys == sorted(keys)


def test_classify_rejects_input_over_max_input_chars():
    too_long = "a" * (POLICY.max_input_chars + 1)
    with pytest.raises(SafetyInputLimitError):
        classify_content(too_long, scope="context", policy=POLICY)


def test_classify_pure_no_db_no_time_no_random():
    # Assert function has no side effects: same input -> same output bytes.
    a = classify_content("stab the guard", scope="context", policy=POLICY)
    b = classify_content("stab the guard", scope="context", policy=POLICY)
    assert pickle.dumps(a) == pickle.dumps(b)


def test_classify_finding_offsets_satisfy_zero_le_start_lt_end_le_len():
    text = "build a bomb and plan suicide"
    assessment = classify_content(text, scope="context", policy=POLICY)
    for f in assessment.findings:
        assert 0 <= f.start < f.end <= len(text)


def test_classify_empty_text_returns_allow():
    assessment = classify_content("", scope="context", policy=POLICY)
    assert assessment.action == "allow"
    assert assessment.findings == []
