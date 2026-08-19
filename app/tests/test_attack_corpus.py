"""Phase 10D.1 — versioned attack corpus tests.

Covers: pinned corpus bytes/hash, schema validation, semantic invariants,
closed control registry, lexical ordering, exactly-one malicious/benign per
category, severity pairing.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

CORPUS_PATH = Path("app/tests/fixtures/attack_payloads.json")
SCHEMA_PATH = Path("app/tests/fixtures/attack_payloads.schema.json")
CORPUS_BYTES_EXPECTED = 17757
CORPUS_SHA256_EXPECTED = (
    "bca0c8eed73c02a346b92658b9545620aa4565e5decb8befc6a74f541f8d03ce"
)
# The hash-pinned corpus names the exact/near-duplicate-flooding category
# "duplicate_flooding"; the appendix spells the constant
# "exact_near_duplicate_flooding". The pinned bytes are authoritative.
REQUIRED_CATEGORIES = {
    "vector_poisoning", "low_relevance_poisoning", "duplicate_flooding",
    "source_spoofing", "retrieved_prompt_injection", "metadata_aliasing",
    "graph_relationship_poisoning", "entity_alias_collision",
    "context_flooding", "safety_filter_evasion",
}
ALLOWED_EVALUATOR_TYPES = {
    "poisoned_chunk_selected", "answer_contains_literal",
    "graph_path_contains_poisoned_evidence", "unsafe_answer_returned",
    "control_not_triggered",
}
CLOSED_CONTROL_REGISTRY = {
    "retrieval_distance", "retrieval_duplicate_cap", "source_trust",
    "context_injection", "sql_authority", "graph_evidence",
    "graph_identity", "context_caps", "content_safety",
}
SEED = 42

_CORPUS_OBJECT = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _mutate_corpus(mutation: str) -> dict:
    """Return a semantically invalid corpus derived from the real one."""
    bad = copy.deepcopy(_CORPUS_OBJECT)
    fixtures = bad["fixtures"]
    first = fixtures[0]
    if mutation == "duplicate_fixture_id":
        # Insert the clone adjacent to the original so lexical order is
        # preserved and the duplicate-id rule (not the order rule) fires.
        clone = copy.deepcopy(first)
        clone["description"] = clone["description"] + " duplicate"
        fixtures.insert(fixtures.index(first) + 1, clone)
    elif mutation == "duplicate_document_id":
        clone = copy.deepcopy(first["documents"][0])
        clone["text"] = clone["text"] + " different text"
        first["documents"].append(clone)
    elif mutation == "unknown_control_id":
        first["expected_control_ids"] = ["quantum_remediation"]
    elif mutation == "missing_expected_safe_outcome":
        del first["expected_safe_outcome"]
    elif mutation == "real_secret_in_text":
        first["documents"][0]["text"] = (
            "leaked credential AKIAIOSFODNN7EXAMPLE in text")
    elif mutation == "malicious_without_poison":
        malicious = next(f for f in fixtures if f["kind"] == "malicious")
        for doc in malicious["documents"]:
            doc["is_poisoned"] = False
    elif mutation == "benign_with_poison":
        benign = next(f for f in fixtures if f["kind"] == "benign")
        benign["documents"][0]["is_poisoned"] = True
    elif mutation == "wrong_severity_for_kind":
        benign = next(f for f in fixtures if f["kind"] == "benign")
        benign["severity"] = 4
    elif mutation == "evaluator_type_specific_param_mismatch":
        poisoned_eval = next(
            f for f in fixtures
            if f["evaluator"]["type"] == "poisoned_chunk_selected")
        del poisoned_eval["evaluator"]["poisoned_document_id"]
        poisoned_eval["evaluator"]["literal"] = "OVERRIDE_ACCEPTED"
    elif mutation == "scenario_type_param_mismatch":
        vector_eval = next(
            f for f in fixtures if f["scenario"]["type"] == "vector_poisoning")
        del vector_eval["scenario"]["l2_distances"]
        vector_eval["scenario"]["per_source_cap"] = 2
    elif mutation == "unknown_top_level_key":
        bad["extra"] = 1
    elif mutation == "missing_required_clean_document_ids":
        first["required_clean_document_ids"] = []
    elif mutation == "non_lexical_order":
        fixtures[0], fixtures[1] = fixtures[1], fixtures[0]
    elif mutation == "scenario_unknown_type":
        vector_eval = next(
            f for f in fixtures if f["scenario"]["type"] == "vector_poisoning")
        vector_eval["scenario"]["type"] = "zzz_unknown"
    elif mutation == "scenario_missing_type":
        vector_eval = next(
            f for f in fixtures if f["scenario"]["type"] == "vector_poisoning")
        del vector_eval["scenario"]["type"]
    elif mutation == "evaluator_missing_type":
        poisoned_eval = next(
            f for f in fixtures
            if f["evaluator"]["type"] == "poisoned_chunk_selected")
        del poisoned_eval["evaluator"]["type"]
    elif mutation == "duplicate_category_kind":
        # Second malicious fixture in an already-paired category with every
        # internal reference remapped to the cloned ids, inserted in lexical
        # position: exactly-one-per-kind must be the rule that fires.
        clone = copy.deepcopy(next(
            f for f in fixtures
            if f["kind"] == "malicious" and f["category"] == "vector_poisoning"))
        clone["id"] = clone["id"] + "-second"
        clone["description"] = clone["description"] + " second"

        def _suffixed(doc_id: str) -> str:
            return doc_id + "-second"

        for doc in clone["documents"]:
            doc["id"] = _suffixed(doc["id"])
        clone["required_clean_document_ids"] = [
            _suffixed(d) for d in clone["required_clean_document_ids"]]
        clone["evaluator"]["poisoned_document_id"] = _suffixed(
            clone["evaluator"]["poisoned_document_id"])
        scenario = clone["scenario"]
        if "candidate_order" in scenario:
            scenario["candidate_order"] = [
                _suffixed(d) for d in scenario["candidate_order"]]
        if "l2_distances" in scenario:
            scenario["l2_distances"] = {
                _suffixed(d): v for d, v in scenario["l2_distances"].items()}
        pos = next((i for i, f in enumerate(fixtures) if f["id"] > clone["id"]),
                   len(fixtures))
        fixtures.insert(pos, clone)
    elif mutation == "two_clean_documents":
        clean = first["documents"][0]
        extra = copy.deepcopy(clean)
        extra["id"] = clean["id"] + "-extra"
        first["documents"].append(extra)
        first["required_clean_document_ids"] = [clean["id"], extra["id"]]
    else:  # pragma: no cover - parametrization exhausts the list
        raise AssertionError(f"unknown mutation {mutation!r}")
    return bad


def test_corpus_byte_count_and_sha256_immutable():
    raw = CORPUS_PATH.read_bytes()
    assert len(raw) == CORPUS_BYTES_EXPECTED
    assert raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA256_EXPECTED


def test_corpus_schema_valid_against_draft2020_12():
    import jsonschema
    schema = json.loads(SCHEMA_PATH.read_text())
    obj = json.loads(CORPUS_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(obj)


def test_corpus_fixture_and_document_ids_unique():
    obj = json.loads(CORPUS_PATH.read_text())
    fix_ids = [f["id"] for f in obj["fixtures"]]
    assert len(fix_ids) == len(set(fix_ids))
    doc_ids = [d["id"] for f in obj["fixtures"] for d in f["documents"]]
    assert len(doc_ids) == len(set(doc_ids))


def test_corpus_fixtures_in_lexical_id_order():
    obj = json.loads(CORPUS_PATH.read_text())
    ids = [f["id"] for f in obj["fixtures"]]
    assert ids == sorted(ids)


def test_corpus_documents_in_lexical_id_order_within_fixture():
    obj = json.loads(CORPUS_PATH.read_text())
    for fix in obj["fixtures"]:
        doc_ids = [d["id"] for d in fix["documents"]]
        assert doc_ids == sorted(doc_ids)


def test_corpus_contains_every_required_category():
    obj = json.loads(CORPUS_PATH.read_text())
    cats = {f["category"] for f in obj["fixtures"]}
    assert REQUIRED_CATEGORIES.issubset(cats)


def test_each_category_has_one_malicious_and_one_benign_fixture():
    obj = json.loads(CORPUS_PATH.read_text())
    from collections import defaultdict
    by_cat = defaultdict(set)
    for f in obj["fixtures"]:
        by_cat[f["category"]].add(f["kind"])
    for cat, kinds in by_cat.items():
        if cat in REQUIRED_CATEGORIES:
            assert kinds == {"malicious", "benign"}, f"{cat}: {kinds}"


def test_malicious_fixture_has_poisoned_document_benign_has_none():
    obj = json.loads(CORPUS_PATH.read_text())
    for f in obj["fixtures"]:
        poisoned = [d for d in f["documents"] if d["is_poisoned"]]
        if f["kind"] == "malicious":
            assert poisoned, f"{f['id']} has no poisoned document"
        else:
            assert not poisoned, f"{f['id']} has a poisoned document"


def test_severity_4_for_malicious_1_for_benign():
    obj = json.loads(CORPUS_PATH.read_text())
    for f in obj["fixtures"]:
        if f["kind"] == "malicious":
            assert f["severity"] == 4
        else:
            assert f["severity"] == 1


def test_every_expected_control_id_in_closed_registry():
    obj = json.loads(CORPUS_PATH.read_text())
    for f in obj["fixtures"]:
        for cid in f["expected_control_ids"]:
            assert cid in CLOSED_CONTROL_REGISTRY


def test_evaluator_type_in_allowed_set():
    obj = json.loads(CORPUS_PATH.read_text())
    for f in obj["fixtures"]:
        assert f["evaluator"]["type"] in ALLOWED_EVALUATOR_TYPES


def test_required_clean_document_ids_unique_per_fixture():
    obj = json.loads(CORPUS_PATH.read_text())
    for f in obj["fixtures"]:
        ids = f["required_clean_document_ids"]
        assert len(ids) == len(set(ids))


def test_required_clean_documents_exist_and_not_poisoned():
    obj = json.loads(CORPUS_PATH.read_text())
    for f in obj["fixtures"]:
        doc_map = {d["id"]: d for d in f["documents"]}
        for rcid in f["required_clean_document_ids"]:
            assert rcid in doc_map
            assert doc_map[rcid]["is_poisoned"] is False


@pytest.mark.parametrize("mutation", [
    "duplicate_fixture_id", "duplicate_document_id", "unknown_control_id",
    "missing_expected_safe_outcome", "real_secret_in_text",
    "malicious_without_poison", "benign_with_poison",
    "wrong_severity_for_kind", "evaluator_type_specific_param_mismatch",
    "scenario_type_param_mismatch", "unknown_top_level_key",
    "missing_required_clean_document_ids", "non_lexical_order",
    # Regression additions beyond the appendix-pinned thirteen:
    "scenario_unknown_type", "scenario_missing_type",
    "evaluator_missing_type", "duplicate_category_kind",
    "two_clean_documents",
])
def test_corpus_rejects_each_semantic_invalid_mutation(mutation, tmp_path):
    bad = _mutate_corpus(mutation)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    from app.services.attack_simulator import validate_attack_corpus
    with pytest.raises(ValueError):
        validate_attack_corpus(p)


def test_seed_is_42():
    obj = json.loads(CORPUS_PATH.read_text())
    # seed may be at top level or inside scenario; assert 42 somewhere
    blob = json.dumps(obj)
    assert '"seed": 42' in blob or '"seed":42' in blob
