"""Phase 10D attack-corpus validation and (from Task 10D.2) simulation.

Task 10D.1 owns :func:`validate_attack_corpus`: structural validation
against the closed Draft 2020-12 schema plus the semantic invariants the
plan pins for the versioned attack corpus. The simulator modes arrive in
Task 10D.2; nothing here executes attacks.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_SCHEMA_PATH = _REPO_ROOT / "app" / "tests" / "fixtures" / "attack_payloads.schema.json"

# Closed control registry: control ID -> owning phase contract (plan 10D.1).
CONTROL_REGISTRY: dict[str, str] = {
    "retrieval_distance": "10B.3 distance",
    "retrieval_duplicate_cap": "10B.3 duplicate/caps",
    "source_trust": "10B.2 server trust",
    "context_injection": "10B.4",
    "sql_authority": "10A.6/10B.3 SQL hydration",
    "graph_evidence": "10A.2 evidence validation",
    "graph_identity": "10A.3 canonical identity",
    "context_caps": "10B.3 caps",
    "content_safety": "10C.1-10C.4",
}

# The hash-pinned corpus names the exact/near-duplicate-flooding category
# "duplicate_flooding" (the appendix spells the constant
# "exact_near_duplicate_flooding"); the pinned bytes are authoritative.
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

# Credential-shaped material must never appear in corpus texts.
_CREDENTIAL_PATTERNS = (
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bpassword\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bapi[_-]?key\s*[:=]\s*\S+"),
)


def _invalid(message: str) -> None:
    raise ValueError(f"attack corpus invalid: {message}")


def _check_structural(obj: Any) -> None:
    import jsonschema

    schema = json.loads(CORPUS_SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.Draft202012Validator(schema).validate(obj)
    except jsonschema.ValidationError as exc:
        _invalid(f"schema violation at {'/'.join(str(p) for p in exc.absolute_path)}: "
                 f"{exc.message}")


def _referenced_document_ids(fixture: dict) -> set[str]:
    scenario = fixture["scenario"]
    ids: set[str] = set(scenario.get("candidate_order", []))
    ids.update(scenario.get("l2_distances", {}).keys())
    ids.update(e["document_id"] for e in scenario.get("entities", []))
    for path_row in scenario.get("paths", []):
        ids.update(path_row["document_ids"])
        ids.add(path_row["evidence_document_id"])
    for key in ("chroma_id", "sql_document_id", "chroma_claimed_document_id",
                "injected_chunk_id"):
        if key in scenario:
            ids.add(scenario[key])
    evaluator = fixture["evaluator"]
    if "poisoned_document_id" in evaluator:
        ids.add(evaluator["poisoned_document_id"])
    return ids


def validate_attack_corpus(path) -> dict:
    """Validate the attack corpus at ``path``; return the parsed object.

    Raises ``ValueError`` on any structural (schema) or semantic
    violation: ordering, global ID uniqueness, reference resolution,
    category pairing, clean requirements, poison presence, severity
    pairing, the closed control registry, finite nonnegative L2
    distances, and absence of credential-shaped material.
    """
    corpus_path = Path(path)
    obj = json.loads(corpus_path.read_text(encoding="utf-8"))
    _check_structural(obj)

    fixtures = obj["fixtures"]

    fixture_ids = [f["id"] for f in fixtures]
    if fixture_ids != sorted(fixture_ids):
        _invalid("fixtures are not in lexical id order")
    if len(fixture_ids) != len(set(fixture_ids)):
        _invalid("duplicate fixture id")

    global_doc_ids: set[str] = set()
    from collections import Counter
    by_category: Counter = Counter()
    for fixture in fixtures:
        docs = fixture["documents"]
        doc_ids = [d["id"] for d in docs]
        if doc_ids != sorted(doc_ids):
            _invalid(f"fixture {fixture['id']}: documents not in lexical id order")
        for doc_id in doc_ids:
            if doc_id in global_doc_ids:
                _invalid(f"duplicate document id {doc_id!r} "
                         f"(fixture {fixture['id']})")
            global_doc_ids.add(doc_id)
        doc_map = {d["id"]: d for d in docs}

        for ref in sorted(_referenced_document_ids(fixture)):
            if ref not in doc_map:
                _invalid(f"fixture {fixture['id']}: scenario/evaluator references "
                         f"unknown document {ref!r}")

        by_category[(fixture["category"], fixture["kind"])] += 1

        poisoned = [d for d in docs if d["is_poisoned"]]
        if fixture["kind"] == "malicious" and not poisoned:
            _invalid(f"fixture {fixture['id']}: malicious fixture has no poisoned document")
        if fixture["kind"] == "benign" and poisoned:
            _invalid(f"fixture {fixture['id']}: benign fixture has a poisoned document")
        if fixture["kind"] == "malicious" and fixture["severity"] != 4:
            _invalid(f"fixture {fixture['id']}: malicious severity must be 4")
        if fixture["kind"] == "benign" and fixture["severity"] != 1:
            _invalid(f"fixture {fixture['id']}: benign severity must be 1")

        clean_ids = fixture["required_clean_document_ids"]
        if len(clean_ids) != 1:
            _invalid(f"fixture {fixture['id']}: exactly one required clean "
                     f"document required, found {len(clean_ids)}")
        for clean_id in clean_ids:
            if clean_id not in doc_map:
                _invalid(f"fixture {fixture['id']}: required clean document "
                         f"{clean_id!r} does not exist")
            if doc_map[clean_id]["is_poisoned"]:
                _invalid(f"fixture {fixture['id']}: required clean document "
                         f"{clean_id!r} is poisoned")

        for control_id in fixture["expected_control_ids"]:
            if control_id not in CONTROL_REGISTRY:
                _invalid(f"fixture {fixture['id']}: unknown control id "
                         f"{control_id!r}")
        if fixture["evaluator"]["type"] not in ALLOWED_EVALUATOR_TYPES:
            _invalid(f"fixture {fixture['id']}: evaluator type not allowed")

        for value in fixture["scenario"].get("l2_distances", {}).values():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                _invalid(f"fixture {fixture['id']}: L2 distance must be numeric")
            if not math.isfinite(value) or value < 0:
                _invalid(f"fixture {fixture['id']}: L2 distance must be "
                         f"finite and nonnegative, got {value!r}")

        for doc in docs:
            for pattern in _CREDENTIAL_PATTERNS:
                if pattern.search(doc["text"]):
                    _invalid(f"fixture {fixture['id']}: document {doc['id']} "
                             f"contains credential-shaped material")

    for category in sorted(REQUIRED_CATEGORIES):
        for kind in ("malicious", "benign"):
            count = by_category.get((category, kind), 0)
            if count != 1:
                _invalid(f"category {category!r} must contain exactly one "
                         f"{kind} fixture, found {count}")
    for category in sorted({c for c, _ in by_category} - REQUIRED_CATEGORIES):
        _invalid(f"unknown category {category!r}")

    return obj
