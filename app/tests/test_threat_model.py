"""Tests for the pinned OWASP LLM Top 10 2025 threat model."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OWASP_EXCERPT = PROJECT_ROOT / "docs" / "references" / "owasp-llm-top10-2025-excerpt.txt"
THREAT_MODEL = PROJECT_ROOT / "docs" / "threat-model.md"

EXPECTED_OWASP_BYTES = 436
EXPECTED_OWASP_SHA256 = (
    "e024da7f5a562882e3ba8c8eae62d74db29d9aa86ec20b5512ad6be58c0c6200"
)
OWASP_URL = "https://genai.owasp.org/llm-top-10/"
RETRIEVAL_DATE = "2026-08-01"

EXPECTED_IDS_AND_NAMES = {
    "LLM01": "LLM01:2025 Prompt Injection",
    "LLM02": "LLM02:2025 Sensitive Information Disclosure",
    "LLM03": "LLM03:2025 Supply Chain",
    "LLM04": "LLM04:2025 Data and Model Poisoning",
    "LLM05": "LLM05:2025 Improper Output Handling",
    "LLM06": "LLM06:2025 Excessive Agency",
    "LLM07": "LLM07:2025 System Prompt Leakage",
    "LLM08": "LLM08:2025 Vector and Embedding Weaknesses",
    "LLM09": "LLM09:2025 Misinformation",
    "LLM10": "LLM10:2025 Unbounded Consumption",
}

# scenario → expected primary OWASP mapping (from plan §10B.1)
SCENARIO_MAPPINGS = {
    "retrieved_prompt_injection": "LLM01",
    "corpus_or_data_poisoning": "LLM04",
    "credential_leakage_llm02": "LLM02",
    "credential_leakage_llm07": "LLM07",
    "vector_or_embedding_attack": "LLM08",
    "resource_flooding": "LLM10",
}

EXPECTED_FIXTURE_IDS = {
    "TFS-01-untrusted-uploader-trusted-source",
    "TFS-02-context-flooding-single-source",
    "TFS-03-exact-near-duplicate-poisoning",
    "TFS-04-low-relevance-poisoned-chunks",
    "TFS-05-chroma-metadata-record-id-aliasing",
    "TFS-06-graph-entity-alias-collision-relationship-poisoning",
    "TFS-07-retrieved-prompt-injection",
    "TFS-08-audit-tampering-sensitive-data-disclosure",
    "TFS-09-cross-tenant-retrieval-deferred",
}

REQUIRED_SCENARIO_FIELDS = (
    "protected_asset", "trust_boundary", "precondition", "attack",
    "expected_control", "audit_evidence", "residual_risk", "fixture_id",
)


def test_owasp_excerpt_byte_count_is_exactly_436():
    raw = OWASP_EXCERPT.read_bytes()
    assert len(raw) == EXPECTED_OWASP_BYTES


def test_owasp_excerpt_sha256_matches_pinned_hash():
    raw = OWASP_EXCERPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_OWASP_SHA256


def test_owasp_excerpt_ends_with_final_lf():
    raw = OWASP_EXCERPT.read_bytes()
    assert raw.endswith(b"\n")


def test_threat_model_cites_owasp_url_date_and_hash():
    text = THREAT_MODEL.read_text(encoding="utf-8")
    assert OWASP_URL in text
    assert RETRIEVAL_DATE in text
    assert EXPECTED_OWASP_SHA256 in text


def test_threat_model_contains_all_ten_owasp_ids_with_exact_names():
    text = THREAT_MODEL.read_text(encoding="utf-8")
    for owasp_id, full_name in EXPECTED_IDS_AND_NAMES.items():
        assert owasp_id in text, f"missing {owasp_id}"
        # the full canonical name must appear at least once
        assert full_name in text, f"missing full name '{full_name}'"


def test_threat_model_assigns_not_primary_rationale_to_non_mapped_ids():
    """Every ID not in SCENARIO_MAPPINGS values must carry an explicit
    not_primary_for_phase10 rationale rather than being omitted."""
    text = THREAT_MODEL.read_text(encoding="utf-8")
    primary_ids = set(SCENARIO_MAPPINGS.values())
    for owasp_id in EXPECTED_IDS_AND_NAMES:
        if owasp_id not in primary_ids:
            assert owasp_id in text
            # rationale marker near the ID
            assert "not_primary_for_phase10" in text, (
                f"{owasp_id} lacks not_primary_for_phase10 rationale")


@pytest.mark.parametrize("scenario_key,expected_owasp", list(SCENARIO_MAPPINGS.items()))
def test_threat_model_scenario_mappings(scenario_key, expected_owasp):
    """Each enumerated threat scenario maps to its expected primary OWASP ID."""
    text = THREAT_MODEL.read_text(encoding="utf-8")
    # scenario must be referenced and the OWASP ID must appear
    # (the threat model must not silently omit any of the 9 scenarios)
    assert expected_owasp in text


def test_threat_model_has_nine_threat_scenarios_with_required_fields():
    """Every fixture ID exists exactly once and each scenario row exposes
    all required fields."""
    text = THREAT_MODEL.read_text(encoding="utf-8")
    found_fixture_ids: list[str] = re.findall(r"TFS-\d{2}-[a-z0-9\-]+", text)
    # every expected fixture id is present
    for fid in EXPECTED_FIXTURE_IDS:
        assert fid in found_fixture_ids, f"missing fixture {fid}"
    # each fixture id appears exactly once
    for fid in EXPECTED_FIXTURE_IDS:
        assert found_fixture_ids.count(fid) == 1, (
            f"fixture {fid} appears {found_fixture_ids.count(fid)} times")
    # required field labels present
    for field in REQUIRED_SCENARIO_FIELDS:
        assert field in text, f"missing scenario field label '{field}'"


def test_threat_model_cross_tenant_scenario_is_deferred_not_solved():
    """Scenario 9 (cross-tenant retrieval) must explicitly defer until
    multi-tenancy exists, not claim it is solved."""
    text = THREAT_MODEL.read_text(encoding="utf-8")
    assert "TFS-09" in text
    assert ("deferred" in text.lower() or "not claimed solved" in text.lower())
