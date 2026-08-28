"""Phase 10B.2 — provenance/trust assessment tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.services.provenance import (
    ProvenanceAssessment,
    SourceTrustPolicy,
    assess_provenance,
    compute_grounding_score,
    content_sha256,
    load_source_trust_policy,
    provenance_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "config" / "source-trust-policy.json"

EXPECTED_POLICY_BYTES = 288
EXPECTED_POLICY_SHA256 = (
    "b61f58f519f0c67c7ac7820417c055329fed7974d3ea94bee4d361299b6a979a"
)


_POLICY_RAW = {
    "version": "source-trust-v1",
    "default": {"tier": "untrusted", "score": 0.2},
    "rules": [
        {"rule_id": "SRC001", "source": "operator-curated", "tier": "trusted", "score": 1.0, "requires_operator": True},
        {"rule_id": "SRC002", "source": "blocked-source", "tier": "blocked", "score": 0.0, "requires_operator": False},
    ],
}


@pytest.fixture
def policy():
    return SourceTrustPolicy(_POLICY_RAW)


def test_default_trust_for_unmatched_source(policy):
    tier, score = policy.assess("unknown-source", is_operator=False)
    assert tier == "untrusted"
    assert score == 0.2


def test_default_trust_for_no_source(policy):
    tier, score = policy.assess(None, is_operator=False)
    assert tier == "untrusted"
    assert score == 0.2


def test_trusted_source_requires_operator(policy):
    tier, score = policy.assess("operator-curated", is_operator=False)
    assert tier == "untrusted"
    assert score == 0.2


def test_trusted_source_with_operator(policy):
    tier, score = policy.assess("operator-curated", is_operator=True)
    assert tier == "trusted"
    assert score == 1.0


def test_blocked_source_rejected(policy):
    assert policy.is_blocked("blocked-source")
    assert not policy.is_blocked("unknown-source")


def test_grounding_score_all_factors_present():
    score, reasons = compute_grounding_score(
        vector_id_matches_sql=True,
        text_hash_matches=True,
        document_ready=True,
        has_ingestion_origin=True,
        has_graph_evidence=True,
    )
    assert score == 1.0
    assert len(reasons) == 5
    assert all(r.startswith("grounding_") for r in reasons)


def test_grounding_score_no_factors():
    score, reasons = compute_grounding_score(
        vector_id_matches_sql=False,
        text_hash_matches=False,
        document_ready=False,
        has_ingestion_origin=False,
        has_graph_evidence=False,
    )
    assert score == 0.0
    assert len(reasons) == 5


def test_grounding_score_partial():
    score, reasons = compute_grounding_score(
        vector_id_matches_sql=True,
        text_hash_matches=True,
        document_ready=True,
        has_ingestion_origin=False,
        has_graph_evidence=False,
    )
    assert score == pytest.approx(0.70)


def test_provenance_score_formula():
    score = provenance_score(trust_score=1.0, grounding_score=1.0)
    assert score == 1.0
    score = provenance_score(trust_score=0.2, grounding_score=0.0)
    assert score == pytest.approx(0.12)


def test_content_sha256_is_deterministic():
    h1 = content_sha256("hello world")
    h2 = content_sha256("hello world")
    assert h1 == h2
    assert len(h1) == 64


def test_policy_load_validates_trusted_requires_operator(tmp_path):
    import json
    bad = dict(_POLICY_RAW)
    bad["rules"] = [
        {"rule_id": "X1", "source": "trusted-no-op", "tier": "trusted", "score": 1.0, "requires_operator": False},
    ]
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="requires_operator"):
        load_source_trust_policy(str(p))


def test_policy_load_rejects_duplicate_source(tmp_path):
    import json
    bad = dict(_POLICY_RAW)
    bad["rules"] = [
        {"rule_id": "X1", "source": "dup", "tier": "untrusted", "score": 0.2, "requires_operator": False},
        {"rule_id": "X2", "source": "dup", "tier": "untrusted", "score": 0.2, "requires_operator": False},
    ]
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="duplicate source"):
        load_source_trust_policy(str(p))


# ---------------------------------------------------------------------------
# Committed-policy artifact contract (appendix 10B.2): the in-repo policy is
# the immutable-image artifact whose canonical bytes are pinned by the plan.

def test_source_trust_policy_byte_count_is_exactly_288():
    raw = POLICY_PATH.read_bytes()
    assert len(raw) == EXPECTED_POLICY_BYTES


def test_source_trust_policy_sha256_matches_pinned_hash():
    raw = POLICY_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_POLICY_SHA256


def test_source_trust_policy_canonical_form():
    obj = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert obj["version"] == "source-trust-v1"
    assert obj["default"] == {"tier": "untrusted", "score": 0.2}
    rules = {r["rule_id"]: r for r in obj["rules"]}
    assert rules["SRC001"] == {
        "rule_id": "SRC001", "source": "operator-curated",
        "tier": "trusted", "score": 1.0, "requires_operator": True,
    }
    assert rules["SRC002"] == {
        "rule_id": "SRC002", "source": "blocked-source",
        "tier": "blocked", "score": 0.0, "requires_operator": False,
    }
    # Canonical serialization: sorted-key minified + final LF reproduces the
    # exact committed bytes.
    canonical = (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    assert canonical == POLICY_PATH.read_bytes()


def test_committed_policy_loads_strict_object():
    loaded = load_source_trust_policy(str(POLICY_PATH))
    assert loaded.version == "source-trust-v1"
    assert len(loaded.rules) == 2


# ---------------------------------------------------------------------------
# assess_provenance: deterministic composition of policy trust and the
# five-component grounding formula.

def _assess(policy, **overrides):
    kwargs = dict(
        document_id=1,
        chunk_id=1,
        source="wiki",
        operator_authenticated=False,
        vector_id_match=True,
        sql_text_hash_match=True,
        document_ready=True,
        ingestion_origin="api",
        has_graph_evidence=False,
        policy_version="source-trust-v1",
        policy=policy,
    )
    kwargs.update(overrides)
    return assess_provenance(**kwargs)


def test_same_sql_chunk_receives_same_score_under_one_policy_version(policy):
    a1 = _assess(policy)
    a2 = _assess(policy)
    assert a1.provenance_score == a2.provenance_score
    assert a1.trust_tier == a2.trust_tier
    assert a1.grounding_score == a2.grounding_score
    assert a1.reasons == a2.reasons


def test_reasons_are_sorted_stable_codes(policy):
    assessment = _assess(policy)
    assert assessment.reasons == sorted(assessment.reasons)
    assert all(code.startswith("grounding_") for code in assessment.reasons)
    # Every component reports exactly one contributing-or-missing code.
    assert len(assessment.reasons) == 5


def test_assess_provenance_composes_policy_trust_and_grounding(policy):
    assessment = _assess(policy)
    # Unmatched public source: default tier/score from the committed policy.
    assert assessment.trust_tier == "untrusted"
    assert assessment.trust_score == 0.2
    # vector(+0.25) + text hash(+0.25) + ready(+0.20) + origin(+0.15) = 0.85
    assert assessment.grounding_score == pytest.approx(0.85)
    assert assessment.provenance_score == round(0.6 * 0.2 + 0.4 * 0.85, 6)


# ---------------------------------------------------------------------------
# Model field validation (R4): scores are bounded to [0,1].

def test_provenance_assessment_rejects_invalid_trust_tier():
    with pytest.raises(ValidationError):
        ProvenanceAssessment(document_id=1, chunk_id=1, trust_tier="god-mode",
                             trust_score=0.5, grounding_score=0.5, reasons=[],
                             policy_version="source-trust-v1")


def test_provenance_assessment_rejects_out_of_range_trust_score():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            ProvenanceAssessment(document_id=1, chunk_id=1, trust_tier="untrusted",
                                 trust_score=bad, grounding_score=0.5, reasons=[],
                                 policy_version="source-trust-v1")


def test_provenance_assessment_rejects_out_of_range_grounding_score():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            ProvenanceAssessment(document_id=1, chunk_id=1, trust_tier="untrusted",
                                 trust_score=0.5, grounding_score=bad, reasons=[],
                                 policy_version="source-trust-v1")


@pytest.mark.parametrize("trust,grounding", [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
def test_provenance_assessment_accepts_boundary_scores(trust, grounding):
    assessment = ProvenanceAssessment(
        document_id=1, chunk_id=1, trust_tier="untrusted",
        trust_score=trust, grounding_score=grounding, reasons=[],
        policy_version="source-trust-v1")
    assert assessment.trust_score == trust
    assert assessment.grounding_score == grounding
    assert assessment.provenance_score == round(0.6 * trust + 0.4 * grounding, 6)
