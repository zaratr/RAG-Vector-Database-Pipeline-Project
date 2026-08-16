"""Phase 10B.2 — provenance/trust assessment tests."""
from __future__ import annotations

import pytest
from app.services.provenance import (
    ProvenanceAssessment,
    SourceTrustPolicy,
    compute_grounding_score,
    content_sha256,
    load_source_trust_policy,
    provenance_score,
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
