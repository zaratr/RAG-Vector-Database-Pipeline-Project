"""Phase 10B remediation — persisted provenance scoring (R6).

The persisted ``provenance_score`` must use the documented five-component
grounding formula (plan L1092): +0.25 canonical vector ID matched SQL, +0.25
authoritative SQL text hash matched, +0.20 ready document, +0.15 nonempty
server-recorded ingestion origin, +0.15 graph evidence with valid FKs
(graph/hybrid only); ``provenance_score = round(0.6*trust + 0.4*grounding, 6)``.
The pre-remediation persisted path credited only 0.20/0.25/0.25 (max 0.70).
"""
from __future__ import annotations

import pytest

from app.services.retrieval_security import (
    Candidate,
    RetrievalSecurityPolicy,
    apply_security_filters,
)

from app.services.provenance import provenance_score


def _policy(**overrides):
    values = dict(
        version="retrieval-security-v1",
        metric="l2",
        max_distance=1.0,
        per_source_cap=5,
        per_document_cap=5,
        max_candidates=50,
        near_duplicate_jaccard=0.90,
        calibration_fixture_sha256="0" * 64,
        calibration_clean_recall=1.0,
        calibration_poison_share=0.0,
        calibration_tool_version="calibrate-v1",
    )
    values.update(overrides)
    return RetrievalSecurityPolicy(**values)


def _candidate(chunk_id=1, *, tier="untrusted", ready=True, origin="",
               graph_evidence=False, text="alpha", source="s1"):
    return Candidate(
        chunk_id=chunk_id, document_id=chunk_id, source=source, text=text,
        native_score=0.10, trust_tier=tier, document_ready=ready,
        origin=origin or "vector", distance=0.10,
        ingestion_origin=origin,
        has_graph_evidence=graph_evidence,
    )


def _selected_score(candidates, policy=None, mode="vector"):
    decisions = apply_security_filters(candidates, policy=policy or _policy(),
                                       mode=mode)
    selected = [d for d in decisions if d.decision == "selected"]
    assert len(selected) == 1
    return selected[0]


# ---------------------------------------------------------------------------
# Each component's contribution, pinned individually.

def test_base_hydrated_identity_components_contribute_half():
    """Vector-ID and SQL-text-hash components alone ground at 0.50 (they hold
    by construction for every SQL-hydrated candidate)."""
    decision = _selected_score([
        _candidate(ready=False, origin="", graph_evidence=False)])
    assert decision.provenance_score == provenance_score(0.2, 0.50)


def test_ready_document_component_contributes_twenty_points():
    decision = _selected_score([
        _candidate(ready=True, origin="", graph_evidence=False)])
    assert decision.provenance_score == provenance_score(0.2, 0.70)


def test_ingestion_origin_component_contributes_fifteen_points():
    decision = _selected_score([
        _candidate(ready=False, origin="api", graph_evidence=False)])
    assert decision.provenance_score == provenance_score(0.2, 0.65)


def test_graph_evidence_component_contributes_fifteen_points():
    decision = _selected_score([
        _candidate(ready=False, origin="", graph_evidence=True)])
    assert decision.provenance_score == provenance_score(0.2, 0.65)


def test_all_five_components_present_ground_at_one():
    decision = _selected_score([
        _candidate(ready=True, origin="api", graph_evidence=True)])
    # trust 0.2 (untrusted) → round(0.6*0.2 + 0.4*1.0, 6)
    assert decision.provenance_score == provenance_score(0.2, 1.0)
    assert decision.provenance_score == pytest.approx(0.52)

    trusted = _selected_score([
        _candidate(tier="trusted", ready=True, origin="operator-api",
                   graph_evidence=True)])
    assert trusted.provenance_score == 1.0


def test_documented_partial_combinations():
    """Documented partial combinations of the five components."""
    cases = [
        # (ready, origin, graph) → grounding
        (True, "", False, 0.70),   # pre-remediation maximum (defect value)
        (True, "api", False, 0.85),  # vector-only production lane
        (True, "api", True, 1.00),   # graph/hybrid production lane
        (False, "", False, 0.50),    # identity-only
        (True, "", True, 0.85),
        (False, "api", True, 0.80),
    ]
    for ready, origin, graph, expected_grounding in cases:
        decision = _selected_score([
            _candidate(ready=ready, origin=origin, graph_evidence=graph)])
        assert decision.provenance_score == provenance_score(0.2, expected_grounding), (
            ready, origin, graph)


def test_trust_weight_is_sixty_percent_of_the_final_score():
    """0.6 trust / 0.4 grounding split documented by the plan."""
    for tier, trust in (("trusted", 1.0), ("standard", 0.5), ("untrusted", 0.2)):
        decision = _selected_score([
            _candidate(tier=tier, ready=True, origin="api")])
        assert decision.provenance_score == provenance_score(trust, 0.85)


def test_rejected_candidates_carry_zero_provenance_score():
    policy = _policy(max_distance=0.05)
    decisions = apply_security_filters(
        [_candidate(1), _candidate(2, text="alpha")], policy=policy, mode="vector")
    by_chunk = {d.chunk_id: d for d in decisions}
    assert by_chunk[1].decision == "rejected_distance"
    assert by_chunk[1].provenance_score == 0.0


# ---------------------------------------------------------------------------
# Production wiring: the SQL document state feeds the origin/readiness
# components and graph support feeds the evidence component.

def _wiring_session(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.db import Base
    from app.persistence import models

    engine = create_engine(f"sqlite:///{tmp_path / 'wiring.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    doc_a = models.Document(
        title="With origin", source="wiki", ingestion_status="ready",
        trust_tier="untrusted", trust_score=0.2,
        trust_policy_version="source-trust-v1", ingestion_origin="api")
    doc_b = models.Document(
        title="Without origin", source="wiki", ingestion_status="ready",
        trust_tier="untrusted", trust_score=0.2,
        trust_policy_version="source-trust-v1", ingestion_origin="")
    session.add_all([doc_a, doc_b])
    session.flush()
    chunk_a = models.Chunk(document_id=doc_a.id, index=0, text="alpha text",
                           start_offset=0, end_offset=10, vector_id="vec-a")
    chunk_b = models.Chunk(document_id=doc_b.id, index=0, text="beta text",
                           start_offset=0, end_offset=9, vector_id="vec-b")
    session.add_all([chunk_a, chunk_b])
    session.flush()
    return session, engine, (doc_a, chunk_a), (doc_b, chunk_b)


def _apply(session, contexts, mode):
    import app.services.retrieval as retrieval

    return retrieval._apply_security_filter(session, contexts, mode=mode)


def test_apply_security_filter_wires_origin_readiness_and_graph(tmp_path, monkeypatch):
    import app.services.retrieval as retrieval

    monkeypatch.setattr(retrieval, "_POLICY_CACHE", _policy(max_distance=1.0))
    session, engine, a, b = _wiring_session(tmp_path)
    try:
        contexts = [
            {"text": "alpha text", "score": 0.10, "metadata": {
                "chunk_id": a[1].id, "document_id": a[0].id, "source": "wiki",
                "retrieval_sources": ["vector"]}},
            {"text": "beta text", "score": 0.20, "metadata": {
                "chunk_id": b[1].id, "document_id": b[0].id, "source": "wiki",
                "retrieval_sources": ["vector"]}},
        ]
        _, decision_rows = _apply(session, contexts, mode="vector")
        scores = {row["chunk_id"]: row["provenance_score"] for row in decision_rows}
        # With server-recorded ingestion origin: grounding 0.85.
        assert scores[a[1].id] == provenance_score(0.2, 0.85)
        # Without one: grounding 0.70 (the pre-remediation persisted value).
        assert scores[b[1].id] == provenance_score(0.2, 0.70)
    finally:
        session.close()
        engine.dispose()


def test_apply_security_filter_graph_supported_candidate_gets_graph_credit(
        tmp_path, monkeypatch):
    import app.services.retrieval as retrieval

    monkeypatch.setattr(retrieval, "_POLICY_CACHE", _policy(max_distance=1.0))
    session, engine, a, _b = _wiring_session(tmp_path)
    try:
        contexts = [{
            "text": "alpha text", "score": 0.10, "metadata": {
                "chunk_id": a[1].id, "document_id": a[0].id, "source": "wiki",
                "retrieval_sources": ["graph"], "graph_score": 0.9, "min_hop": 1},
        }]
        _, decision_rows = _apply(session, contexts, mode="graph")
        assert len(decision_rows) == 1
        assert decision_rows[0]["provenance_score"] == provenance_score(0.2, 1.0)
        # Graph-origin survivors explain distance inapplicability.
        assert "distance_not_applicable_graph_origin" in decision_rows[0]["reason_codes"]
    finally:
        session.close()
        engine.dispose()
