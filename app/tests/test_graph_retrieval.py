"""Tests for bounded relational multi-hop graph traversal."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation
from app.services.graph_retrieval import (
    GraphTraversalLimitError,
    UnsupportedGraphFilter,
    retrieve_graph_contexts,
    retrieve_graph_paths,
    resolve_graph_seeds,
)


def _entity(name):
    return ExtractedEntity(
        name=name, canonical_name=name.casefold(), entity_type="concept"
    )


def _relation(source, predicate, target, text):
    return ExtractedRelation(
        source=_entity(source),
        predicate=predicate,
        target=_entity(target),
        evidence=text,
        evidence_start=0,
        evidence_end=len(text),
        confidence=0.9,
    )


def _session_with_chain():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="Access chain", source="unit", tags="graph")
    session.add(document)
    session.flush()
    triples = [
        ("User", "purchases", "Subscription", "User purchases Subscription."),
        (
            "Subscription",
            "grants",
            "PremiumAccess",
            "Subscription grants PremiumAccess.",
        ),
        (
            "PremiumAccess",
            "unlocks",
            "Dashboard",
            "PremiumAccess unlocks Dashboard.",
        ),
    ]
    chunks = []
    for index, (source, predicate, target, text) in enumerate(triples):
        chunk = models.Chunk(
            document_id=document.id,
            index=index,
            text=text,
            start_offset=0,
            end_offset=len(text),
            vector_id=f"chunk:{document.id}:{index}",
        )
        session.add(chunk)
        session.flush()
        persist_chunk_extraction(
            session,
            chunk=chunk,
            relations=[_relation(source, predicate, target, text)],
            provider="ollama",
            model="gemma4:latest",
        )
        chunks.append(chunk)
    session.commit()
    return session, engine, document, chunks


def test_graph_traversal_enforces_exact_hop_boundaries_and_provenance():
    session, engine, document, chunks = _session_with_chain()

    one_hop = retrieve_graph_contexts(
        session, query="Explain User", max_hops=1, limit=10
    )
    two_hop = retrieve_graph_contexts(
        session, query="Explain User", max_hops=2, limit=10
    )
    three_hop = retrieve_graph_contexts(
        session, query="Explain User", max_hops=3, limit=10
    )

    assert [item["metadata"]["graph"]["predicate"] for item in one_hop] == [
        "purchases"
    ]
    assert [item["metadata"]["graph"]["predicate"] for item in two_hop] == [
        "purchases",
        "grants",
    ]
    assert [item["metadata"]["graph"]["predicate"] for item in three_hop] == [
        "purchases",
        "grants",
        "unlocks",
    ]
    assert [item["metadata"]["graph"]["hop"] for item in three_hop] == [1, 2, 3]
    assert {item["metadata"]["chunk_id"] for item in three_hop} == {
        chunk.id for chunk in chunks
    }
    session.close()
    engine.dispose()


def test_graph_traversal_uses_vector_seed_chunk_mentions():
    session, engine, document, chunks = _session_with_chain()
    contexts = retrieve_graph_contexts(
        session,
        query="How does this work?",
        seed_chunk_ids=[chunks[0].id],
        max_hops=2,
        limit=10,
    )
    assert {item["metadata"]["graph"]["predicate"] for item in contexts} == {
        "purchases",
        "grants",
        "unlocks",
    }
    session.close()
    engine.dispose()


def test_cycles_self_loops_and_parallel_predicates_terminate_deterministically():
    session, engine, document, chunks = _session_with_chain()
    extras = [
        ("Dashboard", "belongs_to", "User", "Dashboard belongs to User."),
        ("User", "knows", "User", "User knows User."),
        ("User", "subscribes_to", "Subscription", "User subscribes to Subscription."),
    ]
    for offset, (source, predicate, target, text) in enumerate(extras, start=3):
        chunk = models.Chunk(
            document_id=document.id,
            index=offset,
            text=text,
            start_offset=0,
            end_offset=len(text),
            vector_id=f"chunk:{document.id}:{offset}",
        )
        session.add(chunk)
        session.flush()
        persist_chunk_extraction(
            session,
            chunk=chunk,
            relations=[_relation(source, predicate, target, text)],
            provider="ollama",
            model="gemma4:latest",
        )
    session.commit()

    first = retrieve_graph_contexts(
        session, query="Explain User", max_hops=3, limit=20
    )
    second = retrieve_graph_contexts(
        session, query="Explain User", max_hops=3, limit=20
    )
    assert [item["metadata"]["evidence_id"] for item in first] == [
        item["metadata"]["evidence_id"] for item in second
    ]
    assert len(first) == 6
    assert len({item["metadata"]["evidence_id"] for item in first}) == 6
    session.close()
    engine.dispose()


def test_failed_documents_are_not_graph_retrievable():
    session, engine, document, chunks = _session_with_chain()
    document.ingestion_status = "failed"
    session.commit()
    assert retrieve_graph_contexts(
        session, query="User", max_hops=2, limit=10
    ) == []
    session.close()
    engine.dispose()


def test_unsupported_hybrid_filter_is_rejected_not_ignored():
    session, engine, document, chunks = _session_with_chain()
    with pytest.raises(UnsupportedGraphFilter):
        retrieve_graph_contexts(
            session,
            query="User",
            max_hops=2,
            limit=10,
            filters={"$or": [{"document_id": 1}]},
        )
    session.close()
    engine.dispose()


@pytest.mark.parametrize("hops", [0, 4])
def test_graph_hops_outside_documented_range_are_rejected(hops):
    session, engine, document, chunks = _session_with_chain()
    with pytest.raises(ValueError):
        retrieve_graph_contexts(session, query="User", max_hops=hops, limit=10)
    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# 10A.5 — directional path traversal (retrieve_graph_paths)
# ---------------------------------------------------------------------------


def test_retrieve_graph_paths_outbound_returns_complete_one_two_three_hop_chains():
    session, engine, document, chunks = _session_with_chain()
    try:
        one_hop = retrieve_graph_paths(
            session, query="Explain User", max_hops=1,
            direction="outbound", limit=10)
        two_hop = retrieve_graph_paths(
            session, query="Explain User", max_hops=2,
            direction="outbound", limit=10)
        three_hop = retrieve_graph_paths(
            session, query="Explain User", max_hops=3,
            direction="outbound", limit=10)

        # hop 1: only User->Subscription
        assert len(one_hop) == 1
        assert one_hop[0].hop_count == 1
        assert one_hop[0].seed_entity_id == one_hop[0].steps[0].source_entity_id
        assert one_hop[0].steps[0].predicate == "purchases"
        assert one_hop[0].steps[0].source == "User"
        assert one_hop[0].steps[0].target == "Subscription"
        assert len(one_hop[0].steps) == 1

        # hop 2: one 2-hop path User->Sub->PremiumAccess
        assert any(p.hop_count == 2 for p in two_hop)
        two_step = next(p for p in two_hop if p.hop_count == 2)
        assert [s.predicate for s in two_step.steps] == ["purchases", "grants"]
        assert two_step.steps[0].source == "User"
        assert two_step.steps[1].target == "PremiumAccess"
        assert two_step.seed_entity_id == two_step.steps[0].source_entity_id
        assert two_step.terminal_entity_id == two_step.steps[-1].target_entity_id

        # hop 3: User->Sub->PremiumAccess->Dashboard
        three_step = next(p for p in three_hop if p.hop_count == 3)
        assert [s.predicate for s in three_step.steps] == [
            "purchases", "grants", "unlocks"]
        assert three_step.hop_count == 3
        first = three_step.steps[0]
        assert {f for f in ("edge_id", "evidence_id", "source_entity_id", "source",
                            "source_type", "predicate", "target_entity_id", "target",
                            "target_type", "chunk_id", "document_id", "evidence",
                            "confidence", "extraction_id", "extraction_model")
               }.issubset(first.model_fields.keys())
        assert first.document_id == document.id
        assert first.evidence == "User purchases Subscription."
        assert first.extraction_model == "gemma4:latest"
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_path_score_is_min_confidence_over_hop_count():
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="Explain User", max_hops=3,
                                     direction="outbound", limit=10)
        for path in paths:
            min_conf = min(step.confidence for step in path.steps)
            assert path.score == min_conf / path.hop_count
        by_hop = {p.hop_count: p.score for p in paths}
        assert by_hop[1] == 0.9
        assert by_hop[2] == 0.45
        assert by_hop[3] == 0.3
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_inbound_preserves_stored_edge_orientation():
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="Explain Dashboard",
                                     max_hops=3, direction="inbound", limit=10)
        # Inbound from Dashboard follows target->source but preserves orientation.
        assert len(paths) >= 1
        first = paths[0]
        # Stored orientation: PremiumAccess unlocks Dashboard.
        assert first.steps[0].predicate == "unlocks"
        assert first.steps[0].source == "PremiumAccess"
        assert first.steps[0].target == "Dashboard"
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_both_direction_traverses_either_way():
    session, engine, document, chunks = _session_with_chain()
    try:
        outbound = retrieve_graph_paths(session, query="Explain User",
                                        max_hops=1, direction="outbound", limit=10)
        both = retrieve_graph_paths(session, query="Explain User",
                                    max_hops=1, direction="both", limit=10)
        assert len(both) >= len(outbound)
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_deterministic_order_across_calls():
    session, engine, document, chunks = _session_with_chain()
    try:
        first = retrieve_graph_paths(session, query="Explain User", max_hops=3,
                                     direction="outbound", limit=10)
        second = retrieve_graph_paths(session, query="Explain User", max_hops=3,
                                      direction="outbound", limit=10)
        assert [p.model_dump() for p in first] == [p.model_dump() for p in second]
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_scalar_document_id_filter():
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="Explain User", max_hops=2,
                                     direction="outbound", limit=10,
                                     filters={"document_id": document.id})
        assert all(s.document_id == document.id for p in paths for s in p.steps)
        assert retrieve_graph_paths(session, query="Explain User", max_hops=2,
                                    direction="outbound", limit=10,
                                    filters={"document_id": 999999}) == []
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_unsupported_filter_raises():
    session, engine, document, chunks = _session_with_chain()
    try:
        with pytest.raises(UnsupportedGraphFilter):
            retrieve_graph_paths(session, query="Explain User", max_hops=2,
                                 direction="outbound", limit=10,
                                 filters={"$or": [{"document_id": 1}]})
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_no_seeds_returns_empty():
    session, engine, document, chunks = _session_with_chain()
    try:
        assert retrieve_graph_paths(session, query="zzznoentzzz", max_hops=2,
                                    direction="outbound", limit=10) == []
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_scalar_tags_filter_matches_membership():
    session, engine, document, chunks = _session_with_chain()
    try:
        # document.tags == "graph"; tags filter splits stored CSV and trims
        matched = retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                       direction="outbound", limit=10,
                                       filters={"tags": "graph"})
        assert matched
        assert retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                    direction="outbound", limit=10,
                                    filters={"tags": "absent"}) == []
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_tags_filter_membership_after_csv_split_and_trim():
    """Plan 10A.5 matrix: `tags` is "one scalar tag; exact membership after
    splitting stored comma-separated tags and trimming whitespace" — not
    whole-column equality."""
    session, engine, document, chunks = _session_with_chain()
    try:
        document.tags = "alpha, graph ,beta"
        session.commit()
        # membership: each trimmed CSV element matches exactly
        for present in ("graph", "alpha", "beta"):
            assert retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                        direction="outbound", limit=10,
                                        filters={"tags": present})
        # exact membership, not substring and not whole-column equality
        for absent in ("alph", "absent", "alpha, graph ,beta", "graph "):
            assert retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                        direction="outbound", limit=10,
                                        filters={"tags": absent}) == []
        # the legacy contexts entry point applies the same membership semantics
        assert retrieve_graph_contexts(session, query="Explain User", max_hops=1,
                                       limit=10, filters={"tags": "graph"})
        assert retrieve_graph_contexts(session, query="Explain User", max_hops=1,
                                       limit=10, filters={"tags": "alph"}) == []
        # chunk-derived seeds are admitted only through matching documents
        assert retrieve_graph_paths(session, query="zzznoentzzz", max_hops=1,
                                    direction="outbound", limit=10,
                                    filters={"tags": "graph"},
                                    seed_chunk_ids=[chunks[0].id])
        assert retrieve_graph_paths(session, query="zzznoentzzz", max_hops=1,
                                    direction="outbound", limit=10,
                                    filters={"tags": "absent"},
                                    seed_chunk_ids=[chunks[0].id]) == []
    finally:
        session.close()
        engine.dispose()
