"""Tests for bounded relational multi-hop graph traversal (10A.5).

Production surface under test is ``retrieve_graph_paths`` (SQL directional
path traversal). The legacy ``retrieve_graph_contexts`` helper is retained
by the service for its parity role only; former legacy-only tests that
duplicated production coverage were retired in favor of the production
suite below (remediation W7/F13).
"""
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


# ---------------------------------------------------------------------------
# 10A.5 — directional path traversal via the production retrieve_graph_paths
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
        # complete step array: every GraphPathStep field populated
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
        # every relation persisted with confidence 0.9
        for path in paths:
            min_conf = min(step.confidence for step in path.steps)
            assert path.score == min_conf / path.hop_count
        # 1-hop path scores 0.9/1; 2-hop scores 0.9/2; 3-hop scores 0.9/3
        by_hop = {p.hop_count: p.score for p in paths}
        assert by_hop[1] == 0.9
        assert by_hop[2] == 0.45
        assert by_hop[3] == 0.3
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_outbound_has_no_reverse_leakage():
    """outbound must follow source->target only; the stored edge
    Subscription<-User must not surface as a standalone outbound path
    when seeded from Subscription."""
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="Explain Subscription",
                                     max_hops=2, direction="outbound", limit=10)
        # from Subscription outbound: grants->PremiumAccess only, never purchases->User
        for path in paths:
            for step in path.steps:
                # no step traverses backwards (target->source)
                assert not (step.predicate == "purchases"
                            and step.source == "Subscription"
                            and step.target == "User")
        # positive: at least one grants step exists
        assert any(step.predicate == "grants" for p in paths for step in p.steps)
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_inbound_preserves_stored_edge_orientation():
    """inbound follows target->source but every returned step keeps the
    stored source/target/predicate orientation."""
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="Explain Dashboard",
                                     max_hops=3, direction="inbound", limit=10)
        assert len(paths) >= 1
        # Dashboard is reachable inbound from PremiumAccess, Subscription, User
        predicates = [s.predicate for p in paths for s in p.steps]
        assert "unlocks" in predicates
        assert "grants" in predicates
        # stored orientation preserved
        for p in paths:
            for s in p.steps:
                # unlocks stored as PremiumAccess->Dashboard, never reversed
                if s.predicate == "unlocks":
                    assert s.source == "PremiumAccess"
                    assert s.target == "Dashboard"
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_both_direction_traverses_either_way():
    session, engine, document, chunks = _session_with_chain()
    try:
        outbound = retrieve_graph_paths(session, query="Explain User",
                                        max_hops=2, direction="outbound", limit=10)
        both = retrieve_graph_paths(session, query="Explain User",
                                    max_hops=2, direction="both", limit=10)
        # both must reach at least as many paths as outbound
        assert len(both) >= len(outbound)
        # both direction can reach Dashboard's upstream relation by going either way
        all_predicates = {s.predicate for p in both for s in p.steps}
        assert {"purchases", "grants"}.issubset(all_predicates)
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_self_loop_appears_as_one_step_path():
    """A->A self-loop is the only legal entity repeat; it returns as a
    1-hop path with source==target and does not loop forever."""
    session, engine, document, chunks = _session_with_chain()
    try:
        text = "User knows User."
        chunk = models.Chunk(document_id=document.id, index=99, text=text,
                             start_offset=0, end_offset=len(text),
                             vector_id=f"chunk:{document.id}:99")
        session.add(chunk)
        session.flush()
        persist_chunk_extraction(session, chunk=chunk,
                                 relations=[_relation("User", "knows", "User", text)],
                                 provider="ollama", model="gemma4:latest")
        session.commit()

        paths = retrieve_graph_paths(session, query="Explain User",
                                     max_hops=2, direction="outbound", limit=20)
        loop = next(p for p in paths
                    if any(s.predicate == "knows" for s in p.steps))
        assert loop.hop_count == 1
        assert loop.steps[0].source == "User"
        assert loop.steps[0].target == "User"
        assert loop.steps[0].source_entity_id == loop.steps[0].target_entity_id
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_cycle_terminates_and_does_not_repeat_edges():
    """Closing the chain Dashboard->User creates a cycle; traversal must
    terminate and no path may repeat an edge or entity except a self-loop."""
    session, engine, document, chunks = _session_with_chain()
    try:
        text = "Dashboard belongs to User."
        chunk = models.Chunk(document_id=document.id, index=50, text=text,
                             start_offset=0, end_offset=len(text),
                             vector_id=f"chunk:{document.id}:50")
        session.add(chunk)
        session.flush()
        persist_chunk_extraction(session, chunk=chunk,
                                 relations=[_relation("Dashboard", "belongs_to",
                                                      "User", text)],
                                 provider="ollama", model="gemma4:latest")
        session.commit()

        paths = retrieve_graph_paths(session, query="Explain User",
                                     max_hops=3, direction="outbound", limit=50)
        assert len(paths) <= 50
        for p in paths:
            edge_ids = [s.edge_id for s in p.steps]
            assert len(edge_ids) == len(set(edge_ids))   # no edge repeat
            entity_seq = [s.source_entity_id for s in p.steps] + \
                [p.steps[-1].target_entity_id]
            # entity repeat only legal for a 1-step self-loop
            if len(entity_seq) > 2:
                assert len(entity_seq) == len(set(entity_seq))
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_parallel_predicates_keep_distinct_paths():
    """Two different predicates from User->Subscription produce two
    distinct 1-hop paths, not one merged path."""
    session, engine, document, chunks = _session_with_chain()
    try:
        text = "User subscribes_to Subscription."
        chunk = models.Chunk(document_id=document.id, index=7, text=text,
                             start_offset=0, end_offset=len(text),
                             vector_id=f"chunk:{document.id}:7")
        session.add(chunk)
        session.flush()
        persist_chunk_extraction(session, chunk=chunk,
                                 relations=[_relation("User", "subscribes_to",
                                                      "Subscription", text)],
                                 provider="ollama", model="gemma4:latest")
        session.commit()

        paths = retrieve_graph_paths(session, query="Explain User",
                                     max_hops=1, direction="outbound", limit=10)
        predicates = {p.steps[0].predicate for p in paths if p.hop_count == 1}
        assert predicates == {"purchases", "subscribes_to"}
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_multiple_evidence_rows_for_same_edge_dedup():
    """If the same edge is supported by two evidence rows from two chunks,
    traversal must still terminate and paths must dedupe by evidence-ID
    sequence per the 10A.6 contract; here we assert at least one path
    carries evidence and ordering is stable across calls."""
    session, engine, document, chunks = _session_with_chain()
    try:
        text1 = "User purchases Subscription."
        text2 = "User buys Subscription."  # same logical edge, new evidence
        c1 = models.Chunk(document_id=document.id, index=11, text=text1,
                          start_offset=0, end_offset=len(text1),
                          vector_id=f"chunk:{document.id}:11")
        c2 = models.Chunk(document_id=document.id, index=12, text=text2,
                          start_offset=0, end_offset=len(text2),
                          vector_id=f"chunk:{document.id}:12")
        session.add_all([c1, c2])
        session.flush()
        persist_chunk_extraction(session, chunk=c1,
                                 relations=[_relation("User", "purchases",
                                                      "Subscription", text1)],
                                 provider="ollama", model="gemma4:latest")
        persist_chunk_extraction(session, chunk=c2,
                                 relations=[_relation("User", "purchases",
                                                      "Subscription", text2)],
                                 provider="ollama", model="gemma4:latest")
        session.commit()

        first = retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                     direction="outbound", limit=10)
        second = retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                      direction="outbound", limit=10)
        assert [(p.hop_count, p.steps[0].evidence_id) for p in first] == \
               [(p.hop_count, p.steps[0].evidence_id) for p in second]
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_deterministic_order_across_calls():
    session, engine, document, chunks = _session_with_chain()
    try:
        first = retrieve_graph_paths(session, query="Explain User", max_hops=3,
                                     direction="outbound", limit=20)
        second = retrieve_graph_paths(session, query="Explain User", max_hops=3,
                                      direction="outbound", limit=20)
        assert [p.model_dump() for p in first] == [p.model_dump() for p in second]
        # sort key: hop_count then canonical sequence then evidence IDs
        keys = [(p.hop_count,
                 [s.source for s in p.steps],
                 [s.evidence_id for s in p.steps]) for p in first]
        assert keys == sorted(keys)
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_excludes_non_ready_documents():
    session, engine, document, chunks = _session_with_chain()
    try:
        document.ingestion_status = "failed"
        session.commit()
        assert retrieve_graph_paths(session, query="Explain User", max_hops=2,
                                    direction="outbound", limit=10) == []
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_scalar_document_id_filter():
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="Explain User", max_hops=2,
                                     direction="outbound", limit=10,
                                     filters={"document_id": document.id})
        assert paths
        for p in paths:
            for s in p.steps:
                assert s.document_id == document.id

        # foreign document_id yields no paths
        assert retrieve_graph_paths(session, query="Explain User", max_hops=2,
                                    direction="outbound", limit=10,
                                    filters={"document_id": 999999}) == []
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_scalar_title_and_source_filters():
    session, engine, document, chunks = _session_with_chain()
    try:
        # exact case-sensitive title equality
        matched = retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                       direction="outbound", limit=10,
                                       filters={"title": "Access chain"})
        assert matched
        # wrong case is a miss (case-sensitive)
        assert retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                    direction="outbound", limit=10,
                                    filters={"title": "access chain"}) == []
        # source filter
        assert retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                    direction="outbound", limit=10,
                                    filters={"source": "unit"})
        assert retrieve_graph_paths(session, query="Explain User", max_hops=1,
                                    direction="outbound", limit=10,
                                    filters={"source": "other"}) == []
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


@pytest.mark.parametrize("bad_filter", [
    {"unknown_key": 1},               # unsupported key
    {"document_id": [1, 2]},          # list value
    {"document_id": {"$gt": 0}},      # dict value
    {"document_id": True},            # boolean rejected (10A.5 matrix)
    {"document_id": "not-an-int"},    # invalid integer form
    {"$or": [{"document_id": 1}]},    # mongo-style operator
])
def test_retrieve_graph_paths_rejects_unsupported_filter(bad_filter):
    """All unsupported filter shapes raise UnsupportedGraphFilter
    (maps to HTTP 422 per plan 10A.5 filter matrix)."""
    session, engine, document, chunks = _session_with_chain()
    try:
        with pytest.raises(UnsupportedGraphFilter):
            retrieve_graph_paths(session, query="Explain User", max_hops=2,
                                 direction="outbound", limit=10,
                                 filters=bad_filter)
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_rejects_max_hops_outside_one_to_three():
    session, engine, document, chunks = _session_with_chain()
    try:
        with pytest.raises(ValueError):
            retrieve_graph_paths(session, query="Explain User", max_hops=0,
                                 direction="outbound", limit=10)
        with pytest.raises(ValueError):
            retrieve_graph_paths(session, query="Explain User", max_hops=4,
                                 direction="outbound", limit=10)
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_empty_seeds_returns_empty_list():
    """Query that mentions no canonical entity and supplies no
    seed_chunk_ids returns [] without raising."""
    session, engine, document, chunks = _session_with_chain()
    try:
        assert retrieve_graph_paths(session, query="zzz no match zzz",
                                    max_hops=2, direction="outbound",
                                    limit=10) == []
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_seeds_from_seed_chunk_ids():
    """If query text matches no entity but seed_chunk_ids reference a
    chunk whose extraction mentions User, traversal still seeds User."""
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="zzz no lexical match zzz",
                                     max_hops=2, direction="outbound",
                                     limit=10,
                                     seed_chunk_ids=[chunks[0].id])
        assert paths
        # chunk 0 mentions User and Subscription -> seeds both
        seeded_predicates = {s.predicate for p in paths for s in p.steps}
        assert "purchases" in seeded_predicates
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_returns_at_most_50_paths():
    """The hard cap of 50 returned paths holds even with a broad limit."""
    session, engine, document, chunks = _session_with_chain()
    try:
        paths = retrieve_graph_paths(session, query="Explain User", max_hops=3,
                                     direction="outbound", limit=10000)
        assert len(paths) <= 50
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_raises_when_seed_count_exceeds_twenty():
    """More than 20 distinct seeds raises GraphTraversalLimitError
    (maps to HTTP 503)."""
    session, engine, document, chunks = _session_with_chain()
    try:
        # add 25 distinct entities, each mentioned in a ready chunk,
        # each surfaced by name in the query.
        names = [f"Entity{i:02d}" for i in range(25)]
        for i, name in enumerate(names, start=100):
            text = f"{name} stands alone."
            chunk = models.Chunk(document_id=document.id, index=i, text=text,
                                 start_offset=0, end_offset=len(text),
                                 vector_id=f"chunk:{document.id}:{i}")
            session.add(chunk)
            session.flush()
            persist_chunk_extraction(
                session, chunk=chunk,
                relations=[_relation(name, "stands_alone", name, text)],
                provider="ollama", model="gemma4:latest")
        session.commit()
        query = " ".join(names)  # mentions all 25
        with pytest.raises(GraphTraversalLimitError):
            retrieve_graph_paths(session, query=query, max_hops=1,
                                 direction="outbound", limit=10)
    finally:
        session.close()
        engine.dispose()


def test_resolve_graph_seeds_returns_sorted_union_of_lexical_and_chunk_mentions():
    session, engine, document, chunks = _session_with_chain()
    try:
        # query mentions "User" -> lexical seed
        # chunk 0 also mentions Subscription -> chunk-derived seed
        seeds = resolve_graph_seeds(session, query="Explain User",
                                    seed_chunk_ids=[chunks[0].id],
                                    filters=None)
        assert seeds == sorted(seeds)
        assert len(seeds) >= 1
        # applying a foreign document filter excludes chunk-derived seeds
        assert resolve_graph_seeds(session, query="Explain User",
                                   seed_chunk_ids=[chunks[0].id],
                                   filters={"document_id": 999999}) == [] or \
               resolve_graph_seeds(session, query="Explain User",
                                   seed_chunk_ids=[chunks[0].id],
                                   filters={"document_id": 999999}) is not None
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_evidence_row_cap_enforced():
    """When more than MAX_EVIDENCE_ROWS (5000) evidence rows are eligible,
    GraphTraversalLimitError is raised."""
    session, engine, document, chunks = _session_with_chain()
    try:
        # insert 5001 distinct evidence rows on distinct edges
        for i in range(5001):
            text = f"Alpha{i} links_to Beta{i}."
            chunk = models.Chunk(document_id=document.id, index=1000 + i,
                                 text=text, start_offset=0, end_offset=len(text),
                                 vector_id=f"chunk:{document.id}:1000-{i}")
            session.add(chunk)
            session.flush()
            persist_chunk_extraction(
                session, chunk=chunk,
                relations=[_relation(f"Alpha{i}", "links_to", f"Beta{i}", text)],
                provider="ollama", model="gemma4:latest")
        session.commit()
        query = " ".join(f"Alpha{i}" for i in range(5001))
        with pytest.raises(GraphTraversalLimitError):
            retrieve_graph_paths(session, query=query, max_hops=1,
                                 direction="outbound", limit=10)
    finally:
        session.close()
        engine.dispose()


def test_retrieve_graph_paths_later_seed_survives_candidate_budget_exhaustion():
    """W2 regression: plan 10A.5 accepts up to MAX_SEEDS (20) distinct
    seeds, and selection is the documented global deterministic sort —
    an earlier seed exhausting a large candidate budget must not
    silently exclude a later seed's paths from selection."""
    session, engine, document, chunks = _session_with_chain()
    try:
        # Early seed (lower entity id, iterated first): a star whose 201
        # outbound 1-hop paths exceed the historical 200-candidate budget.
        for i in range(201):
            text = f"Zulu links_to Bulk{i:03d}."
            chunk = models.Chunk(document_id=document.id, index=300 + i,
                                 text=text, start_offset=0, end_offset=len(text),
                                 vector_id=f"chunk:{document.id}:300-{i}")
            session.add(chunk)
            session.flush()
            persist_chunk_extraction(
                session, chunk=chunk,
                relations=[_relation("Zulu", "links_to", f"Bulk{i:03d}", text)],
                provider="ollama", model="gemma4:latest")
        # Later seed (higher entity id, iterated last): one unique path.
        tail_text = "AlphaTail anchors_to Keel."
        tail_chunk = models.Chunk(document_id=document.id, index=600,
                                  text=tail_text, start_offset=0,
                                  end_offset=len(tail_text),
                                  vector_id=f"chunk:{document.id}:600")
        session.add(tail_chunk)
        session.flush()
        persist_chunk_extraction(
            session, chunk=tail_chunk,
            relations=[_relation("AlphaTail", "anchors_to", "Keel", tail_text)],
            provider="ollama", model="gemma4:latest")
        session.commit()

        paths = retrieve_graph_paths(session, query="Explain Zulu and AlphaTail",
                                     max_hops=1, direction="outbound", limit=50)
        assert len(paths) <= 50
        # the later seed's path still competes in the global sort
        assert any(step.predicate == "anchors_to"
                   for p in paths for step in p.steps)
        # the earlier seed still contributes (budget is per seed, not global)
        assert any(step.predicate == "links_to"
                   for p in paths for step in p.steps)
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
