"""Tests for bounded relational multi-hop graph traversal."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation
from app.services.graph_retrieval import UnsupportedGraphFilter, retrieve_graph_contexts


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
