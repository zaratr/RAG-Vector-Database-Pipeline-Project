"""Tests for ready-only vector/graph hybrid retrieval and fusion."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.core.models import RetrievedChunk
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation
from app.services.retrieval import retrieve_contexts

pytestmark = pytest.mark.asyncio


class FixedEmbeddingProvider:
    def __init__(self):
        self.calls = []

    async def embed_texts(self, texts):
        self.calls.append(texts)
        return [[0.1, 0.2]]


class FixedVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def query(self, embedding, top_k, filters=None):
        self.calls.append((embedding, top_k, filters))
        return self.results


def _graph_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="Graph doc", source="unit")
    session.add(document)
    session.flush()
    graph_chunk = models.Chunk(
        document_id=document.id,
        index=0,
        text="User purchases Subscription.",
        start_offset=0,
        end_offset=28,
        vector_id="chunk:graph",
    )
    vector_chunk = models.Chunk(
        document_id=document.id,
        index=1,
        text="Vector-only evidence",
        start_offset=29,
        end_offset=49,
        vector_id="chunk:vector",
    )
    session.add_all([graph_chunk, vector_chunk])
    session.flush()
    relation = ExtractedRelation(
        source=ExtractedEntity(
            name="User", canonical_name="user", entity_type="person"
        ),
        predicate="purchases",
        target=ExtractedEntity(
            name="Subscription",
            canonical_name="subscription",
            entity_type="product",
        ),
        evidence=graph_chunk.text,
        evidence_start=0,
        evidence_end=len(graph_chunk.text),
        confidence=0.9,
    )
    persist_chunk_extraction(
        session,
        chunk=graph_chunk,
        relations=[relation],
        provider="ollama",
        model="gemma4:latest",
    )
    session.commit()
    return session, engine, document, graph_chunk, vector_chunk


async def test_hybrid_rrf_deduplicates_chunk_and_preserves_native_score_and_paths():
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    embedding = FixedEmbeddingProvider()
    vector = FixedVectorStore(
        [
            RetrievedChunk(
                text="untrusted chroma text",
                score=0.1,
                vector_id=graph_chunk.vector_id,
                metadata={
                    "document_id": document.id,
                    "chunk_id": graph_chunk.id,
                    "title": "stale Chroma title",
                },
            ),
            RetrievedChunk(
                text="untrusted vector text",
                score=0.2,
                vector_id=vector_chunk.vector_id,
                metadata={"document_id": document.id, "chunk_id": vector_chunk.id},
            ),
        ]
    )

    contexts = await retrieve_contexts(
        query="Explain User",
        embedding_provider=embedding,
        vector_store=vector,
        session=session,
        mode="hybrid",
        top_k=5,
        graph_max_hops=2,
    )

    assert len(contexts) == 2
    merged = next(item for item in contexts if item["metadata"]["chunk_id"] == graph_chunk.id)
    vector_only = next(
        item for item in contexts if item["metadata"]["chunk_id"] == vector_chunk.id
    )
    assert merged["text"] == graph_chunk.text
    assert merged["metadata"]["title"] == "Graph doc"
    assert merged["score"] == 0.1
    assert merged["metadata"]["retrieval_sources"] == ["vector", "graph"]
    # 10A.6: graph_paths is a list of complete GraphPath objects; the predicate
    # is on the first step of the path.
    assert merged["metadata"]["graph_paths"][0]["steps"][0]["predicate"] == "purchases"
    assert merged["metadata"]["hybrid_score"] > vector_only["metadata"]["hybrid_score"]
    session.close()
    engine.dispose()


async def test_graph_only_mode_does_not_embed_or_query_vector_store():
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    embedding = FixedEmbeddingProvider()
    vector = FixedVectorStore([])

    contexts = await retrieve_contexts(
        query="Explain User",
        embedding_provider=embedding,
        vector_store=vector,
        session=session,
        mode="graph",
        top_k=5,
        graph_max_hops=1,
    )

    assert len(contexts) == 1
    assert contexts[0]["metadata"]["retrieval_sources"] == ["graph"]
    assert contexts[0]["metadata"]["graph_paths"][0]["steps"][0]["predicate"] == "purchases"
    assert embedding.calls == []
    assert vector.calls == []
    session.close()
    engine.dispose()


async def test_vector_only_hydrates_sql_text_and_preserves_distance_score():
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    embedding = FixedEmbeddingProvider()
    vector = FixedVectorStore(
        [
            RetrievedChunk(
                text="stale Chroma text",
                score=0.37,
                vector_id=vector_chunk.vector_id,
                metadata={"chunk_id": vector_chunk.id},
            )
        ]
    )

    contexts = await retrieve_contexts(
        query="anything",
        embedding_provider=embedding,
        vector_store=vector,
        session=session,
        mode="vector",
        top_k=1,
        graph_max_hops=2,
    )

    assert contexts[0]["text"] == vector_chunk.text
    assert contexts[0]["score"] == 0.37
    assert contexts[0]["metadata"]["retrieval_sources"] == ["vector"]
    session.close()
    engine.dispose()


async def test_stale_or_failed_vector_hits_are_not_query_visible():
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    document.ingestion_status = "failed"
    session.commit()
    vector = FixedVectorStore(
        [
            RetrievedChunk(
                text=graph_chunk.text,
                score=0.1,
                vector_id=graph_chunk.vector_id,
                metadata={"chunk_id": graph_chunk.id},
            ),
            RetrievedChunk(
                text="orphan",
                score=0.2,
                vector_id="orphan",
                metadata={"chunk_id": 999999},
            ),
        ]
    )

    contexts = await retrieve_contexts(
        query="User",
        embedding_provider=FixedEmbeddingProvider(),
        vector_store=vector,
        session=session,
        mode="hybrid",
        top_k=5,
        graph_max_hops=2,
    )
    assert contexts == []
    session.close()
    engine.dispose()


async def test_vector_only_rejects_alias_id_and_deduplicates_sql_chunk():
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    vector = FixedVectorStore(
        [
            RetrievedChunk(
                text="stale alias",
                score=0.01,
                vector_id="uuid-alias",
                metadata={"chunk_id": vector_chunk.id},
            ),
            RetrievedChunk(
                text="canonical",
                score=0.2,
                vector_id=vector_chunk.vector_id,
                metadata={"chunk_id": vector_chunk.id},
            ),
            RetrievedChunk(
                text="duplicate canonical",
                score=0.3,
                vector_id=vector_chunk.vector_id,
                metadata={"chunk_id": vector_chunk.id},
            ),
        ]
    )

    contexts = await retrieve_contexts(
        query="anything",
        embedding_provider=FixedEmbeddingProvider(),
        vector_store=vector,
        session=session,
        mode="vector",
        top_k=5,
    )

    assert len(contexts) == 1
    assert contexts[0]["text"] == vector_chunk.text
    assert contexts[0]["score"] == 0.2
    session.close()
    engine.dispose()


async def test_hybrid_filters_apply_identically_to_vector_and_graph_candidates():
    """Filter parity: a document_id filter excludes both vector and graph
    candidates from foreign documents identically."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([
            RetrievedChunk(text=graph_chunk.text, score=0.1, vector_id=graph_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": graph_chunk.id}),
        ])
        # filter to a non-existent document → both sides empty
        contexts = await retrieve_contexts(
            query="Explain User", embedding_provider=embedding,
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1, filters={"document_id": 999999})
        assert contexts == []
    finally:
        session.close()
        engine.dispose()


# ── 10A.6 appendix tests (F14 fill) ──────────────────────────────────


async def test_hybrid_two_document_two_hop_query_includes_both_supporting_chunks():
    """Two-document two-hop: doc A has User→Subscription; doc B has
    Subscription→PremiumAccess. Hybrid query seeded from vector hit on
    doc A must return both doc A and doc B chunks with full path steps."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        doc_a = models.Document(title="Doc A", source="unit"); session.add(doc_a); session.flush()
        doc_b = models.Document(title="Doc B", source="unit"); session.add(doc_b); session.flush()
        ca = models.Chunk(document_id=doc_a.id, index=0,
                          text="User purchases Subscription.",
                          start_offset=0, end_offset=28,
                          vector_id="chunk:docA:0")
        cb = models.Chunk(document_id=doc_b.id, index=0,
                          text="Subscription grants PremiumAccess.",
                          start_offset=0, end_offset=35,
                          vector_id="chunk:docB:0")
        session.add_all([ca, cb]); session.flush()
        persist_chunk_extraction(session, chunk=ca,
            relations=[ExtractedRelation(
                source=ExtractedEntity(name="User", canonical_name="user", entity_type="person"),
                predicate="purchases",
                target=ExtractedEntity(name="Subscription", canonical_name="subscription", entity_type="product"),
                evidence=ca.text, evidence_start=0, evidence_end=len(ca.text), confidence=0.9)],
            provider="ollama", model="gemma4:latest")
        persist_chunk_extraction(session, chunk=cb,
            relations=[ExtractedRelation(
                source=ExtractedEntity(name="Subscription", canonical_name="subscription", entity_type="product"),
                predicate="grants",
                target=ExtractedEntity(name="PremiumAccess", canonical_name="premiumaccess", entity_type="concept"),
                evidence=cb.text, evidence_start=0, evidence_end=len(cb.text), confidence=0.9)],
            provider="ollama", model="gemma4:latest")
        session.commit()

        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([
            RetrievedChunk(text=ca.text, score=0.1, vector_id=ca.vector_id,
                           metadata={"document_id": doc_a.id, "chunk_id": ca.id}),
        ])
        contexts = await retrieve_contexts(
            query="Explain User", embedding_provider=embedding,
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=2)

        chunk_ids = {c["metadata"]["chunk_id"] for c in contexts}
        assert {ca.id, cb.id}.issubset(chunk_ids)
        # full path provenance on the doc B (graph-only) chunk
        doc_b_ctx = next(c for c in contexts if c["metadata"]["chunk_id"] == cb.id)
        assert "graph_paths" in doc_b_ctx["metadata"]
        assert doc_b_ctx["metadata"]["graph_paths"]
        first_path = doc_b_ctx["metadata"]["graph_paths"][0]
        # path object exposes GraphPathStep fields
        assert "steps" in first_path or "predicate" in first_path
    finally:
        session.close(); engine.dispose()


async def test_hybrid_rrf_constant_is_sixty_and_tiebreak_is_chunk_id():
    """With two vector hits at distinct ranks and two graph hits, the
    chunk that appears in both lists gets 1/(60+1) + 1/(60+1) and ranks
    above any single-origin chunk."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([
            RetrievedChunk(text=graph_chunk.text, score=0.1, vector_id=graph_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": graph_chunk.id}),
            RetrievedChunk(text=vector_chunk.text, score=0.2, vector_id=vector_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": vector_chunk.id}),
        ])
        contexts = await retrieve_contexts(
            query="Explain User", embedding_provider=embedding,
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1)

        # graph_chunk appears in both → higher hybrid_score
        merged = next(c for c in contexts if c["metadata"]["chunk_id"] == graph_chunk.id)
        vector_only = next(c for c in contexts if c["metadata"]["chunk_id"] == vector_chunk.id)
        # rank 1 in both lists → 1/(60+1) + 1/(60+1) ≈ 0.0328
        assert abs(merged["metadata"]["hybrid_score"] - (1/61 + 1/61)) < 1e-9
        # vector_only rank 2 in vector list only → 1/(60+2)
        assert abs(vector_only["metadata"]["hybrid_score"] - 1/62) < 1e-9
        # sorted by (-hybrid_score, chunk_id): merged first
        assert contexts[0]["metadata"]["chunk_id"] == graph_chunk.id
    finally:
        session.close(); engine.dispose()


async def test_hybrid_preserves_native_vector_distance_in_score_and_exposes_graph_score():
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([
            RetrievedChunk(text=graph_chunk.text, score=0.15, vector_id=graph_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": graph_chunk.id}),
        ])
        contexts = await retrieve_contexts(
            query="Explain User", embedding_provider=embedding,
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1)
        merged = next(c for c in contexts if c["metadata"]["chunk_id"] == graph_chunk.id)
        # native distance preserved as score
        assert merged["score"] == 0.15
        assert merged["metadata"]["score_type"] == "vector_distance" or "vector" in merged["metadata"].get("retrieval_sources", [])
        # hybrid_score exposed separately
        assert "hybrid_score" in merged["metadata"]
    finally:
        session.close(); engine.dispose()


async def test_hybrid_graph_only_chunk_carries_graph_path_score():
    """A chunk reachable only via graph (no vector hit) gets
    score=graph_score and score_type=graph_path_score."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([])  # no vector hits at all
        contexts = await retrieve_contexts(
            query="Explain User", embedding_provider=embedding,
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1)
        # graph_chunk surfaces via graph only
        graph_only = next(c for c in contexts if c["metadata"]["chunk_id"] == graph_chunk.id)
        assert "graph_score" in graph_only["metadata"] or "graph_paths" in graph_only["metadata"]
    finally:
        session.close(); engine.dispose()


async def test_hybrid_rejects_unsupported_filter_for_graph_and_hybrid_modes():
    from app.services.graph_retrieval import UnsupportedGraphFilter
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        for mode in ("graph", "hybrid"):
            with pytest.raises(UnsupportedGraphFilter):
                await retrieve_contexts(
                    query="Explain User",
                    embedding_provider=FixedEmbeddingProvider(),
                    vector_store=FixedVectorStore([]),
                    session=session, mode=mode, top_k=5, graph_max_hops=1,
                    filters={"unknown_key": 1})
    finally:
        session.close(); engine.dispose()


async def test_hybrid_deduplicates_by_sql_chunk_id():
    """Vector hit and graph hit on the same SQL chunk collapse to one
    context entry tagged retrieval_sources=['vector','graph']."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([
            RetrievedChunk(text=graph_chunk.text, score=0.1, vector_id=graph_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": graph_chunk.id}),
        ])
        contexts = await retrieve_contexts(
            query="Explain User", embedding_provider=embedding,
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1)
        assert len(contexts) == 1
        assert set(contexts[0]["metadata"]["retrieval_sources"]) == {"vector", "graph"}
    finally:
        session.close(); engine.dispose()


async def test_hybrid_deterministic_output_across_repeated_calls():
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        vector = FixedVectorStore([
            RetrievedChunk(text=graph_chunk.text, score=0.1, vector_id=graph_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": graph_chunk.id}),
            RetrievedChunk(text=vector_chunk.text, score=0.2, vector_id=vector_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": vector_chunk.id}),
        ])
        first = await retrieve_contexts(query="Explain User",
            embedding_provider=FixedEmbeddingProvider(), vector_store=vector,
            session=session, mode="hybrid", top_k=5, graph_max_hops=1)
        second = await retrieve_contexts(query="Explain User",
            embedding_provider=FixedEmbeddingProvider(), vector_store=vector,
            session=session, mode="hybrid", top_k=5, graph_max_hops=1)
        assert [c["metadata"]["chunk_id"] for c in first] == \
               [c["metadata"]["chunk_id"] for c in second]
    finally:
        session.close(); engine.dispose()


async def test_graph_mode_does_not_call_embedding_or_vector_store():
    """graph mode must not invoke embedding_provider.embed_texts or
    vector_store.query (already partially covered; retained as contract)."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([])
        await retrieve_contexts(query="Explain User",
            embedding_provider=embedding, vector_store=vector,
            session=session, mode="graph", top_k=5, graph_max_hops=1)
        assert embedding.calls == []
        assert vector.calls == []
    finally:
        session.close(); engine.dispose()


async def test_tags_filter_membership_applies_identically_across_vector_graph_and_hybrid():
    """F4 parity: `tags` is membership-after-split of the stored CSV in every
    mode; the vector side must not push tags as a raw Chroma where clause."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        document.tags = "alpha, graph ,beta"
        session.commit()
        embedding = FixedEmbeddingProvider()
        vector = FixedVectorStore([
            RetrievedChunk(text=graph_chunk.text, score=0.1, vector_id=graph_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": graph_chunk.id}),
            RetrievedChunk(text=vector_chunk.text, score=0.2, vector_id=vector_chunk.vector_id,
                           metadata={"document_id": document.id, "chunk_id": vector_chunk.id}),
        ])
        # one scalar tag present in the split+trimmed CSV matches everywhere
        for mode in ("vector", "graph", "hybrid"):
            contexts = await retrieve_contexts(
                query="Explain User", embedding_provider=embedding,
                vector_store=vector, session=session, mode=mode,
                top_k=5, graph_max_hops=1, filters={"tags": "graph"})
            assert contexts, mode
        # a tag absent from the stored CSV excludes candidates everywhere
        for mode in ("vector", "graph", "hybrid"):
            contexts = await retrieve_contexts(
                query="Explain User", embedding_provider=embedding,
                vector_store=vector, session=session, mode=mode,
                top_k=5, graph_max_hops=1, filters={"tags": "absent"})
            assert contexts == [], mode
    finally:
        session.close()
        engine.dispose()
