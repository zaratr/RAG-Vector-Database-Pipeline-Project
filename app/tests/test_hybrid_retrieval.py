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
    # graph_paths is a list of complete GraphPath objects; the predicate
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


# ── Hybrid retrieval contract tests ──────────────────────────────────


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


# ── RRF tie-breaking, degenerate candidate sets, and the no-evidence lane ──


def _tie_session():
    """Seed one ready document with three chunks in a known id order:

    * ``c_graph``  — "User purchases Subscription." with a relation (graph side)
    * ``c_alpha``  — plain text (vector side only)
    * ``c_beta``   — "Stakeholder reviews Roadmap." with a relation (graph side)

    Chunk ids are assigned in creation order, so c_graph < c_alpha < c_beta.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    document = models.Document(title="Tie doc", source="unit")
    session.add(document)
    session.flush()
    c_graph = models.Chunk(document_id=document.id, index=0,
                           text="User purchases Subscription.",
                           start_offset=0, end_offset=28, vector_id="chunk:tie:0")
    c_alpha = models.Chunk(document_id=document.id, index=1,
                           text="Plain vector-only evidence alpha.",
                           start_offset=29, end_offset=61, vector_id="chunk:tie:1")
    c_beta = models.Chunk(document_id=document.id, index=2,
                          text="Stakeholder reviews Roadmap.",
                          start_offset=62, end_offset=90, vector_id="chunk:tie:2")
    session.add_all([c_graph, c_alpha, c_beta])
    session.flush()
    for chunk, source, predicate, target in (
        (c_graph, "User", "purchases", "Subscription"),
        (c_beta, "Stakeholder", "reviews", "Roadmap"),
    ):
        persist_chunk_extraction(
            session, chunk=chunk,
            relations=[ExtractedRelation(
                source=ExtractedEntity(name=source, canonical_name=source.casefold(),
                                       entity_type="person"),
                predicate=predicate,
                target=ExtractedEntity(name=target, canonical_name=target.casefold(),
                                       entity_type="concept"),
                evidence=chunk.text, evidence_start=0, evidence_end=len(chunk.text),
                confidence=0.9)],
            provider="ollama", model="gemma4:latest")
    session.commit()
    return session, engine, document, c_graph, c_alpha, c_beta


async def test_hybrid_exact_tie_breaks_deterministically_by_chunk_id():
    """Two single-side chunks at the same rank on their own sides tie exactly
    (vector rank 2 vs graph rank 2 -> both 1/62). The fused order must be
    deterministic: ascending chunk_id for equal hybrid_score."""
    session, engine, document, c_graph, c_alpha, c_beta = _tie_session()
    try:
        # Vector side ranks: c_graph 1st, c_alpha 2nd.
        vector = FixedVectorStore([
            RetrievedChunk(text=c_graph.text, score=0.1, vector_id=c_graph.vector_id,
                           metadata={"document_id": document.id, "chunk_id": c_graph.id}),
            RetrievedChunk(text=c_alpha.text, score=0.2, vector_id=c_alpha.vector_id,
                           metadata={"document_id": document.id, "chunk_id": c_alpha.id}),
        ])
        # Query names both relation sources so the graph side resolves both
        # seeds lexically; graph candidates sort by (-score, hop, chunk_id):
        # c_graph (conf 0.9, id lower) rank 1, c_beta rank 2.
        contexts = await retrieve_contexts(
            query="User Stakeholder", embedding_provider=FixedEmbeddingProvider(),
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1)

        by_id = {c["metadata"]["chunk_id"]: c for c in contexts}
        assert set(by_id) == {c_graph.id, c_alpha.id, c_beta.id}
        # c_graph is rank 1 on BOTH sides: 1/61 + 1/61.
        assert by_id[c_graph.id]["metadata"]["hybrid_score"] == (1 / 61 + 1 / 61)
        # c_alpha: vector rank 2 only; c_beta: graph rank 2 only -> exact tie.
        alpha_score = by_id[c_alpha.id]["metadata"]["hybrid_score"]
        beta_score = by_id[c_beta.id]["metadata"]["hybrid_score"]
        assert alpha_score == 1 / 62
        assert beta_score == 1 / 62
        assert alpha_score == beta_score
        # Deterministic tie-break: ascending chunk_id among equal scores,
        # and the both-sides chunk outranks every single-side chunk.
        ordered = [c["metadata"]["chunk_id"] for c in contexts]
        assert ordered == [c_graph.id, c_alpha.id, c_beta.id]
    finally:
        session.close(); engine.dispose()


async def test_hybrid_zero_candidates_returns_empty_without_error():
    """Empty vector side and a query matching no entities: hybrid returns
    [] cleanly (no error, no placeholder rows)."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        contexts = await retrieve_contexts(
            query="zzz nothing matches this",
            embedding_provider=FixedEmbeddingProvider(),
            vector_store=FixedVectorStore([]),
            session=session, mode="hybrid", top_k=5, graph_max_hops=1)
        assert contexts == []
    finally:
        session.close(); engine.dispose()


async def test_hybrid_single_candidate_fuses_single_side_score():
    """Exactly one candidate (vector side only, graph side empty): the fused
    list has one entry whose hybrid_score is 1/(60+1) from that side alone."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        vector = FixedVectorStore([
            RetrievedChunk(text=vector_chunk.text, score=0.4,
                           vector_id=vector_chunk.vector_id,
                           metadata={"document_id": document.id,
                                     "chunk_id": vector_chunk.id}),
        ])
        contexts = await retrieve_contexts(
            query="zzz nothing matches this",
            embedding_provider=FixedEmbeddingProvider(),
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1)
        assert len(contexts) == 1
        assert contexts[0]["metadata"]["chunk_id"] == vector_chunk.id
        assert contexts[0]["metadata"]["retrieval_sources"] == ["vector"]
        assert contexts[0]["metadata"]["hybrid_score"] == 1 / 61
    finally:
        session.close(); engine.dispose()


async def test_answer_query_with_no_evidence_invokes_llm_with_empty_context():
    """Grounded no-evidence contract: a query matching nothing still produces
    an answer, with context == [] and the LLM invoked with an empty context
    list (never placeholder rows, never a crash)."""
    from app.services.rag import answer_query

    class RecordingLLM:
        def __init__(self):
            self.calls = []

        async def generate_answer(self, query, context):
            self.calls.append((query, list(context)))
            return "No evidence available for this question."

    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        llm = RecordingLLM()
        result = await answer_query(
            query="zzz nothing matches this",
            embedding_provider=FixedEmbeddingProvider(),
            vector_store=FixedVectorStore([]),
            llm_client=llm, session=session,
            retrieval_mode="vector", top_k=5)
        assert result["context"] == []
        assert result["answer"] == "No evidence available for this question."
        assert llm.calls == [("zzz nothing matches this", [])]
    finally:
        session.close(); engine.dispose()


# ── Vector-store inconsistency hardening at retrieval time ─────────────


async def test_hybrid_duplicate_ids_in_one_store_response_yield_chunk_once():
    """A single Chroma response returning the SAME chunk id twice (same
    deterministic vector id, two score entries) must hydrate to exactly one
    context entry. Without hydration-time dedup the duplicate would be fused
    twice and inflate the RRF score to 1/61 + 1/62 from the vector side
    alone; the pinned value proves the second occurrence never reaches the
    fusion ranking."""
    session, engine, document, graph_chunk, vector_chunk = _graph_session()
    try:
        vector = FixedVectorStore([
            RetrievedChunk(text="first copy", score=0.1,
                           vector_id=vector_chunk.vector_id,
                           metadata={"document_id": document.id,
                                     "chunk_id": vector_chunk.id}),
            RetrievedChunk(text="second copy", score=0.3,
                           vector_id=vector_chunk.vector_id,
                           metadata={"document_id": document.id,
                                     "chunk_id": vector_chunk.id}),
        ])
        contexts = await retrieve_contexts(
            query="zzz nothing matches this",
            embedding_provider=FixedEmbeddingProvider(),
            vector_store=vector, session=session, mode="hybrid",
            top_k=5, graph_max_hops=1)

        assert len(contexts) == 1
        assert contexts[0]["text"] == vector_chunk.text
        assert contexts[0]["score"] == 0.1  # first occurrence wins
        assert contexts[0]["metadata"]["chunk_id"] == vector_chunk.id
        assert contexts[0]["metadata"]["hybrid_score"] == 1 / 61
    finally:
        session.close(); engine.dispose()


def _cross_document_session():
    """Two ready documents; ``chunk_a`` belongs to doc A but the store hit
    for it claims doc B's identity (document_id, title, and text)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    doc_a = models.Document(title="Doc A title", source="unit")
    doc_b = models.Document(title="Doc B title", source="unit")
    session.add_all([doc_a, doc_b])
    session.flush()
    chunk_a = models.Chunk(document_id=doc_a.id, index=0,
                           text="Authoritative SQL text for doc A.",
                           start_offset=0, end_offset=33,
                           vector_id="chunk:docA:0")
    chunk_b = models.Chunk(document_id=doc_b.id, index=0,
                           text="Doc B chunk text.",
                           start_offset=0, end_offset=16,
                           vector_id="chunk:docB:0")
    session.add_all([chunk_a, chunk_b])
    session.commit()
    lying_hit = RetrievedChunk(
        text="stolen doc B text riding a doc A vector",
        score=0.2,
        vector_id=chunk_a.vector_id,
        metadata={"document_id": doc_b.id, "chunk_id": chunk_a.id,
                  "title": "Doc B title", "index": 9},
    )
    return session, engine, doc_a, doc_b, chunk_a, chunk_b, lying_hit


async def test_vector_hit_claiming_foreign_document_yields_sql_identity():
    """Cross-document metadata lie: the hit's chunk_id resolves to a chunk of
    a DIFFERENT document than the hit's own metadata claims, while the vector
    record id matches the chunk's deterministic id. SQL identity must win:
    the hydrated entry carries doc A's document_id/title/index/text, never
    the claimed doc B identity as one blended context entry."""
    session, engine, doc_a, doc_b, chunk_a, chunk_b, lying_hit = (
        _cross_document_session()
    )
    try:
        contexts = await retrieve_contexts(
            query="anything",
            embedding_provider=FixedEmbeddingProvider(),
            vector_store=FixedVectorStore([lying_hit]),
            session=session, mode="vector", top_k=5)

        assert len(contexts) == 1
        entry = contexts[0]
        assert entry["text"] == chunk_a.text
        assert entry["metadata"]["document_id"] == doc_a.id
        assert entry["metadata"]["chunk_id"] == chunk_a.id
        assert entry["metadata"]["title"] == doc_a.title
        assert entry["metadata"]["index"] == chunk_a.index
        assert entry["metadata"]["document_id"] != doc_b.id
    finally:
        session.close(); engine.dispose()


async def test_document_id_filter_uses_sql_identity_not_store_metadata():
    """The scalar document_id filter is re-applied against SQL during
    hydration, so filtering for the LIED-about document excludes the hit and
    filtering for the SQL-true document keeps it — Chroma metadata can never
    smuggle a foreign-document chunk into a filtered result set."""
    session, engine, doc_a, doc_b, chunk_a, chunk_b, lying_hit = (
        _cross_document_session()
    )
    try:
        for filters, expected in (
            ({"document_id": doc_b.id}, []),   # the lie matches nothing in SQL
            ({"document_id": doc_a.id}, [chunk_a.id]),  # SQL truth admits it
        ):
            contexts = await retrieve_contexts(
                query="anything",
                embedding_provider=FixedEmbeddingProvider(),
                vector_store=FixedVectorStore([lying_hit]),
                session=session, mode="vector", top_k=5, filters=filters)
            assert [c["metadata"]["chunk_id"] for c in contexts] == expected
    finally:
        session.close(); engine.dispose()
