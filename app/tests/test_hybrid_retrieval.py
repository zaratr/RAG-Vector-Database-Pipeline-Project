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
