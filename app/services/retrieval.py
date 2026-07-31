"""Retrieval utilities."""
from __future__ import annotations

from typing import List, Literal

from sqlalchemy.orm import Session

from app.persistence import models
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore
from app.services.graph_retrieval import retrieve_graph_contexts


async def retrieve(
    *, query: str, embedding_provider: EmbeddingProvider, vector_store: VectorStore, top_k: int = 5, filters: dict | None = None
) -> List[dict]:
    query_embedding = (await embedding_provider.embed_texts([query]))[0]
    results = await vector_store.query(query_embedding, top_k=top_k, filters=filters)
    return [
        {**result.model_dump(), "_vector_id": result.vector_id}
        for result in results
    ]


def _context_key(context: dict) -> tuple:
    metadata = context.get("metadata") or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        return ("chunk", str(chunk_id))
    return ("text", context.get("text", ""))


def _ready_vector_contexts(session: Session, contexts: list[dict]) -> list[dict]:
    """Hydrate Chroma hits from SQL and discard stale/non-ready vectors."""
    requested_ids: list[int] = []
    for context in contexts:
        try:
            requested_ids.append(int((context.get("metadata") or {})["chunk_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not requested_ids:
        return []
    chunks = (
        session.query(models.Chunk)
        .join(models.Chunk.document)
        .filter(
            models.Chunk.id.in_(requested_ids),
            models.Document.ingestion_status == "ready",
        )
        .all()
    )
    by_id = {chunk.id: chunk for chunk in chunks}
    hydrated = []
    seen_chunk_ids: set[int] = set()
    for context in contexts:
        metadata = context.get("metadata") or {}
        try:
            chunk = by_id[int(metadata["chunk_id"])]
        except (KeyError, TypeError, ValueError):
            continue
        if context.get("_vector_id") != chunk.vector_id:
            continue
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)
        # Chroma metadata is an index hint, never the provenance authority.
        # Preserve optional vector-only fields while forcing SQL identity and
        # document metadata to win when stale IDs have been reused.
        hydrated_metadata = dict(metadata)
        hydrated_metadata.update(chunk.get_chunk_metadata())
        hydrated.append(
            {
                "text": chunk.text,
                "score": context["score"],
                "metadata": hydrated_metadata,
            }
        )
    return hydrated


async def retrieve_contexts(
    *,
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    session: Session,
    mode: Literal["vector", "graph", "hybrid"] = "vector",
    top_k: int = 5,
    graph_max_hops: int = 2,
    filters: dict | None = None,
) -> List[dict]:
    """Retrieve vector, graph, or reciprocal-rank-fused hybrid evidence."""
    vector_contexts: list[dict] = []
    graph_contexts: list[dict] = []
    if mode in {"vector", "hybrid"}:
        vector_contexts = await retrieve(
            query=query,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            top_k=top_k,
            filters=filters,
        )
        vector_contexts = _ready_vector_contexts(session, vector_contexts)
        for context in vector_contexts:
            metadata = context.setdefault("metadata", {})
            metadata["retrieval_sources"] = ["vector"]
    if mode in {"graph", "hybrid"}:
        graph_contexts = retrieve_graph_contexts(
            session,
            query=query,
            max_hops=graph_max_hops,
            limit=max(top_k * 3, top_k),
            filters=filters,
            seed_chunk_ids=[
                int(context["metadata"]["chunk_id"])
                for context in vector_contexts
                if (context.get("metadata") or {}).get("chunk_id") is not None
            ],
        )

    if mode == "vector":
        return vector_contexts[:top_k]
    if mode == "graph":
        merged_graph: dict[tuple, dict] = {}
        for context in graph_contexts:
            key = _context_key(context)
            graph_path = context["metadata"]["graph"]
            if key not in merged_graph:
                metadata = dict(context["metadata"])
                metadata.pop("graph")
                metadata["graph_paths"] = [graph_path]
                merged_graph[key] = {
                    "text": context["text"],
                    "score": context["score"],
                    "metadata": metadata,
                }
            else:
                merged_graph[key]["metadata"]["graph_paths"].append(graph_path)
                merged_graph[key]["score"] = max(
                    merged_graph[key]["score"], context["score"]
                )
        return list(merged_graph.values())[:top_k]

    fused: dict[tuple, dict] = {}
    rrf_scores: dict[tuple, float] = {}
    for source_name, contexts in (
        ("vector", vector_contexts),
        ("graph", graph_contexts),
    ):
        for rank, context in enumerate(contexts, start=1):
            key = _context_key(context)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
            graph_path = (context.get("metadata") or {}).get("graph")
            if key not in fused:
                metadata = dict(context.get("metadata") or {})
                metadata.pop("graph", None)
                metadata["retrieval_sources"] = [source_name]
                if graph_path is not None:
                    metadata["graph_paths"] = [graph_path]
                fused[key] = {
                    "text": context["text"],
                    "score": context["score"],
                    "metadata": metadata,
                }
                continue
            metadata = fused[key]["metadata"]
            if source_name not in metadata["retrieval_sources"]:
                metadata["retrieval_sources"].append(source_name)
            if graph_path is not None:
                metadata.setdefault("graph_paths", []).append(graph_path)

    for key, context in fused.items():
        context["metadata"]["hybrid_score"] = rrf_scores[key]
    return sorted(
        fused.values(),
        key=lambda item: (
            -item["metadata"]["hybrid_score"],
            _context_key(item),
        ),
    )[:top_k]
