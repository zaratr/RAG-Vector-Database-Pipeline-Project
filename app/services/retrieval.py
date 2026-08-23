"""Retrieval utilities."""
from __future__ import annotations

from typing import List, Literal

from sqlalchemy.orm import Session

from app.persistence import models
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore
from app.services.graph_retrieval import (
    GraphTraversalLimitError,
    apply_document_filters,
    retrieve_graph_contexts,
    retrieve_graph_paths,
)


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


def _ready_vector_contexts(
    session: Session, contexts: list[dict], filters: dict | None = None
) -> list[dict]:
    """Hydrate Chroma hits from SQL and discard stale/non-ready vectors.

    Scalar filters are re-applied against SQL here so the same filter value
    yields identical candidate semantics in vector, graph, and hybrid modes
    (Chroma metadata is an index hint, never the provenance authority — the
    tags membership filter cannot be expressed as a Chroma where clause).
    """
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
    )
    chunks = apply_document_filters(chunks, filters, session).all()
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


def _graph_contexts_from_paths(
    session: Session,
    *,
    query: str,
    max_hops: int,
    limit: int,
    filters: dict | None,
    seed_chunk_ids: list[int],
) -> list[dict]:
    """Build graph candidate contexts from complete ``GraphPath`` objects.

    Each distinct ``chunk_id`` referenced by a path step becomes a candidate.
    A chunk's ``graph_score`` is the maximum score among paths containing it;
    ``graph_paths`` carries the full path objects (deduplicated by ordered
    evidence-ID sequence, capped at 10 per chunk).
    """
    paths = retrieve_graph_paths(
        session,
        query=query,
        max_hops=max_hops,
        direction="outbound",
        limit=limit,
        filters=filters,
        seed_chunk_ids=seed_chunk_ids or None,
    )
    # Map chunk_id -> (paths containing it, ready chunk).
    chunk_to_paths: dict[int, list] = {}
    for path in paths:
        for step in path.steps:
            chunk_to_paths.setdefault(step.chunk_id, []).append(path)

    if not chunk_to_paths:
        return []
    ready_chunks = (
        session.query(models.Chunk)
        .join(models.Chunk.document)
        .filter(
            models.Chunk.id.in_(list(chunk_to_paths)),
            models.Document.ingestion_status == "ready",
        )
        .all()
    )
    by_id = {chunk.id: chunk for chunk in ready_chunks}

    contexts: list[dict] = []
    for chunk_id, containing in chunk_to_paths.items():
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        # Dedup paths by ordered evidence-ID sequence; sort by score then hops.
        unique_paths: dict[tuple, object] = {}
        for path in containing:
            key = tuple(s.evidence_id for s in path.steps)
            if key not in unique_paths:
                unique_paths[key] = path
        sorted_paths = sorted(
            unique_paths.values(),
            key=lambda p: (-p.score, p.hop_count, tuple(s.evidence_id for s in p.steps)),
        )[:10]
        graph_score = max(p.score for p in sorted_paths)
        min_hop = min(
            (i for p in sorted_paths for i, s in enumerate(p.steps, start=1) if s.chunk_id == chunk_id),
            default=max_hops,
        )
        contexts.append(
            {
                "text": chunk.text,
                "score": graph_score,
                "metadata": {
                    **chunk.get_chunk_metadata(),
                    "retrieval_sources": ["graph"],
                    "graph_paths": [p.model_dump() for p in sorted_paths],
                    "graph_score": graph_score,
                    "score_type": "graph_path_score",
                    "min_hop": min_hop,
                    "chunk_id": chunk.id,
                },
            }
        )
    # Graph candidate order: (-graph_score, minimum hop, chunk_id).
    contexts.sort(key=lambda c: (-c["metadata"]["graph_score"], c["metadata"]["min_hop"], c["metadata"]["chunk_id"]))
    return contexts


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
        vector_contexts = _ready_vector_contexts(session, vector_contexts, filters)
        for context in vector_contexts:
            metadata = context.setdefault("metadata", {})
            metadata["retrieval_sources"] = ["vector"]
            metadata["score_type"] = "vector_distance"
    if mode in {"graph", "hybrid"}:
        seed_chunk_ids = [
            int(context["metadata"]["chunk_id"])
            for context in vector_contexts
            if (context.get("metadata") or {}).get("chunk_id") is not None
        ]
        graph_contexts = _graph_contexts_from_paths(
            session,
            query=query,
            max_hops=graph_max_hops,
            limit=max(top_k * 3, top_k),
            filters=filters,
            seed_chunk_ids=seed_chunk_ids,
        )

    if mode == "vector":
        return vector_contexts[:top_k]
    if mode == "graph":
        # Graph contexts already carry full graph_paths + graph_score metadata.
        return graph_contexts[:top_k]

    fused: dict[tuple, dict] = {}
    rrf_scores: dict[tuple, float] = {}
    for source_name, contexts in (
        ("vector", vector_contexts),
        ("graph", graph_contexts),
    ):
        for rank, context in enumerate(contexts, start=1):
            key = _context_key(context)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
            graph_paths = (context.get("metadata") or {}).get("graph_paths")
            graph_score = (context.get("metadata") or {}).get("graph_score")
            if key not in fused:
                metadata = dict(context.get("metadata") or {})
                metadata["retrieval_sources"] = [source_name]
                if source_name == "graph":
                    metadata["score_type"] = "graph_path_score"
                fused[key] = {
                    "text": context["text"],
                    "score": context["score"],
                    "metadata": metadata,
                }
                continue
            metadata = fused[key]["metadata"]
            if source_name not in metadata["retrieval_sources"]:
                metadata["retrieval_sources"].append(source_name)
            # Hybrid chunk from both: preserve native vector distance in score,
            # add graph_score + hybrid_score separately.
            if source_name == "vector":
                metadata["score"] = context["score"]
                metadata["score_type"] = "vector_distance"
            if graph_paths:
                existing = metadata.setdefault("graph_paths", [])
                for gp in graph_paths:
                    if gp not in existing:
                        existing.append(gp)
            if graph_score is not None:
                metadata["graph_score"] = graph_score

    for key, context in fused.items():
        context["metadata"]["hybrid_score"] = round(rrf_scores[key], 6)
    return sorted(
        fused.values(),
        key=lambda item: (
            -item["metadata"]["hybrid_score"],
            _context_key(item),
        ),
    )[:top_k]
