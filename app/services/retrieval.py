"""Retrieval utilities."""
from __future__ import annotations

import hashlib
import json
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
from app.services.retrieval_security import (
    Candidate,
    RetrievalSecurityPolicy,
    apply_security_filters,
    load_retrieval_security_policy_strict,
)

# Immutable process-wide policy: loaded once, never reloaded per request, and
# never replaced by an arbitrary default (plan §10B.3 fail-closed requirement).
_POLICY_CACHE: RetrievalSecurityPolicy | None = None


def get_retrieval_security_policy() -> RetrievalSecurityPolicy:
    """Return the immutable policy, loading it exactly once; fail closed."""
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        from app.config import get_settings
        from app.services.retrieval_security import (
            assert_policy_matches_runtime_regime,
        )

        settings = get_settings()
        policy = load_retrieval_security_policy_strict(
            settings.retrieval_security_policy_path
        )
        # R2a: the calibrated thresholds are only meaningful under the
        # embedding regime the policy was calibrated for (e.g. the committed
        # policy is fastembed/jina-clip-v1 calibrated; the local hash regime
        # sits far outside it). Refuse at load — typed and fail-closed —
        # instead of silently fail-closing retrieval with misleading
        # rejected_distance decisions. A pre-installed _POLICY_CACHE (the
        # shipped hermetic-test hook) remains a full bypass.
        assert_policy_matches_runtime_regime(policy, settings)
        _POLICY_CACHE = policy
    return _POLICY_CACHE


def reset_retrieval_security_policy_cache() -> None:
    """Test hook: forget the cached policy so the next call reloads strictly."""
    global _POLICY_CACHE
    _POLICY_CACHE = None


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
        # Numeric SQL chunk identity: fused contexts key and tie-break by the
        # integer chunk id (2 before 10). The stringified form would order
        # multi-digit ids lexicographically and invert the deterministic
        # (-hybrid_score, chunk_id) ordering.
        return ("chunk", int(chunk_id))
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
    # 10D union (red-team harness scope): the exact Chroma-native operator
    # form ``{"document_id": {"$in": [<int>, ...]}}`` is pushed to the vector
    # index by ``vector_store.query`` and used by the Phase 10D red-team
    # harness to scope a fixture's candidates before ranking. SQL remains
    # the authority, so the operator is re-applied here as an exact integer
    # membership with the same strict W5 typing ``apply_document_filters``
    # enforces for scalars: booleans and non-integer elements are rejected,
    # never coerced (falling through to the scalar rejection below). Every
    # other key/value keeps its scalar-matrix handling untouched, and graph
    # and hybrid modes still reject operator forms via
    # ``graph_retrieval._validate_filters``.
    sql_filters = dict(filters or {})
    scope = sql_filters.get("document_id")
    if (
        isinstance(scope, dict)
        and set(scope) == {"$in"}
        and isinstance(scope["$in"], list)
        and scope["$in"]
        and all(
            not isinstance(item, bool) and isinstance(item, int)
            for item in scope["$in"]
        )
    ):
        chunks = chunks.filter(models.Document.id.in_(scope["$in"]))
        del sql_filters["document_id"]
    chunks = apply_document_filters(chunks, sql_filters, session).all()
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


async def retrieve_contexts_detailed(
    *,
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    session: Session,
    mode: Literal["vector", "graph", "hybrid"] = "vector",
    top_k: int = 5,
    graph_max_hops: int = 2,
    filters: dict | None = None,
) -> tuple[List[dict], list[dict]]:
    """Retrieve vector, graph, or reciprocal-rank-fused hybrid evidence.

    Returns ``(contexts, retrieval_decisions)`` where contexts are the
    post-security survivors (top_k-limited) and retrieval_decisions is the
    full per-candidate decision list (selected/rejected_*) ready for audit
    persistence.
    """
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
        contexts, decisions = _apply_security_filter(session, vector_contexts, mode="vector")
        return contexts[:top_k], decisions
    if mode == "graph":
        contexts, decisions = _apply_security_filter(session, graph_contexts, mode="graph")
        return contexts[:top_k], decisions

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
        # Exact RRF-60 sum, never rounded: hybrid_score is the
        # sum of 1/(60+rank) over the candidate's sides, and rounding would
        # perturb that arithmetic (and could fabricate ties).
        context["metadata"]["hybrid_score"] = rrf_scores[key]
    hybrid_results = sorted(
        fused.values(),
        key=lambda item: (
            -item["metadata"]["hybrid_score"],
            _context_key(item),
        ),
    )
    contexts, decisions = _apply_security_filter(session, hybrid_results, mode="hybrid")
    return contexts[:top_k], decisions


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
    """Backward-compatible wrapper returning only the survivor contexts."""
    contexts, _ = await retrieve_contexts_detailed(
        query=query,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        session=session,
        mode=mode,
        top_k=top_k,
        graph_max_hops=graph_max_hops,
        filters=filters,
    )
    return contexts


def _apply_security_filter(session: Session, contexts: list[dict], mode: str = "vector") -> tuple[list[dict], list[dict]]:
    """Apply retrieval poisoning controls (distance, duplicate, caps) to contexts.

    Builds Candidate objects from SQL-hydrated contexts, applies the mode-aware
    security filters (native pre-rank order per the §10B.3 ordering table,
    distance applicability by origin, duplicate/cap controls, graph-origin
    annotation), and returns ``(selected_contexts, decisions)`` where
    ``decisions`` is the full per-candidate final decision list ready for
    audit persistence. The immutable policy is loaded once per process and
    fail-closed.
    """
    if not contexts:
        return [], []

    # Immutable, fail-closed: never an arbitrary default threshold.
    policy = get_retrieval_security_policy()

    candidates: list[Candidate] = []
    context_by_chunk: dict[int, dict] = {}
    for ctx in contexts:
        meta = ctx.get("metadata") or {}
        chunk_id = meta.get("chunk_id")
        if chunk_id is None:
            continue
        doc_id = meta.get("document_id")
        source = meta.get("source") or "unknown"
        text = ctx.get("text", "")
        score = ctx.get("score", 1.0)
        # Determine origin from retrieval_sources metadata.
        sources = meta.get("retrieval_sources", ["vector"])
        if "graph" in sources and "vector" in sources:
            origin = "hybrid_both"
        elif "graph" in sources:
            origin = "graph"
        elif "vector" in sources:
            origin = "vector"
        else:
            origin = "vector"

        # Look up trust tier and grounding inputs from SQL.
        trust_tier = "standard"
        document_ready = True
        ingestion_origin = ""
        if doc_id:
            doc = session.get(models.Document, doc_id)
            if doc:
                trust_tier = doc.trust_tier or "standard"
                document_ready = doc.ingestion_status == "ready"
                ingestion_origin = doc.ingestion_origin or ""
        # Graph evidence with valid chunk/document/extraction FKs: candidates
        # reached through the graph lane were hydrated from validated graph
        # paths, so graph support is the recorded evidence indicator.
        has_graph_evidence = "graph" in sources

        # Native L2 distance exists only for vector-supported candidates; the
        # Chroma score IS the distance for vector hits and must never be
        # compared to max_distance for graph-only candidates.
        distance = score if "vector" in sources else None
        cand = Candidate(
            chunk_id=chunk_id,
            document_id=doc_id or 0,
            source=source,
            text=text,
            native_score=score,
            trust_tier=trust_tier,
            document_ready=document_ready,
            origin=origin,
            distance=distance,
            graph_score=meta.get("graph_score"),
            hop=meta.get("min_hop", meta.get("hop")) if isinstance(
                meta.get("min_hop", meta.get("hop")), int
            ) else None,
            hybrid_score=meta.get("hybrid_score"),
            ingestion_origin=ingestion_origin,
            has_graph_evidence=has_graph_evidence,
        )
        candidates.append(cand)
        context_by_chunk[chunk_id] = ctx

    decisions = apply_security_filters(candidates, policy=policy, mode=mode)
    # Build survivor list in the mode's pre-rank order (filter output order)
    # and the persistable decision dicts for every candidate.
    result = []
    decision_rows: list[dict] = []
    for d in decisions:
        ctx = context_by_chunk.get(d.chunk_id)
        if ctx is None:
            continue
        document_id = (ctx.get("metadata") or {}).get("document_id")
        text = ctx.get("text", "")
        decision_rows.append({
            "chunk_id": d.chunk_id,
            "document_id": document_id,
            "document_id_snapshot": document_id or 0,
            "chunk_id_snapshot": d.chunk_id,
            "decision": d.decision,
            "native_score": d.native_score,
            "provenance_score": d.provenance_score,
            "reason_codes": json.dumps(sorted(d.reason_codes)),
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        if d.decision == "selected" and ctx is not None:
            result.append(ctx)
    return result, decision_rows
