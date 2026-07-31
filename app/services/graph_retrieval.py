"""Bounded relational traversal over persisted graph evidence."""
from __future__ import annotations

import re
from collections import deque

from sqlalchemy.orm import Session, joinedload

from app.persistence import models
from app.services.graph_extraction import canonicalize_entity_name

MAX_GRAPH_HOPS = 3
MAX_SEEDS = 20
MAX_EVIDENCE_ROWS = 5000


class GraphTraversalError(RuntimeError):
    """Base graph traversal failure."""


class GraphTraversalLimitError(GraphTraversalError):
    """Graph traversal exceeded a documented safety cap."""


class UnsupportedGraphFilter(GraphTraversalError):
    """A filter cannot be applied consistently to vector and graph evidence."""


def _query_mentions_entity(query: str, canonical_name: str) -> bool:
    canonical_query = canonicalize_entity_name(query)
    return re.search(
        rf"(?<!\w){re.escape(canonical_name)}(?!\w)", canonical_query
    ) is not None


def _validate_filters(filters: dict | None) -> dict:
    if not filters:
        return {}
    allowed = {"document_id", "title", "source", "tags"}
    unsupported = set(filters) - allowed
    if unsupported or any(isinstance(value, (dict, list)) for value in filters.values()):
        raise UnsupportedGraphFilter(
            "Hybrid graph filters support only scalar document_id, title, source, and tags"
        )
    return filters


def retrieve_graph_contexts(
    session: Session,
    *,
    query: str,
    max_hops: int,
    limit: int,
    filters: dict | None = None,
    seed_chunk_ids: list[int] | None = None,
) -> list[dict]:
    """Return deterministic multi-hop evidence from ready documents only."""
    if max_hops < 1 or max_hops > MAX_GRAPH_HOPS:
        raise ValueError(f"max_hops must be between 1 and {MAX_GRAPH_HOPS}")
    if limit < 1:
        return []
    filters = _validate_filters(filters)

    seeds = {
        entity.id
        for entity in session.query(models.GraphEntity).all()
        if _query_mentions_entity(query, entity.canonical_name)
    }
    if seed_chunk_ids:
        mentioned_ids = (
            session.query(models.EntityMention.entity_id)
            .join(models.GraphExtraction)
            .filter(models.GraphExtraction.chunk_id.in_(seed_chunk_ids))
            .distinct()
            .all()
        )
        seeds.update(entity_id for (entity_id,) in mentioned_ids)
    if not seeds:
        return []
    if len(seeds) > MAX_SEEDS:
        raise GraphTraversalLimitError(
            f"Graph seed count exceeds safety cap of {MAX_SEEDS}"
        )

    evidence_query = (
        session.query(models.GraphEdgeEvidence)
        .join(models.GraphEdgeEvidence.extraction)
        .join(models.GraphExtraction.chunk)
        .join(models.Chunk.document)
        .options(
            joinedload(models.GraphEdgeEvidence.edge).joinedload(models.GraphEdge.source),
            joinedload(models.GraphEdgeEvidence.edge).joinedload(models.GraphEdge.target),
            joinedload(models.GraphEdgeEvidence.extraction)
            .joinedload(models.GraphExtraction.chunk)
            .joinedload(models.Chunk.document),
        )
        .filter(models.Document.ingestion_status == "ready")
    )
    if "document_id" in filters:
        try:
            document_id = int(filters["document_id"])
        except (TypeError, ValueError) as exc:
            raise UnsupportedGraphFilter("document_id filter must be an integer") from exc
        evidence_query = evidence_query.filter(models.Document.id == document_id)
    if "title" in filters:
        evidence_query = evidence_query.filter(models.Document.title == str(filters["title"]))
    if "source" in filters:
        evidence_query = evidence_query.filter(models.Document.source == str(filters["source"]))
    if "tags" in filters:
        evidence_query = evidence_query.filter(models.Document.tags == str(filters["tags"]))

    evidence_rows = evidence_query.limit(MAX_EVIDENCE_ROWS + 1).all()
    if len(evidence_rows) > MAX_EVIDENCE_ROWS:
        raise GraphTraversalLimitError(
            f"Graph evidence exceeds safety cap of {MAX_EVIDENCE_ROWS}"
        )
    if not evidence_rows:
        return []

    adjacency: dict[int, set[int]] = {}
    for evidence in evidence_rows:
        edge = evidence.edge
        adjacency.setdefault(edge.source_entity_id, set()).add(edge.target_entity_id)
        adjacency.setdefault(edge.target_entity_id, set()).add(edge.source_entity_id)

    distances: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque((seed_id, 0) for seed_id in sorted(seeds))
    while queue:
        entity_id, distance = queue.popleft()
        previous = distances.get(entity_id)
        if previous is not None and previous <= distance:
            continue
        distances[entity_id] = distance
        if distance == max_hops:
            continue
        for neighbor in sorted(adjacency.get(entity_id, ())):
            queue.append((neighbor, distance + 1))

    candidates: list[tuple[int, models.GraphEdgeEvidence]] = []
    for evidence in evidence_rows:
        edge = evidence.edge
        endpoint_distances = [
            distances[entity_id]
            for entity_id in (edge.source_entity_id, edge.target_entity_id)
            if entity_id in distances
        ]
        if not endpoint_distances:
            continue
        hop = min(endpoint_distances) + 1
        if hop <= max_hops:
            candidates.append((hop, evidence))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].edge.source.canonical_name,
            item[1].edge.predicate,
            item[1].edge.target.canonical_name,
            item[1].edge_id,
            item[1].id,
        )
    )
    contexts = []
    for hop, evidence in candidates[:limit]:
        edge = evidence.edge
        chunk = evidence.extraction.chunk
        document = chunk.document
        contexts.append(
            {
                "text": chunk.text,
                "score": evidence.confidence / hop,
                "metadata": {
                    "document_id": document.id,
                    "chunk_id": chunk.id,
                    "title": document.title,
                    "retrieval_sources": ["graph"],
                    "edge_id": edge.id,
                    "evidence_id": evidence.id,
                    "graph": {
                        "source": edge.source.display_name,
                        "source_type": edge.source.entity_type,
                        "predicate": edge.predicate,
                        "target": edge.target.display_name,
                        "target_type": edge.target.entity_type,
                        "hop": hop,
                        "evidence": evidence.evidence_text,
                        "confidence": evidence.confidence,
                        "extraction_model": evidence.extraction.model,
                    },
                },
            }
        )
    return contexts
