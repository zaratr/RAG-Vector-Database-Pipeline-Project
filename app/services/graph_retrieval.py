"""Bounded relational traversal over persisted graph evidence."""
from __future__ import annotations

import re
from collections import deque
from typing import Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from app.persistence import models
from app.services.graph_extraction import canonicalize_entity_name

MAX_GRAPH_HOPS = 3
MAX_SEEDS = 20
MAX_EVIDENCE_ROWS = 5000
MAX_RETURNED_PATHS = 50
# Per-seed fair candidate budget (W2): every accepted seed contributes up to
# this many candidate paths to the global deterministic sort. Plan 10A.5
# selection semantics consider all seeds (up to MAX_SEEDS) and pick the top
# ``limit`` (<= MAX_RETURNED_PATHS) paths by the documented sort key, so no
# seed may be silently dropped because an earlier seed produced many paths.
# The budget keeps total candidates bounded at MAX_SEEDS * this value while
# preserving the plan's seed and returned-path caps.
MAX_SEED_CANDIDATE_PATHS = MAX_RETURNED_PATHS * 4

TraversalDirection = Literal["outbound", "inbound", "both"]


class GraphTraversalError(RuntimeError):
    """Base graph traversal failure."""


class GraphTraversalLimitError(GraphTraversalError):
    """Graph traversal exceeded a documented safety cap."""


class UnsupportedGraphFilter(GraphTraversalError):
    """A filter cannot be applied consistently to vector and graph evidence."""


class GraphPathStep(BaseModel):
    edge_id: int
    evidence_id: int
    source_entity_id: int
    source: str
    source_type: str
    predicate: str
    target_entity_id: int
    target: str
    target_type: str
    chunk_id: int
    document_id: int
    evidence: str
    confidence: float
    extraction_id: int
    extraction_model: str


class GraphPath(BaseModel):
    seed_entity_id: int
    terminal_entity_id: int
    hop_count: int
    steps: list[GraphPathStep]
    score: float


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
    # Validate document_id's scalar type up front: the SQL-side check in
    # apply_document_filters is skipped when traversal short-circuits (no
    # seeds/empty evidence), and an unsupported filter must be rejected,
    # never silently ignored, regardless of graph content (plan 10A.6).
    if "document_id" in filters and (
        isinstance(filters["document_id"], bool)
        or not isinstance(filters["document_id"], int)
    ):
        raise UnsupportedGraphFilter(
            "document_id filter must be an integer; booleans and "
            "non-integer forms are rejected"
        )
    return filters


def split_document_tags(stored_tags: str | None) -> list[str]:
    """Split stored comma-separated document tags and trim whitespace.

    Plan 10A.5 scalar filter matrix: ``tags`` is "one scalar tag; exact
    membership after splitting stored comma-separated tags and trimming
    whitespace". Empty elements are dropped. Mirrors the split performed by
    ``Chunk.get_chunk_metadata`` and is shared by graph retrieval and vector
    hydration so every side applies identical semantics.
    """
    if not stored_tags:
        return []
    return [tag.strip() for tag in stored_tags.split(",") if tag.strip()]


def document_ids_matching_tag(session: Session, tag: str) -> list[int]:
    """Return sorted ids of documents whose split+trimmed tags contain ``tag``."""
    rows = (
        session.query(models.Document.id, models.Document.tags)
        .filter(models.Document.tags.isnot(None))
        .all()
    )
    return sorted(
        document_id
        for document_id, stored_tags in rows
        if tag in split_document_tags(stored_tags)
    )


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
    evidence_query = apply_document_filters(evidence_query, filters, session)

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


# ---------------------------------------------------------------------------
# 10A.5 — directional path traversal returning complete GraphPath objects
# ---------------------------------------------------------------------------


def apply_document_filters(query, filters: dict | None, session: Session):
    """Apply the plan 10A.5 scalar filter matrix to a Document-joined query.

    ``document_id`` is integer equality with booleans and non-integer forms
    (strings like ``"7"``, floats like ``7.0``) rejected rather than coerced
    (W5), ``title`` and ``source`` are exact case-sensitive SQL equality, and
    ``tags`` is exact membership after splitting the stored comma-separated
    tags and trimming whitespace. Shared by graph traversal and vector
    hydration so identical filter values yield identical candidate semantics
    on both sides.
    """
    if not filters:
        return query
    if "document_id" in filters:
        document_id = filters["document_id"]
        if isinstance(document_id, bool) or not isinstance(document_id, int):
            raise UnsupportedGraphFilter(
                "document_id filter must be an integer; booleans and "
                "non-integer forms are rejected"
            )
        query = query.filter(models.Document.id == document_id)
    if "title" in filters:
        query = query.filter(models.Document.title == str(filters["title"]))
    if "source" in filters:
        query = query.filter(models.Document.source == str(filters["source"]))
    if "tags" in filters:
        matching_ids = document_ids_matching_tag(session, str(filters["tags"]))
        query = query.filter(models.Document.id.in_(matching_ids))
    return query


def _ready_evidence_rows(session: Session, filters: dict):
    query = (
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
    query = apply_document_filters(query, filters, session)
    return query.all()


def resolve_graph_seeds(
    session: Session,
    *,
    query: str,
    seed_chunk_ids: list[int] | None,
    filters: dict | None,
) -> list[int]:
    """Resolve seed entity IDs: lexical matches in the query + ready seed-chunk mentions.

    Apply document filters before accepting chunk-derived seeds. More than 20
    distinct seeds raises ``GraphTraversalLimitError``; no seeds returns ``[]``.
    """
    canonical_query = canonicalize_entity_name(query)
    seeds: set[int] = set()
    for entity in session.query(models.GraphEntity).all():
        if re.search(rf"(?<!\w){re.escape(entity.canonical_name)}(?!\w)", canonical_query):
            seeds.add(entity.id)
    if seed_chunk_ids:
        mentioned = (
            session.query(models.EntityMention.entity_id)
            .join(models.GraphExtraction)
            .join(models.Chunk)
            .join(models.Document)
            .filter(models.GraphExtraction.chunk_id.in_(seed_chunk_ids))
            .filter(models.Document.ingestion_status == "ready")
        )
        mentioned = apply_document_filters(mentioned, filters or {}, session)
        seeds.update(entity_id for (entity_id,) in mentioned.distinct().all())
    if len(seeds) > MAX_SEEDS:
        raise GraphTraversalLimitError(
            f"Graph seed count exceeds safety cap of {MAX_SEEDS}"
        )
    return sorted(seeds)


def _step_from_evidence(evidence: models.GraphEdgeEvidence) -> GraphPathStep:
    edge = evidence.edge
    chunk = evidence.extraction.chunk
    document = chunk.document
    return GraphPathStep(
        edge_id=edge.id,
        evidence_id=evidence.id,
        source_entity_id=edge.source_entity_id,
        source=edge.source.display_name,
        source_type=edge.source.entity_type,
        predicate=edge.predicate,
        target_entity_id=edge.target_entity_id,
        target=edge.target.display_name,
        target_type=edge.target.entity_type,
        chunk_id=chunk.id,
        document_id=document.id,
        evidence=evidence.evidence_text,
        confidence=evidence.confidence,
        extraction_id=evidence.extraction_id,
        extraction_model=evidence.extraction.model,
    )


def _direction_neighbors(edge: models.GraphEdge, direction: TraversalDirection):
    """Return the (from_entity, to_entity) pair honoring direction semantics.

    Stored edge orientation is always preserved in the returned step; direction
    only controls which entity is the traversal source for BFS expansion.
    """
    if direction == "outbound":
        return [(edge.source_entity_id, edge.target_entity_id)]
    if direction == "inbound":
        return [(edge.target_entity_id, edge.source_entity_id)]
    return [
        (edge.source_entity_id, edge.target_entity_id),
        (edge.target_entity_id, edge.source_entity_id),
    ]


def retrieve_graph_paths(
    session: Session,
    *,
    query: str,
    max_hops: int,
    direction: TraversalDirection,
    limit: int,
    filters: dict | None = None,
    seed_chunk_ids: list[int] | None = None,
) -> list[GraphPath]:
    """Return deterministic directional paths over ready evidence only.

    Paths cannot repeat an edge or entity except a one-step self-loop. Sort by
    ``(hop_count, canonical entity/edge sequence, evidence IDs)``. Score is
    ``min(confidence across steps) / hop_count``.
    """
    if max_hops < 1 or max_hops > MAX_GRAPH_HOPS:
        raise ValueError(f"max_hops must be between 1 and {MAX_GRAPH_HOPS}")
    if limit < 1:
        return []
    validated = _validate_filters(filters)
    seeds = resolve_graph_seeds(
        session, query=query, seed_chunk_ids=seed_chunk_ids, filters=validated
    )
    if not seeds:
        return []

    rows = _ready_evidence_rows(session, validated)
    if len(rows) > MAX_EVIDENCE_ROWS:
        raise GraphTraversalLimitError(
            f"Graph evidence exceeds safety cap of {MAX_EVIDENCE_ROWS}"
        )
    if not rows:
        return []

    # Index evidence rows by traversal-source entity for directed BFS.
    evidence_by_source: dict[int, list[models.GraphEdgeEvidence]] = {}
    for evidence in rows:
        for src, _dst in _direction_neighbors(evidence.edge, direction):
            evidence_by_source.setdefault(src, []).append(evidence)
    for key in evidence_by_source:
        evidence_by_source[key].sort(
            key=lambda e: (
                e.edge.target.canonical_name if direction != "inbound" else e.edge.source.canonical_name,
                e.edge.predicate,
                e.edge_id,
                e.id,
            )
        )

    # Every accepted seed (<= MAX_SEEDS) contributes candidates to the global
    # sort; a seed is never silently dropped because an earlier seed produced
    # many paths (W2). Boundedness comes from the per-seed budget inside
    # _bfs_paths, so the worst case is MAX_SEEDS * MAX_SEED_CANDIDATE_PATHS
    # candidates before the deterministic sort selects min(limit, 50).
    paths: list[GraphPath] = []
    for seed in seeds:
        paths.extend(_bfs_paths(seed, max_hops, direction, evidence_by_source))

    # Deduplicate by ordered evidence-ID sequence.
    seen: set[tuple[int, ...]] = set()
    unique: list[GraphPath] = []
    for path in paths:
        key = tuple(step.evidence_id for step in path.steps)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    unique.sort(
        key=lambda p: (
            p.hop_count,
            tuple(s.source for s in p.steps),
            tuple(s.predicate for s in p.steps),
            tuple(s.target for s in p.steps),
            tuple(s.evidence_id for s in p.steps),
        )
    )
    return unique[: min(limit, MAX_RETURNED_PATHS)]


def _bfs_paths(
    seed: int,
    max_hops: int,
    direction: TraversalDirection,
    evidence_by_source: dict[int, list[models.GraphEdgeEvidence]],
) -> list[GraphPath]:
    """Breadth-first expansion from ``seed`` yielding complete GraphPath objects.

    Results are capped at ``MAX_SEED_CANDIDATE_PATHS`` per seed (W2 fair
    budget). The cap is deterministic because both the per-entity evidence
    lists and the FIFO queue are explored in sorted order.
    """
    results: list[GraphPath] = []
    # Each frontier item: (current_entity, steps_so_far, visited_edges, visited_entities)
    initial = (seed, [], set(), {seed})
    queue: deque = deque([initial])
    while queue:
        if len(results) >= MAX_SEED_CANDIDATE_PATHS:
            break
        current, steps, visited_edges, visited_entities = queue.popleft()
        if len(steps) >= max_hops:
            continue
        for evidence in evidence_by_source.get(current, []):
            if len(results) >= MAX_SEED_CANDIDATE_PATHS:
                break
            if evidence.id in visited_edges:
                continue
            edge = evidence.edge
            for src, dst in _direction_neighbors(edge, direction):
                if src != current:
                    continue
                # Self-loop: a one-step path may revisit the same entity once.
                if dst in visited_entities and dst != current:
                    continue
                step = _step_from_evidence(evidence)
                new_steps = steps + [step]
                new_visited_edges = visited_edges | {evidence.id}
                new_visited_entities = visited_entities | {dst}
                min_conf = min(s.confidence for s in new_steps)
                path = GraphPath(
                    seed_entity_id=seed,
                    terminal_entity_id=dst,
                    hop_count=len(new_steps),
                    steps=new_steps,
                    score=min_conf / len(new_steps),
                )
                results.append(path)
                if dst != current:
                    queue.append(
                        (dst, new_steps, new_visited_edges, new_visited_entities)
                    )
    return results

