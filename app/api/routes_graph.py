"""Graph inspection endpoints (Task 10A.7).

Exposes the persisted graph for verifiable entity, relationship, and path
inspection. All inspection routes are read-only over ready-document evidence
and never call the embedding/Chroma layer.
"""
from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.core.db import get_db
from app.persistence import models
from app.services.graph_retrieval import (
    GraphTraversalLimitError,
    UnsupportedGraphFilter,
    retrieve_graph_paths,
)

router = APIRouter(prefix="/graph", tags=["graph"])


# ---------------------------------------------------------------------------
# GET /graph/entities
# ---------------------------------------------------------------------------


class GraphEntityItem(BaseModel):
    id: int
    canonical_name: str
    display_name: str
    entity_type: str
    mention_count: int
    evidence_count: int


class GraphEntityListResponse(BaseModel):
    items: List[GraphEntityItem]
    total: int
    limit: int
    offset: int


_ALLOWED_ENTITY_TYPES = {
    "person", "organization", "location", "product", "project",
    "technology", "concept", "event", "other",
}


@router.get("/entities", response_model=GraphEntityListResponse)
async def list_entities(
    name: str | None = None,
    match: Literal["exact", "prefix", "contains"] = "exact",
    entity_type: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> GraphEntityListResponse:
    if entity_type is not None and entity_type not in _ALLOWED_ENTITY_TYPES:
        raise HTTPException(status_code=422, detail="unsupported entity_type")

    query = (
        session.query(models.GraphEntity)
        .join(models.EntityMention, models.EntityMention.entity_id == models.GraphEntity.id)
        .join(models.GraphExtraction, models.EntityMention.extraction_id == models.GraphExtraction.id)
        .join(models.Chunk, models.GraphExtraction.chunk_id == models.Chunk.id)
        .join(models.Document, models.Chunk.document_id == models.Document.id)
        .filter(models.Document.ingestion_status == "ready")
        .distinct()
    )
    if name is not None:
        if match == "exact":
            query = query.filter(models.GraphEntity.canonical_name == name.casefold())
        elif match == "prefix":
            query = query.filter(models.GraphEntity.canonical_name.like(f"{name.casefold()}%"))
        else:  # contains
            query = query.filter(models.GraphEntity.canonical_name.like(f"%{name.casefold()}%"))
    if entity_type is not None:
        query = query.filter(models.GraphEntity.entity_type == entity_type)

    total = query.count()
    rows = (
        query.order_by(
            models.GraphEntity.canonical_name,
            models.GraphEntity.entity_type,
            models.GraphEntity.id,
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    items: List[GraphEntityItem] = []
    for entity in rows:
        mention_count, evidence_count = _ready_counts(session, entity.id)
        items.append(
            GraphEntityItem(
                id=entity.id,
                canonical_name=entity.canonical_name,
                display_name=entity.display_name,
                entity_type=entity.entity_type,
                mention_count=mention_count,
                evidence_count=evidence_count,
            )
        )
    return GraphEntityListResponse(items=items, total=total, limit=limit, offset=offset)


def _ready_counts(session: Session, entity_id: int) -> tuple[int, int]:
    """Count ready-document mentions and evidence rows for an entity."""
    mention_count = (
        session.query(func.count(models.EntityMention.id))
        .join(models.GraphExtraction, models.EntityMention.extraction_id == models.GraphExtraction.id)
        .join(models.Chunk, models.GraphExtraction.chunk_id == models.Chunk.id)
        .join(models.Document, models.Chunk.document_id == models.Document.id)
        .filter(
            models.EntityMention.entity_id == entity_id,
            models.Document.ingestion_status == "ready",
        )
        .scalar()
        or 0
    )
    evidence_count = (
        session.query(func.count(models.GraphEdgeEvidence.id))
        .join(models.GraphEdge, models.GraphEdgeEvidence.edge_id == models.GraphEdge.id)
        .join(models.GraphExtraction, models.GraphEdgeEvidence.extraction_id == models.GraphExtraction.id)
        .join(models.Chunk, models.GraphExtraction.chunk_id == models.Chunk.id)
        .join(models.Document, models.Chunk.document_id == models.Document.id)
        .filter(
            (models.GraphEdge.source_entity_id == entity_id)
            | (models.GraphEdge.target_entity_id == entity_id),
            models.Document.ingestion_status == "ready",
        )
        .scalar()
        or 0
    )
    return mention_count, evidence_count


# ---------------------------------------------------------------------------
# GET /graph/entities/{entity_id}/relationships
# ---------------------------------------------------------------------------


class RelationshipEvidence(BaseModel):
    evidence_id: int
    document_id: int
    chunk_id: int
    extraction_id: int
    text: str
    start: int
    end: int
    confidence: float
    model: str


class RelationshipEndpoint(BaseModel):
    id: int
    name: str
    type: str


class RelationshipItem(BaseModel):
    edge_id: int
    source: RelationshipEndpoint
    predicate: str
    target: RelationshipEndpoint
    evidence: List[RelationshipEvidence]


class RelationshipListResponse(BaseModel):
    entity_id: int
    direction: Literal["outbound", "inbound", "both"]
    items: List[RelationshipItem]
    total: int
    limit: int
    offset: int


@router.get(
    "/entities/{entity_id}/relationships", response_model=RelationshipListResponse
)
async def list_relationships(
    entity_id: int,
    direction: Literal["outbound", "inbound", "both"] = "outbound",
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> RelationshipListResponse:
    entity = session.get(models.GraphEntity, entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    edge_filter = []
    if direction == "outbound":
        edge_filter.append(models.GraphEdge.source_entity_id == entity_id)
    elif direction == "inbound":
        edge_filter.append(models.GraphEdge.target_entity_id == entity_id)
    else:
        edge_filter.append(
            (models.GraphEdge.source_entity_id == entity_id)
            | (models.GraphEdge.target_entity_id == entity_id)
        )

    all_edges = (
        session.query(models.GraphEdge)
        .filter(*edge_filter)
        .order_by(
            models.GraphEdge.source_entity_id,
            models.GraphEdge.predicate,
            models.GraphEdge.target_entity_id,
            models.GraphEdge.id,
        )
        .all()
    )
    items: List[RelationshipItem] = []
    for edge in all_edges:
        evidence_rows = _ready_edge_evidence(session, edge.id)
        if not evidence_rows:
            continue
        source = session.get(models.GraphEntity, edge.source_entity_id)
        target = session.get(models.GraphEntity, edge.target_entity_id)
        ev_models = [
            RelationshipEvidence(
                evidence_id=ev.id,
                document_id=ev.extraction.chunk.document_id,
                chunk_id=ev.extraction.chunk_id,
                extraction_id=ev.extraction_id,
                text=ev.evidence_text,
                start=ev.evidence_start,
                end=ev.evidence_end,
                confidence=ev.confidence,
                model=ev.extraction.model,
            )
            for ev in evidence_rows
        ]
        items.append(
            RelationshipItem(
                edge_id=edge.id,
                source=RelationshipEndpoint(id=source.id, name=source.display_name, type=source.entity_type),
                predicate=edge.predicate,
                target=RelationshipEndpoint(id=target.id, name=target.display_name, type=target.entity_type),
                evidence=ev_models,
            )
        )
    total = len(items)
    page = items[offset : offset + limit]
    return RelationshipListResponse(
        entity_id=entity_id, direction=direction, items=page, total=total, limit=limit, offset=offset
    )


def _ready_edge_evidence(session: Session, edge_id: int):
    return (
        session.query(models.GraphEdgeEvidence)
        .join(models.GraphExtraction, models.GraphEdgeEvidence.extraction_id == models.GraphExtraction.id)
        .join(models.Chunk, models.GraphExtraction.chunk_id == models.Chunk.id)
        .join(models.Document, models.Chunk.document_id == models.Document.id)
        .options(
            joinedload(models.GraphEdgeEvidence.extraction).joinedload(models.GraphExtraction.chunk),
        )
        .filter(
            models.GraphEdgeEvidence.edge_id == edge_id,
            models.Document.ingestion_status == "ready",
        )
        .order_by(models.Document.id, models.Chunk.id, models.GraphEdgeEvidence.id)
        .all()
    )


# ---------------------------------------------------------------------------
# POST /graph/paths
# ---------------------------------------------------------------------------


class GraphPathsRequest(BaseModel):
    query: str
    max_hops: int = 3
    direction: Literal["outbound", "inbound", "both"] = "outbound"
    limit: int = 20
    filters: dict | None = None


class GraphPathsResponse(BaseModel):
    paths: List[dict]


@router.post("/paths", response_model=GraphPathsResponse)
async def post_paths(
    request: GraphPathsRequest,
    session: Session = Depends(get_db),
) -> GraphPathsResponse:
    try:
        paths = retrieve_graph_paths(
            session,
            query=request.query,
            max_hops=request.max_hops,
            direction=request.direction,
            limit=request.limit,
            filters=request.filters,
        )
    except GraphTraversalLimitError as exc:
        raise HTTPException(status_code=503, detail="Graph traversal limit exceeded") from exc
    except UnsupportedGraphFilter as exc:
        raise HTTPException(status_code=422, detail="Unsupported graph filter") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GraphPathsResponse(paths=[p.model_dump() for p in paths])
