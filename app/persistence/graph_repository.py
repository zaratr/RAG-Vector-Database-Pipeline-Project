"""Persistence operations for normalized graph extraction provenance."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.persistence import models
from app.services.graph_extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ExtractedEntity,
    ExtractedRelation,
)


def _get_or_create_entity(
    session: Session, extracted: ExtractedEntity
) -> models.GraphEntity:
    entity = (
        session.query(models.GraphEntity)
        .filter(
            models.GraphEntity.canonical_name == extracted.canonical_name,
            models.GraphEntity.entity_type == extracted.entity_type,
        )
        .one_or_none()
    )
    if entity is None:
        entity = models.GraphEntity(
            canonical_name=extracted.canonical_name,
            display_name=extracted.name,
            entity_type=extracted.entity_type,
        )
        session.add(entity)
        session.flush()
    return entity


def _mention_offsets(chunk_text: str, surface_form: str) -> tuple[int, int]:
    start = chunk_text.find(surface_form)
    if start < 0:
        raise ValueError(f"Entity surface form is not present in source chunk: {surface_form}")
    return start, start + len(surface_form)


def _get_or_create_mention(
    session: Session,
    *,
    entity: models.GraphEntity,
    extraction: models.GraphExtraction,
    surface_form: str,
    chunk_text: str,
) -> models.EntityMention:
    mention = (
        session.query(models.EntityMention)
        .filter(
            models.EntityMention.entity_id == entity.id,
            models.EntityMention.extraction_id == extraction.id,
        )
        .one_or_none()
    )
    if mention is None:
        start, end = _mention_offsets(chunk_text, surface_form)
        mention = models.EntityMention(
            entity_id=entity.id,
            extraction_id=extraction.id,
            surface_form=surface_form,
            start_offset=start,
            end_offset=end,
        )
        session.add(mention)
    return mention


def _get_or_create_edge(
    session: Session,
    *,
    source: models.GraphEntity,
    predicate: str,
    target: models.GraphEntity,
) -> models.GraphEdge:
    edge = (
        session.query(models.GraphEdge)
        .filter(
            models.GraphEdge.source_entity_id == source.id,
            models.GraphEdge.predicate == predicate,
            models.GraphEdge.target_entity_id == target.id,
        )
        .one_or_none()
    )
    if edge is None:
        edge = models.GraphEdge(
            source_entity_id=source.id,
            predicate=predicate,
            target_entity_id=target.id,
        )
        session.add(edge)
        session.flush()
    return edge


def persist_chunk_extraction(
    session: Session,
    *,
    chunk: models.Chunk,
    relations: list[ExtractedRelation],
    provider: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> models.GraphExtraction:
    """Persist one successful extraction run and its normalized provenance."""
    extraction = models.GraphExtraction(
        chunk_id=chunk.id,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
        status="empty" if not relations else "succeeded",
    )
    session.add(extraction)
    session.flush()

    for relation in relations:
        source = _get_or_create_entity(session, relation.source)
        target = _get_or_create_entity(session, relation.target)
        _get_or_create_mention(
            session,
            entity=source,
            extraction=extraction,
            surface_form=relation.source.name,
            chunk_text=chunk.text,
        )
        _get_or_create_mention(
            session,
            entity=target,
            extraction=extraction,
            surface_form=relation.target.name,
            chunk_text=chunk.text,
        )
        edge = _get_or_create_edge(
            session,
            source=source,
            predicate=relation.predicate,
            target=target,
        )
        existing_evidence = (
            session.query(models.GraphEdgeEvidence)
            .filter(
                models.GraphEdgeEvidence.edge_id == edge.id,
                models.GraphEdgeEvidence.extraction_id == extraction.id,
                models.GraphEdgeEvidence.evidence_start == relation.evidence_start,
                models.GraphEdgeEvidence.evidence_end == relation.evidence_end,
            )
            .one_or_none()
        )
        if existing_evidence is None:
            session.add(
                models.GraphEdgeEvidence(
                    edge_id=edge.id,
                    extraction_id=extraction.id,
                    evidence_text=relation.evidence,
                    evidence_start=relation.evidence_start,
                    evidence_end=relation.evidence_end,
                    confidence=relation.confidence,
                )
            )
        elif relation.confidence > existing_evidence.confidence:
            existing_evidence.confidence = relation.confidence
    session.flush()
    return extraction
