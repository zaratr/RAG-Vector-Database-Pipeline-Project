"""Tests for normalized persisted graph extraction provenance."""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation


def _session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def _document_chunk(session, *, index=0, text="Alice works at Acme Corp."):
    document = models.Document(title="Graph source", source="unit")
    session.add(document)
    session.flush()
    chunk = models.Chunk(
        document_id=document.id,
        index=index,
        text=text,
        start_offset=0,
        end_offset=len(text),
        vector_id=f"chunk:{document.id}:{index}",
    )
    session.add(chunk)
    session.flush()
    return document, chunk


def _relation(confidence=0.9, source_type="person"):
    return ExtractedRelation(
        source=ExtractedEntity(
            name="Alice", canonical_name="alice", entity_type=source_type
        ),
        predicate="works_at",
        target=ExtractedEntity(
            name="Acme Corp",
            canonical_name="acme corp",
            entity_type="organization",
        ),
        evidence="Alice works at Acme Corp.",
        evidence_start=0,
        evidence_end=25,
        confidence=confidence,
    )


def test_persist_extraction_records_run_mentions_edge_and_evidence_provenance():
    session, engine = _session()
    document, chunk = _document_chunk(session)

    extraction = persist_chunk_extraction(
        session,
        chunk=chunk,
        relations=[_relation()],
        provider="ollama",
        model="gemma4:latest",
    )
    session.commit()

    assert extraction.status == "succeeded"
    assert extraction.prompt_version == "graph-v1"
    assert extraction.schema_version == "graph-relations-v1"
    assert session.query(models.GraphEntity).count() == 2
    mentions = session.query(models.EntityMention).all()
    assert {(mention.surface_form, mention.start_offset) for mention in mentions} == {
        ("Alice", 0),
        ("Acme Corp", 15),
    }
    edge = session.query(models.GraphEdge).one()
    evidence = session.query(models.GraphEdgeEvidence).one()
    assert edge.predicate == "works_at"
    assert evidence.edge_id == edge.id
    assert evidence.extraction.chunk.document.id == document.id
    assert evidence.extraction.chunk_id == chunk.id
    assert evidence.evidence_text == chunk.text
    assert evidence.confidence == 0.9
    session.close()
    engine.dispose()


def test_same_logical_edge_from_two_chunks_has_two_evidence_rows():
    session, engine = _session()
    document, first = _document_chunk(session)
    second = models.Chunk(
        document_id=document.id,
        index=1,
        text=first.text,
        start_offset=26,
        end_offset=51,
        vector_id=f"chunk:{document.id}:1",
    )
    session.add(second)
    session.flush()

    for chunk in (first, second):
        persist_chunk_extraction(
            session,
            chunk=chunk,
            relations=[_relation()],
            provider="ollama",
            model="gemma4:latest",
        )
    session.commit()

    assert session.query(models.GraphEdge).count() == 1
    assert session.query(models.GraphEdgeEvidence).count() == 2
    assert session.query(models.GraphExtraction).count() == 2
    assert {
        evidence.extraction.chunk_id
        for evidence in session.query(models.GraphEdgeEvidence).all()
    } == {first.id, second.id}
    session.close()
    engine.dispose()


def test_same_canonical_name_with_different_type_creates_distinct_entities():
    session, engine = _session()
    document, chunk = _document_chunk(session)
    relations = [_relation(source_type="person"), _relation(source_type="account")]

    persist_chunk_extraction(
        session,
        chunk=chunk,
        relations=relations,
        provider="ollama",
        model="gemma4:latest",
    )
    session.commit()

    alice_entities = (
        session.query(models.GraphEntity).filter_by(canonical_name="alice").all()
    )
    assert {entity.entity_type for entity in alice_entities} == {"person", "account"}
    session.close()
    engine.dispose()


def test_empty_extraction_is_recorded_as_successful_empty_run():
    session, engine = _session()
    document, chunk = _document_chunk(session)
    extraction = persist_chunk_extraction(
        session,
        chunk=chunk,
        relations=[],
        provider="ollama",
        model="gemma4:latest",
    )
    session.commit()

    assert extraction.status == "empty"
    assert session.query(models.GraphEdgeEvidence).count() == 0
    session.close()
    engine.dispose()


def test_document_delete_cascades_provenance_but_not_logical_graph():
    session, engine = _session()
    document, chunk = _document_chunk(session)
    persist_chunk_extraction(
        session,
        chunk=chunk,
        relations=[_relation()],
        provider="ollama",
        model="gemma4:latest",
    )
    session.commit()

    session.delete(document)
    session.commit()

    assert session.query(models.GraphExtraction).count() == 0
    assert session.query(models.EntityMention).count() == 0
    assert session.query(models.GraphEdgeEvidence).count() == 0
    assert session.query(models.GraphEdge).count() == 1
    assert session.query(models.GraphEntity).count() == 2
    session.close()
    engine.dispose()
