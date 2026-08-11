"""Phase 10A.7 — graph inspection API tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.db import Base, get_db
from app.main import app
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation


def _entity(name):
    return ExtractedEntity(name=name, canonical_name=name.casefold(), entity_type="concept")


def _relation(source, predicate, target, text):
    return ExtractedRelation(
        source=_entity(source),
        predicate=predicate,
        target=_entity(target),
        evidence=text,
        evidence_start=0,
        evidence_end=len(text),
        confidence=1.0,
    )


@pytest.fixture
def graph_client(tmp_path):
    """A TestClient backed by an isolated in-memory DB seeded with a 2-hop chain."""
    engine = create_engine(f"sqlite:///{tmp_path / 'graph_api.db'}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    document = models.Document(title="Access chain", source="unit", tags="graph")
    document.ingestion_status = "ready"
    session.add(document)
    session.flush()
    triples = [
        ("User", "purchases", "Subscription", "User purchases Subscription."),
        ("Subscription", "grants", "PremiumAccess", "Subscription grants PremiumAccess."),
    ]
    for index, (source, predicate, target, text) in enumerate(triples):
        chunk = models.Chunk(
            document_id=document.id,
            index=index,
            text=text,
            start_offset=0,
            end_offset=len(text),
            media_type="text/plain",
            vector_id=f"chunk:{document.id}:{index}",
        )
        session.add(chunk)
        session.flush()
        persist_chunk_extraction(
            session,
            chunk=chunk,
            relations=[_relation(source, predicate, target, text)],
            provider="ollama",
            model="gemma4:latest",
        )
    session.commit()
    session.close()

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    prior_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    try:
        yield client
    finally:
        # Restore the prior get_db override (other test modules install their own
        # at import time) rather than removing the key entirely.
        if prior_override is not None:
            app.dependency_overrides[get_db] = prior_override
        else:
            app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def test_list_entities_returns_ready_only_with_counts(graph_client):
    response = graph_client.get("/graph/entities")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 3  # User, Subscription, PremiumAccess
    names = {item["display_name"] for item in body["items"]}
    assert {"User", "Subscription", "PremiumAccess"}.issubset(names)
    for item in body["items"]:
        assert item["mention_count"] >= 1
        assert item["evidence_count"] >= 1


def test_list_entities_name_filter_exact(graph_client):
    response = graph_client.get("/graph/entities", params={"name": "User"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert all(i["canonical_name"] == "user" for i in items)
    assert any(i["display_name"] == "User" for i in items)


def test_relationships_unknown_entity_returns_404(graph_client):
    response = graph_client.get("/graph/entities/999999/relationships")
    assert response.status_code == 404


def test_relationships_outbound_returns_evidence_with_complete_fields(graph_client):
    entities = graph_client.get("/graph/entities", params={"name": "User"}).json()["items"]
    user_id = next(e["id"] for e in entities if e["display_name"] == "User")
    response = graph_client.get(f"/graph/entities/{user_id}/relationships")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    item = body["items"][0]
    assert item["predicate"] == "purchases"
    assert item["source"]["name"] == "User"
    assert item["target"]["name"] == "Subscription"
    ev = item["evidence"][0]
    for field in ("evidence_id", "document_id", "chunk_id", "extraction_id",
                  "text", "start", "end", "confidence", "model"):
        assert field in ev
    assert ev["text"] == "User purchases Subscription."


def test_post_paths_returns_complete_graph_path_step_objects(graph_client):
    response = graph_client.post(
        "/graph/paths",
        json={"query": "User", "max_hops": 2, "direction": "outbound", "limit": 10},
    )
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert len(paths) >= 1
    two_step = next((p for p in paths if p["hop_count"] == 2), None)
    assert two_step is not None
    step = two_step["steps"][0]
    for field in ("edge_id", "evidence_id", "source_entity_id", "source",
                  "source_type", "predicate", "target_entity_id", "target",
                  "target_type", "chunk_id", "document_id", "evidence",
                  "confidence", "extraction_id", "extraction_model"):
        assert field in step


def test_post_paths_no_seeds_returns_empty_paths(graph_client):
    response = graph_client.post(
        "/graph/paths",
        json={"query": "zzznoentzzz", "max_hops": 2, "direction": "outbound"},
    )
    assert response.status_code == 200
    assert response.json()["paths"] == []
