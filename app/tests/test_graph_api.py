"""Graph inspection API tests.

Covers the graph inspection surface plus regression tests:

- OpenAPI must document the canonical ``GraphPath``/``GraphPathStep``
  step fields for ``POST /graph/paths`` (impossible while ``paths`` was
  typed ``List[dict]``).
- Relationship items sort by canonical names, not entity IDs.
- ``POST /graph/paths`` ``limit`` outside 1–50 returns 422.
- ``document_id`` filters reject non-integer forms ("7", 7.0, booleans).
- Prefix/contains matching escapes LIKE wildcards so literal ``%`` and
  ``_`` in entity names match literally.
"""
from __future__ import annotations

import chromadb
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_query
from app.core.db import Base, get_db
from app.main import app
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation
from app.services.llm import DummyLLMClient
from app.services.vector_store import ChromaVectorStore


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


def _seed_triples(title, tags, triples):
    """Seed one ready document with one chunk per (source, predicate, target).

    Returns the persisted document ID (the ORM instance itself would be
    detached after the session closes).
    """
    session = TestSessionLocal()
    document = models.Document(title=title, source="unit", tags=tags)
    session.add(document)
    session.flush()
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
    document_id = document.id
    session.close()
    return document_id


test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(test_engine, "connect")
def _enable_foreign_keys(dbapi_connection, connection_record):
    dbapi_connection.execute("PRAGMA foreign_keys=ON")


TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def _override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def graph_api_environment():
    """Bind the app to this module's engine and make /query hermetic.

    ``/query`` is pinned unchanged; its provider collaborators are swapped
    for ephemeral/dummy implementations exactly like test_rag_api.py so the
    backward-compatibility test proves mode acceptance without live
    Ollama/Chroma dependencies. The prior get_db override (other test
    modules install their own at import time) is restored afterwards.
    """
    ephemeral = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(collection_name="test-graph-api", client=ephemeral)
    original_store = routes_query.get_vector_store
    original_llm = routes_query.get_llm_client
    routes_query.get_vector_store = lambda: vector_store
    routes_query.get_llm_client = lambda *args, **kwargs: DummyLLMClient()
    prior_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _override_get_db
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    routes_query.get_vector_store = original_store
    routes_query.get_llm_client = original_llm
    ephemeral.delete_collection("test-graph-api")
    if prior_override is not None:
        app.dependency_overrides[get_db] = prior_override
    else:
        app.dependency_overrides.pop(get_db, None)
    test_engine.dispose()


@pytest.fixture(scope="module")
def seeded_graph():
    """Seed the module-scoped test_engine with the 3-hop chain and yield
    the document + entity ids needed by the API tests."""
    document_id = _seed_triples(
        "Access chain",
        "graph",
        [
            ("User", "purchases", "Subscription", "User purchases Subscription."),
            ("Subscription", "grants", "PremiumAccess", "Subscription grants PremiumAccess."),
            ("PremiumAccess", "unlocks", "Dashboard", "PremiumAccess unlocks Dashboard."),
        ],
    )
    session = TestSessionLocal()
    user_entity = session.query(models.GraphEntity).filter_by(canonical_name="user").one()
    session.close()
    yield document_id, user_entity.id
    # cleanup
    s = TestSessionLocal()
    s.query(models.Document).filter_by(id=document_id).delete()
    s.commit()
    s.close()


# ---------------------------------------------------------------------------
# GET /graph/entities
# ---------------------------------------------------------------------------


def test_get_graph_entities_returns_ready_only_with_counts(seeded_graph):
    document, user_entity_id = seeded_graph
    response = client.get("/graph/entities")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert body["limit"] == 20
    assert body["offset"] == 0
    item = next(i for i in body["items"] if i["id"] == user_entity_id)
    assert item["canonical_name"] == "user"
    assert item["entity_type"] == "concept"
    assert item["mention_count"] >= 1
    assert item["evidence_count"] >= 1


def test_get_graph_entities_pagination_respects_limit_and_offset(seeded_graph):
    response = client.get("/graph/entities", params={"limit": 2, "offset": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 2
    assert len(body["items"]) <= 2
    assert body["offset"] == 0


@pytest.mark.parametrize("limit", [0, 101])
def test_get_graph_entities_rejects_out_of_range_limit(limit, seeded_graph):
    response = client.get("/graph/entities", params={"limit": limit})
    assert response.status_code == 422


def test_get_graph_entities_rejects_negative_offset(seeded_graph):
    response = client.get("/graph/entities", params={"offset": -1})
    assert response.status_code == 422


def test_get_graph_entities_name_filter_exact_match(seeded_graph):
    document, user_entity_id = seeded_graph
    response = client.get("/graph/entities", params={"name": "user", "match": "exact"})
    assert response.status_code == 200
    body = response.json()
    assert any(i["id"] == user_entity_id for i in body["items"])
    assert all(i["canonical_name"] == "user" for i in body["items"])


def test_get_graph_entities_prefix_match(seeded_graph):
    response = client.get("/graph/entities", params={"name": "use", "match": "prefix"})
    assert response.status_code == 200
    body = response.json()
    assert any(i["canonical_name"].startswith("use") for i in body["items"])


def test_get_graph_entities_contains_match(seeded_graph):
    response = client.get("/graph/entities", params={"name": "ser", "match": "contains"})
    assert response.status_code == 200
    body = response.json()
    assert any("ser" in i["canonical_name"] for i in body["items"])


def test_get_graph_entities_excludes_orphans_without_ready_evidence():
    """An entity with no mention/edge evidence in a ready document is
    not inspection-visible."""
    session = TestSessionLocal()
    orphan = models.GraphEntity(canonical_name="orphan", display_name="Orphan",
                                entity_type="concept")
    session.add(orphan)
    session.commit()
    session.close()
    response = client.get("/graph/entities", params={"name": "orphan"})
    assert response.status_code == 200
    assert response.json()["total"] == 0


# ---------------------------------------------------------------------------
# GET /graph/entities/{entity_id}/relationships
# ---------------------------------------------------------------------------


def test_get_relationships_outbound_returns_edges_with_full_evidence(seeded_graph):
    document, user_entity_id = seeded_graph
    response = client.get(f"/graph/entities/{user_entity_id}/relationships",
                          params={"direction": "outbound"})
    assert response.status_code == 200
    body = response.json()
    assert body["entity_id"] == user_entity_id
    assert body["direction"] == "outbound"
    assert body["total"] >= 1
    item = body["items"][0]
    assert item["source"]["id"] == user_entity_id
    assert item["predicate"] == "purchases"
    assert item["evidence"]
    ev = item["evidence"][0]
    for key in ("evidence_id", "document_id", "chunk_id", "extraction_id",
                "text", "start", "end", "confidence", "model"):
        assert key in ev
    assert ev["text"] == "User purchases Subscription."


def test_get_relationships_inbound_for_target_entity(seeded_graph):
    document, user_entity_id = seeded_graph
    session = TestSessionLocal()
    subscription = session.query(models.GraphEntity).filter_by(canonical_name="subscription").one()
    session.close()
    response = client.get(f"/graph/entities/{subscription.id}/relationships",
                          params={"direction": "inbound"})
    assert response.status_code == 200
    body = response.json()
    # Subscription is target of "purchases"
    assert any(i["predicate"] == "purchases" for i in body["items"])


def test_get_relationships_unknown_entity_returns_404(seeded_graph):
    response = client.get("/graph/entities/999999/relationships")
    assert response.status_code == 404


def test_get_relationships_valid_entity_with_no_ready_evidence_returns_200_empty():
    """An entity that exists but has no ready-document evidence returns
    200 and an empty items list."""
    session = TestSessionLocal()
    orphan = models.GraphEntity(canonical_name="lonely", display_name="Lonely",
                                entity_type="concept")
    session.add(orphan)
    session.commit()
    orphan_id = orphan.id
    session.close()
    response = client.get(f"/graph/entities/{orphan_id}/relationships")
    assert response.status_code == 200
    assert response.json()["items"] == []


# ---------------------------------------------------------------------------
# POST /graph/paths
# ---------------------------------------------------------------------------


def test_post_graph_paths_returns_complete_graphpathstep_objects(seeded_graph):
    document, user_entity_id = seeded_graph
    response = client.post("/graph/paths", json={
        "query": "Explain User", "max_hops": 2,
        "direction": "outbound", "limit": 10, "filters": None,
    })
    assert response.status_code == 200
    body = response.json()
    assert "paths" in body
    assert body["paths"]
    first = body["paths"][0]
    for key in ("seed_entity_id", "terminal_entity_id", "hop_count", "steps", "score"):
        assert key in first
    step = first["steps"][0]
    # exact GraphPathStep field set
    for key in ("edge_id", "evidence_id", "source_entity_id", "source",
                "source_type", "predicate", "target_entity_id", "target",
                "target_type", "chunk_id", "document_id", "evidence",
                "confidence", "extraction_id", "extraction_model"):
        assert key in step
    # The 3-hop chain must expose a 2-hop path whose first step is complete.
    two_step = next((p for p in body["paths"] if p["hop_count"] == 2), None)
    assert two_step is not None
    for key in ("edge_id", "evidence_id", "source_entity_id", "source",
                "source_type", "predicate", "target_entity_id", "target",
                "target_type", "chunk_id", "document_id", "evidence",
                "confidence", "extraction_id", "extraction_model"):
        assert key in two_step["steps"][0]


def test_post_graph_paths_validation_error_returns_422(seeded_graph):
    # max_hops out of range
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 5, "direction": "outbound",
        "limit": 10, "filters": None,
    })
    assert response.status_code == 422
    # invalid direction
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 2, "direction": "sideways",
        "limit": 10, "filters": None,
    })
    assert response.status_code == 422


def test_post_graph_paths_no_seed_returns_200_empty_paths(seeded_graph):
    response = client.post("/graph/paths", json={
        "query": "zzz no lexical match zzz", "max_hops": 2,
        "direction": "outbound", "limit": 10, "filters": None,
    })
    assert response.status_code == 200
    assert response.json()["paths"] == []


def test_post_graph_paths_traversal_limit_returns_503(monkeypatch, seeded_graph):
    from app.api import routes_graph
    from app.services.graph_retrieval import GraphTraversalLimitError

    def _boom(*args, **kwargs):
        raise GraphTraversalLimitError("cap exceeded")
    monkeypatch.setattr(routes_graph, "retrieve_graph_paths", _boom)
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 2, "direction": "outbound",
        "limit": 10, "filters": None,
    })
    assert response.status_code == 503


def test_post_graph_paths_unsupported_filter_returns_422(seeded_graph):
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 2, "direction": "outbound",
        "limit": 10, "filters": {"unknown_key": 1},
    })
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /query pin + inspection-route purity + OpenAPI
# ---------------------------------------------------------------------------


def test_post_query_remains_backward_compatible_with_vector_graph_hybrid(seeded_graph):
    """Existing POST /query still accepts vector/graph/hybrid modes."""
    for mode in ("vector", "graph", "hybrid"):
        response = client.post("/query", json={
            "query": "FastAPI", "retrieval_mode": mode, "top_k": 1,
        })
        assert response.status_code == 200


def test_graph_inspection_routes_do_not_call_chroma_or_embedding(monkeypatch):
    """GET /graph/entities and /graph/entities/{id}/relationships must
    not invoke the embedding provider or vector store."""
    from app.api import routes_graph
    calls = []
    class _Spy:
        def embed_texts(self, t): calls.append(("embed", t)); return [[0.0]]
        async def query(self, *a, **k): calls.append(("query", a)); return []
    monkeypatch.setattr(routes_graph, "get_vector_store", lambda: _Spy())
    monkeypatch.setattr(routes_graph, "get_embedding_provider", lambda: _Spy())
    client.get("/graph/entities")
    client.get("/graph/entities/1/relationships")
    assert calls == []


def test_graph_paths_openapi_schema_documents_all_fields():
    """OpenAPI must include the /graph/paths endpoint and its schema."""
    schema = client.get("/openapi.json").json()
    assert "/graph/paths" in schema["paths"]
    assert "post" in schema["paths"]["/graph/paths"]
    # The canonical GraphPath/GraphPathStep models must document every
    # step field (impossible while paths was typed List[dict]).
    components = schema["components"]["schemas"]
    assert "GraphPath" in components
    for key in ("seed_entity_id", "terminal_entity_id", "hop_count", "steps", "score"):
        assert key in components["GraphPath"]["properties"]
    assert "GraphPathStep" in components
    step_properties = components["GraphPathStep"]["properties"]
    for key in ("edge_id", "evidence_id", "source_entity_id", "source",
                "source_type", "predicate", "target_entity_id", "target",
                "target_type", "chunk_id", "document_id", "evidence",
                "confidence", "extraction_id", "extraction_model"):
        assert key in step_properties


# ---------------------------------------------------------------------------
# POST /graph/paths limit bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [0, 51])
def test_post_graph_paths_rejects_out_of_range_limit(limit, seeded_graph):
    """Returned paths are capped at 50; limit outside 1-50 is invalid
    input and must map to 422, never a 200."""
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 2, "direction": "outbound",
        "limit": limit, "filters": None,
    })
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Relationship items sort by canonical names
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reverse_order_targets(seeded_graph):
    """Seed User edges whose targets are created in reverse canonical-name
    order so entity-ID order differs from canonical-name order."""
    return _seed_triples(
        "Sort order",
        "graph-sort",
        [
            ("User", "manages", "ZuluTarget", "User manages ZuluTarget."),
            ("User", "manages", "AlphaTarget", "User manages AlphaTarget."),
        ],
    )


def test_get_relationships_items_sort_by_canonical_names(seeded_graph, reverse_order_targets):
    """Items sort by source canonical name, predicate,
    target canonical name, edge ID — not by entity IDs."""
    document, user_entity_id = seeded_graph
    response = client.get(f"/graph/entities/{user_entity_id}/relationships",
                          params={"direction": "outbound"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 3
    items = body["items"]
    # The alphabetically-later ZuluTarget was created first (lower entity ID);
    # canonical-name ordering must place AlphaTarget before it.
    assert items[0]["target"]["name"] == "AlphaTarget"
    keys = [
        (i["source"]["name"].casefold(), i["predicate"], i["target"]["name"].casefold(), i["edge_id"])
        for i in items
    ]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Typed document_id filter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["7", 7.0, True])
def test_post_graph_paths_rejects_non_integer_document_id_filter(bad_value, seeded_graph):
    """Scalar filter matrix: document_id is integer equality;
    strings, floats, and booleans are invalid integer forms -> 422."""
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 2, "direction": "outbound",
        "limit": 10, "filters": {"document_id": bad_value},
    })
    assert response.status_code == 422


def test_post_graph_paths_accepts_integer_document_id_filter(seeded_graph):
    """Integer document_id equality keeps working (filter parity)."""
    document_id, user_entity_id = seeded_graph
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 2, "direction": "outbound",
        "limit": 10, "filters": {"document_id": document_id},
    })
    assert response.status_code == 200
    assert response.json()["paths"]
    response = client.post("/graph/paths", json={
        "query": "User", "max_hops": 2, "direction": "outbound",
        "limit": 10, "filters": {"document_id": 999999},
    })
    assert response.status_code == 200
    assert response.json()["paths"] == []


# ---------------------------------------------------------------------------
# LIKE wildcard escaping in prefix/contains matching
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wildcard_entities(seeded_graph):
    """Seed entity names containing literal LIKE wildcards ('_', '%') plus
    near-miss siblings that unescaped wildcards would wrongly match."""
    return _seed_triples(
        "Wildcard names",
        "graph-wildcards",
        [
            ("User_Alpha", "knows", "UserBeta", "User_Alpha knows UserBeta."),
            ("Metric_100%", "measures", "Metric_100x", "Metric_100% measures Metric_100x."),
        ],
    )


def test_get_graph_entities_prefix_match_escapes_like_wildcards(seeded_graph, wildcard_entities):
    """'user_' as a prefix must match only the literal 'user_alpha', not
    'userbeta' (an unescaped '_' matches any single character)."""
    response = client.get("/graph/entities", params={"name": "user_", "match": "prefix"})
    assert response.status_code == 200
    body = response.json()
    assert [i["canonical_name"] for i in body["items"]] == ["user_alpha"]


def test_get_graph_entities_contains_match_escapes_like_wildcards(seeded_graph, wildcard_entities):
    """'100%' as a contains needle must match only the literal
    'metric_100%', not 'metric_100x' (an unescaped '%' matches anything)."""
    response = client.get("/graph/entities", params={"name": "100%", "match": "contains"})
    assert response.status_code == 200
    body = response.json()
    assert [i["canonical_name"] for i in body["items"]] == ["metric_100%"]
