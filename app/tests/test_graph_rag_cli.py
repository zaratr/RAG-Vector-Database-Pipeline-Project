"""Phase 10A.5 — CLI tests for src/graph_rag.py (retrieve_graph_paths migration)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI = PROJECT_ROOT / "src" / "graph_rag.py"


def _entity(name):
    return ExtractedEntity(name=name, canonical_name=name.casefold(), entity_type="concept")


def _relation(source, predicate, target, text):
    return ExtractedRelation(
        source=_entity(source), predicate=predicate, target=_entity(target),
        evidence=text, evidence_start=0, evidence_end=len(text), confidence=0.9,
    )


def _seed_db(db_path: Path) -> tuple[int, int]:
    """Seed an on-disk SQLite DB with the canonical 3-hop chain."""
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="Access chain", source="unit", tags="graph")
    session.add(document); session.flush()
    triples = [
        ("User", "purchases", "Subscription", "User purchases Subscription."),
        ("Subscription", "grants", "PremiumAccess", "Subscription grants PremiumAccess."),
        ("PremiumAccess", "unlocks", "Dashboard", "PremiumAccess unlocks Dashboard."),
    ]
    chunks = []
    for index, (source, predicate, target, text) in enumerate(triples):
        chunk = models.Chunk(document_id=document.id, index=index, text=text,
                             start_offset=0, end_offset=len(text),
                             vector_id=f"chunk:{document.id}:{index}")
        session.add(chunk); session.flush()
        persist_chunk_extraction(session, chunk=chunk,
                                 relations=[_relation(source, predicate, target, text)],
                                 provider="ollama", model="gemma4:latest")
        chunks.append(chunk)
    session.commit()
    doc_id = document.id
    chunk_id = chunks[0].id
    session.close(); engine.dispose()
    return doc_id, chunk_id


def test_graph_rag_cli_emits_graphpathstep_fields_for_outbound_traversal(tmp_path):
    db_path = tmp_path / "graph-rag-cli.db"
    _seed_db(db_path)
    env = {**os.environ, "RAG_DATABASE_URL": f"sqlite:///{db_path}"}
    argv = [sys.executable, str(CLI), "Explain User",
            "--hops", "2", "--direction", "outbound", "--limit", "10"]
    result = subprocess.run(argv, env=env, capture_output=True, text=True,
                            cwd=PROJECT_ROOT, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert payload, "expected at least one path"
    first_path = payload[0]
    for key in ("seed_entity_id", "terminal_entity_id", "hop_count", "steps", "score"):
        assert key in first_path
    first_step = first_path["steps"][0]
    for key in ("edge_id", "evidence_id", "source_entity_id", "source",
                "source_type", "predicate", "target_entity_id", "target",
                "target_type", "chunk_id", "document_id", "evidence",
                "confidence", "extraction_id", "extraction_model"):
        assert key in first_step, f"missing GraphPathStep field {key}"
    assert first_step["predicate"] == "purchases"


def test_graph_rag_cli_caps_hops_at_three():
    """--hops choices=range(1,4) means 4 must be rejected by argparse
    with exit code 2 and a usage error."""
    argv = [sys.executable, str(CLI), "User", "--hops", "4"]
    result = subprocess.run(argv, capture_output=True, text=True,
                            cwd=PROJECT_ROOT, check=False)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


@pytest.mark.parametrize("hops", ["1", "2", "3"])
def test_graph_rag_cli_accepts_hops_one_through_three(hops, tmp_path):
    db_path = tmp_path / "graph-rag-cli-hops.db"
    _seed_db(db_path)
    env = {**os.environ, "RAG_DATABASE_URL": f"sqlite:///{db_path}"}
    argv = [sys.executable, str(CLI), "Explain User", "--hops", hops]
    result = subprocess.run(argv, env=env, capture_output=True, text=True,
                            cwd=PROJECT_ROOT, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is not None


def test_graph_rag_cli_direction_inbound_returns_paths(tmp_path):
    db_path = tmp_path / "graph-rag-cli-inbound.db"
    _seed_db(db_path)
    env = {**os.environ, "RAG_DATABASE_URL": f"sqlite:///{db_path}"}
    argv = [sys.executable, str(CLI), "Explain Dashboard",
            "--hops", "3", "--direction", "inbound"]
    result = subprocess.run(argv, env=env, capture_output=True, text=True,
                            cwd=PROJECT_ROOT, check=False)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    assert paths
    assert any(s["predicate"] == "unlocks" for p in paths for s in p["steps"])


def test_graph_rag_cli_does_not_emit_legacy_context_objects(tmp_path):
    """Output must contain GraphPathStep fields, NOT the legacy
    context dict shape (text/score/metadata.graph.hop)."""
    db_path = tmp_path / "graph-rag-cli-shape.db"
    _seed_db(db_path)
    env = {**os.environ, "RAG_DATABASE_URL": f"sqlite:///{db_path}"}
    argv = [sys.executable, str(CLI), "Explain User", "--hops", "1"]
    result = subprocess.run(argv, env=env, capture_output=True, text=True,
                            cwd=PROJECT_ROOT, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    # legacy shape would be a list of dicts with top-level "metadata"
    assert "metadata" not in payload[0]
    assert "graph" not in payload[0]
