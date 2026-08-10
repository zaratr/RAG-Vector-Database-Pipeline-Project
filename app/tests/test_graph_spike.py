"""Subprocess coverage for the persisted GraphRAG operator CLI (10A.5)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from app.core.db import Base, create_database_engine
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_graph_rag_cli_help_executes():
    """Verify the CLI --help works and describes path traversal."""
    result = subprocess.run(
        [sys.executable, "src/graph_rag.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "persisted GraphRAG relationships" in result.stdout


def test_graph_rag_cli_output_contains_graph_path_step_fields(tmp_path):
    """Verify CLI calls retrieve_graph_paths and prints GraphPathStep objects.

    Fixture: two-document two-hop graph seeded via persist_chunk_extraction.
    - Document 1 / Chunk 0: "User purchases Subscription."
    - Document 2 / Chunk 1: "Subscription grants PremiumAccess."
    Query "User" --hops 2 returns at least one GraphPath whose first step has
    every canonical GraphPathStep field.
    """
    database_path = tmp_path / "graph-cli-spike.db"
    database_url = f"sqlite:///{database_path}"
    engine = create_database_engine(database_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="CLI graph spike", source="unit-test")
    session.add(document)
    session.flush()
    chunks = [
        models.Chunk(
            document_id=document.id,
            index=index,
            text=text,
            start_offset=0,
            end_offset=len(text),
        )
        for index, text in enumerate(
            ["User purchases Subscription.", "Subscription grants PremiumAccess."]
        )
    ]
    session.add_all(chunks)
    session.flush()
    for chunk, source, predicate, target in (
        (chunks[0], "User", "purchases", "Subscription"),
        (chunks[1], "Subscription", "grants", "PremiumAccess"),
    ):
        persist_chunk_extraction(
            session,
            chunk=chunk,
            relations=[
                ExtractedRelation(
                    source=ExtractedEntity(
                        name=source,
                        canonical_name=source.casefold(),
                        entity_type="concept",
                    ),
                    predicate=predicate,
                    target=ExtractedEntity(
                        name=target,
                        canonical_name=target.casefold(),
                        entity_type="concept",
                    ),
                    evidence=chunk.text,
                    evidence_start=0,
                    evidence_end=len(chunk.text),
                    confidence=1.0,
                )
            ],
            provider="ollama",
            model="gemma4:latest",
        )
    session.commit()
    session.close()
    engine.dispose()

    env = os.environ.copy()
    env["RAG_DATABASE_URL"] = database_url
    argv = [sys.executable, "src/graph_rag.py", "User", "--hops", "2"]
    result = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout)
    assert isinstance(paths, list)
    assert len(paths) >= 1, "Expected at least one path for two-hop query"
    path = paths[0]
    # GraphPath fields
    assert "seed_entity_id" in path
    assert "terminal_entity_id" in path
    assert "hop_count" in path
    assert "steps" in path
    assert "score" in path
    assert len(path["steps"]) >= 1
    step = path["steps"][0]
    # GraphPathStep canonical fields (must match 10A.5 schema exactly)
    for field in (
        "edge_id", "evidence_id",
        "source_entity_id", "source", "source_type",
        "predicate",
        "target_entity_id", "target", "target_type",
        "chunk_id", "document_id",
        "evidence", "confidence",
        "extraction_id", "extraction_model",
    ):
        assert field in step, f"Missing GraphPathStep field: {field}"


def test_graph_rag_cli_hops_capped_at_three():
    """Verify --hops accepts max 3 and rejects 4+.

    argparse with choices=range(1, 4) means accepted values are 1, 2, 3.
    Passing --hops 4 must exit with code 2 (argparse error).
    """
    result = subprocess.run(
        [sys.executable, "src/graph_rag.py", "test", "--hops", "4"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, f"Expected exit 2 for --hops 4, got {result.returncode}"


def test_networkx_not_imported_after_removal():
    """Verify networkx is not importable after removal from requirements.

    Runs inside the Docker image (not on host) after image rebuild.
    Asserts importlib.util.find_spec('networkx') returns None.
    """
    import importlib.util
    assert importlib.util.find_spec("networkx") is None, \
        "networkx must be absent from installed packages after 10A.5 removal"
