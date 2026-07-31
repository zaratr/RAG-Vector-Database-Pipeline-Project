"""Subprocess coverage for the persisted GraphRAG operator CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from app.core.db import Base, create_database_engine
from app.persistence import models
from app.persistence.graph_repository import persist_chunk_extraction
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_networkx_importable():
    import networkx

    assert networkx.__version__ == "3.6.1"


def test_graph_rag_cli_help_executes():
    result = subprocess.run(
        [sys.executable, "src/graph_rag.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "persisted GraphRAG relationships" in result.stdout


def test_graph_rag_cli_queries_persisted_multihop_provenance(tmp_path):
    database_path = tmp_path / "graph-cli.db"
    database_url = f"sqlite:///{database_path}"
    engine = create_database_engine(database_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="CLI graph", source="unit")
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
    result = subprocess.run(
        [sys.executable, "src/graph_rag.py", "User", "--hops", "2"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    contexts = json.loads(result.stdout)
    assert [context["metadata"]["graph"]["predicate"] for context in contexts] == [
        "purchases",
        "grants",
    ]