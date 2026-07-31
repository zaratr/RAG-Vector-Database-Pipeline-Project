"""Tests for NetworkX dependency and graph_rag.py spike execution.

Covers: dependency availability, happy-path pipeline execution, script
execution via subprocess, and edge cases the spike code already handles
correctly (empty graph, missing entity, 2nd-hop entity, empty doc list,
multi-doc accumulation).
"""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def restore_sys_path():
    """Ensure /app/src is removed from sys.path after each test to prevent leaks."""
    yield
    sys.path[:] = [p for p in sys.path if p != "/app/src"]


def _import_pipeline():
    """Import GraphRAGPipeline from src/graph_rag.py with path isolation."""
    if "/app/src" not in sys.path:
        sys.path.insert(0, "/app/src")
    from graph_rag import GraphRAGPipeline
    return GraphRAGPipeline


def test_networkx_importable():
    """NetworkX must be installed and importable inside the Docker container."""
    import networkx
    assert networkx.__version__ is not None


def test_graph_rag_pipeline_builds_and_queries():
    """The GraphRAGPipeline class must build a graph and produce expected query results."""
    Pipeline = _import_pipeline()
    pipeline = Pipeline()
    docs = ["User purchases Subscription which grants PremiumAccess."]
    pipeline.build_graph(docs)

    result = pipeline.query_graph("User")
    assert len(result) == 1
    assert result[0] == ("User", "Subscription", "purchases")


def test_graph_rag_script_executes():
    """The src/graph_rag.py script must execute via `python src/graph_rag.py` and produce expected stdout."""
    result = subprocess.run(
        [sys.executable, "src/graph_rag.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Graph query for 'User':" in result.stdout
    assert "('User', 'Subscription', 'purchases')" in result.stdout
    assert "GraphRAG local setup complete" in result.stdout


def test_query_empty_graph_returns_empty():
    """Querying a freshly initialized graph (no documents built) returns []."""
    Pipeline = _import_pipeline()
    pipeline = Pipeline()
    result = pipeline.query_graph("User")
    assert result == []


def test_query_missing_entity_returns_empty():
    """Querying for an entity that doesn't exist in the graph returns []."""
    Pipeline = _import_pipeline()
    pipeline = Pipeline()
    pipeline.build_graph(["User purchases Subscription which grants PremiumAccess."])
    result = pipeline.query_graph("NonExistentEntity")
    assert result == []


def test_query_2nd_hop_entity():
    """The intermediate entity (Subscription) must have its own outgoing edges."""
    Pipeline = _import_pipeline()
    pipeline = Pipeline()
    pipeline.build_graph(["User purchases Subscription which grants PremiumAccess."])

    result = pipeline.query_graph("Subscription")
    assert len(result) == 1
    assert result[0] == ("Subscription", "PremiumAccess", "grants")


def test_build_graph_with_empty_doc_list():
    """Building a graph with an empty document list produces an empty graph."""
    Pipeline = _import_pipeline()
    pipeline = Pipeline()
    pipeline.build_graph([])
    assert pipeline.graph.number_of_nodes() == 0
    assert pipeline.graph.number_of_edges() == 0


def test_multi_doc_accumulation_deduplicates_edges():
    """Building from multiple identical documents deduplicates nodes/edges in the DiGraph."""
    Pipeline = _import_pipeline()
    pipeline = Pipeline()
    docs = [
        "User purchases Subscription which grants PremiumAccess.",
        "User purchases Subscription which grants PremiumAccess.",
    ]
    pipeline.build_graph(docs)

    # DiGraph deduplicates: same 3 nodes, same 2 edges regardless of doc count
    assert pipeline.graph.number_of_nodes() == 3
    assert pipeline.graph.number_of_edges() == 2

    result = pipeline.query_graph("User")
    assert len(result) == 1  # No duplicate edges
