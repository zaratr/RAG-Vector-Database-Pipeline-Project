"""Unit tests for ChromaVectorStore client-mode selection and roundtrip."""
import uuid

import pytest
import chromadb

from app.services.vector_store import ChromaVectorStore, _create_client


def _ephemeral_client():
    """Factory for a fresh isolated in-memory Chroma client."""
    return chromadb.EphemeralClient()


@pytest.fixture()
def store():
    """Fresh store with a unique collection name on an ephemeral client."""
    client = _ephemeral_client()
    name = "test-vs-" + uuid.uuid4().hex[:8]
    s = ChromaVectorStore(collection_name=name, client=client)
    yield s
    try:
        client.delete_collection(name)
    except Exception:
        pass


# ── Client selection tests ──────────────────────────────────────────

def _client_backend_name(client):
    """Return the underlying server/backend type name for a Chroma client."""
    return type(getattr(client, "_server", None)).__name__


def test_ephemeral_mode_when_no_host_no_persist(monkeypatch):
    """With no chroma_host and no chroma_persist_directory, client is EphemeralClient."""
    from app.config import Settings
    monkeypatch.setattr("app.services.vector_store.get_settings", lambda: Settings(
        chroma_host=None, chroma_persist_directory=None
    ))
    client = _create_client()
    assert _client_backend_name(client) == "RustBindingsAPI"


def test_http_mode_when_host_set(monkeypatch):
    """When chroma_host is set, _create_client builds an HttpClient bound to
    that host/port. chromadb.HttpClient eagerly connects, so a recording stub
    is installed in its place: the construction contract (client type + bound
    host/port) is pinned without a live Chroma server."""
    from app.config import Settings
    import app.services.vector_store as vector_store_module

    created = {}

    class RecordingHttpClient:
        def __init__(self, host, port):
            created["host"] = host
            created["port"] = port

    monkeypatch.setattr(
        vector_store_module.chromadb, "HttpClient", RecordingHttpClient
    )
    monkeypatch.setattr("app.services.vector_store.get_settings", lambda: Settings(
        chroma_host="vectordb.example", chroma_port=8000
    ))
    client = _create_client()
    assert isinstance(client, RecordingHttpClient)
    assert created == {"host": "vectordb.example", "port": 8000}


def test_persistent_mode_when_directory_set(monkeypatch, tmp_path):
    """When chroma_persist_directory is set (and no host), client is PersistentClient."""
    from app.config import Settings
    monkeypatch.setattr("app.services.vector_store.get_settings", lambda: Settings(
        chroma_host=None, chroma_persist_directory=str(tmp_path)
    ))
    client = _create_client()
    assert _client_backend_name(client) == "RustBindingsAPI"
    # Persistent and ephemeral share the same Rust backend — distinguish by
    # checking that the client's identifier is NOT "ephemeral"
    assert client._identifier != "ephemeral"


@pytest.mark.asyncio
async def test_persistent_client_survives_across_client_instances(tmp_path):
    """Persistent mode is genuinely persistent: records written through one
    client instance are visible to a fresh client opened on the same
    directory (and absent from an unrelated directory)."""
    import chromadb

    name = "persist-" + uuid.uuid4().hex[:8]
    writer = ChromaVectorStore(
        collection_name=name, client=chromadb.PersistentClient(path=str(tmp_path))
    )
    await writer.index_embeddings(
        embeddings=[[0.1] * 8],
        metadatas=[{"doc": "persisted"}],
        ids=["p-1"],
        documents=["persisted payload"],
    )

    reader = ChromaVectorStore(
        collection_name=name, client=chromadb.PersistentClient(path=str(tmp_path))
    )
    hits = await reader.query([0.1] * 8, top_k=1)
    assert len(hits) == 1
    assert hits[0].vector_id == "p-1"
    assert hits[0].text == "persisted payload"
    assert hits[0].metadata["doc"] == "persisted"

    unrelated_dir = tmp_path / "elsewhere"
    unrelated_dir.mkdir()
    stranger = ChromaVectorStore(
        collection_name=name, client=chromadb.PersistentClient(path=str(unrelated_dir))
    )
    assert len(await stranger.query([0.1] * 8, top_k=1)) == 0


# ── Roundtrip tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_index_and_query_roundtrip(store):
    """Index a known embedding + document text, query it back."""
    embedding = [0.1] * 8
    await store.index_embeddings(
        embeddings=[embedding],
        metadatas=[{"doc": "test"}],
        ids=["rt-1"],
        documents=["hello world"],
    )
    results = await store.query(embedding, top_k=1)
    assert len(results) == 1
    assert results[0].text == "hello world"
    assert results[0].score == 0.0
    assert results[0].vector_id == "rt-1"


@pytest.mark.asyncio
async def test_query_empty_collection_returns_empty(store):
    """Querying a collection with no data returns an empty list, not an error."""
    embedding = [0.0] * 8
    results = await store.query(embedding, top_k=5)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_upsert_is_idempotent_for_deterministic_chunk_id(store):
    await store.upsert_embeddings(
        embeddings=[[0.1] * 8],
        metadatas=[{"chunk_id": 7}],
        ids=["chunk:7"],
        documents=["first"],
    )
    await store.upsert_embeddings(
        embeddings=[[0.2] * 8],
        metadatas=[{"chunk_id": 7}],
        ids=["chunk:7"],
        documents=["updated"],
    )

    assert store.collection.count() == 1
    result = await store.query([0.2] * 8, top_k=1)
    assert result[0].text == "updated"


@pytest.mark.asyncio
async def test_collection_isolation():
    """Two different collection names on the same client don't cross-contaminate."""
    client = _ephemeral_client()
    name_a = "iso-a-" + uuid.uuid4().hex[:8]
    name_b = "iso-b-" + uuid.uuid4().hex[:8]
    store_a = ChromaVectorStore(collection_name=name_a, client=client)
    store_b = ChromaVectorStore(collection_name=name_b, client=client)

    await store_a.index_embeddings(
        embeddings=[[0.1] * 8],
        metadatas=[{"src": "a"}],
        ids=["a1"],
        documents=["doc from A"],
    )
    results_b = await store_b.query([0.1] * 8, top_k=5)
    assert len(results_b) == 0

    results_a = await store_a.query([0.1] * 8, top_k=1)
    assert len(results_a) == 1
    assert results_a[0].text == "doc from A"

    client.delete_collection(name_a)
    client.delete_collection(name_b)
