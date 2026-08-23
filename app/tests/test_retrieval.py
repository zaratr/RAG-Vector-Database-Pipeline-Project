import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.persistence import repositories
from app.services.embeddings import HashEmbeddingProvider as LocalEmbeddingProvider
from app.services.retrieval import retrieve, retrieve_contexts
from app.services.vector_store import ChromaVectorStore

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.mark.asyncio
async def test_retrieve_returns_results():
    provider = LocalEmbeddingProvider()
    store = ChromaVectorStore(collection_name="test-retrieval")

    session: Session = TestSessionLocal()
    document = repositories.create_document(session, title="Retrieve", source="test", tags=["demo"])
    chunk = repositories.create_chunks(
        session,
        document=document,
        chunks=[{"index": 0, "text": "this is retrieval", "start_offset": 0, "end_offset": 10}],
    )[0]
    session.commit()

    embeddings = await provider.embed_texts([chunk.text])
    await store.index_embeddings(embeddings, [chunk.get_chunk_metadata()], ["retrieval-chunk"], documents=[chunk.text])

    results = await retrieve(query="retrieval", embedding_provider=provider, vector_store=store, top_k=1)
    assert len(results) == 1
    assert "retrieval" in results[0]["text"]
    session.close()


@pytest.mark.asyncio
async def test_retrieve_contexts_tags_filter_uses_membership_not_raw_where(monkeypatch):
    """F4 vector-side parity: `tags` is membership-after-split of the stored
    CSV. Chroma's where clause cannot express membership over the indexed list
    tags metadata, so the filter must be applied against SQL Document.tags
    during hydration — never pushed as a raw where clause."""
    # The merged pipeline applies 10B's retrieval-security controls on top of
    # SQL-authoritative hydration. This test pins the tags-membership
    # hydration contract, so neutralize ONLY the distance control (the
    # deterministic hash embeddings produce l2 distances far above any
    # realistic threshold); caps/duplicate controls keep their shipped values.
    from app.services import retrieval as retrieval_module
    from app.services.retrieval_security import RetrievalSecurityPolicy

    monkeypatch.setattr(
        retrieval_module,
        "_POLICY_CACHE",
        RetrievalSecurityPolicy(
            version="retrieval-security-v1",
            metric="l2",
            max_distance=float("inf"),
            per_source_cap=2,
            per_document_cap=2,
            max_candidates=50,
            near_duplicate_jaccard=0.9,
            calibration_fixture_sha256="test-only",
            calibration_clean_recall=1.0,
            calibration_poison_share=0.0,
            calibration_tool_version="calibrate-v1",
            calibration_embedding_model="hash-test",
        ),
    )

    provider = LocalEmbeddingProvider()
    store = ChromaVectorStore(collection_name="test-retrieval-tags-parity")

    session: Session = TestSessionLocal()
    tagged = repositories.create_document(
        session, title="Tags match", source="test", tags=["alpha", "graph", "beta"]
    )
    other = repositories.create_document(
        session, title="Other tags", source="test", tags=["other"]
    )
    tagged_chunk = repositories.create_chunks(
        session,
        document=tagged,
        chunks=[{"index": 0, "text": "tagged membership text", "start_offset": 0, "end_offset": 22}],
    )[0]
    other_chunk = repositories.create_chunks(
        session,
        document=other,
        chunks=[{"index": 0, "text": "other tags text", "start_offset": 0, "end_offset": 15}],
    )[0]
    session.commit()

    tagged_chunk.vector_id = f"chunk:{tagged.id}:0"
    other_chunk.vector_id = f"chunk:{other.id}:0"
    session.commit()

    embeddings = await provider.embed_texts(
        [tagged_chunk.text, other_chunk.text]
    )
    await store.index_embeddings(
        embeddings,
        [tagged_chunk.get_chunk_metadata(), other_chunk.get_chunk_metadata()],
        [tagged_chunk.vector_id, other_chunk.vector_id],
        documents=[tagged_chunk.text, other_chunk.text],
    )

    try:
        contexts = await retrieve_contexts(
            query="tags", embedding_provider=provider, vector_store=store,
            session=session, mode="vector", top_k=10, filters={"tags": "graph"},
        )
        assert {c["metadata"]["chunk_id"] for c in contexts} == {tagged_chunk.id}

        contexts = await retrieve_contexts(
            query="tags", embedding_provider=provider, vector_store=store,
            session=session, mode="vector", top_k=10, filters={"tags": "absent"},
        )
        assert contexts == []
    finally:
        session.close()
