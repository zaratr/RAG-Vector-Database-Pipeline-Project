import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.persistence import repositories
from app.services.embeddings import HashEmbeddingProvider as LocalEmbeddingProvider
from app.services.retrieval import retrieve
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
