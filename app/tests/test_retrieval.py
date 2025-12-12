import pytest
from sqlalchemy.orm import Session

from app.core.db import Base, SessionLocal, engine
from app.persistence import repositories
from app.services.embeddings import LocalEmbeddingProvider
from app.services.retrieval import retrieve
from app.services.vector_store import ChromaVectorStore


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.mark.asyncio
async def test_retrieve_returns_results():
    provider = LocalEmbeddingProvider()
    store = ChromaVectorStore(collection_name="test-retrieval")

    session: Session = SessionLocal()
    document = repositories.create_document(session, title="Retrieve", source="test", tags=["demo"])
    chunk = repositories.create_chunks(
        session,
        document=document,
        chunks=[{"index": 0, "text": "this is retrieval", "start_offset": 0, "end_offset": 10}],
    )[0]
    session.commit()

    embeddings = await provider.embed_texts([chunk.text])
    await store.index_embeddings(embeddings, [chunk.metadata()], ["retrieval-chunk"])

    results = await retrieve(query="retrieval", embedding_provider=provider, vector_store=store, top_k=1)
    assert len(results) == 1
    assert "retrieval" in results[0]["text"]
    session.close()
