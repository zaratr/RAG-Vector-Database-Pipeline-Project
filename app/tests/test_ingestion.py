import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.persistence import repositories
from app.services.embeddings import LocalEmbeddingProvider
from app.services.ingestion import ingest_text
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
async def test_ingest_text_creates_document():
    provider = LocalEmbeddingProvider()
    store = ChromaVectorStore(collection_name="test-ingestion")

    session: Session = TestSessionLocal()
    result = await ingest_text(
        title="Test Doc",
        source="unit",
        tags=["one"],
        text="hello world",
        embedding_provider=provider,
        vector_store=store,
        session=session,
    )
    assert result["chunks"] > 0
    docs = repositories.list_documents(session)
    assert len(docs) == 1
    session.close()
