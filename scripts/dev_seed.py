"""Seed the API with a sample document."""
import asyncio

from app.services.embeddings import get_embedding_provider
from app.services.ingestion import ingest_text
from app.services.vector_store import get_vector_store
from app.core.db import session_scope


def main() -> None:
    with session_scope() as session:
        asyncio.run(
            ingest_text(
                title="Sample",
                source="seed",
                tags=["example"],
                text="Hello world. This is a sample document for seeding.",
                embedding_provider=get_embedding_provider(),
                vector_store=get_vector_store(),
                session=session,
            )
        )


if __name__ == "__main__":
    main()
