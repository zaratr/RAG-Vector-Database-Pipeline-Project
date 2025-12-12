"""Document ingestion pipeline."""
from __future__ import annotations

import uuid
from typing import Iterable, List, Optional, Sequence

from app.core.logging import logger
from app.persistence import models, repositories
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore


async def ingest_text(
    *,
    title: str,
    source: Optional[str],
    tags: Optional[Sequence[str]],
    text: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    session,
) -> dict:
    """Ingest raw text into the system."""

    normalized = " ".join(text.split())
    logger.info("Chunking document '%s'", title)
    chunks = chunk_text(normalized)
    document = repositories.create_document(session, title=title, source=source, tags=tags)
    chunk_models: List[models.Chunk] = repositories.create_chunks(session, document=document, chunks=chunks)

    embeddings = await embedding_provider.embed_texts([chunk.text for chunk in chunk_models])
    metadata_entries = [chunk.metadata() for chunk in chunk_models]
    ids = [str(uuid.uuid4()) for _ in chunk_models]
    await vector_store.index_embeddings(embeddings, metadata_entries, ids)

    logger.info("Ingested document %s with %s chunks", document.id, len(chunk_models))
    return {"document_id": document.id, "chunks": len(chunk_models)}


def chunks_for_document(chunks: Iterable[models.Chunk]) -> List[dict]:
    return [
        {
            "id": chunk.id,
            "index": chunk.index,
            "text": chunk.text,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
        }
        for chunk in chunks
    ]
