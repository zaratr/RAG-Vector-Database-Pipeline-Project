"""Document ingestion pipeline."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from app.core.logging import logger
from app.persistence import models, repositories
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingProvider, ImageEmbeddingProvider
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
    metadata_entries = [chunk.get_chunk_metadata() for chunk in chunk_models]
    ids = [str(uuid.uuid4()) for _ in chunk_models]
    texts = [chunk.text for chunk in chunk_models]
    await vector_store.index_embeddings(embeddings, metadata_entries, ids, documents=texts)

    logger.info("Ingested document %s with %s chunks", document.id, len(chunk_models))
    return {"document_id": document.id, "chunks": len(chunk_models)}


async def ingest_image(
    *,
    title: str,
    source: Optional[str],
    tags: Optional[Sequence[str]],
    image_path: str | Path,
    media_type: str,
    embedding_provider: ImageEmbeddingProvider,
    vector_store: VectorStore,
    session,
) -> dict:
    """Ingest an image into the vector store.

    The image is embedded and stored as a single vector entry.
    A placeholder chunk is created in the relational DB for tracking.
    """
    path = Path(image_path)
    logger.info("Embedding image '%s' (%s)", title, media_type)

    embeddings = await embedding_provider.embed_images([path])
    embedding = embeddings[0]

    # Create a document record for the image
    document = repositories.create_document(session, title=title, source=source, tags=tags)
    chunk_text = f"[image:{media_type}] {path.name}"
    chunk_models: List[models.Chunk] = repositories.create_chunks(
        session,
        document=document,
        chunks=[{"index": 0, "text": chunk_text, "start_offset": 0, "end_offset": 0}],
    )
    chunk = chunk_models[0]

    metadata = chunk.get_chunk_metadata()
    metadata["media_type"] = media_type
    metadata["image_name"] = path.name

    chunk_id = str(uuid.uuid4())
    await vector_store.index_embeddings(
        [embedding],
        [metadata],
        [chunk_id],
        documents=[chunk_text],
    )

    logger.info("Ingested image document %s", document.id)
    return {"document_id": document.id, "chunks": 1}


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
