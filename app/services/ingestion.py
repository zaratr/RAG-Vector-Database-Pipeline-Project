"""Document ingestion pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from app.core.logging import logger
from app.persistence import graph_repository, models, repositories
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingProvider, ImageEmbeddingProvider
from app.services.graph_extraction import GraphExtractor
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
    graph_extractor: GraphExtractor | None = None,
    graph_extraction_provider: str = "ollama",
    graph_extraction_model: str = "unknown",
) -> dict:
    """Ingest raw text into the system."""

    normalized = " ".join(text.split())
    logger.info("Chunking document '%s'", title)
    chunk_payloads = chunk_text(normalized)
    chunk_texts = [chunk["text"] for chunk in chunk_payloads]

    # External work happens before any relational write. Provider failures leave
    # neither staged rows nor partially indexed vectors.
    embeddings = await embedding_provider.embed_texts(chunk_texts)
    extracted_by_chunk = []
    if graph_extractor is not None:
        for text_value in chunk_texts:
            extracted_by_chunk.append(await graph_extractor.extract(text_value))
    else:
        extracted_by_chunk = [[] for _ in chunk_texts]

    document_id: int | None = None
    vector_ids: list[str] = []
    try:
        document = repositories.create_document(
            session, title=title, source=source, tags=tags
        )
        document.ingestion_status = "staged"
        chunk_models: List[models.Chunk] = repositories.create_chunks(
            session, document=document, chunks=chunk_payloads
        )
        for chunk_model in chunk_models:
            chunk_model.vector_id = f"chunk:{chunk_model.id}"
        session.flush()
        document_id = document.id
        vector_ids = [chunk.vector_id for chunk in chunk_models]
        session.commit()

        metadata_entries = [chunk.get_chunk_metadata() for chunk in chunk_models]
        await vector_store.upsert_embeddings(
            embeddings,
            metadata_entries,
            vector_ids,
            documents=chunk_texts,
        )

        relation_count = 0
        for chunk_model, relations in zip(chunk_models, extracted_by_chunk):
            graph_repository.persist_chunk_extraction(
                session,
                chunk=chunk_model,
                relations=relations,
                provider=graph_extraction_provider,
                model=graph_extraction_model,
            )
            relation_count += len(relations)
        document.ingestion_status = "ready"
        document.failure_code = None
        session.commit()
    except Exception as exc:
        session.rollback()
        if document_id is not None:
            failed_document = session.get(models.Document, document_id)
            if failed_document is not None:
                failed_document.ingestion_status = "failed"
                failed_document.failure_code = type(exc).__name__[:100]
                session.commit()
        raise

    logger.info("Ingested document %s with %s chunks", document_id, len(chunk_models))
    return {
        "document_id": document_id,
        "chunks": len(chunk_models),
        "relations": relation_count,
    }


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

    document_id: int | None = None
    try:
        document = repositories.create_document(
            session, title=title, source=source, tags=tags
        )
        document.ingestion_status = "staged"
        chunk_text_value = f"[image:{media_type}] {path.name}"
        chunk = repositories.create_chunks(
            session,
            document=document,
            chunks=[
                {
                    "index": 0,
                    "text": chunk_text_value,
                    "start_offset": 0,
                    "end_offset": len(chunk_text_value),
                }
            ],
        )[0]
        chunk.vector_id = f"chunk:{chunk.id}"
        session.flush()
        document_id = document.id
        session.commit()

        metadata = chunk.get_chunk_metadata()
        metadata["media_type"] = media_type
        metadata["image_name"] = path.name
        await vector_store.upsert_embeddings(
            [embedding],
            [metadata],
            [chunk.vector_id],
            documents=[chunk_text_value],
        )
        document.ingestion_status = "ready"
        session.commit()
    except Exception as exc:
        session.rollback()
        if document_id is not None:
            failed_document = session.get(models.Document, document_id)
            if failed_document is not None:
                failed_document.ingestion_status = "failed"
                failed_document.failure_code = type(exc).__name__[:100]
                session.commit()
        raise

    logger.info("Ingested image document %s", document_id)
    return {"document_id": document_id, "chunks": 1}


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
