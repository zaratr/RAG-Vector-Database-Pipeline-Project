"""Repair SQL/Chroma ingestion drift after retries or process crashes."""
from __future__ import annotations

from sqlalchemy.orm import Session, joinedload

from app.persistence import models
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore


async def reconcile_ingestion(
    *,
    session: Session,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    batch_size: int = 64,
) -> dict[str, int]:
    """Make vector visibility converge to relational document readiness."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    nonready = (
        session.query(models.Document)
        .options(joinedload(models.Document.chunks))
        .filter(models.Document.ingestion_status.in_(["staged", "failed"]))
        .order_by(models.Document.id)
        .all()
    )
    nonready_vector_ids = {
        chunk.vector_id
        for document in nonready
        for chunk in document.chunks
        if chunk.vector_id
    }

    staged_count = 0
    for document in nonready:
        if document.ingestion_status == "staged":
            document.ingestion_status = "failed"
            document.failure_code = "reconciled_incomplete"
            staged_count += 1
    session.commit()

    ready_chunks = (
        session.query(models.Chunk)
        .join(models.Chunk.document)
        .options(joinedload(models.Chunk.document))
        .filter(models.Document.ingestion_status == "ready")
        .order_by(models.Chunk.id)
        .all()
    )
    for chunk in ready_chunks:
        if not chunk.vector_id:
            chunk.vector_id = f"chunk:{chunk.id}"
    session.commit()

    ready_vector_ids = {chunk.vector_id for chunk in ready_chunks}
    collection_vector_ids = set(await vector_store.list_ids())
    stale_vector_ids = sorted(collection_vector_ids - ready_vector_ids)
    deleted_nonready = len(set(stale_vector_ids) & nonready_vector_ids)
    deleted_orphans = len(stale_vector_ids) - deleted_nonready
    await vector_store.delete(stale_vector_ids)

    repaired = 0
    for start in range(0, len(ready_chunks), batch_size):
        batch = ready_chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        embeddings = await embedding_provider.embed_texts(texts)
        ids = [chunk.vector_id for chunk in batch]
        await vector_store.upsert_embeddings(
            embeddings,
            [chunk.get_chunk_metadata() for chunk in batch],
            ids,
            documents=texts,
        )
        repaired += len(batch)

    return {
        "nonready_vectors_deleted": deleted_nonready,
        "orphan_vectors_deleted": deleted_orphans,
        "staged_documents_failed": staged_count,
        "ready_chunks_upserted": repaired,
    }
