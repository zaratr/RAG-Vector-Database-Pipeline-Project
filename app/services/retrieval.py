"""Retrieval utilities."""
from __future__ import annotations

from typing import List

from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore


async def retrieve(
    *, query: str, embedding_provider: EmbeddingProvider, vector_store: VectorStore, top_k: int = 5, filters: dict | None = None
) -> List[dict]:
    query_embedding = (await embedding_provider.embed_texts([query]))[0]
    results = await vector_store.query(query_embedding, top_k=top_k, filters=filters)
    return [result.dict() for result in results]
