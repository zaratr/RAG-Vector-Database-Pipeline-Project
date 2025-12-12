"""RAG orchestration logic."""
from __future__ import annotations

from typing import List

from app.services.embeddings import EmbeddingProvider
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore
from app.services import retrieval


async def answer_query(
    *,
    query: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_client: LLMClient,
    top_k: int = 5,
    filters: dict | None = None,
) -> dict:
    contexts = await retrieval.retrieve(
        query=query, embedding_provider=embedding_provider, vector_store=vector_store, top_k=top_k, filters=filters
    )
    context_texts: List[str] = [ctx["text"] for ctx in contexts]
    answer = await llm_client.generate_answer(query, context_texts)
    return {"answer": answer, "context": contexts}
