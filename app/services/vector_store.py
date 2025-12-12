"""Vector store interface and Chroma implementation."""
from __future__ import annotations

from typing import List, Protocol

from chromadb import Client
from chromadb.config import Settings as ChromaSettings
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.logging import logger
from app.core.models import RetrievedChunk

settings = get_settings()


class VectorStore(Protocol):
    async def index_embeddings(self, embeddings: List[List[float]], metadatas: List[dict], ids: List[str]) -> None:
        ...

    async def query(self, embedding: List[float], top_k: int, filters: dict | None = None) -> List[RetrievedChunk]:
        ...


class ChromaVectorStore:
    """Wrapper around Chroma DB."""

    def __init__(self, collection_name: str = "rag-collection") -> None:
        client_settings = ChromaSettings(persist_directory=settings.chroma_persist_directory)
        self.client = Client(settings=client_settings)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=None)

    async def index_embeddings(self, embeddings: List[List[float]], metadatas: List[dict], ids: List[str]) -> None:
        logger.info("Indexing %s embeddings", len(embeddings))
        await run_in_threadpool(self.collection.add, embeddings=embeddings, metadatas=metadatas, ids=ids)

    async def query(self, embedding: List[float], top_k: int, filters: dict | None = None) -> List[RetrievedChunk]:
        logger.info("Querying vector store with top_k=%s", top_k)
        result = await run_in_threadpool(
            self.collection.query, query_embeddings=[embedding], n_results=top_k, where=filters or {}
        )
        contexts = []
        for text, score, metadata in zip(
            result.get("documents", [[]])[0], result.get("distances", [[]])[0], result.get("metadatas", [[]])[0]
        ):
            contexts.append(RetrievedChunk(text=text, score=float(score), metadata=metadata))
        return contexts


def get_vector_store() -> VectorStore:
    return ChromaVectorStore()
