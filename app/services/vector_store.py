"""Vector store interface and Chroma implementation."""
from __future__ import annotations

from typing import List, Protocol

import chromadb
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.logging import logger
from app.core.models import RetrievedChunk


class VectorStore(Protocol):
    async def index_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        documents: List[str] | None = None,
    ) -> None:
        ...

    async def query(self, embedding: List[float], top_k: int, filters: dict | None = None) -> List[RetrievedChunk]:
        ...

    async def upsert_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        documents: List[str] | None = None,
    ) -> None:
        ...

    async def delete(self, ids: List[str]) -> None:
        ...

    async def list_ids(self) -> List[str]:
        ...


def _create_client() -> chromadb.api.ClientAPI:
    """Create the correct Chroma client based on configuration.

    Precedence:
      1. chroma_host set       → HttpClient (Docker/production)
      2. persist_directory set → PersistentClient (standalone)
      3. otherwise             → EphemeralClient (tests/dev)
    """
    settings = get_settings()
    if settings.chroma_host:
        logger.info("Creating Chroma HttpClient(host=%s, port=%s)", settings.chroma_host, settings.chroma_port)
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    if settings.chroma_persist_directory:
        logger.info("Creating Chroma PersistentClient(path=%s)", settings.chroma_persist_directory)
        return chromadb.PersistentClient(path=settings.chroma_persist_directory)
    logger.info("Creating Chroma EphemeralClient")
    return chromadb.EphemeralClient()


# Chroma 1.5.9 rejects batches above its server-side maximum; stay well below.
_UPSERT_BATCH_SIZE = 4096


class ChromaVectorStore:
    """Wrapper around Chroma DB with configurable client mode."""

    def __init__(self, collection_name: str = "rag-collection", client=None) -> None:
        self.collection_name = collection_name
        self.client = client or _create_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )

    def delete_collection(self, name: str | None = None) -> None:
        """Delete a collection by exact name (defaults to this store's collection)."""
        self.client.delete_collection(name or self.collection_name)

    async def index_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        documents: List[str] | None = None,
    ) -> None:
        logger.info("Indexing %s embeddings", len(embeddings))
        await self._add_in_batches(
            self.collection.add, embeddings, metadatas, ids, documents
        )

    async def upsert_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        documents: List[str] | None = None,
    ) -> None:
        logger.info("Upserting %s embeddings", len(embeddings))
        await self._add_in_batches(
            self.collection.upsert, embeddings, metadatas, ids, documents
        )

    async def _add_in_batches(self, operation, embeddings, metadatas, ids, documents):
        """Apply add/upsert in server-safe batches (Chroma caps batch size)."""
        for start in range(0, len(ids), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            kwargs = {
                "embeddings": embeddings[start:end],
                "metadatas": metadatas[start:end],
                "ids": ids[start:end],
            }
            if documents is not None:
                kwargs["documents"] = documents[start:end]
            await run_in_threadpool(operation, **kwargs)

    async def delete(self, ids: List[str]) -> None:
        if ids:
            await run_in_threadpool(self.collection.delete, ids=ids)

    async def list_ids(self) -> List[str]:
        result = await run_in_threadpool(self.collection.get, include=[])
        return list(result.get("ids", []))

    async def query(self, embedding: List[float], top_k: int, filters: dict | None = None) -> List[RetrievedChunk]:
        logger.info("Querying vector store with top_k=%s", top_k)
        where_clause = filters if filters else None
        result = await run_in_threadpool(
            self.collection.query, query_embeddings=[embedding], n_results=top_k, where=where_clause
        )
        contexts = []
        for vector_id, text, score, metadata in zip(
            result.get("ids", [[]])[0],
            result.get("documents", [[]])[0],
            result.get("distances", [[]])[0],
            result.get("metadatas", [[]])[0],
        ):
            contexts.append(
                RetrievedChunk(
                    text=text,
                    score=float(score),
                    metadata=metadata,
                    vector_id=vector_id,
                )
            )
        return contexts


def get_vector_store() -> VectorStore:
    return ChromaVectorStore()
