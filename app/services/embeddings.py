"""Embedding provider interfaces and implementations."""
from __future__ import annotations

import hashlib
from typing import List, Protocol

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from app.config import get_settings

settings = get_settings()


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


class LocalEmbeddingProvider:
    """Deterministic embedding using sentence-transformers if available, otherwise hashing."""

    def __init__(self) -> None:
        if SentenceTransformer:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.model = None

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.model:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        embeddings: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [float(int.from_bytes(digest[i : i + 4], "big") % 1000) for i in range(0, 32, 4)]
            embeddings.append(vector)
        return embeddings


class OpenAIEmbeddingProvider:
    """Placeholder for OpenAI embeddings."""

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider")
        # Stub implementation to avoid external calls during tests.
        return [[float(len(text)) for _ in range(8)] for text in texts]


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings.openai_api_key)
    return LocalEmbeddingProvider()
