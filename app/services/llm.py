"""LLM client interfaces and implementations."""
from __future__ import annotations

from typing import List, Protocol


class LLMClient(Protocol):
    async def generate_answer(self, query: str, context: List[str]) -> str:
        ...


class DummyLLMClient:
    """Simple LLM that echoes query and context."""

    async def generate_answer(self, query: str, context: List[str]) -> str:
        context_preview = "\n".join(context)
        return f"Answer to: {query}\nContext:\n{context_preview}"


class OpenAILLMClient:
    """Placeholder for OpenAI chat completions."""

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key

    async def generate_answer(self, query: str, context: List[str]) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAILLMClient")
        context_preview = " ".join(context)[:2000]
        return f"[OpenAI simulated] {query} | context: {context_preview}"


def get_llm_client(api_key: str | None = None, provider: str = "dummy") -> LLMClient:
    if provider == "openai":
        return OpenAILLMClient(api_key)
    return DummyLLMClient()
