"""LLM client interfaces and implementations."""
from __future__ import annotations

from typing import List, Protocol

import httpx


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

class LLMProviderOutputError(Exception):
    """The answer provider answered but its response envelope was unusable
    (non-JSON body, missing/empty ``choices``, or missing/non-text
    ``message.content``); maps to HTTP 502."""


class OllamaLLMClient:
    """LLM client using Ollama's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport

    async def generate_answer(self, query: str, context: List[str]) -> str:
        context_text = "\n".join(
            f"Evidence [{index}]: {text}" for index, text in enumerate(context, start=1)
        )
        payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer using only the supplied evidence. Combine facts transitively "
                            "when each link in a relationship chain is explicitly supported; for "
                            "example, if A relates to B and B relates to C, explain how A connects "
                            "to C. Do not say a connection is unstated when all of its links appear "
                            "in the evidence. Cite the supporting evidence numbers in the answer. "
                            "If no supported path answers the question, say so."
                        )
                     },
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                     },
                ],
        }
        async with httpx.AsyncClient(timeout=120.0, transport=self.transport) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            try:
                content = resp.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMProviderOutputError(
                    "LLM provider returned a malformed chat completion envelope"
                ) from exc
            if not isinstance(content, str):
                raise LLMProviderOutputError(
                    "LLM provider returned a chat completion without text content"
                )
            return content

def get_llm_client(api_key: str | None = None, provider: str = "dummy",
                   base_url: str | None = None, model: str | None = None) -> LLMClient:
    if provider == "ollama":
        return OllamaLLMClient(
            base_url=base_url or "http://localhost:11434/v1",
            model=model or "gemma4:latest",
        )
            
    if provider == "openai":
        return OpenAILLMClient(api_key)
    return DummyLLMClient()
