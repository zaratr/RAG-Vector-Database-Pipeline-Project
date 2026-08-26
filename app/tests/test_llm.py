"""Tests for grounded Ollama answer generation."""
from __future__ import annotations

import json

import httpx
import pytest

from app.services.llm import LLMProviderOutputError, OllamaLLMClient


def _client_with(handler) -> OllamaLLMClient:
    return OllamaLLMClient(
        base_url="http://ollama.test/v1",
        model="gemma4",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_ollama_prompt_requires_transitive_grounding_and_numbered_evidence():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["temperature"] == 0
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        assert "Combine facts transitively" in system
        assert "Do not say a connection is unstated" in system
        assert "Evidence [1]: Aria manages Project Helios." in user
        assert "Evidence [2]: Project Helios uses Vector Engine." in user
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Aria connects to Vector Engine through Project Helios "
                                "[Evidence 1, Evidence 2]."
                            )
                        }
                    }
                ]
            },
        )

    client = OllamaLLMClient(
        base_url="http://ollama.test/v1",
        model="gemma4",
        transport=httpx.MockTransport(handler),
    )
    answer = await client.generate_answer(
        "How is Aria connected to Vector Engine?",
        ["Aria manages Project Helios.", "Project Helios uses Vector Engine."],
    )
    assert "through Project Helios" in answer


# ── Malformed provider envelopes ──────────────────────────────────────
# A provider that answers garbage (an HTML error page, a JSON body without
# a usable choices/message/content structure) must raise the typed
# LLMProviderOutputError — never an opaque KeyError/JSONDecodeError that
# would leak as an unhandled 500, and never a silent non-string return.


def _static_response(status: int, **kwargs) -> httpx.Response:
    return httpx.Response(status, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        # Non-JSON body: an HTML error page served with status 200.
        _static_response(200, text="<html><body>Bad Gateway</body></html>"),
        # JSON body missing the choices key entirely.
        _static_response(200, json={"error": "nothing here"}),
        # JSON body with an empty choices list.
        _static_response(200, json={"choices": []}),
        # choices[0] without a message object.
        _static_response(200, json={"choices": [{"finish_reason": "stop"}]}),
        # message present but content is None.
        _static_response(200, json={"choices": [{"message": {"content": None}}]}),
        # message present but content is a non-string payload.
        _static_response(
            200, json={"choices": [{"message": {"content": {"unexpected": 1}}}]}
        ),
    ],
    ids=[
        "html_body",
        "missing_choices",
        "empty_choices",
        "missing_message",
        "none_content",
        "non_string_content",
    ],
)
async def test_malformed_envelope_raises_typed_provider_output_error(response):
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    client = _client_with(handler)
    with pytest.raises(LLMProviderOutputError):
        await client.generate_answer("anything", ["evidence text"])


@pytest.mark.asyncio
async def test_http_500_raises_httpx_transport_error_not_output_error():
    """The transport lane stays distinct: a 5xx answer raises
    httpx.HTTPStatusError (which the /query route maps to the 503
    provider-unavailable contract), never LLMProviderOutputError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")

    client = _client_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.generate_answer("anything", ["evidence text"])
