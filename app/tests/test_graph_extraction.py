"""Tests for strict structured Gemma graph extraction."""
from __future__ import annotations

import json

import httpx
import pytest

from app.services.graph_extraction import (
    GraphProviderOutputError,
    GraphProviderUnavailable,
    OllamaGraphExtractor,
)


def _response(content: str, status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, text="failure")
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}
    )


def _valid_relation(**overrides):
    relation = {
        "source": {"name": "Alice", "type": "Person"},
        "predicate": "works at",
        "target": {"name": "Acme Corp", "type": "Organization"},
        "evidence": "Alice works at Acme Corp.",
        "confidence": 0.91,
    }
    relation.update(overrides)
    return relation


def _extractor(handler, **kwargs):
    return OllamaGraphExtractor(
        base_url="http://ollama.test/v1",
        model="gemma4:latest",
        transport=httpx.MockTransport(handler),
        retry_backoff=0,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_strict_extractor_uses_json_schema_and_computes_exact_offsets():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        response_format = payload["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        return _response(json.dumps({"relations": [_valid_relation()]}))

    relations = await _extractor(handler).extract("Alice works at Acme Corp.")

    assert len(relations) == 1
    assert relations[0].source.canonical_name == "alice"
    assert relations[0].source.entity_type == "person"
    assert relations[0].predicate == "works_at"
    assert relations[0].evidence_start == 0
    assert relations[0].evidence_end == 25


@pytest.mark.asyncio
async def test_same_name_different_entity_type_is_not_collapsed():
    relations = [
        _valid_relation(
            source={"name": "Apple", "type": "Company"},
            evidence="Apple works at Acme Corp.",
        ),
        _valid_relation(
            source={"name": "Apple", "type": "Fruit"},
            evidence="Apple works at Acme Corp.",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(json.dumps({"relations": relations}))

    extracted = await _extractor(handler).extract("Apple works at Acme Corp.")
    assert {item.source.entity_type for item in extracted} == {"organization", "other"}


@pytest.mark.asyncio
async def test_semantic_type_aliases_converge_for_cross_chunk_entity_resolution():
    relations = [
        _valid_relation(source={"name": "Alice", "type": "Engine"}),
        _valid_relation(source={"name": "Alice", "type": "System"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(json.dumps({"relations": relations}))

    extracted = await _extractor(handler).extract("Alice works at Acme Corp.")
    assert len(extracted) == 1
    assert extracted[0].source.entity_type == "technology"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "```json\n{\"relations\": []}\n```",
        '{"relations": [], "unexpected": true}',
        json.dumps({"relations": [_valid_relation(unexpected="value")]}),
        json.dumps({"relations": [_valid_relation(confidence=2)]}),
        json.dumps({"relations": [_valid_relation(evidence="invented quote")]}),
        json.dumps({"relations": [_valid_relation(predicate="   ")]}),
    ],
)
async def test_invalid_output_is_rejected_after_one_bounded_repair(content):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(content)

    with pytest.raises(GraphProviderOutputError):
        await _extractor(handler).extract("Alice works at Acme Corp.")
    assert calls == 2


@pytest.mark.asyncio
async def test_oversized_relation_array_is_rejected():
    content = json.dumps({"relations": [_valid_relation()] * 101})

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(content)

    with pytest.raises(GraphProviderOutputError):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_malformed_output_is_repaired_once():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("not-json")
        payload = json.loads(request.content)
        assert "previous response was invalid" in payload["messages"][0]["content"].lower()
        return _response(json.dumps({"relations": [_valid_relation()]}))

    result = await _extractor(handler).extract("Alice works at Acme Corp.")
    assert len(result) == 1
    assert calls == 2


@pytest.mark.asyncio
async def test_transient_503_retries_then_succeeds():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return _response("", status=503)
        return _response('{"relations": []}')

    assert await _extractor(handler).extract("No relation here.") == []
    assert calls == 3


@pytest.mark.asyncio
async def test_persistent_transport_failure_is_typed_unavailable():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response("", status=503)

    with pytest.raises(GraphProviderUnavailable):
        await _extractor(handler).extract("No relation here.")
    assert calls == 3


@pytest.mark.asyncio
async def test_empty_relations_is_success_not_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return _response('{"relations": []}')

    assert await _extractor(handler).extract("No factual relation here.") == []
