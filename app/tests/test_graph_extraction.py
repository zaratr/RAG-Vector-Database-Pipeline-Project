"""Phase 10A.1/10A.2 — structured extraction contract freeze + provider behavior.

All tests use ``httpx.MockTransport`` so no live Ollama call is required. The
contract under test lives in ``app/services/graph_extraction.py``.
"""
from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.services.graph_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    GraphProviderOutputError,
    GraphProviderUnavailable,
    OllamaGraphExtractor,
    canonicalize_entity_type,
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


# ---------------------------------------------------------------------------
# 10A.1 — contract freeze tests
# ---------------------------------------------------------------------------


def test_extracted_entity_rejects_unknown_extra_field():
    with pytest.raises(ValidationError, match="extra"):
        ExtractedEntity(
            name="Alice",
            canonical_name="alice",
            entity_type="person",
            unexpected_field="value",
        )


def test_extracted_relation_rejects_unknown_extra_field():
    with pytest.raises(ValidationError, match="extra"):
        ExtractedRelation(
            source=ExtractedEntity(name="A", canonical_name="a", entity_type="person"),
            predicate="knows",
            target=ExtractedEntity(name="B", canonical_name="b", entity_type="person"),
            evidence="A knows B",
            evidence_start=0,
            evidence_end=8,
            confidence=0.9,
            unexpected="value",
        )


def test_extracted_entity_rejects_blank_name():
    with pytest.raises(ValidationError, match="min_length|string_too_short|blank"):
        ExtractedEntity(name="", canonical_name="alice", entity_type="person")


def test_extracted_entity_rejects_blank_entity_type():
    with pytest.raises(ValidationError, match="min_length|string_too_short|blank"):
        ExtractedEntity(name="Alice", canonical_name="alice", entity_type="")


def test_extracted_relation_rejects_blank_predicate():
    with pytest.raises(ValidationError, match="min_length|string_too_short|blank"):
        ExtractedRelation(
            source=ExtractedEntity(name="A", canonical_name="a", entity_type="person"),
            predicate="",
            target=ExtractedEntity(name="B", canonical_name="b", entity_type="person"),
            evidence="A knows B",
            evidence_start=0,
            evidence_end=8,
            confidence=0.9,
        )


def test_extracted_relation_rejects_blank_evidence():
    with pytest.raises(ValidationError, match="min_length|string_too_short|blank"):
        ExtractedRelation(
            source=ExtractedEntity(name="A", canonical_name="a", entity_type="person"),
            predicate="knows",
            target=ExtractedEntity(name="B", canonical_name="b", entity_type="person"),
            evidence="",
            evidence_start=0,
            evidence_end=8,
            confidence=0.9,
        )


def test_extracted_relation_rejects_confidence_above_one():
    with pytest.raises(ValidationError, match="less_than_equal|le|confidence"):
        ExtractedRelation(
            source=ExtractedEntity(name="A", canonical_name="a", entity_type="person"),
            predicate="knows",
            target=ExtractedEntity(name="B", canonical_name="b", entity_type="person"),
            evidence="A knows B",
            evidence_start=0,
            evidence_end=8,
            confidence=1.5,
        )


def test_extracted_relation_rejects_confidence_below_zero():
    with pytest.raises(ValidationError, match="greater_than_equal|ge|confidence"):
        ExtractedRelation(
            source=ExtractedEntity(name="A", canonical_name="a", entity_type="person"),
            predicate="knows",
            target=ExtractedEntity(name="B", canonical_name="b", entity_type="person"),
            evidence="A knows B",
            evidence_start=0,
            evidence_end=8,
            confidence=-0.1,
        )


@pytest.mark.asyncio
async def test_extractor_rejects_evidence_not_found_in_source_text():
    """Evidence substring that doesn't exist in source text raises GraphProviderOutputError."""
    content = json.dumps({
        "relations": [{
            "source": {"name": "Alice", "type": "person"},
            "predicate": "knows",
            "target": {"name": "Bob", "type": "person"},
            "evidence": "this text does not appear in source",
            "confidence": 0.9,
        }]
    })

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(content)

    with pytest.raises(GraphProviderOutputError, match="evidence.*not.*substring|not an exact"):
        await _extractor(handler).extract("Alice knows Bob.")


@pytest.mark.asyncio
async def test_extractor_rejects_entity_surface_not_found_in_source_text():
    """Entity name that doesn't appear in source text raises GraphProviderOutputError."""
    content = json.dumps({
        "relations": [{
            "source": {"name": "Charlie", "type": "person"},
            "predicate": "knows",
            "target": {"name": "Alice", "type": "person"},
            "evidence": "Alice knows Bob.",
            "confidence": 0.9,
        }]
    })

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(content)

    with pytest.raises(GraphProviderOutputError, match="surface.*substring|entity surface"):
        await _extractor(handler).extract("Alice knows Bob.")


def test_same_canonical_name_with_different_canonical_types_remains_distinct():
    """Two entities with same canonical_name but different entity_type are distinct keys."""
    from app.services.graph_extraction import _ExtractionEnvelope, OllamaGraphExtractor

    source_text = "Apple works at Acme. Apple is tasty."
    relations_data = [
        {
            "source": {"name": "Apple", "type": "Company"},
            "predicate": "works at",
            "target": {"name": "Acme", "type": "Organization"},
            "evidence": "Apple works at Acme.",
            "confidence": 0.8,
        },
        {
            "source": {"name": "Apple", "type": "Fruit"},
            "predicate": "is",
            "target": {"name": "Acme", "type": "Organization"},
            "evidence": "Apple is tasty.",
            "confidence": 0.7,
        },
    ]
    envelope = _ExtractionEnvelope.model_validate({"relations": relations_data})
    normalized = OllamaGraphExtractor._normalize_relations(envelope, source_text)

    assert len(normalized) == 2
    types = {r.source.entity_type for r in normalized}
    assert types == {"organization", "other"}


def test_semantic_aliases_converge_to_same_allowed_type():
    """Aliases like 'human', 'company', 'engine' map to canonical types."""
    assert canonicalize_entity_type("human") == "person"
    assert canonicalize_entity_type("individual") == "person"
    assert canonicalize_entity_type("company") == "organization"
    assert canonicalize_entity_type("org") == "organization"
    assert canonicalize_entity_type("engine") == "technology"
    assert canonicalize_entity_type("system") == "technology"
    assert canonicalize_entity_type("city") == "location"
    assert canonicalize_entity_type("topic") == "concept"


@pytest.mark.asyncio
async def test_duplicate_logical_triplets_retain_only_highest_confidence():
    """Same (source_type, predicate, target_type) deduplicates to highest confidence."""
    source_text = "Alice works at Acme."
    relations_data = [
        {
            "source": {"name": "Alice", "type": "person"},
            "predicate": "works at",
            "target": {"name": "Acme", "type": "Organization"},
            "evidence": "Alice works at Acme.",
            "confidence": 0.7,
        },
        {
            "source": {"name": "Alice", "type": "human"},
            "predicate": "works at",
            "target": {"name": "Acme", "type": "company"},
            "evidence": "Alice works at Acme.",
            "confidence": 0.95,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(json.dumps({"relations": relations_data}))

    result = await _extractor(handler).extract(source_text)

    assert len(result) == 1
    assert result[0].confidence == 0.95


@pytest.mark.asyncio
async def test_empty_relations_is_valid_successful_empty_output():
    """Empty relations array is valid, returns empty list, no error raised."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _response('{"relations": []}')

    result = await _extractor(handler).extract("No relations here at all.")

    assert result == []


@pytest.mark.asyncio
async def test_more_than_100_relations_is_rejected():
    """An envelope with 101 relations raises GraphProviderOutputError after repair."""
    valid_relation = {
        "source": {"name": "Alice", "type": "person"},
        "predicate": "knows",
        "target": {"name": "Bob", "type": "person"},
        "evidence": "Alice knows Bob.",
        "confidence": 0.9,
    }
    content = json.dumps({"relations": [valid_relation] * 101})

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(content)

    with pytest.raises(GraphProviderOutputError):
        await _extractor(handler).extract("Alice knows Bob.")


# ---------------------------------------------------------------------------
# 10A.2 — real Ollama/Gemma provider behavior (mocked transport)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_uses_strict_json_schema_response_format():
    """Request must include response_format type=json_schema, strict=True."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured["response_format"] = payload["response_format"]
        captured["temperature"] = payload["temperature"]
        captured["model"] = payload["model"]
        return _response(json.dumps({"relations": [_valid_relation()]}))

    await _extractor(handler).extract("Alice works at Acme Corp.")

    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["temperature"] == 0
    assert captured["model"] == "gemma4:latest"


@pytest.mark.asyncio
async def test_provider_endpoint_is_chat_completions():
    """POST must target <base_url>/chat/completions."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return _response(json.dumps({"relations": [_valid_relation()]}))

    await _extractor(handler).extract("Alice works at Acme Corp.")

    assert captured["url"].endswith("/chat/completions")


@pytest.mark.asyncio
async def test_provider_timeout_is_120_seconds_per_attempt():
    """Constructor default timeout is 120.0."""
    extractor = OllamaGraphExtractor(
        base_url="http://ollama.test/v1",
        model="gemma4:latest",
        transport=httpx.MockTransport(lambda req: _response('{"relations": []}')),
    )
    assert extractor.timeout == 120.0


@pytest.mark.asyncio
async def test_provider_makes_exactly_3_transport_attempts_on_persistent_503():
    """429/5xx map to GraphProviderUnavailable after exactly 3 attempts."""
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return _response("", status=503)

    with pytest.raises(GraphProviderUnavailable):
        await _extractor(handler).extract("Alice works at Acme Corp.")

    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_provider_429_maps_to_graph_provider_unavailable():
    """HTTP 429 is a transient failure, maps to GraphProviderUnavailable."""
    def handler(request: httpx.Request) -> httpx.Response:
        return _response("", status=429)

    with pytest.raises(GraphProviderUnavailable):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_provider_5xx_maps_to_graph_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return _response("", status=500)

    with pytest.raises(GraphProviderUnavailable):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_provider_other_4xx_maps_to_graph_provider_output_error():
    """HTTP 400/404 (non-429, non-5xx) maps to GraphProviderOutputError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return _response("", status=400)

    with pytest.raises(GraphProviderOutputError, match="HTTP 400"):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_provider_timeout_exception_maps_to_graph_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(GraphProviderUnavailable):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_provider_network_error_maps_to_graph_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(GraphProviderUnavailable):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_provider_malformed_envelope_maps_to_graph_provider_output_error():
    """Missing choices[0].message.content raises GraphProviderOutputError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    with pytest.raises(GraphProviderOutputError, match="envelope"):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_provider_non_text_content_maps_to_graph_provider_output_error():
    """Content that is not a string (e.g., null or dict) raises GraphProviderOutputError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": None}}]}
        )

    with pytest.raises(GraphProviderOutputError, match="not text"):
        await _extractor(handler).extract("Alice works at Acme Corp.")


@pytest.mark.asyncio
async def test_provider_performs_at_most_one_output_repair():
    """Invalid output triggers one repair retry; second failure raises GraphProviderOutputError."""
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return _response("not valid json")

    with pytest.raises(GraphProviderOutputError):
        await _extractor(handler).extract("Alice works at Acme Corp.")

    assert call_count[0] == 2  # 1 initial + 1 repair


@pytest.mark.asyncio
async def test_provider_repair_succeeds_on_second_attempt():
    """First call returns invalid JSON; repair call returns valid relations."""
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] == 1:
            return _response("not-json")
        return _response(json.dumps({"relations": [_valid_relation()]}))

    result = await _extractor(handler).extract("Alice works at Acme Corp.")

    assert len(result) == 1
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_provider_error_string_excludes_source_text_and_payload():
    """Error messages must not include raw source text or provider response payload."""
    source = "Alice works at Acme Corp. SECRET_DATA_HERE"
    content = json.dumps({"relations": [{"bad": "schema"}]})

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(content)

    with pytest.raises(GraphProviderOutputError) as exc_info:
        await _extractor(handler).extract(source)

    error_str = str(exc_info.value)
    assert "SECRET_DATA_HERE" not in error_str


@pytest.mark.asyncio
async def test_provider_transient_503_retries_then_succeeds():
    """Two 503s then 200 should succeed after retries."""
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] < 3:
            return _response("", status=503)
        return _response('{"relations": []}')

    result = await _extractor(handler).extract("No relations here.")
    assert result == []
    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_provider_exponential_backoff_uses_powers_of_two():
    """Retry backoff is exponential: backoff * 2^attempt."""
    sleep_calls = []
    extractor = OllamaGraphExtractor(
        base_url="http://ollama.test/v1",
        model="gemma4:latest",
        transport=httpx.MockTransport(lambda req: _response("", status=503)),
        retry_backoff=0.25,
    )

    original_sleep = __import__("asyncio").sleep

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    import app.services.graph_extraction as ge_mod
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(ge_mod.asyncio, "sleep", fake_sleep)
        with pytest.raises(GraphProviderUnavailable):
            await extractor.extract("test text")

    # 3 attempts = 2 backoff sleeps: 0.25*1, 0.25*2
    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(0.25)
    assert sleep_calls[1] == pytest.approx(0.5)
