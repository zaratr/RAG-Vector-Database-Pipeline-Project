"""Phase 10A.1/10A.2 — structured extraction contract freeze + provider behavior.

All tests use ``httpx.MockTransport`` so no live Ollama call is required. The
contract under test lives in ``app/services/graph_extraction.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

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
    """Error messages must not include raw source text or provider response payload.

    Pydantic ``ValidationError`` strings embed raw input values, so a payload
    marker must never reach ``str``, ``repr``, or ``args`` of the surfaced
    ``GraphProviderOutputError`` — neither via a schema-violating payload nor
    via a malformed-JSON payload.
    """
    source = "Alice works at Acme Corp. SECRET_DATA_HERE"
    payload_marker = "P4YL04D-M4RK3R-7c31f0"
    contents = [
        # Schema-valid JSON whose relation carries an unknown extra field
        # (extra="forbid" -> ValidationError repr embeds the input value).
        json.dumps({"relations": [_valid_relation(unexpected_field=payload_marker)]}),
        # Malformed JSON containing the marker (json_invalid also embeds input).
        '{"relations": [{"predicate": "' + payload_marker + '"',
    ]

    for content in contents:

        def handler(request: httpx.Request) -> httpx.Response:
            return _response(content)

        with pytest.raises(GraphProviderOutputError) as exc_info:
            await _extractor(handler).extract(source)

        error = exc_info.value
        assert str(error).startswith("Invalid graph extraction output")
        for surfaced in (str(error), repr(error), " ".join(str(a) for a in error.args)):
            assert "SECRET_DATA_HERE" not in surfaced
            assert payload_marker not in surfaced


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


# ---------------------------------------------------------------------------
# 10A.2 — live acceptance script CLI contract (appendix exit codes 0/1/2)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_graph_extraction.py"
)
_SCRIPT_FIXTURE_TEXT = "Aria manages Project Helios. Project Helios uses Vector Engine."

# Startup shim loaded via PYTHONPATH inside the subprocess. It patches
# OllamaGraphExtractor.extract as soon as the script itself has imported
# app.services.graph_extraction. It deliberately does NOT put the repository
# root on sys.path: the script must bootstrap its own imports, and cwd is an
# unrelated tmp_path to prove the bootstrap is __file__-relative.
_SITECUSTOMIZE = textwrap.dedent(
    """
    import builtins
    import os
    import sys

    _mode = os.environ.get("GRAPH_EXTRACTION_TEST_MODE")
    _real_import = builtins.__import__
    _patched = False


    def _patch(module):
        if not _mode:
            return
        if _mode == "empty":

            async def extract(self, text):
                return []

        elif _mode == "unavailable":

            async def extract(self, text):
                raise module.GraphProviderUnavailable("connection refused")

        else:

            async def extract(self, text):
                return [
                    module.ExtractedRelation(
                        source=module.ExtractedEntity(
                            name="Aria",
                            canonical_name="aria",
                            entity_type="person",
                        ),
                        predicate="manages",
                        target=module.ExtractedEntity(
                            name="Project Helios",
                            canonical_name="project helios",
                            entity_type="project",
                        ),
                        evidence="Aria manages Project Helios",
                        evidence_start=0,
                        evidence_end=27,
                        confidence=0.9,
                    )
                ]

        module.OllamaGraphExtractor.extract = extract


    def _patching_import(name, *args, **kwargs):
        module = _real_import(name, *args, **kwargs)
        global _patched
        if not _patched:
            target = sys.modules.get("app.services.graph_extraction")
            # Only patch once the module body has finished executing; during
            # its own nested imports the class attribute does not exist yet.
            if target is not None and hasattr(target, "OllamaGraphExtractor"):
                _patched = True
                _patch(target)
        return module


    builtins.__import__ = _patching_import
    """
)


def _run_validation_script(monkeypatch, tmp_path, mode="grounded", provider="ollama"):
    (tmp_path / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    monkeypatch.setenv("RAG_LLM_PROVIDER", provider)
    monkeypatch.setenv("RAG_GRAPH_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("GRAPH_EXTRACTION_TEST_MODE", mode)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--text",
            _SCRIPT_FIXTURE_TEXT,
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )


def test_validate_graph_extraction_exits_0_with_grounded_relation(monkeypatch, tmp_path):
    """Script run with a mocked grounded relation exits 0 with plan-shaped JSON."""
    result = _run_validation_script(monkeypatch, tmp_path, "grounded")

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["provider"] == "ollama"
    assert output["grounding_valid"] is True
    assert len(output["relations"]) >= 1
    for relation in output["relations"]:
        assert "source" in relation
        assert "predicate" in relation
        assert "target" in relation
        assert "evidence_start" in relation
        assert "evidence_end" in relation


def test_validate_graph_extraction_exits_1_on_successful_empty_for_relation_fixture(
    monkeypatch, tmp_path
):
    """Fixture that should produce a relation but provider returns empty → exit 1."""
    result = _run_validation_script(monkeypatch, tmp_path, "empty")

    assert result.returncode == 1, result.stderr
    output = json.loads(result.stdout)
    assert output["relations"] == []
    assert output["grounding_valid"] is True


def test_validate_graph_extraction_exits_2_on_provider_failure(monkeypatch, tmp_path):
    """Provider unavailable/config error → exit 2 with sanitized JSON error to stderr."""
    result = _run_validation_script(monkeypatch, tmp_path, "unavailable")

    assert result.returncode == 2, result.stderr
    error = json.loads(result.stderr)
    assert error["provider"] == "ollama"
    assert "error" in error
    assert "detail" in error
    assert len(error["detail"]) <= 200  # bounded sanitized detail


def test_validate_graph_extraction_exits_2_on_configuration_failure(monkeypatch, tmp_path):
    """Configuration failure (unsupported provider) → exit 2, sanitized stderr, no traceback."""
    result = _run_validation_script(
        monkeypatch, tmp_path, mode=None, provider="openai"
    )

    assert result.returncode == 2, result.stderr
    error = json.loads(result.stderr)
    assert error["provider"] == "openai"
    assert "error" in error
    assert "detail" in error
    assert len(error["detail"]) <= 200  # bounded sanitized detail
    assert "Traceback" not in result.stderr
