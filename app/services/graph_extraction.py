"""Strict structured entity and relationship extraction through Ollama/Gemma."""
from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import get_settings

PROMPT_VERSION = "graph-v1"
SCHEMA_VERSION = "graph-relations-v1"


class GraphExtractionError(RuntimeError):
    """Base class for graph extraction failures."""


class GraphProviderUnavailable(GraphExtractionError):
    """Transient provider/network failure; maps to HTTP 503."""


class GraphProviderOutputError(GraphExtractionError):
    """Invalid provider envelope or structured output; maps to HTTP 502."""


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def canonicalize_entity_name(value: str) -> str:
    return _normalized_text(value).casefold()


def canonicalize_predicate(value: str) -> str:
    normalized = canonicalize_entity_name(value)
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def canonicalize_entity_type(value: str) -> str:
    raw = canonicalize_entity_name(value)
    aliases = {
        "human": "person",
        "individual": "person",
        "company": "organization",
        "org": "organization",
        "business": "organization",
        "institution": "organization",
        "engine": "technology",
        "system": "technology",
        "software": "technology",
        "platform": "technology",
        "tool": "technology",
        "database": "technology",
        "model": "technology",
        "program": "project",
        "initiative": "project",
        "service": "product",
        "city": "location",
        "country": "location",
        "region": "location",
        "place": "location",
        "topic": "concept",
        "idea": "concept",
    }
    allowed = {
        "person",
        "organization",
        "location",
        "product",
        "project",
        "technology",
        "concept",
        "event",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in allowed else "other"


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    canonical_name: str = Field(min_length=1, max_length=255)
    entity_type: str = Field(min_length=1, max_length=100)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: ExtractedEntity
    predicate: str = Field(min_length=1, max_length=255)
    target: ExtractedEntity
    evidence: str = Field(min_length=1)
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)


class _RawEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    type: str = Field(min_length=1, max_length=100)

    @field_validator("name", "type")
    @classmethod
    def normalize_nonempty(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class _RawRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: _RawEntity
    predicate: str = Field(min_length=1, max_length=255)
    target: _RawEntity
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("predicate", "evidence")
    @classmethod
    def normalize_nonempty(cls, value: str) -> str:
        normalized = _normalized_text(value)
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class _ExtractionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relations: list[_RawRelation] = Field(max_length=100)


def _summarize_validation_error(exc: ValidationError) -> str:
    """Bounded, sanitized summary of a provider-output validation failure.

    Pydantic ``ValidationError`` strings embed the raw input values — the
    provider payload — so only field locations, error types, and static
    pydantic messages are surfaced; the input itself is never included.
    """
    parts: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "root"
        summary = f"{location}: {error.get('type', 'unknown')} {error.get('msg', '')}"
        parts.append(summary.rstrip())
    return ("; ".join(parts) or "schema violation")[:200]


class GraphExtractor(Protocol):
    async def extract(self, text: str) -> list[ExtractedRelation]:
        ...


class DisabledGraphExtractor:
    async def extract(self, text: str) -> list[ExtractedRelation]:
        return []


class OllamaGraphExtractor:
    """Extract typed relations using strict JSON-schema output and bounded retries."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_transport_attempts: int = 3,
        max_output_repairs: int = 1,
        retry_backoff: float = 0.25,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.transport = transport
        self.max_transport_attempts = max_transport_attempts
        self.max_output_repairs = max_output_repairs
        self.retry_backoff = retry_backoff

    async def extract(self, text: str) -> list[ExtractedRelation]:
        source_text = text.strip()
        if not source_text:
            return []

        last_output_error: GraphProviderOutputError | None = None
        for repair_attempt in range(self.max_output_repairs + 1):
            content = await self._request(source_text, repair_attempt, last_output_error)
            try:
                envelope = _ExtractionEnvelope.model_validate_json(content)
                return self._normalize_relations(envelope, source_text)
            except ValidationError as exc:
                last_output_error = GraphProviderOutputError(
                    "Invalid graph extraction output: "
                    f"{_summarize_validation_error(exc)}"
                )
            except (ValueError, TypeError) as exc:
                last_output_error = GraphProviderOutputError(
                    f"Invalid graph extraction output: {str(exc)[:200]}"
                )
        assert last_output_error is not None
        raise last_output_error

    async def _request(
        self,
        source_text: str,
        repair_attempt: int,
        previous_error: GraphProviderOutputError | None,
    ) -> str:
        system_prompt = (
            "Extract only factual relationships explicitly supported by the supplied text. "
            "Return JSON matching the supplied schema. Evidence must be an exact substring "
            "of the text. Entity type must be one of person, organization, location, product, "
            "project, technology, concept, event, or other. Return an empty relations array "
            "when no relation exists."
        )
        if repair_attempt:
            system_prompt += (
                " Your previous response was invalid. Return only schema-valid JSON with no "
                "markdown fences, prose, or unknown fields."
            )
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "graph_extraction",
                    "strict": True,
                    "schema": _ExtractionEnvelope.model_json_schema(),
                },
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_text},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(self.max_transport_attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, transport=self.transport
                ) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions", json=payload
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "transient graph provider response",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    raise GraphProviderOutputError(
                        f"Graph provider rejected structured request with HTTP {response.status_code}"
                    )
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise GraphProviderOutputError(
                        f"Invalid graph provider response envelope: {str(exc)[:200]}"
                    ) from exc
                if not isinstance(content, str):
                    raise GraphProviderOutputError(
                        "Invalid graph provider response envelope: content is not text"
                    )
                return content
            except GraphProviderOutputError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt + 1 < self.max_transport_attempts:
                    await asyncio.sleep(self.retry_backoff * (2**attempt))
        raise GraphProviderUnavailable(
            f"Graph extraction provider unavailable after {self.max_transport_attempts} attempts"
        ) from last_error

    @staticmethod
    def _normalize_relations(
        envelope: _ExtractionEnvelope, source_text: str
    ) -> list[ExtractedRelation]:
        deduplicated: dict[tuple[str, str, str], ExtractedRelation] = {}
        for raw in envelope.relations:
            source_name = _normalized_text(raw.source.name)
            target_name = _normalized_text(raw.target.name)
            evidence = _normalized_text(raw.evidence)
            evidence_start = source_text.find(evidence)
            if evidence_start < 0:
                raise ValueError("evidence is not an exact source substring")
            if source_text.find(source_name) < 0 or source_text.find(target_name) < 0:
                raise ValueError("entity surface forms must be exact source substrings")
            predicate = canonicalize_predicate(raw.predicate)
            if not predicate:
                raise ValueError("predicate must contain letters or numbers")
            relation = ExtractedRelation(
                source=ExtractedEntity(
                    name=source_name,
                    canonical_name=canonicalize_entity_name(source_name),
                    entity_type=canonicalize_entity_type(raw.source.type),
                ),
                predicate=predicate,
                target=ExtractedEntity(
                    name=target_name,
                    canonical_name=canonicalize_entity_name(target_name),
                    entity_type=canonicalize_entity_type(raw.target.type),
                ),
                evidence=evidence,
                evidence_start=evidence_start,
                evidence_end=evidence_start + len(evidence),
                confidence=raw.confidence,
            )
            key = (
                relation.source.canonical_name + "\0" + relation.source.entity_type,
                relation.predicate,
                relation.target.canonical_name + "\0" + relation.target.entity_type,
            )
            previous = deduplicated.get(key)
            if previous is None or relation.confidence > previous.confidence:
                deduplicated[key] = relation
        return list(deduplicated.values())


def get_graph_extractor() -> GraphExtractor:
    settings = get_settings()
    if not settings.graph_extraction_enabled:
        return DisabledGraphExtractor()
    if settings.llm_provider != "ollama":
        raise GraphProviderOutputError(
            "Graph extraction requires the configured Ollama/Gemma provider"
        )
    return OllamaGraphExtractor(
        base_url=settings.llm_base_url,
        model=settings.graph_extraction_model or settings.llm_model,
    )
