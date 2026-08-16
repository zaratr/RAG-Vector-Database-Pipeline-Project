"""Optional structured Gemma safety review (Task 10C.3).

Pure typed provider-outcome contract (B-09): the provider adapter, Pydantic
result types, the deterministic/LLM merge algorithm, and the
disabled/rules_only/fail_closed mode semantics. No persistence — all
persistence and enforcement is owned by 10C.4.
"""
from __future__ import annotations

import asyncio
import json
from typing import Literal, Protocol

from pydantic import BaseModel
from jsonschema import Draft202012Validator

from app.services.content_safety import SafetyFinding, classify_content
from app.services.safety_policy import SafetyPolicy, action_precedence

PROMPT_VERSION = "safety-review-v1"
RESULT_SCHEMA_VERSION = "safety-result-v1"
PROVIDER_TIMEOUT_SECONDS = 120
MAX_TRANSPORT_ATTEMPTS = 3
MAX_OUTPUT_REPAIRS = 1

SAFETY_CATEGORIES = (
    "violence", "self_harm", "sexual_content", "hate_harassment",
    "illegal_activity", "privacy_credentials",
)
SAFETY_ACTIONS = ("allow", "warn", "filter", "block")


class LLMSafetyFinding(BaseModel):
    category: Literal["violence", "self_harm", "sexual_content",
                      "hate_harassment", "illegal_activity", "privacy_credentials"]
    severity: int
    action: Literal["allow", "warn", "filter", "block"]
    start: int
    end: int
    evidence: str


class LLMSafetyReviewResult(BaseModel):
    findings: list[LLMSafetyFinding]
    overall_action: Literal["allow", "warn", "filter", "block"]
    rationale_codes: list[str]


#: Plan envelope name; the appendix tests construct LLMSafetyReviewResult.
LLMSafetyResult = LLMSafetyReviewResult

#: Strict JSON Schema for the provider envelope (no extra keys anywhere).
SCHEMA = Draft202012Validator({
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["findings", "overall_action", "rationale_codes"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "severity", "action", "start", "end",
                             "evidence"],
                "properties": {
                    "category": {"enum": list(SAFETY_CATEGORIES)},
                    "severity": {"type": "integer", "minimum": 0, "maximum": 4},
                    "action": {"enum": list(SAFETY_ACTIONS)},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                    "evidence": {"type": "string"},
                },
            },
        },
        "overall_action": {"enum": list(SAFETY_ACTIONS)},
        "rationale_codes": {"type": "array", "items": {"type": "string"}},
    },
})


class LLMSafetyTimeoutError(Exception):
    """Provider exceeded the configured review timeout."""


class LLMSafetyTransportError(Exception):
    """Provider transport failed after the bounded attempt budget."""


class LLMSafetyMalformedOutputError(Exception):
    """Provider output failed schema validation even after repair."""


class LLMSafetyReviewFailure(BaseModel):
    """Typed fail-closed outcome carrying a stable error code."""

    error_code: str
    prompt_version: str = PROMPT_VERSION
    schema_version: str = RESULT_SCHEMA_VERSION


class SafetyReviewOutcome(BaseModel):
    """Successful review outcome: merged findings + provider status."""

    llm_status: Literal["skipped", "failed", "ok"]
    findings: list[SafetyFinding]
    overall_action: Literal["allow", "warn", "filter", "block"]
    policy_version: str


class SafetyReviewProvider(Protocol):
    async def review(self, text: str, scope: str) -> LLMSafetyReviewResult:
        ...


def merge_findings(
    deterministic: list[SafetyFinding],
    llm: list[LLMSafetyFinding],
    policy_version: str = "",
) -> list[SafetyFinding]:
    """Merge deterministic and LLM findings into canonical components.

    Deterministic findings keep ``source_rule_ids=[rule_id]``; LLM findings
    become ``source_rule_ids=["LLM_<category>"]``. Per category, overlapping
    half-open spans (``max(start_a,start_b) < min(end_a,end_b)``; adjacency is
    not overlap) chain into one component emitting the union span, strongest
    action by fixed precedence, maximum severity, and the sorted unique union
    of source rule IDs. Exact duplicate findings collapse. Deterministic
    findings are never downgraded or removed (their actions participate in the
    precedence fold). Output sorted by ``(start, end, category,
    source_rule_ids)``.
    """
    version = policy_version
    if deterministic and deterministic[0].policy_version:
        version = deterministic[0].policy_version

    entries: list[dict] = []
    seen: set = set()
    for finding in deterministic:
        key = (finding.start, finding.end, finding.category,
               tuple(finding.source_rule_ids))
        if key in seen:
            continue
        seen.add(key)
        entries.append({
            "start": finding.start, "end": finding.end,
            "category": finding.category, "action": finding.action,
            "severity": finding.severity,
            "rule_ids": tuple(finding.source_rule_ids),
        })
    for finding in llm:
        rule_id = f"LLM_{finding.category}"
        key = (finding.start, finding.end, finding.category, (rule_id,))
        if key in seen:
            continue
        seen.add(key)
        entries.append({
            "start": finding.start, "end": finding.end,
            "category": finding.category, "action": finding.action,
            "severity": finding.severity, "rule_ids": (rule_id,),
        })

    merged: list[SafetyFinding] = []
    for category in sorted({entry["category"] for entry in entries}):
        spans = sorted(
            (e for e in entries if e["category"] == category),
            key=lambda e: (e["start"], e["end"], e["rule_ids"]),
        )
        component: dict | None = None
        for entry in spans:
            if component is not None and entry["start"] < component["end"]:
                # Overlap (strictly less: adjacency does not merge).
                component["end"] = max(component["end"], entry["end"])
                component["action"] = action_precedence(
                    component["action"], entry["action"])
                component["severity"] = max(component["severity"],
                                            entry["severity"])
                component["rule_ids"] = tuple(sorted(
                    set(component["rule_ids"]) | set(entry["rule_ids"])))
            else:
                if component is not None:
                    merged.append(_component_finding(component, version))
                component = {
                    "start": entry["start"], "end": entry["end"],
                    "category": category, "action": entry["action"],
                    "severity": entry["severity"],
                    "rule_ids": entry["rule_ids"],
                }
        if component is not None:
            merged.append(_component_finding(component, version))

    merged.sort(key=lambda f: (f.start, f.end, f.category,
                               tuple(f.source_rule_ids)))
    return merged


def _component_finding(component: dict, policy_version: str) -> SafetyFinding:
    return SafetyFinding(
        category=component["category"],
        severity=component["severity"],
        action=component["action"],  # type: ignore[arg-type]
        start=component["start"],
        end=component["end"],
        source_rule_ids=list(component["rule_ids"]),
        policy_version=policy_version,
    )


def _validate_provider_result(result: LLMSafetyReviewResult, text: str) -> None:
    """Every non-allow finding needs valid original offsets whose slice equals
    the reported evidence; otherwise the whole output is malformed."""
    for finding in result.findings:
        if finding.action == "allow":
            continue
        if not (0 <= finding.start < finding.end <= len(text)):
            raise LLMSafetyMalformedOutputError(
                f"offset out of range: [{finding.start},{finding.end})"
            )
        if text[finding.start:finding.end] != finding.evidence:
            raise LLMSafetyMalformedOutputError(
                "text[start:end] does not equal reported evidence"
            )


def _overall(findings: list[SafetyFinding]) -> str:
    action = "allow"
    for finding in findings:
        action = action_precedence(action, finding.action)
    return action  # type: ignore[return-value]


async def review_with_llm(
    text: str,
    *,
    scope: str,
    policy: SafetyPolicy,
    mode: str,
    provider: SafetyReviewProvider | None = None,
) -> SafetyReviewOutcome | LLMSafetyReviewFailure:
    """Run the mode-aware safety review over the deterministic first pass.

    ``disabled`` returns deterministic-only findings with ``llm_status=
    "skipped"``. ``rules_only`` returns deterministic findings with
    ``llm_status="failed"`` when the provider fails. ``fail_closed`` returns a
    typed :class:`LLMSafetyReviewFailure` carrying a stable error code.
    Overall action is always recomputed from the merged findings, never
    copied from provider output.
    """
    deterministic = classify_content(text, scope=scope, policy=policy).findings

    if mode == "disabled":
        return SafetyReviewOutcome(
            llm_status="skipped",
            findings=list(deterministic),
            overall_action=_overall(deterministic),  # type: ignore[arg-type]
            policy_version=policy.version,
        )

    if provider is None:
        if mode == "rules_only":
            return SafetyReviewOutcome(
                llm_status="failed",
                findings=list(deterministic),
                overall_action=_overall(deterministic),  # type: ignore[arg-type]
                policy_version=policy.version,
            )
        return LLMSafetyReviewFailure(error_code="provider_unavailable")

    try:
        result = await asyncio.wait_for(
            provider.review(text, scope),
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        result = None  # handled below as provider_timeout
    except LLMSafetyTimeoutError:
        result = None
    except LLMSafetyMalformedOutputError:
        if mode == "rules_only":
            return SafetyReviewOutcome(
                llm_status="failed", findings=list(deterministic),
                overall_action=_overall(deterministic),  # type: ignore[arg-type]
                policy_version=policy.version,
            )
        return LLMSafetyReviewFailure(error_code="provider_output_malformed")
    except LLMSafetyTransportError:
        if mode == "rules_only":
            return SafetyReviewOutcome(
                llm_status="failed", findings=list(deterministic),
                overall_action=_overall(deterministic),  # type: ignore[arg-type]
                policy_version=policy.version,
            )
        return LLMSafetyReviewFailure(error_code="provider_transport_error")
    except Exception:
        if mode == "rules_only":
            return SafetyReviewOutcome(
                llm_status="failed", findings=list(deterministic),
                overall_action=_overall(deterministic),  # type: ignore[arg-type]
                policy_version=policy.version,
            )
        return LLMSafetyReviewFailure(error_code="provider_transport_error")

    if result is None:
        if mode == "rules_only":
            return SafetyReviewOutcome(
                llm_status="failed", findings=list(deterministic),
                overall_action=_overall(deterministic),  # type: ignore[arg-type]
                policy_version=policy.version,
            )
        return LLMSafetyReviewFailure(error_code="provider_timeout")

    try:
        _validate_provider_result(result, text)
    except LLMSafetyMalformedOutputError:
        if mode == "rules_only":
            return SafetyReviewOutcome(
                llm_status="failed", findings=list(deterministic),
                overall_action=_overall(deterministic),  # type: ignore[arg-type]
                policy_version=policy.version,
            )
        return LLMSafetyReviewFailure(error_code="provider_output_malformed")

    merged = merge_findings(deterministic, result.findings, policy.version)
    return SafetyReviewOutcome(
        llm_status="ok",
        findings=merged,
        overall_action=_overall(merged),  # type: ignore[arg-type]
        policy_version=policy.version,
    )


_REVIEW_SYSTEM_PROMPT = (
    "You are a content-safety reviewer. Classify the supplied text for the "
    "six categories (violence, self_harm, sexual_content, hate_harassment, "
    "illegal_activity, privacy_credentials). Return ONLY minified JSON "
    'matching {"findings":[{"category":...,"severity":0-4,"action":'
    '"allow|warn|filter|block","start":int,"end":int,"evidence":str}],'
    '"overall_action":str,"rationale_codes":[str]}. Offsets are half-open '
    "into the ORIGINAL text and evidence must equal text[start:end]."
)


class OllamaSafetyReviewProvider:
    """Structured Gemma review through the OpenAI-compatible Ollama API.

    Bounded retries: at most three transport attempts and one output repair,
    matching graph extraction's envelope.
    """

    def __init__(self, base_url: str, model: str, transport=None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self._transport = transport
        self._http = httpx.AsyncClient(
            timeout=PROVIDER_TIMEOUT_SECONDS, transport=transport
        )

    async def review(self, text: str, scope: str) -> LLMSafetyReviewResult:
        import httpx

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Scope: {scope}\nText:\n{text}"},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(MAX_TRANSPORT_ATTEMPTS):
            try:
                resp = await self._http.post(
                    f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return self._parse(content)
            except (httpx.TransportError, httpx.HTTPError) as exc:
                last_error = exc
                await asyncio.sleep(0.5 * (attempt + 1))
        raise LLMSafetyTransportError(str(last_error))

    def _parse(self, content: str) -> LLMSafetyReviewResult:
        candidates = [content]
        # One bounded repair: strip code fences the model sometimes adds.
        repaired = content.strip()
        if repaired.startswith("```"):
            repaired = repaired.strip("`")
            if repaired.startswith("json"):
                repaired = repaired[4:]
            candidates.append(repaired.strip())
        last_error: Exception | None = None
        for candidate in candidates[: MAX_OUTPUT_REPAIRS + 1]:
            try:
                obj = json.loads(candidate)
                SCHEMA.validate(obj)
                return LLMSafetyReviewResult(**obj)
            except Exception as exc:
                last_error = exc
        raise LLMSafetyMalformedOutputError(str(last_error))
