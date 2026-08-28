"""Server-assigned provenance and trust assessment (Task 10B.2).

Ensures client-supplied ``source`` labels cannot grant trust and every candidate
has a deterministic provenance assessment. The source-trust policy is loaded at
startup and validated; trust is assigned server-side based on the policy, never
client-side.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

TrustTier = Literal["trusted", "standard", "untrusted", "blocked"]

POLICY_VERSION = "source-trust-v1"


class SourceTrustPolicy:
    """Validated, immutable source-trust policy."""

    def __init__(self, raw: dict):
        self.version = raw["version"]
        self.default_tier: TrustTier = raw["default"]["tier"]
        self.default_score: float = float(raw["default"]["score"])
        self.rules: list[dict] = raw["rules"]
        self._by_source: dict[str, dict] = {r["source"].strip(): r for r in self.rules}
        self._by_rule_id: dict[str, dict] = {r["rule_id"]: r for r in self.rules}

    def assess(self, source: str | None, is_operator: bool) -> tuple[TrustTier, float]:
        """Return ``(tier, score)`` for the given source and operator flag."""
        if source is None:
            return self.default_tier, self.default_score
        rule = self._by_source.get(source.strip())
        if rule is None:
            return self.default_tier, self.default_score
        if rule.get("requires_operator") and not is_operator:
            return self.default_tier, self.default_score
        return rule["tier"], float(rule["score"])

    def is_blocked(self, source: str | None) -> bool:
        if source is None:
            return False
        rule = self._by_source.get(source.strip())
        return rule is not None and rule["tier"] == "blocked"

    def requires_operator(self, source: str | None) -> bool:
        if source is None:
            return False
        rule = self._by_source.get(source.strip())
        return rule is not None and rule.get("requires_operator", False)


def load_source_trust_policy(path: str) -> SourceTrustPolicy:
    """Load and validate the source-trust policy from ``path``.

    Raises ``ValueError`` on missing/invalid/unreadable policy, duplicate
    sources or rule IDs, unknown keys, or a trusted rule without
    ``requires_operator``.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"source-trust policy not found: {path}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if "version" not in raw or "default" not in raw or "rules" not in raw:
        raise ValueError("source-trust policy missing required keys")
    default = raw["default"]
    if "tier" not in default or "score" not in default:
        raise ValueError("source-trust policy default missing tier/score")
    sources_seen: set[str] = set()
    rule_ids_seen: set[str] = set()
    for rule in raw["rules"]:
        for key in ("rule_id", "source", "tier", "score", "requires_operator"):
            if key not in rule:
                raise ValueError(f"rule missing key: {key}")
        src = rule["source"].strip()
        if src in sources_seen:
            raise ValueError(f"duplicate source: {src}")
        sources_seen.add(src)
        if rule["rule_id"] in rule_ids_seen:
            raise ValueError(f"duplicate rule_id: {rule['rule_id']}")
        rule_ids_seen.add(rule["rule_id"])
        if rule["tier"] == "trusted" and not rule["requires_operator"]:
            raise ValueError(
                f"rule {rule['rule_id']} assigns trusted without requires_operator"
            )
    return SourceTrustPolicy(raw)


class ProvenanceAssessment(BaseModel):
    document_id: int
    chunk_id: int
    trust_tier: TrustTier
    trust_score: float = Field(ge=0, le=1)   # [0,1]
    grounding_score: float = Field(ge=0, le=1)  # [0,1]
    reasons: list[str]       # sorted stable codes
    policy_version: str

    @property
    def provenance_score(self) -> float:
        """Final ``round(0.6 * trust_score + 0.4 * grounding_score, 6)``."""
        return provenance_score(self.trust_score, self.grounding_score)


def assess_provenance(
    *,
    document_id: int,
    chunk_id: int,
    source: str | None,
    operator_authenticated: bool,
    vector_id_match: bool,
    sql_text_hash_match: bool,
    document_ready: bool,
    ingestion_origin: str | None,
    has_graph_evidence: bool,
    policy_version: str,
    policy: SourceTrustPolicy | None = None,
) -> ProvenanceAssessment:
    """Deterministic server-side provenance assessment for one candidate.

    Composes the trust policy's server-assigned ``(tier, score)`` with the
    five-component grounding score and the documented ``provenance_score``
    formula. When ``policy`` is omitted, the active committed source-trust
    policy is loaded from the configured path (fail-closed on any policy
    error, exactly like API startup).
    """
    if policy is None:
        from app.config import get_settings

        policy = load_source_trust_policy(get_settings().source_trust_policy_path)
    trust_tier, trust_score = policy.assess(source, is_operator=operator_authenticated)
    grounding_score_value, reasons = compute_grounding_score(
        vector_id_matches_sql=vector_id_match,
        text_hash_matches=sql_text_hash_match,
        document_ready=document_ready,
        has_ingestion_origin=bool(ingestion_origin),
        has_graph_evidence=has_graph_evidence,
    )
    return ProvenanceAssessment(
        document_id=document_id,
        chunk_id=chunk_id,
        trust_tier=trust_tier,
        trust_score=trust_score,
        grounding_score=grounding_score_value,
        reasons=reasons,
        policy_version=policy_version,
    )


def compute_grounding_score(
    *,
    vector_id_matches_sql: bool,
    text_hash_matches: bool,
    document_ready: bool,
    has_ingestion_origin: bool,
    has_graph_evidence: bool,
) -> tuple[float, list[str]]:
    """Compute a deterministic grounding score in ``[0,1]``.

    Returns ``(score, reasons)`` where reasons are sorted stable codes for
    contributing and missing factors.
    """
    score = 0.0
    reasons: list[str] = []
    if vector_id_matches_sql:
        score += 0.25
        reasons.append("grounding_vector_id_match")
    else:
        reasons.append("grounding_vector_id_missing")
    if text_hash_matches:
        score += 0.25
        reasons.append("grounding_text_hash_match")
    else:
        reasons.append("grounding_text_hash_missing")
    if document_ready:
        score += 0.20
        reasons.append("grounding_document_ready")
    else:
        reasons.append("grounding_document_not_ready")
    if has_ingestion_origin:
        score += 0.15
        reasons.append("grounding_ingestion_origin")
    else:
        reasons.append("grounding_ingestion_origin_missing")
    if has_graph_evidence:
        score += 0.15
        reasons.append("grounding_graph_evidence")
    else:
        reasons.append("grounding_graph_evidence_missing")
    score = min(1.0, max(0.0, score))
    return score, sorted(reasons)


def provenance_score(trust_score: float, grounding_score: float) -> float:
    """Final ``provenance_score = round(0.6 * trust + 0.4 * grounding, 6)``."""
    return round(0.6 * trust_score + 0.4 * grounding_score, 6)


def content_sha256(text: str) -> str:
    """SHA-256 of the exact authoritative UTF-8 bytes of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
