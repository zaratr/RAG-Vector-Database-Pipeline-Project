"""Retrieval poisoning controls (Task 10B.3).

Applies distance, duplicate, and cap filtering to retrieval candidates after
SQL-authoritative identity validation but before generation. The filtering
sequence is:
identity/readiness → pre-rank → blocked trust → distance → exact duplicate →
near duplicate → per-document cap → per-source cap → emit survivors.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel

Decision = Literal[
    "selected",
    "rejected_distance",
    "rejected_blocked_source",
    "rejected_source_cap",
    "rejected_document_cap",
    "rejected_duplicate",
    "rejected_injection",
    "rejected_safety",
]


class RetrievalSecurityDecision(BaseModel):
    chunk_id: int
    decision: Decision
    native_score: float | None
    provenance_score: float
    policy_version: str
    reason_codes: list[str]


class RetrievalSecurityRegimeError(ValueError):
    """Fail-closed refusal: the policy's calibration regime != the runtime's.

    The retrieval-security thresholds (notably ``max_distance``) are
    calibrated per embedding model. Applying them under a different embedding
    regime would silently fail-close (or fail-open) retrieval with misleading
    ``rejected_distance`` decisions. Raised at policy load — mirroring the
    content-safety fail-closed pattern — naming BOTH regimes.
    """


# Effective model identity of the deterministic local hash embedding. The
# ``local`` provider (HashEmbeddingProvider) ignores the configured model
# name entirely, so its regime identity is the hash embedding itself — never
# the configured model string (R2a: the check is model-identity-based, not a
# blanket ban on the provider name).
LOCAL_HASH_EMBEDDING_MODEL = "local-hash-embedding"


def effective_runtime_embedding_model(settings) -> str:
    """The embedding-model identity the configured provider actually realizes.

    ``fastembed``/``openai`` realize the configured model name; ``local`` is
    the deterministic SHA-256 hash embedding (unnormalized vectors whose l2
    distances sit far outside any real-model calibration), so its identity is
    the hash regime.
    """
    if settings.embedding_provider == "local":
        return LOCAL_HASH_EMBEDDING_MODEL
    return settings.embedding_model


def assert_policy_matches_runtime_regime(
    policy: RetrievalSecurityPolicy, settings
) -> None:
    """Refuse (typed, fail-closed) unless the policy's calibration embedding
    model matches the runtime embedding regime's effective model identity."""
    runtime_model = effective_runtime_embedding_model(settings)
    if policy.calibration_embedding_model != runtime_model:
        raise RetrievalSecurityRegimeError(
            f"retrieval-security policy calibrated for embedding model "
            f"{policy.calibration_embedding_model!r} (normalized fastembed "
            f"regime) but runtime provider is {settings.embedding_provider}/"
            f"{settings.embedding_model!r} (effective embedding model "
            f"{runtime_model!r}); refusing to apply calibrated thresholds "
            f"across embedding regimes"
        )


@dataclass(frozen=True)
class RetrievalSecurityPolicy:
    version: str
    metric: str
    max_distance: float
    per_source_cap: int
    per_document_cap: int
    max_candidates: int
    near_duplicate_jaccard: float
    calibration_fixture_sha256: str
    calibration_clean_recall: float
    calibration_poison_share: float
    calibration_tool_version: str
    calibration_embedding_model: str = ""


# Exact canonical policy payload shape (plan §10B.3): unknown or missing keys
# fail startup; the calibration block carries tool/model/fixture versions and
# both acceptance metrics.
POLICY_TOP_LEVEL_KEYS = frozenset({
    "version", "metric", "max_distance", "per_source_cap", "per_document_cap",
    "max_candidates", "near_duplicate_jaccard", "calibration",
})
POLICY_CALIBRATION_KEYS = frozenset({
    "calibration_tool_version", "clean_recall", "embedding_model",
    "fixture_sha256", "poisoned_context_share",
})


def load_retrieval_security_policy_strict(policy_path) -> RetrievalSecurityPolicy:
    """Strictly load and validate a retrieval security policy; raise ValueError.

    Unknown/missing keys, non-l2 metric, out-of-range values, or a calibration
    fixture hash that does not match the committed fixture are all refused.
    """
    import json
    from pathlib import Path

    path = Path(policy_path)
    if not path.is_file():
        raise ValueError(f"retrieval security policy not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"retrieval security policy is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("retrieval security policy must be a JSON object")
    if set(raw) != POLICY_TOP_LEVEL_KEYS:
        unknown = sorted(set(raw) - POLICY_TOP_LEVEL_KEYS)
        missing = sorted(POLICY_TOP_LEVEL_KEYS - set(raw))
        raise ValueError(f"policy key set mismatch (unknown={unknown}, missing={missing})")
    if raw["metric"] != "l2":
        raise ValueError(f"policy metric must be 'l2', got {raw['metric']!r}")

    calibration = raw["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != POLICY_CALIBRATION_KEYS:
        raise ValueError("policy calibration key set mismatch")

    import math

    if not (isinstance(raw["max_distance"], (int, float)) and math.isfinite(raw["max_distance"]) and raw["max_distance"] > 0):
        raise ValueError("max_distance must be a positive finite number")
    for key in ("per_source_cap", "per_document_cap", "max_candidates"):
        if not (isinstance(raw[key], int) and raw[key] >= 1):
            raise ValueError(f"{key} must be an integer >= 1")
    if not (isinstance(raw["near_duplicate_jaccard"], (int, float)) and 0.0 <= raw["near_duplicate_jaccard"] <= 1.0):
        raise ValueError("near_duplicate_jaccard must be within [0, 1]")
    for key in ("clean_recall", "poisoned_context_share"):
        if not (isinstance(calibration[key], (int, float)) and 0.0 <= calibration[key] <= 1.0):
            raise ValueError(f"calibration.{key} must be within [0, 1]")
    if calibration["embedding_model"] != CALIBRATION_EMBEDDING_MODEL:
        raise ValueError(
            f"calibration.embedding_model must be {CALIBRATION_EMBEDDING_MODEL!r}"
        )

    fixture_path = Path(__file__).resolve().parents[2] / "app/tests/fixtures/retrieval_calibration.json"
    if not fixture_path.is_file():
        raise ValueError(f"calibration fixture missing: {fixture_path}")
    import hashlib as _hashlib
    fixture_sha = _hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if calibration["fixture_sha256"] != fixture_sha:
        raise ValueError("calibration fixture hash does not match the committed fixture")

    return RetrievalSecurityPolicy(
        version=raw["version"],
        metric=raw["metric"],
        max_distance=float(raw["max_distance"]),
        per_source_cap=raw["per_source_cap"],
        per_document_cap=raw["per_document_cap"],
        max_candidates=raw["max_candidates"],
        near_duplicate_jaccard=float(raw["near_duplicate_jaccard"]),
        calibration_fixture_sha256=calibration["fixture_sha256"],
        calibration_clean_recall=float(calibration["clean_recall"]),
        calibration_poison_share=float(calibration["poisoned_context_share"]),
        calibration_tool_version=calibration["calibration_tool_version"],
        calibration_embedding_model=calibration["embedding_model"],
    )


def _normalize_text(text: str) -> str:
    """NFKC + casefold + whitespace collapse."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split())
    return normalized.casefold()


def exact_duplicate_hash(text: str) -> str:
    """SHA-256 of NFKC+casefold+whitespace-collapsed text."""
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _tokenize(text: str) -> set[str]:
    """Unicode alphanumeric word tokenization of normalized text."""
    normalized = _normalize_text(text)
    return set(re.findall(r"[\w]+", normalized, re.UNICODE))


def near_duplicate_jaccard(text_a: str, text_b: str) -> float:
    """Jaccard similarity of tokenized normalized texts. Empty sets = 1.0."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / len(union)


@dataclass
class Candidate:
    """A retrieval candidate with all fields needed for security filtering."""
    chunk_id: int
    document_id: int
    source: str
    text: str
    native_score: float
    trust_tier: str
    document_ready: bool
    origin: str = "vector"  # "vector", "graph", "hybrid_vector", "hybrid_graph", "hybrid_both", "both"
    distance: float | None = None       # native L2 distance (vector/hybrid-vector origins)
    graph_score: float | None = None    # graph relevance score (graph origins)
    hop: int | None = None              # minimum hop count (graph origins)
    hybrid_score: float | None = None   # fused RRF score (hybrid mode)

    def native_distance(self) -> float:
        """The candidate's native L2 distance (explicit distance, else native_score)."""
        return self.distance if self.distance is not None else self.native_score


def prerank_vector(candidates: list[Candidate]) -> list[Candidate]:
    """Vector pre-security total order: (native_l2_distance ASC, chunk_id ASC)."""
    return sorted(candidates, key=lambda c: (c.native_distance(), c.chunk_id))


def prerank_graph(candidates: list[Candidate]) -> list[Candidate]:
    """Graph pre-security total order: (-graph_score, minimum_hop, chunk_id)."""
    return sorted(
        candidates,
        key=lambda c: (-(c.graph_score or 0.0), c.hop if c.hop is not None else 0, c.chunk_id),
    )


def prerank_hybrid(candidates: list[Candidate]) -> list[Candidate]:
    """Hybrid pre-security total order (10A.6): (-hybrid_score, chunk_id)."""
    return sorted(
        candidates,
        key=lambda c: (-(c.hybrid_score if c.hybrid_score is not None else c.native_score), c.chunk_id),
    )


_DISTANCE_ELIGIBLE_ORIGINS = frozenset({"vector", "hybrid_vector"})
_GRAPH_ORIGIN_VALUES = frozenset({"graph", "hybrid_graph", "hybrid_both", "both"})


def apply_security_filters(
    candidates: list[Candidate],
    *,
    policy: RetrievalSecurityPolicy,
    mode: str = "vector",
) -> list[RetrievalSecurityDecision]:
    """Mode-aware security filtering per the 10B.3 production ordering table.

    Each mode supplies its native total order before the order-dependent
    controls: vector ``(distance, chunk_id)``, graph ``(-graph_score, hop,
    chunk_id)``, hybrid ``(-hybrid_score, chunk_id)``. The L2 distance control
    applies only to distance-eligible candidates (all vector candidates;
    hybrid vector-only candidates). Graph-origin or graph-supported survivors
    are annotated with ``distance_not_applicable_graph_origin`` and may remain
    ``selected``. Survivor order preserves the mode's pre-rank order.
    """
    if mode == "vector":
        ranked = prerank_vector(candidates)

        def eligible(cand: Candidate) -> bool:
            return True

        def value(cand: Candidate) -> float:
            return cand.native_distance()
    elif mode == "graph":
        ranked = prerank_graph(candidates)

        def eligible(cand: Candidate) -> bool:
            return False

        def value(cand: Candidate) -> float:
            return cand.native_distance()
    elif mode == "hybrid":
        ranked = prerank_hybrid(candidates)

        def eligible(cand: Candidate) -> bool:
            return cand.origin in _DISTANCE_ELIGIBLE_ORIGINS

        def value(cand: Candidate) -> float:
            return cand.native_distance()
    else:
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    decisions = _sequential_filter(ranked, policy, distance_eligible=eligible, distance_value=value)

    # Annotate graph-origin candidates whose distance control was inapplicable.
    annotated: list[RetrievalSecurityDecision] = []
    by_chunk = {c.chunk_id: c for c in candidates}
    for decision in decisions:
        cand = by_chunk.get(decision.chunk_id)
        graph_origin = cand is not None and cand.origin in _GRAPH_ORIGIN_VALUES
        if mode == "graph" or (mode == "hybrid" and graph_origin):
            if decision.decision != "rejected_distance":
                decision.reason_codes = sorted(
                    decision.reason_codes + ["distance_not_applicable_graph_origin"]
                )
        annotated.append(decision)
    return annotated


def filter_candidates(
    candidates: list[Candidate],
    policy: RetrievalSecurityPolicy,
) -> list[RetrievalSecurityDecision]:
    """Apply the security filtering sequence and return decisions.

    Candidates must be pre-ranked by ``(native_score, document_id)`` ascending.
    The filtering sequence preserves pre-rank order for survivors.
    """
    # Pre-rank is already established by the caller; do NOT re-sort.
    # The plan specifies that each retrieval mode supplies its already-defined
    # native total order before order-dependent controls are applied.
    return _sequential_filter(
        list(candidates),
        policy,
        distance_eligible=lambda cand: cand.origin in _DISTANCE_ELIGIBLE_ORIGINS,
        distance_value=lambda cand: cand.native_score,
    )


def _sequential_filter(
    ranked: list[Candidate],
    policy: RetrievalSecurityPolicy,
    *,
    distance_eligible,
    distance_value,
) -> list[RetrievalSecurityDecision]:
    """Consume pre-ranked candidates sequentially through the common filters."""

    decisions: list[RetrievalSecurityDecision] = []
    seen_exact_hashes: dict[str, int] = {}  # hash → first kept chunk_id
    kept_representatives: list[tuple[int, str]] = []  # (chunk_id, text)
    doc_counts: dict[int, int] = {}
    source_counts: dict[str, int] = {}
    emitted = 0

    for cand in ranked:
        reason_codes: list[str] = []

        # Blocked trust
        if cand.trust_tier == "blocked":
            decisions.append(RetrievalSecurityDecision(
                chunk_id=cand.chunk_id, decision="rejected_blocked_source",
                native_score=cand.native_score, provenance_score=0.0,
                policy_version=policy.version, reason_codes=["blocked_trust"],
            ))
            continue

        # Distance — only applies to distance-eligible candidates (vector and
        # hybrid vector-only origins; graph/hybrid graph-supported candidates
        # are never compared against max_distance).
        if distance_eligible(cand) and distance_value(cand) > policy.max_distance:
            decisions.append(RetrievalSecurityDecision(
                chunk_id=cand.chunk_id, decision="rejected_distance",
                native_score=cand.native_score, provenance_score=0.0,
                policy_version=policy.version, reason_codes=["distance_exceeds_max"],
            ))
            continue

        # Exact duplicate
        text_hash = exact_duplicate_hash(cand.text)
        if text_hash in seen_exact_hashes:
            decisions.append(RetrievalSecurityDecision(
                chunk_id=cand.chunk_id, decision="rejected_duplicate",
                native_score=cand.native_score, provenance_score=0.0,
                policy_version=policy.version, reason_codes=["exact_duplicate"],
            ))
            continue

        # Near duplicate
        is_near_dup = False
        for rep_id, rep_text in kept_representatives:
            if exact_duplicate_hash(rep_text) != text_hash:
                if near_duplicate_jaccard(cand.text, rep_text) >= policy.near_duplicate_jaccard:
                    decisions.append(RetrievalSecurityDecision(
                        chunk_id=cand.chunk_id, decision="rejected_duplicate",
                        native_score=cand.native_score, provenance_score=0.0,
                        policy_version=policy.version, reason_codes=["near_duplicate"],
                    ))
                    is_near_dup = True
                    break
        if is_near_dup:
            continue

        # Per-document cap
        doc_count = doc_counts.get(cand.document_id, 0)
        if doc_count >= policy.per_document_cap:
            decisions.append(RetrievalSecurityDecision(
                chunk_id=cand.chunk_id, decision="rejected_document_cap",
                native_score=cand.native_score, provenance_score=0.0,
                policy_version=policy.version, reason_codes=["document_cap_exceeded"],
            ))
            continue

        # Per-source cap
        source_count = source_counts.get(cand.source, 0)
        if source_count >= policy.per_source_cap:
            decisions.append(RetrievalSecurityDecision(
                chunk_id=cand.chunk_id, decision="rejected_source_cap",
                native_score=cand.native_score, provenance_score=0.0,
                policy_version=policy.version, reason_codes=["source_cap_exceeded"],
            ))
            continue

        # Max candidates
        if emitted >= policy.max_candidates:
            decisions.append(RetrievalSecurityDecision(
                chunk_id=cand.chunk_id, decision="rejected_document_cap",
                native_score=cand.native_score, provenance_score=0.0,
                policy_version=policy.version, reason_codes=["max_candidates_exceeded"],
            ))
            continue

        # Selected
        seen_exact_hashes[text_hash] = cand.chunk_id
        kept_representatives.append((cand.chunk_id, cand.text))
        doc_counts[cand.document_id] = doc_count + 1
        source_counts[cand.source] = source_count + 1
        emitted += 1
        # Compute actual provenance score from trust tier + grounding factors.
        from app.services.provenance import provenance_score as compute_ps
        trust_map = {"trusted": 1.0, "standard": 0.5, "untrusted": 0.2, "blocked": 0.0}
        t_score = trust_map.get(cand.trust_tier, 0.2)
        g_score = 0.0
        if cand.document_ready:
            g_score += 0.20
        g_score += 0.25  # vector ID matched SQL (already hydrated)
        g_score += 0.25  # text hash matched (SQL authoritative)
        ps = compute_ps(trust_score=t_score, grounding_score=min(1.0, g_score))
        decisions.append(RetrievalSecurityDecision(
            chunk_id=cand.chunk_id, decision="selected",
            native_score=cand.native_score, provenance_score=ps,
            policy_version=policy.version, reason_codes=[],
        ))

    return decisions


def default_policy() -> RetrievalSecurityPolicy:
    """Return a default policy with calibration output TBD."""
    return RetrievalSecurityPolicy(
        version="retrieval-security-v1",
        metric="l2",
        max_distance=1.0,
        per_source_cap=2,
        per_document_cap=2,
        max_candidates=50,
        near_duplicate_jaccard=0.90,
        calibration_fixture_sha256="",
        calibration_clean_recall=0.0,
        calibration_poison_share=0.0,
        calibration_tool_version="calibrate-v1",
    )


CALIBRATION_EMBEDDING_MODEL = "jinaai/jina-clip-v1"


def validate_calibration_corpus(corpus: dict, schema: dict | None = None) -> None:
    """Validate the calibration corpus against schema + semantic rules.

    Raises ``jsonschema.ValidationError`` on schema violations and ``ValueError``
    on semantic violations (duplicate IDs, empty text/source, labels outside
    clean|poison, absent/overlapping query references, ordering, model/metric
    mismatch, or top_k out of range).
    """
    if schema:
        from jsonschema import Draft202012Validator
        Draft202012Validator(schema).validate(corpus)

    doc_ids = [d["id"] for d in corpus["documents"]]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("duplicate document IDs")
    doc_map = {d["id"]: d for d in corpus["documents"]}
    for d in corpus["documents"]:
        if not d["text"].strip():
            raise ValueError(f"empty text in {d['id']}")
        if not d["source"].strip():
            raise ValueError(f"empty source in {d['id']}")
        if d["label"] not in ("clean", "poison"):
            raise ValueError(f"invalid label in {d['id']}")
    if sorted(doc_ids) != doc_ids:
        raise ValueError("documents not in lexical id order")
    clean_count = sum(1 for d in corpus["documents"] if d["label"] == "clean")
    poison_count = sum(1 for d in corpus["documents"] if d["label"] == "poison")
    if clean_count < 2:
        raise ValueError("need at least 2 clean docs")
    if poison_count < 2:
        raise ValueError("need at least 2 poison docs")
    query_ids = [q["id"] for q in corpus["queries"]]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query IDs")
    if sorted(query_ids) != query_ids:
        raise ValueError("queries not in lexical id order")
    if len(corpus["queries"]) < 2:
        raise ValueError("need at least 2 queries")
    referenced = set()
    for q in corpus["queries"]:
        for rid in q["required_clean_ids"]:
            if rid not in doc_map:
                raise ValueError(f"query {q['id']} references absent doc {rid}")
            if doc_map[rid]["label"] != "clean":
                raise ValueError(f"required_clean_id {rid} not labeled clean")
            if rid in q["poisoned_ids"]:
                raise ValueError(f"overlap between required and poisoned: {rid}")
        for pid in q["poisoned_ids"]:
            if pid not in doc_map:
                raise ValueError(f"query {q['id']} references absent poison doc {pid}")
            if doc_map[pid]["label"] != "poison":
                raise ValueError(f"poisoned_id {pid} not labeled poison")
        if not q["required_clean_ids"]:
            raise ValueError(f"empty required_clean_ids in {q['id']}")
        if not q["poisoned_ids"]:
            raise ValueError(f"empty poisoned_ids in {q['id']}")
        referenced.update(q["required_clean_ids"])
        referenced.update(q["poisoned_ids"])
    unreferenced = set(doc_ids) - referenced
    if unreferenced:
        raise ValueError(f"unreferenced documents: {sorted(unreferenced)}")
    if corpus["embedding_model"] != CALIBRATION_EMBEDDING_MODEL:
        raise ValueError("embedding_model mismatch")
    if corpus["distance_metric"] != "l2":
        raise ValueError("distance_metric must be l2")
    if not (1 <= corpus["top_k"] <= 50):
        raise ValueError("top_k out of range 1-50")


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration acceptance metrics for one selection set."""

    clean_recall: float
    poisoned_context_share: float
    off_query_poison: list[str] = field(default_factory=list)


def compute_calibration_metrics(
    *,
    selected_ids: list[str],
    required_clean_ids: list[str],
    label_by_id: dict[str, str],
    poisoned_ids: list[str],
) -> CalibrationMetrics:
    """Compute clean recall, poisoned-context share, and off-query diagnostics.

    Clean recall is selected required-clean IDs over total required references.
    Poisoned-context share counts every selected document whose global label is
    ``poison`` (including off-query global poisons) over all selected documents.
    Zero selected yields 0.0 for both rates.
    """
    required_total = len(required_clean_ids)
    required_selected = sum(1 for rid in required_clean_ids if rid in selected_ids)
    selected_total = len(selected_ids)
    poison_selected = [sid for sid in selected_ids if label_by_id.get(sid) == "poison"]
    poisoned_ids_set = set(poisoned_ids)

    clean_recall = required_selected / required_total if required_total > 0 else 0.0
    poison_share = len(poison_selected) / selected_total if selected_total > 0 else 0.0
    off_query = sorted({sid for sid in poison_selected if sid not in poisoned_ids_set})
    return CalibrationMetrics(
        clean_recall=clean_recall,
        poisoned_context_share=poison_share,
        off_query_poison=off_query,
    )


ACCEPTANCE_CLEAN_RECALL = 0.90
ACCEPTANCE_POISON_SHARE = 0.20


def select_max_distance_threshold(
    evaluations: list[tuple[float, float, float]],
) -> float | None:
    """Select the largest threshold meeting both acceptance bounds.

    ``evaluations`` is a list of ``(threshold, clean_recall,
    poisoned_context_share)`` ascending by threshold. Returns the largest
    threshold with recall >= 0.90 and share <= 0.20 (both inclusive), or None.
    """
    chosen = None
    for threshold, recall, share in evaluations:
        if recall >= ACCEPTANCE_CLEAN_RECALL and share <= ACCEPTANCE_POISON_SHARE:
            chosen = threshold
    return chosen
