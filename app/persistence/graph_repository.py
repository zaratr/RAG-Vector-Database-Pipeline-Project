"""Persistence operations for normalized graph extraction provenance.

Task 10A.3 introduces an idempotent extraction lifecycle: every eligible
chunk/version has exactly one durable extraction identity, and the repository
exposes a begin/complete/fail/skip API that returns an :class:`ExtractionLease`
telling the caller whether to call the provider.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.persistence import models
from app.services.graph_extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ExtractedEntity,
    ExtractedRelation,
    GraphExtractionError,
)

_LEASE_DEFAULT_SECONDS = 600

# The only reason codes a skipped extraction may record; mirrors the W4
# lifecycle CHECK (``ck_graph_extractions_lifecycle``) so an invalid code is
# rejected with a typed repository error before any DB write instead of a raw
# IntegrityError at flush time. ``safety_blocked`` is the 10C.4 ingestion-scope
# skip reason carried by d9b5f7c1e4a3.
_SKIP_REASON_CODES = frozenset(
    {"extraction_disabled", "unsupported_media_type", "safety_blocked"}
)


class InvalidExtractionTransition(RuntimeError):
    """Raised when a terminal extraction is mutated or terminalized again."""


class ExtractionLeaseLost(GraphExtractionError):
    """Raised when a stale owner attempts complete/fail after a lease reclaim.

    The caller's ``expected_attempt_count`` no longer matches the row's current
    ``attempt_count`` because another worker reclaimed the expired lease. The
    caller must not retry the provider call but may call ``begin`` to obtain a
    fresh lease.
    """


@dataclass(frozen=True)
class ExtractionIdentity:
    chunk_id: int
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    input_sha256: str


@dataclass(frozen=True)
class ExtractionLease:
    extraction: models.GraphExtraction
    identity: ExtractionIdentity
    should_call_provider: bool
    lease_attempt_count: int


def _default_now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_dt(value: datetime) -> datetime:
    """Return a naive UTC datetime at second precision for storage/comparison."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.replace(microsecond=0)


def derive_extraction_identity(
    *,
    chunk: models.Chunk,
    provider: str,
    model: str,
    prompt_version: str,
    schema_version: str,
) -> ExtractionIdentity:
    """Compute the durable extraction identity for a chunk/version.

    ``input_sha256`` is lowercase SHA-256 of the exact UTF-8 bytes of the
    authoritative persisted ``chunk.text``; callers cannot supply or override it.
    """
    payload = (chunk.text or "").encode("utf-8")
    input_sha = hashlib.sha256(payload).hexdigest()
    return ExtractionIdentity(
        chunk_id=chunk.id,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
        input_sha256=input_sha,
    )


def _find_owner(
    session: Session, identity: ExtractionIdentity
) -> Optional[models.GraphExtraction]:
    """Return the identity-owner extraction row, if any."""
    stmt = select(models.GraphExtraction).where(
        models.GraphExtraction.chunk_id == identity.chunk_id,
        models.GraphExtraction.provider == identity.provider,
        models.GraphExtraction.model == identity.model,
        models.GraphExtraction.prompt_version == identity.prompt_version,
        models.GraphExtraction.schema_version == identity.schema_version,
        models.GraphExtraction.input_sha256 == identity.input_sha256,
        models.GraphExtraction.is_identity_owner.is_(True),
    )
    return session.execute(stmt).scalars().first()


def _lease_seconds() -> int:
    try:
        return int(get_settings().extraction_lease_seconds)
    except Exception:  # pragma: no cover - defensive
        return _LEASE_DEFAULT_SECONDS


def begin_chunk_extraction(
    session: Session,
    *,
    chunk: models.Chunk,
    provider: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    retry_failed: bool = False,
    now_utc: Callable[[], datetime] | None = None,
) -> ExtractionLease:
    """Begin (or resume) extraction for a chunk/version.

    Returns an :class:`ExtractionLease`. The caller branches only on
    ``should_call_provider``. See the state-transition table in the plan.

    When two workers begin the same absent identity concurrently, the partial
    unique index ``uq_graph_extractions_identity_owner`` admits exactly one
    insert. The insert runs inside a savepoint so the loser's IntegrityError
    is contained without poisoning the caller's open transaction (a failed
    bare flush would leave the whole Session requiring rollback); the loser
    reloads the winner's row and must not start a second provider call.
    """
    now = _normalize_dt((now_utc or _default_now_utc)())
    identity = derive_extraction_identity(
        chunk=chunk,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
    )
    existing = _find_owner(session, identity)

    if existing is None:
        try:
            with session.begin_nested():
                extraction = models.GraphExtraction(
                    chunk_id=identity.chunk_id,
                    provider=identity.provider,
                    model=identity.model,
                    prompt_version=identity.prompt_version,
                    schema_version=identity.schema_version,
                    input_sha256=identity.input_sha256,
                    status="pending",
                    attempt_count=1,
                    attempt_started_at=now,
                    completed_at=None,
                    error_code=None,
                    error_detail=None,
                    is_identity_owner=True,
                )
                session.add(extraction)
            return ExtractionLease(extraction, identity, should_call_provider=True,
                                   lease_attempt_count=extraction.attempt_count)
        except IntegrityError:
            # Concurrent same-identity begin: the partial unique index
            # uq_graph_extractions_identity_owner rejected this insert because
            # another worker won the race and created the pending row. The
            # savepoint above rolled the failed insert back; cooperatively
            # reload the winner's row. The loser must NOT start a second
            # provider call (plan: "exactly one caller obtains
            # should_call_provider=True, losers reload the winner and do not
            # call the provider").
            winner = _find_owner(session, identity)
            if winner is None:
                # The violation was not a same-identity race the caller can
                # recover from (or the winner's row is not visible in this
                # transaction); surface the original error, never swallow it.
                raise
            return ExtractionLease(winner, identity, should_call_provider=False,
                                   lease_attempt_count=winner.attempt_count)

    status = existing.status

    # Terminal successful/empty/skipped: return unchanged.
    if status in ("succeeded", "empty", "skipped"):
        return ExtractionLease(existing, identity, should_call_provider=False,
                               lease_attempt_count=existing.attempt_count)

    # Pending: never start a second provider call. Only an expired lease plus
    # retry_failed reclaims it.
    if status == "pending":
        if retry_failed and existing.attempt_started_at is not None:
            lease_seconds = _lease_seconds()
            started = _normalize_dt(existing.attempt_started_at).timestamp()
            # Expiry is strictly started < now - lease_seconds; boundary equality
            # (started == now - lease_seconds) retains the lease.
            if now.timestamp() - lease_seconds <= started:
                return ExtractionLease(existing, identity, should_call_provider=False,
                                       lease_attempt_count=existing.attempt_count)
            # Expired: reclaim.
            existing.attempt_count = (existing.attempt_count or 0) + 1
            existing.attempt_started_at = now
            existing.completed_at = None
            existing.error_code = None
            existing.error_detail = None
            session.flush()
            return ExtractionLease(existing, identity, should_call_provider=True,
                                   lease_attempt_count=existing.attempt_count)
        return ExtractionLease(existing, identity, should_call_provider=False,
                               lease_attempt_count=existing.attempt_count)

    # Failed: only retry_failed reclaims it.
    if status == "failed":
        if not retry_failed:
            return ExtractionLease(existing, identity, should_call_provider=False,
                                   lease_attempt_count=existing.attempt_count)
        existing.status = "pending"
        existing.attempt_count = (existing.attempt_count or 0) + 1
        existing.attempt_started_at = now
        existing.completed_at = None
        existing.error_code = None
        existing.error_detail = None
        session.flush()
        return ExtractionLease(existing, identity, should_call_provider=True,
                               lease_attempt_count=existing.attempt_count)

    # Defensive: any other status returns unchanged.
    return ExtractionLease(existing, identity, should_call_provider=False,
                           lease_attempt_count=existing.attempt_count)


def _diagnose_update_failure(session: Session, extraction_id: int,
                             expected_attempt_count: int) -> None:
    """After a 0-row conditional UPDATE, diagnose whether the row is terminal
    (InvalidExtractionTransition) or a stale-owner (ExtractionLeaseLost).

    This diagnostic read occurs AFTER the failed atomic UPDATE, so no
    read-then-write gap exists.
    """
    current = session.get(models.GraphExtraction, extraction_id)
    if current is None:
        raise InvalidExtractionTransition(
            f"extraction {extraction_id} no longer exists"
        )
    if current.status != "pending":
        raise InvalidExtractionTransition(
            f"extraction {extraction_id} is terminal ({current.status}); "
            "complete/fail require status pending"
        )
    raise ExtractionLeaseLost(
        f"extraction {extraction_id} lease lost: expected attempt_count="
        f"{expected_attempt_count}, actual={current.attempt_count}"
    )


def complete_chunk_extraction(
    session: Session,
    *,
    extraction: models.GraphExtraction,
    relations: list[ExtractedRelation],
    expected_attempt_count: int,
    now_utc: Callable[[], datetime] | None = None,
) -> models.GraphExtraction:
    """Atomically transition pending → succeeded/empty and replace evidence.

    Uses a conditional UPDATE (compare-and-swap) on ``attempt_count`` to prevent
    a stale owner from committing after a lease reclaim. If the row's
    ``attempt_count`` differs from ``expected_attempt_count``, raises
    ``ExtractionLeaseLost``.
    """
    now = _normalize_dt((now_utc or _default_now_utc)())
    new_status = "empty" if not relations else "succeeded"

    result = session.execute(
        update(models.GraphExtraction)
        .where(
            models.GraphExtraction.id == extraction.id,
            models.GraphExtraction.status == "pending",
            models.GraphExtraction.attempt_count == expected_attempt_count,
        )
        .values(
            status=new_status,
            completed_at=now,
            error_code=None,
            error_detail=None,
        )
    )
    if result.rowcount == 0:
        _diagnose_update_failure(session, extraction.id, expected_attempt_count)

    # Refresh the ORM object to reflect the conditional UPDATE.
    session.refresh(extraction)

    # Replace evidence owned by this extraction identity: delete old mentions and
    # evidence created by this extraction, then write the new ones.
    session.query(models.EntityMention).filter(
        models.EntityMention.extraction_id == extraction.id
    ).delete(synchronize_session=False)
    session.query(models.GraphEdgeEvidence).filter(
        models.GraphEdgeEvidence.extraction_id == extraction.id
    ).delete(synchronize_session=False)

    chunk = session.get(models.Chunk, extraction.chunk_id)
    for relation in relations:
        _persist_relation(session, extraction=extraction, chunk=chunk, relation=relation)

    session.flush()
    return extraction


def fail_chunk_extraction(
    session: Session,
    *,
    extraction: models.GraphExtraction,
    error_code: str,
    error_detail: str,
    expected_attempt_count: int,
    now_utc: Callable[[], datetime] | None = None,
) -> models.GraphExtraction:
    """Transition pending → failed with a bounded, sanitized error record.

    Uses the same conditional UPDATE (compare-and-swap) fencing as ``complete``.
    """
    now = _normalize_dt((now_utc or _default_now_utc)())
    bounded_code = (error_code or "")[:100]
    bounded_detail = (error_detail or "")[:1000]

    result = session.execute(
        update(models.GraphExtraction)
        .where(
            models.GraphExtraction.id == extraction.id,
            models.GraphExtraction.status == "pending",
            models.GraphExtraction.attempt_count == expected_attempt_count,
        )
        .values(
            status="failed",
            completed_at=now,
            error_code=bounded_code,
            error_detail=bounded_detail,
        )
    )
    if result.rowcount == 0:
        _diagnose_update_failure(session, extraction.id, expected_attempt_count)

    session.refresh(extraction)
    return extraction


def skip_chunk_extraction(
    session: Session,
    *,
    chunk: models.Chunk,
    provider: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    reason_code: str,
    now_utc: Callable[[], datetime] | None = None,
) -> ExtractionLease:
    """Create or reload the complete skipped identity; never calls the provider."""
    if reason_code not in _SKIP_REASON_CODES:
        raise ValueError(
            f"invalid skip reason_code {reason_code!r}; must be one of "
            f"{sorted(_SKIP_REASON_CODES)} (ck_graph_extractions_lifecycle)"
        )
    now = _normalize_dt((now_utc or _default_now_utc)())
    identity = derive_extraction_identity(
        chunk=chunk,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
    )
    existing = _find_owner(session, identity)
    if existing is None:
        extraction = models.GraphExtraction(
            chunk_id=identity.chunk_id,
            provider=identity.provider,
            model=identity.model,
            prompt_version=identity.prompt_version,
            schema_version=identity.schema_version,
            input_sha256=identity.input_sha256,
            status="skipped",
            attempt_count=0,
            attempt_started_at=now,
            completed_at=now,
            error_code=reason_code,
            error_detail=None,
            is_identity_owner=True,
        )
        session.add(extraction)
        session.flush()
        return ExtractionLease(extraction, identity, should_call_provider=False,
                               lease_attempt_count=extraction.attempt_count)
    # Identity unchanged: return the existing terminal row unchanged.
    return ExtractionLease(existing, identity, should_call_provider=False,
                           lease_attempt_count=existing.attempt_count)


# ---------------------------------------------------------------------------
# Internal relation persistence (shared with the legacy convenience helper)
# ---------------------------------------------------------------------------


def _get_or_create_entity(
    session: Session, extracted: ExtractedEntity
) -> models.GraphEntity:
    entity = (
        session.query(models.GraphEntity)
        .filter(
            models.GraphEntity.canonical_name == extracted.canonical_name,
            models.GraphEntity.entity_type == extracted.entity_type,
        )
        .one_or_none()
    )
    if entity is None:
        entity = models.GraphEntity(
            canonical_name=extracted.canonical_name,
            display_name=extracted.name,
            entity_type=extracted.entity_type,
        )
        session.add(entity)
        session.flush()
    return entity


def _mention_offsets(chunk_text: str, surface_form: str) -> tuple[int, int]:
    start = chunk_text.find(surface_form)
    if start < 0:
        raise ValueError(f"Entity surface form is not present in source chunk: {surface_form}")
    return start, start + len(surface_form)


def _get_or_create_mention(
    session: Session,
    *,
    entity: models.GraphEntity,
    extraction: models.GraphExtraction,
    surface_form: str,
    chunk_text: str,
) -> models.EntityMention:
    mention = (
        session.query(models.EntityMention)
        .filter(
            models.EntityMention.entity_id == entity.id,
            models.EntityMention.extraction_id == extraction.id,
        )
        .one_or_none()
    )
    if mention is None:
        start, end = _mention_offsets(chunk_text, surface_form)
        mention = models.EntityMention(
            entity_id=entity.id,
            extraction_id=extraction.id,
            surface_form=surface_form,
            start_offset=start,
            end_offset=end,
        )
        session.add(mention)
        # Sessions run with autoflush=False; flush so a later relation's
        # dedup query sees this pending mention instead of accumulating a
        # duplicate that violates the unique constraint at commit time.
        session.flush()
    return mention


def _get_or_create_edge(
    session: Session,
    *,
    source: models.GraphEntity,
    predicate: str,
    target: models.GraphEntity,
) -> models.GraphEdge:
    edge = (
        session.query(models.GraphEdge)
        .filter(
            models.GraphEdge.source_entity_id == source.id,
            models.GraphEdge.predicate == predicate,
            models.GraphEdge.target_entity_id == target.id,
        )
        .one_or_none()
    )
    if edge is None:
        edge = models.GraphEdge(
            source_entity_id=source.id,
            predicate=predicate,
            target_entity_id=target.id,
        )
        session.add(edge)
        session.flush()
    return edge


def _persist_relation(
    session: Session,
    *,
    extraction: models.GraphExtraction,
    chunk: models.Chunk,
    relation: ExtractedRelation,
) -> None:
    source = _get_or_create_entity(session, relation.source)
    target = _get_or_create_entity(session, relation.target)
    _get_or_create_mention(
        session,
        entity=source,
        extraction=extraction,
        surface_form=relation.source.name,
        chunk_text=chunk.text,
    )
    _get_or_create_mention(
        session,
        entity=target,
        extraction=extraction,
        surface_form=relation.target.name,
        chunk_text=chunk.text,
    )
    edge = _get_or_create_edge(
        session, source=source, predicate=relation.predicate, target=target
    )
    existing_evidence = (
        session.query(models.GraphEdgeEvidence)
        .filter(
            models.GraphEdgeEvidence.edge_id == edge.id,
            models.GraphEdgeEvidence.extraction_id == extraction.id,
            models.GraphEdgeEvidence.evidence_start == relation.evidence_start,
            models.GraphEdgeEvidence.evidence_end == relation.evidence_end,
        )
        .one_or_none()
    )
    if existing_evidence is None:
        session.add(
            models.GraphEdgeEvidence(
                edge_id=edge.id,
                extraction_id=extraction.id,
                evidence_text=relation.evidence,
                evidence_start=relation.evidence_start,
                evidence_end=relation.evidence_end,
                confidence=relation.confidence,
            )
        )
        # Same autoflush=False invisibility as mentions (D-29/D-33): flush so
        # an identical duplicate relation's dedup query sees this row.
        session.flush()
    elif relation.confidence > existing_evidence.confidence:
        existing_evidence.confidence = relation.confidence


def persist_chunk_extraction(
    session: Session,
    *,
    chunk: models.Chunk,
    relations: list[ExtractedRelation],
    provider: str,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    schema_version: str = SCHEMA_VERSION,
) -> models.GraphExtraction:
    """Convenience helper: begin + complete in one call.

    Kept for the operator CLI and tests that seed a graph in one step. Uses the
    idempotent lifecycle: an existing terminal owner is left unchanged; otherwise
    a pending row is created and immediately completed.
    """
    lease = begin_chunk_extraction(
        session,
        chunk=chunk,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
    )
    if not lease.should_call_provider:
        return lease.extraction
    return complete_chunk_extraction(
        session, extraction=lease.extraction, relations=relations,
        expected_attempt_count=lease.lease_attempt_count,
    )
