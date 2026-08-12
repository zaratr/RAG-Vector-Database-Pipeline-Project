"""Idempotent graph backfill for ready text chunks lacking the current extraction identity.

Task 10A.8: extracts graph facts for ready text chunks that predate GraphRAG or
lack the current extraction identity. Reuses the 10A.3 repository API; never
modifies vector IDs, Chroma records, document readiness, or old extraction
evidence rows from other versions.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from app.persistence import models
from app.persistence.graph_repository import (
    ExtractionLeaseLost,
    InvalidExtractionTransition,
    begin_chunk_extraction,
    complete_chunk_extraction,
    derive_extraction_identity,
    fail_chunk_extraction,
)
from app.services.graph_extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    GraphExtractor,
    GraphExtractionError,
)


@dataclass(frozen=True)
class BackfillReport:
    scanned: int
    eligible: int
    processed: int
    succeeded: int
    skipped: int
    empty: int
    failed: int
    lease_lost: int
    relations: int
    skip_reasons: dict[str, int] = field(default_factory=dict)


def _scan_universe(session: Session, document_id: Optional[int]) -> list[models.Chunk]:
    """Return chunks ordered by document id, chunk index, chunk id."""
    query = (
        session.query(models.Chunk)
        .join(models.Chunk.document)
        .options(joinedload(models.Chunk.document))
        .order_by(models.Document.id, models.Chunk.index, models.Chunk.id)
    )
    if document_id is not None:
        query = query.filter(models.Chunk.document_id == document_id)
    return query.all()


def _has_owner_identity(session: Session, chunk: models.Chunk, provider: str, model: str) -> bool:
    """Return True if any identity-owner extraction row exists for this chunk."""
    # Re-derive the identity and check for an owner row matching current text.
    identity = derive_extraction_identity(
        chunk=chunk,
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    return (
        session.query(models.GraphExtraction)
        .filter(
            models.GraphExtraction.chunk_id == identity.chunk_id,
            models.GraphExtraction.provider == identity.provider,
            models.GraphExtraction.model == identity.model,
            models.GraphExtraction.prompt_version == identity.prompt_version,
            models.GraphExtraction.schema_version == identity.schema_version,
            models.GraphExtraction.input_sha256 == identity.input_sha256,
            models.GraphExtraction.is_identity_owner.is_(True),
        )
        .count()
        > 0
    )


def _terminal_owner_status(
    session: Session, chunk: models.Chunk, provider: str, model: str
) -> Optional[str]:
    """Return the status of the owner row for the current identity, if any."""
    identity = derive_extraction_identity(
        chunk=chunk,
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    row = (
        session.query(models.GraphExtraction)
        .filter(
            models.GraphExtraction.chunk_id == identity.chunk_id,
            models.GraphExtraction.provider == identity.provider,
            models.GraphExtraction.model == identity.model,
            models.GraphExtraction.prompt_version == identity.prompt_version,
            models.GraphExtraction.schema_version == identity.schema_version,
            models.GraphExtraction.input_sha256 == identity.input_sha256,
            models.GraphExtraction.is_identity_owner.is_(True),
        )
        .one_or_none()
    )
    return row.status if row is not None else None


def classify_chunk(
    session: Session,
    chunk: models.Chunk,
    *,
    provider: str,
    model: str,
    retry_failed: bool,
) -> tuple[str, Optional[str]]:
    """Classify a chunk's pre-lease eligibility.

    Returns ``(state, skip_reason)`` where ``state`` is one of
    ``eligible``, ``skipped``. Skip precedence:
    document_not_ready > unsupported_media_type > current_terminal >
    failed_not_retried > pending_active.
    """
    document = chunk.document
    if document is None or document.ingestion_status != "ready":
        return ("skipped", "document_not_ready")
    if not (chunk.media_type or "text/plain").startswith("text/"):
        return ("skipped", "unsupported_media_type")

    identity = derive_extraction_identity(
        chunk=chunk,
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
    )
    owner = (
        session.query(models.GraphExtraction)
        .filter(
            models.GraphExtraction.chunk_id == identity.chunk_id,
            models.GraphExtraction.provider == identity.provider,
            models.GraphExtraction.model == identity.model,
            models.GraphExtraction.prompt_version == identity.prompt_version,
            models.GraphExtraction.schema_version == identity.schema_version,
            models.GraphExtraction.input_sha256 == identity.input_sha256,
            models.GraphExtraction.is_identity_owner.is_(True),
        )
        .one_or_none()
    )
    if owner is None:
        return ("eligible", None)
    if owner.status in ("succeeded", "empty", "skipped"):
        return ("skipped", "current_terminal")
    if owner.status == "failed" and not retry_failed:
        return ("skipped", "failed_not_retried")
    if owner.status == "pending":
        if not retry_failed:
            return ("skipped", "pending_active")
        # With retry_failed, defer to begin_chunk_extraction which checks lease
        # expiry: an expired lease is reclaimed, an active lease retains the hold.
        return ("eligible", None)
    # failed + retry_failed: eligible (lease may reclaim).
    return ("eligible", None)


async def backfill(
    session: Session,
    *,
    extractor: GraphExtractor,
    provider: str,
    model: str,
    document_id: Optional[int] = None,
    batch_size: int = 20,
    retry_failed: bool = False,
    dry_run: bool = False,
) -> BackfillReport:
    """Run an idempotent backfill and return the exact counter report."""
    if batch_size < 1 or batch_size > 100:
        raise ValueError("batch_size must be between 1 and 100")
    if document_id is not None:
        exists = (
            session.query(models.Document).filter(models.Document.id == document_id).count() > 0
        )
        if not exists:
            raise ValueError(f"unknown document id: {document_id}")

    chunks = _scan_universe(session, document_id)
    scanned = len(chunks)

    skip_reasons: dict[str, int] = {}
    eligible = 0
    processed = succeeded = empty = failed = lease_lost = relations = 0
    skipped = 0

    for chunk in chunks:
        state, reason = classify_chunk(
            session, chunk, provider=provider, model=model, retry_failed=retry_failed
        )
        if dry_run:
            if state == "eligible":
                eligible += 1
            else:
                skipped += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        if state == "skipped":
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        eligible += 1
        lease = begin_chunk_extraction(
            session,
            chunk=chunk,
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            retry_failed=retry_failed,
        )
        if not lease.should_call_provider:
            lease_lost += 1
            continue

        processed += 1
        try:
            try:
                result_relations = await extractor.extract(chunk.text)
            except Exception as extract_exc:
                fail_chunk_extraction(
                    session,
                    extraction=lease.extraction,
                    error_code=type(extract_exc).__name__[:100],
                    error_detail=str(extract_exc)[:1000],
                    expected_attempt_count=lease.lease_attempt_count,
                )
                session.commit()
                failed += 1
                continue
            complete_chunk_extraction(
                session, extraction=lease.extraction, relations=result_relations,
                expected_attempt_count=lease.lease_attempt_count,
            )
            session.commit()
            if result_relations:
                succeeded += 1
                relations += len(result_relations)
            else:
                empty += 1
        except (InvalidExtractionTransition, ExtractionLeaseLost):
            # Concurrent winner completed first or lease was reclaimed; this
            # worker loses the lease.
            lease_lost += 1
            session.rollback()

    if dry_run:
        return BackfillReport(
            scanned=scanned,
            eligible=eligible,
            processed=0,
            succeeded=0,
            skipped=skipped,
            empty=0,
            failed=0,
            lease_lost=0,
            relations=0,
            skip_reasons=skip_reasons,
        )

    return BackfillReport(
        scanned=scanned,
        eligible=eligible,
        processed=processed,
        succeeded=succeeded,
        skipped=skipped,
        empty=empty,
        failed=failed,
        lease_lost=lease_lost,
        relations=relations,
        skip_reasons=skip_reasons,
    )
