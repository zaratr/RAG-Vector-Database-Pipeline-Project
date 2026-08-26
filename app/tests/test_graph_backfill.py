"""Idempotent graph backfill tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.persistence.graph_repository import begin_chunk_extraction
from app.services.graph_backfill import backfill
from app.services.graph_extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ExtractedEntity,
    ExtractedRelation,
    GraphExtractionError,
)


class _StaticExtractor:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def extract(self, text: str):
        if self.fail:
            raise GraphExtractionError("provider down")
        return [
            ExtractedRelation(
                source=ExtractedEntity(name="Subject", canonical_name="subject", entity_type="concept"),
                predicate="describes",
                target=ExtractedEntity(name="Object", canonical_name="object", entity_type="concept"),
                evidence=text,
                evidence_start=0,
                evidence_end=len(text),
                confidence=0.9,
            )
        ]


def _session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def _ready_chunk(session, *, text="Subject describes Object.", media_type="text/plain"):
    document = models.Document(title="Backfill doc", source="unit", ingestion_status="ready")
    session.add(document)
    session.flush()
    chunk = models.Chunk(
        document_id=document.id,
        index=0,
        text=text,
        start_offset=0,
        end_offset=len(text),
        media_type=media_type,
        vector_id=f"chunk:{document.id}:0",
    )
    session.add(chunk)
    session.flush()
    return document, chunk


@pytest.mark.asyncio
async def test_backfill_processes_eligible_ready_text_chunk():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    session.commit()
    report = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest"
    )
    assert report.scanned == 1
    assert report.eligible == 1
    assert report.processed == 1
    assert report.succeeded == 1
    assert report.failed == 0
    assert report.relations == 1
    # eligible = processed + lease_lost
    assert report.eligible == report.processed + report.lease_lost
    # scanned = skipped + processed + lease_lost
    assert report.scanned == report.skipped + report.processed + report.lease_lost
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_dry_run_does_no_writes_and_reports_equations():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    session.commit()
    report = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest",
        dry_run=True,
    )
    assert report.processed == 0
    assert report.succeeded == 0
    assert report.failed == 0
    assert report.relations == 0
    assert report.eligible == 1
    # dry-run equation: scanned = skipped + eligible
    assert report.scanned == report.skipped + report.eligible
    # no extraction rows written
    assert session.query(models.GraphExtraction).count() == 0
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_second_run_is_no_op_skip_current_terminal():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    session.commit()
    await backfill(session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest")
    second = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest"
    )
    assert second.eligible == 0
    assert second.processed == 0
    assert second.succeeded == 0
    assert second.skipped == 1
    assert second.skip_reasons == {"current_terminal": 1}
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_skips_non_ready_document():
    session, engine = _session()
    document = models.Document(title="Staged", source="unit", ingestion_status="staged")
    session.add(document)
    session.flush()
    chunk = models.Chunk(
        document_id=document.id, index=0, text="Subject describes Object.",
        start_offset=0, end_offset=24, media_type="text/plain",
        vector_id=f"chunk:{document.id}:0",
    )
    session.add(chunk)
    session.commit()
    report = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest"
    )
    assert report.eligible == 0
    assert report.skip_reasons == {"document_not_ready": 1}
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_skips_unsupported_media_type():
    session, engine = _session()
    _ready_chunk(session, media_type="image/png")
    session.commit()
    report = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest"
    )
    assert report.eligible == 0
    assert report.skip_reasons == {"unsupported_media_type": 1}
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_failed_chunk_not_retried_without_flag():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    session.commit()
    # First run fails the extraction.
    await backfill(
        session, extractor=_StaticExtractor(fail=True), provider="ollama", model="gemma4:latest"
    )
    # Second run without retry_failed skips it.
    second = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest",
        retry_failed=False,
    )
    assert second.eligible == 0
    assert second.skip_reasons == {"failed_not_retried": 1}
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_retry_failed_reclaims_failed_chunk():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    session.commit()
    await backfill(
        session, extractor=_StaticExtractor(fail=True), provider="ollama", model="gemma4:latest"
    )
    second = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest",
        retry_failed=True,
    )
    assert second.eligible == 1
    assert second.processed == 1
    assert second.succeeded == 1
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_preserves_vector_ids_and_document_readiness():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    original_vector_id = chunk.vector_id
    original_status = chunk.document.ingestion_status
    session.commit()
    await backfill(session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest")
    session.refresh(chunk)
    session.refresh(chunk.document)
    assert chunk.vector_id == original_vector_id
    assert chunk.document.ingestion_status == original_status
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_unknown_document_id_is_fatal():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    session.commit()
    with pytest.raises(ValueError, match="unknown document id"):
        await backfill(
            session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest",
            document_id=999999,
        )
    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_empty_relations_counted_as_empty_not_succeeded():
    session, engine = _session()
    _, chunk = _ready_chunk(session)
    session.commit()

    class EmptyExtractor:
        async def extract(self, text):
            return []

    report = await backfill(
        session, extractor=EmptyExtractor(), provider="ollama", model="gemma4:latest"
    )
    assert report.empty == 1
    assert report.succeeded == 0
    assert report.processed == report.succeeded + report.empty + report.failed
    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# CONC-2: Stale-owner fencing through the backfill path
# ---------------------------------------------------------------------------


def test_backfill_stale_owner_after_reclaim_cannot_complete():
    """CONC-2: A backfill worker whose lease was reclaimed by another worker
    cannot overwrite the reclaiming worker's attempt.

    Reproduces: Worker A begins, provider call is slow, lease expires.
    Worker B reclaims. Worker A attempts to complete with stale attempt_count.
    Asserts ExtractionLeaseLost is raised; Worker B's attempt survives.
    """
    from datetime import datetime, timedelta, timezone
    from app.persistence.graph_repository import (
        ExtractionLeaseLost, begin_chunk_extraction,
        complete_chunk_extraction,
    )

    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="Backfill stale", source="unit", ingestion_status="ready")
    session.add(document); session.flush()
    text = "Subject describes Object."
    chunk = models.Chunk(document_id=document.id, index=0, text=text,
                         start_offset=0, end_offset=len(text),
                         media_type="text/plain", vector_id=f"chunk:{document.id}:0")
    session.add(chunk); session.commit()

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    clock_start = lambda: base
    lease_a = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=clock_start,
    )
    stale_attempt = lease_a.lease_attempt_count
    session.commit()

    clock_expired = lambda: base + timedelta(seconds=601)
    begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True, now_utc=clock_expired,
    )
    session.commit()

    with pytest.raises(ExtractionLeaseLost):
        complete_chunk_extraction(
            session, extraction=lease_a.extraction,
            relations=[ExtractedRelation(
                source=ExtractedEntity(name="Subject", canonical_name="subject", entity_type="concept"),
                predicate="describes",
                target=ExtractedEntity(name="Object", canonical_name="object", entity_type="concept"),
                evidence=text, evidence_start=0, evidence_end=len(text), confidence=0.9,
            )],
            expected_attempt_count=stale_attempt,
        )
    session.rollback()

    ext = session.query(models.GraphExtraction).one()
    assert ext.status == "pending"
    assert ext.attempt_count == stale_attempt + 1

    session.close(); engine.dispose()


# ---------------------------------------------------------------------------
# DEFECT-A regression: expired pending lease reclaim via --retry-failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_expired_pending_lease_is_reclaimed_with_retry_failed():
    """An expired pending lease + retry_failed → eligible → reclaimed."""
    from datetime import datetime, timedelta, timezone
    from app.persistence.graph_repository import begin_chunk_extraction

    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="Expired lease", source="unit", ingestion_status="ready")
    session.add(document); session.flush()
    text = "Subject describes Object."
    chunk = models.Chunk(document_id=document.id, index=0, text=text,
                         start_offset=0, end_offset=len(text),
                         media_type="text/plain", vector_id=f"chunk:{document.id}:0")
    session.add(chunk); session.commit()

    # Worker A begins at T0, creating a pending lease.
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=lambda: base,
    )
    session.commit()

    # Worker B runs backfill with retry_failed AFTER lease expiry (T0+601s).
    report = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest",
        retry_failed=True,
    )

    # The expired lease must be reclaimed: processed=1, succeeded=1.
    assert report.eligible == 1
    assert report.processed == 1
    assert report.succeeded == 1
    assert report.lease_lost == 0
    assert report.failed == 0

    session.close(); engine.dispose()


@pytest.mark.asyncio
async def test_backfill_active_pending_lease_not_reclaimed_without_retry():
    """An active pending lease without retry_failed → skipped as pending_active."""
    from datetime import datetime, timezone
    from app.persistence.graph_repository import begin_chunk_extraction

    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = models.Document(title="Active lease", source="unit", ingestion_status="ready")
    session.add(document); session.flush()
    text = "Subject describes Object."
    chunk = models.Chunk(document_id=document.id, index=0, text=text,
                         start_offset=0, end_offset=len(text),
                         media_type="text/plain", vector_id=f"chunk:{document.id}:0")
    session.add(chunk); session.commit()

    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=lambda: base,
    )
    session.commit()

    # Without retry_failed, the pending lease should be skipped.
    report = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest",
        retry_failed=False,
    )
    assert report.eligible == 0
    assert report.processed == 0
    assert report.skipped == 1
    assert report.skip_reasons == {"pending_active": 1}

    session.close(); engine.dispose()


# ---------------------------------------------------------------------------
# F5 regression: conservation equations under expiry-during-processing
# ---------------------------------------------------------------------------


def _transactional_session():
    """In-memory session honoring full transactional semantics.

    The default pysqlite driver (legacy isolation) commits a savepoint's
    inserts when the savepoint is released, so the fenced worker's later
    ``session.rollback()`` cannot undo its pending lease row — production
    Postgres does undo it. This engine uses the documented pysqlite recipe
    (driver-level autocommit plus an explicit ``BEGIN`` when SQLAlchemy
    starts a transaction) so SAVEPOINT/ROLLBACK behave as in production.
    """
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _connect(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _begin(connection):
        connection.execute(text("BEGIN"))

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


class _LeaseReclaimedDuringProviderCallExtractor:
    """Provider stub reproducing expiry-during-processing.

    While the REAL backfill() loop's provider call is in flight, a second
    worker reclaims the lease through the repository API: ``begin`` with
    ``retry_failed=True`` and an injected clock far past this worker's
    ``attempt_started_at`` (offset 7200s exceeds any configured lease). The
    second worker thereby wins the same identity lease and bumps
    ``attempt_count``, so this worker's subsequent complete/fail transition
    is fenced with ``ExtractionLeaseLost``.
    """

    def __init__(self, session, chunk, *, fail: bool = False):
        self._session = session
        self._chunk = chunk
        self.fail = fail
        self.calls = 0

    async def extract(self, text: str):
        self.calls += 1
        begin_chunk_extraction(
            self._session,
            chunk=self._chunk,
            provider="ollama",
            model="gemma4:latest",
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            retry_failed=True,
            now_utc=lambda: datetime.now(timezone.utc) + timedelta(seconds=7200),
        )
        if self.fail:
            raise GraphExtractionError("provider down")
        return [
            ExtractedRelation(
                source=ExtractedEntity(name="Subject", canonical_name="subject", entity_type="concept"),
                predicate="describes",
                target=ExtractedEntity(name="Object", canonical_name="object", entity_type="concept"),
                evidence=text,
                evidence_start=0,
                evidence_end=len(text),
                confidence=0.9,
            )
        ]


@pytest.mark.asyncio
async def test_backfill_conservation_equations_hold_when_lease_lost_mid_processing():
    """F5 regression: a chunk lease-lost mid-processing counts ONLY in lease_lost.

    Drives the REAL backfill() service loop through expiry-during-processing:
    worker A begins the lease and starts the provider call; while the call is
    in flight the lease expires and a second worker reclaims it; worker A's
    complete then raises ExtractionLeaseLost. The conservation
    equations must hold for every report in every interleaving:

        eligible = processed + lease_lost
        processed = succeeded + empty + failed
        scanned = skipped + processed + lease_lost
        skipped = sum(skip_reasons.values())

    Two-worker expectation: exactly one worker reports
    processed=1/lease_lost=0; the other reports lease_lost=1.
    """
    session, engine = _transactional_session()
    _, chunk = _ready_chunk(session)
    session.commit()

    # Worker A: the real loop; the second worker reclaims mid-provider-call.
    worker_a = _LeaseReclaimedDuringProviderCallExtractor(session, chunk)
    report_a = await backfill(
        session, extractor=worker_a, provider="ollama", model="gemma4:latest"
    )
    assert worker_a.calls == 1  # the provider call really started
    assert report_a.eligible == 1
    assert report_a.lease_lost == 1
    # The fenced chunk must NOT also be counted as processed (F5 defect: it
    # landed in BOTH processed and lease_lost).
    assert report_a.processed == 0
    assert report_a.succeeded == 0
    assert report_a.empty == 0
    assert report_a.failed == 0
    assert report_a.skipped == 0
    assert report_a.skip_reasons == {}
    assert report_a.relations == 0
    # All four conservation equations for worker A.
    assert report_a.eligible == report_a.processed + report_a.lease_lost
    assert report_a.processed == report_a.succeeded + report_a.empty + report_a.failed
    assert report_a.scanned == report_a.skipped + report_a.processed + report_a.lease_lost
    assert report_a.skipped == sum(report_a.skip_reasons.values())
    # Worker A was fenced out: its uncommitted lease row was rolled back.
    assert session.query(models.GraphExtraction).count() == 0

    # Worker B: the reclaiming worker re-runs the REAL backfill() loop on the
    # same chunk and, owning the lease, processes it to completion.
    report_b = await backfill(
        session, extractor=_StaticExtractor(), provider="ollama", model="gemma4:latest"
    )
    assert report_b.eligible == 1
    assert report_b.processed == 1
    assert report_b.succeeded == 1
    assert report_b.lease_lost == 0
    assert report_b.failed == 0
    assert report_b.skipped == 0
    assert report_b.relations == 1
    # All four conservation equations for worker B.
    assert report_b.eligible == report_b.processed + report_b.lease_lost
    assert report_b.processed == report_b.succeeded + report_b.empty + report_b.failed
    assert report_b.scanned == report_b.skipped + report_b.processed + report_b.lease_lost
    assert report_b.skipped == sum(report_b.skip_reasons.values())

    # Two-worker expectation: exactly one worker reports processed=1 and
    # lease_lost=0; the other reports lease_lost=1 (processed=0).
    outcomes = sorted((r.processed, r.lease_lost) for r in (report_a, report_b))
    assert outcomes == [(0, 1), (1, 0)]

    # Exactly one durable identity row, terminal succeeded.
    extraction = session.query(models.GraphExtraction).one()
    assert extraction.status == "succeeded"

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_backfill_conservation_equations_hold_when_lease_lost_on_fail_transition():
    """F5 companion: the fail-transition path (provider error + reclaimed lease).

    The provider call raises; worker A attempts fail_chunk_extraction, but the
    lease was reclaimed during the call, so fail raises ExtractionLeaseLost.
    The chunk counts only in lease_lost — a fenced-out failure is not a chunk
    failure (failed stays 0).
    """
    session, engine = _transactional_session()
    _, chunk = _ready_chunk(session)
    session.commit()

    extractor = _LeaseReclaimedDuringProviderCallExtractor(session, chunk, fail=True)
    report = await backfill(
        session, extractor=extractor, provider="ollama", model="gemma4:latest"
    )
    assert extractor.calls == 1
    assert report.eligible == 1
    assert report.lease_lost == 1
    assert report.processed == 0
    assert report.failed == 0
    assert report.succeeded == 0
    assert report.empty == 0
    assert report.skipped == 0
    assert report.skip_reasons == {}
    # All four conservation equations.
    assert report.eligible == report.processed + report.lease_lost
    assert report.processed == report.succeeded + report.empty + report.failed
    assert report.scanned == report.skipped + report.processed + report.lease_lost
    assert report.skipped == sum(report.skip_reasons.values())
    # The fenced worker left no durable row behind.
    assert session.query(models.GraphExtraction).count() == 0

    session.close()
    engine.dispose()
