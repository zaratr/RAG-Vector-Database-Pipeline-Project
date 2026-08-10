"""Phase 10A.8 — idempotent graph backfill tests."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.services.graph_backfill import backfill
from app.services.graph_extraction import (
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
