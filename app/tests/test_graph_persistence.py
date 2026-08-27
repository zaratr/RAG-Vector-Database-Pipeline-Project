"""Idempotent extraction lifecycle persistence tests.

Exercises the repository API: ``derive_extraction_identity``,
``begin_chunk_extraction``, ``complete_chunk_extraction``,
``fail_chunk_extraction``, ``skip_chunk_extraction`` against an in-memory SQLite
database whose schema is created from the current ORM metadata (which mirrors the
migrated ``b7f3d5a9c2e1`` physical schema).
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.persistence.graph_repository import (
    ExtractionLeaseLost,
    InvalidExtractionTransition,
    begin_chunk_extraction,
    complete_chunk_extraction,
    derive_extraction_identity,
    fail_chunk_extraction,
    persist_chunk_extraction,
    skip_chunk_extraction,
)
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation


def _session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def _document_chunk(session, *, index=0, text="Alice works at Acme Corp."):
    document = models.Document(title="Graph source", source="unit")
    session.add(document)
    session.flush()
    chunk = models.Chunk(
        document_id=document.id,
        index=index,
        text=text,
        start_offset=0,
        end_offset=len(text),
        media_type="text/plain",
        vector_id=f"chunk:{document.id}:{index}",
    )
    session.add(chunk)
    session.flush()
    return document, chunk


def _valid_relation(chunk_text="Alice works at Acme Corp.", confidence=0.9):
    return ExtractedRelation(
        source=ExtractedEntity(name="Alice", canonical_name="alice", entity_type="person"),
        predicate="works_at",
        target=ExtractedEntity(
            name="Acme Corp", canonical_name="acme corp", entity_type="organization"
        ),
        evidence=chunk_text,
        evidence_start=0,
        evidence_end=len(chunk_text),
        confidence=confidence,
    )


def _frozen_clock(offset_seconds=0):
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return lambda: base + timedelta(seconds=offset_seconds)


def test_same_chunk_version_text_produces_one_extraction_row_across_repeated_begins():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease1 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    assert lease1.should_call_provider is True

    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    assert lease2.should_call_provider is False
    assert lease2.extraction.id == lease1.extraction.id
    assert session.query(models.GraphExtraction).count() == 1

    session.close()
    engine.dispose()


def test_changed_prompt_version_produces_distinct_extraction_identity():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    id1 = derive_extraction_identity(
        chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    id2 = derive_extraction_identity(
        chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v2", schema_version="graph-relations-v1",
    )

    assert id1.input_sha256 == id2.input_sha256
    assert id1.prompt_version != id2.prompt_version
    assert (id1.chunk_id, id1.provider, id1.model, id1.prompt_version,
            id1.schema_version, id1.input_sha256) != \
           (id2.chunk_id, id2.provider, id2.model, id2.prompt_version,
            id2.schema_version, id2.input_sha256)

    session.close()
    engine.dispose()


def test_changed_model_produces_distinct_extraction_identity():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    id1 = derive_extraction_identity(
        chunk=chunk, provider="ollama", model="gemma3",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    id2 = derive_extraction_identity(
        chunk=chunk, provider="ollama", model="gemma4",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    assert id1.model != id2.model

    session.close()
    engine.dispose()


def test_changed_chunk_text_produces_distinct_input_sha256():
    session, engine = _session()
    _, chunk1 = _document_chunk(session, text="Alice works at Acme.")
    _, chunk2 = _document_chunk(session, text="Bob works at Beta.")

    id1 = derive_extraction_identity(
        chunk=chunk1, provider="ollama", model="gemma4",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    id2 = derive_extraction_identity(
        chunk=chunk2, provider="ollama", model="gemma4",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    assert id1.input_sha256 != id2.input_sha256

    session.close()
    engine.dispose()


def test_input_sha256_is_64_lowercase_hex_characters():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    identity = derive_extraction_identity(
        chunk=chunk, provider="ollama", model="gemma4",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    sha = identity.input_sha256
    assert len(sha) == 64
    assert sha == sha.lower()
    assert all(c in "0123456789abcdef" for c in sha)

    session.close()
    engine.dispose()


def test_complete_chunk_extraction_replaces_evidence_atomically():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    relations = [_valid_relation(chunk.text)]
    extraction = complete_chunk_extraction(session, extraction=lease.extraction, relations=relations, expected_attempt_count=lease.lease_attempt_count)
    session.commit()

    assert extraction.status == "succeeded"
    assert extraction.completed_at is not None
    assert extraction.error_code is None
    assert session.query(models.GraphEdgeEvidence).count() == 1

    session.close()
    engine.dispose()


def test_complete_on_empty_relations_sets_status_empty():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    extraction = complete_chunk_extraction(session, extraction=lease.extraction, relations=[], expected_attempt_count=lease.lease_attempt_count)
    session.commit()

    assert extraction.status == "empty"
    assert extraction.completed_at is not None
    assert session.query(models.GraphEdgeEvidence).count() == 0

    session.close()
    engine.dispose()


def test_complete_on_already_succeeded_raises_invalid_extraction_transition():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    complete_chunk_extraction(session, extraction=lease.extraction, relations=[_valid_relation(chunk.text)], expected_attempt_count=lease.lease_attempt_count)
    session.commit()

    with pytest.raises(InvalidExtractionTransition):
        complete_chunk_extraction(session, extraction=lease.extraction, relations=[_valid_relation(chunk.text)], expected_attempt_count=lease.lease_attempt_count)

    session.close()
    engine.dispose()


def test_complete_on_already_empty_raises_invalid_extraction_transition():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    complete_chunk_extraction(session, extraction=lease.extraction, relations=[], expected_attempt_count=lease.lease_attempt_count)
    session.commit()

    with pytest.raises(InvalidExtractionTransition):
        complete_chunk_extraction(session, extraction=lease.extraction, relations=[], expected_attempt_count=lease.lease_attempt_count)

    session.close()
    engine.dispose()


def test_fail_on_succeeded_raises_invalid_extraction_transition():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    complete_chunk_extraction(session, extraction=lease.extraction, relations=[_valid_relation(chunk.text)], expected_attempt_count=lease.lease_attempt_count)
    session.commit()

    with pytest.raises(InvalidExtractionTransition):
        fail_chunk_extraction(
            session, extraction=lease.extraction,
            error_code="some_error", error_detail="detail",
            expected_attempt_count=lease.lease_attempt_count,
        )

    session.close()
    engine.dispose()


def test_fail_chunk_extraction_stores_bounded_error_code_and_sanitized_detail():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    extraction = fail_chunk_extraction(
        session, extraction=lease.extraction,
        error_code="provider_timeout",
        error_detail="Ollama timed out after 120s",
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()

    assert extraction.status == "failed"
    assert extraction.completed_at is not None
    assert extraction.error_code == "provider_timeout"
    assert "timed out" in extraction.error_detail

    session.close()
    engine.dispose()


def test_fail_truncates_error_detail_to_1000_characters():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    long_detail = "x" * 2000
    extraction = fail_chunk_extraction(
        session, extraction=lease.extraction,
        error_code="error", error_detail=long_detail,
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()

    assert len(extraction.error_detail) <= 1000

    session.close()
    engine.dispose()


def test_failed_and_skipped_runs_have_no_mentions_or_evidence():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    fail_chunk_extraction(
        session, extraction=lease.extraction,
        error_code="test", error_detail="detail",
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()
    assert session.query(models.EntityMention).filter_by(
        extraction_id=lease.extraction.id
    ).count() == 0

    _, chunk2 = _document_chunk(session, text="Different chunk text here.")
    skip_lease = skip_chunk_extraction(
        session, chunk=chunk2, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        reason_code="extraction_disabled",
    )
    session.commit()
    assert skip_lease.extraction.status == "skipped"
    assert skip_lease.extraction.error_code == "extraction_disabled"
    assert skip_lease.extraction.completed_at is not None
    assert session.query(models.EntityMention).filter_by(
        extraction_id=skip_lease.extraction.id
    ).count() == 0

    session.close()
    engine.dispose()


def test_successful_empty_differs_from_failed_and_skipped():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    empty_ext = complete_chunk_extraction(session, extraction=lease.extraction, relations=[], expected_attempt_count=lease.lease_attempt_count)
    assert empty_ext.status == "empty"

    _, chunk2 = _document_chunk(session, text="Second chunk text.")
    lease2 = begin_chunk_extraction(
        session, chunk=chunk2, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    failed_ext = fail_chunk_extraction(
        session, extraction=lease2.extraction,
        error_code="err", error_detail="d",
            expected_attempt_count=lease2.lease_attempt_count,
        )
    assert failed_ext.status == "failed"

    _, chunk3 = _document_chunk(session, text="Third chunk text.")
    skip_lease = skip_chunk_extraction(
        session, chunk=chunk3, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        reason_code="unsupported_media_type",
    )
    assert skip_lease.extraction.status == "skipped"

    statuses = {empty_ext.status, failed_ext.status, skip_lease.extraction.status}
    assert statuses == {"empty", "failed", "skipped"}

    session.close()
    engine.dispose()


def test_concurrent_same_identity_creation_does_not_duplicate_rows():
    """Requirement 6: two begins on same identity produce one row, one provider call."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease1 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )

    assert lease1.should_call_provider is True
    assert lease2.should_call_provider is False
    assert lease1.extraction.id == lease2.extraction.id
    assert session.query(models.GraphExtraction).count() == 1

    session.close()
    engine.dispose()


def test_concurrent_begin_two_sessions_single_winner_loser_reloads(tmp_path):
    """Genuine two-session race on one extraction identity (F3 regression).

    Schedule — the production loser interleave (statement-level READ COMMITTED
    snapshot, as under Postgres), forced deterministically at the engine
    boundary so the losing branch is exercised on every run:

      1. session A (winner, own engine) holds real uncommitted DML in its open
         transaction (as ingestion does: staged document/chunk writes precede
         the extraction begins), then begins: lookup misses, INSERT of the
         pending identity-owner row inside the savepoint succeeds, nothing
         committed yet.
      2. session B (loser, own engine) begins: its lookup runs before A
         commits, so it misses; its INSERT is parked just before execution
         until A commits.
      3. A commits.
      4. B's INSERT executes against A's committed row and violates the
         partial unique index ``uq_graph_extractions_identity_owner``.

    The loser must roll back to its savepoint, reload the winner's pending
    row, and return ``should_call_provider=False`` — no raw IntegrityError
    escaping the repository, no second row, no second provider call, and no
    reclaim of the winner's live attempt.
    """
    db_path = (tmp_path / "begin-race.db").as_posix()
    engine_a = create_engine(f"sqlite:///{db_path}")
    engine_b = create_engine(f"sqlite:///{db_path}")

    loser_insert_attempted = threading.Event()
    winner_committed = threading.Event()

    @event.listens_for(engine_b, "before_cursor_execute")
    def _park_loser_insert(conn, cursor, statement, parameters, context,
                           executemany):
        if statement.lstrip().upper().startswith("INSERT INTO GRAPH_EXTRACTIONS"):
            loser_insert_attempted.set()
            if not winner_committed.wait(timeout=15):
                raise TimeoutError("loser INSERT parked but winner never committed")

    Base.metadata.create_all(engine_a)
    factory_a = sessionmaker(bind=engine_a)
    factory_b = sessionmaker(bind=engine_b)

    session_a = factory_a()
    _, chunk = _document_chunk(session_a)
    session_a.commit()
    chunk_id = chunk.id

    # Winner: open the write transaction with real uncommitted DML first —
    # ingestion stages document/chunk inserts before its extraction begins in
    # the same transaction — then begin and hold everything uncommitted (the
    # provider call happens outside the SQL transaction).
    session_a.add(models.Document(title="winner staging", source="unit"))
    session_a.flush()
    lease_a = begin_chunk_extraction(
        session_a, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    assert lease_a.should_call_provider is True
    winner_attempt_started_at = lease_a.extraction.attempt_started_at

    outcome = {}

    def _loser():
        session_b = factory_b()
        try:
            chunk_b = session_b.get(models.Chunk, chunk_id)
            lease_b = begin_chunk_extraction(
                session_b, chunk=chunk_b, provider="ollama", model="gemma4:latest",
                prompt_version="graph-v1", schema_version="graph-relations-v1",
            )
            session_b.commit()
            # Capture while instances are still bound (session closes below).
            outcome["lease"] = {
                "should_call_provider": lease_b.should_call_provider,
                "id": lease_b.extraction.id,
                "status": lease_b.extraction.status,
                "attempt_count": lease_b.extraction.attempt_count,
                "lease_attempt_count": lease_b.lease_attempt_count,
                "attempt_started_at": lease_b.extraction.attempt_started_at,
            }
        except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            session_b.close()

    loser = threading.Thread(target=_loser)
    loser.start()
    # The loser's identity lookup has already executed (program order: the
    # lookup precedes the INSERT we park on), so committing now guarantees
    # the loser misses on lookup and collides on insert.
    assert loser_insert_attempted.wait(timeout=15), "loser never attempted its INSERT"
    session_a.commit()
    winner_committed.set()
    loser.join(timeout=15)
    assert not loser.is_alive(), "loser thread hung"

    assert "error" not in outcome, f"loser raised: {outcome.get('error')}"
    lease_b = outcome["lease"]
    assert lease_b["should_call_provider"] is False
    assert lease_b["id"] == lease_a.extraction.id
    assert lease_b["status"] == "pending"
    assert lease_b["attempt_count"] == 1
    assert lease_b["lease_attempt_count"] == 1
    # The loser reloaded the winner's attempt untouched (no reclaim).
    assert lease_b["attempt_started_at"] == winner_attempt_started_at

    # Exactly one row exists; the loser created no second row.
    verify = factory_a()
    rows = verify.query(models.GraphExtraction).all()
    assert len(rows) == 1
    assert rows[0].id == lease_a.extraction.id
    assert rows[0].status == "pending"
    assert rows[0].is_identity_owner is True
    verify.close()

    session_a.close()
    engine_a.dispose()
    engine_b.dispose()


def test_begin_on_failed_without_retry_returns_failed_without_provider_call():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    fail_chunk_extraction(
        session, extraction=lease.extraction,
        error_code="err", error_detail="d",
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()

    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=False,
    )
    assert lease2.should_call_provider is False
    assert lease2.extraction.status == "failed"

    session.close()
    engine.dispose()


def test_begin_on_failed_with_retry_resets_to_pending_and_increments_attempt():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    original_attempt = lease.extraction.attempt_count
    fail_chunk_extraction(
        session, extraction=lease.extraction,
        error_code="err", error_detail="d",
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()

    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True,
    )
    assert lease2.should_call_provider is True
    assert lease2.extraction.status == "pending"
    assert lease2.extraction.attempt_count == original_attempt + 1
    assert lease2.extraction.error_code is None
    assert lease2.extraction.completed_at is None

    session.close()
    engine.dispose()


def test_begin_on_succeeded_returns_terminal_without_provider_call():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    complete_chunk_extraction(
        session, extraction=lease.extraction,
        relations=[_valid_relation(chunk.text)],
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()

    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True,
    )
    assert lease2.should_call_provider is False
    assert lease2.extraction.status == "succeeded"

    session.close()
    engine.dispose()


def test_pending_lease_not_expired_is_not_reclaimed():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    clock = _frozen_clock(offset_seconds=0)
    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=clock,
    )
    session.commit()

    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True, now_utc=clock,
    )
    assert lease2.should_call_provider is False

    session.close()
    engine.dispose()


def test_pending_lease_expired_is_reclaimed_with_retry_failed():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    clock_start = _frozen_clock(offset_seconds=0)
    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=clock_start,
    )
    original_attempt = lease.extraction.attempt_count
    session.commit()

    clock_expired = _frozen_clock(offset_seconds=601)
    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True, now_utc=clock_expired,
    )
    assert lease2.should_call_provider is True
    assert lease2.extraction.attempt_count == original_attempt + 1

    session.close()
    engine.dispose()


def test_pending_lease_boundary_equality_retains_lease():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    clock_start = _frozen_clock(offset_seconds=0)
    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=clock_start,
    )
    session.commit()

    clock_boundary = _frozen_clock(offset_seconds=600)
    lease2 = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True, now_utc=clock_boundary,
    )
    assert lease2.should_call_provider is False

    session.close()
    engine.dispose()


def test_skip_chunk_extraction_sets_status_skipped_and_zero_attempts():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = skip_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        reason_code="extraction_disabled",
    )
    session.commit()

    assert lease.extraction.status == "skipped"
    assert lease.extraction.attempt_count == 0
    assert lease.extraction.error_code == "extraction_disabled"
    assert lease.extraction.completed_at is not None
    assert lease.should_call_provider is False

    session.close()
    engine.dispose()


def test_skip_with_unsupported_media_type_reason_code():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = skip_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        reason_code="unsupported_media_type",
    )
    session.commit()

    assert lease.extraction.status == "skipped"
    assert lease.extraction.error_code == "unsupported_media_type"

    session.close()
    engine.dispose()


def test_skip_chunk_extraction_rejects_invalid_reason_code_with_typed_error():
    """An invalid reason_code raises the repository's typed ValueError before
    any DB write instead of surfacing the lifecycle CHECK
    (``skipped -> error_code IN ('extraction_disabled',
    'unsupported_media_type')``) as a raw IntegrityError at flush time."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(ValueError, match="reason_code"):
        skip_chunk_extraction(
            session, chunk=chunk, provider="ollama", model="gemma4:latest",
            prompt_version="graph-v1", schema_version="graph-relations-v1",
            reason_code="not_a_valid_skip_reason",
        )

    session.rollback()
    assert session.query(models.GraphExtraction).count() == 0

    session.close()
    engine.dispose()


def test_direct_sql_rejects_invalid_status_value():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'bogus_status', "
            "'a' * 64, 1, 1)"
        ), {"cid": chunk.id})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_direct_sql_rejects_negative_attempt_count():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', "
            "'a' * 64, -1, 1)"
        ), {"cid": chunk.id})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_direct_sql_rejects_uppercase_input_sha256():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', "
            "'A' * 64, 1, 1)"
        ), {"cid": chunk.id})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_direct_sql_rejects_non_hex_input_sha256():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', "
            "'z' * 64, 1, 1)"
        ), {"cid": chunk.id})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_direct_sql_rejects_short_input_sha256():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', "
            "'abc123', 1, 1)"
        ), {"cid": chunk.id})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_direct_sql_rejects_long_input_sha256():
    """The 64-hex CHECK also rejects values longer than 64."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', "
            ":sha, 1, 1)"
        ), {"cid": chunk.id, "sha": "a" * 65})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_direct_sql_rejects_punctuation_input_sha256():
    """The 64-hex CHECK rejects punctuation inside the value."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', "
            ":sha, 1, 1)"
        ), {"cid": chunk.id, "sha": "a" * 32 + "!" + "a" * 31})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_partial_unique_index_prevents_duplicate_identity_owner_rows():
    session, engine = _session()
    _, chunk = _document_chunk(session)
    sha = "a" * 64

    session.execute(text(
        "INSERT INTO graph_extractions (chunk_id, provider, model, "
        "prompt_version, schema_version, status, input_sha256, "
        "attempt_count, is_identity_owner, completed_at) "
        "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', :sha, 1, 1, "
        "'2026-08-22T00:00:00+00:00')"
    ), {"cid": chunk.id, "sha": sha})

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner, completed_at) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'empty', :sha, 1, 1, "
            "'2026-08-22T00:00:00+00:00')"
        ), {"cid": chunk.id, "sha": sha})
        session.commit()

    session.rollback()
    session.close()
    engine.dispose()


def test_non_owner_rows_allowed_with_same_identity():
    session, engine = _session()
    _, chunk = _document_chunk(session)
    sha = "b" * 64

    session.execute(text(
        "INSERT INTO graph_extractions (chunk_id, provider, model, "
        "prompt_version, schema_version, status, input_sha256, "
        "attempt_count, is_identity_owner, completed_at) "
        "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', :sha, 1, 1, "
        "'2026-08-22T00:00:00+00:00')"
    ), {"cid": chunk.id, "sha": sha})
    session.execute(text(
        "INSERT INTO graph_extractions (chunk_id, provider, model, "
        "prompt_version, schema_version, status, input_sha256, "
        "attempt_count, is_identity_owner, completed_at) "
        "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'empty', :sha, 1, 0, "
        "'2026-08-22T00:00:00+00:00')"
    ), {"cid": chunk.id, "sha": sha})
    session.commit()

    assert session.query(models.GraphExtraction).count() == 2

    session.close()
    engine.dispose()


def test_lifecycle_check_succeeded_has_completed_at_and_null_errors():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    complete_chunk_extraction(
        session, extraction=lease.extraction,
        relations=[_valid_relation(chunk.text)],
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()

    ext = session.query(models.GraphExtraction).one()
    assert ext.status == "succeeded"
    assert ext.completed_at is not None
    assert ext.error_code is None
    assert ext.error_detail is None
    assert ext.attempt_count >= 1

    session.close()
    engine.dispose()


def test_lifecycle_check_failed_has_completed_at_and_non_null_error_code():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    fail_chunk_extraction(
        session, extraction=lease.extraction,
        error_code="test_error", error_detail="detail",
            expected_attempt_count=lease.lease_attempt_count,
        )
    session.commit()

    ext = session.query(models.GraphExtraction).one()
    assert ext.status == "failed"
    assert ext.completed_at is not None
    assert ext.error_code is not None
    assert ext.attempt_count >= 1

    session.close()
    engine.dispose()


def test_persist_chunk_extraction_convenience_helper_round_trips():
    """Legacy convenience helper still seeds a succeeded extraction in one call."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    extraction = persist_chunk_extraction(
        session, chunk=chunk, relations=[_valid_relation(chunk.text)],
        provider="ollama", model="gemma4:latest",
    )
    session.commit()

    assert extraction.status == "succeeded"
    assert session.query(models.GraphEdgeEvidence).count() == 1
    assert session.query(models.GraphExtraction).count() == 1

    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# CONC-1: Stale-owner fencing tests (expected_attempt_count compare-and-swap)
# ---------------------------------------------------------------------------


def test_complete_by_stale_owner_after_reclaim_raises_lease_lost():
    """A worker whose lease was reclaimed cannot complete."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    clock_start = _frozen_clock(offset_seconds=0)
    lease_a = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=clock_start,
    )
    stale_attempt_count = lease_a.lease_attempt_count
    session.commit()

    # Worker B reclaims after lease expiry.
    clock_expired = _frozen_clock(offset_seconds=601)
    begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True, now_utc=clock_expired,
    )
    session.commit()

    # Worker A tries to complete with its stale attempt_count.
    with pytest.raises(ExtractionLeaseLost):
        complete_chunk_extraction(
            session, extraction=lease_a.extraction,
            relations=[_valid_relation(chunk.text)],
            expected_attempt_count=stale_attempt_count,
        )
    session.rollback()

    # Worker B's attempt is still pending; A did not overwrite it.
    ext = session.query(models.GraphExtraction).one()
    assert ext.status == "pending"
    assert ext.attempt_count == stale_attempt_count + 1

    session.close()
    engine.dispose()


def test_complete_by_stale_owner_after_uncommitted_cross_session_reclaim_raises_lease_lost():
    """W-1 regression: the 0-row UPDATE diagnosis must re-read database state.

    The same-session committed-reclaim variant above never exercised the
    defect: committing between the reclaim and the fenced complete expires the
    ORM instance, so SQLAlchemy's evaluate synchronization refreshes it from
    the database and the stale SET values are never applied in memory.

    This test constructs the shape that variant misses. Worker A begins the
    lease and holds its instance un-expired in its own session (no commit).
    Worker B — a SECOND session over the same in-memory engine; both sessions
    share the SingletonThreadPool DBAPI connection, so B's uncommitted reclaim
    flush is visible to A's conditional UPDATE — reclaims the expired lease
    with no commit in between. A's conditional UPDATE then matches 0 rows
    while evaluate-sync applies the UPDATE's SET values to A's un-expired
    in-memory instance (it reads status="succeeded"). The diagnosis must
    therefore consult the database, not the identity map, and raise
    ExtractionLeaseLost — never InvalidExtractionTransition.
    """
    session_a, engine = _session()
    _, chunk = _document_chunk(session_a)

    clock_start = _frozen_clock(offset_seconds=0)
    lease_a = begin_chunk_extraction(
        session_a, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=clock_start,
    )
    stale_attempt_count = lease_a.lease_attempt_count
    # NO commit: worker A's instance stays live (un-expired) in its session.

    # Worker B reclaims in a second session over the same engine without any
    # commit in between; its flush rides the shared DBAPI connection.
    session_b = sessionmaker(bind=engine)()
    chunk_b = session_b.get(models.Chunk, chunk.id)
    lease_b = begin_chunk_extraction(
        session_b, chunk=chunk_b, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True, now_utc=_frozen_clock(offset_seconds=601),
    )
    assert lease_b.should_call_provider is True
    assert lease_b.lease_attempt_count == stale_attempt_count + 1

    # Worker A completes with its stale attempt_count: fenced, lease lost.
    with pytest.raises(ExtractionLeaseLost):
        complete_chunk_extraction(
            session_a, extraction=lease_a.extraction,
            relations=[_valid_relation(chunk.text)],
            expected_attempt_count=stale_attempt_count,
        )

    # Worker B's reclaim survives; worker A's poisoned objects are discarded.
    # B must commit BEFORE A rolls back: both sessions share one DBAPI
    # connection, so A's rollback would otherwise discard B's uncommitted
    # reclaim flush.
    session_b.commit()
    session_a.rollback()

    verify = sessionmaker(bind=engine)()
    ext = verify.query(models.GraphExtraction).one()
    assert ext.status == "pending"
    assert ext.attempt_count == stale_attempt_count + 1
    assert ext.error_code is None
    # The fenced worker wrote no evidence for its failed completion.
    assert verify.query(models.GraphEdgeEvidence).count() == 0
    assert verify.query(models.EntityMention).count() == 0

    verify.close()
    session_b.close()
    session_a.close()
    engine.dispose()


def test_fail_by_stale_owner_after_reclaim_raises_lease_lost():
    """Stale owner cannot fail after reclaim either."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    clock_start = _frozen_clock(offset_seconds=0)
    lease_a = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        now_utc=clock_start,
    )
    stale_attempt_count = lease_a.lease_attempt_count
    session.commit()

    clock_expired = _frozen_clock(offset_seconds=601)
    begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
        retry_failed=True, now_utc=clock_expired,
    )
    session.commit()

    with pytest.raises(ExtractionLeaseLost):
        fail_chunk_extraction(
            session, extraction=lease_a.extraction,
            error_code="stale_error", error_detail="stale",
            expected_attempt_count=stale_attempt_count,
        )
    session.rollback()

    session.close()
    engine.dispose()


def test_complete_with_matching_attempt_count_succeeds():
    """Sanity: complete with the correct expected_attempt_count succeeds."""
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    extraction = complete_chunk_extraction(
        session, extraction=lease.extraction,
        relations=[_valid_relation(chunk.text)],
        expected_attempt_count=lease.lease_attempt_count,
    )
    session.commit()

    assert extraction.status == "succeeded"
    assert extraction.completed_at is not None

    session.close()
    engine.dispose()
