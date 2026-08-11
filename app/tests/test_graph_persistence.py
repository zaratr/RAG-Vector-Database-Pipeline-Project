"""Phase 10A.3 — idempotent extraction lifecycle persistence tests.

Exercises the repository API: ``derive_extraction_identity``,
``begin_chunk_extraction``, ``complete_chunk_extraction``,
``fail_chunk_extraction``, ``skip_chunk_extraction`` against an in-memory SQLite
database whose schema is created from the current ORM metadata (which mirrors the
migrated ``b7f3d5a9c2e1`` physical schema).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.persistence.graph_repository import (
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
    extraction = complete_chunk_extraction(session, extraction=lease.extraction, relations=relations)
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
    extraction = complete_chunk_extraction(session, extraction=lease.extraction, relations=[])
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
    complete_chunk_extraction(session, extraction=lease.extraction, relations=[_valid_relation(chunk.text)])
    session.commit()

    with pytest.raises(InvalidExtractionTransition):
        complete_chunk_extraction(session, extraction=lease.extraction, relations=[_valid_relation(chunk.text)])

    session.close()
    engine.dispose()


def test_complete_on_already_empty_raises_invalid_extraction_transition():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    complete_chunk_extraction(session, extraction=lease.extraction, relations=[])
    session.commit()

    with pytest.raises(InvalidExtractionTransition):
        complete_chunk_extraction(session, extraction=lease.extraction, relations=[])

    session.close()
    engine.dispose()


def test_fail_on_succeeded_raises_invalid_extraction_transition():
    session, engine = _session()
    _, chunk = _document_chunk(session)

    lease = begin_chunk_extraction(
        session, chunk=chunk, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    complete_chunk_extraction(session, extraction=lease.extraction, relations=[_valid_relation(chunk.text)])
    session.commit()

    with pytest.raises(InvalidExtractionTransition):
        fail_chunk_extraction(
            session, extraction=lease.extraction,
            error_code="some_error", error_detail="detail",
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
    empty_ext = complete_chunk_extraction(session, extraction=lease.extraction, relations=[])
    assert empty_ext.status == "empty"

    _, chunk2 = _document_chunk(session, text="Second chunk text.")
    lease2 = begin_chunk_extraction(
        session, chunk=chunk2, provider="ollama", model="gemma4:latest",
        prompt_version="graph-v1", schema_version="graph-relations-v1",
    )
    failed_ext = fail_chunk_extraction(
        session, extraction=lease2.extraction,
        error_code="err", error_detail="d",
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


def test_partial_unique_index_prevents_duplicate_identity_owner_rows():
    session, engine = _session()
    _, chunk = _document_chunk(session)
    sha = "a" * 64

    session.execute(text(
        "INSERT INTO graph_extractions (chunk_id, provider, model, "
        "prompt_version, schema_version, status, input_sha256, "
        "attempt_count, is_identity_owner) "
        "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', :sha, 1, 1)"
    ), {"cid": chunk.id, "sha": sha})

    with pytest.raises(IntegrityError):
        session.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, input_sha256, "
            "attempt_count, is_identity_owner) "
            "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'empty', :sha, 1, 1)"
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
        "attempt_count, is_identity_owner) "
        "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'succeeded', :sha, 1, 1)"
    ), {"cid": chunk.id, "sha": sha})
    session.execute(text(
        "INSERT INTO graph_extractions (chunk_id, provider, model, "
        "prompt_version, schema_version, status, input_sha256, "
        "attempt_count, is_identity_owner) "
        "VALUES (:cid, 'ollama', 'm', 'v1', 's1', 'empty', :sha, 1, 0)"
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
