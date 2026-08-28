"""Phase 10B.2 — security audit lifecycle tests."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.persistence import models
from app.services.security_audit import (
    InvalidAuditTransition,
    SecurityAuditService,
)


def _session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def test_begin_creates_pending_audit():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("test query", "vector", {"provenance": "v1"})
    session.commit()
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.status == "pending"
    assert audit.retrieval_mode == "vector"
    assert audit.candidate_count == 0
    assert audit.completed_at is None
    assert audit.failure_code is None
    session.close(); engine.dispose()


def test_record_decisions_sets_counters():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("q", "vector", {})
    svc.record_decisions(audit_id, [
        {"document_id_snapshot": 1, "chunk_id_snapshot": 1, "decision": "selected",
         "provenance_score": 0.5, "reason_codes": "[]", "content_sha256": "a" * 64},
        {"document_id_snapshot": 2, "chunk_id_snapshot": 2, "decision": "rejected_duplicate",
         "provenance_score": 0.3, "reason_codes": "[]", "content_sha256": "b" * 64},
    ])
    session.commit()
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.candidate_count == 2
    assert audit.selected_count == 1
    assert audit.rejected_count == 1
    assert audit.candidate_count == audit.selected_count + audit.rejected_count
    session.close(); engine.dispose()


def test_complete_transitions_to_completed():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("q", "vector", {})
    svc.complete(audit_id)
    session.commit()
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.status == "completed"
    assert audit.completed_at is not None
    assert audit.failure_code is None
    session.close(); engine.dispose()


def test_fail_transitions_to_failed():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("q", "vector", {})
    svc.fail(audit_id, "provider_error")
    session.commit()
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.status == "failed"
    assert audit.completed_at is not None
    assert audit.failure_code == "provider_error"
    session.close(); engine.dispose()


def test_complete_on_completed_raises():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("q", "vector", {})
    svc.complete(audit_id)
    session.commit()
    with pytest.raises(InvalidAuditTransition):
        svc.complete(audit_id)
    session.rollback()
    session.close(); engine.dispose()


def test_fail_on_failed_raises():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("q", "vector", {})
    svc.fail(audit_id, "error")
    session.commit()
    with pytest.raises(InvalidAuditTransition):
        svc.fail(audit_id, "error2")
    session.rollback()
    session.close(); engine.dispose()


def test_replace_decision_while_pending():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("q", "vector", {})
    svc.record_decisions(audit_id, [
        {"document_id_snapshot": 1, "chunk_id_snapshot": 1, "decision": "selected",
         "provenance_score": 0.5, "reason_codes": "[]", "content_sha256": "a" * 64},
    ])
    # The replacement lane's production value is ``rejected_injection`` (the
    # 10B.3 decision enum is physically enforced by the
    # ck_candidate_decisions_decision CHECK; the earlier ``rejected_safety``
    # literal was never a legal decision value).
    svc.replace_decision(audit_id, 1, "rejected_injection", '["CTX001_instruction_override"]')
    session.commit()
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.selected_count == 0
    assert audit.rejected_count == 1
    session.close(); engine.dispose()


def test_replace_decision_on_terminal_raises():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("q", "vector", {})
    svc.complete(audit_id)
    session.commit()
    with pytest.raises(InvalidAuditTransition):
        svc.replace_decision(audit_id, 1, "rejected", "[]")
    session.rollback()
    session.close(); engine.dispose()


# ---------------------------------------------------------------------------
# Appendix 10B.2 additions: failure preservation, evidence durability under
# document deletion, cascade scope, and raw-query hash distinctness.

def test_fail_transitions_pending_to_failed_and_preserves_decisions():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("failure lane query", "hybrid", {"provenance": "v1"})
    svc.record_decisions(audit_id, [
        {"document_id_snapshot": 1, "chunk_id_snapshot": 1, "decision": "selected",
         "provenance_score": 0.6, "reason_codes": "[]", "content_sha256": "3" * 64},
    ])
    svc.fail(audit_id, "provider_unavailable")
    session.commit()
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.status == "failed"
    assert audit.failure_code == "provider_unavailable"
    assert audit.completed_at is not None
    # Persisted candidate decisions and counters survive the failure.
    assert audit.candidate_count == 1
    assert audit.selected_count == 1
    decisions = session.query(models.RetrievalCandidateDecision).filter_by(
        audit_id=audit_id).all()
    assert len(decisions) == 1
    assert decisions[0].decision == "selected"
    assert decisions[0].chunk_id_snapshot == 1
    session.close(); engine.dispose()


def test_document_deletion_does_not_erase_audit_evidence():
    """Deleting the source document through the ORM (production deletion
    path) nulls only the live FKs; snapshots, hashes, and counters remain."""
    session, engine = _session()
    svc = SecurityAuditService(session)
    doc = models.Document(title="Audit del", source="unit", ingestion_status="ready")
    session.add(doc)
    session.flush()
    chunk = models.Chunk(document_id=doc.id, index=0, text="hello",
                         start_offset=0, end_offset=5, vector_id="v1")
    session.add(chunk)
    session.flush()
    audit_id = svc.begin("audit deletion query", "hybrid", {"provenance": "v1"})
    content_sha = "8" * 64
    svc.record_decisions(audit_id, [
        {"chunk_id": chunk.id, "document_id": doc.id,
         "document_id_snapshot": doc.id, "chunk_id_snapshot": chunk.id,
         "decision": "selected", "native_score": None,
         "provenance_score": 0.5, "reason_codes": "[]",
         "content_sha256": content_sha},
    ])
    svc.complete(audit_id)

    session.delete(doc)
    session.commit()

    decisions = session.query(models.RetrievalCandidateDecision).filter_by(
        audit_id=audit_id).all()
    assert len(decisions) == 1
    survivor = decisions[0]
    assert survivor.document_id is None       # live FK nulled (SET NULL)
    assert survivor.chunk_id is None
    assert survivor.document_id_snapshot == doc.id   # snapshot retained
    assert survivor.chunk_id_snapshot == chunk.id
    assert survivor.content_sha256 == content_sha
    assert survivor.decision == "selected"
    # Counters still derivable and equal.
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.candidate_count == 1
    assert audit.selected_count == 1
    assert audit.candidate_count == audit.selected_count + audit.rejected_count
    session.close(); engine.dispose()


def test_only_deleting_audit_cascades_its_decisions():
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin("cascade scope query", "vector", {"provenance": "v1"})
    svc.record_decisions(audit_id, [
        {"document_id_snapshot": 1, "chunk_id_snapshot": 1, "decision": "selected",
         "provenance_score": 0.5, "reason_codes": "[]", "content_sha256": "9" * 64},
    ])
    svc.complete(audit_id)
    session.commit()

    audit = session.get(models.RetrievalAudit, audit_id)
    session.delete(audit)
    session.commit()

    assert session.query(models.RetrievalCandidateDecision).filter_by(
        audit_id=audit_id).count() == 0
    session.close(); engine.dispose()


def test_query_sha256_distinct_for_whitespace_case_unicode_differences():
    """No normalization before hashing: queries differing only by
    whitespace, case, or Unicode normalization form hash distinctly."""
    import hashlib

    from app.services.security_audit import SecurityAuditService as _Svc

    q1 = "Who manages Helios?"
    q2 = "who manages helios?"
    q3 = "Who  manages Helios?"   # double space
    q4 = "Ｗho manages Helios?"   # full-width W (NFKC expands)
    hashes = [_Svc._query_sha256(q) if hasattr(_Svc, "_query_sha256")
              else hashlib.sha256(q.encode("utf-8")).hexdigest()
              for q in (q1, q2, q3, q4)]
    assert len(set(hashes)) == 4
    for h in hashes:
        assert len(h) == 64

    # The service persists exactly this hash of the exact UTF-8 bytes.
    session, engine = _session()
    svc = SecurityAuditService(session)
    audit_id = svc.begin(q1, "vector", {})
    session.commit()
    audit = session.get(models.RetrievalAudit, audit_id)
    assert audit.query_sha256 == hashlib.sha256(q1.encode("utf-8")).hexdigest()
    # The raw query is never a column on the audit.
    assert not any("query" == c.name for c in models.RetrievalAudit.__table__.columns
                   if c.name != "query_sha256")
    session.close(); engine.dispose()
