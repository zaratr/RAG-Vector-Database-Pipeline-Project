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
    svc.replace_decision(audit_id, 1, "rejected_safety", '["context_safety"]')
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
