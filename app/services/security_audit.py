"""Security audit lifecycle service (Task 10B.2).

Tracks the end-to-end ``/query`` lifecycle: begin (pending) → record decisions
→ complete (completed) or fail (failed). Version-gated completion (B-09): at
``c8...`` head with safety disabled, generation + final decisions suffice; at
``d9...`` head with safety enabled, an answer-safety review is also required.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.persistence import models


class InvalidAuditTransition(RuntimeError):
    """Raised when a terminal audit is mutated or terminalized again."""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class SecurityAuditService:
    def __init__(self, session: Session):
        self.session = session

    def begin(
        self,
        query: str,
        mode: str,
        policy_versions: dict[str, str],
    ) -> str:
        """Create a pending audit and return its UUID."""
        audit_id = str(uuid.uuid4())
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        audit = models.RetrievalAudit(
            id=audit_id,
            query_sha256=query_hash,
            retrieval_mode=mode,
            status="pending",
            provenance_policy_version=policy_versions.get("provenance", "unassigned"),
            retrieval_policy_version=policy_versions.get("retrieval", "unassigned"),
            context_policy_version=policy_versions.get("context", "unassigned"),
            candidate_count=0,
            selected_count=0,
            rejected_count=0,
            failure_code=None,
            completed_at=None,
        )
        self.session.add(audit)
        self.session.flush()
        return audit_id

    def record_decisions(self, audit_id: str, decisions: list[dict]) -> None:
        """Atomically upsert candidate decisions and recalculate counters."""
        audit = self.session.get(models.RetrievalAudit, audit_id)
        if audit is None:
            raise ValueError(f"audit {audit_id} not found")
        if audit.status != "pending":
            raise InvalidAuditTransition(
                f"audit {audit_id} is terminal ({audit.status})"
            )
        # Replace all decisions for this audit.
        self.session.query(models.RetrievalCandidateDecision).filter(
            models.RetrievalCandidateDecision.audit_id == audit_id
        ).delete(synchronize_session=False)

        selected = 0
        rejected = 0
        for dec in decisions:
            decision_val = dec["decision"]
            if decision_val in ("selected", "allow"):
                selected += 1
            else:
                rejected += 1
            self.session.add(
                models.RetrievalCandidateDecision(
                    audit_id=audit_id,
                    document_id=dec.get("document_id"),
                    chunk_id=dec.get("chunk_id"),
                    document_id_snapshot=dec["document_id_snapshot"],
                    chunk_id_snapshot=dec["chunk_id_snapshot"],
                    decision=decision_val,
                    native_score=dec.get("native_score"),
                    provenance_score=dec["provenance_score"],
                    reason_codes=dec["reason_codes"],
                    content_sha256=dec["content_sha256"],
                    bounded_excerpt=None,  # NULL for all untrusted text in v1
                )
            )

        audit.candidate_count = selected + rejected
        audit.selected_count = selected
        audit.rejected_count = rejected
        self.session.flush()

    def replace_decision(
        self, audit_id: str, chunk_id: int, final_decision: str, reasons: str
    ) -> None:
        """Update a single candidate decision while the audit is pending."""
        audit = self.session.get(models.RetrievalAudit, audit_id)
        if audit is None:
            raise ValueError(f"audit {audit_id} not found")
        if audit.status != "pending":
            raise InvalidAuditTransition(
                f"audit {audit_id} is terminal ({audit.status})"
            )
        decision = (
            self.session.query(models.RetrievalCandidateDecision)
            .filter(
                models.RetrievalCandidateDecision.audit_id == audit_id,
                models.RetrievalCandidateDecision.chunk_id_snapshot == chunk_id,
            )
            .one_or_none()
        )
        if decision is None:
            raise ValueError(f"decision for chunk {chunk_id} not found in audit {audit_id}")
        old_decision = decision.decision
        decision.decision = final_decision
        decision.reason_codes = reasons
        # Recalculate counters.
        all_decisions = (
            self.session.query(models.RetrievalCandidateDecision)
            .filter(models.RetrievalCandidateDecision.audit_id == audit_id)
            .all()
        )
        selected = sum(1 for d in all_decisions if d.decision in ("selected", "allow"))
        rejected = sum(1 for d in all_decisions if d.decision not in ("selected", "allow"))
        audit.selected_count = selected
        audit.rejected_count = rejected
        audit.candidate_count = selected + rejected
        self.session.flush()

    def complete(self, audit_id: str) -> None:
        """Transition pending → completed (version-gated per B-09).

        At c8 head (safety disabled): generation + final candidate decisions suffice.
        At d9 head (safety enabled): answer-safety review required (checked by caller).
        The transition guard itself is unconditional on status=pending; the
        version-gating is the caller's responsibility (the /query handler checks
        the feature flag before calling complete).
        """
        result = self.session.execute(
            update(models.RetrievalAudit)
            .where(
                models.RetrievalAudit.id == audit_id,
                models.RetrievalAudit.status == "pending",
            )
            .values(
                status="completed",
                completed_at=_now_utc(),
                failure_code=None,
            )
        )
        if result.rowcount == 0:
            current = self.session.get(models.RetrievalAudit, audit_id)
            if current is None:
                raise ValueError(f"audit {audit_id} not found")
            if current.status != "pending":
                raise InvalidAuditTransition(
                    f"audit {audit_id} is terminal ({current.status})"
                )

    def fail(self, audit_id: str, failure_code: str) -> None:
        """Transition pending → failed."""
        result = self.session.execute(
            update(models.RetrievalAudit)
            .where(
                models.RetrievalAudit.id == audit_id,
                models.RetrievalAudit.status == "pending",
            )
            .values(
                status="failed",
                completed_at=_now_utc(),
                failure_code=failure_code[:100],
            )
        )
        if result.rowcount == 0:
            current = self.session.get(models.RetrievalAudit, audit_id)
            if current is None:
                raise ValueError(f"audit {audit_id} not found")
            if current.status != "pending":
                raise InvalidAuditTransition(
                    f"audit {audit_id} is terminal ({current.status})"
                )
