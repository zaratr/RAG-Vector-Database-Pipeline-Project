"""Operator security audit API (Task 10B.5).

Read-only endpoints exposing persisted retrieval audit decisions. Requires
operator authentication. Never returns raw queries, prompts, credentials,
or full document text.
"""
from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.auth import require_operator
from app.core.db import get_db
from app.persistence import models

router = APIRouter(prefix="/security", tags=["security"])


class AuditDecisionResponse(BaseModel):
    document_id: int | None
    chunk_id: int | None
    decision: str
    native_score: float | None
    provenance_score: float
    reason_codes: list[str]
    content_sha256: str
    excerpt: str | None


class AuditResponse(BaseModel):
    id: str
    query_sha256: str
    mode: str
    status: str
    policy_versions: dict[str, str]
    counts: dict[str, int]
    failure_code: str | None
    created_at: str
    completed_at: str | None
    decisions: list[AuditDecisionResponse]


def _format_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@router.get("/audits/{audit_id}", response_model=AuditResponse)
async def get_audit(
    audit_id: str,
    session: Session = Depends(get_db),
    _operator: bool = Depends(require_operator),
) -> AuditResponse:
    audit = session.get(models.RetrievalAudit, audit_id)
    if audit is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audit not found")

    decisions = (
        session.query(models.RetrievalCandidateDecision)
        .filter(models.RetrievalCandidateDecision.audit_id == audit_id)
        .order_by(models.RetrievalCandidateDecision.chunk_id_snapshot)
        .all()
    )

    return AuditResponse(
        id=audit.id,
        query_sha256=audit.query_sha256,
        mode=audit.retrieval_mode,
        status=audit.status,
        policy_versions={
            "provenance": audit.provenance_policy_version,
            "retrieval": audit.retrieval_policy_version,
            "context": audit.context_policy_version,
        },
        counts={
            "candidates": audit.candidate_count,
            "selected": audit.selected_count,
            "rejected": audit.rejected_count,
        },
        failure_code=audit.failure_code,
        created_at=_format_dt(audit.created_at),
        completed_at=_format_dt(audit.completed_at),
        decisions=[
            AuditDecisionResponse(
                document_id=d.document_id,
                chunk_id=d.chunk_id,
                decision=d.decision,
                native_score=d.native_score,
                provenance_score=d.provenance_score,
                # Storage is canonical sorted JSON text; the API returns the
                # decoded array per the plan payload shape.
                reason_codes=json.loads(d.reason_codes),
                content_sha256=d.content_sha256,
                excerpt=d.bounded_excerpt,
            )
            for d in decisions
        ],
    )
