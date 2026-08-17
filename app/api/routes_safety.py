"""Operator safety APIs and statistics (Task 10C.5).

Read-only endpoints exposing persisted safety findings and aggregates.
Requires operator authentication AND content safety enabled; either disabled
yields 404 before bearer parsing. Never returns full blocked content,
generated answers, raw queries/prompts, secrets, or auth headers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ConfigDict, Field
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.auth import require_operator
from app.config import get_settings
from app.core.db import get_db
from app.persistence import models

router = APIRouter(prefix="/safety", tags=["safety"])

_CATEGORY_ORDER = ("violence", "self_harm", "sexual_content",
                   "hate_harassment", "illegal_activity",
                   "privacy_credentials")
_ACTION_ORDER = ("allow", "warn", "filter", "block")
_SCOPE_ORDER = ("ingestion", "context", "answer")


def derive_source(source_rule_ids: list[str]) -> str:
    """Derive the finding source from rule-ID prefixes.

    All ``SAF`` IDs -> deterministic; all ``LLM_`` IDs -> llm; mixed ->
    merged. Unknown prefixes and empty lists are rejected.
    """
    if not source_rule_ids:
        raise ValueError("empty source_rule_ids")
    saf = [rid for rid in source_rule_ids if rid.startswith("SAF")]
    llm = [rid for rid in source_rule_ids if rid.startswith("LLM_")]
    if len(saf) + len(llm) != len(source_rule_ids):
        raise ValueError(f"unknown rule prefix in {source_rule_ids!r}")
    if saf and llm:
        return "merged"
    if saf:
        return "deterministic"
    return "llm"


def _parse_rfc3339_utc(value: str, *, param: str) -> datetime:
    """Parse an inclusive/exclusive UTC RFC3339-Z timestamp.

    Naive or invalid timestamps are 422; a Z suffix is required.
    """
    if not value.endswith("Z"):
        raise HTTPException(
            status_code=422,
            detail=f"{param} must be an RFC3339 UTC timestamp ending in Z",
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"{param} is not a valid RFC3339 timestamp"
        ) from None
    if parsed.tzinfo is None:
        raise HTTPException(status_code=422, detail=f"{param} must carry Z")
    return parsed.astimezone(timezone.utc)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FindingItem(BaseModel):
    id: int
    review_run_id: int
    scope: str
    document_id: int | None
    chunk_id: int | None
    document_id_snapshot: int | None
    chunk_id_snapshot: int | None
    source_deleted: bool
    retrieval_audit_id: str | None
    category: str
    severity: int
    action: str
    source: str
    source_rule_ids: list[str]
    start_offset: int
    end_offset: int
    excerpt_sha256: str
    bounded_excerpt: str | None
    policy_version: str
    created_at: str


class FindingsPage(BaseModel):
    items: list[FindingItem]
    total: int
    limit: int
    offset: int
    from_: str | None = Field(default=None, alias="from",
                               serialization_alias="from")
    to: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class RunSummary(BaseModel):
    status: str
    llm_status: str
    final_action: str | None
    failure_code: str | None
    provider: str | None
    model: str | None
    prompt_version: str | None
    schema_version: str | None
    completed_at: str | None


class FindingDetail(BaseModel):
    finding: FindingItem
    run: RunSummary


class CountRow(BaseModel):
    count: int


class PolicyVersionRow(CountRow):
    policy_version: str


class CategoryRow(CountRow):
    category: str


class ActionRow(CountRow):
    action: str


class ScopeRow(CountRow):
    scope: str


class StatsResponse(BaseModel):
    from_: str | None = Field(default=None, alias="from",
                               serialization_alias="from")
    to: str | None = None
    total_findings: int

    model_config = ConfigDict(populate_by_name=True)
    by_policy_version: list[PolicyVersionRow]
    by_category: list[CategoryRow]
    by_action: list[ActionRow]
    by_scope: list[ScopeRow]


def _require_safety_routes() -> None:
    """404 before bearer parsing unless content safety is enabled.

    Declared ahead of ``require_operator`` in each route's dependency list so
    a disabled feature flag can never surface as a 401 auth failure.
    """
    settings = get_settings()
    if not settings.content_safety_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Safety API disabled",
        )


def _finding_item(finding: models.SafetyFinding,
                  run: models.SafetyReviewRun) -> FindingItem:
    rule_ids = json.loads(finding.source_rule_ids)
    source_deleted = (
        finding.review_run.document_id is None
        and finding.review_run.document_id_snapshot is not None
    ) or (
        finding.review_run.chunk_id is None
        and finding.review_run.chunk_id_snapshot is not None
    )
    return FindingItem(
        id=finding.id,
        review_run_id=finding.review_run_id,
        scope=run.scope,
        document_id=run.document_id,
        chunk_id=run.chunk_id,
        document_id_snapshot=run.document_id_snapshot,
        chunk_id_snapshot=run.chunk_id_snapshot,
        source_deleted=source_deleted,
        retrieval_audit_id=run.retrieval_audit_id,
        category=finding.category,
        severity=finding.severity,
        action=finding.action,
        source=derive_source(rule_ids),
        source_rule_ids=rule_ids,
        start_offset=finding.start_offset,
        end_offset=finding.end_offset,
        excerpt_sha256=finding.excerpt_sha256,
        bounded_excerpt=finding.bounded_excerpt,
        policy_version=run.policy_version,
        created_at=_format_utc(run.created_at),
    )


def _filtered_findings_query(
    session: Session,
    *,
    category: str | None,
    action: str | None,
    scope: str | None,
    document_id: int | None,
    from_dt: datetime | None,
    to_dt: datetime | None,
    policy_version: str | None,
):
    query = (
        session.query(models.SafetyFinding)
        .join(models.SafetyReviewRun,
              models.SafetyFinding.review_run_id == models.SafetyReviewRun.id)
    )
    if category is not None:
        query = query.filter(models.SafetyFinding.category == category)
    if action is not None:
        query = query.filter(models.SafetyFinding.action == action)
    if scope is not None:
        query = query.filter(models.SafetyReviewRun.scope == scope)
    if document_id is not None:
        query = query.filter(
            models.SafetyReviewRun.document_id_snapshot == document_id)
    if policy_version is not None:
        query = query.filter(
            models.SafetyReviewRun.policy_version == policy_version)
    if from_dt is not None:
        query = query.filter(models.SafetyReviewRun.created_at >= from_dt)
    if to_dt is not None:
        query = query.filter(models.SafetyReviewRun.created_at < to_dt)
    return query


@router.get("/findings", response_model=FindingsPage,
            dependencies=[Depends(_require_safety_routes),
                        Depends(require_operator)])
async def list_findings(
    category: str | None = Query(default=None),
    action: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    document_id: int | None = Query(default=None),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> FindingsPage:
    from_dt = _parse_rfc3339_utc(from_, param="from") if from_ else None
    to_dt = _parse_rfc3339_utc(to, param="to") if to else None
    if from_dt and to_dt and from_dt >= to_dt:
        raise HTTPException(
            status_code=422, detail="from must be strictly before to")

    query = _filtered_findings_query(
        session, category=category, action=action, scope=scope,
        document_id=document_id, from_dt=from_dt, to_dt=to_dt,
        policy_version=None)
    total = query.count()
    rows = (
        query.order_by(
            models.SafetyReviewRun.created_at.desc(),
            models.SafetyFinding.id.desc())
        .offset(offset).limit(limit).all()
    )
    return FindingsPage(
        items=[_finding_item(f, f.review_run) for f in rows],
        total=total, limit=limit, offset=offset,
        from_=_format_utc(from_dt) if from_dt else None,
        to=_format_utc(to_dt) if to_dt else None,
    )


@router.get("/findings/{finding_id}", response_model=FindingDetail,
            dependencies=[Depends(_require_safety_routes),
                        Depends(require_operator)])
async def get_finding(
    finding_id: int,
    session: Session = Depends(get_db),
) -> FindingDetail:
    finding = session.get(models.SafetyFinding, finding_id)
    if finding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Safety finding not found",
        )
    run = finding.review_run
    return FindingDetail(
        finding=_finding_item(finding, run),
        run=RunSummary(
            status=run.status,
            llm_status=run.llm_status,
            final_action=run.final_action,
            failure_code=run.failure_code,
            provider=run.provider,
            model=run.model,
            prompt_version=run.prompt_version,
            schema_version=run.schema_version,
            completed_at=_format_utc(run.completed_at) if run.completed_at else None,
        ),
    )


@router.get("/stats", response_model=StatsResponse,
            dependencies=[Depends(_require_safety_routes),
                        Depends(require_operator)])
async def safety_stats(
    policy_version: str | None = Query(default=None),
    category: str | None = Query(default=None),
    action: str | None = Query(default=None),
    scope: str | None = Query(default=None),
    document_id: int | None = Query(default=None),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    session: Session = Depends(get_db),
) -> StatsResponse:
    from_dt = _parse_rfc3339_utc(from_, param="from") if from_ else None
    to_dt = _parse_rfc3339_utc(to, param="to") if to else None
    if from_dt and to_dt and from_dt >= to_dt:
        raise HTTPException(
            status_code=422, detail="from must be strictly before to")

    query = _filtered_findings_query(
        session, category=category, action=action, scope=scope,
        document_id=document_id, from_dt=from_dt, to_dt=to_dt,
        policy_version=policy_version)
    findings = query.all()

    by_policy: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    for finding in findings:
        run = finding.review_run
        by_policy[run.policy_version] = by_policy.get(run.policy_version, 0) + 1
        by_category[finding.category] = by_category.get(finding.category, 0) + 1
        by_action[finding.action] = by_action.get(finding.action, 0) + 1
        by_scope[run.scope] = by_scope.get(run.scope, 0) + 1

    return StatsResponse(
        from_=_format_utc(from_dt) if from_dt else None,
        to=_format_utc(to_dt) if to_dt else None,
        total_findings=len(findings),
        by_policy_version=[
            PolicyVersionRow(policy_version=v, count=by_policy[v])
            for v in sorted(by_policy)],
        by_category=[
            CategoryRow(category=c, count=by_category[c])
            for c in sorted(by_category, key=_CATEGORY_ORDER.index)],
        by_action=[
            ActionRow(action=a, count=by_action[a])
            for a in sorted(by_action, key=_ACTION_ORDER.index)],
        by_scope=[
            ScopeRow(scope=s, count=by_scope[s])
            for s in sorted(by_scope, key=_SCOPE_ORDER.index)],
    )
