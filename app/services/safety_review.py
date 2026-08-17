"""Persisted safety-review service and enforcement helpers (Task 10C.4).

``SafetyReviewService`` follows the same committed-pending then terminal
transaction contract as security audits: ``begin`` commits a durable pending
row (idempotent per review target through the d9 partial unique indexes, one
creator via savepoint, losers reload without a second detector/provider
call), the detector runs outside that transaction, and ``complete``/``fail``
terminalize conditionally. Enforcement helpers implement the answer-scope
filter transformation (strict-overlap union spans replaced highest-start
first) and the deterministic refusal/withhold messages.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.persistence import models
from app.services.content_safety import (
    SafetyAssessment,
    SafetyInputLimitError,
    classify_content,
)
from app.services.llm_safety_review import PROMPT_VERSION, RESULT_SCHEMA_VERSION
from app.services.safety_policy import SafetyPolicy

DETECTOR_VERSION = "rules-v1"
POLICY_VERSION_FALLBACK = "safety-v1"

NO_SAFE_CONTEXT_ANSWER = "No safe context was available to answer the query."
ANSWER_WITHHELD = "The generated answer was withheld by the content-safety policy."


class SafetyReviewSubsystemFailure(RuntimeError):
    """Safety review failed at any scope; the caller must fail closed (503)."""


def _run_coroutine_sync(coro):
    """Run a coroutine to completion from sync or async caller contexts."""
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: dict = {}

    def _runner():
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagate to the joining thread
            outcome["error"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class IngestionSafetyBlocked(Exception):
    """Ingestion-scope review returned block|filter; maps to HTTP 422."""

    def __init__(self, final_action: str):
        self.final_action = final_action
        super().__init__(f"ingestion safety {final_action}")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def merge_filter_spans(
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, tuple[str, ...]]]:
    """Merge strictly overlapping filter spans into union spans.

    ``spans`` are ``(start, end, category)`` half-open intervals. Sorted by
    ``(start, end, category)``; intervals merge only when they overlap
    (``max(start_a, start_b) < min(end_a, end_b)``) — adjacency keeps them
    separate. Each union span carries the sorted unique categories covering
    it.
    """
    ordered = sorted(spans, key=lambda s: (s[0], s[1], s[2]))
    merged: list[tuple[int, int, tuple[str, ...]]] = []
    for start, end, category in ordered:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_categories = merged[-1]
            merged[-1] = (
                prev_start,
                max(prev_end, end),
                tuple(sorted(set(prev_categories) | {category})),
            )
        else:
            merged.append((start, end, (category,)))
    return merged


def apply_answer_filter(
    answer: str,
    spans: list[tuple[int, int, str]],
) -> str:
    """Replace filter spans with ``[FILTERED:<cat1+cat2>]`` tags.

    Union spans are applied from the highest start to the lowest so no source
    character is replaced twice; adjacent spans remain separate.
    """
    merged = merge_filter_spans(spans)
    result = answer
    for start, end, categories in sorted(merged, key=lambda s: s[0], reverse=True):
        tag = "[FILTERED:" + "+".join(categories) + "]"
        result = result[:start] + tag + result[end:]
    return result


class SafetyReviewService:
    """Persisted safety reviews with idempotent, concurrent-safe ``begin``."""

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def begin(
        self,
        scope: str,
        *,
        input_text: str,
        policy: SafetyPolicy,
        mode: str = "disabled",
        provider=None,
        document_id: int | None = None,
        chunk_id: int | None = None,
        retrieval_audit_id: str | None = None,
        detector_version: str = DETECTOR_VERSION,
        llm_provider_name: str | None = None,
        llm_model_name: str | None = None,
        prompt_version: str | None = None,
        schema_version: str | None = None,
    ) -> models.SafetyReviewRun:
        """Create (committed pending) or reload the review run for a target.

        The creator then runs the deterministic detector (and optional
        provider review) outside the insert transaction and records findings,
        leaving the run pending for ``complete``/``fail``. A concurrent or
        repeated ``begin`` returns the existing row without a second
        detector/provider call.
        """
        input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        run = models.SafetyReviewRun(
            scope=scope,
            status="pending",
            document_id=document_id,
            chunk_id=chunk_id,
            document_id_snapshot=document_id,
            chunk_id_snapshot=chunk_id,
            retrieval_audit_id=retrieval_audit_id,
            input_sha256=input_sha,
            policy_version=policy.version,
            detector_version=detector_version,
            llm_status="skipped",
        )
        created = False
        try:
            nested = self.session.begin_nested()
            self.session.add(run)
            self.session.flush()
            nested.commit()
            created = True
        except IntegrityError:
            self.session.rollback()
            existing = self._select_target(
                scope, document_id, chunk_id, retrieval_audit_id)
            if existing is None:
                raise
            return existing

        if not created:  # pragma: no cover - defensive
            raise SafetyReviewSubsystemFailure("begin neither created nor reloaded")

        # Durable pending row before any detector/provider work.
        self.session.commit()

        # Creator path: run the review and persist findings; still pending.
        # review_with_llm is a coroutine; begin is sync and is invoked from
        # both sync (CLI/tests) and async (rag/ingestion) contexts, so the
        # coroutine runs on a dedicated thread's event loop rather than
        # asyncio.run (which cannot nest inside a running loop).
        from app.services.llm_safety_review import review_with_llm

        try:
            outcome = _run_coroutine_sync(review_with_llm(
                input_text,
                scope=scope,  # type: ignore[arg-type]
                policy=policy,
                mode=mode,
                provider=provider,
            ))
        except SafetyInputLimitError:
            # 10C.2: input over max_input_chars is rejected with the typed
            # error before classification — never silently truncated. Map it
            # to a failed review run so every scope fails closed (503)
            # instead of the error escaping as a 500 with an orphaned
            # pending run (D-64).
            run.review_failure_code = "safety_input_limit"
            run.review_llm_status = "failed"
        else:
            if hasattr(outcome, "error_code"):
                run.review_failure_code = outcome.error_code
                run.review_llm_status = "failed"
            else:
                run.review_failure_code = None
                # The 10C.3 outcome vocabulary is skipped|failed|ok; the d9
                # column check permits skipped|succeeded|failed (D-58).
                run.review_llm_status = (
                    "succeeded" if outcome.llm_status == "ok" else outcome.llm_status)
                run.review_outcome = outcome
                if provider is not None:
                    run.provider = getattr(provider, "provider_name", None) \
                        or type(provider).__name__[:50]
                    run.model = getattr(provider, "model", None)
                    run.prompt_version = PROMPT_VERSION
                    run.schema_version = RESULT_SCHEMA_VERSION
                self._record_findings(run.id, outcome.findings, input_text)
                self.session.commit()
        return run

    def complete(
        self,
        run: models.SafetyReviewRun,
        *,
        final_action: str,
        llm_status: str = "skipped",
    ) -> models.SafetyReviewRun:
        """Transition pending -> succeeded (conditional, terminal)."""
        from sqlalchemy import update

        result = self.session.execute(
            update(models.SafetyReviewRun)
            .where(
                models.SafetyReviewRun.id == run.id,
                models.SafetyReviewRun.status == "pending",
            )
            .values(
                status="succeeded",
                final_action=final_action[:10],
                llm_status=llm_status[:20],
                failure_code=None,
                completed_at=_now_utc(),
            )
        )
        self.session.commit()
        if result.rowcount == 0:
            self.session.rollback()
            current = self.session.get(models.SafetyReviewRun, run.id)
            if current is None:
                raise SafetyReviewSubsystemFailure(
                    f"safety run {run.id} disappeared during complete")
            return current
        self.session.refresh(run)
        return run

    def fail(
        self,
        run: models.SafetyReviewRun,
        *,
        failure_code: str,
        llm_status: str = "skipped",
    ) -> models.SafetyReviewRun:
        """Transition pending -> failed (conditional, terminal)."""
        from sqlalchemy import update

        result = self.session.execute(
            update(models.SafetyReviewRun)
            .where(
                models.SafetyReviewRun.id == run.id,
                models.SafetyReviewRun.status == "pending",
            )
            .values(
                status="failed",
                final_action=None,
                llm_status=llm_status[:20],
                failure_code=failure_code[:100],
                completed_at=_now_utc(),
            )
        )
        self.session.commit()
        if result.rowcount == 0:
            self.session.rollback()
            current = self.session.get(models.SafetyReviewRun, run.id)
            if current is None:
                raise SafetyReviewSubsystemFailure(
                    f"safety run {run.id} disappeared during fail")
            return current
        self.session.refresh(run)
        return run

    # ------------------------------------------------------------------
    # Scope-specific enforcement entry points
    # ------------------------------------------------------------------

    def review_ingestion(
        self, *, document, text: str, policy: SafetyPolicy, mode: str,
        provider=None,
    ) -> models.SafetyReviewRun:
        """Ingestion-scope review for a staged document (idempotent)."""
        run = self.begin(
            "ingestion", input_text=text, policy=policy, mode=mode,
            provider=provider, document_id=document.id,
        )
        return self._finish_review(run)

    def _finish_review(self, run: models.SafetyReviewRun) -> models.SafetyReviewRun:
        """Terminalize a freshly created review run (D-59: a reloaded
        still-pending row — a concurrent creator owns it — fails typed)."""
        if run.status != "pending":
            return run  # terminal from a prior attempt; caller honors it
        failure = getattr(run, "review_failure_code", None)
        if failure is not None:
            return self.fail(run, failure_code=failure, llm_status="failed")
        outcome = getattr(run, "review_outcome", None)
        if outcome is None:
            raise SafetyReviewSubsystemFailure(
                f"safety run {run.id} is pending under a concurrent creator")
        return self.complete(
            run,
            final_action=outcome.overall_action,
            llm_status=run.review_llm_status,
        )

    def review_context(
        self, *, chunk, document_id: int | None, retrieval_audit_id: str,
        text: str, policy: SafetyPolicy, mode: str, provider=None,
    ) -> models.SafetyReviewRun:
        """Context-scope review for one hydrated chunk (idempotent)."""
        run = self.begin(
            "context", input_text=text, policy=policy, mode=mode,
            provider=provider, document_id=document_id, chunk_id=chunk.id,
            retrieval_audit_id=retrieval_audit_id,
        )
        return self._finish_review(run)

    def review_answer(
        self, *, retrieval_audit_id: str, text: str, policy: SafetyPolicy,
        mode: str, provider=None,
    ) -> models.SafetyReviewRun:
        """Answer-scope review of the generated answer (idempotent)."""
        run = self.begin(
            "answer", input_text=text, policy=policy, mode=mode,
            provider=provider, retrieval_audit_id=retrieval_audit_id,
        )
        return self._finish_review(run)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_target(
        self,
        scope: str,
        document_id: int | None,
        chunk_id: int | None,
        retrieval_audit_id: str | None,
    ) -> models.SafetyReviewRun | None:
        query = self.session.query(models.SafetyReviewRun).filter(
            models.SafetyReviewRun.scope == scope
        )
        if scope == "ingestion":
            query = query.filter(
                models.SafetyReviewRun.document_id_snapshot == document_id)
        elif scope == "context":
            query = query.filter(
                models.SafetyReviewRun.retrieval_audit_id == retrieval_audit_id,
                models.SafetyReviewRun.chunk_id_snapshot == chunk_id,
            )
        elif scope == "answer":
            query = query.filter(
                models.SafetyReviewRun.retrieval_audit_id == retrieval_audit_id)
        else:  # pragma: no cover - closed scope set
            raise ValueError(f"unknown scope {scope!r}")
        return query.order_by(models.SafetyReviewRun.id).first()

    def _record_findings(
        self, run_id: int, findings, input_text: str,
    ) -> None:
        for finding in findings:
            excerpt = input_text[finding.start:finding.end]
            self.session.add(models.SafetyFinding(
                review_run_id=run_id,
                category=finding.category,
                severity=finding.severity,
                action=finding.action,
                start_offset=finding.start,
                end_offset=finding.end,
                source_rule_ids=json.dumps(
                    sorted(set(finding.source_rule_ids)),
                    separators=(",", ":"), ensure_ascii=True),
                excerpt_sha256=hashlib.sha256(
                    excerpt.encode("utf-8")).hexdigest(),
                bounded_excerpt=None,  # NULL for all untrusted text in v1
            ))
