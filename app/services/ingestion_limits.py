"""Bounded ingestion controls (Task 10B.3).

Provides request envelope body counting (BoundedReceiveMiddleware) and
fixed-window rate limiting via atomic SQL ``INSERT ... ON CONFLICT ... DO UPDATE``
in ``BEGIN IMMEDIATE`` with opportunistic prune.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings

RATE_WINDOW_SECONDS = 60


def _rate_identity(operator_token: str | None, remote_host: str | None) -> str:
    """Derive the raw rate-limit identity key (NOT pre-hashed).

    Returns ``operator:<token>`` or ``client:<host>`` — the caller hashes once.
    """
    if operator_token:
        return f"operator:{operator_token}"
    if remote_host:
        return f"client:{remote_host}"
    return "client:anonymous"


def acquire_slot(
    session: Session,
    *,
    identity: str,
    limit: int,
    window_seconds: int = RATE_WINDOW_SECONDS,
    clock: Callable[[], int] | None = None,
) -> tuple[bool, int, dict[str, str]]:
    """Atomic fixed-window rate check using INSERT ... ON CONFLICT ... DO UPDATE
    in ``BEGIN IMMEDIATE``.

    Uses a raw DBAPI connection to avoid conflicts with the session's implicit
    transaction. Returns ``(allowed, request_count, headers)``. Always persists
    the count (even for rejected requests). Opportunistically prunes expired buckets.
    """
    epoch = clock() if clock is not None else int(time.time())
    window_start = epoch - (epoch % window_seconds)

    # Hash identity once.
    identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    # Get a raw DBAPI connection for BEGIN IMMEDIATE control.
    engine = session.get_bind()
    raw_conn = engine.raw_connection()
    try:
        # Opportunistic prune of buckets older than two windows, capped at 1000.
        # DELETE ... LIMIT is not portable SQLite; select bounded rowids instead.
        prune_cutoff = window_start - 2 * window_seconds
        raw_conn.execute(
            "DELETE FROM ingestion_rate_buckets WHERE rowid IN ("
            "SELECT rowid FROM ingestion_rate_buckets "
            "WHERE window_start_epoch < ? LIMIT 1000)",
            (prune_cutoff,),
        )
        raw_conn.commit()

        # Atomic upsert in BEGIN IMMEDIATE for serializable write.
        raw_conn.execute("BEGIN IMMEDIATE")
        cursor = raw_conn.cursor()
        cursor.execute(
            "INSERT INTO ingestion_rate_buckets (identity_sha256, window_start_epoch, request_count) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT (identity_sha256, window_start_epoch) "
            "DO UPDATE SET request_count = request_count + 1 "
            "RETURNING request_count",
            (identity_hash, window_start),
        )
        row = cursor.fetchone()
        count = row[0] if row else 1
        raw_conn.execute("COMMIT")
    except Exception:
        try:
            raw_conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        raw_conn.close()

    allowed = count <= limit
    remaining = max(0, limit - count)
    reset_epoch = window_start + window_seconds
    retry_after = max(1, reset_epoch - epoch)
    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_epoch),
        "Retry-After": str(retry_after),
    }
    return allowed, count, headers


def check_rate_limit(
    session: Session,
    *,
    identity: str,
    limit: int,
    window_seconds: int = RATE_WINDOW_SECONDS,
    now_epoch: Optional[int] = None,
) -> tuple[bool, int, dict[str, str]]:
    """Backward-compatible wrapper around acquire_slot with now_epoch clock."""
    clock = (lambda: now_epoch) if now_epoch is not None else None
    return acquire_slot(
        session, identity=identity, limit=limit,
        window_seconds=window_seconds, clock=clock,
    )


def check_rate_limit_http(
    session: Session,
    *,
    operator_token: str | None = None,
    remote_host: str | None = None,
    limit: Optional[int] = None,
    window_seconds: Optional[int] = None,
) -> tuple[bool, dict[str, str]]:
    """HTTP-friendly wrapper that derives identity from operator token or client host."""
    settings = get_settings()
    if limit is None:
        limit = settings.ingest_rate_limit_requests
    if window_seconds is None:
        window_seconds = settings.ingest_rate_limit_window_seconds
    identity = _rate_identity(operator_token, remote_host)
    allowed, _, headers = acquire_slot(
        session, identity=identity, limit=limit, window_seconds=window_seconds,
    )
    return allowed, headers
