"""harden graph extraction lifecycle (idempotent identity, attempts, lease, skip)

Revision ID: b7f3d5a9c2e1
Revises: a6e2c4f8b1d9
Create Date: 2026-08-09

This revision is owned by Task 10A.3. It is a deterministic SQLite table
rebuild that:

* adds ``media_type`` to ``chunks`` (default ``text/plain``);
* adds the lifecycle/identity columns to ``graph_extractions``
  (``input_sha256``, ``attempt_count``, ``completed_at``,
  ``attempt_started_at``, ``is_identity_owner``), adds ``skipped`` to the
  status CHECK, and creates the partial unique index
  ``uq_graph_extractions_identity_owner`` over the identity tuple where
  ``is_identity_owner = 1``.

The predecessor ``a6e2c4f8b1d9`` schema has no unique constraint/index on the
graph-extraction identity columns (only the non-unique
``ix_graph_extractions_id``), so this revision creates the partial unique index
directly; there is no predecessor unique object to drop or replace.

Existing rows are backfilled deterministically per the predecessor conversion
matrix in the plan (status precedence + ``completed_at``/``id`` tie-break for
owner selection, SHA-256 of authoritative ``chunks.text`` for ``input_sha256``).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7f3d5a9c2e1"
down_revision: Union[str, None] = "a6e2c4f8b1d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add chunk media type.
    with op.batch_alter_table("chunks") as batch_op:
        batch_op.add_column(
            sa.Column(
                "media_type",
                sa.String(length=100),
                nullable=False,
                server_default="text/plain",
            )
        )

    # 2. Add the new lifecycle/identity columns with safe defaults so the
    #    backfill UPDATE can set their real values before the CHECKs are added.
    with op.batch_alter_table("graph_extractions") as batch_op:
        batch_op.add_column(
            sa.Column("input_sha256", sa.String(length=64), nullable=False, server_default="0" * 64)
        )
        batch_op.add_column(
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("attempt_started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(
            sa.Column(
                "is_identity_owner",
                sa.Boolean(),
                nullable=False,
                server_default="1",
            )
        )

    # 3. Backfill input_sha256 from authoritative chunks.text (lowercase SHA-256
    #    of the exact UTF-8 bytes). SQLite's lower(hex(sha256(...))) is not
    #    available portably, so compute via a Python-driven update.
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT ge.id AS ge_id, ge.chunk_id AS chunk_id, ge.status AS status, "
            "ge.created_at AS created_at, ge.error_code AS error_code "
            "FROM graph_extractions ge"
        )
    ).fetchall()
    import hashlib

    for row in rows:
        chunk_text = bind.execute(
            sa.text("SELECT text FROM chunks WHERE id = :cid"), {"cid": row.chunk_id}
        ).scalar()
        payload = (chunk_text or "").encode("utf-8")
        input_sha = hashlib.sha256(payload).hexdigest()
        status = row.status
        created = row.created_at
        if status in ("succeeded", "empty"):
            completed_at = created
            attempt_started_at = created
            error_code = None
            error_detail = None
        elif status == "failed":
            completed_at = created
            attempt_started_at = created
            if row.error_code:
                error_code = row.error_code
                error_detail = row.error_detail if hasattr(row, 'error_detail') else None
            else:
                error_code = "predecessor_failed"
                error_detail = "unknown pre-b7 failure"
        elif status == "pending":
            completed_at = None
            attempt_started_at = created
            error_code = None
            error_detail = None
        else:
            # Unknown predecessor status — treat as terminal succeeded defensively.
            completed_at = created
            attempt_started_at = created
            error_code = None
            error_detail = None
        bind.execute(
            sa.text(
                "UPDATE graph_extractions SET input_sha256 = :sha, "
                "attempt_count = 1, completed_at = :comp, "
                "attempt_started_at = :started, error_code = :ec, "
                "error_detail = :ed "
                "WHERE id = :gid"
            ),
            {
                "sha": input_sha,
                "comp": completed_at,
                "started": attempt_started_at,
                "ec": error_code,
                "ed": error_detail,
                "gid": row.ge_id,
            },
        )

    # 4. Owner selection: for each identity collision group, keep one owner by
    #    status precedence (succeeded > empty > failed > pending), then
    #    completed_at DESC (NULL last), then id ASC. Mark the rest non-owner.
    _select_identity_owners(bind)

    # 5. Replace the status CHECK to include 'skipped' and add the remaining
    #    CHECKs, then create the partial unique identity-owner index.
    with op.batch_alter_table("graph_extractions") as batch_op:
        batch_op.drop_constraint("ck_graph_extractions_status", type_="check")
        batch_op.create_check_constraint(
            "ck_graph_extractions_status",
            "status IN ('pending', 'succeeded', 'failed', 'empty', 'skipped')",
        )
        batch_op.create_check_constraint(
            "ck_graph_extractions_attempt_count", "attempt_count >= 0"
        )
        batch_op.create_check_constraint(
            "ck_graph_extractions_is_identity_owner", "is_identity_owner IN (0, 1)"
        )
        batch_op.create_check_constraint(
            "ck_graph_extractions_input_sha256_hex",
            "length(input_sha256) = 64 AND input_sha256 NOT GLOB '*[^0-9a-f]*'",
        )

    op.create_index(
        "uq_graph_extractions_identity_owner",
        "graph_extractions",
        ["chunk_id", "provider", "model", "prompt_version", "schema_version", "input_sha256"],
        unique=True,
        sqlite_where=sa.text("is_identity_owner = 1"),
    )


def _select_identity_owners(bind) -> None:
    """Mark one owner per identity collision group; the rest become non-owner.

    Owner precedence: status (succeeded > empty > failed > pending), then
    ``completed_at`` DESC with NULL last, then ``id`` ASC. Deterministic.
    """
    import datetime as _dt

    precedence = {"succeeded": 0, "empty": 1, "failed": 2, "pending": 3}
    rows = bind.execute(
        sa.text(
            "SELECT id, chunk_id, provider, model, prompt_version, schema_version, "
            "input_sha256, status, completed_at FROM graph_extractions"
        )
    ).fetchall()

    grouped: dict[tuple, list] = {}
    for row in rows:
        key = (
            row.chunk_id,
            row.provider,
            row.model,
            row.prompt_version,
            row.schema_version,
            row.input_sha256,
        )
        grouped.setdefault(key, []).append(row)

    def _epoch(ts) -> float:
        if ts is None:
            return float("-inf")  # NULL sorts last under DESC
        if isinstance(ts, _dt.datetime):
            return ts.timestamp()
        return float("-inf")

    for members in grouped.values():
        owner = min(
            members,
            key=lambda r: (
                precedence.get(r.status, 9),
                0 if r.completed_at is not None else 1,
                -_epoch(r.completed_at),  # DESC: larger epoch -> smaller key
                r.id,
            ),
        )
        for m in members:
            if m.id != owner.id:
                bind.execute(
                    sa.text(
                        "UPDATE graph_extractions SET is_identity_owner = 0 WHERE id = :gid"
                    ),
                    {"gid": m.id},
                )


def downgrade() -> None:
    """Downgrade to the a6e2c4f8b1d9 schema.

    Refuses (via the predecessor CHECK) to proceed while any ``skipped`` row
    exists, because the predecessor ``status`` CHECK cannot represent it. After
    the operator removes only explicitly selected skipped rows, this downgrade
    restores the predecessor table/checks and preserves every representable
    row/ID.
    """
    bind = op.get_bind()
    skipped = bind.execute(
        sa.text("SELECT COUNT(*) FROM graph_extractions WHERE status = 'skipped'")
    ).scalar()
    if skipped:
        raise RuntimeError(
            "downgrade_skipped_rows_present: cannot downgrade b7f3d5a9c2e1 while "
            f"{skipped} skipped row(s) exist; export/delete them via "
            "scripts/export_unrepresentable_rows.py --revision b7f3d5a9c2e1 first"
        )

    op.drop_index("uq_graph_extractions_identity_owner", table_name="graph_extractions")

    with op.batch_alter_table("graph_extractions") as batch_op:
        batch_op.drop_constraint("ck_graph_extractions_input_sha256_hex", type_="check")
        batch_op.drop_constraint("ck_graph_extractions_is_identity_owner", type_="check")
        batch_op.drop_constraint("ck_graph_extractions_attempt_count", type_="check")
        batch_op.drop_constraint("ck_graph_extractions_status", type_="check")
        batch_op.create_check_constraint(
            "ck_graph_extractions_status",
            "status IN ('pending', 'succeeded', 'failed', 'empty')",
        )
        batch_op.drop_column("is_identity_owner")
        batch_op.drop_column("attempt_started_at")
        batch_op.drop_column("completed_at")
        batch_op.drop_column("attempt_count")
        batch_op.drop_column("input_sha256")

    with op.batch_alter_table("chunks") as batch_op:
        batch_op.drop_column("media_type")
