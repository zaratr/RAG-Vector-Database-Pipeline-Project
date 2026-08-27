"""harden graph extraction lifecycle (idempotent identity, attempts, lease, skip)

Revision ID: b7f3d5a9c2e1
Revises: a6e2c4f8b1d9
Create Date: 2026-08-09

The upgrade is a deterministic SQLite
table rebuild of ``graph_extractions``:

* adds ``media_type`` to ``chunks`` (default ``text/plain``);
* creates the staging table ``_b7_new`` with the literal target schema,
  ``INSERT ... SELECT`` every column ordered by PK with the computed
  lifecycle/identity columns set during the copy (per the predecessor
  conversion matrix), asserts source/destination row counts, canonical PK
  sets and per-row content hashes, per-status conversion counts, and the
  synthetic-error patch count;
* drops the predecessor table, renames the staging table and recreates
  every index;
* ``PRAGMA foreign_key_check`` and an exact schema snapshot of the rebuilt
  table must pass before commit.

The predecessor ``a6e2c4f8b1d9`` schema has no unique constraint/index on the
graph-extraction identity columns (only the non-unique
``ix_graph_extractions_id``), so this revision creates the partial unique index
``uq_graph_extractions_identity_owner`` directly; there is no predecessor
unique object to drop or replace.

Rows whose ``status`` is not one of the predecessor statuses fail the upgrade
explicitly (no silent coercion). Legacy collision groups (same identity tuple
once ``input_sha256`` is computed) keep exactly one owner by status precedence
``succeeded > empty > failed > pending``, then ``completed_at`` DESC with NULL
last, then ``id`` ASC; all other rows become immutable non-owner provenance.

The downgrade refuses with ``downgrade_skipped_rows_present`` while any
``skipped`` row exists (the predecessor status CHECK cannot represent it);
after the operator removes explicitly selected skipped rows, the downgrade
rebuilds the exact predecessor table/checks and preserves every representable
row/ID.
"""
from __future__ import annotations

import datetime as _dt
import functools
import hashlib
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7f3d5a9c2e1"
down_revision: Union[str, None] = "a6e2c4f8b1d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Contract constants (mirrored in app.persistence.models.GraphExtraction)
# ---------------------------------------------------------------------------

_PREDECESSOR_STATUSES = ("succeeded", "empty", "failed", "pending")
_OWNER_PRECEDENCE = {"succeeded": 0, "empty": 1, "failed": 2, "pending": 3}
_SYNTHETIC_ERROR_CODE = "predecessor_failed"
_SYNTHETIC_ERROR_DETAIL = "unknown pre-b7 failure"

# Plan-exact database lifecycle CHECK (W4).
_LIFECYCLE_CHECK_SQL = (
    "CASE status "
    "WHEN 'pending' THEN completed_at IS NULL AND error_code IS NULL "
    "AND error_detail IS NULL AND attempt_count >= 1 "
    "WHEN 'succeeded' THEN completed_at IS NOT NULL AND error_code IS NULL "
    "AND error_detail IS NULL AND attempt_count >= 1 "
    "WHEN 'empty' THEN completed_at IS NOT NULL AND error_code IS NULL "
    "AND error_detail IS NULL AND attempt_count >= 1 "
    "WHEN 'failed' THEN completed_at IS NOT NULL AND error_code IS NOT NULL "
    "AND attempt_count >= 1 "
    "WHEN 'skipped' THEN completed_at IS NOT NULL AND error_code IN "
    "('extraction_disabled', 'unsupported_media_type') AND attempt_count = 0 "
    "ELSE 0 END"
)

# Literal target schema for graph_extractions at this revision. ``{table}`` is
# the staging name at creation time and the physical name after the swap.
_TARGET_DDL_TEMPLATE = """CREATE TABLE {table} (
	id INTEGER NOT NULL, 
	chunk_id INTEGER NOT NULL, 
	provider VARCHAR(100) NOT NULL, 
	model VARCHAR(255) NOT NULL, 
	prompt_version VARCHAR(50) NOT NULL, 
	schema_version VARCHAR(50) NOT NULL, 
	status VARCHAR(20) NOT NULL, 
	error_code VARCHAR(100), 
	error_detail TEXT, 
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, 
	input_sha256 VARCHAR(64) NOT NULL, 
	attempt_count INTEGER DEFAULT '0' NOT NULL, 
	completed_at DATETIME, 
	attempt_started_at DATETIME, 
	is_identity_owner BOOLEAN DEFAULT '1' NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_graph_extractions_status CHECK (status IN ('pending', 'succeeded', 'failed', 'empty', 'skipped')), 
	CONSTRAINT ck_graph_extractions_attempt_count CHECK (attempt_count >= 0), 
	CONSTRAINT ck_graph_extractions_is_identity_owner CHECK (is_identity_owner IN (0,1)), 
	CONSTRAINT ck_graph_extractions_input_sha256_hex CHECK (length(input_sha256) = 64 AND input_sha256 NOT GLOB '*[^0-9a-f]*'), 
	CONSTRAINT ck_graph_extractions_lifecycle CHECK ({lifecycle}), 
	FOREIGN KEY(chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
)""".replace("{lifecycle}", _LIFECYCLE_CHECK_SQL)

# Literal predecessor (a6e2c4f8b1d9) schema restored by the downgrade.
_PREDECESSOR_DDL_TEMPLATE = """CREATE TABLE {table} (
	id INTEGER NOT NULL, 
	chunk_id INTEGER NOT NULL, 
	provider VARCHAR(100) NOT NULL, 
	model VARCHAR(255) NOT NULL, 
	prompt_version VARCHAR(50) NOT NULL, 
	schema_version VARCHAR(50) NOT NULL, 
	status VARCHAR(20) NOT NULL, 
	error_code VARCHAR(100), 
	error_detail TEXT, 
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_graph_extractions_status CHECK (status IN ('pending', 'succeeded', 'failed', 'empty')), 
	FOREIGN KEY(chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
)"""

_TARGET_INDEX_DDL = {
    "ix_graph_extractions_id": (
        "CREATE INDEX ix_graph_extractions_id ON graph_extractions (id)"
    ),
    "uq_graph_extractions_identity_owner": (
        "CREATE UNIQUE INDEX uq_graph_extractions_identity_owner "
        "ON graph_extractions (chunk_id, provider, model, prompt_version, "
        "schema_version, input_sha256) WHERE is_identity_owner = 1"
    ),
}
_PREDECESSOR_INDEX_DDL = {
    "ix_graph_extractions_id": (
        "CREATE INDEX ix_graph_extractions_id ON graph_extractions (id)"
    ),
}

_SOURCE_COLUMNS = (
    "id, chunk_id, provider, model, prompt_version, schema_version, "
    "status, error_code, error_detail, created_at"
)


def upgrade() -> None:
    bind = op.get_bind()

    # ------------------------------------------------------------------
    # 0. Self-heal staging debris left by pre-transactional-DDL failures so
    #    a retry can proceed (the migration transaction itself now rolls
    #    back completely; this cleans up historical half-applied attempts).
    # ------------------------------------------------------------------
    bind.exec_driver_sql("DROP TABLE IF EXISTS _b7_new")
    bind.exec_driver_sql("DROP TABLE IF EXISTS _b7_conversion")

    # ------------------------------------------------------------------
    # 1. Refuse explicitly before touching anything if any predecessor row
    #    carries a status outside the predecessor enum (no silent coercion).
    # ------------------------------------------------------------------
    source_rows = bind.execute(
        sa.text(f"SELECT {_SOURCE_COLUMNS} FROM graph_extractions ORDER BY id")
    ).fetchall()
    unknown = sorted({r.status for r in source_rows if r.status not in _PREDECESSOR_STATUSES})
    if unknown:
        raise RuntimeError(
            "b7f3d5a9c2e1: refusing to upgrade graph_extractions rows with "
            f"unrecognized status {unknown!r}; the predecessor a6e2c4f8b1d9 "
            f"contract allows only {list(_PREDECESSOR_STATUSES)!r}. Resolve or "
            "export these rows before upgrading."
        )

    # ------------------------------------------------------------------
    # 2. Add chunk media type (backfills existing rows to 'text/plain').
    # ------------------------------------------------------------------
    with op.batch_alter_table("chunks") as batch_op:
        batch_op.add_column(
            sa.Column(
                "media_type",
                sa.String(length=100),
                nullable=False,
                server_default="text/plain",
            )
        )

    # ------------------------------------------------------------------
    # 3. Compute the deterministic conversion matrix in Python.
    # ------------------------------------------------------------------
    conversions, synthetic_patched, status_counts = _compute_conversions(
        bind, source_rows
    )

    # ------------------------------------------------------------------
    # 4. Deterministic staging rebuild: create _b7_new with the literal
    #    target schema, carry every column via INSERT ... SELECT ordered by
    #    PK with the computed columns set during the copy.
    # ------------------------------------------------------------------
    bind.exec_driver_sql(_TARGET_DDL_TEMPLATE.format(table="_b7_new"))
    bind.exec_driver_sql(
        """CREATE TABLE _b7_conversion (
	extraction_id INTEGER NOT NULL PRIMARY KEY, 
	input_sha256 VARCHAR(64) NOT NULL, 
	attempt_count INTEGER NOT NULL, 
	completed_at DATETIME, 
	attempt_started_at DATETIME, 
	error_code VARCHAR(100), 
	error_detail TEXT, 
	is_identity_owner BOOLEAN NOT NULL
)"""
    )
    conversion_params = [
        {"extraction_id": rid, **fields} for rid, fields in sorted(conversions.items())
    ]
    if conversion_params:  # empty executemany is not a no-op in SQLAlchemy 2.0
        bind.execute(
            sa.text(
                "INSERT INTO _b7_conversion (extraction_id, input_sha256, "
                "attempt_count, completed_at, attempt_started_at, error_code, "
                "error_detail, is_identity_owner) VALUES (:extraction_id, "
                ":input_sha256, :attempt_count, :completed_at, "
                ":attempt_started_at, :error_code, :error_detail, "
                ":is_identity_owner)"
            ),
            conversion_params,
        )
    bind.execute(
        sa.text(
            "INSERT INTO _b7_new (id, chunk_id, provider, model, "
            "prompt_version, schema_version, status, error_code, "
            "error_detail, created_at, input_sha256, attempt_count, "
            "completed_at, attempt_started_at, is_identity_owner) "
            "SELECT ge.id, ge.chunk_id, ge.provider, ge.model, "
            "ge.prompt_version, ge.schema_version, ge.status, "
            "conv.error_code, conv.error_detail, ge.created_at, "
            "conv.input_sha256, conv.attempt_count, conv.completed_at, "
            "conv.attempt_started_at, conv.is_identity_owner "
            "FROM graph_extractions AS ge "
            "JOIN _b7_conversion AS conv ON conv.extraction_id = ge.id "
            "ORDER BY ge.id"
        )
    )

    # ------------------------------------------------------------------
    # 5. Assert the copy is lossless BEFORE the swap.
    # ------------------------------------------------------------------
    _assert_rebuild_lossless(
        bind,
        staging_table="_b7_new",
        conversions=conversions,
        status_counts=status_counts,
        synthetic_patched=synthetic_patched,
    )

    # ------------------------------------------------------------------
    # 6. Swap tables and recreate every index.
    # ------------------------------------------------------------------
    bind.exec_driver_sql("DROP TABLE graph_extractions")
    bind.exec_driver_sql("ALTER TABLE _b7_new RENAME TO graph_extractions")
    for index_ddl in _TARGET_INDEX_DDL.values():
        bind.exec_driver_sql(index_ddl)
    bind.exec_driver_sql("DROP TABLE _b7_conversion")

    # ------------------------------------------------------------------
    # 7. PRAGMA foreign_key_check + exact schema snapshot before commit.
    # ------------------------------------------------------------------
    _assert_foreign_key_check_clean(bind)
    _assert_schema_snapshot(bind, _TARGET_DDL_TEMPLATE.format(table="graph_extractions"), _TARGET_INDEX_DDL)


def downgrade() -> None:
    """Downgrade to the a6e2c4f8b1d9 schema.

    Refuses while any ``skipped`` row exists, because the predecessor
    ``status`` CHECK cannot represent it. After the operator removes only
    explicitly selected skipped rows, this downgrade rebuilds the exact
    predecessor table/checks and preserves every representable row/ID.
    """
    bind = op.get_bind()
    # Self-heal staging debris from pre-transactional-DDL failures (see
    # upgrade step 0).
    bind.exec_driver_sql("DROP TABLE IF EXISTS _b7_old")
    skipped = bind.execute(
        sa.text("SELECT COUNT(*) FROM graph_extractions WHERE status = 'skipped'")
    ).scalar()
    if skipped:
        raise RuntimeError(
            "downgrade_skipped_rows_present: cannot downgrade b7f3d5a9c2e1 while "
            f"{skipped} skipped row(s) exist; export/delete them via "
            "scripts/export_unrepresentable_rows.py --revision b7f3d5a9c2e1 first"
        )

    current_rows = bind.execute(
        sa.text(f"SELECT {_SOURCE_COLUMNS} FROM graph_extractions ORDER BY id")
    ).fetchall()

    bind.exec_driver_sql(_PREDECESSOR_DDL_TEMPLATE.format(table="_b7_old"))
    bind.execute(
        sa.text(
            "INSERT INTO _b7_old (id, chunk_id, provider, model, "
            "prompt_version, schema_version, status, error_code, "
            "error_detail, created_at) "
            f"SELECT {_SOURCE_COLUMNS} FROM graph_extractions ORDER BY id"
        )
    )
    _assert_downgrade_lossless(bind, current_rows)

    bind.exec_driver_sql("DROP TABLE graph_extractions")
    bind.exec_driver_sql("ALTER TABLE _b7_old RENAME TO graph_extractions")
    for index_ddl in _PREDECESSOR_INDEX_DDL.values():
        bind.exec_driver_sql(index_ddl)

    _assert_foreign_key_check_clean(bind)
    _assert_schema_snapshot(
        bind,
        _PREDECESSOR_DDL_TEMPLATE.format(table="graph_extractions"),
        _PREDECESSOR_INDEX_DDL,
    )

    with op.batch_alter_table("chunks") as batch_op:
        batch_op.drop_column("media_type")


# ---------------------------------------------------------------------------
# Conversion matrix
# ---------------------------------------------------------------------------


def _compute_conversions(bind, source_rows) -> tuple[dict, int, dict]:
    """Deterministically compute the converted columns for every row.

    Returns ``(conversions, synthetic_patched, status_counts)`` where
    ``conversions`` maps extraction id to the converted column values.
    """
    chunk_texts = dict(
        bind.execute(sa.text("SELECT id, text FROM chunks ORDER BY id")).fetchall()
    )
    sha_by_chunk = {
        chunk_id: hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        for chunk_id, text in chunk_texts.items()
    }
    empty_sha = hashlib.sha256(b"").hexdigest()

    conversions: dict[int, dict] = {}
    synthetic_patched = 0
    status_counts = {status: 0 for status in _PREDECESSOR_STATUSES}

    for row in source_rows:
        status_counts[row.status] += 1
        input_sha256 = sha_by_chunk.get(row.chunk_id, empty_sha)

        if row.status in ("succeeded", "empty"):
            error_code = None
            error_detail = None
        elif row.status == "failed":
            if row.error_code is None:
                error_code = _SYNTHETIC_ERROR_CODE
                error_detail = _SYNTHETIC_ERROR_DETAIL
                synthetic_patched += 1
            else:
                error_code = row.error_code
                error_detail = row.error_detail
        else:  # pending
            error_code = None
            error_detail = None

        conversions[row.id] = {
            "input_sha256": input_sha256,
            "attempt_count": 1,
            "completed_at": (
                row.created_at if row.status != "pending" else None
            ),
            "attempt_started_at": row.created_at,
            "error_code": error_code,
            "error_detail": error_detail,
            "is_identity_owner": 1,
        }

    _select_identity_owners(source_rows, conversions)
    return conversions, synthetic_patched, status_counts


def _select_identity_owners(source_rows, conversions) -> None:
    """Mark one owner per legacy identity collision group.

    Owner precedence: status ``succeeded > empty > failed > pending``, then
    ``completed_at`` DESC with NULL last, then ``id`` ASC. Deterministic.
    """
    grouped: dict[tuple, list] = {}
    for row in source_rows:
        converted = conversions[row.id]
        key = (
            row.chunk_id,
            row.provider,
            row.model,
            row.prompt_version,
            row.schema_version,
            converted["input_sha256"],
        )
        grouped.setdefault(key, []).append(row)

    def _completed_sort_value(row):
        value = conversions[row.id]["completed_at"]
        if value is None:
            return None
        if isinstance(value, _dt.datetime):
            return value.isoformat(sep=" ")
        return str(value)

    def _compare(left, right) -> int:
        precedence_delta = _OWNER_PRECEDENCE[left.status] - _OWNER_PRECEDENCE[right.status]
        if precedence_delta:
            return -1 if precedence_delta < 0 else 1
        left_completed = _completed_sort_value(left)
        right_completed = _completed_sort_value(right)
        if (left_completed is None) != (right_completed is None):
            return 1 if left_completed is None else -1  # NULL last under DESC
        if left_completed is not None and left_completed != right_completed:
            return 1 if left_completed < right_completed else -1  # DESC
        return -1 if left.id < right.id else (1 if left.id > right.id else 0)

    for members in grouped.values():
        owner = min(members, key=functools.cmp_to_key(_compare))
        for member in members:
            conversions[member.id]["is_identity_owner"] = (
                1 if member.id == owner.id else 0
            )


# ---------------------------------------------------------------------------
# Deterministic rebuild assertions
# ---------------------------------------------------------------------------


def _canonical(value) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, _dt.datetime):
        return value.isoformat(sep=" ")
    return str(value)


def _row_hash(values) -> str:
    payload = "\x1f".join(_canonical(v) for v in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_rebuild_lossless(
    bind,
    *,
    staging_table: str,
    conversions: dict,
    status_counts: dict,
    synthetic_patched: int,
) -> None:
    """Assert row counts, PK sets, per-row content hashes and conversion counts."""
    source_count = bind.execute(
        sa.text("SELECT COUNT(*) FROM graph_extractions")
    ).scalar()
    staged_count = bind.execute(
        sa.text(f"SELECT COUNT(*) FROM {staging_table}")
    ).scalar()
    if source_count != staged_count:
        raise RuntimeError(
            f"b7f3d5a9c2e1 rebuild failed: row count mismatch "
            f"(source={source_count}, staged={staged_count})"
        )

    source_rows = bind.execute(
        sa.text(f"SELECT {_SOURCE_COLUMNS} FROM graph_extractions ORDER BY id")
    ).fetchall()
    staged_rows = bind.execute(
        sa.text(
            "SELECT id, chunk_id, provider, model, prompt_version, "
            "schema_version, status, error_code, error_detail, created_at, "
            "input_sha256, attempt_count, completed_at, attempt_started_at, "
            f"is_identity_owner FROM {staging_table} ORDER BY id"
        )
    ).fetchall()

    source_pks = [r.id for r in source_rows]
    staged_pks = [r.id for r in staged_rows]
    if source_pks != staged_pks:
        raise RuntimeError(
            "b7f3d5a9c2e1 rebuild failed: PK set mismatch "
            f"(source-only={sorted(set(source_pks) - set(staged_pks))}, "
            f"staged-only={sorted(set(staged_pks) - set(source_pks))})"
        )

    staged_by_id = {r.id: r for r in staged_rows}
    for row in source_rows:
        staged = staged_by_id[row.id]
        # Preserved columns must be byte-identical after canonicalization.
        source_preserved = (
            row.id, row.chunk_id, row.provider, row.model, row.prompt_version,
            row.schema_version, row.status, row.created_at,
        )
        staged_preserved = (
            staged.id, staged.chunk_id, staged.provider, staged.model,
            staged.prompt_version, staged.schema_version, staged.status,
            staged.created_at,
        )
        if _row_hash(source_preserved) != _row_hash(staged_preserved):
            raise RuntimeError(
                f"b7f3d5a9c2e1 rebuild failed: content hash mismatch for "
                f"graph_extractions.id={row.id}"
            )
        # Computed columns must equal the deterministic conversion exactly.
        expected = conversions[row.id]
        actual = {
            "input_sha256": staged.input_sha256,
            "attempt_count": staged.attempt_count,
            "completed_at": staged.completed_at,
            "attempt_started_at": staged.attempt_started_at,
            "error_code": staged.error_code,
            "error_detail": staged.error_detail,
            "is_identity_owner": staged.is_identity_owner,
        }
        if actual != expected:
            raise RuntimeError(
                f"b7f3d5a9c2e1 rebuild failed: conversion mismatch for "
                f"graph_extractions.id={row.id}: expected={expected!r}, "
                f"actual={actual!r}"
            )

    # Counting assertion: every predecessor status survives the conversion.
    staged_status_counts = dict(
        bind.execute(
            sa.text(
                f"SELECT status, COUNT(*) FROM {staging_table} GROUP BY status"
            )
        ).fetchall()
    )
    if staged_status_counts != {
        status: count for status, count in status_counts.items() if count
    }:
        raise RuntimeError(
            "b7f3d5a9c2e1 rebuild failed: per-status conversion count mismatch "
            f"(expected={status_counts!r}, staged={staged_status_counts!r})"
        )

    # Counting assertion: synthetic-error patches must equal the number of
    # predecessor failed rows whose error_code was NULL.
    source_null_error_failed = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM graph_extractions "
            "WHERE status = 'failed' AND error_code IS NULL"
        )
    ).scalar()
    if synthetic_patched != source_null_error_failed:
        raise RuntimeError(
            "b7f3d5a9c2e1 rebuild failed: synthetic error patch count "
            f"({synthetic_patched}) does not match predecessor failed rows "
            f"with NULL error_code ({source_null_error_failed})"
        )


def _assert_downgrade_lossless(bind, current_rows) -> None:
    """Assert the downgrade copy carried every row and column unchanged."""
    staged_rows = bind.execute(
        sa.text("SELECT id, chunk_id, provider, model, prompt_version, "
                "schema_version, status, error_code, error_detail, created_at "
                "FROM _b7_old ORDER BY id")
    ).fetchall()
    if len(current_rows) != len(staged_rows):
        raise RuntimeError(
            "b7f3d5a9c2e1 downgrade failed: row count mismatch "
            f"(source={len(current_rows)}, staged={len(staged_rows)})"
        )
    if [r.id for r in current_rows] != [r.id for r in staged_rows]:
        raise RuntimeError(
            "b7f3d5a9c2e1 downgrade failed: PK set mismatch"
        )
    for source, staged in zip(current_rows, staged_rows):
        if _row_hash(tuple(source)) != _row_hash(tuple(staged)):
            raise RuntimeError(
                f"b7f3d5a9c2e1 downgrade failed: content hash mismatch for "
                f"graph_extractions.id={source.id}"
            )


def _assert_foreign_key_check_clean(bind) -> None:
    """Assert the rebuild kept graph_extractions' own foreign keys clean.

    Scoped to the rebuilt table: this migration preserves every row ID, so
    child tables (entity_mentions, graph_edge_evidence) are exactly as
    referentially valid as before the rebuild, and pre-existing orphans in
    unrelated tables are not this revision's invariant to police.
    """
    violations = bind.exec_driver_sql(
        "PRAGMA foreign_key_check(graph_extractions)"
    ).fetchall()
    if violations:
        raise RuntimeError(
            "b7f3d5a9c2e1 rebuild failed: PRAGMA foreign_key_check reported "
            f"violations: {violations[:10]!r}"
        )


def _canonical_ddl(sql: str) -> str:
    """Canonicalize stored DDL for exact-snapshot comparison.

    Collapses whitespace and strips SQLite identifier quoting (``ALTER TABLE
    ... RENAME TO`` stores the new table name quoted), mirroring the
    normalization used by ``app.core.migrations._normalize_sql``.
    """
    return " ".join(sql.replace('"', "").split())


def _assert_schema_snapshot(bind, expected_table_ddl: str, expected_indexes: dict) -> None:
    """Exact schema snapshot: stored CREATE statements must match literally."""
    stored_table_ddl = bind.exec_driver_sql(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'graph_extractions'"
    ).scalar()
    if _canonical_ddl(stored_table_ddl) != _canonical_ddl(expected_table_ddl):
        raise RuntimeError(
            "b7f3d5a9c2e1 rebuild failed: graph_extractions schema snapshot "
            f"differs from the literal target schema.\nExpected:\n"
            f"{expected_table_ddl}\nGot:\n{stored_table_ddl}"
        )
    stored_indexes = {
        row[0]: _canonical_ddl(row[1])
        for row in bind.exec_driver_sql(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'graph_extractions' AND sql IS NOT NULL"
        ).fetchall()
    }
    expected_canonical = {
        name: _canonical_ddl(ddl) for name, ddl in expected_indexes.items()
    }
    if stored_indexes != expected_canonical:
        raise RuntimeError(
            "b7f3d5a9c2e1 rebuild failed: graph_extractions index snapshot "
            f"differs from the expected index set.\nExpected: "
            f"{expected_canonical!r}\nGot: {stored_indexes!r}"
        )
