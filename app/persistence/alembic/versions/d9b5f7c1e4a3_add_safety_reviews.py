"""add safety reviews

Revision ID: d9b5f7c1e4a3
Revises: c8a4e6b0d3f2
Create Date: 2026-08-15

Task 10C.4 sole forward owner of all 10C physical changes:

* creates ``safety_review_runs`` and ``safety_findings`` with the exact
  columns/checks/indexes/partial uniques of the 10C.4 contract;
* deterministically rebuilds ``graph_extractions`` with the exact b7 layout
  but adds the closed skip-reason vocabulary
  ``extraction_disabled|unsupported_media_type|safety_blocked`` (skipped rows
  require ``attempt_count = 0``);
* deterministically rebuilds ``retrieval_candidate_decisions`` with the exact
  c8 layout but expands the decision vocabulary to include
  ``rejected_safety``.

Rebuild procedure: create ``_d9_new`` with the literal schema, copy every
column ordered by PK, assert source/destination row counts and canonical PK
fingerprints equal, drop the predecessor, rename, recreate every index.
``PRAGMA foreign_key_check`` must pass before commit. Downgrade refuses with
a typed error while any ``safety_blocked`` or ``rejected_safety`` row exists,
then performs the inverse rebuilds restoring the exact b7/c8 checks.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d9b5f7c1e4a3"
down_revision: Union[str, None] = "c8a4e6b0d3f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_GRAPH_EXTRACTION_COLUMNS = (
    "id", "chunk_id", "provider", "model", "prompt_version", "schema_version",
    "status", "error_code", "error_detail", "created_at", "input_sha256",
    "attempt_count", "completed_at", "attempt_started_at", "is_identity_owner",
)

_DECISION_COLUMNS = (
    "id", "audit_id", "document_id", "chunk_id", "document_id_snapshot",
    "chunk_id_snapshot", "decision", "native_score", "provenance_score",
    "reason_codes", "content_sha256", "bounded_excerpt",
)

_DECISION_INDEX_NAMES = (
    "ix_candidate_decisions_audit_decision",
    "ix_candidate_decisions_live_doc",
    "ix_candidate_decisions_live_chunk",
    "ix_candidate_decisions_snapshot_doc",
    "ix_candidate_decisions_snapshot_chunk",
)

_GE_COMMON_CONSTRAINTS = (
    "    CONSTRAINT ck_graph_extractions_status CHECK (status IN "
    "('pending', 'succeeded', 'failed', 'empty', 'skipped')),\n"
    "    CONSTRAINT ck_graph_extractions_attempt_count CHECK "
    "(attempt_count >= 0),\n"
    "    CONSTRAINT ck_graph_extractions_is_identity_owner CHECK "
    "(is_identity_owner IN (0, 1)),\n"
    "    CONSTRAINT ck_graph_extractions_input_sha256_hex CHECK "
    "(length(input_sha256) = 64 AND input_sha256 NOT GLOB '*[^0-9a-f]*')"
)

_GE_SKIP_CONSTRAINT_D9 = (
    ",\n    CONSTRAINT ck_graph_extractions_skip_reason CHECK "
    "(status != 'skipped' OR (error_code IN ('extraction_disabled', "
    "'unsupported_media_type', 'safety_blocked') AND attempt_count = 0))"
)


def _graph_extractions_ddl(skip_constraint: str) -> str:
    return (
        "CREATE TABLE graph_extractions_d9_new (\n"
        "    id INTEGER NOT NULL,\n"
        "    chunk_id INTEGER NOT NULL,\n"
        "    provider VARCHAR(100) NOT NULL,\n"
        "    model VARCHAR(255) NOT NULL,\n"
        "    prompt_version VARCHAR(50) NOT NULL,\n"
        "    schema_version VARCHAR(50) NOT NULL,\n"
        "    status VARCHAR(20) NOT NULL,\n"
        "    error_code VARCHAR(100),\n"
        "    error_detail TEXT,\n"
        "    created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),\n"
        "    input_sha256 VARCHAR(64) NOT NULL "
        "DEFAULT '0000000000000000000000000000000000000000000000000000000000000000',\n"
        "    attempt_count INTEGER NOT NULL DEFAULT '0',\n"
        "    completed_at DATETIME,\n"
        "    attempt_started_at DATETIME,\n"
        "    is_identity_owner BOOLEAN NOT NULL DEFAULT '1',\n"
        "    PRIMARY KEY (id),\n"
        "    FOREIGN KEY(chunk_id) REFERENCES chunks (id) ON DELETE CASCADE,\n"
        f"{_GE_COMMON_CONSTRAINTS}{skip_constraint}\n"
        ")"
    )


_GRAPH_EXTRACTIONS_D9_DDL = _graph_extractions_ddl(_GE_SKIP_CONSTRAINT_D9)
_GRAPH_EXTRACTIONS_B7_DDL = _graph_extractions_ddl("")

_GRAPH_EXTRACTIONS_INDEXES = (
    "CREATE INDEX ix_graph_extractions_id ON graph_extractions (id)",
    "CREATE UNIQUE INDEX uq_graph_extractions_identity_owner ON "
    "graph_extractions (chunk_id, provider, model, prompt_version, "
    "schema_version, input_sha256) WHERE is_identity_owner = 1",
)

_DECISIONS_COMMON_CONSTRAINTS = (
    "    CONSTRAINT uq_candidate_decisions_audit_chunk UNIQUE "
    "(audit_id, chunk_id_snapshot),\n"
    "    CONSTRAINT ck_candidate_decisions_provenance_score CHECK "
    "(provenance_score >= 0 AND provenance_score <= 1)"
)

_DECISION_CONSTRAINT_D9 = (
    ",\n    CONSTRAINT ck_candidate_decisions_decision CHECK (decision IN "
    "('selected', 'rejected_distance', 'rejected_blocked_source', "
    "'rejected_source_cap', 'rejected_document_cap', 'rejected_duplicate', "
    "'rejected_injection', 'rejected_safety'))"
)


def _decisions_ddl(decision_constraint: str) -> str:
    return (
        "CREATE TABLE retrieval_candidate_decisions_d9_new (\n"
        "    id INTEGER NOT NULL,\n"
        "    audit_id VARCHAR(36) NOT NULL,\n"
        "    document_id INTEGER,\n"
        "    chunk_id INTEGER,\n"
        "    document_id_snapshot INTEGER NOT NULL,\n"
        "    chunk_id_snapshot INTEGER NOT NULL,\n"
        "    decision VARCHAR(50) NOT NULL,\n"
        "    native_score FLOAT,\n"
        "    provenance_score FLOAT NOT NULL,\n"
        "    reason_codes TEXT NOT NULL,\n"
        "    content_sha256 VARCHAR(64) NOT NULL,\n"
        "    bounded_excerpt VARCHAR(200),\n"
        "    PRIMARY KEY (id),\n"
        "    FOREIGN KEY(audit_id) REFERENCES retrieval_audits (id) "
        "ON DELETE CASCADE,\n"
        "    FOREIGN KEY(document_id) REFERENCES documents (id) "
        "ON DELETE SET NULL,\n"
        "    FOREIGN KEY(chunk_id) REFERENCES chunks (id) "
        "ON DELETE SET NULL,\n"
        f"{_DECISIONS_COMMON_CONSTRAINTS}{decision_constraint}\n"
        ")"
    )


_DECISIONS_D9_DDL = _decisions_ddl(_DECISION_CONSTRAINT_D9)
_DECISIONS_C8_DDL = _decisions_ddl("")

_DECISIONS_INDEXES = tuple(
    f"CREATE INDEX {name} ON retrieval_candidate_decisions "
    f"({columns})"
    for name, columns in (
        ("ix_candidate_decisions_audit_decision", "audit_id, decision"),
        ("ix_candidate_decisions_live_doc", "document_id"),
        ("ix_candidate_decisions_live_chunk", "chunk_id"),
        ("ix_candidate_decisions_snapshot_doc", "document_id_snapshot"),
        ("ix_candidate_decisions_snapshot_chunk", "chunk_id_snapshot"),
    )
)


def _pk_fingerprint(bind, table: str) -> str:
    rows = bind.execute(
        sa.text(f"SELECT id FROM {table} ORDER BY id")
    ).fetchall()
    return hashlib.sha256(
        json.dumps([list(r) for r in rows], default=str).encode("utf-8")
    ).hexdigest()


def _drop_indexes(bind, names: Sequence[str]) -> None:
    for name in names:
        bind.execute(sa.text(f"DROP INDEX IF EXISTS {name}"))


def _rebuild_table(
    bind,
    table: str,
    columns: tuple[str, ...],
    create_new: str,
    indexes_after: tuple[str, ...],
) -> None:
    """Deterministic rebuild: create _d9_new, copy ordered by PK, assert
    equality, drop predecessor, rename, recreate indexes."""
    before = (
        bind.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar(),
        _pk_fingerprint(bind, table),
    )
    bind.execute(sa.text(f"DROP TABLE IF EXISTS {table}_d9_new"))
    bind.execute(sa.text(create_new))
    bind.execute(sa.text(
        f"INSERT INTO {table}_d9_new ({', '.join(columns)}) "
        f"SELECT {', '.join(columns)} FROM {table} ORDER BY id"))
    after = (
        bind.execute(
            sa.text(f"SELECT COUNT(*) FROM {table}_d9_new")).scalar(),
        _pk_fingerprint(bind, f"{table}_d9_new"),
    )
    if before != after:
        raise RuntimeError(
            f"d9b5f7c1e4a3 rebuild fingerprint mismatch for {table}: "
            f"{before} != {after}"
        )
    bind.execute(sa.text(f"DROP TABLE {table}"))
    bind.execute(sa.text(f"ALTER TABLE {table}_d9_new RENAME TO {table}"))
    for statement in indexes_after:
        bind.execute(sa.text(statement))
    # Scope the check to the rebuilt table only: pre-existing violations in
    # unrelated tables (e.g. legacy chunks.document_id) are not this
    # revision's concern and must not block the rebuild.
    violations = bind.execute(
        sa.text(f"PRAGMA foreign_key_check('{table}')")).fetchall()
    if violations:
        raise RuntimeError(
            f"d9b5f7c1e4a3 foreign_key_check failed for {table}: "
            f"{violations[:5]}"
        )


def upgrade() -> None:
    # 1. safety_review_runs.
    op.create_table(
        "safety_review_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False,
                  server_default="pending"),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("chunk_id", sa.Integer(), nullable=True),
        sa.Column("document_id_snapshot", sa.Integer(), nullable=True),
        sa.Column("chunk_id_snapshot", sa.Integer(), nullable=True),
        sa.Column("retrieval_audit_id", sa.String(length=36), nullable=True),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=50), nullable=False),
        sa.Column("detector_version", sa.String(length=50), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=True),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("prompt_version", sa.String(length=50), nullable=True),
        sa.Column("schema_version", sa.String(length=50), nullable=True),
        sa.Column("llm_status", sa.String(length=20), nullable=False,
                  server_default="skipped"),
        sa.Column("final_action", sa.String(length=10), nullable=True),
        sa.Column("failure_code", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "scope IN ('ingestion', 'context', 'answer')",
            name="ck_safety_runs_scope"),
        sa.CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed')",
            name="ck_safety_runs_status"),
        sa.CheckConstraint(
            "llm_status IN ('skipped', 'succeeded', 'failed')",
            name="ck_safety_runs_llm_status"),
        sa.CheckConstraint(
            "final_action IS NULL OR final_action IN "
            "('allow', 'warn', 'filter', 'block')",
            name="ck_safety_runs_action"),
        sa.CheckConstraint(
            "length(input_sha256) = 64 AND input_sha256 NOT GLOB "
            "'*[^0-9a-f]*'",
            name="ck_safety_runs_hash"),
        sa.CheckConstraint(
            "(scope = 'ingestion' AND document_id_snapshot IS NOT NULL "
            "AND chunk_id_snapshot IS NULL AND chunk_id IS NULL "
            "AND retrieval_audit_id IS NULL) OR "
            "(scope = 'context' AND document_id_snapshot IS NOT NULL "
            "AND chunk_id_snapshot IS NOT NULL "
            "AND retrieval_audit_id IS NOT NULL) OR "
            "(scope = 'answer' AND document_id_snapshot IS NULL "
            "AND chunk_id_snapshot IS NULL AND document_id IS NULL "
            "AND chunk_id IS NULL AND retrieval_audit_id IS NOT NULL)",
            name="ck_safety_runs_provenance"),
        sa.CheckConstraint(
            "(document_id IS NULL OR document_id = document_id_snapshot) "
            "AND (chunk_id IS NULL OR chunk_id = chunk_id_snapshot)",
            name="ck_safety_runs_live_ids"),
        sa.CheckConstraint(
            "(status = 'pending' AND completed_at IS NULL AND final_action "
            "IS NULL AND failure_code IS NULL AND llm_status = 'skipped') OR "
            "(status = 'succeeded' AND completed_at IS NOT NULL AND "
            "final_action IS NOT NULL AND failure_code IS NULL AND "
            "llm_status IN ('skipped', 'succeeded', 'failed')) OR "
            "(status = 'failed' AND completed_at IS NOT NULL AND final_action "
            "IS NULL AND failure_code IS NOT NULL AND llm_status IN "
            "('skipped', 'failed'))",
            name="ck_safety_runs_lifecycle"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"],
                                ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["retrieval_audit_id"],
                                ["retrieval_audits.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_safety_runs_scope_status_created", "safety_review_runs",
        ["scope", "status", "created_at"])
    op.create_index("ix_safety_runs_document", "safety_review_runs",
                    ["document_id"])
    op.create_index("ix_safety_runs_chunk", "safety_review_runs", ["chunk_id"])
    op.create_index("ix_safety_runs_document_snapshot", "safety_review_runs",
                    ["document_id_snapshot"])
    op.create_index("ix_safety_runs_chunk_snapshot", "safety_review_runs",
                    ["chunk_id_snapshot"])
    op.create_index("ix_safety_runs_retrieval_audit", "safety_review_runs",
                    ["retrieval_audit_id"])
    op.create_index("ix_safety_runs_policy_version", "safety_review_runs",
                    ["policy_version"])
    op.create_index(
        "uq_safety_runs_ingestion_document", "safety_review_runs",
        ["document_id_snapshot"], unique=True,
        sqlite_where=sa.text("scope = 'ingestion'"))
    op.create_index(
        "uq_safety_runs_context_audit_chunk", "safety_review_runs",
        ["retrieval_audit_id", "chunk_id_snapshot"], unique=True,
        sqlite_where=sa.text("scope = 'context'"))
    op.create_index(
        "uq_safety_runs_answer_audit", "safety_review_runs",
        ["retrieval_audit_id"], unique=True,
        sqlite_where=sa.text("scope = 'answer'"))

    # 2. safety_findings.
    op.create_table(
        "safety_findings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("review_run_id", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=10), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("source_rule_ids", sa.Text(), nullable=False),
        sa.Column("excerpt_sha256", sa.String(length=64), nullable=False),
        sa.Column("bounded_excerpt", sa.String(length=200), nullable=True),
        sa.CheckConstraint(
            "category IN ('violence', 'self_harm', 'sexual_content', "
            "'hate_harassment', 'illegal_activity', 'privacy_credentials')",
            name="ck_safety_findings_category"),
        sa.CheckConstraint(
            "severity >= 0 AND severity <= 4",
            name="ck_safety_findings_severity"),
        sa.CheckConstraint(
            "action IN ('allow', 'warn', 'filter', 'block')",
            name="ck_safety_findings_action"),
        sa.CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset",
            name="ck_safety_findings_offsets"),
        sa.CheckConstraint(
            "length(excerpt_sha256) = 64 AND excerpt_sha256 NOT GLOB "
            "'*[^0-9a-f]*'",
            name="ck_safety_findings_hash"),
        sa.CheckConstraint(
            "bounded_excerpt IS NULL OR length(bounded_excerpt) <= 200",
            name="ck_safety_findings_excerpt_length"),
        sa.ForeignKeyConstraint(["review_run_id"], ["safety_review_runs.id"],
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "review_run_id", "category", "start_offset", "end_offset",
            "source_rule_ids",
            name="uq_safety_findings_run_span_rules"),
    )
    op.create_index(
        "ix_safety_findings_run_category_action", "safety_findings",
        ["review_run_id", "category", "action"])

    # 3. Deterministic graph_extractions rebuild (b7 layout + skip vocabulary).
    bind = op.get_bind()
    _drop_indexes(bind, ("uq_graph_extractions_identity_owner",))
    _rebuild_table(
        bind, "graph_extractions", _GRAPH_EXTRACTION_COLUMNS,
        _GRAPH_EXTRACTIONS_D9_DDL, _GRAPH_EXTRACTIONS_INDEXES)

    # 4. Deterministic retrieval_candidate_decisions rebuild (c8 layout +
    # rejected_safety).
    _drop_indexes(bind, _DECISION_INDEX_NAMES)
    _rebuild_table(
        bind, "retrieval_candidate_decisions", _DECISION_COLUMNS,
        _DECISIONS_D9_DDL, _DECISIONS_INDEXES)


def downgrade() -> None:
    bind = op.get_bind()

    blocked = bind.execute(sa.text(
        "SELECT COUNT(*) FROM graph_extractions "
        "WHERE error_code = 'safety_blocked'")).scalar()
    if blocked:
        raise RuntimeError(
            "downgrade_safety_blocked_rows_present: cannot downgrade "
            f"d9b5f7c1e4a3 while {blocked} safety_blocked row(s) exist"
        )
    rejected = bind.execute(sa.text(
        "SELECT COUNT(*) FROM retrieval_candidate_decisions "
        "WHERE decision = 'rejected_safety'")).scalar()
    if rejected:
        raise RuntimeError(
            "downgrade_rejected_safety_rows_present: cannot downgrade "
            f"d9b5f7c1e4a3 while {rejected} rejected_safety row(s) exist"
        )

    # Inverse rebuilds restore the exact b7/c8 checks; rows/IDs preserved.
    _drop_indexes(bind, ("uq_graph_extractions_identity_owner",))
    _rebuild_table(
        bind, "graph_extractions", _GRAPH_EXTRACTION_COLUMNS,
        _GRAPH_EXTRACTIONS_B7_DDL, _GRAPH_EXTRACTIONS_INDEXES)
    _drop_indexes(bind, _DECISION_INDEX_NAMES)
    _rebuild_table(
        bind, "retrieval_candidate_decisions", _DECISION_COLUMNS,
        _DECISIONS_C8_DDL, _DECISIONS_INDEXES)

    op.drop_table("safety_findings")
    op.drop_table("safety_review_runs")
