"""add security provenance and audits

Revision ID: c8a4e6b0d3f2
Revises: b7f3d5a9c2e1
Create Date: 2026-08-11

Adds document trust fields, retrieval_audits, retrieval_candidate_decisions,
and ingestion_rate_buckets tables for the 10B.2 security/provenance contract.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c8a4e6b0d3f2"
down_revision: Union[str, None] = "b7f3d5a9c2e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add trust/provenance columns to documents.
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column("trust_tier", sa.String(length=20), nullable=False, server_default="untrusted")
        )
        batch_op.add_column(
            sa.Column("trust_score", sa.Float(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("trust_policy_version", sa.String(length=50), nullable=False, server_default="unassigned")
        )
        batch_op.add_column(
            sa.Column("ingestion_origin", sa.String(length=50), nullable=False, server_default="api")
        )
        batch_op.create_index("ix_documents_trust_tier", ["trust_tier"])
        batch_op.create_index("ix_documents_trust_policy_version", ["trust_policy_version"])
        batch_op.create_check_constraint(
            "ck_documents_trust_tier",
            "trust_tier IN ('trusted', 'standard', 'untrusted', 'blocked')",
        )
        batch_op.create_check_constraint(
            "ck_documents_trust_score",
            "trust_score >= 0 AND trust_score <= 1",
        )

    # 2. Create retrieval_audits.
    op.create_table(
        "retrieval_audits",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("query_sha256", sa.String(length=64), nullable=False),
        sa.Column("retrieval_mode", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("provenance_policy_version", sa.String(length=50), nullable=False),
        sa.Column("retrieval_policy_version", sa.String(length=50), nullable=False),
        sa.Column("context_policy_version", sa.String(length=50), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("selected_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rejected_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_code", sa.String(length=100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("retrieval_mode IN ('vector', 'graph', 'hybrid')", name="ck_retrieval_audits_mode"),
        sa.CheckConstraint("status IN ('pending', 'completed', 'failed')", name="ck_retrieval_audits_status"),
        sa.CheckConstraint(
            "candidate_count >= 0 AND selected_count >= 0 AND rejected_count >= 0",
            name="ck_retrieval_audits_counts_nonneg",
        ),
        sa.CheckConstraint(
            "candidate_count = selected_count + rejected_count",
            name="ck_retrieval_audits_counter_equality",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND completed_at IS NULL AND failure_code IS NULL) "
            "OR (status = 'completed' AND completed_at IS NOT NULL AND failure_code IS NULL) "
            "OR (status = 'failed' AND completed_at IS NOT NULL AND failure_code IS NOT NULL)",
            name="ck_retrieval_audits_lifecycle",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # 3. Create retrieval_candidate_decisions.
    op.create_table(
        "retrieval_candidate_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("audit_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("chunk_id", sa.Integer(), nullable=True),
        sa.Column("document_id_snapshot", sa.Integer(), nullable=False),
        sa.Column("chunk_id_snapshot", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=50), nullable=False),
        sa.Column("native_score", sa.Float(), nullable=True),
        sa.Column("provenance_score", sa.Float(), nullable=False),
        sa.Column("reason_codes", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("bounded_excerpt", sa.String(length=200), nullable=True),
        sa.CheckConstraint(
            "provenance_score >= 0 AND provenance_score <= 1",
            name="ck_candidate_decisions_provenance_score",
        ),
        sa.ForeignKeyConstraint(["audit_id"], ["retrieval_audits.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("audit_id", "chunk_id_snapshot", name="uq_candidate_decisions_audit_chunk"),
    )
    op.create_index("ix_candidate_decisions_audit_decision", "retrieval_candidate_decisions", ["audit_id", "decision"])
    op.create_index("ix_candidate_decisions_live_doc", "retrieval_candidate_decisions", ["document_id"])
    op.create_index("ix_candidate_decisions_live_chunk", "retrieval_candidate_decisions", ["chunk_id"])
    op.create_index("ix_candidate_decisions_snapshot_doc", "retrieval_candidate_decisions", ["document_id_snapshot"])
    op.create_index("ix_candidate_decisions_snapshot_chunk", "retrieval_candidate_decisions", ["chunk_id_snapshot"])

    # 4. Create ingestion_rate_buckets.
    op.create_table(
        "ingestion_rate_buckets",
        sa.Column("identity_sha256", sa.String(length=64), nullable=False),
        sa.Column("window_start_epoch", sa.Integer(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False),
        sa.CheckConstraint("request_count > 0", name="ck_ingestion_rate_buckets_count"),
        sa.PrimaryKeyConstraint("identity_sha256", "window_start_epoch"),
    )


def downgrade() -> None:
    op.drop_table("ingestion_rate_buckets")
    op.drop_index("ix_candidate_decisions_snapshot_chunk", table_name="retrieval_candidate_decisions")
    op.drop_index("ix_candidate_decisions_snapshot_doc", table_name="retrieval_candidate_decisions")
    op.drop_index("ix_candidate_decisions_live_chunk", table_name="retrieval_candidate_decisions")
    op.drop_index("ix_candidate_decisions_live_doc", table_name="retrieval_candidate_decisions")
    op.drop_index("ix_candidate_decisions_audit_decision", table_name="retrieval_candidate_decisions")
    op.drop_table("retrieval_candidate_decisions")
    op.drop_table("retrieval_audits")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_constraint("ck_documents_trust_score", type_="check")
        batch_op.drop_constraint("ck_documents_trust_tier", type_="check")
        batch_op.drop_index("ix_documents_trust_policy_version")
        batch_op.drop_index("ix_documents_trust_tier")
        batch_op.drop_column("ingestion_origin")
        batch_op.drop_column("trust_policy_version")
        batch_op.drop_column("trust_score")
        batch_op.drop_column("trust_tier")
