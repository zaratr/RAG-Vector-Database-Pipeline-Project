"""persist normalized graph extraction provenance

Revision ID: 4c9a8d7e6f5b
Revises: dee48bc24a7f
Create Date: 2026-07-30
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "4c9a8d7e6f5b"
down_revision: Union[str, None] = "dee48bc24a7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column(
                "ingestion_status",
                sa.String(length=20),
                nullable=False,
                server_default="ready",
            )
        )
        batch_op.add_column(sa.Column("failure_code", sa.String(length=100)))
        batch_op.create_index("ix_documents_ingestion_status", ["ingestion_status"])

    with op.batch_alter_table("chunks") as batch_op:
        batch_op.add_column(sa.Column("vector_id", sa.String(length=255)))
        batch_op.create_index("ix_chunks_vector_id", ["vector_id"], unique=True)
        batch_op.create_unique_constraint(
            "uq_chunks_document_index", ["document_id", "index"]
        )

    op.create_table(
        "graph_entities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("canonical_name", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("entity_type", sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "canonical_name", "entity_type", name="uq_graph_entities_name_type"
        ),
    )
    op.create_index("ix_graph_entities_id", "graph_entities", ["id"])
    op.create_index(
        "ix_graph_entities_canonical_name", "graph_entities", ["canonical_name"]
    )

    op.create_table(
        "graph_extractions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("schema_version", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_code", sa.String(length=100)),
        sa.Column("error_detail", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed', 'empty')",
            name="ck_graph_extractions_status",
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_graph_extractions_id", "graph_extractions", ["id"])

    op.create_table(
        "entity_mentions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("extraction_id", sa.Integer(), nullable=False),
        sa.Column("surface_form", sa.String(length=255), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset",
            name="ck_entity_mentions_offsets",
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"], ["graph_entities.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["extraction_id"], ["graph_extractions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "entity_id",
            "extraction_id",
            name="uq_entity_mentions_entity_extraction",
        ),
    )
    op.create_index("ix_entity_mentions_id", "entity_mentions", ["id"])

    op.create_table(
        "graph_edges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_entity_id", sa.Integer(), nullable=False),
        sa.Column("target_entity_id", sa.Integer(), nullable=False),
        sa.Column("predicate", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_entity_id"], ["graph_entities.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_entity_id"], ["graph_entities.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_entity_id",
            "predicate",
            "target_entity_id",
            name="uq_graph_edges_triplet",
        ),
    )
    op.create_index("ix_graph_edges_id", "graph_edges", ["id"])
    op.create_index("ix_graph_edges_predicate", "graph_edges", ["predicate"])

    op.create_table(
        "graph_edge_evidence",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("edge_id", sa.Integer(), nullable=False),
        sa.Column("extraction_id", sa.Integer(), nullable=False),
        sa.Column("evidence_text", sa.Text(), nullable=False),
        sa.Column("evidence_start", sa.Integer(), nullable=False),
        sa.Column("evidence_end", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "evidence_start >= 0 AND evidence_end > evidence_start",
            name="ck_graph_edge_evidence_offsets",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_graph_edge_evidence_confidence",
        ),
        sa.ForeignKeyConstraint(
            ["edge_id"], ["graph_edges.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["extraction_id"], ["graph_extractions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "edge_id",
            "extraction_id",
            "evidence_start",
            "evidence_end",
            name="uq_graph_edge_evidence_location",
        ),
    )
    op.create_index("ix_graph_edge_evidence_id", "graph_edge_evidence", ["id"])


def downgrade() -> None:
    op.drop_index("ix_graph_edge_evidence_id", table_name="graph_edge_evidence")
    op.drop_table("graph_edge_evidence")
    op.drop_index("ix_graph_edges_predicate", table_name="graph_edges")
    op.drop_index("ix_graph_edges_id", table_name="graph_edges")
    op.drop_table("graph_edges")
    op.drop_index("ix_entity_mentions_id", table_name="entity_mentions")
    op.drop_table("entity_mentions")
    op.drop_index("ix_graph_extractions_id", table_name="graph_extractions")
    op.drop_table("graph_extractions")
    op.drop_index("ix_graph_entities_canonical_name", table_name="graph_entities")
    op.drop_index("ix_graph_entities_id", table_name="graph_entities")
    op.drop_table("graph_entities")

    with op.batch_alter_table("chunks") as batch_op:
        batch_op.drop_constraint("uq_chunks_document_index", type_="unique")
        batch_op.drop_index("ix_chunks_vector_id")
        batch_op.drop_column("vector_id")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_index("ix_documents_ingestion_status")
        batch_op.drop_column("failure_code")
        batch_op.drop_column("ingestion_status")
