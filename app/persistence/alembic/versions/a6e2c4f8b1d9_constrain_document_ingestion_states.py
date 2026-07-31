"""constrain document ingestion lifecycle states

Revision ID: a6e2c4f8b1d9
Revises: 4c9a8d7e6f5b
Create Date: 2026-07-31
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "a6e2c4f8b1d9"
down_revision: Union[str, None] = "4c9a8d7e6f5b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.create_check_constraint(
            "ck_documents_ingestion_status",
            "ingestion_status IN ('staged', 'ready', 'failed')",
        )


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_constraint(
            "ck_documents_ingestion_status",
            type_="check",
        )
