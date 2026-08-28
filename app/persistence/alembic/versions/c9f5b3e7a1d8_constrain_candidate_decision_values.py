"""constrain candidate decision values

Revision ID: c9f5b3e7a1d8
Revises: c8a4e6b0d3f2
Create Date: 2026-08-27

Adds the plan-mandated 10B.3 decision CHECK to
``retrieval_candidate_decisions.decision`` (plan L1160: "decision VARCHAR(50)
with the complete 10B.3 decision CHECK"; L1205: "Database checks enforce every
contract above"). c8a4e6b0d3f2 is already merged, so the constraint is added
by a follow-up revision using the repository's established SQLite
batch table-rebuild pattern instead of rewriting c8's history.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "c9f5b3e7a1d8"
down_revision: Union[str, None] = "c8a4e6b0d3f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The complete 10B.3 decision enum (RetrievalSecurityDecision.decision).
DECISION_CHECK_SQL = (
    "decision IN ('selected', 'rejected_distance', 'rejected_blocked_source', "
    "'rejected_source_cap', 'rejected_document_cap', 'rejected_duplicate', "
    "'rejected_injection')"
)


def upgrade() -> None:
    with op.batch_alter_table("retrieval_candidate_decisions") as batch_op:
        batch_op.create_check_constraint(
            "ck_candidate_decisions_decision",
            DECISION_CHECK_SQL,
        )


def downgrade() -> None:
    with op.batch_alter_table("retrieval_candidate_decisions") as batch_op:
        batch_op.drop_constraint(
            "ck_candidate_decisions_decision",
            type_="check",
        )
