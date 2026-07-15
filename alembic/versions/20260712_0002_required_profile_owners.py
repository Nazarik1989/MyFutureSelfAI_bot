"""Make one-to-one owner foreign keys required on databases created by early MVP builds."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260712_0002"
down_revision: str | None = "20260712_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table_name in ("vision_profiles", "onboarding_states"):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(
                "user_id",
                existing_type=sa.Integer(),
                nullable=False,
            )


def downgrade() -> None:
    for table_name in ("vision_profiles", "onboarding_states"):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(
                "user_id",
                existing_type=sa.Integer(),
                nullable=True,
            )
