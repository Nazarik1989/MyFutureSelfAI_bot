"""Add durable user access tiers and operator audit records."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260805_0023"
down_revision: str | None = "20260731_0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TIERS = "'guest', 'subscriber', 'admin', 'blocked'"


def upgrade() -> None:
    # Both supported databases can add constant-default, NOT NULL columns in
    # place. Keeping the CHECK as a column constraint avoids rebuilding SQLite's
    # heavily referenced users table while foreign_keys is enabled.
    op.add_column(
        "users",
        sa.Column(
            "access_tier",
            sa.String(16),
            sa.CheckConstraint(
                f"access_tier IN ({_TIERS})",
                name="ck_users_access_tier",
            ),
            nullable=False,
            server_default="guest",
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "access_version",
            sa.Integer(),
            sa.CheckConstraint(
                "access_version > 0",
                name="ck_users_access_version",
            ),
            nullable=False,
            server_default="1",
        ),
    )
    users = sa.table(
        "users",
        sa.column("onboarding_completed", sa.Boolean()),
        sa.column("access_tier", sa.String(16)),
    )
    op.execute(
        users.update()
        .where(users.c.onboarding_completed.is_(True))
        .values(access_tier="subscriber")
    )

    op.create_table(
        "access_tier_changes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("from_tier", sa.String(16), nullable=False),
        sa.Column("to_tier", sa.String(16), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            f"from_tier IN ({_TIERS})",
            name="ck_access_tier_changes_from_tier",
        ),
        sa.CheckConstraint(
            f"to_tier IN ({_TIERS})",
            name="ck_access_tier_changes_to_tier",
        ),
        sa.CheckConstraint(
            "length(source) BETWEEN 1 AND 64",
            name="ck_access_tier_changes_source_length",
        ),
    )
    op.create_index(
        "ix_access_tier_changes_user_id",
        "access_tier_changes",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_access_tier_changes_user_id", table_name="access_tier_changes")
    op.drop_table("access_tier_changes")
    op.drop_column("users", "access_version")
    op.drop_column("users", "access_tier")
