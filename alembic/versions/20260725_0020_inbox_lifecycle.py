"""Add recoverable lifecycle metadata to saved Inbox items."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260725_0020"
down_revision: str | None = "20260722_0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("inbox_items") as batch:
        batch.add_column(sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
        batch.add_column(sa.Column("trashed_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("pre_trash_status", sa.String(20), nullable=True))
        batch.create_check_constraint("ck_inbox_item_version", "version > 0")
        batch.create_check_constraint(
            "ck_inbox_item_trash_state",
            "(status = 'trashed' AND trashed_at IS NOT NULL "
            "AND pre_trash_status IN ('confirmed', 'archived')) OR "
            "(status != 'trashed' AND trashed_at IS NULL "
            "AND pre_trash_status IS NULL)",
        )
    op.create_index(
        "ix_inbox_items_trashed_at",
        "inbox_items",
        ["trashed_at"],
    )
    op.create_index(
        "ix_inbox_items_user_status_id",
        "inbox_items",
        ["user_id", "status", "id"],
    )


def downgrade() -> None:
    # A previous application does not understand recoverable trash metadata.
    # Converting trashed rows to a legacy status would either make deleted data
    # visible again or destroy restore information. Refuse the downgrade until
    # an operator restores the rows explicitly; this keeps rollback fail-closed.
    trashed_count = int(
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM inbox_items WHERE status = 'trashed'"))
        .scalar_one()
    )
    if trashed_count:
        raise RuntimeError(
            "Cannot downgrade inbox lifecycle while the Inbox trash is non-empty; "
            "restore trashed rows first."
        )
    op.drop_index("ix_inbox_items_user_status_id", table_name="inbox_items")
    op.drop_index("ix_inbox_items_trashed_at", table_name="inbox_items")
    with op.batch_alter_table("inbox_items") as batch:
        batch.drop_constraint("ck_inbox_item_trash_state", type_="check")
        batch.drop_constraint("ck_inbox_item_version", type_="check")
        batch.drop_column("pre_trash_status")
        batch.drop_column("trashed_at")
        batch.drop_column("version")
