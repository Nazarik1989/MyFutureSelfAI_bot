"""Add canonical task state, action capabilities, and reminder versions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import sqlalchemy as sa

from alembic import op

revision: str = "20260720_0016"
down_revision: str | None = "20260720_0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _zone(value: object, fallback: object) -> str:
    for candidate in (value, fallback, "UTC"):
        if not isinstance(candidate, str) or not candidate:
            continue
        try:
            ZoneInfo(candidate)
        except ZoneInfoNotFoundError:
            continue
        return candidate
    return "UTC"


def _datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _temporal(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def upgrade() -> None:
    with op.batch_alter_table("inbox_items") as batch:
        batch.create_unique_constraint("uq_inbox_item_id_user", ["id", "user_id"])
    with op.batch_alter_table("task_reminders") as batch:
        batch.add_column(
            sa.Column("task_version", sa.Integer(), nullable=False, server_default="1")
        )

    op.create_table(
        "task_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("inbox_item_id", sa.Integer(), nullable=False, unique=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("owner_id", "inbox_item_id", name="uq_task_state_owner_item"),
        sa.ForeignKeyConstraint(
            ["inbox_item_id", "owner_id"],
            ["inbox_items.id", "inbox_items.user_id"],
            ondelete="CASCADE",
            name="fk_task_state_inbox_owner",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'completed', 'cancelled')",
            name="ck_task_state_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_task_state_version"),
    )
    op.create_index("ix_task_states_owner_id", "task_states", ["owner_id"])
    op.create_index("ix_task_states_inbox_item_id", "task_states", ["inbox_item_id"])
    op.create_index("ix_task_states_status", "task_states", ["status"])
    op.create_index("ix_task_states_event_at", "task_states", ["event_at"])

    op.create_table(
        "task_action_tokens",
        sa.Column("token", sa.String(32), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("inbox_item_id", sa.Integer(), nullable=False),
        sa.Column("task_version", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id", "inbox_item_id"],
            ["task_states.owner_id", "task_states.inbox_item_id"],
            ondelete="CASCADE",
            name="fk_task_action_state_owner",
        ),
        sa.CheckConstraint("task_version > 0", name="ck_task_action_version"),
        sa.CheckConstraint(
            "status IN ('pending', 'awaiting_input', 'consumed')",
            name="ck_task_action_status",
        ),
    )
    op.create_index("ix_task_action_tokens_owner_id", "task_action_tokens", ["owner_id"])
    op.create_index("ix_task_action_tokens_chat_id", "task_action_tokens", ["chat_id"])
    op.create_index(
        "ix_task_action_tokens_inbox_item_id",
        "task_action_tokens",
        ["inbox_item_id"],
    )
    op.create_index("ix_task_action_tokens_action", "task_action_tokens", ["action"])
    op.create_index("ix_task_action_tokens_status", "task_action_tokens", ["status"])
    op.create_index("ix_task_action_tokens_expires_at", "task_action_tokens", ["expires_at"])

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            """
            SELECT i.id AS inbox_item_id,
                   i.user_id AS owner_id,
                   i.resolved_date AS resolved_date,
                   i.temporal_resolution AS temporal_resolution,
                   u.timezone AS owner_timezone,
                   r.event_at AS reminder_event_at,
                   r.timezone AS reminder_timezone
              FROM inbox_items AS i
              JOIN users AS u ON u.id = i.user_id
              LEFT JOIN task_reminders AS r ON r.inbox_item_id = i.id
             WHERE i.kind = 'task'
             ORDER BY i.id
            """
        )
    ).mappings()
    task_states = sa.table(
        "task_states",
        sa.column("owner_id", sa.Integer()),
        sa.column("inbox_item_id", sa.Integer()),
        sa.column("status", sa.String()),
        sa.column("event_at", sa.DateTime(timezone=True)),
        sa.column("timezone", sa.String()),
        sa.column("version", sa.Integer()),
    )
    payload: list[dict[str, object]] = []
    for row in rows:
        temporal = _temporal(row["temporal_resolution"])
        timezone = _zone(
            row["reminder_timezone"] or temporal.get("timezone"),
            row["owner_timezone"],
        )
        event_at = _datetime(row["reminder_event_at"])
        if event_at is None:
            event_at = _datetime(temporal.get("resolved_at"))
        resolved_date = _date(row["resolved_date"])
        if event_at is None and resolved_date is not None:
            event_at = datetime.combine(
                resolved_date,
                time(hour=9),
                tzinfo=ZoneInfo(timezone),
            ).astimezone(UTC)
        payload.append(
            {
                "owner_id": row["owner_id"],
                "inbox_item_id": row["inbox_item_id"],
                "status": "active",
                "event_at": event_at,
                "timezone": timezone,
                "version": 1,
            }
        )
    if payload:
        connection.execute(task_states.insert(), payload)


def downgrade() -> None:
    op.drop_index("ix_task_action_tokens_expires_at", table_name="task_action_tokens")
    op.drop_index("ix_task_action_tokens_status", table_name="task_action_tokens")
    op.drop_index("ix_task_action_tokens_action", table_name="task_action_tokens")
    op.drop_index("ix_task_action_tokens_inbox_item_id", table_name="task_action_tokens")
    op.drop_index("ix_task_action_tokens_chat_id", table_name="task_action_tokens")
    op.drop_index("ix_task_action_tokens_owner_id", table_name="task_action_tokens")
    op.drop_table("task_action_tokens")
    op.drop_index("ix_task_states_event_at", table_name="task_states")
    op.drop_index("ix_task_states_status", table_name="task_states")
    op.drop_index("ix_task_states_inbox_item_id", table_name="task_states")
    op.drop_index("ix_task_states_owner_id", table_name="task_states")
    op.drop_table("task_states")
    with op.batch_alter_table("task_reminders") as batch:
        batch.drop_column("task_version")
    with op.batch_alter_table("inbox_items") as batch:
        batch.drop_constraint("uq_inbox_item_id_user", type_="unique")
