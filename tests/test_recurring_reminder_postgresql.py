import os
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, time
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from future_self.db import Database
from future_self.models import (
    Base,
    InboxItem,
    RecurringTaskReminderOccurrence,
    RecurringTaskReminderSchedule,
    User,
)

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL is not configured for the PostgreSQL contract suite",
)


def asyncpg_url(value: str) -> str:
    clean = value.strip()
    if clean.startswith("postgresql://"):
        return clean.replace("postgresql://", "postgresql+asyncpg://", 1)
    return clean


@pytest_asyncio.fixture
async def postgres_db() -> AsyncIterator[Database]:
    url = asyncpg_url(TEST_POSTGRES_URL or "")
    assert make_url(url).get_backend_name() == "postgresql"
    schema = f"recurring_reminder_contract_{uuid4().hex}"
    admin_engine = create_async_engine(url)
    async with admin_engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    database = Database(url)
    await database.engine.dispose()
    database.engine = create_async_engine(
        url,
        connect_args={"server_settings": {"search_path": f"{schema},public"}},
    )
    database.sessions = async_sessionmaker(database.engine, expire_on_commit=False)
    async with database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield database
    finally:
        await database.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin_engine.dispose()


async def create_owner_and_tasks(postgres_db: Database) -> tuple[int, int, int, int]:
    async with postgres_db.session() as session:
        owner = User(telegram_id=99_101, timezone="UTC", onboarding_completed=True)
        other_owner = User(telegram_id=99_102, timezone="UTC", onboarding_completed=True)
        session.add_all([owner, other_owner])
        await session.flush()
        first_task = InboxItem(
            user_id=owner.id,
            kind="task",
            title="PostgreSQL recurring task one",
            raw_text="PostgreSQL recurring task one",
            source="text",
            status="confirmed",
            version=1,
        )
        second_task = InboxItem(
            user_id=owner.id,
            kind="task",
            title="PostgreSQL recurring task two",
            raw_text="PostgreSQL recurring task two",
            source="text",
            status="confirmed",
            version=1,
        )
        session.add_all([first_task, second_task])
        await session.flush()
        return owner.id, other_owner.id, first_task.id, second_task.id


def schedule(*, owner_id: int, inbox_item_id: int, hour: int) -> RecurringTaskReminderSchedule:
    return RecurringTaskReminderSchedule(
        owner_id=owner_id,
        inbox_item_id=inbox_item_id,
        recurrence_kind="daily",
        local_time=time(hour, 30),
        timezone="UTC",
        timezone_source="explicit",
        start_local_date=date(2026, 8, 10),
        next_occurrence_at=datetime(2026, 8, 10, hour, 30, tzinfo=UTC),
        status="active",
        version=1,
    )


def occurrence(
    *,
    schedule_id: int,
    local_date: date,
    scheduled_for: datetime,
    delivery_key: str,
    schedule_version: int = 1,
    status: str = "pending",
    delivery_started_at: datetime | None = None,
    sent_at: datetime | None = None,
) -> RecurringTaskReminderOccurrence:
    return RecurringTaskReminderOccurrence(
        schedule_id=schedule_id,
        schedule_version=schedule_version,
        scheduled_for=scheduled_for,
        local_date=local_date,
        delivery_key=delivery_key,
        status=status,
        delivery_started_at=delivery_started_at,
        sent_at=sent_at,
        next_attempt_at=scheduled_for,
        attempt_count=0,
    )


async def test_postgresql_recurring_owner_and_occurrence_uniqueness_contract(postgres_db):
    owner_id, other_owner_id, first_task_id, second_task_id = await create_owner_and_tasks(
        postgres_db
    )

    with pytest.raises(IntegrityError):
        async with postgres_db.session() as session:
            session.add(schedule(owner_id=other_owner_id, inbox_item_id=first_task_id, hour=8))
            await session.flush()

    async with postgres_db.session() as session:
        first_schedule = schedule(owner_id=owner_id, inbox_item_id=first_task_id, hour=8)
        second_schedule = schedule(owner_id=owner_id, inbox_item_id=second_task_id, hour=9)
        session.add_all([first_schedule, second_schedule])
        await session.flush()
        first_schedule_id = first_schedule.id
        second_schedule_id = second_schedule.id

    first_local_date = date(2026, 8, 10)
    first_scheduled_for = datetime(2026, 8, 10, 8, 30, tzinfo=UTC)
    async with postgres_db.session() as session:
        session.add(
            occurrence(
                schedule_id=first_schedule_id,
                local_date=first_local_date,
                scheduled_for=first_scheduled_for,
                delivery_key="recurring:postgres:first",
            )
        )

    with pytest.raises(IntegrityError):
        async with postgres_db.session() as session:
            session.add(
                occurrence(
                    schedule_id=first_schedule_id,
                    local_date=first_local_date,
                    scheduled_for=datetime(2026, 8, 10, 8, 45, tzinfo=UTC),
                    delivery_key="recurring:postgres:duplicate-date",
                )
            )
            await session.flush()

    async with postgres_db.session() as session:
        session.add(
            occurrence(
                schedule_id=first_schedule_id,
                schedule_version=2,
                local_date=first_local_date,
                scheduled_for=first_scheduled_for,
                delivery_key="recurring:postgres:generation-two",
            )
        )

    sent_at = datetime(2026, 8, 10, 8, 31, tzinfo=UTC)
    async with postgres_db.session() as session:
        session.add(
            occurrence(
                schedule_id=first_schedule_id,
                schedule_version=3,
                local_date=first_local_date,
                scheduled_for=datetime(2026, 8, 10, 8, 50, tzinfo=UTC),
                delivery_key="recurring:postgres:sent-generation-three",
                status="sent",
                delivery_started_at=sent_at,
                sent_at=sent_at,
            )
        )

    with pytest.raises(IntegrityError):
        async with postgres_db.session() as session:
            session.add(
                occurrence(
                    schedule_id=first_schedule_id,
                    schedule_version=4,
                    local_date=first_local_date,
                    scheduled_for=datetime(2026, 8, 10, 8, 55, tzinfo=UTC),
                    delivery_key="recurring:postgres:duplicate-sent-day",
                    status="sent",
                    delivery_started_at=sent_at,
                    sent_at=sent_at,
                )
            )
            await session.flush()

    with pytest.raises(IntegrityError):
        async with postgres_db.session() as session:
            session.add(
                occurrence(
                    schedule_id=second_schedule_id,
                    local_date=date(2026, 8, 11),
                    scheduled_for=datetime(2026, 8, 11, 9, 30, tzinfo=UTC),
                    delivery_key="recurring:postgres:first",
                )
            )
            await session.flush()
