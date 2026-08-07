import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from future_self.db import Database
from future_self.guest_access import (
    GuestDemoKind,
    GuestProviderStartOutcome,
    GuestQuotaDenialReason,
    GuestQuotaPolicy,
    GuestQuotaService,
    GuestSessionOutcome,
    GuestSessionService,
    GuestUsageStatus,
)
from future_self.models import Base, GuestUsageLedger, User
from future_self.repositories import UserRepository

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
    schema = f"guest_quota_contract_{uuid4().hex}"
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


async def create_guest(db: Database, telegram_id: int) -> int:
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(telegram_id, "UTC")
        return user.id


async def prepare_claimed_session(
    db: Database,
    *,
    user_id: int,
    chat_id: int,
    update_id: int,
    now: datetime,
    policy: GuestQuotaPolicy | None = None,
):
    service = GuestSessionService(db, policy)
    started = await service.start_session(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        demo_kind=GuestDemoKind.FIRST_STEP,
        prompt_message_id=update_id + 10_000,
        now=now,
    )
    assert started.outcome is GuestSessionOutcome.STARTED
    claimed = await service.claim_input(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        telegram_update_id=update_id,
        telegram_message_id=update_id + 20_000,
        now=now,
    )
    assert claimed.outcome is GuestSessionOutcome.CLAIMED
    assert claimed.session is not None
    return service, claimed.session


async def test_postgresql_same_user_and_duplicate_idempotency_lock_contract(postgres_db):
    user_id = await create_guest(postgres_db, 98001)
    now = datetime(2026, 8, 6, 10, tzinfo=UTC)
    different_keys = await asyncio.gather(
        *(
            GuestQuotaService(postgres_db).reserve(
                user_id=user_id,
                demo_kind="first_step",
                idempotency_key=f"postgres:race:{index}",
                telegram_update_id=100 + index,
                now=now,
            )
            for index in range(8)
        )
    )
    assert sum(decision.is_new for decision in different_keys) == 1
    assert {decision.denial_reason for decision in different_keys if not decision.is_new} == {
        GuestQuotaDenialReason.IN_PROGRESS
    }

    second_user = await create_guest(postgres_db, 98002)
    duplicate_key = await asyncio.gather(
        *(
            GuestQuotaService(postgres_db).reserve(
                user_id=second_user,
                demo_kind="thought_breakdown",
                idempotency_key="postgres:duplicate",
                telegram_update_id=200,
                now=now,
            )
            for _ in range(8)
        )
    )
    assert sum(decision.is_new for decision in duplicate_key) == 1
    tokens = {
        decision.reservation.reservation_token
        for decision in duplicate_key
        if decision.reservation is not None
    }
    assert len(tokens) == 1
    assert (
        sum(
            decision.denial_reason is GuestQuotaDenialReason.DUPLICATE_RESERVED
            for decision in duplicate_key
        )
        == 7
    )


async def test_postgresql_provider_start_grants_one_permission(postgres_db):
    user_id = await create_guest(postgres_db, 98003)
    now = datetime(2026, 8, 6, 10, 30, tzinfo=UTC)
    service, claimed = await prepare_claimed_session(
        postgres_db,
        user_id=user_id,
        chat_id=98003,
        update_id=300,
        now=now,
    )
    reserved = await GuestQuotaService(postgres_db).reserve(
        user_id=user_id,
        demo_kind=GuestDemoKind.FIRST_STEP,
        idempotency_key="postgres:provider-start",
        telegram_update_id=300,
        now=now,
    )
    assert reserved.reservation is not None
    bound = await service.bind_reservation(
        user_id=user_id,
        chat_id=98003,
        access_version=1,
        session_version=claimed.version,
        reservation_token=reserved.reservation.reservation_token,
        now=now,
    )
    assert bound.outcome is GuestSessionOutcome.BOUND
    assert bound.session is not None

    decisions = await asyncio.gather(
        *(
            GuestSessionService(postgres_db).begin_provider_call(
                reservation_token=reserved.reservation.reservation_token,
                user_id=user_id,
                chat_id=98003,
                access_version=1,
                session_version=bound.session.version,
                now=now,
            )
            for _ in range(8)
        )
    )
    assert sum(decision.can_invoke_provider for decision in decisions) == 1
    assert (
        sum(decision.outcome is GuestProviderStartOutcome.ALREADY_STARTED for decision in decisions)
        == 7
    )
    assert {decision.provider_started_at for decision in decisions} == {now}


async def test_postgresql_fifty_started_failures_still_exhaust_global_day(postgres_db):
    user_ids = [await create_guest(postgres_db, 98100 + index) for index in range(51)]
    now = datetime(2026, 8, 6, 11, tzinfo=UTC)
    policy = GuestQuotaPolicy(global_daily_limit=50)
    prepared = await asyncio.gather(
        *(
            prepare_claimed_session(
                postgres_db,
                user_id=user_id,
                chat_id=98100 + index,
                update_id=1000 + index,
                now=now,
                policy=policy,
            )
            for index, user_id in enumerate(user_ids)
        )
    )
    decisions = await asyncio.gather(
        *(
            GuestQuotaService(postgres_db, policy).reserve(
                user_id=user_id,
                demo_kind="first_step",
                idempotency_key=f"postgres:global:{index}",
                telegram_update_id=1000 + index,
                now=now,
            )
            for index, user_id in enumerate(user_ids)
        )
    )
    assert sum(decision.is_new for decision in decisions) == 50
    assert (
        sum(
            decision.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED
            for decision in decisions
        )
        == 1
    )
    accepted = [
        (index, decision)
        for index, decision in enumerate(decisions)
        if decision.reservation is not None
    ]
    bound = await asyncio.gather(
        *(
            prepared[index][0].bind_reservation(
                user_id=user_ids[index],
                chat_id=98100 + index,
                access_version=1,
                session_version=prepared[index][1].version,
                reservation_token=decision.reservation.reservation_token,
                now=now,
            )
            for index, decision in accepted
        )
    )
    assert all(item.outcome is GuestSessionOutcome.BOUND for item in bound)
    assert all(item.session is not None for item in bound)
    starts = await asyncio.gather(
        *(
            prepared[index][0].begin_provider_call(
                reservation_token=decision.reservation.reservation_token,
                user_id=user_ids[index],
                chat_id=98100 + index,
                access_version=1,
                session_version=bound[position].session.version,
                now=now,
            )
            for position, (index, decision) in enumerate(accepted)
        )
    )
    assert sum(item.can_invoke_provider for item in starts) == 50
    failures = await asyncio.gather(
        *(
            prepared[index][0].fail_processing(
                reservation_token=decision.reservation.reservation_token,
                user_id=user_ids[index],
                chat_id=98100 + index,
                access_version=1,
                session_version=bound[position].session.version,
                now=now,
            )
            for position, (index, decision) in enumerate(accepted)
        )
    )
    assert all(item.outcome.value == "failed" for item in failures)

    denied_index = next(
        index
        for index, decision in enumerate(decisions)
        if decision.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED
    )
    retry = await GuestQuotaService(postgres_db, policy).reserve(
        user_id=user_ids[denied_index],
        demo_kind=GuestDemoKind.FIRST_STEP,
        idempotency_key="postgres:global:retry",
        telegram_update_id=2000,
        now=now,
    )
    assert retry.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED
    async with postgres_db.sessions() as session:
        failed_started = await session.scalar(
            select(func.count(GuestUsageLedger.id)).where(
                GuestUsageLedger.status == GuestUsageStatus.FAILED.value,
                GuestUsageLedger.provider_started_at.is_not(None),
            )
        )
        assert failed_started == 50
        assert await session.scalar(select(func.count(User.id))) == 51
