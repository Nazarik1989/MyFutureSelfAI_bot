import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from future_self.access import ADMIN, SUBSCRIBER, AccessService
from future_self.db import Database
from future_self.models import AccessTierChange, Base, User
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
    schema = f"access_contract_{uuid4().hex}"
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


async def test_postgresql_get_or_create_and_access_locking_contract(postgres_db):
    async def create() -> int:
        async with postgres_db.session() as session:
            user = await UserRepository(session).get_or_create(81001, "UTC")
            return user.id

    ids = await asyncio.gather(*(create() for _ in range(4)))
    assert len(set(ids)) == 1

    first = AccessService(postgres_db)
    second = AccessService(postgres_db)
    results = await asyncio.gather(
        first.grant_subscriber(81001, source="postgres:first"),
        second.grant_admin(81001, source="postgres:second"),
    )
    assert all(result.changed for result in results)
    status = await first.status(81001)
    assert status is not None
    assert status.access_tier in {SUBSCRIBER, ADMIN}
    assert status.access_version == 3
    async with postgres_db.sessions() as session:
        assert await session.scalar(select(func.count(User.id))) == 1
        assert await session.scalar(select(func.count(AccessTierChange.id))) == 2
