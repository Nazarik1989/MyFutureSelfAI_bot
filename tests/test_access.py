import asyncio

import pytest
from sqlalchemy import event, func, select

from future_self.access import (
    ADMIN,
    BLOCKED,
    GUEST,
    SUBSCRIBER,
    AccessService,
    AccessUserNotFound,
    is_full_access_tier,
)
from future_self.db import Database
from future_self.models import AccessTierChange, InboxItem, User
from future_self.repositories import ProfileRepository, UserRepository
from future_self.schemas import VisionSummary


async def create_user(db: Database, telegram_id: int, *, completed: bool = False) -> User:
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(telegram_id, "Europe/Moscow")
        user.onboarding_completed = completed
        user_id = user.id
    async with db.sessions() as session:
        return await session.get(User, user_id)


def test_full_access_tiers_are_explicit_and_ignore_onboarding_state():
    assert is_full_access_tier(SUBSCRIBER)
    assert is_full_access_tier(ADMIN)
    assert not is_full_access_tier(GUEST)
    assert not is_full_access_tier(BLOCKED)


async def test_access_transitions_version_audit_and_idempotency(db):
    await create_user(db, 91001, completed=True)
    service = AccessService(db)

    initial = await service.status(91001)
    assert initial is not None
    assert (initial.access_tier, initial.access_version, initial.onboarding_completed) == (
        GUEST,
        1,
        True,
    )

    subscriber = await service.grant_subscriber(91001)
    assert subscriber.changed
    assert (subscriber.status.access_tier, subscriber.status.access_version) == (SUBSCRIBER, 2)
    no_op_subscriber = await service.grant_subscriber(91001)
    assert not no_op_subscriber.changed
    assert no_op_subscriber.status.access_version == 2

    admin = await service.grant_admin(91001)
    assert admin.changed and admin.status.access_version == 3
    blocked = await service.block(91001)
    assert blocked.changed and blocked.status.access_tier == BLOCKED
    blocked_again = await service.block(91001)
    assert not blocked_again.changed and blocked_again.status.access_version == 4

    guest = await service.unblock(91001)
    assert guest.changed
    assert (guest.status.access_tier, guest.status.access_version) == (GUEST, 5)
    unblocked_again = await service.unblock(91001)
    assert not unblocked_again.changed and unblocked_again.status.access_version == 5
    guest_again = await service.set_guest(91001)
    assert not guest_again.changed and guest_again.status.access_version == 5

    async with db.sessions() as session:
        changes = list(
            (
                await session.scalars(
                    select(AccessTierChange)
                    .join(User, User.id == AccessTierChange.user_id)
                    .where(User.telegram_id == 91001)
                    .order_by(AccessTierChange.id)
                )
            ).all()
        )
    assert [(row.from_tier, row.to_tier) for row in changes] == [
        (GUEST, SUBSCRIBER),
        (SUBSCRIBER, ADMIN),
        (ADMIN, BLOCKED),
        (BLOCKED, GUEST),
    ]
    assert {row.source for row in changes} == {"operator-cli"}


async def test_set_guest_block_and_unblock_preserve_user_data(db):
    user = await create_user(db, 91002, completed=True)
    async with db.session() as session:
        stored = await session.get(User, user.id)
        stored.display_name = "Private profile marker"
        session.add(
            InboxItem(
                user_id=user.id,
                kind="note",
                title="Private item",
                raw_text="Private content marker",
                source="text",
                status="confirmed",
            )
        )

    service = AccessService(db)
    await service.grant_admin(91002)
    await service.set_guest(91002)
    await service.block(91002)
    await service.unblock(91002)

    async with db.sessions() as session:
        stored = await session.scalar(select(User).where(User.telegram_id == 91002))
        item_count = await session.scalar(
            select(func.count(InboxItem.id)).where(InboxItem.user_id == user.id)
        )
    assert stored.access_tier == GUEST
    assert stored.onboarding_completed is True
    assert stored.display_name == "Private profile marker"
    assert item_count == 1


async def test_missing_user_mutations_do_not_create_user(db):
    service = AccessService(db)
    assert await service.status(999999) is None
    with pytest.raises(AccessUserNotFound):
        await service.grant_subscriber(999999)
    with pytest.raises(AccessUserNotFound):
        await service.unblock(999999)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(User.id))) == 0


async def test_concurrent_access_mutations_do_not_lose_versions(db):
    await create_user(db, 91003)
    first = AccessService(db)
    second = AccessService(db)

    results = await asyncio.gather(
        first.grant_subscriber(91003, source="concurrency:first"),
        second.grant_admin(91003, source="concurrency:second"),
    )

    status = await first.status(91003)
    assert status is not None and status.access_version == 3
    assert all(result.changed for result in results)
    async with db.sessions() as session:
        assert (await session.scalar(select(func.count(AccessTierChange.id)))) == 2


async def test_concurrent_same_tier_mutation_has_one_change(db):
    await create_user(db, 91004)
    service = AccessService(db)
    results = await asyncio.gather(
        service.grant_subscriber(91004, source="concurrency:first"),
        service.grant_subscriber(91004, source="concurrency:second"),
    )
    assert sorted(result.changed for result in results) == [False, True]
    status = await service.status(91004)
    assert status is not None and status.access_version == 2


async def test_concurrent_get_or_create_uses_one_row_and_guest_defaults(db):
    second_db = Database(db.url)

    async def get(database: Database, timezone: str) -> tuple[int, str, int]:
        async with database.session() as session:
            user = await UserRepository(session).get_or_create(92001, timezone)
            return user.id, user.access_tier, user.access_version

    try:
        results = await asyncio.gather(
            get(db, "Europe/Moscow"),
            get(second_db, "UTC"),
            get(db, "Europe/Saratov"),
            get(second_db, "Asia/Yekaterinburg"),
        )
    finally:
        await second_db.dispose()

    assert len({item[0] for item in results}) == 1
    assert {(item[1], item[2]) for item in results} == {(GUEST, 1)}
    async with db.sessions() as session:
        users = list((await session.scalars(select(User).where(User.telegram_id == 92001))).all())
    assert len(users) == 1


async def test_existing_user_get_or_create_uses_read_only_fast_path(db):
    existing = await create_user(db, 92002)
    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement.lstrip().upper())

    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        async with db.session() as session:
            returned = await UserRepository(session).get_or_create(92002, "UTC")
            assert returned.id == existing.id
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    assert any(statement.startswith("SELECT") for statement in statements)
    assert not any(statement.startswith(("INSERT", "UPDATE", "DELETE")) for statement in statements)


async def test_read_only_full_access_checks_cover_all_tiers_without_mutation(db):
    tiers = [GUEST, SUBSCRIBER, ADMIN, BLOCKED]
    user_ids: dict[str, int] = {}
    async with db.session() as session:
        for offset, tier in enumerate(tiers, start=1):
            user = await UserRepository(session).get_or_create(92100 + offset, "Europe/Moscow")
            user.access_tier = tier
            user_ids[tier] = user.id

    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement.lstrip().upper())

    service = AccessService(db)
    event.listen(db.engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        assert not await service.has_full_access_by_user_id(user_ids[GUEST])
        assert await service.has_full_access_by_user_id(user_ids[SUBSCRIBER])
        assert await service.has_full_access_by_user_id(user_ids[ADMIN])
        assert not await service.has_full_access_by_user_id(user_ids[BLOCKED])
        assert not await service.has_full_access_by_user_id(999_999)
        assert not await service.has_full_access_by_telegram_id(92101)
        assert await service.has_full_access_by_telegram_id(92102)
        assert await service.has_full_access_by_telegram_id(92103)
        assert not await service.has_full_access_by_telegram_id(92104)
        assert not await service.has_full_access_by_telegram_id(999_999)
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_statement)

    assert statements and all(statement.startswith("SELECT") for statement in statements)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(AccessTierChange.id))) == 0
        assert set(await session.scalars(select(User.access_version))) == {1}


async def test_read_only_full_access_check_propagates_database_errors(db, monkeypatch):
    class BrokenSession:
        async def __aenter__(self):
            raise RuntimeError("synthetic database failure")

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(db, "sessions", lambda: BrokenSession())
    with pytest.raises(RuntimeError, match="synthetic database failure"):
        await AccessService(db).has_full_access_by_telegram_id(92105)


async def test_existing_user_is_returned_without_changing_access(db):
    await create_user(db, 92003)
    service = AccessService(db)
    await service.grant_admin(92003)
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(92003, "UTC")
        assert (user.access_tier, user.access_version, user.timezone) == (
            ADMIN,
            2,
            "Europe/Moscow",
        )


async def test_profile_completion_does_not_upgrade_guest(db):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(92004, "UTC")
        await ProfileRepository(session).upsert(
            user,
            {"future_life": "Calm future"},
            VisionSummary(summary="Calm future"),
        )
    status = await AccessService(db).status(92004)
    assert status is not None
    assert status.onboarding_completed is True
    assert status.access_tier == GUEST
