from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, func, select

from future_self.models import NovaMemoryChange, NovaMemoryItem, User
from future_self.nova_memory import (
    NovaMemoryService,
    NovaMemoryValidationError,
    normalize_nova_memory_content,
    nova_memory_fingerprint,
)


async def _add_user(
    db,
    telegram_id: int,
    *,
    tier: str = "subscriber",
    access_version: int = 1,
) -> User:
    async with db.session() as session:
        user = User(
            telegram_id=telegram_id,
            access_tier=tier,
            access_version=access_version,
            onboarding_completed=True,
        )
        session.add(user)
        await session.flush()
        return user


async def _create(
    service: NovaMemoryService,
    telegram_id: int,
    content: str,
    *,
    category: str = "about_me",
    important: bool = False,
    access_version: int = 1,
):
    return await service.create(
        telegram_actor_id=telegram_id,
        expected_access_version=access_version,
        category=category,
        content=content,
        important=important,
    )


async def _collection_revision(service: NovaMemoryService, telegram_id: int) -> str:
    status = await service.status(telegram_actor_id=telegram_id)
    assert status.status == "available"
    assert status.collection_revision is not None
    return status.collection_revision


def test_normalization_fingerprint_and_privacy_safe_validation():
    assert normalize_nova_memory_content(" \tＦｏｏ\n  bar  ") == "Foo bar"
    assert nova_memory_fingerprint("Foo bar") == nova_memory_fingerprint("foo BAR")
    assert len(nova_memory_fingerprint("Foo bar")) == 64

    private = "PRIVATE_MEMORY_SENTINEL"
    with pytest.raises(NovaMemoryValidationError) as invalid_control:
        normalize_nova_memory_content(f"{private}\x00")
    assert private not in str(invalid_control.value)
    with pytest.raises(NovaMemoryValidationError):
        normalize_nova_memory_content("")
    with pytest.raises(NovaMemoryValidationError):
        normalize_nova_memory_content("x" * 501)
    with pytest.raises(NovaMemoryValidationError):
        normalize_nova_memory_content(7)  # type: ignore[arg-type]


async def test_create_duplicate_is_noop_and_audit_is_metadata_only(db, caplog):
    telegram_id = 71_001
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    private = "PRIVATE_MEMORY_SENTINEL"

    created = await _create(
        service,
        telegram_id,
        f"  {private}\n likes   tea ",
        category="about_me",
    )
    duplicate = await _create(
        service,
        telegram_id,
        f"{private.casefold()} LIKES TEA",
        category="orientation",
        important=True,
    )

    assert created.status == "created"
    assert created.item is not None
    assert created.item.content == f"{private} likes tea"
    assert created.item.version == 1
    assert str(UUID(created.item.public_id)) == created.item.public_id
    assert duplicate.status == "duplicate"
    assert duplicate.item is not None
    assert duplicate.item.public_id == created.item.public_id
    assert duplicate.item.category == "about_me"
    assert duplicate.item.important is False
    assert private not in repr(created)
    assert private.casefold() not in repr(duplicate).casefold()
    assert private.casefold() not in caplog.text.casefold()

    async with db.sessions() as session:
        items = list((await session.scalars(select(NovaMemoryItem))).all())
        changes = list((await session.scalars(select(NovaMemoryChange))).all())
    assert len(items) == 1
    assert [(change.operation, change.resulting_version) for change in changes] == [("created", 1)]
    assert "content" not in NovaMemoryChange.__table__.columns
    assert "content_fingerprint" not in NovaMemoryChange.__table__.columns


async def test_owner_isolation_and_fail_closed_access(db):
    subscriber_id = 71_010
    admin_id = 71_011
    guest_id = 71_012
    blocked_id = 71_013
    await _add_user(db, subscriber_id)
    await _add_user(db, admin_id, tier="admin")
    await _add_user(db, guest_id, tier="guest")
    await _add_user(db, blocked_id, tier="blocked")
    service = NovaMemoryService(db)
    created = await _create(service, subscriber_id, "Only the owner may read this")
    assert created.item is not None

    assert (
        await service.get(
            telegram_actor_id=admin_id,
            public_id=created.item.public_id,
        )
    ).status == "not_found"
    assert (
        await service.update(
            telegram_actor_id=admin_id,
            public_id=created.item.public_id,
            expected_version=1,
            expected_access_version=1,
            content="Forged cross-owner update",
        )
    ).status == "not_found"
    admin_own = await _create(service, admin_id, "Only the owner may read this")
    assert admin_own.status == "created"
    assert admin_own.item is not None
    assert admin_own.item.public_id != created.item.public_id
    assert (
        await service.get(
            telegram_actor_id=guest_id,
            public_id=created.item.public_id,
        )
    ).status == "access_denied"
    assert (await service.list(telegram_actor_id=blocked_id)).status == "access_denied"
    assert (await service.status(telegram_actor_id=999_999)).status == "access_denied"
    assert (await service.status(telegram_actor_id=admin_id)).status == "available"


async def test_downgrade_preserves_memory_resubscribe_restores_reads_and_old_fence_stales(db):
    telegram_id = 71_020
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    created = await _create(service, telegram_id, "Durable through access changes")
    assert created.item is not None

    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "guest"
        user.access_version = 2
    assert (
        await service.get(telegram_actor_id=telegram_id, public_id=created.item.public_id)
    ).status == "access_denied"
    denied = await service.set_important(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        important=True,
    )
    assert denied.status == "access_denied"

    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "subscriber"
        user.access_version = 3
    assert (
        await service.get(telegram_actor_id=telegram_id, public_id=created.item.public_id)
    ).status == "found"
    stale = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        content="Must not apply",
    )
    assert stale.status == "stale_access"
    changed = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=3,
        category="orientation",
    )
    assert changed.status == "updated"
    assert changed.item is not None and changed.item.version == 2


class _PausingPreflightService(NovaMemoryService):
    def __init__(self, db):
        super().__init__(db)
        self.preflight_complete = asyncio.Event()
        self.continue_to_lock = asyncio.Event()

    async def _preflight_mutation(self, telegram_actor_id, expected_access_version):
        result = await super()._preflight_mutation(telegram_actor_id, expected_access_version)
        self.preflight_complete.set()
        await self.continue_to_lock.wait()
        return result


class _PausingReadFenceService(NovaMemoryService):
    def __init__(self, db):
        super().__init__(db)
        self.payload_materialized = asyncio.Event()
        self.continue_to_final_fence = asyncio.Event()

    async def _before_read_generation_check(self, telegram_actor_id, actor):
        del telegram_actor_id, actor
        self.payload_materialized.set()
        await self.continue_to_final_fence.wait()


class _PausingOwnerLockService(NovaMemoryService):
    def __init__(self, db):
        super().__init__(db)
        self.owner_locked = asyncio.Event()
        self.release_owner_lock = asyncio.Event()

    async def _lock_actor(self, session, telegram_actor_id, expected_access_version):
        result = await super()._lock_actor(
            session,
            telegram_actor_id,
            expected_access_version,
        )
        self.owner_locked.set()
        await self.release_owner_lock.wait()
        return result


class _SignalingOwnerLockService(NovaMemoryService):
    def __init__(self, db):
        super().__init__(db)
        self.owner_lock_attempted = asyncio.Event()

    async def _lock_actor(self, session, telegram_actor_id, expected_access_version):
        self.owner_lock_attempted.set()
        return await super()._lock_actor(
            session,
            telegram_actor_id,
            expected_access_version,
        )


async def test_access_version_bounce_between_preflight_and_owner_lock_is_stale(db):
    telegram_id = 71_021
    await _add_user(db, telegram_id)
    baseline = NovaMemoryService(db)
    created = await _create(baseline, telegram_id, "Access-fenced memory")
    assert created.item is not None
    service = _PausingPreflightService(db)

    mutation = asyncio.create_task(
        service.update(
            telegram_actor_id=telegram_id,
            public_id=created.item.public_id,
            expected_version=1,
            expected_access_version=1,
            content="PRIVATE_STALE_MUTATION",
        )
    )
    await service.preflight_complete.wait()
    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "subscriber"
        user.access_version = 3
    service.continue_to_lock.set()

    assert (await mutation).status == "stale_access"
    stored = await baseline.get(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
    )
    assert stored.item is not None and stored.item.content == "Access-fenced memory"


async def test_get_discards_materialized_item_after_access_downgrade(db):
    telegram_id = 71_022
    await _add_user(db, telegram_id)
    baseline = NovaMemoryService(db)
    created = await _create(baseline, telegram_id, "PRIVATE_GET_RACE_MEMORY")
    assert created.item is not None
    service = _PausingReadFenceService(db)

    read = asyncio.create_task(
        service.get(
            telegram_actor_id=telegram_id,
            public_id=created.item.public_id,
        )
    )
    await service.payload_materialized.wait()
    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "guest"
        user.access_version = 2
    service.continue_to_final_fence.set()

    result = await read
    assert result.status == "access_denied"
    assert result.item is None


async def test_list_discards_materialized_page_after_access_generation_bounce(db):
    telegram_id = 71_023
    await _add_user(db, telegram_id)
    baseline = NovaMemoryService(db)
    await _create(baseline, telegram_id, "PRIVATE_LIST_RACE_MEMORY")
    service = _PausingReadFenceService(db)

    read = asyncio.create_task(service.list(telegram_actor_id=telegram_id))
    await service.payload_materialized.wait()
    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "guest"
        user.access_version = 2
    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "subscriber"
        user.access_version = 3
    service.continue_to_final_fence.set()

    result = await read
    assert result.status == "access_denied"
    assert result.items == ()
    assert result.total == 0
    assert result.next_offset is None
    assert result.collection_revision is None


async def test_status_discards_materialized_counts_after_allowed_tier_change(db):
    telegram_id = 71_024
    await _add_user(db, telegram_id)
    baseline = NovaMemoryService(db)
    await _create(baseline, telegram_id, "PRIVATE_STATUS_RACE_MEMORY")
    service = _PausingReadFenceService(db)

    read = asyncio.create_task(service.status(telegram_actor_id=telegram_id))
    await service.payload_materialized.wait()
    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "admin"
    service.continue_to_final_fence.set()

    result = await read
    assert result.status == "access_denied"
    assert result.count == 0
    assert result.remaining == 0
    assert result.access_version is None
    assert result.collection_revision is None


async def test_get_list_and_status_read_fences_are_select_only(db):
    telegram_id = 71_025
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    created = await _create(service, telegram_id, "Read without DML")
    assert created.item is not None
    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", capture_statement)
    try:
        assert (
            await service.get(
                telegram_actor_id=telegram_id,
                public_id=created.item.public_id,
            )
        ).status == "found"
        assert (await service.list(telegram_actor_id=telegram_id)).status == "ok"
        assert (await service.status(telegram_actor_id=telegram_id)).status == "available"
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture_statement)

    assert statements
    assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)


async def test_version_fences_noops_and_real_transitions_write_exact_audit(db):
    telegram_id = 71_030
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    created = await _create(service, telegram_id, "Work calmly")
    assert created.item is not None

    unchanged = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        content="  Work   calmly ",
        category="about_me",
    )
    unimportant = await service.set_important(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        important=False,
    )
    moved = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        category="interaction",
    )
    assert moved.item is not None
    important = await service.set_important(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=2,
        expected_access_version=1,
        important=True,
    )
    stale = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=2,
        expected_access_version=1,
        content="Stale content",
    )

    assert unchanged.status == unimportant.status == "unchanged"
    assert unchanged.item is not None and unchanged.item.version == 1
    assert moved.status == "updated" and moved.item.version == 2
    assert important.status == "importance_changed"
    assert important.item is not None and important.item.version == 3
    assert stale.status == "stale"
    async with db.sessions() as session:
        changes = list(
            (await session.scalars(select(NovaMemoryChange).order_by(NovaMemoryChange.id))).all()
        )
    assert [(row.operation, row.resulting_version) for row in changes] == [
        ("created", 1),
        ("updated", 2),
        ("importance_changed", 3),
    ]


async def test_update_to_another_owner_fingerprint_is_typed_conflict(db):
    telegram_id = 71_031
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    first = await _create(service, telegram_id, "First wording")
    second = await _create(service, telegram_id, "Second wording", category="orientation")
    assert first.item is not None and second.item is not None

    conflict = await service.update(
        telegram_actor_id=telegram_id,
        public_id=first.item.public_id,
        expected_version=1,
        expected_access_version=1,
        content=" second WORDING ",
        category="interaction",
    )
    assert conflict.status == "conflict"
    assert conflict.item is not None
    assert (conflict.item.content, conflict.item.category, conflict.item.version) == (
        "First wording",
        "about_me",
        1,
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 2


async def test_bounded_pagination_category_filter_and_stable_order(db):
    telegram_id = 71_040
    user = await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    values = [
        ("important older", "about_me", True),
        ("important newer", "orientation", True),
        ("ordinary older", "about_me", False),
        ("ordinary newer", "interaction", False),
    ]
    created = [
        await _create(service, telegram_id, content, category=category, important=important)
        for content, category, important in values
    ]
    public_ids = [result.item.public_id for result in created if result.item is not None]
    assert len(public_ids) == 4
    moments = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
        datetime(2026, 1, 4, tzinfo=UTC),
    ]
    async with db.session() as session:
        rows = list(
            (
                await session.scalars(
                    select(NovaMemoryItem)
                    .where(NovaMemoryItem.owner_id == user.id)
                    .order_by(NovaMemoryItem.id)
                )
            ).all()
        )
        for row, moment in zip(rows, moments, strict=True):
            row.updated_at = moment

    first = await service.list(telegram_actor_id=telegram_id, offset=0, limit=2)
    second = await service.list(telegram_actor_id=telegram_id, offset=2, limit=2)
    about = await service.list(telegram_actor_id=telegram_id, category="about_me", limit=10)
    assert [item.content for item in first.items] == ["important newer", "important older"]
    assert first.total == 4 and first.next_offset == 2
    assert [item.content for item in second.items] == ["ordinary newer", "ordinary older"]
    assert second.next_offset is None
    assert [item.content for item in about.items] == ["important older", "ordinary older"]
    with pytest.raises(NovaMemoryValidationError):
        await service.list(telegram_actor_id=telegram_id, limit=51)


async def test_hard_delete_delete_all_and_idempotent_audit(db):
    telegram_id = 71_050
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    created = [await _create(service, telegram_id, f"Memory {index}") for index in range(3)]
    first = created[0].item
    assert first is not None

    removed = await service.delete(
        telegram_actor_id=telegram_id,
        public_id=first.public_id,
        expected_version=1,
        expected_access_version=1,
    )
    repeated = await service.delete(
        telegram_actor_id=telegram_id,
        public_id=first.public_id,
        expected_version=1,
        expected_access_version=1,
    )
    delete_all_revision = await _collection_revision(service, telegram_id)
    all_removed = await service.delete_all(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        expected_collection_revision=delete_all_revision,
    )
    replay = await service.delete_all(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        expected_collection_revision=delete_all_revision,
    )
    empty_revision = await _collection_revision(service, telegram_id)
    empty = await service.delete_all(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        expected_collection_revision=empty_revision,
    )
    assert (removed.status, removed.affected_count) == ("deleted", 1)
    assert repeated.status == "not_found"
    assert (all_removed.status, all_removed.affected_count) == ("deleted_all", 2)
    assert (replay.status, replay.affected_count) == ("stale", 0)
    assert empty.status == "unchanged"

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        changes = list(
            (await session.scalars(select(NovaMemoryChange).order_by(NovaMemoryChange.id))).all()
        )
    assert [row.operation for row in changes] == [
        "created",
        "created",
        "created",
        "deleted",
        "deleted_all",
    ]
    assert changes[-2].memory_public_id == first.public_id
    assert changes[-2].resulting_version is None
    assert changes[-1].memory_public_id is None
    assert changes[-1].category is None
    assert changes[-1].affected_count == 2


async def test_collection_revision_changes_for_every_real_item_mutation(db):
    telegram_id = 71_051
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    empty_revision = await _collection_revision(service, telegram_id)

    created = await _create(service, telegram_id, "Revision transitions")
    assert created.item is not None
    created_revision = await _collection_revision(service, telegram_id)
    unchanged = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        content="Revision transitions",
    )
    unchanged_revision = await _collection_revision(service, telegram_id)
    updated = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        content="Revision transitions updated",
    )
    assert updated.item is not None
    updated_revision = await _collection_revision(service, telegram_id)
    important = await service.set_important(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=2,
        expected_access_version=1,
        important=True,
    )
    assert important.item is not None
    important_revision = await _collection_revision(service, telegram_id)
    removed = await service.delete(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=3,
        expected_access_version=1,
    )
    deleted_revision = await _collection_revision(service, telegram_id)

    assert unchanged.status == "unchanged"
    assert removed.status == "deleted"
    assert unchanged_revision == created_revision
    assert len({created_revision, updated_revision, important_revision}) == 3
    assert created_revision != empty_revision
    assert deleted_revision == empty_revision


async def test_collection_revision_uses_only_public_ids_and_versions(db):
    telegram_id = 71_052
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    created = await _create(service, telegram_id, "PRIVATE_REVISION_CONTENT_A")
    assert created.item is not None
    before = await _collection_revision(service, telegram_id)

    async with db.session() as session:
        item = await session.scalar(
            select(NovaMemoryItem).where(NovaMemoryItem.public_id == created.item.public_id)
        )
        assert item is not None
        item.content = "PRIVATE_REVISION_CONTENT_B"
        item.content_fingerprint = nova_memory_fingerprint(item.content)
    after = await _collection_revision(service, telegram_id)

    assert before == after
    assert len(before) == 64
    assert "PRIVATE" not in before


async def test_filtered_and_paged_list_exposes_whole_collection_revision(db):
    telegram_id = 71_053
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    await _create(service, telegram_id, "About", category="about_me")
    await _create(service, telegram_id, "Interaction", category="interaction")
    await _create(service, telegram_id, "Orientation", category="orientation")

    expected = await _collection_revision(service, telegram_id)
    filtered = await service.list(
        telegram_actor_id=telegram_id,
        category="about_me",
        limit=1,
    )
    paged = await service.list(telegram_actor_id=telegram_id, offset=1, limit=1)

    assert len(filtered.items) == len(paged.items) == 1
    assert filtered.collection_revision == expected
    assert paged.collection_revision == expected


async def test_delete_all_rejects_new_item_after_preview_without_delete_or_audit(db):
    telegram_id = 71_054
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    await _create(service, telegram_id, "Previewed item")
    preview_revision = await _collection_revision(service, telegram_id)
    await _create(service, telegram_id, "Created after preview")

    result = await service.delete_all(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        expected_collection_revision=preview_revision,
    )

    assert (result.status, result.affected_count) == ("stale", 0)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 2
        assert (
            await session.scalar(
                select(func.count(NovaMemoryChange.id)).where(
                    NovaMemoryChange.operation == "deleted_all"
                )
            )
            == 0
        )


async def test_delete_all_rejects_edit_after_preview_without_delete_or_audit(db):
    telegram_id = 71_055
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    created = await _create(service, telegram_id, "Preview then edit")
    assert created.item is not None
    preview_revision = await _collection_revision(service, telegram_id)
    changed = await service.update(
        telegram_actor_id=telegram_id,
        public_id=created.item.public_id,
        expected_version=1,
        expected_access_version=1,
        content="Edited after preview",
    )
    assert changed.status == "updated"

    result = await service.delete_all(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        expected_collection_revision=preview_revision,
    )

    assert (result.status, result.affected_count) == ("stale", 0)
    assert (await service.status(telegram_actor_id=telegram_id)).count == 1
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(NovaMemoryChange.id)).where(
                    NovaMemoryChange.operation == "deleted_all"
                )
            )
            == 0
        )


async def test_delete_all_rejects_delete_create_same_count_after_preview(db):
    telegram_id = 71_056
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    first = await _create(service, telegram_id, "First preview item")
    await _create(service, telegram_id, "Second preview item")
    assert first.item is not None
    preview_revision = await _collection_revision(service, telegram_id)
    assert (
        await service.delete(
            telegram_actor_id=telegram_id,
            public_id=first.item.public_id,
            expected_version=1,
            expected_access_version=1,
        )
    ).status == "deleted"
    await _create(service, telegram_id, "Replacement with same count")

    result = await service.delete_all(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        expected_collection_revision=preview_revision,
    )

    assert (result.status, result.affected_count) == ("stale", 0)
    assert (await service.status(telegram_actor_id=telegram_id)).count == 2


async def test_concurrent_duplicate_create_has_one_item_and_one_audit(db):
    telegram_id = 71_060
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)

    results = await asyncio.gather(
        _create(service, telegram_id, "Same concurrent memory"),
        _create(service, telegram_id, "same CONCURRENT memory", category="orientation"),
    )
    assert {result.status for result in results} == {"created", "duplicate"}
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1


async def test_delete_all_owner_lock_serializes_concurrent_create_to_next_collection(db):
    telegram_id = 71_062
    await _add_user(db, telegram_id)
    baseline = NovaMemoryService(db)
    await _create(baseline, telegram_id, "Collection before delete-all")
    preview_revision = await _collection_revision(baseline, telegram_id)
    deleting = _PausingOwnerLockService(db)
    creating = _SignalingOwnerLockService(db)

    deletion = asyncio.create_task(
        deleting.delete_all(
            telegram_actor_id=telegram_id,
            expected_access_version=1,
            expected_collection_revision=preview_revision,
        )
    )
    await deleting.owner_locked.wait()
    creation = asyncio.create_task(
        _create(creating, telegram_id, "Concurrent next collection item")
    )
    await creating.owner_lock_attempted.wait()
    deleting.release_owner_lock.set()
    deleted, created = await asyncio.gather(deletion, creation)

    assert (deleted.status, deleted.affected_count) == ("deleted_all", 1)
    assert (created.status, created.affected_count) == ("created", 1)
    page = await baseline.list(telegram_actor_id=telegram_id)
    assert [item.content for item in page.items] == ["Concurrent next collection item"]
    async with db.sessions() as session:
        operations = list(
            await session.scalars(select(NovaMemoryChange.operation).order_by(NovaMemoryChange.id))
        )
    assert operations == ["created", "deleted_all", "created"]


async def test_limit_one_hundred_is_race_safe(db):
    telegram_id = 71_061
    user = await _add_user(db, telegram_id)
    async with db.session() as session:
        session.add_all(
            NovaMemoryItem(
                public_id=str(uuid4()),
                owner_id=user.id,
                category="about_me",
                content=f"Existing memory {index}",
                content_fingerprint=nova_memory_fingerprint(f"Existing memory {index}"),
                important=False,
                version=1,
            )
            for index in range(99)
        )
    service = NovaMemoryService(db)

    results = await asyncio.gather(
        _create(service, telegram_id, "Concurrent candidate A"),
        _create(service, telegram_id, "Concurrent candidate B"),
    )
    assert sorted(result.status for result in results) == ["created", "limit_reached"]
    status = await service.status(telegram_actor_id=telegram_id)
    assert (status.count, status.remaining) == (100, 0)
    duplicate_at_limit = await _create(service, telegram_id, "existing MEMORY 0")
    assert duplicate_at_limit.status == "duplicate"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 100
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1


async def test_concurrent_update_delete_has_one_fenced_winner(db):
    telegram_id = 71_070
    await _add_user(db, telegram_id)
    service = NovaMemoryService(db)
    created = await _create(service, telegram_id, "Raced item")
    assert created.item is not None

    updated, removed = await asyncio.gather(
        service.update(
            telegram_actor_id=telegram_id,
            public_id=created.item.public_id,
            expected_version=1,
            expected_access_version=1,
            content="Updated item",
        ),
        service.delete(
            telegram_actor_id=telegram_id,
            public_id=created.item.public_id,
            expected_version=1,
            expected_access_version=1,
        ),
    )
    winners = [result for result in (updated, removed) if result.status in {"updated", "deleted"}]
    assert len(winners) == 1
    loser = removed if winners[0] is updated else updated
    assert loser.status in {"stale", "not_found"}
    async with db.sessions() as session:
        operations = list(
            await session.scalars(select(NovaMemoryChange.operation).order_by(NovaMemoryChange.id))
        )
    assert operations[0] == "created"
    assert len(operations) == 2


def test_public_crud_is_actor_bound_and_mutations_require_access_version():
    read_methods = (NovaMemoryService.get, NovaMemoryService.list, NovaMemoryService.count)
    mutation_methods = (
        NovaMemoryService.create,
        NovaMemoryService.update,
        NovaMemoryService.set_important,
        NovaMemoryService.delete,
        NovaMemoryService.delete_all,
    )
    for method in (*read_methods, *mutation_methods):
        parameters = inspect.signature(method).parameters
        assert "telegram_actor_id" in parameters
        assert "owner_id" not in parameters
    for method in mutation_methods:
        assert "expected_access_version" in inspect.signature(method).parameters
    delete_all_parameters = inspect.signature(NovaMemoryService.delete_all).parameters
    assert "expected_collection_revision" in delete_all_parameters
    assert delete_all_parameters["expected_collection_revision"].default is inspect.Parameter.empty
