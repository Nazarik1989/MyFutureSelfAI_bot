import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from future_self.collection_commands import CollectionCommandRouter
from future_self.collections_service import (
    STARTER_TEMPLATES,
    CollectionNameError,
    LifeCollectionService,
    normalize_collection_name,
    split_list_items,
)
from future_self.models import (
    InboxItem,
    LifeCollection,
    LifeCollectionActionToken,
    LifeCollectionContext,
    LifeCollectionLink,
    LifeCollectionPreference,
    TaskState,
)
from future_self.repositories import UserRepository
from future_self.tasks import TaskService


async def owner(db, telegram_id=700001):
    async with db.session() as session:
        return await UserRepository(session).get_or_create(telegram_id, "Europe/Moscow")


async def create_collection(db, *, telegram_id=700001, kind="topic", name="Раздел"):
    user = await owner(db, telegram_id)
    result = await LifeCollectionService(db).create_collection(user.id, kind, name)
    assert result.status == "created"
    return user, result.collection


async def test_migration_equivalent_schema_does_not_seed_existing_users(db):
    user = await owner(db)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LifeCollection.id))) == 0
        assert await session.get(LifeCollectionPreference, user.id) is None


async def test_all_selective_and_empty_onboarding_are_explicit_and_idempotent(db):
    first = await owner(db, 700010)
    service = LifeCollectionService(db)
    all_result = await service.create_starters(
        first.id, tuple(item.key for item in STARTER_TEMPLATES)
    )
    assert all_result.status == "created"
    assert len(all_result.item_ids) == 10
    assert (await service.create_starters(first.id, ("shopping",))).status == "already_completed"

    second = await owner(db, 700011)
    selected = await service.create_starters(second.id, ("shopping", "travel"))
    assert selected.status == "created"
    assert len(selected.item_ids) == 2

    third = await owner(db, 700012)
    assert (await service.complete_empty_onboarding(third.id)).status == "completed"
    assert await service.is_onboarded(third.id)
    assert (await service.create_starters(third.id, ("shopping",))).status == "already_completed"
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(LifeCollection.id)).where(LifeCollection.owner_id == third.id)
            )
            == 0
        )


async def test_deleted_starter_does_not_reappear_and_inbox_content_survives(db):
    user = await owner(db)
    service = LifeCollectionService(db)
    await service.create_starters(user.id, ("shopping",))
    collection = (await service.list_page(user.id, 0)).records[0].collection
    created = await service.create_items(
        user.id, 700001, collection.id, collection.version, ("Чай",), source="text"
    )
    assert created.status == "created_items"
    current = (await service.summary(user.id, collection.id)).collection
    deleted = await service.delete_collection(
        user.id, current.id, current.version, unlink_nonempty=True
    )
    assert deleted.status == "deleted"
    assert (await service.create_starters(user.id, ("shopping",))).status == "already_completed"
    async with db.sessions() as session:
        assert await session.get(InboxItem, created.item_ids[0]) is not None
        assert await session.scalar(select(func.count(LifeCollection.id))) == 0


async def test_crud_normalization_aliases_and_same_name_for_different_owners(db):
    first = await owner(db, 700020)
    second = await owner(db, 700021)
    service = LifeCollectionService(db)
    created = await service.create_collection(first.id, "project", "  Наз   и Войд  ")
    assert created.collection.name == "Наз и Войд"
    assert (await service.create_collection(first.id, "topic", "НАЗ И ВОЙД")).status == "conflict"
    assert (await service.create_collection(second.id, "project", "наз и войд")).status == "created"

    aliased = await service.add_alias(
        first.id, created.collection.id, created.collection.version, "Приложение"
    )
    assert aliased.status == "aliased"
    assert (
        await service.add_alias(
            first.id, created.collection.id, aliased.collection.version, "НАЗ И ВОЙД"
        )
    ).status == "conflict"
    assert (
        await service.rename(
            first.id, created.collection.id, aliased.collection.version, "Приложение"
        )
    ).status == "conflict"
    assert (await service.resolve(first.id, "ПРИЛОЖЕНИЕ")).match.id == created.collection.id
    renamed = await service.rename(
        first.id, created.collection.id, aliased.collection.version, "Новый проект"
    )
    assert renamed.status == "renamed"
    assert (await service.resolve(first.id, "Приложение")).match.id == created.collection.id
    with pytest.raises(CollectionNameError):
        await service.create_collection(first.id, "topic", " " * 10)
    with pytest.raises(CollectionNameError):
        await service.create_collection(first.id, "topic", "Я" * 101)


async def test_list_split_creates_distinct_task_hub_items_without_duplicates(db):
    user, collection = await create_collection(db, kind="list", name="Покупки")
    service = LifeCollectionService(db)
    items, ambiguous = split_list_items("чай, сахар, бетономешалку и остров в Индийском океане")
    assert items == (
        "чай",
        "сахар",
        "бетономешалку",
        "остров в Индийском океане",
    )
    assert not ambiguous
    result = await service.create_items(
        user.id, 700001, collection.id, collection.version, items, source="voice"
    )
    assert result.status == "created_items"
    async with db.sessions() as session:
        inbox_count = await session.scalar(
            select(func.count(InboxItem.id)).where(InboxItem.user_id == user.id)
        )
        state_count = await session.scalar(
            select(func.count(TaskState.id)).where(TaskState.owner_id == user.id)
        )
        link_count = await session.scalar(
            select(func.count(LifeCollectionLink.id)).where(LifeCollectionLink.owner_id == user.id)
        )
    assert inbox_count == state_count == link_count == 4

    task_service = TaskService(db)
    first_id = result.item_ids[0]
    state = await task_service.record(user.id, first_id)
    complete = (
        await task_service.issue_actions(
            user.id, 700001, first_id, state.state.version, ("complete",)
        )
    )["complete"]
    assert (await task_service.complete(complete, user.id, 700001)).status == "completed"
    reopened_state = await task_service.record(user.id, first_id)
    reopen = (
        await task_service.issue_actions(
            user.id, 700001, first_id, reopened_state.state.version, ("reopen",)
        )
    )["reopen"]
    assert (await task_service.reopen(reopen, user.id, 700001)).status == "reopened"
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(TaskState.id)).where(TaskState.inbox_item_id == first_id)
            )
            == 1
        )


async def test_topic_notes_are_not_forced_into_tasks(db):
    user, collection = await create_collection(db, kind="topic", name="Личное")
    result = await LifeCollectionService(db).create_items(
        user.id,
        700001,
        collection.id,
        collection.version,
        ("Наблюдение за неделей", "идея сделать деревянный светильник"),
        source="text",
    )
    async with db.sessions() as session:
        kinds = tuple(
            (
                await session.scalars(
                    select(InboxItem.kind)
                    .where(InboxItem.id.in_(result.item_ids))
                    .order_by(InboxItem.id)
                )
            ).all()
        )
        task_count = await session.scalar(select(func.count(TaskState.id)))
    assert kinds == ("note", "idea")
    assert task_count == 0


async def test_link_move_unlink_and_archive_preserve_canonical_item(db):
    user, first = await create_collection(db, kind="project", name="Ремонт")
    service = LifeCollectionService(db)
    second = (await service.create_collection(user.id, "topic", "Дом")).collection
    created = await service.create_items(
        user.id, 700001, first.id, first.version, ("исправить стену",), source="text"
    )
    item_id = created.item_ids[0]
    first = (await service.summary(user.id, first.id)).collection
    assert (await service.link_item(user.id, second.id, second.version, item_id)).status == "linked"
    second = (await service.summary(user.id, second.id)).collection
    first = (await service.summary(user.id, first.id)).collection
    assert (
        await service.move_item(
            user.id, first.id, first.version, second.id, second.version, item_id
        )
    ).status == "moved"
    second = (await service.summary(user.id, second.id)).collection
    assert (
        await service.unlink_item(user.id, second.id, second.version, item_id)
    ).status == "unlinked"
    async with db.sessions() as session:
        assert await session.get(InboxItem, item_id) is not None
        assert (
            await session.scalar(
                select(func.count(TaskState.id)).where(TaskState.inbox_item_id == item_id)
            )
            == 1
        )

    first = (await service.summary(user.id, first.id)).collection
    archived = await service.set_archived(user.id, first.id, first.version, archived=True)
    assert archived.status == "archived"
    restored = await service.set_archived(
        user.id, first.id, archived.collection.version, archived=False
    )
    assert restored.status == "restored"


async def test_context_is_owner_chat_scoped_restart_safe_and_expires(db):
    user, collection = await create_collection(db)
    service = LifeCollectionService(db, context_ttl=timedelta(seconds=2))
    assert await service.set_context(user.id, 800001, collection.id)
    restarted = LifeCollectionService(db, context_ttl=timedelta(seconds=2))
    assert (await restarted.active_context(user.id, 800001)).collection.id == collection.id
    assert await restarted.active_context(user.id, 800002) is None
    assert (
        await restarted.active_context(
            user.id, 800001, now=datetime.now(UTC) + timedelta(seconds=3)
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LifeCollectionContext.id))) == 0


async def test_concurrent_context_switch_keeps_one_owner_chat_row(db):
    user, first = await create_collection(db, name="Первый")
    service = LifeCollectionService(db)
    second = (await service.create_collection(user.id, "topic", "Второй")).collection
    results = await asyncio.gather(
        service.set_context(user.id, 800003, first.id),
        service.set_context(user.id, 800003, second.id),
    )
    assert results == [True, True]
    active = await service.active_context(user.id, 800003)
    assert active.collection.id in {first.id, second.id}
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(LifeCollectionContext.id)).where(
                    LifeCollectionContext.owner_id == user.id,
                    LifeCollectionContext.chat_id == 800003,
                )
            )
            == 1
        )


async def test_action_tokens_reject_forged_cross_owner_cross_chat_replay_and_stale(db):
    first, collection = await create_collection(db, telegram_id=700030)
    second = await owner(db, 700031)
    service = LifeCollectionService(db)
    token = await service.issue_action(first.id, 810001, "archive", collection=collection)
    assert await service.claim_action("forged", first.id, 810001, {"archive"}) is None
    assert await service.claim_action(token, second.id, 810001, {"archive"}) is None
    assert await service.claim_action(token, first.id, 810002, {"archive"}) is None
    claim = await service.claim_action(token, first.id, 810001, {"archive"})
    assert claim is not None
    assert await service.claim_action(token, first.id, 810001, {"archive"}) is None

    stale = await service.issue_action(first.id, 810001, "archive", collection=collection)
    renamed = await service.rename(first.id, collection.id, collection.version, "Другое имя")
    assert renamed.status == "renamed"
    assert await service.claim_action(stale, first.id, 810001, {"archive"}) is None

    expired = await service.issue_action(first.id, 810001, "noop")
    async with db.session() as session:
        capability = await session.get(LifeCollectionActionToken, expired)
        capability.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert await service.claim_action(expired, first.id, 810001, {"noop"}) is None


async def test_pending_input_is_single_restart_safe_and_not_used_after_ttl(db):
    user, collection = await create_collection(db)
    service = LifeCollectionService(db, input_ttl=timedelta(seconds=2))
    first = await service.issue_action(user.id, 820001, "rename_prompt", collection=collection)
    assert (
        await service.begin_input(
            first,
            user.id,
            820001,
            allowed={"rename_prompt"},
            input_action="input_rename",
        )
        is not None
    )
    restarted = LifeCollectionService(db, input_ttl=timedelta(seconds=2))
    pending = await restarted.pending_input(user.id, 820001)
    assert pending is not None and pending.action == "input_rename"
    async with db.session() as session:
        stored = await session.get(LifeCollectionActionToken, pending.token)
        stored.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert await restarted.pending_input(user.id, 820001) is None
    assert (
        await restarted.claim_action(
            pending.token,
            user.id,
            820001,
            {"input_rename"},
            pending_status="awaiting_input",
        )
        is None
    )


async def test_concurrent_token_claim_has_exactly_one_winner(db):
    user, collection = await create_collection(db)
    service = LifeCollectionService(db)
    token = await service.issue_action(user.id, 820002, "archive", collection=collection)
    claims = await asyncio.gather(
        service.claim_action(token, user.id, 820002, {"archive"}),
        service.claim_action(token, user.id, 820002, {"archive"}),
    )
    assert sum(claim is not None for claim in claims) == 1


async def test_suggestions_are_limited_to_owner_and_ambiguity_does_not_choose_silently(db):
    first = await owner(db, 700040)
    second = await owner(db, 700041)
    service = LifeCollectionService(db)
    await service.create_collection(first.id, "topic", "Дом")
    await service.create_collection(first.id, "project", "Домашние дела")
    await service.create_collection(second.id, "project", "Дом мечты")
    resolution = await service.suggest(first.id, "дом дела")
    assert resolution.match is None
    assert {item.name for item in resolution.candidates} == {"Дом", "Домашние дела"}
    assert "Дом мечты" not in {item.name for item in resolution.candidates}


async def test_concurrent_create_and_versioned_mutations_have_single_winner(db):
    user = await owner(db)
    service = LifeCollectionService(db)
    creates = await asyncio.gather(
        service.create_collection(user.id, "project", "Гонка"),
        service.create_collection(user.id, "project", "ГОНКА"),
    )
    assert [result.status for result in creates].count("created") == 1
    collection = next(result.collection for result in creates if result.status == "created")
    mutations = await asyncio.gather(
        service.rename(user.id, collection.id, collection.version, "Победитель"),
        service.set_archived(user.id, collection.id, collection.version, archived=True),
    )
    assert sum(result.status in {"renamed", "archived"} for result in mutations) == 1
    assert sum(result.status == "stale" for result in mutations) == 1


async def test_collection_and_item_pagination_are_stable(db):
    user = await owner(db)
    service = LifeCollectionService(db)
    collections = []
    for index in range(8):
        collections.append(
            (await service.create_collection(user.id, "list", f"Список {index}")).collection
        )
    first = await service.list_page(user.id, 0, kind="list")
    second = await service.list_page(user.id, 1, kind="list")
    assert first.total == 8 and first.pages == 2
    assert len(first.records) == 6 and len(second.records) == 2
    target = collections[0]
    await service.create_items(
        user.id,
        700001,
        target.id,
        target.version,
        tuple(f"Пункт {index}" for index in range(8)),
        source="text",
    )
    assert (await service.item_page(user.id, target.id, 0)).total == 8
    assert len((await service.item_page(user.id, target.id, 1)).records) == 2


def test_collection_command_router_is_conservative_and_deterministic():
    router = CollectionCommandRouter()
    assert router.route("Создай проект Наз и Войд").action == "create"
    assert router.route("Сохрани в проект Наз и Войд: исправить главную страницу").action == "add"
    assert router.route("Добавь в покупки чай, сахар и цемент").action == "add"
    assert router.route("Покажи проект Наз и Войд").action == "show"
    assert router.route("Что находится в покупках?").action == "show"
    assert router.route("Перенеси это в Ремонт").action == "move_last"
    assert router.route("Ещё добавь бетономешалку").action == "add_more"
    assert router.route("Запиши идею для творчества: светильник").forced_item_kind == "idea"
    assert router.route("Просто мысль о проекте") is None
    assert normalize_collection_name("  Ёлка—Дом  ") == "елка дом"
