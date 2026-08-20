import asyncio

from sqlalchemy import func, select, update

from future_self.drafts import DraftInboxService
from future_self.models import DraftInboxItem, InboxItem, User
from future_self.schemas import ParsedThought


async def _owner(
    db,
    *,
    telegram_id: int,
    access_tier: str = "admin",
    access_version: int = 1,
) -> User:
    async with db.session() as session:
        owner = User(
            telegram_id=telegram_id,
            timezone="Europe/Moscow",
            access_tier=access_tier,
            access_version=access_version,
            onboarding_completed=True,
        )
        session.add(owner)
        await session.flush()
        owner_id = owner.id
    async with db.sessions() as session:
        return await session.get(User, owner_id)


async def _counts(db) -> tuple[int, int]:
    async with db.sessions() as session:
        drafts = await session.scalar(select(func.count(DraftInboxItem.id)))
        inbox_items = await session.scalar(select(func.count(InboxItem.id)))
    return int(drafts or 0), int(inbox_items or 0)


def _thought(*, next_step: str = "Open the document") -> ParsedThought:
    return ParsedThought(
        kind="task",
        title="Send the agreement",
        description="Send the signed agreement to Marina",
        next_step=next_step,
    )


async def test_suggestion_creation_is_fenced_to_exact_owner_and_chat(db):
    first = await _owner(db, telegram_id=7101, access_version=3)
    second = await _owner(db, telegram_id=7102, access_version=7)
    service = DraftInboxService(db, 60)

    owner_result = await service.create_or_get_for_suggestion(
        user_id=first.id,
        telegram_user_id=first.telegram_id,
        chat_id=8101,
        expected_access_version=3,
        source="nova_companion",
        raw_text="Send the agreement to Marina",
        parsed=_thought(),
    )
    crossed = await service.create_or_get_for_suggestion(
        user_id=first.id,
        telegram_user_id=second.telegram_id,
        chat_id=8101,
        expected_access_version=7,
        source="nova_companion",
        raw_text="Send the agreement to Marina",
        parsed=_thought(),
    )
    other_result = await service.create_or_get_for_suggestion(
        user_id=second.id,
        telegram_user_id=second.telegram_id,
        chat_id=8102,
        expected_access_version=7,
        source="nova_companion",
        raw_text="Send the agreement to Marina",
        parsed=_thought(),
    )

    assert owner_result.status == "created" and owner_result.ok
    assert crossed.status == "access_changed" and not crossed.ok
    assert crossed.draft is None
    assert other_result.status == "created" and other_result.ok
    assert owner_result.draft.id != other_result.draft.id
    assert owner_result.draft.user_id == first.id
    assert other_result.draft.user_id == second.id
    assert await _counts(db) == (2, 0)


async def test_suggestion_creation_rejects_stale_access_generation_after_bounce(db):
    owner = await _owner(db, telegram_id=7201, access_version=4)
    service = DraftInboxService(db, 60)

    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == owner.id).values(access_tier="blocked", access_version=5)
        )
    blocked = await service.create_or_get_for_suggestion(
        user_id=owner.id,
        telegram_user_id=owner.telegram_id,
        chat_id=8201,
        expected_access_version=4,
        source="nova_companion",
        raw_text="Send the agreement to Marina",
        parsed=_thought(),
    )
    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == owner.id).values(access_tier="admin", access_version=6)
        )
    bounced = await service.create_or_get_for_suggestion(
        user_id=owner.id,
        telegram_user_id=owner.telegram_id,
        chat_id=8201,
        expected_access_version=4,
        source="nova_companion",
        raw_text="Send the agreement to Marina",
        parsed=_thought(),
    )

    assert blocked.status == bounced.status == "access_changed"
    assert blocked.draft is bounced.draft is None
    assert await _counts(db) == (0, 0)


async def test_suggestion_creation_reuses_only_exact_active_draft_and_never_saves(db):
    owner = await _owner(db, telegram_id=7301, access_tier="subscriber", access_version=2)
    service = DraftInboxService(db, 60)
    arguments = {
        "user_id": owner.id,
        "telegram_user_id": owner.telegram_id,
        "chat_id": 8301,
        "expected_access_version": 2,
        "source": "nova_companion",
        "raw_text": "Send the agreement to Marina",
    }

    created = await service.create_or_get_for_suggestion(parsed=_thought(), **arguments)
    replay = await service.create_or_get_for_suggestion(
        parsed=ParsedThought(
            kind="task",
            title="send the agreement!",
            description="send the signed agreement to marina",
            next_step="open the document",
        ),
        **arguments,
    )
    distinct = await service.create_or_get_for_suggestion(
        parsed=_thought(next_step="Ask Marina for her email"),
        **arguments,
    )

    assert created.status == "created" and created.created
    assert replay.status == "reused" and replay.ok and not replay.created
    assert replay.draft.id == created.draft.id
    assert distinct.status == "created" and distinct.draft.id != created.draft.id
    assert await _counts(db) == (2, 0)


async def test_concurrent_suggestion_duplicate_is_serialized_by_owner_lock(db):
    owner = await _owner(db, telegram_id=7401, access_version=9)
    service = DraftInboxService(db, 60)
    arguments = {
        "user_id": owner.id,
        "telegram_user_id": owner.telegram_id,
        "chat_id": 8401,
        "expected_access_version": 9,
        "source": "nova_companion",
        "raw_text": "Send the agreement to Marina",
        "parsed": _thought(),
    }

    first, second = await asyncio.gather(
        service.create_or_get_for_suggestion(**arguments),
        service.create_or_get_for_suggestion(**arguments),
    )

    assert {first.status, second.status} == {"created", "reused"}
    assert first.draft.id == second.draft.id
    assert await _counts(db) == (1, 0)


async def test_preview_message_cas_publishes_once_and_preserves_newer_replacement(db):
    owner = await _owner(db, telegram_id=7501, access_version=3)
    service = DraftInboxService(db, 60)
    creation = await service.create_or_get_for_suggestion(
        user_id=owner.id,
        telegram_user_id=owner.telegram_id,
        chat_id=8501,
        expected_access_version=3,
        source="nova_companion",
        raw_text="Send the agreement to Marina",
        parsed=_thought(),
    )
    draft = creation.draft
    assert draft is not None

    winners = await asyncio.gather(
        service.restore_preview_message_if_current(
            draft.id,
            draft.version,
            owner.telegram_id,
            8501,
            expected_message_id=None,
            restored_message_id=101,
        ),
        service.restore_preview_message_if_current(
            draft.id,
            draft.version,
            owner.telegram_id,
            8501,
            expected_message_id=None,
            restored_message_id=102,
        ),
    )
    assert sorted(winners) == [False, True]
    current = await service.get(draft.id)
    assert current is not None
    published = current.preview_message_id
    assert published in {101, 102}

    replacement = 103
    assert await service.restore_preview_message_if_current(
        draft.id,
        draft.version,
        owner.telegram_id,
        8501,
        expected_message_id=published,
        restored_message_id=replacement,
    )
    assert not await service.restore_preview_message_if_current(
        draft.id,
        draft.version,
        owner.telegram_id,
        8501,
        expected_message_id=published,
        restored_message_id=None,
    )
    current = await service.get(draft.id)
    assert current is not None and current.preview_message_id == replacement
    stale_drop = await service.drop_if_preview_message_current(
        draft.id,
        draft.version,
        owner.telegram_id,
        8501,
        expected_message_id=published,
    )
    assert not stale_drop.ok
    current = await service.get(draft.id)
    assert current is not None
    assert current.status == "preview" and current.preview_message_id == replacement


async def test_preview_message_cas_is_exactly_owner_chat_version_and_preview_status(db):
    owner = await _owner(db, telegram_id=7601, access_version=4)
    service = DraftInboxService(db, 60)
    creation = await service.create_or_get_for_suggestion(
        user_id=owner.id,
        telegram_user_id=owner.telegram_id,
        chat_id=8601,
        expected_access_version=4,
        source="nova_companion",
        raw_text="Send the agreement to Marina",
        parsed=_thought(),
    )
    draft = creation.draft
    assert draft is not None
    assert await service.restore_preview_message_if_current(
        draft.id,
        draft.version,
        owner.telegram_id,
        8601,
        expected_message_id=None,
        restored_message_id=201,
    )

    assert not await service.restore_preview_message_if_current(
        draft.id,
        draft.version + 1,
        owner.telegram_id,
        8601,
        expected_message_id=201,
        restored_message_id=202,
    )
    assert not await service.restore_preview_message_if_current(
        draft.id,
        draft.version,
        owner.telegram_id + 1,
        8601,
        expected_message_id=201,
        restored_message_id=202,
    )
    await service.drop(draft.id, draft.version, owner.telegram_id, 8601)
    assert not await service.restore_preview_message_if_current(
        draft.id,
        draft.version,
        owner.telegram_id,
        8601,
        expected_message_id=201,
        restored_message_id=202,
    )
