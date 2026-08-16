from __future__ import annotations

import asyncio
import logging
from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select
from telegram.error import BadRequest

from future_self.access import ADMIN, GUEST, SUBSCRIBER, AccessService
from future_self.conversation import ConversationContextService, ConversationSnapshot
from future_self.models import NovaMemoryChange, NovaMemoryItem, User
from future_self.nova_memory import (
    NovaMemoryMutation,
    NovaMemoryPage,
    NovaMemoryService,
    NovaMemoryStorageError,
)
from future_self.nova_memory_flow import NovaMemoryFlowPhase, NovaMemoryFlowStore
from future_self.nova_memory_handlers import (
    NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    NOVA_MEMORY_ROOT_TEXT,
    NOVA_MEMORY_STALE_ALERT,
    NovaMemoryHandlers,
)
from future_self.repositories import UserRepository


class MemoryMessage:
    _ids = count(90_000)

    def __init__(self, text: str | None = None, *, message_id: int | None = None) -> None:
        self.text = text
        self.message_id = message_id if message_id is not None else next(self._ids)
        self.chat_id = 0
        self.photo = None
        self.document = None
        self.voice = None
        self.audio = None
        self.replies: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted = 0

    async def reply_text(self, text: str, **kwargs: Any) -> MemoryMessage:
        sent = MemoryMessage(text)
        self.replies.append({"text": text, "message": sent, **kwargs})
        return sent

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})

    async def delete(self) -> None:
        self.deleted += 1


class MemoryQuery:
    def __init__(self, data: str, message: MemoryMessage) -> None:
        self.data = data
        self.message = message
        self.answers: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        self.answers.append({"args": args, **kwargs})

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})


class BlockingMemoryQuery(MemoryQuery):
    def __init__(
        self,
        data: str,
        message: MemoryMessage,
        *,
        first_error: BaseException | None = None,
        compensation_error: BaseException | None = None,
    ) -> None:
        super().__init__(data, message)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.first_error = first_error
        self.compensation_error = compensation_error

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        if not self.edits:
            self.edits.append({"text": text, **kwargs})
            self.started.set()
            await self.release.wait()
            if self.first_error is not None:
                raise self.first_error
            return
        self.edits.append({"text": text, **kwargs})
        if self.compensation_error is not None:
            raise self.compensation_error


class FailingMemoryQuery(MemoryQuery):
    def __init__(self, data: str, message: MemoryMessage, error: BaseException) -> None:
        super().__init__(data, message)
        self.error = error

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        del text, kwargs
        raise self.error


class CanonicalPaintBlockingMemoryQuery(BlockingMemoryQuery):
    def __init__(
        self,
        data: str,
        message: MemoryMessage,
        paint_log: list[str],
    ) -> None:
        super().__init__(data, message)
        self.paint_log = paint_log

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        await super().edit_message_text(text, **kwargs)
        self.paint_log.append(text)


class CanonicalPaintMemoryQuery(MemoryQuery):
    def __init__(
        self,
        data: str,
        message: MemoryMessage,
        paint_log: list[str],
        *,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(data, message)
        self.paint_log = paint_log
        self.error = error
        self.started = asyncio.Event()

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        self.started.set()
        if self.error is not None:
            raise self.error
        await super().edit_message_text(text, **kwargs)
        self.paint_log.append(text)


class MemoryBot:
    def __init__(self) -> None:
        self.edits: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edits.append(kwargs)

    async def delete_message(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs)


class BlockingMemoryBot(MemoryBot):
    def __init__(
        self,
        *,
        first_error: BaseException | None = None,
        compensation_error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.first_error = first_error
        self.compensation_error = compensation_error

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.edits.append(kwargs)
        if len(self.edits) == 1:
            self.started.set()
            await self.release.wait()
            if self.first_error is not None:
                raise self.first_error
            return
        if self.compensation_error is not None:
            raise self.compensation_error


class MemoryHarness(NovaMemoryHandlers):
    def __init__(
        self,
        db,
        *,
        enabled: bool = True,
        admin_only: bool = False,
        application_enabled: bool = False,
        application_admin_only: bool = True,
    ) -> None:
        self.db = db
        self.settings = SimpleNamespace(
            enable_nova_memory=enabled,
            nova_memory_admin_only=admin_only,
            enable_nova_memory_application=application_enabled,
            nova_memory_application_admin_only=application_admin_only,
        )
        self.access_service = AccessService(db)
        self.nova_memory_service = NovaMemoryService(db)
        self.nova_memory_sessions = NovaMemoryFlowStore()
        self._nova_memory_launch_lock = asyncio.Lock()
        self._nova_memory_ui_lock = asyncio.Lock()
        self.conversation = SimpleNamespace(
            get=self._conversation_get,
            latest_nova_memory_candidate=ConversationContextService.latest_nova_memory_candidate,
        )
        self.reference_snapshot = ConversationSnapshot()

    async def _conversation_get(self, telegram_user_id: int, chat_id: int) -> Any:
        del telegram_user_id, chat_id
        return self.reference_snapshot

    def _root_keyboard(self, tier: str | None = None) -> Any:
        return SimpleNamespace(tier=tier)


async def memory_user(db, telegram_id: int, *, tier: str = SUBSCRIBER) -> User:
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(telegram_id, "Europe/Moscow")
        user.access_tier = tier
        user.access_version = 4
        return user


def memory_update(
    message: MemoryMessage,
    *,
    telegram_user_id: int,
    chat_id: int,
    query: MemoryQuery | None = None,
) -> SimpleNamespace:
    message.chat_id = chat_id
    return SimpleNamespace(
        effective_message=query.message if query is not None else message,
        message=message,
        effective_user=SimpleNamespace(id=telegram_user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        callback_query=query,
    )


def memory_context() -> SimpleNamespace:
    return SimpleNamespace(bot=MemoryBot(), user_data={})


def callback_for(markup: Any, label: str) -> str:
    for row in markup.inline_keyboard:
        for button in row:
            if button.text == label:
                assert button.callback_data is not None
                return button.callback_data
    raise AssertionError(f"Button not found: {label}")


async def click(
    bot: MemoryHarness,
    user: User,
    chat_id: int,
    canonical: MemoryMessage,
    context: Any,
    markup: Any,
    label: str,
) -> MemoryQuery:
    query = MemoryQuery(callback_for(markup, label), canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=chat_id,
            query=query,
        ),
        context,
    )
    return query


@pytest.mark.asyncio
async def test_command_root_is_exact_access_aware_and_single_canonical(db):
    subscriber = await memory_user(db, 81_001)
    bot = MemoryHarness(db)
    incoming = MemoryMessage("/mynova")
    update = memory_update(incoming, telegram_user_id=subscriber.telegram_id, chat_id=91_001)

    assert await bot.nova_memory_command(update, memory_context()) is True
    assert len(incoming.replies) == 1
    canonical = incoming.replies[0]["message"]
    assert canonical.edits[-1]["text"] == NOVA_MEMORY_ROOT_TEXT.format(
        count=0,
        max_items=100,
        personalization_status="выключена",
    )
    labels = [
        button.text for row in canonical.edits[-1]["reply_markup"].inline_keyboard for button in row
    ]
    assert labels == [
        "➕ Научить Nova",
        "⭐ Важное",
        "📖 Что Nova знает обо мне",
        "⚙️ Как со мной работать",
        "🧭 Мои ориентиры",
        "❓ Как это работает",
        "🏠 Главное меню",
    ]
    callbacks = [
        button.callback_data
        for row in canonical.edits[-1]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert all(value.startswith("nmem:") and len(value.encode()) <= 64 for value in callbacks)


@pytest.mark.parametrize(
    (
        "enabled",
        "admin_only",
        "application_enabled",
        "application_admin_only",
        "tier",
        "expected",
    ),
    [
        (False, False, True, False, ADMIN, False),
        (True, False, False, False, ADMIN, False),
        (True, True, True, False, SUBSCRIBER, False),
        (True, False, True, True, SUBSCRIBER, False),
        (True, True, True, True, ADMIN, True),
        (True, False, True, False, SUBSCRIBER, True),
        (True, False, True, False, GUEST, False),
    ],
)
def test_application_effective_policy_requires_both_gates_and_tier_policies(
    db,
    enabled: bool,
    admin_only: bool,
    application_enabled: bool,
    application_admin_only: bool,
    tier: str,
    expected: bool,
):
    bot = MemoryHarness(
        db,
        enabled=enabled,
        admin_only=admin_only,
        application_enabled=application_enabled,
        application_admin_only=application_admin_only,
    )

    policy = bot.nova_memory_application_policy()

    assert policy.memory_enabled is enabled
    assert policy.memory_admin_only is admin_only
    assert policy.application_enabled is application_enabled
    assert policy.application_admin_only is application_admin_only
    assert bot.nova_memory_application_available_for_tier(tier) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("application_admin_only", "tier", "expected_status"),
    [
        (True, SUBSCRIBER, "выключена"),
        (True, ADMIN, "включена"),
        (False, SUBSCRIBER, "включена"),
    ],
)
async def test_root_and_help_show_effective_application_status_without_new_routes(
    db,
    application_admin_only: bool,
    tier: str,
    expected_status: str,
):
    user = await memory_user(db, next(MemoryMessage._ids), tier=tier)
    bot = MemoryHarness(
        db,
        application_enabled=True,
        application_admin_only=application_admin_only,
    )
    context = memory_context()
    incoming = MemoryMessage("/mynova")
    chat_id = next(MemoryMessage._ids)

    assert await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=chat_id),
        context,
    )
    canonical = incoming.replies[0]["message"]
    root = canonical.edits[-1]
    assert f"Персонализация AI-ответов: {expected_status}" in root["text"]
    root_labels = [button.text for row in root["reply_markup"].inline_keyboard for button in row]

    help_query = await click(
        bot,
        user,
        chat_id,
        canonical,
        context,
        root["reply_markup"],
        "❓ Как это работает",
    )

    assert f"Персонализация AI-ответов: {expected_status}" in help_query.edits[-1]["text"]
    assert "ограниченная выборка подтверждённых записей" in help_query.edits[-1]["text"]
    assert "не означает\nобязательное упоминание" in help_query.edits[-1]["text"]
    assert root_labels == [
        "➕ Научить Nova",
        "⭐ Важное",
        "📖 Что Nova знает обо мне",
        "⚙️ Как со мной работать",
        "🧭 Мои ориентиры",
        "❓ Как это работает",
        "🏠 Главное меню",
    ]


@pytest.mark.asyncio
async def test_feature_and_admin_only_fail_closed_without_session(db):
    subscriber = await memory_user(db, 81_002)
    bot = MemoryHarness(db, admin_only=True)
    incoming = MemoryMessage("/mynova")
    update = memory_update(incoming, telegram_user_id=subscriber.telegram_id, chat_id=91_002)

    assert await bot.nova_memory_command(update, memory_context())
    assert incoming.replies[-1]["text"] == "Функция «Моя Nova» сейчас недоступна."
    assert await bot.nova_memory_sessions.count() == 0

    admin = await memory_user(db, 81_003, tier=ADMIN)
    admin_update = memory_update(
        MemoryMessage("/mynova"), telegram_user_id=admin.telegram_id, chat_id=91_003
    )
    assert await bot.nova_memory_command(admin_update, memory_context())
    assert await bot.nova_memory_sessions.count() == 1


@pytest.mark.asyncio
async def test_explicit_text_preview_has_zero_dml_then_confirm_is_single_use(db, monkeypatch):
    user = await memory_user(db, 81_010)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("Nova, запомни обо мне: Я люблю тихие прогулки")
    update = memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_010)

    assert await bot.nova_memory_text_gate(update, context)
    assert len(incoming.replies) == 1
    canonical = incoming.replies[0]["message"]
    preview = context.bot.edits[-1]
    assert "Nova запомнит" in preview["text"]
    assert "Я люблю тихие прогулки" in preview["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 0

    original_create = bot.nova_memory_service.create
    domain_calls = 0

    async def counted_create(**kwargs: Any):
        nonlocal domain_calls
        domain_calls += 1
        return await original_create(**kwargs)

    monkeypatch.setattr(bot.nova_memory_service, "create", counted_create)
    data = callback_for(preview["reply_markup"], "✅ Запомнить")
    first = MemoryQuery(data, canonical)
    replay = MemoryQuery(data, canonical)
    first_update = memory_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=91_010,
        query=first,
    )
    replay_update = memory_update(
        canonical,
        telegram_user_id=user.telegram_id,
        chat_id=91_010,
        query=replay,
    )
    await asyncio.gather(
        bot.nova_memory_callback(first_update, context),
        bot.nova_memory_callback(replay_update, context),
    )

    winner, loser = (first, replay) if first.edits else (replay, first)
    assert len(winner.answers) == len(loser.answers) == 1
    assert "Nova запомнила" in winner.edits[-1]["text"]
    assert loser.answers == [{"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}]
    assert loser.edits == []
    assert domain_calls == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1


@pytest.mark.asyncio
async def test_remember_this_previews_only_latest_suitable_conversation_candidate(db):
    user = await memory_user(db, 81_012)
    bot = MemoryHarness(db)
    context = memory_context()
    older = "Старое подходящее сообщение не должно попасть в preview"
    latest = "Я предпочитаю короткие и спокойные ответы"
    bot.reference_snapshot = ConversationSnapshot(
        messages=[
            {
                "role": "user",
                "content": older,
                "source": "text",
                "intent": "conversation",
            },
            {
                "role": "assistant",
                "content": "Поняла.",
                "source": "text",
                "intent": "conversation",
            },
            {
                "role": "user",
                "content": f"  {latest}  ",
                "source": "voice",
                "intent": "conversation",
            },
            {
                "role": "assistant",
                "content": "Хорошо.",
                "source": "text",
                "intent": "conversation",
            },
        ]
    )

    class ProviderTripwire:
        calls = 0

        def __getattr__(self, name: str) -> Any:
            del name
            self.calls += 1
            raise AssertionError("Nova memory reference lookup must not use a provider")

    provider = ProviderTripwire()
    bot.ai = provider
    incoming = MemoryMessage("Nova, запомни это")

    assert await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_012),
        context,
    )
    preview = context.bot.edits[-1]
    assert latest in preview["text"]
    assert older not in preview["text"]
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_012,
    )
    assert current is not None
    assert current.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    assert current.candidate_content == latest
    assert provider.calls == 0
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 0


@pytest.mark.asyncio
async def test_remember_this_fails_closed_when_newest_user_candidate_is_unsuitable(db):
    user = await memory_user(db, 81_013)
    bot = MemoryHarness(db)
    context = memory_context()
    older = "Старое подходящее сообщение нельзя использовать как fallback"
    newest = "Последняя реплика относится к навигации"
    bot.reference_snapshot = ConversationSnapshot(
        messages=[
            {
                "role": "user",
                "content": older,
                "source": "text",
                "intent": "conversation",
            },
            {
                "role": "user",
                "content": newest,
                "source": "text",
                "intent": "navigation",
            },
        ]
    )
    incoming = MemoryMessage("Nova, запомни это")

    assert await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_013),
        context,
    )
    screen = context.bot.edits[-1]
    assert older not in screen["text"]
    assert newest not in screen["text"]
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_013,
    )
    assert current is not None
    assert current.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
    assert current.candidate_content is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 0


@pytest.mark.asyncio
async def test_invalid_explicit_content_is_consumed_without_dml(db):
    user = await memory_user(db, 81_011)
    bot = MemoryHarness(db)
    context = memory_context()
    text = "Nova, запомни: " + "x" * 501
    incoming = MemoryMessage(text)
    update = memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_011)

    assert await bot.nova_memory_owns_text(update, text)
    assert await bot.nova_memory_text_gate(update, context)
    assert "1 до 500" in context.bot.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0


@pytest.mark.asyncio
async def test_voice_uses_progress_as_first_canonical_and_media_is_contained(db):
    user = await memory_user(db, 81_020)
    bot = MemoryHarness(db)
    context = memory_context()
    voice = MemoryMessage()
    update = memory_update(voice, telegram_user_id=user.telegram_id, chat_id=91_020)
    fence = await bot.nova_memory_voice_fence(update, user=user)
    progress = MemoryMessage("Расшифровываю…")
    progress.chat_id = 91_020

    assert await bot.nova_memory_voice_gate(
        update,
        context,
        "Nova, запомни: Голосовой кандидат",
        progress,
        user=user,
        fence=fence,
    )
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_020,
    )
    assert current is not None
    assert current.canonical_message_id == progress.message_id
    assert context.bot.edits[-1]["message_id"] == progress.message_id
    assert "Голосовой кандидат" in context.bot.edits[-1]["text"]

    await bot.nova_memory_sessions.update(
        current,
        phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
        candidate_content=None,
        candidate_category="about_me",
    )
    live = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_020,
    )
    assert live is not None
    photo = MemoryMessage()
    photo.photo = [object()]
    media_update = memory_update(photo, telegram_user_id=user.telegram_id, chat_id=91_020)
    assert await bot.nova_memory_media_gate(media_update, context)
    assert context.bot.edits[-1]["text"].endswith("Пришли текст или голосовое сообщение.")


@pytest.mark.asyncio
async def test_access_bounce_neutralizes_exact_old_canonical(db):
    user = await memory_user(db, 81_030)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("Nova, запомни: PRIVATE_ACCESS_SENTINEL")
    update = memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_030)
    assert await bot.nova_memory_text_gate(update, context)
    canonical = incoming.replies[0]["message"]
    preview = context.bot.edits[-1]
    data = callback_for(preview["reply_markup"], "✅ Запомнить")

    async with db.session() as session:
        stored = await session.scalar(select(User).where(User.id == user.id))
        assert stored is not None
        stored.access_version += 1
    query = MemoryQuery(data, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_030,
            query=query,
        ),
        context,
    )

    assert query.answers == [{"args": ()}]
    assert query.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert "PRIVATE_ACCESS_SENTINEL" not in query.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0


@pytest.mark.asyncio
async def test_list_post_read_downgrade_discards_private_page_and_neutralizes_canonical(
    db,
    monkeypatch,
):
    user = await memory_user(db, 81_031)
    bot = MemoryHarness(db)
    context = memory_context()
    private = "PRIVATE_LIST_POST_READ_SENTINEL"
    await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="about_me",
        content=private,
    )
    incoming = MemoryMessage("/mynova")
    update = memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_031)
    await bot.nova_memory_command(update, context)
    canonical = incoming.replies[0]["message"]
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_031,
    )
    assert current is not None
    token = await bot.nova_memory_sessions.issue(current, action="list_about_me")
    assert token is not None
    original_list = bot.nova_memory_service.list

    async def list_then_downgrade(**kwargs: Any) -> NovaMemoryPage:
        result = await original_list(**kwargs)
        async with db.session() as session:
            stored = await session.scalar(select(User).where(User.id == user.id))
            assert stored is not None
            stored.access_tier = GUEST
            stored.access_version += 1
        return result

    monkeypatch.setattr(bot.nova_memory_service, "list", list_then_downgrade)
    query = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_031,
            query=query,
        ),
        context,
    )

    assert query.answers == [{"args": ()}]
    assert query.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert private not in query.edits[-1]["text"]
    assert private not in str(context.bot.edits)
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=91_031,
        )
        is None
    )


@pytest.mark.asyncio
async def test_detail_post_read_version_bounce_discards_private_item_and_neutralizes_canonical(
    db,
    monkeypatch,
):
    user = await memory_user(db, 81_032)
    bot = MemoryHarness(db)
    context = memory_context()
    private = "PRIVATE_DETAIL_POST_READ_SENTINEL"
    created = await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="about_me",
        content=private,
    )
    assert created.item is not None
    incoming = MemoryMessage("/mynova")
    update = memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_032)
    await bot.nova_memory_command(update, context)
    canonical = incoming.replies[0]["message"]
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_032,
    )
    assert current is not None
    token = await bot.nova_memory_sessions.issue(
        current,
        action="detail",
        public_id=created.item.public_id,
        expected_item_version=created.item.version,
        list_filter="about_me",
        page=0,
    )
    assert token is not None
    original_get = bot.nova_memory_service.get

    async def get_then_bounce(**kwargs: Any):
        result = await original_get(**kwargs)
        async with db.session() as session:
            stored = await session.scalar(select(User).where(User.id == user.id))
            assert stored is not None
            stored.access_version += 1
        return result

    monkeypatch.setattr(bot.nova_memory_service, "get", get_then_bounce)
    query = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_032,
            query=query,
        ),
        context,
    )

    assert query.answers == [{"args": ()}]
    assert query.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert private not in query.edits[-1]["text"]
    assert private not in str(context.bot.edits)
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=91_032,
        )
        is None
    )


@pytest.mark.asyncio
async def test_downgrade_immediately_before_domain_mutation_has_zero_memory_dml(
    db,
    monkeypatch,
):
    user = await memory_user(db, 81_033)
    bot = MemoryHarness(db)
    context = memory_context()
    private = "PRIVATE_PRE_DOMAIN_DOWNGRADE_SENTINEL"
    incoming = MemoryMessage(f"Nova, запомни: {private}")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_033),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(context.bot.edits[-1]["reply_markup"], "✅ Запомнить")
    original_before = bot._nova_memory_before_mutation
    domain_calls = 0

    async def downgrade_at_before_mutation(update: Any, callback_context: Any, session: Any):
        async with db.session() as db_session:
            stored = await db_session.scalar(select(User).where(User.id == user.id))
            assert stored is not None
            stored.access_tier = GUEST
            stored.access_version += 1
        return await original_before(update, callback_context, session)

    async def forbidden_create(**kwargs: Any):
        nonlocal domain_calls
        del kwargs
        domain_calls += 1
        raise AssertionError("domain create must stay behind the final access fence")

    monkeypatch.setattr(bot, "_nova_memory_before_mutation", downgrade_at_before_mutation)
    monkeypatch.setattr(bot.nova_memory_service, "create", forbidden_create)
    query = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_033,
            query=query,
        ),
        context,
    )

    assert domain_calls == 0
    assert query.answers == [{"args": ()}]
    assert query.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert private not in query.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 0


@pytest.mark.asyncio
async def test_post_mutation_version_bounce_keeps_commit_but_neutralizes_private_canonical(
    db,
    monkeypatch,
):
    user = await memory_user(db, 81_034)
    bot = MemoryHarness(db)
    context = memory_context()
    private = "PRIVATE_POST_MUTATION_BOUNCE_SENTINEL"
    incoming = MemoryMessage(f"Nova, запомни: {private}")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_034),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(context.bot.edits[-1]["reply_markup"], "✅ Запомнить")
    original_create = bot.nova_memory_service.create
    domain_calls = 0

    async def create_then_bounce(**kwargs: Any):
        nonlocal domain_calls
        domain_calls += 1
        result = await original_create(**kwargs)
        async with db.session() as session:
            stored = await session.scalar(select(User).where(User.id == user.id))
            assert stored is not None
            stored.access_version += 1
        return result

    monkeypatch.setattr(bot.nova_memory_service, "create", create_then_bounce)
    query = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_034,
            query=query,
        ),
        context,
    )

    assert domain_calls == 1
    assert query.answers == [{"args": ()}]
    assert query.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert private not in query.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1
        committed = await session.scalar(select(NovaMemoryItem))
        assert committed is not None and committed.content == private


@pytest.mark.asyncio
async def test_list_detail_and_replay_safe_importance_toggle(db):
    user = await memory_user(db, 81_040)
    bot = MemoryHarness(db)
    context = memory_context()
    created = await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="about_me",
        content="Особенно важная запись",
        important=True,
    )
    assert created.item is not None
    await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="about_me",
        content="Обычная запись",
    )
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_040),
        context,
    )
    canonical = incoming.replies[0]["message"]
    root_markup = canonical.edits[-1]["reply_markup"]

    listed = await click(bot, user, 91_040, canonical, context, root_markup, "⭐ Важное")
    assert "Записей: 1 · Страница 1/1" in listed.edits[-1]["text"]
    item_button = listed.edits[-1]["reply_markup"].inline_keyboard[0][0]
    detail_query = MemoryQuery(item_button.callback_data, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_040,
            query=detail_query,
        ),
        context,
    )
    assert "🧬 Запись Nova" in detail_query.edits[-1]["text"]
    toggle_data = callback_for(
        detail_query.edits[-1]["reply_markup"],
        "☆ Убрать из важного",
    )
    first = MemoryQuery(toggle_data, canonical)
    replay = MemoryQuery(toggle_data, canonical)
    await asyncio.gather(
        bot.nova_memory_callback(
            memory_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=91_040,
                query=first,
            ),
            context,
        ),
        bot.nova_memory_callback(
            memory_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=91_040,
                query=replay,
            ),
            context,
        ),
    )
    winner, loser = (first, replay) if first.edits else (replay, first)
    assert "⭐ Важное" not in winner.edits[-1]["text"]
    assert loser.answers == [{"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}]
    lookup = await bot.nova_memory_service.get(
        telegram_actor_id=user.telegram_id,
        public_id=created.item.public_id,
    )
    assert lookup.item is not None and lookup.item.important is False


@pytest.mark.asyncio
async def test_edit_text_and_category_require_preview_before_domain_update(db):
    user = await memory_user(db, 81_050)
    bot = MemoryHarness(db)
    context = memory_context()
    created = await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="about_me",
        content="Старый текст",
    )
    assert created.item is not None
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_050),
        context,
    )
    canonical = incoming.replies[0]["message"]
    listed = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        canonical.edits[-1]["reply_markup"],
        "📖 Что Nova знает обо мне",
    )
    item_data = listed.edits[-1]["reply_markup"].inline_keyboard[0][0].callback_data
    detail = MemoryQuery(item_data, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_050,
            query=detail,
        ),
        context,
    )
    editing = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        detail.edits[-1]["reply_markup"],
        "✏️ Изменить",
    )
    assert "Текущий текст:\n«Старый текст»" in editing.edits[-1]["text"]

    category_only = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        editing.edits[-1]["reply_markup"],
        "🗂 Изменить только раздел",
    )
    category_only_preview = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        category_only.edits[-1]["reply_markup"],
        "⚙️ Как со мной работать",
    )
    assert "Станет:\n«Старый текст»" in category_only_preview.edits[-1]["text"]
    detail_again = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        category_only_preview.edits[-1]["reply_markup"],
        "✖️ Отмена",
    )
    editing = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        detail_again.edits[-1]["reply_markup"],
        "✏️ Изменить",
    )

    replacement = MemoryMessage("Новый точный текст")
    assert await bot.nova_memory_text_gate(
        memory_update(replacement, telegram_user_id=user.telegram_id, chat_id=91_050),
        context,
    )
    preview = context.bot.edits[-1]
    assert "Было:\n«Старый текст»" in preview["text"]
    assert "Станет:\n«Новый точный текст»" in preview["text"]
    before = await bot.nova_memory_service.get(
        telegram_actor_id=user.telegram_id,
        public_id=created.item.public_id,
    )
    assert before.item is not None and before.item.content == "Старый текст"

    categories = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        preview["reply_markup"],
        "🗂 Изменить раздел",
    )
    oriented = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        categories.edits[-1]["reply_markup"],
        "🧭 Мой ориентир",
    )
    assert "Раздел: 🧭 Мои ориентиры" in oriented.edits[-1]["text"]
    assert all(
        button.text not in {"⭐ В важное", "☆ Убрать из важного"}
        for row in oriented.edits[-1]["reply_markup"].inline_keyboard
        for button in row
    )
    saved = await click(
        bot,
        user,
        91_050,
        canonical,
        context,
        oriented.edits[-1]["reply_markup"],
        "✅ Сохранить изменения",
    )
    assert "Изменения сохранены" in saved.edits[-1]["text"]
    after = await bot.nova_memory_service.get(
        telegram_actor_id=user.telegram_id,
        public_id=created.item.public_id,
    )
    assert after.item is not None
    assert (after.item.content, after.item.category, after.item.version) == (
        "Новый точный текст",
        "orientation",
        2,
    )


@pytest.mark.asyncio
async def test_delete_all_uses_exact_collection_revision_and_replay_is_stale(db):
    user = await memory_user(db, 81_060)
    bot = MemoryHarness(db)
    context = memory_context()
    for value in ("Первая запись", "Вторая запись"):
        await bot.nova_memory_service.create(
            telegram_actor_id=user.telegram_id,
            expected_access_version=user.access_version,
            category="about_me",
            content=value,
        )
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_060),
        context,
    )
    canonical = incoming.replies[0]["message"]
    preview = await click(
        bot,
        user,
        91_060,
        canonical,
        context,
        canonical.edits[-1]["reply_markup"],
        "🗑 Забыть всё",
    )
    assert "все 2 записей" in preview.edits[-1]["text"]
    stale_data = callback_for(preview.edits[-1]["reply_markup"], "🗑 Да, забыть всё")
    await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="orientation",
        content="Добавлено после preview",
    )
    stale = MemoryQuery(stale_data, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_060,
            query=stale,
        ),
        context,
    )
    assert "Ничего не удалено" in stale.edits[-1]["text"]
    assert (await bot.nova_memory_service.status(telegram_actor_id=user.telegram_id)).count == 3

    current_root = stale.edits[-1]["reply_markup"]
    fresh_preview = await click(
        bot,
        user,
        91_060,
        canonical,
        context,
        current_root,
        "🗑 Забыть всё",
    )
    confirm_data = callback_for(
        fresh_preview.edits[-1]["reply_markup"],
        "🗑 Да, забыть всё",
    )
    confirmed = MemoryQuery(confirm_data, canonical)
    replay = MemoryQuery(confirm_data, canonical)
    await asyncio.gather(
        bot.nova_memory_callback(
            memory_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=91_060,
                query=confirmed,
            ),
            context,
        ),
        bot.nova_memory_callback(
            memory_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=91_060,
                query=replay,
            ),
            context,
        ),
    )
    winner, loser = (confirmed, replay) if confirmed.edits else (replay, confirmed)
    assert "Nova забыла 3 записей" in winner.edits[-1]["text"]
    assert loser.answers == [{"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}]
    assert (await bot.nova_memory_service.status(telegram_actor_id=user.telegram_id)).count == 0


@pytest.mark.asyncio
async def test_wrong_actor_chat_and_canonical_do_not_consume_mutation_capability(db):
    user = await memory_user(db, 81_070)
    intruder = await memory_user(db, 81_071)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("Nova, запомни: Capability остаётся одноразовой")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_070),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(context.bot.edits[-1]["reply_markup"], "✅ Запомнить")

    attempts = (
        memory_update(
            canonical,
            telegram_user_id=intruder.telegram_id,
            chat_id=91_070,
            query=MemoryQuery(token, canonical),
        ),
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_071,
            query=MemoryQuery(token, canonical),
        ),
        memory_update(
            MemoryMessage(message_id=canonical.message_id + 1),
            telegram_user_id=user.telegram_id,
            chat_id=91_070,
            query=MemoryQuery(token, MemoryMessage(message_id=canonical.message_id + 1)),
        ),
    )
    for wrong_update in attempts:
        await bot.nova_memory_callback(wrong_update, context)
        assert wrong_update.callback_query.answers == [
            {"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}
        ]

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0

    rightful = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_070,
            query=rightful,
        ),
        context,
    )
    assert "Nova запомнила" in rightful.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1


@pytest.mark.asyncio
async def test_restart_makes_old_capability_stale_without_domain_write(db):
    user = await memory_user(db, 81_072)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("Nova, запомни: Не переживает рестарт")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_072),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(context.bot.edits[-1]["reply_markup"], "✅ Запомнить")
    bot.nova_memory_sessions = NovaMemoryFlowStore()

    query = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_072,
            query=query,
        ),
        context,
    )
    assert query.answers == [{"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}]
    assert query.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0


@pytest.mark.asyncio
async def test_replacement_during_final_mutation_access_check_prevents_dml(db, monkeypatch):
    user = await memory_user(db, 81_073)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("Nova, запомни: Не уйдёт в заменённую сессию")
    update = memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_073)
    await bot.nova_memory_text_gate(update, context)
    canonical = incoming.replies[0]["message"]
    token = callback_for(context.bot.edits[-1]["reply_markup"], "✅ Запомнить")
    original_access = bot._nova_memory_access
    access_calls = 0

    async def replace_on_before_mutation(callback_update: Any) -> User | None:
        nonlocal access_calls
        access_calls += 1
        actor = await original_access(callback_update)
        if access_calls == 3:
            await bot.nova_memory_sessions.create(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=91_073,
                tier=user.access_tier,
                access_version=user.access_version,
                canonical_message_id=canonical.message_id,
                phase=NovaMemoryFlowPhase.ROOT,
            )
        return actor

    monkeypatch.setattr(bot, "_nova_memory_access", replace_on_before_mutation)
    query = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_073,
            query=query,
        ),
        context,
    )
    assert query.answers == [{"args": ()}]
    assert query.edits == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0


@pytest.mark.asyncio
async def test_callback_edit_errors_have_no_fallback_and_cancelled_error_propagates(db):
    user = await memory_user(db, 81_074)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_074),
        context,
    )
    canonical = incoming.replies[0]["message"]
    help_token = callback_for(canonical.edits[-1]["reply_markup"], "❓ Как это работает")
    not_modified = FailingMemoryQuery(
        help_token,
        canonical,
        BadRequest("Message is not modified"),
    )
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_074,
            query=not_modified,
        ),
        context,
    )
    assert not_modified.answers == [{"args": ()}]
    assert len(incoming.replies) == 1

    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_074,
    )
    assert current is not None
    root_token = await bot.nova_memory_sessions.issue(current, action="root")
    assert root_token is not None
    cancelled = FailingMemoryQuery(root_token, canonical, asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await bot.nova_memory_callback(
            memory_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=91_074,
                query=cancelled,
            ),
            context,
        )


@pytest.mark.asyncio
async def test_mutation_edit_bad_request_commits_once_without_send_fallback(db):
    user = await memory_user(db, 81_075)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("Nova, запомни: Telegram edit может не удаться")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_075),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(context.bot.edits[-1]["reply_markup"], "✅ Запомнить")
    bot_edits_before = len(context.bot.edits)
    query = FailingMemoryQuery(token, canonical, BadRequest("edit failed"))
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_075,
            query=query,
        ),
        context,
    )
    assert query.answers == [{"args": ()}]
    assert len(context.bot.edits) == bot_edits_before
    assert len(incoming.replies) == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1

    replay = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_075,
            query=replay,
        ),
        context,
    )
    assert replay.answers == [{"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}]


@pytest.mark.asyncio
async def test_final_access_bounce_neutralizes_help_before_private_edit(db, monkeypatch):
    user = await memory_user(db, 81_076)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_076),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(canonical.edits[-1]["reply_markup"], "❓ Как это работает")
    calls = 0

    async def access_bounce(telegram_user_id: int, chat_id: int) -> User | None:
        nonlocal calls
        del telegram_user_id, chat_id
        calls += 1
        return user if calls == 1 else None

    monkeypatch.setattr(bot, "_nova_memory_access_values", access_bounce)
    query = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_076,
            query=query,
        ),
        context,
    )
    assert query.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert "Обычная переписка" not in query.edits[-1]["text"]
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=91_076,
        )
        is None
    )


@pytest.mark.asyncio
async def test_important_page_rejects_mixed_collection_revisions(db):
    del db
    bot = SimpleNamespace()
    bot.nova_memory_service = SimpleNamespace()
    revisions = iter(("a" * 64, "b" * 64))

    async def mixed_list(**kwargs: Any) -> NovaMemoryPage:
        offset = kwargs["offset"]
        return NovaMemoryPage(
            status="ok",
            offset=offset,
            limit=50,
            total=51,
            next_offset=50 if offset == 0 else None,
            collection_revision=next(revisions),
        )

    bot.nova_memory_service.list = mixed_list
    with pytest.raises(NovaMemoryStorageError):
        await NovaMemoryHandlers._nova_memory_page(
            bot,
            81_077,
            list_filter="important",
            page=0,
        )


@pytest.mark.asyncio
async def test_duplicate_and_limit_outcomes_are_explicit_and_zero_extra_dml(db, monkeypatch):
    user = await memory_user(db, 81_080)
    bot = MemoryHarness(db)
    context = memory_context()
    original = await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="about_me",
        content="Точное совпадение",
    )
    assert original.item is not None
    incoming = MemoryMessage("Nova, запомни: Точное совпадение")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_080),
        context,
    )
    canonical = incoming.replies[0]["message"]
    duplicate = await click(
        bot,
        user,
        91_080,
        canonical,
        context,
        context.bot.edits[-1]["reply_markup"],
        "✅ Запомнить",
    )
    assert "Эта запись уже сохранена" in duplicate.edits[-1]["text"]
    labels = [
        button.text for row in duplicate.edits[-1]["reply_markup"].inline_keyboard for button in row
    ]
    assert "← К списку" in labels
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1

    second = MemoryMessage("Nova, запомни: Лимит-кандидат")
    await bot.nova_memory_text_gate(
        memory_update(second, telegram_user_id=user.telegram_id, chat_id=91_080),
        context,
    )
    limit_preview = context.bot.edits[-1]

    async def limit_reached(**kwargs: Any) -> NovaMemoryMutation:
        del kwargs
        return NovaMemoryMutation(status="limit_reached")

    monkeypatch.setattr(bot.nova_memory_service, "create", limit_reached)
    limited = await click(
        bot,
        user,
        91_080,
        canonical,
        context,
        limit_preview["reply_markup"],
        "✅ Запомнить",
    )
    assert "Достигнут лимит памяти" in limited.edits[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1


@pytest.mark.asyncio
async def test_category_and_important_lists_page_beyond_five(db):
    user = await memory_user(db, 81_081)
    bot = MemoryHarness(db)
    context = memory_context()
    for index in range(7):
        await bot.nova_memory_service.create(
            telegram_actor_id=user.telegram_id,
            expected_access_version=user.access_version,
            category="about_me",
            content=f"Обо мне номер {index}",
            important=index % 2 == 0,
        )
    await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="interaction",
        content="Работать короткими итерациями",
    )
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_081),
        context,
    )
    canonical = incoming.replies[0]["message"]
    about = await click(
        bot,
        user,
        91_081,
        canonical,
        context,
        canonical.edits[-1]["reply_markup"],
        "📖 Что Nova знает обо мне",
    )
    assert "Записей: 7 · Страница 1/2" in about.edits[-1]["text"]
    assert len(about.edits[-1]["reply_markup"].inline_keyboard) == 7
    next_page = await click(
        bot,
        user,
        91_081,
        canonical,
        context,
        about.edits[-1]["reply_markup"],
        "Далее →",
    )
    assert "Записей: 7 · Страница 2/2" in next_page.edits[-1]["text"]
    root = await click(
        bot,
        user,
        91_081,
        canonical,
        context,
        next_page.edits[-1]["reply_markup"],
        "← К моей Nova",
    )
    important = await click(
        bot,
        user,
        91_081,
        canonical,
        context,
        root.edits[-1]["reply_markup"],
        "⭐ Важное",
    )
    assert "Записей: 4 · Страница 1/1" in important.edits[-1]["text"]
    item_labels = [row[0].text for row in important.edits[-1]["reply_markup"].inline_keyboard[:-1]]
    assert len(item_labels) == 4 and all("⭐" in label for label in item_labels)
    root = await click(
        bot,
        user,
        91_081,
        canonical,
        context,
        important.edits[-1]["reply_markup"],
        "← К моей Nova",
    )
    interaction = await click(
        bot,
        user,
        91_081,
        canonical,
        context,
        root.edits[-1]["reply_markup"],
        "⚙️ Как со мной работать",
    )
    assert "Записей: 1 · Страница 1/1" in interaction.edits[-1]["text"]


@pytest.mark.asyncio
async def test_delete_item_stale_then_success_uses_exact_version(db):
    user = await memory_user(db, 81_082)
    bot = MemoryHarness(db)
    context = memory_context()
    created = await bot.nova_memory_service.create(
        telegram_actor_id=user.telegram_id,
        expected_access_version=user.access_version,
        category="about_me",
        content="Удаляемая запись",
    )
    assert created.item is not None
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_082),
        context,
    )
    canonical = incoming.replies[0]["message"]
    listed = await click(
        bot,
        user,
        91_082,
        canonical,
        context,
        canonical.edits[-1]["reply_markup"],
        "📖 Что Nova знает обо мне",
    )
    detail = MemoryQuery(
        listed.edits[-1]["reply_markup"].inline_keyboard[0][0].callback_data,
        canonical,
    )
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_082,
            query=detail,
        ),
        context,
    )
    preview = await click(
        bot,
        user,
        91_082,
        canonical,
        context,
        detail.edits[-1]["reply_markup"],
        "🗑 Забыть",
    )
    delete_generation = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_082,
    )
    assert delete_generation is not None
    stray_text = MemoryMessage("обычный текст во время подтверждения")
    assert await bot.nova_memory_text_gate(
        memory_update(stray_text, telegram_user_id=user.telegram_id, chat_id=91_082),
        context,
    )
    photo = MemoryMessage()
    photo.photo = [object()]
    assert await bot.nova_memory_media_gate(
        memory_update(photo, telegram_user_id=user.telegram_id, chat_id=91_082),
        context,
    )
    voice_message = MemoryMessage()
    voice_update = memory_update(
        voice_message,
        telegram_user_id=user.telegram_id,
        chat_id=91_082,
    )
    voice_fence = await bot.nova_memory_voice_fence(voice_update, user=user)
    voice_progress = MemoryMessage("Расшифровываю…")
    voice_progress.chat_id = 91_082
    assert await bot.nova_memory_voice_gate(
        voice_update,
        context,
        "обычный голосовой текст",
        voice_progress,
        user=user,
        fence=voice_fence,
    )
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=91_082,
        )
        == delete_generation
    )
    changed = await bot.nova_memory_service.update(
        telegram_actor_id=user.telegram_id,
        public_id=created.item.public_id,
        expected_version=created.item.version,
        expected_access_version=user.access_version,
        content="Запись успела измениться",
        category="about_me",
    )
    assert changed.status == "updated"
    stale = await click(
        bot,
        user,
        91_082,
        canonical,
        context,
        preview.edits[-1]["reply_markup"],
        "🗑 Да, забыть",
    )
    assert "Ничего не перезаписано" in stale.edits[-1]["text"]
    assert (
        await bot.nova_memory_service.get(
            telegram_actor_id=user.telegram_id,
            public_id=created.item.public_id,
        )
    ).status == "found"

    listed = await click(
        bot,
        user,
        91_082,
        canonical,
        context,
        stale.edits[-1]["reply_markup"],
        "📖 Что Nova знает обо мне",
    )
    detail = MemoryQuery(
        listed.edits[-1]["reply_markup"].inline_keyboard[0][0].callback_data,
        canonical,
    )
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_082,
            query=detail,
        ),
        context,
    )
    preview = await click(
        bot,
        user,
        91_082,
        canonical,
        context,
        detail.edits[-1]["reply_markup"],
        "🗑 Забыть",
    )
    deleted = await click(
        bot,
        user,
        91_082,
        canonical,
        context,
        preview.edits[-1]["reply_markup"],
        "🗑 Да, забыть",
    )
    assert "Запись удалена" in deleted.edits[-1]["text"]
    assert (
        await bot.nova_memory_service.get(
            telegram_actor_id=user.telegram_id,
            public_id=created.item.public_id,
        )
    ).status == "not_found"


@pytest.mark.asyncio
async def test_cancel_retires_preview_without_domain_write(db):
    user = await memory_user(db, 81_083)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("Nova, запомни: Отменяемый кандидат")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_083),
        context,
    )
    canonical = incoming.replies[0]["message"]
    cancel = MemoryMessage("/cancel")
    assert await bot.nova_memory_cancel_gate(
        memory_update(cancel, telegram_user_id=user.telegram_id, chat_id=91_083),
        context,
    )
    assert context.bot.edits[-1]["message_id"] == canonical.message_id
    assert "Ничего не сохранено" in context.bot.edits[-1]["text"]
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=91_083,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0


@pytest.mark.asyncio
async def test_voice_preview_confirms_and_nonawaiting_voice_does_not_corrupt_phase(db):
    user = await memory_user(db, 81_084)
    bot = MemoryHarness(db)
    context = memory_context()
    voice = MemoryMessage()
    update = memory_update(voice, telegram_user_id=user.telegram_id, chat_id=91_084)
    fence = await bot.nova_memory_voice_fence(update, user=user)
    progress = MemoryMessage("Расшифровываю…")
    progress.chat_id = 91_084
    assert await bot.nova_memory_voice_gate(
        update,
        context,
        "Nova, запомни: Голос до подтверждения",
        progress,
        user=user,
        fence=fence,
    )
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_084,
    )
    assert current is not None and current.phase is NovaMemoryFlowPhase.CREATE_PREVIEW

    second_progress = MemoryMessage("Расшифровываю…")
    second_progress.chat_id = 91_084
    second_fence = await bot.nova_memory_voice_fence(update, user=user)
    assert await bot.nova_memory_voice_gate(
        update,
        context,
        "обычный голосовой текст",
        second_progress,
        user=user,
        fence=second_fence,
    )
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_084,
    )
    assert current is not None and current.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    refreshed = context.bot.edits[-1]
    confirmed = await click(
        bot,
        user,
        91_084,
        progress,
        context,
        refreshed["reply_markup"],
        "✅ Запомнить",
    )
    assert "Nova запомнила" in confirmed.edits[-1]["text"]
    async with db.sessions() as session:
        item = await session.scalar(select(NovaMemoryItem))
        assert item is not None and item.content == "Голос до подтверждения"


@pytest.mark.asyncio
async def test_candidate_is_absent_from_repr_callbacks_and_storage_error_logs(
    db,
    monkeypatch,
    caplog,
):
    user = await memory_user(db, 81_085)
    bot = MemoryHarness(db)
    context = memory_context()
    sentinel = "PRIVATE_NOVA_MEMORY_SENTINEL"
    incoming = MemoryMessage(f"Nova, запомни: {sentinel}")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_085),
        context,
    )
    canonical = incoming.replies[0]["message"]
    preview = context.bot.edits[-1]
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_085,
    )
    assert current is not None
    assert sentinel not in repr(current)
    callbacks = [
        button.callback_data for row in preview["reply_markup"].inline_keyboard for button in row
    ]
    assert all(value.startswith("nmem:") and sentinel not in value for value in callbacks)

    async def fail_closed(**kwargs: Any) -> NovaMemoryMutation:
        del kwargs
        raise NovaMemoryStorageError(f"database failed with {sentinel}")

    monkeypatch.setattr(bot.nova_memory_service, "create", fail_closed)
    failed = await click(
        bot,
        user,
        91_085,
        canonical,
        context,
        preview["reply_markup"],
        "✅ Запомнить",
    )
    assert "не удалось" in failed.edits[-1]["text"].casefold()
    assert sentinel not in caplog.text
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 0


@pytest.mark.asyncio
async def test_exact_old_access_cleanup_preserves_newer_same_session_generation(db, monkeypatch):
    user = await memory_user(db, 81_086)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_086),
        context,
    )
    canonical = incoming.replies[0]["message"]
    old = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_086,
    )
    assert old is not None
    original_get_exact = bot.nova_memory_sessions.get_exact
    raced = False
    newer = None

    async def replace_after_exact(session):
        nonlocal raced, newer
        live = await original_get_exact(session)
        if live is not None and not raced:
            raced = True
            newer = await bot.nova_memory_sessions.update(live, phase=NovaMemoryFlowPhase.LIST)
        return live

    monkeypatch.setattr(bot.nova_memory_sessions, "get_exact", replace_after_exact)
    edits_before = len(canonical.edits)
    await bot._nova_memory_access_changed(
        context,
        old,
        source_message=canonical,
    )
    assert newer is not None
    assert await original_get_exact(newer) == newer
    assert len(canonical.edits) == edits_before


@pytest.mark.asyncio
async def test_delivery_rechecks_exact_generation_after_final_access_await(db, monkeypatch):
    user = await memory_user(db, 81_087)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_087),
        context,
    )
    canonical = incoming.replies[0]["message"]
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_087,
    )
    assert current is not None
    original_access = bot._nova_memory_access_values
    calls = 0
    newer = None

    async def transition_during_final_access(telegram_user_id: int, chat_id: int):
        nonlocal calls, newer
        calls += 1
        if calls == 2:
            newer = await bot.nova_memory_sessions.update(
                current,
                phase=NovaMemoryFlowPhase.LIST,
            )
            assert newer is not None
        return await original_access(telegram_user_id, chat_id)

    monkeypatch.setattr(
        bot,
        "_nova_memory_access_values",
        transition_during_final_access,
    )
    edits_before = len(canonical.edits)

    delivered = await bot._nova_memory_deliver(
        context,
        current,
        "OBSOLETE_MEMORY_SCREEN",
        None,
        query=None,
        source_message=canonical,
        operation="generation_race",
    )

    assert delivered is False
    assert calls == 2
    assert newer is not None
    assert await bot.nova_memory_sessions.get_exact(newer) == newer
    assert len(canonical.edits) == edits_before


@pytest.mark.asyncio
@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
async def test_query_post_edit_access_race_neutralizes_exact_read_screen(
    db,
    access_change,
):
    user = await memory_user(db, 81_090 if access_change == "downgrade" else 81_091)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_090)
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_090,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.DETAIL,
    )
    query = BlockingMemoryQuery("nmem:unused", canonical)
    private = "PRIVATE_QUERY_POST_EDIT_SENTINEL"
    task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            session,
            private,
            SimpleNamespace(private=True),
            query=query,
            source_message=canonical,
            operation="detail",
        )
    )
    await query.started.wait()
    service = AccessService(db)
    if access_change == "downgrade":
        await service.set_guest(user.telegram_id, source="post-edit-query")
    else:
        await service.set_guest(user.telegram_id, source="post-edit-query")
        await service.grant_subscriber(user.telegram_id, source="post-edit-query")
    query.release.set()

    assert await task is False
    assert [edit["text"] for edit in query.edits] == [
        private,
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    ]
    assert query.edits[-1]["reply_markup"] is None
    assert query.edits[-1]["parse_mode"] is None
    assert await bot.nova_memory_sessions.get_exact(session) is None
    assert canonical.replies == []


@pytest.mark.asyncio
async def test_bot_post_edit_access_bounce_neutralizes_exact_preview_without_source(db):
    user = await memory_user(db, 81_092)
    bot = MemoryHarness(db)
    blocking_bot = BlockingMemoryBot()
    context = SimpleNamespace(bot=blocking_bot, user_data={})
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_092,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=99_092,
        phase=NovaMemoryFlowPhase.CREATE_PREVIEW,
    )
    private = "PRIVATE_BOT_POST_EDIT_SENTINEL"
    task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            session,
            private,
            SimpleNamespace(private=True),
            query=None,
            source_message=None,
            operation="preview",
        )
    )
    await blocking_bot.started.wait()
    service = AccessService(db)
    await service.set_guest(user.telegram_id, source="post-edit-bot")
    await service.grant_subscriber(user.telegram_id, source="post-edit-bot")
    blocking_bot.release.set()

    assert await task is False
    assert [edit["text"] for edit in blocking_bot.edits] == [
        private,
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    ]
    assert blocking_bot.edits[-1]["chat_id"] == session.chat_id
    assert blocking_bot.edits[-1]["message_id"] == session.canonical_message_id
    assert blocking_bot.edits[-1]["reply_markup"] is None
    assert blocking_bot.edits[-1]["parse_mode"] is None
    assert await bot.nova_memory_sessions.get_exact(session) is None


@pytest.mark.asyncio
async def test_post_mutation_query_edit_race_commits_and_audits_exactly_once(
    db,
    monkeypatch,
):
    user = await memory_user(db, 81_097)
    bot = MemoryHarness(db)
    context = memory_context()
    private = "PRIVATE_POST_MUTATION_EDIT_RACE"
    incoming = MemoryMessage(f"Nova, запомни: {private}")
    await bot.nova_memory_text_gate(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_097),
        context,
    )
    canonical = incoming.replies[0]["message"]
    token = callback_for(context.bot.edits[-1]["reply_markup"], "✅ Запомнить")
    original_create = bot.nova_memory_service.create
    domain_calls = 0

    async def counted_create(**kwargs: Any):
        nonlocal domain_calls
        domain_calls += 1
        return await original_create(**kwargs)

    monkeypatch.setattr(bot.nova_memory_service, "create", counted_create)
    query = BlockingMemoryQuery(token, canonical)
    task = asyncio.create_task(
        bot.nova_memory_callback(
            memory_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=91_097,
                query=query,
            ),
            context,
        )
    )
    await query.started.wait()
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1
    await AccessService(db).set_guest(user.telegram_id, source="post-mutation-edit")
    query.release.set()
    await task

    assert domain_calls == 1
    assert query.answers == [{"args": ()}]
    assert len(query.edits) == 2
    assert private in query.edits[0]["text"]
    assert query.edits[1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert query.edits[1]["reply_markup"] is None
    assert query.edits[1]["parse_mode"] is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaMemoryItem.id))) == 1
        assert await session.scalar(select(func.count(NovaMemoryChange.id))) == 1

    replay = MemoryQuery(token, canonical)
    await bot.nova_memory_callback(
        memory_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_097,
            query=replay,
        ),
        context,
    )
    assert replay.answers == [{"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}]
    assert domain_calls == 1


@pytest.mark.asyncio
async def test_not_modified_primary_edit_still_runs_post_edit_access_compensation(db):
    user = await memory_user(db, 81_093)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_093)
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_093,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.LIST,
    )
    query = BlockingMemoryQuery(
        "nmem:unused",
        canonical,
        first_error=BadRequest("Message is not modified"),
    )
    task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            session,
            "PRIVATE_NOT_MODIFIED_SENTINEL",
            None,
            query=query,
            source_message=canonical,
            operation="list",
        )
    )
    await query.started.wait()
    await AccessService(db).set_guest(user.telegram_id, source="post-edit-not-modified")
    query.release.set()

    assert await task is False
    assert query.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert query.edits[-1]["reply_markup"] is None
    assert query.edits[-1]["parse_mode"] is None
    assert await bot.nova_memory_sessions.get_exact(session) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("access_change", ["downgrade", "bounce"])
async def test_same_canonical_old_version_replacement_is_preserved_after_neutralization(
    db,
    access_change,
):
    user = await memory_user(db, 81_094 if access_change == "downgrade" else 81_095)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_094)
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_094 if access_change == "downgrade" else 91_095,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.DETAIL,
    )
    query = BlockingMemoryQuery("nmem:unused", canonical)
    task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            session,
            "PRIVATE_OLD_GENERATION",
            None,
            query=query,
            source_message=canonical,
            operation="detail",
        )
    )
    await query.started.wait()
    replacement = await bot.nova_memory_sessions.update(
        session,
        phase=NovaMemoryFlowPhase.ROOT,
    )
    assert replacement is not None
    access = AccessService(db)
    await access.set_guest(user.telegram_id, source="post-edit-replacement")
    if access_change == "bounce":
        await access.grant_subscriber(user.telegram_id, source="post-edit-replacement")
    query.release.set()

    assert await task is False
    assert [edit["text"] for edit in query.edits] == [
        "PRIVATE_OLD_GENERATION",
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    ]
    assert query.edits[-1]["reply_markup"] is None
    assert query.edits[-1]["parse_mode"] is None
    assert await bot.nova_memory_sessions.get_exact(replacement) == replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_outcome", ["failed", "cancelled"])
async def test_fresh_same_canonical_replacement_render_failure_leaves_screen_neutralized(
    db,
    replacement_outcome,
):
    offset = 0 if replacement_outcome == "failed" else 1
    user = await memory_user(db, 81_110 + offset)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_110 + offset)
    old = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_110 + offset,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.DETAIL,
    )
    canonical_paints: list[str] = []
    old_query = CanonicalPaintBlockingMemoryQuery(
        "nmem:old",
        canonical,
        canonical_paints,
    )
    old_task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            old,
            "PRIVATE_OLD_SAME_CANONICAL",
            None,
            query=old_query,
            source_message=canonical,
            operation="detail",
        )
    )
    await old_query.started.wait()
    access = AccessService(db)
    await access.set_guest(user.telegram_id, source="same-canonical-replacement")
    await access.grant_subscriber(user.telegram_id, source="same-canonical-replacement")
    fresh = await bot._nova_memory_access_values(user.telegram_id, old.chat_id)
    assert fresh is not None
    replacement = await bot.nova_memory_sessions.create(
        owner_id=fresh.id,
        telegram_user_id=fresh.telegram_id,
        chat_id=old.chat_id,
        tier=SUBSCRIBER,
        access_version=fresh.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.ROOT,
    )
    error: BaseException = (
        BadRequest("replacement edit failed")
        if replacement_outcome == "failed"
        else asyncio.CancelledError()
    )
    replacement_query = CanonicalPaintMemoryQuery(
        "nmem:replacement",
        canonical,
        canonical_paints,
        error=error,
    )
    replacement_task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            replacement,
            "PRIVATE_REPLACEMENT_SCREEN",
            None,
            query=replacement_query,
            source_message=canonical,
            operation="root",
        )
    )
    await asyncio.sleep(0)
    assert bot._nova_memory_ui_lock.locked()
    assert not replacement_task.done()
    assert not replacement_query.started.is_set()

    old_query.release.set()
    assert await old_task is False
    assert [edit["text"] for edit in old_query.edits] == [
        "PRIVATE_OLD_SAME_CANONICAL",
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    ]
    assert old_query.edits[-1]["reply_markup"] is None
    assert old_query.edits[-1]["parse_mode"] is None
    assert await bot.nova_memory_sessions.get_exact(replacement) == replacement

    if replacement_outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await replacement_task
    else:
        assert await replacement_task is False
    assert replacement_query.started.is_set()
    assert await bot.nova_memory_sessions.get_exact(replacement) == replacement
    assert canonical_paints == [
        "PRIVATE_OLD_SAME_CANONICAL",
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    ]


@pytest.mark.asyncio
async def test_fresh_permitted_same_canonical_replacement_can_render_after_neutralization(db):
    user = await memory_user(db, 81_122)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_122)
    old = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_122,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.DETAIL,
    )
    canonical_paints: list[str] = []
    old_query = CanonicalPaintBlockingMemoryQuery(
        "nmem:old",
        canonical,
        canonical_paints,
    )
    old_task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            old,
            "PRIVATE_STALE_SCREEN",
            None,
            query=old_query,
            source_message=canonical,
            operation="detail",
        )
    )
    await old_query.started.wait()
    access = AccessService(db)
    await access.set_guest(user.telegram_id, source="fresh-replacement")
    await access.grant_subscriber(user.telegram_id, source="fresh-replacement")
    fresh = await bot._nova_memory_access_values(user.telegram_id, old.chat_id)
    assert fresh is not None
    replacement = await bot.nova_memory_sessions.create(
        owner_id=fresh.id,
        telegram_user_id=fresh.telegram_id,
        chat_id=old.chat_id,
        tier=SUBSCRIBER,
        access_version=fresh.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.ROOT,
    )
    replacement_query = CanonicalPaintMemoryQuery(
        "nmem:fresh",
        canonical,
        canonical_paints,
    )
    replacement_task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            replacement,
            "FRESH_PERMITTED_REPLACEMENT_SCREEN",
            None,
            query=replacement_query,
            source_message=canonical,
            operation="root",
        )
    )
    await asyncio.sleep(0)
    assert bot._nova_memory_ui_lock.locked()
    assert not replacement_task.done()
    assert not replacement_query.started.is_set()

    old_query.release.set()
    assert await old_task is False
    assert await replacement_task is True
    assert replacement_query.started.is_set()
    assert [edit["text"] for edit in replacement_query.edits] == [
        "FRESH_PERMITTED_REPLACEMENT_SCREEN"
    ]
    assert canonical_paints == [
        "PRIVATE_STALE_SCREEN",
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
        "FRESH_PERMITTED_REPLACEMENT_SCREEN",
    ]
    assert await bot.nova_memory_sessions.get_exact(replacement) == replacement


@pytest.mark.asyncio
async def test_external_cancellation_does_not_cancel_blocked_post_edit_access_fence(
    db,
    monkeypatch,
    caplog,
):
    user = await memory_user(db, 81_123)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_123)
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_123,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.DETAIL,
    )
    query = MemoryQuery("nmem:blocked-access", canonical)
    original_access = bot._nova_memory_access_values
    access_started = asyncio.Event()
    access_release = asyncio.Event()
    calls = 0

    async def blocked_post_edit_access(telegram_user_id: int, chat_id: int):
        nonlocal calls
        calls += 1
        if calls == 3:
            access_started.set()
            await access_release.wait()
        return await original_access(telegram_user_id, chat_id)

    monkeypatch.setattr(bot, "_nova_memory_access_values", blocked_post_edit_access)
    captured_inner: list[asyncio.Task[bool]] = []
    original_create_task = asyncio.create_task

    def capture_named_task(coro, *, name=None, context=None):
        task = original_create_task(coro, name=name, context=context)
        if name is not None and "nova-memory-post-edit" in name:
            captured_inner.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", capture_named_task)
    outer = original_create_task(
        bot._nova_memory_deliver(
            context,
            session,
            "PRIVATE_BLOCKED_POST_EDIT_SCREEN",
            None,
            query=query,
            source_message=canonical,
            operation="detail",
        )
    )
    await access_started.wait()
    await AccessService(db).set_guest(user.telegram_id, source="cancelled-post-edit")
    outer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await outer
    assert captured_inner and not captured_inner[0].done()
    assert captured_inner[0] in bot._nova_memory_post_edit_tasks
    assert bot._nova_memory_ui_lock.locked()
    access_release.set()
    with caplog.at_level(logging.WARNING, logger="future_self.nova_memory_handlers"):
        assert await captured_inner[0] is False

    assert [edit["text"] for edit in query.edits] == [
        "PRIVATE_BLOCKED_POST_EDIT_SCREEN",
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    ]
    assert query.edits[-1]["reply_markup"] is None
    assert query.edits[-1]["parse_mode"] is None
    assert await bot.nova_memory_sessions.get_exact(session) is None
    assert not bot._nova_memory_ui_lock.locked()
    assert bot._nova_memory_post_edit_tasks == set()
    assert "Nova memory" not in caplog.text


@pytest.mark.asyncio
async def test_primary_telegram_cancellation_creates_no_post_edit_fence_and_releases_lock(
    db,
    monkeypatch,
):
    user = await memory_user(db, 81_124)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_124)
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_124,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.ROOT,
    )
    query = FailingMemoryQuery(
        "nmem:cancelled-primary",
        canonical,
        asyncio.CancelledError(),
    )
    created_names: list[str | None] = []
    original_create_task = asyncio.create_task

    def capture_task(coro, *, name=None, context=None):
        created_names.append(name)
        return original_create_task(coro, name=name, context=context)

    monkeypatch.setattr(asyncio, "create_task", capture_task)

    with pytest.raises(asyncio.CancelledError):
        await bot._nova_memory_deliver(
            context,
            session,
            "PRIVATE_CANCELLED_PRIMARY_SCREEN",
            None,
            query=query,
            source_message=canonical,
            operation="root",
        )

    assert "nova-memory-post-edit-fence" not in created_names
    assert not bot._nova_memory_ui_lock.locked()
    assert getattr(bot, "_nova_memory_post_edit_tasks", set()) == set()
    assert await bot.nova_memory_sessions.get_exact(session) == session


@pytest.mark.asyncio
async def test_post_edit_access_loss_neutralizes_cleared_old_generation(db):
    user = await memory_user(db, 81_099)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_099)
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_099,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.DETAIL,
    )
    query = BlockingMemoryQuery("nmem:unused", canonical)
    private = "PRIVATE_CLEARED_GENERATION_SENTINEL"
    task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            session,
            private,
            None,
            query=query,
            source_message=canonical,
            operation="detail",
        )
    )
    await query.started.wait()
    assert await bot.nova_memory_sessions.clear_exact(session) is True
    await AccessService(db).set_guest(user.telegram_id, source="post-edit-cleared")
    query.release.set()

    assert await task is False
    assert [edit["text"] for edit in query.edits] == [
        private,
        NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    ]
    assert query.edits[-1]["reply_markup"] is None
    assert query.edits[-1]["parse_mode"] is None
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=session.chat_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_compensation_error_is_safe_and_cancelled_error_propagates(db, caplog):
    user = await memory_user(db, 81_095)
    bot = MemoryHarness(db)
    context = memory_context()
    canonical = MemoryMessage(message_id=99_095)
    session = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_095,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        phase=NovaMemoryFlowPhase.LIST,
    )
    query = BlockingMemoryQuery(
        "nmem:unused",
        canonical,
        compensation_error=RuntimeError("PRIVATE_COMPENSATION_ERROR_BODY"),
    )
    task = asyncio.create_task(
        bot._nova_memory_deliver(
            context,
            session,
            "PRIVATE_COMPENSATION_CONTENT",
            None,
            query=query,
            source_message=canonical,
            operation="list",
        )
    )
    await query.started.wait()
    await AccessService(db).set_guest(user.telegram_id, source="post-edit-compensation")
    query.release.set()
    with caplog.at_level(logging.WARNING, logger="future_self.nova_memory_handlers"):
        assert await task is False
    assert await bot.nova_memory_sessions.get_exact(session) is None
    assert "operation=access_compensation" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "PRIVATE_COMPENSATION_ERROR_BODY" not in caplog.text
    assert "PRIVATE_COMPENSATION_CONTENT" not in caplog.text

    fresh_user = await memory_user(db, 81_096)
    cancel_bot = MemoryHarness(db)
    cancel_context = memory_context()
    cancel_session = await cancel_bot.nova_memory_sessions.create(
        owner_id=fresh_user.id,
        telegram_user_id=fresh_user.telegram_id,
        chat_id=91_096,
        tier=SUBSCRIBER,
        access_version=fresh_user.access_version,
        canonical_message_id=99_096,
        phase=NovaMemoryFlowPhase.ROOT,
    )
    cancelled = FailingMemoryQuery(
        "nmem:unused",
        MemoryMessage(message_id=99_096),
        asyncio.CancelledError(),
    )
    with pytest.raises(asyncio.CancelledError):
        await cancel_bot._nova_memory_deliver(
            cancel_context,
            cancel_session,
            "PRIVATE_CANCELLED_CONTENT",
            None,
            query=cancelled,
            source_message=cancelled.message,
            operation="root",
        )
    assert await cancel_bot.nova_memory_sessions.get_exact(cancel_session) == cancel_session

    compensation_user = await memory_user(db, 81_098)
    compensation_bot = MemoryHarness(db)
    compensation_context = memory_context()
    compensation_session = await compensation_bot.nova_memory_sessions.create(
        owner_id=compensation_user.id,
        telegram_user_id=compensation_user.telegram_id,
        chat_id=91_098,
        tier=SUBSCRIBER,
        access_version=compensation_user.access_version,
        canonical_message_id=99_098,
        phase=NovaMemoryFlowPhase.DETAIL,
    )
    compensation_query = BlockingMemoryQuery(
        "nmem:unused",
        MemoryMessage(message_id=99_098),
        compensation_error=asyncio.CancelledError(),
    )
    compensation_task = asyncio.create_task(
        compensation_bot._nova_memory_deliver(
            compensation_context,
            compensation_session,
            "PRIVATE_COMPENSATION_CANCELLED",
            None,
            query=compensation_query,
            source_message=compensation_query.message,
            operation="detail",
        )
    )
    await compensation_query.started.wait()
    await AccessService(db).set_guest(
        compensation_user.telegram_id,
        source="post-edit-compensation-cancelled",
    )
    compensation_query.release.set()
    with pytest.raises(asyncio.CancelledError):
        await compensation_task
    assert await compensation_bot.nova_memory_sessions.get_exact(compensation_session) is None
    assert len(compensation_query.edits) == 2


@pytest.mark.asyncio
async def test_old_canonical_callback_cannot_clear_access_bounced_replacement(db):
    user = await memory_user(db, 81_088)
    bot = MemoryHarness(db)
    context = memory_context()
    incoming = MemoryMessage("/mynova")
    await bot.nova_memory_command(
        memory_update(incoming, telegram_user_id=user.telegram_id, chat_id=91_088),
        context,
    )
    old_canonical = incoming.replies[0]["message"]
    old_token = callback_for(
        old_canonical.edits[-1]["reply_markup"],
        "❓ Как это работает",
    )
    replacement = await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=91_088,
        tier=SUBSCRIBER,
        access_version=user.access_version,
        canonical_message_id=98_088,
        phase=NovaMemoryFlowPhase.ROOT,
    )
    await AccessService(db).set_guest(user.telegram_id, source="memory-old-callback")
    await AccessService(db).grant_subscriber(user.telegram_id, source="memory-old-callback")
    query = MemoryQuery(old_token, old_canonical)

    await bot.nova_memory_callback(
        memory_update(
            old_canonical,
            telegram_user_id=user.telegram_id,
            chat_id=91_088,
            query=query,
        ),
        context,
    )

    assert query.answers == [{"args": (NOVA_MEMORY_STALE_ALERT,), "show_alert": True}]
    assert query.edits == []
    assert await bot.nova_memory_sessions.get_exact(replacement) == replacement
    assert context.bot.edits == []
