from __future__ import annotations

import asyncio
import logging
from itertools import count
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop

from future_self.access import ADMIN, BLOCKED, GUEST, SUBSCRIBER, AccessService
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import Base, ConversationMessage, DraftInboxItem, Goal, InboxItem
from future_self.nova import NovaSessionStore
from future_self.nova_handlers import (
    NOVA_ACCESS_CHANGED_TEXT,
    NOVA_BUSY_ALERT,
    NOVA_LOCAL_CLARIFY_TEXT,
    NOVA_PROVIDER_FAILURE_TEXT,
    NOVA_ROOT_TEXT,
    NOVA_STALE_ALERT,
)
from future_self.nova_memory_flow import NovaMemoryFlowPhase
from future_self.nova_memory_handlers import (
    NOVA_MEMORY_ACCESS_CHANGED_TEXT,
    NOVA_MEMORY_VOICE_ACCESS_CHANGED_TEXT,
)
from future_self.schemas import NovaHelpPlan
from future_self.transcription import TranscriptionError


class NovaAIStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.other_calls: list[str] = []
        self.result = NovaHelpPlan(
            response="Ответ Nova",
            steps=["Первый шаг"],
            action_id=None,
            kind="clarify",
        )
        self.error: BaseException | None = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def nova_help(self, question: str, catalog: Any) -> NovaHelpPlan:
        self.calls.append((question, catalog))
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def _forbidden(self, name: str) -> None:
        self.other_calls.append(name)
        raise AssertionError(f"Nova called unrelated AI method: {name}")

    async def route_message(self, *_args: Any, **_kwargs: Any) -> None:
        await self._forbidden("route_message")

    async def answer_message(self, *_args: Any, **_kwargs: Any) -> None:
        await self._forbidden("answer_message")

    async def parse_thought(self, *_args: Any, **_kwargs: Any) -> None:
        await self._forbidden("parse_thought")

    async def guest_thought_breakdown(self, *_args: Any, **_kwargs: Any) -> None:
        await self._forbidden("guest_thought_breakdown")

    async def guest_first_step(self, *_args: Any, **_kwargs: Any) -> None:
        await self._forbidden("guest_first_step")


class NovaMessage:
    _ids = count(50_000)

    def __init__(
        self,
        text: str | None = None,
        *,
        message_id: int | None = None,
        voice: Any = None,
        audio: Any = None,
        photo: Any = None,
        document: Any = None,
        edit_error: BaseException | None = None,
    ) -> None:
        self.text = text
        self.message_id = message_id if message_id is not None else next(self._ids)
        self.voice = voice
        self.audio = audio
        self.photo = photo
        self.document = document
        self.edit_error = edit_error
        self.reply_to_message = None
        self.replies: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted = 0

    async def reply_text(self, text: str, **kwargs: Any) -> NovaMessage:
        sent = NovaMessage(text, edit_error=self.edit_error)
        self.replies.append({"text": text, "message": sent, **kwargs})
        return sent

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        self.edits.append({"text": text, **kwargs})

    async def delete(self) -> None:
        self.deleted += 1


class BlockingReplyNovaMessage(NovaMessage):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.reply_started = asyncio.Event()
        self.reply_release = asyncio.Event()

    async def reply_text(self, text: str, **kwargs: Any) -> NovaMessage:
        self.reply_started.set()
        await self.reply_release.wait()
        return await super().reply_text(text, **kwargs)


class NovaTelegramFile:
    async def download_as_bytearray(self) -> bytearray:
        return bytearray(b"nova-voice")


class NovaVoice:
    duration = 3
    file_size = 10
    mime_type = "audio/ogg"
    file_name = "voice.ogg"

    async def get_file(self) -> NovaTelegramFile:
        return NovaTelegramFile()


class NovaTranscription:
    enabled = True

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls.append((audio, filename))
        return self.transcript


class HookedNovaTranscription(NovaTranscription):
    def __init__(self, transcript: str, hook: Any) -> None:
        super().__init__(transcript)
        self.hook = hook

    async def transcribe(self, audio: bytes, filename: str) -> str:
        value = await super().transcribe(audio, filename)
        await self.hook()
        return value


class BarrierNovaTranscription(NovaTranscription):
    def __init__(self, transcript: str, participants: int = 2) -> None:
        super().__init__(transcript)
        self.participants = participants
        self.ready = asyncio.Event()

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls.append((audio, filename))
        if len(self.calls) >= self.participants:
            self.ready.set()
        await self.ready.wait()
        return self.transcript


class BlockingNovaTranscription(NovaTranscription):
    def __init__(self, transcript: str) -> None:
        super().__init__(transcript)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls.append((audio, filename))
        self.started.set()
        await self.release.wait()
        return self.transcript


class FailingNovaTranscription(NovaTranscription):
    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.calls.append((audio, filename))
        raise TranscriptionError("PRIVATE_TRANSCRIPTION_ERROR")


class NovaQuery:
    def __init__(
        self,
        data: str,
        message: NovaMessage,
        *,
        edit_error: BaseException | None = None,
        answer_hook: Any = None,
        edit_started: asyncio.Event | None = None,
        edit_release: asyncio.Event | None = None,
    ) -> None:
        self.data = data
        self.message = message
        self.edit_error = edit_error
        self.answer_hook = answer_hook
        self.edit_started = edit_started
        self.edit_release = edit_release
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[dict[str, Any]] = []
        self.caption_edits: list[dict[str, Any]] = []
        self.retired = 0

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        if self.answer_hook is not None:
            hook = self.answer_hook
            self.answer_hook = None
            await hook()
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        if self.edit_started is not None:
            self.edit_started.set()
        if self.edit_release is not None:
            await self.edit_release.wait()
        self.edits.append({"text": text, **kwargs})

    async def edit_message_caption(self, caption: str, **kwargs: Any) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        self.caption_edits.append({"text": caption, **kwargs})

    async def edit_message_reply_markup(self, reply_markup: Any = None) -> None:
        del reply_markup
        self.retired += 1


class NovaTelegramBot:
    def __init__(self) -> None:
        self.edits: list[dict[str, Any]] = []
        self.caption_edits: list[dict[str, Any]] = []
        self.retired: list[tuple[int, int]] = []
        self.sent: list[dict[str, Any]] = []
        self.deleted: list[tuple[int, int]] = []
        self.edit_error: BaseException | None = None

    async def edit_message_text(self, **kwargs: Any) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        self.edits.append(kwargs)

    async def edit_message_caption(self, **kwargs: Any) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        self.caption_edits.append(kwargs)

    async def edit_message_reply_markup(
        self,
        *,
        chat_id: int,
        message_id: int,
        reply_markup: Any,
    ) -> None:
        del reply_markup
        self.retired.append((chat_id, message_id))

    async def send_message(self, **kwargs: Any) -> NovaMessage:
        sent = NovaMessage(kwargs["text"])
        self.sent.append({"message": sent, **kwargs})
        return sent

    async def delete_message(self, *, chat_id: int, message_id: int) -> None:
        self.deleted.append((chat_id, message_id))


def nova_context(telegram: NovaTelegramBot | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        user_data={},
        args=[],
        bot=telegram or NovaTelegramBot(),
        application=SimpleNamespace(create_task=lambda *_args, **_kwargs: None),
    )


def update_for(
    message: NovaMessage,
    *,
    user_id: int,
    chat_id: int | None = None,
    query: NovaQuery | None = None,
) -> SimpleNamespace:
    private_chat_id = user_id if chat_id is None else chat_id
    return SimpleNamespace(
        update_id=1,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=private_chat_id, type="private"),
        effective_message=message,
        message=message,
        callback_query=query,
    )


def button_matrix(markup: Any) -> list[list[tuple[str, str | None]]]:
    return [
        [(button.text, button.callback_data) for button in row] for row in markup.inline_keyboard
    ]


def callback_with_prefix(markup: Any, prefix: str) -> str:
    for row in markup.inline_keyboard:
        for button in row:
            if button.callback_data and button.callback_data.startswith(prefix):
                return button.callback_data
    raise AssertionError(f"callback with prefix {prefix!r} was not rendered")


def make_bot(db: Any, ai: NovaAIStub, **overrides: Any) -> FutureSelfBot:
    knowledge_asset_root = overrides.pop("_knowledge_asset_root", None)
    transcription = overrides.pop("_transcription", SimpleNamespace(enabled=False))
    values: dict[str, Any] = {
        "_env_file": None,
        "telegram_bot_token": "123456:test-token",
        "ai_api_key": "test-key",
        "database_url": db.url,
    }
    values.update(overrides)
    settings = Settings(**values)
    if knowledge_asset_root is not None:
        settings.knowledge_asset_root = str(knowledge_asset_root)
    return FutureSelfBot(
        settings,
        db,
        ai,
        transcription,
    )


async def content_row_counts(db: Any) -> tuple[int, int]:
    async with db.sessions() as session:
        conversation_messages = await session.scalar(select(func.count(ConversationMessage.id)))
        inbox_items = await session.scalar(select(func.count(InboxItem.id)))
    return int(conversation_messages or 0), int(inbox_items or 0)


async def user_with_tier(bot: FutureSelfBot, telegram_id: int, tier: str) -> Any:
    await bot._user(telegram_id)
    service = AccessService(bot.db)
    if tier == ADMIN:
        await service.grant_admin(telegram_id, source="nova-test")
    elif tier == SUBSCRIBER:
        await service.grant_subscriber(telegram_id, source="nova-test")
    elif tier == BLOCKED:
        await service.block(telegram_id, source="nova-test")
    elif tier != GUEST:
        raise AssertionError(f"unsupported test tier: {tier}")
    return await bot._user(telegram_id)


async def open_nova(
    bot: FutureSelfBot,
    *,
    user_id: int,
    context: SimpleNamespace | None = None,
) -> tuple[NovaMessage, NovaMessage, SimpleNamespace]:
    command = NovaMessage("/help")
    actual_context = context or nova_context()
    await bot.help_command(update_for(command, user_id=user_id), actual_context)
    assert len(command.replies) == 1
    return command, command.replies[0]["message"], actual_context


async def seed_memory_flow(
    bot: FutureSelfBot,
    user: Any,
    *,
    chat_id: int,
    canonical_message_id: int,
    phase: NovaMemoryFlowPhase = NovaMemoryFlowPhase.ROOT,
) -> Any:
    return await bot.nova_memory_sessions.create(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=chat_id,
        tier=user.access_tier,
        access_version=user.access_version,
        canonical_message_id=canonical_message_id,
        phase=phase,
    )


@pytest.mark.asyncio
async def test_help_command_and_navigation_help_open_the_exact_same_nova_root(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_001
    await user_with_tier(bot, user_id, SUBSCRIBER)

    command, _canonical, command_context = await open_nova(bot, user_id=user_id)
    command_screen = command.replies[0]
    assert command_screen["text"] == NOVA_ROOT_TEXT
    assert button_matrix(command_screen["reply_markup"]) == [
        [("🚀 Быстрый старт", "nova:topic:quick")],
        [("🧭 Возможности", "nova:topic:requests")],
        [("💬 Примеры вопросов", "nova:topic:examples")],
        [("🔒 Данные и безопасность", "nova:topic:privacy")],
        [("🏠 Главное меню", "nav:root")],
    ]
    assert "Nova" in command_screen["text"]
    assert all(name not in command_screen["text"] for name in ("Jarvis", "Джарвис", "Нова"))
    assert command_context.bot.sent == []

    menu_message = NovaMessage("Главное меню")
    query = NovaQuery("nav:help", menu_message)
    nav_context = nova_context()
    await bot.navigation_action(
        update_for(menu_message, user_id=user_id, query=query),
        nav_context,
    )

    assert query.answers == [(None, False)]
    assert len(query.edits) == 1
    assert query.edits[0]["text"] == command_screen["text"]
    assert button_matrix(query.edits[0]["reply_markup"]) == button_matrix(
        command_screen["reply_markup"]
    )
    assert menu_message.replies == []
    assert nav_context.bot.sent == []
    assert ai.calls == []


@pytest.mark.asyncio
async def test_local_question_edits_only_the_canonical_message_and_never_calls_ai(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = 61_002
    await user_with_tier(bot, user_id, ADMIN)
    _command, canonical, context = await open_nova(bot, user_id=user_id)

    question = NovaMessage("Как добавить задачу с напоминанием?")
    handled = await bot.nova_text_gate(update_for(question, user_id=user_id), context)

    assert handled is True
    assert ai.calls == []
    assert question.replies == []
    assert context.bot.sent == []
    assert len(context.bot.edits) == 1
    edit = context.bot.edits[0]
    assert edit["message_id"] == canonical.message_id
    assert edit["text"].startswith("✨ Nova\n\n")
    assert "1. " in edit["text"]
    assert callback_with_prefix(edit["reply_markup"], "nova:action:task_create:")


@pytest.mark.asyncio
async def test_guided_nova_question_clears_only_the_exact_memory_flow(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_081
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    exact = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=81_001,
    )
    other = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id + 1,
        canonical_message_id=81_002,
    )
    message = NovaMessage("Nova, как добавить задачу с напоминанием?")

    assert await bot.nova_text_gate(
        update_for(message, user_id=user_id),
        nova_context(),
        user=user,
    )

    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id + 1,
        )
        == other
    )
    assert exact.id != other.id
    assert ai.calls == []


@pytest.mark.asyncio
async def test_explicit_memory_intent_clears_guided_nova_and_reminder_flows(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_082
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    reminder_message = NovaMessage("Каждый день напоминай в 20:30 заполнить дневник")
    assert await bot.reminder_text_gate(
        update_for(reminder_message, user_id=user_id),
        context,
    )
    reminder = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert reminder is not None
    guided = await bot.nova_sessions.create(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
        access_version=user.access_version,
        canonical_message_id=82_001,
        tier=SUBSCRIBER,
    )

    memory_message = NovaMessage("Nova, запомни: Я предпочитаю короткие ответы")
    assert await bot.nova_memory_text_gate(
        update_for(memory_message, user_id=user_id),
        context,
        user=user,
    )

    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    memory = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert memory is not None
    assert memory.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    assert guided.id not in repr(memory)


@pytest.mark.asyncio
async def test_navigation_routes_memory_before_reminder_nova_and_generic(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_091
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    reminder_gate = AsyncMock(return_value=False)
    nova_gate = AsyncMock(return_value=False)
    generic = AsyncMock()
    bot.reminder_text_gate = reminder_gate
    bot.nova_text_gate = nova_gate
    bot._route_message = generic
    message = NovaMessage("Nova, запомни: Я люблю проверяемые планы")

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(
            update_for(message, user_id=user_id),
            nova_context(),
        )

    reminder_gate.assert_not_awaited()
    nova_gate.assert_not_awaited()
    generic.assert_not_awaited()
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_concurrent_memory_and_reminder_launch_leave_one_current_owner(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_093
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    memory_message = BlockingReplyNovaMessage("Nova, запомни: Я предпочитаю точные ответы")
    memory_update = update_for(memory_message, user_id=user_id)
    memory_task = asyncio.create_task(bot.nova_memory_text_gate(memory_update, context, user=user))
    await memory_message.reply_started.wait()

    reminder_message = NovaMessage("Каждый день напоминай в 20:30 заполнить дневник")
    reminder_task = asyncio.create_task(
        bot.reminder_text_gate(
            update_for(reminder_message, user_id=user_id),
            context,
        )
    )
    await asyncio.sleep(0)
    assert not reminder_task.done()

    memory_message.reply_release.set()
    assert await memory_task
    assert await reminder_task

    memory = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    reminder = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert memory is None
    assert reminder is not None
    assert reminder.canonical_message_id == reminder_message.replies[0]["message"].message_id
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_concurrent_memory_and_guided_nova_launch_leave_one_current_owner(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_094
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    memory_message = BlockingReplyNovaMessage("Nova, запомни: Я предпочитаю точные ответы")
    memory_update = update_for(memory_message, user_id=user_id)
    memory_task = asyncio.create_task(bot.nova_memory_text_gate(memory_update, context, user=user))
    await memory_message.reply_started.wait()

    guided_message = NovaMessage("Nova, как добавить задачу с напоминанием?")
    guided_task = asyncio.create_task(
        bot.nova_text_gate(
            update_for(guided_message, user_id=user_id),
            context,
            user=user,
        )
    )
    await asyncio.sleep(0)
    assert not guided_task.done()

    memory_message.reply_release.set()
    assert await memory_task
    assert await guided_task

    memory = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    guided = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert memory is None
    assert guided is not None
    assert guided.canonical_message_id == guided_message.replies[0]["message"].message_id
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_reverse_concurrent_guided_then_memory_launch_keeps_last_publisher(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_095
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    guided_pre_clear_done = asyncio.Event()
    allow_guided_publish = asyncio.Event()
    guided_post_clear_started = asyncio.Event()
    original_memory_clear = bot.nova_memory_clear_current
    original_guided_cleanup = bot._nova_memory_clear_if_guided_current
    clear_calls = 0
    cleanup_calls = 0

    async def controlled_memory_clear(update: Any) -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            await original_memory_clear(update)
            guided_pre_clear_done.set()
            await allow_guided_publish.wait()
            return
        await original_memory_clear(update)

    async def controlled_guided_cleanup(session: Any) -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        guided_post_clear_started.set()
        return await original_guided_cleanup(session)

    bot.nova_memory_clear_current = controlled_memory_clear
    bot._nova_memory_clear_if_guided_current = controlled_guided_cleanup
    guided_message = NovaMessage("Nova, как добавить задачу с напоминанием?")
    guided_task = asyncio.create_task(
        bot.nova_text_gate(
            update_for(guided_message, user_id=user_id),
            context,
            user=user,
        )
    )
    await guided_pre_clear_done.wait()

    memory_message = BlockingReplyNovaMessage("Nova, запомни: Я предпочитаю точные ответы")
    memory_task = asyncio.create_task(
        bot.nova_memory_text_gate(
            update_for(memory_message, user_id=user_id),
            context,
            user=user,
        )
    )
    await memory_message.reply_started.wait()
    allow_guided_publish.set()
    await guided_post_clear_started.wait()
    memory_message.reply_release.set()

    assert await memory_task
    assert await guided_task
    memory = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    guided = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert memory is not None
    assert guided is None
    assert memory.canonical_message_id == memory_message.replies[0]["message"].message_id
    assert clear_calls == 1
    assert cleanup_calls == 1
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_reverse_concurrent_reminder_then_memory_launch_keeps_last_publisher(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_096
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    reminder_pre_clear_done = asyncio.Event()
    allow_reminder_publish = asyncio.Event()
    reminder_post_clear_started = asyncio.Event()
    original_memory_clear = bot.nova_memory_clear_current
    original_reminder_cleanup = bot._nova_memory_clear_if_reminder_current
    clear_calls = 0
    cleanup_calls = 0

    async def controlled_memory_clear(update: Any) -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            await original_memory_clear(update)
            reminder_pre_clear_done.set()
            await allow_reminder_publish.wait()
            return
        await original_memory_clear(update)

    async def controlled_reminder_cleanup(session: Any) -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        reminder_post_clear_started.set()
        return await original_reminder_cleanup(session)

    bot.nova_memory_clear_current = controlled_memory_clear
    bot._nova_memory_clear_if_reminder_current = controlled_reminder_cleanup
    reminder_message = NovaMessage("Каждый день напоминай в 20:30 заполнить дневник")
    reminder_task = asyncio.create_task(
        bot.reminder_text_gate(
            update_for(reminder_message, user_id=user_id),
            context,
        )
    )
    await reminder_pre_clear_done.wait()

    memory_message = BlockingReplyNovaMessage("Nova, запомни: Я предпочитаю точные ответы")
    memory_task = asyncio.create_task(
        bot.nova_memory_text_gate(
            update_for(memory_message, user_id=user_id),
            context,
            user=user,
        )
    )
    await memory_message.reply_started.wait()
    allow_reminder_publish.set()
    await reminder_post_clear_started.wait()
    memory_message.reply_release.set()

    assert await memory_task
    assert await reminder_task
    memory = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    reminder = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert memory is not None
    assert reminder is None
    assert memory.canonical_message_id == memory_message.replies[0]["message"].message_id
    assert clear_calls == 1
    assert cleanup_calls == 1
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_durable_flow_keeps_ownership_over_explicit_memory_text(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_092
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    context.user_data["evening"] = {}
    memory_gate = AsyncMock(return_value=True)
    reminder_gate = AsyncMock(return_value=True)
    nova_gate = AsyncMock(return_value=True)
    bot.nova_memory_text_gate = memory_gate
    bot.reminder_text_gate = reminder_gate
    bot.nova_text_gate = nova_gate
    message = NovaMessage("Nova, запомни: Не потерять активный сценарий")

    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(
            update_for(message, user_id=user_id),
            context,
        )

    memory_gate.assert_not_awaited()
    reminder_gate.assert_not_awaited()
    nova_gate.assert_not_awaited()
    assert message.replies
    assert "не завершён сценарий" in message.replies[-1]["text"]
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_unlisted_goals_command_is_contained_by_active_memory_preview(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_097
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    command = NovaMessage("Nova, запомни: Я предпочитаю точные ответы")
    assert await bot.nova_memory_text_gate(
        update_for(command, user_id=user_id),
        context,
        user=user,
    )
    before = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert before is not None and before.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    canonical = command.replies[0]["message"]
    goals_handler = AsyncMock()
    bot.goals_command = goals_handler

    with pytest.raises(ApplicationHandlerStop):
        await bot.onboarding_command_gate(
            update_for(NovaMessage("/goals"), user_id=user_id),
            context,
        )

    goals_handler.assert_not_awaited()
    after = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert after is not None
    assert after.id == before.id
    assert after.version > before.version
    assert after.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    prompt = context.bot.edits[-1]
    assert prompt["message_id"] == canonical.message_id
    assert "не завершено действие с памятью" in prompt["text"]
    matrix = button_matrix(prompt["reply_markup"])
    assert [[label for label, _callback in row] for row in matrix] == [
        ["▶️ Продолжить"],
        ["🏠 Выйти в главное меню"],
    ]
    callbacks = [callback for row in matrix for _label, callback in row]
    assert all(
        callback is not None and callback.startswith("nmem:") and len(callback.encode()) <= 64
        for callback in callbacks
    )
    assert len(set(callbacks)) == 2
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(Goal.id))) == 0
    assert ai.calls == []
    assert ai.other_calls == []


@pytest.mark.asyncio
async def test_unlisted_goals_command_clears_browsing_memory_and_falls_through(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_098
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    root = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=98_001,
    )
    context = nova_context()

    await bot.onboarding_command_gate(
        update_for(NovaMessage("/goals"), user_id=user_id),
        context,
    )

    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert root.phase is NovaMemoryFlowPhase.ROOT
    assert context.bot.edits == []
    assert ai.calls == []
    assert ai.other_calls == []


@pytest.mark.asyncio
async def test_cancel_keeps_durable_flow_owner_and_retires_memory(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_099
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    memory = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=99_001,
        phase=NovaMemoryFlowPhase.CREATE_PREVIEW,
    )
    context = nova_context()
    context.user_data["evening"] = {}
    message = NovaMessage("/cancel")

    await bot.onboarding_command_gate(
        update_for(message, user_id=user_id),
        context,
    )

    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert memory.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    assert message.replies == []
    assert context.bot.edits == []
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_nmem_callback_bypasses_workspace_and_knowledge_cleanup(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_workspace_access=True,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    workspace_cancel = AsyncMock()
    reminder_clear = AsyncMock()
    guided_clear = AsyncMock()
    knowledge_cancel = AsyncMock()
    bot.workspace_service.cancel_input = workspace_cancel
    bot.reminder_clear_current = reminder_clear
    bot.nova_clear_current = guided_clear
    bot.cancel_knowledge_state = knowledge_cancel
    message = NovaMessage("memory callback")
    query = NovaQuery("nmem:opaque_capability", message)

    await bot.knowledge_other_callback_gate(
        update_for(message, user_id=61_100, query=query),
        nova_context(),
    )

    workspace_cancel.assert_not_awaited()
    reminder_clear.assert_not_awaited()
    guided_clear.assert_not_awaited()
    knowledge_cancel.assert_not_awaited()
    assert query.answers == []
    assert query.edits == []
    assert ai.calls == []
    assert ai.other_calls == []


@pytest.mark.asyncio
async def test_cancel_stops_memory_before_reminder_and_guided_nova(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_083
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    reminder_message = NovaMessage("Каждый день напоминай в 20:30 заполнить дневник")
    assert await bot.reminder_text_gate(
        update_for(reminder_message, user_id=user_id),
        context,
    )
    reminder = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert reminder is not None
    guided = await bot.nova_sessions.create(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
        access_version=user.access_version,
        canonical_message_id=83_001,
        tier=SUBSCRIBER,
    )
    memory = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=83_002,
        phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
    )

    with pytest.raises(ApplicationHandlerStop):
        await bot.nova_cancel_gate(
            update_for(NovaMessage("/cancel"), user_id=user_id),
            context,
        )

    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        == reminder
    )
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        == guided
    )
    assert memory.id not in repr(reminder)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Привет, где у тебя находится визуализация?",
        (
            "Просто я знаю, что в этом боте есть визуализация, но не могу её найти "
            "в менюшке. Подскажи, пожалуйста."
        ),
        "Привет, где у тебя находятся желания?",
    ],
)
async def test_natural_bot_help_question_routes_to_local_nova_without_content_capture(
    db,
    question,
):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True, nova_ai_admin_only=False)
    user_id = 61_060
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    message = NovaMessage(question)
    context = nova_context()

    assert await bot.nova_text_gate(
        update_for(message, user_id=user_id),
        context,
        user=user,
    )

    assert ai.calls == []
    assert ai.other_calls == []
    assert len(message.replies) == 1
    canonical = message.replies[0]["message"]
    assert context.bot.sent == []
    assert len(context.bot.edits) == 1
    edit = context.bot.edits[0]
    assert edit["message_id"] == canonical.message_id
    assert "визуализац" in edit["text"].casefold()
    assert "1. " in edit["text"]
    callback = callback_with_prefix(edit["reply_markup"], "nova:action:vision:")
    assert any(
        button.text == "🎯 Открыть визуализацию" and button.callback_data == callback
        for row in edit["reply_markup"].inline_keyboard
        for button in row
    )
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.last_action_id == "vision"
    assert question not in repr(current)
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_short_follow_up_reuses_visualization_context_and_canonical_message(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True, nova_ai_admin_only=False)
    user_id = 61_061
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    first = NovaMessage("Не могу найти визуализацию в меню, подскажи")

    assert await bot.nova_text_gate(
        update_for(first, user_id=user_id),
        context,
        user=user,
    )
    canonical = first.replies[0]["message"]
    first_edit_count = len(context.bot.edits)
    follow_up = NovaMessage("Ладно, объясни")

    assert await bot.nova_text_gate(
        update_for(follow_up, user_id=user_id),
        context,
        user=user,
    )

    assert follow_up.replies == []
    assert len(context.bot.edits) == first_edit_count + 1
    edit = context.bot.edits[-1]
    assert edit["message_id"] == canonical.message_id
    assert "визуализац" in edit["text"].casefold()
    assert callback_with_prefix(edit["reply_markup"], "nova:action:vision:")
    assert ai.calls == []
    assert ai.other_calls == []
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.last_action_id == "vision"
    assert all(value not in repr(current) for value in (first.text, follow_up.text))
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Сегодня я размышлял о визуализации будущего",
        "Хочу записать идею для новой карты",
        "Мне важно понять, где я вижу себя через год",
        "Не могу найти время на задачу",
        "Как найти время на задачу?",
        "Где находится задача, которую я обещал сделать?",
        "Где находится здоровье человека?",
        "Как открыть референс в Photoshop?",
        "Как в Photoshop открыть раздел референсов?",
        "Как в меню Photoshop открыть референсы?",
        "Как пользоваться функцией задач в Excel?",
        (
            "Сегодня получилась длинная личная мысль про задачу, здоровье и "
            "визуализацию будущего; хочу спокойно сохранить её и вернуться позже."
        ),
        "Ладно, объясни",
    ],
)
async def test_non_help_content_is_not_captured_by_nova_text_gate(db, content):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True, nova_ai_admin_only=False)
    user_id = 61_062
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    message = NovaMessage(content)

    assert not await bot.nova_text_gate(
        update_for(message, user_id=user_id),
        context,
        user=user,
    )

    assert message.replies == []
    assert context.bot.edits == []
    assert context.bot.sent == []
    assert ai.calls == []
    assert ai.other_calls == []
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_oversized_natural_known_help_is_consumed_as_local_clarify(db):
    question = "Не могу найти визуализацию в меню этого бота. " + "пожалуйста " * 60
    assert len(question) > 600
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True, nova_ai_admin_only=False)
    user_id = 61_077
    user = await user_with_tier(bot, user_id, ADMIN)
    context = nova_context()
    message = NovaMessage(question)

    assert await bot.nova_text_gate(
        update_for(message, user_id=user_id),
        context,
        user=user,
    )

    assert len(message.replies) == 1
    assert len(context.bot.edits) == 1
    assert "600" in context.bot.edits[0]["text"]
    assert ai.calls == []
    assert ai.other_calls == []
    assert context.bot.sent == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcript",
    [
        "Привет, где у тебя находится визуализация?",
        (
            "Просто я знаю, что в этом боте есть визуализация, но не могу её найти "
            "в менюшке. Подскажи, пожалуйста."
        ),
        "Привет, где у тебя находятся желания?",
    ],
)
async def test_voice_help_uses_stt_progress_as_local_nova_canonical_without_ai(
    db,
    transcript,
):
    ai = NovaAIStub()
    transcription = NovaTranscription(transcript)
    bot = make_bot(
        db,
        ai,
        enable_nova_ai=True,
        nova_ai_admin_only=False,
        _transcription=transcription,
    )
    user_id = 61_063
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    message = NovaMessage(voice=NovaVoice())
    context = nova_context()

    await bot.voice(update_for(message, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    assert ai.calls == []
    assert ai.other_calls == []
    assert len(message.replies) == 1
    assert message.replies[0]["text"] == "Расшифровываю голосовую мысль…"
    canonical = message.replies[0]["message"]
    assert context.bot.sent == []
    assert context.bot.edits
    assert all(edit["message_id"] == canonical.message_id for edit in context.bot.edits)
    result = context.bot.edits[-1]
    assert "визуализац" in result["text"].casefold()
    assert "Я услышал" not in result["text"]
    assert callback_with_prefix(result["reply_markup"], "nova:action:vision:")
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.canonical_message_id == canonical.message_id
    assert current.last_action_id == "vision"
    assert transcript not in repr(current)
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_voice_follow_up_reuses_existing_canonical_and_last_action_without_new_message(db):
    ai = NovaAIStub()
    transcription = NovaTranscription("Ладно, объясни")
    bot = make_bot(
        db,
        ai,
        enable_nova_ai=True,
        nova_ai_admin_only=False,
        _transcription=transcription,
    )
    user_id = 61_064
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    first = NovaMessage("Где у тебя находится визуализация?")
    assert await bot.nova_text_gate(
        update_for(first, user_id=user_id),
        context,
        user=user,
    )
    canonical = first.replies[0]["message"]
    edits_before = len(context.bot.edits)
    voice = NovaMessage(voice=NovaVoice())

    await bot.voice(update_for(voice, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    assert len(voice.replies) <= 1
    if voice.replies:
        transient = voice.replies[0]["message"]
        assert (
            transient.deleted == 1
            or (
                user_id,
                transient.message_id,
            )
            in context.bot.deleted
        )
    assert len(context.bot.edits) > edits_before
    follow_up_edits = context.bot.edits[edits_before:]
    assert all(edit["message_id"] == canonical.message_id for edit in follow_up_edits)
    result = follow_up_edits[-1]
    assert "визуализац" in result["text"].casefold()
    assert callback_with_prefix(result["reply_markup"], "nova:action:vision:")
    assert ai.calls == []
    assert ai.other_calls == []
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.canonical_message_id == canonical.message_id
    assert current.last_action_id == "vision"
    assert transcription.transcript not in repr(current)
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_active_memory_voice_reuses_old_canonical_and_deletes_stt_progress(db):
    ai = NovaAIStub()
    transcript = "Я предпочитаю короткие практические ответы"
    transcription = NovaTranscription(transcript)
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
        _transcription=transcription,
    )
    user_id = 61_086
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    command = NovaMessage("Nova, запомни")
    assert await bot.nova_memory_text_gate(
        update_for(command, user_id=user_id),
        context,
        user=user,
    )
    canonical = command.replies[0]["message"]
    awaiting = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert awaiting is not None
    assert awaiting.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
    assert awaiting.canonical_message_id == canonical.message_id
    edits_before = len(context.bot.edits)
    route_message = AsyncMock()
    bot._route_message = route_message
    voice = NovaMessage(voice=NovaVoice())

    with pytest.raises(ApplicationHandlerStop):
        await bot.voice(update_for(voice, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    route_message.assert_not_awaited()
    assert len(voice.replies) == 1
    transient = voice.replies[0]["message"]
    assert (
        transient.deleted == 1
        or (
            user_id,
            transient.message_id,
        )
        in context.bot.deleted
    )
    memory_edits = context.bot.edits[edits_before:]
    assert memory_edits
    assert all(edit["message_id"] == canonical.message_id for edit in memory_edits)
    assert transcript in memory_edits[-1]["text"]
    current = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.phase is NovaMemoryFlowPhase.CREATE_PREVIEW
    assert current.canonical_message_id == canonical.message_id
    assert transcript not in repr(current)
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_active_memory_transcription_error_reuses_canonical_and_retires_progress(db):
    ai = NovaAIStub()
    transcription = FailingNovaTranscription("unused")
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
        _transcription=transcription,
    )
    user_id = 61_101
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    command = NovaMessage("Nova, запомни")
    assert await bot.nova_memory_text_gate(
        update_for(command, user_id=user_id),
        context,
        user=user,
    )
    canonical = command.replies[0]["message"]
    before = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert before is not None
    system_action = AsyncMock(return_value=True)
    route_message = AsyncMock()
    bot._try_system_action = system_action
    bot._route_message = route_message
    edits_before = len(context.bot.edits)
    voice = NovaMessage(voice=NovaVoice())

    await bot.voice(update_for(voice, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    system_action.assert_not_awaited()
    route_message.assert_not_awaited()
    assert len(voice.replies) == 1
    transient = voice.replies[0]["message"]
    assert transient.deleted == 1 or (user_id, transient.message_id) in context.bot.deleted
    error_edits = context.bot.edits[edits_before:]
    assert len(error_edits) == 1
    assert error_edits[0]["message_id"] == canonical.message_id
    assert "Не удалось распознать голосовое" in error_edits[0]["text"]
    callbacks = [
        str(button.callback_data)
        for row in error_edits[0]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert len(callbacks) == 1 and callbacks[0].startswith("nmem:")
    after = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert after is not None
    assert after.id == before.id
    assert after.version > before.version
    assert after.phase is NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_active_memory_voice_disabled_edits_only_existing_canonical(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_102
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    command = NovaMessage("Nova, запомни")
    assert await bot.nova_memory_text_gate(
        update_for(command, user_id=user_id),
        context,
        user=user,
    )
    canonical = command.replies[0]["message"]
    edits_before = len(context.bot.edits)
    voice = NovaMessage(voice=NovaVoice())

    await bot.voice(update_for(voice, user_id=user_id), context)

    assert voice.replies == []
    error_edits = context.bot.edits[edits_before:]
    assert len(error_edits) == 1
    assert error_edits[0]["message_id"] == canonical.message_id
    assert "Распознавание голосовых временно не настроено" in error_edits[0]["text"]
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_memory_created_during_stt_fences_destructive_downstream_route(db):
    ai = NovaAIStub()
    transcription = BlockingNovaTranscription("удали все задачи")
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
        _transcription=transcription,
    )
    user_id = 61_103
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    system_action = AsyncMock(return_value=True)
    route_message = AsyncMock()
    bot._try_system_action = system_action
    bot._route_message = route_message
    voice = NovaMessage(voice=NovaVoice())
    task = asyncio.create_task(bot.voice(update_for(voice, user_id=user_id), context))
    await transcription.started.wait()

    replacement = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=103_001,
        phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
    )
    transcription.release.set()
    with pytest.raises(ApplicationHandlerStop):
        await task

    system_action.assert_not_awaited()
    route_message.assert_not_awaited()
    assert len(voice.replies) == 1
    transient = voice.replies[0]["message"]
    assert transient.deleted == 1 or (user_id, transient.message_id) in context.bot.deleted
    assert await bot.nova_memory_sessions.get_exact(replacement) == replacement
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_memory_published_during_post_stt_access_check_fences_system_route(db):
    ai = NovaAIStub()
    transcription = NovaTranscription("удали все задачи")
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
        _transcription=transcription,
    )
    user_id = 61_104
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    original_access = bot._nova_memory_access
    replacement = None

    async def publish_during_access(update):
        nonlocal replacement
        if replacement is None:
            replacement = await seed_memory_flow(
                bot,
                user,
                chat_id=user_id,
                canonical_message_id=104_001,
                phase=NovaMemoryFlowPhase.ROOT,
            )
        return await original_access(update)

    bot._nova_memory_access = publish_during_access
    system_action = AsyncMock(return_value=True)
    route_message = AsyncMock()
    bot._try_system_action = system_action
    bot._route_message = route_message
    voice = NovaMessage(voice=NovaVoice())

    with pytest.raises(ApplicationHandlerStop):
        await bot.voice(update_for(voice, user_id=user_id), context)

    system_action.assert_not_awaited()
    route_message.assert_not_awaited()
    assert replacement is not None
    assert await bot.nova_memory_sessions.get_exact(replacement) == replacement
    transient = voice.replies[0]["message"]
    assert transient.deleted == 1 or (user_id, transient.message_id) in context.bot.deleted
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_second_voice_access_version_bounce_cannot_create_first_memory_session(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_105
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    message = NovaMessage(voice=NovaVoice())
    update = update_for(message, user_id=user_id)
    fence = await bot.nova_memory_voice_fence(update, user=user)
    assert fence is not None and fence.session is None
    original_access = bot._nova_memory_access
    calls = 0

    async def bounce_on_second_access(late_update):
        nonlocal calls
        calls += 1
        if calls == 2:
            service = AccessService(db)
            await service.set_guest(user_id, source="memory-second-voice-access")
            await service.grant_subscriber(user_id, source="memory-second-voice-access")
        return await original_access(late_update)

    bot._nova_memory_access = bounce_on_second_access
    progress = NovaMessage("Расшифровываю…")

    assert await bot.nova_memory_voice_gate(
        update,
        context,
        "Nova, запомни: PRIVATE_SECOND_ACCESS_CONTENT",
        progress,
        user=user,
        fence=fence,
    )

    assert calls == 3
    assert progress.edits[-1]["text"] == NOVA_MEMORY_VOICE_ACCESS_CHANGED_TEXT
    assert await bot.nova_memory_sessions.count() == 0
    assert await content_row_counts(db) == (0, 0)
    assert ai.calls == []
    assert ai.other_calls == []


@pytest.mark.asyncio
async def test_second_voice_access_downgrade_neutralizes_active_memory_canonical(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_106
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    session = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=106_001,
        phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
    )
    message = NovaMessage(voice=NovaVoice())
    update = update_for(message, user_id=user_id)
    fence = await bot.nova_memory_voice_fence(update, user=user)
    assert fence is not None and fence.session == session
    original_access = bot._nova_memory_access
    calls = 0

    async def downgrade_on_second_access(late_update):
        nonlocal calls
        calls += 1
        if calls == 2:
            await AccessService(db).set_guest(
                user_id,
                source="memory-second-voice-access",
            )
        return await original_access(late_update)

    bot._nova_memory_access = downgrade_on_second_access
    progress = NovaMessage("Расшифровываю…")

    assert await bot.nova_memory_voice_gate(
        update,
        context,
        "Nova, запомни: PRIVATE_REPLACEMENT_CONTENT",
        progress,
        user=user,
        fence=fence,
    )

    assert calls == 3
    assert progress.deleted == 1
    assert await bot.nova_memory_sessions.count() == 0
    assert context.bot.edits[-1]["message_id"] == session.canonical_message_id
    assert context.bot.edits[-1]["text"] == NOVA_MEMORY_ACCESS_CHANGED_TEXT
    assert "PRIVATE_REPLACEMENT_CONTENT" not in str(context.bot.edits)
    assert await content_row_counts(db) == (0, 0)
    assert ai.calls == []
    assert ai.other_calls == []


@pytest.mark.asyncio
async def test_voice_access_bounce_during_session_read_is_caught_by_final_fence(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_107
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    message = NovaMessage(voice=NovaVoice())
    update = update_for(message, user_id=user_id)
    fence = await bot.nova_memory_voice_fence(update, user=user)
    assert fence is not None and fence.session is None
    original_current = bot.nova_memory_sessions.current
    current_calls = 0

    async def bounce_during_first_session_read(**kwargs):
        nonlocal current_calls
        current_calls += 1
        if current_calls == 1:
            service = AccessService(db)
            await service.set_guest(user_id, source="memory-voice-session-read")
            await service.grant_subscriber(user_id, source="memory-voice-session-read")
        return await original_current(**kwargs)

    bot.nova_memory_sessions.current = bounce_during_first_session_read
    progress = NovaMessage("Расшифровываю…")

    assert await bot.nova_memory_voice_pre_route(
        update,
        context,
        progress,
        fence=fence,
    )

    assert current_calls == 2
    assert progress.edits[-1]["text"] == NOVA_MEMORY_VOICE_ACCESS_CHANGED_TEXT
    assert await bot.nova_memory_sessions.count() == 0
    assert await content_row_counts(db) == (0, 0)
    assert ai.calls == []
    assert ai.other_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["cleared", "replaced", "access_bounce"])
async def test_stale_memory_voice_is_consumed_without_generic_fallback(db, lifecycle):
    ai = NovaAIStub()
    transcript = "Новый способ общения"
    transcription = BlockingNovaTranscription(transcript)
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
        _transcription=transcription,
    )
    user_id = {
        "cleared": 61_087,
        "replaced": 61_088,
        "access_bounce": 61_089,
    }[lifecycle]
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    command = NovaMessage("Nova, запомни")
    assert await bot.nova_memory_text_gate(
        update_for(command, user_id=user_id),
        context,
        user=user,
    )
    original = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert original is not None
    system_action = AsyncMock(return_value=True)
    bot._try_system_action = system_action
    route_message = AsyncMock()
    bot._route_message = route_message
    voice = NovaMessage(voice=NovaVoice())
    task = asyncio.create_task(bot.voice(update_for(voice, user_id=user_id), context))
    await transcription.started.wait()

    replacement = None
    if lifecycle == "cleared":
        assert await bot.nova_memory_sessions.clear(
            owner_id=original.owner_id,
            telegram_user_id=original.telegram_user_id,
            chat_id=original.chat_id,
            session_id=original.id,
        )
    elif lifecycle == "replaced":
        replacement = await seed_memory_flow(
            bot,
            user,
            chat_id=user_id,
            canonical_message_id=89_001,
            phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
        )
    else:
        service = AccessService(db)
        await service.set_guest(user_id, source="memory-voice-race")
        await service.grant_subscriber(user_id, source="memory-voice-race")

    transcription.release.set()
    with pytest.raises(ApplicationHandlerStop):
        await task

    system_action.assert_not_awaited()
    route_message.assert_not_awaited()
    assert len(voice.replies) == 1
    transient = voice.replies[0]["message"]
    assert (
        transient.deleted == 1
        or (
            user_id,
            transient.message_id,
        )
        in context.bot.deleted
    )
    live = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    if replacement is not None:
        assert live == replacement
    else:
        assert live is None
    assert original.id not in repr(live)
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_memory_negative_voice_falls_through_to_generic_pipeline(db):
    ai = NovaAIStub()
    transcript = "Сегодня я спокойно работал над обычными делами"
    transcription = NovaTranscription(transcript)
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
        _transcription=transcription,
    )
    user_id = 61_090
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    route_message = AsyncMock()
    bot._route_message = route_message
    message = NovaMessage(voice=NovaVoice())

    await bot.voice(update_for(message, user_id=user_id), nova_context())

    route_message.assert_awaited_once()
    assert route_message.await_args.args[2:] == (transcript, "voice")
    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert ai.calls == []
    assert ai.other_calls == []
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcript",
    [
        "Сегодня я размышлял о визуализации будущего",
        "Не могу найти время на задачу",
        "Как открыть референс в Photoshop?",
        "Как открыть раздел здоровья в презентации?",
    ],
)
async def test_non_help_voice_transcript_continues_existing_content_pipeline(db, transcript):
    ai = NovaAIStub()
    transcription = NovaTranscription(transcript)
    bot = make_bot(db, ai, _transcription=transcription)
    user_id = 61_065
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    route_message = AsyncMock()
    bot._route_message = route_message
    message = NovaMessage(voice=NovaVoice())
    context = nova_context()

    await bot.voice(update_for(message, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    route_message.assert_awaited_once()
    assert route_message.await_args.args[2:] == (transcript, "voice")
    assert message.replies[0]["message"].edits[-1]["text"].startswith("Я услышал")
    assert ai.calls == []
    assert ai.other_calls == []
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["downgrade", "version_bounce"])
async def test_voice_help_access_change_during_stt_discards_result_and_context(
    db,
    caplog,
    change,
):
    ai = NovaAIStub()
    user_id = 61_066 if change == "downgrade" else 61_067

    async def mutate_access() -> None:
        service = AccessService(db)
        await service.set_guest(user_id, source="nova-voice-race")
        if change == "version_bounce":
            await service.grant_subscriber(user_id, source="nova-voice-race")

    transcript = "Привет, где у тебя находится визуализация? PRIVATE_VOICE_TRANSCRIPT"
    transcription = HookedNovaTranscription(transcript, mutate_access)
    bot = make_bot(
        db,
        ai,
        enable_nova_ai=True,
        nova_ai_admin_only=False,
        _transcription=transcription,
    )
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    message = NovaMessage(voice=NovaVoice())
    context = nova_context()

    with caplog.at_level(logging.WARNING, logger="future_self.nova_handlers"):
        await bot.voice(update_for(message, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    assert ai.calls == []
    assert ai.other_calls == []
    assert len(message.replies) == 1
    rendered = [edit["text"] for edit in context.bot.edits]
    rendered.extend(edit["text"] for edit in message.replies[0]["message"].edits)
    assert NOVA_ACCESS_CHANGED_TEXT in rendered
    assert all("Открыть визуализацию" not in value for value in rendered)
    assert context.bot.sent == []
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )
    assert transcript not in caplog.text


@pytest.mark.asyncio
async def test_ordinary_voice_access_bounce_during_stt_never_reaches_content_pipeline(db):
    ai = NovaAIStub()
    user_id = 61_077

    async def bounce_access() -> None:
        service = AccessService(db)
        await service.set_guest(user_id, source="ordinary-voice-race")
        await service.grant_subscriber(user_id, source="ordinary-voice-race")

    transcript = "PRIVATE_ORDINARY_VOICE_TRANSCRIPT о моём обычном дне"
    transcription = HookedNovaTranscription(transcript, bounce_access)
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
        _transcription=transcription,
    )
    await user_with_tier(bot, user_id, SUBSCRIBER)
    system_action = AsyncMock(return_value=True)
    bot._try_system_action = system_action
    route_message = AsyncMock()
    bot._route_message = route_message
    message = NovaMessage(voice=NovaVoice())
    context = nova_context()

    with pytest.raises(ApplicationHandlerStop):
        await bot.voice(update_for(message, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    system_action.assert_not_awaited()
    route_message.assert_not_awaited()
    assert ai.calls == []
    assert ai.other_calls == []
    assert len(message.replies) == 1
    progress = message.replies[0]["message"]
    assert len(progress.edits) == 1
    assert progress.edits[0]["text"] == NOVA_MEMORY_VOICE_ACCESS_CHANGED_TEXT
    assert progress.edits[0]["reply_markup"] is None
    assert context.bot.sent == []
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


@pytest.mark.asyncio
async def test_voice_nova_edit_error_has_no_fallback_or_sensitive_log(db, caplog):
    transcript = "Привет, где у тебя находится визуализация? PRIVATE_VOICE_QUESTION"
    error = BadRequest("PRIVATE_VOICE_TELEGRAM_ERROR_BODY")
    ai = NovaAIStub()
    transcription = NovaTranscription(transcript)
    bot = make_bot(db, ai, _transcription=transcription)
    user_id = 61_068
    await user_with_tier(bot, user_id, SUBSCRIBER)
    message = NovaMessage(voice=NovaVoice(), edit_error=error)
    context = nova_context()
    context.bot.edit_error = error

    with caplog.at_level(logging.WARNING):
        await bot.voice(update_for(message, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    assert len(message.replies) == 1
    assert context.bot.sent == []
    assert message.replies[0]["message"].replies == []
    assert ai.calls == []
    assert ai.other_calls == []
    assert transcript not in caplog.text
    assert "PRIVATE_VOICE_TELEGRAM_ERROR_BODY" not in caplog.text
    assert "BadRequest" in caplog.text


@pytest.mark.asyncio
async def test_voice_nova_message_not_modified_keeps_bound_canonical_without_replacement(db):
    ai = NovaAIStub()
    transcription = NovaTranscription("Привет, где у тебя находится визуализация?")
    bot = make_bot(db, ai, _transcription=transcription)
    user_id = 61_069
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    error = BadRequest("Message is not modified")
    message = NovaMessage(voice=NovaVoice(), edit_error=error)
    context = nova_context()
    context.bot.edit_error = error

    await bot.voice(update_for(message, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    assert len(message.replies) == 1
    canonical = message.replies[0]["message"]
    assert context.bot.sent == []
    assert ai.calls == []
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.canonical_message_id == canonical.message_id
    assert current.last_action_id == "vision"


@pytest.mark.asyncio
async def test_cancelled_error_from_voice_nova_delivery_propagates_without_raw_log(db, caplog):
    transcript = "Привет, где у тебя находится визуализация? RAW_CANCELLED_VOICE_SENTINEL"
    ai = NovaAIStub()
    transcription = NovaTranscription(transcript)
    bot = make_bot(db, ai, _transcription=transcription)
    user_id = 61_070
    await user_with_tier(bot, user_id, SUBSCRIBER)
    error = asyncio.CancelledError()
    message = NovaMessage(voice=NovaVoice(), edit_error=error)
    context = nova_context()
    context.bot.edit_error = error

    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.CancelledError):
            await bot.voice(update_for(message, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    assert ai.calls == []
    assert ai.other_calls == []
    assert context.bot.sent == []
    assert transcript not in caplog.text


@pytest.mark.asyncio
async def test_concurrent_duplicate_voice_help_keeps_one_live_canonical(db):
    ai = NovaAIStub()
    transcription = BarrierNovaTranscription("Привет, где у тебя находится визуализация?")
    bot = make_bot(db, ai, _transcription=transcription)
    user_id = 61_071
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    first = NovaMessage(voice=NovaVoice())
    second = NovaMessage(voice=NovaVoice())

    await asyncio.gather(
        bot.voice(update_for(first, user_id=user_id), context),
        bot.voice(update_for(second, user_id=user_id), context),
    )

    assert len(transcription.calls) == 2
    assert ai.calls == []
    assert ai.other_calls == []
    assert context.bot.sent == []
    progresses = [reply["message"] for source in (first, second) for reply in source.replies]
    deleted_ids = {message_id for _chat_id, message_id in context.bot.deleted}
    live_progresses = [
        progress
        for progress in progresses
        if progress.deleted == 0 and progress.message_id not in deleted_ids
    ]
    assert len(live_progresses) == 1
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.canonical_message_id == live_progresses[0].message_id
    assert current.last_action_id == "vision"
    assert all(edit["message_id"] == current.canonical_message_id for edit in context.bot.edits)
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_concurrent_duplicate_unknown_voice_help_calls_only_nova_provider_once(db):
    transcript = "Nova, где находится неизвестный квантовый раздел?"
    ai = NovaAIStub()
    transcription = BarrierNovaTranscription(transcript)
    bot = make_bot(db, ai, enable_nova_ai=True, _transcription=transcription)
    user_id = 61_076
    user = await user_with_tier(bot, user_id, ADMIN)
    context = nova_context()
    first = NovaMessage(voice=NovaVoice())
    second = NovaMessage(voice=NovaVoice())

    await asyncio.gather(
        bot.voice(update_for(first, user_id=user_id), context),
        bot.voice(update_for(second, user_id=user_id), context),
    )

    assert len(transcription.calls) == 2
    assert len(ai.calls) == 1
    assert ai.calls[0][0] == "где находится неизвестный квантовый раздел?"
    assert ai.other_calls == []
    assert context.bot.sent == []
    progresses = [reply["message"] for source in (first, second) for reply in source.replies]
    deleted_ids = {message_id for _chat_id, message_id in context.bot.deleted}
    live_progresses = [
        progress
        for progress in progresses
        if progress.deleted == 0 and progress.message_id not in deleted_ids
    ]
    assert len(live_progresses) == 1
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.canonical_message_id == live_progresses[0].message_id
    assert current.last_action_id is None
    assert all(edit["message_id"] == current.canonical_message_id for edit in context.bot.edits)
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["cancelled", "expired", "replaced"])
async def test_stale_voice_follow_up_cannot_resurrect_or_replace_nova_session(
    db,
    lifecycle,
):
    ai = NovaAIStub()
    transcription = BlockingNovaTranscription("Ладно, объясни")
    bot = make_bot(db, ai, _transcription=transcription)
    clock = [0.0]
    if lifecycle == "expired":
        bot.nova_sessions = NovaSessionStore(ttl_seconds=1, clock=lambda: clock[0])
    user_id = {
        "cancelled": 61_073,
        "expired": 61_074,
        "replaced": 61_075,
    }[lifecycle]
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    first = NovaMessage("Привет, где у тебя находится визуализация?")
    assert await bot.nova_text_gate(
        update_for(first, user_id=user_id),
        context,
        user=user,
    )
    original = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert original is not None
    assert original.last_action_id == "vision"
    edits_before = len(context.bot.edits)
    route_message = AsyncMock()
    bot._route_message = route_message
    voice = NovaMessage(voice=NovaVoice())
    task = asyncio.create_task(bot.voice(update_for(voice, user_id=user_id), context))
    await transcription.started.wait()

    replacement = None
    if lifecycle == "expired":
        clock[0] = 2.0
        assert (
            await bot.nova_sessions.current(
                owner_id=user.id,
                telegram_user_id=user_id,
                chat_id=user_id,
            )
            is None
        )
    elif lifecycle == "replaced":
        replacement = await bot.nova_sessions.create(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
            access_version=user.access_version,
            canonical_message_id=99_000 + user_id,
            tier=SUBSCRIBER,
        )
    else:
        assert await bot.nova_sessions.clear(
            owner_id=user.id,
            chat_id=user_id,
            session_id=original.id,
        )

    transcription.release.set()
    await task

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    route_message.assert_not_awaited()
    assert ai.calls == []
    assert ai.other_calls == []
    assert len(voice.replies) == 1
    transient = voice.replies[0]["message"]
    assert (
        transient.deleted == 1
        or (
            user_id,
            transient.message_id,
        )
        in context.bot.deleted
    )
    assert len(context.bot.edits) == edits_before
    assert context.bot.sent == []
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    if lifecycle == "replaced":
        assert replacement is not None
        assert current == replacement
        assert current.last_action_id is None
    else:
        assert current is None
    assert original.id not in repr(current)
    assert await content_row_counts(db) == (0, 0)


@pytest.mark.asyncio
async def test_guest_is_local_only_and_blocked_is_fail_closed(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True, nova_ai_admin_only=False)

    guest_id = 61_003
    guest = await user_with_tier(bot, guest_id, GUEST)
    _command, _canonical, guest_context = await open_nova(bot, user_id=guest_id)
    question = NovaMessage("Как создать задачу с напоминанием?")
    assert await bot.nova_text_gate(
        update_for(question, user_id=guest_id),
        guest_context,
        user=guest,
    )
    assert ai.calls == []
    guest_edit = guest_context.bot.edits[-1]
    assert callback_with_prefix(guest_edit["reply_markup"], "guest:access") == "guest:access"
    assert not any(
        (button.callback_data or "").startswith("nova:action:")
        for row in guest_edit["reply_markup"].inline_keyboard
        for button in row
    )

    blocked_id = 61_004
    await user_with_tier(bot, blocked_id, BLOCKED)
    blocked_message = NovaMessage("/help")
    await bot.help_command(update_for(blocked_message, user_id=blocked_id), nova_context())
    assert len(blocked_message.replies) == 1
    assert "Nova" not in blocked_message.replies[0]["text"]
    blocked_user = await bot._user(blocked_id)
    assert (
        await bot.nova_sessions.current(
            owner_id=blocked_user.id,
            telegram_user_id=blocked_id,
            chat_id=blocked_id,
        )
        is None
    )
    assert ai.calls == []


@pytest.mark.asyncio
async def test_subscriber_unknown_question_stays_local_during_admin_only_pilot(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True, nova_ai_admin_only=True)
    user_id = 61_005
    await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)

    question = NovaMessage("квантовый гиперкуб без известной функции")
    assert await bot.nova_text_gate(update_for(question, user_id=user_id), context)

    assert ai.calls == []
    assert context.bot.edits == [
        {
            "chat_id": user_id,
            "message_id": canonical.message_id,
            "text": f"✨ Nova\n\n{NOVA_LOCAL_CLARIFY_TEXT}",
            "reply_markup": context.bot.edits[0]["reply_markup"],
        }
    ]


@pytest.mark.asyncio
async def test_admin_unknown_explicit_question_uses_only_one_nova_help_call(db):
    ai = NovaAIStub()
    ai.result = NovaHelpPlan(
        response="PROVIDER_OUTPUT_SENTINEL",
        steps=["Уточни нужный раздел"],
        action_id=None,
        kind="clarify",
    )
    bot = make_bot(db, ai, enable_nova_ai=True, nova_ai_admin_only=True)
    user_id = 61_006
    await user_with_tier(bot, user_id, ADMIN)
    message = NovaMessage("Nova, где спрятан квантовый гиперкуб?")
    context = nova_context()

    assert await bot.nova_text_gate(update_for(message, user_id=user_id), context)

    assert len(message.replies) == 1
    assert message.replies[0]["text"] == "✨ Nova\n\nРазбираю вопрос…"
    assert len(ai.calls) == 1
    assert ai.calls[0][0] == "где спрятан квантовый гиперкуб?"
    assert ai.other_calls == []
    assert len(context.bot.edits) == 1
    assert "PROVIDER_OUTPUT_SENTINEL" in context.bot.edits[0]["text"]
    assert context.bot.sent == []


@pytest.mark.asyncio
async def test_duplicate_concurrent_explicit_questions_create_one_canonical_and_provider_call(db):
    ai = NovaAIStub()
    ai.release.clear()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = 61_007
    user = await user_with_tier(bot, user_id, ADMIN)
    context = nova_context()
    first = NovaMessage("Nova, неизвестный вопрос номер один")
    second = NovaMessage("Nova, неизвестный вопрос номер два")
    original_begin_question = bot.nova_sessions.begin_question
    second_begin_attempted = asyncio.Event()
    begin_calls = 0

    async def observed_begin_question(**kwargs: Any):
        nonlocal begin_calls
        begin_calls += 1
        result = await original_begin_question(**kwargs)
        if begin_calls == 2:
            second_begin_attempted.set()
        return result

    bot.nova_sessions.begin_question = observed_begin_question

    first_task = asyncio.create_task(
        bot.nova_text_gate(update_for(first, user_id=user_id), context, user=user)
    )
    await ai.started.wait()
    second_task = asyncio.create_task(
        bot.nova_text_gate(update_for(second, user_id=user_id), context, user=user)
    )
    await second_begin_attempted.wait()

    assert len(ai.calls) == 1
    assert len(first.replies) + len(second.replies) == 1
    assert await bot.nova_sessions.count() == 1

    ai.release.set()
    assert await asyncio.gather(first_task, second_task) == [True, True]
    assert len(ai.calls) == 1
    assert len(context.bot.edits) == 1
    assert context.bot.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_data", ["nova:root", "nova:topic:privacy", "nav:help"])
async def test_blocked_canonical_edit_cannot_reset_a_concurrent_provider_session(
    db,
    callback_data,
):
    ai = NovaAIStub()
    ai.release.clear()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = {
        "nova:root": 61_008,
        "nova:topic:privacy": 61_009,
        "nav:help": 61_013,
    }[callback_data]
    user = await user_with_tier(bot, user_id, ADMIN)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    original_begin_question = bot.nova_sessions.begin_question
    second_begin_attempted = asyncio.Event()
    begin_calls = 0

    async def observed_begin_question(**kwargs: Any):
        nonlocal begin_calls
        begin_calls += 1
        result = await original_begin_question(**kwargs)
        if begin_calls == 2:
            second_begin_attempted.set()
        return result

    bot.nova_sessions.begin_question = observed_begin_question
    edit_started = asyncio.Event()
    edit_release = asyncio.Event()
    query = NovaQuery(
        callback_data,
        canonical,
        edit_started=edit_started,
        edit_release=edit_release,
    )
    callback_update = update_for(canonical, user_id=user_id, query=query)
    callback_task = asyncio.create_task(
        bot.nova_navigation_help_callback(callback_update, context)
        if callback_data == "nav:help"
        else bot.nova_callback(callback_update, context)
    )
    await edit_started.wait()

    first_question = asyncio.create_task(
        bot.nova_text_gate(
            update_for(NovaMessage("неизвестный параллельный вопрос"), user_id=user_id),
            context,
            user=user,
        )
    )
    await asyncio.sleep(0)
    assert ai.calls == []

    edit_release.set()
    await callback_task
    await ai.started.wait()
    second_question = asyncio.create_task(
        bot.nova_text_gate(
            update_for(NovaMessage("ещё один параллельный вопрос"), user_id=user_id),
            context,
            user=user,
        )
    )
    await second_begin_attempted.wait()
    assert len(ai.calls) == 1

    ai.release.set()
    assert await asyncio.gather(first_question, second_question) == [True, True]
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert current.question_in_progress is False
    assert query.answers == [(None, False)]
    assert len(ai.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_provider", "after_provider", "delivery"])
async def test_access_races_discard_provider_result_and_action(db, phase):
    ai = NovaAIStub()
    ai.result = NovaHelpPlan(
        response="ACCESS_RACE_PROVIDER_RESULT",
        steps=["Недоступный после downgrade шаг"],
        action_id="task_create",
        kind="guide",
    )
    if phase == "after_provider":
        ai.release.clear()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = {"before_provider": 61_010, "after_provider": 61_011, "delivery": 61_012}[phase]
    user = await user_with_tier(bot, user_id, ADMIN)
    _command, _canonical, context = await open_nova(bot, user_id=user_id)
    original_status = bot.access_service.status
    calls = 0

    async def status_with_delivery_fence(telegram_id: int):
        nonlocal calls
        calls += 1
        if phase == "before_provider" and calls == 1:
            await AccessService(db).set_guest(telegram_id, source="nova-race")
        if phase == "delivery" and calls == 5:
            await AccessService(db).set_guest(telegram_id, source="nova-race")
        return await original_status(telegram_id)

    bot.access_service.status = status_with_delivery_fence
    question = NovaMessage("неизвестный вопрос для access race")
    task = asyncio.create_task(
        bot.nova_text_gate(update_for(question, user_id=user_id), context, user=user)
    )
    if phase == "after_provider":
        await ai.started.wait()
        await AccessService(db).set_guest(user_id, source="nova-race")
        ai.release.set()
    assert await task is True

    assert len(ai.calls) == (0 if phase == "before_provider" else 1)
    assert all("ACCESS_RACE_PROVIDER_RESULT" not in edit["text"] for edit in context.bot.edits)
    assert context.bot.edits[-1]["text"] == NOVA_ACCESS_CHANGED_TEXT
    assert context.bot.edits[-1]["reply_markup"] is None
    assert context.bot.sent == []
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_action_token_rejects_forged_id_then_allows_once_and_rejects_replay(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_020
    await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    question = NovaMessage("Как добавить задачу с напоминанием?")
    await bot.nova_text_gate(update_for(question, user_id=user_id), context)
    callback = callback_with_prefix(
        context.bot.edits[-1]["reply_markup"],
        "nova:action:task_create:",
    )
    forged = callback.replace("nova:action:task_create:", "nova:action:vision:")
    dispatch = AsyncMock()
    bot._nova_dispatch_capability = dispatch

    forged_query = NovaQuery(forged, canonical)
    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=forged_query),
        context,
    )
    assert forged_query.answers == [(NOVA_STALE_ALERT, True)]
    dispatch.assert_not_awaited()

    valid_query = NovaQuery(callback, canonical)
    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=valid_query),
        context,
    )
    assert valid_query.answers == [(None, False)]
    dispatch.assert_awaited_once()

    replay_query = NovaQuery(callback, canonical)
    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=replay_query),
        context,
    )
    assert replay_query.answers == [(NOVA_STALE_ALERT, True)]
    assert dispatch.await_count == 1


@pytest.mark.asyncio
async def test_weekly_nova_action_preserves_active_reminder_owner(db):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_weekly_review=True,
        weekly_review_admin_only=False,
    )
    user_id = 61_120
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    reminder_message = NovaMessage("Каждый день напоминай в 20:30 заполнить дневник")
    assert await bot.reminder_text_gate(
        update_for(reminder_message, user_id=user_id),
        context,
    )
    reminder = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert reminder is not None

    canonical = NovaMessage(message_id=88_120)
    guided = await bot.nova_sessions.create(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        tier=SUBSCRIBER,
    )
    token = await bot.nova_sessions.issue_action(
        action_id="weekly_review",
        owner_id=guided.owner_id,
        telegram_user_id=guided.telegram_user_id,
        chat_id=guided.chat_id,
        access_version=guided.access_version,
        canonical_message_id=guided.canonical_message_id,
        tier=guided.tier,
        session_id=guided.id,
    )
    assert token is not None
    query = NovaQuery(f"nova:action:weekly_review:{token}", canonical)

    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=query),
        context,
    )

    assert query.answers == [(None, False)]
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        == reminder
    )
    assert (
        await bot.weekly_review_service.current_session(
            telegram_actor_id=user_id,
            chat_id=user_id,
            expected_access_version=user.access_version,
        )
    ).session is None
    assert ai.calls == []


@pytest.mark.asyncio
async def test_weekly_nova_action_rechecks_false_snapshot_after_late_reminder_acquisition(
    db,
    monkeypatch,
):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_weekly_review=True,
        weekly_review_admin_only=False,
    )
    user_id = 61_121
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    canonical = NovaMessage(message_id=88_121)
    guided = await bot.nova_sessions.create(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
        access_version=user.access_version,
        canonical_message_id=canonical.message_id,
        tier=SUBSCRIBER,
    )
    token = await bot.nova_sessions.issue_action(
        action_id="weekly_review",
        owner_id=guided.owner_id,
        telegram_user_id=guided.telegram_user_id,
        chat_id=guided.chat_id,
        access_version=guided.access_version,
        canonical_message_id=guided.canonical_message_id,
        tier=guided.tier,
        session_id=guided.id,
    )
    assert token is not None
    query = NovaQuery(f"nova:action:weekly_review:{token}", canonical)
    callback_update = update_for(canonical, user_id=user_id, query=query)
    first_snapshot_read = asyncio.Event()
    first_snapshot_release = asyncio.Event()
    original_blocks = bot.reminder_blocks_navigation
    checks = 0

    async def false_then_live(update: Any) -> bool:
        nonlocal checks
        checks += 1
        if checks == 1:
            assert await original_blocks(update) is False
            first_snapshot_read.set()
            await first_snapshot_release.wait()
            return False
        return await original_blocks(update)

    monkeypatch.setattr(bot, "reminder_blocks_navigation", false_then_live)
    processing = asyncio.create_task(bot.nova_callback(callback_update, context))
    try:
        await asyncio.wait_for(first_snapshot_read.wait(), timeout=10)
        reminder_message = NovaMessage("Каждый день напоминай в 20:30 заполнить дневник")
        assert await bot.reminder_text_gate(
            update_for(reminder_message, user_id=user_id),
            context,
        )
        late_reminder = await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        assert late_reminder is not None
        first_snapshot_release.set()
        await asyncio.wait_for(processing, timeout=10)
    finally:
        first_snapshot_release.set()
        if not processing.done():
            await asyncio.gather(processing, return_exceptions=True)

    assert checks == 2
    assert query.answers == [(None, False)]
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        == late_reminder
    )
    assert late_reminder.canonical_message_id == reminder_message.replies[0]["message"].message_id
    assert (
        await bot.weekly_review_service.current_session(
            telegram_actor_id=user_id,
            chat_id=user_id,
            expected_access_version=user.access_version,
        )
    ).session is None
    assert ai.calls == []


@pytest.mark.asyncio
async def test_voice_visualization_cta_dispatches_real_menu_once_and_rejects_replay(db):
    ai = NovaAIStub()
    transcription = NovaTranscription("Привет, где у тебя находится визуализация?")
    bot = make_bot(db, ai, _transcription=transcription)
    user_id = 61_072
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    context = nova_context()
    question = NovaMessage(voice=NovaVoice())

    await bot.voice(update_for(question, user_id=user_id), context)

    assert transcription.calls == [(b"nova-voice", "voice.ogg")]
    canonical = question.replies[0]["message"]
    callback = callback_with_prefix(
        context.bot.edits[-1]["reply_markup"],
        "nova:action:vision:",
    )

    query = NovaQuery(callback, canonical)
    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=query),
        context,
    )

    assert query.answers == [(None, False)]
    assert len(query.edits) == 1
    assert query.edits[0]["text"].startswith("🎯 Желания и визуализация")
    assert button_matrix(query.edits[0]["reply_markup"])[0] == [
        ("➕ Добавить желание", "vision:add")
    ]
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )

    replay = NovaQuery(callback, canonical)
    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=replay),
        context,
    )

    assert replay.answers == [(NOVA_STALE_ALERT, True)]
    assert replay.edits == []
    assert ai.calls == []
    assert ai.other_calls == []


@pytest.mark.asyncio
async def test_action_is_rejected_if_runtime_feature_was_disabled_after_render(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_workspace_access=True)
    user_id = 61_021
    await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    await bot.nova_text_gate(
        update_for(NovaMessage("Где пространства?"), user_id=user_id),
        context,
    )
    callback = callback_with_prefix(
        context.bot.edits[-1]["reply_markup"],
        "nova:action:spaces:",
    )
    bot.settings.enable_workspace_access = False
    dispatch = AsyncMock()
    bot._nova_dispatch_capability = dispatch
    query = NovaQuery(callback, canonical)

    await bot.nova_callback(update_for(canonical, user_id=user_id, query=query), context)

    assert query.answers == [(NOVA_STALE_ALERT, True)]
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["access_version", "feature_flag"])
async def test_action_is_revalidated_after_callback_answer_before_dispatch(db, race):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_workspace_access=True)
    user_id = 61_022 if race == "access_version" else 61_023
    await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    question = (
        "Как добавить задачу с напоминанием?" if race == "access_version" else "Где пространства?"
    )
    prefix = "nova:action:task_create:" if race == "access_version" else "nova:action:spaces:"
    await bot.nova_text_gate(update_for(NovaMessage(question), user_id=user_id), context)
    callback = callback_with_prefix(context.bot.edits[-1]["reply_markup"], prefix)

    async def change_during_answer() -> None:
        if race == "access_version":
            await AccessService(db).set_guest(user_id, source="nova-answer-race")
        else:
            bot.settings.enable_workspace_access = False

    dispatch = AsyncMock()
    bot._nova_dispatch_capability = dispatch
    query = NovaQuery(callback, canonical, answer_hook=change_during_answer)
    await bot.nova_callback(update_for(canonical, user_id=user_id, query=query), context)

    assert query.answers == [(None, False)]
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_conversation_action_is_revalidated_after_callback_answer(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_024
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    token = await bot.nova_sessions.issue_action(
        action_id="evening",
        owner_id=current.owner_id,
        telegram_user_id=current.telegram_user_id,
        chat_id=current.chat_id,
        access_version=current.access_version,
        canonical_message_id=current.canonical_message_id,
        tier=current.tier,
        session_id=current.id,
    )
    assert token is not None
    callback = f"nova:action:evening:{token}"

    async def change_during_answer() -> None:
        await AccessService(db).set_guest(user_id, source="nova-conversation-answer-race")

    start = AsyncMock(return_value=123)
    bot.evening_start = start
    query = NovaQuery(callback, canonical, answer_hook=change_during_answer)
    result = await bot.nova_evening_entry(
        update_for(canonical, user_id=user_id, query=query),
        context,
    )

    assert result is None
    assert query.answers == [(None, False)]
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_conversation_action_launch_blocks_concurrent_nova_question(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = 61_029
    user = await user_with_tier(bot, user_id, ADMIN)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    token = await bot.nova_sessions.issue_action(
        action_id="evening",
        owner_id=current.owner_id,
        telegram_user_id=current.telegram_user_id,
        chat_id=current.chat_id,
        access_version=current.access_version,
        canonical_message_id=current.canonical_message_id,
        tier=current.tier,
        session_id=current.id,
    )
    assert token is not None
    answer_started = asyncio.Event()
    answer_release = asyncio.Event()

    async def block_answer() -> None:
        answer_started.set()
        await answer_release.wait()

    query = NovaQuery(
        f"nova:action:evening:{token}",
        canonical,
        answer_hook=block_answer,
    )
    action_task = asyncio.create_task(
        bot.nova_evening_entry(
            update_for(canonical, user_id=user_id, query=query),
            context,
        )
    )
    await answer_started.wait()
    question = NovaMessage("Nova, совершенно неизвестный параллельный вопрос")
    question_task = asyncio.create_task(
        bot.nova_text_gate(
            update_for(question, user_id=user_id),
            context,
            user=user,
        )
    )
    await asyncio.sleep(0)
    assert not question_task.done()
    assert ai.calls == []
    assert question.replies == []

    answer_release.set()
    assert await action_task is not None
    assert await question_task is False
    assert query.answers == [(None, False)]
    assert "evening" in context.user_data
    assert ai.calls == []
    assert question.replies == []
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_old_nova_controls_are_rejected_during_a_new_active_flow(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_025
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    await bot.nova_text_gate(
        update_for(NovaMessage("Как добавить задачу с напоминанием?"), user_id=user_id),
        context,
    )
    callback = callback_with_prefix(
        context.bot.edits[-1]["reply_markup"],
        "nova:action:task_create:",
    )
    context.user_data["evening"] = {}
    dispatch = AsyncMock()
    bot._nova_dispatch_capability = dispatch

    action_query = NovaQuery(callback, canonical)
    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=action_query),
        context,
    )
    assert action_query.answers == [(NOVA_STALE_ALERT, True)]
    dispatch.assert_not_awaited()

    static_query = NovaQuery("nova:root", canonical)
    await bot.nova_callback(
        update_for(canonical, user_id=user_id, query=static_query),
        context,
    )
    assert static_query.answers == [(NOVA_STALE_ALERT, True)]
    assert static_query.edits == []
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_data", ["labs:add", "collection:new", "kh:capture"])
async def test_legacy_flow_callback_clears_nova_before_lower_handler(
    db,
    tmp_path,
    callback_data,
):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_knowledge_hub=True,
        enable_knowledge_capture=True,
        _knowledge_asset_root=tmp_path / "knowledge",
    )
    user_id = {
        "labs:add": 61_026,
        "collection:new": 61_027,
        "kh:capture": 61_028,
    }[callback_data]
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    query = NovaQuery(callback_data, canonical)

    await bot.knowledge_other_callback_gate(
        update_for(canonical, user_id=user_id, query=query),
        context,
    )

    assert query.answers == []
    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("media_kind", ["photo", "document"])
async def test_photo_and_document_leave_nova_before_existing_media_pipeline(db, media_kind):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_032
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, _canonical, context = await open_nova(bot, user_id=user_id)
    media = {media_kind: [object()] if media_kind == "photo" else object()}

    await bot.nova_non_text_gate(
        update_for(NovaMessage(message_id=61_032, **media), user_id=user_id),
        context,
    )

    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("media_kind", ["photo", "document"])
async def test_photo_and_document_are_contained_by_awaiting_memory_flow(db, media_kind):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_084
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    memory = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=84_001,
        phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
    )
    bot.reminder_clear_current = AsyncMock()
    bot.nova_clear_current = AsyncMock()
    payload = [object()] if media_kind == "photo" else object()
    message = NovaMessage(**{media_kind: payload})
    context = nova_context()

    with pytest.raises(ApplicationHandlerStop):
        await bot.nova_non_text_gate(
            update_for(message, user_id=user_id),
            context,
        )

    live = await bot.nova_memory_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert live is not None and live.id == memory.id
    assert context.bot.edits[-1]["message_id"] == memory.canonical_message_id
    assert context.bot.edits[-1]["text"].endswith("Пришли текст или голосовое сообщение.")
    bot.reminder_clear_current.assert_not_awaited()
    bot.nova_clear_current.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("media_kind", ["voice", "audio"])
async def test_voice_and_audio_do_not_clear_active_nova_before_stt(db, media_kind):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_036
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, _canonical, context = await open_nova(bot, user_id=user_id)
    before = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert before is not None

    await bot.nova_non_text_gate(
        update_for(
            NovaMessage(message_id=61_036, **{media_kind: NovaVoice()}),
            user_id=user_id,
        ),
        context,
    )

    after = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert after == before


@pytest.mark.asyncio
@pytest.mark.parametrize("media_kind", ["voice", "audio"])
async def test_voice_and_audio_leave_awaiting_memory_for_stt(db, media_kind):
    ai = NovaAIStub()
    bot = make_bot(
        db,
        ai,
        enable_nova_memory=True,
        nova_memory_admin_only=False,
    )
    user_id = 61_085
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    memory = await seed_memory_flow(
        bot,
        user,
        chat_id=user_id,
        canonical_message_id=85_001,
        phase=NovaMemoryFlowPhase.AWAITING_CREATE_CONTENT,
    )
    bot.reminder_clear_current = AsyncMock()
    bot.nova_clear_current = AsyncMock()

    await bot.nova_non_text_gate(
        update_for(
            NovaMessage(**{media_kind: NovaVoice()}),
            user_id=user_id,
        ),
        nova_context(),
    )

    assert (
        await bot.nova_memory_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        == memory
    )
    bot.reminder_clear_current.assert_not_awaited()
    bot.nova_clear_current.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["/health_edit 1", "/doctor_prepare_edit 1", "/cleanup_drafts"],
)
async def test_stateful_command_alias_clears_nova_before_conversation_entry(db, command):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = {
        "/health_edit 1": 61_033,
        "/doctor_prepare_edit 1": 61_034,
        "/cleanup_drafts": 61_035,
    }[command]
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, _canonical, context = await open_nova(bot, user_id=user_id)

    await bot.navigation_public_command_gate(
        update_for(NovaMessage(command), user_id=user_id),
        context,
    )

    assert (
        await bot.nova_sessions.current(
            owner_id=user.id,
            telegram_user_id=user_id,
            chat_id=user_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_topic_callback_during_provider_is_busy_and_does_not_replace_result(db):
    ai = NovaAIStub()
    ai.release.clear()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = 61_030
    user = await user_with_tier(bot, user_id, ADMIN)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    question = NovaMessage("совершенно неизвестный вопрос")
    task = asyncio.create_task(
        bot.nova_text_gate(update_for(question, user_id=user_id), context, user=user)
    )
    await ai.started.wait()

    query = NovaQuery("nova:topic:privacy", canonical)
    await bot.nova_callback(update_for(canonical, user_id=user_id, query=query), context)
    assert query.answers == [(NOVA_BUSY_ALERT, True)]
    assert query.edits == []

    ai.release.set()
    assert await task is True
    assert len(ai.calls) == 1
    assert len(context.bot.edits) == 1


@pytest.mark.asyncio
async def test_cancel_during_provider_prevents_stale_result_delivery(db):
    ai = NovaAIStub()
    ai.release.clear()
    ai.result = NovaHelpPlan(
        response="CANCELLED_PROVIDER_RESULT",
        steps=[],
        action_id=None,
        kind="clarify",
    )
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = 61_031
    user = await user_with_tier(bot, user_id, ADMIN)
    _command, _canonical, context = await open_nova(bot, user_id=user_id)
    question_task = asyncio.create_task(
        bot.nova_text_gate(
            update_for(NovaMessage("неизвестный вопрос"), user_id=user_id),
            context,
            user=user,
        )
    )
    await ai.started.wait()

    with pytest.raises(ApplicationHandlerStop):
        await bot.nova_cancel_gate(
            update_for(NovaMessage("/cancel"), user_id=user_id),
            context,
        )
    ai.release.set()
    assert await question_task is True

    assert len(ai.calls) == 1
    assert all("CANCELLED_PROVIDER_RESULT" not in edit["text"] for edit in context.bot.edits)
    assert context.bot.edits[-1]["text"] == "✨ Nova\n\nСессия завершена."
    assert await bot.nova_sessions.count() == 0


@pytest.mark.asyncio
async def test_generic_callback_edit_error_has_no_fallback_spam_and_answers_once(db, caplog):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_040
    await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    query = NovaQuery(
        "nova:topic:privacy",
        canonical,
        edit_error=BadRequest("PRIVATE_TELEGRAM_ERROR_BODY"),
    )

    with caplog.at_level(logging.WARNING, logger="future_self.nova_handlers"):
        await bot.nova_callback(
            update_for(canonical, user_id=user_id, query=query),
            context,
        )

    assert query.answers == [(None, False)]
    assert query.edits == []
    assert canonical.replies == []
    assert context.bot.sent == []
    assert "PRIVATE_TELEGRAM_ERROR_BODY" not in caplog.text
    assert "BadRequest" in caplog.text


@pytest.mark.asyncio
async def test_message_not_modified_is_success_without_replacement(db):
    ai = NovaAIStub()
    bot = make_bot(db, ai)
    user_id = 61_041
    user = await user_with_tier(bot, user_id, SUBSCRIBER)
    _command, canonical, context = await open_nova(bot, user_id=user_id)
    query = NovaQuery(
        "nova:root",
        canonical,
        edit_error=BadRequest("Message is not modified"),
    )

    await bot.nova_callback(update_for(canonical, user_id=user_id, query=query), context)

    assert query.answers == [(None, False)]
    assert canonical.replies == []
    assert context.bot.sent == []
    rebound = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert rebound is not None
    assert rebound.canonical_message_id == canonical.message_id


@pytest.mark.asyncio
async def test_cancelled_error_from_provider_propagates_without_sensitive_logging(db, caplog):
    ai = NovaAIStub()
    ai.error = asyncio.CancelledError()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = 61_050
    await user_with_tier(bot, user_id, ADMIN)
    _command, _canonical, context = await open_nova(bot, user_id=user_id)
    raw_question = "RAW_CANCELLED_QUESTION_SENTINEL"

    with caplog.at_level(logging.WARNING, logger="future_self.nova_handlers"):
        with pytest.raises(asyncio.CancelledError):
            await bot.nova_text_gate(
                update_for(NovaMessage(raw_question), user_id=user_id),
                context,
            )

    assert len(ai.calls) == 1
    assert raw_question not in caplog.text
    assert context.user_data == {}
    assert context.bot.sent == []


@pytest.mark.asyncio
async def test_raw_question_provider_output_and_error_body_are_not_persisted_or_logged(db, caplog):
    ai = NovaAIStub()
    bot = make_bot(db, ai, enable_nova_ai=True)
    user_id = 61_051
    user = await user_with_tier(bot, user_id, ADMIN)
    _command, _canonical, context = await open_nova(bot, user_id=user_id)
    raw = "RAW_NOVA_INPUT_SENTINEL"
    provider_output = "PRIVATE_PROVIDER_OUTPUT_SENTINEL"
    ai.result = NovaHelpPlan(
        response=provider_output,
        steps=[],
        action_id=None,
        kind="clarify",
    )

    with caplog.at_level(logging.WARNING, logger="future_self.nova_handlers"):
        await bot.nova_text_gate(update_for(NovaMessage(raw), user_id=user_id), context)
    current = await bot.nova_sessions.current(
        owner_id=user.id,
        telegram_user_id=user_id,
        chat_id=user_id,
    )
    assert current is not None
    assert all(value not in repr(current) for value in (raw, provider_output))
    assert all(value not in repr(context.user_data) for value in (raw, provider_output))

    database_dump: list[str] = []
    async with db.sessions() as session:
        for table in Base.metadata.sorted_tables:
            rows = (await session.execute(select(table))).all()
            database_dump.append(repr(rows))
    persisted = "\n".join(database_dump)
    assert raw not in persisted
    assert provider_output not in persisted
    assert raw not in caplog.text
    assert provider_output not in caplog.text

    ai.error = RuntimeError("PRIVATE_PROVIDER_ERROR_BODY_SENTINEL")
    with caplog.at_level(logging.WARNING, logger="future_self.nova_handlers"):
        await bot.nova_text_gate(
            update_for(NovaMessage("SECOND_RAW_INPUT_SENTINEL"), user_id=user_id),
            context,
        )
    assert context.bot.edits[-1]["text"] == f"✨ Nova\n\n{NOVA_PROVIDER_FAILURE_TEXT}"
    assert "PRIVATE_PROVIDER_ERROR_BODY_SENTINEL" not in caplog.text
    assert "SECOND_RAW_INPUT_SENTINEL" not in caplog.text
    assert "RuntimeError" in caplog.text
