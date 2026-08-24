from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select, update
from telegram.error import BadRequest
from telegram.ext import ApplicationHandlerStop

from future_self.access import AccessService
from future_self.ai import NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.dates import DateResolver
from future_self.models import (
    ConversationMessage,
    ConversationSession,
    DraftInboxItem,
    Goal,
    InboxItem,
    NovaDialogueState,
    NovaMemoryItem,
    NovaObservedMemory,
    TaskReminder,
    User,
    VisionItem,
    VisionProfile,
    WeeklyFocus,
)
from future_self.nova_companion_flow import CaptureSuggestion, NovaCompanionCaptureStore
from future_self.nova_companion_handlers import (
    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
    NOVA_COMPANION_CONTEXT_CHANGED_TEXT,
    NOVA_COMPANION_DISCOURSE_AMBIGUOUS_TEXT,
    NOVA_COMPANION_NO_ACTIVE_REMINDER_OFFER_TEXT,
    NOVA_COMPANION_NOT_EXECUTED_TEXT,
    NOVA_COMPANION_REMINDER_OFFER_ACTION_TEXT,
    NOVA_COMPANION_UNAVAILABLE_TEXT,
)
from future_self.reminder_flow import ReminderFlowPhase
from future_self.schemas import (
    NovaCompanionCapture,
    NovaCompanionDialogueStateUpdate,
    NovaCompanionMemoryCandidate,
    NovaCompanionProviderCapture,
    NovaCompanionProviderReminderOffer,
    NovaCompanionProviderResponse,
    NovaCompanionReminderOffer,
    NovaCompanionResponse,
    ParsedThought,
)
from future_self.weekly_review import current_week_start


class CompanionMessage:
    _ids = count(81_000)

    def __init__(
        self,
        text: str | None = None,
        *,
        chat_id: int = 71_000,
        reply_error: BaseException | None = None,
        edit_error: BaseException | None = None,
        markup_error: BaseException | None = None,
        block_markup: bool = False,
    ) -> None:
        self.text = text
        self.chat_id = chat_id
        self.chat = SimpleNamespace(id=chat_id)
        self.message_id = next(self._ids)
        self.voice = None
        self.audio = None
        self.photo = []
        self.document = None
        self.reply_to_message = None
        self.reply_error = reply_error
        self.edit_error = edit_error
        self.markup_error = markup_error
        self.block_markup = block_markup
        self.markup_started = asyncio.Event()
        self.markup_release = asyncio.Event()
        if not block_markup:
            self.markup_release.set()
        self.replies: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.markup_edits: list[Any] = []
        self.deleted = 0

    async def reply_text(self, text: str, **kwargs: Any) -> CompanionMessage:
        if self.reply_error is not None:
            raise self.reply_error
        sent = CompanionMessage(text, chat_id=self.chat_id)
        self.replies.append({"text": text, "message": sent, **kwargs})
        return sent

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        if self.edit_error is not None:
            raise self.edit_error
        self.edits.append({"text": text, **kwargs})

    async def edit_reply_markup(self, reply_markup: Any = None) -> None:
        self.markup_edits.append(reply_markup)
        self.markup_started.set()
        if self.markup_error is not None:
            raise self.markup_error
        await self.markup_release.wait()

    async def delete(self) -> bool:
        self.deleted += 1
        return True


class BlockingSuggestionMessage(CompanionMessage):
    """Incoming message whose one primary reply blocks while adding controls."""

    def __init__(self, text: str, *, chat_id: int) -> None:
        super().__init__(text, chat_id=chat_id)
        self.sent: CompanionMessage | None = None
        self.reply_started = asyncio.Event()

    async def reply_text(self, text: str, **kwargs: Any) -> CompanionMessage:
        sent = CompanionMessage(text, chat_id=self.chat_id, block_markup=True)
        self.sent = sent
        self.replies.append({"text": text, "message": sent, **kwargs})
        self.reply_started.set()
        return sent


class CancelledSuggestionMessage(CompanionMessage):
    """Incoming message whose primary reply gets a direct Telegram cancellation."""

    def __init__(self, text: str, *, chat_id: int) -> None:
        super().__init__(text, chat_id=chat_id)
        self.sent: CompanionMessage | None = None

    async def reply_text(self, text: str, **kwargs: Any) -> CompanionMessage:
        sent = CompanionMessage(
            text,
            chat_id=self.chat_id,
            markup_error=asyncio.CancelledError(),
        )
        self.sent = sent
        self.replies.append({"text": text, "message": sent, **kwargs})
        return sent


class CompanionQuery:
    def __init__(
        self,
        data: str,
        message: CompanionMessage,
        *,
        answer_error: BaseException | None = None,
    ) -> None:
        self.data = data
        self.message = message
        self.answer_error = answer_error
        self.answer_attempts = 0
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[dict[str, Any]] = []
        self.markup_removed = 0

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        self.answer_attempts += 1
        if self.answer_error is not None:
            raise self.answer_error
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs: Any) -> None:
        self.edits.append({"text": text, **kwargs})

    async def edit_message_reply_markup(self, reply_markup: Any = None) -> None:
        assert reply_markup is None
        self.markup_removed += 1


class BlockingAnswerQuery(CompanionQuery):
    def __init__(self, data: str, message: CompanionMessage) -> None:
        super().__init__(data, message)
        self.answer_started = asyncio.Event()
        self.answer_release = asyncio.Event()

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:
        self.answer_attempts += 1
        self.answer_started.set()
        await self.answer_release.wait()
        self.answers.append((text, show_alert))


class CompanionTelegram:
    def __init__(self) -> None:
        self.deleted: list[tuple[int, int]] = []
        self.neutralized: list[dict[str, Any]] = []
        self.markup_edits: list[dict[str, Any]] = []

    async def delete_message(self, *, chat_id: int, message_id: int) -> bool:
        self.deleted.append((chat_id, message_id))
        return True

    async def edit_message_text(self, **kwargs: Any) -> None:
        self.neutralized.append(kwargs)

    async def edit_message_reply_markup(self, **kwargs: Any) -> None:
        self.markup_edits.append(kwargs)


class NoopTranscription:
    enabled = False


def companion_settings(db: Any) -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        database_url=db.url,
        enable_nova_companion=True,
        nova_companion_admin_only=False,
        enable_nova_memory_application=False,
    )


def companion_context(telegram: CompanionTelegram | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        user_data={},
        args=[],
        bot=telegram or CompanionTelegram(),
        application=SimpleNamespace(create_task=lambda *_args, **_kwargs: None),
    )


def companion_update(
    message: CompanionMessage,
    *,
    telegram_user_id: int,
    chat_id: int,
    query: CompanionQuery | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        update_id=91_000,
        effective_user=SimpleNamespace(id=telegram_user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
        effective_message=message,
        message=message,
        callback_query=query,
    )


async def ready_companion_bot(
    db: Any,
    ai: Any,
    *,
    telegram_user_id: int = 71_001,
    enable_brain: bool = False,
    brain_admin_only: bool = False,
) -> tuple[FutureSelfBot, User]:
    settings = companion_settings(db).model_copy(
        update={
            "enable_nova_conversation_brain": enable_brain,
            "nova_conversation_brain_admin_only": brain_admin_only,
        }
    )
    bot = FutureSelfBot(settings, db, ai, NoopTranscription())
    await bot._user(telegram_user_id)
    await AccessService(db).grant_subscriber(telegram_user_id, source="companion-handler-test")
    async with db.session() as session:
        await session.execute(
            update(User)
            .where(User.telegram_id == telegram_user_id)
            .values(onboarding_completed=True)
        )
    return bot, await bot._user(telegram_user_id)


async def companion_counts(db: Any) -> tuple[int, int, int]:
    async with db.sessions() as session:
        messages = await session.scalar(select(func.count(ConversationMessage.id)))
        drafts = await session.scalar(select(func.count(DraftInboxItem.id)))
        inbox = await session.scalar(select(func.count(InboxItem.id)))
    return int(messages or 0), int(drafts or 0), int(inbox or 0)


async def active_companion_drafts(db: Any) -> list[DraftInboxItem]:
    async with db.sessions() as session:
        return list(
            (
                await session.scalars(
                    select(DraftInboxItem).where(DraftInboxItem.status.in_(("preview", "editing")))
                )
            ).all()
        )


def callback_by_label(markup: Any, label: str) -> str:
    for row in markup.inline_keyboard:
        for button in row:
            if button.text == label:
                assert isinstance(button.callback_data, str)
                return button.callback_data
    raise AssertionError(f"Missing callback button: {label}")


def capture_result() -> NovaCompanionResponse:
    return NovaCompanionResponse(
        answer="Похоже, здесь уже есть небольшой конкретный шаг.",
        capture=NovaCompanionCapture(
            kind="task",
            title="подготовить письмо Марине",
            next_step="открыть документ",
        ),
    )


async def deliver(
    bot: FutureSelfBot,
    user: User,
    message: CompanionMessage,
    *,
    context: SimpleNamespace | None = None,
    source: str = "text",
    delivery_message: CompanionMessage | None = None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    actual_context = context or companion_context()
    update = companion_update(
        message,
        telegram_user_id=user.telegram_id,
        chat_id=message.chat_id,
    )
    snapshot = await bot.conversation.get(user.telegram_id, message.chat_id)
    handled = await bot.nova_companion_route(
        update,
        actual_context,
        message.text or "",
        source,
        user=user,
        conversation_snapshot=snapshot,
        delivery_message=delivery_message,
    )
    assert handled
    await asyncio.sleep(0)
    assert bot._nova_companion_tasks == set()
    return update, actual_context


def sent_answer(message: CompanionMessage) -> CompanionMessage:
    assert len(message.replies) == 1
    sent = message.replies[0]["message"]
    assert isinstance(sent, CompanionMessage)
    return sent


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_local_wake_is_local_for_text_and_voice_without_provider_or_domain_dml(
    db,
    fake_ai,
    source,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage("Нова, ты тут?", chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
    )

    assert fake_ai.companion_calls == []
    assert await companion_counts(db) == (0, 0, 0)
    if progress is None:
        assert sent_answer(incoming).text == "Да, я здесь 🙂"
    else:
        assert incoming.replies == []
        assert progress.edits == [{"text": "Да, я здесь 🙂", "reply_markup": None}]


async def test_ordinary_conversation_answers_once_and_never_auto_creates_domain_rows(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Сегодня было непросто, хочу просто спокойно поговорить.",
        chat_id=user.telegram_id,
    )

    await deliver(bot, user, incoming)

    assert len(fake_ai.companion_calls) == 1
    assert sent_answer(incoming).text == fake_ai.companion_result.answer
    assert await companion_counts(db) == (2, 0, 0)
    async with db.sessions() as session:
        intents = tuple(
            await session.scalars(
                select(ConversationMessage.intent).order_by(ConversationMessage.id)
            )
        )
    assert intents == ("companion_user", "companion_answer")


async def test_grounded_suggestion_is_answer_first_with_only_opaque_bound_controls(db, fake_ai):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    raw_text = "Я хочу подготовить письмо Марине и открыть документ"
    incoming = CompanionMessage(raw_text, chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    answer = sent_answer(incoming)
    assert incoming.replies[0].get("reply_markup") is None
    assert len(answer.markup_edits) == 1
    markup = answer.markup_edits[0]
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert len(callbacks) == 2
    assert all(data.startswith("ncap:") and len(data.encode()) <= 64 for data in callbacks)
    assert all(raw_text not in data and "Марине" not in data for data in callbacks)
    assert await companion_counts(db) == (2, 0, 0)


async def test_add_creates_only_draft_then_existing_save_creates_inbox_item(db, fake_ai):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    _update, context = await deliver(bot, user, incoming)
    answer = sent_answer(incoming)
    add_data = callback_by_label(answer.markup_edits[-1], "Добавить как задачу")
    add_query = CompanionQuery(add_data, answer)

    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=add_query,
        ),
        context,
    )

    assert add_query.answer_attempts == 1
    assert add_query.markup_removed == 1
    # The existing preview lifecycle records one bounded preview message; it
    # still does not create an InboxItem until its own confirm callback.
    assert await companion_counts(db) == (3, 1, 0)
    assert len(answer.replies) == 1
    preview = answer.replies[0]["message"]
    preview_markup = answer.replies[0]["reply_markup"]
    save_data = callback_by_label(preview_markup, "Сохранить")
    save_query = CompanionQuery(save_data, preview)

    await bot.inbox_action(
        companion_update(
            preview,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=save_query,
        ),
        context,
    )

    assert save_query.answer_attempts == 1
    assert await companion_counts(db) == (3, 1, 1)
    async with db.sessions() as session:
        draft = await session.scalar(select(DraftInboxItem))
        item = await session.scalar(select(InboxItem))
    assert draft is not None and draft.status == "confirmed"
    assert item is not None and item.draft_id == draft.id
    assert item.source == "companion"
    for status_question in (
        "Ты сохранила задачу?",
        "Ты создала задачу?",
        "Нова, ты создала задачу?",
        "Nova, ты создала задачу?",
        "Готово?",
        "Всё готово?",
        "Ну что, всё готово?",
    ):
        status = CompanionMessage(status_question, chat_id=user.telegram_id)
        await deliver(bot, user, status, context=context)
        assert sent_answer(status).text == "Да, задача сохранена."
    assert len(fake_ai.companion_calls) == 1

    receipt_key = (user.id, user.telegram_id, user.telegram_id)
    exact_receipt = bot._nova_companion_status_receipts[receipt_key]
    assert (
        bot.nova_companion_invalidate_status_for_input(
            user.telegram_id,
            user.telegram_id + 1,
            "Создай другую задачу",
        )
        is False
    )
    assert bot._nova_companion_status_receipts[receipt_key] is exact_receipt
    assert (
        bot.nova_companion_invalidate_status_for_input(
            user.telegram_id + 1,
            user.telegram_id,
            "Создай другую задачу",
        )
        is False
    )
    assert bot._nova_companion_status_receipts[receipt_key] is exact_receipt
    assert (
        bot.nova_companion_invalidate_status_for_input(
            user.telegram_id,
            user.telegram_id,
            "Нова, ты сохранила задачу?",
        )
        is False
    )
    assert bot._nova_companion_status_receipts[receipt_key] is exact_receipt
    newer_receipt = replace(
        exact_receipt,
        inbox_item_version=exact_receipt.inbox_item_version + 1,
    )
    bot._nova_companion_status_receipts[receipt_key] = newer_receipt
    assert (
        bot.nova_companion_invalidate_status_for_input(
            user.telegram_id,
            user.telegram_id,
            "Создай другую задачу",
            expected_receipt=exact_receipt,
        )
        is False
    )
    assert bot._nova_companion_status_receipts[receipt_key] is newer_receipt
    bot._nova_companion_status_receipts[receipt_key] = exact_receipt

    direct_task = CompanionMessage(
        "Создай задачу подготовить новую презентацию",
        chat_id=user.telegram_id,
    )
    direct_update = companion_update(
        direct_task,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    await bot.navigation_text_gate(direct_update, context)
    await bot.text(direct_update, context)
    assert len(await active_companion_drafts(db)) == 1
    stale_status = CompanionMessage("Ты сохранила задачу?", chat_id=user.telegram_id)
    await deliver(bot, user, stale_status, context=context)
    assert sent_answer(stale_status).text == "Пока нет — задача ещё не создана."
    assert len(fake_ai.companion_calls) == 1

    bot._nova_companion_status_receipts[receipt_key] = exact_receipt
    await AccessService(db).block(
        user.telegram_id,
        source="companion-status-access-isolation",
    )
    assert (
        await bot._nova_companion_status_answer(
            "Ты сохранила задачу?",
            user=user,
            chat_id=user.telegram_id,
        )
        == "Пока нет — задача ещё не создана."
    )
    assert bot._nova_companion_status_receipts[receipt_key] is exact_receipt


async def test_discarded_pending_capture_cannot_mask_later_exact_confirmed_status(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    fake_ai.companion_result = capture_result()
    first = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    await deliver(bot, user, first, context=context)
    first_answer = sent_answer(first)
    first_add = callback_by_label(
        first_answer.markup_edits[-1],
        "Добавить как задачу",
    )
    await bot.nova_companion_callback(
        companion_update(
            first_answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(first_add, first_answer),
        ),
        context,
    )
    first_preview = first_answer.replies[-1]["message"]
    first_markup = first_answer.replies[-1]["reply_markup"]
    first_drop = callback_by_label(first_markup, "Не сохранять")
    await bot.inbox_action(
        companion_update(
            first_preview,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(first_drop, first_preview),
        ),
        context,
    )

    async with db.sessions() as session:
        discarded = await session.scalar(
            select(DraftInboxItem).where(DraftInboxItem.status == "discarded")
        )
    assert discarded is not None
    assert discarded.id not in bot._nova_companion_pending_capture_status

    fake_ai.companion_result = NovaCompanionResponse(
        answer="Это можно оформить как новую задачу.",
        capture=NovaCompanionCapture(
            kind="task",
            title="позвонить врачу",
            next_step="открыть контакты",
        ),
    )
    second = CompanionMessage(
        "Я хочу позвонить врачу и открыть контакты",
        chat_id=user.telegram_id,
    )
    await deliver(bot, user, second, context=context)
    second_answer = sent_answer(second)
    second_add = callback_by_label(
        second_answer.markup_edits[-1],
        "Добавить как задачу",
    )
    await bot.nova_companion_callback(
        companion_update(
            second_answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(second_add, second_answer),
        ),
        context,
    )
    second_preview = second_answer.replies[-1]["message"]
    second_markup = second_answer.replies[-1]["reply_markup"]
    second_save = callback_by_label(second_markup, "Сохранить")
    await bot.inbox_action(
        companion_update(
            second_preview,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(second_save, second_preview),
        ),
        context,
    )

    status = CompanionMessage("Ты создала задачу?", chat_id=user.telegram_id)
    await deliver(bot, user, status, context=context)
    assert sent_answer(status).text == "Да, задача сохранена."
    assert len(fake_ai.companion_calls) == 2
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
    assert bot._nova_companion_pending_capture_status == {}


@pytest.mark.parametrize("old_action", ["save", "drop"])
@pytest.mark.parametrize("access_bounce", [False, True])
async def test_reused_same_id_version_preview_rejects_old_canonical_terminal_callback(
    db,
    fake_ai,
    old_action,
    access_bounce,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    fake_ai.companion_result = capture_result()
    text = "Я хочу подготовить письмо Марине и открыть документ"

    first = CompanionMessage(text, chat_id=user.telegram_id)
    await deliver(bot, user, first, context=context)
    first_answer = sent_answer(first)
    await bot.nova_companion_callback(
        companion_update(
            first_answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(
                callback_by_label(first_answer.markup_edits[-1], "Добавить как задачу"),
                first_answer,
            ),
        ),
        context,
    )
    preview_a = first_answer.replies[-1]["message"]
    markup_a = first_answer.replies[-1]["reply_markup"]
    async with db.sessions() as session:
        draft_a = await session.scalar(select(DraftInboxItem))
    assert draft_a is not None and draft_a.preview_message_id == preview_a.message_id

    if access_bounce:
        await AccessService(db).block(user.telegram_id, source="pending-preview-bounce")
        await AccessService(db).grant_subscriber(
            user.telegram_id,
            source="pending-preview-bounce",
        )
        user = await bot._user(user.telegram_id)

    second = CompanionMessage(text, chat_id=user.telegram_id)
    await deliver(bot, user, second, context=context)
    second_answer = sent_answer(second)
    await bot.nova_companion_callback(
        companion_update(
            second_answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(
                callback_by_label(second_answer.markup_edits[-1], "Добавить как задачу"),
                second_answer,
            ),
        ),
        context,
    )
    preview_b = second_answer.replies[-1]["message"]
    markup_b = second_answer.replies[-1]["reply_markup"]
    async with db.sessions() as session:
        draft_b = await session.scalar(select(DraftInboxItem))
    assert draft_b is not None
    assert (draft_b.id, draft_b.version) == (draft_a.id, draft_a.version)
    assert draft_b.preview_message_id == preview_b.message_id != preview_a.message_id
    pending_b = bot._nova_companion_pending_capture_status[draft_b.id]
    assert pending_b.canonical_message_id == preview_b.message_id
    assert pending_b.access_version == user.access_version

    old_query = CompanionQuery(
        callback_by_label(
            markup_a,
            "Сохранить" if old_action == "save" else "Не сохранять",
        ),
        preview_a,
    )
    await bot.inbox_action(
        companion_update(
            preview_a,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=old_query,
        ),
        context,
    )

    assert old_query.answer_attempts == 1
    assert old_query.answers[-1][1] is True
    async with db.sessions() as session:
        after_old = await session.get(DraftInboxItem, draft_b.id)
        assert after_old is not None
        assert after_old.status == "preview"
        assert after_old.preview_message_id == preview_b.message_id
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert bot._nova_companion_pending_capture_status[draft_b.id] is pending_b

    current_query = CompanionQuery(callback_by_label(markup_b, "Сохранить"), preview_b)
    await bot.inbox_action(
        companion_update(
            preview_b,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=current_query,
        ),
        context,
    )

    assert current_query.answer_attempts == 1
    async with db.sessions() as session:
        saved = await session.get(DraftInboxItem, draft_b.id)
        assert saved is not None and saved.status == "confirmed"
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
    assert draft_b.id not in bot._nova_companion_pending_capture_status
    assert len(fake_ai.companion_calls) == 2
    assert bot._nova_companion_tasks == set()


async def test_terminal_save_waiting_at_dml_preserves_newer_same_version_preview(
    db,
    fake_ai,
    monkeypatch,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    fake_ai.companion_result = capture_result()
    text = "Я хочу подготовить письмо Марине и открыть документ"

    first = CompanionMessage(text, chat_id=user.telegram_id)
    await deliver(bot, user, first, context=context)
    first_answer = sent_answer(first)
    await bot.nova_companion_callback(
        companion_update(
            first_answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(
                callback_by_label(first_answer.markup_edits[-1], "Добавить как задачу"),
                first_answer,
            ),
        ),
        context,
    )
    preview_a = first_answer.replies[-1]["message"]
    markup_a = first_answer.replies[-1]["reply_markup"]
    save_a = CompanionQuery(callback_by_label(markup_a, "Сохранить"), preview_a)
    save_started = asyncio.Event()
    release_save = asyncio.Event()
    original_confirm = bot.draft_service.confirm

    async def blocked_confirm(*args: Any, **kwargs: Any):
        save_started.set()
        await release_save.wait()
        return await original_confirm(*args, **kwargs)

    monkeypatch.setattr(bot.draft_service, "confirm", blocked_confirm)
    old_task = asyncio.create_task(
        bot.inbox_action(
            companion_update(
                preview_a,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=save_a,
            ),
            context,
        )
    )
    await save_started.wait()

    second = CompanionMessage(text, chat_id=user.telegram_id)
    await deliver(bot, user, second, context=context)
    second_answer = sent_answer(second)
    await bot.nova_companion_callback(
        companion_update(
            second_answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(
                callback_by_label(second_answer.markup_edits[-1], "Добавить как задачу"),
                second_answer,
            ),
        ),
        context,
    )
    preview_b = second_answer.replies[-1]["message"]
    markup_b = second_answer.replies[-1]["reply_markup"]
    pending_b = next(iter(bot._nova_companion_pending_capture_status.values()))
    release_save.set()
    await old_task

    assert save_a.answer_attempts == 1
    assert save_a.answers[-1][1] is True
    async with db.sessions() as session:
        draft = await session.scalar(select(DraftInboxItem))
        assert draft is not None and draft.status == "preview"
        assert draft.preview_message_id == preview_b.message_id
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert bot._nova_companion_pending_capture_status[pending_b.draft_id] is pending_b

    save_b = CompanionQuery(callback_by_label(markup_b, "Сохранить"), preview_b)
    await bot.inbox_action(
        companion_update(
            preview_b,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=save_b,
        ),
        context,
    )
    assert save_b.answer_attempts == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
    assert bot._nova_companion_pending_capture_status == {}


async def test_inbox_terminal_forged_and_cross_user_callbacks_preserve_exact_pending_preview(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    fake_ai.companion_result = capture_result()
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    await deliver(bot, user, incoming, context=context)
    answer = sent_answer(incoming)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=CompanionQuery(
                callback_by_label(answer.markup_edits[-1], "Добавить как задачу"),
                answer,
            ),
        ),
        context,
    )
    preview = answer.replies[-1]["message"]
    markup = answer.replies[-1]["reply_markup"]
    valid_callback = callback_by_label(markup, "Сохранить")
    pending = next(iter(bot._nova_companion_pending_capture_status.values()))
    other_actor = user.telegram_id + 100
    await bot._user(other_actor)
    await AccessService(db).grant_subscriber(other_actor, source="cross-user-terminal")

    forged = CompanionQuery(f"inbox:save:forged:{pending.draft_version}", preview)
    crossed = CompanionQuery(valid_callback, preview)
    await bot.inbox_action(
        companion_update(
            preview,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=forged,
        ),
        context,
    )
    await bot.inbox_action(
        companion_update(
            preview,
            telegram_user_id=other_actor,
            chat_id=user.telegram_id,
            query=crossed,
        ),
        context,
    )

    assert forged.answer_attempts == crossed.answer_attempts == 1
    async with db.sessions() as session:
        draft = await session.get(DraftInboxItem, pending.draft_id)
        assert draft is not None and draft.status == "preview"
        assert draft.preview_message_id == preview.message_id
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert bot._nova_companion_pending_capture_status[pending.draft_id] is pending

    valid = CompanionQuery(valid_callback, preview)
    await bot.inbox_action(
        companion_update(
            preview,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=valid,
        ),
        context,
    )
    assert valid.answer_attempts == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
    assert bot._nova_companion_pending_capture_status == {}


async def test_pending_capture_terminal_cleanup_is_exact_for_replacement_duplicate_and_expiry(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    old = SimpleNamespace(id="draft-exact", version=1, kind="task", title="Та же задача")
    replacement = SimpleNamespace(
        id="draft-exact",
        version=1,
        kind="task",
        title="Та же задача",
    )
    bot._nova_companion_bind_pending_capture_status(
        user,
        user.telegram_id,
        old,
        canonical_message_id=91_001,
    )
    old_pending = bot._nova_companion_pending_capture_status[old.id]
    replacement_user = SimpleNamespace(
        id=user.id,
        telegram_id=user.telegram_id,
        access_version=user.access_version + 1,
    )
    bot._nova_companion_bind_pending_capture_status(
        replacement_user,
        user.telegram_id,
        replacement,
        canonical_message_id=91_002,
    )
    replacement_pending = bot._nova_companion_pending_capture_status[replacement.id]

    assert (
        bot.nova_companion_clear_pending_capture_exact(
            user.telegram_id,
            user.telegram_id,
            old.id,
            old.version,
            expected_pending=old_pending,
        )
        is False
    )
    assert bot._nova_companion_pending_capture_status[replacement.id] is replacement_pending
    assert replacement_pending.access_version == user.access_version + 1

    duplicate_item = SimpleNamespace(
        id="duplicate-item",
        draft_id=replacement.id,
        user_id=user.id,
        kind="task",
        title=replacement.title,
        version=1,
    )
    bot.nova_companion_record_confirmed_capture(
        user.telegram_id,
        user.telegram_id,
        SimpleNamespace(
            result=SimpleNamespace(
                draft=replacement,
                inbox_item=duplicate_item,
                duplicate=True,
            )
        ),
        expected_pending=replacement_pending,
    )
    assert replacement.id not in bot._nova_companion_pending_capture_status
    assert bot._nova_companion_status_receipts == {}

    expired = await bot.draft_service.create(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        source="companion",
        raw_text="Просроченная задача",
        parsed=ParsedThought(kind="task", title="Просроченная задача"),
    )
    bot._nova_companion_bind_pending_capture_status(
        user,
        user.telegram_id,
        expired,
        canonical_message_id=91_003,
    )
    async with db.session() as session:
        await session.execute(
            update(DraftInboxItem)
            .where(DraftInboxItem.id == expired.id)
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )

    answer = await bot._nova_companion_status_answer(
        "Ты создала задачу?",
        user=user,
        chat_id=user.telegram_id,
    )

    assert answer == "Пока нет — задача ещё не создана."
    assert expired.id not in bot._nova_companion_pending_capture_status


async def test_not_now_creates_nothing_and_suppresses_same_topic_generation(db, fake_ai):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    text = "Я хочу подготовить письмо Марине и открыть документ"
    first = CompanionMessage(text, chat_id=user.telegram_id)
    _update, context = await deliver(bot, user, first)
    first_answer = sent_answer(first)
    dismiss_data = callback_by_label(first_answer.markup_edits[-1], "Не сейчас")
    query = CompanionQuery(dismiss_data, first_answer)

    await bot.nova_companion_callback(
        companion_update(
            first_answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=query,
        ),
        context,
    )

    assert query.answer_attempts == 1
    assert query.markup_removed == 1
    assert await companion_counts(db) == (2, 0, 0)

    second = CompanionMessage(text, chat_id=user.telegram_id)
    await deliver(bot, user, second, context=context)
    second_answer = sent_answer(second)
    assert second_answer.markup_edits == []
    assert len(fake_ai.companion_calls) == 2
    assert await companion_counts(db) == (4, 0, 0)


async def test_continuing_conversation_does_not_auto_accept_previous_suggestion(db, fake_ai):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    first_text = "Я хочу подготовить письмо Марине и открыть документ"
    first = CompanionMessage(first_text, chat_id=user.telegram_id)
    _update, context = await deliver(bot, user, first)
    first_answer = sent_answer(first)
    add_data = callback_by_label(first_answer.markup_edits[-1], "Добавить как задачу")

    second_text = "Давай пока просто обсудим, почему мне трудно начать"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Конечно. Что именно сильнее всего мешает начать?"
    )
    second = CompanionMessage(second_text, chat_id=user.telegram_id)
    await deliver(bot, user, second, context=context)

    assert [call[0] for call in fake_ai.companion_calls] == [first_text, second_text]
    assert sent_answer(second).text == fake_ai.companion_result.answer
    assert await companion_counts(db) == (4, 0, 0)
    capability = await bot.nova_companion_captures.peek(
        add_data,
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        canonical_message_id=first_answer.message_id,
        access_version=user.access_version,
        expected_action="add",
    )
    assert capability is not None


async def test_callback_is_owner_chat_message_bound_and_replay_and_forgery_are_noops(
    db,
    fake_ai,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    _update, context = await deliver(bot, user, incoming)
    answer = sent_answer(incoming)
    dismiss_data = callback_by_label(answer.markup_edits[-1], "Не сейчас")

    crossed = CompanionQuery(dismiss_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id + 1,
            query=crossed,
        ),
        context,
    )
    forged_data = dismiss_data[:-1] + ("A" if dismiss_data[-1] != "A" else "B")
    forged = CompanionQuery(forged_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=forged,
        ),
        context,
    )
    valid = CompanionQuery(dismiss_data, answer)
    valid_update = companion_update(
        answer,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        query=valid,
    )
    await bot.nova_companion_callback(valid_update, context)
    replay = CompanionQuery(dismiss_data, answer)
    replay_update = companion_update(
        answer,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        query=replay,
    )
    await bot.nova_companion_callback(replay_update, context)

    assert [query.answer_attempts for query in (crossed, forged, valid, replay)] == [1, 1, 1, 1]
    assert crossed.markup_removed == forged.markup_removed == replay.markup_removed == 0
    assert valid.markup_removed == 1
    assert await companion_counts(db) == (2, 0, 0)
    assert len(fake_ai.companion_calls) == 1


@pytest.mark.parametrize(
    ("checks", "expected_provider_calls", "expected_replies"),
    [
        (("access_changed",), 0, 0),
        (("ready", "access_changed"), 1, 0),
    ],
)
async def test_pre_and_post_provider_access_fences_do_not_deliver_or_mutate(
    db,
    fake_ai,
    monkeypatch,
    checks,
    expected_provider_calls,
    expected_replies,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    outcomes: AsyncIterator[str] = _outcomes(checks)

    async def current_check(_generation: Any) -> str:
        return await anext(outcomes)

    monkeypatch.setattr(bot, "_nova_companion_current_check", current_check)
    incoming = CompanionMessage(
        "Мне важно спокойно выбрать следующий шаг", chat_id=user.telegram_id
    )

    await deliver(bot, user, incoming)

    assert len(fake_ai.companion_calls) == expected_provider_calls
    assert len(incoming.replies) == expected_replies
    assert await companion_counts(db) == (0, 0, 0)


async def _outcomes(values: tuple[str, ...]) -> AsyncIterator[str]:
    for value in values:
        yield value


async def test_post_send_access_bounce_deletes_exact_answer_and_never_falls_back(
    db,
    fake_ai,
    monkeypatch,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    outcomes = iter(("ready", "ready", "ready", "access_changed"))

    async def current_check(_generation: Any) -> str:
        return next(outcomes)

    monkeypatch.setattr(bot, "_nova_companion_current_check", current_check)
    incoming = CompanionMessage(
        "Мне важно спокойно выбрать следующий шаг", chat_id=user.telegram_id
    )
    telegram = CompanionTelegram()
    context = companion_context(telegram)

    await deliver(bot, user, incoming, context=context)

    answer = sent_answer(incoming)
    assert telegram.deleted == [(user.telegram_id, answer.message_id)]
    assert telegram.neutralized == []
    assert len(incoming.replies) == 1
    assert len(fake_ai.companion_calls) == 1
    assert await companion_counts(db) == (0, 0, 0)


async def test_provider_and_primary_send_failures_are_private_single_attempt_noops(
    db,
    fake_ai,
    caplog,
):
    private = "PRIVATE_COMPANION_PROVIDER_PAYLOAD"
    fake_ai.companion_error = RuntimeError(private)
    bot, user = await ready_companion_bot(db, fake_ai)
    provider_message = CompanionMessage("Мне нужен спокойный разговор", chat_id=user.telegram_id)

    with caplog.at_level(logging.WARNING):
        await deliver(bot, user, provider_message)

    assert len(fake_ai.companion_calls) == 1
    assert len(provider_message.replies) == 1
    assert provider_message.replies[0]["text"] == NOVA_COMPANION_UNAVAILABLE_TEXT
    assert provider_message.replies[0].get("reply_markup") is None
    assert private not in caplog.text
    assert await companion_counts(db) == (0, 0, 0)
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert bot._nova_companion_tasks == set()

    fake_ai.companion_error = None
    send_private = "PRIVATE_TELEGRAM_ERROR_TEXT"
    failed_send = CompanionMessage(
        "Мне нужен спокойный разговор",
        chat_id=user.telegram_id,
        reply_error=RuntimeError(send_private),
    )
    with caplog.at_level(logging.WARNING):
        await deliver(bot, user, failed_send)

    assert len(fake_ai.companion_calls) == 2
    assert failed_send.replies == []
    assert send_private not in caplog.text
    assert await companion_counts(db) == (0, 0, 0)
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert bot._nova_companion_tasks == set()


async def test_exchange_second_insert_failure_rolls_back_and_revokes_exact_offer(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_SECOND_EXCHANGE_INSERT"
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    telegram = CompanionTelegram()
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )

    async def fail_second_insert(_fence: Any) -> None:
        raise RuntimeError(private)

    monkeypatch.setattr(
        bot.conversation,
        "_before_exchange_assistant_insert",
        fail_second_insert,
    )
    with caplog.at_level(logging.WARNING):
        await deliver(
            bot,
            user,
            incoming,
            context=companion_context(telegram),
        )

    answer = sent_answer(incoming)
    dismiss_data = callback_by_label(answer.markup_edits[-1], "Не сейчас")
    assert telegram.deleted == [(user.telegram_id, answer.message_id)]
    assert telegram.neutralized == []
    assert answer.replies == []
    assert await companion_counts(db) == (0, 0, 0)
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert len(fake_ai.companion_calls) == 1
    assert private not in caplog.text
    assert any(
        record.getMessage() == "Nova companion failed operation=delivery error_type=RuntimeError"
        for record in caplog.records
    )

    stale = CompanionQuery(dismiss_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=stale,
        ),
        companion_context(telegram),
    )
    assert stale.answer_attempts == 1
    assert stale.markup_removed == 0
    assert await companion_counts(db) == (0, 0, 0)
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("bounce", ["access", "context"])
async def test_exchange_prelock_cas_bounce_preserves_only_concurrent_state(
    db,
    fake_ai,
    monkeypatch,
    bounce,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    telegram = CompanionTelegram()
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    concurrent = "Безопасное более новое сообщение"
    seam_calls = 0

    async def bounce_before_lock(_fence: Any) -> None:
        nonlocal seam_calls
        seam_calls += 1
        if bounce == "access":
            await AccessService(db).block(
                user.telegram_id,
                source="companion-exchange-bounce-test",
            )
        else:
            await bot.conversation.append(
                user.telegram_id,
                user.telegram_id,
                role="user",
                content=concurrent,
                source="text",
                intent="companion_user",
            )

    monkeypatch.setattr(
        bot.conversation,
        "_before_exchange_access_lock",
        bounce_before_lock,
    )
    await deliver(
        bot,
        user,
        incoming,
        context=companion_context(telegram),
    )

    answer = sent_answer(incoming)
    assert seam_calls == 1
    assert telegram.deleted == [(user.telegram_id, answer.message_id)]
    assert telegram.neutralized == []
    assert len(incoming.replies) == 1
    assert answer.replies == []
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert len(fake_ai.companion_calls) == 1
    expected_messages = 0 if bounce == "access" else 1
    assert await companion_counts(db) == (expected_messages, 0, 0)
    if bounce == "context":
        async with db.sessions() as session:
            contents = tuple(
                await session.scalars(
                    select(ConversationMessage.content).order_by(ConversationMessage.id)
                )
            )
        assert contents == (concurrent,)
    assert bot._nova_companion_tasks == set()


async def test_safe_recent_mutation_after_provider_fences_before_delivery(
    db,
    fake_ai,
):
    fake_ai.companion_result = capture_result()
    fake_ai.companion_release.clear()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    context = companion_context()
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    route = asyncio.create_task(
        bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            context,
            incoming.text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )
    )
    await fake_ai.companion_started.wait()
    concurrent = "Новый безопасный контекст после provider"
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="user",
        content=concurrent,
        source="text",
        intent="companion_user",
    )
    fake_ai.companion_release.set()

    assert await route
    await bot._drain_nova_companion_tasks()

    assert len(fake_ai.companion_calls) == 1
    assert incoming.replies == []
    assert await companion_counts(db) == (1, 0, 0)
    async with db.sessions() as session:
        content = await session.scalar(select(ConversationMessage.content))
    assert content == concurrent
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert bot._nova_companion_tasks == set()


async def test_durable_goal_mutation_after_provider_fences_before_delivery(
    db,
    fake_ai,
):
    fake_ai.companion_result = capture_result()
    fake_ai.companion_release.clear()
    bot, user = await ready_companion_bot(db, fake_ai)
    telegram = CompanionTelegram()
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    route = asyncio.create_task(
        bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            companion_context(telegram),
            incoming.text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )
    )
    await fake_ai.companion_started.wait()
    async with db.session() as session:
        session.add(
            Goal(
                user_id=user.id,
                life_area="Здоровье",
                title="Новый durable goal",
                outcome="Спокойно пройти обследование",
                progress_criterion="Записаться к врачу",
                horizon="Месяц",
                status="active",
                priority=1,
                vision_link="Поддерживать здоровье",
            )
        )
    fake_ai.companion_release.set()

    assert await route
    await bot._drain_nova_companion_tasks()

    assert len(fake_ai.companion_calls) == 1
    assert incoming.replies == []
    assert telegram.deleted == []
    assert telegram.neutralized == []
    assert telegram.markup_edits == []
    assert await companion_counts(db) == (0, 0, 0)
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert bot._nova_companion_tasks == set()


async def test_safe_recent_mutation_after_primary_send_retires_exact_answer(
    db,
    fake_ai,
    monkeypatch,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    telegram = CompanionTelegram()
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    concurrent = "Новый безопасный контекст после primary send"
    original_check = bot._nova_companion_current_check
    check_calls = 0

    async def mutate_on_post_send(
        generation: Any,
        *,
        exchange_receipt: Any | None = None,
    ) -> str:
        nonlocal check_calls
        check_calls += 1
        if check_calls == 4:
            assert exchange_receipt is None
            await bot.conversation.append(
                user.telegram_id,
                user.telegram_id,
                role="user",
                content=concurrent,
                source="text",
                intent="companion_user",
            )
        return await original_check(
            generation,
            exchange_receipt=exchange_receipt,
        )

    monkeypatch.setattr(bot, "_nova_companion_current_check", mutate_on_post_send)
    await deliver(
        bot,
        user,
        incoming,
        context=companion_context(telegram),
    )

    answer = sent_answer(incoming)
    assert check_calls == 4
    assert telegram.deleted == [(user.telegram_id, answer.message_id)]
    assert telegram.neutralized == []
    assert len(incoming.replies) == 1
    assert answer.replies == []
    assert answer.markup_edits == []
    assert await companion_counts(db) == (1, 0, 0)
    async with db.sessions() as session:
        content = await session.scalar(select(ConversationMessage.content))
    assert content == concurrent
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_final_check_cancellation_compensates_pair_and_preserves_newer_message(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_NEWER_CONVERSATION_MESSAGE"
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    telegram = CompanionTelegram()
    context = companion_context(telegram)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    original_check = bot._nova_companion_current_check
    cancelled = False

    async def cancel_after_committed_exchange(
        generation: Any,
        *,
        exchange_receipt: Any | None = None,
    ) -> str:
        nonlocal cancelled
        if exchange_receipt is not None and not cancelled:
            cancelled = True
            await bot.conversation.append(
                user.telegram_id,
                user.telegram_id,
                role="user",
                content=private,
                source="text",
                intent="companion_user",
            )
            raise asyncio.CancelledError
        return await original_check(
            generation,
            exchange_receipt=exchange_receipt,
        )

    monkeypatch.setattr(
        bot,
        "_nova_companion_current_check",
        cancel_after_committed_exchange,
    )
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    with caplog.at_level(logging.WARNING), pytest.raises(asyncio.CancelledError):
        await bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            context,
            incoming.text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )
    await bot._drain_nova_companion_tasks()

    answer = sent_answer(incoming)
    dismiss_data = callback_by_label(answer.markup_edits[-1], "Не сейчас")
    assert cancelled
    assert telegram.deleted == [(user.telegram_id, answer.message_id)]
    assert telegram.neutralized == []
    assert len(incoming.replies) == 1
    assert answer.replies == []
    assert await companion_counts(db) == (1, 0, 0)
    async with db.sessions() as session:
        rows = tuple(
            await session.execute(
                select(ConversationMessage.content, ConversationMessage.intent).order_by(
                    ConversationMessage.id
                )
            )
        )
    assert rows == ((private, "companion_user"),)
    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}
    assert len(fake_ai.companion_calls) == 1
    assert private not in caplog.text
    assert "was never awaited" not in caplog.text
    assert "Task was destroyed" not in caplog.text

    stale = CompanionQuery(dismiss_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=stale,
        ),
        context,
    )
    assert stale.answer_attempts == 1
    assert stale.markup_removed == 0
    assert await companion_counts(db) == (1, 0, 0)
    assert bot._nova_companion_tasks == set()


async def test_direct_provider_cancellation_propagates_without_delivery_or_background_tasks(
    db,
    fake_ai,
):
    fake_ai.companion_error = asyncio.CancelledError()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage("Мне нужен спокойный разговор", chat_id=user.telegram_id)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)

    with pytest.raises(asyncio.CancelledError):
        await bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            companion_context(),
            incoming.text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )

    assert len(fake_ai.companion_calls) == 1
    assert incoming.replies == []
    assert bot._nova_companion_tasks == set()
    assert await companion_counts(db) == (0, 0, 0)


async def test_outer_cancel_during_post_send_edit_propagates_then_shielded_delivery_finishes(
    db,
    fake_ai,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = BlockingSuggestionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    telegram = CompanionTelegram()
    context = companion_context(telegram)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    update_ = companion_update(
        incoming,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    route = asyncio.create_task(
        bot.nova_companion_route(
            update_,
            context,
            incoming.text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )
    )
    await incoming.reply_started.wait()
    assert incoming.sent is not None
    await incoming.sent.markup_started.wait()
    markup = incoming.sent.markup_edits[-1]
    dismissed = callback_by_label(markup, "Не сейчас")

    route.cancel()
    with pytest.raises(asyncio.CancelledError):
        await route
    incoming.sent.markup_release.set()
    await bot._drain_nova_companion_tasks()

    assert len(incoming.replies) == 1
    assert telegram.deleted == []
    usable = CompanionQuery(dismissed, incoming.sent)
    await bot.nova_companion_callback(
        companion_update(
            incoming.sent,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=usable,
        ),
        context,
    )
    assert usable.answer_attempts == 1
    assert usable.markup_removed == 1
    assert bot._nova_companion_tasks == set()
    assert await companion_counts(db) == (2, 0, 0)


async def test_direct_cancel_from_post_send_edit_revokes_offer_and_cleans_exact_message(
    db,
    fake_ai,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CancelledSuggestionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    telegram = CompanionTelegram()
    context = companion_context(telegram)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)

    with pytest.raises(asyncio.CancelledError):
        await bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            context,
            incoming.text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )
    await bot._drain_nova_companion_tasks()

    assert incoming.sent is not None
    assert telegram.deleted == [(user.telegram_id, incoming.sent.message_id)]
    dismissed = callback_by_label(incoming.sent.markup_edits[-1], "Не сейчас")
    stale = CompanionQuery(dismissed, incoming.sent)
    await bot.nova_companion_callback(
        companion_update(
            incoming.sent,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=stale,
        ),
        context,
    )
    assert stale.answer_attempts == 1
    assert stale.markup_removed == 0
    assert bot._nova_companion_tasks == set()
    assert await companion_counts(db) == (0, 0, 0)


async def test_callback_answer_cancellation_leaves_capability_usable(db, fake_ai):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    _update, context = await deliver(bot, user, incoming)
    answer = sent_answer(incoming)
    dismiss_data = callback_by_label(answer.markup_edits[-1], "Не сейчас")
    cancelled = CompanionQuery(
        dismiss_data,
        answer,
        answer_error=asyncio.CancelledError(),
    )

    with pytest.raises(asyncio.CancelledError):
        await bot.nova_companion_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=cancelled,
            ),
            context,
        )

    retry = CompanionQuery(dismiss_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=retry,
        ),
        context,
    )

    assert cancelled.answer_attempts == retry.answer_attempts == 1
    assert retry.markup_removed == 1
    assert bot._nova_companion_tasks == set()
    assert await companion_counts(db) == (2, 0, 0)


async def test_callback_answer_bad_request_is_private_and_does_not_abort_lifecycle(
    db,
    fake_ai,
    caplog,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    _update, context = await deliver(bot, user, incoming)
    answer = sent_answer(incoming)
    dismiss_data = callback_by_label(answer.markup_edits[-1], "Не сейчас")
    private = "PRIVATE_CALLBACK_ANSWER_TEXT"
    query = CompanionQuery(
        dismiss_data,
        answer,
        answer_error=BadRequest(private),
    )

    with caplog.at_level(logging.WARNING):
        await bot.nova_companion_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=query,
            ),
            context,
        )

    assert query.answer_attempts == 1
    assert query.markup_removed == 1
    assert private not in caplog.text
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_successful_delivery_applies_exchange_state_and_memory_atomically(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    text = "Я предпочитаю короткие ответы"
    answer = "Могу предложить короткое упражнение. Какой вариант тебе ближе?"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer=answer,
        dialogue_state_update=NovaCompanionDialogueStateUpdate(
            active_topic="короткие ответы",
            last_assistant_offer="Могу предложить короткое упражнение.",
            last_assistant_offer_kinds=["exercise"],
            unresolved_question="Какой вариант тебе ближе?",
        ),
        memory_candidate=NovaCompanionMemoryCandidate(
            category="preference",
            key="response_length",
            value="short",
            evidence=text,
            salience=5,
        ),
    )
    incoming = CompanionMessage(text, chat_id=user.telegram_id)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)

    handled = await bot.nova_companion_route(
        companion_update(
            incoming,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        ),
        companion_context(),
        text,
        "text",
        user=user,
        conversation_snapshot=snapshot,
    )

    assert handled
    assert len(fake_ai.companion_calls) == 1
    assert fake_ai.companion_brain_calls[0] is not None
    assert incoming.replies[0]["text"] == answer
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2
        state = await session.scalar(select(NovaDialogueState))
        memory = await session.scalar(select(NovaObservedMemory))
        assert state is not None and state.active_topic == "короткие ответы"
        assert state.revision == 1
        assert memory is not None and memory.status == "active"
        assert memory.normalized_value == "response_length=short"
        assert memory.source_message_id > 0
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_telegram_failure_leaves_no_exchange_state_or_memory(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    text = "Я предпочитаю короткие ответы"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Поняла.",
        dialogue_state_update=NovaCompanionDialogueStateUpdate(active_topic="короткие ответы"),
        memory_candidate=NovaCompanionMemoryCandidate(
            category="preference",
            key="response_length",
            value="short",
            evidence=text,
        ),
    )
    incoming = CompanionMessage(
        text,
        chat_id=user.telegram_id,
        reply_error=RuntimeError("PRIVATE_SEND_FAILURE"),
    )

    await deliver(bot, user, incoming, context=companion_context())

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_concurrent_newer_state_invalidates_blocked_older_response(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    text = "Обсудим старую тему"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Старый ответ",
        dialogue_state_update=NovaCompanionDialogueStateUpdate(active_topic="старую тему"),
    )
    fake_ai.companion_release.clear()
    incoming = CompanionMessage(text, chat_id=user.telegram_id)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    task = asyncio.create_task(
        bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            companion_context(),
            text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )
    )
    await asyncio.wait_for(fake_ai.companion_started.wait(), timeout=5)
    async with db.session() as session:
        session.add(
            NovaDialogueState(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                access_version=user.access_version,
                active_topic="новая тема",
                last_assistant_offer_kinds=[],
                open_loops=[],
                revision=1,
                expires_at=datetime.now(UTC) + timedelta(days=1),
            )
        )
    fake_ai.companion_release.set()
    await task

    assert incoming.replies == []
    async with db.sessions() as session:
        state = await session.scalar(select(NovaDialogueState))
        assert state is not None and state.active_topic == "новая тема"
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_recall_and_exact_forget_confirmation_are_provider_free(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    async with db.session() as session:
        session.add(
            NovaObservedMemory(
                owner_id=user.id,
                category="preference",
                normalized_value="response_length=short",
                content_fingerprint="a" * 64,
                source_kind="conversation",
                source_session_id=1,
                source_message_id=1,
                source_receipt="b" * 64,
                status="active",
                salience=5,
                revision=1,
            )
        )
    recall = CompanionMessage("Что ты обо мне помнишь?", chat_id=user.telegram_id)
    recall_snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)

    assert await bot.nova_companion_route(
        companion_update(
            recall,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        ),
        companion_context(),
        recall.text or "",
        "text",
        user=user,
        conversation_snapshot=recall_snapshot,
    )
    assert "короткие ответы" in recall.replies[0]["text"]
    assert fake_ai.companion_calls == []

    forget = CompanionMessage("Забудь это", chat_id=user.telegram_id)
    forget_snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    assert await bot.nova_companion_route(
        companion_update(
            forget,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        ),
        companion_context(),
        forget.text or "",
        "text",
        user=user,
        conversation_snapshot=forget_snapshot,
    )
    canonical = forget.replies[0]["message"]
    callback = callback_by_label(forget.replies[0]["reply_markup"], "Забыть")
    query = CompanionQuery(callback, canonical)
    await bot.nova_brain_callback(
        companion_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=query,
        ),
        companion_context(),
    )

    assert query.answer_attempts == 1
    assert query.edits[-1]["text"] == "Забыла выбранную запись."
    async with db.sessions() as session:
        memory = await session.scalar(select(NovaObservedMemory))
        assert memory is not None and memory.status == "forgotten"
    replay = CompanionQuery(callback, canonical)
    await bot.nova_brain_callback(
        companion_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=replay,
        ),
        companion_context(),
    )
    assert replay.answer_attempts == 1
    assert replay.edits == []
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_recall_is_bounded_and_separates_all_current_source_layers(db, fake_ai):
    base, user = await ready_companion_bot(
        db,
        fake_ai,
        telegram_user_id=74_100,
        enable_brain=True,
    )
    settings = base.settings.model_copy(
        update={
            "enable_nova_memory": True,
            "nova_memory_admin_only": False,
            "enable_nova_memory_application": True,
            "nova_memory_application_admin_only": False,
        }
    )
    bot = FutureSelfBot(settings, db, fake_ai, NoopTranscription())
    async with db.session() as session:
        await session.execute(
            update(User)
            .where(User.id == user.id)
            .values(display_name="Назар", location_city="Казань")
        )
        session.add_all(
            [
                VisionProfile(
                    user_id=user.id,
                    raw_answers={},
                    summary="Хочу жить спокойнее и осознаннее",
                    values=["самостоятельность"],
                    desired_identity=["надёжный человек"],
                    constraints=[],
                ),
                VisionItem(
                    owner_id=user.id,
                    category="growth_creativity",
                    wish_text="написать свою книгу",
                    why_text="важно выразить идеи",
                    first_step="составить план",
                    status="active",
                ),
                Goal(
                    user_id=user.id,
                    life_area="Развитие",
                    title="подготовить рукопись",
                    outcome="готовая рукопись",
                    progress_criterion="десять глав",
                    horizon="год",
                    status="active",
                    priority=5,
                    vision_link="написать свою книгу",
                ),
                WeeklyFocus(
                    owner_id=user.id,
                    week_start=current_week_start(
                        "Europe/Moscow",
                        now=datetime.now(UTC),
                    ),
                    focus="первая глава книги",
                    approach="писать по утрам",
                    small_steps=["набросать структуру"],
                    source="text",
                ),
                NovaMemoryItem(
                    owner_id=user.id,
                    category="interaction",
                    content="предпочитает конкретные примеры",
                    content_fingerprint=sha256(
                        "предпочитает конкретные примеры".encode()
                    ).hexdigest(),
                    important=True,
                    version=1,
                ),
                NovaObservedMemory(
                    owner_id=user.id,
                    category="identity",
                    normalized_value=("identity:display_name=назар;grammatical_address=masculine"),
                    content_fingerprint=sha256(
                        "identity:display_name=назар;grammatical_address=masculine".encode()
                    ).hexdigest(),
                    source_kind="conversation",
                    source_session_id=1,
                    source_message_id=1,
                    source_receipt="c" * 64,
                    status="active",
                    salience=5,
                    revision=1,
                ),
            ]
        )
    user = await bot._user(user.telegram_id)
    incoming = CompanionMessage("Что ты обо мне помнишь?", chat_id=user.telegram_id)

    await deliver(bot, user, incoming, context=companion_context())

    answer = incoming.replies[0]["text"]
    for expected in (
        "Подтверждено в профиле",
        "имя: Назар",
        "город: Казань",
        "Анкета и профиль",
        "Текущие планы и ориентиры",
        "первая глава книги",
        "подготовить рукопись",
        "написать свою книгу",
        "Подтверждено тобой в Nova Memory",
        "предпочитает конкретные примеры",
        "Сохранено из твоих слов",
        "имя: Назар; мужское обращение",
    ):
        assert expected in answer
    assert "Ты говорила" not in answer
    assert "всё, что хранится" not in answer
    assert len(answer) <= 1_900
    assert fake_ai.companion_calls == []
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_post_apply_context_change_compensates_exact_turn(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    text = "Я предпочитаю короткие ответы"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Поняла.",
        dialogue_state_update=NovaCompanionDialogueStateUpdate(active_topic="короткие ответы"),
        memory_candidate=NovaCompanionMemoryCandidate(
            category="preference",
            key="response_length",
            value="short",
            evidence=text,
        ),
    )

    async def mutate_authoritative_context(_receipt):
        async with db.session() as session:
            await session.execute(
                update(User).where(User.id == user.id).values(display_name="Новая Лена")
            )

    bot.nova_brain_service._after_apply_commit = mutate_authoritative_context
    incoming = CompanionMessage(text, chat_id=user.telegram_id)
    context = companion_context()

    await deliver(bot, user, incoming, context=context)

    assert len(incoming.replies) == 1
    assert context.bot.deleted == [(user.telegram_id, incoming.replies[0]["message"].message_id)]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_disabled_preserves_stage_b_and_ignores_provider_proposals(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=False)
    text = "Я предпочитаю короткие ответы"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Поняла.",
        dialogue_state_update=NovaCompanionDialogueStateUpdate(active_topic="короткие ответы"),
        memory_candidate=NovaCompanionMemoryCandidate(
            category="preference",
            key="response_length",
            value="short",
            evidence=text,
        ),
    )
    incoming = CompanionMessage(text, chat_id=user.telegram_id)

    await deliver(bot, user, incoming, context=companion_context())

    assert incoming.replies[0]["text"] == "Поняла."
    assert fake_ai.companion_brain_calls == [None]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("gate", ["disabled", "tier_ineligible"])
@pytest.mark.parametrize(
    "proposal_kind",
    ["valid_memory", "invalid_memory", "sensitive_memory", "capture", "reminder"],
)
async def test_nova_brain_gate_is_strict_stage_b_fallback_for_all_provider_proposals(
    db,
    fake_ai,
    source,
    gate,
    proposal_kind,
):
    enabled = gate == "tier_ineligible"
    bot, user = await ready_companion_bot(
        db,
        fake_ai,
        telegram_user_id=72_000
        + (0 if source == "text" else 100)
        + (0 if gate == "disabled" else 10)
        + ["valid_memory", "invalid_memory", "sensitive_memory", "capture", "reminder"].index(
            proposal_kind
        ),
        enable_brain=enabled,
        brain_admin_only=enabled,
    )
    safe_answer = f"Безопасный ответ {proposal_kind}."
    if proposal_kind == "capture":
        text = "Я хочу подготовить письмо Марине и открыть документ; у меня диагноз PRIVATE"
        capture = NovaCompanionProviderCapture(
            kind="task",
            title="подготовить письмо Марине",
            next_step="открыть документ",
            evidence="подготовить письмо Марине и открыть документ",
        )
        reminder = None
        memory = NovaCompanionMemoryCandidate(
            category="fact",
            value="у меня диагноз PRIVATE",
            evidence="у меня диагноз PRIVATE",
        )
    elif proposal_kind == "reminder":
        text = "Завтра в 10:00 позвонить врачу; у меня диагноз PRIVATE"
        capture = None
        reminder = NovaCompanionProviderReminderOffer(
            title="позвонить врачу",
            schedule_wording="Завтра в 10:00",
            evidence=text,
        )
        memory = NovaCompanionMemoryCandidate(
            category="fact",
            value="у меня диагноз PRIVATE",
            evidence="у меня диагноз PRIVATE",
        )
    elif proposal_kind == "valid_memory":
        text = "Я предпочитаю короткие ответы"
        capture = None
        reminder = None
        memory = NovaCompanionMemoryCandidate(
            category="preference",
            key="response_length",
            value="short",
            evidence=text,
        )
    elif proposal_kind == "sensitive_memory":
        text = "У меня депрессия"
        capture = None
        reminder = None
        memory = NovaCompanionMemoryCandidate(
            category="fact",
            value=text,
            evidence=text,
        )
    else:
        text = "Иногда думаю о прогулке"
        capture = None
        reminder = None
        memory = NovaCompanionMemoryCandidate(
            category="fact",
            value="прогулка",
            evidence=text,
        )
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer=safe_answer,
        capture=capture,
        reminder_offer=reminder,
        dialogue_state_update=NovaCompanionDialogueStateUpdate(active_topic=text[:40]),
        memory_candidate=memory,
    )
    incoming = CompanionMessage(text, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
        context=companion_context(),
    )

    delivered = progress.edits[-1]["text"] if progress is not None else incoming.replies[0]["text"]
    assert delivered == safe_answer
    assert len(fake_ai.companion_calls) == 1
    assert fake_ai.companion_brain_calls == [None]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_rejected_private_memory_never_reaches_ui_exchange_or_logs(
    db,
    fake_ai,
    caplog,
):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    private = "PRIVATE_DIAGNOSIS_SENTINEL"
    text = f"У меня диагноз {private}"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer=f"Я запомнила {private}",
        memory_candidate=NovaCompanionMemoryCandidate(
            category="fact",
            value=f"у меня диагноз {private}",
            evidence=text,
        ),
    )
    incoming = CompanionMessage(text, chat_id=user.telegram_id)

    with caplog.at_level(logging.INFO):
        await deliver(bot, user, incoming, context=companion_context())

    assert private not in incoming.replies[0]["text"]
    assert private not in caplog.text
    async with db.sessions() as session:
        rows = list((await session.scalars(select(ConversationMessage))).all())
        assert len(rows) == 2
        assert rows[0].role == "user" and private in rows[0].content
        assert rows[1].role == "assistant" and private not in rows[1].content
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("action", ["capture", "reminder"])
async def test_nova_brain_rejected_memory_preserves_independent_safe_action_and_exchange(
    db,
    fake_ai,
    source,
    action,
):
    bot, user = await ready_companion_bot(
        db,
        fake_ai,
        telegram_user_id=73_000
        + (0 if source == "text" else 10)
        + (0 if action == "capture" else 1),
        enable_brain=True,
    )
    safe_answer = "Я рядом; действие появится только после твоего подтверждения."
    if action == "capture":
        text = "Я хочу подготовить письмо Марине и открыть документ; у меня депрессия"
        capture = NovaCompanionProviderCapture(
            kind="task",
            title="подготовить письмо Марине",
            next_step="открыть документ",
            evidence="подготовить письмо Марине и открыть документ",
        )
        reminder = None
        expected_prefix = "ncap:"
    else:
        text = "Завтра в 10:00 позвонить врачу; у меня депрессия"
        capture = None
        reminder = NovaCompanionProviderReminderOffer(
            title="позвонить врачу",
            schedule_wording="Завтра в 10:00",
            evidence=text,
        )
        expected_prefix = "nrem:"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer=safe_answer,
        capture=capture,
        reminder_offer=reminder,
        memory_candidate=NovaCompanionMemoryCandidate(
            category="fact",
            value="у меня депрессия",
            evidence="у меня депрессия",
        ),
    )
    incoming = CompanionMessage(text, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
        context=companion_context(),
    )

    canonical = progress if progress is not None else sent_answer(incoming)
    delivered = progress.edits[-1]["text"] if progress is not None else incoming.replies[0]["text"]
    assert delivered == safe_answer
    assert len(fake_ai.companion_calls) == 1
    assert canonical.markup_edits
    callbacks = [
        button.callback_data for row in canonical.markup_edits[-1].inline_keyboard for button in row
    ]
    assert any(value.startswith(expected_prefix) for value in callbacks)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert bot._nova_companion_tasks == set()


async def test_nova_brain_working_state_survives_recent_window_and_service_restart(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    async with db.session() as session:
        session.add(
            NovaDialogueState(
                owner_id=user.id,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                access_version=user.access_version,
                active_topic="подготовка к важному выступлению",
                last_assistant_offer="Можем составить простой план.",
                last_assistant_offer_kinds=["plan"],
                unresolved_question="Какой первый шаг тебе ближе?",
                open_loops=["выбрать первый шаг выступления"],
                revision=7,
                expires_at=datetime.now(UTC) + timedelta(days=5),
            )
        )
    for index in range(bot.settings.conversation_context_messages + 4):
        await bot.conversation.append(
            user.telegram_id,
            user.telegram_id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"безопасная короткая реплика {index}",
            source="text",
            intent="companion_user" if index % 2 == 0 else "companion_answer",
        )
    restarted = FutureSelfBot(bot.settings, db, fake_ai, NoopTranscription())
    restarted_user = await restarted._user(user.telegram_id)
    incoming = CompanionMessage("Что же делать?", chat_id=user.telegram_id)

    await deliver(restarted, restarted_user, incoming, context=companion_context())

    projection = fake_ai.companion_brain_calls[-1]
    assert projection is not None
    assert projection.working_state.revision == 7
    assert projection.working_state.active_topic == "подготовка к важному выступлению"
    assert projection.working_state.open_loops == ("выбрать первый шаг выступления",)


async def test_nova_brain_self_declared_identity_survives_window_and_restart_without_profile_write(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(
        db,
        fake_ai,
        telegram_user_id=74_001,
        enable_brain=True,
    )
    async with db.session() as session:
        await session.execute(update(User).where(User.id == user.id).values(display_name=None))
    user = await bot._user(user.telegram_id)
    declaration = "Меня зовут Назар. Я мужчина"
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Назар, рад знакомству. Ты готов продолжить?",
        memory_candidate=NovaCompanionMemoryCandidate(
            category="identity",
            key="identity",
            value="display_name=Назар;grammatical_address=masculine",
            evidence=declaration,
            salience=5,
        ),
    )
    first = CompanionMessage(declaration, chat_id=user.telegram_id)
    await deliver(bot, user, first, context=companion_context())

    async with db.sessions() as session:
        actor = await session.get(User, user.id)
        memory = await session.scalar(select(NovaObservedMemory))
        assert actor is not None and actor.display_name is None
        assert memory is not None
        assert memory.category == "identity"
        assert (
            memory.normalized_value == "identity:display_name=назар;grammatical_address=masculine"
        )
    for index in range(bot.settings.conversation_context_messages + 3):
        await bot.conversation.append(
            user.telegram_id,
            user.telegram_id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"безопасная промежуточная реплика {index}",
            source="text",
            intent="companion_user" if index % 2 == 0 else "companion_answer",
        )
    restarted = FutureSelfBot(bot.settings, db, fake_ai, NoopTranscription())
    restarted_user = await restarted._user(user.telegram_id)
    fake_ai.companion_provider_result = NovaCompanionProviderResponse(
        answer="Назар, ты готов выбрать следующий шаг?",
    )
    follow_up = CompanionMessage("Продолжим?", chat_id=user.telegram_id)

    await deliver(restarted, restarted_user, follow_up, context=companion_context())

    projection = fake_ai.companion_brain_calls[-1]
    assert projection is not None
    assert [item.category for item in projection.memories] == ["identity"]
    assert (
        projection.memories[0].value == "identity:display_name=назар;grammatical_address=masculine"
    )
    assert follow_up.replies[0]["text"] == "Назар, ты готов выбрать следующий шаг?"
    provider_calls = len(fake_ai.companion_calls)
    identity_question = CompanionMessage("Как меня зовут?", chat_id=user.telegram_id)
    await deliver(
        restarted,
        restarted_user,
        identity_question,
        context=companion_context(),
    )
    assert identity_question.replies[0]["text"] == "Из твоих слов: тебя зовут Назар."
    assert len(fake_ai.companion_calls) == provider_calls
    assert restarted._nova_companion_tasks == set()


async def test_nova_brain_forget_direct_cancellation_recovers_same_exact_generation(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    async with db.session() as session:
        session.add(
            NovaObservedMemory(
                owner_id=user.id,
                category="preference",
                normalized_value="response_length=short",
                content_fingerprint="c" * 64,
                source_kind="conversation",
                source_session_id=1,
                source_message_id=1,
                source_receipt="d" * 64,
                status="active",
                salience=5,
                revision=1,
            )
        )
    prompt = CompanionMessage("Забудь это", chat_id=user.telegram_id)
    await deliver(bot, user, prompt, context=companion_context())
    canonical = prompt.replies[0]["message"]
    callback = callback_by_label(prompt.replies[0]["reply_markup"], "Забыть")
    query = CompanionQuery(callback, canonical)

    async def cancel_forget(**_kwargs):
        raise asyncio.CancelledError

    bot.nova_brain_service.forget_exact = cancel_forget
    with pytest.raises(asyncio.CancelledError):
        await bot.nova_brain_callback(
            companion_update(
                canonical,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=query,
            ),
            companion_context(),
        )
    await bot._drain_nova_companion_tasks()

    assert query.answer_attempts == 1
    assert query.edits[-1]["text"] == "Забыть выбранную запись?"
    recovered = callback_by_label(query.edits[-1]["reply_markup"], "Забыть")
    assert recovered != callback
    assert bot._nova_companion_tasks == set()
    async with db.sessions() as session:
        memory = await session.scalar(select(NovaObservedMemory))
        assert memory is not None and memory.status == "active"


async def test_nova_brain_forget_forged_cross_user_chat_and_replay_are_inert(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    other = await bot._user(71_099)
    await AccessService(db).grant_subscriber(other.telegram_id, source="brain-cross-user")
    other = await bot._user(other.telegram_id)
    async with db.session() as session:
        session.add(
            NovaObservedMemory(
                owner_id=user.id,
                category="preference",
                normalized_value="response_length=short",
                content_fingerprint="e" * 64,
                source_kind="conversation",
                source_session_id=1,
                source_message_id=1,
                source_receipt="f" * 64,
                status="active",
                salience=5,
                revision=1,
            )
        )
    prompt = CompanionMessage("Забудь это", chat_id=user.telegram_id)
    await deliver(bot, user, prompt, context=companion_context())
    canonical = prompt.replies[0]["message"]
    callback = callback_by_label(prompt.replies[0]["reply_markup"], "Забыть")
    for actor_id, chat_id, data in (
        (other.telegram_id, other.telegram_id, callback),
        (user.telegram_id, user.telegram_id + 1, callback),
        (user.telegram_id, user.telegram_id, "nbrain:forged_token_value_123456"),
    ):
        query = CompanionQuery(data, canonical)
        await bot.nova_brain_callback(
            companion_update(
                canonical,
                telegram_user_id=actor_id,
                chat_id=chat_id,
                query=query,
            ),
            companion_context(),
        )
        assert query.answer_attempts == 1
        assert query.edits == []
    owner_query = CompanionQuery(callback, canonical)
    await bot.nova_brain_callback(
        companion_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=owner_query,
        ),
        companion_context(),
    )
    replay = CompanionQuery(callback, canonical)
    await bot.nova_brain_callback(
        companion_update(
            canonical,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=replay,
        ),
        companion_context(),
    )
    assert owner_query.edits[-1]["text"] == "Забыла выбранную запись."
    assert replay.edits == []


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Как меня зовут?", "Да, тебя зовут Назар."),
        ("А ты не знала, как меня зовут?", "Да, тебя зовут Назар."),
        ("Где я живу?", "Да, ты живёшь в городе Москва."),
    ],
)
async def test_confirmed_identity_questions_are_local_and_domain_read_only(
    db,
    fake_ai,
    question,
    expected,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    async with db.session() as session:
        await session.execute(
            update(User)
            .where(User.id == user.id)
            .values(display_name="Назар", location_city="Москва")
        )
    user = await bot._user(user.telegram_id)
    incoming = CompanionMessage(question, chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    assert sent_answer(incoming).text == expected
    assert fake_ai.companion_calls == []
    assert await companion_counts(db) == (0, 0, 0)


async def test_authoritative_profile_name_precedes_structured_observed_identity(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai, enable_brain=True)
    structured = "identity:display_name=назар;grammatical_address=masculine"
    async with db.session() as session:
        await session.execute(update(User).where(User.id == user.id).values(display_name="Елена"))
        session.add(
            NovaObservedMemory(
                owner_id=user.id,
                category="identity",
                normalized_value=structured,
                content_fingerprint=sha256(structured.encode()).hexdigest(),
                source_kind="conversation",
                source_session_id=1,
                source_message_id=1,
                source_receipt="7" * 64,
                status="active",
                salience=5,
                revision=1,
            )
        )
    user = await bot._user(user.telegram_id)
    incoming = CompanionMessage("Как меня зовут?", chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    assert sent_answer(incoming).text == "Да, тебя зовут Елена."
    assert fake_ai.companion_calls == []
    async with db.sessions() as session:
        memory = await session.scalar(select(NovaObservedMemory))
        assert memory is not None and memory.normalized_value == structured


async def test_task_status_never_uses_unrelated_confirmed_inbox_state(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai)
    before = CompanionMessage("Ты сохранила задачу?", chat_id=user.telegram_id)
    await deliver(bot, user, before)
    assert sent_answer(before).text == "Пока нет — задача ещё не создана."

    async with db.session() as session:
        session.add(
            InboxItem(
                user_id=user.id,
                kind="task",
                title="Позвонить врачу",
                description=None,
                raw_text="Позвонить врачу",
                next_step=None,
                resolved_date=None,
                temporal_resolution=None,
                source="text",
                status="confirmed",
                version=1,
            )
        )

    new_topic = CompanionMessage(
        "Нужно подготовить новую презентацию, пока ничего не сохраняй",
        chat_id=user.telegram_id,
    )
    await deliver(bot, user, new_topic)
    after = CompanionMessage("Ты сохранила задачу?", chat_id=user.telegram_id)
    await deliver(bot, user, after)
    assert sent_answer(after).text == "Пока нет — задача ещё не создана."
    assert len(fake_ai.companion_calls) == 1


@pytest.mark.parametrize("reminder_status", ["pending", "sent"])
async def test_reminder_status_never_uses_old_unrelated_domain_row(
    db,
    fake_ai,
    reminder_status,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    event_at = datetime(2026, 8, 22, 16, 0, tzinfo=UTC)
    async with db.session() as session:
        old = InboxItem(
            user_id=user.id,
            kind="task",
            title="Позвонить врачу",
            description=None,
            raw_text="Позвонить врачу",
            next_step=None,
            resolved_date=event_at.date(),
            temporal_resolution=None,
            source="text",
            status="confirmed",
            version=1,
        )
        session.add(old)
        await session.flush()
        session.add(
            TaskReminder(
                inbox_item_id=old.id,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                event_at=event_at,
                remind_at=event_at,
                timezone="Europe/Moscow",
                delivery_key=f"old-{reminder_status}-{user.id}",
                task_version=1,
                status=reminder_status,
            )
        )

    new_topic = CompanionMessage(
        "Завтра у меня новая стрижка в 19:00, пока ничего не создавай",
        chat_id=user.telegram_id,
    )
    await deliver(bot, user, new_topic)
    status = CompanionMessage("Оно уже создано?", chat_id=user.telegram_id)
    await deliver(bot, user, status)

    assert sent_answer(status).text == "Пока нет — напоминание ещё не создано."
    assert len(fake_ai.companion_calls) == 1


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Ты поставила напоминание?", "Пока нет — напоминание ещё не создано."),
        ("Ты создала напоминание?", "Пока нет — напоминание ещё не создано."),
        ("Напоминание готово?", "Пока нет — напоминание ещё не создано."),
        ("Готово с напоминанием?", "Пока нет — напоминание ещё не создано."),
        ("Нова, ты поставила напоминание?", "Пока нет — напоминание ещё не создано."),
        ("Nova, напоминание готово?", "Пока нет — напоминание ещё не создано."),
        ("Ты создала задачу?", "Пока нет — задача ещё не создана."),
        ("Нова, ты создала заметку?", "Пока нет — заметка ещё не создана."),
        ("Nova, ты создала идею?", "Пока нет — идея ещё не создана."),
    ],
)
async def test_natural_status_questions_without_exact_anchor_fail_closed(
    db,
    fake_ai,
    question,
    expected,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(question, chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    assert sent_answer(incoming).text == expected
    assert fake_ai.companion_calls == []
    assert await companion_counts(db) == (0, 0, 0)


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("question", ["Готово?", "Всё готово?", "Ну что, всё готово?"])
async def test_generic_ready_question_is_not_promoted_to_status_without_action_anchor(
    db,
    fake_ai,
    source,
    question,
):
    fake_ai.companion_result = NovaCompanionResponse(answer="Уточни, пожалуйста, о чём речь.")
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(question, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(bot, user, incoming, source=source, delivery_message=progress)

    delivered = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert delivered == fake_ai.companion_result.answer
    assert len(fake_ai.companion_calls) == 1
    assert await companion_counts(db) == (2, 0, 0)
    assert bot._nova_companion_status_receipts == {}


async def test_provider_fallback_with_live_action_anchor_still_blocks_elliptical_commitment(
    db,
    fake_ai,
    monkeypatch,
):
    private_claim = "Да, всё готово."
    fake_ai.companion_result = NovaCompanionResponse(answer=private_claim)
    bot, user = await ready_companion_bot(db, fake_ai)
    key = (user.id, user.telegram_id, user.telegram_id)
    receipt = object()
    bot._nova_companion_status_receipts[key] = receipt

    async def missed_status(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(bot, "_nova_companion_status_answer", missed_status)
    incoming = CompanionMessage("Всё готово?", chat_id=user.telegram_id)
    await deliver(bot, user, incoming)

    assert sent_answer(incoming).text == NOVA_COMPANION_NOT_EXECUTED_TEXT
    assert bot._nova_companion_status_receipts[key] is receipt
    async with db.sessions() as session:
        stored = tuple(await session.scalars(select(ConversationMessage.content)))
    assert private_claim not in stored
    assert len(fake_ai.companion_calls) == 1


async def test_substantive_route_exact_invalidation_preserves_concurrent_newer_receipt(
    db,
    fake_ai,
    monkeypatch,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    key = (user.id, user.telegram_id, user.telegram_id)
    old_receipt = object()
    newer_receipt = object()
    bot._nova_companion_status_receipts[key] = old_receipt
    check_started = asyncio.Event()
    check_release = asyncio.Event()

    async def blocked_status(*args, **kwargs):
        del args, kwargs
        check_started.set()
        await check_release.wait()
        return None

    monkeypatch.setattr(bot, "_nova_companion_status_answer", blocked_status)
    incoming = CompanionMessage("Обсудим новый план тренировки", chat_id=user.telegram_id)
    delivery = asyncio.create_task(deliver(bot, user, incoming))
    await asyncio.wait_for(check_started.wait(), timeout=10)
    bot._nova_companion_status_receipts[key] = newer_receipt
    check_release.set()

    await delivery

    assert bot._nova_companion_status_receipts[key] is newer_receipt
    assert len(fake_ai.companion_calls) == 1


@pytest.mark.parametrize("case", ["forged", "stale", "cross_user"])
async def test_rejected_capture_callback_never_invalidates_exact_status_receipt(
    db,
    fake_ai,
    case,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    message = CompanionMessage("Ответ", chat_id=user.telegram_id)
    first = await bot.nova_companion_captures.stage(
        CaptureSuggestion(kind="task", title="Первая задача"),
        raw_text="Первая задача",
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        access_version=user.access_version,
    )
    assert first is not None
    first_bound = await bot.nova_companion_captures.bind(
        first,
        canonical_message_id=message.message_id,
    )
    assert first_bound is not None
    stale_data = first_bound.callback_data("add")
    replacement = await bot.nova_companion_captures.stage(
        CaptureSuggestion(kind="task", title="Новая задача"),
        raw_text="Новая задача",
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        access_version=user.access_version,
    )
    assert replacement is not None
    replacement = await bot.nova_companion_captures.bind(
        replacement,
        canonical_message_id=message.message_id,
    )
    assert replacement is not None
    live_data = replacement.callback_data("add")
    receipt = object()
    key = (user.id, user.telegram_id, user.telegram_id)
    bot._nova_companion_status_receipts[key] = receipt
    callback_data = {
        "forged": "ncap:forged",
        "stale": stale_data,
        "cross_user": live_data,
    }[case]
    actor_id = user.telegram_id + 1 if case == "cross_user" else user.telegram_id
    query = CompanionQuery(callback_data, message)

    await bot.nova_companion_callback(
        companion_update(
            message,
            telegram_user_id=actor_id,
            chat_id=user.telegram_id,
            query=query,
        ),
        companion_context(),
    )

    assert query.answer_attempts == 1
    assert bot._nova_companion_status_receipts[key] is receipt
    live = await bot.nova_companion_captures.peek_bound_identity(
        live_data,
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        canonical_message_id=message.message_id,
    )
    assert live is not None


async def test_grounded_reminder_offer_survives_conversation_and_text_consent_hands_off(
    db,
    fake_ai,
):
    evidence = "Да))) не забыть бы мне завтра на стрижку)"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Тогда лучше действительно поставить напоминание. Могу помочь 🙂",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    first = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, first, context=context)
    offer_message = sent_answer(first)
    offer_markup = offer_message.markup_edits[-1]
    callbacks = [button.callback_data for row in offer_markup.inline_keyboard for button in row]
    assert len(callbacks) == 2
    assert all(callback.startswith("nrem:") for callback in callbacks)
    assert await companion_counts(db) == (2, 0, 0)

    fake_ai.companion_result = NovaCompanionResponse(answer="Стрижка остаётся текущей темой.")
    followup = CompanionMessage("Что же делать?", chat_id=user.telegram_id)
    await deliver(bot, user, followup, context=context)
    assert len(fake_ai.companion_calls) == 2

    consent = CompanionMessage("Поставь плиз", chat_id=user.telegram_id)
    await deliver(bot, user, consent, context=context)
    assert len(fake_ai.companion_calls) == 2
    assert consent.replies == []
    session = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert session is not None
    assert session.title == "стрижку"
    assert session.local_date is not None
    assert session.local_time is None
    assert session.timezone == "Europe/Moscow"
    assert session.phase is ReminderFlowPhase.TIME
    assert context.bot.neutralized[-1]["message_id"] == offer_message.message_id
    assert await companion_counts(db) == (4, 0, 0)


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_grounded_course_offer_without_schedule_enters_existing_when_phase(
    db,
    fake_ai,
    source,
):
    evidence = (
        "Мне нужно напоминать о самом главном, чтобы в суете я не забывал курс, "
        "по которому я могу стать лучше."
    )
    current = "Вот поэтому мне нужно вспоминать об этом почаще, а ты могла бы мне помогать в этом."
    fake_ai.companion_result = NovaCompanionResponse(
        answer=(
            "Понимаю: ты хочешь возвращаться к своему курсу среди потока информации. "
            "Давай настроим настоящее напоминание и выберем расписание."
        ),
        reminder_offer=NovaCompanionReminderOffer(
            title="курс, по которому я могу стать лучше",
            schedule_wording=None,
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="user",
        content=evidence,
        source="text",
        intent="companion_user",
    )
    incoming = CompanionMessage(current, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )
    context = companion_context()

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
        context=context,
    )

    canonical = sent_answer(incoming) if progress is None else progress
    callbacks = [
        button.callback_data for row in canonical.markup_edits[-1].inline_keyboard for button in row
    ]
    assert len(callbacks) == 2
    assert all(str(callback).startswith("nrem:") for callback in callbacks)
    assert len(fake_ai.companion_calls) == 1
    assert await companion_counts(db) == (3, 0, 0)

    consent = CompanionMessage("давай", chat_id=user.telegram_id)
    await deliver(bot, user, consent, context=context)

    session = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert session is not None
    assert session.phase is ReminderFlowPhase.WHEN
    assert session.title == "курс, по которому я могу стать лучше"
    assert context.bot.neutralized[-1]["text"] == "🔔 Когда напомнить?"
    labels = [
        button.text
        for row in context.bot.neutralized[-1]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert "Завтра" in labels
    assert "🔁 Каждый день" in labels
    assert len(fake_ai.companion_calls) == 1
    async with db.sessions() as session_db:
        assert await session_db.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session_db.scalar(select(func.count(InboxItem.id))) == 0
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize(
    "vague_schedule",
    ["постоянно", "почаще", "на первое время", "пока не привыкну"],
)
async def test_vague_frequency_is_never_accepted_as_a_real_schedule(
    db,
    fake_ai,
    vague_schedule,
):
    evidence = f"Хочу вспоминать о своём курсе {vague_schedule}"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Я буду постоянно напоминать — всё настроено.",
        reminder_offer=NovaCompanionReminderOffer(
            title="своём курсе",
            schedule_wording=vague_schedule,
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    answer = sent_answer(incoming)
    assert answer.text == NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT
    assert answer.markup_edits == []
    assert (
        await bot.nova_companion_reminders.active(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            access_tier=user.access_tier,
            access_version=user.access_version,
        )
        is None
    )
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert "Я буду постоянно напоминать — всё настроено." not in contents
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_weak_assent_uses_exact_immediate_offer_and_blocks_meta_question(
    db,
    fake_ai,
    source,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="user",
        content="Как не улетать в мысли постоянно?",
        source="text",
        intent="companion_user",
    )
    offer = "Если хочешь, можем дальше просто подобрать удобный способ."
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="assistant",
        content=offer,
        source="text",
        intent="companion_answer",
    )
    raw_failure = "Только уточни: что именно подобрать — вариант, план или упражнение?"
    fake_ai.companion_result = NovaCompanionResponse(answer=raw_failure)
    incoming = CompanionMessage("ооо, было бы круто)", chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(bot, user, incoming, source=source, delivery_message=progress)

    answer = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert "короткую практику" in answer
    assert "Что для меня сейчас действительно важно?" in answer
    assert raw_failure not in answer
    anchor = fake_ai.companion_discourse_calls[-1]
    assert anchor is not None
    assert anchor.status == "single"
    assert anchor.offer_kinds == ("method",)
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert raw_failure not in contents
        assert answer in contents
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert len(fake_ai.companion_calls) == 1

    fake_ai.companion_result = NovaCompanionResponse(answer="Поговорим о новой теме спокойно.")
    new_topic = CompanionMessage("А теперь поговорим о сне", chat_id=user.telegram_id)
    await deliver(bot, user, new_topic)
    assert fake_ai.companion_discourse_calls[-1] is None
    assert len(fake_ai.companion_calls) == 2
    assert bot._nova_companion_tasks == set()


async def test_two_immediate_offers_produce_one_local_clarification_without_capability(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="assistant",
        content="Могу подобрать удобный способ и могу помочь настроить напоминание.",
        source="text",
        intent="companion_answer",
    )
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Да, звучит неплохо.",
        reminder_offer=NovaCompanionReminderOffer(
            title="способ",
            evidence="давай",
        ),
    )
    incoming = CompanionMessage("давай", chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    answer = sent_answer(incoming)
    assert answer.text == NOVA_COMPANION_DISCOURSE_AMBIGUOUS_TEXT
    assert answer.markup_edits == []
    anchor = fake_ai.companion_discourse_calls[-1]
    assert anchor is not None and anchor.status == "ambiguous"
    assert (
        await bot.nova_companion_reminders.active(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            access_tier=user.access_tier,
            access_version=user.access_version,
        )
        is None
    )
    assert await companion_counts(db) == (3, 0, 0)
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_concurrent_new_topic_invalidates_inflight_discourse_anchor(
    db,
    fake_ai,
    source,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="assistant",
        content="Если хочешь, можем подобрать удобный способ.",
        source="text",
        intent="companion_answer",
    )
    private_answer = "PRIVATE_STALE_ANCHOR_ANSWER"
    fake_ai.companion_result = NovaCompanionResponse(answer=private_answer)
    fake_ai.companion_release.clear()
    incoming = CompanionMessage("давай", chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )
    context = companion_context()
    task = asyncio.create_task(
        deliver(
            bot,
            user,
            incoming,
            context=context,
            source=source,
            delivery_message=progress,
        )
    )
    await fake_ai.companion_started.wait()

    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="user",
        content="Теперь хочу поговорить о сне.",
        source="text",
        intent="companion_user",
    )
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="assistant",
        content="Давай обсудим сон.",
        source="text",
        intent="companion_answer",
    )
    fake_ai.companion_release.set()
    await task

    assert fake_ai.companion_discourse_calls[-1] is not None
    if progress is None:
        assert incoming.replies == []
    else:
        assert context.bot.deleted == [(user.telegram_id, progress.message_id)] or (
            context.bot.neutralized
            and context.bot.neutralized[-1]["text"] == NOVA_COMPANION_CONTEXT_CHANGED_TEXT
        )
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert private_answer not in contents
        assert "Теперь хочу поговорить о сне." in contents
        assert "Давай обсудим сон." in contents
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert bot._nova_companion_tasks == set()


async def test_reminder_callback_answer_delay_to_ttl_boundary_fails_closed(
    db,
    fake_ai,
    monkeypatch,
):
    issued_at = datetime(2026, 8, 21, 9, 0, tzinfo=UTC)
    clock = {"now": issued_at}
    original_utc = NovaCompanionCaptureStore._utc
    monkeypatch.setattr(
        NovaCompanionCaptureStore,
        "_utc",
        staticmethod(lambda value: original_utc(value) if value is not None else clock["now"]),
    )
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    bot.nova_companion_reminders.ttl = timedelta(seconds=1)
    context = companion_context()
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, incoming, context=context)
    answer = sent_answer(incoming)
    callback = callback_by_label(answer.markup_edits[-1], "🔔 Напомнить")
    query = BlockingAnswerQuery(callback, answer)
    callback_task = asyncio.create_task(
        bot.nova_companion_reminder_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=query,
            ),
            context,
        )
    )
    await query.answer_started.wait()
    clock["now"] = issued_at + timedelta(seconds=1)
    query.answer_release.set()
    await callback_task

    assert query.answer_attempts == 1
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_reminder_text_consent_waiting_for_store_lock_expires_locally(
    db,
    fake_ai,
    monkeypatch,
):
    issued_at = datetime(2026, 8, 21, 9, 0, tzinfo=UTC)
    clock = {"now": issued_at}
    original_utc = NovaCompanionCaptureStore._utc
    monkeypatch.setattr(
        NovaCompanionCaptureStore,
        "_utc",
        staticmethod(lambda value: original_utc(value) if value is not None else clock["now"]),
    )
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    bot.nova_companion_reminders.ttl = timedelta(seconds=1)
    context = companion_context()
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, incoming, context=context)

    await bot.nova_companion_reminders._lock.acquire()
    consent = CompanionMessage("Поставь плиз", chat_id=user.telegram_id)
    consent_task = asyncio.create_task(deliver(bot, user, consent, context=context))
    await asyncio.sleep(0)
    assert consent_task.done() is False
    clock["now"] = issued_at + timedelta(seconds=1)
    bot.nova_companion_reminders._lock.release()
    await consent_task

    assert sent_answer(consent).text == NOVA_COMPANION_NO_ACTIVE_REMINDER_OFFER_TEXT
    assert len(fake_ai.companion_calls) == 1
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize(
    ("title", "schedule", "evidence"),
    [
        ("врача", None, "стрижка завтра в 19:00"),
    ],
)
async def test_rejected_reminder_offer_never_delivers_or_persists_dependent_answer(
    db,
    fake_ai,
    title,
    schedule,
    evidence,
):
    raw_answer = "Могу поставить напоминание — нажми кнопку. PRIVATE_REJECTED_OFFER"
    fake_ai.companion_result = NovaCompanionResponse(
        answer=raw_answer,
        reminder_offer=NovaCompanionReminderOffer(
            title=title,
            schedule_wording=schedule,
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage("А вдруг забуду?", chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    answer = sent_answer(incoming)
    assert answer.text == NOVA_COMPANION_REJECTED_REMINDER_OFFER_TEXT
    assert answer.markup_edits == []
    assert raw_answer not in answer.text
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert raw_answer not in contents
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert (
        await bot.nova_companion_reminders.active(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            access_tier=user.access_tier,
            access_version=user.access_version,
        )
        is None
    )
    assert len(fake_ai.companion_calls) == 1


async def test_reminder_offer_callback_answers_once_and_creates_no_domain_row(db, fake_ai):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, incoming, context=context)
    answer = sent_answer(incoming)
    callback = callback_by_label(answer.markup_edits[-1], "🔔 Напомнить")
    offer_status = CompanionMessage("Оно уже создано?", chat_id=user.telegram_id)
    await deliver(bot, user, offer_status, context=context)
    assert sent_answer(offer_status).text == "Пока нет — напоминание ещё не создано."
    query = CompanionQuery(callback, answer)
    update_value = companion_update(
        answer,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        query=query,
    )

    await bot.nova_companion_reminder_callback(update_value, context)

    assert query.answer_attempts == 1
    assert len(fake_ai.companion_calls) == 1
    reminder_session = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert reminder_session is not None
    assert reminder_session.phase is ReminderFlowPhase.PREVIEW
    session_status = CompanionMessage("Оно уже создано?", chat_id=user.telegram_id)
    await deliver(bot, user, session_status, context=context)
    assert sent_answer(session_status).text == "Пока нет — напоминание ещё не создано."
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0

    reminder_markup = context.bot.neutralized[-1]["reply_markup"]
    confirm_callback = reminder_markup.inline_keyboard[0][0].callback_data
    assert isinstance(confirm_callback, str) and confirm_callback.startswith("rmd:")
    confirm_query = CompanionQuery(confirm_callback, answer)
    confirm_update = companion_update(
        answer,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        query=confirm_query,
    )
    await bot.reminder_callback(confirm_update, context)
    assert confirm_query.answer_attempts == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 1
        assert await session.scalar(select(func.count(InboxItem.id))) == 1

    for status_question in (
        "Оно уже создано?",
        "Ты поставила напоминание?",
        "Ты создала напоминание?",
        "Напоминание готово?",
        "Готово с напоминанием?",
        "Нова, ты поставила напоминание?",
        "Nova, напоминание готово?",
        "Готово?",
        "Всё готово?",
        "Ну что, всё готово?",
    ):
        status = CompanionMessage(status_question, chat_id=user.telegram_id)
        await deliver(bot, user, status, context=context)
        assert sent_answer(status).text.startswith("Да, напоминание создано на ")
        assert "19:00" in sent_answer(status).text
    assert len(fake_ai.companion_calls) == 1

    direct = CompanionMessage(
        "Напомни послезавтра в 20:00 купить молоко",
        chat_id=user.telegram_id,
    )
    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_text_gate(
            companion_update(
                direct,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            context,
        )
    replacement = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert replacement is not None and replacement.title.endswith("купить молоко")
    assert await bot.reminder_sessions.clear_exact(replacement) is True
    stale_status = CompanionMessage("Оно уже создано?", chat_id=user.telegram_id)
    await deliver(bot, user, stale_status, context=context)
    assert sent_answer(stale_status).text == "Пока нет — напоминание ещё не создано."
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_reminder_offer_callback_answer_error_continues_and_cancellation_keeps_token(
    db,
    fake_ai,
):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, incoming, context=context)
    answer = sent_answer(incoming)
    callback = callback_by_label(answer.markup_edits[-1], "🔔 Напомнить")

    cancelled = CompanionQuery(callback, answer, answer_error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await bot.nova_companion_reminder_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=cancelled,
            ),
            context,
        )
    assert cancelled.answer_attempts == 1
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )

    retry = CompanionQuery(callback, answer, answer_error=BadRequest("PRIVATE_ANSWER"))
    await bot.nova_companion_reminder_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=retry,
        ),
        context,
    )
    assert retry.answer_attempts == 1
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is not None
    )
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_reminder_offer_callback_access_replacement_is_neutral_and_domain_read_only(
    db,
    fake_ai,
):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, incoming, context=context)
    answer = sent_answer(incoming)
    callback = callback_by_label(answer.markup_edits[-1], "🔔 Напомнить")
    await AccessService(db).block(user.telegram_id, source="nrem-access-test")

    query = CompanionQuery(callback, answer)
    await bot.nova_companion_reminder_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=query,
        ),
        context,
    )

    assert query.answer_attempts == 1
    assert context.bot.neutralized[-1]["text"] == NOVA_COMPANION_ACCESS_CHANGED_TEXT
    assert context.bot.neutralized[-1]["reply_markup"] is None
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_decline_and_unanchored_consent_never_create_reminder(db, fake_ai):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, incoming, context=context)
    offer = sent_answer(incoming)

    decline = CompanionMessage("Не сейчас", chat_id=user.telegram_id)
    await deliver(bot, user, decline, context=context)
    assert len(fake_ai.companion_calls) == 1
    assert decline.replies == []
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )

    fake_ai.companion_result = NovaCompanionResponse(answer="Уточни, что именно ты хочешь.")
    unanchored = CompanionMessage("Да", chat_id=user.telegram_id)
    await deliver(bot, user, unanchored, context=context)
    assert sent_answer(unanchored).text == "Уточни, что именно ты хочешь."
    assert len(fake_ai.companion_calls) == 2
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert offer.markup_edits


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("reply", ["Да", "Давай", "Нет", "Не сейчас", "Не надо"])
async def test_weak_unanchored_reply_continues_ordinary_companion_dialogue(
    db,
    fake_ai,
    source,
    reply,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    fake_ai.companion_result = NovaCompanionResponse(answer="Хочешь рассказать подробнее?")
    opening = CompanionMessage("Мне сегодня было непросто", chat_id=user.telegram_id)
    await deliver(bot, user, opening, context=context)
    fake_ai.companion_result = NovaCompanionResponse(answer=f"Продолжаем разговор после: {reply}")
    incoming = CompanionMessage(reply, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        context=context,
        source=source,
        delivery_message=progress,
    )

    answer = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert answer == f"Продолжаем разговор после: {reply}"
    assert len(fake_ai.companion_calls) == 2
    assert await companion_counts(db) == (4, 0, 0)


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize(
    ("reply", "expected_session"),
    [
        ("Да", True),
        ("Давай", True),
        ("Нет", False),
        ("Не сейчас", False),
        ("Не надо", False),
    ],
)
async def test_weak_reply_with_active_offer_keeps_exact_consent_semantics(
    db,
    fake_ai,
    source,
    reply,
    expected_session,
):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    opening = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, opening, context=context)
    incoming = CompanionMessage(reply, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        context=context,
        source=source,
        delivery_message=progress,
    )

    current = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert (current is not None) is expected_session
    assert len(fake_ai.companion_calls) == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize(
    "command",
    ["Поставь", "Поставь плиз", "Напомни", "Сделай напоминание"],
)
async def test_strong_unanchored_reminder_command_is_local_and_domain_read_only(
    db,
    fake_ai,
    source,
    command,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(command, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
    )

    answer = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert answer == NOVA_COMPANION_NO_ACTIVE_REMINDER_OFFER_TEXT
    assert fake_ai.companion_calls == []
    assert await companion_counts(db) == (0, 0, 0)
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("terminal", ["expired", "replayed"])
async def test_weak_reply_never_revives_expired_or_replayed_reminder_offer(
    db,
    fake_ai,
    monkeypatch,
    source,
    terminal,
):
    issued_at = datetime(2026, 8, 21, 9, 0, tzinfo=UTC)
    clock = {"now": issued_at}
    original_utc = NovaCompanionCaptureStore._utc
    monkeypatch.setattr(
        NovaCompanionCaptureStore,
        "_utc",
        staticmethod(lambda value: original_utc(value) if value is not None else clock["now"]),
    )
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    bot.nova_companion_reminders.ttl = timedelta(seconds=1)
    context = companion_context()
    opening = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, opening, context=context)

    if terminal == "expired":
        clock["now"] = issued_at + timedelta(seconds=1)
    else:
        decline = CompanionMessage("Не сейчас", chat_id=user.telegram_id)
        await deliver(bot, user, decline, context=context)

    fake_ai.companion_result = NovaCompanionResponse(answer="Это обычное продолжение разговора.")
    incoming = CompanionMessage("Да", chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )
    await deliver(
        bot,
        user,
        incoming,
        context=context,
        source=source,
        delivery_message=progress,
    )

    answer = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert answer == "Это обычное продолжение разговора."
    assert len(fake_ai.companion_calls) == 2
    assert (
        await bot.reminder_sessions.current(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert bot._nova_companion_tasks == set()


async def test_reminder_offer_markup_failure_revokes_capability_and_exact_answer(
    db,
    fake_ai,
    monkeypatch,
):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)

    async def fail_markup(*_args: Any, **_kwargs: Any) -> None:
        raise BadRequest("PRIVATE_REMINDER_MARKUP")

    monkeypatch.setattr(bot, "_nova_companion_edit_markup", fail_markup)
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    _update, context = await deliver(bot, user, incoming)

    answer = sent_answer(incoming)
    assert context.bot.deleted == [(user.telegram_id, answer.message_id)]
    assert (
        await bot.nova_companion_reminders.active(
            owner_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            access_tier=user.access_tier,
            access_version=user.access_version,
        )
        is None
    )
    assert await companion_counts(db) == (0, 0, 0)
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_cancelled_reminder_handoff_preserves_exact_replacement_session(
    db,
    fake_ai,
    monkeypatch,
):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer="Могу предложить настоящее напоминание.",
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    context = companion_context()
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    await deliver(bot, user, incoming, context=context)
    answer = sent_answer(incoming)
    callback = callback_by_label(answer.markup_edits[-1], "🔔 Напомнить")
    edit_started = asyncio.Event()
    release_edit = asyncio.Event()

    async def cancel_edit(**_kwargs: Any) -> None:
        edit_started.set()
        await release_edit.wait()
        raise asyncio.CancelledError()

    monkeypatch.setattr(context.bot, "edit_message_text", cancel_edit)
    query = CompanionQuery(callback, answer)
    callback_task = asyncio.create_task(
        bot.nova_companion_reminder_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=query,
            ),
            context,
        )
    )
    await edit_started.wait()
    old = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert old is not None
    replacement = await bot.reminder_sessions.create(
        owner_id=old.owner_id,
        telegram_user_id=old.telegram_user_id,
        chat_id=old.chat_id,
        access_version=old.access_version,
        title="независимое напоминание",
        schedule_kind=old.schedule_kind,
        local_date=old.local_date,
        local_time=old.local_time,
        timezone=old.timezone,
        timezone_source=old.timezone_source,
        phase=old.phase,
        canonical_message_id=old.canonical_message_id,
        profile_timezone=old.profile_timezone,
    )
    release_edit.set()
    with pytest.raises(asyncio.CancelledError):
        await callback_task
    await bot._drain_nova_companion_tasks()

    current = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert current == replacement
    assert query.answer_attempts == 1
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize(
    "private_claim",
    [
        "Я уже поставила напоминание и напомню завтра.",
        "Я держу это в контексте как напоминание.",
        "Я запомнила и обязательно вернусь к этому завтра.",
        "Не переживай, напоминание у меня в голове.",
        "Готово — стрижка отмечена на завтра.",
        "Я не забуду про твою стрижку завтра.",
        "Я точно не забуду о стрижке завтра.",
        "Считай, что напоминание готово.",
        "Считай, напоминание уже готово.",
        "Стрижка уже у меня на контроле.",
        "Стрижку держу на контроле до завтра.",
        "Буду держать это в уме до завтра.",
        "Обязательно буду держать это в уме.",
        "Напоминание готово.",
        "Готово.",
        "Стрижка у меня под контролем.",
        "Возьму стрижку на контроль.",
        "Буду держать это в голове до завтра.",
        "Я буду помнить про стрижку завтра.",
        "Считай, всё готово.",
        "Я прослежу, чтобы ты не забыл.",
        "Да, всё готово.",
        "Конечно, готово.",
        "Готово — напоминание на завтра.",
        "Напоминание установлено.",
        "Стрижка отмечена на завтра.",
        "Я всё отметила.",
        "Я учла стрижку на завтра.",
        "Я зафиксировала это.",
        "Буду иметь это в виду до завтра.",
        "Можешь на меня рассчитывать.",
        "Буду постоянно напоминать.",
        "Буду держать это в поле внимания.",
        "Не дам тебе забыть.",
        "Возьму это на контроль.",
        "Считай, это под контролем.",
        "Напоминание настроено.",
        "Буду возвращать тебя к главному.",
    ],
)
async def test_untrusted_operational_claim_is_never_delivered_or_persisted(
    db,
    fake_ai,
    source,
    private_claim,
):
    fake_ai.companion_result = NovaCompanionResponse(answer=private_claim)
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage("А вдруг забуду?", chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
    )

    answer = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert answer == NOVA_COMPANION_NOT_EXECUTED_TEXT
    async with db.sessions() as session:
        stored = tuple(
            await session.scalars(
                select(ConversationMessage.content).order_by(ConversationMessage.id)
            )
        )
    assert private_claim not in stored
    assert stored[-1] == answer
    assert len(fake_ai.companion_calls) == 1
    assert await companion_counts(db) == (2, 0, 0)


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize(
    ("question", "grounded_answer", "prior"),
    [
        (
            "Ты помнишь, как меня зовут?",
            "Да, я помню, тебя зовут Назар.",
            None,
        ),
        (
            "Что ты помнишь о моей энергии?",
            "Я помню из нашего разговора, что ты хочешь больше энергии.",
            "Я хочу больше энергии.",
        ),
        (
            "Какой стиль ответа я просил?",
            "Помню, ты просил меня быть краткой.",
            "Пожалуйста, отвечай кратко.",
        ),
        (
            "Ты помнишь, что завтра у меня?",
            "Помню, ты говорил, что завтра у тебя стрижка; но напоминание пока не создано.",
            "Завтра у меня стрижка.",
        ),
        (
            "Ты помнишь про мою стрижку?",
            "Да, я помню: завтра у тебя стрижка. Если хочешь, поставим напоминание.",
            "Завтра у меня стрижка.",
        ),
        (
            "Что мне взять с собой?",
            "Не забудь завтра взять паспорт.",
            "Завтра мне нужен паспорт.",
        ),
        (
            "Кто создал задачу?",
            "Ты уже создала задачу сама — я только помогла её сформулировать.",
            "Я сама создала задачу.",
        ),
        (
            "Что ты помнишь о моём питании?",
            "Помню, ты держишь питание под контролем.",
            "Я держу питание под контролем.",
        ),
        (
            "Что было записано вчера?",
            "Ты записала это вчера в блокнот.",
            "Я записала это вчера в блокнот.",
        ),
    ],
)
async def test_grounded_conversational_memory_claim_is_delivered_and_persisted(
    db,
    fake_ai,
    source,
    question,
    grounded_answer,
    prior,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    async with db.session() as session:
        await session.execute(update(User).where(User.id == user.id).values(display_name="Назар"))
    user = await bot._user(user.telegram_id)
    if prior is not None:
        await bot.conversation.append(
            user.telegram_id,
            user.telegram_id,
            role="user",
            content=prior,
            source="text",
            intent="companion_user",
            topic=None,
        )
    fake_ai.companion_result = NovaCompanionResponse(answer=grounded_answer)
    incoming = CompanionMessage(question, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
    )

    answer = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert answer == grounded_answer
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert grounded_answer in contents
        assert await session.scalar(select(func.count(DraftInboxItem.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
    assert len(fake_ai.companion_calls) == 1


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize(
    "private_claim",
    [
        "Напоминание готово.",
        "Готово.",
        "Стрижка у меня под контролем.",
        "Возьму стрижку на контроль.",
        "Буду держать это в голове до завтра.",
        "Я буду помнить про стрижку завтра.",
        "Считай, всё готово.",
        "Я прослежу, чтобы ты не забыл.",
        "Да, всё готово.",
        "Конечно, готово.",
        "Готово — напоминание на завтра.",
        "Напоминание установлено.",
        "Стрижка отмечена на завтра.",
        "Я всё отметила.",
        "Я учла стрижку на завтра.",
        "Я зафиксировала это.",
        "Буду иметь это в виду до завтра.",
        "Можешь на меня рассчитывать.",
        "Буду постоянно напоминать.",
        "Буду держать это в поле внимания.",
        "Не дам тебе забыть.",
        "Возьму это на контроль.",
        "Считай, это под контролем.",
        "Напоминание настроено.",
        "Буду возвращать тебя к главному.",
    ],
)
async def test_untrusted_claim_with_valid_offer_keeps_only_real_offer_action(
    db,
    fake_ai,
    source,
    private_claim,
):
    evidence = "Боюсь забыть стрижку завтра в 19:00"
    fake_ai.companion_result = NovaCompanionResponse(
        answer=private_claim,
        reminder_offer=NovaCompanionReminderOffer(
            title="стрижку",
            schedule_wording="завтра в 19:00",
            evidence=evidence,
        ),
    )
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(evidence, chat_id=user.telegram_id)
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
    )

    answer = sent_answer(incoming).text if progress is None else progress.edits[-1]["text"]
    assert answer == NOVA_COMPANION_REMINDER_OFFER_ACTION_TEXT
    assert private_claim != answer
    live = await bot.nova_companion_reminders.active(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        access_tier=user.access_tier,
        access_version=user.access_version,
    )
    assert live is not None
    async with db.sessions() as session:
        contents = tuple(await session.scalars(select(ConversationMessage.content)))
        assert private_claim not in contents
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(InboxItem.id))) == 0
    assert len(fake_ai.companion_calls) == 1


async def test_add_access_bounce_compensates_only_created_draft_and_neutralizes_consumed_screen(
    db,
    fake_ai,
    monkeypatch,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    _update, context = await deliver(bot, user, incoming)
    answer = sent_answer(incoming)
    add_data = callback_by_label(answer.markup_edits[-1], "Добавить как задачу")
    # Callback-level check, add-level check, then the post-create fence.
    checks = iter((True, True, False))

    async def actor_check(_user: User, _capability: Any) -> bool:
        return next(checks)

    monkeypatch.setattr(bot, "_nova_companion_capability_actor_is_current", actor_check)
    query = CompanionQuery(add_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=query,
        ),
        context,
    )

    assert query.answer_attempts == 1
    assert query.edits[-1]["text"] == NOVA_COMPANION_ACCESS_CHANGED_TEXT
    assert await active_companion_drafts(db) == []
    assert await companion_counts(db) == (2, 1, 0)
    async with db.sessions() as session:
        status = await session.scalar(select(DraftInboxItem.status))
    assert status == "discarded"
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_pending_capture_status == {}
    assert bot._nova_companion_tasks == set()


async def test_vocative_explicit_reminder_keeps_existing_reminder_flow(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Нова, напомни завтра в 10:00 позвонить врачу",
        chat_id=user.telegram_id,
    )

    await deliver(bot, user, incoming)

    reminder = await bot.reminder_sessions.current(
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
    )
    assert reminder is not None
    assert fake_ai.companion_calls == []
    assert await companion_counts(db) == (0, 0, 0)


async def test_explicit_context_capture_uses_only_latest_safe_companion_message(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai)
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="user",
        content="Старая безопасная мысль не должна быть выбрана",
        source="text",
        intent="companion_user",
    )
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="user",
        content="PRIVATE_HEALTH_SENTINEL с чувствительными подробностями",
        source="text",
        intent="health_answer",
    )
    incoming = CompanionMessage("Нова, сохрани это.", chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    assert fake_ai.companion_calls == []
    assert await active_companion_drafts(db) == []
    assert len(incoming.replies) == 1
    assert "уточни" in str(incoming.replies[0]["text"]).casefold()
    assert "PRIVATE_" not in repr(incoming.replies)


async def test_explicit_context_capture_opens_existing_preview_without_companion_provider(
    db,
    fake_ai,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    await bot.conversation.append(
        user.telegram_id,
        user.telegram_id,
        role="user",
        content="Подготовить короткий план разговора с Мариной",
        source="text",
        intent="companion_user",
    )
    incoming = CompanionMessage("Запиши это как заметку.", chat_id=user.telegram_id)

    await deliver(bot, user, incoming)

    drafts = await active_companion_drafts(db)
    assert len(drafts) == 1
    assert drafts[0].title == "Подготовить короткий план разговора с Мариной"
    assert fake_ai.companion_calls == []
    assert await companion_counts(db) == (3, 1, 0)


@pytest.mark.parametrize(
    ("checkpoint", "expected_text", "expected_provider_calls"),
    [
        ("memory_snapshot", NOVA_COMPANION_UNAVAILABLE_TEXT, 0),
        ("context_snapshot", NOVA_COMPANION_UNAVAILABLE_TEXT, 0),
        ("pre_provider_fence", NOVA_COMPANION_ACCESS_CHANGED_TEXT, 0),
        ("post_provider_fence", NOVA_COMPANION_CONTEXT_CHANGED_TEXT, 1),
    ],
)
async def test_voice_early_outcome_neutralizes_exact_stt_progress_without_fallback(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    checkpoint,
    expected_text,
    expected_provider_calls,
):
    private = "PRIVATE_VOICE_EARLY_FAILURE_TEXT"
    bot, user = await ready_companion_bot(db, fake_ai)
    if checkpoint == "memory_snapshot":

        async def memory_failure(_user: User) -> tuple[str, None, None]:
            return "unavailable", None, None

        monkeypatch.setattr(bot, "_nova_companion_memory_projection", memory_failure)
    elif checkpoint == "context_snapshot":

        async def context_failure(**_kwargs: Any) -> None:
            raise RuntimeError(private)

        monkeypatch.setattr(bot.nova_companion_context, "snapshot", context_failure)
    else:
        outcomes = iter(
            ("access_changed",)
            if checkpoint == "pre_provider_fence"
            else ("ready", "context_changed")
        )

        async def current_check(_generation: Any) -> str:
            return next(outcomes)

        monkeypatch.setattr(bot, "_nova_companion_current_check", current_check)

    incoming = CompanionMessage(private, chat_id=user.telegram_id)
    progress = CompanionMessage("Расшифровываю…", chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    context = companion_context(telegram)

    with caplog.at_level(logging.WARNING):
        await deliver(
            bot,
            user,
            incoming,
            context=context,
            source="voice",
            delivery_message=progress,
        )

    assert len(fake_ai.companion_calls) == expected_provider_calls
    assert incoming.replies == []
    assert progress.edits == []
    assert telegram.deleted == []
    assert telegram.neutralized == [
        {
            "chat_id": user.telegram_id,
            "message_id": progress.message_id,
            "text": expected_text,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    assert private not in caplog.text
    assert bot._nova_companion_tasks == set()
    assert await companion_counts(db) == (0, 0, 0)


async def test_voice_outer_cancellation_during_provider_tracks_exact_progress_cleanup(
    db,
    fake_ai,
    caplog,
):
    private = "PRIVATE_CANCELLED_VOICE_TEXT"
    fake_ai.companion_release.clear()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(private, chat_id=user.telegram_id)
    progress = CompanionMessage("Расшифровываю…", chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    context = companion_context(telegram)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    route = asyncio.create_task(
        bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            context,
            private,
            "voice",
            user=user,
            conversation_snapshot=snapshot,
            delivery_message=progress,
        )
    )
    await fake_ai.companion_started.wait()

    with caplog.at_level(logging.WARNING):
        route.cancel()
        with pytest.raises(asyncio.CancelledError):
            await route
        await bot._drain_nova_companion_tasks()

    assert len(fake_ai.companion_calls) == 1
    assert incoming.replies == []
    assert telegram.deleted == []
    assert telegram.neutralized == [
        {
            "chat_id": user.telegram_id,
            "message_id": progress.message_id,
            "text": NOVA_COMPANION_UNAVAILABLE_TEXT,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    assert private not in caplog.text
    assert bot._nova_companion_tasks == set()
    assert await companion_counts(db) == (0, 0, 0)


@pytest.mark.parametrize("failed_check", [2, 4, 7])
async def test_explicit_capture_access_fence_compensates_exact_lifecycle(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    failed_check,
):
    private = "PRIVATE_EXPLICIT_ACCESS_SENTINEL"
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        f"Создай задачу позвонить врачу {private}",
        chat_id=user.telegram_id,
    )
    telegram = CompanionTelegram()
    context = companion_context(telegram)
    checks = 0

    async def actor_check(_user: User) -> bool:
        nonlocal checks
        checks += 1
        return checks != failed_check

    monkeypatch.setattr(bot, "_nova_companion_actor_is_current", actor_check)
    with caplog.at_level(logging.WARNING):
        await deliver(bot, user, incoming, context=context)

    assert fake_ai.companion_calls == []
    assert await active_companion_drafts(db) == []
    assert (await companion_counts(db))[2] == 0
    assert len(incoming.replies) == (1 if failed_check >= 4 else 0)
    if incoming.replies:
        preview = incoming.replies[0]["message"]
        assert telegram.deleted == [(user.telegram_id, preview.message_id)]
    else:
        assert telegram.deleted == []
    async with db.sessions() as session:
        conversation = await session.scalar(select(ConversationSession))
        if conversation is not None:
            assert conversation.active_draft_id is None
            assert conversation.focused_draft_id is None
    assert private not in caplog.text
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("cancellation", ["direct", "outer"])
async def test_explicit_capture_post_send_cancellation_tracks_exact_cleanup(
    db,
    fake_ai,
    monkeypatch,
    cancellation,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Создай задачу позвонить врачу",
        chat_id=user.telegram_id,
    )
    telegram = CompanionTelegram()
    context = companion_context(telegram)
    started = asyncio.Event()

    if cancellation == "direct":

        async def cancel_pointer(*_args: Any, **_kwargs: Any) -> bool:
            started.set()
            raise asyncio.CancelledError

    else:

        async def cancel_pointer(*_args: Any, **_kwargs: Any) -> bool:
            started.set()
            await asyncio.Event().wait()
            return True

    monkeypatch.setattr(
        bot.draft_service,
        "restore_preview_message_if_current",
        cancel_pointer,
    )
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    route = asyncio.create_task(
        bot.nova_companion_route(
            companion_update(
                incoming,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
            ),
            context,
            incoming.text,
            "text",
            user=user,
            conversation_snapshot=snapshot,
        )
    )
    await started.wait()
    if cancellation == "outer":
        route.cancel()
    with pytest.raises(asyncio.CancelledError):
        await route
    await bot._drain_nova_companion_tasks()

    assert len(incoming.replies) == 1
    preview = incoming.replies[0]["message"]
    assert telegram.deleted == [(user.telegram_id, preview.message_id)]
    assert await active_companion_drafts(db) == []
    assert await companion_counts(db) == (0, 1, 0)
    assert fake_ai.companion_calls == []
    assert bot._nova_companion_tasks == set()


async def test_explicit_reused_draft_failure_restores_original_pointer_and_focus(
    db,
    fake_ai,
    monkeypatch,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    first = CompanionMessage("Создай задачу позвонить врачу", chat_id=user.telegram_id)
    await deliver(bot, user, first)
    original_preview = first.replies[0]["message"]
    async with db.sessions() as session:
        original_draft = await session.scalar(select(DraftInboxItem))
        assert original_draft is not None
        original_draft_id = original_draft.id

    checks = 0

    async def actor_check(_user: User) -> bool:
        nonlocal checks
        checks += 1
        return checks != 4

    monkeypatch.setattr(bot, "_nova_companion_actor_is_current", actor_check)
    second = CompanionMessage("Создай задачу позвонить врачу", chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    await deliver(bot, user, second, context=companion_context(telegram))

    replacement_preview = second.replies[0]["message"]
    assert telegram.deleted == [(user.telegram_id, replacement_preview.message_id)]
    async with db.sessions() as session:
        draft = await session.get(DraftInboxItem, original_draft_id)
        conversation = await session.scalar(select(ConversationSession))
        assert draft is not None and draft.status == "preview"
        assert draft.preview_message_id == original_preview.message_id
        assert conversation is not None
        assert conversation.active_draft_id == original_draft_id
        assert conversation.focused_draft_id == original_draft_id
    assert len(await active_companion_drafts(db)) == 1
    assert fake_ai.companion_calls == []
    assert bot._nova_companion_tasks == set()


async def test_explicit_voice_post_send_access_bounce_retires_preview_and_progress(
    db,
    fake_ai,
    monkeypatch,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage("Создай задачу позвонить врачу", chat_id=user.telegram_id)
    progress = CompanionMessage("Распознаю…", chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    checks = 0

    async def actor_check(_user: User) -> bool:
        nonlocal checks
        checks += 1
        return checks != 4

    monkeypatch.setattr(bot, "_nova_companion_actor_is_current", actor_check)
    await deliver(
        bot,
        user,
        incoming,
        context=companion_context(telegram),
        source="voice",
        delivery_message=progress,
    )

    assert incoming.replies == []
    assert len(progress.replies) == 1
    preview = progress.replies[0]["message"]
    assert telegram.deleted == [
        (user.telegram_id, preview.message_id),
        (user.telegram_id, progress.message_id),
    ]
    assert await active_companion_drafts(db) == []
    assert fake_ai.companion_calls == []
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("source", ["text", "voice"])
async def test_dated_explicit_task_preserves_legacy_temporal_resolution(
    db,
    fake_ai,
    source,
):
    fixed_now = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)
    bot, user = await ready_companion_bot(db, fake_ai)
    bot.date_resolver = DateResolver(lambda: fixed_now)
    incoming = CompanionMessage(
        "Создай задачу завтра в 10:00 позвонить врачу",
        chat_id=user.telegram_id,
    )
    progress = (
        CompanionMessage("Распознаю…", chat_id=user.telegram_id) if source == "voice" else None
    )

    await deliver(
        bot,
        user,
        incoming,
        source=source,
        delivery_message=progress,
    )

    drafts = await active_companion_drafts(db)
    assert len(drafts) == 1
    assert drafts[0].resolved_date == (fixed_now + timedelta(days=1)).date()
    assert drafts[0].temporal_resolution is not None
    assert drafts[0].temporal_resolution["resolved_local_time"] == "10:00:00"
    assert drafts[0].temporal_resolution["timezone"] == user.timezone
    assert fake_ai.companion_calls == []


async def test_callback_reused_access_failure_restores_pointer_and_clears_old_focus(
    db,
    fake_ai,
    monkeypatch,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    raw_text = "Я хочу подготовить письмо Марине и открыть документ"
    incoming = CompanionMessage(raw_text, chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    _update, context = await deliver(
        bot,
        user,
        incoming,
        context=companion_context(telegram),
    )
    answer = sent_answer(incoming)
    add_data = callback_by_label(answer.markup_edits[-1], "Добавить как задачу")
    seeded = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="companion",
        raw_text=raw_text,
        parsed=ParsedThought(
            kind="task",
            title="подготовить письмо Марине",
            next_step="открыть документ",
        ),
    )
    assert seeded.draft is not None
    old_pointer = 990_001
    await bot.draft_service.set_preview_message(seeded.draft.id, old_pointer)
    await bot.conversation.set_active_draft(
        user.telegram_id,
        user.telegram_id,
        seeded.draft.id,
    )
    checks = 0

    async def actor_check(_user: User, _capability: Any) -> bool:
        nonlocal checks
        checks += 1
        return checks != 7

    monkeypatch.setattr(bot, "_nova_companion_capability_actor_is_current", actor_check)
    query = CompanionQuery(add_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=query,
        ),
        context,
    )

    assert len(answer.replies) == 1
    transient_preview = answer.replies[0]["message"]
    assert telegram.deleted == [(user.telegram_id, transient_preview.message_id)]
    async with db.sessions() as session:
        draft = await session.get(DraftInboxItem, seeded.draft.id)
        conversation = await session.scalar(select(ConversationSession))
        assert draft is not None and draft.status == "preview"
        assert draft.preview_message_id == old_pointer
        assert conversation is not None
        assert conversation.active_draft_id is None
        assert conversation.focused_draft_id is None
    assert query.edits[-1]["text"] == NOVA_COMPANION_ACCESS_CHANGED_TEXT
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_callback_reused_ordinary_failure_restores_different_prior_focus(
    db,
    fake_ai,
    monkeypatch,
    caplog,
):
    private = "PRIVATE_CALLBACK_FOCUS_LEASE"
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    raw_text = "Я хочу подготовить письмо Марине и открыть документ"
    incoming = CompanionMessage(raw_text, chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    _update, context = await deliver(
        bot,
        user,
        incoming,
        context=companion_context(telegram),
    )
    answer = sent_answer(incoming)
    add_data = callback_by_label(answer.markup_edits[-1], "Добавить как задачу")
    prior = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="companion",
        raw_text="Prior focus",
        parsed=ParsedThought(kind="note", title="Prior focus"),
    )
    reused = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="companion",
        raw_text=raw_text,
        parsed=ParsedThought(
            kind="task",
            title="подготовить письмо Марине",
            next_step="открыть документ",
        ),
    )
    assert prior.draft is not None and reused.draft is not None
    old_pointer = 990_201
    await bot.draft_service.set_preview_message(reused.draft.id, old_pointer)
    await bot.conversation.set_active_draft(
        user.telegram_id,
        user.telegram_id,
        prior.draft.id,
    )
    checks = 0

    async def actor_check(_user: User, _capability: Any) -> bool:
        nonlocal checks
        checks += 1
        if checks == 7:
            raise RuntimeError(private)
        return True

    monkeypatch.setattr(bot, "_nova_companion_capability_actor_is_current", actor_check)
    query = CompanionQuery(add_data, answer)
    with caplog.at_level(logging.WARNING):
        await bot.nova_companion_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=query,
            ),
            context,
        )

    transient = answer.replies[0]["message"]
    assert telegram.deleted == [(user.telegram_id, transient.message_id)]
    async with db.sessions() as session:
        conversation = await session.scalar(select(ConversationSession))
        durable_reused = await session.get(DraftInboxItem, reused.draft.id)
        assert conversation is not None and durable_reused is not None
        assert conversation.active_draft_id == prior.draft.id
        assert conversation.focused_draft_id == prior.draft.id
        assert durable_reused.status == "preview"
        assert durable_reused.preview_message_id == old_pointer
    assert private not in caplog.text
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_callback_cleanup_compare_and_clear_preserves_new_focus_replacement(
    db,
    fake_ai,
    monkeypatch,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    _update, context = await deliver(bot, user, incoming)
    answer = sent_answer(incoming)
    add_data = callback_by_label(answer.markup_edits[-1], "Добавить как задачу")
    checks = 0
    replacement_id: str | None = None

    async def actor_check(_user: User, capability: Any) -> bool:
        nonlocal checks, replacement_id
        checks += 1
        if checks != 7:
            return True
        replacement = await bot.draft_service.create_or_get_for_suggestion(
            user_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            expected_access_version=user.access_version,
            source="companion",
            raw_text="Новый независимый draft",
            parsed=ParsedThought(kind="note", title="Новый независимый draft"),
        )
        assert replacement.draft is not None
        replacement_id = replacement.draft.id
        await bot.conversation.set_active_draft(
            capability.telegram_user_id,
            capability.chat_id,
            replacement_id,
        )
        return False

    monkeypatch.setattr(bot, "_nova_companion_capability_actor_is_current", actor_check)
    query = CompanionQuery(add_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=query,
        ),
        context,
    )

    assert replacement_id is not None
    async with db.sessions() as session:
        conversation = await session.scalar(select(ConversationSession))
        replacement = await session.get(DraftInboxItem, replacement_id)
        assert conversation is not None
        assert conversation.active_draft_id == replacement_id
        assert conversation.focused_draft_id == replacement_id
        assert replacement is not None and replacement.status == "preview"
    assert query.edits[-1]["text"] == NOVA_COMPANION_ACCESS_CHANGED_TEXT
    assert len(await active_companion_drafts(db)) == 1
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("install_replacement", [False, True])
async def test_callback_final_screen_race_refences_access_before_focus_restore(
    db,
    fake_ai,
    monkeypatch,
    install_replacement,
):
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    raw_text = "Я хочу подготовить письмо Марине и открыть документ"
    incoming = CompanionMessage(raw_text, chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    _update, context = await deliver(
        bot,
        user,
        incoming,
        context=companion_context(telegram),
    )
    answer = sent_answer(incoming)
    add_data = answer.markup_edits[-1].inline_keyboard[0][0].callback_data
    assert isinstance(add_data, str)

    prior = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="companion",
        raw_text="Prior focus",
        parsed=ParsedThought(kind="note", title="Prior focus"),
    )
    reused = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="companion",
        raw_text=raw_text,
        parsed=ParsedThought(
            kind="task",
            title="подготовить письмо Марине",
            next_step="открыть документ",
        ),
    )
    replacement = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="companion",
        raw_text="Concurrent replacement",
        parsed=ParsedThought(kind="note", title="Concurrent replacement"),
    )
    assert prior.draft is not None
    assert reused.draft is not None
    assert replacement.draft is not None
    await bot.conversation.set_active_draft(
        user.telegram_id,
        user.telegram_id,
        prior.draft.id,
    )

    focus_installed = asyncio.Event()
    final_screen_started = asyncio.Event()
    release_final_screen = asyncio.Event()
    original_acquire = bot.conversation.acquire_active_draft_focus
    original_screen_check = bot.nova_companion_captures.consumed_screen_is_current

    async def acquire_focus(*args: Any, **kwargs: Any) -> Any:
        lease = await original_acquire(*args, **kwargs)
        focus_installed.set()
        return lease

    async def screen_check(capability: Any) -> bool:
        if focus_installed.is_set() and not final_screen_started.is_set():
            final_screen_started.set()
            await release_final_screen.wait()
            return False
        return await original_screen_check(capability)

    monkeypatch.setattr(bot.conversation, "acquire_active_draft_focus", acquire_focus)
    monkeypatch.setattr(
        bot.nova_companion_captures,
        "consumed_screen_is_current",
        screen_check,
    )
    query = CompanionQuery(add_data, answer)
    callback = asyncio.create_task(
        bot.nova_companion_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=query,
            ),
            context,
        )
    )
    await final_screen_started.wait()
    await AccessService(db).block(user.telegram_id, source="companion-race-test")
    if install_replacement:
        await bot.conversation.set_active_draft(
            user.telegram_id,
            user.telegram_id,
            replacement.draft.id,
        )
    release_final_screen.set()
    await callback

    async with db.sessions() as session:
        conversation = await session.scalar(select(ConversationSession))
        durable_reused = await session.get(DraftInboxItem, reused.draft.id)
        assert conversation is not None
        expected_focus = replacement.draft.id if install_replacement else None
        assert conversation.active_draft_id == expected_focus
        assert conversation.focused_draft_id == expected_focus
        assert durable_reused is not None and durable_reused.status == "preview"
        assert durable_reused.preview_message_id is None
    transient = answer.replies[0]["message"]
    assert telegram.deleted == [(user.telegram_id, transient.message_id)]
    assert query.answer_attempts == 1
    assert len(fake_ai.companion_calls) == 1
    assert bot._nova_companion_tasks == set()


async def test_callback_double_edit_failure_deletes_exact_consumed_canonical(
    db,
    fake_ai,
    caplog,
):
    private = "PRIVATE_CALLBACK_UI_FAILURE"
    fake_ai.companion_result = capture_result()
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage(
        "Я хочу подготовить письмо Марине и открыть документ",
        chat_id=user.telegram_id,
    )
    telegram = CompanionTelegram()
    _update, context = await deliver(
        bot,
        user,
        incoming,
        context=companion_context(telegram),
    )
    answer = sent_answer(incoming)
    dismiss_data = callback_by_label(answer.markup_edits[-1], "Не сейчас")
    query = CompanionQuery(dismiss_data, answer)

    async def fail_edit(*_args: Any, **_kwargs: Any) -> None:
        raise BadRequest(private)

    query.edit_message_reply_markup = fail_edit
    query.edit_message_text = fail_edit
    with caplog.at_level(logging.WARNING):
        await bot.nova_companion_callback(
            companion_update(
                answer,
                telegram_user_id=user.telegram_id,
                chat_id=user.telegram_id,
                query=query,
            ),
            context,
        )

    assert query.answer_attempts == 1
    assert telegram.deleted == [(user.telegram_id, answer.message_id)]
    retry = CompanionQuery(dismiss_data, answer)
    await bot.nova_companion_callback(
        companion_update(
            answer,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            query=retry,
        ),
        context,
    )
    assert retry.answer_attempts == 1 and retry.markup_removed == 0
    assert private not in caplog.text
    assert bot._nova_companion_tasks == set()


@pytest.mark.parametrize("outcome", ["access", "cancel"])
async def test_local_voice_access_and_cancellation_retire_exact_progress(
    db,
    fake_ai,
    monkeypatch,
    outcome,
):
    bot, user = await ready_companion_bot(db, fake_ai)
    incoming = CompanionMessage("Нова, ты тут?", chat_id=user.telegram_id)
    progress = CompanionMessage(
        "Распознаю…",
        chat_id=user.telegram_id,
        edit_error=asyncio.CancelledError() if outcome == "cancel" else None,
    )
    telegram = CompanionTelegram()
    context = companion_context(telegram)
    if outcome == "access":

        async def actor_changed(_user: User) -> bool:
            return False

        monkeypatch.setattr(bot, "_nova_companion_actor_is_current", actor_changed)

    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    route = bot.nova_companion_route(
        companion_update(
            incoming,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
        ),
        context,
        incoming.text,
        "voice",
        user=user,
        conversation_snapshot=snapshot,
        delivery_message=progress,
    )
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await route
    else:
        assert await route
    await bot._drain_nova_companion_tasks()

    assert telegram.deleted == []
    expected_text = (
        NOVA_COMPANION_ACCESS_CHANGED_TEXT
        if outcome == "access"
        else NOVA_COMPANION_UNAVAILABLE_TEXT
    )
    assert telegram.neutralized == [
        {
            "chat_id": user.telegram_id,
            "message_id": progress.message_id,
            "text": expected_text,
            "reply_markup": None,
            "parse_mode": None,
        }
    ]
    assert incoming.replies == []
    assert fake_ai.companion_calls == []
    assert await companion_counts(db) == (0, 0, 0)
    assert bot._nova_companion_tasks == set()


async def test_runtime_maintenance_physically_drops_expired_private_capture_text(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai)
    private = "PRIVATE_EXPIRED_CAPTURE_TEXT"
    staged = await bot.nova_companion_captures.stage(
        CaptureSuggestion("note", private),
        raw_text=private,
        owner_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        access_version=user.access_version,
        now=datetime(2000, 1, 1, tzinfo=UTC),
    )
    assert staged is not None
    assert bot.nova_companion_captures._screens
    assert bot.nova_companion_captures._capabilities

    await bot._cleanup_nova_companion_capabilities_safely()

    assert bot.nova_companion_captures._screens == {}
    assert bot.nova_companion_captures._capabilities == {}


async def test_focus_lease_restores_prior_and_preserves_concurrent_replacement(db, fake_ai):
    bot, user = await ready_companion_bot(db, fake_ai)

    async def create(title: str) -> DraftInboxItem:
        creation = await bot.draft_service.create_or_get_for_suggestion(
            user_id=user.id,
            telegram_user_id=user.telegram_id,
            chat_id=user.telegram_id,
            expected_access_version=user.access_version,
            source="companion",
            raw_text=title,
            parsed=ParsedThought(kind="note", title=title),
        )
        assert creation.draft is not None
        return creation.draft

    first = await create("Первый focus")
    temporary = await create("Временный focus")
    replacement = await create("Новый focus")
    await bot.conversation.set_active_draft(user.telegram_id, user.telegram_id, first.id)
    lease = await bot.conversation.acquire_active_draft_focus(
        user.telegram_id,
        user.telegram_id,
        temporary.id,
    )
    assert lease is not None
    assert first.id not in repr(lease)
    assert temporary.id not in repr(lease)
    assert await bot.conversation.restore_active_draft_focus_if_current(lease)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    assert snapshot.focused_draft_id == first.id

    lease = await bot.conversation.acquire_active_draft_focus(
        user.telegram_id,
        user.telegram_id,
        temporary.id,
    )
    assert lease is not None
    await bot.conversation.set_active_draft(
        user.telegram_id,
        user.telegram_id,
        replacement.id,
    )
    assert not await bot.conversation.restore_active_draft_focus_if_current(lease)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    assert snapshot.focused_draft_id == replacement.id

    frozen = datetime(2099, 8, 18, 12, 0, tzinfo=UTC)
    older_same_draft = await bot.conversation.acquire_active_draft_focus(
        user.telegram_id,
        user.telegram_id,
        temporary.id,
        now=frozen,
    )
    newer_same_draft = await bot.conversation.acquire_active_draft_focus(
        user.telegram_id,
        user.telegram_id,
        temporary.id,
        now=frozen + timedelta(seconds=1),
    )
    assert older_same_draft is not None and newer_same_draft is not None
    assert not await bot.conversation.restore_active_draft_focus_if_current(older_same_draft)
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    assert snapshot.focused_draft_id == temporary.id

    await bot.conversation.set_active_draft(user.telegram_id, user.telegram_id, first.id)
    lease = await bot.conversation.acquire_active_draft_focus(
        user.telegram_id,
        user.telegram_id,
        temporary.id,
    )
    assert lease is not None
    assert await bot.conversation.restore_active_draft_focus_if_current(
        lease,
        restore_prior=False,
    )
    snapshot = await bot.conversation.get(user.telegram_id, user.telegram_id)
    assert snapshot.focused_draft_id is None
    assert snapshot.active_draft is None


@pytest.mark.parametrize("failure", ["ordinary", "access"])
async def test_explicit_reused_focus_lease_restores_only_current_access_generation(
    db,
    fake_ai,
    monkeypatch,
    caplog,
    failure,
):
    private = "PRIVATE_FOCUS_LEASE_FAILURE"
    bot, user = await ready_companion_bot(db, fake_ai)
    prior = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="companion",
        raw_text="Существующий focus A",
        parsed=ParsedThought(kind="note", title="Существующий focus A"),
    )
    reused = await bot.draft_service.create_or_get_for_suggestion(
        user_id=user.id,
        telegram_user_id=user.telegram_id,
        chat_id=user.telegram_id,
        expected_access_version=user.access_version,
        source="text",
        raw_text="позвонить врачу",
        parsed=ParsedThought(kind="task", title="позвонить врачу"),
    )
    assert prior.draft is not None and reused.draft is not None
    old_pointer = 990_101
    await bot.draft_service.set_preview_message(reused.draft.id, old_pointer)
    await bot.conversation.set_active_draft(
        user.telegram_id,
        user.telegram_id,
        prior.draft.id,
    )
    checks = 0

    async def actor_check(_user: User) -> bool:
        nonlocal checks
        checks += 1
        if checks == 7:
            if failure == "ordinary":
                raise RuntimeError(private)
            return False
        return True

    monkeypatch.setattr(bot, "_nova_companion_actor_is_current", actor_check)
    incoming = CompanionMessage("Создай задачу позвонить врачу", chat_id=user.telegram_id)
    telegram = CompanionTelegram()
    with caplog.at_level(logging.WARNING):
        await deliver(bot, user, incoming, context=companion_context(telegram))

    transient = incoming.replies[0]["message"]
    assert telegram.deleted == [(user.telegram_id, transient.message_id)]
    async with db.sessions() as session:
        conversation = await session.scalar(select(ConversationSession))
        durable_reused = await session.get(DraftInboxItem, reused.draft.id)
        assert conversation is not None and durable_reused is not None
        assert durable_reused.status == "preview"
        assert durable_reused.preview_message_id == old_pointer
        expected = prior.draft.id if failure == "ordinary" else None
        assert conversation.active_draft_id == expected
        assert conversation.focused_draft_id == expected
    assert private not in caplog.text
    assert bot._nova_companion_tasks == set()
