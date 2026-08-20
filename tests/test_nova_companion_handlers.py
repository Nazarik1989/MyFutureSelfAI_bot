from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select, update
from telegram.error import BadRequest

from future_self.access import AccessService
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.dates import DateResolver
from future_self.models import (
    ConversationMessage,
    ConversationSession,
    DraftInboxItem,
    Goal,
    InboxItem,
    User,
)
from future_self.nova_companion_flow import CaptureSuggestion
from future_self.nova_companion_handlers import (
    NOVA_COMPANION_ACCESS_CHANGED_TEXT,
    NOVA_COMPANION_CONTEXT_CHANGED_TEXT,
    NOVA_COMPANION_UNAVAILABLE_TEXT,
)
from future_self.schemas import NovaCompanionCapture, NovaCompanionResponse, ParsedThought


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
) -> tuple[FutureSelfBot, User]:
    bot = FutureSelfBot(companion_settings(db), db, ai, NoopTranscription())
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
    assert len(fake_ai.companion_calls) == 1


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
    assert await companion_counts(db) == (2, 0, 0)


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
