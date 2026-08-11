import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.conversation import ConversationContextService
from future_self.dates import DateResolver
from future_self.db import Database
from future_self.models import ConversationMessage, ConversationSession, DraftInboxItem, InboxItem


class CommitGateDatabase(Database):
    """Hold a successful transaction immediately before its real database commit."""

    def __init__(self, url: str):
        super().__init__(url)
        self.before_commit = asyncio.Event()
        self.allow_commit = asyncio.Event()

    @asynccontextmanager
    async def session(self):
        async with super().session() as session:
            yield session
            self.before_commit.set()
            await self.allow_commit.wait()


class SessionObservedDatabase(Database):
    """Expose that a second database instance has entered its transaction."""

    def __init__(self, url: str):
        super().__init__(url)
        self.session_started = asyncio.Event()

    @asynccontextmanager
    async def session(self):
        async with super().session() as session:
            self.session_started.set()
            yield session


class FakeMessage:
    def __init__(self, text: str):
        self.text = text
        self.voice = None
        self.audio = None
        self.replies: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self


class FakeTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return ""


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
        transcription_provider="disabled",
        conversation_context_messages=12,
        conversation_context_ttl_hours=24,
    )


def update_for(message: FakeMessage, user_id: int, chat_id: int):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
    )


async def counts(db) -> tuple[int, int]:
    async with db.sessions() as session:
        drafts = await session.scalar(select(func.count(DraftInboxItem.id)))
        inbox = await session.scalar(select(func.count(InboxItem.id)))
    return int(drafts), int(inbox)


async def test_context_is_persistent_bounded_and_available_after_recreation(db):
    first = ConversationContextService(db, 12, 24)
    for index in range(15):
        await first.append(
            100,
            200,
            role="user" if index % 2 == 0 else "assistant",
            content=f"Сообщение {index}",
            source="text",
            intent="conversation",
            topic="планирование",
        )
    recreated = ConversationContextService(db, 12, 24)
    snapshot = await recreated.get(100, 200)
    assert len(snapshot.messages) == 12
    assert snapshot.messages[-1]["content"] == "Сообщение 14"
    assert snapshot.current_topic == "планирование"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 12


async def test_context_isolated_by_user_and_chat(db):
    service = ConversationContextService(db, 12, 24)
    await service.append(
        100,
        200,
        role="user",
        content="Личный контекст",
        source="text",
        intent="conversation",
    )
    assert (await service.get(100, 200)).messages
    assert not (await service.get(101, 200)).messages
    assert not (await service.get(100, 201)).messages


async def test_context_ttl_excludes_expired_messages(db):
    service = ConversationContextService(db, 12, 24)
    await service.append(
        100,
        200,
        role="user",
        content="Старое сообщение",
        source="text",
        intent="conversation",
    )
    async with db.session() as session:
        conversation = await session.scalar(select(ConversationSession))
        conversation.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert not (await service.get(100, 200)).messages


async def test_expired_context_purge_is_bounded_and_preserves_active_session(db, caplog):
    service = ConversationContextService(db, 12, 24)
    private = "PRIVATE_CONVERSATION_PURGE_SENTINEL"
    for telegram_user_id in range(501, 505):
        await service.append(
            telegram_user_id,
            600,
            role="user",
            content=f"{private}:{telegram_user_id}",
            source="text",
            intent="conversation",
        )

    current = datetime.now(UTC)
    async with db.session() as session:
        rows = list(
            (
                await session.scalars(select(ConversationSession).order_by(ConversationSession.id))
            ).all()
        )
        for offset, row in enumerate(rows[:3], start=1):
            row.expires_at = current - timedelta(minutes=offset)
        rows[3].expires_at = current + timedelta(hours=1)

    with caplog.at_level(logging.INFO):
        result = await service.purge_expired(batch_size=2, now=current)
    assert result == 2
    assert type(result) is int
    assert private not in repr(result)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationSession.id))) == 2
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2

    assert await service.purge_expired(batch_size=2, now=current) == 1
    assert await service.purge_expired(batch_size=2, now=current) == 0
    assert await service.purge_expired(batch_size=100, now=current) == 0
    active = await service.get(504, 600)
    assert len(active.messages) == 1
    assert private not in caplog.text


async def test_expired_context_purge_is_concurrent_and_idempotent(db):
    service = ConversationContextService(db, 12, 24)
    current = datetime.now(UTC)
    for telegram_user_id in range(511, 516):
        await service.append(
            telegram_user_id,
            610,
            role="user",
            content="expired",
            source="text",
            intent="conversation",
        )
    async with db.session() as session:
        rows = list((await session.scalars(select(ConversationSession))).all())
        for row in rows:
            row.expires_at = current - timedelta(seconds=1)

    deleted = await asyncio.gather(
        service.purge_expired(batch_size=5, now=current),
        service.purge_expired(batch_size=5, now=current),
    )
    assert sum(deleted) == 5
    assert await service.purge_expired(batch_size=5, now=current) == 0
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationSession.id))) == 0
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0


async def test_append_refresh_wins_purge_race_across_database_instances(db):
    telegram_user_id = 521
    chat_id = 620
    initial = ConversationContextService(db, 12, 24)
    original_session_id = await initial.append(
        telegram_user_id,
        chat_id,
        role="user",
        content="old expired message",
        source="text",
        intent="conversation",
    )
    current = datetime.now(UTC)
    async with db.session() as session:
        conversation = await session.get(ConversationSession, original_session_id)
        assert conversation is not None
        conversation.expires_at = current - timedelta(seconds=1)

    append_db = CommitGateDatabase(db.url)
    purge_db = SessionObservedDatabase(db.url)
    async with append_db.engine.connect() as connection:
        await connection.execute(select(1))
    async with purge_db.engine.connect() as connection:
        await connection.execute(select(1))
    appender = ConversationContextService(append_db, 12, 24)
    purger = ConversationContextService(purge_db, 12, 24)
    append_task = asyncio.create_task(
        appender.append(
            telegram_user_id,
            chat_id,
            role="user",
            content="fresh message",
            source="text",
            intent="conversation",
        )
    )
    purge_task = None
    try:
        await asyncio.wait_for(append_db.before_commit.wait(), timeout=5)
        purge_task = asyncio.create_task(purger.purge_expired(batch_size=1, now=current))
        await asyncio.wait_for(purge_db.session_started.wait(), timeout=5)
        append_db.allow_commit.set()

        assert await asyncio.wait_for(append_task, timeout=5) == original_session_id
        assert await asyncio.wait_for(purge_task, timeout=5) == 0
    finally:
        append_db.allow_commit.set()
        if not append_task.done():
            append_task.cancel()
            await asyncio.gather(append_task, return_exceptions=True)
        if purge_task is not None and not purge_task.done():
            purge_task.cancel()
            await asyncio.gather(purge_task, return_exceptions=True)
        await append_db.dispose()
        await purge_db.dispose()

    async with db.sessions() as session:
        conversations = list((await session.scalars(select(ConversationSession))).all())
        messages = list((await session.scalars(select(ConversationMessage))).all())
    assert len(conversations) == 1
    assert conversations[0].id == original_session_id
    assert not ConversationContextService._is_expired(conversations[0].expires_at, current)
    assert [message.content for message in messages] == ["fresh message"]


async def test_purge_wins_race_then_append_recreates_once_across_database_instances(db):
    telegram_user_id = 531
    chat_id = 630
    initial = ConversationContextService(db, 12, 24)
    original_session_id = await initial.append(
        telegram_user_id,
        chat_id,
        role="user",
        content="old expired message",
        source="text",
        intent="conversation",
    )
    current = datetime.now(UTC)
    async with db.session() as session:
        conversation = await session.get(ConversationSession, original_session_id)
        assert conversation is not None
        conversation.expires_at = current - timedelta(seconds=1)

    purge_db = CommitGateDatabase(db.url)
    append_db = SessionObservedDatabase(db.url)
    async with purge_db.engine.connect() as connection:
        await connection.execute(select(1))
    async with append_db.engine.connect() as connection:
        await connection.execute(select(1))
    purger = ConversationContextService(purge_db, 12, 24)
    appender = ConversationContextService(append_db, 12, 24)
    purge_task = asyncio.create_task(purger.purge_expired(batch_size=1, now=current))
    append_task = None
    try:
        await asyncio.wait_for(purge_db.before_commit.wait(), timeout=5)
        append_task = asyncio.create_task(
            appender.append(
                telegram_user_id,
                chat_id,
                role="user",
                content="fresh message",
                source="text",
                intent="conversation",
            )
        )
        await asyncio.wait_for(append_db.session_started.wait(), timeout=5)
        purge_db.allow_commit.set()

        assert await asyncio.wait_for(purge_task, timeout=5) == 1
        recreated_session_id = await asyncio.wait_for(append_task, timeout=5)
    finally:
        purge_db.allow_commit.set()
        if not purge_task.done():
            purge_task.cancel()
            await asyncio.gather(purge_task, return_exceptions=True)
        if append_task is not None and not append_task.done():
            append_task.cancel()
            await asyncio.gather(append_task, return_exceptions=True)
        await purge_db.dispose()
        await append_db.dispose()

    async with db.sessions() as session:
        conversations = list((await session.scalars(select(ConversationSession))).all())
        messages = list((await session.scalars(select(ConversationMessage))).all())
    assert len(conversations) == 1
    assert conversations[0].id == recreated_session_id
    # SQLite may reuse an integer primary key after hard deletion; the purge
    # outcome above proves that this is a newly inserted row either way.
    assert not ConversationContextService._is_expired(conversations[0].expires_at, current)
    assert len(messages) == 1
    assert messages[0].session_id == recreated_session_id
    assert messages[0].content == "fresh message"


@pytest.mark.parametrize("batch_size", [True, 0, 101])
async def test_expired_context_purge_rejects_unbounded_batch(db, batch_size):
    service = ConversationContextService(db, 12, 24)
    with pytest.raises(ValueError, match="batch_size"):
        await service.purge_expired(batch_size=batch_size)


async def test_system_action_begin_uses_unique_atomic_versions(db):
    service = ConversationContextService(db, 12, 24)
    assert await service.begin_system_action(110, 210, "bootstrap", [{"id": 0}]) == 1

    requests = (
        ("discard_all_active_drafts", [{"id": "draft"}]),
        ("archive_overdue_tasks", [{"id": 42}]),
    )
    versions = await asyncio.gather(
        *(service.begin_system_action(110, 210, action, snapshot) for action, snapshot in requests)
    )

    assert sorted(versions) == [2, 3]
    winning_action, winning_snapshot = requests[versions.index(3)]
    current = await service.get(110, 210)
    assert current.system_action_version == 3
    assert current.system_pending_action == winning_action
    assert current.system_draft_snapshot == winning_snapshot


async def test_system_action_claim_and_clear_are_bound_to_expected_version(db):
    service = ConversationContextService(db, 12, 24)
    old_version = await service.begin_system_action(
        120,
        220,
        "discard_all_active_drafts",
        [{"id": "old"}],
    )
    new_snapshot = [{"id": 99, "version": 7}]
    new_version = await service.begin_system_action(
        120,
        220,
        "archive_overdue_tasks",
        new_snapshot,
    )

    assert not await service.clear_system_action(
        120,
        220,
        expected_version=old_version,
    )
    assert await service.claim_system_action(120, 220, expected_version=old_version) is None
    still_current = await service.get(120, 220)
    assert still_current.system_action_version == new_version
    assert still_current.system_draft_snapshot == new_snapshot

    competing_claims = await asyncio.gather(
        service.claim_system_action(120, 220, expected_version=new_version),
        service.claim_system_action(120, 220, expected_version=new_version),
    )
    claims = [claim for claim in competing_claims if claim is not None]
    assert len(claims) == 1
    assert claims[0].action == "archive_overdue_tasks"
    assert claims[0].snapshot == new_snapshot
    assert claims[0].version == new_version
    assert (await service.get(120, 220)).system_pending_action is None

    latest_version = await service.begin_system_action(
        120,
        220,
        "trash_inbox_items",
        [{"id": 100}],
    )
    assert not await service.clear_system_action(
        120,
        220,
        expected_version=new_version,
    )
    latest = await service.get(120, 220)
    assert latest.system_action_version == latest_version
    assert latest.system_pending_action == "trash_inbox_items"
    assert await service.clear_system_action(
        120,
        220,
        expected_version=latest_version,
    )
    assert not await service.clear_system_action(
        120,
        220,
        expected_version=latest_version,
    )


async def test_system_action_confirm_and_cancel_have_exactly_one_winner(db):
    service = ConversationContextService(db, 12, 24)
    version = await service.begin_system_action(
        130,
        230,
        "trash_inbox_items",
        [{"id": 1}],
    )

    claim, cancelled = await asyncio.gather(
        service.claim_system_action(130, 230, expected_version=version),
        service.clear_system_action(130, 230, expected_version=version),
    )

    assert (claim is not None) != cancelled
    if claim is not None:
        assert await service.finalize_system_action_claim(
            130,
            230,
            expected_version=version,
        )
    assert (await service.get(130, 230)).system_pending_action is None


async def test_recent_conversation_reaches_next_message_after_bot_recreation(db, fake_ai):
    first_bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    first_message = FakeMessage("Давай обсудим еженедельное планирование")
    await first_bot.text(update_for(first_message, 300, 400), SimpleNamespace(user_data={}))

    second_bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    follow_up = FakeMessage(
        "Ну вот, мы только что общались про то, что еженедельное планирование полезно"
    )
    await second_bot.text(update_for(follow_up, 300, 400), SimpleNamespace(user_data={}))
    assert "обсуждали еженедельное планирование" in str(follow_up.replies[-1]["text"])
    recent = fake_ai.conversation_contexts[-1]["recent_messages"]
    assert any("еженедельное планирование" in row["content"] for row in recent)


async def test_task_question_offers_choices_but_saves_nothing(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    await bot.text(
        update_for(FakeMessage("Давай обсудим еженедельное планирование"), 310, 410),
        SimpleNamespace(user_data={}),
    )
    question = FakeMessage("Ты занесёшь это в задачу?")
    await bot.text(update_for(question, 310, 410), SimpleNamespace(user_data={}))
    labels = {
        button.text
        for row in question.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    }
    assert labels == {"Создать задачу", "Оставить идеей", "Уточнить дату", "Ничего"}
    assert await counts(db) == (0, 0)


async def test_save_this_creates_preview_from_unambiguous_context_only(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    original = "Давай обсудим еженедельное планирование"
    await bot.text(update_for(FakeMessage(original), 320, 420), SimpleNamespace(user_data={}))
    capture = FakeMessage("Да, сохрани это")
    await bot.text(update_for(capture, 320, 420), SimpleNamespace(user_data={}))
    assert "Исходный текст" in str(capture.replies[-1]["text"])
    assert original in str(capture.replies[-1]["text"])
    assert await counts(db) == (1, 0)


async def test_ambiguous_reference_is_not_guessed(db, fake_ai):
    service = ConversationContextService(db, 12, 24)
    for content in ("Первый подробный вариант плана", "Второй подробный вариант плана"):
        await service.append(
            330,
            430,
            role="user",
            content=content,
            source="text",
            intent="conversation",
        )
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    message = FakeMessage("Сохрани это")
    await bot.text(update_for(message, 330, 430), SimpleNamespace(user_data={}))
    assert "Уточни" in str(message.replies[-1]["text"])
    assert await counts(db) == (0, 0)


def test_date_resolver_detects_july_28_2026_and_nearest_sunday():
    resolver = DateResolver()
    result = resolver.resolve(
        "Начну в воскресенье, 28 июля 2026",
        "Europe/Moscow",
        now=datetime(2026, 7, 13, 10, tzinfo=UTC),
    )
    assert result.status == "conflict"
    assert result.actual_weekday == "вторник"
    assert result.options[0].value.isoformat() == "2026-07-26"
    assert result.options[0].weekday == "воскресенье"


def test_date_without_year_uses_nearest_future_and_timezone_affects_tomorrow():
    resolver = DateResolver()
    inferred = resolver.resolve(
        "Начну 28 июля", "Europe/Moscow", now=datetime(2026, 8, 1, tzinfo=UTC)
    )
    assert inferred.target_date.isoformat() == "2027-07-28"
    assert inferred.inferred_year is True
    moment = datetime(2026, 7, 12, 21, 30, tzinfo=UTC)
    assert resolver.resolve("завтра", "Europe/Moscow", now=moment).target_date.isoformat() == (
        "2026-07-14"
    )
    assert resolver.resolve("завтра", "America/New_York", now=moment).target_date.isoformat() == (
        "2026-07-13"
    )


async def test_date_conflict_handler_asks_and_creates_no_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    message = FakeMessage("Начну с воскресенья, 28 июля 2026")
    await bot.text(update_for(message, 340, 440), SimpleNamespace(user_data={}))
    answer = str(message.replies[-1]["text"])
    assert "28.07.2026" in answer and "вторник" in answer
    assert "26.07.2026" in answer and "воскресенье" in answer
    assert await counts(db) == (0, 0)


async def test_context_logging_does_not_include_private_message(db, caplog):
    private = "очень личный полный текст пользователя"
    service = ConversationContextService(db, 12, 24)
    with caplog.at_level(logging.INFO):
        await service.append(
            350,
            450,
            role="user",
            content=private,
            source="text",
            intent="reflection",
        )
    assert private not in caplog.text
