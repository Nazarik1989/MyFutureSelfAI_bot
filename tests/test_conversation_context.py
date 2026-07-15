import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.conversation import ConversationContextService
from future_self.dates import DateResolver
from future_self.models import ConversationMessage, ConversationSession, DraftInboxItem, InboxItem


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
