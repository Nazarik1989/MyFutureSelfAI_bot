import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.domain import IntentRouter
from future_self.models import DraftInboxItem, InboxItem


class FakeMessage:
    def __init__(self, text: str | None = None, *, voice=None):
        self.text = text
        self.voice = voice
        self.audio = None
        self.replies: list[dict[str, object]] = []
        self.edits: list[str] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self

    async def edit_text(self, text: str):
        self.edits.append(text)


class FakeCallbackQuery:
    def __init__(self, data: str, message: FakeMessage):
        self.data = data
        self.message = message
        self.answers: list[tuple[str | None, bool]] = []
        self.edits: list[str] = []
        self.markup_removed = 0

    async def answer(self, text: str | None = None, show_alert: bool = False):
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text: str):
        self.edits.append(text)

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_removed += 1


class FakeTelegramFile:
    async def download_as_bytearray(self):
        return bytearray(b"voice")


class FakeVoice:
    duration = 3
    file_size = 5
    mime_type = "audio/ogg"
    file_name = "voice.ogg"

    async def get_file(self):
        return FakeTelegramFile()


class GreetingTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "Привет"


class IdeaTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "Мне пришла идея сделать совместное пространство для друзей"


class CorrectedTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "нужно заниматься спортом 3 раза в неделю"


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
        transcription_provider="disabled",
        intent_confidence_threshold=0.70,
    )


def update_for(message: FakeMessage, user_id: int = 501):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=user_id + 10_000),
    )


def preview_callback(message: FakeMessage, action: str) -> str:
    markup = message.replies[-1]["reply_markup"]
    return next(
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data.startswith(f"inbox:{action}:")
    )


async def inbox_count(db) -> int:
    async with db.sessions() as session:
        return int(await session.scalar(select(func.count(InboxItem.id))))


async def draft_count(db) -> int:
    async with db.sessions() as session:
        return int(await session.scalar(select(func.count(DraftInboxItem.id))))


async def test_greeting_gets_answer_and_is_not_saved(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Привет")
    await bot.text(update_for(message), SimpleNamespace(user_data={}))
    assert message.replies[-1]["text"] == "Привет!"
    assert await inbox_count(db) == 0
    assert await draft_count(db) == 0


async def test_tomorrow_question_uses_timezone_and_is_not_saved(db, fake_ai):
    router = IntentRouter(fake_ai, 0.70)
    moment = datetime(2026, 7, 12, 21, 30, tzinfo=UTC)
    moscow = await router.route("Какой завтра день недели?", "Europe/Moscow", now=moment)
    new_york = await router.route("Какой завтра день недели?", "America/New_York", now=moment)
    assert moscow.answer == "Завтра вторник."
    assert new_york.answer == "Завтра понедельник."
    assert await inbox_count(db) == 0


async def test_question_handler_creates_no_draft_or_inbox(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Какой завтра день недели?")
    await bot.text(update_for(message, 502), SimpleNamespace(user_data={}))
    assert str(message.replies[-1]["text"]).startswith("Завтра")
    assert await draft_count(db) == 0
    assert await inbox_count(db) == 0


async def test_idea_preview_requires_callback_and_saves_once(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    update = update_for(message)
    await bot.text(update, context)
    assert "Тип: идея" in str(message.replies[-1]["text"])
    assert await inbox_count(db) == 0

    callback_data = preview_callback(message, "save")
    query = FakeCallbackQuery(callback_data, message)
    callback_update = SimpleNamespace(
        callback_query=query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )
    await bot.inbox_action(callback_update, context)
    await bot.inbox_action(callback_update, context)
    assert await inbox_count(db) == 1
    assert query.answers[-1] == ("Эта карточка уже неактуальна. Создай новую.", True)


async def test_drop_preview_never_saves(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    message = FakeMessage("Не забудь сделать звонок")
    update = update_for(message)
    await bot.text(update, context)
    query = FakeCallbackQuery(preview_callback(message, "drop"), message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert await inbox_count(db) == 0


async def test_edit_repreviews_and_only_second_confirmation_saves(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    first_message = FakeMessage("Мне пришла идея создать пространство")
    update = update_for(first_message)
    await bot.text(update, context)
    first_edit_callback = preview_callback(first_message, "edit")
    first_draft_id = first_edit_callback.split(":")[2]
    edit_query = FakeCallbackQuery(first_edit_callback, first_message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=edit_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert await inbox_count(db) == 0

    corrected = "Мне пришла идея создать совместное пространство для друзей"
    corrected_message = FakeMessage(corrected)
    await bot.text(update_for(corrected_message), context)
    save_callback = preview_callback(corrected_message, "save")
    assert save_callback.split(":")[2] == first_draft_id
    assert save_callback.endswith(":2")
    assert "Тип: идея" in str(corrected_message.replies[-1]["text"])
    assert await inbox_count(db) == 0

    save_query = FakeCallbackQuery(save_callback, corrected_message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=save_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert await inbox_count(db) == 1
    async with db.sessions() as session:
        saved = await session.scalar(select(InboxItem))
    assert saved.raw_text == corrected


async def test_text_and_voice_share_intent_router(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    await bot.text(update_for(FakeMessage("Привет"), 601), SimpleNamespace(user_data={}))
    voice_message = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice_message, 601), SimpleNamespace(user_data={}))
    assert [call[0] for call in fake_ai.route_calls[-2:]] == ["Привет", "Привет"]
    assert voice_message.edits[0] == "Я услышал: «Привет»"
    assert voice_message.replies[-1]["text"] == "Привет!"


async def test_low_confidence_shows_choices_and_choice_still_previews(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    context = SimpleNamespace(user_data={})
    message = FakeMessage("непонятно")
    update = update_for(message, 701)
    await bot.text(update, context)
    markup = message.replies[-1]["reply_markup"]
    callback_data = {button.callback_data for row in markup.inline_keyboard for button in row}
    assert {
        next(value for value in callback_data if value.startswith("intent:answer:")),
        next(value for value in callback_data if value.startswith("intent:idea:")),
        next(value for value in callback_data if value.startswith("intent:task:")),
        next(value for value in callback_data if value.startswith("intent:note:")),
        next(value for value in callback_data if value.startswith("intent:drop:")),
    } == callback_data
    assert await inbox_count(db) == 0

    idea_callback = next(value for value in callback_data if value.startswith("intent:idea:"))
    query = FakeCallbackQuery(idea_callback, message)
    await bot.intent_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        context,
    )
    assert any(
        button.callback_data.startswith("inbox:save:")
        for row in message.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    )
    assert await inbox_count(db) == 0


async def test_voice_idea_creates_persistent_preview_without_inbox(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    message = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(message, 801), SimpleNamespace(user_data={}))
    assert message.edits[0].startswith("Я услышал:")
    assert preview_callback(message, "save").startswith("inbox:save:")
    assert await draft_count(db) == 1
    assert await inbox_count(db) == 0


async def test_persistent_draft_can_be_saved_after_bot_restart(db, fake_ai):
    first_bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    update = update_for(message, 811)
    await first_bot.text(update, SimpleNamespace(user_data={}))
    callback_data = preview_callback(message, "save")

    restarted_bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    query = FakeCallbackQuery(callback_data, message)
    await restarted_bot.inbox_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    assert await inbox_count(db) == 1


async def test_foreign_user_cannot_confirm_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    owner_update = update_for(message, 821)
    await bot.text(owner_update, SimpleNamespace(user_data={}))
    query = FakeCallbackQuery(preview_callback(message, "save"), message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=SimpleNamespace(id=999),
            effective_chat=owner_update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    assert await inbox_count(db) == 0
    assert query.answers[-1][1] is True


async def test_two_fast_messages_create_independent_drafts(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    await bot.text(
        update_for(FakeMessage("Мне пришла идея создать пространство А"), 831),
        SimpleNamespace(user_data={}),
    )
    await bot.text(
        update_for(FakeMessage("Мне пришла идея создать пространство Б"), 831),
        SimpleNamespace(user_data={}),
    )
    assert await draft_count(db) == 2
    assert await inbox_count(db) == 0


async def test_concurrent_save_callbacks_create_one_inbox_item(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Мне пришла идея создать совместное пространство")
    update = update_for(message, 841)
    await bot.text(update, SimpleNamespace(user_data={}))
    callback_data = preview_callback(message, "save")
    first_query = FakeCallbackQuery(callback_data, message)
    second_query = FakeCallbackQuery(callback_data, message)

    def callback_update(query):
        return SimpleNamespace(
            callback_query=query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        )

    await asyncio.gather(
        bot.inbox_action(callback_update(first_query), SimpleNamespace(user_data={})),
        bot.inbox_action(callback_update(second_query), SimpleNamespace(user_data={})),
    )
    assert await inbox_count(db) == 1
    assert sorted(len(query.edits) for query in (first_query, second_query)) == [0, 1]


async def test_voice_can_revise_editing_draft_without_saving(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    first_message = FakeMessage("Мне пришла идея создать пространство")
    update = update_for(first_message, 851)
    await bot.text(update, SimpleNamespace(user_data={}))
    edit_query = FakeCallbackQuery(preview_callback(first_message, "edit"), first_message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=edit_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    bot.transcription = CorrectedTranscription()
    voice_message = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice_message, 851), SimpleNamespace(user_data={}))
    assert preview_callback(voice_message, "save").endswith(":2")
    assert await inbox_count(db) == 0


async def test_cancel_discards_persistent_editing_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, GreetingTranscription())
    message = FakeMessage("Мне пришла идея создать пространство")
    update = update_for(message, 861)
    await bot.text(update, SimpleNamespace(user_data={}))
    edit_query = FakeCallbackQuery(preview_callback(message, "edit"), message)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=edit_query,
            effective_user=update.effective_user,
            effective_chat=update.effective_chat,
        ),
        SimpleNamespace(user_data={}),
    )
    cancel_message = FakeMessage()
    await bot.cancel_draft_edit(update_for(cancel_message, 861), SimpleNamespace(user_data={}))
    assert "ничего не сохранено" in str(cancel_message.replies[-1]["text"])
    assert await inbox_count(db) == 0


async def test_local_voice_edit_save_duplicate_flow_counts(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, IdeaTranscription())
    context = SimpleNamespace(user_data={})
    voice_message = FakeMessage(voice=FakeVoice())
    update = update_for(voice_message, 871)
    await bot.voice(update, context)
    counts = [await inbox_count(db)]

    edit_query = FakeCallbackQuery(preview_callback(voice_message, "edit"), voice_message)
    callback_update = SimpleNamespace(
        callback_query=edit_query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )
    await bot.inbox_action(callback_update, context)
    counts.append(await inbox_count(db))

    corrected_message = FakeMessage("нужно заниматься спортом 3 раза в неделю")
    await bot.text(update_for(corrected_message, 871), context)
    counts.append(await inbox_count(db))

    save_query = FakeCallbackQuery(preview_callback(corrected_message, "save"), corrected_message)
    save_update = SimpleNamespace(
        callback_query=save_query,
        effective_user=update.effective_user,
        effective_chat=update.effective_chat,
    )
    await bot.inbox_action(save_update, context)
    counts.append(await inbox_count(db))
    await bot.inbox_action(save_update, context)
    counts.append(await inbox_count(db))

    print("handler_flow_counts=" + " -> ".join(map(str, counts)))
    assert counts == [0, 0, 0, 1, 1]
