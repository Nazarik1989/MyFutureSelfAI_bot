from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import ConversationSession, DraftInboxItem, InboxItem
from future_self.schemas import ParsedThought, TemporalResolution


class FakeMessage:
    _next_id = 100

    def __init__(self, text: str | None = None, *, voice=None, reply_to_message=None):
        self.text = text
        self.voice = voice
        self.audio = None
        self.reply_to_message = reply_to_message
        self.replies: list[dict[str, object]] = []
        self.edits: list[str] = []
        self.message_id = self._next_id
        FakeMessage._next_id += 1

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


class FakeBot:
    def __init__(self):
        self.removed: list[tuple[int, int]] = []

    async def edit_message_reply_markup(self, *, chat_id, message_id, reply_markup):
        self.removed.append((chat_id, message_id))


class FakeFile:
    async def download_as_bytearray(self):
        return bytearray(b"voice")


class FakeVoice:
    duration = 2
    file_size = 5
    mime_type = "audio/ogg"
    file_name = "voice.ogg"

    async def get_file(self):
        return FakeFile()


class SaveTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return "сохрани"


class PhraseTranscription:
    enabled = True

    def __init__(self, phrase: str):
        self.phrase = phrase

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return self.phrase


class NoopTranscription:
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
    )


def update_for(message: FakeMessage, user_id: int, chat_id: int):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
    )


def context_with_bot() -> SimpleNamespace:
    return SimpleNamespace(user_data={}, bot=FakeBot())


async def counts(db) -> tuple[int, int]:
    async with db.sessions() as session:
        drafts = await session.scalar(select(func.count(DraftInboxItem.id)))
        inbox = await session.scalar(select(func.count(InboxItem.id)))
    return int(drafts), int(inbox)


async def active_count(db, user_id: int, chat_id: int) -> int:
    async with db.sessions() as session:
        return int(
            await session.scalar(
                select(func.count(DraftInboxItem.id)).where(
                    DraftInboxItem.telegram_user_id == user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status.in_(("preview", "editing")),
                )
            )
        )


async def make_preview(bot, user_id: int, chat_id: int, context):
    message = FakeMessage("Мне пришла идея планировать следующую неделю")
    await bot.text(update_for(message, user_id, chat_id), context)
    return message


async def make_named_preview(bot, user_id, chat_id, context, title, raw_text):
    user = await bot._user(user_id)
    message = FakeMessage(raw_text)
    await bot._show_preview(
        message,
        user.id,
        user_id,
        chat_id,
        raw_text,
        "text",
        ParsedThought(kind="idea", title=title),
    )
    return message


@pytest.mark.parametrize(
    "command", ["сохрани", "да, сохрани", "можешь сохранить", "можешь сохранить?"]
)
async def test_explicit_text_save_confirms_active_preview(db, fake_ai, command):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_preview(bot, 1001, 2001, context)
    command_message = FakeMessage(command)
    await bot.text(update_for(command_message, 1001, 2001), context)
    assert await counts(db) == (1, 1)
    assert "Сохранено в inbox по текстовой команде" in command_message.replies[-1]["text"]


@pytest.mark.parametrize(
    "command",
    ["ты это сохранишь?", "не сохраняй пока", "можно будет сохранить?"],
)
async def test_questions_and_deferred_phrases_do_not_change_preview(db, fake_ai, command):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_preview(bot, 1002, 2002, context)
    await bot.text(update_for(FakeMessage(command), 1002, 2002), context)
    assert await counts(db) == (1, 0)
    async with db.sessions() as session:
        draft = await session.scalar(select(DraftInboxItem))
    assert draft.status == "preview"


async def test_high_confidence_do_not_save_discards_preview(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_preview(bot, 1003, 2003, context)
    await bot.text(update_for(FakeMessage("не сохраняй"), 1003, 2003), context)
    async with db.sessions() as session:
        draft = await session.scalar(select(DraftInboxItem))
    assert draft.status == "discarded"
    assert await counts(db) == (1, 0)


async def test_voice_and_callback_save_share_atomic_confirm(db, fake_ai, monkeypatch):
    bot = FutureSelfBot(settings(), db, fake_ai, SaveTranscription())
    context = context_with_bot()
    calls = 0
    original = bot.draft_service.confirm

    async def counted_confirm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(bot.draft_service, "confirm", counted_confirm)
    first = await make_preview(bot, 1004, 2004, context)
    save_data = first.replies[-1]["reply_markup"].inline_keyboard[0][0].callback_data
    query = FakeCallbackQuery(save_data, first)
    await bot.inbox_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=SimpleNamespace(id=1004),
            effective_chat=SimpleNamespace(id=2004),
        ),
        context,
    )
    await make_preview(bot, 1004, 2004, context)
    voice = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice, 1004, 2004), context)
    assert calls == 2
    assert await counts(db) == (2, 2)


async def test_repeated_voice_save_is_idempotent_and_removes_preview_keyboard(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, SaveTranscription())
    context = context_with_bot()
    preview = await make_preview(bot, 1005, 2005, context)
    first = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(first, 1005, 2005), context)
    second = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(second, 1005, 2005), context)
    assert await counts(db) == (1, 1)
    assert (2005, preview.message_id) in context.bot.removed


async def test_foreign_voice_command_cannot_confirm_owner_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, SaveTranscription())
    context = context_with_bot()
    await make_preview(bot, 1006, 2006, context)
    await bot.voice(update_for(FakeMessage(voice=FakeVoice()), 9999, 2006), context)
    assert await counts(db) == (1, 0)


async def test_two_active_drafts_require_clarification(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_preview(bot, 1007, 2007, context)
    second = FakeMessage("Мне пришла идея организовать пространство для спорта")
    await bot.text(update_for(second, 1007, 2007), context)
    await bot.conversation.clear_focus(1007, 2007)
    command = FakeMessage("сохрани")
    await bot.text(update_for(command, 1007, 2007), context)
    assert command.replies[-1]["text"] == "К какой карточке применить команду?"
    assert await counts(db) == (2, 0)


async def test_date_choice_stays_july_26_and_task_uses_active_topic(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await bot.conversation.append(
        1008,
        2008,
        role="user",
        content="Обсудим еженедельное планирование",
        source="text",
        intent="conversation",
        topic="еженедельное планирование",
    )
    conflict = FakeMessage("Начну в воскресенье, 28 июля 2026")
    await bot.text(update_for(conflict, 1008, 2008), context)
    assert "26.07.2026" in conflict.replies[-1]["text"]

    choice = FakeMessage("26, значит, с воскресенья именно")
    await bot.text(update_for(choice, 1008, 2008), context)
    assert "26.07.2026" in choice.replies[-1]["text"]
    assert "19.07.2026" not in choice.replies[-1]["text"]
    async with db.sessions() as session:
        draft = await session.scalar(select(DraftInboxItem))
    assert draft.resolved_date.isoformat() == "2026-07-26"
    assert draft.resolved_date.weekday() == 6

    task_command = FakeMessage("Добавь к задачам")
    await bot.text(update_for(task_command, 1008, 2008), context)
    async with db.sessions() as session:
        draft = await session.scalar(select(DraftInboxItem))
    assert draft.kind == "task"
    assert draft.title == "Еженедельное планирование"
    assert draft.title != task_command.text
    assert draft.description == (
        "Каждое воскресенье составлять план следующей недели и подводить итоги предыдущей"
    )
    assert draft.resolved_date.isoformat() == "2026-07-26"
    preview_text = task_command.replies[-1]["text"]
    assert "26.07.2026" in preview_text
    assert "черновик задачи" in preview_text
    assert "напоминание ещё не настроено" in preview_text
    assert await counts(db) == (1, 0)


async def test_ambiguous_save_command_requires_button_confirmation(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_preview(bot, 1009, 2009, context)
    command = FakeMessage("Наверное, сохрани")
    await bot.text(update_for(command, 1009, 2009), context)
    assert command.replies[-1]["text"].startswith("Сохранить идею")
    labels = {
        button.text for row in command.replies[-1]["reply_markup"].inline_keyboard for button in row
    }
    assert labels == {"Да, сохранить", "Нет"}
    assert await counts(db) == (1, 0)


async def test_new_preview_is_focused_and_saves_despite_older_drafts(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(
        bot, 1101, 2101, context, "Еженедельное планирование", "Планировать неделю"
    )
    latest = await make_named_preview(
        bot,
        1101,
        2101,
        context,
        "Записывать победу дня",
        "Записывать одну победу дня каждый вечер",
    )
    snapshot = await bot.conversation.get(1101, 2101)
    assert snapshot.focused_draft_id is not None
    command = FakeMessage("сохрани")
    await bot.text(update_for(command, 1101, 2101), context)
    async with db.sessions() as session:
        saved = await session.scalar(select(InboxItem))
    assert saved.title == "Записывать победу дня"
    assert (2101, latest.message_id) in context.bot.removed


async def test_full_disambiguation_last_confirm_and_repeat_is_safe(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(
        bot, 1102, 2102, context, "Еженедельное планирование", "Планировать неделю"
    )
    await make_named_preview(
        bot,
        1102,
        2102,
        context,
        "Записывать победу дня",
        "Записывать одну победу дня каждый вечер",
    )
    await bot.conversation.clear_focus(1102, 2102)

    save = FakeMessage("сохрани")
    await bot.text(update_for(save, 1102, 2102), context)
    assert save.replies[-1]["text"] == "К какой карточке применить команду?"
    assert await counts(db) == (2, 0)

    selection = FakeMessage("последнюю")
    await bot.text(update_for(selection, 1102, 2102), context)
    assert selection.replies[-1]["text"] == ("Сохранить идею «Записывать победу дня»?")
    focused = await bot.conversation.get(1102, 2102)
    assert focused.pending_action == "save"
    assert focused.focused_draft_version == 1

    confirmation = FakeMessage("да, всё правильно")
    await bot.text(update_for(confirmation, 1102, 2102), context)
    async with db.sessions() as session:
        saved = list((await session.scalars(select(InboxItem))).all())
    assert [item.title for item in saved] == ["Записывать победу дня"]
    cleared = await bot.conversation.get(1102, 2102)
    assert cleared.focused_draft_id is None
    assert cleared.pending_action is None

    repeated = FakeMessage("сохрани")
    await bot.text(update_for(repeated, 1102, 2102), context)
    assert await counts(db) == (2, 1)
    assert repeated.replies[-1]["text"].startswith("Сохранить идею")


async def test_topic_selection_focuses_matching_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(
        bot, 1103, 2103, context, "Еженедельное планирование", "Планировать неделю"
    )
    await make_named_preview(
        bot,
        1103,
        2103,
        context,
        "Записывать победу дня",
        "Каждый вечер записывать победу дня",
    )
    await bot.conversation.clear_focus(1103, 2103)
    await bot.text(update_for(FakeMessage("сохрани"), 1103, 2103), context)
    choice = FakeMessage("про победу дня")
    await bot.text(update_for(choice, 1103, 2103), context)
    snapshot = await bot.conversation.get(1103, 2103)
    draft = await bot.draft_service.get(snapshot.focused_draft_id)
    assert draft.title == "Записывать победу дня"
    assert snapshot.pending_action == "save"


async def test_reply_to_preview_is_strongest_save_signal(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    first = await make_named_preview(
        bot, 1104, 2104, context, "Первая идея", "Первая подробная идея"
    )
    await make_named_preview(bot, 1104, 2104, context, "Вторая идея", "Вторая подробная идея")
    command = FakeMessage("сохрани", reply_to_message=first)
    await bot.text(update_for(command, 1104, 2104), context)
    async with db.sessions() as session:
        saved = await session.scalar(select(InboxItem))
    assert saved.title == "Первая идея"


async def test_control_language_and_duplicate_preview_create_nothing_new(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(
        bot, 1105, 2105, context, "Победа дня", "Записывать победу каждый вечер"
    )
    await make_named_preview(
        bot, 1105, 2105, context, "Победа дня", "Записывать победу каждый вечер"
    )
    before = await counts(db)
    for phrase in (
        "ну ты запишешь или как?",
        "я же сказал сохранить",
        "какую именно?",
        "да, её",
    ):
        await bot.text(update_for(FakeMessage(phrase), 1105, 2105), context)
    assert before == (1, 0)
    assert await counts(db) == before


async def test_expired_foreign_and_stale_drafts_are_rejected(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    message = await make_named_preview(
        bot, 1106, 2106, context, "Истекающая идея", "Старое содержание"
    )
    async with db.session() as session:
        draft = await session.scalar(select(DraftInboxItem))
        draft.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert await bot.draft_service.active_previews(1106, 2106) == []
    async with db.sessions() as session:
        expired = await session.scalar(select(DraftInboxItem))
    assert expired.status == "expired"

    active = await make_named_preview(bot, 1106, 2106, context, "Новая идея", "Новое содержание")
    callback = active.replies[-1]["reply_markup"].inline_keyboard[0][0].callback_data
    _, _, draft_id, raw_version = callback.split(":")
    stale_query = FakeCallbackQuery(f"draftfocus:save:{draft_id}:{int(raw_version) + 1}", active)
    await bot.draft_focus_action(
        SimpleNamespace(
            callback_query=stale_query,
            effective_user=SimpleNamespace(id=1106),
            effective_chat=SimpleNamespace(id=2106),
        ),
        context,
    )
    assert stale_query.answers[-1][1] is True

    foreign_query = FakeCallbackQuery(f"draftfocus:save:{draft_id}:{raw_version}", message)
    await bot.draft_focus_action(
        SimpleNamespace(
            callback_query=foreign_query,
            effective_user=SimpleNamespace(id=9999),
            effective_chat=SimpleNamespace(id=2106),
        ),
        context,
    )
    assert foreign_query.answers[-1][1] is True
    assert await counts(db) == (2, 0)


async def test_cancel_clears_pending_and_drafts_lists_only_owner_chat(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(bot, 1107, 2107, context, "Мой черновик", "Моё содержание")
    await make_named_preview(bot, 9998, 2107, context, "Чужой черновик", "Чужое содержание")
    await bot.conversation.set_pending_action(1107, 2107, "save")
    cancel = FakeMessage()
    await bot.cancel_draft_edit(update_for(cancel, 1107, 2107), context)
    snapshot = await bot.conversation.get(1107, 2107)
    assert snapshot.focused_draft_id is None
    assert snapshot.pending_action is None
    assert await counts(db) == (2, 0)

    listing = FakeMessage()
    await bot.drafts_command(update_for(listing, 1107, 2107), context)
    text = listing.replies[-1]["text"]
    assert "Мой черновик" in text
    assert "Чужой черновик" not in text


async def test_system_cleanup_never_reaches_intent_and_preserves_inbox_and_foreign(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    saved_preview = await make_named_preview(
        bot, 1201, 2201, context, "Сохранённая победа", "Сохранить победу дня"
    )
    await bot.text(update_for(FakeMessage("сохрани"), 1201, 2201), context)
    assert saved_preview is not None
    for index in range(3):
        await make_named_preview(
            bot,
            1201,
            2201,
            context,
            f"Мой черновик {index}",
            f"Несохранённое содержание {index}",
        )
    await make_named_preview(bot, 9997, 2201, context, "Чужой черновик", "Чужое содержание")
    routed_before = len(fake_ai.route_calls)
    request = FakeMessage(
        "Можешь список этих всех несохранённых задач удалить? Если что, я их вспомню"
    )
    await bot.text(update_for(request, 1201, 2201), context)
    assert request.replies[-1]["text"].startswith("Удалить 3 активных черновиков?")
    assert len(fake_ai.route_calls) == routed_before
    assert await active_count(db, 1201, 2201) == 3

    blocked_save = FakeMessage("сохрани")
    await bot.text(update_for(blocked_save, 1201, 2201), context)
    assert "Ожидаю отдельное подтверждение удаления" in blocked_save.replies[-1]["text"]
    assert await active_count(db, 1201, 2201) == 3

    confirmation = FakeMessage("Да-да, я имею в виду вот из черновиков хочу все удалить этот мусор")
    await bot.text(update_for(confirmation, 1201, 2201), context)
    assert confirmation.replies[-1]["text"] == (
        "Удалено 3 черновиков. Сохранённые записи не затронуты"
    )
    assert await active_count(db, 1201, 2201) == 0
    assert await active_count(db, 9997, 2201) == 1
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1
    assert len(fake_ai.route_calls) == routed_before


async def test_system_cleanup_cancel_expiry_and_repeat_are_safe(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    for index in range(2):
        await make_named_preview(
            bot,
            1202,
            2202,
            context,
            f"Черновик {index}",
            f"Содержание {index}",
        )
    request = FakeMessage("удали все черновики")
    await bot.text(update_for(request, 1202, 2202), context)
    cancel = FakeMessage("нет")
    await bot.text(update_for(cancel, 1202, 2202), context)
    assert await active_count(db, 1202, 2202) == 2

    await bot.text(update_for(FakeMessage("очисти черновики"), 1202, 2202), context)
    async with db.session() as session:
        state = await session.scalar(select(ConversationSession))
        state.system_action_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    expired = FakeMessage("да, удалить")
    await bot.text(update_for(expired, 1202, 2202), context)
    assert "Ожидаю отдельное подтверждение" in expired.replies[-1]["text"]
    assert await active_count(db, 1202, 2202) == 2

    prompt = FakeMessage("удали все черновики")
    await bot.text(update_for(prompt, 1202, 2202), context)
    callback = prompt.replies[-1]["reply_markup"].inline_keyboard[0][0].callback_data
    query = FakeCallbackQuery(callback, prompt)
    callback_update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=1202),
        effective_chat=SimpleNamespace(id=2202),
    )
    await bot.system_draft_action(callback_update, context)
    await bot.system_draft_action(callback_update, context)
    assert await active_count(db, 1202, 2202) == 0
    assert query.answers[-1][1] is True


async def test_cleanup_snapshot_change_aborts_without_partial_delete(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(bot, 1203, 2203, context, "Первый", "Первое содержание")
    request = FakeMessage("удали все черновики")
    await bot.text(update_for(request, 1203, 2203), context)
    await make_named_preview(bot, 1203, 2203, context, "Второй", "Второе содержание")
    confirmation = FakeMessage("подтверждаю удаление")
    await bot.text(update_for(confirmation, 1203, 2203), context)
    assert "Набор черновиков изменился" in confirmation.replies[-1]["text"]
    assert await active_count(db, 1203, 2203) == 2


async def test_drafts_pagination_groups_duplicates_and_isolates_user(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    user = await bot._user(1204)
    for index in range(5):
        await make_named_preview(
            bot,
            1204,
            2204,
            context,
            f"Уникальный {index}",
            f"Уникальное содержание {index}",
        )
    duplicate = ParsedThought(kind="idea", title="Повторяющаяся победа")
    for _ in range(2):
        await bot.draft_service.create(
            user_id=user.id,
            telegram_user_id=1204,
            chat_id=2204,
            source="text",
            raw_text="Записывать победу дня",
            parsed=duplicate,
        )
    await make_named_preview(bot, 9996, 2204, context, "Чужая карточка", "Чужое содержание")
    listing = FakeMessage()
    await bot.drafts_command(update_for(listing, 1204, 2204), context)
    text = listing.replies[-1]["text"]
    item_lines = [line for line in text.splitlines() if line[:1].isdigit() and ". " in line]
    assert len(item_lines) == 5
    assert "Активных черновиков: 7" in text
    assert "×2" in text
    assert "Чужая карточка" not in text
    labels = {
        button.text for row in listing.replies[-1]["reply_markup"].inline_keyboard for button in row
    }
    assert "Далее" in labels
    assert "Очистить активные" in labels
    next_data = next(
        button.callback_data
        for row in listing.replies[-1]["reply_markup"].inline_keyboard
        for button in row
        if button.text == "Далее"
    )
    query = FakeCallbackQuery(next_data, listing)
    await bot.drafts_action(
        SimpleNamespace(
            callback_query=query,
            effective_user=SimpleNamespace(id=1204),
            effective_chat=SimpleNamespace(id=2204),
        ),
        context,
    )
    assert "Страница 2/2" in listing.replies[-1]["text"]
    assert "Чужая карточка" not in listing.replies[-1]["text"]


def temporal_resolution(
    *,
    resolved_at: datetime = datetime(2026, 7, 20, 18, tzinfo=UTC),
    timezone: str = "Europe/Moscow",
    local_date: date = date(2026, 7, 20),
    local_time: time | None = time(21),
    precision: str = "datetime",
    original_expression: str = "вечером 20 июля",
) -> TemporalResolution:
    return TemporalResolution(
        resolved_at=resolved_at,
        timezone=timezone,
        resolved_local_date=local_date,
        resolved_local_time=local_time,
        precision=precision,
        original_expression=original_expression,
        resolution_status="resolved",
    )


async def test_drafts_group_same_canonical_semantics_despite_raw_audit_variation(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    user = await bot._user(1210)
    variants = (
        (
            "Записывать одну победу дня каждый вечер.",
            ParsedThought(
                kind="idea",
                title="Записывать одну победу дня каждый вечер",
                description="Каждый вечер записывать одну победу дня",
                next_step="Начать сегодня вечером",
                resolved_date=date(2026, 7, 20),
                temporal_resolution=temporal_resolution(
                    original_expression="вечером двадцатого июля"
                ),
            ),
        ),
        (
            "Я хочу каждый вечер записывать одну свою победу дня",
            ParsedThought(
                kind="idea",
                title="Записывать одну победу дня каждый вечер",
                description="Каждый вечер записывать одну победу дня",
                next_step="Открыть дневник после ужина",
                resolved_date=date(2026, 7, 20),
                temporal_resolution=temporal_resolution(
                    original_expression="20.07 в девять вечера"
                ),
            ),
        ),
    )
    for raw_text, parsed in variants:
        await bot.draft_service.create(
            user_id=user.id,
            telegram_user_id=1210,
            chat_id=2210,
            source="voice",
            raw_text=raw_text,
            parsed=parsed,
        )

    listing = FakeMessage()
    await bot.drafts_command(update_for(listing, 1210, 2210), context_with_bot())

    text = listing.replies[-1]["text"]
    item_lines = [line for line in text.splitlines() if line[:1].isdigit() and ". " in line]
    assert len(item_lines) == 1
    assert "[идея] Записывать одну победу дня каждый вечер ×2" in item_lines[0]
    assert await active_count(db, 1210, 2210) == 2


async def test_drafts_do_not_group_different_descriptions(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    user = await bot._user(1211)
    for description in (
        "Записывать личную победу дня",
        "Записывать победу команды за день",
    ):
        await bot.draft_service.create(
            user_id=user.id,
            telegram_user_id=1211,
            chat_id=2211,
            source="text",
            raw_text="Победа дня",
            parsed=ParsedThought(
                kind="idea",
                title="Победа дня",
                description=description,
            ),
        )

    listing = FakeMessage()
    await bot.drafts_command(update_for(listing, 1211, 2211), context_with_bot())

    text = listing.replies[-1]["text"]
    item_lines = [line for line in text.splitlines() if line[:1].isdigit() and ". " in line]
    assert len(item_lines) == 2
    assert "×" not in text


@pytest.mark.parametrize(
    "variant",
    [
        ParsedThought(
            kind="task",
            title="Победа дня",
            description="Записать победу дня",
            resolved_date=date(2026, 7, 20),
            temporal_resolution=temporal_resolution(),
        ),
        ParsedThought(
            kind="idea",
            title="Победа дня",
            description="Записать победу дня",
            resolved_date=date(2026, 7, 21),
            temporal_resolution=temporal_resolution(
                resolved_at=datetime(2026, 7, 21, 18, tzinfo=UTC),
                local_date=date(2026, 7, 21),
            ),
        ),
        ParsedThought(
            kind="idea",
            title="Победа дня",
            description="Записать победу дня",
            resolved_date=date(2026, 7, 20),
            temporal_resolution=temporal_resolution(
                resolved_at=datetime(2026, 7, 20, 19, tzinfo=UTC),
                local_time=time(22),
            ),
        ),
        ParsedThought(
            kind="idea",
            title="Победа дня",
            description="Записать победу дня",
            resolved_date=date(2026, 7, 20),
            temporal_resolution=temporal_resolution(timezone="Asia/Tbilisi"),
        ),
        ParsedThought(
            kind="idea",
            title="Победа дня",
            description="Записать победу дня",
            resolved_date=date(2026, 7, 20),
            temporal_resolution=temporal_resolution(
                resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
                local_time=None,
                precision="date",
            ),
        ),
    ],
    ids=("kind", "date", "time", "timezone", "precision"),
)
async def test_drafts_do_not_group_different_canonical_semantics(db, fake_ai, variant):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    user = await bot._user(1212)
    base = ParsedThought(
        kind="idea",
        title="Победа дня",
        description="Записать победу дня",
        resolved_date=date(2026, 7, 20),
        temporal_resolution=temporal_resolution(),
    )
    for parsed in (base, variant):
        await bot.draft_service.create(
            user_id=user.id,
            telegram_user_id=1212,
            chat_id=2212,
            source="text",
            raw_text="Записать победу дня",
            parsed=parsed,
        )

    listing = FakeMessage()
    await bot.drafts_command(update_for(listing, 1212, 2212), context_with_bot())

    text = listing.replies[-1]["text"]
    item_lines = [line for line in text.splitlines() if line[:1].isdigit() and ". " in line]
    assert len(item_lines) == 2
    assert "×" not in text


async def test_saved_receipt_and_last_saved_phrase_use_inbox(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(bot, 1205, 2205, context, "Победа дня", "Записать сегодняшнюю победу")
    save = FakeMessage("сохрани")
    await bot.text(update_for(save, 1205, 2205), context)
    assert "Сохранено в inbox:\nидея — Победа дня" in save.replies[-1]["text"]
    snapshot = await bot.conversation.get(1205, 2205)
    assert snapshot.last_saved_inbox_item_id is not None

    routed_before = len(fake_ai.route_calls)
    last = FakeMessage("что ты сохранил?")
    await bot.text(update_for(last, 1205, 2205), context)
    assert last.replies[-1]["text"] == ("Последняя сохранённая запись:\nидея — Победа дня")
    assert len(fake_ai.route_calls) == routed_before
    assert await counts(db) == (1, 1)


@pytest.mark.parametrize(
    ("phrase", "handler_name"),
    [
        ("покажи drafts", "drafts_command"),
        ("что в inbox?", "inbox"),
    ],
)
async def test_voice_natural_read_uses_same_slash_handler(
    db, fake_ai, monkeypatch, phrase, handler_name
):
    bot = FutureSelfBot(settings(), db, fake_ai, PhraseTranscription(phrase))
    calls = []

    async def handler(update, context):
        calls.append((update.effective_user.id, update.effective_chat.id))

    monkeypatch.setattr(bot, handler_name, handler)
    voice = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice, 1301, 2301), context_with_bot())
    assert calls == [(1301, 2301)]
    assert await counts(db) == (0, 0)
    assert fake_ai.route_calls == []


@pytest.mark.parametrize(
    ("phrase", "handler_name"),
    [
        ("  ЧТО   У МЕНЯ СОХРАНЕНО?!  ", "inbox"),
        ("Что сохранилось?", "last_saved_command"),
        ("Что сохранилось последним...", "last_saved_command"),
        ("Какие у тебя есть команды?", "help_command"),
        ("Покажи мои черновики", "drafts_command"),
        ("Покажи фокус дня", "today"),
    ],
)
@pytest.mark.parametrize("source", ["text", "voice"])
async def test_natural_read_commands_are_identical_for_text_and_voice(
    db, fake_ai, monkeypatch, phrase, handler_name, source
):
    transcription = PhraseTranscription(phrase) if source == "voice" else NoopTranscription()
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    calls = []

    async def handler(update, context):
        calls.append((update.effective_user.id, update.effective_chat.id))

    monkeypatch.setattr(bot, handler_name, handler)
    message = FakeMessage(voice=FakeVoice()) if source == "voice" else FakeMessage(phrase)
    update = update_for(message, 1310, 2310)
    if source == "voice":
        await bot.voice(update, context_with_bot())
    else:
        await bot.text(update, context_with_bot())

    assert calls == [(1310, 2310)]
    assert fake_ai.route_calls == []
    assert await counts(db) == (0, 0)


@pytest.mark.parametrize("command", ["Сохрани инбокс", "Сохрани в инбокс", "Сохрани в inbox"])
@pytest.mark.parametrize("source", ["text", "voice"])
async def test_natural_save_inbox_confirms_focused_draft_once(db, fake_ai, command, source):
    transcription = PhraseTranscription(command) if source == "voice" else NoopTranscription()
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    context = context_with_bot()
    await make_named_preview(
        bot,
        1311,
        2311,
        context,
        "Победа дня",
        "Записывать одну победу дня каждый вечер",
    )
    routed_before = len(fake_ai.route_calls)

    first = FakeMessage(voice=FakeVoice()) if source == "voice" else FakeMessage(command)
    second = FakeMessage(voice=FakeVoice()) if source == "voice" else FakeMessage(command)
    route = bot.voice if source == "voice" else bot.text
    await route(update_for(first, 1311, 2311), context)
    await route(update_for(second, 1311, 2311), context)

    assert (
        f"Сохранено в inbox по {'голосовой' if source == 'voice' else 'текстовой'} команде"
        in (first.replies[-1]["text"])
    )
    assert "Нет одной актуальной" in second.replies[-1]["text"]
    assert await counts(db) == (1, 1)
    assert len(fake_ai.route_calls) == routed_before


async def test_natural_save_inbox_does_not_guess_between_drafts_or_create_preview(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(bot, 1312, 2312, context, "Первый", "Первое содержание")
    await make_named_preview(bot, 1312, 2312, context, "Второй", "Второе содержание")
    await bot.conversation.clear_focus(1312, 2312)
    routed_before = len(fake_ai.route_calls)

    command = FakeMessage("Сохрани в инбокс")
    await bot.text(update_for(command, 1312, 2312), context)

    assert command.replies[-1]["text"] == "К какой карточке применить команду?"
    assert await counts(db) == (2, 0)
    assert len(fake_ai.route_calls) == routed_before


async def test_natural_inbox_is_isolated_and_write_phrase_is_not_read_command(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    await make_named_preview(bot, 1302, 2302, context, "Моя запись", "Моё содержание")
    await bot.text(update_for(FakeMessage("сохрани"), 1302, 2302), context)
    await make_named_preview(bot, 9995, 2302, context, "Чужая запись", "Чужое содержание")
    await bot.text(update_for(FakeMessage("сохрани"), 9995, 2302), context)

    show = FakeMessage("покажи сохранённые записи")
    routed_before = len(fake_ai.route_calls)
    await bot.text(update_for(show, 1302, 2302), context)
    assert "Моя запись" in show.replies[-1]["text"]
    assert "Чужая запись" not in show.replies[-1]["text"]
    assert len(fake_ai.route_calls) == routed_before

    write = FakeMessage("сохрани в inbox")
    await bot.text(update_for(write, 1302, 2302), context)
    assert "Inbox пока пуст" not in write.replies[-1]["text"]
    assert "Нет одной актуальной" in write.replies[-1]["text"]


async def test_conflict_choice_regenerates_all_temporal_fields_and_survives_restart(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    context = context_with_bot()
    original = (
        "Мне 18 июля 2026, в воскресенье, в 18:00 нужно будет на стрижку. Напомни, пожалуйста"
    )
    conflict = FakeMessage(original)
    await bot.text(update_for(conflict, 1303, 2303), context)
    assert "18.07.2026" in conflict.replies[-1]["text"]
    assert "19.07.2026" in conflict.replies[-1]["text"]

    choice = FakeMessage("19 июля")
    await bot.text(update_for(choice, 1303, 2303), context)
    preview = choice.replies[-1]["text"]
    assert "Стрижка — 19 июля в 18:00" in preview
    assert "19.07.2026 18:00" in preview
    assert "Europe/Moscow" in preview
    assert "18 июля" not in preview
    assert "18.07.2026" not in preview

    async with db.sessions() as session:
        draft = await session.scalar(select(DraftInboxItem))
    current_fields = " ".join(
        value
        for value in (draft.title, draft.description, draft.raw_text, draft.next_step)
        if value
    )
    assert "18 июля" not in current_fields
    assert "18.07.2026" not in current_fields
    assert draft.title == "Стрижка — 19 июля в 18:00"
    temporal = draft.temporal_resolution
    assert temporal["timezone"] == "Europe/Moscow"
    assert temporal["resolved_local_date"] == "2026-07-19"
    assert temporal["resolved_local_time"] == "18:00:00"
    assert temporal["precision"] == "datetime"
    assert temporal["resolution_status"] == "resolved"
    assert "18 июля" in temporal["original_expression"]
    resolved_at = temporal["resolved_at"]

    restarted = FutureSelfBot(settings(), db, fake_ai, NoopTranscription())
    persisted = await restarted.draft_service.get(draft.id)
    assert persisted.temporal_resolution["resolved_at"] == resolved_at
    assert persisted.title == "Стрижка — 19 июля в 18:00"

    save = FakeMessage("сохрани")
    await restarted.text(update_for(save, 1303, 2303), context)
    receipt = save.replies[-1]["text"]
    assert "задача — Стрижка — 19 июля в 18:00" in receipt
    assert "Задача сохранена. Автоматическое напоминание пока не настроено" in receipt
    assert "напоминание создано" not in receipt.lower()
    async with db.sessions() as session:
        item = await session.scalar(select(InboxItem))
    assert item.temporal_resolution["resolved_at"] == resolved_at
