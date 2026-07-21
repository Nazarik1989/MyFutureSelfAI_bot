from asyncio import gather
from datetime import date
from types import SimpleNamespace

import pytest
from autotester.fakes import (
    FakeCallbackQuery,
    FakeMessage,
    FakeVoice,
    ScriptedTranscription,
)
from sqlalchemy import func, select
from telegram.ext import ApplicationHandlerStop

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import InboxItem, TaskReminder, TaskState, VisionDraft, VisionItem
from future_self.vision import CATEGORY_META


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def update_for(message, *, user_id=7001, chat_id=17001, chat_type="private"):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=None,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
    )


def callback_update(data, message, *, user_id=7001, chat_id=17001):
    query = FakeCallbackQuery(data, message)
    update = SimpleNamespace(
        effective_message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
    )
    return update, query


def callback_from(message, prefix: str) -> str:
    for reply in reversed(message.replies):
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        for row in markup.inline_keyboard:
            for button in row:
                if button.callback_data and button.callback_data.startswith(prefix):
                    return button.callback_data
    raise AssertionError(f"Missing callback {prefix!r}")


async def start_draft(bot, *, user_id=7001, chat_id=17001):
    menu = FakeMessage("/vision")
    await bot.vision_command(update_for(menu, user_id=user_id, chat_id=chat_id), None)
    add_update, _ = callback_update(
        callback_from(menu, "vision:add"), menu, user_id=user_id, chat_id=chat_id
    )
    await bot.vision_action(add_update, None)
    category_update, _ = callback_update(
        callback_from(menu, "vision:cat:"),
        menu,
        user_id=user_id,
        chat_id=chat_id,
    )
    await bot.vision_action(category_update, None)
    return menu


async def create_item(
    bot,
    wish: str,
    *,
    user_id=7001,
    chat_id=17001,
    first_step="Сделать первый шаг",
):
    await start_draft(bot, user_id=user_id, chat_id=chat_id)
    user = await bot._user(user_id)
    await bot.vision_service.consume_text(user.id, chat_id, wish)
    draft = await bot.vision_service.draft(user.id, chat_id)
    await bot.vision_service.skip(user.id, chat_id, draft.id, draft.version)
    draft = await bot.vision_service.draft(user.id, chat_id)
    await bot.vision_service.skip(user.id, chat_id, draft.id, draft.version)
    if first_step is None:
        draft = await bot.vision_service.draft(user.id, chat_id)
        await bot.vision_service.skip(user.id, chat_id, draft.id, draft.version)
    else:
        await bot.vision_service.consume_text(user.id, chat_id, first_step)
    draft = await bot.vision_service.draft(user.id, chat_id)
    return await bot.vision_service.confirm(user.id, chat_id, draft.id, draft.version)


async def test_full_voice_skip_preview_confirm_is_idempotent_and_no_llm(db, fake_ai):
    transcription = ScriptedTranscription()
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    menu = await start_draft(bot)

    transcription.queue("Побывать у океана")
    voice = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice), SimpleNamespace(user_data={}))
    assert transcription.calls
    skip_why, _ = callback_update(callback_from(voice, "vision:skip:"), voice)
    await bot.vision_action(skip_why, None)
    skip_date, _ = callback_update(callback_from(voice, "vision:skip:"), voice)
    await bot.vision_action(skip_date, None)

    step = FakeMessage("Открыть календарь и выбрать неделю")
    await bot.text(update_for(step), SimpleNamespace(user_data={}))
    confirm_data = callback_from(step, "vision:confirm:")
    confirm_update, confirm_query = callback_update(confirm_data, step)
    await bot.vision_action(confirm_update, None)
    assert any("Желание сохранено" in edit for edit in confirm_query.edits)

    repeated_update, repeated_query = callback_update(confirm_data, step)
    await bot.vision_action(repeated_update, None)
    assert any(text and "устарело" in text for text, _show_alert in repeated_query.answers)
    assert fake_ai.route_calls == []
    async with db.sessions() as session:
        items = list((await session.scalars(select(VisionItem))).all())
        assert len(items) == 1
        assert items[0].wish_text == "Побывать у океана"
        assert items[0].why_text is None
        assert items[0].target_date is None
        assert await session.scalar(select(func.count(VisionDraft.id))) == 0
    assert callback_from(menu, "vision:add") == "vision:add"


async def test_text_date_and_cancel_do_not_save_partial_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await start_draft(bot)
    for value in ("Создать уютный дом", "Чтобы семье было спокойно", "31.12.2030"):
        await bot.text(update_for(FakeMessage(value)), SimpleNamespace(user_data={}))
    cancel = FakeMessage("/cancel")
    await bot.cancel_draft_edit(update_for(cancel), SimpleNamespace(user_data={}))
    assert "ничего не сохранено" in cancel.replies[-1]["text"]
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(VisionItem.id))) == 0
        assert await session.scalar(select(func.count(VisionDraft.id))) == 0


async def test_draft_survives_bot_restart_and_vision_resumes(db, fake_ai):
    first_bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await start_draft(first_bot)
    await first_bot.text(
        update_for(FakeMessage("Освоить акварель")),
        SimpleNamespace(user_data={}),
    )
    second_bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    resume = FakeMessage("/vision")
    await second_bot.vision_command(update_for(resume), None)
    output = "\n".join(reply["text"] for reply in resume.replies)
    assert "незавершённая" in output
    assert "Почему это важно" in output


async def test_early_text_gate_routes_active_vision_before_other_conversations(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await start_draft(bot)
    message = FakeMessage("Текст для карты, а не для onboarding")
    with pytest.raises(ApplicationHandlerStop):
        await bot.vision_text_gate(
            update_for(message),
            SimpleNamespace(user_data={"unrelated_conversation": "paused"}),
        )
    user = await bot._user(7001)
    draft = await bot.vision_service.draft(user.id, 17001)
    assert draft.wish_text == "Текст для карты, а не для onboarding"
    assert fake_ai.route_calls == []


async def test_draft_cannot_leak_to_another_chat_or_be_overwritten_by_edit(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await start_draft(bot, user_id=7051, chat_id=17051)
    user = await bot._user(7051)

    with pytest.raises(ValueError, match="другом чате"):
        await bot.vision_service.begin(user.id, 27051)

    async with db.session() as session:
        item = VisionItem(
            owner_id=user.id,
            category="home",
            wish_text="Существующая карточка",
            status="active",
        )
        session.add(item)
        await session.flush()
        item_id = item.id
    outcome = await bot.vision_service.start_edit(user.id, 17051, item_id, "wish")
    assert outcome.status == "busy"
    draft = await bot.vision_service.draft(user.id, 17051)
    assert draft is not None
    assert draft.step == "wish"


async def test_two_owners_can_have_identical_wishes_and_forged_callback_is_private(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    first = await create_item(bot, "Одинаковое приватное желание", user_id=7101, chat_id=17101)
    second = await create_item(bot, "Одинаковое приватное желание", user_id=7102, chat_id=17102)
    assert first.item.owner_id != second.item.owner_id

    forged_message = FakeMessage()
    forged_update, forged_query = callback_update(
        f"vision:view:{first.item.id}",
        forged_message,
        user_id=7102,
        chat_id=17102,
    )
    await bot.vision_action(forged_update, None)
    assert forged_message.replies == []
    assert any(text and "недоступна" in text for text, _show_alert in forged_query.answers)
    own_items, own_total = await bot.vision_service.page(second.item.owner_id, "active", 0)
    assert own_total == 1
    assert [item.id for item in own_items] == [second.item.id]


async def test_foreign_draft_callback_cannot_advance_or_reveal_private_state(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(7151)
    draft = await bot.vision_service.begin(owner.id, 17151)
    forged_data = f"vision:cat:{draft.id}:{draft.version}:travel"

    forged_update, forged_query = callback_update(
        forged_data,
        FakeMessage(),
        user_id=7152,
        chat_id=17152,
    )
    await bot.vision_action(forged_update, None)

    unchanged = await bot.vision_service.draft(owner.id, 17151)
    assert unchanged is not None
    assert unchanged.step == "category"
    assert unchanged.category is None
    assert any(text and "недоступна" in text for text, _show_alert in forged_query.answers)


async def test_status_edit_archive_delete_and_pagination_are_owner_scoped(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    user = await bot._user(7201)
    for index in range(6):
        async with db.session() as session:
            session.add(
                VisionItem(
                    owner_id=user.id,
                    category="money" if index < 3 else "travel",
                    wish_text=f"Желание {index}",
                    status="active",
                )
            )
    first_page, total = await bot.vision_service.page(user.id, "active", 0)
    second_page, _ = await bot.vision_service.page(user.id, "active", 1)
    assert total == 6
    assert len(first_page) == 5
    assert len(second_page) == 1
    assert await bot.vision_service.category_counts(user.id, "active") == {
        "money": 3,
        "travel": 3,
    }

    item = first_page[0]
    assert (await bot.vision_service.set_status(user.id, item.id, "achieved")).status == "achieved"
    edit = await bot.vision_service.start_edit(user.id, 17201, item.id, "wish")
    assert edit.status == "editing"
    edited = await bot.vision_service.consume_text(user.id, 17201, "Новое желание")
    assert edited.status == "edited"
    assert edited.item.wish_text == "Новое желание"
    assert (await bot.vision_service.set_status(user.id, item.id, "archived")).status == "archived"
    assert await bot.vision_service.delete_item(user.id, item.id) is True
    assert await bot.vision_service.delete_item(user.id, item.id) is False


async def test_all_editable_fields_use_owned_persistent_business_logic(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_item(bot, "Начальная формулировка")
    owner_id = created.item.owner_id
    item_id = created.item.id

    why = await bot.vision_service.start_edit(owner_id, 17001, item_id, "why")
    assert why.status == "editing"
    assert (
        await bot.vision_service.consume_text(owner_id, 17001, "Это действительно важно")
    ).status == "edited"

    target = await bot.vision_service.start_edit(owner_id, 17001, item_id, "target_date")
    assert target.status == "editing"
    assert (await bot.vision_service.consume_text(owner_id, 17001, "31.12.2030")).status == "edited"

    category = await bot.vision_service.start_edit(owner_id, 17001, item_id, "category")
    assert category.status == "editing"
    changed = await bot.vision_service.choose_category(
        owner_id,
        17001,
        "health_energy",
        draft_id=category.draft.id,
    )
    assert changed.status == "edited"
    assert changed.item.category == "health_energy"

    clear_why = await bot.vision_service.start_edit(owner_id, 17001, item_id, "why")
    cleared = await bot.vision_service.skip(
        owner_id,
        17001,
        clear_why.draft.id,
        clear_why.draft.version,
    )
    assert cleared.status == "edited"
    assert cleared.item.why_text is None
    assert cleared.item.target_date == date(2030, 12, 31)


async def test_archive_is_listable_and_restorable_and_invalid_list_callback_is_stale(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_item(bot, "Сохранить в архиве")
    item = await bot.vision_service.set_status(created.item.owner_id, created.item.id, "archived")
    archived, total = await bot.vision_service.page(item.owner_id, "archived", 0)
    assert total == 1
    assert archived[0].id == item.id

    restored = await bot.vision_service.set_status(item.owner_id, item.id, "active")
    assert restored.status == "active"

    message = FakeMessage()
    update, query = callback_update("vision:list:not-a-status:0", message)
    await bot.vision_action(update, None)
    assert message.replies == []
    assert any(text and "устарело" in text for text, _show_alert in query.answers)


async def test_delete_requires_exact_persistent_confirmation_and_is_idempotent(db, fake_ai):
    first_bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_item(first_bot, "Удалить только после подтверждения")
    item_id = created.item.id
    owner_id = created.item.owner_id

    forged_message = FakeMessage()
    forged_update, forged_query = callback_update(
        f"vision:delete:{item_id}:999999:1",
        forged_message,
    )
    await first_bot.vision_action(forged_update, None)
    assert await first_bot.vision_service.get_item(owner_id, item_id) is not None
    assert any(text and "устарело" in text for text, _show_alert in forged_query.answers)

    card = FakeMessage()
    await first_bot._vision_send_item(card, created.item)
    ask_update, _ = callback_update(callback_from(card, "vision:deleteask:"), card)
    await first_bot.vision_action(ask_update, None)
    confirm_data = callback_from(card, "vision:delete:")

    restarted_bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    resume = FakeMessage("/vision")
    await restarted_bot.vision_command(update_for(resume), None)
    assert "Удаление ожидает явного подтверждения" in "\n".join(
        reply["text"] for reply in resume.replies
    )

    unrelated = FakeMessage("Это не должно изменить карточку")
    with pytest.raises(ApplicationHandlerStop):
        await restarted_bot.vision_text_gate(
            update_for(unrelated),
            SimpleNamespace(user_data={}),
        )
    assert await restarted_bot.vision_service.get_item(owner_id, item_id) is not None
    assert fake_ai.route_calls == []

    confirm_update, confirm_query = callback_update(confirm_data, card)
    await restarted_bot.vision_action(confirm_update, None)
    assert any("Карточка удалена" in edit for edit in confirm_query.edits)
    assert await restarted_bot.vision_service.get_item(owner_id, item_id) is None

    repeated_update, repeated_query = callback_update(confirm_data, card)
    await restarted_bot.vision_action(repeated_update, None)
    assert any(text and "устарело" in text for text, _show_alert in repeated_query.answers)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(VisionDraft.id))) == 0


async def test_delete_cancel_and_foreign_callbacks_cannot_change_owner_item(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_item(bot, "Приватная карточка", user_id=7251, chat_id=17251)
    item_id = created.item.id

    card = FakeMessage()
    await bot._vision_send_item(card, created.item)
    ask_update, _ = callback_update(
        callback_from(card, "vision:deleteask:"),
        card,
        user_id=7251,
        chat_id=17251,
    )
    await bot.vision_action(ask_update, None)
    confirm_data = callback_from(card, "vision:delete:")
    cancel_data = callback_from(card, "vision:deletecancel:")

    foreign_update, foreign_query = callback_update(
        confirm_data,
        FakeMessage(),
        user_id=7252,
        chat_id=17252,
    )
    await bot.vision_action(foreign_update, None)
    assert any(text and "устарело" in text for text, _show_alert in foreign_query.answers)
    assert await bot.vision_service.get_item(created.item.owner_id, item_id) is not None

    cancel_update, cancel_query = callback_update(
        cancel_data,
        card,
        user_id=7251,
        chat_id=17251,
    )
    await bot.vision_action(cancel_update, None)
    assert any("Удаление отменено" in edit for edit in cancel_query.edits)
    assert await bot.vision_service.get_item(created.item.owner_id, item_id) is not None

    stale_update, stale_query = callback_update(
        confirm_data,
        card,
        user_id=7251,
        chat_id=17251,
    )
    await bot.vision_action(stale_update, None)
    assert any(text and "устарело" in text for text, _show_alert in stale_query.answers)


async def test_create_task_is_atomic_idempotent_and_never_creates_reminder(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_item(bot, "Запустить проект", first_step="Написать план")
    item = created.item
    results = await gather(
        bot.vision_service.create_task(item.owner_id, item.id),
        bot.vision_service.create_task(item.owner_id, item.id),
    )
    assert {result.status for result in results} == {"created", "existing"}
    async with db.sessions() as session:
        tasks = list((await session.scalars(select(InboxItem))).all())
        assert len(tasks) == 1
        assert tasks[0].source == "vision"
        assert tasks[0].next_step == "Написать план"
        assert await session.scalar(select(func.count(TaskReminder.id))) == 0
        assert await session.scalar(select(func.count(TaskState.id))) == 1


async def test_missing_first_step_does_not_create_task(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    created = await create_item(bot, "Новая привычка", first_step=None)
    result = await bot.vision_service.create_task(created.item.owner_id, created.item.id)
    assert result.status == "missing_step"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 0


async def test_group_guard_blocks_vision_before_content_is_processed(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    message = FakeMessage("/vision")
    update = update_for(message, chat_type="group")
    with pytest.raises(ApplicationHandlerStop):
        await bot.private_chat_guard(update, None)
    assert "только в личном чате" in message.replies[-1]["text"]


def test_target_date_parser_rejects_past_and_accepts_supported_formats():
    from future_self.vision import parse_target_date

    assert parse_target_date("31.12.2030", today=date(2026, 1, 1)) == date(2030, 12, 31)
    assert parse_target_date("2030-12-31", today=date(2026, 1, 1)) == date(2030, 12, 31)
    with pytest.raises(ValueError, match="будущем"):
        parse_target_date("01.01.2020", today=date(2026, 1, 1))


def test_vision_categories_have_stable_machine_codes_and_russian_labels():
    assert CATEGORY_META == {
        "health_energy": ("🌿", "Здоровье и энергия"),
        "relationships_family": ("❤️", "Отношения и семья"),
        "work_purpose": ("💼", "Работа и предназначение"),
        "money": ("💰", "Деньги"),
        "home": ("🏡", "Дом"),
        "travel": ("✈️", "Путешествия"),
        "growth_creativity": ("🎨", "Развитие и творчество"),
        "other": ("✨", "Другое"),
    }
