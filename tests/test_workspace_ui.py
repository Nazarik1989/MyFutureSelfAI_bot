import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from autotester.fakes import (
    FakeBot,
    FakeCallbackQuery,
    FakeMessage,
    FakeVoice,
    ScriptedTranscription,
)
from telegram.ext import CallbackQueryHandler, CommandHandler, ConversationHandler

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.navigation import (
    navigation_actions,
    navigation_sections,
    public_commands,
    validate_catalog,
)
from future_self.workspace_access import InvitationPreview
from future_self.workspace_handlers import (
    CHARACTER_LABELS,
    INVITATION_TEMPLATES,
    PRIVACY_FOOTER,
    WorkspaceHandlers,
)


class WorkspaceFakeBot(FakeBot):
    username = "future_self_test_bot"


def settings(*, enabled: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        database_url="sqlite+aiosqlite:///:memory:",
        enable_workspace_access=enabled,
    )


def context(*, args: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(user_data={}, args=args or [], bot=WorkspaceFakeBot())


def update_for(
    message: FakeMessage,
    *,
    user_id: int = 880001,
    chat_id: int | None = None,
    query: FakeCallbackQuery | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id or user_id, type="private"),
    )


def callback_by_label(message: FakeMessage, label: str, *, contains: bool = False) -> str:
    for reply in reversed(message.replies):
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        for row in markup.inline_keyboard:
            for button in row:
                if button.callback_data and (
                    button.text == label or (contains and label in button.text)
                ):
                    return button.callback_data
    raise AssertionError(f"Missing callback for {label!r}")


async def click(
    bot: FutureSelfBot,
    message: FakeMessage,
    label: str,
    *,
    user_id: int = 880001,
    chat_id: int | None = None,
    contains: bool = False,
    ctx: SimpleNamespace | None = None,
) -> FakeCallbackQuery:
    data = callback_by_label(message, label, contains=contains)
    query = FakeCallbackQuery(data, message)
    await bot.workspace_callback(
        update_for(message, user_id=user_id, chat_id=chat_id, query=query),
        ctx or context(),
    )
    return query


async def create_pair_workspace(
    bot: FutureSelfBot,
    *,
    user_id: int = 880001,
    name: str = "Наше будущее",
) -> FakeMessage:
    hub = FakeMessage("/spaces")
    await bot.spaces_command(update_for(hub, user_id=user_id), context())
    await click(bot, hub, "＋ Создать", user_id=user_id)
    await click(bot, hub, "Для пары", user_id=user_id)
    if name == "Наше будущее":
        await click(bot, hub, name, user_id=user_id)
    else:
        await click(bot, hub, "Своё название", user_id=user_id)
        title = FakeMessage(name)
        assert await bot.workspace_pending_text(update_for(title, user_id=user_id), name, "text")
    description = FakeMessage("-")
    assert await bot.workspace_pending_text(update_for(description, user_id=user_id), "-", "text")
    return description


@pytest.mark.autotester
async def test_workspace_catalog_is_flag_aware_and_has_no_dead_disabled_surface(db, fake_ai):
    disabled = FutureSelfBot(settings(enabled=False), db, fake_ai, ScriptedTranscription())
    enabled = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())

    assert "spaces" not in {item.command for item in public_commands(False)}
    assert "spaces" not in navigation_sections(False)
    assert "spaces" not in navigation_actions(False)
    assert disabled.natural_command_router.route("покажи мои пространства") is None

    validate_catalog(True)
    assert "spaces" in {item.command for item in public_commands(True)}
    assert "spaces" in navigation_sections(True)
    assert enabled.natural_command_router.route("покажи мои пространства").action == "show_spaces"
    assert enabled.natural_command_router.route("создай совместное пространство").action == (
        "create_space"
    )

    disabled_handlers = disabled.build().handlers[0]
    enabled_handlers = enabled.build().handlers[0]
    assert not any(
        isinstance(handler, CommandHandler) and "spaces" in handler.commands
        for handler in disabled_handlers
    )
    assert any(
        isinstance(handler, CommandHandler) and "spaces" in handler.commands
        for handler in enabled_handlers
    )
    assert not any(
        isinstance(handler, CallbackQueryHandler)
        and getattr(handler, "pattern", None)
        and "space" in str(handler.pattern)
        for handler in disabled_handlers
    )


@pytest.mark.autotester
async def test_create_flow_and_pending_input_survive_restart_without_llm(db, fake_ai):
    first = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    hub = FakeMessage("/spaces")
    await first.spaces_command(update_for(hub), context())
    await click(first, hub, "＋ Создать")
    await click(first, hub, "Для семьи")
    await click(first, hub, "Своё название")

    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    title = FakeMessage("<b>Наш & дом</b>")
    assert await restarted.workspace_pending_text(update_for(title), title.text, "text")
    description = FakeMessage("Короткое <i>описание</i>")
    assert await restarted.workspace_pending_text(update_for(description), description.text, "text")
    reply = description.replies[-1]
    assert "&lt;b&gt;Наш &amp; дом&lt;/b&gt;" in reply["text"]
    assert "&lt;i&gt;описание&lt;/i&gt;" in reply["text"]
    labels = {button.text for row in reply["reply_markup"].inline_keyboard for button in row}
    assert {"Участники", "Проекты", "Пригласить", "Переименовать"} <= labels
    assert not any(
        marker in " ".join(labels).lower()
        for marker in ("knowledge", "council", "загрузить pdf", "материалы")
    )
    assert all(
        len(button.text) <= 64 for row in reply["reply_markup"].inline_keyboard for button in row
    )
    assert fake_ai.route_calls == []


@pytest.mark.autotester
async def test_invite_preview_edit_confirm_and_deep_link_accept_survive_restart(db, fake_ai):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    card = await create_pair_workspace(owner)
    await click(owner, card, "Пригласить")
    await click(owner, card, "Редактор")
    await click(owner, card, "Вариант 1")
    assert "Личные записи и визуализации" in card.replies[-1]["text"]
    await click(owner, card, "Изменить текст")

    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    custom = FakeMessage("Вместе <b>строим</b> планы")
    assert await restarted.workspace_pending_text(update_for(custom), custom.text, "text")
    preview = custom.replies[-1]
    assert "&lt;b&gt;строим&lt;/b&gt;" in preview["text"]

    await click(
        FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription()),
        custom,
        "Поделиться приглашением",
        ctx=context(),
    )
    issued_text = custom.replies[-1]["text"]
    match = re.search(r"[?&]start=space_([A-Za-z0-9_-]+)", issued_text)
    assert match is not None
    raw_token = match.group(1)
    assert "Пересланную ссылку" in issued_text

    recipient = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    invitation = FakeMessage("/start")
    result = await recipient.start(
        update_for(invitation, user_id=880002),
        context(args=[f"space_{raw_token}"]),
    )
    assert result == ConversationHandler.END
    assert "Вместе &lt;b&gt;строим&lt;/b&gt; планы" in invitation.replies[-1]["text"]
    await click(recipient, invitation, "Подробнее", user_id=880002)
    assert "Health, Doctor и Labs не передаются" in invitation.replies[-1]["text"]
    await click(recipient, invitation, "Присоединиться", user_id=880002)
    assert "Ты присоединился" in invitation.replies[-1]["text"]
    assert fake_ai.route_calls == []


@pytest.mark.autotester
async def test_workspace_natural_and_voice_routes_are_deterministic_without_llm(db, fake_ai):
    transcription = ScriptedTranscription()
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    natural = FakeMessage("создай совместное пространство")
    await bot.text(update_for(natural), context())
    assert callback_by_label(natural, "Для команды").startswith("space:")

    await bot.workspace_service.cancel_input((await bot._user(880001)).id, 880001)
    transcription.queue("покажи мои пространства")
    voice = FakeMessage(voice=FakeVoice())
    await bot.voice(update_for(voice), context())
    assert any("Совместные пространства" in reply["text"] for reply in voice.replies)
    assert fake_ai.route_calls == []


async def test_project_context_archive_restore_and_cancel_are_reachable(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    card = await create_pair_workspace(bot, name="A" * 100)
    await click(bot, card, "Проекты")
    await click(bot, card, "＋ Проект")
    project_name = "P" * 100
    project_input = FakeMessage(project_name)
    assert await bot.workspace_pending_text(update_for(project_input), project_name, "text")
    await click(bot, project_input, "P" * 20, contains=True)
    await click(bot, project_input, "Выбрать контекст проекта")
    assert "Выбран контекст проекта" in project_input.replies[-1]["text"]
    await click(bot, project_input, "Архивировать")
    assert "Проект архивирован" in project_input.replies[-1]["text"]
    await click(bot, project_input, "Архив проектов")
    await click(bot, project_input, "P" * 20, contains=True)
    assert "Проект восстановлен" in project_input.replies[-1]["text"]

    await click(bot, project_input, "＋ Проект")
    cancel = FakeMessage("/cancel")
    await bot.cancel_draft_edit(update_for(cancel), context())
    assert "Операция с пространством отменена" in cancel.replies[-1]["text"]
    assert await bot.workspace_service.pending_input((await bot._user(880001)).id, 880001) is None


@pytest.mark.parametrize("character", tuple(CHARACTER_LABELS))
def test_all_character_invitation_templates_are_complete_and_have_privacy_footer(character):
    templates = INVITATION_TEMPLATES[character]
    assert len(templates) == 4
    for index, template in enumerate(templates, start=1):
        rendered = WorkspaceHandlers._format_invitation(
            template, "Приглашающий", "Общее пространство"
        )
        assert "{inviter}" not in rendered
        assert "{workspace}" not in rendered
        preview = InvitationPreview(
            inviter_display_name="Приглашающий",
            workspace_name="Общее пространство",
            character=character,
            role="editor",
            template_key=f"{character}_{index}",
            custom_text=None,
            expires_at=datetime.now(UTC) + timedelta(days=1),
            version=1,
        )
        recipient = WorkspaceHandlers._incoming_invitation_text(preview)
        assert rendered in recipient
        assert PRIVACY_FOOTER in recipient


async def test_invalid_workspace_input_is_rearmed_and_can_finish_after_restart(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    hub = FakeMessage("/spaces")
    await bot.spaces_command(update_for(hub), context())
    await click(bot, hub, "＋ Создать")
    await click(bot, hub, "Свой вариант")
    await click(bot, hub, "Своё название")

    invalid = FakeMessage("   ")
    assert await bot.workspace_pending_text(update_for(invalid), invalid.text, "text")
    assert "не может быть пустым" in invalid.replies[-1]["text"]
    user = await bot._user(880001)
    assert await bot.workspace_service.pending_input(user.id, 880001) is not None

    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    valid = FakeMessage("Совместный дом")
    assert await restarted.workspace_pending_text(update_for(valid), valid.text, "text")
    description = FakeMessage("-")
    assert await restarted.workspace_pending_text(update_for(description), description.text, "text")
    assert "Пространство создано" in description.replies[-1]["text"]
