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
from telegram.error import TelegramError
from telegram.ext import CallbackQueryHandler, CommandHandler, ConversationHandler

from future_self.access import AccessService
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


class DirectDeliveryBot(WorkspaceFakeBot):
    def __init__(self, *, fail: bool = False, ambiguous: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.ambiguous = ambiguous
        self.sent: list[dict[str, object]] = []

    async def send_message(self, **kwargs: object) -> SimpleNamespace:
        if self.fail:
            raise TelegramError("delivery unavailable")
        self.sent.append(dict(kwargs))
        if self.ambiguous:
            raise RuntimeError("ambiguous transport result")
        return SimpleNamespace(message_id=99001)


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


def delivery_context(bot: DirectDeliveryBot) -> SimpleNamespace:
    return SimpleNamespace(user_data={}, args=[], bot=bot)


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


async def open_direct_recipient_picker(
    bot: FutureSelfBot,
    delivery_bot: DirectDeliveryBot,
    *,
    card: FakeMessage | None = None,
) -> tuple[FakeMessage, int]:
    card = card or await create_pair_workspace(bot)
    ctx = delivery_context(delivery_bot)
    await click(bot, card, "Пригласить", ctx=ctx)
    await click(bot, card, "Редактор", ctx=ctx)
    await click(bot, card, "Вариант 1", ctx=ctx)
    await click(bot, card, "Отправить через бота", ctx=ctx)
    picker = card.replies[-1]["reply_markup"]
    request = picker.keyboard[0][0].request_users
    assert request is not None
    assert request.request_name is None
    assert request.request_username is None
    assert request.request_photo is None
    return card, request.request_id


def users_shared_message(request_id: int, telegram_id: int) -> FakeMessage:
    message = FakeMessage()
    message.users_shared = SimpleNamespace(
        request_id=request_id,
        users=(SimpleNamespace(user_id=telegram_id),),
    )
    return message


@pytest.mark.autotester
async def test_workspace_catalog_is_flag_aware_and_has_no_dead_disabled_surface(db, fake_ai):
    disabled = FutureSelfBot(settings(enabled=False), db, fake_ai, ScriptedTranscription())
    enabled = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())

    assert "spaces" not in {item.command for item in public_commands(False)}
    assert "spaces" not in navigation_sections(False)
    assert "spaces" not in navigation_actions(False)
    assert disabled.natural_command_router.route("покажи мои пространства").action == "show_spaces"

    validate_catalog(True)
    assert "spaces" not in {item.command for item in public_commands(True)}
    assert "spaces" not in navigation_sections(True)
    assert "spaces" in navigation_sections(True)["sections"].actions
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
    await recipient._user(880002)
    await AccessService(db).grant_subscriber(880002, source="test")
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
async def test_known_recipient_gets_in_bot_invitation_without_link_or_identity_leak(
    db, fake_ai, monkeypatch, caplog
):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await owner._user(880002)
    transport = DirectDeliveryBot()
    card, request_id = await open_direct_recipient_picker(owner, transport)

    captured_tokens: list[str] = []
    create_invitation = owner.workspace_service.create_invitation

    async def capture_invitation(*args, **kwargs):
        issued = await create_invitation(*args, **kwargs)
        captured_tokens.append(issued.token)
        return issued

    monkeypatch.setattr(owner.workspace_service, "create_invitation", capture_invitation)
    shared = users_shared_message(request_id, 880002)
    await owner.workspace_users_shared(update_for(shared), delivery_context(transport))

    assert len(transport.sent) == 1
    delivered = transport.sent[0]
    assert delivered["chat_id"] == 880002
    assert "Приглашение в совместное пространство" in str(delivered["text"])
    delivered_labels = {
        button.text for row in delivered["reply_markup"].inline_keyboard for button in row
    }
    assert {"Присоединиться", "Отклонить", "Подробнее", "Не сейчас"} <= delivered_labels
    assert "https://" not in str(delivered["text"])
    assert "Готово — бот отправил приглашение" in shared.replies[-1]["text"]

    visible_text = "\n".join(
        [*(reply["text"] for reply in card.replies), *(reply["text"] for reply in shared.replies)]
    )
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert captured_tokens
    assert all(token not in visible_text and token not in log_text for token in captured_tokens)
    assert "880002" not in visible_text
    assert "880002" not in log_text

    invitation_card = FakeMessage()
    invitation_card.replies.append(delivered)
    accept = FakeCallbackQuery(
        callback_by_label(invitation_card, "Присоединиться"), invitation_card
    )
    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await restarted.workspace_callback(
        update_for(invitation_card, user_id=880002, query=accept), context()
    )
    assert any("Ты присоединился" in reply["text"] for reply in invitation_card.replies)


@pytest.mark.autotester
async def test_unknown_recipient_gets_honest_fallback_without_global_user_directory(db, fake_ai):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    transport = DirectDeliveryBot()
    _card, request_id = await open_direct_recipient_picker(owner, transport)
    shared = users_shared_message(request_id, 999_999_123)

    await owner.workspace_users_shared(update_for(shared), delivery_context(transport))

    assert transport.sent == []
    assert "Бот не смог отправить сообщение" in shared.replies[-1]["text"]
    assert "ещё не запускал бота или запретил сообщения" in shared.replies[-1]["text"]
    assert "999999123" not in "\n".join(reply["text"] for reply in shared.replies)
    assert "https://" not in "\n".join(reply["text"] for reply in shared.replies)
    assert callback_by_label(shared, "Создать ссылку для передачи").startswith("space:")
    user = await owner._user(880001)
    active = await owner.workspace_service.active_context(user.id, 880001)
    assert active is not None
    assert await owner.workspace_service.list_invitations(active.access_context) == ()


@pytest.mark.autotester
async def test_failed_direct_delivery_revokes_invitation_before_offering_link(db, fake_ai):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await owner._user(880002)
    transport = DirectDeliveryBot(fail=True)
    _card, request_id = await open_direct_recipient_picker(owner, transport)
    shared = users_shared_message(request_id, 880002)

    await owner.workspace_users_shared(update_for(shared), delivery_context(transport))

    user = await owner._user(880001)
    active = await owner.workspace_service.active_context(user.id, 880001)
    assert active is not None
    assert await owner.workspace_service.list_invitations(active.access_context) == ()
    revoked = await owner.workspace_service.list_invitations(
        active.access_context, status="revoked"
    )
    assert len(revoked) == 1
    assert revoked[0].delivery_mode == "direct"
    assert "Адресное приглашение не осталось активным" in shared.replies[-1]["text"]
    assert callback_by_label(shared, "Создать ссылку для передачи").startswith("space:")


@pytest.mark.autotester
async def test_direct_invitation_management_requires_revoke_and_fresh_delivery(db, fake_ai):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    card = await create_pair_workspace(owner)
    owner_user = await owner._user(880001)
    recipient = await owner._user(880002)
    active = await owner.workspace_service.active_context(owner_user.id, 880001)
    assert active is not None
    workspace = await owner.workspace_service.get_workspace(active.access_context)
    direct = await owner.workspace_service.create_invitation(
        active.access_context,
        delivery_mode="direct",
        intended_user_id=recipient.id,
        role="editor",
        template_key="pair_1",
    )
    preexisting_renew = await owner.workspace_service.issue_action(
        owner_user.id,
        880001,
        "invite_renew",
        payload={
            "invitation_id": direct.invitation.id,
            "invitation_version": direct.invitation.version,
        },
        context=active.access_context,
        workspace_version=workspace.version,
    )

    await click(owner, card, "Приглашения")
    await click(owner, card, "Приглашение 1", contains=True)
    direct_view = card.replies[-1]
    direct_labels = {
        button.text for row in direct_view["reply_markup"].inline_keyboard for button in row
    }
    assert "Обновить" not in direct_labels
    assert "Отозвать" in direct_labels
    assert "отзови это и создай новое" in direct_view["text"]

    stale_query = FakeCallbackQuery(preexisting_renew, card)
    await owner.workspace_callback(update_for(card, query=stale_query), context())
    assert any(show_alert for _text, show_alert in stale_query.answers)
    pending = await owner.workspace_service.list_invitations(active.access_context)
    assert [(item.id, item.version, item.delivery_mode) for item in pending] == [
        (direct.invitation.id, direct.invitation.version, "direct")
    ]

    await owner.workspace_service.revoke_invitation(
        active.access_context, direct.invitation.id, direct.invitation.version
    )
    await owner.workspace_service.create_invitation(
        active.access_context,
        delivery_mode="share",
        role="editor",
        template_key="pair_1",
    )
    fresh = FakeMessage("/spaces")
    await owner.spaces_command(update_for(fresh), context())
    await click(owner, fresh, "Наше будущее", contains=True)
    await click(owner, fresh, "Приглашения")
    await click(owner, fresh, "Приглашение 1", contains=True)
    share_labels = {
        button.text for row in fresh.replies[-1]["reply_markup"].inline_keyboard for button in row
    }
    assert "Обновить" in share_labels


@pytest.mark.autotester
async def test_ambiguous_delivery_revokes_invite_and_delivered_buttons_fail_closed(db, fake_ai):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await owner._user(880002)
    transport = DirectDeliveryBot(ambiguous=True)
    _card, request_id = await open_direct_recipient_picker(owner, transport)
    shared = users_shared_message(request_id, 880002)

    await owner.workspace_users_shared(update_for(shared), delivery_context(transport))

    assert len(transport.sent) == 1
    accidentally_delivered = FakeMessage()
    accidentally_delivered.replies.append(transport.sent[0])
    stale_accept = FakeCallbackQuery(
        callback_by_label(accidentally_delivered, "Присоединиться"), accidentally_delivered
    )
    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await restarted.workspace_callback(
        update_for(accidentally_delivered, user_id=880002, query=stale_accept), context()
    )

    assert any(show_alert for _text, show_alert in stale_accept.answers)
    assert all("Наше будущее" not in (text or "") for text, _show_alert in stale_accept.answers)
    owner_user = await owner._user(880001)
    active = await owner.workspace_service.active_context(owner_user.id, 880001)
    assert active is not None
    assert await owner.workspace_service.list_invitations(active.access_context) == ()


@pytest.mark.autotester
async def test_recipient_picker_is_actor_chat_and_request_bound(db, fake_ai):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    await owner._user(880002)
    transport = DirectDeliveryBot()
    _card, request_id = await open_direct_recipient_picker(owner, transport)

    forged_actor = users_shared_message(request_id, 880002)
    await owner.workspace_users_shared(
        update_for(forged_actor, user_id=880099), delivery_context(transport)
    )
    forged_chat = users_shared_message(request_id, 880002)
    await owner.workspace_users_shared(
        update_for(forged_chat, chat_id=880099), delivery_context(transport)
    )
    forged_request = users_shared_message(request_id + 1, 880002)
    await owner.workspace_users_shared(update_for(forged_request), delivery_context(transport))

    assert transport.sent == []
    assert all(
        "Наше будущее" not in reply["text"]
        for message in (forged_actor, forged_chat, forged_request)
        for reply in message.replies
    )
    user = await owner._user(880001)
    pending = await owner.workspace_service.pending_input(user.id, 880001)
    assert pending is not None
    assert pending.action == "input:invite_recipient"

    valid = users_shared_message(request_id, 880002)
    await owner.workspace_users_shared(update_for(valid), delivery_context(transport))
    assert len(transport.sent) == 1


@pytest.mark.autotester
async def test_self_and_existing_member_do_not_create_or_fallback_to_share_invites(db, fake_ai):
    owner = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    transport = DirectDeliveryBot()
    _card, request_id = await open_direct_recipient_picker(owner, transport)
    self_selection = users_shared_message(request_id, 880001)
    await owner.workspace_users_shared(update_for(self_selection), delivery_context(transport))
    assert "Себя приглашать не нужно" in self_selection.replies[-1]["text"]
    assert transport.sent == []

    owner_user = await owner._user(880001)
    member_user = await owner._user(880002)
    active = await owner.workspace_service.active_context(owner_user.id, 880001)
    assert active is not None
    existing = await owner.workspace_service.create_invitation(
        active.access_context,
        delivery_mode="direct",
        intended_user_id=member_user.id,
        role="editor",
        template_key="pair_1",
    )
    await owner.workspace_service.accept_invitation(member_user.id, existing.token)

    fresh = FakeMessage("/spaces")
    await owner.spaces_command(update_for(fresh), context())
    await click(owner, fresh, "Наше будущее", contains=True)
    _card, member_request_id = await open_direct_recipient_picker(owner, transport, card=fresh)
    member_selection = users_shared_message(member_request_id, 880002)
    await owner.workspace_users_shared(update_for(member_selection), delivery_context(transport))
    assert "уже участвует" in member_selection.replies[-1]["text"]
    assert all(
        button.text != "Создать ссылку для передачи"
        for reply in member_selection.replies
        for row in getattr(reply.get("reply_markup"), "inline_keyboard", ())
        for button in row
    )
    assert transport.sent == []


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
    assert voice.reply_text_calls == 1
    assert voice.replies[0]["text"] == "Расшифровываю голосовую мысль…"
    assert voice.edits[-1].startswith("🗂 Мои разделы")
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
