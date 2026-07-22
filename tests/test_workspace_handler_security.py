import logging
import re
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest
from autotester.fakes import FakeBot, FakeCallbackQuery, FakeMessage, ScriptedTranscription
from sqlalchemy import select, update
from telegram.ext import ApplicationHandlerStop, ConversationHandler

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.labs import LabUploadSessionStore
from future_self.models import (
    DoctorVisitPrep,
    HealthCheckIn,
    InboxItem,
    LabDocument,
    OnboardingState,
    TaskActionToken,
    TaskState,
    User,
    WorkspaceActionToken,
)

OWNER_TELEGRAM_ID = 9_910_001
RECIPIENT_TELEGRAM_ID = 9_910_002
OTHER_TELEGRAM_ID = 9_910_003
OWNER_CHAT_ID = 9_920_001
RECIPIENT_CHAT_ID = 9_920_002


class WorkspaceFakeBot(FakeBot):
    username = "future_self_security_bot"


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        database_url="sqlite+aiosqlite:///:memory:",
        enable_workspace_access=True,
    )


def context(*, args: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(user_data={}, args=args or [], bot=WorkspaceFakeBot())


def update_for(
    message: FakeMessage,
    *,
    user_id: int = OWNER_TELEGRAM_ID,
    chat_id: int = OWNER_CHAT_ID,
    query: FakeCallbackQuery | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
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


def callback_data(message: FakeMessage) -> list[str]:
    values: list[str] = []
    for reply in message.replies:
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        values.extend(
            button.callback_data
            for row in markup.inline_keyboard
            for button in row
            if button.callback_data
        )
    return values


async def click(
    bot: FutureSelfBot,
    message: FakeMessage,
    label: str,
    *,
    user_id: int = OWNER_TELEGRAM_ID,
    chat_id: int = OWNER_CHAT_ID,
    contains: bool = False,
    ctx: SimpleNamespace | None = None,
) -> FakeCallbackQuery:
    query = FakeCallbackQuery(callback_by_label(message, label, contains=contains), message)
    await bot.workspace_callback(
        update_for(message, user_id=user_id, chat_id=chat_id, query=query),
        ctx or context(),
    )
    return query


async def set_display_name(db, user_id: int, value: str) -> None:
    async with db.session() as session:
        user = await session.get(User, user_id)
        assert user is not None
        user.display_name = value


async def seed_private_medical_markers(db, owner_id: int, marker: str) -> None:
    async with db.session() as session:
        session.add_all(
            [
                HealthCheckIn(
                    user_id=owner_id,
                    local_date=date(2026, 7, 22),
                    timezone="Europe/Moscow",
                    energy=5,
                    sleep=5,
                    mood=5,
                    stress=5,
                    physical_wellbeing=5,
                    symptoms=marker,
                    state_score=50,
                ),
                DoctorVisitPrep(
                    user_id=owner_id,
                    timezone="Europe/Moscow",
                    reason=marker,
                    duration="one day",
                    symptoms=marker,
                    medications=None,
                    questions=None,
                    health_snapshot={"private_marker": marker},
                    summary=marker,
                ),
                LabDocument(
                    owner_id=owner_id,
                    title=marker,
                    document_date=date(2026, 7, 22),
                    source_type="image",
                    page_count=1,
                    status="saved",
                    version=1,
                ),
            ]
        )


async def test_direct_deep_link_precedes_onboarding_and_capabilities_survive_restart(
    db, fake_ai, caplog
):
    caplog.set_level(logging.DEBUG, logger="future_self")
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(OWNER_TELEGRAM_ID)
    recipient = await bot._user(RECIPIENT_TELEGRAM_ID)
    await set_display_name(db, owner.id, "<b>Alice & Bob</b>")
    workspace = await bot.workspace_service.create_workspace(
        owner.id,
        "pair",
        "<i>Room & Family</i>",
        "<script>workspace description</script>",
    )
    owner_access = await bot.workspace_service.access_context(owner.id, workspace.id)
    private_marker = "PRIVATE-MEDICAL-MARKER-41f38b"
    await seed_private_medical_markers(db, owner.id, private_marker)
    invitation = await bot.workspace_service.create_invitation(
        owner_access,
        delivery_mode="direct",
        intended_user_id=recipient.id,
        role="viewer",
        template_key="pair_1",
    )

    wrong_message = FakeMessage("/start")
    wrong_result = await bot.start(
        update_for(
            wrong_message,
            user_id=OTHER_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID,
        ),
        context(args=[f"space_{invitation.token}"]),
    )
    assert wrong_result == ConversationHandler.END
    assert wrong_message.replies[-1]["text"] == "Приглашение недействительно."
    assert "Room" not in wrong_message.replies[-1]["text"]

    incoming = FakeMessage("/start")
    result = await bot.start(
        update_for(
            incoming,
            user_id=RECIPIENT_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID,
        ),
        context(args=[f"space_{invitation.token}"]),
    )
    assert result == ConversationHandler.END
    rendered = incoming.replies[-1]["text"]
    assert "&lt;b&gt;Alice &amp; Bob&lt;/b&gt;" in rendered
    assert "&lt;i&gt;Room &amp; Family&lt;/i&gt;" in rendered
    assert "<b>Alice" not in rendered
    assert "<i>Room" not in rendered
    assert invitation.token not in rendered
    assert private_marker not in rendered
    assert "настройка профиля" not in rendered.casefold()

    incoming_callbacks = callback_data(incoming)
    assert len(incoming_callbacks) == 4
    assert all(re.fullmatch(r"spacei:[A-Za-z0-9_-]{20,48}", item) for item in incoming_callbacks)
    assert all(len(item.encode("utf-8")) <= 64 for item in incoming_callbacks)
    assert all(invitation.token not in item for item in incoming_callbacks)

    details_query = await click(
        bot,
        incoming,
        "Подробнее",
        user_id=RECIPIENT_TELEGRAM_ID,
        chat_id=RECIPIENT_CHAT_ID,
    )
    assert details_query.answers == [(None, False)]
    assert "Health, Doctor и Labs не передаются" in incoming.replies[-1]["text"]
    assert private_marker not in incoming.replies[-1]["text"]

    accept_callback = callback_by_label(incoming, "Присоединиться")
    wrong_actor_query = FakeCallbackQuery(accept_callback, incoming)
    await bot.workspace_callback(
        update_for(
            incoming,
            user_id=OTHER_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID,
            query=wrong_actor_query,
        ),
        context(),
    )
    assert any(show_alert for _text, show_alert in wrong_actor_query.answers)

    wrong_chat_query = FakeCallbackQuery(accept_callback, incoming)
    await bot.workspace_callback(
        update_for(
            incoming,
            user_id=RECIPIENT_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID + 1,
            query=wrong_chat_query,
        ),
        context(),
    )
    assert any(show_alert for _text, show_alert in wrong_chat_query.answers)

    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    accepted_query = FakeCallbackQuery(accept_callback, incoming)
    await restarted.workspace_callback(
        update_for(
            incoming,
            user_id=RECIPIENT_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID,
            query=accepted_query,
        ),
        context(),
    )
    assert accepted_query.answers == [(None, False)]
    assert "Ты присоединился" in incoming.replies[-1]["text"]
    assert private_marker not in "\n".join(reply["text"] for reply in incoming.replies)

    replay_query = FakeCallbackQuery(accept_callback, incoming)
    await restarted.workspace_callback(
        update_for(
            incoming,
            user_id=RECIPIENT_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID,
            query=replay_query,
        ),
        context(),
    )
    assert any(show_alert for _text, show_alert in replay_query.answers)

    async with db.sessions() as session:
        onboarding = await session.scalar(
            select(OnboardingState).where(OnboardingState.user_id == recipient.id)
        )
    assert onboarding is None
    assert fake_ai.route_calls == []
    log_output = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "future_self" or record.name.startswith("future_self.")
    )
    for secret in (
        invitation.token,
        private_marker,
        str(OWNER_TELEGRAM_ID),
        str(RECIPIENT_TELEGRAM_ID),
        "<script>workspace description</script>",
    ):
        assert secret not in log_output


async def test_revoked_invitation_action_is_generic_and_unusable(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(OWNER_TELEGRAM_ID)
    target = await bot._user(RECIPIENT_TELEGRAM_ID)
    workspace = await bot.workspace_service.create_workspace(owner.id, "team", "Revocation")
    access = await bot.workspace_service.access_context(owner.id, workspace.id)
    issued = await bot.workspace_service.create_invitation(
        access,
        delivery_mode="direct",
        intended_user_id=target.id,
        role="editor",
        template_key="team_1",
    )
    incoming = FakeMessage("/start")
    await bot.start(
        update_for(
            incoming,
            user_id=RECIPIENT_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID,
        ),
        context(args=[f"space_{issued.token}"]),
    )
    accept_callback = callback_by_label(incoming, "Присоединиться")
    await bot.workspace_service.revoke_invitation(
        access, issued.invitation.id, issued.invitation.version
    )

    query = FakeCallbackQuery(accept_callback, incoming)
    await FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription()).workspace_callback(
        update_for(
            incoming,
            user_id=RECIPIENT_TELEGRAM_ID,
            chat_id=RECIPIENT_CHAT_ID,
            query=query,
        ),
        context(),
    )
    assert query.answers == [("Эта кнопка устарела или недоступна.", True)]
    assert "Revocation" not in (query.answers[-1][0] or "")
    assert fake_ai.route_calls == []


async def test_archived_workspace_buttons_are_scoped_reachable_and_truncated(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(OWNER_TELEGRAM_ID)
    long_name = "W" * 100
    workspace = await bot.workspace_service.create_workspace(owner.id, "custom", long_name)

    message = FakeMessage("/spaces")
    await bot.spaces_command(update_for(message), context())
    latest_markup = message.replies[-1]["reply_markup"]
    workspace_button = next(
        button
        for row in latest_markup.inline_keyboard
        for button in row
        if long_name[:20] in button.text
    )
    assert len(workspace_button.text) <= 56
    assert workspace_button.text.endswith("…")
    assert re.fullmatch(r"space:[A-Za-z0-9_-]{20,48}", workspace_button.callback_data)

    await click(bot, message, long_name[:20], contains=True)
    stale_members = callback_by_label(message, "Участники")
    await click(bot, message, "Архивировать")
    await click(bot, message, "Архивировать")
    assert "Пространство перемещено в архив" in message.replies[-1]["text"]

    stale_query = FakeCallbackQuery(stale_members, message)
    await bot.workspace_callback(
        update_for(message, query=stale_query),
        context(),
    )
    assert stale_query.answers == [("Эта кнопка устарела или недоступна.", True)]

    await click(bot, message, "Архив")
    await click(bot, message, long_name[:20], contains=True)
    archived_labels = {
        button.text for row in message.replies[-1]["reply_markup"].inline_keyboard for button in row
    }
    assert "Восстановить" in archived_labels
    await click(bot, message, "Восстановить")
    assert "Пространство восстановлено" in message.replies[-1]["text"]

    refreshed = await bot.workspace_service.access_context(owner.id, workspace.id)
    assert (await bot.workspace_service.get_workspace(refreshed)).status == "active"
    for item in callback_data(message):
        assert len(item.encode("utf-8")) <= 64
        assert item.startswith(("space:", "nav:"))
        if item.startswith("space:"):
            assert item.count(":") == 1
            assert re.fullmatch(r"space:[A-Za-z0-9_-]{20,48}", item)
    assert fake_ai.route_calls == []


async def test_navigation_gates_vision_labs_and_workspace_until_explicit_exit(
    db, fake_ai, tmp_path
):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    bot.lab_uploads = LabUploadSessionStore(root=tmp_path / "labs")
    owner = await bot._user(OWNER_TELEGRAM_ID)

    await bot.vision_service.begin(owner.id, OWNER_CHAT_ID)
    vision_message = FakeMessage("/spaces")
    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_public_command_gate(update_for(vision_message), context())
    assert "создание карточки желания" in vision_message.replies[-1]["text"]
    assert await bot.vision_service.draft(owner.id, OWNER_CHAT_ID) is not None
    assert await bot.workspace_service.pending_input(owner.id, OWNER_CHAT_ID) is None
    await bot.vision_service.cancel(owner.id, OWNER_CHAT_ID)

    assert await bot.lab_uploads.start(owner.id, OWNER_CHAT_ID)
    labs_message = FakeMessage("/spaces")
    with pytest.raises(ApplicationHandlerStop):
        await bot.navigation_public_command_gate(update_for(labs_message), context())
    assert "загрузка результатов анализов" in labs_message.replies[-1]["text"]
    assert await bot.lab_uploads.has_active(owner.id, OWNER_CHAT_ID)
    assert await bot.workspace_service.pending_input(owner.id, OWNER_CHAT_ID) is None
    await bot.lab_uploads.cancel_active(owner.id, OWNER_CHAT_ID)

    await bot.workspace_service.begin_input(
        owner.id,
        OWNER_CHAT_ID,
        "create_name",
        payload={"character": "pair"},
    )
    navigation_message = FakeMessage()
    navigation_query = FakeCallbackQuery("nav:root", navigation_message)
    ctx = context()
    await bot.navigation_action(update_for(navigation_message, query=navigation_query), ctx)
    assert "операция с совместным пространством" in navigation_message.replies[-1]["text"]
    assert await bot.workspace_service.pending_input(owner.id, OWNER_CHAT_ID) is not None

    exit_callback = callback_by_label(navigation_message, "Выйти в меню")
    exit_query = FakeCallbackQuery(exit_callback, navigation_message)
    result = await bot.navigation_action(
        update_for(navigation_message, query=exit_query),
        ctx,
    )
    assert result == ConversationHandler.END
    assert await bot.workspace_service.pending_input(owner.id, OWNER_CHAT_ID) is None
    assert fake_ai.route_calls == []


@pytest.mark.autotester
async def test_expired_workspace_input_is_consumed_without_falling_through_to_llm(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(OWNER_TELEGRAM_ID)
    await bot.workspace_service.begin_input(
        owner.id,
        OWNER_CHAT_ID,
        "create_name",
        payload={"character": "pair"},
    )
    async with db.session() as session:
        await session.execute(
            update(WorkspaceActionToken)
            .where(
                WorkspaceActionToken.actor_user_id == owner.id,
                WorkspaceActionToken.chat_id == OWNER_CHAT_ID,
                WorkspaceActionToken.status == "awaiting_input",
            )
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    await bot.workspace_service.cleanup()

    message = FakeMessage("Не отправляй это в маршрутизацию")
    await bot.text(update_for(message), context())

    assert "Время ввода для пространства истекло" in message.replies[-1]["text"]
    assert await bot.workspace_service.pending_input(owner.id, OWNER_CHAT_ID) is None
    assert fake_ai.route_calls == []


async def test_workspace_input_replaces_task_and_collection_inputs(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(OWNER_TELEGRAM_ID)
    async with db.session() as session:
        item = InboxItem(
            user_id=owner.id,
            kind="task",
            title="Проверить пересечение потоков",
            raw_text="Проверить пересечение потоков",
            source="text",
            status="confirmed",
        )
        session.add(item)
        await session.flush()
        session.add(
            TaskState(
                owner_id=owner.id,
                inbox_item_id=item.id,
                status="active",
                timezone=owner.timezone,
                version=1,
            )
        )
        await session.flush()
        session.add(
            TaskActionToken(
                token="task-workspace-intersection-01",
                owner_id=owner.id,
                chat_id=OWNER_CHAT_ID,
                inbox_item_id=item.id,
                task_version=1,
                action="event_input",
                payload=None,
                status="awaiting_input",
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )
        )
    await bot.collection_service.issue_action(
        owner.id,
        OWNER_CHAT_ID,
        "input_rename",
        status="awaiting_input",
        ttl=timedelta(minutes=5),
    )

    await bot._begin_workspace_input(
        owner.id,
        OWNER_CHAT_ID,
        "input_create_name",
        payload={"character": "pair"},
    )

    assert await bot.task_service.pending_input(owner.id, OWNER_CHAT_ID) is None
    assert await bot.collection_service.pending_input(owner.id, OWNER_CHAT_ID) is None
    pending = await bot.workspace_service.pending_input(owner.id, OWNER_CHAT_ID)
    assert pending is not None and pending.action == "input:create_name"
    assert fake_ai.route_calls == []


async def test_archived_member_can_leave_but_last_owner_remains_protected(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(OWNER_TELEGRAM_ID)
    member = await bot._user(RECIPIENT_TELEGRAM_ID)
    workspace = await bot.workspace_service.create_workspace(owner.id, "team", "Archive Team")
    owner_access = await bot.workspace_service.access_context(owner.id, workspace.id)
    invitation = await bot.workspace_service.create_invitation(
        owner_access,
        delivery_mode="direct",
        intended_user_id=member.id,
        role="editor",
        template_key="team_1",
    )
    await bot.workspace_service.accept_invitation(member.id, invitation.token)
    owner_access = await bot.workspace_service.access_context(owner.id, workspace.id)
    current = await bot.workspace_service.get_workspace(owner_access)
    await bot.workspace_service.set_workspace_archived(
        owner_access,
        current.version,
        archived=True,
    )

    member_access = await bot.workspace_service.access_context(member.id, workspace.id)
    message = FakeMessage("/spaces")
    await bot._send_workspace(message, member.id, RECIPIENT_CHAT_ID, member_access)
    assert callback_by_label(message, "Выйти").startswith("space:")
    await click(
        bot,
        message,
        "Выйти",
        user_id=RECIPIENT_TELEGRAM_ID,
        chat_id=RECIPIENT_CHAT_ID,
    )
    assert "Последний владелец выйти не сможет" in message.replies[-1]["text"]
    await click(
        bot,
        message,
        "Выйти",
        user_id=RECIPIENT_TELEGRAM_ID,
        chat_id=RECIPIENT_CHAT_ID,
    )
    assert "Ты вышел из пространства" in message.replies[-1]["text"]

    owner_access = await bot.workspace_service.access_context(owner.id, workspace.id)
    owner_message = FakeMessage("/spaces")
    await bot._send_workspace(owner_message, owner.id, OWNER_CHAT_ID, owner_access)
    labels = {
        button.text
        for row in owner_message.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    }
    assert "Выйти" not in labels
    assert "сначала назначь другого участника владельцем" in owner_message.replies[-1]["text"]
    assert fake_ai.route_calls == []
