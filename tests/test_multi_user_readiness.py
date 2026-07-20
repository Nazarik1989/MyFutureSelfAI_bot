from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from telegram.ext import ApplicationHandlerStop

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.doctor_prep import DoctorVisitPrepService
from future_self.drafts import DraftInboxService
from future_self.health import HealthService
from future_self.models import OnboardingState, TaskReminder
from future_self.reminders import TaskReminderEngine
from future_self.repositories import OnboardingRepository, ProfileRepository, UserRepository
from future_self.schemas import ParsedThought, TemporalResolution, VisionSummary


class NoopTranscription:
    enabled = False


class CaptureMessage:
    def __init__(self, text: str = ""):
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append(text)
        return self


class CaptureQuery:
    def __init__(self, data: str, message: CaptureMessage):
        self.data = data
        self.message = message
        self.answers: list[tuple[str | None, bool]] = []
        self.markup_removed = 0

    async def answer(self, text: str | None = None, show_alert: bool = False):
        self.answers.append((text, show_alert))

    async def edit_message_reply_markup(self, reply_markup=None):
        self.markup_removed += 1


def bot_settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def update_for(user_id: int, message: CaptureMessage, *, chat_id: int | None = None):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id or user_id, type="private"),
        effective_message=message,
        callback_query=None,
    )


def health_answers(marker: str) -> dict[str, object]:
    return {
        "energy": 7,
        "sleep": 6,
        "mood": 8,
        "stress": 3,
        "physical_wellbeing": 7,
        "symptoms": marker,
    }


def doctor_answers(marker: str) -> dict[str, object]:
    return {
        "reason": marker,
        "duration": "две недели",
        "symptoms": f"наблюдение {marker}",
        "medications": "нет",
        "questions": "нет",
    }


@pytest.mark.parametrize("chat_type", ["group", "supergroup", "channel"])
async def test_non_private_telegram_updates_are_stopped_before_feature_handlers(
    db, fake_ai, chat_type
):
    bot = FutureSelfBot(bot_settings(), db, fake_ai, NoopTranscription())
    message = CaptureMessage("/health")
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=-100_123, type=chat_type),
        effective_message=message,
        callback_query=None,
    )

    with pytest.raises(ApplicationHandlerStop):
        await bot.private_chat_guard(update, SimpleNamespace())

    assert message.replies == [
        "Из соображений приватности бот работает только в личном чате. "
        "Открой диалог с ботом напрямую."
    ]


async def test_two_users_are_isolated_across_all_private_data_paths(db, fake_ai):
    first_tg, second_tg = 91_000_001, 91_000_002
    first_marker, second_marker = "PRIVATE-ALPHA-7f3", "PRIVATE-BETA-9c1"
    bot = FutureSelfBot(bot_settings(), db, fake_ai, NoopTranscription())

    async with db.session() as session:
        first = await UserRepository(session).get_or_create(first_tg, "Europe/Moscow")
        second = await UserRepository(session).get_or_create(second_tg, "Europe/Moscow")
        first_state = await OnboardingRepository(session).get_or_create(first.id)
        second_state = await OnboardingRepository(session).get_or_create(second.id)
        first_state.answers = {"future_life": first_marker}
        second_state.answers = {"future_life": second_marker}
        first_state.current_step = 3
        second_state.current_step = 5
        await ProfileRepository(session).upsert(
            first,
            {"future_life": first_marker},
            VisionSummary(
                summary=first_marker,
                values=[first_marker],
                desired_identity=[first_marker],
                constraints=[],
            ),
        )
        await ProfileRepository(session).upsert(
            second,
            {"future_life": second_marker},
            VisionSummary(
                summary=second_marker,
                values=[second_marker],
                desired_identity=[second_marker],
                constraints=[],
            ),
        )
        first_id, second_id = first.id, second.id

    # Onboarding and profiles remain owner-scoped.
    async with db.sessions() as session:
        states = {
            state.user_id: state for state in (await session.scalars(select(OnboardingState))).all()
        }
    assert states[first_id].answers == {"future_life": first_marker}
    assert states[second_id].answers == {"future_life": second_marker}
    second_profile_message = CaptureMessage("/profile")
    await bot.profile(
        update_for(second_tg, second_profile_message),
        SimpleNamespace(user_data={}),
    )
    assert second_marker in second_profile_message.replies[-1]
    assert first_marker not in second_profile_message.replies[-1]

    drafts = DraftInboxService(db, 60)
    first_draft = await drafts.create(
        user_id=first_id,
        telegram_user_id=first_tg,
        chat_id=first_tg,
        source="text",
        raw_text=first_marker,
        parsed=ParsedThought(kind="task", title=first_marker),
    )
    second_draft = await drafts.create(
        user_id=second_id,
        telegram_user_id=second_tg,
        chat_id=second_tg,
        source="text",
        raw_text=second_marker,
        parsed=ParsedThought(kind="task", title=second_marker),
    )
    assert (await drafts.confirm(first_draft.id, 1, first_tg, first_tg)).ok
    assert (await drafts.confirm(second_draft.id, 1, second_tg, second_tg)).ok

    # A forged callback containing another user's UUID cannot save or mutate it.
    foreign_draft = await drafts.create(
        user_id=first_id,
        telegram_user_id=first_tg,
        chat_id=first_tg,
        source="text",
        raw_text=f"callback-{first_marker}",
        parsed=ParsedThought(kind="note", title=f"callback-{first_marker}"),
    )
    query = CaptureQuery(
        f"inbox:save:{foreign_draft.id}:{foreign_draft.version}",
        CaptureMessage(),
    )
    callback_update = SimpleNamespace(
        effective_user=SimpleNamespace(id=second_tg),
        effective_chat=SimpleNamespace(id=second_tg, type="private"),
        effective_message=query.message,
        callback_query=query,
    )
    await bot.inbox_action(callback_update, SimpleNamespace(user_data={}))
    assert query.answers[-1][1] is True
    assert (await drafts.get(foreign_draft.id)).status == "preview"

    second_inbox_message = CaptureMessage("/inbox")
    await bot.inbox(
        update_for(second_tg, second_inbox_message),
        SimpleNamespace(user_data={}),
    )
    assert second_marker in second_inbox_message.replies[-1]
    assert first_marker not in second_inbox_message.replies[-1]

    # Draft focus and conversational context cannot be pointed at a foreign draft.
    await bot.conversation.append(
        first_tg,
        first_tg,
        role="user",
        content=first_marker,
        source="text",
        intent="conversation",
    )
    await bot.conversation.append(
        second_tg,
        second_tg,
        role="user",
        content=second_marker,
        source="text",
        intent="conversation",
    )
    await bot.conversation.set_active_draft(second_tg, second_tg, foreign_draft.id)
    await bot.conversation.set_focus(
        second_tg,
        second_tg,
        foreign_draft.id,
        foreign_draft.version,
        "save",
    )
    second_context = await bot.conversation.get(second_tg, second_tg)
    assert second_context.active_draft is None
    assert second_context.focused_draft_id is None
    assert second_marker in str(second_context.messages)
    assert first_marker not in str(second_context.messages)

    await bot.focus_service.generate(second_id)
    assert second_marker in str(fake_ai.last_today_context)
    assert first_marker not in str(fake_ai.last_today_context)

    health = HealthService(db)
    first_health = await health.save(
        user_id=first_id,
        timezone="Europe/Moscow",
        answers=health_answers(first_marker),
    )
    await health.save(
        user_id=second_id,
        timezone="Europe/Moscow",
        answers=health_answers(second_marker),
    )
    assert await health.get_owned(second_id, first_health.id) is None
    assert await health.delete_owned(second_id, first_health.id) is False
    second_health_message = CaptureMessage("/health")
    await bot.health_command(
        update_for(second_tg, second_health_message),
        SimpleNamespace(user_data={}),
    )
    assert second_marker in second_health_message.replies[-1]
    assert first_marker not in second_health_message.replies[-1]

    # Persisted health chat IDs are normalized to the owner's private chat.
    preference = await health.set_reminder(
        user_id=second_id,
        telegram_user_id=second_tg,
        chat_id=-100_999_888,
        timezone="Europe/Moscow",
        local_time=datetime.now().time().replace(second=0, microsecond=0),
        enabled=True,
    )
    assert preference.telegram_user_id == second_tg
    assert preference.chat_id == second_tg
    with pytest.raises(ValueError):
        await health.set_reminder(
            user_id=first_id,
            telegram_user_id=second_tg,
            chat_id=second_tg,
            timezone="Europe/Moscow",
            local_time=preference.local_time,
            enabled=True,
        )

    doctor = DoctorVisitPrepService(db)
    first_doctor = await doctor.save(
        user_id=first_id,
        timezone="Europe/Moscow",
        answers=doctor_answers(first_marker),
    )
    second_doctor = await doctor.save(
        user_id=second_id,
        timezone="Europe/Moscow",
        answers=doctor_answers(second_marker),
    )
    assert await doctor.get_owned(second_id, first_doctor.id) is None
    assert await doctor.delete_owned(second_id, first_doctor.id) is False
    foreign_show = CaptureMessage(f"/doctor_prepare_show {first_doctor.id}")
    await bot.doctor_prepare_show(
        update_for(second_tg, foreign_show),
        SimpleNamespace(args=[str(first_doctor.id)], user_data={}),
    )
    assert first_marker not in foreign_show.replies[-1]
    own_show = CaptureMessage(f"/doctor_prepare_show {second_doctor.id}")
    await bot.doctor_prepare_show(
        update_for(second_tg, own_show),
        SimpleNamespace(args=[str(second_doctor.id)], user_data={}),
    )
    assert second_marker in own_show.replies[-1]
    assert first_marker not in own_show.replies[-1]

    # Reminder delivery derives the destination from the inbox owner, not from
    # a stale or malicious persisted group chat ID.
    now = datetime.now(UTC)
    temporal = TemporalResolution(
        resolved_at=now + timedelta(hours=1),
        remind_at=now - timedelta(minutes=1),
        timezone="Europe/Moscow",
        resolved_local_date=(now + timedelta(hours=1)).date(),
        resolved_local_time=(now + timedelta(hours=1)).time().replace(microsecond=0),
        precision="datetime",
        original_expression="через час",
    )
    reminder_draft = await drafts.create(
        user_id=second_id,
        telegram_user_id=second_tg,
        chat_id=-100_777_666,
        source="text",
        raw_text=f"reminder-{second_marker}",
        parsed=ParsedThought(
            kind="task",
            title=f"reminder-{second_marker}",
            temporal_resolution=temporal,
        ),
    )
    assert (
        await drafts.confirm(
            reminder_draft.id,
            reminder_draft.version,
            second_tg,
            -100_777_666,
        )
    ).ok
    async with db.sessions() as session:
        stored = await session.scalar(
            select(TaskReminder).where(TaskReminder.chat_id == -100_777_666)
        )
    assert stored is not None
    deliveries: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> int:
        deliveries.append((chat_id, text))
        return 1

    assert await TaskReminderEngine(db, send).deliver_due(now=now) == 1
    assert deliveries[0][0] == second_tg
    assert second_marker in deliveries[0][1]
    assert first_marker not in deliveries[0][1]
