from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import and_, select, update

from .ai import AIService
from .db import Database
from .location import parse_location
from .models import (
    DailyCheckIn,
    Goal,
    InboxItem,
    RecurringTaskReminderSchedule,
    Routine,
    TaskReminder,
    TaskState,
    User,
    VisionCompanionPreference,
    VisionItem,
    VisionProfile,
    WeeklyFocus,
)
from .nova_memory_application import NovaMemoryProjection
from .repositories import ProfileRepository
from .schemas import AssistantAnswer, IntentResult, ParsedThought, TodayPlan

ONBOARDING_QUESTIONS: tuple[tuple[str, str, bool], ...] = (
    ("display_name", "Как мне к тебе обращаться?", True),
    (
        "timezone",
        "В каком городе ты сейчас живёшь? Напиши обычное название, например: Казань, Берлин или Алматы. Я сам определю часовой пояс.",
        True,
    ),
    ("future_life", "Как выглядит твоя жизнь через три года?", True),
    ("residence", "Где ты живёшь в этом образе?", False),
    ("work_income", "Чем занимаешься и какой уровень дохода хочешь?", False),
    ("health_body", "Как ты описал(а) бы желаемое здоровье и состояние тела?", False),
    ("relationships", "Какие отношения и окружение тебя поддерживают?", False),
    ("ideal_day", "Как проходит твой идеальный обычный день?", True),
    ("values", "Какие ценности для тебя главные?", True),
    ("obstacles", "Что сейчас чаще всего мешает двигаться к этому?", False),
    ("support_style", "Какой стиль поддержки тебе подходит?", True),
    (
        "location",
        "В каком городе искать врачей? Можно указать маршрут: основной город → запасной.",
        True,
    ),
)
DISPLAY_NAME_MAX_CHARS = 120
ONBOARDING_ANSWER_MAX_CHARS = 8_000
ONBOARDING_TOTAL_MAX_CHARS = 30_000
TODAY_URGENT_TASK_LIMIT = 3
TODAY_REMINDER_LIMIT = 5


class TodayApplicationStatus(StrEnum):
    CURRENT = "current"
    ACTOR_CHANGED = "actor_changed"
    ACCESS_CHANGED = "access_changed"
    TIMEZONE_CHANGED = "timezone_changed"
    WEEK_CHANGED = "week_changed"
    FOCUS_CHANGED = "focus_changed"


@dataclass(frozen=True, slots=True)
class TodayApplicationCheck:
    status: TodayApplicationStatus

    @property
    def is_current(self) -> bool:
        return self.status is TodayApplicationStatus.CURRENT


@dataclass(frozen=True, slots=True)
class TodayApplicationSnapshot:
    actor_id: int
    telegram_id: int
    access_tier: str
    access_version: int
    timezone: str
    local_week_start: date
    includes_weekly_focus: bool
    weekly_focus_public_id: str | None
    weekly_focus_version: int | None
    weekly_focus: str | None = field(repr=False)
    _provider_context: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.includes_weekly_focus:
            has_identity = (
                self.weekly_focus_public_id is not None
                and self.weekly_focus_version is not None
                and self.weekly_focus is not None
            )
            is_confirmed_absence = (
                self.weekly_focus_public_id is None
                and self.weekly_focus_version is None
                and self.weekly_focus is None
            )
            if not (has_identity or is_confirmed_absence):
                raise ValueError(
                    "weekly focus snapshot must contain an exact generation or absence"
                )
        elif any(
            value is not None
            for value in (
                self.weekly_focus_public_id,
                self.weekly_focus_version,
                self.weekly_focus,
            )
        ):
            raise ValueError("excluded weekly focus snapshot cannot contain weekly data")
        object.__setattr__(
            self,
            "_provider_context",
            _freeze_today_context(self._provider_context),
        )

    def provider_context(self) -> dict[str, object]:
        context = _thaw_today_context(self._provider_context)
        if not isinstance(context, dict):  # pragma: no cover - constructor invariant
            raise RuntimeError("today provider context is not a mapping")
        return context


def _freeze_today_context(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_today_context(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_today_context(item) for item in value)
    return value


def _thaw_today_context(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_today_context(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_today_context(item) for item in value]
    return value


def normalize_display_name(value: str, *, clip_legacy: bool = False) -> str:
    clean = " ".join(value.split())
    if not clean:
        if clip_legacy:
            return "друг"
        raise ValueError("Имя не должно быть пустым.")
    if len(clean) > DISPLAY_NAME_MAX_CHARS:
        if clip_legacy:
            return clean[:DISPLAY_NAME_MAX_CHARS].rstrip()
        raise ValueError(
            f"Имя слишком длинное. Укажи не больше {DISPLAY_NAME_MAX_CHARS} символов; "
            "длинный рассказ можно будет написать в следующих ответах."
        )
    return clean


class OnboardingFlow:
    @staticmethod
    def next_step(step: int) -> int:
        return min(step + 1, len(ONBOARDING_QUESTIONS))

    @staticmethod
    def previous_step(step: int) -> int:
        return max(step - 1, 0)

    @staticmethod
    def answer(answers: dict[str, Any], step: int, value: str | None) -> dict[str, Any]:
        if not 0 <= step < len(ONBOARDING_QUESTIONS):
            raise ValueError("Текущий шаг регистрации устарел. Запусти /start, чтобы продолжить.")
        result = dict(answers)
        key, _, required = ONBOARDING_QUESTIONS[step]
        if value is None and required:
            raise ValueError("Этот вопрос нельзя пропустить")
        if value is None:
            result.pop(key, None)
        else:
            clean = value.strip()
            if not clean:
                raise ValueError("Ответ получился пустым. Напиши ответ или используй /skip.")
            if key == "display_name":
                clean = normalize_display_name(clean)
            elif len(clean) > ONBOARDING_ANSWER_MAX_CHARS:
                raise ValueError(
                    f"Ответ слишком длинный. Сократи его до {ONBOARDING_ANSWER_MAX_CHARS} "
                    "символов; многоабзацный текст поддерживается."
                )
            result[key] = clean
            answer_keys = {item[0] for item in ONBOARDING_QUESTIONS}
            total_chars = sum(
                len(answer)
                for answer_key, answer in result.items()
                if answer_key in answer_keys and isinstance(answer, str)
            )
            if total_chars > ONBOARDING_TOTAL_MAX_CHARS:
                raise ValueError(
                    "Ответы вместе получились слишком длинными. Сократи текущий ответ; "
                    "предыдущие ответы сохранены и шаг не изменён."
                )
        return result


@dataclass(slots=True)
class PendingIntent:
    token: str
    raw_text: str
    source: str
    result: IntentResult
    handled: bool = False
    canonical_chat_id: int | None = None
    canonical_message_id: int | None = None


WEEKDAYS_RU = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)


def temporal_context(timezone_name: str, *, now: datetime | None = None) -> dict[str, str]:
    zone = validate_timezone(timezone_name)
    current = (now or datetime.now(UTC)).astimezone(zone)
    tomorrow = current + timedelta(days=1)
    return {
        "timezone": timezone_name,
        "local_datetime": current.isoformat(timespec="seconds"),
        "today_date": current.date().isoformat(),
        "today_weekday": WEEKDAYS_RU[current.weekday()],
        "tomorrow_date": tomorrow.date().isoformat(),
        "tomorrow_weekday": WEEKDAYS_RU[tomorrow.weekday()],
    }


class IntentRouter:
    CAPTURE_INTENTS = {
        "inbox_idea": "idea",
        "inbox_task": "task",
        "inbox_desire": "desire",
        "inbox_note": "note",
        "reflection": "note",
    }

    def __init__(self, ai: AIService, confidence_threshold: float):
        self.ai = ai
        self.confidence_threshold = confidence_threshold

    async def route(
        self,
        text: str,
        timezone_name: str,
        *,
        now: datetime | None = None,
        conversation_context: dict[str, object] | None = None,
        defer_answer: bool = False,
    ) -> IntentResult:
        clean = text.strip()
        if not clean:
            raise ValueError("Пустое сообщение нельзя обработать")
        context = temporal_context(timezone_name, now=now)
        result = await self.ai.route_message(clean, context, conversation_context)
        if result.confidence < self.confidence_threshold or result.intent == "shared_idea":
            return result.model_copy(
                update={"intent": "unknown", "inbox_kind": None, "answer": None}
            )
        if result.intent in {"conversation", "question"} and not result.answer and not defer_answer:
            answer = await self.ai.answer_message(clean, context, conversation_context)
            return result.model_copy(update={"answer": answer.answer})
        kind = self.CAPTURE_INTENTS.get(result.intent)
        if result.intent == "explicit_capture":
            kind = result.inbox_kind or "note"
        if kind and result.inbox_kind != kind:
            return result.model_copy(update={"inbox_kind": kind})
        return result

    async def answer(
        self,
        text: str,
        timezone_name: str,
        *,
        now: datetime | None = None,
        conversation_context: dict[str, object] | None = None,
        confirmed_memory: NovaMemoryProjection | None = None,
    ) -> AssistantAnswer:
        if confirmed_memory is not None and confirmed_memory.records:
            return await self.ai.answer_message(
                text.strip(),
                temporal_context(timezone_name, now=now),
                conversation_context,
                confirmed_memory=confirmed_memory,
            )
        return await self.ai.answer_message(
            text.strip(), temporal_context(timezone_name, now=now), conversation_context
        )


class InboxService:
    def __init__(self, db: Database, ai: AIService, default_timezone: str):
        self.db, self.ai, self.default_timezone = db, ai, default_timezone

    async def classify(self, text: str) -> ParsedThought:
        clean = text.strip()
        if not clean:
            raise ValueError("Пустую мысль нельзя обработать")
        return await self.ai.parse_thought(clean)


class ProfileService:
    def __init__(self, db: Database, ai: AIService):
        self.db, self.ai = db, ai

    async def create(self, user_id: int, answers: dict[str, str]) -> VisionProfile:
        summary = await self.ai.summarize_vision(answers)
        async with self.db.session() as session:
            locked_owner_id = await session.scalar(
                update(User)
                .where(User.id == user_id)
                .values(updated_at=User.updated_at)
                .returning(User.id)
            )
            if locked_owner_id is None:
                raise ValueError("User not found")
            user = await session.get(User, locked_owner_id)
            if user is None:
                raise RuntimeError("Profile owner disappeared")
            if timezone := answers.get("timezone"):
                from .recurring_reminders import RecurringTaskReminderService

                await RecurringTaskReminderService(self.db).refresh_profile_timezone_in_session(
                    session,
                    user.id,
                    canonical_timezone(timezone),
                )
            user.display_name = (
                normalize_display_name(display_name, clip_legacy=True)
                if (display_name := answers.get("display_name"))
                else None
            )
            if location_value := answers.get("location"):
                location = parse_location(location_value)
                user.location_city = location.city
                user.location_fallback_city = location.fallback_city
            return await ProfileRepository(session).upsert(user, answers, summary)


class FocusService:
    def __init__(self, db: Database, ai: AIService):
        self.db, self.ai = db, ai

    async def generate(self, user_id: int, *, now: datetime | None = None) -> TodayPlan:
        plan, _weekly_focus = await self.generate_with_weekly_focus(user_id, now=now)
        return plan

    async def generate_with_weekly_focus(
        self,
        user_id: int,
        *,
        now: datetime | None = None,
    ) -> tuple[TodayPlan, str | None]:
        snapshot = await self.materialize_today_application(
            user_id,
            include_weekly_focus=True,
            now=now,
        )
        plan = await self.generate_today_plan(snapshot)
        return plan, snapshot.weekly_focus

    async def materialize_today_application(
        self,
        user_id: int,
        *,
        include_weekly_focus: bool,
        now: datetime | None = None,
    ) -> TodayApplicationSnapshot:
        current = self._utc_now(now)
        async with self.db.sessions() as session:
            user = await session.get(User, user_id)
            if user is None:
                raise ValueError("User not found")
            zone = ZoneInfo(user.timezone)
            local_today = current.astimezone(zone).date()
            local_week_start = local_today - timedelta(days=local_today.weekday())
            local_tomorrow = datetime.combine(
                local_today + timedelta(days=1),
                time.min,
                tzinfo=zone,
            ).astimezone(UTC)
            weekly_focus_public_id: str | None = None
            weekly_focus_version: int | None = None
            weekly_focus: str | None = None
            if include_weekly_focus:
                weekly_focus_row = (
                    await session.execute(
                        select(
                            WeeklyFocus.public_id,
                            WeeklyFocus.version,
                            WeeklyFocus.focus,
                        ).where(
                            WeeklyFocus.owner_id == user_id,
                            WeeklyFocus.week_start == local_week_start,
                        )
                    )
                ).one_or_none()
                if weekly_focus_row is not None:
                    weekly_focus_public_id = weekly_focus_row.public_id
                    weekly_focus_version = weekly_focus_row.version
                    weekly_focus = weekly_focus_row.focus
            profile = await session.scalar(
                select(VisionProfile).where(VisionProfile.user_id == user_id)
            )
            goals = (
                await session.scalars(
                    select(Goal)
                    .where(Goal.user_id == user_id, Goal.status == "active")
                    .order_by(Goal.priority.desc())
                    .limit(5)
                )
            ).all()
            routines = (
                await session.scalars(
                    select(Routine)
                    .where(Routine.user_id == user_id, Routine.status == "active")
                    .limit(3)
                )
            ).all()
            tasks = (
                await session.scalars(
                    select(InboxItem)
                    .where(
                        InboxItem.user_id == user_id,
                        InboxItem.status == "confirmed",
                        InboxItem.kind == "task",
                    )
                    .order_by(InboxItem.id.desc())
                    .limit(3)
                )
            ).all()
            urgent_task_rows = (
                await session.execute(
                    select(InboxItem.title, TaskState.event_at)
                    .join(
                        TaskState,
                        and_(
                            TaskState.inbox_item_id == InboxItem.id,
                            TaskState.owner_id == InboxItem.user_id,
                        ),
                    )
                    .where(
                        InboxItem.user_id == user_id,
                        InboxItem.status == "confirmed",
                        InboxItem.kind == "task",
                        TaskState.status == "active",
                        TaskState.event_at.is_not(None),
                        TaskState.event_at < local_tomorrow,
                    )
                    .order_by(TaskState.event_at, InboxItem.id)
                    .limit(TODAY_URGENT_TASK_LIMIT)
                )
            ).all()
            one_shot_rows = (
                await session.execute(
                    select(InboxItem.title, TaskReminder.remind_at)
                    .join(
                        TaskState,
                        and_(
                            TaskState.inbox_item_id == InboxItem.id,
                            TaskState.owner_id == InboxItem.user_id,
                        ),
                    )
                    .join(TaskReminder, TaskReminder.inbox_item_id == InboxItem.id)
                    .where(
                        InboxItem.user_id == user_id,
                        InboxItem.status == "confirmed",
                        InboxItem.kind == "task",
                        TaskState.status == "active",
                        TaskReminder.status.in_({"pending", "processing"}),
                        TaskReminder.remind_at >= current,
                    )
                    .order_by(TaskReminder.remind_at, TaskReminder.id)
                    .limit(TODAY_REMINDER_LIMIT)
                )
            ).all()
            daily_rows = (
                await session.execute(
                    select(
                        InboxItem.title,
                        RecurringTaskReminderSchedule.next_occurrence_at,
                        RecurringTaskReminderSchedule.local_time,
                        RecurringTaskReminderSchedule.timezone,
                    )
                    .join(
                        TaskState,
                        and_(
                            TaskState.inbox_item_id == InboxItem.id,
                            TaskState.owner_id == InboxItem.user_id,
                        ),
                    )
                    .join(
                        RecurringTaskReminderSchedule,
                        and_(
                            RecurringTaskReminderSchedule.inbox_item_id == InboxItem.id,
                            RecurringTaskReminderSchedule.owner_id == InboxItem.user_id,
                        ),
                    )
                    .where(
                        InboxItem.user_id == user_id,
                        InboxItem.status == "confirmed",
                        InboxItem.kind == "task",
                        TaskState.status == "active",
                        RecurringTaskReminderSchedule.status == "active",
                        RecurringTaskReminderSchedule.recurrence_kind == "daily",
                        RecurringTaskReminderSchedule.next_occurrence_at >= current,
                    )
                    .order_by(
                        RecurringTaskReminderSchedule.next_occurrence_at,
                        RecurringTaskReminderSchedule.id,
                    )
                    .limit(TODAY_REMINDER_LIMIT)
                )
            ).all()
            history = (
                await session.scalars(
                    select(DailyCheckIn)
                    .where(DailyCheckIn.user_id == user_id)
                    .order_by(DailyCheckIn.checkin_date.desc())
                    .limit(7)
                )
            ).all()
            companion = await session.scalar(
                select(VisionCompanionPreference).where(
                    VisionCompanionPreference.owner_id == user_id,
                    VisionCompanionPreference.enabled.is_(True),
                )
            )
            vision_focus = None
            if companion is not None:
                item = await session.scalar(
                    select(VisionItem).where(
                        VisionItem.id == companion.vision_item_id,
                        VisionItem.owner_id == user_id,
                        VisionItem.status == "active",
                    )
                )
                if item is not None:
                    vision_focus = {
                        "wish": item.wish_text,
                        "why": item.why_text,
                        "first_step": item.first_step,
                    }
            reminders = [
                {
                    "kind": "one_shot",
                    "title": row.title,
                    "next_at": self._utc_isoformat(row.remind_at),
                }
                for row in one_shot_rows
            ]
            reminders.extend(
                {
                    "kind": "daily",
                    "title": row.title,
                    "next_at": self._utc_isoformat(row.next_occurrence_at),
                    "local_time": row.local_time.isoformat(timespec="minutes"),
                    "timezone": row.timezone,
                }
                for row in daily_rows
            )
            reminders.sort(key=lambda reminder: str(reminder["next_at"]))
            context = {
                "profile": profile.summary if profile else None,
                "goals": [goal.title for goal in goals],
                "routines": [routine.normal_version for routine in routines],
                "confirmed_tasks": [task.title for task in tasks],
                "urgent_confirmed_tasks": [
                    {
                        "title": row.title,
                        "event_at": self._utc_isoformat(row.event_at),
                    }
                    for row in urgent_task_rows
                ],
                "upcoming_reminders": reminders[:TODAY_REMINDER_LIMIT],
                "recent_completed": [x for row in history for x in row.completed_actions],
                "recent_skipped": [x for row in history for x in row.skipped_actions],
                "vision_focus": vision_focus,
            }
            if include_weekly_focus:
                context["weekly_focus"] = weekly_focus
            return TodayApplicationSnapshot(
                actor_id=user.id,
                telegram_id=user.telegram_id,
                access_tier=user.access_tier,
                access_version=user.access_version,
                timezone=user.timezone,
                local_week_start=local_week_start,
                includes_weekly_focus=include_weekly_focus,
                weekly_focus_public_id=weekly_focus_public_id,
                weekly_focus_version=weekly_focus_version,
                weekly_focus=weekly_focus,
                _provider_context=context,
            )

    async def check_today_application(
        self,
        snapshot: TodayApplicationSnapshot,
        *,
        now: datetime | None = None,
    ) -> TodayApplicationCheck:
        current = self._utc_now(now)
        async with self.db.sessions() as session:
            actor = (
                await session.execute(
                    select(
                        User.id,
                        User.access_tier,
                        User.access_version,
                        User.timezone,
                    ).where(User.telegram_id == snapshot.telegram_id)
                )
            ).one_or_none()
            if actor is None or actor.id != snapshot.actor_id:
                return TodayApplicationCheck(TodayApplicationStatus.ACTOR_CHANGED)
            if (
                actor.access_tier != snapshot.access_tier
                or actor.access_version != snapshot.access_version
            ):
                return TodayApplicationCheck(TodayApplicationStatus.ACCESS_CHANGED)
            if actor.timezone != snapshot.timezone:
                return TodayApplicationCheck(TodayApplicationStatus.TIMEZONE_CHANGED)
            try:
                zone = ZoneInfo(actor.timezone)
            except ZoneInfoNotFoundError:
                return TodayApplicationCheck(TodayApplicationStatus.TIMEZONE_CHANGED)
            local_today = current.astimezone(zone).date()
            local_week_start = local_today - timedelta(days=local_today.weekday())
            if local_week_start != snapshot.local_week_start:
                return TodayApplicationCheck(TodayApplicationStatus.WEEK_CHANGED)
            if not snapshot.includes_weekly_focus:
                return TodayApplicationCheck(TodayApplicationStatus.CURRENT)
            weekly_focus_row = (
                await session.execute(
                    select(WeeklyFocus.public_id, WeeklyFocus.version).where(
                        WeeklyFocus.owner_id == actor.id,
                        WeeklyFocus.week_start == local_week_start,
                    )
                )
            ).one_or_none()
            if weekly_focus_row is None:
                focus_matches = (
                    snapshot.weekly_focus_public_id is None
                    and snapshot.weekly_focus_version is None
                )
            else:
                focus_matches = (
                    weekly_focus_row.public_id == snapshot.weekly_focus_public_id
                    and weekly_focus_row.version == snapshot.weekly_focus_version
                )
            return TodayApplicationCheck(
                TodayApplicationStatus.CURRENT
                if focus_matches
                else TodayApplicationStatus.FOCUS_CHANGED
            )

    async def generate_today_plan(self, snapshot: TodayApplicationSnapshot) -> TodayPlan:
        return await self.ai.make_today_plan(snapshot.provider_context())

    @staticmethod
    def _utc_isoformat(value: datetime) -> str:
        aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return aware.isoformat(timespec="seconds")

    @staticmethod
    def _utc_now(value: datetime | None) -> datetime:
        current = value or datetime.now(UTC)
        return current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)


_TIMEZONE_ALIASES = {
    "moscow": "Europe/Moscow",
    "москва": "Europe/Moscow",
    "мск": "Europe/Moscow",
    "msk": "Europe/Moscow",
    "europe/moscow": "Europe/Moscow",
    "gmt+3": "Europe/Moscow",
    "gmt+03": "Europe/Moscow",
    "gmt+03:00": "Europe/Moscow",
    "utc+3": "Europe/Moscow",
    "utc+03": "Europe/Moscow",
    "utc+03:00": "Europe/Moscow",
    "saratov": "Europe/Saratov",
    "саратов": "Europe/Saratov",
    "europe/saratov": "Europe/Saratov",
    "gmt+4": "Europe/Saratov",
    "gmt+04": "Europe/Saratov",
    "gmt+04:00": "Europe/Saratov",
    "utc+4": "Europe/Saratov",
    "utc+04": "Europe/Saratov",
    "utc+04:00": "Europe/Saratov",
}


def canonical_timezone(value: str) -> str:
    clean = value.strip()
    alias_key = "".join(clean.casefold().split())
    if alias := _TIMEZONE_ALIASES.get(alias_key):
        return alias
    try:
        return ZoneInfo(clean).key
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            "Неизвестный часовой пояс. Примеры: Москва, Moscow, GMT+4, "
            "Europe/Moscow или Europe/Saratov."
        ) from exc


def validate_timezone(value: str) -> ZoneInfo:
    return ZoneInfo(canonical_timezone(value))


def next_notification_utc(
    timezone_name: str, local_hour: int, *, now: datetime | None = None
) -> datetime:
    zone = validate_timezone(timezone_name)
    current = (now or datetime.now(UTC)).astimezone(zone)
    target = datetime.combine(current.date(), time(local_hour), tzinfo=zone)
    if target <= current:
        target = datetime.combine(
            date.fromordinal(current.date().toordinal() + 1), time(local_hour), tzinfo=zone
        )
    return target.astimezone(UTC)
