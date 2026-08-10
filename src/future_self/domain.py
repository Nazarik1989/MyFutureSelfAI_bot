from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select, update

from .ai import AIService
from .db import Database
from .location import parse_location
from .models import (
    DailyCheckIn,
    Goal,
    InboxItem,
    Routine,
    User,
    VisionCompanionPreference,
    VisionItem,
    VisionProfile,
)
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
        if result.intent in {"conversation", "question"} and not result.answer:
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
    ) -> AssistantAnswer:
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

    async def generate(self, user_id: int) -> TodayPlan:
        async with self.db.sessions() as session:
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
            context = {
                "profile": profile.summary if profile else None,
                "goals": [goal.title for goal in goals],
                "routines": [routine.normal_version for routine in routines],
                "confirmed_tasks": [task.title for task in tasks],
                "recent_completed": [x for row in history for x in row.completed_actions],
                "recent_skipped": [x for row in history for x in row.skipped_actions],
                "vision_focus": vision_focus,
            }
        return await self.ai.make_today_plan(context)


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
