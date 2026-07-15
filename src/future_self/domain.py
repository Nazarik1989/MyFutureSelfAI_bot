from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select

from .ai import AIService
from .db import Database
from .models import DailyCheckIn, Goal, InboxItem, Routine, User, VisionProfile
from .repositories import ProfileRepository
from .schemas import AssistantAnswer, IntentResult, ParsedThought, TodayPlan

ONBOARDING_QUESTIONS: tuple[tuple[str, str, bool], ...] = (
    ("display_name", "Как мне к тебе обращаться?", True),
    ("timezone", "В каком часовом поясе ты живёшь? Например, Europe/Moscow.", True),
    ("future_life", "Как выглядит твоя жизнь через три года?", True),
    ("residence", "Где ты живёшь в этом образе?", False),
    ("work_income", "Чем занимаешься и какой уровень дохода хочешь?", False),
    ("health_body", "Как ты описал(а) бы желаемое здоровье и состояние тела?", False),
    ("relationships", "Какие отношения и окружение тебя поддерживают?", False),
    ("ideal_day", "Как проходит твой идеальный обычный день?", True),
    ("values", "Какие ценности для тебя главные?", True),
    ("obstacles", "Что сейчас чаще всего мешает двигаться к этому?", False),
    ("support_style", "Какой стиль поддержки тебе подходит?", True),
)


class OnboardingFlow:
    @staticmethod
    def next_step(step: int) -> int:
        return min(step + 1, len(ONBOARDING_QUESTIONS))

    @staticmethod
    def previous_step(step: int) -> int:
        return max(step - 1, 0)

    @staticmethod
    def answer(answers: dict[str, Any], step: int, value: str | None) -> dict[str, Any]:
        result = dict(answers)
        key, _, required = ONBOARDING_QUESTIONS[step]
        if value is None and required:
            raise ValueError("Этот вопрос нельзя пропустить")
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value.strip()
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
            user = await session.get(User, user_id)
            if user is None:
                raise ValueError("User not found")
            if timezone := answers.get("timezone"):
                validate_timezone(timezone)
                user.timezone = timezone
            user.display_name = answers.get("display_name")
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
            context = {
                "profile": profile.summary if profile else None,
                "goals": [goal.title for goal in goals],
                "routines": [routine.normal_version for routine in routines],
                "confirmed_tasks": [task.title for task in tasks],
                "recent_completed": [x for row in history for x in row.completed_actions],
                "recent_skipped": [x for row in history for x in row.skipped_actions],
            }
        return await self.ai.make_today_plan(context)


def validate_timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Неизвестный часовой пояс") from exc


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
