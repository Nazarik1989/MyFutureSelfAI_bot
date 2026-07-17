from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .db import Database
from .models import HealthCheckIn, HealthReminderPreference

METRICS = ("energy", "sleep", "mood", "stress", "physical_wellbeing")
METRIC_LABELS = {
    "energy": "энергия",
    "sleep": "сон",
    "mood": "настроение",
    "stress": "стресс",
    "physical_wellbeing": "физическое самочувствие",
    "state_score": "линейка состояния",
}
EMERGENCY_MARKERS = (
    r"(?:сильн\w*\s+)?(?:боль|давлен\w*|сдавлен\w*)\s+в\s+груд",
    r"(?:тяжел\w*|сильн\w*|резк\w*)\s+(?:одышк\w*|затруднен\w*\s+дыхан\w*)",
    r"не\s+могу\s+(?:дышать|вдохнуть|говорить)",
    r"(?:задыха\w*|не\s+хватает\s+воздух\w*)",
    r"потер\w*\s+сознани",
    r"(?:перекос\w*\s+лиц|неразборчив\w*\s+реч|онемел\w*\s+(?:рук|ног)|слабост\w*\s+одн\w*\s+сторон)",
    r"(?:судорог\w*|приступ\w*\s+судорог)",
    r"сильн\w*\s+кровотеч",
    r"(?:тяжел\w*|сильн\w*)\s+аллерг\w*\s+реакц",
    r"(?:мысл\w*\s+о\s+самоубийств|хочу\s+умереть|навредить\s+себе|"
    r"(?:причинить|нанести)\s+себе\s+вред)",
)
NEGATION = re.compile(
    r"\b(?:нет|без|не|не было|не испытываю|не чувствую)\b",
    re.IGNORECASE,
)
WEAKNESS = re.compile(r"\b(?:слабост\w*|усталост\w*|нет\s+сил)\b", re.IGNORECASE)
LONG_DURATION = re.compile(
    r"\b(?:несколько\s+недель|недел\w+|долго|длительн\w*|не\s+проход\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class WeeklyHealthReport:
    current_count: int
    previous_count: int
    current: dict[str, float]
    changes: dict[str, float | None]


def subjective_score(
    energy: int,
    sleep: int,
    mood: int,
    stress: int,
    physical_wellbeing: int,
) -> int:
    values = (energy, sleep, mood, stress, physical_wellbeing)
    if any(value < 0 or value > 10 for value in values):
        raise ValueError("Health ratings must be from 0 to 10")
    positive_total = energy + sleep + mood + physical_wellbeing + (10 - stress)
    return round(positive_total * 2)


def urgent_safety_message(symptoms: str | None) -> str | None:
    if not symptoms:
        return None
    normalized = " ".join(symptoms.lower().replace("ё", "е").split())
    for pattern in EMERGENCY_MARKERS:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match is None:
            continue
        prefix = normalized[max(0, match.start() - 40) : match.start()]
        local_clause = re.split(r"[,;.]|\bно\b", prefix)[-1]
        if NEGATION.search(local_clause):
            continue
        return (
            "⚠️ Такие симптомы могут требовать срочной оценки. Если это происходит сейчас "
            "или состояние ухудшается, немедленно обратитесь в местную экстренную медицинскую "
            "службу. Не оставайтесь в одиночестве. Бот не ставит диагноз."
        )
    return None


def prolonged_weakness_message(symptoms: str | None, recent_weakness_days: int) -> str | None:
    if not symptoms or not WEAKNESS.search(symptoms):
        return None
    if not LONG_DURATION.search(symptoms) and recent_weakness_days < 3:
        return None
    return (
        "Слабость держится достаточно долго, чтобы обсудить её с врачом. Можно записаться "
        "на приём и подготовить наблюдения: когда началось, как меняется, сон, аппетит, "
        "температура, одышка или боль, что облегчает или усиливает состояние, какие лекарства "
        "вы уже принимаете. Это не диагноз и не назначение лечения или анализов."
    )


class HealthService:
    def __init__(self, db: Database):
        self.db = db

    async def save(
        self,
        *,
        user_id: int,
        timezone: str,
        answers: dict[str, object],
        record_id: int | None = None,
        now: datetime | None = None,
    ) -> HealthCheckIn | None:
        local_date = (now or datetime.now(UTC)).astimezone(ZoneInfo(timezone)).date()
        async with self.db.session() as session:
            record = (
                await session.get(HealthCheckIn, record_id)
                if record_id is not None
                else await session.scalar(
                    select(HealthCheckIn).where(
                        HealthCheckIn.user_id == user_id,
                        HealthCheckIn.local_date == local_date,
                    )
                )
            )
            if record_id is not None and record is None:
                return None
            if record is not None and record.user_id != user_id:
                return None
            values = {name: int(answers[name]) for name in METRICS}
            score = subjective_score(**values)
            symptoms = str(answers.get("symptoms") or "").strip()[:1000] or None
            if record is None:
                record = HealthCheckIn(
                    user_id=user_id,
                    local_date=local_date,
                    timezone=timezone,
                    **values,
                    symptoms=symptoms,
                    state_score=score,
                )
                session.add(record)
            else:
                for name, value in values.items():
                    setattr(record, name, value)
                record.symptoms = symptoms
                record.state_score = score
                record.timezone = timezone
            await session.flush()
            return record

    async def history(self, user_id: int, *, limit: int = 14) -> list[HealthCheckIn]:
        async with self.db.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(HealthCheckIn)
                        .where(HealthCheckIn.user_id == user_id)
                        .order_by(HealthCheckIn.local_date.desc(), HealthCheckIn.id.desc())
                        .limit(limit)
                    )
                ).all()
            )

    async def get_owned(self, user_id: int, record_id: int) -> HealthCheckIn | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(HealthCheckIn).where(
                    HealthCheckIn.id == record_id,
                    HealthCheckIn.user_id == user_id,
                )
            )

    async def delete_owned(self, user_id: int, record_id: int) -> bool:
        async with self.db.session() as session:
            record = await session.scalar(
                select(HealthCheckIn).where(
                    HealthCheckIn.id == record_id,
                    HealthCheckIn.user_id == user_id,
                )
            )
            if record is None:
                return False
            await session.delete(record)
            return True

    async def weekly_report(
        self,
        user_id: int,
        timezone: str,
        *,
        now: datetime | None = None,
    ) -> WeeklyHealthReport:
        today = (now or datetime.now(UTC)).astimezone(ZoneInfo(timezone)).date()
        records = await self.history(user_id, limit=30)
        current_records = [
            record for record in records if today - timedelta(days=6) <= record.local_date <= today
        ]
        previous_records = [
            record
            for record in records
            if today - timedelta(days=13) <= record.local_date <= today - timedelta(days=7)
        ]
        names = (*METRICS, "state_score")
        current = {
            name: mean(getattr(record, name) for record in current_records)
            for name in names
            if current_records
        }
        previous = {
            name: mean(getattr(record, name) for record in previous_records)
            for name in names
            if previous_records
        }
        changes = {
            name: current[name] - previous[name] if name in previous else None for name in current
        }
        return WeeklyHealthReport(
            current_count=len(current_records),
            previous_count=len(previous_records),
            current=current,
            changes=changes,
        )

    async def recent_weakness_days(self, user_id: int) -> int:
        records = await self.history(user_id, limit=7)
        return sum(bool(record.symptoms and WEAKNESS.search(record.symptoms)) for record in records)

    async def set_reminder(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        chat_id: int,
        timezone: str,
        local_time: time,
        enabled: bool,
    ) -> HealthReminderPreference:
        async with self.db.session() as session:
            preference = await session.scalar(
                select(HealthReminderPreference).where(HealthReminderPreference.user_id == user_id)
            )
            if preference is None:
                preference = HealthReminderPreference(
                    user_id=user_id,
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    timezone=timezone,
                    local_time=local_time,
                    enabled=enabled,
                )
                session.add(preference)
            else:
                preference.telegram_user_id = telegram_user_id
                preference.chat_id = chat_id
                preference.timezone = timezone
                preference.local_time = local_time
                preference.enabled = enabled
            await session.flush()
            return preference

    async def disable_reminder(self, user_id: int) -> bool:
        async with self.db.session() as session:
            preference = await session.scalar(
                select(HealthReminderPreference).where(HealthReminderPreference.user_id == user_id)
            )
            if preference is None:
                return False
            preference.enabled = False
            return True

    async def reminder_preferences(self) -> list[HealthReminderPreference]:
        async with self.db.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(HealthReminderPreference).where(
                            HealthReminderPreference.enabled.is_(True)
                        )
                    )
                ).all()
            )
