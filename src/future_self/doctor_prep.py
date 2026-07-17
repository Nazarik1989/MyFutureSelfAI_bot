from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .db import Database
from .health import METRIC_LABELS, HealthService
from .models import DoctorVisitPrep, InboxItem, TaskReminder
from .reminders import reminder_for_inbox_item
from .schemas import TemporalResolution

EMPTY_ANSWERS = {"нет", "не принимаю", "нет лекарств", "нет вопросов", "-"}


@dataclass(frozen=True, slots=True)
class DoctorTaskResult:
    status: str
    inbox_item: InboxItem | None = None
    reminder: TaskReminder | None = None


class _TaskAlreadyCreated(RuntimeError):
    pass


class DoctorVisitPrepService:
    def __init__(
        self,
        db: Database,
        *,
        task_date_event_hour: int = 9,
        task_reminder_lead_minutes: int = 30,
    ):
        self.db = db
        self.health = HealthService(db)
        self.task_date_event_hour = task_date_event_hour
        self.task_reminder_lead_minutes = task_reminder_lead_minutes

    async def save(
        self,
        *,
        user_id: int,
        timezone: str,
        answers: dict[str, object],
        record_id: int | None = None,
    ) -> DoctorVisitPrep | None:
        values = {
            "reason": self._required(answers, "reason", 1000),
            "duration": self._required(answers, "duration", 500),
            "symptoms": self._required(answers, "symptoms", 2000),
            "medications": self._optional(answers, "medications", 2000),
            "questions": self._optional(answers, "questions", 2000),
        }
        snapshot = await self._health_snapshot(user_id, timezone)
        summary = self.format_summary(values, snapshot)
        async with self.db.session() as session:
            record = (
                await session.get(DoctorVisitPrep, record_id) if record_id is not None else None
            )
            if record_id is not None and (record is None or record.user_id != user_id):
                return None
            if record is None:
                record = DoctorVisitPrep(
                    user_id=user_id,
                    timezone=timezone,
                    health_snapshot=snapshot,
                    summary=summary,
                    **values,
                )
                session.add(record)
            else:
                for name, value in values.items():
                    setattr(record, name, value)
                record.timezone = timezone
                record.health_snapshot = snapshot
                record.summary = summary
            await session.flush()
            return record

    async def history(self, user_id: int, *, limit: int = 10) -> list[DoctorVisitPrep]:
        async with self.db.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(DoctorVisitPrep)
                        .where(DoctorVisitPrep.user_id == user_id)
                        .order_by(DoctorVisitPrep.created_at.desc(), DoctorVisitPrep.id.desc())
                        .limit(limit)
                    )
                ).all()
            )

    async def get_owned(self, user_id: int, record_id: int) -> DoctorVisitPrep | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(DoctorVisitPrep).where(
                    DoctorVisitPrep.id == record_id,
                    DoctorVisitPrep.user_id == user_id,
                )
            )

    async def delete_owned(self, user_id: int, record_id: int) -> bool:
        async with self.db.session() as session:
            record = await session.scalar(
                select(DoctorVisitPrep).where(
                    DoctorVisitPrep.id == record_id,
                    DoctorVisitPrep.user_id == user_id,
                )
            )
            if record is None:
                return False
            await session.delete(record)
            return True

    async def create_appointment_task(
        self,
        *,
        user_id: int,
        record_id: int,
        telegram_user_id: int,
        chat_id: int,
        temporal: TemporalResolution,
    ) -> DoctorTaskResult:
        try:
            async with self.db.session() as session:
                record = await session.scalar(
                    select(DoctorVisitPrep).where(
                        DoctorVisitPrep.id == record_id,
                        DoctorVisitPrep.user_id == user_id,
                    )
                )
                if record is None:
                    return DoctorTaskResult("missing")
                if record.appointment_inbox_item_id is not None:
                    return await self._existing_task(session, record.appointment_inbox_item_id)

                item = InboxItem(
                    user_id=user_id,
                    kind="task",
                    title="Записаться к врачу",
                    description=f"Подготовка к визиту #{record.id}",
                    raw_text=f"doctor_prepare:{record.id}:appointment",
                    next_step="Связаться с клиникой",
                    resolved_date=temporal.resolved_local_date,
                    temporal_resolution=temporal.model_dump(mode="json"),
                    source="doctor_prepare",
                    status="confirmed",
                )
                session.add(item)
                await session.flush()
                reminder = reminder_for_inbox_item(
                    item,
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    date_event_hour=self.task_date_event_hour,
                    lead_minutes=self.task_reminder_lead_minutes,
                )
                if reminder is not None:
                    session.add(reminder)
                claimed = await session.execute(
                    update(DoctorVisitPrep)
                    .where(
                        DoctorVisitPrep.id == record_id,
                        DoctorVisitPrep.user_id == user_id,
                        DoctorVisitPrep.appointment_inbox_item_id.is_(None),
                    )
                    .values(appointment_inbox_item_id=item.id)
                    .returning(DoctorVisitPrep.id)
                )
                if claimed.scalar_one_or_none() is None:
                    raise _TaskAlreadyCreated
                await session.flush()
                return DoctorTaskResult("created", item, reminder)
        except _TaskAlreadyCreated:
            async with self.db.sessions() as session:
                record = await session.scalar(
                    select(DoctorVisitPrep).where(
                        DoctorVisitPrep.id == record_id,
                        DoctorVisitPrep.user_id == user_id,
                    )
                )
                if record is None or record.appointment_inbox_item_id is None:
                    return DoctorTaskResult("missing")
                return await self._existing_task(session, record.appointment_inbox_item_id)

    @staticmethod
    async def _existing_task(session: AsyncSession, inbox_item_id: int) -> DoctorTaskResult:
        item = await session.get(InboxItem, inbox_item_id)
        reminder = await session.scalar(
            select(TaskReminder).where(TaskReminder.inbox_item_id == inbox_item_id)
        )
        return DoctorTaskResult("existing", item, reminder)

    async def _health_snapshot(self, user_id: int, timezone: str) -> dict[str, Any]:
        records = await self.health.history(user_id, limit=14)
        report = await self.health.weekly_report(user_id, timezone)
        latest = records[0] if records else None
        return {
            "checkin_count": len(records),
            "latest_date": latest.local_date.isoformat() if latest else None,
            "latest": (
                {
                    "energy": latest.energy,
                    "sleep": latest.sleep,
                    "mood": latest.mood,
                    "stress": latest.stress,
                    "physical_wellbeing": latest.physical_wellbeing,
                    "state_score": latest.state_score,
                }
                if latest
                else None
            ),
            "week_count": report.current_count,
            "week_average": {name: round(value, 1) for name, value in report.current.items()},
            "week_change": {
                name: round(value, 1) if value is not None else None
                for name, value in report.changes.items()
            },
        }

    @staticmethod
    def format_summary(values: dict[str, str | None], snapshot: dict[str, Any]) -> str:
        lines = [
            "Краткое фактическое резюме для врача",
            f"Причина обращения: {DoctorVisitPrepService._brief(values['reason'], 250)}",
            f"Длительность: {DoctorVisitPrepService._brief(values['duration'], 120)}",
            f"Симптомы и наблюдения: {DoctorVisitPrepService._brief(values['symptoms'], 500)}",
            "Лекарства и добавки: "
            f"{DoctorVisitPrepService._brief(values['medications'], 350) or 'не указаны'}",
            "Вопросы врачу: "
            f"{DoctorVisitPrepService._brief(values['questions'], 350) or 'не указаны'}",
        ]
        latest = snapshot.get("latest")
        if latest:
            lines.append(
                "Health Track: "
                f"{snapshot['checkin_count']} check-in; последнее {snapshot['latest_date']}; "
                f"линейка {latest['state_score']}/100; энергия {latest['energy']}/10; "
                f"сон {latest['sleep']}/10; настроение {latest['mood']}/10; "
                f"стресс {latest['stress']}/10; "
                f"физическое самочувствие {latest['physical_wellbeing']}/10."
            )
            averages = snapshot.get("week_average") or {}
            changes = snapshot.get("week_change") or {}
            if averages:
                metric_parts = []
                for name in ("state_score", "energy", "sleep", "mood", "stress"):
                    if name not in averages:
                        continue
                    change = changes.get(name)
                    suffix = "" if change is None else f" ({change:+.1f})"
                    metric_parts.append(f"{METRIC_LABELS[name]} {averages[name]:.1f}{suffix}")
                lines.append(
                    f"Среднее за 7 дней ({snapshot['week_count']} check-in): "
                    + "; ".join(metric_parts)
                    + "."
                )
        else:
            lines.append("Health Track: данных check-in пока нет.")
        lines.append(
            "Это пользовательские наблюдения для подготовки к визиту, не медицинский диагноз."
        )
        return "\n".join(lines)

    @staticmethod
    def _required(answers: dict[str, object], name: str, limit: int) -> str:
        value = str(answers.get(name) or "").strip()
        if not value:
            raise ValueError(f"{name} is required")
        return value[:limit]

    @staticmethod
    def _optional(answers: dict[str, object], name: str, limit: int) -> str | None:
        value = str(answers.get(name) or "").strip()
        return None if value.lower() in EMPTY_ANSWERS or not value else value[:limit]

    @staticmethod
    def _brief(value: str | None, limit: int) -> str | None:
        if value is None:
            return None
        clean = " ".join(value.split())
        return clean if len(clean) <= limit else f"{clean[: limit - 3]}..."
