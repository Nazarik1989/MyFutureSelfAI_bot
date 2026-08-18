from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from .domain import next_notification_utc
from .reminders import TaskReminderEngine

SendCallback = Callable[[int, str], Awaitable[int | None]]
VisionSendCallback = Callable[[int, str], Awaitable[None]]
CanSendCallback = Callable[[int], Awaitable[bool]]
WeeklyReviewSendCallback = Callable[[int, str], Awaitable[None]]


class Scheduler(Protocol):
    def schedule_user(self, telegram_id: int, timezone: str) -> None: ...

    def remove_user(self, telegram_id: int) -> None: ...


class VisionCompanionSchedule(Protocol):
    id: int
    owner_id: int
    telegram_user_id: int
    timezone: str
    morning_time: time
    evening_time: time
    extra_times: list[str]


class DeliverDueEngine(Protocol):
    async def deliver_due(self) -> int: ...


class JobQueueScheduler:
    """Small adapter around PTB JobQueue, replaceable by a worker later."""

    def __init__(
        self,
        job_queue: object,
        send: SendCallback,
        morning_hour: int,
        evening_hour: int,
        weekly_weekday: int,
        weekly_enabled: bool = True,
        vision_send: VisionSendCallback | None = None,
        can_send: CanSendCallback | None = None,
        weekly_review_send: WeeklyReviewSendCallback | None = None,
        weekly_can_send: CanSendCallback | None = None,
    ):
        self.job_queue = job_queue
        self.send = send
        self.morning_hour = morning_hour
        self.evening_hour = evening_hour
        self.weekly_weekday = weekly_weekday
        self.weekly_enabled = weekly_enabled
        self.vision_send = vision_send
        self.can_send = can_send
        self.weekly_review_send = weekly_review_send
        self.weekly_can_send = weekly_can_send

    @staticmethod
    def next_run(timezone: str, hour: int, now: datetime | None = None) -> datetime:
        return next_notification_utc(timezone, hour, now=now)

    def schedule_user(self, telegram_id: int, timezone: str) -> None:
        self.remove_user(telegram_id)
        self._schedule_daily(telegram_id, timezone, "morning", self.morning_hour)
        self._schedule_daily(telegram_id, timezone, "evening", self.evening_hour)
        if self.weekly_enabled:
            self._schedule_weekly(telegram_id, timezone)

    def _schedule_daily(self, telegram_id: int, timezone: str, kind: str, hour: int) -> None:
        callback = self._morning if kind == "morning" else self._evening
        self.job_queue.run_once(
            callback,
            when=self.next_run(timezone, hour),
            data={"telegram_id": telegram_id, "timezone": timezone},
            name=f"user:{telegram_id}:{kind}",
        )

    def _schedule_weekly(self, telegram_id: int, timezone: str) -> None:
        zone = ZoneInfo(timezone)
        now = datetime.now(UTC).astimezone(zone)
        days = (self.weekly_weekday - now.weekday()) % 7
        target = datetime.combine(now.date() + timedelta(days=days), time(18), tzinfo=zone)
        if target <= now:
            target += timedelta(days=7)
        self.job_queue.run_once(
            self._weekly,
            when=target.astimezone(UTC),
            data={"telegram_id": telegram_id, "timezone": timezone},
            name=f"user:{telegram_id}:weekly",
        )

    async def _morning(self, context: object) -> None:
        data = context.job.data
        if await self._can_send(data["telegram_id"]):
            await self.send(data["telegram_id"], "/today — выбери небольшой фокус на сегодня.")
        self._schedule_daily(data["telegram_id"], data["timezone"], "morning", self.morning_hour)

    async def _evening(self, context: object) -> None:
        data = context.job.data
        if await self._can_send(data["telegram_id"]):
            await self.send(data["telegram_id"], "Время короткой рефлексии: /evening")
        self._schedule_daily(data["telegram_id"], data["timezone"], "evening", self.evening_hour)

    async def _weekly(self, context: object) -> None:
        data = context.job.data
        try:
            if self.weekly_enabled and await self._can_send_weekly(data["telegram_id"]):
                if self.weekly_review_send is not None:
                    await self.weekly_review_send(data["telegram_id"], data["timezone"])
                else:
                    await self.send(
                        data["telegram_id"],
                        "Пора спокойно посмотреть на неделю и скорректировать систему.",
                    )
        finally:
            self._schedule_weekly(data["telegram_id"], data["timezone"])

    def remove_user(self, telegram_id: int) -> None:
        get_jobs = getattr(self.job_queue, "get_jobs_by_name", None)
        if get_jobs:
            for suffix in ("morning", "evening", "weekly"):
                for job in get_jobs(f"user:{telegram_id}:{suffix}"):
                    job.schedule_removal()

    def start_task_reminders(
        self,
        engine: TaskReminderEngine,
        *,
        interval_seconds: int,
    ) -> None:
        async def deliver_due(context: object) -> None:
            await engine.deliver_due()

        self.job_queue.run_repeating(
            deliver_due,
            interval=interval_seconds,
            first=interval_seconds,
            name="task-reminders:persistent-outbox",
        )

    def start_recurring_task_reminders(
        self,
        engine: DeliverDueEngine,
        *,
        interval_seconds: int,
    ) -> None:
        async def deliver_due(context: object) -> None:
            del context
            await engine.deliver_due()

        self.job_queue.run_repeating(
            deliver_due,
            interval=interval_seconds,
            first=interval_seconds,
            name="recurring-task-reminders:persistent-outbox",
        )

    def schedule_health_reminder(
        self,
        *,
        user_id: int,
        chat_id: int,
        timezone: str,
        local_time: time,
    ) -> None:
        self.remove_health_reminder(user_id)
        zone = ZoneInfo(timezone)
        self.job_queue.run_daily(
            self._health_checkin,
            time=local_time.replace(tzinfo=zone),
            data={
                "user_id": user_id,
                "chat_id": chat_id,
            },
            name=f"health:{user_id}:daily",
        )

    async def _health_checkin(self, context: object) -> None:
        data = context.job.data
        if await self._can_send(data["chat_id"]):
            await self.send(
                data["chat_id"],
                "Добровольный health check-in: /checkin. Это самонаблюдение, не медицинский диагноз.",
            )

    async def _can_send(self, telegram_id: int) -> bool:
        return self.can_send is None or await self.can_send(telegram_id)

    async def _can_send_weekly(self, telegram_id: int) -> bool:
        if self.weekly_can_send is not None:
            return await self.weekly_can_send(telegram_id)
        return await self._can_send(telegram_id)

    def remove_health_reminder(self, user_id: int) -> None:
        get_jobs = getattr(self.job_queue, "get_jobs_by_name", None)
        if get_jobs:
            for job in get_jobs(f"health:{user_id}:daily"):
                job.schedule_removal()

    def schedule_vision_companion(self, preference: VisionCompanionSchedule) -> None:
        if self.vision_send is None:
            return
        preference_id = int(preference.id)
        owner_id = int(preference.owner_id)
        telegram_user_id = int(preference.telegram_user_id)
        timezone = str(preference.timezone)
        zone = ZoneInfo(timezone)
        self.remove_vision_companion(owner_id)
        get_jobs = getattr(self.job_queue, "get_jobs_by_name", None)
        if get_jobs:
            for suffix in ("morning", "evening"):
                for job in get_jobs(f"user:{telegram_user_id}:{suffix}"):
                    job.schedule_removal()
        moments = [
            ("morning", preference.morning_time),
            ("evening", preference.evening_time),
        ]
        moments.extend(
            (f"extra-{index}", time.fromisoformat(raw))
            for index, raw in enumerate(preference.extra_times, start=1)
        )
        for moment, local_time in moments:
            self.job_queue.run_daily(
                self._vision_companion,
                time=local_time.replace(tzinfo=zone),
                data={"preference_id": preference_id, "moment": moment},
                name=f"vision-companion:{owner_id}:{moment}",
            )

    async def _vision_companion(self, context: object) -> None:
        if self.vision_send is None:
            return
        data = context.job.data
        await self.vision_send(data["preference_id"], data["moment"])

    def remove_vision_companion(self, owner_id: int) -> None:
        get_jobs = getattr(self.job_queue, "jobs", None)
        if get_jobs is not None:
            prefix = f"vision-companion:{owner_id}:"
            for job in tuple(get_jobs()):
                if getattr(job, "name", "").startswith(prefix):
                    job.schedule_removal()
            return
        by_name = getattr(self.job_queue, "get_jobs_by_name", None)
        if by_name:
            for suffix in ("morning", "evening", "extra-1", "extra-2", "extra-3"):
                for job in by_name(f"vision-companion:{owner_id}:{suffix}"):
                    job.schedule_removal()
