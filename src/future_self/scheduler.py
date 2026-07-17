from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from .domain import next_notification_utc
from .reminders import TaskReminderEngine

SendCallback = Callable[[int, str], Awaitable[int | None]]


class Scheduler(Protocol):
    def schedule_user(self, telegram_id: int, timezone: str) -> None: ...

    def remove_user(self, telegram_id: int) -> None: ...


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
    ):
        self.job_queue = job_queue
        self.send = send
        self.morning_hour = morning_hour
        self.evening_hour = evening_hour
        self.weekly_weekday = weekly_weekday
        self.weekly_enabled = weekly_enabled

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
        await self.send(data["telegram_id"], "/today — выбери небольшой фокус на сегодня.")
        self._schedule_daily(data["telegram_id"], data["timezone"], "morning", self.morning_hour)

    async def _evening(self, context: object) -> None:
        data = context.job.data
        await self.send(data["telegram_id"], "Время короткой рефлексии: /evening")
        self._schedule_daily(data["telegram_id"], data["timezone"], "evening", self.evening_hour)

    async def _weekly(self, context: object) -> None:
        data = context.job.data
        await self.send(
            data["telegram_id"], "Пора спокойно посмотреть на неделю и скорректировать систему."
        )
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
