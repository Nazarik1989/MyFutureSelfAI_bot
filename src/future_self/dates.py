import re
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from .schemas import TemporalResolution

WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)
WEEKDAY_FORMS = {
    "понедельник": 0,
    "понедельника": 0,
    "вторник": 1,
    "вторника": 1,
    "среда": 2,
    "среду": 2,
    "среды": 2,
    "четверг": 3,
    "четверга": 3,
    "пятница": 4,
    "пятницу": 4,
    "пятницы": 4,
    "суббота": 5,
    "субботу": 5,
    "субботы": 5,
    "воскресенье": 6,
    "воскресенья": 6,
}
MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


class DateOption(BaseModel):
    value: date
    weekday: str


class DateResolution(BaseModel):
    status: Literal["none", "resolved", "conflict"]
    target_date: date | None = None
    stated_weekday: str | None = None
    actual_weekday: str | None = None
    inferred_year: bool = False
    options: list[DateOption] = Field(default_factory=list)


class DateResolver:
    DATE_PATTERN = re.compile(
        r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|"
        r"сентября|октября|ноября|декабря)(?:\s+(\d{4}))?\b",
        re.IGNORECASE,
    )

    def resolve(
        self, text: str, timezone_name: str, *, now: datetime | None = None
    ) -> DateResolution:
        local_now = (now or datetime.now(UTC)).astimezone(ZoneInfo(timezone_name))
        today = local_now.date()
        lowered = text.lower().replace("ё", "е")
        relative = self._relative(lowered, today)
        stated_weekday = self._weekday(lowered)
        match = self.DATE_PATTERN.search(lowered)
        inferred_year = False
        target = relative
        if match:
            day = int(match.group(1))
            month = MONTHS[match.group(2)]
            if match.group(3):
                year = int(match.group(3))
            else:
                year = today.year
                inferred_year = True
            try:
                target = date(year, month, day)
                if inferred_year and target < today:
                    target = date(year + 1, month, day)
            except ValueError:
                return DateResolution(status="none")
        elif target is None and stated_weekday is not None:
            days = (stated_weekday - today.weekday()) % 7
            target = today + timedelta(days=days or 7)
        if target is None:
            return DateResolution(status="none")
        actual = target.weekday()
        if stated_weekday is not None and stated_weekday != actual:
            nearest = self._nearest_weekday(target, stated_weekday)
            return DateResolution(
                status="conflict",
                target_date=target,
                stated_weekday=WEEKDAYS[stated_weekday],
                actual_weekday=WEEKDAYS[actual],
                inferred_year=inferred_year,
                options=[
                    DateOption(value=nearest, weekday=WEEKDAYS[nearest.weekday()]),
                    DateOption(value=target, weekday=WEEKDAYS[actual]),
                ],
            )
        return DateResolution(
            status="resolved",
            target_date=target,
            stated_weekday=WEEKDAYS[stated_weekday] if stated_weekday is not None else None,
            actual_weekday=WEEKDAYS[actual],
            inferred_year=inferred_year,
        )

    def choose_option(self, text: str, options: list[dict[str, str]]) -> DateOption | None:
        """Select only among persisted conflict options; never recalculate a weekday."""
        lowered = text.lower().replace("ё", "е")
        matches: list[DateOption] = []
        stated_weekday = self._weekday(lowered)
        numbers = {int(value) for value in re.findall(r"\b\d{1,2}\b", lowered)}
        for raw in options:
            value = date.fromisoformat(raw["value"])
            weekday = WEEKDAYS[value.weekday()]
            if raw["weekday"] != weekday:
                continue
            day_matches = value.day in numbers
            weekday_matches = stated_weekday is not None and value.weekday() == stated_weekday
            iso_matches = value.isoformat() in lowered
            short_matches = value.strftime("%d.%m.%Y") in lowered
            if iso_matches or short_matches or day_matches or weekday_matches:
                matches.append(DateOption(value=value, weekday=weekday))
        unique = {match.value: match for match in matches}
        return next(iter(unique.values())) if len(unique) == 1 else None

    @staticmethod
    def extract_local_time(text: str) -> time | None:
        colon = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
        if colon:
            return time(int(colon.group(1)), int(colon.group(2)))
        hour = re.search(
            r"\bв\s+([01]?\d|2[0-3])(?:\s*(?:час(?:а|ов)?))?\b",
            text.lower(),
        )
        if hour:
            return time(int(hour.group(1)), 0)
        return None

    @staticmethod
    def temporal_resolution(
        selected_date: date,
        timezone_name: str,
        original_expression: str,
        local_time: time | None,
    ) -> TemporalResolution:
        zone = ZoneInfo(timezone_name)
        local_datetime = datetime.combine(selected_date, local_time or time.min, tzinfo=zone)
        return TemporalResolution(
            resolved_at=local_datetime.astimezone(UTC),
            timezone=timezone_name,
            resolved_local_date=selected_date,
            resolved_local_time=local_time,
            precision="datetime" if local_time else "date",
            original_expression=original_expression,
            resolution_status="resolved",
        )

    @staticmethod
    def conflict_message(result: DateResolution) -> str:
        first, second = result.options
        return (
            f"Есть несоответствие: {result.target_date.strftime('%d.%m.%Y')} — "
            f"{result.actual_weekday}, а не {result.stated_weekday}. "
            f"Ближайшие варианты: {first.weekday} {first.value.strftime('%d.%m.%Y')} "
            f"или {second.weekday} {second.value.strftime('%d.%m.%Y')}. Что выбрать?"
        )

    @staticmethod
    def interpretation_message(result: DateResolution) -> str | None:
        if result.status != "resolved" or not result.inferred_year:
            return None
        return (
            f"Понимаю дату как {result.target_date.strftime('%d.%m.%Y')}, {result.actual_weekday}."
        )

    @staticmethod
    def _relative(text: str, today: date) -> date | None:
        if "послезавтра" in text:
            return today + timedelta(days=2)
        if "завтра" in text:
            return today + timedelta(days=1)
        if "сегодня" in text:
            return today
        return None

    @staticmethod
    def _weekday(text: str) -> int | None:
        for form, number in WEEKDAY_FORMS.items():
            if re.search(rf"\b{form}\b", text):
                return number
        return None

    @staticmethod
    def _nearest_weekday(target: date, weekday: int) -> date:
        before = target - timedelta(days=(target.weekday() - weekday) % 7)
        after = target + timedelta(days=(weekday - target.weekday()) % 7)
        if before == target:
            return target
        return before if (target - before) <= (after - target) else after
