from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select

from .access import FULL_ACCESS_TIERS
from .db import Database
from .models import (
    User,
    VisionCompanionCheckIn,
    VisionCompanionPreference,
    VisionItem,
    VisionItemImage,
)

MOMENTS = {"morning", "evening", "extra"}
RESPONSES = {"committed", "pause", "done", "partial", "missed", "later"}


@dataclass(frozen=True, slots=True)
class CompanionSnapshot:
    preference_id: int
    owner_id: int
    item_id: int
    chat_id: int
    timezone: str
    wish_text: str
    why_text: str | None
    first_step: str | None
    image_bytes: bytes | None


def companion_extra_times(morning: time, evening: time, count: int) -> list[time]:
    if count not in {0, 1, 2, 3}:
        raise ValueError("Допустимо от 0 до 3 дополнительных напоминаний.")
    if count == 0:
        return []
    start = morning.hour * 60 + morning.minute
    end = evening.hour * 60 + evening.minute
    if end <= start:
        end += 24 * 60
    span = end - start
    return [
        time(
            hour=((start + round(span * index / (count + 1))) // 60) % 24,
            minute=(start + round(span * index / (count + 1))) % 60,
        )
        for index in range(1, count + 1)
    ]


class VisionCompanionService:
    """Owner-scoped, opt-in accompaniment for one active vision item per user."""

    def __init__(self, db: Database):
        self.db = db

    async def get(self, owner_id: int) -> VisionCompanionPreference | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(VisionCompanionPreference).where(
                    VisionCompanionPreference.owner_id == owner_id
                )
            )

    async def enable(
        self,
        *,
        owner_id: int,
        item_id: int,
        telegram_user_id: int,
        chat_id: int,
        timezone: str,
        morning_time: time,
        evening_time: time,
    ) -> VisionCompanionPreference | None:
        async with self.db.session() as session:
            await session.scalar(select(User).where(User.id == owner_id).with_for_update())
            item = await session.scalar(
                select(VisionItem).where(
                    VisionItem.id == item_id,
                    VisionItem.owner_id == owner_id,
                    VisionItem.status == "active",
                )
            )
            if item is None:
                return None
            preference = await session.scalar(
                select(VisionCompanionPreference).where(
                    VisionCompanionPreference.owner_id == owner_id
                )
            )
            if preference is None:
                preference = VisionCompanionPreference(
                    owner_id=owner_id,
                    vision_item_id=item_id,
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    timezone=timezone,
                    morning_time=morning_time,
                    evening_time=evening_time,
                    extra_per_day=0,
                    extra_times=[],
                    enabled=True,
                )
                session.add(preference)
            else:
                preference.vision_item_id = item_id
                preference.telegram_user_id = telegram_user_id
                preference.chat_id = chat_id
                preference.timezone = timezone
                preference.morning_time = morning_time
                preference.evening_time = evening_time
                preference.enabled = True
            await session.flush()
            return preference

    async def set_frequency(
        self, owner_id: int, item_id: int, extra_per_day: int
    ) -> VisionCompanionPreference | None:
        async with self.db.session() as session:
            await session.scalar(select(User).where(User.id == owner_id).with_for_update())
            preference = await session.scalar(
                select(VisionCompanionPreference).where(
                    VisionCompanionPreference.owner_id == owner_id,
                    VisionCompanionPreference.vision_item_id == item_id,
                    VisionCompanionPreference.enabled.is_(True),
                )
            )
            if preference is None:
                return None
            times = companion_extra_times(
                preference.morning_time, preference.evening_time, extra_per_day
            )
            preference.extra_per_day = extra_per_day
            preference.extra_times = [value.strftime("%H:%M") for value in times]
            await session.flush()
            return preference

    async def disable(self, owner_id: int, item_id: int | None = None) -> bool:
        async with self.db.session() as session:
            await session.scalar(select(User).where(User.id == owner_id).with_for_update())
            conditions = [VisionCompanionPreference.owner_id == owner_id]
            if item_id is not None:
                conditions.append(VisionCompanionPreference.vision_item_id == item_id)
            preference = await session.scalar(select(VisionCompanionPreference).where(*conditions))
            if preference is None or not preference.enabled:
                return False
            preference.enabled = False
            return True

    async def enabled_preferences(self) -> list[VisionCompanionPreference]:
        async with self.db.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(VisionCompanionPreference)
                        .join(User, User.id == VisionCompanionPreference.owner_id)
                        .where(
                            VisionCompanionPreference.enabled.is_(True),
                            User.access_tier.in_(FULL_ACCESS_TIERS),
                        )
                    )
                ).all()
            )

    async def snapshot(self, preference_id: int) -> CompanionSnapshot | None:
        async with self.db.session() as session:
            row = await session.execute(
                select(VisionCompanionPreference, User)
                .join(User, User.id == VisionCompanionPreference.owner_id)
                .where(
                    VisionCompanionPreference.id == preference_id,
                    VisionCompanionPreference.enabled.is_(True),
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                )
            )
            current = row.one_or_none()
            if current is None:
                return None
            preference, owner = current
            item = await session.scalar(
                select(VisionItem).where(
                    VisionItem.id == preference.vision_item_id,
                    VisionItem.owner_id == preference.owner_id,
                )
            )
            if item is None or item.status != "active":
                preference.enabled = False
                return None
            image = await session.scalar(
                select(VisionItemImage).where(
                    VisionItemImage.vision_item_id == item.id,
                    VisionItemImage.owner_id == preference.owner_id,
                )
            )
            return CompanionSnapshot(
                preference_id=preference.id,
                owner_id=preference.owner_id,
                item_id=item.id,
                chat_id=owner.telegram_id,
                timezone=preference.timezone,
                wish_text=item.wish_text,
                why_text=item.why_text,
                first_step=item.first_step,
                image_bytes=image.image_bytes if image is not None else None,
            )

    async def record(
        self,
        *,
        owner_id: int,
        item_id: int,
        moment: str,
        response: str,
        now: datetime | None = None,
    ) -> VisionCompanionCheckIn | None:
        if moment not in MOMENTS or response not in RESPONSES:
            return None
        async with self.db.session() as session:
            await session.scalar(select(User).where(User.id == owner_id).with_for_update())
            preference = await session.scalar(
                select(VisionCompanionPreference).where(
                    VisionCompanionPreference.owner_id == owner_id,
                    VisionCompanionPreference.vision_item_id == item_id,
                    VisionCompanionPreference.enabled.is_(True),
                )
            )
            if preference is None:
                return None
            try:
                zone = ZoneInfo(preference.timezone)
            except (TypeError, ZoneInfoNotFoundError):
                zone = ZoneInfo("UTC")
            current = now.astimezone(zone) if now is not None else datetime.now(zone)
            checkin = await session.scalar(
                select(VisionCompanionCheckIn).where(
                    VisionCompanionCheckIn.owner_id == owner_id,
                    VisionCompanionCheckIn.vision_item_id == item_id,
                    VisionCompanionCheckIn.local_date == current.date(),
                    VisionCompanionCheckIn.moment == moment,
                )
            )
            if checkin is None:
                checkin = VisionCompanionCheckIn(
                    owner_id=owner_id,
                    vision_item_id=item_id,
                    local_date=current.date(),
                    moment=moment,
                    response=response,
                )
                session.add(checkin)
            else:
                checkin.response = response
            await session.flush()
            return checkin
