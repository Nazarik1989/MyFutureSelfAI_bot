from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select

from .db import Database
from .models import OnboardingState, User

_ROUTE_SEPARATOR = re.compile(r"\s*(?:→|->)\s*")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class UserLocation:
    city: str
    fallback_city: str | None = None

    @property
    def label(self) -> str:
        return f"{self.city} → {self.fallback_city}" if self.fallback_city else self.city

    @property
    def key(self) -> tuple[str, str | None]:
        return normalize_city(self.city), (
            normalize_city(self.fallback_city) if self.fallback_city else None
        )


def normalize_city(value: str) -> str:
    return _SPACE.sub(" ", value.strip()).casefold().replace("ё", "е")


def parse_location(value: str) -> UserLocation:
    clean = _SPACE.sub(" ", value.strip())
    if not clean:
        raise ValueError("Укажи город, например: /location Саратов.")
    parts = _ROUTE_SEPARATOR.split(clean)
    if len(parts) > 2 or any(not part.strip() for part in parts):
        raise ValueError(
            "Локация должна быть городом или маршрутом из двух городов, "
            "например: /location Саратов → Энгельс."
        )
    cities = tuple(_clean_city(part) for part in parts)
    return UserLocation(cities[0], cities[1] if len(cities) == 2 else None)


def location_from_user(user: User) -> UserLocation | None:
    if not user.location_city:
        return None
    return UserLocation(user.location_city, user.location_fallback_city)


class LocationService:
    def __init__(self, db: Database):
        self.db = db

    async def get(self, user_id: int) -> UserLocation | None:
        async with self.db.sessions() as session:
            user = await session.get(User, user_id)
            return location_from_user(user) if user is not None else None

    async def set(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        value: str,
    ) -> UserLocation:
        location = parse_location(value)
        async with self.db.session() as session:
            user = await session.scalar(
                select(User).where(
                    User.id == user_id,
                    User.telegram_id == telegram_user_id,
                )
            )
            if user is None:
                raise ValueError("Пользователь не найден.")
            user.location_city = location.city
            user.location_fallback_city = location.fallback_city
            state = await session.scalar(
                select(OnboardingState).where(OnboardingState.user_id == user.id)
            )
            if state is not None:
                answers = dict(state.answers)
                answers["location"] = location.label
                state.answers = answers
        return location


def _clean_city(value: str) -> str:
    city = _SPACE.sub(" ", value.strip())
    if len(city) > 120:
        raise ValueError("Название города слишком длинное.")
    if not all(character.isalpha() or character in {" ", "-", "."} for character in city):
        raise ValueError("В названии города допустимы только буквы, пробел, дефис и точка.")
    if not any(character.isalpha() for character in city):
        raise ValueError("Название города должно содержать буквы.")
    return city
