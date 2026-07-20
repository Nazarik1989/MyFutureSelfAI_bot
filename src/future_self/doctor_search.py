from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import select, update

from .db import Database
from .location import UserLocation, location_from_user
from .models import InboxItem, TaskReminder, User
from .reminders import reminder_for_inbox_item
from .schemas import TemporalResolution

BOOKING_PHONE = "122"
GOSUSLUGI_BOOKING_URL = "https://www.gosuslugi.ru/"
SARATOV_BOOKING_URL = "https://er.med.saratov.gov.ru/"


@dataclass(frozen=True, slots=True)
class DoctorOption:
    city: str
    facility: str
    address: str
    phone: str
    phone_label: str
    source_url: str


@dataclass(frozen=True, slots=True)
class DoctorDirectory:
    location: UserLocation
    options: tuple[DoctorOption, ...]
    booking_url: str
    booking_phone: str
    official_sources: tuple[str, ...]
    verified_on: str
    task_key: str
    task_title: str
    next_step: str


@dataclass(frozen=True, slots=True)
class SearchTaskResult:
    status: str
    inbox_item: InboxItem | None = None
    reminder: TaskReminder | None = None


class DoctorSearchService:
    """Owner-location doctor route; no ranking, LLM, or health-data processing."""

    def __init__(
        self,
        db: Database,
        *,
        task_date_event_hour: int = 9,
        task_reminder_lead_minutes: int = 30,
    ):
        self.db = db
        self.task_date_event_hour = task_date_event_hour
        self.task_reminder_lead_minutes = task_reminder_lead_minutes

    @classmethod
    def directory(cls, location: UserLocation) -> DoctorDirectory:
        if location.key == ("светогорск", "выборг"):
            return cls._svetogorsk_vyborg(location)
        if location.key == ("саратов", None):
            return cls._saratov(location)
        return cls._generic(location)

    @classmethod
    def format_directory(cls, location: UserLocation) -> str:
        directory = cls.directory(location)
        if location.key == ("светогорск", "выборг"):
            first, second = directory.options
            return (
                f"Терапевт — официальный маршрут {location.label}\n\n"
                "Сначала ближайший вариант:\n"
                f"1. {first.facility}\n"
                f"{first.address}\n"
                f"{first.phone} — {first.phone_label}\n"
                f"{first.source_url}\n\n"
                f"Если в {location.city} нет подходящего времени:\n"
                f"2. {second.facility}\n"
                f"{second.address}\n"
                f"{second.phone} — {second.phone_label}\n"
                f"{second.source_url}\n\n"
                f"Официальная запись: {directory.booking_phone} или "
                f"{directory.booking_url}\n"
                f"Проверено: {directory.verified_on}. Наличие талонов уточняй при записи.\n\n"
                "Изменить город: /location. Создать задачу: "
                "/doctor_find_task через 2 часа"
            )
        if location.key == ("саратов", None):
            return (
                f"Терапевт — {location.label}\n\n"
                "Запись зависит от поликлиники, к которой ты прикреплён(а):\n"
                f"1. Электронная регистратура Саратовской области: "
                f"{directory.booking_url}\n"
                f"2. Госуслуги: {GOSUSLUGI_BOOKING_URL}\n"
                f"3. Единый номер записи: {directory.booking_phone}\n\n"
                "Если портал не показывает время, уточни прикрепление и позвони "
                "в регистратуру своей поликлиники.\n"
                f"Проверено: {directory.verified_on}.\n\n"
                "Изменить город: /location. Создать задачу: "
                "/doctor_find_task через 2 часа"
            )
        return (
            f"Терапевт — {location.label}\n\n"
            "Используй поликлинику по месту прикрепления:\n"
            f"1. Госуслуги: {directory.booking_url}\n"
            f"2. Единый номер записи: {directory.booking_phone}\n\n"
            "Если запись недоступна, уточни прикрепление и контакты регистратуры "
            "своей поликлиники.\n\n"
            "Изменить город: /location. Создать задачу: "
            "/doctor_find_task через 2 часа"
        )

    async def create_booking_task(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        chat_id: int,
        temporal: TemporalResolution,
    ) -> SearchTaskResult:
        async with self.db.session() as session:
            # Serialize location reads and idempotent task creation on the owner.
            locked = await session.execute(
                update(User)
                .where(
                    User.id == user_id,
                    User.telegram_id == telegram_user_id,
                )
                .values(updated_at=User.updated_at)
                .returning(User.id)
            )
            if locked.scalar_one_or_none() is None:
                return SearchTaskResult("missing_location")
            owner = await session.get(User, user_id)
            location = location_from_user(owner)
            if location is None:
                return SearchTaskResult("missing_location")
            directory = self.directory(location)
            existing = await session.scalar(
                select(InboxItem)
                .where(
                    InboxItem.user_id == user_id,
                    InboxItem.source == "doctor_search",
                    InboxItem.raw_text == directory.task_key,
                    InboxItem.status == "confirmed",
                )
                .order_by(InboxItem.id)
            )
            if existing is not None:
                reminder = await session.scalar(
                    select(TaskReminder).where(TaskReminder.inbox_item_id == existing.id)
                )
                return SearchTaskResult("existing", existing, reminder)

            item = InboxItem(
                user_id=user_id,
                kind="task",
                title=directory.task_title,
                description=(
                    f"Локация: {location.label}. Официальная запись: "
                    f"{directory.booking_phone}; {directory.booking_url}"
                ),
                raw_text=directory.task_key,
                next_step=directory.next_step,
                resolved_date=temporal.resolved_local_date,
                temporal_resolution=temporal.model_dump(mode="json"),
                source="doctor_search",
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
                await session.flush()
            return SearchTaskResult("created", item, reminder)

    @staticmethod
    def _svetogorsk_vyborg(location: UserLocation) -> DoctorDirectory:
        svetogorsk_source = "https://mb.vbglenobl.ru/polikliniki/poliklinika-v-g-svetogorske"
        vyborg_source = "https://mb.vbglenobl.ru/polikliniki/poliklinika-v-g-vyborge/kontakty"
        booking_url = "https://zdrav.lenreg.ru/"
        return DoctorDirectory(
            location=location,
            options=(
                DoctorOption(
                    city=location.city,
                    facility="Поликлиника Светогорской больницы",
                    address="г. Светогорск, ул. Пограничная, д. 13",
                    phone="+7 (81378) 36-268",
                    phone_label="контакт поликлиники",
                    source_url=svetogorsk_source,
                ),
                DoctorOption(
                    city=location.fallback_city or "Выборг",
                    facility="Городская поликлиника Выборгской межрайонной больницы",
                    address="г. Выборг, ул. Ильинская, д. 8",
                    phone="+7 (81378) 2-83-46",
                    phone_label="контакт поликлиники",
                    source_url=vyborg_source,
                ),
            ),
            booking_url=booking_url,
            booking_phone=BOOKING_PHONE,
            official_sources=(svetogorsk_source, vyborg_source, booking_url),
            verified_on="20.07.2026",
            # Preserve idempotency for the task created by the previous release.
            task_key="doctor_search:ru:svetogorsk:vyborg:therapist:v1",
            task_title=f"Записаться к терапевту: {location.label}",
            next_step=f"Проверить {location.city}; если нет талона — {location.fallback_city}",
        )

    @staticmethod
    def _saratov(location: UserLocation) -> DoctorDirectory:
        return DoctorDirectory(
            location=location,
            options=(),
            booking_url=SARATOV_BOOKING_URL,
            booking_phone=BOOKING_PHONE,
            official_sources=(SARATOV_BOOKING_URL, GOSUSLUGI_BOOKING_URL),
            verified_on="20.07.2026",
            task_key=DoctorSearchService._task_key(location),
            task_title=f"Записаться к терапевту: {location.label}",
            next_step="Открыть электронную регистратуру и выбрать поликлинику прикрепления",
        )

    @staticmethod
    def _generic(location: UserLocation) -> DoctorDirectory:
        return DoctorDirectory(
            location=location,
            options=(),
            booking_url=GOSUSLUGI_BOOKING_URL,
            booking_phone=BOOKING_PHONE,
            official_sources=(GOSUSLUGI_BOOKING_URL,),
            verified_on="20.07.2026",
            task_key=DoctorSearchService._task_key(location),
            task_title=f"Записаться к терапевту: {location.label}",
            next_step="Открыть Госуслуги и выбрать поликлинику прикрепления",
        )

    @staticmethod
    def _task_key(location: UserLocation) -> str:
        canonical = "|".join(part or "" for part in location.key)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"doctor_search:ru:{digest}:therapist:v2"
