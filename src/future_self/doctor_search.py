from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, update

from .db import Database
from .models import InboxItem, TaskReminder, User
from .reminders import reminder_for_inbox_item
from .schemas import TemporalResolution

OFFICIAL_BOOKING_URL = "https://zdrav.lenreg.ru/"
OFFICIAL_BOOKING_PHONE = "122"
SEARCH_TASK_TITLE = "Записаться к терапевту: Светогорск → Выборг"
SEARCH_TASK_KEY = "doctor_search:ru:svetogorsk:vyborg:therapist:v1"


@dataclass(frozen=True, slots=True)
class DoctorOption:
    city: str
    facility: str
    specialty: str
    address: str
    phone: str
    phone_label: str
    source_url: str


@dataclass(frozen=True, slots=True)
class DoctorDirectory:
    country: str
    specialty: str
    options: tuple[DoctorOption, ...]
    official_sources: tuple[str, ...]
    verified_on: str


@dataclass(frozen=True, slots=True)
class SearchTaskResult:
    status: str
    inbox_item: InboxItem
    reminder: TaskReminder | None


class DoctorSearchService:
    """Curated official route; no ranking, LLM, or health-data processing."""

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

    @staticmethod
    def directory() -> DoctorDirectory:
        svetogorsk_source = "https://mb.vbglenobl.ru/polikliniki/poliklinika-v-g-svetogorske"
        vyborg_source = "https://mb.vbglenobl.ru/polikliniki/poliklinika-v-g-vyborge/kontakty"
        lofoms_source = "https://lofoms.spb.ru/disp_time"
        return DoctorDirectory(
            country="Россия",
            specialty="терапевт",
            options=(
                DoctorOption(
                    city="Светогорск",
                    facility="Поликлиника Светогорской больницы",
                    specialty="терапевт",
                    address="г. Светогорск, ул. Пограничная, д. 13",
                    phone="+7 (81378) 36-268",
                    phone_label="контакт поликлиники в перечне ЛОФОМС",
                    source_url=svetogorsk_source,
                ),
                DoctorOption(
                    city="Выборг",
                    facility="Городская поликлиника Выборгской межрайонной больницы",
                    specialty="терапевт",
                    address="г. Выборг, ул. Ильинская, д. 8",
                    phone="+7 (81378) 2-83-46",
                    phone_label="контакт поликлиники на официальном сайте больницы",
                    source_url=vyborg_source,
                ),
            ),
            official_sources=(
                svetogorsk_source,
                vyborg_source,
                lofoms_source,
                OFFICIAL_BOOKING_URL,
            ),
            verified_on="17.07.2026",
        )

    @classmethod
    def format_directory(cls) -> str:
        directory = cls.directory()
        first, second = directory.options
        return (
            "Терапевт — официальный маршрут Светогорск → Выборг\n\n"
            "Сначала ближайший вариант:\n"
            f"1. {first.facility}\n"
            f"{first.address}\n"
            f"{first.phone} — {first.phone_label}\n"
            f"{first.source_url}\n\n"
            "Если в Светогорске нет подходящего времени:\n"
            f"2. {second.facility}\n"
            f"{second.address}\n"
            f"{second.phone} — {second.phone_label}\n"
            f"{second.source_url}\n\n"
            f"Официальная запись: {OFFICIAL_BOOKING_PHONE} или {OFFICIAL_BOOKING_URL}\n"
            f"Адреса дополнительно сверены по ЛОФОМС: {directory.official_sources[2]}\n"
            f"Проверено: {directory.verified_on}. Наличие талонов уточняй при записи.\n\n"
            "Создать задачу и reminder: /doctor_find_task через 2 часа"
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
            # A harmless owner-row update serializes concurrent task creation on
            # both SQLite and PostgreSQL without adding search-specific schema.
            await session.execute(
                update(User).where(User.id == user_id).values(updated_at=User.updated_at)
            )
            existing = await session.scalar(
                select(InboxItem)
                .where(
                    InboxItem.user_id == user_id,
                    InboxItem.source == "doctor_search",
                    InboxItem.raw_text == SEARCH_TASK_KEY,
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
                title=SEARCH_TASK_TITLE,
                description=f"Официальная запись: {OFFICIAL_BOOKING_PHONE}; {OFFICIAL_BOOKING_URL}",
                raw_text=SEARCH_TASK_KEY,
                next_step="Проверить Светогорск; если нет талона — Выборг",
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
