from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.orm import selectinload

from .db import Database
from .models import InboxItem, User, VisionDraft, VisionItem, VisionItemImage

CATEGORY_META: dict[str, tuple[str, str]] = {
    "health_energy": ("🌿", "Здоровье и энергия"),
    "relationships_family": ("❤️", "Отношения и семья"),
    "work_purpose": ("💼", "Работа и предназначение"),
    "money": ("💰", "Деньги"),
    "home": ("🏡", "Дом"),
    "travel": ("✈️", "Путешествия"),
    "growth_creativity": ("🎨", "Развитие и творчество"),
    "other": ("✨", "Другое"),
}
VISION_STATUSES = {"active", "achieved", "archived"}
OPTIONAL_STEPS = {"why", "target_date", "first_step"}
EDITABLE_FIELDS = {"wish", "why", "target_date", "first_step", "category"}
PAGE_SIZE = 5


@dataclass(frozen=True, slots=True)
class DraftAdvance:
    status: str
    draft: VisionDraft | None = None
    item: VisionItem | None = None


@dataclass(frozen=True, slots=True)
class TaskLinkResult:
    status: str
    item: VisionItem | None = None
    task: InboxItem | None = None


class VisionService:
    """Deterministic, owner-scoped vision board without LLM calls."""

    def __init__(self, db: Database):
        self.db = db

    async def draft(self, owner_id: int, chat_id: int) -> VisionDraft | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(VisionDraft).where(
                    VisionDraft.owner_id == owner_id,
                    VisionDraft.chat_id == chat_id,
                )
            )

    async def begin(self, owner_id: int, chat_id: int) -> VisionDraft:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            existing = await session.scalar(
                select(VisionDraft).where(VisionDraft.owner_id == owner_id)
            )
            if existing is not None:
                if existing.chat_id != chat_id:
                    raise ValueError("Незавершённая карточка открыта в другом чате.")
                return existing
            draft = VisionDraft(owner_id=owner_id, chat_id=chat_id, step="category")
            session.add(draft)
            await session.flush()
            return draft

    async def cancel(self, owner_id: int, chat_id: int) -> bool:
        async with self.db.session() as session:
            result = await session.execute(
                delete(VisionDraft).where(
                    VisionDraft.owner_id == owner_id,
                    VisionDraft.chat_id == chat_id,
                )
            )
            return bool(result.rowcount)

    async def choose_category(
        self,
        owner_id: int,
        chat_id: int,
        category: str,
        *,
        draft_id: int | None = None,
    ) -> DraftAdvance:
        if category not in CATEGORY_META:
            return DraftAdvance("invalid")
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            draft = await self._owned_draft(session, owner_id, chat_id, draft_id)
            if draft is None:
                return DraftAdvance("stale")
            if draft.step == "edit_value" and draft.edit_field == "category":
                item = await self._owned_item(session, owner_id, draft.editing_item_id)
                if item is None:
                    await session.delete(draft)
                    return DraftAdvance("stale")
                item.category = category
                await session.delete(draft)
                return DraftAdvance("edited", item=item)
            if draft.step != "category":
                return DraftAdvance("stale")
            draft.category = category
            draft.step = "wish"
            draft.version += 1
            return DraftAdvance("advanced", draft=draft)

    async def consume_text(self, owner_id: int, chat_id: int, value: str) -> DraftAdvance:
        clean = value.strip()
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            owner = await session.get(User, owner_id)
            owner_today = self._owner_today(owner)
            draft = await self._owned_draft(session, owner_id, chat_id)
            if draft is None:
                return DraftAdvance("none")
            if draft.step == "category":
                return DraftAdvance("need_category", draft=draft)
            if draft.step == "preview":
                return DraftAdvance("need_confirm", draft=draft)
            if draft.step == "delete_confirm":
                return DraftAdvance("need_delete_confirm", draft=draft)
            if draft.step == "edit_value":
                return await self._apply_edit(session, owner_id, draft, clean, today=owner_today)
            if not clean:
                return DraftAdvance("invalid", draft=draft)
            if draft.step == "wish":
                draft.wish_text = self._bounded(clean, 1000, "Желание")
                draft.step = "why"
            elif draft.step == "why":
                draft.why_text = self._bounded(clean, 1500, "Причина")
                draft.step = "target_date"
            elif draft.step == "target_date":
                draft.target_date = parse_target_date(clean, today=owner_today)
                draft.step = "first_step"
            elif draft.step == "first_step":
                draft.first_step = self._bounded(clean, 1000, "Первый шаг")
                draft.step = "preview"
            else:
                return DraftAdvance("stale")
            draft.version += 1
            return DraftAdvance("advanced", draft=draft)

    async def skip(self, owner_id: int, chat_id: int, draft_id: int, version: int) -> DraftAdvance:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            draft = await self._owned_draft(session, owner_id, chat_id, draft_id)
            if draft is None or draft.version != version:
                return DraftAdvance("stale")
            if draft.step == "edit_value" and draft.edit_field in OPTIONAL_STEPS:
                item = await self._owned_item(session, owner_id, draft.editing_item_id)
                if item is None:
                    await session.delete(draft)
                    return DraftAdvance("stale")
                setattr(item, field_attribute(draft.edit_field), None)
                await session.delete(draft)
                return DraftAdvance("edited", item=item)
            if draft.step not in OPTIONAL_STEPS:
                return DraftAdvance("stale")
            next_step = {
                "why": "target_date",
                "target_date": "first_step",
                "first_step": "preview",
            }
            setattr(draft, field_attribute(draft.step), None)
            draft.step = next_step[draft.step]
            draft.version += 1
            return DraftAdvance("advanced", draft=draft)

    async def confirm(
        self, owner_id: int, chat_id: int, draft_id: int, version: int
    ) -> DraftAdvance:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            draft = await self._owned_draft(session, owner_id, chat_id, draft_id)
            if (
                draft is None
                or draft.step != "preview"
                or draft.version != version
                or draft.category not in CATEGORY_META
                or not draft.wish_text
            ):
                return DraftAdvance("stale")
            item = VisionItem(
                owner_id=owner_id,
                category=draft.category,
                wish_text=draft.wish_text,
                why_text=draft.why_text,
                target_date=draft.target_date,
                first_step=draft.first_step,
                status="active",
            )
            session.add(item)
            await session.delete(draft)
            await session.flush()
            return DraftAdvance("created", item=item)

    async def get_item(self, owner_id: int, item_id: int) -> VisionItem | None:
        async with self.db.sessions() as session:
            return await self._owned_item(session, owner_id, item_id)

    async def page(self, owner_id: int, status: str, page: int) -> tuple[list[VisionItem], int]:
        if status not in VISION_STATUSES:
            return [], 0
        page = max(page, 0)
        async with self.db.sessions() as session:
            total = int(
                await session.scalar(
                    select(func.count(VisionItem.id)).where(
                        VisionItem.owner_id == owner_id,
                        VisionItem.status == status,
                    )
                )
                or 0
            )
            items = list(
                (
                    await session.scalars(
                        select(VisionItem)
                        .where(
                            VisionItem.owner_id == owner_id,
                            VisionItem.status == status,
                        )
                        .order_by(VisionItem.category, VisionItem.id)
                        .offset(page * PAGE_SIZE)
                        .limit(PAGE_SIZE)
                    )
                ).all()
            )
            return items, total

    async def category_counts(self, owner_id: int, status: str) -> dict[str, int]:
        if status not in VISION_STATUSES:
            return {}
        async with self.db.sessions() as session:
            rows = (
                await session.execute(
                    select(VisionItem.category, func.count(VisionItem.id))
                    .where(
                        VisionItem.owner_id == owner_id,
                        VisionItem.status == status,
                    )
                    .group_by(VisionItem.category)
                )
            ).all()
            return {category: int(count) for category, count in rows}

    async def active_for_render(
        self,
        owner_id: int,
        *,
        category: str | None,
        limit: int,
    ) -> tuple[list[VisionItem], int]:
        if category is not None and category not in CATEGORY_META:
            return [], 0
        category_order = case(
            {code: index for index, code in enumerate(CATEGORY_META)},
            value=VisionItem.category,
            else_=len(CATEGORY_META),
        )
        conditions = [
            VisionItem.owner_id == owner_id,
            VisionItem.status == "active",
        ]
        if category is not None:
            conditions.append(VisionItem.category == category)
        async with self.db.sessions() as session:
            total = int(
                await session.scalar(select(func.count(VisionItem.id)).where(*conditions)) or 0
            )
            items = list(
                (
                    await session.scalars(
                        select(VisionItem)
                        .options(selectinload(VisionItem.image))
                        .where(*conditions)
                        .order_by(
                            category_order,
                            VisionItem.target_date.is_(None),
                            VisionItem.target_date,
                            VisionItem.id,
                        )
                        .limit(max(limit, 1))
                    )
                ).all()
            )
            return items, total

    async def set_status(self, owner_id: int, item_id: int, status: str) -> VisionItem | None:
        if status not in VISION_STATUSES:
            return None
        async with self.db.session() as session:
            item = await self._owned_item(session, owner_id, item_id)
            if item is None:
                return None
            item.status = status
            return item

    async def start_edit(
        self, owner_id: int, chat_id: int, item_id: int, field: str
    ) -> DraftAdvance:
        if field not in EDITABLE_FIELDS:
            return DraftAdvance("invalid")
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            item = await self._owned_item(session, owner_id, item_id)
            if item is None:
                return DraftAdvance("stale")
            existing = await session.scalar(
                select(VisionDraft).where(VisionDraft.owner_id == owner_id)
            )
            if existing is not None:
                return DraftAdvance("busy", draft=existing, item=item)
            draft = VisionDraft(
                owner_id=owner_id,
                chat_id=chat_id,
                step="edit_value",
                editing_item_id=item.id,
                edit_field=field,
            )
            session.add(draft)
            await session.flush()
            return DraftAdvance("editing", draft=draft, item=item)

    async def start_delete(self, owner_id: int, chat_id: int, item_id: int) -> DraftAdvance:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            item = await self._owned_item(session, owner_id, item_id)
            if item is None:
                return DraftAdvance("stale")
            existing = await session.scalar(
                select(VisionDraft).where(VisionDraft.owner_id == owner_id)
            )
            if existing is not None:
                if (
                    existing.chat_id == chat_id
                    and existing.step == "delete_confirm"
                    and existing.editing_item_id == item.id
                ):
                    return DraftAdvance("confirming", draft=existing, item=item)
                return DraftAdvance("busy", draft=existing, item=item)
            draft = VisionDraft(
                owner_id=owner_id,
                chat_id=chat_id,
                step="delete_confirm",
                editing_item_id=item.id,
            )
            session.add(draft)
            await session.flush()
            return DraftAdvance("confirming", draft=draft, item=item)

    async def confirm_delete(
        self,
        owner_id: int,
        chat_id: int,
        item_id: int,
        draft_id: int,
        version: int,
    ) -> DraftAdvance:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            draft = await self._owned_draft(session, owner_id, chat_id, draft_id)
            if (
                draft is None
                or draft.step != "delete_confirm"
                or draft.editing_item_id != item_id
                or draft.version != version
            ):
                return DraftAdvance("stale")
            item = await self._owned_item(session, owner_id, item_id)
            await session.delete(draft)
            if item is None:
                return DraftAdvance("stale")
            await session.execute(
                delete(VisionItemImage).where(
                    VisionItemImage.owner_id == owner_id,
                    VisionItemImage.vision_item_id == item_id,
                )
            )
            await session.delete(item)
            return DraftAdvance("deleted", item=item)

    async def cancel_delete(
        self,
        owner_id: int,
        chat_id: int,
        item_id: int,
        draft_id: int,
        version: int,
    ) -> DraftAdvance:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            draft = await self._owned_draft(session, owner_id, chat_id, draft_id)
            if (
                draft is None
                or draft.step != "delete_confirm"
                or draft.editing_item_id != item_id
                or draft.version != version
            ):
                return DraftAdvance("stale")
            item = await self._owned_item(session, owner_id, item_id)
            await session.delete(draft)
            if item is None:
                return DraftAdvance("stale")
            return DraftAdvance("cancelled", item=item)

    async def delete_item(self, owner_id: int, item_id: int) -> bool:
        async with self.db.session() as session:
            await session.execute(
                delete(VisionItemImage).where(
                    VisionItemImage.owner_id == owner_id,
                    VisionItemImage.vision_item_id == item_id,
                )
            )
            result = await session.execute(
                delete(VisionItem).where(
                    VisionItem.id == item_id,
                    VisionItem.owner_id == owner_id,
                )
            )
            return bool(result.rowcount)

    async def create_task(self, owner_id: int, item_id: int) -> TaskLinkResult:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            item = await self._owned_item(session, owner_id, item_id)
            if item is None:
                return TaskLinkResult("stale")
            if not item.first_step:
                return TaskLinkResult("missing_step", item=item)
            if item.linked_task_id is not None:
                task = await session.get(InboxItem, item.linked_task_id)
                return TaskLinkResult("existing", item, task)
            key = f"vision:{item.id}:first-step:v1"
            existing = await session.scalar(
                select(InboxItem).where(
                    InboxItem.user_id == owner_id,
                    InboxItem.source == "vision",
                    InboxItem.raw_text == key,
                )
            )
            if existing is not None:
                item.linked_task_id = existing.id
                return TaskLinkResult("existing", item, existing)
            task = InboxItem(
                user_id=owner_id,
                kind="task",
                title=f"Шаг к желанию: {item.wish_text[:160]}",
                description=None,
                raw_text=key,
                next_step=item.first_step,
                resolved_date=None,
                temporal_resolution=None,
                source="vision",
                status="confirmed",
            )
            session.add(task)
            await session.flush()
            item.linked_task_id = task.id
            return TaskLinkResult("created", item, task)

    async def _apply_edit(
        self,
        session: object,
        owner_id: int,
        draft: VisionDraft,
        clean: str,
        *,
        today: date,
    ) -> DraftAdvance:
        item = await self._owned_item(session, owner_id, draft.editing_item_id)
        if item is None or draft.edit_field not in EDITABLE_FIELDS:
            await session.delete(draft)
            return DraftAdvance("stale")
        if draft.edit_field == "category":
            return DraftAdvance("need_category", draft=draft)
        if not clean:
            return DraftAdvance("invalid", draft=draft)
        if draft.edit_field == "target_date":
            item.target_date = parse_target_date(clean, today=today)
        else:
            limit = 1000 if draft.edit_field in {"wish", "first_step"} else 1500
            setattr(
                item,
                field_attribute(draft.edit_field),
                self._bounded(clean, limit, "Поле"),
            )
        await session.delete(draft)
        return DraftAdvance("edited", item=item)

    @staticmethod
    async def _lock_owner(session: object, owner_id: int) -> None:
        result = await session.execute(
            update(User)
            .where(User.id == owner_id)
            .values(updated_at=User.updated_at)
            .returning(User.id)
        )
        if result.scalar_one_or_none() is None:
            raise ValueError("Owner not found")

    @staticmethod
    async def _owned_draft(
        session: object,
        owner_id: int,
        chat_id: int,
        draft_id: int | None = None,
    ) -> VisionDraft | None:
        conditions = [
            VisionDraft.owner_id == owner_id,
            VisionDraft.chat_id == chat_id,
        ]
        if draft_id is not None:
            conditions.append(VisionDraft.id == draft_id)
        return await session.scalar(select(VisionDraft).where(*conditions))

    @staticmethod
    async def _owned_item(session: object, owner_id: int, item_id: int | None) -> VisionItem | None:
        if item_id is None:
            return None
        return await session.scalar(
            select(VisionItem).where(
                VisionItem.id == item_id,
                VisionItem.owner_id == owner_id,
            )
        )

    @staticmethod
    def _bounded(value: str, limit: int, label: str) -> str:
        if len(value) > limit:
            raise ValueError(f"{label} слишком длинное: максимум {limit} символов.")
        return value

    @staticmethod
    def _owner_today(owner: User | None) -> date:
        try:
            return datetime.now(ZoneInfo(owner.timezone if owner else "UTC")).date()
        except ZoneInfoNotFoundError:
            return date.today()


def parse_target_date(value: str, *, today: date | None = None) -> date:
    clean = value.strip()
    parsed: date | None = None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            parsed = datetime.strptime(clean, fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError("Не понял дату. Используй ДД.ММ.ГГГГ, например 31.12.2027.")
    if parsed < (today or date.today()):
        raise ValueError("Желаемая дата должна быть сегодня или в будущем.")
    return parsed


def field_attribute(field: str) -> str:
    return {
        "wish": "wish_text",
        "why": "why_text",
        "target_date": "target_date",
        "first_step": "first_step",
    }[field]
