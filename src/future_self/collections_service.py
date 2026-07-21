from __future__ import annotations

import re
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .db import Database
from .models import (
    InboxItem,
    LifeCollection,
    LifeCollectionActionToken,
    LifeCollectionAlias,
    LifeCollectionContext,
    LifeCollectionLink,
    LifeCollectionPreference,
    TaskState,
    User,
)
from .tasks import add_task_state

CollectionKind = Literal["topic", "project", "list"]
CollectionStatus = Literal["active", "archived"]

COLLECTION_KIND_LABELS: dict[CollectionKind, str] = {
    "topic": "тема",
    "project": "проект",
    "list": "список",
}


@dataclass(frozen=True, slots=True)
class StarterTemplate:
    key: str
    name: str
    kind: CollectionKind
    aliases: tuple[str, ...] = ()


STARTER_TEMPLATES: tuple[StarterTemplate, ...] = (
    StarterTemplate("work_projects", "Работа и проекты", "topic", ("работа", "проекты")),
    StarterTemplate("home", "Дом и быт", "topic", ("дом", "быт")),
    StarterTemplate("shopping", "Покупки", "list", ("покупка", "покупках", "список покупок")),
    StarterTemplate("finance", "Финансы", "topic", ("деньги", "бюджет")),
    StarterTemplate(
        "health",
        "Здоровье и самочувствие",
        "topic",
        ("здоровье", "самочувствие", "заметки по здоровью"),
    ),
    StarterTemplate("family", "Семья и отношения", "topic", ("семья", "отношения")),
    StarterTemplate("learning", "Обучение и развитие", "topic", ("обучение", "развитие", "учеба")),
    StarterTemplate(
        "creativity",
        "Творчество и идеи",
        "topic",
        ("творчество", "творчества", "идеи творчества"),
    ),
    StarterTemplate("travel", "Отдых и путешествия", "topic", ("отдых", "путешествия", "поездки")),
    StarterTemplate("personal", "Личное", "topic", ("личные", "личное")),
)
STARTER_BY_KEY = {item.key: item for item in STARTER_TEMPLATES}


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    collection: LifeCollection
    item_count: int


@dataclass(frozen=True, slots=True)
class CollectionPage:
    records: tuple[CollectionSummary, ...]
    page: int
    pages: int
    total: int
    kind: CollectionKind | None
    status: CollectionStatus


@dataclass(frozen=True, slots=True)
class CollectionItemRecord:
    item: InboxItem
    task_state: TaskState | None


@dataclass(frozen=True, slots=True)
class CollectionItemPage:
    collection: LifeCollection
    records: tuple[CollectionItemRecord, ...]
    page: int
    pages: int
    total: int


@dataclass(frozen=True, slots=True)
class CollectionResolution:
    match: LifeCollection | None
    candidates: tuple[LifeCollection, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class CollectionContextSnapshot:
    collection: LifeCollection
    last_inbox_item_id: int | None


@dataclass(frozen=True, slots=True)
class CollectionClaim:
    action: str
    collection_id: int | None
    collection_version: int | None
    inbox_item_id: int | None
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CollectionMutation:
    status: str
    collection: LifeCollection | None = None
    item_ids: tuple[int, ...] = ()


class CollectionNameError(ValueError):
    pass


class CollectionConflictError(CollectionNameError):
    pass


def normalize_collection_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    value = "".join(character if character.isalnum() else " " for character in value)
    return re.sub(r"\s+", " ", value).strip()


def clean_collection_name(value: str) -> tuple[str, str]:
    display = unicodedata.normalize("NFKC", value)
    display = re.sub(r"\s+", " ", display).strip(" \t\r\n.,:;!?—–-")
    normalized = normalize_collection_name(display)
    if not display or not normalized:
        raise CollectionNameError("Название раздела не может быть пустым.")
    if len(display) > 100 or len(normalized) > 100:
        raise CollectionNameError("Название раздела должно быть не длиннее 100 символов.")
    if any(unicodedata.category(character).startswith("C") for character in display):
        raise CollectionNameError("Название раздела содержит недопустимые символы.")
    return display, normalized


def split_list_items(value: str) -> tuple[tuple[str, ...], bool]:
    """Split clear comma/semicolon lists; return ambiguity for unsafe bulk input."""
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:")
    if not cleaned:
        return (), False
    if any(mark in cleaned for mark in ('"', "«", "»", "(", ")")):
        return (cleaned,), True
    if "," not in cleaned and ";" not in cleaned:
        return (cleaned,), False
    parts = [part.strip() for part in re.split(r"\s*[,;]\s*", cleaned) if part.strip()]
    if parts:
        tail = re.split(r"\s+и\s+", parts[-1], maxsplit=1, flags=re.IGNORECASE)
        if len(tail) == 2 and all(tail):
            parts[-1:] = [tail[0].strip(), tail[1].strip()]
    if len(parts) > 20 or any(len(part) > 200 for part in parts):
        return tuple(parts), True
    return tuple(parts), False


def infer_inbox_kind(text: str, collection_kind: CollectionKind) -> str:
    normalized = normalize_collection_name(text)
    if normalized.startswith(("идея ", "идею ")):
        return "idea"
    if collection_kind == "list":
        return "task"
    task_starts = (
        "сделать ",
        "исправить ",
        "купить ",
        "позвонить ",
        "проверить ",
        "подготовить ",
        "записаться ",
        "добавить ",
    )
    return "task" if normalized.startswith(task_starts) else "note"


class LifeCollectionService:
    PAGE_SIZE = 6
    ITEM_PAGE_SIZE = 6
    ACTION_TTL = timedelta(minutes=15)
    INPUT_TTL = timedelta(minutes=20)
    CONTEXT_TTL = timedelta(minutes=20)

    def __init__(
        self,
        db: Database,
        *,
        action_ttl: timedelta | None = None,
        input_ttl: timedelta | None = None,
        context_ttl: timedelta | None = None,
        task_date_event_hour: int = 9,
    ):
        self.db = db
        self.action_ttl = action_ttl or self.ACTION_TTL
        self.input_ttl = input_ttl or self.INPUT_TTL
        self.context_ttl = context_ttl or self.CONTEXT_TTL
        self.task_date_event_hour = task_date_event_hour

    async def cleanup(self, *, now: datetime | None = None) -> tuple[int, int]:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            tokens = await session.execute(
                delete(LifeCollectionActionToken).where(
                    LifeCollectionActionToken.expires_at <= current
                )
            )
            contexts = await session.execute(
                delete(LifeCollectionContext).where(LifeCollectionContext.expires_at <= current)
            )
            return int(tokens.rowcount or 0), int(contexts.rowcount or 0)

    async def is_onboarded(self, owner_id: int) -> bool:
        async with self.db.sessions() as session:
            value = await session.scalar(
                select(LifeCollectionPreference.onboarding_completed).where(
                    LifeCollectionPreference.owner_id == owner_id
                )
            )
            return bool(value)

    async def complete_empty_onboarding(self, owner_id: int) -> CollectionMutation:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            preference = await session.get(LifeCollectionPreference, owner_id)
            if preference is not None and preference.onboarding_completed:
                return CollectionMutation("already_completed")
            if preference is None:
                preference = LifeCollectionPreference(
                    owner_id=owner_id, onboarding_completed=True, version=1
                )
                session.add(preference)
            else:
                preference.onboarding_completed = True
                preference.version += 1
            return CollectionMutation("completed")

    async def create_starters(self, owner_id: int, keys: tuple[str, ...]) -> CollectionMutation:
        unique_keys = tuple(dict.fromkeys(keys))
        if any(key not in STARTER_BY_KEY for key in unique_keys):
            return CollectionMutation("invalid")
        try:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                preference = await session.get(LifeCollectionPreference, owner_id)
                if preference is not None and preference.onboarding_completed:
                    return CollectionMutation("already_completed")
                created: list[int] = []
                for key in unique_keys:
                    template = STARTER_BY_KEY[key]
                    collection = await self._create_collection(
                        session,
                        owner_id,
                        template.kind,
                        template.name,
                        starter_key=template.key,
                        aliases=template.aliases,
                    )
                    created.append(collection.id)
                if preference is None:
                    session.add(
                        LifeCollectionPreference(
                            owner_id=owner_id, onboarding_completed=True, version=1
                        )
                    )
                else:
                    preference.onboarding_completed = True
                    preference.version += 1
                return CollectionMutation("created", item_ids=tuple(created))
        except (CollectionNameError, IntegrityError):
            return CollectionMutation("conflict")

    async def create_collection(
        self,
        owner_id: int,
        kind: CollectionKind,
        name: str,
        *,
        mark_onboarded: bool = True,
    ) -> CollectionMutation:
        if kind not in COLLECTION_KIND_LABELS:
            return CollectionMutation("invalid")
        try:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                collection = await self._create_collection(session, owner_id, kind, name)
                if mark_onboarded:
                    await self._mark_onboarded(session, owner_id)
                await session.flush()
                return CollectionMutation("created", collection)
        except CollectionConflictError:
            return CollectionMutation("conflict")
        except CollectionNameError:
            raise
        except IntegrityError:
            return CollectionMutation("conflict")

    async def add_alias(
        self, owner_id: int, collection_id: int, version: int, alias: str
    ) -> CollectionMutation:
        display, normalized = clean_collection_name(alias)
        try:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                collection = await self._versioned_collection(
                    session, owner_id, collection_id, version
                )
                if collection is None:
                    return CollectionMutation("stale")
                if await self._name_taken(session, owner_id, normalized):
                    return CollectionMutation("conflict")
                session.add(
                    LifeCollectionAlias(
                        collection_id=collection.id,
                        owner_id=owner_id,
                        alias=display,
                        normalized_alias=normalized,
                    )
                )
                collection.version += 1
                await session.flush()
                return CollectionMutation("aliased", collection)
        except IntegrityError:
            return CollectionMutation("conflict")

    async def list_page(
        self,
        owner_id: int,
        page: int,
        *,
        kind: CollectionKind | None = None,
        status: CollectionStatus = "active",
    ) -> CollectionPage:
        if kind is not None and kind not in COLLECTION_KIND_LABELS:
            raise ValueError("Unknown collection kind")
        if status not in {"active", "archived"}:
            raise ValueError("Unknown collection status")
        count_subquery = (
            select(func.count(LifeCollectionLink.id))
            .where(
                LifeCollectionLink.owner_id == owner_id,
                LifeCollectionLink.collection_id == LifeCollection.id,
            )
            .correlate(LifeCollection)
            .scalar_subquery()
        )
        async with self.db.sessions() as session:
            statement = select(LifeCollection, count_subquery).where(
                LifeCollection.owner_id == owner_id,
                LifeCollection.status == status,
            )
            if kind is not None:
                statement = statement.where(LifeCollection.kind == kind)
            rows = (
                await session.execute(
                    statement.order_by(
                        LifeCollection.normalized_name.asc(), LifeCollection.id.asc()
                    )
                )
            ).all()
        total = len(rows)
        pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        safe_page = min(max(page, 0), pages - 1)
        start = safe_page * self.PAGE_SIZE
        return CollectionPage(
            records=tuple(
                CollectionSummary(collection, int(count or 0))
                for collection, count in rows[start : start + self.PAGE_SIZE]
            ),
            page=safe_page,
            pages=pages,
            total=total,
            kind=kind,
            status=status,
        )

    async def summary(self, owner_id: int, collection_id: int) -> CollectionSummary | None:
        async with self.db.sessions() as session:
            collection = await session.scalar(
                select(LifeCollection).where(
                    LifeCollection.id == collection_id,
                    LifeCollection.owner_id == owner_id,
                )
            )
            if collection is None:
                return None
            count = await session.scalar(
                select(func.count(LifeCollectionLink.id)).where(
                    LifeCollectionLink.owner_id == owner_id,
                    LifeCollectionLink.collection_id == collection_id,
                )
            )
            return CollectionSummary(collection, int(count or 0))

    async def item_page(
        self, owner_id: int, collection_id: int, page: int
    ) -> CollectionItemPage | None:
        async with self.db.sessions() as session:
            collection = await session.scalar(
                select(LifeCollection).where(
                    LifeCollection.id == collection_id,
                    LifeCollection.owner_id == owner_id,
                )
            )
            if collection is None:
                return None
            rows = (
                await session.execute(
                    select(InboxItem, TaskState)
                    .join(
                        LifeCollectionLink,
                        LifeCollectionLink.inbox_item_id == InboxItem.id,
                    )
                    .outerjoin(TaskState, TaskState.inbox_item_id == InboxItem.id)
                    .where(
                        LifeCollectionLink.owner_id == owner_id,
                        LifeCollectionLink.collection_id == collection_id,
                        InboxItem.user_id == owner_id,
                    )
                    .order_by(LifeCollectionLink.created_at.desc(), InboxItem.id.desc())
                )
            ).all()
        total = len(rows)
        pages = max(1, (total + self.ITEM_PAGE_SIZE - 1) // self.ITEM_PAGE_SIZE)
        safe_page = min(max(page, 0), pages - 1)
        start = safe_page * self.ITEM_PAGE_SIZE
        return CollectionItemPage(
            collection=collection,
            records=tuple(
                CollectionItemRecord(item, task_state)
                for item, task_state in rows[start : start + self.ITEM_PAGE_SIZE]
            ),
            page=safe_page,
            pages=pages,
            total=total,
        )

    async def item_record(
        self, owner_id: int, collection_id: int, inbox_item_id: int
    ) -> CollectionItemRecord | None:
        async with self.db.sessions() as session:
            row = (
                await session.execute(
                    select(InboxItem, TaskState)
                    .join(
                        LifeCollectionLink,
                        LifeCollectionLink.inbox_item_id == InboxItem.id,
                    )
                    .outerjoin(TaskState, TaskState.inbox_item_id == InboxItem.id)
                    .where(
                        LifeCollectionLink.owner_id == owner_id,
                        LifeCollectionLink.collection_id == collection_id,
                        LifeCollectionLink.inbox_item_id == inbox_item_id,
                        InboxItem.user_id == owner_id,
                    )
                )
            ).one_or_none()
            return CollectionItemRecord(*row) if row is not None else None

    async def active_collections(
        self, owner_id: int, *, exclude_id: int | None = None
    ) -> tuple[LifeCollection, ...]:
        async with self.db.sessions() as session:
            statement = select(LifeCollection).where(
                LifeCollection.owner_id == owner_id,
                LifeCollection.status == "active",
            )
            if exclude_id is not None:
                statement = statement.where(LifeCollection.id != exclude_id)
            return tuple(
                (
                    await session.scalars(
                        statement.order_by(
                            LifeCollection.normalized_name.asc(), LifeCollection.id.asc()
                        )
                    )
                ).all()
            )

    async def resolve(self, owner_id: int, value: str) -> CollectionResolution:
        normalized = normalize_collection_name(value)
        if not normalized:
            return CollectionResolution(None)
        async with self.db.sessions() as session:
            exact = await session.scalar(
                select(LifeCollection).where(
                    LifeCollection.owner_id == owner_id,
                    LifeCollection.status == "active",
                    LifeCollection.normalized_name == normalized,
                )
            )
            if exact is not None:
                return CollectionResolution(exact, confidence=1.0)
            alias = await session.scalar(
                select(LifeCollection)
                .join(
                    LifeCollectionAlias,
                    LifeCollectionAlias.collection_id == LifeCollection.id,
                )
                .where(
                    LifeCollection.owner_id == owner_id,
                    LifeCollectionAlias.owner_id == owner_id,
                    LifeCollection.status == "active",
                    LifeCollectionAlias.normalized_alias == normalized,
                )
            )
            if alias is not None:
                return CollectionResolution(alias, confidence=1.0)
        return await self.suggest(owner_id, value)

    async def resolve_leading(self, owner_id: int, value: str) -> tuple[CollectionResolution, str]:
        normalized_tokens = normalize_collection_name(value).split()
        if not normalized_tokens:
            return CollectionResolution(None), ""
        names = await self._search_names(owner_id)
        matches: list[tuple[int, LifeCollection]] = []
        for normalized, collection in names:
            candidate = normalized.split()
            if normalized_tokens[: len(candidate)] == candidate:
                matches.append((len(candidate), collection))
        if not matches:
            return await self.suggest(owner_id, value), ""
        matches.sort(key=lambda pair: (-pair[0], pair[1].id))
        best_size = matches[0][0]
        best = {pair[1].id: pair[1] for pair in matches if pair[0] == best_size}
        if len(best) != 1:
            return CollectionResolution(None, tuple(best.values()), 0.5), ""
        original_tokens = re.sub(r"\s+", " ", value).strip().split(" ")
        remainder = " ".join(original_tokens[best_size:]).strip(" \t\r\n:,-")
        return CollectionResolution(next(iter(best.values())), confidence=1.0), remainder

    async def suggest(self, owner_id: int, value: str) -> CollectionResolution:
        query_tokens = set(normalize_collection_name(value).split())
        if not query_tokens:
            return CollectionResolution(None)
        scores: dict[int, tuple[float, LifeCollection]] = {}
        for normalized, collection in await self._search_names(owner_id):
            name_tokens = set(normalized.split())
            overlap = len(query_tokens & name_tokens)
            if not overlap:
                continue
            score = overlap / max(len(query_tokens), len(name_tokens))
            previous = scores.get(collection.id)
            if previous is None or score > previous[0]:
                scores[collection.id] = (score, collection)
        ordered = sorted(scores.values(), key=lambda pair: (-pair[0], pair[1].id))
        if not ordered:
            return CollectionResolution(None)
        best_score = ordered[0][0]
        close = tuple(collection for score, collection in ordered if best_score - score < 0.15)[:4]
        if best_score >= 0.74 and len(close) == 1:
            return CollectionResolution(close[0], close, best_score)
        return CollectionResolution(None, close, best_score)

    async def create_items(
        self,
        owner_id: int,
        chat_id: int,
        collection_id: int,
        expected_version: int,
        contents: tuple[str, ...],
        *,
        source: str,
        forced_kind: str | None = None,
    ) -> CollectionMutation:
        cleaned = tuple(re.sub(r"\s+", " ", item).strip() for item in contents if item.strip())
        if not cleaned or len(cleaned) > 20 or any(len(item) > 2000 for item in cleaned):
            return CollectionMutation("invalid")
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            collection = await self._versioned_collection(
                session, owner_id, collection_id, expected_version, active_only=True
            )
            if collection is None:
                return CollectionMutation("stale")
            owner = await session.get(User, owner_id)
            if owner is None:
                return CollectionMutation("stale")
            item_ids: list[int] = []
            for content in cleaned:
                kind = forced_kind or infer_inbox_kind(content, collection.kind)
                if kind not in {"task", "note", "idea"}:
                    return CollectionMutation("invalid")
                title = content[:200]
                item = InboxItem(
                    user_id=owner_id,
                    kind=kind,
                    title=title,
                    description=None,
                    raw_text=content,
                    next_step=None,
                    resolved_date=None,
                    temporal_resolution=None,
                    source=f"collection_{source}",
                    status="confirmed",
                )
                session.add(item)
                await session.flush()
                if kind == "task":
                    await add_task_state(
                        session,
                        item,
                        owner_timezone=owner.timezone,
                        date_event_hour=self.task_date_event_hour,
                    )
                session.add(
                    LifeCollectionLink(
                        collection_id=collection.id,
                        owner_id=owner_id,
                        inbox_item_id=item.id,
                    )
                )
                item_ids.append(item.id)
            collection.version += 1
            await session.flush()
            new_version = collection.version
        await self.set_context(
            owner_id,
            chat_id,
            collection_id,
            last_inbox_item_id=item_ids[-1],
        )
        refreshed = await self._collection(owner_id, collection_id)
        if refreshed is not None and refreshed.version != new_version:
            return CollectionMutation("stale")
        return CollectionMutation("created_items", refreshed, tuple(item_ids))

    async def link_item(
        self,
        owner_id: int,
        collection_id: int,
        expected_version: int,
        inbox_item_id: int,
    ) -> CollectionMutation:
        try:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                collection = await self._versioned_collection(
                    session, owner_id, collection_id, expected_version, active_only=True
                )
                item = await session.scalar(
                    select(InboxItem).where(
                        InboxItem.id == inbox_item_id, InboxItem.user_id == owner_id
                    )
                )
                if collection is None or item is None:
                    return CollectionMutation("stale")
                existing = await session.scalar(
                    select(LifeCollectionLink.id).where(
                        LifeCollectionLink.owner_id == owner_id,
                        LifeCollectionLink.collection_id == collection_id,
                        LifeCollectionLink.inbox_item_id == inbox_item_id,
                    )
                )
                if existing is not None:
                    return CollectionMutation("already_linked", collection)
                session.add(
                    LifeCollectionLink(
                        owner_id=owner_id,
                        collection_id=collection_id,
                        inbox_item_id=inbox_item_id,
                    )
                )
                collection.version += 1
                await session.flush()
                return CollectionMutation("linked", collection, (inbox_item_id,))
        except IntegrityError:
            return CollectionMutation("stale")

    async def unlink_item(
        self,
        owner_id: int,
        collection_id: int,
        expected_version: int,
        inbox_item_id: int,
    ) -> CollectionMutation:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            collection = await self._versioned_collection(
                session, owner_id, collection_id, expected_version
            )
            if collection is None:
                return CollectionMutation("stale")
            result = await session.execute(
                delete(LifeCollectionLink).where(
                    LifeCollectionLink.owner_id == owner_id,
                    LifeCollectionLink.collection_id == collection_id,
                    LifeCollectionLink.inbox_item_id == inbox_item_id,
                )
            )
            if not result.rowcount:
                return CollectionMutation("stale")
            collection.version += 1
            await session.flush()
            return CollectionMutation("unlinked", collection, (inbox_item_id,))

    async def move_item(
        self,
        owner_id: int,
        source_id: int,
        source_version: int,
        target_id: int,
        target_version: int,
        inbox_item_id: int,
    ) -> CollectionMutation:
        if source_id == target_id:
            return CollectionMutation("same")
        try:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                source = await self._versioned_collection(
                    session, owner_id, source_id, source_version
                )
                target = await self._versioned_collection(
                    session, owner_id, target_id, target_version, active_only=True
                )
                link = await session.scalar(
                    select(LifeCollectionLink).where(
                        LifeCollectionLink.owner_id == owner_id,
                        LifeCollectionLink.collection_id == source_id,
                        LifeCollectionLink.inbox_item_id == inbox_item_id,
                    )
                )
                if source is None or target is None or link is None:
                    return CollectionMutation("stale")
                duplicate = await session.scalar(
                    select(LifeCollectionLink.id).where(
                        LifeCollectionLink.owner_id == owner_id,
                        LifeCollectionLink.collection_id == target_id,
                        LifeCollectionLink.inbox_item_id == inbox_item_id,
                    )
                )
                if duplicate is None:
                    link.collection_id = target_id
                else:
                    await session.delete(link)
                source.version += 1
                target.version += 1
                await session.flush()
                return CollectionMutation("moved", target, (inbox_item_id,))
        except IntegrityError:
            return CollectionMutation("stale")

    async def rename(
        self, owner_id: int, collection_id: int, expected_version: int, name: str
    ) -> CollectionMutation:
        display, normalized = clean_collection_name(name)
        try:
            async with self.db.session() as session:
                await self._lock_owner(session, owner_id)
                collection = await self._versioned_collection(
                    session, owner_id, collection_id, expected_version
                )
                if collection is None:
                    return CollectionMutation("stale")
                if await self._name_taken(session, owner_id, normalized, collection_id):
                    return CollectionMutation("conflict")
                collection.name = display
                collection.normalized_name = normalized
                collection.version += 1
                await session.flush()
                return CollectionMutation("renamed", collection)
        except IntegrityError:
            return CollectionMutation("conflict")

    async def set_archived(
        self,
        owner_id: int,
        collection_id: int,
        expected_version: int,
        *,
        archived: bool,
    ) -> CollectionMutation:
        expected_status = "active" if archived else "archived"
        new_status = "archived" if archived else "active"
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            collection = await self._versioned_collection(
                session, owner_id, collection_id, expected_version
            )
            if collection is None or collection.status != expected_status:
                return CollectionMutation("stale")
            collection.status = new_status
            collection.version += 1
            await session.execute(
                delete(LifeCollectionContext).where(
                    LifeCollectionContext.owner_id == owner_id,
                    LifeCollectionContext.collection_id == collection_id,
                )
            )
            await session.flush()
            return CollectionMutation("archived" if archived else "restored", collection)

    async def delete_collection(
        self,
        owner_id: int,
        collection_id: int,
        expected_version: int,
        *,
        unlink_nonempty: bool,
    ) -> CollectionMutation:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            collection = await self._versioned_collection(
                session, owner_id, collection_id, expected_version
            )
            if collection is None:
                return CollectionMutation("stale")
            count = await session.scalar(
                select(func.count(LifeCollectionLink.id)).where(
                    LifeCollectionLink.owner_id == owner_id,
                    LifeCollectionLink.collection_id == collection_id,
                )
            )
            if count and not unlink_nonempty:
                return CollectionMutation("nonempty", collection)
            await session.delete(collection)
            await session.flush()
            return CollectionMutation("deleted")

    async def delete_item(
        self, owner_id: int, collection_id: int, expected_version: int, inbox_item_id: int
    ) -> CollectionMutation:
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            collection = await self._versioned_collection(
                session, owner_id, collection_id, expected_version
            )
            link = await session.scalar(
                select(LifeCollectionLink).where(
                    LifeCollectionLink.owner_id == owner_id,
                    LifeCollectionLink.collection_id == collection_id,
                    LifeCollectionLink.inbox_item_id == inbox_item_id,
                )
            )
            item = await session.scalar(
                select(InboxItem).where(
                    InboxItem.id == inbox_item_id, InboxItem.user_id == owner_id
                )
            )
            if collection is None or link is None or item is None:
                return CollectionMutation("stale")
            await session.delete(item)
            collection.version += 1
            await session.flush()
            return CollectionMutation("item_deleted", collection)

    async def set_context(
        self,
        owner_id: int,
        chat_id: int,
        collection_id: int,
        *,
        last_inbox_item_id: int | None = None,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_owner(session, owner_id)
            collection = await session.scalar(
                select(LifeCollection.id).where(
                    LifeCollection.id == collection_id,
                    LifeCollection.owner_id == owner_id,
                    LifeCollection.status == "active",
                )
            )
            if collection is None:
                return False
            if last_inbox_item_id is not None:
                owned_item = await session.scalar(
                    select(InboxItem.id).where(
                        InboxItem.id == last_inbox_item_id, InboxItem.user_id == owner_id
                    )
                )
                if owned_item is None:
                    return False
            context = await session.scalar(
                select(LifeCollectionContext).where(
                    LifeCollectionContext.owner_id == owner_id,
                    LifeCollectionContext.chat_id == chat_id,
                )
            )
            if context is None:
                session.add(
                    LifeCollectionContext(
                        owner_id=owner_id,
                        chat_id=chat_id,
                        collection_id=collection_id,
                        last_inbox_item_id=last_inbox_item_id,
                        version=1,
                        expires_at=current + self.context_ttl,
                    )
                )
            else:
                context.collection_id = collection_id
                context.last_inbox_item_id = last_inbox_item_id
                context.version += 1
                context.expires_at = current + self.context_ttl
            return True

    async def active_context(
        self, owner_id: int, chat_id: int, *, now: datetime | None = None
    ) -> CollectionContextSnapshot | None:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            row = (
                await session.execute(
                    select(LifeCollectionContext, LifeCollection)
                    .join(
                        LifeCollection,
                        LifeCollection.id == LifeCollectionContext.collection_id,
                    )
                    .where(
                        LifeCollectionContext.owner_id == owner_id,
                        LifeCollectionContext.chat_id == chat_id,
                    )
                )
            ).one_or_none()
            if row is None:
                return None
            context, collection = row
            if self._utc(context.expires_at) <= current:
                await session.execute(
                    delete(LifeCollectionContext)
                    .where(
                        LifeCollectionContext.id == context.id,
                        LifeCollectionContext.expires_at <= current,
                    )
                    .execution_options(synchronize_session=False)
                )
                return None
            if collection.status != "active":
                return None
            return CollectionContextSnapshot(collection, context.last_inbox_item_id)

    async def clear_context(self, owner_id: int, chat_id: int) -> bool:
        async with self.db.session() as session:
            result = await session.execute(
                delete(LifeCollectionContext).where(
                    LifeCollectionContext.owner_id == owner_id,
                    LifeCollectionContext.chat_id == chat_id,
                )
            )
            return bool(result.rowcount)

    async def issue_action(
        self,
        owner_id: int,
        chat_id: int,
        action: str,
        *,
        collection: LifeCollection | None = None,
        inbox_item_id: int | None = None,
        payload: dict[str, Any] | None = None,
        status: str = "pending",
        ttl: timedelta | None = None,
    ) -> str:
        async with self.db.session() as session:
            if await session.get(User, owner_id) is None:
                raise ValueError("Unknown collection owner")
            if collection is not None:
                current = await self._versioned_collection(
                    session, owner_id, collection.id, collection.version
                )
                if current is None:
                    raise ValueError("Stale collection")
            if inbox_item_id is not None:
                owned_item = await session.scalar(
                    select(InboxItem.id).where(
                        InboxItem.id == inbox_item_id, InboxItem.user_id == owner_id
                    )
                )
                if owned_item is None:
                    raise ValueError("Unknown collection item")
            if status == "awaiting_input":
                await session.execute(
                    update(LifeCollectionActionToken)
                    .where(
                        LifeCollectionActionToken.owner_id == owner_id,
                        LifeCollectionActionToken.chat_id == chat_id,
                        LifeCollectionActionToken.status == "awaiting_input",
                    )
                    .values(status="consumed", consumed_at=datetime.now(UTC))
                )
            return await self._new_token(
                session,
                owner_id,
                chat_id,
                action,
                collection=collection,
                inbox_item_id=inbox_item_id,
                payload=payload,
                status=status,
                ttl=ttl,
            )

    async def capability_action(self, token: str, owner_id: int, chat_id: int) -> str | None:
        async with self.db.sessions() as session:
            capability = await self._token(session, token, owner_id, chat_id)
            if (
                capability is None
                or capability.status == "consumed"
                or self._utc(capability.expires_at) <= datetime.now(UTC)
            ):
                return None
            return capability.action

    async def claim_action(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
        allowed: set[str],
        *,
        pending_status: str = "pending",
    ) -> CollectionClaim | None:
        async with self.db.session() as session:
            capability = await self._token(session, token, owner_id, chat_id)
            current = datetime.now(UTC)
            if (
                capability is None
                or capability.action not in allowed
                or capability.status != pending_status
                or self._utc(capability.expires_at) <= current
            ):
                return None
            if capability.collection_id is not None:
                collection = await self._versioned_collection(
                    session,
                    owner_id,
                    capability.collection_id,
                    capability.collection_version or 0,
                )
                if collection is None:
                    return None
            if capability.inbox_item_id is not None:
                item = await session.scalar(
                    select(InboxItem.id).where(
                        InboxItem.id == capability.inbox_item_id,
                        InboxItem.user_id == owner_id,
                    )
                )
                if item is None:
                    return None
            consumed = await session.execute(
                update(LifeCollectionActionToken)
                .where(
                    LifeCollectionActionToken.token == token,
                    LifeCollectionActionToken.owner_id == owner_id,
                    LifeCollectionActionToken.chat_id == chat_id,
                    LifeCollectionActionToken.action.in_(allowed),
                    LifeCollectionActionToken.status == pending_status,
                    LifeCollectionActionToken.expires_at > current,
                )
                .values(status="consumed", consumed_at=current)
                .returning(LifeCollectionActionToken.token)
                .execution_options(synchronize_session=False)
            )
            if consumed.scalar_one_or_none() is None:
                return None
            return CollectionClaim(
                action=capability.action,
                collection_id=capability.collection_id,
                collection_version=capability.collection_version,
                inbox_item_id=capability.inbox_item_id,
                payload=dict(capability.payload or {}),
            )

    async def begin_input(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
        *,
        allowed: set[str],
        input_action: str,
    ) -> str | None:
        claim = await self.claim_action(token, owner_id, chat_id, allowed)
        if claim is None:
            return None
        collection = (
            await self._collection(owner_id, claim.collection_id)
            if claim.collection_id is not None
            else None
        )
        try:
            return await self.issue_action(
                owner_id,
                chat_id,
                input_action,
                collection=collection,
                inbox_item_id=claim.inbox_item_id,
                payload=claim.payload,
                status="awaiting_input",
                ttl=self.input_ttl,
            )
        except ValueError:
            return None

    async def pending_input(
        self, owner_id: int, chat_id: int, *, now: datetime | None = None
    ) -> LifeCollectionActionToken | None:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            expired = await session.execute(
                delete(LifeCollectionActionToken).where(
                    LifeCollectionActionToken.owner_id == owner_id,
                    LifeCollectionActionToken.chat_id == chat_id,
                    LifeCollectionActionToken.status == "awaiting_input",
                    LifeCollectionActionToken.expires_at <= current,
                )
            )
            del expired
            return await session.scalar(
                select(LifeCollectionActionToken)
                .where(
                    LifeCollectionActionToken.owner_id == owner_id,
                    LifeCollectionActionToken.chat_id == chat_id,
                    LifeCollectionActionToken.status == "awaiting_input",
                    LifeCollectionActionToken.expires_at > current,
                )
                .order_by(LifeCollectionActionToken.created_at.desc())
                .limit(1)
            )

    async def cancel_input(self, owner_id: int, chat_id: int) -> bool:
        async with self.db.session() as session:
            result = await session.execute(
                update(LifeCollectionActionToken)
                .where(
                    LifeCollectionActionToken.owner_id == owner_id,
                    LifeCollectionActionToken.chat_id == chat_id,
                    LifeCollectionActionToken.status == "awaiting_input",
                )
                .values(status="consumed", consumed_at=datetime.now(UTC))
            )
            return bool(result.rowcount)

    async def _create_collection(
        self,
        session: AsyncSession,
        owner_id: int,
        kind: CollectionKind,
        name: str,
        *,
        starter_key: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> LifeCollection:
        display, normalized = clean_collection_name(name)
        if await self._name_taken(session, owner_id, normalized):
            raise CollectionConflictError("Раздел с таким названием или alias уже существует.")
        collection = LifeCollection(
            owner_id=owner_id,
            kind=kind,
            name=display,
            normalized_name=normalized,
            starter_key=starter_key,
            status="active",
            version=1,
        )
        session.add(collection)
        await session.flush()
        seen = {normalized}
        for alias in aliases:
            alias_display, alias_normalized = clean_collection_name(alias)
            if alias_normalized in seen or await self._name_taken(
                session, owner_id, alias_normalized
            ):
                continue
            seen.add(alias_normalized)
            session.add(
                LifeCollectionAlias(
                    collection_id=collection.id,
                    owner_id=owner_id,
                    alias=alias_display,
                    normalized_alias=alias_normalized,
                )
            )
        await session.flush()
        return collection

    async def _name_taken(
        self,
        session: AsyncSession,
        owner_id: int,
        normalized: str,
        except_collection_id: int | None = None,
    ) -> bool:
        collection_query = select(LifeCollection.id).where(
            LifeCollection.owner_id == owner_id,
            LifeCollection.normalized_name == normalized,
        )
        alias_query = select(LifeCollectionAlias.collection_id).where(
            LifeCollectionAlias.owner_id == owner_id,
            LifeCollectionAlias.normalized_alias == normalized,
        )
        if except_collection_id is not None:
            collection_query = collection_query.where(LifeCollection.id != except_collection_id)
        return (
            await session.scalar(collection_query.limit(1)) is not None
            or await session.scalar(alias_query.limit(1)) is not None
        )

    async def _mark_onboarded(self, session: AsyncSession, owner_id: int) -> None:
        preference = await session.get(LifeCollectionPreference, owner_id)
        if preference is None:
            session.add(
                LifeCollectionPreference(owner_id=owner_id, onboarding_completed=True, version=1)
            )
        elif not preference.onboarding_completed:
            preference.onboarding_completed = True
            preference.version += 1

    async def _search_names(self, owner_id: int) -> list[tuple[str, LifeCollection]]:
        async with self.db.sessions() as session:
            collections = list(
                (
                    await session.scalars(
                        select(LifeCollection).where(
                            LifeCollection.owner_id == owner_id,
                            LifeCollection.status == "active",
                        )
                    )
                ).all()
            )
            aliases = (
                await session.execute(
                    select(LifeCollectionAlias, LifeCollection)
                    .join(
                        LifeCollection,
                        LifeCollection.id == LifeCollectionAlias.collection_id,
                    )
                    .where(
                        LifeCollectionAlias.owner_id == owner_id,
                        LifeCollection.owner_id == owner_id,
                        LifeCollection.status == "active",
                    )
                )
            ).all()
        values = [(collection.normalized_name, collection) for collection in collections]
        values.extend((alias.normalized_alias, collection) for alias, collection in aliases)
        return values

    async def _collection(self, owner_id: int, collection_id: int | None) -> LifeCollection | None:
        if collection_id is None:
            return None
        async with self.db.sessions() as session:
            return await session.scalar(
                select(LifeCollection).where(
                    LifeCollection.id == collection_id,
                    LifeCollection.owner_id == owner_id,
                )
            )

    @staticmethod
    async def _versioned_collection(
        session: AsyncSession,
        owner_id: int,
        collection_id: int,
        version: int,
        *,
        active_only: bool = False,
    ) -> LifeCollection | None:
        statement = select(LifeCollection).where(
            LifeCollection.id == collection_id,
            LifeCollection.owner_id == owner_id,
            LifeCollection.version == version,
        )
        if active_only:
            statement = statement.where(LifeCollection.status == "active")
        return await session.scalar(statement)

    @staticmethod
    async def _lock_owner(session: AsyncSession, owner_id: int) -> None:
        changed = await session.execute(
            update(User).where(User.id == owner_id).values(updated_at=User.updated_at)
        )
        if not changed.rowcount:
            raise ValueError("Unknown collection owner")

    async def _token(
        self, session: AsyncSession, token: str, owner_id: int, chat_id: int
    ) -> LifeCollectionActionToken | None:
        return await session.scalar(
            select(LifeCollectionActionToken).where(
                LifeCollectionActionToken.token == token,
                LifeCollectionActionToken.owner_id == owner_id,
                LifeCollectionActionToken.chat_id == chat_id,
            )
        )

    async def _new_token(
        self,
        session: AsyncSession,
        owner_id: int,
        chat_id: int,
        action: str,
        *,
        collection: LifeCollection | None,
        inbox_item_id: int | None,
        payload: dict[str, Any] | None,
        status: str,
        ttl: timedelta | None,
    ) -> str:
        token = secrets.token_urlsafe(18)
        session.add(
            LifeCollectionActionToken(
                token=token,
                owner_id=owner_id,
                chat_id=chat_id,
                collection_id=collection.id if collection else None,
                collection_version=collection.version if collection else None,
                inbox_item_id=inbox_item_id,
                action=action,
                payload=payload,
                status=status,
                expires_at=datetime.now(UTC) + (ttl or self.action_ttl),
            )
        )
        await session.flush()
        return token

    @staticmethod
    def _utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
