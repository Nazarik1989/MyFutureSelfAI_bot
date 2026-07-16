import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update

from .db import Database
from .models import DraftInboxItem, InboxItem
from .schemas import ParsedThought

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DraftResult:
    ok: bool
    draft: DraftInboxItem | None = None
    inbox_item: InboxItem | None = None


@dataclass(slots=True)
class DraftCreation:
    draft: DraftInboxItem
    created: bool


@dataclass(slots=True)
class BatchDraftResult:
    ok: bool
    count: int = 0
    preview_message_ids: list[int] | None = None


class DraftSnapshotChanged(RuntimeError):
    pass


def masked_user(telegram_user_id: int) -> str:
    return hashlib.sha256(str(telegram_user_id).encode()).hexdigest()[:8]


def log_transition(
    draft_id: str,
    telegram_user_id: int,
    old_status: str,
    new_status: str,
    action: str,
    *,
    inbox_created: bool = False,
) -> None:
    logger.info(
        "draft=%s user=%s transition=%s->%s action=%s inbox_created=%s",
        draft_id[:8],
        masked_user(telegram_user_id),
        old_status,
        new_status,
        action,
        inbox_created,
    )


class DraftInboxService:
    """Persistent draft state machine. Only confirm creates an InboxItem."""

    def __init__(self, db: Database, ttl_minutes: int):
        self.db = db
        self.ttl = timedelta(minutes=ttl_minutes)

    async def create(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        chat_id: int,
        source: str,
        raw_text: str,
        parsed: ParsedThought,
    ) -> DraftInboxItem:
        now = datetime.now(UTC)
        draft = DraftInboxItem(
            id=str(uuid4()),
            user_id=user_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            source=source,
            raw_text=raw_text,
            kind=parsed.kind,
            title=parsed.title,
            description=parsed.description,
            next_step=parsed.next_step,
            resolved_date=parsed.resolved_date,
            temporal_resolution=(
                parsed.temporal_resolution.model_dump(mode="json")
                if parsed.temporal_resolution
                else None
            ),
            status="preview",
            expires_at=now + self.ttl,
            version=1,
        )
        async with self.db.session() as session:
            session.add(draft)
            await session.flush()
        log_transition(draft.id, telegram_user_id, "none", "preview", "create")
        return draft

    async def create_or_get(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        chat_id: int,
        source: str,
        raw_text: str,
        parsed: ParsedThought,
    ) -> DraftCreation:
        active = await self.active_previews(telegram_user_id, chat_id)
        normalized_title = self._normalize(parsed.title)
        normalized_raw = self._normalize(raw_text)
        for draft in active:
            if (
                draft.kind == parsed.kind
                and self._normalize(draft.title) == normalized_title
                and self._normalize(draft.raw_text) == normalized_raw
            ):
                log_transition(
                    draft.id,
                    telegram_user_id,
                    "preview",
                    "preview",
                    "reuse_duplicate",
                )
                return DraftCreation(draft=draft, created=False)
        return DraftCreation(
            draft=await self.create(
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source=source,
                raw_text=raw_text,
                parsed=parsed,
            ),
            created=True,
        )

    async def set_preview_message(self, draft_id: str, message_id: int) -> None:
        async with self.db.session() as session:
            await session.execute(
                update(DraftInboxItem)
                .where(DraftInboxItem.id == draft_id, DraftInboxItem.status == "preview")
                .values(preview_message_id=message_id)
            )

    async def _mark_expired(self, draft: DraftInboxItem, now: datetime) -> bool:
        expires_at = draft.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if expires_at > now:
            return False
        async with self.db.session() as session:
            await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft.id,
                    DraftInboxItem.status.in_(("preview", "editing")),
                )
                .values(status="expired")
            )
        log_transition(draft.id, draft.telegram_user_id, draft.status, "expired", "expire")
        return True

    async def get(self, draft_id: str) -> DraftInboxItem | None:
        async with self.db.sessions() as session:
            return await session.get(DraftInboxItem, draft_id)

    async def begin_edit(
        self, draft_id: str, version: int, telegram_user_id: int, chat_id: int
    ) -> DraftResult:
        now = datetime.now(UTC)
        draft = await self.get(draft_id)
        if not self._matches(draft, version, telegram_user_id, chat_id, "preview"):
            return DraftResult(False)
        if await self._mark_expired(draft, now):
            return DraftResult(False)
        async with self.db.session() as session:
            await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "editing",
                    DraftInboxItem.id != draft_id,
                )
                .values(status="discarded")
            )
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "preview",
                    DraftInboxItem.version == version,
                    DraftInboxItem.expires_at > now,
                )
                .values(status="editing")
                .returning(DraftInboxItem.id)
            )
            if changed.scalar_one_or_none() is None:
                return DraftResult(False)
        draft = await self.get(draft_id)
        log_transition(draft_id, telegram_user_id, "preview", "editing", "edit")
        return DraftResult(True, draft=draft)

    async def editing(self, telegram_user_id: int, chat_id: int) -> DraftInboxItem | None:
        async with self.db.sessions() as session:
            draft = await session.scalar(
                select(DraftInboxItem)
                .where(
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "editing",
                )
                .order_by(DraftInboxItem.created_at.desc())
                .limit(1)
            )
        if draft and await self._mark_expired(draft, datetime.now(UTC)):
            return None
        return draft

    async def active_previews(self, telegram_user_id: int, chat_id: int) -> list[DraftInboxItem]:
        """Return every unexpired preview; command callers must require exactly one."""
        now = datetime.now(UTC)
        await self.expire_stale(telegram_user_id, chat_id, now=now)
        async with self.db.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(DraftInboxItem)
                        .where(
                            DraftInboxItem.telegram_user_id == telegram_user_id,
                            DraftInboxItem.chat_id == chat_id,
                            DraftInboxItem.status == "preview",
                            DraftInboxItem.expires_at > now,
                        )
                        .order_by(
                            DraftInboxItem.expires_at.desc(),
                            DraftInboxItem.created_at.desc(),
                        )
                    )
                ).all()
            )

    async def active_drafts(self, telegram_user_id: int, chat_id: int) -> list[DraftInboxItem]:
        now = datetime.now(UTC)
        await self.expire_stale(telegram_user_id, chat_id, now=now)
        async with self.db.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(DraftInboxItem)
                        .where(
                            DraftInboxItem.telegram_user_id == telegram_user_id,
                            DraftInboxItem.chat_id == chat_id,
                            DraftInboxItem.status.in_(("preview", "editing")),
                            DraftInboxItem.expires_at > now,
                        )
                        .order_by(
                            DraftInboxItem.expires_at.desc(),
                            DraftInboxItem.created_at.desc(),
                        )
                    )
                ).all()
            )

    async def discard_snapshot(
        self,
        telegram_user_id: int,
        chat_id: int,
        snapshot: list[dict[str, object]],
    ) -> BatchDraftResult:
        """Discard an unchanged active-set snapshot atomically."""
        now = datetime.now(UTC)
        expected = {(str(item["id"]), int(item["version"])) for item in snapshot}
        affected = {str(item["id"]) for item in snapshot if bool(item.get("affected"))}
        if not affected:
            return BatchDraftResult(False)
        try:
            async with self.db.session() as session:
                rows = list(
                    (
                        await session.scalars(
                            select(DraftInboxItem).where(
                                DraftInboxItem.telegram_user_id == telegram_user_id,
                                DraftInboxItem.chat_id == chat_id,
                                DraftInboxItem.status.in_(("preview", "editing")),
                                DraftInboxItem.expires_at > now,
                            )
                        )
                    ).all()
                )
                current = {(draft.id, draft.version) for draft in rows}
                if current != expected or not affected <= {draft.id for draft in rows}:
                    raise DraftSnapshotChanged
                message_ids = [
                    draft.preview_message_id
                    for draft in rows
                    if draft.id in affected and draft.preview_message_id is not None
                ]
                changed = await session.execute(
                    update(DraftInboxItem)
                    .where(
                        DraftInboxItem.id.in_(affected),
                        DraftInboxItem.telegram_user_id == telegram_user_id,
                        DraftInboxItem.chat_id == chat_id,
                        DraftInboxItem.status.in_(("preview", "editing")),
                        DraftInboxItem.expires_at > now,
                    )
                    .values(status="discarded")
                    .returning(DraftInboxItem.id)
                    .execution_options(synchronize_session=False)
                )
                changed_ids = list(changed.scalars())
                if set(changed_ids) != affected:
                    raise DraftSnapshotChanged
        except DraftSnapshotChanged:
            return BatchDraftResult(False)
        for draft_id in affected:
            log_transition(
                draft_id,
                telegram_user_id,
                "preview",
                "discarded",
                "batch_discard",
            )
        return BatchDraftResult(True, len(affected), message_ids)

    async def expire_stale(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        now: datetime | None = None,
    ) -> int:
        current = now or datetime.now(UTC)
        async with self.db.session() as session:
            result = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status.in_(("preview", "editing")),
                    DraftInboxItem.expires_at <= current,
                )
                .values(status="expired")
                .returning(DraftInboxItem.id)
            )
            expired = list(result.scalars())
        for draft_id in expired:
            log_transition(draft_id, telegram_user_id, "preview", "expired", "expire")
        return len(expired)

    async def active_by_id(
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
    ) -> DraftInboxItem | None:
        await self.expire_stale(telegram_user_id, chat_id)
        draft = await self.get(draft_id)
        return (
            draft if self._matches(draft, version, telegram_user_id, chat_id, "preview") else None
        )

    async def by_preview_message(
        self, telegram_user_id: int, chat_id: int, message_id: int
    ) -> DraftInboxItem | None:
        await self.expire_stale(telegram_user_id, chat_id)
        async with self.db.sessions() as session:
            return await session.scalar(
                select(DraftInboxItem).where(
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.preview_message_id == message_id,
                    DraftInboxItem.status == "preview",
                    DraftInboxItem.expires_at > datetime.now(UTC),
                )
            )

    async def revise(
        self,
        draft_id: str,
        telegram_user_id: int,
        chat_id: int,
        raw_text: str,
        source: str,
        parsed: ParsedThought,
    ) -> DraftResult:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "editing",
                    DraftInboxItem.expires_at > now,
                )
                .values(
                    raw_text=raw_text,
                    source=source,
                    kind=parsed.kind,
                    title=parsed.title,
                    description=parsed.description,
                    next_step=parsed.next_step,
                    resolved_date=parsed.resolved_date,
                    temporal_resolution=(
                        parsed.temporal_resolution.model_dump(mode="json")
                        if parsed.temporal_resolution
                        else None
                    ),
                    status="preview",
                    version=DraftInboxItem.version + 1,
                    preview_message_id=None,
                    expires_at=now + self.ttl,
                )
                .returning(DraftInboxItem.id)
            )
            if changed.scalar_one_or_none() is None:
                return DraftResult(False)
        draft = await self.get(draft_id)
        log_transition(draft_id, telegram_user_id, "editing", "preview", "revise")
        return DraftResult(True, draft=draft)

    async def transform(
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        parsed: ParsedThought,
        *,
        raw_text: str | None = None,
    ) -> DraftResult:
        """Atomically replace a preview with a new version of the same draft."""
        now = datetime.now(UTC)
        values: dict[str, object] = {
            "kind": parsed.kind,
            "title": parsed.title,
            "description": parsed.description,
            "next_step": parsed.next_step,
            "resolved_date": parsed.resolved_date,
            "temporal_resolution": (
                parsed.temporal_resolution.model_dump(mode="json")
                if parsed.temporal_resolution
                else None
            ),
            "version": DraftInboxItem.version + 1,
            "preview_message_id": None,
            "expires_at": now + self.ttl,
        }
        if raw_text is not None:
            values["raw_text"] = raw_text
        async with self.db.session() as session:
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "preview",
                    DraftInboxItem.version == version,
                    DraftInboxItem.expires_at > now,
                )
                .values(**values)
                .returning(DraftInboxItem.id)
            )
            if changed.scalar_one_or_none() is None:
                return DraftResult(False)
        draft = await self.get(draft_id)
        log_transition(draft_id, telegram_user_id, "preview", "preview", "transform")
        return DraftResult(True, draft=draft)

    async def apply_resolved_date(
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        resolved_date: date,
    ) -> DraftResult:
        draft = await self.get(draft_id)
        if not self._matches(draft, version, telegram_user_id, chat_id, "preview"):
            return DraftResult(False)
        parsed = ParsedThought(
            kind=draft.kind,
            title=draft.title,
            description=draft.description,
            next_step=draft.next_step,
            resolved_date=resolved_date,
            temporal_resolution=draft.temporal_resolution,
        )
        return await self.transform(draft_id, version, telegram_user_id, chat_id, parsed)

    async def drop(
        self, draft_id: str, version: int, telegram_user_id: int, chat_id: int
    ) -> DraftResult:
        return await self._transition_preview(
            draft_id, version, telegram_user_id, chat_id, "discarded", "drop"
        )

    async def cancel_editing(self, telegram_user_id: int, chat_id: int) -> bool:
        async with self.db.session() as session:
            result = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "editing",
                )
                .values(status="discarded")
                .returning(DraftInboxItem.id)
            )
            draft_id = result.scalar_one_or_none()
        if draft_id:
            log_transition(draft_id, telegram_user_id, "editing", "discarded", "cancel")
        return draft_id is not None

    async def confirm(
        self, draft_id: str, version: int, telegram_user_id: int, chat_id: int
    ) -> DraftResult:
        """The sole atomic path allowed to construct an InboxItem."""
        now = datetime.now(UTC)
        async with self.db.session() as session:
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "preview",
                    DraftInboxItem.version == version,
                    DraftInboxItem.expires_at > now,
                )
                .values(status="confirmed")
                .returning(DraftInboxItem.id)
            )
            if changed.scalar_one_or_none() is None:
                return DraftResult(False)
            draft = await session.get(DraftInboxItem, draft_id)
            inbox_item = InboxItem(
                draft_id=draft.id,
                user_id=draft.user_id,
                kind=draft.kind,
                title=draft.title,
                description=draft.description,
                raw_text=draft.raw_text,
                next_step=draft.next_step,
                resolved_date=draft.resolved_date,
                temporal_resolution=draft.temporal_resolution,
                source=draft.source,
                status="confirmed",
            )
            session.add(inbox_item)
            await session.flush()
        log_transition(
            draft_id,
            telegram_user_id,
            "preview",
            "confirmed",
            "save",
            inbox_created=True,
        )
        return DraftResult(True, draft=draft, inbox_item=inbox_item)

    async def _transition_preview(
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        new_status: str,
        action: str,
    ) -> DraftResult:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "preview",
                    DraftInboxItem.version == version,
                    DraftInboxItem.expires_at > now,
                )
                .values(status=new_status)
                .returning(DraftInboxItem.id)
            )
            if changed.scalar_one_or_none() is None:
                return DraftResult(False)
        draft = await self.get(draft_id)
        log_transition(draft_id, telegram_user_id, "preview", new_status, action)
        return DraftResult(True, draft=draft)

    @staticmethod
    def _matches(
        draft: DraftInboxItem | None,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        status: str,
    ) -> bool:
        return bool(
            draft
            and draft.telegram_user_id == telegram_user_id
            and draft.chat_id == chat_id
            and draft.status == status
            and draft.version == version
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-zа-я0-9]+", " ", value.lower().replace("ё", "е")).strip()

    @classmethod
    def semantic_key(cls, draft: DraftInboxItem) -> tuple[str, ...]:
        temporal = draft.temporal_resolution or {}
        canonical_temporal = tuple(
            str(temporal.get(field) or "")
            for field in (
                "resolved_at",
                "resolved_local_date",
                "resolved_local_time",
                "timezone",
                "precision",
                "resolution_status",
            )
        )
        return (
            draft.kind,
            cls._normalize(draft.title),
            cls._normalize(draft.description or ""),
            draft.resolved_date.isoformat() if draft.resolved_date else "",
            *canonical_temporal,
        )
