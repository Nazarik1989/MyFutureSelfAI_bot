import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from uuid import uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .access import FULL_ACCESS_TIERS
from .db import Database
from .models import DraftInboxItem, InboxItem, TaskReminder, TaskState, User
from .reminders import reminder_for_inbox_item
from .schemas import ParsedThought
from .tasks import add_task_state

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DraftResult:
    ok: bool
    draft: DraftInboxItem | None = None
    inbox_item: InboxItem | None = None
    reminder: TaskReminder | None = None
    duplicate: bool = False


@dataclass(slots=True)
class DraftCreation:
    draft: DraftInboxItem
    created: bool


type FencedDraftCreationStatus = Literal["created", "reused", "access_changed"]


@dataclass(frozen=True, slots=True)
class FencedDraftCreation:
    status: FencedDraftCreationStatus
    draft: DraftInboxItem | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"created", "reused"} and self.draft is not None

    @property
    def created(self) -> bool:
        return self.status == "created"


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

    SAVED_DEDUP_WINDOW = timedelta(minutes=10)

    def __init__(
        self,
        db: Database,
        ttl_minutes: int,
        *,
        task_date_event_hour: int = 9,
        task_reminder_lead_minutes: int = 30,
    ):
        self.db = db
        self.ttl = timedelta(minutes=ttl_minutes)
        self.task_date_event_hour = task_date_event_hour
        self.task_reminder_lead_minutes = task_reminder_lead_minutes

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
        async with self.db.session() as session:
            draft = await self.create_in_session(
                session,
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                source=source,
                raw_text=raw_text,
                parsed=parsed,
            )
        log_transition(draft.id, telegram_user_id, "none", "preview", "create")
        return draft

    async def create_in_session(
        self,
        session: AsyncSession,
        *,
        user_id: int,
        telegram_user_id: int,
        chat_id: int,
        source: str,
        raw_text: str,
        parsed: ParsedThought,
        now: datetime | None = None,
    ) -> DraftInboxItem:
        """Create a canonical draft without committing the caller transaction."""

        owner_exists = await session.scalar(
            select(User.id).where(
                User.id == user_id,
                User.telegram_id == telegram_user_id,
            )
        )
        if owner_exists is None:
            raise ValueError("Telegram user does not own this draft")
        current = now or datetime.now(UTC)
        draft = DraftInboxItem(
            id=str(uuid4()),
            user_id=user_id,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            source=source,
            # Reminder flow callers pass the normalized bounded title here,
            # never the raw command or voice transcript.
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
            expires_at=current + self.ttl,
            version=1,
        )
        session.add(draft)
        await session.flush()
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

    async def create_or_get_for_suggestion(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        chat_id: int,
        expected_access_version: int,
        source: str,
        raw_text: str,
        parsed: ParsedThought,
    ) -> FencedDraftCreation:
        """Create/reuse a suggestion draft for one exact full-access generation.

        The owner write lock serializes access changes and concurrent suggestion
        callbacks before any draft read or write.  A mismatched owner or access
        generation therefore fails without creating or reusing domain state.
        """

        identifiers = (user_id, telegram_user_id, chat_id, expected_access_version)
        if any(type(value) is not int or value <= 0 for value in identifiers):
            return FencedDraftCreation("access_changed")

        current = datetime.now(UTC)
        transition: tuple[str, str, str] | None = None
        async with self.db.session() as session:
            locked_owner = await session.execute(
                update(User)
                .where(
                    User.id == user_id,
                    User.telegram_id == telegram_user_id,
                    User.access_tier.in_(FULL_ACCESS_TIERS),
                    User.access_version == expected_access_version,
                )
                .values(updated_at=User.updated_at)
                .returning(User.id)
                .execution_options(synchronize_session=False)
            )
            if locked_owner.scalar_one_or_none() is None:
                return FencedDraftCreation("access_changed")

            active = tuple(
                (
                    await session.scalars(
                        select(DraftInboxItem)
                        .where(
                            DraftInboxItem.user_id == user_id,
                            DraftInboxItem.telegram_user_id == telegram_user_id,
                            DraftInboxItem.chat_id == chat_id,
                            DraftInboxItem.status == "preview",
                            DraftInboxItem.expires_at > current,
                        )
                        .order_by(DraftInboxItem.created_at.desc(), DraftInboxItem.id.desc())
                    )
                ).all()
            )
            draft = next(
                (
                    candidate
                    for candidate in active
                    if self._matches_suggestion_creation(
                        candidate,
                        source=source,
                        raw_text=raw_text,
                        parsed=parsed,
                    )
                ),
                None,
            )
            if draft is not None:
                transition = (draft.id, "preview", "reuse_duplicate")
                result = FencedDraftCreation("reused", draft)
            else:
                draft = await self.create_in_session(
                    session,
                    user_id=user_id,
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    source=source,
                    raw_text=raw_text,
                    parsed=parsed,
                    now=current,
                )
                transition = (draft.id, "none", "create_suggestion")
                result = FencedDraftCreation("created", draft)

        if transition is not None:
            draft_id, old_status, action = transition
            log_transition(draft_id, telegram_user_id, old_status, "preview", action)
        return result

    async def set_preview_message(self, draft_id: str, message_id: int) -> None:
        async with self.db.session() as session:
            await session.execute(
                update(DraftInboxItem)
                .where(DraftInboxItem.id == draft_id, DraftInboxItem.status == "preview")
                .values(preview_message_id=message_id)
            )

    async def restore_preview_message_if_current(
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_message_id: int | None,
        restored_message_id: int | None,
    ) -> bool:
        """CAS one preview pointer without overwriting a newer canonical message."""

        if (
            not isinstance(draft_id, str)
            or not draft_id
            or type(version) is not int
            or version <= 0
            or type(telegram_user_id) is not int
            or type(chat_id) is not int
            or (
                expected_message_id is not None
                and (type(expected_message_id) is not int or expected_message_id <= 0)
            )
            or (
                restored_message_id is not None
                and (type(restored_message_id) is not int or restored_message_id <= 0)
            )
        ):
            return False
        expected_pointer = (
            DraftInboxItem.preview_message_id.is_(None)
            if expected_message_id is None
            else DraftInboxItem.preview_message_id == expected_message_id
        )
        async with self.db.session() as session:
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.version == version,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "preview",
                    expected_pointer,
                )
                .values(preview_message_id=restored_message_id)
                .returning(DraftInboxItem.id)
            )
            return changed.scalar_one_or_none() is not None

    async def drop_if_preview_message_current(
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_message_id: int | None,
        expected_access_version: int | None = None,
    ) -> DraftResult:
        """Discard only the exact preview canonical owned by a failed delivery."""

        if (
            not isinstance(draft_id, str)
            or not draft_id
            or type(version) is not int
            or version <= 0
            or type(telegram_user_id) is not int
            or type(chat_id) is not int
            or (
                expected_message_id is not None
                and (type(expected_message_id) is not int or expected_message_id <= 0)
            )
        ):
            return DraftResult(False)
        expected_pointer = (
            DraftInboxItem.preview_message_id.is_(None)
            if expected_message_id is None
            else DraftInboxItem.preview_message_id == expected_message_id
        )
        owner_filter = select(User.id).where(User.telegram_id == telegram_user_id)
        if expected_access_version is not None:
            owner_filter = owner_filter.where(
                User.access_tier.in_(FULL_ACCESS_TIERS),
                User.access_version == expected_access_version,
            )
        async with self.db.session() as session:
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.user_id.in_(owner_filter),
                    DraftInboxItem.version == version,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "preview",
                    expected_pointer,
                )
                .values(status="discarded")
                .returning(DraftInboxItem.id)
            )
            if changed.scalar_one_or_none() is None:
                return DraftResult(False)
        draft = await self.get(draft_id)
        log_transition(draft_id, telegram_user_id, "preview", "discarded", "drop")
        return DraftResult(True, draft=draft)

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
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_preview_message_id: int | None = None,
        expected_access_version: int | None = None,
    ) -> DraftResult:
        now = datetime.now(UTC)
        draft = await self.get(draft_id)
        if not self._matches(draft, version, telegram_user_id, chat_id, "preview"):
            return DraftResult(False)
        owner_filter = select(User.id).where(User.telegram_id == telegram_user_id)
        if expected_access_version is not None:
            owner_filter = owner_filter.where(
                User.access_tier.in_(FULL_ACCESS_TIERS),
                User.access_version == expected_access_version,
            )
        preview_filter = (
            DraftInboxItem.id == draft_id
            if expected_preview_message_id is None
            else DraftInboxItem.preview_message_id == expected_preview_message_id
        )
        async with self.db.session() as session:
            changed = await session.execute(
                update(DraftInboxItem)
                .where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.user_id.in_(owner_filter),
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "preview",
                    DraftInboxItem.version == version,
                    DraftInboxItem.expires_at > now,
                    preview_filter,
                )
                .values(status="editing")
                .returning(DraftInboxItem.id)
            )
            if changed.scalar_one_or_none() is None:
                return DraftResult(False)
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
                    DraftInboxItem.user_id.in_(
                        select(User.id).where(User.telegram_id == telegram_user_id)
                    ),
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
        self,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_preview_message_id: int | None = None,
        expected_access_version: int | None = None,
    ) -> DraftResult:
        """The sole atomic path allowed to construct an InboxItem."""
        async with self.db.session() as session:
            result = await self.confirm_in_session(
                session,
                draft_id,
                version,
                telegram_user_id,
                chat_id,
                expected_preview_message_id=expected_preview_message_id,
                expected_access_version=expected_access_version,
            )
        if not result.ok:
            return result
        log_transition(
            draft_id,
            telegram_user_id,
            "preview",
            "confirmed",
            "reuse_saved_duplicate" if result.duplicate else "save",
            inbox_created=not result.duplicate,
        )
        return result

    async def confirm_in_session(
        self,
        session: AsyncSession,
        draft_id: str,
        version: int,
        telegram_user_id: int,
        chat_id: int,
        *,
        owner_locked: bool = False,
        allow_saved_dedup: bool = True,
        return_existing: bool = False,
        expected_access_version: int | None = None,
        expected_preview_message_id: int | None = None,
        now: datetime | None = None,
    ) -> DraftResult:
        """Confirm through the canonical Inbox/Task path inside a caller transaction.

        ``allow_saved_dedup=False`` is used for recurring creation because a
        semantically equal one-shot task must never be reused as a daily task.
        ``return_existing`` makes an exact draft replay idempotent without
        broadening the normal preview-confirm contract.
        """

        current = now or datetime.now(UTC)
        owner_filter = select(User.id).where(User.telegram_id == telegram_user_id)
        if expected_access_version is not None:
            owner_filter = owner_filter.where(
                User.access_tier.in_(FULL_ACCESS_TIERS),
                User.access_version == expected_access_version,
            )
        preview_filter = (
            DraftInboxItem.id == draft_id
            if expected_preview_message_id is None
            else DraftInboxItem.preview_message_id == expected_preview_message_id
        )
        changed = await session.execute(
            update(DraftInboxItem)
            .where(
                DraftInboxItem.id == draft_id,
                DraftInboxItem.user_id.in_(owner_filter),
                DraftInboxItem.telegram_user_id == telegram_user_id,
                DraftInboxItem.chat_id == chat_id,
                DraftInboxItem.status == "preview",
                DraftInboxItem.version == version,
                DraftInboxItem.expires_at > current,
                preview_filter,
            )
            .values(status="confirmed")
            .returning(DraftInboxItem.id)
        )
        if changed.scalar_one_or_none() is None:
            if not return_existing:
                return DraftResult(False)
            draft = await session.scalar(
                select(DraftInboxItem).where(
                    DraftInboxItem.id == draft_id,
                    DraftInboxItem.telegram_user_id == telegram_user_id,
                    DraftInboxItem.chat_id == chat_id,
                    DraftInboxItem.status == "confirmed",
                    DraftInboxItem.version == version,
                )
            )
            if draft is None:
                return DraftResult(False)
            existing = await session.scalar(
                select(InboxItem).where(
                    InboxItem.draft_id == draft_id,
                    InboxItem.user_id == draft.user_id,
                )
            )
            if existing is None:
                return DraftResult(False)
            reminder = await session.scalar(
                select(TaskReminder).where(TaskReminder.inbox_item_id == existing.id)
            )
            return DraftResult(
                True,
                draft=draft,
                inbox_item=existing,
                reminder=reminder,
                duplicate=True,
            )

        draft = await session.get(DraftInboxItem, draft_id)
        if draft is None:
            return DraftResult(False)
        if not owner_locked:
            # Cross-dialect owner lock used by every create/confirm coordinator.
            await session.execute(
                update(User).where(User.id == draft.user_id).values(updated_at=User.updated_at)
            )

        if allow_saved_dedup:
            recent_items = (
                await session.execute(
                    select(InboxItem, TaskState)
                    .outerjoin(
                        TaskState,
                        and_(
                            TaskState.inbox_item_id == InboxItem.id,
                            TaskState.owner_id == InboxItem.user_id,
                        ),
                    )
                    .where(
                        InboxItem.user_id == draft.user_id,
                        or_(
                            InboxItem.status == "confirmed",
                            and_(
                                InboxItem.kind == "task",
                                InboxItem.status == "archived",
                            ),
                        ),
                        InboxItem.created_at >= current - self.SAVED_DEDUP_WINDOW,
                    )
                    .order_by(InboxItem.id.desc())
                )
            ).all()
            duplicate = next(
                (
                    item
                    for item, task_state in recent_items
                    if (item.kind != "task" or task_state is not None)
                    and (item.kind != "task" or task_state.status == "active")
                    if self.saved_semantic_key(item) == self.saved_semantic_key(draft)
                ),
                None,
            )
            if duplicate is not None:
                reminder = await session.scalar(
                    select(TaskReminder).where(TaskReminder.inbox_item_id == duplicate.id)
                )
                return DraftResult(
                    True,
                    draft=draft,
                    inbox_item=duplicate,
                    reminder=reminder,
                    duplicate=True,
                )

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
        reminder = reminder_for_inbox_item(
            inbox_item,
            telegram_user_id=telegram_user_id,
            chat_id=chat_id,
            date_event_hour=self.task_date_event_hour,
            lead_minutes=self.task_reminder_lead_minutes,
        )
        if reminder is not None:
            session.add(reminder)
            await session.flush()
        owner = await session.get(User, draft.user_id)
        if owner is None:
            raise RuntimeError("Draft owner disappeared during confirmation")
        await add_task_state(
            session,
            inbox_item,
            owner_timezone=owner.timezone,
            reminder=reminder,
            date_event_hour=self.task_date_event_hour,
        )
        return DraftResult(True, draft=draft, inbox_item=inbox_item, reminder=reminder)

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
    def _matches_suggestion_creation(
        cls,
        draft: DraftInboxItem,
        *,
        source: str,
        raw_text: str,
        parsed: ParsedThought,
    ) -> bool:
        temporal_resolution = (
            parsed.temporal_resolution.model_dump(mode="json")
            if parsed.temporal_resolution
            else None
        )
        return bool(
            draft.source == source
            and draft.kind == parsed.kind
            and cls._normalize(draft.raw_text) == cls._normalize(raw_text)
            and cls._normalize(draft.title) == cls._normalize(parsed.title)
            and cls._normalize(draft.description or "") == cls._normalize(parsed.description or "")
            and cls._normalize(draft.next_step or "") == cls._normalize(parsed.next_step or "")
            and draft.resolved_date == parsed.resolved_date
            and (draft.temporal_resolution or None) == temporal_resolution
        )

    @classmethod
    def semantic_key(cls, draft: DraftInboxItem | InboxItem) -> tuple[str, ...]:
        temporal = draft.temporal_resolution or {}
        canonical_temporal = tuple(
            str(temporal.get(field) or "")
            for field in (
                "resolved_at",
                "remind_at",
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

    @classmethod
    def saved_semantic_key(cls, item: DraftInboxItem | InboxItem) -> tuple[str, ...]:
        """Canonical fields that define a short-window confirmed duplicate."""

        return (
            *cls.semantic_key(item),
            cls._normalize(item.next_step or ""),
            cls._normalize(item.raw_text),
        )
