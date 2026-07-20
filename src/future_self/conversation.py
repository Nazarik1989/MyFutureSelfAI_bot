from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import delete, select

from .db import Database
from .models import ConversationMessage, ConversationSession, DraftInboxItem, InboxItem, User


@dataclass(slots=True)
class ConversationSnapshot:
    session_id: int | None = None
    current_topic: str | None = None
    summary: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    active_draft: dict[str, object] | None = None
    pending_date_options: list[dict[str, str]] = field(default_factory=list)
    resolved_date: str | None = None
    focused_draft_id: str | None = None
    focused_draft_version: int | None = None
    pending_action: str | None = None
    focus_expires_at: str | None = None
    system_pending_action: str | None = None
    system_draft_snapshot: list[dict[str, object]] = field(default_factory=list)
    system_action_version: int | None = None
    system_action_expires_at: str | None = None
    last_saved_inbox_item_id: int | None = None
    last_saved_at: str | None = None

    def for_prompt(self) -> dict[str, object]:
        return {
            "current_topic": self.current_topic,
            "summary": self.summary,
            "recent_messages": self.messages,
            "active_draft": self.active_draft,
            "pending_date_options": self.pending_date_options,
            "resolved_date": self.resolved_date,
            "focused_draft_id": self.focused_draft_id,
            "focused_draft_version": self.focused_draft_version,
            "pending_action": self.pending_action,
            "system_pending_action": self.system_pending_action,
        }


class ConversationContextService:
    def __init__(
        self,
        db: Database,
        message_limit: int,
        ttl_hours: int,
        focus_ttl_minutes: int = 15,
        system_action_ttl_minutes: int = 10,
    ):
        self.db = db
        self.message_limit = message_limit
        self.ttl = timedelta(hours=ttl_hours)
        self.focus_ttl = timedelta(minutes=focus_ttl_minutes)
        self.system_action_ttl = timedelta(minutes=system_action_ttl_minutes)

    async def get(self, telegram_user_id: int, chat_id: int) -> ConversationSnapshot:
        now = datetime.now(UTC)
        async with self.db.sessions() as session:
            conversation = await session.scalar(
                select(ConversationSession).where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
            )
            if conversation is None or self._is_expired(conversation.expires_at, now):
                return ConversationSnapshot()
            rows = list(
                (
                    await session.scalars(
                        select(ConversationMessage)
                        .where(ConversationMessage.session_id == conversation.id)
                        .order_by(ConversationMessage.id.desc())
                        .limit(self.message_limit)
                    )
                ).all()
            )
            active = None
            if conversation.active_draft_id:
                draft = await session.get(DraftInboxItem, conversation.active_draft_id)
                if (
                    draft
                    and draft.telegram_user_id == telegram_user_id
                    and draft.chat_id == chat_id
                    and draft.status in {"preview", "editing"}
                ):
                    active = {
                        "id": draft.id,
                        "version": draft.version,
                        "status": draft.status,
                        "kind": draft.kind,
                        "title": draft.title,
                        "description": draft.description,
                        "resolved_date": (
                            draft.resolved_date.isoformat() if draft.resolved_date else None
                        ),
                        "temporal_resolution": draft.temporal_resolution,
                    }
            focused_id = None
            focused_version = None
            pending_action = None
            focus_expires_at = None
            if self._focus_is_current(conversation, now):
                pending_action = conversation.pending_action
                focus_expires_at = conversation.focus_expires_at.isoformat()
                if conversation.focused_draft_id and conversation.focused_draft_version:
                    focused = await session.get(DraftInboxItem, conversation.focused_draft_id)
                    if (
                        focused
                        and focused.telegram_user_id == telegram_user_id
                        and focused.chat_id == chat_id
                        and focused.status == "preview"
                        and focused.version == conversation.focused_draft_version
                        and not self._is_expired(focused.expires_at, now)
                    ):
                        focused_id = focused.id
                        focused_version = focused.version
            system_pending_action = None
            system_snapshot: list[dict[str, object]] = []
            system_action_version = None
            system_expires_at = None
            if self._system_action_is_current(conversation, now):
                system_pending_action = conversation.system_pending_action
                system_snapshot = conversation.system_draft_snapshot or []
                system_action_version = conversation.system_action_version
                system_expires_at = conversation.system_action_expires_at.isoformat()
            return ConversationSnapshot(
                session_id=conversation.id,
                current_topic=conversation.current_topic,
                summary=conversation.summary,
                messages=[
                    {
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp.isoformat(),
                        "source": row.source,
                        "intent": row.intent,
                    }
                    for row in reversed(rows)
                ],
                active_draft=active,
                pending_date_options=conversation.pending_date_options or [],
                resolved_date=(
                    conversation.resolved_date.isoformat() if conversation.resolved_date else None
                ),
                focused_draft_id=focused_id,
                focused_draft_version=focused_version,
                pending_action=pending_action,
                focus_expires_at=focus_expires_at,
                system_pending_action=system_pending_action,
                system_draft_snapshot=system_snapshot,
                system_action_version=system_action_version,
                system_action_expires_at=system_expires_at,
                last_saved_inbox_item_id=conversation.last_saved_inbox_item_id,
                last_saved_at=(
                    conversation.last_saved_at.isoformat() if conversation.last_saved_at else None
                ),
            )

    async def append(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        role: str,
        content: str,
        source: str,
        intent: str,
        topic: str | None = None,
    ) -> int:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            conversation = await session.scalar(
                select(ConversationSession).where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
            )
            if conversation is None:
                conversation = ConversationSession(
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    expires_at=now + self.ttl,
                )
                session.add(conversation)
                await session.flush()
            elif self._is_expired(conversation.expires_at, now):
                await session.execute(
                    delete(ConversationMessage).where(
                        ConversationMessage.session_id == conversation.id
                    )
                )
                conversation.summary = None
                conversation.current_topic = None
                conversation.active_draft_id = None
                conversation.pending_date_options = None
                conversation.resolved_date = None
                conversation.focused_draft_id = None
                conversation.focused_draft_version = None
                conversation.pending_action = None
                conversation.focus_expires_at = None
                conversation.system_pending_action = None
                conversation.system_draft_snapshot = None
                conversation.system_action_expires_at = None
            conversation.expires_at = now + self.ttl
            if topic:
                conversation.current_topic = topic[:200]
            session.add(
                ConversationMessage(
                    session_id=conversation.id,
                    role=role,
                    content=content,
                    source=source,
                    intent=intent,
                )
            )
            await session.flush()
            keep_ids = (
                select(ConversationMessage.id)
                .where(ConversationMessage.session_id == conversation.id)
                .order_by(ConversationMessage.id.desc())
                .limit(self.message_limit)
            )
            await session.execute(
                delete(ConversationMessage).where(
                    ConversationMessage.session_id == conversation.id,
                    ConversationMessage.id.not_in(keep_ids),
                )
            )
            return conversation.id

    async def set_active_draft(
        self, telegram_user_id: int, chat_id: int, draft_id: str | None
    ) -> None:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            conversation = await session.scalar(
                select(ConversationSession).where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
            )
            if conversation is None:
                conversation = ConversationSession(
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    expires_at=now + self.ttl,
                )
                session.add(conversation)
            if draft_id is None:
                conversation.active_draft_id = None
                conversation.focused_draft_id = None
                conversation.focused_draft_version = None
                conversation.pending_action = None
                conversation.focus_expires_at = None
            else:
                draft = await session.get(DraftInboxItem, draft_id)
                if (
                    draft
                    and draft.telegram_user_id == telegram_user_id
                    and draft.chat_id == chat_id
                    and draft.status == "preview"
                ):
                    conversation.active_draft_id = draft.id
                    conversation.focused_draft_id = draft.id
                    conversation.focused_draft_version = draft.version
                    conversation.pending_action = None
                    conversation.focus_expires_at = now + self.focus_ttl
                else:
                    conversation.active_draft_id = None
                    conversation.focused_draft_id = None
                    conversation.focused_draft_version = None
                    conversation.pending_action = None
                    conversation.focus_expires_at = None
            conversation.expires_at = now + self.ttl

    async def set_pending_action(self, telegram_user_id: int, chat_id: int, action: str) -> None:
        await self._set_focus_state(
            telegram_user_id,
            chat_id,
            focused_draft_id=None,
            focused_draft_version=None,
            pending_action=action,
        )

    async def set_focus(
        self,
        telegram_user_id: int,
        chat_id: int,
        draft_id: str,
        version: int,
        pending_action: str | None,
    ) -> None:
        await self._set_focus_state(
            telegram_user_id,
            chat_id,
            focused_draft_id=draft_id,
            focused_draft_version=version,
            pending_action=pending_action,
        )

    async def clear_focus(self, telegram_user_id: int, chat_id: int) -> None:
        await self._set_focus_state(
            telegram_user_id,
            chat_id,
            focused_draft_id=None,
            focused_draft_version=None,
            pending_action=None,
            expires=False,
        )

    async def begin_system_action(
        self,
        telegram_user_id: int,
        chat_id: int,
        action: str,
        draft_snapshot: list[dict[str, object]],
    ) -> int:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            conversation = await self._get_or_create_session(
                session, telegram_user_id, chat_id, now
            )
            conversation.system_action_version = (conversation.system_action_version or 0) + 1
            conversation.system_pending_action = action
            conversation.system_draft_snapshot = draft_snapshot
            conversation.system_action_expires_at = now + self.system_action_ttl
            conversation.expires_at = now + self.ttl
            return conversation.system_action_version

    async def clear_system_action(self, telegram_user_id: int, chat_id: int) -> None:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            conversation = await self._get_or_create_session(
                session, telegram_user_id, chat_id, now
            )
            conversation.system_pending_action = None
            conversation.system_draft_snapshot = None
            conversation.system_action_expires_at = None

    async def record_saved(
        self,
        telegram_user_id: int,
        chat_id: int,
        inbox_item_id: int,
    ) -> None:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            conversation = await self._get_or_create_session(
                session, telegram_user_id, chat_id, now
            )
            owned_item = await session.scalar(
                select(InboxItem.id)
                .join(User, User.id == InboxItem.user_id)
                .where(
                    InboxItem.id == inbox_item_id,
                    User.telegram_id == telegram_user_id,
                )
            )
            if owned_item is None:
                return
            conversation.last_saved_inbox_item_id = inbox_item_id
            conversation.last_saved_at = now
            conversation.expires_at = now + self.ttl

    async def _get_or_create_session(
        self, session: object, telegram_user_id: int, chat_id: int, now: datetime
    ) -> ConversationSession:
        conversation = await session.scalar(
            select(ConversationSession).where(
                ConversationSession.telegram_user_id == telegram_user_id,
                ConversationSession.chat_id == chat_id,
            )
        )
        if conversation is None:
            conversation = ConversationSession(
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                expires_at=now + self.ttl,
            )
            session.add(conversation)
            await session.flush()
        return conversation

    async def _set_focus_state(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        focused_draft_id: str | None,
        focused_draft_version: int | None,
        pending_action: str | None,
        expires: bool = True,
    ) -> None:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            conversation = await session.scalar(
                select(ConversationSession).where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
            )
            if conversation is None:
                conversation = ConversationSession(
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    expires_at=now + self.ttl,
                )
                session.add(conversation)
            if focused_draft_id is not None:
                draft = await session.scalar(
                    select(DraftInboxItem).where(
                        DraftInboxItem.id == focused_draft_id,
                        DraftInboxItem.telegram_user_id == telegram_user_id,
                        DraftInboxItem.chat_id == chat_id,
                        DraftInboxItem.status == "preview",
                        DraftInboxItem.version == focused_draft_version,
                    )
                )
                if draft is None:
                    focused_draft_id = None
                    focused_draft_version = None
                    pending_action = None
                    expires = False
            conversation.focused_draft_id = focused_draft_id
            conversation.focused_draft_version = focused_draft_version
            conversation.pending_action = pending_action
            conversation.focus_expires_at = now + self.focus_ttl if expires else None
            conversation.expires_at = now + self.ttl

    async def set_date_conflict(
        self,
        telegram_user_id: int,
        chat_id: int,
        options: list[dict[str, str]],
    ) -> None:
        await self._set_date_state(
            telegram_user_id,
            chat_id,
            pending_date_options=options,
            resolved_date=None,
        )

    async def set_resolved_date(
        self, telegram_user_id: int, chat_id: int, resolved_date: date
    ) -> None:
        await self._set_date_state(
            telegram_user_id,
            chat_id,
            pending_date_options=None,
            resolved_date=resolved_date,
        )

    async def _set_date_state(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        pending_date_options: list[dict[str, str]] | None,
        resolved_date: date | None,
    ) -> None:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            conversation = await session.scalar(
                select(ConversationSession).where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
            )
            if conversation is None:
                conversation = ConversationSession(
                    telegram_user_id=telegram_user_id,
                    chat_id=chat_id,
                    expires_at=now + self.ttl,
                )
                session.add(conversation)
            conversation.pending_date_options = pending_date_options
            conversation.resolved_date = resolved_date
            conversation.expires_at = now + self.ttl

    async def by_id(
        self, session_id: int, telegram_user_id: int, chat_id: int
    ) -> ConversationSession | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(ConversationSession).where(
                    ConversationSession.id == session_id,
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
            )

    @staticmethod
    def reference_candidate(snapshot: ConversationSnapshot) -> str | None:
        candidates = [
            message["content"]
            for message in snapshot.messages
            if message["role"] == "user"
            and len(message["content"].strip()) > 12
            and not any(
                marker in message["content"].lower()
                for marker in (
                    "сохрани это",
                    "добавь туда",
                    "как я говорил",
                    "продолжим",
                    "ты занес",
                )
            )
        ]
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _is_expired(value: datetime, now: datetime) -> bool:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value <= now

    @classmethod
    def _focus_is_current(cls, conversation: ConversationSession, now: datetime) -> bool:
        return bool(
            conversation.focus_expires_at
            and not cls._is_expired(conversation.focus_expires_at, now)
            and (conversation.pending_action or conversation.focused_draft_id)
        )

    @classmethod
    def _system_action_is_current(cls, conversation: ConversationSession, now: datetime) -> bool:
        return bool(
            conversation.system_pending_action
            and conversation.system_action_expires_at
            and not cls._is_expired(conversation.system_action_expires_at, now)
        )
