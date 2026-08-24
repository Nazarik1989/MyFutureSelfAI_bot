import asyncio
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256

from sqlalchemy import delete, select, update

from .access import FULL_ACCESS_TIERS
from .db import Database
from .models import ConversationMessage, ConversationSession, DraftInboxItem, InboxItem, User
from .nova_memory import NovaMemoryValidationError, normalize_nova_memory_content

COMPANION_PROMPT_MAX_MESSAGES = 20
COMPANION_PROMPT_MESSAGE_MAX_CHARS = 600
COMPANION_PROMPT_CONTEXT_MAX_BYTES = 32 * 1024
COMPANION_REFERENCE_MAX_CHARS = 2_000
COMPANION_CONTEXT_MIN_RAW_MESSAGES = 10
COMPANION_CONTEXT_MAX_RAW_MESSAGES = 20

_COMPANION_PROMPT_SAFE_INTENTS = frozenset(
    {
        "answer",
        "companion",
        "companion_answer",
        "companion_user",
        "conversation",
        "nova_companion",
        "nova_companion_answer",
        "nova_companion_user",
        "question",
        "reflection",
    }
)


def _bounded_companion_prompt_text(value: object, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value)
    if any(
        unicodedata.category(character).startswith("C") and character not in {"\t", "\n", "\r"}
        for character in normalized
    ):
        return None
    cleaned = re.sub(r"\s+", " ", normalized).strip()
    if not cleaned:
        return None
    return cleaned[:max_chars].rstrip()


def build_companion_prompt_context(
    messages: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Build the exact bounded safe conversation payload eligible for a provider."""

    safe_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role_value = message.get("role")
        if not isinstance(role_value, str):
            continue
        role = role_value.strip().casefold()
        if role not in {"user", "assistant"}:
            continue
        intent_value = message.get("intent")
        if not isinstance(intent_value, str):
            continue
        intent = intent_value.strip().casefold()
        if not intent or intent not in _COMPANION_PROMPT_SAFE_INTENTS:
            continue
        content = _bounded_companion_prompt_text(
            message.get("content"), COMPANION_PROMPT_MESSAGE_MAX_CHARS
        )
        if content is None:
            continue
        safe_messages.append({"role": role, "content": content})

    if not safe_messages:
        return {}
    return {
        "recent_messages": [
            dict(message) for message in safe_messages[-COMPANION_PROMPT_MAX_MESSAGES:]
        ]
    }


def companion_conversation_revision(
    *,
    owner_id: int,
    telegram_user_id: int,
    chat_id: int,
    context: Mapping[str, object],
) -> str:
    """Return an opaque identity-bound digest of the exact safe provider payload."""

    manifest = {
        "owner": owner_id,
        "telegram_actor": telegram_user_id,
        "chat": chat_id,
        "recent_conversation": context,
    }
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def fit_companion_prompt_context(
    context: Mapping[str, object],
    max_payload_bytes: int,
) -> dict[str, object]:
    """Fit the safe recent-conversation value to its frozen provider byte budget."""

    if (
        type(max_payload_bytes) is not int
        or not 0 <= max_payload_bytes <= COMPANION_PROMPT_CONTEXT_MAX_BYTES
    ):
        raise ValueError("Invalid companion conversation payload budget")
    if not isinstance(context, Mapping) or set(context) - {"recent_messages"}:
        raise ValueError("Invalid companion conversation context")
    messages = context.get("recent_messages")
    if messages is None:
        return {}
    if not isinstance(messages, list):
        raise ValueError("Invalid companion conversation context")
    selected: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("Invalid companion conversation context")
        role = message.get("role")
        content = message.get("content")
        if (
            role not in {"user", "assistant"}
            or not isinstance(content, str)
            or not content
            or len(content) > COMPANION_PROMPT_MESSAGE_MAX_CHARS
        ):
            raise ValueError("Invalid companion conversation context")
        selected.append({"role": str(role), "content": content})
    if len(selected) > COMPANION_PROMPT_MAX_MESSAGES:
        raise ValueError("Invalid companion conversation context")
    while selected:
        projected = {"recent_messages": selected}
        serialized = json.dumps(
            projected,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        if len(serialized.encode("utf-8")) <= max_payload_bytes:
            return {"recent_messages": [dict(message) for message in selected]}
        selected.pop(0)
    return {}


@dataclass(slots=True)
class ConversationSnapshot:
    session_id: int | None = None
    current_topic: str | None = None
    summary: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list, repr=False)
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
        provider_messages = [
            dict(message) for message in self.messages if message.get("intent") != "memory_answer"
        ]
        return {
            "current_topic": self.current_topic,
            "summary": self.summary,
            "recent_messages": provider_messages,
            "active_draft": self.active_draft,
            "pending_date_options": self.pending_date_options,
            "resolved_date": self.resolved_date,
            "focused_draft_id": self.focused_draft_id,
            "focused_draft_version": self.focused_draft_version,
            "pending_action": self.pending_action,
            "system_pending_action": self.system_pending_action,
        }

    def for_companion_prompt(self) -> dict[str, object]:
        """Return a bounded, content-only view of ordinary recent conversation.

        Companion context is deliberately fail-closed: persisted messages carrying
        an unknown or control-flow intent are not eligible.  In particular this
        keeps drafts, previews, commands, reminder/date state, navigation and
        privacy-sensitive feature flows outside the provider projection.
        """

        # These session-level fields do not record which intent last wrote
        # them. They may describe a draft or another excluded durable flow,
        # even when an older surviving message is ordinary conversation.
        # Per-message role/content is the only provenance we can prove here.
        return build_companion_prompt_context(self.messages)


@dataclass(frozen=True, slots=True)
class SystemActionClaim:
    action: str
    snapshot: list[dict[str, object]]
    version: int


@dataclass(frozen=True, slots=True)
class ActiveDraftFocusLease:
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    installed_draft_id: str = field(repr=False)
    installed_draft_version: int = field(repr=False)
    installed_focus_expires_at: datetime = field(repr=False)
    previous_active_draft_id: str | None = field(default=None, repr=False)
    previous_focused_draft_id: str | None = field(default=None, repr=False)
    previous_focused_draft_version: int | None = field(default=None, repr=False)
    previous_pending_action: str | None = field(default=None, repr=False)
    previous_focus_expires_at: datetime | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class CompanionConversationFence:
    """Opaque owner/chat/access identity for one bounded safe conversation view."""

    owner_id: int = field(repr=False)
    telegram_user_id: int = field(repr=False)
    chat_id: int = field(repr=False)
    access_version: int = field(repr=False)
    access_tier: str = field(repr=False)
    raw_message_limit: int = field(repr=False)
    conversation_payload_max_bytes: int = field(repr=False)
    revision: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.owner_id,
                    self.telegram_user_id,
                    self.chat_id,
                    self.access_version,
                    self.raw_message_limit,
                )
            )
            or self.access_tier not in FULL_ACCESS_TIERS
            or not (
                COMPANION_CONTEXT_MIN_RAW_MESSAGES
                <= self.raw_message_limit
                <= COMPANION_CONTEXT_MAX_RAW_MESSAGES
            )
            or type(self.conversation_payload_max_bytes) is not int
            or not (0 <= self.conversation_payload_max_bytes <= COMPANION_PROMPT_CONTEXT_MAX_BYTES)
            or len(self.revision) != 64
            or any(character not in "0123456789abcdef" for character in self.revision)
        ):
            raise ValueError("Invalid companion conversation fence")


@dataclass(frozen=True, slots=True)
class _ConversationMessageBackup:
    message_id: int = field(repr=False)
    role: str = field(repr=False)
    content: str = field(repr=False)
    timestamp: datetime = field(repr=False)
    source: str = field(repr=False)
    intent: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ConversationExchangeReceipt:
    """Opaque exact-generation receipt for one committed companion exchange."""

    _prior_fence: CompanionConversationFence = field(repr=False)
    _result_fence: CompanionConversationFence = field(repr=False)
    _session_id: int = field(repr=False)
    _user_message_id: int = field(repr=False)
    _assistant_message_id: int = field(repr=False)
    _inserted_messages: tuple[_ConversationMessageBackup, ...] = field(repr=False)
    _prior_messages: tuple[_ConversationMessageBackup, ...] = field(repr=False)
    _post_message_ids: tuple[int, ...] = field(repr=False)
    _previous_expires_at: datetime | None = field(repr=False)
    _installed_expires_at: datetime = field(repr=False)

    def result_fence_for(
        self,
        expected: CompanionConversationFence,
    ) -> CompanionConversationFence | None:
        """Return the post-exchange fence only for the exact prior generation."""

        if type(expected) is not CompanionConversationFence or expected != self._prior_fence:
            return None
        return self._result_fence

    def source_identity_for(
        self,
        expected: CompanionConversationFence,
    ) -> "ConversationExchangeSource | None":
        """Expose only a content-free provenance receipt for the exact prior fence."""

        if type(expected) is not CompanionConversationFence or expected != self._prior_fence:
            return None
        manifest = (
            f"{self._prior_fence.owner_id}:{self._session_id}:"
            f"{self._user_message_id}:{self._assistant_message_id}:"
            f"{self._result_fence.revision}"
        )
        return ConversationExchangeSource(
            session_id=self._session_id,
            user_message_id=self._user_message_id,
            assistant_message_id=self._assistant_message_id,
            receipt=sha256(manifest.encode("ascii")).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class ConversationExchangeSource:
    """Content-free durable provenance for a committed companion exchange."""

    session_id: int = field(repr=False)
    user_message_id: int = field(repr=False)
    assistant_message_id: int = field(repr=False)
    receipt: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.session_id,
                    self.user_message_id,
                    self.assistant_message_id,
                )
            )
            or len(self.receipt) != 64
            or any(character not in "0123456789abcdef" for character in self.receipt)
        ):
            raise ValueError("Invalid companion exchange source")


class _ConversationExchangeChanged(Exception):
    """Content-free sentinel used to roll back an access/context CAS miss."""


class ConversationContextService:
    MAX_PURGE_BATCH_SIZE = 100

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
            # Take the database write/row lock before reading the session.  PostgreSQL
            # serializes the no-op UPDATE with a concurrent DELETE on this row, while
            # SQLite serializes the writers at the database level.  Consequently the
            # SELECT below observes either the still-current row or its committed
            # deletion, even when append and retention use different processes.
            await session.execute(
                update(ConversationSession)
                .where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
                .values(
                    expires_at=ConversationSession.expires_at,
                    updated_at=ConversationSession.updated_at,
                )
                .execution_options(synchronize_session=False)
            )
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

    async def append_exchange(
        self,
        fence: CompanionConversationFence,
        *,
        user_content: str,
        assistant_content: str,
        user_source: str,
    ) -> ConversationExchangeReceipt | None:
        """Atomically append one owner/access/context-fenced companion exchange.

        ``None`` is a normal CAS miss: the actor's access generation or the
        exact bounded conversation projection changed before the write lock.
        The opaque receipt can later compensate only this exact committed pair.
        """

        if type(fence) is not CompanionConversationFence:
            raise ValueError("Invalid companion conversation fence")
        if fence.raw_message_limit != self.message_limit:
            return None
        user_text = self._exchange_text(user_content, "User content")
        assistant_text = self._exchange_text(assistant_content, "Assistant content")
        source = self._exchange_source(user_source)
        await self._before_exchange_access_lock(fence)
        now = datetime.now(UTC)
        receipt: ConversationExchangeReceipt | None = None
        try:
            async with self.db.sessions() as session:
                async with session.begin():
                    actor_lock = await session.execute(
                        update(User)
                        .where(
                            User.id == fence.owner_id,
                            User.telegram_id == fence.telegram_user_id,
                            User.access_version == fence.access_version,
                            User.access_tier == fence.access_tier,
                            User.access_tier.in_(FULL_ACCESS_TIERS),
                        )
                        .values(updated_at=User.updated_at)
                        .execution_options(synchronize_session=False)
                    )
                    if actor_lock.rowcount != 1:
                        raise _ConversationExchangeChanged

                    await session.execute(
                        update(ConversationSession)
                        .where(
                            ConversationSession.telegram_user_id == fence.telegram_user_id,
                            ConversationSession.chat_id == fence.chat_id,
                        )
                        .values(
                            expires_at=ConversationSession.expires_at,
                            updated_at=ConversationSession.updated_at,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    conversation = await session.scalar(
                        select(ConversationSession).where(
                            ConversationSession.telegram_user_id == fence.telegram_user_id,
                            ConversationSession.chat_id == fence.chat_id,
                        )
                    )
                    previous_expires_at: datetime | None = None
                    if conversation is None:
                        conversation = ConversationSession(
                            telegram_user_id=fence.telegram_user_id,
                            chat_id=fence.chat_id,
                            expires_at=now + self.ttl,
                        )
                        session.add(conversation)
                        await session.flush()
                    elif self._is_expired(conversation.expires_at, now):
                        await self._reset_expired_conversation(session, conversation)
                    else:
                        previous_expires_at = conversation.expires_at

                    prior_rows = await self._conversation_messages(session, conversation.id)
                    prior_rows = prior_rows[-fence.raw_message_limit :]
                    prior_context = build_companion_prompt_context(
                        self._message_projection(row) for row in prior_rows
                    )
                    current_fence = self._companion_fence_like(fence, prior_context)
                    if current_fence != fence:
                        raise _ConversationExchangeChanged

                    installed_expires_at = now + self.ttl
                    conversation.expires_at = installed_expires_at
                    user_message = ConversationMessage(
                        session_id=conversation.id,
                        role="user",
                        content=user_text,
                        source=source,
                        intent="companion_user",
                    )
                    session.add(user_message)
                    await session.flush()

                    await self._before_exchange_assistant_insert(fence)
                    assistant_message = ConversationMessage(
                        session_id=conversation.id,
                        role="assistant",
                        content=assistant_text,
                        source="text",
                        intent="companion_answer",
                    )
                    session.add(assistant_message)
                    await session.flush()

                    result_rows = await self._conversation_messages(session, conversation.id)
                    keep_limit = max(2, self.message_limit)
                    evicted_rows = result_rows[:-keep_limit]
                    result_rows = result_rows[-keep_limit:]
                    if evicted_rows:
                        await session.execute(
                            delete(ConversationMessage).where(
                                ConversationMessage.session_id == conversation.id,
                                ConversationMessage.id.in_(tuple(row.id for row in evicted_rows)),
                            )
                        )
                        await session.flush()

                    result_context = build_companion_prompt_context(
                        self._message_projection(row) for row in result_rows
                    )
                    result_fence = self._companion_fence_like(fence, result_context)
                    inserted = (
                        self._message_backup(user_message),
                        self._message_backup(assistant_message),
                    )
                    receipt = ConversationExchangeReceipt(
                        _prior_fence=fence,
                        _result_fence=result_fence,
                        _session_id=conversation.id,
                        _user_message_id=user_message.id,
                        _assistant_message_id=assistant_message.id,
                        _inserted_messages=inserted,
                        _prior_messages=tuple(self._message_backup(row) for row in prior_rows),
                        _post_message_ids=tuple(row.id for row in result_rows),
                        _previous_expires_at=previous_expires_at,
                        _installed_expires_at=installed_expires_at,
                    )
            await self._after_exchange_commit(receipt)
            return receipt
        except _ConversationExchangeChanged:
            if receipt is not None:
                _cleaned, cancellation_observed = await self._shielded_exchange_compensation(
                    receipt
                )
                if cancellation_observed:
                    raise asyncio.CancelledError from None
            return None
        except (Exception, asyncio.CancelledError) as exc:
            cancellation_observed = isinstance(exc, asyncio.CancelledError)
            if receipt is not None:
                _cleaned, cleanup_cancelled = await self._shielded_exchange_compensation(receipt)
                cancellation_observed = cancellation_observed or cleanup_cancelled
            if cancellation_observed and not isinstance(exc, asyncio.CancelledError):
                raise asyncio.CancelledError from None
            raise

    async def compensate_exchange(self, receipt: ConversationExchangeReceipt) -> bool:
        """CAS-remove one exact exchange without touching any newer generation."""

        if type(receipt) is not ConversationExchangeReceipt:
            raise ValueError("Invalid companion exchange receipt")
        fence = receipt._result_fence
        try:
            async with self.db.sessions() as session:
                async with session.begin():
                    owner_lock = await session.execute(
                        update(User)
                        .where(
                            User.id == fence.owner_id,
                            User.telegram_id == fence.telegram_user_id,
                        )
                        .values(updated_at=User.updated_at)
                        .execution_options(synchronize_session=False)
                    )
                    if owner_lock.rowcount != 1:
                        raise _ConversationExchangeChanged
                    session_lock = await session.execute(
                        update(ConversationSession)
                        .where(
                            ConversationSession.id == receipt._session_id,
                            ConversationSession.telegram_user_id == fence.telegram_user_id,
                            ConversationSession.chat_id == fence.chat_id,
                        )
                        .values(
                            expires_at=ConversationSession.expires_at,
                            updated_at=ConversationSession.updated_at,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    if session_lock.rowcount != 1:
                        raise _ConversationExchangeChanged
                    conversation = await session.get(ConversationSession, receipt._session_id)
                    if conversation is None:
                        raise _ConversationExchangeChanged
                    rows = await self._conversation_messages(session, conversation.id)
                    generation_is_current = (
                        tuple(row.id for row in rows) == receipt._post_message_ids
                    )
                    inserted_by_id = {
                        backup.message_id: backup for backup in receipt._inserted_messages
                    }
                    if set(inserted_by_id) != {
                        receipt._user_message_id,
                        receipt._assistant_message_id,
                    }:
                        raise _ConversationExchangeChanged
                    actual_by_id = {row.id: row for row in rows}
                    surviving_inserted = {
                        message_id: actual_by_id[message_id]
                        for message_id in inserted_by_id
                        if message_id in actual_by_id
                    }
                    if not surviving_inserted:
                        raise _ConversationExchangeChanged
                    for message_id, row in surviving_inserted.items():
                        if self._message_backup(row) != inserted_by_id[message_id]:
                            raise _ConversationExchangeChanged

                    prior_by_id = {backup.message_id: backup for backup in receipt._prior_messages}
                    if len(prior_by_id) != len(receipt._prior_messages):
                        raise _ConversationExchangeChanged
                    current_non_receipt = [
                        self._message_backup(row) for row in rows if row.id not in inserted_by_id
                    ]
                    combined = dict(prior_by_id)
                    for backup in current_non_receipt:
                        previous = combined.get(backup.message_id)
                        if previous is not None and previous != backup:
                            raise _ConversationExchangeChanged
                        combined[backup.message_id] = backup
                    desired = tuple(
                        sorted(combined.values(), key=lambda item: item.message_id)[
                            -receipt._result_fence.raw_message_limit :
                        ]
                    )
                    desired_by_id = {backup.message_id: backup for backup in desired}

                    if surviving_inserted:
                        removed = await session.execute(
                            delete(ConversationMessage).where(
                                ConversationMessage.session_id == conversation.id,
                                ConversationMessage.id.in_(tuple(surviving_inserted)),
                            )
                        )
                        if removed.rowcount != len(surviving_inserted):
                            raise _ConversationExchangeChanged

                    current_non_receipt_ids = {backup.message_id for backup in current_non_receipt}
                    missing_prior = [
                        backup
                        for backup in desired
                        if backup.message_id in prior_by_id
                        and backup.message_id not in current_non_receipt_ids
                    ]
                    if missing_prior:
                        collisions = set(
                            (
                                await session.scalars(
                                    select(ConversationMessage.id).where(
                                        ConversationMessage.id.in_(
                                            tuple(backup.message_id for backup in missing_prior)
                                        )
                                    )
                                )
                            ).all()
                        )
                        if collisions:
                            raise _ConversationExchangeChanged
                        for backup in missing_prior:
                            session.add(
                                ConversationMessage(
                                    id=backup.message_id,
                                    session_id=conversation.id,
                                    role=backup.role,
                                    content=backup.content,
                                    timestamp=backup.timestamp,
                                    source=backup.source,
                                    intent=backup.intent,
                                )
                            )
                    if (
                        generation_is_current
                        and receipt._previous_expires_at is not None
                        and conversation.expires_at == receipt._installed_expires_at
                    ):
                        conversation.expires_at = receipt._previous_expires_at
                    await session.flush()

                    restored_rows = await self._conversation_messages(session, conversation.id)
                    restored_by_id = {row.id: self._message_backup(row) for row in restored_rows}
                    if any(message_id in restored_by_id for message_id in inserted_by_id):
                        raise _ConversationExchangeChanged
                    for backup in current_non_receipt:
                        if restored_by_id.get(backup.message_id) != backup:
                            raise _ConversationExchangeChanged
                    for message_id, backup in desired_by_id.items():
                        if restored_by_id.get(message_id) != backup:
                            raise _ConversationExchangeChanged
            return True
        except _ConversationExchangeChanged:
            return False

    async def _before_exchange_access_lock(self, fence: CompanionConversationFence) -> None:
        """Deterministic test seam before the exact access-generation write lock."""

        del fence

    async def _before_exchange_assistant_insert(
        self,
        fence: CompanionConversationFence,
    ) -> None:
        """Deterministic test seam after the user row but before the assistant row."""

        del fence

    async def _after_exchange_commit(self, receipt: ConversationExchangeReceipt) -> None:
        """Deterministic test seam after commit but before receipt publication."""

        del receipt

    async def _shielded_exchange_compensation(
        self,
        receipt: ConversationExchangeReceipt,
    ) -> tuple[bool, bool]:
        coroutine = self.compensate_exchange(receipt)
        try:
            task = asyncio.create_task(
                coroutine,
                name="companion-exchange-post-commit-compensation",
            )
        except BaseException:
            coroutine.close()
            return False, False
        cancellation_observed = False
        while True:
            try:
                return await asyncio.shield(task), cancellation_observed
            except asyncio.CancelledError:
                cancellation_observed = True
                if not task.done():
                    continue
                try:
                    return task.result(), cancellation_observed
                except (Exception, asyncio.CancelledError):
                    return False, cancellation_observed
            except Exception:
                return False, cancellation_observed

    async def set_active_draft(
        self, telegram_user_id: int, chat_id: int, draft_id: str | None
    ) -> bool:
        """Set the active draft and report whether this call acquired the focus.

        A ``False`` result for a valid ``draft_id`` means the exact same draft
        generation was already active.  Compensation callers use this ownership
        signal so a failed replay does not clear focus that predated the replay.
        """

        now = datetime.now(UTC)
        focus_acquired = False
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
                focus_acquired = any(
                    value is not None
                    for value in (
                        conversation.active_draft_id,
                        conversation.focused_draft_id,
                        conversation.focused_draft_version,
                        conversation.pending_action,
                        conversation.focus_expires_at,
                    )
                )
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
                    focus_acquired = (
                        conversation.active_draft_id != draft.id
                        or conversation.focused_draft_id != draft.id
                        or conversation.focused_draft_version != draft.version
                        or conversation.pending_action is not None
                        or conversation.focus_expires_at is None
                    )
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
        return focus_acquired

    async def acquire_active_draft_focus(
        self,
        telegram_user_id: int,
        chat_id: int,
        draft_id: str,
        *,
        now: datetime | None = None,
    ) -> ActiveDraftFocusLease | None:
        """Install one preview focus and freeze the exact prior state for CAS restore."""

        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        else:
            current = current.astimezone(UTC)
        async with self.db.session() as session:
            await session.execute(
                update(ConversationSession)
                .where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                )
                .values(
                    expires_at=ConversationSession.expires_at,
                    updated_at=ConversationSession.updated_at,
                )
                .execution_options(synchronize_session=False)
            )
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
                    expires_at=current + self.ttl,
                )
                session.add(conversation)
            draft = await session.get(DraftInboxItem, draft_id)
            if (
                draft is None
                or draft.telegram_user_id != telegram_user_id
                or draft.chat_id != chat_id
                or draft.status != "preview"
            ):
                return None
            installed_focus_expires_at = current + self.focus_ttl
            lease = ActiveDraftFocusLease(
                telegram_user_id=telegram_user_id,
                chat_id=chat_id,
                installed_draft_id=draft.id,
                installed_draft_version=draft.version,
                previous_active_draft_id=conversation.active_draft_id,
                previous_focused_draft_id=conversation.focused_draft_id,
                previous_focused_draft_version=conversation.focused_draft_version,
                previous_pending_action=conversation.pending_action,
                previous_focus_expires_at=conversation.focus_expires_at,
                installed_focus_expires_at=installed_focus_expires_at,
            )
            conversation.active_draft_id = draft.id
            conversation.focused_draft_id = draft.id
            conversation.focused_draft_version = draft.version
            conversation.pending_action = None
            conversation.focus_expires_at = installed_focus_expires_at
            conversation.expires_at = current + self.ttl
            return lease

    async def restore_active_draft_focus_if_current(
        self,
        lease: ActiveDraftFocusLease,
        *,
        restore_prior: bool = True,
    ) -> bool:
        """Restore a lease's prior state only while its installed focus is current."""

        if not isinstance(lease, ActiveDraftFocusLease):
            return False
        async with self.db.session() as session:
            previous_active = lease.previous_active_draft_id if restore_prior else None
            if previous_active is not None:
                active = await session.get(DraftInboxItem, previous_active)
                if (
                    active is None
                    or active.telegram_user_id != lease.telegram_user_id
                    or active.chat_id != lease.chat_id
                    or active.status != "preview"
                ):
                    previous_active = None
            previous_focused = lease.previous_focused_draft_id if restore_prior else None
            previous_version = lease.previous_focused_draft_version if restore_prior else None
            if previous_focused is not None:
                focused = await session.get(DraftInboxItem, previous_focused)
                if (
                    focused is None
                    or focused.telegram_user_id != lease.telegram_user_id
                    or focused.chat_id != lease.chat_id
                    or focused.status != "preview"
                    or focused.version != previous_version
                ):
                    previous_focused = None
                    previous_version = None
            pending_action = lease.previous_pending_action if previous_focused is not None else None
            focus_expires_at = (
                lease.previous_focus_expires_at if previous_focused is not None else None
            )
            changed = await session.execute(
                update(ConversationSession)
                .where(
                    ConversationSession.telegram_user_id == lease.telegram_user_id,
                    ConversationSession.chat_id == lease.chat_id,
                    ConversationSession.active_draft_id == lease.installed_draft_id,
                    ConversationSession.focused_draft_id == lease.installed_draft_id,
                    ConversationSession.focused_draft_version == lease.installed_draft_version,
                    ConversationSession.pending_action.is_(None),
                    ConversationSession.focus_expires_at == lease.installed_focus_expires_at,
                )
                .values(
                    active_draft_id=previous_active,
                    focused_draft_id=previous_focused,
                    focused_draft_version=previous_version,
                    pending_action=pending_action,
                    focus_expires_at=focus_expires_at,
                )
                .returning(ConversationSession.id)
            )
            return changed.scalar_one_or_none() is not None

    async def clear_active_draft_if_current(
        self,
        telegram_user_id: int,
        chat_id: int,
        draft_id: str,
        draft_version: int,
    ) -> bool:
        """Clear only the exact focus generation installed by a failed flow."""

        if (
            type(telegram_user_id) is not int
            or type(chat_id) is not int
            or not isinstance(draft_id, str)
            or not draft_id
            or type(draft_version) is not int
            or draft_version <= 0
        ):
            return False
        async with self.db.session() as session:
            changed = await session.execute(
                update(ConversationSession)
                .where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                    ConversationSession.active_draft_id == draft_id,
                    ConversationSession.focused_draft_id == draft_id,
                    ConversationSession.focused_draft_version == draft_version,
                )
                .values(
                    active_draft_id=None,
                    focused_draft_id=None,
                    focused_draft_version=None,
                    pending_action=None,
                    focus_expires_at=None,
                )
                .returning(ConversationSession.id)
            )
            return changed.scalar_one_or_none() is not None

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
            bumped = await session.execute(
                update(ConversationSession)
                .where(ConversationSession.id == conversation.id)
                .values(
                    system_action_version=ConversationSession.system_action_version + 1,
                    system_pending_action=action,
                    system_draft_snapshot=draft_snapshot,
                    system_action_expires_at=now + self.system_action_ttl,
                    expires_at=now + self.ttl,
                )
                .returning(ConversationSession.system_action_version)
                .execution_options(synchronize_session=False)
            )
            return int(bumped.scalar_one())

    async def claim_system_action(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_version: int,
    ) -> SystemActionClaim | None:
        """Atomically consume one current system action and return its immutable payload."""

        if (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version <= 0
        ):
            return None
        now = datetime.now(UTC)
        async with self.db.session() as session:
            result = await session.execute(
                update(ConversationSession)
                .where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                    ConversationSession.system_action_version == expected_version,
                    ConversationSession.system_pending_action.is_not(None),
                    ConversationSession.system_action_expires_at.is_not(None),
                    ConversationSession.system_action_expires_at > now,
                    ConversationSession.expires_at > now,
                )
                # Keep the payload available to RETURNING, but make the row
                # immediately ineligible for a replaying claim.
                .values(system_action_expires_at=None)
                .returning(
                    ConversationSession.system_pending_action,
                    ConversationSession.system_draft_snapshot,
                    ConversationSession.system_action_version,
                )
                .execution_options(synchronize_session=False)
            )
            claimed = result.mappings().one_or_none()
            if claimed is None:
                return None
            return SystemActionClaim(
                action=str(claimed["system_pending_action"]),
                snapshot=list(claimed["system_draft_snapshot"] or []),
                version=int(claimed["system_action_version"]),
            )

    async def clear_system_action(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_version: int | None = None,
    ) -> bool:
        """Cancel an unclaimed action, optionally only at one exact version."""

        if expected_version is not None and (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version <= 0
        ):
            return False
        async with self.db.session() as session:
            statement = update(ConversationSession).where(
                ConversationSession.telegram_user_id == telegram_user_id,
                ConversationSession.chat_id == chat_id,
                ConversationSession.system_pending_action.is_not(None),
                ConversationSession.system_action_expires_at.is_not(None),
            )
            if expected_version is not None:
                statement = statement.where(
                    ConversationSession.system_action_version == expected_version,
                )
            cleared = await session.execute(
                statement.values(
                    system_pending_action=None,
                    system_draft_snapshot=None,
                    system_action_expires_at=None,
                )
                .returning(ConversationSession.id)
                .execution_options(synchronize_session=False)
            )
            return cleared.scalar_one_or_none() is not None

    async def finalize_system_action_claim(
        self,
        telegram_user_id: int,
        chat_id: int,
        *,
        expected_version: int,
    ) -> bool:
        """Clear only the exact action already consumed by ``claim_system_action``."""

        if (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version <= 0
        ):
            return False
        async with self.db.session() as session:
            cleared = await session.execute(
                update(ConversationSession)
                .where(
                    ConversationSession.telegram_user_id == telegram_user_id,
                    ConversationSession.chat_id == chat_id,
                    ConversationSession.system_action_version == expected_version,
                    ConversationSession.system_pending_action.is_not(None),
                    ConversationSession.system_action_expires_at.is_(None),
                )
                .values(
                    system_pending_action=None,
                    system_draft_snapshot=None,
                )
                .returning(ConversationSession.id)
                .execution_options(synchronize_session=False)
            )
            return cleared.scalar_one_or_none() is not None

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
                    InboxItem.status == "confirmed",
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

    async def purge_expired(
        self,
        batch_size: int = MAX_PURGE_BATCH_SIZE,
        *,
        now: datetime | None = None,
    ) -> int:
        """Hard-delete one bounded batch of expired conversation sessions.

        Conversation messages are removed by the existing database cascade. The
        return value intentionally contains only an aggregate session count.
        """

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= self.MAX_PURGE_BATCH_SIZE
        ):
            raise ValueError(f"batch_size must be between 1 and {self.MAX_PURGE_BATCH_SIZE}")
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        else:
            current = current.astimezone(UTC)

        expired_ids = (
            select(ConversationSession.id)
            .where(ConversationSession.expires_at <= current)
            .order_by(ConversationSession.expires_at, ConversationSession.id)
            .limit(batch_size)
        )
        async with self.db.session() as session:
            deleted = await session.execute(
                delete(ConversationSession)
                .where(
                    ConversationSession.id.in_(expired_ids),
                    ConversationSession.expires_at <= current,
                )
                .returning(ConversationSession.id)
                .execution_options(synchronize_session=False)
            )
            return len(deleted.scalars().all())

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
    def companion_reference_candidate(snapshot: ConversationSnapshot) -> str | None:
        """Return only the latest ordinary user message for explicit capture.

        Unlike the legacy helper, this never skips an excluded durable/private
        message to reach older content.  The same intent provenance used by the
        companion prompt therefore fences references such as ``save this``.
        """

        latest_user = next(
            (message for message in reversed(snapshot.messages) if message.get("role") == "user"),
            None,
        )
        if latest_user is None or latest_user.get("source") not in {"text", "voice"}:
            return None
        intent_value = latest_user.get("intent")
        if not isinstance(intent_value, str):
            return None
        intent = intent_value.strip().casefold()
        if not intent or intent not in _COMPANION_PROMPT_SAFE_INTENTS:
            return None
        content = _bounded_companion_prompt_text(
            latest_user.get("content"),
            COMPANION_REFERENCE_MAX_CHARS,
        )
        if content is None:
            return None
        lowered = content.casefold()
        if any(
            marker in lowered
            for marker in (
                "сохрани это",
                "добавь туда",
                "как я говорил",
                "продолжим",
                "ты занес",
            )
        ):
            return None
        return content

    @staticmethod
    def latest_nova_memory_candidate(snapshot: ConversationSnapshot) -> str | None:
        """Return only the latest bounded user reply when it is safe to preview.

        The helper deliberately does not skip an unsuitable latest user message in
        search of older content.  That keeps the meaning of ``remember this``
        bounded without claiming Telegram-level adjacency.
        """

        latest_user = next(
            (message for message in reversed(snapshot.messages) if message.get("role") == "user"),
            None,
        )
        if latest_user is None:
            return None
        source = latest_user.get("source")
        intent = latest_user.get("intent")
        blocked_intents = {
            "command",
            "control",
            "error",
            "navigation",
            "system",
            "system_action",
            "relative_reminder",
            "date_conflict",
            "confirm_date",
            "explicit_capture",
        }
        intent_key = intent.casefold() if isinstance(intent, str) else ""
        if (
            source not in {"text", "voice"}
            or intent_key in blocked_intents
            or any(
                intent_key.startswith(f"{prefix}:")
                for prefix in ("command", "control", "error", "navigation", "system")
            )
        ):
            return None
        content = latest_user.get("content")
        if not isinstance(content, str):
            return None
        try:
            normalized = normalize_nova_memory_content(content)
        except NovaMemoryValidationError:
            return None

        # Keep the dependency local so the durable conversation service remains
        # independent from the process-local memory flow lifecycle.
        from .nova_memory_flow import NovaMemoryIntentKind, classify_nova_memory_intent

        if classify_nova_memory_intent(normalized).kind is not NovaMemoryIntentKind.NONE:
            return None
        return normalized

    @staticmethod
    async def _reset_expired_conversation(
        session: object, conversation: ConversationSession
    ) -> None:
        await session.execute(
            delete(ConversationMessage).where(ConversationMessage.session_id == conversation.id)
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

    @staticmethod
    async def _conversation_messages(
        session: object,
        session_id: int,
    ) -> list[ConversationMessage]:
        return list(
            (
                await session.scalars(
                    select(ConversationMessage)
                    .where(ConversationMessage.session_id == session_id)
                    .order_by(ConversationMessage.id)
                )
            ).all()
        )

    @staticmethod
    def _message_projection(message: ConversationMessage) -> dict[str, object]:
        return {
            "role": message.role,
            "content": message.content,
            "intent": message.intent,
        }

    @staticmethod
    def _message_backup(message: ConversationMessage) -> _ConversationMessageBackup:
        return _ConversationMessageBackup(
            message_id=message.id,
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
            source=message.source,
            intent=message.intent,
        )

    @staticmethod
    def _companion_fence_like(
        identity: CompanionConversationFence,
        context: Mapping[str, object],
    ) -> CompanionConversationFence:
        projected = fit_companion_prompt_context(
            context,
            identity.conversation_payload_max_bytes,
        )
        return CompanionConversationFence(
            owner_id=identity.owner_id,
            telegram_user_id=identity.telegram_user_id,
            chat_id=identity.chat_id,
            access_version=identity.access_version,
            access_tier=identity.access_tier,
            raw_message_limit=identity.raw_message_limit,
            conversation_payload_max_bytes=identity.conversation_payload_max_bytes,
            revision=companion_conversation_revision(
                owner_id=identity.owner_id,
                telegram_user_id=identity.telegram_user_id,
                chat_id=identity.chat_id,
                context=projected,
            ),
        )

    @staticmethod
    def _exchange_text(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty")
        return value

    @staticmethod
    def _exchange_source(value: object) -> str:
        if value not in {"text", "voice"}:
            raise ValueError("User source must be text or voice")
        return str(value)

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
