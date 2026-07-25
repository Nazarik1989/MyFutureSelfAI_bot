from __future__ import annotations

import hashlib
import json
import re
import secrets
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .db import Database
from .models import (
    KnowledgeActionToken,
    KnowledgeAuditEvent,
    KnowledgeCaptureDraft,
    KnowledgeIngestionJob,
    KnowledgeQuotaReservation,
    KnowledgeRuntimeState,
    KnowledgeSource,
    KnowledgeSourceRevision,
    KnowledgeSpace,
    User,
    Workspace,
    WorkspaceMember,
    WorkspaceProject,
)

KnowledgeRole = Literal[
    "foundation", "trusted", "perspective", "discussion", "counterpoint", "hypothesis"
]
KnowledgePriority = Literal["high", "normal", "low"]
KnowledgeClassification = Literal["general", "health_private"]
CaptureKind = Literal["text", "forward", "document", "image", "url"]
FailureKind = Literal["retryable", "permanent", "quarantine"]

KNOWLEDGE_ROLES = frozenset(
    {"foundation", "trusted", "perspective", "discussion", "counterpoint", "hypothesis"}
)
KNOWLEDGE_PRIORITIES = frozenset({"high", "normal", "low"})
KNOWLEDGE_CLASSIFICATIONS = frozenset({"general", "health_private"})
CAPTURE_KINDS = frozenset({"text", "forward", "document", "image", "url"})
SPACE_ROLES = frozenset({"owner", "editor", "viewer"})
EDIT_ROLES = frozenset({"owner", "editor"})
OWNER_ROLES = frozenset({"owner"})
SOURCE_LIFECYCLES = frozenset({"active", "trashed", "purge_pending", "purge_failed", "purged"})
SOURCE_PROCESSING_STATUSES = frozenset(
    {"queued", "processing", "ready", "partial", "failed", "quarantined", "cancelled"}
)
_SAFE_CODE = re.compile(r"[a-z0-9][a-z0-9_]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ACTION = re.compile(r"[a-z0-9][a-z0-9_:.-]{0,47}\Z")
_UNSET = object()


class KnowledgeError(ValueError):
    """Base class for deliberately safe Knowledge-domain failures."""


class KnowledgeAccessDenied(KnowledgeError):
    """Used for both absent and inaccessible objects to avoid an ID oracle."""


class KnowledgeConflictError(KnowledgeError):
    pass


class KnowledgeStaleError(KnowledgeError):
    pass


class KnowledgeQuotaError(KnowledgeError):
    pass


class KnowledgeCaptureError(KnowledgeError):
    pass


class KnowledgeJobError(KnowledgeError):
    pass


@dataclass(frozen=True, slots=True)
class KnowledgeQuotaPolicy:
    max_source_bytes: int = 25 * 1024 * 1024
    max_extracted_bytes: int = 5 * 1024 * 1024
    daily_ingest_bytes_per_user: int = 100 * 1024 * 1024
    storage_bytes_per_user: int = 1024 * 1024 * 1024
    daily_sources_per_user: int = 20
    max_pending_jobs_per_user: int = 4
    daily_ingest_bytes_per_space: int = 100 * 1024 * 1024
    storage_bytes_per_space: int = 1024 * 1024 * 1024
    daily_sources_per_space: int = 100
    max_pending_jobs_per_space: int = 20

    def __post_init__(self) -> None:
        values = (
            self.max_source_bytes,
            self.max_extracted_bytes,
            self.daily_ingest_bytes_per_user,
            self.storage_bytes_per_user,
            self.daily_sources_per_user,
            self.max_pending_jobs_per_user,
            self.daily_ingest_bytes_per_space,
            self.storage_bytes_per_space,
            self.daily_sources_per_space,
            self.max_pending_jobs_per_space,
        )
        if any(value < 1 for value in values):
            raise ValueError("Knowledge quotas must be positive")
        if self.max_source_bytes > self.daily_ingest_bytes_per_user:
            raise ValueError("Source limit cannot exceed the user daily quota")
        if self.max_source_bytes > self.daily_ingest_bytes_per_space:
            raise ValueError("Source limit cannot exceed the space daily quota")
        if self.daily_ingest_bytes_per_user > self.storage_bytes_per_user:
            raise ValueError("User daily quota cannot exceed user storage quota")
        if self.daily_ingest_bytes_per_space > self.storage_bytes_per_space:
            raise ValueError("Space daily quota cannot exceed space storage quota")
        if self.max_source_bytes + self.max_extracted_bytes > self.storage_bytes_per_user:
            raise ValueError("One source and its extraction must fit the user storage quota")
        if self.max_source_bytes + self.max_extracted_bytes > self.storage_bytes_per_space:
            raise ValueError("One source and its extraction must fit the space storage quota")


@dataclass(frozen=True, slots=True)
class KnowledgeAccessContext:
    actor_user_id: int
    knowledge_space_id: int
    space_public_id: str
    kind: str
    role: str
    space_version: int
    workspace_id: int | None
    workspace_access_epoch: int | None
    workspace_project_id: int | None
    workspace_project_version: int | None


@dataclass(frozen=True, slots=True)
class KnowledgeSpaceRecord:
    access: KnowledgeAccessContext
    name: str
    status: str


@dataclass(frozen=True, slots=True)
class StoredKnowledgeOriginal:
    storage_key: str
    sha256: str
    size_bytes: int
    declared_mime: str | None
    detected_mime: str
    detected_format: str
    safe_display_name: str
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeCapturePreview:
    draft_public_id: str
    version: int
    status: str
    capture_kind: str
    target_space_public_id: str | None
    target_name: str | None
    title: str | None
    knowledge_role: str
    priority: str
    system_classification: str
    user_classification: str | None
    content_preview: str | None = field(repr=False)
    declared_mime: str | None
    declared_size_bytes: int | None
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeCaptureState:
    preview: KnowledgeCapturePreview | None
    expired_now: bool


@dataclass(frozen=True, slots=True)
class KnowledgeCaptureReservation:
    public_id: str
    draft_public_id: str
    draft_version: int
    reserved_bytes: int
    expires_at: datetime
    material: KnowledgeCaptureMaterial = field(repr=False)


@dataclass(frozen=True, slots=True)
class KnowledgeCaptureMaterial:
    capture_kind: str
    text_content: str | None = field(default=None, repr=False)
    source_url: str | None = field(default=None, repr=False)
    telegram_file_id: str | None = field(default=None, repr=False)
    declared_mime: str | None = None
    safe_display_name: str | None = None
    declared_size_bytes: int | None = None
    provenance: dict[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class KnowledgeSourceRecord:
    source: KnowledgeSource
    revision: KnowledgeSourceRevision | None
    active_job: KnowledgeIngestionJob | None
    role: str


@dataclass(frozen=True, slots=True)
class KnowledgeSourcePage:
    items: tuple[KnowledgeSourceRecord, ...]
    page: int
    pages: int
    total: int


@dataclass(frozen=True, slots=True)
class KnowledgeSourceReceipt:
    source_public_id: str
    revision_public_id: str
    job_public_id: str
    processing_status: str
    source_version: int


@dataclass(frozen=True, slots=True)
class ClaimedKnowledgeJob:
    id: int
    public_id: str
    source_id: int
    source_public_id: str
    source_version: int
    revision_id: int | None
    revision_number: int | None
    knowledge_space_id: int
    job_type: str
    lease_token: str
    original_storage_key: str | None
    original_sha256: str | None
    declared_mime: str | None
    detected_mime: str | None
    detected_format: str | None
    size_bytes: int | None
    attempt_count: int
    max_attempts: int
    cancel_requested: bool
    asset_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeExtractionResult:
    status: Literal["ready", "partial"]
    extracted_storage_key: str | None = None
    extracted_sha256: str | None = None
    extracted_size_bytes: int | None = None
    metadata: dict[str, Any] | None = None
    safe_error_code: str | None = None


@dataclass(frozen=True, slots=True)
class IssuedKnowledgeAction:
    token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ClaimedKnowledgeAction:
    action: str
    payload: dict[str, Any]
    scope_kind: str
    space_public_id: str
    capture_draft_public_id: str | None
    source_public_id: str | None


class KnowledgeService:
    CAPTURE_TTL = timedelta(hours=1)
    RESERVATION_TTL = timedelta(minutes=20)
    ACTION_TTL = timedelta(minutes=15)
    PAGE_SIZE = 8

    def __init__(
        self,
        db: Database,
        *,
        quota_policy: KnowledgeQuotaPolicy | None = None,
    ):
        self.db = db
        self.quota = quota_policy or KnowledgeQuotaPolicy()

    async def ensure_personal_space(self, actor_user_id: int) -> KnowledgeSpaceRecord:
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            space = await session.scalar(
                select(KnowledgeSpace).where(
                    KnowledgeSpace.kind == "personal",
                    KnowledgeSpace.personal_owner_user_id == actor_user_id,
                )
            )
            created = space is None
            if space is None:
                space = KnowledgeSpace(
                    public_id=str(uuid4()),
                    kind="personal",
                    personal_owner_user_id=actor_user_id,
                    status="active",
                    version=1,
                )
                session.add(space)
                await session.flush()
            await self._repair_space_public_id(session, space)
            if created:
                self._audit(
                    session,
                    "space.created",
                    actor_user_id=actor_user_id,
                    knowledge_space_id=space.id,
                    safe_metadata={"classification": "personal"},
                )
            access = self._context(actor_user_id, space, "owner", None, None)
            return KnowledgeSpaceRecord(access, "Личная база знаний", space.status)

    async def list_spaces(self, actor_user_id: int) -> tuple[KnowledgeSpaceRecord, ...]:
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            rows = (
                await session.execute(self._space_query(actor_user_id, SPACE_ROLES, True))
            ).all()
            records: list[KnowledgeSpaceRecord] = []
            for space, member_role, workspace, project in rows:
                await self._repair_space_public_id(session, space)
                role = "owner" if space.kind == "personal" else str(member_role)
                access = self._context(actor_user_id, space, role, workspace, project)
                if space.kind == "personal":
                    name = "Личная база знаний"
                elif space.kind == "workspace":
                    name = workspace.name
                else:
                    name = f"{workspace.name} / {project.name}"
                records.append(KnowledgeSpaceRecord(access, name, space.status))
            return tuple(records)

    async def resolve_space(
        self,
        actor_user_id: int,
        space_public_id: str,
        *,
        roles: frozenset[str] = SPACE_ROLES,
        require_active: bool = True,
    ) -> KnowledgeAccessContext:
        clean_id = self._public_id(space_public_id)
        async with self.db.session() as session:
            row = (
                await session.execute(
                    self._space_query(actor_user_id, roles, require_active).where(
                        KnowledgeSpace.public_id == clean_id
                    )
                )
            ).one_or_none()
            if row is None:
                raise KnowledgeAccessDenied("Область знаний недоступна.")
            space, member_role, workspace, project = row
            await self._repair_space_public_id(session, space)
            role = "owner" if space.kind == "personal" else str(member_role)
            return self._context(actor_user_id, space, role, workspace, project)

    async def begin_empty_capture(
        self,
        actor_user_id: int,
        chat_id: int,
        *,
        ttl: timedelta | None = None,
    ) -> KnowledgeCapturePreview:
        personal = await self.ensure_personal_space(actor_user_id)
        current = datetime.now(UTC)
        expires = current + self._ttl(ttl, self.CAPTURE_TTL, timedelta(days=1))
        async with self.db.session() as session:
            await self._lock_runtime(session)
            await self._lock_user(session, actor_user_id)
            existing, _ = await self._active_capture_row(
                session, actor_user_id, chat_id, current, expire=True
            )
            if existing is not None:
                return await self._capture_preview(session, existing)
            draft = KnowledgeCaptureDraft(
                public_id=str(uuid4()),
                actor_user_id=actor_user_id,
                chat_id=chat_id,
                capture_kind="text",
                knowledge_space_id=personal.access.knowledge_space_id,
                knowledge_space_version=personal.access.space_version,
                status="collecting",
                version=1,
                expires_at=expires,
            )
            session.add(draft)
            await session.flush()
            self._audit(
                session,
                "capture.started",
                actor_user_id=actor_user_id,
                knowledge_space_id=draft.knowledge_space_id,
                capture_draft_id=draft.id,
                safe_metadata={"capture_kind": "pending"},
            )
            return await self._capture_preview(session, draft)

    async def begin_capture(
        self,
        actor_user_id: int,
        chat_id: int,
        *,
        capture_kind: CaptureKind | str,
        text_content: str | None = None,
        source_url: str | None = None,
        telegram_file_id: str | None = None,
        telegram_file_unique_id_hash: str | None = None,
        telegram_message_id: int | None = None,
        declared_mime: str | None = None,
        safe_display_name: str | None = None,
        declared_size_bytes: int | None = None,
        provenance: dict[str, Any] | None = None,
        target_space_public_id: str | None = None,
        title: str | None = None,
        knowledge_role: KnowledgeRole | str = "trusted",
        priority: KnowledgePriority | str = "normal",
        system_classification: KnowledgeClassification | str = "general",
        user_classification: str | None = None,
        ttl: timedelta | None = None,
    ) -> KnowledgeCapturePreview:
        kind = self._capture_kind(capture_kind)
        payload = self._capture_payload(
            kind,
            text_content=text_content,
            source_url=source_url,
            telegram_file_id=telegram_file_id,
            telegram_file_unique_id_hash=telegram_file_unique_id_hash,
            declared_mime=declared_mime,
            safe_display_name=safe_display_name,
            declared_size_bytes=declared_size_bytes,
            provenance=provenance,
        )
        if target_space_public_id is None:
            target = (await self.ensure_personal_space(actor_user_id)).access
        else:
            target = await self.resolve_space(
                actor_user_id, target_space_public_id, roles=EDIT_ROLES
            )
        clean_classification = self._classification(system_classification)
        self._medical_scope(clean_classification, target.kind)
        clean_title = self._title(title or self._default_title(kind, payload))
        current = datetime.now(UTC)
        expires = current + self._ttl(ttl, self.CAPTURE_TTL, timedelta(days=1))
        async with self.db.session() as session:
            await self._lock_runtime(session)
            await self._lock_user(session, actor_user_id)
            existing, _ = await self._active_capture_row(
                session, actor_user_id, chat_id, current, expire=True
            )
            if existing is not None:
                raise KnowledgeConflictError("Сначала завершите текущий сбор материала.")
            draft = KnowledgeCaptureDraft(
                public_id=str(uuid4()),
                actor_user_id=actor_user_id,
                chat_id=chat_id,
                capture_kind=kind,
                telegram_message_id=telegram_message_id,
                knowledge_space_id=target.knowledge_space_id,
                knowledge_space_version=target.space_version,
                workspace_access_epoch=target.workspace_access_epoch,
                workspace_project_version=target.workspace_project_version,
                title=clean_title,
                knowledge_role=self._knowledge_role(knowledge_role),
                priority=self._priority(priority),
                system_classification=clean_classification,
                user_classification=self._user_classification(user_classification),
                status="awaiting_confirmation",
                version=1,
                expires_at=expires,
                **payload,
            )
            session.add(draft)
            await session.flush()
            self._audit(
                session,
                "capture.started",
                actor_user_id=actor_user_id,
                knowledge_space_id=target.knowledge_space_id,
                capture_draft_id=draft.id,
                safe_metadata={"capture_kind": kind},
            )
            return await self._capture_preview(session, draft)

    async def set_capture_payload(
        self,
        actor_user_id: int,
        chat_id: int,
        draft_public_id: str,
        expected_version: int,
        *,
        capture_kind: CaptureKind | str,
        text_content: str | None = None,
        source_url: str | None = None,
        telegram_file_id: str | None = None,
        telegram_file_unique_id_hash: str | None = None,
        telegram_message_id: int | None = None,
        declared_mime: str | None = None,
        safe_display_name: str | None = None,
        declared_size_bytes: int | None = None,
        provenance: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> KnowledgeCapturePreview:
        kind = self._capture_kind(capture_kind)
        payload = self._capture_payload(
            kind,
            text_content=text_content,
            source_url=source_url,
            telegram_file_id=telegram_file_id,
            telegram_file_unique_id_hash=telegram_file_unique_id_hash,
            declared_mime=declared_mime,
            safe_display_name=safe_display_name,
            declared_size_bytes=declared_size_bytes,
            provenance=provenance,
        )
        current = datetime.now(UTC)
        async with self.db.session() as session:
            await self._lock_runtime(session)
            draft = await self._lock_capture(
                session,
                actor_user_id,
                chat_id,
                draft_public_id,
                expected_version,
                statuses={"collecting"},
                current=current,
            )
            draft.capture_kind = kind
            draft.text_content = payload["text_content"]
            draft.source_url = payload["source_url"]
            draft.telegram_file_id = payload["telegram_file_id"]
            draft.telegram_file_unique_id_hash = payload["telegram_file_unique_id_hash"]
            draft.telegram_message_id = telegram_message_id
            draft.declared_mime = payload["declared_mime"]
            draft.safe_display_name = payload["safe_display_name"]
            draft.declared_size_bytes = payload["declared_size_bytes"]
            draft.provenance = payload["provenance"]
            draft.title = self._title(title or self._default_title(kind, payload))
            draft.status = "awaiting_confirmation"
            draft.version += 1
            await session.flush()
            return await self._capture_preview(session, draft)

    async def update_capture(
        self,
        actor_user_id: int,
        chat_id: int,
        draft_public_id: str,
        expected_version: int,
        *,
        target_space_public_id: str | None = None,
        title: str | None = None,
        knowledge_role: KnowledgeRole | str | None = None,
        priority: KnowledgePriority | str | None = None,
        system_classification: KnowledgeClassification | str | None = None,
        user_classification: str | None | object = _UNSET,
    ) -> KnowledgeCapturePreview:
        current = datetime.now(UTC)
        async with self.db.session() as session:
            await self._lock_runtime(session)
            draft = await self._lock_capture(
                session,
                actor_user_id,
                chat_id,
                draft_public_id,
                expected_version,
                statuses={"collecting", "awaiting_confirmation"},
                current=current,
            )
            target = await self._resolve_space_session(
                session,
                actor_user_id,
                draft.knowledge_space_id,
                EDIT_ROLES,
                True,
            )
            if target_space_public_id is not None:
                target = await self._resolve_space_public_session(
                    session,
                    actor_user_id,
                    target_space_public_id,
                    EDIT_ROLES,
                    True,
                )
            classification = (
                self._classification(system_classification)
                if system_classification is not None
                else draft.system_classification
            )
            self._medical_scope(classification, target.kind)
            draft.knowledge_space_id = target.knowledge_space_id
            draft.knowledge_space_version = target.space_version
            draft.workspace_access_epoch = target.workspace_access_epoch
            draft.workspace_project_version = target.workspace_project_version
            if title is not None:
                draft.title = self._title(title)
            if knowledge_role is not None:
                draft.knowledge_role = self._knowledge_role(knowledge_role)
            if priority is not None:
                draft.priority = self._priority(priority)
            draft.system_classification = classification
            if user_classification is not _UNSET:
                draft.user_classification = self._user_classification(user_classification)
            draft.version += 1
            await session.flush()
            return await self._capture_preview(session, draft)

    async def capture_state(
        self,
        actor_user_id: int,
        chat_id: int,
        *,
        now: datetime | None = None,
    ) -> KnowledgeCaptureState:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            draft, expired = await self._active_capture_row(
                session, actor_user_id, chat_id, current, expire=True
            )
            preview = await self._capture_preview(session, draft) if draft is not None else None
            return KnowledgeCaptureState(preview, expired)

    async def get_active_capture(
        self, actor_user_id: int, chat_id: int
    ) -> KnowledgeCapturePreview | None:
        return (await self.capture_state(actor_user_id, chat_id)).preview

    async def cancel_capture(
        self,
        actor_user_id: int,
        chat_id: int,
        draft_public_id: str,
        expected_version: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            draft = await self._lock_capture(
                session,
                actor_user_id,
                chat_id,
                draft_public_id,
                expected_version,
                statuses={"collecting", "awaiting_confirmation"},
                current=current,
            )
            draft.status = "cancelled"
            draft.completed_at = current
            draft.version += 1
            self._scrub_capture(draft)
            self._audit(
                session,
                "capture.cancelled",
                actor_user_id=actor_user_id,
                knowledge_space_id=draft.knowledge_space_id,
                capture_draft_id=draft.id,
            )
            return True

    async def reserve_capture(
        self,
        actor_user_id: int,
        chat_id: int,
        draft_public_id: str,
        expected_version: int,
        *,
        reserved_bytes: int,
        idempotency_key: str,
        ttl: timedelta | None = None,
        now: datetime | None = None,
    ) -> KnowledgeCaptureReservation:
        if not 0 <= reserved_bytes <= self.quota.max_source_bytes:
            raise KnowledgeQuotaError("Материал превышает допустимый размер.")
        clean_key = self._idempotency_key(idempotency_key)
        current = self._utc(now or datetime.now(UTC))
        expires = current + self._ttl(ttl, self.RESERVATION_TTL, timedelta(hours=2))
        async with self.db.session() as session:
            await self._lock_runtime(session)
            await self._lock_user(session, actor_user_id)
            existing = await session.scalar(
                select(KnowledgeQuotaReservation).where(
                    KnowledgeQuotaReservation.idempotency_key == clean_key,
                    KnowledgeQuotaReservation.actor_user_id == actor_user_id,
                )
            )
            if existing is not None:
                if existing.status == "reserved" and self._utc(existing.expires_at) > current:
                    draft = await session.get(KnowledgeCaptureDraft, existing.capture_draft_id)
                    if draft is None or draft.chat_id != chat_id:
                        raise KnowledgeCaptureError("Подтверждение недействительно.")
                    return KnowledgeCaptureReservation(
                        existing.public_id,
                        draft.public_id,
                        draft.version,
                        existing.reserved_bytes,
                        self._utc(existing.expires_at),
                        self._capture_material(draft),
                    )
                raise KnowledgeConflictError("Подтверждение уже было использовано.")
            draft = await self._lock_capture(
                session,
                actor_user_id,
                chat_id,
                draft_public_id,
                expected_version,
                statuses={"awaiting_confirmation"},
                current=current,
                lock_user=False,
            )
            if draft.declared_size_bytes is not None and draft.declared_size_bytes > reserved_bytes:
                raise KnowledgeQuotaError("Резерв меньше заявленного размера материала.")
            access = await self._resolve_space_session(
                session,
                actor_user_id,
                draft.knowledge_space_id,
                EDIT_ROLES,
                True,
            )
            self._capture_snapshot(draft, access)
            self._medical_scope(draft.system_classification, access.kind)
            await self._lock_space(session, access.knowledge_space_id)
            await self._expire_reservations(session, current)
            await self._check_quota(
                session,
                actor_user_id,
                access.knowledge_space_id,
                reserved_bytes,
                current,
            )
            reservation = KnowledgeQuotaReservation(
                public_id=str(uuid4()),
                idempotency_key=clean_key,
                actor_user_id=actor_user_id,
                knowledge_space_id=access.knowledge_space_id,
                capture_draft_id=draft.id,
                reserved_bytes=reserved_bytes,
                reserved_sources=1,
                reserved_jobs=1,
                status="reserved",
                expires_at=expires,
            )
            draft.status = "confirming"
            draft.version += 1
            session.add(reservation)
            await session.flush()
            return KnowledgeCaptureReservation(
                reservation.public_id,
                draft.public_id,
                draft.version,
                reservation.reserved_bytes,
                expires,
                self._capture_material(draft),
            )

    async def commit_capture(
        self,
        actor_user_id: int,
        chat_id: int,
        reservation_public_id: str,
        *,
        original: StoredKnowledgeOriginal,
        pipeline_version: str = "v1",
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> KnowledgeSourceReceipt:
        clean_reservation_id = self._public_id(reservation_public_id)
        clean_original = self._original(original)
        clean_pipeline = self._pipeline_version(pipeline_version)
        if not 1 <= max_attempts <= 20:
            raise KnowledgeJobError("Некорректный лимит попыток.")
        current = self._utc(now or datetime.now(UTC))
        try:
            async with self.db.session() as session:
                await self._lock_runtime(session)
                await self._lock_user(session, actor_user_id)
                reservation = await session.scalar(
                    select(KnowledgeQuotaReservation).where(
                        KnowledgeQuotaReservation.public_id == clean_reservation_id,
                        KnowledgeQuotaReservation.actor_user_id == actor_user_id,
                    )
                )
                if reservation is None:
                    raise KnowledgeCaptureError("Подтверждение недействительно.")
                if reservation.status == "committed":
                    return await self._receipt_for_reservation(session, reservation, chat_id)
                if reservation.status != "reserved" or self._utc(reservation.expires_at) <= current:
                    raise KnowledgeCaptureError("Время подтверждения истекло.")
                draft = await session.get(KnowledgeCaptureDraft, reservation.capture_draft_id)
                if (
                    draft is None
                    or draft.actor_user_id != actor_user_id
                    or draft.chat_id != chat_id
                    or draft.status != "confirming"
                    or draft.knowledge_space_id != reservation.knowledge_space_id
                ):
                    raise KnowledgeCaptureError("Подтверждение недействительно.")
                access = await self._resolve_space_session(
                    session,
                    actor_user_id,
                    reservation.knowledge_space_id,
                    EDIT_ROLES,
                    True,
                )
                self._capture_snapshot(draft, access)
                self._medical_scope(draft.system_classification, access.kind)
                await self._lock_space(session, access.knowledge_space_id)
                if clean_original.size_bytes > reservation.reserved_bytes:
                    raise KnowledgeQuotaError("Материал превысил зарезервированный размер.")
                source = KnowledgeSource(
                    public_id=str(uuid4()),
                    knowledge_space_id=access.knowledge_space_id,
                    space_kind=access.kind,
                    created_by_user_id=actor_user_id,
                    source_type=draft.capture_kind,
                    title=self._title(draft.title or "Материал"),
                    provenance_kind=self._provenance_kind(draft.capture_kind),
                    provenance=self._json_value(draft.provenance, maximum=4096),
                    processing_status=("partial" if draft.capture_kind == "url" else "queued"),
                    lifecycle_status="active",
                    knowledge_role=draft.knowledge_role,
                    priority=draft.priority,
                    publication_state="draft",
                    system_classification=draft.system_classification,
                    user_classification=draft.user_classification,
                    current_revision_number=1,
                    version=1,
                )
                session.add(source)
                await session.flush()
                revision = KnowledgeSourceRevision(
                    public_id=str(uuid4()),
                    source_id=source.id,
                    knowledge_space_id=source.knowledge_space_id,
                    revision_number=1,
                    sha256=clean_original.sha256,
                    original_storage_key=clean_original.storage_key,
                    declared_mime=clean_original.declared_mime,
                    detected_mime=clean_original.detected_mime,
                    detected_format=clean_original.detected_format,
                    safe_display_name=clean_original.safe_display_name,
                    size_bytes=clean_original.size_bytes,
                    extraction_status=("partial" if draft.capture_kind == "url" else "pending"),
                    provenance=self._json_value(clean_original.provenance, maximum=4096),
                    created_by_user_id=actor_user_id,
                    finalized_at=current if draft.capture_kind == "url" else None,
                )
                session.add(revision)
                await session.flush()
                job = KnowledgeIngestionJob(
                    public_id=str(uuid4()),
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                    revision_id=revision.id,
                    requested_by_user_id=actor_user_id,
                    job_type="extract",
                    status=("partial" if draft.capture_kind == "url" else "queued"),
                    attempt_count=0,
                    max_attempts=max_attempts,
                    available_at=current,
                    idempotency_key=f"extract:{revision.public_id}:{clean_pipeline}",
                    pipeline_version=clean_pipeline,
                    source_version=source.version,
                    version=1,
                    safe_error_code=(
                        "external_fetch_disabled" if draft.capture_kind == "url" else None
                    ),
                    finished_at=current if draft.capture_kind == "url" else None,
                )
                session.add(job)
                await session.flush()
                draft.status = "confirmed"
                draft.confirmed_source_id = source.id
                draft.completed_at = current
                draft.version += 1
                self._scrub_capture(draft)
                reservation.status = "committed"
                reservation.source_id = source.id
                reservation.revision_id = revision.id
                reservation.completed_at = current
                self._audit(
                    session,
                    "source.created",
                    actor_user_id=actor_user_id,
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                    safe_metadata={"source_type": source.source_type},
                )
                self._audit(
                    session,
                    "revision.created",
                    actor_user_id=actor_user_id,
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                    revision_id=revision.id,
                    safe_metadata={"revision_number": 1},
                )
                self._audit(
                    session,
                    "capture.confirmed",
                    actor_user_id=actor_user_id,
                    knowledge_space_id=source.knowledge_space_id,
                    capture_draft_id=draft.id,
                    source_id=source.id,
                )
                return KnowledgeSourceReceipt(
                    source.public_id,
                    revision.public_id,
                    job.public_id,
                    source.processing_status,
                    source.version,
                )
        except IntegrityError as exc:
            raise KnowledgeConflictError("Материал уже был подтверждён.") from exc

    async def release_capture_reservation(
        self,
        actor_user_id: int,
        chat_id: int,
        reservation_public_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now or datetime.now(UTC))
        clean_id = self._public_id(reservation_public_id)
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            reservation = await session.scalar(
                select(KnowledgeQuotaReservation).where(
                    KnowledgeQuotaReservation.public_id == clean_id,
                    KnowledgeQuotaReservation.actor_user_id == actor_user_id,
                    KnowledgeQuotaReservation.status == "reserved",
                )
            )
            if reservation is None:
                return False
            draft = await session.get(KnowledgeCaptureDraft, reservation.capture_draft_id)
            if draft is None or draft.chat_id != chat_id:
                raise KnowledgeCaptureError("Подтверждение недействительно.")
            reservation.status = "released"
            reservation.completed_at = current
            if draft.status == "confirming":
                draft.status = "awaiting_confirmation"
                draft.version += 1
            return True

    async def cleanup_expired(self, *, now: datetime | None = None) -> tuple[int, int, int]:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            captures = list(
                (
                    await session.scalars(
                        select(KnowledgeCaptureDraft).where(
                            KnowledgeCaptureDraft.status.in_(
                                ("collecting", "awaiting_confirmation")
                            ),
                            KnowledgeCaptureDraft.expires_at <= current,
                        )
                    )
                ).all()
            )
            for draft in captures:
                draft.status = "expired"
                draft.completed_at = current
                draft.version += 1
                self._scrub_capture(draft)
                self._audit(
                    session,
                    "capture.expired",
                    actor_user_id=draft.actor_user_id,
                    knowledge_space_id=draft.knowledge_space_id,
                    capture_draft_id=draft.id,
                )
            reservations = await self._expire_reservations(session, current)
            actions = await session.execute(
                delete(KnowledgeActionToken).where(
                    or_(
                        KnowledgeActionToken.expires_at <= current,
                        KnowledgeActionToken.status == "consumed",
                    )
                )
            )
            return len(captures), reservations, actions.rowcount or 0

    async def claim_next_job(
        self,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> ClaimedKnowledgeJob | None:
        clean_worker = self._bounded_key(worker_id, "worker", 64)
        if not 30 <= lease_seconds <= 3600:
            raise KnowledgeJobError("Некорректный срок аренды задания.")
        current = self._utc(now or datetime.now(UTC))
        lease_until = current + timedelta(seconds=lease_seconds)
        due = or_(
            and_(
                KnowledgeIngestionJob.status == "queued",
                KnowledgeIngestionJob.available_at <= current,
            ),
            and_(
                KnowledgeIngestionJob.status == "processing",
                KnowledgeIngestionJob.lease_expires_at <= current,
            ),
        )
        async with self.db.sessions() as read_session:
            candidates = list(
                (
                    await read_session.scalars(
                        select(KnowledgeIngestionJob.id)
                        .where(
                            due,
                            KnowledgeIngestionJob.attempt_count
                            < KnowledgeIngestionJob.max_attempts,
                        )
                        .order_by(
                            KnowledgeIngestionJob.available_at,
                            KnowledgeIngestionJob.id,
                        )
                        .limit(20)
                    )
                ).all()
            )
        for job_id in candidates:
            token = secrets.token_hex(24)
            async with self.db.session() as session:
                if not await self._runtime_available(session):
                    return None
                locked = await session.execute(
                    update(KnowledgeIngestionJob)
                    .where(
                        KnowledgeIngestionJob.id == job_id,
                        due,
                        KnowledgeIngestionJob.attempt_count < KnowledgeIngestionJob.max_attempts,
                    )
                    .values(
                        status="processing",
                        lease_owner=clean_worker,
                        lease_token=token,
                        lease_expires_at=lease_until,
                        heartbeat_at=current,
                        started_at=func.coalesce(KnowledgeIngestionJob.started_at, current),
                        attempt_count=KnowledgeIngestionJob.attempt_count + 1,
                        safe_error_code=None,
                        version=KnowledgeIngestionJob.version + 1,
                    )
                    .returning(KnowledgeIngestionJob.id)
                )
                if locked.scalar_one_or_none() is None:
                    continue
                row = (
                    await session.execute(
                        select(KnowledgeIngestionJob, KnowledgeSource, KnowledgeSourceRevision)
                        .join(
                            KnowledgeSource, KnowledgeSource.id == KnowledgeIngestionJob.source_id
                        )
                        .outerjoin(
                            KnowledgeSourceRevision,
                            KnowledgeSourceRevision.id == KnowledgeIngestionJob.revision_id,
                        )
                        .where(KnowledgeIngestionJob.id == job_id)
                    )
                ).one()
                job, source, revision = row
                valid_extract = (
                    job.job_type == "extract"
                    and revision is not None
                    and source.lifecycle_status == "active"
                    and source.version == job.source_version
                    and source.current_revision_number == revision.revision_number
                    and revision.extraction_status == "pending"
                )
                valid_purge = (
                    job.job_type == "purge"
                    and revision is None
                    and source.lifecycle_status == "purge_pending"
                    and source.version == job.source_version
                )
                if not (valid_extract or valid_purge):
                    job.status = "cancelled"
                    job.lease_owner = None
                    job.lease_token = None
                    job.lease_expires_at = None
                    job.heartbeat_at = None
                    job.finished_at = current
                    job.safe_error_code = None
                    job.version += 1
                    continue
                original_revision: KnowledgeSourceRevision | None = None
                if job.job_type == "extract":
                    source.processing_status = "processing"
                    asset_keys: tuple[str, ...] = ()
                    original_revision = revision
                    if revision is not None and revision.original_revision_id is not None:
                        original_revision = await session.get(
                            KnowledgeSourceRevision, revision.original_revision_id
                        )
                else:
                    asset_keys = tuple(
                        key
                        for original_key, extracted_key in (
                            await session.execute(
                                select(
                                    KnowledgeSourceRevision.original_storage_key,
                                    KnowledgeSourceRevision.extracted_storage_key,
                                ).where(KnowledgeSourceRevision.source_id == source.id)
                            )
                        ).all()
                        for key in (original_key, extracted_key)
                        if key is not None
                    )
                self._audit(
                    session,
                    "ingestion.status_changed",
                    actor_user_id=None,
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                    revision_id=revision.id if revision is not None else None,
                    job_id=job.id,
                    safe_metadata={"status": "processing", "job_type": job.job_type},
                )
                return ClaimedKnowledgeJob(
                    id=job.id,
                    public_id=job.public_id,
                    source_id=source.id,
                    source_public_id=source.public_id,
                    source_version=job.source_version,
                    revision_id=revision.id if revision is not None else None,
                    revision_number=revision.revision_number if revision is not None else None,
                    knowledge_space_id=source.knowledge_space_id,
                    job_type=job.job_type,
                    lease_token=token,
                    original_storage_key=(
                        original_revision.original_storage_key
                        if original_revision is not None
                        else None
                    ),
                    original_sha256=revision.sha256 if revision is not None else None,
                    declared_mime=revision.declared_mime if revision is not None else None,
                    detected_mime=revision.detected_mime if revision is not None else None,
                    detected_format=revision.detected_format if revision is not None else None,
                    size_bytes=revision.size_bytes if revision is not None else None,
                    attempt_count=job.attempt_count,
                    max_attempts=job.max_attempts,
                    cancel_requested=job.cancel_requested_at is not None,
                    asset_keys=asset_keys,
                )
        await self._fail_exhausted_jobs(current)
        return None

    async def heartbeat_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> bool:
        if not 30 <= lease_seconds <= 3600:
            raise KnowledgeJobError("Некорректный срок аренды задания.")
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            changed = await session.execute(
                update(KnowledgeIngestionJob)
                .where(
                    KnowledgeIngestionJob.id == job_id,
                    KnowledgeIngestionJob.status == "processing",
                    KnowledgeIngestionJob.lease_token == lease_token,
                    KnowledgeIngestionJob.lease_expires_at > current,
                    KnowledgeIngestionJob.cancel_requested_at.is_(None),
                )
                .values(
                    heartbeat_at=current,
                    lease_expires_at=current + timedelta(seconds=lease_seconds),
                    version=KnowledgeIngestionJob.version + 1,
                )
                .returning(KnowledgeIngestionJob.id)
            )
            return changed.scalar_one_or_none() is not None

    async def finalize_job(
        self,
        job_id: int,
        lease_token: str,
        result: KnowledgeExtractionResult,
        *,
        now: datetime | None = None,
    ) -> bool:
        clean_result = self._extraction_result(result)
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            job = await self._lock_claim(session, job_id, lease_token, current, "extract")
            if job is None:
                return False
            row = (
                await session.execute(
                    select(KnowledgeSource, KnowledgeSourceRevision)
                    .join(
                        KnowledgeSourceRevision,
                        KnowledgeSourceRevision.source_id == KnowledgeSource.id,
                    )
                    .where(
                        KnowledgeSource.id == job.source_id,
                        KnowledgeSource.version == job.source_version,
                        KnowledgeSource.lifecycle_status == "active",
                        KnowledgeSourceRevision.id == job.revision_id,
                        KnowledgeSourceRevision.revision_number
                        == KnowledgeSource.current_revision_number,
                        KnowledgeSourceRevision.extraction_status == "pending",
                    )
                )
            ).one_or_none()
            if row is None or job.cancel_requested_at is not None:
                await self._cancel_job_locked(session, job, current)
                # The runner must remove a just-created extracted asset because
                # no result was committed to the immutable revision.
                return False
            source, revision = row
            revision.extraction_status = clean_result.status
            revision.extracted_storage_key = clean_result.extracted_storage_key
            revision.extracted_sha256 = clean_result.extracted_sha256
            revision.extracted_size_bytes = clean_result.extracted_size_bytes
            revision.extraction_metadata = self._json_value(clean_result.metadata, maximum=8192)
            revision.finalized_at = current
            source.processing_status = clean_result.status
            self._finish_job(job, clean_result.status, current, clean_result.safe_error_code)
            self._audit(
                session,
                "ingestion.status_changed",
                actor_user_id=None,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                revision_id=revision.id,
                job_id=job.id,
                safe_metadata={"status": clean_result.status, "job_type": "extract"},
            )
            return True

    async def fail_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        failure_kind: FailureKind | str,
        safe_error_code: str,
        now: datetime | None = None,
        base_backoff_seconds: int = 5,
    ) -> bool:
        if failure_kind not in {"retryable", "permanent", "quarantine"}:
            raise KnowledgeJobError("Некорректный тип ошибки.")
        clean_code = self._safe_code(safe_error_code)
        if not 1 <= base_backoff_seconds <= 300:
            raise KnowledgeJobError("Некорректная задержка повтора.")
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            job = await self._lock_claim(session, job_id, lease_token, current, None)
            if job is None:
                return False
            source = await session.get(KnowledgeSource, job.source_id)
            if source is None or source.version != job.source_version:
                return await self._cancel_job_locked(session, job, current)
            retry = failure_kind == "retryable" and job.attempt_count < job.max_attempts
            if retry:
                backoff = min(
                    3600,
                    base_backoff_seconds * (2 ** min(job.attempt_count - 1, 8)),
                ) + secrets.randbelow(max(2, base_backoff_seconds + 1))
                job.status = "queued"
                job.available_at = current + timedelta(seconds=backoff)
                job.lease_owner = None
                job.lease_token = None
                job.lease_expires_at = None
                job.heartbeat_at = None
                job.safe_error_code = clean_code
                job.version += 1
                source.processing_status = "queued" if job.job_type == "extract" else "cancelled"
                terminal_status = "queued"
            else:
                terminal_status = "quarantined" if failure_kind == "quarantine" else "failed"
                self._finish_job(job, terminal_status, current, clean_code)
                if job.job_type == "extract":
                    revision = await session.get(KnowledgeSourceRevision, job.revision_id)
                    if revision is not None and revision.extraction_status == "pending":
                        revision.extraction_status = terminal_status
                        revision.finalized_at = current
                    source.processing_status = terminal_status
                else:
                    source.lifecycle_status = "purge_failed"
                    source.version += 1
            self._audit(
                session,
                "ingestion.status_changed",
                actor_user_id=None,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                revision_id=job.revision_id,
                job_id=job.id,
                safe_metadata={"status": terminal_status, "job_type": job.job_type},
            )
            if job.job_type == "purge" and not retry:
                self._audit(
                    session,
                    "source.purge_failed",
                    actor_user_id=None,
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                )
            return True

    async def cancel_claimed_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            job = await self._lock_claim(session, job_id, lease_token, current, None)
            if job is None:
                return False
            return await self._cancel_job_locked(session, job, current)

    async def finalize_purge_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            job = await self._lock_claim(session, job_id, lease_token, current, "purge")
            if job is None:
                return False
            source = await session.scalar(
                select(KnowledgeSource).where(
                    KnowledgeSource.id == job.source_id,
                    KnowledgeSource.version == job.source_version,
                    KnowledgeSource.lifecycle_status == "purge_pending",
                )
            )
            if source is None:
                return await self._cancel_job_locked(session, job, current)
            await session.execute(
                delete(KnowledgeIngestionJob).where(
                    KnowledgeIngestionJob.source_id == source.id,
                    KnowledgeIngestionJob.id != job.id,
                )
            )
            await session.execute(
                delete(KnowledgeSourceRevision).where(
                    KnowledgeSourceRevision.source_id == source.id
                )
            )
            source.lifecycle_status = "purged"
            source.current_revision_number = None
            source.processing_status = "cancelled"
            source.purged_at = current
            source.version += 1
            self._finish_job(job, "ready", current, None)
            self._audit(
                session,
                "source.purged",
                actor_user_id=None,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                job_id=job.id,
            )
            return True

    async def list_sources(
        self,
        actor_user_id: int,
        space_public_id: str,
        *,
        lifecycle_status: str = "active",
        page: int = 1,
        page_size: int = PAGE_SIZE,
    ) -> KnowledgeSourcePage:
        if lifecycle_status not in SOURCE_LIFECYCLES - {"purged"}:
            raise KnowledgeError("Некорректный раздел материалов.")
        page, page_size = self._page(page, page_size)
        clean_space_public_id = self._public_id(space_public_id)
        async with self.db.sessions() as session:
            await self._resolve_space_public_session(
                session,
                actor_user_id,
                clean_space_public_id,
                SPACE_ROLES,
                True,
            )
            lifecycle_scope = (
                KnowledgeSource.lifecycle_status.in_(("trashed", "purge_failed"))
                if lifecycle_status == "trashed"
                else KnowledgeSource.lifecycle_status == lifecycle_status
            )
            scoped_sources = self._source_query(actor_user_id, SPACE_ROLES).where(
                KnowledgeSpace.public_id == clean_space_public_id,
                lifecycle_scope,
            )
            scoped_ids = scoped_sources.with_only_columns(KnowledgeSource.id).order_by(None)
            total = int(
                await session.scalar(select(func.count()).select_from(scoped_ids.subquery())) or 0
            )
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            source_rows = list(
                (
                    await session.execute(
                        scoped_sources.order_by(
                            KnowledgeSource.updated_at.desc(), KnowledgeSource.id.desc()
                        )
                        .offset((page - 1) * page_size)
                        .limit(page_size)
                    )
                ).all()
            )
            record_parts: list[
                tuple[
                    KnowledgeSource,
                    KnowledgeSourceRevision | None,
                    KnowledgeIngestionJob | None,
                ]
            ] = []
            for source, _space, _member_role, _workspace, _project in source_rows:
                revision = await session.scalar(
                    select(KnowledgeSourceRevision).where(
                        KnowledgeSourceRevision.source_id == source.id,
                        KnowledgeSourceRevision.revision_number == source.current_revision_number,
                    )
                )
                job = await session.scalar(
                    select(KnowledgeIngestionJob)
                    .where(KnowledgeIngestionJob.source_id == source.id)
                    .order_by(KnowledgeIngestionJob.id.desc())
                    .limit(1)
                )
                record_parts.append((source, revision, job))
            access = await self._resolve_space_public_session(
                session,
                actor_user_id,
                clean_space_public_id,
                SPACE_ROLES,
                True,
            )
            records = tuple(
                KnowledgeSourceRecord(source, revision, job, access.role)
                for source, revision, job in record_parts
            )
            return KnowledgeSourcePage(records, page, pages, total)

    async def get_source(
        self,
        actor_user_id: int,
        source_public_id: str,
        *,
        include_trashed: bool = False,
    ) -> KnowledgeSourceRecord:
        async with self.db.sessions() as session:
            source, access = await self._source_access_row(
                session,
                actor_user_id,
                source_public_id,
                SPACE_ROLES,
                include_trashed=include_trashed,
            )
            revision = await session.scalar(
                select(KnowledgeSourceRevision).where(
                    KnowledgeSourceRevision.source_id == source.id,
                    KnowledgeSourceRevision.revision_number == source.current_revision_number,
                )
            )
            job = await session.scalar(
                select(KnowledgeIngestionJob)
                .where(KnowledgeIngestionJob.source_id == source.id)
                .order_by(KnowledgeIngestionJob.id.desc())
                .limit(1)
            )
            return KnowledgeSourceRecord(source, revision, job, access.role)

    async def find_duplicate(
        self,
        actor_user_id: int,
        space_public_id: str,
        sha256: str,
    ) -> KnowledgeSourceRecord | None:
        clean_sha = self._sha256(sha256)
        clean_space_public_id = self._public_id(space_public_id)
        async with self.db.sessions() as session:
            await self._resolve_space_public_session(
                session,
                actor_user_id,
                clean_space_public_id,
                SPACE_ROLES,
                True,
            )
            row = (
                await session.execute(
                    self._source_query(actor_user_id, SPACE_ROLES)
                    .join(
                        KnowledgeSourceRevision,
                        and_(
                            KnowledgeSourceRevision.source_id == KnowledgeSource.id,
                            KnowledgeSourceRevision.knowledge_space_id == KnowledgeSpace.id,
                        ),
                    )
                    .add_columns(KnowledgeSourceRevision)
                    .where(
                        KnowledgeSpace.public_id == clean_space_public_id,
                        KnowledgeSource.lifecycle_status != "purged",
                        KnowledgeSourceRevision.sha256 == clean_sha,
                    )
                    .order_by(KnowledgeSourceRevision.id.desc())
                    .limit(1)
                )
            ).one_or_none()
            if row is None:
                await self._resolve_space_public_session(
                    session,
                    actor_user_id,
                    clean_space_public_id,
                    SPACE_ROLES,
                    True,
                )
                return None
            source, _space, _member_role, _workspace, _project, revision = row
            job = await session.scalar(
                select(KnowledgeIngestionJob)
                .where(KnowledgeIngestionJob.source_id == source.id)
                .order_by(KnowledgeIngestionJob.id.desc())
                .limit(1)
            )
            access = await self._resolve_space_public_session(
                session,
                actor_user_id,
                clean_space_public_id,
                SPACE_ROLES,
                True,
            )
            return KnowledgeSourceRecord(source, revision, job, access.role)

    async def trash_source(
        self,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
        *,
        now: datetime | None = None,
    ) -> KnowledgeSource:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            source, access = await self._lock_source_access(
                session,
                actor_user_id,
                source_public_id,
                expected_version,
                EDIT_ROLES,
                lifecycles={"active"},
            )
            source.lifecycle_status = "trashed"
            source.trashed_at = current
            source.trashed_by_user_id = actor_user_id
            source.processing_status = "cancelled"
            source.version += 1
            await self._cancel_open_jobs(session, source.id, current)
            self._audit(
                session,
                "source.trashed",
                actor_user_id=actor_user_id,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                workspace_id=access.workspace_id,
            )
            return source

    async def restore_source(
        self,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
    ) -> KnowledgeSource:
        async with self.db.session() as session:
            source, access = await self._lock_source_access(
                session,
                actor_user_id,
                source_public_id,
                expected_version,
                EDIT_ROLES,
                lifecycles={"trashed"},
            )
            source.lifecycle_status = "active"
            source.trashed_at = None
            source.trashed_by_user_id = None
            source.version += 1
            self._audit(
                session,
                "source.restored",
                actor_user_id=actor_user_id,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                workspace_id=access.workspace_id,
            )
            return source

    async def request_permanent_delete(
        self,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
        *,
        pipeline_version: str = "v1",
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> KnowledgeIngestionJob:
        clean_pipeline = self._pipeline_version(pipeline_version)
        if not 1 <= max_attempts <= 20:
            raise KnowledgeJobError("Некорректный лимит попыток.")
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_runtime(session)
            source, access = await self._lock_source_access(
                session,
                actor_user_id,
                source_public_id,
                expected_version,
                OWNER_ROLES,
                lifecycles={"trashed", "purge_failed"},
            )
            await self._cancel_open_jobs(session, source.id, current)
            source.lifecycle_status = "purge_pending"
            source.purge_requested_at = current
            source.version += 1
            job = KnowledgeIngestionJob(
                public_id=str(uuid4()),
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                revision_id=None,
                requested_by_user_id=actor_user_id,
                job_type="purge",
                status="queued",
                attempt_count=0,
                max_attempts=max_attempts,
                available_at=current,
                idempotency_key=f"purge:{source.public_id}:v{source.version}",
                pipeline_version=clean_pipeline,
                source_version=source.version,
                version=1,
            )
            session.add(job)
            await session.flush()
            self._audit(
                session,
                "source.purge_requested",
                actor_user_id=actor_user_id,
                workspace_id=access.workspace_id,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                job_id=job.id,
            )
            return job

    async def cancel_source_job(
        self,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            source, _ = await self._lock_source_access(
                session,
                actor_user_id,
                source_public_id,
                expected_version,
                EDIT_ROLES,
                lifecycles={"active"},
            )
            job = await session.scalar(
                select(KnowledgeIngestionJob)
                .where(
                    KnowledgeIngestionJob.source_id == source.id,
                    KnowledgeIngestionJob.status.in_(("queued", "processing")),
                )
                .order_by(KnowledgeIngestionJob.id.desc())
                .limit(1)
            )
            if job is None:
                return False
            if job.status == "queued":
                self._finish_job(job, "cancelled", current, None)
                revision = await session.get(KnowledgeSourceRevision, job.revision_id)
                if revision is not None and revision.extraction_status == "pending":
                    revision.extraction_status = "cancelled"
                    revision.finalized_at = current
                source.processing_status = "cancelled"
            else:
                job.cancel_requested_at = current
                job.version += 1
            return True

    async def retry_source(
        self,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
        *,
        pipeline_version: str = "v1",
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> KnowledgeIngestionJob:
        clean_pipeline = self._pipeline_version(pipeline_version)
        if not 1 <= max_attempts <= 20:
            raise KnowledgeJobError("Некорректный лимит попыток.")
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_runtime(session)
            source, _ = await self._lock_source_access(
                session,
                actor_user_id,
                source_public_id,
                expected_version,
                EDIT_ROLES,
                lifecycles={"active"},
            )
            if source.processing_status not in {"failed", "cancelled"}:
                raise KnowledgeConflictError("Этот материал нельзя повторить.")
            revision = await session.scalar(
                select(KnowledgeSourceRevision).where(
                    KnowledgeSourceRevision.source_id == source.id,
                    KnowledgeSourceRevision.revision_number == source.current_revision_number,
                )
            )
            if revision is None or revision.extraction_status not in {"failed", "cancelled"}:
                raise KnowledgeConflictError("Эту ревизию нельзя повторить.")
            original_revision = (
                await session.get(KnowledgeSourceRevision, revision.original_revision_id)
                if revision.original_revision_id is not None
                else revision
            )
            if (
                original_revision is None
                or original_revision.source_id != source.id
                or original_revision.knowledge_space_id != source.knowledge_space_id
                or original_revision.original_storage_key is None
            ):
                raise KnowledgeConflictError("Оригинал этой ревизии недоступен.")
            await self._lock_space(session, source.knowledge_space_id)
            await self._check_quota(
                session,
                actor_user_id,
                source.knowledge_space_id,
                0,
                current,
                new_sources=0,
                new_extract_jobs=1,
            )
            next_number = source.current_revision_number + 1
            source.processing_status = "queued"
            source.current_revision_number = next_number
            source.version += 1
            retry_revision = KnowledgeSourceRevision(
                public_id=str(uuid4()),
                source_id=source.id,
                knowledge_space_id=source.knowledge_space_id,
                revision_number=next_number,
                sha256=original_revision.sha256,
                original_revision_id=original_revision.id,
                original_storage_key=None,
                declared_mime=original_revision.declared_mime,
                detected_mime=original_revision.detected_mime,
                detected_format=original_revision.detected_format,
                safe_display_name=original_revision.safe_display_name,
                size_bytes=original_revision.size_bytes,
                extraction_status="pending",
                provenance=original_revision.provenance,
                created_by_user_id=actor_user_id,
            )
            session.add(retry_revision)
            await session.flush()
            job = KnowledgeIngestionJob(
                public_id=str(uuid4()),
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                revision_id=retry_revision.id,
                requested_by_user_id=actor_user_id,
                job_type="extract",
                status="queued",
                attempt_count=0,
                max_attempts=max_attempts,
                available_at=current,
                idempotency_key=(
                    f"retry:{retry_revision.public_id}:{clean_pipeline}:v{source.version}"
                ),
                pipeline_version=clean_pipeline,
                source_version=source.version,
                version=1,
            )
            session.add(job)
            await session.flush()
            self._audit(
                session,
                "revision.created",
                actor_user_id=actor_user_id,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
                revision_id=retry_revision.id,
                job_id=job.id,
                safe_metadata={"revision_number": next_number},
            )
            return job

    async def update_source_classification(
        self,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
        *,
        knowledge_role: KnowledgeRole | str | None = None,
        priority: KnowledgePriority | str | None = None,
        publication_state: str | None = None,
        system_classification: KnowledgeClassification | str | None = None,
        user_classification: str | None | object = _UNSET,
    ) -> KnowledgeSource:
        async with self.db.session() as session:
            source, access = await self._lock_source_access(
                session,
                actor_user_id,
                source_public_id,
                expected_version,
                EDIT_ROLES,
                lifecycles={"active"},
            )
            classification = (
                self._classification(system_classification)
                if system_classification is not None
                else source.system_classification
            )
            self._medical_scope(classification, access.kind)
            if knowledge_role is not None:
                source.knowledge_role = self._knowledge_role(knowledge_role)
            if priority is not None:
                source.priority = self._priority(priority)
            if publication_state is not None:
                if publication_state not in {"draft", "publication_ready"}:
                    raise KnowledgeError("Некорректный статус публикации.")
                source.publication_state = publication_state
            if classification == "health_private" and source.publication_state != "draft":
                raise KnowledgeAccessDenied("Медицинский материал нельзя подготовить к публикации.")
            source.system_classification = classification
            if user_classification is not _UNSET:
                source.user_classification = self._user_classification(user_classification)
            source.version += 1
            self._audit(
                session,
                "source.classification_changed",
                actor_user_id=actor_user_id,
                knowledge_space_id=source.knowledge_space_id,
                source_id=source.id,
            )
            return source

    async def append_revision(
        self,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
        *,
        original: StoredKnowledgeOriginal,
        idempotency_key: str,
        pipeline_version: str = "v1",
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> KnowledgeSourceReceipt:
        clean_original = self._original(original)
        clean_key = self._idempotency_key(idempotency_key)
        clean_pipeline = self._pipeline_version(pipeline_version)
        if not 1 <= max_attempts <= 20:
            raise KnowledgeJobError("Некорректный лимит попыток.")
        current = self._utc(now or datetime.now(UTC))
        try:
            async with self.db.session() as session:
                await self._lock_runtime(session)
                source, access = await self._lock_source_access(
                    session,
                    actor_user_id,
                    source_public_id,
                    expected_version,
                    EDIT_ROLES,
                    lifecycles={"active"},
                )
                await self._lock_space(session, source.knowledge_space_id)
                existing_job = await session.scalar(
                    select(KnowledgeIngestionJob).where(
                        KnowledgeIngestionJob.idempotency_key == clean_key,
                        KnowledgeIngestionJob.source_id == source.id,
                    )
                )
                if existing_job is not None:
                    revision = await session.get(KnowledgeSourceRevision, existing_job.revision_id)
                    if revision is None:
                        raise KnowledgeConflictError("Ревизия недоступна.")
                    return KnowledgeSourceReceipt(
                        source.public_id,
                        revision.public_id,
                        existing_job.public_id,
                        source.processing_status,
                        source.version,
                    )
                await self._check_quota(
                    session,
                    actor_user_id,
                    source.knowledge_space_id,
                    clean_original.size_bytes,
                    current,
                )
                next_number = (source.current_revision_number or 0) + 1
                source.current_revision_number = next_number
                source.processing_status = "queued"
                source.version += 1
                revision = KnowledgeSourceRevision(
                    public_id=str(uuid4()),
                    source_id=source.id,
                    knowledge_space_id=source.knowledge_space_id,
                    revision_number=next_number,
                    sha256=clean_original.sha256,
                    original_storage_key=clean_original.storage_key,
                    declared_mime=clean_original.declared_mime,
                    detected_mime=clean_original.detected_mime,
                    detected_format=clean_original.detected_format,
                    safe_display_name=clean_original.safe_display_name,
                    size_bytes=clean_original.size_bytes,
                    extraction_status="pending",
                    provenance=self._json_value(clean_original.provenance, maximum=4096),
                    created_by_user_id=actor_user_id,
                )
                session.add(revision)
                await session.flush()
                job = KnowledgeIngestionJob(
                    public_id=str(uuid4()),
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                    revision_id=revision.id,
                    requested_by_user_id=actor_user_id,
                    job_type="extract",
                    status="queued",
                    attempt_count=0,
                    max_attempts=max_attempts,
                    available_at=current,
                    idempotency_key=clean_key,
                    pipeline_version=clean_pipeline,
                    source_version=source.version,
                    version=1,
                )
                session.add(job)
                await session.flush()
                self._audit(
                    session,
                    "revision.created",
                    actor_user_id=actor_user_id,
                    workspace_id=access.workspace_id,
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                    revision_id=revision.id,
                    job_id=job.id,
                    safe_metadata={"revision_number": next_number},
                )
                return KnowledgeSourceReceipt(
                    source.public_id,
                    revision.public_id,
                    job.public_id,
                    source.processing_status,
                    source.version,
                )
        except IntegrityError as exc:
            raise KnowledgeConflictError("Ревизия уже была создана.") from exc

    async def issue_action(
        self,
        actor_user_id: int,
        chat_id: int,
        action: str,
        space_public_id: str,
        *,
        capture_draft_public_id: str | None = None,
        source_public_id: str | None = None,
        payload: dict[str, Any] | None = None,
        status: Literal["pending", "awaiting_input"] = "pending",
        ttl: timedelta | None = None,
    ) -> IssuedKnowledgeAction:
        clean_action = self._action(action)
        if status not in {"pending", "awaiting_input"}:
            raise KnowledgeError("Некорректный статус действия.")
        if capture_draft_public_id is not None and source_public_id is not None:
            raise KnowledgeError("Некорректная область действия.")
        current = datetime.now(UTC)
        expires = current + self._ttl(ttl, self.ACTION_TTL, timedelta(hours=24))
        raw_token = secrets.token_urlsafe(24)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            access = await self._resolve_space_public_session(
                session, actor_user_id, space_public_id, SPACE_ROLES, True
            )
            capture: KnowledgeCaptureDraft | None = None
            source: KnowledgeSource | None = None
            if capture_draft_public_id is not None:
                capture = await session.scalar(
                    select(KnowledgeCaptureDraft).where(
                        KnowledgeCaptureDraft.public_id == self._public_id(capture_draft_public_id),
                        KnowledgeCaptureDraft.actor_user_id == actor_user_id,
                        KnowledgeCaptureDraft.chat_id == chat_id,
                        KnowledgeCaptureDraft.knowledge_space_id == access.knowledge_space_id,
                        KnowledgeCaptureDraft.status.in_(("collecting", "awaiting_confirmation")),
                    )
                )
                if capture is None:
                    raise KnowledgeAccessDenied("Действие недоступно.")
                scope_kind = "capture"
            elif source_public_id is not None:
                source, _ = await self._source_access_row(
                    session,
                    actor_user_id,
                    source_public_id,
                    SPACE_ROLES,
                    include_trashed=True,
                )
                if source.knowledge_space_id != access.knowledge_space_id:
                    raise KnowledgeAccessDenied("Действие недоступно.")
                scope_kind = "source"
            else:
                scope_kind = "space"
            if status == "awaiting_input":
                await session.execute(
                    update(KnowledgeActionToken)
                    .where(
                        KnowledgeActionToken.actor_user_id == actor_user_id,
                        KnowledgeActionToken.chat_id == chat_id,
                        KnowledgeActionToken.status == "awaiting_input",
                    )
                    .values(status="consumed", consumed_at=current)
                )
            session.add(
                KnowledgeActionToken(
                    token_hash=token_hash,
                    actor_user_id=actor_user_id,
                    chat_id=chat_id,
                    scope_kind=scope_kind,
                    knowledge_space_id=access.knowledge_space_id,
                    knowledge_space_version=access.space_version,
                    workspace_access_epoch=access.workspace_access_epoch,
                    capture_draft_id=capture.id if capture is not None else None,
                    capture_version=capture.version if capture is not None else None,
                    source_id=source.id if source is not None else None,
                    source_version=source.version if source is not None else None,
                    action=clean_action,
                    payload=self._json_value(payload, maximum=2048),
                    status=status,
                    expires_at=expires,
                )
            )
            return IssuedKnowledgeAction(raw_token, expires)

    async def claim_action(
        self,
        token: str,
        actor_user_id: int,
        chat_id: int,
        *,
        expected_action: str | None = None,
        now: datetime | None = None,
    ) -> ClaimedKnowledgeAction | None:
        if len(token) > 256:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            row = await session.scalar(
                select(KnowledgeActionToken).where(
                    KnowledgeActionToken.token_hash == token_hash,
                    KnowledgeActionToken.actor_user_id == actor_user_id,
                    KnowledgeActionToken.chat_id == chat_id,
                    KnowledgeActionToken.status == "pending",
                    KnowledgeActionToken.expires_at > current,
                )
            )
            if row is None:
                return None
            if expected_action is not None and row.action != self._action(expected_action):
                return None
            claim = await self._validate_action(session, row, actor_user_id)
            if claim is None:
                return None
            changed = await session.execute(
                update(KnowledgeActionToken)
                .where(
                    KnowledgeActionToken.token_hash == token_hash,
                    KnowledgeActionToken.status == "pending",
                    KnowledgeActionToken.expires_at > current,
                )
                .values(status="consumed", consumed_at=current)
                .returning(KnowledgeActionToken.token_hash)
                .execution_options(synchronize_session=False)
            )
            return claim if changed.scalar_one_or_none() is not None else None

    async def pending_input(
        self,
        actor_user_id: int,
        chat_id: int,
        *,
        now: datetime | None = None,
    ) -> ClaimedKnowledgeAction | None:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.sessions() as session:
            row = await session.scalar(
                select(KnowledgeActionToken)
                .where(
                    KnowledgeActionToken.actor_user_id == actor_user_id,
                    KnowledgeActionToken.chat_id == chat_id,
                    KnowledgeActionToken.status == "awaiting_input",
                    KnowledgeActionToken.expires_at > current,
                )
                .order_by(KnowledgeActionToken.created_at.desc())
                .limit(1)
            )
            return await self._validate_action(session, row, actor_user_id) if row else None

    async def claim_pending_input(
        self,
        actor_user_id: int,
        chat_id: int,
        *,
        now: datetime | None = None,
    ) -> ClaimedKnowledgeAction | None:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            row = await session.scalar(
                select(KnowledgeActionToken)
                .where(
                    KnowledgeActionToken.actor_user_id == actor_user_id,
                    KnowledgeActionToken.chat_id == chat_id,
                    KnowledgeActionToken.status == "awaiting_input",
                    KnowledgeActionToken.expires_at > current,
                )
                .order_by(KnowledgeActionToken.created_at.desc())
                .limit(1)
            )
            if row is None:
                return None
            claim = await self._validate_action(session, row, actor_user_id)
            if claim is None:
                return None
            changed = await session.execute(
                update(KnowledgeActionToken)
                .where(
                    KnowledgeActionToken.token_hash == row.token_hash,
                    KnowledgeActionToken.status == "awaiting_input",
                    KnowledgeActionToken.expires_at > current,
                )
                .values(status="consumed", consumed_at=current)
                .returning(KnowledgeActionToken.token_hash)
                .execution_options(synchronize_session=False)
            )
            return claim if changed.scalar_one_or_none() is not None else None

    async def cancel_pending_input(
        self,
        actor_user_id: int,
        chat_id: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            changed = await session.execute(
                update(KnowledgeActionToken)
                .where(
                    KnowledgeActionToken.actor_user_id == actor_user_id,
                    KnowledgeActionToken.chat_id == chat_id,
                    KnowledgeActionToken.status == "awaiting_input",
                )
                .values(status="consumed", consumed_at=current)
            )
            return bool(changed.rowcount)

    def _space_query(
        self,
        actor_user_id: int,
        roles: frozenset[str],
        require_active: bool,
    ):
        personal = and_(
            KnowledgeSpace.kind == "personal",
            KnowledgeSpace.personal_owner_user_id == actor_user_id,
        )
        if "owner" not in roles:
            personal = and_(personal, False)
        shared = and_(
            KnowledgeSpace.kind.in_(("workspace", "project")),
            Workspace.id == KnowledgeSpace.workspace_id,
            WorkspaceMember.workspace_id == Workspace.id,
            WorkspaceMember.user_id == actor_user_id,
            WorkspaceMember.status == "active",
            WorkspaceMember.role.in_(roles),
        )
        project = and_(
            shared,
            KnowledgeSpace.kind == "project",
            WorkspaceProject.id == KnowledgeSpace.workspace_project_id,
            WorkspaceProject.workspace_id == Workspace.id,
        )
        workspace = and_(shared, KnowledgeSpace.kind == "workspace")
        if require_active:
            personal = and_(personal, KnowledgeSpace.status == "active")
            workspace = and_(
                workspace,
                KnowledgeSpace.status == "active",
                Workspace.status == "active",
            )
            project = and_(
                project,
                KnowledgeSpace.status == "active",
                Workspace.status == "active",
                WorkspaceProject.status == "active",
            )
        return (
            select(
                KnowledgeSpace,
                WorkspaceMember.role,
                Workspace,
                WorkspaceProject,
            )
            .select_from(KnowledgeSpace)
            .outerjoin(Workspace, Workspace.id == KnowledgeSpace.workspace_id)
            .outerjoin(
                WorkspaceMember,
                and_(
                    WorkspaceMember.workspace_id == Workspace.id,
                    WorkspaceMember.user_id == actor_user_id,
                ),
            )
            .outerjoin(
                WorkspaceProject,
                and_(
                    WorkspaceProject.id == KnowledgeSpace.workspace_project_id,
                    WorkspaceProject.workspace_id == Workspace.id,
                ),
            )
            .where(or_(personal, workspace, project))
        )

    async def _resolve_space_session(
        self,
        session: AsyncSession,
        actor_user_id: int,
        knowledge_space_id: int | None,
        roles: frozenset[str],
        require_active: bool,
    ) -> KnowledgeAccessContext:
        if knowledge_space_id is None:
            raise KnowledgeAccessDenied("Область знаний недоступна.")
        row = (
            await session.execute(
                self._space_query(actor_user_id, roles, require_active).where(
                    KnowledgeSpace.id == knowledge_space_id
                )
            )
        ).one_or_none()
        if row is None:
            raise KnowledgeAccessDenied("Область знаний недоступна.")
        space, member_role, workspace, project = row
        await self._repair_space_public_id(session, space)
        role = "owner" if space.kind == "personal" else str(member_role)
        return self._context(actor_user_id, space, role, workspace, project)

    async def _resolve_space_public_session(
        self,
        session: AsyncSession,
        actor_user_id: int,
        space_public_id: str,
        roles: frozenset[str],
        require_active: bool,
    ) -> KnowledgeAccessContext:
        row = (
            await session.execute(
                self._space_query(actor_user_id, roles, require_active).where(
                    KnowledgeSpace.public_id == self._public_id(space_public_id)
                )
            )
        ).one_or_none()
        if row is None:
            raise KnowledgeAccessDenied("Область знаний недоступна.")
        space, member_role, workspace, project = row
        await self._repair_space_public_id(session, space)
        role = "owner" if space.kind == "personal" else str(member_role)
        return self._context(actor_user_id, space, role, workspace, project)

    @staticmethod
    def _context(
        actor_user_id: int,
        space: KnowledgeSpace,
        role: str,
        workspace: Workspace | None,
        project: WorkspaceProject | None,
    ) -> KnowledgeAccessContext:
        if space.public_id is None:
            raise KnowledgeAccessDenied("Область знаний недоступна.")
        return KnowledgeAccessContext(
            actor_user_id=actor_user_id,
            knowledge_space_id=space.id,
            space_public_id=space.public_id,
            kind=space.kind,
            role=role,
            space_version=space.version,
            workspace_id=workspace.id if workspace is not None else None,
            workspace_access_epoch=(workspace.access_epoch if workspace is not None else None),
            workspace_project_id=project.id if project is not None else None,
            workspace_project_version=project.version if project is not None else None,
        )

    async def _repair_space_public_id(self, session: AsyncSession, space: KnowledgeSpace) -> None:
        if space.public_id is not None:
            return
        for _ in range(3):
            candidate = str(uuid4())
            changed = await session.execute(
                update(KnowledgeSpace)
                .where(KnowledgeSpace.id == space.id, KnowledgeSpace.public_id.is_(None))
                .values(public_id=candidate)
                .returning(KnowledgeSpace.public_id)
            )
            value = changed.scalar_one_or_none()
            if value is not None:
                space.public_id = value
                return
            value = await session.scalar(
                select(KnowledgeSpace.public_id).where(KnowledgeSpace.id == space.id)
            )
            if value is not None:
                space.public_id = value
                return
        raise KnowledgeConflictError("Не удалось назначить идентификатор области.")

    def _source_query(
        self,
        actor_user_id: int,
        roles: frozenset[str],
    ):
        personal = and_(
            KnowledgeSpace.kind == "personal",
            KnowledgeSpace.personal_owner_user_id == actor_user_id,
        )
        if "owner" not in roles:
            personal = and_(personal, False)
        shared_base = and_(
            Workspace.id == KnowledgeSpace.workspace_id,
            Workspace.status == "active",
            WorkspaceMember.workspace_id == Workspace.id,
            WorkspaceMember.user_id == actor_user_id,
            WorkspaceMember.status == "active",
            WorkspaceMember.role.in_(roles),
        )
        workspace = and_(KnowledgeSpace.kind == "workspace", shared_base)
        project = and_(
            KnowledgeSpace.kind == "project",
            shared_base,
            WorkspaceProject.id == KnowledgeSpace.workspace_project_id,
            WorkspaceProject.workspace_id == Workspace.id,
            WorkspaceProject.status == "active",
        )
        return (
            select(
                KnowledgeSource,
                KnowledgeSpace,
                WorkspaceMember.role,
                Workspace,
                WorkspaceProject,
            )
            .join(KnowledgeSpace, KnowledgeSpace.id == KnowledgeSource.knowledge_space_id)
            .outerjoin(Workspace, Workspace.id == KnowledgeSpace.workspace_id)
            .outerjoin(
                WorkspaceMember,
                and_(
                    WorkspaceMember.workspace_id == Workspace.id,
                    WorkspaceMember.user_id == actor_user_id,
                ),
            )
            .outerjoin(
                WorkspaceProject,
                and_(
                    WorkspaceProject.id == KnowledgeSpace.workspace_project_id,
                    WorkspaceProject.workspace_id == Workspace.id,
                ),
            )
            .where(
                KnowledgeSpace.status == "active",
                or_(personal, workspace, project),
            )
        )

    async def _source_access_row(
        self,
        session: AsyncSession,
        actor_user_id: int,
        source_public_id: str,
        roles: frozenset[str],
        *,
        include_trashed: bool,
    ) -> tuple[KnowledgeSource, KnowledgeAccessContext]:
        query = self._source_query(actor_user_id, roles).where(
            KnowledgeSource.public_id == self._public_id(source_public_id),
            KnowledgeSource.lifecycle_status != "purged",
        )
        if not include_trashed:
            query = query.where(KnowledgeSource.lifecycle_status == "active")
        row = (await session.execute(query)).one_or_none()
        if row is None:
            raise KnowledgeAccessDenied("Материал не найден.")
        source, space, member_role, workspace, project = row
        await self._repair_space_public_id(session, space)
        role = "owner" if space.kind == "personal" else str(member_role)
        context = self._context(actor_user_id, space, role, workspace, project)
        return source, KnowledgeAccessContext(
            actor_user_id=actor_user_id,
            knowledge_space_id=context.knowledge_space_id,
            space_public_id=context.space_public_id,
            kind=context.kind,
            role=context.role,
            space_version=context.space_version,
            workspace_id=context.workspace_id,
            workspace_access_epoch=context.workspace_access_epoch,
            workspace_project_id=context.workspace_project_id,
            workspace_project_version=context.workspace_project_version,
        )

    async def _lock_source_access(
        self,
        session: AsyncSession,
        actor_user_id: int,
        source_public_id: str,
        expected_version: int,
        roles: frozenset[str],
        *,
        lifecycles: set[str],
    ) -> tuple[KnowledgeSource, KnowledgeAccessContext]:
        await self._lock_user(session, actor_user_id)
        source, access = await self._source_access_row(
            session,
            actor_user_id,
            source_public_id,
            roles,
            include_trashed=True,
        )
        await self._lock_access_context(session, access, roles)
        locked = await session.execute(
            update(KnowledgeSource)
            .where(
                KnowledgeSource.id == source.id,
                KnowledgeSource.version == expected_version,
                KnowledgeSource.lifecycle_status.in_(lifecycles),
            )
            .values(updated_at=KnowledgeSource.updated_at)
        )
        if locked.rowcount != 1:
            raise KnowledgeStaleError("Материал уже изменился.")
        await session.refresh(source)
        return source, access

    async def _lock_access_context(
        self,
        session: AsyncSession,
        access: KnowledgeAccessContext,
        roles: frozenset[str],
    ) -> None:
        if access.kind == "personal":
            return
        member = select(WorkspaceMember.id).where(
            WorkspaceMember.workspace_id == access.workspace_id,
            WorkspaceMember.user_id == access.actor_user_id,
            WorkspaceMember.status == "active",
            WorkspaceMember.role.in_(roles),
        )
        locked = await session.execute(
            update(Workspace)
            .where(
                Workspace.id == access.workspace_id,
                Workspace.status == "active",
                Workspace.access_epoch == access.workspace_access_epoch,
                member.exists(),
            )
            .values(updated_at=Workspace.updated_at)
        )
        if locked.rowcount != 1:
            raise KnowledgeAccessDenied("Область знаний недоступна.")
        if access.kind == "project":
            project = await session.scalar(
                select(WorkspaceProject.id).where(
                    WorkspaceProject.id == access.workspace_project_id,
                    WorkspaceProject.workspace_id == access.workspace_id,
                    WorkspaceProject.status == "active",
                    WorkspaceProject.version == access.workspace_project_version,
                )
            )
            if project is None:
                raise KnowledgeAccessDenied("Область знаний недоступна.")

    async def _lock_capture(
        self,
        session: AsyncSession,
        actor_user_id: int,
        chat_id: int,
        draft_public_id: str,
        expected_version: int,
        *,
        statuses: set[str],
        current: datetime,
        lock_user: bool = True,
    ) -> KnowledgeCaptureDraft:
        if lock_user:
            await self._lock_user(session, actor_user_id)
        draft = await session.scalar(
            select(KnowledgeCaptureDraft).where(
                KnowledgeCaptureDraft.public_id == self._public_id(draft_public_id),
                KnowledgeCaptureDraft.actor_user_id == actor_user_id,
                KnowledgeCaptureDraft.chat_id == chat_id,
            )
        )
        if draft is None:
            raise KnowledgeAccessDenied("Сбор материала недоступен.")
        if self._utc(draft.expires_at) <= current:
            if draft.status in {"collecting", "awaiting_confirmation", "confirming"}:
                draft.status = "expired"
                draft.completed_at = current
                draft.version += 1
                self._scrub_capture(draft)
                self._audit(
                    session,
                    "capture.expired",
                    actor_user_id=actor_user_id,
                    knowledge_space_id=draft.knowledge_space_id,
                    capture_draft_id=draft.id,
                )
            raise KnowledgeStaleError("Время сбора материала истекло.")
        if draft.version != expected_version or draft.status not in statuses:
            raise KnowledgeStaleError("Сбор материала уже изменился.")
        changed = await session.execute(
            update(KnowledgeCaptureDraft)
            .where(
                KnowledgeCaptureDraft.id == draft.id,
                KnowledgeCaptureDraft.version == expected_version,
                KnowledgeCaptureDraft.status.in_(statuses),
            )
            .values(updated_at=KnowledgeCaptureDraft.updated_at)
        )
        if changed.rowcount != 1:
            raise KnowledgeStaleError("Сбор материала уже изменился.")
        return draft

    async def _active_capture_row(
        self,
        session: AsyncSession,
        actor_user_id: int,
        chat_id: int,
        current: datetime,
        *,
        expire: bool,
    ) -> tuple[KnowledgeCaptureDraft | None, bool]:
        draft = await session.scalar(
            select(KnowledgeCaptureDraft)
            .where(
                KnowledgeCaptureDraft.actor_user_id == actor_user_id,
                KnowledgeCaptureDraft.chat_id == chat_id,
                KnowledgeCaptureDraft.status.in_(
                    ("collecting", "awaiting_confirmation", "confirming")
                ),
            )
            .order_by(KnowledgeCaptureDraft.id.desc())
            .limit(1)
        )
        if draft is None:
            return None, False
        if self._utc(draft.expires_at) > current:
            return draft, False
        if not expire:
            return None, True
        draft.status = "expired"
        draft.completed_at = current
        draft.version += 1
        self._scrub_capture(draft)
        reservations = list(
            (
                await session.scalars(
                    select(KnowledgeQuotaReservation).where(
                        KnowledgeQuotaReservation.capture_draft_id == draft.id,
                        KnowledgeQuotaReservation.status == "reserved",
                    )
                )
            ).all()
        )
        for reservation in reservations:
            reservation.status = "expired"
            reservation.completed_at = current
        self._audit(
            session,
            "capture.expired",
            actor_user_id=actor_user_id,
            knowledge_space_id=draft.knowledge_space_id,
            capture_draft_id=draft.id,
        )
        return None, True

    async def _capture_preview(
        self, session: AsyncSession, draft: KnowledgeCaptureDraft
    ) -> KnowledgeCapturePreview:
        target_name: str | None = None
        target_public_id: str | None = None
        if draft.knowledge_space_id is not None:
            access = await self._resolve_space_session(
                session,
                draft.actor_user_id,
                draft.knowledge_space_id,
                SPACE_ROLES,
                True,
            )
            target_public_id = access.space_public_id
            if access.kind == "personal":
                target_name = "Личная база знаний"
            elif access.kind == "workspace":
                target_name = await session.scalar(
                    select(Workspace.name).where(Workspace.id == access.workspace_id)
                )
            else:
                workspace_name = await session.scalar(
                    select(Workspace.name).where(Workspace.id == access.workspace_id)
                )
                project_name = await session.scalar(
                    select(WorkspaceProject.name).where(
                        WorkspaceProject.id == access.workspace_project_id
                    )
                )
                target_name = f"{workspace_name} / {project_name}"
        if draft.capture_kind in {"text", "forward"}:
            content_preview = self._preview(draft.text_content)
        elif draft.capture_kind == "url":
            content_preview = self._preview(draft.source_url)
        else:
            content_preview = draft.safe_display_name or "Файл"
        return KnowledgeCapturePreview(
            draft_public_id=draft.public_id,
            version=draft.version,
            status=draft.status,
            capture_kind=draft.capture_kind,
            target_space_public_id=target_public_id,
            target_name=target_name,
            title=draft.title,
            knowledge_role=draft.knowledge_role,
            priority=draft.priority,
            system_classification=draft.system_classification,
            user_classification=draft.user_classification,
            content_preview=content_preview,
            declared_mime=draft.declared_mime,
            declared_size_bytes=draft.declared_size_bytes,
            expires_at=self._utc(draft.expires_at),
        )

    @staticmethod
    def _capture_material(draft: KnowledgeCaptureDraft) -> KnowledgeCaptureMaterial:
        return KnowledgeCaptureMaterial(
            capture_kind=draft.capture_kind,
            text_content=draft.text_content,
            source_url=draft.source_url,
            telegram_file_id=draft.telegram_file_id,
            declared_mime=draft.declared_mime,
            safe_display_name=draft.safe_display_name,
            declared_size_bytes=draft.declared_size_bytes,
            provenance=draft.provenance,
        )

    @staticmethod
    def _capture_snapshot(draft: KnowledgeCaptureDraft, access: KnowledgeAccessContext) -> None:
        if (
            draft.knowledge_space_version != access.space_version
            or draft.workspace_access_epoch != access.workspace_access_epoch
            or draft.workspace_project_version != access.workspace_project_version
        ):
            raise KnowledgeStaleError("Доступ или назначение материала изменились.")

    async def _expire_reservations(self, session: AsyncSession, current: datetime) -> int:
        reservations = list(
            (
                await session.scalars(
                    select(KnowledgeQuotaReservation).where(
                        KnowledgeQuotaReservation.status == "reserved",
                        KnowledgeQuotaReservation.expires_at <= current,
                    )
                )
            ).all()
        )
        for reservation in reservations:
            reservation.status = "expired"
            reservation.completed_at = current
            draft = await session.get(KnowledgeCaptureDraft, reservation.capture_draft_id)
            if draft is not None and draft.status == "confirming":
                if self._utc(draft.expires_at) <= current:
                    draft.status = "expired"
                    draft.completed_at = current
                    self._scrub_capture(draft)
                    self._audit(
                        session,
                        "capture.expired",
                        actor_user_id=draft.actor_user_id,
                        knowledge_space_id=draft.knowledge_space_id,
                        capture_draft_id=draft.id,
                    )
                else:
                    draft.status = "awaiting_confirmation"
                draft.version += 1
        return len(reservations)

    async def _check_quota(
        self,
        session: AsyncSession,
        actor_user_id: int,
        knowledge_space_id: int,
        requested_bytes: int,
        current: datetime,
        *,
        new_sources: int = 1,
        new_extract_jobs: int = 1,
    ) -> None:
        if new_sources not in {0, 1} or new_extract_jobs not in {0, 1}:
            raise ValueError("Knowledge quota increments must be zero or one")
        day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
        active_source = KnowledgeSource.lifecycle_status != "purged"
        stored_bytes = case(
            (
                KnowledgeSourceRevision.original_storage_key.is_not(None),
                KnowledgeSourceRevision.size_bytes,
            ),
            else_=0,
        ) + func.coalesce(KnowledgeSourceRevision.extracted_size_bytes, 0)
        stored_user = int(
            await session.scalar(
                select(
                    func.coalesce(
                        func.sum(stored_bytes),
                        0,
                    )
                )
                .join(KnowledgeSource, KnowledgeSource.id == KnowledgeSourceRevision.source_id)
                .where(
                    KnowledgeSourceRevision.created_by_user_id == actor_user_id,
                    active_source,
                )
            )
            or 0
        )
        stored_space = int(
            await session.scalar(
                select(
                    func.coalesce(
                        func.sum(stored_bytes),
                        0,
                    )
                )
                .join(KnowledgeSource, KnowledgeSource.id == KnowledgeSourceRevision.source_id)
                .where(
                    KnowledgeSourceRevision.knowledge_space_id == knowledge_space_id,
                    active_source,
                )
            )
            or 0
        )
        daily_user = int(
            await session.scalar(
                select(func.coalesce(func.sum(KnowledgeSourceRevision.size_bytes), 0)).where(
                    KnowledgeSourceRevision.created_by_user_id == actor_user_id,
                    KnowledgeSourceRevision.original_storage_key.is_not(None),
                    KnowledgeSourceRevision.created_at >= day_start,
                )
            )
            or 0
        )
        daily_space = int(
            await session.scalar(
                select(func.coalesce(func.sum(KnowledgeSourceRevision.size_bytes), 0)).where(
                    KnowledgeSourceRevision.knowledge_space_id == knowledge_space_id,
                    KnowledgeSourceRevision.original_storage_key.is_not(None),
                    KnowledgeSourceRevision.created_at >= day_start,
                )
            )
            or 0
        )
        reserved_user = int(
            await session.scalar(
                select(func.coalesce(func.sum(KnowledgeQuotaReservation.reserved_bytes), 0)).where(
                    KnowledgeQuotaReservation.actor_user_id == actor_user_id,
                    KnowledgeQuotaReservation.status == "reserved",
                    KnowledgeQuotaReservation.expires_at > current,
                )
            )
            or 0
        )
        reserved_space = int(
            await session.scalar(
                select(func.coalesce(func.sum(KnowledgeQuotaReservation.reserved_bytes), 0)).where(
                    KnowledgeQuotaReservation.knowledge_space_id == knowledge_space_id,
                    KnowledgeQuotaReservation.status == "reserved",
                    KnowledgeQuotaReservation.expires_at > current,
                )
            )
            or 0
        )
        sources_user = int(
            await session.scalar(
                select(func.count(KnowledgeSourceRevision.id)).where(
                    KnowledgeSourceRevision.created_by_user_id == actor_user_id,
                    KnowledgeSourceRevision.original_storage_key.is_not(None),
                    KnowledgeSourceRevision.created_at >= day_start,
                )
            )
            or 0
        )
        sources_space = int(
            await session.scalar(
                select(func.count(KnowledgeSourceRevision.id)).where(
                    KnowledgeSourceRevision.knowledge_space_id == knowledge_space_id,
                    KnowledgeSourceRevision.original_storage_key.is_not(None),
                    KnowledgeSourceRevision.created_at >= day_start,
                )
            )
            or 0
        )
        reservations_user = int(
            await session.scalar(
                select(
                    func.coalesce(func.sum(KnowledgeQuotaReservation.reserved_sources), 0)
                ).where(
                    KnowledgeQuotaReservation.actor_user_id == actor_user_id,
                    KnowledgeQuotaReservation.status == "reserved",
                    KnowledgeQuotaReservation.expires_at > current,
                )
            )
            or 0
        )
        reservations_space = int(
            await session.scalar(
                select(
                    func.coalesce(func.sum(KnowledgeQuotaReservation.reserved_sources), 0)
                ).where(
                    KnowledgeQuotaReservation.knowledge_space_id == knowledge_space_id,
                    KnowledgeQuotaReservation.status == "reserved",
                    KnowledgeQuotaReservation.expires_at > current,
                )
            )
            or 0
        )
        pending_user = int(
            await session.scalar(
                select(func.count(KnowledgeIngestionJob.id)).where(
                    KnowledgeIngestionJob.requested_by_user_id == actor_user_id,
                    KnowledgeIngestionJob.status.in_(("queued", "processing")),
                )
            )
            or 0
        )
        pending_space = int(
            await session.scalar(
                select(func.count(KnowledgeIngestionJob.id)).where(
                    KnowledgeIngestionJob.knowledge_space_id == knowledge_space_id,
                    KnowledgeIngestionJob.status.in_(("queued", "processing")),
                )
            )
            or 0
        )
        pending_extract_user = int(
            await session.scalar(
                select(func.count(KnowledgeIngestionJob.id)).where(
                    KnowledgeIngestionJob.requested_by_user_id == actor_user_id,
                    KnowledgeIngestionJob.job_type == "extract",
                    KnowledgeIngestionJob.status.in_(("queued", "processing")),
                )
            )
            or 0
        )
        pending_extract_space = int(
            await session.scalar(
                select(func.count(KnowledgeIngestionJob.id)).where(
                    KnowledgeIngestionJob.knowledge_space_id == knowledge_space_id,
                    KnowledgeIngestionJob.job_type == "extract",
                    KnowledgeIngestionJob.status.in_(("queued", "processing")),
                )
            )
            or 0
        )
        reserved_jobs_user = int(
            await session.scalar(
                select(func.coalesce(func.sum(KnowledgeQuotaReservation.reserved_jobs), 0)).where(
                    KnowledgeQuotaReservation.actor_user_id == actor_user_id,
                    KnowledgeQuotaReservation.status == "reserved",
                    KnowledgeQuotaReservation.expires_at > current,
                )
            )
            or 0
        )
        reserved_jobs_space = int(
            await session.scalar(
                select(func.coalesce(func.sum(KnowledgeQuotaReservation.reserved_jobs), 0)).where(
                    KnowledgeQuotaReservation.knowledge_space_id == knowledge_space_id,
                    KnowledgeQuotaReservation.status == "reserved",
                    KnowledgeQuotaReservation.expires_at > current,
                )
            )
            or 0
        )
        user_headroom = (
            pending_extract_user + reserved_jobs_user + new_extract_jobs
        ) * self.quota.max_extracted_bytes
        space_headroom = (
            pending_extract_space + reserved_jobs_space + new_extract_jobs
        ) * self.quota.max_extracted_bytes
        if (
            stored_user + reserved_user + requested_bytes + user_headroom
            > self.quota.storage_bytes_per_user
        ):
            raise KnowledgeQuotaError("Личная квота хранения исчерпана.")
        if (
            stored_space + reserved_space + requested_bytes + space_headroom
            > self.quota.storage_bytes_per_space
        ):
            raise KnowledgeQuotaError("Квота пространства исчерпана.")
        if daily_user + reserved_user + requested_bytes > self.quota.daily_ingest_bytes_per_user:
            raise KnowledgeQuotaError("Дневная квота загрузки исчерпана.")
        if daily_space + reserved_space + requested_bytes > self.quota.daily_ingest_bytes_per_space:
            raise KnowledgeQuotaError("Дневная квота пространства исчерпана.")
        if sources_user + reservations_user + new_sources > self.quota.daily_sources_per_user:
            raise KnowledgeQuotaError("Дневной лимит материалов исчерпан.")
        if sources_space + reservations_space + new_sources > self.quota.daily_sources_per_space:
            raise KnowledgeQuotaError("Дневной лимит пространства исчерпан.")
        if (
            pending_user + reserved_jobs_user + new_extract_jobs
            > self.quota.max_pending_jobs_per_user
        ):
            raise KnowledgeQuotaError("Слишком много ожидающих заданий.")
        if (
            pending_space + reserved_jobs_space + new_extract_jobs
            > self.quota.max_pending_jobs_per_space
        ):
            raise KnowledgeQuotaError("Очередь пространства заполнена.")

    async def set_maintenance_paused(
        self,
        paused: bool,
        *,
        expected_version: int | None = None,
    ) -> int:
        async with self.db.session() as session:
            await self._ensure_runtime_state(session)
            query = update(KnowledgeRuntimeState).where(KnowledgeRuntimeState.id == 1)
            if expected_version is not None:
                query = query.where(KnowledgeRuntimeState.version == expected_version)
            changed = await session.execute(
                query.values(
                    maintenance_paused=bool(paused),
                    version=KnowledgeRuntimeState.version + 1,
                    updated_at=datetime.now(UTC),
                ).returning(KnowledgeRuntimeState.version)
            )
            version = changed.scalar_one_or_none()
            if version is None:
                raise KnowledgeStaleError("Режим обслуживания уже изменился.")
            return int(version)

    async def runtime_paused(self) -> bool:
        async with self.db.session() as session:
            await self._ensure_runtime_state(session)
            value = await session.scalar(
                select(KnowledgeRuntimeState.maintenance_paused).where(
                    KnowledgeRuntimeState.id == 1
                )
            )
            if value is None:
                raise KnowledgeConflictError("Состояние Knowledge runtime отсутствует.")
            return bool(value)

    async def _runtime_available(self, session: AsyncSession) -> bool:
        # A no-op UPDATE is the cross-dialect mutex shared with the backup
        # pause transaction. Once this succeeds, pause cannot overtake this tx.
        await self._ensure_runtime_state(session)
        changed = await session.execute(
            update(KnowledgeRuntimeState)
            .where(
                KnowledgeRuntimeState.id == 1,
                KnowledgeRuntimeState.maintenance_paused.is_(False),
            )
            .values(updated_at=KnowledgeRuntimeState.updated_at)
            .returning(KnowledgeRuntimeState.id)
        )
        return changed.scalar_one_or_none() is not None

    @staticmethod
    async def _ensure_runtime_state(session: AsyncSession) -> None:
        if (
            await session.scalar(
                select(KnowledgeRuntimeState.id).where(KnowledgeRuntimeState.id == 1)
            )
            is None
        ):
            session.add(
                KnowledgeRuntimeState(
                    id=1,
                    maintenance_paused=False,
                    version=1,
                )
            )
            await session.flush()

    async def _lock_runtime(self, session: AsyncSession) -> None:
        if not await self._runtime_available(session):
            raise KnowledgeConflictError("База знаний временно приостановлена.")

    async def _receipt_for_reservation(
        self,
        session: AsyncSession,
        reservation: KnowledgeQuotaReservation,
        chat_id: int,
    ) -> KnowledgeSourceReceipt:
        draft = await session.get(KnowledgeCaptureDraft, reservation.capture_draft_id)
        source = await session.get(KnowledgeSource, reservation.source_id)
        revision = await session.get(KnowledgeSourceRevision, reservation.revision_id)
        if draft is None or draft.chat_id != chat_id or source is None or revision is None:
            raise KnowledgeCaptureError("Подтверждение недействительно.")
        job = await session.scalar(
            select(KnowledgeIngestionJob)
            .where(
                KnowledgeIngestionJob.source_id == source.id,
                or_(
                    KnowledgeIngestionJob.revision_id == revision.id,
                    KnowledgeIngestionJob.job_type == "purge",
                ),
            )
            .order_by(KnowledgeIngestionJob.id.desc())
            .limit(1)
        )
        if job is None:
            raise KnowledgeCaptureError("Задание материала недоступно.")
        return KnowledgeSourceReceipt(
            source.public_id,
            revision.public_id,
            job.public_id,
            source.processing_status,
            source.version,
        )

    async def _lock_claim(
        self,
        session: AsyncSession,
        job_id: int,
        lease_token: str,
        current: datetime,
        job_type: str | None,
    ) -> KnowledgeIngestionJob | None:
        conditions = [
            KnowledgeIngestionJob.id == job_id,
            KnowledgeIngestionJob.status == "processing",
            KnowledgeIngestionJob.lease_token == lease_token,
            KnowledgeIngestionJob.lease_expires_at > current,
        ]
        if job_type is not None:
            conditions.append(KnowledgeIngestionJob.job_type == job_type)
        locked = await session.execute(
            update(KnowledgeIngestionJob)
            .where(*conditions)
            .values(updated_at=KnowledgeIngestionJob.updated_at)
            .returning(KnowledgeIngestionJob.id)
        )
        if locked.scalar_one_or_none() is None:
            return None
        return await session.get(KnowledgeIngestionJob, job_id)

    @staticmethod
    def _finish_job(
        job: KnowledgeIngestionJob,
        status: str,
        current: datetime,
        safe_error_code: str | None,
    ) -> None:
        job.status = status
        job.lease_owner = None
        job.lease_token = None
        job.lease_expires_at = None
        job.heartbeat_at = None
        job.finished_at = current
        job.safe_error_code = safe_error_code
        job.version += 1

    async def _cancel_job_locked(
        self,
        session: AsyncSession,
        job: KnowledgeIngestionJob,
        current: datetime,
    ) -> bool:
        self._finish_job(job, "cancelled", current, None)
        source = await session.get(KnowledgeSource, job.source_id)
        if job.job_type == "extract":
            revision = await session.get(KnowledgeSourceRevision, job.revision_id)
            if revision is not None and revision.extraction_status == "pending":
                revision.extraction_status = "cancelled"
                revision.finalized_at = current
            if source is not None and source.lifecycle_status == "active":
                source.processing_status = "cancelled"
        elif source is not None and source.lifecycle_status == "purge_pending":
            source.lifecycle_status = "trashed"
            source.purge_requested_at = None
            source.version += 1
        return True

    async def _cancel_open_jobs(
        self, session: AsyncSession, source_id: int, current: datetime
    ) -> None:
        jobs = list(
            (
                await session.scalars(
                    select(KnowledgeIngestionJob).where(
                        KnowledgeIngestionJob.source_id == source_id,
                        KnowledgeIngestionJob.status.in_(("queued", "processing")),
                    )
                )
            ).all()
        )
        for job in jobs:
            if job.status == "queued":
                self._finish_job(job, "cancelled", current, None)
                revision = await session.get(KnowledgeSourceRevision, job.revision_id)
                if revision is not None and revision.extraction_status == "pending":
                    revision.extraction_status = "cancelled"
                    revision.finalized_at = current
            else:
                job.cancel_requested_at = current
                job.version += 1

    async def _fail_exhausted_jobs(self, current: datetime) -> None:
        exhausted = or_(
            and_(
                KnowledgeIngestionJob.status == "queued",
                KnowledgeIngestionJob.attempt_count >= KnowledgeIngestionJob.max_attempts,
            ),
            and_(
                KnowledgeIngestionJob.status == "processing",
                KnowledgeIngestionJob.lease_expires_at <= current,
                KnowledgeIngestionJob.attempt_count >= KnowledgeIngestionJob.max_attempts,
            ),
        )
        async with self.db.session() as session:
            if not await self._runtime_available(session):
                return
            jobs = list(
                (await session.scalars(select(KnowledgeIngestionJob).where(exhausted))).all()
            )
            for job in jobs:
                self._finish_job(job, "failed", current, "attempts_exhausted")
                source = await session.get(KnowledgeSource, job.source_id)
                if source is None:
                    continue
                if job.job_type == "extract":
                    revision = await session.get(KnowledgeSourceRevision, job.revision_id)
                    if revision is not None and revision.extraction_status == "pending":
                        revision.extraction_status = "failed"
                        revision.finalized_at = current
                    source.processing_status = "failed"
                elif source.lifecycle_status == "purge_pending":
                    source.lifecycle_status = "purge_failed"
                    source.version += 1
                self._audit(
                    session,
                    "ingestion.status_changed",
                    actor_user_id=None,
                    knowledge_space_id=source.knowledge_space_id,
                    source_id=source.id,
                    revision_id=job.revision_id,
                    job_id=job.id,
                    safe_metadata={"status": "failed", "job_type": job.job_type},
                )
                if job.job_type == "purge":
                    self._audit(
                        session,
                        "source.purge_failed",
                        actor_user_id=None,
                        knowledge_space_id=source.knowledge_space_id,
                        source_id=source.id,
                    )

    async def _validate_action(
        self,
        session: AsyncSession,
        row: KnowledgeActionToken,
        actor_user_id: int,
    ) -> ClaimedKnowledgeAction | None:
        try:
            access = await self._resolve_space_session(
                session,
                actor_user_id,
                row.knowledge_space_id,
                SPACE_ROLES,
                True,
            )
        except KnowledgeAccessDenied:
            return None
        if (
            access.space_version != row.knowledge_space_version
            or access.workspace_access_epoch != row.workspace_access_epoch
        ):
            return None
        capture_public_id: str | None = None
        source_public_id: str | None = None
        if row.scope_kind == "capture":
            capture = await session.scalar(
                select(KnowledgeCaptureDraft).where(
                    KnowledgeCaptureDraft.id == row.capture_draft_id,
                    KnowledgeCaptureDraft.actor_user_id == actor_user_id,
                    KnowledgeCaptureDraft.knowledge_space_id == access.knowledge_space_id,
                    KnowledgeCaptureDraft.version == row.capture_version,
                    KnowledgeCaptureDraft.status.in_(("collecting", "awaiting_confirmation")),
                )
            )
            if capture is None:
                return None
            capture_public_id = capture.public_id
        elif row.scope_kind == "source":
            source = await session.scalar(
                select(KnowledgeSource).where(
                    KnowledgeSource.id == row.source_id,
                    KnowledgeSource.knowledge_space_id == access.knowledge_space_id,
                    KnowledgeSource.version == row.source_version,
                    KnowledgeSource.lifecycle_status != "purged",
                )
            )
            if source is None:
                return None
            source_public_id = source.public_id
        return ClaimedKnowledgeAction(
            action=row.action,
            payload=dict(row.payload or {}),
            scope_kind=row.scope_kind,
            space_public_id=access.space_public_id,
            capture_draft_public_id=capture_public_id,
            source_public_id=source_public_id,
        )

    @staticmethod
    async def _lock_user(session: AsyncSession, user_id: int) -> None:
        changed = await session.execute(
            update(User).where(User.id == user_id).values(updated_at=User.updated_at)
        )
        if changed.rowcount != 1:
            raise KnowledgeAccessDenied("Пользователь недоступен.")

    @staticmethod
    async def _lock_space(session: AsyncSession, knowledge_space_id: int) -> None:
        changed = await session.execute(
            update(KnowledgeSpace)
            .where(
                KnowledgeSpace.id == knowledge_space_id,
                KnowledgeSpace.status == "active",
            )
            .values(updated_at=KnowledgeSpace.updated_at)
        )
        if changed.rowcount != 1:
            raise KnowledgeAccessDenied("Область знаний недоступна.")

    @staticmethod
    def _audit(
        session: AsyncSession,
        event_type: str,
        *,
        actor_user_id: int | None,
        workspace_id: int | None = None,
        knowledge_space_id: int | None = None,
        capture_draft_id: int | None = None,
        source_id: int | None = None,
        revision_id: int | None = None,
        job_id: int | None = None,
        safe_metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = KnowledgeService._safe_audit_metadata(safe_metadata)
        session.add(
            KnowledgeAuditEvent(
                public_id=str(uuid4()),
                event_type=event_type,
                actor_user_id=actor_user_id,
                workspace_id=workspace_id,
                knowledge_space_id=knowledge_space_id,
                capture_draft_id=capture_draft_id,
                source_id=source_id,
                revision_id=revision_id,
                job_id=job_id,
                safe_metadata=metadata,
            )
        )

    @staticmethod
    def _safe_audit_metadata(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        allowed_keys = {
            "capture_kind",
            "source_type",
            "revision_number",
            "status",
            "job_type",
            "role",
            "priority",
            "classification",
        }
        if not set(value).issubset(allowed_keys):
            raise KnowledgeError("Audit metadata contains a forbidden field")
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, bool | int):
                clean[key] = item
            elif (
                isinstance(item, str)
                and len(item) <= 64
                and not any(marker in item for marker in ("/", "\\", "://", "@"))
            ):
                clean[key] = item
            else:
                raise KnowledgeError("Audit metadata contains a sensitive value")
        return clean

    @staticmethod
    def _scrub_capture(draft: KnowledgeCaptureDraft) -> None:
        draft.text_content = None
        draft.source_url = None
        draft.telegram_file_id = None
        draft.telegram_file_unique_id_hash = None
        draft.telegram_message_id = None
        draft.declared_mime = None
        draft.safe_display_name = None
        draft.declared_size_bytes = None
        draft.provenance = None

    @staticmethod
    def _capture_kind(value: CaptureKind | str) -> str:
        clean = value.strip().casefold()
        if clean not in CAPTURE_KINDS:
            raise KnowledgeCaptureError("Неподдерживаемый тип материала.")
        return clean

    def _capture_payload(
        self,
        kind: str,
        *,
        text_content: str | None,
        source_url: str | None,
        telegram_file_id: str | None,
        telegram_file_unique_id_hash: str | None,
        declared_mime: str | None,
        safe_display_name: str | None,
        declared_size_bytes: int | None,
        provenance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        text_value: str | None = None
        url_value: str | None = None
        file_id_value: str | None = None
        unique_hash: str | None = None
        if kind in {"text", "forward"}:
            if text_content is None or not text_content.strip():
                raise KnowledgeCaptureError("Текст материала пуст.")
            text_value = unicodedata.normalize("NFKC", text_content).strip()
            if len(text_value.encode("utf-8")) > self.quota.max_source_bytes:
                raise KnowledgeQuotaError("Материал превышает допустимый размер.")
        elif kind == "url":
            url_value = self._url(source_url)
        else:
            file_id_value = self._bounded_key(telegram_file_id, "file", 512)
            if telegram_file_unique_id_hash is not None:
                unique_hash = self._sha256(telegram_file_unique_id_hash)
        if declared_size_bytes is not None and not (
            0 <= declared_size_bytes <= self.quota.max_source_bytes
        ):
            raise KnowledgeQuotaError("Материал превышает допустимый размер.")
        return {
            "text_content": text_value,
            "source_url": url_value,
            "telegram_file_id": file_id_value,
            "telegram_file_unique_id_hash": unique_hash,
            "declared_mime": self._mime(declared_mime),
            "safe_display_name": (
                self._safe_display_name(safe_display_name)
                if safe_display_name is not None
                else None
            ),
            "declared_size_bytes": declared_size_bytes,
            "provenance": self._json_value(provenance, maximum=4096),
        }

    @staticmethod
    def _default_title(kind: str, payload: dict[str, Any]) -> str:
        if payload.get("safe_display_name"):
            return str(payload["safe_display_name"])
        if kind == "url":
            hostname = urlsplit(str(payload["source_url"])).hostname
            return hostname or "Ссылка"
        if kind in {"text", "forward"}:
            text_value = str(payload["text_content"])
            return text_value[:80]
        return "Материал"

    @staticmethod
    def _title(value: str) -> str:
        clean = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
        if not 1 <= len(clean) <= 200:
            raise KnowledgeError("Название должно быть от 1 до 200 символов.")
        if any(unicodedata.category(character).startswith("C") for character in clean):
            raise KnowledgeError("Название содержит недопустимые символы.")
        return clean

    @staticmethod
    def _knowledge_role(value: KnowledgeRole | str) -> str:
        clean = value.strip().casefold()
        if clean not in KNOWLEDGE_ROLES:
            raise KnowledgeError("Некорректная роль знания.")
        return clean

    @staticmethod
    def _priority(value: KnowledgePriority | str) -> str:
        clean = value.strip().casefold()
        if clean not in KNOWLEDGE_PRIORITIES:
            raise KnowledgeError("Некорректный приоритет.")
        return clean

    @staticmethod
    def _classification(value: KnowledgeClassification | str) -> str:
        clean = value.strip().casefold()
        if clean not in KNOWLEDGE_CLASSIFICATIONS:
            raise KnowledgeError("Некорректная классификация.")
        return clean

    @staticmethod
    def _medical_scope(classification: str, space_kind: str) -> None:
        if classification == "health_private" and space_kind != "personal":
            raise KnowledgeAccessDenied(
                "Медицинский материал можно сохранить только в личной области."
            )

    @staticmethod
    def _user_classification(value: str | None | object) -> str | None:
        if value is None or value is _UNSET:
            return None
        clean = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()
        if not 1 <= len(clean) <= 64:
            raise KnowledgeError("Пользовательская классификация слишком длинная.")
        if any(unicodedata.category(character).startswith("C") for character in clean):
            raise KnowledgeError("Классификация содержит недопустимые символы.")
        return clean

    @staticmethod
    def _public_id(value: str) -> str:
        try:
            parsed = UUID(value.strip())
        except (AttributeError, ValueError) as exc:
            raise KnowledgeAccessDenied("Объект недоступен.") from exc
        clean = str(parsed)
        if len(clean) != 36:
            raise KnowledgeAccessDenied("Объект недоступен.")
        return clean

    @staticmethod
    def _bounded_key(value: str | None, field_name: str, maximum: int) -> str:
        if value is None:
            raise KnowledgeError(f"Отсутствует значение: {field_name}.")
        clean = value.strip()
        if not 1 <= len(clean) <= maximum or any(
            unicodedata.category(character).startswith("C") for character in clean
        ):
            raise KnowledgeError(f"Некорректное значение: {field_name}.")
        return clean

    @staticmethod
    def _idempotency_key(value: str) -> str:
        clean = value.strip()
        if not 8 <= len(clean) <= 128 or any(character.isspace() for character in clean):
            raise KnowledgeError("Некорректный ключ повторяемости.")
        return clean

    @staticmethod
    def _pipeline_version(value: str) -> str:
        clean = value.strip().casefold()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,31}", clean):
            raise KnowledgeJobError("Некорректная версия pipeline.")
        return clean

    @staticmethod
    def _safe_code(value: str) -> str:
        clean = value.strip().casefold()
        if not _SAFE_CODE.fullmatch(clean):
            raise KnowledgeJobError("Некорректный безопасный код ошибки.")
        return clean

    @staticmethod
    def _action(value: str) -> str:
        clean = value.strip().casefold()
        if not _ACTION.fullmatch(clean):
            raise KnowledgeError("Некорректное действие.")
        return clean

    @staticmethod
    def _sha256(value: str) -> str:
        clean = value.strip().casefold()
        if not _SHA256.fullmatch(clean):
            raise KnowledgeError("Некорректная контрольная сумма.")
        return clean

    @staticmethod
    def _storage_key(value: str) -> str:
        clean = value.strip()
        path = PurePosixPath(clean)
        if (
            not 1 <= len(clean) <= 512
            or path.is_absolute()
            or clean.startswith("/")
            or "\\" in clean
            or ".." in path.parts
            or "." in path.parts
            or any(not part for part in path.parts)
            or any(unicodedata.category(character).startswith("C") for character in clean)
        ):
            raise KnowledgeError("Некорректный storage key.")
        return str(path)

    @staticmethod
    def _mime(value: str | None) -> str | None:
        if value is None:
            return None
        clean = value.strip().casefold()
        if not re.fullmatch(
            r"[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,62}", clean
        ):
            raise KnowledgeError("Некорректный MIME.")
        return clean

    @staticmethod
    def _safe_display_name(value: str) -> str:
        clean = unicodedata.normalize("NFKC", value).strip()
        if (
            not 1 <= len(clean) <= 255
            or "/" in clean
            or "\\" in clean
            or clean in {".", ".."}
            or any(unicodedata.category(character).startswith("C") for character in clean)
        ):
            raise KnowledgeError("Некорректное отображаемое имя.")
        return clean

    @staticmethod
    def _url(value: str | None) -> str:
        if value is None:
            raise KnowledgeCaptureError("Ссылка отсутствует.")
        clean = value.strip()
        if not 1 <= len(clean) <= 2048:
            raise KnowledgeCaptureError("Ссылка слишком длинная.")
        parsed = urlsplit(clean)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise KnowledgeCaptureError("Допустима только обычная HTTP(S)-ссылка.")
        return clean

    @staticmethod
    def _provenance_kind(capture_kind: str) -> str:
        return {
            "text": "manual_text",
            "forward": "telegram_forward",
            "document": "telegram_document",
            "image": "telegram_image",
            "url": "user_url",
        }[capture_kind]

    def _original(self, value: StoredKnowledgeOriginal) -> StoredKnowledgeOriginal:
        if not isinstance(value, StoredKnowledgeOriginal):
            raise KnowledgeError("Некорректный оригинал материала.")
        if not 0 <= value.size_bytes <= self.quota.max_source_bytes:
            raise KnowledgeQuotaError("Материал превышает допустимый размер.")
        clean_format = value.detected_format.strip().casefold()
        if clean_format not in {"text", "txt", "markdown", "pdf", "docx", "epub", "image", "url"}:
            raise KnowledgeError("Неподдерживаемый формат материала.")
        return StoredKnowledgeOriginal(
            storage_key=self._storage_key(value.storage_key),
            sha256=self._sha256(value.sha256),
            size_bytes=value.size_bytes,
            declared_mime=self._mime(value.declared_mime),
            detected_mime=self._mime(value.detected_mime) or "application/octet-stream",
            detected_format=clean_format,
            safe_display_name=self._safe_display_name(value.safe_display_name),
            provenance=self._json_value(value.provenance, maximum=4096),
        )

    def _extraction_result(self, value: KnowledgeExtractionResult) -> KnowledgeExtractionResult:
        if value.status not in {"ready", "partial"}:
            raise KnowledgeJobError("Некорректный результат извлечения.")
        tuple_values = (
            value.extracted_storage_key,
            value.extracted_sha256,
            value.extracted_size_bytes,
        )
        if any(item is not None for item in tuple_values) and not all(
            item is not None for item in tuple_values
        ):
            raise KnowledgeJobError("Неполный результат извлечения.")
        if value.status == "ready" and value.extracted_storage_key is None:
            raise KnowledgeJobError("Готовый результат должен содержать извлечённый текст.")
        if value.status == "ready" and value.safe_error_code is not None:
            raise KnowledgeJobError("Готовый результат не может содержать ошибку.")
        extracted_size = value.extracted_size_bytes
        if extracted_size is not None and not 0 <= extracted_size <= self.quota.max_extracted_bytes:
            raise KnowledgeJobError("Извлечённый результат слишком велик.")
        return KnowledgeExtractionResult(
            status=value.status,
            extracted_storage_key=(
                self._storage_key(value.extracted_storage_key)
                if value.extracted_storage_key is not None
                else None
            ),
            extracted_sha256=(
                self._sha256(value.extracted_sha256) if value.extracted_sha256 is not None else None
            ),
            extracted_size_bytes=extracted_size,
            metadata=self._json_value(value.metadata, maximum=8192),
            safe_error_code=(
                self._safe_code(value.safe_error_code)
                if value.safe_error_code is not None
                else None
            ),
        )

    @staticmethod
    def _json_value(value: dict[str, Any] | None, *, maximum: int) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            decoded = json.loads(serialized)
        except (TypeError, ValueError) as exc:
            raise KnowledgeError("Некорректные метаданные.") from exc
        if not isinstance(decoded, dict) or len(serialized.encode("utf-8")) > maximum:
            raise KnowledgeError("Метаданные слишком велики.")
        return decoded

    @staticmethod
    def _preview(value: str | None) -> str | None:
        if value is None:
            return None
        clean = re.sub(r"\s+", " ", value).strip()
        return clean if len(clean) <= 200 else f"{clean[:197]}…"

    @staticmethod
    def _ttl(
        value: timedelta | None,
        default: timedelta,
        maximum: timedelta,
    ) -> timedelta:
        result = value or default
        if result < timedelta(minutes=1) or result > maximum:
            raise KnowledgeError("Некорректный срок действия.")
        return result

    @staticmethod
    def _page(page: int, page_size: int) -> tuple[int, int]:
        if page < 1 or not 1 <= page_size <= 50:
            raise KnowledgeError("Некорректная страница.")
        return page, page_size

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
