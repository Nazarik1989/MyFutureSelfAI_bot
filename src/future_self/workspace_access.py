from __future__ import annotations

import hashlib
import json
import re
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from .db import Database
from .models import (
    KnowledgeAuditEvent,
    KnowledgeSpace,
    User,
    Workspace,
    WorkspaceActionToken,
    WorkspaceContext,
    WorkspaceInvitation,
    WorkspaceMember,
    WorkspaceProject,
)

WorkspaceCharacter = Literal["pair", "friends", "family", "team", "custom"]
WorkspaceRole = Literal["owner", "editor", "viewer"]
InvitationRole = Literal["editor", "viewer"]
InvitationDelivery = Literal["direct", "share"]
WorkspaceStatus = Literal["active", "archived"]

WORKSPACE_CHARACTERS = frozenset({"pair", "friends", "family", "team", "custom"})
WORKSPACE_ROLES = frozenset({"owner", "editor", "viewer"})
INVITATION_ROLES = frozenset({"editor", "viewer"})
INVITATION_DELIVERY_MODES = frozenset({"direct", "share"})
INVITATION_ACTIONS = frozenset({"accept", "decline", "details", "later"})
_OWNER_ROLES = frozenset({"owner"})
_EDIT_ROLES = frozenset({"owner", "editor"})
_ALL_ROLES = frozenset({"owner", "editor", "viewer"})
_KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]{0,63}\Z")
_ACTION_PATTERN = re.compile(r"[a-z0-9][a-z0-9_:.-]{0,47}\Z")


class WorkspaceAccessError(ValueError):
    """Base class for safe user-facing workspace failures."""


class WorkspaceAccessDenied(WorkspaceAccessError):
    """The same error is used for absent and inaccessible objects (no ID oracle)."""


class WorkspaceConflictError(WorkspaceAccessError):
    pass


class WorkspaceStaleError(WorkspaceAccessError):
    pass


class WorkspaceLastOwnerError(WorkspaceConflictError):
    pass


class WorkspaceInvitationError(WorkspaceAccessError):
    """Generic invalid/expired/consumed/wrong-recipient invitation error."""


@dataclass(frozen=True, slots=True)
class AccessContext:
    actor_user_id: int
    workspace_id: int
    access_epoch: int


@dataclass(frozen=True, slots=True)
class WorkspacePage:
    items: tuple[Workspace, ...]
    page: int
    pages: int
    total: int


@dataclass(frozen=True, slots=True)
class WorkspaceMemberRecord:
    member: WorkspaceMember
    display_name: str


@dataclass(frozen=True, slots=True)
class IssuedInvitation:
    invitation: WorkspaceInvitation
    token: str


@dataclass(frozen=True, slots=True)
class InvitationPreview:
    inviter_display_name: str
    workspace_name: str
    character: str
    role: str
    template_key: str
    custom_text: str | None
    expires_at: datetime
    version: int


@dataclass(frozen=True, slots=True)
class IncomingInvitationActions:
    preview: InvitationPreview
    actions: dict[str, str]


@dataclass(frozen=True, slots=True)
class InvitationActionResult:
    action: str
    status: str
    preview: InvitationPreview
    access_context: AccessContext | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceContextSnapshot:
    access_context: AccessContext
    context_version: int
    workspace_project: WorkspaceProject | None


@dataclass(frozen=True, slots=True)
class WorkspaceActionClaim:
    action: str
    payload: dict[str, Any]
    access_context: AccessContext | None
    workspace_version: int | None
    workspace_project_id: int | None
    workspace_project_version: int | None


@dataclass(frozen=True, slots=True)
class WorkspaceCleanupResult:
    action_tokens: int
    contexts: int
    invitations: int


def normalize_workspace_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    normalized = "".join(character if character.isalnum() else " " for character in normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def clean_workspace_name(value: str) -> tuple[str, str]:
    display = _clean_text(value, maximum=100, field="Название", allow_empty=False)
    normalized = normalize_workspace_name(display)
    if not normalized:
        raise WorkspaceAccessError("Название не может быть пустым.")
    if len(normalized) > 100:
        raise WorkspaceAccessError("Название должно быть не длиннее 100 символов.")
    return display, normalized


def clean_workspace_description(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _clean_text(value, maximum=500, field="Описание", allow_empty=True)
    return cleaned or None


def clean_invitation_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _clean_text(value, maximum=1000, field="Текст приглашения", allow_empty=True)
    return cleaned or None


def _clean_text(value: str, *, maximum: int, field: str, allow_empty: bool) -> str:
    cleaned = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
    if not cleaned and not allow_empty:
        raise WorkspaceAccessError(f"{field} не может быть пустым.")
    if len(cleaned) > maximum:
        raise WorkspaceAccessError(f"{field} должно быть не длиннее {maximum} символов.")
    if any(unicodedata.category(character).startswith("C") for character in cleaned):
        raise WorkspaceAccessError(f"{field} содержит недопустимые символы.")
    return cleaned


def _clean_key(value: str, *, field: str) -> str:
    cleaned = value.strip().casefold()
    if not _KEY_PATTERN.fullmatch(cleaned):
        raise WorkspaceAccessError(f"Некорректное значение: {field}.")
    return cleaned


def _clean_action(value: str) -> str:
    cleaned = value.strip().casefold()
    if not _ACTION_PATTERN.fullmatch(cleaned):
        raise WorkspaceAccessError("Некорректное действие.")
    return cleaned


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class WorkspaceAccessService:
    PAGE_SIZE = 6
    INVITATION_TTL = timedelta(days=7)
    ACTION_TTL = timedelta(minutes=15)
    CONTEXT_TTL = timedelta(hours=8)
    MIN_INVITATION_TTL = timedelta(minutes=5)
    MAX_INVITATION_TTL = timedelta(days=30)
    MAX_ACTION_TTL = timedelta(hours=24)
    MAX_CONTEXT_TTL = timedelta(days=7)

    def __init__(self, db: Database):
        self.db = db

    async def cleanup(self, *, now: datetime | None = None) -> WorkspaceCleanupResult:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            actions = await session.execute(
                delete(WorkspaceActionToken).where(
                    WorkspaceActionToken.expires_at <= current,
                    WorkspaceActionToken.status != "awaiting_input",
                )
            )
            valid_context = exists(
                select(Workspace.id)
                .join(
                    WorkspaceMember,
                    and_(
                        WorkspaceMember.workspace_id == Workspace.id,
                        WorkspaceMember.user_id == WorkspaceContext.actor_user_id,
                        WorkspaceMember.status == "active",
                    ),
                )
                .outerjoin(
                    WorkspaceProject,
                    and_(
                        WorkspaceProject.id == WorkspaceContext.workspace_project_id,
                        WorkspaceProject.workspace_id == Workspace.id,
                    ),
                )
                .where(
                    Workspace.id == WorkspaceContext.workspace_id,
                    Workspace.status == "active",
                    Workspace.access_epoch == WorkspaceContext.workspace_access_epoch,
                    or_(
                        WorkspaceContext.workspace_project_id.is_(None),
                        and_(
                            WorkspaceProject.status == "active",
                            WorkspaceProject.version == WorkspaceContext.workspace_project_version,
                        ),
                    ),
                )
            )
            contexts = await session.execute(
                delete(WorkspaceContext).where(
                    or_(WorkspaceContext.expires_at <= current, ~valid_context)
                )
            )
            invitations = await session.execute(
                update(WorkspaceInvitation)
                .where(
                    WorkspaceInvitation.status == "pending",
                    WorkspaceInvitation.expires_at <= current,
                )
                .values(status="expired", version=WorkspaceInvitation.version + 1)
            )
            return WorkspaceCleanupResult(
                actions.rowcount or 0,
                contexts.rowcount or 0,
                invitations.rowcount or 0,
            )

    async def create_workspace(
        self,
        actor_user_id: int,
        character: WorkspaceCharacter | str,
        name: str,
        description: str | None = None,
    ) -> Workspace:
        display, normalized = clean_workspace_name(name)
        clean_character = self._character(character)
        clean_description = clean_workspace_description(description)
        try:
            async with self.db.session() as session:
                await self._lock_user(session, actor_user_id)
                workspace = Workspace(
                    name=display,
                    normalized_name=normalized,
                    character=clean_character,
                    description=clean_description,
                    created_by_user_id=actor_user_id,
                    status="active",
                    access_epoch=1,
                    version=1,
                )
                session.add(workspace)
                await session.flush()
                session.add(
                    WorkspaceMember(
                        workspace_id=workspace.id,
                        user_id=actor_user_id,
                        role="owner",
                        status="active",
                        invited_by_user_id=actor_user_id,
                        joined_at=datetime.now(UTC),
                        version=1,
                    )
                )
                knowledge_space = KnowledgeSpace(
                    kind="workspace",
                    workspace_id=workspace.id,
                    status="active",
                    version=1,
                )
                session.add(knowledge_space)
                await session.flush()
                await self._knowledge_access_audit(
                    session,
                    "workspace.created",
                    actor_user_id=actor_user_id,
                    workspace_id=workspace.id,
                    knowledge_space_id=knowledge_space.id,
                )
                await self._knowledge_access_audit(
                    session,
                    "space.created",
                    actor_user_id=actor_user_id,
                    workspace_id=workspace.id,
                    knowledge_space_id=knowledge_space.id,
                )
                return workspace
        except IntegrityError as exc:
            raise WorkspaceConflictError("Пространство с таким названием уже существует.") from exc

    async def ensure_personal_knowledge_space(self, actor_user_id: int) -> KnowledgeSpace:
        """Explicit lazy creation; never called by migration or user onboarding."""
        try:
            async with self.db.session() as session:
                await self._lock_user(session, actor_user_id)
                existing = await session.scalar(
                    select(KnowledgeSpace).where(
                        KnowledgeSpace.kind == "personal",
                        KnowledgeSpace.personal_owner_user_id == actor_user_id,
                    )
                )
                if existing is not None:
                    return existing
                space = KnowledgeSpace(
                    kind="personal",
                    personal_owner_user_id=actor_user_id,
                    status="active",
                    version=1,
                )
                session.add(space)
                await session.flush()
                await self._knowledge_access_audit(
                    session,
                    "space.created",
                    actor_user_id=actor_user_id,
                    workspace_id=None,
                    knowledge_space_id=space.id,
                )
                return space
        except IntegrityError as exc:
            raise WorkspaceConflictError("Не удалось создать личную область.") from exc

    async def access_context(self, actor_user_id: int, workspace_id: int) -> AccessContext:
        async with self.db.sessions() as session:
            row = (
                await session.execute(
                    select(Workspace.id, Workspace.access_epoch)
                    .join(
                        WorkspaceMember,
                        and_(
                            WorkspaceMember.workspace_id == Workspace.id,
                            WorkspaceMember.user_id == actor_user_id,
                            WorkspaceMember.status == "active",
                        ),
                    )
                    .where(Workspace.id == workspace_id)
                )
            ).one_or_none()
            if row is None:
                raise WorkspaceAccessDenied("Пространство недоступно.")
            return AccessContext(actor_user_id, row.id, row.access_epoch)

    async def list_workspaces(
        self,
        actor_user_id: int,
        *,
        page: int = 1,
        page_size: int = PAGE_SIZE,
        status: WorkspaceStatus | str = "active",
    ) -> WorkspacePage:
        if status not in {"active", "archived"}:
            raise WorkspaceAccessError("Некорректный статус пространства.")
        page, page_size = self._page(page, page_size)
        scope = and_(
            WorkspaceMember.workspace_id == Workspace.id,
            WorkspaceMember.user_id == actor_user_id,
            WorkspaceMember.status == "active",
        )
        async with self.db.sessions() as session:
            total = int(
                await session.scalar(
                    select(func.count(Workspace.id))
                    .join(WorkspaceMember, scope)
                    .where(Workspace.status == status)
                )
                or 0
            )
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            records = (
                await session.scalars(
                    select(Workspace)
                    .join(WorkspaceMember, scope)
                    .where(Workspace.status == status)
                    .order_by(Workspace.updated_at.desc(), Workspace.id.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
            ).all()
        return WorkspacePage(tuple(records), page, pages, total)

    async def get_workspace(self, context: AccessContext) -> Workspace:
        async with self.db.sessions() as session:
            workspace, _ = await self._read_access(session, context, roles=_ALL_ROLES)
            return workspace

    async def rename_workspace(
        self,
        context: AccessContext,
        expected_version: int,
        name: str,
        *,
        description: str | None = None,
        character: WorkspaceCharacter | str | None = None,
    ) -> Workspace:
        display, normalized = clean_workspace_name(name)
        clean_description = clean_workspace_description(description)
        clean_character = self._character(character) if character is not None else None
        try:
            async with self.db.session() as session:
                workspace, _ = await self._lock_access(session, context, roles=_OWNER_ROLES)
                if workspace.version != expected_version:
                    raise WorkspaceStaleError("Пространство уже изменилось.")
                workspace.name = display
                workspace.normalized_name = normalized
                workspace.description = clean_description
                if clean_character is not None:
                    workspace.character = clean_character
                workspace.version += 1
                # Recipient capabilities confirm the name/character shown in their preview.
                # Keep the raw invitation valid, but make every pre-change button stale so
                # the recipient must open a fresh preview before a terminal decision.
                await session.execute(
                    update(WorkspaceInvitation)
                    .where(
                        WorkspaceInvitation.workspace_id == workspace.id,
                        WorkspaceInvitation.status == "pending",
                    )
                    .values(version=WorkspaceInvitation.version + 1)
                    .execution_options(synchronize_session=False)
                )
                await session.flush()
                return workspace
        except IntegrityError as exc:
            raise WorkspaceConflictError("Пространство с таким названием уже существует.") from exc

    async def set_workspace_archived(
        self,
        context: AccessContext,
        expected_version: int,
        *,
        archived: bool,
    ) -> Workspace:
        expected_status = "active" if archived else "archived"
        new_status = "archived" if archived else "active"
        async with self.db.session() as session:
            workspace, _ = await self._lock_access(session, context, roles=_OWNER_ROLES)
            current = datetime.now(UTC)
            if workspace.version != expected_version or workspace.status != expected_status:
                raise WorkspaceStaleError("Пространство уже изменилось.")
            workspace.status = new_status
            workspace.version += 1
            workspace.access_epoch += 1
            if archived:
                await self._revoke_pending_invitations(session, workspace.id, current=current)
            await session.execute(
                update(KnowledgeSpace)
                .where(
                    KnowledgeSpace.workspace_id == workspace.id,
                    KnowledgeSpace.kind == "workspace",
                )
                .values(status=new_status, version=KnowledgeSpace.version + 1)
            )
            if archived:
                await session.execute(
                    update(KnowledgeSpace)
                    .where(
                        KnowledgeSpace.workspace_id == workspace.id,
                        KnowledgeSpace.kind == "project",
                    )
                    .values(status="archived", version=KnowledgeSpace.version + 1)
                )
            else:
                active_projects = select(WorkspaceProject.id).where(
                    WorkspaceProject.workspace_id == workspace.id,
                    WorkspaceProject.status == "active",
                )
                await session.execute(
                    update(KnowledgeSpace)
                    .where(
                        KnowledgeSpace.workspace_id == workspace.id,
                        KnowledgeSpace.kind == "project",
                        KnowledgeSpace.workspace_project_id.in_(active_projects),
                    )
                    .values(status="active", version=KnowledgeSpace.version + 1)
                )
            await self._knowledge_access_audit(
                session,
                "workspace.archived" if archived else "workspace.restored",
                actor_user_id=context.actor_user_id,
                workspace_id=workspace.id,
            )
            await session.flush()
            return workspace

    async def list_members(
        self, context: AccessContext, *, include_inactive: bool = False
    ) -> tuple[WorkspaceMemberRecord, ...]:
        actor_member = aliased(WorkspaceMember)
        actor_roles = _OWNER_ROLES if include_inactive else _ALL_ROLES
        async with self.db.sessions() as session:
            query = (
                select(WorkspaceMember, User.display_name)
                .join(User, User.id == WorkspaceMember.user_id)
                .join(Workspace, Workspace.id == WorkspaceMember.workspace_id)
                .join(
                    actor_member,
                    and_(
                        actor_member.workspace_id == Workspace.id,
                        actor_member.user_id == context.actor_user_id,
                        actor_member.status == "active",
                        actor_member.role.in_(actor_roles),
                    ),
                )
                .where(
                    Workspace.id == context.workspace_id,
                    Workspace.access_epoch == context.access_epoch,
                )
            )
            if not include_inactive:
                query = query.where(WorkspaceMember.status == "active")
            rows = (
                await session.execute(
                    query.order_by(
                        WorkspaceMember.status,
                        WorkspaceMember.role,
                        WorkspaceMember.joined_at,
                    )
                )
            ).all()
            if not rows:
                raise WorkspaceAccessDenied("Пространство недоступно.")
            return tuple(
                WorkspaceMemberRecord(member, display_name or "Участник")
                for member, display_name in rows
            )

    async def change_member_role(
        self,
        context: AccessContext,
        member_user_id: int,
        role: WorkspaceRole | str,
        expected_version: int,
    ) -> WorkspaceMember:
        clean_role = self._role(role)
        async with self.db.session() as session:
            workspace, _ = await self._lock_access(session, context, roles=_OWNER_ROLES)
            current = datetime.now(UTC)
            member = await session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == workspace.id,
                    WorkspaceMember.user_id == member_user_id,
                    WorkspaceMember.status == "active",
                    WorkspaceMember.version == expected_version,
                )
            )
            if member is None:
                raise WorkspaceStaleError("Участник уже изменился.")
            if member.role == clean_role:
                return member
            if member.role == "owner" and clean_role != "owner":
                await self._require_another_owner(session, workspace.id, member.user_id)
                await self._revoke_pending_invitations(
                    session,
                    workspace.id,
                    current=current,
                    inviter_user_id=member.user_id,
                )
            member.role = clean_role
            member.version += 1
            workspace.access_epoch += 1
            workspace.version += 1
            await self._knowledge_access_audit(
                session,
                "workspace.role_changed",
                actor_user_id=context.actor_user_id,
                workspace_id=workspace.id,
                safe_metadata={"role": clean_role},
            )
            await session.flush()
            return member

    async def revoke_member(
        self,
        context: AccessContext,
        member_user_id: int,
        expected_version: int,
    ) -> WorkspaceMember:
        async with self.db.session() as session:
            workspace, _ = await self._lock_access(session, context, roles=_OWNER_ROLES)
            current = datetime.now(UTC)
            member = await session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == workspace.id,
                    WorkspaceMember.user_id == member_user_id,
                    WorkspaceMember.status == "active",
                    WorkspaceMember.version == expected_version,
                )
            )
            if member is None:
                raise WorkspaceStaleError("Участник уже изменился.")
            if member.role == "owner":
                await self._require_another_owner(session, workspace.id, member.user_id)
            await self._revoke_pending_invitations(
                session,
                workspace.id,
                current=current,
                inviter_user_id=member.user_id,
            )
            member.status = "revoked"
            member.revoked_at = current
            member.version += 1
            workspace.access_epoch += 1
            workspace.version += 1
            await self._knowledge_access_audit(
                session,
                "workspace.member_revoked",
                actor_user_id=context.actor_user_id,
                workspace_id=workspace.id,
            )
            await session.flush()
            return member

    async def leave_workspace(
        self, context: AccessContext, expected_version: int
    ) -> WorkspaceMember:
        async with self.db.session() as session:
            workspace, actor = await self._lock_access(session, context, roles=_ALL_ROLES)
            current = datetime.now(UTC)
            if actor.version != expected_version:
                raise WorkspaceStaleError("Участие уже изменилось.")
            if actor.role == "owner":
                await self._require_another_owner(session, workspace.id, actor.user_id)
            await self._revoke_pending_invitations(
                session,
                workspace.id,
                current=current,
                inviter_user_id=actor.user_id,
            )
            actor.status = "left"
            actor.revoked_at = current
            actor.version += 1
            workspace.access_epoch += 1
            workspace.version += 1
            await self._knowledge_access_audit(
                session,
                "workspace.member_left",
                actor_user_id=context.actor_user_id,
                workspace_id=workspace.id,
            )
            await session.flush()
            return actor

    async def create_invitation(
        self,
        context: AccessContext,
        *,
        delivery_mode: InvitationDelivery | str,
        template_key: str,
        role: InvitationRole | str = "editor",
        intended_user_id: int | None = None,
        custom_text: str | None = None,
        ttl: timedelta | None = None,
    ) -> IssuedInvitation:
        clean_role = self._invitation_role(role)
        clean_delivery = self._delivery(delivery_mode)
        clean_template = _clean_key(template_key, field="шаблон приглашения")
        clean_text = clean_invitation_text(custom_text)
        invitation_ttl = self._invitation_ttl(ttl)
        if clean_delivery == "direct" and intended_user_id is None:
            raise WorkspaceInvitationError("Получатель приглашения недоступен.")
        if clean_delivery == "share" and intended_user_id is not None:
            raise WorkspaceInvitationError("Получатель приглашения недоступен.")
        token = secrets.token_urlsafe(32)
        try:
            async with self.db.session() as session:
                workspace, _ = await self._lock_access(
                    session, context, roles=_OWNER_ROLES, require_active=True
                )
                current = datetime.now(UTC)
                if intended_user_id is not None:
                    intended_exists = await session.scalar(
                        select(User.id).where(
                            User.id == intended_user_id,
                            User.id != context.actor_user_id,
                        )
                    )
                    already_member = await session.scalar(
                        select(WorkspaceMember.id).where(
                            WorkspaceMember.workspace_id == workspace.id,
                            WorkspaceMember.user_id == intended_user_id,
                            WorkspaceMember.status == "active",
                        )
                    )
                    if intended_exists is None or already_member is not None:
                        raise WorkspaceInvitationError("Получатель приглашения недоступен.")
                    await session.execute(
                        update(WorkspaceInvitation)
                        .where(
                            WorkspaceInvitation.workspace_id == workspace.id,
                            WorkspaceInvitation.intended_user_id == intended_user_id,
                            WorkspaceInvitation.status == "pending",
                        )
                        .values(
                            status="revoked",
                            revoked_at=current,
                            version=WorkspaceInvitation.version + 1,
                        )
                    )
                invitation = WorkspaceInvitation(
                    workspace_id=workspace.id,
                    inviter_user_id=context.actor_user_id,
                    intended_user_id=intended_user_id,
                    role=clean_role,
                    delivery_mode=clean_delivery,
                    template_key=clean_template,
                    custom_text=clean_text,
                    token_hash=_token_hash(token),
                    status="pending",
                    expires_at=current + invitation_ttl,
                    version=1,
                    created_at=current,
                    updated_at=current,
                )
                session.add(invitation)
                await session.flush()
                return IssuedInvitation(invitation, token)
        except IntegrityError as exc:
            raise WorkspaceConflictError("Не удалось создать приглашение.") from exc

    async def list_invitations(
        self,
        context: AccessContext,
        *,
        status: str = "pending",
    ) -> tuple[WorkspaceInvitation, ...]:
        if status not in {"pending", "accepted", "declined", "revoked", "expired"}:
            raise WorkspaceAccessError("Некорректный статус приглашения.")
        async with self.db.session() as session:
            await self._lock_access(session, context, roles=_OWNER_ROLES)
            current = datetime.now(UTC)
            await self._expire_invitations(session, context.workspace_id, current)
            return tuple(
                (
                    await session.scalars(
                        select(WorkspaceInvitation)
                        .where(
                            WorkspaceInvitation.workspace_id == context.workspace_id,
                            WorkspaceInvitation.status == status,
                        )
                        .order_by(WorkspaceInvitation.created_at.desc())
                    )
                ).all()
            )

    async def revoke_invitation(
        self,
        context: AccessContext,
        invitation_id: int,
        expected_version: int,
    ) -> WorkspaceInvitation:
        async with self.db.session() as session:
            await self._lock_access(session, context, roles=_OWNER_ROLES)
            current = datetime.now(UTC)
            invitation = await session.scalar(
                select(WorkspaceInvitation).where(
                    WorkspaceInvitation.id == invitation_id,
                    WorkspaceInvitation.workspace_id == context.workspace_id,
                    WorkspaceInvitation.status == "pending",
                    WorkspaceInvitation.version == expected_version,
                )
            )
            if invitation is None:
                raise WorkspaceStaleError("Приглашение уже недействительно.")
            invitation.status = "revoked"
            invitation.revoked_at = current
            invitation.version += 1
            await session.flush()
            return invitation

    async def renew_invitation(
        self,
        context: AccessContext,
        invitation_id: int,
        expected_version: int,
        *,
        ttl: timedelta | None = None,
    ) -> IssuedInvitation:
        invitation_ttl = self._invitation_ttl(ttl)
        token = secrets.token_urlsafe(32)
        try:
            async with self.db.session() as session:
                await self._lock_access(session, context, roles=_OWNER_ROLES, require_active=True)
                current = datetime.now(UTC)
                old = await session.scalar(
                    select(WorkspaceInvitation).where(
                        WorkspaceInvitation.id == invitation_id,
                        WorkspaceInvitation.workspace_id == context.workspace_id,
                        WorkspaceInvitation.status == "pending",
                        WorkspaceInvitation.version == expected_version,
                    )
                )
                if old is None:
                    raise WorkspaceStaleError("Приглашение уже недействительно.")
                if old.intended_user_id is not None:
                    intended_exists = await session.scalar(
                        select(User.id).where(User.id == old.intended_user_id)
                    )
                    if intended_exists is None:
                        raise WorkspaceInvitationError("Получатель приглашения недоступен.")
                old.status = "revoked"
                old.revoked_at = current
                old.version += 1
                renewed = WorkspaceInvitation(
                    workspace_id=old.workspace_id,
                    inviter_user_id=context.actor_user_id,
                    intended_user_id=old.intended_user_id,
                    role=old.role,
                    delivery_mode=old.delivery_mode,
                    template_key=old.template_key,
                    custom_text=old.custom_text,
                    token_hash=_token_hash(token),
                    status="pending",
                    expires_at=current + invitation_ttl,
                    version=1,
                    created_at=current,
                    updated_at=current,
                )
                session.add(renewed)
                await session.flush()
                return IssuedInvitation(renewed, token)
        except IntegrityError as exc:
            raise WorkspaceConflictError("Не удалось обновить приглашение.") from exc

    async def invitation_preview(
        self, actor_user_id: int, token: str, *, now: datetime | None = None
    ) -> InvitationPreview:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.sessions() as session:
            row = await self._invitation_row(session, actor_user_id, token, current)
            if row is None:
                raise WorkspaceInvitationError("Приглашение недействительно.")
            invitation, workspace, inviter_name = row
            return self._invitation_preview(invitation, workspace, inviter_name)

    async def accept_invitation(
        self, actor_user_id: int, token: str, *, now: datetime | None = None
    ) -> AccessContext:
        result = await self._mutate_raw_invitation(actor_user_id, token, action="accept", now=now)
        if result.access_context is None:
            raise WorkspaceInvitationError("Приглашение недействительно.")
        return result.access_context

    async def decline_invitation(
        self, actor_user_id: int, token: str, *, now: datetime | None = None
    ) -> WorkspaceInvitation:
        result = await self._mutate_raw_invitation(actor_user_id, token, action="decline", now=now)
        async with self.db.sessions() as session:
            invitation = await session.scalar(
                select(WorkspaceInvitation).where(
                    WorkspaceInvitation.token_hash == _token_hash(token),
                    WorkspaceInvitation.status == "declined",
                )
            )
            if invitation is None or result.status != "declined":
                raise WorkspaceInvitationError("Приглашение недействительно.")
            return invitation

    async def issue_incoming_actions(
        self,
        actor_user_id: int,
        chat_id: int,
        raw_invitation_token: str,
        *,
        actions: tuple[str, ...] = ("accept", "details", "later", "decline"),
        ttl: timedelta | None = None,
        now: datetime | None = None,
    ) -> IncomingInvitationActions:
        action_ttl = self._action_ttl(ttl)
        requested = tuple(dict.fromkeys(action.strip().casefold() for action in actions))
        if not requested or any(action not in INVITATION_ACTIONS for action in requested):
            raise WorkspaceAccessError("Некорректное действие приглашения.")
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            row = await self._invitation_row(session, actor_user_id, raw_invitation_token, current)
            if row is None:
                raise WorkspaceInvitationError("Приглашение недействительно.")
            invitation, workspace, inviter_name = row
            await self._cleanup_action_tokens(session, actor_user_id, chat_id, current)
            capabilities: dict[str, str] = {}
            for action in requested:
                raw_action_token = self._new_action_token()
                session.add(
                    WorkspaceActionToken(
                        token_hash=_token_hash(raw_action_token),
                        actor_user_id=actor_user_id,
                        chat_id=chat_id,
                        scope_kind="invitation",
                        workspace_id=workspace.id,
                        invitation_id=invitation.id,
                        invitation_version=invitation.version,
                        action=action,
                        status="pending",
                        expires_at=min(current + action_ttl, self._utc(invitation.expires_at)),
                    )
                )
                capabilities[action] = raw_action_token
            await session.flush()
            return IncomingInvitationActions(
                self._invitation_preview(invitation, workspace, inviter_name), capabilities
            )

    async def perform_invitation_action(
        self,
        capability_token: str,
        actor_user_id: int,
        chat_id: int,
        *,
        expected_action: str | None = None,
        now: datetime | None = None,
    ) -> InvitationActionResult:
        current = self._utc(now or datetime.now(UTC))
        action_hash = _token_hash(capability_token)
        async with self.db.session() as session:
            row = await self._incoming_action_row(
                session, action_hash, actor_user_id, chat_id, current
            )
            if row is None:
                raise WorkspaceInvitationError("Действие приглашения недействительно.")
            action_token, invitation, workspace, inviter_name = row
            if expected_action is not None and action_token.action != expected_action:
                raise WorkspaceInvitationError("Действие приглашения недействительно.")
            await self._lock_workspace_unscoped(session, workspace.id)
            current = self._utc(now or datetime.now(UTC))
            row = await self._incoming_action_row(
                session, action_hash, actor_user_id, chat_id, current
            )
            if row is None:
                raise WorkspaceInvitationError("Действие приглашения недействительно.")
            action_token, invitation, workspace, inviter_name = row
            claimed = await session.execute(
                update(WorkspaceActionToken)
                .where(
                    WorkspaceActionToken.token_hash == action_hash,
                    WorkspaceActionToken.status == "pending",
                )
                .values(status="consumed", consumed_at=current)
            )
            if claimed.rowcount != 1:
                raise WorkspaceInvitationError("Действие приглашения недействительно.")
            preview = self._invitation_preview(invitation, workspace, inviter_name)
            access_context: AccessContext | None = None
            if action_token.action in {"accept", "decline"}:
                access_context = await self._consume_invitation(
                    session,
                    invitation,
                    workspace,
                    actor_user_id,
                    action=action_token.action,
                    current=current,
                )
            await session.flush()
            status = (
                "accepted"
                if action_token.action == "accept"
                else "declined"
                if action_token.action == "decline"
                else "pending"
            )
            return InvitationActionResult(
                action_token.action, status, preview, access_context=access_context
            )

    async def accept_from_action(
        self, capability_token: str, actor_user_id: int, chat_id: int
    ) -> AccessContext:
        result = await self.perform_invitation_action(
            capability_token,
            actor_user_id,
            chat_id,
            expected_action="accept",
        )
        if result.access_context is None:
            raise WorkspaceInvitationError("Приглашение недействительно.")
        return result.access_context

    async def decline_from_action(
        self, capability_token: str, actor_user_id: int, chat_id: int
    ) -> InvitationActionResult:
        return await self.perform_invitation_action(
            capability_token,
            actor_user_id,
            chat_id,
            expected_action="decline",
        )

    async def create_project(self, context: AccessContext, name: str) -> WorkspaceProject:
        display, normalized = clean_workspace_name(name)
        try:
            async with self.db.session() as session:
                workspace, _ = await self._lock_access(
                    session, context, roles=_EDIT_ROLES, require_active=True
                )
                project = WorkspaceProject(
                    workspace_id=workspace.id,
                    name=display,
                    normalized_name=normalized,
                    status="active",
                    version=1,
                )
                session.add(project)
                await session.flush()
                knowledge_space = KnowledgeSpace(
                    kind="project",
                    workspace_id=workspace.id,
                    workspace_project_id=project.id,
                    status="active",
                    version=1,
                )
                session.add(knowledge_space)
                workspace.version += 1
                await session.flush()
                await self._knowledge_access_audit(
                    session,
                    "workspace.project_created",
                    actor_user_id=context.actor_user_id,
                    workspace_id=workspace.id,
                    knowledge_space_id=knowledge_space.id,
                )
                await self._knowledge_access_audit(
                    session,
                    "space.created",
                    actor_user_id=context.actor_user_id,
                    workspace_id=workspace.id,
                    knowledge_space_id=knowledge_space.id,
                )
                return project
        except IntegrityError as exc:
            raise WorkspaceConflictError("Проект с таким названием уже существует.") from exc

    async def list_projects(
        self,
        context: AccessContext,
        *,
        status: WorkspaceStatus | str = "active",
    ) -> tuple[WorkspaceProject, ...]:
        if status not in {"active", "archived"}:
            raise WorkspaceAccessError("Некорректный статус проекта.")
        async with self.db.sessions() as session:
            conditions = [
                Workspace.id == context.workspace_id,
                Workspace.access_epoch == context.access_epoch,
                WorkspaceMember.user_id == context.actor_user_id,
                WorkspaceMember.status == "active",
                WorkspaceMember.role.in_(_ALL_ROLES),
            ]
            if status == "active":
                conditions.append(Workspace.status == "active")
            rows = (
                (
                    await session.execute(
                        select(WorkspaceProject)
                        .select_from(Workspace)
                        .join(
                            WorkspaceMember,
                            WorkspaceMember.workspace_id == Workspace.id,
                        )
                        .outerjoin(
                            WorkspaceProject,
                            and_(
                                WorkspaceProject.workspace_id == Workspace.id,
                                WorkspaceProject.status == status,
                            ),
                        )
                        .where(*conditions)
                        .order_by(WorkspaceProject.updated_at.desc(), WorkspaceProject.id.desc())
                    )
                )
                .scalars()
                .all()
            )
            if not rows:
                raise WorkspaceAccessDenied("Пространство недоступно.")
            return tuple(project for project in rows if project is not None)

    async def get_project(
        self,
        context: AccessContext,
        workspace_project_id: int,
        *,
        require_active: bool = True,
    ) -> WorkspaceProject:
        async with self.db.sessions() as session:
            query = (
                select(WorkspaceProject)
                .join(Workspace, Workspace.id == WorkspaceProject.workspace_id)
                .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
                .where(
                    WorkspaceProject.id == workspace_project_id,
                    WorkspaceProject.workspace_id == context.workspace_id,
                    Workspace.id == context.workspace_id,
                    Workspace.access_epoch == context.access_epoch,
                    WorkspaceMember.user_id == context.actor_user_id,
                    WorkspaceMember.status == "active",
                    WorkspaceMember.role.in_(_ALL_ROLES),
                )
            )
            if require_active:
                query = query.where(
                    Workspace.status == "active",
                    WorkspaceProject.status == "active",
                )
            project = await session.scalar(query)
            if project is None:
                raise WorkspaceAccessDenied("Проект недоступен.")
            return project

    async def rename_project(
        self,
        context: AccessContext,
        workspace_project_id: int,
        expected_version: int,
        name: str,
    ) -> WorkspaceProject:
        display, normalized = clean_workspace_name(name)
        try:
            async with self.db.session() as session:
                await self._lock_access(session, context, roles=_EDIT_ROLES, require_active=True)
                project = await session.scalar(
                    select(WorkspaceProject).where(
                        WorkspaceProject.id == workspace_project_id,
                        WorkspaceProject.workspace_id == context.workspace_id,
                        WorkspaceProject.version == expected_version,
                    )
                )
                if project is None:
                    raise WorkspaceStaleError("Проект уже изменился.")
                project.name = display
                project.normalized_name = normalized
                project.version += 1
                await self._knowledge_access_audit(
                    session,
                    "workspace.project_renamed",
                    actor_user_id=context.actor_user_id,
                    workspace_id=context.workspace_id,
                    workspace_project_id=project.id,
                )
                await session.flush()
                return project
        except IntegrityError as exc:
            raise WorkspaceConflictError("Проект с таким названием уже существует.") from exc

    async def set_project_archived(
        self,
        context: AccessContext,
        workspace_project_id: int,
        expected_version: int,
        *,
        archived: bool,
    ) -> WorkspaceProject:
        expected_status = "active" if archived else "archived"
        new_status = "archived" if archived else "active"
        async with self.db.session() as session:
            await self._lock_access(session, context, roles=_EDIT_ROLES, require_active=True)
            project = await session.scalar(
                select(WorkspaceProject).where(
                    WorkspaceProject.id == workspace_project_id,
                    WorkspaceProject.workspace_id == context.workspace_id,
                    WorkspaceProject.version == expected_version,
                    WorkspaceProject.status == expected_status,
                )
            )
            if project is None:
                raise WorkspaceStaleError("Проект уже изменился.")
            project.status = new_status
            project.version += 1
            await session.execute(
                update(KnowledgeSpace)
                .where(
                    KnowledgeSpace.kind == "project",
                    KnowledgeSpace.workspace_id == context.workspace_id,
                    KnowledgeSpace.workspace_project_id == project.id,
                )
                .values(status=new_status, version=KnowledgeSpace.version + 1)
            )
            await self._knowledge_access_audit(
                session,
                "workspace.project_archived" if archived else "workspace.project_restored",
                actor_user_id=context.actor_user_id,
                workspace_id=context.workspace_id,
                workspace_project_id=project.id,
            )
            await session.flush()
            return project

    async def set_context(
        self,
        context: AccessContext,
        chat_id: int,
        *,
        workspace_project_id: int | None = None,
        ttl: timedelta | None = None,
    ) -> WorkspaceContextSnapshot:
        context_ttl = self._context_ttl(ttl)
        current = datetime.now(UTC)
        async with self.db.session() as session:
            await self._lock_access(session, context, roles=_ALL_ROLES, require_active=True)
            project: WorkspaceProject | None = None
            if workspace_project_id is not None:
                project = await session.scalar(
                    select(WorkspaceProject).where(
                        WorkspaceProject.id == workspace_project_id,
                        WorkspaceProject.workspace_id == context.workspace_id,
                        WorkspaceProject.status == "active",
                    )
                )
                if project is None:
                    raise WorkspaceAccessDenied("Проект недоступен.")
            stored = await session.scalar(
                select(WorkspaceContext).where(
                    WorkspaceContext.actor_user_id == context.actor_user_id,
                    WorkspaceContext.chat_id == chat_id,
                )
            )
            if stored is None:
                stored = WorkspaceContext(
                    actor_user_id=context.actor_user_id,
                    chat_id=chat_id,
                    workspace_id=context.workspace_id,
                    workspace_access_epoch=context.access_epoch,
                    workspace_project_id=project.id if project else None,
                    workspace_project_version=project.version if project else None,
                    version=1,
                    expires_at=current + context_ttl,
                )
                session.add(stored)
            else:
                stored.workspace_id = context.workspace_id
                stored.workspace_access_epoch = context.access_epoch
                stored.workspace_project_id = project.id if project else None
                stored.workspace_project_version = project.version if project else None
                stored.version += 1
                stored.expires_at = current + context_ttl
            await session.flush()
            return WorkspaceContextSnapshot(context, stored.version, project)

    async def active_context(
        self, actor_user_id: int, chat_id: int, *, now: datetime | None = None
    ) -> WorkspaceContextSnapshot | None:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.sessions() as session:
            row = (
                await session.execute(
                    select(WorkspaceContext, WorkspaceProject)
                    .join(
                        Workspace,
                        and_(
                            Workspace.id == WorkspaceContext.workspace_id,
                            Workspace.status == "active",
                            Workspace.access_epoch == WorkspaceContext.workspace_access_epoch,
                        ),
                    )
                    .join(
                        WorkspaceMember,
                        and_(
                            WorkspaceMember.workspace_id == Workspace.id,
                            WorkspaceMember.user_id == actor_user_id,
                            WorkspaceMember.status == "active",
                        ),
                    )
                    .outerjoin(
                        WorkspaceProject,
                        and_(
                            WorkspaceProject.id == WorkspaceContext.workspace_project_id,
                            WorkspaceProject.workspace_id == Workspace.id,
                        ),
                    )
                    .where(
                        WorkspaceContext.actor_user_id == actor_user_id,
                        WorkspaceContext.chat_id == chat_id,
                        WorkspaceContext.expires_at > current,
                        or_(
                            WorkspaceContext.workspace_project_id.is_(None),
                            and_(
                                WorkspaceProject.status == "active",
                                WorkspaceProject.version
                                == WorkspaceContext.workspace_project_version,
                            ),
                        ),
                    )
                )
            ).one_or_none()
            if row is None:
                return None
            stored, project = row
            access = AccessContext(
                actor_user_id, stored.workspace_id, stored.workspace_access_epoch
            )
            return WorkspaceContextSnapshot(access, stored.version, project)

    async def clear_context(self, actor_user_id: int, chat_id: int) -> bool:
        async with self.db.session() as session:
            result = await session.execute(
                delete(WorkspaceContext).where(
                    WorkspaceContext.actor_user_id == actor_user_id,
                    WorkspaceContext.chat_id == chat_id,
                )
            )
            return result.rowcount == 1

    async def issue_action(
        self,
        actor_user_id: int,
        chat_id: int,
        action: str,
        *,
        payload: dict[str, Any] | None = None,
        context: AccessContext | None = None,
        workspace_version: int | None = None,
        workspace_project_id: int | None = None,
        workspace_project_version: int | None = None,
        ttl: timedelta | None = None,
        initial_status: Literal["pending", "awaiting_input"] = "pending",
    ) -> str:
        clean_action = _clean_action(action)
        clean_payload = self._payload(payload)
        action_ttl = self._action_ttl(ttl)
        if initial_status not in {"pending", "awaiting_input"}:
            raise WorkspaceAccessError("Некорректный статус действия.")
        if (initial_status == "awaiting_input") != clean_action.startswith("input:"):
            raise WorkspaceAccessError("Некорректный тип действия ввода.")
        raw_token = self._new_action_token()
        async with self.db.session() as session:
            # This portable row update is the actor-scoped mutex for action issuance.
            # It prevents two PostgreSQL transactions from leaving two live inputs.
            await self._lock_user(session, actor_user_id)
            current = datetime.now(UTC)
            await self._cleanup_action_tokens(session, actor_user_id, chat_id, current)
            current = datetime.now(UTC)
            if context is None:
                if any(
                    value is not None
                    for value in (
                        workspace_version,
                        workspace_project_id,
                        workspace_project_version,
                    )
                ):
                    raise WorkspaceAccessError("Действие имеет некорректную область.")
                scope_kind = "wizard"
                workspace_id = None
                access_epoch = None
                bound_workspace_version = None
                workspace_status_snapshot = None
                project_status_snapshot = None
            else:
                if context.actor_user_id != actor_user_id:
                    raise WorkspaceAccessDenied("Действие недоступно.")
                workspace, _ = await self._read_access(session, context, roles=_ALL_ROLES)
                if workspace_version is not None and workspace.version != workspace_version:
                    raise WorkspaceStaleError("Пространство уже изменилось.")
                bound_workspace_version = workspace.version
                workspace_status_snapshot = workspace.status
                workspace_id = workspace.id
                access_epoch = workspace.access_epoch
                scope_kind = "workspace"
                if workspace_project_id is None and workspace_project_version is not None:
                    raise WorkspaceAccessError("Действие имеет некорректный проект.")
                if workspace_project_id is not None:
                    project = await session.scalar(
                        select(WorkspaceProject).where(
                            WorkspaceProject.id == workspace_project_id,
                            WorkspaceProject.workspace_id == workspace.id,
                        )
                    )
                    if project is None or (
                        workspace_project_version is not None
                        and project.version != workspace_project_version
                    ):
                        raise WorkspaceStaleError("Проект уже изменился.")
                    workspace_project_version = project.version
                    project_status_snapshot = project.status
                else:
                    project_status_snapshot = None
            if initial_status == "awaiting_input":
                await session.execute(
                    update(WorkspaceActionToken)
                    .where(
                        WorkspaceActionToken.actor_user_id == actor_user_id,
                        WorkspaceActionToken.chat_id == chat_id,
                        WorkspaceActionToken.status == "awaiting_input",
                        WorkspaceActionToken.action.like("input:%"),
                    )
                    .values(status="consumed", consumed_at=current)
                )
            session.add(
                WorkspaceActionToken(
                    token_hash=_token_hash(raw_token),
                    actor_user_id=actor_user_id,
                    chat_id=chat_id,
                    scope_kind=scope_kind,
                    workspace_id=workspace_id,
                    workspace_access_epoch=access_epoch,
                    workspace_version=bound_workspace_version,
                    workspace_status_snapshot=workspace_status_snapshot,
                    workspace_project_id=workspace_project_id,
                    workspace_project_version=workspace_project_version,
                    workspace_project_status_snapshot=project_status_snapshot,
                    action=clean_action,
                    payload=clean_payload,
                    status=initial_status,
                    expires_at=current + action_ttl,
                )
            )
            await session.flush()
            return raw_token

    async def claim_action(
        self,
        capability_token: str,
        actor_user_id: int,
        chat_id: int,
        *,
        expected_action: str | None = None,
        now: datetime | None = None,
    ) -> WorkspaceActionClaim | None:
        token_hash = _token_hash(capability_token)
        clean_expected = _clean_action(expected_action) if expected_action is not None else None
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            locked = await session.execute(
                update(WorkspaceActionToken)
                .where(
                    WorkspaceActionToken.token_hash == token_hash,
                    WorkspaceActionToken.actor_user_id == actor_user_id,
                    WorkspaceActionToken.chat_id == chat_id,
                    WorkspaceActionToken.status == "pending",
                    WorkspaceActionToken.scope_kind.in_(("wizard", "workspace")),
                )
                .values(status=WorkspaceActionToken.status)
            )
            if locked.rowcount != 1:
                return None
            current = self._utc(now or datetime.now(UTC))
            query = self._action_query(token_hash, actor_user_id, chat_id, current)
            row = (await session.execute(query)).one_or_none()
            if row is None:
                return None
            stored, workspace, _member, project = row
            if clean_expected is not None and stored.action != clean_expected:
                return None
            if workspace is not None:
                await self._lock_access(
                    session,
                    AccessContext(actor_user_id, workspace.id, stored.workspace_access_epoch),
                    roles=_ALL_ROLES,
                )
                current = self._utc(now or datetime.now(UTC))
                query = self._action_query(token_hash, actor_user_id, chat_id, current)
                row = (await session.execute(query)).one_or_none()
                if row is None:
                    return None
                stored, workspace, _member, project = row
            current = self._utc(now or datetime.now(UTC))
            claimed = await session.execute(
                update(WorkspaceActionToken)
                .where(
                    WorkspaceActionToken.token_hash == token_hash,
                    WorkspaceActionToken.actor_user_id == actor_user_id,
                    WorkspaceActionToken.chat_id == chat_id,
                    WorkspaceActionToken.scope_kind == stored.scope_kind,
                    WorkspaceActionToken.action == stored.action,
                    WorkspaceActionToken.status == "pending",
                    WorkspaceActionToken.expires_at > current,
                )
                .values(status="consumed", consumed_at=current)
                .execution_options(synchronize_session=False)
            )
            if claimed.rowcount != 1:
                return None
            access_context = (
                AccessContext(actor_user_id, workspace.id, workspace.access_epoch)
                if workspace is not None
                else None
            )
            return WorkspaceActionClaim(
                stored.action,
                dict(stored.payload or {}),
                access_context,
                stored.workspace_version,
                project.id if project is not None else None,
                stored.workspace_project_version,
            )

    async def begin_input(
        self,
        actor_user_id: int,
        chat_id: int,
        action: str,
        *,
        payload: dict[str, Any] | None = None,
        context: AccessContext | None = None,
        workspace_version: int | None = None,
        workspace_project_id: int | None = None,
        workspace_project_version: int | None = None,
        ttl: timedelta | None = None,
    ) -> str:
        clean = _clean_action(action)
        if not clean.startswith("input:"):
            clean = f"input:{clean}"
        return await self.issue_action(
            actor_user_id,
            chat_id,
            clean,
            payload=payload,
            context=context,
            workspace_version=workspace_version,
            workspace_project_id=workspace_project_id,
            workspace_project_version=workspace_project_version,
            ttl=ttl,
            initial_status="awaiting_input",
        )

    async def pending_input(
        self,
        actor_user_id: int,
        chat_id: int,
        *,
        now: datetime | None = None,
    ) -> WorkspaceActionClaim | None:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.sessions() as session:
            row = await self._input_action_row(
                session, actor_user_id, chat_id, current, expected_action=None
            )
            return self._action_claim(row, actor_user_id) if row is not None else None

    async def claim_pending_input(
        self,
        actor_user_id: int,
        chat_id: int,
        expected_action: str,
        *,
        now: datetime | None = None,
    ) -> WorkspaceActionClaim | None:
        clean = _clean_action(expected_action)
        if not clean.startswith("input:"):
            clean = f"input:{clean}"
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            current = self._utc(now or datetime.now(UTC))
            row = await self._input_action_row(
                session, actor_user_id, chat_id, current, expected_action=clean
            )
            if row is None:
                return None
            stored, workspace, _member, _project = row
            locked = await session.execute(
                update(WorkspaceActionToken)
                .where(
                    WorkspaceActionToken.token_hash == stored.token_hash,
                    WorkspaceActionToken.actor_user_id == actor_user_id,
                    WorkspaceActionToken.chat_id == chat_id,
                    WorkspaceActionToken.status == "awaiting_input",
                    WorkspaceActionToken.action == clean,
                )
                .values(status=WorkspaceActionToken.status)
            )
            if locked.rowcount != 1:
                return None
            current = self._utc(now or datetime.now(UTC))
            row = await self._input_action_row(
                session, actor_user_id, chat_id, current, expected_action=clean
            )
            if row is None:
                return None
            stored, workspace, _member, _project = row
            if workspace is not None:
                await self._lock_access(
                    session,
                    AccessContext(actor_user_id, workspace.id, stored.workspace_access_epoch),
                    roles=_ALL_ROLES,
                )
                current = self._utc(now or datetime.now(UTC))
                row = await self._input_action_row(
                    session, actor_user_id, chat_id, current, expected_action=clean
                )
                if row is None:
                    return None
                stored, _workspace, _member, _project = row
            current = self._utc(now or datetime.now(UTC))
            claimed = await session.execute(
                update(WorkspaceActionToken)
                .where(
                    WorkspaceActionToken.token_hash == stored.token_hash,
                    WorkspaceActionToken.actor_user_id == actor_user_id,
                    WorkspaceActionToken.chat_id == chat_id,
                    WorkspaceActionToken.scope_kind == stored.scope_kind,
                    WorkspaceActionToken.action == clean,
                    WorkspaceActionToken.status == "awaiting_input",
                    WorkspaceActionToken.expires_at > current,
                )
                .values(status="consumed", consumed_at=current)
                .execution_options(synchronize_session=False)
            )
            if claimed.rowcount != 1:
                return None
            return self._action_claim(row, actor_user_id)

    async def cancel_input(self, actor_user_id: int, chat_id: int) -> bool:
        async with self.db.session() as session:
            await self._lock_user(session, actor_user_id)
            current = datetime.now(UTC)
            cancelled = await session.execute(
                update(WorkspaceActionToken)
                .where(
                    WorkspaceActionToken.actor_user_id == actor_user_id,
                    WorkspaceActionToken.chat_id == chat_id,
                    WorkspaceActionToken.status == "awaiting_input",
                    WorkspaceActionToken.action.like("input:%"),
                )
                .values(status="consumed", consumed_at=current)
            )
            return bool(cancelled.rowcount)

    @staticmethod
    def _action_query(token_hash: str, actor_user_id: int, chat_id: int, current: datetime) -> Any:
        return (
            select(WorkspaceActionToken, Workspace, WorkspaceMember, WorkspaceProject)
            .outerjoin(Workspace, Workspace.id == WorkspaceActionToken.workspace_id)
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
                    WorkspaceProject.id == WorkspaceActionToken.workspace_project_id,
                    WorkspaceProject.workspace_id == Workspace.id,
                ),
            )
            .where(
                WorkspaceActionToken.token_hash == token_hash,
                WorkspaceActionToken.actor_user_id == actor_user_id,
                WorkspaceActionToken.chat_id == chat_id,
                WorkspaceActionToken.status == "pending",
                WorkspaceActionToken.expires_at > current,
                WorkspaceActionToken.scope_kind.in_(("wizard", "workspace")),
                or_(
                    WorkspaceActionToken.scope_kind == "wizard",
                    and_(
                        WorkspaceActionToken.scope_kind == "workspace",
                        Workspace.status == WorkspaceActionToken.workspace_status_snapshot,
                        Workspace.access_epoch == WorkspaceActionToken.workspace_access_epoch,
                        Workspace.version == WorkspaceActionToken.workspace_version,
                        WorkspaceMember.status == "active",
                        or_(
                            WorkspaceActionToken.workspace_project_id.is_(None),
                            and_(
                                WorkspaceProject.status
                                == WorkspaceActionToken.workspace_project_status_snapshot,
                                WorkspaceProject.version
                                == WorkspaceActionToken.workspace_project_version,
                            ),
                        ),
                    ),
                ),
            )
        )

    async def _input_action_row(
        self,
        session: AsyncSession,
        actor_user_id: int,
        chat_id: int,
        current: datetime,
        *,
        expected_action: str | None,
    ) -> (
        tuple[
            WorkspaceActionToken,
            Workspace | None,
            WorkspaceMember | None,
            WorkspaceProject | None,
        ]
        | None
    ):
        query = (
            select(WorkspaceActionToken, Workspace, WorkspaceMember, WorkspaceProject)
            .outerjoin(Workspace, Workspace.id == WorkspaceActionToken.workspace_id)
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
                    WorkspaceProject.id == WorkspaceActionToken.workspace_project_id,
                    WorkspaceProject.workspace_id == Workspace.id,
                ),
            )
            .where(
                WorkspaceActionToken.actor_user_id == actor_user_id,
                WorkspaceActionToken.chat_id == chat_id,
                WorkspaceActionToken.status == "awaiting_input",
                WorkspaceActionToken.action.like("input:%"),
                WorkspaceActionToken.expires_at > current,
                WorkspaceActionToken.scope_kind.in_(("wizard", "workspace")),
                or_(
                    WorkspaceActionToken.scope_kind == "wizard",
                    and_(
                        WorkspaceActionToken.scope_kind == "workspace",
                        Workspace.status == WorkspaceActionToken.workspace_status_snapshot,
                        Workspace.access_epoch == WorkspaceActionToken.workspace_access_epoch,
                        Workspace.version == WorkspaceActionToken.workspace_version,
                        WorkspaceMember.status == "active",
                        or_(
                            WorkspaceActionToken.workspace_project_id.is_(None),
                            and_(
                                WorkspaceProject.status
                                == WorkspaceActionToken.workspace_project_status_snapshot,
                                WorkspaceProject.version
                                == WorkspaceActionToken.workspace_project_version,
                            ),
                        ),
                    ),
                ),
            )
            .order_by(WorkspaceActionToken.created_at.desc())
            .limit(1)
        )
        if expected_action is not None:
            query = query.where(WorkspaceActionToken.action == expected_action)
        return (await session.execute(query)).one_or_none()

    @staticmethod
    def _action_claim(
        row: tuple[
            WorkspaceActionToken,
            Workspace | None,
            WorkspaceMember | None,
            WorkspaceProject | None,
        ],
        actor_user_id: int,
    ) -> WorkspaceActionClaim:
        stored, workspace, _member, project = row
        access_context = (
            AccessContext(actor_user_id, workspace.id, workspace.access_epoch)
            if workspace is not None
            else None
        )
        return WorkspaceActionClaim(
            stored.action,
            dict(stored.payload or {}),
            access_context,
            stored.workspace_version,
            project.id if project is not None else None,
            stored.workspace_project_version,
        )

    async def _mutate_raw_invitation(
        self,
        actor_user_id: int,
        raw_token: str,
        *,
        action: Literal["accept", "decline"],
        now: datetime | None,
    ) -> InvitationActionResult:
        current = self._utc(now or datetime.now(UTC))
        async with self.db.session() as session:
            row = await self._invitation_row(session, actor_user_id, raw_token, current)
            if row is None:
                raise WorkspaceInvitationError("Приглашение недействительно.")
            invitation, workspace, inviter_name = row
            await self._lock_workspace_unscoped(session, workspace.id)
            current = self._utc(now or datetime.now(UTC))
            row = await self._invitation_row(session, actor_user_id, raw_token, current)
            if row is None:
                raise WorkspaceInvitationError("Приглашение недействительно.")
            invitation, workspace, inviter_name = row
            preview = self._invitation_preview(invitation, workspace, inviter_name)
            access = await self._consume_invitation(
                session,
                invitation,
                workspace,
                actor_user_id,
                action=action,
                current=current,
            )
            await session.flush()
            status = "accepted" if action == "accept" else "declined"
            return InvitationActionResult(action, status, preview, access)

    async def _consume_invitation(
        self,
        session: AsyncSession,
        invitation: WorkspaceInvitation,
        workspace: Workspace,
        actor_user_id: int,
        *,
        action: Literal["accept", "decline"],
        current: datetime,
    ) -> AccessContext | None:
        intended_scope = (
            WorkspaceInvitation.intended_user_id.is_(None)
            if invitation.delivery_mode == "share"
            else WorkspaceInvitation.intended_user_id == actor_user_id
        )
        consumed = await session.execute(
            update(WorkspaceInvitation)
            .where(
                WorkspaceInvitation.id == invitation.id,
                WorkspaceInvitation.workspace_id == workspace.id,
                WorkspaceInvitation.version == invitation.version,
                WorkspaceInvitation.status == "pending",
                WorkspaceInvitation.expires_at > current,
                intended_scope,
            )
            .values(
                status="accepted" if action == "accept" else "declined",
                consumed_at=current,
                version=WorkspaceInvitation.version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        if consumed.rowcount != 1:
            raise WorkspaceInvitationError("Приглашение недействительно.")
        if action == "decline":
            return None
        actor_exists = await session.scalar(select(User.id).where(User.id == actor_user_id))
        existing = await session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == actor_user_id,
            )
        )
        if actor_exists is None or (existing is not None and existing.status == "active"):
            raise WorkspaceInvitationError("Приглашение недействительно.")
        if existing is None:
            session.add(
                WorkspaceMember(
                    workspace_id=workspace.id,
                    user_id=actor_user_id,
                    role=invitation.role,
                    status="active",
                    invited_by_user_id=invitation.inviter_user_id,
                    joined_at=current,
                    version=1,
                )
            )
        else:
            existing.role = invitation.role
            existing.status = "active"
            existing.invited_by_user_id = invitation.inviter_user_id
            existing.joined_at = current
            existing.revoked_at = None
            existing.version += 1
        workspace.access_epoch += 1
        workspace.version += 1
        await self._knowledge_access_audit(
            session,
            "workspace.member_added",
            actor_user_id=actor_user_id,
            workspace_id=workspace.id,
            safe_metadata={"role": invitation.role},
        )
        await session.flush()
        return AccessContext(actor_user_id, workspace.id, workspace.access_epoch)

    async def _invitation_row(
        self,
        session: AsyncSession,
        actor_user_id: int,
        raw_token: str,
        current: datetime,
    ) -> tuple[WorkspaceInvitation, Workspace, str | None] | None:
        active_membership = exists(
            select(WorkspaceMember.id).where(
                WorkspaceMember.workspace_id == WorkspaceInvitation.workspace_id,
                WorkspaceMember.user_id == actor_user_id,
                WorkspaceMember.status == "active",
            )
        )
        stale_revoked_membership = exists(
            select(WorkspaceMember.id).where(
                WorkspaceMember.workspace_id == WorkspaceInvitation.workspace_id,
                WorkspaceMember.user_id == actor_user_id,
                WorkspaceMember.status.in_(("revoked", "left")),
                WorkspaceMember.revoked_at >= WorkspaceInvitation.created_at,
            )
        )
        return (
            await session.execute(
                select(WorkspaceInvitation, Workspace, User.display_name)
                .join(
                    Workspace,
                    and_(
                        Workspace.id == WorkspaceInvitation.workspace_id,
                        Workspace.status == "active",
                    ),
                )
                .join(User, User.id == WorkspaceInvitation.inviter_user_id)
                .where(
                    WorkspaceInvitation.token_hash == _token_hash(raw_token),
                    WorkspaceInvitation.status == "pending",
                    WorkspaceInvitation.expires_at > current,
                    or_(
                        and_(
                            WorkspaceInvitation.delivery_mode == "direct",
                            WorkspaceInvitation.intended_user_id == actor_user_id,
                        ),
                        and_(
                            WorkspaceInvitation.delivery_mode == "share",
                            WorkspaceInvitation.intended_user_id.is_(None),
                        ),
                    ),
                    ~active_membership,
                    ~stale_revoked_membership,
                )
            )
        ).one_or_none()

    async def _incoming_action_row(
        self,
        session: AsyncSession,
        action_hash: str,
        actor_user_id: int,
        chat_id: int,
        current: datetime,
    ) -> (
        tuple[
            WorkspaceActionToken,
            WorkspaceInvitation,
            Workspace,
            str | None,
        ]
        | None
    ):
        active_membership = exists(
            select(WorkspaceMember.id).where(
                WorkspaceMember.workspace_id == WorkspaceInvitation.workspace_id,
                WorkspaceMember.user_id == actor_user_id,
                WorkspaceMember.status == "active",
            )
        )
        stale_revoked_membership = exists(
            select(WorkspaceMember.id).where(
                WorkspaceMember.workspace_id == WorkspaceInvitation.workspace_id,
                WorkspaceMember.user_id == actor_user_id,
                WorkspaceMember.status.in_(("revoked", "left")),
                WorkspaceMember.revoked_at >= WorkspaceInvitation.created_at,
            )
        )
        return (
            await session.execute(
                select(
                    WorkspaceActionToken,
                    WorkspaceInvitation,
                    Workspace,
                    User.display_name,
                )
                .join(
                    WorkspaceInvitation,
                    and_(
                        WorkspaceInvitation.id == WorkspaceActionToken.invitation_id,
                        WorkspaceInvitation.workspace_id == WorkspaceActionToken.workspace_id,
                        WorkspaceInvitation.version == WorkspaceActionToken.invitation_version,
                    ),
                )
                .join(
                    Workspace,
                    and_(
                        Workspace.id == WorkspaceInvitation.workspace_id,
                        Workspace.status == "active",
                    ),
                )
                .join(User, User.id == WorkspaceInvitation.inviter_user_id)
                .where(
                    WorkspaceActionToken.token_hash == action_hash,
                    WorkspaceActionToken.actor_user_id == actor_user_id,
                    WorkspaceActionToken.chat_id == chat_id,
                    WorkspaceActionToken.scope_kind == "invitation",
                    WorkspaceActionToken.status == "pending",
                    WorkspaceActionToken.expires_at > current,
                    WorkspaceInvitation.status == "pending",
                    WorkspaceInvitation.expires_at > current,
                    or_(
                        and_(
                            WorkspaceInvitation.delivery_mode == "direct",
                            WorkspaceInvitation.intended_user_id == actor_user_id,
                        ),
                        and_(
                            WorkspaceInvitation.delivery_mode == "share",
                            WorkspaceInvitation.intended_user_id.is_(None),
                        ),
                    ),
                    ~active_membership,
                    ~stale_revoked_membership,
                )
            )
        ).one_or_none()

    async def _read_access(
        self,
        session: AsyncSession,
        context: AccessContext,
        *,
        roles: frozenset[str],
        require_active: bool = False,
    ) -> tuple[Workspace, WorkspaceMember]:
        query = (
            select(Workspace, WorkspaceMember)
            .join(
                WorkspaceMember,
                and_(
                    WorkspaceMember.workspace_id == Workspace.id,
                    WorkspaceMember.user_id == context.actor_user_id,
                    WorkspaceMember.status == "active",
                    WorkspaceMember.role.in_(roles),
                ),
            )
            .where(
                Workspace.id == context.workspace_id,
                Workspace.access_epoch == context.access_epoch,
            )
        )
        if require_active:
            query = query.where(Workspace.status == "active")
        row = (await session.execute(query)).one_or_none()
        if row is None:
            raise WorkspaceAccessDenied("Пространство недоступно.")
        return row

    async def _lock_access(
        self,
        session: AsyncSession,
        context: AccessContext,
        *,
        roles: frozenset[str],
        require_active: bool = False,
    ) -> tuple[Workspace, WorkspaceMember]:
        member_scope = exists(
            select(WorkspaceMember.id).where(
                WorkspaceMember.workspace_id == Workspace.id,
                WorkspaceMember.user_id == context.actor_user_id,
                WorkspaceMember.status == "active",
                WorkspaceMember.role.in_(roles),
            )
        )
        conditions = [
            Workspace.id == context.workspace_id,
            Workspace.access_epoch == context.access_epoch,
            member_scope,
        ]
        if require_active:
            conditions.append(Workspace.status == "active")
        locked = await session.execute(
            update(Workspace).where(*conditions).values(updated_at=Workspace.updated_at)
        )
        if locked.rowcount != 1:
            raise WorkspaceAccessDenied("Пространство недоступно.")
        return await self._read_access(session, context, roles=roles, require_active=require_active)

    @staticmethod
    async def _knowledge_access_audit(
        session: AsyncSession,
        event_type: str,
        *,
        actor_user_id: int,
        workspace_id: int | None,
        knowledge_space_id: int | None = None,
        workspace_project_id: int | None = None,
        safe_metadata: dict[str, Any] | None = None,
    ) -> None:
        if knowledge_space_id is None and workspace_id is not None:
            conditions = [KnowledgeSpace.workspace_id == workspace_id]
            if workspace_project_id is None:
                conditions.extend(
                    [
                        KnowledgeSpace.kind == "workspace",
                        KnowledgeSpace.workspace_project_id.is_(None),
                    ]
                )
            else:
                conditions.extend(
                    [
                        KnowledgeSpace.kind == "project",
                        KnowledgeSpace.workspace_project_id == workspace_project_id,
                    ]
                )
            knowledge_space_id = await session.scalar(select(KnowledgeSpace.id).where(*conditions))
        metadata = None
        if safe_metadata is not None:
            role = safe_metadata.get("role")
            if set(safe_metadata) != {"role"} or role not in WORKSPACE_ROLES:
                raise WorkspaceAccessError("Некорректные audit metadata.")
            metadata = {"role": role}
        session.add(
            KnowledgeAuditEvent(
                public_id=str(uuid4()),
                event_type=event_type,
                actor_user_id=actor_user_id,
                workspace_id=workspace_id,
                knowledge_space_id=knowledge_space_id,
                safe_metadata=metadata,
            )
        )

    @staticmethod
    async def _lock_workspace_unscoped(session: AsyncSession, workspace_id: int) -> None:
        locked = await session.execute(
            update(Workspace)
            .where(Workspace.id == workspace_id, Workspace.status == "active")
            .values(updated_at=Workspace.updated_at)
        )
        if locked.rowcount != 1:
            raise WorkspaceInvitationError("Приглашение недействительно.")

    @staticmethod
    async def _require_another_owner(
        session: AsyncSession, workspace_id: int, excluded_user_id: int
    ) -> None:
        owners = int(
            await session.scalar(
                select(func.count(WorkspaceMember.id)).where(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.status == "active",
                    WorkspaceMember.role == "owner",
                    WorkspaceMember.user_id != excluded_user_id,
                )
            )
            or 0
        )
        if owners < 1:
            raise WorkspaceLastOwnerError("Последний владелец не может покинуть пространство.")

    @staticmethod
    async def _expire_invitations(
        session: AsyncSession, workspace_id: int, current: datetime
    ) -> None:
        await session.execute(
            update(WorkspaceInvitation)
            .where(
                WorkspaceInvitation.workspace_id == workspace_id,
                WorkspaceInvitation.status == "pending",
                WorkspaceInvitation.expires_at <= current,
            )
            .values(status="expired", version=WorkspaceInvitation.version + 1)
        )

    @staticmethod
    async def _revoke_pending_invitations(
        session: AsyncSession,
        workspace_id: int,
        *,
        current: datetime,
        inviter_user_id: int | None = None,
    ) -> None:
        query = update(WorkspaceInvitation).where(
            WorkspaceInvitation.workspace_id == workspace_id,
            WorkspaceInvitation.status == "pending",
        )
        if inviter_user_id is not None:
            query = query.where(WorkspaceInvitation.inviter_user_id == inviter_user_id)
        await session.execute(
            query.values(
                status="revoked",
                revoked_at=current,
                version=WorkspaceInvitation.version + 1,
            ).execution_options(synchronize_session=False)
        )

    @staticmethod
    async def _cleanup_action_tokens(
        session: AsyncSession,
        actor_user_id: int,
        chat_id: int,
        current: datetime,
    ) -> None:
        await session.execute(
            delete(WorkspaceActionToken).where(
                WorkspaceActionToken.actor_user_id == actor_user_id,
                WorkspaceActionToken.chat_id == chat_id,
                or_(
                    WorkspaceActionToken.expires_at <= current,
                    WorkspaceActionToken.status == "consumed",
                ),
            )
        )

    @staticmethod
    async def _lock_user(session: AsyncSession, user_id: int) -> None:
        locked = await session.execute(
            update(User).where(User.id == user_id).values(updated_at=User.updated_at)
        )
        if locked.rowcount != 1:
            raise WorkspaceAccessDenied("Пользователь недоступен.")

    @staticmethod
    async def _require_user(session: AsyncSession, user_id: int) -> None:
        if await session.scalar(select(User.id).where(User.id == user_id)) is None:
            raise WorkspaceAccessDenied("Пользователь недоступен.")

    @staticmethod
    def _invitation_preview(
        invitation: WorkspaceInvitation,
        workspace: Workspace,
        inviter_name: str | None,
    ) -> InvitationPreview:
        return InvitationPreview(
            inviter_display_name=inviter_name or "Участник",
            workspace_name=workspace.name,
            character=workspace.character,
            role=invitation.role,
            template_key=invitation.template_key,
            custom_text=invitation.custom_text,
            expires_at=invitation.expires_at,
            version=invitation.version,
        )

    @staticmethod
    def _character(value: WorkspaceCharacter | str) -> str:
        clean = value.strip().casefold()
        if clean not in WORKSPACE_CHARACTERS:
            raise WorkspaceAccessError("Некорректный характер пространства.")
        return clean

    @staticmethod
    def _role(value: WorkspaceRole | str) -> str:
        clean = value.strip().casefold()
        if clean not in WORKSPACE_ROLES:
            raise WorkspaceAccessError("Некорректная роль.")
        return clean

    @staticmethod
    def _invitation_role(value: InvitationRole | str) -> str:
        clean = value.strip().casefold()
        if clean not in INVITATION_ROLES:
            raise WorkspaceAccessError("Некорректная роль приглашения.")
        return clean

    @staticmethod
    def _delivery(value: InvitationDelivery | str) -> str:
        clean = value.strip().casefold()
        if clean not in INVITATION_DELIVERY_MODES:
            raise WorkspaceAccessError("Некорректный способ приглашения.")
        return clean

    @staticmethod
    def _page(page: int, page_size: int) -> tuple[int, int]:
        if page < 1:
            raise WorkspaceAccessError("Некорректная страница.")
        if not 1 <= page_size <= 50:
            raise WorkspaceAccessError("Некорректный размер страницы.")
        return page, page_size

    def _invitation_ttl(self, value: timedelta | None) -> timedelta:
        ttl = value or self.INVITATION_TTL
        if not self.MIN_INVITATION_TTL <= ttl <= self.MAX_INVITATION_TTL:
            raise WorkspaceAccessError("Некорректный срок приглашения.")
        return ttl

    def _action_ttl(self, value: timedelta | None) -> timedelta:
        ttl = value or self.ACTION_TTL
        if not timedelta(seconds=1) <= ttl <= self.MAX_ACTION_TTL:
            raise WorkspaceAccessError("Некорректный срок действия.")
        return ttl

    def _context_ttl(self, value: timedelta | None) -> timedelta:
        ttl = value or self.CONTEXT_TTL
        if not timedelta(minutes=1) <= ttl <= self.MAX_CONTEXT_TTL:
            raise WorkspaceAccessError("Некорректный срок контекста.")
        return ttl

    @staticmethod
    def _payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise WorkspaceAccessError("Некорректные данные действия.") from exc
        if len(encoded.encode("utf-8")) > 4096:
            raise WorkspaceAccessError("Данные действия слишком велики.")
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise WorkspaceAccessError("Некорректные данные действия.")
        return decoded

    @staticmethod
    def _new_action_token() -> str:
        # 24 URL-safe characters, comfortably below Telegram callback_data limits.
        return secrets.token_urlsafe(18)

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


WorkspaceService = WorkspaceAccessService
