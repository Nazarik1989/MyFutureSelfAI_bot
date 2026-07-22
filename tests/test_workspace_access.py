from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update

from future_self.models import (
    KnowledgeSpace,
    User,
    Workspace,
    WorkspaceActionToken,
    WorkspaceInvitation,
)
from future_self.workspace_access import (
    AccessContext,
    WorkspaceAccessDenied,
    WorkspaceAccessService,
    WorkspaceConflictError,
    WorkspaceInvitationError,
    WorkspaceLastOwnerError,
)


async def make_users(db, count: int) -> tuple[User, ...]:
    async with db.session() as session:
        records = tuple(
            User(telegram_id=8_800_000 + index, display_name=f"User {index}")
            for index in range(1, count + 1)
        )
        session.add_all(records)
        await session.flush()
        return records


async def join_direct(
    service: WorkspaceAccessService,
    owner_context: AccessContext,
    target_user_id: int,
    *,
    role: str = "editor",
) -> AccessContext:
    issued = await service.create_invitation(
        owner_context,
        delivery_mode="direct",
        intended_user_id=target_user_id,
        role=role,
        template_key="neutral_1",
    )
    return await service.accept_invitation(target_user_id, issued.token)


@pytest.mark.parametrize("character", ["pair", "friends", "family", "team", "custom"])
async def test_workspace_creation_is_explicit_and_character_is_presentation_only(db, character):
    (owner,) = await make_users(db, 1)
    service = WorkspaceAccessService(db)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(Workspace.id))) == 0
        assert await session.scalar(select(func.count(KnowledgeSpace.id))) == 0

    workspace = await service.create_workspace(
        owner.id, character, f"  Future   {character}  ", "  A shared   plan  "
    )

    assert workspace.character == character
    assert workspace.name == f"Future {character}"
    assert workspace.description == "A shared plan"
    context = await service.access_context(owner.id, workspace.id)
    members = await service.list_members(context)
    assert [(record.member.role, record.member.status) for record in members] == [
        ("owner", "active")
    ]
    async with db.sessions() as session:
        knowledge = await session.scalar(
            select(KnowledgeSpace).where(KnowledgeSpace.workspace_id == workspace.id)
        )
        assert knowledge is not None
        assert (knowledge.kind, knowledge.status) == ("workspace", "active")


async def test_workspace_normalization_conflict_and_pagination_are_owner_scoped(db):
    first, second = await make_users(db, 2)
    service = WorkspaceAccessService(db)
    created = await service.create_workspace(first.id, "custom", "Ａlpha")
    assert created.normalized_name == "alpha"
    with pytest.raises(WorkspaceConflictError):
        await service.create_workspace(first.id, "team", "  ALPHA  ")
    other = await service.create_workspace(second.id, "team", "alpha")
    assert other.created_by_user_id == second.id
    for index in range(7):
        await service.create_workspace(first.id, "custom", f"Space {index}")

    page = await service.list_workspaces(first.id, page=99, page_size=3)
    assert (page.page, page.pages, page.total, len(page.items)) == (3, 3, 8, 2)


async def test_personal_knowledge_space_is_lazy_idempotent_and_does_not_link_old_content(db):
    (owner,) = await make_users(db, 1)
    service = WorkspaceAccessService(db)
    first = await service.ensure_personal_knowledge_space(owner.id)
    second = await service.ensure_personal_knowledge_space(owner.id)
    assert first.id == second.id
    assert (first.kind, first.personal_owner_user_id, first.workspace_id) == (
        "personal",
        owner.id,
        None,
    )
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(KnowledgeSpace.id)).where(KnowledgeSpace.kind == "personal")
            )
            == 1
        )


async def test_direct_invitation_is_hashed_bound_single_use_and_epoch_invalidating(db):
    owner, intended, wrong = await make_users(db, 3)
    service = WorkspaceAccessService(db)
    workspace = await service.create_workspace(owner.id, "pair", "Together")
    owner_context = await service.access_context(owner.id, workspace.id)
    issued = await service.create_invitation(
        owner_context,
        delivery_mode="direct",
        intended_user_id=intended.id,
        role="viewer",
        template_key="pair_2",
        custom_text="  Join   our future  ",
    )
    async with db.sessions() as session:
        stored = await session.get(WorkspaceInvitation, issued.invitation.id)
        assert stored is not None
        assert stored.token_hash != issued.token
        assert len(stored.token_hash) == 64

    with pytest.raises(WorkspaceInvitationError):
        await service.invitation_preview(wrong.id, issued.token)
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_invitation(wrong.id, issued.token)
    preview = await service.invitation_preview(intended.id, issued.token)
    assert (preview.role, preview.template_key, preview.custom_text) == (
        "viewer",
        "pair_2",
        "Join our future",
    )

    intended_context = await service.accept_invitation(intended.id, issued.token)
    assert intended_context.access_epoch == owner_context.access_epoch + 1
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_invitation(intended.id, issued.token)
    with pytest.raises(WorkspaceAccessDenied):
        await service.get_workspace(owner_context)
    refreshed_owner = await service.access_context(owner.id, workspace.id)
    members = await service.list_members(refreshed_owner)
    assert {record.member.role for record in members} == {"owner", "viewer"}


async def test_share_decline_expiry_and_incoming_action_capabilities(db):
    owner, recipient = await make_users(db, 2)
    service = WorkspaceAccessService(db)
    workspace = await service.create_workspace(owner.id, "friends", "Adventures")
    owner_context = await service.access_context(owner.id, workspace.id)
    declined = await service.create_invitation(
        owner_context, delivery_mode="share", template_key="friends_1"
    )
    result = await service.decline_invitation(recipient.id, declined.token)
    assert result.status == "declined"
    with pytest.raises(WorkspaceInvitationError):
        await service.decline_invitation(recipient.id, declined.token)

    issued = await service.create_invitation(
        owner_context,
        delivery_mode="share",
        template_key="friends_2",
        ttl=timedelta(minutes=5),
    )
    incoming = await service.issue_incoming_actions(
        recipient.id, 4001, issued.token, actions=("details", "accept")
    )
    assert set(incoming.actions) == {"details", "accept"}
    with pytest.raises(WorkspaceInvitationError):
        await service.perform_invitation_action(incoming.actions["accept"], recipient.id, 4002)
    details = await service.perform_invitation_action(
        incoming.actions["details"], recipient.id, 4001
    )
    assert (details.action, details.status, details.access_context) == (
        "details",
        "pending",
        None,
    )
    accepted = await service.accept_from_action(incoming.actions["accept"], recipient.id, 4001)
    assert accepted.actor_user_id == recipient.id
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_from_action(incoming.actions["accept"], recipient.id, 4001)


async def test_roles_projects_last_owner_and_immediate_revoke(db):
    owner, editor, viewer = await make_users(db, 3)
    service = WorkspaceAccessService(db)
    workspace = await service.create_workspace(owner.id, "team", "Team")
    owner_context = await service.access_context(owner.id, workspace.id)
    editor_context = await join_direct(service, owner_context, editor.id)
    owner_context = await service.access_context(owner.id, workspace.id)
    viewer_context = await join_direct(service, owner_context, viewer.id, role="viewer")
    owner_context = await service.access_context(owner.id, workspace.id)
    editor_context = await service.access_context(editor.id, workspace.id)
    stale_share = await service.create_invitation(
        owner_context, delivery_mode="share", template_key="team_1"
    )

    project = await service.create_project(editor_context, " Launch   Plan ")
    assert project.normalized_name == "launch plan"
    with pytest.raises(WorkspaceAccessDenied):
        await service.create_project(viewer_context, "Forbidden")
    with pytest.raises(WorkspaceConflictError):
        await service.create_project(owner_context, "LAUNCH PLAN")
    assert [item.id for item in await service.list_projects(viewer_context)] == [project.id]
    with pytest.raises(WorkspaceAccessDenied):
        await service.list_members(viewer_context, include_inactive=True)

    owner_member = next(
        record.member
        for record in await service.list_members(owner_context)
        if record.member.user_id == owner.id
    )
    with pytest.raises(WorkspaceLastOwnerError):
        await service.leave_workspace(owner_context, owner_member.version)

    editor_member = next(
        record.member
        for record in await service.list_members(owner_context)
        if record.member.user_id == editor.id
    )
    await service.change_member_role(owner_context, editor.id, "owner", editor_member.version)
    owner_context = await service.access_context(owner.id, workspace.id)
    owner_member = next(
        record.member
        for record in await service.list_members(owner_context)
        if record.member.user_id == owner.id
    )
    await service.leave_workspace(owner_context, owner_member.version)
    with pytest.raises(WorkspaceAccessDenied):
        await service.get_workspace(owner_context)

    editor_context = await service.access_context(editor.id, workspace.id)
    viewer_member = next(
        record.member
        for record in await service.list_members(editor_context)
        if record.member.user_id == viewer.id
    )
    await service.revoke_member(editor_context, viewer.id, viewer_member.version)
    with pytest.raises(WorkspaceAccessDenied):
        await service.get_workspace(viewer_context)
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_invitation(viewer.id, stale_share.token)
    assert await service.active_context(viewer.id, 99) is None
    editor_context = await service.access_context(editor.id, workspace.id)
    rejoin = await service.create_invitation(
        editor_context,
        delivery_mode="direct",
        intended_user_id=viewer.id,
        template_key="team_2",
    )
    rejoined = await service.accept_invitation(viewer.id, rejoin.token)
    assert rejoined.actor_user_id == viewer.id


async def test_context_and_hashed_actions_are_restart_safe_and_stale_fail_closed(db):
    owner, member = await make_users(db, 2)
    service = WorkspaceAccessService(db)
    workspace = await service.create_workspace(owner.id, "family", "Home")
    owner_context = await service.access_context(owner.id, workspace.id)
    member_context = await join_direct(service, owner_context, member.id)
    owner_context = await service.access_context(owner.id, workspace.id)
    context = await service.set_context(member_context, 7001)
    assert (await service.active_context(member.id, 7001)) == context

    first = await service.begin_input(member.id, 7001, "rename", payload={"step": 1})
    second = await service.begin_input(member.id, 7001, "rename", payload={"step": 2})
    assert first != second
    pending = await service.pending_input(member.id, 7001)
    assert pending is not None and pending.payload == {"step": 2}
    assert await service.pending_input(member.id, 7002) is None
    claimed = await service.claim_pending_input(member.id, 7001, "rename")
    assert claimed is not None and claimed.action == "input:rename"
    assert await service.pending_input(member.id, 7001) is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WorkspaceActionToken.token_hash))) == 2
        assert (
            await session.scalar(
                select(func.count(WorkspaceActionToken.token_hash)).where(
                    WorkspaceActionToken.status == "awaiting_input"
                )
            )
            == 0
        )

    await service.issue_action(
        member.id,
        7001,
        "open",
        context=member_context,
        workspace_version=workspace.version + 1,
    )
    async with db.session() as session:
        await session.execute(
            update(WorkspaceActionToken)
            .where(
                WorkspaceActionToken.actor_user_id == member.id,
                WorkspaceActionToken.chat_id == 7001,
                WorkspaceActionToken.status == "pending",
            )
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    capability = await service.issue_action(
        member.id,
        7001,
        "open",
        context=member_context,
        workspace_version=workspace.version + 1,
    )
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(WorkspaceActionToken.token_hash)).where(
                    WorkspaceActionToken.actor_user_id == member.id,
                    WorkspaceActionToken.chat_id == 7001,
                )
            )
            == 1
        )
    member_row = next(
        record.member
        for record in await service.list_members(owner_context)
        if record.member.user_id == member.id
    )
    await service.revoke_member(owner_context, member.id, member_row.version)
    assert await service.claim_action(capability, member.id, 7001) is None
    assert await service.active_context(member.id, 7001) is None
    cleanup = await service.cleanup(now=datetime.now(UTC) + timedelta(days=8))
    assert cleanup.action_tokens >= 1
    assert cleanup.contexts >= 1


async def test_archive_revokes_pending_invites_and_has_owner_restore_capability(db):
    owner, recipient = await make_users(db, 2)
    service = WorkspaceAccessService(db)
    workspace = await service.create_workspace(owner.id, "custom", "Archive")
    context = await service.access_context(owner.id, workspace.id)
    project = await service.create_project(context, "Project")
    issued = await service.create_invitation(
        context, delivery_mode="share", template_key="custom_1"
    )
    incoming = await service.issue_incoming_actions(
        recipient.id, 991, issued.token, actions=("accept",)
    )
    current = await service.get_workspace(context)
    archived = await service.set_workspace_archived(context, current.version, archived=True)
    archived_context = await service.access_context(owner.id, workspace.id)
    capability = await service.issue_action(
        owner.id,
        100,
        "restore",
        context=archived_context,
        workspace_version=archived.version,
    )
    assert await service.claim_action(capability, owner.id, 100) is not None
    with pytest.raises(WorkspaceAccessDenied):
        await service.get_project(archived_context, project.id)
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_invitation(recipient.id, issued.token)
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_from_action(incoming.actions["accept"], recipient.id, 991)
    restored = await service.set_workspace_archived(
        archived_context, archived.version, archived=False
    )
    assert restored.status == "active"
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_invitation(recipient.id, issued.token)
