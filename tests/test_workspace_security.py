from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, func, select, update

from future_self.models import (
    KnowledgeSpace,
    User,
    Workspace,
    WorkspaceActionToken,
    WorkspaceContext,
    WorkspaceInvitation,
    WorkspaceMember,
)
from future_self.workspace_access import (
    AccessContext,
    WorkspaceAccessDenied,
    WorkspaceAccessError,
    WorkspaceAccessService,
    WorkspaceConflictError,
    WorkspaceInvitationError,
    WorkspaceLastOwnerError,
    WorkspaceStaleError,
    normalize_workspace_name,
)


async def create_user(db, telegram_id: int, display_name: str | None = None) -> User:
    async with db.session() as session:
        user = User(
            telegram_id=telegram_id,
            display_name=display_name,
            timezone="Europe/Moscow",
            onboarding_completed=True,
        )
        session.add(user)
        await session.flush()
        return user


async def create_space(
    service: WorkspaceAccessService,
    owner: User,
    *,
    name: str = "Наше будущее",
    character: str = "pair",
) -> tuple[Workspace, AccessContext]:
    workspace = await service.create_workspace(owner.id, character, name)
    return workspace, await service.access_context(owner.id, workspace.id)


async def direct_join(
    service: WorkspaceAccessService,
    owner_context: AccessContext,
    target: User,
    *,
    role: str = "editor",
) -> tuple[str, AccessContext]:
    issued = await service.create_invitation(
        owner_context,
        delivery_mode="direct",
        intended_user_id=target.id,
        role=role,
        template_key="direct_test",
    )
    return issued.token, await service.accept_invitation(target.id, issued.token)


async def invitation_status(db, token: str) -> str | None:
    async with db.sessions() as session:
        return await session.scalar(
            select(WorkspaceInvitation.status).where(
                WorkspaceInvitation.token_hash == hashlib.sha256(token.encode("utf-8")).hexdigest()
            )
        )


async def active_member_count(db, workspace_id: int, user_id: int) -> int:
    async with db.sessions() as session:
        return int(
            await session.scalar(
                select(func.count(WorkspaceMember.id)).where(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == user_id,
                    WorkspaceMember.status == "active",
                )
            )
            or 0
        )


class PausingWorkspaceAccessService(WorkspaceAccessService):
    """Pause one mutation before its workspace lock to control commit ordering."""

    def __init__(self, db):
        super().__init__(db)
        self.before_lock = asyncio.Event()
        self.release_lock = asyncio.Event()
        self._pause_once = True

    async def _lock_access(self, *args, **kwargs):
        if self._pause_once:
            self._pause_once = False
            self.before_lock.set()
            await self.release_lock.wait()
        return await super()._lock_access(*args, **kwargs)


async def test_explicit_creation_all_characters_normalization_and_no_seed(db):
    service = WorkspaceAccessService(db)
    owners = [await create_user(db, 910_000 + index) for index in range(5)]

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(Workspace.id))) == 0
        assert await session.scalar(select(func.count(KnowledgeSpace.id))) == 0

    for index, character in enumerate(("pair", "friends", "family", "team", "custom")):
        workspace = await service.create_workspace(
            owners[index].id,
            character,
            f"  Пространство\t{index}  ",
            "  Короткое\nописание  ",
        )
        assert workspace.character == character
        assert workspace.name == f"Пространство {index}"
        assert workspace.description == "Короткое описание"

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(Workspace.id))) == 5
        assert await session.scalar(select(func.count(WorkspaceMember.id))) == 5
        assert await session.scalar(select(func.count(KnowledgeSpace.id))) == 5

    personal = await service.ensure_personal_knowledge_space(owners[0].id)
    repeated = await service.ensure_personal_knowledge_space(owners[0].id)
    assert repeated.id == personal.id
    assert personal.kind == "personal"


async def test_unicode_collision_bounds_and_no_cross_owner_collision(db):
    first = await create_user(db, 911_001)
    second = await create_user(db, 911_002)
    service = WorkspaceAccessService(db)
    workspace = await service.create_workspace(first.id, "custom", "  МОЁ\tБУДУЩЕЕ  ")
    assert workspace.normalized_name == normalize_workspace_name("моё будущее")

    with pytest.raises(WorkspaceConflictError):
        await service.create_workspace(first.id, "custom", "мое будущее")
    other = await service.create_workspace(second.id, "custom", "МОЁ БУДУЩЕЕ")
    assert other.normalized_name == workspace.normalized_name

    for invalid in ("", " \n\t ", "<>&", "x" * 101, "имя\u200bскрытое"):
        with pytest.raises(WorkspaceAccessError):
            await service.create_workspace(first.id, "custom", invalid)


async def test_direct_invite_is_bound_hashed_expiring_and_single_use(db):
    owner = await create_user(db, 912_001, "Владелец")
    intended = await create_user(db, 912_002, "Адресат")
    wrong = await create_user(db, 912_003, "Другой")
    service = WorkspaceAccessService(db)
    workspace, context = await create_space(service, owner)

    issued = await service.create_invitation(
        context,
        delivery_mode="direct",
        intended_user_id=intended.id,
        role="viewer",
        template_key="pair_1",
    )
    assert len(issued.token) >= 40
    assert issued.invitation.token_hash == hashlib.sha256(issued.token.encode()).hexdigest()
    assert issued.token not in issued.invitation.token_hash

    for actor in (owner.id, wrong.id):
        with pytest.raises(WorkspaceInvitationError, match="недействительно"):
            await service.invitation_preview(actor, issued.token)
    with pytest.raises(WorkspaceInvitationError, match="недействительно"):
        await service.invitation_preview(
            intended.id,
            issued.token,
            now=datetime.now(UTC) + timedelta(days=8),
        )

    preview = await service.invitation_preview(intended.id, issued.token)
    assert (preview.workspace_name, preview.role, preview.template_key) == (
        workspace.name,
        "viewer",
        "pair_1",
    )
    joined = await service.accept_invitation(intended.id, issued.token)
    assert joined.actor_user_id == intended.id
    assert await active_member_count(db, workspace.id, intended.id) == 1
    with pytest.raises(WorkspaceInvitationError, match="недействительно"):
        await service.accept_invitation(intended.id, issued.token)


async def test_deleted_direct_recipient_never_degrades_to_share(db):
    owner = await create_user(db, 913_001)
    intended = await create_user(db, 913_002)
    attacker = await create_user(db, 913_003)
    service = WorkspaceAccessService(db)
    _workspace, context = await create_space(service, owner)
    issued = await service.create_invitation(
        context,
        delivery_mode="direct",
        intended_user_id=intended.id,
        template_key="pair_1",
    )

    async with db.session() as session:
        await session.execute(delete(User).where(User.id == intended.id))

    assert await invitation_status(db, issued.token) is None
    with pytest.raises(WorkspaceInvitationError, match="недействительно"):
        await service.accept_invitation(attacker.id, issued.token)


async def test_share_actions_are_actor_chat_bound_restart_safe_and_terminal(db):
    owner = await create_user(db, 914_001, "<b>Владелец</b>")
    recipient = await create_user(db, 914_002)
    other = await create_user(db, 914_003)
    service = WorkspaceAccessService(db)
    workspace, context = await create_space(service, owner, name="<i>Общее</i>")
    issued = await service.create_invitation(
        context,
        delivery_mode="share",
        role="editor",
        template_key="custom_warm",
        custom_text="<script>alert(1)</script>",
    )
    actions = await service.issue_incoming_actions(recipient.id, 44_001, issued.token)
    assert set(actions.actions) == {"accept", "details", "later", "decline"}
    assert actions.preview.custom_text == "<script>alert(1)</script>"

    async with db.sessions() as session:
        hashes = set(
            await session.scalars(
                select(WorkspaceActionToken.token_hash).where(
                    WorkspaceActionToken.actor_user_id == recipient.id
                )
            )
        )
    assert hashes == {
        hashlib.sha256(token.encode()).hexdigest() for token in actions.actions.values()
    }
    assert not (hashes & set(actions.actions.values()))

    restarted = WorkspaceAccessService(db)
    later = actions.actions["later"]
    with pytest.raises(WorkspaceInvitationError):
        await restarted.perform_invitation_action(later, other.id, 44_001)
    with pytest.raises(WorkspaceInvitationError):
        await restarted.perform_invitation_action(later, recipient.id, 44_002)
    result = await restarted.perform_invitation_action(later, recipient.id, 44_001)
    assert (result.action, result.status) == ("later", "pending")
    assert await invitation_status(db, issued.token) == "pending"
    with pytest.raises(WorkspaceInvitationError):
        await restarted.perform_invitation_action(later, recipient.id, 44_001)

    joined = await restarted.accept_from_action(actions.actions["accept"], recipient.id, 44_001)
    assert joined.workspace_id == workspace.id
    with pytest.raises(WorkspaceInvitationError):
        await restarted.accept_from_action(actions.actions["accept"], recipient.id, 44_001)


async def test_workspace_presentation_change_invalidates_old_invitation_preview(db):
    owner = await create_user(db, 914_101, "Владелец")
    recipient = await create_user(db, 914_102)
    service = WorkspaceAccessService(db)
    workspace, context = await create_space(service, owner, name="Старое имя")
    issued = await service.create_invitation(
        context,
        delivery_mode="share",
        role="viewer",
        template_key="pair_1",
    )
    old_actions = await service.issue_incoming_actions(recipient.id, 44_101, issued.token)

    await service.rename_workspace(
        context,
        workspace.version,
        "Новое имя",
        character="family",
    )
    with pytest.raises(WorkspaceInvitationError, match="недействительно"):
        await service.accept_from_action(old_actions.actions["accept"], recipient.id, 44_101)

    preview = await service.invitation_preview(recipient.id, issued.token)
    assert (preview.workspace_name, preview.character) == ("Новое имя", "family")
    refreshed = await service.issue_incoming_actions(recipient.id, 44_101, issued.token)
    joined = await service.accept_from_action(refreshed.actions["accept"], recipient.id, 44_101)
    assert joined.actor_user_id == recipient.id


async def test_decline_revoke_renew_and_old_capabilities_fail_closed(db):
    owner = await create_user(db, 915_001)
    first_recipient = await create_user(db, 915_002)
    second_recipient = await create_user(db, 915_003)
    service = WorkspaceAccessService(db)
    _workspace, context = await create_space(service, owner)

    direct = await service.create_invitation(
        context,
        delivery_mode="direct",
        intended_user_id=first_recipient.id,
        template_key="pair_1",
    )
    declined = await service.decline_invitation(first_recipient.id, direct.token)
    assert declined.status == "declined"
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_invitation(first_recipient.id, direct.token)

    share = await service.create_invitation(
        context,
        delivery_mode="share",
        template_key="pair_2",
    )
    old_actions = await service.issue_incoming_actions(second_recipient.id, 45_001, share.token)
    revoked = await service.revoke_invitation(
        context, share.invitation.id, share.invitation.version
    )
    assert revoked.status == "revoked"
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_from_action(old_actions.actions["accept"], second_recipient.id, 45_001)

    new_share = await service.create_invitation(
        context,
        delivery_mode="share",
        template_key="pair_3",
    )
    renewed = await service.renew_invitation(
        context, new_share.invitation.id, new_share.invitation.version
    )
    with pytest.raises(WorkspaceInvitationError):
        await service.accept_invitation(second_recipient.id, new_share.token)
    joined = await service.accept_invitation(second_recipient.id, renewed.token)
    assert joined.actor_user_id == second_recipient.id


async def test_concurrent_double_accept_and_accept_vs_revoke_have_one_winner(db):
    owner = await create_user(db, 916_001)
    first = await create_user(db, 916_002)
    second = await create_user(db, 916_003)
    service = WorkspaceAccessService(db)
    workspace, context = await create_space(service, owner)

    double = await service.create_invitation(
        context,
        delivery_mode="share",
        template_key="team_1",
    )
    outcomes = await asyncio.gather(
        service.accept_invitation(first.id, double.token),
        service.accept_invitation(first.id, double.token),
        return_exceptions=True,
    )
    assert sum(isinstance(result, AccessContext) for result in outcomes) == 1
    assert sum(isinstance(result, WorkspaceInvitationError) for result in outcomes) == 1
    assert await active_member_count(db, workspace.id, first.id) == 1

    context = await service.access_context(owner.id, workspace.id)
    racing = await service.create_invitation(
        context,
        delivery_mode="direct",
        intended_user_id=second.id,
        template_key="team_2",
    )
    outcomes = await asyncio.gather(
        service.accept_invitation(second.id, racing.token),
        service.revoke_invitation(context, racing.invitation.id, racing.invitation.version),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in outcomes) == 1
    assert await invitation_status(db, racing.token) in {"accepted", "revoked"}
    assert await active_member_count(db, workspace.id, second.id) in {0, 1}
    assert (await invitation_status(db, racing.token) == "accepted") == (
        await active_member_count(db, workspace.id, second.id) == 1
    )


@pytest.mark.parametrize("membership_transition", ["revoke", "leave"])
@pytest.mark.parametrize("invitation_transition", ["create", "renew"])
async def test_invite_commit_order_cannot_reactivate_a_concurrently_removed_member(
    db,
    membership_transition,
    invitation_transition,
):
    owner = await create_user(db, 916_100)
    recipient = await create_user(db, 916_101)
    service = WorkspaceAccessService(db)
    workspace, owner_context = await create_space(service, owner, name="Commit ordering")
    _token, recipient_context = await direct_join(service, owner_context, recipient)
    owner_context = await service.access_context(owner.id, workspace.id)
    recipient_context = await service.access_context(recipient.id, workspace.id)
    member = next(
        record.member
        for record in await service.list_members(owner_context)
        if record.member.user_id == recipient.id
    )
    original = await service.create_invitation(
        owner_context,
        delivery_mode="share",
        template_key="team_1",
    )

    paused = PausingWorkspaceAccessService(db)
    if membership_transition == "revoke":
        removal = asyncio.create_task(
            paused.revoke_member(owner_context, recipient.id, member.version)
        )
    else:
        removal = asyncio.create_task(paused.leave_workspace(recipient_context, member.version))
    await asyncio.wait_for(paused.before_lock.wait(), timeout=2)

    if invitation_transition == "create":
        raced = await service.create_invitation(
            owner_context,
            delivery_mode="share",
            template_key="team_2",
        )
    else:
        raced = await service.renew_invitation(
            owner_context,
            original.invitation.id,
            original.invitation.version,
        )
    paused.release_lock.set()
    await removal

    async with db.sessions() as session:
        removed_member = await session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == recipient.id,
            )
        )
        raced_invitation = await session.scalar(
            select(WorkspaceInvitation).where(
                WorkspaceInvitation.token_hash
                == hashlib.sha256(raced.token.encode("utf-8")).hexdigest()
            )
        )
    assert removed_member.status in {"revoked", "left"}
    assert removed_member.revoked_at >= raced_invitation.created_at
    with pytest.raises(WorkspaceInvitationError, match="недействительно"):
        await service.accept_invitation(recipient.id, raced.token)
    assert await active_member_count(db, workspace.id, recipient.id) == 0

    # A new owner-issued capability after the completed removal is the explicit re-entry path.
    await asyncio.sleep(0.002)
    owner_context = await service.access_context(owner.id, workspace.id)
    fresh = await service.create_invitation(
        owner_context,
        delivery_mode="share",
        template_key="team_3",
    )
    joined = await service.accept_invitation(recipient.id, fresh.token)
    assert joined.actor_user_id == recipient.id


async def test_acl_role_matrix_last_owner_and_epoch_invalidation(db):
    owner = await create_user(db, 917_001)
    editor = await create_user(db, 917_002)
    viewer = await create_user(db, 917_003)
    outsider = await create_user(db, 917_004)
    service = WorkspaceAccessService(db)
    workspace, owner_context = await create_space(service, owner)
    _editor_token, editor_context = await direct_join(service, owner_context, editor, role="editor")
    owner_context = await service.access_context(owner.id, workspace.id)
    _viewer_token, viewer_context = await direct_join(service, owner_context, viewer, role="viewer")
    owner_context = await service.access_context(owner.id, workspace.id)
    editor_context = await service.access_context(editor.id, workspace.id)

    with pytest.raises(WorkspaceLastOwnerError):
        owner_member = next(
            record.member
            for record in await service.list_members(owner_context)
            if record.member.user_id == owner.id
        )
        await service.leave_workspace(owner_context, owner_member.version)
    with pytest.raises(WorkspaceAccessDenied, match="недоступно") as foreign:
        await service.access_context(outsider.id, workspace.id)
    with pytest.raises(WorkspaceAccessDenied, match="недоступно") as absent:
        await service.access_context(outsider.id, workspace.id + 999_999)
    assert str(foreign.value) == str(absent.value)

    project = await service.create_project(editor_context, "Общий проект")
    assert (await service.get_project(viewer_context, project.id)).id == project.id
    for operation in (
        lambda: service.create_project(viewer_context, "Нельзя"),
        lambda: service.rename_workspace(viewer_context, workspace.version, "Нельзя переименовать"),
        lambda: service.list_invitations(viewer_context),
    ):
        with pytest.raises(WorkspaceAccessDenied):
            await operation()

    context_snapshot = await service.set_context(viewer_context, 47_001)
    stale_action = await service.issue_action(
        viewer.id,
        47_001,
        "open",
        context=viewer_context,
        workspace_version=(await service.get_workspace(viewer_context)).version,
    )
    viewer_member = next(
        record.member
        for record in await service.list_members(owner_context)
        if record.member.user_id == viewer.id
    )
    await service.revoke_member(owner_context, viewer.id, viewer_member.version)
    with pytest.raises(WorkspaceAccessDenied):
        await service.get_workspace(context_snapshot.access_context)
    assert await service.active_context(viewer.id, 47_001) is None
    assert await service.claim_action(stale_action, viewer.id, 47_001) is None


async def test_projects_are_workspace_scoped_normalized_and_versioned(db):
    first_owner = await create_user(db, 918_001)
    second_owner = await create_user(db, 918_002)
    service = WorkspaceAccessService(db)
    first_space, first_context = await create_space(
        service, first_owner, name="Первое", character="team"
    )
    _second_space, second_context = await create_space(
        service, second_owner, name="Второе", character="team"
    )

    first = await service.create_project(first_context, "  ПРОЁКТ\tАльфа ")
    second = await service.create_project(second_context, "проект альфа")
    assert first.normalized_name == second.normalized_name
    with pytest.raises(WorkspaceConflictError):
        await service.create_project(first_context, "проект альфа")
    with pytest.raises(WorkspaceAccessDenied):
        await service.get_project(second_context, first.id)

    first_context = await service.access_context(first_owner.id, first_space.id)
    mutations = await asyncio.gather(
        service.rename_project(first_context, first.id, first.version, "Победитель"),
        service.set_project_archived(first_context, first.id, first.version, archived=True),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in mutations) == 1
    assert sum(isinstance(result, WorkspaceStaleError) for result in mutations) == 1


async def test_concurrent_workspace_and_role_mutations_have_one_committed_winner(db):
    owner = await create_user(db, 918_101)
    editor = await create_user(db, 918_102)
    service = WorkspaceAccessService(db)
    workspace, owner_context = await create_space(service, owner, name="До гонки")
    _token, _editor_context = await direct_join(service, owner_context, editor)
    owner_context = await service.access_context(owner.id, workspace.id)
    current = await service.get_workspace(owner_context)

    workspace_results = await asyncio.gather(
        service.rename_workspace(owner_context, current.version, "После гонки"),
        service.set_workspace_archived(
            owner_context,
            current.version,
            archived=True,
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in workspace_results) == 1
    assert (
        sum(
            isinstance(result, (WorkspaceStaleError, WorkspaceAccessDenied))
            for result in workspace_results
        )
        == 1
    )
    async with db.sessions() as session:
        persisted = await session.get(Workspace, workspace.id)
    assert persisted.version == current.version + 1
    assert (persisted.status, persisted.name, persisted.access_epoch) in {
        ("active", "После гонки", owner_context.access_epoch),
        ("archived", "До гонки", owner_context.access_epoch + 1),
    }

    if persisted.status == "archived":
        archived_context = await service.access_context(owner.id, workspace.id)
        await service.set_workspace_archived(
            archived_context,
            persisted.version,
            archived=False,
        )
    owner_context = await service.access_context(owner.id, workspace.id)
    member = next(
        record.member
        for record in await service.list_members(owner_context)
        if record.member.user_id == editor.id
    )
    before_role_epoch = owner_context.access_epoch
    role_results = await asyncio.gather(
        service.change_member_role(owner_context, editor.id, "viewer", member.version),
        service.change_member_role(owner_context, editor.id, "owner", member.version),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in role_results) == 1
    assert (
        sum(
            isinstance(result, (WorkspaceStaleError, WorkspaceAccessDenied))
            for result in role_results
        )
        == 1
    )
    async with db.sessions() as session:
        persisted_member = await session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == editor.id,
            )
        )
        persisted_workspace = await session.get(Workspace, workspace.id)
    assert persisted_member.role in {"owner", "viewer"}
    assert persisted_member.version == member.version + 1
    assert persisted_workspace.access_epoch == before_role_epoch + 1


async def test_persistent_capabilities_reject_forged_cross_chat_stale_replay_and_expiry(db):
    owner = await create_user(db, 919_001)
    other = await create_user(db, 919_002)
    service = WorkspaceAccessService(db)
    workspace, context = await create_space(service, owner)
    token = await service.issue_action(
        owner.id,
        49_001,
        "open",
        payload={"safe": "value"},
        context=context,
        workspace_version=workspace.version,
    )
    assert len(f"space:{token}".encode()) <= 64

    restarted = WorkspaceAccessService(db)
    assert await restarted.claim_action("forged", owner.id, 49_001) is None
    assert await restarted.claim_action(token, other.id, 49_001) is None
    assert await restarted.claim_action(token, owner.id, 49_002) is None
    claim = await restarted.claim_action(token, owner.id, 49_001, expected_action="open")
    assert claim is not None and claim.payload == {"safe": "value"}
    assert await restarted.claim_action(token, owner.id, 49_001) is None

    fresh_context = await service.access_context(owner.id, workspace.id)
    stale = await service.issue_action(
        owner.id,
        49_001,
        "rename",
        context=fresh_context,
        workspace_version=(await service.get_workspace(fresh_context)).version,
    )
    current = await service.get_workspace(fresh_context)
    await service.rename_workspace(fresh_context, current.version, "Новое имя")
    assert await service.claim_action(stale, owner.id, 49_001) is None

    expired = await service.issue_action(owner.id, 49_001, "wizard")
    assert (
        await service.claim_action(
            expired,
            owner.id,
            49_001,
            now=datetime.now(UTC) + timedelta(hours=1),
        )
        is None
    )


async def test_concurrent_begin_input_keeps_one_deterministic_live_prompt(db):
    owner = await create_user(db, 919_101)
    service = WorkspaceAccessService(db)

    tokens = await asyncio.gather(
        service.begin_input(owner.id, 49_101, "create_name", payload={"attempt": 1}),
        service.begin_input(owner.id, 49_101, "create_name", payload={"attempt": 2}),
    )

    assert tokens[0] != tokens[1]
    async with db.sessions() as session:
        live_count = await session.scalar(
            select(func.count(WorkspaceActionToken.token_hash)).where(
                WorkspaceActionToken.actor_user_id == owner.id,
                WorkspaceActionToken.chat_id == 49_101,
                WorkspaceActionToken.status == "awaiting_input",
            )
        )
    assert live_count == 1
    pending = await service.pending_input(owner.id, 49_101)
    assert pending is not None
    assert pending.action == "input:create_name"
    assert pending.payload in ({"attempt": 1}, {"attempt": 2})
    claimed = await service.claim_pending_input(owner.id, 49_101, pending.action)
    assert claimed is not None and claimed.payload == pending.payload
    assert await service.pending_input(owner.id, 49_101) is None


@pytest.mark.parametrize("capability_kind", ["action", "input"])
async def test_expired_wizard_capability_cannot_commit_after_waiting_on_a_row_lock(
    db,
    capability_kind,
):
    owner = await create_user(db, 919_101)
    service = WorkspaceAccessService(db)
    if capability_kind == "action":
        token = await service.issue_action(
            owner.id,
            49_101,
            "wizard",
            ttl=timedelta(seconds=1),
        )
        expected_status = "pending"
    else:
        token = await service.begin_input(
            owner.id,
            49_101,
            "rename",
            ttl=timedelta(seconds=1),
        )
        expected_status = "awaiting_input"
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    async with db.session() as blocker:
        await blocker.execute(
            update(WorkspaceActionToken)
            .where(WorkspaceActionToken.token_hash == token_hash)
            .values(status=WorkspaceActionToken.status)
        )
        claim_task = asyncio.create_task(
            service.claim_action(token, owner.id, 49_101)
            if capability_kind == "action"
            else service.claim_pending_input(owner.id, 49_101, "rename")
        )
        await asyncio.sleep(1.25)

    assert await claim_task is None
    async with db.sessions() as session:
        status = await session.scalar(
            select(WorkspaceActionToken.status).where(WorkspaceActionToken.token_hash == token_hash)
        )
    assert status == expected_status


async def test_input_is_single_restart_safe_and_cleanup_removes_stale_state(db):
    owner = await create_user(db, 920_001)
    service = WorkspaceAccessService(db)
    workspace, context = await create_space(service, owner)
    first = await service.begin_input(
        owner.id,
        50_001,
        "rename",
        payload={"value": "first"},
        context=context,
        workspace_version=workspace.version,
    )
    second = await service.begin_input(
        owner.id,
        50_001,
        "rename",
        payload={"value": "second"},
        context=context,
        workspace_version=workspace.version,
    )
    assert first != second

    restarted = WorkspaceAccessService(db)
    pending = await restarted.pending_input(owner.id, 50_001)
    assert pending is not None and pending.payload == {"value": "second"}
    claimed = await restarted.claim_pending_input(owner.id, 50_001, "rename")
    assert claimed is not None and claimed.payload == {"value": "second"}
    assert await restarted.pending_input(owner.id, 50_001) is None

    context = await service.access_context(owner.id, workspace.id)
    await service.set_context(context, 50_002, ttl=timedelta(minutes=1))
    await service.issue_action(owner.id, 50_002, "short", ttl=timedelta(seconds=1))
    cleaned = await service.cleanup(now=datetime.now(UTC) + timedelta(minutes=2))
    assert cleaned.action_tokens >= 1
    assert cleaned.contexts == 1
    assert await service.active_context(owner.id, 50_002) is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WorkspaceContext.id))) == 0
