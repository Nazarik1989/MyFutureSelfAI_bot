from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, func, select, update

from future_self.knowledge import (
    KnowledgeAccessDenied,
    KnowledgeConflictError,
    KnowledgeExtractionResult,
    KnowledgeQuotaError,
    KnowledgeQuotaPolicy,
    KnowledgeService,
    StoredKnowledgeOriginal,
)
from future_self.models import (
    KnowledgeAuditEvent,
    KnowledgeCaptureDraft,
    KnowledgeIngestionJob,
    KnowledgeRuntimeState,
    KnowledgeSource,
    KnowledgeSourceRevision,
    KnowledgeSpace,
    User,
    Workspace,
    WorkspaceMember,
)
from future_self.workspace_access import AccessContext, WorkspaceAccessService


async def make_users(db, count: int) -> tuple[User, ...]:
    async with db.session() as session:
        users = tuple(
            User(telegram_id=9_400_000 + index, display_name=f"Knowledge {index}")
            for index in range(count)
        )
        session.add_all(users)
        await session.flush()
        return users


async def shared_contexts(db, *, viewer: bool = False) -> tuple[AccessContext, AccessContext]:
    owner, member = await make_users(db, 2)
    workspace_service = WorkspaceAccessService(db)
    workspace = await workspace_service.create_workspace(owner.id, "team", "Library")
    owner_context = await workspace_service.access_context(owner.id, workspace.id)
    invitation = await workspace_service.create_invitation(
        owner_context,
        delivery_mode="direct",
        intended_user_id=member.id,
        role="viewer" if viewer else "editor",
        template_key="neutral_1",
    )
    member_context = await workspace_service.accept_invitation(member.id, invitation.token)
    return await workspace_service.access_context(owner.id, workspace.id), member_context


def original(number: int = 1, *, size: int = 12) -> StoredKnowledgeOriginal:
    return StoredKnowledgeOriginal(
        storage_key=f"space/source/original-{number}.txt",
        sha256=f"{number:x}" * 64,
        size_bytes=size,
        declared_mime="text/plain",
        detected_mime="text/plain",
        detected_format="txt",
        safe_display_name=f"note-{number}.txt",
        provenance={"transport": "test"},
    )


async def confirmed_text(db, owner: User):
    service = KnowledgeService(db)
    preview = await service.begin_capture(
        owner.id,
        owner.telegram_id,
        capture_kind="text",
        text_content="Private source body",
        title="Private note",
    )
    reservation = await service.reserve_capture(
        owner.id,
        owner.telegram_id,
        preview.draft_public_id,
        preview.version,
        reserved_bytes=12,
        idempotency_key=f"capture-{preview.draft_public_id}",
    )
    receipt = await service.commit_capture(
        owner.id,
        owner.telegram_id,
        reservation.public_id,
        original=original(),
    )
    return service, receipt


async def confirmed_shared_text(db):
    owner_context, member_context = await shared_contexts(db)
    service = KnowledgeService(db)
    spaces = await service.list_spaces(owner_context.actor_user_id)
    shared = next(space for space in spaces if space.access.kind == "workspace")
    preview = await service.begin_capture(
        owner_context.actor_user_id,
        9_490_001,
        capture_kind="text",
        text_content="Shared source body",
        title="Shared note",
        target_space_public_id=shared.access.space_public_id,
    )
    reservation = await service.reserve_capture(
        owner_context.actor_user_id,
        9_490_001,
        preview.draft_public_id,
        preview.version,
        reserved_bytes=12,
        idempotency_key=f"shared-{preview.draft_public_id}",
    )
    receipt = await service.commit_capture(
        owner_context.actor_user_id,
        9_490_001,
        reservation.public_id,
        original=original(7),
    )
    return service, owner_context, member_context, shared.access.space_public_id, receipt


async def test_personal_space_is_idempotent_public_and_repairs_legacy_null(db):
    (owner,) = await make_users(db, 1)
    service = KnowledgeService(db)
    first = await service.ensure_personal_space(owner.id)
    second = await service.ensure_personal_space(owner.id)
    assert first.access.knowledge_space_id == second.access.knowledge_space_id
    assert len(first.access.space_public_id) == 36

    async with db.session() as session:
        await session.execute(
            update(KnowledgeSpace)
            .where(KnowledgeSpace.id == first.access.knowledge_space_id)
            .values(public_id=None)
        )
    repaired = await service.list_spaces(owner.id)
    assert len(repaired) == 1
    assert repaired[0].access.space_public_id
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(KnowledgeSpace.id))) == 1
        assert await session.scalar(select(func.count(KnowledgeAuditEvent.id))) == 1


async def test_capture_requires_confirmation_commits_atomically_and_scrubs_raw_payload(db):
    (owner,) = await make_users(db, 1)
    service = KnowledgeService(db)
    preview = await service.begin_capture(
        owner.id,
        owner.telegram_id,
        capture_kind="text",
        text_content="Do not retain this draft body",
        title="Source title",
        knowledge_role="perspective",
        priority="high",
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(KnowledgeSource.id))) == 0

    reservation = await service.reserve_capture(
        owner.id,
        owner.telegram_id,
        preview.draft_public_id,
        preview.version,
        reserved_bytes=12,
        idempotency_key=f"capture-{preview.draft_public_id}",
    )
    assert reservation.material.text_content == "Do not retain this draft body"
    receipt = await service.commit_capture(
        owner.id,
        owner.telegram_id,
        reservation.public_id,
        original=original(),
    )
    assert receipt.processing_status == "queued"
    async with db.sessions() as session:
        draft = await session.scalar(
            select(KnowledgeCaptureDraft).where(
                KnowledgeCaptureDraft.public_id == preview.draft_public_id
            )
        )
        assert draft.status == "confirmed"
        assert (
            draft.text_content,
            draft.source_url,
            draft.telegram_file_id,
            draft.provenance,
        ) == (None, None, None, None)
        assert await session.scalar(select(func.count(KnowledgeSource.id))) == 1
        assert await session.scalar(select(func.count(KnowledgeSourceRevision.id))) == 1
        assert await session.scalar(select(func.count(KnowledgeIngestionJob.id))) == 1


async def test_cancel_and_expiry_create_no_source_and_timeout_is_one_shot(db):
    (owner,) = await make_users(db, 1)
    service = KnowledgeService(db)
    draft = await service.begin_capture(
        owner.id,
        owner.telegram_id,
        capture_kind="text",
        text_content="Cancel me",
    )
    assert await service.cancel_capture(
        owner.id, owner.telegram_id, draft.draft_public_id, draft.version
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(KnowledgeSource.id))) == 0
        cancelled = await session.scalar(
            select(KnowledgeCaptureDraft).where(
                KnowledgeCaptureDraft.public_id == draft.draft_public_id
            )
        )
        assert cancelled.text_content is None

    expiring = await service.begin_empty_capture(owner.id, owner.telegram_id)
    async with db.session() as session:
        await session.execute(
            update(KnowledgeCaptureDraft)
            .where(KnowledgeCaptureDraft.public_id == expiring.draft_public_id)
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    first = await service.capture_state(owner.id, owner.telegram_id)
    second = await service.capture_state(owner.id, owner.telegram_id)
    assert (first.preview, first.expired_now) == (None, True)
    assert (second.preview, second.expired_now) == (None, False)


async def test_runner_lease_finalize_and_new_revision_are_immutable(db):
    (owner,) = await make_users(db, 1)
    service, receipt = await confirmed_text(db, owner)
    claim = await service.claim_next_job("runner-a")
    assert claim is not None
    assert claim.source_public_id == receipt.source_public_id
    assert await service.claim_next_job("runner-b") is None
    assert await service.heartbeat_job(claim.id, claim.lease_token)
    assert await service.finalize_job(
        claim.id,
        claim.lease_token,
        KnowledgeExtractionResult(
            status="ready",
            extracted_storage_key="space/source/extracted-1.txt",
            extracted_sha256="e" * 64,
            extracted_size_bytes=20,
            metadata={"extractor": "txt-v1"},
        ),
    )
    ready = await service.get_source(owner.id, receipt.source_public_id)
    assert ready.source.processing_status == "ready"
    assert ready.revision.extraction_status == "ready"
    first_revision_id = ready.revision.id

    next_receipt = await service.append_revision(
        owner.id,
        receipt.source_public_id,
        ready.source.version,
        original=original(2),
        idempotency_key=f"revision-{receipt.source_public_id}-2",
    )
    assert next_receipt.revision_public_id != receipt.revision_public_id
    async with db.sessions() as session:
        revisions = list(
            (
                await session.scalars(
                    select(KnowledgeSourceRevision)
                    .where(KnowledgeSourceRevision.source_id == ready.source.id)
                    .order_by(KnowledgeSourceRevision.revision_number)
                )
            ).all()
        )
        assert [revision.revision_number for revision in revisions] == [1, 2]
        assert revisions[0].id == first_revision_id
        assert revisions[0].extraction_status == "ready"
        assert revisions[0].sha256 == "1" * 64


async def test_manual_retry_creates_new_revision_and_preserves_failed_revision(db):
    (owner,) = await make_users(db, 1)
    service, receipt = await confirmed_text(db, owner)
    first_claim = await service.claim_next_job("runner-fail")
    assert first_claim is not None
    assert await service.fail_job(
        first_claim.id,
        first_claim.lease_token,
        failure_kind="permanent",
        safe_error_code="parser_failure",
    )
    failed = await service.get_source(owner.id, receipt.source_public_id)
    first_finalized_at = failed.revision.finalized_at

    retry_job = await service.retry_source(
        owner.id,
        receipt.source_public_id,
        failed.source.version,
        pipeline_version="v2",
        max_attempts=5,
    )
    retry_claim = await service.claim_next_job("runner-retry")

    assert retry_job.max_attempts == 5
    assert retry_job.pipeline_version == "v2"
    assert retry_claim is not None
    assert retry_claim.revision_number == 2
    assert retry_claim.original_storage_key == original().storage_key
    async with db.sessions() as session:
        revisions = list(
            (
                await session.scalars(
                    select(KnowledgeSourceRevision)
                    .where(KnowledgeSourceRevision.source_id == failed.source.id)
                    .order_by(KnowledgeSourceRevision.revision_number)
                )
            ).all()
        )
        assert [row.extraction_status for row in revisions] == ["failed", "pending"]
        assert revisions[0].finalized_at == first_finalized_at
        assert revisions[0].original_storage_key == original().storage_key
        assert revisions[1].original_storage_key is None
        assert revisions[1].original_revision_id == revisions[0].id


async def test_expired_final_attempt_is_failed_with_durable_audit(db):
    (owner,) = await make_users(db, 1)
    service = KnowledgeService(db)
    preview = await service.begin_capture(
        owner.id,
        owner.telegram_id,
        capture_kind="text",
        text_content="Audit exhausted retry",
    )
    reservation = await service.reserve_capture(
        owner.id,
        owner.telegram_id,
        preview.draft_public_id,
        preview.version,
        reserved_bytes=12,
        idempotency_key=f"exhaust-{preview.draft_public_id}",
    )
    receipt = await service.commit_capture(
        owner.id,
        owner.telegram_id,
        reservation.public_id,
        original=original(),
        max_attempts=1,
    )
    started = datetime(2030, 1, 1, tzinfo=UTC)
    claim = await service.claim_next_job("crashing-runner", now=started, lease_seconds=30)
    assert claim is not None

    assert (
        await service.claim_next_job(
            "recovery-runner", now=started + timedelta(seconds=31), lease_seconds=30
        )
        is None
    )

    async with db.sessions() as session:
        job = await session.get(KnowledgeIngestionJob, claim.id)
        source = await session.scalar(
            select(KnowledgeSource).where(KnowledgeSource.public_id == receipt.source_public_id)
        )
        events = list(
            (
                await session.scalars(
                    select(KnowledgeAuditEvent).where(
                        KnowledgeAuditEvent.job_id == claim.id,
                        KnowledgeAuditEvent.event_type == "ingestion.status_changed",
                    )
                )
            ).all()
        )
        assert job.status == "failed"
        assert source.processing_status == "failed"
        assert any(
            event.safe_metadata == {"status": "failed", "job_type": "extract"} for event in events
        )


async def test_shared_acl_viewer_cannot_write_editor_can_and_health_is_personal_only(db):
    owner_context, viewer_context = await shared_contexts(db, viewer=True)
    service = KnowledgeService(db)
    spaces = await service.list_spaces(viewer_context.actor_user_id)
    shared = next(space for space in spaces if space.access.kind == "workspace")
    assert shared.access.role == "viewer"
    with pytest.raises(KnowledgeAccessDenied):
        await service.begin_capture(
            viewer_context.actor_user_id,
            9_400_001,
            capture_kind="text",
            text_content="Viewer write",
            target_space_public_id=shared.access.space_public_id,
        )
    owner_spaces = await service.list_spaces(owner_context.actor_user_id)
    owner_shared = next(space for space in owner_spaces if space.access.kind == "workspace")
    with pytest.raises(KnowledgeAccessDenied):
        await service.begin_capture(
            owner_context.actor_user_id,
            9_400_000,
            capture_kind="text",
            text_content="Medical",
            target_space_public_id=owner_shared.access.space_public_id,
            system_classification="health_private",
        )


async def test_list_sources_rechecks_revoked_membership_in_scoped_source_sql(db, monkeypatch):
    service, _owner, member, space_public_id, receipt = await confirmed_shared_text(db)
    initial = await service.list_sources(member.actor_user_id, space_public_id)
    assert initial.total == 1
    assert initial.items[0].source.public_id == receipt.source_public_id

    statements: list[str] = []

    def record_sql(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(" ".join(statement.casefold().split()))

    original_resolve = service._resolve_space_public_session
    revoked = False

    async def revoke_after_precheck(session, actor_user_id, public_id, roles, require_active):
        nonlocal revoked
        access = await original_resolve(session, actor_user_id, public_id, roles, require_active)
        if actor_user_id == member.actor_user_id and not revoked:
            revoked = True
            await session.execute(
                update(WorkspaceMember)
                .where(
                    WorkspaceMember.workspace_id == member.workspace_id,
                    WorkspaceMember.user_id == member.actor_user_id,
                )
                .values(
                    status="revoked",
                    revoked_at=datetime.now(UTC),
                    version=WorkspaceMember.version + 1,
                )
            )
        return access

    monkeypatch.setattr(service, "_resolve_space_public_session", revoke_after_precheck)
    event.listen(db.engine.sync_engine, "before_cursor_execute", record_sql)
    try:
        with pytest.raises(KnowledgeAccessDenied):
            await service.list_sources(member.actor_user_id, space_public_id)
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_sql)

    source_selects = [
        statement
        for statement in statements
        if statement.startswith("select") and "knowledge_sources" in statement
    ]
    assert source_selects
    assert all("workspace_members.status" in statement for statement in source_selects)
    assert all("workspaces.status" in statement for statement in source_selects)
    assert all("knowledge_spaces.status" in statement for statement in source_selects)


async def test_find_duplicate_rechecks_archived_workspace_in_same_acl_query(db, monkeypatch):
    service, _owner, member, space_public_id, receipt = await confirmed_shared_text(db)
    initial = await service.find_duplicate(member.actor_user_id, space_public_id, "7" * 64)
    assert initial is not None and initial.source.public_id == receipt.source_public_id

    statements: list[str] = []

    def record_sql(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(" ".join(statement.casefold().split()))

    original_resolve = service._resolve_space_public_session
    archived = False

    async def archive_after_precheck(session, actor_user_id, public_id, roles, require_active):
        nonlocal archived
        access = await original_resolve(session, actor_user_id, public_id, roles, require_active)
        if actor_user_id == member.actor_user_id and not archived:
            archived = True
            await session.execute(
                update(Workspace)
                .where(Workspace.id == member.workspace_id)
                .values(
                    status="archived",
                    access_epoch=Workspace.access_epoch + 1,
                    version=Workspace.version + 1,
                )
            )
        return access

    monkeypatch.setattr(service, "_resolve_space_public_session", archive_after_precheck)
    event.listen(db.engine.sync_engine, "before_cursor_execute", record_sql)
    try:
        with pytest.raises(KnowledgeAccessDenied):
            await service.find_duplicate(member.actor_user_id, space_public_id, "7" * 64)
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", record_sql)

    duplicate_selects = [
        statement
        for statement in statements
        if statement.startswith("select")
        and "knowledge_sources" in statement
        and "knowledge_source_revisions" in statement
    ]
    assert duplicate_selects
    assert all("workspace_members.status" in statement for statement in duplicate_selects)
    assert all("workspaces.status" in statement for statement in duplicate_selects)
    assert all("knowledge_spaces.status" in statement for statement in duplicate_selects)


async def test_trashed_listing_includes_purge_failed_but_not_pending(db):
    (owner,) = await make_users(db, 1)
    service, receipt = await confirmed_text(db, owner)
    space = (await service.list_spaces(owner.id))[0]
    record = await service.get_source(owner.id, receipt.source_public_id)
    trashed = await service.trash_source(owner.id, receipt.source_public_id, record.source.version)
    await service.request_permanent_delete(
        owner.id,
        receipt.source_public_id,
        trashed.version,
    )

    pending_page = await service.list_sources(
        owner.id,
        space.access.space_public_id,
        lifecycle_status="trashed",
    )
    assert pending_page.total == 0

    claim = await service.claim_next_job("purge-failure-runner")
    assert claim is not None and claim.job_type == "purge"
    assert await service.fail_job(
        claim.id,
        claim.lease_token,
        failure_kind="permanent",
        safe_error_code="asset_delete_failed",
    )

    trash_page = await service.list_sources(
        owner.id,
        space.access.space_public_id,
        lifecycle_status="trashed",
    )
    assert trash_page.total == 1
    assert [item.source.public_id for item in trash_page.items] == [receipt.source_public_id]
    assert trash_page.items[0].source.lifecycle_status == "purge_failed"


async def test_workspace_membership_role_and_project_changes_are_audited(db):
    owner_context, member_context = await shared_contexts(db)
    workspace_service = WorkspaceAccessService(db)
    owner_context = await workspace_service.access_context(
        owner_context.actor_user_id, owner_context.workspace_id
    )
    members = await workspace_service.list_members(owner_context)
    member = next(
        row.member for row in members if row.member.user_id == member_context.actor_user_id
    )
    await workspace_service.change_member_role(
        owner_context,
        member.user_id,
        "viewer",
        member.version,
    )
    owner_context = await workspace_service.access_context(
        owner_context.actor_user_id, owner_context.workspace_id
    )
    project = await workspace_service.create_project(owner_context, "Audited project")
    await workspace_service.set_project_archived(
        owner_context,
        project.id,
        project.version,
        archived=True,
    )

    async with db.sessions() as session:
        event_types = set(
            (
                await session.scalars(
                    select(KnowledgeAuditEvent.event_type).where(
                        KnowledgeAuditEvent.workspace_id == owner_context.workspace_id
                    )
                )
            ).all()
        )
        assert {
            "workspace.created",
            "workspace.member_added",
            "workspace.role_changed",
            "workspace.project_created",
            "workspace.project_archived",
            "space.created",
        }.issubset(event_types)


async def test_quota_reservation_and_maintenance_fence_are_atomic(db):
    (owner,) = await make_users(db, 1)
    service = KnowledgeService(
        db,
        quota_policy=KnowledgeQuotaPolicy(
            max_source_bytes=10,
            max_extracted_bytes=1,
            daily_ingest_bytes_per_user=10,
            storage_bytes_per_user=12,
            daily_sources_per_user=2,
            max_pending_jobs_per_user=2,
            daily_ingest_bytes_per_space=10,
            storage_bytes_per_space=12,
            daily_sources_per_space=2,
            max_pending_jobs_per_space=2,
        ),
    )
    draft = await service.begin_capture(
        owner.id,
        owner.telegram_id,
        capture_kind="text",
        text_content="1234567890",
    )
    reservation = await service.reserve_capture(
        owner.id,
        owner.telegram_id,
        draft.draft_public_id,
        draft.version,
        reserved_bytes=10,
        idempotency_key=f"quota-{draft.draft_public_id}",
    )
    second = await service.begin_capture(
        owner.id,
        owner.telegram_id + 1,
        capture_kind="text",
        text_content="x",
    )
    with pytest.raises(KnowledgeQuotaError):
        await service.reserve_capture(
            owner.id,
            owner.telegram_id + 1,
            second.draft_public_id,
            second.version,
            reserved_bytes=1,
            idempotency_key=f"quota-second-{second.draft_public_id}",
        )
    await service.release_capture_reservation(owner.id, owner.telegram_id, reservation.public_id)
    version = await service.set_maintenance_paused(True)
    assert version >= 2 and await service.runtime_paused()
    with pytest.raises(KnowledgeConflictError):
        await service.begin_empty_capture(owner.id, owner.telegram_id + 1)
    async with db.sessions() as session:
        state = await session.get(KnowledgeRuntimeState, 1)
        assert state.maintenance_paused is True


async def test_capability_is_actor_chat_version_bound_and_single_use(db):
    owner, stranger = await make_users(db, 2)
    service = KnowledgeService(db)
    draft = await service.begin_empty_capture(owner.id, owner.telegram_id)
    issued = await service.issue_action(
        owner.id,
        owner.telegram_id,
        "capture_title",
        draft.target_space_public_id,
        capture_draft_public_id=draft.draft_public_id,
        status="pending",
        payload={"field": "title"},
    )
    assert await service.claim_action(issued.token, stranger.id, stranger.telegram_id) is None
    claim = await service.claim_action(issued.token, owner.id, owner.telegram_id)
    assert claim is not None and claim.action == "capture_title"
    assert await service.claim_action(issued.token, owner.id, owner.telegram_id) is None
