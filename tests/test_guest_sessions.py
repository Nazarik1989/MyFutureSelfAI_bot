import asyncio
import inspect
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from future_self.access import ADMIN, BLOCKED, SUBSCRIBER
from future_self.db import Database
from future_self.guest_access import (
    GuestDemoKind,
    GuestProviderStartOutcome,
    GuestQuotaDenialReason,
    GuestQuotaPolicy,
    GuestQuotaService,
    GuestReservationOutcome,
    GuestSessionOutcome,
    GuestSessionService,
    GuestSessionStatus,
    GuestUsageStatus,
)
from future_self.models import GuestDemoSession, GuestUsageLedger, User
from future_self.repositories import UserRepository


async def create_guest(db, telegram_id: int) -> tuple[int, int]:
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(telegram_id, "UTC")
        return user.id, user.access_version


async def prepare_processing(
    db,
    *,
    user_id: int,
    chat_id: int,
    access_version: int,
    update_id: int,
    now: datetime,
    policy: GuestQuotaPolicy | None = None,
):
    service = GuestSessionService(db, policy)
    started = await service.start_session(
        user_id=user_id,
        chat_id=chat_id,
        access_version=access_version,
        demo_kind=GuestDemoKind.THOUGHT_BREAKDOWN,
        prompt_message_id=100 + update_id,
        now=now,
    )
    assert started.outcome is GuestSessionOutcome.STARTED
    claimed = await service.claim_input(
        user_id=user_id,
        chat_id=chat_id,
        access_version=access_version,
        telegram_update_id=update_id,
        telegram_message_id=1000 + update_id,
        now=now,
    )
    assert claimed.outcome is GuestSessionOutcome.CLAIMED
    assert claimed.session is not None
    reservation = await GuestQuotaService(db, policy).reserve(
        user_id=user_id,
        demo_kind=GuestDemoKind.THOUGHT_BREAKDOWN,
        idempotency_key=f"session:{update_id}",
        telegram_update_id=update_id,
        now=now,
    )
    assert reservation.can_bind_session and reservation.reservation is not None
    bound = await service.bind_reservation(
        user_id=user_id,
        chat_id=chat_id,
        access_version=access_version,
        session_version=claimed.session.version,
        reservation_token=reservation.reservation.reservation_token,
        now=now,
    )
    assert bound.outcome is GuestSessionOutcome.BOUND
    assert bound.session is not None
    return service, bound, reservation.reservation


async def begin_bound_provider(
    service: GuestSessionService,
    *,
    reservation_token: str,
    user_id: int,
    chat_id: int,
    access_version: int,
    session_version: int,
    now: datetime,
):
    decision = await service.begin_provider_call(
        reservation_token=reservation_token,
        user_id=user_id,
        chat_id=chat_id,
        access_version=access_version,
        session_version=session_version,
        now=now,
    )
    assert decision.outcome is GuestProviderStartOutcome.STARTED
    assert decision.can_invoke_provider
    assert decision.provider_started_at == now
    return decision


async def test_provider_start_is_the_only_provider_permission_and_is_idempotent(db):
    user_id, access_version = await create_guest(db, 96030)
    now = datetime(2026, 8, 6, 8, tzinfo=UTC)
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96030,
        access_version=access_version,
        update_id=101,
        now=now,
    )
    assert bound.session is not None

    direct_completion = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96030,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "must be rejected before provider start"},
        now=now,
    )
    wrong_token = await service.begin_provider_call(
        reservation_token="x" * 43,
        user_id=user_id,
        chat_id=96030,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    wrong_version = await service.begin_provider_call(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96030,
        access_version=access_version,
        session_version=bound.session.version + 1,
        now=now,
    )
    assert direct_completion.outcome is GuestReservationOutcome.STALE
    assert not direct_completion.changed
    assert wrong_token.outcome is GuestProviderStartOutcome.STALE
    assert not wrong_token.can_invoke_provider
    assert wrong_version.outcome is GuestProviderStartOutcome.STALE
    assert not wrong_version.changed

    started = await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96030,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    repeated = await service.begin_provider_call(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96030,
        access_version=access_version,
        session_version=bound.session.version,
        now=now + timedelta(seconds=1),
    )
    assert repeated.outcome is GuestProviderStartOutcome.ALREADY_STARTED
    assert not repeated.changed
    assert not repeated.can_invoke_provider
    assert repeated.provider_started_at == started.provider_started_at == now

    completion = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96030,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "valid after provider start"},
        now=now + timedelta(seconds=1),
    )
    assert completion.outcome is GuestReservationOutcome.SUCCEEDED
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        assert usage is not None and usage.provider_started_at is not None


async def test_provider_start_expiry_and_access_outcomes_are_typed(db):
    now = datetime(2026, 8, 6, 8, 30, tzinfo=UTC)
    policy = GuestQuotaPolicy(reservation_ttl=timedelta(minutes=2))
    expired_user, expired_access = await create_guest(db, 96031)
    expired_service, expired_bound, expired_reservation = await prepare_processing(
        db,
        user_id=expired_user,
        chat_id=96031,
        access_version=expired_access,
        update_id=102,
        now=now,
        policy=policy,
    )
    assert expired_bound.session is not None
    expired = await expired_service.begin_provider_call(
        reservation_token=expired_reservation.reservation_token,
        user_id=expired_user,
        chat_id=96031,
        access_version=expired_access,
        session_version=expired_bound.session.version,
        now=now + timedelta(minutes=2),
    )
    assert expired.outcome is GuestProviderStartOutcome.EXPIRED
    assert expired.changed and not expired.can_invoke_provider
    assert expired.provider_started_at is None
    assert expired.session is not None
    assert expired.session.status is GuestSessionStatus.AWAITING_INPUT
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, expired_reservation.usage_id)
        assert usage is not None and usage.status == GuestUsageStatus.EXPIRED.value
        assert usage.provider_started_at is None

    changed_user, changed_access = await create_guest(db, 96032)
    changed_service, changed_bound, changed_reservation = await prepare_processing(
        db,
        user_id=changed_user,
        chat_id=96032,
        access_version=changed_access,
        update_id=103,
        now=now,
    )
    assert changed_bound.session is not None
    async with db.session() as session:
        user = await session.get(User, changed_user)
        assert user is not None
        user.access_version += 1
    access_changed = await changed_service.begin_provider_call(
        reservation_token=changed_reservation.reservation_token,
        user_id=changed_user,
        chat_id=96032,
        access_version=changed_access,
        session_version=changed_bound.session.version,
        now=now,
    )
    assert access_changed.outcome is GuestProviderStartOutcome.ACCESS_CHANGED
    assert not access_changed.changed and not access_changed.can_invoke_provider

    blocked_user, blocked_access = await create_guest(db, 96033)
    blocked_service, blocked_bound, blocked_reservation = await prepare_processing(
        db,
        user_id=blocked_user,
        chat_id=96033,
        access_version=blocked_access,
        update_id=104,
        now=now,
    )
    assert blocked_bound.session is not None
    async with db.session() as session:
        user = await session.get(User, blocked_user)
        assert user is not None
        user.access_tier = BLOCKED
        user.access_version += 1
    not_guest = await blocked_service.begin_provider_call(
        reservation_token=blocked_reservation.reservation_token,
        user_id=blocked_user,
        chat_id=96033,
        access_version=blocked_access,
        session_version=blocked_bound.session.version,
        now=now,
    )
    assert not_guest.outcome is GuestProviderStartOutcome.NOT_GUEST
    assert not not_guest.changed and not not_guest.can_invoke_provider


async def test_concurrent_provider_start_has_one_permission_across_database_instances(db):
    user_id, access_version = await create_guest(db, 96034)
    now = datetime(2026, 8, 6, 8, 45, tzinfo=UTC)
    _service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96034,
        access_version=access_version,
        update_id=105,
        now=now,
    )
    assert bound.session is not None
    second_db = Database(db.url)
    try:
        decisions = await asyncio.gather(
            *(
                GuestSessionService(db if index % 2 == 0 else second_db).begin_provider_call(
                    reservation_token=reservation.reservation_token,
                    user_id=user_id,
                    chat_id=96034,
                    access_version=access_version,
                    session_version=bound.session.version,
                    now=now,
                )
                for index in range(8)
            )
        )
        assert sum(decision.can_invoke_provider for decision in decisions) == 1
        assert (
            sum(
                decision.outcome is GuestProviderStartOutcome.ALREADY_STARTED
                for decision in decisions
            )
            == 7
        )
        assert {decision.provider_started_at for decision in decisions} == {now}
        async with second_db.sessions() as session:
            usage = await session.get(GuestUsageLedger, reservation.usage_id)
            assert usage is not None
            assert usage.provider_started_at is not None
    finally:
        await second_db.dispose()


async def test_start_claim_are_bound_to_owner_chat_and_access_version(db):
    user_id, access_version = await create_guest(db, 96001)
    other_id, other_version = await create_guest(db, 96002)
    service = GuestSessionService(db)
    now = datetime(2026, 8, 6, 9, tzinfo=UTC)
    started = await service.start_session(
        user_id=user_id,
        chat_id=96001,
        access_version=access_version,
        demo_kind="first_step",
        prompt_message_id=10,
        now=now,
    )
    assert started.outcome is GuestSessionOutcome.STARTED
    assert started.session is not None
    assert started.session.status is GuestSessionStatus.AWAITING_INPUT

    wrong_chat = await service.claim_input(
        user_id=user_id,
        chat_id=99999,
        access_version=access_version,
        telegram_update_id=1,
        telegram_message_id=2,
        now=now,
    )
    wrong_owner = await service.claim_input(
        user_id=other_id,
        chat_id=96001,
        access_version=other_version,
        telegram_update_id=1,
        telegram_message_id=2,
        now=now,
    )
    stale_access = await service.claim_input(
        user_id=user_id,
        chat_id=96001,
        access_version=access_version + 1,
        telegram_update_id=1,
        telegram_message_id=2,
        now=now,
    )
    assert wrong_chat.outcome is GuestSessionOutcome.STALE
    assert wrong_owner.outcome is GuestSessionOutcome.STALE
    assert stale_access.outcome is GuestSessionOutcome.STALE

    claimed = await service.claim_input(
        user_id=user_id,
        chat_id=96001,
        access_version=access_version,
        telegram_update_id=1,
        telegram_message_id=2,
        now=now,
    )
    assert claimed.outcome is GuestSessionOutcome.CLAIMED
    assert claimed.session is not None
    assert claimed.session.consumed_update_id == 1
    assert claimed.session.consumed_message_id == 2


async def test_concurrent_and_duplicate_claims_start_only_one_processing_flow(db):
    user_id, access_version = await create_guest(db, 96003)
    now = datetime(2026, 8, 6, 10, tzinfo=UTC)
    service = GuestSessionService(db)
    await service.start_session(
        user_id=user_id,
        chat_id=96003,
        access_version=access_version,
        demo_kind="thought_breakdown",
        prompt_message_id=20,
        now=now,
    )
    claims = await asyncio.gather(
        *(
            GuestSessionService(db).claim_input(
                user_id=user_id,
                chat_id=96003,
                access_version=access_version,
                telegram_update_id=10,
                telegram_message_id=11,
                now=now,
            )
            for _ in range(8)
        )
    )
    assert sum(item.outcome is GuestSessionOutcome.CLAIMED for item in claims) == 1
    assert sum(item.outcome is GuestSessionOutcome.DUPLICATE for item in claims) == 7
    other_input = await service.claim_input(
        user_id=user_id,
        chat_id=96003,
        access_version=access_version,
        telegram_update_id=12,
        telegram_message_id=13,
        now=now,
    )
    assert other_input.outcome is GuestSessionOutcome.IN_PROGRESS


async def test_session_survives_restart_and_live_processing_is_not_overwritten(db):
    user_id, access_version = await create_guest(db, 96004)
    now = datetime(2026, 8, 6, 11, tzinfo=UTC)
    first = GuestSessionService(db)
    started = await first.start_session(
        user_id=user_id,
        chat_id=96004,
        access_version=access_version,
        demo_kind="first_step",
        prompt_message_id=30,
        now=now,
    )
    restarted = GuestSessionService(db)
    claimed = await restarted.claim_input(
        user_id=user_id,
        chat_id=96004,
        access_version=access_version,
        telegram_update_id=20,
        telegram_message_id=21,
        now=now,
    )
    assert claimed.outcome is GuestSessionOutcome.CLAIMED
    assert claimed.session is not None and started.session is not None
    in_progress = await GuestSessionService(db).start_session(
        user_id=user_id,
        chat_id=96004,
        access_version=access_version,
        demo_kind="thought_breakdown",
        prompt_message_id=31,
        now=now,
    )
    assert in_progress.outcome is GuestSessionOutcome.IN_PROGRESS
    assert in_progress.session is not None
    assert in_progress.session.demo_kind is GuestDemoKind.FIRST_STEP
    assert in_progress.session.prompt_message_id == 30
    assert in_progress.session.version == claimed.session.version


async def test_provider_failure_is_retry_ready_but_old_input_stays_duplicate(db):
    user_id, access_version = await create_guest(db, 96005)
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96005,
        access_version=access_version,
        update_id=30,
        now=now,
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96005,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    failed = await service.fail_processing(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96005,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    assert failed.outcome is GuestReservationOutcome.FAILED
    assert failed.session is not None
    assert failed.session.status is GuestSessionStatus.AWAITING_INPUT
    assert failed.session.result_payload is None

    duplicate = await service.claim_input(
        user_id=user_id,
        chat_id=96005,
        access_version=access_version,
        telegram_update_id=30,
        telegram_message_id=1030,
        now=now,
    )
    fresh = await service.claim_input(
        user_id=user_id,
        chat_id=96005,
        access_version=access_version,
        telegram_update_id=31,
        telegram_message_id=1031,
        now=now,
    )
    assert duplicate.outcome is GuestSessionOutcome.DUPLICATE
    assert fresh.outcome is GuestSessionOutcome.CLAIMED


async def test_processing_cancel_is_noop_and_valid_completion_charges_once(db):
    user_id, access_version = await create_guest(db, 96024)
    now = datetime(2026, 8, 6, 12, 30, tzinfo=UTC)
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96024,
        access_version=access_version,
        update_id=32,
        now=now,
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96024,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )

    cancel_attempt = await service.cancel(user_id=user_id, chat_id=96024, now=now)
    assert cancel_attempt.outcome is GuestSessionOutcome.IN_PROGRESS
    assert not cancel_attempt.changed
    assert cancel_attempt.session == bound.session
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        stored = await session.get(GuestDemoSession, bound.session.session_id)
        assert usage is not None and usage.status == GuestUsageStatus.RESERVED.value
        assert usage.completed_at is None
        assert stored is not None
        assert GuestSessionService._snapshot(stored) == bound.session

    completion = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96024,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "charged exactly once"},
        now=now,
    )
    assert completion.outcome is GuestReservationOutcome.SUCCEEDED
    assert completion.changed
    assert completion.session is not None
    assert completion.session.status is GuestSessionStatus.RESULT_READY
    assert completion.session.result_payload == {"summary": "charged exactly once"}
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        succeeded = await session.scalar(
            select(func.count(GuestUsageLedger.id)).where(
                GuestUsageLedger.user_id == user_id,
                GuestUsageLedger.status == GuestUsageStatus.SUCCEEDED.value,
            )
        )
        assert usage is not None and usage.status == GuestUsageStatus.SUCCEEDED.value
        assert usage.completed_at is not None
        assert succeeded == 1

    repeated = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96024,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "must not charge twice"},
        now=now,
    )
    assert repeated.outcome is GuestReservationOutcome.STALE
    assert not repeated.changed
    snapshot = await GuestQuotaService(db).snapshot(user_id, now=now)
    assert snapshot.successful_lifetime_count == 1
    assert snapshot.remaining_operations == 1


async def test_cancel_state_machine_preserves_processing_and_terminal_states(db):
    now = datetime(2026, 8, 6, 12, 45, tzinfo=UTC)

    awaiting_user, awaiting_access = await create_guest(db, 96025)
    awaiting_service = GuestSessionService(db)
    awaiting = await awaiting_service.start_session(
        user_id=awaiting_user,
        chat_id=96025,
        access_version=awaiting_access,
        demo_kind="first_step",
        prompt_message_id=1,
        now=now,
    )
    assert awaiting.session is not None
    cancelled = await awaiting_service.cancel(
        user_id=awaiting_user,
        chat_id=96025,
        now=now,
    )
    repeated_cancel = await awaiting_service.cancel(
        user_id=awaiting_user,
        chat_id=96025,
        now=now,
    )
    assert cancelled.outcome is GuestSessionOutcome.CANCELLED and cancelled.changed
    assert repeated_cancel.outcome is GuestSessionOutcome.CANCELLED
    assert not repeated_cancel.changed

    prebind_user, prebind_access = await create_guest(db, 96026)
    prebind_service = GuestSessionService(db)
    await prebind_service.start_session(
        user_id=prebind_user,
        chat_id=96026,
        access_version=prebind_access,
        demo_kind="thought_breakdown",
        prompt_message_id=2,
        now=now,
    )
    claimed = await prebind_service.claim_input(
        user_id=prebind_user,
        chat_id=96026,
        access_version=prebind_access,
        telegram_update_id=33,
        telegram_message_id=34,
        now=now,
    )
    assert claimed.session is not None
    prebind_cancel = await prebind_service.cancel(
        user_id=prebind_user,
        chat_id=96026,
        now=now,
    )
    assert prebind_cancel.outcome is GuestSessionOutcome.IN_PROGRESS
    assert not prebind_cancel.changed
    assert prebind_cancel.session == claimed.session

    completed_user, completed_access = await create_guest(db, 96027)
    completed_service, completed_bound, completed_reservation = await prepare_processing(
        db,
        user_id=completed_user,
        chat_id=96027,
        access_version=completed_access,
        update_id=35,
        now=now,
    )
    assert completed_bound.session is not None
    await begin_bound_provider(
        completed_service,
        reservation_token=completed_reservation.reservation_token,
        user_id=completed_user,
        chat_id=96027,
        access_version=completed_access,
        session_version=completed_bound.session.version,
        now=now,
    )
    result = await completed_service.complete_with_result(
        reservation_token=completed_reservation.reservation_token,
        user_id=completed_user,
        chat_id=96027,
        access_version=completed_access,
        session_version=completed_bound.session.version,
        result_payload={"summary": "delivered"},
        now=now,
    )
    assert result.session is not None
    delivered = await completed_service.mark_delivered(
        user_id=completed_user,
        chat_id=96027,
        session_version=result.session.version,
        now=now,
    )
    assert delivered.session is not None
    completed_cancel = await completed_service.cancel(
        user_id=completed_user,
        chat_id=96027,
        now=now,
    )
    assert completed_cancel.outcome is GuestSessionOutcome.COMPLETED
    assert not completed_cancel.changed
    assert completed_cancel.session == delivered.session

    expired_user, expired_access = await create_guest(db, 96028)
    expired_service = GuestSessionService(db)
    await expired_service.start_session(
        user_id=expired_user,
        chat_id=96028,
        access_version=expired_access,
        demo_kind="first_step",
        prompt_message_id=3,
        now=now,
    )
    expired = await expired_service.cleanup_expired(
        user_id=expired_user,
        chat_id=96028,
        now=now + timedelta(minutes=15),
    )
    assert expired.session is not None
    expired_cancel = await expired_service.cancel(
        user_id=expired_user,
        chat_id=96028,
        now=now + timedelta(minutes=15),
    )
    assert expired_cancel.outcome is GuestSessionOutcome.EXPIRED
    assert not expired_cancel.changed
    assert expired_cancel.session == expired.session


async def test_cancel_and_completion_race_never_refunds_a_valid_provider_result(db):
    user_id, access_version = await create_guest(db, 96029)
    now = datetime(2026, 8, 6, 12, 50, tzinfo=UTC)
    _service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96029,
        access_version=access_version,
        update_id=36,
        now=now,
    )
    assert bound.session is not None
    await begin_bound_provider(
        GuestSessionService(db),
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96029,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )

    cancel_result, completion = await asyncio.gather(
        GuestSessionService(db).cancel(user_id=user_id, chat_id=96029, now=now),
        GuestSessionService(db).complete_with_result(
            reservation_token=reservation.reservation_token,
            user_id=user_id,
            chat_id=96029,
            access_version=access_version,
            session_version=bound.session.version,
            result_payload={"summary": "race result"},
            now=now,
        ),
    )
    assert completion.outcome is GuestReservationOutcome.SUCCEEDED
    assert cancel_result.outcome in {
        GuestSessionOutcome.IN_PROGRESS,
        GuestSessionOutcome.CANCELLED,
    }
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        stored = await session.get(GuestDemoSession, bound.session.session_id)
        succeeded = await session.scalar(
            select(func.count(GuestUsageLedger.id)).where(
                GuestUsageLedger.user_id == user_id,
                GuestUsageLedger.status == GuestUsageStatus.SUCCEEDED.value,
            )
        )
        assert usage is not None and usage.status == GuestUsageStatus.SUCCEEDED.value
        assert usage.completed_at is not None
        assert succeeded == 1
        assert stored is not None
        assert stored.status in {
            GuestSessionStatus.RESULT_READY.value,
            GuestSessionStatus.CANCELLED.value,
        }
        if stored.status == GuestSessionStatus.RESULT_READY.value:
            assert stored.result_payload == {"summary": "race result"}
        else:
            assert stored.result_payload is None

    repeated = await GuestSessionService(db).complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96029,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "duplicate race result"},
        now=now,
    )
    assert repeated.outcome is GuestReservationOutcome.STALE
    assert not repeated.changed
    async with db.sessions() as session:
        succeeded = await session.scalar(
            select(func.count(GuestUsageLedger.id)).where(
                GuestUsageLedger.user_id == user_id,
                GuestUsageLedger.status == GuestUsageStatus.SUCCEEDED.value,
            )
        )
        assert succeeded == 1


async def test_completion_is_atomic_and_result_survives_delivery_failure_restart(db):
    user_id, access_version = await create_guest(db, 96006)
    now = datetime(2026, 8, 6, 13, tzinfo=UTC)
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96006,
        access_version=access_version,
        update_id=40,
        now=now,
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96006,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    payload = {"summary": "short", "steps": ["one", "two"]}
    completed = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96006,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload=payload,
        now=now,
    )
    assert completed.outcome is GuestReservationOutcome.SUCCEEDED
    assert completed.session is not None
    assert completed.session.status is GuestSessionStatus.RESULT_READY
    assert completed.session.result_payload == payload
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        stored = await session.get(GuestDemoSession, completed.session.session_id)
        assert usage is not None and usage.status == GuestUsageStatus.SUCCEEDED.value
        assert stored is not None and stored.status == GuestSessionStatus.RESULT_READY.value
        assert stored.result_payload == payload

    not_overwritten = await GuestSessionService(db).start_session(
        user_id=user_id,
        chat_id=96006,
        access_version=access_version,
        demo_kind="first_step",
        prompt_message_id=999,
        now=now + timedelta(seconds=1),
    )
    assert not_overwritten.outcome is GuestSessionOutcome.RESULT_PENDING
    assert not_overwritten.session is not None
    assert not_overwritten.session.version == completed.session.version
    assert not_overwritten.session.result_payload == payload

    # A Telegram edit failure is represented by deliberately not marking delivery.
    pending = await GuestSessionService(db).pending_result(
        user_id=user_id,
        chat_id=96006,
        access_version=access_version,
        now=now + timedelta(minutes=1),
    )
    assert pending.outcome is GuestSessionOutcome.RESULT_READY
    assert pending.session is not None and pending.session.result_payload == payload
    delivered = await GuestSessionService(db).mark_delivered(
        user_id=user_id,
        chat_id=96006,
        session_version=pending.session.version,
        now=now + timedelta(minutes=1),
    )
    assert delivered.outcome is GuestSessionOutcome.COMPLETED
    assert delivered.session is not None
    assert delivered.session.result_payload is None
    assert delivered.session.result_expires_at is None
    repeated = await GuestSessionService(db).mark_delivered(
        user_id=user_id,
        chat_id=96006,
        session_version=delivered.session.version,
        now=now + timedelta(minutes=1),
    )
    assert repeated.outcome is GuestSessionOutcome.COMPLETED
    assert not repeated.changed


async def test_result_ttl_and_cancel_clear_payload_without_refunding_success(db):
    user_id, access_version = await create_guest(db, 96007)
    now = datetime(2026, 8, 6, 14, tzinfo=UTC)
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96007,
        access_version=access_version,
        update_id=50,
        now=now,
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96007,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    completed = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96007,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "temporary"},
        now=now,
    )
    assert completed.session is not None
    expired = await GuestSessionService(db).pending_result(
        user_id=user_id,
        chat_id=96007,
        access_version=access_version,
        now=now + timedelta(minutes=15),
    )
    assert expired.outcome is GuestSessionOutcome.EXPIRED
    assert expired.session is not None and expired.session.result_payload is None
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        assert usage is not None and usage.status == GuestUsageStatus.SUCCEEDED.value

    # A new result is also removed immediately by explicit cancellation.
    service, bound, second = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96007,
        access_version=access_version,
        update_id=51,
        now=now + timedelta(minutes=16),
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=second.reservation_token,
        user_id=user_id,
        chat_id=96007,
        access_version=access_version,
        session_version=bound.session.version,
        now=now + timedelta(minutes=16),
    )
    ready = await service.complete_with_result(
        reservation_token=second.reservation_token,
        user_id=user_id,
        chat_id=96007,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "cancel me"},
        now=now + timedelta(minutes=16),
    )
    assert ready.session is not None
    cancelled = await service.cancel(
        user_id=user_id,
        chat_id=96007,
        now=now + timedelta(minutes=16),
    )
    assert cancelled.outcome is GuestSessionOutcome.CANCELLED
    assert cancelled.changed
    assert cancelled.session is not None and cancelled.session.result_payload is None
    repeated_cancel = await service.cancel(
        user_id=user_id,
        chat_id=96007,
        now=now + timedelta(minutes=16),
    )
    assert repeated_cancel.outcome is GuestSessionOutcome.CANCELLED
    assert not repeated_cancel.changed
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, second.usage_id)
        assert usage is not None and usage.status == GuestUsageStatus.SUCCEEDED.value


@pytest.mark.parametrize("new_tier", [SUBSCRIBER, ADMIN, BLOCKED])
async def test_access_change_during_processing_consumes_credit_but_drops_result(db, new_tier):
    user_id, access_version = await create_guest(db, 96010 + len(new_tier))
    now = datetime(2026, 8, 6, 15, tzinfo=UTC)
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96010 + len(new_tier),
        access_version=access_version,
        update_id=60 + len(new_tier),
        now=now,
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96010 + len(new_tier),
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    async with db.session() as session:
        user = await session.get(User, user_id)
        assert user is not None
        user.access_tier = new_tier
        user.access_version += 1
    completion = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96010 + len(new_tier),
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "must not be delivered"},
        now=now,
    )
    assert completion.outcome is GuestReservationOutcome.ACCESS_CHANGED
    assert completion.session is not None
    assert completion.session.status is GuestSessionStatus.CANCELLED
    assert completion.session.result_payload is None
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        assert usage is not None and usage.status == GuestUsageStatus.SUCCEEDED.value
    denied = await service.pending_result(
        user_id=user_id,
        chat_id=96010 + len(new_tier),
        access_version=access_version,
        now=now,
    )
    assert denied.outcome is GuestSessionOutcome.NOT_GUEST
    assert denied.session is not None and denied.session.result_payload is None


async def test_late_or_wrong_fencing_completion_changes_nothing(db):
    user_id, access_version = await create_guest(db, 96020)
    now = datetime(2026, 8, 6, 16, tzinfo=UTC)
    policy = GuestQuotaPolicy(reservation_ttl=timedelta(minutes=2))
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96020,
        access_version=access_version,
        update_id=70,
        now=now,
        policy=policy,
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96020,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    wrong = await service.complete_with_result(
        reservation_token="x" * 43,
        user_id=user_id,
        chat_id=96020,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "wrong"},
        now=now,
    )
    wrong_version = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96020,
        access_version=access_version,
        session_version=bound.session.version + 1,
        result_payload={"summary": "wrong version"},
        now=now,
    )
    late = await service.complete_with_result(
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96020,
        access_version=access_version,
        session_version=bound.session.version,
        result_payload={"summary": "late"},
        now=now + timedelta(minutes=2),
    )
    assert wrong.outcome is GuestReservationOutcome.STALE and not wrong.changed
    assert wrong_version.outcome is GuestReservationOutcome.STALE
    assert not wrong_version.changed
    assert late.outcome is GuestReservationOutcome.STALE and not late.changed
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        stored = await session.get(GuestDemoSession, bound.session.session_id)
        assert usage is not None and usage.status == GuestUsageStatus.RESERVED.value
        assert stored is not None and stored.status == GuestSessionStatus.PROCESSING.value
        assert stored.result_payload is None


async def test_result_validation_happens_before_any_database_mutation(db, caplog):
    user_id, access_version = await create_guest(db, 96021)
    now = datetime(2026, 8, 6, 17, tzinfo=UTC)
    policy = GuestQuotaPolicy(max_result_payload_bytes=1024)
    service, bound, reservation = await prepare_processing(
        db,
        user_id=user_id,
        chat_id=96021,
        access_version=access_version,
        update_id=80,
        now=now,
        policy=policy,
    )
    assert bound.session is not None
    await begin_bound_provider(
        service,
        reservation_token=reservation.reservation_token,
        user_id=user_id,
        chat_id=96021,
        access_version=access_version,
        session_version=bound.session.version,
        now=now,
    )
    invalid_payloads = [
        [],
        {},
        {"nested": {"prompt": "secret"}},
        {"summary": float("nan")},
        {"summary": "x" * 1100},
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            await service.complete_with_result(
                reservation_token=reservation.reservation_token,
                user_id=user_id,
                chat_id=96021,
                access_version=access_version,
                session_version=bound.session.version,
                result_payload=payload,
                now=now,
            )
    async with db.sessions() as session:
        usage = await session.get(GuestUsageLedger, reservation.usage_id)
        stored = await session.get(GuestDemoSession, bound.session.session_id)
        assert usage is not None and usage.status == GuestUsageStatus.RESERVED.value
        assert stored is not None and stored.status == GuestSessionStatus.PROCESSING.value
        assert stored.result_payload is None
    assert "secret" not in caplog.text


async def test_raw_input_has_no_session_api_or_database_column(db, caplog):
    sentinel = "RAW_GUEST_INPUT_MUST_NOT_PERSIST"
    user_id, access_version = await create_guest(db, 96022)
    now = datetime(2026, 8, 6, 18, tzinfo=UTC)
    service = GuestSessionService(db)
    await service.start_session(
        user_id=user_id,
        chat_id=96022,
        access_version=access_version,
        demo_kind="first_step",
        prompt_message_id=90,
        now=now,
    )
    assert "input" not in inspect.signature(service.claim_input).parameters
    assert "text" not in inspect.signature(service.claim_input).parameters
    columns = set(GuestDemoSession.__table__.columns.keys())
    assert not columns & {"input", "raw_input", "prompt", "provider_response", "error_body"}
    async with db.sessions() as session:
        row = await session.scalar(
            select(GuestDemoSession).where(GuestDemoSession.user_id == user_id)
        )
        assert row is not None
        assert sentinel not in repr(row.__dict__)
    assert sentinel not in caplog.text


async def test_quota_denial_resolves_processing_session_explicitly(db):
    user_id, access_version = await create_guest(db, 96023)
    now = datetime(2026, 8, 6, 19, tzinfo=UTC)
    service = GuestSessionService(db)
    await service.start_session(
        user_id=user_id,
        chat_id=96023,
        access_version=access_version,
        demo_kind="first_step",
        prompt_message_id=None,
        now=now,
    )
    claimed = await service.claim_input(
        user_id=user_id,
        chat_id=96023,
        access_version=access_version,
        telegram_update_id=91,
        telegram_message_id=92,
        now=now,
    )
    assert claimed.session is not None
    retry = await service.resolve_reservation_denial(
        user_id=user_id,
        chat_id=96023,
        access_version=access_version,
        session_version=claimed.session.version,
        denial_reason=GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED,
        now=now,
    )
    assert retry.outcome is GuestSessionOutcome.RETRY_READY
    assert retry.session is not None
    claimed_again = await service.claim_input(
        user_id=user_id,
        chat_id=96023,
        access_version=access_version,
        telegram_update_id=93,
        telegram_message_id=94,
        now=now,
    )
    assert claimed_again.session is not None
    cancelled = await service.resolve_reservation_denial(
        user_id=user_id,
        chat_id=96023,
        access_version=access_version,
        session_version=claimed_again.session.version,
        denial_reason=GuestQuotaDenialReason.DISABLED,
        now=now,
    )
    assert cancelled.outcome is GuestSessionOutcome.CANCELLED
    assert cancelled.session is not None
    assert cancelled.session.status is GuestSessionStatus.CANCELLED
