import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, func, select, update

from future_self.access import ADMIN, BLOCKED, SUBSCRIBER
from future_self.config import Settings
from future_self.db import Database
from future_self.guest_access import (
    GuestDemoKind,
    GuestQuotaDenialReason,
    GuestQuotaPolicy,
    GuestQuotaService,
    GuestReservationOutcome,
    GuestSessionService,
    GuestUsageStatus,
)
from future_self.models import GuestQuotaDay, GuestUsageLedger, User
from future_self.repositories import UserRepository


async def create_user(db, telegram_id: int, *, tier: str = "guest", completed: bool = False):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(telegram_id, "Europe/Moscow")
        user.access_tier = tier
        user.onboarding_completed = completed
        user_id = user.id
    return user_id


async def complete_demo(
    db,
    *,
    user_id: int,
    chat_id: int,
    update_id: int,
    key: str,
    now: datetime,
    policy: GuestQuotaPolicy | None = None,
):
    quota = GuestQuotaService(db, policy)
    sessions = GuestSessionService(db, policy)
    started = await sessions.start_session(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        demo_kind=GuestDemoKind.THOUGHT_BREAKDOWN,
        prompt_message_id=update_id,
        now=now,
    )
    claimed = await sessions.claim_input(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        telegram_update_id=update_id,
        telegram_message_id=update_id,
        now=now,
    )
    reserved = await quota.reserve(
        user_id=user_id,
        demo_kind=GuestDemoKind.THOUGHT_BREAKDOWN,
        idempotency_key=key,
        telegram_update_id=update_id,
        now=now,
    )
    assert started.session is not None
    assert claimed.session is not None
    assert reserved.reservation is not None
    bound = await sessions.bind_reservation(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        session_version=claimed.session.version,
        reservation_token=reserved.reservation.reservation_token,
        now=now,
    )
    assert bound.session is not None
    provider_start = await sessions.begin_provider_call(
        reservation_token=reserved.reservation.reservation_token,
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        session_version=bound.session.version,
        now=now,
    )
    assert provider_start.can_invoke_provider
    completed = await sessions.complete_with_result(
        reservation_token=reserved.reservation.reservation_token,
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        session_version=bound.session.version,
        result_payload={"summary": "structured", "next_step": "small step"},
        now=now,
    )
    assert completed.outcome is GuestReservationOutcome.SUCCEEDED
    assert completed.session is not None
    await sessions.mark_delivered(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        session_version=completed.session.version,
        now=now,
    )
    return reserved.reservation


async def prepare_bound_demo(
    db,
    *,
    user_id: int,
    chat_id: int,
    update_id: int,
    key: str,
    now: datetime,
    policy: GuestQuotaPolicy | None = None,
):
    sessions = GuestSessionService(db, policy)
    started = await sessions.start_session(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        demo_kind=GuestDemoKind.FIRST_STEP,
        prompt_message_id=update_id,
        now=now,
    )
    claimed = await sessions.claim_input(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        telegram_update_id=update_id,
        telegram_message_id=update_id,
        now=now,
    )
    reservation = await GuestQuotaService(db, policy).reserve(
        user_id=user_id,
        demo_kind=GuestDemoKind.FIRST_STEP,
        idempotency_key=key,
        telegram_update_id=update_id,
        now=now,
    )
    assert started.session is not None
    assert claimed.session is not None
    assert reservation.can_bind_session and reservation.reservation is not None
    bound = await sessions.bind_reservation(
        user_id=user_id,
        chat_id=chat_id,
        access_version=1,
        session_version=claimed.session.version,
        reservation_token=reservation.reservation.reservation_token,
        now=now,
    )
    assert bound.session is not None
    return sessions, bound, reservation.reservation


def test_guest_settings_are_typed_bounded_and_do_not_add_an_access_gate_switch():
    settings = Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
    )
    assert settings.guest_ai_enabled is True
    assert settings.guest_operation_limit == 2
    assert settings.guest_global_daily_limit == 50
    assert settings.guest_reservation_ttl_minutes == 10
    assert not hasattr(settings, "enable_guest_mode")
    for field, value in (
        ("guest_operation_limit", 0),
        ("guest_global_daily_limit", 0),
        ("guest_reservation_ttl_minutes", 1),
        ("guest_input_ttl_minutes", 4),
        ("guest_result_ttl_minutes", 121),
    ):
        with pytest.raises(ValueError):
            Settings(
                _env_file=None,
                telegram_bot_token="123456:TEST",
                ai_api_key="test-key",
                **{field: value},
            )

    disabled = GuestQuotaService(None, GuestQuotaPolicy(enabled=False))
    decision = asyncio.run(
        disabled.reserve(
            user_id=0,
            demo_kind="unsupported",
            idempotency_key="not valid",
            telegram_update_id=-1,
        )
    )
    assert decision.denial_reason is GuestQuotaDenialReason.DISABLED


async def test_two_successes_consume_lifetime_quota_and_third_is_denied(db):
    user_id = await create_user(db, 95001)
    now = datetime(2026, 8, 6, 9, tzinfo=UTC)
    await complete_demo(db, user_id=user_id, chat_id=95001, update_id=1, key="op:1", now=now)
    await complete_demo(db, user_id=user_id, chat_id=95001, update_id=2, key="op:2", now=now)

    third = await GuestQuotaService(db).reserve(
        user_id=user_id,
        demo_kind=GuestDemoKind.FIRST_STEP,
        idempotency_key="op:3",
        telegram_update_id=3,
        now=now,
    )
    assert third.denial_reason is GuestQuotaDenialReason.LIFETIME_EXHAUSTED
    snapshot = await GuestQuotaService(db).snapshot(user_id, now=now)
    assert snapshot.successful_lifetime_count == 2
    assert snapshot.remaining_operations == 0
    assert snapshot.exhausted is True


async def test_failed_and_expired_reservations_do_not_consume_attempts(db):
    user_id = await create_user(db, 95002)
    now = datetime(2026, 8, 6, 9, tzinfo=UTC)
    quota = GuestQuotaService(db, GuestQuotaPolicy(reservation_ttl=timedelta(minutes=2)))

    failed = await quota.reserve(
        user_id=user_id,
        demo_kind="thought_breakdown",
        idempotency_key="failed:1",
        telegram_update_id=1,
        now=now,
    )
    assert failed.reservation is not None
    transition = await quota.fail(failed.reservation.reservation_token, now=now)
    assert (transition.outcome, transition.changed) == (GuestReservationOutcome.FAILED, True)
    repeated = await quota.fail(failed.reservation.reservation_token, now=now)
    assert (repeated.outcome, repeated.changed) == (GuestReservationOutcome.FAILED, False)

    expiring = await quota.reserve(
        user_id=user_id,
        demo_kind="first_step",
        idempotency_key="expired:1",
        telegram_update_id=2,
        now=now,
    )
    assert expiring.reservation is not None
    expired = await quota.expire(
        expiring.reservation.reservation_token,
        now=now + timedelta(minutes=2),
    )
    assert (expired.outcome, expired.changed) == (GuestReservationOutcome.EXPIRED, True)

    fresh = await quota.reserve(
        user_id=user_id,
        demo_kind="first_step",
        idempotency_key="fresh:1",
        telegram_update_id=3,
        now=now + timedelta(minutes=3),
    )
    assert fresh.can_bind_session
    snapshot = await quota.snapshot(user_id, now=now + timedelta(minutes=3))
    assert snapshot.successful_lifetime_count == 0
    assert snapshot.active_reservation_count == 1
    assert snapshot.remaining_operations == 1


async def test_active_reservation_and_idempotency_decisions_are_distinct(db):
    user_id = await create_user(db, 95003)
    now = datetime(2026, 8, 6, 10, tzinfo=UTC)
    quota = GuestQuotaService(db)
    first = await quota.reserve(
        user_id=user_id,
        demo_kind="thought_breakdown",
        idempotency_key="same:1",
        telegram_update_id=10,
        now=now,
    )
    duplicate = await quota.reserve(
        user_id=user_id,
        demo_kind="thought_breakdown",
        idempotency_key="same:1",
        telegram_update_id=10,
        now=now,
    )
    other = await quota.reserve(
        user_id=user_id,
        demo_kind="first_step",
        idempotency_key="other:1",
        telegram_update_id=11,
        now=now,
    )
    assert first.can_bind_session
    assert duplicate.denial_reason is GuestQuotaDenialReason.DUPLICATE_RESERVED
    assert duplicate.reservation is not None and first.reservation is not None
    assert duplicate.reservation.reservation_token == first.reservation.reservation_token
    assert other.denial_reason is GuestQuotaDenialReason.IN_PROGRESS
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 1


async def test_duplicate_succeeded_failed_and_expired_keys_are_terminal(db):
    user_id = await create_user(db, 95004)
    now = datetime(2026, 8, 6, 11, tzinfo=UTC)
    await complete_demo(
        db,
        user_id=user_id,
        chat_id=95004,
        update_id=1,
        key="done:1",
        now=now,
    )
    quota = GuestQuotaService(db, GuestQuotaPolicy(reservation_ttl=timedelta(minutes=2)))
    succeeded = await quota.reserve(
        user_id=user_id,
        demo_kind="thought_breakdown",
        idempotency_key="done:1",
        telegram_update_id=1,
        now=now,
    )
    assert succeeded.denial_reason is GuestQuotaDenialReason.DUPLICATE_SUCCEEDED

    failed = await quota.reserve(
        user_id=user_id,
        demo_kind="first_step",
        idempotency_key="terminal:failed",
        telegram_update_id=2,
        now=now,
    )
    assert failed.reservation is not None
    await quota.fail(failed.reservation.reservation_token, now=now)
    failed_duplicate = await quota.reserve(
        user_id=user_id,
        demo_kind="first_step",
        idempotency_key="terminal:failed",
        telegram_update_id=2,
        now=now,
    )
    assert failed_duplicate.denial_reason is GuestQuotaDenialReason.DUPLICATE_TERMINAL

    expired = await quota.reserve(
        user_id=user_id,
        demo_kind="first_step",
        idempotency_key="terminal:expired",
        telegram_update_id=3,
        now=now,
    )
    assert expired.reservation is not None
    await quota.expire(expired.reservation.reservation_token, now=now + timedelta(minutes=2))
    expired_duplicate = await quota.reserve(
        user_id=user_id,
        demo_kind="first_step",
        idempotency_key="terminal:expired",
        telegram_update_id=3,
        now=now + timedelta(minutes=2),
    )
    assert expired_duplicate.denial_reason is GuestQuotaDenialReason.DUPLICATE_TERMINAL


async def test_emergency_switch_tiers_and_onboarding_are_independent(db):
    guest_incomplete = await create_user(db, 95010, completed=False)
    guest_complete = await create_user(db, 95011, completed=True)
    subscriber = await create_user(db, 95012, tier=SUBSCRIBER)
    admin = await create_user(db, 95013, tier=ADMIN)
    blocked = await create_user(db, 95014, tier=BLOCKED)
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)

    disabled = GuestQuotaService(db, GuestQuotaPolicy(enabled=False))
    decision = await disabled.reserve(
        user_id=guest_incomplete,
        demo_kind="first_step",
        idempotency_key="disabled:1",
        telegram_update_id=1,
        now=now,
    )
    assert decision.denial_reason is GuestQuotaDenialReason.DISABLED
    for offset, user_id in enumerate((subscriber, admin, blocked), start=2):
        denied = await GuestQuotaService(db).reserve(
            user_id=user_id,
            demo_kind="first_step",
            idempotency_key=f"tier:{offset}",
            telegram_update_id=offset,
            now=now,
        )
        assert denied.denial_reason is GuestQuotaDenialReason.NOT_GUEST
    for offset, user_id in enumerate((guest_incomplete, guest_complete), start=10):
        allowed = await GuestQuotaService(db).reserve(
            user_id=user_id,
            demo_kind="first_step",
            idempotency_key=f"guest:{offset}",
            telegram_update_id=offset,
            now=now,
        )
        assert allowed.can_bind_session


async def test_same_user_concurrency_allows_one_live_reservation(db):
    user_id = await create_user(db, 95020)
    now = datetime(2026, 8, 6, 13, tzinfo=UTC)
    services = [GuestQuotaService(db) for _ in range(8)]
    decisions = await asyncio.gather(
        *(
            service.reserve(
                user_id=user_id,
                demo_kind="thought_breakdown",
                idempotency_key=f"race:{index}",
                telegram_update_id=100 + index,
                now=now,
            )
            for index, service in enumerate(services)
        )
    )
    assert sum(decision.is_new for decision in decisions) == 1
    assert {decision.denial_reason for decision in decisions if not decision.is_new} == {
        GuestQuotaDenialReason.IN_PROGRESS
    }


async def test_duplicate_key_concurrency_has_one_new_row_and_one_token(db):
    user_id = await create_user(db, 95021)
    now = datetime(2026, 8, 6, 14, tzinfo=UTC)
    decisions = await asyncio.gather(
        *(
            GuestQuotaService(db).reserve(
                user_id=user_id,
                demo_kind="thought_breakdown",
                idempotency_key="duplicate:race",
                telegram_update_id=200,
                now=now,
            )
            for _ in range(8)
        )
    )
    assert sum(decision.is_new for decision in decisions) == 1
    tokens = {
        decision.reservation.reservation_token
        for decision in decisions
        if decision.reservation is not None
    }
    assert len(tokens) == 1
    assert (
        sum(
            decision.denial_reason is GuestQuotaDenialReason.DUPLICATE_RESERVED
            for decision in decisions
        )
        == 7
    )


async def test_global_limit_tracks_provisional_and_started_calls_across_utc_midnight(db):
    first = await create_user(db, 95030)
    second = await create_user(db, 95031)
    prefailed = await create_user(db, 95032)
    after_prefailure = await create_user(db, 95033)
    policy = GuestQuotaPolicy(global_daily_limit=1, reservation_ttl=timedelta(minutes=2))
    quota = GuestQuotaService(db, policy)
    before_midnight = datetime(2026, 8, 6, 23, 59, tzinfo=UTC)
    sessions, bound, reserved = await prepare_bound_demo(
        db,
        user_id=first,
        chat_id=95030,
        update_id=1,
        key="day:one",
        now=before_midnight,
        policy=policy,
    )
    midnight = datetime(2026, 8, 7, 0, 0, tzinfo=UTC)
    before_start = await quota.snapshot(first, now=midnight)
    denied_while_provisional = await quota.reserve(
        user_id=second,
        demo_kind="first_step",
        idempotency_key="day:two",
        telegram_update_id=2,
        now=midnight,
    )
    assert before_start.global_used == 1
    assert denied_while_provisional.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED

    assert bound.session is not None
    provider_start = await sessions.begin_provider_call(
        reservation_token=reserved.reservation_token,
        user_id=first,
        chat_id=95030,
        access_version=1,
        session_version=bound.session.version,
        now=midnight,
    )
    assert provider_start.can_invoke_provider
    after_start = await quota.snapshot(first, now=midnight)
    assert after_start.global_used == 1
    failed = await sessions.fail_processing(
        reservation_token=reserved.reservation_token,
        user_id=first,
        chat_id=95030,
        access_version=1,
        session_version=bound.session.version,
        now=midnight,
    )
    assert failed.outcome is GuestReservationOutcome.FAILED
    after_started_failure = await quota.snapshot(first, now=midnight)
    assert after_started_failure.global_used == 1
    assert after_started_failure.successful_lifetime_count == 0
    assert after_started_failure.remaining_operations == 2
    still_denied = await quota.reserve(
        user_id=second,
        demo_kind="first_step",
        idempotency_key="day:still-used",
        telegram_update_id=3,
        now=midnight,
    )
    assert still_denied.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED

    next_boundary = datetime(2026, 8, 8, 23, 59, tzinfo=UTC)
    pre_sessions, pre_bound, pre_reservation = await prepare_bound_demo(
        db,
        user_id=prefailed,
        chat_id=95032,
        update_id=4,
        key="day:pre-failure",
        now=next_boundary,
        policy=policy,
    )
    next_midnight = datetime(2026, 8, 9, 0, 0, tzinfo=UTC)
    assert (await quota.snapshot(prefailed, now=next_midnight)).global_used == 1
    assert pre_bound.session is not None
    pre_failure = await pre_sessions.fail_processing(
        reservation_token=pre_reservation.reservation_token,
        user_id=prefailed,
        chat_id=95032,
        access_version=1,
        session_version=pre_bound.session.version,
        now=next_midnight,
    )
    assert pre_failure.outcome is GuestReservationOutcome.FAILED
    assert (await quota.snapshot(prefailed, now=next_midnight)).global_used == 0
    released = await quota.reserve(
        user_id=after_prefailure,
        demo_kind="first_step",
        idempotency_key="day:released",
        telegram_update_id=5,
        now=next_midnight,
    )
    assert released.can_bind_session


async def test_fifty_cross_midnight_provider_starts_and_failures_exhaust_global_day(db):
    user_ids = [await create_user(db, 95100 + index) for index in range(51)]
    before_midnight = datetime(2026, 8, 6, 23, 59, tzinfo=UTC)
    midnight = datetime(2026, 8, 7, 0, 0, tzinfo=UTC)
    services = [GuestSessionService(db) for _ in user_ids]
    await asyncio.gather(
        *(
            service.start_session(
                user_id=user_id,
                chat_id=95100 + index,
                access_version=1,
                demo_kind="first_step",
                prompt_message_id=1000 + index,
                now=before_midnight,
            )
            for index, (service, user_id) in enumerate(zip(services, user_ids, strict=True))
        )
    )
    claims = await asyncio.gather(
        *(
            service.claim_input(
                user_id=user_id,
                chat_id=95100 + index,
                access_version=1,
                telegram_update_id=1000 + index,
                telegram_message_id=2000 + index,
                now=before_midnight,
            )
            for index, (service, user_id) in enumerate(zip(services, user_ids, strict=True))
        )
    )
    decisions = await asyncio.gather(
        *(
            GuestQuotaService(db).reserve(
                user_id=user_id,
                demo_kind="first_step",
                idempotency_key=f"global:{index}",
                telegram_update_id=1000 + index,
                now=before_midnight,
            )
            for index, user_id in enumerate(user_ids)
        )
    )
    assert sum(decision.is_new for decision in decisions) == 50
    assert (
        sum(
            decision.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED
            for decision in decisions
        )
        == 1
    )
    allowed = [
        (index, decision) for index, decision in enumerate(decisions) if decision.can_bind_session
    ]
    denied_index = next(
        index
        for index, decision in enumerate(decisions)
        if decision.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED
    )
    bound_decisions = await asyncio.gather(
        *(
            services[index].bind_reservation(
                user_id=user_ids[index],
                chat_id=95100 + index,
                access_version=1,
                session_version=claims[index].session.version,
                reservation_token=decision.reservation.reservation_token,
                now=before_midnight,
            )
            for index, decision in allowed
        )
    )
    denied_at_midnight = await GuestQuotaService(db).reserve(
        user_id=user_ids[denied_index],
        demo_kind="first_step",
        idempotency_key="global:midnight-denied",
        telegram_update_id=9000,
        now=midnight,
    )
    assert denied_at_midnight.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED

    starts = await asyncio.gather(
        *(
            services[index].begin_provider_call(
                reservation_token=decision.reservation.reservation_token,
                user_id=user_ids[index],
                chat_id=95100 + index,
                access_version=1,
                session_version=bound.session.version,
                now=midnight,
            )
            for (index, decision), bound in zip(allowed, bound_decisions, strict=True)
        )
    )
    assert sum(start.can_invoke_provider for start in starts) == 50
    failures = await asyncio.gather(
        *(
            services[index].fail_processing(
                reservation_token=decision.reservation.reservation_token,
                user_id=user_ids[index],
                chat_id=95100 + index,
                access_version=1,
                session_version=bound.session.version,
                now=midnight,
            )
            for (index, decision), bound in zip(allowed, bound_decisions, strict=True)
        )
    )
    assert all(failure.outcome is GuestReservationOutcome.FAILED for failure in failures)
    denied_after_failures = await GuestQuotaService(db).reserve(
        user_id=user_ids[denied_index],
        demo_kind="first_step",
        idempotency_key="global:after-started-failures",
        telegram_update_id=9001,
        now=midnight,
    )
    assert denied_after_failures.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED
    user_snapshot = await GuestQuotaService(db).snapshot(user_ids[allowed[0][0]], now=midnight)
    assert user_snapshot.successful_lifetime_count == 0
    assert user_snapshot.active_reservation_count == 0
    assert user_snapshot.remaining_operations == 2
    assert user_snapshot.global_used == 50
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(GuestUsageLedger.id)).where(
                    GuestUsageLedger.status == GuestUsageStatus.FAILED.value,
                    GuestUsageLedger.provider_started_at.is_not(None),
                )
            )
            == 50
        )
        assert await session.scalar(select(func.count(GuestQuotaDay.quota_day))) == 2


async def test_pre_provider_failures_release_global_slots_without_spending_lifetime(db):
    user_ids = [await create_user(db, 95210 + index) for index in range(4)]
    now = datetime(2026, 8, 10, 10, tzinfo=UTC)
    policy = GuestQuotaPolicy(global_daily_limit=1)
    quota = GuestQuotaService(db, policy)
    for index, user_id in enumerate(user_ids[:3]):
        decision = await quota.reserve(
            user_id=user_id,
            demo_kind="first_step",
            idempotency_key=f"pre-failure:{index}",
            telegram_update_id=3000 + index,
            now=now,
        )
        assert decision.can_bind_session and decision.reservation is not None
        failed = await quota.fail(decision.reservation.reservation_token, now=now)
        assert failed.outcome is GuestReservationOutcome.FAILED
        snapshot = await quota.snapshot(user_id, now=now)
        assert snapshot.global_used == 0
        assert snapshot.successful_lifetime_count == 0
        assert snapshot.remaining_operations == 2

    final = await quota.reserve(
        user_id=user_ids[3],
        demo_kind="first_step",
        idempotency_key="pre-failure:final",
        telegram_update_id=3004,
        now=now,
    )
    assert final.can_bind_session
    async with db.sessions() as session:
        rows = (
            await session.scalars(
                select(GuestUsageLedger).where(
                    GuestUsageLedger.status == GuestUsageStatus.FAILED.value
                )
            )
        ).all()
        assert len(rows) == 3
        assert all(row.provider_started_at is None for row in rows)


async def test_succeeded_started_failure_and_live_unstarted_share_one_global_limit(db):
    users = [await create_user(db, 95220 + index) for index in range(4)]
    now = datetime(2026, 8, 10, 11, tzinfo=UTC)
    policy = GuestQuotaPolicy(global_daily_limit=3)
    await complete_demo(
        db,
        user_id=users[0],
        chat_id=95220,
        update_id=4000,
        key="mixed:succeeded",
        now=now,
        policy=policy,
    )
    sessions, bound, reservation = await prepare_bound_demo(
        db,
        user_id=users[1],
        chat_id=95221,
        update_id=4001,
        key="mixed:started-failed",
        now=now,
        policy=policy,
    )
    assert bound.session is not None
    provider_start = await sessions.begin_provider_call(
        reservation_token=reservation.reservation_token,
        user_id=users[1],
        chat_id=95221,
        access_version=1,
        session_version=bound.session.version,
        now=now,
    )
    assert provider_start.can_invoke_provider
    failure = await sessions.fail_processing(
        reservation_token=reservation.reservation_token,
        user_id=users[1],
        chat_id=95221,
        access_version=1,
        session_version=bound.session.version,
        now=now,
    )
    assert failure.outcome is GuestReservationOutcome.FAILED
    live = await GuestQuotaService(db, policy).reserve(
        user_id=users[2],
        demo_kind="first_step",
        idempotency_key="mixed:live-unstarted",
        telegram_update_id=4002,
        now=now,
    )
    assert live.can_bind_session
    denied = await GuestQuotaService(db, policy).reserve(
        user_id=users[3],
        demo_kind="first_step",
        idempotency_key="mixed:denied",
        telegram_update_id=4003,
        now=now,
    )
    assert denied.denial_reason is GuestQuotaDenialReason.GLOBAL_DAILY_EXHAUSTED
    failed_snapshot = await GuestQuotaService(db, policy).snapshot(users[1], now=now)
    assert failed_snapshot.global_used == 3
    assert failed_snapshot.successful_lifetime_count == 0
    assert failed_snapshot.remaining_operations == 2


async def test_file_backed_independent_databases_share_lock_and_restart_counts(db):
    user_id = await create_user(db, 95200)
    second_db = Database(db.url)
    now = datetime(2026, 8, 6, 16, tzinfo=UTC)
    try:
        first, duplicate = await asyncio.gather(
            GuestQuotaService(db).reserve(
                user_id=user_id,
                demo_kind="thought_breakdown",
                idempotency_key="independent:1",
                telegram_update_id=1,
                now=now,
            ),
            GuestQuotaService(second_db).reserve(
                user_id=user_id,
                demo_kind="thought_breakdown",
                idempotency_key="independent:1",
                telegram_update_id=1,
                now=now,
            ),
        )
        assert sorted((first.is_new, duplicate.is_new)) == [False, True]
        restarted = GuestQuotaService(second_db)
        snapshot = await restarted.snapshot(user_id, now=now)
        assert snapshot.active_reservation_count == 1
        assert snapshot.remaining_operations == 1
    finally:
        await second_db.dispose()


async def test_sqlite_writer_timeout_denies_without_creating_a_reservation(db):
    user_id = await create_user(db, 95201)
    contender = Database(db.url, sqlite_busy_timeout_ms=1_000)
    now = datetime(2026, 8, 6, 17, tzinfo=UTC)
    try:
        async with db.engine.connect() as connection:
            transaction = await connection.begin()
            await connection.execute(
                update(User).where(User.id == user_id).values(updated_at=User.updated_at)
            )
            decision = await GuestQuotaService(contender).reserve(
                user_id=user_id,
                demo_kind="first_step",
                idempotency_key="locked:1",
                telegram_update_id=1,
                now=now,
            )
            await transaction.rollback()
        assert decision.denial_reason is GuestQuotaDenialReason.UNAVAILABLE
        assert decision.reservation is None
        async with db.sessions() as session:
            assert await session.scalar(select(func.count(GuestUsageLedger.id))) == 0
    finally:
        await contender.dispose()


async def test_snapshot_is_read_only_and_does_not_create_a_user(db):
    statements: list[str] = []

    def observe_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement.lstrip().split(maxsplit=1)[0].upper())

    event.listen(db.engine.sync_engine, "before_cursor_execute", observe_sql)
    try:
        snapshot = await GuestQuotaService(db).snapshot(999_999)
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", observe_sql)

    assert snapshot.successful_lifetime_count == 0
    assert snapshot.active_reservation_count == 0
    assert snapshot.remaining_operations == 0
    assert snapshot.global_used == 0
    assert statements and set(statements) == {"SELECT"}
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(User.id))) == 0


async def test_ledger_schema_contains_no_input_result_prompt_or_error_body(db):
    columns = set(GuestUsageLedger.__table__.columns.keys())
    assert columns == {
        "id",
        "user_id",
        "demo_kind",
        "status",
        "idempotency_key",
        "telegram_update_id",
        "reservation_token",
        "quota_day",
        "reserved_at",
        "expires_at",
        "provider_started_at",
        "completed_at",
    }
    assert not columns & {"input", "prompt", "result_payload", "response", "error_body"}
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(User.id))) == 0
