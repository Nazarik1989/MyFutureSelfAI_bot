from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID

import pytest
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import func, select, update

from future_self.db import Database
from future_self.models import (
    InboxItem,
    RecurringTaskReminderSchedule,
    TaskReminder,
    TaskState,
    User,
    WeeklyFocus,
    WeeklyFocusChange,
    WeeklyReviewSession,
)
from future_self.weekly_review import (
    WeeklyReminderCandidate,
    WeeklyReviewPhase,
    WeeklyReviewService,
    WeeklyReviewValidationError,
    current_week_start,
    normalize_weekly_candidates,
    normalize_weekly_focus,
    normalize_weekly_steps,
    target_week_start,
    weekly_review_week,
)

NOW = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
WEEK = date(2026, 8, 17)


async def _add_user(
    db,
    telegram_id: int,
    *,
    tier: str = "subscriber",
    access_version: int = 1,
    timezone: str = "Europe/Moscow",
) -> User:
    async with db.session() as session:
        user = User(
            telegram_id=telegram_id,
            access_tier=tier,
            access_version=access_version,
            onboarding_completed=True,
            timezone=timezone,
        )
        session.add(user)
        await session.flush()
        return user


async def _after_locked_owner_write(
    db,
    peer: Database,
    *,
    owner_id: int,
    operation,
    clock_value: list[datetime],
    after_lock: datetime,
):
    """Run a peer mutation only after proving it waited on the owner lock."""

    attempted = asyncio.Event()

    def observe_owner_lock(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("UPDATE USERS"):
            attempted.set()

    sqlalchemy_event.listen(peer.engine.sync_engine, "before_cursor_execute", observe_owner_lock)
    task: asyncio.Task | None = None
    try:
        async with db.sessions() as blocker:
            await blocker.execute(
                update(User).where(User.id == owner_id).values(updated_at=User.updated_at)
            )
            task = asyncio.create_task(operation())
            await asyncio.wait_for(attempted.wait(), timeout=2)
            assert not task.done()
            clock_value[0] = after_lock
            await blocker.commit()
            return await asyncio.wait_for(task, timeout=5)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        sqlalchemy_event.remove(
            peer.engine.sync_engine,
            "before_cursor_execute",
            observe_owner_lock,
        )


async def _preview_session(
    service: WeeklyReviewService,
    telegram_id: int,
    *,
    chat_id: int | None = None,
    focus: str = "Спокойно закрывать подтверждённые дела",
    approach: str | None = "Двигаться через один небольшой шаг",
    steps: tuple[str, ...] = ("Открыть список", "Выбрать одно дело"),
    candidates: tuple[WeeklyReminderCandidate, ...] = (),
    source: str = "text",
    now: datetime = NOW,
):
    destination = chat_id or telegram_id
    created = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=destination,
        expected_access_version=1,
        target_week_start=WEEK,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
        now=now,
    )
    assert created.session is not None
    processing = await service.mark_processing(
        telegram_actor_id=telegram_id,
        chat_id=destination,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=501,
        now=now,
    )
    assert processing.session is not None
    preview = await service.store_extraction(
        telegram_actor_id=telegram_id,
        chat_id=destination,
        expected_access_version=1,
        session_public_id=processing.session.public_id,
        expected_session_version=processing.session.version,
        canonical_message_id=501,
        focus=focus,
        approach=approach,
        small_steps=steps,
        reminder_candidates=candidates,
        source=source,
        now=now,
    )
    assert preview.status == "updated"
    assert preview.session is not None
    return preview.session


def test_calendar_cycle_uses_local_iana_week_and_exact_date_range():
    monday = datetime(2026, 8, 17, 8, 0, tzinfo=UTC)
    saturday = datetime(2026, 8, 22, 8, 0, tzinfo=UTC)
    sunday = datetime(2026, 8, 23, 8, 0, tzinfo=UTC)
    utc_sunday_but_moscow_monday = datetime(2026, 8, 23, 21, 30, tzinfo=UTC)

    assert current_week_start("Europe/Moscow", now=monday) == date(2026, 8, 17)
    assert target_week_start("Europe/Moscow", now=monday) == date(2026, 8, 17)
    assert target_week_start("Europe/Moscow", now=saturday) == date(2026, 8, 17)
    assert target_week_start("Europe/Moscow", now=sunday) == date(2026, 8, 17)
    assert target_week_start(
        "Europe/Moscow",
        now=sunday,
        review_weekday=0,
    ) == date(2026, 8, 17)
    assert target_week_start("Europe/Moscow", now=monday, scheduled=True) == date(2026, 8, 24)
    assert target_week_start(
        "Europe/Moscow",
        now=utc_sunday_but_moscow_monday,
    ) == date(2026, 8, 24)
    assert weekly_review_week(date(2026, 8, 17)).end == date(2026, 8, 23)

    with pytest.raises(WeeklyReviewValidationError):
        weekly_review_week(date(2026, 8, 18))
    with pytest.raises(WeeklyReviewValidationError):
        target_week_start("Not/A-Timezone", now=monday)


def test_local_week_bounds_follow_dst_without_changing_mon_sun_cycles():
    spring_start, spring_end = WeeklyReviewService._local_week_bounds_utc(
        "America/New_York",
        date(2026, 3, 2),
    )
    autumn_start, autumn_end = WeeklyReviewService._local_week_bounds_utc(
        "America/New_York",
        date(2026, 10, 26),
    )
    assert spring_end - spring_start == timedelta(hours=167)
    assert autumn_end - autumn_start == timedelta(hours=169)
    assert current_week_start(
        "America/New_York",
        now=datetime(2026, 3, 9, 3, 30, tzinfo=UTC),
    ) == date(2026, 3, 2)
    assert current_week_start(
        "America/New_York",
        now=datetime(2026, 3, 9, 4, 30, tzinfo=UTC),
    ) == date(2026, 3, 9)


def test_normalization_is_bounded_strict_and_privacy_safe():
    private = "PRIVATE_WEEKLY_FOCUS_SENTINEL"
    assert normalize_weekly_focus(f"  {private}\n next  ") == f"{private} next"
    assert normalize_weekly_steps([" один ", "два"]) == ("один", "два")
    candidates = normalize_weekly_candidates(
        [{"title": " Позвонить ", "schedule_wording": " в 15:05 "}]
    )
    assert candidates == (WeeklyReminderCandidate("Позвонить", "в 15:05"),)
    assert private not in repr(candidates)

    for invalid in ("", "x" * 301, f"{private}\x00"):
        with pytest.raises(WeeklyReviewValidationError) as error:
            normalize_weekly_focus(invalid)
        assert private not in str(error.value)
    with pytest.raises(WeeklyReviewValidationError):
        normalize_weekly_steps(["1", "2", "3", "4"])
    with pytest.raises(WeeklyReviewValidationError):
        normalize_weekly_steps(["x" * 201])
    with pytest.raises(WeeklyReviewValidationError):
        normalize_weekly_candidates(
            [
                WeeklyReminderCandidate("Позвонить", "в 15:05"),
                WeeklyReminderCandidate("ПОЗВОНИТЬ", "В 15:05"),
            ]
        )
    for forbidden_key in (
        "evidence",
        "evidence_quote",
        "raw",
        "private",
        "secret",
        "unknown",
    ):
        with pytest.raises(WeeklyReviewValidationError):
            normalize_weekly_candidates(
                [
                    {
                        "title": "Позвонить",
                        "schedule_wording": "в 15:05",
                        forbidden_key: "PRIVATE_CANDIDATE_SENTINEL",
                    }
                ]
            )


async def test_durable_session_stores_only_normalized_preview_and_exact_generation(db):
    telegram_id = 80_001
    await _add_user(db, telegram_id)
    service = WeeklyReviewService(db)
    private = "PRIVATE_WEEKLY_PREVIEW_SENTINEL"
    preview = await _preview_session(
        service,
        telegram_id,
        focus=f" {private}\n focus ",
        candidates=(WeeklyReminderCandidate("Сказать Назару", "в 15:05"),),
    )

    assert preview.phase is WeeklyReviewPhase.PREVIEW
    assert preview.focus == f"{private} focus"
    assert preview.version == 3
    assert preview.canonical_chat_id == telegram_id
    assert preview.canonical_message_id == 501
    assert preview.base_focus_public_id is None
    assert preview.base_focus_version is None
    assert str(UUID(preview.public_id)) == preview.public_id
    assert private not in repr(preview)
    assert preview.reminder_candidates[0].schedule_wording == "в 15:05"

    exact = await service.get_session_exact(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=501,
        expected_phase=WeeklyReviewPhase.PREVIEW,
        now=NOW,
    )
    wrong_canonical = await service.get_session_exact(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=999,
        now=NOW,
    )
    replayed_transition = await service.store_extraction(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=2,
        canonical_message_id=501,
        focus="Must not replace",
        approach=None,
        small_steps=(),
        reminder_candidates=(),
        source="text",
        now=NOW,
    )
    assert exact.status == "found"
    assert wrong_canonical.status == "stale"
    assert replayed_transition.status == "stale"

    columns = set(WeeklyReviewSession.__table__.columns.keys())
    assert {
        "raw_text",
        "raw_input",
        "transcript",
        "provider_output",
        "model_output",
        "evidence",
        "evidence_quote",
    }.isdisjoint(columns)


async def test_scheduled_session_freezes_next_week_across_exact_reads_and_confirm(db):
    telegram_id = 80_002
    await _add_user(db, telegram_id, timezone="UTC")
    service = WeeklyReviewService(db)
    scheduled_week = WEEK + timedelta(days=7)

    rejected_manual = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        target_week_start=scheduled_week,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
        scheduled=False,
        now=NOW,
    )
    assert rejected_manual.status == "week_changed"

    created = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        target_week_start=scheduled_week,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
        scheduled=True,
        now=NOW,
    )
    assert created.status == "created"
    assert created.session is not None
    assert created.session.week_start == scheduled_week

    current = await service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        now=NOW,
    )
    exact = await service.get_session_exact(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=501,
        expected_phase=WeeklyReviewPhase.AWAITING_INPUT,
        now=NOW,
    )
    snapshot = await service.system_snapshot(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        target_week_start=scheduled_week,
        now=NOW,
    )
    assert current.status == "found" and current.session == created.session
    assert exact.status == "found" and exact.session == created.session
    assert snapshot.status == "ready"
    assert snapshot.week_start == scheduled_week

    processing = await service.mark_processing(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=501,
        now=NOW,
    )
    assert processing.session is not None
    preview = await service.store_extraction(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=processing.session.public_id,
        expected_session_version=processing.session.version,
        canonical_message_id=501,
        focus="Scheduled next-week focus",
        approach=None,
        small_steps=(),
        reminder_candidates=(),
        source="text",
        now=NOW,
    )
    assert preview.session is not None
    confirmed = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.session.public_id,
        expected_session_version=preview.session.version,
        canonical_message_id=501,
        expected_week_start=scheduled_week,
        now=NOW,
    )
    assert confirmed.status == "created"
    assert confirmed.focus is not None
    assert confirmed.focus.week_start == scheduled_week
    assert confirmed.audit_written is True

    live_saved = await service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        now=NOW,
    )
    assert live_saved.status == "found"
    assert live_saved.session == confirmed.session
    assert confirmed.session is not None
    delete_preview = await service.transition_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=confirmed.session.public_id,
        expected_session_version=confirmed.session.version,
        expected_canonical_message_id=501,
        expected_phase=WeeklyReviewPhase.SAVED,
        phase=WeeklyReviewPhase.DELETE_PREVIEW,
        now=NOW,
    )
    assert delete_preview.session is not None
    deleted = await service.confirm_delete(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=delete_preview.session.public_id,
        expected_session_version=delete_preview.session.version,
        canonical_message_id=501,
        expected_week_start=scheduled_week,
        focus_public_id=confirmed.focus.public_id,
        expected_focus_version=confirmed.focus.version,
        now=NOW,
    )
    assert deleted.status == "deleted"
    assert deleted.audit_written is True


async def test_scheduled_capability_frozen_week_survives_local_monday_and_confirms(db):
    telegram_id = 80_020
    await _add_user(db, telegram_id, timezone="Europe/Moscow")
    service = WeeklyReviewService(db, review_weekday=2)
    sunday_2350 = datetime(2026, 8, 23, 20, 50, tzinfo=UTC)
    monday_0005 = datetime(2026, 8, 23, 21, 5, tzinfo=UTC)
    frozen_week = target_week_start(
        "Europe/Moscow",
        now=sunday_2350,
        scheduled=True,
        review_weekday=6,
    )
    assert frozen_week == date(2026, 8, 24)

    created = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        target_week_start=frozen_week,
        canonical_message_id=520,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
        scheduled=True,
        now=monday_0005,
    )
    assert created.status == "created"
    assert created.session is not None
    assert created.session.week_start == frozen_week

    processing = await service.mark_processing(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=520,
        now=monday_0005,
    )
    assert processing.session is not None
    preview = await service.store_extraction(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=processing.session.public_id,
        expected_session_version=processing.session.version,
        canonical_message_id=520,
        focus="Frozen scheduled focus",
        approach=None,
        small_steps=(),
        reminder_candidates=(),
        source="text",
        now=monday_0005,
    )
    assert preview.session is not None
    confirmed = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.session.public_id,
        expected_session_version=preview.session.version,
        canonical_message_id=520,
        expected_week_start=frozen_week,
        now=monday_0005,
    )
    assert confirmed.status == "created"
    assert confirmed.focus is not None
    assert confirmed.focus.week_start == frozen_week
    assert confirmed.audit_written is True


@pytest.mark.parametrize(
    ("scheduled", "requested_week"),
    [
        (False, date(2026, 8, 17)),
        (False, date(2026, 8, 31)),
        (True, date(2026, 8, 17)),
        (True, date(2026, 9, 7)),
    ],
)
async def test_session_creation_rejects_non_live_manual_and_scheduled_weeks(
    db,
    scheduled,
    requested_week,
):
    telegram_id = 80_021
    await _add_user(db, telegram_id, timezone="Europe/Moscow")
    service = WeeklyReviewService(db, review_weekday=0)
    monday_0005 = datetime(2026, 8, 23, 21, 5, tzinfo=UTC)

    result = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        target_week_start=requested_week,
        canonical_message_id=521,
        scheduled=scheduled,
        now=monday_0005,
    )

    assert result.status == "week_changed"
    assert result.session is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0


async def test_manual_session_still_fails_closed_at_review_week_boundary(db):
    telegram_id = 80_003
    await _add_user(db, telegram_id, timezone="UTC")
    service = WeeklyReviewService(db)
    sunday = datetime(2026, 8, 23, 23, 50, tzinfo=UTC)
    monday = datetime(2026, 8, 24, 0, 5, tzinfo=UTC)
    created = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        target_week_start=WEEK,
        canonical_message_id=502,
        scheduled=False,
        now=sunday,
    )
    assert created.status == "created"

    changed = await service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        now=monday,
    )
    assert changed.status == "week_changed"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0


async def test_manual_transition_rechecks_week_under_lock_after_boundary_race(db):
    telegram_id = 80_004
    await _add_user(db, telegram_id, timezone="UTC")
    service = WeeklyReviewService(db)
    sunday = datetime(2026, 8, 23, 23, 50, tzinfo=UTC)
    before_boundary = datetime(2026, 8, 23, 23, 59, tzinfo=UTC)
    monday = datetime(2026, 8, 24, 0, 5, tzinfo=UTC)
    created = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        target_week_start=WEEK,
        canonical_message_id=503,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
        scheduled=False,
        now=sunday,
    )
    assert created.session is not None
    observed = await service.get_session_exact(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=503,
        expected_phase=WeeklyReviewPhase.AWAITING_INPUT,
        now=before_boundary,
    )
    assert observed.status == "found"

    transition = await service.mark_processing(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=503,
        now=monday,
    )
    assert transition.status == "week_changed"
    assert transition.session is None
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0


async def test_authoritative_clock_is_sampled_after_create_and_transition_locks(db):
    telegram_id = 80_005
    owner = await _add_user(db, telegram_id, timezone="Europe/Moscow")
    peer = Database(db.url)
    local_sunday = datetime(2026, 8, 23, 20, 55, tzinfo=UTC)
    local_monday = datetime(2026, 8, 23, 21, 5, tzinfo=UTC)
    clock_value = [local_sunday]
    service = WeeklyReviewService(peer, clock=lambda: clock_value[0])
    try:
        create_after_boundary = await _after_locked_owner_write(
            db,
            peer,
            owner_id=owner.id,
            operation=lambda: service.create_session(
                telegram_actor_id=telegram_id,
                chat_id=81_201,
                expected_access_version=1,
                target_week_start=WEEK,
                canonical_message_id=501,
            ),
            clock_value=clock_value,
            after_lock=local_monday,
        )
        assert create_after_boundary.status == "week_changed"

        created = await service.create_session(
            telegram_actor_id=telegram_id,
            chat_id=81_202,
            expected_access_version=1,
            target_week_start=WEEK,
            canonical_message_id=501,
            phase=WeeklyReviewPhase.AWAITING_INPUT,
            now=local_sunday,
        )
        assert created.session is not None
        clock_value[0] = local_sunday
        transitioned_after_boundary = await _after_locked_owner_write(
            db,
            peer,
            owner_id=owner.id,
            operation=lambda: service.mark_processing(
                telegram_actor_id=telegram_id,
                chat_id=81_202,
                expected_access_version=1,
                session_public_id=created.session.public_id,
                expected_session_version=created.session.version,
                canonical_message_id=501,
            ),
            clock_value=clock_value,
            after_lock=local_monday,
        )
        assert transitioned_after_boundary.status == "week_changed"

        ttl_start = datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
        ttl_session = await service.create_session(
            telegram_actor_id=telegram_id,
            chat_id=81_203,
            expected_access_version=1,
            target_week_start=WEEK,
            canonical_message_id=501,
            phase=WeeklyReviewPhase.AWAITING_INPUT,
            now=ttl_start,
        )
        assert ttl_session.session is not None
        clock_value[0] = ttl_start + timedelta(minutes=29)
        transitioned_after_ttl = await _after_locked_owner_write(
            db,
            peer,
            owner_id=owner.id,
            operation=lambda: service.mark_processing(
                telegram_actor_id=telegram_id,
                chat_id=81_203,
                expected_access_version=1,
                session_public_id=ttl_session.session.public_id,
                expected_session_version=ttl_session.session.version,
                canonical_message_id=501,
            ),
            clock_value=clock_value,
            after_lock=ttl_start + timedelta(minutes=31),
        )
        assert transitioned_after_ttl.status == "expired"

        async with db.sessions() as session:
            assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0
            assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
            assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0
    finally:
        await peer.dispose()


async def test_authoritative_clock_fences_confirm_delete_and_recovery_after_locks(db):
    telegram_id = 80_006
    owner = await _add_user(db, telegram_id, timezone="Europe/Moscow")
    peer = Database(db.url)
    local_sunday = datetime(2026, 8, 23, 20, 55, tzinfo=UTC)
    local_monday = datetime(2026, 8, 23, 21, 5, tzinfo=UTC)
    clock_value = [local_sunday]
    service = WeeklyReviewService(peer, clock=lambda: clock_value[0])
    try:
        preview = await _preview_session(
            service,
            telegram_id,
            chat_id=81_211,
            focus="Не должен сохраниться после границы",
            now=local_sunday,
        )
        confirm_after_boundary = await _after_locked_owner_write(
            db,
            peer,
            owner_id=owner.id,
            operation=lambda: service.confirm_focus(
                telegram_actor_id=telegram_id,
                chat_id=81_211,
                expected_access_version=1,
                session_public_id=preview.public_id,
                expected_session_version=preview.version,
                canonical_message_id=501,
                expected_week_start=WEEK,
            ),
            clock_value=clock_value,
            after_lock=local_monday,
        )
        assert confirm_after_boundary.status == "week_changed"

        confirmed_preview = await _preview_session(
            service,
            telegram_id,
            chat_id=81_212,
            focus="Существующий фокус",
            now=local_sunday,
        )
        confirmed = await service.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=81_212,
            expected_access_version=1,
            session_public_id=confirmed_preview.public_id,
            expected_session_version=confirmed_preview.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=local_sunday,
        )
        assert confirmed.focus is not None
        delete_root = await service.create_session(
            telegram_actor_id=telegram_id,
            chat_id=81_213,
            expected_access_version=1,
            target_week_start=WEEK,
            canonical_message_id=501,
            now=local_sunday,
        )
        assert delete_root.session is not None
        delete_preview = await service.transition_session(
            telegram_actor_id=telegram_id,
            chat_id=81_213,
            expected_access_version=1,
            session_public_id=delete_root.session.public_id,
            expected_session_version=delete_root.session.version,
            expected_canonical_message_id=501,
            phase=WeeklyReviewPhase.DELETE_PREVIEW,
            now=local_sunday,
        )
        assert delete_preview.session is not None
        clock_value[0] = local_sunday
        delete_after_boundary = await _after_locked_owner_write(
            db,
            peer,
            owner_id=owner.id,
            operation=lambda: service.confirm_delete(
                telegram_actor_id=telegram_id,
                chat_id=81_213,
                expected_access_version=1,
                session_public_id=delete_preview.session.public_id,
                expected_session_version=delete_preview.session.version,
                canonical_message_id=501,
                expected_week_start=WEEK,
            ),
            clock_value=clock_value,
            after_lock=local_monday,
        )
        assert delete_after_boundary.status == "week_changed"

        processing_root = await service.create_session(
            telegram_actor_id=telegram_id,
            chat_id=81_214,
            expected_access_version=1,
            target_week_start=WEEK,
            canonical_message_id=501,
            phase=WeeklyReviewPhase.AWAITING_INPUT,
            now=local_sunday,
        )
        assert processing_root.session is not None
        processing = await service.mark_processing(
            telegram_actor_id=telegram_id,
            chat_id=81_214,
            expected_access_version=1,
            session_public_id=processing_root.session.public_id,
            expected_session_version=processing_root.session.version,
            canonical_message_id=501,
            now=local_sunday,
        )
        assert processing.session is not None
        clock_value[0] = local_sunday
        recovered_after_boundary = await _after_locked_owner_write(
            db,
            peer,
            owner_id=owner.id,
            operation=lambda: service.recover_processing_session_snapshots(),
            clock_value=clock_value,
            after_lock=local_monday,
        )
        assert recovered_after_boundary == ()

        async with db.sessions() as session:
            focus = await session.scalar(
                select(WeeklyFocus).where(
                    WeeklyFocus.owner_id == owner.id,
                    WeeklyFocus.week_start == WEEK,
                )
            )
            assert focus is not None and focus.version == 1
            assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1
            assert (
                await session.scalar(
                    select(func.count(WeeklyReviewSession.id)).where(
                        WeeklyReviewSession.chat_id == 81_214
                    )
                )
                == 0
            )
    finally:
        await peer.dispose()


async def test_confirm_is_exactly_once_version_fenced_and_audit_is_metadata_only(db):
    telegram_id = 80_010
    owner = await _add_user(db, telegram_id)
    service = WeeklyReviewService(db)
    private = "PRIVATE_CONFIRMED_WEEKLY_SENTINEL"
    preview = await _preview_session(service, telegram_id, focus=private)

    first, second = await asyncio.gather(
        service.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=1,
            session_public_id=preview.public_id,
            expected_session_version=preview.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=NOW,
        ),
        service.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=1,
            session_public_id=preview.public_id,
            expected_session_version=preview.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=NOW,
        ),
    )
    created = first if first.status == "created" else second
    replay = second if first.status == "created" else first

    assert created.status == "created"
    assert created.audit_written is True
    assert created.focus is not None and created.focus.version == 1
    assert created.session is not None and created.session.phase is WeeklyReviewPhase.SAVED
    assert replay.status == "replay"
    assert replay.audit_written is False
    assert private not in repr(created)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 1
        changes = list(
            (
                await session.scalars(
                    select(WeeklyFocusChange).where(WeeklyFocusChange.owner_id == owner.id)
                )
            ).all()
        )
    assert [(change.operation, change.resulting_version) for change in changes] == [("created", 1)]
    assert {
        "focus",
        "approach",
        "small_steps",
        "raw_text",
        "provider_output",
        "evidence",
    }.isdisjoint(WeeklyFocusChange.__table__.columns.keys())

    duplicate_preview = await _preview_session(service, telegram_id, focus=private)
    duplicate = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=duplicate_preview.public_id,
        expected_session_version=duplicate_preview.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        now=NOW,
    )
    assert duplicate.status == "duplicate"
    assert duplicate.audit_written is False
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 1

    update_preview = await _preview_session(service, telegram_id, focus="Новый ориентир")
    updated = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=update_preview.public_id,
        expected_session_version=update_preview.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        now=NOW,
    )
    assert updated.status == "updated"
    assert updated.focus is not None and updated.focus.version == 2
    async with db.sessions() as session:
        changes = list(
            (await session.scalars(select(WeeklyFocusChange).order_by(WeeklyFocusChange.id))).all()
        )
    assert [(row.operation, row.resulting_version) for row in changes] == [
        ("created", 1),
        ("updated", 2),
    ]


async def test_focus_generation_cas_across_two_databases_and_chats(db):
    telegram_id = 80_011
    owner = await _add_user(db, telegram_id, timezone="UTC")
    peer = Database(db.url)
    primary = WeeklyReviewService(db)
    secondary = WeeklyReviewService(peer)
    try:
        absent_a = await _preview_session(
            primary,
            telegram_id,
            chat_id=81_101,
            focus="Первый фокус",
        )
        absent_b = await _preview_session(
            secondary,
            telegram_id,
            chat_id=81_102,
            focus="Конкурирующий фокус",
        )
        assert absent_a.base_focus_public_id is None
        assert absent_b.base_focus_public_id is None

        created = await primary.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=81_101,
            expected_access_version=1,
            session_public_id=absent_a.public_id,
            expected_session_version=absent_a.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=NOW,
        )
        stale_absence = await secondary.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=81_102,
            expected_access_version=1,
            session_public_id=absent_b.public_id,
            expected_session_version=absent_b.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=NOW,
        )
        assert created.status == "created"
        assert created.focus is not None
        assert stale_absence.status == "focus_changed"
        assert stale_absence.session == absent_b
        assert stale_absence.audit_written is False

        update_a = await _preview_session(
            primary,
            telegram_id,
            chat_id=81_103,
            focus="Обновление A",
        )
        update_b = await _preview_session(
            secondary,
            telegram_id,
            chat_id=81_104,
            focus="Обновление B",
        )
        expected_base = (created.focus.public_id, created.focus.version)
        assert (update_a.base_focus_public_id, update_a.base_focus_version) == expected_base
        assert (update_b.base_focus_public_id, update_b.base_focus_version) == expected_base

        updated = await primary.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=81_103,
            expected_access_version=1,
            session_public_id=update_a.public_id,
            expected_session_version=update_a.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=NOW,
        )
        stale_version = await secondary.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=81_104,
            expected_access_version=1,
            session_public_id=update_b.public_id,
            expected_session_version=update_b.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=NOW,
        )
        assert updated.status == "updated"
        assert updated.focus is not None and updated.focus.version == 2
        assert stale_version.status == "focus_changed"
        assert stale_version.session == update_b

        delete_root = await secondary.create_session(
            telegram_actor_id=telegram_id,
            chat_id=81_105,
            expected_access_version=1,
            target_week_start=WEEK,
            canonical_message_id=501,
            now=NOW,
        )
        assert delete_root.session is not None
        delete_preview = await secondary.transition_session(
            telegram_actor_id=telegram_id,
            chat_id=81_105,
            expected_access_version=1,
            session_public_id=delete_root.session.public_id,
            expected_session_version=delete_root.session.version,
            expected_canonical_message_id=501,
            phase=WeeklyReviewPhase.DELETE_PREVIEW,
            now=NOW,
        )
        assert delete_preview.session is not None
        assert (
            delete_preview.session.base_focus_public_id,
            delete_preview.session.base_focus_version,
        ) == (updated.focus.public_id, updated.focus.version)

        competing = await _preview_session(
            primary,
            telegram_id,
            chat_id=81_106,
            focus="Обновление после delete preview",
        )
        replacement = await primary.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=81_106,
            expected_access_version=1,
            session_public_id=competing.public_id,
            expected_session_version=competing.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            now=NOW,
        )
        assert replacement.focus is not None and replacement.focus.version == 3
        stale_delete = await secondary.confirm_delete(
            telegram_actor_id=telegram_id,
            chat_id=81_105,
            expected_access_version=1,
            session_public_id=delete_preview.session.public_id,
            expected_session_version=delete_preview.session.version,
            canonical_message_id=501,
            expected_week_start=WEEK,
            focus_public_id=replacement.focus.public_id,
            expected_focus_version=replacement.focus.version,
            now=NOW,
        )
        assert stale_delete.status == "focus_changed"
        assert stale_delete.session == delete_preview.session
        assert stale_delete.audit_written is False

        async with db.sessions() as session:
            focus = await session.scalar(
                select(WeeklyFocus).where(
                    WeeklyFocus.owner_id == owner.id,
                    WeeklyFocus.week_start == WEEK,
                )
            )
            changes = tuple(
                (
                    await session.scalars(
                        select(WeeklyFocusChange)
                        .where(WeeklyFocusChange.owner_id == owner.id)
                        .order_by(WeeklyFocusChange.id)
                    )
                ).all()
            )
        assert focus is not None and focus.version == 3
        assert [change.operation for change in changes] == ["created", "updated", "updated"]
    finally:
        await peer.dispose()


async def test_access_bounce_wrong_actor_and_week_change_fail_closed(db):
    telegram_id = 80_020
    other_id = 80_021
    await _add_user(db, telegram_id)
    await _add_user(db, other_id, tier="admin")
    service = WeeklyReviewService(db)
    preview = await _preview_session(service, telegram_id)

    forged = await service.confirm_focus(
        telegram_actor_id=other_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        now=NOW,
    )
    assert forged.status == "not_found"

    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_version = 3
    bounced = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        now=NOW,
    )
    assert bounced.status == "access_changed"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0

    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_version = 1
    changed_week = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=501,
        expected_week_start=WEEK + timedelta(days=7),
        now=NOW,
    )
    assert changed_week.status == "week_changed"


async def test_replacement_generation_is_not_cleared_by_exact_old_cleanup(db):
    telegram_id = 80_030
    await _add_user(db, telegram_id)
    service = WeeklyReviewService(db)
    old = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.ROOT,
        now=NOW,
    )
    replacement = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
        now=NOW,
    )
    assert old.session is not None and replacement.session is not None

    assert not await service.clear_session_exact(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        session_public_id=old.session.public_id,
        expected_session_version=old.session.version,
    )
    current = await service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        now=NOW,
    )
    assert current.session == replacement.session
    assert await service.clear_session_exact(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        session_public_id=replacement.session.public_id,
        expected_session_version=replacement.session.version,
    )


async def test_access_retirement_returns_old_binding_and_preserves_fresh_replacement(db):
    telegram_id = 80_031
    await _add_user(db, telegram_id)
    service = WeeklyReviewService(db)
    old = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.ROOT,
        now=NOW,
    )
    assert old.session is not None
    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "guest"
        user.access_version = 2
    retired = await service.retire_access_changed_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        current_access_version=2,
    )
    assert retired.status == "access_changed"
    assert retired.session == old.session
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyReviewSession.id))) == 0

    async with db.session() as session:
        user = await session.scalar(select(User).where(User.telegram_id == telegram_id))
        assert user is not None
        user.access_tier = "subscriber"
        user.access_version = 3
    replacement = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=3,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.ROOT,
        now=NOW,
    )
    assert replacement.session is not None
    stale_observer = await service.retire_access_changed_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        current_access_version=2,
    )
    assert stale_observer.status == "access_changed"
    assert stale_observer.session is None
    current = await service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=3,
        now=NOW,
    )
    assert current.session == replacement.session


async def test_delete_requires_separate_preview_and_is_exactly_once(db):
    telegram_id = 80_040
    await _add_user(db, telegram_id)
    service = WeeklyReviewService(db)
    preview = await _preview_session(service, telegram_id)
    created = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        now=NOW,
    )
    assert created.focus is not None and created.session is not None

    premature = await service.confirm_delete(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        focus_public_id=created.focus.public_id,
        expected_focus_version=created.focus.version,
        now=NOW,
    )
    assert premature.status == "stale"
    delete_preview = await service.transition_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        expected_canonical_message_id=501,
        expected_phase=WeeklyReviewPhase.SAVED,
        phase=WeeklyReviewPhase.DELETE_PREVIEW,
        now=NOW,
    )
    assert delete_preview.session is not None
    deleted = await service.confirm_delete(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=delete_preview.session.public_id,
        expected_session_version=delete_preview.session.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        focus_public_id=created.focus.public_id,
        expected_focus_version=created.focus.version,
        now=NOW,
    )
    replay = await service.confirm_delete(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=delete_preview.session.public_id,
        expected_session_version=delete_preview.session.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        focus_public_id=created.focus.public_id,
        expected_focus_version=created.focus.version,
        now=NOW,
    )
    assert deleted.status == "deleted" and deleted.audit_written is True
    assert replay.status == "replay"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
        changes = list(
            (await session.scalars(select(WeeklyFocusChange).order_by(WeeklyFocusChange.id))).all()
        )
    assert [(row.operation, row.resulting_version) for row in changes] == [
        ("created", 1),
        ("deleted", 2),
    ]


async def test_processing_recovery_and_expiry_cleanup_are_bounded_and_idempotent(db):
    telegram_id = 80_050
    await _add_user(db, telegram_id)
    service = WeeklyReviewService(db)
    created = await service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        canonical_message_id=501,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
        now=NOW,
    )
    assert created.session is not None
    processing = await service.mark_processing(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=501,
        now=NOW,
    )
    assert processing.session is not None

    assert (
        await service.recover_processing_sessions(
            now=NOW + timedelta(seconds=1),
            updated_before=NOW - timedelta(seconds=34),
        )
        == 0
    )
    live = await service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        now=NOW + timedelta(seconds=1),
    )
    assert live.session == processing.session
    assert live.session.phase is WeeklyReviewPhase.PROCESSING

    assert await service.recover_processing_sessions(now=NOW + timedelta(minutes=1)) == 1
    assert await service.recover_processing_sessions(now=NOW + timedelta(minutes=1)) == 0
    recovered = await service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        now=NOW + timedelta(minutes=1),
    )
    assert recovered.session is not None
    assert recovered.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert recovered.session.version == processing.session.version + 1
    assert recovered.session.focus is None
    assert recovered.session.reminder_candidates == ()

    assert await service.cleanup_expired(now=NOW + timedelta(minutes=32), limit=1) == 1
    assert await service.cleanup_expired(now=NOW + timedelta(minutes=32), limit=1) == 0
    assert (
        await service.current_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=1,
            now=NOW + timedelta(minutes=32),
        )
    ).status == "not_found"


async def test_recovery_and_cleanup_respect_effective_allowed_tiers(db):
    subscriber_id = 80_051
    admin_id = 80_052
    await _add_user(db, subscriber_id, tier="subscriber")
    await _add_user(db, admin_id, tier="admin")
    service = WeeklyReviewService(db)
    processing_by_actor = {}
    for telegram_id in (subscriber_id, admin_id):
        created = await service.create_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=1,
            canonical_message_id=501,
            phase=WeeklyReviewPhase.AWAITING_INPUT,
            now=NOW,
        )
        assert created.session is not None
        processing = await service.mark_processing(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=1,
            session_public_id=created.session.public_id,
            expected_session_version=created.session.version,
            canonical_message_id=501,
            now=NOW,
        )
        assert processing.session is not None
        processing_by_actor[telegram_id] = processing.session

    recovered = await service.recover_processing_session_snapshots(
        now=NOW + timedelta(minutes=1),
        allowed_tiers=frozenset({"admin"}),
    )
    assert [session.telegram_user_id for session in recovered] == [admin_id]
    subscriber = await service.current_session(
        telegram_actor_id=subscriber_id,
        chat_id=subscriber_id,
        expected_access_version=1,
        now=NOW + timedelta(minutes=1),
    )
    assert subscriber.session == processing_by_actor[subscriber_id]
    assert subscriber.session.phase is WeeklyReviewPhase.PROCESSING

    assert (
        await service.cleanup_expired(
            now=NOW + timedelta(minutes=32),
            allowed_tiers=frozenset({"admin"}),
        )
        == 1
    )
    async with db.sessions() as session:
        rows = tuple(
            (
                await session.scalars(select(WeeklyReviewSession).order_by(WeeklyReviewSession.id))
            ).all()
        )
    assert [row.telegram_user_id for row in rows] == [subscriber_id]
    assert (
        await service.cleanup_expired(
            now=NOW + timedelta(minutes=32),
            allowed_tiers=frozenset({"subscriber"}),
        )
        == 1
    )
    with pytest.raises(WeeklyReviewValidationError):
        await service.recover_processing_sessions(allowed_tiers=frozenset())


async def test_snapshot_is_bounded_owner_scoped_and_contains_no_raw_text(db):
    telegram_id = 80_060
    owner = await _add_user(db, telegram_id, timezone="UTC")
    service = WeeklyReviewService(db, task_snapshot_limit=2, reminder_snapshot_limit=2)
    preview = await _preview_session(service, telegram_id, now=NOW)
    confirmed = await service.confirm_focus(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=1,
        session_public_id=preview.public_id,
        expected_session_version=preview.version,
        canonical_message_id=501,
        expected_week_start=WEEK,
        now=NOW,
    )
    assert confirmed.status == "created"

    async with db.session() as session:
        completed = InboxItem(
            user_id=owner.id,
            kind="task",
            title="Выполнено",
            raw_text="PRIVATE_RAW_COMPLETED",
            source="text",
            status="confirmed",
        )
        one_shot_item = InboxItem(
            user_id=owner.id,
            kind="task",
            title="Срочное дело",
            raw_text="PRIVATE_RAW_ACTIVE",
            source="text",
            status="confirmed",
        )
        daily_item = InboxItem(
            user_id=owner.id,
            kind="task",
            title="Ежедневное дело",
            raw_text="PRIVATE_RAW_DAILY",
            source="text",
            status="confirmed",
        )
        session.add_all([completed, one_shot_item, daily_item])
        await session.flush()
        completed_state = TaskState(
            owner_id=owner.id,
            inbox_item_id=completed.id,
            status="completed",
            timezone="UTC",
            completed_at=NOW - timedelta(days=3),
        )
        one_shot_state = TaskState(
            owner_id=owner.id,
            inbox_item_id=one_shot_item.id,
            status="active",
            timezone="UTC",
            event_at=NOW + timedelta(days=1),
        )
        daily_state = TaskState(
            owner_id=owner.id,
            inbox_item_id=daily_item.id,
            status="active",
            timezone="UTC",
        )
        session.add_all([completed_state, one_shot_state, daily_state])
        await session.flush()
        session.add(
            TaskReminder(
                inbox_item_id=one_shot_item.id,
                telegram_user_id=telegram_id,
                chat_id=telegram_id,
                event_at=NOW + timedelta(days=1),
                remind_at=NOW + timedelta(hours=2),
                timezone="UTC",
                delivery_key="weekly-snapshot-one-shot",
                task_version=one_shot_state.version,
                status="pending",
            )
        )
        session.add(
            RecurringTaskReminderSchedule(
                owner_id=owner.id,
                inbox_item_id=daily_item.id,
                recurrence_kind="daily",
                local_time=time(15, 5),
                timezone="UTC",
                timezone_source="profile",
                start_local_date=WEEK,
                next_occurrence_at=NOW + timedelta(hours=5),
                status="active",
                version=1,
            )
        )

    snapshot = await service.system_snapshot(
        telegram_actor_id=telegram_id,
        expected_access_version=1,
        target_week_start=WEEK,
        now=NOW,
    )
    assert snapshot.status == "ready"
    assert snapshot.current_focus is not None
    assert snapshot.completed_previous_cycle == 1
    assert len(snapshot.active_tasks) == 2
    assert any(task.requires_attention for task in snapshot.active_tasks)
    assert {reminder.kind for reminder in snapshot.reminders} == {"one_shot", "daily"}
    assert "PRIVATE_RAW" not in repr(snapshot)


async def test_confirmed_history_is_preserved_but_current_lookup_never_uses_expired_week(db):
    telegram_id = 80_070
    owner = await _add_user(db, telegram_id, timezone="UTC")
    async with db.session() as session:
        session.add(
            WeeklyFocus(
                public_id="00000000-0000-4000-8000-000000000070",
                owner_id=owner.id,
                week_start=WEEK - timedelta(days=7),
                focus="Прошлый фокус",
                approach=None,
                small_steps=[],
                source="text",
                version=1,
            )
        )
    service = WeeklyReviewService(db)
    current = await service.get_focus(telegram_actor_id=telegram_id, now=NOW)
    history = await service.get_focus(
        telegram_actor_id=telegram_id,
        week_start=WEEK - timedelta(days=7),
        now=NOW,
    )
    assert current.status == "not_found"
    assert history.status == "found"
    assert history.focus is not None and history.focus.focus == "Прошлый фокус"
