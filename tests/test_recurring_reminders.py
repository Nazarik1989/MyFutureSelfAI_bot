from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta

import pytest
from sqlalchemy import delete, func, select, update

from future_self.access import BLOCKED, GUEST, SUBSCRIBER, AccessService
from future_self.models import (
    InboxItem,
    RecurringTaskReminderOccurrence,
    RecurringTaskReminderSchedule,
    TaskState,
    User,
)
from future_self.recurring_reminders import (
    RecurringFenceLost,
    RecurringScheduleConflict,
    RecurringTaskNotEligible,
    RecurringTaskReminderEngine,
    RecurringTaskReminderService,
    as_utc,
    calculate_next_daily_occurrence,
    format_recurring_time_for_profile,
)


async def create_task(
    db,
    *,
    telegram_id: int = 61_001,
    timezone: str = "Europe/Moscow",
    tier: str = SUBSCRIBER,
    kind: str = "task",
    item_status: str = "confirmed",
    task_status: str = "active",
    title: str = "Заполнить дневник благодарностей",
) -> tuple[User, InboxItem, TaskState]:
    async with db.session() as session:
        owner = User(
            telegram_id=telegram_id,
            timezone=timezone,
            access_tier=tier,
            onboarding_completed=True,
        )
        session.add(owner)
        await session.flush()
        item = InboxItem(
            user_id=owner.id,
            kind=kind,
            title=title,
            raw_text="private recurring command sentinel",
            source="text",
            status=item_status,
            trashed_at=(datetime(2026, 8, 1, tzinfo=UTC) if item_status == "trashed" else None),
            pre_trash_status="confirmed" if item_status == "trashed" else None,
        )
        session.add(item)
        await session.flush()
        state = TaskState(
            owner_id=owner.id,
            inbox_item_id=item.id,
            status=task_status,
            timezone=timezone,
        )
        session.add(state)
        await session.flush()
        return owner, item, state


async def create_due_schedule(
    db,
    *,
    telegram_id: int = 61_001,
    tier: str = SUBSCRIBER,
    due: datetime = datetime(2026, 8, 10, 11, tzinfo=UTC),
) -> tuple[RecurringTaskReminderService, User, InboxItem, datetime]:
    owner, item, _state = await create_task(db, telegram_id=telegram_id, tier=tier, timezone="UTC")
    service = RecurringTaskReminderService(db)
    created = await service.create_daily(
        owner.id,
        item.id,
        due.time().replace(tzinfo=None),
        now=due - timedelta(hours=1),
    )
    assert created.schedule is not None
    assert created.schedule.next_occurrence_at == due
    return service, owner, item, due


async def occurrence_for(db, schedule_id: int) -> RecurringTaskReminderOccurrence:
    async with db.sessions() as session:
        occurrence = await session.scalar(
            select(RecurringTaskReminderOccurrence).where(
                RecurringTaskReminderOccurrence.schedule_id == schedule_id
            )
        )
    assert occurrence is not None
    return occurrence


async def occurrences_for(db, schedule_id: int) -> list[RecurringTaskReminderOccurrence]:
    async with db.sessions() as session:
        return list(
            (
                await session.scalars(
                    select(RecurringTaskReminderOccurrence)
                    .where(RecurringTaskReminderOccurrence.schedule_id == schedule_id)
                    .order_by(RecurringTaskReminderOccurrence.id)
                )
            ).all()
        )


def test_daily_next_occurrence_before_after_and_utc_midnight():
    before = calculate_next_daily_occurrence(
        time(19, 30),
        "Europe/Moscow",
        now=datetime(2026, 8, 10, 16, 29, tzinfo=UTC),
    )
    assert before.scheduled_for == datetime(2026, 8, 10, 16, 30, tzinfo=UTC)
    assert before.local_date == date(2026, 8, 10)

    exact = calculate_next_daily_occurrence(
        time(19, 30),
        "Europe/Moscow",
        now=datetime(2026, 8, 10, 16, 30, tzinfo=UTC),
    )
    assert exact.scheduled_for == datetime(2026, 8, 11, 16, 30, tzinfo=UTC)

    midnight = calculate_next_daily_occurrence(
        time(0, 30),
        "Europe/Saratov",
        now=datetime(2026, 8, 9, 19, tzinfo=UTC),
    )
    assert midnight.local_date == date(2026, 8, 10)
    assert midnight.scheduled_for == datetime(2026, 8, 9, 20, 30, tzinfo=UTC)


def test_daily_dst_policy_uses_fold_zero_and_skips_nonexistent_day():
    ambiguous = calculate_next_daily_occurrence(
        time(1, 30),
        "America/New_York",
        now=datetime(2026, 11, 1, 4, tzinfo=UTC),
    )
    assert ambiguous.local_date == date(2026, 11, 1)
    assert ambiguous.scheduled_for == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    assert ambiguous.fold == 0

    nonexistent = calculate_next_daily_occurrence(
        time(2, 30),
        "America/New_York",
        now=datetime(2026, 3, 8, 5, tzinfo=UTC),
    )
    assert nonexistent.local_date == date(2026, 3, 9)
    assert nonexistent.scheduled_for == datetime(2026, 3, 9, 6, 30, tzinfo=UTC)


def test_timezone_presentation_uses_occurrence_date_conversion():
    assert (
        format_recurring_time_for_profile(
            time(19, 30),
            "Europe/Moscow",
            "Europe/Saratov",
            occurrence_date=date(2026, 8, 10),
        )
        == "19:30 по Москве — 20:30 по Саратову"
    )
    assert (
        format_recurring_time_for_profile(
            time(19, 30),
            "Europe/Moscow",
            "Europe/Moscow",
            occurrence_date=date(2026, 8, 10),
        )
        == "19:30 (Europe/Moscow)"
    )


@pytest.mark.parametrize("value", [True, 4, 361])
def test_recurring_grace_validation_matches_settings_bounds(db, value):
    with pytest.raises(ValueError, match="between 5 and 360"):
        RecurringTaskReminderService(db, grace_minutes=value)

    RecurringTaskReminderService(db, grace_minutes=5)
    RecurringTaskReminderService(db, grace_minutes=360)


async def test_create_daily_is_owner_fenced_idempotent_and_lists_active(db):
    owner, item, _state = await create_task(db)
    service = RecurringTaskReminderService(db)
    now = datetime(2026, 8, 10, 15, tzinfo=UTC)

    first = await service.create_daily(owner.id, item.id, time(19, 30), now=now)
    duplicate = await service.create_daily(owner.id, item.id, time(19, 30), now=now)

    assert first.changed is True
    assert duplicate.changed is False
    assert duplicate.schedule == first.schedule
    assert first.schedule is not None
    assert first.schedule.timezone == "Europe/Moscow"
    assert first.schedule.timezone_source == "profile"
    assert await service.status(owner.id, item.id) == first.schedule
    assert await service.get(owner.id + 1, item.id) is None
    assert await service.list_active(owner.id) == (first.schedule,)


async def test_create_daily_in_session_uses_caller_lock_and_is_idempotent(db):
    owner, item, _state = await create_task(db)
    service = RecurringTaskReminderService(db)
    now = datetime(2026, 8, 10, 15, tzinfo=UTC)

    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == owner.id).values(updated_at=User.updated_at)
        )
        first = await service.create_daily_in_session(
            session,
            owner.id,
            item.id,
            time(19, 30),
            now=now,
        )
        repeated = await service.create_daily_in_session(
            session,
            owner.id,
            item.id,
            time(19, 30),
            now=now,
        )

    assert first.changed is True and first.schedule is not None
    assert repeated.changed is False
    assert repeated.schedule == first.schedule
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 1


async def test_create_daily_in_session_never_commits_caller_transaction(db):
    owner, item, _state = await create_task(db)
    service = RecurringTaskReminderService(db)

    with pytest.raises(RuntimeError, match="rollback sentinel"):
        async with db.session() as session:
            await session.execute(
                update(User).where(User.id == owner.id).values(updated_at=User.updated_at)
            )
            await service.create_daily_in_session(
                session,
                owner.id,
                item.id,
                time(19, 30),
                now=datetime(2026, 8, 10, 15, tzinfo=UTC),
            )
            raise RuntimeError("rollback sentinel")

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) == 0


async def test_explicit_timezone_wins_and_profile_timezone_is_authoritative(db):
    owner, item, _state = await create_task(db, timezone="Europe/Saratov")
    service = RecurringTaskReminderService(db)
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)

    explicit = await service.create_daily(
        owner.id,
        item.id,
        time(19, 30),
        timezone="Europe/Moscow",
        timezone_source="explicit",
        now=now,
    )
    assert explicit.schedule is not None
    assert (explicit.schedule.timezone, explicit.schedule.timezone_source) == (
        "Europe/Moscow",
        "explicit",
    )

    other, other_item, _state = await create_task(
        db,
        telegram_id=61_002,
        timezone="Europe/Saratov",
    )
    profile = await service.create_daily(
        other.id,
        other_item.id,
        time(19, 30),
        timezone="Europe/Moscow",
        timezone_source="profile",
        now=now,
    )
    assert profile.schedule is not None
    assert profile.schedule.timezone == "Europe/Saratov"


async def test_profile_timezone_refresh_moves_only_active_profile_generation(db):
    owner, profile_item, _state = await create_task(db, timezone="Europe/Moscow")
    async with db.session() as session:
        explicit_item = InboxItem(
            user_id=owner.id,
            kind="task",
            title="Explicit timezone task",
            raw_text="explicit timezone task",
            source="text",
            status="confirmed",
        )
        session.add(explicit_item)
        await session.flush()
        session.add(
            TaskState(
                owner_id=owner.id,
                inbox_item_id=explicit_item.id,
                status="active",
                timezone="Europe/Moscow",
            )
        )
        await session.flush()

    service = RecurringTaskReminderService(db)
    before_due = datetime(2026, 8, 10, 15, tzinfo=UTC)
    due = datetime(2026, 8, 10, 16, 30, tzinfo=UTC)
    profile = await service.create_daily(
        owner.id,
        profile_item.id,
        time(19, 30),
        now=before_due,
    )
    explicit = await service.create_daily(
        owner.id,
        explicit_item.id,
        time(19, 30),
        timezone="Europe/Moscow",
        timezone_source="explicit",
        now=before_due,
    )
    assert profile.schedule is not None and explicit.schedule is not None
    assert await service.materialize_due(now=due) == 2

    changed = await service.refresh_profile_timezone(
        owner.id,
        "Europe/Saratov",
        now=due + timedelta(minutes=1),
    )

    assert len(changed) == 1
    assert changed[0].id == profile.schedule.id
    assert (changed[0].timezone, changed[0].version) == ("Europe/Saratov", 2)
    assert changed[0].next_occurrence_at == datetime(2026, 8, 11, 15, 30, tzinfo=UTC)
    stored_profile = await service.get(owner.id, profile_item.id)
    stored_explicit = await service.get(owner.id, explicit_item.id)
    assert stored_profile == changed[0]
    assert stored_explicit == explicit.schedule
    assert (await occurrence_for(db, profile.schedule.id)).status == "cancelled"
    assert (await occurrence_for(db, explicit.schedule.id)).status == "pending"
    async with db.sessions() as session:
        stored_owner = await session.get(User, owner.id)
    assert stored_owner is not None and stored_owner.timezone == "Europe/Saratov"


async def test_in_session_timezone_refresh_defers_disabled_schedule_until_reenable(db):
    owner, item, _state = await create_task(db, timezone="Europe/Moscow")
    service = RecurringTaskReminderService(db)
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    created = await service.create_daily(owner.id, item.id, time(19, 30), now=now)
    assert created.schedule is not None
    disabled = await service.disable(
        owner.id,
        item.id,
        expected_version=created.schedule.version,
    )
    assert disabled.schedule is not None

    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == owner.id).values(updated_at=User.updated_at)
        )
        changed = await service.refresh_profile_timezone_in_session(
            session,
            owner.id,
            "Europe/Saratov",
            now=now,
        )

    assert changed == ()
    still_disabled = await service.get(owner.id, item.id)
    assert still_disabled == disabled.schedule
    reenabled = await service.reenable(
        owner.id,
        item.id,
        expected_version=disabled.schedule.version,
        now=now,
    )
    assert reenabled.schedule is not None
    assert (reenabled.schedule.timezone, reenabled.schedule.version) == ("Europe/Saratov", 3)


@pytest.mark.parametrize(
    ("kind", "item_status", "task_status"),
    [
        ("note", "confirmed", "active"),
        ("task", "archived", "active"),
        ("task", "trashed", "active"),
        ("task", "confirmed", "completed"),
        ("task", "confirmed", "cancelled"),
    ],
)
async def test_ineligible_items_cannot_get_active_schedule(
    db,
    kind,
    item_status,
    task_status,
):
    owner, item, _state = await create_task(
        db,
        kind=kind,
        item_status=item_status,
        task_status=task_status,
    )
    with pytest.raises(RecurringTaskNotEligible):
        await RecurringTaskReminderService(db).create_daily(owner.id, item.id, time(19, 30))


async def test_wrong_owner_cannot_create_or_mutate_schedule(db):
    owner, item, _state = await create_task(db)
    intruder, _other, _other_state = await create_task(db, telegram_id=61_002)
    service = RecurringTaskReminderService(db)
    await service.create_daily(owner.id, item.id, time(19, 30))

    with pytest.raises(RecurringTaskNotEligible):
        await service.create_daily(intruder.id, item.id, time(19, 30))
    with pytest.raises(RecurringScheduleConflict):
        await service.disable(intruder.id, item.id)
    assert (await service.get(owner.id, item.id)).status == "active"  # type: ignore[union-attr]


async def test_concurrent_duplicate_create_produces_one_schedule(db):
    owner, item, _state = await create_task(db)
    service = RecurringTaskReminderService(db)
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)

    results = await asyncio.gather(
        service.create_daily(owner.id, item.id, time(19, 30), now=now),
        service.create_daily(owner.id, item.id, time(19, 30), now=now),
    )

    assert sorted(result.changed for result in results) == [False, True]
    async with db.sessions() as session:
        count = await session.scalar(select(func.count(RecurringTaskReminderSchedule.id)))
    assert count == 1


async def test_update_noop_and_version_fences_pending_occurrence(db):
    service, owner, item, due = await create_due_schedule(db)
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    await service.materialize_due(now=due)

    no_op = await service.update_time(owner.id, item.id, time(11), now=due)
    assert no_op.changed is False
    assert no_op.schedule.version == 1  # type: ignore[union-attr]

    updated = await service.update_time(owner.id, item.id, time(12), now=due)
    assert updated.changed is True
    assert updated.schedule is not None
    assert updated.schedule.version == 2
    stored = await occurrence_for(db, schedule.id)
    assert stored.status == "cancelled"
    assert stored.schedule_version == 1
    assert as_utc(stored.scheduled_for) == due


async def test_mutations_reject_stale_expected_schedule_version(db):
    service, owner, item, due = await create_due_schedule(db)
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None

    updated = await service.update_time(
        owner.id,
        item.id,
        time(12),
        expected_version=schedule.version,
        now=due,
    )
    assert updated.changed is True and updated.schedule is not None
    assert updated.schedule.version == 2

    with pytest.raises(RecurringFenceLost, match="schedule version changed"):
        await service.disable(
            owner.id,
            item.id,
            expected_version=schedule.version,
        )
    unchanged = await service.get(owner.id, item.id)
    assert unchanged == updated.schedule

    disabled = await service.disable(
        owner.id,
        item.id,
        expected_version=updated.schedule.version,
    )
    assert disabled.changed is True and disabled.schedule is not None
    assert (disabled.schedule.status, disabled.schedule.version) == ("disabled", 3)

    with pytest.raises(RecurringFenceLost, match="schedule version changed"):
        await service.reenable(
            owner.id,
            item.id,
            expected_version=updated.schedule.version,
            now=due,
        )
    reenabled = await service.reenable(
        owner.id,
        item.id,
        expected_version=disabled.schedule.version,
        now=due,
    )
    assert reenabled.changed is True and reenabled.schedule is not None
    assert (reenabled.schedule.status, reenabled.schedule.version) == ("active", 4)


async def test_concurrent_expected_version_mutations_allow_one_generation_change(db):
    service, owner, item, due = await create_due_schedule(db)
    first = RecurringTaskReminderService(db)
    second = RecurringTaskReminderService(db)

    outcomes = await asyncio.gather(
        first.update_time(
            owner.id,
            item.id,
            time(12),
            expected_version=1,
            now=due,
        ),
        second.disable(owner.id, item.id, expected_version=1),
        return_exceptions=True,
    )

    assert sum(isinstance(outcome, RecurringFenceLost) for outcome in outcomes) == 1
    mutations = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    assert len(mutations) == 1
    assert mutations[0].changed is True
    stored = await service.get(owner.id, item.id)
    assert stored is not None
    assert stored.version == 2


async def test_update_time_retains_existing_explicit_timezone_when_omitted(db):
    owner, item, _state = await create_task(db, timezone="Europe/Saratov")
    service = RecurringTaskReminderService(db)
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    await service.create_daily(
        owner.id,
        item.id,
        time(19, 30),
        timezone="Europe/Moscow",
        timezone_source="explicit",
        now=now,
    )

    updated = await service.update_time(owner.id, item.id, time(20), now=now)

    assert updated.schedule is not None
    assert (updated.schedule.local_time, updated.schedule.timezone) == (
        time(20),
        "Europe/Moscow",
    )
    assert updated.schedule.timezone_source == "explicit"


async def test_same_day_update_preserves_cancelled_history_and_appends_generation(db):
    due = datetime(2026, 8, 10, 10, tzinfo=UTC)
    service, owner, item, _due = await create_due_schedule(db, due=due)
    await service.materialize_due(now=due)
    before = await occurrence_for(db, (await service.get(owner.id, item.id)).id)  # type: ignore[union-attr]
    original = (
        before.id,
        before.schedule_version,
        as_utc(before.scheduled_for),
        before.delivery_key,
    )

    updated = await service.update_time(
        owner.id,
        item.id,
        time(11),
        now=due + timedelta(minutes=1),
    )

    assert updated.schedule is not None
    assert updated.schedule.next_occurrence_at == due + timedelta(hours=1)
    rows_before_due = await occurrences_for(db, updated.schedule.id)
    assert len(rows_before_due) == 1
    assert (
        rows_before_due[0].id,
        rows_before_due[0].schedule_version,
        as_utc(rows_before_due[0].scheduled_for),
        rows_before_due[0].delivery_key,
    ) == original
    assert rows_before_due[0].status == "cancelled"

    assert await service.materialize_due(now=due + timedelta(hours=1)) == 1
    old, fresh = await occurrences_for(db, updated.schedule.id)
    assert (old.id, old.schedule_version, as_utc(old.scheduled_for), old.delivery_key) == original
    assert old.status == "cancelled"
    assert fresh.id != old.id
    assert fresh.status == "pending"
    assert fresh.schedule_version == updated.schedule.version == 2
    assert as_utc(fresh.scheduled_for) == due + timedelta(hours=1)
    assert fresh.delivery_key != old.delivery_key


async def test_same_day_reenable_preserves_cancelled_history_and_appends_generation(db):
    due = datetime(2026, 8, 10, 10, tzinfo=UTC)
    service, owner, item, _due = await create_due_schedule(db, due=due)
    schedule = await service.get(owner.id, item.id)
    await service.materialize_due(now=due)
    before = await occurrence_for(db, schedule.id)  # type: ignore[union-attr]
    original = (
        before.id,
        before.schedule_version,
        as_utc(before.scheduled_for),
        before.delivery_key,
    )
    await service.disable(owner.id, item.id)
    await service.update_time(
        owner.id,
        item.id,
        time(11),
        now=due + timedelta(minutes=1),
    )

    reenabled = await service.reenable(
        owner.id,
        item.id,
        now=due + timedelta(minutes=2),
    )

    assert reenabled.schedule is not None
    assert reenabled.schedule.next_occurrence_at == due + timedelta(hours=1)
    rows_before_due = await occurrences_for(db, reenabled.schedule.id)
    assert len(rows_before_due) == 1
    assert (
        rows_before_due[0].id,
        rows_before_due[0].schedule_version,
        as_utc(rows_before_due[0].scheduled_for),
        rows_before_due[0].delivery_key,
    ) == original
    assert rows_before_due[0].status == "cancelled"

    assert await service.materialize_due(now=due + timedelta(hours=1)) == 1
    old, fresh = await occurrences_for(db, reenabled.schedule.id)
    assert (old.id, old.schedule_version, as_utc(old.scheduled_for), old.delivery_key) == original
    assert old.status == "cancelled"
    assert fresh.id != old.id
    assert fresh.status == "pending"
    assert fresh.schedule_version == reenabled.schedule.version == 4
    assert as_utc(fresh.scheduled_for) == due + timedelta(hours=1)
    assert fresh.delivery_key != old.delivery_key


async def test_same_time_disable_reenable_appends_without_rewriting_cancelled_row(db):
    due = datetime(2026, 8, 10, 10, tzinfo=UTC)
    service, owner, item, _due = await create_due_schedule(db, due=due)
    schedule = await service.get(owner.id, item.id)
    await service.materialize_due(now=due)
    before = await occurrence_for(db, schedule.id)  # type: ignore[union-attr]
    original = (
        before.id,
        before.schedule_version,
        as_utc(before.scheduled_for),
        before.delivery_key,
    )

    await service.disable(owner.id, item.id)
    reenabled = await service.reenable(
        owner.id,
        item.id,
        now=due - timedelta(seconds=1),
    )

    assert reenabled.schedule is not None
    assert reenabled.schedule.next_occurrence_at == due
    assert reenabled.schedule.version == 3
    cancelled = await occurrence_for(db, reenabled.schedule.id)
    assert (
        cancelled.id,
        cancelled.schedule_version,
        as_utc(cancelled.scheduled_for),
        cancelled.delivery_key,
    ) == original
    assert cancelled.status == "cancelled"

    assert await service.materialize_due(now=due) == 1
    old, fresh = await occurrences_for(db, reenabled.schedule.id)
    assert (old.id, old.schedule_version, as_utc(old.scheduled_for), old.delivery_key) == original
    assert old.status == "cancelled"
    assert fresh.id != old.id
    assert fresh.status == "pending"
    assert fresh.schedule_version == 3
    assert as_utc(fresh.scheduled_for) == due
    assert fresh.delivery_key != old.delivery_key


async def test_same_day_update_never_reuses_sent_occurrence(db):
    due = datetime(2026, 8, 10, 10, tzinfo=UTC)
    service, owner, item, _due = await create_due_schedule(db, due=due)

    async def send(delivery):
        return 906

    assert await RecurringTaskReminderEngine(db, send).deliver_due(now=due) == 1
    updated = await service.update_time(
        owner.id,
        item.id,
        time(11),
        now=due + timedelta(minutes=1),
    )

    assert updated.schedule is not None
    assert updated.schedule.next_occurrence_at == datetime(2026, 8, 11, 11, tzinfo=UTC)


async def test_same_day_update_never_reuses_skipped_occurrence(db):
    due = datetime(2026, 8, 10, 10, tzinfo=UTC)
    service, owner, item, _due = await create_due_schedule(db, due=due, tier=GUEST)

    async def send(delivery):
        raise AssertionError("guest delivery must not run")

    engine = RecurringTaskReminderEngine(db, send)
    assert await engine.deliver_due(now=due) == 0
    assert await engine.deliver_due(now=due + timedelta(minutes=121)) == 0
    updated = await service.update_time(
        owner.id,
        item.id,
        time(13),
        now=due + timedelta(minutes=122),
    )

    assert updated.schedule is not None
    assert updated.schedule.next_occurrence_at == datetime(2026, 8, 11, 13, tzinfo=UTC)


async def test_disable_during_claim_forbids_mark_and_reenable_is_fresh_generation(db):
    service, owner, item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    claimed = (await service.claim_due(now=due))[0]

    disabled = await service.disable(owner.id, item.id)
    assert disabled.schedule is not None
    assert (disabled.schedule.status, disabled.schedule.version) == ("disabled", 2)
    assert await service.mark_sent(claimed, 77, now=due) is False

    reenabled = await service.reenable(owner.id, item.id, now=due)
    assert reenabled.schedule is not None
    assert (reenabled.schedule.status, reenabled.schedule.version) == ("active", 3)
    assert reenabled.schedule.start_local_date == date(2026, 8, 11)
    stored = await occurrence_for(db, claimed.schedule_id)
    assert stored.status == "cancelled"


async def test_terminal_completion_and_reopen_require_explicit_reenable(db):
    service, owner, item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    async with db.session() as session:
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item.id))
        state.status = "completed"
        state.completed_at = due
        state.version += 1

    completed = await service.complete_for_terminal_task(owner.id, item.id)
    assert completed.schedule is not None
    assert completed.schedule.status == "completed"
    assert await service.list_active(owner.id) == ()
    stored = await occurrence_for(db, completed.schedule.id)
    assert stored.status == "cancelled"

    async with db.session() as session:
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item.id))
        state.status = "active"
        state.completed_at = None
        state.version += 1
    assert (await service.get(owner.id, item.id)).status == "completed"  # type: ignore[union-attr]

    reenabled = await service.reenable(owner.id, item.id, now=due)
    assert reenabled.schedule is not None
    assert reenabled.schedule.status == "active"
    assert reenabled.schedule.version == 3


async def test_same_session_terminalization_is_idempotent_and_cancels_pending(db):
    service, owner, item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)

    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == owner.id).values(updated_at=User.updated_at)
        )
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item.id))
        state.status = "completed"
        state.completed_at = due
        state.version += 1

        first = await RecurringTaskReminderService.complete_for_terminal_task_in_session(
            session,
            owner.id,
            item.id,
        )
        second = await service.complete_for_terminal_task_in_session(
            session,
            owner.id,
            item.id,
        )

    assert first.changed is True and first.schedule is not None
    assert second.changed is False and second.schedule is not None
    assert first.schedule.status == second.schedule.status == "completed"
    assert first.schedule.version == second.schedule.version == 2
    stored = await occurrence_for(db, first.schedule.id)
    assert stored.status == "cancelled"
    assert stored.claim_token is None


async def test_same_session_terminalization_rejects_live_and_nonowner(db):
    service, owner, item, _due = await create_due_schedule(db)
    intruder, _other, _state = await create_task(db, telegram_id=61_041)

    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == owner.id).values(updated_at=User.updated_at)
        )
        with pytest.raises(RecurringTaskNotEligible) as live:
            await service.complete_for_terminal_task_in_session(
                session,
                owner.id,
                item.id,
            )
    assert live.value.code == "task_is_not_terminal"

    async with db.session() as session:
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item.id))
        state.status = "completed"
        state.completed_at = datetime(2026, 8, 10, tzinfo=UTC)
        state.version += 1

    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == intruder.id).values(updated_at=User.updated_at)
        )
        with pytest.raises(RecurringTaskNotEligible) as wrong_owner:
            await service.complete_for_terminal_task_in_session(
                session,
                intruder.id,
                item.id,
            )
    assert wrong_owner.value.code == "task_not_found"

    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    assert (schedule.status, schedule.version) == ("active", 1)


async def test_same_session_terminalization_fences_io_started_occurrence(db):
    service, owner, item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    claimed = (await service.claim_due(now=due))[0]
    assert await service.begin_delivery(claimed, now=due)

    async with db.session() as session:
        await session.execute(
            update(User).where(User.id == owner.id).values(updated_at=User.updated_at)
        )
        state = await session.scalar(select(TaskState).where(TaskState.inbox_item_id == item.id))
        state.status = "completed"
        state.completed_at = due
        state.version += 1
        completed = await service.complete_for_terminal_task_in_session(
            session,
            owner.id,
            item.id,
        )

    assert completed.changed is True and completed.schedule is not None
    assert (completed.schedule.status, completed.schedule.version) == ("completed", 2)
    stored = await occurrence_for(db, completed.schedule.id)
    assert stored.status == "skipped_stale"
    assert stored.claim_token is None
    assert stored.delivery_started_at is None


async def test_trashed_task_completes_schedule_and_cancels_pending_occurrence(db):
    service, owner, item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    async with db.session() as session:
        stored_item = await session.get(InboxItem, item.id)
        stored_item.pre_trash_status = stored_item.status
        stored_item.status = "trashed"
        stored_item.trashed_at = due
        stored_item.version += 1

    completed = await service.complete_for_terminal_task(owner.id, item.id)

    assert completed.schedule is not None
    assert completed.schedule.status == "completed"
    stored = await occurrence_for(db, completed.schedule.id)
    assert stored.status == "cancelled"


async def test_stale_materialization_records_one_day_and_does_not_catch_up_flood(db):
    service, owner, item, due = await create_due_schedule(
        db,
        due=datetime(2026, 7, 1, 11, tzinfo=UTC),
    )
    now = datetime(2026, 8, 10, 10, tzinfo=UTC)
    assert await service.materialize_due(now=now) == 1

    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    assert schedule.next_occurrence_at == datetime(2026, 8, 10, 11, tzinfo=UTC)
    async with db.sessions() as session:
        rows = (
            await session.scalars(
                select(RecurringTaskReminderOccurrence).where(
                    RecurringTaskReminderOccurrence.schedule_id == schedule.id
                )
            )
        ).all()
    assert [(row.local_date, row.status) for row in rows] == [(date(2026, 7, 1), "skipped_stale")]


async def test_long_downtime_delivers_only_current_day_when_it_is_within_grace(db):
    service, owner, item, _due = await create_due_schedule(
        db,
        due=datetime(2026, 7, 1, 11, tzinfo=UTC),
    )
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    delivered = []

    async def send(delivery):
        delivered.append(delivery.local_date)
        return 904

    assert await RecurringTaskReminderEngine(db, send).deliver_due(now=now) == 1
    assert delivered == [date(2026, 8, 10)]
    schedule = await service.get(owner.id, item.id)
    async with db.sessions() as session:
        rows = (
            await session.scalars(
                select(RecurringTaskReminderOccurrence)
                .where(RecurringTaskReminderOccurrence.schedule_id == schedule.id)  # type: ignore[union-attr]
                .order_by(RecurringTaskReminderOccurrence.local_date)
            )
        ).all()
    assert [(row.local_date, row.status) for row in rows] == [
        (date(2026, 7, 1), "skipped_stale"),
        (date(2026, 8, 10), "sent"),
    ]


async def test_concurrent_workers_claim_one_occurrence(db):
    service, _owner, _item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)

    first, second = await asyncio.gather(
        service.claim_due(now=due),
        service.claim_due(now=due),
    )
    assert len(first) + len(second) == 1


async def test_concurrent_engines_invoke_send_once_for_the_same_occurrence(db):
    _service, _owner, _item, due = await create_due_schedule(db)
    calls = []

    async def send(delivery):
        calls.append(delivery.delivery_key)
        await asyncio.sleep(0)
        return 906

    first = RecurringTaskReminderEngine(db, send)
    second = RecurringTaskReminderEngine(db, send)
    results = await asyncio.gather(
        first.deliver_due(now=due),
        second.deliver_due(now=due),
    )

    assert sum(results) == 1
    assert len(calls) == 1


async def test_lease_recovers_only_before_delivery_started(db):
    service, _owner, _item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    await service.claim_due(now=due)

    reclaimed = (await service.claim_due(now=due + timedelta(seconds=121)))[0]
    assert await service.begin_delivery(reclaimed, now=due + timedelta(seconds=121)) is True
    assert await service.claim_due(now=due + timedelta(seconds=242)) == ()


async def test_begin_delivery_atomically_skips_when_fresh_grace_expired(db):
    service, owner, item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    claimed = (await service.claim_due(now=due))[0]

    assert await service.begin_delivery(claimed, now=due + timedelta(minutes=121)) is False

    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    stored = await occurrence_for(db, schedule.id)
    assert stored.status == "skipped_stale"
    assert stored.delivery_started_at is None
    assert schedule.next_occurrence_at == due + timedelta(days=1)


async def test_pre_transport_retry_keeps_delivery_key_and_backoff(db):
    service, _owner, _item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    claimed = (await service.claim_due(now=due))[0]
    assert await service.release_retry(claimed, RuntimeError("private provider body"), now=due)

    assert await service.claim_due(now=due + timedelta(seconds=4)) == ()
    retried = (await service.claim_due(now=due + timedelta(seconds=5)))[0]
    assert retried.delivery_key == claimed.delivery_key
    async with db.sessions() as session:
        stored = await session.get(RecurringTaskReminderOccurrence, claimed.id)
    assert stored.last_error_type == "RuntimeError"
    assert stored.delivery_started_at is None


async def test_repeated_mark_sent_is_idempotent_and_advances_once(db):
    service, owner, item, due = await create_due_schedule(db)
    await service.materialize_due(now=due)
    claimed = (await service.claim_due(now=due))[0]
    assert await service.begin_delivery(claimed, now=due)

    assert await service.mark_sent(claimed, 909, now=due) is True
    assert await service.mark_sent(claimed, 909, now=due) is False

    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    assert schedule.next_occurrence_at == due + timedelta(days=1)
    stored = await occurrence_for(db, schedule.id)
    assert (stored.status, stored.telegram_message_id) == ("sent", 909)


async def test_access_downgrade_preserves_pending_and_resubscribe_uses_fresh_snapshot(db):
    service, owner, _item, due = await create_due_schedule(db, telegram_id=61_010)
    await service.materialize_due(now=due)
    claimed = (await service.claim_due(now=due))[0]
    await AccessService(db).set_guest(owner.telegram_id, source="test")

    assert await service.delivery_readiness(claimed) == "access_denied"
    assert await service.release_claim(claimed, attempted=False)
    await AccessService(db).grant_subscriber(owner.telegram_id, source="test")
    refreshed = (await service.claim_due(now=due))[0]
    assert refreshed.access_version > claimed.access_version
    assert refreshed.delivery_key == claimed.delivery_key


async def test_engine_sends_once_to_current_user_destination_and_advances_exactly_once(db):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_020)
    deliveries = []

    async def send(delivery):
        deliveries.append(delivery)
        return 901

    engine = RecurringTaskReminderEngine(db, send)
    assert await engine.deliver_due(now=due) == 1
    assert await engine.deliver_due(now=due + timedelta(minutes=1)) == 0
    assert len(deliveries) == 1
    assert deliveries[0].destination_id == owner.telegram_id
    assert deliveries[0].title == item.title
    assert deliveries[0].delivery_key.startswith("recurring:")

    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    assert schedule.next_occurrence_at == due + timedelta(days=1)
    stored = await occurrence_for(db, schedule.id)
    assert (stored.status, stored.telegram_message_id) == ("sent", 901)


async def test_delayed_multi_item_worker_never_sends_late_second_claim(db):
    due = datetime(2026, 8, 10, 11, tzinfo=UTC)
    first_service, first_owner, first_item, _due = await create_due_schedule(
        db,
        telegram_id=61_030,
        due=due,
    )
    second_service, second_owner, second_item, _due = await create_due_schedule(
        db,
        telegram_id=61_031,
        due=due,
    )
    clock = [due]
    sent = []

    async def send(delivery):
        sent.append(delivery.destination_id)
        clock[0] = due + timedelta(minutes=121)
        return 907

    engine = RecurringTaskReminderEngine(db, send, now_provider=lambda: clock[0])
    assert await engine.deliver_due(now=due) == 1
    assert sent == [first_owner.telegram_id]

    first_schedule = await first_service.get(first_owner.id, first_item.id)
    second_schedule = await second_service.get(second_owner.id, second_item.id)
    assert first_schedule is not None and second_schedule is not None
    assert first_schedule.next_occurrence_at == due + timedelta(days=1)
    assert second_schedule.next_occurrence_at == due + timedelta(days=1)
    second_occurrence = await occurrence_for(db, second_schedule.id)
    assert second_occurrence.status == "skipped_stale"


async def test_title_edit_between_claim_and_send_is_version_fenced(db, monkeypatch):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_032)
    sent_titles = []

    async def send(delivery):
        sent_titles.append(delivery.title)
        return 908

    engine = RecurringTaskReminderEngine(db, send)
    original_claim_due = engine.service.claim_due

    async def claim_then_edit(*, now=None):
        claimed = await original_claim_due(now=now)
        async with db.session() as session:
            stored_item = await session.get(InboxItem, item.id)
            stored_item.title = "Новое название"
            stored_item.version += 1
        return claimed

    monkeypatch.setattr(engine.service, "claim_due", claim_then_edit)
    assert await engine.deliver_due(now=due) == 0
    assert sent_titles == []

    monkeypatch.setattr(engine.service, "claim_due", original_claim_due)
    assert await engine.deliver_due(now=due) == 1
    assert sent_titles == ["Новое название"]
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None


@pytest.mark.parametrize("tier", [GUEST, BLOCKED])
async def test_engine_never_sends_guest_or_blocked_and_skips_after_grace(db, tier):
    service, owner, item, due = await create_due_schedule(
        db,
        telegram_id=61_021,
        tier=tier,
    )
    sent = []

    async def send(delivery):
        sent.append(delivery)
        return 1

    engine = RecurringTaskReminderEngine(db, send)
    assert await engine.deliver_due(now=due) == 0
    assert sent == []
    assert await engine.deliver_due(now=due + timedelta(minutes=121)) == 0
    stored = await occurrence_for(db, (await service.get(owner.id, item.id)).id)  # type: ignore[union-attr]
    assert stored.status == "skipped_stale"
    assert (await service.get(owner.id, item.id)).next_occurrence_at == due + timedelta(days=1)  # type: ignore[union-attr]


async def test_missing_user_produces_zero_send_calls(db):
    _service, owner, _item, due = await create_due_schedule(db, telegram_id=61_033)
    sent = []

    async def send(delivery):
        sent.append(delivery)
        return 1

    async with db.session() as session:
        await session.execute(delete(User).where(User.id == owner.id))

    assert await RecurringTaskReminderEngine(db, send).deliver_due(now=due) == 0
    assert sent == []


async def test_access_loss_before_claim_produces_zero_send_and_zero_attempts(db, monkeypatch):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_034)
    sent = []

    async def send(delivery):
        sent.append(delivery)
        return 1

    engine = RecurringTaskReminderEngine(db, send)
    original_materialize = engine.service.materialize_due

    async def materialize_then_downgrade(*, now=None):
        result = await original_materialize(now=now)
        await AccessService(db).set_guest(owner.telegram_id, source="test")
        return result

    monkeypatch.setattr(engine.service, "materialize_due", materialize_then_downgrade)
    assert await engine.deliver_due(now=due) == 0
    assert sent == []
    schedule = await service.get(owner.id, item.id)
    stored = await occurrence_for(db, schedule.id)  # type: ignore[union-attr]
    assert (stored.status, stored.attempt_count, stored.claim_token) == ("pending", 0, None)


async def test_access_loss_after_claim_produces_zero_send_and_releases_claim(db, monkeypatch):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_035)
    sent = []

    async def send(delivery):
        sent.append(delivery)
        return 1

    engine = RecurringTaskReminderEngine(db, send)
    original_claim = engine.service.claim_due

    async def claim_then_downgrade(*, now=None):
        claimed = await original_claim(now=now)
        await AccessService(db).block(owner.telegram_id, source="test")
        return claimed

    monkeypatch.setattr(engine.service, "claim_due", claim_then_downgrade)
    assert await engine.deliver_due(now=due) == 0
    assert sent == []
    schedule = await service.get(owner.id, item.id)
    stored = await occurrence_for(db, schedule.id)  # type: ignore[union-attr]
    assert (stored.status, stored.attempt_count, stored.claim_token) == ("pending", 0, None)


async def test_changed_trusted_destination_fences_stale_claim(db, monkeypatch):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_036)
    destinations = []

    async def send(delivery):
        destinations.append(delivery.destination_id)
        return 910

    engine = RecurringTaskReminderEngine(db, send)
    original_claim = engine.service.claim_due

    async def claim_then_change_destination(*, now=None):
        claimed = await original_claim(now=now)
        async with db.session() as session:
            stored_owner = await session.get(User, owner.id)
            stored_owner.telegram_id = 71_036
        return claimed

    monkeypatch.setattr(engine.service, "claim_due", claim_then_change_destination)
    assert await engine.deliver_due(now=due) == 0
    assert destinations == []

    monkeypatch.setattr(engine.service, "claim_due", original_claim)
    assert await engine.deliver_due(now=due) == 1
    assert destinations == [71_036]
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None


async def test_access_version_bounce_during_send_is_compensated_and_not_retried(db):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_022)
    deleted = []

    async def send(delivery):
        await AccessService(db).set_guest(owner.telegram_id, source="test")
        await AccessService(db).grant_subscriber(owner.telegram_id, source="test")
        return 902

    async def delete_sent(destination_id, message_id):
        deleted.append((destination_id, message_id))

    engine = RecurringTaskReminderEngine(db, send, delete_sent=delete_sent)
    assert await engine.deliver_due(now=due) == 0
    assert deleted == [(owner.telegram_id, 902)]
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    stored = await occurrence_for(db, schedule.id)
    assert stored.status == "skipped_stale"
    assert schedule.next_occurrence_at == due + timedelta(days=1)


async def test_persistent_access_loss_during_send_is_compensated_and_advanced(db):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_037)
    deleted = []

    async def send(delivery):
        await AccessService(db).block(owner.telegram_id, source="test")
        return 911

    async def delete_sent(destination_id, message_id):
        deleted.append((destination_id, message_id))

    engine = RecurringTaskReminderEngine(db, send, delete_sent=delete_sent)
    assert await engine.deliver_due(now=due) == 0
    assert deleted == [(owner.telegram_id, 911)]
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    assert schedule.status == "active"
    assert schedule.next_occurrence_at == due + timedelta(days=1)
    stored = await occurrence_for(db, schedule.id)
    assert stored.status == "skipped_stale"


async def test_task_completion_during_send_completes_schedule_and_compensates(db):
    service, owner, item, due = await create_due_schedule(db, telegram_id=61_023)
    deleted = []

    async def send(delivery):
        async with db.session() as session:
            state = await session.scalar(
                select(TaskState).where(TaskState.inbox_item_id == item.id)
            )
            state.status = "completed"
            state.completed_at = due
            state.version += 1
        return 905

    async def delete_sent(destination_id, message_id):
        deleted.append((destination_id, message_id))

    engine = RecurringTaskReminderEngine(db, send, delete_sent=delete_sent)
    assert await engine.deliver_due(now=due) == 0
    assert deleted == [(owner.telegram_id, 905)]
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    assert schedule.status == "completed"
    stored = await occurrence_for(db, schedule.id)
    assert stored.status == "skipped_stale"


async def test_transport_exception_is_uncertain_and_never_replays_occurrence(db):
    service, owner, item, due = await create_due_schedule(db)
    calls = []
    clock = [due]

    def at(moment):
        clock[0] = moment
        return moment

    async def send(delivery):
        calls.append((delivery.occurrence_id, delivery.delivery_key))
        if len(calls) == 1:
            raise RuntimeError("transport outcome is private and uncertain")
        return 912

    engine = RecurringTaskReminderEngine(
        db,
        send,
        lease_seconds=30,
        now_provider=lambda: clock[0],
    )
    assert await engine.deliver_due(now=at(due)) == 0

    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    first = await occurrence_for(db, schedule.id)
    first_key = first.delivery_key
    assert (first.status, first.attempt_count) == ("processing", 1)
    assert first.claim_token is not None
    assert as_utc(first.claimed_at) == due
    assert as_utc(first.delivery_started_at) == due
    assert first_key == f"recurring:{schedule.id}:v1:{due.date().isoformat()}"

    assert await engine.deliver_due(now=at(due + timedelta(seconds=29))) == 0
    assert await engine.deliver_due(now=at(due + timedelta(seconds=31))) == 0
    restarted = RecurringTaskReminderEngine(
        db,
        send,
        lease_seconds=30,
        now_provider=lambda: clock[0],
    )
    assert await restarted.deliver_due(now=at(due + timedelta(seconds=60))) == 0
    assert calls == [(first.id, first_key)]

    assert await restarted.deliver_due(now=at(due + timedelta(minutes=121))) == 0
    skipped = await occurrence_for(db, schedule.id)
    refreshed = await service.get(owner.id, item.id)
    assert refreshed is not None
    assert (skipped.status, skipped.attempt_count, skipped.delivery_key) == (
        "skipped_stale",
        1,
        first_key,
    )
    assert skipped.claim_token is None
    assert skipped.delivery_started_at is None
    assert refreshed.next_occurrence_at == due + timedelta(days=1)

    next_due = due + timedelta(days=1)
    assert await restarted.deliver_due(now=at(next_due)) == 1
    assert await restarted.deliver_due(now=at(next_due + timedelta(minutes=1))) == 0
    occurrences = await occurrences_for(db, schedule.id)
    assert len(occurrences) == 2
    first, second = occurrences
    assert (first.status, first.attempt_count, first.delivery_key) == (
        "skipped_stale",
        1,
        first_key,
    )
    assert (second.status, second.attempt_count, second.telegram_message_id) == (
        "sent",
        1,
        912,
    )
    assert second.delivery_key == f"recurring:{schedule.id}:v1:{next_due.date().isoformat()}"
    assert calls == [
        (first.id, first.delivery_key),
        (second.id, second.delivery_key),
    ]


async def test_cancelled_send_keeps_uncertainty_fence_and_propagates(db):
    service, owner, item, due = await create_due_schedule(db)
    calls = []
    clock = [due]

    def at(moment):
        clock[0] = moment
        return moment

    async def send(delivery):
        calls.append((delivery.occurrence_id, delivery.delivery_key))
        if len(calls) == 1:
            raise asyncio.CancelledError
        return 913

    engine = RecurringTaskReminderEngine(
        db,
        send,
        lease_seconds=30,
        now_provider=lambda: clock[0],
    )
    with pytest.raises(asyncio.CancelledError):
        await engine.deliver_due(now=at(due))

    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    first = await occurrence_for(db, schedule.id)
    first_key = first.delivery_key
    assert (first.status, first.attempt_count) == ("processing", 1)
    assert first.claim_token is not None
    assert as_utc(first.delivery_started_at) == due

    assert await engine.deliver_due(now=at(due + timedelta(seconds=29))) == 0
    assert await engine.deliver_due(now=at(due + timedelta(seconds=31))) == 0
    restarted = RecurringTaskReminderEngine(
        db,
        send,
        lease_seconds=30,
        now_provider=lambda: clock[0],
    )
    assert await restarted.deliver_due(now=at(due + timedelta(seconds=60))) == 0
    assert calls == [(first.id, first_key)]

    assert await restarted.deliver_due(now=at(due + timedelta(minutes=121))) == 0
    stored = await occurrence_for(db, schedule.id)
    assert stored.status == "skipped_stale"
    assert stored.attempt_count == 1
    assert stored.delivery_key == first_key
    assert stored.claim_token is None
    assert stored.delivery_started_at is None
    refreshed = await service.get(owner.id, item.id)
    assert refreshed is not None
    assert refreshed.next_occurrence_at == due + timedelta(days=1)

    next_due = due + timedelta(days=1)
    assert await restarted.deliver_due(now=at(next_due)) == 1
    assert await restarted.deliver_due(now=at(next_due + timedelta(minutes=1))) == 0
    occurrences = await occurrences_for(db, schedule.id)
    assert [(row.status, row.attempt_count) for row in occurrences] == [
        ("skipped_stale", 1),
        ("sent", 1),
    ]
    assert calls == [
        (occurrences[0].id, occurrences[0].delivery_key),
        (occurrences[1].id, occurrences[1].delivery_key),
    ]


@pytest.mark.parametrize("stage", ["readiness", "begin"])
async def test_cancelled_before_transport_safely_releases_claim(db, monkeypatch, stage):
    service, owner, item, due = await create_due_schedule(db)
    calls = []

    async def send(delivery):
        calls.append(delivery)
        return 1

    engine = RecurringTaskReminderEngine(db, send)

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        engine.service,
        "delivery_readiness" if stage == "readiness" else "begin_delivery",
        cancel,
    )
    with pytest.raises(asyncio.CancelledError):
        await engine.deliver_due(now=due)

    assert calls == []
    schedule = await service.get(owner.id, item.id)
    assert schedule is not None
    stored = await occurrence_for(db, schedule.id)
    assert (stored.status, stored.attempt_count, stored.claim_token) == ("pending", 0, None)
    assert stored.delivery_started_at is None


async def test_readiness_db_error_causes_zero_send_and_safe_release(db, monkeypatch):
    service, owner, item, due = await create_due_schedule(db)
    calls = []

    async def send(delivery):
        calls.append(delivery)
        return 1

    engine = RecurringTaskReminderEngine(db, send)

    async def fail_readiness(*args, **kwargs):
        raise RuntimeError("private database exception")

    monkeypatch.setattr(engine.service, "delivery_readiness", fail_readiness)
    assert await engine.deliver_due(now=due) == 0
    assert calls == []
    schedule = await service.get(owner.id, item.id)
    stored = await occurrence_for(db, schedule.id)  # type: ignore[union-attr]
    assert stored.status == "pending"
    assert stored.claim_token is None


async def test_send_success_mark_failure_is_uncertain_and_not_lease_reclaimed(db, monkeypatch):
    service, owner, item, due = await create_due_schedule(db)
    calls = []

    async def send(delivery):
        calls.append(delivery.delivery_key)
        return 903

    engine = RecurringTaskReminderEngine(db, send, lease_seconds=30)

    async def fail_mark(*args, **kwargs):
        raise RuntimeError("commit unavailable")

    monkeypatch.setattr(engine.service, "mark_sent", fail_mark)
    assert await engine.deliver_due(now=due) == 0
    assert await engine.deliver_due(now=due + timedelta(seconds=31)) == 0
    assert len(calls) == 1

    schedule = await service.get(owner.id, item.id)
    stored = await occurrence_for(db, schedule.id)  # type: ignore[union-attr]
    assert stored.status == "processing"
    assert stored.delivery_started_at is not None


async def test_private_command_and_transport_error_are_not_persisted_or_logged(db, caplog):
    service, owner, item, due = await create_due_schedule(db)
    sentinel = "PRIVATE_PROVIDER_RESPONSE_SENTINEL"
    title_sentinel = "PRIVATE_TASK_TITLE_SENTINEL"
    async with db.session() as session:
        stored_item = await session.get(InboxItem, item.id)
        assert stored_item is not None
        stored_item.title = title_sentinel

    async def send(delivery):
        raise RuntimeError(sentinel)

    with caplog.at_level("WARNING"):
        assert await RecurringTaskReminderEngine(db, send).deliver_due(now=due) == 0

    schedule = await service.get(owner.id, item.id)
    stored = await occurrence_for(db, schedule.id)  # type: ignore[union-attr]
    assert (stored.status, stored.attempt_count) == ("processing", 1)
    assert stored.delivery_started_at is not None
    assert stored.last_error_type is None
    assert f"occurrence_id={stored.id}" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert sentinel not in caplog.text
    assert title_sentinel not in caplog.text
    assert "private recurring command sentinel" not in caplog.text
    async with db.sessions() as session:
        persisted = await session.scalar(
            select(RecurringTaskReminderSchedule).where(
                RecurringTaskReminderSchedule.inbox_item_id == item.id
            )
        )
    assert not hasattr(persisted, "raw_text")
    assert not hasattr(stored, "title")
