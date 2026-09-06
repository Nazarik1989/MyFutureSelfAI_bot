import asyncio
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, time, timedelta

import pytest
from sqlalchemy import delete, event, func, select

from future_self.domain import (
    FocusService,
    InboxService,
    OnboardingFlow,
    TodayApplicationStatus,
    next_notification_utc,
)
from future_self.drafts import DraftInboxService
from future_self.models import (
    Goal,
    InboxItem,
    RecurringTaskReminderSchedule,
    Routine,
    TaskReminder,
    TaskState,
    User,
    WeeklyFocus,
)
from future_self.repositories import ProfileRepository, UserRepository
from future_self.schemas import ParsedThought, VisionSummary


async def test_creates_user_once(db):
    async with db.session() as session:
        repository = UserRepository(session)
        first = await repository.get_or_create(1001, "Europe/Moscow")
        first_id = first.id
    async with db.session() as session:
        second = await UserRepository(session).get_or_create(1001, "UTC")
        count = await session.scalar(select(func.count(User.id)))
    assert second.id == first_id
    assert second.timezone == "Europe/Moscow"
    assert count == 1


async def test_saves_and_updates_vision_profile(db):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(1, "UTC")
        profile = await ProfileRepository(session).upsert(
            user,
            {"future_life": "Живу у моря"},
            VisionSummary(summary="Живу у моря", values=["свобода"]),
        )
        profile_id = profile.id
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(1, "UTC")
        updated = await ProfileRepository(session).upsert(
            user,
            {"future_life": "Живу у моря и работаю"},
            VisionSummary(summary="Живу у моря и работаю", values=["свобода"]),
        )
    assert updated.id == profile_id
    assert updated.summary == "Живу у моря и работаю"
    assert user.onboarding_completed is True


def test_onboarding_transitions_and_required_skip():
    answers = OnboardingFlow.answer({}, 0, "Аня")
    assert answers == {"display_name": "Аня"}
    assert OnboardingFlow.next_step(0) == 1
    assert OnboardingFlow.previous_step(1) == 0
    with pytest.raises(ValueError, match="нельзя пропустить"):
        OnboardingFlow.answer(answers, 0, None)
    optional = OnboardingFlow.answer(answers, 3, None)
    assert "residence" not in optional


async def test_classifies_thought_with_fake_ai(db, fake_ai):
    service = InboxService(db, fake_ai, "UTC")
    parsed = await service.classify("Нужно сделать отчёт")
    assert parsed.kind == "task"
    assert parsed.title == "Нужно сделать отчёт"


async def test_confirmation_and_discard_are_idempotent(db, fake_ai):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(10, "UTC")
        user_id = user.id
    service = DraftInboxService(db, 60)
    parsed = ParsedThought(kind="task", title="Отчёт", next_step="Открыть документ")
    accepted = await service.create(
        user_id=user_id,
        telegram_user_id=10,
        chat_id=100,
        source="text",
        raw_text="Сделать отчёт",
        parsed=parsed,
    )
    assert (await service.confirm(accepted.id, 1, 10, 100)).ok is True
    assert (await service.confirm(accepted.id, 1, 10, 100)).ok is False
    dropped = await service.create(
        user_id=user_id,
        telegram_user_id=10,
        chat_id=100,
        source="text",
        raw_text="Идея",
        parsed=parsed,
    )
    assert (await service.drop(dropped.id, 1, 10, 100)).ok is True
    assert (await service.drop(dropped.id, 1, 10, 100)).ok is False
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1


async def test_recent_exact_confirmed_duplicate_reuses_live_inbox_item(db):
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(11, "UTC")
        owner_id = owner.id
    service = DraftInboxService(db, 60)
    parsed = ParsedThought(
        kind="task",
        title="Позвонить Варваре",
        description="Обсудить планы",
        next_step="Позвонить",
    )
    first = await service.create(
        user_id=owner_id,
        telegram_user_id=11,
        chat_id=101,
        source="text",
        raw_text="Позвонить Варваре",
        parsed=parsed,
    )
    second = await service.create(
        user_id=owner_id,
        telegram_user_id=11,
        chat_id=101,
        source="voice",
        raw_text="позвонить Варваре!",
        parsed=parsed,
    )

    created = await service.confirm(first.id, first.version, 11, 101)
    duplicate = await service.confirm(second.id, second.version, 11, 101)

    assert created.ok and not created.duplicate
    assert duplicate.ok and duplicate.duplicate
    assert duplicate.inbox_item.id == created.inbox_item.id
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 1


async def test_recent_completed_task_is_not_reused_as_active_duplicate(db):
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(12, "UTC")
        owner_id = owner.id
    service = DraftInboxService(db, 60)
    parsed = ParsedThought(
        kind="task",
        title="Позвонить Варваре",
        description="Обсудить планы",
        next_step="Позвонить",
    )
    first = await service.create(
        user_id=owner_id,
        telegram_user_id=12,
        chat_id=102,
        source="text",
        raw_text="Позвонить Варваре",
        parsed=parsed,
    )
    created = await service.confirm(first.id, first.version, 12, 102)
    async with db.session() as session:
        state = await session.scalar(
            select(TaskState).where(TaskState.inbox_item_id == created.inbox_item.id)
        )
        state.status = "completed"
        state.completed_at = datetime.now(UTC)
        state.version += 1

    second = await service.create(
        user_id=owner_id,
        telegram_user_id=12,
        chat_id=102,
        source="text",
        raw_text="Позвонить Варваре",
        parsed=parsed,
    )
    repeated = await service.confirm(second.id, second.version, 12, 102)

    assert repeated.ok and not repeated.duplicate
    assert repeated.inbox_item.id != created.inbox_item.id
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(InboxItem.id))) == 2


async def test_recent_legacy_archived_active_task_is_reused(db):
    async with db.session() as session:
        owner = await UserRepository(session).get_or_create(13, "UTC")
        owner_id = owner.id
    service = DraftInboxService(db, 60)
    parsed = ParsedThought(kind="task", title="Подать документы", next_step="Собрать пакет")
    first = await service.create(
        user_id=owner_id,
        telegram_user_id=13,
        chat_id=103,
        source="text",
        raw_text="Подать документы",
        parsed=parsed,
    )
    created = await service.confirm(first.id, first.version, 13, 103)
    async with db.session() as session:
        item = await session.get(InboxItem, created.inbox_item.id)
        item.status = "archived"

    second = await service.create(
        user_id=owner_id,
        telegram_user_id=13,
        chat_id=103,
        source="text",
        raw_text="Подать документы",
        parsed=parsed,
    )
    repeated = await service.confirm(second.id, second.version, 13, 103)

    assert repeated.ok and repeated.duplicate
    assert repeated.inbox_item.id == created.inbox_item.id


async def test_user_data_is_isolated(db, fake_ai):
    async with db.session() as session:
        first_user = await UserRepository(session).get_or_create(10, "UTC")
        second_user = await UserRepository(session).get_or_create(20, "UTC")
        first_id, second_id = first_user.id, second_user.id
    service = DraftInboxService(db, 60)
    first_draft = await service.create(
        user_id=first_id,
        telegram_user_id=10,
        chat_id=100,
        source="text",
        raw_text="Сделать А",
        parsed=ParsedThought(kind="task", title="А"),
    )
    second_draft = await service.create(
        user_id=second_id,
        telegram_user_id=20,
        chat_id=200,
        source="text",
        raw_text="Сделать Б",
        parsed=ParsedThought(kind="task", title="Б"),
    )
    await service.confirm(first_draft.id, 1, 10, 100)
    await service.confirm(second_draft.id, 1, 20, 200)
    async with db.sessions() as session:
        first = await session.scalar(select(User).where(User.telegram_id == 10))
        titles = list(
            (
                await session.scalars(select(InboxItem.title).where(InboxItem.user_id == first.id))
            ).all()
        )
    assert titles == ["А"]


async def test_today_uses_only_confirmed_tasks_and_active_items(db, fake_ai):
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(30, "UTC")
        session.add_all(
            [
                Goal(
                    user_id=user.id,
                    life_area="здоровье",
                    title="Двигаться",
                    outcome="Больше энергии",
                    progress_criterion="3 раза",
                    horizon="месяц",
                    status="active",
                    priority=5,
                    vision_link="Энергичная жизнь",
                ),
                InboxItem(
                    user_id=user.id,
                    kind="task",
                    title="Подтверждено",
                    raw_text="текст",
                    source="text",
                    status="confirmed",
                ),
                InboxItem(
                    user_id=user.id,
                    kind="task",
                    title="Черновик",
                    raw_text="текст",
                    source="text",
                    status="pending",
                ),
            ]
        )
        await session.flush()
        goal = await session.scalar(select(Goal).where(Goal.user_id == user.id))
        session.add(
            Routine(
                user_id=user.id,
                goal_id=goal.id,
                frequency="ежедневно",
                minimum_version="2 минуты",
                normal_version="15 минут ходьбы",
                status="active",
            )
        )
        user_id = user.id
    plan = await FocusService(db, fake_ai).generate(user_id)
    assert plan.main_focus == "Один устойчивый шаг"
    assert fake_ai.last_today_context["confirmed_tasks"] == ["Подтверждено"]
    assert fake_ai.last_today_context["goals"] == ["Двигаться"]


async def test_today_uses_only_confirmed_focus_for_current_local_week(db, fake_ai):
    local_today = datetime.now(UTC).date()
    current_week = local_today - timedelta(days=local_today.weekday())
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(31, "UTC")
        session.add_all(
            [
                WeeklyFocus(
                    owner_id=user.id,
                    week_start=current_week - timedelta(days=7),
                    focus="Прошлая неделя",
                    approach=None,
                    small_steps=[],
                    source="text",
                ),
                WeeklyFocus(
                    owner_id=user.id,
                    week_start=current_week,
                    focus="Текущий подтверждённый ориентир",
                    approach="Спокойно",
                    small_steps=["Один шаг"],
                    source="text",
                ),
            ]
        )
        user_id = user.id

    plan, weekly_focus = await FocusService(db, fake_ai).generate_with_weekly_focus(user_id)

    assert plan.main_focus == "Один устойчивый шаг"
    assert weekly_focus == "Текущий подтверждённый ориентир"
    assert fake_ai.last_today_context["weekly_focus"] == weekly_focus


async def test_today_application_snapshot_is_immutable_and_fences_exact_generation(
    db,
    fake_ai,
):
    now = datetime(2026, 8, 17, 8, tzinfo=UTC)
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(311, "Europe/Moscow")
        user.access_tier = "subscriber"
        user.access_version = 7
        focus = WeeklyFocus(
            owner_id=user.id,
            week_start=now.date(),
            focus="Точный ориентир",
            approach=None,
            small_steps=[],
            source="text",
            version=4,
        )
        session.add(focus)
        await session.flush()
        user_id = user.id
        focus_public_id = focus.public_id

    service = FocusService(db, fake_ai)
    snapshot = await service.materialize_today_application(
        user_id,
        include_weekly_focus=True,
        now=now,
    )

    assert snapshot.actor_id == user_id
    assert snapshot.telegram_id == 311
    assert snapshot.access_tier == "subscriber"
    assert snapshot.access_version == 7
    assert snapshot.timezone == "Europe/Moscow"
    assert snapshot.local_week_start == now.date()
    assert snapshot.weekly_focus_public_id == focus_public_id
    assert snapshot.weekly_focus_version == 4
    assert snapshot.weekly_focus == "Точный ориентир"
    assert (await service.check_today_application(snapshot, now=now)).is_current is True

    detached = snapshot.provider_context()
    detached["weekly_focus"] = "Подмена"
    assert snapshot.provider_context()["weekly_focus"] == "Точный ориентир"
    with pytest.raises(FrozenInstanceError):
        snapshot.timezone = "UTC"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("actor_missing", TodayApplicationStatus.ACTOR_CHANGED),
        ("downgrade", TodayApplicationStatus.ACCESS_CHANGED),
        ("bounce", TodayApplicationStatus.ACCESS_CHANGED),
        ("timezone", TodayApplicationStatus.TIMEZONE_CHANGED),
    ],
)
async def test_today_application_pre_provider_actor_access_and_timezone_fences(
    db,
    fake_ai,
    mutation,
    expected,
):
    now = datetime(2026, 8, 12, 8, tzinfo=UTC)
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(312, "Europe/Moscow")
        user.access_tier = "subscriber"
        user.access_version = 11
        user_id = user.id

    service = FocusService(db, fake_ai)
    snapshot = await service.materialize_today_application(
        user_id,
        include_weekly_focus=True,
        now=now,
    )
    async with db.session() as session:
        user = await session.get(User, user_id)
        assert user is not None
        if mutation == "actor_missing":
            await session.execute(delete(User).where(User.id == user_id))
        elif mutation == "downgrade":
            user.access_tier = "guest"
            user.access_version += 1
        elif mutation == "bounce":
            user.access_version += 2
        else:
            user.timezone = "Asia/Tokyo"

    check = await service.check_today_application(snapshot, now=now)

    assert check.status is expected
    assert check.is_current is False


async def test_today_application_fence_rejects_local_monday_rollover(db, fake_ai):
    local_sunday = datetime(2026, 8, 16, 20, 59, tzinfo=UTC)
    local_monday = datetime(2026, 8, 16, 21, 0, tzinfo=UTC)
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(313, "Europe/Moscow")
        user_id = user.id

    service = FocusService(db, fake_ai)
    snapshot = await service.materialize_today_application(
        user_id,
        include_weekly_focus=True,
        now=local_sunday,
    )

    assert snapshot.local_week_start.isoformat() == "2026-08-10"
    assert (
        await service.check_today_application(snapshot, now=local_sunday)
    ).status is TodayApplicationStatus.CURRENT
    assert (
        await service.check_today_application(snapshot, now=local_monday)
    ).status is TodayApplicationStatus.WEEK_CHANGED


@pytest.mark.parametrize("mutation", ["edit", "delete"])
async def test_today_application_detects_focus_change_while_provider_is_running(
    db,
    fake_ai,
    monkeypatch,
    mutation,
):
    now = datetime(2026, 8, 17, 8, tzinfo=UTC)
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(314, "Europe/Moscow")
        focus = WeeklyFocus(
            owner_id=user.id,
            week_start=now.date(),
            focus="Исходный ориентир",
            approach=None,
            small_steps=[],
            source="text",
        )
        session.add(focus)
        await session.flush()
        user_id = user.id
        focus_id = focus.id

    service = FocusService(db, fake_ai)
    snapshot = await service.materialize_today_application(
        user_id,
        include_weekly_focus=True,
        now=now,
    )
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()
    provider_calls = 0
    original_make_today_plan = fake_ai.make_today_plan

    async def blocked_provider(context):
        nonlocal provider_calls
        provider_calls += 1
        provider_started.set()
        await provider_release.wait()
        return await original_make_today_plan(context)

    monkeypatch.setattr(fake_ai, "make_today_plan", blocked_provider)
    provider_task = asyncio.create_task(service.generate_today_plan(snapshot))
    await provider_started.wait()
    async with db.session() as session:
        focus = await session.get(WeeklyFocus, focus_id)
        assert focus is not None
        if mutation == "edit":
            focus.focus = "Новая версия"
            focus.version += 1
        else:
            await session.delete(focus)
    provider_release.set()

    await provider_task
    post_provider = await service.check_today_application(snapshot, now=now)

    assert provider_calls == 1
    assert post_provider.status is TodayApplicationStatus.FOCUS_CHANGED


async def test_today_application_detects_focus_created_after_confirmed_absence(db, fake_ai):
    now = datetime(2026, 8, 17, 8, tzinfo=UTC)
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(315, "Europe/Moscow")
        user_id = user.id

    service = FocusService(db, fake_ai)
    snapshot = await service.materialize_today_application(
        user_id,
        include_weekly_focus=True,
        now=now,
    )
    assert snapshot.weekly_focus_public_id is None
    assert snapshot.weekly_focus_version is None

    async with db.session() as session:
        session.add(
            WeeklyFocus(
                owner_id=user_id,
                week_start=now.date(),
                focus="Появившийся ориентир",
                approach=None,
                small_steps=[],
                source="text",
            )
        )

    assert (
        await service.check_today_application(snapshot, now=now)
    ).status is TodayApplicationStatus.FOCUS_CHANGED


async def test_today_legacy_snapshot_omits_weekly_context_and_reads(db, fake_ai):
    now = datetime(2026, 8, 17, 8, tzinfo=UTC)
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(316, "Europe/Moscow")
        user.access_tier = "subscriber"
        session.add(
            WeeklyFocus(
                owner_id=user.id,
                week_start=now.date(),
                focus="Не должен читаться",
                approach=None,
                small_steps=[],
                source="text",
            )
        )
        user_id = user.id

    statements: list[str] = []

    def capture_statement(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(db.engine.sync_engine, "before_cursor_execute", capture_statement)
    try:
        service = FocusService(db, fake_ai)
        snapshot = await service.materialize_today_application(
            user_id,
            include_weekly_focus=False,
            now=now,
        )
        assert (await service.check_today_application(snapshot, now=now)).is_current is True
        await service.generate_today_plan(snapshot)
    finally:
        event.remove(db.engine.sync_engine, "before_cursor_execute", capture_statement)

    assert snapshot.includes_weekly_focus is False
    assert snapshot.weekly_focus is None
    assert "weekly_focus" not in fake_ai.last_today_context
    assert not any("weekly_focuses" in statement.casefold() for statement in statements)


async def test_today_weekly_focus_keeps_urgent_tasks_and_nearest_reminders_bounded(
    db,
    fake_ai,
    monkeypatch,
):
    now = datetime(2026, 8, 17, 8, tzinfo=UTC)
    week_start = now.date()
    async with db.session() as session:
        user = await UserRepository(session).get_or_create(32, "Europe/Moscow")
        session.add(
            WeeklyFocus(
                owner_id=user.id,
                week_start=week_start,
                focus="Главный ориентир недели",
                approach=None,
                small_steps=[],
                source="text",
            )
        )

        async def add_task(
            title: str,
            event_at: datetime,
            *,
            inbox_status: str = "confirmed",
            state_status: str = "active",
        ) -> tuple[InboxItem, TaskState]:
            item = InboxItem(
                user_id=user.id,
                kind="task",
                title=title,
                raw_text=f"PRIVATE_RAW_{title}",
                source="text",
                status=inbox_status,
            )
            session.add(item)
            await session.flush()
            state = TaskState(
                owner_id=user.id,
                inbox_item_id=item.id,
                status=state_status,
                event_at=event_at,
                timezone="Europe/Moscow",
            )
            session.add(state)
            await session.flush()
            return item, state

        await add_task("Просроченная подтверждённая", now - timedelta(hours=2))
        await add_task("Срочная подтверждённая", now + timedelta(hours=1))
        await add_task("Будущая подтверждённая", now + timedelta(days=2))
        await add_task("Неподтверждённая", now, inbox_status="pending")
        await add_task("Завершённая", now, state_status="completed")

        reminder_specs = (
            ("one_shot", "Разовое 1", 1),
            ("daily", "Ежедневное 2", 2),
            ("one_shot", "Разовое 3", 3),
            ("daily", "Ежедневное 4", 4),
            ("one_shot", "Разовое 5", 5),
            ("one_shot", "Разовое вне лимита", 6),
        )
        for index, (kind, title, offset_hours) in enumerate(reminder_specs, start=1):
            next_at = now + timedelta(hours=offset_hours)
            item, state = await add_task(title, now + timedelta(days=2, hours=index))
            if kind == "one_shot":
                session.add(
                    TaskReminder(
                        inbox_item_id=item.id,
                        telegram_user_id=user.telegram_id,
                        chat_id=user.telegram_id,
                        event_at=next_at + timedelta(hours=1),
                        remind_at=next_at,
                        timezone="Europe/Moscow",
                        delivery_key=f"today-context-{index}",
                        task_version=state.version,
                        status="pending",
                    )
                )
            else:
                session.add(
                    RecurringTaskReminderSchedule(
                        owner_id=user.id,
                        inbox_item_id=item.id,
                        recurrence_kind="daily",
                        local_time=time(15, index),
                        timezone="Europe/Moscow",
                        timezone_source="profile",
                        start_local_date=week_start,
                        next_occurrence_at=next_at,
                        status="active",
                        version=1,
                    )
                )
        user_id = user.id

    transaction_open = False
    original_sessions = db.sessions

    @asynccontextmanager
    async def monitored_sessions():
        nonlocal transaction_open
        async with original_sessions() as session:
            transaction_open = True
            try:
                yield session
            finally:
                transaction_open = False

    original_make_today_plan = fake_ai.make_today_plan

    async def assert_ai_outside_transaction(context):
        assert transaction_open is False
        return await original_make_today_plan(context)

    monkeypatch.setattr(db, "sessions", monitored_sessions)
    monkeypatch.setattr(fake_ai, "make_today_plan", assert_ai_outside_transaction)

    plan, weekly_focus = await FocusService(db, fake_ai).generate_with_weekly_focus(
        user_id,
        now=now,
    )

    assert plan.main_focus == "Один устойчивый шаг"
    assert weekly_focus == "Главный ориентир недели"
    assert fake_ai.last_today_context["weekly_focus"] == weekly_focus
    assert fake_ai.last_today_context["urgent_confirmed_tasks"] == [
        {
            "title": "Просроченная подтверждённая",
            "event_at": (now - timedelta(hours=2)).isoformat(timespec="seconds"),
        },
        {
            "title": "Срочная подтверждённая",
            "event_at": (now + timedelta(hours=1)).isoformat(timespec="seconds"),
        },
    ]
    reminders = fake_ai.last_today_context["upcoming_reminders"]
    assert len(reminders) == 5
    assert [(reminder["kind"], reminder["title"]) for reminder in reminders] == [
        ("one_shot", "Разовое 1"),
        ("daily", "Ежедневное 2"),
        ("one_shot", "Разовое 3"),
        ("daily", "Ежедневное 4"),
        ("one_shot", "Разовое 5"),
    ]
    assert "PRIVATE_RAW" not in repr(fake_ai.last_today_context)


@pytest.mark.parametrize(
    ("zone", "hour", "now", "expected"),
    [
        (
            "Europe/Moscow",
            8,
            datetime(2026, 1, 10, 2, tzinfo=UTC),
            datetime(2026, 1, 10, 5, tzinfo=UTC),
        ),
        (
            "Europe/Berlin",
            8,
            datetime(2026, 7, 10, 5, tzinfo=UTC),
            datetime(2026, 7, 10, 6, tzinfo=UTC),
        ),
        (
            "America/New_York",
            8,
            datetime(2026, 1, 10, 14, tzinfo=UTC),
            datetime(2026, 1, 11, 13, tzinfo=UTC),
        ),
    ],
)
def test_notification_timezones(zone, hour, now, expected):
    assert next_notification_utc(zone, hour, now=now) == expected
