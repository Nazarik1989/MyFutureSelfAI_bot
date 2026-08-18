from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import func, select
from telegram import User as TelegramUser
from telegram.ext import ExtBot

from future_self.access import AccessService
from future_self.bot import FutureSelfBot
from future_self.models import (
    DraftInboxItem,
    InboxItem,
    RecurringTaskReminderSchedule,
    TaskReminder,
    WeeklyFocus,
    WeeklyFocusChange,
)
from future_self.schemas import WeeklyReviewExtraction
from future_self.weekly_review import WeeklyReviewPhase
from future_self.weekly_review_handlers import WEEKLY_REVIEW_QUESTION
from tests.test_runtime import (
    RuntimeTranscription,
    _patch_runtime_weekly_transport,
    _runtime_callback_for_fragment,
    _runtime_subscriber,
    _runtime_voice_update,
    _runtime_weekly_callback_update,
    _runtime_weekly_command_update,
    _runtime_weekly_text_update,
    runtime_settings,
)

_WEEKLY_TABLES = (
    "weekly_review_sessions",
    "weekly_focuses",
    "weekly_focus_changes",
)


async def _application_for(
    db,
    fake_ai,
    transcription,
    *,
    telegram_id: int,
    enabled: bool = True,
    admin_only: bool = False,
    tier: str = "subscriber",
):
    core = FutureSelfBot(
        runtime_settings(
            database_url=db.url,
            enable_weekly_review=enabled,
            weekly_review_admin_only=admin_only,
        ),
        db,
        fake_ai,
        transcription,
    )
    application = core.build()
    application._initialized = True
    owner = await _runtime_subscriber(
        core,
        db,
        telegram_id,
        onboarding_completed=True,
    )
    if tier == "admin":
        await AccessService(db).grant_admin(telegram_id, source="stage8a-correction-test")
        owner = await core._user(telegram_id)
    return core, application, owner


async def _current(core, owner, telegram_id: int):
    return await core.weekly_review_service.current_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
    )


async def _write_counts(db) -> tuple[int, int, int, int, int, int]:
    async with db.sessions() as session:
        return (
            int(await session.scalar(select(func.count(WeeklyFocus.id))) or 0),
            int(await session.scalar(select(func.count(WeeklyFocusChange.id))) or 0),
            int(await session.scalar(select(func.count(DraftInboxItem.id))) or 0),
            int(await session.scalar(select(func.count(InboxItem.id))) or 0),
            int(await session.scalar(select(func.count(TaskReminder.id))) or 0),
            int(await session.scalar(select(func.count(RecurringTaskReminderSchedule.id))) or 0),
        )


@pytest.mark.parametrize(
    ("enabled", "admin_only", "tier", "allowed"),
    [
        pytest.param(False, False, "subscriber", False, id="disabled-subscriber"),
        pytest.param(False, False, "admin", False, id="disabled-admin"),
        pytest.param(False, True, "subscriber", False, id="disabled-admin-only-subscriber"),
        pytest.param(False, True, "admin", False, id="disabled-admin-only-admin"),
        pytest.param(True, False, "subscriber", True, id="enabled-subscriber"),
        pytest.param(True, False, "admin", True, id="enabled-admin"),
        pytest.param(True, True, "subscriber", False, id="pilot-subscriber"),
        pytest.param(True, True, "admin", True, id="pilot-admin"),
    ],
)
async def test_real_application_weekly_policy_full_flag_tier_surface_matrix(
    db,
    fake_ai,
    monkeypatch,
    enabled,
    admin_only,
    tier,
    allowed,
):
    telegram_id = 719_000 + int(enabled) * 100 + int(admin_only) * 10 + (tier == "admin")
    transcription = RuntimeTranscription("Спланируем неделю")
    core, application, owner = await _application_for(
        db,
        fake_ai,
        transcription,
        telegram_id=telegram_id,
        enabled=enabled,
        admin_only=admin_only,
        tier=tier,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    statements: list[str] = []

    def capture_sql(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(str(statement).casefold())

    sqlalchemy_event.listen(db.engine.sync_engine, "before_cursor_execute", capture_sql)
    try:
        await core.weekly_review_scheduled_notification(
            application.bot,
            telegram_id,
            owner.timezone,
        )
        assert len(transport.sent) == int(allowed)
        if allowed:
            assert (await _current(core, owner, telegram_id)).status == "not_found"

        await application.process_update(
            _runtime_weekly_command_update(
                application,
                telegram_id,
                "/week",
                update_id=telegram_id * 10 + 1,
                source_message_id=71_901,
            )
        )
        if allowed:
            root = await _current(core, owner, telegram_id)
            assert root.status == "found"
            assert root.session is not None
            assert root.session.phase is WeeklyReviewPhase.ROOT
            callback_data = _runtime_callback_for_fragment(
                transport.edits[-1]["reply_markup"],
                "Начать обзор",
                prefix="wrev:",
            )
            callback_message = transport.sent_messages[-1]
        else:
            callback_message = transport.make_message(
                telegram_id,
                97_100 + telegram_id,
                "stale weekly screen",
            )
            tokens = await core.weekly_review_capabilities.issue(
                actions=("start",),
                owner_id=owner.id,
                telegram_user_id=telegram_id,
                chat_id=telegram_id,
                canonical_message_id=callback_message.message_id,
                access_version=owner.access_version,
                week_start=core.weekly_review_service.target_week_start(owner.timezone),
            )
            callback_data = f"wrev:{tokens['start']}"

        callback_update, query = _runtime_weekly_callback_update(
            application,
            TelegramUser(telegram_id, "Варвара", False),
            callback_message,
            callback_data,
            update_id=telegram_id * 10 + 2,
        )
        answers_before = len(transport.answers)
        await application.process_update(callback_update)
        assert len(transport.answers) == answers_before + 1
        assert transport.answers[-1]["callback_query_id"] == query.id

        await application.process_update(
            _runtime_weekly_text_update(
                application,
                telegram_id,
                "Давай скорректируем систему",
                update_id=telegram_id * 10 + 3,
                source_message_id=71_903,
            )
        )
        voice_update, _progress = _runtime_voice_update(
            application,
            telegram_id,
            update_id=telegram_id * 10 + 4,
            source_message_id=71_904,
            progress_message_id=71_905,
        )
        await application.process_update(voice_update)
        if allowed:
            current = await _current(core, owner, telegram_id)
            assert current.status == "found"
            assert current.session is not None
            assert current.session.phase is WeeklyReviewPhase.AWAITING_INPUT
        await application.process_update(
            _runtime_weekly_command_update(
                application,
                telegram_id,
                "/today",
                update_id=telegram_id * 10 + 5,
                source_message_id=71_906,
            )
        )
    finally:
        sqlalchemy_event.remove(db.engine.sync_engine, "before_cursor_execute", capture_sql)

    weekly_sql = [
        statement
        for statement in statements
        if any(table_name in statement for table_name in _WEEKLY_TABLES)
    ]
    assert bool(weekly_sql) is allowed
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert fake_ai.last_today_context is not None
    assert ("weekly_focus" in fake_ai.last_today_context) is allowed
    if not allowed:
        assert weekly_sql == []
        async with db.sessions() as session:
            assert await session.scalar(select(func.count(WeeklyFocus.id))) == 0
            assert await session.scalar(select(func.count(WeeklyFocusChange.id))) == 0


async def test_real_application_shared_reducer_root_text_stt_view_and_reprompts(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 719_200
    transcription = RuntimeTranscription(
        "Обычный длинный ответ в ROOT не должен запускать weekly extraction"
    )
    core, application, owner = await _application_for(
        db,
        fake_ai,
        transcription,
        telegram_id=telegram_id,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            "Обзор недели",
            update_id=72_001,
            source_message_id=72_101,
        )
    )
    root = await _current(core, owner, telegram_id)
    assert root.status == "found"
    assert root.session is not None
    assert root.session.phase is WeeklyReviewPhase.ROOT

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            "Обычный длинный ответ в ROOT не должен запускать weekly extraction",
            update_id=72_002,
            source_message_id=72_102,
        )
    )
    assert (await _current(core, owner, telegram_id)).session == root.session

    voice_update, _progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=72_003,
        source_message_id=72_103,
        progress_message_id=72_203,
    )
    await application.process_update(voice_update)
    assert (await _current(core, owner, telegram_id)).session == root.session

    transcription.transcript = "Давай скорректируем систему"
    voice_start, _progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=72_004,
        source_message_id=72_104,
        progress_message_id=72_204,
    )
    await application.process_update(voice_start)
    awaiting = await _current(core, owner, telegram_id)
    assert awaiting.session is not None
    assert awaiting.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert awaiting.session.version == root.session.version + 1
    assert str(transport.edits[-1]["text"]).endswith(WEEKLY_REVIEW_QUESTION)

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            "Начать обзор недели",
            update_id=72_005,
            source_message_id=72_105,
        )
    )
    assert (await _current(core, owner, telegram_id)).session == awaiting.session
    assert str(transport.edits[-1]["text"]).endswith(WEEKLY_REVIEW_QUESTION)

    transcription.transcript = "Не знаю"
    voice_nonanswer, _progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=72_006,
        source_message_id=72_106,
        progress_message_id=72_206,
    )
    await application.process_update(voice_nonanswer)
    assert (await _current(core, owner, telegram_id)).session == awaiting.session

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            "Покажи фокус недели",
            update_id=72_007,
            source_message_id=72_107,
        )
    )
    assert (await _current(core, owner, telegram_id)).session == awaiting.session
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert transcription.calls == [
        (b"runtime-voice", "voice.ogg"),
        (b"runtime-voice", "voice.ogg"),
        (b"runtime-voice", "voice.ogg"),
    ]
    assert await _write_counts(db) == (0, 0, 0, 0, 0, 0)


async def test_real_application_shared_reducer_controls_and_processing_absorption(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 719_201
    transcription = RuntimeTranscription("Спланируем неделю")
    core, application, owner = await _application_for(
        db,
        fake_ai,
        transcription,
        telegram_id=telegram_id,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    async def send_text(value: str, index: int):
        await application.process_update(
            _runtime_weekly_text_update(
                application,
                telegram_id,
                value,
                update_id=73_000 + index,
                source_message_id=73_100 + index,
            )
        )
        result = await _current(core, owner, telegram_id)
        assert result.session is not None
        return result.session

    awaiting = await send_text("Давай скорректируем систему", 1)
    assert awaiting.phase is WeeklyReviewPhase.AWAITING_INPUT
    root_after_back = await send_text("Назад", 2)
    assert root_after_back.phase is WeeklyReviewPhase.ROOT
    assert root_after_back.version == awaiting.version + 1
    awaiting_after_edit = await send_text("Фокус на неделю", 3)
    assert awaiting_after_edit.phase is WeeklyReviewPhase.AWAITING_INPUT
    root_after_skip = await send_text("Пропустить", 4)
    assert root_after_skip.phase is WeeklyReviewPhase.ROOT
    awaiting_again = await send_text("Спланируем неделю", 5)
    assert awaiting_again.phase is WeeklyReviewPhase.AWAITING_INPUT
    completed = await send_text("Отменить", 6)
    assert completed.phase is WeeklyReviewPhase.COMPLETED

    canonical_message_id = 97_201
    transport.make_message(telegram_id, canonical_message_id, "processing")
    created = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=canonical_message_id,
        phase=WeeklyReviewPhase.PROCESSING,
    )
    assert created.session is not None
    frozen = created.session
    edit_count = len(transport.edits)
    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            "Начать обзор недели",
            update_id=73_007,
            source_message_id=73_107,
        )
    )
    assert (await _current(core, owner, telegram_id)).session == frozen
    assert len(transport.edits) == edit_count

    voice_update, _progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=73_008,
        source_message_id=73_108,
        progress_message_id=73_208,
    )
    await application.process_update(voice_update)
    assert (await _current(core, owner, telegram_id)).session == frozen
    assert len(transport.edits) == edit_count
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    assert await _write_counts(db) == (0, 0, 0, 0, 0, 0)


async def _preview_session(core, owner, telegram_id: int, canonical_message_id: int):
    created = await core.weekly_review_service.create_session(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        canonical_message_id=canonical_message_id,
        phase=WeeklyReviewPhase.AWAITING_INPUT,
    )
    assert created.session is not None
    processing = await core.weekly_review_service.mark_processing(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        session_public_id=created.session.public_id,
        expected_session_version=created.session.version,
        canonical_message_id=canonical_message_id,
    )
    assert processing.session is not None
    preview = await core.weekly_review_service.store_extraction(
        telegram_actor_id=telegram_id,
        chat_id=telegram_id,
        expected_access_version=owner.access_version,
        session_public_id=processing.session.public_id,
        expected_session_version=processing.session.version,
        canonical_message_id=canonical_message_id,
        focus="Сохранить точный preview",
        approach="Двигаться спокойно",
        small_steps=("Один небольшой шаг",),
        reminder_candidates=({"title": "Позвонить Назару", "schedule_wording": "Завтра в 15:05"},),
        source="text",
    )
    assert preview.session is not None
    return preview.session


@pytest.mark.parametrize(
    "phase",
    [
        pytest.param(WeeklyReviewPhase.PREVIEW, id="preview"),
        pytest.param(WeeklyReviewPhase.DELETE_PREVIEW, id="delete-preview"),
    ],
)
async def test_real_application_preview_and_delete_preview_voice_preserve_exact_session(
    db,
    fake_ai,
    monkeypatch,
    phase,
):
    telegram_id = 719_300 + (phase is WeeklyReviewPhase.DELETE_PREVIEW)
    canonical_message_id = 97_300 + (phase is WeeklyReviewPhase.DELETE_PREVIEW)
    transcription = RuntimeTranscription(
        "Новый длинный приватный ответ не должен менять уже показанный preview"
    )
    core, application, owner = await _application_for(
        db,
        fake_ai,
        transcription,
        telegram_id=telegram_id,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)
    before = await _preview_session(core, owner, telegram_id, canonical_message_id)
    if phase is WeeklyReviewPhase.DELETE_PREVIEW:
        confirmed = await core.weekly_review_service.confirm_focus(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
            session_public_id=before.public_id,
            expected_session_version=before.version,
            canonical_message_id=canonical_message_id,
            expected_week_start=before.week_start,
        )
        assert confirmed.session is not None
        delete_preview = await core.weekly_review_service.transition_session(
            telegram_actor_id=telegram_id,
            chat_id=telegram_id,
            expected_access_version=owner.access_version,
            session_public_id=confirmed.session.public_id,
            expected_session_version=confirmed.session.version,
            expected_canonical_message_id=canonical_message_id,
            phase=WeeklyReviewPhase.DELETE_PREVIEW,
        )
        assert delete_preview.session is not None
        before = delete_preview.session
    assert before.phase is phase
    transport.make_message(telegram_id, canonical_message_id, "canonical preview")
    counts_before = await _write_counts(db)

    async def forbidden_mark_processing(**_kwargs):
        raise AssertionError("PREVIEW/DELETE_PREVIEW voice must not mark processing")

    monkeypatch.setattr(
        core.weekly_review_service,
        "mark_processing",
        forbidden_mark_processing,
    )
    original_send: Callable = ExtBot.send_message
    original_delete: Callable = ExtBot.delete_message
    original_edit: Callable = ExtBot.edit_message_text
    io_order: list[tuple[str, int]] = []

    async def ordered_send(self, *args, **kwargs):
        message = await original_send(self, *args, **kwargs)
        io_order.append(("send", message.message_id))
        return message

    async def ordered_delete(self, *args, **kwargs):
        result = await original_delete(self, *args, **kwargs)
        io_order.append(("delete", int(kwargs["message_id"])))
        return result

    async def ordered_edit(self, *args, **kwargs):
        result = await original_edit(self, *args, **kwargs)
        io_order.append(("edit", int(kwargs["message_id"])))
        return result

    monkeypatch.setattr(ExtBot, "send_message", ordered_send)
    monkeypatch.setattr(ExtBot, "delete_message", ordered_delete)
    monkeypatch.setattr(ExtBot, "edit_message_text", ordered_edit)
    voice_update, _progress = _runtime_voice_update(
        application,
        telegram_id,
        update_id=74_000 + telegram_id,
        source_message_id=74_100,
        progress_message_id=74_200,
    )

    await application.process_update(voice_update)

    after = await _current(core, owner, telegram_id)
    assert after.status == "found"
    assert after.session == before
    assert await _write_counts(db) == counts_before
    assert fake_ai.weekly_review_calls == []
    assert fake_ai.route_calls == []
    assert transcription.calls == [(b"runtime-voice", "voice.ogg")]
    progress_message_id = transport.sent_messages[-1].message_id
    assert io_order == [
        ("send", progress_message_id),
        ("delete", progress_message_id),
        ("edit", canonical_message_id),
    ]


async def test_real_application_varvara_scheduled_nudge_two_independent_confirms(
    db,
    fake_ai,
    monkeypatch,
):
    telegram_id = 719_400
    exact_focus = "спокойно закрывать подтверждённые дела дня"
    evidence = "В 15:05 сказать Назару, что я люблю его."
    long_answer = (
        "Хочу скорректировать систему и двигаться к будущему небольшими шагами.\n"
        f"Фокус: {exact_focus}\n"
        "Подход: сохранять спокойный темп.\n"
        "Шаги:\n- закрыть одно подтверждённое дело.\n"
        f"Напоминание: {evidence}"
    )
    fake_ai.weekly_review_result = WeeklyReviewExtraction(
        focus="provider не заменяет явный фокус",
        approach="сохранять спокойный темп",
        small_steps=["закрыть одно подтверждённое дело"],
        reminder_candidates=[
            {
                "title": "сказать Назару, что я люблю его",
                "schedule_wording": "В 15:05",
                "evidence": evidence,
            }
        ],
    )
    core, application, owner = await _application_for(
        db,
        fake_ai,
        RuntimeTranscription("unused"),
        telegram_id=telegram_id,
    )
    transport = _patch_runtime_weekly_transport(monkeypatch, application)

    await core.weekly_review_scheduled_notification(
        application.bot,
        telegram_id,
        owner.timezone,
    )
    assert len(transport.sent) == 1
    assert (await _current(core, owner, telegram_id)).status == "not_found"

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            "Давай скорректируем систему",
            update_id=75_001,
            source_message_id=75_101,
        )
    )
    assert len(transport.sent) == 2
    canonical = transport.sent_messages[-1]
    question = str(transport.edits[-1]["text"])
    assert question.endswith(WEEKLY_REVIEW_QUESTION)
    assert "Что именно?" not in question
    awaiting = await _current(core, owner, telegram_id)
    assert awaiting.session is not None
    assert awaiting.session.phase is WeeklyReviewPhase.AWAITING_INPUT
    assert fake_ai.weekly_review_calls == []

    await application.process_update(
        _runtime_weekly_text_update(
            application,
            telegram_id,
            long_answer,
            update_id=75_002,
            source_message_id=75_102,
        )
    )
    preview = await _current(core, owner, telegram_id)
    assert preview.session is not None
    assert preview.session.phase is WeeklyReviewPhase.PREVIEW
    assert preview.session.focus == exact_focus
    assert [call[0] for call in fake_ai.weekly_review_calls] == [long_answer]
    assert await _write_counts(db) == (0, 0, 0, 0, 0, 0)

    telegram_user = TelegramUser(telegram_id, "Варвара", False)
    callback_ids: list[str] = []

    async def click(data: str, update_id: int) -> None:
        callback_update, query = _runtime_weekly_callback_update(
            application,
            telegram_user,
            canonical,
            data,
            update_id=update_id,
        )
        before_answers = len(transport.answers)
        await application.process_update(callback_update)
        assert len(transport.answers) == before_answers + 1
        assert transport.answers[-1]["callback_query_id"] == query.id
        callback_ids.append(query.id)

    save_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Сохранить на неделю",
        prefix="wrev:",
    )
    await click(save_data, 75_003)
    assert await _write_counts(db) == (1, 1, 0, 0, 0, 0)

    configure_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Настроить найденные (1)",
        prefix="wrev:",
    )
    await click(configure_data, 75_004)
    candidate_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Назару",
        prefix="wrev:",
    )
    await click(candidate_data, 75_005)
    assert await _write_counts(db) == (1, 1, 0, 0, 0, 0)

    tomorrow_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Завтра",
        prefix="rmd:",
    )
    await click(tomorrow_data, 75_006)
    assert await _write_counts(db) == (1, 1, 0, 0, 0, 0)
    confirm_data = _runtime_callback_for_fragment(
        transport.edits[-1]["reply_markup"],
        "Создать",
        prefix="rmd:",
    )
    await click(confirm_data, 75_007)

    assert await _write_counts(db) == (1, 1, 1, 1, 1, 0)
    assert [entry["callback_query_id"] for entry in transport.answers] == callback_ids
    assert len(transport.sent) == 2
    assert all(entry["message_id"] == canonical.message_id for entry in transport.edits[1:])
    assert fake_ai.route_calls == []
