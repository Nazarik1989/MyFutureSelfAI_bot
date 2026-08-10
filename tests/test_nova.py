from __future__ import annotations

import asyncio
import inspect
from dataclasses import fields

import pytest

from future_self.access import ADMIN, BLOCKED, GUEST, SUBSCRIBER
from future_self.navigation import help_topics, navigation_actions, navigation_sections
from future_self.nova import (
    NOVA_SESSION_TTL_SECONDS,
    NovaBackTarget,
    NovaResolutionKind,
    NovaRuntimeFlags,
    NovaSession,
    NovaSessionStore,
    build_nova_catalog,
    extract_explicit_nova_question,
    is_explicit_nova_invocation,
    is_nova_help_intent,
    resolve_nova_question,
)


def test_catalog_uses_canonical_navigation_labels_and_descriptions():
    flags = NovaRuntimeFlags(
        enable_workspace_access=True,
        enable_knowledge_hub=True,
        enable_knowledge_capture=True,
    )
    catalog = build_nova_catalog(ADMIN, flags)
    actions = navigation_actions(True, True, True)
    sections = navigation_sections(True, True, True)
    topics = help_topics(True, True, True)

    for action_id in (
        "task_create",
        "task_reminder_guide",
        "vision",
        "spaces",
        "knowledge",
        "capture",
    ):
        capability = catalog.capability(action_id)
        assert capability is not None
        assert (capability.label, capability.description) == (
            actions[action_id].label,
            actions[action_id].description,
        )

    settings = catalog.capability("section:settings")
    assert settings is not None
    assert settings.label == f"{sections['settings'].emoji} {sections['settings'].label}"
    assert settings.description == sections["settings"].description

    privacy = catalog.capability("help:privacy")
    assert privacy is not None
    assert (privacy.label, privacy.description) == topics["privacy"]


def test_catalog_respects_tier_and_runtime_flags():
    disabled = build_nova_catalog(SUBSCRIBER)
    assert disabled.capability("spaces") is None
    assert disabled.capability("knowledge") is None
    assert disabled.capability("capture") is None

    enabled = build_nova_catalog(
        ADMIN,
        NovaRuntimeFlags(
            enable_workspace_access=True,
            enable_knowledge_hub=True,
            enable_knowledge_capture=True,
            enable_vision_image_generation=True,
            enable_nova_ai=True,
        ),
    )
    assert {"workspace", "knowledge", "knowledge_capture", "nova_ai"} <= set(
        enabled.enabled_features
    )
    assert "vision_image_generation" in enabled.enabled_features

    subscriber = build_nova_catalog(
        SUBSCRIBER,
        NovaRuntimeFlags(enable_nova_ai=True, nova_ai_admin_only=True),
    )
    assert "nova_ai" not in subscriber.enabled_features

    guest = build_nova_catalog(GUEST, NovaRuntimeFlags(guest_ai_enabled=True))
    assert {item.id for item in guest.capabilities} == {
        "guest:features",
        "guest:demos",
        "guest:demo:thought",
        "guest:demo:first-step",
        "guest:access",
    }
    assert build_nova_catalog(BLOCKED).capabilities == ()

    disabled_guest = build_nova_catalog(GUEST, NovaRuntimeFlags(guest_ai_enabled=False))
    assert {item.id for item in disabled_guest.capabilities} == {
        "guest:features",
        "guest:access",
    }
    disabled_demo = resolve_nova_question("Как попробовать демо?", disabled_guest)
    assert disabled_demo is not None
    assert disabled_demo.kind is NovaResolutionKind.UNSUPPORTED
    assert disabled_demo.action_id is None


@pytest.mark.parametrize(
    ("question", "expected_action"),
    [
        ("Как открыть главное меню?", "menu"),
        ("Где раздел Сегодня?", "section:today"),
        ("Как добавить задачу с напоминанием?", "task_create"),
        ("Где мои задачи?", "section:tasks"),
        ("Где мои записи и идеи?", "inbox"),
        ("Как пройти check-in?", "checkin"),
        ("Где раздел здоровья?", "section:health"),
        ("Как найти врача?", "doctor_find"),
        ("Как подготовиться к приёму врача?", "doctor_prepare"),
        ("Где мои анализы?", "labs"),
        ("Где мои желания?", "vision"),
        ("Как открыть визуализацию?", "vision"),
        ("Как загрузить референс?", "vision"),
        ("Как собрать текущую локальную PNG-карту?", "vision"),
        ("Где мой профиль?", "profile"),
        ("Как изменить часовой пояс?", "timezone"),
        ("Где изменить локацию?", "location"),
        ("Как открыть настройки?", "section:settings"),
        ("Где мои разделы?", "collections"),
        ("Как пользоваться этим ботом?", "help:quick"),
        ("Что умеет бот?", "help:requests"),
        ("Покажи примеры вопросов", "help:examples"),
        ("Где справка про данные и безопасность?", "help:privacy"),
        ("Что делать, если бот не понял?", "help:troubleshooting"),
    ],
)
def test_required_known_questions_resolve_locally(question, expected_action):
    catalog = build_nova_catalog(SUBSCRIBER)

    result = resolve_nova_question(question, catalog)

    assert result is not None
    assert result.kind is NovaResolutionKind.GUIDE
    assert result.action_id == expected_action
    expected_cta = (
        "🎯 Открыть визуализацию"
        if expected_action == "vision"
        else catalog.capability(expected_action).label
    )
    assert result.cta_label == expected_cta
    assert 1 <= len(result.steps) <= 3


@pytest.mark.parametrize(
    ("question", "expected_fragment"),
    [
        (
            "Как в боте создать ежедневное напоминание?",
            "создаётся обычной фразой",
        ),
        (
            "Как в боте изменить время ежедневного напоминания?",
            "меняется в карточке",
        ),
        (
            "Как в боте отключить ежедневное напоминание?",
            "сама задача остаётся",
        ),
    ],
)
def test_daily_reminder_help_stays_local_when_nova_ai_is_enabled(
    question,
    expected_fragment,
):
    catalog = build_nova_catalog(
        SUBSCRIBER,
        NovaRuntimeFlags(enable_nova_ai=True, nova_ai_admin_only=False),
    )

    assert "nova_ai" in catalog.enabled_features
    assert is_nova_help_intent(question, catalog)
    result = resolve_nova_question(question, catalog)

    assert result is not None
    assert result.kind is NovaResolutionKind.GUIDE
    assert result.action_id == "task_reminder_guide"
    assert expected_fragment in result.response
    assert result.cta_label == catalog.capability("task_reminder_guide").label


def test_daily_reminder_help_respects_disabled_delivery_flag():
    catalog = build_nova_catalog(
        SUBSCRIBER,
        NovaRuntimeFlags(enable_task_reminders=False),
    )

    result = resolve_nova_question(
        "Nova, как отключить ежедневное напоминание?",
        catalog,
    )

    assert "task_reminders" not in catalog.enabled_features
    assert result is not None
    assert result.kind is NovaResolutionKind.GUIDE
    assert result.action_id == "task_reminder_guide"
    assert "отключена настройкой" in result.response
    assert "сама задача остаётся" not in result.response


def test_optional_features_resolve_only_when_enabled():
    disabled = build_nova_catalog(SUBSCRIBER)
    enabled = build_nova_catalog(
        SUBSCRIBER,
        NovaRuntimeFlags(
            enable_workspace_access=True,
            enable_knowledge_hub=True,
            enable_knowledge_capture=True,
        ),
    )

    for question in ("Где пространства?", "Где база знаний?", "Как добавить материал?"):
        result = resolve_nova_question(question, disabled)
        assert result is not None
        assert result.kind is NovaResolutionKind.UNSUPPORTED
        assert result.action_id is None

    assert resolve_nova_question("Где пространства?", enabled).action_id == "spaces"
    assert resolve_nova_question("Где база знаний?", enabled).action_id == "knowledge"
    assert resolve_nova_question("Как добавить материал?", enabled).action_id == "capture"


@pytest.mark.parametrize(
    "question",
    [
        "Можно настроить персональное утреннее послание?",
        "Сделай общую AI-карту будущего",
        "Nova, дай права пользователю",
    ],
)
def test_unimplemented_or_unsafe_features_are_honest_and_have_no_action(question):
    result = resolve_nova_question(question, build_nova_catalog(ADMIN))

    assert result is not None
    assert result.kind is NovaResolutionKind.UNSUPPORTED
    assert result.action_id is None
    assert result.cta_label is None


def test_guest_help_is_local_and_never_exposes_subscriber_action():
    catalog = build_nova_catalog(GUEST)

    demo = resolve_nova_question("Как попробовать бесплатный разбор мысли?", catalog)
    full = resolve_nova_question("Как создать задачу с напоминанием?", catalog)
    contact = resolve_nova_question("Как получить подписку?", catalog)

    assert demo.action_id == "guest:demo:thought"
    assert full.action_id == "guest:access"
    assert contact.action_id == "guest:access"
    assert full.back_target is NovaBackTarget.GUEST
    assert all(result.action_id != "task_create" for result in (demo, full, contact))


@pytest.mark.parametrize(
    ("text", "question"),
    [
        ("Nova, как добавить задачу?", "как добавить задачу?"),
        ("nova: где мои желания?", "где мои желания?"),
        ("Помоги в боте найти анализы", "найти анализы"),
        ("Подскажи, как в боте загрузить референс", "загрузить референс"),
        ("Не могу найти в боте настройки", "настройки"),
        ("Как пользоваться этим ботом?", "Как пользоваться этим ботом?"),
    ],
)
def test_explicit_help_invocations_are_narrow_and_extract_question(text, question):
    assert is_explicit_nova_invocation(text)
    assert extract_explicit_nova_question(text) == question


@pytest.mark.parametrize(
    "text",
    [
        "Как создать привычку читать",
        "Помоги мне сформулировать мысль",
        "Открой окно и проветри комнату",
        "Как найти время на спорт",
        "Покажи презентацию клиенту",
        "Где поставить коробки после переезда",
        "Хочу понять, как улучшить отношения",
        "Нова, где задачи?",
    ],
)
def test_normal_content_is_not_captured_as_explicit_nova_help(text):
    assert not is_explicit_nova_invocation(text)
    assert extract_explicit_nova_question(text) is None


def test_oversized_explicit_invocation_is_still_captured_for_local_validation():
    oversized = "Nova, " + "x" * 601

    assert is_explicit_nova_invocation(oversized)
    assert extract_explicit_nova_question(oversized) == "x" * 601
    result = resolve_nova_question(oversized, build_nova_catalog(ADMIN))
    assert result is not None
    assert result.kind is NovaResolutionKind.CLARIFY
    assert result.action_id is None


@pytest.mark.parametrize(
    ("text", "expected_action"),
    [
        ("Привет, где у тебя находится визуализация?", "vision"),
        (
            "Просто я знаю, что в этом боте есть визуализация, но не могу её найти "
            "в менюшке. Подскажи, пожалуйста.",
            "vision",
        ),
        ("Не могу найти референсы в меню, подскажи, пожалуйста", "vision"),
        ("Подскажи, где мои задачи", "section:tasks"),
        ("Где у тебя находятся напоминания?", "task_reminder_guide"),
        ("Как в боте настроить часовой пояс?", "timezone"),
        ("Не могу найти настройки в меню, подскажи", "section:settings"),
        ("Привет, где у тебя находятся желания?", "vision"),
        ("Как в боте открыть раздел Сегодня?", "section:today"),
        ("Подскажи, где мои записи", "inbox"),
    ],
)
def test_shared_help_intent_recognizes_natural_known_capability_questions(
    text,
    expected_action,
):
    catalog = build_nova_catalog(SUBSCRIBER)

    assert is_nova_help_intent(text, catalog)
    resolution = resolve_nova_question(text, catalog)
    assert resolution is not None
    assert resolution.kind is NovaResolutionKind.GUIDE
    assert resolution.action_id == expected_action


def test_shared_help_intent_accepts_explicit_nova_and_active_session():
    catalog = build_nova_catalog(SUBSCRIBER)

    assert is_nova_help_intent("Nova, где находится неизвестная кнопка?", catalog)
    assert is_nova_help_intent(
        "Ладно, объясни",
        catalog,
        active_session=True,
    )


def test_shared_help_intent_requires_capability_in_current_catalog():
    disabled = build_nova_catalog(SUBSCRIBER)
    enabled = build_nova_catalog(
        SUBSCRIBER,
        NovaRuntimeFlags(enable_knowledge_hub=True),
    )

    question = "Подскажи, где в боте находится база знаний?"
    assert not is_nova_help_intent(question, disabled)
    assert is_nova_help_intent(question, enabled)


@pytest.mark.parametrize(
    "text",
    [
        "Сегодня я размышлял о визуализации будущего",
        "Хочу записать идею для новой карты",
        "Мне важно понять, где я вижу себя через год",
        (
            "Сегодня я долго записывал обычную мысль о том, как мои задачи, здоровье "
            "и визуализация будущего связаны с важными для меня переменами."
        ),
        "Не могу найти время на задачу",
        "Как найти время на задачу?",
        "Где находится задача, которую я обещал сделать?",
        "Где находится здоровье человека?",
        "Как открыть референс в Photoshop?",
        "Как в Photoshop открыть раздел референсов?",
        "Как в меню Photoshop открыть референсы?",
        "Подскажи, как открыть раздел задачи в учебнике?",
        "Как открыть раздел здоровья в презентации?",
        "Как пользоваться функцией задач в Excel?",
        "Ладно, объясни",
        "Расскажи подробнее",
        "Как это работает?",
        "Покажи, где это",
        "Напоминай каждый день в 19:30 позвонить маме",
    ],
)
def test_shared_help_intent_does_not_capture_content_or_contextless_follow_up(text):
    assert not is_nova_help_intent(text, build_nova_catalog(SUBSCRIBER))


@pytest.mark.parametrize(
    "text",
    [
        "НЕ МОГУ ЕЁ НАЙТИ!!! ВИЗУАЛИЗАЦИЮ В МЕНЮ???",
        "не могу ее найти, визуализацию в меню",
    ],
)
def test_shared_help_intent_normalizes_case_yo_and_punctuation(text):
    catalog = build_nova_catalog(SUBSCRIBER)

    assert is_nova_help_intent(text, catalog)
    resolution = resolve_nova_question(text, catalog)
    assert resolution is not None
    assert resolution.action_id == "vision"


def test_shared_help_intent_keeps_oversized_known_question_in_nova_for_local_clarify():
    catalog = build_nova_catalog(SUBSCRIBER)
    text = "Где у тебя находится визуализация в меню? " + "пожалуйста " * 70

    assert len(text) > 600
    assert is_nova_help_intent(text, catalog)
    resolution = resolve_nova_question(text, catalog)
    assert resolution is not None
    assert resolution.kind is NovaResolutionKind.CLARIFY
    assert resolution.action_id is None


@pytest.mark.parametrize(
    "follow_up",
    [
        "Ладно, объясни",
        "Расскажи подробнее",
        "Как это работает?",
        "Покажи, где это",
    ],
)
def test_follow_up_resolves_only_from_safe_last_action_context(follow_up):
    catalog = build_nova_catalog(SUBSCRIBER)

    contextless = resolve_nova_question(follow_up, catalog)
    assert contextless is not None
    assert contextless.kind is NovaResolutionKind.CLARIFY
    assert contextless.action_id is None
    assert is_nova_help_intent(
        follow_up,
        catalog,
        active_session=True,
        last_action_id="vision",
    )
    resolution = resolve_nova_question(
        follow_up,
        catalog,
        last_action_id="vision",
    )
    assert resolution is not None
    assert resolution.kind is NovaResolutionKind.GUIDE
    assert resolution.action_id == "vision"
    assert resolution.cta_label == "🎯 Открыть визуализацию"


def _binding(**overrides):
    values = {
        "owner_id": 7,
        "telegram_user_id": 1001,
        "chat_id": 2002,
        "access_version": 4,
        "canonical_message_id": 55,
        "tier": ADMIN,
    }
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_session_has_fifteen_minute_ttl_and_stores_no_question_or_response():
    now = 100.0
    store = NovaSessionStore(clock=lambda: now)

    session = await store.create(**_binding())

    assert session.expires_at - session.created_at == NOVA_SESSION_TTL_SECONDS
    assert {field.name for field in fields(NovaSession)} == {
        "id",
        "owner_id",
        "telegram_user_id",
        "chat_id",
        "access_version",
        "canonical_message_id",
        "tier",
        "created_at",
        "expires_at",
        "question_in_progress",
        "last_action_id",
    }
    assert "question" not in inspect.signature(store.create).parameters
    assert "transcript" not in inspect.signature(store.create).parameters
    assert "response" not in inspect.signature(store.create).parameters
    assert session.last_action_id is None


@pytest.mark.asyncio
async def test_last_action_context_is_fenced_and_contains_no_raw_content():
    raw_transcript = "Привет, где у тебя находится визуализация? PRIVATE_TRANSCRIPT_SENTINEL"
    catalog = build_nova_catalog(SUBSCRIBER)
    assert is_nova_help_intent(raw_transcript, catalog)
    resolution = resolve_nova_question(raw_transcript, catalog)
    assert resolution is not None
    assert resolution.action_id == "vision"
    store = NovaSessionStore()
    session = await store.create(**_binding())
    assert await store.begin_question(session_id=session.id, **_binding()) is not None

    remembered = await store.remember_action(
        session_id=session.id,
        last_action_id=resolution.action_id,
        **_binding(),
    )

    assert remembered is not None
    assert remembered.last_action_id == "vision"
    current = await store.current(owner_id=7, telegram_user_id=1001, chat_id=2002)
    assert current is not None
    assert current.last_action_id == "vision"
    assert raw_transcript not in repr(current)
    assert raw_transcript not in repr(store._sessions)
    assert "transcript" not in inspect.signature(store.remember_action).parameters
    assert "question" not in inspect.signature(store.remember_action).parameters
    assert "response" not in inspect.signature(store.remember_action).parameters

    assert (
        await store.remember_action(
            session_id="wrong-session",
            last_action_id="timezone",
            **_binding(),
        )
        is None
    )
    unchanged = await store.current(owner_id=7, telegram_user_id=1001, chat_id=2002)
    assert unchanged is not None
    assert unchanged.last_action_id == "vision"

    assert (
        await store.remember_action(
            session_id=session.id,
            last_action_id="vision",
            **_binding(access_version=5),
        )
        is None
    )
    assert await store.current(owner_id=7, telegram_user_id=1001, chat_id=2002) is None


@pytest.mark.asyncio
async def test_replacing_or_clearing_session_removes_last_action_context():
    store = NovaSessionStore()
    first = await store.create(**_binding())
    assert await store.begin_question(session_id=first.id, **_binding()) is not None
    assert (
        await store.remember_action(
            session_id=first.id,
            last_action_id="vision",
            **_binding(),
        )
        is not None
    )

    replacement = await store.create(**_binding(canonical_message_id=56))

    assert replacement.id != first.id
    assert replacement.last_action_id is None
    assert await store.clear(owner_id=7, chat_id=2002, session_id=replacement.id)
    assert await store.current(owner_id=7, telegram_user_id=1001, chat_id=2002) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_action_id",
    [
        "",
        "Visual action",
        "vision?private-question",
        "PRIVATE_TRANSCRIPT_SENTINEL где визуализация",
        "x" * 101,
    ],
)
async def test_last_action_context_rejects_raw_or_unsafe_identifiers(unsafe_action_id):
    store = NovaSessionStore()
    session = await store.create(**_binding())
    assert await store.begin_question(session_id=session.id, **_binding()) is not None

    with pytest.raises(ValueError, match="action"):
        await store.remember_action(
            session_id=session.id,
            last_action_id=unsafe_action_id,
            **_binding(),
        )

    current = await store.current(owner_id=7, telegram_user_id=1001, chat_id=2002)
    assert current is not None
    assert current.last_action_id is None


@pytest.mark.asyncio
async def test_new_owner_chat_session_replaces_old_session_and_actions():
    store = NovaSessionStore()
    old = await store.create(**_binding())
    old_token = await store.issue_action(action_id="vision", session_id=old.id, **_binding())

    new = await store.create(**_binding(canonical_message_id=56))

    assert new.id != old.id
    assert await store.get(session_id=old.id, **_binding(canonical_message_id=56)) is None
    assert await store.consume_action(token=old_token, **_binding()) is None
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_action_token_is_random_owner_chat_bound_and_single_use():
    store = NovaSessionStore()
    session = await store.create(**_binding())
    first = await store.issue_action(action_id="task_create", session_id=session.id, **_binding())
    second = await store.issue_action(action_id="vision", session_id=session.id, **_binding())

    assert first and second and first != second
    assert len(first) >= 20
    assert (
        await store.consume_action(
            token=first,
            **_binding(owner_id=8, telegram_user_id=3003, chat_id=4004),
        )
        is None
    )

    claim = await store.consume_action(token=first, **_binding())
    assert claim is not None
    assert claim.action_id == "task_create"
    assert await store.consume_action(token=first, **_binding()) is None


@pytest.mark.asyncio
async def test_forged_action_id_does_not_burn_valid_token():
    store = NovaSessionStore()
    session = await store.create(**_binding())
    token = await store.issue_action(action_id="task_create", session_id=session.id, **_binding())

    assert (
        await store.consume_action(
            token=token,
            expected_action_id="vision",
            **_binding(),
        )
        is None
    )
    claim = await store.consume_action(
        token=token,
        expected_action_id="task_create",
        **_binding(),
    )
    assert claim is not None
    assert claim.action_id == "task_create"


@pytest.mark.asyncio
async def test_access_version_or_tier_mismatch_clears_session_and_actions():
    store = NovaSessionStore()
    session = await store.create(**_binding())
    token = await store.issue_action(action_id="vision", session_id=session.id, **_binding())

    assert await store.get(**_binding(access_version=5)) is None
    assert await store.current(owner_id=7, telegram_user_id=1001, chat_id=2002) is None
    assert await store.consume_action(token=token, **_binding()) is None

    await store.create(**_binding())
    assert await store.get(**_binding(tier=SUBSCRIBER)) is None
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_expired_session_and_action_are_pruned():
    now = [10.0]
    store = NovaSessionStore(ttl_seconds=5, clock=lambda: now[0])
    session = await store.create(**_binding())
    assert await store.begin_question(session_id=session.id, **_binding()) is not None
    assert (
        await store.remember_action(
            session_id=session.id,
            last_action_id="vision",
            **_binding(),
        )
        is not None
    )
    assert await store.finish_question(session_id=session.id, **_binding())
    token = await store.issue_action(action_id="vision", session_id=session.id, **_binding())

    now[0] = 15.0

    assert await store.current(owner_id=7, telegram_user_id=1001, chat_id=2002) is None
    assert await store.consume_action(token=token, **_binding()) is None
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_rebind_canonical_preserves_token_for_only_the_new_message():
    store = NovaSessionStore()
    session = await store.create(**_binding())
    token = await store.issue_action(action_id="vision", session_id=session.id, **_binding())

    rebound = await store.rebind_canonical(
        owner_id=7,
        telegram_user_id=1001,
        chat_id=2002,
        access_version=4,
        tier=ADMIN,
        expected_message_id=55,
        new_message_id=56,
        session_id=session.id,
    )

    assert rebound is not None
    assert rebound.canonical_message_id == 56
    assert await store.consume_action(token=token, **_binding()) is None
    claim = await store.consume_action(token=token, **_binding(canonical_message_id=56))
    assert claim is not None
    assert claim.action_id == "vision"
    assert await store.consume_action(token=token, **_binding(canonical_message_id=56)) is None


@pytest.mark.asyncio
async def test_session_can_be_reserved_then_bound_after_telegram_returns_message():
    store = NovaSessionStore()

    reserved = await store.reserve(
        owner_id=7,
        telegram_user_id=1001,
        chat_id=2002,
        access_version=4,
        tier=ADMIN,
    )
    assert reserved.canonical_message_id is None
    assert (
        await store.issue_action(
            action_id="vision",
            session_id=reserved.id,
            **_binding(),
        )
        is None
    )

    bound = await store.bind_canonical(
        owner_id=7,
        telegram_user_id=1001,
        chat_id=2002,
        access_version=4,
        tier=ADMIN,
        new_message_id=55,
        session_id=reserved.id,
    )
    assert bound is not None
    assert bound.id == reserved.id
    assert bound.canonical_message_id == 55


@pytest.mark.asyncio
async def test_duplicate_concurrent_question_has_one_winner():
    store = NovaSessionStore()
    session = await store.create(**_binding())
    old_token = await store.issue_action(action_id="vision", session_id=session.id, **_binding())

    attempts = await asyncio.gather(
        *(store.begin_question(session_id=session.id, **_binding()) for _ in range(20))
    )

    winners = [result for result in attempts if result is not None]
    assert len(winners) == 1
    assert winners[0].question_in_progress
    assert await store.consume_action(token=old_token, **_binding()) is None
    assert await store.finish_question(session_id=session.id, **_binding())
    assert await store.begin_question(session_id=session.id, **_binding()) is not None


@pytest.mark.asyncio
async def test_store_is_bounded_and_evicts_oldest_session():
    store = NovaSessionStore(max_sessions=2)
    first = await store.create(**_binding(owner_id=1, chat_id=11, telegram_user_id=10))
    await store.create(**_binding(owner_id=2, chat_id=22, telegram_user_id=20))
    await store.create(**_binding(owner_id=3, chat_id=33, telegram_user_id=30))

    assert await store.count() == 2
    assert (
        await store.get(
            **_binding(owner_id=1, chat_id=11, telegram_user_id=10, session_id=first.id)
        )
        is None
    )


@pytest.mark.asyncio
async def test_session_identifiers_are_positive_but_private_chat_is_bound_separately():
    store = NovaSessionStore()

    with pytest.raises(ValueError, match="positive"):
        await store.create(**_binding(chat_id=0))

    session = await store.create(**_binding(telegram_user_id=101, chat_id=202))
    assert session.telegram_user_id == 101
    assert session.chat_id == 202
