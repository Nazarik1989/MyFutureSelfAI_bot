import asyncio
import inspect
from dataclasses import fields

import pytest

from future_self.access import ADMIN, BLOCKED, SUBSCRIBER
from future_self.nova_memory import NovaMemoryValidationError
from future_self.nova_memory_flow import (
    NOVA_MEMORY_CALLBACK_PREFIX,
    NOVA_MEMORY_FLOW_MAX_SESSIONS,
    NOVA_MEMORY_FLOW_TTL_SECONDS,
    NovaMemoryFlowPhase,
    NovaMemoryFlowSession,
    NovaMemoryFlowStore,
    NovaMemoryIntentClassifier,
    NovaMemoryIntentKind,
    classify_nova_memory_intent,
)


def _binding(**overrides):
    values = {
        "owner_id": 7,
        "telegram_user_id": 1001,
        "chat_id": 1001,
        "tier": ADMIN,
        "access_version": 4,
        "canonical_message_id": 55,
    }
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_flow_defaults_are_bounded_fifteen_minute_process_local_state():
    now = 100.0
    store = NovaMemoryFlowStore(clock=lambda: now)

    session = await store.create(**_binding())

    assert store.max_sessions == NOVA_MEMORY_FLOW_MAX_SESSIONS == 128
    assert session.expires_at - session.created_at == NOVA_MEMORY_FLOW_TTL_SECONDS == 900
    assert session.version == 1
    assert session.phase is NovaMemoryFlowPhase.ROOT
    assert session.canonical_message_id == 55
    with pytest.raises(ValueError, match="128"):
        NovaMemoryFlowStore(max_sessions=129)
    with pytest.raises(ValueError, match="15 minutes"):
        NovaMemoryFlowStore(ttl_seconds=901)


@pytest.mark.asyncio
async def test_candidate_is_normalized_temporary_and_excluded_from_all_reprs():
    sentinel = "PRIVATE_MEMORY_SENTINEL"
    store = NovaMemoryFlowStore()
    session = await store.create(**_binding())

    preview = await store.update(
        session,
        phase=NovaMemoryFlowPhase.CREATE_PREVIEW,
        candidate_content=f"  {sentinel}\n  likes   tea  ",
        candidate_category="about_me",
        candidate_important=True,
    )

    assert preview is not None
    assert preview.candidate_content == f"{sentinel} likes tea"
    assert sentinel not in repr(preview)
    assert sentinel not in repr(store._sessions)
    assert "audio" not in inspect.signature(store.create).parameters
    assert "transcript" not in inspect.signature(store.create).parameters
    assert "provider" not in inspect.signature(store.create).parameters
    assert "command" not in inspect.signature(store.create).parameters

    root = await store.update(preview, phase=NovaMemoryFlowPhase.ROOT)
    assert root is not None
    assert root.candidate_content is None
    assert root.candidate_category is None
    assert not root.candidate_important


@pytest.mark.asyncio
async def test_owner_chat_key_replaces_only_the_exact_flow_and_evicts_oldest():
    store = NovaMemoryFlowStore(max_sessions=2)
    first = await store.create(
        **_binding(owner_id=1, telegram_user_id=11, chat_id=101, canonical_message_id=1)
    )
    replacement = await store.create(
        **_binding(owner_id=1, telegram_user_id=12, chat_id=101, canonical_message_id=2)
    )
    assert replacement.id != first.id
    assert await store.current(owner_id=1, telegram_user_id=11, chat_id=101) is None

    await store.create(
        **_binding(owner_id=2, telegram_user_id=22, chat_id=202, canonical_message_id=3)
    )
    await store.create(
        **_binding(owner_id=3, telegram_user_id=33, chat_id=303, canonical_message_id=4)
    )

    assert await store.count() == 2
    assert await store.current(owner_id=1, telegram_user_id=12, chat_id=101) is None


@pytest.mark.asyncio
async def test_guest_or_blocked_cannot_create_and_access_generation_mismatch_clears():
    store = NovaMemoryFlowStore()
    with pytest.raises(ValueError, match="full access"):
        await store.create(**_binding(tier=BLOCKED))

    session = await store.create(**_binding(tier=SUBSCRIBER))
    assert (
        await store.get(
            **_binding(tier=SUBSCRIBER),
            session_id=session.id,
            session_version=session.version,
        )
        == session
    )
    assert await store.get(**_binding(tier=SUBSCRIBER, access_version=5)) is None
    assert await store.current(owner_id=7, telegram_user_id=1001, chat_id=1001) is None


@pytest.mark.asyncio
async def test_reserve_then_bind_increments_generation_and_requires_exact_snapshot():
    store = NovaMemoryFlowStore()
    reserved = await store.reserve(
        owner_id=7,
        telegram_user_id=1001,
        chat_id=1001,
        tier=ADMIN,
        access_version=4,
    )

    assert reserved.canonical_message_id is None
    assert await store.issue(reserved, action="root") is None

    bound = await store.bind_canonical(reserved, canonical_message_id=55)
    assert bound is not None
    assert bound.version == reserved.version + 1
    assert bound.canonical_message_id == 55
    assert await store.get_exact(reserved) is None
    assert await store.get_exact(bound) == bound


@pytest.mark.asyncio
async def test_capability_is_opaque_short_and_binds_only_safe_fences():
    store = NovaMemoryFlowStore()
    session = await store.create(**_binding())
    callback_data = await store.issue(
        session,
        action="item",
        public_id="private:item:id",
        expected_item_version=3,
        list_filter="important",
        page=2,
    )

    assert callback_data is not None
    assert callback_data.startswith(NOVA_MEMORY_CALLBACK_PREFIX)
    assert len(callback_data.encode("utf-8")) <= 64
    assert callback_data.count(":") == 1
    assert "private:item:id" not in callback_data

    claim = await store.claim(callback_data, **_binding())
    assert claim is not None
    assert claim.capability.action == "item"
    assert claim.capability.public_id == "private:item:id"
    assert claim.capability.expected_item_version == 3
    assert claim.capability.list_filter == "important"
    assert claim.capability.page == 2


@pytest.mark.asyncio
async def test_screen_transition_invalidates_every_capability_from_previous_version():
    store = NovaMemoryFlowStore()
    session = await store.create(**_binding())
    first = await store.issue(session, action="create")
    second = await store.issue(session, action="list_about")
    assert first and second

    moved = await store.update(
        session,
        phase=NovaMemoryFlowPhase.LIST,
        list_filter="about_me",
        page=0,
    )

    assert moved is not None
    assert moved.version == session.version + 1
    assert await store.claim(first, **_binding()) is None
    assert await store.claim(second, **_binding()) is None


@pytest.mark.asyncio
async def test_wrong_owner_user_chat_or_canonical_does_not_consume_mutation():
    store = NovaMemoryFlowStore()
    session = await store.create(**_binding())
    callback_data = await store.issue(
        session,
        action="confirm_delete",
        mutation=True,
        public_id="item-1",
        expected_item_version=8,
    )
    assert callback_data is not None

    assert (
        await store.claim(
            callback_data,
            **_binding(owner_id=8, telegram_user_id=2002, chat_id=2002),
        )
        is None
    )
    assert await store.claim(callback_data, **_binding(telegram_user_id=2002)) is None
    assert await store.claim(callback_data, **_binding(chat_id=2002)) is None
    assert await store.claim(callback_data, **_binding(canonical_message_id=56)) is None
    assert (
        await store.claim(
            callback_data,
            expected_action="confirm_create",
            **_binding(),
        )
        is None
    )

    claim = await store.claim(
        callback_data,
        expected_action="confirm_delete",
        **_binding(),
    )
    assert claim is not None
    assert claim.session.phase is NovaMemoryFlowPhase.PROCESSING
    assert claim.capability.public_id == "item-1"


@pytest.mark.asyncio
async def test_mutation_capability_has_one_atomic_winner_and_burns_entire_screen():
    store = NovaMemoryFlowStore()
    session = await store.create(**_binding())
    mutation = await store.issue(session, action="confirm_create", mutation=True)
    sibling = await store.issue(session, action="cancel")
    assert mutation and sibling

    attempts = await asyncio.gather(*(store.claim(mutation, **_binding()) for _ in range(30)))

    winners = [claim for claim in attempts if claim is not None]
    assert len(winners) == 1
    assert winners[0].session.phase is NovaMemoryFlowPhase.PROCESSING
    assert winners[0].session.version == session.version + 1
    assert await store.claim(mutation, **_binding()) is None
    assert await store.claim(sibling, **_binding()) is None


@pytest.mark.asyncio
async def test_non_mutation_claim_waits_for_winning_transition_to_invalidate_screen():
    store = NovaMemoryFlowStore()
    session = await store.create(**_binding())
    callback_data = await store.issue(session, action="help")
    assert callback_data is not None

    first = await store.claim(callback_data, **_binding())
    second = await store.claim(callback_data, **_binding())
    assert first is not None and second is not None

    transitioned = await store.update(first.session, phase=NovaMemoryFlowPhase.ROOT)
    lost_race = await store.update(second.session, phase=NovaMemoryFlowPhase.ROOT)
    assert transitioned is not None
    assert lost_race is None
    assert await store.claim(callback_data, **_binding()) is None


@pytest.mark.asyncio
async def test_delete_all_capability_keeps_exact_revision_out_of_callback_and_repr():
    revision = "a" * 64
    store = NovaMemoryFlowStore()
    session = await store.create(**_binding())
    preview = await store.update(
        session,
        phase=NovaMemoryFlowPhase.DELETE_ALL_PREVIEW,
        collection_revision=revision,
        collection_count=5,
    )
    assert preview is not None
    callback_data = await store.issue(
        preview,
        action="confirm_delete_all",
        mutation=True,
        expected_collection_revision=revision,
    )
    assert callback_data is not None
    assert revision not in callback_data
    assert revision not in repr(preview)
    assert revision not in repr(store._capabilities)

    claim = await store.claim(callback_data, **_binding())
    assert claim is not None
    assert claim.capability.expected_collection_revision == revision


@pytest.mark.asyncio
async def test_expiry_and_restart_make_sessions_and_callbacks_stale():
    now = [10.0]
    store = NovaMemoryFlowStore(ttl_seconds=5, clock=lambda: now[0])
    session = await store.create(**_binding())
    callback_data = await store.issue(session, action="root")
    assert callback_data is not None

    restarted = NovaMemoryFlowStore()
    assert await restarted.claim(callback_data, **_binding()) is None

    now[0] = 15.0
    assert await store.current(owner_id=7, telegram_user_id=1001, chat_id=1001) is None
    assert await store.claim(callback_data, **_binding()) is None
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_clear_exact_is_generation_cas_and_preserves_newer_same_id_session():
    store = NovaMemoryFlowStore()
    original = await store.create(**_binding())
    newer = await store.update(original, phase=NovaMemoryFlowPhase.CREATE_PREVIEW)
    assert newer is not None
    assert newer.id == original.id and newer.version == original.version + 1

    assert await store.clear_exact(original) is False
    assert await store.get_exact(newer) == newer
    assert await store.clear_exact(newer) is True
    assert await store.current(owner_id=7, telegram_user_id=1001, chat_id=1001) is None


@pytest.mark.parametrize(
    ("text", "category", "important", "content"),
    [
        ("Nova, запомни: я люблю чай", "about_me", False, "я люблю чай"),
        ("Нова, запомни: я люблю чай", "about_me", False, "я люблю чай"),
        ("Научи Nova: отвечай кратко", "about_me", False, "отвечай кратко"),
        ("Запомни для Nova: я сова", "about_me", False, "я сова"),
        ("Nova, запомни обо мне: я сова", "about_me", False, "я сова"),
        (
            "Nova, запомни, как со мной работать: без спешки",
            "interaction",
            False,
            "без спешки",
        ),
        (
            "Nova, запомни мой ориентир: семья",
            "orientation",
            False,
            "семья",
        ),
        ("Nova, сохрани в важное: звонить маме", "about_me", True, "звонить маме"),
        ("Нова, запомни мой ориентир: честность", "orientation", False, "честность"),
        ("Нова, сохрани в важное: семья", "about_me", True, "семья"),
    ],
)
def test_classifier_accepts_only_exact_create_prefixes(text, category, important, content):
    result = classify_nova_memory_intent(text)

    assert result.kind is NovaMemoryIntentKind.CREATE
    assert result.category == category
    assert result.important is important
    assert result.content == content
    assert content not in repr(result)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Nova, запомни", NovaMemoryIntentKind.AWAIT_CONTENT),
        ("Nova, запомни это", NovaMemoryIntentKind.REMEMBER_THIS),
        ("Сохрани это для Nova", NovaMemoryIntentKind.REMEMBER_THIS),
        ("Nova, забудь всё", NovaMemoryIntentKind.DELETE_ALL),
        ("Открой мою Nova", NovaMemoryIntentKind.OPEN),
        ("Что Nova помнит обо мне?", NovaMemoryIntentKind.OPEN),
        ("Покажи память Nova", NovaMemoryIntentKind.OPEN),
    ],
)
def test_classifier_returns_typed_control_intents(text, kind):
    result = NovaMemoryIntentClassifier().classify(text)
    assert result.kind is kind
    assert result.content is None


@pytest.mark.parametrize(
    "text",
    [
        "Я хочу запомнить этот день",
        "Как запомнить английские слова?",
        "Напомни завтра купить чай",
        "Не забудь купить молоко",
        "Эта песня напомнила школу",
        "Он сказал: Nova, запомни: это цитата",
        "Сегодня важное событие",
        "Мой ориентир изменился",
        "Расскажи, что такое память",
        "Nova, запомни пожалуйста: слишком широкая фраза",
        "Nova, запомни это: лишний текст",
        "Открой мою Nova и покажи задачи",
    ],
)
def test_classifier_fails_closed_for_conversation_reminders_quotes_and_near_matches(text):
    assert classify_nova_memory_intent(text).kind is NovaMemoryIntentKind.NONE


def test_classifier_normalizes_candidate_and_rejects_oversized_explicit_content():
    result = classify_nova_memory_intent("  NOVA,   ЗАПОМНИ:  люблю\nзелёный   чай ")
    assert result.kind is NovaMemoryIntentKind.CREATE
    assert result.content == "люблю зелёный чай"

    with pytest.raises(NovaMemoryValidationError, match="too long"):
        classify_nova_memory_intent(f"Nova, запомни: {'я' * 501}")


def test_session_schema_has_no_raw_media_or_provider_fields():
    names = {item.name for item in fields(NovaMemoryFlowSession)}
    assert "candidate_content" in names
    assert not names & {
        "audio",
        "audio_bytes",
        "raw_command",
        "transcript",
        "provider_output",
    }
