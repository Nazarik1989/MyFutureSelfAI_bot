import asyncio
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from sqlalchemy import func, select, update

from future_self.conversation import (
    CompanionConversationFence,
    ConversationContextService,
    companion_conversation_revision,
)
from future_self.models import NovaDialogueState, NovaObservedMemory, User
from future_self.nova_brain import (
    NovaBrainFence,
    NovaBrainForgetStore,
    NovaBrainPolicy,
    NovaBrainProjection,
    NovaBrainService,
    NovaDialogueStateView,
    NovaObservedMemoryView,
    validate_dialogue_state_update,
    validate_memory_candidate,
)
from future_self.schemas import NovaCompanionDialogueStateUpdate, NovaCompanionMemoryCandidate


async def _actor(db, telegram_id: int, *, tier: str = "subscriber") -> User:
    async with db.session() as session:
        actor = User(
            telegram_id=telegram_id,
            display_name="Лена",
            timezone="Europe/Moscow",
            onboarding_completed=True,
            access_tier=tier,
            access_version=1,
        )
        session.add(actor)
        await session.flush()
        return actor


def _conversation_fence(actor: User, chat_id: int) -> CompanionConversationFence:
    context: dict[str, object] = {}
    return CompanionConversationFence(
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=chat_id,
        access_version=actor.access_version,
        access_tier=actor.access_tier,
        raw_message_limit=20,
        conversation_payload_max_bytes=32 * 1024,
        revision=companion_conversation_revision(
            owner_id=actor.id,
            telegram_user_id=actor.telegram_id,
            chat_id=chat_id,
            context=context,
        ),
    )


async def _source(db, actor: User, chat_id: int, user_text: str, answer: str):
    conversation = ConversationContextService(db, 20, 24)
    snapshot = await conversation.get(actor.telegram_id, chat_id)
    context = snapshot.for_companion_prompt()
    fence = CompanionConversationFence(
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=chat_id,
        access_version=actor.access_version,
        access_tier=actor.access_tier,
        raw_message_limit=20,
        conversation_payload_max_bytes=32 * 1024,
        revision=companion_conversation_revision(
            owner_id=actor.id,
            telegram_user_id=actor.telegram_id,
            chat_id=chat_id,
            context=context,
        ),
    )
    receipt = await conversation.append_exchange(
        fence,
        user_content=user_text,
        assistant_content=answer,
        user_source="text",
    )
    assert receipt is not None
    source = receipt.source_identity_for(fence)
    assert source is not None
    return source


def _policy() -> NovaBrainPolicy:
    return NovaBrainPolicy(enabled=True, admin_only=False)


def _response_length_memory(
    evidence: str,
    *,
    value: str = "short",
    supersedes_value: str | None = None,
) -> NovaCompanionMemoryCandidate:
    return NovaCompanionMemoryCandidate(
        category="preference",
        key="response_length",
        value=value,
        evidence=evidence,
        salience=5,
        supersedes_value=supersedes_value,
    )


def _structured_memory(
    key: str,
    value: str,
    evidence: str,
    *,
    supersedes_value: str | None = None,
) -> NovaCompanionMemoryCandidate:
    return NovaCompanionMemoryCandidate(
        category="identity" if key == "identity" else "preference",
        key=key,
        value=value,
        evidence=evidence,
        salience=5,
        supersedes_value=supersedes_value,
    )


async def _apply_structured_memory(
    db,
    service: NovaBrainService,
    actor: User,
    text: str,
    candidate: NovaCompanionMemoryCandidate,
    *,
    chat_id: int | None = None,
):
    target_chat_id = actor.telegram_id if chat_id is None else chat_id
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=target_chat_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text=text,
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(db, actor, target_chat_id, text, "Поняла настройку.")
    return await service.apply_turn(
        snapshot.fence,
        source,
        state_update=None,
        memory_candidate=candidate,
        user_text=text,
        policy=_policy(),
    )


def test_nova_brain_server_validation_rejects_inferred_sensitive_and_one_off_memory():
    assert (
        validate_memory_candidate(
            NovaCompanionMemoryCandidate(
                category="preference",
                key="response_length",
                value="short",
                evidence="Я предпочитаю короткие ответы",
            ),
            user_text="Я предпочитаю короткие ответы",
        )
        is not None
    )
    for text, value in (
        ("Кажется, пользователь любит чай", "пользователь любит чай"),
        ("Сегодня я очень устала", "сегодня я очень устала"),
        ("У меня диагноз PRIVATE", "у меня диагноз private"),
        ("Моя подруга любит чай", "моя подруга любит чай"),
    ):
        assert (
            validate_memory_candidate(
                NovaCompanionMemoryCandidate(
                    category="fact",
                    value=value,
                    evidence=text,
                ),
                user_text=text,
            )
            is None
        )
    assert (
        validate_memory_candidate(
            NovaCompanionMemoryCandidate(
                category="fact",
                value="работаю редактором",
                evidence="Теперь я работаю редактором",
            ),
            user_text="Теперь я работаю редактором",
        )
        is None
    )
    assert (
        validate_memory_candidate(
            NovaCompanionMemoryCandidate(
                category="fact",
                value="работаю редактором",
                evidence="Я всегда работаю редактором",
                supersedes_value="работаю дизайнером",
            ),
            user_text="Я всегда работаю редактором",
        )
        is None
    )


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("У меня депрессия", "fact"),
        ("Депрессия диагностирована у меня", "fact"),
        ("У меня зарплата 500000 рублей", "fact"),
        ("500000 рублей — моя зарплата", "fact"),
        ("У меня почта user@example.com", "fact"),
        ("Мой номер: +7 999 123-45-67", "fact"),
        ("Я живу по адресу Ленина 10", "fact"),
        ("Ленина, дом 10 — мой адрес", "fact"),
        ("У меня есть дочь Маша", "fact"),
        ("Машу зовут моя дочь", "identity"),
        ("Я голосую за эту партию", "orientation"),
        ("Мой паспорт 1234 567890", "fact"),
    ],
)
def test_nova_brain_observed_memory_is_fail_closed_for_sensitive_and_third_party_data(
    text,
    category,
):
    assert (
        validate_memory_candidate(
            NovaCompanionMemoryCandidate(
                category=category,
                value=text,
                evidence=text,
            ),
            user_text=text,
        )
        is None
    )


def test_nova_brain_identity_allows_only_exact_self_declared_identity():
    self_declared = "Меня зовут Назар. Я мужчина"
    accepted = validate_memory_candidate(
        NovaCompanionMemoryCandidate(
            category="identity",
            key="identity",
            value="display_name=Назар;grammatical_address=masculine",
            evidence=self_declared,
            salience=5,
        ),
        user_text=self_declared,
    )
    assert accepted is not None
    assert accepted.value == "identity:display_name=Назар;grammatical_address=masculine"

    third_party = "Мою дочь зовут Маша"
    assert (
        validate_memory_candidate(
            NovaCompanionMemoryCandidate(
                category="identity",
                value=third_party,
                evidence=third_party,
            ),
            user_text=third_party,
        )
        is None
    )


@pytest.mark.parametrize(
    ("user_text", "evidence", "key", "value", "category"),
    [
        (
            "Дочь сказала: «Меня зовут Маша»",
            "Меня зовут Маша",
            "identity",
            "display_name=Маша",
            "identity",
        ),
        (
            "Мой муж говорит: «Я мужчина»",
            "Я мужчина",
            "identity",
            "grammatical_address=masculine",
            "identity",
        ),
        (
            "Коллега просит: «Отвечай мне коротко»",
            "Отвечай мне коротко",
            "response_length",
            "short",
            "preference",
        ),
        (
            "Не повторяй фразу «Говори со мной спокойно»",
            "Говори со мной спокойно",
            "tone",
            "calm",
            "preference",
        ),
        (
            "В инструкции написано: «Напоминай мне мягко»",
            "Напоминай мне мягко",
            "reminder_style",
            "gentle",
            "preference",
        ),
        (
            "Она повторила («Меня зовут Маша»)",
            "Меня зовут Маша",
            "identity",
            "display_name=Маша",
            "identity",
        ),
        (
            "Пример — отвечай мне коротко",
            "отвечай мне коротко",
            "response_length",
            "short",
            "preference",
        ),
        (
            "Я не мужчина",
            "Я мужчина",
            "identity",
            "grammatical_address=masculine",
            "identity",
        ),
        (
            "Меня зовут Назар?",
            "Меня зовут Назар",
            "identity",
            "display_name=Назар",
            "identity",
        ),
        (
            "Возможно, меня зовут Назар",
            "меня зовут Назар",
            "identity",
            "display_name=Назар",
            "identity",
        ),
        (
            "Если понадобится, говори со мной спокойно",
            "говори со мной спокойно",
            "tone",
            "calm",
            "preference",
        ),
        (
            "Отвечай мне коротко, это только пример",
            "Отвечай мне коротко",
            "response_length",
            "short",
            "preference",
        ),
        (
            "(Напоминай мне мягко)",
            "Напоминай мне мягко",
            "reminder_style",
            "gentle",
            "preference",
        ),
    ],
)
def test_nova_brain_automatic_memory_rejects_provider_evidence_slicing(
    user_text,
    evidence,
    key,
    value,
    category,
):
    proposal = NovaCompanionMemoryCandidate(
        category=category,
        key=key,
        value=value,
        evidence=evidence,
    )

    assert validate_memory_candidate(proposal, user_text=user_text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Я постоянно лечусь от биполярки",
        "Я часто употребляю кокаин",
        "Я люблю секс без презерватива",
        "Я часто хожу в мечеть",
        "Я часто поддерживаю Навального",
        "Я постоянно получаю 500 тысяч",
        "Я постоянно забираю Машу из школы",
        "Каждую неделю прохожу длительное лечение",
        "Регулярно принимаю сильные вещества",
        "Мне нравится близость без защиты",
        "По пятницам бываю на общей молитве",
        "Я за известного оппозиционного деятеля",
        "Каждый месяц мне приходит очень крупная сумма",
        "После уроков всегда встречаю Машу",
        "Я люблю готовить вместе с Мариной",
    ],
)
def test_nova_brain_free_text_can_never_impersonate_a_structured_setting(text):
    malicious = NovaCompanionMemoryCandidate(
        category="preference",
        key="tone",
        value="calm",
        evidence=text,
    )

    assert validate_memory_candidate(malicious, user_text=text) is None


@pytest.mark.parametrize(
    ("text", "key", "value", "canonical"),
    [
        (
            "Я предпочитаю короткие ответы",
            "response_length",
            "short",
            "response_length=short",
        ),
        (
            "Нова, отвечай мне коротко, пожалуйста",
            "response_length",
            "short",
            "response_length=short",
        ),
        ("Говори со мной спокойно", "tone", "calm", "tone=calm"),
        ("Напоминай мне мягко", "reminder_style", "gentle", "reminder_style=gentle"),
    ],
)
def test_nova_brain_accepts_only_server_parsed_communication_enums(
    text,
    key,
    value,
    canonical,
):
    accepted = validate_memory_candidate(
        NovaCompanionMemoryCandidate(
            category="preference",
            key=key,
            value=value,
            evidence=text,
        ),
        user_text=text,
    )

    assert accepted is not None
    assert accepted.value == canonical
    assert accepted.salience == 4


def test_nova_brain_state_validation_requires_exact_user_and_assistant_grounding():
    proposal = NovaCompanionDialogueStateUpdate(
        active_topic="ментальные тренировки",
        last_assistant_offer="Могу предложить короткое упражнение.",
        last_assistant_offer_kinds=["exercise"],
        unresolved_question="Какой вариант тебе ближе?",
    )
    assert (
        validate_dialogue_state_update(
            proposal,
            user_text="Хочу обсудить ментальные тренировки",
            assistant_answer=("Могу предложить короткое упражнение. Какой вариант тебе ближе?"),
            visible_action=None,
        )
        is not None
    )
    assert (
        validate_dialogue_state_update(
            proposal,
            user_text="Другая тема",
            assistant_answer=("Могу предложить короткое упражнение. Какой вариант тебе ближе?"),
            visible_action=None,
        )
        is None
    )
    clear_attempt = validate_dialogue_state_update(
        NovaCompanionDialogueStateUpdate(clear_fields=["active_topic", "open_loops"]),
        user_text="Продолжим",
        assistant_answer="Продолжим",
        visible_action=None,
    )
    assert clear_attempt is not None and clear_attempt.clear_fields == []


async def test_nova_brain_apply_persists_state_and_observed_memory_across_service_restart(db):
    actor = await _actor(db, 810_001)
    text = "Я предпочитаю короткие ответы"
    answer = "Поняла. Какой вариант тебе ближе?"
    source = await _source(db, actor, actor.telegram_id, text, answer)
    service = NovaBrainService(db)
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text=text,
        policy=_policy(),
    )
    assert snapshot.fence is not None
    state = NovaCompanionDialogueStateUpdate(
        active_topic="короткие ответы",
        unresolved_question="Какой вариант тебе ближе?",
    )
    memory = NovaCompanionMemoryCandidate(
        category="preference",
        key="response_length",
        value="short",
        evidence=text,
        salience=5,
    )

    receipt = await service.apply_turn(
        snapshot.fence,
        source,
        state_update=state,
        memory_candidate=memory,
        user_text=text,
        policy=_policy(),
    )

    assert receipt is not None
    recreated = NovaBrainService(db)
    after = await recreated.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Как тебе лучше отвечать?",
        policy=_policy(),
    )
    assert after.projection is not None
    assert after.projection.working_state.active_topic == "короткие ответы"
    assert [item.value for item in after.projection.memories] == ["response_length=short"]


async def test_nova_brain_stale_generation_cannot_overwrite_newer_state(db):
    actor = await _actor(db, 810_002)
    service = NovaBrainService(db)
    first = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Первая тема",
        policy=_policy(),
    )
    assert first.fence is not None
    source = await _source(db, actor, actor.telegram_id, "Первая тема", "Ответ")
    applied = await service.apply_turn(
        first.fence,
        source,
        state_update=NovaCompanionDialogueStateUpdate(active_topic="Первая тема"),
        memory_candidate=None,
        user_text="Первая тема",
        policy=_policy(),
    )
    assert applied is not None

    stale = await service.apply_turn(
        first.fence,
        source,
        state_update=NovaCompanionDialogueStateUpdate(active_topic="Старая тема"),
        memory_candidate=None,
        user_text="Старая тема",
        policy=_policy(),
    )

    assert stale is None
    current = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="",
        policy=_policy(),
    )
    assert current.projection is not None
    assert current.projection.working_state.active_topic == "Первая тема"


async def test_nova_brain_compensation_removes_only_exact_generation(db):
    actor = await _actor(db, 810_003)
    service = NovaBrainService(db)
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Я предпочитаю короткие ответы",
        "Поняла",
    )
    receipt = await service.apply_turn(
        snapshot.fence,
        source,
        state_update=NovaCompanionDialogueStateUpdate(active_topic="короткие ответы"),
        memory_candidate=_response_length_memory("Я предпочитаю короткие ответы"),
        user_text="Я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert receipt is not None

    assert await service.compensate_turn(receipt)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0


async def test_nova_brain_forget_is_owner_revision_and_access_scoped(db):
    actor = await _actor(db, 810_004)
    other = await _actor(db, 810_005)
    service = NovaBrainService(db)
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(db, actor, actor.telegram_id, "Я предпочитаю короткие ответы", "Поняла")
    receipt = await service.apply_turn(
        snapshot.fence,
        source,
        state_update=None,
        memory_candidate=_response_length_memory("Я предпочитаю короткие ответы"),
        user_text="Я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert receipt is not None
    memories = await service.list_current(
        telegram_actor_id=actor.telegram_id,
        expected_access_version=actor.access_version,
        policy=_policy(),
    )
    item = memories[0]

    cross = await service.forget_exact(
        telegram_actor_id=other.telegram_id,
        public_id=item.public_id,
        expected_revision=item.revision,
        expected_access_version=other.access_version,
        policy=_policy(),
    )
    assert cross.status == "not_found"
    stale = await service.forget_exact(
        telegram_actor_id=actor.telegram_id,
        public_id=item.public_id,
        expected_revision=item.revision + 1,
        expected_access_version=actor.access_version,
        policy=_policy(),
    )
    assert stale.status == "stale"
    forgotten = await service.forget_exact(
        telegram_actor_id=actor.telegram_id,
        public_id=item.public_id,
        expected_revision=item.revision,
        expected_access_version=actor.access_version,
        policy=_policy(),
    )
    assert forgotten.status == "applied"
    assert (
        await service.list_current(
            telegram_actor_id=actor.telegram_id,
            expected_access_version=actor.access_version,
            policy=_policy(),
        )
        == ()
    )


async def test_nova_brain_guest_and_blocked_are_fail_closed(db):
    guest = await _actor(db, 810_006, tier="guest")
    service = NovaBrainService(db)
    snapshot = await service.snapshot(
        telegram_actor_id=guest.telegram_id,
        chat_id=guest.telegram_id,
        expected_tier=guest.access_tier,
        expected_access_version=guest.access_version,
        current_text="Я предпочитаю чай",
        policy=_policy(),
        now=datetime(2026, 8, 22, tzinfo=UTC),
    )
    assert snapshot.status == "access_changed"


async def test_nova_brain_forget_capability_is_opaque_owner_chat_exact_and_single_winner(db):
    actor = await _actor(db, 810_007)
    service = NovaBrainService(db)
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(db, actor, actor.telegram_id, "Я предпочитаю короткие ответы", "Поняла")
    assert (
        await service.apply_turn(
            snapshot.fence,
            source,
            state_update=None,
            memory_candidate=_response_length_memory("Я предпочитаю короткие ответы"),
            user_text="Я предпочитаю короткие ответы",
            policy=_policy(),
        )
        is not None
    )
    memory = (
        await service.list_current(
            telegram_actor_id=actor.telegram_id,
            expected_access_version=actor.access_version,
            policy=_policy(),
        )
    )[0]
    store = NovaBrainForgetStore()
    stage = await store.stage(
        memory,
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        access_version=actor.access_version,
    )
    assert memory.public_id not in repr(stage)
    bound = await store.bind(stage, canonical_message_id=99)
    assert bound is not None
    confirm = await store.peek(
        bound.callback_data("confirm"),
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        canonical_message_id=99,
    )
    assert confirm is not None
    assert (
        await store.peek(
            bound.callback_data("confirm"),
            owner_id=actor.id,
            telegram_user_id=actor.telegram_id,
            chat_id=actor.telegram_id + 1,
            canonical_message_id=99,
        )
        is None
    )
    assert await store.consume(confirm)
    assert (
        await store.peek(
            bound.callback_data("cancel"),
            owner_id=actor.id,
            telegram_user_id=actor.telegram_id,
            chat_id=actor.telegram_id,
            canonical_message_id=99,
        )
        is None
    )
    assert not await store.consume(confirm)


@pytest.mark.parametrize(
    ("key", "first_text", "first_value", "second_text", "second_value"),
    [
        (
            "response_length",
            "Отвечай мне коротко",
            "short",
            "Отвечай мне подробно",
            "detailed",
        ),
        ("tone", "Говори со мной спокойно", "calm", "Говори со мной прямо", "direct"),
        (
            "reminder_style",
            "Напоминай мне мягко",
            "gentle",
            "Напоминай мне кратко",
            "brief",
        ),
    ],
)
async def test_nova_brain_structured_setting_replacement_is_server_owned_singleton(
    db,
    key,
    first_text,
    first_value,
    second_text,
    second_value,
):
    actor = await _actor(db, 811_100 + len(key))
    service = NovaBrainService(db)
    assert await _apply_structured_memory(
        db,
        service,
        actor,
        first_text,
        _structured_memory(key, first_value, first_text),
    )
    assert await _apply_structured_memory(
        db,
        service,
        actor,
        second_text,
        _structured_memory(key, second_value, second_text),
    )

    expected = f"{key}={second_value}"
    current = await service.list_current(
        telegram_actor_id=actor.telegram_id,
        expected_access_version=actor.access_version,
        policy=_policy(),
    )
    assert [item.value for item in current] == [expected]
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text=second_text,
        policy=_policy(),
    )
    assert snapshot.projection is not None
    assert [item.value for item in snapshot.projection.memories] == [expected]
    async with db.sessions() as session:
        rows = list(
            (
                await session.scalars(select(NovaObservedMemory).order_by(NovaObservedMemory.id))
            ).all()
        )
        assert [(row.normalized_value, row.status) for row in rows] == [
            (f"{key}={first_value}", "superseded"),
            (expected, "active"),
        ]
        assert [row.semantic_key for row in rows] == [key, key]


async def test_nova_brain_exact_structured_repeat_is_memory_noop(db):
    actor = await _actor(db, 811_140)
    service = NovaBrainService(db)
    text = "Отвечай мне коротко"
    candidate = _structured_memory("response_length", "short", text)
    first = await _apply_structured_memory(db, service, actor, text, candidate)
    assert first is not None and first.memory_id is not None
    async with db.sessions() as session:
        row = await session.scalar(select(NovaObservedMemory))
        assert row is not None
        exact_generation = (row.id, row.revision, row.source_receipt, row.updated_at)

    repeated = await _apply_structured_memory(db, service, actor, text, candidate)
    assert repeated is not None and repeated.memory_id is None
    async with db.sessions() as session:
        rows = list((await session.scalars(select(NovaObservedMemory))).all())
        assert len(rows) == 1
        assert (rows[0].id, rows[0].revision, rows[0].source_receipt, rows[0].updated_at) == (
            exact_generation
        )


async def test_nova_brain_identity_is_field_merged_into_one_effective_row(db):
    actor = await _actor(db, 811_150)
    service = NovaBrainService(db)
    turns = (
        ("Меня зовут Назар", "display_name=Назар", "identity:display_name=назар"),
        (
            "Я мужчина",
            "grammatical_address=masculine",
            "identity:display_name=назар;grammatical_address=masculine",
        ),
        (
            "Теперь меня зовут Иван",
            "display_name=Иван",
            "identity:display_name=иван;grammatical_address=masculine",
        ),
        (
            "Обращайся ко мне в женском роде",
            "grammatical_address=feminine",
            "identity:display_name=иван;grammatical_address=feminine",
        ),
    )
    for text, value, expected in turns:
        receipt = await _apply_structured_memory(
            db,
            service,
            actor,
            text,
            _structured_memory("identity", value, text),
        )
        assert receipt is not None
        current = await service.list_current(
            telegram_actor_id=actor.telegram_id,
            expected_access_version=actor.access_version,
            policy=_policy(),
        )
        assert [item.value for item in current] == [expected]

    async with db.sessions() as session:
        rows = list((await session.scalars(select(NovaObservedMemory))).all())
        assert len(rows) == 4
        assert sum(row.status == "active" for row in rows) == 1
        active = next(row for row in rows if row.status == "active")
        assert active.semantic_key == "identity"
        assert active.normalized_value == turns[-1][2]


async def test_nova_brain_provider_supersedes_hint_cannot_cross_semantic_key_or_owner(db):
    first_actor = await _actor(db, 811_160)
    second_actor = await _actor(db, 811_161)
    service = NovaBrainService(db)
    calm = "Говори со мной спокойно"
    short = "Отвечай мне коротко"
    assert await _apply_structured_memory(
        db,
        service,
        first_actor,
        calm,
        _structured_memory("tone", "calm", calm),
    )
    assert await _apply_structured_memory(
        db,
        service,
        first_actor,
        short,
        _structured_memory("response_length", "short", short),
    )
    other = "Напоминай мне мягко"
    assert await _apply_structured_memory(
        db,
        service,
        second_actor,
        other,
        _structured_memory("reminder_style", "gentle", other),
    )

    detailed = "Отвечай мне подробно"
    forged = _structured_memory(
        "response_length",
        "detailed",
        detailed,
        supersedes_value="tone=calm",
    )
    validated = validate_memory_candidate(forged, user_text=detailed)
    assert validated is not None and validated.supersedes_value is None
    assert await _apply_structured_memory(db, service, first_actor, detailed, forged)

    first_values = {
        item.value
        for item in await service.list_current(
            telegram_actor_id=first_actor.telegram_id,
            expected_access_version=first_actor.access_version,
            policy=_policy(),
        )
    }
    second_values = {
        item.value
        for item in await service.list_current(
            telegram_actor_id=second_actor.telegram_id,
            expected_access_version=second_actor.access_version,
            policy=_policy(),
        )
    }
    assert first_values == {"tone=calm", "response_length=detailed"}
    assert second_values == {"reminder_style=gentle"}


async def test_nova_brain_replacement_is_owner_chat_and_access_fenced(db):
    first_actor = await _actor(db, 811_165)
    second_actor = await _actor(db, 811_166)
    shared_chat_id = 811_999
    service = NovaBrainService(db)
    short = "Отвечай мне коротко"
    calm = "Говори со мной спокойно"
    assert await _apply_structured_memory(
        db,
        service,
        first_actor,
        short,
        _structured_memory("response_length", "short", short),
        chat_id=shared_chat_id,
    )
    assert await _apply_structured_memory(
        db,
        service,
        second_actor,
        calm,
        _structured_memory("tone", "calm", calm),
        chat_id=shared_chat_id,
    )

    detailed = "Отвечай мне подробно"
    stale = await service.snapshot(
        telegram_actor_id=first_actor.telegram_id,
        chat_id=shared_chat_id,
        expected_tier=first_actor.access_tier,
        expected_access_version=first_actor.access_version,
        current_text=detailed,
        policy=_policy(),
    )
    assert stale.fence is not None
    source = await _source(db, first_actor, shared_chat_id, detailed, "Поняла настройку.")
    async with db.session() as session:
        await session.execute(
            update(User)
            .where(User.id == first_actor.id)
            .values(access_version=first_actor.access_version + 1)
        )
    assert (
        await service.apply_turn(
            stale.fence,
            source,
            state_update=None,
            memory_candidate=_structured_memory("response_length", "detailed", detailed),
            user_text=detailed,
            policy=_policy(),
        )
        is None
    )
    first_current = await service.list_current(
        telegram_actor_id=first_actor.telegram_id,
        expected_access_version=first_actor.access_version + 1,
        policy=_policy(),
    )
    second_current = await service.list_current(
        telegram_actor_id=second_actor.telegram_id,
        expected_access_version=second_actor.access_version,
        policy=_policy(),
    )
    assert [item.value for item in first_current] == ["response_length=short"]
    assert [item.value for item in second_current] == ["tone=calm"]


async def test_nova_brain_concurrent_same_key_replacements_have_one_fenced_winner(db):
    actor = await _actor(db, 811_170)
    service = NovaBrainService(db)
    initial = "Отвечай мне коротко"
    assert await _apply_structured_memory(
        db,
        service,
        actor,
        initial,
        _structured_memory("response_length", "short", initial),
    )
    snapshots = [
        await service.snapshot(
            telegram_actor_id=actor.telegram_id,
            chat_id=actor.telegram_id,
            expected_tier=actor.access_tier,
            expected_access_version=actor.access_version,
            current_text=text,
            policy=_policy(),
        )
        for text in ("Отвечай мне подробно", "Отвечай мне обычно")
    ]
    assert all(snapshot.fence is not None for snapshot in snapshots)
    sources = [
        await _source(db, actor, actor.telegram_id, text, "Поняла настройку.")
        for text in ("Отвечай мне подробно", "Отвечай мне обычно")
    ]
    results = await asyncio.gather(
        *(
            service.apply_turn(
                snapshot.fence,
                source,
                state_update=None,
                memory_candidate=_structured_memory("response_length", value, text),
                user_text=text,
                policy=_policy(),
            )
            for snapshot, source, text, value in zip(
                snapshots,
                sources,
                ("Отвечай мне подробно", "Отвечай мне обычно"),
                ("detailed", "normal"),
                strict=True,
            )
            if snapshot.fence is not None
        )
    )
    assert sum(result is not None for result in results) == 1
    current = await service.list_current(
        telegram_actor_id=actor.telegram_id,
        expected_access_version=actor.access_version,
        policy=_policy(),
    )
    assert len(current) == 1
    assert current[0].value in {"response_length=detailed", "response_length=normal"}
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(NovaObservedMemory.id)).where(
                    NovaObservedMemory.status == "active",
                    NovaObservedMemory.semantic_key == "response_length",
                )
            )
            == 1
        )


@pytest.mark.parametrize("failure", [RuntimeError("PRIVATE_REPLACE"), asyncio.CancelledError()])
async def test_nova_brain_replacement_failure_or_cancellation_restores_prior_value(db, failure):
    actor = await _actor(db, 811_180)
    service = NovaBrainService(db)
    short = "Отвечай мне коротко"
    assert await _apply_structured_memory(
        db,
        service,
        actor,
        short,
        _structured_memory("response_length", "short", short),
    )

    async def fail_after_commit(_receipt):
        raise failure

    service._after_apply_commit = fail_after_commit
    detailed = "Отвечай мне подробно"
    with pytest.raises(type(failure)):
        await _apply_structured_memory(
            db,
            service,
            actor,
            detailed,
            _structured_memory("response_length", "detailed", detailed),
        )
    current = await service.list_current(
        telegram_actor_id=actor.telegram_id,
        expected_access_version=actor.access_version,
        policy=_policy(),
    )
    assert [item.value for item in current] == ["response_length=short"]
    async with db.sessions() as session:
        rows = list((await session.scalars(select(NovaObservedMemory))).all())
        assert len(rows) == 1
        assert rows[0].status == "active"


async def test_nova_brain_correction_supersedes_exact_value_and_compensates_atomically(db):
    actor = await _actor(db, 810_008)
    service = NovaBrainService(db)
    first = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я предпочитаю ответы средней длины",
        policy=_policy(),
    )
    assert first.fence is not None
    first_source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Я предпочитаю ответы средней длины",
        "Поняла",
    )
    assert (
        await service.apply_turn(
            first.fence,
            first_source,
            state_update=None,
            memory_candidate=_response_length_memory(
                "Я предпочитаю ответы средней длины",
                value="normal",
            ),
            user_text="Я предпочитаю ответы средней длины",
            policy=_policy(),
        )
        is not None
    )
    second = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Теперь я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert second.fence is not None
    second_source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Теперь я предпочитаю короткие ответы",
        "Учту поправку",
    )
    correction = await service.apply_turn(
        second.fence,
        second_source,
        state_update=None,
        memory_candidate=_response_length_memory(
            "Теперь я предпочитаю короткие ответы",
            supersedes_value="response_length=normal",
        ),
        user_text="Теперь я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert correction is not None
    async with db.sessions() as session:
        rows = list(
            (
                await session.scalars(select(NovaObservedMemory).order_by(NovaObservedMemory.id))
            ).all()
        )
        assert [(row.normalized_value, row.status) for row in rows] == [
            ("response_length=normal", "superseded"),
            ("response_length=short", "active"),
        ]

    assert await service.compensate_turn(correction)
    async with db.sessions() as session:
        rows = list(
            (
                await session.scalars(select(NovaObservedMemory).order_by(NovaObservedMemory.id))
            ).all()
        )
        assert [(row.normalized_value, row.status) for row in rows] == [
            ("response_length=normal", "active")
        ]


async def test_nova_brain_compensation_never_reactivates_old_value_when_newer_changed(db):
    actor = await _actor(db, 810_009)
    service = NovaBrainService(db)
    first = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я предпочитаю ответы средней длины",
        policy=_policy(),
    )
    assert first.fence is not None
    source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Я предпочитаю ответы средней длины",
        "Поняла",
    )
    assert await service.apply_turn(
        first.fence,
        source,
        state_update=None,
        memory_candidate=_response_length_memory(
            "Я предпочитаю ответы средней длины",
            value="normal",
        ),
        user_text="Я предпочитаю ответы средней длины",
        policy=_policy(),
    )
    second = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Теперь я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert second.fence is not None
    source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Теперь я предпочитаю короткие ответы",
        "Поняла",
    )
    correction = await service.apply_turn(
        second.fence,
        source,
        state_update=None,
        memory_candidate=_response_length_memory(
            "Теперь я предпочитаю короткие ответы",
            supersedes_value="response_length=normal",
        ),
        user_text="Теперь я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert correction is not None
    async with db.session() as session:
        current = await session.scalar(
            select(NovaObservedMemory).where(
                NovaObservedMemory.normalized_value == "response_length=short"
            )
        )
        assert current is not None
        current.revision += 1

    assert not await service.compensate_turn(correction)
    async with db.sessions() as session:
        rows = list(
            (
                await session.scalars(select(NovaObservedMemory).order_by(NovaObservedMemory.id))
            ).all()
        )
        assert [(row.normalized_value, row.status) for row in rows] == [
            ("response_length=normal", "superseded"),
            ("response_length=short", "active"),
        ]


async def test_nova_brain_revived_memory_compensation_restores_full_prior_row(db):
    actor = await _actor(db, 810_015)
    prior_updated_at = datetime(2026, 8, 1, tzinfo=UTC)
    value = "response_length=short"
    async with db.session() as session:
        session.add(
            NovaObservedMemory(
                public_id="00000000-0000-0000-0000-000000000015",
                owner_id=actor.id,
                category="fact",
                normalized_value=value,
                content_fingerprint=sha256(value.encode()).hexdigest(),
                source_kind="conversation",
                source_session_id=11,
                source_message_id=12,
                source_receipt="1" * 64,
                status="forgotten",
                salience=1,
                revision=4,
                updated_at=prior_updated_at,
            )
        )
    service = NovaBrainService(db)
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я всегда предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Я всегда предпочитаю короткие ответы",
        "Поняла",
    )
    receipt = await service.apply_turn(
        snapshot.fence,
        source,
        state_update=None,
        memory_candidate=_response_length_memory("Я предпочитаю короткие ответы"),
        user_text="Я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert receipt is not None and await service.compensate_turn(receipt)
    async with db.sessions() as session:
        row = await session.scalar(select(NovaObservedMemory))
        assert row is not None
        assert (
            row.category,
            row.status,
            row.salience,
            row.revision,
            row.source_session_id,
            row.source_message_id,
            row.source_receipt,
        ) == ("fact", "forgotten", 1, 4, 11, 12, "1" * 64)
        assert service._aware(row.updated_at) == prior_updated_at


@pytest.mark.parametrize("failure", [RuntimeError("PRIVATE_POST_COMMIT"), asyncio.CancelledError()])
async def test_nova_brain_post_commit_failure_or_cancellation_compensates_exact_turn(
    db,
    failure,
):
    actor = await _actor(db, 810_017)
    service = NovaBrainService(db)
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я всегда предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Я всегда предпочитаю короткие ответы",
        "Поняла",
    )

    async def fail_after_commit(_receipt):
        raise failure

    service._after_apply_commit = fail_after_commit
    with pytest.raises(type(failure)):
        await service.apply_turn(
            snapshot.fence,
            source,
            state_update=NovaCompanionDialogueStateUpdate(active_topic="короткие ответы"),
            memory_candidate=_response_length_memory("Я предпочитаю короткие ответы"),
            user_text="Я предпочитаю короткие ответы",
            policy=_policy(),
        )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0
        assert await session.scalar(select(func.count(NovaObservedMemory.id))) == 0


async def test_nova_brain_retrieval_is_relevant_deterministic_diverse_and_bounded(db):
    actor = await _actor(db, 810_010)
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    values = (
        ("preference", "response_length=short", 5),
        ("identity", "identity:display_name=назар", 4),
        ("orientation", "ментальные тренировки помогают держать фокус", 4),
        ("theme", "люблю обсуждать садовые цветы", 5),
    )
    async with db.session() as session:
        for index, (category, value, salience) in enumerate(values, start=1):
            session.add(
                NovaObservedMemory(
                    public_id=f"00000000-0000-0000-0000-{index:012d}",
                    owner_id=actor.id,
                    category=category,
                    semantic_key=(
                        "response_length"
                        if category == "preference"
                        else "identity"
                        if category == "identity"
                        else None
                    ),
                    normalized_value=value,
                    content_fingerprint=sha256(value.encode()).hexdigest(),
                    source_kind="conversation",
                    source_session_id=1,
                    source_message_id=index,
                    source_receipt=sha256(f"receipt-{index}".encode()).hexdigest(),
                    status="active",
                    salience=salience,
                    revision=1,
                    created_at=now - timedelta(days=index),
                    updated_at=now - timedelta(days=index),
                )
            )
    service = NovaBrainService(db, retrieval_max_items=2, context_max_bytes=1024)
    first = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Хочу продолжить ментальные тренировки и держать фокус",
        policy=_policy(),
        now=now,
    )
    second = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Хочу продолжить ментальные тренировки и держать фокус",
        policy=_policy(),
        now=now,
    )
    assert first.projection == second.projection
    assert first.projection is not None
    assert [item.category for item in first.projection.memories] == [
        "preference",
        "identity",
    ]
    assert first.projection.payload_bytes <= 1024
    assert all(
        "сад" not in item.value and "чай" not in item.value for item in first.projection.memories
    )


async def test_nova_brain_new_substantive_topic_closes_prior_offer_and_open_loop(db):
    actor = await _actor(db, 810_016)
    service = NovaBrainService(db)
    async with db.session() as session:
        session.add(
            NovaDialogueState(
                owner_id=actor.id,
                telegram_user_id=actor.telegram_id,
                chat_id=actor.telegram_id,
                access_version=actor.access_version,
                active_topic="ментальные тренировки",
                last_assistant_offer="Можем составить простой план.",
                last_assistant_offer_kinds=["plan"],
                unresolved_question="С чего начнём?",
                requested_action="plan",
                open_loops=["выбрать упражнение"],
                revision=3,
                expires_at=datetime.now(UTC) + timedelta(days=1),
            )
        )
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Теперь хочу обсудить подготовку к выступлению",
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Теперь хочу обсудить подготовку к выступлению",
        "Давай",
    )
    assert await service.apply_turn(
        snapshot.fence,
        source,
        state_update=NovaCompanionDialogueStateUpdate(active_topic="подготовку к выступлению"),
        memory_candidate=None,
        user_text="Теперь хочу обсудить подготовку к выступлению",
        policy=_policy(),
    )
    current = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="",
        policy=_policy(),
    )
    assert current.projection is not None
    state = current.projection.working_state
    assert state.active_topic == "подготовку к выступлению"
    assert state.last_assistant_offer is None
    assert state.last_assistant_offer_kinds == ()
    assert state.unresolved_question is None
    assert state.requested_action is None
    assert state.open_loops == ()


async def test_nova_brain_snapshot_physically_scrubs_state_at_exact_expiry(db):
    actor = await _actor(db, 810_116)
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    async with db.session() as session:
        session.add(
            NovaDialogueState(
                owner_id=actor.id,
                telegram_user_id=actor.telegram_id,
                chat_id=actor.telegram_id,
                access_version=actor.access_version,
                active_topic="устаревшая приватная тема",
                last_assistant_offer_kinds=[],
                open_loops=[],
                revision=4,
                expires_at=now,
            )
        )

    snapshot = await NovaBrainService(db).snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Продолжим",
        policy=_policy(),
        now=now,
    )

    assert snapshot.status == "ready"
    assert snapshot.projection is not None
    assert snapshot.projection.working_state.revision == 0
    async with db.session() as session:
        assert await session.scalar(select(func.count(NovaDialogueState.id))) == 0


async def test_nova_brain_expiry_cleanup_cas_preserves_concurrent_newer_generation(db):
    actor = await _actor(db, 810_117)
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    async with db.session() as session:
        state = NovaDialogueState(
            owner_id=actor.id,
            telegram_user_id=actor.telegram_id,
            chat_id=actor.telegram_id,
            access_version=actor.access_version,
            active_topic="старая тема",
            last_assistant_offer_kinds=[],
            open_loops=[],
            revision=1,
            expires_at=now,
        )
        session.add(state)
        await session.flush()
        state_id = state.id
    service = NovaBrainService(db)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def block_cleanup(candidate_id, revision):
        assert (candidate_id, revision) == (state_id, 1)
        cleanup_started.set()
        await cleanup_release.wait()

    service._before_expired_state_cleanup = block_cleanup
    pending = asyncio.create_task(
        service.snapshot(
            telegram_actor_id=actor.telegram_id,
            chat_id=actor.telegram_id,
            expected_tier=actor.access_tier,
            expected_access_version=actor.access_version,
            current_text="Продолжим",
            policy=_policy(),
            now=now,
        )
    )
    await cleanup_started.wait()
    async with db.session() as session:
        await session.execute(
            update(NovaDialogueState)
            .where(NovaDialogueState.id == state_id)
            .values(
                active_topic="новая тема",
                revision=2,
                expires_at=now + timedelta(days=1),
            )
        )
    cleanup_release.set()
    snapshot = await pending

    assert snapshot.projection is not None
    assert snapshot.projection.working_state.active_topic == "новая тема"
    assert snapshot.projection.working_state.revision == 2
    async with db.session() as session:
        current = await session.get(NovaDialogueState, state_id)
        assert current is not None
        assert current.revision == 2


def test_nova_brain_projection_rejects_forged_nested_or_oversized_values():
    with pytest.raises(ValueError):
        NovaDialogueStateView(active_topic={"private_id": 7})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NovaObservedMemoryView(
            public_id="00000000-0000-0000-0000-000000000001",
            category="preference",
            value={"private": "value"},  # type: ignore[arg-type]
            salience=3,
            revision=1,
            updated_at=datetime.now(UTC),
        )
    with pytest.raises(ValueError):
        NovaDialogueStateView(last_assistant_offer_kinds=["plan"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NovaDialogueStateView(open_loops=["private"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NovaObservedMemoryView(
            public_id="00000000-0000-0000-0000-000000000001",
            category="preference",
            value="short answers",
            salience=3,
            revision=1,
            updated_at=datetime.now(),
        )
    with pytest.raises(ValueError):
        NovaBrainFence(
            owner_id=1,
            telegram_user_id=2,
            chat_id=3,
            access_tier="subscriber",
            access_version=1,
            state_revision=0,
            memory_revision=object(),  # type: ignore[arg-type]
        )
    projection = NovaBrainProjection(NovaDialogueStateView(), ())
    assert projection.provider_payload() == {"working_dialogue_state": {"revision": 0}}


def test_nova_brain_projection_drops_whole_optional_fields_to_hard_byte_budget():
    service = NovaBrainService.__new__(NovaBrainService)
    service.context_max_bytes = 1024
    state = NovaDialogueStateView(
        active_topic="т" * 200,
        current_user_goal="ц" * 300,
        last_assistant_offer="п" * 600,
        last_assistant_offer_kinds=("plan",),
        unresolved_question="в" * 300,
        requested_action="plan",
        open_loops=tuple("о" * 200 for _ in range(5)),
        revision=9,
    )

    projection = service._fit_projection(state, ())

    assert projection.payload_bytes <= 1024
    assert projection.working_state.revision == 9
    assert projection.working_state.active_topic == "т" * 200
    payload = projection.provider_payload()
    assert len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) < 1200


async def test_nova_brain_policy_matrix_admin_subscriber_guest(db):
    admin = await _actor(db, 810_011, tier="admin")
    subscriber = await _actor(db, 810_012, tier="subscriber")
    guest = await _actor(db, 810_013, tier="guest")
    service = NovaBrainService(db)
    for actor, policy, expected in (
        (admin, NovaBrainPolicy(enabled=True, admin_only=True), "ready"),
        (subscriber, NovaBrainPolicy(enabled=True, admin_only=True), "access_changed"),
        (subscriber, NovaBrainPolicy(enabled=True, admin_only=False), "ready"),
        (guest, NovaBrainPolicy(enabled=True, admin_only=False), "access_changed"),
        (admin, NovaBrainPolicy(enabled=False, admin_only=False), "access_changed"),
    ):
        result = await service.snapshot(
            telegram_actor_id=actor.telegram_id,
            chat_id=actor.telegram_id,
            expected_tier=actor.access_tier,
            expected_access_version=actor.access_version,
            current_text="Тема",
            policy=policy,
        )
        assert result.status == expected


async def test_nova_brain_forget_recovery_is_exact_generation_and_preserves_replacement(db):
    actor = await _actor(db, 810_014)
    memory = NovaObservedMemoryView(
        public_id="00000000-0000-0000-0000-000000000014",
        category="preference",
        value="response_length=short",
        salience=5,
        revision=1,
        updated_at=datetime.now(UTC),
    )
    store = NovaBrainForgetStore()
    stage = await store.stage(
        memory,
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        access_version=actor.access_version,
    )
    bound = await store.bind(stage, canonical_message_id=414)
    assert bound is not None
    capability = await store.peek(
        bound.callback_data("confirm"),
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        canonical_message_id=414,
    )
    assert capability is not None and await store.consume(capability)
    recovered = await store.stage_recovery(capability)
    assert recovered is not None
    assert not await store.consumed_screen_is_current(capability)
    assert await store.peek(
        recovered.callback_data("confirm"),
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        canonical_message_id=414,
    )
    assert await store.stage_recovery(capability) is None


async def test_nova_brain_forget_store_is_bounded_without_evicting_live_screen(db):
    actor = await _actor(db, 810_018)
    memory = NovaObservedMemoryView(
        public_id="00000000-0000-0000-0000-000000000018",
        category="preference",
        value="response_length=short",
        salience=5,
        revision=1,
        updated_at=datetime.now(UTC),
    )
    store = NovaBrainForgetStore(max_screens=1)
    first = await store.stage(
        memory,
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        access_version=actor.access_version,
    )
    bound = await store.bind(first, canonical_message_id=818)
    assert bound is not None
    with pytest.raises(RuntimeError, match="capacity"):
        await store.stage(
            memory,
            owner_id=actor.id,
            telegram_user_id=actor.telegram_id,
            chat_id=actor.telegram_id,
            access_version=actor.access_version,
        )
    assert await store.peek(
        bound.callback_data("confirm"),
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        canonical_message_id=818,
    )


async def test_nova_brain_bounded_history_pruning_is_exactly_compensatable(db):
    actor = await _actor(db, 810_019)
    original_ids = (
        "00000000-0000-0000-0000-000000000191",
        "00000000-0000-0000-0000-000000000192",
    )
    async with db.session() as session:
        for index, public_id in enumerate(original_ids, start=1):
            value = f"старое забытое сведение {index}"
            session.add(
                NovaObservedMemory(
                    public_id=public_id,
                    owner_id=actor.id,
                    category="theme",
                    normalized_value=value,
                    content_fingerprint=sha256(value.encode()).hexdigest(),
                    source_kind="conversation",
                    source_session_id=1,
                    source_message_id=index,
                    source_receipt=sha256(public_id.encode()).hexdigest(),
                    status="forgotten",
                    salience=1,
                    revision=2,
                    updated_at=datetime(2026, 8, index, tzinfo=UTC),
                )
            )
    service = NovaBrainService(db, max_memories=1)
    snapshot = await service.snapshot(
        telegram_actor_id=actor.telegram_id,
        chat_id=actor.telegram_id,
        expected_tier=actor.access_tier,
        expected_access_version=actor.access_version,
        current_text="Я всегда предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert snapshot.fence is not None
    source = await _source(
        db,
        actor,
        actor.telegram_id,
        "Я всегда предпочитаю короткие ответы",
        "Поняла",
    )
    receipt = await service.apply_turn(
        snapshot.fence,
        source,
        state_update=None,
        memory_candidate=_response_length_memory("Я предпочитаю короткие ответы"),
        user_text="Я предпочитаю короткие ответы",
        policy=_policy(),
    )
    assert receipt is not None
    async with db.sessions() as session:
        rows = list((await session.scalars(select(NovaObservedMemory))).all())
        assert len(rows) == 2
        assert sum(row.status == "active" for row in rows) == 1
    assert await service.compensate_turn(receipt)
    async with db.sessions() as session:
        rows = list(
            (
                await session.scalars(
                    select(NovaObservedMemory).order_by(NovaObservedMemory.public_id)
                )
            ).all()
        )
        assert [row.public_id for row in rows] == list(original_ids)
        assert all(row.status == "forgotten" for row in rows)
