import asyncio
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace

import pytest

from future_self.dates import DateOption, DateResolution, DateResolver
from future_self.nova_companion_flow import (
    CAPTURE_DATE_ACTIONS,
    NOVA_COMPANION_CALLBACK_PREFIX,
    CaptureSuggestion,
    ExplicitCaptureClassifier,
    NovaAddressClassifier,
    NovaAddressKind,
    NovaCompanionCaptureStore,
    NovaCompanionCaptureTemporal,
    NovaCompanionDiscourseAnchor,
    NovaCompanionDiscourseReducer,
    NovaCompanionPolicy,
    NovaCompanionReminderCandidate,
    NovaCompanionReminderStore,
    should_offer_capture,
    validate_capture_suggestion,
)


@pytest.mark.parametrize(
    "reply",
    [
        "ооо, было бы круто)",
        "давай",
        "ну давай",
        "подбери",
        "ну давай подбери)",
        "да, хочу",
        "помоги тогда",
    ],
)
def test_discourse_reducer_binds_weak_assent_to_one_immediate_offer(reply):
    anchor = NovaCompanionDiscourseReducer.reduce(
        reply,
        [
            {"role": "user", "content": "Как не улетать в мысли?"},
            {
                "role": "assistant",
                "content": "Если хочешь, можем дальше просто подобрать удобный способ.",
            },
        ],
    )

    assert anchor is not None
    assert anchor.status == "single"
    assert anchor.offer_kinds == ("method",)
    assert anchor.provider_payload() == {
        "status": "single",
        "offer_kinds": ["method"],
        "offer_text": "Если хочешь, можем дальше просто подобрать удобный способ.",
    }
    assert "offer_text" not in repr(anchor)


def test_discourse_reducer_marks_two_real_offers_ambiguous():
    anchor = NovaCompanionDiscourseReducer.reduce(
        "давай",
        [
            {
                "role": "assistant",
                "content": ("Могу подобрать удобный способ и могу помочь настроить напоминание."),
            }
        ],
    )

    assert anchor is not None
    assert anchor.status == "ambiguous"
    assert anchor.offer_kinds == ("method", "reminder_setup")


def test_discourse_reducer_keeps_three_real_offers_fail_closed_and_bounded():
    anchor = NovaCompanionDiscourseReducer.reduce(
        "да, хочу",
        [
            {
                "role": "assistant",
                "content": (
                    "Могу подобрать удобный способ, могу выбрать упражнение и могу помочь "
                    "настроить напоминание."
                ),
            }
        ],
    )

    assert anchor is not None
    assert anchor.status == "ambiguous"
    assert anchor.offer_kinds == ("method", "exercise", "reminder_setup")


@pytest.mark.parametrize(
    "assistant_text",
    [
        "Я не могу подобрать подходящий способ.",
        "Не можем выбрать упражнение.",
        "Я не могу предложить упражнение без деталей.",
        "Не могу помочь настроить напоминание.",
        "Не могу составить план.",
        "Ты спрашивал, могу ли я подобрать способ.",
    ],
)
def test_discourse_reducer_rejects_negated_and_reported_assistant_offers(assistant_text):
    assert (
        NovaCompanionDiscourseReducer.reduce(
            "давай",
            [{"role": "assistant", "content": assistant_text}],
        )
        is None
    )


@pytest.mark.parametrize(
    ("assistant_text", "expected_kind"),
    [
        ("Если хочешь, можем подобрать удобный способ.", "method"),
        ("Могу предложить короткое упражнение.", "exercise"),
        ("Могу помочь настроить настоящее напоминание.", "reminder_setup"),
        ("Можем составить простой план.", "plan"),
    ],
)
def test_discourse_reducer_accepts_only_affirmative_assistant_offers(assistant_text, expected_kind):
    anchor = NovaCompanionDiscourseReducer.reduce(
        "давай",
        [{"role": "assistant", "content": assistant_text}],
    )

    assert anchor is not None
    assert anchor.offer_kinds == (expected_kind,)


def test_discourse_anchor_rejects_forged_offer_kind():
    with pytest.raises(ValueError, match="Invalid companion discourse offer"):
        NovaCompanionDiscourseAnchor(
            status="single",
            offer_kinds=("forged",),  # type: ignore[arg-type]
            offer_text="Могу подобрать способ.",
        )


@pytest.mark.parametrize(
    ("reply", "messages"),
    [
        ("Расскажи про новую тему", [{"role": "assistant", "content": "Могу выбрать способ."}]),
        ("давай", [{"role": "user", "content": "Могу выбрать способ."}]),
        (
            "давай",
            [
                {"role": "assistant", "content": "Могу подобрать удобный способ."},
                {"role": "user", "content": "Новая тема"},
            ],
        ),
        ("давай", [{"role": "assistant", "content": "Могу помочь."}]),
    ],
)
def test_discourse_reducer_fails_closed_without_immediate_single_offer(reply, messages):
    assert NovaCompanionDiscourseReducer.reduce(reply, messages) is None


def _reminder_candidate() -> NovaCompanionReminderCandidate:
    resolver = DateResolver(now_provider=lambda: datetime(2026, 8, 21, 9, 0, tzinfo=UTC))
    evidence = "Не забыть бы мне завтра на стрижку в 19:00"
    resolution = resolver.resolve(
        evidence,
        "Europe/Moscow",
        now=datetime(2026, 8, 21, 9, 0, tzinfo=UTC),
    )
    return NovaCompanionReminderCandidate(
        title="стрижку",
        schedule_wording="завтра",
        evidence=evidence,
        timezone="Europe/Moscow",
        temporal=NovaCompanionCaptureTemporal(
            timezone="Europe/Moscow",
            resolution=resolution,
            local_time=time(19, 0),
        ),
    )


@pytest.mark.parametrize(
    ("text", "kind", "response"),
    [
        ("Нова", NovaAddressKind.WAKE, "Да, я здесь 🙂"),
        ("Nova", NovaAddressKind.WAKE, "Да, я здесь 🙂"),
        ("Нова ты тут?", NovaAddressKind.PRESENCE, "Да, я здесь 🙂"),
        ("Nova, ты здесь?", NovaAddressKind.PRESENCE, "Да, я здесь 🙂"),
        (
            "Ты Нова?",
            NovaAddressKind.IDENTITY,
            "Да, я Nova. Я рядом — о чём хочешь поговорить?",
        ),
        (
            "Are you Nova?",
            NovaAddressKind.IDENTITY,
            "Да, я Nova. Я рядом — о чём хочешь поговорить?",
        ),
    ],
)
def test_nova_address_local_wake_and_identity(
    text: str,
    kind: NovaAddressKind,
    response: str,
) -> None:
    result = NovaAddressClassifier.classify(text)

    assert result.kind is kind
    assert result.is_local_response is True
    assert result.local_response == response
    assert result.content is None


@pytest.mark.parametrize("source", ["text", "voice"])
def test_nova_vocative_has_text_voice_parity(source: str) -> None:
    # Both transport paths pass their resulting text to this exact classifier.
    result = NovaAddressClassifier.classify(
        "Нова, я постоянно забываю о главном, из-за каждодневной суеты"
    )

    assert source in {"text", "voice"}
    assert result.kind is NovaAddressKind.VOCATIVE
    assert result.content == "я постоянно забываю о главном, из-за каждодневной суеты"
    assert result.local_response is None


@pytest.mark.parametrize(
    "text",
    [
        "Иванова написала сообщение",
        "Моя новая задача",
        "Ты тут?",
        "Что умеет Nova в этом боте?",
        "",
        None,
    ],
)
def test_nova_address_does_not_broadly_capture_content(text: object) -> None:
    assert NovaAddressClassifier.classify(text).kind is NovaAddressKind.NONE


async def test_reminder_offer_store_is_opaque_exact_single_winner_and_ttl_fenced() -> None:
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    store = NovaCompanionReminderStore(ttl=timedelta(minutes=5), max_capabilities=8)
    fence = object()
    staged = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="subscriber",
        access_version=4,
        context_fence=fence,
        memory_revision="opaque-memory-revision",
        now=now,
    )
    assert staged is not None and staged.is_bound is False
    assert all(
        callback.startswith("nrem:")
        and "стриж" not in callback
        and "завтра" not in callback
        and "19:00" not in callback
        for callback in staged.callbacks.values()
    )
    assert (
        await store.peek_bound_identity(
            staged.callback_data("accept"),
            owner_id=1,
            telegram_user_id=2,
            chat_id=3,
            canonical_message_id=10,
            now=now,
        )
        is None
    )

    bound = await store.bind(staged, canonical_message_id=10, now=now)
    assert bound is not None
    accept = await store.peek_bound_identity(
        bound.callback_data("accept"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=10,
        now=now,
    )
    decline = await store.peek_bound_identity(
        bound.callback_data("not_now"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=10,
        now=now,
    )
    assert accept is not None and decline is not None
    first, second = await asyncio.gather(
        store.consume(accept, now=now), store.consume(decline, now=now)
    )
    assert sorted((first, second)) == [False, True]
    assert await store.consume(accept, now=now) is False
    assert await store.consumed_screen_is_current(accept, now=now) is True

    expiring = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="subscriber",
        access_version=4,
        context_fence=fence,
        memory_revision=None,
        now=now,
    )
    assert expiring is not None
    expiring = await store.bind(expiring, canonical_message_id=11, now=now)
    assert expiring is not None
    token = await store.peek_bound_identity(
        expiring.callback_data("accept"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=11,
        now=now,
    )
    assert token is not None
    assert await store.consume(token, now=now + timedelta(minutes=5)) is False


async def test_reminder_offer_consume_samples_runtime_clock_after_waiting_for_lock(
    monkeypatch,
) -> None:
    issued_at = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    expires_at = issued_at + timedelta(seconds=1)
    store = NovaCompanionReminderStore(ttl=timedelta(seconds=1), max_capabilities=8)
    staged = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="subscriber",
        access_version=4,
        context_fence=object(),
        memory_revision=None,
        now=issued_at,
    )
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=10, now=issued_at)
    assert bound is not None
    token = await store.peek_bound_identity(
        bound.callback_data("accept"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=10,
        now=issued_at,
    )
    assert token is not None

    await store._lock.acquire()
    consume_task = asyncio.create_task(store.consume(token))
    await asyncio.sleep(0)
    assert consume_task.done() is False
    monkeypatch.setattr(
        NovaCompanionCaptureStore,
        "_utc",
        staticmethod(lambda _value: expires_at),
    )
    store._lock.release()

    assert await consume_task is False
    assert store._capabilities == {}
    assert store._screens == {}


async def test_reminder_offer_exact_expiry_rejects_old_and_preserves_replacement() -> None:
    issued_at = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    expires_at = issued_at + timedelta(seconds=1)
    store = NovaCompanionReminderStore(ttl=timedelta(seconds=1), max_capabilities=8)
    old_screen = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="subscriber",
        access_version=4,
        context_fence=object(),
        memory_revision=None,
        now=issued_at,
    )
    assert old_screen is not None
    old_screen = await store.bind(old_screen, canonical_message_id=10, now=issued_at)
    assert old_screen is not None
    old_token = await store.peek_bound_identity(
        old_screen.callback_data("accept"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=10,
        now=issued_at,
    )
    assert old_token is not None

    replacement = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="subscriber",
        access_version=4,
        context_fence=object(),
        memory_revision=None,
        now=issued_at + timedelta(microseconds=1),
    )
    assert replacement is not None
    replacement = await store.bind(
        replacement,
        canonical_message_id=11,
        now=issued_at + timedelta(microseconds=1),
    )
    assert replacement is not None

    assert await store.consume(old_token, now=expires_at) is False
    live = await store.peek_bound_identity(
        replacement.callback_data("accept"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=11,
        now=expires_at,
    )
    assert live is not None
    assert await store.consume(live, now=expires_at) is True


async def test_reminder_offer_old_callback_cannot_consume_replacement_generation() -> None:
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    store = NovaCompanionReminderStore(max_capabilities=8)
    first = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="admin",
        access_version=1,
        context_fence=object(),
        memory_revision=None,
        now=now,
    )
    assert first is not None
    first = await store.bind(first, canonical_message_id=20, now=now)
    assert first is not None
    old = await store.peek_bound_identity(
        first.callback_data("accept"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=20,
        now=now,
    )
    assert old is not None
    second = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="admin",
        access_version=1,
        context_fence=object(),
        memory_revision=None,
        now=now,
    )
    assert second is not None
    second = await store.bind(second, canonical_message_id=20, now=now)
    assert second is not None
    assert await store.consume(old, now=now) is False
    assert await store.consumed_screen_is_current(old, now=now) is False
    live = await store.active(
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="admin",
        access_version=1,
        now=now,
    )
    assert live is not None and live.screen_order == second.screen_order


async def test_reminder_offer_stage_failure_does_not_evict_live_generation(
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    store = NovaCompanionReminderStore(max_capabilities=2)
    first = await store.stage(
        _reminder_candidate(),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        access_tier="subscriber",
        access_version=4,
        context_fence=object(),
        memory_revision=None,
        now=now,
    )
    assert first is not None
    first = await store.bind(first, canonical_message_id=10, now=now)
    assert first is not None
    live_token = await store.peek_bound_identity(
        first.callback_data("accept"),
        owner_id=1,
        telegram_user_id=2,
        chat_id=3,
        canonical_message_id=10,
        now=now,
    )
    assert live_token is not None

    values = iter(("new-screen", "new-accept"))

    def fail_mid_batch(_size: int) -> str:
        try:
            return next(values)
        except StopIteration:
            raise OSError("PRIVATE_ENTROPY_FAILURE") from None

    monkeypatch.setattr(
        "future_self.nova_companion_flow.secrets.token_urlsafe",
        fail_mid_batch,
    )
    with pytest.raises(OSError, match="PRIVATE_ENTROPY_FAILURE"):
        await store.stage(
            _reminder_candidate(),
            owner_id=1,
            telegram_user_id=2,
            chat_id=3,
            access_tier="subscriber",
            access_version=4,
            context_fence=object(),
            memory_revision=None,
            now=now,
        )

    assert await store.consume(live_token, now=now) is True


def test_address_result_repr_hides_vocative_content() -> None:
    secret = "PRIVATE_VOCATIVE_PAYLOAD"
    result = NovaAddressClassifier.classify(f"Nova, {secret}")

    assert result.content == secret
    assert secret not in repr(result)


@pytest.mark.parametrize(
    ("text", "kind", "content", "references_context"),
    [
        ("Запиши это", "note", None, True),
        ("Запиши это как заметку", "note", None, True),
        ("Запиши это.", "note", None, True),
        ("Добавь как заметку.", "note", None, True),
        ("Нова, запиши это.", "note", None, True),
        ("Добавь как заметку", "note", None, True),
        ("Добавь как идею сделать тихую комнату", "idea", "сделать тихую комнату", False),
        ("Создай задачу позвонить врачу", "task", "позвонить врачу", False),
        ("Создай желание поехать к морю", "desire", "поехать к морю", False),
        ("Нова, создай задачу позвонить врачу", "task", "позвонить врачу", False),
        (
            "Добавь в мысли: я лучше работаю, когда утром не читаю новости.",
            "note",
            "я лучше работаю, когда утром не читаю новости",
            False,
        ),
        ("добавь в ЗАМЕТКИ — одна цель", "note", "одна цель", False),
        ("Нова, добавь в идеи: тихая комната", "idea", "тихая комната", False),
        ("Добавь в задачи позвонить врачу", "task", "позвонить врачу", False),
    ],
)
def test_explicit_capture_classifier_matches_only_clear_commands(
    text: str,
    kind: str,
    content: str | None,
    references_context: bool,
) -> None:
    result = ExplicitCaptureClassifier.classify(text)

    assert result is not None
    assert result.kind == kind
    assert result.content == content
    assert result.references_context is references_context


@pytest.mark.parametrize("source", ["text", "voice"])
@pytest.mark.parametrize("address", ["", "Нова, ", "Nova, ", "Нова ", "Nova "])
@pytest.mark.parametrize(
    ("command", "kind", "content", "references_context"),
    [
        ("Добавь задачу позвонить врачу", "task", "позвонить врачу", False),
        ("Добавь идею открыть кофейню", "idea", "открыть кофейню", False),
        ("Сохрани желание увидеть океан", "desire", "увидеть океан", False),
        ("Запиши заметку купить молоко", "note", "купить молоко", False),
        ("Запиши задачу позвонить врачу", "task", "позвонить врачу", False),
        (
            "Создай задачу завтра в 10:00 позвонить врачу",
            "task",
            "завтра в 10:00 позвонить врачу",
            False,
        ),
        ("Сохрани это", "note", None, True),
    ],
)
def test_explicit_capture_classifier_supports_legacy_direct_commands_with_transport_parity(
    source: str,
    address: str,
    command: str,
    kind: str,
    content: str | None,
    references_context: bool,
) -> None:
    # Text and transcribed voice use the same deterministic classifier.
    result = ExplicitCaptureClassifier.classify(f"{address}{command}")

    assert source in {"text", "voice"}
    assert result is not None
    assert result.kind == kind
    assert result.content == content
    assert result.references_context is references_context


@pytest.mark.parametrize(
    "text",
    [
        "Я постоянно забываю о главном, из-за каждодневной суеты",
        "Мне нужно позвонить врачу",
        "Я хочу поехать к морю",
        "Кажется, это интересная идея",
        "Как записать задачу?",
        "Напомни завтра в 10:00 позвонить врачу",
        "Открой мои заметки",
        "Да не, я хотел просто пообщаться",
        "Я добавил задачу позвонить врачу",
        "Я добавила задачу позвонить врачу",
        "Кажется, это хорошая идея",
        "Как добавить задачу?",
    ],
)
def test_explicit_capture_classifier_leaves_reflection_reminder_and_navigation_alone(
    text: str,
) -> None:
    assert ExplicitCaptureClassifier.classify(text) is None


def test_explicit_capture_repr_hides_content() -> None:
    secret = "PRIVATE_EXPLICIT_CAPTURE"
    result = ExplicitCaptureClassifier.classify(f"Создай задачу {secret}")

    assert result is not None and result.content == secret
    assert secret not in repr(result)


@pytest.mark.parametrize(
    "text",
    [
        "Я постоянно забываю о главном, из-за каждодневной суеты",
        "Да не, я хотел просто пообщаться",
        "Не сохраняй эту мысль",
        "Мне грустно",
        "Наверное, я мог бы чаще гулять",
        "Как лучше отдыхать?",
        "Привет!",
        "Сегодня идёт дождь",
        "Я обычно пью чай утром",
        "Нова",
        "Ты Nova?",
    ],
)
def test_capture_offer_is_suppressed_for_non_concrete_conversation(text: str) -> None:
    assert should_offer_capture(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Хочу в сентябре пройти курс первой помощи",
        "Нужно до пятницы отправить договор Марине",
        "Идея: сделать тихую комнату для чтения",
    ],
)
def test_capture_offer_allows_concrete_non_command_content(text: str) -> None:
    assert should_offer_capture(text) is True


def test_suggestion_validation_is_bounded_normalized_and_privacy_safe() -> None:
    suggestion = validate_capture_suggestion(
        kind="task",
        title="  Отправить   договор Марине  ",
        next_step="  Найти адрес  ",
        user_text="Нужно до пятницы отправить договор Марине и найти адрес",
    )

    assert suggestion is not None
    assert suggestion.title == "Отправить договор Марине"
    assert suggestion.next_step == "Найти адрес"
    assert suggestion.fingerprint not in repr(suggestion)
    assert suggestion.title not in repr(suggestion)
    assert suggestion.next_step not in repr(suggestion)
    assert (
        CaptureSuggestion("task", "ОТПРАВИТЬ ДОГОВОР МАРИНЕ").fingerprint == suggestion.fingerprint
    )


@pytest.mark.parametrize(
    ("title", "next_step"),
    [
        ("Позвонить нотариусу", None),
        ("Отправить договор Марине", "Выдуманный следующий шаг"),
    ],
)
def test_suggestion_validation_rejects_ungrounded_provider_fields(
    title: str, next_step: str | None
) -> None:
    assert (
        validate_capture_suggestion(
            kind="task",
            title=title,
            next_step=next_step,
            user_text="Нужно до пятницы отправить договор Марине",
        )
        is None
    )


@pytest.mark.parametrize(
    "values",
    [
        {"kind": "memory", "title": "Секрет", "next_step": None},
        {"kind": "task", "title": "x" * 121, "next_step": None},
        {"kind": "task", "title": "Задача", "next_step": "x" * 241},
        {"kind": "task", "title": "Задача\x00", "next_step": None},
        {"kind": "task", "title": 42, "next_step": None},
    ],
)
def test_untrusted_invalid_suggestion_fails_closed(values: dict[str, object]) -> None:
    assert (
        validate_capture_suggestion(
            **values,
            user_text="Нужно до пятницы отправить договор Марине",
        )
        is None
    )


def test_companion_policy_is_fail_closed_and_version_fenced() -> None:
    admin = SimpleNamespace(access_tier="admin", access_version=4, telegram_id=999)
    subscriber = SimpleNamespace(access_tier="subscriber", access_version=4)

    assert NovaCompanionPolicy(enabled=False).allows_actor(admin) is False
    pilot = NovaCompanionPolicy(enabled=True)
    assert pilot.allows_actor(admin, expected_access_version=4) is True
    assert pilot.allows_actor(admin, expected_access_version=5) is False
    assert pilot.allows_actor(subscriber) is False
    rollout = NovaCompanionPolicy(enabled=True, admin_only=False)
    assert rollout.allows_actor(subscriber) is True
    assert rollout.allows_tier("guest") is False
    assert "telegram_id" not in repr(pilot)


def suggestion() -> CaptureSuggestion:
    return CaptureSuggestion(
        "task",
        "Отправить договор Марине",
        "Найти адрес",
    )


def binding() -> dict[str, int]:
    return {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "access_version": 4,
    }


def resolved_temporal() -> NovaCompanionCaptureTemporal:
    now = datetime(2026, 8, 18, 7, tzinfo=UTC)
    resolution = DateResolver().resolve(
        "завтра в 10:00 позвонить врачу",
        "Europe/Moscow",
        now=now,
    )
    return NovaCompanionCaptureTemporal(
        timezone="Europe/Moscow",
        resolution=resolution,
        local_time=time(10),
    )


def conflict_temporal() -> NovaCompanionCaptureTemporal:
    resolution = DateResolution(
        status="conflict",
        target_date=date(2026, 8, 20),
        stated_weekday="среда",
        actual_weekday="четверг",
        options=[
            DateOption(value=date(2026, 8, 19), weekday="среда"),
            DateOption(value=date(2026, 8, 20), weekday="четверг"),
        ],
    )
    return NovaCompanionCaptureTemporal(
        timezone="Europe/Moscow",
        resolution=resolution,
        local_time=time(10),
    )


def test_capture_temporal_payload_is_validated_frozen_and_privacy_safe() -> None:
    resolution = DateResolution(
        status="resolved",
        target_date=date(2026, 8, 19),
        actual_weekday="среда",
    )
    payload = NovaCompanionCaptureTemporal(
        timezone="Europe/Moscow",
        resolution=resolution,
        local_time=time(10),
    )

    resolution.target_date = date(2030, 1, 1)
    first_copy = payload.resolution
    first_copy.target_date = date(2031, 1, 1)

    assert payload.timezone == "Europe/Moscow"
    assert payload.local_time == time(10)
    assert payload.resolution.target_date == date(2026, 8, 19)
    rendered = repr(payload)
    assert "Europe/Moscow" not in rendered
    assert "2026" not in rendered
    assert "10:00" not in rendered


@pytest.mark.parametrize(
    ("timezone", "resolution", "local_time"),
    [
        ("Not/A_Timezone", DateResolution(status="resolved"), time(10)),
        ("Europe/Moscow", DateResolution(status="none"), time(10)),
        ("Europe/Moscow", DateResolution(status="resolved"), time(10)),
        (
            "Europe/Moscow",
            DateResolution(
                status="conflict",
                target_date=date(2026, 8, 20),
                stated_weekday="среда",
                actual_weekday="четверг",
            ),
            time(10),
        ),
        (
            "Europe/Moscow",
            DateResolution(
                status="resolved",
                target_date=date(2026, 8, 19),
                actual_weekday="среда",
            ),
            time(10, tzinfo=UTC),
        ),
    ],
)
def test_capture_temporal_payload_rejects_invalid_date_resolver_outcomes(
    timezone: str,
    resolution: DateResolution,
    local_time: time,
) -> None:
    with pytest.raises(ValueError):
        NovaCompanionCaptureTemporal(
            timezone=timezone,
            resolution=resolution,
            local_time=local_time,
        )


@pytest.mark.asyncio
async def test_temporal_payload_survives_stage_bind_capability_and_exact_recovery() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    temporal = conflict_temporal()
    staged = await store.stage(
        suggestion(),
        raw_text="Задача с конфликтующей датой",
        temporal=temporal,
        now=now,
        **binding(),
    )
    assert staged is not None
    assert staged.temporal == temporal
    assert set(staged.callbacks) == {"add", "not_now"}
    assert "Europe/Moscow" not in repr(staged)

    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None and bound.temporal == temporal
    consumed = await store.claim(
        bound.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert consumed is not None and consumed.temporal == temporal
    assert "Europe/Moscow" not in repr(consumed)

    recovery = await store.stage_recovery(
        consumed,
        actions=CAPTURE_DATE_ACTIONS,
        now=now,
    )
    assert recovery is not None
    assert recovery.canonical_message_id == 301
    assert recovery.temporal == temporal
    assert tuple(recovery.callbacks) == CAPTURE_DATE_ACTIONS
    for action, callback_data in recovery.callbacks.items():
        assert action not in callback_data
        exact = await store.peek(
            callback_data,
            canonical_message_id=301,
            expected_action=action,
            now=now,
            **binding(),
        )
        assert exact is not None and exact.temporal == temporal


@pytest.mark.asyncio
async def test_custom_date_actions_have_one_whole_screen_winner_without_suppression() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(
        suggestion(),
        raw_text="Задача с выбором даты",
        temporal=conflict_temporal(),
        actions=CAPTURE_DATE_ACTIONS,
        now=now,
        **binding(),
    )
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None

    winners = await asyncio.gather(
        *(
            store.claim(
                bound.callback_data(action),
                canonical_message_id=301,
                expected_action=action,
                now=now,
                **binding(),
            )
            for action in CAPTURE_DATE_ACTIONS
        )
    )

    winner = next(result for result in winners if result is not None)
    assert sum(result is not None for result in winners) == 1
    assert winner.action in CAPTURE_DATE_ACTIONS
    assert store._capabilities == {}
    assert store._screens == {}
    if winner.action != "not_now":
        assert not await store.is_suppressed(
            fingerprint=suggestion().fingerprint,
            now=now,
            **binding(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["date_first", "date_second"])
async def test_date_choice_actions_never_mark_suggestion_suppressed(action: str) -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(
        suggestion(),
        raw_text="Задача с выбором даты",
        temporal=conflict_temporal(),
        actions=CAPTURE_DATE_ACTIONS,
        now=now,
        **binding(),
    )
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None

    consumed = await store.claim(
        bound.callback_data(action),  # type: ignore[arg-type]
        canonical_message_id=301,
        expected_action=action,  # type: ignore[arg-type]
        now=now,
        **binding(),
    )

    assert consumed is not None and consumed.action == action
    assert not await store.is_suppressed(
        fingerprint=suggestion().fingerprint,
        now=now,
        **binding(),
    )


@pytest.mark.asyncio
async def test_invalid_custom_actions_are_rejected_atomically_for_stage_and_recovery() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(
        suggestion(),
        raw_text="Исходная задача",
        temporal=conflict_temporal(),
        now=now,
        **binding(),
    )
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None
    original_count = len(store._capabilities)

    invalid_actions = (
        (),
        ("add", "add"),
        ("date_first", "date_second"),
        ("date_second", "date_first", "not_now"),
        ("date_first", "date_second", "add"),
        ["add", "not_now"],
    )
    for actions in invalid_actions:
        with pytest.raises(ValueError):
            await store.stage(
                CaptureSuggestion("task", "Другая задача"),
                raw_text="Другая задача",
                actions=actions,  # type: ignore[arg-type]
                now=now,
                **binding(),
            )
        assert len(store._capabilities) == original_count

    with pytest.raises(ValueError):
        await store.stage(
            CaptureSuggestion("task", "Другая задача"),
            raw_text="Другая задача",
            temporal=resolved_temporal(),
            actions=CAPTURE_DATE_ACTIONS,
            now=now,
            **binding(),
        )
    assert len(store._capabilities) == original_count

    consumed = await store.claim(
        bound.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert consumed is not None
    with pytest.raises(ValueError):
        await store.stage_recovery(
            consumed,
            actions=("date_first", "not_now"),  # type: ignore[arg-type]
            now=now,
        )
    assert store._capabilities == {}
    assert store._screens == {}

    recovery = await store.stage_recovery(
        consumed,
        actions=CAPTURE_DATE_ACTIONS,
        now=now,
    )
    assert recovery is not None
    assert tuple(recovery.callbacks) == CAPTURE_DATE_ACTIONS


@pytest.mark.asyncio
async def test_staged_callbacks_are_opaque_and_unclaimable_until_exact_bind() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    raw_text = "Нужно до пятницы отправить договор Марине"
    staged = await store.stage(suggestion(), raw_text=raw_text, now=now, **binding())
    assert staged is not None

    assert staged.is_bound is False
    assert set(staged.callbacks) == {"add", "not_now"}
    for action, callback_data in staged.callbacks.items():
        assert callback_data.startswith(NOVA_COMPANION_CALLBACK_PREFIX)
        assert action not in callback_data
        assert suggestion().title not in callback_data
        assert raw_text not in callback_data
        assert len(callback_data.encode("utf-8")) <= 64
        assert (
            await store.claim(
                callback_data,
                canonical_message_id=301,
                expected_action=action,
                now=now,
                **binding(),
            )
            is None
        )

    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None and bound.is_bound
    claim = await store.claim(
        bound.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert claim is not None
    assert claim.suggestion == suggestion()
    assert claim.raw_text == raw_text
    assert claim.canonical_message_id == 301
    assert claim.action == "add"
    assert raw_text not in repr(claim)
    assert suggestion().title not in repr(claim)
    # One valid action spends the entire exact screen.
    assert (
        await store.claim(
            bound.callback_data("not_now"),
            canonical_message_id=301,
            expected_action="not_now",
            now=now,
            **binding(),
        )
        is None
    )


@pytest.mark.asyncio
async def test_wrong_identity_canonical_and_action_do_not_spend_owner_capability() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(suggestion(), raw_text="Конкретная задача", now=now, **binding())
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None
    callback_data = bound.callback_data("add")

    wrong_bindings = [
        {**binding(), "owner_id": 2},
        {**binding(), "telegram_user_id": 102},
        {**binding(), "chat_id": 202},
    ]
    for wrong in wrong_bindings:
        assert (
            await store.claim(
                callback_data,
                canonical_message_id=301,
                expected_action="add",
                now=now,
                **wrong,
            )
            is None
        )
    assert (
        await store.claim(
            callback_data,
            canonical_message_id=999,
            expected_action="add",
            now=now,
            **binding(),
        )
        is None
    )
    assert (
        await store.claim(
            callback_data,
            canonical_message_id=301,
            expected_action="not_now",
            now=now,
            **binding(),
        )
        is None
    )
    assert (
        await store.claim(
            callback_data,
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_bound_identity_peek_supports_access_cleanup_without_spending_or_crossing_owner() -> (
    None
):
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(
        suggestion(), raw_text="РљРѕРЅРєСЂРµС‚РЅР°СЏ Р·Р°РґР°С‡Р°", now=now, **binding()
    )
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None
    callback_data = bound.callback_data("add")

    identity = await store.peek_bound_identity(
        callback_data,
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        expected_action="add",
        now=now,
    )
    assert identity is not None
    assert identity.access_version == 4
    for wrong in (
        {"owner_id": 2, "telegram_user_id": 101, "chat_id": 201},
        {"owner_id": 1, "telegram_user_id": 102, "chat_id": 201},
        {"owner_id": 1, "telegram_user_id": 101, "chat_id": 202},
    ):
        assert (
            await store.peek_bound_identity(
                callback_data,
                canonical_message_id=301,
                expected_action="add",
                now=now,
                **wrong,
            )
            is None
        )
    assert (
        await store.peek_bound_identity(
            callback_data,
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=999,
            expected_action="add",
            now=now,
        )
        is None
    )
    assert (
        await store.peek_bound_identity(
            callback_data,
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            expected_action="not_now",
            now=now,
        )
        is None
    )
    # Identity lookup is non-consuming and does not apply the changed access
    # value, so the handler can decide whether to neutralize the old screen.
    assert (
        await store.peek(
            callback_data,
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **binding(),
        )
        == identity
    )


@pytest.mark.asyncio
async def test_bound_identity_peek_cannot_observe_expired_or_replaced_generation() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(
        suggestion(), raw_text="РљРѕРЅРєСЂРµС‚РЅР°СЏ Р·Р°РґР°С‡Р°", now=now, **binding()
    )
    assert staged is not None
    old = await store.bind(staged, canonical_message_id=301, now=now)
    assert old is not None
    old_callback = old.callback_data("add")
    replacement_stage = await store.stage(
        CaptureSuggestion("idea", "РЎРґРµР»Р°С‚СЊ С‚РёС…СѓСЋ РєРѕРјРЅР°С‚Сѓ"),
        raw_text="РЎРІРµР¶Р°СЏ РёРґРµСЏ",
        now=now,
        **binding(),
    )
    assert replacement_stage is not None
    replacement = await store.bind(replacement_stage, canonical_message_id=301, now=now)
    assert replacement is not None

    assert (
        await store.peek_bound_identity(
            old_callback,
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )
    assert (
        await store.peek_bound_identity(
            replacement.callback_data("add"),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=replacement.expires_at,
        )
        is None
    )


@pytest.mark.asyncio
async def test_access_version_mismatch_revokes_only_exact_old_screen() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    old = await store.stage(suggestion(), raw_text="Старая задача", now=now, **binding())
    assert old is not None
    old_bound = await store.bind(old, canonical_message_id=301, now=now)
    assert old_bound is not None
    fresh_binding = {**binding(), "access_version": 5}
    fresh = await store.stage(
        CaptureSuggestion("idea", "Сделать тихую комнату"),
        raw_text="Свежая идея",
        now=now,
        **fresh_binding,
    )
    assert fresh is not None
    fresh_bound = await store.bind(fresh, canonical_message_id=302, now=now)
    assert fresh_bound is not None

    assert (
        await store.claim(
            old_bound.callback_data("add"),
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **fresh_binding,
        )
        is None
    )
    assert (
        await store.claim(
            fresh_bound.callback_data("add"),
            canonical_message_id=302,
            expected_action="add",
            now=now,
            **fresh_binding,
        )
        is not None
    )


@pytest.mark.asyncio
async def test_binding_new_generation_atomically_retires_old_and_stale_does_not_clear_fresh() -> (
    None
):
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    old_stage = await store.stage(suggestion(), raw_text="Старая задача", now=now, **binding())
    assert old_stage is not None
    old = await store.bind(old_stage, canonical_message_id=301, now=now)
    assert old is not None
    old_expected = await store.peek(
        old.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert old_expected is not None

    fresh_stage = await store.stage(
        CaptureSuggestion("idea", "Сделать тихую комнату"),
        raw_text="Свежая идея",
        now=now,
        **binding(),
    )
    assert fresh_stage is not None
    # Staging cannot make the currently rendered controls dead.
    assert (
        await store.peek(
            old.callback_data("add"),
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **binding(),
        )
        is not None
    )
    fresh = await store.bind(fresh_stage, canonical_message_id=301, now=now)
    assert fresh is not None
    assert await store.consume(old_expected, now=now) is False
    assert (
        await store.claim(
            fresh.callback_data("add"),
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_older_stage_cannot_replace_already_bound_newer_generation() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    older = await store.stage(suggestion(), raw_text="Старая задача", now=now, **binding())
    newer = await store.stage(
        CaptureSuggestion("idea", "Сделать тихую комнату"),
        raw_text="Свежая идея",
        now=now,
        **binding(),
    )
    assert older is not None and newer is not None
    newer_bound = await store.bind(newer, canonical_message_id=301, now=now)
    assert newer_bound is not None

    assert await store.bind(older, canonical_message_id=301, now=now) is None
    assert (
        await store.claim(
            newer_bound.callback_data("add"),
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_consumed_generation_tombstone_rejects_consumed_replacement_overwrite() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    first_stage = await store.stage(
        suggestion(),
        raw_text="Первая задача",
        now=now,
        **binding(),
    )
    assert first_stage is not None
    first = await store.bind(first_stage, canonical_message_id=301, now=now)
    assert first is not None
    first_capability = await store.peek(
        first.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert first_capability is not None
    assert await store.consume(first_capability, now=now)
    assert await store.consumed_screen_is_current(first_capability, now=now)

    delayed = await store.stage(
        CaptureSuggestion("idea", "Сделать тихую комнату"),
        raw_text="Отложенная идея",
        now=now,
        **binding(),
    )
    replacement_stage = await store.stage(
        CaptureSuggestion("desire", "Поехать к морю"),
        raw_text="Свежая замена",
        now=now,
        **binding(),
    )
    assert delayed is not None and replacement_stage is not None
    replacement = await store.bind(replacement_stage, canonical_message_id=301, now=now)
    assert replacement is not None
    replacement_capability = await store.peek(
        replacement.callback_data("not_now"),
        canonical_message_id=301,
        expected_action="not_now",
        now=now,
        **binding(),
    )
    assert replacement_capability is not None
    assert await store.consume(replacement_capability, now=now)

    assert not await store.consumed_screen_is_current(first_capability, now=now)
    assert await store.consumed_screen_is_current(replacement_capability, now=now)
    # The newer generation remains authoritative even after its capabilities
    # were consumed and its live screen was removed.
    assert await store.bind(delayed, canonical_message_id=301, now=now) is None
    assert not await store.consumed_screen_is_current(
        replacement_capability,
        now=replacement_capability.expires_at,
    )


@pytest.mark.asyncio
async def test_stage_recovery_atomically_publishes_bound_newer_generation() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(
        suggestion(),
        raw_text="РџРµСЂРІР°СЏ Р·Р°РґР°С‡Р°",
        now=now,
        **binding(),
    )
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None
    consumed = await store.peek(
        bound.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert consumed is not None

    # A live capability is not a consumed recovery source.
    assert await store.stage_recovery(consumed, now=now) is None
    assert await store.consume(consumed, now=now)
    recovered = await store.stage_recovery(consumed, now=now)

    assert recovered is not None and recovered.is_bound
    assert recovered.canonical_message_id == 301
    assert recovered.screen_order > consumed.screen_order
    assert recovered.suggestion == consumed.suggestion
    assert recovered.raw_text == consumed.raw_text
    assert not await store.consumed_screen_is_current(consumed, now=now)
    assert (
        await store.claim(
            recovered.callback_data("add"),
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **binding(),
        )
        is not None
    )
    # The exact consumed generation can recover at most once.
    assert await store.stage_recovery(consumed, now=now) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_first", [False, True])
async def test_stage_recovery_and_replacement_bind_have_one_deterministic_winner(
    replacement_first: bool,
) -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    first_stage = await store.stage(
        suggestion(),
        raw_text="РџРµСЂРІР°СЏ Р·Р°РґР°С‡Р°",
        now=now,
        **binding(),
    )
    assert first_stage is not None
    first = await store.bind(first_stage, canonical_message_id=301, now=now)
    assert first is not None
    consumed = await store.peek(
        first.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert consumed is not None
    assert await store.consume(consumed, now=now)

    replacement_stage = await store.stage(
        CaptureSuggestion("idea", "РЎРґРµР»Р°С‚СЊ С‚РёС…СѓСЋ РєРѕРјРЅР°С‚Сѓ"),
        raw_text="РЎРІРµР¶Р°СЏ РёРґРµСЏ",
        now=now,
        **binding(),
    )
    assert replacement_stage is not None

    # Queue both contenders behind the same held lock. asyncio.Lock wakes
    # waiters FIFO, making both possible interleavings deterministic.
    await store._lock.acquire()
    try:
        if replacement_first:
            replacement_task = asyncio.create_task(
                store.bind(replacement_stage, canonical_message_id=301, now=now)
            )
            await asyncio.sleep(0)
            recovery_task = asyncio.create_task(store.stage_recovery(consumed, now=now))
        else:
            recovery_task = asyncio.create_task(store.stage_recovery(consumed, now=now))
            await asyncio.sleep(0)
            replacement_task = asyncio.create_task(
                store.bind(replacement_stage, canonical_message_id=301, now=now)
            )
        await asyncio.sleep(0)
    finally:
        store._lock.release()
    recovered, replacement = await asyncio.gather(recovery_task, replacement_task)

    if replacement_first:
        assert replacement is not None
        assert recovered is None
        winner = replacement
    else:
        assert recovered is not None
        assert replacement is None
        winner = recovered
    assert winner.is_bound
    assert (
        await store.claim(
            winner.callback_data("add"),
            canonical_message_id=301,
            expected_action="add",
            now=now,
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_not_now_suppression_cannot_be_undone_by_recovery() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(
        suggestion(), raw_text="РљРѕРЅРєСЂРµС‚РЅР°СЏ Р·Р°РґР°С‡Р°", now=now, **binding()
    )
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None
    consumed = await store.peek(
        bound.callback_data("not_now"),
        canonical_message_id=301,
        expected_action="not_now",
        now=now,
        **binding(),
    )
    assert consumed is not None
    assert await store.consume(consumed, now=now)

    assert await store.stage_recovery(consumed, now=now) is None
    assert await store.is_suppressed(
        fingerprint=consumed.suggestion.fingerprint,
        now=now,
        **binding(),
    )


@pytest.mark.asyncio
async def test_consumed_generation_tombstones_are_bounded() -> None:
    store = NovaCompanionCaptureStore(max_capabilities=2)
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    consumed = []
    for offset in range(5):
        staged = await store.stage(
            suggestion(),
            raw_text=f"Задача {offset}",
            now=now,
            **binding(),
        )
        assert staged is not None
        screen = await store.bind(staged, canonical_message_id=400 + offset, now=now)
        assert screen is not None
        capability = await store.peek(
            screen.callback_data("add"),
            canonical_message_id=400 + offset,
            expected_action="add",
            now=now,
            **binding(),
        )
        assert capability is not None
        assert await store.consume(capability, now=now)
        consumed.append(capability)

    assert len(store._canonical_generations) == 2
    assert not await store.consumed_screen_is_current(consumed[0], now=now)
    assert await store.consumed_screen_is_current(consumed[-1], now=now)


@pytest.mark.asyncio
async def test_tombstone_bound_never_evicts_a_live_canonical_generation() -> None:
    store = NovaCompanionCaptureStore(max_capabilities=4)
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    live_stage = await store.stage(
        CaptureSuggestion("task", "Р–РёРІР°СЏ Р·Р°РґР°С‡Р°"),
        raw_text="Р–РёРІР°СЏ Р·Р°РґР°С‡Р°",
        now=now,
        **binding(),
    )
    assert live_stage is not None
    live = await store.bind(live_stage, canonical_message_id=301, now=now)
    assert live is not None
    live_capability = await store.peek(
        live.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert live_capability is not None

    for offset in range(4):
        transient_stage = await store.stage(
            CaptureSuggestion("idea", f"Р’СЂРµРјРµРЅРЅР°СЏ РёРґРµСЏ {offset}"),
            raw_text=f"Р’СЂРµРјРµРЅРЅР°СЏ РёРґРµСЏ {offset}",
            now=now,
            **binding(),
        )
        assert transient_stage is not None
        transient = await store.bind(
            transient_stage,
            canonical_message_id=400 + offset,
            now=now,
        )
        assert transient is not None
        transient_capability = await store.peek(
            transient.callback_data("add"),
            canonical_message_id=400 + offset,
            expected_action="add",
            now=now,
            **binding(),
        )
        assert transient_capability is not None
        assert await store.consume(transient_capability, now=now)

    live_key = (1, 101, 201, 301)
    assert len(store._canonical_generations) == store.max_capabilities
    assert live_key in store._canonical_generations
    assert await store.consume(live_capability, now=now)
    assert await store.consumed_screen_is_current(live_capability, now=now)


@pytest.mark.asyncio
async def test_failed_delivery_can_revoke_only_exact_unbound_stage() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    stage = await store.stage(suggestion(), raw_text="Конкретная задача", now=now, **binding())
    assert stage is not None

    assert await store.revoke_screen(stage, now=now) is True
    assert await store.bind(stage, canonical_message_id=301, now=now) is None
    assert await store.revoke_screen(stage, now=now) is False


@pytest.mark.asyncio
async def test_stage_entropy_failure_is_atomic_and_preserves_published_screen(
    monkeypatch,
) -> None:
    store = NovaCompanionCaptureStore(max_capabilities=2)
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    existing = await store.stage(suggestion(), raw_text="Конкретная задача", now=now, **binding())
    assert existing is not None
    existing = await store.bind(existing, canonical_message_id=301, now=now)
    assert existing is not None

    generated = iter(("new-screen", "same-token", "same-token"))

    def fail_after_batch_collision(_size: int) -> str:
        try:
            return next(generated)
        except StopIteration:
            raise OSError("PRIVATE_ENTROPY_FAILURE") from None

    monkeypatch.setattr(
        "future_self.nova_companion_flow.secrets.token_urlsafe",
        fail_after_batch_collision,
    )
    with pytest.raises(OSError, match="PRIVATE_ENTROPY_FAILURE"):
        await store.stage(
            CaptureSuggestion("task", "Нужно отправить договор"),
            raw_text="Нужно отправить договор",
            now=now,
            **binding(),
        )

    assert (
        await store.peek(
            existing.callback_data("add"),
            canonical_message_id=301,
            now=now,
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_not_now_atomically_suppresses_same_owner_topic_until_ttl() -> None:
    store = NovaCompanionCaptureStore(ttl=timedelta(seconds=10))
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    capture = suggestion()
    staged = await store.stage(capture, raw_text="Конкретная задача", now=now, **binding())
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None

    dismissed = await store.claim(
        bound.callback_data("not_now"),
        canonical_message_id=301,
        expected_action="not_now",
        now=now,
        **binding(),
    )
    assert dismissed is not None
    assert (
        await store.is_suppressed(
            fingerprint=capture.fingerprint,
            now=now,
            **binding(),
        )
        is True
    )
    assert await store.stage(capture, raw_text="Повтор той же темы", now=now, **binding()) is None
    other = {**binding(), "owner_id": 2, "telegram_user_id": 102}
    assert await store.stage(capture, raw_text="Та же тема другого owner", now=now, **other)
    assert (
        await store.stage(
            capture,
            raw_text="Тема после TTL",
            now=now + timedelta(seconds=10),
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_concurrent_same_topic_stage_has_one_renderable_generation() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    capture = suggestion()
    candidates = await asyncio.gather(
        *(
            store.stage(
                capture,
                raw_text=f"Конкретная задача {index}",
                now=now,
                **binding(),
            )
            for index in range(8)
        )
    )
    winners = [candidate for candidate in candidates if candidate is not None]
    assert len(winners) == 1

    winner = await store.bind(winners[0], canonical_message_id=301, now=now)
    assert winner is not None
    assert (
        await store.stage(
            capture,
            raw_text="Повтор после публикации",
            now=now,
            **binding(),
        )
        is None
    )

    unrelated = await store.stage(
        CaptureSuggestion("idea", "Идея сделать тихую комнату"),
        raw_text="Идея сделать тихую комнату",
        now=now,
        **binding(),
    )
    assert unrelated is not None
    unrelated = await store.bind(unrelated, canonical_message_id=303, now=now)
    assert unrelated is not None

    dismissed = await store.claim(
        winner.callback_data("not_now"),
        canonical_message_id=301,
        expected_action="not_now",
        now=now,
        **binding(),
    )

    assert dismissed is not None
    assert (
        await store.peek(
            winner.callback_data("add"),
            canonical_message_id=301,
            now=now,
            **binding(),
        )
        is None
    )
    assert (
        await store.peek(
            unrelated.callback_data("add"),
            canonical_message_id=303,
            now=now,
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_manual_suppression_uses_only_privacy_safe_fingerprint() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    capture = suggestion()

    await store.mark_suppressed(fingerprint=capture.fingerprint, now=now, **binding())

    assert await store.is_suppressed(fingerprint=capture.fingerprint, now=now, **binding())
    assert capture.title not in repr(store._suppressions)


@pytest.mark.asyncio
async def test_expiry_is_strict_and_does_not_consume_fresh_neighbor() -> None:
    store = NovaCompanionCaptureStore(ttl=timedelta(seconds=1))
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    old_stage = await store.stage(suggestion(), raw_text="Старая задача", now=now, **binding())
    assert old_stage is not None
    old = await store.bind(old_stage, canonical_message_id=301, now=now)
    assert old is not None
    expected = await store.peek(
        old.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now + timedelta(microseconds=999_999),
        **binding(),
    )
    assert expected is not None
    fresh_stage = await store.stage(
        CaptureSuggestion("idea", "Сделать тихую комнату"),
        raw_text="Свежая идея",
        now=now + timedelta(milliseconds=500),
        **binding(),
    )
    assert fresh_stage is not None
    fresh = await store.bind(
        fresh_stage,
        canonical_message_id=302,
        now=now + timedelta(milliseconds=500),
    )
    assert fresh is not None

    assert await store.consume(expected, now=expected.expires_at) is False
    assert (
        await store.claim(
            fresh.callback_data("add"),
            canonical_message_id=302,
            expected_action="add",
            now=now + timedelta(seconds=1),
            **binding(),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_concurrent_claim_has_one_whole_screen_winner() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(suggestion(), raw_text="Конкретная задача", now=now, **binding())
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None

    winners = await asyncio.gather(
        *(
            store.claim(
                bound.callback_data(action),
                canonical_message_id=301,
                expected_action=action,
                now=now,
                **binding(),
            )
            for action in ("add", "not_now")
        )
    )

    assert sum(result is not None for result in winners) == 1
    assert store._capabilities == {}
    assert store._screens == {}


@pytest.mark.asyncio
async def test_cancellation_waiting_for_store_lock_does_not_publish_or_spend() -> None:
    store = NovaCompanionCaptureStore()
    now = datetime(2026, 8, 18, 10, tzinfo=UTC)
    staged = await store.stage(suggestion(), raw_text="Конкретная задача", now=now, **binding())
    assert staged is not None
    bound = await store.bind(staged, canonical_message_id=301, now=now)
    assert bound is not None
    expected = await store.peek(
        bound.callback_data("add"),
        canonical_message_id=301,
        expected_action="add",
        now=now,
        **binding(),
    )
    assert expected is not None

    await store._lock.acquire()
    task = asyncio.create_task(store.consume(expected, now=now))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    store._lock.release()

    assert await store.consume(expected, now=now) is True


def test_capture_screen_callbacks_are_read_only_and_repr_is_privacy_safe() -> None:
    async def scenario():
        store = NovaCompanionCaptureStore()
        staged = await store.stage(
            suggestion(),
            raw_text="PRIVATE_SOURCE_TEXT",
            now=datetime(2026, 8, 18, 10, tzinfo=UTC),
            **binding(),
        )
        assert staged is not None
        with pytest.raises(TypeError):
            staged.callbacks["add"] = "forged"  # type: ignore[index]
        rendered = repr(staged)
        assert "PRIVATE_SOURCE_TEXT" not in rendered
        assert suggestion().title not in rendered
        assert "owner_id" not in rendered
        assert "chat_id" not in rendered

    asyncio.run(scenario())
