from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

import future_self.weekly_review_flow as weekly_review_flow_module
from future_self.weekly_review_flow import (
    WeeklyReviewCapabilityStore,
    WeeklyReviewDisposition,
    WeeklyReviewIntent,
    WeeklyReviewPolicy,
    classify_weekly_review_intent,
    reduce_weekly_review_input,
)


class _Actor:
    def __init__(self, tier: str, access_version: int = 1) -> None:
        self.access_tier = tier
        self.access_version = access_version


@pytest.mark.parametrize(
    ("enabled", "admin_only", "tier", "allowed"),
    [
        (False, False, "admin", False),
        (False, True, "admin", False),
        (True, True, "subscriber", False),
        (True, True, "admin", True),
        (True, False, "subscriber", True),
        (True, False, "admin", True),
        (True, False, "guest", False),
        (True, False, "blocked", False),
    ],
)
def test_weekly_policy_flag_and_tier_matrix(
    enabled: bool,
    admin_only: bool,
    tier: str,
    allowed: bool,
) -> None:
    policy = WeeklyReviewPolicy(enabled=enabled, admin_only=admin_only)

    assert policy.allows_tier(tier) is allowed
    assert policy.allows_actor(_Actor(tier)) is allowed


def test_weekly_policy_requires_actor_and_exact_access_generation() -> None:
    policy = WeeklyReviewPolicy(enabled=True, admin_only=True)
    actor = _Actor("admin", access_version=4)

    assert policy.allows_actor(None) is False
    assert policy.allows_actor(actor, expected_access_version=3) is False
    assert policy.allows_actor(actor, expected_access_version=4) is True


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Обзор недели", WeeklyReviewIntent.OPEN),
        ("Давай скорректируем систему", WeeklyReviewIntent.START),
        ("Спланируем неделю", WeeklyReviewIntent.START),
        ("Начать обзор недели", WeeklyReviewIntent.START),
        ("Фокус на неделю", WeeklyReviewIntent.EDIT_FOCUS),
        ("Изменить фокус недели", WeeklyReviewIntent.EDIT_FOCUS),
        ("Покажи фокус недели", WeeklyReviewIntent.VIEW),
        ("Назад", WeeklyReviewIntent.BACK),
        ("Пропустить", WeeklyReviewIntent.SKIP),
        ("Отменить", WeeklyReviewIntent.CANCEL),
        ("Обычный содержательный ответ", WeeklyReviewIntent.NONE),
    ],
)
def test_weekly_intent_classifier_is_exact(text: str, intent: WeeklyReviewIntent) -> None:
    assert classify_weekly_review_intent(text) is intent


@pytest.mark.parametrize("phase", ["preview", "delete_preview"])
@pytest.mark.parametrize(
    "text",
    ["Обычный длинный ответ", "Назад", "Отменить", "Спланируем неделю"],
)
def test_weekly_reducer_never_extracts_from_preview_phases(phase: str, text: str) -> None:
    assert (
        reduce_weekly_review_input(phase, text).disposition
        is WeeklyReviewDisposition.RENDER_CURRENT
    )


def test_weekly_reducer_absorbs_duplicate_processing_update() -> None:
    assert (
        reduce_weekly_review_input("processing", "Ещё один длинный ответ").disposition
        is WeeklyReviewDisposition.ABSORB
    )


def test_weekly_reducer_root_ordinary_text_preserves_owner_without_extraction() -> None:
    decision = reduce_weekly_review_input("root", "Обычная мысль для другой функции")

    assert decision.intent is WeeklyReviewIntent.NONE
    assert decision.disposition is WeeklyReviewDisposition.RENDER_CURRENT


@pytest.mark.parametrize(
    "text",
    ["Давай скорректируем систему", "Изменить фокус недели"],
)
def test_weekly_reducer_reprompts_repeated_start_in_awaiting_input(text: str) -> None:
    assert (
        reduce_weekly_review_input("awaiting_input", text).disposition
        is WeeklyReviewDisposition.REPROMPT
    )


@pytest.mark.parametrize(
    "text",
    ["Не знаю", "Не знаю пока", "Пока не знаю", "Затрудняюсь ответить"],
)
def test_weekly_reducer_reprompts_exact_non_answers(text: str) -> None:
    decision = reduce_weekly_review_input("awaiting_input", text)

    assert decision.intent is WeeklyReviewIntent.NONE
    assert decision.disposition is WeeklyReviewDisposition.REPROMPT


@pytest.mark.asyncio
async def test_weekly_capability_is_opaque_exact_bound_and_single_use() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    tokens = await store.issue(
        actions=("start", "close"),
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        access_version=4,
        week_start=date(2026, 8, 17),
        now=now,
    )

    assert set(tokens) == {"start", "close"}
    assert all(token and action not in token for action, token in tokens.items())
    # Wrong identity/canonical does not spend the owner's capability.
    assert (
        await store.claim(
            tokens["start"],
            telegram_user_id=999,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )
    claim = await store.claim(
        tokens["start"],
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=now,
    )
    assert claim is not None
    assert claim.action == "start"
    assert claim.week_start == date(2026, 8, 17)
    assert claim.access_version == 4
    assert (
        await store.claim(
            tokens["start"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )
    # The first valid launch action also retires sibling controls from the
    # same canonical screen, before a durable session id exists.
    assert (
        await store.claim(
            tokens["close"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )


@pytest.mark.asyncio
async def test_weekly_session_generation_retires_older_actions() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    old = await store.issue(
        actions=("save",),
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        access_version=4,
        week_start=date(2026, 8, 17),
        session_public_id="00000000-0000-4000-8000-000000000001",
        session_version=2,
        now=now,
    )
    new = await store.issue(
        actions=("save", "cancel"),
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        access_version=4,
        week_start=date(2026, 8, 17),
        session_public_id="00000000-0000-4000-8000-000000000001",
        session_version=3,
        now=now,
    )

    assert (
        await store.claim(
            old["save"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )
    claim = await store.claim(
        new["cancel"],
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=now,
    )
    assert claim is not None and claim.session_version == 3
    # A valid claim retires sibling controls from the same screen.
    assert (
        await store.claim(
            new["save"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )


@pytest.mark.asyncio
async def test_staged_session_controls_do_not_revoke_visible_screen_before_render() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    binding = {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "canonical_message_id": 301,
        "access_version": 4,
        "week_start": date(2026, 8, 17),
        "session_public_id": "00000000-0000-4000-8000-000000000001",
        "now": now,
    }
    visible = await store.issue(actions=("save",), session_version=2, **binding)
    staged = await store.issue(
        actions=("cancel",),
        session_version=3,
        replace_session=False,
        **binding,
    )

    assert (
        await store.peek(
            visible["save"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is not None
    )
    assert (
        await store.peek(
            staged["cancel"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is not None
    )


@pytest.mark.asyncio
async def test_failed_staged_screen_revokes_only_itself() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    binding = {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "canonical_message_id": 301,
        "access_version": 4,
        "week_start": date(2026, 8, 17),
        "session_public_id": "00000000-0000-4000-8000-000000000001",
        "now": now,
    }
    visible = await store.issue(actions=("save",), session_version=2, **binding)
    staged = await store.issue(
        actions=("cancel",),
        session_version=3,
        replace_session=False,
        **binding,
    )
    staged_screen = await store.screen_for_tokens(tuple(staged.values()), now=now)
    assert staged_screen is not None

    assert await store.revoke_screen(staged_screen, now=now) is True
    assert (
        await store.peek(
            visible["save"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is not None
    )
    assert (
        await store.peek(
            staged["cancel"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )


@pytest.mark.asyncio
async def test_successful_rerender_activates_one_screen_without_growth() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    binding = {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "canonical_message_id": 301,
        "access_version": 4,
        "week_start": date(2026, 8, 17),
        "session_public_id": "00000000-0000-4000-8000-000000000001",
        "session_version": 2,
        "now": now,
    }
    visible = await store.issue(actions=("save", "cancel"), **binding)
    previous = visible
    for _index in range(3):
        staged = await store.issue(
            actions=("save", "cancel"),
            replace_session=False,
            **binding,
        )
        screen = await store.screen_for_tokens(tuple(staged.values()), now=now)
        assert screen is not None
        assert await store.activate_screen(screen, now=now) is True
        assert (
            await store.peek(
                previous["save"],
                telegram_user_id=101,
                chat_id=201,
                canonical_message_id=301,
                now=now,
            )
            is None
        )
        previous = staged

    assert len(store._capabilities) == 2
    assert len(store._screens) == 1


@pytest.mark.asyncio
async def test_older_activation_preserves_newer_concurrent_stage() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    binding = {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "canonical_message_id": 301,
        "access_version": 4,
        "week_start": date(2026, 8, 17),
        "session_public_id": "00000000-0000-4000-8000-000000000001",
        "session_version": 2,
        "now": now,
    }
    visible = await store.issue(actions=("save",), **binding)
    older = await store.issue(actions=("cancel",), replace_session=False, **binding)
    newer = await store.issue(actions=("edit",), replace_session=False, **binding)
    older_screen = await store.screen_for_tokens(tuple(older.values()), now=now)
    newer_screen = await store.screen_for_tokens(tuple(newer.values()), now=now)
    assert older_screen is not None and newer_screen is not None

    assert await store.activate_screen(older_screen, now=now) is True
    assert (
        await store.peek(
            visible["save"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )
    assert await store.screen_is_live(newer_screen, now=now) is True
    assert await store.activate_screen(newer_screen, now=now) is True
    assert await store.screen_is_live(older_screen, now=now) is False
    assert len(store._capabilities) == 1


@pytest.mark.asyncio
async def test_stale_replacement_token_cannot_consume_fresh_screen_capability() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    binding = {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "canonical_message_id": 301,
        "access_version": 4,
        "week_start": date(2026, 8, 17),
        "now": now,
    }
    old = await store.issue(
        actions=("save",),
        session_public_id="00000000-0000-4000-8000-000000000001",
        session_version=2,
        **binding,
    )
    fresh = await store.issue(
        actions=("save",),
        session_public_id="00000000-0000-4000-8000-000000000002",
        session_version=1,
        **binding,
    )

    old_claim = await store.claim(
        old["save"],
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=now,
    )
    assert old_claim is not None
    fresh_claim = await store.claim(
        fresh["save"],
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=now,
    )
    assert fresh_claim is not None
    assert fresh_claim.session_public_id == "00000000-0000-4000-8000-000000000002"


@pytest.mark.asyncio
async def test_weekly_capability_expiry_is_bounded() -> None:
    store = WeeklyReviewCapabilityStore(ttl=timedelta(seconds=1))
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    token = (
        await store.issue(
            actions=("start",),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            access_version=4,
            week_start=date(2026, 8, 17),
            now=now,
        )
    )["start"]

    assert (
        await store.claim(
            token,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now + timedelta(seconds=2),
        )
        is None
    )
    assert store._canonical_generations == {}


@pytest.mark.asyncio
async def test_consume_rechecks_expiry_after_peek_and_cleans_only_expired_screen() -> None:
    store = WeeklyReviewCapabilityStore(ttl=timedelta(seconds=1))
    issued_at = datetime(2026, 8, 17, 10, tzinfo=UTC)
    expired = await store.issue(
        actions=("save", "cancel"),
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        access_version=4,
        week_start=date(2026, 8, 17),
        session_public_id="00000000-0000-4000-8000-000000000001",
        session_version=1,
        now=issued_at,
    )
    fresh = await store.issue(
        actions=("edit",),
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        access_version=4,
        week_start=date(2026, 8, 17),
        session_public_id="00000000-0000-4000-8000-000000000002",
        session_version=1,
        now=issued_at + timedelta(milliseconds=500),
    )
    expected = await store.peek(
        expired["save"],
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=issued_at + timedelta(microseconds=999_999),
    )
    assert expected is not None
    fresh_screen = await store.screen_for_tokens(
        tuple(fresh.values()),
        now=issued_at + timedelta(microseconds=999_999),
    )
    assert fresh_screen is not None

    assert await store.consume(expected, now=expected.expires_at) is False
    assert set(store._capabilities) == set(fresh.values())
    assert set(store._screens) == {fresh_screen.screen_id}


@pytest.mark.asyncio
async def test_concurrent_consume_just_before_expiry_has_one_whole_screen_winner() -> None:
    store = WeeklyReviewCapabilityStore(ttl=timedelta(seconds=1))
    issued_at = datetime(2026, 8, 17, 10, tzinfo=UTC)
    tokens = await store.issue(
        actions=("save", "cancel"),
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        access_version=4,
        week_start=date(2026, 8, 17),
        session_public_id="00000000-0000-4000-8000-000000000001",
        session_version=1,
        now=issued_at,
    )
    before_expiry = issued_at + timedelta(microseconds=999_999)
    claims = await asyncio.gather(
        *(
            store.peek(
                token,
                telegram_user_id=101,
                chat_id=201,
                canonical_message_id=301,
                now=before_expiry,
            )
            for token in tokens.values()
        )
    )
    assert all(claim is not None for claim in claims)

    winners = await asyncio.gather(
        *(store.consume(claim, now=before_expiry) for claim in claims if claim is not None)
    )

    assert winners.count(True) == 1
    assert store._capabilities == {}
    assert store._screens == {}


@pytest.mark.asyncio
async def test_consume_at_exact_expiry_fails_closed() -> None:
    store = WeeklyReviewCapabilityStore(ttl=timedelta(seconds=1))
    issued_at = datetime(2026, 8, 17, 10, tzinfo=UTC)
    token = (
        await store.issue(
            actions=("start",),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            access_version=4,
            week_start=date(2026, 8, 17),
            now=issued_at,
        )
    )["start"]
    expected = await store.peek(
        token,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=issued_at + timedelta(microseconds=999_999),
    )
    assert expected is not None

    assert await store.consume(expected, now=expected.expires_at) is False
    assert store._capabilities == {}


@pytest.mark.parametrize(
    ("elapsed", "claimed"),
    [
        pytest.param(timedelta(microseconds=999_999), True, id="before"),
        pytest.param(timedelta(seconds=1), False, id="at"),
        pytest.param(timedelta(seconds=1, microseconds=1), False, id="after"),
    ],
)
@pytest.mark.asyncio
async def test_claim_uses_injected_clock_for_peek_and_consume(
    elapsed: timedelta,
    claimed: bool,
) -> None:
    store = WeeklyReviewCapabilityStore(ttl=timedelta(seconds=1))
    issued_at = datetime(2000, 1, 1, tzinfo=UTC)
    token = (
        await store.issue(
            actions=("start",),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            access_version=4,
            week_start=date(2000, 1, 3),
            now=issued_at,
        )
    )["start"]

    result = await store.claim(
        token,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=issued_at + elapsed,
    )

    assert (result is not None) is claimed
    assert store._capabilities == {}


@pytest.mark.asyncio
async def test_claim_runtime_clock_rechecks_expiry_between_peek_and_consume(
    monkeypatch,
) -> None:
    store = WeeklyReviewCapabilityStore(ttl=timedelta(seconds=1))
    issued_at = datetime(2026, 8, 17, 10, tzinfo=UTC)
    token = (
        await store.issue(
            actions=("start",),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            access_version=4,
            week_start=date(2026, 8, 17),
            now=issued_at,
        )
    )["start"]
    moments = iter(
        (
            issued_at + timedelta(microseconds=999_999),
            issued_at + timedelta(seconds=1),
        )
    )
    sampled: list[datetime] = []
    normalize = store._utc

    def advancing_clock(value: datetime | None) -> datetime:
        if value is not None:
            return normalize(value)
        current = next(moments)
        sampled.append(current)
        return current

    monkeypatch.setattr(store, "_utc", advancing_clock)

    assert (
        await store.claim(
            token,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
        )
        is None
    )
    assert sampled == [
        issued_at + timedelta(microseconds=999_999),
        issued_at + timedelta(seconds=1),
    ]
    assert store._capabilities == {}


@pytest.mark.asyncio
async def test_concurrent_consume_around_expiry_never_lets_expired_call_win() -> None:
    store = WeeklyReviewCapabilityStore(ttl=timedelta(seconds=1))
    issued_at = datetime(2026, 8, 17, 10, tzinfo=UTC)
    tokens = await store.issue(
        actions=("save", "cancel"),
        owner_id=1,
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        access_version=4,
        week_start=date(2026, 8, 17),
        session_public_id="00000000-0000-4000-8000-000000000001",
        session_version=1,
        now=issued_at,
    )
    before_expiry = issued_at + timedelta(microseconds=999_999)
    claims = await asyncio.gather(
        *(
            store.peek(
                token,
                telegram_user_id=101,
                chat_id=201,
                canonical_message_id=301,
                now=before_expiry,
            )
            for token in tokens.values()
        )
    )
    assert all(claim is not None for claim in claims)
    first, second = claims
    assert first is not None and second is not None

    before_result, expired_result = await asyncio.gather(
        store.consume(first, now=before_expiry),
        store.consume(second, now=issued_at + timedelta(seconds=1)),
    )

    assert expired_result is False
    assert int(before_result) + int(expired_result) <= 1
    assert store._capabilities == {}
    assert store._screens == {}


@pytest.mark.asyncio
async def test_consumed_newer_screen_tombstone_blocks_old_exact_cleanup() -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    binding = {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "canonical_message_id": 301,
        "access_version": 4,
        "week_start": date(2026, 8, 24),
        "scheduled": True,
        "now": now,
    }
    old = await store.issue(actions=("old",), **binding)
    newer = await store.issue(actions=("newer",), **binding)

    newer_claim = await store.claim(
        newer["newer"],
        telegram_user_id=101,
        chat_id=201,
        canonical_message_id=301,
        now=now,
    )
    assert newer_claim is not None
    assert (
        await store.peek(
            newer["newer"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is None
    )

    assert (
        await store.revoke_tokens_if_current_screen(
            tuple(old.values()),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            access_version=4,
            week_start=date(2026, 8, 24),
            scheduled=True,
            now=now,
        )
        is False
    )
    assert store._capabilities == {}
    assert store._canonical_generations[(101, 201, 301)][0] == newer_claim.screen_order


@pytest.mark.asyncio
async def test_canonical_generation_tombstones_are_bounded() -> None:
    store = WeeklyReviewCapabilityStore(max_capabilities=2)
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    for offset in range(3):
        await store.issue(
            actions=("start",),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301 + offset,
            access_version=4,
            week_start=date(2026, 8, 24),
            scheduled=True,
            now=now + timedelta(microseconds=offset),
        )

    assert len(store._canonical_generations) == 2
    assert (101, 201, 301) not in store._canonical_generations


@pytest.mark.asyncio
async def test_tombstone_eviction_never_discards_a_live_canonical_generation() -> None:
    store = WeeklyReviewCapabilityStore(max_capabilities=2)
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)

    async def issue(message_id: int, action: str) -> dict[str, str]:
        return await store.issue(
            actions=(action,),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=message_id,
            access_version=4,
            week_start=date(2026, 8, 24),
            scheduled=True,
            now=now,
        )

    live_a = await issue(301, "a")
    consumed_b = await issue(302, "b")
    assert (
        await store.claim(
            consumed_b["b"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=302,
            now=now,
        )
        is not None
    )
    consumed_c = await issue(303, "c")
    assert (
        await store.claim(
            consumed_c["c"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=303,
            now=now,
        )
        is not None
    )

    assert len(store._canonical_generations) == 2
    assert (101, 201, 301) in store._canonical_generations
    assert (
        await store.revoke_tokens_if_current_screen(
            tuple(live_a.values()),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            access_version=4,
            week_start=date(2026, 8, 24),
            scheduled=True,
            now=now,
        )
        is True
    )
    assert store._capabilities == {}
    assert len(store._canonical_generations) <= store.max_capabilities


@pytest.mark.asyncio
async def test_issue_mid_generation_failure_publishes_nothing_and_preserves_existing_screen(
    monkeypatch,
) -> None:
    store = WeeklyReviewCapabilityStore()
    now = datetime(2026, 8, 17, 10, tzinfo=UTC)
    session_public_id = "00000000-0000-4000-8000-000000000099"
    binding = {
        "owner_id": 1,
        "telegram_user_id": 101,
        "chat_id": 201,
        "canonical_message_id": 301,
        "access_version": 4,
        "week_start": date(2026, 8, 24),
        "session_public_id": session_public_id,
        "session_version": 3,
        "now": now,
    }
    existing = await store.issue(actions=("existing",), **binding)
    capabilities_before = dict(store._capabilities)
    screens_before = dict(store._screens)
    generations_before = dict(store._canonical_generations)
    screen_order_before = store._next_screen_order
    generated = iter(("staged-screen", "staged-first"))

    def fail_mid_batch(_length: int) -> str:
        try:
            return next(generated)
        except StopIteration:
            raise RuntimeError("PRIVATE_MID_ISSUE_FAILURE") from None

    monkeypatch.setattr(weekly_review_flow_module.secrets, "token_urlsafe", fail_mid_batch)

    with pytest.raises(RuntimeError, match="PRIVATE_MID_ISSUE_FAILURE"):
        await store.issue(actions=("first", "second"), **binding)

    assert store._capabilities == capabilities_before
    assert store._screens == screens_before
    assert store._canonical_generations == generations_before
    assert store._next_screen_order == screen_order_before
    assert (
        await store.peek(
            existing["existing"],
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            now=now,
        )
        is not None
    )


@pytest.mark.asyncio
async def test_issue_rejects_action_batch_larger_than_capability_limit() -> None:
    store = WeeklyReviewCapabilityStore(max_capabilities=1)

    with pytest.raises(ValueError, match="batch exceeds capability limit"):
        await store.issue(
            actions=("first", "second"),
            owner_id=1,
            telegram_user_id=101,
            chat_id=201,
            canonical_message_id=301,
            access_version=4,
            week_start=date(2026, 8, 24),
        )

    assert store._capabilities == {}
    assert store._screens == {}
    assert store._canonical_generations == {}
