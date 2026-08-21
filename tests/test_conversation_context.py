import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from future_self.access import AccessService
from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.conversation import (
    COMPANION_PROMPT_CONTEXT_MAX_BYTES,
    COMPANION_PROMPT_MAX_MESSAGES,
    COMPANION_PROMPT_MESSAGE_MAX_CHARS,
    CompanionConversationFence,
    ConversationContextService,
    ConversationSnapshot,
    companion_conversation_revision,
)
from future_self.dates import DateResolver
from future_self.db import Database
from future_self.models import (
    ConversationMessage,
    ConversationSession,
    DraftInboxItem,
    InboxItem,
    User,
)


class CommitGateDatabase(Database):
    """Hold a successful transaction immediately before its real database commit."""

    def __init__(self, url: str):
        super().__init__(url)
        self.before_commit = asyncio.Event()
        self.allow_commit = asyncio.Event()

    @asynccontextmanager
    async def session(self):
        async with super().session() as session:
            yield session
            self.before_commit.set()
            await self.allow_commit.wait()


class SessionObservedDatabase(Database):
    """Expose that a second database instance has entered its transaction."""

    def __init__(self, url: str):
        super().__init__(url)
        self.session_started = asyncio.Event()

    @asynccontextmanager
    async def session(self):
        async with super().session() as session:
            self.session_started.set()
            yield session


class FakeMessage:
    def __init__(self, text: str):
        self.text = text
        self.voice = None
        self.audio = None
        self.replies: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs):
        self.replies.append({"text": text, **kwargs})
        return self


class FakeTranscription:
    enabled = True

    async def transcribe(self, audio: bytes, filename: str) -> str:
        return ""


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
        transcription_provider="disabled",
        conversation_context_messages=12,
        conversation_context_ttl_hours=24,
    )


def update_for(message: FakeMessage, user_id: int, chat_id: int):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
    )


async def counts(db) -> tuple[int, int]:
    async with db.sessions() as session:
        drafts = await session.scalar(select(func.count(DraftInboxItem.id)))
        inbox = await session.scalar(select(func.count(InboxItem.id)))
    return int(drafts), int(inbox)


async def exchange_actor(db, telegram_user_id: int = 701) -> User:
    async with db.session() as session:
        actor = User(
            telegram_id=telegram_user_id,
            timezone="Europe/Moscow",
            access_tier="admin",
            access_version=1,
            onboarding_completed=True,
        )
        session.add(actor)
        await session.flush()
        owner_id = actor.id
    async with db.sessions() as session:
        stored = await session.get(User, owner_id)
        assert stored is not None
        return stored


async def exchange_fence(
    service: ConversationContextService,
    actor: User,
    chat_id: int,
) -> CompanionConversationFence:
    context = (await service.get(actor.telegram_id, chat_id)).for_companion_prompt()
    return CompanionConversationFence(
        owner_id=actor.id,
        telegram_user_id=actor.telegram_id,
        chat_id=chat_id,
        access_version=actor.access_version,
        access_tier=actor.access_tier,
        raw_message_limit=service.message_limit,
        conversation_payload_max_bytes=COMPANION_PROMPT_CONTEXT_MAX_BYTES,
        revision=companion_conversation_revision(
            owner_id=actor.id,
            telegram_user_id=actor.telegram_id,
            chat_id=chat_id,
            context=context,
        ),
    )


def test_snapshot_for_prompt_excludes_memory_answers_without_mutating_local_history():
    user_message = {
        "role": "user",
        "content": "What should I do next?",
        "source": "text",
        "intent": "question",
    }
    memory_answer = {
        "role": "assistant",
        "content": "MEMORY_DERIVED_ANSWER_SENTINEL",
        "source": "text",
        "intent": "memory_answer",
    }
    ordinary_answer = {
        "role": "assistant",
        "content": "An ordinary answer remains available.",
        "source": "text",
        "intent": "answer",
    }
    snapshot = ConversationSnapshot(
        messages=[user_message, memory_answer, ordinary_answer],
    )

    prompt = snapshot.for_prompt()

    assert prompt["recent_messages"] == [user_message, ordinary_answer]
    assert snapshot.messages == [user_message, memory_answer, ordinary_answer]
    assert snapshot.messages[1]["content"] == "MEMORY_DERIVED_ANSWER_SENTINEL"
    assert "MEMORY_DERIVED_ANSWER_SENTINEL" not in repr(prompt)
    assert "MEMORY_DERIVED_ANSWER_SENTINEL" not in repr(snapshot)
    assert ConversationContextService.latest_nova_memory_candidate(snapshot) == (
        "What should I do next?"
    )


def test_snapshot_for_prompt_returns_fresh_message_list_and_dicts_on_every_call():
    original_message = {
        "role": "assistant",
        "content": "Keep this ordinary answer.",
        "source": "text",
        "intent": "answer",
    }
    snapshot = ConversationSnapshot(messages=[original_message])

    first_messages = snapshot.for_prompt()["recent_messages"]
    second_messages = snapshot.for_prompt()["recent_messages"]

    assert isinstance(first_messages, list)
    assert isinstance(second_messages, list)
    assert first_messages is not snapshot.messages
    assert second_messages is not first_messages
    assert first_messages[0] is not original_message
    assert second_messages[0] is not first_messages[0]

    first_messages[0]["content"] = "mutated provider copy"
    first_messages.append({"role": "assistant", "content": "provider-only"})

    assert snapshot.messages == [original_message]
    assert second_messages == [original_message]


def test_snapshot_for_prompt_filter_is_independent_of_application_feature_state():
    messages = [
        {
            "role": "user",
            "content": "Current user request",
            "source": "text",
            "intent": "question",
        },
        {
            "role": "assistant",
            "content": "STALE_MEMORY_ANSWER_SENTINEL",
            "source": "text",
            "intent": "memory_answer",
        },
    ]

    contexts_by_application_state = {
        application_enabled: ConversationSnapshot(messages=messages).for_prompt()
        for application_enabled in (False, True)
    }

    expected = [messages[0]]
    assert contexts_by_application_state[False]["recent_messages"] == expected
    assert contexts_by_application_state[True]["recent_messages"] == expected


def test_snapshot_for_companion_prompt_is_bounded_content_only_and_fail_closed():
    safe_messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"ordinary-{index} " + "x" * 900,
            "intent": "conversation" if index % 2 == 0 else "answer",
            "timestamp": f"PRIVATE_TIMESTAMP_{index}",
            "source": "PRIVATE_SOURCE",
            "telegram_id": 999,
        }
        for index in range(12)
    ]
    control_intents = (
        "preview",
        "relative_reminder",
        "date_interpretation",
        "navigation",
        "system_action",
        "onboarding",
        "health_checkin",
        "labs_upload",
        "doctor_visit",
        "knowledge_search",
        "nova_memory_add",
        "explicit_capture",
        "unknown_future_control",
    )
    control_messages = [
        {
            "role": "user",
            "content": f"PRIVATE_CONTROL_{intent}",
            "intent": intent,
        }
        for intent in control_intents
    ]
    snapshot = ConversationSnapshot(
        session_id=123,
        current_topic="  Current\n topic  " + "t" * 400,
        summary="  Stable\t summary  " + "s" * 900,
        messages=[
            {"role": "user", "content": "PRIVATE_MISSING_INTENT"},
            {"role": "assistant", "content": "PRIVATE_EMPTY_INTENT", "intent": "  "},
            {"role": "system", "content": "PRIVATE_SYSTEM_ROLE"},
            *control_messages,
            *safe_messages,
        ],
        active_draft={"id": "PRIVATE_DRAFT", "title": "PRIVATE_DRAFT_TITLE"},
        pending_date_options=[{"label": "PRIVATE_DATE"}],
        focused_draft_id="PRIVATE_FOCUS",
        pending_action="PRIVATE_ACTION",
        system_pending_action="PRIVATE_SYSTEM_ACTION",
    )

    prompt = snapshot.for_companion_prompt()

    assert set(prompt) == {"recent_messages"}
    messages = prompt["recent_messages"]
    assert len(messages) == min(len(safe_messages), COMPANION_PROMPT_MAX_MESSAGES)
    first_safe_index = max(0, len(safe_messages) - COMPANION_PROMPT_MAX_MESSAGES)
    assert messages[0]["content"].startswith(f"ordinary-{first_safe_index} ")
    assert messages[-1]["content"].startswith("ordinary-11 ")
    assert all(set(message) == {"role", "content"} for message in messages)
    assert all(
        len(message["content"]) <= COMPANION_PROMPT_MESSAGE_MAX_CHARS for message in messages
    )
    serialized = repr(prompt)
    for forbidden in (
        "PRIVATE_CONTROL_",
        "PRIVATE_SYSTEM_ROLE",
        "PRIVATE_MISSING_INTENT",
        "PRIVATE_EMPTY_INTENT",
        "PRIVATE_TIMESTAMP",
        "PRIVATE_SOURCE",
        "PRIVATE_DRAFT",
        "PRIVATE_DATE",
        "PRIVATE_FOCUS",
        "PRIVATE_ACTION",
        "PRIVATE_SYSTEM_ACTION",
        "telegram_id",
        "session_id",
        "intent",
    ):
        assert forbidden not in serialized


def test_snapshot_for_companion_prompt_is_deep_detached_and_deterministic():
    original = {
        "role": "USER",
        "content": "  A\n private but ordinary thought  ",
        "intent": "companion_user",
        "source": "voice",
    }
    snapshot = ConversationSnapshot(
        current_topic="  one   topic ",
        summary=" one\nsummary ",
        messages=[original],
    )

    first = snapshot.for_companion_prompt()
    second = snapshot.for_companion_prompt()

    assert (
        first
        == second
        == {"recent_messages": [{"role": "user", "content": "A private but ordinary thought"}]}
    )
    assert first is not second
    assert first["recent_messages"] is not second["recent_messages"]
    assert first["recent_messages"][0] is not second["recent_messages"][0]
    first["recent_messages"][0]["content"] = "MUTATED"
    first["recent_messages"].append({"role": "assistant", "content": "MUTATED"})

    assert snapshot.messages == [original]
    assert snapshot.for_companion_prompt() == second


def test_snapshot_for_companion_prompt_never_leaks_unattributed_topic_or_summary() -> None:
    snapshot = ConversationSnapshot(
        current_topic="PRIVATE_UNCONFIRMED_DRAFT_TITLE",
        summary="PRIVATE_EXCLUDED_FLOW_SUMMARY",
        messages=[
            {"role": "user", "content": "Обычная реплика", "intent": "conversation"},
        ],
    )

    prompt = snapshot.for_companion_prompt()

    assert prompt == {"recent_messages": [{"role": "user", "content": "Обычная реплика"}]}
    assert "PRIVATE_" not in repr(prompt)


def test_companion_reference_uses_only_latest_proven_ordinary_user_message() -> None:
    snapshot = ConversationSnapshot(
        messages=[
            {
                "role": "user",
                "content": "Старая обычная мысль не должна быть выбрана",
                "source": "text",
                "intent": "conversation",
            },
            {
                "role": "assistant",
                "content": "Продолжай.",
                "source": "text",
                "intent": "companion_answer",
            },
            {
                "role": "user",
                "content": "  Конкретная последняя мысль для сохранения  ",
                "source": "voice",
                "intent": "companion_user",
            },
        ]
    )

    assert ConversationContextService.companion_reference_candidate(snapshot) == (
        "Конкретная последняя мысль для сохранения"
    )


@pytest.mark.parametrize(
    "intent",
    [
        "health_answer",
        "knowledge_capture",
        "memory_answer",
        "preview",
        "relative_reminder",
        "navigation",
        "",
        None,
    ],
)
def test_companion_reference_never_skips_excluded_latest_private_flow(intent) -> None:
    snapshot = ConversationSnapshot(
        messages=[
            {
                "role": "user",
                "content": "Старая обычная мысль не должна просочиться",
                "source": "text",
                "intent": "conversation",
            },
            {
                "role": "user",
                "content": "PRIVATE_DURABLE_SENTINEL с чувствительными деталями",
                "source": "text",
                "intent": intent,
            },
        ]
    )

    assert ConversationContextService.companion_reference_candidate(snapshot) is None


async def test_persisted_memory_answer_remains_local_but_is_excluded_from_prompt(db):
    service = ConversationContextService(db, 12, 24)
    await service.append(
        901,
        902,
        role="user",
        content="Current question",
        source="text",
        intent="question",
    )
    await service.append(
        901,
        902,
        role="assistant",
        content="PERSISTED_MEMORY_ANSWER_SENTINEL",
        source="text",
        intent="memory_answer",
    )

    snapshot = await service.get(901, 902)

    assert [message["intent"] for message in snapshot.messages] == [
        "question",
        "memory_answer",
    ]
    assert snapshot.messages[-1]["content"] == "PERSISTED_MEMORY_ANSWER_SENTINEL"
    assert snapshot.for_prompt()["recent_messages"] == [snapshot.messages[0]]


async def test_context_is_persistent_bounded_and_available_after_recreation(db):
    first = ConversationContextService(db, 12, 24)
    for index in range(15):
        await first.append(
            100,
            200,
            role="user" if index % 2 == 0 else "assistant",
            content=f"Сообщение {index}",
            source="text",
            intent="conversation",
            topic="планирование",
        )
    recreated = ConversationContextService(db, 12, 24)
    snapshot = await recreated.get(100, 200)
    assert len(snapshot.messages) == 12
    assert snapshot.messages[-1]["content"] == "Сообщение 14"
    assert snapshot.current_topic == "планирование"
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 12


async def test_context_isolated_by_user_and_chat(db):
    service = ConversationContextService(db, 12, 24)
    await service.append(
        100,
        200,
        role="user",
        content="Личный контекст",
        source="text",
        intent="conversation",
    )
    assert (await service.get(100, 200)).messages
    assert not (await service.get(101, 200)).messages
    assert not (await service.get(100, 201)).messages


async def test_context_ttl_excludes_expired_messages(db):
    service = ConversationContextService(db, 12, 24)
    await service.append(
        100,
        200,
        role="user",
        content="Старое сообщение",
        source="text",
        intent="conversation",
    )
    async with db.session() as session:
        conversation = await session.scalar(select(ConversationSession))
        conversation.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert not (await service.get(100, 200)).messages


async def test_append_exchange_is_atomic_bounded_and_exactly_compensable(db):
    actor = await exchange_actor(db)
    chat_id = actor.telegram_id
    service = ConversationContextService(db, 10, 24)
    for index in range(10):
        await service.append(
            actor.telegram_id,
            chat_id,
            role="user",
            content=f"prior-{index}",
            source="text",
            intent="conversation",
        )
    prior = (await service.get(actor.telegram_id, chat_id)).messages
    fence = await exchange_fence(service, actor, chat_id)

    receipt = await service.append_exchange(
        fence,
        user_content="PRIVATE_EXCHANGE_USER",
        assistant_content="PRIVATE_EXCHANGE_ASSISTANT",
        user_source="voice",
    )

    assert receipt is not None
    assert repr(receipt) == "ConversationExchangeReceipt()"
    assert "PRIVATE_EXCHANGE" not in repr(receipt)
    committed = (await service.get(actor.telegram_id, chat_id)).messages
    assert len(committed) == 10
    assert [(message["role"], message["intent"]) for message in committed[-2:]] == [
        ("user", "companion_user"),
        ("assistant", "companion_answer"),
    ]
    assert await service.compensate_exchange(receipt)
    assert (await service.get(actor.telegram_id, chat_id)).messages == prior
    assert not await service.compensate_exchange(receipt)


async def test_append_exchange_second_insert_failure_rolls_back_user_row(db):
    actor = await exchange_actor(db, 702)

    class FailingExchangeService(ConversationContextService):
        async def _before_exchange_assistant_insert(self, fence):
            del fence
            raise RuntimeError("PRIVATE_SECOND_INSERT_FAILURE")

    service = FailingExchangeService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    with pytest.raises(RuntimeError, match="PRIVATE_SECOND_INSERT_FAILURE"):
        await service.append_exchange(
            fence,
            user_content="user",
            assistant_content="assistant",
            user_source="text",
        )

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(ConversationSession.id))) == 0


async def test_append_exchange_cancellation_propagates_and_rolls_back(db):
    actor = await exchange_actor(db, 703)

    class CancellingExchangeService(ConversationContextService):
        async def _before_exchange_assistant_insert(self, fence):
            del fence
            raise asyncio.CancelledError

    service = CancellingExchangeService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    with pytest.raises(asyncio.CancelledError):
        await service.append_exchange(
            fence,
            user_content="user",
            assistant_content="assistant",
            user_source="text",
        )

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        assert await session.scalar(select(func.count(ConversationSession.id))) == 0


@pytest.mark.parametrize("failure", ["ordinary", "cancel"])
async def test_append_exchange_post_commit_failure_shields_exact_compensation(
    db,
    failure,
):
    actor = await exchange_actor(db, 708)

    class PostCommitFailureService(ConversationContextService):
        async def _after_exchange_commit(self, receipt):
            assert receipt is not None
            await self.append(
                actor.telegram_id,
                actor.telegram_id,
                role="user",
                content="newer concurrent message",
                source="text",
                intent="conversation",
            )
            if failure == "cancel":
                raise asyncio.CancelledError
            raise RuntimeError("PRIVATE_POST_COMMIT_FAILURE")

    service = PostCommitFailureService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    expected_error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(expected_error):
        await service.append_exchange(
            fence,
            user_content="exchange user",
            assistant_content="exchange assistant",
            user_source="text",
        )

    async with db.sessions() as session:
        messages = list(
            (
                await session.scalars(select(ConversationMessage).order_by(ConversationMessage.id))
            ).all()
        )
    assert [(message.content, message.intent) for message in messages] == [
        ("newer concurrent message", "conversation")
    ]
    await asyncio.sleep(0)
    assert not any(
        task.get_name() == "companion-exchange-post-commit-compensation" and not task.done()
        for task in asyncio.all_tasks()
    )


async def test_append_exchange_cleanup_preserves_later_outer_cancellation(db):
    actor = await exchange_actor(db, 711)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    class BlockingCompensationService(ConversationContextService):
        async def _after_exchange_commit(self, receipt):
            del receipt
            raise RuntimeError("ordinary post-commit failure")

        async def compensate_exchange(self, receipt):
            cleanup_started.set()
            await cleanup_release.wait()
            return await super().compensate_exchange(receipt)

    service = BlockingCompensationService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    append = asyncio.create_task(
        service.append_exchange(
            fence,
            user_content="exchange user",
            assistant_content="exchange assistant",
            user_source="text",
        )
    )
    await cleanup_started.wait()
    append.cancel()
    await asyncio.sleep(0)
    assert not append.done()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await append

    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
    assert not any(
        task.get_name() == "companion-exchange-post-commit-compensation" and not task.done()
        for task in asyncio.all_tasks()
    )


async def test_append_exchange_access_bounce_fails_before_any_message(db):
    actor = await exchange_actor(db, 704)

    class BouncingExchangeService(ConversationContextService):
        async def _before_exchange_access_lock(self, fence):
            await AccessService(self.db).block(
                fence.telegram_user_id,
                source="exchange-test",
            )

    service = BouncingExchangeService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    assert (
        await service.append_exchange(
            fence,
            user_content="user",
            assistant_content="assistant",
            user_source="text",
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0
        stored = await session.get(User, actor.id)
        assert stored is not None
        assert stored.access_tier == "blocked"
        assert stored.access_version == 2


async def test_append_exchange_requires_exact_tier_even_if_version_is_malformed_unchanged(db):
    actor = await exchange_actor(db, 707)
    service = ConversationContextService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    async with db.session() as session:
        stored = await session.get(User, actor.id)
        assert stored is not None
        stored.access_tier = "subscriber"

    assert (
        await service.append_exchange(
            fence,
            user_content="user",
            assistant_content="assistant",
            user_source="text",
        )
        is None
    )
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0


async def test_append_exchange_rejects_stale_safe_context_without_partial_dml(db):
    actor = await exchange_actor(db, 705)
    service = ConversationContextService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    await service.append(
        actor.telegram_id,
        actor.telegram_id,
        role="user",
        content="concurrent safe mutation",
        source="text",
        intent="conversation",
    )

    assert (
        await service.append_exchange(
            fence,
            user_content="stale user",
            assistant_content="stale assistant",
            user_source="text",
        )
        is None
    )
    snapshot = await service.get(actor.telegram_id, actor.telegram_id)
    assert [message["content"] for message in snapshot.messages] == ["concurrent safe mutation"]


async def test_exchange_compensation_deletes_exact_pair_and_preserves_newer_messages(db):
    actor = await exchange_actor(db, 706)
    service = ConversationContextService(db, 10, 24)
    fence = await exchange_fence(service, actor, actor.telegram_id)
    receipt = await service.append_exchange(
        fence,
        user_content="exchange user",
        assistant_content="exchange assistant",
        user_source="text",
    )
    assert receipt is not None
    await service.append(
        actor.telegram_id,
        actor.telegram_id,
        role="user",
        content="newer concurrent message",
        source="text",
        intent="conversation",
    )

    assert await service.compensate_exchange(receipt)
    messages = (await service.get(actor.telegram_id, actor.telegram_id)).messages
    assert [(message["content"], message["intent"]) for message in messages] == [
        ("newer concurrent message", "conversation")
    ]


async def test_exchange_compensation_reconciles_full_window_counterfactual(db):
    actor = await exchange_actor(db, 709)
    service = ConversationContextService(db, 10, 24)
    for index in range(10):
        await service.append(
            actor.telegram_id,
            actor.telegram_id,
            role="user",
            content=f"prior-{index}",
            source="text",
            intent="conversation",
        )
    fence = await exchange_fence(service, actor, actor.telegram_id)
    receipt = await service.append_exchange(
        fence,
        user_content="exchange user",
        assistant_content="exchange assistant",
        user_source="text",
    )
    assert receipt is not None
    await service.append(
        actor.telegram_id,
        actor.telegram_id,
        role="user",
        content="newer",
        source="text",
        intent="conversation",
    )

    assert await service.compensate_exchange(receipt)
    messages = (await service.get(actor.telegram_id, actor.telegram_id)).messages
    assert [message["content"] for message in messages] == [
        *[f"prior-{index}" for index in range(1, 10)],
        "newer",
    ]


async def test_exchange_compensation_removes_surviving_half_after_concurrent_pruning(db):
    actor = await exchange_actor(db, 710)
    service = ConversationContextService(db, 10, 24)
    for index in range(10):
        await service.append(
            actor.telegram_id,
            actor.telegram_id,
            role="user",
            content=f"prior-{index}",
            source="text",
            intent="conversation",
        )
    fence = await exchange_fence(service, actor, actor.telegram_id)
    receipt = await service.append_exchange(
        fence,
        user_content="exchange user",
        assistant_content="exchange assistant",
        user_source="text",
    )
    assert receipt is not None
    for index in range(9):
        await service.append(
            actor.telegram_id,
            actor.telegram_id,
            role="user",
            content=f"newer-{index}",
            source="text",
            intent="conversation",
        )

    before = (await service.get(actor.telegram_id, actor.telegram_id)).messages
    assert "exchange user" not in [message["content"] for message in before]
    assert "exchange assistant" in [message["content"] for message in before]
    assert await service.compensate_exchange(receipt)
    messages = (await service.get(actor.telegram_id, actor.telegram_id)).messages
    assert [message["content"] for message in messages] == [
        "prior-9",
        *[f"newer-{index}" for index in range(9)],
    ]


async def test_expired_context_purge_is_bounded_and_preserves_active_session(db, caplog):
    service = ConversationContextService(db, 12, 24)
    private = "PRIVATE_CONVERSATION_PURGE_SENTINEL"
    for telegram_user_id in range(501, 505):
        await service.append(
            telegram_user_id,
            600,
            role="user",
            content=f"{private}:{telegram_user_id}",
            source="text",
            intent="conversation",
        )

    current = datetime.now(UTC)
    async with db.session() as session:
        rows = list(
            (
                await session.scalars(select(ConversationSession).order_by(ConversationSession.id))
            ).all()
        )
        for offset, row in enumerate(rows[:3], start=1):
            row.expires_at = current - timedelta(minutes=offset)
        rows[3].expires_at = current + timedelta(hours=1)

    with caplog.at_level(logging.INFO):
        result = await service.purge_expired(batch_size=2, now=current)
    assert result == 2
    assert type(result) is int
    assert private not in repr(result)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationSession.id))) == 2
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 2

    assert await service.purge_expired(batch_size=2, now=current) == 1
    assert await service.purge_expired(batch_size=2, now=current) == 0
    assert await service.purge_expired(batch_size=100, now=current) == 0
    active = await service.get(504, 600)
    assert len(active.messages) == 1
    assert private not in caplog.text


async def test_expired_context_purge_is_concurrent_and_idempotent(db):
    service = ConversationContextService(db, 12, 24)
    current = datetime.now(UTC)
    for telegram_user_id in range(511, 516):
        await service.append(
            telegram_user_id,
            610,
            role="user",
            content="expired",
            source="text",
            intent="conversation",
        )
    async with db.session() as session:
        rows = list((await session.scalars(select(ConversationSession))).all())
        for row in rows:
            row.expires_at = current - timedelta(seconds=1)

    deleted = await asyncio.gather(
        service.purge_expired(batch_size=5, now=current),
        service.purge_expired(batch_size=5, now=current),
    )
    assert sum(deleted) == 5
    assert await service.purge_expired(batch_size=5, now=current) == 0
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(ConversationSession.id))) == 0
        assert await session.scalar(select(func.count(ConversationMessage.id))) == 0


async def test_append_refresh_wins_purge_race_across_database_instances(db):
    telegram_user_id = 521
    chat_id = 620
    initial = ConversationContextService(db, 12, 24)
    original_session_id = await initial.append(
        telegram_user_id,
        chat_id,
        role="user",
        content="old expired message",
        source="text",
        intent="conversation",
    )
    current = datetime.now(UTC)
    async with db.session() as session:
        conversation = await session.get(ConversationSession, original_session_id)
        assert conversation is not None
        conversation.expires_at = current - timedelta(seconds=1)

    append_db = CommitGateDatabase(db.url)
    purge_db = SessionObservedDatabase(db.url)
    async with append_db.engine.connect() as connection:
        await connection.execute(select(1))
    async with purge_db.engine.connect() as connection:
        await connection.execute(select(1))
    appender = ConversationContextService(append_db, 12, 24)
    purger = ConversationContextService(purge_db, 12, 24)
    append_task = asyncio.create_task(
        appender.append(
            telegram_user_id,
            chat_id,
            role="user",
            content="fresh message",
            source="text",
            intent="conversation",
        )
    )
    purge_task = None
    try:
        await asyncio.wait_for(append_db.before_commit.wait(), timeout=5)
        purge_task = asyncio.create_task(purger.purge_expired(batch_size=1, now=current))
        await asyncio.wait_for(purge_db.session_started.wait(), timeout=5)
        append_db.allow_commit.set()

        assert await asyncio.wait_for(append_task, timeout=5) == original_session_id
        assert await asyncio.wait_for(purge_task, timeout=5) == 0
    finally:
        append_db.allow_commit.set()
        if not append_task.done():
            append_task.cancel()
            await asyncio.gather(append_task, return_exceptions=True)
        if purge_task is not None and not purge_task.done():
            purge_task.cancel()
            await asyncio.gather(purge_task, return_exceptions=True)
        await append_db.dispose()
        await purge_db.dispose()

    async with db.sessions() as session:
        conversations = list((await session.scalars(select(ConversationSession))).all())
        messages = list((await session.scalars(select(ConversationMessage))).all())
    assert len(conversations) == 1
    assert conversations[0].id == original_session_id
    assert not ConversationContextService._is_expired(conversations[0].expires_at, current)
    assert [message.content for message in messages] == ["fresh message"]


async def test_purge_wins_race_then_append_recreates_once_across_database_instances(db):
    telegram_user_id = 531
    chat_id = 630
    initial = ConversationContextService(db, 12, 24)
    original_session_id = await initial.append(
        telegram_user_id,
        chat_id,
        role="user",
        content="old expired message",
        source="text",
        intent="conversation",
    )
    current = datetime.now(UTC)
    async with db.session() as session:
        conversation = await session.get(ConversationSession, original_session_id)
        assert conversation is not None
        conversation.expires_at = current - timedelta(seconds=1)

    purge_db = CommitGateDatabase(db.url)
    append_db = SessionObservedDatabase(db.url)
    async with purge_db.engine.connect() as connection:
        await connection.execute(select(1))
    async with append_db.engine.connect() as connection:
        await connection.execute(select(1))
    purger = ConversationContextService(purge_db, 12, 24)
    appender = ConversationContextService(append_db, 12, 24)
    purge_task = asyncio.create_task(purger.purge_expired(batch_size=1, now=current))
    append_task = None
    try:
        await asyncio.wait_for(purge_db.before_commit.wait(), timeout=5)
        append_task = asyncio.create_task(
            appender.append(
                telegram_user_id,
                chat_id,
                role="user",
                content="fresh message",
                source="text",
                intent="conversation",
            )
        )
        await asyncio.wait_for(append_db.session_started.wait(), timeout=5)
        purge_db.allow_commit.set()

        assert await asyncio.wait_for(purge_task, timeout=5) == 1
        recreated_session_id = await asyncio.wait_for(append_task, timeout=5)
    finally:
        purge_db.allow_commit.set()
        if not purge_task.done():
            purge_task.cancel()
            await asyncio.gather(purge_task, return_exceptions=True)
        if append_task is not None and not append_task.done():
            append_task.cancel()
            await asyncio.gather(append_task, return_exceptions=True)
        await purge_db.dispose()
        await append_db.dispose()

    async with db.sessions() as session:
        conversations = list((await session.scalars(select(ConversationSession))).all())
        messages = list((await session.scalars(select(ConversationMessage))).all())
    assert len(conversations) == 1
    assert conversations[0].id == recreated_session_id
    # SQLite may reuse an integer primary key after hard deletion; the purge
    # outcome above proves that this is a newly inserted row either way.
    assert not ConversationContextService._is_expired(conversations[0].expires_at, current)
    assert len(messages) == 1
    assert messages[0].session_id == recreated_session_id
    assert messages[0].content == "fresh message"


@pytest.mark.parametrize("batch_size", [True, 0, 101])
async def test_expired_context_purge_rejects_unbounded_batch(db, batch_size):
    service = ConversationContextService(db, 12, 24)
    with pytest.raises(ValueError, match="batch_size"):
        await service.purge_expired(batch_size=batch_size)


async def test_system_action_begin_uses_unique_atomic_versions(db):
    service = ConversationContextService(db, 12, 24)
    assert await service.begin_system_action(110, 210, "bootstrap", [{"id": 0}]) == 1

    requests = (
        ("discard_all_active_drafts", [{"id": "draft"}]),
        ("archive_overdue_tasks", [{"id": 42}]),
    )
    versions = await asyncio.gather(
        *(service.begin_system_action(110, 210, action, snapshot) for action, snapshot in requests)
    )

    assert sorted(versions) == [2, 3]
    winning_action, winning_snapshot = requests[versions.index(3)]
    current = await service.get(110, 210)
    assert current.system_action_version == 3
    assert current.system_pending_action == winning_action
    assert current.system_draft_snapshot == winning_snapshot


async def test_system_action_claim_and_clear_are_bound_to_expected_version(db):
    service = ConversationContextService(db, 12, 24)
    old_version = await service.begin_system_action(
        120,
        220,
        "discard_all_active_drafts",
        [{"id": "old"}],
    )
    new_snapshot = [{"id": 99, "version": 7}]
    new_version = await service.begin_system_action(
        120,
        220,
        "archive_overdue_tasks",
        new_snapshot,
    )

    assert not await service.clear_system_action(
        120,
        220,
        expected_version=old_version,
    )
    assert await service.claim_system_action(120, 220, expected_version=old_version) is None
    still_current = await service.get(120, 220)
    assert still_current.system_action_version == new_version
    assert still_current.system_draft_snapshot == new_snapshot

    competing_claims = await asyncio.gather(
        service.claim_system_action(120, 220, expected_version=new_version),
        service.claim_system_action(120, 220, expected_version=new_version),
    )
    claims = [claim for claim in competing_claims if claim is not None]
    assert len(claims) == 1
    assert claims[0].action == "archive_overdue_tasks"
    assert claims[0].snapshot == new_snapshot
    assert claims[0].version == new_version
    assert (await service.get(120, 220)).system_pending_action is None

    latest_version = await service.begin_system_action(
        120,
        220,
        "trash_inbox_items",
        [{"id": 100}],
    )
    assert not await service.clear_system_action(
        120,
        220,
        expected_version=new_version,
    )
    latest = await service.get(120, 220)
    assert latest.system_action_version == latest_version
    assert latest.system_pending_action == "trash_inbox_items"
    assert await service.clear_system_action(
        120,
        220,
        expected_version=latest_version,
    )
    assert not await service.clear_system_action(
        120,
        220,
        expected_version=latest_version,
    )


async def test_system_action_confirm_and_cancel_have_exactly_one_winner(db):
    service = ConversationContextService(db, 12, 24)
    version = await service.begin_system_action(
        130,
        230,
        "trash_inbox_items",
        [{"id": 1}],
    )

    claim, cancelled = await asyncio.gather(
        service.claim_system_action(130, 230, expected_version=version),
        service.clear_system_action(130, 230, expected_version=version),
    )

    assert (claim is not None) != cancelled
    if claim is not None:
        assert await service.finalize_system_action_claim(
            130,
            230,
            expected_version=version,
        )
    assert (await service.get(130, 230)).system_pending_action is None


async def test_recent_conversation_reaches_next_message_after_bot_recreation(db, fake_ai):
    first_bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    first_message = FakeMessage("Давай обсудим еженедельное планирование")
    await first_bot.text(update_for(first_message, 300, 400), SimpleNamespace(user_data={}))

    second_bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    follow_up = FakeMessage(
        "Ну вот, мы только что общались про то, что еженедельное планирование полезно"
    )
    await second_bot.text(update_for(follow_up, 300, 400), SimpleNamespace(user_data={}))
    assert "обсуждали еженедельное планирование" in str(follow_up.replies[-1]["text"])
    recent = fake_ai.conversation_contexts[-1]["recent_messages"]
    assert any("еженедельное планирование" in row["content"] for row in recent)


async def test_task_question_offers_choices_but_saves_nothing(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    await bot.text(
        update_for(FakeMessage("Давай обсудим еженедельное планирование"), 310, 410),
        SimpleNamespace(user_data={}),
    )
    question = FakeMessage("Ты занесёшь это в задачу?")
    await bot.text(update_for(question, 310, 410), SimpleNamespace(user_data={}))
    labels = {
        button.text
        for row in question.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    }
    assert labels == {"Создать задачу", "Оставить идеей", "Уточнить дату", "Ничего"}
    assert await counts(db) == (0, 0)


async def test_save_this_creates_preview_from_unambiguous_context_only(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    original = "Давай обсудим еженедельное планирование"
    await bot.text(update_for(FakeMessage(original), 320, 420), SimpleNamespace(user_data={}))
    capture = FakeMessage("Да, сохрани это")
    await bot.text(update_for(capture, 320, 420), SimpleNamespace(user_data={}))
    assert "Исходный текст" in str(capture.replies[-1]["text"])
    assert original in str(capture.replies[-1]["text"])
    assert await counts(db) == (1, 0)


async def test_ambiguous_reference_is_not_guessed(db, fake_ai):
    service = ConversationContextService(db, 12, 24)
    for content in ("Первый подробный вариант плана", "Второй подробный вариант плана"):
        await service.append(
            330,
            430,
            role="user",
            content=content,
            source="text",
            intent="conversation",
        )
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    message = FakeMessage("Сохрани это")
    await bot.text(update_for(message, 330, 430), SimpleNamespace(user_data={}))
    assert "Уточни" in str(message.replies[-1]["text"])
    assert await counts(db) == (0, 0)


def test_date_resolver_detects_july_28_2026_and_nearest_sunday():
    resolver = DateResolver()
    result = resolver.resolve(
        "Начну в воскресенье, 28 июля 2026",
        "Europe/Moscow",
        now=datetime(2026, 7, 13, 10, tzinfo=UTC),
    )
    assert result.status == "conflict"
    assert result.actual_weekday == "вторник"
    assert result.options[0].value.isoformat() == "2026-07-26"
    assert result.options[0].weekday == "воскресенье"


def test_date_without_year_uses_nearest_future_and_timezone_affects_tomorrow():
    resolver = DateResolver()
    inferred = resolver.resolve(
        "Начну 28 июля", "Europe/Moscow", now=datetime(2026, 8, 1, tzinfo=UTC)
    )
    assert inferred.target_date.isoformat() == "2027-07-28"
    assert inferred.inferred_year is True
    moment = datetime(2026, 7, 12, 21, 30, tzinfo=UTC)
    assert resolver.resolve("завтра", "Europe/Moscow", now=moment).target_date.isoformat() == (
        "2026-07-14"
    )
    assert resolver.resolve("завтра", "America/New_York", now=moment).target_date.isoformat() == (
        "2026-07-13"
    )


async def test_date_conflict_handler_asks_and_creates_no_draft(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, FakeTranscription())
    message = FakeMessage("Начну с воскресенья, 28 июля 2026")
    await bot.text(update_for(message, 340, 440), SimpleNamespace(user_data={}))
    answer = str(message.replies[-1]["text"])
    assert "28.07.2026" in answer and "вторник" in answer
    assert "26.07.2026" in answer and "воскресенье" in answer
    assert await counts(db) == (0, 0)


async def test_context_logging_does_not_include_private_message(db, caplog):
    private = "очень личный полный текст пользователя"
    service = ConversationContextService(db, 12, 24)
    with caplog.at_level(logging.INFO):
        await service.append(
            350,
            450,
            role="user",
            content=private,
            source="text",
            intent="reflection",
        )
    assert private not in caplog.text


def test_latest_nova_memory_candidate_uses_only_latest_bounded_user_reply():
    snapshot = ConversationSnapshot(
        messages=[
            {
                "role": "user",
                "content": "  Я предпочитаю короткие ответы  ",
                "source": "voice",
                "intent": "conversation",
            },
            {
                "role": "assistant",
                "content": "Поняла.",
                "source": "text",
                "intent": "conversation",
            },
        ]
    )

    assert (
        ConversationContextService.latest_nova_memory_candidate(snapshot)
        == "Я предпочитаю короткие ответы"
    )


@pytest.mark.parametrize(
    ("content", "source", "intent"),
    [
        ("Nova, запомни: отвечай кратко", "text", "conversation"),
        ("обычная последняя реплика", "photo", "conversation"),
        ("обычная последняя реплика", "text", "navigation"),
        ("Напомни завтра позвонить", "text", "relative_reminder"),
        ("Выбираю первый вариант", "text", "confirm_date"),
        ("Сохрани это", "text", "explicit_capture"),
        ("текст с\x00управляющим символом", "text", "conversation"),
        ("x" * 501, "voice", "conversation"),
    ],
)
def test_latest_nova_memory_candidate_fails_closed_without_older_fallback(
    content,
    source,
    intent,
):
    snapshot = ConversationSnapshot(
        messages=[
            {
                "role": "user",
                "content": "старую подходящую реплику нельзя подставлять",
                "source": "text",
                "intent": "conversation",
            },
            {
                "role": "user",
                "content": content,
                "source": source,
                "intent": intent,
            },
        ]
    )

    assert ConversationContextService.latest_nova_memory_candidate(snapshot) is None
