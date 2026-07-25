import logging
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from autotester.fakes import FakeCallbackQuery, FakeMessage, FakeVoice, ScriptedTranscription
from sqlalchemy import func, select, update

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.knowledge_handlers import KnowledgeTelegramDownloadError
from future_self.models import (
    KnowledgeCaptureDraft,
    KnowledgeIngestionJob,
    KnowledgeSource,
    KnowledgeSourceRevision,
)

OWNER_ID = 8_220_001
OWNER_CHAT_ID = 8_230_001
OTHER_ID = 8_220_002


def settings(tmp_path) -> Settings:
    result = Settings(
        _env_file=None,
        telegram_bot_token="123456:test-token",
        ai_api_key="test-key",
        enable_knowledge_hub=True,
        enable_knowledge_capture=True,
        knowledge_asset_root="/tmp/myfutureselfai-tests/knowledge",
        runtime_min_free_bytes=100_000_000,
    )
    # Production config deliberately requires an absolute POSIX mount. Tests use
    # pytest's isolated native directory after Settings has validated the policy.
    result.knowledge_asset_root = str(tmp_path / "knowledge")
    return result


def context(*, args: list[str] | None = None, bot: Any = None) -> SimpleNamespace:
    return SimpleNamespace(user_data={}, args=args or [], bot=bot)


def update_for(
    message: FakeMessage,
    *,
    user_id: int = OWNER_ID,
    chat_id: int = OWNER_CHAT_ID,
    query: FakeCallbackQuery | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
    )


def callbacks(message: FakeMessage) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for reply in message.replies:
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        result.extend(
            (button.text, button.callback_data)
            for row in markup.inline_keyboard
            for button in row
            if button.callback_data
        )
    return result


def callback_by_label(message: FakeMessage, label: str) -> str:
    for button_label, data in reversed(callbacks(message)):
        if button_label == label:
            return data
    raise AssertionError(f"Missing callback for {label!r}")


async def click(
    bot: FutureSelfBot,
    message: FakeMessage,
    label: str,
    *,
    user_id: int = OWNER_ID,
    chat_id: int = OWNER_CHAT_ID,
) -> FakeCallbackQuery:
    query = FakeCallbackQuery(callback_by_label(message, label), message)
    await bot.knowledge_callback(
        update_for(message, user_id=user_id, chat_id=chat_id, query=query),
        context(),
    )
    return query


async def knowledge_counts(db) -> tuple[int, int, int]:
    async with db.sessions() as session:
        return (
            int(await session.scalar(select(func.count(KnowledgeSource.id))) or 0),
            int(await session.scalar(select(func.count(KnowledgeSourceRevision.id))) or 0),
            int(await session.scalar(select(func.count(KnowledgeIngestionJob.id))) or 0),
        )


async def test_text_capture_requires_explicit_confirm_and_uses_opaque_bound_callbacks(
    db, fake_ai, tmp_path
):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    assert (
        bot.knowledge_service.quota.max_extracted_bytes
        == bot.settings.knowledge_extraction_max_text_bytes
    )
    assert (
        bot.knowledge_service.quota.max_pending_jobs_per_space
        == bot.settings.knowledge_max_pending_jobs_per_space
    )
    assert (
        bot.knowledge_storage.max_extracted_bytes
        == bot.settings.knowledge_extraction_max_text_bytes
    )
    message = FakeMessage("/capture")
    private_text = "<script>private & source</script>"
    await bot.capture_command(update_for(message), context(args=[private_text]))

    assert await knowledge_counts(db) == (0, 0, 0)
    assert fake_ai.route_calls == []
    preview = message.replies[-1]["text"]
    assert "<script>" not in preview
    assert "&lt;script&gt;" in preview
    kh_callbacks = [data for _label, data in callbacks(message) if data.startswith("kh:")]
    assert kh_callbacks
    assert all(re.fullmatch(r"kh:[A-Za-z0-9_-]{20,48}", data) for data in kh_callbacks)
    assert all(len(data.encode("utf-8")) <= 64 for data in kh_callbacks)
    assert all(private_text not in data for data in kh_callbacks)

    confirm = callback_by_label(message, "Сохранить и обработать")
    wrong_actor = FakeCallbackQuery(confirm, message)
    await bot.knowledge_callback(
        update_for(message, user_id=OTHER_ID, query=wrong_actor), context()
    )
    assert any(show_alert for _text, show_alert in wrong_actor.answers)
    assert await knowledge_counts(db) == (0, 0, 0)

    wrong_chat = FakeCallbackQuery(confirm, message)
    await bot.knowledge_callback(
        update_for(message, chat_id=OWNER_CHAT_ID + 1, query=wrong_chat), context()
    )
    assert any(show_alert for _text, show_alert in wrong_chat.answers)
    assert await knowledge_counts(db) == (0, 0, 0)

    owner = FakeCallbackQuery(confirm, message)
    await bot.knowledge_callback(update_for(message, query=owner), context())
    assert await knowledge_counts(db) == (1, 1, 1)
    assert fake_ai.route_calls == []
    assert any("Материал сохранён" in reply["text"] for reply in message.replies)

    replay = FakeCallbackQuery(confirm, message)
    await bot.knowledge_callback(update_for(message, query=replay), context())
    assert any(show_alert for _text, show_alert in replay.answers)
    assert await knowledge_counts(db) == (1, 1, 1)


class MetadataOnlyDocument:
    file_id = "telegram-file-id"
    file_unique_id = "telegram-unique-id"
    file_name = "notes.txt"
    mime_type = "text/plain"
    file_size = 12

    def __init__(self) -> None:
        self.download_calls = 0

    async def get_file(self):
        self.download_calls += 1
        raise AssertionError("metadata-only preview downloaded a Telegram file")


class CountingVoice(FakeVoice):
    def __init__(self) -> None:
        self.download_calls = 0

    async def get_file(self):
        self.download_calls += 1
        return await super().get_file()


@pytest.mark.parametrize("pending_service", ["task_service", "collection_service"])
async def test_unclaimed_media_defers_to_task_and_collection_inputs_without_download(
    db, fake_ai, tmp_path, monkeypatch, pending_service
):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    media = MetadataOnlyDocument()

    async def pending(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(getattr(bot, pending_service), "pending_input", pending)
    message = FakeMessage(document=media)
    await bot.knowledge_media_gate(update_for(message), context())

    owner = await bot._user(OWNER_ID)
    state = await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)
    assert state.preview is None
    assert media.download_calls == 0
    assert await knowledge_counts(db) == (0, 0, 0)
    assert fake_ai.route_calls == []
    assert "активен другой сценарий" in message.replies[-1]["text"]


async def test_unclaimed_media_is_metadata_only_until_cancel(db, fake_ai, tmp_path):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    media = MetadataOnlyDocument()
    message = FakeMessage(document=media)
    await bot.knowledge_media_gate(update_for(message), context())

    owner = await bot._user(OWNER_ID)
    state = await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)
    assert state.preview is not None
    assert state.preview.capture_kind == "document"
    assert media.download_calls == 0
    assert await knowledge_counts(db) == (0, 0, 0)
    assert not any(bot.knowledge_storage.originals_root.iterdir())

    cancel_message = FakeMessage("/cancel")
    await bot.cancel_draft_edit(update_for(cancel_message), context())
    assert (await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)).preview is None
    assert await knowledge_counts(db) == (0, 0, 0)
    assert "Capture отменён" in cancel_message.replies[-1]["text"]


@pytest.mark.parametrize("flow_key", ["health_checkin", "doctor_prepare"])
async def test_medical_flow_keeps_capture_and_media_download_outside_the_flow(
    db, fake_ai, tmp_path, flow_key
):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    media = MetadataOnlyDocument()
    message = FakeMessage(document=media)
    medical_context = context()
    medical_context.user_data[flow_key] = {"active": True}
    await bot.knowledge_media_gate(update_for(message), medical_context)

    owner = await bot._user(OWNER_ID)
    assert (await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)).preview is None
    assert media.download_calls == 0
    assert await knowledge_counts(db) == (0, 0, 0)
    assert fake_ai.route_calls == []


@pytest.mark.parametrize("capture_enabled", [False, True])
@pytest.mark.parametrize("medical_flow", ["health_checkin", "doctor_prepare", "labs"])
async def test_medical_flow_voice_is_rejected_before_download_stt_and_llm(
    db, fake_ai, tmp_path, monkeypatch, medical_flow, capture_enabled
):
    transcription = ScriptedTranscription()
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, transcription)
    bot.settings.enable_knowledge_capture = capture_enabled
    medical_context = context()
    if medical_flow == "labs":

        async def active_lab_upload(*_args, **_kwargs):
            return True

        monkeypatch.setattr(bot.lab_uploads, "has_active", active_lab_upload)
    else:
        medical_context.user_data[medical_flow] = {"active": True}

    voice = CountingVoice()
    message = FakeMessage(voice=voice)
    await bot.voice(update_for(message), medical_context)

    assert voice.download_calls == 0
    assert transcription.calls == []
    assert fake_ai.route_calls == []
    assert "не отправляется на распознавание или в LLM" in message.replies[-1]["text"]


async def test_nonmedical_specialized_flow_keeps_existing_voice_behavior(db, fake_ai, tmp_path):
    transcription = ScriptedTranscription()
    transcription.queue("меню")
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, transcription)
    flow_context = context()
    flow_context.user_data["onboarding_user_id"] = OWNER_ID
    voice = CountingVoice()
    message = FakeMessage(voice=voice)

    await bot.voice(update_for(message), flow_context)

    assert voice.download_calls == 1
    assert len(transcription.calls) == 1
    assert fake_ai.route_calls == []


async def test_normal_text_stays_generic_but_forwarded_text_opens_unconfirmed_capture(
    db, fake_ai, tmp_path
):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    normal = FakeMessage("привет")
    await bot.text(update_for(normal), context())
    assert len(fake_ai.route_calls) == 1
    assert await knowledge_counts(db) == (0, 0, 0)

    forwarded = FakeMessage("Forwarded source text")
    forwarded.forward_origin = SimpleNamespace(date=datetime.now(UTC))
    await bot.text(update_for(forwarded), context())
    owner = await bot._user(OWNER_ID)
    preview = (await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)).preview
    assert preview is not None and preview.capture_kind == "forward"
    assert preview.declared_mime == "text/plain"
    assert len(fake_ai.route_calls) == 1
    assert await knowledge_counts(db) == (0, 0, 0)


async def test_url_capture_never_fetches_external_page(db, fake_ai, tmp_path, monkeypatch):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    url = "https://example.com/private/path?opaque=value"
    message = FakeMessage("/capture")
    await bot.capture_command(update_for(message), context(args=[url]))
    assert all(url not in data for _label, data in callbacks(message))

    class NoNetworkClient:
        def __init__(self, **_kwargs):
            raise AssertionError("URL Capture attempted an external HTTP request")

    monkeypatch.setattr("future_self.knowledge_handlers.httpx.AsyncClient", NoNetworkClient)
    await click(bot, message, "Сохранить и обработать")

    async with db.sessions() as session:
        source = await session.scalar(select(KnowledgeSource))
        job = await session.scalar(select(KnowledgeIngestionJob))
    assert source is not None and source.source_type == "url"
    assert source.processing_status == "partial"
    assert job is not None and job.safe_error_code == "external_fetch_disabled"
    assert fake_ai.route_calls == []


async def test_expired_capture_and_open_capture_voice_never_fall_through(db, fake_ai, tmp_path):
    transcription = ScriptedTranscription()
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, transcription)
    command = FakeMessage("/capture")
    await bot.capture_command(update_for(command), context())

    voice = CountingVoice()
    voice_message = FakeMessage(voice=voice)
    await bot.voice(update_for(voice_message), context())
    assert voice.download_calls == 0
    assert transcription.calls == []
    assert fake_ai.route_calls == []

    owner = await bot._user(OWNER_ID)
    async with db.session() as session:
        await session.execute(
            update(KnowledgeCaptureDraft)
            .where(
                KnowledgeCaptureDraft.actor_user_id == owner.id,
                KnowledgeCaptureDraft.chat_id == OWNER_CHAT_ID,
            )
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    text = FakeMessage("Это не должно попасть в LLM")
    await bot.text(update_for(text), context())
    assert fake_ai.route_calls == []
    assert "не отправлено в LLM" in text.replies[-1]["text"]


async def test_old_capture_callback_is_fail_closed_after_runtime_disable(db, fake_ai, tmp_path):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    message = FakeMessage("/knowledge")
    await bot.knowledge_command(update_for(message), context())
    old_callback = callback_by_label(message, "➕ Добавить материал")
    bot.settings.enable_knowledge_capture = False

    query = FakeCallbackQuery(old_callback, message)
    await bot.knowledge_callback(update_for(message, query=query), context())

    assert any(
        show_alert and text and "старое действие не выполнено" in text
        for text, show_alert in query.answers
    )
    owner = await bot._user(OWNER_ID)
    assert (await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)).preview is None
    async with db.sessions() as session:
        assert int(await session.scalar(select(func.count(KnowledgeCaptureDraft.id))) or 0) == 0


async def test_purge_failed_card_is_owner_retryable_without_restore(db, fake_ai, tmp_path):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    capture = FakeMessage("/capture")
    await bot.capture_command(update_for(capture), context(args=["material to purge"]))
    await click(bot, capture, "Сохранить и обработать")
    owner = await bot._user(OWNER_ID)
    personal = await bot.knowledge_service.ensure_personal_space(owner.id)
    async with db.sessions() as session:
        source = await session.scalar(select(KnowledgeSource))
        assert source is not None
        source_public_id = source.public_id
        source_version = source.version

    trashed = await bot.knowledge_service.trash_source(owner.id, source_public_id, source_version)
    await bot.knowledge_service.request_permanent_delete(
        owner.id,
        source_public_id,
        trashed.version,
        max_attempts=bot.settings.knowledge_runner_max_attempts,
    )
    failed_at = datetime.now(UTC)
    async with db.session() as session:
        source = await session.scalar(
            select(KnowledgeSource).where(KnowledgeSource.public_id == source_public_id)
        )
        purge_job = await session.scalar(
            select(KnowledgeIngestionJob)
            .where(
                KnowledgeIngestionJob.source_id == source.id,
                KnowledgeIngestionJob.job_type == "purge",
            )
            .order_by(KnowledgeIngestionJob.id.desc())
        )
        assert source is not None and purge_job is not None
        source.lifecycle_status = "purge_failed"
        source.version += 1
        purge_job.status = "failed"
        purge_job.safe_error_code = "purge_io_failed"
        purge_job.finished_at = failed_at

    record = await bot.knowledge_service.get_source(
        owner.id, source_public_id, include_trashed=True
    )
    message = FakeMessage()
    query = FakeCallbackQuery("unused", message)
    await bot._render_source_card(
        query,
        owner.id,
        OWNER_CHAT_ID,
        personal.access.space_public_id,
        record,
    )

    assert "Окончательное удаление не завершено" in message.replies[-1]["text"]
    labels = [label for label, _data in callbacks(message)]
    assert "Повторить удаление" in labels
    assert "Восстановить" not in labels

    await click(bot, message, "Повторить удаление")
    async with db.sessions() as session:
        source = await session.scalar(
            select(KnowledgeSource).where(KnowledgeSource.public_id == source_public_id)
        )
        purge_jobs = list(
            (
                await session.scalars(
                    select(KnowledgeIngestionJob)
                    .where(
                        KnowledgeIngestionJob.source_id == source.id,
                        KnowledgeIngestionJob.job_type == "purge",
                    )
                    .order_by(KnowledgeIngestionJob.id)
                )
            ).all()
        )
    assert source is not None and source.lifecycle_status == "purge_pending"
    assert len(purge_jobs) == 2 and purge_jobs[-1].status == "queued"


async def test_cross_feature_callback_cancels_capture_state(db, fake_ai, tmp_path):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    command = FakeMessage("/capture")
    await bot.capture_command(update_for(command), context())
    owner = await bot._user(OWNER_ID)
    assert (await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)).preview

    query = FakeCallbackQuery("task:opaque-other-feature-token", command)
    await bot.knowledge_other_callback_gate(update_for(command, query=query), context())
    assert (await bot.knowledge_service.capture_state(owner.id, OWNER_CHAT_ID)).preview is None
    assert await knowledge_counts(db) == (0, 0, 0)


class StreamResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def aiter_bytes(self, _size):
        yield b"safe-stream"


class RecordingClient:
    created_with: dict[str, Any] = {}
    requested_url = ""

    def __init__(self, **kwargs):
        type(self).created_with = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, method: str, url: str):
        assert method == "GET"
        type(self).requested_url = url
        return StreamResponse()


async def test_telegram_download_boundary_is_streamed_strict_and_log_safe(
    db, fake_ai, tmp_path, monkeypatch, caplog
):
    bot = FutureSelfBot(settings(tmp_path), db, fake_ai, ScriptedTranscription())
    fake_token = "123456:FAKE_PRIVATE_TOKEN"

    class DownloadBot:
        base_file_url = f"https://api.telegram.org/file/bot{fake_token}"

        async def get_file(self, _file_id):
            return SimpleNamespace(file_path=f"{self.base_file_url}/documents/source.txt")

    monkeypatch.setattr("future_self.knowledge_handlers.httpx.AsyncClient", RecordingClient)
    caplog.set_level(logging.DEBUG)
    chunks = [chunk async for chunk in bot._telegram_file_chunks(DownloadBot(), "opaque-id")]

    assert chunks == [b"safe-stream"]
    assert RecordingClient.created_with["follow_redirects"] is False
    assert RecordingClient.created_with["trust_env"] is False
    assert RecordingClient.requested_url.startswith("https://api.telegram.org/file/bot")
    assert fake_token not in caplog.text

    class EvilDownloadBot(DownloadBot):
        async def get_file(self, _file_id):
            return SimpleNamespace(file_path="https://evil.example/file.txt")

    with pytest.raises(KnowledgeTelegramDownloadError):
        _ = [chunk async for chunk in bot._telegram_file_chunks(EvilDownloadBot(), "opaque-id")]
    assert fake_token not in caplog.text
