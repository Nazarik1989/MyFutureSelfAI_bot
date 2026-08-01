import base64
import json
import logging
from asyncio import gather
from datetime import date
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from autotester.fakes import (
    FakeCallbackQuery,
    FakeImageMedia,
    FakeMediaCallbackQuery,
    FakeMessage,
    ScriptedTranscription,
)
from PIL import Image, PngImagePlugin
from sqlalchemy import func, select
from telegram.ext import ApplicationHandlerStop

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.image_generation import (
    ImageGenerationError,
    ImageReferenceInput,
    OpenRouterImageGenerationService,
    build_vision_image_prompt,
    create_image_generation_service,
)
from future_self.models import VisionItem, VisionItemImage
from future_self.vision_images import (
    MAX_IMAGE_DISPLAY_DIMENSION,
    MAX_IMAGE_INPUT_BYTES,
    MAX_IMAGE_OUTPUT_BYTES,
    MAX_IMAGE_PIXELS,
    NormalizedVisionImage,
    TelegramImageMetadata,
    VisionImageError,
    VisionImageSessionStore,
    normalize_vision_image,
    validate_telegram_metadata,
)
from future_self.vision_renderer import VisionBoardRenderer, VisionRenderItem


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


class FakeImageGeneration:
    enabled = True
    model = "openai/gpt-image-2"
    quality = "medium"
    size = "1024x1024"

    def __init__(self, result: bytes | None = None, error: str | None = None):
        self.result = result
        self.error = error
        self.prompts: list[str] = []
        self.reference_batches: list[tuple[ImageReferenceInput, ...]] = []

    async def generate(self, prompt: str, *, references=()) -> bytes:
        self.prompts.append(prompt)
        self.reference_batches.append(tuple(references))
        if self.error:
            raise ImageGenerationError(self.error)
        assert self.result is not None
        return self.result


def image_bytes(
    image_format: str,
    *,
    size: tuple[int, int] = (320, 180),
    color: tuple[int, ...] = (40, 120, 200),
    metadata: bool = False,
) -> bytes:
    mode = "RGBA" if len(color) == 4 else "RGB"
    image = Image.new(mode, size, color)
    output = BytesIO()
    kwargs: dict[str, object] = {}
    if image_format == "JPEG" and metadata:
        exif = Image.Exif()
        exif[274] = 6
        exif[270] = "private-description"
        kwargs = {"exif": exif, "icc_profile": b"private-icc"}
    elif image_format == "PNG" and metadata:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", "private-comment")
        kwargs = {"pnginfo": info, "icc_profile": b"private-icc"}
    image.save(output, format=image_format, **kwargs)
    image.close()
    return output.getvalue()


def callback_from(message: FakeMessage, prefix: str) -> str:
    for reply in reversed(message.replies):
        markup = reply.get("reply_markup")
        if markup is None:
            continue
        for row in markup.inline_keyboard:
            for button in row:
                if button.callback_data and button.callback_data.startswith(prefix):
                    return button.callback_data
    raise AssertionError(f"Missing callback {prefix!r}")


def update_for(message: FakeMessage, *, user_id: int, chat_id: int, chat_type="private"):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=None,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
    )


def callback_update(
    data: str,
    message: FakeMessage,
    *,
    user_id: int,
    chat_id: int,
    media: bool = False,
):
    query = FakeMediaCallbackQuery(data, message) if media else FakeCallbackQuery(data, message)
    return (
        SimpleNamespace(
            effective_message=message,
            callback_query=query,
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=chat_id, type="private"),
        ),
        query,
    )


async def add_item(db, owner_id: int, wish: str = "Личное желание") -> VisionItem:
    async with db.session() as session:
        item = VisionItem(
            owner_id=owner_id,
            category="travel",
            wish_text=wish,
            status="active",
        )
        session.add(item)
        await session.flush()
        return item


@pytest.mark.parametrize(
    ("image_format", "mime_type"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_normalization_accepts_static_formats_and_emits_bounded_metadata_free_rgb(
    image_format,
    mime_type,
):
    raw = image_bytes(image_format, size=(2200, 1200), metadata=image_format != "WEBP")
    normalized = normalize_vision_image(raw, declared_mime=mime_type)
    assert normalized.mime_type == "image/jpeg"
    assert len(normalized.image_bytes) <= MAX_IMAGE_OUTPUT_BYTES
    assert max(normalized.width, normalized.height) <= MAX_IMAGE_DISPLAY_DIMENSION
    assert len(normalized.sha256) == 64

    result = Image.open(BytesIO(normalized.image_bytes))
    assert result.format == "JPEG"
    assert result.mode == "RGB"
    assert result.size == (normalized.width, normalized.height)
    assert not result.getexif()
    assert "exif" not in result.info
    assert "icc_profile" not in result.info
    assert "comment" not in result.info
    result.verify()


def test_exif_orientation_is_applied_before_metadata_is_removed():
    raw = image_bytes("JPEG", size=(300, 120), metadata=True)
    normalized = normalize_vision_image(raw, declared_mime="image/jpeg")
    assert normalized.height > normalized.width
    result = Image.open(BytesIO(normalized.image_bytes))
    assert not result.getexif()


def test_alpha_is_flattened_to_safe_rgb_and_output_is_deterministic():
    raw = image_bytes("PNG", color=(10, 50, 200, 70))
    first = normalize_vision_image(raw, declared_mime="image/png")
    second = normalize_vision_image(raw, declared_mime="image/png")
    assert first == second
    assert Image.open(BytesIO(first.image_bytes)).mode == "RGB"


@pytest.mark.parametrize(
    ("payload", "mime_type"),
    [
        (b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/png"),
        (b"%PDF-1.7", "image/png"),
        (b"not-an-image", "image/jpeg"),
    ],
)
def test_corrupt_and_non_raster_payloads_are_rejected(payload, mime_type):
    with pytest.raises(VisionImageError):
        normalize_vision_image(payload, declared_mime=mime_type)


def test_mime_magic_mismatch_and_animation_are_rejected():
    png = image_bytes("PNG")
    with pytest.raises(VisionImageError, match="mime_mismatch"):
        normalize_vision_image(png, declared_mime="image/jpeg")

    first = Image.new("RGB", (24, 24), "red")
    second = Image.new("RGB", (24, 24), "blue")
    output = BytesIO()
    first.save(output, format="GIF", save_all=True, append_images=[second], duration=100, loop=0)
    first.close()
    second.close()
    with pytest.raises(VisionImageError):
        normalize_vision_image(output.getvalue(), declared_mime=None)


def test_decompression_bomb_warning_is_rejected(monkeypatch):
    raw = image_bytes("PNG", size=(40, 40))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(VisionImageError, match="decompression_bomb"):
        normalize_vision_image(raw, declared_mime="image/png")


def test_metadata_precheck_rejects_size_pixels_and_unsupported_documents():
    validate_telegram_metadata(TelegramImageMetadata("photo", 1000, None, width=100, height=100))
    validate_telegram_metadata(TelegramImageMetadata("document", 1000, "image/webp"))
    with pytest.raises(VisionImageError, match="input_too_large"):
        validate_telegram_metadata(
            TelegramImageMetadata("document", MAX_IMAGE_INPUT_BYTES + 1, "image/jpeg")
        )
    with pytest.raises(VisionImageError, match="too_many_pixels"):
        validate_telegram_metadata(
            TelegramImageMetadata("photo", 1000, None, width=MAX_IMAGE_PIXELS, height=2)
        )
    for mime_type in ("image/svg+xml", "image/gif", "application/pdf", "image/heic"):
        with pytest.raises(VisionImageError, match="unsupported_mime"):
            validate_telegram_metadata(TelegramImageMetadata("document", 1000, mime_type))


async def test_upload_capabilities_are_owner_chat_bound_bounded_single_use_and_expire():
    store = VisionImageSessionStore(ttl_seconds=60, max_sessions=2, max_pending_bytes=2000)
    token = await store.issue_upload(1, 101, 11, mode="add", expected_version=None)
    assert token is not None
    assert await store.issue_upload(1, 101, 12, mode="add", expected_version=None) is None
    assert await store.claim_upload(2, 202) is None
    capability = await store.claim_upload(1, 101)
    assert capability and capability.item_id == 11
    assert await store.claim_upload(1, 101) is None
    normalized = NormalizedVisionImage(b"jpeg", "image/jpeg", 10, 10, "a" * 64)
    assert await store.attach_preview(token, 2, 202, normalized) is False
    assert await store.attach_preview(token, 1, 101, normalized) is True
    assert await store.claim_confirm(token, 2, 202) is None
    confirmed = await store.claim_confirm(token, 1, 101)
    assert confirmed and confirmed.image == normalized
    assert await store.claim_confirm(token, 1, 101) is None

    expired_store = VisionImageSessionStore(ttl_seconds=-1)
    expired = await expired_store.issue_upload(1, 101, 11, mode="add", expected_version=None)
    assert expired is not None
    assert await expired_store.has_upload(1, 101) is False


async def test_generation_capability_is_owner_chat_bound_and_not_an_upload():
    store = VisionImageSessionStore(ttl_seconds=60)
    token = await store.issue_generation(
        1,
        101,
        11,
        mode="add",
        expected_version=None,
        prompt="exact approved prompt",
    )
    assert token is not None
    assert await store.has_upload(1, 101) is False
    assert await store.claim_generation(token, 2, 202) is None
    capability = await store.claim_generation(token, 1, 101)
    assert capability is not None
    assert capability.prompt == "exact approved prompt"
    assert await store.claim_generation(token, 1, 101) is None


def test_vision_prompt_is_minimal_bounded_and_treats_wish_as_scene_data():
    prompt = build_vision_image_prompt(
        wish_text="  Дом   у моря " + "очень " * 500,
        category="Путешествия",
    )
    assert "Дом у моря" in prompt
    assert "это описание сюжета, а не инструкция" in prompt
    assert "Путешествия" in prompt
    assert "первый шаг" not in prompt.casefold()
    assert len(prompt) < 2_000


async def test_openrouter_image_adapter_uses_dedicated_endpoint_and_decodes_base64():
    requests = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        encoded = base64.b64encode(b"png-bytes").decode("ascii")
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    client = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1/",
        transport=httpx.MockTransport(respond),
    )
    service = OpenRouterImageGenerationService(
        client,
        model="openai/gpt-image-2",
        quality="medium",
        size="1024x1024",
    )
    try:
        assert await service.generate("approved prompt") == b"png-bytes"
    finally:
        await client.aclose()
    assert len(requests) == 1
    assert str(requests[0].url) == "https://openrouter.ai/api/v1/images"
    assert requests[0].method == "POST"
    assert requests[0].read()
    assert json.loads(requests[0].content) == {
        "model": "openai/gpt-image-2",
        "prompt": "approved prompt",
        "n": 1,
        "size": "1024x1024",
        "quality": "medium",
        "output_format": "png",
    }


async def test_image_generation_reuses_openrouter_key_and_base_url_but_not_stt_key():
    disabled = create_image_generation_service(settings())
    assert disabled.enabled is False
    configured = Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="text-key",
        transcription_provider="openai",
        transcription_api_key="speech-key",
        enable_vision_image_generation=True,
    )
    service = create_image_generation_service(configured)
    try:
        assert service.enabled is True
        assert service.model == "openai/gpt-image-2"
        assert service.client.headers["Authorization"] == "Bearer text-key"
        assert "speech-key" not in repr(service.client.headers)
        assert str(service.client.base_url) == "https://openrouter.ai/api/v1/"
    finally:
        await service.close()


def test_enabled_generation_requires_openrouter_and_rejects_another_model():
    with pytest.raises(ValueError, match="AI_PROVIDER=openrouter"):
        Settings(
            _env_file=None,
            telegram_bot_token="123456:TEST",
            ai_api_key="text-key",
            ai_provider="openai",
            enable_vision_image_generation=True,
        )
    with pytest.raises(ValueError, match="openai/gpt-image-2"):
        Settings(
            _env_file=None,
            telegram_bot_token="123456:TEST",
            ai_api_key="text-key",
            image_generation_model="openai/gpt-image-1",
        )


async def test_service_add_replace_delete_are_owner_scoped_versioned_and_idempotent(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(9101)
    foreign = await bot._user(9102)
    item = await add_item(db, owner.id)
    first = normalize_vision_image(
        image_bytes("JPEG", color=(20, 40, 60)), declared_mime="image/jpeg"
    )
    second = normalize_vision_image(
        image_bytes("PNG", color=(80, 100, 120)), declared_mime="image/png"
    )

    created = await bot.vision_image_service.save(
        owner.id, item.id, expected_version=None, normalized=first
    )
    assert created.status == "created"
    assert created.image.version == 1
    duplicate = await bot.vision_image_service.save(
        owner.id, item.id, expected_version=None, normalized=first
    )
    assert duplicate.status == "existing"
    assert (
        await bot.vision_image_service.save(
            foreign.id, item.id, expected_version=None, normalized=second
        )
    ).status == "stale"
    assert (
        await bot.vision_image_service.save(
            owner.id, item.id, expected_version=99, normalized=second
        )
    ).status == "stale"
    replaced = await bot.vision_image_service.save(
        owner.id, item.id, expected_version=1, normalized=second
    )
    assert replaced.status == "replaced"
    assert replaced.image.version == 2
    assert (
        await bot.vision_image_service.delete(foreign.id, item.id, expected_version=2)
    ).status == "stale"
    assert (
        await bot.vision_image_service.delete(owner.id, item.id, expected_version=1)
    ).status == "stale"
    assert (
        await bot.vision_image_service.delete(owner.id, item.id, expected_version=2)
    ).status == "deleted"
    assert await bot.vision_image_service.get(owner.id, item.id) is None


async def test_concurrent_add_has_one_atomic_winner_and_card_delete_cascades(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(9201)
    item = await add_item(db, owner.id)
    first = normalize_vision_image(image_bytes("JPEG", color=(1, 2, 3)), declared_mime="image/jpeg")
    second = normalize_vision_image(
        image_bytes("JPEG", color=(3, 2, 1)), declared_mime="image/jpeg"
    )
    results = await gather(
        bot.vision_image_service.save(owner.id, item.id, expected_version=None, normalized=first),
        bot.vision_image_service.save(owner.id, item.id, expected_version=None, normalized=second),
    )
    assert sorted(result.status for result in results) == ["created", "stale"]
    assert await bot.vision_service.delete_item(owner.id, item.id) is True
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(VisionItemImage.id))) == 0


async def test_concurrent_replace_and_delete_cannot_resurrect_or_partially_update(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(9251)
    item = await add_item(db, owner.id)
    first = normalize_vision_image(
        image_bytes("JPEG", color=(5, 10, 15)), declared_mime="image/jpeg"
    )
    second = normalize_vision_image(
        image_bytes("PNG", color=(15, 10, 5)), declared_mime="image/png"
    )
    assert (
        await bot.vision_image_service.save(
            owner.id, item.id, expected_version=None, normalized=first
        )
    ).status == "created"
    replace, removal = await gather(
        bot.vision_image_service.save(
            owner.id,
            item.id,
            expected_version=1,
            normalized=second,
        ),
        bot.vision_image_service.delete(owner.id, item.id, expected_version=1),
    )
    statuses = {replace.status, removal.status}
    assert statuses in ({"replaced", "stale"}, {"deleted", "stale"})
    stored = await bot.vision_image_service.get(owner.id, item.id)
    if stored is not None:
        assert stored.version == 2
        assert stored.sha256 == second.sha256


async def test_handler_photo_preview_confirm_repeat_replace_cancel_and_delete(
    db,
    fake_ai,
    tmp_path,
):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    telegram_id, chat_id = 9301, 19301
    owner = await bot._user(telegram_id)
    item = await add_item(db, owner.id)
    before = set(tmp_path.iterdir())
    card = FakeMessage()
    await bot._vision_send_item(card, item)

    add_update, _ = callback_update(
        callback_from(card, "vision:imageadd:"), card, user_id=telegram_id, chat_id=chat_id
    )
    await bot.vision_action(add_update, None)
    jpeg = image_bytes("JPEG", color=(10, 100, 200))
    upload = FakeMessage(photo=[FakeImageMedia(jpeg, mime_type=None, width=320, height=180)])
    with pytest.raises(ApplicationHandlerStop):
        await bot.vision_image_gate(update_for(upload, user_id=telegram_id, chat_id=chat_id), None)
    preview = [reply for reply in upload.replies if reply.get("kind") == "photo"]
    assert len(preview) == 1
    confirm = callback_from(upload, "vision:imageconfirm:")
    confirm_update, _ = callback_update(confirm, upload, user_id=telegram_id, chat_id=chat_id)
    await bot.vision_action(confirm_update, None)
    stored = await bot.vision_image_service.get(owner.id, item.id)
    assert stored and stored.version == 1
    assert stored.image_bytes != jpeg
    replay_update, replay_query = callback_update(
        confirm, upload, user_id=telegram_id, chat_id=chat_id
    )
    await bot.vision_action(replay_update, None)
    assert any(show_alert for _text, show_alert in replay_query.answers)
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(VisionItemImage.id))) == 1

    latest_card = upload
    replace_update, _ = callback_update(
        callback_from(latest_card, "vision:imagereplace:"),
        latest_card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(replace_update, None)
    png = image_bytes("PNG", color=(200, 80, 20))
    replacement = FakeMessage(document=FakeImageMedia(png, mime_type="image/png"))
    with pytest.raises(ApplicationHandlerStop):
        await bot.vision_image_gate(
            update_for(replacement, user_id=telegram_id, chat_id=chat_id), None
        )
    cancel_update, cancel_query = callback_update(
        callback_from(replacement, "vision:imagecancel:"),
        replacement,
        user_id=telegram_id,
        chat_id=chat_id,
        media=True,
    )
    await bot.vision_action(cancel_update, None)
    assert cancel_query.text_attempts == 1
    assert any("Действие с изображением отменено" in text for text in cancel_query.caption_edits)
    unchanged = await bot.vision_image_service.get(owner.id, item.id)
    assert unchanged and unchanged.version == 1 and unchanged.sha256 == stored.sha256

    delete_update, _ = callback_update(
        callback_from(latest_card, "vision:imagedeleteask:"),
        latest_card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(delete_update, None)
    delete_data = callback_from(latest_card, "vision:imagedelete:")
    forged_update, forged_query = callback_update(
        delete_data, latest_card, user_id=9302, chat_id=19302
    )
    await bot.vision_action(forged_update, None)
    assert any(show_alert for _text, show_alert in forged_query.answers)
    assert await bot.vision_image_service.get(owner.id, item.id) is not None
    owner_delete, _ = callback_update(
        delete_data, latest_card, user_id=telegram_id, chat_id=chat_id
    )
    await bot.vision_action(owner_delete, None)
    assert await bot.vision_image_service.get(owner.id, item.id) is None
    assert fake_ai.route_calls == []
    assert set(tmp_path.iterdir()) == before


async def test_gpt_image_2_requires_consent_previews_then_saves_once(db, fake_ai):
    generated_png = image_bytes("PNG", size=(1024, 1024), color=(70, 130, 210))
    generator = FakeImageGeneration(generated_png)
    configured = Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="text-key",
        enable_vision_image_generation=True,
    )
    bot = FutureSelfBot(
        configured,
        db,
        fake_ai,
        ScriptedTranscription(),
        image_generation=generator,
    )
    telegram_id, chat_id = 9351, 19351
    owner = await bot._user(telegram_id)
    async with db.session() as session:
        item = VisionItem(
            owner_id=owner.id,
            category="travel",
            wish_text="Уютный дом у океана",
            why_text="секретная личная причина",
            first_step="секретный первый шаг",
            status="active",
        )
        session.add(item)
        await session.flush()

    card = FakeMessage()
    await bot._vision_send_item(card, item)
    ask_update, _ = callback_update(
        callback_from(card, "vision:imagegenerateask:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(ask_update, None)
    assert generator.prompts == []
    disclosure = card.replies[-1]["text"]
    assert "OpenRouter" in disclosure
    assert "gpt-image-2" in disclosure
    assert "Точный запрос" in disclosure
    assert "Уютный дом у океана" in disclosure
    assert "секретная личная причина" not in disclosure
    assert "секретный первый шаг" not in disclosure

    generate_data = callback_from(card, "vision:imagegenerate:")
    generate_update, _ = callback_update(
        generate_data,
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(generate_update, None)
    assert len(generator.prompts) == 1
    assert await bot.vision_image_service.get(owner.id, item.id) is None
    assert any(reply.get("kind") == "photo" for reply in card.replies)

    replay_update, replay_query = callback_update(
        generate_data,
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(replay_update, None)
    assert len(generator.prompts) == 1
    assert any(show_alert for _text, show_alert in replay_query.answers)

    confirm_data = callback_from(card, "vision:imageconfirm:")
    forged_update, forged_query = callback_update(
        confirm_data,
        card,
        user_id=9352,
        chat_id=19352,
    )
    await bot.vision_action(forged_update, None)
    assert any(show_alert for _text, show_alert in forged_query.answers)
    assert await bot.vision_image_service.get(owner.id, item.id) is None

    confirm_update, _ = callback_update(
        confirm_data,
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(confirm_update, None)
    stored = await bot.vision_image_service.get(owner.id, item.id)
    assert stored is not None
    assert stored.mime_type == "image/jpeg"


async def test_main_visualization_entry_picks_active_wish_without_rendering_png(db, fake_ai):
    generator = FakeImageGeneration()
    bot = FutureSelfBot(
        settings(),
        db,
        fake_ai,
        ScriptedTranscription(),
        image_generation=generator,
    )
    telegram_id, chat_id = 9355, 19355
    owner = await bot._user(telegram_id)
    active = await add_item(db, owner.id, "Я работаю над любимым проектом у океана")
    archived = await add_item(db, owner.id, "Устаревшее желание")
    await bot.vision_service.set_status(owner.id, archived.id, "archived")

    message = FakeMessage("/vision")
    await bot.vision_command(update_for(message, user_id=telegram_id, chat_id=chat_id), None)
    picker_data = callback_from(message, "vision:imagepick:0")
    assert not any(
        button.callback_data == "vision:render"
        for row in message.replies[-1]["reply_markup"].inline_keyboard
        for button in row
    )

    picker_update, _ = callback_update(
        picker_data,
        message,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(picker_update, None)
    picker_text = message.replies[-1]["text"]
    assert "Выбери желание для AI-визуализации" in picker_text
    assert active.wish_text in picker_text
    assert archived.wish_text not in picker_text
    assert not any(reply.get("kind") in {"photo", "document"} for reply in message.replies)
    assert generator.prompts == []

    item_update, _ = callback_update(
        callback_from(message, f"vision:imagepickitem:{active.id}"),
        message,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(item_update, None)
    disclosure = message.replies[-1]["text"]
    assert "Точный запрос" in disclosure
    assert active.wish_text in disclosure
    assert "gpt-image-2" in disclosure
    assert generator.prompts == []


async def test_moderation_failure_is_safe_not_retried_or_saved(db, fake_ai, caplog):
    generator = FakeImageGeneration(error="moderation_blocked")
    configured = Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="text-key",
        enable_vision_image_generation=True,
    )
    bot = FutureSelfBot(
        configured,
        db,
        fake_ai,
        ScriptedTranscription(),
        image_generation=generator,
    )
    telegram_id, chat_id = 9361, 19361
    owner = await bot._user(telegram_id)
    item = await add_item(db, owner.id, "Не выводить личное желание в лог")
    card = FakeMessage()
    await bot._vision_send_item(card, item)
    ask, _ = callback_update(
        callback_from(card, "vision:imagegenerateask:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(ask, None)
    generate, _ = callback_update(
        callback_from(card, "vision:imagegenerate:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    with caplog.at_level(logging.ERROR):
        await bot.vision_action(generate, None)
    assert len(generator.prompts) == 1
    assert "Не выводить личное желание в лог" not in caplog.text
    assert "moderation_blocked" in caplog.text
    assert await bot.vision_image_service.get(owner.id, item.id) is None
    assert any("Автоповтора не было" in reply.get("text", "") for reply in card.replies)


async def test_invalid_upload_is_not_logged_or_saved_and_can_be_retried(db, fake_ai, caplog):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    telegram_id, chat_id = 9401, 19401
    owner = await bot._user(telegram_id)
    item = await add_item(db, owner.id, "Не выводить это желание")
    card = FakeMessage()
    await bot._vision_send_item(card, item)
    begin, _ = callback_update(
        callback_from(card, "vision:imageadd:"), card, user_id=telegram_id, chat_id=chat_id
    )
    await bot.vision_action(begin, None)
    invalid = FakeMessage(document=FakeImageMedia(b"secret-payload", mime_type="image/jpeg"))
    with caplog.at_level(logging.ERROR), pytest.raises(ApplicationHandlerStop):
        await bot.vision_image_gate(update_for(invalid, user_id=telegram_id, chat_id=chat_id), None)
    assert "secret-payload" not in caplog.text
    assert "Не выводить это желание" not in caplog.text
    assert str(telegram_id) not in caplog.text
    assert await bot.vision_image_service.get(owner.id, item.id) is None
    assert await bot.vision_image_sessions.has_upload(owner.id, chat_id) is True


async def test_out_of_flow_group_foreign_and_restart_uploads_fail_closed(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(9501)
    foreign = await bot._user(9502)
    item = await add_item(db, owner.id)
    raw = image_bytes("JPEG")
    outside = FakeMessage(photo=[FakeImageMedia(raw, mime_type=None, width=320, height=180)])
    await bot.vision_image_gate(update_for(outside, user_id=9501, chat_id=19501), None)
    assert outside.replies == []
    assert await bot.vision_image_service.get(owner.id, item.id) is None

    forged = FakeMessage()
    forged_update, forged_query = callback_update(
        f"vision:imageadd:{item.id}", forged, user_id=9502, chat_id=19502
    )
    await bot.vision_action(forged_update, None)
    assert any(show_alert for _text, show_alert in forged_query.answers)
    assert await bot.vision_image_service.get(foreign.id, item.id) is None

    card = FakeMessage()
    await bot._vision_send_item(card, item)
    start, _ = callback_update(
        callback_from(card, "vision:imageadd:"), card, user_id=9501, chat_id=19501
    )
    await bot.vision_action(start, None)
    token = callback_from(card, "vision:imagecancel:").split(":")[-1]
    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    stale_update, stale_query = callback_update(
        f"vision:imageconfirm:{token}", card, user_id=9501, chat_id=19501
    )
    await restarted.vision_action(stale_update, None)
    assert any(show_alert for _text, show_alert in stale_query.answers)
    after_restart = FakeMessage(photo=[FakeImageMedia(raw, mime_type=None, width=320, height=180)])
    await restarted.vision_image_gate(update_for(after_restart, user_id=9501, chat_id=19501), None)
    assert await restarted.vision_image_service.get(owner.id, item.id) is None


def test_renderer_crops_normalized_photo_without_changing_paginated_contract():
    normalized = normalize_vision_image(
        image_bytes("PNG", size=(500, 200), color=(220, 30, 40)),
        declared_mime="image/png",
    )
    renderer = VisionBoardRenderer()
    with_photo = renderer.render(
        [
            VisionRenderItem(
                "travel",
                "Желание с фото",
                date(2030, 1, 1),
                1,
                normalized.image_bytes,
            )
        ],
        created_on=date(2026, 7, 21),
        category=None,
    )
    without_photo = renderer.render(
        [VisionRenderItem("travel", "Желание с фото", date(2030, 1, 1), 1)],
        created_on=date(2026, 7, 21),
        category=None,
    )
    assert with_photo.pages[0].png != without_photo.pages[0].png
    rendered = Image.open(BytesIO(with_photo.pages[0].png))
    assert rendered.size == (1080, 1350)
    assert rendered.getpixel((900, 260))[0] > rendered.getpixel((900, 260))[2]
