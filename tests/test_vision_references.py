import base64
import json
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
from PIL import Image
from telegram.ext import ApplicationHandlerStop

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.image_generation import ImageReferenceInput, OpenRouterImageGenerationService
from future_self.models import VisionItem
from future_self.vision_images import normalize_vision_image
from future_self.vision_references import VisionReferenceService, VisionReferenceSessionStore


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def image_bytes(color=(40, 120, 200)) -> bytes:
    image = Image.new("RGB", (320, 240), color)
    output = BytesIO()
    image.save(output, format="JPEG")
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


def update_for(message: FakeMessage, *, user_id: int, chat_id: int):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=None,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
    )


def callback_update(data: str, message: FakeMessage, *, user_id: int, chat_id: int):
    query = FakeCallbackQuery(data, message)
    return SimpleNamespace(
        effective_message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
    )


def media_callback_update(data: str, message: FakeMessage, *, user_id: int, chat_id: int):
    query = FakeMediaCallbackQuery(data, message)
    update = SimpleNamespace(
        effective_message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
    )
    return update, query


class FakeGeneration:
    enabled = True
    model = "openai/gpt-image-2"
    quality = "medium"
    size = "1024x1024"

    def __init__(self):
        self.calls: list[tuple[str, tuple[ImageReferenceInput, ...]]] = []

    async def generate(self, prompt: str, *, references=()) -> bytes:
        self.calls.append((prompt, tuple(references)))
        output = BytesIO()
        image = Image.new("RGB", (512, 512), (90, 140, 180))
        image.save(output, format="PNG")
        image.close()
        return output.getvalue()


async def add_item(db, owner_id: int) -> VisionItem:
    async with db.session() as session:
        item = VisionItem(
            owner_id=owner_id,
            category="travel",
            wish_text="Увидеть океан и почувствовать спокойствие",
            status="active",
        )
        session.add(item)
        await session.flush()
        return item


async def test_reference_service_persists_deduplicates_and_is_owner_scoped(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(8101)
    foreign = await bot._user(8102)
    normalized = normalize_vision_image(image_bytes(), declared_mime="image/jpeg")

    created = await bot.vision_reference_service.save(
        owner.id, kind="self", name="Я сейчас", normalized=normalized
    )
    duplicate = await bot.vision_reference_service.save(
        owner.id, kind="style", name="Дубль", normalized=normalized
    )
    assert created.status == "created"
    assert duplicate.status == "existing"
    assert created.reference is not None
    assert await bot.vision_reference_service.get(foreign.id, created.reference.id) is None

    restarted = VisionReferenceService(db)
    rows = await restarted.list(owner.id)
    assert [(row.kind, row.name, row.image_bytes) for row in rows] == [
        ("self", "Я сейчас", normalized.image_bytes)
    ]
    renamed = await restarted.rename(
        owner.id, rows[0].id, expected_version=rows[0].version, name="Я в будущем"
    )
    assert renamed.status == "renamed"
    assert renamed.reference is not None
    replacement = normalize_vision_image(image_bytes((100, 40, 180)), declared_mime="image/jpeg")
    replaced = await restarted.replace(
        owner.id,
        rows[0].id,
        expected_version=renamed.reference.version,
        normalized=replacement,
    )
    assert replaced.status == "replaced"
    assert replaced.reference is not None
    deleted = await restarted.delete(
        owner.id, rows[0].id, expected_version=replaced.reference.version
    )
    assert deleted.status == "deleted"
    assert await restarted.list(owner.id) == []


async def test_reference_session_is_owner_chat_bound_and_expires_closed():
    store = VisionReferenceSessionStore(ttl_seconds=60)
    token = await store.issue_create(1, 101)
    assert token is not None
    assert await store.choose_kind(token, 2, 202, "self") is None
    chosen = await store.choose_kind(token, 1, 101, "self")
    assert chosen and chosen.kind == "self"
    named = await store.set_name(token, 1, 101, "  Я   сейчас  ")
    assert named and named.name == "Я сейчас"
    claimed = await store.claim_upload(1, 101)
    assert claimed and claimed.name == "Я сейчас"
    normalized = normalize_vision_image(image_bytes(), declared_mime="image/jpeg")
    assert await store.attach_preview(token, 2, 202, normalized) is False
    assert await store.attach_preview(token, 1, 101, normalized) is True
    assert await store.claim_confirm(token, 2, 202) is None
    confirmed = await store.claim_confirm(token, 1, 101)
    assert confirmed and confirmed.image == normalized

    expired = VisionReferenceSessionStore(ttl_seconds=-1)
    expired_token = await expired.issue_create(1, 101)
    assert expired_token is not None
    assert await expired.has_active(1, 101) is False


async def test_openrouter_adapter_sends_private_references_as_data_urls():
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(b"png").decode("ascii")}]},
        )

    client = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1/", transport=httpx.MockTransport(respond)
    )
    service = OpenRouterImageGenerationService(
        client, model="openai/gpt-image-2", quality="medium", size="1024x1024"
    )
    try:
        result = await service.generate(
            "approved prompt",
            references=[ImageReferenceInput(b"private-jpeg", "image/jpeg")],
        )
    finally:
        await client.aclose()
    assert result == b"png"
    payload = json.loads(requests[0].content)
    assert payload["input_references"] == [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(b"private-jpeg").decode("ascii")
            },
        }
    ]


async def test_telegram_library_selects_saved_reference_only_after_explicit_choices(db, fake_ai):
    generator = FakeGeneration()
    bot = FutureSelfBot(
        settings(), db, fake_ai, ScriptedTranscription(), image_generation=generator
    )
    telegram_id, chat_id = 8201, 18201
    await bot._user(telegram_id)
    await bot.access_service.grant_admin(telegram_id, source="vision-test")
    owner = await bot._user(telegram_id)
    item = await add_item(db, owner.id)

    library = FakeMessage()
    await bot.vision_action(
        callback_update("vision:refadd", library, user_id=telegram_id, chat_id=chat_id),
        None,
    )
    kind_callback = callback_from(library, "vision:refkind:")
    kind_token = kind_callback.split(":")[2]
    kind_update, kind_query = media_callback_update(
        f"vision:refkind:{kind_token}:self",
        library,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(kind_update, None)
    assert kind_query.text_attempts == 1
    assert any("Тип: Я / моя внешность" in text for text in kind_query.caption_edits)
    name = FakeMessage("Я в будущем")
    with pytest.raises(ApplicationHandlerStop):
        await bot.vision_text_gate(update_for(name, user_id=telegram_id, chat_id=chat_id), None)
    upload = FakeMessage(
        photo=[FakeImageMedia(image_bytes(), mime_type=None, width=320, height=240)]
    )
    with pytest.raises(ApplicationHandlerStop):
        await bot.vision_image_gate(update_for(upload, user_id=telegram_id, chat_id=chat_id), None)
    confirm_update, confirm_query = media_callback_update(
        callback_from(upload, "vision:refconfirm:"),
        upload,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(confirm_update, None)
    references = await bot.vision_reference_service.list(owner.id)
    assert len(references) == 1
    assert references[0].name == "Я в будущем"
    assert confirm_query.text_attempts == 1
    assert any("Референс сохранён" in text for text in confirm_query.caption_edits)
    assert any("Мои референсы" in text for text in confirm_query.caption_edits)

    card = FakeMessage()
    await bot._vision_send_item(card, item)
    ask_update, _ask_query = media_callback_update(
        callback_from(card, "vision:imagegenerateask:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(ask_update, None)
    assert generator.calls == []
    refs_update, _refs_query = media_callback_update(
        callback_from(card, "vision:genrefs:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(refs_update, None)
    toggle_update, _toggle_query = media_callback_update(
        callback_from(card, "vision:genreftoggle:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(toggle_update, None)
    done_update, _done_query = media_callback_update(
        callback_from(card, "vision:genrefdone:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(done_update, None)
    assert generator.calls == []
    generate_update, generate_query = media_callback_update(
        callback_from(card, "vision:imagegenerate:"),
        card,
        user_id=telegram_id,
        chat_id=chat_id,
    )
    await bot.vision_action(generate_update, None)
    assert generate_query.text_attempts == 1
    assert any("Создаю изображение" in text for text in generate_query.caption_edits)
    assert len(generator.calls) == 1
    prompt, sent_references = generator.calls[0]
    assert "Изображение 1" in prompt
    assert "Я в будущем" in prompt
    assert len(sent_references) == 1
    assert sent_references[0].image_bytes == references[0].image_bytes


async def test_foreign_reference_id_cannot_be_selected_for_generation(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(8301)
    foreign = await bot._user(8302)
    normalized = normalize_vision_image(image_bytes(), declared_mime="image/jpeg")
    foreign_reference = await bot.vision_reference_service.save(
        foreign.id, kind="self", name="Чужое фото", normalized=normalized
    )
    assert foreign_reference.reference is not None
    assert (
        await bot.vision_reference_service.get_many(owner.id, (foreign_reference.reference.id,))
        == []
    )
