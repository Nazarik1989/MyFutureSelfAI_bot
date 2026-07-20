import logging
from datetime import date
from io import BytesIO
from types import SimpleNamespace

import pytest
from autotester.fakes import FakeCallbackQuery, FakeMessage, ScriptedTranscription
from PIL import Image, ImageColor, ImageDraw

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.models import User, VisionItem
from future_self.vision_renderer import (
    BACKGROUND,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    CATEGORY_COLORS,
    MAX_CARDS_PER_PAGE,
    MAX_PAGES,
    MAX_PNG_BYTES,
    VisionBoardRenderer,
    VisionRenderError,
    VisionRenderItem,
    VisionRenderLimiter,
    VisionRenderSessionStore,
    clean_render_text,
    fit_text_lines,
)


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def callback_update(data: str, message: FakeMessage, *, user_id: int, chat_id: int):
    query = FakeCallbackQuery(data, message)
    return (
        SimpleNamespace(
            effective_message=message,
            callback_query=query,
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=chat_id, type="private"),
        ),
        query,
    )


def command_update(message: FakeMessage, *, user_id: int, chat_id: int):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=None,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type="private"),
    )


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


async def add_item(
    db,
    owner_id: int,
    wish: str,
    *,
    category: str = "travel",
    status: str = "active",
    target_date: date | None = None,
) -> None:
    async with db.session() as session:
        session.add(
            VisionItem(
                owner_id=owner_id,
                category=category,
                wish_text=wish,
                target_date=target_date,
                status=status,
            )
        )


def test_renderer_is_deterministic_valid_and_keeps_layout_inside_safe_bounds():
    renderer = VisionBoardRenderer()
    items = [
        VisionRenderItem(
            "travel",
            "Увидеть северное сияние ✨\nи спокойно спланировать путешествие",
            date(2030, 12, 31),
            3,
        ),
        VisionRenderItem(
            "health_energy",
            "Больше энергии\u202e\x00 через бережный режим и прогулки",
            None,
            2,
        ),
        VisionRenderItem("home", "Создать уютный дом", None, 1),
    ]

    first = renderer.render(items, created_on=date(2026, 7, 20), category=None)
    second = renderer.render(list(reversed(items)), created_on=date(2026, 7, 20), category=None)

    assert first.pages[0].png == second.pages[0].png
    assert first.included_count == 3
    assert first.omitted_count == 0
    image = Image.open(BytesIO(first.pages[0].png))
    assert image.format == "PNG"
    assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == ImageColor.getrgb(BACKGROUND)
    assert image.getpixel((70, 245)) == ImageColor.getrgb(CATEGORY_COLORS["health_energy"])
    boxes = first.pages[0].card_boxes
    assert len(boxes) == 3
    assert all(0 <= box.left < box.right <= CANVAS_WIDTH for box in boxes)
    assert all(190 <= box.top < box.bottom < 1235 for box in boxes)
    assert all(
        previous.bottom < current.top for previous, current in zip(boxes, boxes[1:], strict=False)
    )
    assert len(first.pages[0].png) < MAX_PNG_BYTES


def test_renderer_caps_pages_reports_omitted_and_has_stable_sorting():
    renderer = VisionBoardRenderer()
    items = [
        VisionRenderItem(
            "money" if index % 2 else "travel",
            f"Желание {index}",
            None,
            index,
        )
        for index in range(MAX_CARDS_PER_PAGE * MAX_PAGES + 7)
    ]
    board = renderer.render(
        list(reversed(items)),
        created_on=date(2026, 7, 20),
        category=None,
    )
    repeated = renderer.render(items, created_on=date(2026, 7, 20), category=None)
    assert len(board.pages) == MAX_PAGES
    assert board.included_count == MAX_CARDS_PER_PAGE * MAX_PAGES
    assert board.omitted_count == 7
    assert [page.png for page in board.pages] == [page.png for page in repeated.pages]
    assert all(1 <= len(page.card_boxes) <= MAX_CARDS_PER_PAGE for page in board.pages)


def test_text_cleaning_and_pixel_wrapping_handle_controls_emoji_and_long_words():
    renderer = VisionBoardRenderer()
    image = Image.new("RGB", (600, 300), "white")
    draw = ImageDraw.Draw(image)
    font = renderer._font(30)
    cleaned = clean_render_text("  Мечта\n✨\x00\u202e   безопасно  ")
    assert cleaned == "Мечта ✨ безопасно"
    lines = fit_text_lines(
        draw,
        "Сверхдлинноесловобезпробелов" * 8,
        font,
        max_width=420,
        max_lines=3,
    )
    assert len(lines) == 3
    assert lines[-1].endswith("…")
    assert all(draw.textlength(line, font=font) <= 420 for line in lines)
    image.close()


def test_missing_font_fails_with_safe_renderer_error(tmp_path):
    with pytest.raises(VisionRenderError, match="font_unavailable"):
        VisionBoardRenderer(font_path=tmp_path / "missing-font.ttf")


def test_category_accents_have_high_contrast_on_card_background():
    def relative_luminance(color: str) -> float:
        channels = [value / 255 for value in ImageColor.getrgb(color)]
        linear = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    paper_luminance = relative_luminance("#FFFDFC")
    for accent in CATEGORY_COLORS.values():
        contrast = (paper_luminance + 0.05) / (relative_luminance(accent) + 0.05)
        assert contrast >= 4.5


async def test_render_capabilities_are_owner_chat_bound_single_use_and_expire():
    store = VisionRenderSessionStore(ttl_seconds=60)
    token = await store.issue(1, 101, {"travel"})
    assert await store.claim_selection(token, 2, 202, "travel") is None
    assert await store.claim_selection(token, 1, 101, "money") is None
    assert await store.claim_selection(token, 1, 101, "travel") == "travel"
    assert await store.claim_selection(token, 1, 101, "travel") is None
    assert await store.claim_download(token, 2, 202) is None
    assert await store.claim_download(token, 1, 101) == "travel"
    assert await store.claim_download(token, 1, 101) is None

    cancelled = await store.issue(1, 101, {"travel"})
    assert await store.cancel(cancelled, 2, 202) is False
    assert await store.cancel(cancelled, 1, 101) is True
    assert await store.claim_selection(cancelled, 1, 101, "travel") is None

    expired_store = VisionRenderSessionStore(ttl_seconds=-1)
    expired = await expired_store.issue(1, 101, {"travel"})
    assert await expired_store.claim_selection(expired, 1, 101, "travel") is None


async def test_render_limiter_rejects_same_owner_and_excess_global_work():
    limiter = VisionRenderLimiter(max_concurrent=2)
    assert await limiter.acquire(1) is True
    assert await limiter.acquire(1) is False
    assert await limiter.acquire(2) is True
    assert await limiter.acquire(3) is False
    await limiter.release(1)
    assert await limiter.acquire(3) is True
    await limiter.release(2)
    await limiter.release(3)


async def test_render_query_is_owner_active_category_scoped_and_bounded(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(8051)
    other = await bot._user(8052)
    for index in range(35):
        await add_item(
            db,
            owner.id,
            f"Активное желание {index}",
            category="travel" if index % 2 else "money",
        )
    await add_item(db, owner.id, "Архивное", category="travel", status="archived")
    await add_item(db, other.id, "Чужое", category="travel")

    all_items, all_total = await bot.vision_service.active_for_render(
        owner.id,
        category=None,
        limit=30,
    )
    travel_items, travel_total = await bot.vision_service.active_for_render(
        owner.id,
        category="travel",
        limit=30,
    )
    forged_items, forged_total = await bot.vision_service.active_for_render(
        owner.id,
        category="not-a-category",
        limit=30,
    )
    assert len(all_items) == 30
    assert all_total == 35
    assert travel_total == 17
    assert all(item.category == "travel" for item in travel_items)
    assert forged_items == []
    assert forged_total == 0


async def test_handler_renders_only_active_owner_items_and_downloads_documents(
    db,
    fake_ai,
    tmp_path,
    monkeypatch,
):
    import future_self.vision_handlers as vision_handlers_module

    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(8101)
    async with db.session() as session:
        stored_owner = await session.get(User, owner.id)
        assert stored_owner is not None
        stored_owner.timezone = "Europe/Saratov"
    other = await bot._user(8102)
    await add_item(db, owner.id, "Поехать в Карелию", category="travel")
    await add_item(db, owner.id, "Создать уют", category="home")
    await add_item(db, owner.id, "Уже достигнуто", status="achieved")
    await add_item(db, other.id, "Чужое приватное желание", category="travel")
    before = set(tmp_path.iterdir())
    captured_dates: list[date] = []
    real_renderer = bot.vision_renderer

    class FixedDateTime:
        @classmethod
        def now(cls, timezone):
            assert timezone.key in {"Europe/Saratov", "UTC"}
            selected = date(2026, 7, 21) if timezone.key == "Europe/Saratov" else date(2026, 7, 20)
            return SimpleNamespace(date=lambda: selected)

    class CapturingRenderer:
        def render(self, items, **kwargs):
            captured_dates.append(kwargs["created_on"])
            return real_renderer.render(items, **kwargs)

    monkeypatch.setattr(vision_handlers_module, "datetime", FixedDateTime)
    bot.vision_renderer = CapturingRenderer()

    message = FakeMessage("/vision")
    await bot.vision_command(command_update(message, user_id=8101, chat_id=18101), None)
    render_update, _ = callback_update(
        callback_from(message, "vision:render"),
        message,
        user_id=8101,
        chat_id=18101,
    )
    await bot.vision_action(render_update, None)
    selection_markup = message.replies[-1]["reply_markup"]
    assert all(
        len(button.callback_data.encode("utf-8")) <= 64
        for row in selection_markup.inline_keyboard
        for button in row
    )
    pick = callback_from(message, "vision:renderpick:")
    assert pick.endswith(":all")
    pick_update, _ = callback_update(pick, message, user_id=8101, chat_id=18101)
    await bot.vision_action(pick_update, None)

    photos = [reply for reply in message.replies if reply.get("kind") == "photo"]
    assert len(photos) == 1
    assert captured_dates == [date(2026, 7, 21)]
    assert "Активных желаний: 2" in photos[0]["text"]
    image = Image.open(BytesIO(photos[0]["data"]))
    assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
    assert image.format == "PNG"

    download = callback_from(message, "vision:renderdownload:")
    download_update, _ = callback_update(download, message, user_id=8101, chat_id=18101)
    await bot.vision_action(download_update, None)
    documents = [reply for reply in message.replies if reply.get("kind") == "document"]
    assert len(documents) == 1
    assert captured_dates == [date(2026, 7, 21), date(2026, 7, 21)]
    assert documents[0]["filename"].endswith(".png")
    assert Image.open(BytesIO(documents[0]["data"])).size == (CANVAS_WIDTH, CANVAS_HEIGHT)

    repeat_update, repeat_query = callback_update(
        download,
        message,
        user_id=8101,
        chat_id=18101,
    )
    await bot.vision_action(repeat_update, None)
    assert any(text and "устарел" in text for text, _alert in repeat_query.answers)
    assert fake_ai.route_calls == []
    assert set(tmp_path.iterdir()) == before


async def test_empty_category_forged_owner_and_restart_callbacks_fail_closed(db, fake_ai):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(8201)
    await bot._user(8202)

    empty = FakeMessage("/vision")
    await bot.vision_command(command_update(empty, user_id=8202, chat_id=18202), None)
    empty_update, _ = callback_update(
        callback_from(empty, "vision:render"),
        empty,
        user_id=8202,
        chat_id=18202,
    )
    await bot.vision_action(empty_update, None)
    assert "Сначала добавь желание" in empty.replies[-1]["text"]

    await add_item(db, owner.id, "Владелец видит свою карточку")
    message = FakeMessage("/vision")
    await bot.vision_command(command_update(message, user_id=8201, chat_id=18201), None)
    render_update, _ = callback_update(
        callback_from(message, "vision:render"),
        message,
        user_id=8201,
        chat_id=18201,
    )
    await bot.vision_action(render_update, None)
    pick = callback_from(message, "vision:renderpick:")

    forged_update, forged_query = callback_update(
        pick,
        message,
        user_id=8202,
        chat_id=18202,
    )
    await bot.vision_action(forged_update, None)
    assert any(text and "устарел" in text for text, _alert in forged_query.answers)
    assert not any(reply.get("kind") == "photo" for reply in message.replies)

    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    stale_update, stale_query = callback_update(
        pick,
        message,
        user_id=8201,
        chat_id=18201,
    )
    await restarted.vision_action(stale_update, None)
    assert any(text and "устарел" in text for text, _alert in stale_query.answers)


async def test_renderer_failure_and_stream_failure_are_safe_and_leave_no_temp_files(
    db,
    fake_ai,
    tmp_path,
    caplog,
):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    owner = await bot._user(8301)
    await add_item(db, owner.id, "Секретное желание не должно попасть в лог")
    before = set(tmp_path.iterdir())

    def fail_render(*args, **kwargs):
        raise RuntimeError("C:/private/internal/path and private wish")

    bot.vision_renderer.render = fail_render
    message = FakeMessage()
    with caplog.at_level(logging.ERROR):
        await bot._vision_render_and_send(
            message,
            owner,
            None,
            token="safe-token",
            as_document=False,
        )
    assert "Не удалось создать визуализацию" in message.replies[-1]["text"]
    assert "private/internal" not in caplog.text
    assert "private wish" not in caplog.text
    assert "Секретное желание" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert set(tmp_path.iterdir()) == before

    healthy = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())

    class FailingPhotoMessage(FakeMessage):
        def __init__(self):
            super().__init__()
            self.captured_stream = None

        async def reply_photo(self, photo, **kwargs):
            self.captured_stream = photo
            raise RuntimeError("transport failed")

    failing_message = FailingPhotoMessage()
    await healthy._vision_render_and_send(
        failing_message,
        owner,
        None,
        token="safe-token",
        as_document=False,
    )
    assert failing_message.captured_stream.closed is True
    assert "Не удалось создать визуализацию" in failing_message.replies[-1]["text"]
