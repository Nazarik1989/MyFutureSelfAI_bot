import asyncio
import hashlib
import os
import stat
from datetime import date
from io import BytesIO
from types import SimpleNamespace

import pytest
from autotester.fakes import (
    FakeCallbackQuery,
    FakeImageMedia,
    FakeMessage,
    FakeVoice,
    ScriptedTranscription,
)
from PIL import Image, PngImagePlugin
from pypdf import PdfWriter
from sqlalchemy import func, select
from telegram.ext import ApplicationHandlerStop

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.lab_media import (
    MAX_LAB_INPUT_BYTES,
    MAX_PDF_PAGES,
    LabMediaError,
    NormalizedLabPage,
    TelegramLabMetadata,
    process_lab_upload,
    validate_telegram_lab_metadata,
)
from future_self.labs import LabDocumentService, LabUploadSessionStore
from future_self.models import LabDeleteConfirmation, LabDocument, LabDocumentPage


def settings() -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token="123456:TEST",
        ai_api_key="test-key",
        ai_model="test-model",
    )


def image_bytes(image_format: str, *, metadata: bool = False, size=(320, 180)) -> bytes:
    image = Image.new("RGB", size, "navy")
    output = BytesIO()
    kwargs = {}
    if image_format == "JPEG" and metadata:
        exif = Image.Exif()
        exif[270] = "sensitive-comment"
        kwargs = {"exif": exif, "icc_profile": b"private-profile"}
    elif image_format == "PNG" and metadata:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", "sensitive-comment")
        kwargs = {"pnginfo": info}
    image.save(output, format=image_format, **kwargs)
    image.close()
    return output.getvalue()


def pdf_bytes(page_count: int = 1, *, encrypted=False, javascript=False, attachment=False) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    if javascript:
        writer.add_js("app.alert('no')")
    if attachment:
        writer.add_attachment("private.txt", b"private")
    if encrypted:
        writer.encrypt("password")
    output = BytesIO()
    writer.write(output)
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
    raise AssertionError(f"Missing callback {prefix}")


def update_for(message, *, user_id=101, chat_id=201, query=None, chat_type="private"):
    return SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=query,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
    )


def context() -> SimpleNamespace:
    return SimpleNamespace(user_data={}, args=[], bot=SimpleNamespace())


def callback_update(data: str, message: FakeMessage, *, user_id=101, chat_id=201):
    query = FakeCallbackQuery(data, message)
    return update_for(message, user_id=user_id, chat_id=chat_id, query=query), query


@pytest.mark.parametrize(
    ("image_format", "mime_type"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_supported_images_are_reencoded_without_metadata(tmp_path, image_format, mime_type):
    raw = image_bytes(image_format, metadata=image_format != "WEBP")
    processed = process_lab_upload(
        raw,
        TelegramLabMetadata("document", len(raw), mime_type),
        temp_root=tmp_path,
    )
    assert processed.source_type == "image"
    assert len(processed.pages) == 1
    page = processed.pages[0]
    assert page.mime_type == "image/jpeg"
    assert page.image_bytes != raw
    decoded = Image.open(BytesIO(page.image_bytes))
    assert decoded.mode == "RGB"
    assert not decoded.getexif()
    assert "icc_profile" not in decoded.info
    assert "comment" not in decoded.info
    decoded.verify()


@pytest.mark.parametrize("page_count", [1, 3])
def test_pdf_is_locally_rasterized_to_bounded_jpeg_pages(tmp_path, page_count):
    raw = pdf_bytes(page_count)
    processed = process_lab_upload(
        raw,
        TelegramLabMetadata("document", len(raw), "application/pdf"),
        temp_root=tmp_path,
    )
    assert processed.source_type == "pdf"
    assert len(processed.pages) == page_count
    assert all(page.image_bytes.startswith(b"\xff\xd8\xff") for page in processed.pages)
    assert all(
        page.sha256 == hashlib.sha256(page.image_bytes).hexdigest() for page in processed.pages
    )
    assert list(tmp_path.rglob("input.pdf")) == []
    assert list(tmp_path.rglob("page-*.jpg")) == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(pdf_bytes(encrypted=True), id="encrypted"),
        pytest.param(pdf_bytes(javascript=True), id="javascript"),
        pytest.param(pdf_bytes(attachment=True), id="attachment"),
        pytest.param(pdf_bytes()[:-8], id="truncated"),
        pytest.param(pdf_bytes(MAX_PDF_PAGES + 1), id="too-many-pages"),
    ],
)
def test_unsafe_or_invalid_pdfs_fail_closed_and_leave_no_original(tmp_path, payload):
    with pytest.raises(LabMediaError):
        process_lab_upload(
            payload,
            TelegramLabMetadata("document", len(payload), "application/pdf"),
            temp_root=tmp_path,
        )
    assert list(tmp_path.rglob("input.pdf")) == []


def test_mime_format_mismatch_svg_animation_size_and_bomb_are_rejected(tmp_path, monkeypatch):
    png = image_bytes("PNG")
    with pytest.raises(LabMediaError, match="mime_mismatch"):
        process_lab_upload(
            png,
            TelegramLabMetadata("document", len(png), "image/jpeg"),
            temp_root=tmp_path,
        )
    with pytest.raises(LabMediaError, match="mime_mismatch"):
        process_lab_upload(
            pdf_bytes(),
            TelegramLabMetadata("document", len(pdf_bytes()), "image/jpeg"),
            temp_root=tmp_path,
        )
    with pytest.raises(LabMediaError, match="unsupported_mime"):
        validate_telegram_lab_metadata(TelegramLabMetadata("document", 100, "image/svg+xml"))
    with pytest.raises(LabMediaError, match="input_too_large"):
        validate_telegram_lab_metadata(
            TelegramLabMetadata("document", MAX_LAB_INPUT_BYTES + 1, "application/pdf")
        )

    first = Image.new("RGB", (20, 20), "red")
    second = Image.new("RGB", (20, 20), "blue")
    output = BytesIO()
    first.save(output, format="WEBP", save_all=True, append_images=[second])
    first.close()
    second.close()
    with pytest.raises(LabMediaError, match="animated"):
        process_lab_upload(
            output.getvalue(),
            TelegramLabMetadata("document", len(output.getvalue()), "image/webp"),
            temp_root=tmp_path,
        )

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    tiny = image_bytes("PNG", size=(40, 40))
    with pytest.raises(LabMediaError, match="decompression_bomb"):
        process_lab_upload(
            tiny,
            TelegramLabMetadata("document", len(tiny), "image/png"),
            temp_root=tmp_path,
        )


def test_renderer_timeout_and_failure_fail_closed(tmp_path, monkeypatch):
    raw = pdf_bytes()

    def timeout(*args, **kwargs):
        del args, kwargs
        raise __import__("subprocess").TimeoutExpired("worker", 1)

    monkeypatch.setattr("future_self.safe_media.subprocess.subprocess.run", timeout)
    with pytest.raises(LabMediaError, match="renderer_timeout"):
        process_lab_upload(
            raw,
            TelegramLabMetadata("document", len(raw), "application/pdf"),
            temp_root=tmp_path,
        )
    assert list(tmp_path.rglob("input.pdf")) == []


async def test_secure_temp_sessions_are_bounded_bound_and_cleaned(tmp_path):
    root = tmp_path / "private-labs"
    store = LabUploadSessionStore(root=root, ttl_seconds=60, max_sessions=1)
    if os.name == "posix":
        assert stat.S_IMODE(root.stat().st_mode) & 0o077 == 0
    token = await store.start(1, 10)
    assert token is not None
    assert await store.start(2, 20) is None
    assert await store.claim_upload(2, 20) is None
    claim = await store.claim_upload(1, 10)
    assert claim and claim.token == token
    raw = image_bytes("JPEG")
    processed = process_lab_upload(
        raw,
        TelegramLabMetadata("document", len(raw), "image/jpeg"),
        temp_root=root,
    )
    snapshot = await store.attach(token, 1, 10, processed, title="Документ")
    assert snapshot and snapshot.first_page
    page_path = next((root / token).glob("page-*.jpg"))
    if os.name == "posix":
        assert stat.S_IMODE(page_path.stat().st_mode) & 0o077 == 0
    assert await store.claim_confirm(token, 2, 20) is None
    assert (await store.claim_confirm(token, 1, 10)) is not None
    assert await store.claim_confirm(token, 1, 10) is None
    await store.finish(token, 1, 10)
    assert not (root / token).exists()

    orphan = await store.start(1, 10)
    assert orphan and (root / orphan).exists()
    LabUploadSessionStore(root=root, ttl_seconds=60)
    assert not (root / orphan).exists()

    expired = LabUploadSessionStore(root=tmp_path / "expired", ttl_seconds=-1)
    stale = await expired.start(1, 10)
    assert stale is not None
    assert await expired.has_active(1, 10) is False
    assert not (expired.root / stale).exists()


async def test_service_owner_isolation_duplicates_pagination_edits_and_delete(db):
    bot = FutureSelfBot(settings(), db, SimpleNamespace(), ScriptedTranscription())
    owner_a = await bot._user(1001)
    owner_b = await bot._user(1002)
    service = LabDocumentService(db)
    payload = b"\xff\xd8\xffnormalized"
    page = NormalizedLabPage(payload, "image/jpeg", 10, 10, hashlib.sha256(payload).hexdigest())
    first = await service.create(owner_a.id, "Одинаковый", date(2026, 7, 1), "image", (page,))
    second = await service.create(owner_b.id, "Одинаковый", date(2026, 7, 1), "image", (page,))
    assert await service.get(owner_b.id, first.id) is None
    assert await service.get_page(owner_b.id, first.id, 0) is None
    assert (await service.get(owner_a.id, first.id)).pages[0].image_bytes == payload
    assert (await service.get(owner_b.id, second.id)).pages[0].image_bytes == payload

    for index in range(7):
        await service.create(owner_a.id, f"Документ {index}", None, "image", (page,))
    listed, total = await service.page(owner_a.id, 0)
    next_page, _ = await service.page(owner_a.id, 1)
    assert len(listed) == 6 and len(next_page) == 2 and total == 8

    current = await service.get(owner_a.id, first.id)
    assert await service.rename(owner_b.id, first.id, current.version, "Чужое") is False
    assert await service.rename(owner_a.id, first.id, current.version, "Новое название") is True
    assert await service.rename(owner_a.id, first.id, current.version, "Повтор") is False
    current = await service.get(owner_a.id, first.id)
    assert await service.set_date(owner_a.id, first.id, current.version, None) is True

    token = await service.issue_delete(owner_a.id, 5001, first.id)
    assert token is not None
    restarted_service = LabDocumentService(db)
    assert await restarted_service.confirm_delete(token, owner_b.id, 5001) is False
    results = await asyncio.gather(
        restarted_service.confirm_delete(token, owner_a.id, 5001),
        restarted_service.confirm_delete(token, owner_a.id, 5001),
    )
    assert sorted(results) == [False, True]
    assert await service.get(owner_a.id, first.id) is None
    async with db.sessions() as session:
        assert (
            await session.scalar(
                select(func.count(LabDocumentPage.id)).where(
                    LabDocumentPage.document_id == first.id
                )
            )
            == 0
        )


async def test_stale_delete_confirmation_cannot_remove_new_version(db):
    bot = FutureSelfBot(settings(), db, SimpleNamespace(), ScriptedTranscription())
    owner = await bot._user(2001)
    payload = b"\xff\xd8\xffsafe"
    page = NormalizedLabPage(payload, "image/jpeg", 10, 10, hashlib.sha256(payload).hexdigest())
    document = await bot.lab_documents.create(owner.id, "Старое", None, "image", (page,))
    token = await bot.lab_documents.issue_delete(owner.id, 3001, document.id)
    assert token
    assert await bot.lab_documents.rename(owner.id, document.id, document.version, "Новое")
    assert await bot.lab_documents.confirm_delete(token, owner.id, 3001) is False
    assert await bot.lab_documents.get(owner.id, document.id) is not None


async def test_full_photo_preview_edit_confirm_view_and_delete_flow(db, fake_ai, tmp_path):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    bot.lab_uploads = LabUploadSessionStore(root=tmp_path / "labs")
    ctx = context()
    menu = FakeMessage("/labs")
    await bot.labs_command(update_for(menu), ctx)
    start_update, _ = callback_update(callback_from(menu, "labs:add"), menu)
    await bot.labs_action(start_update, ctx)

    raw = image_bytes("PNG", metadata=True)
    upload = FakeMessage(document=FakeImageMedia(raw, mime_type="image/png"))
    with pytest.raises(ApplicationHandlerStop):
        await bot.labs_media_gate(update_for(upload), ctx)
    assert upload.replies[-1]["kind"] == "photo"
    assert "Сохранение произойдёт только" in upload.replies[-1]["text"]

    title_update, _ = callback_update(callback_from(upload, "labs:draft:title:"), upload)
    await bot.labs_action(title_update, ctx)
    title = FakeMessage("Общий анализ крови")
    with pytest.raises(ApplicationHandlerStop):
        await bot.labs_text_gate(update_for(title), ctx)
    date_update, _ = callback_update(callback_from(title, "labs:draft:date:"), title)
    await bot.labs_action(date_update, ctx)
    document_date = FakeMessage("20.07.2026")
    with pytest.raises(ApplicationHandlerStop):
        await bot.labs_text_gate(update_for(document_date), ctx)

    save_data = callback_from(document_date, "labs:draft:save:")
    save_update, _ = callback_update(save_data, document_date)
    await bot.labs_action(save_update, ctx)
    owner = await bot._user(101)
    items, total = await bot.lab_documents.page(owner.id, 0)
    assert total == 1
    document = items[0]
    assert document.title == "Общий анализ крови"
    assert document.document_date == date(2026, 7, 20)
    assert not any(bot.lab_uploads.root.iterdir())
    assert fake_ai.route_calls == []

    details_message = FakeMessage()
    open_update, _ = callback_update(f"labs:open:{document.id}", details_message)
    await bot.labs_action(open_update, ctx)
    assert "Общий анализ крови" in details_message.replies[-1]["text"]
    view_update, _ = callback_update(f"labs:view:{document.id}:0", details_message)
    await bot.labs_action(view_update, ctx)
    assert details_message.replies[-1]["kind"] == "photo"

    delete_update, _ = callback_update(f"labs:delete:{document.id}", details_message)
    await bot.labs_action(delete_update, ctx)
    confirm = callback_from(details_message, "labs:deleteconfirm:")
    forged, forged_query = callback_update(confirm, details_message, user_id=999, chat_id=999)
    await bot.labs_action(forged, ctx)
    assert any(show for _text, show in forged_query.answers)
    confirm_update, _ = callback_update(confirm, details_message)
    await bot.labs_action(confirm_update, ctx)
    assert await bot.lab_documents.get(owner.id, document.id) is None
    repeated, repeated_query = callback_update(confirm, details_message)
    await bot.labs_action(repeated, ctx)
    assert any(show for _text, show in repeated_query.answers)


async def test_cancel_restart_direct_upload_and_group_privacy(db, fake_ai, tmp_path):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    bot.lab_uploads = LabUploadSessionStore(root=tmp_path / "labs")
    direct = FakeMessage(document=FakeImageMedia(image_bytes("JPEG"), mime_type="image/jpeg"))
    await bot.labs_media_gate(update_for(direct), context())
    assert direct.replies == []

    owner = await bot._user(101)
    token = await bot.lab_uploads.start(owner.id, 201)
    assert token
    restarted = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    restarted.lab_uploads = LabUploadSessionStore(root=tmp_path / "restart")
    stale_message = FakeMessage()
    stale_update, stale_query = callback_update(f"labs:draft:save:{token}", stale_message)
    await restarted.labs_action(stale_update, context())
    assert any(show for _text, show in stale_query.answers)

    cancelled = await bot.lab_uploads.cancel(token, owner.id, 201)
    assert cancelled and not (bot.lab_uploads.root / token).exists()

    group = FakeMessage("/labs")
    with pytest.raises(ApplicationHandlerStop):
        await bot.private_chat_guard(update_for(group, chat_type="group"), context())
    assert "только в личном чате" in group.replies[-1]["text"]


async def test_database_contains_no_document_before_explicit_confirm(db, fake_ai, tmp_path):
    bot = FutureSelfBot(settings(), db, fake_ai, ScriptedTranscription())
    bot.lab_uploads = LabUploadSessionStore(root=tmp_path / "labs")
    owner = await bot._user(101)
    token = await bot.lab_uploads.start(owner.id, 201)
    claim = await bot.lab_uploads.claim_upload(owner.id, 201)
    assert claim and claim.token == token
    raw = image_bytes("JPEG")
    processed = process_lab_upload(
        raw,
        TelegramLabMetadata("document", len(raw), "image/jpeg"),
        temp_root=bot.lab_uploads.root,
    )
    await bot.lab_uploads.attach(token, owner.id, 201, processed, title="Preview")
    async with db.sessions() as session:
        assert await session.scalar(select(func.count(LabDocument.id))) == 0
        assert await session.scalar(select(func.count(LabDocumentPage.id))) == 0
        assert await session.scalar(select(func.count(LabDeleteConfirmation.token))) == 0


async def test_document_edit_blocks_voice_and_media_from_other_feature_routes(
    db, fake_ai, tmp_path
):
    transcription = ScriptedTranscription()
    bot = FutureSelfBot(settings(), db, fake_ai, transcription)
    bot.lab_uploads = LabUploadSessionStore(root=tmp_path / "labs")
    owner = await bot._user(101)
    ctx = context()
    ctx.user_data["lab_document_edit"] = {
        "owner_id": owner.id,
        "chat_id": 201,
        "document_id": 1,
        "version": 1,
        "field": "title",
        "expires_at": float("inf"),
    }
    voice = FakeMessage(voice=FakeVoice())
    with pytest.raises(ApplicationHandlerStop):
        await bot.labs_voice_gate(update_for(voice), ctx)
    media = FakeMessage(document=FakeImageMedia(image_bytes("JPEG"), mime_type="image/jpeg"))
    with pytest.raises(ApplicationHandlerStop):
        await bot.labs_media_gate(update_for(media), ctx)
    assert transcription.calls == []
    assert fake_ai.route_calls == []
