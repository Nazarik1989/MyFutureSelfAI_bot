from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from future_self.knowledge_extraction import (
    ExtractionLimits,
    KnowledgeExtractionError,
    KnowledgeExtractor,
    extract_url_note,
    inspect_upload,
)
from future_self.safe_media.knowledge_worker import _block_network


def private_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    path.chmod(0o600)
    return path


def docx_bytes(text: str, *, active: bool = False, external_relationship: bool = False) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>"
            if not active
            else "<Types>macroEnabled</Types>",
        )
        relationship = (
            "<Relationship Target='https://private.invalid/path' TargetMode='External'/>"
            if external_relationship
            else ""
        )
        archive.writestr("_rels/.rels", f"<Relationships>{relationship}</Relationships>")
        archive.writestr(
            "word/document.xml",
            "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return output.getvalue()


def docx_with_document_xml(document_xml: bytes) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>",
        )
        archive.writestr("_rels/.rels", "<Relationships/>")
        archive.writestr("word/document.xml", document_xml)
    return output.getvalue()


def epub_bytes(text: str, *, doctype: bool = False) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        mime = ZipInfo("mimetype")
        mime.compress_type = ZIP_STORED
        archive.writestr(mime, b"application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            "<container><rootfiles><rootfile full-path='OPS/package.opf'/></rootfiles></container>",
            compress_type=ZIP_DEFLATED,
        )
        archive.writestr(
            "OPS/package.opf",
            "<package><manifest><item id='c1' href='chapter.xhtml' "
            "media-type='application/xhtml+xml'/></manifest>"
            "<spine><itemref idref='c1'/></spine></package>",
            compress_type=ZIP_DEFLATED,
        )
        prefix = "<!DOCTYPE html>" if doctype else ""
        archive.writestr(
            "OPS/chapter.xhtml",
            prefix + f"<html><body><p>{text}</p><script>secret()</script></body></html>",
            compress_type=ZIP_DEFLATED,
        )
    return output.getvalue()


def pdf_bytes(*, text: str | None = None, javascript: bool = False) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 200 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    if javascript:
        writer.add_js("app.alert('blocked')")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def extractor(tmp_path: Path, **limits: int) -> KnowledgeExtractor:
    return KnowledgeExtractor(
        tmp_path / "worker-temp",
        limits=ExtractionLimits(**limits),
    )


def test_upload_detection_uses_magic_mime_and_only_filename_suffix(tmp_path: Path) -> None:
    pdf = private_file(tmp_path / "opaque", pdf_bytes())
    detected = inspect_upload(
        pdf,
        declared_mime="application/pdf",
        display_name="../../private/report.pdf",
    )
    assert (detected.source_format, detected.detected_mime) == ("pdf", "application/pdf")

    with pytest.raises(KnowledgeExtractionError, match="mime_mismatch"):
        inspect_upload(pdf, declared_mime="image/png", display_name="report.png")
    with pytest.raises(KnowledgeExtractionError, match="unsupported_format"):
        inspect_upload(
            private_file(tmp_path / "binary", b"not-text\x00payload"),
            declared_mime="application/octet-stream",
            display_name="payload.bin",
        )


def test_txt_and_markdown_are_utf8_normalized_deterministically(tmp_path: Path) -> None:
    source = private_file(tmp_path / "note", "Первая  \r\n\r\nСтрока".encode())
    detected = inspect_upload(source, declared_mime="text/markdown", display_name="note.md")
    result = extractor(tmp_path).extract_path(source, detected.source_format)

    assert result.status == "ready"
    assert result.text_bytes.decode() == "Первая\n\nСтрока"
    assert result.text_sha256 is not None
    assert "Первая" not in repr(result)
    assert extractor(tmp_path).extract_path(source, "text").status == "ready"

    with pytest.raises(KnowledgeExtractionError, match="source_hash_mismatch"):
        extractor(tmp_path).extract_path(
            source,
            "markdown",
            expected_size=source.stat().st_size,
            expected_sha256="0" * 64,
        )


def test_text_pdf_and_image_only_pdf_have_honest_statuses(tmp_path: Path) -> None:
    text_pdf = private_file(tmp_path / "text.pdf", pdf_bytes(text="Deterministic text"))
    image_pdf = private_file(tmp_path / "scan.pdf", pdf_bytes())

    ready = extractor(tmp_path).extract_path(text_pdf, "pdf")
    partial = extractor(tmp_path).extract_path(image_pdf, "pdf")

    assert ready.status == "ready"
    assert b"Deterministic text" in ready.text_bytes
    assert ready.metadata["page_count"] == 1
    assert partial.status == "partial"
    assert partial.error_code == "no_text_layer"
    assert not partial.text_bytes


def test_active_pdf_is_quarantined_without_technical_output(tmp_path: Path) -> None:
    source = private_file(tmp_path / "active.pdf", pdf_bytes(javascript=True))
    result = extractor(tmp_path).extract_path(source, "pdf")

    assert result.status == "quarantined"
    assert result.error_code == "pdf_active_content"
    assert not result.text_bytes


def test_docx_extracts_known_xml_without_following_external_relationships(tmp_path: Path) -> None:
    private_url = "https://private.invalid/never-fetch"
    source = private_file(
        tmp_path / "book.docx",
        docx_bytes("Документ", external_relationship=True).replace(
            b"https://private.invalid/path", private_url.encode()
        ),
    )
    result = extractor(tmp_path).extract_path(source, "docx")

    assert result.status == "ready"
    assert result.text_bytes.decode() == "Документ"
    assert private_url not in repr(result)


def test_docx_macro_and_archive_bomb_are_quarantined(tmp_path: Path) -> None:
    active = private_file(tmp_path / "active.docx", docx_bytes("text", active=True))
    bomb = private_file(tmp_path / "bomb.docx", docx_bytes("x" * (2 * 1024 * 1024)))

    assert extractor(tmp_path).extract_path(active, "docx").status == "quarantined"
    bomb_result = extractor(tmp_path, max_unpacked_bytes=1024 * 1024).extract_path(bomb, "docx")
    assert bomb_result.status == "quarantined"
    assert bomb_result.error_code in {"archive_unpacked_limit", "archive_ratio_limit"}


def test_docx_rejects_utf16_dtd_and_entity_before_xml_parse(tmp_path: Path) -> None:
    active_xml = (
        '<?xml version="1.0" encoding="UTF-16"?>'
        '<!DOCTYPE document [<!ENTITY private "expanded-secret">]>'
        '<document xmlns:w="urn:test"><w:t>&private;</w:t></document>'
    ).encode("utf-16")
    source = private_file(tmp_path / "utf16-active.docx", docx_with_document_xml(active_xml))

    result = extractor(tmp_path).extract_path(source, "docx")

    assert result.status == "quarantined"
    assert result.error_code in {"unsupported_xml_encoding", "xml_active_content"}
    assert b"expanded-secret" not in result.text_bytes


def test_epub_follows_only_local_spine_and_rejects_doctype(tmp_path: Path) -> None:
    source = private_file(tmp_path / "book.epub", epub_bytes("Chapter text"))
    active = private_file(tmp_path / "active.epub", epub_bytes("unsafe", doctype=True))

    ready = extractor(tmp_path).extract_path(source, "epub")
    quarantined = extractor(tmp_path).extract_path(active, "epub")

    assert ready.status == "ready"
    assert ready.text_bytes.decode() == "Chapter text"
    assert b"secret" not in ready.text_bytes
    assert quarantined.status == "quarantined"
    assert quarantined.error_code == "epub_active_content"


def test_image_is_verified_and_saved_without_ocr_claim(tmp_path: Path) -> None:
    output = BytesIO()
    Image.new("RGB", (24, 12), "white").save(output, format="PNG")
    source = private_file(tmp_path / "image", output.getvalue())

    detected = inspect_upload(source, declared_mime="image/png", display_name="scan.png")
    result = extractor(tmp_path).extract_path(source, detected.source_format)

    assert result.status == "partial"
    assert result.error_code == "no_ocr"
    assert result.metadata == {"height": 12, "image_format": "png", "width": 24}


def test_url_is_stored_as_note_without_network_or_url_in_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls: list[object] = []
    monkeypatch.setattr(
        "socket.create_connection",
        lambda *args, **kwargs: network_calls.append((args, kwargs)),
    )
    private_url = "https://example.invalid/private?token=do-not-log"
    result = extract_url_note(private_url, "Моя заметка")

    assert result.status == "partial"
    assert result.error_code == "external_fetch_disabled"
    assert result.text_bytes.decode() == "Моя заметка"
    assert result.metadata == {"network_fetched": False}
    assert private_url not in repr(result)
    assert network_calls == []


def test_worker_python_network_guard_blocks_socket_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    with monkeypatch.context():
        _block_network(set_attribute=monkeypatch.setattr)
        with pytest.raises(RuntimeError, match="network_disabled"):
            socket.socket()


def test_symlink_source_is_rejected_before_worker(tmp_path: Path) -> None:
    target = private_file(tmp_path / "target.txt", b"private")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(KnowledgeExtractionError, match="unsafe_source"):
        extractor(tmp_path).extract_path(link, "txt")
