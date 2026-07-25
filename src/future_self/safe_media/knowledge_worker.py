from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import socket
import stat
import sys
import unicodedata
import warnings
import zipfile
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

from .subprocess import regular_private_file, write_private_file

_PROTOCOL_VERSION = 1
_SUPPORTED_FORMATS = frozenset({"text", "txt", "markdown", "pdf", "docx", "epub", "image"})
_MAX_INPUT_BYTES = 100 * 1024 * 1024
_MAX_XML_BYTES = 16 * 1024 * 1024
_MAX_OBJECTS = 50_000
_MAX_DEPTH = 100
_ARCHIVE_RATIO = 100
_ARCHIVE_RATIO_ALLOWANCE = 1024 * 1024
_IMAGE_MAX_PIXELS = 40_000_000
_IMAGE_MAX_DIMENSION = 24_000
_XML_UNSAFE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_PDF_UNSAFE_RAW = re.compile(
    rb"/(?:AA|AcroForm|EmbeddedFile|Filespec|GoToE|ImportData|JavaScript|JS|Launch|"
    rb"OpenAction|RichMedia|SubmitForm|XFA)\b"
)
_PDF_UNSAFE_KEYS = frozenset(
    {
        "/AA",
        "/AcroForm",
        "/EmbeddedFile",
        "/Filespec",
        "/GoToE",
        "/ImportData",
        "/JavaScript",
        "/JS",
        "/Launch",
        "/OpenAction",
        "/RichMedia",
        "/SubmitForm",
        "/XFA",
    }
)

Image.MAX_IMAGE_PIXELS = _IMAGE_MAX_PIXELS


class WorkerFailure(ValueError):
    def __init__(self, code: str, *, quarantined: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.quarantined = quarantined


class _SafeHtmlText(HTMLParser):
    _IGNORED = frozenset({"script", "style", "noscript", "svg", "math", "object", "embed"})
    _BLOCKS = frozenset(
        {
            "address",
            "article",
            "aside",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "figcaption",
            "figure",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "nav",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "td",
            "th",
            "tr",
            "ul",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered in self._IGNORED:
            self.ignored_depth += 1
        elif self.ignored_depth == 0 and lowered in self._BLOCKS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() in self._IGNORED:
            self.ignored_depth = max(0, self.ignored_depth - 1)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in self._IGNORED:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        elif self.ignored_depth == 0 and lowered in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth == 0:
            self.parts.append(data)


def _set_resource_limits() -> None:
    try:
        import resource
    except ImportError:
        return
    limits = (
        (resource.RLIMIT_CPU, (30, 30)),
        (resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024)),
        (resource.RLIMIT_FSIZE, (24 * 1024 * 1024, 24 * 1024 * 1024)),
        (resource.RLIMIT_NOFILE, (64, 64)),
    )
    for resource_id, value in limits:
        try:
            resource.setrlimit(resource_id, value)
        except (OSError, ValueError):
            pass


def _block_network(*, set_attribute: Callable[[object, str, object], object] = setattr) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("network_disabled")

    # Deny socket construction itself, including UDP/sendto paths. Docker's
    # network-less namespace remains the independent OS boundary.
    set_attribute(socket, "socket", forbidden)
    set_attribute(socket, "socketpair", forbidden)
    set_attribute(socket, "fromfd", forbidden)
    set_attribute(socket, "create_connection", forbidden)


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned: list[str] = []
    blank_count = 0
    for raw_line in value.split("\n"):
        line = "".join(
            character
            for character in raw_line
            if character == "\t" or unicodedata.category(character) not in {"Cc", "Cs"}
        ).rstrip()
        if line:
            blank_count = 0
            cleaned.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
    return "\n".join(cleaned).strip()


def _read_regular(path: Path, *, limit: int = _MAX_INPUT_BYTES) -> bytes:
    if not regular_private_file(path, max_bytes=limit):
        raise WorkerFailure("unsafe_input", quarantined=True)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WorkerFailure("input_unavailable") from exc


def _safe_xml(data: bytes) -> ElementTree.Element:
    if not data or len(data) > _MAX_XML_BYTES:
        raise WorkerFailure("xml_size", quarantined=True)
    try:
        # Office/EPUB package XML is required to be UTF-8 at this boundary. This
        # makes the active-content scan encoding-independent and fail-closed;
        # ElementTree otherwise accepts UTF-16 declarations that hide DTD bytes.
        decoded = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise WorkerFailure("unsupported_xml_encoding", quarantined=True) from exc
    if "\x00" in decoded or _XML_UNSAFE.search(decoded):
        raise WorkerFailure("xml_active_content", quarantined=True)
    try:
        return ElementTree.fromstring(decoded)
    except ElementTree.ParseError as exc:
        raise WorkerFailure("invalid_xml") from exc


def _local_name(tag: object) -> str:
    value = str(tag)
    return value.rsplit("}", 1)[-1].casefold()


def _extract_plain(path: Path) -> tuple[str, dict[str, object]]:
    data = _read_regular(path)
    if b"\x00" in data:
        raise WorkerFailure("binary_text", quarantined=True)
    try:
        text = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise WorkerFailure("unsupported_text_encoding") from exc
    return _normalize_text(text), {"encoding": "utf-8"}


def _walk_pdf_objects(root: object) -> None:
    stack: list[tuple[object, int]] = [(root, 0)]
    indirect_seen: set[tuple[int, int]] = set()
    visited = 0
    while stack:
        value, depth = stack.pop()
        visited += 1
        if visited > _MAX_OBJECTS or depth > _MAX_DEPTH:
            raise WorkerFailure("pdf_object_limit", quarantined=True)
        if isinstance(value, IndirectObject):
            key = (value.idnum, value.generation)
            if key in indirect_seen:
                continue
            indirect_seen.add(key)
            try:
                stack.append((value.get_object(), depth + 1))
            except Exception as exc:
                raise WorkerFailure("invalid_pdf") from exc
        elif isinstance(value, DictionaryObject):
            for key, child in value.items():
                if str(key) in _PDF_UNSAFE_KEYS:
                    raise WorkerFailure("pdf_active_content", quarantined=True)
                stack.append((child, depth + 1))
        elif isinstance(value, ArrayObject):
            stack.extend((child, depth + 1) for child in value)


def _extract_pdf(path: Path, *, max_pages: int) -> tuple[str, dict[str, object]]:
    raw = _read_regular(path)
    if not raw.startswith(b"%PDF-") or not raw.rstrip().endswith(b"%%EOF"):
        raise WorkerFailure("invalid_pdf")
    if _PDF_UNSAFE_RAW.search(raw):
        raise WorkerFailure("pdf_active_content", quarantined=True)
    try:
        reader = PdfReader(str(path), strict=True)
        if reader.is_encrypted:
            raise WorkerFailure("pdf_encrypted", quarantined=True)
        page_count = len(reader.pages)
        if page_count < 1 or page_count > max_pages:
            raise WorkerFailure("pdf_page_limit", quarantined=True)
        _walk_pdf_objects(reader.trailer)
        pages: list[str] = []
        for page in reader.pages:
            width = abs(float(page.mediabox.width))
            height = abs(float(page.mediabox.height))
            if (
                not math.isfinite(width)
                or not math.isfinite(height)
                or width <= 0
                or height <= 0
                or width > 20_000
                or height > 20_000
            ):
                raise WorkerFailure("pdf_page_dimensions", quarantined=True)
            extracted = page.extract_text() or ""
            pages.append(extracted)
    except WorkerFailure:
        raise
    except Exception as exc:
        raise WorkerFailure("invalid_pdf") from exc
    return _normalize_text("\n\n".join(pages)), {"page_count": page_count}


def _safe_archive_name(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise WorkerFailure("archive_unsafe_path", quarantined=True)
    decoded = unquote(name)
    path = PurePosixPath(decoded)
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise WorkerFailure("archive_unsafe_path", quarantined=True)
    return path


def _inspect_archive(
    archive: zipfile.ZipFile,
    *,
    max_files: int,
    max_unpacked_bytes: int,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > max_files:
        raise WorkerFailure("archive_file_limit", quarantined=True)
    members: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in infos:
        path = _safe_archive_name(info.filename)
        normalized = path.as_posix()
        if normalized in members:
            raise WorkerFailure("archive_duplicate_member", quarantined=True)
        if info.flag_bits & 0x1:
            raise WorkerFailure("archive_encrypted", quarantined=True)
        unix_mode = (info.external_attr >> 16) & 0o170000
        if unix_mode == stat.S_IFLNK:
            raise WorkerFailure("archive_link", quarantined=True)
        if info.file_size < 0 or info.compress_size < 0:
            raise WorkerFailure("archive_invalid_size", quarantined=True)
        total += info.file_size
        if total > max_unpacked_bytes:
            raise WorkerFailure("archive_unpacked_limit", quarantined=True)
        if (
            info.file_size > _ARCHIVE_RATIO_ALLOWANCE
            and info.file_size > max(1, info.compress_size) * _ARCHIVE_RATIO
        ):
            raise WorkerFailure("archive_ratio_limit", quarantined=True)
        members[normalized] = info
    return members


def _read_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
) -> bytes:
    if info.file_size > max_bytes:
        raise WorkerFailure("archive_member_limit", quarantined=True)
    output = bytearray()
    try:
        with archive.open(info, "r") as stream:
            while chunk := stream.read(64 * 1024):
                output.extend(chunk)
                if len(output) > max_bytes or len(output) > info.file_size:
                    raise WorkerFailure("archive_member_limit", quarantined=True)
    except WorkerFailure:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise WorkerFailure("invalid_archive") from exc
    if len(output) != info.file_size:
        raise WorkerFailure("archive_size_mismatch", quarantined=True)
    return bytes(output)


def _docx_xml_text(root: ElementTree.Element) -> str:
    parts: list[str] = []
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "t" and element.text:
            parts.append(element.text)
        elif name in {"tab"}:
            parts.append("\t")
        elif name in {"br", "cr", "p"}:
            parts.append("\n")
    return "".join(parts)


def _extract_docx(
    path: Path,
    *,
    max_files: int,
    max_unpacked_bytes: int,
) -> tuple[str, dict[str, object]]:
    if not _read_regular(path, limit=_MAX_INPUT_BYTES).startswith(b"PK"):
        raise WorkerFailure("invalid_docx")
    try:
        with zipfile.ZipFile(path) as archive:
            members = _inspect_archive(
                archive,
                max_files=max_files,
                max_unpacked_bytes=max_unpacked_bytes,
            )
            required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
            if not required.issubset(members):
                raise WorkerFailure("invalid_docx")
            lowered_names = {name.casefold() for name in members}
            if any(
                "vbaproject" in name or "/embeddings/" in f"/{name}" or "/activex/" in f"/{name}"
                for name in lowered_names
            ):
                raise WorkerFailure("docx_active_content", quarantined=True)
            content_types = _read_member(
                archive, members["[Content_Types].xml"], max_bytes=_MAX_XML_BYTES
            )
            if b"macroenabled" in content_types.lower():
                raise WorkerFailure("docx_active_content", quarantined=True)
            _safe_xml(content_types)
            xml_members = ["word/document.xml"]
            xml_members.extend(
                sorted(
                    name
                    for name in members
                    if re.fullmatch(
                        r"word/(?:footnotes|endnotes|header[0-9]+|footer[0-9]+)\.xml",
                        name,
                        flags=re.IGNORECASE,
                    )
                )
            )
            parts: list[str] = []
            for name in xml_members:
                data = _read_member(archive, members[name], max_bytes=_MAX_XML_BYTES)
                parts.append(_docx_xml_text(_safe_xml(data)))
    except WorkerFailure:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise WorkerFailure("invalid_docx") from exc
    return _normalize_text("\n\n".join(parts)), {"part_count": len(xml_members)}


def _safe_archive_join(base: PurePosixPath, reference: str) -> str:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc:
        raise WorkerFailure("epub_external_reference", quarantined=True)
    decoded = unquote(parsed.path)
    if not decoded or "\\" in decoded or "\x00" in decoded:
        raise WorkerFailure("archive_unsafe_path", quarantined=True)
    relative = PurePosixPath(decoded)
    if relative.is_absolute() or any(":" in part for part in relative.parts):
        raise WorkerFailure("archive_unsafe_path", quarantined=True)
    combined = base.joinpath(relative)
    normalized_parts: list[str] = []
    for part in combined.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                raise WorkerFailure("archive_unsafe_path", quarantined=True)
            normalized_parts.pop()
        else:
            normalized_parts.append(part)
    if not normalized_parts:
        raise WorkerFailure("archive_unsafe_path", quarantined=True)
    return PurePosixPath(*normalized_parts).as_posix()


def _extract_html(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise WorkerFailure("unsupported_epub_encoding") from exc
    if "\x00" in text or _XML_UNSAFE.search(text):
        raise WorkerFailure("epub_active_content", quarantined=True)
    parser = _SafeHtmlText()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise WorkerFailure("invalid_epub_content") from exc
    return "".join(parser.parts)


def _extract_epub(
    path: Path,
    *,
    max_files: int,
    max_unpacked_bytes: int,
) -> tuple[str, dict[str, object]]:
    if not _read_regular(path, limit=_MAX_INPUT_BYTES).startswith(b"PK"):
        raise WorkerFailure("invalid_epub")
    try:
        with zipfile.ZipFile(path) as archive:
            members = _inspect_archive(
                archive,
                max_files=max_files,
                max_unpacked_bytes=max_unpacked_bytes,
            )
            if "mimetype" not in members or "META-INF/container.xml" not in members:
                raise WorkerFailure("invalid_epub")
            mime_info = members["mimetype"]
            mime = _read_member(archive, mime_info, max_bytes=128)
            if (
                mime != b"application/epub+zip"
                or mime_info.compress_type != zipfile.ZIP_STORED
                or mime_info.header_offset != 0
            ):
                raise WorkerFailure("invalid_epub")
            container = _safe_xml(
                _read_member(archive, members["META-INF/container.xml"], max_bytes=_MAX_XML_BYTES)
            )
            rootfiles = [
                element for element in container.iter() if _local_name(element.tag) == "rootfile"
            ]
            if len(rootfiles) != 1:
                raise WorkerFailure("invalid_epub")
            opf_path = _safe_archive_name(str(rootfiles[0].attrib.get("full-path", ""))).as_posix()
            if opf_path not in members:
                raise WorkerFailure("invalid_epub")
            opf = _safe_xml(_read_member(archive, members[opf_path], max_bytes=_MAX_XML_BYTES))
            manifest: dict[str, tuple[str, str]] = {}
            spine: list[str] = []
            for element in opf.iter():
                name = _local_name(element.tag)
                if name == "item":
                    item_id = str(element.attrib.get("id", ""))
                    href = str(element.attrib.get("href", ""))
                    media_type = str(element.attrib.get("media-type", "")).casefold()
                    if not item_id or not href or item_id in manifest:
                        raise WorkerFailure("invalid_epub")
                    manifest[item_id] = (href, media_type)
                elif name == "itemref":
                    item_id = str(element.attrib.get("idref", ""))
                    if item_id:
                        spine.append(item_id)
            if not spine or len(spine) > max_files:
                raise WorkerFailure("invalid_epub")
            opf_parent = PurePosixPath(opf_path).parent
            parts: list[str] = []
            for item_id in spine:
                item = manifest.get(item_id)
                if item is None:
                    raise WorkerFailure("invalid_epub")
                href, media_type = item
                if media_type not in {"application/xhtml+xml", "text/html"}:
                    continue
                content_path = _safe_archive_join(opf_parent, href)
                if content_path not in members:
                    raise WorkerFailure("invalid_epub")
                content = _read_member(archive, members[content_path], max_bytes=_MAX_XML_BYTES)
                parts.append(_extract_html(content))
    except WorkerFailure:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise WorkerFailure("invalid_epub") from exc
    return _normalize_text("\n\n".join(parts)), {"spine_items": len(spine)}


def _inspect_image(path: Path) -> tuple[str, dict[str, object]]:
    if not regular_private_file(path, max_bytes=_MAX_INPUT_BYTES):
        raise WorkerFailure("unsafe_input", quarantined=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                actual_format = str(image.format or "").upper()
                width, height = image.size
                animated = (
                    bool(getattr(image, "is_animated", False))
                    or int(getattr(image, "n_frames", 1)) != 1
                )
                if actual_format not in {"JPEG", "PNG", "WEBP"}:
                    raise WorkerFailure("unsupported_image")
                if (
                    animated
                    or width <= 0
                    or height <= 0
                    or width > _IMAGE_MAX_DIMENSION
                    or height > _IMAGE_MAX_DIMENSION
                    or width * height > _IMAGE_MAX_PIXELS
                ):
                    raise WorkerFailure("unsafe_image", quarantined=True)
                image.verify()
    except WorkerFailure:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise WorkerFailure("unsafe_image", quarantined=True) from exc
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError) as exc:
        raise WorkerFailure("invalid_image") from exc
    return "", {"image_format": actual_format.casefold(), "width": width, "height": height}


def _write_result(
    output_path: Path,
    *,
    source_format: str,
    status: str,
    text: str = "",
    error_code: str | None = None,
    metadata: dict[str, object] | None = None,
    max_chars: int,
) -> None:
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
        status = "partial"
        error_code = "text_limit_reached"
    encoded = text.encode("utf-8")
    text_name: str | None = None
    text_hash: str | None = None
    if encoded:
        text_name = "text.txt"
        text_hash = hashlib.sha256(encoded).hexdigest()
        write_private_file(output_path / text_name, encoded)
    manifest = {
        "protocol": _PROTOCOL_VERSION,
        "format": source_format,
        "status": status,
        "text_file": text_name,
        "text_bytes": len(encoded),
        "text_sha256": text_hash,
        "error_code": error_code,
        "metadata": metadata or {},
    }
    write_private_file(
        output_path / "manifest.json",
        json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        ),
    )


def main() -> int:
    if len(sys.argv) != 8:
        return 2
    _set_resource_limits()
    _block_network()
    logging.disable(logging.CRITICAL)
    source_format = sys.argv[1]
    input_path = Path(sys.argv[2]).resolve()
    output_path = Path(sys.argv[3]).resolve()
    try:
        max_chars = int(sys.argv[4])
        max_archive_files = int(sys.argv[5])
        max_unpacked_bytes = int(sys.argv[6])
        max_pdf_pages = int(sys.argv[7])
    except ValueError:
        return 2
    if (
        source_format not in _SUPPORTED_FORMATS
        or not 1_000 <= max_chars <= 5_000_000
        or not 10 <= max_archive_files <= 5_000
        or not 1024 * 1024 <= max_unpacked_bytes <= 256 * 1024 * 1024
        or not 1 <= max_pdf_pages <= 500
        or input_path.parent != output_path.parent
        or not output_path.is_dir()
        or output_path.is_symlink()
        or not regular_private_file(input_path, max_bytes=_MAX_INPUT_BYTES)
    ):
        return 2
    try:
        if source_format in {"text", "txt", "markdown"}:
            text, metadata = _extract_plain(input_path)
        elif source_format == "pdf":
            text, metadata = _extract_pdf(input_path, max_pages=max_pdf_pages)
        elif source_format == "docx":
            text, metadata = _extract_docx(
                input_path,
                max_files=max_archive_files,
                max_unpacked_bytes=max_unpacked_bytes,
            )
        elif source_format == "epub":
            text, metadata = _extract_epub(
                input_path,
                max_files=max_archive_files,
                max_unpacked_bytes=max_unpacked_bytes,
            )
        else:
            text, metadata = _inspect_image(input_path)
        status = "ready" if text else "partial"
        reason = None if text else ("no_ocr" if source_format == "image" else "no_text_layer")
        _write_result(
            output_path,
            source_format=source_format,
            status=status,
            text=text,
            error_code=reason,
            metadata=metadata,
            max_chars=max_chars,
        )
    except WorkerFailure as exc:
        _write_result(
            output_path,
            source_format=source_format,
            status="quarantined" if exc.quarantined else "failed",
            error_code=exc.code,
            max_chars=max_chars,
        )
    except BaseException:
        _write_result(
            output_path,
            source_format=source_format,
            status="failed",
            error_code="parser_failure",
            max_chars=max_chars,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
