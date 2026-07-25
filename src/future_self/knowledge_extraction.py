from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from .safe_media.subprocess import (
    SafeSubprocessError,
    private_temporary_directory,
    regular_private_file,
    run_isolated_python_module,
)

SourceFormat = Literal["text", "txt", "markdown", "pdf", "docx", "epub", "image", "url"]
ExtractionStatus = Literal["ready", "partial", "failed", "quarantined"]

_WORKER_MODULE = "future_self.safe_media.knowledge_worker"
_MANIFEST_MAX_BYTES = 16 * 1024
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)
_ALLOWED_METADATA = frozenset(
    {
        "encoding",
        "page_count",
        "part_count",
        "spine_items",
        "image_format",
        "width",
        "height",
        "network_fetched",
    }
)

_FORMAT_MIMES: dict[SourceFormat, frozenset[str]] = {
    "text": frozenset({"text/plain"}),
    "pdf": frozenset({"application/pdf"}),
    "txt": frozenset({"text/plain"}),
    "markdown": frozenset({"text/markdown", "text/plain", "text/x-markdown"}),
    "docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
    "epub": frozenset({"application/epub+zip"}),
    "image": frozenset({"image/jpeg", "image/png", "image/webp"}),
    "url": frozenset(),
}


class KnowledgeExtractionError(ValueError):
    """A non-sensitive boundary error with runner retry guidance."""

    def __init__(self, code: str, *, retryable: bool = False, quarantined: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.quarantined = quarantined


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    max_source_bytes: int = 25 * 1024 * 1024
    max_text_chars: int = 2_000_000
    max_archive_files: int = 1_000
    max_unpacked_bytes: int = 64 * 1024 * 1024
    max_pdf_pages: int = 100
    timeout_seconds: int = 45

    def __post_init__(self) -> None:
        if not 1_000_000 <= self.max_source_bytes <= 100 * 1024 * 1024:
            raise ValueError("max_source_bytes is outside the safe worker range")
        if not 1_000 <= self.max_text_chars <= 5_000_000:
            raise ValueError("max_text_chars is outside the safe worker range")
        if not 10 <= self.max_archive_files <= 5_000:
            raise ValueError("max_archive_files is outside the safe worker range")
        if not 1024 * 1024 <= self.max_unpacked_bytes <= 256 * 1024 * 1024:
            raise ValueError("max_unpacked_bytes is outside the safe worker range")
        if not 1 <= self.max_pdf_pages <= 500:
            raise ValueError("max_pdf_pages is outside the safe worker range")
        if not 5 <= self.timeout_seconds <= 300:
            raise ValueError("timeout_seconds is outside the safe worker range")


@dataclass(frozen=True, slots=True)
class DetectedUpload:
    source_format: SourceFormat
    detected_mime: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    status: ExtractionStatus
    source_format: SourceFormat
    text_bytes: bytes = field(default=b"", repr=False)
    text_sha256: str | None = None
    error_code: str | None = None
    metadata: Mapping[str, str | int | bool] = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return bool(self.text_bytes)


def inspect_upload(
    path: Path,
    *,
    declared_mime: str | None,
    display_name: str | None,
    max_source_bytes: int = 25 * 1024 * 1024,
) -> DetectedUpload:
    """Sniff a bounded envelope; archive internals stay in the runner worker."""

    source = Path(path)
    info = _private_regular_info(source, max_bytes=max_source_bytes)
    prefix = _read_prefix(source, 64 * 1024)
    normalized_mime = _normalize_mime(declared_mime)
    safe_suffix = _suffix_only(display_name)

    actual_mime: str
    source_format: SourceFormat
    if prefix.startswith(b"%PDF-"):
        source_format, actual_mime = "pdf", "application/pdf"
    elif prefix.startswith(_IMAGE_SIGNATURES[0][0]):
        source_format, actual_mime = "image", "image/jpeg"
    elif prefix.startswith(_IMAGE_SIGNATURES[1][0]):
        source_format, actual_mime = "image", "image/png"
    elif len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        source_format, actual_mime = "image", "image/webp"
    elif prefix.startswith(b"PK\x03\x04"):
        if normalized_mime == "application/epub+zip" or safe_suffix == ".epub":
            source_format, actual_mime = "epub", "application/epub+zip"
        elif (
            normalized_mime
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or safe_suffix == ".docx"
        ):
            source_format = "docx"
            actual_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            raise KnowledgeExtractionError("ambiguous_zip")
    else:
        if b"\x00" in prefix:
            raise KnowledgeExtractionError("unsupported_format", quarantined=True)
        try:
            codecs.getincrementaldecoder("utf-8-sig")("strict").decode(prefix, final=False)
        except UnicodeDecodeError as exc:
            raise KnowledgeExtractionError("unsupported_format") from exc
        source_format = (
            "markdown"
            if normalized_mime in {"text/markdown", "text/x-markdown"}
            or safe_suffix in {".md", ".markdown"}
            else "txt"
        )
        actual_mime = "text/markdown" if source_format == "markdown" else "text/plain"

    unknown_mimes = {None, "application/octet-stream", "binary/octet-stream"}
    if normalized_mime not in unknown_mimes and normalized_mime not in _FORMAT_MIMES[source_format]:
        raise KnowledgeExtractionError("mime_mismatch", quarantined=True)
    if source_format == "image" and normalized_mime not in unknown_mimes | {actual_mime}:
        raise KnowledgeExtractionError("mime_mismatch", quarantined=True)
    if safe_suffix and not _suffix_matches(source_format, safe_suffix):
        raise KnowledgeExtractionError("extension_mismatch", quarantined=True)
    return DetectedUpload(source_format, actual_mime, info.st_size)


def extract_url_note(
    url: str, note: str | None = None, *, max_chars: int = 20_000
) -> ExtractionResult:
    """Validate a link without fetching it; the private URL is never returned."""

    if not isinstance(url, str) or len(url) > 4_096 or any(ord(char) < 32 for char in url):
        raise KnowledgeExtractionError("invalid_url")
    parsed = urlsplit(url)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise KnowledgeExtractionError("invalid_url")
    normalized_note = _normalize_text(note or "")
    truncated = len(normalized_note) > max_chars
    if truncated:
        normalized_note = normalized_note[:max_chars]
    encoded = normalized_note.encode("utf-8")
    return ExtractionResult(
        status="partial",
        source_format="url",
        text_bytes=encoded,
        text_sha256=hashlib.sha256(encoded).hexdigest() if encoded else None,
        error_code="text_limit_reached" if truncated else "external_fetch_disabled",
        metadata={"network_fetched": False},
    )


class KnowledgeExtractor:
    def __init__(self, temp_root: Path, *, limits: ExtractionLimits | None = None) -> None:
        self.temp_root = Path(temp_root)
        self.limits = limits or ExtractionLimits()

    async def extract_path_async(
        self,
        path: Path,
        source_format: SourceFormat,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> ExtractionResult:
        return await asyncio.to_thread(
            self.extract_path,
            path,
            source_format,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )

    def extract_path(
        self,
        path: Path,
        source_format: SourceFormat,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> ExtractionResult:
        if source_format not in {"text", "txt", "markdown", "pdf", "docx", "epub", "image"}:
            raise KnowledgeExtractionError("unsupported_worker_format")
        source = Path(path)
        _private_regular_info(source, max_bytes=self.limits.max_source_bytes)
        try:
            with private_temporary_directory(self.temp_root, prefix="knowledge-extract-") as work:
                worker_input = work / "input.bin"
                worker_output = work / "output"
                worker_output.mkdir(mode=0o700)
                copied_size, copied_sha256 = _copy_private(
                    source, worker_input, max_bytes=self.limits.max_source_bytes
                )
                if expected_size is not None and copied_size != expected_size:
                    raise KnowledgeExtractionError("source_size_mismatch", quarantined=True)
                if expected_sha256 is not None:
                    if not _SHA256.fullmatch(expected_sha256):
                        raise KnowledgeExtractionError("invalid_expected_hash")
                    if copied_sha256 != expected_sha256:
                        raise KnowledgeExtractionError("source_hash_mismatch", quarantined=True)
                try:
                    completed = run_isolated_python_module(
                        _WORKER_MODULE,
                        (
                            source_format,
                            str(worker_input),
                            str(worker_output),
                            str(self.limits.max_text_chars),
                            str(self.limits.max_archive_files),
                            str(self.limits.max_unpacked_bytes),
                            str(self.limits.max_pdf_pages),
                        ),
                        cwd=work,
                        timeout_seconds=self.limits.timeout_seconds,
                    )
                finally:
                    worker_input.unlink(missing_ok=True)
                if completed.returncode != 0:
                    raise KnowledgeExtractionError("worker_failed", retryable=True)
                return _read_worker_result(worker_output, expected_format=source_format)
        except KnowledgeExtractionError:
            raise
        except SafeSubprocessError as exc:
            if str(exc) == "worker_timeout":
                raise KnowledgeExtractionError(
                    "worker_timeout", retryable=False, quarantined=True
                ) from None
            raise KnowledgeExtractionError("worker_boundary_failed", retryable=True) from None
        except OSError as exc:
            raise KnowledgeExtractionError("worker_io_failed", retryable=True) from exc


def _read_worker_result(output: Path, *, expected_format: SourceFormat) -> ExtractionResult:
    manifest_path = output / "manifest.json"
    if not regular_private_file(manifest_path, max_bytes=_MANIFEST_MAX_BYTES):
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True) from exc
    if not isinstance(manifest, dict):
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    expected_keys = {
        "protocol",
        "format",
        "status",
        "text_file",
        "text_bytes",
        "text_sha256",
        "error_code",
        "metadata",
    }
    if set(manifest) != expected_keys or manifest.get("protocol") != 1:
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    status = manifest.get("status")
    source_format = manifest.get("format")
    error_code = manifest.get("error_code")
    text_name = manifest.get("text_file")
    text_size = manifest.get("text_bytes")
    text_hash = manifest.get("text_sha256")
    metadata = manifest.get("metadata")
    if (
        source_format != expected_format
        or status not in {"ready", "partial", "failed", "quarantined"}
        or not isinstance(text_size, int)
        or text_size < 0
        or (error_code is not None and not isinstance(error_code, str))
        or (isinstance(error_code, str) and not _ERROR_CODE.fullmatch(error_code))
        or not isinstance(metadata, dict)
    ):
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    safe_metadata: dict[str, str | int | bool] = {}
    for key, value in metadata.items():
        if (
            key not in _ALLOWED_METADATA
            or not isinstance(value, (str, int, bool))
            or (isinstance(value, str) and len(value) > 64)
        ):
            raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
        safe_metadata[key] = value

    text = b""
    if text_name is None:
        if text_size != 0 or text_hash is not None:
            raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    elif text_name == "text.txt" and isinstance(text_hash, str) and _SHA256.fullmatch(text_hash):
        text_path = output / text_name
        if not regular_private_file(text_path, max_bytes=20 * 1024 * 1024):
            raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
        text = text_path.read_bytes()
        try:
            text.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise KnowledgeExtractionError("invalid_worker_output", retryable=True) from exc
        if len(text) != text_size or hashlib.sha256(text).hexdigest() != text_hash:
            raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    else:
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    if status == "ready" and not text:
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    if status in {"failed", "quarantined"} and text:
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    allowed_files = {"manifest.json"} | ({"text.txt"} if text else set())
    if {path.name for path in output.iterdir()} != allowed_files:
        raise KnowledgeExtractionError("invalid_worker_output", retryable=True)
    return ExtractionResult(
        status=status,
        source_format=expected_format,
        text_bytes=text,
        text_sha256=text_hash,
        error_code=error_code,
        metadata=safe_metadata,
    )


def _copy_private(source: Path, destination: Path, *, max_bytes: int) -> tuple[int, str]:
    read_flags = os.O_RDONLY
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
        write_flags |= os.O_NOFOLLOW
    source_descriptor = -1
    destination_descriptor = -1
    digest = hashlib.sha256()
    try:
        source_descriptor = os.open(source, read_flags)
        source_info = os.fstat(source_descriptor)
        _validate_info(source_info, max_bytes=max_bytes)
        destination_descriptor = os.open(destination, write_flags, 0o600)
        with (
            os.fdopen(source_descriptor, "rb") as input_stream,
            os.fdopen(destination_descriptor, "wb") as output_stream,
        ):
            source_descriptor = -1
            destination_descriptor = -1
            copied = 0
            while chunk := input_stream.read(64 * 1024):
                copied += len(chunk)
                if copied > max_bytes:
                    raise KnowledgeExtractionError("source_too_large")
                output_stream.write(chunk)
                digest.update(chunk)
            if copied != source_info.st_size:
                raise KnowledgeExtractionError("source_changed")
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    return copied, digest.hexdigest()


def _read_prefix(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read(limit)
    except OSError as exc:
        raise KnowledgeExtractionError("source_unavailable", retryable=True) from exc


def _private_regular_info(path: Path, *, max_bytes: int) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise KnowledgeExtractionError("source_missing", retryable=True) from None
    except OSError as exc:
        raise KnowledgeExtractionError("source_unavailable", retryable=True) from exc
    if path.is_symlink():
        raise KnowledgeExtractionError("unsafe_source", quarantined=True)
    try:
        _validate_info(info, max_bytes=max_bytes)
    except KnowledgeExtractionError:
        raise
    return info


def _validate_info(info: os.stat_result, *, max_bytes: int) -> None:
    private = os.name != "posix" or stat.S_IMODE(info.st_mode) & 0o077 == 0
    if (
        not stat.S_ISREG(info.st_mode)
        or not private
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > max_bytes
    ):
        raise KnowledgeExtractionError("unsafe_source", quarantined=True)


def _normalize_mime(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.split(";", 1)[0].strip().casefold()
    return normalized or None


def _suffix_only(display_name: str | None) -> str:
    if not display_name or len(display_name) > 512:
        return ""
    leaf = display_name.replace("\\", "/").rsplit("/", 1)[-1]
    return Path(leaf).suffix.casefold()


def _suffix_matches(source_format: SourceFormat, suffix: str) -> bool:
    allowed = {
        "pdf": {".pdf"},
        "text": {".txt", ""},
        "txt": {".txt", ""},
        "markdown": {".md", ".markdown", ".txt"},
        "docx": {".docx"},
        "epub": {".epub"},
        "image": {".jpg", ".jpeg", ".png", ".webp"},
        "url": {""},
    }
    return suffix in allowed[source_format]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
