from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .vision_images import (
    MAX_IMAGE_INPUT_BYTES,
    MAX_IMAGE_OUTPUT_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SOURCE_DIMENSION,
    NormalizedVisionImage,
    VisionImageError,
    normalize_vision_image,
)

MAX_LAB_INPUT_BYTES = MAX_IMAGE_INPUT_BYTES
MAX_PDF_PAGES = 10
MAX_PDF_PAGE_POINTS = 2_000
MAX_RENDERED_PAGE_DIMENSION = 3_000
MAX_RENDERED_PAGE_PIXELS = 12_000_000
MAX_NORMALIZED_TOTAL_BYTES = 8 * 1024 * 1024
PDF_RENDER_TIMEOUT_SECONDS = 30
LAB_UPLOAD_TTL_SECONDS = 20 * 60
MAX_PENDING_LAB_SESSIONS = 16
MAX_PENDING_LAB_BYTES = 32 * 1024 * 1024

IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/webp"})
PDF_MIME = "application/pdf"


class LabMediaError(ValueError):
    """A non-sensitive, user-safe media validation failure code."""


@dataclass(frozen=True, slots=True)
class TelegramLabMetadata:
    source: str
    file_size: int | None
    mime_type: str | None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class NormalizedLabPage:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ProcessedLabDocument:
    source_type: str
    pages: tuple[NormalizedLabPage, ...]


def validate_telegram_lab_metadata(metadata: TelegramLabMetadata) -> None:
    if metadata.source not in {"photo", "document"}:
        raise LabMediaError("unsupported_source")
    if metadata.file_size is None or metadata.file_size <= 0:
        raise LabMediaError("missing_size")
    if metadata.file_size > MAX_LAB_INPUT_BYTES:
        raise LabMediaError("input_too_large")
    if metadata.source == "photo":
        if metadata.mime_type not in {None, "image/jpeg"}:
            raise LabMediaError("unsupported_mime")
        if metadata.width is None or metadata.height is None:
            raise LabMediaError("missing_dimensions")
        if (
            metadata.width <= 0
            or metadata.height <= 0
            or metadata.width > MAX_IMAGE_SOURCE_DIMENSION
            or metadata.height > MAX_IMAGE_SOURCE_DIMENSION
            or metadata.width * metadata.height > MAX_IMAGE_PIXELS
        ):
            raise LabMediaError("too_many_pixels")
        return
    if metadata.mime_type not in IMAGE_MIMES | {PDF_MIME}:
        raise LabMediaError("unsupported_mime")


def process_lab_upload(
    data: bytes,
    metadata: TelegramLabMetadata,
    *,
    temp_root: Path,
) -> ProcessedLabDocument:
    validate_telegram_lab_metadata(metadata)
    if not data or len(data) > MAX_LAB_INPUT_BYTES:
        raise LabMediaError("invalid_input_size")
    actual_pdf = data.startswith(b"%PDF-")
    declared_pdf = metadata.mime_type == PDF_MIME
    if actual_pdf != declared_pdf:
        raise LabMediaError("mime_mismatch")
    if actual_pdf:
        return ProcessedLabDocument("pdf", _render_pdf(data, temp_root=temp_root))
    declared_mime = "image/jpeg" if metadata.source == "photo" else metadata.mime_type
    try:
        normalized = normalize_vision_image(data, declared_mime=declared_mime)
    except VisionImageError as exc:
        raise LabMediaError(str(exc)) from None
    return ProcessedLabDocument("image", (_lab_page(normalized),))


def _lab_page(image: NormalizedVisionImage) -> NormalizedLabPage:
    return NormalizedLabPage(
        image_bytes=image.image_bytes,
        mime_type=image.mime_type,
        width=image.width,
        height=image.height,
        sha256=image.sha256,
    )


def _render_pdf(data: bytes, *, temp_root: Path) -> tuple[NormalizedLabPage, ...]:
    root = temp_root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _tighten_directory(root)
    with tempfile.TemporaryDirectory(prefix="render-", dir=root) as directory:
        work = Path(directory).resolve()
        _ensure_child(root, work)
        _tighten_directory(work)
        input_path = work / "input.pdf"
        output_path = work / "pages"
        output_path.mkdir(mode=0o700)
        _write_private_file(input_path, data)
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "PATH",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "LD_LIBRARY_PATH",
            }
        }
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "future_self.lab_pdf_worker",
                    str(input_path),
                    str(output_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                timeout=PDF_RENDER_TIMEOUT_SECONDS,
                check=False,
                env=env,
                cwd=work,
            )
        except subprocess.TimeoutExpired:
            raise LabMediaError("renderer_timeout") from None
        finally:
            input_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise LabMediaError("unsafe_or_unrenderable_pdf")
        manifest_path = output_path / "manifest.json"
        manifest = _read_manifest(manifest_path)
        pages: list[NormalizedLabPage] = []
        total = 0
        for index, item in enumerate(manifest):
            expected_name = f"page-{index:04d}.jpg"
            if item.get("file") != expected_name:
                raise LabMediaError("invalid_renderer_output")
            page_path = (output_path / expected_name).resolve()
            _ensure_child(output_path.resolve(), page_path)
            info = page_path.lstat()
            if not stat.S_ISREG(info.st_mode) or page_path.is_symlink():
                raise LabMediaError("invalid_renderer_output")
            page_bytes = page_path.read_bytes()
            total += len(page_bytes)
            if (
                not page_bytes.startswith(b"\xff\xd8\xff")
                or len(page_bytes) > MAX_IMAGE_OUTPUT_BYTES
                or total > MAX_NORMALIZED_TOTAL_BYTES
            ):
                raise LabMediaError("normalized_too_large")
            width = int(item.get("width", 0))
            height = int(item.get("height", 0))
            sha256 = str(item.get("sha256", ""))
            if (
                width <= 0
                or height <= 0
                or width > MAX_RENDERED_PAGE_DIMENSION
                or height > MAX_RENDERED_PAGE_DIMENSION
                or width * height > MAX_RENDERED_PAGE_PIXELS
                or hashlib.sha256(page_bytes).hexdigest() != sha256
            ):
                raise LabMediaError("invalid_renderer_output")
            pages.append(NormalizedLabPage(page_bytes, "image/jpeg", width, height, sha256))
        if not pages or len(pages) > MAX_PDF_PAGES:
            raise LabMediaError("invalid_page_count")
        return tuple(pages)


def _read_manifest(path: Path) -> list[dict[str, object]]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > 16_384:
            raise LabMediaError("invalid_renderer_output")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise LabMediaError("invalid_renderer_output") from None
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise LabMediaError("invalid_renderer_output")
    return value


def _write_private_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        path.unlink(missing_ok=True)
        raise


def _tighten_directory(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise LabMediaError("unsafe_temporary_storage") from exc


def _ensure_child(parent: Path, child: Path) -> None:
    if child == parent or not child.is_relative_to(parent):
        raise LabMediaError("unsafe_temporary_storage")
