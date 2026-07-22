from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .safe_media import images as safe_images
from .safe_media import pdf as safe_pdf

MAX_LAB_INPUT_BYTES = safe_images.MAX_IMAGE_INPUT_BYTES
MAX_PDF_PAGES = safe_pdf.MAX_PDF_PAGES
MAX_NORMALIZED_TOTAL_BYTES = safe_pdf.MAX_NORMALIZED_TOTAL_BYTES
PDF_RENDER_TIMEOUT_SECONDS = safe_pdf.PDF_RENDER_TIMEOUT_SECONDS
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
            or metadata.width > safe_images.MAX_IMAGE_SOURCE_DIMENSION
            or metadata.height > safe_images.MAX_IMAGE_SOURCE_DIMENSION
            or metadata.width * metadata.height > safe_images.MAX_IMAGE_PIXELS
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
    """Adapt the shared fail-closed media boundary to the private Labs domain."""

    validate_telegram_lab_metadata(metadata)
    if not data or len(data) > MAX_LAB_INPUT_BYTES:
        raise LabMediaError("invalid_input_size")
    actual_pdf = data.startswith(b"%PDF-")
    declared_pdf = metadata.mime_type == PDF_MIME
    if actual_pdf != declared_pdf:
        raise LabMediaError("mime_mismatch")
    if actual_pdf:
        try:
            pages = safe_pdf.render_pdf_pages(data, temp_root=temp_root)
        except safe_pdf.SafePdfError as exc:
            raise LabMediaError(str(exc)) from None
        return ProcessedLabDocument("pdf", tuple(_lab_pdf_page(page) for page in pages))
    declared_mime = "image/jpeg" if metadata.source == "photo" else metadata.mime_type
    try:
        normalized = safe_images.normalize_image(data, declared_mime=declared_mime)
    except safe_images.SafeImageError as exc:
        raise LabMediaError(str(exc)) from None
    return ProcessedLabDocument("image", (_lab_image_page(normalized),))


def _lab_image_page(image: safe_images.NormalizedImage) -> NormalizedLabPage:
    return NormalizedLabPage(
        image.image_bytes,
        image.mime_type,
        image.width,
        image.height,
        image.sha256,
    )


def _lab_pdf_page(page: safe_pdf.NormalizedPdfPage) -> NormalizedLabPage:
    return NormalizedLabPage(
        page.image_bytes,
        page.mime_type,
        page.width,
        page.height,
        page.sha256,
    )
