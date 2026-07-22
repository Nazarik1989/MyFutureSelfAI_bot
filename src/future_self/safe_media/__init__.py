"""Domain-neutral, fail-closed primitives for untrusted media."""

from .images import (
    MAX_IMAGE_DISPLAY_DIMENSION,
    MAX_IMAGE_INPUT_BYTES,
    MAX_IMAGE_OUTPUT_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SOURCE_DIMENSION,
    NormalizedImage,
    SafeImageError,
    normalize_image,
)
from .pdf import (
    MAX_NORMALIZED_TOTAL_BYTES,
    MAX_PDF_PAGES,
    PDF_RENDER_TIMEOUT_SECONDS,
    NormalizedPdfPage,
    SafePdfError,
    render_pdf_pages,
)

__all__ = [
    "MAX_IMAGE_DISPLAY_DIMENSION",
    "MAX_IMAGE_INPUT_BYTES",
    "MAX_IMAGE_OUTPUT_BYTES",
    "MAX_IMAGE_PIXELS",
    "MAX_IMAGE_SOURCE_DIMENSION",
    "MAX_NORMALIZED_TOTAL_BYTES",
    "MAX_PDF_PAGES",
    "PDF_RENDER_TIMEOUT_SECONDS",
    "NormalizedImage",
    "NormalizedPdfPage",
    "SafeImageError",
    "SafePdfError",
    "normalize_image",
    "render_pdf_pages",
]
