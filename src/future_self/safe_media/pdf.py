from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .images import MAX_IMAGE_INPUT_BYTES, MAX_IMAGE_OUTPUT_BYTES
from .subprocess import (
    SafeSubprocessError,
    ensure_child,
    private_temporary_directory,
    regular_private_file,
    run_isolated_python_module,
    write_private_file,
)

MAX_PDF_PAGES = 10
MAX_PDF_PAGE_POINTS = 2_000
MAX_RENDERED_PAGE_DIMENSION = 3_000
MAX_RENDERED_PAGE_PIXELS = 12_000_000
MAX_NORMALIZED_TOTAL_BYTES = 8 * 1024 * 1024
PDF_RENDER_TIMEOUT_SECONDS = 30
_MAX_MANIFEST_BYTES = 16_384


class SafePdfError(ValueError):
    """A stable, non-sensitive PDF validation failure code."""


@dataclass(frozen=True, slots=True)
class NormalizedPdfPage:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    sha256: str


def render_pdf_pages(data: bytes, *, temp_root: Path) -> tuple[NormalizedPdfPage, ...]:
    """Inspect and rasterize a PDF in a bounded, secret-free local subprocess."""

    if not data or len(data) > MAX_IMAGE_INPUT_BYTES or not data.startswith(b"%PDF-"):
        raise SafePdfError("invalid_input_size")
    try:
        with private_temporary_directory(temp_root, prefix="render-") as work:
            input_path = work / "input.pdf"
            output_path = work / "pages"
            output_path.mkdir(mode=0o700)
            write_private_file(input_path, data)
            try:
                result = run_isolated_python_module(
                    "future_self.safe_media.pdf_worker",
                    (str(input_path), str(output_path)),
                    cwd=work,
                    timeout_seconds=PDF_RENDER_TIMEOUT_SECONDS,
                )
            finally:
                input_path.unlink(missing_ok=True)
            if result.returncode != 0:
                raise SafePdfError("unsafe_or_unrenderable_pdf")
            manifest = _read_manifest(output_path / "manifest.json")
            pages: list[NormalizedPdfPage] = []
            total = 0
            for index, item in enumerate(manifest):
                expected_name = f"page-{index:04d}.jpg"
                if item.get("file") != expected_name:
                    raise SafePdfError("invalid_renderer_output")
                page_path = (output_path / expected_name).resolve()
                ensure_child(output_path.resolve(), page_path)
                if not regular_private_file(page_path, max_bytes=MAX_IMAGE_OUTPUT_BYTES):
                    raise SafePdfError("invalid_renderer_output")
                page_bytes = page_path.read_bytes()
                total += len(page_bytes)
                if not page_bytes.startswith(b"\xff\xd8\xff") or total > MAX_NORMALIZED_TOTAL_BYTES:
                    raise SafePdfError("normalized_too_large")
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
                    raise SafePdfError("invalid_renderer_output")
                pages.append(NormalizedPdfPage(page_bytes, "image/jpeg", width, height, sha256))
            if not pages or len(pages) > MAX_PDF_PAGES:
                raise SafePdfError("invalid_page_count")
            return tuple(pages)
    except SafePdfError:
        raise
    except SafeSubprocessError as exc:
        code = "renderer_timeout" if str(exc) == "worker_timeout" else str(exc)
        raise SafePdfError(code) from None
    except (OSError, TypeError, ValueError):
        raise SafePdfError("invalid_renderer_output") from None


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not regular_private_file(path, max_bytes=_MAX_MANIFEST_BYTES):
        raise SafePdfError("invalid_renderer_output")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SafePdfError("invalid_renderer_output") from None
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SafePdfError("invalid_renderer_output")
    return value
