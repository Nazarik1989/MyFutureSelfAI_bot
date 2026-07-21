from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import sys
from pathlib import Path
from time import monotonic
from typing import Any

import pypdfium2 as pdfium
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

from .lab_media import (
    MAX_NORMALIZED_TOTAL_BYTES,
    MAX_PDF_PAGE_POINTS,
    MAX_PDF_PAGES,
    MAX_RENDERED_PAGE_DIMENSION,
    MAX_RENDERED_PAGE_PIXELS,
    PDF_RENDER_TIMEOUT_SECONDS,
)
from .vision_images import _bounded_jpeg, _safe_rgb

_DANGEROUS_KEYS = frozenset(
    {
        "/A",
        "/AA",
        "/AcroForm",
        "/Annots",
        "/Collection",
        "/EmbeddedFile",
        "/Filespec",
        "/GoToE",
        "/ImportData",
        "/JavaScript",
        "/JS",
        "/Launch",
        "/Movie",
        "/OpenAction",
        "/RichMedia",
        "/Sound",
        "/SubmitForm",
        "/XFA",
    }
)
_DANGEROUS_RAW = re.compile(
    rb"/(?:AA|AcroForm|Annots|Collection|EmbeddedFile|Filespec|GoToE|ImportData|"
    rb"JavaScript|JS|Launch|Movie|OpenAction|RichMedia|Sound|SubmitForm|XFA)\b"
)
_MAX_OBJECTS = 20_000
_MAX_DEPTH = 80


def _set_resource_limits() -> None:
    try:
        import resource
    except ImportError:
        return
    limits = (
        (resource.RLIMIT_CPU, (20, 20)),
        (resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024)),
        (resource.RLIMIT_FSIZE, (12 * 1024 * 1024, 12 * 1024 * 1024)),
        (resource.RLIMIT_NOFILE, (64, 64)),
    )
    for resource_id, value in limits:
        try:
            resource.setrlimit(resource_id, value)
        except (OSError, ValueError):
            pass


def _block_network() -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("network_disabled")

    socket.socket.connect = forbidden  # type: ignore[method-assign]
    socket.socket.connect_ex = forbidden  # type: ignore[method-assign]
    socket.create_connection = forbidden  # type: ignore[assignment]


def _inspect_pdf(input_path: Path) -> tuple[PdfReader, list[tuple[float, float]]]:
    raw = input_path.read_bytes()
    if not raw.startswith(b"%PDF-") or not raw.rstrip().endswith(b"%%EOF"):
        raise ValueError("invalid_pdf_envelope")
    if _DANGEROUS_RAW.search(raw):
        raise ValueError("active_content")
    reader = PdfReader(str(input_path), strict=True)
    if reader.is_encrypted:
        raise ValueError("encrypted")
    count = len(reader.pages)
    if count < 1 or count > MAX_PDF_PAGES:
        raise ValueError("page_count")
    _walk_pdf_objects(reader.trailer)
    dimensions: list[tuple[float, float]] = []
    for page in reader.pages:
        width = abs(float(page.mediabox.width))
        height = abs(float(page.mediabox.height))
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0
            or height <= 0
            or width > MAX_PDF_PAGE_POINTS
            or height > MAX_PDF_PAGE_POINTS
        ):
            raise ValueError("page_dimensions")
        dimensions.append((width, height))
    return reader, dimensions


def _walk_pdf_objects(root: Any) -> None:
    stack: list[tuple[Any, int]] = [(root, 0)]
    indirect_seen: set[tuple[int, int]] = set()
    visited = 0
    while stack:
        value, depth = stack.pop()
        visited += 1
        if visited > _MAX_OBJECTS or depth > _MAX_DEPTH:
            raise ValueError("object_graph_limit")
        if isinstance(value, IndirectObject):
            key = (value.idnum, value.generation)
            if key in indirect_seen:
                continue
            indirect_seen.add(key)
            stack.append((value.get_object(), depth + 1))
        elif isinstance(value, DictionaryObject):
            for key, child in value.items():
                if str(key) in _DANGEROUS_KEYS:
                    raise ValueError("active_content")
                stack.append((child, depth + 1))
        elif isinstance(value, ArrayObject):
            stack.extend((child, depth + 1) for child in value)


def _render(input_path: Path, output_path: Path, dimensions: list[tuple[float, float]]) -> None:
    started = monotonic()
    document = pdfium.PdfDocument(str(input_path))
    manifest: list[dict[str, object]] = []
    total = 0
    try:
        if len(document) != len(dimensions):
            raise ValueError("renderer_page_mismatch")
        for index, (width_points, height_points) in enumerate(dimensions):
            if monotonic() - started > PDF_RENDER_TIMEOUT_SECONDS - 2:
                raise TimeoutError
            scale = min(
                150 / 72,
                MAX_RENDERED_PAGE_DIMENSION / max(width_points, height_points),
                math.sqrt(MAX_RENDERED_PAGE_PIXELS / (width_points * height_points)),
            )
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("render_scale")
            page = document[index]
            bitmap = page.render(scale=scale)
            try:
                rendered = bitmap.to_pil()
                safe = _safe_rgb(rendered)
                try:
                    encoded, width, height = _bounded_jpeg(safe)
                finally:
                    safe.close()
                    rendered.close()
            finally:
                bitmap.close()
                page.close()
            total += len(encoded)
            if total > MAX_NORMALIZED_TOTAL_BYTES:
                raise ValueError("normalized_total")
            name = f"page-{index:04d}.jpg"
            _write_private(output_path / name, encoded)
            manifest.append(
                {
                    "file": name,
                    "width": width,
                    "height": height,
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
    finally:
        document.close()
    _write_private(
        output_path / "manifest.json",
        json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
    )


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    _set_resource_limits()
    _block_network()
    input_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    try:
        if input_path.parent != output_path.parent or not output_path.is_dir():
            return 2
        _reader, dimensions = _inspect_pdf(input_path)
        _render(input_path, output_path, dimensions)
    except BaseException:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
