from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_INPUT_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_SOURCE_DIMENSION = 20_000
MAX_IMAGE_DISPLAY_DIMENSION = 1600
MAX_IMAGE_OUTPUT_BYTES = 768 * 1024

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MIME_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

# Pillow checks this before allocating decoded pixel buffers. Generated canvases
# are materially smaller, so the process-wide ceiling is safe for all domains.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class SafeImageError(ValueError):
    """A stable, non-sensitive image validation failure code."""


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    sha256: str


def normalize_image(data: bytes, *, declared_mime: str | None) -> NormalizedImage:
    """Decode once, reject ambiguous media, strip metadata, and emit bounded JPEG."""

    if not data or len(data) > MAX_IMAGE_INPUT_BYTES:
        raise SafeImageError("invalid_input_size")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                actual_format = probe.format
                width, height = probe.size
                animated = (
                    bool(getattr(probe, "is_animated", False))
                    or int(getattr(probe, "n_frames", 1)) != 1
                )
                if actual_format not in ALLOWED_FORMATS:
                    raise SafeImageError("unsupported_format")
                if declared_mime is not None and MIME_FORMATS.get(declared_mime) != actual_format:
                    raise SafeImageError("mime_mismatch")
                if animated:
                    raise SafeImageError("animated")
                if (
                    width <= 0
                    or height <= 0
                    or width > MAX_IMAGE_SOURCE_DIMENSION
                    or height > MAX_IMAGE_SOURCE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise SafeImageError("too_many_pixels")
                probe.verify()

            with Image.open(BytesIO(data)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                try:
                    safe = safe_rgb(oriented)
                    try:
                        safe.thumbnail(
                            (MAX_IMAGE_DISPLAY_DIMENSION, MAX_IMAGE_DISPLAY_DIMENSION),
                            Image.Resampling.LANCZOS,
                        )
                        encoded, final_width, final_height = bounded_jpeg(safe)
                    finally:
                        safe.close()
                finally:
                    if oriented is not source:
                        oriented.close()
    except SafeImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise SafeImageError("decompression_bomb") from None
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError):
        raise SafeImageError("corrupt_image") from None
    return NormalizedImage(
        image_bytes=encoded,
        mime_type="image/jpeg",
        width=final_width,
        height=final_height,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def safe_rgb(image: Image.Image) -> Image.Image:
    """Flatten transparency without retaining source metadata."""

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        safe = Image.new("RGB", rgba.size, "white")
        safe.paste(rgba, mask=rgba.getchannel("A"))
        rgba.close()
        return safe
    return image.convert("RGB")


def bounded_jpeg(image: Image.Image) -> tuple[bytes, int, int]:
    """Encode deterministic metadata-free JPEG within the shared output quota."""

    working = image
    while True:
        for quality in (85, 75, 65, 55):
            output = BytesIO()
            working.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=False,
                progressive=False,
                subsampling=2,
            )
            encoded = output.getvalue()
            output.close()
            if len(encoded) <= MAX_IMAGE_OUTPUT_BYTES:
                final_size = working.size
                if working is not image:
                    working.close()
                return encoded, final_size[0], final_size[1]
        width, height = working.size
        if max(width, height) <= 480:
            if working is not image:
                working.close()
            raise SafeImageError("normalized_too_large")
        resized = working.resize(
            (max(int(width * 0.8), 1), max(int(height * 0.8), 1)),
            Image.Resampling.LANCZOS,
        )
        if working is not image:
            working.close()
        working = resized
