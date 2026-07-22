"""Compatibility entrypoint; the domain-neutral worker lives in safe_media."""

from .safe_media.pdf_worker import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
