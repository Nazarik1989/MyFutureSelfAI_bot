from __future__ import annotations

import asyncio
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from time import monotonic

from PIL import Image, ImageDraw, ImageFont

from .vision import CATEGORY_META

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1350
MAX_CARDS_PER_PAGE = 5
MAX_PAGES = 6
MAX_RENDER_ITEMS = MAX_CARDS_PER_PAGE * MAX_PAGES
MAX_PNG_BYTES = 5 * 1024 * 1024
MAX_RENDER_TEXT_CHARS = 1200
RENDER_SESSION_TTL_SECONDS = 10 * 60

BACKGROUND = "#F4F1EA"
PAPER = "#FFFDFC"
INK = "#20242A"
MUTED = "#5D6670"
FOOTER = "#4C5B52"
CATEGORY_COLORS = {
    "health_energy": "#3F7555",
    "relationships_family": "#A94D62",
    "work_purpose": "#4E6F9E",
    "money": "#806325",
    "home": "#8C583A",
    "travel": "#397E91",
    "growth_creativity": "#8266A3",
    "other": "#68717A",
}


class VisionRenderError(RuntimeError):
    """Safe renderer failure without user content or filesystem details."""


class VisionRenderBusy(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VisionRenderItem:
    category: str
    wish_text: str
    target_date: date | None = None
    sort_id: int = 0


@dataclass(frozen=True, slots=True)
class LayoutBox:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class RenderedPage:
    png: bytes
    card_boxes: tuple[LayoutBox, ...]


@dataclass(frozen=True, slots=True)
class RenderedBoard:
    pages: tuple[RenderedPage, ...]
    included_count: int
    omitted_count: int
    category: str | None
    created_on: date


@dataclass(slots=True)
class _RenderSession:
    owner_id: int
    chat_id: int
    allowed_categories: frozenset[str]
    expires_at: float
    selection: str | None = None
    selection_claimed: bool = False
    download_claimed: bool = False


class VisionRenderSessionStore:
    """Small bounded, process-local capability store for render callbacks."""

    def __init__(self, *, ttl_seconds: int = RENDER_SESSION_TTL_SECONDS, max_sessions: int = 256):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, _RenderSession] = {}
        self._lock = asyncio.Lock()

    async def issue(
        self,
        owner_id: int,
        chat_id: int,
        allowed_categories: set[str],
    ) -> str:
        async with self._lock:
            self._prune()
            while len(self._sessions) >= self.max_sessions:
                oldest = next(iter(self._sessions))
                self._sessions.pop(oldest, None)
            token = secrets.token_urlsafe(9)
            while token in self._sessions:
                token = secrets.token_urlsafe(9)
            self._sessions[token] = _RenderSession(
                owner_id=owner_id,
                chat_id=chat_id,
                allowed_categories=frozenset(allowed_categories),
                expires_at=monotonic() + self.ttl_seconds,
            )
            return token

    async def claim_selection(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
        category: str,
    ) -> str | None:
        async with self._lock:
            session = self._get(token, owner_id, chat_id)
            if session is None or session.selection_claimed:
                return None
            if category != "all" and category not in session.allowed_categories:
                return None
            session.selection = category
            session.selection_claimed = True
            return category

    async def claim_download(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
    ) -> str | None:
        async with self._lock:
            session = self._get(token, owner_id, chat_id)
            if (
                session is None
                or session.selection is None
                or not session.selection_claimed
                or session.download_claimed
            ):
                return None
            session.download_claimed = True
            return session.selection

    async def cancel(self, token: str, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            session = self._get(token, owner_id, chat_id)
            if session is None or session.selection_claimed:
                return False
            self._sessions.pop(token, None)
            return True

    def _get(self, token: str, owner_id: int, chat_id: int) -> _RenderSession | None:
        self._prune()
        session = self._sessions.get(token)
        if session is None or session.owner_id != owner_id or session.chat_id != chat_id:
            return None
        return session

    def _prune(self) -> None:
        now = monotonic()
        expired = [token for token, session in self._sessions.items() if session.expires_at <= now]
        for token in expired:
            self._sessions.pop(token, None)


class VisionRenderLimiter:
    """Immediate per-owner and global admission control for CPU-bound renders."""

    def __init__(self, *, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self._owners: set[int] = set()
        self._lock = asyncio.Lock()

    async def acquire(self, owner_id: int) -> bool:
        async with self._lock:
            if owner_id in self._owners or len(self._owners) >= self.max_concurrent:
                return False
            self._owners.add(owner_id)
            return True

    async def release(self, owner_id: int) -> None:
        async with self._lock:
            self._owners.discard(owner_id)


class VisionBoardRenderer:
    """Deterministic in-memory 1080×1350 PNG renderer."""

    def __init__(
        self,
        *,
        font_path: str | Path | None = None,
        bold_font_path: str | Path | None = None,
    ):
        self.font_path = self._resolve_font(font_path, bold=False)
        self.bold_font_path = self._resolve_font(bold_font_path, bold=True)

    def render(
        self,
        items: list[VisionRenderItem],
        *,
        created_on: date,
        category: str | None,
        total_count: int | None = None,
    ) -> RenderedBoard:
        if not items:
            raise VisionRenderError("empty")
        ordered = sorted(
            items,
            key=lambda item: (
                self._category_rank(item.category),
                item.target_date is None,
                item.target_date or date.max,
                item.sort_id,
            ),
        )
        limited = ordered[:MAX_RENDER_ITEMS]
        omitted = max((total_count if total_count is not None else len(ordered)) - len(limited), 0)
        pages: list[RenderedPage] = []
        page_count = (len(limited) + MAX_CARDS_PER_PAGE - 1) // MAX_CARDS_PER_PAGE
        for page_index in range(page_count):
            start = page_index * MAX_CARDS_PER_PAGE
            pages.append(
                self._render_page(
                    limited[start : start + MAX_CARDS_PER_PAGE],
                    created_on=created_on,
                    page_index=page_index,
                    page_count=page_count,
                )
            )
        return RenderedBoard(
            pages=tuple(pages),
            included_count=len(limited),
            omitted_count=omitted,
            category=category,
            created_on=created_on,
        )

    def _render_page(
        self,
        items: list[VisionRenderItem],
        *,
        created_on: date,
        page_index: int,
        page_count: int,
    ) -> RenderedPage:
        image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(image)
        title_font = self._font(54, bold=True)
        meta_font = self._font(24)
        category_font = self._font(23, bold=True)
        wish_font = self._font(31)
        date_font = self._font(21)
        footer_font = self._font(22)

        draw.text((64, 54), "Моя карта желаний", font=title_font, fill=INK)
        page_label = f" · страница {page_index + 1}/{page_count}" if page_count > 1 else ""
        draw.text(
            (66, 130),
            f"Создано {created_on.strftime('%d.%m.%Y')}{page_label}",
            font=meta_font,
            fill=MUTED,
        )

        content_top = 205
        card_height = 184
        card_gap = 16
        card_left = 64
        card_right = CANVAS_WIDTH - 64
        boxes: list[LayoutBox] = []
        for index, item in enumerate(items):
            top = content_top + index * (card_height + card_gap)
            bottom = top + card_height
            box = LayoutBox(card_left, top, card_right, bottom)
            boxes.append(box)
            accent = CATEGORY_COLORS[item.category]
            draw.rounded_rectangle(
                (box.left, box.top, box.right, box.bottom),
                radius=24,
                fill=PAPER,
            )
            draw.rounded_rectangle(
                (box.left, box.top, box.left + 14, box.bottom),
                radius=7,
                fill=accent,
            )
            _emoji, label = CATEGORY_META[item.category]
            draw.text(
                (box.left + 36, box.top + 18),
                label,
                font=category_font,
                fill=accent,
            )
            wish = clean_render_text(item.wish_text)
            lines = fit_text_lines(
                draw,
                wish,
                wish_font,
                max_width=box.right - box.left - 72,
                max_lines=2 if item.target_date is not None else 3,
            )
            draw.multiline_text(
                (box.left + 36, box.top + 53),
                "\n".join(lines),
                font=wish_font,
                fill=INK,
                spacing=5,
            )
            if item.target_date is not None:
                date_text = f"Желаемая дата: {item.target_date.strftime('%d.%m.%Y')}"
                draw.text(
                    (box.left + 36, box.bottom - 32),
                    date_text,
                    font=date_font,
                    fill=MUTED,
                )

        draw.line((64, 1235, CANVAS_WIDTH - 64, 1235), fill="#D8D2C7", width=2)
        draw.text(
            (64, 1262),
            "Маленькие шаги помогают двигаться к важному.",
            font=footer_font,
            fill=FOOTER,
        )

        output = BytesIO()
        image.save(output, format="PNG", optimize=False, compress_level=6)
        image.close()
        png = output.getvalue()
        output.close()
        if len(png) > MAX_PNG_BYTES:
            raise VisionRenderError("too_large")
        return RenderedPage(png=png, card_boxes=tuple(boxes))

    def _font(self, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
        path = self.bold_font_path if bold else self.font_path
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError as exc:
            raise VisionRenderError("font_unavailable") from exc

    @staticmethod
    def _category_rank(category: str) -> int:
        try:
            return list(CATEGORY_META).index(category)
        except ValueError as exc:
            raise VisionRenderError("invalid_category") from exc

    @staticmethod
    def _resolve_font(path: str | Path | None, *, bold: bool) -> Path:
        if path is not None:
            candidate = Path(path)
            if candidate.is_file():
                return candidate
            raise VisionRenderError("font_unavailable")
        names = (
            (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
            )
            if bold
            else (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
            )
        )
        for name in names:
            candidate = Path(name)
            if candidate.is_file():
                return candidate
        raise VisionRenderError("font_unavailable")


def clean_render_text(value: str) -> str:
    cleaned: list[str] = []
    for character in value[:MAX_RENDER_TEXT_CHARS]:
        category = unicodedata.category(character)
        if character.isspace():
            cleaned.append(" ")
        elif category in {"Cc", "Cf", "Cs"}:
            continue
        else:
            cleaned.append(character)
    return " ".join("".join(cleaned).split()) or "Желание без текста"


def fit_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
    max_lines: int,
) -> tuple[str, ...]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        chunks = _split_long_word(draw, word, font, max_width)
        for chunk in chunks:
            candidate = f"{current} {chunk}".strip()
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = chunk
            else:
                current = candidate
    if current:
        lines.append(current)
    if not lines:
        lines = ["Желание без текста"]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _with_ellipsis(draw, lines[-1], font, max_width)
    return tuple(lines)


def _split_long_word(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    if draw.textlength(word, font=font) <= max_width:
        return [word]
    chunks: list[str] = []
    current = ""
    for character in word:
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            chunks.append(current)
            current = character
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _with_ellipsis(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    value = text.rstrip("…")
    while value and draw.textlength(value + "…", font=font) > max_width:
        value = value[:-1]
    return (value.rstrip() or "Желание") + "…"
