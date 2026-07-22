from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from time import monotonic

from sqlalchemy import select, update

from .db import Database
from .models import User, VisionItem, VisionItemImage
from .safe_media import images as safe_images

MIME_FORMATS = safe_images.MIME_FORMATS
MAX_IMAGE_INPUT_BYTES = safe_images.MAX_IMAGE_INPUT_BYTES
MAX_IMAGE_PIXELS = safe_images.MAX_IMAGE_PIXELS
MAX_IMAGE_SOURCE_DIMENSION = safe_images.MAX_IMAGE_SOURCE_DIMENSION
MAX_IMAGE_DISPLAY_DIMENSION = safe_images.MAX_IMAGE_DISPLAY_DIMENSION
MAX_IMAGE_OUTPUT_BYTES = safe_images.MAX_IMAGE_OUTPUT_BYTES
NormalizedVisionImage = safe_images.NormalizedImage
VisionImageError = safe_images.SafeImageError
normalize_vision_image = safe_images.normalize_image
_bounded_jpeg = safe_images.bounded_jpeg
_safe_rgb = safe_images.safe_rgb

IMAGE_UPLOAD_TTL_SECONDS = 10 * 60
MAX_PENDING_IMAGE_SESSIONS = 32
MAX_PENDING_IMAGE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TelegramImageMetadata:
    source: str
    file_size: int | None
    mime_type: str | None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class VisionImageMutation:
    status: str
    image: VisionItemImage | None = None


@dataclass(frozen=True, slots=True)
class VisionImageCapability:
    token: str
    owner_id: int
    chat_id: int
    item_id: int
    mode: str
    expected_version: int | None
    image: NormalizedVisionImage | None = None


@dataclass(slots=True)
class _VisionImageSession:
    owner_id: int
    chat_id: int
    item_id: int
    mode: str
    expected_version: int | None
    expires_at: float
    stage: str
    image: NormalizedVisionImage | None = None


class VisionImageSessionStore:
    """Bounded process-local capabilities; restarts make unfinished flows stale."""

    def __init__(
        self,
        *,
        ttl_seconds: int = IMAGE_UPLOAD_TTL_SECONDS,
        max_sessions: int = MAX_PENDING_IMAGE_SESSIONS,
        max_pending_bytes: int = MAX_PENDING_IMAGE_BYTES,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_pending_bytes = max_pending_bytes
        self._sessions: dict[str, _VisionImageSession] = {}
        self._lock = asyncio.Lock()

    async def issue_upload(
        self,
        owner_id: int,
        chat_id: int,
        item_id: int,
        *,
        mode: str,
        expected_version: int | None,
    ) -> str | None:
        if mode not in {"add", "replace"}:
            return None
        return await self._issue(
            owner_id,
            chat_id,
            item_id,
            mode=mode,
            expected_version=expected_version,
            stage="awaiting_upload",
        )

    async def issue_delete(
        self,
        owner_id: int,
        chat_id: int,
        item_id: int,
        *,
        expected_version: int,
    ) -> str | None:
        return await self._issue(
            owner_id,
            chat_id,
            item_id,
            mode="delete",
            expected_version=expected_version,
            stage="delete_confirm",
        )

    async def _issue(
        self,
        owner_id: int,
        chat_id: int,
        item_id: int,
        *,
        mode: str,
        expected_version: int | None,
        stage: str,
    ) -> str | None:
        async with self._lock:
            self._prune()
            if any(session.owner_id == owner_id for session in self._sessions.values()):
                return None
            while len(self._sessions) >= self.max_sessions:
                self._sessions.pop(next(iter(self._sessions)), None)
            token = secrets.token_urlsafe(9)
            while token in self._sessions:
                token = secrets.token_urlsafe(9)
            self._sessions[token] = _VisionImageSession(
                owner_id=owner_id,
                chat_id=chat_id,
                item_id=item_id,
                mode=mode,
                expected_version=expected_version,
                expires_at=monotonic() + self.ttl_seconds,
                stage=stage,
            )
            return token

    async def has_upload(self, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            self._prune()
            return any(
                session.owner_id == owner_id
                and session.chat_id == chat_id
                and session.stage in {"awaiting_upload", "processing"}
                for session in self._sessions.values()
            )

    async def has_active(self, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            self._prune()
            return any(
                session.owner_id == owner_id and session.chat_id == chat_id
                for session in self._sessions.values()
            )

    async def cancel_active(self, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            self._prune()
            tokens = [
                token
                for token, session in self._sessions.items()
                if session.owner_id == owner_id and session.chat_id == chat_id
            ]
            for token in tokens:
                self._sessions.pop(token, None)
            return bool(tokens)

    async def claim_upload(self, owner_id: int, chat_id: int) -> VisionImageCapability | None:
        async with self._lock:
            self._prune()
            for token, session in self._sessions.items():
                if (
                    session.owner_id == owner_id
                    and session.chat_id == chat_id
                    and session.stage == "awaiting_upload"
                ):
                    session.stage = "processing"
                    return self._snapshot(token, session)
            return None

    async def retry_upload(self, token: str, owner_id: int, chat_id: int) -> None:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is not None and session.stage == "processing":
                session.stage = "awaiting_upload"

    async def attach_preview(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
        image: NormalizedVisionImage,
    ) -> bool:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is None or session.stage != "processing":
                return False
            retained = sum(
                len(value.image.image_bytes)
                for value in self._sessions.values()
                if value.image is not None
            )
            if retained + len(image.image_bytes) > self.max_pending_bytes:
                self._sessions.pop(token, None)
                return False
            session.image = image
            session.stage = "preview"
            session.expires_at = monotonic() + self.ttl_seconds
            return True

    async def claim_confirm(
        self, token: str, owner_id: int, chat_id: int
    ) -> VisionImageCapability | None:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is None or session.stage != "preview" or session.image is None:
                return None
            snapshot = self._snapshot(token, session)
            self._sessions.pop(token, None)
            return snapshot

    async def claim_delete(
        self, token: str, owner_id: int, chat_id: int
    ) -> VisionImageCapability | None:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is None or session.stage != "delete_confirm":
                return None
            snapshot = self._snapshot(token, session)
            self._sessions.pop(token, None)
            return snapshot

    async def cancel(self, token: str, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is None:
                return False
            self._sessions.pop(token, None)
            return True

    def _owned(self, token: str, owner_id: int, chat_id: int) -> _VisionImageSession | None:
        self._prune()
        session = self._sessions.get(token)
        if session is None or session.owner_id != owner_id or session.chat_id != chat_id:
            return None
        return session

    @staticmethod
    def _snapshot(token: str, session: _VisionImageSession) -> VisionImageCapability:
        return VisionImageCapability(
            token=token,
            owner_id=session.owner_id,
            chat_id=session.chat_id,
            item_id=session.item_id,
            mode=session.mode,
            expected_version=session.expected_version,
            image=session.image,
        )

    def _prune(self) -> None:
        now = monotonic()
        for token in [
            token for token, session in self._sessions.items() if session.expires_at <= now
        ]:
            self._sessions.pop(token, None)


class VisionImageService:
    """Short owner-locked transactions for normalized image BLOBs."""

    def __init__(self, db: Database):
        self.db = db

    async def get(self, owner_id: int, item_id: int) -> VisionItemImage | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(VisionItemImage).where(
                    VisionItemImage.owner_id == owner_id,
                    VisionItemImage.vision_item_id == item_id,
                )
            )

    async def save(
        self,
        owner_id: int,
        item_id: int,
        *,
        expected_version: int | None,
        normalized: NormalizedVisionImage,
    ) -> VisionImageMutation:
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return VisionImageMutation("stale")
            item = await session.scalar(
                select(VisionItem).where(
                    VisionItem.id == item_id,
                    VisionItem.owner_id == owner_id,
                )
            )
            if item is None:
                return VisionImageMutation("stale")
            image = await session.scalar(
                select(VisionItemImage).where(
                    VisionItemImage.owner_id == owner_id,
                    VisionItemImage.vision_item_id == item_id,
                )
            )
            if expected_version is None:
                if image is not None:
                    return VisionImageMutation(
                        "existing" if image.sha256 == normalized.sha256 else "stale",
                        image,
                    )
                image = VisionItemImage(
                    owner_id=owner_id,
                    vision_item_id=item_id,
                    image_bytes=normalized.image_bytes,
                    mime_type=normalized.mime_type,
                    width=normalized.width,
                    height=normalized.height,
                    sha256=normalized.sha256,
                    version=1,
                )
                session.add(image)
                await session.flush()
                return VisionImageMutation("created", image)
            if image is None or image.version != expected_version:
                return VisionImageMutation("stale")
            if image.sha256 == normalized.sha256:
                return VisionImageMutation("existing", image)
            image.image_bytes = normalized.image_bytes
            image.mime_type = normalized.mime_type
            image.width = normalized.width
            image.height = normalized.height
            image.sha256 = normalized.sha256
            image.version += 1
            return VisionImageMutation("replaced", image)

    async def delete(
        self,
        owner_id: int,
        item_id: int,
        *,
        expected_version: int,
    ) -> VisionImageMutation:
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return VisionImageMutation("stale")
            image = await session.scalar(
                select(VisionItemImage).where(
                    VisionItemImage.owner_id == owner_id,
                    VisionItemImage.vision_item_id == item_id,
                    VisionItemImage.version == expected_version,
                    VisionItemImage.vision_item_id.in_(
                        select(VisionItem.id).where(VisionItem.owner_id == owner_id)
                    ),
                )
            )
            if image is None:
                return VisionImageMutation("stale")
            await session.delete(image)
            return VisionImageMutation("deleted", image)

    @staticmethod
    async def _lock_owner(session: object, owner_id: int) -> bool:
        result = await session.execute(
            update(User)
            .where(User.id == owner_id)
            .values(updated_at=User.updated_at)
            .returning(User.id)
        )
        return result.scalar_one_or_none() is not None


def validate_telegram_metadata(metadata: TelegramImageMetadata) -> None:
    if metadata.source not in {"photo", "document"}:
        raise VisionImageError("unsupported_source")
    if metadata.file_size is None or metadata.file_size <= 0:
        raise VisionImageError("missing_size")
    if metadata.file_size > MAX_IMAGE_INPUT_BYTES:
        raise VisionImageError("input_too_large")
    if metadata.source == "document" and metadata.mime_type not in MIME_FORMATS:
        raise VisionImageError("unsupported_mime")
    if metadata.source == "photo" and metadata.mime_type not in {None, "image/jpeg"}:
        raise VisionImageError("unsupported_mime")
    if metadata.width is not None or metadata.height is not None:
        if not metadata.width or not metadata.height:
            raise VisionImageError("invalid_dimensions")
        if (
            metadata.width > MAX_IMAGE_SOURCE_DIMENSION
            or metadata.height > MAX_IMAGE_SOURCE_DIMENSION
            or metadata.width * metadata.height > MAX_IMAGE_PIXELS
        ):
            raise VisionImageError("too_many_pixels")
