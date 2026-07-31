from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from time import monotonic

from sqlalchemy import func, select, update

from .db import Database
from .models import User, VisionReference
from .vision_images import NormalizedVisionImage

REFERENCE_KINDS = {
    "self": "Я / моя внешность",
    "person": "Другой человек",
    "place": "Место",
    "object": "Предмет",
    "style": "Стиль / атмосфера",
}
MAX_VISION_REFERENCES = 12
MAX_GENERATION_REFERENCES = 4
REFERENCE_SESSION_TTL_SECONDS = 10 * 60
MAX_REFERENCE_SESSIONS = 32
MAX_PENDING_REFERENCE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VisionReferenceMutation:
    status: str
    reference: VisionReference | None = None


@dataclass(frozen=True, slots=True)
class VisionReferenceCapability:
    token: str
    owner_id: int
    chat_id: int
    stage: str
    mode: str
    kind: str | None = None
    name: str | None = None
    image: NormalizedVisionImage | None = None
    reference_id: int | None = None
    expected_version: int | None = None


@dataclass(slots=True)
class _ReferenceSession:
    owner_id: int
    chat_id: int
    stage: str
    mode: str
    expires_at: float
    kind: str | None = None
    name: str | None = None
    image: NormalizedVisionImage | None = None
    reference_id: int | None = None
    expected_version: int | None = None


class VisionReferenceSessionStore:
    """Owner/chat-bound capabilities for private reference creation and deletion."""

    def __init__(
        self,
        *,
        ttl_seconds: int = REFERENCE_SESSION_TTL_SECONDS,
        max_sessions: int = MAX_REFERENCE_SESSIONS,
        max_pending_bytes: int = MAX_PENDING_REFERENCE_BYTES,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_pending_bytes = max_pending_bytes
        self._sessions: dict[str, _ReferenceSession] = {}
        self._lock = asyncio.Lock()

    async def issue_create(self, owner_id: int, chat_id: int) -> str | None:
        return await self._issue(owner_id, chat_id, stage="kind", mode="create")

    async def issue_replace(
        self,
        owner_id: int,
        chat_id: int,
        reference_id: int,
        *,
        expected_version: int,
        kind: str,
        name: str,
    ) -> str | None:
        token = await self._issue(
            owner_id,
            chat_id,
            stage="awaiting_upload",
            mode="replace",
            reference_id=reference_id,
            expected_version=expected_version,
        )
        if token is None:
            return None
        async with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            session.kind = kind
            session.name = name
        return token

    async def issue_rename(
        self,
        owner_id: int,
        chat_id: int,
        reference_id: int,
        *,
        expected_version: int,
        kind: str,
        name: str,
    ) -> str | None:
        token = await self._issue(
            owner_id,
            chat_id,
            stage="rename_name",
            mode="rename",
            reference_id=reference_id,
            expected_version=expected_version,
        )
        if token is None:
            return None
        async with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            session.kind = kind
            session.name = name
        return token

    async def issue_delete(
        self,
        owner_id: int,
        chat_id: int,
        reference_id: int,
        *,
        expected_version: int,
    ) -> str | None:
        return await self._issue(
            owner_id,
            chat_id,
            stage="delete_confirm",
            mode="delete",
            reference_id=reference_id,
            expected_version=expected_version,
        )

    async def _issue(
        self,
        owner_id: int,
        chat_id: int,
        *,
        stage: str,
        mode: str,
        reference_id: int | None = None,
        expected_version: int | None = None,
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
            self._sessions[token] = _ReferenceSession(
                owner_id=owner_id,
                chat_id=chat_id,
                stage=stage,
                mode=mode,
                expires_at=monotonic() + self.ttl_seconds,
                reference_id=reference_id,
                expected_version=expected_version,
            )
            return token

    async def choose_kind(
        self, token: str, owner_id: int, chat_id: int, kind: str
    ) -> VisionReferenceCapability | None:
        if kind not in REFERENCE_KINDS:
            return None
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is None or session.stage != "kind":
                return None
            session.kind = kind
            session.stage = "name"
            return self._snapshot(token, session)

    async def set_name(
        self, token: str, owner_id: int, chat_id: int, name: str | None
    ) -> VisionReferenceCapability | None:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is None or session.stage != "name" or session.kind is None:
                return None
            cleaned = " ".join((name or "").split())
            if not cleaned:
                cleaned = REFERENCE_KINDS[session.kind]
            if len(cleaned) > 60:
                return None
            session.name = cleaned
            session.stage = "awaiting_upload"
            session.expires_at = monotonic() + self.ttl_seconds
            return self._snapshot(token, session)

    async def awaiting_name(self, owner_id: int, chat_id: int) -> VisionReferenceCapability | None:
        async with self._lock:
            self._prune()
            for token, session in self._sessions.items():
                if (
                    session.owner_id == owner_id
                    and session.chat_id == chat_id
                    and session.stage == "name"
                ):
                    return self._snapshot(token, session)
            return None

    async def awaiting_rename(
        self, owner_id: int, chat_id: int
    ) -> VisionReferenceCapability | None:
        async with self._lock:
            self._prune()
            for token, session in self._sessions.items():
                if (
                    session.owner_id == owner_id
                    and session.chat_id == chat_id
                    and session.stage == "rename_name"
                ):
                    return self._snapshot(token, session)
            return None

    async def claim_rename(
        self, token: str, owner_id: int, chat_id: int, name: str
    ) -> VisionReferenceCapability | None:
        cleaned = " ".join(name.split())
        if not cleaned or len(cleaned) > 60:
            return None
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is None or session.stage != "rename_name":
                return None
            session.name = cleaned
            snapshot = self._snapshot(token, session)
            self._sessions.pop(token, None)
            return snapshot

    async def has_upload(self, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            self._prune()
            return any(
                session.owner_id == owner_id
                and session.chat_id == chat_id
                and session.stage in {"awaiting_upload", "processing_upload"}
                for session in self._sessions.values()
            )

    async def has_active(self, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            self._prune()
            return any(
                session.owner_id == owner_id and session.chat_id == chat_id
                for session in self._sessions.values()
            )

    async def claim_upload(self, owner_id: int, chat_id: int) -> VisionReferenceCapability | None:
        async with self._lock:
            self._prune()
            for token, session in self._sessions.items():
                if (
                    session.owner_id == owner_id
                    and session.chat_id == chat_id
                    and session.stage == "awaiting_upload"
                ):
                    session.stage = "processing_upload"
                    return self._snapshot(token, session)
            return None

    async def retry_upload(self, token: str, owner_id: int, chat_id: int) -> None:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if session is not None and session.stage == "processing_upload":
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
            if session is None or session.stage != "processing_upload":
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
    ) -> VisionReferenceCapability | None:
        async with self._lock:
            session = self._owned(token, owner_id, chat_id)
            if (
                session is None
                or session.stage != "preview"
                or session.image is None
                or session.kind is None
                or session.name is None
            ):
                return None
            snapshot = self._snapshot(token, session)
            self._sessions.pop(token, None)
            return snapshot

    async def claim_delete(
        self, token: str, owner_id: int, chat_id: int
    ) -> VisionReferenceCapability | None:
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

    def _owned(self, token: str, owner_id: int, chat_id: int) -> _ReferenceSession | None:
        self._prune()
        session = self._sessions.get(token)
        if session is None or session.owner_id != owner_id or session.chat_id != chat_id:
            return None
        return session

    @staticmethod
    def _snapshot(token: str, session: _ReferenceSession) -> VisionReferenceCapability:
        return VisionReferenceCapability(
            token=token,
            owner_id=session.owner_id,
            chat_id=session.chat_id,
            stage=session.stage,
            mode=session.mode,
            kind=session.kind,
            name=session.name,
            image=session.image,
            reference_id=session.reference_id,
            expected_version=session.expected_version,
        )

    def _prune(self) -> None:
        now = monotonic()
        for token in [
            token for token, session in self._sessions.items() if session.expires_at <= now
        ]:
            self._sessions.pop(token, None)


class VisionReferenceService:
    """Persistent, owner-scoped normalized reference image library."""

    def __init__(self, db: Database):
        self.db = db

    async def count(self, owner_id: int) -> int:
        async with self.db.sessions() as session:
            return int(
                await session.scalar(
                    select(func.count(VisionReference.id)).where(
                        VisionReference.owner_id == owner_id
                    )
                )
                or 0
            )

    async def list(self, owner_id: int) -> list[VisionReference]:
        async with self.db.sessions() as session:
            return list(
                (
                    await session.scalars(
                        select(VisionReference)
                        .where(VisionReference.owner_id == owner_id)
                        .order_by(VisionReference.created_at, VisionReference.id)
                    )
                ).all()
            )

    async def get(self, owner_id: int, reference_id: int) -> VisionReference | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(VisionReference).where(
                    VisionReference.owner_id == owner_id,
                    VisionReference.id == reference_id,
                )
            )

    async def get_many(
        self, owner_id: int, reference_ids: tuple[int, ...]
    ) -> list[VisionReference]:
        if not reference_ids:
            return []
        unique_ids = tuple(dict.fromkeys(reference_ids))[:MAX_GENERATION_REFERENCES]
        async with self.db.sessions() as session:
            rows = list(
                (
                    await session.scalars(
                        select(VisionReference).where(
                            VisionReference.owner_id == owner_id,
                            VisionReference.id.in_(unique_ids),
                        )
                    )
                ).all()
            )
        by_id = {row.id: row for row in rows}
        return [by_id[reference_id] for reference_id in unique_ids if reference_id in by_id]

    async def save(
        self,
        owner_id: int,
        *,
        kind: str,
        name: str,
        normalized: NormalizedVisionImage,
    ) -> VisionReferenceMutation:
        cleaned_name = " ".join(name.split())
        if kind not in REFERENCE_KINDS or not cleaned_name or len(cleaned_name) > 60:
            return VisionReferenceMutation("invalid")
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return VisionReferenceMutation("stale")
            existing = await session.scalar(
                select(VisionReference).where(
                    VisionReference.owner_id == owner_id,
                    VisionReference.sha256 == normalized.sha256,
                )
            )
            if existing is not None:
                return VisionReferenceMutation("existing", existing)
            total = int(
                await session.scalar(
                    select(func.count(VisionReference.id)).where(
                        VisionReference.owner_id == owner_id
                    )
                )
                or 0
            )
            if total >= MAX_VISION_REFERENCES:
                return VisionReferenceMutation("limit")
            reference = VisionReference(
                owner_id=owner_id,
                kind=kind,
                name=cleaned_name,
                image_bytes=normalized.image_bytes,
                mime_type=normalized.mime_type,
                width=normalized.width,
                height=normalized.height,
                sha256=normalized.sha256,
                version=1,
            )
            session.add(reference)
            await session.flush()
            return VisionReferenceMutation("created", reference)

    async def delete(
        self, owner_id: int, reference_id: int, *, expected_version: int
    ) -> VisionReferenceMutation:
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return VisionReferenceMutation("stale")
            reference = await session.scalar(
                select(VisionReference).where(
                    VisionReference.owner_id == owner_id,
                    VisionReference.id == reference_id,
                    VisionReference.version == expected_version,
                )
            )
            if reference is None:
                return VisionReferenceMutation("stale")
            await session.delete(reference)
            return VisionReferenceMutation("deleted", reference)

    async def rename(
        self,
        owner_id: int,
        reference_id: int,
        *,
        expected_version: int,
        name: str,
    ) -> VisionReferenceMutation:
        cleaned = " ".join(name.split())
        if not cleaned or len(cleaned) > 60:
            return VisionReferenceMutation("invalid")
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return VisionReferenceMutation("stale")
            reference = await session.scalar(
                select(VisionReference).where(
                    VisionReference.owner_id == owner_id,
                    VisionReference.id == reference_id,
                    VisionReference.version == expected_version,
                )
            )
            if reference is None:
                return VisionReferenceMutation("stale")
            if reference.name == cleaned:
                return VisionReferenceMutation("existing", reference)
            reference.name = cleaned
            reference.version += 1
            return VisionReferenceMutation("renamed", reference)

    async def replace(
        self,
        owner_id: int,
        reference_id: int,
        *,
        expected_version: int,
        normalized: NormalizedVisionImage,
    ) -> VisionReferenceMutation:
        async with self.db.session() as session:
            if not await self._lock_owner(session, owner_id):
                return VisionReferenceMutation("stale")
            duplicate = await session.scalar(
                select(VisionReference).where(
                    VisionReference.owner_id == owner_id,
                    VisionReference.sha256 == normalized.sha256,
                    VisionReference.id != reference_id,
                )
            )
            if duplicate is not None:
                return VisionReferenceMutation("duplicate", duplicate)
            reference = await session.scalar(
                select(VisionReference).where(
                    VisionReference.owner_id == owner_id,
                    VisionReference.id == reference_id,
                    VisionReference.version == expected_version,
                )
            )
            if reference is None:
                return VisionReferenceMutation("stale")
            if reference.sha256 == normalized.sha256:
                return VisionReferenceMutation("existing", reference)
            reference.image_bytes = normalized.image_bytes
            reference.mime_type = normalized.mime_type
            reference.width = normalized.width
            reference.height = normalized.height
            reference.sha256 = normalized.sha256
            reference.version += 1
            return VisionReferenceMutation("replaced", reference)

    @staticmethod
    async def _lock_owner(session: object, owner_id: int) -> bool:
        result = await session.execute(
            update(User)
            .where(User.id == owner_id)
            .values(updated_at=User.updated_at)
            .returning(User.id)
        )
        return result.scalar_one_or_none() is not None
