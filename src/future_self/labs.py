from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import time

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import selectinload

from .db import Database
from .lab_media import (
    LAB_UPLOAD_TTL_SECONDS,
    MAX_NORMALIZED_TOTAL_BYTES,
    MAX_PENDING_LAB_BYTES,
    MAX_PENDING_LAB_SESSIONS,
    NormalizedLabPage,
    ProcessedLabDocument,
)
from .models import LabDeleteConfirmation, LabDocument, LabDocumentPage

LAB_LIST_PAGE_SIZE = 6
DELETE_CONFIRM_TTL_MINUTES = 10


@dataclass(frozen=True, slots=True)
class LabDraftSnapshot:
    token: str
    owner_id: int
    chat_id: int
    stage: str
    title: str | None
    document_date: date | None
    source_type: str | None
    page_count: int
    first_page: bytes | None = None


@dataclass(frozen=True, slots=True)
class LabSaveCapability:
    token: str
    owner_id: int
    chat_id: int
    title: str
    document_date: date | None
    source_type: str
    pages: tuple[NormalizedLabPage, ...]


@dataclass(slots=True)
class _LabDraft:
    owner_id: int
    chat_id: int
    expires_at: float
    stage: str
    directory: Path
    title: str | None = None
    document_date: date | None = None
    source_type: str | None = None
    pages: tuple[tuple[Path, str, int, int, str], ...] = ()


class LabUploadSessionStore:
    """Bounded process-local capabilities backed by private, expiring page files."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        ttl_seconds: int = LAB_UPLOAD_TTL_SECONDS,
        max_sessions: int = MAX_PENDING_LAB_SESSIONS,
        max_pending_bytes: int = MAX_PENDING_LAB_BYTES,
    ):
        self.root = (root or Path(tempfile.gettempdir()) / "myfutureselfai-labs").resolve()
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_pending_bytes = max_pending_bytes
        self._sessions: dict[str, _LabDraft] = {}
        self._lock = asyncio.Lock()
        self._prepare_root()
        self.cleanup_startup()

    async def start(self, owner_id: int, chat_id: int) -> str | None:
        async with self._lock:
            self._purge_expired_locked()
            self._cancel_owner_locked(owner_id, chat_id)
            if len(self._sessions) >= self.max_sessions:
                return None
            token = secrets.token_hex(12)
            directory = (self.root / token).resolve()
            if directory.parent != self.root:
                return None
            directory.mkdir(mode=0o700)
            directory.chmod(0o700)
            self._sessions[token] = _LabDraft(
                owner_id=owner_id,
                chat_id=chat_id,
                expires_at=time() + self.ttl_seconds,
                stage="awaiting_file",
                directory=directory,
            )
            return token

    async def has_active(self, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            self._purge_expired_locked()
            return self._active_locked(owner_id, chat_id) is not None

    async def claim_upload(self, owner_id: int, chat_id: int) -> LabDraftSnapshot | None:
        async with self._lock:
            self._purge_expired_locked()
            found = self._active_locked(owner_id, chat_id)
            if found is None or found[1].stage != "awaiting_file":
                return None
            token, session = found
            session.stage = "processing"
            return self._snapshot(token, session)

    async def retry_upload(self, token: str, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            session = self._owned_locked(token, owner_id, chat_id)
            if session is None or session.stage != "processing":
                return False
            session.stage = "awaiting_file"
            return True

    async def attach(
        self,
        token: str,
        owner_id: int,
        chat_id: int,
        processed: ProcessedLabDocument,
        *,
        title: str,
    ) -> LabDraftSnapshot | None:
        async with self._lock:
            self._purge_expired_locked()
            session = self._owned_locked(token, owner_id, chat_id)
            if session is None or session.stage != "processing" or session.pages:
                return None
            pending = sum(
                self._size(item) for draft in self._sessions.values() for item in draft.pages
            )
            incoming = sum(len(page.image_bytes) for page in processed.pages)
            if incoming <= 0 or incoming > MAX_NORMALIZED_TOTAL_BYTES:
                return None
            if pending + incoming > self.max_pending_bytes:
                return None
            stored: list[tuple[Path, str, int, int, str]] = []
            try:
                for index, page in enumerate(processed.pages):
                    path = session.directory / f"page-{index:04d}.jpg"
                    self._write_private(path, page.image_bytes)
                    stored.append((path, page.mime_type, page.width, page.height, page.sha256))
            except Exception:
                for path, *_rest in stored:
                    path.unlink(missing_ok=True)
                raise
            session.pages = tuple(stored)
            session.source_type = processed.source_type
            session.title = sanitize_lab_title(title)
            session.stage = "ready"
            return self._snapshot(token, session, with_preview=True)

    async def begin_edit(
        self, token: str, owner_id: int, chat_id: int, field: str
    ) -> LabDraftSnapshot | None:
        if field not in {"title", "date"}:
            return None
        async with self._lock:
            self._purge_expired_locked()
            session = self._owned_locked(token, owner_id, chat_id)
            if session is None or session.stage != "ready":
                return None
            session.stage = f"edit_{field}"
            return self._snapshot(token, session)

    async def apply_title(self, owner_id: int, chat_id: int, title: str) -> LabDraftSnapshot | None:
        clean = sanitize_lab_title(title)
        async with self._lock:
            self._purge_expired_locked()
            found = self._active_locked(owner_id, chat_id)
            if found is None or found[1].stage != "edit_title":
                return None
            token, session = found
            session.title = clean
            session.stage = "ready"
            return self._snapshot(token, session, with_preview=True)

    async def apply_date(
        self, owner_id: int, chat_id: int, value: date | None
    ) -> LabDraftSnapshot | None:
        async with self._lock:
            self._purge_expired_locked()
            found = self._active_locked(owner_id, chat_id)
            if found is None or found[1].stage != "edit_date":
                return None
            token, session = found
            session.document_date = value
            session.stage = "ready"
            return self._snapshot(token, session, with_preview=True)

    async def active(self, owner_id: int, chat_id: int) -> LabDraftSnapshot | None:
        async with self._lock:
            self._purge_expired_locked()
            found = self._active_locked(owner_id, chat_id)
            return None if found is None else self._snapshot(found[0], found[1])

    async def claim_confirm(
        self, token: str, owner_id: int, chat_id: int
    ) -> LabSaveCapability | None:
        async with self._lock:
            self._purge_expired_locked()
            session = self._owned_locked(token, owner_id, chat_id)
            if (
                session is None
                or session.stage != "ready"
                or session.title is None
                or session.source_type is None
                or not session.pages
            ):
                return None
            session.stage = "saving"
            try:
                pages = tuple(
                    NormalizedLabPage(path.read_bytes(), mime_type, width, height, sha256)
                    for path, mime_type, width, height, sha256 in session.pages
                )
            except OSError:
                self._remove_locked(token)
                return None
            return LabSaveCapability(
                token,
                owner_id,
                chat_id,
                session.title,
                session.document_date,
                session.source_type,
                pages,
            )

    async def cancel(self, token: str, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            session = self._owned_locked(token, owner_id, chat_id)
            if session is None:
                return False
            self._remove_locked(token)
            return True

    async def cancel_active(self, owner_id: int, chat_id: int) -> bool:
        async with self._lock:
            found = self._active_locked(owner_id, chat_id)
            if found is None:
                return False
            self._remove_locked(found[0])
            return True

    async def finish(self, token: str, owner_id: int, chat_id: int) -> None:
        async with self._lock:
            if self._owned_locked(token, owner_id, chat_id) is not None:
                self._remove_locked(token)

    def cleanup_startup(self) -> None:
        for entry in self.root.iterdir():
            try:
                entry.lstat()
                # Process-local capabilities never survive a restart, so every
                # leftover entry is orphaned regardless of its age.
                self._remove_path(entry)
            except OSError:
                continue

    async def cleanup_expired(self) -> None:
        async with self._lock:
            self._purge_expired_locked()

    def _prepare_root(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise RuntimeError("Unsafe lab temporary directory")
        self.root.chmod(0o700)

    def _purge_expired_locked(self) -> None:
        expired = [
            token for token, session in self._sessions.items() if session.expires_at <= time()
        ]
        for token in expired:
            self._remove_locked(token)

    def _cancel_owner_locked(self, owner_id: int, chat_id: int) -> None:
        found = self._active_locked(owner_id, chat_id)
        if found is not None:
            self._remove_locked(found[0])

    def _active_locked(self, owner_id: int, chat_id: int) -> tuple[str, _LabDraft] | None:
        return next(
            (
                (token, session)
                for token, session in self._sessions.items()
                if session.owner_id == owner_id and session.chat_id == chat_id
            ),
            None,
        )

    def _owned_locked(self, token: str, owner_id: int, chat_id: int) -> _LabDraft | None:
        session = self._sessions.get(token)
        if session is None or session.owner_id != owner_id or session.chat_id != chat_id:
            return None
        return session

    def _remove_locked(self, token: str) -> None:
        session = self._sessions.pop(token, None)
        if session is not None:
            self._remove_path(session.directory)

    def _remove_path(self, path: Path) -> None:
        if path.absolute().parent.resolve() != self.root:
            return
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)

    @staticmethod
    def _write_private(path: Path, data: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)

    @staticmethod
    def _size(item: tuple[Path, str, int, int, str]) -> int:
        try:
            return item[0].stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _snapshot(
        token: str, session: _LabDraft, *, with_preview: bool = False
    ) -> LabDraftSnapshot:
        first_page = None
        if with_preview and session.pages:
            try:
                first_page = session.pages[0][0].read_bytes()
            except OSError:
                first_page = None
        return LabDraftSnapshot(
            token,
            session.owner_id,
            session.chat_id,
            session.stage,
            session.title,
            session.document_date,
            session.source_type,
            len(session.pages),
            first_page,
        )


class LabDocumentService:
    def __init__(self, db: Database):
        self.db = db

    async def create(
        self,
        owner_id: int,
        title: str,
        document_date: date | None,
        source_type: str,
        pages: tuple[NormalizedLabPage, ...],
    ) -> LabDocument:
        clean = sanitize_lab_title(title)
        if source_type not in {"image", "pdf"} or not pages:
            raise ValueError("invalid_document")
        async with self.db.session() as session:
            document = LabDocument(
                owner_id=owner_id,
                title=clean,
                document_date=document_date,
                source_type=source_type,
                page_count=len(pages),
                status="saved",
                version=1,
            )
            session.add(document)
            await session.flush()
            session.add_all(
                LabDocumentPage(
                    document_id=document.id,
                    owner_id=owner_id,
                    page_index=index,
                    image_bytes=page.image_bytes,
                    mime_type=page.mime_type,
                    width=page.width,
                    height=page.height,
                    sha256=page.sha256,
                )
                for index, page in enumerate(pages)
            )
            await session.flush()
            return document

    async def page(self, owner_id: int, page: int) -> tuple[list[LabDocument], int]:
        safe_page = max(page, 0)
        async with self.db.sessions() as session:
            total = int(
                await session.scalar(
                    select(func.count(LabDocument.id)).where(LabDocument.owner_id == owner_id)
                )
                or 0
            )
            items = list(
                (
                    await session.scalars(
                        select(LabDocument)
                        .where(LabDocument.owner_id == owner_id)
                        .order_by(LabDocument.created_at.desc(), LabDocument.id.desc())
                        .offset(safe_page * LAB_LIST_PAGE_SIZE)
                        .limit(LAB_LIST_PAGE_SIZE)
                    )
                ).all()
            )
            return items, total

    async def get(self, owner_id: int, document_id: int) -> LabDocument | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(LabDocument)
                .options(selectinload(LabDocument.pages))
                .where(
                    LabDocument.id == document_id,
                    LabDocument.owner_id == owner_id,
                )
            )

    async def get_page(
        self, owner_id: int, document_id: int, page_index: int
    ) -> LabDocumentPage | None:
        async with self.db.sessions() as session:
            return await session.scalar(
                select(LabDocumentPage)
                .join(
                    LabDocument,
                    (LabDocument.id == LabDocumentPage.document_id)
                    & (LabDocument.owner_id == LabDocumentPage.owner_id),
                )
                .where(
                    LabDocument.owner_id == owner_id,
                    LabDocumentPage.owner_id == owner_id,
                    LabDocumentPage.document_id == document_id,
                    LabDocumentPage.page_index == page_index,
                )
            )

    async def rename(
        self, owner_id: int, document_id: int, expected_version: int, title: str
    ) -> bool:
        clean = sanitize_lab_title(title)
        async with self.db.session() as session:
            result = await session.execute(
                update(LabDocument)
                .where(
                    LabDocument.id == document_id,
                    LabDocument.owner_id == owner_id,
                    LabDocument.version == expected_version,
                )
                .values(title=clean, version=LabDocument.version + 1, updated_at=func.now())
            )
            return result.rowcount == 1

    async def set_date(
        self,
        owner_id: int,
        document_id: int,
        expected_version: int,
        document_date: date | None,
    ) -> bool:
        async with self.db.session() as session:
            result = await session.execute(
                update(LabDocument)
                .where(
                    LabDocument.id == document_id,
                    LabDocument.owner_id == owner_id,
                    LabDocument.version == expected_version,
                )
                .values(
                    document_date=document_date,
                    version=LabDocument.version + 1,
                    updated_at=func.now(),
                )
            )
            return result.rowcount == 1

    async def issue_delete(self, owner_id: int, chat_id: int, document_id: int) -> str | None:
        async with self.db.session() as session:
            document = await session.scalar(
                select(LabDocument).where(
                    LabDocument.id == document_id,
                    LabDocument.owner_id == owner_id,
                )
            )
            if document is None:
                return None
            token = secrets.token_hex(12)
            session.add(
                LabDeleteConfirmation(
                    token=token,
                    owner_id=owner_id,
                    chat_id=chat_id,
                    document_id=document.id,
                    document_version=document.version,
                    status="pending",
                    expires_at=datetime.now(UTC) + timedelta(minutes=DELETE_CONFIRM_TTL_MINUTES),
                )
            )
            return token

    async def confirm_delete(self, token: str, owner_id: int, chat_id: int) -> bool:
        now = datetime.now(UTC)
        async with self.db.session() as session:
            capability = await session.get(LabDeleteConfirmation, token)
            if (
                capability is None
                or capability.owner_id != owner_id
                or capability.chat_id != chat_id
                or capability.status != "pending"
                or _as_utc(capability.expires_at) <= now
            ):
                return False
            claimed = await session.execute(
                update(LabDeleteConfirmation)
                .where(
                    LabDeleteConfirmation.token == token,
                    LabDeleteConfirmation.status == "pending",
                )
                .values(status="consumed", consumed_at=now)
            )
            if claimed.rowcount != 1:
                return False
            removed = await session.execute(
                delete(LabDocument).where(
                    LabDocument.id == capability.document_id,
                    LabDocument.owner_id == owner_id,
                    LabDocument.version == capability.document_version,
                )
            )
            if removed.rowcount != 1:
                return False
            # SQLite connections may be opened with foreign-key enforcement disabled
            # by external maintenance tooling. Keep the application-level deletion
            # complete while the schema's ON DELETE CASCADE remains the primary guard.
            await session.execute(
                delete(LabDocumentPage).where(
                    LabDocumentPage.document_id == capability.document_id,
                    LabDocumentPage.owner_id == owner_id,
                )
            )
            return True

    async def cleanup_confirmations(self) -> int:
        async with self.db.session() as session:
            result = await session.execute(
                delete(LabDeleteConfirmation).where(
                    (LabDeleteConfirmation.expires_at <= datetime.now(UTC))
                    | (LabDeleteConfirmation.status == "consumed")
                )
            )
            return int(result.rowcount or 0)


def sanitize_lab_title(value: str) -> str:
    clean = " ".join(value.split())
    if not clean or len(clean) > 200 or any(ord(char) < 32 for char in clean):
        raise ValueError("invalid_title")
    return clean


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
