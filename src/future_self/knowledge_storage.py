from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import AsyncIterable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

AssetKind = Literal["original", "extracted"]

_OPAQUE_ID = re.compile(r"^[0-9a-f]{32}$")
_STAGING_NAME = re.compile(r"^[0-9a-f]{32}\.part$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 64 * 1024


class KnowledgeStorageError(ValueError):
    """A stable, non-sensitive storage failure code."""


@dataclass(frozen=True, slots=True)
class StagedAsset:
    token: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class StoredAsset:
    storage_key: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class InspectedStoredAsset[Inspection]:
    asset: StoredAsset
    inspection: Inspection


@dataclass(frozen=True, slots=True)
class StorageAudit:
    missing: tuple[str, ...]
    orphaned: tuple[str, ...]
    unsafe: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.orphaned or self.unsafe)


class KnowledgeAssetStore:
    """Private file storage with opaque keys and fail-closed path handling.

    Quota accounting intentionally stays in the database service. This boundary
    enforces actual streamed byte limits, durable staging, atomic same-filesystem
    publication, private modes, and idempotent cleanup.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_source_bytes: int = 25 * 1024 * 1024,
        max_extracted_bytes: int = 8 * 1024 * 1024,
        min_free_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        if max_source_bytes <= 0 or max_extracted_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("knowledge storage limits must be positive")
        self.root = Path(root)
        self.max_source_bytes = max_source_bytes
        self.max_extracted_bytes = max_extracted_bytes
        self.min_free_bytes = min_free_bytes
        self._initialize_layout()

    @property
    def staging_root(self) -> Path:
        return self._child_directory(".staging")

    @property
    def originals_root(self) -> Path:
        return self._child_directory("originals")

    @property
    def extracted_root(self) -> Path:
        return self._child_directory("extracted")

    async def stage_async(
        self,
        chunks: AsyncIterable[bytes],
        *,
        declared_size: int | None = None,
    ) -> StagedAsset:
        """Stream untrusted bytes to a private staging file and hash actual bytes."""

        if declared_size is not None and (
            declared_size <= 0 or declared_size > self.max_source_bytes
        ):
            raise KnowledgeStorageError("declared_size_rejected")
        token, descriptor = self._new_staging_file()
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                async for chunk in chunks:
                    size = self._write_chunk(output, chunk, size, digest, self.max_source_bytes)
                self._finish_staging(output, size)
        except BaseException:
            self._remove_staging_token(token)
            raise
        return StagedAsset(token, size, digest.hexdigest())

    def stage(
        self,
        chunks: Iterable[bytes],
        *,
        declared_size: int | None = None,
        max_bytes: int | None = None,
    ) -> StagedAsset:
        """Synchronous variant used by local capture adapters and backup tooling."""

        limit = max_bytes if max_bytes is not None else self.max_source_bytes
        if limit <= 0 or limit > max(self.max_source_bytes, self.max_extracted_bytes):
            raise ValueError("invalid staging byte limit")
        if declared_size is not None and (declared_size <= 0 or declared_size > limit):
            raise KnowledgeStorageError("declared_size_rejected")
        token, descriptor = self._new_staging_file()
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                for chunk in chunks:
                    size = self._write_chunk(output, chunk, size, digest, limit)
                self._finish_staging(output, size)
        except BaseException:
            self._remove_staging_token(token)
            raise
        return StagedAsset(token, size, digest.hexdigest())

    def stage_bytes(self, data: bytes, *, extracted: bool = False) -> StagedAsset:
        limit = self.max_extracted_bytes if extracted else self.max_source_bytes
        return self.stage((data,), declared_size=len(data), max_bytes=limit)

    def finalize(self, staged: StagedAsset, *, kind: AssetKind = "original") -> StoredAsset:
        """Publish staged bytes under a fresh opaque key without overwriting."""

        if kind not in {"original", "extracted"}:
            raise KnowledgeStorageError("invalid_asset_kind")
        self._validate_staged(staged)
        if staged.size_bytes > self._limit_for(kind):
            raise KnowledgeStorageError("asset_too_large")
        source = self._staging_path(staged.token)
        for _attempt in range(8):
            opaque_id = secrets.token_hex(16)
            key = self._key_for(kind, opaque_id)
            destination = self._path_for_key(key, create_parent=True)
            try:
                # An exclusive hard-link plus unlink gives atomic visibility without
                # os.replace's overwrite semantics. Both paths are on the same root.
                os.link(source, destination, follow_symlinks=False)
            except FileExistsError:
                continue
            except OSError as exc:
                raise KnowledgeStorageError("asset_finalize_failed") from exc
            try:
                source.unlink()
                self._fsync_directory(destination.parent)
                self._fsync_directory(self.staging_root)
                self._verify_regular(destination, max_bytes=self._limit_for(kind))
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            return StoredAsset(key, staged.size_bytes, staged.sha256)
        raise KnowledgeStorageError("storage_key_collision")

    def discard_staged(self, staged: StagedAsset) -> None:
        """Idempotently remove a known staging object."""

        if not _OPAQUE_ID.fullmatch(staged.token):
            raise KnowledgeStorageError("invalid_staging_token")
        self._remove_staging_token(staged.token)

    def staged_path_for_inspection(self, staged: StagedAsset) -> Path:
        """Return a revalidated private path for a read-only magic inspector.

        The caller must not persist this process-local path. The extraction
        boundary opens it again with no-follow semantics and rechecks its size.
        """

        self._validate_staged(staged)
        return self._staging_path(staged.token)

    def inspect_and_finalize[Inspection](
        self,
        staged: StagedAsset,
        *,
        declared_mime: str | None,
        display_name: str | None,
        inspector: Callable[..., Inspection],
    ) -> InspectedStoredAsset[Inspection]:
        """Run the format allowlist before publishing an original.

        `inspector` is injected to keep storage domain-neutral. It must implement
        the keyword contract of `knowledge_extraction.inspect_upload`. Any failed
        inspection discards staging and no permanent key becomes visible.
        """

        try:
            inspection = inspector(
                self.staged_path_for_inspection(staged),
                declared_mime=declared_mime,
                display_name=display_name,
                max_source_bytes=self.max_source_bytes,
            )
            asset = self.finalize(staged, kind="original")
        except BaseException:
            self.discard_staged(staged)
            raise
        return InspectedStoredAsset(asset, inspection)

    def asset_path_for_inspection(
        self,
        storage_key: str,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        """Return a verified immutable asset path for the ingestion worker."""

        self.verify_asset(
            storage_key,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        return self._path_for_key(storage_key)

    @contextmanager
    def open_asset(self, storage_key: str) -> Iterator[BinaryIO]:
        """Open a private regular asset with no-follow semantics."""

        kind = self._kind_for_key(storage_key)
        path = self._path_for_key(storage_key)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            raise KnowledgeStorageError("asset_missing") from None
        except OSError as exc:
            raise KnowledgeStorageError("asset_open_failed") from exc
        try:
            info = os.fstat(descriptor)
            self._validate_file_info(info, max_bytes=self._limit_for(kind))
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                yield stream
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def copy_asset_to(self, storage_key: str, destination: Path) -> StoredAsset:
        """Copy an asset to a new private path while re-hashing its actual bytes."""

        target = Path(destination)
        if target.exists() or target.is_symlink():
            raise KnowledgeStorageError("unsafe_copy_target")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        digest = hashlib.sha256()
        size = 0
        try:
            descriptor = os.open(target, flags, 0o600)
            with self.open_asset(storage_key) as source, os.fdopen(descriptor, "wb") as output:
                for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return StoredAsset(storage_key, size, digest.hexdigest())

    def verify_asset(
        self,
        storage_key: str,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> StoredAsset:
        if expected_sha256 is not None and not _SHA256.fullmatch(expected_sha256):
            raise KnowledgeStorageError("invalid_expected_hash")
        digest = hashlib.sha256()
        size = 0
        with self.open_asset(storage_key) as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
                size += len(chunk)
        actual_hash = digest.hexdigest()
        if expected_size is not None and size != expected_size:
            raise KnowledgeStorageError("asset_size_mismatch")
        if expected_sha256 is not None and actual_hash != expected_sha256:
            raise KnowledgeStorageError("asset_hash_mismatch")
        return StoredAsset(storage_key, size, actual_hash)

    def delete_asset(self, storage_key: str) -> bool:
        """Idempotently unlink one validated asset; never follows attacker paths."""

        kind = self._kind_for_key(storage_key)
        path = self._path_for_key(storage_key)
        try:
            self._verify_regular(path, max_bytes=self._limit_for(kind))
        except KnowledgeStorageError as exc:
            if str(exc) == "asset_missing":
                return False
            raise
        try:
            path.unlink()
            self._fsync_directory(path.parent)
        except OSError as exc:
            raise KnowledgeStorageError("asset_delete_failed") from exc
        return True

    def cleanup_staging(
        self, *, older_than_seconds: int, now: float | None = None
    ) -> tuple[str, ...]:
        if older_than_seconds < 0:
            raise ValueError("older_than_seconds must be non-negative")
        cutoff = (time.time() if now is None else now) - older_than_seconds
        removed: list[str] = []
        try:
            entries = tuple(self.staging_root.iterdir())
        except OSError as exc:
            raise KnowledgeStorageError("staging_scan_failed") from exc
        for entry in entries:
            if not _STAGING_NAME.fullmatch(entry.name):
                continue
            try:
                info = entry.lstat()
                self._validate_file_info(info, max_bytes=self.max_source_bytes)
                if info.st_mtime > cutoff:
                    continue
                entry.unlink()
                removed.append(entry.name[:-5])
            except FileNotFoundError:
                continue
            except KnowledgeStorageError:
                # Unknown links or non-regular entries are evidence, not cleanup
                # targets. The audit path reports them to an operator.
                continue
            except OSError as exc:
                raise KnowledgeStorageError("staging_cleanup_failed") from exc
        if removed:
            self._fsync_directory(self.staging_root)
        return tuple(sorted(removed))

    def audit(self, referenced_keys: Iterable[str]) -> StorageAudit:
        expected: set[str] = set()
        unsafe: set[str] = set()
        for key in referenced_keys:
            try:
                self._kind_for_key(key)
                expected.add(key)
            except KnowledgeStorageError:
                unsafe.add(str(key))
        actual: set[str] = set()
        for kind, base in (("original", self.originals_root), ("extracted", self.extracted_root)):
            for path in self._walk_private_files(base, unsafe):
                relative = path.relative_to(self.root.resolve()).as_posix()
                try:
                    if self._kind_for_key(relative) != kind:
                        raise KnowledgeStorageError("invalid_storage_key")
                    self._verify_regular(path, max_bytes=self._limit_for(kind))
                    actual.add(relative)
                except KnowledgeStorageError:
                    unsafe.add(relative)
        return StorageAudit(
            tuple(sorted(expected - actual)),
            tuple(sorted(actual - expected)),
            tuple(sorted(unsafe)),
        )

    def _initialize_layout(self) -> None:
        if self.root.exists() and self.root.is_symlink():
            raise KnowledgeStorageError("unsafe_storage_root")
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._tighten_directory(self.root)
            for name in (".staging", "originals", "extracted"):
                path = self.root / name
                path.mkdir(mode=0o700, exist_ok=True)
                self._tighten_directory(path)
        except KnowledgeStorageError:
            raise
        except OSError as exc:
            raise KnowledgeStorageError("unsafe_storage_root") from exc

    def _child_directory(self, name: str) -> Path:
        root = self.root.resolve()
        child = root / name
        if child.is_symlink() or not child.is_dir():
            raise KnowledgeStorageError("unsafe_storage_root")
        return child

    def _new_staging_file(self) -> tuple[str, int]:
        self._require_capacity(1)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        for _attempt in range(8):
            token = secrets.token_hex(16)
            try:
                descriptor = os.open(self._staging_path(token), flags, 0o600)
            except FileExistsError:
                continue
            except OSError as exc:
                raise KnowledgeStorageError("staging_create_failed") from exc
            self._fsync_directory(self.staging_root)
            return token, descriptor
        raise KnowledgeStorageError("staging_collision")

    def _write_chunk(
        self,
        output: BinaryIO,
        chunk: bytes,
        current_size: int,
        digest: object,
        limit: int,
    ) -> int:
        if not isinstance(chunk, bytes):
            raise KnowledgeStorageError("invalid_stream_chunk")
        next_size = current_size + len(chunk)
        if next_size > limit:
            raise KnowledgeStorageError("source_too_large")
        if chunk:
            self._require_capacity(len(chunk))
            try:
                output.write(chunk)
            except OSError as exc:
                raise KnowledgeStorageError("staging_write_failed") from exc
            digest.update(chunk)  # type: ignore[attr-defined]
        return next_size

    @staticmethod
    def _finish_staging(output: BinaryIO, size: int) -> None:
        if size <= 0:
            raise KnowledgeStorageError("empty_source")
        try:
            output.flush()
            os.fsync(output.fileno())
        except OSError as exc:
            raise KnowledgeStorageError("staging_write_failed") from exc

    def _require_capacity(self, incoming_bytes: int) -> None:
        try:
            available = shutil.disk_usage(self.root).free
        except OSError as exc:
            raise KnowledgeStorageError("storage_capacity_unavailable") from exc
        if available - incoming_bytes < self.min_free_bytes:
            raise KnowledgeStorageError("insufficient_storage")

    def _validate_staged(self, staged: StagedAsset) -> None:
        if (
            not _OPAQUE_ID.fullmatch(staged.token)
            or staged.size_bytes <= 0
            or staged.size_bytes > max(self.max_source_bytes, self.max_extracted_bytes)
            or not _SHA256.fullmatch(staged.sha256)
        ):
            raise KnowledgeStorageError("invalid_staged_asset")
        path = self._staging_path(staged.token)
        self._verify_regular(path, max_bytes=max(self.max_source_bytes, self.max_extracted_bytes))
        verified = self._hash_path(path)
        if verified.size_bytes != staged.size_bytes or verified.sha256 != staged.sha256:
            raise KnowledgeStorageError("staged_asset_changed")

    def _remove_staging_token(self, token: str) -> None:
        try:
            self._staging_path(token).unlink(missing_ok=True)
            self._fsync_directory(self.staging_root)
        except OSError as exc:
            raise KnowledgeStorageError("staging_cleanup_failed") from exc

    def _staging_path(self, token: str) -> Path:
        if not _OPAQUE_ID.fullmatch(token):
            raise KnowledgeStorageError("invalid_staging_token")
        return self.staging_root / f"{token}.part"

    @staticmethod
    def _key_for(kind: AssetKind, opaque_id: str) -> str:
        suffix = ".txt" if kind == "extracted" else ""
        prefix = "originals" if kind == "original" else "extracted"
        return f"{prefix}/{opaque_id[:2]}/{opaque_id[2:4]}/{opaque_id}{suffix}"

    def _kind_for_key(self, storage_key: str) -> AssetKind:
        if not isinstance(storage_key, str) or "\\" in storage_key:
            raise KnowledgeStorageError("invalid_storage_key")
        path = PurePosixPath(storage_key)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 4:
            raise KnowledgeStorageError("invalid_storage_key")
        prefix, first, second, filename = path.parts
        if prefix == "originals":
            opaque_id = filename
            kind: AssetKind = "original"
        elif prefix == "extracted" and filename.endswith(".txt"):
            opaque_id = filename[:-4]
            kind = "extracted"
        else:
            raise KnowledgeStorageError("invalid_storage_key")
        if (
            not _OPAQUE_ID.fullmatch(opaque_id)
            or first != opaque_id[:2]
            or second != opaque_id[2:4]
        ):
            raise KnowledgeStorageError("invalid_storage_key")
        return kind

    def _path_for_key(self, storage_key: str, *, create_parent: bool = False) -> Path:
        self._kind_for_key(storage_key)
        root = self.root.resolve()
        path = root.joinpath(*PurePosixPath(storage_key).parts)
        if create_parent:
            current = root
            for part in PurePosixPath(storage_key).parts[:-1]:
                current = current / part
                try:
                    current.mkdir(mode=0o700, exist_ok=True)
                except OSError as exc:
                    raise KnowledgeStorageError("unsafe_storage_path") from exc
                self._tighten_directory(current)
        try:
            resolved_parent = path.parent.resolve()
        except OSError as exc:
            raise KnowledgeStorageError("unsafe_storage_path") from exc
        if resolved_parent == root or not resolved_parent.is_relative_to(root):
            raise KnowledgeStorageError("unsafe_storage_path")
        return path

    @staticmethod
    def _tighten_directory(path: Path) -> None:
        try:
            info = path.lstat()
            if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
                raise KnowledgeStorageError("unsafe_storage_path")
            path.chmod(0o700)
        except KnowledgeStorageError:
            raise
        except OSError as exc:
            raise KnowledgeStorageError("unsafe_storage_path") from exc

    @staticmethod
    def _validate_file_info(info: os.stat_result, *, max_bytes: int) -> None:
        private = os.name != "posix" or stat.S_IMODE(info.st_mode) & 0o077 == 0
        if (
            not stat.S_ISREG(info.st_mode)
            or not private
            or info.st_nlink != 1
            or info.st_size < 0
            or info.st_size > max_bytes
        ):
            raise KnowledgeStorageError("unsafe_asset")

    def _verify_regular(self, path: Path, *, max_bytes: int) -> os.stat_result:
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise KnowledgeStorageError("asset_missing") from None
        except OSError as exc:
            raise KnowledgeStorageError("asset_stat_failed") from exc
        if path.is_symlink():
            raise KnowledgeStorageError("unsafe_asset")
        self._validate_file_info(info, max_bytes=max_bytes)
        return info

    def _hash_path(self, path: Path) -> StoredAsset:
        digest = hashlib.sha256()
        size = 0
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            self._validate_file_info(
                os.fstat(descriptor),
                max_bytes=max(self.max_source_bytes, self.max_extracted_bytes),
            )
            with os.fdopen(descriptor, "rb") as stream:
                for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except KnowledgeStorageError:
            raise
        except OSError as exc:
            raise KnowledgeStorageError("asset_read_failed") from exc
        return StoredAsset("", size, digest.hexdigest())

    def _walk_private_files(self, root: Path, unsafe: set[str]) -> Iterator[Path]:
        resolved_root = self.root.resolve()
        for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            safe_directories: list[str] = []
            for name in directory_names:
                candidate = current_path / name
                relative = candidate.relative_to(resolved_root).as_posix()
                try:
                    info = candidate.lstat()
                    private = os.name != "posix" or stat.S_IMODE(info.st_mode) & 0o077 == 0
                    if candidate.is_symlink() or not stat.S_ISDIR(info.st_mode) or not private:
                        raise KnowledgeStorageError("unsafe_storage_path")
                    safe_directories.append(name)
                except (KnowledgeStorageError, OSError):
                    unsafe.add(relative)
            directory_names[:] = safe_directories
            for name in file_names:
                yield current_path / name

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name != "posix":
            return
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        try:
            descriptor = os.open(path, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise KnowledgeStorageError("directory_sync_failed") from exc

    def _limit_for(self, kind: AssetKind | str) -> int:
        return self.max_source_bytes if kind == "original" else self.max_extracted_bytes
