import hashlib
import os
import time
from pathlib import Path

import pytest

from future_self.knowledge_extraction import KnowledgeExtractionError, inspect_upload
from future_self.knowledge_storage import KnowledgeAssetStore, KnowledgeStorageError


def store(tmp_path: Path, *, max_bytes: int = 1024, min_free_bytes: int = 0) -> KnowledgeAssetStore:
    return KnowledgeAssetStore(
        tmp_path / "knowledge",
        max_source_bytes=max_bytes,
        max_extracted_bytes=max_bytes,
        min_free_bytes=min_free_bytes,
    )


async def chunks(*values: bytes):
    for value in values:
        yield value


async def test_streaming_staging_hashes_actual_bytes_and_finalizes_under_opaque_key(
    tmp_path: Path,
) -> None:
    storage = store(tmp_path)
    staged = await storage.stage_async(chunks(b"hello", b" ", b"world"), declared_size=1)

    assert staged.size_bytes == 11
    assert staged.sha256 == hashlib.sha256(b"hello world").hexdigest()

    stored = storage.finalize(staged)
    assert stored.storage_key.startswith("originals/")
    assert "hello" not in stored.storage_key
    assert not (storage.staging_root / f"{staged.token}.part").exists()
    assert (
        storage.verify_asset(
            stored.storage_key,
            expected_size=11,
            expected_sha256=staged.sha256,
        )
        == stored
    )
    with storage.open_asset(stored.storage_key) as source:
        assert source.read() == b"hello world"


async def test_actual_stream_limit_is_enforced_and_staging_is_removed(tmp_path: Path) -> None:
    storage = store(tmp_path, max_bytes=8)

    with pytest.raises(KnowledgeStorageError, match="source_too_large"):
        await storage.stage_async(chunks(b"1234", b"56789"), declared_size=1)

    assert list(storage.staging_root.iterdir()) == []


def test_storage_rejects_traversal_symlink_and_hardlink_assets(tmp_path: Path) -> None:
    storage = store(tmp_path)
    stored = storage.finalize(storage.stage_bytes(b"private"))

    with pytest.raises(KnowledgeStorageError, match="invalid_storage_key"):
        with storage.open_asset("../outside"):
            pass

    original = storage.root.joinpath(*stored.storage_key.split("/"))
    linked = tmp_path / "linked"
    try:
        os.link(original, linked)
    except OSError:
        pytest.skip("hardlinks are unavailable")
    with pytest.raises(KnowledgeStorageError, match="unsafe_asset"):
        storage.verify_asset(stored.storage_key)
    linked.unlink()

    target = tmp_path / "target"
    target.write_bytes(b"outside")
    symlink = storage.originals_root / "bad-link"
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    audit = storage.audit({stored.storage_key})
    assert "originals/bad-link" in audit.unsafe
    assert target.read_bytes() == b"outside"


def test_low_disk_guard_and_private_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = store(tmp_path, min_free_bytes=100)

    class Usage:
        free = 50

    monkeypatch.setattr("future_self.knowledge_storage.shutil.disk_usage", lambda _path: Usage())
    with pytest.raises(KnowledgeStorageError, match="insufficient_storage"):
        storage.stage_bytes(b"x")

    if os.name == "posix":
        assert storage.root.stat().st_mode & 0o777 == 0o700


def test_staging_cleanup_and_orphan_missing_audit_are_fail_closed(tmp_path: Path) -> None:
    storage = store(tmp_path)
    staged = storage.stage_bytes(b"abandoned")
    staged_path = storage.staging_root / f"{staged.token}.part"
    old = time.time() - 600
    os.utime(staged_path, (old, old))
    assert storage.cleanup_staging(older_than_seconds=60) == (staged.token,)

    referenced = storage.finalize(storage.stage_bytes(b"referenced"))
    orphaned = storage.finalize(storage.stage_bytes(b"orphaned"))
    audit = storage.audit({referenced.storage_key, "originals/aa/aa/" + "a" * 32})
    assert audit.missing == ("originals/aa/aa/" + "a" * 32,)
    assert audit.orphaned == (orphaned.storage_key,)
    assert not audit.ok


def test_extracted_assets_copy_verify_and_delete_idempotently(tmp_path: Path) -> None:
    storage = store(tmp_path)
    staged = storage.stage_bytes("Текст".encode(), extracted=True)
    stored = storage.finalize(staged, kind="extracted")
    destination = tmp_path / "copy.txt"

    copied = storage.copy_asset_to(stored.storage_key, destination)
    assert copied.sha256 == stored.sha256
    assert destination.read_bytes() == "Текст".encode()
    assert storage.delete_asset(stored.storage_key) is True
    assert storage.delete_asset(stored.storage_key) is False

    bounded = KnowledgeAssetStore(
        tmp_path / "bounded",
        max_source_bytes=16,
        max_extracted_bytes=8,
        min_free_bytes=0,
    )
    oversized = bounded.stage_bytes(b"x" * 9)
    with pytest.raises(KnowledgeStorageError, match="asset_too_large"):
        bounded.finalize(oversized, kind="extracted")
    bounded.discard_staged(oversized)


def test_inspect_and_finalize_rejects_mime_spoof_before_publication(tmp_path: Path) -> None:
    storage = store(tmp_path)
    staged = storage.stage_bytes(b"%PDF-1.4\n%%EOF")

    with pytest.raises(KnowledgeExtractionError, match="mime_mismatch"):
        storage.inspect_and_finalize(
            staged,
            declared_mime="image/png",
            display_name="scan.png",
            inspector=inspect_upload,
        )

    assert list(storage.staging_root.iterdir()) == []
    assert list(storage.originals_root.rglob("*")) == []
