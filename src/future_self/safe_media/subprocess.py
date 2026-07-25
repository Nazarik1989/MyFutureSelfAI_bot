from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
    }
)
_MODULE_PATTERN = re.compile(r"^future_self\.safe_media\.[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*$")


class SafeSubprocessError(ValueError):
    """A stable failure code from the isolated subprocess boundary."""


def sanitized_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return only process/runtime variables; provider keys and user data are excluded."""

    values = source if source is not None else os.environ
    environment = {key: value for key, value in values.items() if key.upper() in _SAFE_ENV_KEYS}
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    return environment


def run_isolated_python_module(
    module: str,
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    """Run a fixed internal worker without a shell, inherited stdin, or inherited secrets."""

    if not _MODULE_PATTERN.fullmatch(module):
        raise SafeSubprocessError("invalid_worker")
    requested = Path(cwd)
    if requested.is_symlink():
        raise SafeSubprocessError("unsafe_work_directory")
    work = requested.resolve()
    if not work.is_dir():
        raise SafeSubprocessError("unsafe_work_directory")
    environment = sanitized_environment()
    environment.update({"TEMP": str(work), "TMP": str(work), "TMPDIR": str(work)})
    try:
        return subprocess.run(
            [sys.executable, "-m", module, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            timeout=timeout_seconds,
            check=False,
            env=environment,
            cwd=work,
        )
    except subprocess.TimeoutExpired:
        raise SafeSubprocessError("worker_timeout") from None


@contextmanager
def private_temporary_directory(root: Path, *, prefix: str) -> Iterator[Path]:
    """Create a private non-symlink work directory below an explicit trusted root."""

    requested = Path(root)
    if requested.exists() and requested.is_symlink():
        raise SafeSubprocessError("unsafe_temporary_storage")
    requested.mkdir(mode=0o700, parents=True, exist_ok=True)
    if requested.is_symlink():
        raise SafeSubprocessError("unsafe_temporary_storage")
    tighten_directory(requested)
    resolved_root = requested.resolve()
    with tempfile.TemporaryDirectory(prefix=prefix, dir=resolved_root) as directory:
        work = Path(directory).resolve()
        ensure_child(resolved_root, work)
        tighten_directory(work)
        yield work


def write_private_file(path: Path, data: bytes) -> None:
    """Create a new regular file with no-follow semantics and durable contents."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SafeSubprocessError("unsafe_output_path") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def regular_private_file(path: Path, *, max_bytes: int) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    private_permissions = os.name != "posix" or stat.S_IMODE(info.st_mode) & 0o077 == 0
    return (
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_nlink == 1
        and private_permissions
        and 0 <= info.st_size <= max_bytes
    )


def tighten_directory(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise SafeSubprocessError("unsafe_temporary_storage") from exc


def ensure_child(parent: Path, child: Path) -> None:
    resolved_parent = parent.resolve()
    resolved_child = child.resolve()
    if resolved_child == resolved_parent or not resolved_child.is_relative_to(resolved_parent):
        raise SafeSubprocessError("unsafe_temporary_storage")
