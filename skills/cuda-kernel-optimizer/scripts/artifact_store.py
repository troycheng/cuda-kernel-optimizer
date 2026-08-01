#!/usr/bin/env python3
"""Versioned, traversal-safe storage for optimizer run artifacts."""

from __future__ import annotations

import copy
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import Any, Optional, Union


CURRENT_SCHEMA_VERSION = 2
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9._-]+")
_PathLike = Union[str, os.PathLike]


class StaleReferenceError(ValueError):
    """Raised when a compare-and-swap reference no longer matches."""


def publish_directory_noreplace(source: _PathLike, destination: _PathLike) -> None:
    """Atomically publish one directory and fail if the destination exists."""
    source_path = Path(os.path.abspath(os.path.expanduser(os.fspath(source))))
    destination_path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(destination)))
    )
    if source_path.parent != destination_path.parent:
        raise ValueError("directory publication requires one parent directory")
    if not source_path.is_dir() or source_path.is_symlink():
        raise ValueError("published source must be a real directory")
    library = ctypes.CDLL(None, use_errno=True)
    old = os.fsencode(source_path)
    new = os.fsencode(destination_path)
    if sys.platform == "darwin":
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        returncode = function(old, new, 0x00000004)
    elif sys.platform.startswith("linux"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        returncode = function(-100, old, -100, new, 1)
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
        )
    if returncode != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(code, os.strerror(code), destination_path)
        raise OSError(code, os.strerror(code), destination_path)
    parent_fd = os.open(
        source_path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def sha256_file(path: _PathLike) -> str:
    """Return a stable SHA-256 digest without following path symlinks."""
    return hashlib.sha256(read_regular_bytes(path)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("artifact value is not finite JSON") from error


def _open_parent_directory(
    path: _PathLike, *, create: bool
) -> tuple[int, str, Path]:
    """Open a stable parent dirfd without following any path component."""
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not target.name:
        raise ValueError("artifact path must name a file")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(target.anchor, flags)
    try:
        for index, component in enumerate(target.parts[1:-1]):
            # macOS exposes root-owned compatibility aliases such as /var ->
            # /private/var.  Permit only that filesystem-root boundary; every
            # user-controlled descendant remains no-follow.
            component_flags = flags if index == 0 else flags | nofollow
            try:
                child_fd = os.open(
                    component, component_flags, dir_fd=directory_fd
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=directory_fd)
                child_fd = os.open(
                    component, component_flags, dir_fd=directory_fd
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"artifact parent path contains a symlink or non-directory: {target}"
                    ) from error
                raise
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd, target.name, target
    except BaseException:
        os.close(directory_fd)
        raise


def _read_regular_leaf(
    directory_fd: int,
    leaf: str,
    target: Path,
    *,
    missing_ok: bool,
    maximum_bytes: Optional[int] = None,
) -> Optional[bytes]:
    """Read one no-follow regular leaf from an already stable parent dirfd."""
    fd = None
    try:
        try:
            fd = os.open(
                leaf,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except OSError as error:
            if missing_ok and error.errno == errno.ENOENT:
                return None
            if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
                raise ValueError(
                    f"artifact file is missing, a symlink, or unsafe: {target}"
                ) from error
            raise
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"artifact path is not a regular file: {target}")
        if (
            maximum_bytes is not None
            and (
                isinstance(maximum_bytes, bool)
                or not isinstance(maximum_bytes, int)
                or maximum_bytes < 0
            )
        ):
            raise ValueError("maximum_bytes must be a non-negative integer")
        if maximum_bytes is not None and metadata.st_size > maximum_bytes:
            raise ValueError(f"artifact file exceeds byte limit: {target}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum_bytes is not None and total > maximum_bytes:
                raise ValueError(f"artifact file exceeds byte limit: {target}")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if fd is not None:
            os.close(fd)


def read_regular_bytes(
    path: _PathLike,
    *,
    maximum_bytes: Optional[int] = None,
) -> bytes:
    """Read regular-file bytes through no-follow parent and leaf descriptors."""
    try:
        directory_fd, leaf, target = _open_parent_directory(path, create=False)
    except FileNotFoundError as error:
        raise ValueError(f"artifact file does not exist: {path}") from error
    try:
        return _read_regular_leaf(
            directory_fd,
            leaf,
            target,
            missing_ok=False,
            maximum_bytes=maximum_bytes,
        )
    finally:
        os.close(directory_fd)


def read_regular_with_optional_sibling(
    path: _PathLike, sibling_name: str
) -> tuple[bytes, Optional[bytes]]:
    """Read a regular file and optional sibling through one stable parent dirfd."""
    sibling = Path(sibling_name)
    if (
        not sibling_name
        or sibling.is_absolute()
        or len(sibling.parts) != 1
        or sibling_name in {".", ".."}
    ):
        raise ValueError("sibling_name must name one relative file")
    try:
        directory_fd, leaf, target = _open_parent_directory(path, create=False)
    except FileNotFoundError as error:
        raise ValueError(f"artifact file does not exist: {path}") from error

    try:
        primary = _read_regular_leaf(
            directory_fd, leaf, target, missing_ok=False
        )
        sibling_payload = _read_regular_leaf(
            directory_fd,
            sibling_name,
            target.with_name(sibling_name),
            missing_ok=True,
        )
        return primary, sibling_payload
    finally:
        os.close(directory_fd)


_BUNDLE_GENERATION = re.compile(r"[0-9a-f]{32}")


def _open_bundle_generation(directory_fd: int, generation: str, directory_path: Path) -> int:
    """Open one immutable bundle generation below an already-open bundle root."""
    if _BUNDLE_GENERATION.fullmatch(generation) is None:
        raise ValueError("artifact bundle current pointer has an invalid generation")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    generations_fd = None
    try:
        generations_fd = os.open("generations", flags, dir_fd=directory_fd)
        return os.open(generation, flags, dir_fd=generations_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
            raise ValueError(
                f"artifact bundle generation is missing, a symlink, or unsafe: {directory_path}"
            ) from error
        raise
    finally:
        if generations_fd is not None:
            os.close(generations_fd)


def _read_bundle_pointer(
    directory_fd: int, directory_path: Path, *, missing_ok: bool = False
) -> Optional[tuple[str, str]]:
    payload = _read_regular_leaf(
        directory_fd, "current", directory_path / "current", missing_ok=missing_ok
    )
    if payload is None:
        return None
    try:
        pointer = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact bundle current pointer is invalid JSON") from error
    if type(pointer) is not dict or set(pointer) != {"generation"}:
        raise ValueError("artifact bundle current pointer fields are invalid")
    generation = pointer["generation"]
    if type(generation) is not str or _BUNDLE_GENERATION.fullmatch(generation) is None:
        raise ValueError("artifact bundle current pointer has an invalid generation")
    return generation, hashlib.sha256(payload).hexdigest()


def read_regular_bundle(directory: _PathLike, names) -> dict[str, bytes]:
    """Read named files from the immutable generation named by current once."""
    if isinstance(names, (str, bytes, bytearray, Mapping)):
        raise ValueError("artifact bundle names must be a sequence")
    try:
        clean_names = [_validate_leaf_name(name) for name in names]
    except TypeError as error:
        raise ValueError("artifact bundle names must be a sequence") from error
    if len(clean_names) != len(set(clean_names)):
        raise ValueError("artifact bundle names must be unique")
    directory_path = Path(os.path.abspath(os.path.expanduser(os.fspath(directory))))
    marker = directory_path / ".read-bundle"
    try:
        directory_fd, _leaf, _target = _open_parent_directory(marker, create=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"artifact bundle directory is missing, a symlink, or unsafe: {directory_path}"
        ) from error
    generation_fd = None
    try:
        pointer = _read_bundle_pointer(
            directory_fd, directory_path
        )
        assert pointer is not None
        generation, _pointer_digest = pointer
        generation_fd = _open_bundle_generation(
            directory_fd, generation, directory_path
        )
        return {
            name: _read_regular_leaf(
                generation_fd,
                name,
                directory_path / "generations" / generation / name,
                missing_ok=False,
            )
            for name in clean_names
        }
    finally:
        if generation_fd is not None:
            os.close(generation_fd)
        os.close(directory_fd)


def _validate_leaf_name(value: str) -> str:
    if type(value) is not str or not value or value in {".", ".."}:
        raise ValueError("artifact bundle names must be non-empty file names")
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or len(path.parts) != 1
        or len(windows_path.parts) != 1
    ):
        raise ValueError("artifact bundle names must contain one relative component")
    return value


def _atomic_write_leaf(
    directory_fd: int, leaf: str, target: Path, payload: bytes
) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    temporary_leaf = f".{leaf}.{secrets.token_hex(8)}.tmp"
    fd = None
    try:
        try:
            existing = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise ValueError(f"artifact target path contains a symlink: {target}")
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(f"artifact target must be a regular file: {target}")
        fd = os.open(
            temporary_leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("atomic artifact write made no progress")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(
            temporary_leaf,
            leaf,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary_leaf, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def _remove_regular_leaf(
    directory_fd: int, leaf: str, target: Path, *, missing_ok: bool
) -> bool:
    try:
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        if missing_ok:
            return False
        raise ValueError(f"artifact file does not exist: {target}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact path is not a regular file: {target}")
    os.unlink(leaf, dir_fd=directory_fd)
    return True


def remove_regular_file(path: _PathLike, *, missing_ok: bool = True) -> bool:
    """Remove one regular file relative to a stable, no-follow parent dirfd."""
    try:
        directory_fd, leaf, target = _open_parent_directory(path, create=False)
    except FileNotFoundError as error:
        if missing_ok:
            return False
        raise ValueError(f"artifact file does not exist: {path}") from error
    try:
        removed = _remove_regular_leaf(
            directory_fd, leaf, target, missing_ok=missing_ok
        )
        os.fsync(directory_fd)
        return removed
    finally:
        os.close(directory_fd)


def atomic_write_bytes(path: _PathLike, payload: bytes) -> None:
    """Atomically replace a file relative to one stable, no-follow parent dirfd."""
    if not isinstance(payload, bytes):
        raise TypeError("atomic payload must be bytes")
    directory_fd, leaf, target = _open_parent_directory(path, create=True)
    try:
        _atomic_write_leaf(directory_fd, leaf, target, payload)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_regular_bytes(path: _PathLike, payload: bytes) -> None:
    """Durably append bytes to one regular no-follow file."""
    if not isinstance(payload, bytes):
        raise TypeError("append payload must be bytes")
    directory_fd, leaf, target = _open_parent_directory(path, create=True)
    descriptor = None
    try:
        descriptor = os.open(
            leaf,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact append path is not regular: {target}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("artifact append made no progress")
            offset += written
        os.fsync(descriptor)
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(directory_fd)


def create_regular_directory(
    path: _PathLike,
    *,
    mode: int = 0o700,
    exist_ok: bool = False,
) -> Path:
    """Create one directory below a stable no-follow parent."""
    directory_fd, leaf, target = _open_parent_directory(path, create=True)
    try:
        try:
            os.mkdir(leaf, mode, dir_fd=directory_fd)
        except FileExistsError:
            if not exist_ok:
                raise
            metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"artifact directory path is not a real directory: {target}"
                )
        os.fsync(directory_fd)
        return target
    finally:
        os.close(directory_fd)


def list_regular_directories(
    path: _PathLike,
    *,
    prefix: str = "",
) -> list[str]:
    """List direct child directories without following the container or children."""
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    parent_fd, leaf, _ = _open_parent_directory(target, create=False)
    directory_fd = None
    try:
        directory_fd = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        names = []
        for name in os.listdir(directory_fd):
            if not name.startswith(prefix):
                continue
            metadata = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"artifact directory entry is not a real directory: {target / name}"
                )
            names.append(name)
        return sorted(names)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"artifact directory is missing, a symlink, or unsafe: {target}"
            ) from error
        raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def _relative_artifact_path(value: _PathLike) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise ValueError("artifact relative path must be a non-empty string") from error
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact relative path must be a non-empty string")
    path = Path(raw)
    windows_path = PureWindowsPath(raw)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part in {"", ".", ".."} for part in windows_path.parts)
    ):
        raise ValueError("artifact relative path must stay inside its root")
    return path


@contextmanager
def _locked_reference(artifact_root: Path, relative: Path):
    """Serialize all reads and writes of one mutable root-relative reference."""
    lock_key = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()
    lock_path = artifact_root / ".locks" / f"cas-{lock_key}.lock"
    lock_directory_fd, lock_leaf, lock_target = _open_parent_directory(
        lock_path, create=True
    )
    lock_fd = None
    try:
        try:
            lock_fd = os.open(
                lock_leaf,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=lock_directory_fd,
            )
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    f"artifact reference lock is a symlink or unsafe: {lock_target}"
                ) from error
            raise
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ValueError(
                f"artifact reference lock is not regular: {lock_target}"
            )
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(lock_directory_fd)


def _reference_digest(artifact_root: Path, relative: Path) -> Optional[str]:
    target = artifact_root / relative
    target_directory_fd, target_leaf, target_path = _open_parent_directory(
        target, create=True
    )
    try:
        current = _read_regular_leaf(
            target_directory_fd,
            target_leaf,
            target_path,
            missing_ok=True,
        )
    finally:
        os.close(target_directory_fd)
    return None if current is None else hashlib.sha256(current).hexdigest()


def compare_and_swap_ref(
    artifact_root: _PathLike,
    relative_path: _PathLike,
    expected_digest: Optional[str],
    replacement: Mapping[str, Any],
) -> str:
    """Atomically replace one root-relative JSON reference if its digest matches."""
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    relative = _relative_artifact_path(relative_path)
    if (
        expected_digest is not None
        and (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        )
    ):
        raise ValueError("expected reference digest must be SHA-256 or null")
    if not isinstance(replacement, Mapping):
        raise ValueError("replacement reference must be an object")
    try:
        payload = (
            json.dumps(
                dict(replacement),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("replacement reference is not serializable") from error

    with _locked_reference(root, relative):
        target = root / relative
        target_directory_fd, target_leaf, target_path = _open_parent_directory(
            target, create=True
        )
        try:
            current = _read_regular_leaf(
                target_directory_fd,
                target_leaf,
                target_path,
                missing_ok=True,
            )
            current_digest = (
                None if current is None else hashlib.sha256(current).hexdigest()
            )
            if current_digest != expected_digest:
                raise StaleReferenceError(
                    "artifact reference changed before compare-and-swap"
                )
            _atomic_write_leaf(
                target_directory_fd,
                target_leaf,
                target_path,
                payload,
            )
            os.fsync(target_directory_fd)
        finally:
            os.close(target_directory_fd)
        return hashlib.sha256(payload).hexdigest()


def create_regular_json_if_ref_digest(
    artifact_root: _PathLike,
    reference_path: _PathLike,
    expected_digest: Optional[str],
    output_path: _PathLike,
    payload: Any,
) -> None:
    """Create one immutable record only while a mutable reference is unchanged."""
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    reference = _relative_artifact_path(reference_path)
    output = _relative_artifact_path(output_path)
    if (
        expected_digest is not None
        and (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        )
    ):
        raise ValueError("expected reference digest must be SHA-256 or null")
    with _locked_reference(root, reference):
        if _reference_digest(root, reference) != expected_digest:
            raise StaleReferenceError(
                "artifact reference changed before immutable record publication"
            )
        create_regular_json(root / output, payload)


def _scan_limits(value) -> tuple[int, int, float]:
    if type(value) is not dict:
        raise ValueError("scan_limits must be an object")
    fields = {"max_files", "max_total_bytes", "max_wall_seconds"}
    if set(value) != fields:
        raise ValueError("scan_limits fields are incomplete or unknown")
    max_files = value["max_files"]
    max_total_bytes = value["max_total_bytes"]
    max_wall_seconds = value["max_wall_seconds"]
    if (
        isinstance(max_files, bool)
        or not isinstance(max_files, int)
        or max_files < 1
    ):
        raise ValueError("scan_limits.max_files must be a positive integer")
    if (
        isinstance(max_total_bytes, bool)
        or not isinstance(max_total_bytes, int)
        or max_total_bytes < 1
    ):
        raise ValueError("scan_limits.max_total_bytes must be a positive integer")
    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or max_wall_seconds <= 0
    ):
        raise ValueError("scan_limits.max_wall_seconds must be positive")
    return max_files, max_total_bytes, float(max_wall_seconds)


def _read_snapshot_file(
    directory_fd: int,
    name: str,
    display_path: Path,
    *,
    deadline: float,
    remaining_bytes: int,
) -> tuple[bytes, int]:
    if time.monotonic() >= deadline:
        raise TimeoutError("artifact scan exceeded max_wall_seconds")
    descriptor = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"snapshot input is not a regular file: {display_path}")
        if before.st_size > remaining_bytes:
            raise ValueError("artifact scan exceeded max_total_bytes")
        chunks = []
        total = 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("artifact scan exceeded max_wall_seconds")
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > remaining_bytes:
                raise ValueError("artifact scan exceeded max_total_bytes")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"snapshot input changed while reading: {display_path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ValueError(f"snapshot input size changed: {display_path}")
        return payload, stat.S_IMODE(before.st_mode)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _scan_directory(
    directory_fd: int,
    relative: Path,
    display_root: Path,
    *,
    deadline: float,
    records: list[dict],
    payloads: dict[str, bytes],
    counters: dict[str, int],
    max_files: int,
    max_total_bytes: int,
) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("artifact scan exceeded max_wall_seconds")
    for name in sorted(os.listdir(directory_fd)):
        if name in {".", ".."}:
            continue
        child_relative = relative / name
        display_path = display_root / child_relative
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"snapshot input contains a symlink: {display_path}")
        if stat.S_ISDIR(metadata.st_mode):
            records.append(
                {
                    "path": child_relative.as_posix(),
                    "kind": "directory",
                    "mode": stat.S_IMODE(metadata.st_mode),
                }
            )
            child_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                _scan_directory(
                    child_fd,
                    child_relative,
                    display_root,
                    deadline=deadline,
                    records=records,
                    payloads=payloads,
                    counters=counters,
                    max_files=max_files,
                    max_total_bytes=max_total_bytes,
                )
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"snapshot input contains an unsupported entry: {display_path}")
        payload, mode = _read_snapshot_file(
            directory_fd,
            name,
            display_path,
            deadline=deadline,
            remaining_bytes=max_total_bytes - counters["bytes"],
        )
        counters["files"] += 1
        counters["bytes"] += len(payload)
        if counters["files"] > max_files:
            raise ValueError("artifact scan exceeded max_files")
        if counters["bytes"] > max_total_bytes:
            raise ValueError("artifact scan exceeded max_total_bytes")
        key = child_relative.as_posix()
        payloads[key] = payload
        records.append(
            {
                "path": key,
                "kind": "file",
                "mode": mode,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )


def _snapshot_source(source, scan_limits) -> tuple[dict, dict[str, bytes]]:
    max_files, max_total_bytes, max_wall_seconds = _scan_limits(scan_limits)
    deadline = time.monotonic() + max_wall_seconds
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(source))))
    parent_fd, leaf, _target = _open_parent_directory(target, create=False)
    records = []
    payloads = {}
    counters = {"files": 0, "bytes": 0}
    try:
        metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"snapshot input is a symlink: {target}")
        if stat.S_ISREG(metadata.st_mode):
            payload, mode = _read_snapshot_file(
                parent_fd,
                leaf,
                target,
                deadline=deadline,
                remaining_bytes=max_total_bytes,
            )
            counters = {"files": 1, "bytes": len(payload)}
            if counters["files"] > max_files or counters["bytes"] > max_total_bytes:
                raise ValueError("artifact scan exceeded configured limits")
            payloads[target.name] = payload
            records.append(
                {
                    "path": target.name,
                    "kind": "file",
                    "mode": mode,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            source_kind = "file"
        elif stat.S_ISDIR(metadata.st_mode):
            source_fd = os.open(
                leaf,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            try:
                _scan_directory(
                    source_fd,
                    Path(),
                    target,
                    deadline=deadline,
                    records=records,
                    payloads=payloads,
                    counters=counters,
                    max_files=max_files,
                    max_total_bytes=max_total_bytes,
                )
            finally:
                os.close(source_fd)
            source_kind = "directory"
        else:
            raise ValueError(f"snapshot input has unsupported type: {target}")
    finally:
        os.close(parent_fd)
    manifest = {
        "format_version": "cuda-kernel-optimizer/object-manifest-v1",
        "source_kind": source_kind,
        "entries": records,
        "file_count": counters["files"],
        "total_bytes": counters["bytes"],
    }
    return manifest, payloads


def _validated_snapshot(manifest, payloads) -> tuple[dict, dict[str, bytes]]:
    """Validate one self-contained snapshot before it enters the object store."""
    if type(manifest) is not dict or set(manifest) != {
        "format_version", "source_kind", "entries", "file_count", "total_bytes"
    }:
        raise ValueError("snapshot manifest fields are incomplete or unknown")
    if manifest["format_version"] != "cuda-kernel-optimizer/object-manifest-v1":
        raise ValueError("snapshot manifest format_version is unsupported")
    if manifest["source_kind"] not in {"file", "directory"}:
        raise ValueError("snapshot manifest source_kind is unsupported")
    if type(manifest["entries"]) is not list:
        raise ValueError("snapshot manifest entries must be a list")
    if not isinstance(payloads, Mapping):
        raise ValueError("snapshot payloads must be a mapping")

    files: dict[str, bytes] = {}
    directories = set()
    total_bytes = 0
    for entry in manifest["entries"]:
        if type(entry) is not dict:
            raise ValueError("snapshot manifest entry must be an object")
        kind = entry.get("kind")
        path = _relative_artifact_path(entry.get("path"))
        key = path.as_posix()
        mode = entry.get("mode")
        if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
            raise ValueError("snapshot manifest entry mode is invalid")
        if kind == "directory":
            if (
                set(entry) != {"path", "kind", "mode"}
                or key in directories
                or key in files
            ):
                raise ValueError("snapshot directory entry fields are invalid")
            directories.add(key)
            continue
        if kind != "file" or set(entry) != {
            "path", "kind", "mode", "size_bytes", "sha256"
        }:
            raise ValueError("snapshot file entry fields are invalid")
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or key in files
            or key in directories
        ):
            raise ValueError("snapshot file entry metadata is invalid")
        try:
            payload = payloads[key]
        except KeyError as error:
            raise ValueError(f"snapshot payload is missing: {key}") from error
        if not isinstance(payload, bytes):
            raise ValueError("snapshot payload must be bytes")
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("snapshot payload does not match manifest")
        files[key] = payload
        total_bytes += size

    if set(payloads) != set(files):
        raise ValueError("snapshot payload paths do not match manifest")
    if any(not isinstance(key, str) for key in payloads):
        raise ValueError("snapshot payload paths must be strings")
    if (
        isinstance(manifest["file_count"], bool)
        or not isinstance(manifest["file_count"], int)
        or manifest["file_count"] != len(files)
        or isinstance(manifest["total_bytes"], bool)
        or not isinstance(manifest["total_bytes"], int)
        or manifest["total_bytes"] != total_bytes
    ):
        raise ValueError("snapshot manifest summary does not match entries")
    all_paths = set(files) | directories
    for key in all_paths:
        parent = Path(key).parent
        while parent != Path("."):
            parent_key = parent.as_posix()
            if parent_key not in directories:
                raise ValueError("snapshot manifest omits a parent directory")
            parent = parent.parent
    if any(
        path in files and any(other.startswith(path + "/") for other in all_paths)
        for path in files
    ):
        raise ValueError("snapshot file entry cannot contain child entries")
    if manifest["source_kind"] == "file" and (len(files) != 1 or directories):
        raise ValueError("snapshot file source must contain exactly one file")
    return manifest, files


def freeze_snapshot(artifact_root, manifest, payloads) -> dict:
    """Publish one already-captured snapshot without reading its source again."""
    manifest, payloads = _validated_snapshot(manifest, payloads)
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    digest = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    object_parent = root / "objects" / "sha256"
    object_parent.mkdir(parents=True, exist_ok=True)
    destination = object_parent / digest
    if destination.exists():
        existing = read_regular_bytes(destination / "manifest.json")
        if existing != _canonical_json_bytes(manifest) + b"\n":
            raise ValueError("content-addressed object manifest does not match")
    else:
        temporary = object_parent / f".{digest}.{secrets.token_hex(8)}.tmp"
        try:
            temporary.mkdir(mode=0o700)
            payload_root = temporary / "payload"
            payload_root.mkdir(mode=0o700)
            for record in manifest["entries"]:
                target = payload_root / record["path"]
                if record["kind"] == "directory":
                    target.mkdir(mode=record["mode"], parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                create_regular_bytes(target, payloads[record["path"]])
                os.chmod(target, record["mode"], follow_symlinks=False)
            create_regular_bytes(
                temporary / "manifest.json",
                _canonical_json_bytes(manifest) + b"\n",
            )
            try:
                publish_directory_noreplace(temporary, destination)
            except FileExistsError:
                existing = read_regular_bytes(destination / "manifest.json")
                if existing != _canonical_json_bytes(manifest) + b"\n":
                    raise ValueError(
                        "content-addressed object manifest does not match"
                    )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return {
        "digest": digest,
        "locator": str(Path("objects") / "sha256" / digest),
        "source_kind": manifest["source_kind"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def freeze_path(artifact_root, source, scan_limits) -> dict:
    """Capture a bounded source once, then publish that exact snapshot."""
    manifest, payloads = _snapshot_source(source, scan_limits)
    return freeze_snapshot(artifact_root, manifest, payloads)


def materialize_object(artifact_root, object_ref, destination) -> Path:
    """Verify and copy one frozen object into a new isolated path."""
    if type(object_ref) is not dict:
        raise ValueError("object_ref must be an object")
    required = {"digest", "locator"}
    optional = {"source_kind", "file_count", "total_bytes"}
    if not required.issubset(object_ref) or set(object_ref) - required - optional:
        raise ValueError("object_ref fields are incomplete or unknown")
    digest = object_ref["digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("object_ref.digest must be a lowercase SHA-256")
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    locator = _relative_artifact_path(object_ref["locator"])
    expected_locator = Path("objects") / "sha256" / digest
    if locator != expected_locator:
        raise ValueError("object_ref locator does not match its digest")
    object_directory = root / locator
    manifest_payload = read_regular_bytes(object_directory / "manifest.json")
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("object manifest is invalid JSON") from error
    if type(manifest) is not dict:
        raise ValueError("object manifest must be an object")
    if hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest() != digest:
        raise ValueError("object manifest digest does not match object_ref")

    raw_entries = manifest.get("entries")
    if type(raw_entries) is not list:
        raise ValueError("object manifest entries must be a list")
    payload_root = object_directory / "payload"
    payloads = {}
    for entry in raw_entries:
        if type(entry) is not dict or entry.get("kind") != "file":
            continue
        path = _relative_artifact_path(entry.get("path"))
        payloads[path.as_posix()] = read_regular_bytes(payload_root / path)
    manifest, payloads = _validated_snapshot(manifest, payloads)
    for field in optional.intersection(object_ref):
        if (
            type(manifest[field]) is not type(object_ref[field])
            or manifest[field] != object_ref[field]
        ):
            raise ValueError("object_ref summary does not match manifest")
    entries = manifest["entries"]

    target = Path(os.path.abspath(os.path.expanduser(os.fspath(destination))))
    if os.path.lexists(target):
        raise FileExistsError(f"materialization destination exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
    replacement = None
    try:
        temporary.mkdir(mode=0o700)
        directories = [entry for entry in entries if entry["kind"] == "directory"]
        for entry in sorted(directories, key=lambda item: item["path"]):
            path = _relative_artifact_path(entry["path"])
            materialized = temporary / path
            materialized.mkdir(mode=entry["mode"], parents=True, exist_ok=True)
            os.chmod(materialized, entry["mode"], follow_symlinks=False)
        for entry in (entry for entry in entries if entry["kind"] == "file"):
            path = _relative_artifact_path(entry["path"])
            materialized = temporary / path
            payload = payloads[path.as_posix()]
            materialized.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            create_regular_bytes(materialized, payload)
            os.chmod(materialized, entry["mode"], follow_symlinks=False)
        if manifest["source_kind"] == "file":
            file_entries = [entry for entry in entries if entry["kind"] == "file"]
            source = temporary / file_entries[0]["path"]
            replacement = target.parent / f".{target.name}.{secrets.token_hex(8)}.file"
            os.rename(source, replacement)
            shutil.rmtree(temporary)
            os.rename(replacement, target)
        elif manifest["source_kind"] == "directory":
            os.rename(temporary, target)
        else:
            raise ValueError("object manifest source_kind is unsupported")
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if replacement is not None:
            try:
                os.unlink(replacement)
            except FileNotFoundError:
                pass
        raise


def create_regular_bytes(path: _PathLike, payload: bytes) -> None:
    """Create one durable regular file without following any path component."""
    if not isinstance(payload, bytes):
        raise TypeError("create-once payload must be bytes")
    directory_fd, leaf, target = _open_parent_directory(path, create=True)
    descriptor = None
    created = False
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            created = True
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise FileExistsError(f"artifact already exists: {target}") from error
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(f"artifact target is a symlink or unsafe: {target}") from error
            raise
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("create-once artifact write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(directory_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(leaf, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(directory_fd)


def create_regular_json(path: _PathLike, payload: Any) -> None:
    """Create one strict, formatted JSON document exactly once."""
    try:
        encoded = (
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("JSON document is not serializable") from error
    create_regular_bytes(path, encoded)


def publish_regular_bundle(directory: _PathLike, writes, removals=()) -> dict:
    """Publish one immutable bundle generation, then CAS the current pointer."""
    if not isinstance(writes, Mapping):
        raise ValueError("artifact bundle writes must be a mapping")
    if isinstance(removals, (str, bytes, bytearray, Mapping)):
        raise ValueError("artifact bundle removals must be a sequence")
    try:
        removal_names = list(removals)
    except TypeError as error:
        raise ValueError("artifact bundle removals must be a sequence") from error
    clean_writes = []
    expected_hashes = {}
    for name, payload in writes.items():
        leaf = _validate_leaf_name(name)
        if not isinstance(payload, bytes):
            raise TypeError("artifact bundle payloads must be bytes")
        clean_writes.append((leaf, payload))
        expected_hashes[leaf] = hashlib.sha256(payload).hexdigest()
    clean_removals = [_validate_leaf_name(name) for name in removal_names]
    if len(set(clean_removals)) != len(clean_removals):
        raise ValueError("artifact bundle removals must be unique")
    if set(expected_hashes).intersection(clean_removals):
        raise ValueError("artifact bundle cannot write and remove the same file")

    directory_path = Path(
        os.path.abspath(os.path.expanduser(os.fspath(directory)))
    )
    marker = directory_path / ".publish-bundle"
    try:
        directory_fd, _leaf, _target = _open_parent_directory(
            marker, create=False
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"artifact publish directory is missing, a symlink, or unsafe: {directory_path}"
        ) from error
    if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
        os.close(directory_fd)
        raise ValueError(f"artifact publish path is not a directory: {directory_path}")
    lock_fd = None
    generations_fd = None
    generation_fd = None
    temporary_leaf = None
    try:
        lock_fd = os.open(
            ".bundle-publish.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ValueError("artifact bundle publish lock is not regular")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        pointer = _read_bundle_pointer(directory_fd, directory_path, missing_ok=True)
        if pointer is None:
            previous_generation, expected_pointer_digest = None, None
        else:
            previous_generation, expected_pointer_digest = pointer
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        lock_fd = None

        contents = {}
        if previous_generation is not None:
            generation_fd = _open_bundle_generation(
                directory_fd, previous_generation, directory_path
            )
            for leaf in os.listdir(generation_fd):
                _validate_leaf_name(leaf)
                contents[leaf] = _read_regular_leaf(
                    generation_fd,
                    leaf,
                    directory_path / "generations" / previous_generation / leaf,
                    missing_ok=False,
                )
            os.close(generation_fd)
            generation_fd = None
        for leaf in clean_removals:
            contents.pop(leaf, None)
        contents.update(clean_writes)

        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            os.mkdir("generations", 0o700, dir_fd=directory_fd)
        except FileExistsError:
            pass
        try:
            generations_fd = os.open("generations", flags, dir_fd=directory_fd)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("artifact bundle generations path is unsafe") from error
            raise
        generation = secrets.token_hex(16)
        temporary_leaf = f".{generation}.tmp"
        os.mkdir(temporary_leaf, 0o700, dir_fd=generations_fd)
        generation_fd = os.open(temporary_leaf, flags, dir_fd=generations_fd)
        for leaf, payload in sorted(contents.items()):
            target = directory_path / "generations" / temporary_leaf / leaf
            _atomic_write_leaf(generation_fd, leaf, target, payload)
            published = _read_regular_leaf(
                generation_fd, leaf, target, missing_ok=False
            )
            if hashlib.sha256(published).hexdigest() != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"published artifact does not match captured payload: {target}")
        os.fsync(generation_fd)
        os.close(generation_fd)
        generation_fd = None
        os.rename(
            temporary_leaf,
            generation,
            src_dir_fd=generations_fd,
            dst_dir_fd=generations_fd,
        )
        temporary_leaf = None
        os.fsync(generations_fd)
        compare_and_swap_ref(
            directory_path,
            "current",
            expected_pointer_digest,
            {"generation": generation},
        )
        return expected_hashes
    finally:
        if generation_fd is not None:
            os.close(generation_fd)
        if temporary_leaf is not None and generations_fd is not None:
            try:
                shutil.rmtree(directory_path / "generations" / temporary_leaf)
            except FileNotFoundError:
                pass
        if generations_fd is not None:
            os.close(generations_fd)
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


def atomic_write_json(path: _PathLike, payload: Any) -> None:
    """Atomically replace *path* with a formatted UTF-8 JSON document."""
    try:
        encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    except (ValueError, OverflowError) as error:
        raise ValueError("JSON document is not serializable") from error
    atomic_write_bytes(path, encoded)


def atomic_write_jsonl(path: _PathLike, records) -> None:
    """Atomically replace *path* with strict JSON Lines and durable metadata."""
    if isinstance(records, (str, bytes, bytearray, dict)):
        raise ValueError("JSONL records must be a sequence")
    try:
        snapshot = list(records)
    except TypeError as error:
        raise ValueError("JSONL records must be a sequence") from error
    encoded = []
    for index, record in enumerate(snapshot):
        try:
            encoded.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"JSONL record {index} is not strict JSON") from error

    document = ("\n".join(encoded) + ("\n" if encoded else "")).encode("utf-8")
    atomic_write_bytes(path, document)


def write_paired_samples(
    path: _PathLike,
    pairs,
    *,
    kind: str,
    input_hash: str,
    iteration: int,
    candidate_id,
    candidate_file: _PathLike,
    baseline_file: Optional[_PathLike] = None,
    classifier_config=None,
) -> dict:
    """Persist raw paired observations with candidate/input/iteration bindings."""
    if kind not in {"kernel", "workload"}:
        raise ValueError("paired sample kind must be kernel or workload")
    if not isinstance(input_hash, str) or not input_hash.strip():
        raise ValueError("paired sample input_hash must be non-empty")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
        raise ValueError("paired sample iteration must be positive")
    if candidate_id is None or isinstance(candidate_id, bool):
        raise ValueError("paired sample candidate_id must be non-empty")
    candidate_name = str(candidate_id).strip()
    if not candidate_name:
        raise ValueError("paired sample candidate_id must be non-empty")
    if not isinstance(classifier_config, dict) or not classifier_config:
        raise ValueError("paired sample classifier_config must be a non-empty mapping")
    try:
        classifier = json.loads(
            json.dumps(classifier_config, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("paired sample classifier_config must be strict JSON") from error
    candidate = Path(candidate_file).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("paired sample candidate must be a regular non-symlink file")
    candidate = candidate.resolve(strict=True)
    candidate_sha256 = sha256_file(candidate)
    baseline = None
    baseline_sha256 = None
    if baseline_file is not None:
        baseline = Path(baseline_file).expanduser()
        if baseline.is_symlink() or not baseline.is_file():
            raise ValueError(
                "paired sample baseline must be a regular non-symlink file"
            )
        baseline = baseline.resolve(strict=True)
        baseline_sha256 = sha256_file(baseline)
    if isinstance(pairs, (str, bytes, bytearray, dict)):
        raise ValueError("paired samples must be a sequence")
    try:
        raw_pairs = copy.deepcopy(list(pairs))
    except TypeError as error:
        raise ValueError("paired samples must be a sequence") from error
    records = [
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": kind,
            "input_hash": input_hash,
            "iteration": iteration,
            "candidate_id": candidate_name,
            "candidate_file": str(candidate),
            "candidate_sha256": candidate_sha256,
            **(
                {
                    "baseline_file": str(baseline),
                    "baseline_sha256": baseline_sha256,
                }
                if baseline is not None
                else {}
            ),
            "classifier": copy.deepcopy(classifier),
            "pair_index": index,
            "pair": pair,
        }
        for index, pair in enumerate(raw_pairs)
    ]
    target = Path(path).expanduser().absolute()
    atomic_write_jsonl(target, records)
    target = target.resolve(strict=True)
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "kind": kind,
        "path": str(target),
        "sha256": hashlib.sha256(read_regular_bytes(target)).hexdigest(),
        "pairs": len(records),
        "input_hash": input_hash,
        "iteration": iteration,
        "candidate_id": candidate_name,
        "candidate_file": str(candidate),
        "candidate_sha256": candidate_sha256,
        **(
            {
                "baseline_file": str(baseline),
                "baseline_sha256": baseline_sha256,
            }
            if baseline is not None
            else {}
        ),
        "classifier": classifier,
    }


class ArtifactStore:
    """Own the durable artifacts beneath one optimizer run directory."""

    def __init__(self, root: _PathLike) -> None:
        self.root = Path(root).expanduser().resolve()

    def initialize(
        self,
        *,
        inputs: dict,
        budget: dict,
        environment: Optional[dict] = None,
    ) -> dict:
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a dict containing baseline and ref")
        missing = [name for name in ("baseline", "ref") if name not in inputs]
        if missing:
            raise ValueError(
                "inputs must contain baseline and ref; missing: " + ", ".join(missing)
            )

        for directory in (
            self.root,
            self.root / "workload",
            self.root / "baseline",
            self.root / "candidates",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        input_records = {}
        for name, value in inputs.items():
            if not isinstance(name, str) or not name:
                raise ValueError("input keys must be non-empty strings")
            source = Path(value).expanduser().resolve()
            digest = sha256_file(source)
            input_records[name] = {
                "path": str(source),
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            }

        sha_mapping = {
            name: input_records[name]["sha256"] for name in sorted(input_records)
        }
        stable_json = json.dumps(
            sha_mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        input_hash = hashlib.sha256(stable_json.encode("utf-8")).hexdigest()
        manifest = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "inputs": input_records,
            "budget": copy.deepcopy(budget),
            "environment": copy.deepcopy(environment) if environment is not None else {},
            "input_hash": input_hash,
        }
        atomic_write_json(self.root / "manifest.json", manifest)
        return manifest

    def candidate_dir(self, candidate_id: str) -> Path:
        if (
            not isinstance(candidate_id, str)
            or candidate_id in {".", ".."}
            or not _CANDIDATE_ID.fullmatch(candidate_id)
        ):
            raise ValueError(
                "candidate_id must match [A-Za-z0-9._-]+ and cannot be '.' or '..'"
            )
        path = self._resolve_relative(Path("candidates") / candidate_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_relative(self, relative_path: _PathLike) -> Path:
        text = os.fspath(relative_path)
        if not text:
            raise ValueError("artifact path must be a non-empty relative path")
        relative = Path(text)
        if relative.is_absolute() or PureWindowsPath(text).is_absolute():
            raise ValueError(f"artifact path must be relative to {self.root}: {text}")

        target = (self.root / relative).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                f"artifact path escapes run root {self.root}: {text}"
            ) from error
        if target == self.root:
            raise ValueError(f"artifact path must name a file below {self.root}: {text}")
        return target

    def write_json(self, relative_path: _PathLike, payload: Any) -> Path:
        target = self._resolve_relative(relative_path)
        atomic_write_json(target, payload)
        return target

    def append_jsonl(self, relative_path: _PathLike, payload: Any) -> Path:
        target = self._resolve_relative(relative_path)
        line = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        directory_fd, leaf, stable_target = _open_parent_directory(
            target, create=True
        )
        fd = None
        try:
            try:
                fd = os.open(
                    leaf,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_APPEND
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o644,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
                    raise ValueError(
                        f"JSONL target is missing, a symlink, or unsafe: {stable_target}"
                    ) from error
                raise
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(
                    f"JSONL target is not a regular file: {stable_target}"
                )
            fcntl.flock(fd, fcntl.LOCK_EX)
            offset = 0
            while offset < len(line):
                offset += os.write(fd, line[offset:])
            os.fsync(fd)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.fsync(directory_fd)
            return stable_target
        finally:
            if fd is not None:
                os.close(fd)
            os.close(directory_fd)

    def read_jsonl(self, relative_path: _PathLike) -> list:
        target = self._resolve_relative(relative_path)
        if not target.exists():
            return []
        records = []
        try:
            text = read_regular_bytes(target).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"JSONL artifact is not UTF-8: {target}") from error
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {target} at line {line_number}: {error.msg}"
                ) from error
        return records

    def write_checkpoint(self, payload: dict) -> Path:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a dict")
        checkpoint = copy.deepcopy(payload)
        checkpoint["schema_version"] = CURRENT_SCHEMA_VERSION
        path = self._resolve_relative("checkpoint.json")
        atomic_write_json(path, checkpoint)
        return path

    def load_checkpoint(self, *, expected_input_hash: str) -> dict:
        path = self._resolve_relative("checkpoint.json")
        try:
            checkpoint = json.loads(read_regular_bytes(path).decode("utf-8"))
        except ValueError as error:
            if "artifact file does not exist" in str(error):
                raise ValueError(f"checkpoint not found: {path}") from error
            raise
        except UnicodeDecodeError as error:
            raise ValueError(f"checkpoint is not UTF-8: {path}") from error
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint must contain a JSON object: {path}")
        if type(checkpoint.get("schema_version")) is not int or checkpoint.get(
            "schema_version"
        ) != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version must be {CURRENT_SCHEMA_VERSION}: {path}"
            )
        if checkpoint.get("input_hash") != expected_input_hash:
            raise ValueError(
                "checkpoint does not match the frozen input; "
                f"expected {expected_input_hash!r}, got {checkpoint.get('input_hash')!r}"
            )
        return checkpoint
