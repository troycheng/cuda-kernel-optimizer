#!/usr/bin/env python3
"""Versioned, traversal-safe storage for optimizer run artifacts."""

from __future__ import annotations

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


_PathLike = Union[str, os.PathLike]


class StaleReferenceError(ValueError):
    """Raised when a compare-and-swap reference no longer matches."""


def _publish_directory_fd_noreplace(
    source_parent_fd: int,
    source_leaf: str,
    destination_parent_fd: int,
    destination_leaf: str,
    destination_path: Path,
) -> None:
    """Atomically publish a directory relative to already-open parents."""
    library = ctypes.CDLL(None, use_errno=True)
    old = os.fsencode(source_leaf)
    new = os.fsencode(destination_leaf)
    if sys.platform == "darwin":
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        returncode = function(
            source_parent_fd,
            old,
            destination_parent_fd,
            new,
            0x00000004,
        )
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
        returncode = function(
            source_parent_fd,
            old,
            destination_parent_fd,
            new,
            1,
        )
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


def publish_directory_noreplace(source: _PathLike, destination: _PathLike) -> None:
    """Atomically publish one directory and fail if the destination exists."""
    source_parent_fd, source_leaf, source_path = _open_parent_directory(
        source, create=False
    )
    destination_parent_fd = None
    try:
        source_metadata = os.stat(
            source_leaf, dir_fd=source_parent_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(source_metadata.st_mode):
            raise ValueError("published source must be a real directory")
        destination_parent_fd, destination_leaf, destination_path = (
            _open_parent_directory(destination, create=True)
        )
        _publish_directory_fd_noreplace(
            source_parent_fd,
            source_leaf,
            destination_parent_fd,
            destination_leaf,
            destination_path,
        )
        os.fsync(source_parent_fd)
        if destination_parent_fd != source_parent_fd:
            os.fsync(destination_parent_fd)
    finally:
        os.close(source_parent_fd)
        if destination_parent_fd is not None:
            os.close(destination_parent_fd)


def sha256_file(path: _PathLike) -> str:
    """Return a stable SHA-256 digest without following path symlinks."""
    try:
        directory_fd, leaf, target = _open_parent_directory(path, create=False)
    except FileNotFoundError as error:
        raise ValueError(f"artifact file does not exist: {path}") from error
    descriptor = None
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR}:
                raise ValueError(
                    f"artifact file is missing, a symlink, or unsafe: {target}"
                ) from error
            raise
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact path is not a regular file: {target}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _STREAM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or total != before.st_size:
            raise ValueError(f"artifact file changed while hashing: {target}")
        return digest.hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


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


_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_OBJECT_MANIFEST_BYTES = 16 * 1024 * 1024


def _validated_manifest(manifest) -> dict:
    """Validate canonical object metadata without loading payloads."""
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
    files: dict[str, dict] = {}
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
        files[key] = entry
        total_bytes += size

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
    return manifest


def _object_reference(object_ref) -> tuple[str, Path, dict]:
    if type(object_ref) is not dict:
        raise ValueError("object_ref must be an object")
    required = {"digest", "locator"}
    optional = {"source_kind", "file_count", "total_bytes"}
    if not required.issubset(object_ref) or set(object_ref) - required - optional:
        raise ValueError("object_ref fields are incomplete or unknown")
    digest = object_ref["digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("object_ref.digest must be a lowercase SHA-256")
    locator = _relative_artifact_path(object_ref["locator"])
    expected_locator = Path("objects") / "sha256" / digest
    if locator != expected_locator:
        raise ValueError("object_ref locator does not match its digest")
    return digest, locator, {field: object_ref[field] for field in optional if field in object_ref}


def _safe_directory(path: Path) -> None:
    directory_fd, _leaf, _target = _open_parent_directory(
        path / ".directory-probe", create=True
    )
    try:
        metadata = os.fstat(directory_fd)
    finally:
        os.close(directory_fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"object store path is not a safe directory: {path}")


def _copy_source_file(
    directory_fd: int,
    name: str,
    destination: Path,
    display_path: Path,
    *,
    deadline: float,
    remaining_bytes: int,
) -> dict:
    if time.monotonic() >= deadline:
        raise TimeoutError("artifact scan exceeded max_wall_seconds")
    source_fd = destination_fd = None
    try:
        source_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"snapshot input is not a regular file: {display_path}")
        if before.st_size > remaining_bytes:
            raise ValueError("artifact scan exceeded max_total_bytes")
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("artifact scan exceeded max_wall_seconds")
            chunk = os.read(source_fd, _STREAM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > remaining_bytes:
                raise ValueError("artifact scan exceeded max_total_bytes")
            digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_fd, chunk[offset:])
                if written <= 0:
                    raise OSError("object staging write made no progress")
                offset += written
        after = os.fstat(source_fd)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or total != before.st_size:
            raise ValueError(f"snapshot input changed while reading: {display_path}")
        os.fchmod(destination_fd, stat.S_IMODE(before.st_mode))
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = None
        return {
            "mode": stat.S_IMODE(before.st_mode),
            "size_bytes": total,
            "sha256": digest.hexdigest(),
        }
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _scan_directory_stream(
    directory_fd: int,
    relative: Path,
    display_root: Path,
    payload_root: Path,
    *,
    deadline: float,
    records: list[dict],
    counters: dict[str, int],
    max_files: int,
    max_total_bytes: int,
) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("artifact scan exceeded max_wall_seconds")
    before = os.fstat(directory_fd)
    names = sorted(name for name in os.listdir(directory_fd) if name not in {".", ".."})
    for name in names:
        child_relative = relative / name
        display_path = display_root / child_relative
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"snapshot input contains a symlink: {display_path}")
        if stat.S_ISDIR(metadata.st_mode):
            target = payload_root / child_relative
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
            records.append({
                "path": child_relative.as_posix(),
                "kind": "directory",
                "mode": stat.S_IMODE(metadata.st_mode),
            })
            child_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(child_fd)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise ValueError(f"snapshot directory changed while opening: {display_path}")
                _scan_directory_stream(
                    child_fd, child_relative, display_root, payload_root,
                    deadline=deadline, records=records, counters=counters,
                    max_files=max_files, max_total_bytes=max_total_bytes,
                )
            finally:
                os.close(child_fd)
            os.chmod(target, stat.S_IMODE(metadata.st_mode), follow_symlinks=False)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"snapshot input contains an unsupported entry: {display_path}")
        counters["files"] += 1
        if counters["files"] > max_files:
            raise ValueError("artifact scan exceeded max_files")
        details = _copy_source_file(
            directory_fd, name, payload_root / child_relative, display_path,
            deadline=deadline,
            remaining_bytes=max_total_bytes - counters["bytes"],
        )
        counters["bytes"] += details["size_bytes"]
        records.append({
            "path": child_relative.as_posix(), "kind": "file", **details,
        })
    after = os.fstat(directory_fd)
    if (
        before.st_dev, before.st_ino, before.st_mtime_ns, names
    ) != (
        after.st_dev, after.st_ino, after.st_mtime_ns,
        sorted(name for name in os.listdir(directory_fd) if name not in {".", ".."}),
    ):
        raise ValueError(f"snapshot directory changed while reading: {display_root / relative}")


def _hash_object_member(payload_root: Path, entry: dict) -> None:
    relative = _relative_artifact_path(entry["path"])
    parent_fd, leaf, target = _open_parent_directory(payload_root / relative, create=False)
    descriptor = None
    try:
        descriptor = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != entry["size_bytes"]:
            raise ValueError("object payload does not match manifest")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _STREAM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or total != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("object payload does not match manifest")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _verify_payload_root(
    payload_root: Path, manifest: dict, *, verify_hashes: bool = True
) -> None:
    expected_files = {
        entry["path"] for entry in manifest["entries"] if entry["kind"] == "file"
    }
    expected_directories = {
        entry["path"]
        for entry in manifest["entries"]
        if entry["kind"] == "directory"
    }
    actual_files = set()
    actual_directories = set()
    root_metadata = os.lstat(payload_root)
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("object payload root is unsafe")
    for directory, names, filenames in os.walk(payload_root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("object payload contains an unsafe directory")
            actual_directories.add(path.relative_to(payload_root).as_posix())
        for name in filenames:
            path = directory_path / name
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("object payload contains an unsafe file")
            actual_files.add(path.relative_to(payload_root).as_posix())
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("object payload paths do not match manifest")
    if verify_hashes:
        for entry in manifest["entries"]:
            if entry["kind"] == "file":
                _hash_object_member(payload_root, entry)


def _fsync_directory_tree(root: Path) -> None:
    directories = [Path(directory) for directory, _names, _files in os.walk(root)]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _load_object_manifest(artifact_root, object_ref, *, verify_payload: bool) -> dict:
    digest, locator, summaries = _object_reference(object_ref)
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    object_directory = root / locator
    manifest_payload = read_regular_bytes(
        object_directory / "manifest.json", maximum_bytes=_MAX_OBJECT_MANIFEST_BYTES
    )
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("object manifest is invalid JSON") from error
    if type(manifest) is not dict:
        raise ValueError("object manifest must be an object")
    if hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest() != digest:
        raise ValueError("object manifest digest does not match object_ref")

    manifest = _validated_manifest(manifest)
    for field, expected in summaries.items():
        if type(manifest[field]) is not type(expected) or manifest[field] != expected:
            raise ValueError("object_ref summary does not match manifest")
    if verify_payload:
        payload_root = object_directory / "payload"
        _verify_payload_root(payload_root, manifest)
    return manifest


def freeze_path(artifact_root, source, scan_limits) -> dict:
    """Stream one bounded source into the content-addressed object store."""
    max_files, max_total_bytes, max_wall_seconds = _scan_limits(scan_limits)
    deadline = time.monotonic() + max_wall_seconds
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    object_parent = root / "objects" / "sha256"
    _safe_directory(object_parent)
    temporary = object_parent / f".capture-{secrets.token_hex(8)}.tmp"
    temporary.mkdir(mode=0o700)
    payload_root = temporary / "payload"
    payload_root.mkdir(mode=0o700)
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(source))))
    records: list[dict] = []
    counters = {"files": 0, "bytes": 0}
    try:
        parent_fd, leaf, _target = _open_parent_directory(target, create=False)
        try:
            metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"snapshot input is a symlink: {target}")
            if stat.S_ISREG(metadata.st_mode):
                details = _copy_source_file(
                    parent_fd, leaf, payload_root / target.name, target,
                    deadline=deadline, remaining_bytes=max_total_bytes,
                )
                counters = {"files": 1, "bytes": details["size_bytes"]}
                if counters["files"] > max_files:
                    raise ValueError("artifact scan exceeded max_files")
                records.append({"path": target.name, "kind": "file", **details})
                source_kind = "file"
            elif stat.S_ISDIR(metadata.st_mode):
                source_fd = os.open(
                    leaf,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    opened = os.fstat(source_fd)
                    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                        raise ValueError(f"snapshot directory changed while opening: {target}")
                    _scan_directory_stream(
                        source_fd, Path(), target, payload_root,
                        deadline=deadline, records=records, counters=counters,
                        max_files=max_files, max_total_bytes=max_total_bytes,
                    )
                finally:
                    os.close(source_fd)
                source_kind = "directory"
            else:
                raise ValueError(f"snapshot input has unsupported type: {target}")
        finally:
            os.close(parent_fd)
        manifest = _validated_manifest({
            "format_version": "cuda-kernel-optimizer/object-manifest-v1",
            "source_kind": source_kind,
            "entries": records,
            "file_count": counters["files"],
            "total_bytes": counters["bytes"],
        })
        digest = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
        create_regular_bytes(temporary / "manifest.json", _canonical_json_bytes(manifest) + b"\n")
        _verify_payload_root(payload_root, manifest)
        _fsync_directory_tree(temporary)
        destination = object_parent / digest
        object_ref = {
            "digest": digest,
            "locator": str(Path("objects") / "sha256" / digest),
            "source_kind": source_kind,
            "file_count": counters["files"],
            "total_bytes": counters["bytes"],
        }
        try:
            publish_directory_noreplace(temporary, destination)
        except FileExistsError:
            _load_object_manifest(root, object_ref, verify_payload=True)
        return object_ref
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _open_relative_directory(
    root_fd: int, relative_path: _PathLike, *, create: bool
) -> int:
    """Open one relative directory chain without following any component."""
    raw = os.fspath(relative_path)
    if raw in {"", "."}:
        return os.dup(root_fd)
    relative = _relative_artifact_path(raw)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = os.dup(root_fd)
    try:
        for component in relative.parts:
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        "materialization path contains a symlink or non-directory"
                    ) from error
                raise
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _require_missing_leaf(parent_fd: int, leaf: str, target: Path) -> None:
    try:
        os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(f"materialization destination exists: {target}")


def _same_open_directory(parent_fd: int, leaf: str, directory_fd: int) -> bool:
    try:
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    opened = os.fstat(directory_fd)
    return (
        stat.S_ISDIR(current.st_mode)
        and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
    )


def _clear_open_directory(directory_fd: int) -> None:
    """Remove children through a stable directory descriptor without following links."""
    os.fchmod(directory_fd, 0o700)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for leaf in os.listdir(directory_fd):
        metadata = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = os.open(leaf, flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                if (metadata.st_dev, metadata.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    raise ValueError(
                        "materialization directory changed during cleanup"
                    )
                os.fchmod(child_fd, 0o700)
                _clear_open_directory(child_fd)
                if _same_open_directory(directory_fd, leaf, child_fd):
                    os.rmdir(leaf, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        else:
            os.unlink(leaf, dir_fd=directory_fd)


def _remove_open_directory(parent_fd: int, leaf: str, directory_fd: int) -> None:
    _clear_open_directory(directory_fd)
    if _same_open_directory(parent_fd, leaf, directory_fd):
        os.rmdir(leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)


def _remove_directory_nofollow(path: _PathLike) -> None:
    parent_fd, leaf, _target = _open_parent_directory(path, create=False)
    directory_fd = None
    try:
        directory_fd = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        _remove_open_directory(parent_fd, leaf, directory_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def _copy_frozen_member(
    payload_root: Path,
    entry: dict,
    destination_parent_fd: int,
    destination_leaf: str,
) -> None:
    source_path = payload_root / _relative_artifact_path(entry["path"])
    source_parent_fd, source_leaf, _ = _open_parent_directory(source_path, create=False)
    source_fd = destination_fd = None
    try:
        source_fd = os.open(source_leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=source_parent_fd)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != entry["size_bytes"]:
            raise ValueError("object payload does not match manifest")
        destination_fd = os.open(
            destination_leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=destination_parent_fd,
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_fd, _STREAM_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_fd, chunk[offset:])
                if written <= 0:
                    raise OSError("object materialization write made no progress")
                offset += written
        after = os.fstat(source_fd)
        if (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or total != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("object payload does not match manifest")
        os.fchmod(destination_fd, entry["mode"])
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = None
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_parent_fd)


def _publish_materialized_file_fd(
    source_parent_fd: int,
    source_leaf: str,
    target_parent_fd: int,
    target_leaf: str,
    target_path: Path,
) -> None:
    try:
        os.link(
            source_leaf,
            target_leaf,
            src_dir_fd=source_parent_fd,
            dst_dir_fd=target_parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"materialization destination exists: {target_path}"
        ) from error
    os.fsync(target_parent_fd)


def materialize_object(artifact_root, object_ref, destination) -> Path:
    """Verify and stream one frozen object into a new isolated path."""
    manifest = _load_object_manifest(artifact_root, object_ref, verify_payload=False)
    digest, locator, _summaries = _object_reference(object_ref)
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    payload_root = root / locator / "payload"
    entries = manifest["entries"]
    _verify_payload_root(payload_root, manifest, verify_hashes=False)

    target_parent_fd = temporary_fd = None
    temporary_leaf = None
    directory_published = False
    try:
        target_parent_fd, target_leaf, target = _open_parent_directory(
            destination, create=True
        )
        _require_missing_leaf(target_parent_fd, target_leaf, target)
        temporary_leaf = f".{target_leaf}.{secrets.token_hex(8)}.tmp"
        os.mkdir(temporary_leaf, 0o700, dir_fd=target_parent_fd)
        temporary_fd = os.open(
            temporary_leaf,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=target_parent_fd,
        )
        directories = [entry for entry in entries if entry["kind"] == "directory"]
        for entry in sorted(directories, key=lambda item: item["path"]):
            directory_fd = _open_relative_directory(
                temporary_fd, entry["path"], create=True
            )
            os.close(directory_fd)
        for entry in (entry for entry in entries if entry["kind"] == "file"):
            path = _relative_artifact_path(entry["path"])
            parent_fd = _open_relative_directory(
                temporary_fd, path.parent, create=False
            )
            try:
                _copy_frozen_member(payload_root, entry, parent_fd, path.name)
            finally:
                os.close(parent_fd)
        for entry in sorted(
            directories,
            key=lambda item: len(Path(item["path"]).parts),
            reverse=True,
        ):
            directory_fd = _open_relative_directory(
                temporary_fd, entry["path"], create=False
            )
            try:
                os.fchmod(directory_fd, entry["mode"])
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        os.fsync(temporary_fd)
        if manifest["source_kind"] == "file":
            file_entries = [entry for entry in entries if entry["kind"] == "file"]
            path = _relative_artifact_path(file_entries[0]["path"])
            source_parent_fd = _open_relative_directory(
                temporary_fd, path.parent, create=False
            )
            try:
                _publish_materialized_file_fd(
                    source_parent_fd,
                    path.name,
                    target_parent_fd,
                    target_leaf,
                    target,
                )
            finally:
                os.close(source_parent_fd)
        elif manifest["source_kind"] == "directory":
            _publish_directory_fd_noreplace(
                target_parent_fd,
                temporary_leaf,
                target_parent_fd,
                target_leaf,
                target,
            )
            directory_published = True
            os.fsync(target_parent_fd)
        else:
            raise ValueError("object manifest source_kind is unsupported")
        return target
    finally:
        try:
            if temporary_fd is not None:
                try:
                    if not directory_published:
                        _remove_open_directory(
                            target_parent_fd, temporary_leaf, temporary_fd
                        )
                finally:
                    os.close(temporary_fd)
        finally:
            if target_parent_fd is not None:
                os.close(target_parent_fd)


def materialize_object_member(artifact_root, object_ref, relative_path, destination) -> dict:
    """Stream one exact regular member from a verified frozen object."""
    manifest = _load_object_manifest(artifact_root, object_ref, verify_payload=False)
    relative = _relative_artifact_path(relative_path).as_posix()
    matches = [entry for entry in manifest["entries"] if entry["kind"] == "file" and entry["path"] == relative]
    if len(matches) != 1:
        raise ValueError("object member is not one exact regular file")
    _digest, locator, _summaries = _object_reference(object_ref)
    target_parent_fd = temporary_fd = None
    temporary_leaf = None
    try:
        target_parent_fd, target_leaf, target = _open_parent_directory(
            destination, create=True
        )
        _require_missing_leaf(target_parent_fd, target_leaf, target)
        temporary_leaf = f".{target_leaf}.{secrets.token_hex(8)}.tmp"
        os.mkdir(temporary_leaf, 0o700, dir_fd=target_parent_fd)
        temporary_fd = os.open(
            temporary_leaf,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=target_parent_fd,
        )
        _copy_frozen_member(
            Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root)))) / locator / "payload",
            matches[0],
            temporary_fd,
            "member",
        )
        os.fsync(temporary_fd)
        _publish_materialized_file_fd(
            temporary_fd,
            "member",
            target_parent_fd,
            target_leaf,
            target,
        )
        return dict(matches[0])
    finally:
        try:
            if temporary_fd is not None:
                try:
                    _remove_open_directory(
                        target_parent_fd, temporary_leaf, temporary_fd
                    )
                finally:
                    os.close(temporary_fd)
        finally:
            if target_parent_fd is not None:
                os.close(target_parent_fd)


def _promote_staged_object(artifact_root, staging_root, object_ref) -> dict:
    """Atomically promote one verified object from an artifact-root staging area."""
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    staging = Path(os.path.abspath(os.path.expanduser(os.fspath(staging_root))))
    expected_parent = root / ".staging"
    if staging.parent != expected_parent:
        raise ValueError("staged object root is outside the private staging area")
    _load_object_manifest(staging, object_ref, verify_payload=True)
    _digest, locator, _summaries = _object_reference(object_ref)
    object_parent = root / "objects" / "sha256"
    _safe_directory(object_parent)
    source = staging / locator
    destination = root / locator
    published = True
    try:
        publish_directory_noreplace(source, destination)
    except FileExistsError:
        _load_object_manifest(root, object_ref, verify_payload=True)
        published = False
    return {"object_ref": dict(object_ref), "published": published}


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
