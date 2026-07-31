#!/usr/bin/env python3
"""Reliable execution primitives for one V1.4 invocation.

This module owns process creation, process-group cleanup, shared GPU locks and
the small guardian/worker transport.  It never interprets optimization facts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import select
import secrets
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
import stat


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runtime sibling: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_invocation_store")
_MAX_RECORD_BYTES = 4 * 1024 * 1024


_COMMAND_FIELDS = {
    "argv",
    "cwd",
    "env",
    "output_limit_bytes",
    "required_gpu_uuids",
}
_LIMIT_FIELDS = {
    "operation_timeout_seconds",
    "command_timeout_seconds",
    "resource_wait_timeout_seconds",
    "cleanup_timeout_seconds",
}
_RESOURCE_FIELDS = {"host_id", "gpu_uuids"}
_REQUEST_DIGEST_FIELDS_IGNORED_BY_RUNTIME = {
    "operation_timeout_seconds",
    "command_timeout_seconds",
    "resource_wait_timeout_seconds",
    "cleanup_timeout_seconds",
    "launch_deadline",
    "absolute_deadline",
    "retry_of",
    "request_digest",
}
_WORKER_REQUEST_FD = "CKO_WORKER_REQUEST_FD"
_WORKER_RESPONSE_FD = "CKO_WORKER_RESPONSE_FD"
_INVOCATION_DIR = "CKO_INVOCATION_DIR"
_ARTIFACT_ROOT = "CKO_ARTIFACT_ROOT"
_INVOCATION_ID = "CKO_INVOCATION_ID"
_WORKER_GATE_FD = "CKO_WORKER_GATE_FD"


def _closed(value, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    return value


def _positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _validate_gpu_uuids(value, label: str) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise ValueError(f"{label} must be a string list")
    normalized = list(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError(f"{label} must be a string list")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must not contain duplicates")
    return sorted(normalized)


def _validate_command(value) -> dict:
    command = _closed(value, _COMMAND_FIELDS, "command_spec")
    argv = command["argv"]
    if isinstance(argv, (str, bytes, bytearray)) or not isinstance(argv, Sequence):
        raise ValueError("command_spec.argv must be a non-empty string list")
    normalized_argv = list(argv)
    if not normalized_argv or any(
        not isinstance(item, str) or not item for item in normalized_argv
    ):
        raise ValueError("command_spec.argv must be a non-empty string list")
    cwd = command["cwd"]
    if not isinstance(cwd, str) or not os.path.isabs(cwd):
        raise ValueError("command_spec.cwd must be an absolute path")
    environment = command["env"]
    if type(environment) is not dict or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        for key, item in environment.items()
    ):
        raise ValueError("command_spec.env must map non-empty strings to strings")
    output_limit = command["output_limit_bytes"]
    if (
        isinstance(output_limit, bool)
        or not isinstance(output_limit, int)
        or output_limit < 1
    ):
        raise ValueError("command_spec.output_limit_bytes must be a positive integer")
    gpu_uuids = _validate_gpu_uuids(
        command["required_gpu_uuids"],
        "command_spec.required_gpu_uuids",
    )
    return {
        "argv": normalized_argv,
        "cwd": cwd,
        "env": dict(environment),
        "output_limit_bytes": output_limit,
        "required_gpu_uuids": gpu_uuids,
    }


def _validate_limits(value) -> dict:
    limits = _closed(value, _LIMIT_FIELDS, "runtime_limits")
    return {
        field: _positive_number(limits[field], f"runtime_limits.{field}")
        for field in sorted(_LIMIT_FIELDS)
    }


def _validate_resources(value) -> dict:
    resources = _closed(value, _RESOURCE_FIELDS, "resources")
    host_id = resources["host_id"]
    if not isinstance(host_id, str) or not host_id:
        raise ValueError("resources.host_id must be a non-empty string")
    normalized = _validate_gpu_uuids(
        resources["gpu_uuids"],
        "resources.gpu_uuids",
    )
    return {"host_id": host_id, "gpu_uuids": normalized}


def _strict_json_bytes(value) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invocation value must be finite JSON") from error


def _is_regular_file(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode)


def _regular_file_or_absent(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"runtime record is not a regular file: {path}")
    return True


def _is_real_directory(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(metadata.st_mode)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(STORE.read_regular_bytes(path, maximum_bytes=_MAX_RECORD_BYTES).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON document is invalid: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"JSON document must contain an object: {path}")
    return value


def _atomic_json(path: Path, value) -> None:
    STORE.atomic_write_bytes(path, _strict_json_bytes(value) + b"\n")


def _create_json(path: Path, value) -> None:
    STORE.create_regular_bytes(path, _strict_json_bytes(value) + b"\n")


def _remove_record(path: Path) -> None:
    STORE.remove_regular_file(path, missing_ok=True)


def request_digest(value: Mapping) -> str:
    """Return the semantic digest after excluding runtime-only limits."""
    if not isinstance(value, Mapping):
        raise ValueError("invocation request must be an object")
    semantic = {
        key: item
        for key, item in value.items()
        if key not in _REQUEST_DIGEST_FIELDS_IGNORED_BY_RUNTIME
    }
    return hashlib.sha256(_strict_json_bytes(semantic)).hexdigest()


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group_gone(process_group: int, deadline: float) -> bool:
    while _group_exists(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _terminate_group(
    process: subprocess.Popen,
    process_group: int,
    cleanup_timeout_seconds: float,
) -> str:
    deadline = time.monotonic() + cleanup_timeout_seconds
    if process.poll() is None or _group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    graceful_deadline = min(deadline, time.monotonic() + 0.25)
    try:
        process.wait(timeout=max(0.0, graceful_deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pass
    if _group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    remaining = max(0.0, deadline - time.monotonic())
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return "unconfirmed"
    return "confirmed" if _wait_group_gone(process_group, deadline) else "unconfirmed"


def _read_prefix(stream, limit: int) -> str:
    stream.seek(0)
    payload = stream.read(limit + 1)
    truncated = len(payload) > limit
    payload = payload[:limit]
    text = payload.decode("utf-8", errors="replace")
    return text + ("\n[output truncated]" if truncated else "")


def _lock_root() -> Path:
    root = Path.home() / ".cache" / "cuda-kernel-optimizer" / "locks"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _resource_lock_path(root: Path, host_id: str, gpu_uuid: str) -> Path:
    key = hashlib.sha256((host_id + "\0" + gpu_uuid).encode("utf-8")).hexdigest()
    return root / f"gpu-{key}.lock"


def _occupancy_path(root: Path, host_id: str, gpu_uuid: str) -> Path:
    key = hashlib.sha256((host_id + "\0" + gpu_uuid).encode("utf-8")).hexdigest()
    return root / f"gpu-{key}.occupancy.json"


def _command_resources(resources: dict, command: dict) -> dict:
    required = set(command["required_gpu_uuids"])
    declared = set(resources["gpu_uuids"])
    if not required.issubset(declared):
        raise ValueError("child command requests GPUs not declared by invocation")
    return {"host_id": resources["host_id"], "gpu_uuids": sorted(required)}


def _clear_occupancies(lock_root: Path, resources: dict) -> None:
    for gpu_uuid in resources["gpu_uuids"]:
        _remove_record(_occupancy_path(lock_root, resources["host_id"], gpu_uuid))


def _recover_occupancies(lock_root: Path, resources: dict) -> bool:
    """Clear only credentials whose local process group is provably gone."""
    for gpu_uuid in resources["gpu_uuids"]:
        path = _occupancy_path(lock_root, resources["host_id"], gpu_uuid)
        try:
            present = _regular_file_or_absent(path)
        except ValueError:
            return False
        if not present:
            continue
        try:
            record = _read_json(path)
            pid = record["child_pid"]
            token = record["child_start_token"]
            process_group = record["process_group"]
        except (OSError, KeyError, ValueError, TypeError):
            return False
        if _process_matches(pid, token):
            try:
                os.killpg(process_group, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            if not _wait_group_gone(process_group, time.monotonic() + 1.0):
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            if not _wait_group_gone(process_group, time.monotonic() + 1.0):
                return False
        elif _group_exists(process_group):
            return False
        _remove_record(path)
    return True


def _recover_active_child(
    tracking_dir: Path,
    cleanup_timeout_seconds: float,
) -> bool:
    path = tracking_dir / "active-child.json"
    try:
        present = _regular_file_or_absent(path)
    except ValueError:
        return False
    if not present:
        return True
    try:
        record = _read_json(path)
        pid = record["child_pid"]
        token = record["child_start_token"]
        process_group = record["process_group"]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    if _process_matches(pid, token):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + cleanup_timeout_seconds
        if not _wait_group_gone(process_group, min(deadline, time.monotonic() + 0.25)):
            try:
                os.killpg(process_group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if not _wait_group_gone(process_group, deadline):
            return False
    elif _group_exists(process_group):
        return False
    _remove_record(path)
    return True


def _write_occupancies(
    lock_root: Path, resources: dict, *, invocation_dir: Path, process_group: int, child_pid: int
) -> None:
    record = {
        "invocation_id": invocation_dir.name,
        "child_pid": child_pid,
        "child_start_token": _process_start_token(child_pid),
        "process_group": process_group,
        "resources": resources,
    }
    for gpu_uuid in resources["gpu_uuids"]:
        _create_json(_occupancy_path(lock_root, resources["host_id"], gpu_uuid), record)


def _release_locks(descriptors: list[int]) -> None:
    for descriptor in reversed(descriptors):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _acquire_resource_locks(resources: dict, deadline: float, cancel_path=None):
    if not resources["gpu_uuids"]:
        return []
    root = _lock_root()
    descriptors = []
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for gpu_uuid in resources["gpu_uuids"]:
            descriptor = os.open(
                _resource_lock_path(root, resources["host_id"], gpu_uuid),
                flags,
                0o600,
            )
            while True:
                if cancel_path is not None and _regular_file_or_absent(Path(cancel_path)):
                    os.close(descriptor)
                    _release_locks(descriptors)
                    return "cancelled"
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    descriptors.append(descriptor)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        os.close(descriptor)
                        _release_locks(descriptors)
                        return None
                    time.sleep(0.01)
        return descriptors
    except BaseException:
        _release_locks(descriptors)
        raise


def _command_deadline(
    *,
    operation_deadline: float,
    command_timeout_seconds: float,
    absolute_deadline: float | None,
) -> tuple[float, str]:
    now_mono = time.monotonic()
    deadlines = [
        (operation_deadline, "operation_timeout"),
        (now_mono + command_timeout_seconds, "command_timeout"),
    ]
    if absolute_deadline is not None:
        remaining = absolute_deadline - time.time()
        deadlines.append((now_mono + max(0.0, remaining), "absolute_deadline"))
    return min(deadlines, key=lambda item: item[0])


def _execute_validated(
    command: dict,
    *,
    resources: dict,
    operation_started: float,
    operation_deadline: float,
    limits: dict,
    absolute_deadline: float | None,
    cancel_path=None,
    tracking_dir: Path | None = None,
    owner_fd: int | None = None,
) -> dict:
    deadline, timeout_reason = _command_deadline(
        operation_deadline=operation_deadline,
        command_timeout_seconds=limits["command_timeout_seconds"],
        absolute_deadline=absolute_deadline,
    )
    if deadline <= time.monotonic():
        return {
            "status": "timed_out",
            "stop_reason": timeout_reason,
            "returncode": None,
            "elapsed_seconds": time.monotonic() - operation_started,
            "cleanup_status": "confirmed",
            "stdout": "",
            "stderr": "",
        }

    environment = os.environ.copy()
    environment.update(command["env"])
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(command["required_gpu_uuids"])
    if tracking_dir is None:
        raise ValueError("tracked child execution requires a tracking directory")
    child_token = secrets.token_hex(8)
    child_argv_path = tracking_dir / f".child-{child_token}.json"
    _create_json(child_argv_path, {"argv": command["argv"]})
    gate_read, gate_write = os.pipe()
    environment[_WORKER_GATE_FD] = str(gate_read)
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_child_gate",
                    "--invocation-dir",
                    str(tracking_dir),
                    "--artifact-root",
                    str(_lock_root()),
                    "--worker-argv-file",
                    str(child_argv_path),
                ],
                cwd=command["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                pass_fds=(gate_read,),
            )
        finally:
            os.close(gate_read)
        process_group = process.pid
        try:
            _atomic_json(
                tracking_dir / "active-child.json",
                {
                    "child_pid": process.pid,
                    "child_start_token": _process_start_token(process.pid),
                    "process_group": process_group,
                    "required_gpu_uuids": command["required_gpu_uuids"],
                },
            )
            if command["required_gpu_uuids"]:
                _write_occupancies(
                    _lock_root(),
                    resources,
                    invocation_dir=tracking_dir,
                    process_group=process_group,
                    child_pid=process.pid,
                )
            os.write(gate_write, b"1")
        except BaseException:
            os.close(gate_write)
            _terminate_group(
                process,
                process_group,
                limits["cleanup_timeout_seconds"],
            )
            _remove_record(child_argv_path)
            raise
        os.close(gate_write)
        stop_reason = None
        while process.poll() is None:
            if owner_fd is not None:
                readable, _, _ = select.select([owner_fd], [], [], 0)
                if readable and not os.read(owner_fd, 1):
                    stop_reason = "owner_lost"
                    break
            if cancel_path is not None and _regular_file_or_absent(Path(cancel_path)):
                stop_reason = "cancelled"
                break
            if time.monotonic() >= deadline:
                stop_reason = timeout_reason
                break
            time.sleep(min(0.02, max(0.001, deadline - time.monotonic())))

        if stop_reason is not None:
            cleanup_status = _terminate_group(
                process,
                process_group,
                limits["cleanup_timeout_seconds"],
            )
            if cleanup_status == "confirmed" and command["required_gpu_uuids"]:
                _clear_occupancies(_lock_root(), resources)
            response = {
                "status": (
                    "cancelled"
                    if stop_reason in {"cancelled", "owner_lost"}
                    else "timed_out"
                ),
                "stop_reason": stop_reason,
                "returncode": process.returncode,
                "elapsed_seconds": time.monotonic() - operation_started,
                "cleanup_status": cleanup_status,
                "stdout": _read_prefix(stdout, command["output_limit_bytes"]),
                "stderr": _read_prefix(stderr, command["output_limit_bytes"]),
            }
        else:
            returncode = process.returncode
            cleanup_status = (
                "confirmed"
                if not _group_exists(process_group)
                else _terminate_group(
                    process,
                    process_group,
                    limits["cleanup_timeout_seconds"],
                )
            )
            if cleanup_status == "confirmed" and command["required_gpu_uuids"]:
                _clear_occupancies(_lock_root(), resources)
            response = {
                "status": "completed" if returncode == 0 else "failed",
                "stop_reason": "completed" if returncode == 0 else "command_failed",
                "returncode": returncode,
                "elapsed_seconds": time.monotonic() - operation_started,
                "cleanup_status": cleanup_status,
                "stdout": _read_prefix(stdout, command["output_limit_bytes"]),
                "stderr": _read_prefix(stderr, command["output_limit_bytes"]),
            }
        if response["cleanup_status"] == "confirmed":
            _remove_record(tracking_dir / "active-child.json")
        _remove_record(child_argv_path)
        return response


def probe(command_spec, runtime_limits, resources) -> dict:
    """Run one synchronous, bounded readiness command through a guardian."""
    command = _validate_command(command_spec)
    limits = _validate_limits(runtime_limits)
    claimed_resources = _validate_resources(resources)
    probe_dir = Path(tempfile.mkdtemp(prefix="cko-probe-"))
    owner_read, owner_write = os.pipe()
    try:
        _create_json(
            probe_dir / "probe.json",
            {"command": command, "limits": limits, "resources": claimed_resources},
        )
        guardian = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_probe_guardian",
                "--invocation-dir",
                str(probe_dir),
                "--artifact-root",
                str(_lock_root()),
                "--owner-fd",
                str(owner_read),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            pass_fds=(owner_read,),
        )
        os.close(owner_read)
        owner_read = None
        guardian_timed_out = _wait_probe_guardian(
            guardian,
            owner_write,
            operation_timeout_seconds=limits["operation_timeout_seconds"],
            cleanup_timeout_seconds=limits["cleanup_timeout_seconds"],
        )
        if guardian_timed_out:
            owner_write = None
        result_path = probe_dir / "result.json"
        if _regular_file_or_absent(result_path):
            return _read_json(result_path)
        if guardian_timed_out:
            cleaned = _recover_active_child(
                probe_dir,
                limits["cleanup_timeout_seconds"],
            )
            return {
                "status": "timed_out",
                "stop_reason": "probe_guardian_timeout",
                "returncode": guardian.returncode,
                "elapsed_seconds": limits["operation_timeout_seconds"],
                "cleanup_status": "confirmed" if cleaned else "unknown",
                "stdout": "",
                "stderr": "",
            }
        if not _regular_file_or_absent(result_path):
            cleaned = _recover_active_child(
                probe_dir,
                limits["cleanup_timeout_seconds"],
            )
            return {
                "status": "failed",
                "stop_reason": "guardian_lost",
                "returncode": guardian.returncode,
                "elapsed_seconds": 0.0,
                "cleanup_status": "confirmed" if cleaned else "unknown",
                "stdout": "",
                "stderr": "",
            }
        return _read_json(result_path)
    finally:
        if owner_read is not None:
            os.close(owner_read)
        if owner_write is not None:
            os.close(owner_write)
        shutil.rmtree(probe_dir, ignore_errors=True)


def _wait_probe_guardian(
    guardian,
    owner_write: int,
    *,
    operation_timeout_seconds: float,
    cleanup_timeout_seconds: float,
) -> bool:
    """Bound guardian completion and make caller loss trigger child cleanup."""
    try:
        guardian.wait(timeout=operation_timeout_seconds)
        return False
    except subprocess.TimeoutExpired:
        os.close(owner_write)
        cleanup_started = time.monotonic()
        try:
            guardian.wait(timeout=cleanup_timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_group(
                guardian,
                guardian.pid,
                max(0.0, cleanup_timeout_seconds - (time.monotonic() - cleanup_started)),
            )
        return True


def _append_event(invocation_dir: Path, event: str, **details) -> None:
    path = invocation_dir / "events.jsonl"
    record = {
        "event": event,
        "at_epoch": time.time(),
        **details,
    }
    payload = (_strict_json_bytes(record) + b"\n")
    STORE.append_regular_bytes(path, payload)


def _process_start_token(pid: int) -> str | None:
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        raw = proc_stat.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        raw = None
    if raw is not None:
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        if len(fields) > 19:
            return "linux:" + fields[19]
    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return None if not value else "ps:" + value


def _process_matches(pid, token) -> bool:
    return (
        isinstance(pid, int)
        and pid > 0
        and isinstance(token, str)
        and _process_start_token(pid) == token
    )


def _limits_from_request(request: Mapping) -> dict:
    return _validate_limits(
        {
            field: request[field]
            for field in _LIMIT_FIELDS
        }
    )


def _resources_from_request(request: Mapping) -> dict:
    return _validate_resources(request["resources"])


def _result_ref(invocation_id: str, result_path: Path) -> dict:
    digest = hashlib.sha256(
        STORE.read_regular_bytes(result_path, maximum_bytes=_MAX_RECORD_BYTES)
    ).hexdigest()
    return {"invocation_id": invocation_id, "sha256": digest}


def _has_reusable_result(current: dict) -> bool:
    if current.get("query_status") != "completed":
        return False
    result = current.get("result")
    if type(result) is not dict or result.get("execution_status") != "succeeded":
        return False
    return result.get("measurement_validity", "valid") == "valid"


def _status_from_dir(invocation_dir: Path) -> dict:
    request = _read_json(invocation_dir / "request.json")
    invocation_id = invocation_dir.name
    result_path = invocation_dir / "result.json"
    created = float(request["created_at_epoch"])
    if _regular_file_or_absent(result_path):
        result = _read_json(result_path)
        active = invocation_dir / "active-child.json"
        cleanup_status = result.get("cleanup_status", "unknown")
        if _regular_file_or_absent(active):
            try:
                child = _read_json(active)
                if _process_matches(
                    child.get("child_pid"),
                    child.get("child_start_token"),
                ) or _group_exists(child.get("process_group")):
                    cleanup_status = "unknown"
                else:
                    cleanup_status = "confirmed"
            except (OSError, TypeError, ValueError):
                cleanup_status = "unknown"
        return {
            "query_status": "completed",
            "invocation_id": invocation_id,
            "elapsed_seconds": max(
                0.0,
                float(result.get("finished_at_epoch", time.time())) - created,
            ),
            "cleanup_status": cleanup_status,
            "result_ref": _result_ref(invocation_id, result_path),
            "result": result,
        }

    worker_path = invocation_dir / "worker.json"
    if _regular_file_or_absent(worker_path):
        worker = _read_json(worker_path)
        worker_live = _process_matches(
            worker.get("worker_pid"), worker.get("worker_start_token")
        )
        guardian_live = _process_matches(
            worker.get("guardian_pid"), worker.get("guardian_start_token")
        )
        if worker_live or guardian_live:
            return {
                "query_status": "running",
                "invocation_id": invocation_id,
                "elapsed_seconds": max(0.0, time.time() - created),
                "cleanup_status": "pending",
            }
        return {
            "query_status": "worker_lost",
            "invocation_id": invocation_id,
            "elapsed_seconds": max(0.0, time.time() - created),
            "cleanup_status": "unknown",
            "stop_reason": "worker_lost",
        }

    if time.time() <= float(request["launch_deadline"]):
        return {
            "query_status": "starting",
            "invocation_id": invocation_id,
            "elapsed_seconds": max(0.0, time.time() - created),
            "cleanup_status": "not_required",
        }
    return {
        "query_status": "worker_lost",
        "invocation_id": invocation_id,
        "elapsed_seconds": max(0.0, time.time() - created),
        "cleanup_status": "not_required",
        "stop_reason": "worker_lost",
    }


def status(artifact_root, invocation_id: str) -> dict:
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    if (
        not isinstance(invocation_id, str)
        or not invocation_id.startswith("inv-")
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in invocation_id)
    ):
        raise ValueError("invocation_id is invalid")
    invocation_dir = root / "invocations" / invocation_id
    if not _is_real_directory(invocation_dir):
        raise ValueError("invocation not found")
    return _status_from_dir(invocation_dir)


def wait(artifact_root, invocation_id: str) -> dict:
    while True:
        current = status(artifact_root, invocation_id)
        if current["query_status"] == "completed":
            return {
                **current["result"],
                "invocation_id": invocation_id,
                "result_ref": current["result_ref"],
            }
        if current["query_status"] == "worker_lost":
            return current
        time.sleep(0.02)


def _runtime_lock(path: Path):
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _launch_file(path: Path, role: str) -> dict:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"worker launch {role} does not exist") from error
    if not _is_regular_file(resolved):
        raise ValueError(f"worker launch {role} is not a regular file")
    return {
        "role": role,
        "path": str(resolved),
        "sha256": STORE.sha256_file(resolved),
    }


def _build_worker_launch(worker_argv: Sequence[str]) -> dict:
    """Normalize and bind every executable source used by the worker gate."""
    command = list(worker_argv)
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("worker_argv must be a non-empty string list")
    executable = Path(command[0])
    if not executable.is_absolute():
        located = shutil.which(command[0])
        if located is None:
            raise ValueError("worker launch executable does not exist")
        executable = Path(located)
    executable_record = _launch_file(executable, "executable")
    command[0] = executable_record["path"]
    files = [executable_record]
    seen_paths = {executable_record["path"]}
    for index, argument in enumerate(command[1:], start=1):
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not _is_regular_file(resolved):
            continue
        record = _launch_file(resolved, f"argv[{index}]")
        command[index] = record["path"]
        if record["path"] not in seen_paths:
            files.append(record)
            seen_paths.add(record["path"])
    runtime_record = _launch_file(Path(__file__), "runtime")
    files.append(runtime_record)
    material = {"argv": command, "files": files}
    return {
        **material,
        "digest": hashlib.sha256(_strict_json_bytes(material)).hexdigest(),
    }


def _verify_worker_launch(launch: Mapping) -> list[str]:
    if type(launch) is not dict or set(launch) != {"argv", "files", "digest"}:
        raise ValueError("worker launch record is invalid")
    command = launch["argv"]
    files = launch["files"]
    digest = launch["digest"]
    if (
        isinstance(command, (str, bytes, bytearray))
        or not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
        or isinstance(files, (str, bytes, bytearray))
        or not isinstance(files, list)
        or not isinstance(digest, str)
    ):
        raise ValueError("worker launch record is invalid")
    material = {"argv": command, "files": files}
    if hashlib.sha256(_strict_json_bytes(material)).hexdigest() != digest:
        raise ValueError("worker launch identity changed")
    seen_roles = set()
    for record in files:
        if type(record) is not dict or set(record) != {"role", "path", "sha256"}:
            raise ValueError("worker launch record is invalid")
        role, path, expected = record["role"], record["path"], record["sha256"]
        if (
            not isinstance(role, str)
            or role in seen_roles
            or not isinstance(path, str)
            or not os.path.isabs(path)
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError("worker launch record is invalid")
        seen_roles.add(role)
        try:
            actual = _launch_file(Path(path), role)["sha256"]
        except ValueError as error:
            raise ValueError("worker launch identity changed") from error
        if actual != expected:
            raise ValueError("worker launch identity changed")
    if {"executable", "runtime"} - seen_roles:
        raise ValueError("worker launch record is invalid")
    return list(command)


def _invocation_digest(request_digest_value: str, launch_digest: str) -> str:
    return hashlib.sha256(
        _strict_json_bytes(
            {"request_digest": request_digest_value, "launch_digest": launch_digest}
        )
    ).hexdigest()


def _verify_invocation_launch(
    invocation_dir: Path,
    request: Mapping,
    launch: Mapping,
) -> list[str]:
    command = _verify_worker_launch(launch)
    request_value = request.get("request_digest")
    if not isinstance(request_value, str):
        raise ValueError("worker launch identity changed")
    expected_prefix = "inv-" + _invocation_digest(
        request_value, launch["digest"]
    )[:32]
    if not invocation_dir.name.startswith(expected_prefix):
        raise ValueError("worker launch identity changed")
    return command


def submit(artifact_root, frozen_request, worker_argv, wait_for_result: bool):
    """Create or reuse one invocation and start its detached guardian."""
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    if type(frozen_request) is not dict:
        raise ValueError("frozen_request must be an object")
    if isinstance(worker_argv, (str, bytes, bytearray)) or not isinstance(
        worker_argv, Sequence
    ):
        raise ValueError("worker_argv must be a non-empty string list")
    worker_launch = _build_worker_launch(worker_argv)

    request = dict(frozen_request)
    expected_digest = request_digest(request)
    if request.get("request_digest") != expected_digest:
        raise ValueError("frozen request digest does not match its semantics")
    _limits_from_request(request)
    _resources_from_request(request)
    launch_deadline = request.get("launch_deadline")
    if (
        isinstance(launch_deadline, bool)
        or not isinstance(launch_deadline, (int, float))
        or not math.isfinite(float(launch_deadline))
        or float(launch_deadline) <= time.time()
    ):
        raise ValueError("launch_deadline must be a future epoch")
    absolute = request.get("absolute_deadline")
    if absolute is not None and (
        isinstance(absolute, bool)
        or not isinstance(absolute, (int, float))
        or not math.isfinite(float(absolute))
        or float(absolute) <= time.time()
    ):
        raise ValueError("absolute_deadline must be a future epoch when present")

    invocations = root / "invocations"
    locks = root / ".locks"
    STORE.create_regular_directory(invocations, exist_ok=True)
    STORE.create_regular_directory(locks, exist_ok=True)
    combined_digest = _invocation_digest(expected_digest, worker_launch["digest"])
    invocation_id = "inv-" + combined_digest[:32]
    lock_fd = _runtime_lock(locks / f"invocation-{combined_digest}.lock")
    try:
        matching = [
            invocations / name
            for name in STORE.list_regular_directories(invocations, prefix=invocation_id)
            if _regular_file_or_absent(invocations / name / "request.json")
        ]
        if matching:
            records = []
            for candidate in matching:
                existing = _read_json(candidate / "request.json")
                if existing.get("request_digest") != expected_digest:
                    raise ValueError("invocation digest directory is inconsistent")
                existing_launch = _read_json(candidate / "worker-launch.json")
                if existing_launch.get("digest") != worker_launch["digest"]:
                    raise ValueError("invocation launch directory is inconsistent")
                records.append((candidate, existing, _status_from_dir(candidate)))

            retry_of = request.get("retry_of")
            if retry_of is not None:
                if not isinstance(retry_of, str):
                    raise ValueError("retry_of must be an invocation id")
                duplicate = next(
                    (
                        (candidate, current)
                        for candidate, existing, current in records
                        if existing.get("retry_of") == retry_of
                    ),
                    None,
                )
                if duplicate is not None:
                    selected, current = duplicate
                    if current["query_status"] == "completed" and wait_for_result:
                        return {
                            **current["result"],
                            "invocation_id": selected.name,
                            "result_ref": current["result_ref"],
                        }
                    return current if not wait_for_result else wait(root, selected.name)
                previous = next(
                    (
                        current
                        for candidate, _existing, current in records
                        if candidate.name == retry_of
                    ),
                    None,
                )
                if previous is None:
                    raise ValueError("retry_of does not name a matching invocation")
                if _has_reusable_result(previous):
                    return previous
                if previous.get("cleanup_status") not in {"not_required", "confirmed"}:
                    raise ValueError("retry requires confirmed cleanup of the previous invocation")
                invocation_id = f"{invocation_id}-retry-{secrets.token_hex(4)}"
            else:
                for selected, _existing, current in records:
                    if _has_reusable_result(current):
                        if wait_for_result:
                            return {
                                **current["result"],
                                "invocation_id": selected.name,
                                "result_ref": current["result_ref"],
                            }
                        return current
                    if current["query_status"] in {"starting", "running"}:
                        return (
                            wait(root, selected.name)
                            if wait_for_result
                            else {**current, "already_running": True}
                        )
                raise ValueError("failed invocation requires explicit retry_of")

        invocation_dir = invocations / invocation_id
        STORE.create_regular_directory(invocation_dir, mode=0o700)
        request["created_at_epoch"] = time.time()
        _create_json(invocation_dir / "request.json", request)
        _create_json(invocation_dir / "worker-launch.json", worker_launch)
        guardian_argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_guardian",
            "--invocation-dir",
            str(invocation_dir),
            "--artifact-root",
            str(root),
        ]
        subprocess.Popen(
            guardian_argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return (
        wait(root, invocation_id)
        if wait_for_result
        else status(root, invocation_id)
    )


def _write_frame(descriptor: int, value) -> None:
    payload = _strict_json_bytes(value) + b"\n"
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise BrokenPipeError("worker transport made no progress")
        offset += written


def _read_frame(descriptor: int, *, maximum: int = 4 * 1024 * 1024) -> dict:
    chunks = []
    size = 0
    while True:
        chunk = os.read(descriptor, 1)
        if not chunk:
            raise EOFError("worker transport closed")
        if chunk == b"\n":
            break
        size += 1
        if size > maximum:
            raise ValueError("worker transport frame is too large")
        chunks.append(chunk)
    value = json.loads(b"".join(chunks).decode("utf-8"))
    if type(value) is not dict:
        raise ValueError("worker transport frame must be an object")
    return value


def run_child(command_spec) -> dict:
    """Ask this invocation's guardian to run one bounded child command."""
    request_fd = os.environ.get(_WORKER_REQUEST_FD)
    response_fd = os.environ.get(_WORKER_RESPONSE_FD)
    if request_fd is None or response_fd is None:
        raise RuntimeError("run_child is only available inside an invocation worker")
    command = _validate_command(command_spec)
    _write_frame(int(request_fd), {"kind": "run_child", "command": command})
    response = _read_frame(int(response_fd))
    if set(response) != {"kind", "result"} or response["kind"] != "child_result":
        raise RuntimeError("guardian returned an invalid child response")
    if type(response["result"]) is not dict:
        raise RuntimeError("guardian child result is invalid")
    return response["result"]


def acquire_resources(gpu_uuids) -> dict:
    """Acquire this invocation's declared GPU set before semantic revalidation."""
    request_fd = os.environ.get(_WORKER_REQUEST_FD)
    response_fd = os.environ.get(_WORKER_RESPONSE_FD)
    if request_fd is None or response_fd is None:
        raise RuntimeError(
            "acquire_resources is only available inside an invocation worker"
        )
    required = _validate_gpu_uuids(gpu_uuids, "required_gpu_uuids")
    _write_frame(
        int(request_fd),
        {"kind": "acquire_resources", "gpu_uuids": required},
    )
    response = _read_frame(int(response_fd))
    if (
        set(response) != {"kind", "result"}
        or response["kind"] != "resource_result"
        or type(response["result"]) is not dict
    ):
        raise RuntimeError("guardian returned an invalid resource response")
    return response["result"]


def check_cancelled() -> bool:
    invocation_dir = os.environ.get(_INVOCATION_DIR)
    return invocation_dir is not None and _regular_file_or_absent(
        Path(invocation_dir) / "cancel"
    )


def current_cleanup_status() -> str:
    """Return the worker-visible cleanup fact without modifying runtime state."""
    invocation_dir = os.environ.get(_INVOCATION_DIR)
    if invocation_dir is None:
        return "unknown"
    return (
        "unknown"
        if _regular_file_or_absent(Path(invocation_dir) / "active-child.json")
        else "confirmed"
    )


def _guardian(invocation_dir: Path, artifact_root: Path) -> int:
    request = _read_json(invocation_dir / "request.json")
    try:
        _verify_invocation_launch(
            invocation_dir,
            request,
            _read_json(invocation_dir / "worker-launch.json"),
        )
    except ValueError as error:
        _atomic_json(
            invocation_dir / "result.json",
            {
                "status": "failed",
                "stop_reason": "worker_launch_identity_changed",
                "returncode": None,
                "elapsed_seconds": 0.0,
                "cleanup_status": "confirmed",
                "stdout": "",
                "stderr": str(error)[:1024],
                "finished_at_epoch": time.time(),
            },
        )
        _append_event(invocation_dir, "worker_launch_rejected")
        return 1
    limits = _limits_from_request(request)
    resources = _resources_from_request(request)
    cancel_path = invocation_dir / "cancel"
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    gate_read, gate_write = os.pipe()
    worker_env = os.environ.copy()
    worker_env.update(
        {
            _WORKER_REQUEST_FD: str(request_write),
            _WORKER_RESPONSE_FD: str(response_read),
            _INVOCATION_DIR: str(invocation_dir),
            _ARTIFACT_ROOT: str(artifact_root),
            _INVOCATION_ID: invocation_dir.name,
            _WORKER_GATE_FD: str(gate_read),
        }
    )
    guardian_token = _process_start_token(os.getpid())
    worker = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker_gate",
            "--invocation-dir",
            str(invocation_dir),
            "--artifact-root",
            str(artifact_root),
            "--worker-launch-file",
            str(invocation_dir / "worker-launch.json"),
        ],
        cwd=str(invocation_dir),
        env=worker_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(request_write, response_read, gate_read),
    )
    os.close(request_write)
    os.close(response_read)
    os.close(gate_read)
    worker_token = _process_start_token(worker.pid)
    _create_json(
        invocation_dir / "worker.json",
        {
            "guardian_pid": os.getpid(),
            "guardian_start_token": guardian_token,
            "worker_pid": worker.pid,
            "worker_start_token": worker_token,
            "started_at_epoch": time.time(),
        },
    )
    _append_event(invocation_dir, "worker_started", worker_pid=worker.pid)
    os.write(gate_write, b"1")
    os.close(gate_write)
    started = time.monotonic()
    operation_deadline = started + limits["operation_timeout_seconds"]
    absolute_deadline = request.get("absolute_deadline")
    locks = False
    try:
        while worker.poll() is None:
            if _regular_file_or_absent(cancel_path):
                try:
                    worker.terminate()
                except ProcessLookupError:
                    pass
                break
            if time.monotonic() >= operation_deadline or (
                absolute_deadline is not None and time.time() >= absolute_deadline
            ):
                try:
                    worker.terminate()
                except ProcessLookupError:
                    pass
                break
            readable, _, _ = select.select([request_read], [], [], 0.05)
            if not readable:
                continue
            try:
                frame = _read_frame(request_read)
            except EOFError:
                break
            if (
                set(frame) == {"kind", "gpu_uuids"}
                and frame["kind"] == "acquire_resources"
            ):
                required_gpu_uuids = _validate_gpu_uuids(
                    frame["gpu_uuids"],
                    "required_gpu_uuids",
                )
                _command_resources(
                    resources,
                    {"required_gpu_uuids": required_gpu_uuids},
                )
                if locks is False and required_gpu_uuids:
                    resource_deadline = min(
                        operation_deadline,
                        time.monotonic()
                        + limits["resource_wait_timeout_seconds"],
                    )
                    locks = _acquire_resource_locks(
                        resources,
                        resource_deadline,
                        cancel_path,
                    )
                    if locks is None:
                        locks = "timeout"
                    if isinstance(locks, list) and not _recover_occupancies(
                        _lock_root(), resources
                    ):
                        _release_locks(locks)
                        locks = "unknown"
                if isinstance(locks, str) and locks in {
                    "timeout",
                    "cancelled",
                    "unknown",
                }:
                    resource_result = {
                        "status": (
                            "cancelled" if locks == "cancelled" else "failed"
                        ),
                        "stop_reason": (
                            "cancelled"
                            if locks == "cancelled"
                            else "resource_cleanup_unknown"
                            if locks == "unknown"
                            else "resource_wait_timeout"
                        ),
                        "cleanup_status": (
                            "unknown" if locks == "unknown" else "confirmed"
                        ),
                    }
                else:
                    resource_result = {
                        "status": "acquired",
                        "stop_reason": "resources_acquired",
                        "cleanup_status": "confirmed",
                    }
                try:
                    _write_frame(
                        response_write,
                        {
                            "kind": "resource_result",
                            "result": resource_result,
                        },
                    )
                except BrokenPipeError:
                    break
                continue
            if set(frame) != {"kind", "command"} or frame["kind"] != "run_child":
                result = {
                    "status": "failed",
                    "stop_reason": "invalid_worker_request",
                    "returncode": None,
                    "elapsed_seconds": time.monotonic() - started,
                    "cleanup_status": "confirmed",
                    "stdout": "",
                    "stderr": "",
                }
            else:
                command = _validate_command(frame["command"])
                command_resources = _command_resources(resources, command)
                if locks is False and command["required_gpu_uuids"]:
                    resource_deadline = min(
                        operation_deadline,
                        time.monotonic()
                        + limits["resource_wait_timeout_seconds"],
                    )
                    locks = _acquire_resource_locks(
                        resources,
                        resource_deadline,
                        cancel_path,
                    )
                    if locks is None:
                        locks = "timeout"
                    if isinstance(locks, list) and not _recover_occupancies(
                        _lock_root(), resources
                    ):
                        _release_locks(locks)
                        locks = "unknown"
                if locks == "timeout" or locks == "cancelled" or locks == "unknown":
                    result = {
                        "status": (
                            "cancelled" if locks == "cancelled" else "failed"
                        ),
                        "stop_reason": (
                            "cancelled"
                            if locks == "cancelled"
                            else (
                                "resource_cleanup_unknown"
                                if locks == "unknown"
                            else "resource_wait_timeout"
                            )
                        ),
                        "returncode": None,
                        "elapsed_seconds": time.monotonic() - started,
                        "cleanup_status": (
                            "unknown" if locks == "unknown" else "confirmed"
                        ),
                        "stdout": "",
                        "stderr": "",
                    }
                else:
                    try:
                        result = _execute_validated(
                            command,
                            resources=resources,
                            operation_started=started,
                            operation_deadline=operation_deadline,
                            limits=limits,
                            absolute_deadline=absolute_deadline,
                            cancel_path=cancel_path,
                            tracking_dir=invocation_dir,
                            owner_fd=request_read,
                        )
                    except (OSError, ValueError) as error:
                        result = {
                            "status": "failed",
                            "stop_reason": "invalid_child_command",
                            "returncode": None,
                            "elapsed_seconds": time.monotonic() - started,
                            "cleanup_status": "confirmed",
                            "stdout": "",
                            "stderr": str(error)[:1024],
                        }
                if result["status"] in {"timed_out", "cancelled"}:
                    _append_event(
                        invocation_dir,
                        "child_stopped",
                        stop_reason=result["stop_reason"],
                    )
            try:
                _write_frame(
                    response_write,
                    {"kind": "child_result", "result": result},
                )
            except BrokenPipeError:
                break

        if worker.poll() is None:
            try:
                worker.wait(timeout=min(0.25, limits["cleanup_timeout_seconds"]))
            except subprocess.TimeoutExpired:
                try:
                    worker.kill()
                except ProcessLookupError:
                    pass
                try:
                    worker.wait(timeout=limits["cleanup_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    pass
        _append_event(
            invocation_dir,
            "guardian_finished",
            worker_returncode=worker.returncode,
            result_published=_regular_file_or_absent(invocation_dir / "result.json"),
        )
    finally:
        try:
            os.close(gate_write)
        except OSError:
            pass
        os.close(request_read)
        os.close(response_write)
        if isinstance(locks, list):
            _release_locks(locks)
    return 0


def _probe_guardian(probe_dir: Path, owner_fd: int) -> int:
    probe = _read_json(probe_dir / "probe.json")
    command = _validate_command(probe["command"])
    limits = _validate_limits(probe["limits"])
    resources = _command_resources(_validate_resources(probe["resources"]), command)
    started = time.monotonic()
    operation_deadline = started + limits["operation_timeout_seconds"]
    locks = _acquire_resource_locks(
        resources,
        min(operation_deadline, started + limits["resource_wait_timeout_seconds"]),
    )
    if locks is None:
        result = {
            "status": "timed_out", "stop_reason": "resource_wait_timeout",
            "returncode": None, "elapsed_seconds": time.monotonic() - started,
            "cleanup_status": "confirmed", "stdout": "", "stderr": "",
        }
    elif not _recover_occupancies(_lock_root(), resources):
        _release_locks(locks)
        result = {
            "status": "failed", "stop_reason": "resource_cleanup_unknown",
            "returncode": None, "elapsed_seconds": time.monotonic() - started,
            "cleanup_status": "unknown", "stdout": "", "stderr": "",
        }
    else:
        try:
            result = _execute_validated(
                command, resources=resources, operation_started=started,
                operation_deadline=operation_deadline, limits=limits,
                absolute_deadline=None, tracking_dir=probe_dir, owner_fd=owner_fd,
            )
        finally:
            _release_locks(locks)
    _atomic_json(probe_dir / "result.json", result)
    return 0


def _worker_gate(worker_launch_file: Path) -> int:
    """Keep a semantic worker inert until its durable identity exists."""
    descriptor = os.environ.get(_WORKER_GATE_FD)
    if descriptor is None or os.read(int(descriptor), 1) != b"1":
        return 1
    try:
        invocation_dir = os.environ.get(_INVOCATION_DIR)
        if invocation_dir is None:
            return 1
        invocation = Path(invocation_dir)
        command = _verify_invocation_launch(
            invocation,
            _read_json(invocation / "request.json"),
            _read_json(worker_launch_file),
        )
    except ValueError:
        return 1
    os.execvpe(command[0], command, os.environ.copy())
    return 1


def _child_gate(worker_argv_file: Path) -> int:
    descriptor = os.environ.get(_WORKER_GATE_FD)
    if descriptor is None or os.read(int(descriptor), 1) != b"1":
        return 1
    command = _read_json(worker_argv_file).get("argv")
    if isinstance(command, (str, bytes, bytearray)) or not isinstance(command, list):
        return 1
    if not command or any(not isinstance(item, str) or not item for item in command):
        return 1
    os.execvpe(command[0], command, os.environ.copy())
    return 1


def cancel(artifact_root, invocation_id: str) -> dict:
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    current = status(root, invocation_id)
    if (
        current["query_status"] == "completed"
        and current["cleanup_status"] == "confirmed"
    ):
        return current
    invocation_dir = root / "invocations" / invocation_id
    cancel_path = invocation_dir / "cancel"
    try:
        STORE.create_regular_bytes(cancel_path, b"")
    except FileExistsError:
        pass
    else:
        _append_event(invocation_dir, "cancel_requested")
    if current["query_status"] in {"completed", "worker_lost"}:
        request = _read_json(invocation_dir / "request.json")
        limits = _limits_from_request(request)
        resources = _resources_from_request(request)
        cleaned = _recover_active_child(
            invocation_dir,
            limits["cleanup_timeout_seconds"],
        )
        if cleaned:
            _clear_occupancies(_lock_root(), resources)
            _append_event(invocation_dir, "cleanup_confirmed")
    return status(root, invocation_id)


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "mode",
        choices=("_guardian", "_worker_gate", "_child_gate", "_probe_guardian"),
    )
    parser.add_argument("--invocation-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--worker-argv-file")
    parser.add_argument("--worker-launch-file")
    parser.add_argument("--owner-fd", type=int)
    args = parser.parse_args(argv)
    if args.mode == "_worker_gate":
        if args.worker_launch_file is None:
            return 2
        return _worker_gate(Path(args.worker_launch_file))
    if args.mode == "_child_gate":
        if args.worker_argv_file is None:
            return 2
        return _child_gate(Path(args.worker_argv_file))
    if args.mode == "_probe_guardian":
        if args.owner_fd is None:
            return 2
        return _probe_guardian(Path(args.invocation_dir), args.owner_fd)
    return _guardian(
        Path(args.invocation_dir),
        Path(args.artifact_root),
    )


if __name__ == "__main__":
    raise SystemExit(_main())
