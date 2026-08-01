#!/usr/bin/env python3
"""Copyable V1.4 command-driver skeleton.

The optimizer writes a closed request and invokes this file as:

    python3 workload_driver.py --request /absolute/request.json

Implement the three functions below for the real workload.  Do not infer GPU,
driver, framework, container, correctness, or measurement facts: return only
facts the workload actually observed.  This skeleton deliberately refuses to
write a result until those functions are implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
from pathlib import Path, PurePosixPath


DRIVER_PROTOCOL = "cuda-kernel-optimizer/driver-v1"
REQUEST_PROTOCOL = "cuda-kernel-optimizer/driver-request-v1"
RESULT_PROTOCOL = "cuda-kernel-optimizer/driver-result-v1"

_REQUEST_FIELDS = {
    "protocol_version", "request_digest", "target_id", "execution_id",
    "operation", "variant", "test_suite", "correctness", "objective",
    "role", "mode", "case", "sampling", "output_path", "driver_identity",
}
_RESULT_BASE_FIELDS = {
    "protocol_version", "request_digest", "target_id", "execution_id",
    "variant_digest", "role", "mode", "case_id", "artifacts", "cleanup",
    "driver_identity", "environment",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACTS = 128
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024


class DriverError(ValueError):
    pass


def run_correctness(request: dict) -> dict:
    """Return {"status": "passed"|"failed", "metrics": {name: finite_number}}.

    TODO: run the user's actual precision/correctness check.  Do not report
    "passed" merely because a command exited successfully.
    """
    raise NotImplementedError("TODO: implement run_correctness for this workload")


def run_measurements(request: dict) -> dict:
    """Return primary and constraint sample arrays from the real workload.

    TODO: run the user's actual measurement.  All samples must be finite and
    the returned units must be the units actually observed.
    """
    raise NotImplementedError("TODO: implement run_measurements for this workload")


def collect_environment(request: dict) -> dict:
    """Return the exact environment object observed for this invocation.

    TODO: obtain the real GPU, driver, CUDA runtime, framework, and container
    identity.  Never insert placeholders or values copied from another host.
    """
    raise NotImplementedError("TODO: implement collect_environment for this workload")


def cleanup(request: dict) -> dict:
    """Return {"status": "confirmed", "live_tasks": []} only after cleanup.

    TODO: terminate and verify every child task started by this driver.  Every
    task must remain in the Invocation process group.  Do not claim
    confirmation while any task may remain alive.
    """
    raise NotImplementedError("TODO: implement cleanup for this workload")


def collect_artifacts(request: dict) -> list[dict]:
    """Return raw invocation files as kind, relative_path, and sha256 records.

    Paths are relative to the result file's directory.  A driver that declares
    ``pytorch_chrome_trace_v1`` must return exactly one
    ``pytorch_chrome_trace`` artifact for profiler collection operations.
    """
    return []


def _closed(value, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise DriverError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise DriverError(f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}")
    return value


def _text(value, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise DriverError(f"{label} must be a non-empty bounded string")
    return value


def _sha256(value, label: str) -> str:
    value = _text(value, label, 64)
    if _SHA256.fullmatch(value) is None:
        raise DriverError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise DriverError(f"{label} must be a finite number")
    return value


def _json_value(value, label: str):
    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise DriverError(f"{label} must be finite JSON") from error


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read_request(path) -> dict:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    descriptor = None
    try:
        metadata = os.lstat(target)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise DriverError("request must be a regular non-symlink file")
        if metadata.st_size > _MAX_JSON_BYTES:
            raise DriverError("request exceeds the byte limit")
        descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_JSON_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_JSON_BYTES:
                raise DriverError("request exceeds the byte limit")
        raw = b"".join(chunks)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_object(pairs, "request"),
            parse_constant=lambda token: (_ for _ in ()).throw(DriverError(f"request contains non-finite number: {token}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DriverError("request is invalid JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(value) is not dict:
        raise DriverError("request root must be an object")
    return value


def _unique_object(pairs, label: str) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise DriverError(f"{label} contains duplicate key: {key}")
        value[key] = item
    return value


def _validate_request(value: dict) -> dict:
    request = _closed(value, _REQUEST_FIELDS, "request")
    if request["protocol_version"] != REQUEST_PROTOCOL:
        raise DriverError("request protocol_version is unsupported")
    for field in ("request_digest", "driver_identity"):
        _sha256(request[field], field)
    for field in ("target_id", "execution_id", "operation", "output_path"):
        _text(request[field], field)
    variant = _closed(request["variant"], {"kind", "digest", "locator"}, "request.variant")
    if variant["kind"] not in {"source_snapshot", "artifact", "deployment"}:
        raise DriverError("request.variant.kind is unsupported")
    _sha256(variant["digest"], "request.variant.digest")
    _text(variant["locator"], "request.variant.locator")
    suite = _closed(request["test_suite"], {"digest", "locator", "case_ids"}, "request.test_suite")
    _sha256(suite["digest"], "request.test_suite.digest")
    _text(suite["locator"], "request.test_suite.locator")
    if type(suite["case_ids"]) is not list or not suite["case_ids"]:
        raise DriverError("request.test_suite.case_ids must be a non-empty string list")
    if len(suite["case_ids"]) != len(set(suite["case_ids"])):
        raise DriverError("request.test_suite.case_ids must not contain duplicates")
    for index, case_id in enumerate(suite["case_ids"]):
        _text(case_id, f"request.test_suite.case_ids[{index}]")
    correctness = _closed(request["correctness"], {"reference", "method", "acceptance"}, "request.correctness")
    reference = _closed(correctness["reference"], {"digest", "locator"}, "request.correctness.reference")
    _sha256(reference["digest"], "request.correctness.reference.digest")
    _text(reference["locator"], "request.correctness.reference.locator")
    _text(correctness["method"], "request.correctness.method")
    objective = _closed(request["objective"], {"primary_metric", "constraints"}, "request.objective")
    _json_value(correctness["acceptance"], "request.correctness.acceptance")
    _json_value(objective["primary_metric"], "request.objective.primary_metric")
    _json_value(objective["constraints"], "request.objective.constraints")
    if request["role"] not in {"original", "reference", "candidate"}:
        raise DriverError("request.role is unsupported")
    if request["mode"] not in {"correctness", "measure", "combined"}:
        raise DriverError("request.mode is unsupported")
    if type(request["case"]) is not dict or type(request["sampling"]) is not dict:
        raise DriverError("request.case and request.sampling must be objects")
    _json_value(request["case"], "request.case")
    _json_value(request["sampling"], "request.sampling")
    core = {key: request[key] for key in _REQUEST_FIELDS if key != "request_digest"}
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != request["request_digest"]:
        raise DriverError("request_digest does not match request identity")
    return request


def _metrics(value: object, label: str) -> dict:
    if type(value) is not dict:
        raise DriverError(f"{label} must be an object")
    return {_text(name, f"{label} name", 128): _finite(metric, f"{label}.{name}") for name, metric in value.items()}


def _validate_correctness(value: object) -> dict:
    value = _closed(value, {"status", "metrics"}, "correctness")
    if value["status"] not in {"passed", "failed"}:
        raise DriverError("correctness.status is unsupported")
    return {"status": value["status"], "metrics": _metrics(value["metrics"], "correctness.metrics")}


def _samples(value: object, label: str) -> list[float]:
    if type(value) is not list or not value:
        raise DriverError(f"{label} must be a non-empty number list")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _validate_measurements(value: object) -> dict:
    value = _closed(value, {"primary", "constraints"}, "measurements")
    primary = _closed(value["primary"], {"name", "unit", "samples"}, "measurements.primary")
    if type(value["constraints"]) is not list:
        raise DriverError("measurements.constraints must be a list")
    constraints = []
    names = set()
    for index, item in enumerate(value["constraints"]):
        item = _closed(item, {"name", "unit", "samples"}, f"measurements.constraints[{index}]")
        name = _text(item["name"], f"measurements.constraints[{index}].name", 128)
        if name in names:
            raise DriverError("measurements.constraints names must be unique")
        names.add(name)
        constraints.append({"name": name, "unit": _text(item["unit"], f"measurements.constraints[{index}].unit", 64), "samples": _samples(item["samples"], f"measurements.constraints[{index}].samples")})
    return {
        "primary": {"name": _text(primary["name"], "measurements.primary.name", 128), "unit": _text(primary["unit"], "measurements.primary.unit", 64), "samples": _samples(primary["samples"], "measurements.primary.samples")},
        "constraints": constraints,
    }


def _validate_environment(value: object) -> dict:
    fields = {"gpu_uuids", "gpu_models", "gpu_architectures", "driver_version", "cuda_runtime_version", "frameworks", "container"}
    value = _closed(value, fields, "environment")
    for field in ("gpu_uuids", "gpu_models", "gpu_architectures"):
        if type(value[field]) is not list or any(not isinstance(item, str) or not item for item in value[field]):
            raise DriverError(f"environment.{field} must be a string list")
    if len(value["gpu_uuids"]) != len(set(value["gpu_uuids"])):
        raise DriverError("environment.gpu_uuids must be unique")
    if len(value["gpu_models"]) != len(value["gpu_uuids"]) or len(value["gpu_architectures"]) != len(value["gpu_uuids"]):
        raise DriverError("environment GPU arrays must align")
    if type(value["frameworks"]) is not dict:
        raise DriverError("environment.frameworks must be an object")
    for name, version in value["frameworks"].items():
        _text(name, "environment.frameworks name", 128)
        _text(version, f"environment.frameworks.{name}", 256)
    container = _closed(value["container"], {"kind", "identity"}, "environment.container")
    return {
        "gpu_uuids": list(value["gpu_uuids"]), "gpu_models": list(value["gpu_models"]),
        "gpu_architectures": list(value["gpu_architectures"]),
        "driver_version": _text(value["driver_version"], "environment.driver_version", 256),
        "cuda_runtime_version": _text(value["cuda_runtime_version"], "environment.cuda_runtime_version", 256),
        "frameworks": dict(sorted(value["frameworks"].items())),
        "container": {"kind": _text(container["kind"], "environment.container.kind"), "identity": _text(container["identity"], "environment.container.identity")},
    }


def _artifact_digest(root: Path, parts: tuple[str, ...]) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    directories = []
    try:
        descriptor = os.open(root, directory_flags)
        directories.append(descriptor)
        for part in parts[:-1]:
            descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            directories.append(descriptor)
        file_descriptor = os.open(parts[-1], flags, dir_fd=descriptor)
        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise DriverError("artifact must be a regular file")
            if metadata.st_size > _MAX_ARTIFACT_BYTES:
                raise DriverError("artifact exceeds the byte limit")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            return digest.hexdigest()
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise DriverError("artifact is unavailable or unsafe") from error
    finally:
        for descriptor in reversed(directories):
            os.close(descriptor)


def _validate_artifacts(value: object, output_path: str) -> list[dict]:
    if type(value) is not list or len(value) > _MAX_ARTIFACTS:
        raise DriverError("result.artifacts must be a bounded list")
    root = Path(output_path).parent
    normalized = []
    seen = set()
    for index, item in enumerate(value):
        item = _closed(
            item,
            {"kind", "relative_path", "sha256"},
            f"result.artifacts[{index}]",
        )
        kind = _text(item["kind"], f"result.artifacts[{index}].kind", 128)
        relative = _text(
            item["relative_path"],
            f"result.artifacts[{index}].relative_path",
        )
        if "\\" in relative or "\x00" in relative:
            raise DriverError("artifact path must be a canonical POSIX relative path")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or str(path) != relative
            or relative in seen
        ):
            raise DriverError("artifact path must be unique and relative")
        seen.add(relative)
        actual = _artifact_digest(root, path.parts)
        expected = _sha256(item["sha256"], f"result.artifacts[{index}].sha256")
        if actual != expected:
            raise DriverError("artifact digest does not match")
        normalized.append(
            {"kind": kind, "relative_path": relative, "sha256": expected}
        )
    return normalized


def _validate_result(value: object, request: dict) -> dict:
    required = set(_RESULT_BASE_FIELDS)
    if request["mode"] in {"correctness", "combined"}:
        required.add("correctness")
    if request["mode"] in {"measure", "combined"}:
        required.add("measurements")
    value = _closed(value, required, "result")
    if value["protocol_version"] != RESULT_PROTOCOL:
        raise DriverError("result protocol_version is unsupported")
    expected = {
        "request_digest": request["request_digest"], "target_id": request["target_id"],
        "execution_id": request["execution_id"], "variant_digest": request["variant"]["digest"],
        "role": request["role"], "mode": request["mode"], "case_id": request["case"].get("id"),
        "driver_identity": request["driver_identity"],
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise DriverError(f"result {field} does not match request")
    artifacts = _validate_artifacts(value["artifacts"], request["output_path"])
    _validate_cleanup(value["cleanup"])
    result = {field: _json_value(value[field], f"result.{field}") for field in _RESULT_BASE_FIELDS}
    result["artifacts"] = artifacts
    result["environment"] = _validate_environment(value["environment"])
    if "correctness" in required:
        result["correctness"] = _validate_correctness(value["correctness"])
    if "measurements" in required:
        result["measurements"] = _validate_measurements(value["measurements"])
    return result


def _validate_cleanup(value: object) -> dict:
    cleanup = _closed(value, {"status", "live_tasks"}, "cleanup")
    if cleanup["status"] != "confirmed" or type(cleanup["live_tasks"]) is not list or cleanup["live_tasks"]:
        raise DriverError("cleanup must be confirmed with no live tasks")
    return {"status": "confirmed", "live_tasks": []}


def _atomic_json(path, value: dict) -> None:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    parent = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = f".{target.name}.{secrets.token_hex(8)}.tmp"
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        payload = _canonical_bytes(value)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("result write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            target.name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
        os.fsync(parent)
        os.unlink(temporary, dir_fd=parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one closed workload-driver request.")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        request = _validate_request(_read_request(args.request))
        result = None
        primary_error = None
        try:
            result = {
                "protocol_version": RESULT_PROTOCOL,
                "request_digest": request["request_digest"], "target_id": request["target_id"],
                "execution_id": request["execution_id"], "variant_digest": request["variant"]["digest"],
                "role": request["role"], "mode": request["mode"], "case_id": request["case"].get("id"),
                "artifacts": collect_artifacts(request),
                "driver_identity": request["driver_identity"],
            }
            if request["mode"] in {"correctness", "combined"}:
                result["correctness"] = run_correctness(request)
            if request["mode"] in {"measure", "combined"}:
                result["measurements"] = run_measurements(request)
            result["environment"] = collect_environment(request)
        except BaseException as error:
            primary_error = error
        try:
            cleanup_result = _validate_cleanup(cleanup(request))
        except BaseException as error:
            if primary_error is not None:
                raise DriverError(f"{primary_error}; cleanup failed: {error}") from error
            raise
        if primary_error is not None:
            raise primary_error
        assert result is not None
        result["cleanup"] = cleanup_result
        _atomic_json(request["output_path"], _validate_result(result, request))
    except (DriverError, NotImplementedError, OSError, ValueError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)[:1024]}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
