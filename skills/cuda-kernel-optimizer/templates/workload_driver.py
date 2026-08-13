#!/usr/bin/env python3
"""Copyable V2 workload-driver skeleton.

Implement the workload-specific hooks below. One invocation returns correctness
and measurements for every requested subject. A two-subject request must keep
both runs inside this process and preserve the declared shared state.
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


DRIVER_PROTOCOL = "cuda-kernel-optimizer/driver-v2"
REQUEST_PROTOCOL = "cuda-kernel-optimizer/driver-request-v2"
RESULT_PROTOCOL = "cuda-kernel-optimizer/driver-result-v2"

_REQUEST_FIELDS = {
    "protocol_version", "request_digest", "target_id", "execution_id",
    "operation", "subjects", "test_suite", "correctness", "objective",
    "acquisition", "case", "sampling", "output_path", "driver_identity",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACTS = 128
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024


class DriverError(ValueError):
    pass


def prepare_acquisition(request: dict):
    """Create state shared by subjects in this one driver invocation."""
    return None


def run_subject_evidence(request: dict, subject: dict, shared_state) -> dict:
    """Return one subject's correctness and measurements from the real workload.

    Return:
      {
        "correctness": {"status": "passed"|"failed", "metrics": {...}},
        "measurements": {
          "primary": {"name": ..., "unit": ..., "samples": [...]},
          "constraints": [...]
        }
      }

    Resolve request["case"]["id"] from the frozen test suite. Correctness and
    measurement should come from the same workload execution whenever the
    workload permits. For a same_process request, reuse shared_state and do not
    launch an independent top-level service for each subject.
    """
    raise NotImplementedError("TODO: implement run_subject_evidence for this workload")


def collect_environment(request: dict) -> dict:
    """Return the runtime identity actually used by this invocation."""
    raise NotImplementedError("TODO: implement collect_environment for this workload")


def collect_artifacts(request: dict, shared_state) -> list[dict]:
    """Return raw files as kind, relative_path and sha256 records."""
    return []


def cleanup(request: dict, shared_state) -> dict:
    """Confirm that all child tasks created by this invocation are gone."""
    raise NotImplementedError("TODO: implement cleanup for this workload")


def _closed(value, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise DriverError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise DriverError(
            f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}"
        )
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
    number = float(value)
    if not math.isfinite(number):
        raise DriverError(f"{label} must be a finite number")
    return number


def _json_value(value, label: str):
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DriverError(f"{label} must be finite JSON") from error


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs, label: str) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise DriverError(f"{label} contains duplicate key: {key}")
        value[key] = item
    return value


def _read_request(path) -> dict:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    descriptor = None
    try:
        metadata = os.lstat(target)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise DriverError("request must be a regular non-symlink file")
        if metadata.st_size > _MAX_JSON_BYTES:
            raise DriverError("request exceeds the byte limit")
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        chunks = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_JSON_BYTES + 1 - total),
            )
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
            parse_constant=lambda token: (_ for _ in ()).throw(
                DriverError(f"request contains non-finite number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DriverError("request is invalid JSON") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(value) is not dict:
        raise DriverError("request root must be an object")
    return value


def _string_list(value, label: str) -> list[str]:
    if type(value) is not list or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise DriverError(f"{label} must be a string list")
    if len(value) != len(set(value)):
        raise DriverError(f"{label} must not contain duplicates")
    return list(value)


def _validate_variant(value, label: str) -> dict:
    value = _closed(value, {"kind", "digest", "locator"}, label)
    if value["kind"] not in {"source_snapshot", "artifact", "deployment"}:
        raise DriverError(f"{label}.kind is unsupported")
    return {
        "kind": value["kind"],
        "digest": _sha256(value["digest"], f"{label}.digest"),
        "locator": _text(value["locator"], f"{label}.locator"),
    }


def _validate_request(value: dict) -> dict:
    request = _closed(value, _REQUEST_FIELDS, "request")
    if request["protocol_version"] != REQUEST_PROTOCOL:
        raise DriverError("request protocol_version is unsupported")
    for field in ("request_digest", "driver_identity"):
        _sha256(request[field], field)
    for field in ("target_id", "execution_id", "operation", "output_path"):
        _text(request[field], field)

    subjects = request["subjects"]
    if type(subjects) is not list or len(subjects) not in {1, 2}:
        raise DriverError("request.subjects must contain one or two subjects")
    roles = set()
    normalized_subjects = []
    for index, subject in enumerate(subjects):
        subject = _closed(subject, {"role", "variant"}, f"request.subjects[{index}]")
        role = subject["role"]
        if role not in {"original", "reference", "candidate"} or role in roles:
            raise DriverError("request subject roles must be unique and supported")
        roles.add(role)
        normalized_subjects.append(
            {
                "role": role,
                "variant": _validate_variant(
                    subject["variant"], f"request.subjects[{index}].variant"
                ),
            }
        )

    acquisition = _closed(
        request["acquisition"],
        {"lifecycle", "shared_state", "rebuilt_state"},
        "request.acquisition",
    )
    shared = _string_list(acquisition["shared_state"], "request.acquisition.shared_state")
    rebuilt = _string_list(
        acquisition["rebuilt_state"], "request.acquisition.rebuilt_state"
    )
    if set(shared) & set(rebuilt):
        raise DriverError("acquisition state cannot be both shared and rebuilt")
    if len(subjects) == 1 and (
        acquisition["lifecycle"] != "isolated_process" or shared
    ):
        raise DriverError("single-subject acquisition must be isolated")
    if len(subjects) == 2 and (
        acquisition["lifecycle"] != "same_process" or not shared
    ):
        raise DriverError("paired acquisition must be same-process with shared state")

    suite = _closed(
        request["test_suite"], {"digest", "locator", "case_ids"}, "request.test_suite"
    )
    _sha256(suite["digest"], "request.test_suite.digest")
    _text(suite["locator"], "request.test_suite.locator")
    case_ids = _string_list(suite["case_ids"], "request.test_suite.case_ids")
    if not case_ids:
        raise DriverError("request.test_suite.case_ids must not be empty")

    correctness = _closed(
        request["correctness"],
        {"reference", "method", "acceptance"},
        "request.correctness",
    )
    reference = _closed(
        correctness["reference"],
        {"digest", "locator"},
        "request.correctness.reference",
    )
    _sha256(reference["digest"], "request.correctness.reference.digest")
    _text(reference["locator"], "request.correctness.reference.locator")
    _text(correctness["method"], "request.correctness.method")
    objective = _closed(
        request["objective"], {"primary_metric", "constraints"}, "request.objective"
    )
    for label, item in (
        ("request.correctness.acceptance", correctness["acceptance"]),
        ("request.objective.primary_metric", objective["primary_metric"]),
        ("request.objective.constraints", objective["constraints"]),
        ("request.case", request["case"]),
        ("request.sampling", request["sampling"]),
    ):
        _json_value(item, label)
    if type(request["case"]) is not dict or type(request["sampling"]) is not dict:
        raise DriverError("request.case and request.sampling must be objects")

    normalized = dict(request)
    normalized["subjects"] = normalized_subjects
    normalized["acquisition"] = {
        "lifecycle": acquisition["lifecycle"],
        "shared_state": shared,
        "rebuilt_state": rebuilt,
    }
    core = {key: normalized[key] for key in _REQUEST_FIELDS if key != "request_digest"}
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != request["request_digest"]:
        raise DriverError("request_digest does not match request identity")
    return normalized


def _metrics(value, label: str) -> dict:
    if type(value) is not dict:
        raise DriverError(f"{label} must be an object")
    return {
        _text(name, f"{label} name", 128): _finite(metric, f"{label}.{name}")
        for name, metric in value.items()
    }


def _validate_correctness(value) -> dict:
    value = _closed(value, {"status", "metrics"}, "correctness")
    if value["status"] not in {"passed", "failed"}:
        raise DriverError("correctness.status is unsupported")
    return {
        "status": value["status"],
        "metrics": _metrics(value["metrics"], "correctness.metrics"),
    }


def _samples(value, label: str) -> list[float]:
    if type(value) is not list or not value:
        raise DriverError(f"{label} must be a non-empty number list")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _validate_measurements(value) -> dict:
    value = _closed(value, {"primary", "constraints"}, "measurements")
    primary = _closed(
        value["primary"], {"name", "unit", "samples"}, "measurements.primary"
    )
    if type(value["constraints"]) is not list:
        raise DriverError("measurements.constraints must be a list")
    constraints = []
    names = set()
    for index, item in enumerate(value["constraints"]):
        item = _closed(
            item,
            {"name", "unit", "samples"},
            f"measurements.constraints[{index}]",
        )
        name = _text(item["name"], f"measurements.constraints[{index}].name", 128)
        if name in names:
            raise DriverError("measurements constraint names must be unique")
        names.add(name)
        constraints.append(
            {
                "name": name,
                "unit": _text(item["unit"], "constraint unit", 64),
                "samples": _samples(item["samples"], "constraint samples"),
            }
        )
    return {
        "primary": {
            "name": _text(primary["name"], "measurements.primary.name", 128),
            "unit": _text(primary["unit"], "measurements.primary.unit", 64),
            "samples": _samples(primary["samples"], "measurements.primary.samples"),
        },
        "constraints": constraints,
    }


def _validate_environment(value) -> dict:
    fields = {
        "gpu_uuids", "gpu_models", "gpu_architectures", "driver_version",
        "cuda_runtime_version", "frameworks", "runtime_provenance",
    }
    value = _closed(value, fields, "environment")
    for field in ("gpu_uuids", "gpu_models", "gpu_architectures"):
        _string_list(value[field], f"environment.{field}")
    if (
        len(value["gpu_models"]) != len(value["gpu_uuids"])
        or len(value["gpu_architectures"]) != len(value["gpu_uuids"])
    ):
        raise DriverError("environment GPU arrays must align")
    if type(value["frameworks"]) is not dict:
        raise DriverError("environment.frameworks must be an object")
    for name, version in value["frameworks"].items():
        _text(name, "environment.frameworks name", 128)
        _text(version, f"environment.frameworks.{name}", 256)
    provenance = _closed(
        value["runtime_provenance"],
        {"kind", "identity", "lineage_complete", "lineage", "components"},
        "environment.runtime_provenance",
    )
    if provenance["kind"] not in {"host", "container"}:
        raise DriverError("runtime provenance kind is unsupported")
    if type(provenance["lineage_complete"]) is not bool:
        raise DriverError("runtime provenance lineage_complete must be boolean")
    if type(provenance["lineage"]) is not list or type(provenance["components"]) is not list:
        raise DriverError("runtime provenance lineage and components must be lists")
    if provenance["kind"] == "host" and (
        not provenance["lineage_complete"] or provenance["lineage"]
    ):
        raise DriverError("host runtime provenance must be complete without image lineage")
    return _json_value(value, "environment")


def _validate_cleanup(value) -> dict:
    value = _closed(value, {"status", "live_tasks"}, "cleanup")
    if value != {"status": "confirmed", "live_tasks": []}:
        raise DriverError("cleanup must be confirmed with no live tasks")
    return value


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


def _validate_artifacts(value, output_path: str) -> list[dict]:
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
        relative = _text(item["relative_path"], "artifact relative_path")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or str(path) != relative
            or relative in seen
            or relative == "result.json"
        ):
            raise DriverError("artifact relative_path is invalid or reserved")
        seen.add(relative)
        expected = _sha256(item["sha256"], "artifact sha256")
        if _artifact_digest(root, path.parts) != expected:
            raise DriverError("artifact digest does not match")
        normalized.append(
            {
                "kind": _text(item["kind"], "artifact kind", 128),
                "relative_path": relative,
                "sha256": expected,
            }
        )
    return normalized


def _atomic_json(path, value: dict) -> None:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    parent = os.open(
        target.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = f".{target.name}.{secrets.token_hex(8)}.tmp"
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
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
    parser = argparse.ArgumentParser(description="Run one V2 workload evidence request.")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        request = _validate_request(_read_request(args.request))
        shared_state = prepare_acquisition(request)
        evidence = {"correctness": [], "measurements": []}
        primary_error = None
        try:
            for subject in request["subjects"]:
                observed = _closed(
                    run_subject_evidence(request, subject, shared_state),
                    {"correctness", "measurements"},
                    f"evidence for {subject['role']}",
                )
                evidence["correctness"].append(
                    {
                        "role": subject["role"],
                        "result": _validate_correctness(observed["correctness"]),
                    }
                )
                evidence["measurements"].append(
                    {
                        "role": subject["role"],
                        "result": _validate_measurements(observed["measurements"]),
                    }
                )
            result = {
                "protocol_version": RESULT_PROTOCOL,
                "request_digest": request["request_digest"],
                "target_id": request["target_id"],
                "execution_id": request["execution_id"],
                "subject_digests": [
                    {
                        "role": subject["role"],
                        "digest": subject["variant"]["digest"],
                    }
                    for subject in request["subjects"]
                ],
                "case_id": request["case"].get("id"),
                "artifacts": _validate_artifacts(
                    collect_artifacts(request, shared_state),
                    request["output_path"],
                ),
                "driver_identity": request["driver_identity"],
                "environment": _validate_environment(collect_environment(request)),
                "acquisition": request["acquisition"],
                "evidence": evidence,
            }
        except BaseException as error:
            primary_error = error
            result = None
        try:
            cleanup_result = _validate_cleanup(cleanup(request, shared_state))
        except BaseException as error:
            if primary_error is not None:
                raise DriverError(f"{primary_error}; cleanup failed: {error}") from error
            raise
        if primary_error is not None:
            raise primary_error
        assert result is not None
        result["cleanup"] = cleanup_result
        _atomic_json(request["output_path"], result)
    except (DriverError, NotImplementedError, OSError, ValueError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)[:1024]}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
