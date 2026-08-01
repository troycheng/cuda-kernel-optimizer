#!/usr/bin/env python3
"""Analyze one frozen, version-bound Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote


INPUT_VERSION = "cuda-kernel-optimizer/nsys-input-v1"
RESULT_VERSION = "cuda-kernel-optimizer/profiler-result-v1"
PARSER_VERSION = "nsys-sqlite-3.25-v1"
TOOL_IDENTITY_VERSION = "cuda-kernel-optimizer/nsys-tool-v1"
_VERSION = re.compile(r"2026\.2(?:\.\d+)*\Z")
_ANALYZE_INPUT = {
    "format_version", "operation", "artifact_root", "target_ref", "report_ref",
    "resources", "operation_timeout_seconds", "command_timeout_seconds",
    "resource_wait_timeout_seconds", "cleanup_timeout_seconds", "launch_deadline",
}
_INPUT_OPTIONAL = {"absolute_deadline", "retry_of"}
_COLLECT_INPUT = {
    "format_version", "operation", "artifact_root", "target_ref", "baseline_ref",
    "role", "case_id", "resources", "operation_timeout_seconds",
    "command_timeout_seconds", "resource_wait_timeout_seconds",
    "cleanup_timeout_seconds", "launch_deadline",
}
_COLLECT_OPTIONAL = {"experiment_ref", "correctness_ref", "absolute_deadline", "retry_of"}
_MATERIAL = {"id", "sha256", "kind", "tool", "tool_version", "dialect", "object_ref"}
_KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
_KERNEL_COLUMNS = {"start", "end", "demangledName"}
_DIALECT = "nsys-sqlite-3.25-v1"
_MAX_SQLITE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_TABLES, _MAX_ROWS = 128, 10_000
_MAX_UNMODELED = 128
_DRIVER_OUTPUT_FREEZE_LIMITS = {
    "max_files": 16,
    "max_total_bytes": 16 * 1024 * 1024,
    "max_wall_seconds": 5.0,
}
_NSYS_ARTIFACT_FREEZE_LIMITS = {
    "max_files": 1,
    "max_total_bytes": 8 * 1024 * 1024 * 1024,
    "max_wall_seconds": 120.0,
}


def _load_sibling(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_nsys_store")
RUNTIME = _load_sibling("_invocation_runtime.py", "cuda_optimizer_nsys_runtime")
ADAPTER = _load_sibling("workload_adapter.py", "cuda_optimizer_nsys_adapter")


class NsysError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _closed(value, required, optional=(), label="value") -> dict:
    if type(value) is not dict:
        raise NsysError("invalid_nsys_input", f"{label} must be an object")
    missing, unknown = set(required) - set(value), set(value) - set(required) - set(optional)
    if missing or unknown:
        raise NsysError("invalid_nsys_input", f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}")
    return value


def _text(value, label: str, maximum=4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise NsysError("invalid_nsys_input", f"{label} must be a non-empty bounded string")
    return value


def _sha(value, label: str) -> str:
    value = _text(value, label, 64)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NsysError("invalid_nsys_input", f"{label} must be a SHA-256")
    return value


def _finite(value, label: str, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise NsysError("invalid_nsys_input", f"{label} must be finite")
    if positive and value <= 0:
        raise NsysError("invalid_nsys_input", f"{label} must be positive")
    return float(value)


def _strict_json(path) -> dict:
    try:
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise NsysError("invalid_nsys_input", "request contains duplicate key")
                value[key] = item
            return value
        value = json.loads(STORE.read_regular_bytes(path).decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(NsysError("invalid_nsys_input", "request contains non-finite number")))
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise NsysError("invalid_nsys_input", "request is invalid JSON") from error
    if type(value) is not dict:
        raise NsysError("invalid_nsys_input", "request must be an object")
    return value


def _target(root: Path, reference) -> dict:
    reference = _closed(reference, {"id", "sha256"}, label="target_ref")
    try:
        payload = STORE.read_regular_bytes(root / "target.json")
        target = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise NsysError("target_not_found", "target record is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != _sha(reference["sha256"], "target_ref.sha256"):
        raise NsysError("target_changed", "target record digest changed")
    if type(target) is not dict or target.get("record_type") != "target" or target.get("format_version") != "cuda-kernel-optimizer/target-v1" or target.get("id") != _text(reference["id"], "target_ref.id") or target.get("target_mode") != "diagnostic":
        raise NsysError("target_invalid", "target is not a frozen diagnostic target")
    return target


def _material(target: dict, reference) -> dict:
    reference = _closed(reference, {"id", "sha256"}, label="report_ref")
    report_id, digest = _text(reference["id"], "report_ref.id"), _sha(reference["sha256"], "report_ref.sha256")
    matches = [item for item in target.get("diagnostic_materials", []) if type(item) is dict and item.get("id") == report_id]
    if len(matches) != 1:
        raise NsysError("report_not_found", "report_ref does not select one frozen material")
    material = _closed(matches[0], _MATERIAL, label="diagnostic material")
    if material["sha256"] != digest or material["kind"] != "report" or material["tool"] != "nsys":
        raise NsysError("unsupported_report", "material is not the requested Nsys report")
    if material["dialect"] != _DIALECT:
        raise NsysError("unsupported_report", "material is not the accepted Nsys SQLite dialect")
    if not isinstance(material["tool_version"], str) or _VERSION.fullmatch(material["tool_version"]) is None:
        raise NsysError("unsupported_tool_version", "Nsys version is not supported")
    object_ref = material["object_ref"]
    expected_locator = str(Path("objects") / "sha256" / digest)
    if type(object_ref) is not dict or object_ref.get("digest") != digest or object_ref.get("locator") != expected_locator or set(object_ref) - {"digest", "locator", "source_kind", "file_count", "total_bytes"}:
        raise NsysError("report_changed", "report object reference is invalid")
    identity = {key: material[key] for key in ("kind", "tool", "tool_version", "dialect", "object_ref")}
    expected_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if report_id != expected_id:
        raise NsysError("report_changed", "report material identity is invalid")
    return dict(material)


def _resources(value) -> dict:
    value = _closed(value, {"host_id", "gpu_uuids"}, label="resources")
    host_id = _text(value["host_id"], "resources.host_id", maximum=256)
    gpu_uuids = value["gpu_uuids"]
    if type(gpu_uuids) is not list or any(
        not isinstance(item, str) or not item for item in gpu_uuids
    ):
        raise NsysError("invalid_nsys_input", "resources.gpu_uuids must be a string list")
    if len(gpu_uuids) != len(set(gpu_uuids)):
        raise NsysError("invalid_nsys_input", "resources.gpu_uuids must not contain duplicates")
    return {"host_id": host_id, "gpu_uuids": sorted(gpu_uuids)}


def _nsys_tool(target: dict) -> dict:
    """Read and verify the Target-frozen Nsys executable without PATH lookup."""
    try:
        value = target["environment"]["host"]["tools"]["nsys"]
    except (KeyError, TypeError) as error:
        raise NsysError("nsys_not_frozen", "Target has no frozen Nsys executable") from error
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise NsysError("nsys_not_frozen", "Target Nsys identity is invalid")
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value["path"]))))
    if not path.is_file() or path.is_symlink():
        raise NsysError("nsys_changed", "frozen Nsys executable is unavailable")
    digest = _sha(value["sha256"], "Target Nsys SHA-256")
    if STORE.sha256_file(path) != digest:
        raise NsysError("nsys_changed", "frozen Nsys executable digest changed")
    return {"path": str(path), "sha256": digest}


def _collect_resources(target: dict, resources: dict) -> dict:
    try:
        host = target["environment"]["host"]
        expected = {
            "host_id": _text(host["host_id"], "Target host_id", maximum=256),
            "gpu_uuids": sorted(host["gpu_uuids"]),
        }
    except (KeyError, TypeError) as error:
        raise NsysError("target_invalid", "Target host resources are invalid") from error
    if any(not isinstance(item, str) or not item for item in expected["gpu_uuids"]):
        raise NsysError("target_invalid", "Target GPU identities are invalid")
    if len(expected["gpu_uuids"]) != len(set(expected["gpu_uuids"])):
        raise NsysError("target_invalid", "Target GPU identities are duplicated")
    if resources != expected:
        raise NsysError("resource_mismatch", "collect resources do not equal Target host and GPUs")
    return expected


def _resolve_collect(request: dict) -> dict:
    try:
        resolved = ADAPTER.resolve_profile_collection(
            artifact_root=request["artifact_root"],
            target_ref=request["target_ref"],
            baseline_ref=request["baseline_ref"],
            role=request["role"],
            case_id=request["case_id"],
            capability="nsys_wrap_v1",
            experiment_ref=request.get("experiment_ref"),
            correctness_ref=request.get("correctness_ref"),
        )
    except ValueError as error:
        raise NsysError("collection_rejected", str(error)) from error
    resources = _collect_resources(resolved["target"], request["resources"])
    return {**resolved, "resources": resources, "nsys_tool": _nsys_tool(resolved["target"])}


def parse_nsys_sqlite(path, tool_version: str) -> dict:
    """Extract kernel durations from exactly one Nsys SQLite 3.25 export dialect."""
    raw_path = Path(path).expanduser()
    if raw_path.suffix == ".nsys-rep":
        raise NsysError("private_report", "private .nsys-rep is not an SQLite export")
    if _VERSION.fullmatch(tool_version) is None:
        raise NsysError("unsupported_tool_version", "Nsys version is not supported")
    connection = None
    try:
        identity = raw_path.lstat()
        if not raw_path.is_file() or raw_path.is_symlink() or identity.st_size > _MAX_SQLITE_BYTES:
            raise NsysError("invalid_nsys_sqlite", "SQLite export is unsafe or exceeds size limit")
        candidate = raw_path.resolve(strict=True)
        uri = f"file:{quote(str(candidate), safe='/')}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if len(tables) > _MAX_TABLES:
            raise NsysError("invalid_nsys_sqlite", "SQLite export exceeds table limit")
        if "META_DATA_EXPORT" not in tables or "StringIds" not in tables:
            raise NsysError("missing_required_table", "Nsys SQLite is missing required metadata tables")
        metadata_columns = {row[1] for row in connection.execute("PRAGMA table_info(META_DATA_EXPORT)")}
        if {"name", "value"} - metadata_columns:
            raise NsysError("missing_required_columns", "Nsys SQLite is missing metadata columns")
        metadata = list(connection.execute("SELECT value FROM META_DATA_EXPORT WHERE name='EXPORT_SCHEMA_VERSION'"))
        if len(metadata) != 1 or metadata[0][0] != "3.25.0":
            raise NsysError("unsupported_schema", "Nsys export schema version is unsupported")
        if _KERNEL_TABLE not in tables:
            raise NsysError("missing_required_table", "Nsys SQLite is missing the kernel table")
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({_KERNEL_TABLE})")}
        if not _KERNEL_COLUMNS.issubset(columns):
            raise NsysError("missing_required_columns", "Nsys SQLite is missing required kernel columns")
        string_columns = {row[1] for row in connection.execute("PRAGMA table_info(StringIds)")}
        if {"id", "value"} - string_columns:
            raise NsysError("missing_required_columns", "Nsys SQLite is missing StringIds columns")
        rows = list(connection.execute(f"SELECT k.start, k.end, s.value FROM {_KERNEL_TABLE} k LEFT JOIN StringIds s ON k.demangledName=s.id LIMIT {_MAX_ROWS + 1}"))
    except NsysError:
        raise
    except (OSError, ValueError, sqlite3.Error) as error:
        raise NsysError("invalid_nsys_sqlite", "report is not a readable SQLite export") from error
    finally:
        if connection is not None:
            connection.close()
    if len(rows) > _MAX_ROWS:
        raise NsysError("row_limit_exceeded", "Nsys SQLite exceeds kernel row limit")
    observations = []
    for start, end, name in rows:
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not math.isfinite(float(start)) or not math.isfinite(float(end)) or end < start or not isinstance(name, str) or not name:
            raise NsysError("invalid_kernel_row", "Nsys kernel row has invalid time or name")
        observations.append({"semantic_id": "kernel.duration", "value": float(end - start), "unit": "ns", "scope": ["kernel", name], "aggregation": "single_kernel_row", "source": {"table": _KERNEL_TABLE, "start_column": "start", "end_column": "end", "name_column": "demangledName"}, "tool": {"name": "nsys", "version": tool_version}})
    if not observations:
        raise NsysError("missing_kernel_rows", "Nsys SQLite has no kernel rows")
    return {
        "observations": observations,
        "unmodeled": {
            "tables": sorted(
                tables - {_KERNEL_TABLE, "META_DATA_EXPORT", "StringIds"}
            )[:_MAX_UNMODELED],
            "kernel_columns": sorted(columns - _KERNEL_COLUMNS)[:_MAX_UNMODELED],
        },
    }


def _tool_identity(operation: str) -> dict:
    if operation not in {"analyze", "collect"}:
        raise NsysError("invalid_nsys_input", "tool identity operation is unsupported")
    files = []
    names = ["profile_nsys.py", "_invocation_runtime.py", "artifact_store.py"]
    if operation == "collect":
        names.append("workload_adapter.py")
    for name in names:
        files.append({"name": name, "sha256": STORE.sha256_file(Path(__file__).with_name(name))})
    identity = {
        "version": TOOL_IDENTITY_VERSION,
        "implementations": files,
    }
    if operation == "collect":
        identity["result_contract"] = RESULT_VERSION
    identity["digest"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def _validate_analyze(value) -> tuple[dict, Path, dict]:
    request = _closed(value, _ANALYZE_INPUT, _INPUT_OPTIONAL, "analyze input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "analyze":
        raise NsysError("invalid_nsys_input", "analyze operation is unsupported")
    root = Path(os.path.abspath(os.path.expanduser(_text(request["artifact_root"], "artifact_root"))))
    if not root.is_dir():
        raise NsysError("target_not_found", "artifact_root is unavailable")
    resources = _closed(request["resources"], {"host_id", "gpu_uuids"}, label="resources")
    if resources["gpu_uuids"] != []:
        raise NsysError("invalid_nsys_input", "read-only Nsys analysis must not request GPUs")
    normalized = {**request, "artifact_root": str(root), "resources": {"host_id": _text(resources["host_id"], "resources.host_id"), "gpu_uuids": []}}
    for field in ("operation_timeout_seconds", "command_timeout_seconds", "resource_wait_timeout_seconds", "cleanup_timeout_seconds"):
        normalized[field] = _finite(request[field], field, True)
    normalized["launch_deadline"] = _finite(request["launch_deadline"], "launch_deadline")
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(request["absolute_deadline"], "absolute_deadline")
    return normalized, root, _material(_target(root, request["target_ref"]), request["report_ref"])


def _validate_collect(value) -> tuple[dict, dict]:
    request = _closed(value, _COLLECT_INPUT, _COLLECT_OPTIONAL, "collect input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "collect":
        raise NsysError("invalid_nsys_input", "collect operation is unsupported")
    normalized = {
        **request,
        "artifact_root": str(Path(os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"]))))),
        "resources": _resources(request["resources"]),
        "role": _text(request["role"], "role", maximum=32),
        "case_id": _text(request["case_id"], "case_id", maximum=128),
    }
    for field in (
        "operation_timeout_seconds", "command_timeout_seconds",
        "resource_wait_timeout_seconds", "cleanup_timeout_seconds",
    ):
        normalized[field] = _finite(request[field], field, True)
    normalized["launch_deadline"] = _finite(request["launch_deadline"], "launch_deadline")
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(request["absolute_deadline"], "absolute_deadline")
    return normalized, _resolve_collect(normalized)


def analyze(value, *, wait_for_result: bool) -> dict:
    request, root, material = _validate_analyze(value)
    frozen = {**request, "report_material": material, "tool_identity": _tool_identity("analyze")}
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return RUNTIME.submit(root, frozen, [sys.executable, str(Path(__file__).resolve()), "_worker"], wait_for_result)


def _frozen_collect_request(request: dict, resolved: dict) -> dict:
    frozen = {
        "operation": "collect",
        "target_ref": resolved["target_ref"],
        "baseline_ref": resolved["baseline_ref"],
        "role": resolved["role"],
        "case_id": resolved["case_id"],
        "variant": resolved["variant"],
        "driver": resolved["driver"],
        "nsys_tool": resolved["nsys_tool"],
        "resources": resolved["resources"],
        "tool_identity": _tool_identity("collect"),
        "operation_timeout_seconds": request["operation_timeout_seconds"],
        "command_timeout_seconds": request["command_timeout_seconds"],
        "resource_wait_timeout_seconds": request["resource_wait_timeout_seconds"],
        "cleanup_timeout_seconds": request["cleanup_timeout_seconds"],
        "launch_deadline": request["launch_deadline"],
    }
    if resolved["role"] == "candidate":
        frozen.update({"experiment_ref": resolved["experiment_ref"], "correctness_ref": resolved["correctness_ref"]})
    for field in ("absolute_deadline", "retry_of"):
        if field in request:
            frozen[field] = request[field]
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return frozen


def collect(value, *, wait_for_result: bool) -> dict:
    request, resolved = _validate_collect(value)
    return RUNTIME.submit(
        resolved["artifact_root"],
        _frozen_collect_request(request, resolved),
        [sys.executable, str(Path(__file__).resolve()), "_worker"],
        wait_for_result,
    )


def _command_spec(argv: list[str], workspace: Path, gpu_uuids: list[str]) -> dict:
    return {
        "argv": argv,
        "cwd": str(workspace),
        "env": {},
        "output_limit_bytes": 64 * 1024,
        "required_gpu_uuids": gpu_uuids,
    }


def _record_child_failure(result: dict, child: dict, label: str) -> bool:
    if child.get("status") == "completed":
        return False
    diagnostic = (child.get("stderr", "") + "\n" + child.get("stdout", ""))[:1024]
    status = child.get("status")
    result["execution_status"] = status if status in {"timed_out", "cancelled"} else "failed"
    result["measurement_validity"] = "invalid"
    result["stop_reason"] = child.get("stop_reason", status) if status in {"timed_out", "cancelled"} else "nsys_command_failed"
    result["cleanup_status"] = child.get("cleanup_status", "unknown")
    result["diagnostic"] = {"error": f"{label} failed: {diagnostic}"[:1024]}
    return True


def _nsys_version(stdout: str) -> str:
    matches = re.findall(r"2026\.2(?:\.\d+)*", stdout)
    if len(matches) != 1 or _VERSION.fullmatch(matches[0]) is None:
        raise NsysError("unsupported_tool_version", "only Nsys 2026.2.x is supported")
    return matches[0]


def _revalidate_collect_worker(request: dict, artifact_root: Path) -> dict:
    resolved = _resolve_collect({**request, "artifact_root": str(artifact_root)})
    expected = {
        key: request[key]
        for key in (
            "target_ref", "baseline_ref", "role", "case_id", "variant",
            "driver", "nsys_tool", "resources",
        )
    }
    if request["role"] == "candidate":
        expected.update({"experiment_ref": request["experiment_ref"], "correctness_ref": request["correctness_ref"]})
    if {key: resolved[key] for key in expected} != expected:
        raise NsysError("collection_changed", "frozen collection bindings changed")
    return resolved


def _base_result(request: dict, started_epoch: float) -> dict:
    result = {
        "record_type": "profiler_result",
        "format_version": RESULT_VERSION,
        "operation": request["operation"],
        "target_ref": request["target_ref"],
        "started_at_epoch": started_epoch,
        "finished_at_epoch": None,
        "elapsed_seconds": None,
        "execution_status": "invalid",
        "measurement_validity": "invalid",
        "stop_reason": None,
        "cleanup_status": "not_required",
        "observations": [],
        "unmodeled": [],
        "provenance": {},
    }
    if request["operation"] == "analyze":
        result["report_ref"] = request["report_ref"]
    else:
        result.update(
            {
                "baseline_ref": request["baseline_ref"],
                "role": request["role"],
                "case_id": request["case_id"],
            }
        )
        if request["role"] == "candidate":
            result.update({"experiment_ref": request["experiment_ref"], "correctness_ref": request["correctness_ref"]})
    return result


def _collect_worker(request: dict, artifact_root: Path, invocation: Path, result: dict) -> None:
    acquired = RUNTIME.acquire_resources(request["resources"]["gpu_uuids"])
    if acquired.get("status") != "acquired":
        result["execution_status"] = acquired.get("status", "failed")
        result["stop_reason"] = acquired.get("stop_reason", "resource_acquisition_failed")
        result["cleanup_status"] = acquired.get("cleanup_status", "unknown")
        return
    resolved = _revalidate_collect_worker(request, artifact_root)
    workspace = invocation / "workspace"
    workspace.mkdir()
    tool = _nsys_tool(resolved["target"])
    if tool != request["nsys_tool"]:
        raise NsysError("nsys_changed", "frozen Nsys executable changed before collection")
    receipts = []
    result["provenance"] = {
        "tool": {"name": "nsys", "path": tool["path"], "sha256": tool["sha256"]},
        "parser_version": PARSER_VERSION,
        "tool_identity": request["tool_identity"],
        "command_receipts": receipts,
    }
    version_argv = [tool["path"], "--version"]
    version_result = RUNTIME.run_child(_command_spec(version_argv, workspace, []))
    receipts.append({"argv": version_argv, "result": version_result})
    if _record_child_failure(result, version_result, "Nsys version query"):
        return
    version = _nsys_version(version_result["stdout"])
    result["provenance"]["tool"]["version"] = version
    variant = ADAPTER.materialize_variant(artifact_root, workspace, resolved["variant"], "variant")
    target_inputs = ADAPTER.materialize_target_inputs(artifact_root, workspace, resolved["target"])
    driver_output = workspace / "driver-output"
    driver_output.mkdir()
    driver_request = ADAPTER.build_driver_request(
        target_id=resolved["target"]["id"],
        execution_id=os.environ["CKO_INVOCATION_ID"],
        operation="profile_nsys_collect",
        driver=resolved["driver"],
        variant=variant,
        test_suite=target_inputs["test_suite"],
        correctness=target_inputs["correctness"],
        objective=target_inputs["objective"],
        role=resolved["role"],
        mode="measure" if resolved["driver"]["execution_mode"] == "separate" else "combined",
        case={"id": resolved["case_id"]},
        sampling={"kind": "nsys_collect"},
        output_path=driver_output / "result.json",
    )
    driver_request_path = workspace / "driver-request.json"
    STORE.create_regular_bytes(
        driver_request_path,
        json.dumps(driver_request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n",
    )
    driver_argv = ADAPTER.build_argv(resolved["driver"], driver_request_path)
    prefix = workspace / "collection"
    profile_argv = [
        tool["path"], "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
        "--cpuctxsw=none", "--stats=false", "--wait=all", "--output", str(prefix),
        *driver_argv,
    ]
    tool = _nsys_tool(resolved["target"])
    if tool != request["nsys_tool"]:
        raise NsysError("nsys_changed", "frozen Nsys executable changed before profile")
    profile_result = RUNTIME.run_child(
        _command_spec(profile_argv, workspace, request["resources"]["gpu_uuids"])
    )
    receipts.append({"argv": profile_argv, "result": profile_result})
    if _record_child_failure(result, profile_result, "Nsys collection"):
        return
    reports = sorted(path for path in workspace.glob("collection*.nsys-rep") if path.is_file() and not path.is_symlink())
    report_path = prefix.with_suffix(".nsys-rep")
    if reports != [report_path]:
        raise NsysError("invalid_report_count", "Nsys collection produced zero or multiple reports")
    driver_object = STORE.freeze_path(artifact_root, driver_output, _DRIVER_OUTPUT_FREEZE_LIMITS)
    result["provenance"]["driver_output"] = driver_object
    frozen_driver_output = STORE.materialize_object(
        artifact_root, driver_object, workspace / "frozen-driver-output"
    )
    try:
        driver_result = ADAPTER.validate_driver_result(frozen_driver_output / "result.json", driver_request)
    except ValueError as error:
        raise NsysError("driver_result_invalid", str(error)) from error
    if driver_result["environment"] != resolved["target"]["environment"]["runtime"]:
        raise NsysError("environment_changed", "driver runtime identity changed during collection")
    report_object = STORE.freeze_path(artifact_root, report_path, _NSYS_ARTIFACT_FREEZE_LIMITS)
    result["provenance"]["report"] = report_object
    frozen_report = STORE.materialize_object(
        artifact_root, report_object, workspace / "frozen-report.nsys-rep"
    )
    sqlite_path = workspace / "collection.sqlite"
    export_argv = [tool["path"], "export", "--type", "sqlite", "--output", str(sqlite_path), str(frozen_report)]
    tool = _nsys_tool(resolved["target"])
    if tool != request["nsys_tool"]:
        raise NsysError("nsys_changed", "frozen Nsys executable changed before export")
    export_result = RUNTIME.run_child(_command_spec(export_argv, workspace, []))
    receipts.append({"argv": export_argv, "result": export_result})
    if _record_child_failure(result, export_result, "Nsys SQLite export"):
        return
    sqlite_object = STORE.freeze_path(artifact_root, sqlite_path, _NSYS_ARTIFACT_FREEZE_LIMITS)
    result["provenance"]["sqlite"] = sqlite_object
    frozen_sqlite = STORE.materialize_object(artifact_root, sqlite_object, workspace / "frozen-sqlite.sqlite")
    facts = parse_nsys_sqlite(frozen_sqlite, version)
    result.update(
        {
            "execution_status": "succeeded",
            "measurement_validity": "valid",
            "stop_reason": "completed",
            "cleanup_status": RUNTIME.current_cleanup_status(),
            "observations": facts["observations"],
            "unmodeled": facts["unmodeled"],
        }
    )


def _worker_main() -> int:
    root, invocation = Path(os.environ["CKO_ARTIFACT_ROOT"]), Path(os.environ["CKO_INVOCATION_DIR"])
    request, started = _strict_json(invocation / "request.json"), time.monotonic()
    result = _base_result(request, time.time())
    try:
        if request.get("tool_identity") != _tool_identity(request["operation"]):
            raise NsysError("tool_identity_changed", "implementation changed before analysis")
        if request["operation"] == "collect":
            _collect_worker(request, root, invocation, result)
        else:
            material = _material(_target(root, request["target_ref"]), request["report_ref"])
            if material != request.get("report_material"):
                raise NsysError("report_changed", "frozen report material changed")
            report = STORE.materialize_object(root, material["object_ref"], invocation / "workspace" / "report.sqlite")
            facts = parse_nsys_sqlite(report, material["tool_version"])
            result.update({"execution_status": "succeeded", "measurement_validity": "valid", "stop_reason": "completed", "observations": facts["observations"], "unmodeled": facts["unmodeled"], "provenance": {"report_ref": request["report_ref"], "report_object_ref": material["object_ref"], "tool": {"name": "nsys", "version": material["tool_version"]}, "dialect": material["dialect"], "parser_version": PARSER_VERSION, "tool_identity": request["tool_identity"]}})
    except NsysError as error:
        result["execution_status"] = "invalid"
        result["measurement_validity"] = "invalid"
        result["stop_reason"] = error.code
        if request["operation"] == "collect":
            result["cleanup_status"] = RUNTIME.current_cleanup_status()
        result["diagnostic"] = {"error": str(error)[:1024]}
    except BaseException as error:
        result.update({"execution_status": "failed", "measurement_validity": "invalid", "stop_reason": "worker_error", "diagnostic": {"error": str(error)[:1024]}})
        if request["operation"] == "collect":
            result["cleanup_status"] = RUNTIME.current_cleanup_status()
    result["finished_at_epoch"], result["elapsed_seconds"] = time.time(), time.monotonic() - started
    STORE.create_regular_json(invocation / "result.json", result)
    return 0


def _status_or_cancel(value, operation: str) -> dict:
    value = _closed(value, {"format_version", "operation", "artifact_root", "invocation_id"}, label=f"{operation} input")
    if value["format_version"] != INPUT_VERSION or value["operation"] != operation:
        raise NsysError("invalid_nsys_input", f"{operation} input is unsupported")
    return RUNTIME.status(value["artifact_root"], value["invocation_id"]) if operation == "status" else RUNTIME.cancel(value["artifact_root"], value["invocation_id"])


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["_worker"]:
        return _worker_main()
    parser = argparse.ArgumentParser(description="Analyze or collect frozen V1.4 Nsight Systems facts.")
    parser.add_argument("operation", choices=("analyze", "collect", "status", "cancel"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(arguments)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise NsysError("invalid_nsys_input", "CLI operation does not match request")
        if args.operation == "analyze":
            result = analyze(request, wait_for_result=args.wait)
        elif args.operation == "collect":
            result = collect(request, wait_for_result=args.wait)
        else:
            if args.wait:
                raise NsysError("invalid_nsys_input", "--wait is only valid for analyze or collect")
            result = _status_or_cancel(request, args.operation)
    except (NsysError, OSError, ValueError, TimeoutError) as error:
        print(json.dumps({"status": "rejected", "error_code": getattr(error, "code", "nsys_error"), "error": str(error)[:1024]}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
