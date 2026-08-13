#!/usr/bin/env python3
"""Parse one frozen, version-bound Nsight Compute CSV report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/ncu-input-v1"
RESULT_VERSION = "cuda-kernel-optimizer/profiler-result-v1"
TOOL_IDENTITY_VERSION = "cuda-kernel-optimizer/ncu-tool-v1"
PARSER_VERSION = "cuda-kernel-optimizer/ncu-csv-long-v1"

_INPUT_REQUIRED = {
    "format_version",
    "operation",
    "artifact_root",
    "target_ref",
    "report_ref",
    "kernel_name_hints",
    "resources",
    "operation_timeout_seconds",
    "command_timeout_seconds",
    "resource_wait_timeout_seconds",
    "cleanup_timeout_seconds",
    "launch_deadline",
}
_INPUT_OPTIONAL = {"absolute_deadline", "retry_of"}
_COLLECT_REQUIRED = {
    "format_version", "operation", "artifact_root", "target_ref", "baseline_ref",
    "role", "case_id", "kernel_name_hints", "resources",
    "operation_timeout_seconds", "command_timeout_seconds",
    "resource_wait_timeout_seconds", "cleanup_timeout_seconds", "launch_deadline",
}
_COLLECT_OPTIONAL = {
    "experiment_ref", "correctness_ref", "absolute_deadline", "retry_of",
}
_TARGET_REF_FIELDS = {"id", "sha256"}
_REPORT_REF_FIELDS = {"id", "sha256"}
_RESOURCE_FIELDS = {"host_id", "gpu_uuids"}
_MATERIAL_FIELDS = {
    "id",
    "sha256",
    "kind",
    "tool",
    "tool_version",
    "dialect",
    "object_ref",
}
_OBJECT_REF_REQUIRED = {"digest", "locator"}
_CSV_COLUMNS = {"Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
_VERSION = re.compile(r"2026\.2(?:\.\d+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NUMBER = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?\Z"
)
_MAX_CSV_BYTES = 16 * 1024 * 1024
_MAX_CSV_ROWS = 100_000
_MAX_UNMODELED = 128
_COLLECTION_FREEZE_LIMITS = {
    "max_files": 16,
    "max_total_bytes": 16 * 1024 * 1024,
    "max_wall_seconds": 5.0,
}

# The seven semantics with accepted, fixture-backed NCU 2026.2 long-form names.
_METRICS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": (
        "kernel.dram_throughput_pct",
        "%",
    ),
    "dram__bytes.sum": ("kernel.dram_bytes", "byte"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": (
        "kernel.occupancy_pct",
        "%",
    ),
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": (
        "kernel.sm_active_pct",
        "%",
    ),
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": (
        "kernel.tensor_pipe_pct",
        "%",
    ),
    "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct": (
        "kernel.barrier_stall_pct",
        "%",
    ),
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": (
        "kernel.long_scoreboard_pct",
        "%",
    ),
}


def _load_sibling(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load NCU dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_ncu_store")
RUNTIME = _load_sibling("_invocation_runtime.py", "cuda_optimizer_ncu_runtime")
ADAPTER = _load_sibling("workload_adapter.py", "cuda_optimizer_ncu_adapter")


class NcuError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _closed(value, required: set[str], optional: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise NcuError("invalid_ncu_input", f"{label} must be an object")
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise NcuError(
            "invalid_ncu_input",
            f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}",
        )
    return value


def _text(value, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise NcuError("invalid_ncu_input", f"{label} must be a non-empty bounded string")
    return value


def _sha256(value, label: str) -> str:
    text = _text(value, label, maximum=64)
    if _SHA256.fullmatch(text) is None:
        raise NcuError("invalid_ncu_input", f"{label} must be a lowercase SHA-256")
    return text


def _finite(value, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NcuError("invalid_ncu_input", f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise NcuError("invalid_ncu_input", f"{label} must be a finite number")
    return number


def _strict_json(path) -> dict:
    raw = STORE.read_regular_bytes(path)

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise NcuError("invalid_ncu_input", f"request contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                NcuError("invalid_ncu_input", f"request contains non-finite number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NcuError("invalid_ncu_input", "request is invalid JSON") from error
    if type(value) is not dict:
        raise NcuError("invalid_ncu_input", "request root must be an object")
    return value


def _target(root: Path, reference) -> dict:
    reference = _closed(reference, _TARGET_REF_FIELDS, set(), "target_ref")
    target_path = root / "target.json"
    try:
        payload = STORE.read_regular_bytes(target_path)
    except (OSError, ValueError) as error:
        raise NcuError("target_not_found", "target record is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != _sha256(reference["sha256"], "target_ref.sha256"):
        raise NcuError("target_changed", "target record digest changed")
    try:
        target = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NcuError("target_invalid", "target record is invalid") from error
    if (
        type(target) is not dict
        or target.get("record_type") != "target"
        or target.get("format_version") != "cuda-kernel-optimizer/target-v2"
        or target.get("id") != _text(reference["id"], "target_ref.id", maximum=128)
        or target.get("target_mode") != "diagnostic"
    ):
        raise NcuError("target_invalid", "target record identity is invalid")
    return target


def _material(target: dict, report_ref: dict) -> dict:
    report_ref = _closed(report_ref, _REPORT_REF_FIELDS, set(), "report_ref")
    report_id = _text(report_ref["id"], "report_ref.id", maximum=256)
    report_digest = _sha256(report_ref["sha256"], "report_ref.sha256")
    materials = target.get("diagnostic_materials")
    if type(materials) is not list:
        raise NcuError("report_not_found", "target has no diagnostic materials")
    matches = []
    for candidate in materials:
        if type(candidate) is dict and candidate.get("id") == report_id:
            matches.append(candidate)
    if len(matches) != 1:
        raise NcuError("report_not_found", "report_ref does not select one frozen material")
    material = _closed(matches[0], _MATERIAL_FIELDS, set(), "diagnostic material")
    if material["sha256"] != report_digest:
        raise NcuError("report_changed", "report material digest does not match report_ref")
    if material["kind"] != "report" or material["tool"] != "ncu":
        raise NcuError("unsupported_report", "material is not an NCU report")
    if material["dialect"] != "ncu-csv-long-v1":
        raise NcuError("unsupported_report", "material is not the accepted NCU CSV dialect")
    version = _text(material["tool_version"], "material.tool_version", maximum=64)
    if _VERSION.fullmatch(version) is None:
        raise NcuError("unsupported_tool_version", "only NCU 2026.2.x CSV is supported")
    object_ref = material["object_ref"]
    if type(object_ref) is not dict or not _OBJECT_REF_REQUIRED.issubset(object_ref):
        raise NcuError("report_invalid", "material object_ref is invalid")
    if object_ref["digest"] != report_digest:
        raise NcuError("report_changed", "material object_ref does not match report_ref")
    identity = {
        "kind": material["kind"],
        "tool": material["tool"],
        "tool_version": material["tool_version"],
        "dialect": material["dialect"],
        "object_ref": object_ref,
    }
    expected_id = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if report_id != expected_id:
        raise NcuError("report_changed", "report material identity is invalid")
    return {
        "id": report_id,
        "sha256": report_digest,
        "kind": "report",
        "tool": "ncu",
        "tool_version": version,
        "dialect": "ncu-csv-long-v1",
        "object_ref": dict(object_ref),
    }


def _resources(value) -> dict:
    value = _closed(value, _RESOURCE_FIELDS, set(), "resources")
    host_id = _text(value["host_id"], "resources.host_id", maximum=256)
    gpu_uuids = value["gpu_uuids"]
    if type(gpu_uuids) is not list or any(not isinstance(item, str) or not item for item in gpu_uuids):
        raise NcuError("invalid_ncu_input", "resources.gpu_uuids must be a string list")
    if len(gpu_uuids) != len(set(gpu_uuids)):
        raise NcuError("invalid_ncu_input", "resources.gpu_uuids must not contain duplicates")
    return {"host_id": host_id, "gpu_uuids": sorted(gpu_uuids)}


def _hints(value) -> list[str]:
    if type(value) is not list or any(not isinstance(item, str) or not item or len(item) > 512 for item in value):
        raise NcuError("invalid_ncu_input", "kernel_name_hints must be a bounded string list")
    if len(value) > 64 or len(value) != len(set(value)):
        raise NcuError("invalid_ncu_input", "kernel_name_hints must be unique and bounded")
    return list(value)


def _tool_identity(operation: str) -> dict:
    if operation not in {"analyze", "collect"}:
        raise NcuError("invalid_ncu_input", "tool identity operation is unsupported")
    implementations = []
    names = ["profile_ncu.py", "_invocation_runtime.py", "artifact_store.py"]
    if operation == "collect":
        names.append("workload_adapter.py")
    for name in names:
        path = Path(__file__).with_name(name)
        implementations.append({"name": name, "sha256": STORE.sha256_file(path)})
    identity = {
        "version": TOOL_IDENTITY_VERSION,
        "result_contract": RESULT_VERSION,
        "implementations": implementations,
    }
    identity["digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity


def _validate_analyze(value) -> tuple[dict, Path, dict]:
    request = _closed(value, _INPUT_REQUIRED, _INPUT_OPTIONAL, "analyze input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "analyze":
        raise NcuError("invalid_ncu_input", "analyze input version or operation is unsupported")
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"]))))
    if not root.is_dir():
        raise NcuError("target_not_found", "artifact_root is unavailable")
    target = _target(root, request["target_ref"])
    material = _material(target, request["report_ref"])
    normalized = {
        **request,
        "artifact_root": str(root),
        "kernel_name_hints": _hints(request["kernel_name_hints"]),
        "resources": _resources(request["resources"]),
    }
    if normalized["resources"]["gpu_uuids"]:
        raise NcuError(
            "invalid_ncu_input",
            "read-only NCU analysis must not request GPU resources",
        )
    for field in (
        "operation_timeout_seconds",
        "command_timeout_seconds",
        "resource_wait_timeout_seconds",
        "cleanup_timeout_seconds",
    ):
        normalized[field] = _finite(request[field], field, positive=True)
    normalized["launch_deadline"] = _finite(request["launch_deadline"], "launch_deadline")
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(request["absolute_deadline"], "absolute_deadline")
    return normalized, root, material


def _ncu_tool(target: dict) -> dict:
    """Read and verify the Target-frozen NCU executable without consulting PATH."""
    try:
        value = target["environment"]["host"]["tools"]["ncu"]
    except (KeyError, TypeError) as error:
        raise NcuError("ncu_not_frozen", "Target has no frozen NCU executable") from error
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise NcuError("ncu_not_frozen", "Target NCU identity is invalid")
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value["path"]))))
    if not path.is_file() or path.is_symlink():
        raise NcuError("ncu_changed", "frozen NCU executable is unavailable")
    digest = _sha256(value["sha256"], "Target NCU SHA-256")
    if STORE.sha256_file(path) != digest:
        raise NcuError("ncu_changed", "frozen NCU executable digest changed")
    return {"path": str(path), "sha256": digest}


def _collect_resources(target: dict, resources: dict) -> dict:
    try:
        host = target["environment"]["host"]
        expected = {
            "host_id": _text(host["host_id"], "Target host_id", maximum=256),
            "gpu_uuids": sorted(host["gpu_uuids"]),
        }
    except (KeyError, TypeError) as error:
        raise NcuError("target_invalid", "Target host resources are invalid") from error
    if any(not isinstance(item, str) or not item for item in expected["gpu_uuids"]):
        raise NcuError("target_invalid", "Target GPU identities are invalid")
    if len(expected["gpu_uuids"]) != len(set(expected["gpu_uuids"])):
        raise NcuError("target_invalid", "Target GPU identities are duplicated")
    if resources != expected:
        raise NcuError("resource_mismatch", "collect resources do not equal Target host and GPUs")
    return expected


def _resolve_collect(request: dict) -> dict:
    try:
        resolved = ADAPTER.resolve_profile_collection(
            artifact_root=request["artifact_root"],
            target_ref=request["target_ref"],
            baseline_ref=request["baseline_ref"],
            role=request["role"],
            case_id=request["case_id"],
            capability="ncu_wrap_v1",
            experiment_ref=request.get("experiment_ref"),
            correctness_ref=request.get("correctness_ref"),
        )
    except ValueError as error:
        raise NcuError("collection_rejected", str(error)) from error
    resources = _collect_resources(resolved["target"], request["resources"])
    return {**resolved, "resources": resources, "ncu_tool": _ncu_tool(resolved["target"])}


def _validate_collect(value) -> tuple[dict, dict]:
    request = _closed(value, _COLLECT_REQUIRED, _COLLECT_OPTIONAL, "collect input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "collect":
        raise NcuError("invalid_ncu_input", "collect input version or operation is unsupported")
    normalized = {
        **request,
        "artifact_root": str(Path(os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"]))))),
        "kernel_name_hints": _hints(request["kernel_name_hints"]),
        "resources": _resources(request["resources"]),
        "role": _text(request["role"], "role", maximum=32),
        "case_id": _text(request["case_id"], "case_id", maximum=128),
    }
    for field in (
        "operation_timeout_seconds", "command_timeout_seconds",
        "resource_wait_timeout_seconds", "cleanup_timeout_seconds",
    ):
        normalized[field] = _finite(request[field], field, positive=True)
    normalized["launch_deadline"] = _finite(request["launch_deadline"], "launch_deadline")
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(request["absolute_deadline"], "absolute_deadline")
    return normalized, _resolve_collect(normalized)


def _frozen_request(request: dict, material: dict) -> dict:
    frozen = {
        "operation": "analyze",
        "target_ref": request["target_ref"],
        "report_ref": request["report_ref"],
        "report_material": material,
        "kernel_name_hints": request["kernel_name_hints"],
        "resources": request["resources"],
        "tool_identity": _tool_identity("analyze"),
        "operation_timeout_seconds": request["operation_timeout_seconds"],
        "command_timeout_seconds": request["command_timeout_seconds"],
        "resource_wait_timeout_seconds": request["resource_wait_timeout_seconds"],
        "cleanup_timeout_seconds": request["cleanup_timeout_seconds"],
        "launch_deadline": request["launch_deadline"],
    }
    for field in ("absolute_deadline", "retry_of"):
        if field in request:
            frozen[field] = request[field]
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return frozen


def _frozen_collect_request(request: dict, resolved: dict) -> dict:
    frozen = {
        "operation": "collect",
        "target_ref": resolved["target_ref"],
        "baseline_ref": resolved["baseline_ref"],
        "role": resolved["role"],
        "case_id": resolved["case_id"],
        "variant": resolved["variant"],
        "driver": resolved["driver"],
        "ncu_tool": resolved["ncu_tool"],
        "kernel_name_hints": request["kernel_name_hints"],
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


def _number(value: str, label: str) -> float:
    if not _NUMBER.fullmatch(value):
        raise NcuError("invalid_metric_value", f"{label} is not a finite number")
    try:
        number = float(value.replace(",", ""))
    except (TypeError, ValueError) as error:
        raise NcuError("invalid_metric_value", f"{label} is not a finite number") from error
    if not math.isfinite(number):
        raise NcuError("invalid_metric_value", f"{label} is not a finite number")
    return number


def parse_ncu_csv(csv_text: str, tool_version: str, kernel_name_hints: list[str]) -> dict:
    """Return only the fixed 2026.2 metric observations from a long CSV export."""
    if _VERSION.fullmatch(tool_version) is None:
        raise NcuError("unsupported_tool_version", "only NCU 2026.2.x CSV is supported")
    if not isinstance(csv_text, str) or len(csv_text.encode("utf-8")) > _MAX_CSV_BYTES:
        raise NcuError("invalid_ncu_csv", "NCU CSV is invalid or exceeds the byte limit")
    reader = csv.DictReader(io.StringIO(csv_text))
    if (
        reader.fieldnames is None
        or not _CSV_COLUMNS.issubset(set(reader.fieldnames))
        or len(reader.fieldnames) != len(set(reader.fieldnames))
    ):
        raise NcuError("missing_required_column", "NCU CSV is missing required long-form columns")
    unmodeled_columns = sorted(set(reader.fieldnames) - _CSV_COLUMNS)
    values = {name: [] for name in _METRICS}
    unmodeled = set()
    seen_rows = 0
    matching_rows = 0
    for row in reader:
        seen_rows += 1
        if seen_rows > _MAX_CSV_ROWS:
            raise NcuError("invalid_ncu_csv", "NCU CSV exceeds the row limit")
        if None in row:
            raise NcuError("invalid_ncu_csv", "NCU CSV row has an unexpected column")
        kernel_name = row["Kernel Name"].strip()
        if kernel_name_hints and not any(hint in kernel_name for hint in kernel_name_hints):
            continue
        matching_rows += 1
        name = row["Metric Name"].strip()
        unit = row["Metric Unit"].strip()
        if name not in _METRICS:
            unmodeled.add(name)
            continue
        semantic_id, expected_unit = _METRICS[name]
        if unit != expected_unit:
            raise NcuError("unexpected_unit", f"known NCU metric has an unexpected_unit: {name}")
        values[name].append(_number(row["Metric Value"].strip(), f"metric {name}"))
    if not matching_rows:
        raise NcuError("missing_matching_kernel", "NCU CSV has no matching kernel rows")
    if not any(values.values()):
        raise NcuError("missing_modeled_metric", "NCU CSV has no modeled metrics")
    observations = []
    for name, (semantic_id, unit) in sorted(_METRICS.items(), key=lambda item: item[1][0]):
        if not values[name]:
            continue
        observations.append(
            {
                "semantic_id": semantic_id,
                "value": statistics.fmean(values[name]),
                "unit": unit,
                "scope": ["kernel"],
                "aggregation": "mean_across_matching_rows",
                "source_metric": name,
                "tool": {"name": "ncu", "version": tool_version},
            }
        )
    unmodeled_facts = [
        {"column_name": name, "reason": "unmodeled_column"}
        for name in unmodeled_columns
    ]
    unmodeled_facts.extend(
        {"metric_name": name, "reason": "unknown_metric"}
        for name in sorted(unmodeled)
    )
    return {
        "observations": observations,
        "unmodeled": unmodeled_facts[:_MAX_UNMODELED],
    }


def analyze(value, *, wait_for_result: bool) -> dict:
    request, root, material = _validate_analyze(value)
    frozen = _frozen_request(request, material)
    return RUNTIME.submit(
        root,
        frozen,
        [sys.executable, str(Path(__file__).resolve()), "_worker"],
        wait_for_result,
    )


def collect(value, *, wait_for_result: bool) -> dict:
    request, resolved = _validate_collect(value)
    frozen = _frozen_collect_request(request, resolved)
    return RUNTIME.submit(
        resolved["artifact_root"],
        frozen,
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
    """Publish a failed child fact without discarding its runtime outcome."""
    if child.get("status") == "completed":
        return False
    diagnostic = (child.get("stderr", "") + "\n" + child.get("stdout", ""))[:1024]
    status = child.get("status")
    if "ERR_NVGPUCTRPERM" in diagnostic:
        result["execution_status"] = "failed"
        result["stop_reason"] = "ncu_counter_access_denied"
    elif status in {"timed_out", "cancelled"}:
        result["execution_status"] = status
        result["stop_reason"] = child.get("stop_reason", status)
    else:
        result["execution_status"] = "failed"
        result["stop_reason"] = "ncu_command_failed"
    result["measurement_validity"] = "invalid"
    result["cleanup_status"] = child.get("cleanup_status", "unknown")
    result["diagnostic"] = {"error": f"{label} failed: {diagnostic}"[:1024]}
    return True


def _ncu_version(stdout: str) -> str:
    matches = re.findall(r"2026\.2(?:\.\d+)*", stdout)
    if len(matches) != 1 or _VERSION.fullmatch(matches[0]) is None:
        raise NcuError("unsupported_tool_version", "only NCU 2026.2.x is supported")
    return matches[0]


def _revalidate_collect_worker(request: dict, artifact_root: Path) -> dict:
    resolved = _resolve_collect({**request, "artifact_root": str(artifact_root)})
    expected = {
        key: request[key]
        for key in (
            "target_ref", "baseline_ref", "role", "case_id", "variant",
            "driver", "ncu_tool", "resources",
        )
    }
    if request["role"] == "candidate":
        expected.update({"experiment_ref": request["experiment_ref"], "correctness_ref": request["correctness_ref"]})
    actual = {key: resolved[key] for key in expected}
    if actual != expected:
        raise NcuError("collection_changed", "frozen collection bindings changed")
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
        "execution_status": None,
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


def _finish(result: dict, started_mono: float) -> dict:
    result["finished_at_epoch"] = time.time()
    result["elapsed_seconds"] = time.monotonic() - started_mono
    return result


def _collect_worker(request: dict, artifact_root: Path, invocation_dir: Path, result: dict) -> None:
    acquired = RUNTIME.acquire_resources(request["resources"]["gpu_uuids"])
    if acquired.get("status") != "acquired":
        result["execution_status"] = acquired.get("status", "failed")
        result["stop_reason"] = acquired.get("stop_reason", "resource_acquisition_failed")
        result["cleanup_status"] = acquired.get("cleanup_status", "unknown")
        return
    resolved = _revalidate_collect_worker(request, artifact_root)
    workspace = invocation_dir / "workspace"
    workspace.mkdir()
    tool = _ncu_tool(resolved["target"])
    if tool != request["ncu_tool"]:
        raise NcuError("ncu_changed", "frozen NCU executable changed before collection")
    receipts = []
    result["provenance"] = {
        "tool": {"name": "ncu", "path": tool["path"], "sha256": tool["sha256"]},
        "metrics": list(_METRICS),
        "parser_version": PARSER_VERSION,
        "tool_identity": request["tool_identity"],
        "command_receipts": receipts,
    }
    version_argv = [tool["path"], "--version"]
    version_result = RUNTIME.run_child(
        _command_spec(version_argv, workspace, [])
    )
    receipts.append({"argv": version_argv, "result": version_result})
    if _record_child_failure(result, version_result, "NCU version query"):
        return
    version = _ncu_version(version_result["stdout"])
    result["provenance"]["tool"]["version"] = version
    variant = ADAPTER.materialize_variant(
        artifact_root, workspace, resolved["variant"], "variant"
    )
    target_inputs = ADAPTER.materialize_target_inputs(
        artifact_root, workspace, resolved["target"]
    )
    driver_output = workspace / "driver-output"
    driver_output.mkdir()
    driver_request = ADAPTER.build_driver_request(
        target_id=resolved["target"]["id"],
        execution_id=os.environ["CKO_INVOCATION_ID"],
        operation="profile_ncu_collect",
        driver=resolved["driver"],
        subjects=[{"role": resolved["role"], "variant": variant}],
        test_suite=target_inputs["test_suite"],
        correctness=target_inputs["correctness"],
        objective=target_inputs["objective"],
        acquisition={
            "lifecycle": "isolated_process",
            "shared_state": [],
            "rebuilt_state": ["process"],
        },
        case={"id": resolved["case_id"]},
        sampling={"kind": "ncu_collect"},
        output_path=driver_output / "result.json",
    )
    driver_request_path = workspace / "driver-request.json"
    STORE.create_regular_bytes(
        driver_request_path,
        json.dumps(driver_request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n",
    )
    driver_argv = ADAPTER.build_argv(resolved["driver"], driver_request_path)
    report_path = workspace / "collection.ncu-rep"
    collect_argv = [
        tool["path"], "--config-file", "off", "--metrics", ",".join(_METRICS),
        "--target-processes", "all", "--export", str(report_path), *driver_argv,
    ]
    tool = _ncu_tool(resolved["target"])
    if tool != request["ncu_tool"]:
        raise NcuError("ncu_changed", "frozen NCU executable changed before collection")
    collect_result = RUNTIME.run_child(
        _command_spec(collect_argv, workspace, request["resources"]["gpu_uuids"])
    )
    receipts.append({"argv": collect_argv, "result": collect_result})
    if _record_child_failure(result, collect_result, "NCU collection"):
        return
    driver_object = STORE.freeze_path(
        artifact_root, driver_output, _COLLECTION_FREEZE_LIMITS
    )
    result["provenance"]["driver_output"] = driver_object
    frozen_driver_output = STORE.materialize_object(
        artifact_root, driver_object, workspace / "frozen-driver-output"
    )
    try:
        driver_result = ADAPTER.validate_driver_result(
            frozen_driver_output / "result.json", driver_request
        )
    except ValueError as error:
        raise NcuError("driver_result_invalid", str(error)) from error
    if driver_result["environment"] != resolved["target"]["environment"]["runtime"]:
        raise NcuError("environment_changed", "driver runtime identity changed during collection")
    report_object = STORE.freeze_path(
        artifact_root, report_path, _COLLECTION_FREEZE_LIMITS
    )
    result["provenance"]["report"] = report_object
    frozen_report = STORE.materialize_object(
        artifact_root, report_object, workspace / "frozen-report"
    )
    csv_path = workspace / "collection.csv"
    import_argv = [
        tool["path"], "--config-file", "off", "--import", str(frozen_report),
        "--csv", "--page", "raw", "--print-units", "base",
        "--print-metric-name", "name", "--log-file", str(csv_path),
    ]
    tool = _ncu_tool(resolved["target"])
    if tool != request["ncu_tool"]:
        raise NcuError("ncu_changed", "frozen NCU executable changed before import")
    import_result = RUNTIME.run_child(_command_spec(import_argv, workspace, []))
    receipts.append({"argv": import_argv, "result": import_result})
    if _record_child_failure(result, import_result, "NCU CSV import"):
        return
    csv_object = STORE.freeze_path(
        artifact_root, csv_path, _COLLECTION_FREEZE_LIMITS
    )
    result["provenance"]["csv"] = csv_object
    frozen_csv = STORE.materialize_object(
        artifact_root, csv_object, workspace / "frozen-csv"
    )
    csv_bytes = STORE.read_regular_bytes(frozen_csv, maximum_bytes=_MAX_CSV_BYTES)
    try:
        csv_text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NcuError("invalid_ncu_csv", "NCU CSV is not UTF-8") from error
    facts = parse_ncu_csv(csv_text, version, request["kernel_name_hints"])
    result["execution_status"] = "succeeded"
    result["measurement_validity"] = "valid"
    result["stop_reason"] = "completed"
    result["cleanup_status"] = RUNTIME.current_cleanup_status()
    result["observations"] = facts["observations"]
    result["unmodeled"] = facts["unmodeled"]
    result["provenance"].update(
        {
            "tool": {"name": "ncu", "version": version, "path": tool["path"], "sha256": tool["sha256"]},
        }
    )


def _worker_main() -> int:
    artifact_root = Path(os.environ["CKO_ARTIFACT_ROOT"])
    invocation_dir = Path(os.environ["CKO_INVOCATION_DIR"])
    request = _strict_json(invocation_dir / "request.json")
    started_epoch = time.time()
    started_mono = time.monotonic()
    result = _base_result(request, started_epoch)
    try:
        if request.get("tool_identity") != _tool_identity(request["operation"]):
            result["execution_status"] = "invalid"
            result["stop_reason"] = "tool_identity_changed"
        elif request.get("operation") == "collect":
            _collect_worker(request, artifact_root, invocation_dir, result)
        else:
            target = _target(artifact_root, request["target_ref"])
            material = _material(target, request["report_ref"])
            if material != request.get("report_material"):
                raise NcuError("report_changed", "frozen report material changed before analysis")
            workspace = invocation_dir / "workspace"
            workspace.mkdir()
            report_path = STORE.materialize_object(
                artifact_root, material["object_ref"], workspace / "report.csv"
            )
            csv_bytes = STORE.read_regular_bytes(report_path)
            try:
                csv_text = csv_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise NcuError("invalid_ncu_csv", "NCU CSV is not UTF-8") from error
            facts = parse_ncu_csv(csv_text, material["tool_version"], request["kernel_name_hints"])
            result["execution_status"] = "succeeded"
            result["measurement_validity"] = "valid"
            result["stop_reason"] = "completed"
            result["observations"] = facts["observations"]
            result["unmodeled"] = facts["unmodeled"]
            result["provenance"] = {
                "report_ref": request["report_ref"],
                "report_object_ref": material["object_ref"],
                "tool": {"name": "ncu", "version": material["tool_version"]},
                "dialect": material["dialect"],
                "parser_version": PARSER_VERSION,
                "tool_identity": request["tool_identity"],
            }
    except NcuError as error:
        result["execution_status"] = "invalid"
        result["measurement_validity"] = "invalid"
        result["stop_reason"] = error.code
        result["cleanup_status"] = RUNTIME.current_cleanup_status()
        result["diagnostic"] = {"error": str(error)[:1024]}
    except BaseException as error:
        result["execution_status"] = "failed"
        result["measurement_validity"] = "invalid"
        result["stop_reason"] = "worker_error"
        result["cleanup_status"] = RUNTIME.current_cleanup_status()
        result["diagnostic"] = {"error": str(error)[:1024]}
    STORE.create_regular_json(invocation_dir / "result.json", _finish(result, started_mono))
    return 0


def _status_or_cancel(value, operation: str) -> dict:
    fields = {"format_version", "operation", "artifact_root", "invocation_id"}
    value = _closed(value, fields, set(), f"{operation} input")
    if value["format_version"] != INPUT_VERSION or value["operation"] != operation:
        raise NcuError("invalid_ncu_input", f"{operation} input is unsupported")
    return (
        RUNTIME.status(value["artifact_root"], value["invocation_id"])
        if operation == "status"
        else RUNTIME.cancel(value["artifact_root"], value["invocation_id"])
    )


def _emit_error(error: BaseException) -> int:
    code = error.code if isinstance(error, NcuError) else "ncu_error"
    print(json.dumps({"status": "rejected", "error_code": code, "error": str(error)[:1024]}, sort_keys=True), file=sys.stderr)
    return 2


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["_worker"]:
        return _worker_main()
    parser = argparse.ArgumentParser(description="Analyze or collect frozen V1.4 NCU facts.")
    parser.add_argument("operation", choices=("analyze", "collect", "status", "cancel"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(arguments)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise NcuError("invalid_ncu_input", "CLI operation does not match request")
        if args.operation == "analyze":
            result = analyze(request, wait_for_result=args.wait)
        elif args.operation == "collect":
            result = collect(request, wait_for_result=args.wait)
        else:
            if args.wait:
                raise NcuError("invalid_ncu_input", "--wait is only valid for analyze or collect")
            result = _status_or_cancel(request, args.operation)
    except (NcuError, OSError, ValueError, TimeoutError) as error:
        return _emit_error(error)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
