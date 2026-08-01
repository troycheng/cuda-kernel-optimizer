#!/usr/bin/env python3
"""Analyze one frozen, version-bound PyTorch Chrome trace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/pytorch-input-v1"
RESULT_VERSION = "cuda-kernel-optimizer/profiler-result-v1"
PARSER_VERSION = "chrome-trace-v1"
_DIALECT = "chrome-trace-v1"
_VERSION = re.compile(r"2\.13(?:\.\d+)*(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INPUT = {
    "format_version", "operation", "artifact_root", "target_ref", "report_ref",
    "resources", "operation_timeout_seconds", "command_timeout_seconds",
    "resource_wait_timeout_seconds", "cleanup_timeout_seconds", "launch_deadline",
}
_OPTIONAL = {"absolute_deadline", "retry_of"}
_COLLECT = {
    "format_version", "operation", "artifact_root", "target_ref", "baseline_ref",
    "role", "case_id", "resources", "operation_timeout_seconds",
    "command_timeout_seconds", "resource_wait_timeout_seconds",
    "cleanup_timeout_seconds", "launch_deadline",
}
_COLLECT_OPTIONAL = {"experiment_ref", "correctness_ref", "absolute_deadline", "retry_of"}
_MATERIAL = {"id", "sha256", "kind", "tool", "tool_version", "dialect", "object_ref"}
_OBJECT_REF = {"digest", "locator", "source_kind", "file_count", "total_bytes"}
_COMPLETE_FIELDS = {"name", "cat", "ph", "pid", "tid", "ts", "dur"}
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_TRACE_BYTES = 512 * 1024 * 1024
_MAX_TOP_LEVEL_FIELDS = 32
_MAX_EVENTS = 100_000
_MAX_EVENT_FIELDS = 32
_MAX_TEXT = 512
_MAX_OUTPUT = 10_000
_MAX_UNMODELED = 128
_DRIVER_OUTPUT_FREEZE_LIMITS = {
    "max_files": 16,
    "max_total_bytes": 256 * 1024 * 1024,
    "max_wall_seconds": 30.0,
}


def _load_sibling(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_pytorch_store")
RUNTIME = _load_sibling("_invocation_runtime.py", "cuda_optimizer_pytorch_runtime")
ADAPTER = _load_sibling("workload_adapter.py", "cuda_optimizer_pytorch_adapter")


class PyTorchError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _closed(value, required, optional=(), label="value") -> dict:
    if type(value) is not dict:
        raise PyTorchError("invalid_pytorch_input", f"{label} must be an object")
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    if missing or unknown:
        raise PyTorchError("invalid_pytorch_input", f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}")
    return value


def _text(value, label: str, maximum=_MAX_TEXT) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise PyTorchError("invalid_pytorch_input", f"{label} must be a non-empty bounded string")
    return value


def _sha(value, label: str) -> str:
    value = _text(value, label, 64)
    if _SHA256.fullmatch(value) is None:
        raise PyTorchError("invalid_pytorch_input", f"{label} must be a SHA-256")
    return value


def _finite(value, label: str, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PyTorchError("invalid_pytorch_input", f"{label} must be finite")
    if positive and value <= 0:
        raise PyTorchError("invalid_pytorch_input", f"{label} must be positive")
    return float(value)


def _torch_version(value, label: str) -> str:
    value = _text(value, label, 128)
    if _VERSION.fullmatch(value) is None:
        raise PyTorchError("unsupported_tool_version", "PyTorch version is not supported")
    return value


def _strict_json_bytes(
    payload: bytes,
    label: str,
    *,
    maximum_bytes: int = _MAX_TRACE_BYTES,
) -> dict:
    if len(payload) > maximum_bytes:
        raise PyTorchError("trace_limit_exceeded", f"{label} exceeds byte limit")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise PyTorchError("invalid_pytorch_trace", f"{label} contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PyTorchError("invalid_pytorch_trace", f"{label} contains non-finite number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PyTorchError("invalid_pytorch_trace", f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise PyTorchError("invalid_pytorch_trace", f"{label} must be an object")
    return value


def _strict_json(path) -> dict:
    try:
        return _strict_json_bytes(
            STORE.read_regular_bytes(
                path,
                maximum_bytes=_MAX_REQUEST_BYTES,
            ),
            "request",
            maximum_bytes=_MAX_REQUEST_BYTES,
        )
    except PyTorchError as error:
        raise PyTorchError("invalid_pytorch_input", str(error)) from error


def _target(root: Path, reference) -> dict:
    reference = _closed(reference, {"id", "sha256"}, label="target_ref")
    try:
        payload = STORE.read_regular_bytes(root / "target.json")
    except (OSError, ValueError) as error:
        raise PyTorchError("target_not_found", "target record is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != _sha(reference["sha256"], "target_ref.sha256"):
        raise PyTorchError("target_changed", "target record digest changed")
    try:
        target = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PyTorchError("target_invalid", "target record is invalid") from error
    if (
        type(target) is not dict
        or target.get("record_type") != "target"
        or target.get("format_version") != "cuda-kernel-optimizer/target-v1"
        or target.get("id") != _text(reference["id"], "target_ref.id")
        or target.get("target_mode") != "diagnostic"
    ):
        raise PyTorchError("target_invalid", "target is not a frozen diagnostic target")
    return target


def _material(target: dict, reference) -> dict:
    reference = _closed(reference, {"id", "sha256"}, label="report_ref")
    report_id = _text(reference["id"], "report_ref.id")
    digest = _sha(reference["sha256"], "report_ref.sha256")
    matches = [item for item in target.get("diagnostic_materials", []) if type(item) is dict and item.get("id") == report_id]
    if len(matches) != 1:
        raise PyTorchError("report_not_found", "report_ref does not select one frozen material")
    material = _closed(matches[0], _MATERIAL, label="diagnostic material")
    if material["sha256"] != digest or material["kind"] != "report" or material["tool"] != "pytorch_profiler":
        raise PyTorchError("unsupported_report", "material is not the requested PyTorch profiler report")
    if material["dialect"] != _DIALECT:
        raise PyTorchError("unsupported_report", "material is not the accepted Chrome trace dialect")
    _torch_version(material["tool_version"], "diagnostic material.tool_version")
    object_ref = material["object_ref"]
    expected_locator = str(Path("objects") / "sha256" / digest)
    if type(object_ref) is not dict or set(object_ref) - _OBJECT_REF or object_ref.get("digest") != digest or object_ref.get("locator") != expected_locator:
        raise PyTorchError("report_changed", "report object reference is invalid")
    identity = {key: material[key] for key in ("kind", "tool", "tool_version", "dialect", "object_ref")}
    expected_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if report_id != expected_id:
        raise PyTorchError("report_changed", "report material identity is invalid")
    return dict(material)


def _trace_text(value, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT:
        raise PyTorchError("invalid_trace_field", f"{label} must be a non-empty bounded string")
    return value


def _trace_integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PyTorchError("invalid_trace_field", f"{label} must be an integer")
    return value


def _trace_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PyTorchError("invalid_trace_field", f"{label} must be finite")
    return float(value)


def parse_chrome_trace(trace: dict, tool_version: str) -> dict:
    """Return one exact observation per complete event in a Chrome Trace v1 object."""
    tool_version = _torch_version(tool_version, "PyTorch version")
    if (
        type(trace) is not dict
        or len(trace) > _MAX_TOP_LEVEL_FIELDS
        or not {"schemaVersion", "traceEvents"}.issubset(trace)
        or type(trace.get("schemaVersion")) is not int
        or trace["schemaVersion"] != 1
    ):
        raise PyTorchError("unsupported_schema", "Chrome trace schema is unsupported")
    events = trace["traceEvents"]
    if type(events) is not list:
        raise PyTorchError("invalid_pytorch_trace", "traceEvents must be a list")
    if len(events) > _MAX_EVENTS:
        raise PyTorchError("event_limit_exceeded", "Chrome trace exceeds event limit")
    observations, phases, unmodeled_complete = [], {}, {}
    for index, event in enumerate(events):
        if type(event) is not dict or not event or len(event) > _MAX_EVENT_FIELDS:
            raise PyTorchError("invalid_trace_event", f"trace event {index} is invalid")
        phase = _trace_text(event.get("ph"), f"trace event {index}.ph")
        if phase != "X":
            phases[phase] = phases.get(phase, 0) + 1
            continue
        missing = _COMPLETE_FIELDS - set(event)
        if missing:
            raise PyTorchError("missing_trace_field", f"trace event {index} missing={sorted(missing)}")
        name = _trace_text(event["name"], f"trace event {index}.name")
        category = _trace_text(event["cat"], f"trace event {index}.cat")
        if (
            isinstance(event["pid"], bool)
            or isinstance(event["tid"], bool)
            or not isinstance(event["pid"], int)
            or not isinstance(event["tid"], int)
        ):
            if category != "Trace":
                raise PyTorchError(
                    "invalid_trace_field",
                    f"trace event {index} has a non-numeric process or thread id",
                )
            unmodeled_complete[category] = (
                unmodeled_complete.get(category, 0) + 1
            )
            continue
        pid = _trace_integer(event["pid"], f"trace event {index}.pid")
        tid = _trace_integer(event["tid"], f"trace event {index}.tid")
        start = _trace_number(event["ts"], f"trace event {index}.ts")
        duration = _trace_number(event["dur"], f"trace event {index}.dur")
        if duration < 0 or not math.isfinite(start + duration):
            raise PyTorchError("invalid_trace_field", f"trace event {index} has invalid duration")
        observations.append(
            {
                "semantic_id": "pytorch.trace.complete_event",
                "value": duration,
                "unit": "us",
                "interval": {"start_us": start, "end_us": start + duration},
                "duration_us": duration,
                "category": category,
                "name": name,
                "scope": ["process", pid, "thread", tid],
                "aggregation": "single_complete_event",
                "source": {"phase": "X", "timestamp_unit": "us"},
                "tool": {"name": "pytorch_profiler", "version": tool_version},
            }
        )
        if len(observations) > _MAX_OUTPUT:
            raise PyTorchError("output_limit_exceeded", "Chrome trace observations exceed output limit")
    if not observations:
        raise PyTorchError("missing_complete_events", "Chrome trace has no complete events")
    unmodeled = []
    extra_top_level = sorted(set(trace) - {"schemaVersion", "traceEvents"})
    if extra_top_level:
        unmodeled.append(
            {
                "kind": "top_level_metadata",
                "fields": extra_top_level[:_MAX_UNMODELED],
            }
        )
    unmodeled.extend(
        {"kind": "event_phase", "phase": phase, "count": phases[phase]}
        for phase in sorted(phases)
    )
    unmodeled.extend(
        {
            "kind": "complete_event_scope",
            "category": category,
            "reason": "non_numeric_process_or_thread_id",
            "count": unmodeled_complete[category],
        }
        for category in sorted(unmodeled_complete)
    )
    return {
        "observations": observations,
        "unmodeled": unmodeled[:_MAX_UNMODELED],
    }


def _tool_identity(operation: str) -> dict:
    if operation not in {"analyze", "collect"}:
        raise PyTorchError("invalid_pytorch_input", "tool identity operation is unsupported")
    names = ["profile_pytorch.py", "_invocation_runtime.py", "artifact_store.py"]
    if operation == "collect":
        names.append("workload_adapter.py")
    implementations = [{"name": name, "sha256": STORE.sha256_file(Path(__file__).with_name(name))} for name in names]
    identity = {"version": "cuda-kernel-optimizer/pytorch-tool-v1", "result_contract": RESULT_VERSION, "implementations": implementations}
    identity["digest"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return identity


def _validate_analyze(value) -> tuple[dict, Path, dict]:
    request = _closed(value, _INPUT, _OPTIONAL, "analyze input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "analyze":
        raise PyTorchError("invalid_pytorch_input", "analyze operation is unsupported")
    root = Path(os.path.abspath(os.path.expanduser(_text(request["artifact_root"], "artifact_root", 4096))))
    if not root.is_dir():
        raise PyTorchError("target_not_found", "artifact_root is unavailable")
    resources = _closed(request["resources"], {"host_id", "gpu_uuids"}, label="resources")
    if resources["gpu_uuids"] != []:
        raise PyTorchError("invalid_pytorch_input", "read-only PyTorch analysis must not request GPU resources")
    normalized = {**request, "artifact_root": str(root), "resources": {"host_id": _text(resources["host_id"], "resources.host_id"), "gpu_uuids": []}}
    for field in ("operation_timeout_seconds", "command_timeout_seconds", "resource_wait_timeout_seconds", "cleanup_timeout_seconds"):
        normalized[field] = _finite(request[field], field, True)
    normalized["launch_deadline"] = _finite(request["launch_deadline"], "launch_deadline")
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(request["absolute_deadline"], "absolute_deadline")
    return normalized, root, _material(_target(root, request["target_ref"]), request["report_ref"])


def _resources(value) -> dict:
    value = _closed(value, {"host_id", "gpu_uuids"}, label="resources")
    gpu_uuids = value["gpu_uuids"]
    if type(gpu_uuids) is not list or any(not isinstance(item, str) or not item for item in gpu_uuids):
        raise PyTorchError("invalid_pytorch_input", "resources.gpu_uuids must be a string list")
    if len(gpu_uuids) != len(set(gpu_uuids)):
        raise PyTorchError("invalid_pytorch_input", "resources.gpu_uuids must not contain duplicates")
    return {"host_id": _text(value["host_id"], "resources.host_id", 256), "gpu_uuids": sorted(gpu_uuids)}


def _resolve_collect(request: dict) -> dict:
    try:
        resolved = ADAPTER.resolve_profile_collection(
            artifact_root=request["artifact_root"], target_ref=request["target_ref"],
            baseline_ref=request["baseline_ref"], role=request["role"], case_id=request["case_id"],
            capability="pytorch_chrome_trace_v1", experiment_ref=request.get("experiment_ref"),
            correctness_ref=request.get("correctness_ref"),
        )
        host = resolved["target"]["environment"]["host"]
        expected = {"host_id": _text(host["host_id"], "Target host_id", 256), "gpu_uuids": sorted(host["gpu_uuids"])}
        if request["resources"] != expected:
            raise PyTorchError("resource_mismatch", "collect resources do not equal Target host and GPUs")
        torch_version = _torch_version(resolved["target"]["environment"]["runtime"]["frameworks"]["torch"], "Target torch version")
    except PyTorchError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise PyTorchError("collection_rejected", str(error)) from error
    if any(not isinstance(item, str) or not item for item in expected["gpu_uuids"]) or len(expected["gpu_uuids"]) != len(set(expected["gpu_uuids"])):
        raise PyTorchError("target_invalid", "Target GPU identities are invalid")
    return {**resolved, "resources": expected, "torch_version": torch_version}


def _validate_collect(value) -> tuple[dict, dict]:
    request = _closed(value, _COLLECT, _COLLECT_OPTIONAL, "collect input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "collect":
        raise PyTorchError("invalid_pytorch_input", "collect operation is unsupported")
    root = Path(os.path.abspath(os.path.expanduser(_text(request["artifact_root"], "artifact_root", 4096))))
    normalized = {**request, "artifact_root": str(root), "resources": _resources(request["resources"]), "role": _text(request["role"], "role", 32), "case_id": _text(request["case_id"], "case_id", 128)}
    for field in ("operation_timeout_seconds", "command_timeout_seconds", "resource_wait_timeout_seconds", "cleanup_timeout_seconds"):
        normalized[field] = _finite(request[field], field, True)
    normalized["launch_deadline"] = _finite(request["launch_deadline"], "launch_deadline")
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(request["absolute_deadline"], "absolute_deadline")
    return normalized, _resolve_collect(normalized)


def _frozen_collect_request(request: dict, resolved: dict) -> dict:
    frozen = {"operation": "collect", "target_ref": resolved["target_ref"], "baseline_ref": resolved["baseline_ref"], "role": resolved["role"], "case_id": resolved["case_id"], "variant": resolved["variant"], "driver": resolved["driver"], "resources": resolved["resources"], "torch_version": resolved["torch_version"], "tool_identity": _tool_identity("collect"), "operation_timeout_seconds": request["operation_timeout_seconds"], "command_timeout_seconds": request["command_timeout_seconds"], "resource_wait_timeout_seconds": request["resource_wait_timeout_seconds"], "cleanup_timeout_seconds": request["cleanup_timeout_seconds"], "launch_deadline": request["launch_deadline"]}
    if resolved["role"] == "candidate":
        frozen.update({"experiment_ref": resolved["experiment_ref"], "correctness_ref": resolved["correctness_ref"]})
    for field in ("absolute_deadline", "retry_of"):
        if field in request:
            frozen[field] = request[field]
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return frozen


def analyze(value, *, wait_for_result: bool) -> dict:
    request, root, material = _validate_analyze(value)
    frozen = {**request, "report_material": material, "tool_identity": _tool_identity("analyze")}
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return RUNTIME.submit(root, frozen, [sys.executable, str(Path(__file__).resolve()), "_worker"], wait_for_result)


def collect(value, *, wait_for_result: bool) -> dict:
    request, resolved = _validate_collect(value)
    return RUNTIME.submit(resolved["artifact_root"], _frozen_collect_request(request, resolved), [sys.executable, str(Path(__file__).resolve()), "_worker"], wait_for_result)


def _command_spec(argv: list[str], workspace: Path, gpu_uuids: list[str]) -> dict:
    return {"argv": argv, "cwd": str(workspace), "env": {}, "output_limit_bytes": 64 * 1024, "required_gpu_uuids": gpu_uuids}


def _revalidate_collect_worker(request: dict, root: Path) -> dict:
    resolved = _resolve_collect({**request, "artifact_root": str(root)})
    keys = ("target_ref", "baseline_ref", "role", "case_id", "variant", "driver", "resources", "torch_version")
    if request["role"] == "candidate":
        keys += ("experiment_ref", "correctness_ref")
    if {key: resolved[key] for key in keys} != {key: request[key] for key in keys}:
        raise PyTorchError("collection_changed", "frozen collection bindings changed")
    return resolved


def _collect_worker(request: dict, root: Path, invocation: Path, result: dict) -> None:
    acquired = RUNTIME.acquire_resources(request["resources"]["gpu_uuids"])
    if acquired.get("status") != "acquired":
        result.update({"execution_status": acquired.get("status", "failed"), "stop_reason": acquired.get("stop_reason", "resource_acquisition_failed"), "cleanup_status": acquired.get("cleanup_status", "unknown")})
        return
    resolved = _revalidate_collect_worker(request, root)
    workspace = invocation / "workspace"
    workspace.mkdir()
    variant = ADAPTER.materialize_variant(root, workspace, resolved["variant"], "variant")
    inputs = ADAPTER.materialize_target_inputs(root, workspace, resolved["target"])
    driver_output = workspace / "driver-output"
    driver_output.mkdir()
    driver_request = ADAPTER.build_driver_request(
        target_id=resolved["target"]["id"], execution_id=os.environ["CKO_INVOCATION_ID"], operation="profile_pytorch_collect",
        driver=resolved["driver"], variant=variant, test_suite=inputs["test_suite"], correctness=inputs["correctness"], objective=inputs["objective"],
        role=resolved["role"], mode="measure" if resolved["driver"]["execution_mode"] == "separate" else "combined", case={"id": resolved["case_id"]},
        sampling={"kind": "pytorch_chrome_trace_v1"}, output_path=driver_output / "result.json",
    )
    request_path = workspace / "driver-request.json"
    STORE.create_regular_bytes(request_path, json.dumps(driver_request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
    argv = ADAPTER.build_argv(resolved["driver"], request_path)
    receipt = RUNTIME.run_child(_command_spec(argv, workspace, request["resources"]["gpu_uuids"]))
    result["provenance"] = {"tool_identity": request["tool_identity"], "parser_version": PARSER_VERSION, "command_receipts": [{"argv": argv, "result": receipt}]}
    if receipt.get("status") != "completed":
        result.update({"execution_status": receipt.get("status", "failed"), "stop_reason": receipt.get("stop_reason", "driver_command_failed"), "cleanup_status": receipt.get("cleanup_status", "unknown"), "diagnostic": {"error": (receipt.get("stderr", "") + "\n" + receipt.get("stdout", ""))[:1024]}})
        return
    driver_object = STORE.freeze_path(root, driver_output, _DRIVER_OUTPUT_FREEZE_LIMITS)
    result["provenance"]["driver_output"] = driver_object
    frozen = STORE.materialize_object(root, driver_object, workspace / "frozen-driver-output")
    try:
        driver_result = ADAPTER.validate_driver_result(frozen / "result.json", driver_request)
    except ValueError as error:
        raise PyTorchError("driver_result_invalid", str(error)) from error
    if driver_result["environment"] != resolved["target"]["environment"]["runtime"]:
        raise PyTorchError("environment_changed", "driver runtime identity changed during collection")
    version = _torch_version(driver_result["environment"]["frameworks"].get("torch"), "driver torch version")
    if version != request["torch_version"]:
        raise PyTorchError("environment_changed", "driver torch version changed during collection")
    artifacts = driver_result["artifacts"]
    if len(artifacts) != 1 or artifacts[0].get("kind") != "pytorch_chrome_trace":
        raise PyTorchError("invalid_trace_artifact", "driver result must contain exactly one PyTorch Chrome trace")
    artifact = artifacts[0]
    trace = frozen / artifact["relative_path"]
    facts = parse_chrome_trace(_strict_json_bytes(STORE.read_regular_bytes(trace, maximum_bytes=_MAX_TRACE_BYTES), "Chrome trace"), version)
    result.update({"execution_status": "succeeded", "measurement_validity": "valid", "stop_reason": "completed", "cleanup_status": RUNTIME.current_cleanup_status(), "observations": facts["observations"], "unmodeled": facts["unmodeled"]})
    result["provenance"].update({"trace_artifact": artifact, "tool": {"name": "pytorch_profiler", "version": version}, "dialect": _DIALECT, "timestamp_unit": "us"})


def _worker_main() -> int:
    root, invocation = Path(os.environ["CKO_ARTIFACT_ROOT"]), Path(os.environ["CKO_INVOCATION_DIR"])
    request, started = _strict_json(invocation / "request.json"), time.monotonic()
    result = {"record_type": "profiler_result", "format_version": RESULT_VERSION, "operation": request["operation"], "target_ref": request["target_ref"], "execution_status": "invalid", "measurement_validity": "invalid", "stop_reason": None, "cleanup_status": "not_required", "observations": [], "unmodeled": [], "provenance": {}, "started_at_epoch": time.time()}
    if request["operation"] == "analyze":
        result["report_ref"] = request["report_ref"]
    else:
        result.update({"baseline_ref": request["baseline_ref"], "role": request["role"], "case_id": request["case_id"]})
        if request["role"] == "candidate":
            result.update({"experiment_ref": request["experiment_ref"], "correctness_ref": request["correctness_ref"]})
    try:
        if request.get("tool_identity") != _tool_identity(request["operation"]):
            raise PyTorchError("tool_identity_changed", "implementation changed before analysis")
        if request["operation"] == "collect":
            _collect_worker(request, root, invocation, result)
        else:
            material = _material(_target(root, request["target_ref"]), request["report_ref"])
            if material != request.get("report_material"):
                raise PyTorchError("report_changed", "frozen report material changed")
            report = STORE.materialize_object(root, material["object_ref"], invocation / "workspace" / "trace.json")
            facts = parse_chrome_trace(_strict_json_bytes(STORE.read_regular_bytes(report, maximum_bytes=_MAX_TRACE_BYTES), "Chrome trace"), material["tool_version"])
            result.update({"execution_status": "succeeded", "measurement_validity": "valid", "stop_reason": "completed", "observations": facts["observations"], "unmodeled": facts["unmodeled"], "provenance": {"report_ref": request["report_ref"], "report_object_ref": material["object_ref"], "tool": {"name": "pytorch_profiler", "version": material["tool_version"]}, "dialect": material["dialect"], "parser_version": PARSER_VERSION, "timestamp_unit": "us", "tool_identity": request["tool_identity"]}})
    except PyTorchError as error:
        result["stop_reason"] = error.code
        result["diagnostic"] = {"error": str(error)[:1024]}
        if request["operation"] == "collect":
            result["cleanup_status"] = RUNTIME.current_cleanup_status()
    except BaseException as error:
        result.update({"execution_status": "failed", "stop_reason": "worker_error", "diagnostic": {"error": str(error)[:1024]}})
        if request["operation"] == "collect":
            result["cleanup_status"] = RUNTIME.current_cleanup_status()
    result["finished_at_epoch"], result["elapsed_seconds"] = time.time(), time.monotonic() - started
    STORE.create_regular_json(invocation / "result.json", result)
    return 0


def _status_or_cancel(value, operation: str) -> dict:
    value = _closed(value, {"format_version", "operation", "artifact_root", "invocation_id"}, label=f"{operation} input")
    if value["format_version"] != INPUT_VERSION or value["operation"] != operation:
        raise PyTorchError("invalid_pytorch_input", f"{operation} input is unsupported")
    return RUNTIME.status(value["artifact_root"], value["invocation_id"]) if operation == "status" else RUNTIME.cancel(value["artifact_root"], value["invocation_id"])


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["_worker"]:
        return _worker_main()
    parser = argparse.ArgumentParser(description="Analyze one frozen V1.4 PyTorch Chrome trace.")
    parser.add_argument("operation", choices=("analyze", "collect", "status", "cancel"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(arguments)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise PyTorchError("invalid_pytorch_input", "CLI operation does not match request")
        if args.operation == "analyze":
            result = analyze(request, wait_for_result=args.wait)
        elif args.operation == "collect":
            result = collect(request, wait_for_result=args.wait)
        else:
            if args.wait:
                raise PyTorchError("invalid_pytorch_input", "--wait is only valid for analyze or collect")
            result = _status_or_cancel(request, args.operation)
    except (PyTorchError, OSError, ValueError, TimeoutError) as error:
        print(json.dumps({"status": "rejected", "error_code": getattr(error, "code", "pytorch_error"), "error": str(error)[:1024]}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
