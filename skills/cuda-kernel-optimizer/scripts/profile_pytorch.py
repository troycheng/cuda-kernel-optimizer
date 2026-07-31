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
_VERSION = re.compile(r"2\.13(?:\.\d+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INPUT = {
    "format_version", "operation", "artifact_root", "target_ref", "report_ref",
    "resources", "operation_timeout_seconds", "command_timeout_seconds",
    "resource_wait_timeout_seconds", "cleanup_timeout_seconds", "launch_deadline",
}
_OPTIONAL = {"absolute_deadline", "retry_of"}
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


def _load_sibling(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_pytorch_store")
RUNTIME = _load_sibling("_invocation_runtime.py", "cuda_optimizer_pytorch_runtime")


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
    if not isinstance(material["tool_version"], str) or _VERSION.fullmatch(material["tool_version"]) is None:
        raise PyTorchError("unsupported_tool_version", "PyTorch version is not supported")
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
    if _VERSION.fullmatch(tool_version) is None:
        raise PyTorchError("unsupported_tool_version", "PyTorch version is not supported")
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


def _tool_identity() -> dict:
    implementations = [{"name": name, "sha256": STORE.sha256_file(Path(__file__).with_name(name))} for name in ("profile_pytorch.py", "_invocation_runtime.py", "artifact_store.py")]
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


def analyze(value, *, wait_for_result: bool) -> dict:
    request, root, material = _validate_analyze(value)
    frozen = {**request, "report_material": material, "tool_identity": _tool_identity()}
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return RUNTIME.submit(root, frozen, [sys.executable, str(Path(__file__).resolve()), "_worker"], wait_for_result)


def _worker_main() -> int:
    root, invocation = Path(os.environ["CKO_ARTIFACT_ROOT"]), Path(os.environ["CKO_INVOCATION_DIR"])
    request, started = _strict_json(invocation / "request.json"), time.monotonic()
    result = {"record_type": "profiler_result", "format_version": RESULT_VERSION, "operation": "analyze", "target_ref": request["target_ref"], "report_ref": request["report_ref"], "execution_status": "invalid", "measurement_validity": "invalid", "stop_reason": None, "cleanup_status": "not_required", "observations": [], "unmodeled": [], "provenance": {}, "started_at_epoch": time.time()}
    try:
        if request.get("tool_identity") != _tool_identity():
            raise PyTorchError("tool_identity_changed", "implementation changed before analysis")
        material = _material(_target(root, request["target_ref"]), request["report_ref"])
        if material != request.get("report_material"):
            raise PyTorchError("report_changed", "frozen report material changed")
        report = STORE.materialize_object(root, material["object_ref"], invocation / "workspace" / "trace.json")
        facts = parse_chrome_trace(
            _strict_json_bytes(
                STORE.read_regular_bytes(
                    report,
                    maximum_bytes=_MAX_TRACE_BYTES,
                ),
                "Chrome trace",
            ),
            material["tool_version"],
        )
        result.update({"execution_status": "succeeded", "measurement_validity": "valid", "stop_reason": "completed", "observations": facts["observations"], "unmodeled": facts["unmodeled"], "provenance": {"report_ref": request["report_ref"], "report_object_ref": material["object_ref"], "tool": {"name": "pytorch_profiler", "version": material["tool_version"]}, "dialect": material["dialect"], "parser_version": PARSER_VERSION, "timestamp_unit": "us", "tool_identity": request["tool_identity"]}})
    except PyTorchError as error:
        result["stop_reason"] = error.code
        result["diagnostic"] = {"error": str(error)[:1024]}
    except BaseException as error:
        result.update({"execution_status": "failed", "stop_reason": "worker_error", "diagnostic": {"error": str(error)[:1024]}})
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
    parser.add_argument("operation", choices=("analyze", "status", "cancel"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(arguments)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise PyTorchError("invalid_pytorch_input", "CLI operation does not match request")
        if args.operation == "analyze":
            result = analyze(request, wait_for_result=args.wait)
        else:
            if args.wait:
                raise PyTorchError("invalid_pytorch_input", "--wait is only valid for analyze")
            result = _status_or_cancel(request, args.operation)
    except (PyTorchError, OSError, ValueError, TimeoutError) as error:
        print(json.dumps({"status": "rejected", "error_code": getattr(error, "code", "pytorch_error"), "error": str(error)[:1024]}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
