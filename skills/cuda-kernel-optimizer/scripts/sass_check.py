#!/usr/bin/env python3
"""Report bounded SASS facts for one explicitly selected frozen CUDA binary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import time
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/sass-input-v1"
RESULT_VERSION = "cuda-kernel-optimizer/sass-result-v1"
TOOL_VERSION = "cuda-kernel-optimizer/sass-tool-v1"
PARSER_VERSION = "cuobjdump-sass-v1"
SIGNATURES = Path(__file__).resolve().parent.parent / "references" / "sass_signatures.json"
_ANALYZE_FIELDS = {
    "format_version", "operation", "artifact_root", "target_ref", "artifact_ref",
    "resources", "operation_timeout_seconds", "command_timeout_seconds",
    "resource_wait_timeout_seconds", "cleanup_timeout_seconds", "launch_deadline",
}
_OPTIONAL_FIELDS = {"absolute_deadline", "retry_of"}
_STATUS_FIELDS = {"format_version", "operation", "artifact_root", "invocation_id"}
_BINDING_KEYS = (
    "target_ref", "source", "material_ref", "invocation_ref", "receipt_index",
    "object_ref", "artifact", "role", "variant", "experiment_ref", "mechanism_key",
    "environment",
)
_SOURCE_KEYS = (
    "source", "material_ref", "invocation_ref", "receipt_index", "role", "variant",
    "experiment_ref", "mechanism_key",
)
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_MAX_CAPTURE_BYTES = 512 * 1024 * 1024
_MAX_SASS_LINE_BYTES = 64 * 1024
_MAX_ARCHITECTURES = 32
_MAX_OPCODES = 512
_SAFE_SIGNATURE_RE = re.compile(r"(?:[A-Za-z0-9_]+|\\\.|\.\*)+\Z")
_VERSION_RE = re.compile(
    r"cuobjdump: NVIDIA \(R\) fat binary listing tool\n"
    r"Copyright \(c\) 2005-\d{4} NVIDIA Corporation\n"
    r"Built on [^\n]{1,128}\n"
    r"Cuda compilation tools, release (\d+\.\d+), V(\d+\.\d+\.\d+)\n"
    r"Build cuda_[A-Za-z0-9_./-]{1,128}\n?\Z"
)
_ARCH_RE = re.compile(r"arch\s*=\s*(sm_[0-9]+)\Z")
_CODE_ARCH_RE = re.compile(r"code for (sm_[0-9]+)\Z")
_FUNCTION_RE = re.compile(r"Function\s*:\s*\S.*\Z")
_INSTRUCTION_RE = re.compile(
    r"\s*/\*[0-9a-fA-F]+\*/\s+(?:@[!A-Za-z0-9_.]+\s+)?([A-Z][A-Z0-9_.]*)\b"
    r".*;\s*(?:/\*\s*0x[0-9a-fA-F ]+\s*\*/)?\s*\Z"
)
_HEX_RE = re.compile(r"\s*/\*\s*0x[0-9a-fA-F ]+\s*\*/\s*\Z")
_METADATA_RE = re.compile(
    r"(?:=+|Fatbin elf code:|(?:code version|host|compile_size|identifier|producer)\s*=.*|"
    r"\.target\s+sm_[0-9]+|\.headerflags.*|Section:.*)\Z"
)


def _load_sibling(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_sass_store")
RUNTIME = _load_sibling("_invocation_runtime.py", "cuda_optimizer_sass_runtime")
ADAPTER = _load_sibling("workload_adapter.py", "cuda_optimizer_sass_adapter")


class SassError(ValueError):
    def __init__(self, code: str, message: str, child_fact=None) -> None:
        super().__init__(message)
        self.code = code
        self.child_fact = child_fact


def _closed(value, required, optional=(), label="value") -> dict:
    if type(value) is not dict:
        raise SassError("invalid_sass_input", f"{label} must be an object")
    missing, unknown = set(required) - set(value), set(value) - set(required) - set(optional)
    if missing or unknown:
        raise SassError(
            "invalid_sass_input",
            f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}",
        )
    return value


def _text(value, label: str, maximum=4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise SassError("invalid_sass_input", f"{label} must be a non-empty bounded string")
    return value


def _strict_json(path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SassError("invalid_sass_input", "JSON contains a duplicate key")
            result[key] = value
        return result
    try:
        payload = STORE.read_regular_bytes(path, maximum_bytes=_MAX_RECORD_BYTES)
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SassError("invalid_sass_input", "JSON contains a non-finite number")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, SassError):
            raise
        raise SassError("invalid_sass_input", "request is invalid JSON") from error
    if type(value) is not dict:
        raise SassError("invalid_sass_input", "request must be an object")
    return value


def _artifact_ref(value) -> dict:
    value = _closed(value, {"source"}, {
        "material_ref", "invocation_ref", "receipt_index", "relative_path"
    }, "artifact_ref")
    if value["source"] == "target_material":
        expected = {"source", "material_ref"}
    elif value["source"] == "invocation_driver_artifact":
        expected = {"source", "invocation_ref", "receipt_index", "relative_path"}
    else:
        raise SassError("unsupported_artifact_source", "artifact_ref.source is unsupported")
    if set(value) != expected:
        raise SassError("invalid_sass_input", "artifact_ref is not one closed source variant")
    return dict(value)


def _subset(value: dict, keys: tuple) -> dict:
    return {key: value[key] for key in keys if key in value}


def _resolve(root: Path, target_ref: dict, artifact_ref: dict) -> tuple[dict, dict]:
    try:
        resolved = ADAPTER.resolve_analysis_artifact(
            artifact_root=root, target_ref=target_ref, artifact_ref=artifact_ref
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SassError("artifact_resolution_rejected", str(error)) from error
    artifact = resolved.get("artifact")
    if type(artifact) is not dict:
        raise SassError("artifact_binding_invalid", "resolver did not bind one artifact")
    if resolved.get("source") == "target_material":
        if artifact.get("dialect") != "cuda-binary-v1":
            raise SassError("binary_dialect_mismatch", "Target material is not a CUDA binary")
        member = artifact.get("path")
    else:
        if artifact.get("kind") != "binary":
            raise SassError("binary_kind_mismatch", "driver artifact is not a binary")
        member = artifact.get("relative_path")
    size, digest = artifact.get("size_bytes"), artifact.get("sha256")
    if not isinstance(member, str) or not member or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SassError("empty_binary", "selected CUDA binary is empty or unavailable")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SassError("artifact_binding_invalid", "selected binary digest is invalid")
    return resolved, {"member": member, "size_bytes": size, "sha256": digest}


def _cuobjdump(resolved: dict) -> dict:
    environment = resolved.get("environment")
    host = environment.get("host") if type(environment) is dict else None
    tools = host.get("tools") if type(host) is dict else None
    tool = tools.get("cuobjdump") if type(tools) is dict else None
    if type(tool) is not dict or set(tool) != {"path", "sha256"}:
        raise SassError("cuobjdump_binding_missing", "Target does not bind cuobjdump")
    path, digest = tool["path"], tool["sha256"]
    if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
        raise SassError("cuobjdump_binding_invalid", "Target cuobjdump binding is invalid")
    try:
        current = STORE.sha256_file(path)
    except (OSError, ValueError) as error:
        raise SassError("cuobjdump_identity_changed", "Target cuobjdump is unavailable") from error
    if current != digest:
        raise SassError("cuobjdump_identity_changed", "Target cuobjdump digest changed")
    return {"path": path, "sha256": digest}


def _tool_identity() -> dict:
    files = ("sass_check.py", "_invocation_runtime.py", "artifact_store.py", "workload_adapter.py")
    identity = {
        "version": TOOL_VERSION, "parser_version": PARSER_VERSION,
        "result_contract": RESULT_VERSION,
        "implementations": [
            {"name": name, "sha256": STORE.sha256_file(Path(__file__).with_name(name))}
            for name in files
        ],
        "signatures_sha256": STORE.sha256_file(SIGNATURES),
    }
    identity["digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity


def _validate_analyze(value) -> tuple[dict, Path, dict]:
    request = _closed(value, _ANALYZE_FIELDS, _OPTIONAL_FIELDS, "analyze input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "analyze":
        raise SassError("invalid_sass_input", "analyze operation is unsupported")
    root = Path(os.path.abspath(os.path.expanduser(_text(request["artifact_root"], "artifact_root"))))
    if not root.is_dir():
        raise SassError("target_not_found", "artifact_root is unavailable")
    artifact_ref = _artifact_ref(request["artifact_ref"])
    resources = _closed(request["resources"], {"host_id", "gpu_uuids"}, label="resources")
    if resources["gpu_uuids"] != []:
        raise SassError("invalid_sass_input", "read-only SASS analysis must not request GPUs")
    resolved, _selected = _resolve(root, request["target_ref"], artifact_ref)
    host = resolved["environment"].get("host")
    host_id = _text(resources["host_id"], "resources.host_id", 256)
    if type(host) is not dict or host.get("host_id") != host_id:
        raise SassError("host_binding_mismatch", "resources.host_id does not match Target host")
    _cuobjdump(resolved)
    normalized = dict(request)
    normalized.update({
        "artifact_root": str(root), "artifact_ref": artifact_ref,
        "target_ref": resolved["target_ref"], "resources": {"host_id": host_id, "gpu_uuids": []},
    })
    return normalized, root, _subset(resolved, _BINDING_KEYS)


def analyze(value, *, wait_for_result: bool) -> dict:
    request, root, binding = _validate_analyze(value)
    frozen = {**request, "artifact_binding": binding, "tool_identity": _tool_identity()}
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return RUNTIME.submit(
        root, frozen, [sys.executable, str(Path(__file__).resolve()), "_worker"], wait_for_result
    )


def _version(stdout: str) -> dict:
    match = _VERSION_RE.fullmatch(stdout)
    if match is None or not match.group(2).startswith(match.group(1) + "."):
        raise SassError("unsupported_cuobjdump_version", "cuobjdump version output is unsupported")
    return {"release": match.group(1), "version": match.group(2)}


def _sass_facts(path: Path) -> dict:
    digest, size, functions, instructions = hashlib.sha256(), 0, 0, 0
    architectures, opcodes = set(), {}
    saw_header = False
    try:
        with path.open("rb") as stream:
            while True:
                raw = stream.readline(_MAX_SASS_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > _MAX_SASS_LINE_BYTES:
                    raise SassError("sass_line_too_long", "SASS capture line exceeds limit")
                size += len(raw); digest.update(raw)
                if b"\x00" in raw:
                    raise SassError("invalid_sass_text", "SASS capture contains NUL")
                line = raw.decode("utf-8", errors="strict").rstrip("\r\n")
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == "Fatbin elf code:":
                    saw_header = True; continue
                architecture = _ARCH_RE.fullmatch(stripped)
                if architecture:
                    if architecture.group(1) not in architectures and len(architectures) >= _MAX_ARCHITECTURES:
                        raise SassError("sass_fact_limit_exceeded", "SASS architecture limit exceeded")
                    architectures.add(architecture.group(1)); continue
                code_architecture = _CODE_ARCH_RE.fullmatch(stripped)
                if code_architecture:
                    if code_architecture.group(1) not in architectures and len(architectures) >= _MAX_ARCHITECTURES:
                        raise SassError("sass_fact_limit_exceeded", "SASS architecture limit exceeded")
                    architectures.add(code_architecture.group(1)); continue
                if _FUNCTION_RE.fullmatch(stripped):
                    functions += 1; continue
                instruction = _INSTRUCTION_RE.fullmatch(line)
                if instruction:
                    opcode = instruction.group(1).split(".", 1)[0]
                    if opcode not in opcodes and len(opcodes) >= _MAX_OPCODES:
                        raise SassError("sass_fact_limit_exceeded", "SASS opcode limit exceeded")
                    opcodes[opcode] = opcodes.get(opcode, 0) + 1
                    instructions += 1; continue
                if _HEX_RE.fullmatch(line) or _METADATA_RE.fullmatch(stripped):
                    continue
                raise SassError("unrecognized_sass_dialect", "cuobjdump SASS line is unsupported")
    except UnicodeDecodeError as error:
        raise SassError("invalid_sass_text", "SASS capture is not UTF-8") from error
    if not saw_header or not architectures or functions == 0 or instructions == 0:
        raise SassError("unrecognized_sass_dialect", "cuobjdump SASS structure is incomplete")
    return {
        "kind": "sass_facts", "content_sha256": digest.hexdigest(), "size_bytes": size,
        "architectures": sorted(architectures), "function_count": functions,
        "instruction_count": instructions, "opcode_counts": dict(sorted(opcodes.items())),
    }


def _signature_facts(path: Path, mechanism_key: str) -> dict:
    try:
        signatures = _strict_json(SIGNATURES)
    except (SassError, OSError, ValueError) as error:
        raise SassError("signature_catalog_invalid", "SASS signature catalog is invalid") from error
    if set(signatures) != {"$note", "$version", "methods"} or signatures["$version"] != "4.0" or type(signatures["methods"]) is not dict:
        raise SassError("signature_catalog_invalid", "SASS signature catalog dialect is unsupported")
    method = signatures["methods"].get(mechanism_key)
    base = {"kind": "sass_signature_facts", "mechanism_key": mechanism_key}
    if method is None:
        return {**base, "applicability": "not_applicable", "reason": "signature_unavailable", "patterns_found": [], "patterns_missing": []}
    if type(method) is not dict or set(method) != {"sass_patterns", "require_any", "note"}:
        raise SassError("signature_catalog_invalid", "selected SASS signature is malformed")
    patterns = method["sass_patterns"]
    if (
        type(patterns) is not list or len(patterns) > 128
        or any(not isinstance(item, str) or not item or len(item) > 256 or _SAFE_SIGNATURE_RE.fullmatch(item) is None for item in patterns)
        or type(method["require_any"]) is not bool
        or not isinstance(method["note"], str) or not method["note"] or len(method["note"]) > 2048
    ):
        raise SassError("signature_catalog_invalid", "selected SASS signature is malformed")
    if not patterns:
        return {**base, "applicability": "not_applicable", "reason": "no_sass_patterns", "patterns_found": [], "patterns_missing": []}
    try:
        compiled = [(pattern, re.compile(pattern, re.IGNORECASE)) for pattern in patterns]
    except re.error as error:
        raise SassError("signature_catalog_invalid", "selected SASS regex is invalid") from error
    found = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            for pattern, expression in compiled:
                if pattern not in found and expression.search(line):
                    found.add(pattern)
    return {
        **base, "applicability": "applicable", "require_any": method["require_any"],
        "patterns_found": [item for item in patterns if item in found],
        "patterns_missing": [item for item in patterns if item not in found],
    }


def _child(command: dict, code: str) -> dict:
    result = RUNTIME.run_child(command)
    if result.get("status") != "completed" or result.get("cleanup_status") != "confirmed":
        fact = {
            "status": result.get("status"), "stop_reason": result.get("stop_reason"),
            "cleanup_status": result.get("cleanup_status"), "returncode": result.get("returncode"),
        }
        raise SassError(code, f"guarded command failed: {result.get('stop_reason')}", fact)
    return result


def _worker_main() -> int:
    root, invocation = Path(os.environ["CKO_ARTIFACT_ROOT"]), Path(os.environ["CKO_INVOCATION_DIR"])
    request, started = _strict_json(invocation / "request.json"), time.monotonic()
    result = {
        "record_type": "sass_result", "format_version": RESULT_VERSION, "operation": "analyze",
        "target_ref": request["target_ref"], "artifact_ref": request["artifact_ref"],
        "execution_status": "invalid", "evidence_validity": "invalid", "stop_reason": None,
        "cleanup_status": "not_required", "observations": [],
        "source_binding": _subset(request["artifact_binding"], _SOURCE_KEYS),
        "environment_binding": request["artifact_binding"]["environment"],
        "provenance": {"tool_identity": request["tool_identity"]}, "started_at_epoch": time.time(),
    }
    try:
        if request["tool_identity"] != _tool_identity():
            raise SassError("tool_identity_changed", "implementation changed before analysis")
        resolved, selected = _resolve(root, request["target_ref"], request["artifact_ref"])
        if _subset(resolved, _BINDING_KEYS) != request["artifact_binding"]:
            raise SassError("artifact_binding_changed", "frozen artifact bindings changed")
        tool = _cuobjdump(resolved)
        workspace = invocation / "workspace"; STORE.create_regular_directory(workspace)
        binary = workspace / "selected-binary"
        STORE.materialize_object_member(root, resolved["object_ref"], selected["member"], binary)
        if STORE.sha256_file(binary) != selected["sha256"] or binary.stat().st_size != selected["size_bytes"]:
            raise SassError("binary_changed", "materialized binary does not match frozen metadata")
        common = {"cwd": str(workspace), "env": {}, "required_gpu_uuids": []}
        version_result = _child({
            **common, "argv": [tool["path"], "--version"], "output_limit_bytes": 4096,
        }, "cuobjdump_version_failed")
        result["cleanup_status"] = version_result["cleanup_status"]
        version = _version(version_result["stdout"])
        result["provenance"]["cuobjdump"] = {**tool, **version}
        if _cuobjdump(resolved) != tool:
            raise SassError("cuobjdump_identity_changed", "cuobjdump changed during analysis")
        dump = _child({
            **common, "argv": [tool["path"], "--dump-sass", str(binary)],
            "output_limit_bytes": 4096,
            "stdout_capture": {"relative_path": "artifacts/sass.txt", "max_bytes": _MAX_CAPTURE_BYTES},
        }, "cuobjdump_dump_failed")
        result["cleanup_status"] = dump["cleanup_status"]
        if _cuobjdump(resolved) != tool:
            raise SassError("cuobjdump_identity_changed", "cuobjdump changed after SASS dump")
        capture = dump.get("stdout_capture")
        if (
            type(capture) is not dict
            or set(capture) != {"relative_path", "size_bytes", "sha256"}
            or capture.get("relative_path") != "artifacts/sass.txt"
            or isinstance(capture.get("size_bytes"), bool)
            or not isinstance(capture.get("size_bytes"), int)
            or not 1 <= capture["size_bytes"] <= _MAX_CAPTURE_BYTES
            or re.fullmatch(r"[0-9a-f]{64}", capture.get("sha256", "")) is None
        ):
            raise SassError("capture_metadata_invalid", "guardian omitted closed capture metadata")
        capture_path = invocation / capture["relative_path"]
        metadata = os.lstat(capture_path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != capture["size_bytes"] or STORE.sha256_file(capture_path) != capture["sha256"]:
            raise SassError("capture_identity_changed", "capture does not match guardian metadata")
        capture_ref = STORE.freeze_path(root, capture_path, {
            "max_files": 1, "max_total_bytes": _MAX_CAPTURE_BYTES, "max_wall_seconds": 30.0,
        })
        manifest = STORE._load_object_manifest(root, capture_ref, verify_payload=True)
        files = [item for item in manifest["entries"] if item["kind"] == "file"]
        if len(files) != 1 or files[0]["size_bytes"] != capture["size_bytes"] or files[0]["sha256"] != capture["sha256"]:
            raise SassError("capture_freeze_invalid", "frozen capture does not match guardian metadata")
        result["provenance"]["capture"] = {**capture, "object_ref": capture_ref}
        frozen = workspace / "frozen-sass"
        STORE.materialize_object_member(root, capture_ref, files[0]["path"], frozen)
        observations = [_sass_facts(frozen)]
        if resolved.get("role") == "candidate":
            observations.append(_signature_facts(frozen, resolved["mechanism_key"]))
        if request["tool_identity"] != _tool_identity():
            raise SassError("tool_identity_changed", "implementation changed during analysis")
        result.update({
            "execution_status": "succeeded", "evidence_validity": "valid",
            "stop_reason": "completed", "cleanup_status": RUNTIME.current_cleanup_status(),
            "observations": observations,
        })
        result["provenance"].update({
            "object_ref": resolved["object_ref"], "selected_member": selected["member"],
        })
    except SassError as error:
        result["stop_reason"] = error.code; result["diagnostic"] = {"error": str(error)[:1024]}
        if error.child_fact is not None:
            result["command_failure"] = error.child_fact
            result["cleanup_status"] = error.child_fact.get("cleanup_status", "unknown")
    except (KeyError, OSError, TypeError, ValueError) as error:
        result["stop_reason"] = "sass_analysis_invalid"; result["diagnostic"] = {"error": str(error)[:1024]}
    except BaseException as error:
        result.update({"execution_status": "failed", "stop_reason": "worker_error", "diagnostic": {"error": str(error)[:1024]}})
    result["finished_at_epoch"], result["elapsed_seconds"] = time.time(), time.monotonic() - started
    STORE.create_regular_json(invocation / "result.json", result)
    return 0


def _status_or_cancel(value, operation: str) -> dict:
    request = _closed(value, _STATUS_FIELDS, label=f"{operation} input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != operation:
        raise SassError("invalid_sass_input", f"{operation} input is unsupported")
    return RUNTIME.status(request["artifact_root"], request["invocation_id"]) if operation == "status" else RUNTIME.cancel(request["artifact_root"], request["invocation_id"])


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["_worker"]:
        return _worker_main()
    parser = argparse.ArgumentParser(description="Analyze one frozen CUDA binary's SASS.")
    parser.add_argument("operation", choices=("analyze", "status", "cancel"))
    parser.add_argument("--request", required=True); parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(arguments)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise SassError("invalid_sass_input", "CLI operation does not match request")
        if args.operation != "analyze" and args.wait:
            raise SassError("invalid_sass_input", "--wait is only valid for analyze")
        result = analyze(request, wait_for_result=args.wait) if args.operation == "analyze" else _status_or_cancel(request, args.operation)
    except (SassError, OSError, TimeoutError, ValueError) as error:
        print(json.dumps({"status": "rejected", "error_code": getattr(error, "code", "sass_error"), "error": str(error)[:1024]}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
