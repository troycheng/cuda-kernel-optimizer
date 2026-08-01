#!/usr/bin/env python3
"""Report bounded facts for one explicitly selected frozen compiler artifact."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/compiler-input-v1"
RESULT_VERSION = "cuda-kernel-optimizer/compiler-result-v1"
TOOL_VERSION = "cuda-kernel-optimizer/compiler-tool-v1"
STAGES = ("source", "ttir", "ttgir", "llvm_ir", "ptx", "sass", "binary")
TEXT_STAGES = frozenset(STAGES) - {"binary"}
DIALECT_STAGES = {
    "cuda-source-v1": "source",
    "triton-ttir-v1": "ttir",
    "triton-ttgir-v1": "ttgir",
    "llvm-ir-v1": "llvm_ir",
    "ptx-v1": "ptx",
    "sass-text-v1": "sass",
    "cuda-binary-v1": "binary",
}
STRUCTURAL_MARKERS = {
    "ttir": ((b"module", "module"), (b"tt.func", "tt.func")),
    "ttgir": ((b"module", "module"), (b"ttg.", "ttg.")),
    "llvm_ir": ((b"define", "define"), (b"{", "function_body")),
    "ptx": ((b".version", ".version"), (b".target", ".target")),
}
_ANALYZE_FIELDS = {
    "format_version", "operation", "artifact_root", "target_ref", "artifact_ref",
    "resources", "operation_timeout_seconds", "command_timeout_seconds",
    "resource_wait_timeout_seconds", "cleanup_timeout_seconds", "launch_deadline",
}
_OPTIONAL_FIELDS = {"absolute_deadline", "retry_of"}
_STATUS_FIELDS = {"format_version", "operation", "artifact_root", "invocation_id"}
_BINDING_KEYS = (
    "target_ref", "requested_stage", "source", "material_ref", "invocation_ref",
    "receipt_index", "object_ref", "artifact", "role", "variant", "experiment_ref",
    "mechanism_key", "environment",
)
_SOURCE_KEYS = (
    "source", "material_ref", "invocation_ref", "receipt_index", "role", "variant",
    "experiment_ref", "mechanism_key",
)
_CHUNK_BYTES, _MAX_RECORD_BYTES = 1024 * 1024, 16 * 1024 * 1024


def _load_sibling(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_compiler_store")
RUNTIME = _load_sibling("_invocation_runtime.py", "cuda_optimizer_compiler_runtime")
ADAPTER = _load_sibling("workload_adapter.py", "cuda_optimizer_compiler_adapter")


class CompilerError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _closed(value, required, optional=(), label="value") -> dict:
    if type(value) is not dict:
        raise CompilerError("invalid_compiler_input", f"{label} must be an object")
    missing, unknown = set(required) - set(value), set(value) - set(required) - set(optional)
    if missing or unknown:
        raise CompilerError(
            "invalid_compiler_input",
            f"{label} has missing={sorted(missing)} unknown={sorted(unknown)}",
        )
    return value


def _text(value, label: str, maximum=4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CompilerError("invalid_compiler_input", f"{label} must be a non-empty bounded string")
    return value


def _strict_json(path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CompilerError("invalid_compiler_input", "JSON contains a duplicate key")
            result[key] = value
        return result
    try:
        payload = STORE.read_regular_bytes(path, maximum_bytes=_MAX_RECORD_BYTES)
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CompilerError("invalid_compiler_input", "JSON contains a non-finite number")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, CompilerError):
            raise
        raise CompilerError("invalid_compiler_input", "request is invalid JSON") from error
    if type(value) is not dict:
        raise CompilerError("invalid_compiler_input", "request must be an object")
    return value


def _artifact_ref(value) -> dict:
    value = _closed(value, {"source", "stage"}, {
        "material_ref", "invocation_ref", "receipt_index", "relative_path"
    }, "artifact_ref")
    stage = _text(value["stage"], "artifact_ref.stage", 64)
    if stage not in STAGES:
        raise CompilerError("unsupported_stage", "artifact_ref.stage is unsupported")
    if value["source"] == "target_material":
        expected = {"source", "stage", "material_ref"}
    elif value["source"] == "invocation_driver_artifact":
        expected = {"source", "stage", "invocation_ref", "receipt_index", "relative_path"}
    else:
        raise CompilerError("unsupported_artifact_source", "artifact_ref.source is unsupported")
    if set(value) != expected:
        raise CompilerError("invalid_compiler_input", "artifact_ref is not one closed source variant")
    return dict(value)


def _subset(resolved: dict, keys: tuple) -> dict:
    return {key: resolved[key] for key in keys if key in resolved}


def _validate_resolution(resolved: dict, stage: str) -> dict:
    artifact = resolved.get("artifact")
    if type(artifact) is not dict or resolved.get("requested_stage") != stage:
        raise CompilerError("artifact_binding_invalid", "resolver did not bind the requested stage")
    if resolved.get("source") == "target_material":
        dialect = artifact.get("dialect")
        if DIALECT_STAGES.get(dialect) != stage:
            raise CompilerError("stage_dialect_mismatch", "Target material dialect does not match stage")
        member = artifact.get("path")
    else:
        if artifact.get("kind") != stage:
            raise CompilerError("stage_kind_mismatch", "driver artifact kind does not match stage")
        member = artifact.get("relative_path")
    if not isinstance(member, str) or not member:
        raise CompilerError("artifact_binding_invalid", "selected object member is unavailable")
    size = artifact.get("size_bytes")
    digest = artifact.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise CompilerError("empty_artifact", "selected compiler artifact is empty")
    if not isinstance(digest, str) or len(digest) != 64:
        raise CompilerError("artifact_binding_invalid", "selected artifact digest is invalid")
    return {"member": member, "size_bytes": size, "sha256": digest}


def _resolve(root: Path, target_ref: dict, artifact_ref: dict) -> tuple[dict, dict]:
    try:
        resolved = ADAPTER.resolve_analysis_artifact(
            artifact_root=root, target_ref=target_ref, artifact_ref=artifact_ref
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CompilerError("artifact_resolution_rejected", str(error)) from error
    return resolved, _validate_resolution(resolved, artifact_ref["stage"])


def _tool_identity() -> dict:
    implementations = [
        {"name": name, "sha256": STORE.sha256_file(Path(__file__).with_name(name))}
        for name in (
            "compiler_evidence.py", "_invocation_runtime.py", "artifact_store.py",
            "workload_adapter.py",
        )
    ]
    identity = {
        "version": TOOL_VERSION,
        "result_contract": RESULT_VERSION,
        "implementations": implementations,
    }
    identity["digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity


def _validate_analyze(value) -> tuple[dict, Path, dict]:
    request = _closed(value, _ANALYZE_FIELDS, _OPTIONAL_FIELDS, "analyze input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != "analyze":
        raise CompilerError("invalid_compiler_input", "analyze operation is unsupported")
    root = Path(os.path.abspath(os.path.expanduser(_text(request["artifact_root"], "artifact_root"))))
    if not root.is_dir():
        raise CompilerError("target_not_found", "artifact_root is unavailable")
    artifact_ref = _artifact_ref(request["artifact_ref"])
    resources = _closed(request["resources"], {"host_id", "gpu_uuids"}, label="resources")
    if resources["gpu_uuids"] != []:
        raise CompilerError("invalid_compiler_input", "read-only compiler analysis must not request GPUs")
    normalized = dict(request)
    normalized.update({
        "artifact_root": str(root), "artifact_ref": artifact_ref,
        "resources": {"host_id": _text(resources["host_id"], "resources.host_id", 256), "gpu_uuids": []},
    })
    resolved, _selected = _resolve(root, request["target_ref"], artifact_ref)
    normalized["target_ref"] = resolved["target_ref"]
    return normalized, root, _subset(resolved, _BINDING_KEYS)


def analyze(value, *, wait_for_result: bool) -> dict:
    request, root, binding = _validate_analyze(value)
    frozen = {**request, "artifact_binding": binding, "tool_identity": _tool_identity()}
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return RUNTIME.submit(
        root, frozen, [sys.executable, str(Path(__file__).resolve()), "_worker"],
        wait_for_result,
    )


def _facts(path: Path, stage: str, selected: dict) -> dict:
    digest = hashlib.sha256()
    total = newlines = 0
    last = b""
    decoder = codecs.getincrementaldecoder("utf-8")("strict") if stage in TEXT_STAGES else None
    required = list(STRUCTURAL_MARKERS.get(stage, ()))
    found = {name: False for _marker, name in required}
    ptx_entry = False
    tail = b""
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                last = chunk[-1:]
                if decoder is not None:
                    if b"\x00" in chunk:
                        raise CompilerError("invalid_text_artifact", "text artifact contains NUL")
                    decoder.decode(chunk, final=False)
                    newlines += chunk.count(b"\n")
                    searchable = tail + chunk
                    for marker, name in required:
                        if marker in searchable:
                            found[name] = True
                    ptx_entry = ptx_entry or stage == "ptx" and (
                        b".entry" in searchable or b".func" in searchable
                    )
                    tail = searchable[-32:]
        if decoder is not None:
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as error:
        raise CompilerError("invalid_text_artifact", "text artifact is not UTF-8") from error
    if total != selected["size_bytes"] or digest.hexdigest() != selected["sha256"]:
        raise CompilerError("artifact_changed", "materialized artifact does not match frozen metadata")
    missing = [name for name, present in found.items() if not present]
    if stage == "ptx" and not ptx_entry:
        missing.append(".entry_or_func")
    if missing:
        raise CompilerError("unrecognized_dialect", f"missing structural marker: {missing[0]}")
    observation = {
        "kind": "compiler_artifact_facts", "declared_stage": stage,
        "content_sha256": digest.hexdigest(), "size_bytes": total,
    }
    if decoder is not None:
        markers = [name for _marker, name in required]
        if stage == "ptx":
            markers.append(".entry_or_func")
        observation["text"] = {
            "encoding": "utf-8", "line_count": newlines + (0 if last == b"\n" else 1),
            "ends_with_newline": last == b"\n", "structural_markers": markers,
        }
    return observation


def _worker_main() -> int:
    root = Path(os.environ["CKO_ARTIFACT_ROOT"])
    invocation = Path(os.environ["CKO_INVOCATION_DIR"])
    request = _strict_json(invocation / "request.json")
    started = time.monotonic()
    result = {
        "record_type": "compiler_result", "format_version": RESULT_VERSION,
        "operation": "analyze", "target_ref": request["target_ref"],
        "artifact_ref": request["artifact_ref"], "execution_status": "invalid",
        "evidence_validity": "invalid", "stop_reason": None,
        "cleanup_status": "not_required", "observations": [],
        "source_binding": _subset(request["artifact_binding"], _SOURCE_KEYS),
        "environment_binding": request["artifact_binding"]["environment"],
        "provenance": {"tool_identity": request["tool_identity"]},
        "started_at_epoch": time.time(),
    }
    try:
        if request["tool_identity"] != _tool_identity():
            raise CompilerError("tool_identity_changed", "implementation changed before analysis")
        resolved, selected = _resolve(root, request["target_ref"], request["artifact_ref"])
        if _subset(resolved, _BINDING_KEYS) != request["artifact_binding"]:
            raise CompilerError("artifact_binding_changed", "frozen artifact bindings changed")
        workspace = invocation / "workspace"
        STORE.create_regular_directory(workspace)
        STORE.materialize_object_member(
            root, resolved["object_ref"], selected["member"], workspace / "selected-artifact"
        )
        observation = _facts(workspace / "selected-artifact", request["artifact_ref"]["stage"], selected)
        if request["tool_identity"] != _tool_identity():
            raise CompilerError("tool_identity_changed", "implementation changed during analysis")
        result.update({
            "execution_status": "succeeded", "evidence_validity": "valid",
            "stop_reason": "completed", "observations": [observation],
        })
        result["provenance"].update({
            "object_ref": resolved["object_ref"], "selected_member": selected["member"],
        })
    except CompilerError as error:
        result["stop_reason"] = error.code
        result["diagnostic"] = {"error": str(error)[:1024]}
    except (KeyError, OSError, TypeError, ValueError) as error:
        result["stop_reason"] = "artifact_analysis_invalid"
        result["diagnostic"] = {"error": str(error)[:1024]}
    except BaseException as error:
        result.update({
            "execution_status": "failed", "stop_reason": "worker_error",
            "diagnostic": {"error": str(error)[:1024]},
        })
    result["finished_at_epoch"], result["elapsed_seconds"] = time.time(), time.monotonic() - started
    STORE.create_regular_json(invocation / "result.json", result)
    return 0


def _status_or_cancel(value, operation: str) -> dict:
    request = _closed(value, _STATUS_FIELDS, label=f"{operation} input")
    if request["format_version"] != INPUT_VERSION or request["operation"] != operation:
        raise CompilerError("invalid_compiler_input", f"{operation} input is unsupported")
    return (
        RUNTIME.status(request["artifact_root"], request["invocation_id"])
        if operation == "status"
        else RUNTIME.cancel(request["artifact_root"], request["invocation_id"])
    )


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["_worker"]:
        return _worker_main()
    parser = argparse.ArgumentParser(description="Analyze one frozen compiler artifact.")
    parser.add_argument("operation", choices=("analyze", "status", "cancel"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(arguments)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise CompilerError("invalid_compiler_input", "CLI operation does not match request")
        if args.operation == "analyze":
            result = analyze(request, wait_for_result=args.wait)
        else:
            if args.wait:
                raise CompilerError("invalid_compiler_input", "--wait is only valid for analyze")
            result = _status_or_cancel(request, args.operation)
    except (CompilerError, OSError, TimeoutError, ValueError) as error:
        print(json.dumps({
            "status": "rejected", "error_code": getattr(error, "code", "compiler_error"),
            "error": str(error)[:1024],
        }, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
