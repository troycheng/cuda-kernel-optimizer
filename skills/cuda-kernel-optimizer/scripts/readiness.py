#!/usr/bin/env python3
"""Validate one optimization target and publish it atomically."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import shutil
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/readiness-input-v2"
TARGET_VERSION = "cuda-kernel-optimizer/target-v2"

_INPUT_FIELDS = {
    "format_version",
    "operation",
    "artifact_root",
    "target_mode",
    "claim_layer",
    "test_suite",
    "correctness",
    "original",
    "objective",
    "driver",
    "environment_requirements",
    "validity_requirements",
    "smoke",
    "scan_limits",
}
_DIAGNOSTIC_INPUT_FIELDS = {
    "format_version",
    "operation",
    "artifact_root",
    "target_mode",
    "claim_layer",
    "original",
    "materials",
    "environment_requirements",
    "scan_limits",
}
_TEST_SUITE_FIELDS = {"path", "case_ids"}
_CORRECTNESS_FIELDS = {"reference_path", "method", "acceptance"}
_ACCEPTANCE_FIELDS = {"metric", "operator", "value"}
_ORIGINAL_FIELDS = {"kind", "path"}
_OBJECTIVE_FIELDS = {
    "primary_metric",
    "minimum_effect",
    "constraints",
}
_PRIMARY_FIELDS = {"name", "unit", "direction", "aggregation"}
_EFFECT_FIELDS = {"value", "unit"}
_CONSTRAINT_FIELDS = {
    "name",
    "unit",
    "direction",
    "aggregation",
    "max_regression_pct",
}
_ENVIRONMENT_FIELDS = {"gpu_uuids", "required_tools"}
_VALIDITY_FIELDS = {"minimum_pairs", "confidence", "bootstrap_samples"}
_SMOKE_FIELDS = {"case_id", "resources", "runtime_limits"}
_MATERIAL_FIELDS = {"kind", "path", "tool", "tool_version", "dialect"}
_UNAVAILABLE_REASON = "diagnostic_mode"


def _load_sibling(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load readiness dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_readiness_store")
RUNTIME = _load_sibling(
    "_invocation_runtime.py", "cuda_optimizer_readiness_runtime"
)
ADAPTER = _load_sibling(
    "workload_adapter.py", "cuda_optimizer_readiness_adapter"
)


class InputError(ValueError):
    pass


def _closed(value, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise InputError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise InputError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise InputError(f"{label} contains unknown fields: {sorted(unknown)}")
    return value


def _text(value, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise InputError(f"{label} must be a non-empty bounded string")
    return value


def _string_list(value, label: str) -> list[str]:
    if type(value) is not list or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise InputError(f"{label} must be a string list")
    if len(value) != len(set(value)):
        raise InputError(f"{label} must not contain duplicates")
    return list(value)


def _positive_integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError(f"{label} must be a positive integer")
    return value


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{label} must be a finite number")
    number = float(value)
    if not (float("-inf") < number < float("inf")):
        raise InputError(f"{label} must be a finite number")
    return number


def _canonical_bytes(value) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError("readiness input must be finite JSON") from error


def _bounded_stream(value: str, limit: int = 256) -> str:
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    marker = "...<truncated>"
    return rendered[: limit - len(marker)] + marker


def _strict_json(path) -> dict:
    raw = STORE.read_regular_bytes(path)

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise InputError(f"request contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                InputError(f"request contains non-finite number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InputError("request is invalid JSON") from error
    if type(value) is not dict:
        raise InputError("request root must be an object")
    return value


def _validate_input(value) -> dict:
    if type(value) is not dict:
        raise InputError("readiness input must be an object")
    target_mode = value.get("target_mode")
    fields = _DIAGNOSTIC_INPUT_FIELDS if target_mode == "diagnostic" else _INPUT_FIELDS
    request = _closed(value, fields, "readiness input")
    if request["format_version"] != INPUT_VERSION:
        raise InputError("readiness input format_version is unsupported")
    if request["operation"] != "check":
        raise InputError("readiness input operation must be check")
    artifact_root = Path(
        os.path.abspath(os.path.expanduser(_text(request["artifact_root"], "artifact_root")))
    )
    if os.path.lexists(artifact_root):
        raise InputError("artifact_root already exists")
    if request["target_mode"] not in {"optimization", "diagnostic"}:
        raise InputError("target_mode is unsupported")
    if request["claim_layer"] not in {
        "diagnostic",
        "kernel",
        "workload",
        "serving",
    }:
        raise InputError("claim_layer is unsupported")
    if request["target_mode"] == "diagnostic":
        if request["claim_layer"] != "diagnostic":
            raise InputError(
                "diagnostic Target claim_layer must be diagnostic"
            )
        original = _closed(request["original"], _ORIGINAL_FIELDS, "original")
        if original["kind"] not in {"source_snapshot", "artifact"}:
            raise InputError("diagnostic original.kind is unsupported")
        materials = request["materials"]
        if type(materials) is not list or not materials:
            raise InputError("materials must be a non-empty list")
        normalized_materials = []
        for index, item in enumerate(materials):
            item = _closed(item, _MATERIAL_FIELDS, f"materials[{index}]")
            if item["kind"] not in {"report", "artifact"}:
                raise InputError("diagnostic material.kind is unsupported")
            normalized_materials.append(
                {
                    "kind": item["kind"],
                    "path": _text(item["path"], f"materials[{index}].path"),
                    "tool": _text(item["tool"], f"materials[{index}].tool", maximum=128),
                    "tool_version": _text(
                        item["tool_version"],
                        f"materials[{index}].tool_version",
                        maximum=128,
                    ),
                    "dialect": _text(
                        item["dialect"],
                        f"materials[{index}].dialect",
                        maximum=128,
                    ),
                }
            )
        environment = _closed(
            request["environment_requirements"],
            _ENVIRONMENT_FIELDS,
            "environment_requirements",
        )
        _string_list(environment["gpu_uuids"], "environment_requirements.gpu_uuids")
        _string_list(environment["required_tools"], "environment_requirements.required_tools")
        STORE._scan_limits(request["scan_limits"])
        return {
            **request,
            "artifact_root": str(artifact_root),
            "original": original,
            "materials": normalized_materials,
        }

    test_suite = _closed(
        request["test_suite"], _TEST_SUITE_FIELDS, "test_suite"
    )
    case_ids = _string_list(test_suite["case_ids"], "test_suite.case_ids")
    if not case_ids:
        raise InputError("test_suite.case_ids must not be empty")
    correctness = _closed(
        request["correctness"], _CORRECTNESS_FIELDS, "correctness"
    )
    if correctness["method"] != "driver":
        raise InputError("correctness.method must be driver")
    acceptance = _closed(
        correctness["acceptance"], _ACCEPTANCE_FIELDS, "correctness.acceptance"
    )
    _text(acceptance["metric"], "correctness.acceptance.metric", maximum=128)
    if acceptance["operator"] not in {
        "greater_or_equal",
        "less_or_equal",
        "equal",
    }:
        raise InputError("correctness.acceptance.operator is unsupported")
    _finite(acceptance["value"], "correctness.acceptance.value")
    original = _closed(request["original"], _ORIGINAL_FIELDS, "original")
    if original["kind"] not in {"source_snapshot", "artifact", "deployment"}:
        raise InputError("original.kind is unsupported")
    if original["kind"] == "deployment":
        raise InputError("deployment readiness is not implemented")

    objective = _closed(request["objective"], _OBJECTIVE_FIELDS, "objective")
    primary = _closed(
        objective["primary_metric"], _PRIMARY_FIELDS, "objective.primary_metric"
    )
    _text(primary["name"], "objective.primary_metric.name", maximum=128)
    _text(primary["unit"], "objective.primary_metric.unit", maximum=64)
    if primary["direction"] not in {"lower", "higher"}:
        raise InputError("objective.primary_metric.direction is unsupported")
    if primary["aggregation"] != "median":
        raise InputError("objective.primary_metric.aggregation must be median")
    minimum = _closed(
        objective["minimum_effect"], _EFFECT_FIELDS, "objective.minimum_effect"
    )
    if minimum["unit"] != "percent":
        raise InputError("objective.minimum_effect.unit must be percent")
    if _finite(minimum["value"], "objective.minimum_effect.value") < 0:
        raise InputError("objective.minimum_effect.value must be non-negative")
    if type(objective["constraints"]) is not list:
        raise InputError("objective.constraints must be a list")
    normalized_constraints = []
    constraint_names = set()
    for index, item in enumerate(objective["constraints"]):
        item = _closed(
            item,
            _CONSTRAINT_FIELDS,
            f"objective.constraints[{index}]",
        )
        name = _text(
            item["name"],
            f"objective.constraints[{index}].name",
            maximum=128,
        )
        if name in constraint_names:
            raise InputError("objective constraint names must be unique")
        constraint_names.add(name)
        if item["direction"] not in {"lower", "higher"}:
            raise InputError("objective constraint direction is unsupported")
        if item["aggregation"] != "median":
            raise InputError("objective constraint aggregation must be median")
        maximum_regression = _finite(
            item["max_regression_pct"],
            f"objective.constraints[{index}].max_regression_pct",
        )
        if maximum_regression < 0:
            raise InputError("objective constraint regression must be non-negative")
        normalized_constraints.append(
            {
                "name": name,
                "unit": _text(
                    item["unit"],
                    f"objective.constraints[{index}].unit",
                    maximum=64,
                ),
                "direction": item["direction"],
                "aggregation": "median",
                "max_regression_pct": maximum_regression,
            }
        )

    environment = _closed(
        request["environment_requirements"],
        _ENVIRONMENT_FIELDS,
        "environment_requirements",
    )
    _string_list(environment["gpu_uuids"], "environment_requirements.gpu_uuids")
    _string_list(
        environment["required_tools"], "environment_requirements.required_tools"
    )
    validity = _closed(
        request["validity_requirements"],
        _VALIDITY_FIELDS,
        "validity_requirements",
    )
    _positive_integer(validity["minimum_pairs"], "validity_requirements.minimum_pairs")
    confidence = _finite(
        validity["confidence"], "validity_requirements.confidence"
    )
    if not 0 < confidence < 1:
        raise InputError("validity_requirements.confidence must be between 0 and 1")
    _positive_integer(
        validity["bootstrap_samples"],
        "validity_requirements.bootstrap_samples",
    )
    smoke = _closed(request["smoke"], _SMOKE_FIELDS, "smoke")
    if smoke["case_id"] not in case_ids:
        raise InputError("smoke.case_id is not in test_suite.case_ids")
    STORE._scan_limits(request["scan_limits"])
    driver = ADAPTER.validate_driver(request["driver"])
    return {
        **request,
        "artifact_root": str(artifact_root),
        "driver": driver,
        "objective": {
            **objective,
            "constraints": normalized_constraints,
        },
    }


def _check_storage(parent: Path, stem: str) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    probe = parent / f".{stem}.storage-{secrets.token_hex(8)}"
    probe.mkdir(mode=0o700)
    lock_fd = None
    try:
        value = probe / "value"
        STORE.create_regular_bytes(value, b"first")
        try:
            STORE.create_regular_bytes(value, b"second")
        except FileExistsError:
            pass
        else:
            raise InputError("artifact storage lacks create-exclusive semantics")
        STORE.atomic_write_bytes(value, b"replaced")
        if STORE.read_regular_bytes(value) != b"replaced":
            raise InputError("artifact storage lacks atomic replacement")
        append = probe / "append"
        STORE.append_regular_bytes(append, b"a\n")
        STORE.append_regular_bytes(append, b"b\n")
        if STORE.read_regular_bytes(append) != b"a\nb\n":
            raise InputError("artifact storage lacks append semantics")
        lock_fd = os.open(
            probe / "lock",
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        shutil.rmtree(probe, ignore_errors=True)


def _environment(requirements: dict) -> dict:
    tools = {}
    for name in requirements["required_tools"]:
        path = shutil.which(name)
        if path is None:
            raise InputError(f"required tool is unavailable: {name}")
        tools[name] = {
            "path": str(Path(path).resolve()),
            "sha256": STORE.sha256_file(path),
        }
    return {
        "host_id": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
        },
        "gpu_uuids": list(requirements["gpu_uuids"]),
        "tools": tools,
    }


def _object_variant(object_ref: dict, kind: str) -> dict:
    return {
        "role": "original",
        "kind": kind,
        "digest": object_ref["digest"],
        "locator": object_ref["locator"],
    }


def _diagnostic_material(material: dict, object_ref: dict) -> dict:
    identity = {
        "kind": material["kind"],
        "tool": material["tool"],
        "tool_version": material["tool_version"],
        "dialect": material["dialect"],
        "object_ref": object_ref,
    }
    return {
        "id": hashlib.sha256(_canonical_bytes(identity)).hexdigest(),
        "sha256": object_ref["digest"],
        **identity,
    }


def _publish_root(temporary: Path, destination: Path) -> None:
    try:
        STORE.publish_directory_noreplace(temporary, destination)
    except FileExistsError as error:
        raise InputError("artifact_root appeared during readiness") from error


def check(value) -> dict:
    request = _validate_input(value)
    artifact_root = Path(request["artifact_root"])
    _check_storage(artifact_root.parent, artifact_root.name)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_root.name}.readiness-",
            dir=artifact_root.parent,
        )
    )
    published = False
    try:
        for relative in (
            "experiments",
            "invocations",
            "champion/selections",
            ".locks",
        ):
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        if request["target_mode"] == "diagnostic":
            original_object = STORE.freeze_path(
                temporary, request["original"]["path"], request["scan_limits"]
            )
            diagnostic_materials = []
            for material in request["materials"]:
                diagnostic_materials.append(
                    _diagnostic_material(
                        material,
                        STORE.freeze_path(
                            temporary, material["path"], request["scan_limits"]
                        ),
                    )
                )
            host_evidence = _environment(request["environment_requirements"])
            unavailable = {"status": "unavailable", "reason": _UNAVAILABLE_REASON}
            original_variant = _object_variant(
                original_object, request["original"]["kind"]
            )
            identity = {
                "target_mode": "diagnostic",
                "claim_layer": request["claim_layer"],
                "original": original_variant,
                "diagnostic_materials": diagnostic_materials,
                "environment": {"host": host_evidence, "runtime": unavailable},
                "test_suite": unavailable,
                "correctness": unavailable,
                "driver": unavailable,
                "objective": unavailable,
                "validity_requirements": unavailable,
            }
            target_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
            target = {
                "record_type": "target",
                "format_version": TARGET_VERSION,
                "id": target_id,
                **identity,
                "readiness_evidence": {
                    "checked_at_epoch": time.time(),
                    "smoke": unavailable,
                },
            }
            STORE.create_regular_json(temporary / "target.json", target)
            _publish_root(temporary, artifact_root)
            published = True
            target_path = artifact_root / "target.json"
            return {
                "status": "ready",
                "target_ref": {
                    "id": target_id,
                    "sha256": STORE.sha256_file(target_path),
                },
                "highest_claim_layer": request["claim_layer"],
                "missing": [
                    "test_suite",
                    "correctness",
                    "driver",
                    "objective",
                    "validity_requirements",
                    "environment.runtime",
                ],
            }
        original_object = STORE.freeze_path(
            temporary,
            request["original"]["path"],
            request["scan_limits"],
        )
        test_object = STORE.freeze_path(
            temporary,
            request["test_suite"]["path"],
            request["scan_limits"],
        )
        correctness_object = STORE.freeze_path(
            temporary,
            request["correctness"]["reference_path"],
            request["scan_limits"],
        )
        host_evidence = _environment(request["environment_requirements"])
        original_variant = _object_variant(
            original_object, request["original"]["kind"]
        )
        identity_without_runtime = {
            "target_mode": request["target_mode"],
            "claim_layer": request["claim_layer"],
            "test_suite": {
                "object_ref": test_object,
                "case_ids": request["test_suite"]["case_ids"],
            },
            "correctness": {
                "reference": correctness_object,
                "method": request["correctness"]["method"],
                "acceptance": request["correctness"]["acceptance"],
            },
            "original": original_variant,
            "primary_metric": request["objective"]["primary_metric"],
            "minimum_effect": request["objective"]["minimum_effect"],
            "constraints": request["objective"]["constraints"],
            "host_evidence": host_evidence,
            "driver": request["driver"],
            "validity_requirements": request["validity_requirements"],
        }
        provisional_target_id = hashlib.sha256(
            _canonical_bytes(identity_without_runtime)
        ).hexdigest()
        probe_id = "probe-" + hashlib.sha256(
            _canonical_bytes(
                {
                    "provisional_target_id": provisional_target_id,
                    "smoke": request["smoke"],
                    "driver_identity": request["driver"]["identity"],
                }
            )
        ).hexdigest()[:24]
        smoke_root = temporary / ".readiness-smoke"
        smoke_root.mkdir()
        inputs_root = smoke_root / "inputs"
        inputs_root.mkdir()
        smoke_original = STORE.materialize_object(
            temporary,
            original_object,
            inputs_root / "original",
        )
        smoke_test_suite = STORE.materialize_object(
            temporary,
            test_object,
            inputs_root / "test-suite",
        )
        smoke_correctness = STORE.materialize_object(
            temporary,
            correctness_object,
            inputs_root / "correctness-reference",
        )
        driver_output_dir = smoke_root / "output"
        driver_output_dir.mkdir()
        driver_output = driver_output_dir / "result.json"
        driver_request = ADAPTER.build_driver_request(
            target_id=provisional_target_id,
            execution_id=probe_id,
            operation="readiness_smoke",
            driver=request["driver"],
            subjects=[
                {
                    "role": "original",
                    "variant": {
                        "kind": original_variant["kind"],
                        "digest": original_variant["digest"],
                        "locator": str(smoke_original),
                    },
                }
            ],
            test_suite={
                "digest": test_object["digest"],
                "locator": str(smoke_test_suite),
                "case_ids": request["test_suite"]["case_ids"],
            },
            correctness={
                "reference": {
                    "digest": correctness_object["digest"],
                    "locator": str(smoke_correctness),
                },
                "method": request["correctness"]["method"],
                "acceptance": request["correctness"]["acceptance"],
            },
            objective={
                "primary_metric": request["objective"]["primary_metric"],
                "constraints": request["objective"]["constraints"],
            },
            acquisition={
                "lifecycle": "isolated_process",
                "shared_state": [],
                "rebuilt_state": ["process"],
            },
            case={"id": request["smoke"]["case_id"]},
            sampling={"kind": "smoke", "repetitions": 2},
            output_path=driver_output,
        )
        driver_request_path = smoke_root / "driver-request.json"
        STORE.atomic_write_bytes(
            driver_request_path,
            _canonical_bytes(driver_request) + b"\n",
        )
        command_result = RUNTIME.probe(
            {
                "argv": ADAPTER.build_argv(
                    request["driver"], driver_request_path
                ),
                "cwd": str(smoke_root),
                "env": {},
                "output_limit_bytes": 64 * 1024,
                "required_gpu_uuids": request["smoke"]["resources"][
                    "gpu_uuids"
                ],
            },
            request["smoke"]["runtime_limits"],
            request["smoke"]["resources"],
        )
        if command_result["status"] != "completed":
            raise InputError(
                "driver smoke did not complete: "
                f"stop_reason={command_result['stop_reason']} "
                f"returncode={command_result['returncode']!r} "
                f"stdout={_bounded_stream(command_result['stdout'])} "
                f"stderr={_bounded_stream(command_result['stderr'])} "
                f"cleanup_status={command_result['cleanup_status']}"
            )
        smoke_result_object = STORE.freeze_path(
            temporary,
            driver_output_dir,
            request["scan_limits"],
        )
        frozen_smoke_output = STORE.materialize_object(
            temporary,
            smoke_result_object,
            smoke_root / "frozen-output",
        )
        smoke_output_manifest = STORE._load_object_manifest(
            temporary,
            smoke_result_object,
            verify_payload=False,
        )
        driver_result = ADAPTER.validate_driver_result(
            frozen_smoke_output / "result.json",
            driver_request,
            bundle_manifest=smoke_output_manifest,
        )
        smoke_evidence = ADAPTER.evidence_for_role(driver_result, "original")
        ADAPTER.validate_measurement_contract(
            smoke_evidence["measurements"],
            driver_request["objective"],
            driver_request["sampling"],
        )
        correctness = smoke_evidence["correctness"]
        correctness_gate = ADAPTER.evaluate_correctness(
            correctness,
            request["correctness"]["acceptance"],
        )
        if not correctness_gate["passed"]:
            raise InputError("driver smoke correctness failed")
        runtime_environment = driver_result["environment"]
        if sorted(runtime_environment["gpu_uuids"]) != sorted(
            request["environment_requirements"]["gpu_uuids"]
        ):
            raise InputError("driver smoke GPU identity does not match requirements")
        identity = {
            **identity_without_runtime,
            "environment": {
                "host": host_evidence,
                "runtime": runtime_environment,
            },
        }
        identity.pop("host_evidence")
        target_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        shutil.rmtree(smoke_root)
        target = {
            "record_type": "target",
            "format_version": TARGET_VERSION,
            "id": target_id,
            **identity,
            "readiness_evidence": {
                "checked_at_epoch": time.time(),
                "probe_id": probe_id,
                "command": driver_request,
                "command_result": command_result,
                "driver_output_ref": smoke_result_object,
                "driver_artifacts": driver_result["artifacts"],
                "correctness_gate": correctness_gate,
            },
        }
        STORE.create_regular_json(temporary / "target.json", target)
        _publish_root(temporary, artifact_root)
        published = True
        target_path = artifact_root / "target.json"
        return {
            "status": "ready",
            "target_ref": {
                "id": target_id,
                "sha256": STORE.sha256_file(target_path),
            },
            "probe_id": probe_id,
            "highest_claim_layer": request["claim_layer"],
            "missing": [],
        }
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _emit_error(error: BaseException) -> int:
    payload = {
        "status": "rejected",
        "error_code": "invalid_readiness_input",
        "error": str(error)[:1024],
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and freeze one V1.4 optimization target."
    )
    parser.add_argument("operation", choices=("check",))
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        value = _strict_json(args.request)
        if value.get("operation") != args.operation:
            raise InputError("CLI operation does not match request")
        result = check(value)
    except (InputError, OSError, ValueError, TimeoutError) as error:
        return _emit_error(error)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
