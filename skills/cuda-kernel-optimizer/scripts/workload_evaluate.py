#!/usr/bin/env python3
"""Execute explicit V1.4 workload measurements.

The module records baseline and candidate facts.  It does not choose the next
optimization action or adopt a candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import statistics
import sys
import time
from pathlib import Path


INPUT_VERSION = "cuda-kernel-optimizer/evaluator-input-v1"
RESULT_VERSION = "cuda-kernel-optimizer/evaluator-result-v1"
TOOL_IDENTITY_VERSION = "cuda-kernel-optimizer/evaluator-tool-v1"

_BASELINE_REQUIRED = {
    "format_version",
    "operation",
    "artifact_root",
    "target_ref",
    "sampling_design",
    "resources",
    "operation_timeout_seconds",
    "command_timeout_seconds",
    "resource_wait_timeout_seconds",
    "cleanup_timeout_seconds",
    "launch_deadline",
}
_BASELINE_OPTIONAL = {"absolute_deadline", "retry_of"}
_TARGET_REF_FIELDS = {"id", "sha256"}
_SAMPLING_FIELDS = {"case_ids", "samples_per_case", "seed"}
_RESOURCE_FIELDS = {"host_id", "gpu_uuids"}
_EXPERIMENT_FIELDS = {
    "format_version",
    "operation",
    "artifact_root",
    "target_ref",
    "baseline_ref",
    "source_base",
    "candidate",
    "hypothesis",
    "mechanism_key",
    "claim_layer",
    "cheapest_falsifier",
    "screen_design",
    "estimated_cost",
    "minimum_effect",
    "reject_if",
    "promote_if",
    "change_scope",
    "max_risk",
}
_INVOCATION_REF_FIELDS = {"invocation_id", "sha256"}
_SOURCE_FIELDS = {"kind", "path"}
_FALSIFIER_FIELDS = {"kind", "reason"}
_SCREEN_DESIGN_FIELDS = {"enabled", "kind", "reason", "claim"}
_COST_FIELDS = {"screen", "target"}
_COST_RANGE_FIELDS = {"p50_seconds", "p90_seconds", "gpu_count", "basis"}
_EFFECT_FIELDS = {"value", "unit"}
_CONDITION_FIELDS = {"kind"}
_SCREEN_REQUIRED = {
    "format_version",
    "operation",
    "artifact_root",
    "target_ref",
    "experiment_ref",
    "sampling_design",
    "resources",
    "operation_timeout_seconds",
    "command_timeout_seconds",
    "resource_wait_timeout_seconds",
    "cleanup_timeout_seconds",
    "launch_deadline",
}
_SCREEN_OPTIONAL = {"absolute_deadline", "retry_of"}
_TARGET_REQUIRED = set(_SCREEN_REQUIRED)
_TARGET_OPTIONAL = set(_SCREEN_OPTIONAL)
_FINAL_AUDIT_REQUIRED = _TARGET_REQUIRED - {"experiment_ref"}
_FINAL_AUDIT_OPTIONAL = set(_SCREEN_OPTIONAL)
_EXPERIMENT_REF_FIELDS = {"id", "sha256"}
_SCREEN_SAMPLING_FIELDS = {"case_ids", "pairs", "seed"}
_LIFECYCLE_FIELDS = {
    "format_version",
    "operation",
    "artifact_root",
    "invocation_id",
}
_OBJECT_SCAN_LIMITS = {
    "max_files": 1,
    "max_total_bytes": 4 * 1024 * 1024,
    "max_wall_seconds": 2.0,
}


def _load_sibling(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_sibling("artifact_store.py", "cuda_optimizer_evaluator_store")
RUNTIME = _load_sibling(
    "_invocation_runtime.py", "cuda_optimizer_evaluator_runtime"
)
ADAPTER = _load_sibling(
    "workload_adapter.py", "cuda_optimizer_evaluator_adapter"
)
DESIGN = _load_sibling(
    "experiment_design.py", "cuda_optimizer_evaluator_design"
)
PAIRED = _load_sibling("paired_stats.py", "cuda_optimizer_evaluator_stats")


class EvaluatorError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _closed(value, required: set[str], optional: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise EvaluatorError("invalid_evaluator_input", f"{label} must be an object")
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise EvaluatorError(
            "invalid_evaluator_input",
            f"{label} is missing fields: {sorted(missing)}",
        )
    if unknown:
        raise EvaluatorError(
            "invalid_evaluator_input",
            f"{label} contains unknown fields: {sorted(unknown)}",
        )
    return value


def _finite(value, label: str, *, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label} must be a finite number"
        )
    return number


def _positive_integer(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label} must be a positive integer"
        )
    return value


def _string_list(value, label: str) -> list[str]:
    if type(value) is not list or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label} must be a string list"
        )
    if len(value) != len(set(value)):
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label} must not contain duplicates"
        )
    return list(value)


def _strict_json(path) -> dict:
    raw = STORE.read_regular_bytes(path)

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise EvaluatorError(
                    "invalid_evaluator_input",
                    f"request contains duplicate key: {key}",
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvaluatorError(
                    "invalid_evaluator_input",
                    f"request contains non-finite number: {token}",
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorError(
            "invalid_evaluator_input", "request is invalid JSON"
        ) from error
    if type(value) is not dict:
        raise EvaluatorError(
            "invalid_evaluator_input", "request root must be an object"
        )
    return value


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
        raise EvaluatorError(
            "invalid_evaluator_input", "request must contain finite JSON"
        ) from error


def _sha256(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label} must be a lowercase SHA-256"
        )
    return value


def _load_target(root: Path, reference) -> dict:
    reference = _closed(
        reference,
        _TARGET_REF_FIELDS,
        set(),
        "target_ref",
    )
    expected_id = reference["id"]
    if not isinstance(expected_id, str) or not expected_id:
        raise EvaluatorError(
            "invalid_evaluator_input", "target_ref.id must be a non-empty string"
        )
    expected_digest = _sha256(reference["sha256"], "target_ref.sha256")
    path = root / "target.json"
    try:
        payload = STORE.read_regular_bytes(path)
    except (OSError, ValueError) as error:
        raise EvaluatorError("target_not_found", "target record is unavailable") from error
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise EvaluatorError("target_changed", "target record digest changed")
    try:
        target = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorError("target_invalid", "target record is invalid") from error
    if (
        type(target) is not dict
        or target.get("record_type") != "target"
        or target.get("id") != expected_id
    ):
        raise EvaluatorError("target_invalid", "target record identity is invalid")
    if target.get("target_mode") != "optimization":
        raise EvaluatorError(
            "target_not_optimizable",
            "diagnostic target cannot run a workload baseline",
        )
    return target


def _load_baseline(root: Path, reference, target_ref: dict) -> dict:
    reference = _closed(
        reference,
        _INVOCATION_REF_FIELDS,
        set(),
        "baseline_ref",
    )
    invocation_id = reference["invocation_id"]
    if (
        not isinstance(invocation_id, str)
        or not invocation_id
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in invocation_id
        )
    ):
        raise EvaluatorError(
            "invalid_evaluator_input", "baseline_ref.invocation_id is invalid"
        )
    expected_digest = _sha256(reference["sha256"], "baseline_ref.sha256")
    path = root / "invocations" / invocation_id / "result.json"
    try:
        payload = STORE.read_regular_bytes(path)
    except (OSError, ValueError) as error:
        raise EvaluatorError(
            "baseline_not_found", "baseline result is unavailable"
        ) from error
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise EvaluatorError("baseline_changed", "baseline result digest changed")
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorError("baseline_invalid", "baseline result is invalid") from error
    if (
        type(result) is not dict
        or result.get("operation") != "baseline"
        or result.get("target_ref") != target_ref
        or result.get("execution_status") != "succeeded"
        or result.get("measurement_validity") != "valid"
        or result.get("verdict") != "passed"
        or result.get("cleanup_status") != "confirmed"
    ):
        raise EvaluatorError(
            "baseline_invalid", "baseline is not a valid original measurement"
        )
    return result


def _load_experiment(root: Path, reference, target_ref: dict) -> dict:
    reference = _closed(
        reference,
        _EXPERIMENT_REF_FIELDS,
        set(),
        "experiment_ref",
    )
    experiment_id = reference["id"]
    if (
        not isinstance(experiment_id, str)
        or not experiment_id.startswith("exp-")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in experiment_id
        )
    ):
        raise EvaluatorError(
            "invalid_evaluator_input", "experiment_ref.id is invalid"
        )
    expected_digest = _sha256(reference["sha256"], "experiment_ref.sha256")
    path = root / "experiments" / f"{experiment_id}.json"
    try:
        payload = STORE.read_regular_bytes(path)
    except (OSError, ValueError) as error:
        raise EvaluatorError(
            "experiment_not_found", "experiment record is unavailable"
        ) from error
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise EvaluatorError(
            "experiment_changed", "experiment record digest changed"
        )
    try:
        experiment = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorError(
            "experiment_invalid", "experiment record is invalid"
        ) from error
    if (
        type(experiment) is not dict
        or experiment.get("record_type") != "experiment"
        or experiment.get("id") != experiment_id
        or experiment.get("target_ref") != target_ref
    ):
        raise EvaluatorError(
            "experiment_invalid", "experiment record identity is invalid"
        )
    return experiment


def _validate_resources(value) -> dict:
    resources = _closed(value, _RESOURCE_FIELDS, set(), "resources")
    host_id = resources["host_id"]
    if not isinstance(host_id, str) or not host_id:
        raise EvaluatorError(
            "invalid_evaluator_input",
            "resources.host_id must be a non-empty string",
        )
    return {
        "host_id": host_id,
        "gpu_uuids": sorted(_string_list(resources["gpu_uuids"], "resources.gpu_uuids")),
    }


def _validate_baseline_input(value) -> tuple[dict, Path, dict]:
    request = _closed(
        value,
        _BASELINE_REQUIRED,
        _BASELINE_OPTIONAL,
        "baseline input",
    )
    if request["format_version"] != INPUT_VERSION or request["operation"] != "baseline":
        raise EvaluatorError(
            "invalid_evaluator_input",
            "baseline input version or operation is unsupported",
        )
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"])))
    )
    if not root.is_dir():
        raise EvaluatorError("target_not_found", "artifact_root is unavailable")
    target = _load_target(root, request["target_ref"])
    sampling = _closed(
        request["sampling_design"],
        _SAMPLING_FIELDS,
        set(),
        "sampling_design",
    )
    case_ids = _string_list(sampling["case_ids"], "sampling_design.case_ids")
    if not case_ids:
        raise EvaluatorError(
            "invalid_evaluator_input", "sampling_design.case_ids must not be empty"
        )
    frozen_cases = target["test_suite"]["case_ids"]
    if any(case_id not in frozen_cases for case_id in case_ids):
        raise EvaluatorError(
            "invalid_evaluator_input",
            "sampling_design contains a case outside the frozen test suite",
        )
    samples_per_case = _positive_integer(
        sampling["samples_per_case"],
        "sampling_design.samples_per_case",
    )
    seed = sampling["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluatorError(
            "invalid_evaluator_input",
            "sampling_design.seed must be a non-negative integer",
        )
    resources = _validate_resources(request["resources"])
    if resources["gpu_uuids"] != sorted(
        target["environment"]["runtime"]["gpu_uuids"]
    ):
        raise EvaluatorError(
            "invalid_evaluator_input",
            "resources.gpu_uuids must match the frozen target environment",
        )
    normalized = {
        **request,
        "artifact_root": str(root),
        "sampling_design": {
            "case_ids": case_ids,
            "samples_per_case": samples_per_case,
            "seed": seed,
        },
        "resources": resources,
    }
    for field in (
        "operation_timeout_seconds",
        "command_timeout_seconds",
        "resource_wait_timeout_seconds",
        "cleanup_timeout_seconds",
    ):
        normalized[field] = _finite(request[field], field, positive=True)
    normalized["launch_deadline"] = _finite(
        request["launch_deadline"], "launch_deadline"
    )
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(
            request["absolute_deadline"], "absolute_deadline"
        )
    return normalized, root, target


def _tool_identity() -> dict:
    names = (
        "workload_evaluate.py",
        "_invocation_runtime.py",
        "workload_adapter.py",
        "artifact_store.py",
        "experiment_design.py",
        "paired_stats.py",
    )
    implementations = []
    for name in names:
        path = Path(__file__).with_name(name)
        implementations.append(
            {
                "name": name,
                "sha256": STORE.sha256_file(path),
            }
        )
    identity = {
        "version": TOOL_IDENTITY_VERSION,
        "result_contract": RESULT_VERSION,
        "implementations": implementations,
    }
    identity["digest"] = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    return identity


def _frozen_baseline_request(request: dict, target: dict) -> dict:
    frozen = {
        "operation": "baseline",
        "target_ref": request["target_ref"],
        "variant_refs": [target["original"]],
        "driver": ADAPTER.verify_driver(target["driver"]),
        "objective": {
            "primary_metric": target["primary_metric"],
            "constraints": target["constraints"],
        },
        "sampling_design": request["sampling_design"],
        "resources": request["resources"],
        "cleanup": target["driver"]["cleanup_contract"],
        "tool_identity": _tool_identity(),
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


def baseline(value, *, wait_for_result: bool) -> dict:
    request, root, target = _validate_baseline_input(value)
    frozen = _frozen_baseline_request(request, target)
    worker_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
    ]
    return RUNTIME.submit(root, frozen, worker_argv, wait_for_result)


def _text(value, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise EvaluatorError(
            "invalid_evaluator_input",
            f"{label} must be a non-empty bounded string",
        )
    return value


def _source(value, label: str) -> dict:
    source = _closed(value, _SOURCE_FIELDS, set(), label)
    if source["kind"] not in {"source_snapshot", "artifact"}:
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label}.kind is unsupported"
        )
    return {
        "kind": source["kind"],
        "path": str(
            Path(
                os.path.abspath(
                    os.path.expanduser(os.fspath(source["path"]))
                )
            )
        ),
    }


def _cost_range(value, label: str) -> dict:
    value = _closed(value, _COST_RANGE_FIELDS, set(), label)
    p50 = _finite(value["p50_seconds"], f"{label}.p50_seconds", positive=True)
    p90 = _finite(value["p90_seconds"], f"{label}.p90_seconds", positive=True)
    if p90 < p50:
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label}.p90_seconds must be at least p50"
        )
    gpu_count = value["gpu_count"]
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
        raise EvaluatorError(
            "invalid_evaluator_input",
            f"{label}.gpu_count must be a non-negative integer",
        )
    return {
        "p50_seconds": p50,
        "p90_seconds": p90,
        "gpu_count": gpu_count,
        "basis": _text(value["basis"], f"{label}.basis"),
    }


def _conditions(value, label: str) -> list[dict]:
    if type(value) is not list or not value:
        raise EvaluatorError(
            "invalid_evaluator_input", f"{label} must be a non-empty list"
        )
    normalized = []
    for index, item in enumerate(value):
        item = _closed(item, _CONDITION_FIELDS, set(), f"{label}[{index}]")
        normalized.append(
            {"kind": _text(item["kind"], f"{label}[{index}].kind", maximum=128)}
        )
    return normalized


def _manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def _changed_paths(
    base_manifest: dict,
    candidate_manifest: dict,
) -> list[str]:
    def entries(manifest):
        return {
            entry["path"]: {
                key: value
                for key, value in entry.items()
                if key not in {"path"}
            }
            for entry in manifest["entries"]
        }

    base = entries(base_manifest)
    candidate = entries(candidate_manifest)
    return sorted(
        path
        for path in set(base) | set(candidate)
        if base.get(path) != candidate.get(path)
    )


def _current_reference(root: Path, target: dict) -> tuple[dict, dict | None]:
    target_ref = {
        "id": target["id"],
        "sha256": STORE.sha256_file(root / "target.json"),
    }
    current_path = root / "champion" / "current.json"
    try:
        os.lstat(current_path)
    except FileNotFoundError:
        return target["original"], None
    except OSError as error:
        raise EvaluatorError(
            "champion_invalid", "current Champion selection is invalid"
        ) from error
    try:
        pointer = json.loads(STORE.read_regular_bytes(current_path).decode("utf-8"))
        if (
            type(pointer) is not dict
            or set(pointer)
            != {
                "record_type",
                "format_version",
                "target_ref",
                "selection_ref",
            }
            or pointer["record_type"] != "champion_pointer"
            or pointer["format_version"]
            != "cuda-kernel-optimizer/champion-pointer-v1"
            or pointer["target_ref"] != target_ref
            or type(pointer["selection_ref"]) is not dict
            or set(pointer["selection_ref"]) != {"id", "sha256"}
        ):
            raise ValueError("invalid Champion pointer")
        selection_ref = pointer["selection_ref"]
        selection_id = selection_ref["id"]
        selection_digest = selection_ref["sha256"]
        if (
            not isinstance(selection_id, str)
            or re.fullmatch(r"sel-[a-z0-9-]+", selection_id) is None
            or not isinstance(selection_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", selection_digest) is None
        ):
            raise ValueError("invalid Champion selection reference")
        selection_payload = STORE.read_regular_bytes(
            root / "champion" / "selections" / f"{selection_id}.json"
        )
        if hashlib.sha256(selection_payload).hexdigest() != selection_digest:
            raise ValueError("Champion selection digest changed")
        selection = json.loads(selection_payload.decode("utf-8"))
        variant = selection.get("selected_variant") if type(selection) is dict else None
        if (
            type(selection) is not dict
            or selection.get("record_type") != "champion_selection"
            or selection.get("format_version")
            != "cuda-kernel-optimizer/champion-selection-v1"
            or selection.get("id") != selection_id
            or selection.get("target_ref") != target_ref
            or type(variant) is not dict
            or set(variant) != {"role", "kind", "digest", "locator"}
            or not isinstance(variant["digest"], str)
            or re.fullmatch(r"[0-9a-f]{64}", variant["digest"]) is None
            or not isinstance(variant["locator"], str)
        ):
            raise ValueError("invalid Champion selection")
        return variant, dict(selection_ref)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluatorError(
            "champion_invalid", "current Champion selection is invalid"
        ) from error


def _current_pointer_digest(root: Path) -> str | None:
    path = root / "champion" / "current.json"
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise EvaluatorError(
            "champion_invalid", "current Champion pointer is unavailable"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise EvaluatorError(
            "champion_invalid", "current Champion pointer is invalid"
        )
    try:
        return STORE.sha256_file(path)
    except (OSError, ValueError) as error:
        raise EvaluatorError(
            "champion_invalid", "current Champion pointer is invalid"
        ) from error


def create_experiment(value) -> dict:
    request = _closed(
        value,
        _EXPERIMENT_FIELDS,
        set(),
        "experiment input",
    )
    if (
        request["format_version"] != INPUT_VERSION
        or request["operation"] != "experiment"
    ):
        raise EvaluatorError(
            "invalid_evaluator_input",
            "experiment input version or operation is unsupported",
        )
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"])))
    )
    target = _load_target(root, request["target_ref"])
    _load_baseline(root, request["baseline_ref"], request["target_ref"])
    source_base = _source(request["source_base"], "source_base")
    candidate = _source(request["candidate"], "candidate")
    hypothesis = _text(request["hypothesis"], "hypothesis")
    mechanism_key = _text(
        request["mechanism_key"], "mechanism_key", maximum=256
    )
    if re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", mechanism_key) is None:
        raise EvaluatorError(
            "invalid_evaluator_input", "mechanism_key is not normalized"
        )
    claim_layer = request["claim_layer"]
    layer_order = {"kernel": 1, "workload": 2, "serving": 3}
    if claim_layer not in layer_order or layer_order[claim_layer] > layer_order[
        target["claim_layer"]
    ]:
        raise EvaluatorError(
            "invalid_evaluator_input",
            "claim_layer exceeds the frozen target claim layer",
        )
    falsifier = _closed(
        request["cheapest_falsifier"],
        _FALSIFIER_FIELDS,
        set(),
        "cheapest_falsifier",
    )
    if falsifier["kind"] not in {"none", "command"}:
        raise EvaluatorError(
            "invalid_evaluator_input", "cheapest_falsifier.kind is unsupported"
        )
    falsifier = {
        "kind": falsifier["kind"],
        "reason": _text(falsifier["reason"], "cheapest_falsifier.reason"),
    }
    screen = _closed(
        request["screen_design"],
        _SCREEN_DESIGN_FIELDS,
        set(),
        "screen_design",
    )
    if type(screen["enabled"]) is not bool:
        raise EvaluatorError(
            "invalid_evaluator_input", "screen_design.enabled must be boolean"
        )
    if screen["kind"] not in {"diagnostic_proxy", "conservative_bound"}:
        raise EvaluatorError(
            "invalid_evaluator_input", "screen_design.kind is unsupported"
        )
    screen = {
        "enabled": screen["enabled"],
        "kind": screen["kind"],
        "reason": _text(screen["reason"], "screen_design.reason"),
        "claim": _text(screen["claim"], "screen_design.claim"),
    }
    costs = _closed(
        request["estimated_cost"],
        _COST_FIELDS,
        set(),
        "estimated_cost",
    )
    costs = {
        "screen": _cost_range(costs["screen"], "estimated_cost.screen"),
        "target": _cost_range(costs["target"], "estimated_cost.target"),
    }
    minimum = _closed(
        request["minimum_effect"],
        _EFFECT_FIELDS,
        set(),
        "minimum_effect",
    )
    minimum = {
        "value": _finite(minimum["value"], "minimum_effect.value"),
        "unit": minimum["unit"],
    }
    if minimum["value"] < 0 or minimum != target["minimum_effect"]:
        raise EvaluatorError(
            "invalid_evaluator_input",
            "minimum_effect must equal the frozen target threshold",
        )
    reject_if = _conditions(request["reject_if"], "reject_if")
    promote_if = _conditions(request["promote_if"], "promote_if")
    change_scope = _string_list(request["change_scope"], "change_scope")
    if not change_scope:
        raise EvaluatorError(
            "invalid_evaluator_input", "change_scope must not be empty"
        )
    for path in change_scope:
        candidate_path = Path(path)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise EvaluatorError(
                "invalid_evaluator_input", "change_scope must contain safe relative paths"
            )
    if request["max_risk"] not in {"low", "medium", "high"}:
        raise EvaluatorError(
            "invalid_evaluator_input", "max_risk is unsupported"
        )
    reference_variant, reference_selection_ref = _current_reference(root, target)

    try:
        base_manifest, _base_payloads = STORE._snapshot_source(
            source_base["path"],
            {
                "max_files": 10000,
                "max_total_bytes": 2 * 1024 * 1024 * 1024,
                "max_wall_seconds": 30.0,
            },
        )
        candidate_manifest, candidate_payloads = STORE._snapshot_source(
            candidate["path"],
            {
                "max_files": 10000,
                "max_total_bytes": 2 * 1024 * 1024 * 1024,
                "max_wall_seconds": 30.0,
            },
        )
    except (OSError, ValueError, TimeoutError) as error:
        raise EvaluatorError(
            "candidate_invalid", "source base or candidate cannot be frozen"
        ) from error
    if _manifest_digest(base_manifest) != target["original"]["digest"]:
        raise EvaluatorError(
            "source_base_changed",
            "source_base does not match the frozen original Variant",
        )
    changed_paths = _changed_paths(base_manifest, candidate_manifest)
    if not changed_paths:
        raise EvaluatorError("candidate_unchanged", "candidate has no content change")
    if any(
        not any(
            path == allowed or path.startswith(allowed.rstrip("/") + "/")
            for allowed in change_scope
        )
        for path in changed_paths
    ):
        raise EvaluatorError(
            "change_scope_exceeded",
            "candidate changes paths outside change_scope",
        )
    candidate_object = STORE.freeze_snapshot(
        root,
        candidate_manifest,
        candidate_payloads,
    )
    candidate_variant = {
        "role": "candidate",
        "kind": candidate["kind"],
        "digest": candidate_object["digest"],
        "locator": candidate_object["locator"],
    }
    core = {
        "target_ref": request["target_ref"],
        "baseline_ref": request["baseline_ref"],
        "source_base": target["original"],
        "reference_variant": reference_variant,
        "reference_selection_ref": reference_selection_ref,
        "candidate": candidate_variant,
        "hypothesis": hypothesis,
        "mechanism_key": mechanism_key,
        "claim_layer": claim_layer,
        "cheapest_falsifier": falsifier,
        "screen_design": screen,
        "estimated_cost": costs,
        "minimum_effect": minimum,
        "reject_if": reject_if,
        "promote_if": promote_if,
        "change_scope": change_scope,
        "changed_paths": changed_paths,
        "max_risk": request["max_risk"],
    }
    experiment_id = "exp-" + hashlib.sha256(_canonical_bytes(core)).hexdigest()
    record = {
        "record_type": "experiment",
        "format_version": "cuda-kernel-optimizer/experiment-v1",
        "id": experiment_id,
        **core,
    }
    path = root / "experiments" / f"{experiment_id}.json"
    try:
        STORE.create_regular_json(path, record)
    except FileExistsError:
        existing = STORE.read_regular_bytes(path)
        expected = (
            json.dumps(
                record,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if existing != expected:
            raise EvaluatorError(
                "experiment_conflict",
                "existing experiment does not match deterministic identity",
            )
    return {
        "status": "created",
        "experiment_ref": {
            "id": experiment_id,
            "sha256": STORE.sha256_file(path),
        },
        "candidate_ref": {
            "digest": candidate_variant["digest"],
            "locator": candidate_variant["locator"],
        },
        "reference_selection_ref": reference_selection_ref,
    }


def _validate_screen_input(value) -> tuple[dict, Path, dict, dict]:
    request = _closed(
        value,
        _SCREEN_REQUIRED,
        _SCREEN_OPTIONAL,
        "screen input",
    )
    if request["format_version"] != INPUT_VERSION or request["operation"] != "screen":
        raise EvaluatorError(
            "invalid_evaluator_input",
            "screen input version or operation is unsupported",
        )
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"])))
    )
    target = _load_target(root, request["target_ref"])
    experiment = _load_experiment(
        root,
        request["experiment_ref"],
        request["target_ref"],
    )
    _load_baseline(root, experiment["baseline_ref"], request["target_ref"])
    current_variant, current_selection = _current_reference(root, target)
    if (
        current_variant != experiment["reference_variant"]
        or current_selection != experiment["reference_selection_ref"]
    ):
        raise EvaluatorError(
            "stale_reference",
            "experiment reference is no longer the current Champion",
        )
    sampling = _closed(
        request["sampling_design"],
        _SCREEN_SAMPLING_FIELDS,
        set(),
        "sampling_design",
    )
    case_ids = _string_list(sampling["case_ids"], "sampling_design.case_ids")
    if not case_ids or any(
        case_id not in target["test_suite"]["case_ids"] for case_id in case_ids
    ):
        raise EvaluatorError(
            "invalid_evaluator_input",
            "sampling_design cases must belong to the frozen test suite",
        )
    pairs = _positive_integer(sampling["pairs"], "sampling_design.pairs")
    seed = sampling["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluatorError(
            "invalid_evaluator_input",
            "sampling_design.seed must be a non-negative integer",
        )
    resources = _validate_resources(request["resources"])
    if resources["gpu_uuids"] != sorted(
        target["environment"]["runtime"]["gpu_uuids"]
    ):
        raise EvaluatorError(
            "invalid_evaluator_input",
            "resources.gpu_uuids must match the frozen target environment",
        )
    normalized = {
        **request,
        "artifact_root": str(root),
        "sampling_design": {
            "case_ids": case_ids,
            "pairs": pairs,
            "seed": seed,
            "orders": DESIGN.balanced_pair_orders(pairs, seed=seed),
        },
        "resources": resources,
    }
    for field in (
        "operation_timeout_seconds",
        "command_timeout_seconds",
        "resource_wait_timeout_seconds",
        "cleanup_timeout_seconds",
    ):
        normalized[field] = _finite(request[field], field, positive=True)
    normalized["launch_deadline"] = _finite(
        request["launch_deadline"], "launch_deadline"
    )
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(
            request["absolute_deadline"], "absolute_deadline"
        )
    return normalized, root, target, experiment


def _frozen_screen_request(
    request: dict,
    target: dict,
    experiment: dict,
) -> dict:
    frozen = {
        "operation": "screen",
        "target_ref": request["target_ref"],
        "experiment_ref": request["experiment_ref"],
        "baseline_ref": experiment["baseline_ref"],
        "variant_refs": [
            experiment["reference_variant"],
            experiment["candidate"],
        ],
        "reference_selection_ref": experiment["reference_selection_ref"],
        "driver": ADAPTER.verify_driver(target["driver"]),
        "objective": {
            "primary_metric": target["primary_metric"],
            "minimum_effect": experiment["minimum_effect"],
            "constraints": target["constraints"],
        },
        "screen_design": experiment["screen_design"],
        "cheapest_falsifier": experiment["cheapest_falsifier"],
        "reject_if": experiment["reject_if"],
        "sampling_design": request["sampling_design"],
        "validity_requirements": target["validity_requirements"],
        "resources": request["resources"],
        "cleanup": target["driver"]["cleanup_contract"],
        "tool_identity": _tool_identity(),
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


def screen(value, *, wait_for_result: bool) -> dict:
    request, root, target, experiment = _validate_screen_input(value)
    if not experiment["screen_design"]["enabled"]:
        raise EvaluatorError(
            "screen_not_enabled",
            "experiment does not define a screen operation",
        )
    if experiment["cheapest_falsifier"]["kind"] != "none":
        raise EvaluatorError(
            "falsifier_not_executable",
            "the frozen falsifier has no supported executable contract",
        )
    frozen = _frozen_screen_request(request, target, experiment)
    worker_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
    ]
    return RUNTIME.submit(root, frozen, worker_argv, wait_for_result)


def _comparison_input(
    value,
    *,
    operation: str,
) -> tuple[dict, Path, dict, dict | None]:
    required = (
        _TARGET_REQUIRED
        if operation == "target"
        else _FINAL_AUDIT_REQUIRED
    )
    optional = (
        _TARGET_OPTIONAL
        if operation == "target"
        else _FINAL_AUDIT_OPTIONAL
    )
    request = _closed(value, required, optional, f"{operation} input")
    if (
        request["format_version"] != INPUT_VERSION
        or request["operation"] != operation
    ):
        raise EvaluatorError(
            "invalid_evaluator_input",
            f"{operation} input version or operation is unsupported",
        )
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(request["artifact_root"])))
    )
    target = _load_target(root, request["target_ref"])
    experiment = None
    if operation == "target":
        experiment = _load_experiment(
            root,
            request["experiment_ref"],
            request["target_ref"],
        )
        _load_baseline(root, experiment["baseline_ref"], request["target_ref"])
    sampling = _closed(
        request["sampling_design"],
        _SCREEN_SAMPLING_FIELDS,
        set(),
        "sampling_design",
    )
    case_ids = _string_list(sampling["case_ids"], "sampling_design.case_ids")
    if not case_ids or any(
        case_id not in target["test_suite"]["case_ids"] for case_id in case_ids
    ):
        raise EvaluatorError(
            "invalid_evaluator_input",
            "sampling_design cases must belong to the frozen test suite",
        )
    pairs = _positive_integer(sampling["pairs"], "sampling_design.pairs")
    if pairs < target["validity_requirements"]["minimum_pairs"]:
        raise EvaluatorError(
            "invalid_evaluator_input",
            "formal comparison pairs are below the Target minimum",
        )
    seed = sampling["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluatorError(
            "invalid_evaluator_input",
            "sampling_design.seed must be a non-negative integer",
        )
    resources = _validate_resources(request["resources"])
    if resources["gpu_uuids"] != sorted(
        target["environment"]["runtime"]["gpu_uuids"]
    ):
        raise EvaluatorError(
            "invalid_evaluator_input",
            "resources.gpu_uuids must match the frozen target environment",
        )
    normalized = {
        **request,
        "artifact_root": str(root),
        "sampling_design": {
            "case_ids": case_ids,
            "pairs": pairs,
            "seed": seed,
            "orders": DESIGN.balanced_pair_orders(pairs, seed=seed),
        },
        "resources": resources,
    }
    for field in (
        "operation_timeout_seconds",
        "command_timeout_seconds",
        "resource_wait_timeout_seconds",
        "cleanup_timeout_seconds",
    ):
        normalized[field] = _finite(request[field], field, positive=True)
    normalized["launch_deadline"] = _finite(
        request["launch_deadline"], "launch_deadline"
    )
    if "absolute_deadline" in request:
        normalized["absolute_deadline"] = _finite(
            request["absolute_deadline"], "absolute_deadline"
        )
    return normalized, root, target, experiment


def _screen_gate_refs(root: Path, experiment: dict) -> list[dict]:
    if not experiment["screen_design"]["enabled"]:
        return []
    experiment_ref = {
        "id": experiment["id"],
        "sha256": STORE.sha256_file(
            root / "experiments" / f"{experiment['id']}.json"
        ),
    }
    current_tool_identity = _tool_identity()
    records = {}
    invocation_root = root / "invocations"
    for invocation_dir in sorted(invocation_root.glob("inv-*")):
        request_path = invocation_dir / "request.json"
        result_path = invocation_dir / "result.json"
        if not request_path.is_file():
            continue
        try:
            frozen = json.loads(
                STORE.read_regular_bytes(request_path).decode("utf-8")
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            continue
        if (
            type(frozen) is not dict
            or frozen.get("operation") != "screen"
            or frozen.get("experiment_ref") != experiment_ref
            or frozen.get("variant_refs")
            != [experiment["reference_variant"], experiment["candidate"]]
            or frozen.get("reference_selection_ref")
            != experiment["reference_selection_ref"]
            or frozen.get("tool_identity") != current_tool_identity
        ):
            continue
        result = None
        result_payload = None
        if result_path.is_file():
            try:
                result_payload = STORE.read_regular_bytes(result_path)
                result = json.loads(result_payload.decode("utf-8"))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                result = None
                result_payload = None
        records[invocation_dir.name] = {
            "request": frozen,
            "result": result,
            "result_payload": result_payload,
        }
    if not records:
        raise EvaluatorError(
            "screen_required",
            "a valid non-rejected screen result is required",
        )
    semantic_digests = {
        record["request"].get("request_digest") for record in records.values()
    }
    if len(semantic_digests) != 1 or None in semantic_digests:
        raise EvaluatorError(
            "screen_attempt_changed",
            "screen sampling changed without a superseding Experiment",
        )

    rejected = []
    eligible = []
    for invocation_id, record in records.items():
        result = record["result"]
        payload = record["result_payload"]
        if (
            type(result) is not dict
            or payload is None
            or result.get("operation") != "screen"
            or result.get("experiment_ref") != experiment_ref
            or result.get("target_ref") != experiment["target_ref"]
            or result.get("execution_status") != "succeeded"
            or result.get("measurement_validity") != "valid"
            or result.get("cleanup_status") != "confirmed"
            or result.get("performance_receipt", {}).get("reference_status")
            != "current"
        ):
            continue
        reference = {
            "invocation_id": invocation_id,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if result.get("verdict") == "rejected":
            rejected.append(reference)
        elif result.get("verdict") == "inconclusive":
            eligible.append(reference)
    if rejected:
        raise EvaluatorError(
            "screen_rejected",
            "a valid screen result rejected this Experiment",
        )
    if len(eligible) != 1:
        raise EvaluatorError(
            "screen_required",
            "one valid non-rejected screen result is required",
        )
    terminal_id = eligible[0]["invocation_id"]
    retry_chain = set()
    cursor = terminal_id
    while cursor is not None:
        if cursor in retry_chain or cursor not in records:
            raise EvaluatorError(
                "screen_retry_invalid",
                "screen retry lineage is incomplete or cyclic",
            )
        retry_chain.add(cursor)
        retry_of = records[cursor]["request"].get("retry_of")
        if retry_of is not None and not isinstance(retry_of, str):
            raise EvaluatorError(
                "screen_retry_invalid",
                "screen retry lineage is invalid",
            )
        cursor = retry_of
    if retry_chain != set(records):
        raise EvaluatorError(
            "screen_attempt_changed",
            "screen attempts are not one explicit retry chain",
        )
    return eligible


def _frozen_formal_request(
    request: dict,
    target: dict,
    experiment: dict | None,
    screen_refs: list[dict],
) -> dict:
    if experiment is None:
        current_variant, current_selection = _current_reference(
            Path(request["artifact_root"]),
            target,
        )
        if current_selection is None or current_variant == target["original"]:
            raise EvaluatorError(
                "champion_required",
                "final_audit requires a selected non-original Champion",
            )
        variant_refs = [target["original"], current_variant]
        baseline_ref = None
        minimum_effect = target["minimum_effect"]
    else:
        current_variant, current_selection = _current_reference(
            Path(request["artifact_root"]),
            target,
        )
        if (
            current_variant != experiment["reference_variant"]
            or current_selection != experiment["reference_selection_ref"]
        ):
            raise EvaluatorError(
                "stale_reference",
                "experiment reference is no longer the current Champion",
            )
        variant_refs = [current_variant, experiment["candidate"]]
        baseline_ref = experiment["baseline_ref"]
        minimum_effect = experiment["minimum_effect"]
    reference_pointer_digest = _current_pointer_digest(
        Path(request["artifact_root"])
    )
    if (current_selection is None) != (reference_pointer_digest is None):
        raise EvaluatorError(
            "stale_reference",
            "Champion changed while the formal request was frozen",
        )
    frozen = {
        "operation": request["operation"],
        "target_ref": request["target_ref"],
        "variant_refs": variant_refs,
        "reference_selection_ref": current_selection,
        "reference_pointer_sha256": reference_pointer_digest,
        "driver": ADAPTER.verify_driver(target["driver"]),
        "objective": {
            "primary_metric": target["primary_metric"],
            "minimum_effect": minimum_effect,
            "constraints": target["constraints"],
        },
        "sampling_design": request["sampling_design"],
        "validity_requirements": target["validity_requirements"],
        "resources": request["resources"],
        "cleanup": target["driver"]["cleanup_contract"],
        "tool_identity": _tool_identity(),
        "operation_timeout_seconds": request["operation_timeout_seconds"],
        "command_timeout_seconds": request["command_timeout_seconds"],
        "resource_wait_timeout_seconds": request["resource_wait_timeout_seconds"],
        "cleanup_timeout_seconds": request["cleanup_timeout_seconds"],
        "launch_deadline": request["launch_deadline"],
    }
    if experiment is not None:
        frozen["experiment_ref"] = request["experiment_ref"]
        frozen["baseline_ref"] = baseline_ref
        frozen["screen_result_refs"] = screen_refs
    for field in ("absolute_deadline", "retry_of"):
        if field in request:
            frozen[field] = request[field]
    frozen["request_digest"] = RUNTIME.request_digest(frozen)
    return frozen


def target(value, *, wait_for_result: bool) -> dict:
    request, root, target_record, experiment = _comparison_input(
        value,
        operation="target",
    )
    assert experiment is not None
    screen_refs = _screen_gate_refs(root, experiment)
    frozen = _frozen_formal_request(
        request,
        target_record,
        experiment,
        screen_refs,
    )
    return RUNTIME.submit(
        root,
        frozen,
        [sys.executable, str(Path(__file__).resolve()), "_worker"],
        wait_for_result,
    )


def final_audit(value, *, wait_for_result: bool) -> dict:
    request, root, target_record, _ = _comparison_input(
        value,
        operation="final_audit",
    )
    frozen = _frozen_formal_request(request, target_record, None, [])
    return RUNTIME.submit(
        root,
        frozen,
        [sys.executable, str(Path(__file__).resolve()), "_worker"],
        wait_for_result,
    )


def invocation_status(value) -> dict:
    request = _closed(
        value,
        _LIFECYCLE_FIELDS,
        set(),
        "status input",
    )
    if request["format_version"] != INPUT_VERSION or request["operation"] != "status":
        raise EvaluatorError(
            "invalid_evaluator_input",
            "status input version or operation is unsupported",
        )
    return RUNTIME.status(request["artifact_root"], request["invocation_id"])


def cancel_invocation(value) -> dict:
    request = _closed(
        value,
        _LIFECYCLE_FIELDS,
        set(),
        "cancel input",
    )
    if request["format_version"] != INPUT_VERSION or request["operation"] != "cancel":
        raise EvaluatorError(
            "invalid_evaluator_input",
            "cancel input version or operation is unsupported",
        )
    return RUNTIME.cancel(request["artifact_root"], request["invocation_id"])


def _driver_call(
    *,
    invocation_dir: Path,
    request: dict,
    target: dict,
    driver_inputs: dict,
    variant: dict,
    role: str,
    mode: str,
    case_id: str,
    index: int,
) -> tuple[dict | None, dict, dict]:
    call_dir = invocation_dir / "driver" / f"{index:04d}-{mode}-{case_id}"
    call_dir.mkdir(parents=True)
    output_path = call_dir / "result.json"
    driver_request = ADAPTER.build_driver_request(
        target_id=request["target_ref"]["id"],
        execution_id=invocation_dir.name,
        operation=request["operation"],
        driver=request["driver"],
        variant=variant,
        test_suite=driver_inputs["test_suite"],
        correctness=driver_inputs["correctness"],
        objective=driver_inputs["objective"],
        role=role,
        mode=mode,
        case={"id": case_id},
        sampling=request["sampling_design"],
        output_path=output_path,
    )
    request_path = call_dir / "request.json"
    STORE.create_regular_json(request_path, driver_request)
    command_result = RUNTIME.run_child(
        {
            "argv": ADAPTER.build_argv(request["driver"], request_path),
            "cwd": str(call_dir),
            "env": {},
            "output_limit_bytes": 64 * 1024,
            "required_gpu_uuids": request["resources"]["gpu_uuids"],
        }
    )
    if command_result["status"] != "completed":
        return None, command_result, {
            "request": driver_request,
            "command_result": command_result,
        }
    try:
        driver_result = ADAPTER.validate_driver_result(
            output_path,
            driver_request,
        )
    except (OSError, ValueError) as error:
        invalid = {
            **command_result,
            "status": "failed",
            "stop_reason": "driver_result_invalid",
            "stderr": str(error)[:1024],
        }
        return None, invalid, {
            "request": driver_request,
            "command_result": invalid,
        }
    if driver_result["environment"] != target["environment"]["runtime"]:
        invalid = {
            **command_result,
            "status": "failed",
            "stop_reason": "environment_identity_changed",
            "stderr": "driver environment does not match the frozen Target",
        }
        return None, invalid, {
            "request": driver_request,
            "command_result": invalid,
        }
    evidence_ref = STORE.freeze_path(
        os.environ["CKO_ARTIFACT_ROOT"],
        output_path,
        _OBJECT_SCAN_LIMITS,
    )
    return driver_result, command_result, {
        "request": driver_request,
        "command_result": command_result,
        "driver_result_ref": evidence_ref,
    }


def _base_result(request: dict, started_epoch: float) -> dict:
    return {
        "record_type": "invocation_result",
        "format_version": RESULT_VERSION,
        "operation": request["operation"],
        "target_ref": request["target_ref"],
        "variant_refs": request["variant_refs"],
        "started_at_epoch": started_epoch,
        "finished_at_epoch": None,
        "elapsed_seconds": None,
        "execution_status": None,
        "measurement_validity": "invalid",
        "verdict": "not_evaluated",
        "stop_reason": None,
        "cleanup_status": "confirmed",
        "correctness_receipts": [],
        "performance_receipt": {
            "status": "not_run",
            "acceptance": request["objective"]["primary_metric"],
            "evidence_refs": [],
        },
        "skipped_expensive_stages": [],
        "command_receipts": [],
    }


def _finish(result: dict, *, started_mono: float) -> dict:
    result["finished_at_epoch"] = time.time()
    result["elapsed_seconds"] = time.monotonic() - started_mono
    return result


def _command_failure(
    result: dict,
    command_result: dict,
    *,
    skipped: list[str],
    started_mono: float,
) -> dict:
    status = command_result["status"]
    result["execution_status"] = (
        "timed_out"
        if status == "timed_out"
        else "cancelled"
        if status == "cancelled"
        else "failed"
    )
    result["stop_reason"] = command_result["stop_reason"]
    result["cleanup_status"] = command_result["cleanup_status"]
    result["skipped_expensive_stages"] = skipped
    return _finish(result, started_mono=started_mono)


def _run_baseline_worker(
    *,
    artifact_root: Path,
    invocation_dir: Path,
    request: dict,
) -> dict:
    started_epoch = time.time()
    started_mono = time.monotonic()
    result = _base_result(request, started_epoch)
    target = _load_target(artifact_root, request["target_ref"])
    driver = ADAPTER.verify_driver(request["driver"])
    original = target["original"]
    workspace = invocation_dir / "workspace"
    workspace.mkdir()
    driver_inputs = ADAPTER.materialize_target_inputs(
        artifact_root, workspace, target
    )
    variant = ADAPTER.materialize_variant(
        artifact_root, workspace, original, "original"
    )
    cases = request["sampling_design"]["case_ids"]
    call_index = 0
    measurements = []
    correctness_receipts = []

    if driver["execution_mode"] == "combined":
        modes = ("combined",)
    else:
        modes = ("correctness", "measure")
    if modes == ("correctness", "measure"):
        for case_id in cases:
            driver_result, command_result, receipt = _driver_call(
                invocation_dir=invocation_dir,
                request=request,
                target=target,
                driver_inputs=driver_inputs,
                variant=variant,
                role="original",
                mode="correctness",
                case_id=case_id,
                index=call_index,
            )
            call_index += 1
            result["command_receipts"].append(receipt)
            if driver_result is None:
                return _command_failure(
                    result,
                    command_result,
                    skipped=["measure"],
                    started_mono=started_mono,
                )
            correctness = driver_result["correctness"]
            correctness_gate = ADAPTER.evaluate_correctness(
                correctness,
                target["correctness"]["acceptance"],
            )
            correctness_receipts.append(
                {
                    "variant": original,
                    "case_id": case_id,
                    "status": "valid",
                    "passed": correctness_gate["passed"],
                    "acceptance": target["correctness"]["acceptance"],
                    "metrics": correctness["metrics"],
                    "gate": correctness_gate,
                    "evidence_refs": [receipt["driver_result_ref"]],
                }
            )
            if not correctness_gate["passed"]:
                result["correctness_receipts"] = correctness_receipts
                result["execution_status"] = "succeeded"
                result["measurement_validity"] = "invalid"
                result["verdict"] = "failed"
                result["stop_reason"] = "correctness_failed"
                result["skipped_expensive_stages"] = ["measure"]
                return _finish(result, started_mono=started_mono)

        for case_id in cases:
            driver_result, command_result, receipt = _driver_call(
                invocation_dir=invocation_dir,
                request=request,
                target=target,
                driver_inputs=driver_inputs,
                variant=variant,
                role="original",
                mode="measure",
                case_id=case_id,
                index=call_index,
            )
            call_index += 1
            result["command_receipts"].append(receipt)
            if driver_result is None:
                result["correctness_receipts"] = correctness_receipts
                return _command_failure(
                    result,
                    command_result,
                    skipped=[],
                    started_mono=started_mono,
                )
            measurements.append(
                {
                    "case_id": case_id,
                    **driver_result["measurements"],
                    "evidence_ref": receipt["driver_result_ref"],
                }
            )
    else:
        for case_id in cases:
            driver_result, command_result, receipt = _driver_call(
                invocation_dir=invocation_dir,
                request=request,
                target=target,
                driver_inputs=driver_inputs,
                variant=variant,
                role="original",
                mode="combined",
                case_id=case_id,
                index=call_index,
            )
            call_index += 1
            result["command_receipts"].append(receipt)
            if driver_result is None:
                return _command_failure(
                    result,
                    command_result,
                    skipped=[],
                    started_mono=started_mono,
                )
            correctness = driver_result["correctness"]
            correctness_gate = ADAPTER.evaluate_correctness(
                correctness,
                target["correctness"]["acceptance"],
            )
            correctness_receipts.append(
                {
                    "variant": original,
                    "case_id": case_id,
                    "status": "valid",
                    "passed": correctness_gate["passed"],
                    "acceptance": target["correctness"]["acceptance"],
                    "metrics": correctness["metrics"],
                    "gate": correctness_gate,
                    "evidence_refs": [receipt["driver_result_ref"]],
                }
            )
            if not correctness_gate["passed"]:
                result["correctness_receipts"] = correctness_receipts
                result["execution_status"] = "succeeded"
                result["measurement_validity"] = "invalid"
                result["verdict"] = "failed"
                result["stop_reason"] = "correctness_failed"
                return _finish(result, started_mono=started_mono)
            measurements.append(
                {
                    "case_id": case_id,
                    **driver_result["measurements"],
                    "evidence_ref": receipt["driver_result_ref"],
                }
            )

    primary = target["primary_metric"]
    for measurement in measurements:
        try:
            _measurement_values({"measurements": measurement}, target)
        except EvaluatorError:
            result["correctness_receipts"] = correctness_receipts
            result["execution_status"] = "invalid"
            result["stop_reason"] = "measurement_contract_mismatch"
            return _finish(result, started_mono=started_mono)
    result["correctness_receipts"] = correctness_receipts
    result["performance_receipt"] = {
        "status": "valid",
        "acceptance": primary,
        "measurements": measurements,
        "evidence_refs": [
            measurement["evidence_ref"] for measurement in measurements
        ],
    }
    result["execution_status"] = "succeeded"
    result["measurement_validity"] = "valid"
    result["verdict"] = "passed"
    result["stop_reason"] = "completed"
    return _finish(result, started_mono=started_mono)


def _current_reference_matches(
    artifact_root: Path,
    target: dict,
    request: dict,
) -> bool:
    variant, selection = _current_reference(artifact_root, target)
    return (
        variant == request["variant_refs"][0]
        and selection == request["reference_selection_ref"]
    )


def _formal_binding_matches(
    artifact_root: Path,
    target: dict,
    request: dict,
) -> bool:
    variant, selection = _current_reference(artifact_root, target)
    expected_variant = (
        request["variant_refs"][1]
        if request["operation"] == "final_audit"
        else request["variant_refs"][0]
    )
    return (
        variant == expected_variant
        and selection == request["reference_selection_ref"]
    )


def _measurement_values(driver_result: dict, target: dict) -> dict:
    measurements = driver_result["measurements"]
    primary = target["primary_metric"]
    observed = measurements["primary"]
    if observed["name"] != primary["name"] or observed["unit"] != primary["unit"]:
        raise EvaluatorError(
            "primary_metric_mismatch",
            "driver primary measurement does not match the Target",
        )
    expected_constraints = {
        constraint["name"]: constraint for constraint in target["constraints"]
    }
    observed_constraints = {
        constraint["name"]: constraint
        for constraint in measurements["constraints"]
    }
    if set(expected_constraints) != set(observed_constraints):
        raise EvaluatorError(
            "constraint_metric_mismatch",
            "driver constraint measurements do not match the Target",
        )
    values = {
        "primary": statistics.median(observed["samples"]),
        "constraints": {},
    }
    for name, contract in expected_constraints.items():
        measurement = observed_constraints[name]
        if measurement["unit"] != contract["unit"]:
            raise EvaluatorError(
                "constraint_unit_mismatch",
                f"driver constraint unit does not match the Target: {name}",
            )
        values["constraints"][name] = statistics.median(measurement["samples"])
    return values


def _comparison_statistics(
    pairs: list[dict],
    *,
    target: dict,
    objective: dict,
    validity_requirements: dict,
    seed: int,
) -> tuple[dict, list[dict]]:
    primary = objective["primary_metric"]
    primary_statistics = PAIRED.classify_pairs(
        [
            {
                "baseline": pair["reference"],
                "candidate": pair["candidate"],
                "valid": pair["valid"],
            }
            for pair in pairs
        ],
        direction=primary["direction"],
        min_effect_pct=objective["minimum_effect"]["value"],
        confidence=validity_requirements["confidence"],
        bootstrap_samples=validity_requirements["bootstrap_samples"],
        min_valid_pairs=validity_requirements["minimum_pairs"],
        seed=seed,
    )
    constraint_statistics = []
    for index, constraint in enumerate(target["constraints"]):
        regressions = []
        for pair in pairs:
            reference_value = pair["constraints"][constraint["name"]]["reference"]
            candidate_value = pair["constraints"][constraint["name"]]["candidate"]
            if reference_value == 0:
                raise EvaluatorError(
                    "constraint_reference_zero",
                    f"constraint reference is zero: {constraint['name']}",
                )
            if constraint["direction"] == "lower":
                regression = (
                    candidate_value - reference_value
                ) / abs(reference_value) * 100.0
            else:
                regression = (
                    reference_value - candidate_value
                ) / abs(reference_value) * 100.0
            regressions.append(regression)
        estimate = statistics.median(regressions)
        ci_low, ci_high = PAIRED.bootstrap_median_ci(
            regressions,
            confidence=validity_requirements["confidence"],
            samples=validity_requirements["bootstrap_samples"],
            seed=seed + index + 1,
        )
        if len(regressions) < validity_requirements["minimum_pairs"]:
            status = "inconclusive"
        elif ci_high <= constraint["max_regression_pct"]:
            status = "passed"
        elif ci_low > constraint["max_regression_pct"]:
            status = "failed"
        else:
            status = "inconclusive"
        constraint_statistics.append(
            {
                "name": constraint["name"],
                "max_regression_pct": constraint["max_regression_pct"],
                "estimate_pct": estimate,
                "ci_low_pct": ci_low,
                "ci_high_pct": ci_high,
                "status": status,
                "values_pct": regressions,
            }
        )
    return primary_statistics, constraint_statistics


def _run_screen_worker(
    *,
    artifact_root: Path,
    invocation_dir: Path,
    request: dict,
) -> dict:
    started_epoch = time.time()
    started_mono = time.monotonic()
    result = _base_result(request, started_epoch)
    result["experiment_ref"] = request["experiment_ref"]
    target = _load_target(artifact_root, request["target_ref"])
    driver = ADAPTER.verify_driver(request["driver"])
    reference_frozen, candidate_frozen = request["variant_refs"]
    workspace = invocation_dir / "workspace"
    workspace.mkdir()
    driver_inputs = ADAPTER.materialize_target_inputs(
        artifact_root, workspace, target
    )
    reference = ADAPTER.materialize_variant(
        artifact_root, workspace, reference_frozen, "reference"
    )
    candidate = ADAPTER.materialize_variant(
        artifact_root, workspace, candidate_frozen, "candidate"
    )
    if not _current_reference_matches(artifact_root, target, request):
        result["execution_status"] = "invalid"
        result["stop_reason"] = "stale_reference"
        result["skipped_expensive_stages"] = ["correctness", "paired_measurement"]
        return _finish(result, started_mono=started_mono)

    call_index = 0
    correctness_receipts = []
    cases = request["sampling_design"]["case_ids"]
    if driver["execution_mode"] == "separate":
        for case_id in cases:
            driver_result, command_result, receipt = _driver_call(
                invocation_dir=invocation_dir,
                request=request,
                target=target,
                driver_inputs=driver_inputs,
                variant=candidate,
                role="candidate",
                mode="correctness",
                case_id=case_id,
                index=call_index,
            )
            call_index += 1
            result["command_receipts"].append(receipt)
            if driver_result is None:
                return _command_failure(
                    result,
                    command_result,
                    skipped=["paired_measurement"],
                    started_mono=started_mono,
                )
            correctness = driver_result["correctness"]
            correctness_gate = ADAPTER.evaluate_correctness(
                correctness,
                target["correctness"]["acceptance"],
            )
            correctness_receipts.append(
                {
                    "variant": candidate_frozen,
                    "case_id": case_id,
                    "status": "valid",
                    "passed": correctness_gate["passed"],
                    "acceptance": target["correctness"]["acceptance"],
                    "metrics": correctness["metrics"],
                    "gate": correctness_gate,
                    "evidence_refs": [receipt["driver_result_ref"]],
                }
            )
            if not correctness_gate["passed"]:
                result["correctness_receipts"] = correctness_receipts
                result["execution_status"] = "succeeded"
                result["measurement_validity"] = "valid"
                result["verdict"] = "rejected"
                result["stop_reason"] = "correctness_failed"
                result["skipped_expensive_stages"] = ["paired_measurement"]
                return _finish(result, started_mono=started_mono)
        if not _current_reference_matches(artifact_root, target, request):
            result["correctness_receipts"] = correctness_receipts
            result["execution_status"] = "succeeded"
            result["stop_reason"] = "stale_reference"
            result["skipped_expensive_stages"] = ["paired_measurement"]
            return _finish(result, started_mono=started_mono)

    pairs = []
    for pair_index, order in enumerate(request["sampling_design"]["orders"]):
        for case_id in cases:
            values = {}
            pair_evidence = []
            roles = (
                (("reference", reference), ("candidate", candidate))
                if order == "AB"
                else (("candidate", candidate), ("reference", reference))
            )
            for role, variant in roles:
                mode = (
                    "measure"
                    if driver["execution_mode"] == "separate"
                    else "combined"
                )
                driver_result, command_result, receipt = _driver_call(
                    invocation_dir=invocation_dir,
                    request=request,
                    target=target,
                    driver_inputs=driver_inputs,
                    variant=variant,
                    role=role,
                    mode=mode,
                    case_id=case_id,
                    index=call_index,
                )
                call_index += 1
                result["command_receipts"].append(receipt)
                if driver_result is None:
                    result["correctness_receipts"] = correctness_receipts
                    return _command_failure(
                        result,
                        command_result,
                        skipped=[],
                        started_mono=started_mono,
                    )
                if driver["execution_mode"] == "combined":
                    correctness = driver_result["correctness"]
                    correctness_gate = ADAPTER.evaluate_correctness(
                        correctness,
                        target["correctness"]["acceptance"],
                    )
                    correctness_receipts.append(
                        {
                            "variant": (
                                reference_frozen
                                if role == "reference"
                                else candidate_frozen
                            ),
                            "case_id": case_id,
                            "status": "valid",
                            "passed": correctness_gate["passed"],
                            "acceptance": target["correctness"]["acceptance"],
                            "metrics": correctness["metrics"],
                            "gate": correctness_gate,
                            "evidence_refs": [receipt["driver_result_ref"]],
                        }
                    )
                    if not correctness_gate["passed"]:
                        result["correctness_receipts"] = correctness_receipts
                        result["execution_status"] = "succeeded"
                        result["measurement_validity"] = "valid"
                        result["verdict"] = "rejected"
                        result["stop_reason"] = "correctness_failed"
                        return _finish(result, started_mono=started_mono)
                values[role] = _measurement_values(driver_result, target)
                pair_evidence.append(receipt["driver_result_ref"])
            pairs.append(
                {
                    "pair_index": pair_index,
                    "case_id": case_id,
                    "order": order,
                    "reference": values["reference"]["primary"],
                    "candidate": values["candidate"]["primary"],
                    "constraints": {
                        name: {
                            "reference": values["reference"]["constraints"][name],
                            "candidate": values["candidate"]["constraints"][name],
                        }
                        for name in values["reference"]["constraints"]
                    },
                    "valid": True,
                    "evidence_refs": pair_evidence,
                }
            )

    statistics_result, constraint_statistics = _comparison_statistics(
        pairs,
        target=target,
        objective=request["objective"],
        validity_requirements=request["validity_requirements"],
        seed=request["sampling_design"]["seed"],
    )
    result["correctness_receipts"] = correctness_receipts
    result["performance_receipt"] = {
        "status": "valid",
        "reference": reference_frozen,
        "candidate": candidate_frozen,
        "reference_status": (
            "current"
            if _current_reference_matches(artifact_root, target, request)
            else "stale_reference"
        ),
        "acceptance": request["objective"]["minimum_effect"],
        "pairs": pairs,
        "statistics": statistics_result,
        "constraint_statistics": constraint_statistics,
        "evidence_refs": [
            evidence
            for pair in pairs
            for evidence in pair["evidence_refs"]
        ],
    }
    if result["performance_receipt"]["reference_status"] != "current":
        result["execution_status"] = "succeeded"
        result["measurement_validity"] = "invalid"
        result["verdict"] = "not_evaluated"
        result["stop_reason"] = "stale_reference"
    else:
        result["execution_status"] = "succeeded"
        result["measurement_validity"] = "valid"
        result["screen_claim_status"] = (
            "confirmed"
            if statistics_result["status"] == "confirmed_win"
            else "falsified"
            if statistics_result["status"] == "confirmed_loss"
            else "inconclusive"
        )
        reject_kinds = {condition["kind"] for condition in request["reject_if"]}
        if any(item["status"] == "failed" for item in constraint_statistics):
            result["verdict"] = "rejected"
            result["stop_reason"] = "constraint_failed"
        elif (
            result["screen_claim_status"] == "falsified"
            and "screen_claim_falsified" in reject_kinds
        ):
            result["verdict"] = "rejected"
            result["stop_reason"] = "screen_claim_falsified"
        elif (
            request["screen_design"]["kind"] == "conservative_bound"
            and statistics_result["ci_high_pct"] is not None
            and statistics_result["ci_high_pct"]
            < request["objective"]["minimum_effect"]["value"]
        ):
            result["verdict"] = "rejected"
            result["stop_reason"] = "screen_upper_bound_below_minimum"
        else:
            result["verdict"] = "inconclusive"
            result["stop_reason"] = "completed"
    return _finish(result, started_mono=started_mono)


def _correctness_receipt(
    *,
    frozen_variant: dict,
    case_id: str,
    target: dict,
    driver_result: dict,
    evidence_ref: dict,
) -> dict:
    correctness = driver_result["correctness"]
    gate = ADAPTER.evaluate_correctness(
        correctness,
        target["correctness"]["acceptance"],
    )
    return {
        "variant": frozen_variant,
        "case_id": case_id,
        "status": "valid",
        "passed": gate["passed"],
        "acceptance": target["correctness"]["acceptance"],
        "metrics": correctness["metrics"],
        "gate": gate,
        "evidence_refs": [evidence_ref],
    }


def _finish_stale_formal(
    result: dict,
    *,
    started_mono: float,
    commands_started: bool,
    skipped: list[str],
) -> dict:
    _mark_stale_formal_result(result, skipped=skipped)
    result["execution_status"] = "succeeded" if commands_started else "invalid"
    return _finish(result, started_mono=started_mono)


def _mark_stale_formal_result(
    result: dict,
    *,
    skipped: list[str],
) -> None:
    result["measurement_validity"] = "invalid"
    result["verdict"] = "not_evaluated"
    result["stop_reason"] = "stale_reference"
    result["reference_status"] = "stale_reference"
    result["skipped_expensive_stages"] = skipped
    performance = result.get("performance_receipt")
    if type(performance) is dict and "reference_status" in performance:
        performance["reference_status"] = "stale_reference"
    if result["operation"] == "final_audit":
        result["restore_supported"] = False


def _run_formal_worker(
    *,
    artifact_root: Path,
    invocation_dir: Path,
    request: dict,
) -> dict:
    started_epoch = time.time()
    started_mono = time.monotonic()
    result = _base_result(request, started_epoch)
    result["reference_selection_ref"] = request["reference_selection_ref"]
    if request["operation"] == "target":
        result["experiment_ref"] = request["experiment_ref"]
        result["screen_result_refs"] = request["screen_result_refs"]
    else:
        result["restore_supported"] = False
    target = _load_target(artifact_root, request["target_ref"])
    driver = ADAPTER.verify_driver(request["driver"])
    reference_frozen, candidate_frozen = request["variant_refs"]

    resource_result = RUNTIME.acquire_resources(
        request["resources"]["gpu_uuids"]
    )
    if resource_result["status"] != "acquired":
        return _command_failure(
            result,
            resource_result,
            skipped=["correctness", "paired_measurement"],
            started_mono=started_mono,
        )
    if not _formal_binding_matches(artifact_root, target, request):
        return _finish_stale_formal(
            result,
            started_mono=started_mono,
            commands_started=False,
            skipped=["correctness", "paired_measurement"],
        )
    result["reference_status"] = "current"

    workspace = invocation_dir / "workspace"
    workspace.mkdir()
    driver_inputs = ADAPTER.materialize_target_inputs(
        artifact_root, workspace, target
    )
    reference = ADAPTER.materialize_variant(
        artifact_root, workspace, reference_frozen, "reference"
    )
    candidate = ADAPTER.materialize_variant(
        artifact_root, workspace, candidate_frozen, "candidate"
    )
    if request["operation"] == "final_audit":
        correctness_plan = (
            ("original", reference_frozen, reference),
            ("candidate", candidate_frozen, candidate),
        )
    else:
        correctness_plan = (
            ("candidate", candidate_frozen, candidate),
            ("reference", reference_frozen, reference),
        )
    correctness_receipts = []
    call_index = 0
    commands_started = False
    correctness_mode = (
        "correctness"
        if driver["execution_mode"] == "separate"
        else "combined"
    )
    cases = request["sampling_design"]["case_ids"]
    for role, frozen_variant, materialized_variant in correctness_plan:
        for case_id in cases:
            if not _formal_binding_matches(artifact_root, target, request):
                result["correctness_receipts"] = correctness_receipts
                return _finish_stale_formal(
                    result,
                    started_mono=started_mono,
                    commands_started=commands_started,
                    skipped=["paired_measurement"],
                )
            driver_result, command_result, receipt = _driver_call(
                invocation_dir=invocation_dir,
                request=request,
                target=target,
                driver_inputs=driver_inputs,
                variant=materialized_variant,
                role=role,
                mode=correctness_mode,
                case_id=case_id,
                index=call_index,
            )
            commands_started = True
            call_index += 1
            result["command_receipts"].append(receipt)
            if driver_result is None:
                result["correctness_receipts"] = correctness_receipts
                return _command_failure(
                    result,
                    command_result,
                    skipped=["paired_measurement"],
                    started_mono=started_mono,
                )
            correctness = _correctness_receipt(
                frozen_variant=frozen_variant,
                case_id=case_id,
                target=target,
                driver_result=driver_result,
                evidence_ref=receipt["driver_result_ref"],
            )
            correctness_receipts.append(correctness)
            if not correctness["passed"]:
                result["correctness_receipts"] = correctness_receipts
                if not _formal_binding_matches(
                    artifact_root,
                    target,
                    request,
                ):
                    return _finish_stale_formal(
                        result,
                        started_mono=started_mono,
                        commands_started=True,
                        skipped=["paired_measurement"],
                    )
                result["execution_status"] = "succeeded"
                result["measurement_validity"] = "valid"
                result["verdict"] = "rejected"
                result["stop_reason"] = "correctness_failed"
                result["skipped_expensive_stages"] = ["paired_measurement"]
                if request["operation"] == "final_audit":
                    original_passed = all(
                        any(
                            item["variant"] == reference_frozen
                            and item["case_id"] == expected_case
                            and item["passed"]
                            for item in correctness_receipts
                        )
                        for expected_case in cases
                    )
                    result["restore_supported"] = (
                        frozen_variant == candidate_frozen and original_passed
                    )
                return _finish(result, started_mono=started_mono)

    if not _formal_binding_matches(artifact_root, target, request):
        result["correctness_receipts"] = correctness_receipts
        return _finish_stale_formal(
            result,
            started_mono=started_mono,
            commands_started=True,
            skipped=["paired_measurement"],
        )

    pairs = []
    for pair_index, order in enumerate(request["sampling_design"]["orders"]):
        for case_id in cases:
            values = {}
            pair_evidence = []
            roles = (
                (("reference", reference), ("candidate", candidate))
                if order == "AB"
                else (("candidate", candidate), ("reference", reference))
            )
            for role, variant in roles:
                if not _formal_binding_matches(artifact_root, target, request):
                    result["correctness_receipts"] = correctness_receipts
                    return _finish_stale_formal(
                        result,
                        started_mono=started_mono,
                        commands_started=True,
                        skipped=["remaining_paired_measurement"],
                    )
                mode = (
                    "measure"
                    if driver["execution_mode"] == "separate"
                    else "combined"
                )
                driver_result, command_result, receipt = _driver_call(
                    invocation_dir=invocation_dir,
                    request=request,
                    target=target,
                    driver_inputs=driver_inputs,
                    variant=variant,
                    role=role,
                    mode=mode,
                    case_id=case_id,
                    index=call_index,
                )
                call_index += 1
                result["command_receipts"].append(receipt)
                if driver_result is None:
                    result["correctness_receipts"] = correctness_receipts
                    return _command_failure(
                        result,
                        command_result,
                        skipped=[],
                        started_mono=started_mono,
                    )
                if driver["execution_mode"] == "combined":
                    frozen_variant = (
                        reference_frozen
                        if role == "reference"
                        else candidate_frozen
                    )
                    correctness = _correctness_receipt(
                        frozen_variant=frozen_variant,
                        case_id=case_id,
                        target=target,
                        driver_result=driver_result,
                        evidence_ref=receipt["driver_result_ref"],
                    )
                    correctness_receipts.append(correctness)
                    if not correctness["passed"]:
                        result["correctness_receipts"] = correctness_receipts
                        if not _formal_binding_matches(
                            artifact_root,
                            target,
                            request,
                        ):
                            return _finish_stale_formal(
                                result,
                                started_mono=started_mono,
                                commands_started=True,
                                skipped=["remaining_paired_measurement"],
                            )
                        result["execution_status"] = "succeeded"
                        result["measurement_validity"] = "valid"
                        result["verdict"] = "rejected"
                        result["stop_reason"] = "correctness_failed"
                        if request["operation"] == "final_audit":
                            result["restore_supported"] = (
                                frozen_variant == candidate_frozen
                            )
                        return _finish(result, started_mono=started_mono)
                values[role] = _measurement_values(driver_result, target)
                pair_evidence.append(receipt["driver_result_ref"])
            pairs.append(
                {
                    "pair_index": pair_index,
                    "case_id": case_id,
                    "order": order,
                    "reference": values["reference"]["primary"],
                    "candidate": values["candidate"]["primary"],
                    "constraints": {
                        name: {
                            "reference": values["reference"]["constraints"][name],
                            "candidate": values["candidate"]["constraints"][name],
                        }
                        for name in values["reference"]["constraints"]
                    },
                    "valid": True,
                    "evidence_refs": pair_evidence,
                }
            )

    primary_statistics, constraint_statistics = _comparison_statistics(
        pairs,
        target=target,
        objective=request["objective"],
        validity_requirements=request["validity_requirements"],
        seed=request["sampling_design"]["seed"],
    )
    reference_status = (
        "current"
        if _formal_binding_matches(artifact_root, target, request)
        else "stale_reference"
    )
    result["correctness_receipts"] = correctness_receipts
    result["performance_receipt"] = {
        "status": "valid",
        "reference": reference_frozen,
        "candidate": candidate_frozen,
        "reference_status": reference_status,
        "acceptance": request["objective"]["minimum_effect"],
        "pairs": pairs,
        "statistics": primary_statistics,
        "constraint_statistics": constraint_statistics,
        "evidence_refs": [
            evidence
            for pair in pairs
            for evidence in pair["evidence_refs"]
        ],
    }
    if reference_status != "current":
        return _finish_stale_formal(
            result,
            started_mono=started_mono,
            commands_started=True,
            skipped=[],
        )

    constraint_failed = any(
        item["status"] == "failed" for item in constraint_statistics
    )
    constraint_inconclusive = any(
        item["status"] == "inconclusive" for item in constraint_statistics
    )
    result["execution_status"] = "succeeded"
    result["measurement_validity"] = "valid"
    if (
        primary_statistics["status"] == "confirmed_win"
        and not constraint_failed
        and not constraint_inconclusive
    ):
        result["verdict"] = "passed"
        result["stop_reason"] = "completed"
    elif (
        primary_statistics["status"] == "confirmed_loss"
        or (
            primary_statistics["ci_high_pct"] is not None
            and primary_statistics["ci_high_pct"]
            < request["objective"]["minimum_effect"]["value"]
        )
        or constraint_failed
    ):
        result["verdict"] = "rejected"
        result["stop_reason"] = (
            "constraint_failed"
            if constraint_failed
            else "minimum_effect_not_met"
        )
    else:
        result["verdict"] = "inconclusive"
        result["stop_reason"] = "completed"
    if request["operation"] == "final_audit":
        result["restore_supported"] = result["verdict"] == "rejected"
    return _finish(result, started_mono=started_mono)


def _worker_main() -> int:
    artifact_root = Path(os.environ["CKO_ARTIFACT_ROOT"])
    invocation_dir = Path(os.environ["CKO_INVOCATION_DIR"])
    request = _strict_json(invocation_dir / "request.json")
    started_mono = time.monotonic()
    try:
        if request.get("tool_identity") != _tool_identity():
            result = _base_result(request, time.time())
            result["execution_status"] = "invalid"
            result["stop_reason"] = "tool_identity_changed"
            result["skipped_expensive_stages"] = [
                "correctness",
                "paired_measurement",
            ]
            result = _finish(result, started_mono=started_mono)
        elif request.get("operation") == "baseline":
            result = _run_baseline_worker(
                artifact_root=artifact_root,
                invocation_dir=invocation_dir,
                request=request,
            )
        elif request.get("operation") == "screen":
            result = _run_screen_worker(
                artifact_root=artifact_root,
                invocation_dir=invocation_dir,
                request=request,
            )
        elif request.get("operation") in {"target", "final_audit"}:
            result = _run_formal_worker(
                artifact_root=artifact_root,
                invocation_dir=invocation_dir,
                request=request,
            )
        else:
            raise EvaluatorError(
                "unsupported_worker_operation",
                "worker operation is unsupported",
            )
    except BaseException as error:
        result = {
            "record_type": "invocation_result",
            "format_version": RESULT_VERSION,
            "operation": request.get("operation"),
            "target_ref": request.get("target_ref"),
            "variant_refs": request.get("variant_refs", []),
            "started_at_epoch": time.time() - (time.monotonic() - started_mono),
            "finished_at_epoch": time.time(),
            "elapsed_seconds": time.monotonic() - started_mono,
            "execution_status": "failed",
            "measurement_validity": "invalid",
            "verdict": "not_evaluated",
            "stop_reason": "worker_error",
            "cleanup_status": RUNTIME.current_cleanup_status(),
            "correctness_receipts": [],
            "performance_receipt": {
                "status": "not_run",
                "evidence_refs": [],
            },
            "skipped_expensive_stages": [],
            "command_receipts": [],
            "diagnostic": {
                "error_type": type(error).__name__[:128],
                "message": str(error)[:1024],
            },
        }
    if request.get("tool_identity") != _tool_identity():
        result["execution_status"] = "invalid"
        result["measurement_validity"] = "invalid"
        result["verdict"] = "not_evaluated"
        result["stop_reason"] = "tool_identity_changed"
        result["cleanup_status"] = RUNTIME.current_cleanup_status()
    if (
        request.get("operation") in {"target", "final_audit"}
        and result.get("reference_status") == "current"
    ):
        try:
            STORE.create_regular_json_if_ref_digest(
                artifact_root,
                "champion/current.json",
                request["reference_pointer_sha256"],
                f"invocations/{invocation_dir.name}/result.json",
                result,
            )
            return 0
        except STORE.StaleReferenceError:
            _mark_stale_formal_result(
                result,
                skipped=result.get("skipped_expensive_stages", []),
            )
    STORE.create_regular_json(invocation_dir / "result.json", result)
    return 0


def _emit_error(error: BaseException) -> int:
    code = error.code if isinstance(error, EvaluatorError) else "evaluator_error"
    print(
        json.dumps(
            {
                "status": "rejected",
                "error_code": code,
                "error": str(error)[:1024],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["_worker"]:
        return _worker_main()
    parser = argparse.ArgumentParser(
        description="Run one explicit V1.4 workload evaluation operation."
    )
    parser.add_argument(
        "operation",
        choices=(
            "baseline",
            "experiment",
            "screen",
            "target",
            "final_audit",
            "status",
            "cancel",
        ),
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(arguments)
    try:
        request = _strict_json(args.request)
        if request.get("operation") != args.operation:
            raise EvaluatorError(
                "invalid_evaluator_input",
                "CLI operation does not match request",
            )
        if args.operation == "baseline":
            result = baseline(request, wait_for_result=args.wait)
        elif args.operation == "experiment":
            if args.wait:
                raise EvaluatorError(
                    "invalid_evaluator_input",
                    "experiment is synchronous and does not accept --wait",
                )
            result = create_experiment(request)
        elif args.operation == "screen":
            result = screen(request, wait_for_result=args.wait)
        elif args.operation == "target":
            result = target(request, wait_for_result=args.wait)
        elif args.operation == "final_audit":
            result = final_audit(request, wait_for_result=args.wait)
        elif args.operation == "status":
            if args.wait:
                raise EvaluatorError(
                    "invalid_evaluator_input",
                    "status does not accept --wait",
                )
            result = invocation_status(request)
        elif args.operation == "cancel":
            if args.wait:
                raise EvaluatorError(
                    "invalid_evaluator_input",
                    "cancel does not accept --wait",
                )
            result = cancel_invocation(request)
        else:
            raise EvaluatorError(
                "operation_not_implemented",
                f"{args.operation} is not implemented",
            )
    except (EvaluatorError, OSError, ValueError, TimeoutError) as error:
        return _emit_error(error)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
