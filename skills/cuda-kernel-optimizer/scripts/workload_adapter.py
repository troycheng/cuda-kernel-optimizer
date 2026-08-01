#!/usr/bin/env python3
"""Validate and materialize the V1.4 command-driver protocol.

This module never starts a process.  Readiness, workload evaluation, and
profiler tools use it to build one identical driver request and argv.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath


DRIVER_PROTOCOL = "cuda-kernel-optimizer/driver-v1"
REQUEST_PROTOCOL = "cuda-kernel-optimizer/driver-request-v1"
RESULT_PROTOCOL = "cuda-kernel-optimizer/driver-result-v1"

_DRIVER_FIELDS = {
    "command",
    "request_argument",
    "execution_mode",
    "protocol_version",
    "profiler_capabilities",
    "side_effects",
    "cleanup_contract",
}
_CLEANUP_CONTRACT_FIELDS = {"kind", "external_tasks"}
_REQUEST_FIELDS = {
    "protocol_version",
    "request_digest",
    "target_id",
    "execution_id",
    "operation",
    "variant",
    "test_suite",
    "correctness",
    "objective",
    "role",
    "mode",
    "case",
    "sampling",
    "output_path",
    "driver_identity",
}
_RESULT_BASE_FIELDS = {
    "protocol_version",
    "request_digest",
    "target_id",
    "execution_id",
    "variant_digest",
    "role",
    "mode",
    "case_id",
    "artifacts",
    "cleanup",
    "driver_identity",
    "environment",
}
_VARIANT_FIELDS = {"kind", "digest", "locator"}
_CLEANUP_RESULT_FIELDS = {"status", "live_tasks"}
_CORRECTNESS_FIELDS = {"status", "metrics"}
_MEASUREMENTS_FIELDS = {"primary", "constraints"}
_PRIMARY_MEASUREMENT_FIELDS = {"name", "unit", "samples"}
_TEST_SUITE_FIELDS = {"digest", "locator", "case_ids"}
_CORRECTNESS_INPUT_FIELDS = {"reference", "method", "acceptance"}
_REFERENCE_INPUT_FIELDS = {"digest", "locator"}
_OBJECTIVE_INPUT_FIELDS = {"primary_metric", "constraints"}
_ENVIRONMENT_FIELDS = {
    "gpu_uuids",
    "gpu_models",
    "gpu_architectures",
    "driver_version",
    "cuda_runtime_version",
    "frameworks",
    "container",
}
_CONTAINER_FIELDS = {"kind", "identity"}
_CONSTRAINT_MEASUREMENT_FIELDS = {"name", "unit", "samples"}
_ARTIFACT_FIELDS = {"kind", "relative_path", "sha256"}
_PROFILE_RECEIPT_FIELDS = {
    "variant",
    "case_id",
    "status",
    "passed",
    "acceptance",
    "metrics",
    "gate",
    "evidence_refs",
}
_PROFILE_GATE_FIELDS = {
    "passed",
    "driver_status",
    "metric",
    "operator",
    "threshold",
    "observed",
    "status_consistent",
}
_PROFILE_EVIDENCE_REF_FIELDS = {
    "digest",
    "locator",
    "source_kind",
    "file_count",
    "total_bytes",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROFILER_CAPABILITIES = {
    "ncu_wrap_v1",
    "nsys_wrap_v1",
    "pytorch_chrome_trace_v1",
}
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_MAX_ARTIFACTS = 128
_MAX_METRICS = 256
_MAX_SAMPLES = 100_000
_MAX_CONSTRAINTS = 128
_MAX_GPUS = 64
_MAX_FRAMEWORKS = 128


def _load_artifact_store():
    path = Path(__file__).with_name("artifact_store.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_workload_adapter_store", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load artifact store: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load_artifact_store()


def _closed(value, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
    return value


def _text(value, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{label} must be a non-empty bounded string")
    return value


def _sha256(value, label: str) -> str:
    text = _text(value, label, maximum=64)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _json_copy(value, label: str):
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite JSON") from error


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stable_regular_file(path, label: str) -> dict:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    before = os.lstat(target)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    digest = STORE.sha256_file(target)
    after = os.lstat(target)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"{label} changed while being inspected")
    return {
        "path": str(target),
        "sha256": digest,
        "size_bytes": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
    }


def _command(value) -> tuple[list[str], list[dict]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise ValueError("driver.command must be a non-empty string list")
    argv = list(value)
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("driver.command must be a non-empty string list")
    executable = shutil.which(argv[0])
    if executable is None:
        raise ValueError(f"driver executable is unavailable: {argv[0]}")
    normalized = list(argv)
    normalized[0] = str(Path(executable).resolve())
    sources = [_stable_regular_file(normalized[0], "driver executable")]
    cwd = Path.cwd()
    for index, argument in enumerate(normalized[1:], start=1):
        if argument.startswith("-"):
            continue
        candidate = Path(argument).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            metadata = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("driver command contains a symlink input")
        if stat.S_ISREG(metadata.st_mode):
            source = _stable_regular_file(candidate, "driver command input")
            normalized[index] = source["path"]
            sources.append(source)
    unique = {item["path"]: item for item in sources}
    return normalized, [unique[path] for path in sorted(unique)]


def validate_driver(value) -> dict:
    """Return one closed, content-bound command driver."""
    driver = _closed(value, _DRIVER_FIELDS, "driver")
    if driver["protocol_version"] != DRIVER_PROTOCOL:
        raise ValueError("driver protocol_version is unsupported")
    command, sources = _command(driver["command"])
    request_argument = _text(
        driver["request_argument"], "driver.request_argument", maximum=64
    )
    if not request_argument.startswith("-"):
        raise ValueError("driver.request_argument must be an option")
    if driver["execution_mode"] not in {"separate", "combined"}:
        raise ValueError("driver.execution_mode must be separate or combined")
    profiler_capabilities = driver["profiler_capabilities"]
    if (
        type(profiler_capabilities) is not list
        or any(
            not isinstance(item, str) or item not in _PROFILER_CAPABILITIES
            for item in profiler_capabilities
        )
        or len(profiler_capabilities) != len(set(profiler_capabilities))
    ):
        raise ValueError(
            "driver.profiler_capabilities must be a unique supported string list"
        )
    side_effects = driver["side_effects"]
    if isinstance(side_effects, (str, bytes, bytearray)) or not isinstance(
        side_effects, Sequence
    ):
        raise ValueError("driver.side_effects must be a string list")
    normalized_side_effects = list(side_effects)
    if any(not isinstance(item, str) or not item for item in normalized_side_effects):
        raise ValueError("driver.side_effects must be a string list")
    if len(normalized_side_effects) != len(set(normalized_side_effects)):
        raise ValueError("driver.side_effects must not contain duplicates")
    cleanup = _closed(
        driver["cleanup_contract"],
        _CLEANUP_CONTRACT_FIELDS,
        "driver.cleanup_contract",
    )
    if type(cleanup["external_tasks"]) is not bool:
        raise ValueError("driver.cleanup_contract.external_tasks must be boolean")
    if cleanup != {"kind": "process_group_only", "external_tasks": False}:
        raise ValueError(
            "driver external tasks are unsupported; commands must remain in the "
            "invocation process group"
        )
    normalized = {
        "command": command,
        "request_argument": request_argument,
        "execution_mode": driver["execution_mode"],
        "protocol_version": DRIVER_PROTOCOL,
        "profiler_capabilities": sorted(profiler_capabilities),
        "side_effects": normalized_side_effects,
        "cleanup_contract": dict(cleanup),
        "source_files": sources,
    }
    normalized["identity"] = hashlib.sha256(
        _canonical_bytes(normalized)
    ).hexdigest()
    return normalized


def verify_driver(value) -> dict:
    """Recheck every content-bound command source before use."""
    if type(value) is not dict:
        raise ValueError("frozen driver must be an object")
    expected_fields = _DRIVER_FIELDS | {"source_files", "identity"}
    frozen = _closed(value, expected_fields, "frozen driver")
    rebuilt = validate_driver(
        {field: frozen[field] for field in _DRIVER_FIELDS}
    )
    if rebuilt != frozen:
        raise ValueError("driver identity changed after readiness")
    return rebuilt


def validate_variant(value) -> dict:
    variant = _closed(value, _VARIANT_FIELDS, "variant")
    if variant["kind"] not in {"source_snapshot", "artifact", "deployment"}:
        raise ValueError("variant.kind is unsupported")
    return {
        "kind": variant["kind"],
        "digest": _sha256(variant["digest"], "variant.digest"),
        "locator": _text(variant["locator"], "variant.locator"),
    }


def materialize_variant(
    artifact_root,
    workspace,
    frozen_variant: Mapping,
    name: str,
) -> dict:
    """Materialize one frozen Variant for the sole command-driver path."""
    frozen = dict(frozen_variant)
    unknown = set(frozen) - _VARIANT_FIELDS - {"role"}
    if unknown:
        raise ValueError(
            f"frozen variant contains unknown fields: {sorted(unknown)}"
        )
    if "role" in frozen and frozen["role"] not in {
        "original",
        "reference",
        "candidate",
    }:
        raise ValueError("frozen variant role is unsupported")
    variant = validate_variant(
        {field: frozen[field] for field in _VARIANT_FIELDS}
    )
    materialized = STORE.materialize_object(
        artifact_root,
        {"digest": variant["digest"], "locator": variant["locator"]},
        Path(workspace) / _text(name, "variant materialization name", maximum=128),
    )
    return {**variant, "locator": str(materialized)}


def materialize_target_inputs(
    artifact_root,
    workspace,
    target: Mapping,
) -> dict:
    """Materialize the Target inputs shared by evaluator and profilers."""
    if not isinstance(target, Mapping):
        raise ValueError("target must be an object")
    test_suite = target.get("test_suite")
    correctness = target.get("correctness")
    if not isinstance(test_suite, Mapping) or not isinstance(correctness, Mapping):
        raise ValueError("target driver inputs are unavailable")
    test_object = test_suite.get("object_ref")
    correctness_object = correctness.get("reference")
    if not isinstance(test_object, Mapping) or not isinstance(
        correctness_object, Mapping
    ):
        raise ValueError("target driver object references are unavailable")
    root = Path(workspace)
    materialized_test_suite = STORE.materialize_object(
        artifact_root,
        dict(test_object),
        root / "test-suite",
    )
    materialized_correctness = STORE.materialize_object(
        artifact_root,
        dict(correctness_object),
        root / "correctness-reference",
    )
    return {
        "test_suite": {
            "digest": _sha256(
                test_object.get("digest"),
                "target.test_suite.object_ref.digest",
            ),
            "locator": str(materialized_test_suite),
            "case_ids": _json_copy(
                test_suite.get("case_ids"),
                "target.test_suite.case_ids",
            ),
        },
        "correctness": {
            "reference": {
                "digest": _sha256(
                    correctness_object.get("digest"),
                    "target.correctness.reference.digest",
                ),
                "locator": str(materialized_correctness),
            },
            "method": _text(
                correctness.get("method"),
                "target.correctness.method",
                maximum=64,
            ),
            "acceptance": _json_copy(
                correctness.get("acceptance"),
                "target.correctness.acceptance",
            ),
        },
        "objective": {
            "primary_metric": _json_copy(
                target.get("primary_metric"),
                "target.primary_metric",
            ),
            "constraints": _json_copy(
                target.get("constraints"),
                "target.constraints",
            ),
        },
    }


def build_driver_request(
    *,
    target_id: str,
    execution_id: str,
    operation: str,
    driver: Mapping,
    variant: Mapping,
    test_suite: Mapping,
    correctness: Mapping,
    objective: Mapping,
    role: str,
    mode: str,
    case: Mapping,
    sampling: Mapping,
    output_path,
) -> dict:
    frozen_driver = verify_driver(driver)
    normalized_variant = validate_variant(variant)
    test_suite = _closed(dict(test_suite), _TEST_SUITE_FIELDS, "test_suite")
    test_suite = {
        "digest": _sha256(test_suite["digest"], "test_suite.digest"),
        "locator": _text(test_suite["locator"], "test_suite.locator"),
        "case_ids": _json_copy(test_suite["case_ids"], "test_suite.case_ids"),
    }
    if (
        type(test_suite["case_ids"]) is not list
        or not test_suite["case_ids"]
        or any(
            not isinstance(case_id, str) or not case_id
            for case_id in test_suite["case_ids"]
        )
        or len(test_suite["case_ids"]) != len(set(test_suite["case_ids"]))
    ):
        raise ValueError("test_suite.case_ids must be a non-empty unique string list")
    correctness = _closed(
        dict(correctness),
        _CORRECTNESS_INPUT_FIELDS,
        "correctness input",
    )
    reference = _closed(
        dict(correctness["reference"]),
        _REFERENCE_INPUT_FIELDS,
        "correctness reference",
    )
    correctness = {
        "reference": {
            "digest": _sha256(reference["digest"], "correctness.reference.digest"),
            "locator": _text(
                reference["locator"], "correctness.reference.locator"
            ),
        },
        "method": _text(correctness["method"], "correctness.method", maximum=64),
        "acceptance": _json_copy(
            correctness["acceptance"], "correctness.acceptance"
        ),
    }
    objective = _closed(
        dict(objective), _OBJECTIVE_INPUT_FIELDS, "objective input"
    )
    objective = {
        "primary_metric": _json_copy(
            objective["primary_metric"], "objective.primary_metric"
        ),
        "constraints": _json_copy(
            objective["constraints"], "objective.constraints"
        ),
    }
    if role not in {"original", "reference", "candidate"}:
        raise ValueError("driver role is unsupported")
    if mode not in {"correctness", "measure", "combined"}:
        raise ValueError("driver mode is unsupported")
    if frozen_driver["execution_mode"] == "separate" and mode == "combined":
        raise ValueError("separate driver cannot run combined mode")
    if frozen_driver["execution_mode"] == "combined" and mode != "combined":
        raise ValueError("combined driver requires combined mode")
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(output_path))))
    core = {
        "protocol_version": REQUEST_PROTOCOL,
        "target_id": _text(target_id, "target_id", maximum=128),
        "execution_id": _text(execution_id, "execution_id", maximum=128),
        "operation": _text(operation, "operation", maximum=64),
        "variant": normalized_variant,
        "test_suite": test_suite,
        "correctness": correctness,
        "objective": objective,
        "role": role,
        "mode": mode,
        "case": _json_copy(case, "case"),
        "sampling": _json_copy(sampling, "sampling"),
        "output_path": str(target),
        "driver_identity": frozen_driver["identity"],
    }
    request_digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    return {
        "protocol_version": REQUEST_PROTOCOL,
        "request_digest": request_digest,
        **{key: value for key, value in core.items() if key != "protocol_version"},
    }


def build_argv(driver: Mapping, request_path) -> list[str]:
    frozen = verify_driver(driver)
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(request_path))))
    return frozen["command"] + [frozen["request_argument"], str(path)]


def _strict_json(path) -> dict:
    raw = STORE.read_regular_bytes(path, maximum_bytes=_MAX_RESULT_BYTES)

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"driver result contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"driver result contains non-finite number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("driver result is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError("driver result root must be an object")
    return value


def _read_bound_json(path, expected_digest: str, label: str) -> dict:
    raw = STORE.read_regular_bytes(path, maximum_bytes=_MAX_RESULT_BYTES)
    if hashlib.sha256(raw).hexdigest() != _sha256(
        expected_digest, f"{label}.sha256"
    ):
        raise ValueError(f"{label} digest changed")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"{label} contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _invocation_reference(value, label: str, *, include_case: bool) -> dict:
    fields = {"invocation_id", "sha256"}
    if include_case:
        fields.add("case_id")
    reference = _closed(value, fields, label)
    invocation_id = _text(
        reference["invocation_id"],
        f"{label}.invocation_id",
        maximum=128,
    )
    if (
        not invocation_id.startswith("inv-")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in invocation_id
        )
    ):
        raise ValueError(f"{label}.invocation_id is invalid")
    normalized = {
        "invocation_id": invocation_id,
        "sha256": _sha256(reference["sha256"], f"{label}.sha256"),
    }
    if include_case:
        normalized["case_id"] = _text(
            reference["case_id"],
            f"{label}.case_id",
            maximum=128,
        )
    return normalized


def _profile_target(root: Path, reference) -> dict:
    reference = _closed(reference, {"id", "sha256"}, "target_ref")
    expected_id = _text(reference["id"], "target_ref.id", maximum=128)
    target = _read_bound_json(
        root / "target.json",
        reference["sha256"],
        "target_ref",
    )
    if (
        target.get("record_type") != "target"
        or target.get("format_version")
        != "cuda-kernel-optimizer/target-v1"
        or target.get("id") != expected_id
        or target.get("target_mode") != "optimization"
    ):
        raise ValueError("target_ref is not a frozen optimization Target")
    return target


def _profile_baseline(
    root: Path,
    reference,
    target_ref: dict,
    target: dict,
) -> dict:
    normalized = _invocation_reference(
        reference, "baseline_ref", include_case=False
    )
    result = _read_bound_json(
        root
        / "invocations"
        / normalized["invocation_id"]
        / "result.json",
        normalized["sha256"],
        "baseline_ref",
    )
    if (
        result.get("operation") != "baseline"
        or result.get("target_ref") != target_ref
        or result.get("execution_status") != "succeeded"
        or result.get("measurement_validity") != "valid"
        or result.get("verdict") != "passed"
        or result.get("cleanup_status") != "confirmed"
        or result.get("variant_refs") != [target.get("original")]
    ):
        raise ValueError("baseline_ref is not a valid original baseline")
    return normalized


def _profile_experiment(
    root: Path,
    reference,
    target_ref: dict,
    baseline_ref: dict,
) -> tuple[dict, dict]:
    reference = _closed(reference, {"id", "sha256"}, "experiment_ref")
    experiment_id = _text(
        reference["id"], "experiment_ref.id", maximum=128
    )
    if (
        not experiment_id.startswith("exp-")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in experiment_id
        )
    ):
        raise ValueError("experiment_ref.id is invalid")
    normalized = {
        "id": experiment_id,
        "sha256": _sha256(
            reference["sha256"], "experiment_ref.sha256"
        ),
    }
    experiment = _read_bound_json(
        root / "experiments" / f"{experiment_id}.json",
        normalized["sha256"],
        "experiment_ref",
    )
    if (
        experiment.get("record_type") != "experiment"
        or experiment.get("format_version")
        != "cuda-kernel-optimizer/experiment-v1"
        or experiment.get("id") != experiment_id
        or experiment.get("target_ref") != target_ref
        or experiment.get("baseline_ref") != baseline_ref
    ):
        raise ValueError("experiment_ref is not bound to this Target and baseline")
    candidate = experiment.get("candidate")
    if type(candidate) is not dict:
        raise ValueError("experiment_ref has no candidate Variant")
    return normalized, {
        **experiment,
        "candidate": _profile_variant(
            candidate,
            "experiment_ref.candidate",
            expected_role="candidate",
        ),
    }


def _profile_variant(value, label: str, *, expected_role: str | None = None) -> dict:
    frozen = _closed(value, _VARIANT_FIELDS | {"role"}, label)
    if frozen["role"] not in {"original", "reference", "candidate"}:
        raise ValueError(f"{label}.role is unsupported")
    if expected_role is not None and frozen["role"] != expected_role:
        raise ValueError(f"{label}.role must be {expected_role}")
    return {
        **validate_variant({field: frozen[field] for field in _VARIANT_FIELDS}),
        "role": frozen["role"],
    }


def _profile_evidence_ref(root: Path, value, command_receipts) -> dict:
    evidence_ref = _closed(
        value,
        _PROFILE_EVIDENCE_REF_FIELDS,
        "correctness receipt evidence_ref",
    )
    digest = _sha256(
        evidence_ref["digest"], "correctness receipt evidence_ref.digest"
    )
    locator = _text(
        evidence_ref["locator"], "correctness receipt evidence_ref.locator"
    )
    expected_locator = f"objects/sha256/{digest}"
    if locator != expected_locator:
        raise ValueError("correctness receipt evidence_ref locator is not content-bound")
    if evidence_ref["source_kind"] != "directory":
        raise ValueError("correctness receipt evidence_ref must reference one driver bundle")
    if (
        type(evidence_ref["file_count"]) is not int
        or not 1 <= evidence_ref["file_count"] <= _MAX_ARTIFACTS + 1
        or type(evidence_ref["total_bytes"]) is not int
        or not 0 < evidence_ref["total_bytes"] <= _MAX_ARTIFACT_BYTES + _MAX_RESULT_BYTES
    ):
        raise ValueError("correctness receipt evidence_ref summary is invalid")
    if type(command_receipts) is not list:
        raise ValueError("correctness result command_receipts must be a list")
    matches = [
        receipt
        for receipt in command_receipts
        if type(receipt) is dict
        and receipt.get("driver_output_ref") == evidence_ref
    ]
    if len(matches) != 1 or set(matches[0]) != {
        "request", "command_result", "driver_output_ref", "driver_artifacts"
    }:
        raise ValueError("correctness evidence is not bound to one driver receipt")
    receipt = matches[0]
    if (
        type(receipt["command_result"]) is not dict
        or receipt["command_result"].get("status") != "completed"
    ):
        raise ValueError("correctness driver receipt did not complete")
    with tempfile.TemporaryDirectory(prefix="cko-profile-evidence-") as temporary:
        manifest = STORE._load_object_manifest(
            root,
            evidence_ref,
            verify_payload=True,
        )
        materialized = Path(temporary) / "result.json"
        STORE.materialize_object_member(
            root,
            evidence_ref,
            "result.json",
            materialized,
        )
        driver_result = validate_driver_result(
            materialized,
            receipt["request"],
            bundle_manifest=manifest,
        )
    if driver_result["artifacts"] != receipt["driver_artifacts"]:
        raise ValueError("correctness driver artifact receipt changed")
    return dict(evidence_ref)


def _profile_correctness(
    root: Path,
    reference,
    *,
    target_ref: dict,
    experiment_ref: dict,
    candidate: dict,
    target: dict,
    case_id: str,
) -> dict:
    normalized = _invocation_reference(
        reference, "correctness_ref", include_case=True
    )
    if normalized["case_id"] != case_id:
        raise ValueError("correctness_ref is for a different workload case")
    result = _read_bound_json(
        root
        / "invocations"
        / normalized["invocation_id"]
        / "result.json",
        normalized["sha256"],
        "correctness_ref",
    )
    if (
        result.get("operation") not in {"screen", "target"}
        or result.get("target_ref") != target_ref
        or result.get("experiment_ref") != experiment_ref
        or result.get("execution_status") != "succeeded"
        or result.get("cleanup_status") != "confirmed"
    ):
        raise ValueError("correctness_ref is not bound to this candidate")
    variants = result.get("variant_refs")
    if type(variants) is not list or sum(item == candidate for item in variants) != 1:
        raise ValueError("correctness_ref does not bind this candidate Variant")
    receipts = result.get("correctness_receipts")
    if type(receipts) is not list:
        raise ValueError("correctness_ref has no correctness receipts")
    matches = []
    expected_acceptance = _closed(
        target.get("correctness", {}).get("acceptance"),
        {"metric", "operator", "value"},
        "Target correctness acceptance",
    )
    for index, value in enumerate(receipts):
        receipt = _closed(
            value,
            _PROFILE_RECEIPT_FIELDS,
            f"correctness receipt[{index}]",
        )
        if receipt["variant"] != candidate or receipt["case_id"] != case_id:
            continue
        if receipt["status"] != "valid" or receipt["passed"] is not True:
            continue
        acceptance = _closed(
            receipt["acceptance"],
            {"metric", "operator", "value"},
            f"correctness receipt[{index}].acceptance",
        )
        if acceptance != expected_acceptance:
            raise ValueError("correctness receipt acceptance is not bound to Target")
        gate = _closed(
            receipt["gate"],
            _PROFILE_GATE_FIELDS,
            f"correctness receipt[{index}].gate",
        )
        computed = evaluate_correctness(
            {
                "status": gate["driver_status"],
                "metrics": receipt["metrics"],
            },
            acceptance,
        )
        if gate != computed or not computed["passed"]:
            raise ValueError("correctness receipt gate is not a passing result")
        if type(receipt["evidence_refs"]) is not list or len(receipt["evidence_refs"]) != 1:
            raise ValueError("correctness receipt must contain one content-bound evidence_ref")
        _profile_evidence_ref(
            root,
            receipt["evidence_refs"][0],
            result.get("command_receipts"),
        )
        matches.append(receipt)
    if len(matches) != 1:
        raise ValueError(
            "correctness_ref does not contain one passing candidate receipt"
        )
    return normalized


def resolve_profile_collection(
    *,
    artifact_root,
    target_ref,
    baseline_ref,
    role: str,
    case_id: str,
    capability: str,
    experiment_ref=None,
    correctness_ref=None,
) -> dict:
    """Resolve immutable records needed before any profiler command can start."""
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(artifact_root)))
    )
    if not root.is_dir():
        raise ValueError("artifact_root is unavailable")
    normalized_target_ref = _closed(
        target_ref, {"id", "sha256"}, "target_ref"
    )
    normalized_target_ref = {
        "id": _text(
            normalized_target_ref["id"], "target_ref.id", maximum=128
        ),
        "sha256": _sha256(
            normalized_target_ref["sha256"], "target_ref.sha256"
        ),
    }
    target = _profile_target(root, normalized_target_ref)
    driver = verify_driver(target.get("driver"))
    if capability not in _PROFILER_CAPABILITIES:
        raise ValueError("profiler capability is unsupported")
    if capability not in driver["profiler_capabilities"]:
        raise ValueError(
            f"driver does not declare profiler capability: {capability}"
        )
    case_id = _text(case_id, "case_id", maximum=128)
    if case_id not in target["test_suite"]["case_ids"]:
        raise ValueError("case_id is outside the frozen test suite")
    normalized_baseline_ref = _profile_baseline(
        root, baseline_ref, normalized_target_ref, target
    )
    if role == "original":
        if experiment_ref is not None or correctness_ref is not None:
            raise ValueError(
                "original collection must not include candidate references"
            )
        variant = target.get("original")
        if type(variant) is not dict:
            raise ValueError("Target has no original Variant")
        variant = _profile_variant(
            variant,
            "Target original Variant",
            expected_role="original",
        )
        normalized_experiment_ref = None
        normalized_correctness_ref = None
    elif role == "candidate":
        if experiment_ref is None or correctness_ref is None:
            raise ValueError(
                "candidate collection requires experiment_ref and correctness_ref"
            )
        normalized_experiment_ref, experiment = _profile_experiment(
            root,
            experiment_ref,
            normalized_target_ref,
            normalized_baseline_ref,
        )
        variant = experiment["candidate"]
        normalized_correctness_ref = _profile_correctness(
            root,
            correctness_ref,
            target_ref=normalized_target_ref,
            experiment_ref=normalized_experiment_ref,
            candidate=variant,
            target=target,
            case_id=case_id,
        )
    else:
        raise ValueError("profile collection role is unsupported")
    return {
        "artifact_root": str(root),
        "target_ref": normalized_target_ref,
        "target": target,
        "baseline_ref": normalized_baseline_ref,
        "role": role,
        "case_id": case_id,
        "variant": dict(variant),
        "experiment_ref": normalized_experiment_ref,
        "correctness_ref": normalized_correctness_ref,
        "driver": driver,
    }


def _analysis_target(root: Path, reference) -> tuple[dict, dict]:
    normalized = _closed(reference, {"id", "sha256"}, "target_ref")
    normalized = {
        "id": _text(normalized["id"], "target_ref.id", maximum=128),
        "sha256": _sha256(normalized["sha256"], "target_ref.sha256"),
    }
    target = _read_bound_json(
        root / "target.json",
        normalized["sha256"],
        "target_ref",
    )
    if (
        target.get("record_type") != "target"
        or target.get("format_version") != "cuda-kernel-optimizer/target-v1"
        or target.get("id") != normalized["id"]
        or target.get("target_mode") not in {"optimization", "diagnostic"}
    ):
        raise ValueError("target_ref is not one frozen Target")
    return normalized, target


def _analysis_artifact_ref(value) -> dict:
    if type(value) is not dict:
        raise ValueError("artifact_ref must be an object")
    source = value.get("source")
    optional = {"stage"}
    if source == "target_material":
        required = {"source", "material_ref"}
    elif source == "invocation_driver_artifact":
        required = {
            "source", "invocation_ref", "receipt_index", "relative_path"
        }
    else:
        raise ValueError("artifact_ref.source is unsupported")
    if not required.issubset(value) or set(value) - required - optional:
        raise ValueError("artifact_ref fields are incomplete or unknown")
    normalized = {field: value[field] for field in required}
    if "stage" in value:
        normalized["stage"] = _text(value["stage"], "artifact_ref.stage", maximum=64)
    return normalized


def resolve_analysis_artifact(*, artifact_root, target_ref, artifact_ref) -> dict:
    """Resolve one immutable compiler/SASS input without starting a process."""
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(artifact_root))))
    if not root.is_dir():
        raise ValueError("artifact_root is unavailable")
    normalized_target_ref, target = _analysis_target(root, target_ref)
    selected = _analysis_artifact_ref(artifact_ref)
    response = {
        "artifact_root": str(root),
        "target_ref": normalized_target_ref,
        "target": target,
        "environment": _json_copy(target.get("environment"), "Target environment"),
    }
    if "stage" in selected:
        response["requested_stage"] = selected["stage"]

    if selected["source"] == "target_material":
        if target.get("target_mode") != "diagnostic":
            raise ValueError("target_material requires a diagnostic Target")
        material_ref = _closed(
            selected["material_ref"], {"id", "sha256"}, "material_ref"
        )
        material_ref = {
            "id": _text(material_ref["id"], "material_ref.id", maximum=128),
            "sha256": _sha256(material_ref["sha256"], "material_ref.sha256"),
        }
        matches = [
            material
            for material in target.get("diagnostic_materials", [])
            if type(material) is dict and material.get("id") == material_ref["id"]
        ]
        if (
            len(matches) != 1
            or matches[0].get("sha256") != material_ref["sha256"]
            or type(matches[0].get("object_ref")) is not dict
            or matches[0]["object_ref"].get("digest") != material_ref["sha256"]
        ):
            raise ValueError("material_ref is not bound to this Target")
        material = matches[0]
        manifest = STORE._load_object_manifest(
            root, material["object_ref"], verify_payload=True
        )
        files = [entry for entry in manifest["entries"] if entry["kind"] == "file"]
        if manifest["source_kind"] != "file" or len(files) != 1:
            raise ValueError("target material is not one frozen file")
        return {
            **response,
            "source": "target_material",
            "material_ref": material_ref,
            "material": _json_copy(material, "diagnostic material"),
            "object_ref": dict(material["object_ref"]),
            "artifact": {
                "kind": material.get("kind"),
                "dialect": material.get("dialect"),
                **dict(files[0]),
            },
        }

    invocation_ref = _invocation_reference(
        selected["invocation_ref"], "invocation_ref", include_case=False
    )
    result = _read_bound_json(
        root / "invocations" / invocation_ref["invocation_id"] / "result.json",
        invocation_ref["sha256"],
        "invocation_ref",
    )
    if (
        result.get("record_type") != "invocation_result"
        or result.get("format_version")
        != "cuda-kernel-optimizer/evaluator-result-v1"
        or result.get("operation")
        not in {"baseline", "screen", "target", "final_audit"}
        or result.get("target_ref") != normalized_target_ref
        or result.get("execution_status") != "succeeded"
        or result.get("cleanup_status") != "confirmed"
    ):
        raise ValueError("invocation_ref is not an accepted evaluator result")
    receipts = result.get("command_receipts")
    index = selected["receipt_index"]
    if (
        type(receipts) is not list
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(receipts)
    ):
        raise ValueError("artifact_ref.receipt_index is invalid")
    receipt = receipts[index]
    if type(receipt) is not dict or set(receipt) != {
        "request", "command_result", "driver_output_ref", "driver_artifacts"
    }:
        raise ValueError("selected command receipt is not a closed driver receipt")
    if (
        type(receipt["command_result"]) is not dict
        or receipt["command_result"].get("status") != "completed"
    ):
        raise ValueError("selected driver command did not complete")
    driver_request = receipt["request"]
    if (
        type(driver_request) is not dict
        or driver_request.get("execution_id") != invocation_ref["invocation_id"]
        or driver_request.get("target_id") != normalized_target_ref["id"]
        or driver_request.get("operation") != result["operation"]
        or driver_request.get("role") not in {"original", "reference", "candidate"}
    ):
        raise ValueError("selected driver receipt is not invocation-bound")
    relative_path = "/".join(
        _artifact_relative_path(
            selected["relative_path"], "artifact_ref.relative_path"
        )
    )
    if relative_path == "result.json":
        raise ValueError("artifact_ref.relative_path is reserved")
    output_ref = _closed(
        receipt["driver_output_ref"],
        _PROFILE_EVIDENCE_REF_FIELDS,
        "driver_output_ref",
    )
    manifest = STORE._load_object_manifest(root, output_ref, verify_payload=True)
    with tempfile.TemporaryDirectory(prefix="cko-analysis-artifact-") as temporary:
        STORE.materialize_object_member(
            root, output_ref, "result.json", Path(temporary) / "result.json"
        )
        normalized_driver_result = validate_driver_result(
            Path(temporary) / "result.json",
            driver_request,
            bundle_manifest=manifest,
        )
    if normalized_driver_result["artifacts"] != receipt["driver_artifacts"]:
        raise ValueError("driver artifact receipt changed")
    artifacts = [
        artifact
        for artifact in normalized_driver_result["artifacts"]
        if artifact["relative_path"] == relative_path
    ]
    if len(artifacts) != 1:
        raise ValueError("artifact_ref does not select one declared driver artifact")
    artifact = artifacts[0]
    members = [
        entry
        for entry in manifest["entries"]
        if entry["kind"] == "file" and entry["path"] == relative_path
    ]
    if (
        len(members) != 1
        or members[0]["sha256"] != artifact["sha256"]
        or members[0]["size_bytes"] < 0
    ):
        raise ValueError("selected driver artifact member is not manifest-bound")
    artifact = {**artifact, "size_bytes": members[0]["size_bytes"]}
    variants = result.get("variant_refs")
    role = driver_request.get("role")
    request_variant = driver_request.get("variant")
    matches = []
    if type(variants) is list and type(request_variant) is dict:
        matches = [
            variant
            for variant in variants
            if type(variant) is dict
            and variant.get("role") == role
            and variant.get("kind") == request_variant.get("kind")
            and variant.get("digest") == request_variant.get("digest")
        ]
    if len(matches) != 1:
        raise ValueError("selected driver receipt Variant is not result-bound")
    bound = {
        **response,
        "source": "invocation_driver_artifact",
        "invocation_ref": invocation_ref,
        "receipt_index": index,
        "object_ref": dict(output_ref),
        "artifact": dict(artifact),
        "role": role,
        "variant": dict(matches[0]),
    }
    if role == "candidate":
        if result.get("operation") not in {"screen", "target"}:
            raise ValueError("candidate artifact operation is invalid")
        experiment_ref = result.get("experiment_ref")
        if type(experiment_ref) is not dict or set(experiment_ref) != {"id", "sha256"}:
            raise ValueError("candidate artifact has no Experiment binding")
        experiment = _read_bound_json(
            root / "experiments" / f"{experiment_ref['id']}.json",
            experiment_ref["sha256"],
            "experiment_ref",
        )
        if (
            experiment.get("record_type") != "experiment"
            or experiment.get("format_version")
            != "cuda-kernel-optimizer/experiment-v1"
            or experiment.get("id") != experiment_ref["id"]
            or experiment.get("target_ref") != normalized_target_ref
            or experiment.get("candidate") != matches[0]
        ):
            raise ValueError("candidate artifact Experiment binding is invalid")
        bound["experiment_ref"] = dict(experiment_ref)
        bound["mechanism_key"] = _text(
            experiment.get("mechanism_key"),
            "experiment mechanism_key",
            maximum=256,
        )
    return bound


def _validate_metrics(value, label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    if len(value) > _MAX_METRICS:
        raise ValueError(f"{label} exceeds the metric limit")
    return {
        _text(name, f"{label} metric name", maximum=128): _finite(
            metric, f"{label}.{name}"
        )
        for name, metric in value.items()
    }


def _validate_correctness(value) -> dict:
    correctness = _closed(value, _CORRECTNESS_FIELDS, "correctness")
    if correctness["status"] not in {"passed", "failed"}:
        raise ValueError("correctness.status must be passed or failed")
    return {
        "status": correctness["status"],
        "metrics": _validate_metrics(correctness["metrics"], "correctness.metrics"),
    }


def _validate_measurements(value) -> dict:
    measurements = _closed(value, _MEASUREMENTS_FIELDS, "measurements")
    primary = _closed(
        measurements["primary"],
        _PRIMARY_MEASUREMENT_FIELDS,
        "measurements.primary",
    )
    samples = primary["samples"]
    if isinstance(samples, (str, bytes, bytearray)) or not isinstance(
        samples, Sequence
    ):
        raise ValueError("measurements.primary.samples must be a number list")
    normalized_samples = [
        _finite(item, f"measurements.primary.samples[{index}]")
        for index, item in enumerate(samples)
    ]
    if not normalized_samples:
        raise ValueError("measurements.primary.samples must not be empty")
    if len(normalized_samples) > _MAX_SAMPLES:
        raise ValueError("measurements.primary.samples exceeds the sample limit")
    constraints = measurements["constraints"]
    if type(constraints) is not list:
        raise ValueError("measurements.constraints must be a list")
    if len(constraints) > _MAX_CONSTRAINTS:
        raise ValueError("measurements.constraints exceeds the constraint limit")
    normalized_constraints = []
    names = set()
    for index, item in enumerate(constraints):
        item = _closed(
            item,
            _CONSTRAINT_MEASUREMENT_FIELDS,
            f"measurements.constraints[{index}]",
        )
        name = _text(
            item["name"],
            f"measurements.constraints[{index}].name",
            maximum=128,
        )
        if name in names:
            raise ValueError("measurements.constraints names must be unique")
        names.add(name)
        values = item["samples"]
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(
            values, Sequence
        ):
            raise ValueError(
                f"measurements.constraints[{index}].samples must be a number list"
            )
        normalized_values = [
            _finite(
                sample,
                f"measurements.constraints[{index}].samples[{sample_index}]",
            )
            for sample_index, sample in enumerate(values)
        ]
        if not normalized_values:
            raise ValueError(
                f"measurements.constraints[{index}].samples must not be empty"
            )
        if len(normalized_values) > _MAX_SAMPLES:
            raise ValueError(
                f"measurements.constraints[{index}].samples exceeds the sample limit"
            )
        normalized_constraints.append(
            {
                "name": name,
                "unit": _text(
                    item["unit"],
                    f"measurements.constraints[{index}].unit",
                    maximum=64,
                ),
                "samples": normalized_values,
            }
        )
    return {
        "primary": {
            "name": _text(primary["name"], "measurements.primary.name", maximum=128),
            "unit": _text(primary["unit"], "measurements.primary.unit", maximum=64),
            "samples": normalized_samples,
        },
        "constraints": normalized_constraints,
    }


def _validate_environment(value) -> dict:
    environment = _closed(value, _ENVIRONMENT_FIELDS, "driver environment")
    gpu_uuids = environment["gpu_uuids"]
    gpu_models = environment["gpu_models"]
    gpu_architectures = environment["gpu_architectures"]
    if (
        type(gpu_uuids) is not list
        or any(not isinstance(item, str) or not item for item in gpu_uuids)
        or len(gpu_uuids) != len(set(gpu_uuids))
    ):
        raise ValueError("driver environment gpu_uuids must be a unique string list")
    if len(gpu_uuids) > _MAX_GPUS:
        raise ValueError("driver environment exceeds the GPU limit")
    if (
        type(gpu_models) is not list
        or any(not isinstance(item, str) or not item for item in gpu_models)
        or len(gpu_models) != len(gpu_uuids)
    ):
        raise ValueError("driver environment gpu_models must align with gpu_uuids")
    if (
        type(gpu_architectures) is not list
        or any(
            not isinstance(item, str) or not item
            for item in gpu_architectures
        )
        or len(gpu_architectures) != len(gpu_uuids)
    ):
        raise ValueError(
            "driver environment gpu_architectures must align with gpu_uuids"
        )
    frameworks = environment["frameworks"]
    if type(frameworks) is not dict or any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in frameworks.items()
    ):
        raise ValueError("driver environment frameworks must map names to versions")
    if len(frameworks) > _MAX_FRAMEWORKS:
        raise ValueError("driver environment exceeds the framework limit")
    container = _closed(
        environment["container"], _CONTAINER_FIELDS, "driver environment container"
    )
    return {
        "gpu_uuids": list(gpu_uuids),
        "gpu_models": list(gpu_models),
        "gpu_architectures": list(gpu_architectures),
        "driver_version": _text(
            environment["driver_version"],
            "driver environment driver_version",
            maximum=256,
        ),
        "cuda_runtime_version": _text(
            environment["cuda_runtime_version"],
            "driver environment cuda_runtime_version",
            maximum=256,
        ),
        "frameworks": dict(sorted(frameworks.items())),
        "container": {
            "kind": _text(container["kind"], "driver environment container.kind"),
            "identity": _text(
                container["identity"],
                "driver environment container.identity",
            ),
        },
    }


def _artifact_relative_path(value, label: str) -> tuple[str, ...]:
    relative = _text(value, label)
    if "\\" in relative or "\x00" in relative:
        raise ValueError(f"{label} must be a canonical POSIX relative path")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != relative
    ):
        raise ValueError(f"{label} must be a canonical POSIX relative path")
    return path.parts


def _sha256_relative_regular_file(
    root: Path,
    parts: tuple[str, ...],
    label: str,
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, directory_flags)
        descriptors.append(descriptor)
        for part in parts[:-1]:
            descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        file_descriptor = os.open(parts[-1], flags, dir_fd=descriptor)
        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"{label} must select a regular file")
            if metadata.st_size > _MAX_ARTIFACT_BYTES:
                raise ValueError(f"{label} exceeds the artifact byte limit")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(file_descriptor)
            if (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"{label} changed while being inspected")
            return digest.hexdigest()
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise ValueError(f"{label} is unavailable or unsafe") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _validate_artifacts(value, result_path, manifest_files=None) -> list[dict]:
    if type(value) is not list:
        raise ValueError("driver result artifacts must be a list")
    if len(value) > _MAX_ARTIFACTS:
        raise ValueError("driver result artifacts exceeds the artifact limit")
    root = Path(
        os.path.abspath(os.path.expanduser(os.fspath(result_path)))
    ).parent
    normalized = []
    relative_paths = set()
    for index, item in enumerate(value):
        artifact = _closed(
            item,
            _ARTIFACT_FIELDS,
            f"driver result artifacts[{index}]",
        )
        kind = _text(
            artifact["kind"],
            f"driver result artifacts[{index}].kind",
            maximum=128,
        )
        parts = _artifact_relative_path(
            artifact["relative_path"],
            f"driver result artifacts[{index}].relative_path",
        )
        relative_path = "/".join(parts)
        if relative_path == "result.json":
            raise ValueError("driver result artifact path is reserved: result.json")
        if relative_path in relative_paths:
            raise ValueError("driver result artifact paths must be unique")
        relative_paths.add(relative_path)
        expected = _sha256(
            artifact["sha256"],
            f"driver result artifacts[{index}].sha256",
        )
        if manifest_files is None:
            actual = _sha256_relative_regular_file(
                root,
                parts,
                f"driver result artifacts[{index}]",
            )
        else:
            entry = manifest_files.get(relative_path)
            if type(entry) is not dict or entry.get("kind") != "file":
                raise ValueError(
                    f"driver result artifacts[{index}] is absent from frozen bundle"
                )
            actual = entry.get("sha256")
        if actual != expected:
            raise ValueError(
                f"driver result artifacts[{index}] digest does not match"
            )
        normalized.append(
            {
                "kind": kind,
                "relative_path": relative_path,
                "sha256": expected,
            }
        )
    return normalized


def evaluate_correctness(correctness: Mapping, acceptance: Mapping) -> dict:
    """Apply the frozen correctness rule instead of trusting driver status."""
    normalized = _validate_correctness(dict(correctness))
    rule = _closed(
        dict(acceptance),
        {"metric", "operator", "value"},
        "correctness acceptance",
    )
    metric = _text(rule["metric"], "correctness acceptance metric", maximum=128)
    if metric not in normalized["metrics"]:
        raise ValueError("correctness acceptance metric is missing")
    threshold = _finite(rule["value"], "correctness acceptance value")
    observed = normalized["metrics"][metric]
    operator = rule["operator"]
    if operator == "greater_or_equal":
        computed = observed >= threshold
    elif operator == "less_or_equal":
        computed = observed <= threshold
    elif operator == "equal":
        computed = observed == threshold
    else:
        raise ValueError("correctness acceptance operator is unsupported")
    return {
        "passed": bool(computed and normalized["status"] == "passed"),
        "driver_status": normalized["status"],
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "status_consistent": (
            normalized["status"] == ("passed" if computed else "failed")
        ),
    }


def validate_driver_result(path, expected_request: Mapping, *, bundle_manifest=None) -> dict:
    """Validate one driver result against the exact request that produced it."""
    request = _closed(
        dict(expected_request), _REQUEST_FIELDS, "expected driver request"
    )
    result = _strict_json(path)
    required = set(_RESULT_BASE_FIELDS)
    mode = request["mode"]
    if mode in {"correctness", "combined"}:
        required.add("correctness")
    if mode in {"measure", "combined"}:
        required.add("measurements")
    result = _closed(result, required, "driver result")
    if result["protocol_version"] != RESULT_PROTOCOL:
        raise ValueError("driver result protocol_version is unsupported")
    expected_echoes = {
        "request_digest": request["request_digest"],
        "target_id": request["target_id"],
        "execution_id": request["execution_id"],
        "variant_digest": request["variant"]["digest"],
        "role": request["role"],
        "mode": request["mode"],
        "case_id": request["case"].get("id"),
        "driver_identity": request["driver_identity"],
    }
    for field, expected in expected_echoes.items():
        if result[field] != expected:
            raise ValueError(f"driver result {field} does not match request")
    manifest_files = None
    manifest_directories = None
    if bundle_manifest is not None:
        bundle_manifest = STORE._validated_manifest(bundle_manifest)
        if (
            bundle_manifest.get("source_kind") != "directory"
        ):
            raise ValueError("driver output bundle manifest is invalid")
        manifest_files = {}
        manifest_directories = set()
        for entry in bundle_manifest["entries"]:
            if type(entry) is not dict or entry.get("kind") not in {"file", "directory"}:
                raise ValueError("driver output bundle manifest is invalid")
            relative = "/".join(
                _artifact_relative_path(
                    entry.get("path"), "driver output bundle manifest path"
                )
            )
            if entry["kind"] == "file":
                if relative in manifest_files:
                    raise ValueError("driver output bundle contains duplicate files")
                manifest_files[relative] = entry
            else:
                manifest_directories.add(relative)
    artifacts = _validate_artifacts(result["artifacts"], path, manifest_files)
    if manifest_files is not None:
        expected_files = {"result.json"} | {
            artifact["relative_path"] for artifact in artifacts
        }
        expected_directories = set()
        for relative in expected_files:
            parent = PurePosixPath(relative).parent
            while str(parent) != ".":
                expected_directories.add(str(parent))
                parent = parent.parent
        if set(manifest_files) != expected_files or manifest_directories != expected_directories:
            raise ValueError("driver output bundle contains undeclared members")
        result_entry = manifest_files.get("result.json")
        if type(result_entry) is not dict or result_entry.get("kind") != "file":
            raise ValueError("driver output bundle omits result.json")
    cleanup = _closed(
        result["cleanup"], _CLEANUP_RESULT_FIELDS, "driver result cleanup"
    )
    if cleanup["status"] != "confirmed":
        raise ValueError("driver cleanup is not confirmed")
    if type(cleanup["live_tasks"]) is not list or cleanup["live_tasks"]:
        raise ValueError("driver reports live tasks after completion")
    normalized = {
        field: _json_copy(result[field], f"driver result {field}")
        for field in _RESULT_BASE_FIELDS
    }
    normalized["artifacts"] = artifacts
    normalized["environment"] = _validate_environment(result["environment"])
    if "correctness" in required:
        normalized["correctness"] = _validate_correctness(result["correctness"])
    if "measurements" in required:
        normalized["measurements"] = _validate_measurements(result["measurements"])
    return normalized


__all__ = [
    "DRIVER_PROTOCOL",
    "REQUEST_PROTOCOL",
    "RESULT_PROTOCOL",
    "build_argv",
    "build_driver_request",
    "evaluate_correctness",
    "materialize_target_inputs",
    "materialize_variant",
    "resolve_analysis_artifact",
    "resolve_profile_collection",
    "validate_driver",
    "validate_driver_result",
    "validate_variant",
    "verify_driver",
]
