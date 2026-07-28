#!/usr/bin/env python3
"""Strict contracts and orchestration entry point for workload optimization."""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import stat
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


CONTROL_SCHEMA_V1 = "cuda-workload-optimizer/control-v1"
CONTROL_SCHEMA_V2 = "cuda-workload-optimizer/control-v2"
CONTROL_SCHEMA = CONTROL_SCHEMA_V1
CHANGE_SCHEMA = "cuda-workload-optimizer/change-v1"
_BUDGETS = {"fast", "quick", "balanced", "thorough"}
_PROBE_KINDS = {
    "environment",
    "timeline",
    "framework",
    "cpu_data",
    "transfer",
    "communication",
    "io",
    "custom",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REVIEWER_PROVIDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_MAX_REVIEWERS = 8
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|cookie|credential|password|secret|token)($|[_-])",
    re.IGNORECASE,
)
_LOG_SECRET = re.compile(
    r'''(?i)(["']?\b[A-Z0-9_]{0,128}(?:API[_-]?KEY|AUTH|COOKIE|CREDENTIAL|PASSWORD|SECRET|TOKEN)[A-Z0-9_]{0,128}\b["']?\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\r\n,;}]+)'''
)
_DEFAULT_LOG_LIMIT = 64 * 1024
_OUTPUT_LIMIT = 1024 * 1024
_EVIDENCE_TERMINATION_RESERVE_SECONDS = 0.05
_EVIDENCE_ACCOUNTING_MARGIN_SECONDS = 0.05
_REVIEW_CALL_INTENT_SCHEMA = (
    "cuda-workload-optimizer/review-call-intent-v1"
)
_REVIEW_CALL_COMPLETE_SCHEMA = (
    "cuda-workload-optimizer/review-call-complete-v1"
)
_DIAGNOSIS_PUBLISH_INTENT_SCHEMA = (
    "cuda-workload-optimizer/diagnosis-publish-intent-v1"
)
_SAFE_ENV = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "TMPDIR",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
}
_DIAGNOSIS_MODULE = None
_REVIEWER_MODULE = None
_WORKLOAD_MODULE = None
_EVALUATE_MODULE = None
_BUDGET_MODULE = None
_READINESS_CONTRACT_MODULE = None
_READINESS_GATE_MODULE = None
_READINESS_IDENTITY_MODULE = None
_CHECK_ENV_MODULE = None
_ANALYSIS_EPOCH_MODULE = None
_EXECUTION_MAP_MODULE = None
_HYPOTHESIS_SPACE_MODULE = None
_EVIDENCE_SELECTOR_MODULE = None
_PERFORMANCE_MODEL_MODULE = None
_DIAGNOSTIC_DECISION_MODULE = None
_KNOWLEDGE_ADAPTER_MODULE = None
_ACTIVE_DIAGNOSIS_CONTRACT_SCHEMA = "cuda-optimizer/active-diagnosis-contract-v1"
_GLOBAL_SCAN_DRAFT_SCHEMA = "cuda-optimizer/global-scan-draft-v1"
_RUN_AUTHORIZATION_SCHEMA = "cuda-workload-optimizer/authorization-v1"
_RUN_AUTHORIZATION_INPUT_FIELDS = {
    "schema_version",
    "grant_id",
    "source",
    "interaction_mode",
    "max_controlled_seconds",
    "allowed_mutation_scopes",
    "max_risk",
    "max_stage",
}
_RUN_AUTHORIZATION_BINDING_FIELDS = {
    "control_digest",
    "workload_source_hash",
    "baseline_identity_digest",
    "baseline_environment_identity_digest",
    "analysis_epoch_sha256",
    "previous_grant_sha256",
    "sealed_at_epoch",
}
_RUN_AUTHORIZATION_STAGES = (
    "diagnosis",
    "static_review",
    "build_correctness",
    "short_paired",
    "profiler",
    "formal_paired",
)
_BUDGET_RUNTIME = {
    "quick": {
        "soft_target_seconds": 900,
        "hard_ceiling_seconds": 2700,
        "blocks": 3,
        "retries": 0,
        "bootstrap": 200,
    },
    "balanced": {
        "soft_target_seconds": 3600,
        "hard_ceiling_seconds": 10800,
        "blocks": 5,
        "retries": 1,
        "bootstrap": 1000,
    },
    "thorough": {
        "soft_target_seconds": 14400,
        "hard_ceiling_seconds": 36000,
        "blocks": 9,
        "retries": 2,
        "bootstrap": 5000,
    },
}


class ValidationError(ValueError):
    """Raised when a workload-controller contract is not closed and safe."""


class _CandidateStageTimeout(TimeoutError):
    """Signal a known, bounded candidate command timeout to the stage committer."""


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: os.PathLike[str] | str) -> dict:
    """Load one JSON object while rejecting duplicate keys and non-finite numbers."""
    source = Path(path)
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"cannot read JSON file {source}: {error}") from error
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda token: (_raise_invalid_number(token)),
        )
    except ValidationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValidationError(f"invalid JSON in {source}: {error}") from error
    if type(value) is not dict:
        raise ValidationError(f"JSON root must be an object: {source}")
    return value


def _raise_invalid_number(token: str):
    raise ValidationError(f"JSON number must be finite: {token}")


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{field} must be an object")
    return value


def _closed(value: dict, allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValidationError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _required(value: dict, required: set[str], field: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError(f"{field} is missing required fields: {', '.join(missing)}")


def _string(value: Any, field: str, *, max_length: int = 4096) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    if len(value) > max_length:
        raise ValidationError(f"{field} exceeds {max_length} characters")
    return value


def _identifier(value: Any, field: str) -> str:
    text = _string(value, field, max_length=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValidationError(f"{field} must be a safe identifier")
    return text


def _timeout(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not 1 <= number <= 3600:
        raise ValidationError(f"{field} must be between 1 and 3600 seconds")
    return number


def _argv(value: Any, field: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{field} argv must be a non-empty array")
    result = []
    for index, item in enumerate(value):
        result.append(_string(item, f"{field} argv[{index}]"))
    return result


def _absolute(value: Any, field: str) -> Path:
    text = _string(value, field)
    expanded = Path(os.path.expanduser(text))
    if not expanded.is_absolute():
        raise ValidationError(f"{field} must be an absolute path")
    return expanded.resolve(strict=False)


def _relative(value: Any, field: str) -> Path:
    text = _string(value, field)
    path = Path(text)
    if path.is_absolute() or text in {".", ".."} or ".." in path.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    normalized = Path(os.path.normpath(text))
    if str(normalized) in {"", ".", ".."} or ".." in normalized.parts:
        raise ValidationError(f"{field} must be a contained relative path")
    return normalized


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _json_copy(value: Any, field: str, *, reject_sensitive: bool = False) -> Any:
    if value is None or type(value) in {bool, str, int}:
        return copy.deepcopy(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{field} numbers must be finite")
        return value
    if type(value) is list:
        return [
            _json_copy(item, f"{field}[{index}]", reject_sensitive=reject_sensitive)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValidationError(f"{field} keys must be non-empty strings")
            if reject_sensitive and _SENSITIVE_KEY.search(key):
                raise ValidationError(f"{field} must not contain credentials: {key}")
            result[key] = _json_copy(
                item, f"{field}.{key}", reject_sensitive=reject_sensitive
            )
        return result
    raise ValidationError(f"{field} must contain JSON-compatible values")


def _string_list(value: Any, field: str, *, identifiers: bool = False) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{field} must be a non-empty array")
    result = []
    for index, item in enumerate(value):
        if identifiers:
            result.append(_identifier(item, f"{field}[{index}]"))
        else:
            result.append(_string(item, f"{field}[{index}]"))
    if len(set(result)) != len(result):
        raise ValidationError(f"{field} must not contain duplicates")
    return result


def _sha256(value: Any, field: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationError(f"{field} must be lowercase SHA-256")
    return value


def _validate_run_authorization_input(value: Mapping[str, Any]) -> dict:
    grant = _object(value, "run authorization")
    _closed(grant, _RUN_AUTHORIZATION_INPUT_FIELDS, "run authorization")
    _required(grant, _RUN_AUTHORIZATION_INPUT_FIELDS, "run authorization")
    if grant["schema_version"] != _RUN_AUTHORIZATION_SCHEMA:
        raise ValidationError("run authorization schema is invalid")
    grant_id = _identifier(grant["grant_id"], "run authorization grant_id")
    source = grant["source"]
    if source not in {"initial_request", "interactive_confirmation"}:
        raise ValidationError("run authorization source is invalid")
    interaction_mode = grant["interaction_mode"]
    if interaction_mode not in {"interactive", "unattended"}:
        raise ValidationError("run authorization interaction_mode is invalid")
    maximum = grant["max_controlled_seconds"]
    if (
        type(maximum) not in {int, float}
        or not math.isfinite(float(maximum))
        or float(maximum) <= 0
    ):
        raise ValidationError(
            "run authorization max_controlled_seconds must be positive and finite"
        )
    scopes = grant["allowed_mutation_scopes"]
    if (
        type(scopes) is not list
        or any(type(scope) is not str for scope in scopes)
        or len(scopes) != len(set(scopes))
    ):
        raise ValidationError(
            "run authorization allowed_mutation_scopes must be a unique list"
        )
    if any(scope not in {"project", "isolated_environment"} for scope in scopes):
        raise ValidationError("run authorization mutation scope is invalid")
    max_risk = grant["max_risk"]
    if max_risk not in {"none", "low", "medium", "high"}:
        raise ValidationError("run authorization max_risk is invalid")
    max_stage = grant["max_stage"]
    if max_stage not in _RUN_AUTHORIZATION_STAGES:
        raise ValidationError("run authorization max_stage is invalid")
    return {
        "schema_version": _RUN_AUTHORIZATION_SCHEMA,
        "grant_id": grant_id,
        "source": source,
        "interaction_mode": interaction_mode,
        "max_controlled_seconds": float(maximum),
        "allowed_mutation_scopes": copy.deepcopy(scopes),
        "max_risk": max_risk,
        "max_stage": max_stage,
    }


def _validate_run_authorization_record(
    value: Mapping[str, Any],
    label: str = "sealed run authorization",
) -> dict:
    record = _object(value, label)
    fields = _RUN_AUTHORIZATION_INPUT_FIELDS | _RUN_AUTHORIZATION_BINDING_FIELDS
    _closed(record, fields, label)
    _required(record, fields, label)
    normalized = _validate_run_authorization_input(
        {field: record[field] for field in _RUN_AUTHORIZATION_INPUT_FIELDS}
    )
    for field in (
        "control_digest",
        "workload_source_hash",
        "baseline_identity_digest",
        "analysis_epoch_sha256",
    ):
        _sha256(record[field], f"{label}.{field}")
    environment_digest = record["baseline_environment_identity_digest"]
    if environment_digest is not None:
        _sha256(environment_digest, f"{label}.baseline_environment_identity_digest")
    previous = record["previous_grant_sha256"]
    if previous is not None:
        _sha256(previous, f"{label}.previous_grant_sha256")
    sealed_at = record["sealed_at_epoch"]
    if (
        type(sealed_at) not in {int, float}
        or not math.isfinite(float(sealed_at))
        or float(sealed_at) < 0
    ):
        raise ValidationError(f"{label}.sealed_at_epoch is invalid")
    return {
        **normalized,
        "control_digest": record["control_digest"],
        "workload_source_hash": record["workload_source_hash"],
        "baseline_identity_digest": record["baseline_identity_digest"],
        "baseline_environment_identity_digest": environment_digest,
        "analysis_epoch_sha256": record["analysis_epoch_sha256"],
        "previous_grant_sha256": previous,
        "sealed_at_epoch": float(sealed_at),
    }


def _validate_active_diagnosis_contract(value: Mapping[str, Any]) -> dict:
    contract = _object(value, "analysis_contract")
    fields = {
        "schema_version",
        "global_scan_probe_id",
        "adapter_path",
        "analysis_policy_sha256",
        "minimum_effect_us",
        "source",
        "actions",
        "selection_policy",
    }
    _closed(contract, fields, "analysis_contract")
    _required(contract, fields - {"minimum_effect_us"}, "analysis_contract")
    if contract["schema_version"] != _ACTIVE_DIAGNOSIS_CONTRACT_SCHEMA:
        raise ValidationError(
            f"analysis_contract.schema_version must be {_ACTIVE_DIAGNOSIS_CONTRACT_SCHEMA}"
        )
    _identifier(contract["global_scan_probe_id"], "analysis_contract.global_scan_probe_id")
    _absolute(contract["adapter_path"], "analysis_contract.adapter_path")
    _sha256(
        contract["analysis_policy_sha256"],
        "analysis_contract.analysis_policy_sha256",
    )
    minimum_effect_us = contract.get("minimum_effect_us", 1.0)
    if (
        type(minimum_effect_us) not in {int, float}
        or not math.isfinite(float(minimum_effect_us))
        or float(minimum_effect_us) <= 0
    ):
        raise ValidationError("analysis_contract.minimum_effect_us must be positive and finite")
    source = _object(contract["source"], "analysis_contract.source")
    source_fields = {
        "profiler",
        "profiler_version",
        "export_schema",
        "adapter_id",
        "adapter_version",
        "adapter_sha256",
    }
    _closed(source, source_fields, "analysis_contract.source")
    _required(source, source_fields, "analysis_contract.source")
    if source["profiler"] not in {"nsys", "pytorch", "perfetto", "custom"}:
        raise ValidationError("analysis_contract.source.profiler is unsupported")
    for field in ("profiler_version", "export_schema", "adapter_version"):
        _string(source[field], f"analysis_contract.source.{field}", max_length=256)
    _identifier(source["adapter_id"], "analysis_contract.source.adapter_id")
    _sha256(source["adapter_sha256"], "analysis_contract.source.adapter_sha256")
    if type(contract["actions"]) is not list or not contract["actions"]:
        raise ValidationError("analysis_contract.actions must be a non-empty array")
    actions = []
    action_ids = set()
    for index, raw in enumerate(contract["actions"]):
        action = _object(raw, f"analysis_contract.actions[{index}]")
        action_fields = {
            "action_id",
            "adapter_path",
            "adapter_sha256",
            "argv",
            "timeout_seconds",
            "cost_bound",
        }
        _closed(action, action_fields, f"analysis_contract.actions[{index}]")
        _required(
            action,
            action_fields - {"cost_bound"},
            f"analysis_contract.actions[{index}]",
        )
        action_id = _identifier(
            action["action_id"], f"analysis_contract.actions[{index}].action_id"
        )
        if action_id in action_ids:
            raise ValidationError("analysis_contract action ids must be unique")
        action_ids.add(action_id)
        normalized_action = {
            "action_id": action_id,
            "adapter_path": str(
                _absolute(
                    action["adapter_path"],
                    f"analysis_contract.actions[{index}].adapter_path",
                )
            ),
            "adapter_sha256": _sha256(
                action["adapter_sha256"],
                f"analysis_contract.actions[{index}].adapter_sha256",
            ),
            "argv": _argv(
                action["argv"], f"analysis_contract.actions[{index}]"
            ),
            "timeout_seconds": _timeout(
                action["timeout_seconds"],
                f"analysis_contract.actions[{index}].timeout_seconds",
            ),
        }
        if "cost_bound" in action:
            bound = _object(
                action["cost_bound"],
                f"analysis_contract.actions[{index}].cost_bound",
            )
            bound_fields = {"p50_seconds", "p90_seconds", "basis"}
            _closed(
                bound,
                bound_fields,
                f"analysis_contract.actions[{index}].cost_bound",
            )
            _required(
                bound,
                bound_fields,
                f"analysis_contract.actions[{index}].cost_bound",
            )
            p50 = bound["p50_seconds"]
            p90 = bound["p90_seconds"]
            if (
                type(p50) not in {int, float}
                or type(p90) not in {int, float}
                or not math.isfinite(float(p50))
                or not math.isfinite(float(p90))
                or float(p50) <= 0
                or float(p90) <= 0
                or float(p50) > float(p90)
            ):
                raise ValidationError(
                    "analysis action cost_bound requires positive P50 <= P90"
                )
            if bound["basis"] != "user_authorized_upper_bound":
                raise ValidationError(
                    "analysis action cost_bound basis must be user_authorized_upper_bound"
                )
            normalized_action["cost_bound"] = {
                "p50_seconds": float(p50),
                "p90_seconds": float(p90),
                "basis": bound["basis"],
            }
        actions.append(normalized_action)
    actions.sort(key=lambda item: item["action_id"])
    try:
        policy = _load_evidence_selector_module()._validate_policy(
            contract["selection_policy"]
        )
    except ValueError as error:
        raise ValidationError(f"invalid analysis selection policy: {error}") from error
    normalized = copy.deepcopy(dict(contract))
    normalized["minimum_effect_us"] = float(minimum_effect_us)
    normalized["actions"] = actions
    # Capability admission is Controller-owned and is rebuilt from the current
    # readiness report when the diagnosis context is created.
    policy["available_capability_ids"] = []
    normalized["selection_policy"] = policy
    return normalized


def validate_control_manifest(value: Mapping[str, Any], source_path=None) -> dict:
    """Validate and detach the closed v2.4 controller manifest."""
    control = _object(value, "control")
    allowed = {
        "schema_version",
        "project_root",
        "workload_manifest",
        "baseline_candidate",
        "budget",
        "evaluation_gate",
        "mutation",
        "probes",
        "reviewer",
        "reviewers",
        "readiness_contract",
        "analysis_contract",
    }
    schema_version = control.get("schema_version")
    if schema_version not in {CONTROL_SCHEMA_V1, CONTROL_SCHEMA_V2}:
        raise ValidationError(
            f"schema_version must be {CONTROL_SCHEMA_V1} or {CONTROL_SCHEMA_V2}"
        )
    required = allowed - {
        "reviewer",
        "reviewers",
        "evaluation_gate",
        "readiness_contract",
        "analysis_contract",
    }
    if schema_version == CONTROL_SCHEMA_V2:
        required.add("readiness_contract")
    _closed(control, allowed, "control")
    _required(control, required, "control")
    if schema_version == CONTROL_SCHEMA_V1 and "readiness_contract" in control:
        raise ValidationError("control-v1 must not contain readiness_contract")
    if schema_version == CONTROL_SCHEMA_V1 and "analysis_contract" in control:
        raise ValidationError("control-v1 must not contain analysis_contract")

    project_root = _absolute(control["project_root"], "project_root")
    workload_manifest = _absolute(
        control["workload_manifest"], "workload_manifest"
    )
    if not _is_within(workload_manifest, project_root):
        raise ValidationError("workload_manifest must be inside project_root")
    if schema_version == CONTROL_SCHEMA_V2:
        readiness_contract = _absolute(
            control["readiness_contract"], "readiness_contract"
        )
        if not _is_within(readiness_contract, project_root):
            raise ValidationError(
                "readiness_contract must be inside project_root"
            )
        if "analysis_contract" in control:
            analysis_contract = _absolute(
                control["analysis_contract"], "analysis_contract"
            )
            if not _is_within(analysis_contract, project_root):
                raise ValidationError(
                    "analysis_contract must be inside project_root"
                )
    baseline = _object(control["baseline_candidate"], "baseline_candidate")
    if not baseline:
        raise ValidationError("baseline_candidate must not be empty")
    _json_copy(baseline, "baseline_candidate", reject_sensitive=True)
    if control["budget"] not in _BUDGETS:
        raise ValidationError("budget must be quick, balanced, or thorough")
    if control.get("evaluation_gate", "promotion") not in {
        "promotion",
        "reject_only",
    }:
        raise ValidationError(
            "evaluation_gate must be promotion or reject_only"
        )

    mutation = _object(control["mutation"], "mutation")
    mutation_fields = {"project_paths", "environment_root", "host_policy"}
    _closed(mutation, mutation_fields, "mutation")
    _required(mutation, mutation_fields, "mutation")
    project_paths = _string_list(mutation["project_paths"], "project_paths")
    normalized_roots = []
    for index, item in enumerate(project_paths):
        relative = _relative(item, f"project_paths[{index}]")
        candidate = (project_root / relative).resolve(strict=False)
        if not _is_within(candidate, project_root):
            raise ValidationError(f"project_paths[{index}] escapes project_root")
        normalized_roots.append(relative)
    for index, root in enumerate(normalized_roots):
        for other in normalized_roots[index + 1 :]:
            if root == other or _is_within(root, other) or _is_within(other, root):
                raise ValidationError("project_paths must not overlap")
    environment_root = _absolute(mutation["environment_root"], "environment_root")
    protected_environment_roots = {
        Path(path).resolve(strict=False)
        for path in (
            "/System",
            "/Library",
            "/Applications",
            "/bin",
            "/sbin",
            "/usr",
            "/etc",
            "/private/etc",
        )
    }
    if (
        _is_within(environment_root, project_root)
        or _is_within(project_root, environment_root)
        or environment_root == Path("/")
        or any(_is_within(environment_root, root) for root in protected_environment_roots)
    ):
        raise ValidationError(
            "environment_root must be isolated from project_root and host system roots"
        )
    allowed_workspace_roots = [
        Path(tempfile.gettempdir()).resolve(strict=False),
        Path("/workspace").resolve(strict=False),
        Path("/data").resolve(strict=False),
    ]
    if os.geteuid() != 0:
        allowed_workspace_roots.append(Path.home().resolve(strict=False))
    if not any(
        environment_root != root and _is_within(environment_root, root)
        for root in allowed_workspace_roots
    ):
        raise ValidationError(
            "environment_root must be below a user workspace, data, or temporary root"
        )
    if mutation["host_policy"] != "recommend_only":
        raise ValidationError("host_policy must be recommend_only")

    probes = control["probes"]
    if type(probes) is not list or not probes:
        raise ValidationError("probes must be a non-empty array")
    probe_ids = set()
    for index, item in enumerate(probes):
        probe = _object(item, f"probes[{index}]")
        fields = {"id", "kind", "argv", "timeout_seconds"}
        _closed(probe, fields, f"probes[{index}]")
        _required(probe, fields, f"probes[{index}]")
        probe_id = _identifier(probe["id"], f"probes[{index}].id")
        if probe_id in probe_ids:
            raise ValidationError("probe ids must be unique")
        probe_ids.add(probe_id)
        if probe["kind"] not in _PROBE_KINDS:
            raise ValidationError(f"probes[{index}].kind is unsupported")
        _argv(probe["argv"], f"probes[{index}]")
        _timeout(probe["timeout_seconds"], f"probes[{index}].timeout_seconds")

    if "reviewer" in control and "reviewers" in control:
        raise ValidationError("reviewer and reviewers are mutually exclusive")
    reviewer = control.get("reviewer")
    if reviewer is not None:
        reviewer = _object(reviewer, "reviewer")
        fields = {"argv", "timeout_seconds", "include_diff"}
        _closed(reviewer, fields, "reviewer")
        _required(reviewer, {"argv", "timeout_seconds"}, "reviewer")
        _argv(reviewer["argv"], "reviewer")
        _timeout(reviewer["timeout_seconds"], "reviewer.timeout_seconds")
        if "include_diff" in reviewer and type(reviewer["include_diff"]) is not bool:
            raise ValidationError("reviewer.include_diff must be a boolean")
    reviewers = control.get("reviewers")
    if reviewers is not None:
        if (
            type(reviewers) is not list
            or not reviewers
            or len(reviewers) > _MAX_REVIEWERS
        ):
            raise ValidationError(
                f"reviewers must contain between 1 and {_MAX_REVIEWERS} entries"
            )
        reviewer_keys = set()
        for index, item in enumerate(reviewers):
            reviewer = _object(item, f"reviewers[{index}]")
            fields = {
                "provider",
                "underlying_model",
                "argv",
                "timeout_seconds",
                "include_diff",
            }
            _closed(reviewer, fields, f"reviewers[{index}]")
            _required(
                reviewer,
                {"provider", "argv", "timeout_seconds"},
                f"reviewers[{index}]",
            )
            provider = reviewer["provider"]
            if (
                type(provider) is not str
                or _REVIEWER_PROVIDER.fullmatch(provider) is None
            ):
                raise ValidationError(
                    f"reviewers[{index}].provider must be a safe 64-character identifier"
                )
            model = reviewer.get("underlying_model", "unknown")
            if (
                type(model) is not str
                or _REVIEWER_PROVIDER.fullmatch(model) is None
            ):
                raise ValidationError(
                    f"reviewers[{index}].underlying_model must be a known-safe identifier, auto, or unknown"
                )
            reviewer_key = (provider.lower(), model.lower())
            if reviewer_key in reviewer_keys:
                raise ValidationError("reviewer provider/model pairs must not be duplicate")
            reviewer_keys.add(reviewer_key)
            _argv(reviewer["argv"], f"reviewers[{index}]")
            _timeout(
                reviewer["timeout_seconds"],
                f"reviewers[{index}].timeout_seconds",
            )
            if (
                "include_diff" in reviewer
                and type(reviewer["include_diff"]) is not bool
            ):
                raise ValidationError(
                    f"reviewers[{index}].include_diff must be a boolean"
                )

    normalized = _json_copy(control, "control", reject_sensitive=True)
    if normalized["budget"] == "fast":
        normalized["budget"] = "quick"
    return normalized


def validate_change_set(value: Mapping[str, Any], control: Mapping[str, Any]) -> dict:
    """Validate a bounded project or isolated-environment ChangeSet."""
    change = _object(value, "change_set")
    fields = {
        "schema_version",
        "id",
        "hypothesis",
        "diagnosis_ids",
        "scope",
        "risk",
        "candidate",
        "paths",
        "commands",
        "rollback",
        "expected_metrics",
    }
    _closed(change, fields, "change_set")
    _required(change, fields, "change_set")
    if change["schema_version"] != CHANGE_SCHEMA:
        raise ValidationError(f"change_set.schema_version must be {CHANGE_SCHEMA}")
    _identifier(change["id"], "change_set.id")
    _string(change["hypothesis"], "change_set.hypothesis")
    _string_list(change["diagnosis_ids"], "change_set.diagnosis_ids")
    if change["scope"] not in {"project", "isolated_environment"}:
        raise ValidationError("change_set.scope must be project or isolated_environment")
    if change["risk"] not in {"none", "low", "medium", "high"}:
        raise ValidationError("change_set.risk must be none, low, medium, or high")
    candidate = _object(change["candidate"], "change_set.candidate")
    if not candidate:
        raise ValidationError("change_set.candidate must not be empty")
    if "_cuda_optimizer_identity_digest" in candidate:
        raise ValidationError("change_set.candidate uses a reserved identity field")
    _json_copy(candidate, "change_set.candidate", reject_sensitive=True)
    runtime = _BUDGET_RUNTIME[control["budget"]]
    gate_contract = {
        "soft_target_seconds": runtime["soft_target_seconds"],
        "hard_ceiling_seconds": runtime["hard_ceiling_seconds"],
        "minimum_effect": {"mechanism_us": 1.0, "service_pct": 0.5},
    }
    try:
        clean_candidate = _load_budget_module().validate_candidate_declaration(
            candidate, gate_contract
        )
    except ValueError as error:
        raise ValidationError(f"candidate declaration is invalid: {error}") from error
    if clean_candidate["claim_layer"] != "workload":
        raise ValidationError(
            "workload Controller accepts only workload claims; use the kernel "
            "or serving workflow for other units and evidence"
        )

    paths = _string_list(change["paths"], "change_set.paths")
    relative_paths = [
        _relative(item, f"change_set.paths[{index}]")
        for index, item in enumerate(paths)
    ]
    if change["scope"] == "project":
        allowed_roots = [
            _relative(item, "control.mutation.project_paths")
            for item in control["mutation"]["project_paths"]
        ]
        for index, path in enumerate(relative_paths):
            if not any(path == root or _is_within(path, root) for root in allowed_roots):
                raise ValidationError(
                    f"change_set.paths[{index}] is outside declared project_paths"
                )

    commands = change["commands"]
    if type(commands) is not list:
        raise ValidationError("change_set.commands must be an array of argv arrays")
    if commands:
        raise ValidationError(
            "change_set.commands must be empty; correctness runs through the workload adapter"
        )
    for index, command in enumerate(commands):
        if type(command) is not list:
            raise ValidationError("change_set.commands must contain argv arrays")
        _argv(command, f"change_set.commands[{index}]")
    if change["rollback"] != "restore_frozen_snapshot":
        raise ValidationError("change_set.rollback must be restore_frozen_snapshot")
    _string_list(change["expected_metrics"], "change_set.expected_metrics")
    return _json_copy(change, "change_set", reject_sensitive=True)


def _load_diagnosis_module():
    global _DIAGNOSIS_MODULE
    if _DIAGNOSIS_MODULE is not None:
        return _DIAGNOSIS_MODULE
    path = Path(__file__).with_name("workload_diagnosis.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_workload_diagnosis_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workload diagnosis module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    _DIAGNOSIS_MODULE = module
    return module


def _load_reviewer_module():
    global _REVIEWER_MODULE
    if _REVIEWER_MODULE is not None:
        return _REVIEWER_MODULE
    path = Path(__file__).with_name("workload_reviewer.py")
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_workload_reviewer_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workload reviewer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    _REVIEWER_MODULE = module
    return module


def _load_sibling_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load controller dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_workload_module():
    global _WORKLOAD_MODULE
    if _WORKLOAD_MODULE is None:
        script_dir = str(Path(__file__).resolve().parent)
        inserted = script_dir not in sys.path
        if inserted:
            sys.path.insert(0, script_dir)
        try:
            _WORKLOAD_MODULE = _load_sibling_module(
                "workload_adapter.py", "workload_adapter"
            )
        finally:
            if inserted:
                sys.path.remove(script_dir)
    return _WORKLOAD_MODULE


def _load_evaluate_module():
    global _EVALUATE_MODULE
    if _EVALUATE_MODULE is None:
        # workload_evaluate imports workload_adapter and paired_stats by module
        # name, so expose this directory only while loading the trusted sibling.
        script_dir = str(Path(__file__).resolve().parent)
        inserted = script_dir not in sys.path
        if inserted:
            sys.path.insert(0, script_dir)
        try:
            _EVALUATE_MODULE = _load_sibling_module(
                "workload_evaluate.py", "cuda_optimizer_workload_evaluate_controller"
            )
        finally:
            if inserted:
                sys.path.remove(script_dir)
    return _EVALUATE_MODULE


def _load_budget_module():
    global _BUDGET_MODULE
    if _BUDGET_MODULE is None:
        _BUDGET_MODULE = _load_sibling_module(
            "budget.py", "cuda_optimizer_budget_controller"
        )
    return _BUDGET_MODULE


def _load_readiness_contract_module():
    global _READINESS_CONTRACT_MODULE
    if _READINESS_CONTRACT_MODULE is None:
        _READINESS_CONTRACT_MODULE = _load_sibling_module(
            "readiness_contract.py",
            "cuda_optimizer_readiness_contract_controller",
        )
    return _READINESS_CONTRACT_MODULE


def _load_readiness_gate_module():
    global _READINESS_GATE_MODULE
    if _READINESS_GATE_MODULE is None:
        _READINESS_GATE_MODULE = _load_sibling_module(
            "readiness_gate.py",
            "cuda_optimizer_readiness_gate_controller",
        )
    return _READINESS_GATE_MODULE


def _load_readiness_identity_module():
    global _READINESS_IDENTITY_MODULE
    if _READINESS_IDENTITY_MODULE is None:
        _READINESS_IDENTITY_MODULE = _load_sibling_module(
            "readiness_identity.py",
            "cuda_optimizer_readiness_identity_controller",
        )
    return _READINESS_IDENTITY_MODULE


def _load_check_env_module():
    global _CHECK_ENV_MODULE
    if _CHECK_ENV_MODULE is None:
        _CHECK_ENV_MODULE = _load_sibling_module(
            "check_env.py", "cuda_optimizer_check_env_controller"
        )
    return _CHECK_ENV_MODULE


def _load_analysis_epoch_module():
    global _ANALYSIS_EPOCH_MODULE
    if _ANALYSIS_EPOCH_MODULE is None:
        _ANALYSIS_EPOCH_MODULE = _load_sibling_module(
            "analysis_epoch.py", "cuda_optimizer_analysis_epoch_controller"
        )
    return _ANALYSIS_EPOCH_MODULE


def _load_execution_map_module():
    global _EXECUTION_MAP_MODULE
    if _EXECUTION_MAP_MODULE is None:
        _EXECUTION_MAP_MODULE = _load_sibling_module(
            "execution_map.py", "cuda_optimizer_execution_map_controller"
        )
    return _EXECUTION_MAP_MODULE


def _load_hypothesis_space_module():
    global _HYPOTHESIS_SPACE_MODULE
    if _HYPOTHESIS_SPACE_MODULE is None:
        _HYPOTHESIS_SPACE_MODULE = _load_sibling_module(
            "hypothesis_space.py", "cuda_optimizer_hypothesis_space_controller"
        )
    return _HYPOTHESIS_SPACE_MODULE


def _load_evidence_selector_module():
    global _EVIDENCE_SELECTOR_MODULE
    if _EVIDENCE_SELECTOR_MODULE is None:
        _EVIDENCE_SELECTOR_MODULE = _load_sibling_module(
            "evidence_selector.py", "cuda_optimizer_evidence_selector_controller"
        )
    return _EVIDENCE_SELECTOR_MODULE


def _load_performance_model_module():
    global _PERFORMANCE_MODEL_MODULE
    if _PERFORMANCE_MODEL_MODULE is None:
        _PERFORMANCE_MODEL_MODULE = _load_sibling_module(
            "performance_model.py", "cuda_optimizer_performance_model_controller"
        )
    return _PERFORMANCE_MODEL_MODULE


def _load_diagnostic_decision_module():
    global _DIAGNOSTIC_DECISION_MODULE
    if _DIAGNOSTIC_DECISION_MODULE is None:
        _DIAGNOSTIC_DECISION_MODULE = _load_sibling_module(
            "diagnostic_decision.py", "cuda_optimizer_diagnostic_decision_controller"
        )
    return _DIAGNOSTIC_DECISION_MODULE


def _load_diagnostic_knowledge_module():
    return _load_sibling_module(
        "diagnostic_knowledge.py", "cuda_optimizer_diagnostic_knowledge_controller"
    )


def _load_knowledge_adapter_module():
    global _KNOWLEDGE_ADAPTER_MODULE
    if _KNOWLEDGE_ADAPTER_MODULE is None:
        _KNOWLEDGE_ADAPTER_MODULE = _load_sibling_module(
            "knowledge_adapter.py", "cuda_optimizer_knowledge_adapter_controller"
        )
    return _KNOWLEDGE_ADAPTER_MODULE


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _run_lock(run_root: Path):
    """Serialize initialization and active-diagnosis mutations for one run."""
    lock_path = run_root / ".workload-controller.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValidationError("active diagnosis lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class _BoundedLog:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.value = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        available = max(0, self.limit - len(self.value))
        self.value.extend(chunk[:available])
        if len(chunk) > available:
            self.truncated = True

    def text(self) -> str:
        decoded = bytes(self.value).decode("utf-8", errors="replace")
        return decoded + ("...[truncated]" if self.truncated else "")


def _drain(stream, capture: _BoundedLog) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            capture.append(chunk)
    finally:
        stream.close()


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(
    process: Any,
    *,
    term_deadline_monotonic: float,
    deadline_monotonic: float,
) -> None:
    process_group = process.pid
    term_deadline = float(term_deadline_monotonic)
    deadline = float(deadline_monotonic)
    if (
        not math.isfinite(term_deadline)
        or not math.isfinite(deadline)
        or term_deadline > deadline
    ):
        raise ValidationError(
            "process termination deadlines are invalid"
        )

    try:
        os.killpg(process_group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    while _process_group_exists(process_group):
        remaining = term_deadline - time.monotonic()
        if remaining <= 0.0:
            break
        process.poll()
        if not _process_group_exists(process_group):
            break
        time.sleep(min(0.01, remaining))
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    while True:
        process.poll()
        group_exists = _process_group_exists(process_group)
        if process.returncode is not None and not group_exists:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise ValidationError(
                "process group termination exceeded the safety deadline"
            )
        if process.returncode is None:
            try:
                process.wait(timeout=min(0.01, remaining))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.01, remaining))


def _wait_process_with_heartbeats(
    process: Any,
    *,
    timeout_seconds: float,
    label: str,
    heartbeat_interval_seconds: float = 30.0,
    termination_grace_seconds: float = 0.25,
    accounting_margin_seconds: float = 0.05,
    event_sink=None,
) -> tuple[int | None, bool, float, str]:
    """Wait visibly and preserve the existing process-group hard stop."""
    timeout = float(timeout_seconds)
    interval = float(heartbeat_interval_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValidationError("process timeout must be a positive finite number")
    if not math.isfinite(interval) or interval <= 0:
        raise ValidationError("heartbeat interval must be a positive finite number")
    termination_grace = float(termination_grace_seconds)
    if not math.isfinite(termination_grace) or termination_grace <= 0:
        raise ValidationError(
            "process termination grace must be a positive finite number"
        )
    accounting_margin = float(accounting_margin_seconds)
    if not math.isfinite(accounting_margin) or accounting_margin <= 0:
        raise ValidationError(
            "process accounting margin must be a positive finite number"
        )
    task = _identifier(label, "heartbeat label")

    def emit(event: Mapping[str, Any]) -> None:
        if event_sink is not None:
            if not callable(event_sink):
                raise ValidationError("event_sink must be callable")
            event_sink(copy.deepcopy(dict(event)))
        print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)

    started = time.monotonic()
    timed_out = False
    while True:
        elapsed = max(0.0, time.monotonic() - started)
        remaining = timeout - elapsed
        if remaining <= 0:
            timed_out = True
            term_deadline = time.monotonic() + termination_grace
            _stop_group(
                process,
                term_deadline_monotonic=term_deadline,
                deadline_monotonic=term_deadline + accounting_margin,
            )
            break
        try:
            process.wait(timeout=min(interval, remaining))
            if _process_group_exists(process.pid):
                term_deadline = time.monotonic() + termination_grace
                _stop_group(
                    process,
                    term_deadline_monotonic=term_deadline,
                    deadline_monotonic=term_deadline + accounting_margin,
                )
            break
        except subprocess.TimeoutExpired:
            elapsed = max(0.0, time.monotonic() - started)
            if elapsed >= timeout:
                timed_out = True
                term_deadline = time.monotonic() + termination_grace
                _stop_group(
                    process,
                    term_deadline_monotonic=term_deadline,
                    deadline_monotonic=term_deadline + accounting_margin,
                )
                break
            emit(
                {
                    "event": "heartbeat",
                    "task": task,
                    "elapsed_seconds": elapsed,
                }
            )
    elapsed = max(0.0, time.monotonic() - started)
    reason = (
        "hard_deadline_exceeded"
        if timed_out
        else "completed"
        if process.returncode == 0
        else "command_failed"
    )
    emit(
        {
            "event": "terminal",
            "task": task,
            "elapsed_seconds": elapsed,
            "stop_reason": reason,
        }
    )
    return process.returncode, timed_out, elapsed, reason


def _is_secret_name(name: str) -> bool:
    return _SENSITIVE_KEY.search(name) is not None


def _probe_environment(overrides: Mapping[str, str]) -> tuple[dict, tuple[str, ...]]:
    inherited = dict(os.environ)
    explicit = {
        name.strip()
        for name in inherited.get("CUDA_OPTIMIZER_PASS_ENV", "").split(",")
        if name.strip()
    }
    allowed = _SAFE_ENV | explicit
    environment = {
        name: value
        for name, value in inherited.items()
        if name in allowed and not _is_secret_name(name)
    }
    environment.update(overrides)
    secrets = tuple(
        value
        for name, value in inherited.items()
        if _is_secret_name(name) and value
    )
    return environment, secrets


def _redact_log(value: str, secrets: Sequence[str]) -> str:
    result = _LOG_SECRET.sub(lambda match: f"{match.group(1)}[REDACTED]", value)
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _failure_probe(probe: Mapping[str, Any], status: str, issue_id: str, message: str) -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/probe-v1",
        "probe_id": probe["id"],
        "kind": probe["kind"],
        "status": status,
        "metrics": {},
        "issues": [
            {
                "id": issue_id,
                "category": "environment",
                "severity": "error",
                "message": message,
            }
        ],
        "artifacts": [],
    }


def _read_probe_output(path: Path) -> dict:
    try:
        info = path.stat()
    except FileNotFoundError as error:
        raise ValidationError("probe did not create CUDA_OPTIMIZER_OUTPUT") from error
    if not path.is_file() or info.st_size > _OUTPUT_LIMIT:
        raise ValidationError(f"probe output must be a regular file under {_OUTPUT_LIMIT} bytes")
    return load_json_object(path)


def _run_probe_unchecked(
    probe: Mapping[str, Any],
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    *,
    log_limit_bytes: int = _DEFAULT_LOG_LIMIT,
    deadline_epoch: float | None = None,
) -> dict:
    """Execute one probe after the owning Controller boundary admitted it."""
    if isinstance(log_limit_bytes, bool) or not isinstance(log_limit_bytes, int):
        raise ValidationError("log_limit_bytes must be a positive integer")
    if log_limit_bytes <= 0 or log_limit_bytes > _OUTPUT_LIMIT:
        raise ValidationError("log_limit_bytes must be between 1 and 1048576")
    normalized_control = validate_control_manifest(control)
    matching = [item for item in normalized_control["probes"] if item["id"] == probe.get("id")]
    if len(matching) != 1 or matching[0] != probe:
        raise ValidationError("probe must exactly match one validated control probe")
    selected = matching[0]
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    actual_timeout = float(selected["timeout_seconds"])
    if deadline_epoch is not None:
        remaining = float(deadline_epoch) - time.time()
        if remaining <= 0:
            raise ValidationError("workload optimization budget deadline has expired")
        actual_timeout = min(actual_timeout, remaining)
    probes_dir = run_root / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    output_path = probes_dir / f".{selected['id']}.output.json"
    active_output_path = None
    environment_overrides = {
        "CUDA_OPTIMIZER_OUTPUT": str(output_path),
        "CUDA_OPTIMIZER_RUN_DIR": str(run_root),
        "CUDA_OPTIMIZER_PROJECT_ROOT": normalized_control["project_root"],
    }
    if "analysis_contract" in normalized_control:
        frozen_analysis_contract = (
            run_root / "active_diagnosis" / "analysis_contract.json"
        )
        contract_path = (
            frozen_analysis_contract
            if frozen_analysis_contract.is_file()
            else Path(normalized_control["analysis_contract"])
        )
        active_contract = _validate_active_diagnosis_contract(
            load_json_object(contract_path)
        )
        if selected["id"] == active_contract["global_scan_probe_id"]:
            active_output_path = probes_dir / ".active-diagnosis.output.json"
            environment_overrides["CUDA_OPTIMIZER_ACTIVE_DIAGNOSIS_OUTPUT"] = str(
                active_output_path
            )
            if frozen_analysis_contract.is_file():
                state = read_run_state(run_root)
                bindings = _load_frozen_execution_bindings(run_root, state)
                _verify_adapter_execution_binding(
                    bindings["global_scan"],
                    Path(active_contract["adapter_path"]),
                    selected["argv"],
                    "analysis_contract global scan",
                )
    for transient in (output_path, active_output_path):
        if transient is None:
            continue
        try:
            transient.unlink()
        except FileNotFoundError:
            pass
    environment, secret_values = _probe_environment(environment_overrides)
    stdout = _BoundedLog(log_limit_bytes)
    stderr = _BoundedLog(log_limit_bytes)
    started = time.monotonic()
    timed_out = False
    exit_code = None
    stop_reason = "launch_failed"
    events: list[dict] = []
    process = None
    try:
        process = subprocess.Popen(
            selected["argv"],
            cwd=normalized_control["project_root"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        readers = [
            threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()
        exit_code, timed_out, _elapsed, stop_reason = _wait_process_with_heartbeats(
            process,
            timeout_seconds=actual_timeout,
            label=f"probe-{selected['id']}",
            event_sink=events.append,
        )
        for reader in readers:
            reader.join(timeout=1)
    except FileNotFoundError as error:
        result = _failure_probe(
            selected, "unavailable", "environment:probe-unavailable", str(error)
        )
    except OSError as error:
        result = _failure_probe(
            selected, "failed", "environment:probe-launch", str(error)
        )
    else:
        if timed_out:
            result = _failure_probe(
                selected,
                "unavailable",
                "environment:probe-timeout",
                f"probe exceeded {actual_timeout:.6g} seconds",
            )
        elif exit_code != 0:
            result = _failure_probe(
                selected,
                "failed",
                "environment:probe-exit",
                f"probe exited with status {exit_code}",
            )
        else:
            try:
                result = _load_diagnosis_module().validate_probe(
                    _read_probe_output(output_path)
                )
                if result["probe_id"] != selected["id"] or result["kind"] != selected["kind"]:
                    raise ValidationError("probe output identity does not match control")
            except (ValidationError, ValueError) as error:
                result = _failure_probe(
                    selected,
                    "failed",
                    "environment:probe-output",
                    f"invalid normalized probe output: {error}",
                )
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass

    if active_output_path is not None and result.get("status") in {"ok", "degraded"}:
        try:
            active_draft = _read_probe_output(active_output_path)
            _atomic_json(
                run_root / "active_diagnosis" / "global_scan.json",
                active_draft,
            )
        except (ValidationError, ValueError) as error:
            result = _failure_probe(
                selected,
                "failed",
                "environment:active-diagnosis-output",
                f"invalid active diagnosis output: {error}",
            )
    if active_output_path is not None:
        try:
            active_output_path.unlink()
        except FileNotFoundError:
            pass

    result = _load_diagnosis_module().validate_probe(result)
    duration = time.monotonic() - started
    execution = {
        "schema_version": "cuda-workload-optimizer/probe-execution-v1",
        "probe_id": selected["id"],
        "argv_sha256": _canonical_digest(selected["argv"]),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "stop_reason": stop_reason,
        "events": events,
        "stdout": _redact_log(stdout.text(), secret_values),
        "stderr": _redact_log(stderr.text(), secret_values),
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
    }
    _atomic_json(probes_dir / f"{selected['id']}.json", result)
    _atomic_json(probes_dir / f"{selected['id']}.execution.json", execution)
    return result


def run_probe(
    probe: Mapping[str, Any],
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    *,
    log_limit_bytes: int = _DEFAULT_LOG_LIMIT,
    deadline_epoch: float | None = None,
) -> dict:
    """Admit and execute one public run-bound probe."""
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    _require_run_grant_investment_control(read_run_state(run_root))
    return _run_probe_unchecked(
        probe,
        control,
        run_root,
        log_limit_bytes=log_limit_bytes,
        deadline_epoch=deadline_epoch,
    )


def run_probes(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    *,
    deadline_epoch: float | None = None,
) -> list[dict]:
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    _require_run_grant_investment_control(read_run_state(run_root))
    return [
        _run_probe_unchecked(
            probe,
            normalized,
            run_root,
            deadline_epoch=deadline_epoch,
        )
        for probe in normalized["probes"]
    ]


def diagnose_run(run_dir: os.PathLike[str] | str) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    _require_run_grant_investment_control(read_run_state(run_root))
    probes_dir = run_root / "probes"
    values = []
    for path in sorted(probes_dir.glob("*.json")):
        if path.name.endswith(".execution.json"):
            continue
        values.append(load_json_object(path))
    policy_path = Path(__file__).resolve().parents[1] / "references" / "workload_diagnosis_policy.json"
    diagnosis_module = _load_diagnosis_module()
    result = diagnosis_module.diagnose(values, diagnosis_module.load_policy(policy_path))
    _atomic_json(run_root / "diagnosis.json", result)
    return result


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ValidationError(f"cannot hash artifact {path}: {error}") from error
    return digest.hexdigest()


def review_change(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    change_set: Mapping[str, Any],
    *,
    deadline_epoch: float | None = None,
) -> dict:
    """Record a safe standalone summary; provider calls are Controller-managed."""
    normalized = validate_control_manifest(control)
    change = validate_change_set(change_set, normalized)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    _require_run_grant_investment_control(read_run_state(run_root))
    reviewer = _load_reviewer_module()
    request = _final_review_request(
        run_root,
        change,
        {"status": "unknown", "primary": {}, "constraints": []},
    )
    _atomic_json(run_root / "review_request.json", request)
    return reviewer.write_skipped_review(
        request,
        run_root,
        reason="controller_managed",
    )


def _final_review_request(
    run_root: Path,
    change: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict:
    reviewer = _load_reviewer_module()
    diagnosis_path = run_root / "diagnosis.json"
    primary = evaluation.get("primary")
    if not isinstance(primary, Mapping):
        primary = {}
    constraints = evaluation.get("constraints")
    if type(constraints) is not list:
        constraints = []
    candidate_diff = run_root / "candidate.diff"
    hashes = {
        "diagnosis.json": _sha256_path(diagnosis_path),
        "change_set.json": _canonical_digest(change),
    }
    if candidate_diff.is_file() and not candidate_diff.is_symlink():
        hashes["candidate.diff"] = _sha256_path(candidate_diff)
    return reviewer.validate_review_request(
        reviewer.build_review_request(
            diagnosis={"review_kind": "final"},
            change_set={
                "scope": change["scope"],
                "risk": change["risk"],
                "candidate": {
                    "effect_pct": primary.get("estimate_pct"),
                    "ci_low_pct": primary.get("ci_low_pct"),
                    "ci_high_pct": primary.get("ci_high_pct"),
                },
            },
            redacted_diff=(
                "present" if "candidate.diff" in hashes else ""
            ),
            experiment={
                "evaluation_status": evaluation.get("status", "unknown"),
                "primary_status": primary.get("status", "unknown"),
                "constraint_statuses": [
                    item.get("status", "unknown")
                    for item in constraints
                    if isinstance(item, Mapping)
                ],
            },
            artifact_hashes=hashes,
        )
    )


def _completed_final_review_aggregate(
    run_root: Path,
    state: Mapping[str, Any],
    request: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict | None:
    completions = state.get("review_call_completions", {})
    if type(completions) is not dict:
        raise ValidationError("review completion state is invalid")
    review_root = _review_call_root(run_root, "final")
    for complete_digest in completions.values():
        path = review_root / "generations" / "completions" / f"{complete_digest}.json"
        if not path.is_file() or path.is_symlink():
            continue
        complete = _review_generation(
            review_root, "complete", complete_digest
        )
        intent_digest = _sha256(
            complete.get("intent_sha256"),
            "final review intent",
        )
        intent = _validate_review_call_intent(
            "final",
            _review_generation(review_root, "intent", intent_digest),
        )
        if intent["request"] != request:
            continue
        base = load_json_object(
            run_root
            / "state_generations"
            / f"{intent['base_state_sha256']}.json"
        )
        if any(
            base.get(field) != state.get(field)
            for field in (
                "authorization_grant_sha256",
                "candidate_digest",
                "change_set_digest",
            )
        ):
            continue
        aggregate_digest = _sha256(
            complete.get("aggregate_sha256"),
            "final review aggregate",
        )
        aggregate = _validate_review_aggregate(
            intent,
            _review_generation(review_root, "aggregate", aggregate_digest),
            _reviewer_configs(control),
        )
        _validate_review_call_complete(
            "final",
            intent,
            complete,
            aggregate,
        )
        if _canonical_digest(complete) != complete_digest:
            raise ValidationError("final review completion digest drifted")
        return aggregate
    return None


def _run_final_managed_review(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    change: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> tuple[dict, dict]:
    reviewer = _load_reviewer_module()
    request = _final_review_request(run_root, change, evaluation)
    _atomic_json(run_root / "review_request.json", request)
    prior = _completed_final_review_aggregate(
        run_root,
        state,
        request,
        control,
    )
    if prior is not None:
        _atomic_json(run_root / "review.json", prior)
        return copy.deepcopy(dict(state)), prior
    if not _reviewer_configs(control):
        skipped = reviewer.write_skipped_review(request, run_root)
        return copy.deepcopy(dict(state)), skipped
    committed, aggregate = _managed_review_call(
        run_root,
        state,
        control,
        "final",
        request,
        trigger="final",
    )
    _atomic_json(run_root / "review.json", aggregate)
    return committed, aggregate


def _write_final_review_adjudication(
    run_root: Path,
    review_artifact: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict:
    """Preserve external challenges and state the local evidence decision."""
    raw_reviews = review_artifact.get("reviews")
    if type(raw_reviews) is not list:
        raw_reviews = [review_artifact]
    challenges = []
    for item in raw_reviews:
        if not isinstance(item, Mapping) or item.get("status") != "completed":
            continue
        response = item.get("response")
        if not isinstance(response, Mapping) or response.get("verdict") != "challenge":
            continue
        concerns = response.get("concerns", [])
        experiments = response.get("suggested_experiments", [])
        challenges.append(
            {
                "provider": item.get("provider"),
                "concerns": copy.deepcopy(concerns) if type(concerns) is list else [],
                "suggested_experiments": (
                    copy.deepcopy(experiments) if type(experiments) is list else []
                ),
            }
        )
    primary_status = evaluation.get("primary", {}).get("status")
    constraints_passed = all(
        item.get("status") == "passed"
        for item in evaluation.get("constraints", [])
    )
    if challenges and primary_status == "confirmed_win" and constraints_passed:
        status = "retained_but_non_blocking"
    elif challenges:
        status = "retained_for_local_review"
    else:
        status = "not_required"
    artifact = {
        "schema_version": "cuda-workload-optimizer/review-adjudication-v1",
        "status": status,
        "advisory_only": True,
        "challenges": challenges,
        "local_evidence": {
            "primary_status": primary_status,
            "constraints_passed": constraints_passed,
            "evaluation_sha256": _canonical_digest(evaluation),
        },
    }
    _atomic_json(run_root / "review_adjudication.json", artifact)
    return artifact


def _scope_layout(control: Mapping[str, Any], scope: str) -> tuple[Path, list[str], str]:
    if scope == "project":
        return (
            Path(control["project_root"]),
            list(control["mutation"]["project_paths"]),
            "project",
        )
    if scope == "isolated_environment":
        return Path(control["mutation"]["environment_root"]), ["."], "environment"
    raise ValidationError("unsupported ChangeSet scope")


def _identity(control: Mapping[str, Any], scope: str) -> dict:
    base, roots, _snapshot_name = _scope_layout(control, scope)
    files = {}
    missing_roots = []
    for relative_root in roots:
        root = base if relative_root == "." else base / relative_root
        if not root.exists():
            missing_roots.append(relative_root)
            continue
        if relative_root == "." and not root.is_dir():
            raise ValidationError("environment_root must be a directory")
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_symlink():
                raise ValidationError(f"mutation root contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(base).as_posix()
            files[relative] = {
                "sha256": _sha256_path(path),
                "size_bytes": path.stat().st_size,
                "mode": path.stat().st_mode & 0o777,
            }
    return {
        "schema_version": "cuda-workload-optimizer/project-identity-v1",
        "scope": scope,
        "roots": roots,
        "missing_roots": sorted(missing_roots),
        "files": files,
        "digest": _canonical_digest(
            {"missing_roots": sorted(missing_roots), "files": files}
        ),
    }


def _project_surface_identity(project_root: Path) -> dict:
    root = project_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValidationError("project_root must be a non-symlink directory")
    entries = {}
    excluded_directories = {".git", ".worktrees", "__pycache__"}
    try:
        for current, raw_directories, raw_files in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories = []
            for name in sorted(raw_directories):
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                if name in excluded_directories:
                    continue
                if path.is_symlink():
                    metadata = path.lstat()
                    entries[relative] = {
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": metadata.st_mode & 0o777,
                    }
                else:
                    directories.append(name)
            raw_directories[:] = directories
            for name in sorted(raw_files):
                if name.endswith(".pyc"):
                    continue
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    entries[relative] = {
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": metadata.st_mode & 0o777,
                    }
                elif stat.S_ISREG(metadata.st_mode):
                    entries[relative] = {
                        "type": "file",
                        "size_bytes": metadata.st_size,
                        "mode": metadata.st_mode & 0o777,
                        "mtime_ns": metadata.st_mtime_ns,
                        "sha256": _sha256_path(path),
                    }
                else:
                    entries[relative] = {
                        "type": "other",
                        "mode": metadata.st_mode,
                    }
    except OSError as error:
        raise ValidationError(f"cannot identify the complete project surface: {error}") from error
    return {
        "schema_version": "cuda-workload-optimizer/project-surface-identity-v1",
        "entries": entries,
        "digest": _canonical_digest(entries),
    }


def _snapshot_scope(
    control: Mapping[str, Any], run_root: Path, scope: str
) -> dict:
    base, roots, snapshot_name = _scope_layout(control, scope)
    snapshot = run_root / "snapshot" / snapshot_name
    if snapshot.exists():
        raise ValidationError("frozen ChangeSet snapshot already exists")
    identity = _identity(control, scope)
    if scope == "isolated_environment":
        if identity["missing_roots"]:
            raise ValidationError("environment_root must exist before registration")
        if base.is_symlink() or base.stat().st_uid != os.getuid():
            raise ValidationError(
                "environment_root must be a user-owned non-symlink directory"
            )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, snapshot, symlinks=False)
        _atomic_json(
            run_root / "rounds" / "round-1" / "before_identity.json", identity
        )
        return identity
    snapshot.mkdir(parents=True)
    for relative_root in roots:
        source = base / relative_root
        destination = snapshot / relative_root
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=False)
        else:
            shutil.copy2(source, destination)
    _atomic_json(run_root / "rounds" / "round-1" / "before_identity.json", identity)
    return identity


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _frozen_snapshot_identity(
    control: Mapping[str, Any],
    run_root: Path,
    scope: str,
    expected_identity_digest: str,
) -> dict:
    _base, _roots, snapshot_name = _scope_layout(control, scope)
    snapshot = run_root / "snapshot" / snapshot_name
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValidationError("frozen snapshot must be a regular directory")
    snapshot_control = copy.deepcopy(control)
    if scope == "project":
        snapshot_control["project_root"] = str(snapshot)
    else:
        snapshot_control["mutation"]["environment_root"] = str(snapshot)
    snapshot_identity = _identity(snapshot_control, scope)
    if snapshot_identity["digest"] != expected_identity_digest:
        raise ValidationError("frozen snapshot identity does not match registration")
    return snapshot_identity


def _restore_snapshot(
    control: Mapping[str, Any],
    run_root: Path,
    scope: str,
    expected_identity_digest: str,
) -> None:
    base, roots, snapshot_name = _scope_layout(control, scope)
    snapshot = run_root / "snapshot" / snapshot_name
    _frozen_snapshot_identity(
        control,
        run_root,
        scope,
        expected_identity_digest,
    )
    if scope == "isolated_environment":
        if base.exists() or base.is_symlink():
            _remove_path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, base, symlinks=False)
        return
    for relative_root in roots:
        current = base / relative_root
        frozen = snapshot / relative_root
        if current.exists() or current.is_symlink():
            _remove_path(current)
        if not frozen.exists():
            continue
        current.parent.mkdir(parents=True, exist_ok=True)
        if frozen.is_dir():
            shutil.copytree(frozen, current, symlinks=False)
        else:
            shutil.copy2(frozen, current)


def _changed_paths(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    names = set(before["files"]) | set(after["files"])
    return sorted(
        name for name in names if before["files"].get(name) != after["files"].get(name)
    )


def _path_allowed(relative: str, allowed: Sequence[str]) -> bool:
    path = Path(relative)
    return any(
        path == Path(root) or _is_within(path, Path(root))
        for root in allowed
    )


def _candidate_diff(
    control: Mapping[str, Any],
    run_root: Path,
    changed: Sequence[str],
    scope: str,
) -> str:
    base, _roots, snapshot_name = _scope_layout(control, scope)
    snapshot = run_root / "snapshot" / snapshot_name
    chunks = []
    for relative in changed:
        before_path = snapshot / relative
        after_path = base / relative
        try:
            before = before_path.read_text("utf-8").splitlines(keepends=True) if before_path.exists() else []
            after = after_path.read_text("utf-8").splitlines(keepends=True) if after_path.exists() else []
        except (OSError, UnicodeError):
            before_hash = _sha256_path(before_path) if before_path.exists() else "missing"
            after_hash = _sha256_path(after_path) if after_path.exists() else "missing"
            chunks.append(f"binary {relative}: {before_hash} -> {after_hash}\n")
            continue
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return _redact_log("".join(chunks), ())


def _static_review_changed_files(
    control: Mapping[str, Any],
    *,
    scope: str,
    changed: Sequence[str],
    candidate_binding: Mapping[str, Any],
    change_set_digest: str,
    after_identity_digest: str,
) -> dict:
    """Apply cheap parsers plus binding checks before any workload command."""
    base, _roots, _snapshot_name = _scope_layout(control, scope)
    checks = []
    for relative in changed:
        path = base / relative
        if not path.is_file() or path.is_symlink():
            continue
        kind = None
        try:
            source = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".py":
                kind = "python_ast"
                ast.parse(source, filename=relative)
            elif path.suffix.lower() == ".json":
                kind = "json_parse"
                json.loads(
                    source,
                    object_pairs_hook=_pairs_without_duplicates,
                    parse_constant=lambda token: _raise_invalid_number(token),
                )
        except (OSError, UnicodeError, SyntaxError, json.JSONDecodeError, ValidationError) as error:
            checks.append(
                {
                    "kind": kind or "text_decode",
                    "path": relative,
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
            )
        else:
            if kind is not None:
                checks.append(
                    {"kind": kind, "path": relative, "status": "passed"}
                )
        if path.suffix.lower() in {
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
        }:
            checks.append(
                {
                    "kind": "source_syntax",
                    "path": relative,
                    "status": "not_applicable",
                    "reason": "no_configured_static_checker",
                }
            )

    binding_without_digest = dict(candidate_binding)
    binding_digest = binding_without_digest.pop("digest", None)
    binding_checks = {
        "candidate_digest": candidate_binding.get("candidate_digest")
        == _canonical_digest(candidate_binding.get("candidate")),
        "binding_digest": binding_digest
        == _canonical_digest(binding_without_digest),
        "change_set_digest": candidate_binding.get("change_set_digest")
        == change_set_digest,
        "after_identity_digest": candidate_binding.get("after_identity_digest")
        == after_identity_digest,
    }
    checks.extend(
        {
            "kind": kind,
            "status": "passed" if passed else "failed",
        }
        for kind, passed in binding_checks.items()
    )
    status = "passed"
    if any(check["status"] == "failed" for check in checks):
        status = "failed"
    return {
        "status": status,
        "candidate_digest": candidate_binding.get("candidate_digest"),
        "change_set_digest": change_set_digest,
        "after_identity_digest": after_identity_digest,
        "changed_paths": list(changed),
        "checks": checks,
    }


def read_run_state(run_dir: os.PathLike[str] | str) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    commit = load_json_object(run_root / "state_commit.json")
    if set(commit) != {"schema_version", "state_digest"} or commit.get(
        "schema_version"
    ) != "cuda-workload-optimizer/state-commit-v1":
        raise ValidationError("state commit marker is invalid")
    digest = commit.get("state_digest")
    if type(digest) is not str or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ValidationError("state commit digest is invalid")
    state = load_json_object(run_root / "state_generations" / f"{digest}.json")
    if _canonical_digest(state) != digest:
        raise ValidationError("committed state generation digest is invalid")
    for name in ("state.json", "checkpoint.json"):
        path = run_root / name
        try:
            mirror = load_json_object(path)
        except (OSError, ValidationError):
            mirror = None
        if mirror != state:
            _atomic_json(path, state)
    return state


def _write_state(run_root: Path, state: Mapping[str, Any]) -> dict:
    detached = _json_copy(state, "state")
    digest = _canonical_digest(detached)
    _atomic_json(run_root / "state_generations" / f"{digest}.json", detached)
    _atomic_json(
        run_root / "state_commit.json",
        {
            "schema_version": "cuda-workload-optimizer/state-commit-v1",
            "state_digest": digest,
        },
    )
    _atomic_json(run_root / "state.json", detached)
    _atomic_json(run_root / "checkpoint.json", detached)
    return detached


def _require_run_grant_investment_control(state: Mapping[str, Any]) -> None:
    if state.get("investment_control_version") != "run-grant-v1":
        raise ValidationError("legacy investment control requires restart")


def _candidate_authorization_digest(state: Mapping[str, Any]) -> str:
    return _sha256(
        state.get("authorization_grant_sha256", state.get("control_digest")),
        "candidate authorization",
    )


def _advance(
    run_root: Path,
    state: Mapping[str, Any],
    completed: str,
    *,
    stage: str,
    next_action: str,
) -> dict:
    updated = copy.deepcopy(state)
    if completed not in updated["completed_stages"]:
        updated["completed_stages"].append(completed)
    updated["stage"] = stage
    updated["next_action"] = next_action
    updated["updated_at_epoch"] = time.time()
    return _write_state(run_root, updated)


def _load_frozen_control(run_root: Path, state: Mapping[str, Any] | None = None) -> dict:
    active_state = read_run_state(run_root) if state is None else state
    control = validate_control_manifest(
        load_json_object(run_root / "control_manifest.json")
    )
    if _canonical_digest(control) != active_state.get("control_digest"):
        raise ValidationError("frozen control manifest digest does not match state")
    return control


def _normalize_frozen_workload(control: Mapping[str, Any]):
    try:
        return _load_workload_module().normalize_workload(
            workload_manifest=control["workload_manifest"]
        )
    except (OSError, ValueError) as error:
        raise ValidationError(f"workload identity validation failed: {error}") from error


def _validate_workload_candidate_minimum_effect(
    change: Mapping[str, Any], workload: Any
) -> None:
    project_minimum = max(0.5, float(workload.objective["min_effect_pct"]))
    candidate_minimum = float(change["candidate"]["minimum_effect"]["value"])
    if candidate_minimum < project_minimum:
        raise ValidationError(
            "candidate minimum_effect is below the project contract "
            f"({candidate_minimum:g}% < {project_minimum:g}%)"
        )


def _check_deadline(state: Mapping[str, Any]) -> None:
    if time.time() > state["deadline_epoch"]:
        raise ValidationError("workload optimization budget deadline has expired")


def start_optimization_budget_after_readiness(
    state: Mapping[str, Any], runtime: Mapping[str, Any], *, now: float
) -> dict:
    """Start performance time only after the environment admission succeeds."""
    if not isinstance(state, Mapping) or not isinstance(runtime, Mapping):
        raise ValidationError("state and runtime must be mappings")
    if isinstance(now, bool) or not isinstance(now, (int, float)):
        raise ValidationError("now must be a finite epoch")
    timestamp = float(now)
    if not math.isfinite(timestamp):
        raise ValidationError("now must be a finite epoch")
    started = state.get("started_at_epoch")
    if isinstance(started, bool) or not isinstance(started, (int, float)):
        raise ValidationError("state.started_at_epoch must be finite")
    started = float(started)
    def duration(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{field} must be a positive finite number")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValidationError(f"{field} must be a positive finite number")
        return parsed

    soft = duration(runtime.get("soft_target_seconds"), "soft_target_seconds")
    hard = duration(runtime.get("hard_ceiling_seconds"), "hard_ceiling_seconds")
    if soft > hard:
        raise ValidationError("soft target must not exceed hard ceiling")
    updated = copy.deepcopy(dict(state))
    updated["environment_seconds"] = max(0.0, timestamp - started)
    updated["optimization_started_at_epoch"] = timestamp
    updated["soft_target_epoch"] = timestamp + soft
    updated["deadline_epoch"] = timestamp + hard
    updated["updated_at_epoch"] = timestamp
    return updated


def migrate_completed_readiness_budget(
    state: Mapping[str, Any], runtime: Mapping[str, Any], *, completed_at: float
) -> dict:
    """Migrate a pre-fix V3.1 state once without charging readiness time."""
    if "readiness" not in state.get("completed_stages", []):
        return copy.deepcopy(dict(state))
    if "optimization_started_at_epoch" in state:
        return copy.deepcopy(dict(state))
    updated = start_optimization_budget_after_readiness(
        state, runtime, now=completed_at
    )
    updated["optimization_timer_migration"] = "readiness-report-mtime-v1"
    return updated


def _current_readiness_identity(control: Mapping[str, Any]) -> dict:
    environment_root = Path(control["mutation"]["environment_root"])
    if not environment_root.is_dir() or environment_root.is_symlink():
        raise ValidationError(
            "control-v2 environment_root must be an existing non-symlink directory"
        )
    inventory = _load_check_env_module().collect_identity_inventory()
    return _load_readiness_identity_module().build_identity(
        environment_root=environment_root,
        inventory=inventory,
        run=_load_check_env_module()._run,
    )


def _load_frozen_readiness_contract(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    module = _load_readiness_contract_module()
    path = run_root / "readiness_contract.json"
    value = module.load_contract(path)
    validated = module.validate_contract(
        value,
        project_root=Path(control["project_root"]),
        environment_root=Path(control["mutation"]["environment_root"]),
    )
    if module.contract_digest(validated) != state.get(
        "readiness_contract_digest"
    ):
        raise ValidationError("frozen readiness contract digest does not match state")
    return validated


def _run_readiness_gate(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    gate = _load_readiness_gate_module()
    contract = _load_frozen_readiness_contract(control, run_root, state)
    identity = _current_readiness_identity(control)
    return gate.run_gate(
        contract=contract,
        control={
            "project_root": control["project_root"],
            "environment_root": control["mutation"]["environment_root"],
            "environment_identity": identity,
        },
        run_dir=run_root,
        identity_provider=lambda: _current_readiness_identity(control),
    )


def _readiness_execution_records(run_root: Path) -> list[dict]:
    execution_root = run_root / "readiness" / "executions"
    if not execution_root.exists():
        return []
    if execution_root.is_symlink() or not execution_root.is_dir():
        raise ValidationError("readiness execution directory is invalid")
    records = []
    for expected_sequence, path in enumerate(
        sorted(execution_root.glob("*.json")),
        1,
    ):
        record = load_json_object(path)
        fields = {
            "schema_version",
            "sequence",
            "outcome",
            "report_sha256",
            "duration_seconds",
        }
        _closed(record, fields, f"readiness execution {path.name}")
        _required(record, fields, f"readiness execution {path.name}")
        if (
            record["schema_version"]
            != "cuda-workload-optimizer/readiness-execution-v1"
        ):
            raise ValidationError("readiness execution schema is invalid")
        if (
            path.name != f"{expected_sequence:06d}.json"
            or record["sequence"] != expected_sequence
        ):
            raise ValidationError("readiness execution sequence is not contiguous")
        if record["outcome"] not in {"completed", "failed"}:
            raise ValidationError("readiness execution outcome is invalid")
        report_sha = record["report_sha256"]
        if report_sha is not None:
            _sha256(report_sha, "readiness execution report_sha256")
        if record["outcome"] == "completed" and report_sha is None:
            raise ValidationError(
                "completed readiness execution must bind its report"
            )
        duration = record["duration_seconds"]
        if (
            type(duration) not in {int, float}
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise ValidationError("readiness execution duration is invalid")
        records.append(record)
    return records


def _record_readiness_execution(
    run_root: Path,
    *,
    duration_seconds: float,
    outcome: str,
    report: Mapping[str, Any] | None,
) -> dict:
    records = _readiness_execution_records(run_root)
    sequence = len(records) + 1
    record = {
        "schema_version": "cuda-workload-optimizer/readiness-execution-v1",
        "sequence": sequence,
        "outcome": outcome,
        "report_sha256": (
            None if report is None else _canonical_digest(dict(report))
        ),
        "duration_seconds": max(0.0, float(duration_seconds)),
    }
    _atomic_json(
        run_root
        / "readiness"
        / "executions"
        / f"{sequence:06d}.json",
        record,
    )
    return record


def _run_readiness_gate_checked(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    surface_before = (
        _project_surface_identity(Path(control["project_root"]))
        if "analysis_contract" in control
        else None
    )
    started = time.monotonic()
    try:
        report = _run_readiness_gate(control, run_root, state)
    except BaseException:
        _record_readiness_execution(
            run_root,
            duration_seconds=time.monotonic() - started,
            outcome="failed",
            report=None,
        )
        raise
    _record_readiness_execution(
        run_root,
        duration_seconds=time.monotonic() - started,
        outcome="completed",
        report=report,
    )
    if (
        surface_before is not None
        and _project_surface_identity(Path(control["project_root"]))
        != surface_before
    ):
        raise ValidationError("readiness modified the complete project surface")
    return report


def _readiness_report_digest(run_root: Path, report: Mapping[str, Any]) -> str:
    path = run_root / "readiness" / "report.json"
    return _sha256_path(path) if path.is_file() else _canonical_digest(report)


def _verify_readiness_report(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> bool:
    gate = _load_readiness_gate_module()
    try:
        report = gate._load_prior_report(run_root / "readiness")
    except ValueError as error:
        raise ValidationError(f"readiness report verification failed: {error}") from error
    if report is None:
        raise ValidationError("completed readiness stage is missing its report")
    if _readiness_report_digest(run_root, report) != state.get(
        "readiness_report_digest"
    ):
        raise ValidationError("readiness report digest does not match state")
    contract = _load_frozen_readiness_contract(control, run_root, state)
    contract_digest = _load_readiness_contract_module().contract_digest(contract)
    if report.get("contract_digest") != contract_digest:
        raise ValidationError("readiness report contract digest drifted")
    identity_digest = gate.environment_identity_digest(
        _current_readiness_identity(control)
    )
    if report.get("environment_identity_digest") != identity_digest:
        return False
    if not report.get("can_start_diagnosis"):
        return False
    now = time.time()
    required_ids = {
        item["id"] for item in contract["requirements"] if item["necessity"] == "required"
    }
    ready_ids = {
        item.get("requirement_id")
        for item in report.get("results", [])
        if type(item) is dict
        and item.get("necessity") == "required"
        and item.get("admission_status") == "ready"
        and isinstance(item.get("valid_until"), (int, float))
        and not isinstance(item.get("valid_until"), bool)
        and math.isfinite(float(item["valid_until"]))
        and float(item["valid_until"]) > now
    }
    return ready_ids == required_ids


def _load_frozen_analysis_contract(run_root: Path, state: Mapping[str, Any]) -> dict:
    contract = _validate_active_diagnosis_contract(
        load_json_object(run_root / "active_diagnosis" / "analysis_contract.json")
    )
    if _canonical_digest(contract) != state.get("analysis_contract_digest"):
        raise ValidationError("frozen analysis contract digest does not match state")
    return contract


def _adapter_execution_binding(
    adapter_path: Path, argv: Sequence[str], field: str
) -> dict:
    adapter = adapter_path.resolve(strict=True)
    adapter_text = str(adapter_path)
    if argv[0] == adapter_text:
        if not os.access(adapter, os.X_OK):
            raise ValidationError(f"{field} direct adapter must be executable")
        launcher = adapter
        mode = "direct"
        adapter_arg_index = 0
    elif len(argv) >= 2 and argv[1] == adapter_text:
        raw_launcher = Path(argv[0]).expanduser()
        if not raw_launcher.is_absolute():
            located = shutil.which(argv[0])
            if located is None:
                raise ValidationError(f"{field} interpreter cannot be resolved")
            raw_launcher = Path(located)
        launcher = raw_launcher.resolve(strict=True)
        basename = launcher.name.lower()
        suffix = adapter.suffix.lower()
        python_launcher = re.fullmatch(r"python(?:3(?:\.\d+)*)?", basename) is not None
        shell_launcher = basename in {"sh", "bash"}
        if not ((suffix == ".py" and python_launcher) or (suffix == ".sh" and shell_launcher)):
            raise ValidationError(
                f"{field} must execute the adapter directly or through a matching Python/shell interpreter"
            )
        mode = "interpreter"
        adapter_arg_index = 1
    else:
        raise ValidationError(
            f"{field} must place adapter_path at argv[0], or argv[1] after a matching interpreter"
        )
    if not launcher.is_file():
        raise ValidationError(f"{field} launcher must be a regular file")
    return {
        "adapter_path": adapter_text,
        "adapter_sha256": _sha256_path(adapter),
        "launcher_path": str(launcher),
        "launcher_sha256": _sha256_path(launcher),
        "mode": mode,
        "adapter_arg_index": adapter_arg_index,
    }


def _load_frozen_execution_bindings(
    run_root: Path, state: Mapping[str, Any]
) -> dict:
    bindings = load_json_object(
        run_root / "active_diagnosis" / "execution_bindings.json"
    )
    if _canonical_digest(bindings) != state.get("analysis_execution_bindings_digest"):
        raise ValidationError("frozen analysis execution bindings drifted from state")
    return bindings


def _verify_adapter_execution_binding(
    expected: Mapping[str, Any], adapter_path: Path, argv: Sequence[str], field: str
) -> None:
    actual = _adapter_execution_binding(adapter_path, argv, field)
    if actual != expected:
        raise ValidationError(f"{field} adapter or launcher identity drifted")


def _ready_capability_ids(report: Mapping[str, Any], *, now: float | None = None) -> list[str]:
    current = time.time() if now is None else float(now)
    identity_digest = report.get("environment_identity_digest")
    return sorted(
        item["requirement_id"]
        for item in report.get("results", [])
        if type(item) is dict
        and item.get("admission_status") == "ready"
        and item.get("identity_digest") == identity_digest
        and type(item.get("valid_until")) in {int, float}
        and float(item["valid_until"]) > current
    )


def _validate_global_scan_draft(value: Mapping[str, Any]) -> dict:
    draft = _object(value, "global_scan")
    fields = {
        "schema_version",
        "regime",
        "boundary_ambiguous",
        "window",
        "coverage",
        "nodes",
        "edges",
        "hot_path",
        "uncovered_intervals",
        "conclusion_level",
    }
    _closed(draft, fields, "global_scan")
    _required(draft, fields, "global_scan")
    if draft["schema_version"] != _GLOBAL_SCAN_DRAFT_SCHEMA:
        raise ValidationError(
            f"global_scan.schema_version must be {_GLOBAL_SCAN_DRAFT_SCHEMA}"
        )
    regime = _object(draft["regime"], "global_scan.regime")
    regime_fields = {
        "shape_distribution_sha256",
        "dynamic_branch_sha256",
        "execution_regime_sha256",
    }
    _closed(regime, regime_fields, "global_scan.regime")
    _required(regime, regime_fields, "global_scan.regime")
    for field in regime_fields:
        _sha256(regime[field], f"global_scan.regime.{field}")
    if type(draft["boundary_ambiguous"]) is not bool:
        raise ValidationError("global_scan.boundary_ambiguous must be a boolean")
    window = _object(draft["window"], "global_scan.window")
    _closed(window, {"start_us", "end_us"}, "global_scan.window")
    _required(window, {"start_us", "end_us"}, "global_scan.window")
    for field in ("start_us", "end_us"):
        value = window[field]
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ValidationError(f"global_scan.window.{field} must be finite")
    if float(window["start_us"]) < 0 or float(window["end_us"]) <= float(
        window["start_us"]
    ):
        raise ValidationError("global_scan window must be positive")
    return _json_copy(draft, "global_scan", reject_sensitive=True)


def _verify_active_diagnosis_ledger(run_root: Path) -> list[dict]:
    ledger_dir = run_root / "active_diagnosis" / "ledger"
    if not ledger_dir.exists():
        return []
    events = []
    previous = None
    for expected_sequence, path in enumerate(sorted(ledger_dir.glob("*.json")), 1):
        event = load_json_object(path)
        fields = {
            "schema_version",
            "sequence",
            "event_type",
            "previous_event_sha256",
            "payload_sha256",
            "created_at_epoch",
        }
        _closed(event, fields, f"active diagnosis ledger event {path.name}")
        _required(event, fields, f"active diagnosis ledger event {path.name}")
        if event["schema_version"] != "cuda-optimizer/active-diagnosis-event-v1":
            raise ValidationError("active diagnosis ledger schema is invalid")
        if event["sequence"] != expected_sequence:
            raise ValidationError("active diagnosis ledger sequence is not contiguous")
        _identifier(event["event_type"], "active diagnosis event_type")
        if event["previous_event_sha256"] != previous:
            raise ValidationError("active diagnosis ledger hash chain is invalid")
        _sha256(event["payload_sha256"], "active diagnosis payload_sha256")
        created = event["created_at_epoch"]
        if type(created) not in {int, float} or not math.isfinite(float(created)):
            raise ValidationError("active diagnosis event time must be finite")
        previous = _canonical_digest(event)
        events.append(event)
    return events


def _active_ledger_binding(events: Sequence[Mapping[str, Any]]) -> dict:
    if not events:
        raise ValidationError("active diagnosis ledger is empty")
    return {
        "active_diagnosis_ledger_sequence": len(events),
        "active_diagnosis_ledger_head_sha256": _canonical_digest(events[-1]),
    }


def _verify_committed_active_ledger(
    state: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    sequence = state.get("active_diagnosis_ledger_sequence")
    head = state.get("active_diagnosis_ledger_head_sha256")
    if type(sequence) is not int or sequence < 1 or type(head) is not str:
        raise ValidationError("run state is missing its active diagnosis ledger binding")
    if len(events) < sequence:
        raise ValidationError("committed active diagnosis ledger tail is missing")
    if _canonical_digest(events[sequence - 1]) != head:
        raise ValidationError("committed active diagnosis ledger head drifted")


def _active_ledger_append_boundary(
    run_root: Path,
    event_type: str,
) -> tuple[list[dict], int]:
    event_type = _identifier(event_type, "active diagnosis event_type")
    state = read_run_state(run_root)
    events = _verify_active_diagnosis_ledger(run_root)
    sequence = state.get("active_diagnosis_ledger_sequence")
    head = state.get("active_diagnosis_ledger_head_sha256")
    if sequence is None and head is None:
        committed_sequence = 0
    else:
        if type(sequence) is not int or sequence < 1 or type(head) is not str:
            raise ValidationError(
                "run state active diagnosis ledger binding is invalid"
            )
        committed_sequence = sequence
        if len(events) < committed_sequence:
            raise ValidationError("committed active diagnosis ledger tail is missing")
        if _canonical_digest(events[committed_sequence - 1]) != head:
            raise ValidationError("committed active diagnosis ledger head drifted")
    if len(events) == committed_sequence:
        return events, committed_sequence
    if (
        len(events) == committed_sequence + 1
        and events[-1]["event_type"] == event_type
    ):
        return events, committed_sequence
    raise ValidationError(
        "active diagnosis ledger has an uncommitted foreign tail"
    )


def _append_active_diagnosis_event(
    run_root: Path, event_type: str, payload: Mapping[str, Any]
) -> dict:
    event_type = _identifier(event_type, "active diagnosis event_type")
    payload_sha = _canonical_digest(_json_copy(payload, "active diagnosis payload"))
    events, committed_sequence = _active_ledger_append_boundary(
        run_root,
        event_type,
    )
    if len(events) == committed_sequence + 1:
        if events[-1]["payload_sha256"] == payload_sha:
            return events[-1]
        raise ValidationError(
            "active diagnosis ledger recovery payload conflicts with tail"
        )
    sequence = committed_sequence + 1
    event = {
        "schema_version": "cuda-optimizer/active-diagnosis-event-v1",
        "sequence": sequence,
        "event_type": event_type,
        "previous_event_sha256": (
            None if not events else _canonical_digest(events[-1])
        ),
        "payload_sha256": payload_sha,
        "created_at_epoch": time.time(),
    }
    path = (
        run_root
        / "active_diagnosis"
        / "ledger"
        / f"{sequence:06d}-{event_type}.json"
    )
    if path.exists():
        raise ValidationError("active diagnosis ledger event already exists")
    _atomic_json(path, event)
    return event


def _prepare_active_diagnosis_event(
    run_root: Path,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    created_at_epoch: float,
) -> tuple[str, dict]:
    event_type = _identifier(event_type, "active diagnosis event_type")
    payload_sha = _canonical_digest(_json_copy(payload, "active diagnosis payload"))
    events, committed_sequence = _active_ledger_append_boundary(
        run_root,
        event_type,
    )
    if len(events) == committed_sequence + 1:
        if events[-1]["payload_sha256"] != payload_sha:
            raise ValidationError(
                "active diagnosis ledger recovery payload conflicts with tail"
            )
        event = copy.deepcopy(events[-1])
    else:
        event = {
            "schema_version": "cuda-optimizer/active-diagnosis-event-v1",
            "sequence": committed_sequence + 1,
            "event_type": event_type,
            "previous_event_sha256": (
                None if not events else _canonical_digest(events[-1])
            ),
            "payload_sha256": payload_sha,
            "created_at_epoch": float(created_at_epoch),
        }
    relative = (
        Path("active_diagnosis")
        / "ledger"
        / f"{event['sequence']:06d}-{event_type}.json"
    )
    return relative.as_posix(), event




def _baseline_execution_records(run_root: Path) -> list[dict]:
    execution_root = run_root / "baseline" / "executions"
    if not execution_root.exists():
        return []
    if execution_root.is_symlink() or not execution_root.is_dir():
        raise ValidationError("baseline execution directory is invalid")
    records = []
    for expected_sequence, path in enumerate(
        sorted(execution_root.iterdir()),
        1,
    ):
        if path.is_symlink() or not path.is_file():
            raise ValidationError("baseline execution record is invalid")
        record = load_json_object(path)
        fields = {
            "schema_version",
            "sequence",
            "workload_kind",
            "status",
            "duration_seconds",
        }
        _closed(record, fields, f"baseline execution {path.name}")
        _required(record, fields, f"baseline execution {path.name}")
        if (
            record["schema_version"]
            != "cuda-workload-optimizer/baseline-execution-v1"
        ):
            raise ValidationError("baseline execution schema is invalid")
        if (
            path.name != f"{expected_sequence:06d}.json"
            or record["sequence"] != expected_sequence
        ):
            raise ValidationError("baseline execution sequence is not contiguous")
        if record["workload_kind"] not in {"python", "command"}:
            raise ValidationError("baseline execution workload kind is invalid")
        if record["status"] not in {"failed", "measured"}:
            raise ValidationError("baseline execution status is invalid")
        duration = record["duration_seconds"]
        if (
            type(duration) not in {int, float}
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise ValidationError("baseline execution duration is invalid")
        records.append(record)
    if len({record["workload_kind"] for record in records}) > 1:
        raise ValidationError("baseline execution workload kind drifted")
    return records


def _record_baseline_execution(
    run_root: Path,
    *,
    workload_kind: str,
    status: str,
    duration_seconds: float,
) -> dict:
    if workload_kind not in {"python", "command"}:
        raise ValidationError("baseline execution workload kind is invalid")
    if status not in {"failed", "measured"}:
        raise ValidationError("baseline execution status is invalid")
    if (
        type(duration_seconds) not in {int, float}
        or not math.isfinite(float(duration_seconds))
        or float(duration_seconds) < 0
    ):
        raise ValidationError("baseline execution duration is invalid")
    records = _baseline_execution_records(run_root)
    if records and records[-1]["workload_kind"] != workload_kind:
        raise ValidationError("baseline execution workload kind drifted")
    sequence = len(records) + 1
    record = {
        "schema_version": "cuda-workload-optimizer/baseline-execution-v1",
        "sequence": sequence,
        "workload_kind": workload_kind,
        "status": status,
        "duration_seconds": float(duration_seconds),
    }
    _atomic_json(
        run_root
        / "baseline"
        / "executions"
        / f"{sequence:06d}.json",
        record,
    )
    return record


def _bootstrap_execution_seconds(
    run_root: Path,
    state: Mapping[str, Any],
) -> float:
    readiness_records = _readiness_execution_records(run_root)
    if not readiness_records:
        raise ValidationError("readiness execution record is missing")
    readiness_seconds = sum(
        float(record["duration_seconds"])
        for record in readiness_records
    )
    baseline_records = _baseline_execution_records(run_root)
    if not baseline_records:
        raise ValidationError("baseline execution record is missing")
    if baseline_records[-1]["status"] != "measured":
        raise ValidationError("latest baseline execution was not measured")
    baseline_seconds = sum(
        float(record["duration_seconds"])
        for record in baseline_records
    )
    baseline_observation = load_json_object(
        run_root / "baseline" / "observation.json"
    )
    if baseline_observation.get("status") != "measured":
        raise ValidationError("baseline observation is not measured")
    contract = _load_frozen_analysis_contract(run_root, state)
    global_scan_id = contract["global_scan_probe_id"]
    profile_execution = load_json_object(
        run_root / "probes" / f"{global_scan_id}.execution.json"
    )
    profile_seconds = profile_execution.get("duration_seconds")
    if (
        type(profile_seconds) not in {int, float}
        or not math.isfinite(float(profile_seconds))
        or float(profile_seconds) < 0
    ):
        raise ValidationError("global profile execution time is invalid")
    return (
        float(readiness_seconds)
        + float(baseline_seconds)
        + float(profile_seconds)
    )


def _unknown_knowledge_fact() -> dict:
    return {
        "value": None,
        "status": "unknown",
        "source_kind": "unknown",
        "source_sha256": None,
    }


def _readiness_knowledge_identity(
    run_root: Path,
    report: Mapping[str, Any],
    contract: Mapping[str, Any],
    identities: Mapping[str, str],
) -> dict:
    """Extract only marker-bound readiness observations; never re-run probes."""
    scalar_names = {"sm_arch": "gpu_architecture", "driver_version": "driver_version",
                    "cuda_runtime_version": "cuda_runtime_version"}
    grouped_names = {"framework_versions", "compiler_versions", "profiler_versions"}
    values: dict[str, list[tuple[Any, str]]] = {name: [] for name in scalar_names}
    groups: dict[str, list[tuple[dict, str]]] = {name: [] for name in grouped_names}
    for raw in report.get("results", []):
        if (type(raw) is not dict or raw.get("admission_status") != "ready"
                or raw.get("identity_digest") != report.get("environment_identity_digest")):
            continue
        requirement_id = _identifier(raw.get("requirement_id"), "readiness requirement_id")
        relative = _relative(raw.get("evidence_path"), "readiness evidence_path")
        probe_path = run_root / relative
        if (probe_path.is_symlink() or not probe_path.is_file()
                or not _is_within(probe_path.resolve(strict=False), run_root)):
            raise ValidationError("readiness evidence path is not a contained regular file")
        execution_path = probe_path.with_name(f"{requirement_id}.execution.json")
        marker_path = probe_path.with_name(f"{requirement_id}.complete.json")
        if any(path.is_symlink() or not path.is_file() for path in (execution_path, marker_path)):
            raise ValidationError("readiness completion chain is incomplete")
        probe = load_json_object(probe_path)
        execution = load_json_object(execution_path)
        marker = load_json_object(marker_path)
        if (probe.get("requirement_id") != requirement_id
                or execution.get("requirement_id") != requirement_id
                or marker.get("requirement_id") != requirement_id
                or marker.get("schema_version") != "cuda-workload-optimizer/readiness-completion-v1"
                or marker.get("probe_sha256") != _sha256_path(probe_path)
                or marker.get("execution_sha256") != _sha256_path(execution_path)
                or execution.get("environment_identity_digest") != raw.get("identity_digest")):
            raise ValidationError("readiness completion provenance does not match report")
        observations = probe.get("observations", {})
        if type(observations) is not dict:
            continue
        source_sha = marker["probe_sha256"]
        for name in scalar_names:
            if type(observations.get(name)) is str and observations[name]:
                values[name].append((observations[name], source_sha))
        for name in grouped_names:
            group = observations.get(name)
            if type(group) is dict and all(type(key) is str and type(value) is str and value for key, value in group.items()):
                groups[name].append((group, source_sha))
    def fact(items: list[tuple[Any, str]]) -> dict:
        if not items or any(value != items[0][0] for value, _sha in items):
            return _unknown_knowledge_fact()
        return {"value": items[0][0], "status": "verified", "source_kind": "readiness_probe", "source_sha256": items[0][1]}
    identity = {
        "schema_version": "cuda-optimizer/knowledge-identity-v1",
        **{target: fact(values[name]) for name, target in scalar_names.items()},
        **{name: {} for name in grouped_names},
        "workload_contract_sha256": identities["workload_contract_sha256"],
        "source_sha256": identities["source_sha256"],
        "environment_sha256": identities["environment_sha256"],
    }
    for name, entries in groups.items():
        merged: dict[str, list[tuple[str, str]]] = {}
        for group, source_sha in entries:
            for key, value in group.items():
                merged.setdefault(key, []).append((value, source_sha))
        identity[name] = {key: fact(items) for key, items in merged.items()}
    profiler = contract["source"]
    contract_sha = _canonical_digest(contract)
    existing_profiler = identity["profiler_versions"].get(profiler["profiler"])
    if existing_profiler is None:
        identity["profiler_versions"][profiler["profiler"]] = {
            "value": profiler["profiler_version"], "status": "verified",
            "source_kind": "analysis_contract", "source_sha256": contract_sha,
        }
    elif existing_profiler["value"] != profiler["profiler_version"]:
        identity["profiler_versions"][profiler["profiler"]] = _unknown_knowledge_fact()
    return identity


def _rebuild_knowledge_context(
    run_root: Path, context: Mapping[str, Any], contract: Mapping[str, Any],
    epoch: Mapping[str, Any], execution_map: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any], selection_policy: Mapping[str, Any],
    performance_model: Mapping[str, Any],
    pending_summary: Mapping[str, Any] | None = None,
) -> dict:
    catalog = load_json_object(run_root / "active_diagnosis" / "action_catalog.json")
    ready = sorted(selection_policy.get("available_capability_ids", []))
    contract_ids = sorted(action["action_id"] for action in contract["actions"])
    available = sorted(action["action_id"] for action in catalog["actions"]
                       if action.get("control_scope") == "read_only"
                       and set(action.get("required_capability_ids", [])).issubset(ready))
    envelopes = []
    by_action = {item["action_id"]: item for item in catalog["actions"]}
    contract_actions = {item["action_id"]: item for item in contract["actions"]}
    ledger = _verify_active_diagnosis_ledger(run_root)
    summaries = context.get("evidence_results", [])
    for index, summary in enumerate(summaries):
        signature = _sha256(
            summary["request_signature"],
            "knowledge evidence request_signature",
        )
        result_path = run_root / _relative(
            summary["result_path"], "knowledge result path"
        )
        expected_result_path = (
            run_root
            / "active_diagnosis"
            / "evidence"
            / signature
            / "result.json"
        )
        if (result_path.is_symlink() or not result_path.is_file()
                or result_path != expected_result_path
                or not _is_within(result_path.resolve(strict=False), run_root)):
            raise ValidationError("sealed evidence result path is unsafe")
        result = load_json_object(result_path)
        if _canonical_digest(result) != summary["result_sha256"]:
            raise ValidationError("sealed evidence result digest does not match summary")
        action_id = summary["action_id"]
        action = by_action.get(action_id)
        contract_action = contract_actions.get(action_id)
        if not isinstance(action, Mapping) or not isinstance(contract_action, Mapping):
            raise ValidationError("sealed evidence action is not catalog and contract bound")
        attempt_root = result_path.parent
        request_path = attempt_root / "request.json"
        execution_path = attempt_root / "execution.json"
        intent_path = attempt_root / "intent.json"
        complete_path = attempt_root / "complete.json"
        if any(path.is_symlink() or not path.is_file()
               for path in (request_path, execution_path, intent_path)):
            raise ValidationError("sealed evidence provenance path is unsafe")
        request = load_json_object(request_path)
        execution = load_json_object(execution_path)
        intent = load_json_object(intent_path)
        expected_signature = _load_evidence_selector_module()._request_signature(
            epoch["epoch_id"], action, request
        )
        bindings = {
            "request_signature": request.get("request_signature") == summary["request_signature"],
            "request_action": request.get("action_id") == action_id,
            "selector_signature": expected_signature == summary["request_signature"],
            "catalog_action": request.get("controller_action") == action,
            "intent_action": intent.get("action_sha256") == _canonical_digest(contract_action),
            "execution_action": execution.get("action_id") == action_id,
            "execution_adapter": execution.get("adapter_sha256") == contract_action["adapter_sha256"],
            "result_signature": result.get("request_signature") == summary["request_signature"],
            "result_status": result.get("status") == summary["status"],
            "result_outcome": result.get("outcome_id") == summary["outcome_id"],
        }
        if not all(bindings.values()):
            raise ValidationError(
                "sealed evidence request, action, or adapter binding drifted: "
                + ",".join(name for name, valid in bindings.items() if not valid)
            )
        pending = (pending_summary is not None and index == len(summaries) - 1
                   and summary == pending_summary)
        if complete_path.is_file() and not complete_path.is_symlink():
            complete = load_json_object(complete_path)
            expected = {
                "result_sha256": summary["result_sha256"],
                "execution_sha256": _canonical_digest(execution),
            }
            completion_context_sha = complete.get("context_sha256")
            if (complete.get("schema_version") != "cuda-optimizer/evidence-completion-v1"
                    or complete.get("request_signature") != summary["request_signature"]
                    or not isinstance(completion_context_sha, str)
                    or len(completion_context_sha) != 64
                    or any(character not in "0123456789abcdef"
                           for character in completion_context_sha)
                    or any(complete.get(field) != digest for field, digest in expected.items())):
                raise ValidationError("evidence completion digest binding drifted")
            payload = {"request_signature": summary["request_signature"], **expected,
                       "context_sha256": completion_context_sha}
            if not any(event["event_type"] == "evidence"
                       and event["payload_sha256"] == _canonical_digest(payload)
                       for event in ledger):
                raise ValidationError("evidence completion has no matching ledger payload")
        elif not pending:
            raise ValidationError("historical evidence has no sealed completion marker")
        envelopes.append({"action_id": action_id, "evidence_kind": action["evidence_kind"],
                          "adapter_implementation_sha256": contract_action["adapter_sha256"],
                          "result_sha256": summary["result_sha256"], "status": summary["status"],
                          "observations": copy.deepcopy(result["observations"])})
    frozen = {
        "knowledge_identity": copy.deepcopy(context["knowledge_identity"]),
        "diagnosis": load_json_object(run_root / "diagnosis.json"),
        "analysis_epoch": copy.deepcopy(dict(epoch)), "evidence_catalog": copy.deepcopy(dict(evidence_catalog)),
        "execution_map": copy.deepcopy(dict(execution_map)), "performance_model": copy.deepcopy(dict(performance_model)),
        # The global scan is already bound through diagnosis, the execution map,
        # and the performance model.  It is not a diagnostic-evidence-v1
        # artifact, so do not infer or duplicate semantic observations from it.
        "diagnostic_evidence": [], "active_evidence_results": envelopes,
        "requested_claim": context["requested_claim"],
        "ready_capability_ids": ready, "contract_action_ids": contract_ids,
        "available_actions": available,
        "closed_mechanism_keys": copy.deepcopy(context.get("closed_mechanism_keys", [])),
        "candidate_history": copy.deepcopy(context.get("candidate_history", [])),
    }
    try:
        return _load_diagnostic_knowledge_module().build_knowledge_context(frozen, limit=3)
    except ValueError as error:
        raise ValidationError(f"invalid frozen knowledge context: {error}") from error


def _build_active_diagnosis_context(
    control: Mapping[str, Any], run_root: Path, state: Mapping[str, Any]
) -> dict:
    contract = _load_frozen_analysis_contract(run_root, state)
    active_root = run_root / "active_diagnosis"
    scan_path = active_root / "global_scan.json"
    draft = _validate_global_scan_draft(load_json_object(scan_path))
    identities = {
        "workload_contract_sha256": _sha256_path(Path(control["workload_manifest"])),
        "environment_sha256": _sha256(
            state.get("baseline_environment_identity_digest"),
            "baseline environment identity",
        ),
        "source_sha256": _sha256(
            state.get("baseline_identity_digest"), "baseline source identity"
        ),
        "analysis_policy_sha256": contract["analysis_policy_sha256"],
    }
    epoch_seed = {
        "identities": identities,
        "source": contract["source"],
        "regime": draft["regime"],
        "boundary_ambiguous": draft["boundary_ambiguous"],
    }
    epoch_id = f"epoch-{_canonical_digest(epoch_seed)[:16]}"
    epoch = {
        "schema_version": "cuda-optimizer/analysis-epoch-v1",
        "epoch_id": epoch_id,
        "sequence": 1,
        "trigger": "initial",
        "parent_epoch_id": None,
        "started_at": state["started_at_epoch"],
        "identities": identities,
        "source": copy.deepcopy(contract["source"]),
        "regime": copy.deepcopy(draft["regime"]),
        "boundary_ambiguous": draft["boundary_ambiguous"],
    }
    epoch_module = _load_analysis_epoch_module()
    try:
        epoch = epoch_module.validate_epoch(epoch, expected_identities=identities)
    except ValueError as error:
        raise ValidationError(f"invalid Controller analysis epoch: {error}") from error
    epoch_sha = epoch_module.epoch_digest(epoch)
    evidence_catalog = {
        "ev-global-scan": {
            "epoch_id": epoch_id,
            "kind": "nsys_timeline" if contract["source"]["profiler"] == "nsys" else "global_scan",
            "artifact_sha256": _sha256_path(scan_path),
        }
    }
    execution_map = {
        "schema_version": "cuda-optimizer/execution-map-v1",
        "map_id": f"map-{epoch_id.removeprefix('epoch-')}",
        "epoch_id": epoch_id,
        "epoch_sha256": epoch_sha,
        "identities": copy.deepcopy(identities),
        "window": {
            **copy.deepcopy(draft["window"]),
            "boundary_ambiguous": draft["boundary_ambiguous"],
        },
        "coverage": copy.deepcopy(draft["coverage"]),
        "nodes": copy.deepcopy(draft["nodes"]),
        "edges": copy.deepcopy(draft["edges"]),
        "hot_path": copy.deepcopy(draft["hot_path"]),
        "uncovered_intervals": copy.deepcopy(draft["uncovered_intervals"]),
        "conclusion_level": draft["conclusion_level"],
    }
    map_module = _load_execution_map_module()
    try:
        map_result = map_module.validate_execution_map(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        )
    except ValueError as error:
        raise ValidationError(f"invalid global scan execution map: {error}") from error
    execution_map = map_result["execution_map"]
    action_catalog = load_json_object(
        Path(__file__).resolve().parents[1]
        / "references"
        / "evidence_action_catalog.json"
    )
    enabled_action_ids = {item["action_id"] for item in contract["actions"]}
    action_catalog["actions"] = [
        item
        for item in action_catalog["actions"]
        if item.get("action_id") in enabled_action_ids
    ]
    if not action_catalog["actions"]:
        raise ValidationError("analysis contract enables no catalog evidence action")
    selection_policy = copy.deepcopy(contract["selection_policy"])
    readiness_report = _load_readiness_gate_module()._load_prior_report(
        run_root / "readiness"
    )
    if readiness_report is None:
        raise ValidationError("active diagnosis requires a completed readiness report")
    selection_policy["available_capability_ids"] = _ready_capability_ids(
        readiness_report
    )
    # Replay both Controller-owned inputs now, before an AI proposal exists.
    selector = _load_evidence_selector_module()
    try:
        action_catalog = selector._validate_catalog(action_catalog)[0]
        selection_policy = selector._validate_policy(selection_policy)
    except ValueError as error:
        raise ValidationError(f"invalid active diagnosis selection inputs: {error}") from error
    _atomic_json(active_root / "epoch.json", epoch)
    _atomic_json(active_root / "evidence_catalog.json", evidence_catalog)
    _atomic_json(active_root / "execution_map.json", execution_map)
    _atomic_json(active_root / "action_catalog.json", action_catalog)
    _atomic_json(active_root / "selection_policy.json", selection_policy)
    request_history = []
    _atomic_json(active_root / "request_history.json", request_history)
    completed_action_ids = (
        ["nsys-global-timeline"]
        if contract["source"]["profiler"] == "nsys"
        and "nsys-global-timeline" in enabled_action_ids
        else []
    )
    _atomic_json(
        active_root / "completed_action_ids.json",
        completed_action_ids,
    )
    performance_model = _load_performance_model_module().build_performance_model(
        execution_map,
        minimum_effect_us=contract["minimum_effect_us"],
        action_timings=[],
    )
    _atomic_json(active_root / "performance_model.json", performance_model)
    try:
        initial_brief = (
            _load_diagnostic_decision_module().build_initial_investment_brief(
                performance_model,
                _bootstrap_execution_seconds(run_root, state),
            )
        )
    except ValueError as error:
        raise ValidationError(
            f"invalid initial investment brief: {error}"
        ) from error
    _atomic_json(
        active_root / "initial_investment_brief.json",
        initial_brief,
    )
    diagnosis = load_json_object(run_root / "diagnosis.json")
    knowledge_identity = _readiness_knowledge_identity(
        run_root, readiness_report, contract, identities
    )
    project_surface_identity = _project_surface_identity(Path(control["project_root"]))
    _atomic_json(
        active_root / "project_surface_identity.json", project_surface_identity
    )
    context = {
        "schema_version": "cuda-optimizer/diagnosis-context-v1",
        "epoch_id": epoch_id,
        "epoch_sha256": epoch_sha,
        "execution_map_sha256": map_module.execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        ),
        "performance_model_sha256": _canonical_digest(performance_model),
        "evidence_catalog_sha256": _canonical_digest(evidence_catalog),
        "action_catalog_sha256": _canonical_digest(action_catalog),
        "selection_policy_sha256": _canonical_digest(selection_policy),
        "request_history_sha256": _canonical_digest(request_history),
        "completed_action_ids_sha256": _canonical_digest(completed_action_ids),
        "diagnosis_sha256": _canonical_digest(diagnosis),
        "project_surface_identity_sha256": _canonical_digest(
            project_surface_identity
        ),
        "knowledge_identity": knowledge_identity,
        "requested_claim": readiness_report["requested_claim"],
        "knowledge_context": {},
        "requires_unmodeled_hypothesis": map_result[
            "requires_unmodeled_hypothesis"
        ],
        "evidence_results": [],
        "candidate_history": [],
        "closed_mechanism_keys": [],
        "closed_scope_records": [],
    }
    context["knowledge_context"] = _rebuild_knowledge_context(
        run_root, context, contract, epoch, execution_map, evidence_catalog,
        selection_policy, performance_model,
    )
    _atomic_json(run_root / "diagnosis_context.json", context)
    _atomic_json(active_root / "knowledge_context.json", context["knowledge_context"])
    _append_active_diagnosis_event(run_root, "context", context)
    return context


def _load_active_diagnosis_context(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    *,
    verify_current_project_surface: bool = True,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    if _identity(control, "project")["digest"] != state.get(
        "baseline_identity_digest"
    ):
        raise ValidationError("project identity drifted after diagnosis context")
    workload = _normalize_frozen_workload(control)
    if workload.source_hash != state.get("workload_source_hash"):
        raise ValidationError("workload identity drifted after diagnosis context")
    if _identity(control, "isolated_environment")["digest"] != state.get(
        "baseline_environment_identity_digest"
    ):
        raise ValidationError("environment identity drifted after diagnosis context")
    _load_frozen_analysis_contract(run_root, state)
    events = _verify_active_diagnosis_ledger(run_root)
    _verify_committed_active_ledger(state, events)
    active_root = run_root / "active_diagnosis"
    epoch = load_json_object(active_root / "epoch.json")
    evidence_catalog = load_json_object(active_root / "evidence_catalog.json")
    execution_map = load_json_object(active_root / "execution_map.json")
    action_catalog = load_json_object(active_root / "action_catalog.json")
    selection_policy = load_json_object(active_root / "selection_policy.json")
    request_history = json.loads(
        (active_root / "request_history.json").read_text(encoding="utf-8")
    )
    completed_action_ids = json.loads(
        (active_root / "completed_action_ids.json").read_text(encoding="utf-8")
    )
    project_surface_identity = load_json_object(
        active_root / "project_surface_identity.json"
    )
    context = load_json_object(run_root / "diagnosis_context.json")
    if _canonical_digest(context) != state.get("diagnosis_context_sha256"):
        raise ValidationError("diagnosis context digest does not match state")
    if _canonical_digest(project_surface_identity) != context.get(
        "project_surface_identity_sha256"
    ):
        raise ValidationError("diagnosis context project surface identity drifted")
    if (
        verify_current_project_surface
        and _project_surface_identity(Path(control["project_root"]))
        != project_surface_identity
    ):
        raise ValidationError("complete project surface drifted after diagnosis context")
    if type(request_history) is not list or type(completed_action_ids) is not list:
        raise ValidationError("active diagnosis histories are invalid")
    result_summaries = context.get("evidence_results", [])
    if type(result_summaries) is not list:
        raise ValidationError("diagnosis context evidence_results is invalid")
    candidate_history = context.get("candidate_history", [])
    if type(candidate_history) is not list:
        raise ValidationError("diagnosis context candidate_history is invalid")
    for index, raw_record in enumerate(candidate_history):
        record = _object(raw_record, f"candidate_history[{index}]")
        fields = {
            "hypothesis_id",
            "action_id",
            "implementation_status",
            "identity_digest",
            "elapsed_seconds",
            "candidate_digest",
            "decision_digest",
            "failure_reason",
        }
        _closed(record, fields, f"candidate_history[{index}]")
        _required(record, fields, f"candidate_history[{index}]")
        _identifier(record["hypothesis_id"], f"candidate_history[{index}].hypothesis_id")
        _identifier(record["action_id"], f"candidate_history[{index}].action_id")
        if record["implementation_status"] != "failed":
            raise ValidationError("candidate history records must describe failed candidates")
        for field in ("identity_digest", "candidate_digest", "decision_digest"):
            _sha256(record[field], f"candidate_history[{index}].{field}")
        elapsed = record["elapsed_seconds"]
        if (
            type(elapsed) not in {int, float}
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise ValidationError("candidate history elapsed_seconds is invalid")
        _identifier(record["failure_reason"], f"candidate_history[{index}].failure_reason")
    closed_mechanism_keys = context.get("closed_mechanism_keys", [])
    if type(closed_mechanism_keys) is not list:
        raise ValidationError("diagnosis context closed_mechanism_keys is invalid")
    hypothesis_module = _load_hypothesis_space_module()
    normalized_closed = [
        hypothesis_module.canonical_mechanism_key(item)
        for item in closed_mechanism_keys
    ]
    if normalized_closed != sorted(set(normalized_closed)):
        raise ValidationError("diagnosis context closed_mechanism_keys is not canonical")
    closed_scope_records = context.get("closed_scope_records", [])
    if type(closed_scope_records) is not list:
        raise ValidationError("diagnosis context closed_scope_records is invalid")
    seen_closed_scope_records = set()
    for index, raw_record in enumerate(closed_scope_records):
        record = _object(raw_record, f"closed_scope_records[{index}]")
        fields = {
            "hypothesis_id",
            "mechanism_key",
            "kind",
            "claim_layer",
            "scope_node_ids",
            "known_evidence_ids",
        }
        _closed(record, fields, f"closed_scope_records[{index}]")
        _required(record, fields, f"closed_scope_records[{index}]")
        for field in ("hypothesis_id", "kind", "claim_layer"):
            if type(record[field]) is not str or not record[field]:
                raise ValidationError(
                    f"closed_scope_records[{index}].{field} is invalid"
                )
        mechanism_key = hypothesis_module.canonical_mechanism_key(
            record["mechanism_key"]
        )
        if mechanism_key != record["mechanism_key"]:
            raise ValidationError(
                "diagnosis context closed_scope_records mechanism key is not canonical"
            )
        scope = record["scope_node_ids"]
        if (
            type(scope) is not list
            or not scope
            or any(type(item) is not str or not item for item in scope)
            or scope != sorted(set(scope))
        ):
            raise ValidationError(
                "diagnosis context closed_scope_records scope is not canonical"
            )
        known_evidence = record["known_evidence_ids"]
        if (
            type(known_evidence) is not list
            or any(type(item) is not str or not item for item in known_evidence)
            or known_evidence != sorted(set(known_evidence))
            or not set(known_evidence).issubset(evidence_catalog)
        ):
            raise ValidationError(
                "diagnosis context closed_scope_records evidence history is invalid"
            )
        record_key = (
            record["hypothesis_id"],
            mechanism_key,
            record["kind"],
            record["claim_layer"],
            tuple(scope),
        )
        if record_key in seen_closed_scope_records:
            raise ValidationError(
                "diagnosis context closed_scope_records contains duplicates"
            )
        seen_closed_scope_records.add(record_key)
    for index, raw_summary in enumerate(result_summaries):
        summary = _object(raw_summary, f"evidence_results[{index}]")
        summary_fields = {
            "request_signature",
            "action_id",
            "evidence_id",
            "status",
            "outcome_id",
            "result_path",
            "result_sha256",
            "duration_seconds",
        }
        _closed(summary, summary_fields, f"evidence_results[{index}]")
        _required(summary, summary_fields, f"evidence_results[{index}]")
        relative = _relative(summary["result_path"], f"evidence_results[{index}].result_path")
        result_path = run_root / relative
        resolved_result = result_path.resolve(strict=False)
        if (
            not _is_within(resolved_result, run_root)
            or result_path.is_symlink()
            or not result_path.is_file()
        ):
            raise ValidationError("evidence result path is not a contained regular file")
        result = load_json_object(result_path)
        if _canonical_digest(result) != summary["result_sha256"]:
            raise ValidationError("evidence result content digest does not match context")
        if result.get("request_signature") != summary["request_signature"]:
            raise ValidationError("evidence result request signature does not match context")
        duration = summary["duration_seconds"]
        if (
            type(duration) not in {int, float}
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise ValidationError("evidence result duration_seconds must be non-negative and finite")
        artifacts = result.get("artifacts")
        if type(artifacts) is not list:
            raise ValidationError("evidence result artifacts are invalid")
        for artifact_index, raw_artifact in enumerate(artifacts):
            artifact = _object(
                raw_artifact,
                f"evidence_results[{index}].artifacts[{artifact_index}]",
            )
            _closed(
                artifact,
                {"path", "sha256"},
                f"evidence_results[{index}].artifacts[{artifact_index}]",
            )
            _required(
                artifact,
                {"path", "sha256"},
                f"evidence_results[{index}].artifacts[{artifact_index}]",
            )
            relative_artifact = _relative(
                artifact["path"],
                f"evidence_results[{index}].artifacts[{artifact_index}].path",
            )
            artifact_path = result_path.parent / relative_artifact
            if (
                not _is_within(artifact_path.resolve(strict=False), result_path.parent)
                or artifact_path.is_symlink()
                or not artifact_path.is_file()
            ):
                raise ValidationError("evidence artifact path is not a contained regular file")
            if _sha256_path(artifact_path) != artifact["sha256"]:
                raise ValidationError("evidence artifact content digest does not match result")
        evidence_id = summary["evidence_id"]
        if evidence_id is not None:
            catalog_item = evidence_catalog.get(evidence_id)
            if type(catalog_item) is not dict:
                raise ValidationError("evidence result is missing from evidence catalog")
            if catalog_item.get("artifact_sha256") != _sha256_path(result_path):
                raise ValidationError("evidence result artifact digest does not match catalog")
    epoch_module = _load_analysis_epoch_module()
    expected_identities = {
        "workload_contract_sha256": _sha256_path(Path(control["workload_manifest"])),
        "environment_sha256": state["baseline_environment_identity_digest"],
        "source_sha256": state["baseline_identity_digest"],
        "analysis_policy_sha256": _load_frozen_analysis_contract(run_root, state)[
            "analysis_policy_sha256"
        ],
    }
    try:
        epoch = epoch_module.validate_epoch(
            epoch, expected_identities=expected_identities
        )
        execution_map = _load_execution_map_module().validate_execution_map(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        )["execution_map"]
    except ValueError as error:
        raise ValidationError(f"active diagnosis context validation failed: {error}") from error
    contract = _load_frozen_analysis_contract(run_root, state)
    action_timings = [
        {
            "action_id": item["action_id"],
            "identities": copy.deepcopy(execution_map["identities"]),
            "elapsed_seconds": item["duration_seconds"],
        }
        for item in result_summaries
        if item["duration_seconds"] > 0
    ]
    try:
        expected_performance_model = _load_performance_model_module().build_performance_model(
            execution_map,
            minimum_effect_us=contract["minimum_effect_us"],
            action_timings=action_timings,
        )
    except ValueError as error:
        raise ValidationError(f"active diagnosis performance model is invalid: {error}") from error
    performance_model = load_json_object(active_root / "performance_model.json")
    if performance_model != expected_performance_model:
        raise ValidationError("active diagnosis performance model drifted")
    expected = {
        "epoch_sha256": epoch_module.epoch_digest(epoch),
        "execution_map_sha256": _load_execution_map_module().execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        ),
        "performance_model_sha256": _canonical_digest(performance_model),
        "evidence_catalog_sha256": _canonical_digest(evidence_catalog),
        "action_catalog_sha256": _canonical_digest(action_catalog),
        "selection_policy_sha256": _canonical_digest(selection_policy),
        "request_history_sha256": _canonical_digest(request_history),
        "completed_action_ids_sha256": _canonical_digest(completed_action_ids),
    }
    for field, digest in expected.items():
        if context.get(field) != digest:
            raise ValidationError(f"diagnosis context {field} drifted")
    _object(context.get("knowledge_context"), "diagnosis context knowledge_context")
    mirror_path = active_root / "knowledge_context.json"
    if mirror_path.is_symlink() or (
        mirror_path.exists() and not mirror_path.is_file()
    ):
        raise ValidationError("knowledge context mirror is unsafe")
    mirror = None
    if mirror_path.is_file():
        try:
            mirror = load_json_object(mirror_path)
        except ValidationError:
            mirror = None
    if mirror != context["knowledge_context"]:
        _atomic_json(mirror_path, context["knowledge_context"])
    return (
        context,
        epoch,
        execution_map,
        evidence_catalog,
        action_catalog,
        selection_policy,
    )


def start_run(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    project_root = Path(normalized["project_root"])
    if _is_within(run_root, project_root):
        raise ValidationError("run_dir must be outside project_root")
    run_root.mkdir(parents=True, exist_ok=True)
    with _run_lock(run_root):
        return _start_run_unlocked(normalized, run_root)


def _start_run_unlocked(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    """Initialize evidence, baseline, probes, and diagnosis up to the change boundary."""
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    project_root = Path(normalized["project_root"])
    if _is_within(run_root, project_root):
        raise ValidationError("run_dir must be outside project_root")
    control_digest = _canonical_digest(normalized)
    state_path = run_root / "state.json"
    if any(
        path.exists() or path.is_symlink()
        for path in (
            state_path,
            run_root / "state_commit.json",
            run_root / "state_generations",
        )
    ):
        state = read_run_state(run_root)
        _require_run_grant_investment_control(state)
        if state["control_digest"] != control_digest:
            raise ValidationError("control manifest drifted after run initialization")
        _load_frozen_control(run_root, state)
        if state["next_action"] in {
            "readiness_action",
            "propose_hypotheses",
            "collect_evidence",
            "evidence_gap",
            "review_required",
            "register_change",
            "edit_then_evaluate",
            "done",
            "manual_recovery",
        }:
            return state
    else:
        run_root.mkdir(parents=True, exist_ok=True)
        readiness_contract = None
        readiness_contract_digest = None
        analysis_contract = None
        analysis_contract_digest = None
        analysis_execution_bindings = None
        if normalized["schema_version"] == CONTROL_SCHEMA_V2:
            readiness_module = _load_readiness_contract_module()
            readiness_contract = readiness_module.validate_contract(
                readiness_module.load_contract(normalized["readiness_contract"]),
                project_root=project_root,
                environment_root=Path(
                    normalized["mutation"]["environment_root"]
                ),
            )
            readiness_contract_digest = readiness_module.contract_digest(
                readiness_contract
            )
        if "analysis_contract" in normalized:
            analysis_contract = _validate_active_diagnosis_contract(
                load_json_object(normalized["analysis_contract"])
            )
            matching_global_probes = [
                item
                for item in normalized["probes"]
                if item["id"] == analysis_contract["global_scan_probe_id"]
            ]
            if len(matching_global_probes) != 1:
                raise ValidationError(
                    "analysis_contract global_scan_probe_id must name a control probe"
                )
            adapter_path = _absolute(
                analysis_contract["adapter_path"],
                "analysis_contract.adapter_path",
            )
            if not _is_within(adapter_path, project_root):
                raise ValidationError(
                    "analysis_contract adapter_path must be inside project_root"
                )
            if (
                not adapter_path.is_file()
                or adapter_path.is_symlink()
                or adapter_path.stat().st_uid != os.getuid()
            ):
                raise ValidationError(
                    "analysis_contract adapter must be a user-owned regular file"
                )
            if _sha256_path(adapter_path) != analysis_contract["source"][
                "adapter_sha256"
            ]:
                raise ValidationError(
                    "analysis_contract adapter digest does not match adapter_path"
                )
            analysis_execution_bindings = {
                "schema_version": "cuda-optimizer/analysis-execution-bindings-v1",
                "global_scan": _adapter_execution_binding(
                    adapter_path,
                    matching_global_probes[0]["argv"],
                    "analysis_contract global scan",
                ),
                "actions": {},
            }
            for action in analysis_contract["actions"]:
                action_adapter = Path(action["adapter_path"])
                if not _is_within(action_adapter, project_root):
                    raise ValidationError(
                        "analysis_contract action adapter_path must be inside project_root"
                    )
                if (
                    not action_adapter.is_file()
                    or action_adapter.is_symlink()
                    or action_adapter.stat().st_uid != os.getuid()
                ):
                    raise ValidationError(
                        "analysis_contract action adapter must be a user-owned regular file"
                    )
                if _sha256_path(action_adapter) != action["adapter_sha256"]:
                    raise ValidationError(
                        "analysis_contract action adapter digest does not match adapter_path"
                    )
                analysis_execution_bindings["actions"][action["action_id"]] = (
                    _adapter_execution_binding(
                        action_adapter,
                        action["argv"],
                        f"analysis_contract action {action['action_id']}",
                    )
                )
            analysis_contract_digest = _canonical_digest(analysis_contract)
        baseline_identity = _identity(normalized, "project")
        environment_root = Path(normalized["mutation"]["environment_root"])
        environment_identity = None
        if environment_root.exists() or environment_root.is_symlink():
            if (
                environment_root.is_symlink()
                or not environment_root.is_dir()
                or environment_root.stat().st_uid != os.getuid()
            ):
                raise ValidationError(
                    "existing environment_root must be a user-owned non-symlink directory"
                )
            environment_identity = _identity(normalized, "isolated_environment")
        workload = _normalize_frozen_workload(normalized)
        if _identity(normalized, "project")["digest"] != baseline_identity["digest"]:
            raise ValidationError(
                "declared project identity changed while loading the workload adapter"
            )
        if environment_identity is not None and _identity(
            normalized, "isolated_environment"
        )["digest"] != environment_identity["digest"]:
            raise ValidationError(
                "isolated environment changed while loading the workload adapter"
            )
        now = time.time()
        runtime = _BUDGET_RUNTIME[normalized["budget"]]
        state = {
            "schema_version": "cuda-workload-optimizer/state-v1",
            "status": "active",
            "stage": (
                "readiness"
                if normalized["schema_version"] == CONTROL_SCHEMA_V2
                else "baseline"
            ),
            "round": 1,
            "completed_stages": [],
            "next_action": (
                "readiness"
                if normalized["schema_version"] == CONTROL_SCHEMA_V2
                else "baseline"
            ),
            "control_digest": control_digest,
            "workload_source_hash": workload.source_hash,
            "investment_control_version": "run-grant-v1",
            "controlled_spend_seconds": 0.0,
            "started_at_epoch": now,
            "updated_at_epoch": now,
            "soft_target_epoch": now + runtime["soft_target_seconds"],
            "deadline_epoch": now + runtime["hard_ceiling_seconds"],
        }
        _atomic_json(run_root / "control_manifest.json", normalized)
        if readiness_contract is not None:
            _atomic_json(
                run_root / "readiness_contract.json", readiness_contract
            )
            state["readiness_contract_digest"] = readiness_contract_digest
            state["readiness_report_digest"] = None
        if analysis_contract is not None:
            _atomic_json(
                run_root / "active_diagnosis" / "analysis_contract.json",
                analysis_contract,
            )
            state["analysis_contract_digest"] = analysis_contract_digest
            _atomic_json(
                run_root / "active_diagnosis" / "execution_bindings.json",
                analysis_execution_bindings,
            )
            state["analysis_execution_bindings_digest"] = _canonical_digest(
                analysis_execution_bindings
            )
        _atomic_json(run_root / "baseline_identity.json", baseline_identity)
        state["baseline_identity_digest"] = baseline_identity["digest"]
        state["baseline_environment_identity_digest"] = (
            None if environment_identity is None else environment_identity["digest"]
        )
        if environment_identity is not None:
            _atomic_json(
                run_root / "baseline_environment_identity.json",
                environment_identity,
            )
        (run_root / "host_recommendations.md").write_text(
            "# Host recommendations\n\nNo host mutation was executed. Add evidence-backed suggestions here for manual review.\n",
            encoding="utf-8",
        )
        state = _write_state(run_root, state)

    runtime = _BUDGET_RUNTIME[normalized["budget"]]
    if (
        normalized["schema_version"] == CONTROL_SCHEMA_V2
        and "readiness" in state["completed_stages"]
        and "optimization_started_at_epoch" not in state
    ):
        report_path = run_root / "readiness" / "report.json"
        if not report_path.is_file() or report_path.is_symlink():
            raise ValidationError(
                "legacy readiness state lacks a regular completion report"
            )
        state = migrate_completed_readiness_budget(
            state,
            runtime,
            completed_at=report_path.stat().st_mtime,
        )
        state = _write_state(run_root, state)
    if not (
        normalized["schema_version"] == CONTROL_SCHEMA_V2
        and "readiness" not in state["completed_stages"]
    ):
        _check_deadline(state)
    _load_frozen_control(run_root, state)
    workload = _normalize_frozen_workload(normalized)
    if workload.source_hash != state["workload_source_hash"]:
        raise ValidationError("workload identity drifted after run initialization")

    if (
        normalized["schema_version"] == CONTROL_SCHEMA_V2
        and "readiness" not in state["completed_stages"]
    ):
        report = _run_readiness_gate_checked(normalized, run_root, state)
        state = copy.deepcopy(state)
        state["readiness_report_digest"] = _readiness_report_digest(
            run_root, report
        )
        state["readiness_environment_identity_digest"] = report.get(
            "environment_identity_digest"
        )
        if not report.get("can_start_diagnosis"):
            state["stage"] = "readiness"
            state["next_action"] = "readiness_action"
            state["updated_at_epoch"] = time.time()
            return _write_state(run_root, state)
        if _identity(normalized, "project")["digest"] != state.get(
            "baseline_identity_digest"
        ):
            raise ValidationError(
                "declared project identity drifted during readiness"
            )
        if _normalize_frozen_workload(normalized).source_hash != state.get(
            "workload_source_hash"
        ):
            raise ValidationError("workload identity drifted during readiness")
        refreshed_environment = _identity(
            normalized, "isolated_environment"
        )
        _atomic_json(
            run_root / "baseline_environment_identity.json",
            refreshed_environment,
        )
        state["baseline_environment_identity_digest"] = refreshed_environment[
            "digest"
        ]
        state = start_optimization_budget_after_readiness(
            state, runtime, now=time.time()
        )
        state = _advance(
            run_root,
            state,
            "readiness",
            stage="baseline",
            next_action="baseline",
        )

    if normalized["schema_version"] == CONTROL_SCHEMA_V2:
        if not _verify_readiness_report(normalized, run_root, state):
            report = _run_readiness_gate_checked(normalized, run_root, state)
            state = copy.deepcopy(state)
            state["readiness_report_digest"] = _readiness_report_digest(
                run_root, report
            )
            state["readiness_environment_identity_digest"] = report.get(
                "environment_identity_digest"
            )
            state["updated_at_epoch"] = time.time()
            state = _write_state(run_root, state)
            if not report.get("can_start_diagnosis"):
                state["stage"] = "readiness"
                state["next_action"] = "readiness_action"
                return _write_state(run_root, state)

    if "baseline" not in state["completed_stages"]:
        baseline_surface_before = (
            _project_surface_identity(Path(normalized["project_root"]))
            if "analysis_contract" in normalized
            else None
        )
        timeout = min(
            120.0, max(0.001, state["deadline_epoch"] - time.time())
        )
        baseline_attempt = 0

        def run_baseline_once(
            evaluation_workload: Any,
            *,
            candidate: Any,
            role: str,
            case: Mapping[str, Any] | None = None,
            timeout: float | None = None,
        ) -> dict:
            nonlocal baseline_attempt
            if evaluation_workload.source_hash != workload.source_hash:
                raise ValidationError("workload changed inside baseline capture")
            effective_timeout = min(
                max(0.001, state["deadline_epoch"] - time.time()),
                120.0 if timeout is None else float(timeout),
            )
            baseline_attempt += 1
            return _run_python_workload_once_bounded(
                normalized,
                run_root,
                candidate=candidate,
                role=role,
                case=case,
                timeout_seconds=effective_timeout,
                task=f"baseline-workload-{baseline_attempt}",
            )

        baseline_started = time.monotonic()
        baseline = _load_evaluate_module().measure_candidate(
            workload,
            normalized["baseline_candidate"],
            role="baseline",
            retries=runtime["retries"],
            timeout=timeout,
            deadline_epoch=state["deadline_epoch"],
            runner=run_baseline_once if workload.kind == "python" else None,
        )
        baseline_duration = max(0.0, time.monotonic() - baseline_started)
        _record_baseline_execution(
            run_root,
            workload_kind=workload.kind,
            status=baseline.get("status"),
            duration_seconds=baseline_duration,
        )
        if (
            baseline_surface_before is not None
            and _project_surface_identity(Path(normalized["project_root"]))
            != baseline_surface_before
        ):
            raise ValidationError("baseline modified the complete project surface")
        _atomic_json(run_root / "baseline" / "observation.json", baseline)
        if baseline["status"] != "measured":
            raise ValidationError("baseline workload failed; see baseline/observation.json")
        state = _advance(
            run_root, state, "baseline", stage="probes", next_action="probes"
        )
    _check_deadline(state)
    if "probes" not in state["completed_stages"]:
        if (
            normalized["schema_version"] == CONTROL_SCHEMA_V2
            and not _verify_readiness_report(normalized, run_root, state)
        ):
            report = _run_readiness_gate_checked(normalized, run_root, state)
            state = copy.deepcopy(state)
            state["readiness_report_digest"] = _readiness_report_digest(
                run_root, report
            )
            state["readiness_environment_identity_digest"] = report.get(
                "environment_identity_digest"
            )
            state["updated_at_epoch"] = time.time()
            if not report.get("can_start_diagnosis"):
                state["stage"] = "readiness"
                state["next_action"] = "readiness_action"
                return _write_state(run_root, state)
            state = _write_state(run_root, state)
        probe_surface_before = (
            _project_surface_identity(Path(normalized["project_root"]))
            if "analysis_contract" in normalized
            else None
        )
        probe_identity_before = _identity(normalized, "project")
        run_probes(
            normalized,
            run_root,
            deadline_epoch=state["deadline_epoch"],
        )
        if _identity(normalized, "project") != probe_identity_before:
            raise ValidationError("diagnosis probes modified declared project inputs")
        if (
            probe_surface_before is not None
            and _project_surface_identity(Path(normalized["project_root"]))
            != probe_surface_before
        ):
            raise ValidationError("diagnosis probes modified the complete project surface")
        state = _advance(
            run_root, state, "probes", stage="diagnosis", next_action="diagnosis"
        )
    _check_deadline(state)
    if "diagnosis" not in state["completed_stages"]:
        diagnose_run(run_root)
        if "analysis_contract" in normalized:
            state = _advance(
                run_root,
                state,
                "diagnosis",
                stage="active_diagnosis",
                next_action="diagnosis_context",
            )
        else:
            state = _advance(
                run_root,
                state,
                "diagnosis",
                stage="change",
                next_action="register_change",
            )
    if (
        "analysis_contract" in normalized
        and "diagnosis_context" not in state["completed_stages"]
    ):
        context = _build_active_diagnosis_context(normalized, run_root, state)
        updated = copy.deepcopy(state)
        if "diagnosis_context" not in updated["completed_stages"]:
            updated["completed_stages"].append("diagnosis_context")
        updated["stage"] = "active_diagnosis"
        updated["next_action"] = "propose_hypotheses"
        updated["updated_at_epoch"] = time.time()
        updated["diagnosis_context_sha256"] = _canonical_digest(context)
        initial_brief = load_json_object(
            run_root
            / "active_diagnosis"
            / "initial_investment_brief.json"
        )
        updated["initial_investment_brief_sha256"] = _canonical_digest(
            initial_brief
        )
        updated.update(
            _active_ledger_binding(_verify_active_diagnosis_ledger(run_root))
        )
        state = _write_state(run_root, updated)
    return state


def _adapt_controller_knowledge(
    knowledge_inputs: Mapping[str, Any] | None,
    *,
    contract: Mapping[str, Any],
    execution_map: Mapping[str, Any],
    action_catalog: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    knowledge_identity: Mapping[str, Any],
    local_knowledge_context: Mapping[str, Any],
    hypothesis_set: Mapping[str, Any],
    request_set: Mapping[str, Any],
) -> tuple[dict, dict, dict]:
    """Merge state-bound local candidates with neutralized external shadows."""
    if knowledge_inputs is None:
        raw_inputs = {}
    else:
        raw_inputs = _object(knowledge_inputs, "knowledge_inputs")
        _closed(raw_inputs, {"bundled", "searched", "external"}, "knowledge_inputs")
    contract_action_ids = {item["action_id"] for item in contract["actions"]}
    ready_capabilities = set(selection_policy.get("available_capability_ids", []))
    by_action = {
        item["action_id"]: item
        for item in action_catalog.get("actions", [])
        if item.get("control_scope") == "read_only"
        and item.get("action_id") in contract_action_ids
        and set(item.get("required_capability_ids", [])).issubset(ready_capabilities)
    }
    nodes = {
        item["node_id"]: item for item in execution_map.get("nodes", [])
    }
    hypothesis_module = _load_hypothesis_space_module()
    formal_context = _object(
        local_knowledge_context, "local_knowledge_context"
    )
    if formal_context.get("schema_version") != "cuda-optimizer/knowledge-context-v1":
        raise ValidationError("local knowledge context schema is unsupported")
    raw_local_candidates = formal_context.get("candidates")
    if type(raw_local_candidates) is not list:
        raise ValidationError("local knowledge context candidates are invalid")
    local_candidates = []
    for index, raw_candidate in enumerate(raw_local_candidates):
        candidate = _object(
            raw_candidate, f"local_knowledge_context.candidates[{index}]"
        )
        mechanism_key = candidate.get("mechanism_key")
        if (
            type(mechanism_key) is not str
            or not mechanism_key
            or hypothesis_module.canonical_mechanism_key(mechanism_key)
            != mechanism_key
            or candidate.get("confidence") != "inconclusive"
            or candidate.get("promotion_authority") != "none"
        ):
            raise ValidationError("formal local knowledge candidate is not canonical")
        scope_node_ids = candidate.get("scope_node_ids")
        execution_layers = candidate.get("execution_layers")
        if (
            type(scope_node_ids) is not list
            or not scope_node_ids
            or scope_node_ids != sorted(set(scope_node_ids))
            or not set(scope_node_ids).issubset(nodes)
            or type(execution_layers) is not list
            or not execution_layers
            or not {
                nodes[node_id]["layer"] for node_id in scope_node_ids
            }.issubset(set(execution_layers))
        ):
            raise ValidationError("formal local knowledge scope is invalid")
        falsifier = candidate.get("cheapest_falsifier")
        if not isinstance(falsifier, Mapping):
            raise ValidationError("formal local knowledge falsifier is invalid")
        action = by_action.get(falsifier.get("action_id"))
        if not isinstance(action, Mapping):
            raise ValidationError(
                "formal local knowledge action is outside current authorization"
            )
        statement = candidate.get("statement")
        rationale = falsifier.get("rationale")
        if (
            type(statement) is not str
            or not statement
            or type(rationale) is not str
            or not rationale
        ):
            raise ValidationError("formal local knowledge wording is invalid")
        local_candidates.append(
            {
                "mechanism_id": mechanism_key,
                "mechanism_key": mechanism_key,
                "statement": statement,
                "scope_node_ids": copy.deepcopy(scope_node_ids),
                "execution_layers": copy.deepcopy(execution_layers),
                "unmodeled_interval_id": None,
                "falsification_question": rationale,
                "evidence_action": {
                    "action_id": action["action_id"],
                    "evidence_kind": action["evidence_kind"],
                    "outcomes": ["falsified", "inconclusive"],
                    "risk": action["risk"],
                    "control_scope": action["control_scope"],
                },
                "cheapest_falsifier": {
                    "action_id": action["action_id"],
                    "rationale": rationale,
                },
                "risk": action["risk"],
                "origin": "local",
                "claim_layer": "workload",
                "confidence": "inconclusive",
                "promotion_authority": "none",
            }
        )
    architecture = knowledge_identity.get("gpu_architecture", {})
    runtime = knowledge_identity.get("cuda_runtime_version", {})
    if architecture.get("status") != "verified" or runtime.get("status") != "verified":
        local_candidates = []
        raw_adaptation = {
            "knowledge_support": "unavailable",
            "candidates": [],
            "rejections": [
                {
                    "origin": "controller",
                    "reason": "knowledge_identity_unverified",
                }
            ],
        }
    else:
        adapter_context = {
            "architecture": architecture["value"], "software_version": runtime["value"],
            "execution_node_ids": sorted(item["node_id"] for item in execution_map.get("nodes", [])),
            "execution_node_layers": {item["node_id"]: item["layer"] for item in execution_map.get("nodes", [])},
            "uncovered_interval_ids": [], "available_evidence_action_ids": sorted(by_action),
            "authorized_risk": contract["selection_policy"]["max_risk"], "authorized_scope": "read_only",
        }
    if architecture.get("status") == "verified" and runtime.get("status") == "verified" and not adapter_context["available_evidence_action_ids"]:
        raw_adaptation = {
            "knowledge_support": "unavailable",
            "candidates": [],
            "rejections": [
                {"origin": "controller", "reason": "no_local_read_only_action"}
            ],
        }
    elif architecture.get("status") == "verified" and runtime.get("status") == "verified":
        try:
            raw_adaptation = _load_knowledge_adapter_module().recommend(
                adapter_context,
                bundled=raw_inputs.get("bundled", ()),
                searched=raw_inputs.get("searched", ()),
                external=raw_inputs.get("external", ()),
                limit=3,
            )
        except ValueError as error:
            raw_adaptation = {
                "knowledge_support": "unavailable",
                "candidates": [],
                "rejections": [
                    {
                        "origin": "controller",
                        "reason": f"adapter_unavailable:{type(error).__name__}",
                    }
                ],
            }
    external_candidates = []
    rejections = copy.deepcopy(raw_adaptation.get("rejections", []))
    for candidate in raw_adaptation.get("candidates", []):
        action = candidate.get("evidence_action")
        local = by_action.get(action.get("action_id")) if isinstance(action, Mapping) else None
        if (
            not isinstance(local, Mapping)
            or action.get("evidence_kind") != local.get("evidence_kind")
            or action.get("risk") != local.get("risk")
            or action.get("control_scope") != local.get("control_scope")
            or sorted(action.get("outcomes", [])) != ["falsified", "inconclusive"]
        ):
            rejections.append(
                {
                    "origin": candidate.get("origin", "unknown"),
                    "reason": "controller_action_or_outcome_mismatch",
                }
            )
            continue
        normalized = copy.deepcopy(candidate)
        normalized["claim_layer"] = "workload"
        external_candidates.append(normalized)
    valid_candidates = []
    seen_mechanisms = set()
    for candidate in (*local_candidates, *external_candidates):
        mechanism_key = candidate["mechanism_key"]
        if mechanism_key in seen_mechanisms:
            rejections.append(
                {
                    "origin": candidate.get("origin", "unknown"),
                    "reason": "canonical_duplicate",
                }
            )
            continue
        if len(valid_candidates) == 3:
            break
        seen_mechanisms.add(mechanism_key)
        valid_candidates.append(candidate)
    adaptation = {
        "knowledge_support": "available" if valid_candidates else "unavailable",
        "candidates": valid_candidates,
        "rejections": rejections,
    }
    augmented_hypotheses = _json_copy(
        hypothesis_set, "hypothesis_set", reject_sensitive=True
    )
    augmented_requests = _json_copy(request_set, "request_set", reject_sensitive=True)
    active_count = sum(
        item.get("disposition") == "active"
        for item in augmented_hypotheses.get("hypotheses", [])
    )
    existing_keys = {
        hypothesis_module.canonical_mechanism_key(item["mechanism"])
        for item in augmented_hypotheses.get("hypotheses", [])
        if item.get("disposition") == "active"
    }
    slots = max(0, 3 - active_count)
    for candidate in valid_candidates:
        mechanism_key = candidate["mechanism_key"]
        if slots <= 0 or mechanism_key in existing_keys:
            continue
        shadow_id = f"knowledge-{_canonical_digest(candidate)[:16]}"
        action = candidate["evidence_action"]
        augmented_hypotheses["hypotheses"].append(
            {
                "hypothesis_id": shadow_id,
                "kind": "mechanism",
                "scope_node_ids": copy.deepcopy(candidate["scope_node_ids"]),
                "statement": candidate["statement"],
                "mechanism": candidate["mechanism_id"],
                "claim_layer": "workload",
                "disposition": "active",
                "confidence": "inconclusive",
                "support_evidence_ids": [],
                "oppose_evidence_ids": [],
                "missing_evidence_kinds": [action["evidence_kind"]],
                "falsification_question": candidate["falsification_question"],
            }
        )
        augmented_requests.setdefault("requests", []).append(
            {
                "request_id": f"request-{shadow_id}",
                "action_id": action["action_id"],
                "question": candidate["falsification_question"],
                "target_hypothesis_ids": [shadow_id],
                "exclusive_pairs": [],
                "outcomes": [
                    {
                        "outcome_id": "falsified",
                        "supports": [],
                        "opposes": [shadow_id],
                    },
                    {
                        "outcome_id": "inconclusive",
                        "supports": [],
                        "opposes": [],
                    },
                ],
            }
        )
        existing_keys.add(mechanism_key)
        slots -= 1
    return augmented_hypotheses, augmented_requests, adaptation


def _load_initial_investment_brief(
    run_root: Path,
    state: Mapping[str, Any],
) -> dict:
    path = run_root / "active_diagnosis" / "initial_investment_brief.json"
    if path.is_symlink() or not path.is_file():
        raise ValidationError("initial investment brief must be a regular file")
    brief = load_json_object(path)
    if _canonical_digest(brief) != state.get(
        "initial_investment_brief_sha256"
    ):
        raise ValidationError("initial investment brief digest drifted")
    if (
        brief.get("schema_version")
        != "cuda-optimizer/initial-investment-brief-v1"
        or brief.get("next_checkpoint") != "propose_hypotheses"
    ):
        raise ValidationError("initial investment brief contract is invalid")
    return brief


def _run_authorization_binding_facts(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    *,
    active_scope_identity_digest: str | None = None,
    allow_active_scope_identity_drift: bool = False,
) -> dict:
    _require_run_grant_investment_control(state)
    if "analysis_contract" not in control:
        raise ValidationError("run authorization requires active diagnosis")
    if state.get("control_digest") != _canonical_digest(control):
        raise ValidationError("control manifest drifted before run authorization")
    _load_frozen_control(run_root, state)
    _load_initial_investment_brief(run_root, state)
    expected_project_identity = state.get("baseline_identity_digest")
    expected_environment_identity = state.get(
        "baseline_environment_identity_digest"
    )
    active_scope = state.get("change_scope")
    if active_scope_identity_digest is not None:
        active_identity = _sha256(
            active_scope_identity_digest,
            "active candidate scope identity",
        )
        if active_scope == "project":
            expected_project_identity = active_identity
        elif active_scope == "isolated_environment":
            expected_environment_identity = active_identity
        else:
            raise ValidationError(
                "active candidate authorization lacks a valid change scope"
            )
    elif allow_active_scope_identity_drift:
        raise ValidationError(
            "active scope drift allowance requires a candidate identity"
        )
    project_identity_drifted = (
        _identity(control, "project")["digest"] != expected_project_identity
    )
    if project_identity_drifted and not (
        allow_active_scope_identity_drift and active_scope == "project"
    ):
        raise ValidationError("run authorization project identity drifted")
    workload = _normalize_frozen_workload(control)
    if workload.source_hash != state.get("workload_source_hash"):
        raise ValidationError("run authorization workload identity drifted")
    environment_identity = state.get("baseline_environment_identity_digest")
    if (
        expected_environment_identity is not None
        and _identity(control, "isolated_environment")["digest"]
        != expected_environment_identity
        and not (
            allow_active_scope_identity_drift
            and active_scope == "isolated_environment"
        )
    ):
        raise ValidationError("run authorization environment identity drifted")
    contract = _load_frozen_analysis_contract(run_root, state)
    epoch_path = run_root / "active_diagnosis" / "epoch.json"
    if epoch_path.is_symlink() or not epoch_path.is_file():
        raise ValidationError("run authorization analysis epoch must be a regular file")
    epoch = load_json_object(epoch_path)
    epoch_module = _load_analysis_epoch_module()
    expected_identities = {
        "workload_contract_sha256": _sha256_path(Path(control["workload_manifest"])),
        "environment_sha256": environment_identity,
        "source_sha256": state["baseline_identity_digest"],
        "analysis_policy_sha256": contract["analysis_policy_sha256"],
    }
    try:
        epoch = epoch_module.validate_epoch(
            epoch,
            expected_identities=expected_identities,
        )
    except ValueError as error:
        raise ValidationError(
            f"run authorization analysis epoch is invalid: {error}"
        ) from error
    return {
        "control_digest": state["control_digest"],
        "workload_source_hash": state["workload_source_hash"],
        "baseline_identity_digest": state["baseline_identity_digest"],
        "baseline_environment_identity_digest": state[
            "baseline_environment_identity_digest"
        ],
        "analysis_epoch_sha256": epoch_module.epoch_digest(epoch),
    }


def _authorization_grant_artifacts(run_root: Path) -> dict[str, dict]:
    grant_root = run_root / "active_diagnosis" / "authorization_grants"
    if not grant_root.exists():
        return {}
    if grant_root.is_symlink() or not grant_root.is_dir():
        raise ValidationError("authorization grant directory is invalid")
    grants = {}
    ids = set()
    for path in sorted(grant_root.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValidationError("authorization grant must be a regular file")
        grant = _validate_run_authorization_record(load_json_object(path))
        if path.stem != grant["grant_id"]:
            raise ValidationError("authorization grant filename does not match grant id")
        if grant["grant_id"] in ids:
            raise ValidationError("authorization grant id is duplicated")
        ids.add(grant["grant_id"])
        digest = _canonical_digest(grant)
        if digest in grants:
            raise ValidationError("authorization grant digest is duplicated")
        grants[digest] = grant
    return grants


def _authorization_chain_digests(
    grants: Mapping[str, Mapping[str, Any]],
    current_digest: str,
) -> list[str]:
    chain = []
    seen = set()
    digest: str | None = current_digest
    while digest is not None:
        if digest in seen:
            raise ValidationError("authorization grant chain contains a cycle")
        seen.add(digest)
        grant = grants.get(digest)
        if grant is None:
            raise ValidationError("state-bound authorization grant artifact is missing")
        chain.append(digest)
        digest = grant["previous_grant_sha256"]
    return chain


def _run_authorization_payload(
    grant: Mapping[str, Any],
    grant_digest: str,
) -> dict:
    return {
        "grant_id": grant["grant_id"],
        "grant_sha256": grant_digest,
        "previous_grant_sha256": grant["previous_grant_sha256"],
        "max_controlled_seconds": grant["max_controlled_seconds"],
    }


def _load_bound_authorization_grant(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any] | None = None,
    *,
    active_scope_identity_digest: str | None = None,
    allow_active_scope_identity_drift: bool = False,
) -> dict:
    digest = _sha256(
        state.get("authorization_grant_sha256"),
        "state authorization_grant_sha256",
    )
    normalized = (
        _load_frozen_control(run_root, state)
        if control is None
        else validate_control_manifest(control)
    )
    facts = _run_authorization_binding_facts(
        normalized,
        run_root,
        state,
        active_scope_identity_digest=active_scope_identity_digest,
        allow_active_scope_identity_drift=allow_active_scope_identity_drift,
    )
    grants = _authorization_grant_artifacts(run_root)
    chain = _authorization_chain_digests(grants, digest)
    for grant_digest in chain:
        grant = grants[grant_digest]
        if any(grant.get(field) != value for field, value in facts.items()):
            raise ValidationError("authorization grant identity drifted")
    events = _verify_active_diagnosis_ledger(run_root)
    _verify_committed_active_ledger(state, events)
    committed_events = events[: int(state["active_diagnosis_ledger_sequence"])]
    committed_authorizations = [
        event["payload_sha256"]
        for event in committed_events
        if event["event_type"] == "run-authorization"
    ]
    expected_authorizations = [
        _canonical_digest(
            _run_authorization_payload(grants[grant_digest], grant_digest)
        )
        for grant_digest in reversed(chain)
    ]
    if committed_authorizations != expected_authorizations:
        raise ValidationError(
            "state-bound authorization grant ledger chain drifted"
        )
    return copy.deepcopy(grants[digest])


def _run_authorization_event(
    state: Mapping[str, Any],
    grant: Mapping[str, Any],
    grant_digest: str,
) -> tuple[Path, dict]:
    sequence = int(state["active_diagnosis_ledger_sequence"]) + 1
    payload = _run_authorization_payload(grant, grant_digest)
    event = {
        "schema_version": "cuda-optimizer/active-diagnosis-event-v1",
        "sequence": sequence,
        "event_type": "run-authorization",
        "previous_event_sha256": state[
            "active_diagnosis_ledger_head_sha256"
        ],
        "payload_sha256": _canonical_digest(payload),
        "created_at_epoch": grant["sealed_at_epoch"],
    }
    relative = (
        Path("active_diagnosis")
        / "ledger"
        / f"{sequence:06d}-run-authorization.json"
    )
    return relative, event


def authorize_run(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    grant: Mapping[str, Any],
) -> dict:
    """Seal or replay one run-level investment authorization."""
    normalized = validate_control_manifest(control)
    requested = _validate_run_authorization_input(grant)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        state = read_run_state(run_root)
        _require_run_grant_investment_control(state)
        frozen_control = _load_frozen_control(run_root, state)
        (
            state,
            _recovered_failure_decision,
            unbound_candidate_failure,
        ) = _candidate_recovery_preflight(run_root, state, frozen_control)
        if unbound_candidate_failure:
            raise ValidationError(
                "resume the unbound candidate failure before authorization"
            )
        if state.get("next_action") == "manual_recovery":
            return state
        state, diagnosis_recovery = _recover_diagnosis_publish(
            run_root,
            state,
        )
        if diagnosis_recovery != "none":
            return state
        for review_kind in ("direction", "final"):
            state, review_recovery, _aggregate = (
                _recover_reviewer_checkpoint(
                    run_root,
                    state,
                    review_kind,
                )
            )
            if review_recovery == "waiting":
                raise ValidationError(
                    "pending direction review rejects a new grant"
                )
            if review_recovery != "none":
                return state
        authorization_identity_kwargs = {}
        candidate_binding_path = run_root / "candidate_binding.json"
        if (
            state.get("change_scope") in {"project", "isolated_environment"}
            and candidate_binding_path.is_file()
            and not candidate_binding_path.is_symlink()
        ):
            change = _load_registered_change_set(run_root, state, normalized)
            candidate_binding = _validate_candidate_binding(
                load_json_object(candidate_binding_path),
                candidate=change["candidate"],
                change_set_sha256=_canonical_digest(change),
            )
            candidate_identity = candidate_binding["after_identity_digest"]
            authorization_identity_kwargs = {
                "active_scope_identity_digest": candidate_identity,
                "allow_active_scope_identity_drift": True,
            }
        facts = _run_authorization_binding_facts(
            normalized,
            run_root,
            state,
            **authorization_identity_kwargs,
        )
        grants = _authorization_grant_artifacts(run_root)
        grant_path = (
            run_root
            / "active_diagnosis"
            / "authorization_grants"
            / f"{requested['grant_id']}.json"
        )
        existing = next(
            (
                (digest, item)
                for digest, item in grants.items()
                if item["grant_id"] == requested["grant_id"]
            ),
            None,
        )
        current_digest = state.get("authorization_grant_sha256")
        if current_digest is not None:
            _sha256(current_digest, "state authorization_grant_sha256")
            current = _load_bound_authorization_grant(
                run_root,
                state,
                normalized,
                **authorization_identity_kwargs,
            )
            chain = _authorization_chain_digests(grants, current_digest)
        else:
            current = None
            chain = []

        if existing is not None:
            existing_digest, sealed = existing
            existing_input = {
                field: sealed[field]
                for field in _RUN_AUTHORIZATION_INPUT_FIELDS
            }
            if existing_input != requested:
                raise ValidationError(
                    "grant id is already sealed with different content"
                )
            if existing_digest in chain:
                return state

        committed_spend = state.get("controlled_spend_seconds", 0.0)
        if (
            type(committed_spend) not in {int, float}
            or not math.isfinite(float(committed_spend))
            or float(committed_spend) < 0
        ):
            raise ValidationError("committed controlled spend is invalid")
        if requested["max_controlled_seconds"] < float(committed_spend):
            raise ValidationError(
                "run authorization cannot be below committed controlled spend"
            )

        if existing is not None:
            if any(sealed.get(field) != value for field, value in facts.items()):
                raise ValidationError("authorization grant identity drifted")
            if sealed["previous_grant_sha256"] != current_digest:
                raise ValidationError(
                    "authorization grant previous digest drifted"
                )
        else:
            sealed_at = time.time()
            if current is not None:
                sealed_at = max(sealed_at, float(current["sealed_at_epoch"]))
            sealed = {
                **requested,
                **facts,
                "previous_grant_sha256": current_digest,
                "sealed_at_epoch": sealed_at,
            }
            _atomic_json(grant_path, sealed)
            sealed = _validate_run_authorization_record(
                load_json_object(grant_path)
            )
            existing_digest = _canonical_digest(sealed)
            grants[existing_digest] = sealed

        ledger_path, ledger_event = _run_authorization_event(
            state,
            sealed,
            existing_digest,
        )
        events = _verify_active_diagnosis_ledger(run_root)
        _verify_committed_active_ledger(state, events)
        expected_prior_count = int(state["active_diagnosis_ledger_sequence"])
        if len(events) == expected_prior_count:
            _atomic_json(run_root / ledger_path, ledger_event)
        elif (
            len(events) != expected_prior_count + 1
            or events[-1] != ledger_event
        ):
            raise ValidationError(
                "authorization ledger tail conflicts with sealed grant"
            )
        events = _verify_active_diagnosis_ledger(run_root)
        if (
            len(events) != expected_prior_count + 1
            or events[-1] != ledger_event
        ):
            raise ValidationError("authorization ledger event was not committed")

        updated = copy.deepcopy(state)
        updated.update(
            {
                "authorization_grant_sha256": existing_digest,
                "active_diagnosis_ledger_sequence": ledger_event["sequence"],
                "active_diagnosis_ledger_head_sha256": _canonical_digest(
                    ledger_event
                ),
                "updated_at_epoch": sealed["sealed_at_epoch"],
            }
        )
        _load_bound_authorization_grant(
            run_root,
            updated,
            normalized,
            **authorization_identity_kwargs,
        )
        committed = _write_state(run_root, updated)
        _load_bound_authorization_grant(
            run_root,
            committed,
            normalized,
            **authorization_identity_kwargs,
        )
        return committed


def register_active_diagnosis_proposal(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    hypothesis_set: Mapping[str, Any],
    request_set: Mapping[str, Any],
    *,
    knowledge_inputs: Mapping[str, Any] | None = None,
) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        _require_run_grant_investment_control(read_run_state(run_root))
        return _register_active_diagnosis_proposal_unlocked(
            control,
            run_root,
            hypothesis_set,
            request_set,
            knowledge_inputs=knowledge_inputs,
        )


def _validate_hypothesis_evolution(
    prior_result: Mapping[str, Any] | None,
    current_result: Mapping[str, Any],
    closed_mechanism_keys: Sequence[str],
    *,
    closed_scope_records: Sequence[Mapping[str, Any]] = (),
    evidence_catalog: Mapping[str, Any] | None = None,
    execution_map: Mapping[str, Any] | None = None,
) -> dict:
    """Keep live identities stable while allowing evidence-closed slots to rotate."""
    module = _load_hypothesis_space_module()
    catalog = {} if evidence_catalog is None else evidence_catalog
    scoped_evidence = {}
    if execution_map is not None:
        scoped_evidence = {
            item["node_id"]: set(item.get("evidence_ids", []))
            for item in execution_map.get("nodes", [])
        }
    closed = {module.canonical_mechanism_key(item) for item in closed_mechanism_keys}
    records = [copy.deepcopy(dict(item)) for item in closed_scope_records]
    record_keys = {
        (
            item["hypothesis_id"],
            item["mechanism_key"],
            item["kind"],
            item["claim_layer"],
            tuple(item["scope_node_ids"]),
        )
        for item in records
    }

    def remember_closed(item: Mapping[str, Any]) -> None:
        mechanism_key = module.canonical_mechanism_key(item["mechanism"])
        record = {
            "hypothesis_id": item["hypothesis_id"],
            "mechanism_key": mechanism_key,
            "kind": item["kind"],
            "claim_layer": item["claim_layer"],
            "scope_node_ids": sorted(item["scope_node_ids"]),
            "known_evidence_ids": sorted(catalog),
        }
        key = (
            record["hypothesis_id"],
            record["mechanism_key"],
            record["kind"],
            record["claim_layer"],
            tuple(record["scope_node_ids"]),
        )
        if key not in record_keys:
            records.append(record)
            record_keys.add(key)

    current_set = current_result["hypothesis_set"]
    current_by_id = {
        item["hypothesis_id"]: item for item in current_set["hypotheses"]
    }
    if prior_result is not None:
        prior_set = prior_result["hypothesis_set"]
        prior_by_id = {
            item["hypothesis_id"]: item for item in prior_set["hypotheses"]
        }
        identity_fields = (
            "kind",
            "scope_node_ids",
            "statement",
            "mechanism",
            "claim_layer",
        )
        current_mechanisms = {
            module.canonical_mechanism_key(item["mechanism"]): item["hypothesis_id"]
            for item in current_set["hypotheses"]
        }
        retained_ids = set(prior_by_id) & set(current_by_id)
        for hypothesis_id, prior in prior_by_id.items():
            mechanism_key = module.canonical_mechanism_key(prior["mechanism"])
            if prior["disposition"] != "active":
                closed.add(mechanism_key)
                remember_closed(prior)
                current = current_by_id.get(hypothesis_id)
                if current is not None and (
                    current["disposition"] == "active"
                    or any(current[field] != prior[field] for field in identity_fields)
                ):
                    raise ValidationError(
                        "closed hypothesis identity cannot be reused or reactivated"
                    )
                continue
            current = current_by_id.get(hypothesis_id)
            if current is None:
                if mechanism_key in current_mechanisms:
                    raise ValidationError(
                        "live hypothesis identity cannot be renamed inside an analysis epoch"
                    )
                closed.add(mechanism_key)
                remember_closed(prior)
                continue
            if any(current[field] != prior[field] for field in identity_fields):
                raise ValidationError(
                    "live hypothesis identity cannot change inside an analysis epoch"
                )
            if current["disposition"] != "active":
                remember_closed(prior)
        prior_relationships = {
            (item["relation"], item["left"], item["right"])
            for item in prior_set["relationships"]
            if item["left"] in retained_ids and item["right"] in retained_ids
        }
        current_relationships = {
            (item["relation"], item["left"], item["right"])
            for item in current_set["relationships"]
            if item["left"] in retained_ids and item["right"] in retained_ids
        }
        if current_relationships != prior_relationships:
            raise ValidationError(
                "relationships between retained hypotheses cannot change inside an analysis epoch"
            )
    for item in current_set["hypotheses"]:
        if item["disposition"] != "active":
            closed.add(module.canonical_mechanism_key(item["mechanism"]))
            remember_closed(item)
    prior_ids = set() if prior_result is None else set(prior_by_id)
    historical_closed_ids = {record["hypothesis_id"] for record in records}
    for item in current_set["hypotheses"]:
        if item["disposition"] != "active" or item["hypothesis_id"] in prior_ids:
            continue
        if item["hypothesis_id"] in historical_closed_ids:
            raise ValidationError(
                "historical hypothesis identity cannot be reused after closure"
            )
        mechanism_key = module.canonical_mechanism_key(item["mechanism"])
        if mechanism_key in closed:
            raise ValidationError(
                "closed mechanism cannot be reactivated with a new hypothesis id"
            )
        item_scope = set(item["scope_node_ids"])
        matching_records = [
            record
            for record in records
            if item_scope & set(record["scope_node_ids"])
        ]
        if not matching_records:
            continue
        for record in matching_records:
            fresh_support = set(item["support_evidence_ids"]) - set(
                record["known_evidence_ids"]
            )
            overlap = item_scope & set(record["scope_node_ids"])
            has_material_support = False
            for evidence_id in fresh_support:
                evidence = catalog.get(evidence_id, {})
                if item["hypothesis_id"] in evidence.get(
                    "supports_hypothesis_ids", []
                ):
                    has_material_support = True
                    break
                if evidence.get("supports_hypothesis_ids", []) or evidence.get(
                    "opposes_hypothesis_ids", []
                ):
                    continue
                if any(
                    evidence_id in scoped_evidence.get(node_id, set())
                    for node_id in overlap
                ):
                    has_material_support = True
                    break
            if not has_material_support:
                raise ValidationError(
                    "new mechanism on a closed scope requires materially new evidence "
                    "that is outcome-bound to the replacement hypothesis or "
                    "Controller-scoped to the overlapping execution-map nodes"
                )
    records.sort(
        key=lambda item: (
            item["kind"],
            item["claim_layer"],
            item["scope_node_ids"],
            item["mechanism_key"],
            item["hypothesis_id"],
        )
    )
    return {
        "closed_mechanism_keys": sorted(closed),
        "closed_scope_records": records,
    }


def _load_bound_diagnostic_artifacts(
    run_root: Path,
    state: Mapping[str, Any],
    *,
    expected_decision: str,
) -> dict:
    """Recheck the published decision and brief before a downstream stage."""
    active_root = run_root / "active_diagnosis"
    decision_path = active_root / "decision.json"
    brief_path = active_root / "investment_brief.json"
    for path, label in (
        (decision_path, "diagnostic decision"),
        (brief_path, "investment brief"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"{label} must be a regular file")
    decision = load_json_object(decision_path)
    brief = load_json_object(brief_path)
    if _canonical_digest(decision) != state.get("diagnostic_decision_sha256"):
        raise ValidationError("diagnostic decision digest drifted before execution")
    if _canonical_digest(brief) != state.get("investment_brief_sha256"):
        raise ValidationError("investment brief digest drifted before execution")
    if decision.get("decision") != expected_decision:
        raise ValidationError(
            f"diagnostic decision does not authorize {expected_decision} execution"
        )
    expected_brief = decision.get("investment_brief")
    if brief != expected_brief:
        raise ValidationError("investment brief does not match diagnostic decision")
    return decision


def _controlled_spend_seconds(state: Mapping[str, Any]) -> float:
    value = state.get("controlled_spend_seconds")
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValidationError("run state controlled_spend_seconds is invalid")
    return float(value)


def _review_call_root(run_root: Path, review_kind: str) -> Path:
    if review_kind == "direction":
        return run_root / "active_diagnosis" / "direction_review"
    if review_kind == "final":
        return run_root / "final_review"
    raise ValidationError("review kind must be direction or final")


def _review_generation(
    review_root: Path,
    generation_kind: str,
    digest: str,
    value: Mapping[str, Any] | None = None,
) -> dict:
    directory = {
        "intent": "intents",
        "aggregate": "aggregates",
        "complete": "completions",
    }.get(generation_kind)
    if directory is None:
        raise ValidationError("review generation kind is invalid")
    digest = _sha256(digest, f"review {generation_kind} generation")
    path = review_root / "generations" / directory / f"{digest}.json"
    if value is None:
        loaded = load_json_object(path)
        if _canonical_digest(loaded) != digest:
            raise ValidationError(f"review {generation_kind} generation drifted")
        return loaded
    detached = copy.deepcopy(dict(value))
    if _canonical_digest(detached) != digest:
        raise ValidationError(f"review {generation_kind} digest is invalid")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or load_json_object(path) != detached:
            raise ValidationError(f"review {generation_kind} generation conflicts")
    else:
        _atomic_json(path, detached)
    return detached


def _reviewer_configs(control: Mapping[str, Any]) -> list[dict]:
    if type(control.get("reviewers")) is list:
        source = control["reviewers"]
    elif isinstance(control.get("reviewer"), Mapping):
        item = control["reviewer"]
        source = [
            {
                "provider": "local-reviewer",
                "underlying_model": "unknown",
                "argv": item["argv"],
                "timeout_seconds": item["timeout_seconds"],
            }
        ]
    else:
        return []
    return [
        {
            "provider": item["provider"],
            "underlying_model": item.get("underlying_model", "unknown"),
            "argv": copy.deepcopy(item["argv"]),
            "timeout_seconds": float(item["timeout_seconds"]),
        }
        for item in source
    ]


def _review_maximum_total_wait_seconds(
    configs: Sequence[Mapping[str, Any]],
) -> float:
    seen = set()
    total = 0.0
    for item in configs:
        key = (
            str(item["provider"]).strip().lower(),
            str(item.get("underlying_model", "unknown")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        timeout = float(item["timeout_seconds"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValidationError("reviewer timeout is invalid")
        total += timeout
    return max(1.0, min(180.0, total))


def _review_authorization_grant(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any] | None,
    review_kind: str,
) -> dict:
    kwargs = {}
    if review_kind == "final":
        kwargs = {
            "active_scope_identity_digest": _sha256(
                state.get("candidate_identity_digest"),
                "final review candidate identity",
            ),
            "allow_active_scope_identity_drift": True,
        }
    return _load_bound_authorization_grant(
        run_root,
        state,
        control,
        **kwargs,
    )


def _validate_review_call_intent(
    review_kind: str,
    value: Mapping[str, Any],
) -> dict:
    intent = _object(value, f"{review_kind} review intent")
    fields = {
        "schema_version",
        "review_kind",
        "review_id",
        "base_state_sha256",
        "authorization_grant_sha256",
        "request",
        "request_sha256",
        "maximum_total_wait_seconds",
        "created_at_epoch",
    }
    _closed(intent, fields, f"{review_kind} review intent")
    _required(intent, fields, f"{review_kind} review intent")
    if (
        intent["schema_version"] != _REVIEW_CALL_INTENT_SCHEMA
        or intent["review_kind"] != review_kind
    ):
        raise ValidationError(f"{review_kind} review intent identity drifted")
    reviewer = _load_reviewer_module()
    try:
        request = reviewer.validate_review_request(intent["request"])
    except ValueError as error:
        raise ValidationError(f"{review_kind} review request is invalid") from error
    base_sha = _sha256(intent["base_state_sha256"], "review base state")
    grant_sha = _sha256(
        intent["authorization_grant_sha256"],
        "review authorization grant",
    )
    expected_review_id = _canonical_digest(
        {
            "review_kind": review_kind,
            "base_state_sha256": base_sha,
            "request_digest": request["request_digest"],
        }
    )
    maximum = intent["maximum_total_wait_seconds"]
    created = intent["created_at_epoch"]
    if (
        request["review_summary"]["review_kind"] != review_kind
        or intent["request_sha256"] != _canonical_digest(request)
        or intent["review_id"] != expected_review_id
        or type(maximum) not in {int, float}
        or not math.isfinite(float(maximum))
        or not 1 <= float(maximum) <= 180
        or grant_sha != intent["authorization_grant_sha256"]
        or type(created) not in {int, float}
        or not math.isfinite(float(created))
    ):
        raise ValidationError(f"{review_kind} review intent binding is invalid")
    normalized = copy.deepcopy(dict(intent))
    normalized["request"] = request
    normalized["maximum_total_wait_seconds"] = float(maximum)
    return normalized


def _validate_review_aggregate(
    intent: Mapping[str, Any],
    value: Mapping[str, Any],
    configs: Sequence[Mapping[str, Any]],
) -> dict:
    kind = intent["review_kind"]
    aggregate = _object(value, f"{kind} review aggregate")
    fields = {
        "schema_version",
        "status",
        "request_digest",
        "trigger",
        "target_completed_provider_count",
        "providers_requested",
        "providers_completed",
        "failed_providers",
        "heterogeneous_models",
        "total_timeout_seconds",
        "total_wait_seconds",
        "reviews",
    }
    _closed(aggregate, fields, f"{kind} review aggregate")
    _required(aggregate, fields, f"{kind} review aggregate")
    if (
        aggregate["schema_version"]
        != "cuda-workload-optimizer/review-aggregate-v1"
        or aggregate["status"] not in {"completed", "unavailable", "skipped"}
        or aggregate["request_digest"]
        != intent["request"]["request_digest"]
    ):
        raise ValidationError(f"{kind} review aggregate identity drifted")
    allowed_triggers = (
        {"ordinary", "major", "plateau"}
        if kind == "direction"
        else {"final"}
    )
    target = aggregate["target_completed_provider_count"]
    allowed_providers = {str(item["provider"]) for item in configs}
    for field in (
        "providers_requested",
        "providers_completed",
        "failed_providers",
        "heterogeneous_models",
        "reviews",
    ):
        if type(aggregate[field]) is not list:
            raise ValidationError(f"{kind} review aggregate array is invalid")
    requested = aggregate["providers_requested"]
    completed = aggregate["providers_completed"]
    failed = aggregate["failed_providers"]
    if (
        aggregate["trigger"] not in allowed_triggers
        or type(target) is not int
        or not 0 <= target <= 3
        or len(requested) != len(set(requested))
        or any(
            type(provider) is not str
            or provider not in allowed_providers
            for provider in requested
        )
        or any(provider not in requested for provider in completed + failed)
        or set(completed) & set(failed)
    ):
        raise ValidationError(f"{kind} review provider coverage is invalid")
    timeout = aggregate["total_timeout_seconds"]
    waited = aggregate["total_wait_seconds"]
    if (
        type(timeout) not in {int, float}
        or type(waited) not in {int, float}
        or not math.isfinite(float(timeout))
        or not math.isfinite(float(waited))
        or not 0 <= float(waited) <= float(timeout)
        or float(timeout) > float(intent["maximum_total_wait_seconds"])
    ):
        raise ValidationError(f"{kind} review aggregate wait is invalid")
    if any(
        _object(item, f"{kind} review result").get("provider") not in requested
        for item in aggregate["reviews"]
    ):
        raise ValidationError(f"{kind} review result provider is invalid")
    return copy.deepcopy(dict(aggregate))


def _validate_review_call_complete(
    review_kind: str,
    intent: Mapping[str, Any],
    value: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict:
    complete = _object(value, f"{review_kind} review complete")
    fields = {
        "schema_version",
        "review_kind",
        "review_id",
        "intent_sha256",
        "request_sha256",
        "aggregate_sha256",
        "total_wait_seconds",
        "completed_at_epoch",
    }
    _closed(complete, fields, f"{review_kind} review complete")
    _required(complete, fields, f"{review_kind} review complete")
    if (
        complete["schema_version"] != _REVIEW_CALL_COMPLETE_SCHEMA
        or complete["review_kind"] != review_kind
        or complete["review_id"] != intent["review_id"]
        or complete["intent_sha256"] != _canonical_digest(intent)
        or complete["request_sha256"] != intent["request_sha256"]
        or complete["aggregate_sha256"] != _canonical_digest(aggregate)
    ):
        raise ValidationError(f"{review_kind} review complete binding drifted")
    waited = complete["total_wait_seconds"]
    if (
        type(waited) not in {int, float}
        or not math.isfinite(float(waited))
        or float(waited) != float(aggregate["total_wait_seconds"])
        or not 0
        <= float(waited)
        <= float(intent["maximum_total_wait_seconds"])
    ):
        raise ValidationError(f"{review_kind} review complete wait is invalid")
    completed = complete["completed_at_epoch"]
    if (
        type(completed) not in {int, float}
        or not math.isfinite(float(completed))
        or float(completed) < float(intent["created_at_epoch"])
    ):
        raise ValidationError(f"{review_kind} review complete time is invalid")
    return copy.deepcopy(dict(complete))


def _persist_task6_manual_recovery(
    run_root: Path,
    state: Mapping[str, Any],
    *,
    reason: str,
) -> dict:
    blocked = copy.deepcopy(dict(state))
    blocked.update(
        {
            "status": "manual_recovery_required",
            "stage": "recovery",
            "next_action": "manual_recovery",
            "manual_recovery_reason": _identifier(
                reason, "task 6 manual recovery reason"
            ),
            "updated_at_epoch": time.time(),
        }
    )
    return _write_state(run_root, blocked)


def _load_review_call_evidence(
    run_root: Path,
    review_kind: str,
    state: Mapping[str, Any],
) -> tuple[dict | None, dict | None, dict | None]:
    root = _review_call_root(run_root, review_kind)
    intent_path, complete_path = root / "intent.json", root / "complete.json"
    if intent_path.is_symlink() or complete_path.is_symlink():
        raise ValidationError(f"{review_kind} review marker must be regular")
    intent = load_json_object(intent_path) if intent_path.is_file() else None
    complete = load_json_object(complete_path) if complete_path.is_file() else None
    binding = state.get("review_call_intent_sha256")
    if binding is not None:
        binding = _sha256(binding, "state review intent")
    if intent is None and complete is not None:
        intent = _review_generation(
            root,
            "intent",
            _sha256(complete.get("intent_sha256"), "review complete intent"),
        )
    if intent is None and binding is not None:
        generation_path = (
            root / "generations" / "intents" / f"{binding}.json"
        )
        if generation_path.is_file() and not generation_path.is_symlink():
            intent = _review_generation(
                root,
                "intent",
                _sha256(binding, "state review intent"),
            )
    if intent is None:
        return None, None, None
    intent = _validate_review_call_intent(review_kind, intent)
    intent_digest = _canonical_digest(intent)
    if _review_generation(root, "intent", intent_digest) != intent:
        raise ValidationError(f"{review_kind} review intent generation drifted")
    if complete is None:
        completions = state.get("review_call_completions", {})
        digest = (
            completions.get(intent["review_id"])
            if isinstance(completions, Mapping)
            else None
        )
        if digest is not None:
            complete = _review_generation(
                root, "complete", _sha256(digest, "state review completion")
            )
    if complete is None and binding == intent_digest:
        matches = []
        generation_root = root / "generations" / "completions"
        if generation_root.exists() or generation_root.is_symlink():
            if generation_root.is_symlink() or not generation_root.is_dir():
                raise ValidationError(
                    f"{review_kind} review completion generations are invalid"
                )
            for path in sorted(generation_root.glob("*.json")):
                if path.is_symlink() or not path.is_file():
                    raise ValidationError(
                        f"{review_kind} review completion generation is invalid"
                    )
                candidate = load_json_object(path)
                if candidate.get("intent_sha256") != intent_digest:
                    continue
                if path.stem != _canonical_digest(candidate):
                    raise ValidationError(
                        f"{review_kind} review completion generation drifted"
                    )
                matches.append(candidate)
        if len(matches) > 1:
            raise ValidationError(
                f"{review_kind} review has conflicting completions"
            )
        if matches:
            complete = matches[0]
    if complete is None:
        return intent, None, None
    aggregate = _review_generation(
        root,
        "aggregate",
        _sha256(complete.get("aggregate_sha256"), "review aggregate"),
    )
    configs = _reviewer_configs(_load_frozen_control(run_root, state))
    aggregate = _validate_review_aggregate(intent, aggregate, configs)
    complete = _validate_review_call_complete(
        review_kind, intent, complete, aggregate
    )
    if _review_generation(root, "complete", _canonical_digest(complete)) != complete:
        raise ValidationError(
            f"{review_kind} review completion generation drifted"
        )
    return intent, complete, aggregate


def _cleanup_review_call_markers(
    run_root: Path,
    review_kind: str,
) -> None:
    review_root = _review_call_root(run_root, review_kind)
    _remove_path(review_root / "intent.json")
    _remove_path(review_root / "complete.json")


def _consume_review_call_complete(
    run_root: Path,
    state: Mapping[str, Any],
    review_kind: str,
    intent: Mapping[str, Any],
    complete: Mapping[str, Any],
) -> dict:
    intent_digest = _canonical_digest(intent)
    complete_digest = _canonical_digest(complete)
    completions = copy.deepcopy(state.get("review_call_completions", {}))
    if type(completions) is not dict:
        raise ValidationError("review completion state is invalid")
    prior = completions.get(intent["review_id"])
    if prior is not None and prior != complete_digest:
        raise ValidationError(f"{review_kind} review completion conflicts")
    if prior == complete_digest:
        return copy.deepcopy(dict(state))
    if state.get("review_call_intent_sha256") != intent_digest:
        raise ValidationError(f"{review_kind} review intent is not state-bound")
    base = load_json_object(
        run_root
        / "state_generations"
        / f"{intent['base_state_sha256']}.json"
    )
    if (
        _canonical_digest(base) != intent["base_state_sha256"]
        or base.get("authorization_grant_sha256")
        != intent["authorization_grant_sha256"]
    ):
        raise ValidationError(f"{review_kind} review base state drifted")
    grant = _review_authorization_grant(
        run_root,
        state,
        None,
        review_kind,
    )
    spend = _controlled_spend_seconds(state) + float(
        complete["total_wait_seconds"]
    )
    if spend > float(grant["max_controlled_seconds"]):
        raise ValidationError(f"{review_kind} review exceeds authorization")
    completions[intent["review_id"]] = complete_digest
    updated = copy.deepcopy(dict(state))
    updated["review_call_completions"] = completions
    updated["controlled_spend_seconds"] = spend
    updated.pop("review_call_intent_sha256", None)
    updated["updated_at_epoch"] = max(
        float(updated.get("updated_at_epoch", 0.0)),
        float(complete["completed_at_epoch"]),
    )
    return _write_state(run_root, updated)


def _recover_reviewer_checkpoint(
    run_root: Path,
    state: Mapping[str, Any],
    review_kind: str,
) -> tuple[dict, str, dict | None]:
    if (
        state.get("status") == "completed"
        and state.get("terminal_reason") == "candidate_abandoned"
    ):
        return copy.deepcopy(dict(state)), "abandoned", None
    root = _review_call_root(run_root, review_kind)
    markers_exist = any(
        path.exists() or path.is_symlink()
        for path in (root / "intent.json", root / "complete.json")
    )
    binding = state.get("review_call_intent_sha256")
    if binding is not None and (
        type(binding) is not str
        or re.fullmatch(r"[a-f0-9]{64}", binding) is None
    ):
        blocked = _persist_task6_manual_recovery(
            run_root,
            state,
            reason="review_intent_binding_invalid",
        )
        return blocked, "manual", None
    bound_generation = (
        root / "generations" / "intents" / f"{binding}.json"
        if binding is not None
        else None
    )
    if not markers_exist and (
        bound_generation is None
        or not bound_generation.is_file()
        or bound_generation.is_symlink()
    ):
        if binding is not None and review_kind == "final":
            blocked = _persist_task6_manual_recovery(
                run_root,
                state,
                reason="review_intent_binding_unresolved",
            )
            return blocked, "manual", None
        return copy.deepcopy(dict(state)), "none", None
    try:
        intent, complete, aggregate = _load_review_call_evidence(
            run_root, review_kind, state
        )
        if intent is None:
            raise ValidationError(f"{review_kind} review evidence is incomplete")
        intent_digest = _canonical_digest(intent)
        completions = state.get("review_call_completions", {})
        if type(completions) is not dict:
            raise ValidationError("review completion state is invalid")
        consumed = completions.get(intent["review_id"])
        if consumed is not None:
            _sha256(consumed, "consumed review completion")
        binding = state.get("review_call_intent_sha256")
        if binding is None:
            if complete is not None and consumed == _canonical_digest(complete):
                _cleanup_review_call_markers(run_root, review_kind)
                return copy.deepcopy(dict(state)), "cleaned", aggregate
            if (
                complete is None
                and _canonical_digest(state) == intent["base_state_sha256"]
            ):
                _cleanup_review_call_markers(run_root, review_kind)
                return copy.deepcopy(dict(state)), "discarded_unbound", None
            raise ValidationError(
                f"{review_kind} review marker is not bound to state"
            )
        if (
            binding != intent_digest
            or
            state.get("authorization_grant_sha256")
            != intent["authorization_grant_sha256"]
        ):
            raise ValidationError(f"{review_kind} review state binding drifted")
        if complete is None:
            blocked = _persist_task6_manual_recovery(
                run_root, state, reason=f"{review_kind}_review_outcome_unknown"
            )
            return blocked, "manual", None
        if review_kind == "direction":
            return copy.deepcopy(dict(state)), "waiting", aggregate
        committed = _consume_review_call_complete(
            run_root, state, review_kind, intent, complete
        )
        _cleanup_review_call_markers(run_root, review_kind)
        return committed, "consumed", aggregate
    except (KeyError, OSError, ValidationError, ValueError):
        blocked = _persist_task6_manual_recovery(
            run_root, state, reason=f"{review_kind}_review_recovery_invalid"
        )
        return blocked, "manual", None


def _empty_review_aggregate(
    request_digest: str,
    trigger: str,
    status: str,
    total_timeout_seconds: float,
) -> dict:
    return {
        "schema_version": "cuda-workload-optimizer/review-aggregate-v1",
        "status": status,
        "request_digest": request_digest,
        "trigger": trigger,
        "target_completed_provider_count": 0,
        "providers_requested": [],
        "providers_completed": [],
        "failed_providers": [],
        "heterogeneous_models": [],
        "total_timeout_seconds": float(total_timeout_seconds),
        "total_wait_seconds": 0.0,
        "reviews": [],
    }


def _managed_review_call(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    review_kind: str,
    request: Mapping[str, Any],
    *,
    trigger: str,
) -> tuple[dict, dict]:
    reviewer = _load_reviewer_module()
    try:
        safe_request = reviewer.validate_review_request(request)
    except ValueError as error:
        raise ValidationError(f"{review_kind} review request is invalid") from error
    if safe_request["review_summary"]["review_kind"] != review_kind:
        raise ValidationError(f"{review_kind} review request kind is invalid")
    current, outcome, recovered_aggregate = _recover_reviewer_checkpoint(
        run_root,
        state,
        review_kind,
    )
    if outcome == "manual":
        return current, _empty_review_aggregate(
            safe_request["request_digest"], trigger, "unavailable", 0.0
        )
    if outcome == "waiting":
        existing_intent, _complete, _aggregate = _load_review_call_evidence(
            run_root,
            review_kind,
            current,
        )
        if (
            existing_intent is None
            or existing_intent["request"] != safe_request
        ):
            raise ValidationError(
                "direction review requires the same proposal and request"
            )
        return current, copy.deepcopy(recovered_aggregate)
    configs = _reviewer_configs(control)
    if not configs:
        raise ValidationError("managed review requires configured reviewers")
    if current.get("review_call_intent_sha256") is not None:
        raise ValidationError("another reviewer call is already pending")
    maximum_wait = _review_maximum_total_wait_seconds(configs)
    grant = _review_authorization_grant(
        run_root,
        current,
        control,
        review_kind,
    )
    created = time.time()
    base_digest = _canonical_digest(current)
    review_id = _canonical_digest(
        {
            "review_kind": review_kind,
            "base_state_sha256": base_digest,
            "request_digest": safe_request["request_digest"],
        }
    )
    intent = {
        "schema_version": _REVIEW_CALL_INTENT_SCHEMA,
        "review_kind": review_kind,
        "review_id": review_id,
        "base_state_sha256": base_digest,
        "authorization_grant_sha256": current[
            "authorization_grant_sha256"
        ],
        "request": safe_request,
        "request_sha256": _canonical_digest(safe_request),
        "maximum_total_wait_seconds": maximum_wait,
        "created_at_epoch": created,
    }
    intent = _validate_review_call_intent(review_kind, intent)
    intent_digest = _canonical_digest(intent)
    review_root = _review_call_root(run_root, review_kind)
    _review_generation(review_root, "intent", intent_digest, intent)
    _atomic_json(review_root / "intent.json", intent)
    bound = copy.deepcopy(dict(current))
    bound["review_call_intent_sha256"] = intent_digest
    bound["updated_at_epoch"] = created
    bound = _write_state(run_root, bound)

    remaining = (
        float(grant["max_controlled_seconds"])
        - _controlled_spend_seconds(bound)
    )
    if remaining < maximum_wait:
        aggregate = _empty_review_aggregate(
            safe_request["request_digest"], trigger, "skipped", maximum_wait
        )
    else:
        aggregate = reviewer.run_prioritized_reviewers(
            configs,
            safe_request,
            review_root / "provider",
            trigger=trigger,
            total_timeout_seconds=maximum_wait,
        )
    aggregate = _validate_review_aggregate(intent, aggregate, configs)
    aggregate_digest = _canonical_digest(aggregate)
    _review_generation(
        review_root, "aggregate", aggregate_digest, aggregate
    )
    _atomic_json(review_root / "review.json", aggregate)
    complete = {
        "schema_version": _REVIEW_CALL_COMPLETE_SCHEMA,
        "review_kind": review_kind,
        "review_id": review_id,
        "intent_sha256": intent_digest,
        "request_sha256": intent["request_sha256"],
        "aggregate_sha256": aggregate_digest,
        "total_wait_seconds": float(aggregate["total_wait_seconds"]),
        "completed_at_epoch": time.time(),
    }
    complete = _validate_review_call_complete(
        review_kind,
        intent,
        complete,
        aggregate,
    )
    complete_digest = _canonical_digest(complete)
    _review_generation(
        review_root, "complete", complete_digest, complete
    )
    _atomic_json(review_root / "complete.json", complete)
    if review_kind == "direction":
        return bound, aggregate
    committed = _consume_review_call_complete(
        run_root,
        bound,
        review_kind,
        intent,
        complete,
    )
    _cleanup_review_call_markers(run_root, review_kind)
    return committed, aggregate


def _direction_review_for_diagnosis_publish(
    run_root: Path,
    state: Mapping[str, Any],
) -> tuple[dict | None, dict | None]:
    if state.get("review_call_intent_sha256") is None:
        return None, None
    intent, complete, aggregate = _load_review_call_evidence(
        run_root,
        "direction",
        state,
    )
    if (
        intent is None
        or complete is None
        or aggregate is None
        or state.get("review_call_intent_sha256")
        != _canonical_digest(intent)
    ):
        raise ValidationError(
            "direction review is not complete for diagnosis publication"
        )
    return complete, aggregate


def _direction_review_generation(
    run_root: Path,
    base: Mapping[str, Any],
    complete_digest: str,
) -> tuple[dict, dict]:
    root = _review_call_root(run_root, "direction")
    complete = _review_generation(
        root, "complete", _sha256(complete_digest, "direction completion")
    )
    intent_digest = _sha256(
        complete.get("intent_sha256"), "direction review intent"
    )
    intent = _validate_review_call_intent(
        "direction", _review_generation(root, "intent", intent_digest)
    )
    aggregate = _validate_review_aggregate(
        intent,
        _review_generation(
            root,
            "aggregate",
            _sha256(complete.get("aggregate_sha256"), "direction aggregate"),
        ),
        _reviewer_configs(_load_frozen_control(run_root, base)),
    )
    complete = _validate_review_call_complete(
        "direction", intent, complete, aggregate
    )
    if (
        _canonical_digest(complete) != complete_digest
        or base.get("review_call_intent_sha256") != intent_digest
        or base.get("authorization_grant_sha256")
        != intent["authorization_grant_sha256"]
    ):
        raise ValidationError("diagnosis direction review binding drifted")
    return complete, aggregate


def _diagnosis_proposal_binding(
    bundle: Mapping[str, Mapping[str, Any]],
    direction_review_aggregate: Mapping[str, Any] | None,
) -> dict:
    return {
        "context_sha256": _canonical_digest(bundle["diagnosis_context"]),
        "hypothesis_set_sha256": bundle["hypothesis_generation"][
            "hypothesis_set_sha256"
        ],
        "request_set_sha256": _canonical_digest(bundle["request_set"]),
        "selection_sha256": _canonical_digest(bundle["evidence_selection"]),
        "decision_sha256": _canonical_digest(bundle["decision"]),
        "investment_brief_sha256": _canonical_digest(
            bundle["investment_brief"]
        ),
        "external_direction_review_sha256": (
            None
            if direction_review_aggregate is None
            else _canonical_digest(direction_review_aggregate)
        ),
        "knowledge_adaptation_sha256": _canonical_digest(
            bundle["knowledge_adaptation"]
        ),
    }


def _diagnosis_publish_target_state(
    base: Mapping[str, Any],
    bundle: Mapping[str, Mapping[str, Any]],
    proposal_ledger_event: Mapping[str, Any],
    committed_at_epoch: float,
    *,
    direction_review_complete: Mapping[str, Any] | None,
    direction_review_aggregate: Mapping[str, Any] | None,
) -> dict:
    context = bundle["diagnosis_context"]
    generation = bundle["hypothesis_generation"]
    selection = bundle["evidence_selection"]
    decision = bundle["decision"]
    brief = bundle["investment_brief"]
    next_action = {
        "MEASURE": "collect_evidence",
        "PURSUE": "register_change",
        "REVIEW_REQUIRED": "review_required",
        "STOP": "done",
    }[decision["decision"]]
    target = copy.deepcopy(dict(base))
    selected = brief["selected_action"]
    blocked = brief["blocked_action"]
    target.update(
        {
            "stage": "active_diagnosis",
            "next_action": next_action,
            "updated_at_epoch": float(committed_at_epoch),
            "hypothesis_set_sha256": generation["hypothesis_set_sha256"],
            "evidence_selection_sha256": _canonical_digest(selection),
            "diagnostic_decision_sha256": _canonical_digest(decision),
            "investment_brief_sha256": _canonical_digest(brief),
            "terminal_reason": decision["terminal_reason"],
            "diagnosis_context_sha256": _canonical_digest(context),
            "active_diagnosis_ledger_sequence": proposal_ledger_event["sequence"],
            "active_diagnosis_ledger_head_sha256": _canonical_digest(
                proposal_ledger_event
            ),
            "investment_summary": {
                "decision": brief["decision"],
                "cumulative_investment": copy.deepcopy(brief["cumulative_investment"]),
                "selected_action_id": None if selected is None else selected["action_id"],
                "blocked_action_id": None if blocked is None else blocked["action_id"],
                "next_feedback_point": brief["next_feedback_point"],
            },
        }
    )
    if decision["decision"] == "STOP":
        target["status"] = "completed"
    if (
        decision["decision"] == "MEASURE"
        and selection["selected_request"] is not None
    ):
        target["selected_request_signature"] = selection[
            "selected_request"
        ]["request_signature"]
    if "diagnosis_proposal" not in target["completed_stages"]:
        target["completed_stages"].append("diagnosis_proposal")
    if direction_review_complete is not None:
        if direction_review_aggregate is None:
            raise ValidationError(
                "direction review completion is missing its aggregate"
            )
        completions = copy.deepcopy(
            target.get("review_call_completions", {})
        )
        if type(completions) is not dict:
            raise ValidationError("review completion state is invalid")
        complete_digest = _canonical_digest(direction_review_complete)
        prior = completions.get(direction_review_complete["review_id"])
        if prior is not None and prior != complete_digest:
            raise ValidationError("direction review completion conflicts")
        completions[direction_review_complete["review_id"]] = complete_digest
        target["review_call_completions"] = completions
        target["controlled_spend_seconds"] = (
            _controlled_spend_seconds(base)
            + float(direction_review_complete["total_wait_seconds"])
        )
        target["external_direction_review_sha256"] = _canonical_digest(
            direction_review_aggregate
        )
        target.pop("review_call_intent_sha256", None)
    target.pop("diagnosis_publish_intent_sha256", None)
    return target


def _validate_diagnosis_publish_intent(
    run_root: Path,
    value: Mapping[str, Any],
) -> tuple[dict, dict, dict | None, dict | None]:
    intent = _object(value, "diagnosis publish intent")
    fields = {
        "schema_version", "base_state_sha256", "base_ledger_sequence",
        "base_ledger_head_sha256", "direction_review_complete_sha256",
        "diagnosis_context", "knowledge_adaptation", "hypothesis_generation",
        "hypothesis_result", "request_set", "evidence_selection", "decision",
        "investment_brief", "proposal_ledger_path", "proposal_ledger_event",
        "target_state", "created_at_epoch",
    }
    _closed(intent, fields, "diagnosis publish intent")
    _required(intent, fields, "diagnosis publish intent")
    if intent["schema_version"] != _DIAGNOSIS_PUBLISH_INTENT_SCHEMA:
        raise ValidationError("diagnosis publish intent schema is invalid")
    base_digest = _sha256(intent["base_state_sha256"], "diagnosis base state")
    base = load_json_object(run_root / "state_generations" / f"{base_digest}.json")
    if _canonical_digest(base) != base_digest:
        raise ValidationError("diagnosis publish base state drifted")
    if (
        type(intent["base_ledger_sequence"]) is not int
        or intent["base_ledger_sequence"] < 1
        or intent["base_ledger_sequence"] != base.get("active_diagnosis_ledger_sequence")
        or intent["base_ledger_head_sha256"]
        != base.get("active_diagnosis_ledger_head_sha256")
    ):
        raise ValidationError("diagnosis publish ledger base drifted")
    _sha256(intent["base_ledger_head_sha256"], "diagnosis base ledger head")
    for field in (
        "diagnosis_context", "knowledge_adaptation", "hypothesis_generation",
        "hypothesis_result", "request_set", "evidence_selection", "decision",
        "investment_brief", "proposal_ledger_event", "target_state",
    ):
        _object(intent[field], f"diagnosis publish {field}")
    bundle = {
        field: intent[field]
        for field in (
            "diagnosis_context", "knowledge_adaptation",
            "hypothesis_generation", "hypothesis_result", "request_set",
            "evidence_selection", "decision", "investment_brief",
        )
    }
    generation = intent["hypothesis_generation"]
    if (
        generation != intent["hypothesis_result"]
        or generation.get("hypothesis_set_sha256")
        != _canonical_digest(generation.get("hypothesis_set"))
    ):
        raise ValidationError("diagnosis publish hypothesis generation drifted")
    if intent["investment_brief"] != intent["decision"].get("investment_brief"):
        raise ValidationError("diagnosis publish investment brief drifted")
    event = intent["proposal_ledger_event"]
    event_fields = {
        "schema_version", "sequence", "event_type",
        "previous_event_sha256", "payload_sha256", "created_at_epoch",
    }
    _closed(event, event_fields, "diagnosis publish proposal ledger event")
    _required(event, event_fields, "diagnosis publish proposal ledger event")
    if (
        event["schema_version"] != "cuda-optimizer/active-diagnosis-event-v1"
        or event["sequence"] != intent["base_ledger_sequence"] + 1
        or event["event_type"] != "proposal"
        or event["previous_event_sha256"] != intent["base_ledger_head_sha256"]
    ):
        raise ValidationError("diagnosis publish proposal ledger drifted")
    expected_ledger_path = (
        Path("active_diagnosis")
        / "ledger"
        / f"{event['sequence']:06d}-proposal.json"
    ).as_posix()
    if intent["proposal_ledger_path"] != expected_ledger_path:
        raise ValidationError("diagnosis publish ledger path drifted")
    created = intent["created_at_epoch"]
    if (
        type(created) not in {int, float}
        or not math.isfinite(float(created))
        or float(event["created_at_epoch"]) != float(created)
    ):
        raise ValidationError("diagnosis publish time drifted")

    direction_complete = None
    direction_aggregate = None
    complete_digest = intent["direction_review_complete_sha256"]
    if complete_digest is not None:
        direction_complete, direction_aggregate = (
            _direction_review_generation(run_root, base, complete_digest)
        )
        direction_root = _review_call_root(run_root, "direction")
        direction_intent = _validate_review_call_intent(
            "direction",
            _review_generation(
                direction_root,
                "intent",
                _sha256(
                    direction_complete.get("intent_sha256"),
                    "direction review intent",
                ),
            ),
        )
        proposal_hashes = direction_intent["request"]["review_summary"][
            "artifact_hashes"
        ]
        if (
            proposal_hashes.get("request_set.json")
            != _canonical_digest(intent["request_set"])
            or proposal_hashes.get("knowledge_adaptation.json")
            != _canonical_digest(intent["knowledge_adaptation"])
        ):
            raise ValidationError(
                "diagnosis direction review proposal binding drifted"
            )
    elif base.get("review_call_intent_sha256") is not None:
        raise ValidationError(
            "diagnosis publish omitted a pending direction review"
        )

    proposal_binding = _diagnosis_proposal_binding(bundle, direction_aggregate)
    if event["payload_sha256"] != _canonical_digest(proposal_binding):
        raise ValidationError("diagnosis publish proposal payload binding drifted")
    expected_target = _diagnosis_publish_target_state(
        base,
        bundle,
        event,
        float(created),
        direction_review_complete=direction_complete,
        direction_review_aggregate=direction_aggregate,
    )
    if intent["target_state"] != expected_target:
        raise ValidationError("diagnosis publish target state drifted")
    return copy.deepcopy(dict(intent)), base, direction_complete, direction_aggregate


def _materialize_diagnosis_publish(
    run_root: Path,
    intent: Mapping[str, Any],
) -> None:
    active_root = run_root / "active_diagnosis"
    generation = intent["hypothesis_generation"]
    generation_path = (
        active_root
        / "hypothesis_generations"
        / f"{generation['hypothesis_set_sha256']}.json"
    )
    proposal_path = (
        active_root
        / "proposal_generations"
        / f"{intent['proposal_ledger_event']['payload_sha256']}.json"
    )
    fixed_artifacts = (
        (run_root / "diagnosis_context.json", intent["diagnosis_context"]),
        (
            active_root / "knowledge_context.json",
            intent["diagnosis_context"]["knowledge_context"],
        ),
        (
            active_root / "knowledge_adaptation.json",
            intent["knowledge_adaptation"],
        ),
        (active_root / "hypothesis_result.json", intent["hypothesis_result"]),
        (active_root / "request_set.json", intent["request_set"]),
        (
            active_root / "evidence_selection.json",
            intent["evidence_selection"],
        ),
        (active_root / "decision.json", intent["decision"]),
        (
            active_root / "investment_brief.json",
            intent["investment_brief"],
        ),
    )
    immutable_artifacts = (
        (
            generation_path,
            generation,
            "diagnosis immutable hypothesis generation conflicts",
        ),
        (
            proposal_path,
            intent,
            "diagnosis immutable proposal generation conflicts",
        ),
    )
    for path, artifact, conflict_message in immutable_artifacts:
        if path.exists() or path.is_symlink():
            if (
                path.is_symlink()
                or not path.is_file()
                or load_json_object(path) != artifact
            ):
                raise ValidationError(conflict_message)
        else:
            _atomic_json(path, artifact)
    for path, artifact in fixed_artifacts:
        _atomic_json(path, artifact)
    events = _verify_active_diagnosis_ledger(run_root)
    base_sequence = intent["base_ledger_sequence"]
    event = intent["proposal_ledger_event"]
    if len(events) == base_sequence:
        _atomic_json(run_root / intent["proposal_ledger_path"], event)
    elif len(events) != base_sequence + 1 or events[-1] != event:
        raise ValidationError(
            "diagnosis publish found a foreign proposal ledger tail"
        )
    for path, artifact in (
        (generation_path, generation),
        (proposal_path, intent),
        *fixed_artifacts,
    ):
        if (
            path.is_symlink()
            or not path.is_file()
            or load_json_object(path) != artifact
        ):
            raise ValidationError("diagnosis publish artifact drifted")
    events = _verify_active_diagnosis_ledger(run_root)
    if len(events) != base_sequence + 1 or events[-1] != event:
        raise ValidationError("diagnosis publish ledger was not materialized")


def _cleanup_diagnosis_publish(
    run_root: Path,
    *,
    consumed_direction_review: bool,
) -> None:
    _remove_path(
        run_root / "active_diagnosis" / "diagnosis_publish_intent.json"
    )
    if consumed_direction_review:
        _cleanup_review_call_markers(run_root, "direction")


def _publish_diagnosis_bundle(
    run_root: Path,
    base: Mapping[str, Any],
    *,
    diagnosis_context: Mapping[str, Any],
    knowledge_adaptation: Mapping[str, Any],
    hypothesis_generation: Mapping[str, Any],
    request_set: Mapping[str, Any],
    evidence_selection: Mapping[str, Any],
    decision: Mapping[str, Any],
    investment_brief: Mapping[str, Any],
) -> dict:
    direction_complete, direction_aggregate = (
        _direction_review_for_diagnosis_publish(run_root, base)
    )
    bundle = {
        "diagnosis_context": copy.deepcopy(dict(diagnosis_context)),
        "knowledge_adaptation": copy.deepcopy(dict(knowledge_adaptation)),
        "hypothesis_generation": copy.deepcopy(dict(hypothesis_generation)),
        "hypothesis_result": copy.deepcopy(dict(hypothesis_generation)),
        "request_set": copy.deepcopy(dict(request_set)),
        "evidence_selection": copy.deepcopy(dict(evidence_selection)),
        "decision": copy.deepcopy(dict(decision)),
        "investment_brief": copy.deepcopy(dict(investment_brief)),
    }
    proposal_binding = _diagnosis_proposal_binding(
        bundle, direction_aggregate
    )
    created = time.time()
    ledger_path, ledger_event = _prepare_active_diagnosis_event(
        run_root,
        "proposal",
        proposal_binding,
        created_at_epoch=created,
    )
    target = _diagnosis_publish_target_state(
        base,
        bundle,
        ledger_event,
        created,
        direction_review_complete=direction_complete,
        direction_review_aggregate=direction_aggregate,
    )
    intent = {
        "schema_version": _DIAGNOSIS_PUBLISH_INTENT_SCHEMA,
        "base_state_sha256": _canonical_digest(base),
        "base_ledger_sequence": base[
            "active_diagnosis_ledger_sequence"
        ],
        "base_ledger_head_sha256": base[
            "active_diagnosis_ledger_head_sha256"
        ],
        "direction_review_complete_sha256": (
            None
            if direction_complete is None
            else _canonical_digest(direction_complete)
        ),
        **bundle,
        "proposal_ledger_path": ledger_path,
        "proposal_ledger_event": ledger_event,
        "target_state": target,
        "created_at_epoch": created,
    }
    intent, _base, _complete, _aggregate = (
        _validate_diagnosis_publish_intent(run_root, intent)
    )
    intent_path = (
        run_root / "active_diagnosis" / "diagnosis_publish_intent.json"
    )
    _atomic_json(intent_path, intent)
    bound = copy.deepcopy(dict(base))
    bound["diagnosis_publish_intent_sha256"] = _canonical_digest(intent)
    bound["updated_at_epoch"] = created
    _write_state(run_root, bound)
    _materialize_diagnosis_publish(run_root, intent)
    _validate_diagnosis_publish_intent(run_root, intent)
    committed = _write_state(run_root, target)
    _cleanup_diagnosis_publish(
        run_root,
        consumed_direction_review=direction_complete is not None,
    )
    return committed


def _recover_diagnosis_publish(
    run_root: Path,
    state: Mapping[str, Any],
) -> tuple[dict, str]:
    intent_path = (
        run_root / "active_diagnosis" / "diagnosis_publish_intent.json"
    )
    if not intent_path.exists() and not intent_path.is_symlink():
        if state.get("diagnosis_publish_intent_sha256") is not None:
            return (
                _persist_task6_manual_recovery(
                    run_root,
                    state,
                    reason="diagnosis_publish_intent_missing",
                ),
                "manual",
            )
        return copy.deepcopy(dict(state)), "none"
    try:
        if intent_path.is_symlink() or not intent_path.is_file():
            raise ValidationError(
                "diagnosis publish intent must be a regular file"
            )
        intent, base, direction_complete, _direction_aggregate = (
            _validate_diagnosis_publish_intent(
                run_root,
                load_json_object(intent_path),
            )
        )
        intent_digest = _canonical_digest(intent)
        target = intent["target_state"]
        current_digest = _canonical_digest(state)
        binding = state.get("diagnosis_publish_intent_sha256")
        if binding is None:
            if current_digest == intent["base_state_sha256"]:
                _remove_path(intent_path)
                return copy.deepcopy(dict(state)), "discarded_unbound"
            if current_digest == _canonical_digest(target):
                _materialize_diagnosis_publish(run_root, intent)
                _cleanup_diagnosis_publish(
                    run_root,
                    consumed_direction_review=direction_complete is not None,
                )
                return copy.deepcopy(dict(state)), "cleaned"
            raise ValidationError(
                "diagnosis publish intent is not bound to current state"
            )
        if binding != intent_digest:
            raise ValidationError("diagnosis publish intent binding drifted")
        expected_bound = copy.deepcopy(base)
        expected_bound["diagnosis_publish_intent_sha256"] = intent_digest
        expected_bound["updated_at_epoch"] = intent["created_at_epoch"]
        if state != expected_bound:
            raise ValidationError("diagnosis publish bound state drifted")
        _materialize_diagnosis_publish(run_root, intent)
        _validate_diagnosis_publish_intent(run_root, intent)
        committed = _write_state(run_root, target)
        _cleanup_diagnosis_publish(
            run_root,
            consumed_direction_review=direction_complete is not None,
        )
        return committed, "recovered"
    except (KeyError, OSError, ValidationError, ValueError):
        return (
            _persist_task6_manual_recovery(
                run_root,
                state,
                reason="diagnosis_publish_recovery_invalid",
            ),
            "manual",
        )


def _controlled_spend_after_execution(
    state: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    max_controlled_seconds: float | None = None,
) -> float:
    duration = execution.get("duration_seconds")
    if (
        type(duration) not in {int, float}
        or not math.isfinite(float(duration))
        or float(duration) < 0
    ):
        raise ValidationError("evidence execution duration_seconds is invalid")
    total = _controlled_spend_seconds(state) + float(duration)
    if not math.isfinite(total):
        raise ValidationError("run controlled spend overflowed")
    if (
        max_controlled_seconds is not None
        and total > float(max_controlled_seconds)
    ):
        raise ValidationError("evidence execution exceeds controlled authorization")
    return total


def _diagnostic_investment_inputs(
    run_root: Path, state: Mapping[str, Any], context: Mapping[str, Any]
) -> dict:
    now = time.time()
    started = state.get("optimization_started_at_epoch", state["started_at_epoch"])
    wall_elapsed = max(0.0, now - float(started))
    grant = _load_bound_authorization_grant(run_root, state)
    controlled_spend = _controlled_spend_seconds(state)
    contract = load_json_object(
        run_root / "active_diagnosis" / "analysis_contract.json"
    )
    action_bounds = {
        item["action_id"]: copy.deepcopy(item["cost_bound"])
        for item in contract["actions"]
        if "cost_bound" in item
    }
    return {
        "authorization": {
            "max_seconds": float(grant["max_controlled_seconds"]),
            "max_risk": grant["max_risk"],
        },
        "spend": {"elapsed_seconds": controlled_spend},
        "wall_elapsed_seconds": wall_elapsed,
        "candidate_history": copy.deepcopy(context.get("candidate_history", [])),
        "action_bounds": action_bounds,
        "knowledge_adaptation": copy.deepcopy(
            context.get("knowledge_adaptation", context.get("knowledge_context"))
        ),
    }


def _review_diagnostic_direction(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    performance_model: Mapping[str, Any],
    hypothesis_result: Mapping[str, Any],
    request_set: Mapping[str, Any],
    knowledge_adaptation: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict | None:
    """Run an optional state-bound direction challenge."""
    if not _reviewer_configs(control):
        return None
    active = [
        item
        for item in hypothesis_result["hypothesis_set"]["hypotheses"]
        if item["disposition"] == "active"
    ]
    if state.get("hypothesis_set_sha256") is None:
        trigger = (
            "major"
            if len({item["claim_layer"] for item in active}) > 1
            else "ordinary"
        )
    elif selection.get("status") == "evidence_gap":
        trigger = "plateau"
    else:
        return None
    reviewer = _load_reviewer_module()
    selected = selection.get("selected_request")
    selected_summary = None
    if isinstance(selected, Mapping):
        action = selected.get("controller_action", {})
        selected_summary = {
            "evidence_kind": action.get("evidence_kind"),
            "cost": action.get("cost"),
        }
    request = reviewer.build_review_request(
        diagnosis={
            "review_kind": "direction",
            "performance_summary": performance_model,
        },
        change_set={"hypotheses": active},
        redacted_diff="",
        experiment={
            "selection_status": selection.get("status"),
            "selected_action": selected_summary,
        },
        artifact_hashes={
            "performance_model.json": _canonical_digest(performance_model),
            "hypothesis_result.json": _canonical_digest(hypothesis_result),
            "evidence_selection.json": _canonical_digest(selection),
            "request_set.json": _canonical_digest(request_set),
            "knowledge_adaptation.json": _canonical_digest(
                knowledge_adaptation
            ),
        },
    )
    _managed_state, aggregate = _managed_review_call(
        run_root,
        state,
        control,
        "direction",
        request,
        trigger=trigger,
    )
    return aggregate


def _register_active_diagnosis_proposal_unlocked(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    hypothesis_set: Mapping[str, Any],
    request_set: Mapping[str, Any],
    *,
    knowledge_inputs: Mapping[str, Any] | None = None,
) -> dict:
    """Replay an AI proposal against Controller-owned context and policy."""
    normalized = validate_control_manifest(control)
    if "analysis_contract" not in normalized:
        raise ValidationError("control does not enable active diagnosis")
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["control_digest"] != _canonical_digest(normalized):
        raise ValidationError("control manifest drifted before diagnosis proposal")
    frozen_control = _load_frozen_control(run_root, state)
    (
        state,
        recovered_failure_decision,
        unbound_candidate_failure,
    ) = _candidate_recovery_preflight(run_root, state, frozen_control)
    if unbound_candidate_failure:
        raise ValidationError(
            "resume the unbound candidate failure before diagnosis proposal"
        )
    if recovered_failure_decision is not None:
        return state
    if state.get("next_action") == "manual_recovery":
        return state
    state, diagnosis_recovery = _recover_diagnosis_publish(run_root, state)
    if diagnosis_recovery != "none":
        return state
    state, review_recovery, _review_aggregate = (
        _recover_reviewer_checkpoint(run_root, state, "direction")
    )
    if review_recovery == "manual":
        return state
    if review_recovery not in {"none", "waiting"}:
        return state
    if (
        review_recovery == "none"
        and state.get("review_call_intent_sha256") is not None
    ):
        return _persist_task6_manual_recovery(
            run_root,
            state,
            reason="review_intent_binding_unresolved",
        )
    if state["next_action"] != "propose_hypotheses":
        raise ValidationError("run is not ready for an active diagnosis proposal")
    _active_ledger_append_boundary(run_root, "proposal")
    _load_bound_authorization_grant(run_root, state, normalized)
    (
        context,
        epoch,
        execution_map,
        evidence_catalog,
        action_catalog,
        selection_policy,
    ) = _load_active_diagnosis_context(normalized, run_root, state)
    contract = _load_frozen_analysis_contract(run_root, state)
    (
        adapted_hypothesis_set,
        adapted_request_set,
        knowledge_adaptation,
    ) = _adapt_controller_knowledge(
        knowledge_inputs,
        contract=contract,
        execution_map=execution_map,
        action_catalog=action_catalog,
        selection_policy=selection_policy,
        knowledge_identity=context["knowledge_identity"],
        local_knowledge_context=context["knowledge_context"],
        hypothesis_set=hypothesis_set,
        request_set=request_set,
    )
    prior_result = None
    frozen_hypothesis_sha = state.get("hypothesis_set_sha256")
    if frozen_hypothesis_sha is not None:
        prior_result = load_json_object(
            run_root
            / "active_diagnosis"
            / "hypothesis_generations"
            / f"{frozen_hypothesis_sha}.json"
        )
        prior_set = prior_result.get("hypothesis_set")
        if (
            type(prior_set) is not dict
            or prior_result.get("hypothesis_set_sha256") != frozen_hypothesis_sha
            or _canonical_digest(prior_set) != frozen_hypothesis_sha
        ):
            raise ValidationError("frozen hypothesis result drifted from run state")
    try:
        hypothesis_result = _load_hypothesis_space_module().validate_hypothesis_set(
            adapted_hypothesis_set,
            epoch=epoch,
            execution_map=execution_map,
            evidence_catalog=evidence_catalog,
            closed_mechanism_keys=context.get("closed_mechanism_keys", []),
        )
        evolution = _validate_hypothesis_evolution(
            prior_result,
            hypothesis_result,
            context.get("closed_mechanism_keys", []),
            closed_scope_records=context.get("closed_scope_records", []),
            evidence_catalog=evidence_catalog,
            execution_map=execution_map,
        )
        adapted_request_set["hypothesis_set_sha256"] = hypothesis_result[
            "hypothesis_set_sha256"
        ]
        selection = _load_evidence_selector_module().select_evidence_request(
            adapted_request_set,
            epoch=epoch,
            execution_map=execution_map,
            hypothesis_result=hypothesis_result,
            evidence_catalog=evidence_catalog,
            action_catalog=action_catalog,
            policy=selection_policy,
            request_history=json.loads(
                (run_root / "active_diagnosis" / "request_history.json").read_text(
                    encoding="utf-8"
                )
            ),
            completed_action_ids=json.loads(
                (run_root / "active_diagnosis" / "completed_action_ids.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        performance_model = load_json_object(
            run_root / "active_diagnosis" / "performance_model.json"
        )
        external_review = _review_diagnostic_direction(
            normalized,
            run_root,
            state,
            performance_model,
            hypothesis_result,
            adapted_request_set,
            knowledge_adaptation,
            selection,
        )
        state = read_run_state(run_root)
        if state.get("next_action") == "manual_recovery":
            return state
        decision_state = state
        direction_complete, _direction_aggregate = (
            _direction_review_for_diagnosis_publish(run_root, state)
        )
        if direction_complete is not None:
            decision_state = copy.deepcopy(dict(state))
            decision_state["controlled_spend_seconds"] = (
                _controlled_spend_seconds(state)
                + float(direction_complete["total_wait_seconds"])
            )
        decision_context = copy.deepcopy(context)
        decision_context["knowledge_adaptation"] = copy.deepcopy(
            knowledge_adaptation
        )
        investment_inputs = _diagnostic_investment_inputs(
            run_root, decision_state, decision_context
        )
        decision = _load_diagnostic_decision_module().decide_next_step(
            performance_model,
            hypothesis_result,
            selection,
            external_review=external_review,
            **investment_inputs,
        )
    except ValueError as error:
        raise ValidationError(f"active diagnosis proposal rejected: {error}") from error
    context = copy.deepcopy(context)
    context["closed_mechanism_keys"] = evolution["closed_mechanism_keys"]
    context["closed_scope_records"] = evolution["closed_scope_records"]
    context["knowledge_context"] = _rebuild_knowledge_context(
        run_root,
        context,
        contract,
        epoch,
        execution_map,
        evidence_catalog,
        selection_policy,
        performance_model,
    )
    context["knowledge_adaptation"] = copy.deepcopy(knowledge_adaptation)
    investment_brief = copy.deepcopy(decision["investment_brief"])
    return _publish_diagnosis_bundle(
        run_root,
        state,
        diagnosis_context=context,
        knowledge_adaptation=knowledge_adaptation,
        hypothesis_generation=hypothesis_result,
        request_set=adapted_request_set,
        evidence_selection=selection,
        decision=decision,
        investment_brief=investment_brief,
    )


def _validate_evidence_result(
    value: Mapping[str, Any], selected: Mapping[str, Any], attempt_root: Path
) -> dict:
    result = _object(value, "evidence_result")
    fields = {
        "schema_version",
        "request_signature",
        "status",
        "outcome_id",
        "observations",
        "artifacts",
    }
    _closed(result, fields, "evidence_result")
    _required(result, fields, "evidence_result")
    if result["schema_version"] != "cuda-optimizer/evidence-result-v1":
        raise ValidationError("evidence_result schema_version is unsupported")
    if result["request_signature"] != selected["request_signature"]:
        raise ValidationError("evidence result request signature does not match selection")
    if result["status"] not in {"observed", "inconclusive", "unavailable", "failed"}:
        raise ValidationError("evidence_result.status is unsupported")
    outcome_ids = {item["outcome_id"] for item in selected["outcomes"]}
    outcome_id = result["outcome_id"]
    if result["status"] == "observed":
        if outcome_id not in outcome_ids:
            raise ValidationError("observed evidence must name a selected outcome")
    elif outcome_id is not None:
        raise ValidationError("non-observed evidence must use a null outcome_id")
    observations = _object(result["observations"], "evidence_result.observations")
    raw_updates = observations.get("execution_map_node_updates")
    if raw_updates is not None:
        if result["status"] != "observed":
            raise ValidationError(
                "only observed evidence can update execution-map nodes"
            )
        if type(raw_updates) is not list or not raw_updates or len(raw_updates) > 256:
            raise ValidationError(
                "execution_map_node_updates must contain 1 to 256 entries"
            )
        seen_node_ids = set()
        for index, raw_update in enumerate(raw_updates):
            update = _object(
                raw_update,
                f"evidence_result.observations.execution_map_node_updates[{index}]",
            )
            _closed(
                update,
                {"node_id", "duration_us", "first_start_us", "last_end_us"},
                f"evidence_result.observations.execution_map_node_updates[{index}]",
            )
            _required(
                update,
                {"node_id", "duration_us"},
                f"evidence_result.observations.execution_map_node_updates[{index}]",
            )
            node_id = _identifier(update["node_id"], f"node update {index} node_id")
            if node_id in seen_node_ids:
                raise ValidationError("execution_map_node_updates contains duplicate nodes")
            seen_node_ids.add(node_id)
            duration = update["duration_us"]
            if (
                type(duration) not in {int, float}
                or not math.isfinite(float(duration))
                or float(duration) <= 0
            ):
                raise ValidationError("node update duration_us must be positive and finite")
            has_start = "first_start_us" in update
            has_end = "last_end_us" in update
            if has_start != has_end:
                raise ValidationError(
                    "node update timing bounds must be supplied together"
                )
            if has_start:
                start = update["first_start_us"]
                end = update["last_end_us"]
                if (
                    type(start) not in {int, float}
                    or type(end) not in {int, float}
                    or not math.isfinite(float(start))
                    or not math.isfinite(float(end))
                    or float(start) < 0
                    or float(end) <= float(start)
                ):
                    raise ValidationError(
                        "node update timing bounds must be finite and increasing"
                    )
    artifacts = result["artifacts"]
    if type(artifacts) is not list:
        raise ValidationError("evidence_result.artifacts must be an array")
    sealed_artifacts = []
    resolved_attempt = attempt_root.resolve(strict=True)
    reserved_artifact_paths = {
        ".output.json",
        "request.json",
        "intent.json",
        "execution.json",
        "result.json",
        "complete.json",
    }
    for index, item in enumerate(artifacts):
        artifact = _object(item, f"evidence_result.artifacts[{index}]")
        _closed(artifact, {"path", "sha256"}, f"evidence_result.artifacts[{index}]")
        _required(artifact, {"path"}, f"evidence_result.artifacts[{index}]")
        if "sha256" in artifact:
            _sha256(
                artifact["sha256"], f"evidence_result.artifacts[{index}].sha256"
            )
        raw_path = Path(
            _string(artifact["path"], f"evidence_result.artifacts[{index}].path")
        )
        artifact_path = raw_path if raw_path.is_absolute() else attempt_root / raw_path
        resolved_artifact = artifact_path.resolve(strict=False)
        if (
            not _is_within(resolved_artifact, resolved_attempt)
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
        ):
            raise ValidationError("evidence artifact must be a contained regular file")
        relative = resolved_artifact.relative_to(resolved_attempt)
        if relative.as_posix() in reserved_artifact_paths:
            raise ValidationError(
                "evidence artifact cannot use a Controller-reserved path"
            )
        sealed_artifacts.append(
            {"path": str(relative), "sha256": _sha256_path(artifact_path)}
        )
    return {
        **copy.deepcopy(dict(result)),
        "observations": _json_copy(
            observations, "evidence_result.observations", reject_sensitive=True
        ),
        "artifacts": sealed_artifacts,
    }


def _apply_execution_map_node_updates(
    execution_map: Mapping[str, Any], result: Mapping[str, Any], evidence_id: str | None
) -> dict:
    """Apply only sealed, observed node durations to existing map nodes."""
    updated = copy.deepcopy(dict(execution_map))
    changes = result.get("observations", {}).get("execution_map_node_updates")
    if changes is None:
        return updated
    if result.get("status") != "observed" or evidence_id is None:
        raise ValidationError("execution-map updates require admitted observed evidence")
    nodes = {item["node_id"]: item for item in updated["nodes"]}
    for change in changes:
        node = nodes.get(change["node_id"])
        if node is None:
            raise ValidationError("execution-map update references an unknown node")
        if node["timing_status"] != "observed":
            raise ValidationError("execution-map update requires observed node timing")
        duration = float(change["duration_us"])
        first_start = float(change.get("first_start_us", node["first_start_us"]))
        last_end = float(change.get("last_end_us", node["last_end_us"]))
        window = updated["window"]
        if first_start < float(window["start_us"]) or last_end > float(window["end_us"]):
            raise ValidationError("execution-map update leaves the analysis window")
        if duration > last_end - first_start:
            raise ValidationError("execution-map update exceeds the observed node span")
        node["duration_us"] = duration
        node["first_start_us"] = first_start
        node["last_end_us"] = last_end
        if evidence_id not in node["evidence_ids"]:
            node["evidence_ids"].append(evidence_id)
            node["evidence_ids"].sort()
    return updated


def _active_evidence_execution_budget(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    grant: Mapping[str, Any],
) -> dict | None:
    """Close execution, termination, and accounting jitter inside the grant."""
    remaining_authorization = (
        float(grant["max_controlled_seconds"])
        - _controlled_spend_seconds(state)
    )
    cost_bound = _object(
        action.get("cost_bound"),
        "evidence action cost_bound",
    )
    action_p90 = cost_bound.get("p90_seconds")
    if (
        type(action_p90) not in {int, float}
        or not math.isfinite(float(action_p90))
        or float(action_p90) <= 0.0
    ):
        raise ValidationError("evidence action p90 cost bound is invalid")
    termination_reserve = _EVIDENCE_TERMINATION_RESERVE_SECONDS
    accounting_margin = _EVIDENCE_ACCOUNTING_MARGIN_SECONDS
    required_authorization = (
        float(action_p90)
        + termination_reserve
        + accounting_margin
    )
    if required_authorization > remaining_authorization:
        return None
    execution_timeout = min(
        float(action["timeout_seconds"]),
        remaining_authorization - termination_reserve - accounting_margin,
    )
    if execution_timeout <= 0.0:
        return None
    return {
        "execution_timeout_seconds": execution_timeout,
        "termination_reserve_seconds": termination_reserve,
        "accounting_margin_seconds": accounting_margin,
    }


def _run_active_evidence_adapter(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    selected: Mapping[str, Any],
    attempt_root: Path,
    execution_budget: Mapping[str, Any],
) -> tuple[dict, dict]:
    output_path = attempt_root / ".output.json"
    request_path = attempt_root / "request.json"
    _atomic_json(request_path, selected)
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    project_root = Path(control["project_root"])
    execution_root = project_root
    execution_argv = list(action["argv"])
    project_identity_before = _identity(control, "project")
    project_surface_before = _project_surface_identity(project_root)
    if selected["controller_action"]["control_scope"] == "project_copy":
        execution_root = attempt_root / "project_copy"
        if execution_root.exists() or execution_root.is_symlink():
            raise ValidationError("direction experiment project copy already exists")
        shutil.copytree(
            project_root,
            execution_root,
            symlinks=False,
            ignore=shutil.ignore_patterns(
                ".git", ".worktrees", "__pycache__", "*.pyc"
            ),
        )
        mapped_argv = []
        for token in execution_argv:
            token_path = Path(token)
            if token_path.is_absolute() and _is_within(
                token_path.resolve(strict=False), project_root
            ):
                mapped_argv.append(
                    str(execution_root / token_path.resolve(strict=False).relative_to(project_root))
                )
            else:
                mapped_argv.append(token)
        execution_argv = mapped_argv
    environment, secret_values = _probe_environment(
        {
            "CUDA_OPTIMIZER_EVIDENCE_OUTPUT": str(output_path),
            "CUDA_OPTIMIZER_EVIDENCE_REQUEST": str(request_path),
            "CUDA_OPTIMIZER_EVIDENCE_DIR": str(attempt_root),
            "CUDA_OPTIMIZER_RUN_DIR": str(run_root),
            "CUDA_OPTIMIZER_PROJECT_ROOT": str(execution_root),
        }
    )
    stdout = _BoundedLog(_DEFAULT_LOG_LIMIT)
    stderr = _BoundedLog(_DEFAULT_LOG_LIMIT)
    budget = _object(execution_budget, "evidence execution budget")
    if set(budget) != {
        "execution_timeout_seconds",
        "termination_reserve_seconds",
        "accounting_margin_seconds",
    }:
        raise ValidationError("evidence execution budget fields are invalid")
    timeout = float(budget["execution_timeout_seconds"])
    termination_reserve = float(budget["termination_reserve_seconds"])
    if (
        not math.isfinite(timeout)
        or timeout <= 0.0
        or not math.isfinite(termination_reserve)
        or termination_reserve <= 0.0
    ):
        raise ValidationError(
            "evidence execution budget values are invalid"
        )
    accounting_margin = float(budget["accounting_margin_seconds"])
    if (
        not math.isfinite(accounting_margin)
        or accounting_margin <= 0.0
    ):
        raise ValidationError(
            "evidence execution accounting margin is invalid"
        )
    started = time.monotonic()
    exit_code = None
    timed_out = False
    events: list[dict] = []
    process = subprocess.Popen(
        execution_argv,
        cwd=execution_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    exit_code, timed_out, elapsed, stop_reason = _wait_process_with_heartbeats(
        process,
        timeout_seconds=timeout,
        termination_grace_seconds=termination_reserve,
        accounting_margin_seconds=accounting_margin,
        label=f"evidence-{action['action_id']}",
        event_sink=events.append,
    )
    for reader in readers:
        reader.join(timeout=1)
    execution = {
        "schema_version": "cuda-optimizer/evidence-execution-v1",
        "action_id": action["action_id"],
        "argv_sha256": _canonical_digest(action["argv"]),
        "execution_argv_sha256": _canonical_digest(execution_argv),
        "adapter_sha256": action["adapter_sha256"],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": elapsed,
        "stop_reason": stop_reason,
        "events": events,
        "stdout": _redact_log(stdout.text(), secret_values),
        "stderr": _redact_log(stderr.text(), secret_values),
    }
    _atomic_json(attempt_root / "execution.json", execution)
    if _identity(control, "project")["digest"] != project_identity_before["digest"]:
        raise ValidationError("evidence action modified the frozen project")
    if _project_surface_identity(project_root) != project_surface_before:
        raise ValidationError("evidence action modified the complete project surface")
    if timed_out:
        result = {
            "schema_version": "cuda-optimizer/evidence-result-v1",
            "request_signature": selected["request_signature"],
            "status": "unavailable",
            "outcome_id": None,
            "observations": {"reason": "timeout"},
            "artifacts": [],
        }
    elif exit_code != 0:
        result = {
            "schema_version": "cuda-optimizer/evidence-result-v1",
            "request_signature": selected["request_signature"],
            "status": "failed",
            "outcome_id": None,
            "observations": {"reason": "nonzero_exit"},
            "artifacts": [],
        }
    else:
        result = _validate_evidence_result(
            _read_probe_output(output_path), selected, attempt_root
        )
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    return result, execution


def _refresh_active_diagnosis_context(
    run_root: Path,
    context: Mapping[str, Any],
    epoch: Mapping[str, Any],
    execution_map: Mapping[str, Any],
    evidence_catalog: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    result_summary: Mapping[str, Any],
    minimum_effect_us: float,
    contract: Mapping[str, Any],
) -> dict:
    refreshed = copy.deepcopy(dict(context))
    refreshed["epoch_id"] = epoch["epoch_id"]
    refreshed["epoch_sha256"] = _load_analysis_epoch_module().epoch_digest(epoch)
    refreshed["evidence_catalog_sha256"] = _canonical_digest(evidence_catalog)
    refreshed["selection_policy_sha256"] = _canonical_digest(selection_policy)
    refreshed["request_history_sha256"] = _canonical_digest(
        json.loads(
            (run_root / "active_diagnosis" / "request_history.json").read_text(
                encoding="utf-8"
            )
        )
    )
    refreshed["completed_action_ids_sha256"] = _canonical_digest(
        json.loads(
            (run_root / "active_diagnosis" / "completed_action_ids.json").read_text(
                encoding="utf-8"
            )
        )
    )
    refreshed["execution_map_sha256"] = (
        _load_execution_map_module().execution_map_digest(
            execution_map, epoch=epoch, evidence_catalog=evidence_catalog
        )
    )
    evidence_results = refreshed.get("evidence_results", [])
    if type(evidence_results) is not list:
        raise ValidationError("diagnosis context evidence_results is invalid")
    evidence_results.append(copy.deepcopy(dict(result_summary)))
    refreshed["evidence_results"] = evidence_results
    action_timings = [
        {
            "action_id": item["action_id"],
            "identities": copy.deepcopy(execution_map["identities"]),
            "elapsed_seconds": item["duration_seconds"],
        }
        for item in evidence_results
        if item["duration_seconds"] > 0
    ]
    performance_model = _load_performance_model_module().build_performance_model(
        execution_map,
        minimum_effect_us=minimum_effect_us,
        action_timings=action_timings,
    )
    _atomic_json(
        run_root / "active_diagnosis" / "performance_model.json",
        performance_model,
    )
    refreshed["performance_model_sha256"] = _canonical_digest(performance_model)
    refreshed["knowledge_context"] = _rebuild_knowledge_context(
        run_root, refreshed, contract, epoch, execution_map, evidence_catalog,
        selection_policy, performance_model, pending_summary=result_summary,
    )
    _atomic_json(run_root / "diagnosis_context.json", refreshed)
    _atomic_json(
        run_root / "active_diagnosis" / "knowledge_context.json",
        refreshed["knowledge_context"],
    )
    return refreshed


def _recover_or_block_active_evidence_attempt(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    selected: Mapping[str, Any],
    attempt_root: Path,
) -> dict | None:
    intent_path = attempt_root / "intent.json"
    complete_path = attempt_root / "complete.json"
    if not intent_path.exists() and not complete_path.exists():
        return None
    if intent_path.is_symlink() or not intent_path.is_file():
        raise ValidationError("evidence intent is not a regular file")
    if not complete_path.exists():
        updated = copy.deepcopy(dict(state))
        updated.update(
            {
                "status": "blocked",
                "stage": "active_diagnosis",
                "next_action": "manual_recovery",
                "updated_at_epoch": time.time(),
                "manual_recovery_reason": "evidence_action_interrupted_not_reexecuted",
            }
        )
        return _write_state(run_root, updated)
    if complete_path.is_symlink() or not complete_path.is_file():
        raise ValidationError("evidence completion is not a regular file")
    completion = load_json_object(complete_path)
    fields = {
        "schema_version",
        "request_signature",
        "result_sha256",
        "execution_sha256",
        "context_sha256",
        "completed_at_epoch",
    }
    _closed(completion, fields, "evidence_completion")
    _required(completion, fields, "evidence_completion")
    if completion["schema_version"] != "cuda-optimizer/evidence-completion-v1":
        raise ValidationError("evidence completion schema_version is unsupported")
    signature = selected["request_signature"]
    if completion["request_signature"] != signature:
        raise ValidationError("evidence completion request signature drifted")
    result = load_json_object(attempt_root / "result.json")
    execution = load_json_object(attempt_root / "execution.json")
    context = load_json_object(run_root / "diagnosis_context.json")
    expected_digests = {
        "result_sha256": _canonical_digest(result),
        "execution_sha256": _canonical_digest(execution),
        "context_sha256": _canonical_digest(context),
    }
    for field, digest in expected_digests.items():
        if completion[field] != digest:
            raise ValidationError(f"evidence completion {field} drifted")
    event_payload = {
        "request_signature": signature,
        **expected_digests,
    }
    payload_sha = _canonical_digest(event_payload)
    events, committed_sequence = _active_ledger_append_boundary(
        run_root,
        "evidence",
    )
    if (
        len(events) != committed_sequence + 1
        or events[-1]["payload_sha256"] != payload_sha
    ):
        raise ValidationError("evidence completion has no matching ledger event")
    grant = _load_bound_authorization_grant(run_root, state, control)
    timed_out = execution.get("timed_out") is True
    recovered = copy.deepcopy(dict(state))
    recovered.update(
        {
            "stage": "active_diagnosis",
            "next_action": (
                "review_required" if timed_out else "propose_hypotheses"
            ),
            "updated_at_epoch": time.time(),
            "diagnosis_context_sha256": completion["context_sha256"],
            "last_request_signature": signature,
            "active_diagnosis_round": int(state.get("active_diagnosis_round", 1)) + 1,
            "controlled_spend_seconds": _controlled_spend_after_execution(
                state,
                execution,
                max_controlled_seconds=grant["max_controlled_seconds"],
            ),
        }
    )
    if timed_out:
        recovered["terminal_reason"] = "evidence_action_timeout"
    recovered.update(_active_ledger_binding(events))
    _load_active_diagnosis_context(control, run_root, recovered)
    return _write_state(run_root, recovered)


def collect_active_diagnosis_evidence(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        _require_run_grant_investment_control(read_run_state(run_root))
        return _collect_active_diagnosis_evidence_unlocked(control, run_root)


def _collect_active_diagnosis_evidence_unlocked(
    control: Mapping[str, Any], run_dir: os.PathLike[str] | str
) -> dict:
    """Execute one frozen evidence action and checkpoint the next diagnosis round."""
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["control_digest"] != _canonical_digest(normalized):
        raise ValidationError("control manifest drifted before evidence collection")
    if (
        state["next_action"] == "propose_hypotheses"
        and state.get("last_request_signature")
    ):
        return state
    if state["next_action"] != "collect_evidence":
        raise ValidationError("run is not ready to collect active diagnosis evidence")
    grant = _load_bound_authorization_grant(run_root, state, normalized)
    decision = _load_bound_diagnostic_artifacts(
        run_root, state, expected_decision="MEASURE"
    )
    selection = load_json_object(
        run_root / "active_diagnosis" / "evidence_selection.json"
    )
    if _canonical_digest(selection) != state.get("evidence_selection_sha256"):
        raise ValidationError("evidence selection digest drifted before collection")
    selected = selection.get("selected_request")
    if type(selected) is not dict or selection.get("status") != "selected":
        raise ValidationError("evidence selection contains no executable request")
    authorized_action = decision.get("next_action")
    if not isinstance(authorized_action, Mapping) or any(
        authorized_action.get(field) != selected.get(field)
        for field in ("request_id", "action_id")
    ):
        raise ValidationError("diagnostic decision does not match evidence selection")
    signature = selected["request_signature"]
    attempt_root = run_root / "active_diagnosis" / "evidence" / signature
    recovered = _recover_or_block_active_evidence_attempt(
        normalized, run_root, state, selected, attempt_root
    )
    if recovered is not None:
        return recovered
    if not _verify_readiness_report(normalized, run_root, state):
        report = _run_readiness_gate_checked(normalized, run_root, state)
        refreshed_state = copy.deepcopy(state)
        refreshed_state["readiness_report_digest"] = _readiness_report_digest(
            run_root, report
        )
        refreshed_state["readiness_environment_identity_digest"] = report.get(
            "environment_identity_digest"
        )
        refreshed_state["updated_at_epoch"] = time.time()
        if not report.get("can_start_diagnosis"):
            refreshed_state["stage"] = "readiness"
            refreshed_state["next_action"] = "readiness_action"
            return _write_state(run_root, refreshed_state)
        if report.get("environment_identity_digest") != state.get(
            "baseline_environment_identity_digest"
        ):
            raise ValidationError(
                "environment identity changed after baseline; create a child run"
            )
        state = _write_state(run_root, refreshed_state)
    (
        context,
        epoch,
        execution_map,
        evidence_catalog,
        _action_catalog,
        selection_policy,
    ) = _load_active_diagnosis_context(normalized, run_root, state)
    contract = _load_frozen_analysis_contract(run_root, state)
    action_by_id = {item["action_id"]: item for item in contract["actions"]}
    action = action_by_id.get(selected["action_id"])
    if action is None:
        raise ValidationError("selected evidence action has no frozen adapter")
    execution_budget = _active_evidence_execution_budget(
        state,
        action,
        grant,
    )
    if execution_budget is None:
        updated = copy.deepcopy(state)
        updated.update(
            {
                "status": "active",
                "stage": "active_diagnosis",
                "next_action": "review_required",
                "updated_at_epoch": time.time(),
                "terminal_reason": (
                    "evidence_action_authorization_insufficient"
                ),
            }
        )
        return _write_state(run_root, updated)
    required = set(selected["controller_action"]["required_capability_ids"])
    readiness_report = _load_readiness_gate_module()._load_prior_report(
        run_root / "readiness"
    )
    if readiness_report is None:
        raise ValidationError("evidence collection requires a readiness report")
    available = set(_ready_capability_ids(readiness_report))
    if not required.issubset(available):
        updated = copy.deepcopy(state)
        updated.update(
            {
                "stage": "active_diagnosis",
                "next_action": "evidence_gap",
                "updated_at_epoch": time.time(),
                "missing_capability_ids": sorted(required - available),
            }
        )
        return _write_state(run_root, updated)
    adapter_path = Path(action["adapter_path"])
    if _sha256_path(adapter_path) != action["adapter_sha256"]:
        raise ValidationError("evidence action adapter drifted before execution")
    bindings = _load_frozen_execution_bindings(run_root, state)
    expected_binding = bindings.get("actions", {}).get(action["action_id"])
    if type(expected_binding) is not dict:
        raise ValidationError("evidence action has no frozen execution binding")
    _verify_adapter_execution_binding(
        expected_binding,
        adapter_path,
        action["argv"],
        f"analysis_contract action {action['action_id']}",
    )

    intent_path = attempt_root / "intent.json"
    complete_path = attempt_root / "complete.json"
    intent = {
        "schema_version": "cuda-optimizer/evidence-intent-v1",
        "request_signature": signature,
        "selection_sha256": _canonical_digest(selection),
        "action_sha256": _canonical_digest(action),
        "created_at_epoch": time.time(),
    }
    _atomic_json(intent_path, intent)
    result, execution = _run_active_evidence_adapter(
        normalized,
        run_root,
        state,
        action,
        selected,
        attempt_root,
        execution_budget,
    )
    evidence_id = None
    if result["status"] == "observed":
        evidence_id = f"ev-{signature[:16]}"
    execution_map = _apply_execution_map_node_updates(
        execution_map, result, evidence_id
    )
    _atomic_json(attempt_root / "result.json", result)
    if result["status"] == "observed":
        outcome = next(
            item for item in selected["outcomes"] if item["outcome_id"] == result["outcome_id"]
        )
        evidence_catalog[evidence_id] = {
            "epoch_id": epoch["epoch_id"],
            "kind": selected["controller_action"]["evidence_kind"],
            "artifact_sha256": _sha256_path(attempt_root / "result.json"),
            "supports_hypothesis_ids": sorted(outcome["supports"]),
            "opposes_hypothesis_ids": sorted(outcome["opposes"]),
        }
    try:
        execution_map = _load_execution_map_module().validate_execution_map(
            execution_map,
            epoch=epoch,
            evidence_catalog=evidence_catalog,
        )["execution_map"]
    except ValueError as error:
        raise ValidationError(
            f"evidence produced an invalid execution-map update: {error}"
        ) from error
    _atomic_json(run_root / "active_diagnosis" / "execution_map.json", execution_map)
    _atomic_json(run_root / "active_diagnosis" / "evidence_catalog.json", evidence_catalog)
    history_path = run_root / "active_diagnosis" / "request_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if type(history) is not list:
        raise ValidationError("active diagnosis request history is invalid")
    if signature not in history:
        history.append(signature)
    history.sort()
    _atomic_json(history_path, history)
    completed_path = run_root / "active_diagnosis" / "completed_action_ids.json"
    completed_actions = json.loads(completed_path.read_text(encoding="utf-8"))
    if type(completed_actions) is not list:
        raise ValidationError("active diagnosis completed action history is invalid")
    if selected["action_id"] not in completed_actions:
        completed_actions.append(selected["action_id"])
    completed_actions.sort()
    _atomic_json(completed_path, completed_actions)
    selection_policy = copy.deepcopy(selection_policy)
    if _load_evidence_selector_module().action_consumes_profile_budget(
        selected["controller_action"]
    ):
        selection_policy["remaining_profile_actions"] = max(
            0, int(selection_policy["remaining_profile_actions"]) - 1
        )
    _atomic_json(
        run_root / "active_diagnosis" / "selection_policy.json", selection_policy
    )
    refreshed_context = _refresh_active_diagnosis_context(
        run_root,
        context,
        epoch,
        execution_map,
        evidence_catalog,
        selection_policy,
        {
            "request_signature": signature,
            "action_id": selected["action_id"],
            "evidence_id": evidence_id,
            "status": result["status"],
            "outcome_id": result["outcome_id"],
            "result_path": str(
                (attempt_root / "result.json").relative_to(run_root)
            ),
            "result_sha256": _canonical_digest(result),
            "duration_seconds": execution["duration_seconds"],
        },
        contract["minimum_effect_us"],
        contract,
    )
    event_payload = {
        "request_signature": signature,
        "result_sha256": _canonical_digest(result),
        "execution_sha256": _canonical_digest(execution),
        "context_sha256": _canonical_digest(refreshed_context),
    }
    _append_active_diagnosis_event(run_root, "evidence", event_payload)
    _atomic_json(
        complete_path,
        {
            "schema_version": "cuda-optimizer/evidence-completion-v1",
            **event_payload,
            "completed_at_epoch": time.time(),
        },
    )
    updated = copy.deepcopy(state)
    timed_out = execution.get("timed_out") is True
    updated.update(
        {
            "stage": "active_diagnosis",
            "next_action": (
                "review_required" if timed_out else "propose_hypotheses"
            ),
            "updated_at_epoch": time.time(),
            "diagnosis_context_sha256": _canonical_digest(refreshed_context),
            "last_request_signature": signature,
            "active_diagnosis_round": int(state.get("active_diagnosis_round", 1)) + 1,
            "controlled_spend_seconds": _controlled_spend_after_execution(
                state,
                execution,
                max_controlled_seconds=grant["max_controlled_seconds"],
            ),
        }
    )
    if timed_out:
        updated["terminal_reason"] = "evidence_action_timeout"
    updated.update(_active_ledger_binding(_verify_active_diagnosis_ledger(run_root)))
    return _write_state(run_root, updated)


def register_change(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    change_set: Mapping[str, Any],
) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        _require_run_grant_investment_control(read_run_state(run_root))
        return _register_change_unlocked(control, run_root, change_set)


def _register_change_unlocked(
    control: Mapping[str, Any],
    run_dir: os.PathLike[str] | str,
    change_set: Mapping[str, Any],
) -> dict:
    normalized = validate_control_manifest(control)
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["control_digest"] != _canonical_digest(normalized):
        raise ValidationError("control manifest drifted before ChangeSet registration")
    _load_frozen_control(run_root, state)
    diagnostic_decision = None
    if "analysis_contract" in normalized:
        diagnostic_decision = _load_bound_diagnostic_artifacts(
            run_root, state, expected_decision="PURSUE"
        )
    change = validate_change_set(change_set, normalized)
    candidate_hypothesis_id = None
    candidate_identity_digest = None
    grant = None
    if diagnostic_decision is not None:
        next_diagnostic_action = diagnostic_decision.get("next_action")
        if not isinstance(next_diagnostic_action, Mapping):
            raise ValidationError("diagnostic decision lacks a bound candidate action")
        candidate_hypothesis_id = _identifier(
            next_diagnostic_action.get("hypothesis_id"),
            "diagnostic candidate hypothesis_id",
        )
        if change["diagnosis_ids"] != [candidate_hypothesis_id]:
            raise ValidationError(
                "ChangeSet diagnosis_ids must contain exactly the authorized diagnostic candidate"
            )
        bound_basis = diagnostic_decision.get("investment_brief", {}).get(
            "bound_basis", {}
        )
        candidate_identity_digest = _sha256(
            bound_basis.get("identity_digest"),
            "diagnostic candidate identity_digest",
        )
        grant = _load_bound_authorization_grant(
            run_root,
            state,
            normalized,
        )
    workload = _normalize_frozen_workload(normalized)
    if workload.source_hash != state["workload_source_hash"]:
        raise ValidationError("workload identity drifted before ChangeSet registration")
    _validate_workload_candidate_minimum_effect(change, workload)
    change_digest = _canonical_digest(change)
    pending_path = run_root / "registration_pending.json"
    if state["next_action"] == "edit_then_evaluate":
        if state.get("change_set_digest") != change_digest:
            raise ValidationError("a different ChangeSet is already registered")
        try:
            pending_path.unlink()
        except FileNotFoundError:
            pass
        return state
    registration_pause = (
        state["next_action"] == "review_required"
        and str(state.get("terminal_reason", "")).startswith(
            "candidate_registration_"
        )
    )
    if state["next_action"] != "register_change" and not registration_pause:
        raise ValidationError("run is not ready to register a ChangeSet")
    if grant is not None:
        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        first_stage = "static_review"
        first_stage_p90 = float(
            change["candidate"]["estimated_cost"][first_stage][
                "p90_seconds"
            ]
        )
        remaining = (
            float(grant["max_controlled_seconds"])
            - _controlled_spend_seconds(state)
        )
        blocked_reason = None
        if change["scope"] not in grant["allowed_mutation_scopes"]:
            blocked_reason = "candidate_registration_scope_unauthorized"
        elif risk_order[change["risk"]] > risk_order[grant["max_risk"]]:
            blocked_reason = "candidate_registration_risk_unauthorized"
        elif _RUN_AUTHORIZATION_STAGES.index(
            grant["max_stage"]
        ) < _RUN_AUTHORIZATION_STAGES.index(first_stage):
            blocked_reason = "candidate_registration_stage_unauthorized"
        elif first_stage_p90 > remaining:
            blocked_reason = "candidate_registration_time_unauthorized"
        if blocked_reason is not None:
            paused = copy.deepcopy(state)
            paused.update(
                {
                    "status": "active",
                    "stage": "active_diagnosis",
                    "next_action": "review_required",
                    "terminal_reason": blocked_reason,
                    "updated_at_epoch": time.time(),
                }
            )
            return _write_state(run_root, paused)
    if change["scope"] == "project":
        current_identity = _identity(normalized, "project")
        if current_identity["digest"] != state.get("baseline_identity_digest"):
            raise ValidationError(
                "declared project identity drifted after baseline capture"
            )
    else:
        expected_environment = state.get("baseline_environment_identity_digest")
        if expected_environment is None:
            raise ValidationError(
                "environment_root must exist before baseline capture for isolated changes"
            )
        if _identity(normalized, "isolated_environment")["digest"] != expected_environment:
            raise ValidationError(
                "isolated environment drifted after baseline capture"
            )
    pending = {
        "schema_version": "cuda-workload-optimizer/registration-pending-v1",
        "change_set_digest": change_digest,
        "scope": change["scope"],
    }
    if pending_path.exists():
        if load_json_object(pending_path) != pending:
            raise ValidationError("pending ChangeSet registration does not match retry")
        before_path = run_root / "rounds" / "round-1" / "before_identity.json"
        if before_path.exists():
            before_value = load_json_object(before_path)
            before = _validated_identity_artifact(
                before_value, before_value.get("digest", "")
            )
        else:
            snapshot_name = "project" if change["scope"] == "project" else "environment"
            incomplete_snapshot = run_root / "snapshot" / snapshot_name
            if incomplete_snapshot.exists() or incomplete_snapshot.is_symlink():
                _remove_path(incomplete_snapshot)
            before = _snapshot_scope(normalized, run_root, change["scope"])
    else:
        _atomic_json(pending_path, pending)
        before = _snapshot_scope(normalized, run_root, change["scope"])
    _atomic_json(run_root / "change_set.json", change)
    _atomic_json(run_root / "rounds" / "round-1" / "change_set.json", change)
    updated = copy.deepcopy(state)
    for field in _CANDIDATE_STAGE_STATE_FIELDS:
        updated.pop(field, None)
    if "change" not in updated["completed_stages"]:
        updated["completed_stages"].append("change")
    updated.update(
        {
            "stage": "review",
            "next_action": "edit_then_evaluate",
            "updated_at_epoch": time.time(),
            "before_identity_digest": before["digest"],
            "change_set_digest": change_digest,
            "change_scope": change["scope"],
            "candidate_stage": "static_review",
        }
    )
    if diagnostic_decision is not None:
        updated.update(
            {
                "candidate_hypothesis_id": candidate_hypothesis_id,
                "candidate_identity_digest": candidate_identity_digest,
                "candidate_digest": _canonical_digest(change["candidate"]),
                "candidate_started_at_epoch": time.time(),
            }
        )
    committed = _write_state(run_root, updated)
    try:
        pending_path.unlink()
    except FileNotFoundError:
        pass
    return committed


def _run_python_workload_once_bounded(
    control: Mapping[str, Any],
    run_root: Path,
    *,
    candidate: Any,
    role: str,
    case: Mapping[str, Any] | None,
    timeout_seconds: float,
    task: str,
) -> dict:
    """Run a Python adapter outside the Controller process under a hard stop."""
    label = _identifier(task, "workload task")
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise ValidationError("workload timeout must be a positive finite number")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 3600:
        raise ValidationError("workload timeout must be positive and at most 3600 seconds")
    attempt_dir = run_root / "workload_attempts" / label
    request_path = attempt_dir / "request.json"
    output_path = attempt_dir / "output.json"
    _atomic_json(
        request_path,
        {"candidate": candidate, "role": role, "case": case},
    )
    environment, _secrets = _probe_environment({})
    events: list[dict] = []

    def emit(event: Mapping[str, Any]) -> None:
        record = {**event, "task": label}
        events.append(record)
        if event.get("event") == "heartbeat" or event.get("stop_reason") != "completed":
            print(
                json.dumps(record, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )

    result = _load_budget_module().run_budgeted_command(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_workload-once",
            "--control",
            str(run_root / "control_manifest.json"),
            "--request",
            str(request_path),
            "--out",
            str(output_path),
        ],
        timeout_seconds=timeout,
        grace_seconds=min(0.5, timeout),
        popen_options={
            "cwd": control["project_root"],
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        },
        heartbeat_interval_seconds=min(30.0, timeout),
        event_sink=emit,
    )
    _atomic_json(
        attempt_dir / "execution.json",
        {
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.stop_reason,
            "events": events,
        },
    )
    if result.timed_out:
        raise TimeoutError("Python workload reached the hard deadline")
    if result.returncode != 0 or not output_path.is_file():
        raise RuntimeError("Python workload child failed")
    return load_json_object(output_path)


def _run_candidate_static_review_bounded(
    control: Mapping[str, Any],
    run_root: Path,
    *,
    timeout_seconds: float,
) -> dict:
    """Run the static candidate falsifier in a killable process group."""
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise ValidationError(
            "candidate static review timeout must be a positive finite number"
        )
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValidationError(
            "candidate static review timeout must be a positive finite number"
        )
    artifact_path = run_root / _CANDIDATE_STAGE_ARTIFACTS["static_review"]
    environment, _secrets = _probe_environment({})

    def emit(event: Mapping[str, Any]) -> None:
        if (
            event.get("event") == "heartbeat"
            or event.get("stop_reason") != "completed"
        ):
            print(
                json.dumps(
                    {**event, "task": "candidate-static-review"},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    result = _load_budget_module().run_budgeted_command(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_candidate-static-review",
            "--run-dir",
            str(run_root),
            "--out",
            str(artifact_path),
        ],
        timeout_seconds=timeout,
        grace_seconds=min(0.5, timeout),
        popen_options={
            "cwd": control["project_root"],
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        },
        heartbeat_interval_seconds=min(30.0, timeout),
        event_sink=emit,
    )
    if result.timed_out:
        raise _CandidateStageTimeout(
            "candidate static review exhausted its controlled timeout"
        )
    if result.returncode != 0:
        raise RuntimeError("candidate static review child failed")
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise RuntimeError("candidate static review child produced no artifact")
    return load_json_object(artifact_path)


def _candidate_failure_removals(
    control: Mapping[str, Any], scope: str
) -> tuple[str, ...]:
    _base, _roots, snapshot_name = _scope_layout(control, scope)
    return (
        f"snapshot/{snapshot_name}",
        "rounds/round-1",
        "registration_pending.json",
        "change_set.json",
        "candidate_binding.json",
        "candidate.diff",
        "static_review.json",
        "correctness.json",
        "short_paired_evaluation.json",
        "profiler_stage.json",
        "formal_paired_evaluation.json",
        "candidate_stage_intent.json",
        "candidate_stage_complete.json",
        "evaluation.json",
        "time_gate.json",
        "review.json",
    )


_CANDIDATE_FAILURE_STATE_REMOVALS = (
    "before_identity_digest",
    "change_set_digest",
    "change_scope",
    "candidate_hypothesis_id",
    "candidate_identity_digest",
    "candidate_digest",
    "candidate_started_at_epoch",
    "candidate_pause_authorization_sha256",
    "candidate_stage",
    "candidate_stage_intent_sha256",
    "candidate_stage_intent_stage",
    "candidate_stage_completions",
    "candidate_failure_pending_sha256",
)

_UNBOUND_CANDIDATE_FAILURE = object()


def _candidate_failure_record(
    base_state: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict:
    decision_digest = _canonical_digest(decision)
    elapsed = decision.get("elapsed_seconds")
    if elapsed is None:
        elapsed = _controlled_spend_seconds(base_state)
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValidationError("candidate failure elapsed time is invalid")
    hypothesis_id = _identifier(
        base_state.get("candidate_hypothesis_id"),
        "candidate hypothesis_id",
    )
    identity_digest = _sha256(
        base_state.get("candidate_identity_digest"),
        "candidate identity_digest",
    )
    candidate_digest = _sha256(
        base_state.get("candidate_digest"),
        "candidate digest",
    )
    return {
        "hypothesis_id": hypothesis_id,
        "action_id": f"implement-{hypothesis_id}",
        "implementation_status": "failed",
        "identity_digest": identity_digest,
        "elapsed_seconds": float(elapsed),
        "candidate_digest": candidate_digest,
        "decision_digest": decision_digest,
        "failure_reason": _identifier(
            decision.get("reason"),
            "candidate failure reason",
        ),
    }


def _candidate_failure_target_state(
    base_state: Mapping[str, Any],
    *,
    decision_sha256: str,
    context_sha256: str,
    ledger_event: Mapping[str, Any],
) -> dict:
    updated = copy.deepcopy(dict(base_state))
    completed = updated.get("completed_stages")
    if type(completed) is not list:
        raise ValidationError("candidate failure base completed stages are invalid")
    updated["completed_stages"] = [
        item
        for item in completed
        if item not in {"change", "review", "evaluation", "decision"}
    ]
    for field in _CANDIDATE_FAILURE_STATE_REMOVALS:
        updated.pop(field, None)
    updated.update(
        {
            "status": "active",
            "stage": "active_diagnosis",
            "next_action": "propose_hypotheses",
            "updated_at_epoch": ledger_event["created_at_epoch"],
            "decision_digest": decision_sha256,
            "diagnosis_context_sha256": context_sha256,
            "active_diagnosis_round": int(
                base_state.get("active_diagnosis_round", 1)
            )
            + 1,
            "active_diagnosis_ledger_sequence": ledger_event["sequence"],
            "active_diagnosis_ledger_head_sha256": _canonical_digest(
                ledger_event
            ),
        }
    )
    return updated


def _candidate_failure_bound_state(
    base_state: Mapping[str, Any],
    pending_sha256: str,
) -> dict:
    if base_state.get("candidate_failure_pending_sha256") is not None:
        raise ValidationError("candidate failure base state is already bound")
    bound = copy.deepcopy(dict(base_state))
    bound["candidate_failure_pending_sha256"] = _sha256(
        pending_sha256,
        "candidate failure pending digest",
    )
    return bound


def _candidate_rollback_identity(
    control: Mapping[str, Any],
    run_root: Path,
    state: Mapping[str, Any],
    *,
    scope: str,
    before_identity_digest: str,
) -> str:
    binding_path = run_root / "candidate_binding.json"
    if not binding_path.exists() and not binding_path.is_symlink():
        return "unbound"
    if binding_path.is_symlink() or not binding_path.is_file():
        raise ValidationError("candidate rollback binding is invalid")
    change = _load_registered_change_set(run_root, state, control)
    binding = _validate_candidate_binding(
        load_json_object(binding_path),
        candidate=change["candidate"],
        change_set_sha256=_canonical_digest(change),
    )
    _validated_identity_artifact(
        load_json_object(
            run_root / "rounds" / "round-1" / "after_identity.json"
        ),
        binding["after_identity_digest"],
    )
    current_identity = _identity(control, scope)["digest"]
    if current_identity == binding["after_identity_digest"]:
        return "candidate"
    if current_identity == before_identity_digest:
        return "before"
    raise ValidationError("candidate rollback identity drifted")


def _validate_candidate_failure_pending(
    run_root: Path,
    value: Mapping[str, Any],
    *,
    current_state: Mapping[str, Any],
) -> dict:
    pending = _object(value, "candidate failure pending")
    fields = {
        "schema_version",
        "base_state_sha256",
        "scope",
        "before_identity_digest",
        "decision",
        "decision_sha256",
        "context",
        "context_sha256",
        "ledger_path",
        "ledger_event",
        "target_state",
    }
    _closed(pending, fields, "candidate failure pending")
    _required(pending, fields, "candidate failure pending")
    if (
        pending["schema_version"]
        != "cuda-workload-optimizer/candidate-failure-pending-v1"
    ):
        raise ValidationError("candidate failure pending schema is invalid")
    base_digest = _sha256(
        pending["base_state_sha256"],
        "candidate failure pending base state",
    )
    base_state = load_json_object(
        run_root / "state_generations" / f"{base_digest}.json"
    )
    if _canonical_digest(base_state) != base_digest:
        raise ValidationError("candidate failure base state drifted")
    scope = pending["scope"]
    if (
        scope not in {"project", "isolated_environment"}
        or scope != base_state.get("change_scope")
    ):
        raise ValidationError("candidate failure scope drifted")
    before_identity = _sha256(
        pending["before_identity_digest"],
        "candidate failure before identity",
    )
    if before_identity != base_state.get("before_identity_digest"):
        raise ValidationError("candidate failure snapshot binding drifted")
    decision = _object(pending["decision"], "candidate failure decision")
    if (
        _canonical_digest(decision)
        != _sha256(
            pending["decision_sha256"],
            "candidate failure decision digest",
        )
        or decision.get("status") != "rejected"
        or decision.get("rolled_back") is not True
    ):
        raise ValidationError("candidate failure decision drifted")
    context = _object(pending["context"], "candidate failure context")
    context_digest = _sha256(
        pending["context_sha256"],
        "candidate failure context digest",
    )
    if _canonical_digest(context) != context_digest:
        raise ValidationError("candidate failure context drifted")
    ledger_event = _object(
        pending["ledger_event"], "candidate failure ledger event"
    )
    ledger_fields = {
        "schema_version",
        "sequence",
        "event_type",
        "previous_event_sha256",
        "payload_sha256",
        "created_at_epoch",
    }
    _closed(ledger_event, ledger_fields, "candidate failure ledger event")
    _required(ledger_event, ledger_fields, "candidate failure ledger event")
    expected_sequence = int(base_state["active_diagnosis_ledger_sequence"]) + 1
    expected_ledger_path = (
        Path("active_diagnosis")
        / "ledger"
        / f"{expected_sequence:06d}-candidate.json"
    ).as_posix()
    if (
        ledger_event.get("schema_version")
        != "cuda-optimizer/active-diagnosis-event-v1"
        or ledger_event.get("sequence") != expected_sequence
        or ledger_event.get("event_type") != "candidate"
        or ledger_event.get("previous_event_sha256")
        != base_state.get("active_diagnosis_ledger_head_sha256")
        or pending["ledger_path"] != expected_ledger_path
    ):
        raise ValidationError("candidate failure ledger binding drifted")
    history = context.get("candidate_history")
    if type(history) is not list or not history:
        raise ValidationError("candidate failure history is missing")
    base_context = copy.deepcopy(context)
    base_history = list(base_context["candidate_history"])
    record = base_history.pop()
    base_context["candidate_history"] = base_history
    active_root = run_root / "active_diagnosis"
    base_context["knowledge_context"] = _rebuild_knowledge_context(
        run_root,
        base_context,
        _load_frozen_analysis_contract(run_root, base_state),
        load_json_object(active_root / "epoch.json"),
        load_json_object(active_root / "execution_map.json"),
        load_json_object(active_root / "evidence_catalog.json"),
        load_json_object(active_root / "selection_policy.json"),
        load_json_object(active_root / "performance_model.json"),
    )
    if _canonical_digest(base_context) != base_state.get(
        "diagnosis_context_sha256"
    ):
        raise ValidationError("candidate failure context prefix drifted")
    expected_record = _candidate_failure_record(base_state, decision)
    if record != expected_record:
        raise ValidationError("candidate failure history record drifted")
    expected_payload = {
        "candidate_history_record": expected_record,
        "context_sha256": context_digest,
    }
    if ledger_event.get("payload_sha256") != _canonical_digest(expected_payload):
        raise ValidationError("candidate failure ledger payload drifted")
    target_state = _object(
        pending["target_state"], "candidate failure target state"
    )
    expected_target = _candidate_failure_target_state(
        base_state,
        decision_sha256=pending["decision_sha256"],
        context_sha256=context_digest,
        ledger_event=ledger_event,
    )
    if target_state != expected_target:
        raise ValidationError("candidate failure target state drifted")
    current_digest = _canonical_digest(current_state)
    target_digest = _canonical_digest(expected_target)
    pending_digest = _canonical_digest(pending)
    state_pending_digest = current_state.get(
        "candidate_failure_pending_sha256"
    )
    if (
        state_pending_digest is not None
        and _sha256(
            state_pending_digest,
            "state candidate failure pending digest",
        )
        != pending_digest
    ):
        raise ValidationError("candidate failure pending state binding drifted")
    events = _verify_active_diagnosis_ledger(run_root)
    _verify_committed_active_ledger(base_state, events)
    base_sequence = int(base_state["active_diagnosis_ledger_sequence"])
    if len(events) not in {base_sequence, base_sequence + 1}:
        raise ValidationError("candidate failure ledger tail drifted")
    if len(events) == base_sequence + 1 and events[-1] != ledger_event:
        raise ValidationError("candidate failure ledger event drifted")
    context_path = run_root / "diagnosis_context.json"
    if context_path.is_symlink() or not context_path.is_file():
        raise ValidationError("candidate failure context artifact is invalid")
    current_context = load_json_object(context_path)
    if current_digest == target_digest:
        if current_context != context:
            raise ValidationError("candidate failure committed context drifted")
    elif current_context != base_context and current_context != context:
        raise ValidationError("candidate failure context artifact drifted")
    return copy.deepcopy(dict(pending))


def _apply_candidate_failure_pending(
    run_root: Path,
    control: Mapping[str, Any],
    pending: Mapping[str, Any],
    *,
    bind_unbound: bool,
) -> dict:
    state = read_run_state(run_root)
    plan = _validate_candidate_failure_pending(
        run_root,
        pending,
        current_state=state,
    )
    base_digest = plan["base_state_sha256"]
    base_state = load_json_object(
        run_root / "state_generations" / f"{base_digest}.json"
    )
    pending_digest = _canonical_digest(plan)
    bound_state = _candidate_failure_bound_state(
        base_state,
        pending_digest,
    )
    bound_digest = _canonical_digest(bound_state)
    target_state = plan["target_state"]
    target_digest = _canonical_digest(target_state)
    current_digest = _canonical_digest(state)
    if current_digest not in {base_digest, bound_digest, target_digest}:
        raise ValidationError("candidate failure state binding drifted")
    if current_digest == base_digest:
        if not bind_unbound:
            raise ValidationError("candidate failure pending is not state-bound")
        state = _write_state(run_root, bound_state)
        current_digest = _canonical_digest(state)
    if current_digest == bound_digest:
        rollback_identity = _candidate_rollback_identity(
            control,
            run_root,
            base_state,
            scope=plan["scope"],
            before_identity_digest=plan["before_identity_digest"],
        )
        if rollback_identity != "before":
            _restore_snapshot(
                control,
                run_root,
                plan["scope"],
                plan["before_identity_digest"],
            )
        for path, value in (
            (run_root / "decision.json", plan["decision"]),
            (run_root / "diagnosis_context.json", plan["context"]),
        ):
            if (
                path.is_symlink()
                or not path.is_file()
                or load_json_object(path) != value
            ):
                _atomic_json(path, value)
        ledger_path = run_root / plan["ledger_path"]
        if ledger_path.exists():
            if (
                ledger_path.is_symlink()
                or load_json_object(ledger_path) != plan["ledger_event"]
            ):
                raise ValidationError("candidate failure ledger event drifted")
        else:
            _atomic_json(ledger_path, plan["ledger_event"])
        events = _verify_active_diagnosis_ledger(run_root)
        if events[plan["ledger_event"]["sequence"] - 1] != plan["ledger_event"]:
            raise ValidationError("candidate failure ledger event is not committed")
        _write_state(run_root, target_state)
    else:
        if (
            load_json_object(run_root / "decision.json") != plan["decision"]
            or load_json_object(run_root / "diagnosis_context.json")
            != plan["context"]
        ):
            raise ValidationError("candidate failure committed artifact drifted")
    mirror_path = run_root / "active_diagnosis" / "knowledge_context.json"
    if mirror_path.is_symlink() or (
        mirror_path.exists() and not mirror_path.is_file()
    ):
        raise ValidationError("candidate failure knowledge mirror is unsafe")
    mirror = None
    if mirror_path.is_file():
        try:
            mirror = load_json_object(mirror_path)
        except ValidationError:
            mirror = None
    if mirror != plan["context"]["knowledge_context"]:
        _atomic_json(
            mirror_path,
            plan["context"]["knowledge_context"],
        )
    for relative in _candidate_failure_removals(control, plan["scope"]):
        path = run_root / relative
        if path.exists() or path.is_symlink():
            _remove_path(path)
    pending_path = run_root / "candidate_failure_pending.json"
    try:
        pending_path.unlink()
    except FileNotFoundError:
        pass
    return read_run_state(run_root)


def _commit_candidate_failure(
    run_root: Path,
    control: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> dict:
    state = read_run_state(run_root)
    plan = _validate_candidate_failure_pending(
        run_root,
        pending,
        current_state=state,
    )
    pending_path = run_root / "candidate_failure_pending.json"
    if pending_path.exists():
        if (
            pending_path.is_symlink()
            or load_json_object(pending_path) != plan
        ):
            raise ValidationError("a different candidate failure is pending")
    else:
        _atomic_json(pending_path, plan)
    return _apply_candidate_failure_pending(
        run_root,
        control,
        plan,
        bind_unbound=True,
    )


def _recover_candidate_failure(
    run_root: Path,
    control: Mapping[str, Any],
    *,
    discard_unbound: bool,
) -> dict | object | None:
    pending_path = run_root / "candidate_failure_pending.json"
    if not pending_path.exists() and not pending_path.is_symlink():
        return None
    if pending_path.is_symlink() or not pending_path.is_file():
        raise ValidationError("candidate failure pending must be a regular file")
    pending = load_json_object(pending_path)
    state = read_run_state(run_root)
    if state.get("candidate_failure_pending_sha256") is None:
        base_digest = _sha256(
            pending.get("base_state_sha256"),
            "candidate failure pending base state",
        )
        if _canonical_digest(state) == base_digest:
            if discard_unbound:
                pending_path.unlink()
            return _UNBOUND_CANDIDATE_FAILURE
    return _apply_candidate_failure_pending(
        run_root,
        control,
        pending,
        bind_unbound=False,
    )


def _resume_active_diagnosis_after_candidate_rejection(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    scope: str,
    reason: str,
    time_gate: Mapping[str, Any] | None,
) -> dict:
    context = load_json_object(run_root / "diagnosis_context.json")
    if _canonical_digest(context) != state.get("diagnosis_context_sha256"):
        raise ValidationError("diagnosis context digest drifted before candidate failure")
    if decision.get("reason") != reason:
        raise ValidationError("candidate failure reason drifted before commit")
    record = _candidate_failure_record(state, decision)
    transition_at = time.time()
    refreshed = copy.deepcopy(context)
    refreshed.setdefault("candidate_history", []).append(record)
    active_root = run_root / "active_diagnosis"
    contract = _load_frozen_analysis_contract(run_root, state)
    epoch = load_json_object(active_root / "epoch.json")
    execution_map = load_json_object(active_root / "execution_map.json")
    evidence_catalog = load_json_object(active_root / "evidence_catalog.json")
    selection_policy = load_json_object(active_root / "selection_policy.json")
    performance_model = load_json_object(active_root / "performance_model.json")
    refreshed["knowledge_context"] = _rebuild_knowledge_context(
        run_root,
        refreshed,
        contract,
        epoch,
        execution_map,
        evidence_catalog,
        selection_policy,
        performance_model,
    )
    context_digest = _canonical_digest(refreshed)
    ledger_path, ledger_event = _prepare_active_diagnosis_event(
        run_root,
        "candidate",
        {
            "candidate_history_record": record,
            "context_sha256": context_digest,
        },
        created_at_epoch=transition_at,
    )
    decision_digest = _canonical_digest(decision)
    updated = _candidate_failure_target_state(
        state,
        decision_sha256=decision_digest,
        context_sha256=context_digest,
        ledger_event=ledger_event,
    )
    pending = {
        "schema_version": (
            "cuda-workload-optimizer/candidate-failure-pending-v1"
        ),
        "base_state_sha256": _canonical_digest(state),
        "scope": scope,
        "before_identity_digest": state["before_identity_digest"],
        "decision": copy.deepcopy(dict(decision)),
        "decision_sha256": decision_digest,
        "context": refreshed,
        "context_sha256": context_digest,
        "ledger_path": ledger_path,
        "ledger_event": ledger_event,
        "target_state": updated,
    }
    _commit_candidate_failure(run_root, control, pending)
    return copy.deepcopy(dict(decision))

def _finish_rejected(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    scope: str,
    reason: str,
    primary_status: str | None,
    time_gate: Mapping[str, Any] | None = None,
) -> dict:
    intrinsic_failure_reasons = {
        "candidate_below_project_contract",
        "no_scoped_changes",
        "static_falsified",
        "correctness_failed",
        "short_pair_failed",
        "effect_upper_bound_below_minimum",
        "formal_pair_failed",
        "primary_not_confirmed",
        "constraint_failed",
    }
    intrinsic_failure = reason in intrinsic_failure_reasons
    if intrinsic_failure and reason not in {
        "candidate_below_project_contract",
        "no_scoped_changes",
    }:
        binding_path = run_root / "candidate_binding.json"
        if not binding_path.is_file() or binding_path.is_symlink():
            intrinsic_failure = False
        else:
            binding = load_json_object(binding_path)
            intrinsic_failure = (
                binding.get("candidate_digest") == state.get("candidate_digest")
                and binding.get("change_set_digest") == state.get("change_set_digest")
                and _identity(control, scope)["digest"]
                == binding.get("after_identity_digest")
            )
    active_candidate = (
        "analysis_contract" in control and state.get("candidate_hypothesis_id")
    )
    if active_candidate and intrinsic_failure:
        decision = {
            "schema_version": "cuda-workload-optimizer/decision-v1",
            "status": "rejected",
            "reason": reason,
            "primary_status": primary_status,
            "rolled_back": True,
        }
        if time_gate is not None:
            decision.update(
                {
                    "elapsed_seconds": time_gate["elapsed_seconds"],
                    "stop_reason": time_gate["stop_reason"],
                    "skipped_expensive_stages": time_gate[
                        "skipped_expensive_stages"
                    ],
                }
            )
        return _resume_active_diagnosis_after_candidate_rejection(
            run_root,
            state,
            control,
            decision,
            scope=scope,
            reason=reason,
            time_gate=time_gate,
        )
    try:
        _restore_snapshot(
            control,
            run_root,
            scope,
            state["before_identity_digest"],
        )
    except (OSError, ValidationError) as error:
        decision = {
            "schema_version": "cuda-workload-optimizer/decision-v1",
            "status": "manual_recovery_required",
            "reason": "rollback_failed",
            "rejected_reason": reason,
            "primary_status": primary_status,
            "rolled_back": False,
            "error": f"{type(error).__name__}: {error}",
            "snapshot": str(run_root / "snapshot" / ("project" if scope == "project" else "environment")),
        }
        _atomic_json(run_root / "decision.json", decision)
        updated = copy.deepcopy(state)
        updated.update(
            {
                "status": "manual_recovery_required",
                "stage": "decision",
                "next_action": "manual_recovery",
                "updated_at_epoch": time.time(),
                "decision_digest": _canonical_digest(decision),
            }
        )
        _write_state(run_root, updated)
        return decision
    if active_candidate and not intrinsic_failure:
        decision = {
            "schema_version": "cuda-workload-optimizer/decision-v1",
            "status": "review_required",
            "reason": reason,
            "primary_status": primary_status,
            "rolled_back": True,
            "refresh_required": True,
        }
        if time_gate is not None:
            decision.update(
                {
                    "elapsed_seconds": time_gate["elapsed_seconds"],
                    "stop_reason": time_gate["stop_reason"],
                    "skipped_expensive_stages": time_gate[
                        "skipped_expensive_stages"
                    ],
                }
            )
        _atomic_json(run_root / "decision.json", decision)
        updated = copy.deepcopy(state)
        updated.update(
            {
                "status": "active",
                "stage": "decision",
                "next_action": "refresh_required",
                "updated_at_epoch": time.time(),
                "decision_digest": _canonical_digest(decision),
                "terminal_reason": reason,
            }
        )
        _write_state(run_root, updated)
        return decision
    decision = {
        "schema_version": "cuda-workload-optimizer/decision-v1",
        "status": "rejected",
        "reason": reason,
        "primary_status": primary_status,
        "rolled_back": True,
    }
    if time_gate is not None:
        decision.update(
            {
                "elapsed_seconds": time_gate["elapsed_seconds"],
                "stop_reason": time_gate["stop_reason"],
                "skipped_expensive_stages": time_gate[
                    "skipped_expensive_stages"
                ],
            }
        )
    _atomic_json(run_root / "decision.json", decision)
    if active_candidate:
        _resume_active_diagnosis_after_candidate_rejection(
            run_root,
            state,
            control,
            decision,
            scope=scope,
            reason=reason,
            time_gate=time_gate,
        )
        return decision
    updated = copy.deepcopy(state)
    for stage in ("review", "evaluation", "decision"):
        if stage not in updated["completed_stages"]:
            updated["completed_stages"].append(stage)
    updated.update(
        {
            "status": "completed",
            "stage": "decision",
            "next_action": "done",
            "updated_at_epoch": time.time(),
            "decision_digest": _canonical_digest(decision),
        }
    )
    _write_state(run_root, updated)
    return decision


def _finish_review_required(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    scope: str,
    primary_status: str | None,
    time_gate: Mapping[str, Any],
) -> dict:
    """Pause one candidate in place until a covering grant is bound."""
    review_reason = time_gate.get("stop_reason")
    if not isinstance(review_reason, str) or not review_reason.strip():
        raise ValidationError("review-required time gate stop reason is invalid")
    next_stage = time_gate.get("next_stage")
    if next_stage not in _CANDIDATE_STAGE_ARTIFACTS:
        raise ValidationError("review-required candidate stage is invalid")
    evidence = {}
    for name in (
        "candidate_binding.json",
        "static_review.json",
        "correctness.json",
        "short_paired_evaluation.json",
        "profiler_stage.json",
        "formal_paired_evaluation.json",
    ):
        path = run_root / name
        if path.is_file():
            evidence[name] = _sha256_path(path)
    decision = {
        "schema_version": "cuda-workload-optimizer/decision-v1",
        "status": "review_required",
        "reason": review_reason,
        "primary_status": primary_status,
        "rolled_back": False,
        "candidate_diff_sha256": _sha256_path(run_root / "candidate.diff"),
        "time_gate_sha256": _canonical_digest(time_gate),
        "next_stage": next_stage,
        "blocked_action": copy.deepcopy(time_gate["blocked_action"]),
        "projected_spend": copy.deepcopy(time_gate["projected_spend"]),
        "elapsed_seconds": time_gate["elapsed_seconds"],
        "stop_reason": review_reason,
        "skipped_expensive_stages": copy.deepcopy(
            time_gate["skipped_expensive_stages"]
        ),
        "evidence": evidence,
    }
    _atomic_json(run_root / "decision.json", decision)
    paused = copy.deepcopy(dict(state))
    paused.update(
        {
            "status": "active",
            "stage": "candidate_validation",
            "next_action": "review_required",
            "candidate_stage": next_stage,
            "candidate_pause_authorization_sha256": (
                _candidate_authorization_digest(state)
            ),
            "updated_at_epoch": time.time(),
            "decision_digest": _canonical_digest(decision),
            "terminal_reason": review_reason,
        }
    )
    _write_state(run_root, paused)
    return decision



def _validated_identity_artifact(value: Mapping[str, Any], expected_digest: str) -> dict:
    identity = _object(value, "identity artifact")
    fields = {"schema_version", "scope", "roots", "missing_roots", "files", "digest"}
    _closed(identity, fields, "identity artifact")
    _required(identity, fields, "identity artifact")
    if identity["schema_version"] != "cuda-workload-optimizer/project-identity-v1":
        raise ValidationError("identity artifact schema is invalid")
    computed = _canonical_digest(
        {
            "missing_roots": identity["missing_roots"],
            "files": identity["files"],
        }
    )
    if identity["digest"] != computed or computed != expected_digest:
        raise ValidationError("frozen identity artifact digest does not match state")
    return copy.deepcopy(identity)


def _explicit_profiler_uncertainty(
    diagnosis: Mapping[str, Any], investment_brief: Mapping[str, Any] | None = None
) -> set[str]:
    """Return only live uncertainties with a directly matching profiler kind."""
    supported = {"timeline", "framework", "custom"}
    uncertainty = {
        item
        for item in diagnosis.get("suggested_probes", [])
        if item in supported
    }
    if investment_brief is not None:
        uncertainty.update(
            item
            for item in investment_brief.get("uncertainty", [])
            if item in supported
        )
    return uncertainty


_CANDIDATE_STAGE_ARTIFACTS = {
    "static_review": "static_review.json",
    "build_correctness": "correctness.json",
    "short_paired": "short_paired_evaluation.json",
    "profiler": "profiler_stage.json",
    "formal_paired": "formal_paired_evaluation.json",
}
_CANDIDATE_STAGE_STATE_FIELDS = {
    "candidate_stage",
    "candidate_stage_intent_sha256",
    "candidate_stage_intent_stage",
    "candidate_stage_completions",
}


def _load_registered_change_set(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict:
    path = run_root / "rounds" / "round-1" / "change_set.json"
    if path.is_symlink() or not path.is_file():
        raise ValidationError("registered ChangeSet must be a regular file")
    frozen = validate_change_set(load_json_object(path), control)
    if (
        _canonical_digest(frozen) != state.get("change_set_digest")
        or frozen["scope"] != state.get("change_scope")
    ):
        raise ValidationError("registered ChangeSet artifact drifted")
    return frozen


def _validate_candidate_binding(
    value: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    change_set_sha256: str,
) -> dict:
    binding = _object(value, "candidate binding")
    fields = {
        "schema_version",
        "candidate",
        "candidate_digest",
        "change_set_digest",
        "after_identity_digest",
        "digest",
    }
    _closed(binding, fields, "candidate binding")
    _required(binding, fields, "candidate binding")
    if (
        binding["schema_version"]
        != "cuda-workload-optimizer/candidate-binding-v1"
    ):
        raise ValidationError("candidate binding schema_version is unsupported")
    if binding["candidate"] != candidate:
        raise ValidationError("candidate binding candidate drifted")
    if binding["candidate_digest"] != _canonical_digest(candidate):
        raise ValidationError("candidate binding candidate digest drifted")
    if binding["change_set_digest"] != change_set_sha256:
        raise ValidationError("candidate binding ChangeSet drifted")
    _sha256(binding["after_identity_digest"], "candidate binding identity")
    without_digest = dict(binding)
    digest = without_digest.pop("digest")
    if digest != _canonical_digest(without_digest):
        raise ValidationError("candidate binding digest drifted")
    return copy.deepcopy(binding)


def _candidate_stage_admission_view(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    applicable_stages: Sequence[str],
    candidate_identity_sha256: str,
    *,
    allow_active_scope_identity_drift: bool = False,
) -> tuple[dict, str]:
    """Build the closed, non-persisted Gate view from the authoritative ceiling."""
    stages = list(applicable_stages)
    if "analysis_contract" in control:
        grant = _load_bound_authorization_grant(
            run_root,
            state,
            control,
            active_scope_identity_digest=candidate_identity_sha256,
            allow_active_scope_identity_drift=(
                allow_active_scope_identity_drift
            ),
        )
        authorization_sha256 = _sha256(
            state.get("authorization_grant_sha256"),
            "state authorization_grant_sha256",
        )
        maximum = float(grant["max_controlled_seconds"])
        max_stage = grant["max_stage"]
        change = _load_registered_change_set(run_root, state, control)
        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        if (
            change["scope"] not in grant["allowed_mutation_scopes"]
            or risk_order[change["risk"]] > risk_order[grant["max_risk"]]
        ):
            max_stage = "diagnosis"
    else:
        runtime = _BUDGET_RUNTIME[control["budget"]]
        maximum = float(runtime["hard_ceiling_seconds"])
        max_stage = "formal_paired"
        authorization_sha256 = _sha256(
            state.get("control_digest"),
            "state control_digest",
        )
    return (
        {
            "max_controlled_seconds": maximum,
            "max_stage": max_stage,
            "applicable_stages": stages,
        },
        authorization_sha256,
    )


def _candidate_stage_marker(
    run_root: Path, name: str
) -> Path:
    return run_root / name


def _load_candidate_stage_marker(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} must be a regular file")
    return load_json_object(path)


def _validate_candidate_stage_intent(
    run_root: Path,
    value: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
) -> dict:
    intent = _object(value, "candidate stage intent")
    fields = {
        "schema_version",
        "stage",
        "base_state_sha256",
        "authorization",
        "authorization_sha256",
        "authorization_view_sha256",
        "change_set_sha256",
        "candidate_digest",
        "candidate_identity_sha256",
        "candidate_binding_sha256",
        "stage_artifact_path",
        "created_at_epoch",
    }
    _closed(intent, fields, "candidate stage intent")
    _required(intent, fields, "candidate stage intent")
    if (
        intent["schema_version"]
        != "cuda-workload-optimizer/candidate-stage-intent-v1"
    ):
        raise ValidationError("candidate stage intent schema_version is unsupported")
    stage = intent["stage"]
    if stage not in _CANDIDATE_STAGE_ARTIFACTS:
        raise ValidationError("candidate stage intent stage is invalid")
    if state.get("candidate_stage") != stage:
        raise ValidationError("candidate stage intent does not match state stage")
    if intent["stage_artifact_path"] != _CANDIDATE_STAGE_ARTIFACTS[stage]:
        raise ValidationError("candidate stage intent artifact path drifted")
    authorization = _object(
        intent["authorization"], "candidate stage intent authorization"
    )
    if set(authorization) != {
        "max_controlled_seconds",
        "max_stage",
        "applicable_stages",
    }:
        raise ValidationError("candidate stage intent authorization fields are invalid")
    if _canonical_digest(authorization) != _sha256(
        intent["authorization_view_sha256"],
        "candidate stage intent authorization view",
    ):
        raise ValidationError("candidate stage intent authorization view drifted")
    for field in (
        "base_state_sha256",
        "authorization_sha256",
        "change_set_sha256",
        "candidate_digest",
        "candidate_identity_sha256",
        "candidate_binding_sha256",
    ):
        _sha256(intent[field], f"candidate stage intent {field}")
    try:
        base_state = load_json_object(
            run_root
            / "state_generations"
            / f"{intent['base_state_sha256']}.json"
        )
    except (OSError, ValidationError) as error:
        raise ValidationError(
            "candidate stage intent base state is unavailable"
        ) from error
    if _canonical_digest(base_state) != intent["base_state_sha256"]:
        raise ValidationError("candidate stage intent base state drifted")
    current = copy.deepcopy(dict(state))
    if current.get("candidate_stage_intent_sha256") is None:
        if _canonical_digest(current) != intent["base_state_sha256"]:
            raise ValidationError("candidate stage intent base state is not current")
    else:
        current.pop("candidate_stage_intent_sha256", None)
        current.pop("candidate_stage_intent_stage", None)
        current["updated_at_epoch"] = base_state.get("updated_at_epoch")
        if current != base_state:
            raise ValidationError("candidate stage intent state binding drifted")
    if intent["change_set_sha256"] != state.get("change_set_digest"):
        raise ValidationError("candidate stage intent ChangeSet drifted")
    if intent["candidate_digest"] != candidate_binding.get("candidate_digest"):
        raise ValidationError("candidate stage intent candidate drifted")
    if (
        intent["candidate_identity_sha256"]
        != candidate_binding.get("after_identity_digest")
    ):
        raise ValidationError("candidate stage intent identity drifted")
    if intent["candidate_binding_sha256"] != candidate_binding.get("digest"):
        raise ValidationError("candidate stage intent binding drifted")
    created = intent["created_at_epoch"]
    if type(created) not in {int, float} or not math.isfinite(float(created)):
        raise ValidationError("candidate stage intent time is invalid")
    return copy.deepcopy(intent)


def _validate_candidate_stage_completion(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    intent: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict:
    completion = _object(value, "candidate stage completion")
    fields = {
        "schema_version",
        "stage",
        "intent_sha256",
        "authorization_sha256",
        "authorization_view_sha256",
        "change_set_sha256",
        "candidate_digest",
        "candidate_identity_sha256",
        "candidate_binding_sha256",
        "stage_artifact_path",
        "stage_artifact_sha256",
        "result",
        "duration_seconds",
        "completed_at_epoch",
    }
    _closed(completion, fields, "candidate stage completion")
    _required(completion, fields, "candidate stage completion")
    if (
        completion["schema_version"]
        != "cuda-workload-optimizer/candidate-stage-completion-v1"
    ):
        raise ValidationError(
            "candidate stage completion schema_version is unsupported"
        )
    if completion["stage"] != intent["stage"]:
        raise ValidationError("candidate stage completion stage drifted")
    if completion["intent_sha256"] != _canonical_digest(intent):
        raise ValidationError("candidate stage completion intent drifted")
    for field in (
        "authorization_sha256",
        "authorization_view_sha256",
        "change_set_sha256",
        "candidate_digest",
        "candidate_identity_sha256",
        "candidate_binding_sha256",
    ):
        if completion[field] != intent[field]:
            raise ValidationError(f"candidate stage completion {field} drifted")
    result = _object(
        completion["result"], "candidate stage completion result"
    )
    identity_drift_completion = (
        result.get("reason") == "candidate_identity_drift"
    )
    admission, authorization_sha256 = _candidate_stage_admission_view(
        run_root,
        state,
        control,
        intent["authorization"]["applicable_stages"],
        intent["candidate_identity_sha256"],
        allow_active_scope_identity_drift=identity_drift_completion,
    )
    if (
        authorization_sha256 != intent["authorization_sha256"]
        or _canonical_digest(admission) != intent["authorization_view_sha256"]
    ):
        raise ValidationError("candidate stage completion authorization drifted")
    artifact_path = intent["stage_artifact_path"]
    if completion["stage_artifact_path"] != artifact_path:
        raise ValidationError("candidate stage completion artifact path drifted")
    artifact = run_root / artifact_path
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or completion["stage_artifact_sha256"] != _sha256_path(artifact)
    ):
        raise ValidationError("candidate stage completion artifact drifted")
    duration = completion["duration_seconds"]
    if (
        type(duration) not in {int, float}
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
    ):
        raise ValidationError("candidate stage completion duration is invalid")
    if (
        _controlled_spend_seconds(state) + float(duration)
        > float(intent["authorization"]["max_controlled_seconds"])
    ):
        raise ValidationError(
            "candidate stage completion exceeds controlled authorization"
        )
    completed_at = completion["completed_at_epoch"]
    if (
        type(completed_at) not in {int, float}
        or not math.isfinite(float(completed_at))
    ):
        raise ValidationError("candidate stage completion time is invalid")
    current_identity = _identity(control, state["change_scope"])["digest"]
    identity_drifted = (
        current_identity != intent["candidate_identity_sha256"]
    )
    if identity_drifted and not identity_drift_completion:
        raise ValidationError("candidate identity drifted after stage execution")
    if identity_drift_completion and not identity_drifted:
        raise ValidationError(
            "candidate stage completion claims identity drift without drift"
        )
    return copy.deepcopy(completion)


def _candidate_stage_heartbeat(stage: str, completion_sha256: str) -> None:
    print(
        json.dumps(
            {
                "event": "heartbeat",
                "checkpoint": "candidate_stage_committed",
                "stage": stage,
                "completion_sha256": completion_sha256,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _cleanup_candidate_stage_markers(run_root: Path) -> None:
    for name in (
        "candidate_stage_complete.json",
        "candidate_stage_intent.json",
    ):
        try:
            _candidate_stage_marker(run_root, name).unlink()
        except FileNotFoundError:
            pass


def _commit_candidate_stage_completion(
    run_root: Path,
    state: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict:
    stage = completion["stage"]
    completion_sha256 = _canonical_digest(completion)
    completions = dict(state.get("candidate_stage_completions", {}))
    if stage in completions:
        if _canonical_digest(completions[stage]) != completion_sha256:
            raise ValidationError("candidate stage completion digest conflicts with state")
        return copy.deepcopy(dict(state))
    updated = copy.deepcopy(dict(state))
    completions[stage] = copy.deepcopy(dict(completion))
    updated["candidate_stage_completions"] = completions
    updated["controlled_spend_seconds"] = (
        _controlled_spend_seconds(state)
        + float(completion["duration_seconds"])
    )
    if not math.isfinite(updated["controlled_spend_seconds"]):
        raise ValidationError("candidate controlled spend overflowed")
    updated.pop("candidate_stage_intent_sha256", None)
    updated.pop("candidate_stage_intent_stage", None)
    updated["updated_at_epoch"] = time.time()
    committed = _write_state(run_root, updated)
    _candidate_stage_heartbeat(stage, completion_sha256)
    return committed


def _validated_candidate_stage_results(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
) -> dict:
    completions = state.get("candidate_stage_completions", {})
    if not isinstance(completions, Mapping):
        raise ValidationError("candidate stage state is invalid")
    validated = {}
    for stage, raw_completion in completions.items():
        if stage not in _CANDIDATE_STAGE_ARTIFACTS:
            raise ValidationError("candidate stage state contains an invalid stage")
        completion = _object(raw_completion, "candidate stage state completion")
        if completion.get("change_set_sha256") != state.get("change_set_digest"):
            raise ValidationError("candidate stage state ChangeSet drifted")
        if completion.get("candidate_digest") != candidate_binding.get(
            "candidate_digest"
        ):
            raise ValidationError("candidate stage state candidate drifted")
        if completion.get("candidate_identity_sha256") != candidate_binding.get(
            "after_identity_digest"
        ):
            raise ValidationError("candidate stage state identity drifted")
        if completion.get("candidate_binding_sha256") != candidate_binding.get(
            "digest"
        ):
            raise ValidationError("candidate stage state binding drifted")
        artifact = run_root / _CANDIDATE_STAGE_ARTIFACTS[stage]
        if (
            completion.get("stage_artifact_path")
            != _CANDIDATE_STAGE_ARTIFACTS[stage]
            or artifact.is_symlink()
            or not artifact.is_file()
            or completion.get("stage_artifact_sha256") != _sha256_path(artifact)
        ):
            raise ValidationError("candidate stage sealed artifact drifted")
        validated[stage] = copy.deepcopy(
            _object(completion.get("result"), "candidate stage state result")
        )
    return validated


def _recover_candidate_stage_checkpoint(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
) -> dict:
    intent_path = _candidate_stage_marker(
        run_root, "candidate_stage_intent.json"
    )
    complete_path = _candidate_stage_marker(
        run_root, "candidate_stage_complete.json"
    )
    pending_sha256 = state.get("candidate_stage_intent_sha256")
    pending_stage = state.get("candidate_stage_intent_stage")
    if pending_sha256 is None and pending_stage is not None:
        raise ValidationError("candidate stage intent state binding is incomplete")
    if (
        pending_sha256 is None
        and (intent_path.exists() or intent_path.is_symlink())
        and not (complete_path.exists() or complete_path.is_symlink())
    ):
        intent_path.unlink()
        return copy.deepcopy(dict(state))

    raw_intent = None
    if intent_path.exists() or intent_path.is_symlink():
        raw_intent = _load_candidate_stage_marker(
            intent_path, "candidate stage intent"
        )

    raw_completion = None
    if complete_path.exists() or complete_path.is_symlink():
        if raw_intent is None:
            raise ValidationError("candidate stage completion lacks its intent")
        raw_completion = _load_candidate_stage_marker(
            complete_path, "candidate stage completion"
        )

    if pending_sha256 is None:
        completions = state.get("candidate_stage_completions", {})
        if not isinstance(completions, Mapping):
            raise ValidationError("candidate stage consumed state is invalid")
        if raw_completion is not None:
            stage = raw_completion.get("stage")
            if (
                not isinstance(stage, str)
                or stage not in _CANDIDATE_STAGE_ARTIFACTS
                or completions.get(stage) != raw_completion
                or raw_completion.get("intent_sha256")
                != _canonical_digest(raw_intent)
            ):
                raise ValidationError(
                    "unbound candidate stage completion is not consumed"
                )
            _cleanup_candidate_stage_markers(run_root)
            return copy.deepcopy(dict(state))
        if raw_intent is not None:
            intent_sha256 = _canonical_digest(raw_intent)
            if any(
                isinstance(completion, Mapping)
                and completion.get("intent_sha256") == intent_sha256
                for completion in completions.values()
            ):
                intent_path.unlink()
                return copy.deepcopy(dict(state))
            _validate_candidate_stage_intent(
                run_root,
                raw_intent,
                state=state,
                candidate_binding=candidate_binding,
            )
            intent_path.unlink()
        return copy.deepcopy(dict(state))

    intent = None
    if raw_intent is not None:
        intent = _validate_candidate_stage_intent(
            run_root,
            raw_intent,
            state=state,
            candidate_binding=candidate_binding,
        )
    completion = None
    if raw_completion is not None:
        completion = _validate_candidate_stage_completion(
            run_root,
            state,
            control,
            intent,
            raw_completion,
        )

    if pending_sha256 is not None:
        _sha256(pending_sha256, "state candidate stage intent")
        if (
            intent is None
            or _canonical_digest(intent) != pending_sha256
            or pending_stage != intent["stage"]
        ):
            raise ValidationError("state-bound candidate stage intent drifted")
        if completion is None:
            blocked = copy.deepcopy(dict(state))
            blocked.update(
                {
                    "status": "manual_recovery_required",
                    "stage": "candidate_validation",
                    "next_action": "manual_recovery",
                    "manual_recovery_reason": (
                        "candidate_stage_interrupted_not_reexecuted"
                    ),
                    "updated_at_epoch": time.time(),
                }
            )
            return _write_state(run_root, blocked)
        committed = _commit_candidate_stage_completion(
            run_root,
            state,
            completion,
        )
        _cleanup_candidate_stage_markers(run_root)
        return committed


def _persist_candidate_manual_recovery(
    run_root: Path,
    state: Mapping[str, Any],
    *,
    reason: str,
) -> dict:
    blocked = copy.deepcopy(dict(state))
    blocked.update(
        {
            "status": "manual_recovery_required",
            "stage": "candidate_validation",
            "next_action": "manual_recovery",
            "manual_recovery_reason": _identifier(
                reason, "candidate manual recovery reason"
            ),
            "updated_at_epoch": time.time(),
        }
    )
    return _write_state(run_root, blocked)


def _has_candidate_stage_checkpoint(
    run_root: Path,
    state: Mapping[str, Any],
) -> bool:
    return (
        state.get("candidate_stage_intent_sha256") is not None
        or bool(state.get("candidate_stage_completions"))
        or (run_root / "candidate_stage_intent.json").exists()
        or (run_root / "candidate_stage_intent.json").is_symlink()
        or (run_root / "candidate_stage_complete.json").exists()
        or (run_root / "candidate_stage_complete.json").is_symlink()
    )


def _candidate_recovery_preflight(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
) -> tuple[dict, dict | None, bool]:
    try:
        recovered_failure = _recover_candidate_failure(
            run_root,
            control,
            discard_unbound=False,
        )
    except (KeyError, OSError, ValidationError):
        return (
            _persist_candidate_manual_recovery(
                run_root,
                state,
                reason="candidate_failure_recovery_invalid",
            ),
            None,
            False,
        )
    if recovered_failure is _UNBOUND_CANDIDATE_FAILURE:
        return read_run_state(run_root), None, True
    if recovered_failure is not None:
        decision = load_json_object(run_root / "decision.json")
        if (
            _canonical_digest(decision)
            != recovered_failure.get("decision_digest")
            or decision.get("status") != "rejected"
            or decision.get("rolled_back") is not True
        ):
            return (
                _persist_candidate_manual_recovery(
                    run_root,
                    recovered_failure,
                    reason="candidate_failure_recovery_invalid",
                ),
                None,
                False,
            )
        return recovered_failure, decision, False
    current = copy.deepcopy(dict(state))
    if not _has_candidate_stage_checkpoint(run_root, current):
        return current, None, False
    try:
        change = _load_registered_change_set(
            run_root,
            current,
            control,
        )
        binding = _validate_candidate_binding(
            _load_candidate_stage_marker(
                run_root / "candidate_binding.json",
                "candidate binding",
            ),
            candidate=change["candidate"],
            change_set_sha256=_canonical_digest(change),
        )
        _validated_identity_artifact(
            load_json_object(
                run_root / "rounds" / "round-1" / "after_identity.json"
            ),
            binding["after_identity_digest"],
        )
        recovered_stage = _recover_candidate_stage_checkpoint(
            run_root,
            current,
            control,
            binding,
        )
    except (KeyError, OSError, ValidationError):
        return (
            _persist_candidate_manual_recovery(
                run_root,
                current,
                reason="candidate_stage_recovery_invalid",
            ),
            None,
            False,
        )
    return recovered_stage, None, False


def _execute_candidate_stage(
    run_root: Path,
    state: Mapping[str, Any],
    control: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    admission: Mapping[str, Any],
    authorization_sha256: str,
    stage: str,
    runner,
    command_safety_timeout_seconds: float,
) -> tuple[dict, str | None]:
    current_identity = _identity(control, state["change_scope"])["digest"]
    if current_identity != candidate_binding["after_identity_digest"]:
        return (
            _persist_candidate_manual_recovery(
                run_root,
                state,
                reason="candidate_identity_drift",
            ),
            "manual_recovery",
        )
    try:
        _load_registered_change_set(
            run_root,
            state,
            control,
        )
    except ValidationError:
        return copy.deepcopy(dict(state)), "frozen_artifact_drift"
    remaining_authorization = (
        float(admission["max_controlled_seconds"])
        - _controlled_spend_seconds(state)
    )
    if remaining_authorization <= 0.0:
        raise ValidationError(
            "candidate stage lacks remaining controlled authorization"
        )
    if (
        isinstance(command_safety_timeout_seconds, bool)
        or not isinstance(command_safety_timeout_seconds, (int, float))
        or not math.isfinite(float(command_safety_timeout_seconds))
        or float(command_safety_timeout_seconds) <= 0.0
    ):
        raise ValidationError("candidate stage command safety timeout is invalid")
    total_stage_window = min(
        float(command_safety_timeout_seconds),
        remaining_authorization,
    )
    termination_reserve = min(0.5, total_stage_window * 0.6)
    execution_timeout = total_stage_window - termination_reserve
    if execution_timeout <= 0.0:
        raise ValidationError(
            "candidate stage lacks executable controlled authorization"
        )
    intent = {
        "schema_version": "cuda-workload-optimizer/candidate-stage-intent-v1",
        "stage": stage,
        "base_state_sha256": _canonical_digest(state),
        "authorization": copy.deepcopy(dict(admission)),
        "authorization_sha256": authorization_sha256,
        "authorization_view_sha256": _canonical_digest(admission),
        "change_set_sha256": state["change_set_digest"],
        "candidate_digest": candidate_binding["candidate_digest"],
        "candidate_identity_sha256": candidate_binding["after_identity_digest"],
        "candidate_binding_sha256": candidate_binding["digest"],
        "stage_artifact_path": _CANDIDATE_STAGE_ARTIFACTS[stage],
        "created_at_epoch": time.time(),
    }
    intent = _validate_candidate_stage_intent(
        run_root,
        intent,
        state=state,
        candidate_binding=candidate_binding,
    )
    intent_path = _candidate_stage_marker(
        run_root, "candidate_stage_intent.json"
    )
    complete_path = _candidate_stage_marker(
        run_root, "candidate_stage_complete.json"
    )
    _atomic_json(intent_path, intent)
    intent_sha256 = _canonical_digest(intent)
    bound = copy.deepcopy(dict(state))
    bound.update(
        {
            "candidate_stage_intent_sha256": intent_sha256,
            "candidate_stage_intent_stage": stage,
            "updated_at_epoch": time.time(),
        }
    )
    bound = _write_state(run_root, bound)

    current_admission, current_authorization_sha256 = (
        _candidate_stage_admission_view(
            run_root,
            bound,
            control,
            admission["applicable_stages"],
            candidate_binding["after_identity_digest"],
        )
    )
    if (
        current_authorization_sha256 != authorization_sha256
        or current_admission != admission
    ):
        raise ValidationError("candidate stage authorization drifted before runner")

    artifact_path = run_root / _CANDIDATE_STAGE_ARTIFACTS[stage]
    if _identity(control, state["change_scope"])["digest"] != candidate_binding[
        "after_identity_digest"
    ]:
        return (
            _persist_candidate_manual_recovery(
                run_root,
                bound,
                reason="candidate_identity_drift",
            ),
            "manual_recovery",
        )
    else:
        try:
            _load_registered_change_set(
                run_root,
                bound,
                control,
            )
        except ValidationError:
            unbound = copy.deepcopy(bound)
            unbound.pop("candidate_stage_intent_sha256", None)
            unbound.pop("candidate_stage_intent_stage", None)
            unbound["updated_at_epoch"] = time.time()
            unbound = _write_state(run_root, unbound)
            try:
                intent_path.unlink()
            except FileNotFoundError:
                pass
            return unbound, "frozen_artifact_drift"
        started = time.monotonic()
        try:
            result = runner(execution_timeout)
            if not isinstance(result, Mapping):
                raise ValueError(f"{stage} result must be a mapping")
            result = copy.deepcopy(dict(result))
        except _CandidateStageTimeout:
            result = {
                "status": "review_required",
                "reason": "candidate_stage_timeout",
                "timed_out": True,
            }
            _atomic_json(artifact_path, result)
        except Exception as error:
            result = {
                "status": "failed",
                "action_failed": True,
                "failure_type": type(error).__name__,
            }
            _atomic_json(artifact_path, result)
        duration = max(0.0, time.monotonic() - started)
        identity_drifted_after_runner = (
            _identity(control, state["change_scope"])["digest"]
            != candidate_binding["after_identity_digest"]
        )
        if identity_drifted_after_runner:
            result = {
                "status": "manual_recovery_required",
                "reason": "candidate_identity_drift",
                "stage_result": result,
            }
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValidationError("candidate stage runner did not write its artifact")
    completion = {
        "schema_version": (
            "cuda-workload-optimizer/candidate-stage-completion-v1"
        ),
        "stage": stage,
        "intent_sha256": intent_sha256,
        "authorization_sha256": authorization_sha256,
        "authorization_view_sha256": _canonical_digest(admission),
        "change_set_sha256": state["change_set_digest"],
        "candidate_digest": candidate_binding["candidate_digest"],
        "candidate_identity_sha256": candidate_binding[
            "after_identity_digest"
        ],
        "candidate_binding_sha256": candidate_binding["digest"],
        "stage_artifact_path": _CANDIDATE_STAGE_ARTIFACTS[stage],
        "stage_artifact_sha256": _sha256_path(artifact_path),
        "result": result,
        "duration_seconds": duration,
        "completed_at_epoch": time.time(),
    }
    _atomic_json(complete_path, completion)
    completion = _validate_candidate_stage_completion(
        run_root,
        bound,
        control,
        intent,
        _load_candidate_stage_marker(
            complete_path, "candidate stage completion"
        ),
    )
    committed = _commit_candidate_stage_completion(
        run_root,
        bound,
        completion,
    )
    _cleanup_candidate_stage_markers(run_root)
    if identity_drifted_after_runner:
        return (
            _persist_candidate_manual_recovery(
                run_root,
                committed,
                reason="candidate_identity_drift",
            ),
            "manual_recovery",
        )
    return committed, None


def evaluate_change(run_dir: os.PathLike[str] | str) -> dict:
    """Serialize and evaluate one registered ChangeSet exactly once."""
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        _require_run_grant_investment_control(read_run_state(run_root))
        return _evaluate_change_unlocked(run_root)


def _evaluate_change_unlocked(run_dir: os.PathLike[str] | str) -> dict:
    """Verify the bounded diff, review it, run paired evaluation, and decide."""
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    state = read_run_state(run_root)
    if state["status"] == "completed":
        decision = load_json_object(run_root / "decision.json")
        if _canonical_digest(decision) != state.get("decision_digest"):
            raise ValidationError("decision artifact digest does not match state")
        return decision
    control = _load_frozen_control(run_root, state)
    (
        state,
        recovered_failure_decision,
        unbound_candidate_failure,
    ) = _candidate_recovery_preflight(run_root, state, control)
    if unbound_candidate_failure:
        raise ValidationError(
            "resume the unbound candidate failure before evaluation"
        )
    if recovered_failure_decision is not None:
        return recovered_failure_decision
    if state.get("next_action") == "manual_recovery":
        return state
    state, diagnosis_recovery = _recover_diagnosis_publish(run_root, state)
    if diagnosis_recovery != "none":
        return state
    state, direction_recovery, _direction_aggregate = (
        _recover_reviewer_checkpoint(run_root, state, "direction")
    )
    if direction_recovery in {"manual", "waiting"}:
        return state
    state, final_recovery, _final_aggregate = (
        _recover_reviewer_checkpoint(run_root, state, "final")
    )
    if final_recovery == "manual":
        return state
    if state.get("stage") == "decision" and state["next_action"] == "review_required":
        raise ValidationError(
            "legacy review-required run cannot resume under candidate lifecycle v1"
        )
    if state["next_action"] == "review_required":
        pause_authorization = _sha256(
            state.get("candidate_pause_authorization_sha256"),
            "candidate pause authorization",
        )
        decision = load_json_object(run_root / "decision.json")
        if (
            _canonical_digest(decision) != state.get("decision_digest")
            or decision.get("status") != "review_required"
            or decision.get("rolled_back") is not False
            or decision.get("next_stage") != state.get("candidate_stage")
        ):
            raise ValidationError("paused candidate decision drifted")
        if _candidate_authorization_digest(state) == pause_authorization:
            return decision
        control = _load_frozen_control(run_root, state)
        try:
            paused_change = _load_registered_change_set(
                run_root,
                state,
                control,
            )
            paused_binding = _validate_candidate_binding(
                _load_candidate_stage_marker(
                    run_root / "candidate_binding.json",
                    "candidate binding",
                ),
                candidate=paused_change["candidate"],
                change_set_sha256=_canonical_digest(paused_change),
            )
            _validated_identity_artifact(
                load_json_object(
                    run_root / "rounds" / "round-1" / "after_identity.json"
                ),
                paused_binding["after_identity_digest"],
            )
            _frozen_snapshot_identity(
                control,
                run_root,
                paused_change["scope"],
                state["before_identity_digest"],
            )
            candidate_diff = run_root / "candidate.diff"
            if (
                candidate_diff.is_symlink()
                or not candidate_diff.is_file()
                or _sha256_path(candidate_diff)
                != decision.get("candidate_diff_sha256")
            ):
                raise ValidationError("paused candidate diff drifted")
        except (OSError, KeyError, ValidationError):
            return _persist_candidate_manual_recovery(
                run_root,
                state,
                reason="candidate_pause_artifact_drift",
            )
        if (
            _identity(control, paused_change["scope"])["digest"]
            != paused_binding["after_identity_digest"]
        ):
            return _persist_candidate_manual_recovery(
                run_root,
                state,
                reason="candidate_identity_drift",
            )
        resumed = copy.deepcopy(dict(state))
        resumed.update(
            {
                "stage": "candidate_validation",
                "next_action": "edit_then_evaluate",
                "updated_at_epoch": time.time(),
            }
        )
        resumed.pop("candidate_pause_authorization_sha256", None)
        resumed.pop("terminal_reason", None)
        state = _write_state(run_root, resumed)
    if state["next_action"] != "edit_then_evaluate":
        raise ValidationError("run is not ready to evaluate a ChangeSet")
    control = _load_frozen_control(run_root, state)
    try:
        change = _load_registered_change_set(
            run_root,
            state,
            control,
        )
    except ValidationError:
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=state["change_scope"],
            reason="frozen_artifact_drift",
            primary_status=None,
        )
    change_digest = _canonical_digest(change)
    workload = _normalize_frozen_workload(control)
    if workload.source_hash != state["workload_source_hash"]:
        decision = _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="workload_identity_drift",
            primary_status=None,
        )
        if decision["status"] == "manual_recovery_required":
            return decision
        if decision["status"] == "review_required":
            return decision
        raise ValidationError("workload identity drifted before evaluation")
    try:
        _validate_workload_candidate_minimum_effect(change, workload)
    except ValidationError:
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="candidate_below_project_contract",
            primary_status=None,
        )
    try:
        before = _validated_identity_artifact(
            load_json_object(
                run_root / "rounds" / "round-1" / "before_identity.json"
            ),
            state["before_identity_digest"],
        )
    except (OSError, ValidationError, KeyError):
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="frozen_artifact_drift",
            primary_status=None,
        )
    after = _identity(control, change["scope"])
    changed = _changed_paths(before, after)
    if not changed:
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="no_scoped_changes",
            primary_status=None,
        )
    outside = [path for path in changed if not _path_allowed(path, change["paths"])]
    if outside:
        decision = _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="change_set_path_escape",
            primary_status=None,
        )
        if decision["status"] == "manual_recovery_required":
            return decision
        if decision["status"] == "review_required":
            return decision
        raise ValidationError(
            "actual scoped diff is outside ChangeSet paths: " + ", ".join(outside)
        )
    bound_candidate = copy.deepcopy(change["candidate"])
    has_candidate_checkpoint = (
        state.get("candidate_stage_intent_sha256") is not None
        or bool(state.get("candidate_stage_completions"))
        or (run_root / "candidate_stage_intent.json").exists()
        or (run_root / "candidate_stage_complete.json").exists()
    )
    if has_candidate_checkpoint:
        candidate_binding = _validate_candidate_binding(
            _load_candidate_stage_marker(
                run_root / "candidate_binding.json", "candidate binding"
            ),
            candidate=bound_candidate,
            change_set_sha256=change_digest,
        )
        _validated_identity_artifact(
            load_json_object(
                run_root / "rounds" / "round-1" / "after_identity.json"
            ),
            candidate_binding["after_identity_digest"],
        )
    else:
        _atomic_json(
            run_root / "rounds" / "round-1" / "after_identity.json",
            after,
        )
        candidate_binding = {
            "schema_version": "cuda-workload-optimizer/candidate-binding-v1",
            "candidate": bound_candidate,
            "candidate_digest": _canonical_digest(bound_candidate),
            "change_set_digest": change_digest,
            "after_identity_digest": after["digest"],
        }
        candidate_binding["digest"] = _canonical_digest(candidate_binding)
        _atomic_json(run_root / "candidate_binding.json", candidate_binding)
        (run_root / "candidate.diff").write_text(
            _candidate_diff(control, run_root, changed, change["scope"]),
            encoding="utf-8",
        )

    runtime = _BUDGET_RUNTIME[control["budget"]]
    workload_attempt = 0

    def static_review_stage(timeout_seconds: float) -> dict:
        return _run_candidate_static_review_bounded(
            control,
            run_root,
            timeout_seconds=timeout_seconds,
        )

    def run_workload_once(
        evaluation_workload: Any,
        *,
        candidate: Any,
        role: str,
        case: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        controlled_deadline_epoch: float | None = None,
    ) -> dict:
        nonlocal workload_attempt
        if evaluation_workload.source_hash != workload.source_hash:
            raise ValidationError("workload changed inside the evaluation stage")
        remaining = (
            None
            if controlled_deadline_epoch is None
            else float(controlled_deadline_epoch) - time.time()
        )
        if remaining is not None and remaining <= 0.0:
            raise _CandidateStageTimeout(
                "candidate workload exhausted its controlled timeout"
            )
        if timeout is None:
            if remaining is None:
                raise ValidationError(
                    "candidate workload requires a controlled timeout"
                )
            effective_timeout = remaining
        else:
            effective_timeout = (
                float(timeout)
                if remaining is None
                else min(float(timeout), remaining)
            )
        if workload.kind != "python":
            return _load_workload_module().run_spec_once(
                workload,
                candidate=candidate,
                role=role,
                case=case,
                timeout=effective_timeout,
            )
        workload_attempt += 1
        return _run_python_workload_once_bounded(
            control,
            run_root,
            candidate=candidate,
            role=role,
            case=case,
            timeout_seconds=effective_timeout,
            task=f"workload-{role}-{workload_attempt}",
        )

    def evaluate_pairs(
        stage: str,
        blocks: int,
        timeout_seconds: float,
    ) -> dict:
        controlled_deadline_epoch = time.time() + float(timeout_seconds)

        def paired_runner(
            evaluation_workload: Any,
            *,
            candidate: Any,
            role: str,
            case: Mapping[str, Any] | None = None,
            timeout: float | None = None,
        ) -> dict:
            return run_workload_once(
                evaluation_workload,
                candidate=candidate,
                role=role,
                case=case,
                timeout=timeout,
                controlled_deadline_epoch=controlled_deadline_epoch,
            )

        evaluation = _load_evaluate_module().evaluate_pairs(
            workload,
            control["baseline_candidate"],
            bound_candidate,
            blocks=blocks,
            retries=runtime["retries"],
            seed=0,
            timeout=float(timeout_seconds),
            deadline_epoch=controlled_deadline_epoch,
            bootstrap_samples=runtime["bootstrap"],
            runner=paired_runner,
        )
        if evaluation.get("failure", {}).get("error_type") in {
            "TimeoutError",
            "_CandidateStageTimeout",
        }:
            raise _CandidateStageTimeout(
                "paired evaluation exhausted its controlled timeout"
            )
        _atomic_json(run_root / f"{stage}_evaluation.json", evaluation)
        primary = evaluation.get("primary", {})
        constraints_passed = all(
            item.get("status") == "passed"
            for item in evaluation.get("constraints", [])
        )
        passed = (
            evaluation.get("status") == "evaluated"
            and constraints_passed
        )
        if stage == "formal_paired":
            passed = (
                passed
                and primary.get("status") == "confirmed_win"
                and constraints_passed
            )
        return {
            "status": "passed" if passed else "failed",
            "estimate": primary.get("estimate_pct"),
            "lower_bound": primary.get("ci_low_pct"),
            "upper_bound": primary.get("ci_high_pct"),
        }

    diagnosis = load_json_object(run_root / "diagnosis.json")
    investment_brief = None
    if "analysis_contract" in control:
        diagnostic_decision = _load_bound_diagnostic_artifacts(
            run_root, state, expected_decision="PURSUE"
        )
        investment_brief = diagnostic_decision["investment_brief"]
    explicit_uncertainty = _explicit_profiler_uncertainty(
        diagnosis, investment_brief
    )
    selected_profiler = next(
        (
            probe
            for probe in sorted(
                control["probes"],
                key=lambda item: (float(item["timeout_seconds"]), item["id"]),
            )
            if probe["kind"] in explicit_uncertainty
        ),
        None,
    )

    def profiler_stage(timeout_seconds: float) -> dict:
        if selected_profiler is None:
            artifact = {
                "status": "failed",
                "reason": "live_uncertainty_has_no_configured_profiler_action",
                "unresolved_uncertainty": sorted(explicit_uncertainty),
            }
            _atomic_json(run_root / "profiler_stage.json", artifact)
            return artifact
        profile_root = run_root / "candidate_profile"
        result = _run_probe_unchecked(
            selected_profiler,
            control,
            profile_root,
            deadline_epoch=time.time() + float(timeout_seconds),
        )
        result_path = (
            profile_root / "probes" / f"{selected_profiler['id']}.json"
        )
        execution = load_json_object(
            profile_root
            / "probes"
            / f"{selected_profiler['id']}.execution.json"
        )
        if execution.get("timed_out") is True:
            raise _CandidateStageTimeout(
                "candidate profiler exhausted its controlled timeout"
            )
        artifact = {
            "status": (
                "passed"
                if result.get("status") in {"ok", "degraded"}
                else "failed"
            ),
            "reason": "candidate_profiler_executed",
            "unresolved_uncertainty": sorted(explicit_uncertainty),
            "probe_id": selected_profiler["id"],
            "probe_kind": selected_profiler["kind"],
            "evidence_path": str(result_path.relative_to(run_root)),
            "evidence_sha256": _sha256_path(result_path),
        }
        _atomic_json(run_root / "profiler_stage.json", artifact)
        return artifact

    def minimum_correctness_stage(timeout_seconds: float) -> dict:
        cases = list(workload.cases) or [{}]
        try:
            observation = run_workload_once(
                workload,
                candidate=bound_candidate,
                role="candidate",
                case=cases[0],
                timeout=float(timeout_seconds),
                controlled_deadline_epoch=(
                    time.time() + float(timeout_seconds)
                ),
            )
        except TimeoutError as error:
            raise _CandidateStageTimeout(
                "candidate correctness exhausted its controlled timeout"
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            artifact = {
                "status": "failed",
                "failure": type(error).__name__,
            }
        else:
            artifact = {
                "status": "passed",
                "validation": observation.get("validation"),
            }
        _atomic_json(run_root / "correctness.json", artifact)
        return artifact

    gate_contract = {
        "soft_target_seconds": runtime["soft_target_seconds"],
        "hard_ceiling_seconds": runtime["hard_ceiling_seconds"],
        "minimum_effect": {
            "mechanism_us": 1.0,
            "service_pct": max(0.5, float(workload.objective["min_effect_pct"])),
        },
    }
    gate = _load_budget_module().CandidateGate(
        gate_contract,
        bound_candidate,
    )
    candidate_actions = {
        "static_review": static_review_stage,
        "build_correctness": minimum_correctness_stage,
        "short_paired": lambda timeout_seconds: evaluate_pairs(
            "short_paired",
            min(2, runtime["blocks"]),
            timeout_seconds,
        ),
        "formal_paired": lambda timeout_seconds: evaluate_pairs(
            "formal_paired",
            runtime["blocks"],
            timeout_seconds,
        ),
    }
    candidate_action_safety_timeouts = {
        "static_review": 60.0,
        "build_correctness": 60.0,
        "short_paired": 120.0,
        "formal_paired": 120.0,
    }
    applicable_stages = [
        "static_review",
        "build_correctness",
        "short_paired",
        "formal_paired",
    ]
    if explicit_uncertainty and selected_profiler is not None:
        applicable_stages.insert(3, "profiler")
        candidate_actions["profiler"] = profiler_stage
        candidate_action_safety_timeouts["profiler"] = float(
            selected_profiler["timeout_seconds"]
        )

    state = _recover_candidate_stage_checkpoint(
        run_root,
        state,
        control,
        candidate_binding,
    )
    if state.get("next_action") == "manual_recovery":
        return state
    while True:
        if (
            _identity(control, change["scope"])["digest"]
            != candidate_binding["after_identity_digest"]
        ):
            return _persist_candidate_manual_recovery(
                run_root,
                state,
                reason="candidate_identity_drift",
            )
        completed_results = _validated_candidate_stage_results(
            run_root,
            state,
            control,
            candidate_binding,
        )
        admission, authorization_sha256 = _candidate_stage_admission_view(
            run_root,
            state,
            control,
            applicable_stages,
            candidate_binding["after_identity_digest"],
        )
        gate_result = gate.decide(
            completed_results,
            _controlled_spend_seconds(state),
            admission,
        )
        next_stage = gate_result.get("next_stage")
        if next_stage is not None:
            if next_stage not in applicable_stages:
                raise ValidationError("candidate gate selected an invalid stage")
            if state.get("candidate_stage") != next_stage:
                staged = copy.deepcopy(dict(state))
                staged["candidate_stage"] = next_stage
                staged["updated_at_epoch"] = time.time()
                state = _write_state(run_root, staged)
        if gate_result["decision"] != "RUN_STAGE":
            break
        if (
            _identity(control, change["scope"])["digest"]
            != candidate_binding["after_identity_digest"]
        ):
            return _persist_candidate_manual_recovery(
                run_root,
                state,
                reason="candidate_identity_drift",
            )
        state, change_set_drift = _execute_candidate_stage(
            run_root,
            state,
            control,
            candidate_binding,
            admission,
            authorization_sha256,
            next_stage,
            candidate_actions[next_stage],
            candidate_action_safety_timeouts[next_stage],
        )
        if change_set_drift == "manual_recovery":
            return state
        if change_set_drift is not None:
            return _finish_rejected(
                run_root,
                state,
                control,
                scope=change["scope"],
                reason=change_set_drift,
                primary_status=None,
            )

    formal_path = run_root / "formal_paired_evaluation.json"
    short_path = run_root / "short_paired_evaluation.json"
    if formal_path.is_file() and not formal_path.is_symlink():
        evaluation = load_json_object(formal_path)
    elif short_path.is_file() and not short_path.is_symlink():
        evaluation = load_json_object(short_path)
    else:
        evaluation = {
            "schema_version": "cuda-workload-optimizer/evaluation-v1",
            "status": gate_result["stop_reason"],
        }
    _atomic_json(run_root / "evaluation.json", evaluation)
    _atomic_json(run_root / "time_gate.json", gate_result)
    if gate_result["decision"] == "REVIEW_REQUIRED":
        _atomic_json(
            run_root / "review.json",
            {
                "schema_version": "cuda-workload-optimizer/review-artifact-v1",
                "status": "skipped",
                "request_digest": None,
                "response": None,
                "execution": {"reason": gate_result["stop_reason"]},
            },
        )
        return _finish_review_required(
            run_root,
            state,
            control,
            scope=change["scope"],
            primary_status=evaluation.get("primary", {}).get("status"),
            time_gate=gate_result,
        )
    if gate_result["decision"] != "PROMOTE":
        rejection_reason = gate_result["stop_reason"]
        if (
            gate_result["stop_reason"]
            in {"short_pair_failed", "formal_pair_failed"}
            and evaluation.get("status") != "evaluated"
        ):
            rejection_reason = "workload_failed"
        _atomic_json(
            run_root / "review.json",
            {
                "schema_version": "cuda-workload-optimizer/review-artifact-v1",
                "status": "skipped",
                "request_digest": None,
                "response": None,
                "execution": {"reason": gate_result["stop_reason"]},
            },
        )
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason=rejection_reason,
            primary_status=evaluation.get("primary", {}).get("status"),
            time_gate=gate_result,
        )

    if control.get("evaluation_gate", "promotion") != "promotion":
        _atomic_json(
            run_root / "review.json",
            {
                "schema_version": "cuda-workload-optimizer/review-artifact-v1",
                "status": "skipped",
                "request_digest": None,
                "response": None,
                "execution": {"reason": "reject_only_stage_cannot_promote"},
            },
        )
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="reject_only_stage_cannot_promote",
            primary_status=evaluation.get("primary", {}).get("status"),
            time_gate=gate_result,
        )

    _candidate_stage_admission_view(
        run_root,
        state,
        control,
        applicable_stages,
        candidate_binding["after_identity_digest"],
    )
    if (
        _identity(control, change["scope"])["digest"]
        != candidate_binding["after_identity_digest"]
    ):
        return _persist_candidate_manual_recovery(
            run_root,
            state,
            reason="candidate_identity_drift",
        )
    try:
        change = _load_registered_change_set(
            run_root,
            state,
            control,
        )
    except ValidationError:
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="frozen_artifact_drift",
            primary_status=evaluation.get("primary", {}).get("status"),
        )
    state, review_artifact = _run_final_managed_review(
        run_root,
        state,
        control,
        change,
        evaluation,
    )
    if state.get("next_action") == "manual_recovery":
        return state
    _write_final_review_adjudication(run_root, review_artifact, evaluation)
    if (
        _identity(control, change["scope"])["digest"]
        != candidate_binding["after_identity_digest"]
    ):
        return _persist_candidate_manual_recovery(
            run_root,
            state,
            reason="candidate_identity_drift",
        )
    try:
        change = _load_registered_change_set(
            run_root,
            state,
            control,
        )
    except ValidationError:
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason="frozen_artifact_drift",
            primary_status=evaluation.get("primary", {}).get("status"),
        )
    primary_status = evaluation.get("primary", {}).get("status")
    constraints = evaluation.get("constraints", [])
    promoted = (
        evaluation.get("status") == "evaluated"
        and primary_status == "confirmed_win"
        and all(item.get("status") == "passed" for item in constraints)
        and control.get("evaluation_gate", "promotion") == "promotion"
    )
    if not promoted:
        if evaluation.get("status") != "evaluated":
            reason = "workload_failed"
        elif primary_status != "confirmed_win":
            reason = "primary_not_confirmed"
        elif control.get("evaluation_gate", "promotion") == "reject_only":
            reason = "reject_only_stage_cannot_promote"
        else:
            reason = "constraint_failed"
        return _finish_rejected(
            run_root,
            state,
            control,
            scope=change["scope"],
            reason=reason,
            primary_status=primary_status,
        )

    evaluation_digest = _canonical_digest(evaluation)
    decision = {
        "schema_version": "cuda-workload-optimizer/decision-v1",
        "status": "promoted",
        "reason": "paired_workload_win",
        "primary_status": primary_status,
        "rolled_back": False,
        "change_set_digest": change_digest,
        "candidate_binding_digest": candidate_binding["digest"],
        "after_identity_digest": candidate_binding["after_identity_digest"],
        "evaluation_digest": evaluation_digest,
        "elapsed_seconds": gate_result["elapsed_seconds"],
        "stop_reason": gate_result["stop_reason"],
        "skipped_expensive_stages": gate_result["skipped_expensive_stages"],
    }
    _atomic_json(run_root / "decision.json", decision)
    updated = copy.deepcopy(state)
    for stage in ("review", "evaluation", "decision"):
        if stage not in updated["completed_stages"]:
            updated["completed_stages"].append(stage)
    updated.update(
        {
            "status": "completed",
            "stage": "decision",
            "next_action": "done",
            "updated_at_epoch": time.time(),
            "decision_digest": _canonical_digest(decision),
        }
    )
    _write_state(run_root, updated)
    return decision


def abandon(run_dir: os.PathLike[str] | str) -> dict:
    """Roll back one registered candidate and seal an idempotent terminal decision."""
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        state = read_run_state(run_root)
        decision_path = run_root / "decision.json"
        control = _load_frozen_control(run_root, state)
        (
            state,
            recovered_failure_decision,
            unbound_candidate_failure,
        ) = _candidate_recovery_preflight(run_root, state, control)
        if unbound_candidate_failure:
            raise ValidationError(
                "resume the unbound candidate failure before abandon"
            )
        if recovered_failure_decision is not None:
            return recovered_failure_decision
        reviewer_manual_reasons = {
            "direction_review_outcome_unknown",
            "direction_review_recovery_invalid",
            "final_review_outcome_unknown",
            "final_review_recovery_invalid",
            "review_intent_binding_unresolved",
            "review_intent_binding_invalid",
        }
        if (
            state.get("next_action") == "manual_recovery"
            and state.get("manual_recovery_reason")
            not in reviewer_manual_reasons
        ):
            return state
        if state.get("status") == "completed":
            decision = load_json_object(decision_path)
            if (
                _canonical_digest(decision) != state.get("decision_digest")
                or decision.get("status") != "abandoned"
                or decision.get("rolled_back") is not True
            ):
                raise ValidationError("completed run cannot be abandoned")
            return decision
        scope = state.get("change_scope")
        before_identity = state.get("before_identity_digest")
        if scope not in {"project", "isolated_environment"}:
            raise ValidationError("run has no registered candidate to abandon")
        _sha256(before_identity, "candidate before identity")
        rollback_identity = "unbound"
        binding_path = run_root / "candidate_binding.json"
        if binding_path.exists() or binding_path.is_symlink():
            try:
                rollback_identity = _candidate_rollback_identity(
                    control,
                    run_root,
                    state,
                    scope=scope,
                    before_identity_digest=before_identity,
                )
            except (KeyError, OSError, ValidationError):
                return _persist_candidate_manual_recovery(
                    run_root,
                    state,
                    reason="candidate_abandon_identity_or_artifact_drift",
                )
        try:
            if rollback_identity != "before":
                _restore_snapshot(
                    control,
                    run_root,
                    scope,
                    before_identity,
                )
            if _identity(control, scope)["digest"] != before_identity:
                raise ValidationError(
                    "abandon rollback did not restore the frozen identity"
                )
        except (OSError, ValidationError) as error:
            decision = {
                "schema_version": "cuda-workload-optimizer/decision-v1",
                "status": "manual_recovery_required",
                "reason": "abandon_rollback_failed",
                "rolled_back": False,
                "error": f"{type(error).__name__}: {error}",
                "snapshot": str(
                    run_root
                    / "snapshot"
                    / ("project" if scope == "project" else "environment")
                ),
            }
            _atomic_json(decision_path, decision)
            blocked = copy.deepcopy(dict(state))
            blocked.update(
                {
                    "status": "manual_recovery_required",
                    "stage": "decision",
                    "next_action": "manual_recovery",
                    "decision_digest": _canonical_digest(decision),
                    "manual_recovery_reason": "abandon_rollback_failed",
                    "updated_at_epoch": time.time(),
                }
            )
            _write_state(run_root, blocked)
            return decision
        decision = {
            "schema_version": "cuda-workload-optimizer/decision-v1",
            "status": "abandoned",
            "reason": "candidate_abandoned",
            "rolled_back": True,
            "change_set_digest": state.get("change_set_digest"),
            "candidate_stage": state.get("candidate_stage"),
        }
        _atomic_json(decision_path, decision)
        abandoned = copy.deepcopy(dict(state))
        for completed_stage in ("review", "evaluation", "decision"):
            if completed_stage not in abandoned["completed_stages"]:
                abandoned["completed_stages"].append(completed_stage)
        abandoned.update(
            {
                "status": "completed",
                "stage": "decision",
                "next_action": "done",
                "decision_digest": _canonical_digest(decision),
                "terminal_reason": "candidate_abandoned",
                "updated_at_epoch": time.time(),
            }
        )
        abandoned.pop("candidate_pause_authorization_sha256", None)
        abandoned.pop("candidate_stage_intent_sha256", None)
        abandoned.pop("candidate_stage_intent_stage", None)
        abandoned.pop("manual_recovery_reason", None)
        _write_state(run_root, abandoned)
        _cleanup_candidate_stage_markers(run_root)
        return decision


def resume_run(run_dir: os.PathLike[str] | str) -> dict:
    run_root = Path(run_dir).expanduser().resolve(strict=False)
    with _run_lock(run_root):
        _require_run_grant_investment_control(read_run_state(run_root))
        return _resume_run_unlocked(run_root)


def _resume_run_unlocked(run_root: Path) -> dict:
    state = read_run_state(run_root)
    control = _load_frozen_control(run_root, state)
    recovered = _recover_candidate_failure(
        run_root,
        control,
        discard_unbound=True,
    )
    if recovered is _UNBOUND_CANDIDATE_FAILURE:
        state = read_run_state(run_root)
    elif recovered is not None:
        state = recovered
    if state.get("next_action") == "manual_recovery":
        return state
    state, diagnosis_recovery = _recover_diagnosis_publish(run_root, state)
    if diagnosis_recovery in {"recovered", "cleaned", "manual"}:
        return state
    state, direction_recovery, _direction_aggregate = (
        _recover_reviewer_checkpoint(run_root, state, "direction")
    )
    if direction_recovery in {"waiting", "cleaned", "manual"}:
        return state
    state, final_recovery, _final_aggregate = (
        _recover_reviewer_checkpoint(run_root, state, "final")
    )
    if final_recovery == "manual":
        return state
    mirror_only_actions = {
        "propose_hypotheses",
        "evidence_gap",
        "refresh_required",
        "register_change",
        "done",
        "manual_recovery",
    }
    if (
        state.get("stage") == "active_diagnosis"
        and (run_root / "diagnosis_context.json").is_file()
        and state.get("next_action") in mirror_only_actions
        and state.get("diagnosis_context_sha256") is not None
        and state.get("active_diagnosis_ledger_sequence") is not None
    ):
        # These branches return without executing a Controller action.  This
        # permits state-bound knowledge validation and mirror repair while
        # preserving the normal full-surface gate on every execution path.
        _load_active_diagnosis_context(
            control,
            run_root,
            state,
            verify_current_project_surface=False,
        )
        return state
    if state["next_action"] == "collect_evidence":
        return _collect_active_diagnosis_evidence_unlocked(control, run_root)
    if state["next_action"] == "edit_then_evaluate" and (
        state.get("candidate_stage_intent_sha256") is not None
        or bool(state.get("candidate_stage_completions"))
        or (run_root / "candidate_stage_intent.json").exists()
        or (run_root / "candidate_stage_complete.json").exists()
    ):
        return _evaluate_change_unlocked(run_root)
    if state["next_action"] == "review_required":
        pause_authorization = state.get("candidate_pause_authorization_sha256")
        if pause_authorization is None:
            return state
        _sha256(pause_authorization, "candidate pause authorization")
        decision = load_json_object(run_root / "decision.json")
        if (
            _canonical_digest(decision) != state.get("decision_digest")
            or decision.get("status") != "review_required"
            or decision.get("rolled_back") is not False
            or decision.get("next_stage") != state.get("candidate_stage")
        ):
            raise ValidationError("paused candidate decision drifted")
        if _candidate_authorization_digest(state) == pause_authorization:
            return state
        return _evaluate_change_unlocked(run_root)
    if state["next_action"] in {
        "propose_hypotheses",
        "evidence_gap",
        "review_required",
        "refresh_required",
        "register_change",
        "edit_then_evaluate",
        "done",
        "manual_recovery",
    }:
        return state
    return _start_run_unlocked(control, run_root)


def _cmd_candidate_static_review(args: argparse.Namespace) -> None:
    run_root = Path(args.run_dir).expanduser().resolve(strict=False)
    output_path = Path(args.out).expanduser().resolve(strict=False)
    expected_output = (
        run_root / _CANDIDATE_STAGE_ARTIFACTS["static_review"]
    ).resolve(strict=False)
    if output_path != expected_output:
        raise ValidationError(
            "candidate static review output must be its stage artifact"
        )
    state = read_run_state(run_root)
    control = _load_frozen_control(run_root, state)
    change = _load_registered_change_set(run_root, state, control)
    change_digest = _canonical_digest(change)
    candidate_binding = _validate_candidate_binding(
        _load_candidate_stage_marker(
            run_root / "candidate_binding.json", "candidate binding"
        ),
        candidate=change["candidate"],
        change_set_sha256=change_digest,
    )
    intent = _validate_candidate_stage_intent(
        run_root,
        _load_candidate_stage_marker(
            run_root / "candidate_stage_intent.json",
            "candidate stage intent",
        ),
        state=state,
        candidate_binding=candidate_binding,
    )
    if intent["stage"] != "static_review":
        raise ValidationError(
            "candidate static review lacks a bound static stage intent"
        )
    before = _validated_identity_artifact(
        load_json_object(
            run_root / "rounds" / "round-1" / "before_identity.json"
        ),
        state["before_identity_digest"],
    )
    after = _validated_identity_artifact(
        load_json_object(
            run_root / "rounds" / "round-1" / "after_identity.json"
        ),
        candidate_binding["after_identity_digest"],
    )
    if (
        _identity(control, change["scope"])["digest"]
        != candidate_binding["after_identity_digest"]
    ):
        raise ValidationError(
            "candidate identity drifted inside static review child"
        )
    changed = _changed_paths(before, after)
    outside = [
        path for path in changed if not _path_allowed(path, change["paths"])
    ]
    if not changed or outside:
        raise ValidationError(
            "candidate static review changed paths drifted"
        )
    artifact = _static_review_changed_files(
        control,
        scope=change["scope"],
        changed=changed,
        candidate_binding=candidate_binding,
        change_set_digest=change_digest,
        after_identity_digest=candidate_binding["after_identity_digest"],
    )
    _atomic_json(output_path, artifact)


def _cmd_workload_once(args: argparse.Namespace) -> None:
    control = validate_control_manifest(
        load_json_object(args.control), args.control
    )
    request = load_json_object(args.request)
    if set(request) != {"candidate", "role", "case"}:
        raise ValidationError("workload child request fields are invalid")
    workload = _normalize_frozen_workload(control)
    if workload.kind != "python":
        raise ValidationError("workload child accepts only Python adapters")
    observation = _load_workload_module().run_spec_once(
        workload,
        candidate=request["candidate"],
        role=request["role"],
        case=request["case"],
        timeout=None,
    )
    _atomic_json(Path(args.out), observation)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and run bounded GPU workload optimization rounds."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    workload_once = subparsers.add_parser("_workload-once", help=argparse.SUPPRESS)
    workload_once.add_argument("--control", required=True)
    workload_once.add_argument("--request", required=True)
    workload_once.add_argument("--out", required=True)
    candidate_static_review = subparsers.add_parser(
        "_candidate-static-review", help=argparse.SUPPRESS
    )
    candidate_static_review.add_argument("--run-dir", required=True)
    candidate_static_review.add_argument("--out", required=True)
    validate = subparsers.add_parser("validate", help="validate controller JSON")
    validate.add_argument("--control", required=True)
    validate.add_argument("--change-set")
    probe = subparsers.add_parser("probe", help="collect normalized probe evidence")
    probe.add_argument("--control", required=True)
    probe.add_argument("--run-dir", required=True)
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="classify stored normalized probe evidence"
    )
    diagnose_parser.add_argument("--run-dir", required=True)
    review = subparsers.add_parser("review", help="request optional advisory review")
    review.add_argument("--control", required=True)
    review.add_argument("--run-dir", required=True)
    review.add_argument("--change-set", required=True)
    run = subparsers.add_parser("run", help="collect baseline evidence and diagnosis")
    run.add_argument("--control", required=True)
    run.add_argument("--run-dir", required=True)
    status = subparsers.add_parser("status", help="read the current run checkpoint")
    status.add_argument("--run-dir", required=True)
    register = subparsers.add_parser(
        "register-change", help="freeze and register a bounded ChangeSet"
    )
    register.add_argument("--control", required=True)
    register.add_argument("--run-dir", required=True)
    register.add_argument("--change-set", required=True)
    diagnosis_proposal = subparsers.add_parser(
        "register-diagnosis",
        help="validate and freeze an active diagnosis proposal",
    )
    diagnosis_proposal.add_argument("--control", required=True)
    diagnosis_proposal.add_argument("--run-dir", required=True)
    diagnosis_proposal.add_argument("--hypothesis-set", required=True)
    diagnosis_proposal.add_argument("--request-set", required=True)
    diagnosis_proposal.add_argument("--knowledge-inputs")
    authorize = subparsers.add_parser(
        "authorize-run",
        help="seal a run-level investment authorization",
    )
    authorize.add_argument("--control", required=True)
    authorize.add_argument("--run-dir", required=True)
    authorize.add_argument("--grant", required=True)
    collect_evidence = subparsers.add_parser(
        "collect-evidence",
        help="execute the selected frozen active-diagnosis evidence action",
    )
    collect_evidence.add_argument("--control", required=True)
    collect_evidence.add_argument("--run-dir", required=True)
    evaluate = subparsers.add_parser(
        "evaluate", help="verify, evaluate, promote, or roll back a candidate"
    )
    evaluate.add_argument("--run-dir", required=True)
    abandon_parser = subparsers.add_parser(
        "abandon", help="roll back and abandon a registered candidate"
    )
    abandon_parser.add_argument("--run-dir", required=True)
    resume = subparsers.add_parser("resume", help="resume from the last checkpoint")
    resume.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "_workload-once":
            _cmd_workload_once(args)
            return 0
        if args.command == "_candidate-static-review":
            _cmd_candidate_static_review(args)
            return 0
        if args.command == "validate":
            control = validate_control_manifest(
                load_json_object(args.control), args.control
            )
            if args.change_set:
                validate_change_set(load_json_object(args.change_set), control)
            print(json.dumps({"status": "valid"}, sort_keys=True))
            return 0
        if args.command == "probe":
            values = run_probes(load_json_object(args.control), args.run_dir)
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "probe_count": len(values),
                        "available_probe_count": sum(
                            item["status"] in {"ok", "degraded"} for item in values
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "diagnose":
            print(json.dumps(diagnose_run(args.run_dir), sort_keys=True))
            return 0
        if args.command == "review":
            artifact = review_change(
                load_json_object(args.control),
                args.run_dir,
                load_json_object(args.change_set),
            )
            print(json.dumps(artifact, sort_keys=True))
            return 0
        if args.command == "run":
            control = validate_control_manifest(load_json_object(args.control))
            if control["schema_version"] != CONTROL_SCHEMA_V2:
                raise ValidationError(
                    "new controller runs require control-v2 with readiness_contract; "
                    "control-v1 remains available for validate and historical resume"
                )
            print(
                json.dumps(
                    start_run(control, args.run_dir),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "status":
            print(json.dumps(read_run_state(args.run_dir), sort_keys=True))
            return 0
        if args.command == "register-change":
            print(
                json.dumps(
                    register_change(
                        load_json_object(args.control),
                        args.run_dir,
                        load_json_object(args.change_set),
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "register-diagnosis":
            print(
                json.dumps(
                    register_active_diagnosis_proposal(
                        load_json_object(args.control),
                        args.run_dir,
                        load_json_object(args.hypothesis_set),
                        load_json_object(args.request_set),
                        knowledge_inputs=(
                            load_json_object(args.knowledge_inputs)
                            if args.knowledge_inputs
                            else None
                        ),
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "authorize-run":
            print(
                json.dumps(
                    authorize_run(
                        load_json_object(args.control),
                        args.run_dir,
                        load_json_object(args.grant),
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "collect-evidence":
            print(
                json.dumps(
                    collect_active_diagnosis_evidence(
                        load_json_object(args.control), args.run_dir
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "evaluate":
            print(json.dumps(evaluate_change(args.run_dir), sort_keys=True))
            return 0
        if args.command == "abandon":
            print(json.dumps(abandon(args.run_dir), sort_keys=True))
            return 0
        if args.command == "resume":
            print(json.dumps(resume_run(args.run_dir), sort_keys=True))
            return 0
    except ValidationError as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
