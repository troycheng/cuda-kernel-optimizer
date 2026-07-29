#!/usr/bin/env python3
"""Route a bounded set of offline diagnostic cards into active diagnosis."""

from __future__ import annotations

import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path


CARDS_PATH = Path(__file__).resolve().parents[1] / "references" / "diagnostic_cards.json"
REFERENCE_DIR = CARDS_PATH.parent
_MAX_REFERENCE_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_LAYERS = {
    "cpu",
    "gpu",
    "framework",
    "transfer",
    "communication",
    "io",
    "synchronization",
    "idle",
}
_REQUESTED_CLAIMS = {"kernel", "workload", "serving"}
_LAYER_TO_DIAGNOSIS_CATEGORY = {
    "communication": "communication",
    "framework": "framework",
    "gpu": "kernel",
    "io": "io",
    "transfer": "transfer",
}
_DIAGNOSTIC_PRODUCERS = {
    "nsys_timeline": "nsys-timeline-adapter",
    "pytorch_profile": "pytorch-profile-adapter",
}
_DIAGNOSTIC_SIGNALS = {
    "nsys_timeline": {
        "launch_gap_short_context": (
            "runtime.launch_gap_short_context",
            ["cpu-submit", "gpu-kernel"],
        ),
        "gpu_idle_gap": ("runtime.gpu_idle_gap", ["gpu-kernel"]),
        "cpu_launch_overhead": ("runtime.cpu_launch_overhead", ["cpu-submit"]),
    },
    "pytorch_profile": {
        "gqa_head_ratio": (
            "framework.gqa_head_ratio",
            ["framework", "gpu-kernel"],
        ),
        "shape_fragmentation": ("framework.shape_fragmentation", ["framework"]),
        "framework_dispatch_overhead": (
            "framework.dispatch_overhead",
            ["framework", "cpu-submit"],
        ),
    },
}
_ACTIVE_ACTIONS = {
    "compiler-sass-inspection": "compiler_sass",
    "direction-experiment-project-copy": "direction_experiment",
    "ncu-targeted-kernel": "ncu_kernel",
    "nsys-global-timeline": "nsys_timeline",
    "nsys-os-runtime-slice": "os_runtime",
    "pytorch-operator-trace": "framework_trace",
}
_KNOWLEDGE_IDENTITY_FIELDS = {
    "schema_version",
    "gpu_architecture",
    "driver_version",
    "cuda_runtime_version",
    "framework_versions",
    "compiler_versions",
    "profiler_versions",
    "workload_contract_sha256",
    "source_sha256",
    "environment_sha256",
}


def _closed(value: object, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strings(
    value: object, label: str, *, nonempty: bool = True, allowed: set[str] | None = None
) -> list[str]:
    if type(value) is not list or (nonempty and not value):
        raise ValueError(f"{label} must be a{' non-empty' if nonempty else ''} string list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    if allowed is not None and not set(value).issubset(allowed):
        raise ValueError(f"{label} contains unsupported values")
    return value


def _canonical_strings(
    value: object,
    label: str,
    *,
    nonempty: bool = True,
    allowed: set[str] | None = None,
) -> list[str]:
    items = _strings(value, label, nonempty=nonempty, allowed=allowed)
    if items != sorted(items):
        raise ValueError(f"{label} must be sorted")
    return items


def _iso_date(value: object, label: str) -> datetime.date:
    try:
        parsed = datetime.date.fromisoformat(_text(value, label))
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must be a valid ISO date")
    return parsed


def _sha256(value: object, label: str) -> str:
    value = _text(value, label)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_reference(reference_dir: Path, filename: str) -> dict:
    root = Path(os.path.abspath(reference_dir))
    try:
        root_status = os.lstat(root)
    except OSError as exc:
        raise ValueError(f"knowledge reference root is unavailable: {root}") from exc
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
        raise ValueError("knowledge reference root is a symlink or unsafe directory")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    root_fd = file_fd = None
    try:
        root_fd = os.open(root, directory_flags)
        file_fd = os.open(filename, flags, dir_fd=root_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_REFERENCE_BYTES:
            raise ValueError(f"knowledge reference is not a bounded regular file: {filename}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, _MAX_REFERENCE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_REFERENCE_BYTES:
                raise ValueError(f"knowledge reference exceeds size limit: {filename}")
        after = os.fstat(file_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"knowledge reference changed while reading: {filename}")
    except OSError as exc:
        raise ValueError(
            f"knowledge reference contains a symlink or unsafe component: {filename}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if root_fd is not None:
            os.close(root_fd)
    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"knowledge reference is invalid JSON: {filename}") from exc
    if type(value) is not dict:
        raise ValueError(f"knowledge reference root must be an object: {filename}")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_sibling(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load knowledge dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_identity_fact(value: object, label: str) -> dict:
    fact = _closed(
        value,
        {"value", "status", "source_kind", "source_sha256"},
        label,
    )
    if fact["status"] == "verified":
        _text(fact["value"], f"{label}.value")
        if fact["source_kind"] == "unknown":
            raise ValueError(f"{label}.source_kind cannot be unknown when verified")
        _text(fact["source_kind"], f"{label}.source_kind")
        _sha256(fact["source_sha256"], f"{label}.source_sha256")
    elif fact["status"] == "unknown":
        if (
            fact["value"] is not None
            or fact["source_kind"] != "unknown"
            or fact["source_sha256"] is not None
        ):
            raise ValueError(f"{label} unknown fact must not invent an identity")
    else:
        raise ValueError(f"{label}.status is unsupported")
    return copy.deepcopy(fact)


def _validate_knowledge_identity(value: object, epoch: Mapping[str, object]) -> dict:
    identity = _closed(value, _KNOWLEDGE_IDENTITY_FIELDS, "knowledge_identity")
    if identity["schema_version"] != "cuda-optimizer/knowledge-identity-v1":
        raise ValueError("knowledge_identity schema is unsupported")
    for field in ("gpu_architecture", "driver_version", "cuda_runtime_version"):
        _validate_identity_fact(identity[field], f"knowledge_identity.{field}")
    for group in ("framework_versions", "compiler_versions", "profiler_versions"):
        versions = identity[group]
        if type(versions) is not dict:
            raise ValueError(f"knowledge_identity.{group} must be an object")
        for name, fact in versions.items():
            _observation_identifier(name, f"knowledge_identity.{group} key")
            _validate_identity_fact(
                fact, f"knowledge_identity.{group}.{name}"
            )
    bindings = {
        "workload_contract_sha256": "workload_contract_sha256",
        "source_sha256": "source_sha256",
        "environment_sha256": "environment_sha256",
    }
    epoch_identities = epoch["identities"]
    for field, epoch_field in bindings.items():
        digest = _sha256(identity[field], f"knowledge_identity.{field}")
        if digest != epoch_identities[epoch_field]:
            raise ValueError(f"knowledge_identity {field} does not match epoch")
    return copy.deepcopy(identity)


def _observation_identifier(value: object, label: str) -> str:
    value = _text(value, label)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _observation_value(value: object, label: str) -> object:
    if value is None or type(value) in {bool, str}:
        return value
    if type(value) in {int, float} and math.isfinite(value):
        return value
    raise ValueError(f"{label} must be a finite scalar")


def _append_observation(observations: dict[tuple, dict], item: dict) -> None:
    key = (
        item["semantic_id"],
        tuple(item["scope"]),
        item["source_digest"],
    )
    previous = observations.get(key)
    if previous is not None and previous != item:
        raise ValueError(
            "semantic observation conflict for the same semantic_id, scope, "
            "and source_digest"
        )
    observations[key] = item


def _validate_observation_rules(value: object, label: str) -> None:
    rules = _closed(
        value,
        {"positive", "counter", "invalidators"},
        label,
    )
    rule_fields = {
        "semantic_id",
        "statuses",
        "scope_all",
        "unit",
        "aggregation",
        "comparison",
    }
    seen_match_keys: dict[tuple, str] = {}
    for group in ("positive", "counter", "invalidators"):
        entries = rules[group]
        if type(entries) is not list:
            raise ValueError(f"{label}.{group} must be an array")
        for index, raw in enumerate(entries):
            rule_label = f"{label}.{group}[{index}]"
            rule = _closed(raw, rule_fields, rule_label)
            semantic_id = _observation_identifier(
                rule["semantic_id"], f"{rule_label}.semantic_id"
            )
            statuses = _strings(
                rule["statuses"],
                f"{rule_label}.statuses",
                allowed={"present", "observed", "absent", "unavailable"},
            )
            if "unavailable" in statuses:
                raise ValueError(
                    f"{rule_label}.statuses cannot contain unavailable"
                )
            scope_all = _strings(rule["scope_all"], f"{rule_label}.scope_all")
            unit = _text(rule["unit"], f"{rule_label}.unit")
            aggregation = _text(
                rule["aggregation"], f"{rule_label}.aggregation"
            )
            comparison = rule["comparison"]
            if comparison is None:
                comparison_key = None
            else:
                comparison = _closed(
                    comparison,
                    {"op", "value"},
                    f"{rule_label}.comparison",
                )
                if comparison["op"] not in {"eq", "lt", "lte", "gt", "gte"}:
                    raise ValueError(f"{rule_label}.comparison op is unsupported")
                if (
                    type(comparison["value"]) not in {int, float}
                    or not math.isfinite(comparison["value"])
                ):
                    raise ValueError(
                        f"{rule_label}.comparison value must be finite"
                    )
                comparison_key = (comparison["op"], comparison["value"])
            match_key = (
                semantic_id,
                tuple(sorted(statuses)),
                tuple(sorted(scope_all)),
                unit,
                aggregation,
                comparison_key,
            )
            if match_key in seen_match_keys:
                raise ValueError(
                    f"{rule_label} duplicates observation rule in "
                    f"{seen_match_keys[match_key]}"
                )
            seen_match_keys[match_key] = f"{label}.{group}"


def _validate_identity_constraints(value: object, label: str) -> None:
    constraints = _closed(
        value,
        {
            "match",
            "gpu_architecture",
            "driver_version",
            "cuda_runtime_version",
            "framework_versions",
            "compiler_versions",
            "profiler_versions",
        },
        label,
    )
    if constraints["match"] != "exact_only":
        raise ValueError(f"{label}.match must be exact_only")
    for field in (
        "gpu_architecture",
        "driver_version",
        "cuda_runtime_version",
    ):
        _strings(
            constraints[field],
            f"{label}.{field}",
            nonempty=False,
        )
    for field in (
        "framework_versions",
        "compiler_versions",
        "profiler_versions",
    ):
        versions = constraints[field]
        if type(versions) is not dict:
            raise ValueError(f"{label}.{field} must be an object")
        for component, allowed_values in versions.items():
            _observation_identifier(component, f"{label}.{field} component")
            _strings(
                allowed_values,
                f"{label}.{field}.{component}",
            )


def normalize_observations(
    *,
    diagnostic_evidence: Sequence[Mapping[str, object]] = (),
    active_evidence_results: Sequence[Mapping[str, object]] = (),
) -> list[dict]:
    """Convert already-validated evidence into stable semantic observations."""
    if isinstance(diagnostic_evidence, (str, bytes)) or not isinstance(
        diagnostic_evidence, Sequence
    ):
        raise ValueError("diagnostic_evidence must be a sequence")
    if isinstance(active_evidence_results, (str, bytes)) or not isinstance(
        active_evidence_results, Sequence
    ):
        raise ValueError("active_evidence_results must be a sequence")

    normalized: dict[tuple, dict] = {}
    diagnostic_fields = {
        "kind",
        "layer",
        "summary",
        "signals",
        "producer",
        "adapter_request_sha256",
        "recorded_at",
        "subject",
        "result",
    }
    for index, raw in enumerate(diagnostic_evidence):
        item = _closed(raw, diagnostic_fields, f"diagnostic evidence {index}")
        kind = item["kind"]
        if kind not in _DIAGNOSTIC_PRODUCERS:
            raise ValueError(f"diagnostic evidence {index} has unknown producer kind")
        producer = _closed(
            item["producer"],
            {"id", "version", "implementation_sha256"},
            f"diagnostic evidence {index} producer",
        )
        if (
            producer["id"] != _DIAGNOSTIC_PRODUCERS[kind]
            or producer["version"] != "1.0.0"
        ):
            raise ValueError(f"diagnostic evidence {index} producer/version is unknown")
        implementation_sha = _sha256(
            producer["implementation_sha256"],
            f"diagnostic evidence {index} producer implementation_sha256",
        )
        request_sha = _sha256(
            item["adapter_request_sha256"],
            f"diagnostic evidence {index} adapter_request_sha256",
        )
        subject = _closed(
            item["subject"],
            {"target_sha256"},
            f"diagnostic evidence {index} subject",
        )
        _sha256(
            subject["target_sha256"],
            f"diagnostic evidence {index} subject target_sha256",
        )
        result = _closed(
            item["result"],
            {"artifact_sha256", "events_total"},
            f"diagnostic evidence {index} result",
        )
        artifact_sha = _sha256(
            result["artifact_sha256"],
            f"diagnostic evidence {index} result artifact_sha256",
        )
        if type(result["events_total"]) is not int or result["events_total"] < 1:
            raise ValueError(
                f"diagnostic evidence {index} result events_total must be positive"
            )
        if item["layer"] != "workload":
            raise ValueError(f"diagnostic evidence {index} layer is unsupported")
        _text(item["summary"], f"diagnostic evidence {index} summary")
        if (
            type(item["recorded_at"]) not in {int, float}
            or not math.isfinite(item["recorded_at"])
            or item["recorded_at"] < 0
        ):
            raise ValueError(f"diagnostic evidence {index} recorded_at is invalid")
        signals = _strings(
            item["signals"],
            f"diagnostic evidence {index} signals",
            nonempty=False,
        )
        unknown_signals = set(signals) - set(_DIAGNOSTIC_SIGNALS[kind])
        if unknown_signals:
            raise ValueError(
                f"diagnostic evidence {index} has unknown signals: "
                f"{sorted(unknown_signals)}"
            )
        source_digest = _canonical_sha256(
            {
                "kind": kind,
                "producer": {
                    "id": producer["id"],
                    "version": producer["version"],
                    "implementation_sha256": implementation_sha,
                },
                "adapter_request_sha256": request_sha,
                "report_artifact_sha256": artifact_sha,
            }
        )
        for signal in signals:
            semantic_id, scope = _DIAGNOSTIC_SIGNALS[kind][signal]
            _append_observation(
                normalized,
                {
                    "semantic_id": semantic_id,
                    "status": "present",
                    "value": True,
                    "unit": "state",
                    "scope": list(scope),
                    "aggregation": "presence",
                    "tool": {
                        "name": producer["id"],
                        "version": producer["version"],
                    },
                    "quality": "validated",
                    "source_digest": source_digest,
                },
            )

    envelope_fields = {
        "action_id",
        "evidence_kind",
        "adapter_implementation_sha256",
        "result_sha256",
        "status",
        "observations",
    }
    semantic_fields = {
        "semantic_id",
        "status",
        "value",
        "unit",
        "scope",
        "aggregation",
        "tool",
        "quality",
    }
    for index, raw in enumerate(active_evidence_results):
        envelope = _closed(raw, envelope_fields, f"active evidence result {index}")
        action_id = _observation_identifier(
            envelope["action_id"], f"active evidence result {index} action_id"
        )
        evidence_kind = _observation_identifier(
            envelope["evidence_kind"],
            f"active evidence result {index} evidence_kind",
        )
        if _ACTIVE_ACTIONS.get(action_id) != evidence_kind:
            raise ValueError(f"active evidence result {index} action identity is unknown")
        _sha256(
            envelope["adapter_implementation_sha256"],
            f"active evidence result {index} adapter identity SHA",
        )
        result_sha = _sha256(
            envelope["result_sha256"],
            f"active evidence result {index} result_sha256",
        )
        if envelope["status"] not in {
            "observed",
            "inconclusive",
            "unavailable",
            "failed",
        }:
            raise ValueError(f"active evidence result {index} status is unsupported")
        observations = envelope["observations"]
        if type(observations) is not dict:
            raise ValueError(f"active evidence result {index} observations must be an object")
        semantic_items = observations.get("semantic_observations")
        if semantic_items is None:
            continue
        if type(semantic_items) is not list:
            raise ValueError(
                f"active evidence result {index} semantic_observations must be an array"
            )
        for semantic_index, raw_semantic in enumerate(semantic_items):
            label = (
                f"active evidence result {index} semantic observation {semantic_index}"
            )
            semantic = _closed(raw_semantic, semantic_fields, label)
            semantic_id = _observation_identifier(
                semantic["semantic_id"], f"{label} semantic_id"
            )
            status = _observation_identifier(semantic["status"], f"{label} status")
            if status not in {"observed", "present", "absent", "unavailable"}:
                raise ValueError(f"{label} status is unsupported")
            value = _observation_value(semantic["value"], f"{label} value")
            unit = _text(semantic["unit"], f"{label} unit")
            scope = _strings(semantic["scope"], f"{label} scope")
            aggregation = _text(
                semantic["aggregation"], f"{label} aggregation"
            )
            tool = _closed(semantic["tool"], {"name", "version"}, f"{label} tool")
            tool_value = {
                "name": _text(tool["name"], f"{label} tool name"),
                "version": _text(tool["version"], f"{label} tool version"),
            }
            quality = _text(semantic["quality"], f"{label} quality")
            if value == "ERR_NVGPUCTRPERM":
                semantic_id = "profile.counter_access"
                status = "unavailable"
                unit = "state"
                scope = ["profile"]
                aggregation = "presence"
            _append_observation(
                normalized,
                {
                    "semantic_id": semantic_id,
                    "status": status,
                    "value": value,
                    "unit": unit,
                    "scope": list(scope),
                    "aggregation": aggregation,
                    "tool": tool_value,
                    "quality": quality,
                    "source_digest": result_sha,
                },
            )

    return [
        normalized[key]
        for key in sorted(
            normalized,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


def _diagnosis_categories(value: object) -> list[str]:
    if type(value) is not dict:
        raise ValueError("diagnosis must be an object")
    categories = []
    primary = value.get("primary_category")
    if isinstance(primary, str):
        categories.append(primary)
    ranked = value.get("ranked_categories")
    if type(ranked) is not list:
        raise ValueError("diagnosis.ranked_categories must be an array")
    for index, item in enumerate(ranked):
        if type(item) is not dict or not isinstance(item.get("category"), str):
            raise ValueError(f"diagnosis.ranked_categories[{index}] is invalid")
        categories.append(item["category"])
    return list(dict.fromkeys(categories or ["unknown"]))


def _effective_categories(
    diagnosis: object, performance_model: Mapping[str, object]
) -> list[str]:
    categories = _diagnosis_categories(diagnosis)
    if categories != ["unknown"]:
        return categories
    inferred = sorted(
        {
            _LAYER_TO_DIAGNOSIS_CATEGORY[item["layer"]]
            for item in performance_model["layer_directions"]
            if item["qualifies_minimum_effect"]
            and item["layer"] in _LAYER_TO_DIAGNOSIS_CATEGORY
        }
    )
    if not inferred:
        return categories
    return (["mixed"] if len(inferred) > 1 else []) + inferred


def _identity_constraint_result(
    constraints: Mapping[str, object], identity: Mapping[str, object]
) -> tuple[str, list[str]]:
    unknown = []
    mismatch = []
    for field in ("gpu_architecture", "driver_version", "cuda_runtime_version"):
        allowed = constraints[field]
        if not allowed:
            continue
        fact = identity[field]
        if fact["status"] != "verified":
            unknown.append(field)
        elif fact["value"] not in allowed:
            mismatch.append(field)
    for group in ("framework_versions", "compiler_versions", "profiler_versions"):
        for component, allowed in constraints[group].items():
            label = f"{group}.{component}"
            fact = identity[group].get(component)
            if fact is None or fact["status"] != "verified":
                unknown.append(label)
            elif fact["value"] not in allowed:
                mismatch.append(label)
    if mismatch:
        return "mismatch", sorted(mismatch)
    if unknown:
        return "unknown", sorted(unknown)
    return "matched", []


def _rule_matches(rule: Mapping[str, object], observation: Mapping[str, object]) -> bool:
    if (
        observation["semantic_id"] != rule["semantic_id"]
        or observation["status"] not in rule["statuses"]
        or observation["unit"] != rule["unit"]
        or observation["aggregation"] != rule["aggregation"]
        or not set(rule["scope_all"]).issubset(observation["scope"])
    ):
        return False
    comparison = rule["comparison"]
    if comparison is None:
        return True
    value = observation["value"]
    if type(value) not in {int, float} or not math.isfinite(value):
        return False
    threshold = comparison["value"]
    return {
        "eq": value == threshold,
        "lt": value < threshold,
        "lte": value <= threshold,
        "gt": value > threshold,
        "gte": value >= threshold,
    }[comparison["op"]]


def _rule_evidence(
    rules: Sequence[Mapping[str, object]],
    observations: Sequence[Mapping[str, object]],
) -> tuple[list[dict], list[dict]]:
    matched = []
    missing = []
    for rule in rules:
        hits = [item for item in observations if _rule_matches(rule, item)]
        if hits:
            matched.extend(
                {
                    "semantic_id": item["semantic_id"],
                    "source_digest": item["source_digest"],
                }
                for item in hits
            )
        else:
            missing.append({"semantic_id": rule["semantic_id"]})
    unique_matches = {
        (item["semantic_id"], item["source_digest"]): item for item in matched
    }
    unique_missing = {item["semantic_id"]: item for item in missing}
    return (
        [unique_matches[key] for key in sorted(unique_matches)],
        [unique_missing[key] for key in sorted(unique_missing)],
    )


def _dominates(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_dims = left["_dimensions"]
    right_dims = right["_dimensions"]
    no_worse = (
        left_dims["benefit"] >= right_dims["benefit"]
        and left_dims["grade"] >= right_dims["grade"]
        and left_dims["cost"] <= right_dims["cost"]
        and left_dims["risk"] <= right_dims["risk"]
    )
    strictly_better = (
        left_dims["benefit"] > right_dims["benefit"]
        or left_dims["grade"] > right_dims["grade"]
        or left_dims["cost"] < right_dims["cost"]
        or left_dims["risk"] < right_dims["risk"]
    )
    return no_worse and strictly_better


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_action_timing_estimates(value: object) -> None:
    if type(value) is not dict:
        raise ValueError("performance_model.action_timing_estimates must be an object")
    for action_id, raw in value.items():
        _observation_identifier(
            action_id, "performance_model.action_timing_estimates action_id"
        )
        estimate = _closed(
            raw,
            {"sample_count", "p50_seconds", "p90_seconds", "basis"},
            f"performance_model.action_timing_estimates.{action_id}",
        )
        if (
            type(estimate["sample_count"]) is not int
            or isinstance(estimate["sample_count"], bool)
            or estimate["sample_count"] < 1
        ):
            raise ValueError(f"action timing {action_id} sample_count is invalid")
        for field in ("p50_seconds", "p90_seconds"):
            if (
                type(estimate[field]) not in {int, float}
                or not math.isfinite(estimate[field])
                or estimate[field] <= 0
            ):
                raise ValueError(f"action timing {action_id} {field} is invalid")
        if estimate["p90_seconds"] < estimate["p50_seconds"]:
            raise ValueError(f"action timing {action_id} percentile order is invalid")
        if estimate["basis"] != "identity_matched_history":
            raise ValueError(f"action timing {action_id} basis is unsupported")


def build_knowledge_context(
    frozen_inputs: Mapping[str, object],
    *,
    limit: int = 3,
    max_bytes: int = 12 * 1024,
) -> dict:
    """Return a deterministic, identity-bound, evidence-only knowledge context."""
    if type(frozen_inputs) is not dict:
        raise ValueError("frozen_inputs must be an object")
    fields = {
        "knowledge_identity",
        "diagnosis",
        "analysis_epoch",
        "evidence_catalog",
        "execution_map",
        "performance_model",
        "diagnostic_evidence",
        "active_evidence_results",
        "requested_claim",
        "ready_capability_ids",
        "contract_action_ids",
        "available_actions",
        "closed_mechanism_keys",
        "candidate_history",
    }
    _closed(frozen_inputs, fields, "frozen_inputs")
    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 3:
        raise ValueError("limit must be between 1 and 3")
    if (
        type(max_bytes) is not int
        or isinstance(max_bytes, bool)
        or max_bytes < 1024
    ):
        raise ValueError("max_bytes must be an integer of at least 1024")

    epoch_module = _load_sibling(
        "analysis_epoch.py", "cuda_optimizer_epoch_knowledge_runtime"
    )
    map_module = _load_sibling(
        "execution_map.py", "cuda_optimizer_map_knowledge_runtime"
    )
    model_module = _load_sibling(
        "performance_model.py", "cuda_optimizer_model_knowledge_runtime"
    )
    hypothesis_module = _load_sibling(
        "hypothesis_space.py", "cuda_optimizer_hypothesis_knowledge_runtime"
    )
    epoch = epoch_module.validate_epoch(frozen_inputs["analysis_epoch"])
    identity = _validate_knowledge_identity(
        frozen_inputs["knowledge_identity"], epoch
    )
    unverified_core_identity_fields = [
        field
        for field in ("gpu_architecture", "cuda_runtime_version")
        if identity[field]["status"] != "verified"
    ]
    identity_sha = _canonical_sha256(identity)
    validated_map = map_module.validate_execution_map(
        frozen_inputs["execution_map"],
        epoch=epoch,
        evidence_catalog=frozen_inputs["evidence_catalog"],
    )["execution_map"]
    map_module.execution_map_digest(
        validated_map,
        epoch=epoch,
        evidence_catalog=frozen_inputs["evidence_catalog"],
    )
    supplied_model = frozen_inputs["performance_model"]
    if type(supplied_model) is not dict:
        raise ValueError("performance_model must be an object")
    minimum_effect = supplied_model.get("minimum_effect_us")
    rebuilt_model = model_module.build_performance_model(
        validated_map,
        minimum_effect_us=minimum_effect,
    )
    _validate_action_timing_estimates(
        supplied_model.get("action_timing_estimates")
    )
    comparable_model = copy.deepcopy(supplied_model)
    comparable_model["action_timing_estimates"] = {}
    if _canonical_sha256(rebuilt_model) != _canonical_sha256(comparable_model):
        raise ValueError("performance_model does not match execution_map")
    performance_model_sha = _canonical_sha256(supplied_model)

    package = validate_knowledge_package(REFERENCE_DIR)
    source_document = _read_reference(REFERENCE_DIR, "knowledge_sources.json")
    card_document = _read_reference(REFERENCE_DIR, "diagnostic_cards.json")
    case_document = _read_reference(REFERENCE_DIR, "case_memory.json")
    action_document = _read_reference(
        REFERENCE_DIR, "evidence_action_catalog.json"
    )
    reread_package = {
        "sources": source_document,
        "cards": card_document,
        "cases": case_document,
        "actions": action_document,
    }
    if _canonical_sha256(reread_package) != package["content_sha256"]:
        raise ValueError("knowledge package changed after validation")
    cards = card_document["cards"]
    cases = {item["id"]: item for item in case_document["cases"]}
    actions = {item["action_id"]: item for item in action_document["actions"]}
    eligible_card_ids = set(package["runtime_candidate_card_ids"])

    requested_claim = _text(
        frozen_inputs["requested_claim"], "frozen_inputs.requested_claim"
    )
    if requested_claim not in _REQUESTED_CLAIMS:
        raise ValueError("frozen_inputs.requested_claim is unsupported")
    ready_capability_ids = set(
        _canonical_strings(
            frozen_inputs["ready_capability_ids"],
            "frozen_inputs.ready_capability_ids",
            nonempty=False,
        )
    )
    contract_action_ids = set(
        _canonical_strings(
            frozen_inputs["contract_action_ids"],
            "frozen_inputs.contract_action_ids",
            nonempty=False,
        )
    )
    unknown_contract_actions = contract_action_ids - set(actions)
    if unknown_contract_actions:
        raise ValueError(
            f"unknown contract action ids: {sorted(unknown_contract_actions)}"
        )
    available_actions = _canonical_strings(
        frozen_inputs["available_actions"],
        "frozen_inputs.available_actions",
        nonempty=False,
    )
    available_action_ids = set(available_actions)
    unknown_available_actions = available_action_ids - set(actions)
    if unknown_available_actions:
        raise ValueError(
            f"unknown available action ids: {sorted(unknown_available_actions)}"
        )
    derived_available_actions = sorted(
        action_id
        for action_id, action in actions.items()
        if action_id in contract_action_ids
        and action["control_scope"] == "read_only"
        and set(action["required_capability_ids"]).issubset(ready_capability_ids)
    )
    if available_actions != derived_available_actions:
        raise ValueError("available_actions must equal derived availability")
    closed_keys = {
        hypothesis_module.canonical_mechanism_key(item)
        for item in _strings(
            frozen_inputs["closed_mechanism_keys"],
            "frozen_inputs.closed_mechanism_keys",
            nonempty=False,
        )
    }
    history = frozen_inputs["candidate_history"]
    if type(history) is not list:
        raise ValueError("candidate_history must be an array")
    for index, item in enumerate(history):
        item = _closed(
            item,
            {
                "hypothesis_id",
                "action_id",
                "implementation_status",
                "identity_digest",
                "elapsed_seconds",
                "candidate_digest",
                "decision_digest",
                "failure_reason",
            },
            f"candidate_history[{index}]",
        )
        for field in ("hypothesis_id", "action_id", "failure_reason"):
            _observation_identifier(
                item[field], f"candidate_history[{index}].{field}"
            )
        if item["implementation_status"] != "failed":
            raise ValueError(
                f"candidate_history[{index}].implementation_status must be failed"
            )
        for field in ("identity_digest", "candidate_digest", "decision_digest"):
            _sha256(item[field], f"candidate_history[{index}].{field}")
        elapsed = item["elapsed_seconds"]
        if (
            type(elapsed) not in {int, float}
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(
                f"candidate_history[{index}].elapsed_seconds is invalid"
            )

    observations = normalize_observations(
        diagnostic_evidence=frozen_inputs["diagnostic_evidence"],
        active_evidence_results=frozen_inputs["active_evidence_results"],
    )
    evidence_sha = _canonical_sha256(observations)
    categories = _effective_categories(
        frozen_inputs["diagnosis"], rebuilt_model
    )
    observed_layers = {
        item["layer"]
        for item in validated_map["coverage"]
        if item["status"] == "observed"
    }
    layer_directions = {
        item["layer"]: item for item in rebuilt_model["layer_directions"]
    }
    cost_rank = {"low": 0, "medium": 1, "high": 2}
    risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    filtered_counts = {
        "category_mismatch": 0,
        "task_mismatch": 0,
        "identity_mismatch": 0,
        "identity_unverified": 0,
        "closed_mechanism": 0,
        "execution_layer_unobserved": 0,
        "invalidator_observed": 0,
        "positive_observation_missing": 0,
        "exact_case_rejection": 0,
        "content_status": 0,
        "read_only_action_unavailable": 0,
        "below_minimum_effect": 0,
        "canonical_duplicate": 0,
        "pareto_dominated": 0,
        "byte_trimmed": 0,
    }
    explanations = []
    rejections = []
    records = []

    for card in cards:
        canonical_key = hypothesis_module.canonical_mechanism_key(
            card["mechanism_key"]
        )
        base = {"card_id": card["id"], "mechanism_key": canonical_key}
        if canonical_key in closed_keys:
            rejections.append({**base, "reason": "closed_mechanism", "details": []})
            filtered_counts["closed_mechanism"] += 1
            continue
        if not set(card["categories"]) & set(categories):
            rejections.append({**base, "reason": "category_mismatch", "details": []})
            filtered_counts["category_mismatch"] += 1
            continue
        if (
            card["requested_claims"]
            and requested_claim not in card["requested_claims"]
        ):
            rejections.append(
                {
                    **base,
                    "reason": "task_mismatch",
                    "details": [requested_claim],
                }
            )
            filtered_counts["task_mismatch"] += 1
            continue
        if unverified_core_identity_fields:
            explanations.append(
                {
                    **base,
                    "reason": "identity_unverified",
                    "details": copy.deepcopy(unverified_core_identity_fields),
                }
            )
            filtered_counts["identity_unverified"] += 1
            continue
        identity_status, identity_details = _identity_constraint_result(
            card["identity_constraints"], identity
        )
        if identity_status == "mismatch":
            rejections.append(
                {**base, "reason": "identity_mismatch", "details": identity_details}
            )
            filtered_counts["identity_mismatch"] += 1
            continue
        if identity_status == "unknown":
            explanations.append(
                {
                    **base,
                    "reason": "identity_unverified",
                    "details": identity_details,
                }
            )
            filtered_counts["identity_unverified"] += 1
            continue
        if not set(card["execution_layers"]) & observed_layers:
            rejections.append(
                {
                    **base,
                    "reason": "execution_layer_unobserved",
                    "details": [],
                }
            )
            filtered_counts["execution_layer_unobserved"] += 1
            continue
        positive, missing = _rule_evidence(
            card["observation_rules"]["positive"], observations
        )
        counter, _ = _rule_evidence(
            card["observation_rules"]["counter"], observations
        )
        invalidators, _ = _rule_evidence(
            card["observation_rules"]["invalidators"], observations
        )
        if invalidators:
            rejections.append(
                {
                    **base,
                    "reason": "invalidator_observed",
                    "details": sorted(
                        {item["semantic_id"] for item in invalidators}
                    ),
                }
            )
            filtered_counts["invalidator_observed"] += 1
            continue
        card_cases = [cases[case_id] for case_id in card["case_ids"]]
        exact_rejection_ids = sorted(
            item["id"]
            for item in card_cases
            if item["identity_match"] == "exact"
            and item["knowledge_identity_sha256"] == identity_sha
            and item["content_status"] == "locally_measured"
            and item["outcome_type"] == "rejection"
        )
        if exact_rejection_ids:
            rejections.append(
                {
                    **base,
                    "reason": "exact_case_rejection",
                    "details": exact_rejection_ids,
                }
            )
            filtered_counts["exact_case_rejection"] += 1
            continue

        action_id = card["cheapest_falsifier"]["action_id"]
        action = actions[action_id]
        action_available = (
            action_id in available_action_ids
            and action["control_scope"] == "read_only"
        )
        candidate_eligible = card["id"] in eligible_card_ids
        if not candidate_eligible:
            record_kind = "explanation"
            explanation_reason = "content_status"
            explanation_details = [card["content_status"]]
        elif card["observation_rules"]["positive"] and not positive:
            record_kind = "explanation"
            explanation_reason = "positive_observation_missing"
            explanation_details = sorted(
                item["semantic_id"] for item in missing
            )
        elif not action_available:
            record_kind = "explanation"
            explanation_reason = "read_only_action_unavailable"
            explanation_details = [action_id]
        else:
            record_kind = "candidate"
            explanation_reason = None
            explanation_details = []

        exact_case_ids = sorted(
            item["id"]
            for item in card_cases
            if item["identity_match"] == "exact"
            and item["knowledge_identity_sha256"] == identity_sha
            and item["content_status"] == "locally_measured"
        )
        analogous_case_ids = sorted(
            item["id"] for item in card_cases if item["identity_match"] == "analogous"
        )
        if positive and exact_case_ids:
            grade, grade_name = 4, "current_local_exact_case"
        elif positive:
            grade, grade_name = 3, "current_local"
        elif missing:
            grade, grade_name = 2, "plausible_missing_observation"
        else:
            grade, grade_name = 1, "source_or_analogous_case_only"
        layer_options = [
            layer_directions[layer]
            for layer in card["execution_layers"]
            if layer in layer_directions
        ]
        layer_options.sort(
            key=lambda item: (-item["benefit_ceiling_us"], item["layer"])
        )
        direction = layer_options[0] if layer_options else None
        benefit = float(direction["benefit_ceiling_us"]) if direction else 0.0
        qualifies_minimum_effect = bool(
            direction and direction["qualifies_minimum_effect"]
        )
        if record_kind == "candidate" and not qualifies_minimum_effect:
            record_kind = "explanation"
            explanation_reason = "below_minimum_effect"
            explanation_details = [
                direction["layer"] if direction else "no_current_direction"
            ]
        scope_node_ids = sorted(
            item["node_id"]
            for item in validated_map["nodes"]
            if direction is not None
            and item["layer"] == direction["layer"]
            and item["node_id"] in validated_map["hot_path"]
        )
        benefit_reference = {
            "performance_model_sha256": performance_model_sha,
            "layer": direction["layer"] if direction else None,
            "benefit_ceiling_us": benefit,
            "qualifies_minimum_effect": qualifies_minimum_effect,
            "basis": direction["basis"] if direction else "no_current_direction",
        }
        record = {
            **base,
            "statement": card["mechanism"],
            "execution_layers": list(card["execution_layers"]),
            "scope_node_ids": scope_node_ids,
            "content_status": card["content_status"],
            "confidence": "inconclusive",
            "promotion_authority": "none",
            "evidence_grade": grade_name,
            "evidence": {
                "positive": positive,
                "counter": counter,
                "missing": missing,
            },
            "case_support": {
                "exact": exact_case_ids,
                "analogous": analogous_case_ids,
            },
            "benefit_ceiling": benefit_reference,
            "cheapest_falsifier": {
                "action_id": action_id,
                "rationale": card["cheapest_falsifier"]["rationale"],
                "cost": action["cost"],
                "risk": action["risk"],
                "control_scope": action["control_scope"],
            },
            "source_ids": list(card["source_ids"]),
            "case_ids": list(card["case_ids"]),
            "_kind": record_kind,
            "_explanation_reason": explanation_reason,
            "_explanation_details": explanation_details,
            "_dimensions": {
                "benefit": benefit,
                "grade": grade,
                "cost": cost_rank[action["cost"]],
                "risk": risk_rank[action["risk"]],
            },
        }
        records.append(record)

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["mechanism_key"], []).append(record)
    deduplicated = []
    for key in sorted(grouped):
        group = grouped[key]
        group.sort(
            key=lambda item: (
                0 if item["_kind"] == "candidate" else 1,
                -item["_dimensions"]["grade"],
                item["card_id"],
            )
        )
        deduplicated.append(group[0])
        for duplicate in group[1:]:
            rejections.append(
                {
                    "card_id": duplicate["card_id"],
                    "mechanism_key": key,
                    "reason": "canonical_duplicate",
                    "details": [group[0]["card_id"]],
                }
            )
            filtered_counts["canonical_duplicate"] += 1

    for record in deduplicated:
        if record["_kind"] != "explanation":
            continue
        reason = record["_explanation_reason"]
        explanations.append(
            {
                "card_id": record["card_id"],
                "mechanism_key": record["mechanism_key"],
                "reason": reason,
                "details": record["_explanation_details"],
            }
        )
        filtered_counts[reason] += 1

    candidate_records = [
        item for item in deduplicated if item["_kind"] == "candidate"
    ]
    frontier = []
    for candidate in candidate_records:
        dominators = [
            other
            for other in candidate_records
            if other is not candidate and _dominates(other, candidate)
        ]
        if dominators:
            rejections.append(
                {
                    "card_id": candidate["card_id"],
                    "mechanism_key": candidate["mechanism_key"],
                    "reason": "pareto_dominated",
                    "details": sorted(item["card_id"] for item in dominators),
                }
            )
            filtered_counts["pareto_dominated"] += 1
        else:
            frontier.append(candidate)
    frontier.sort(key=lambda item: (item["mechanism_key"], item["card_id"]))
    selected = []
    used_layers = set()
    for candidate in frontier:
        layer = candidate["benefit_ceiling"]["layer"]
        if layer not in used_layers:
            selected.append(candidate)
            used_layers.add(layer)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for candidate in frontier:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == limit:
                break

    public_candidates = []
    for candidate in selected:
        public = {
            key: copy.deepcopy(value)
            for key, value in candidate.items()
            if not key.startswith("_")
        }
        public_candidates.append(public)
    explanations.sort(key=lambda item: (item["mechanism_key"], item["card_id"]))
    rejections.sort(
        key=lambda item: (item["reason"], item["mechanism_key"], item["card_id"])
    )
    context = {
        "schema_version": "cuda-optimizer/knowledge-context-v1",
        "input_sha256": _canonical_sha256(frozen_inputs),
        "knowledge_package": {
            "version": package["package_version"],
            "sha256": package["content_sha256"],
        },
        "knowledge_identity_sha256": identity_sha,
        "evidence_sha256": evidence_sha,
        "performance_model_sha256": performance_model_sha,
        "categories": categories,
        "candidates": public_candidates,
        "explanations": explanations,
        "rejections": rejections,
        "filtered_counts": filtered_counts,
        "promotion_authority": "none",
    }
    while len(_canonical_bytes(context)) > max_bytes:
        removed = False
        for field in ("explanations", "rejections", "candidates"):
            if context[field]:
                context[field].pop()
                context["filtered_counts"]["byte_trimmed"] += 1
                removed = True
                break
        if not removed:
            raise ValueError("max_bytes is too small for the closed context")
    return context


def validate_knowledge_package(reference_dir: Path = REFERENCE_DIR) -> dict:
    """Read and cross-check the offline knowledge references without side effects."""
    sources = _read_reference(reference_dir, "knowledge_sources.json")
    cards = _read_reference(reference_dir, "diagnostic_cards.json")
    cases = _read_reference(reference_dir, "case_memory.json")
    actions = _read_reference(reference_dir, "evidence_action_catalog.json")

    _closed(sources, {"schema_version", "as_of", "staleness_policy", "sources"}, "knowledge sources")
    if sources["schema_version"] != "cuda-optimizer/knowledge-sources-v1":
        raise ValueError("knowledge source schema is unsupported")
    source_as_of = _iso_date(sources["as_of"], "knowledge sources as_of")
    _text(sources["staleness_policy"], "knowledge sources staleness_policy")
    source_items = sources["sources"]
    if type(source_items) is not list or len(source_items) != 14:
        raise ValueError("knowledge sources must contain exactly 14 entries")
    source_fields = {
        "id",
        "title",
        "url",
        "version",
        "source_kind",
        "locator",
        "summary",
        "summary_sha256",
        "last_verified",
        "status",
    }
    source_ids: set[str] = set()
    for index, item in enumerate(source_items):
        item = _closed(item, source_fields, f"knowledge source {index}")
        source_id = _text(item["id"], f"knowledge source {index} id")
        if source_id in source_ids:
            raise ValueError(f"duplicate knowledge source id: {source_id}")
        source_ids.add(source_id)
        for field in ("title", "version", "locator", "summary"):
            _text(item[field], f"knowledge source {source_id} {field}")
        if not _text(item["url"], f"knowledge source {source_id} url").startswith(
            ("https://", "http://")
        ):
            raise ValueError(f"knowledge source {source_id} url must be HTTP(S)")
        if item["source_kind"] != "primary":
            raise ValueError(f"knowledge source {source_id} must be primary")
        if item["status"] != "verified":
            raise ValueError(f"knowledge source {source_id} status must be verified")
        expected_summary_sha = hashlib.sha256(item["summary"].encode("utf-8")).hexdigest()
        if _sha256(
            item["summary_sha256"], f"knowledge source {source_id} summary_sha256"
        ) != expected_summary_sha:
            raise ValueError(f"knowledge source {source_id} summary_sha256 does not match")
        if _iso_date(
            item["last_verified"], f"knowledge source {source_id} last_verified"
        ) > source_as_of:
            raise ValueError(f"knowledge source {source_id} is verified after as_of")

    _closed(actions, {"schema_version", "catalog_id", "actions"}, "evidence action catalog")
    if actions["schema_version"] != "cuda-optimizer/evidence-action-catalog-v1":
        raise ValueError("evidence action catalog schema is unsupported")
    action_items = actions["actions"]
    if type(action_items) is not list or not action_items:
        raise ValueError("evidence action catalog is empty")
    action_fields = {
        "action_id",
        "evidence_kind",
        "required_capability_ids",
        "cost",
        "perturbation",
        "risk",
        "control_scope",
        "repeatable",
    }
    action_scopes: dict[str, str] = {}
    for index, item in enumerate(action_items):
        item = _closed(item, action_fields, f"evidence action {index}")
        action_id = _text(item["action_id"], f"evidence action {index} action_id")
        if action_id in action_scopes:
            raise ValueError(f"duplicate evidence action id: {action_id}")
        _text(item["evidence_kind"], f"evidence action {action_id} evidence_kind")
        _strings(
            item["required_capability_ids"],
            f"evidence action {action_id} required_capability_ids",
        )
        for field in ("cost", "perturbation", "risk"):
            _text(item[field], f"evidence action {action_id} {field}")
        if item["control_scope"] not in {"read_only", "project_copy"}:
            raise ValueError(f"evidence action {action_id} control_scope is unsupported")
        if type(item["repeatable"]) is not bool:
            raise ValueError(f"evidence action {action_id} repeatable must be boolean")
        action_scopes[action_id] = item["control_scope"]

    _closed(cases, {"schema_version", "cases"}, "case memory")
    if cases["schema_version"] != "cuda-optimizer/case-memory-v1":
        raise ValueError("case memory schema is unsupported")
    case_items = cases["cases"]
    if type(case_items) is not list or not case_items:
        raise ValueError("case memory is empty")
    case_fields = {
        "id",
        "replay_case_id",
        "replay_status",
        "replay_case_sha256",
        "identity_match",
        "predecision_evidence_sha256",
        "knowledge_identity_sha256",
        "outcome_type",
        "source_ids",
        "content_status",
    }
    case_by_id: dict[str, dict] = {}
    for index, item in enumerate(case_items):
        item = _closed(item, case_fields, f"case memory entry {index}")
        case_id = _text(item["id"], f"case memory entry {index} id")
        if case_id in case_by_id:
            raise ValueError(f"duplicate case memory id: {case_id}")
        case_by_id[case_id] = item
        if item["replay_case_id"] != case_id:
            raise ValueError("case memory ids must match replay_case_id")
        if item["replay_status"] not in {
            "scoreable",
            "package_regression",
            "partial",
            "rejection_only",
            "protocol_only",
        }:
            raise ValueError(f"case memory {case_id} replay_status is unsupported")
        _sha256(item["replay_case_sha256"], f"case memory {case_id} replay_case_sha256")
        _sha256(
            item["predecision_evidence_sha256"],
            f"case memory {case_id} predecision_evidence_sha256",
        )
        if item["knowledge_identity_sha256"] is not None:
            _sha256(
                item["knowledge_identity_sha256"],
                f"case memory {case_id} knowledge_identity_sha256",
            )
        if item["identity_match"] not in {"unknown", "analogous", "exact"}:
            raise ValueError(f"case memory {case_id} identity_match is unsupported")
        if item["outcome_type"] not in {
            "historical_outcome",
            "rejection",
            "protocol_only",
        }:
            raise ValueError(f"case memory {case_id} outcome_type is unsupported")
        referenced_sources = _strings(
            item["source_ids"], f"case memory {case_id} source_ids"
        )
        if not set(referenced_sources).issubset(source_ids):
            raise ValueError(f"case memory {case_id} references unknown source")
        if item["content_status"] not in {
            "source_verified",
            "replay_verified",
            "locally_measured",
        }:
            raise ValueError(f"case memory {case_id} content_status is unsupported")
        if (
            item["content_status"] in {"replay_verified", "locally_measured"}
            and item["replay_status"]
            not in {"scoreable", "package_regression", "rejection_only"}
        ):
            raise ValueError(
                f"case memory {case_id} {item['content_status']} requires a "
                "supporting or rejecting replay"
            )
        if item["content_status"] == "locally_measured" and (
            item["identity_match"] != "exact"
            or item["knowledge_identity_sha256"] is None
        ):
            raise ValueError(
                f"case memory {case_id} locally_measured requires exact identity "
                "and knowledge_identity_sha256"
            )
    _closed(cards, {"schema_version", "as_of", "cards"}, "diagnostic cards")
    if cards["schema_version"] != "cuda-optimizer/diagnostic-cards-v1":
        raise ValueError("diagnostic card schema is unsupported")
    cards_as_of = _iso_date(cards["as_of"], "diagnostic cards as_of")
    card_items = cards["cards"]
    if type(card_items) is not list or not card_items:
        raise ValueError("diagnostic card registry is empty")
    card_fields = {
        "id",
        "mechanism_key",
        "status",
        "content_status",
        "observation_rules",
        "identity_constraints",
        "requested_claims",
        "categories",
        "execution_layers",
        "priority",
        "match_terms",
        "applies_when",
        "mechanism",
        "competing_explanations",
        "distinguishing_question",
        "preferred_actions",
        "cheapest_falsifier",
        "required_evidence",
        "positive_signals",
        "counter_signals",
        "invalidators",
        "source_ids",
        "case_ids",
        "last_reviewed",
    }
    card_ids: set[str] = set()
    mechanism_keys: set[str] = set()
    runtime_candidates = []
    for index, item in enumerate(card_items):
        item = _closed(item, card_fields, f"diagnostic card {index}")
        card_id = _text(item["id"], f"diagnostic card {index} id")
        mechanism_key = _text(
            item["mechanism_key"], f"diagnostic card {card_id} mechanism_key"
        )
        if card_id in card_ids or mechanism_key in mechanism_keys:
            raise ValueError("diagnostic card ids and mechanism_key values must be unique")
        card_ids.add(card_id)
        mechanism_keys.add(mechanism_key)
        if item["status"] != "routing_only":
            raise ValueError(f"diagnostic card {card_id} status must preserve routing_only")
        if item["content_status"] not in {
            "source_verified",
            "replay_verified",
            "locally_measured",
        }:
            raise ValueError(f"diagnostic card {card_id} content_status is unsupported")
        _validate_observation_rules(
            item["observation_rules"],
            f"diagnostic card {card_id} observation_rules",
        )
        _validate_identity_constraints(
            item["identity_constraints"],
            f"diagnostic card {card_id} identity_constraints",
        )
        _canonical_strings(
            item["requested_claims"],
            f"diagnostic card {card_id} requested_claims",
            nonempty=False,
            allowed=_REQUESTED_CLAIMS,
        )
        _strings(item["categories"], f"diagnostic card {card_id} categories")
        _strings(
            item["execution_layers"],
            f"diagnostic card {card_id} execution_layers",
            allowed=_LAYERS,
        )
        if type(item["priority"]) is not int or isinstance(item["priority"], bool):
            raise ValueError(f"diagnostic card {card_id} priority must be an integer")
        _strings(
            item["match_terms"],
            f"diagnostic card {card_id} match_terms",
            nonempty=False,
        )
        for field in (
            "applies_when",
            "competing_explanations",
            "required_evidence",
            "positive_signals",
            "counter_signals",
            "invalidators",
        ):
            _strings(item[field], f"diagnostic card {card_id} {field}")
        for field in ("mechanism", "distinguishing_question"):
            _text(item[field], f"diagnostic card {card_id} {field}")
        preferred_actions = _strings(
            item["preferred_actions"], f"diagnostic card {card_id} preferred_actions"
        )
        if not set(preferred_actions).issubset(action_scopes):
            raise ValueError(f"diagnostic card {card_id} references unknown action")
        falsifier = _closed(
            item["cheapest_falsifier"],
            {"action_id", "rationale"},
            f"diagnostic card {card_id} cheapest_falsifier",
        )
        falsifier_id = _text(
            falsifier["action_id"],
            f"diagnostic card {card_id} cheapest_falsifier action_id",
        )
        if falsifier_id not in action_scopes:
            raise ValueError(f"diagnostic card {card_id} references unknown action")
        if action_scopes[falsifier_id] != "read_only":
            raise ValueError(f"diagnostic card {card_id} cheapest falsifier must be read_only")
        _text(
            falsifier["rationale"],
            f"diagnostic card {card_id} cheapest_falsifier rationale",
        )
        referenced_sources = _strings(
            item["source_ids"], f"diagnostic card {card_id} source_ids"
        )
        if not set(referenced_sources).issubset(source_ids):
            raise ValueError(f"diagnostic card {card_id} references unknown source")
        referenced_cases = _strings(
            item["case_ids"], f"diagnostic card {card_id} case_ids", nonempty=False
        )
        if not set(referenced_cases).issubset(case_by_id):
            raise ValueError(f"diagnostic card {card_id} references unknown case")
        if item["content_status"] == "replay_verified":
            eligible_replays = [
                case_by_id[case_id]
                for case_id in referenced_cases
                if (
                    case_by_id[case_id]["replay_status"]
                    in {"scoreable", "package_regression", "rejection_only"}
                    and case_by_id[case_id]["content_status"]
                    in {"replay_verified", "locally_measured"}
                )
            ]
            if not eligible_replays:
                raise ValueError(
                    f"diagnostic card {card_id} replay_verified requires a "
                    "non-protocol supporting or rejecting replay"
                )
            runtime_candidates.append(card_id)
        if item["content_status"] == "locally_measured":
            locally_measured_cases = [
                case_by_id[case_id]
                for case_id in referenced_cases
                if (
                    case_by_id[case_id]["content_status"] == "locally_measured"
                    and case_by_id[case_id]["replay_status"]
                    in {"scoreable", "package_regression", "rejection_only"}
                    and case_by_id[case_id]["identity_match"] == "exact"
                    and case_by_id[case_id]["knowledge_identity_sha256"] is not None
                )
            ]
            if not locally_measured_cases:
                raise ValueError(
                    f"diagnostic card {card_id} locally_measured requires an exact "
                    "locally_measured case with a complete identity digest"
                )
            runtime_candidates.append(card_id)
        if _iso_date(
            item["last_reviewed"], f"diagnostic card {card_id} last_reviewed"
        ) > cards_as_of:
            raise ValueError(f"diagnostic card {card_id} is reviewed after as_of")

    package = {"sources": sources, "cards": cards, "cases": cases, "actions": actions}
    return {
        "status": "passed",
        "package_version": "cuda-optimizer/knowledge-package-v1",
        "source_count": len(source_items),
        "card_count": len(card_items),
        "case_count": len(case_items),
        "runtime_candidate_card_ids": sorted(runtime_candidates),
        "content_sha256": _canonical_sha256(package),
    }


def _load_cards() -> list[dict]:
    value = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "cuda-optimizer/diagnostic-cards-v1":
        raise ValueError("diagnostic card registry schema is unsupported")
    cards = value.get("cards")
    if type(cards) is not list or not cards:
        raise ValueError("diagnostic card registry is empty")
    return cards


def route_cards(
    diagnosis: Mapping[str, object],
    execution_map: Mapping[str, object],
    *,
    limit: int = 3,
) -> dict:
    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= 5:
        raise ValueError("limit must be between 1 and 5")
    primary = diagnosis.get("primary_category")
    categories = []
    if isinstance(primary, str) and primary != "mixed":
        categories.append(primary)
    ranked = diagnosis.get("ranked_categories", [])
    if type(ranked) is list:
        categories.extend(
            item.get("category")
            for item in ranked
            if type(item) is dict and isinstance(item.get("category"), str)
        )
    if not categories:
        categories = ["unknown"]
    elif primary == "mixed":
        categories.insert(0, "mixed")
    categories = list(dict.fromkeys(categories))
    labels = " ".join(
        str(item.get("label", "")).lower()
        for item in execution_map.get("nodes", [])
        if type(item) is dict
    )
    ranked_cards = []
    for card in _load_cards():
        category_rank = min(
            (
                categories.index(category)
                for category in card["categories"]
                if category in categories
            ),
            default=99,
        )
        if category_rank == 99:
            continue
        term_match = any(term in labels for term in card["match_terms"])
        ranked_cards.append(
            ((category_rank, 0 if term_match else 1, card["priority"], card["id"]), card)
        )
    if not ranked_cards:
        fallback = next(
            card for card in _load_cards() if card["id"] == "diagnostic.cross-layer.triage"
        )
        ranked_cards = [((0, 0, 0, fallback["id"]), fallback)]
    ranked_cards.sort(key=lambda item: item[0])
    return {
        "schema_version": "cuda-optimizer/diagnostic-knowledge-context-v1",
        "categories": categories,
        "cards": [copy.deepcopy(card) for _, card in ranked_cards[:limit]],
        "promotion_authority": "none",
    }
