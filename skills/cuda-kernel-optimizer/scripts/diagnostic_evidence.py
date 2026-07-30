#!/usr/bin/env python3
"""Validate Controller-derived PyTorch and Nsys diagnostic observations."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any


EVIDENCE_SCHEMA = "cuda-optimizer/diagnostic-evidence-v1"
MEASUREMENT_SCHEMA = "cuda-optimizer/diagnostic-measurement-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PRODUCERS = {
    "nsys_timeline": "nsys-timeline-adapter",
    "pytorch_profile": "pytorch-profile-adapter",
}
_DIAGNOSTIC_SIGNAL_SEMANTICS = {
    "nsys_timeline": {
        "launch_gap_short_context": {
            "semantic_id": "runtime.launch_gap_short_context",
            "scope": ["cpu-submit", "gpu-kernel"],
        },
        "gpu_idle_gap": {
            "semantic_id": "runtime.gpu_idle_gap",
            "scope": ["gpu-kernel"],
        },
        "cpu_launch_overhead": {
            "semantic_id": "runtime.cpu_launch_overhead",
            "scope": ["cpu-submit"],
        },
    },
    "pytorch_profile": {
        "gqa_head_ratio": {
            "semantic_id": "framework.gqa_head_ratio",
            "scope": ["framework", "gpu-kernel"],
        },
        "shape_fragmentation": {
            "semantic_id": "framework.shape_fragmentation",
            "scope": ["framework"],
        },
        "framework_dispatch_overhead": {
            "semantic_id": "framework.dispatch_overhead",
            "scope": ["framework", "cpu-submit"],
        },
    },
}
_SIGNALS = {
    kind: set(signals)
    for kind, signals in _DIAGNOSTIC_SIGNAL_SEMANTICS.items()
}
_DIAGNOSTIC_SIGNAL_CONTRACT = {
    kind: {
        "producer_id": _PRODUCERS[kind],
        "producer_version": "1.0.0",
        "signals": signals,
    }
    for kind, signals in _DIAGNOSTIC_SIGNAL_SEMANTICS.items()
}
_NCU_MAPPING_VERSION = "ncu-semantic-v1"
_NCU_VERSION = re.compile(r"(?<!\d)(\d{4}\.\d+)(?:\.\d+)*(?!\d)")
_NCU_METRIC_MAPPINGS = {
    "2026.2": {
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": {
            "semantic_id": "kernel.dram_throughput_pct",
            "unit": "%",
        },
        "dram__bytes.sum": {
            "semantic_id": "kernel.dram_bytes",
            "unit": "byte",
        },
        "sm__warps_active.avg.pct_of_peak_sustained_active": {
            "semantic_id": "kernel.occupancy_pct",
            "unit": "%",
        },
        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed": {
            "semantic_id": "kernel.sm_active_pct",
            "unit": "%",
        },
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": {
            "semantic_id": "kernel.tensor_pipe_pct",
            "unit": "%",
        },
        "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct": {
            "semantic_id": "kernel.barrier_stall_pct",
            "unit": "%",
        },
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": {
            "semantic_id": "kernel.long_scoreboard_pct",
            "unit": "%",
        },
        "smsp__average_warp_latency_issue_stalled_long_scoreboard_per_warp_active.pct": {
            "semantic_id": "kernel.long_scoreboard_pct",
            "unit": "%",
        },
    }
}


def diagnostic_signal_contract() -> dict:
    """Return the closed diagnostic producer/signal routing contract."""
    return json.loads(json.dumps(_DIAGNOSTIC_SIGNAL_CONTRACT))


def semantic_producer_contract() -> dict:
    """Declare direct raw semantics separately from adapter-derived vocabulary."""
    ncu_raw_semantics = sorted(
        {
            rule["semantic_id"]
            for mapping in _NCU_METRIC_MAPPINGS.values()
            for rule in mapping.values()
        }
    )
    return {
        "ncu-targeted-kernel": {
            "evidence_kind": "ncu_kernel",
            "raw_semantic_ids": ncu_raw_semantics,
            "derived_semantic_ids": sorted(
                {
                    "kernel.dependency_scope_unavailable",
                    "kernel.dram_byte_lower_bound_unavailable",
                    "kernel.global_memory_transaction_amplification",
                    "kernel.launch_shape_unavailable",
                    "kernel.memory_access_path_unmodeled",
                    "kernel.redundant_dram_traffic",
                    "kernel.memory_latency_hiding_insufficient",
                    "kernel.precision_contract_unavailable",
                    "kernel.register_or_shared_pressure",
                    "kernel.parallelism_or_wave_tail",
                    "kernel.compute_pipeline_or_dtype_mismatch",
                    "kernel.static_resource_identity_mismatch",
                    "kernel.synchronization_or_atomic_contention",
                    "kernel.warp_stall_mapping_unmodeled",
                }
            ),
            "source_version_coverage": {
                "ncu": {
                    "2026.2": ["nvidia-nsight-compute"],
                },
            },
        },
        "nsys-global-timeline": {
            "evidence_kind": "nsys_timeline",
            "raw_semantic_ids": [],
            "derived_semantic_ids": sorted(
                {
                    "communication.rank_timeline_unaligned",
                    "runtime.launch_gap_short_context",
                    "runtime.gpu_idle_gap",
                    "runtime.cpu_launch_overhead",
                    "runtime.timeline_boundary_ambiguous",
                    "serving.request_corpus_changed",
                    "transfer.boundary_ambiguous",
                    "transfer.h2d_serialized",
                    "runtime.gpu_waiting_for_input",
                    "communication.rank_arrival_skew",
                    "serving.queue_or_request_path_dominant",
                }
            ),
            "source_version_coverage": {
                "nsys": {
                    "2026.3": ["nvidia-nsight-systems"],
                },
            },
        },
        "nsys-os-runtime-slice": {
            "evidence_kind": "os_runtime",
            "raw_semantic_ids": [],
            "derived_semantic_ids": sorted(
                {
                    "communication.rank_timeline_unaligned",
                    "runtime.launch_gap_short_context",
                    "runtime.gpu_idle_gap",
                    "runtime.cpu_launch_overhead",
                    "runtime.timeline_boundary_ambiguous",
                    "serving.request_corpus_changed",
                    "transfer.boundary_ambiguous",
                    "transfer.h2d_serialized",
                    "runtime.gpu_waiting_for_input",
                    "communication.rank_arrival_skew",
                    "serving.queue_or_request_path_dominant",
                }
            ),
            "source_version_coverage": {
                "nsys": {
                    "2026.3": ["nvidia-nsight-systems"],
                },
            },
        },
        "pytorch-operator-trace": {
            "evidence_kind": "framework_trace",
            "raw_semantic_ids": [],
            "derived_semantic_ids": sorted(
                {
                    "framework.gqa_head_ratio",
                    "framework.shape_fragmentation",
                    "framework.dispatch_overhead",
                    "runtime.gpu_waiting_for_input",
                    "runtime.input_workload_changed",
                    "runtime.timeline_boundary_ambiguous",
                }
            ),
            "source_version_coverage": {
                "pytorch": {
                    "2.11.0": [
                        "pytorch-compile-profiling",
                        "pytorch-profiler",
                    ],
                },
            },
        },
    }


class ValidationError(ValueError):
    pass


def _ncu_major_minor(tool_version: Any) -> str | None:
    if type(tool_version) is not str:
        return None
    match = _NCU_VERSION.search(tool_version)
    return match.group(1) if match is not None else None


def _ncu_metric_value(raw: Any) -> tuple[float | None, str | None, str | None]:
    if isinstance(raw, Mapping):
        value = raw.get("value")
        units = raw.get("sample_units")
        if type(units) is not list or not units:
            return None, None, "missing_unit"
        if any(type(unit) is not str or not unit.strip() for unit in units):
            return None, None, "missing_unit"
        unique_units = set(units)
        if len(unique_units) != 1:
            return None, None, "inconsistent_units"
        unit = next(iter(unique_units))
        invalid_samples = raw.get("invalid_value_samples", 0)
        if type(invalid_samples) is not int or invalid_samples < 0:
            return None, None, "invalid_metric_value"
        if invalid_samples:
            return None, None, "invalid_metric_value"
    elif type(raw) in {tuple, list} and len(raw) == 2:
        value, unit = raw
        if type(unit) is not str or not unit.strip():
            return None, None, "missing_unit"
    else:
        return None, None, "invalid_metric_value"

    if type(value) not in {int, float}:
        return None, None, "invalid_metric_value"
    if not math.isfinite(value):
        return None, None, "non_finite_value"
    return float(value), unit, None


def normalize_ncu_metrics(metrics: Mapping[str, Any], tool_version: str) -> dict:
    """Map versioned raw NCU metrics to stable routing-only observations."""
    if not isinstance(metrics, Mapping):
        raise ValidationError("NCU metrics must be a mapping")

    version = _ncu_major_minor(tool_version)
    mapping = _NCU_METRIC_MAPPINGS.get(version)
    unmodeled = []
    if mapping is None:
        for metric_name in sorted(metrics):
            unmodeled.append(
                {
                    "metric_name": metric_name,
                    "reason": "unsupported_tool_version",
                }
            )
        return {
            "semantic_observations": [],
            "unmodeled_metrics": unmodeled,
            "mapping_version": _NCU_MAPPING_VERSION,
        }

    candidates: dict[str, list[tuple[str, float, str]]] = {}
    invalid_semantics = set()
    for metric_name in sorted(metrics):
        rule = mapping.get(metric_name)
        if rule is None:
            unmodeled.append(
                {"metric_name": metric_name, "reason": "unknown_metric"}
            )
            continue
        semantic_id = rule["semantic_id"]
        value, unit, reason = _ncu_metric_value(metrics[metric_name])
        if reason is not None:
            invalid_semantics.add(semantic_id)
            unmodeled.append({"metric_name": metric_name, "reason": reason})
            continue
        assert value is not None and unit is not None
        if unit != rule["unit"]:
            invalid_semantics.add(semantic_id)
            unmodeled.append(
                {"metric_name": metric_name, "reason": "unexpected_unit"}
            )
            continue
        candidates.setdefault(semantic_id, []).append((metric_name, value, unit))

    observations = []
    for semantic_id in sorted(candidates):
        entries = candidates[semantic_id]
        if semantic_id in invalid_semantics:
            for metric_name, _value, _unit in entries:
                unmodeled.append(
                    {
                        "metric_name": metric_name,
                        "reason": "semantic_input_invalid",
                    }
                )
            continue
        values = {(value, unit) for _name, value, unit in entries}
        if len(values) != 1:
            for metric_name, _value, _unit in entries:
                unmodeled.append(
                    {
                        "metric_name": metric_name,
                        "reason": "conflicting_semantic_values",
                    }
                )
            continue
        value, unit = next(iter(values))
        observations.append(
            {
                "semantic_id": semantic_id,
                "status": "observed",
                "value": value,
                "unit": unit,
                "scope": ["kernel"],
                "aggregation": "ncu_metric",
                "tool": {"name": "ncu", "version": version},
                "quality": "validated",
            }
        )

    return {
        "semantic_observations": observations,
        "unmodeled_metrics": sorted(
            unmodeled, key=lambda item: (item["metric_name"], item["reason"])
        ),
        "mapping_version": _NCU_MAPPING_VERSION,
    }


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"duplicate diagnostic key: {key}")
        value[key] = item
    return value


def _invalid_number(token: str):
    raise ValidationError(f"diagnostic number must be finite: {token}")


def _strict_json(raw: bytes, label: str) -> dict:
    if not isinstance(raw, bytes):
        raise ValidationError(f"{label} must be bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_invalid_number,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} must be strict JSON") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    return value


def _closed(value: Any, fields: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise ValidationError(
            f"{label} must be closed; missing={sorted(missing)} extra={sorted(extra)}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a safe identifier")
    return value


def _signals(kind: str, value: Any) -> list[str]:
    if type(value) is not list:
        raise ValidationError("diagnostic signals must be an array")
    result = [_identifier(item, "diagnostic signal") for item in value]
    if len(result) != len(set(result)):
        raise ValidationError("diagnostic signals must not contain duplicates")
    unknown = set(result) - set(
        _DIAGNOSTIC_SIGNAL_CONTRACT[kind]["signals"]
    )
    if unknown:
        raise ValidationError(f"unsupported signals for {kind}: {sorted(unknown)}")
    return sorted(result)


def _subject(value: Any) -> dict:
    subject = _closed(value, {"target_sha256"}, "diagnostic subject")
    _sha(subject["target_sha256"], "subject.target_sha256")
    return dict(subject)


def _report(value: Any) -> dict:
    report = _closed(
        value, {"artifact_sha256", "events_total"}, "diagnostic report"
    )
    _sha(report["artifact_sha256"], "report.artifact_sha256")
    if type(report["events_total"]) is not int or report["events_total"] < 1:
        raise ValidationError("report.events_total must be a positive integer")
    return dict(report)


def _checks(value: Any) -> None:
    if type(value) is not list or not value:
        raise ValidationError("diagnostic checks must be a non-empty array")
    names = set()
    for index, raw in enumerate(value):
        check = _closed(raw, {"name", "passed"}, f"diagnostic check {index}")
        name = _identifier(check["name"], f"diagnostic check {index}.name")
        if name in names:
            raise ValidationError("diagnostic check names must be unique")
        names.add(name)
        if check["passed"] is not True:
            raise ValidationError("diagnostic checks do not support a usable observation")


def derive_diagnostic_evidence(
    raw_measurement: bytes,
    *,
    kind: str,
    producer_id: str,
    producer_version: str,
    implementation_sha256: str,
    adapter_request_sha256: str,
    contract_sha256: str,
    environment_sha256: str,
    recorded_at: float,
) -> bytes:
    if kind not in _DIAGNOSTIC_SIGNAL_CONTRACT:
        raise ValidationError(f"unsupported diagnostic kind: {kind}")
    producer = _DIAGNOSTIC_SIGNAL_CONTRACT[kind]
    if (
        producer_id != producer["producer_id"]
        or producer_version != producer["producer_version"]
    ):
        raise ValidationError(f"untrusted producer for {kind}")
    measurement = _strict_json(raw_measurement, "diagnostic measurement")
    _closed(
        measurement,
        {"schema_version", "subject", "report", "signals", "checks"},
        "diagnostic measurement",
    )
    if measurement["schema_version"] != MEASUREMENT_SCHEMA:
        raise ValidationError("unsupported diagnostic measurement schema")
    _checks(measurement["checks"])
    if type(recorded_at) not in {int, float} or not math.isfinite(recorded_at) or recorded_at < 0:
        raise ValidationError("controller recorded_at must be non-negative and finite")
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "kind": kind,
        "producer": {
            "id": producer_id,
            "version": producer_version,
            "implementation_sha256": _sha(
                implementation_sha256, "producer.implementation_sha256"
            ),
        },
        "adapter_request_sha256": _sha(
            adapter_request_sha256, "adapter_request_sha256"
        ),
        "contract_sha256": _sha(contract_sha256, "contract_sha256"),
        "environment_sha256": _sha(
            environment_sha256, "environment_sha256"
        ),
        "recorded_at": float(recorded_at),
        "subject": _subject(measurement["subject"]),
        "report": _report(measurement["report"]),
        "signals": _signals(kind, measurement["signals"]),
    }
    return (
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_diagnostic_evidence(
    raw: bytes,
    *,
    expected_contract_sha256: str,
    expected_environment_sha256: str,
) -> dict:
    evidence = _strict_json(raw, "diagnostic evidence artifact")
    _closed(
        evidence,
        {
            "schema_version",
            "kind",
            "producer",
            "adapter_request_sha256",
            "contract_sha256",
            "environment_sha256",
            "recorded_at",
            "subject",
            "report",
            "signals",
        },
        "diagnostic evidence",
    )
    if evidence["schema_version"] != EVIDENCE_SCHEMA:
        raise ValidationError("unsupported diagnostic evidence schema")
    kind = evidence["kind"]
    if kind not in _DIAGNOSTIC_SIGNAL_CONTRACT:
        raise ValidationError(f"unsupported diagnostic kind: {kind}")
    producer_contract = _DIAGNOSTIC_SIGNAL_CONTRACT[kind]
    producer = _closed(
        evidence["producer"],
        {"id", "version", "implementation_sha256"},
        "diagnostic producer",
    )
    if (
        producer["id"] != producer_contract["producer_id"]
        or producer["version"] != producer_contract["producer_version"]
    ):
        raise ValidationError(f"untrusted producer for {kind}")
    _sha(producer["implementation_sha256"], "producer.implementation_sha256")
    request_sha = _sha(evidence["adapter_request_sha256"], "adapter_request_sha256")
    contract = _sha(evidence["contract_sha256"], "contract_sha256")
    environment = _sha(evidence["environment_sha256"], "environment_sha256")
    if contract != _sha(expected_contract_sha256, "expected_contract_sha256"):
        raise ValidationError("diagnostic contract identity mismatch")
    if environment != _sha(
        expected_environment_sha256, "expected_environment_sha256"
    ):
        raise ValidationError("diagnostic environment identity mismatch")
    recorded_at = evidence["recorded_at"]
    if type(recorded_at) not in {int, float} or not math.isfinite(recorded_at) or recorded_at < 0:
        raise ValidationError("diagnostic recorded_at must be non-negative and finite")
    return {
        "kind": kind,
        "layer": "workload",
        "summary": (
            f"Validated {kind} observation from "
            f"{producer_contract['producer_id']}."
        ),
        "signals": _signals(kind, evidence["signals"]),
        "producer": dict(producer),
        "adapter_request_sha256": request_sha,
        "recorded_at": float(recorded_at),
        "subject": _subject(evidence["subject"]),
        "result": _report(evidence["report"]),
    }
