#!/usr/bin/env python3
"""Build deterministic performance facts from a validated execution map."""

from __future__ import annotations

import copy
import importlib.util
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MODEL_SCHEMA = "cuda-optimizer/performance-model-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


class ValidationError(ValueError):
    """Raised when deterministic model inputs are incomplete or ambiguous."""


def _load_execution_map_module():
    path = Path(__file__).with_name("execution_map.py")
    name = "cuda_optimizer_execution_map_performance_model"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load execution map module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXECUTION_MAP = _load_execution_map_module()


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValidationError(f"{field} must be finite")
    result = float(value)
    if result < 0 or (positive and result <= 0):
        raise ValidationError(f"{field} must be {'positive' if positive else 'non-negative'}")
    return result


def _identifier(value: Any, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValidationError(f"{field} must be a safe identifier")
    return value


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _timing_estimates(
    records: Sequence[Mapping[str, Any]], identities: Mapping[str, Any]
) -> dict[str, dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"action_timings[{index}] must be an object")
        unknown = set(raw) - {"action_id", "identities", "elapsed_seconds"}
        missing = {"action_id", "identities", "elapsed_seconds"} - set(raw)
        if missing or unknown:
            raise ValidationError(
                f"action_timings[{index}] fields are invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        action_id = _identifier(raw["action_id"], f"action_timings[{index}].action_id")
        record_identities = raw["identities"]
        if not isinstance(record_identities, Mapping):
            raise ValidationError(f"action_timings[{index}].identities must be an object")
        elapsed = _number(
            raw["elapsed_seconds"], f"action_timings[{index}].elapsed_seconds", positive=True
        )
        if dict(record_identities) == dict(identities):
            grouped[action_id].append(elapsed)
    return {
        action_id: {
            "sample_count": len(values),
            "p50_seconds": _nearest_rank(values, 0.50),
            "p90_seconds": _nearest_rank(values, 0.90),
            "basis": "identity_matched_history",
        }
        for action_id, values in sorted(grouped.items())
    }


def build_performance_model(
    execution_map: Mapping[str, Any],
    *,
    minimum_effect_us: float,
    action_timings: Sequence[Mapping[str, Any]] = (),
) -> dict:
    """Compute bounded headroom and uncertainty without model-owned facts."""
    if not isinstance(execution_map, Mapping):
        raise ValidationError("execution_map must be an object")
    threshold = _number(minimum_effect_us, "minimum_effect_us", positive=True)
    window = execution_map.get("window")
    if not isinstance(window, Mapping):
        raise ValidationError("execution_map.window must be an object")
    start = _number(window.get("start_us"), "window.start_us")
    end = _number(window.get("end_us"), "window.end_us")
    if end <= start:
        raise ValidationError("execution_map window must be positive")
    window_duration = end - start
    coverage = execution_map.get("coverage")
    nodes = execution_map.get("nodes")
    hot_path = execution_map.get("hot_path")
    identities = execution_map.get("identities")
    if type(coverage) is not list or type(nodes) is not list or type(hot_path) is not list:
        raise ValidationError("execution_map must contain validated coverage, nodes, and hot_path")
    if not isinstance(identities, Mapping):
        raise ValidationError("execution_map.identities must be an object")

    observed_layers = sorted(item["layer"] for item in coverage if item.get("status") == "observed")
    missing_layers = sorted(item["layer"] for item in coverage if item.get("status") != "observed")
    by_id = {item["node_id"]: item for item in nodes}
    hot_nodes = [by_id[node_id] for node_id in hot_path]
    layer_active: dict[str, float] = defaultdict(float)
    node_directions = []
    for node in hot_nodes:
        duration = min(_number(node["duration_us"], f"{node['node_id']}.duration_us", positive=True), window_duration)
        layer_active[node["layer"]] += duration
        node_directions.append(
            {
                "node_id": node["node_id"],
                "layer": node["layer"],
                "benefit_ceiling_us": duration,
                "first_start_us": node["first_start_us"],
                "last_end_us": node["last_end_us"],
                "qualifies_minimum_effect": duration >= threshold,
                "basis": "observed_active_time_upper_bound",
                "evidence_ids": sorted(node["evidence_ids"]),
            }
        )
    node_directions.sort(key=lambda item: (-item["benefit_ceiling_us"], item["node_id"]))
    layer_directions = [
        {
            "layer": layer,
            "benefit_ceiling_us": min(duration, window_duration),
            "qualifies_minimum_effect": min(duration, window_duration) >= threshold,
            "basis": "summed_hot_path_active_time_capped_by_window",
        }
        for layer, duration in sorted(layer_active.items())
    ]
    layer_directions.sort(key=lambda item: (-item["benefit_ceiling_us"], item["layer"]))

    critical = _EXECUTION_MAP.critical_path_accounting(execution_map)
    uncertainties = []
    if critical["accounting_status"] == "envelope_upper_bound":
        uncertainties.append("hot_path_timing_envelopes_exceed_active_time")
    if window.get("boundary_ambiguous"):
        uncertainties.append("analysis_window_boundary_ambiguous")
    if execution_map.get("uncovered_intervals"):
        uncertainties.append("execution_map_contains_uncovered_intervals")
    if execution_map.get("conclusion_level") != "observed":
        uncertainties.append("execution_map_is_inconclusive")

    return {
        "schema_version": MODEL_SCHEMA,
        "map_id": execution_map.get("map_id"),
        "epoch_id": execution_map.get("epoch_id"),
        "identities": copy.deepcopy(dict(identities)),
        "window_duration_us": window_duration,
        "minimum_effect_us": threshold,
        "observed_layers": observed_layers,
        "missing_layers": missing_layers,
        "critical_path": critical,
        "layer_directions": layer_directions,
        "node_directions": node_directions,
        "qualifying_direction_count": sum(
            item["qualifies_minimum_effect"] for item in node_directions
        ),
        "uncertainties": sorted(set(uncertainties)),
        "action_timing_estimates": _timing_estimates(action_timings, identities),
    }
