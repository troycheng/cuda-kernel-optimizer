"""Pure coverage and overlap accounting for known profiler observations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


_OBSERVATION_FIELDS = {
    "semantic_id",
    "value",
    "unit",
    "interval",
    "duration_us",
    "category",
    "name",
    "scope",
    "aggregation",
    "source",
    "tool",
}
_MAX_OBSERVATIONS = 10_000
_MAX_TEXT = 512
_NCU_VERSION = re.compile(r"2026\.2(?:\.\d+)*\Z")
_NCU_FIELDS = {"semantic_id", "value", "unit", "scope", "aggregation", "source_metric", "tool"}
_NCU_METRICS = {
    "kernel.tensor_pipe_pct": ("%", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"),
    "kernel.sm_active_pct": ("%", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"),
    "kernel.dram_throughput_pct": ("%", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
    "kernel.barrier_stall_pct": ("%", "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct"),
    "kernel.long_scoreboard_pct": ("%", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
    "kernel.dram_bytes": ("byte", "dram__bytes.sum"),
    "kernel.occupancy_pct": ("%", "sm__warps_active.avg.pct_of_peak_sustained_active"),
}


class ValidationError(ValueError):
    """Raised when an observation cannot support deterministic accounting."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT:
        raise ValidationError(f"{label} must be a non-empty bounded string")
    return value


def _number(value: Any, label: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{label} must be finite")
    number = float(value)
    if non_negative and number < 0:
        raise ValidationError(f"{label} must be non-negative")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


def _complete_event(raw: Any, index: int) -> dict:
    if type(raw) is not dict:
        raise ValidationError(f"observations[{index}] must be an object")
    missing = _OBSERVATION_FIELDS - set(raw)
    unknown = set(raw) - _OBSERVATION_FIELDS
    if missing or unknown:
        raise ValidationError(f"observations[{index}] has missing={sorted(missing)} unknown={sorted(unknown)}")
    if raw["semantic_id"] != "pytorch.trace.complete_event":
        raise ValidationError("only known complete-event observations are supported")
    if raw["unit"] != "us" or raw["aggregation"] != "single_complete_event":
        raise ValidationError(f"observations[{index}] has unsupported unit or aggregation")
    interval = raw["interval"]
    if type(interval) is not dict or set(interval) != {"start_us", "end_us"}:
        raise ValidationError(f"observations[{index}].interval is invalid")
    start = _number(interval["start_us"], f"observations[{index}].interval.start_us", non_negative=True)
    end = _number(interval["end_us"], f"observations[{index}].interval.end_us", non_negative=True)
    if end < start:
        raise ValidationError(f"observations[{index}].interval is inverted")
    duration = _number(raw["duration_us"], f"observations[{index}].duration_us", non_negative=True)
    value = _number(raw["value"], f"observations[{index}].value", non_negative=True)
    span = end - start
    if not math.isclose(duration, span, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(value, duration, rel_tol=1e-9, abs_tol=1e-9):
        raise ValidationError(f"observations[{index}] duration does not match its interval")
    source = raw["source"]
    if type(source) is not dict or source != {"phase": "X", "timestamp_unit": "us"}:
        raise ValidationError(f"observations[{index}].source is not a known complete-event source")
    tool = raw["tool"]
    if type(tool) is not dict or set(tool) != {"name", "version"} or tool["name"] != "pytorch_profiler":
        raise ValidationError(f"observations[{index}].tool is unsupported")
    version = _text(tool["version"], f"observations[{index}].tool.version")
    scope = raw["scope"]
    if type(scope) is not list or len(scope) != 4 or scope[0] != "process" or scope[2] != "thread":
        raise ValidationError(f"observations[{index}].scope is invalid")
    pid = _integer(scope[1], f"observations[{index}].scope.pid")
    tid = _integer(scope[3], f"observations[{index}].scope.tid")
    return {
        "start_us": start,
        "end_us": end,
        "duration_us": duration,
        "category": _text(raw["category"], f"observations[{index}].category"),
        "name": _text(raw["name"], f"observations[{index}].name"),
        "pid": pid,
        "tid": tid,
        "tool_version": version,
    }


def validate_observations(observations: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Validate and detach the sole observation dialect supported by this module."""
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes, bytearray)):
        raise ValidationError("observations must be a non-empty sequence")
    if not observations:
        raise ValidationError("observations must be non-empty")
    if len(observations) > _MAX_OBSERVATIONS:
        raise ValidationError("observations exceed limit")
    return [_complete_event(item, index) for index, item in enumerate(observations)]


def account_time(observations: Sequence[Mapping[str, Any]]) -> dict:
    """Return exact interval coverage and explicitly labelled duration upper bounds."""
    events = validate_observations(observations)
    boundaries: dict[float, list[int]] = {}
    grouped: dict[tuple[str, str], list[float]] = {}
    for event in events:
        start, end = event["start_us"], event["end_us"]
        boundaries.setdefault(start, [0, 0])[0] += 1
        boundaries.setdefault(end, [0, 0])[1] += 1
        grouped.setdefault((event["category"], event["name"]), []).append(event["duration_us"])

    covered, overlap_excess, active, maximum = 0.0, 0.0, 0, 0
    points = sorted(boundaries)
    for index, point in enumerate(points[:-1]):
        starts, ends = boundaries[point]
        active += starts - ends
        maximum = max(maximum, active)
        width = points[index + 1] - point
        if active > 0:
            covered += width
        if active > 1:
            overlap_excess += (active - 1) * width
    if points:
        maximum = max(maximum, boundaries[points[-1]][0] - boundaries[points[-1]][1])
    bounds = [
        {
            "category": category,
            "name": name,
            "count": len(durations),
            "duration_upper_bound_us": math.fsum(durations),
        }
        for (category, name), durations in sorted(grouped.items())
    ]
    return {
        "observation_count": len(events),
        "window": {"start_us": points[0], "end_us": points[-1]},
        "covered_union_us": covered,
        "overlap_excess_us": overlap_excess,
        "maximum_concurrent_events": maximum,
        "serial_duration_upper_bound_us": math.fsum(event["duration_us"] for event in events),
        "event_duration_upper_bounds_us": bounds,
    }


def _ncu_observation(raw: Any, index: int) -> tuple[str, float]:
    if type(raw) is not dict:
        raise ValidationError(f"observations[{index}] must be an object")
    missing, unknown = _NCU_FIELDS - set(raw), set(raw) - _NCU_FIELDS
    if missing or unknown:
        raise ValidationError(f"observations[{index}] has missing={sorted(missing)} unknown={sorted(unknown)}")
    semantic_id = raw["semantic_id"]
    if semantic_id not in _NCU_METRICS:
        raise ValidationError("only known NCU observations are supported")
    expected_unit, expected_source = _NCU_METRICS[semantic_id]
    if raw["unit"] != expected_unit or raw["source_metric"] != expected_source:
        raise ValidationError(f"observations[{index}] does not match its known NCU metric")
    if raw["scope"] != ["kernel"] or raw["aggregation"] != "mean_across_matching_rows":
        raise ValidationError(f"observations[{index}] has unsupported NCU scope or aggregation")
    tool = raw["tool"]
    if type(tool) is not dict or set(tool) != {"name", "version"} or tool["name"] != "ncu" or not isinstance(tool["version"], str) or _NCU_VERSION.fullmatch(tool["version"]) is None:
        raise ValidationError(f"observations[{index}].tool is unsupported")
    value = _number(raw["value"], f"observations[{index}].value", non_negative=True)
    if expected_unit == "%" and value > 100:
        raise ValidationError(f"observations[{index}] percentage must be at most 100")
    return semantic_id, value


def account_ncu_utilization(observations: Sequence[Mapping[str, Any]]) -> dict:
    """Return observed NCU utilization and stall complements without any allocation policy."""
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes, bytearray)):
        raise ValidationError("observations must be a non-empty sequence")
    if not observations:
        raise ValidationError("observations must be non-empty")
    if len(observations) > _MAX_OBSERVATIONS:
        raise ValidationError("observations exceed limit")
    values = {}
    for index, raw in enumerate(observations):
        semantic_id, value = _ncu_observation(raw, index)
        if semantic_id in values:
            raise ValidationError("NCU observations must not repeat one semantic id")
        values[semantic_id] = value
    axes = []
    compute = [values[key] for key in ("kernel.tensor_pipe_pct", "kernel.sm_active_pct") if key in values]
    if compute:
        utilization = max(compute)
        axes.append({"axis": "compute", "utilization_percent": utilization, "unutilized_percent": 100.0 - utilization})
    if "kernel.dram_throughput_pct" in values:
        utilization = values["kernel.dram_throughput_pct"]
        axes.append({"axis": "memory", "utilization_percent": utilization, "unutilized_percent": 100.0 - utilization})
    stalls = [values[key] for key in ("kernel.barrier_stall_pct", "kernel.long_scoreboard_pct") if key in values]
    if stalls:
        stall = max(stalls)
        axes.append({"axis": "latency", "stall_percent": stall, "non_stall_percent": 100.0 - stall})
    return {
        "observed_axes": sorted(axes, key=lambda item: item["axis"]),
        "missing_axes": [axis for axis in ("compute", "latency", "memory") if axis not in {item["axis"] for item in axes}],
    }
