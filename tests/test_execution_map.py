from __future__ import annotations

import copy
import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "execution_map.py"
PYTORCH_PROFILE = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "profile_pytorch.py"


def _load():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_execution_map", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_pytorch_profile():
    spec = importlib.util.spec_from_file_location("cuda_optimizer_profile_pytorch", PYTORCH_PROFILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _event(*, name="aten::add", category="cpu_op", pid=1, tid=7, start=0.0, end=10.0) -> dict:
    duration = end - start
    return {
        "semantic_id": "pytorch.trace.complete_event",
        "value": duration,
        "unit": "us",
        "interval": {"start_us": start, "end_us": end},
        "duration_us": duration,
        "category": category,
        "name": name,
        "scope": ["process", pid, "thread", tid],
        "aggregation": "single_complete_event",
        "source": {"phase": "X", "timestamp_unit": "us"},
        "tool": {"name": "pytorch_profiler", "version": "2.13.1"},
    }


def _ncu_percent(semantic_id: str, value: float) -> dict:
    metrics = {
        "kernel.tensor_pipe_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "kernel.sm_active_pct": "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "kernel.dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "kernel.barrier_stall_pct": "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct",
        "kernel.long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    }
    return {
        "semantic_id": semantic_id,
        "value": value,
        "unit": "%",
        "scope": ["kernel"],
        "aggregation": "mean_across_matching_rows",
        "source_metric": metrics[semantic_id],
        "tool": {"name": "ncu", "version": "2026.2.1"},
    }


class ExecutionMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load()

    def test_time_coverage_overlap_and_serial_upper_bound_are_exact(self) -> None:
        result = self.module.account_time(
            [
                _event(name="aten::add", start=0.0, end=10.0),
                _event(name="aten::mul", start=5.0, end=15.0),
                _event(name="aten::sum", start=15.0, end=20.0),
            ]
        )

        self.assertEqual(result["window"], {"start_us": 0.0, "end_us": 20.0})
        self.assertEqual(result["covered_union_us"], 20.0)
        self.assertEqual(result["overlap_excess_us"], 5.0)
        self.assertEqual(result["maximum_concurrent_events"], 2)
        self.assertEqual(result["serial_duration_upper_bound_us"], 25.0)
        self.assertEqual(
            result["event_duration_upper_bounds_us"],
            [
                {"category": "cpu_op", "name": "aten::add", "count": 1, "duration_upper_bound_us": 10.0},
                {"category": "cpu_op", "name": "aten::mul", "count": 1, "duration_upper_bound_us": 10.0},
                {"category": "cpu_op", "name": "aten::sum", "count": 1, "duration_upper_bound_us": 5.0},
            ],
        )

    def test_identical_input_is_detached_and_deterministically_ordered(self) -> None:
        events = [_event(name="z", start=3.0, end=5.0), _event(name="a", start=0.0, end=1.0)]
        first = self.module.account_time(events)
        second = self.module.account_time(list(reversed(events)))

        self.assertEqual(first, second)
        events[0]["interval"]["end_us"] = 99.0
        self.assertEqual(first["window"]["end_us"], 5.0)

    def test_only_known_complete_event_observations_are_accepted(self) -> None:
        invalid = _event()
        invalid["semantic_id"] = "kernel.duration"
        with self.assertRaisesRegex(self.module.ValidationError, "known"):
            self.module.account_time([invalid])
        invalid = _event()
        invalid["source"]["phase"] = "B"
        with self.assertRaisesRegex(self.module.ValidationError, "source"):
            self.module.account_time([invalid])

    def test_pytorch_profiler_complete_event_output_is_the_supported_input(self) -> None:
        profiler = _load_pytorch_profile()
        observations = profiler.parse_chrome_trace(
            {
                "schemaVersion": 1,
                "traceEvents": [
                    {"name": "aten::add", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 7, "ts": 2.0, "dur": 3.0}
                ],
            },
            "2.13.1",
        )["observations"]
        result = self.module.account_time(observations)
        self.assertEqual(result["covered_union_us"], 3.0)

    def test_normalized_ncu_percentages_produce_only_observed_utilization_gaps(self) -> None:
        result = self.module.account_ncu_utilization(
            [
                _ncu_percent("kernel.tensor_pipe_pct", 30.0),
                _ncu_percent("kernel.sm_active_pct", 42.0),
                _ncu_percent("kernel.dram_throughput_pct", 60.0),
                _ncu_percent("kernel.barrier_stall_pct", 9.0),
                _ncu_percent("kernel.long_scoreboard_pct", 25.0),
            ]
        )
        self.assertEqual(
            result,
            {
                "observed_axes": [
                    {"axis": "compute", "utilization_percent": 42.0, "unutilized_percent": 58.0},
                    {"axis": "latency", "stall_percent": 25.0, "non_stall_percent": 75.0},
                    {"axis": "memory", "utilization_percent": 60.0, "unutilized_percent": 40.0},
                ],
                "missing_axes": [],
            },
        )

    def test_ncu_utilization_never_invents_missing_axis_or_accepts_bad_percent(self) -> None:
        result = self.module.account_ncu_utilization(
            [_ncu_percent("kernel.dram_throughput_pct", 60.0)]
        )
        self.assertEqual(result["missing_axes"], ["compute", "latency"])
        with self.assertRaisesRegex(self.module.ValidationError, "finite"):
            self.module.account_ncu_utilization(
                [_ncu_percent("kernel.dram_throughput_pct", math.inf)]
            )

    def test_nonfinite_inconsistent_duration_and_unknown_fields_fail_closed(self) -> None:
        for mutate, expression in (
            (lambda item: item["interval"].update({"end_us": math.inf}), "finite"),
            (lambda item: item.update({"duration_us": 1.0}), "duration"),
            (lambda item: item.update({"custom": True}), "unknown"),
        ):
            value = _event()
            mutate(value)
            with self.subTest(mutate=expression), self.assertRaisesRegex(self.module.ValidationError, expression):
                self.module.account_time([value])

    def test_empty_or_too_many_observations_fail_closed(self) -> None:
        with self.assertRaisesRegex(self.module.ValidationError, "non-empty"):
            self.module.account_time([])
        original = self.module._MAX_OBSERVATIONS
        self.module._MAX_OBSERVATIONS = 1
        try:
            with self.assertRaisesRegex(self.module.ValidationError, "limit"):
                self.module.account_time([_event(), _event(start=20.0, end=21.0)])
        finally:
            self.module._MAX_OBSERVATIONS = original

    def test_module_stays_pure_and_does_not_return_decision_language(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("import os", "import subprocess", "import pathlib", "bottleneck", "suggested_probe", "direction", "next_step"):
            self.assertNotIn(forbidden, source)
        result = self.module.account_time([_event()])
        self.assertNotIn("primary_bottleneck", result)
        self.assertNotIn("suggested_probe", result)


if __name__ == "__main__":
    unittest.main()
