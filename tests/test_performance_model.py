from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

from tests.test_execution_map import map_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def _load(filename: str, name: str):
    path = SCRIPTS / filename
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class PerformanceModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.map_module = _load("execution_map.py", "performance_model_execution_map")
        self.module = _load("performance_model.py", "performance_model_test")
        self.execution_map = map_fixture(self.map_module)

    def build(self, execution_map=None, *, minimum_effect_us=1.0, timings=()):
        return self.module.build_performance_model(
            self.execution_map if execution_map is None else execution_map,
            minimum_effect_us=minimum_effect_us,
            action_timings=list(timings),
        )

    def test_overlapping_hot_path_is_not_double_counted(self) -> None:
        result = self.build()

        critical = result["critical_path"]
        self.assertEqual(critical["covered_union_upper_bound_us"], 1000.0)
        self.assertEqual(critical["overlap_upper_bound_us"], 800.0)
        self.assertEqual(critical["exclusive_span_us"]["cpu-launch"], 100.0)
        self.assertEqual(critical["exclusive_span_us"]["gpu-kernel"], 100.0)
        self.assertLessEqual(
            critical["covered_union_upper_bound_us"], result["window_duration_us"]
        )

    def test_layer_and_node_headroom_are_explicit_upper_bounds(self) -> None:
        result = self.build(minimum_effect_us=100.0)

        by_layer = {item["layer"]: item for item in result["layer_directions"]}
        by_node = {item["node_id"]: item for item in result["node_directions"]}
        self.assertEqual(by_layer["cpu"]["benefit_ceiling_us"], 900.0)
        self.assertEqual(by_layer["gpu"]["benefit_ceiling_us"], 900.0)
        self.assertEqual(by_node["gpu-kernel"]["benefit_ceiling_us"], 900.0)
        self.assertTrue(by_node["gpu-kernel"]["qualifies_minimum_effect"])
        self.assertEqual(by_node["gpu-kernel"]["basis"], "observed_active_time_upper_bound")

    def test_direction_below_minimum_effect_is_not_qualified(self) -> None:
        value = copy.deepcopy(self.execution_map)
        value["nodes"][1].update(
            {
                "duration_us": 0.5,
                "first_start_us": 999.5,
                "last_end_us": 1000.0,
            }
        )
        value["nodes"][0].update(
            {"duration_us": 0.5, "first_start_us": 0.0, "last_end_us": 0.5}
        )

        result = self.build(value, minimum_effect_us=1.0)

        self.assertFalse(any(item["qualifies_minimum_effect"] for item in result["node_directions"]))
        self.assertEqual(result["qualifying_direction_count"], 0)

    def test_missing_layers_and_ambiguous_envelopes_are_reported(self) -> None:
        value = copy.deepcopy(self.execution_map)
        value["nodes"][0]["duration_us"] = 100.0

        result = self.build(value)

        self.assertIn("framework", result["missing_layers"])
        self.assertEqual(result["critical_path"]["accounting_status"], "envelope_upper_bound")
        self.assertIn("hot_path_timing_envelopes_exceed_active_time", result["uncertainties"])

    def test_only_identity_matched_action_timings_produce_numeric_range(self) -> None:
        identities = copy.deepcopy(self.execution_map["identities"])
        matching = [
            {"action_id": "ncu-targeted", "identities": identities, "elapsed_seconds": value}
            for value in (10.0, 12.0, 20.0)
        ]
        mismatched = copy.deepcopy(matching[0])
        mismatched["action_id"] = "global-profile"
        mismatched["identities"]["source_sha256"] = "9" * 64

        result = self.build(timings=[*matching, mismatched])

        estimate = result["action_timing_estimates"]["ncu-targeted"]
        self.assertEqual(estimate["p50_seconds"], 12.0)
        self.assertEqual(estimate["p90_seconds"], 20.0)
        self.assertEqual(estimate["basis"], "identity_matched_history")
        self.assertNotIn("global-profile", result["action_timing_estimates"])

    def test_no_timing_history_does_not_invent_numeric_cost(self) -> None:
        result = self.build()

        self.assertEqual(result["action_timing_estimates"], {})
        self.assertNotIn("estimated_seconds", result)


if __name__ == "__main__":
    unittest.main()
