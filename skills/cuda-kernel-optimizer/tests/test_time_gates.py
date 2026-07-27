from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = SKILL_ROOT / "scripts" / "budget.py"


def _load_budget():
    name = "cuda_optimizer_installed_time_gate_tests"
    spec = importlib.util.spec_from_file_location(name, BUDGET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TimeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = _load_budget()
        if not hasattr(self.budget, "CandidateGate"):
            self.fail("budget.py must expose CandidateGate")
        self.contract = {
            "soft_target_seconds": 30.0,
            "hard_ceiling_seconds": 300.0,
            "minimum_effect": {"mechanism_us": 1.0, "service_pct": 0.5},
        }
        self.candidate = {
            "claim_layer": "kernel",
            "cheapest_falsifier": "static_review",
            "estimated_cost": {
                "static_review": {
                    "p90_seconds": 1.0,
                    "basis": "declared_upper_bound",
                },
                "build_correctness": {
                    "p90_seconds": 4.0,
                    "basis": "declared_upper_bound",
                },
                "short_paired": {
                    "p90_seconds": 5.0,
                    "basis": "declared_upper_bound",
                },
                "profiler": {
                    "p90_seconds": 10.0,
                    "basis": "declared_upper_bound",
                },
                "formal_paired": {
                    "p90_seconds": 20.0,
                    "basis": "declared_upper_bound",
                },
            },
            "minimum_effect": {"metric": "mechanism_us", "value": 1.0},
            "rejection_condition": "upper_bound_below_minimum_or_gate_failed",
            "promotion_condition": "all_required_gates_passed",
        }

    def _gate(self):
        return self.budget.CandidateGate(self.contract, self.candidate)

    def _authorization(self, **overrides):
        authorization = {
            "max_controlled_seconds": self.contract["hard_ceiling_seconds"],
            "max_stage": "formal_paired",
            "applicable_stages": [
                "static_review",
                "build_correctness",
                "short_paired",
                "formal_paired",
            ],
        }
        authorization.update(overrides)
        return authorization

    @staticmethod
    def _passed_through_short(*, lower_bound=0.5, upper_bound=2.0):
        return {
            "static_review": {"status": "passed"},
            "build_correctness": {"status": "passed"},
            "short_paired": {
                "status": "passed",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        }

    def test_static_is_blocked_before_action_when_cumulative_p90_is_not_authorized(self) -> None:
        result = self._gate().decide({}, 299.5, self._authorization())

        self.assertEqual(result["decision"], "REVIEW_REQUIRED")
        self.assertEqual(result["blocked_action"]["action_id"], "static_review")
        self.assertEqual(result["elapsed_seconds"], 299.5)
        self.assertEqual(result["projected_spend"]["p90_seconds"], 300.5)

    def test_authorized_static_is_selected_without_charging_spend(self) -> None:
        gate = self._gate()
        authorization = self._authorization()

        first = gate.decide({}, 10.0, authorization)
        second = gate.decide({}, 10.0, authorization)

        self.assertEqual(first, second)
        self.assertEqual(first["decision"], "RUN_STAGE")
        self.assertEqual(first["next_stage"], "static_review")
        self.assertEqual(first["elapsed_seconds"], 10.0)
        self.assertEqual(first["projected_spend"]["p90_seconds"], 11.0)

    def test_static_falsification_does_not_select_gpu_benchmark(self) -> None:
        result = self._gate().decide(
            {"static_review": {"status": "failed"}},
            1.0,
            self._authorization(),
        )

        self.assertEqual(result["decision"], "STOP")
        self.assertIsNone(result["next_stage"])
        self.assertIn("build_correctness", result["skipped_expensive_stages"])

    def test_correctness_failure_does_not_select_profiler(self) -> None:
        result = self._gate().decide(
            {
                "static_review": {"status": "passed"},
                "build_correctness": {"status": "failed"},
            },
            5.0,
            self._authorization(),
        )

        self.assertIsNone(result["next_stage"])
        self.assertIn("profiler", result["skipped_expensive_stages"])
        self.assertEqual(result["stop_reason"], "correctness_failed")

    def test_profiler_is_skipped_without_declared_live_uncertainty(self) -> None:
        completed = {
            **self._passed_through_short(),
            "formal_paired": {"status": "passed", "lower_bound": 1.5},
        }

        result = self._gate().decide(
            completed,
            30.0,
            self._authorization(),
        )

        self.assertEqual(
            result["completed_stages"],
            [
                "static_review",
                "build_correctness",
                "short_paired",
                "formal_paired",
            ],
        )
        self.assertIn("profiler", result["skipped_expensive_stages"])
        self.assertEqual(result["decision"], "PROMOTE")

    def test_cross_threshold_short_interval_selects_one_bounded_follow_up(self) -> None:
        completed = self._passed_through_short(
            lower_bound=0.5,
            upper_bound=1.5,
        )
        gate = self._gate()
        authorization = self._authorization()

        first = gate.decide(completed, 10.0, authorization)
        second = gate.decide(completed, 10.0, authorization)

        self.assertEqual(first, second)
        self.assertEqual(first["decision"], "RUN_STAGE")
        self.assertEqual(first["next_stage"], "formal_paired")

    def test_follow_up_uses_grant_not_preset_hard_ceiling(self) -> None:
        self.contract["hard_ceiling_seconds"] = 22.0
        self.contract["soft_target_seconds"] = 20.0

        result = self._gate().decide(
            self._passed_through_short(lower_bound=0.5, upper_bound=1.5),
            3.0,
            self._authorization(max_controlled_seconds=23.0),
        )

        self.assertEqual(result["decision"], "RUN_STAGE")
        self.assertEqual(result["stop_reason"], "next_stage_authorized")
        self.assertEqual(result["next_stage"], "formal_paired")
        self.assertIsNone(result["blocked_action"])
        self.assertEqual(result["projected_spend"]["p90_seconds"], 23.0)
        self.assertEqual(result["elapsed_seconds"], 3.0)

    def test_authoritative_spend_is_reflected_once_in_projection_and_soft_target(self) -> None:
        self.contract["soft_target_seconds"] = 50.0
        self.contract["hard_ceiling_seconds"] = 122.0

        result = self._gate().decide(
            self._passed_through_short(lower_bound=0.5, upper_bound=1.5),
            103.0,
            self._authorization(),
        )

        self.assertEqual(result["elapsed_seconds"], 103.0)
        self.assertEqual(result["projected_spend"]["p90_seconds"], 123.0)
        self.assertTrue(result["soft_target_exceeded"])

    def test_short_pair_missing_lower_bound_never_selects_formal(self) -> None:
        completed = self._passed_through_short()
        del completed["short_paired"]["lower_bound"]

        result = self._gate().decide(
            completed,
            10.0,
            self._authorization(),
        )

        self.assertIsNone(result["next_stage"])
        self.assertEqual(result["stop_reason"], "short_pair_missing_lower_bound")

    def test_short_pair_invalid_interval_never_selects_formal(self) -> None:
        result = self._gate().decide(
            self._passed_through_short(lower_bound=2.0, upper_bound=1.5),
            10.0,
            self._authorization(),
        )

        self.assertIsNone(result["next_stage"])
        self.assertEqual(result["stop_reason"], "short_pair_invalid_interval")

    def test_short_pair_qualifying_lower_bound_selects_formal_confirmation(self) -> None:
        result = self._gate().decide(
            self._passed_through_short(lower_bound=1.0, upper_bound=1.5),
            10.0,
            self._authorization(),
        )

        self.assertEqual(result["decision"], "RUN_STAGE")
        self.assertEqual(result["next_stage"], "formal_paired")

    def test_repeated_decide_returns_same_terminal_result(self) -> None:
        gate = self._gate()
        completed = {"static_review": {"status": "failed"}}
        authorization = self._authorization()

        first = gate.decide(completed, 1.0, authorization)
        second = gate.decide(completed, 1.0, authorization)

        self.assertEqual(first, second)

    def test_short_pair_upper_bound_below_threshold_skips_formal_test(self) -> None:
        result = self._gate().decide(
            self._passed_through_short(lower_bound=0.2, upper_bound=0.8),
            10.0,
            self._authorization(),
        )

        self.assertEqual(result["decision"], "STOP")
        self.assertEqual(
            result["stop_reason"], "effect_upper_bound_below_minimum"
        )
        self.assertIn("formal_paired", result["skipped_expensive_stages"])

    def test_conclusive_stop_is_well_before_hard_ceiling(self) -> None:
        result = self._gate().decide(
            {"static_review": {"status": "failed"}},
            2.0,
            self._authorization(),
        )

        self.assertLess(
            result["elapsed_seconds"],
            self.contract["hard_ceiling_seconds"] / 10,
        )
        self.assertEqual(result["stop_reason"], "static_falsified")

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_runner_hard_deadline_kills_the_process_group(self) -> None:
        if not hasattr(self.budget, "run_budgeted_command"):
            self.fail("budget.py must expose run_budgeted_command")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pids = root / "pids.json"
            script = root / "hang.py"
            script.write_text(
                "import json, os, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "open(sys.argv[1], 'w').write(json.dumps({'parent': os.getpid(), 'child': child.pid}))\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            result = self.budget.run_budgeted_command(
                [sys.executable, str(script), str(pids)],
                timeout_seconds=0.4,
            )
            payload = json.loads(pids.read_text("utf-8"))
            time.sleep(0.05)

        self.assertTrue(result.timed_out)
        for pid in payload.values():
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_maintenance_soft_limit_does_not_kill_progressing_setup(self) -> None:
        if not hasattr(self.budget, "run_maintenance_command"):
            self.fail("budget.py must expose run_maintenance_command")

        started = time.monotonic()
        result = self.budget.run_maintenance_command(
            [sys.executable, "-c", "import time; time.sleep(0.35)"],
            hard_ceiling_seconds=2,
        )

        self.assertFalse(result.timed_out)
        self.assertTrue(result.soft_limit_exceeded)
        self.assertEqual(result.stop_reason, "completed")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_declared_cheapest_falsifier_must_match_candidate_stage_order(self) -> None:
        self.candidate["cheapest_falsifier"] = "formal_paired"

        with self.assertRaisesRegex(ValueError, "cheapest_falsifier"):
            self._gate()

    def test_cheaper_late_stage_cannot_bypass_cost_order(self) -> None:
        self.candidate["cheapest_falsifier"] = "profiler"
        self.candidate["estimated_cost"]["static_review"]["p90_seconds"] = 100.0
        self.candidate["estimated_cost"]["profiler"]["p90_seconds"] = 0.5

        with self.assertRaisesRegex(ValueError, "cost|cheapest_falsifier"):
            self._gate()

    def test_long_command_emits_heartbeats_and_a_terminal_reason(self) -> None:
        events = []
        result = self.budget.run_budgeted_command(
            [sys.executable, "-c", "import time; time.sleep(0.3)"],
            timeout_seconds=2,
            heartbeat_interval_seconds=0.1,
            event_sink=events.append,
        )

        self.assertGreaterEqual(
            sum(event["event"] == "heartbeat" for event in events),
            2,
        )
        self.assertEqual(events[-1]["event"], "terminal")
        self.assertEqual(events[-1]["stop_reason"], "completed")
        self.assertEqual(result.stop_reason, "completed")

    def test_output_always_exposes_time_stop_and_skipped_stages(self) -> None:
        result = self._gate().decide(
            {"static_review": {"status": "failed"}},
            1.0,
            self._authorization(),
        )

        self.assertIsInstance(result["elapsed_seconds"], float)
        self.assertEqual(result["stop_reason"], "static_falsified")
        self.assertIsInstance(result["skipped_expensive_stages"], list)

    def test_no_qualified_direction_returns_stop(self) -> None:
        result = self._gate().decide(
            self._passed_through_short(lower_bound=0.1, upper_bound=0.2),
            10.0,
            self._authorization(),
        )

        self.assertEqual(result["decision"], "STOP")
        self.assertNotEqual(result.get("next_action"), "continue_next_round")

    def test_nonfinite_performance_bounds_fail_closed(self) -> None:
        for stage, field, value, expected_reason in (
            (
                "short_paired",
                "upper_bound",
                math.nan,
                "short_pair_invalid_upper_bound",
            ),
            (
                "short_paired",
                "upper_bound",
                math.inf,
                "short_pair_invalid_upper_bound",
            ),
            (
                "formal_paired",
                "lower_bound",
                math.nan,
                "formal_paired_invalid_lower_bound",
            ),
            (
                "formal_paired",
                "lower_bound",
                math.inf,
                "formal_paired_invalid_lower_bound",
            ),
        ):
            with self.subTest(stage=stage, value=value):
                completed = self._passed_through_short()
                if stage == "short_paired":
                    completed[stage][field] = value
                else:
                    completed[stage] = {"status": "passed", field: value}

                result = self._gate().decide(
                    completed,
                    30.0,
                    self._authorization(),
                )

                self.assertEqual(result["decision"], "STOP")
                self.assertEqual(result["stop_reason"], expected_reason)

    def test_mandatory_stage_cannot_be_not_applicable(self) -> None:
        for stage, expected_reason in (
            ("static_review", "static_falsified"),
            ("build_correctness", "correctness_failed"),
            ("short_paired", "short_pair_failed"),
            ("formal_paired", "formal_pair_failed"),
        ):
            with self.subTest(stage=stage):
                completed = self._passed_through_short()
                completed["formal_paired"] = {
                    "status": "passed",
                    "lower_bound": 2.0,
                }
                stage_index = self._authorization()["applicable_stages"].index(stage)
                allowed_stages = self._authorization()["applicable_stages"][
                    : stage_index + 1
                ]
                completed = {
                    name: completed[name]
                    for name in allowed_stages
                }
                completed[stage] = {"status": "not_applicable"}

                result = self._gate().decide(
                    completed,
                    30.0,
                    self._authorization(),
                )

                self.assertEqual(result["decision"], "STOP")
                self.assertEqual(result["stop_reason"], expected_reason)

    def test_soft_target_is_guidance_not_a_direction_timeout(self) -> None:
        self.contract["soft_target_seconds"] = 2.0
        completed = {
            **self._passed_through_short(lower_bound=0.5, upper_bound=2.5),
            "formal_paired": {
                "status": "passed",
                "estimate": 1.4,
                "lower_bound": 1.1,
            },
        }

        result = self._gate().decide(
            completed,
            4.0,
            self._authorization(),
        )

        self.assertGreater(
            result["elapsed_seconds"],
            self.contract["soft_target_seconds"],
        )
        self.assertTrue(result["soft_target_exceeded"])
        self.assertEqual(result["decision"], "PROMOTE")

    def test_candidate_declaration_can_bind_real_candidate_fields(self) -> None:
        self.candidate.update({"name": "optimized", "revision": "worktree"})

        try:
            gate = self._gate()
        except ValueError as error:
            self.fail(f"real candidate fields were rejected: {error}")

        self.assertEqual(gate.candidate["name"], "optimized")


if __name__ == "__main__":
    unittest.main()
