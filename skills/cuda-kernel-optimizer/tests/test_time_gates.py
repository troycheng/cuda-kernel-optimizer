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


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def now(self) -> float:
        return self.value

    def action(self, calls: list[str], name: str, result: dict, seconds: float = 1.0):
        def run():
            calls.append(name)
            self.value += seconds
            return result

        return run


class TimeGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = _load_budget()
        if not hasattr(self.budget, "CandidateGate"):
            self.fail("budget.py must expose CandidateGate")
        self.clock = FakeClock()
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

    def _gate(self, *, pre_gate_elapsed_seconds: float = 0.0):
        return self.budget.CandidateGate(
            self.contract,
            self.candidate,
            now=self.clock.now,
            pre_gate_elapsed_seconds=pre_gate_elapsed_seconds,
        )

    def test_static_is_blocked_before_action_when_cumulative_p90_is_not_authorized(self) -> None:
        calls = []

        result = self._gate(pre_gate_elapsed_seconds=299.5).run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "passed"}
                )
            }
        )

        self.assertEqual(calls, [])
        self.assertEqual(result["decision"], "REVIEW_REQUIRED")
        self.assertEqual(result["blocked_action"]["action_id"], "static_review")
        self.assertEqual(result["elapsed_seconds"], 299.5)
        self.assertEqual(result["projected_spend"]["p90_seconds"], 300.5)

    def test_authorized_static_runs_once_and_is_charged_once(self) -> None:
        calls = []

        result = self._gate(pre_gate_elapsed_seconds=10.0).run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "failed"}
                )
            }
        )

        self.assertEqual(calls, ["static_review"])
        self.assertEqual(result["elapsed_seconds"], 11.0)

    def test_static_falsification_does_not_start_gpu_benchmark(self) -> None:
        calls = []
        actions = {
            "static_review": self.clock.action(
                calls, "static_review", {"status": "failed"}
            ),
            "build_correctness": self.clock.action(
                calls, "build_correctness", {"status": "passed"}
            ),
            "short_paired": self.clock.action(
                calls, "short_paired", {"status": "passed", "lower_bound": 0.5, "upper_bound": 2.0}
            ),
        }

        result = self._gate().run(actions)

        self.assertEqual(calls, ["static_review"])
        self.assertEqual(result["decision"], "STOP")
        self.assertIn("build_correctness", result["skipped_expensive_stages"])

    def test_correctness_failure_does_not_start_profiler(self) -> None:
        calls = []
        actions = {
            "static_review": self.clock.action(calls, "static_review", {"status": "passed"}),
            "build_correctness": self.clock.action(calls, "build_correctness", {"status": "failed"}),
            "short_paired": self.clock.action(calls, "short_paired", {"status": "passed", "lower_bound": 0.5, "upper_bound": 2.0}),
            "profiler": self.clock.action(calls, "profiler", {"status": "passed"}),
        }

        result = self._gate().run(actions)

        self.assertEqual(calls, ["static_review", "build_correctness"])
        self.assertNotIn("profiler", calls)
        self.assertEqual(result["stop_reason"], "correctness_failed")

    def test_profiler_is_skipped_without_declared_live_uncertainty(self) -> None:
        calls = []
        result = self._gate().run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "passed"}
                ),
                "build_correctness": self.clock.action(
                    calls, "build_correctness", {"status": "passed"}
                ),
                "short_paired": self.clock.action(
                    calls,
                    "short_paired",
                    {"status": "passed", "lower_bound": 0.5, "upper_bound": 2.0},
                ),
                "formal_paired": self.clock.action(
                    calls,
                    "formal_paired",
                    {"status": "passed", "lower_bound": 1.5},
                ),
            }
        )

        self.assertEqual(
            calls,
            [
                "static_review",
                "build_correctness",
                "short_paired",
                "formal_paired",
            ],
        )
        self.assertEqual(result["decision"], "PROMOTE")

    def test_cross_threshold_short_interval_allows_one_bounded_follow_up(self) -> None:
        calls = []
        actions = {
            "static_review": self.clock.action(
                calls, "static_review", {"status": "passed"}
            ),
            "build_correctness": self.clock.action(
                calls, "build_correctness", {"status": "passed"}
            ),
            "short_paired": self.clock.action(
                calls,
                "short_paired",
                {"status": "passed", "lower_bound": 0.5, "upper_bound": 1.5},
            ),
            "formal_paired": self.clock.action(
                calls,
                "formal_paired",
                {"status": "passed", "lower_bound": 1.1},
            ),
        }

        result = self._gate().run(actions)

        self.assertEqual(calls.count("formal_paired"), 1)
        self.assertEqual(result["decision"], "PROMOTE")

    def test_follow_up_is_blocked_before_action_when_p90_exceeds_authorization(self) -> None:
        calls = []
        self.contract["hard_ceiling_seconds"] = 22.0
        self.contract["soft_target_seconds"] = 20.0
        actions = {
            "static_review": self.clock.action(
                calls, "static_review", {"status": "passed"}
            ),
            "build_correctness": self.clock.action(
                calls, "build_correctness", {"status": "passed"}
            ),
            "short_paired": self.clock.action(
                calls,
                "short_paired",
                {"status": "passed", "lower_bound": 0.5, "upper_bound": 1.5},
            ),
            "formal_paired": self.clock.action(
                calls,
                "formal_paired",
                {"status": "passed", "lower_bound": 1.1},
            ),
        }

        result = self._gate().run(actions)

        self.assertEqual(
            calls, ["static_review", "build_correctness", "short_paired"]
        )
        self.assertEqual(result["decision"], "REVIEW_REQUIRED")
        self.assertEqual(
            result["stop_reason"], "authorization_insufficient_for_next_action"
        )
        self.assertEqual(result["next_stage"], "formal_paired")
        self.assertEqual(result["blocked_action"]["action_id"], "formal_paired")
        self.assertEqual(result["projected_spend"]["p90_seconds"], 23.0)
        self.assertEqual(result["elapsed_seconds"], 3.0)
        self.assertIn("formal_paired", result["skipped_expensive_stages"])

    def test_pre_gate_actual_is_included_once_in_elapsed_projected_and_soft_target(self) -> None:
        calls = []
        self.contract["soft_target_seconds"] = 50.0
        self.contract["hard_ceiling_seconds"] = 122.0
        result = self._gate(pre_gate_elapsed_seconds=100.0).run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "passed"}
                ),
                "build_correctness": self.clock.action(
                    calls, "build_correctness", {"status": "passed"}
                ),
                "short_paired": self.clock.action(
                    calls,
                    "short_paired",
                    {"status": "passed", "lower_bound": 0.5, "upper_bound": 1.5},
                ),
                "formal_paired": self.clock.action(
                    calls,
                    "formal_paired",
                    {"status": "passed", "lower_bound": 1.1},
                ),
            }
        )

        self.assertEqual(
            calls, ["static_review", "build_correctness", "short_paired"]
        )
        self.assertEqual(result["elapsed_seconds"], 103.0)
        self.assertEqual(result["projected_spend"]["p90_seconds"], 123.0)
        self.assertTrue(result["soft_target_exceeded"])

    def test_short_pair_missing_lower_bound_never_starts_formal(self) -> None:
        calls = []
        result = self._gate().run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "passed"}
                ),
                "build_correctness": self.clock.action(
                    calls, "build_correctness", {"status": "passed"}
                ),
                "short_paired": self.clock.action(
                    calls, "short_paired", {"status": "passed", "upper_bound": 1.5}
                ),
                "formal_paired": self.clock.action(
                    calls,
                    "formal_paired",
                    {"status": "passed", "lower_bound": 1.1},
                ),
            }
        )

        self.assertEqual(calls.count("formal_paired"), 0)
        self.assertEqual(result["stop_reason"], "short_pair_missing_lower_bound")

    def test_short_pair_invalid_interval_never_starts_formal(self) -> None:
        calls = []
        result = self._gate().run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "passed"}
                ),
                "build_correctness": self.clock.action(
                    calls, "build_correctness", {"status": "passed"}
                ),
                "short_paired": self.clock.action(
                    calls,
                    "short_paired",
                    {"status": "passed", "lower_bound": 2.0, "upper_bound": 1.5},
                ),
                "formal_paired": self.clock.action(
                    calls,
                    "formal_paired",
                    {"status": "passed", "lower_bound": 1.1},
                ),
            }
        )

        self.assertEqual(calls.count("formal_paired"), 0)
        self.assertEqual(result["stop_reason"], "short_pair_invalid_interval")

    def test_short_pair_qualifying_lower_bound_starts_one_formal_confirmation(self) -> None:
        calls = []
        result = self._gate().run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "passed"}
                ),
                "build_correctness": self.clock.action(
                    calls, "build_correctness", {"status": "passed"}
                ),
                "short_paired": self.clock.action(
                    calls,
                    "short_paired",
                    {"status": "passed", "lower_bound": 1.0, "upper_bound": 1.5},
                ),
                "formal_paired": self.clock.action(
                    calls,
                    "formal_paired",
                    {"status": "passed", "lower_bound": 1.1},
                ),
            }
        )

        self.assertEqual(calls.count("formal_paired"), 1)
        self.assertEqual(result["decision"], "PROMOTE")

    def test_repeated_run_returns_same_terminal_result_without_restarting_actions(self) -> None:
        calls = []
        gate = self._gate()
        actions = {
            "static_review": self.clock.action(
                calls, "static_review", {"status": "failed"}
            )
        }

        first = gate.run(actions)
        second = gate.run(actions)

        self.assertEqual(first, second)
        self.assertEqual(calls, ["static_review"])

    def test_short_pair_upper_bound_below_threshold_skips_formal_test(self) -> None:
        calls = []
        actions = {
            "static_review": self.clock.action(calls, "static_review", {"status": "passed"}),
            "build_correctness": self.clock.action(calls, "build_correctness", {"status": "passed"}),
            "short_paired": self.clock.action(
                calls,
                "short_paired",
                {"status": "passed", "estimate": 0.4, "lower_bound": 0.2, "upper_bound": 0.8},
            ),
            "profiler": self.clock.action(calls, "profiler", {"status": "passed"}),
            "formal_paired": self.clock.action(calls, "formal_paired", {"status": "passed"}),
        }

        result = self._gate().run(actions)

        self.assertEqual(calls, ["static_review", "build_correctness", "short_paired"])
        self.assertEqual(result["stop_reason"], "effect_upper_bound_below_minimum")
        self.assertIn("formal_paired", result["skipped_expensive_stages"])

    def test_conclusive_stop_is_well_before_hard_ceiling(self) -> None:
        calls = []
        result = self._gate().run(
            {
                "static_review": self.clock.action(
                    calls, "static_review", {"status": "failed"}, seconds=2.0
                )
            }
        )

        self.assertLess(result["elapsed_seconds"], self.contract["hard_ceiling_seconds"] / 10)
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

    def test_declared_cheapest_falsifier_must_match_the_executable_plan(self) -> None:
        self.candidate["cheapest_falsifier"] = "formal_paired"
        calls = []

        with self.assertRaisesRegex(ValueError, "cheapest_falsifier"):
            self._gate().run(
                {
                    "static_review": self.clock.action(
                        calls, "static_review", {"status": "passed"}
                    ),
                    "build_correctness": self.clock.action(
                        calls, "build_correctness", {"status": "passed"}
                    ),
                    "short_paired": self.clock.action(
                        calls,
                        "short_paired",
                        {"status": "passed", "lower_bound": 0.5, "upper_bound": 2.0},
                    ),
                    "formal_paired": self.clock.action(
                        calls,
                        "formal_paired",
                        {"status": "passed", "lower_bound": 1.1},
                    ),
                }
            )

        self.assertEqual(calls, [])

    def test_cheaper_late_stage_cannot_bypass_cost_order_or_missing_action(self) -> None:
        self.candidate["cheapest_falsifier"] = "profiler"
        self.candidate["estimated_cost"]["static_review"]["p90_seconds"] = 100.0
        self.candidate["estimated_cost"]["profiler"]["p90_seconds"] = 0.5
        calls = []

        with self.assertRaisesRegex(ValueError, "cost|cheapest_falsifier"):
            self._gate().run(
                {
                    "static_review": self.clock.action(
                        calls, "static_review", {"status": "passed"}
                    ),
                    "build_correctness": self.clock.action(
                        calls, "build_correctness", {"status": "passed"}
                    ),
                    "short_paired": self.clock.action(
                        calls,
                        "short_paired",
                        {"status": "passed", "lower_bound": 0.5, "upper_bound": 2.0},
                    ),
                    "formal_paired": self.clock.action(
                        calls,
                        "formal_paired",
                        {"status": "passed", "lower_bound": 1.1},
                    ),
                }
            )

        self.assertEqual(calls, [])

    def test_long_command_emits_heartbeats_and_a_terminal_reason(self) -> None:
        events = []
        result = self.budget.run_budgeted_command(
            [sys.executable, "-c", "import time; time.sleep(0.3)"],
            timeout_seconds=2,
            heartbeat_interval_seconds=0.1,
            event_sink=events.append,
        )

        self.assertGreaterEqual(
            sum(event["event"] == "heartbeat" for event in events), 2
        )
        self.assertEqual(events[-1]["event"], "terminal")
        self.assertEqual(events[-1]["stop_reason"], "completed")
        self.assertEqual(result.stop_reason, "completed")

    def test_output_always_exposes_time_stop_and_skipped_stages(self) -> None:
        result = self._gate().run(
            {"static_review": lambda: {"status": "failed"}}
        )

        self.assertIsInstance(result["elapsed_seconds"], float)
        self.assertEqual(result["stop_reason"], "static_falsified")
        self.assertIsInstance(result["skipped_expensive_stages"], list)

    def test_no_qualified_direction_returns_stop(self) -> None:
        calls = []
        result = self._gate().run(
            {
                "static_review": self.clock.action(calls, "static_review", {"status": "passed"}),
                "build_correctness": self.clock.action(calls, "build_correctness", {"status": "passed"}),
                "short_paired": self.clock.action(
                    calls, "short_paired", {"status": "passed", "lower_bound": 0.1, "upper_bound": 0.2}
                ),
            }
        )

        self.assertEqual(result["decision"], "STOP")
        self.assertNotEqual(result.get("next_action"), "continue_next_round")

    def test_nonfinite_performance_bounds_fail_closed(self) -> None:
        for stage, field, value, expected_reason in (
            ("short_paired", "upper_bound", math.nan, "short_pair_invalid_upper_bound"),
            ("short_paired", "upper_bound", math.inf, "short_pair_invalid_upper_bound"),
            ("formal_paired", "lower_bound", math.nan, "formal_paired_invalid_lower_bound"),
            ("formal_paired", "lower_bound", math.inf, "formal_paired_invalid_lower_bound"),
        ):
            with self.subTest(stage=stage, value=value):
                calls = []
                actions = {
                    "static_review": self.clock.action(
                        calls, "static_review", {"status": "passed"}
                    ),
                    "build_correctness": self.clock.action(
                        calls, "build_correctness", {"status": "passed"}
                    ),
                    "short_paired": self.clock.action(
                        calls,
                        "short_paired",
                        {"status": "passed", "lower_bound": 0.5, "upper_bound": 2.0},
                    ),
                    "profiler": self.clock.action(
                        calls, "profiler", {"status": "passed"}
                    ),
                    "formal_paired": self.clock.action(
                        calls,
                        "formal_paired",
                        {"status": "passed", "lower_bound": 2.0},
                    ),
                }
                outcome = {"status": "passed", field: value}
                if stage == "short_paired":
                    outcome["lower_bound"] = 0.5
                actions[stage] = self.clock.action(calls, stage, outcome)

                result = self._gate().run(actions)

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
                calls = []
                actions = {
                    "static_review": self.clock.action(
                        calls, "static_review", {"status": "passed"}
                    ),
                    "build_correctness": self.clock.action(
                        calls, "build_correctness", {"status": "passed"}
                    ),
                    "short_paired": self.clock.action(
                        calls,
                        "short_paired",
                        {"status": "passed", "lower_bound": 0.5, "upper_bound": 2.0},
                    ),
                    "profiler": self.clock.action(
                        calls, "profiler", {"status": "not_applicable"}
                    ),
                    "formal_paired": self.clock.action(
                        calls,
                        "formal_paired",
                        {"status": "passed", "lower_bound": 2.0},
                    ),
                }
                actions[stage] = self.clock.action(
                    calls, stage, {"status": "not_applicable"}
                )

                result = self._gate().run(actions)

                self.assertEqual(result["decision"], "STOP")
                self.assertEqual(result["stop_reason"], expected_reason)

    def test_soft_target_is_guidance_not_a_direction_timeout(self) -> None:
        self.contract["soft_target_seconds"] = 2.0
        calls = []
        actions = {
            "static_review": self.clock.action(calls, "static_review", {"status": "passed"}),
            "build_correctness": self.clock.action(calls, "build_correctness", {"status": "passed"}),
            "short_paired": self.clock.action(
                calls,
                "short_paired",
                {"status": "passed", "estimate": 1.1, "lower_bound": 0.5, "upper_bound": 2.5},
            ),
            "profiler": self.clock.action(calls, "profiler", {"status": "passed"}),
            "formal_paired": self.clock.action(
                calls,
                "formal_paired",
                {"status": "passed", "estimate": 1.4, "lower_bound": 1.1},
            ),
        }

        result = self._gate().run(actions)

        self.assertIn("formal_paired", calls)
        self.assertGreater(result["elapsed_seconds"], self.contract["soft_target_seconds"])
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
