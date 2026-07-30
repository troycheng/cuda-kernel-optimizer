from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cuda_optimizer_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeProfiler:
    def __init__(self, calls) -> None:
        self.calls = calls

    def cudaProfilerStart(self) -> None:
        self.calls.append("start")

    def cudaProfilerStop(self) -> None:
        self.calls.append("stop")


class _FakeCuda:
    def __init__(self, calls) -> None:
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append("sync")


class ProfileNcuTests(unittest.TestCase):
    def test_ncu_profiles_only_the_explicit_target_range(self) -> None:
        profile_ncu = _load("profile_ncu")
        cmd = profile_ncu._build_profile_command(
            ncu_bin="ncu",
            rep_path="out.ncu-rep",
            benchmark_py="benchmark.py",
            solution="kernel.py",
            dims={"M": 128},
            warmup=3,
            launch_count=1,
        )
        self.assertEqual(cmd[cmd.index("--profile-from-start") + 1], "off")
        self.assertEqual(cmd[cmd.index("--launch-count") + 1], "1")
        self.assertIn("--profile-only", cmd)
        self.assertIn("--target-processes", cmd)

    def test_profile_target_uses_start_then_one_call_then_stop(self) -> None:
        benchmark = _load("benchmark")
        calls = []
        benchmark._profile_target_once(
            lambda: calls.append("kernel"),
            profiler=_FakeProfiler(calls),
            cuda=_FakeCuda(calls),
        )
        self.assertEqual(calls, ["start", "kernel", "sync", "stop"])

    def test_ncu_metric_query_does_not_claim_counter_permission(self) -> None:
        check_env = _load("check_env")
        with mock.patch.object(check_env.shutil, "which", return_value="/usr/bin/ncu"), mock.patch.object(
            check_env,
            "_run",
            side_effect=[(0, "NVIDIA Nsight Compute 13.3", ""), (0, "metric", "")],
        ):
            result = check_env._detect_ncu()
        self.assertTrue(result["metrics_query_available"])
        self.assertIsNone(result["can_read_counters"])

    def test_real_counter_permission_failure_is_recorded_in_state_and_env(self) -> None:
        profile_ncu = _load("profile_ncu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / "env.json"
            state_path = root / "state.json"
            env = {"ncu": {"available": True, "can_read_counters": None}}
            state = {
                "env": env,
                "env_path": str(env_path),
            }
            env_path.write_text(json.dumps(env), encoding="utf-8")
            state_path.write_text(json.dumps(state), encoding="utf-8")

            verdict, note = profile_ncu._counter_access_verdict(
                1,
                "==ERROR== ERR_NVGPUCTRPERM - permission denied",
                report_exists=False,
            )
            profile_ncu._record_counter_access(
                str(state_path), state, verdict, note
            )

            recorded_state = json.loads(state_path.read_text(encoding="utf-8"))
            recorded_env = json.loads(env_path.read_text(encoding="utf-8"))

        self.assertFalse(recorded_state["env"]["ncu"]["can_read_counters"])
        self.assertFalse(recorded_env["ncu"]["can_read_counters"])
        self.assertEqual(recorded_env["ncu"]["counter_access_error"], "ERR_NVGPUCTRPERM")

    def test_successful_real_profile_records_counter_access(self) -> None:
        profile_ncu = _load("profile_ncu")
        verdict, note = profile_ncu._counter_access_verdict(
            0, "==PROF== Disconnected", report_exists=True
        )
        self.assertTrue(verdict)
        self.assertIsNone(note)

    def test_long_form_csv_is_normalized(self) -> None:
        profile_ncu = _load("profile_ncu")
        text = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"target","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","75"\n'
        )
        rows = profile_ncu._parse_ncu_csv(text)
        self.assertEqual(rows[0]["Metric Name"], "dram__throughput.avg.pct_of_peak_sustained_elapsed")

    def test_wide_form_csv_selects_target_kernel(self) -> None:
        profile_ncu = _load("profile_ncu")
        text = (
            '"Kernel Name","gpu__time_duration.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed"\n'
            '"rng_setup","100","10"\n'
            '"target_kernel","20","75"\n'
        )
        rows = profile_ncu._parse_ncu_csv(text, kernel_name_hints=["target_kernel"])
        self.assertTrue(rows)
        self.assertEqual({row["Kernel Name"] for row in rows}, {"target_kernel"})

    def test_profile_csv_emits_stable_semantics_without_heuristic_axis(self) -> None:
        profile_ncu = _load("profile_ncu")
        text = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"target","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","75"\n'
            '"target","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct","%","32"\n'
            '"target","unknown__metric","%","9"\n'
        )

        result = profile_ncu._semantic_ncu_observations(text, "2026.2")

        self.assertEqual(
            [item["semantic_id"] for item in result["semantic_observations"]],
            ["kernel.dram_throughput_pct", "kernel.long_scoreboard_pct"],
        )
        self.assertNotIn("primary_axis", result)
        self.assertEqual(
            result["unmodeled_metrics"],
            [{"metric_name": "unknown__metric", "reason": "unknown_metric"}],
        )

    def test_real_profile_csv_covers_all_kernel_card_positive_semantics(self) -> None:
        profile_ncu = _load("profile_ncu")
        rows = [
            ("dram__bytes.sum", "byte", "4096"),
            ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "75"),
            (
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                "%",
                "32",
            ),
            (
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "%",
                "61",
            ),
            (
                "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
                "%",
                "84",
            ),
            (
                "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
                "%",
                "47",
            ),
            (
                "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct",
                "%",
                "9",
            ),
        ]
        csv_text = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            + "".join(
                f'"target","{metric}","{unit}","{value}"\n'
                for metric, unit, value in rows
            )
        )

        result = profile_ncu._semantic_ncu_observations(csv_text, "2026.2")

        self.assertEqual(
            {
                item["semantic_id"]
                for item in result["semantic_observations"]
            },
            {
                "kernel.dram_bytes",
                "kernel.dram_throughput_pct",
                "kernel.long_scoreboard_pct",
                "kernel.occupancy_pct",
                "kernel.sm_active_pct",
                "kernel.tensor_pipe_pct",
                "kernel.barrier_stall_pct",
            },
        )

    def test_zero_raw_metrics_remain_stable_inputs_not_mechanism_positives(
        self,
    ) -> None:
        profile_ncu = _load("profile_ncu")
        cards = json.loads(
            (
                SCRIPTS.parent / "references" / "diagnostic_cards.json"
            ).read_text(encoding="utf-8")
        )["cards"]
        kernel_mechanisms = {
            "global_memory_transactions",
            "redundant_dram_traffic",
            "memory_latency_hiding",
            "register_or_shared_pressure",
            "parallelism_or_wave_tail",
            "compute_pipeline_or_dtype",
            "synchronization_or_atomic_contention",
        }
        card_positives = {
            rule["semantic_id"]
            for card in cards
            if card["mechanism_key"] in kernel_mechanisms
            for rule in card["observation_rules"]["positive"]
        }
        metrics = [
            ("dram__bytes.sum", "byte"),
            ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%"),
            (
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                "%",
            ),
            ("sm__warps_active.avg.pct_of_peak_sustained_active", "%"),
            ("sm__cycles_active.avg.pct_of_peak_sustained_elapsed", "%"),
            (
                "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
                "%",
            ),
            (
                "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct",
                "%",
            ),
        ]
        csv_text = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            + "".join(
                f'"target","{metric}","{unit}","0"\n'
                for metric, unit in metrics
            )
        )

        result = profile_ncu._semantic_ncu_observations(csv_text, "2026.2")
        raw_semantics = {
            item["semantic_id"] for item in result["semantic_observations"]
        }

        self.assertEqual(len(raw_semantics), 7)
        self.assertTrue(card_positives.isdisjoint(raw_semantics))

    def test_profile_main_uses_actual_ncu_version_not_state_snapshot(self) -> None:
        profile_ncu = _load("profile_ncu")
        csv_text = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"target","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","75"\n'
        )
        cases = [
            ("2026.3", "2026.2", ["kernel.dram_throughput_pct"]),
            ("2026.2", "2026.3", []),
        ]
        for state_version, actual_version, expected_ids in cases:
            with self.subTest(
                state_version=state_version, actual_version=actual_version
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                solution = root / "kernel.py"
                solution.write_text("def solve(): pass\n", encoding="utf-8")
                ncu = root / "ncu"
                ncu.write_text(
                    "#!/bin/sh\n"
                    f"printf 'NVIDIA Nsight Compute {actual_version}\\n'\n",
                    encoding="utf-8",
                )
                ncu.chmod(0o700)
                state_path = root / "state.json"
                state_path.write_text(
                    json.dumps(
                        {
                            "run_dir": str(root),
                            "best_file": str(solution),
                            "dims": {},
                            "ptr_size": 0,
                            "ncu_num": 5,
                            "env": {
                                "ncu": {
                                    "available": True,
                                    "path": str(ncu),
                                    "version": state_version,
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                def profile(**kwargs):
                    Path(kwargs["rep_path"]).write_bytes(b"report")
                    return 0, "profiled"

                argv = [
                    "profile_ncu.py",
                    "--state",
                    str(state_path),
                    "--iter",
                    "1",
                    "--which",
                    "best_input",
                ]
                with mock.patch.object(
                    profile_ncu, "_run_ncu_profile", side_effect=profile
                ), mock.patch.object(
                    profile_ncu,
                    "_import_metrics_csv",
                    return_value=(0, csv_text, ""),
                ), mock.patch.object(profile_ncu.sys, "argv", argv):
                    profile_ncu.main()

                result = json.loads(
                    (root / "iterv1" / "ncu_top.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    [
                        item["semantic_id"]
                        for item in result["semantic_observations"]
                    ],
                    expected_ids,
                )

    def test_ncu_version_query_timeout_returns_none(self) -> None:
        profile_ncu = _load("profile_ncu")
        with mock.patch.object(
            profile_ncu.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["ncu", "--version"], 2),
        ) as run:
            self.assertIsNone(profile_ncu._query_ncu_version("ncu", timeout=2))
        run.assert_called_once_with(
            ["ncu", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
        )

    def test_all_metric_sample_units_fail_closed_independent_of_row_order(self) -> None:
        profile_ncu = _load("profile_ncu")
        metric = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        cases = [
            (("%", ""), "missing_unit"),
            (("", "%"), "missing_unit"),
            (("%", "percent"), "inconsistent_units"),
            (("percent", "%"), "inconsistent_units"),
        ]
        for units, reason in cases:
            with self.subTest(units=units, reason=reason):
                text = (
                    '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                    f'"kernel-a","{metric}","{units[0]}","70"\n'
                    f'"kernel-b","{metric}","{units[1]}","80"\n'
                )
                result = profile_ncu._semantic_ncu_observations(text, "2026.2")
                self.assertEqual(result["semantic_observations"], [])
                self.assertEqual(
                    result["unmodeled_metrics"],
                    [{"metric_name": metric, "reason": reason}],
                )

    def test_invalid_value_sample_cannot_hide_its_unit_failure(self) -> None:
        profile_ncu = _load("profile_ncu")
        metric = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        cases = [
            ((("%", "70"), ("", "invalid")), "missing_unit"),
            ((("", "invalid"), ("%", "70")), "missing_unit"),
            ((("%", "70"), ("percent", "invalid")), "inconsistent_units"),
            ((("percent", "invalid"), ("%", "70")), "inconsistent_units"),
        ]
        for samples, reason in cases:
            with self.subTest(samples=samples, reason=reason):
                text = (
                    '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                    f'"kernel-a","{metric}","{samples[0][0]}","{samples[0][1]}"\n'
                    f'"kernel-b","{metric}","{samples[1][0]}","{samples[1][1]}"\n'
                )
                result = profile_ncu._semantic_ncu_observations(text, "2026.2")
                self.assertEqual(result["semantic_observations"], [])
                self.assertEqual(
                    result["unmodeled_metrics"],
                    [{"metric_name": metric, "reason": reason}],
                )


if __name__ == "__main__":
    unittest.main()
