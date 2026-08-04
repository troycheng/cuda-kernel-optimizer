import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stdout


class TargetBaselineBlackBoxTests(unittest.TestCase):
    def test_diagnostic_target_is_frozen_without_smoke_or_baseline_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            request = {
                "format_version": "cuda-kernel-optimizer/readiness-input-v1",
                "operation": "check",
                "artifact_root": str(project.artifact_root),
                "target_mode": "diagnostic",
                "claim_layer": "diagnostic",
                "original": {"kind": "source_snapshot", "path": str(project.original)},
                "materials": [
                    {
                        "kind": "report",
                        "path": str(project.correctness_reference),
                        "tool": "pytorch_profiler",
                        "tool_version": "2.13.0",
                        "dialect": "chrome-trace-v1",
                    }
                ],
                "environment_requirements": {
                    "gpu_uuids": [], "required_tools": []
                },
                "scan_limits": {
                    "max_files": 100,
                    "max_total_bytes": 1024 * 1024,
                    "max_wall_seconds": 2.0,
                },
            }
            invalid = project.run_tool(
                "readiness.py",
                "check",
                {**request, "claim_layer": "workload"},
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertIn(
                "claim_layer must be diagnostic",
                invalid.stderr,
            )
            self.assertFalse(project.artifact_root.exists())

            checked = project.run_tool("readiness.py", "check", request)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            result = decode_stdout(checked)
            self.assertEqual(result["status"], "ready")
            target = __import__("json").loads(
                (project.artifact_root / "target.json").read_text(encoding="utf-8")
            )
            self.assertEqual(target["target_mode"], "diagnostic")
            self.assertEqual(target["environment"]["runtime"]["status"], "unavailable")
            self.assertEqual(target["driver"]["status"], "unavailable")
            self.assertEqual(len(target["diagnostic_materials"]), 1)
            material = target["diagnostic_materials"][0]
            self.assertRegex(material["id"], r"^[0-9a-f]{64}$")
            self.assertEqual(material["sha256"], material["object_ref"]["digest"])
            self.assertEqual(material["tool_version"], "2.13.0")
            self.assertEqual(material["dialect"], "chrome-trace-v1")
            self.assertEqual(list((project.artifact_root / "invocations").iterdir()), [])

            baseline = project.run_tool(
                "workload_evaluate.py", "baseline", project.baseline_input(), wait=True
            )
            self.assertNotEqual(baseline.returncode, 0)
            self.assertIn("target_not_optimizable", baseline.stderr)
            self.assertEqual(project.driver_events(), [])

    def test_failed_correctness_invalidates_combined_baseline_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))

            checked = project.run_tool(
                "readiness.py",
                "check",
                project.readiness_input(),
            )
            self.assertEqual(
                checked.returncode,
                0,
                f"readiness failed:\nstdout={checked.stdout}\nstderr={checked.stderr}",
            )
            check_result = decode_stdout(checked)
            self.assertEqual(check_result["status"], "ready")
            self.assertTrue((project.artifact_root / "target.json").is_file())

            project.set_behavior(correctness="failed")
            baseline = project.run_tool(
                "workload_evaluate.py",
                "baseline",
                project.baseline_input(),
                wait=True,
            )
            self.assertEqual(
                baseline.returncode,
                0,
                f"baseline failed:\nstdout={baseline.stdout}\nstderr={baseline.stderr}",
            )
            baseline_result = decode_stdout(baseline)
            self.assertEqual(baseline_result["measurement_validity"], "invalid")
            self.assertEqual(baseline_result["verdict"], "failed")
            self.assertEqual(baseline_result["skipped_expensive_stages"], [])

            baseline_events = [
                event
                for event in project.driver_events()
                if event["execution_id"] != check_result["probe_id"]
            ]
            self.assertEqual(
                [event["mode"] for event in baseline_events],
                ["combined"],
            )

    def test_driver_pass_status_cannot_override_frozen_correctness_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            checked = project.check()
            project.set_behavior(
                correctness="passed",
                correctness_metric=0.0,
            )
            baseline = project.run_tool(
                "workload_evaluate.py",
                "baseline",
                project.baseline_input(),
                wait=True,
            )
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            result = decode_stdout(baseline)
            self.assertEqual(result["measurement_validity"], "invalid")
            self.assertEqual(result["stop_reason"], "correctness_failed")
            events = [
                event
                for event in project.driver_events()
                if event["execution_id"] != checked["probe_id"]
            ]
            self.assertEqual([event["mode"] for event in events], ["combined"])

    def test_baseline_rejects_sample_count_mismatch_before_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.set_behavior(original_samples=[10.0, 10.1])
            readiness = project.readiness_input()
            readiness["driver"]["execution_mode"] = "combined"
            readiness["smoke"]["mode"] = "combined"
            checked = project.run_tool("readiness.py", "check", readiness)
            self.assertEqual(checked.returncode, 0, checked.stderr)

            project.set_behavior(original_samples=[10.0])
            baseline_request = project.baseline_input()
            baseline_request["sampling_design"]["samples_per_case"] = 3
            baseline = project.run_tool(
                "workload_evaluate.py",
                "baseline",
                baseline_request,
                wait=True,
            )

            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            result = decode_stdout(baseline)
            self.assertEqual(result["execution_status"], "failed")
            self.assertEqual(result["measurement_validity"], "invalid")
            self.assertEqual(result["stop_reason"], "sample_count_mismatch")
            self.assertEqual(len(result["command_receipts"]), 1)
            command_result = result["command_receipts"][0]["command_result"]
            self.assertEqual(command_result["stop_reason"], "sample_count_mismatch")
            self.assertIn(
                "path=measurements.primary.samples expected=3 observed=1",
                command_result["stderr"],
            )

    def test_baseline_preserves_exact_constraint_contract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            project.set_behavior(
                constraints=[
                    {
                        "name": "p99_ttft",
                        "unit": "ms",
                        "samples": [2.0, 2.1],
                    }
                ]
            )

            baseline = project.run_tool(
                "workload_evaluate.py",
                "baseline",
                project.baseline_input(),
                wait=True,
            )

            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            result = decode_stdout(baseline)
            self.assertEqual(result["stop_reason"], "constraint_metric_mismatch")
            command_result = result["command_receipts"][0]["command_result"]
            self.assertEqual(
                command_result["stop_reason"], "constraint_metric_mismatch"
            )
            self.assertIn("observed=['p99_ttft']", command_result["stderr"])


if __name__ == "__main__":
    unittest.main()
