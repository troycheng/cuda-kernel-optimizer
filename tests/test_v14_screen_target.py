import json
import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stderr, decode_stdout


class ScreenTargetBlackBoxTests(unittest.TestCase):
    def test_diagnostic_proxy_screen_does_not_start_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            baseline = project.baseline()
            created = project.run_tool(
                "workload_evaluate.py",
                "experiment",
                project.experiment_input(baseline["result_ref"]),
            )
            self.assertEqual(
                created.returncode,
                0,
                f"experiment failed:\nstdout={created.stdout}\nstderr={created.stderr}",
            )
            experiment = decode_stdout(created)

            project.set_behavior(candidate_samples=[10.1])
            screened = project.run_tool(
                "workload_evaluate.py",
                "screen",
                project.screen_input(experiment["experiment_ref"]),
                wait=True,
            )
            self.assertEqual(
                screened.returncode,
                0,
                f"screen failed:\nstdout={screened.stdout}\nstderr={screened.stderr}",
            )
            result = decode_stdout(screened)
            self.assertEqual(result["measurement_validity"], "valid")
            self.assertEqual(result["verdict"], "inconclusive")

            operations = {
                json.loads(path.read_text(encoding="utf-8"))["operation"]
                for path in (project.artifact_root / "invocations").glob(
                    "*/request.json"
                )
            }
            self.assertEqual(operations, {"baseline", "screen"})
            self.assertFalse(
                (project.artifact_root / "champion" / "current.json").exists()
            )

    def test_valid_non_rejected_screen_allows_one_formal_target_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            baseline = project.baseline()
            created = project.run_tool(
                "workload_evaluate.py",
                "experiment",
                project.experiment_input(baseline["result_ref"]),
            )
            experiment_ref = decode_stdout(created)["experiment_ref"]
            screened = project.run_tool(
                "workload_evaluate.py",
                "screen",
                project.screen_input(experiment_ref),
                wait=True,
            )
            self.assertEqual(screened.returncode, 0, screened.stderr)

            compared = project.run_tool(
                "workload_evaluate.py",
                "target",
                project.target_input(experiment_ref),
                wait=True,
            )
            self.assertEqual(
                compared.returncode,
                0,
                f"target failed:\nstdout={compared.stdout}\nstderr={compared.stderr}",
            )
            result = decode_stdout(compared)
            self.assertEqual(result["operation"], "target")
            self.assertEqual(result["measurement_validity"], "valid")
            self.assertEqual(result["verdict"], "passed")
            self.assertEqual(
                result["performance_receipt"]["reference_status"],
                "current",
            )
            self.assertFalse(
                (project.artifact_root / "champion" / "current.json").exists()
            )

    def test_failed_screen_cannot_be_bypassed_by_changing_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            baseline = project.baseline()
            created = project.run_tool(
                "workload_evaluate.py",
                "experiment",
                project.experiment_input(baseline["result_ref"]),
            )
            experiment_ref = decode_stdout(created)["experiment_ref"]
            first_input = project.screen_input(experiment_ref)
            first_input["command_timeout_seconds"] = 0.000001
            first = project.run_tool(
                "workload_evaluate.py",
                "screen",
                first_input,
                wait=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(
                decode_stdout(first)["measurement_validity"],
                "valid",
            )

            changed_input = project.screen_input(experiment_ref)
            changed_input["sampling_design"]["seed"] = 7
            second = project.run_tool(
                "workload_evaluate.py",
                "screen",
                changed_input,
                wait=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            compared = project.run_tool(
                "workload_evaluate.py",
                "target",
                project.target_input(experiment_ref),
                wait=True,
            )
            self.assertEqual(compared.returncode, 2)
            self.assertEqual(
                decode_stderr(compared)["error_code"],
                "screen_attempt_changed",
            )


if __name__ == "__main__":
    unittest.main()
