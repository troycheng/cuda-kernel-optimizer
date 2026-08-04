from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stderr


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "readiness.py"


class ReadinessPublicSurfaceTests(unittest.TestCase):
    def test_help_exposes_only_check(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("{check}", completed.stdout)

    def test_unknown_operation_is_rejected_before_artifact_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            completed = project.run_tool(
                "readiness.py", "unknown", project.readiness_input()
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(project.artifact_root.exists())

    def test_legacy_field_is_rejected_before_artifact_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            request = project.readiness_input()
            request["state"] = "legacy-run-state"
            completed = project.run_tool("readiness.py", "check", request)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(project.artifact_root.exists())

    def test_optimization_rejects_separate_driver_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            request = project.readiness_input()
            request["driver"]["execution_mode"] = "separate"
            request["smoke"]["mode"] = "correctness"

            completed = project.run_tool(
                "readiness.py", "check", request
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("combined driver", completed.stderr)
            self.assertEqual(project.driver_events(), [])
            self.assertFalse(project.artifact_root.exists())

    def test_combined_smoke_rejects_an_undeclared_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.set_behavior(
                constraints=[
                    {
                        "name": "p99_ttft",
                        "unit": "ms",
                        "samples": [2.0, 2.1],
                    }
                ]
            )
            request = project.readiness_input()
            request["driver"]["execution_mode"] = "combined"
            request["smoke"]["mode"] = "combined"

            completed = project.run_tool("readiness.py", "check", request)

            self.assertEqual(completed.returncode, 2)
            self.assertIn("measurements.constraints", completed.stderr)
            self.assertIn("expected=[]", completed.stderr)
            self.assertIn("observed=['p99_ttft']", completed.stderr)
            self.assertFalse(project.artifact_root.exists())

    def test_combined_smoke_requires_two_measurement_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.set_behavior(original_samples=[10.0])
            request = project.readiness_input()
            request["driver"]["execution_mode"] = "combined"
            request["smoke"]["mode"] = "combined"

            completed = project.run_tool("readiness.py", "check", request)

            self.assertEqual(completed.returncode, 2)
            self.assertIn("sample_count_mismatch", completed.stderr)
            self.assertIn("path=measurements.primary.samples", completed.stderr)
            self.assertIn("expected=2 observed=1", completed.stderr)
            self.assertFalse(project.artifact_root.exists())

    def test_smoke_command_failure_preserves_actionable_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.driver.write_text(
                "import sys\n"
                "print('ready-out')\n"
                "print('ready-err', file=sys.stderr)\n"
                "raise SystemExit(7)\n",
                encoding="utf-8",
            )
            request = project.readiness_input()
            request["driver"]["execution_mode"] = "combined"
            request["smoke"]["mode"] = "combined"

            completed = project.run_tool("readiness.py", "check", request)

            self.assertEqual(completed.returncode, 2)
            error = decode_stderr(completed)["error"]
            for expected in (
                "stop_reason=command_failed",
                "returncode=7",
                "stdout='ready-out\\n'",
                "stderr='ready-err\\n'",
                "cleanup_status=confirmed",
            ):
                self.assertIn(expected, error)
            self.assertFalse(project.artifact_root.exists())

    def test_smoke_command_failure_bounds_streams_before_the_error_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.driver.write_text(
                "import sys\n"
                "print('X' * 2000)\n"
                "print('ACTIONABLE-ERR-' + 'Y' * 2000, file=sys.stderr)\n"
                "raise SystemExit(7)\n",
                encoding="utf-8",
            )

            completed = project.run_tool(
                "readiness.py", "check", project.readiness_input()
            )

            self.assertEqual(completed.returncode, 2)
            error = decode_stderr(completed)["error"]
            for expected in (
                "stop_reason=command_failed",
                "returncode=7",
                "stdout=",
                "stderr='ACTIONABLE-ERR-",
                "<truncated>",
                "cleanup_status=confirmed",
            ):
                self.assertIn(expected, error)
            self.assertFalse(project.artifact_root.exists())

    def test_combined_readiness_accepts_a_closed_two_sample_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))

            completed = project.run_tool(
                "readiness.py", "check", project.readiness_input()
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(project.artifact_root.joinpath("target.json").is_file())

    def test_optimization_rejects_correctness_only_smoke_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            request = project.readiness_input()
            request["smoke"]["mode"] = "correctness"

            completed = project.run_tool("readiness.py", "check", request)

            self.assertEqual(completed.returncode, 2)
            self.assertIn("combined smoke", completed.stderr)
            self.assertEqual(project.driver_events(), [])
            self.assertFalse(project.artifact_root.exists())


if __name__ == "__main__":
    unittest.main()
