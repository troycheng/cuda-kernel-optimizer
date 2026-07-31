from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "workload_evaluate.py"


class EvaluatorPublicSurfaceTests(unittest.TestCase):
    def test_help_exposes_only_v14_operations(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "{baseline,experiment,screen,target,final_audit,status,cancel}",
            completed.stdout,
        )

    def test_unknown_operation_does_not_create_an_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            before = list((project.artifact_root / "invocations").iterdir())
            completed = project.run_tool(
                "workload_evaluate.py", "legacy_measure", project.baseline_input()
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(list((project.artifact_root / "invocations").iterdir()), before)

    def test_legacy_field_is_rejected_before_invocation_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            request = project.baseline_input()
            request["state"] = "legacy-run-state"
            before = list((project.artifact_root / "invocations").iterdir())
            completed = project.run_tool("workload_evaluate.py", "baseline", request)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(list((project.artifact_root / "invocations").iterdir()), before)


if __name__ == "__main__":
    unittest.main()
