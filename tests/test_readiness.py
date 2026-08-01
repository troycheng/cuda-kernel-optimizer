from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project


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


if __name__ == "__main__":
    unittest.main()
