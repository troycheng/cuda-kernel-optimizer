import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stderr


class VariantExperimentBlackBoxTests(unittest.TestCase):
    def test_experiment_is_not_created_without_a_valid_original_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            missing_baseline = {
                "invocation_id": "missing-baseline",
                "sha256": "0" * 64,
            }
            completed = project.run_tool(
                "workload_evaluate.py",
                "experiment",
                project.experiment_input(missing_baseline),
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(
                decode_stderr(completed)["error_code"],
                "baseline_not_found",
            )
            experiments = project.artifact_root / "experiments"
            self.assertEqual(
                [] if not experiments.exists() else list(experiments.glob("*.json")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
