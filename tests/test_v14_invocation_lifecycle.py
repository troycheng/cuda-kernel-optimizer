import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project


class InvocationLifecycleBlackBoxTests(unittest.TestCase):
    def test_identical_baseline_request_reuses_one_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            check = project.check()
            operation_input = project.baseline_input()
            first_completed = project.run_tool(
                "workload_evaluate.py",
                "baseline",
                operation_input,
                wait=True,
            )
            second_completed = project.run_tool(
                "workload_evaluate.py",
                "baseline",
                operation_input,
                wait=True,
            )
            self.assertEqual(
                [first_completed.returncode, second_completed.returncode],
                [0, 0],
            )
            import json

            first = json.loads(first_completed.stdout)
            second = json.loads(second_completed.stdout)
            self.assertEqual(first["invocation_id"], second["invocation_id"])

            events = [
                event
                for event in project.driver_events()
                if event["execution_id"] != check["probe_id"]
            ]
            self.assertEqual(
                [(event["operation"], event["roles"]) for event in events],
                [("baseline", ["original"])],
            )
            invocation_dirs = list(
                (project.artifact_root / "invocations").glob("*")
            )
            self.assertEqual(len(invocation_dirs), 1)


if __name__ == "__main__":
    unittest.main()
