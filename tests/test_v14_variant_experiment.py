import json
import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stderr, decode_stdout


class VariantExperimentBlackBoxTests(unittest.TestCase):
    def test_material_premise_cannot_claim_tool_verified_fact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            baseline = project.baseline()
            request = project.experiment_input(baseline["result_ref"])
            request["material_premises"] = [{
                "statement": "an unverified statement",
                "component": "cuda",
                "version": "unknown",
                "status": "documented_fact",
                "source": "not verified",
                "decision_effect": "remove synchronization",
            }]
            completed = project.run_tool("workload_evaluate.py", "experiment", request)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("status is unsupported", completed.stderr)

    def test_comparison_meaning_and_unresolved_premises_are_frozen_per_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            baseline = project.baseline()
            gate_input = project.experiment_input(baseline["result_ref"])
            gate_input["comparison_contract"]["additional_gates"] = [
                {"metric": "fidelity", "operator": "greater_or_equal", "value": 1.0}
            ]
            diagnostic_input = project.experiment_input(baseline["result_ref"])
            diagnostic_input["comparison_contract"]["diagnostics"] = ["fidelity"]
            diagnostic_input["material_premises"] = [{
                "statement": "the runtime may select the candidate path",
                "component": "fixture-runtime",
                "version": "test",
                "status": "unresolved_hypothesis",
                "source": "not yet verified",
                "decision_effect": "requires a discriminating runtime observation",
            }]

            references = []
            for request in (gate_input, diagnostic_input):
                completed = project.run_tool(
                    "workload_evaluate.py", "experiment", request
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                references.append(decode_stdout(completed)["experiment_ref"])

            records = [
                json.loads(
                    (project.artifact_root / "experiments" / f"{ref['id']}.json").read_text("utf-8")
                )
                for ref in references
            ]
            self.assertNotEqual(references[0], references[1])
            self.assertEqual(records[0]["comparison_contract"]["additional_gates"][0]["metric"], "fidelity")
            self.assertEqual(records[1]["comparison_contract"]["diagnostics"], ["fidelity"])
            self.assertEqual(records[1]["material_premises"][0]["status"], "unresolved_hypothesis")

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
