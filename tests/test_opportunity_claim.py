from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

from tests.v14_support import V14Project, decode_stderr, decode_stdout


EVALUATOR = Path(__file__).resolve().parents[1] / "skills" / "cuda-kernel-optimizer" / "scripts" / "workload_evaluate.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("opportunity_evaluator", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OpportunityClaimTests(unittest.TestCase):
    def _request(self, directory: str) -> tuple[V14Project, dict]:
        project = V14Project(Path(directory))
        self.assertEqual(project.run_tool("readiness.py", "check", project.readiness_input()).returncode, 0)
        baseline_ref = project.baseline()["result_ref"]
        return project, project.experiment_input(baseline_ref)

    def test_real_production_numbers_are_below_the_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, request = self._request(directory)
            claim = request["opportunity_claim"]
            claim["denominator_us"] = 4314.0
            claim["pools"][0].update(reference_time_us=2.899, candidate_time_us=2.316, occurrences=10)
            completed = project.run_tool("workload_evaluate.py", "experiment", request)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(decode_stderr(completed)["error_code"], "opportunity_below_minimum_effect")

    def test_real_case_arithmetic_distinguishes_full_removal_from_prototype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _project, request = self._request(directory)
            claim = request["opportunity_claim"]
            claim["denominator_us"] = 4314.0
            pool = claim["pools"][0]
            pool.update(reference_time_us=2.899, candidate_time_us=None, candidate_evidence=None, occurrences=10)
            module = _load_evaluator()
            target = {"test_suite": {"case_ids": ["main"]}, "minimum_effect": {"value": 0.5, "unit": "percent"}, "primary_metric": {"direction": "higher"}}
            claim["primary_model"] = "inverse_time"
            full_removal = module._opportunity_claim(claim, target)
            self.assertAlmostEqual(full_removal["bound"]["e2e_ceiling_percent"], 0.6765, places=3)
            self.assertAlmostEqual(full_removal["bound"]["required_candidate_time_us"], 0.7527, places=3)
            pool["candidate_time_us"] = 2.316
            pool["candidate_evidence"] = pool["reference_evidence"]
            target["minimum_effect"]["value"] = 0.1
            prototype = module._opportunity_claim(claim, target)
            self.assertAlmostEqual(prototype["bound"]["e2e_ceiling_percent"], 0.1353, places=3)

    def test_eager_evidence_cannot_claim_an_inductor_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, request = self._request(directory)
            request["opportunity_claim"]["pools"][0]["reference_evidence"]["execution_form"]["lowering"] = "eager"
            completed = project.run_tool("workload_evaluate.py", "experiment", request)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(decode_stderr(completed)["error_code"], "opportunity_execution_mismatch")

    def test_candidate_scope_cannot_inherit_another_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, request = self._request(directory)
            request["opportunity_claim"]["candidate_components"] = ["w2"]
            completed = project.run_tool("workload_evaluate.py", "experiment", request)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(decode_stderr(completed)["error_code"], "opportunity_scope_mismatch")

    def test_a_justified_conservative_reference_remains_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project, request = self._request(directory)
            evidence = request["opportunity_claim"]["pools"][0]["reference_evidence"]
            evidence["relationship"] = "conservative_upper_bound"
            evidence["execution_form"]["lowering"] = "narrow-production-trace"
            evidence["reason"] = "measured superset bounds the selected production component"
            completed = project.run_tool("workload_evaluate.py", "experiment", request)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("experiment_ref", decode_stdout(completed))


if __name__ == "__main__":
    unittest.main()
