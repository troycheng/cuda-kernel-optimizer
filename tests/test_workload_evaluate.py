from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.v14_support import V14Project


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "workload_evaluate.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("workload_evaluate_unit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_experiment_rejects_an_unexecutable_falsifier_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            request = project.experiment_input(project.baseline()["result_ref"])
            request["cheapest_falsifier"] = {
                "kind": "command",
                "reason": "no command contract exists in this record",
            }

            with self.assertRaisesRegex(ValueError, "must be none"):
                _load_evaluator().create_experiment(request)

    def test_experiment_record_failure_leaves_no_candidate_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            baseline_ref = project.baseline()["result_ref"]
            evaluator = _load_evaluator()
            object_root = project.artifact_root / "objects" / "sha256"
            before = {path.name for path in object_root.iterdir()}
            real_create = evaluator.STORE.create_regular_json

            def fail_experiment(path, value):
                if Path(path).parent.name == "experiments":
                    raise OSError("simulated experiment publication failure")
                return real_create(path, value)

            with mock.patch.object(
                evaluator.STORE,
                "create_regular_json",
                side_effect=fail_experiment,
            ):
                with self.assertRaisesRegex(OSError, "simulated experiment"):
                    evaluator.create_experiment(
                        project.experiment_input(baseline_ref)
                    )

            self.assertEqual(
                {path.name for path in object_root.iterdir()},
                before,
            )

    def test_experiment_record_failure_preserves_concurrently_reused_candidate_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            baseline_ref = project.baseline()["result_ref"]
            evaluator = _load_evaluator()
            real_promote = evaluator.STORE._promote_staged_object
            real_create = evaluator.STORE.create_regular_json
            promotions = []

            def concurrent_reuse(artifact_root, staging_root, object_ref):
                source = Path(staging_root) / object_ref["locator"]
                destination = Path(artifact_root) / object_ref["locator"]
                if not destination.exists():
                    shutil.copytree(source, destination)
                outcome = real_promote(artifact_root, staging_root, object_ref)
                promotions.append(outcome)
                return outcome

            def fail_experiment(path, value):
                if Path(path).parent.name == "experiments":
                    raise OSError("simulated experiment publication failure")
                return real_create(path, value)

            with mock.patch.object(
                evaluator.STORE,
                "_promote_staged_object",
                side_effect=concurrent_reuse,
            ), mock.patch.object(
                evaluator.STORE,
                "create_regular_json",
                side_effect=fail_experiment,
            ):
                with self.assertRaisesRegex(OSError, "simulated experiment"):
                    evaluator.create_experiment(
                        project.experiment_input(baseline_ref)
                    )

            self.assertEqual(len(promotions), 1)
            self.assertFalse(promotions[0]["published"])
            reused_ref = promotions[0]["object_ref"]
            self.assertTrue(
                (project.artifact_root / reused_ref["locator"] / "manifest.json").is_file()
            )

    def test_readiness_and_evaluator_use_only_driver_output_bundle_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            target = json.loads(
                (project.artifact_root / "target.json").read_text(encoding="utf-8")
            )
            readiness = target["readiness_evidence"]
            self.assertEqual(
                {"driver_output_ref", "driver_artifacts"}.intersection(readiness),
                {"driver_output_ref", "driver_artifacts"},
            )
            self.assertEqual(readiness["driver_output_ref"]["source_kind"], "directory")

            baseline = project.baseline()
            for receipt in baseline["command_receipts"]:
                self.assertEqual(
                    set(receipt),
                    {"request", "command_result", "driver_output_ref", "driver_artifacts"},
                )
                self.assertEqual(receipt["driver_output_ref"]["source_kind"], "directory")
            self.assertNotIn("driver_result_ref", json.dumps(target))
            self.assertNotIn("driver_result_ref", json.dumps(baseline))


if __name__ == "__main__":
    unittest.main()
