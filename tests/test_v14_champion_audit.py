import json
import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stderr, decode_stdout, sha256_file, write_json


class ChampionAuditBlackBoxTests(unittest.TestCase):
    def test_dangling_current_pointer_is_not_treated_as_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            champion_dir = project.artifact_root / "champion"
            champion_dir.mkdir(exist_ok=True)
            (champion_dir / "current.json").symlink_to("missing.json")

            completed = project.run_tool(
                "champion.py",
                "show",
                {
                    "format_version": "cuda-kernel-optimizer/champion-input-v1",
                    "operation": "show",
                    "artifact_root": str(project.artifact_root),
                    "target_ref": project.target_ref(),
                },
            )

            self.assertEqual(completed.returncode, 2)
            self.assertEqual(
                decode_stderr(completed)["error_code"],
                "selection_invalid",
            )

    def _formal_target(self, project: V14Project) -> dict:
        baseline = project.baseline()
        experiment = decode_stdout(
            project.run_tool(
                "workload_evaluate.py",
                "experiment",
                project.experiment_input(baseline["result_ref"]),
            )
        )
        experiment_ref = experiment["experiment_ref"]
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
        self.assertEqual(compared.returncode, 0, compared.stderr)
        return decode_stdout(compared)

    def _result_ref(self, project: V14Project, name: str, result: dict) -> dict:
        invocation = project.artifact_root / "invocations" / name
        path = invocation / "result.json"
        write_json(path, result)
        return {"invocation_id": name, "sha256": sha256_file(path)}

    def _candidate(self) -> dict:
        return {
            "role": "candidate",
            "kind": "source_snapshot",
            "digest": "b" * 64,
            "locator": "objects/sha256/" + "b" * 64,
        }

    def _target_result(
        self, project: V14Project, *, reference: dict, selection_ref
    ) -> dict:
        return {
            "record_type": "invocation_result",
            "format_version": "cuda-kernel-optimizer/evaluator-result-v1",
            "operation": "target",
            "target_ref": project.target_ref(),
            "variant_refs": [reference, self._candidate()],
            "reference_selection_ref": selection_ref,
            "experiment_ref": {"id": "exp-test", "sha256": "a" * 64},
            "cleanup_status": "confirmed",
            "execution_status": "succeeded",
            "measurement_validity": "valid",
            "verdict": "passed",
            "reference_status": "current",
            "performance_receipt": {
                "status": "valid",
                "reference_status": "current",
                "reference": reference,
                "candidate": self._candidate(),
            },
        }

    def _champion_request(
        self, project: V14Project, operation: str, result_ref: dict, expected
    ) -> dict:
        return {
            "format_version": "cuda-kernel-optimizer/champion-input-v1",
            "operation": operation,
            "artifact_root": str(project.artifact_root),
            "target_ref": project.target_ref(),
            "result_ref": result_ref,
            "expected_selection_ref": expected,
        }

    def test_champion_starts_as_original_without_creating_a_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            completed = project.run_tool(
                "champion.py",
                "show",
                {
                    "format_version": "cuda-kernel-optimizer/champion-input-v1",
                    "operation": "show",
                    "artifact_root": str(project.artifact_root),
                    "target_ref": project.target_ref(),
                },
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"show failed:\nstdout={completed.stdout}\nstderr={completed.stderr}",
            )
            result = decode_stdout(completed)
            self.assertEqual(result["status"], "current")
            self.assertIsNone(result["selection_ref"])
            self.assertEqual(result["variant"]["role"], "original")
            self.assertFalse(
                (project.artifact_root / "champion" / "current.json").exists()
            )

    def test_select_adopts_the_passed_candidate_bound_to_current_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            target_result = self._formal_target(project)

            completed = project.run_tool(
                "champion.py",
                "select",
                self._champion_request(
                    project,
                    "select",
                    target_result["result_ref"],
                    None,
                ),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = decode_stdout(completed)
            self.assertEqual(result["status"], "selected")
            self.assertEqual(result["target_ref"], project.target_ref())
            self.assertEqual(
                result["variant"],
                target_result["variant_refs"][1],
            )
            self.assertTrue((project.artifact_root / "champion" / "current.json").is_file())

    def test_select_rejects_result_bound_to_a_different_current_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            result_ref = self._result_ref(
                project,
                "inv-target",
                self._target_result(
                    project,
                    reference={"role": "candidate", "digest": "c" * 64},
                    selection_ref=None,
                ),
            )

            completed = project.run_tool(
                "champion.py",
                "select",
                self._champion_request(project, "select", result_ref, None),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertEqual(decode_stderr(completed)["error_code"], "result_invalid")
            self.assertFalse((project.artifact_root / "champion" / "current.json").exists())

    def test_restore_original_accepts_a_rejecting_audit_bound_to_current_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            original = json.loads(
                (project.artifact_root / "target.json").read_text("utf-8")
            )["original"]
            target_result = self._formal_target(project)
            selected = decode_stdout(
                project.run_tool(
                    "champion.py",
                    "select",
                    self._champion_request(
                        project,
                        "select",
                        target_result["result_ref"],
                        None,
                    ),
                )
            )
            selection_ref = selected["selection_ref"]
            project.set_behavior(
                original_samples=[10.0],
                candidate_samples=[11.0],
            )
            audited = project.run_tool(
                "workload_evaluate.py",
                "final_audit",
                project.final_audit_input(),
                wait=True,
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            audit_result = decode_stdout(audited)
            self.assertEqual(audit_result["verdict"], "rejected")
            self.assertTrue(audit_result["restore_supported"])

            completed = project.run_tool(
                "champion.py",
                "restore-original",
                self._champion_request(
                    project,
                    "restore-original",
                    audit_result["result_ref"],
                    selection_ref,
                ),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(decode_stdout(completed)["status"], "restored_original")
            shown = decode_stdout(
                project.run_tool(
                    "champion.py",
                    "show",
                    {
                        "format_version": "cuda-kernel-optimizer/champion-input-v1",
                        "operation": "show",
                        "artifact_root": str(project.artifact_root),
                        "target_ref": project.target_ref(),
                    },
                )
            )
            self.assertEqual(shown["variant"], original)

    def test_restore_original_rejects_audit_bound_to_a_different_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            original = json.loads(
                (project.artifact_root / "target.json").read_text("utf-8")
            )["original"]
            target_ref = self._result_ref(
                project,
                "inv-target",
                self._target_result(project, reference=original, selection_ref=None),
            )
            selected = decode_stdout(
                project.run_tool(
                    "champion.py",
                    "select",
                    self._champion_request(project, "select", target_ref, None),
                )
            )
            audit_ref = self._result_ref(
                project,
                "inv-audit",
                {
                    "record_type": "invocation_result",
                    "format_version": "cuda-kernel-optimizer/evaluator-result-v1",
                    "operation": "final_audit",
                    "target_ref": project.target_ref(),
                    "variant_refs": [original, self._candidate()],
                    "reference_selection_ref": {"id": "sel-other", "sha256": "d" * 64},
                    "cleanup_status": "confirmed",
                    "execution_status": "succeeded",
                    "measurement_validity": "valid",
                    "verdict": "rejected",
                    "reference_status": "current",
                    "performance_receipt": {
                        "status": "not_run",
                        "reference_status": "current",
                    },
                    "restore_supported": True,
                },
            )

            completed = project.run_tool(
                "champion.py",
                "restore-original",
                self._champion_request(
                    project,
                    "restore-original",
                    audit_ref,
                    selected["selection_ref"],
                ),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertEqual(decode_stderr(completed)["error_code"], "result_invalid")

    def test_champion_correctness_failure_can_restore_without_performance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = V14Project(Path(directory))
            project.check()
            target_result = self._formal_target(project)
            selected = decode_stdout(
                project.run_tool(
                    "champion.py",
                    "select",
                    self._champion_request(
                        project,
                        "select",
                        target_result["result_ref"],
                        None,
                    ),
                )
            )
            project.set_behavior(
                correctness_by_role={
                    "original": "passed",
                    "candidate": "failed",
                }
            )
            audited = project.run_tool(
                "workload_evaluate.py",
                "final_audit",
                project.final_audit_input(),
                wait=True,
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            result = decode_stdout(audited)
            self.assertEqual(result["verdict"], "rejected")
            self.assertEqual(result["reference_status"], "current")
            self.assertTrue(result["restore_supported"])
            self.assertEqual(result["performance_receipt"]["status"], "not_run")

            restored = project.run_tool(
                "champion.py",
                "restore-original",
                self._champion_request(
                    project,
                    "restore-original",
                    result["result_ref"],
                    selected["selection_ref"],
                ),
            )
            self.assertEqual(restored.returncode, 0, restored.stderr)


if __name__ == "__main__":
    unittest.main()
