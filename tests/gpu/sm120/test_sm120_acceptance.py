from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.v14_support import V14Project, decode_stdout, write_json


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUNNER = Path(__file__).resolve().parent / "remote" / "run_lane.sh"
NCU_AUTHORIZED_RUNNER = (
    Path(__file__).resolve().parent / "remote" / "run_ncu_authorized_smoke.sh"
)
ARTIFACTS = Path(os.environ.get("CUDA_E2E_ARTIFACTS", "/tmp/cuda-sm120-acceptance"))
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
PHYSICAL = os.environ.get("CUDA_SM120_E2E") == "1"


class Sm120AcceptanceHelperTests(unittest.TestCase):
    def test_runner_contract_is_fail_closed_and_uses_immutable_image(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")

        self.assertNotIn("|| true", source)
        self.assertGreaterEqual(source.count("assert_gpu_idle"), 3)
        self.assertIn("resolved_image_id", source)
        self.assertIn("requested_ref", source)
        self.assertIn('"$resolved_image_id"', source)
        self.assertIn("must be empty", source)
        self.assertNotIn("CUTLASS_PATH", source)
        self.assertIn(
            "retained replay root must not overlap writable artifacts", source
        )

    def test_authorized_ncu_runner_is_explicit_ephemeral_and_does_not_change_host_policy(self) -> None:
        source = NCU_AUTHORIZED_RUNNER.read_text(encoding="utf-8")

        self.assertIn("CUDA_E2E_ALLOW_SYS_ADMIN", source)
        self.assertIn("--cap-drop ALL", source)
        self.assertIn("--cap-add SYS_ADMIN", source)
        self.assertIn("--rm", source)
        self.assertIn("--network none", source)
        self.assertIn("resolved_image_id", source)
        self.assertGreaterEqual(source.count("assert_gpu_idle"), 3)
        self.assertNotIn("--privileged", source)
        self.assertNotIn("sysctl", source)
        self.assertNotIn("RmProfilingAdminOnly", source)

    def test_runner_rejects_a_writable_artifact_mount_inside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = root / "artifacts" / "repo"
            runner = repository / "tests" / "gpu" / "sm120" / "remote" / "run_lane.sh"
            runner.parent.mkdir(parents=True)
            shutil.copy2(RUNNER, runner)
            handoff = root / "handoff"
            handoff.mkdir()
            (handoff / "blind-run-summary.json").write_text("{}\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_E2E_ROOT": str(root),
                    "CUDA_E2E_ARTIFACTS": str(repository / "writable"),
                    "CUDA_V14_HANDOFF_ROOT": str(handoff),
                }
            )

            completed = subprocess.run(
                [str(runner), "compat"],
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("must not overlap", completed.stderr)


class Sm120V14Project(V14Project):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        marker = '{"value":1}\n'
        (self.original / "implementation.json").write_text(marker, encoding="utf-8")
        (self.candidate / "implementation.json").write_text(marker, encoding="utf-8")
        shutil.copy2(FIXTURES / "triton_vector_slow.py", self.original / "kernel.py")
        shutil.copy2(FIXTURES / "triton_vector.py", self.candidate / "kernel.py")
        shutil.copy2(FIXTURES / "v14_workload_driver.py", self.driver)
        write_json(
            self.test_suite,
            {"cases": [{"id": "main", "size": 1_048_576, "seed": 20260801}]},
        )
        write_json(
            self.correctness_reference,
            {"expression": "x*x+1", "atol": 1e-5},
        )
        self.gpu_uuid = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).splitlines()[0].strip()

    def _resources(self) -> dict:
        return {"host_id": "sm120-physical", "gpu_uuids": [self.gpu_uuid]}

    @staticmethod
    def _runtime_limits(value: dict) -> dict:
        value["operation_timeout_seconds"] = 240.0
        value["command_timeout_seconds"] = 180.0
        value["resource_wait_timeout_seconds"] = 30.0
        value["cleanup_timeout_seconds"] = 15.0
        value["launch_deadline"] = time.time() + 60.0
        return value

    def readiness_input(self) -> dict:
        value = super().readiness_input()
        value["environment_requirements"]["gpu_uuids"] = [self.gpu_uuid]
        value["smoke"]["resources"] = self._resources()
        value["smoke"]["runtime_limits"] = {
            "operation_timeout_seconds": 240.0,
            "command_timeout_seconds": 180.0,
            "resource_wait_timeout_seconds": 30.0,
            "cleanup_timeout_seconds": 15.0,
        }
        return value

    def baseline_input(self) -> dict:
        value = super().baseline_input()
        value["resources"] = self._resources()
        return self._runtime_limits(value)

    def experiment_input(self, baseline_ref: dict) -> dict:
        value = super().experiment_input(baseline_ref)
        value["change_scope"] = ["kernel.py"]
        for stage in value["estimated_cost"].values():
            stage["gpu_count"] = 1
            stage["basis"] = "real SM120 Triton command driver"
        return value

    def screen_input(self, experiment_ref: dict) -> dict:
        value = super().screen_input(experiment_ref)
        value["resources"] = self._resources()
        return self._runtime_limits(value)

    def target_input(self, experiment_ref: dict) -> dict:
        value = super().target_input(experiment_ref)
        value["resources"] = self._resources()
        return self._runtime_limits(value)

    def final_audit_input(self) -> dict:
        value = super().final_audit_input()
        value["resources"] = self._resources()
        return self._runtime_limits(value)

    def run_tool(
        self,
        filename: str,
        operation: str,
        request: dict,
        *,
        wait: bool = False,
    ) -> subprocess.CompletedProcess:
        request_path = self.root / f"{filename}-{operation}-input.json"
        write_json(request_path, request)
        command = [
            sys.executable,
            str(SCRIPTS / filename),
            operation,
            "--request",
            str(request_path),
        ]
        if wait:
            command.append("--wait")
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )


@unittest.skipUnless(PHYSICAL, "set CUDA_SM120_E2E=1 for physical SM120 tests")
class Sm120V14PhysicalTests(unittest.TestCase):
    def test_real_triton_candidate_runs_the_complete_v14_record_path(self) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        case_root = ARTIFACTS / "v14-real-triton"
        if case_root.exists():
            self.fail(f"physical acceptance requires a fresh case root: {case_root}")
        project = Sm120V14Project(case_root)
        checked = project.check()
        self.assertEqual(checked["status"], "ready")
        baseline = project.baseline()
        self.assertEqual(baseline["measurement_validity"], "valid")

        created = project.run_tool(
            "workload_evaluate.py",
            "experiment",
            project.experiment_input(baseline["result_ref"]),
        )
        self.assertEqual(created.returncode, 0, created.stderr)
        experiment_ref = decode_stdout(created)["experiment_ref"]

        screened = project.run_tool(
            "workload_evaluate.py",
            "screen",
            project.screen_input(experiment_ref),
            wait=True,
        )
        self.assertEqual(screened.returncode, 0, screened.stderr)
        screen_result = decode_stdout(screened)
        self.assertEqual(screen_result["measurement_validity"], "valid")
        self.assertEqual(screen_result["verdict"], "inconclusive")
        screen_receipt = screen_result["performance_receipt"]
        self.assertGreater(
            screen_receipt["statistics"]["estimate_pct"],
            screen_receipt["acceptance"]["value"],
        )

        # One diagnostic pair cannot confirm a win. This explicit test path
        # chooses a formal comparison because the valid signal is large; the
        # screen tool itself does not promote or start target measurement.
        compared = project.run_tool(
            "workload_evaluate.py",
            "target",
            project.target_input(experiment_ref),
            wait=True,
        )
        self.assertEqual(compared.returncode, 0, compared.stderr)
        target_result = decode_stdout(compared)
        self.assertEqual(target_result["measurement_validity"], "valid")
        self.assertEqual(target_result["verdict"], "passed")

        selected = project.run_tool(
            "champion.py",
            "select",
            {
                "format_version": "cuda-kernel-optimizer/champion-input-v1",
                "operation": "select",
                "artifact_root": str(project.artifact_root),
                "target_ref": project.target_ref(),
                "result_ref": target_result["result_ref"],
                "expected_selection_ref": None,
            },
        )
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(decode_stdout(selected)["status"], "selected")

        audited = project.run_tool(
            "workload_evaluate.py",
            "final_audit",
            project.final_audit_input(),
            wait=True,
        )
        self.assertEqual(audited.returncode, 0, audited.stderr)
        audit_result = decode_stdout(audited)
        self.assertEqual(audit_result["measurement_validity"], "valid")
        self.assertEqual(audit_result["verdict"], "passed")

        operations = {
            json.loads(path.read_text(encoding="utf-8"))["operation"]
            for path in (project.artifact_root / "invocations").glob("*/request.json")
        }
        self.assertEqual(operations, {"baseline", "screen", "target", "final_audit"})

    def test_retained_iter0_replay_recomputes_both_rejections_without_writes(self) -> None:
        replay = Path(os.environ["CUDA_V14_HANDOFF_ROOT"])
        files = [
            replay / "blind-run-summary.json",
            replay / "candidates" / "candidate-01-deferred-output" / "result.json",
            replay / "candidates" / "candidate-02-native-warp-min" / "result.json",
        ]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        summary, first, second = [
            json.loads(path.read_text(encoding="utf-8")) for path in files
        ]

        spec = importlib.util.spec_from_file_location(
            "sm120_v14_paired_stats", SCRIPTS / "paired_stats.py"
        )
        stats = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(stats)

        first_pair = first["evidence"]["short_paired_complete_service"]
        first_result = stats.classify_pairs(
            [{
                "baseline": first_pair["baseline_endpoint_qps"],
                "candidate": first_pair["candidate_endpoint_qps"],
            }],
            direction="higher",
            min_effect_pct=first_pair["minimum_effect_percent"],
            min_valid_pairs=2,
            bootstrap_samples=200,
            seed=1,
        )
        second_pairs = second["evidence"]["formal_paired_complete_service"]
        second_result = stats.classify_pairs(
            [
                {
                    "baseline": pair["baseline_endpoint_qps"],
                    "candidate": pair["candidate_endpoint_qps"],
                }
                for pair in second_pairs
            ],
            direction="higher",
            min_effect_pct=second["minimum_effect_percent"],
            min_valid_pairs=2,
            bootstrap_samples=1000,
            seed=2,
        )

        self.assertEqual(first_result["status"], "inconclusive")
        self.assertLess(first_result["estimate_pct"], first_pair["minimum_effect_percent"])
        self.assertEqual(second_result["status"], "inconclusive")
        self.assertAlmostEqual(
            second_result["estimate_pct"],
            second["formal_observed_median_gain_percent"],
            places=6,
        )
        self.assertEqual(summary["terminal_decision"], "STOP")
        self.assertEqual([first["outcome"], second["outcome"]], ["REJECT", "REJECT"])
        self.assertEqual(
            {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            before,
        )


if __name__ == "__main__":
    unittest.main()
