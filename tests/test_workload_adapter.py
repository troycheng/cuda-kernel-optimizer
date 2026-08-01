from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.v14_support import V14Project, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
TEMPLATE_DIR = ROOT / "skills" / "cuda-kernel-optimizer" / "templates"


def _load_adapter():
    path = SCRIPT_DIR / "workload_adapter.py"
    spec = importlib.util.spec_from_file_location("workload_adapter_template_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _driver(adapter) -> dict:
    return adapter.validate_driver(
        {
            "command": [sys.executable],
            "request_argument": "--request",
            "execution_mode": "combined",
            "protocol_version": adapter.DRIVER_PROTOCOL,
            "profiler_capabilities": ["pytorch_chrome_trace_v1"],
            "side_effects": [],
            "cleanup_contract": {"kind": "process_group_only", "external_tasks": False},
        }
    )


def _request(adapter, output_path: Path) -> dict:
    return adapter.build_driver_request(
        target_id="target-1",
        execution_id="execution-1",
        operation="baseline",
        driver=_driver(adapter),
        variant={"kind": "source_snapshot", "digest": "a" * 64, "locator": "original"},
        test_suite={"digest": "b" * 64, "locator": "tests", "case_ids": ["case-1"]},
        correctness={
            "reference": {"digest": "c" * 64, "locator": "reference"},
            "method": "relative_error",
            "acceptance": {"metric": "max_error", "operator": "less_or_equal", "value": 0.01},
        },
        objective={"primary_metric": {"name": "latency", "unit": "ms"}, "constraints": []},
        role="original",
        mode="combined",
        case={"id": "case-1"},
        sampling={"repetitions": 3},
        output_path=output_path,
    )


def _result(adapter, request: dict) -> dict:
    return {
        "protocol_version": adapter.RESULT_PROTOCOL,
        "request_digest": request["request_digest"],
        "target_id": request["target_id"],
        "execution_id": request["execution_id"],
        "variant_digest": request["variant"]["digest"],
        "role": request["role"],
        "mode": request["mode"],
        "case_id": request["case"]["id"],
        "artifacts": [],
        "cleanup": {"status": "confirmed", "live_tasks": []},
        "driver_identity": request["driver_identity"],
        "environment": {
            "gpu_uuids": ["GPU-1"],
            "gpu_models": ["Test GPU"],
            "gpu_architectures": ["sm_test"],
            "driver_version": "1",
            "cuda_runtime_version": "1",
            "frameworks": {"torch": "1"},
            "container": {"kind": "none", "identity": "host"},
        },
        "correctness": {"status": "passed", "metrics": {"max_error": 0.0}},
        "measurements": {
            "primary": {"name": "latency", "unit": "ms", "samples": [1.0, 1.1]},
            "constraints": [],
        },
    }


class WorkloadDriverProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _load_adapter()

    def test_driver_identity_is_frozen_into_request_and_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            driver = _driver(self.adapter)
            request = _request(self.adapter, output)

            argv = self.adapter.build_argv(driver, Path(temporary) / "request.json")

        self.assertEqual(request["protocol_version"], self.adapter.REQUEST_PROTOCOL)
        self.assertEqual(request["driver_identity"], driver["identity"])
        self.assertEqual(argv[:-2], driver["command"])
        self.assertEqual(argv[-2], driver["request_argument"])
        self.assertTrue(Path(argv[-1]).is_absolute())
        self.assertEqual(
            driver["profiler_capabilities"],
            ["pytorch_chrome_trace_v1"],
        )

    def test_driver_capabilities_and_artifacts_are_closed_and_content_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "profiler_capabilities"):
            self.adapter.validate_driver(
                {
                    "command": [sys.executable],
                    "request_argument": "--request",
                    "execution_mode": "combined",
                    "protocol_version": self.adapter.DRIVER_PROTOCOL,
                    "profiler_capabilities": ["unknown_profiler"],
                    "side_effects": [],
                    "cleanup_contract": {
                        "kind": "process_group_only",
                        "external_tasks": False,
                    },
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            trace = root / "trace.json"
            trace.write_bytes(b'{"traceEvents":[]}')
            request = _request(self.adapter, output)
            result = _result(self.adapter, request)
            result["artifacts"] = [
                {
                    "kind": "pytorch_chrome_trace",
                    "relative_path": "trace.json",
                    "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                }
            ]
            output.write_text(json.dumps(result), encoding="utf-8")
            normalized = self.adapter.validate_driver_result(output, request)
            self.assertEqual(normalized["artifacts"], result["artifacts"])

            escaped = copy.deepcopy(result)
            escaped["artifacts"][0]["relative_path"] = "../trace.json"
            output.write_text(json.dumps(escaped), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative_path"):
                self.adapter.validate_driver_result(output, request)

            changed = copy.deepcopy(result)
            changed["artifacts"][0]["sha256"] = "0" * 64
            output.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                self.adapter.validate_driver_result(output, request)

    def test_result_rejects_nonfinite_measurements_and_invented_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            request = _request(self.adapter, path)
            result = _result(self.adapter, request)
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(
                self.adapter.validate_driver_result(path, request)["environment"],
                result["environment"],
            )

            nonfinite = copy.deepcopy(result)
            nonfinite["measurements"]["primary"]["samples"] = [math.nan]
            path.write_text(json.dumps(nonfinite), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.adapter.validate_driver_result(path, request)

            invented = copy.deepcopy(result)
            del invented["environment"]["driver_version"]
            path.write_text(json.dumps(invented), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "environment"):
                self.adapter.validate_driver_result(path, request)

            unbounded = copy.deepcopy(result)
            unbounded["measurements"]["primary"]["samples"] = [
                1.0
            ] * (self.adapter._MAX_SAMPLES + 1)
            path.write_text(json.dumps(unbounded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample limit"):
                self.adapter.validate_driver_result(path, request)

    def test_three_driver_templates_match_the_closed_adapter_protocol(self) -> None:
        expected = {
            "workload_driver.py",
            "workload_driver_request.schema.json",
            "workload_driver_result.schema.json",
        }
        self.assertEqual({path.name for path in TEMPLATE_DIR.iterdir()}, expected)

        request_schema = json.loads(
            (TEMPLATE_DIR / "workload_driver_request.schema.json").read_text("utf-8")
        )
        result_schema = json.loads(
            (TEMPLATE_DIR / "workload_driver_result.schema.json").read_text("utf-8")
        )
        self.assertFalse(request_schema["additionalProperties"])
        self.assertEqual(set(request_schema["required"]), self.adapter._REQUEST_FIELDS)
        self.assertEqual(set(request_schema["properties"]), self.adapter._REQUEST_FIELDS)
        self.assertFalse(result_schema["additionalProperties"])
        self.assertEqual(set(result_schema["required"]), self.adapter._RESULT_BASE_FIELDS)
        self.assertEqual(
            set(result_schema["properties"]),
            self.adapter._RESULT_BASE_FIELDS | {"correctness", "measurements"},
        )

        template = (TEMPLATE_DIR / "workload_driver.py").read_text("utf-8")
        for value in (
            self.adapter.DRIVER_PROTOCOL,
            self.adapter.REQUEST_PROTOCOL,
            self.adapter.RESULT_PROTOCOL,
            "--request",
            "run_correctness",
            "run_measurements",
            "collect_environment",
            "os.link",
        ):
            self.assertIn(value, template)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            output_path = root / "result.json"
            request_path.write_text(
                json.dumps(_request(self.adapter, output_path)), encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, str(TEMPLATE_DIR / "workload_driver.py"), "--request", str(request_path)],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("TODO: implement run_correctness", completed.stdout)
            self.assertFalse(output_path.exists())


class ProfileCollectionBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _load_adapter()

    def _ready_project(self, root: Path) -> tuple[V14Project, dict]:
        project = V14Project(root)
        request = project.readiness_input()
        request["driver"]["profiler_capabilities"] = [
            "pytorch_chrome_trace_v1"
        ]
        checked = project.run_tool("readiness.py", "check", request)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        return project, project.baseline()

    def test_original_collection_resolves_and_candidate_needs_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, baseline = self._ready_project(Path(temporary))
            events_before = project.driver_events()

            resolved = self.adapter.resolve_profile_collection(
                artifact_root=project.artifact_root,
                target_ref=project.target_ref(),
                baseline_ref=baseline["result_ref"],
                role="original",
                case_id="main",
                capability="pytorch_chrome_trace_v1",
            )

            self.assertEqual(resolved["role"], "original")
            self.assertEqual(resolved["variant"], resolved["target"]["original"])
            with self.assertRaisesRegex(ValueError, "correctness_ref"):
                self.adapter.resolve_profile_collection(
                    artifact_root=project.artifact_root,
                    target_ref=project.target_ref(),
                    baseline_ref=baseline["result_ref"],
                    role="candidate",
                    case_id="main",
                    capability="pytorch_chrome_trace_v1",
                    experiment_ref={"id": "exp-missing", "sha256": "0" * 64},
                )
            self.assertEqual(project.driver_events(), events_before)

    def test_candidate_collection_rejects_mismatched_or_unknown_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project, baseline = self._ready_project(Path(temporary))
            created = project.run_tool(
                "workload_evaluate.py",
                "experiment",
                project.experiment_input(baseline["result_ref"]),
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            experiment_ref = json.loads(created.stdout)["experiment_ref"]
            screened = project.run_tool(
                "workload_evaluate.py",
                "screen",
                project.screen_input(experiment_ref),
                wait=True,
            )
            self.assertEqual(screened.returncode, 0, screened.stderr)
            screen = json.loads(screened.stdout)
            valid_correctness_ref = {
                **screen["result_ref"],
                "case_id": "main",
            }
            resolved = self.adapter.resolve_profile_collection(
                artifact_root=project.artifact_root,
                target_ref=project.target_ref(),
                baseline_ref=baseline["result_ref"],
                role="candidate",
                case_id="main",
                capability="pytorch_chrome_trace_v1",
                experiment_ref=experiment_ref,
                correctness_ref=valid_correctness_ref,
            )
            self.assertEqual(resolved["variant"]["role"], "candidate")
            result_path = (
                project.artifact_root
                / "invocations"
                / screen["invocation_id"]
                / "result.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["correctness_receipts"][0]["variant"] = json.loads(
                (project.artifact_root / "target.json").read_text(encoding="utf-8")
            )["original"]
            write_json(result_path, result)
            mismatched_ref = {
                "invocation_id": screen["invocation_id"],
                "sha256": sha256_file(result_path),
                "case_id": "main",
            }
            events_before = project.driver_events()
            with self.assertRaisesRegex(ValueError, "one passing candidate receipt"):
                self.adapter.resolve_profile_collection(
                    artifact_root=project.artifact_root,
                    target_ref=project.target_ref(),
                    baseline_ref=baseline["result_ref"],
                    role="candidate",
                    case_id="main",
                    capability="pytorch_chrome_trace_v1",
                    experiment_ref=experiment_ref,
                    correctness_ref=mismatched_ref,
                )
            self.assertEqual(project.driver_events(), events_before)

            result["correctness_receipts"][0]["variant"] = json.loads(
                (project.artifact_root / "experiments" / f"{experiment_ref['id']}.json").read_text(
                    encoding="utf-8"
                )
            )["candidate"]
            result["correctness_receipts"][0]["unexpected"] = True
            write_json(result_path, result)
            correctness_ref = {
                "invocation_id": screen["invocation_id"],
                "sha256": sha256_file(result_path),
                "case_id": "main",
            }

            with self.assertRaisesRegex(ValueError, "correctness receipt.*unknown"):
                self.adapter.resolve_profile_collection(
                    artifact_root=project.artifact_root,
                    target_ref=project.target_ref(),
                    baseline_ref=baseline["result_ref"],
                    role="candidate",
                    case_id="main",
                    capability="pytorch_chrome_trace_v1",
                    experiment_ref=experiment_ref,
                    correctness_ref=correctness_ref,
                )

    def test_template_never_publishes_when_cleanup_is_not_implemented(self) -> None:
        template = (TEMPLATE_DIR / "workload_driver.py").read_text("utf-8")
        implemented = template.replace(
            '    raise NotImplementedError("TODO: implement run_correctness for this workload")',
            '    return {"status": "passed", "metrics": {"max_error": 0.0}}',
        ).replace(
            '    raise NotImplementedError("TODO: implement run_measurements for this workload")',
            '    return {"primary": {"name": "latency", "unit": "ms", "samples": [1.0]}, "constraints": []}',
        ).replace(
            '    raise NotImplementedError("TODO: implement collect_environment for this workload")',
            '    return {"gpu_uuids": ["GPU-1"], "gpu_models": ["Test GPU"], "gpu_architectures": ["sm_test"], "driver_version": "1", "cuda_runtime_version": "1", "frameworks": {"torch": "1"}, "container": {"kind": "none", "identity": "host"}}',
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = root / "workload_driver.py"
            request_path = root / "request.json"
            output_path = root / "result.json"
            driver.write_text(implemented, encoding="utf-8")
            request_path.write_text(
                json.dumps(_request(self.adapter, output_path)), encoding="utf-8"
            )

            completed = subprocess.run(
                [sys.executable, str(driver), "--request", str(request_path)],
                capture_output=True,
                check=False,
                text=True,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("TODO: implement cleanup", completed.stdout)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
