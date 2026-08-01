from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cuda_optimizer_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _trace(events=None, *, schema_version=1) -> dict:
    return {
        "schemaVersion": schema_version,
        "deviceProperties": [
            {
                "id": 0,
                "name": "NVIDIA test GPU",
                "computeMajor": 12,
                "computeMinor": 0,
            }
        ],
        "displayTimeUnit": "ms",
        "traceEvents": events if events is not None else [
            {
                "name": "aten::add",
                "cat": "cpu_op",
                "ph": "X",
                "pid": 1,
                "tid": 7,
                "ts": 120.5,
                "dur": 3.25,
                "args": {"External id": 4},
            },
            {
                "name": "PyTorch Profiler (0)",
                "cat": "Trace",
                "ph": "X",
                "pid": "Spans",
                "tid": "PyTorch Profiler",
                "ts": 100.0,
                "dur": 30.0,
            },
            {"name": "process_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "python"}},
        ],
    }


def _canonical_id(material: dict) -> str:
    identity = {key: material[key] for key in ("kind", "tool", "tool_version", "dialect", "object_ref")}
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _collection_fixture(temporary: str, *, ambiguous_trace: bool = False) -> tuple[dict, dict, Path]:
    """Create one frozen target whose driver owns the Chrome trace export."""
    store = _load("artifact_store")
    adapter = _load("workload_adapter")
    root = Path(temporary) / "artifacts"
    (root / "invocations" / "inv-baseline").mkdir(parents=True)
    (root / ".locks").mkdir()
    project = Path(temporary) / "project"
    project.mkdir()
    original = project / "original.json"
    test_suite = project / "test-suite.json"
    correctness = project / "correctness.json"
    _write_json(original, {"implementation": "original"})
    _write_json(test_suite, {"cases": [{"id": "main"}]})
    _write_json(correctness, {"expected": "fixture"})
    limits = {"max_files": 8, "max_total_bytes": 1024 * 1024, "max_wall_seconds": 2.0}
    original_object = store.freeze_path(root, original, limits)
    test_object = store.freeze_path(root, test_suite, limits)
    correctness_object = store.freeze_path(root, correctness, limits)
    events = project / "driver-events.jsonl"
    driver = project / "driver.py"
    trace_payload = json.dumps(_trace(), sort_keys=True)
    driver.write_text(
        "\n".join(
            [
                "import argparse",
                "import hashlib",
                "import json",
                "import sys",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--request', required=True)",
                "request = json.loads(Path(parser.parse_args().request).read_text('utf-8'))",
                f"events = Path({str(events)!r})",
                "with events.open('a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')",
                "assert request['operation'] == 'profile_pytorch_collect'",
                "assert request['sampling'] == {'kind': 'pytorch_chrome_trace_v1'}",
                "assert Path(request['variant']['locator']).exists()",
                "assert Path(request['test_suite']['locator']).exists()",
                "assert Path(request['correctness']['reference']['locator']).exists()",
                "output = Path(request['output_path'])",
                "trace = output.parent / 'trace.json'",
                f"trace.write_text({trace_payload!r}, encoding='utf-8')",
                "artifact = {'kind': 'pytorch_chrome_trace', 'relative_path': 'trace.json', 'sha256': hashlib.sha256(trace.read_bytes()).hexdigest()}",
                f"artifacts = [artifact, {{**artifact, 'relative_path': 'trace-duplicate.json'}}] if {ambiguous_trace!r} else [artifact]",
                f"if {ambiguous_trace!r}:",
                "    duplicate = output.parent / 'trace-duplicate.json'",
                "    duplicate.write_bytes(trace.read_bytes())",
                "    artifacts[1]['sha256'] = hashlib.sha256(duplicate.read_bytes()).hexdigest()",
                "result = {",
                "  'protocol_version': 'cuda-kernel-optimizer/driver-result-v1',",
                "  'request_digest': request['request_digest'],",
                "  'target_id': request['target_id'],",
                "  'execution_id': request['execution_id'],",
                "  'variant_digest': request['variant']['digest'],",
                "  'role': request['role'],",
                "  'mode': request['mode'],",
                "  'case_id': request['case']['id'],",
                "  'artifacts': artifacts,",
                "  'cleanup': {'status': 'confirmed', 'live_tasks': []},",
                "  'driver_identity': request['driver_identity'],",
                "  'environment': {",
                "    'gpu_uuids': ['GPU-0'], 'gpu_models': ['Fixture GPU'],",
                "    'gpu_architectures': ['sm_fixture'], 'driver_version': 'fixture',",
                "    'cuda_runtime_version': 'fixture', 'frameworks': {'torch': '2.13.0+cu130'},",
                "    'container': {'kind': 'none', 'identity': 'fixture'},",
                "  },",
                "  'measurements': {'primary': {'name': 'latency_ms', 'unit': 'ms', 'samples': [1.0]}, 'constraints': []},",
                "}",
                "output.write_text(json.dumps(result, sort_keys=True), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    frozen_driver = adapter.validate_driver(
        {
            "command": [sys.executable, str(driver)],
            "request_argument": "--request",
            "execution_mode": "separate",
            "protocol_version": adapter.DRIVER_PROTOCOL,
            "profiler_capabilities": ["pytorch_chrome_trace_v1"],
            "side_effects": [],
            "cleanup_contract": {"kind": "process_group_only", "external_tasks": False},
        }
    )
    original_variant = {"role": "original", "kind": "source_snapshot", "digest": original_object["digest"], "locator": original_object["locator"]}
    runtime_environment = {
        "gpu_uuids": ["GPU-0"], "gpu_models": ["Fixture GPU"],
        "gpu_architectures": ["sm_fixture"], "driver_version": "fixture",
        "cuda_runtime_version": "fixture", "frameworks": {"torch": "2.13.0+cu130"},
        "container": {"kind": "none", "identity": "fixture"},
    }
    target = {
        "record_type": "target", "format_version": "cuda-kernel-optimizer/target-v1",
        "id": "target-collect", "target_mode": "optimization", "original": original_variant,
        "driver": frozen_driver,
        "test_suite": {"object_ref": test_object, "case_ids": ["main"]},
        "correctness": {"reference": correctness_object, "method": "driver", "acceptance": {"metric": "exact_match", "operator": "greater_or_equal", "value": 1.0}},
        "objective": {"primary_metric": {"name": "latency_ms", "unit": "ms"}, "constraints": []},
        "environment": {"host": {"host_id": "fixture-host", "gpu_uuids": ["GPU-0"]}, "runtime": runtime_environment},
    }
    target_path = root / "target.json"
    store.create_regular_json(target_path, target)
    target_ref = {"id": target["id"], "sha256": store.sha256_file(target_path)}
    baseline = {
        "operation": "baseline", "target_ref": target_ref, "execution_status": "succeeded",
        "measurement_validity": "valid", "verdict": "passed", "cleanup_status": "confirmed",
        "variant_refs": [original_variant],
    }
    baseline_path = root / "invocations" / "inv-baseline" / "result.json"
    _write_json(baseline_path, baseline)
    request = {
        "format_version": "cuda-kernel-optimizer/pytorch-input-v1", "operation": "collect",
        "artifact_root": str(root), "target_ref": target_ref,
        "baseline_ref": {"invocation_id": "inv-baseline", "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()},
        "role": "original", "case_id": "main",
        "resources": {"host_id": "fixture-host", "gpu_uuids": ["GPU-0"]},
        "operation_timeout_seconds": 5.0, "command_timeout_seconds": 1.0,
        "resource_wait_timeout_seconds": 1.0, "cleanup_timeout_seconds": 1.0,
        "launch_deadline": time.time() + 3.0,
    }
    return request, {"events": events, "root": root, "driver": frozen_driver}, root


class ProfilePyTorchTests(unittest.TestCase):
    def test_public_collect_runs_exact_driver_argv_and_parses_one_frozen_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, fixture, root = _collection_fixture(temporary)
            request_path = Path(temporary) / "collect.json"
            _write_json(request_path, request)

            completed = subprocess.run(
                [sys.executable, str(SCRIPTS / "profile_pytorch.py"), "collect", "--request", str(request_path), "--wait"],
                check=False, capture_output=True, text=True, timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["operation"], "collect")
            self.assertEqual(result["execution_status"], "succeeded")
            self.assertEqual(result["measurement_validity"], "valid")
            self.assertEqual(result["provenance"]["tool"], {"name": "pytorch_profiler", "version": "2.13.0+cu130"})
            self.assertEqual([item["semantic_id"] for item in result["observations"]], ["pytorch.trace.complete_event"])
            self.assertIn("driver_output", result["provenance"])
            self.assertEqual(result["provenance"]["trace_artifact"]["kind"], "pytorch_chrome_trace")
            self.assertTrue((root / result["provenance"]["driver_output"]["locator"]).is_dir())
            self.assertEqual(len(result["provenance"]["command_receipts"]), 1)
            self.assertEqual(
                result["provenance"]["command_receipts"][0]["argv"][:-2],
                fixture["driver"]["command"],
            )
            self.assertEqual(
                result["provenance"]["command_receipts"][0]["argv"][-2],
                fixture["driver"]["request_argument"],
            )
            self.assertTrue(Path(result["provenance"]["command_receipts"][0]["argv"][-1]).is_absolute())
            self.assertEqual(
                [json.loads(line) for line in fixture["events"].read_text(encoding="utf-8").splitlines()],
                [["--request", result["provenance"]["command_receipts"][0]["argv"][-1]]],
            )

    def test_public_collect_rejects_ambiguous_trace_without_second_workload_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, fixture, _root = _collection_fixture(temporary, ambiguous_trace=True)
            request_path = Path(temporary) / "collect.json"
            _write_json(request_path, request)

            completed = subprocess.run(
                [sys.executable, str(SCRIPTS / "profile_pytorch.py"), "collect", "--request", str(request_path), "--wait"],
                check=False, capture_output=True, text=True, timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["execution_status"], "invalid")
            self.assertEqual(result["measurement_validity"], "invalid")
            self.assertEqual(result["stop_reason"], "invalid_trace_artifact")
            self.assertEqual(result["observations"], [])
            self.assertEqual(len(result["provenance"]["command_receipts"]), 1)
            self.assertEqual(len(fixture["events"].read_text(encoding="utf-8").splitlines()), 1)

    def test_chrome_trace_complete_events_emit_interval_duration_category_and_name(self) -> None:
        profile = _load("profile_pytorch")

        facts = profile.parse_chrome_trace(_trace(), "2.13.1")

        observation = facts["observations"][0]
        self.assertEqual(observation["duration_us"], 3.25)
        self.assertEqual(observation["interval"], {"start_us": 120.5, "end_us": 123.75})
        self.assertEqual(observation["category"], "cpu_op")
        self.assertEqual(observation["name"], "aten::add")
        self.assertEqual(
            facts["unmodeled"],
            [
                {
                    "kind": "top_level_metadata",
                    "fields": ["deviceProperties", "displayTimeUnit"],
                },
                {"kind": "event_phase", "phase": "M", "count": 1},
                {
                    "kind": "complete_event_scope",
                    "category": "Trace",
                    "reason": "non_numeric_process_or_thread_id",
                    "count": 1,
                },
            ],
        )

    def test_unknown_schema_missing_complete_field_and_nonfinite_fail_closed(self) -> None:
        profile = _load("profile_pytorch")
        with self.assertRaisesRegex(profile.PyTorchError, "version"):
            profile.parse_chrome_trace(_trace(), "2.14")
        with self.assertRaisesRegex(profile.PyTorchError, "schema"):
            profile.parse_chrome_trace(_trace(schema_version=2), "2.13")
        missing = _trace([{"name": "aten::add", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 1, "ts": 0}])
        with self.assertRaisesRegex(profile.PyTorchError, "missing"):
            profile.parse_chrome_trace(missing, "2.13")
        nonfinite = _trace([{"name": "aten::add", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 1, "ts": 0, "dur": float("inf")}])
        with self.assertRaisesRegex(profile.PyTorchError, "finite"):
            profile.parse_chrome_trace(nonfinite, "2.13")
        duplicate = b'{"schemaVersion":1,"schemaVersion":1,"traceEvents":[]}'
        with self.assertRaisesRegex(profile.PyTorchError, "duplicate"):
            profile._strict_json_bytes(duplicate, "Chrome trace")

    def test_empty_complete_events_and_event_limit_fail_closed(self) -> None:
        profile = _load("profile_pytorch")
        with self.assertRaisesRegex(profile.PyTorchError, "complete"):
            profile.parse_chrome_trace(_trace([{"name": "thread_name", "ph": "M"}]), "2.13")
        original = profile._MAX_EVENTS
        profile._MAX_EVENTS = 1
        try:
            with self.assertRaisesRegex(profile.PyTorchError, "event limit"):
                profile.parse_chrome_trace(_trace(), "2.13")
        finally:
            profile._MAX_EVENTS = original

    def test_public_analyze_status_and_cancel_materialize_frozen_trace(self) -> None:
        profile = _load("profile_pytorch")
        store = _load("artifact_store")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            root.mkdir()
            report = Path(temporary) / "trace.json"
            report.write_text(json.dumps(_trace()), encoding="utf-8")
            object_ref = store.freeze_path(
                root, report, {"max_files": 1, "max_total_bytes": 1024 * 1024, "max_wall_seconds": 1.0}
            )
            material = {
                "sha256": object_ref["digest"],
                "kind": "report",
                "tool": "pytorch_profiler",
                "tool_version": "2.13.1",
                "dialect": "chrome-trace-v1",
                "object_ref": object_ref,
            }
            material["id"] = _canonical_id(material)
            target = {
                "record_type": "target",
                "format_version": "cuda-kernel-optimizer/target-v1",
                "id": "diagnostic-target",
                "target_mode": "diagnostic",
                "diagnostic_materials": [material],
            }
            store.create_regular_json(root / "target.json", target)
            request = {
                "format_version": "cuda-kernel-optimizer/pytorch-input-v1",
                "operation": "analyze",
                "artifact_root": str(root),
                "target_ref": {"id": target["id"], "sha256": hashlib.sha256((root / "target.json").read_bytes()).hexdigest()},
                "report_ref": {"id": material["id"], "sha256": material["sha256"]},
                "resources": {"host_id": "test-host", "gpu_uuids": []},
                "operation_timeout_seconds": 5.0,
                "command_timeout_seconds": 1.0,
                "resource_wait_timeout_seconds": 1.0,
                "cleanup_timeout_seconds": 1.0,
                "launch_deadline": time.time() + 3.0,
            }
            gpu_request = {**request, "resources": {"host_id": "test-host", "gpu_uuids": ["GPU-0"]}}
            with self.assertRaisesRegex(profile.PyTorchError, "must not request GPU"):
                profile._validate_analyze(gpu_request)
            request_path = Path(temporary) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            completed = subprocess.run(
                ["python3", str(SCRIPTS / "profile_pytorch.py"), "analyze", "--request", str(request_path), "--wait"],
                check=False, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["execution_status"], "succeeded")
            self.assertEqual(result["measurement_validity"], "valid")
            self.assertEqual(result["provenance"]["tool"], {"name": "pytorch_profiler", "version": "2.13.1"})
            status = profile._status_or_cancel(
                {"format_version": request["format_version"], "operation": "status", "artifact_root": str(root), "invocation_id": result["invocation_id"]}, "status"
            )
            self.assertEqual(status["query_status"], "completed")
            cancelled = profile._status_or_cancel(
                {"format_version": request["format_version"], "operation": "cancel", "artifact_root": str(root), "invocation_id": result["invocation_id"]}, "cancel"
            )
            self.assertEqual(cancelled["query_status"], "completed")


if __name__ == "__main__":
    unittest.main()
