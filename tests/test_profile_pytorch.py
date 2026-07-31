from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
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


class ProfilePyTorchTests(unittest.TestCase):
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
