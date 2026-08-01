from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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


def _csv(*, bad_unit: bool = False) -> str:
    rows = [
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "75"),
        ("dram__bytes.sum", "byte", "4,096"),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "%", "61"),
        ("sm__cycles_active.avg.pct_of_peak_sustained_elapsed", "%", "84"),
        (
            "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
            "%",
            "47",
        ),
        (
            "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct",
            "%",
            "9",
        ),
        (
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "cycle" if bad_unit else "%",
            "32",
        ),
        ("unknown__future_metric", "widget", "3"),
    ]
    return '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n' + "".join(
        f'"target_kernel","{name}","{unit}","{value}"\n'
        for name, unit, value in rows
    )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _collection_fixture(temporary: str, *, counter_denied: bool = False) -> tuple[dict, dict, Path]:
    """Create one frozen Target and fake NCU/driver pair for public collection."""
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
    events = project / "ncu-events.jsonl"
    driver = project / "driver.py"
    driver.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--request', required=True)",
                "request = json.loads(Path(parser.parse_args().request).read_text('utf-8'))",
                "assert Path(request['variant']['locator']).exists()",
                "assert Path(request['test_suite']['locator']).exists()",
                "assert Path(request['correctness']['reference']['locator']).exists()",
                "result = {",
                "  'protocol_version': 'cuda-kernel-optimizer/driver-result-v1',",
                "  'request_digest': request['request_digest'],",
                "  'target_id': request['target_id'],",
                "  'execution_id': request['execution_id'],",
                "  'variant_digest': request['variant']['digest'],",
                "  'role': request['role'],",
                "  'mode': request['mode'],",
                "  'case_id': request['case']['id'],",
                "  'artifacts': [],",
                "  'cleanup': {'status': 'confirmed', 'live_tasks': []},",
                "  'driver_identity': request['driver_identity'],",
                "  'environment': {",
                "    'gpu_uuids': ['GPU-0'], 'gpu_models': ['Fixture GPU'],",
                "    'gpu_architectures': ['sm_fixture'], 'driver_version': 'fixture',",
                "    'cuda_runtime_version': 'fixture', 'frameworks': {},",
                "    'container': {'kind': 'none', 'identity': 'fixture'},",
                "  },",
                "  'measurements': {'primary': {'name': 'latency_ms', 'unit': 'ms', 'samples': [1.0]}, 'constraints': []},",
                "}",
                "Path(request['output_path']).write_text(json.dumps(result, sort_keys=True), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ncu = project / "ncu"
    ncu.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, subprocess, sys",
                "from pathlib import Path",
                f"events = Path({str(events)!r})",
                "args = sys.argv[1:]",
                "with events.open('a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps(args) + '\\n')",
                "if args == ['--version']:",
                "    print('NVIDIA Nsight Compute version 2026.2.1')",
                "elif '--import' in args:",
                f"    Path(args[args.index('--log-file') + 1]).write_text({_csv()!r}, encoding='utf-8')",
                "else:",
                f"    if {counter_denied!r}:",
                "        print('ERR_NVGPUCTRPERM', file=sys.stderr)",
                "        raise SystemExit(1)",
                "    Path(args[args.index('--export') + 1]).write_bytes(b'fixture-ncu-report')",
                "    subprocess.run(args[12:], check=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(ncu, 0o700)
    frozen_driver = adapter.validate_driver(
        {
            "command": [sys.executable, str(driver)],
            "request_argument": "--request",
            "execution_mode": "separate",
            "protocol_version": adapter.DRIVER_PROTOCOL,
            "profiler_capabilities": ["ncu_wrap_v1"],
            "side_effects": [],
            "cleanup_contract": {"kind": "process_group_only", "external_tasks": False},
        }
    )
    original_variant = {
        "role": "original",
        "kind": "source_snapshot",
        "digest": original_object["digest"],
        "locator": original_object["locator"],
    }
    runtime_environment = {
        "gpu_uuids": ["GPU-0"], "gpu_models": ["Fixture GPU"],
        "gpu_architectures": ["sm_fixture"], "driver_version": "fixture",
        "cuda_runtime_version": "fixture", "frameworks": {},
        "container": {"kind": "none", "identity": "fixture"},
    }
    target = {
        "record_type": "target", "format_version": "cuda-kernel-optimizer/target-v1",
        "id": "target-collect", "target_mode": "optimization", "original": original_variant,
        "driver": frozen_driver,
        "test_suite": {"object_ref": test_object, "case_ids": ["main"]},
        "correctness": {"reference": correctness_object, "method": "driver", "acceptance": {"metric": "exact_match", "operator": "greater_or_equal", "value": 1.0}},
        "objective": {"primary_metric": {"name": "latency_ms", "unit": "ms"}, "constraints": []},
        "environment": {"host": {"host_id": "fixture-host", "gpu_uuids": ["GPU-0"], "tools": {"ncu": {"path": str(ncu), "sha256": store.sha256_file(ncu)}}}, "runtime": runtime_environment},
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
        "format_version": "cuda-kernel-optimizer/ncu-input-v1", "operation": "collect",
        "artifact_root": str(root), "target_ref": target_ref,
        "baseline_ref": {"invocation_id": "inv-baseline", "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()},
        "role": "original", "case_id": "main", "kernel_name_hints": ["target_kernel"],
        "resources": {"host_id": "fixture-host", "gpu_uuids": ["GPU-0"]},
        "operation_timeout_seconds": 5.0, "command_timeout_seconds": 1.0,
        "resource_wait_timeout_seconds": 1.0, "cleanup_timeout_seconds": 1.0,
        "launch_deadline": time.time() + 3.0,
    }
    return request, {"events": events, "root": root, "target": target, "driver": frozen_driver}, root


class ProfileNcuTests(unittest.TestCase):
    def test_public_collect_wraps_the_only_driver_argv_and_freezes_raw_facts(self) -> None:
        profile_ncu = _load("profile_ncu")
        with tempfile.TemporaryDirectory() as temporary:
            request, fixture, root = _collection_fixture(temporary)
            request_path = Path(temporary) / "collect.json"
            _write_json(request_path, request)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "profile_ncu.py"),
                    "collect",
                    "--request",
                    str(request_path),
                    "--wait",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["operation"], "collect")
            self.assertEqual(result["execution_status"], "succeeded")
            self.assertEqual(result["measurement_validity"], "valid")
            self.assertEqual(
                [item["semantic_id"] for item in result["observations"]],
                [
                    "kernel.barrier_stall_pct",
                    "kernel.dram_bytes",
                    "kernel.dram_throughput_pct",
                    "kernel.long_scoreboard_pct",
                    "kernel.occupancy_pct",
                    "kernel.sm_active_pct",
                    "kernel.tensor_pipe_pct",
                ],
            )
            provenance = result["provenance"]
            self.assertEqual(provenance["tool"]["version"], "2026.2.1")
            self.assertEqual(provenance["metrics"], list(profile_ncu._METRICS))
            self.assertIn("driver_result", provenance)
            self.assertIn("report", provenance)
            self.assertIn("csv", provenance)
            self.assertTrue((root / provenance["report"]["locator"]).is_dir())
            self.assertTrue((root / provenance["csv"]["locator"]).is_dir())
            ncu_events = [
                json.loads(line)
                for line in fixture["events"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(ncu_events[0], ["--version"])
            self.assertEqual(
                ncu_events[1][0:11],
                [
                    "--config-file", "off", "--metrics", ",".join(profile_ncu._METRICS),
                    "--print-units", "base", "--print-metric-name", "name",
                    "--target-processes", "all", "--export",
                ],
            )
            self.assertEqual(ncu_events[1][12:-2], fixture["driver"]["command"])
            self.assertEqual(ncu_events[1][-2], fixture["driver"]["request_argument"])
            self.assertTrue(Path(ncu_events[1][-1]).is_absolute())
            self.assertEqual(ncu_events[2][0:6], ["--config-file", "off", "--import", ncu_events[1][ncu_events[1].index("--export") + 1], "--csv", "--page"])
            self.assertEqual(ncu_events[2][6:8], ["raw", "--log-file"])
            self.assertTrue(Path(ncu_events[2][8]).is_absolute())

    def test_candidate_collect_missing_correctness_rejects_before_invocation_or_ncu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, fixture, root = _collection_fixture(temporary)
            request.update(
                {
                    "role": "candidate",
                    "experiment_ref": {"id": "exp-fixture", "sha256": "a" * 64},
                }
            )
            request_path = Path(temporary) / "candidate-collect.json"
            _write_json(request_path, request)
            before = sorted(path.name for path in (root / "invocations").iterdir())

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "profile_ncu.py"),
                    "collect",
                    "--request",
                    str(request_path),
                    "--wait",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("requires experiment_ref and correctness_ref", completed.stderr)
            self.assertEqual(sorted(path.name for path in (root / "invocations").iterdir()), before)
            self.assertFalse(fixture["events"].exists())

    def test_collect_counter_permission_denied_is_an_invalid_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, _fixture, _root = _collection_fixture(temporary, counter_denied=True)
            request_path = Path(temporary) / "counter-denied.json"
            _write_json(request_path, request)

            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPTS / "profile_ncu.py"), "collect",
                    "--request", str(request_path), "--wait",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["measurement_validity"], "invalid")
            self.assertEqual(result["stop_reason"], "ncu_counter_access_denied")
            self.assertEqual(result["observations"], [])
            self.assertIn("ERR_NVGPUCTRPERM", result["diagnostic"]["error"])
            self.assertEqual(
                [receipt["argv"][1] for receipt in result["provenance"]["command_receipts"]],
                ["--version", "--config-file"],
            )

    def test_a_partial_metric_set_reports_only_the_metrics_that_exist(self) -> None:
        profile_ncu = _load("profile_ncu")
        csv_text = (
            '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"target_kernel","dram__bytes.sum","byte","4096"\n'
        )

        result = profile_ncu.parse_ncu_csv(csv_text, "2026.2.1", [])

        self.assertEqual(
            [item["semantic_id"] for item in result["observations"]],
            ["kernel.dram_bytes"],
        )

    def test_supported_2026_2_long_csv_produces_only_stable_observations(self) -> None:
        profile_ncu = _load("profile_ncu")

        result = profile_ncu.parse_ncu_csv(_csv(), "2026.2.1", ["target_kernel"])

        self.assertEqual(
            [item["semantic_id"] for item in result["observations"]],
            [
                "kernel.barrier_stall_pct",
                "kernel.dram_bytes",
                "kernel.dram_throughput_pct",
                "kernel.long_scoreboard_pct",
                "kernel.occupancy_pct",
                "kernel.sm_active_pct",
                "kernel.tensor_pipe_pct",
            ],
        )
        self.assertEqual(
            result["unmodeled"],
            [{"metric_name": "unknown__future_metric", "reason": "unknown_metric"}],
        )
        self.assertEqual(
            next(item["value"] for item in result["observations"] if item["semantic_id"] == "kernel.dram_bytes"),
            4096.0,
        )

    def test_known_metric_with_wrong_unit_fails_closed(self) -> None:
        profile_ncu = _load("profile_ncu")

        with self.assertRaisesRegex(profile_ncu.NcuError, "unexpected_unit"):
            profile_ncu.parse_ncu_csv(_csv(bad_unit=True), "2026.2", [])

    def test_public_analyze_materializes_the_frozen_report_and_writes_facts(self) -> None:
        profile_ncu = _load("profile_ncu")
        store = _load("artifact_store")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            (root / "invocations").mkdir(parents=True)
            (root / ".locks").mkdir()
            report = Path(temporary) / "report.csv"
            report.write_text(_csv(), encoding="utf-8")
            report_object = store.freeze_path(
                root,
                report,
                {"max_files": 1, "max_total_bytes": 1024 * 1024, "max_wall_seconds": 2.0},
            )
            material = {
                "sha256": report_object["digest"],
                "kind": "report",
                "tool": "ncu",
                "tool_version": "2026.2.1",
                "dialect": "ncu-csv-long-v1",
                "object_ref": report_object,
            }
            material_identity = {
                key: material[key]
                for key in ("kind", "tool", "tool_version", "dialect", "object_ref")
            }
            material["id"] = hashlib.sha256(
                json.dumps(
                    material_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            target = {
                "record_type": "target",
                "format_version": "cuda-kernel-optimizer/target-v1",
                "id": "diagnostic-target",
                "target_mode": "diagnostic",
                "diagnostic_materials": [material],
            }
            target_path = root / "target.json"
            store.create_regular_json(target_path, target)
            request = {
                "format_version": "cuda-kernel-optimizer/ncu-input-v1",
                "operation": "analyze",
                "artifact_root": str(root),
                "target_ref": {
                    "id": target["id"],
                    "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                },
                "report_ref": {"id": material["id"], "sha256": material["sha256"]},
                "kernel_name_hints": ["target_kernel"],
                "resources": {"host_id": "test-host", "gpu_uuids": []},
                "operation_timeout_seconds": 5.0,
                "command_timeout_seconds": 1.0,
                "resource_wait_timeout_seconds": 1.0,
                "cleanup_timeout_seconds": 1.0,
                "launch_deadline": time.time() + 3.0,
            }
            request_path = Path(temporary) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            gpu_request = {
                **request,
                "resources": {
                    "host_id": "test-host",
                    "gpu_uuids": ["GPU-0"],
                },
            }
            with self.assertRaisesRegex(
                profile_ncu.NcuError,
                "must not request GPU",
            ):
                profile_ncu._validate_analyze(gpu_request)

            completed = subprocess.run(
                [
                    "python3",
                    str(SCRIPTS / "profile_ncu.py"),
                    "analyze",
                    "--request",
                    str(request_path),
                    "--wait",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["measurement_validity"], "valid")
            self.assertEqual(result["provenance"]["tool"]["version"], "2026.2.1")
            self.assertFalse((root / "objects" / "sha256" / report_object["digest"] / "payload" / "report.csv").samefile(report))


if __name__ == "__main__":
    unittest.main()
