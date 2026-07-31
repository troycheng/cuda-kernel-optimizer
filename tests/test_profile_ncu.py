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


class ProfileNcuTests(unittest.TestCase):
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
