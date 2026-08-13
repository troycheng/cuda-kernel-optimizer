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
from unittest import mock

from tests.v14_support import V14Project, decode_stdout, write_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
TOOL = SCRIPTS / "compiler_evidence.py"


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load("artifact_store.py", "compiler_evidence_test_store")


class CompilerProject:
    def __init__(self, root: Path, materials: dict[str, tuple[str, bytes]]) -> None:
        self.root = root
        self.artifact_root = root / "artifacts"
        records = []
        for stage, (dialect, content) in materials.items():
            path = root / f"{stage}.artifact"
            path.write_bytes(content)
            object_ref = STORE.freeze_path(
                self.artifact_root,
                path,
                {"max_files": 1, "max_total_bytes": 1024 * 1024, "max_wall_seconds": 2.0},
            )
            records.append(
                {
                    "id": f"material-{stage}",
                    "sha256": object_ref["digest"],
                    "kind": "compiler",
                    "tool": "fixture",
                    "tool_version": "1",
                    "dialect": dialect,
                    "object_ref": object_ref,
                }
            )
        target = {
            "record_type": "target",
            "format_version": "cuda-kernel-optimizer/target-v2",
            "id": "target-compiler",
            "target_mode": "diagnostic",
            "diagnostic_materials": records,
            "environment": {
                "host": {"host_id": "local-test", "gpu_uuids": [], "tools": {}},
                "runtime": {"status": "unavailable"},
            },
        }
        STORE.create_regular_json(self.artifact_root / "target.json", target)
        self.target_ref = {
            "id": target["id"],
            "sha256": STORE.sha256_file(self.artifact_root / "target.json"),
        }

    def request(self, selected_stage: str, **artifact_changes) -> dict:
        artifact_ref = {
            "source": "target_material",
            "material_ref": {
                "id": f"material-{selected_stage}",
                "sha256": next(
                    item["sha256"]
                    for item in json.loads(
                        (self.artifact_root / "target.json").read_text("utf-8")
                    )["diagnostic_materials"]
                    if item["id"] == f"material-{selected_stage}"
                ),
            },
            "stage": selected_stage,
        }
        artifact_ref.update(artifact_changes)
        return {
            "format_version": "cuda-kernel-optimizer/compiler-input-v1",
            "operation": "analyze",
            "artifact_root": str(self.artifact_root),
            "target_ref": self.target_ref,
            "artifact_ref": artifact_ref,
            "resources": {"host_id": "local-test", "gpu_uuids": []},
            "operation_timeout_seconds": 3.0,
            "command_timeout_seconds": 1.0,
            "resource_wait_timeout_seconds": 1.0,
            "cleanup_timeout_seconds": 1.0,
            "launch_deadline": time.time() + 2.0,
        }

    def run(self, request: dict, operation="analyze", wait=True):
        request_path = self.root / f"compiler-{operation}-{time.time_ns()}.json"
        write_json(request_path, request)
        argv = [sys.executable, str(TOOL), operation, "--request", str(request_path)]
        if wait:
            argv.append("--wait")
        return subprocess.run(argv, text=True, capture_output=True, timeout=10)


class CompilerEvidenceTests(unittest.TestCase):
    def test_cli_exposes_only_analyze_status_cancel_and_has_no_legacy_api(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(TOOL), "--help"], text=True, capture_output=True, timeout=5
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("{analyze,status,cancel}", help_result.stdout)
        for forbidden in ("collect", "cache", "workload"):
            self.assertNotIn(forbidden, help_result.stdout.lower())
        self.assertNotIn("{compile", help_result.stdout.lower())

        module = _load("compiler_evidence.py", "compiler_evidence_surface_test")
        for removed in (
            "collect", "update_manifest", "write_fresh_manifest", "load_manifest",
            "snapshot_cache", "discover_triton_cache", "publish_triton_stages",
            "same_artifact", "atomic_write_text",
        ):
            self.assertFalse(hasattr(module, removed), removed)

    def test_diagnostic_ptx_analyze_status_and_cancel_report_bounded_facts(self) -> None:
        content = b".version 8.0\n.target sm_90\n.visible .entry kernel() {\n  ret;\n}\n"
        with tempfile.TemporaryDirectory() as temporary:
            project = CompilerProject(Path(temporary), {"ptx": ("ptx-v1", content)})
            analyzed = project.run(project.request("ptx"))
            self.assertEqual(analyzed.returncode, 0, analyzed.stderr)
            self.assertTrue(analyzed.stdout, "analyze must emit one JSON result")
            result = decode_stdout(analyzed)

            self.assertEqual(result["execution_status"], "succeeded")
            self.assertEqual(result["evidence_validity"], "valid")
            self.assertEqual(result["observations"], [{
                "kind": "compiler_artifact_facts",
                "declared_stage": "ptx",
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "text": {
                    "encoding": "utf-8",
                    "line_count": 5,
                    "ends_with_newline": True,
                    "structural_markers": [".version", ".target", ".entry_or_func"],
                },
            }])
            self.assertEqual(result["source_binding"]["source"], "target_material")
            self.assertEqual(result["environment_binding"]["host"]["host_id"], "local-test")

            status_request = {
                "format_version": "cuda-kernel-optimizer/compiler-input-v1",
                "operation": "status",
                "artifact_root": str(project.artifact_root),
                "invocation_id": result["invocation_id"],
            }
            status = project.run(status_request, "status", wait=False)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertEqual(decode_stdout(status)["query_status"], "completed")
            status_request["operation"] = "cancel"
            cancelled = project.run(status_request, "cancel", wait=False)
            self.assertEqual(cancelled.returncode, 0, cancelled.stderr)
            self.assertEqual(decode_stdout(cancelled)["query_status"], "completed")

    def test_known_text_dialects_and_binary_emit_only_structural_facts(self) -> None:
        fixtures = {
            "source": ("cuda-source-v1", b"extern \"C\" __global__ void k() {}\n"),
            "ttir": ("triton-ttir-v1", b"module { tt.func public @k() { tt.return } }\n"),
            "ttgir": ("triton-ttgir-v1", b"module attributes {ttg.num-warps = 4 : i32} { tt.func @k() }\n"),
            "llvm_ir": ("llvm-ir-v1", b"define void @k() {\nentry:\n  ret void\n}\n"),
            "sass": ("sass-text-v1", b"Function : k\n/*0000*/ RET;\n"),
            "binary": ("cuda-binary-v1", b"\x7fCUBIN\x00\x01"),
        }
        expected_markers = {
            "ttir": ["module", "tt.func"],
            "ttgir": ["module", "ttg."],
            "llvm_ir": ["define", "function_body"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            project = CompilerProject(Path(temporary), fixtures)
            for stage, (_dialect, content) in fixtures.items():
                with self.subTest(stage=stage):
                    completed = project.run(project.request(stage))
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertTrue(completed.stdout, "analyze must emit one JSON result")
                    observation = decode_stdout(completed)["observations"][0]
                    self.assertEqual(observation["declared_stage"], stage)
                    self.assertEqual(observation["size_bytes"], len(content))
                    if stage == "binary":
                        self.assertNotIn("text", observation)
                    else:
                        self.assertEqual(
                            observation["text"].get("structural_markers", []),
                            expected_markers.get(stage, []),
                        )

    def test_stage_mismatch_fails_before_submit_and_invalid_text_has_no_observation(self) -> None:
        fixtures = {
            "ptx": ("ptx-v1", b"not structurally ptx\n"),
            "ttir": ("triton-ttir-v1", b"module { tt.func @k() }\x00"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            project = CompilerProject(Path(temporary), fixtures)
            completed = project.run(project.request("ptx", stage="llvm_ir"))
            self.assertEqual(completed.returncode, 2)
            authority_mismatch = project.request("ptx")
            authority_mismatch["artifact_ref"]["material_ref"]["sha256"] = "0" * 64
            completed = project.run(authority_mismatch)
            self.assertEqual(completed.returncode, 2)
            self.assertFalse((project.artifact_root / "invocations").exists())

            for stage in ("ptx", "ttir"):
                with self.subTest(stage=stage):
                    completed = project.run(project.request(stage))
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = decode_stdout(completed)
                    self.assertEqual(result["execution_status"], "invalid")
                    self.assertEqual(result["evidence_validity"], "invalid")
                    self.assertEqual(result["observations"], [])

    def test_worker_re_resolves_frozen_bindings_before_materialization(self) -> None:
        content = b".version 8.0\n.target sm_90\n.entry k() { ret; }\n"
        with tempfile.TemporaryDirectory() as temporary:
            project = CompilerProject(Path(temporary), {"ptx": ("ptx-v1", content)})
            module = _load("compiler_evidence.py", "compiler_evidence_worker_test")
            captured = {}

            def capture_submit(_root, frozen, _argv, _wait):
                captured.update(frozen)
                return {"query_status": "starting"}

            with mock.patch.object(module.RUNTIME, "submit", side_effect=capture_submit):
                module.analyze(project.request("ptx"), wait_for_result=False)
            invocation = project.artifact_root / "invocations" / "inv-worker-test"
            invocation.mkdir(parents=True)
            STORE.create_regular_json(invocation / "request.json", captured)
            target = json.loads((project.artifact_root / "target.json").read_text("utf-8"))
            target["environment"]["runtime"] = {"status": "changed"}
            write_json(project.artifact_root / "target.json", target)

            with mock.patch.dict(os.environ, {
                "CKO_ARTIFACT_ROOT": str(project.artifact_root),
                "CKO_INVOCATION_DIR": str(invocation),
            }):
                self.assertEqual(module._worker_main(), 0)
            result = json.loads((invocation / "result.json").read_text("utf-8"))
            self.assertEqual(result["execution_status"], "invalid")
            self.assertEqual(result["observations"], [])
            self.assertFalse((invocation / "workspace").exists())

    def test_evaluator_artifact_is_authority_bound_and_only_selected_member_materializes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = V14Project(Path(temporary))
            driver = project.driver.read_text("utf-8")
            driver = driver.replace("import json\n", "import json\nimport hashlib\n")
            driver = driver.replace(
                "result = {\n",
                "output_dir = Path(request['output_path']).parent\n"
                "ptx = output_dir / 'kernel.ptx'\n"
                "ptx.write_text('.version 8.0\\n.target sm_90\\n.entry k() { ret; }\\n', encoding='utf-8')\n"
                "neighbor = output_dir / 'neighbor.bin'\n"
                "neighbor.write_bytes(b'x' * 4096)\n"
                "result = {\n",
            ).replace(
                "    'artifacts': [],\n",
                "    'artifacts': [\n"
                "        {'kind': 'ptx', 'relative_path': 'kernel.ptx', 'sha256': hashlib.sha256(ptx.read_bytes()).hexdigest()},\n"
                "        {'kind': 'binary', 'relative_path': 'neighbor.bin', 'sha256': hashlib.sha256(neighbor.read_bytes()).hexdigest()},\n"
                "    ],\n",
            )
            project.driver.write_text(driver, encoding="utf-8")
            project.check()
            baseline = project.baseline()
            events_before = project.driver_events()
            request = {
                **project.baseline_input(),
                "format_version": "cuda-kernel-optimizer/compiler-input-v1",
                "operation": "analyze",
                "artifact_ref": {
                    "source": "invocation_driver_artifact",
                    "invocation_ref": baseline["result_ref"],
                    "receipt_index": 0,
                    "relative_path": "kernel.ptx",
                    "stage": "ptx",
                },
            }
            request.pop("sampling_design")
            completed = project.run_tool("compiler_evidence.py", "analyze", request, wait=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(completed.stdout, "analyze must emit one JSON result")
            result = decode_stdout(completed)
            self.assertEqual(result["source_binding"]["role"], "original")
            self.assertEqual(project.driver_events(), events_before)

            workspace = project.artifact_root / "invocations" / result["invocation_id"] / "workspace"
            files = [path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()]
            self.assertEqual(files, ["selected-artifact"])
            self.assertFalse((workspace / "neighbor.bin").exists())

            bad = json.loads(json.dumps(request))
            bad["artifact_ref"]["relative_path"] = "../kernel.ptx"
            invocations_before = {path.name for path in (project.artifact_root / "invocations").iterdir()}
            for changed in (
                {**bad, "artifact_ref": {**bad["artifact_ref"], "relative_path": "../kernel.ptx"}},
                {**request, "artifact_ref": {**request["artifact_ref"], "stage": "binary"}},
            ):
                rejected = project.run_tool("compiler_evidence.py", "analyze", changed, wait=True)
                self.assertEqual(rejected.returncode, 2)
            self.assertEqual(
                {path.name for path in (project.artifact_root / "invocations").iterdir()},
                invocations_before,
            )


if __name__ == "__main__":
    unittest.main()
