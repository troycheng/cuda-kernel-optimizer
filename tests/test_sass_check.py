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

from tests.v14_support import V14Project, decode_stdout, write_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"
TOOL = SCRIPTS / "sass_check.py"


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STORE = _load("artifact_store.py", "sass_check_test_store")
VERSION = (
    "cuobjdump: NVIDIA (R) fat binary listing tool\n"
    "Copyright (c) 2005-2026 NVIDIA Corporation\n"
    "Built on Tue_Jun_02_12:00:00_PDT_2026\n"
    "Cuda compilation tools, release 13.2, V13.2.0\n"
    "Build cuda_13.2.r13.2/compiler.37668154_0\n"
)
SASS = (
    "Fatbin elf code:\n"
    "================\n"
    "arch = sm_120\n"
    "code version = [1,7]\n"
    "host = linux\n"
    "compile_size = 64bit\n"
    "code for sm_120\n"
    ".target\tsm_120\n"
    "\t\t..........\n"
    "Function : kernel\n"
    ".headerflags @\"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)\"\n"
    "/*0000*/ MOV R1, c[0x0][0x28]; /* 0x00000a0000017a02 */\n"
    "/*0010*/ FFMA R0, R1, R2, R3; /* 0x0000000201007223 */\n"
)


def _fake_cuobjdump(
    root: Path, calls: Path, *, version=VERSION, sass=SASS,
    dump_exit=0, mutate_after_dump=False, name="fake-cuobjdump",
) -> Path:
    tool = root / name
    tool.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"calls=Path({str(calls)!r})\n"
        "with calls.open('a', encoding='utf-8') as stream:\n"
        " stream.write(json.dumps(sys.argv[1:])+'\\n')\n"
        f"version={version!r}; sass={sass!r}; dump_exit={dump_exit!r}; mutate={mutate_after_dump!r}\n"
        "if sys.argv[1:] == ['--version']:\n"
        " sys.stdout.write(version); raise SystemExit(0)\n"
        "if len(sys.argv) == 3 and sys.argv[1] == '--dump-sass':\n"
        " sys.stdout.write(sass)\n"
        " if mutate: Path(sys.argv[0]).write_text('#!/bin/sh\\nexit 9\\n', encoding='utf-8')\n"
        " raise SystemExit(dump_exit)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    return tool


class SassProject:
    def __init__(
        self, root: Path, *, version=VERSION, sass=SASS,
        dump_exit=0, mutate_after_dump=False,
    ) -> None:
        self.root = root
        self.artifact_root = root / "artifacts"
        self.calls = root / "cuobjdump-calls.jsonl"
        self.tool = _fake_cuobjdump(
            root, self.calls, version=version, sass=sass, dump_exit=dump_exit,
            mutate_after_dump=mutate_after_dump,
        )
        binary = root / "kernel.cubin"
        binary.write_bytes(b"\x7fCUBIN\x00fixture")
        object_ref = STORE.freeze_path(
            self.artifact_root,
            binary,
            {"max_files": 1, "max_total_bytes": 1024, "max_wall_seconds": 2.0},
        )
        material = {
            "id": "material-binary",
            "sha256": object_ref["digest"],
            "kind": "compiler",
            "tool": "fixture",
            "tool_version": "1",
            "dialect": "cuda-binary-v1",
            "object_ref": object_ref,
        }
        target = {
            "record_type": "target",
            "format_version": "cuda-kernel-optimizer/target-v1",
            "id": "target-sass",
            "target_mode": "diagnostic",
            "diagnostic_materials": [material],
            "environment": {
                "host": {
                    "host_id": "local-test",
                    "gpu_uuids": [],
                    "tools": {
                        "cuobjdump": {
                            "path": str(self.tool.resolve()),
                            "sha256": hashlib.sha256(self.tool.read_bytes()).hexdigest(),
                        }
                    },
                },
                "runtime": {"status": "unavailable"},
            },
        }
        STORE.create_regular_json(self.artifact_root / "target.json", target)
        self.target_ref = {
            "id": target["id"],
            "sha256": STORE.sha256_file(self.artifact_root / "target.json"),
        }
        self.material_ref = {"id": material["id"], "sha256": material["sha256"]}

    def request(self) -> dict:
        return {
            "format_version": "cuda-kernel-optimizer/sass-input-v1",
            "operation": "analyze",
            "artifact_root": str(self.artifact_root),
            "target_ref": self.target_ref,
            "artifact_ref": {
                "source": "target_material",
                "material_ref": self.material_ref,
            },
            "resources": {"host_id": "local-test", "gpu_uuids": []},
            "operation_timeout_seconds": 5.0,
            "command_timeout_seconds": 3.0,
            "resource_wait_timeout_seconds": 1.0,
            "cleanup_timeout_seconds": 1.0,
            "launch_deadline": time.time() + 5.0,
        }

    def run(self, request: dict, operation="analyze", wait=True):
        path = self.root / f"sass-{operation}-{time.time_ns()}.json"
        write_json(path, request)
        argv = [sys.executable, str(TOOL), operation, "--request", str(path)]
        if wait:
            argv.append("--wait")
        return subprocess.run(argv, text=True, capture_output=True, timeout=10)


class SassCheckTests(unittest.TestCase):
    def test_cli_only_analyzes_frozen_binary_with_two_guarded_commands(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(TOOL), "--help"], text=True, capture_output=True, timeout=5
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("{analyze,status,cancel}", help_result.stdout)
        for forbidden in ("build", "compile", "workload", "state"):
            self.assertNotIn(forbidden, help_result.stdout.lower())
        module = _load("sass_check.py", "sass_check_surface_test")
        for removed in ("run", "_dump_sass", "_find_so_file", "check_method_sass"):
            self.assertFalse(hasattr(module, removed), removed)

        with tempfile.TemporaryDirectory() as temporary:
            project = SassProject(Path(temporary))
            completed = project.run(project.request())
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = decode_stdout(completed)
            self.assertEqual(result["execution_status"], "succeeded")
            self.assertEqual(result["evidence_validity"], "valid")
            self.assertEqual(result["observations"], [{
                "kind": "sass_facts",
                "content_sha256": hashlib.sha256(SASS.encode()).hexdigest(),
                "size_bytes": len(SASS.encode()),
                "architectures": ["sm_120"],
                "function_count": 1,
                "instruction_count": 2,
                "opcode_counts": {"FFMA": 1, "MOV": 1},
            }])
            self.assertNotIn(SASS, completed.stdout)
            calls = [json.loads(line) for line in project.calls.read_text("utf-8").splitlines()]
            self.assertEqual(calls[0], ["--version"])
            self.assertEqual(calls[1][0], "--dump-sass")
            self.assertEqual(len(calls), 2)
            provenance = result["provenance"]
            self.assertEqual(provenance["capture"]["sha256"], hashlib.sha256(SASS.encode()).hexdigest())
            self.assertEqual(provenance["capture"]["size_bytes"], len(SASS.encode()))
            self.assertEqual(provenance["cuobjdump"]["path"], str(project.tool.resolve()))

            status_request = {
                "format_version": "cuda-kernel-optimizer/sass-input-v1",
                "operation": "status",
                "artifact_root": str(project.artifact_root),
                "invocation_id": result["invocation_id"],
            }
            self.assertEqual(decode_stdout(project.run(status_request, "status", False))["query_status"], "completed")
            status_request["operation"] = "cancel"
            self.assertEqual(decode_stdout(project.run(status_request, "cancel", False))["query_status"], "completed")

    def test_host_gpu_version_and_output_dialect_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SassProject(Path(temporary), version="unknown tool\n")
            for changed in (
                {**project.request(), "resources": {"host_id": "other", "gpu_uuids": []}},
                {**project.request(), "resources": {"host_id": "local-test", "gpu_uuids": ["GPU-1"]}},
            ):
                rejected = project.run(changed)
                self.assertEqual(rejected.returncode, 2)
            completed = project.run(project.request())
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = decode_stdout(completed)
            self.assertEqual(result["execution_status"], "invalid")
            self.assertEqual(result["observations"], [])
            self.assertEqual(result["stop_reason"], "unsupported_cuobjdump_version")
            self.assertEqual(result["cleanup_status"], "confirmed")
            self.assertEqual(
                [json.loads(line) for line in project.calls.read_text("utf-8").splitlines()],
                [["--version"]],
            )

        with tempfile.TemporaryDirectory() as temporary:
            project = SassProject(Path(temporary), sass="arbitrary text\n")
            result = decode_stdout(project.run(project.request()))
            self.assertEqual(result["execution_status"], "invalid")
            self.assertEqual(result["observations"], [])
            self.assertEqual(result["stop_reason"], "unrecognized_sass_dialect")
            self.assertEqual(result["cleanup_status"], "confirmed")
            self.assertEqual(result["provenance"]["capture"]["sha256"], hashlib.sha256(b"arbitrary text\n").hexdigest())

        with tempfile.TemporaryDirectory() as temporary:
            project = SassProject(Path(temporary), dump_exit=3)
            result = decode_stdout(project.run(project.request()))
            self.assertEqual(result["execution_status"], "invalid")
            self.assertEqual(result["observations"], [])
            self.assertEqual(result["stop_reason"], "cuobjdump_dump_failed")
            self.assertEqual(result["cleanup_status"], "confirmed")
            self.assertEqual(result["command_failure"]["stop_reason"], "command_failed")
            self.assertNotIn("capture", result["provenance"])

        with tempfile.TemporaryDirectory() as temporary:
            project = SassProject(Path(temporary), mutate_after_dump=True)
            result = decode_stdout(project.run(project.request()))
            self.assertEqual(result["execution_status"], "invalid")
            self.assertEqual(result["stop_reason"], "cuobjdump_identity_changed")
            self.assertEqual(result["cleanup_status"], "confirmed")
            self.assertIn("cuobjdump", result["provenance"])
            self.assertNotIn("capture", result["provenance"])

    def test_candidate_signature_reports_patterns_as_facts_without_verdict(self) -> None:
        module = _load("sass_check.py", "sass_check_signature_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "sass.txt"
            path.write_text(SASS, encoding="utf-8")
            for separator in (".........", "..........."):
                unsupported = root / f"separator-{len(separator)}.txt"
                unsupported.write_text(
                    SASS.replace("..........", separator), encoding="utf-8"
                )
                with self.subTest(separator=separator), self.assertRaises(module.SassError) as separator_error:
                    module._sass_facts(unsupported)
                self.assertEqual(separator_error.exception.code, "unrecognized_sass_dialect")
            observed = module._signature_facts(path, "compute.fma_and_fast_math")
            self.assertEqual(observed["mechanism_key"], "compute.fma_and_fast_math")
            self.assertIn("FFMA", observed["patterns_found"])
            self.assertNotIn("verified", observed)
            self.assertNotIn("status", observed)
            unavailable = module._signature_facts(path, "test.fixture.latency")
            self.assertEqual(unavailable["applicability"], "not_applicable")
            self.assertEqual(unavailable["reason"], "signature_unavailable")
            oversized = root / "oversized.txt"
            oversized.write_bytes(b"x" * (module._MAX_SASS_LINE_BYTES + 1))
            with self.assertRaises(module.SassError) as line_error:
                module._sass_facts(oversized)
            self.assertEqual(line_error.exception.code, "sass_line_too_long")
            malformed = root / "signatures.json"
            malformed.write_text("{}\n", encoding="utf-8")
            original = module.SIGNATURES
            try:
                module.SIGNATURES = malformed
                with self.assertRaises(module.SassError) as catalog_error:
                    module._signature_facts(path, "compute.fma_and_fast_math")
            finally:
                module.SIGNATURES = original
            self.assertEqual(catalog_error.exception.code, "signature_catalog_invalid")
            malformed.write_text(json.dumps({
                "$note": "fixture", "$version": "4.0",
                "methods": {"unsafe": {
                    "sass_patterns": ["(A+)+"], "require_any": True, "note": "unsafe",
                }},
            }), encoding="utf-8")
            module.SIGNATURES = malformed
            try:
                with self.assertRaises(module.SassError) as unsafe_error:
                    module._signature_facts(path, "unsafe")
            finally:
                module.SIGNATURES = original
            self.assertEqual(unsafe_error.exception.code, "signature_catalog_invalid")
            excessive = root / "architectures.txt"
            excessive.write_text(
                "Fatbin elf code:\n" + "".join(
                    f"arch = sm_{index}\n" for index in range(module._MAX_ARCHITECTURES + 1)
                ), encoding="utf-8",
            )
            with self.assertRaises(module.SassError) as fact_limit:
                module._sass_facts(excessive)
            self.assertEqual(fact_limit.exception.code, "sass_fact_limit_exceeded")
            original_child = module.RUNTIME.run_child
            try:
                module.RUNTIME.run_child = lambda _command: {
                    "status": "timed_out", "stop_reason": "command_timeout",
                    "cleanup_status": "unknown", "returncode": None,
                }
                with self.assertRaises(module.SassError) as child_error:
                    module._child({}, "cuobjdump_dump_failed")
            finally:
                module.RUNTIME.run_child = original_child
            self.assertEqual(child_error.exception.child_fact["cleanup_status"], "unknown")
            self.assertEqual(child_error.exception.child_fact["stop_reason"], "command_timeout")

    def test_candidate_binary_is_experiment_bound_without_new_driver_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = V14Project(root)
            calls = root / "candidate-cuobjdump-calls.jsonl"
            _fake_cuobjdump(root, calls, name="cuobjdump")
            driver = project.driver.read_text("utf-8")
            driver = driver.replace("import json\n", "import json\nimport hashlib\n")
            driver = driver.replace(
                "result = {\n",
                "output_dir = Path(request['output_path']).parent\n"
                "binary = output_dir / 'kernel.cubin'\n"
                "binary.write_bytes(b'\\x7fCUBIN-candidate')\n"
                "result = {\n",
            ).replace(
                "    'artifacts': [],\n",
                "    'artifacts': [{'kind': 'binary', 'relative_path': 'kernel.cubin', "
                "'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}],\n",
            )
            project.driver.write_text(driver, encoding="utf-8")
            environment = {**os.environ, "PATH": f"{root}:{os.environ.get('PATH', '')}"}
            readiness = project.readiness_input()
            readiness["environment_requirements"]["required_tools"] = ["cuobjdump"]
            request_path = root / "readiness.json"
            write_json(request_path, readiness)
            checked = subprocess.run(
                [sys.executable, str(SCRIPTS / "readiness.py"), "check", "--request", str(request_path)],
                text=True, capture_output=True, timeout=10, env=environment,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            baseline = project.baseline()
            experiment_input = project.experiment_input(baseline["result_ref"])
            experiment_input["mechanism_key"] = "compute.fma_and_fast_math"
            experiment = decode_stdout(project.run_tool(
                "workload_evaluate.py", "experiment", experiment_input
            ))
            screened = decode_stdout(project.run_tool(
                "workload_evaluate.py", "screen", project.screen_input(experiment["experiment_ref"]), wait=True
            ))
            receipt_index = next(
                index for index, receipt in enumerate(screened["command_receipts"])
                if receipt["request"]["role"] == "candidate"
            )
            target = json.loads((project.artifact_root / "target.json").read_text("utf-8"))
            request = {
                "format_version": "cuda-kernel-optimizer/sass-input-v1",
                "operation": "analyze", "artifact_root": str(project.artifact_root),
                "target_ref": project.target_ref(),
                "artifact_ref": {
                    "source": "invocation_driver_artifact", "invocation_ref": screened["result_ref"],
                    "receipt_index": receipt_index, "relative_path": "kernel.cubin",
                },
                "resources": {"host_id": target["environment"]["host"]["host_id"], "gpu_uuids": []},
                "operation_timeout_seconds": 5.0, "command_timeout_seconds": 3.0,
                "resource_wait_timeout_seconds": 1.0, "cleanup_timeout_seconds": 1.0,
                "launch_deadline": time.time() + 5.0,
            }
            events_before = project.driver_events()
            completed = project.run_tool("sass_check.py", "analyze", request, wait=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = decode_stdout(completed)
            self.assertEqual(result["execution_status"], "succeeded")
            self.assertEqual(result["source_binding"]["role"], "candidate")
            self.assertEqual(result["source_binding"]["mechanism_key"], "compute.fma_and_fast_math")
            self.assertEqual(result["observations"][1]["applicability"], "applicable")
            self.assertIn("FFMA", result["observations"][1]["patterns_found"])
            self.assertEqual(project.driver_events(), events_before)


if __name__ == "__main__":
    unittest.main()
