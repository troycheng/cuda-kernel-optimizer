from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import sqlite3
import tempfile
import time
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "profile_nsys.py"


def _load():
    spec = importlib.util.spec_from_file_location("v14_profile_nsys", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _canonical_id(material: dict) -> str:
    identity = {key: material[key] for key in ("kind", "tool", "tool_version", "dialect", "object_ref")}
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _nsys_sqlite(path: Path, rows=((10, 35, 1),), *, schema_version="3.25.0", kernel_extra=True) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    connection.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_SCHEMA_VERSION', ?)", (schema_version,))
    connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO StringIds VALUES (1, 'kernel_a')")
    extra = ", extra TEXT" if kernel_extra else ""
    connection.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        f"(start INTEGER, end INTEGER, demangledName INTEGER{extra})"
    )
    for start, end, name_id in rows:
        values = (start, end, name_id, "x") if kernel_extra else (start, end, name_id)
        placeholders = ", ".join("?" for _ in values)
        connection.execute(f"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({placeholders})", values)
    connection.execute("CREATE TABLE EXTRA_TABLE (value TEXT)")
    connection.commit()
    connection.close()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _collection_fixture(temporary: str) -> tuple[dict, dict, Path]:
    """Create one frozen Target and fake Nsys/driver pair for public collection."""
    store = _load_sibling("artifact_store")
    adapter = _load_sibling("workload_adapter")
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
    events = project / "nsys-events.jsonl"
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
    nsys = project / "nsys"
    nsys.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sqlite3, subprocess, sys",
                "from pathlib import Path",
                f"events = Path({str(events)!r})",
                "args = sys.argv[1:]",
                "with events.open('a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps(args) + '\\n')",
                "if args == ['--version']:",
                "    print('NVIDIA Nsight Systems version 2026.2.1')",
                "elif args[0] == 'export':",
                "    output = Path(args[args.index('--output') + 1])",
                "    connection = sqlite3.connect(output)",
                "    connection.execute(\"CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)\")",
                "    connection.execute(\"INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_SCHEMA_VERSION', '3.25.0')\")",
                "    connection.execute(\"CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)\")",
                "    connection.execute(\"INSERT INTO StringIds VALUES (1, 'kernel_a')\")",
                "    connection.execute(\"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)\")",
                "    connection.execute(\"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10, 35, 1)\")",
                "    connection.commit()",
                "    connection.close()",
                "else:",
                "    prefix = Path(args[args.index('--output') + 1])",
                "    prefix.with_suffix('.nsys-rep').write_bytes(b'fixture-nsys-report')",
                f"    index = args.index({str(driver)!r})",
                "    subprocess.run(args[index - 1:], check=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(nsys, 0o700)
    frozen_driver = adapter.validate_driver(
        {
            "command": [sys.executable, str(driver)],
            "request_argument": "--request",
            "execution_mode": "separate",
            "protocol_version": adapter.DRIVER_PROTOCOL,
            "profiler_capabilities": ["nsys_wrap_v1"],
            "side_effects": [],
            "cleanup_contract": {"kind": "process_group_only", "external_tasks": False},
        }
    )
    original_variant = {"role": "original", "kind": "source_snapshot", "digest": original_object["digest"], "locator": original_object["locator"]}
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
        "environment": {"host": {"host_id": "fixture-host", "gpu_uuids": ["GPU-0"], "tools": {"nsys": {"path": str(nsys), "sha256": store.sha256_file(nsys)}}}, "runtime": runtime_environment},
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
        "format_version": "cuda-kernel-optimizer/nsys-input-v1", "operation": "collect",
        "artifact_root": str(root), "target_ref": target_ref,
        "baseline_ref": {"invocation_id": "inv-baseline", "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()},
        "role": "original", "case_id": "main",
        "resources": {"host_id": "fixture-host", "gpu_uuids": ["GPU-0"]},
        "operation_timeout_seconds": 5.0, "command_timeout_seconds": 1.0,
        "resource_wait_timeout_seconds": 1.0, "cleanup_timeout_seconds": 1.0,
        "launch_deadline": time.time() + 3.0,
    }
    return request, {"events": events, "root": root, "driver": frozen_driver}, root


def _load_sibling(name: str):
    path = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"v14_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NsysAnalyzeTests(unittest.TestCase):
    def test_public_collect_wraps_the_only_driver_argv_and_freezes_raw_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request, fixture, root = _collection_fixture(temporary)
            request_path = Path(temporary) / "collect.json"
            _write_json(request_path, request)

            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "collect", "--request",
                    str(request_path), "--wait",
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
                ["kernel.duration"],
            )
            provenance = result["provenance"]
            self.assertEqual(provenance["tool"]["version"], "2026.2.1")
            self.assertIn("driver_output", provenance)
            self.assertIn("report", provenance)
            self.assertIn("sqlite", provenance)
            self.assertTrue((root / provenance["report"]["locator"]).is_dir())
            self.assertTrue((root / provenance["sqlite"]["locator"]).is_dir())
            events = [
                json.loads(line)
                for line in fixture["events"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[0], ["--version"])
            self.assertEqual(
                events[1][:7],
                [
                    "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                    "--cpuctxsw=none", "--stats=false", "--wait=all",
                    "--output",
                ],
            )
            self.assertTrue(Path(events[1][7]).is_absolute())
            self.assertEqual(events[1][8:-2], fixture["driver"]["command"])
            self.assertEqual(events[1][-2], fixture["driver"]["request_argument"])
            self.assertTrue(Path(events[1][-1]).is_absolute())
            self.assertEqual(events[2][:4], ["export", "--type", "sqlite", "--output"])
            self.assertTrue(Path(events[2][4]).is_absolute())
            self.assertTrue(events[2][4].endswith(".sqlite"))
            self.assertTrue(events[2][5].endswith(".nsys-rep"))
            self.assertNotEqual(Path(events[2][5]), Path(events[1][7]).with_suffix(".nsys-rep"))

    def test_candidate_collect_missing_correctness_rejects_before_invocation_or_nsys(self) -> None:
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
                    sys.executable, str(SCRIPT), "collect", "--request",
                    str(request_path), "--wait",
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

    def test_official_sqlite_dialect_joins_string_ids_and_emits_facts(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "report.sqlite"
            _nsys_sqlite(database)
            facts = module.parse_nsys_sqlite(database, "2026.2.1")
            self.assertEqual(facts["observations"][0]["value"], 25.0)
            self.assertEqual(facts["observations"][0]["unit"], "ns")
            self.assertEqual(facts["observations"][0]["scope"], ["kernel", "kernel_a"])
            self.assertIn("EXTRA_TABLE", facts["unmodeled"]["tables"])
            self.assertIn("extra", facts["unmodeled"]["kernel_columns"])

    def test_unsupported_version_and_private_nsys_rep_fail_closed(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / "raw.nsys-rep"
            private.write_bytes(b"private")
            with self.assertRaisesRegex(module.NsysError, "SQLite"):
                module.parse_nsys_sqlite(private, "2026.2.1")
            database = Path(directory) / "report.sqlite"
            _nsys_sqlite(database)
            with self.assertRaisesRegex(module.NsysError, "version"):
                module.parse_nsys_sqlite(database, "2026.3.0")

    def test_dangling_string_id_and_row_limit_fail_closed(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as directory:
            dangling = Path(directory) / "dangling.sqlite"
            _nsys_sqlite(dangling, rows=((1, 9, 99),))
            with self.assertRaisesRegex(module.NsysError, "name"):
                module.parse_nsys_sqlite(dangling, "2026.2")

            too_many = Path(directory) / "too_many.sqlite"
            _nsys_sqlite(too_many, rows=((1, 2, 1), (3, 4, 1)))
            original_limit = module._MAX_ROWS
            module._MAX_ROWS = 1
            try:
                with self.assertRaisesRegex(module.NsysError, "row limit"):
                    module.parse_nsys_sqlite(too_many, "2026.2")
            finally:
                module._MAX_ROWS = original_limit

    def test_duplicate_or_unknown_schema_metadata_fail_closed(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.sqlite"
            _nsys_sqlite(duplicate)
            connection = sqlite3.connect(duplicate)
            connection.execute("INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_SCHEMA_VERSION', '3.25.0')")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(module.NsysError, "schema"):
                module.parse_nsys_sqlite(duplicate, "2026.2")

            unknown = Path(directory) / "unknown.sqlite"
            _nsys_sqlite(unknown, schema_version="3.26.0")
            with self.assertRaisesRegex(module.NsysError, "schema"):
                module.parse_nsys_sqlite(unknown, "2026.2")

    def test_analyze_status_and_cancel_use_one_read_only_invocation(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            root.mkdir()
            database = Path(directory) / "report.sqlite"
            _nsys_sqlite(database, rows=((1, 9, 1),), kernel_extra=False)
            object_ref = module.STORE.freeze_path(
                root,
                database,
                {"max_files": 10, "max_total_bytes": 1024 * 1024, "max_wall_seconds": 1.0},
            )
            material = {
                "sha256": object_ref["digest"],
                "kind": "report",
                "tool": "nsys",
                "tool_version": "2026.2.1",
                "dialect": "nsys-sqlite-3.25-v1",
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
            module.STORE.create_regular_json(root / "target.json", target)
            target_sha = hashlib.sha256((root / "target.json").read_bytes()).hexdigest()
            request = {
                "format_version": "cuda-kernel-optimizer/nsys-input-v1",
                "operation": "analyze",
                "artifact_root": str(root),
                "target_ref": {"id": "diagnostic-target", "sha256": target_sha},
                "report_ref": {"id": material["id"], "sha256": object_ref["digest"]},
                "resources": {"host_id": "local", "gpu_uuids": []},
                "operation_timeout_seconds": 2.0,
                "command_timeout_seconds": 1.0,
                "resource_wait_timeout_seconds": 1.0,
                "cleanup_timeout_seconds": 1.0,
                "launch_deadline": time.time() + 1.0,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                result = module.analyze(request, wait_for_result=True)
            self.assertEqual(result["execution_status"], "succeeded")
            time.sleep(0.1)
            status = module._status_or_cancel(
                {"format_version": request["format_version"], "operation": "status", "artifact_root": str(root), "invocation_id": result["invocation_id"]},
                "status",
            )
            self.assertEqual(status["query_status"], "completed")
            cancelled = module._status_or_cancel(
                {"format_version": request["format_version"], "operation": "cancel", "artifact_root": str(root), "invocation_id": result["invocation_id"]},
                "cancel",
            )
            self.assertEqual(cancelled["query_status"], "completed")


if __name__ == "__main__":
    unittest.main()
