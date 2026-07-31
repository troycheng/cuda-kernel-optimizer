from __future__ import annotations

import hashlib
import importlib.util
import json
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


class NsysAnalyzeTests(unittest.TestCase):
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
