from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_STORE_PATH = (
    ROOT / "skills" / "cuda-kernel-optimizer" / "scripts" / "artifact_store.py"
)


def _load_artifact_store():
    spec = importlib.util.spec_from_file_location(
        "cuda_optimizer_artifact_store", ARTIFACT_STORE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts = _load_artifact_store()

    def test_sha256_file_rejects_missing_path_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.py"
            with self.assertRaisesRegex(ValueError, str(missing)):
                self.artifacts.sha256_file(missing)
            with self.assertRaisesRegex(ValueError, str(root)):
                self.artifacts.sha256_file(root)

    def test_sha256_file_rejects_symlinked_parent_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real = root / "real"
            real.mkdir()
            source = real / "value.bin"
            source.write_bytes(b"value")
            linked = root / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                self.artifacts.sha256_file(linked / "value.bin")

    def test_create_regular_json_is_create_once_and_no_follow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root / "decision.json"
            self.artifacts.create_regular_json(target, {"ok": True})
            self.assertEqual(json.loads(target.read_text("utf-8")), {"ok": True})
            with self.assertRaises(FileExistsError):
                self.artifacts.create_regular_json(target, {"ok": False})
            self.assertEqual(json.loads(target.read_text("utf-8")), {"ok": True})

            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                self.artifacts.create_regular_json(linked / "escaped.json", {})

    def test_snapshot_rejects_known_oversize_file_before_first_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve() / "oversize.bin"
            source.write_bytes(b"xx")
            real_read = self.artifacts.os.read
            read_calls = 0

            def tracked_read(*args, **kwargs):
                nonlocal read_calls
                read_calls += 1
                return real_read(*args, **kwargs)

            with mock.patch.object(self.artifacts.os, "read", side_effect=tracked_read):
                with self.assertRaisesRegex(ValueError, "exceeded"):
                    self.artifacts.freeze_path(
                        Path(tmp).resolve() / "artifacts",
                        source,
                        {
                            "max_files": 1,
                            "max_total_bytes": 1,
                            "max_wall_seconds": 1.0,
                        },
                    )

            self.assertEqual(read_calls, 0)

    def test_freeze_path_rejects_source_mutation_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source.bin"
            source.write_bytes(b"x" * (2 * 1024 * 1024))
            real_read = self.artifacts.os.read
            mutated = False

            def mutate_after_first_read(*args, **kwargs):
                nonlocal mutated
                chunk = real_read(*args, **kwargs)
                if chunk and not mutated:
                    mutated = True
                    with source.open("ab") as stream:
                        stream.write(b"changed")
                return chunk

            with mock.patch.object(
                self.artifacts.os, "read", side_effect=mutate_after_first_read
            ):
                with self.assertRaisesRegex(ValueError, "changed while reading"):
                    self.artifacts.freeze_path(
                        root / "artifacts",
                        source,
                        {
                            "max_files": 1,
                            "max_total_bytes": 3 * 1024 * 1024,
                            "max_wall_seconds": 2.0,
                        },
                    )

            linked = root / "linked.bin"
            try:
                linked.symlink_to(source)
            except OSError:
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.artifacts.freeze_path(
                    root / "artifacts",
                    linked,
                    {
                        "max_files": 1,
                        "max_total_bytes": 3 * 1024 * 1024,
                        "max_wall_seconds": 2.0,
                    },
                )

            durable_source = root / "durable.bin"
            durable_source.write_bytes(b"durable")
            events = []
            real_fchmod = self.artifacts.os.fchmod
            real_fsync = self.artifacts.os.fsync

            def tracked_fchmod(fd, mode):
                events.append(("fchmod", fd))
                return real_fchmod(fd, mode)

            def tracked_fsync(fd):
                events.append(("fsync", fd))
                return real_fsync(fd)

            with mock.patch.object(
                self.artifacts.os, "fchmod", side_effect=tracked_fchmod
            ), mock.patch.object(
                self.artifacts.os, "fsync", side_effect=tracked_fsync
            ):
                self.artifacts.freeze_path(
                    root / "durable-artifacts",
                    durable_source,
                    {
                        "max_files": 1,
                        "max_total_bytes": 1024,
                        "max_wall_seconds": 1.0,
                    },
                )

            fchmod_index, destination_fd = next(
                (index, fd)
                for index, (event, fd) in enumerate(events)
                if event == "fchmod"
            )
            self.assertTrue(
                any(
                    event == "fsync" and fd == destination_fd
                    for event, fd in events[fchmod_index + 1 :]
                )
            )

    def test_materialize_object_rejects_tampered_payload_and_cleans_temporary_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source.txt"
            source.write_bytes(b"original")
            frozen = self.artifacts.freeze_path(
                root / "artifacts",
                source,
                {
                    "max_files": 1,
                    "max_total_bytes": 1024,
                    "max_wall_seconds": 1.0,
                },
            )
            object_root = root / "artifacts" / frozen["locator"]
            (object_root / "payload" / "source.txt").write_bytes(b"tampered")
            destination = root / "output.txt"

            with self.assertRaisesRegex(ValueError, "payload does not match"):
                self.artifacts.materialize_object(
                    root / "artifacts", frozen, destination
                )

            self.assertFalse(destination.exists())
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")), []
            )

    def test_materialize_rejects_symlink_parent_before_staging_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source.txt"
            source.write_bytes(b"original")
            frozen = self.artifacts.freeze_path(
                root / "artifacts",
                source,
                {
                    "max_files": 1,
                    "max_total_bytes": 1024,
                    "max_wall_seconds": 1.0,
                },
            )
            outside = root / "outside"
            outside.mkdir()
            linked_parent = root / "linked-parent"
            try:
                linked_parent.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks are unavailable")

            real_copy = self.artifacts._copy_frozen_member
            with mock.patch.object(
                self.artifacts,
                "_copy_frozen_member",
                wraps=real_copy,
            ) as copy_member:
                with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                    self.artifacts.materialize_object(
                        root / "artifacts", frozen, linked_parent / "all.txt"
                    )
                with self.assertRaisesRegex(ValueError, "parent.*symlink|unsafe"):
                    self.artifacts.materialize_object_member(
                        root / "artifacts",
                        frozen,
                        "source.txt",
                        linked_parent / "member.txt",
                    )

            copy_member.assert_not_called()
            self.assertEqual(list(outside.iterdir()), [])

    def test_materialize_nested_path_swap_cannot_escape_private_dirfd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source"
            (source / "nested").mkdir(parents=True)
            (source / "nested" / "payload.bin").write_bytes(b"frozen")
            frozen = self.artifacts.freeze_path(
                root / "artifacts",
                source,
                {
                    "max_files": 2,
                    "max_total_bytes": 1024,
                    "max_wall_seconds": 1.0,
                },
            )
            outside = root / "outside"
            outside.mkdir()
            destination = root / "materialized"
            real_mkdir = Path.mkdir
            swapped = False

            def swap_path_after_mkdir(path, mode=0o777, parents=False, exist_ok=False):
                nonlocal swapped
                result = real_mkdir(
                    path, mode=mode, parents=parents, exist_ok=exist_ok
                )
                materialized = Path(path)
                if (
                    not swapped
                    and materialized.name == "nested"
                    and materialized.parent.name.endswith(".tmp")
                ):
                    moved = materialized.with_name("nested-moved")
                    materialized.rename(moved)
                    materialized.symlink_to(outside, target_is_directory=True)
                    swapped = True
                return result

            with mock.patch.object(Path, "mkdir", new=swap_path_after_mkdir):
                self.artifacts.materialize_object(
                    root / "artifacts", frozen, destination
                )

            self.assertEqual(
                (destination / "nested" / "payload.bin").read_bytes(), b"frozen"
            )
            self.assertEqual(list(outside.iterdir()), [])

    def test_large_object_freeze_and_materialize_use_bounded_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "large.bin"
            chunk = b"x" * (1024 * 1024)
            with source.open("wb") as stream:
                for _ in range(24):
                    stream.write(chunk)

            tracemalloc.start()
            frozen = self.artifacts.freeze_path(
                root / "artifacts",
                source,
                {
                    "max_files": 1,
                    "max_total_bytes": 32 * 1024 * 1024,
                    "max_wall_seconds": 5.0,
                },
            )
            destination = root / "materialized.bin"
            self.artifacts.materialize_object(
                root / "artifacts", frozen, destination
            )
            destination_digest = self.artifacts.sha256_file(destination)
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.assertLess(peak, 12 * 1024 * 1024)
            self.assertEqual(
                self.artifacts.sha256_file(source),
                destination_digest,
            )

    def test_atomic_write_bytes_fsyncs_file_before_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve() / "result.bin"
            events = []
            real_replace = self.artifacts.os.replace
            real_fsync = self.artifacts.os.fsync

            def tracked_replace(source, destination, **kwargs):
                real_replace(source, destination, **kwargs)
                events.append("replace")

            def tracked_fsync(fd):
                mode = os.fstat(fd).st_mode
                events.append("dir_fsync" if stat.S_ISDIR(mode) else "file_fsync")
                return real_fsync(fd)

            with mock.patch.object(
                self.artifacts.os, "replace", side_effect=tracked_replace
            ), mock.patch.object(
                self.artifacts.os, "fsync", side_effect=tracked_fsync
            ):
                self.artifacts.atomic_write_bytes(target, b"complete")

            self.assertLess(events.index("file_fsync"), events.index("replace"))
            self.assertLess(events.index("replace"), events.index("dir_fsync"))
            self.assertEqual(target.read_bytes(), b"complete")

    def test_append_regular_bytes_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve() / "events.jsonl"
            self.artifacts.append_regular_bytes(target, b'{"index":1}\n')
            self.artifacts.append_regular_bytes(target, b'{"index":2}\n')
            self.assertEqual(
                self.artifacts.read_regular_bytes(target),
                b'{"index":1}\n{"index":2}\n',
            )

    def test_directory_and_file_primitives_have_closed_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "invocations"
            root.mkdir()
            first = self.artifacts.create_regular_directory(root / "inv-b")
            second = self.artifacts.create_regular_directory(root / "inv-a")
            self.assertEqual(
                self.artifacts.list_regular_directories(root, prefix="inv-"),
                ["inv-a", "inv-b"],
            )
            with self.assertRaises(FileExistsError):
                self.artifacts.create_regular_directory(first)
            self.assertEqual(
                self.artifacts.create_regular_directory(second, exist_ok=True),
                second,
            )

            result = first / "result.json"
            self.artifacts.create_regular_bytes(result, b"{}")
            self.assertTrue(self.artifacts.remove_regular_file(result))
            self.assertFalse(self.artifacts.remove_regular_file(result))

    def test_compare_and_swap_allows_only_one_concurrent_reference_update(self) -> None:
        self.assertTrue(
            hasattr(self.artifacts, "compare_and_swap_ref"),
            "V1.4 compare-and-swap reference primitive is not implemented",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "artifacts"
            (root / "champion").mkdir(parents=True)
            initial_digest = self.artifacts.compare_and_swap_ref(
                root,
                "champion/current.json",
                None,
                {"selection_ref": "initial"},
            )
            invoke = root.parent / "cas.py"
            invoke.write_text(
                "\n".join(
                    [
                        "import importlib.util",
                        "import json",
                        "import sys",
                        "import time",
                        "from pathlib import Path",
                        "module_path, root, expected, label, go, out = sys.argv[1:]",
                        "spec = importlib.util.spec_from_file_location(",
                        "    'cuda_optimizer_artifact_store_subprocess', module_path",
                        ")",
                        "store = importlib.util.module_from_spec(spec)",
                        "spec.loader.exec_module(store)",
                        "while not Path(go).exists():",
                        "    time.sleep(0.005)",
                        "try:",
                        "    digest = store.compare_and_swap_ref(",
                        "        Path(root),",
                        "        'champion/current.json',",
                        "        expected,",
                        "        {'selection_ref': label},",
                        "    )",
                        "    result = {'status': 'success', 'digest': digest}",
                        "except store.StaleReferenceError:",
                        "    result = {'status': 'stale'}",
                        "Path(out).write_text(json.dumps(result), encoding='utf-8')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            go = root.parent / "go"
            processes = []
            for label in ("first", "second"):
                output = root.parent / f"{label}.json"
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(invoke),
                            str(ARTIFACT_STORE_PATH),
                            str(root),
                            initial_digest,
                            label,
                            str(go),
                            str(output),
                        ]
                    )
                )
            go.write_text("go\n", encoding="utf-8")
            returncodes = [process.wait(timeout=5) for process in processes]
            self.assertEqual(returncodes, [0, 0])

            outcomes = [
                json.loads((root.parent / f"{label}.json").read_text("utf-8"))
                for label in ("first", "second")
            ]
            self.assertEqual(
                sorted(outcome["status"] for outcome in outcomes),
                ["stale", "success"],
            )
            current = json.loads(
                (root / "champion" / "current.json").read_text("utf-8")
            )
            self.assertIn(current["selection_ref"], {"first", "second"})
            successful = next(
                outcome for outcome in outcomes if outcome["status"] == "success"
            )
            self.assertEqual(
                successful["digest"],
                self.artifacts.sha256_file(root / "champion" / "current.json"),
            )

    def test_immutable_result_publication_is_guarded_by_reference_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            (root / "champion").mkdir(parents=True)
            digest = self.artifacts.compare_and_swap_ref(
                root,
                "champion/current.json",
                None,
                {"selection_ref": "first"},
            )
            self.artifacts.create_regular_json_if_ref_digest(
                root,
                "champion/current.json",
                digest,
                "invocations/inv-a/result.json",
                {"status": "current"},
            )
            self.assertEqual(
                json.loads(
                    (root / "invocations" / "inv-a" / "result.json").read_text(
                        "utf-8"
                    )
                ),
                {"status": "current"},
            )

            self.artifacts.compare_and_swap_ref(
                root,
                "champion/current.json",
                digest,
                {"selection_ref": "second"},
            )
            with self.assertRaises(self.artifacts.StaleReferenceError):
                self.artifacts.create_regular_json_if_ref_digest(
                    root,
                    "champion/current.json",
                    digest,
                    "invocations/inv-b/result.json",
                    {"status": "incorrectly_current"},
                )
            self.assertFalse(
                (root / "invocations" / "inv-b" / "result.json").exists()
            )

    def test_directory_publication_never_replaces_an_existing_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent / "temporary"
            source.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            destination = parent / "artifacts"
            destination.mkdir()
            (destination / "existing.txt").write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                self.artifacts.publish_directory_noreplace(source, destination)

            self.assertTrue(source.is_dir())
            self.assertEqual(
                (destination / "existing.txt").read_text(encoding="utf-8"),
                "existing",
            )
            self.assertFalse((destination / "new.txt").exists())


if __name__ == "__main__":
    unittest.main()
