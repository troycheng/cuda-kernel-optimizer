from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUNNER = Path(__file__).resolve().parent / "remote" / "run_lane.sh"
NCU_AUTHORIZED_RUNNER = (
    Path(__file__).resolve().parent / "remote" / "run_ncu_authorized_smoke.sh"
)
ARTIFACTS = Path(os.environ.get("CUDA_E2E_ARTIFACTS", "/tmp/cuda-sm120-acceptance"))


def _run(command: list[str], json_path: Path) -> dict:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json_path.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(old.st_mode):
            raise AssertionError(f"refusing symlink JSON output: {json_path}")
        if not stat.S_ISREG(old.st_mode):
            raise AssertionError(f"refusing non-regular JSON output: {json_path}")
        json_path.unlink()
    started_ns = time.time_ns()
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(json_path, flags)
    except OSError as error:
        raise AssertionError(f"fresh JSON output is missing or unsafe: {json_path}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise AssertionError(f"fresh JSON output is not regular: {json_path}")
        if opened.st_mtime_ns < started_ns:
            raise AssertionError(f"JSON output is not fresh: {json_path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(opened, field) != getattr(after, field) for field in identity_fields):
            raise AssertionError(f"JSON output changed while reading: {json_path}")
    finally:
        os.close(descriptor)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"fresh JSON output is invalid: {json_path}") from error


def _stage_fixture_workspace(
    name: str, *, artifacts=ARTIFACTS, fixtures=FIXTURES
) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or any(
            not (character.isalnum() or character in "._-")
            for character in name
        )
    ):
        raise ValueError("workspace name must contain only letters, digits, ., _, or -")
    artifact_root = Path(artifacts).expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    fixture_root = Path(fixtures).expanduser().resolve(strict=True)
    if not fixture_root.is_dir():
        raise ValueError("fixtures must be a directory")
    case_root = artifact_root / name
    if case_root.is_symlink():
        raise ValueError("artifact case directory must not be a symlink")
    case_root.mkdir(parents=True, exist_ok=True)
    if case_root.resolve(strict=True).parent != artifact_root:
        raise ValueError("artifact case directory escapes the artifact root")
    workspace = case_root / "workspace"
    if workspace.is_symlink():
        raise ValueError("artifact workspace must not be a symlink")
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(
        fixture_root,
        workspace,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", "*.so", "compiler_evidence", ".*"
        ),
    )
    return workspace


class Sm120AcceptanceHelperTests(unittest.TestCase):
    def test_fixture_workspace_copies_relative_dependencies_under_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = root / "source-fixtures"
            fixtures.mkdir()
            (fixtures / "kernel.cu").write_text("// source", encoding="utf-8")
            (fixtures / "reference.py").write_text("# reference", encoding="utf-8")
            (fixtures / "nested").mkdir()
            (fixtures / "nested" / "helper.py").write_text(
                "VALUE = 1", encoding="utf-8"
            )
            artifacts = root / "artifacts"

            workspace = _stage_fixture_workspace(
                "cuda_case", artifacts=artifacts, fixtures=fixtures
            )
            evidence = workspace / "compiler_evidence" / "manifest.json"
            evidence.parent.mkdir()
            evidence.write_text("{}", encoding="utf-8")

            self.assertEqual(
                workspace, artifacts.resolve() / "cuda_case" / "workspace"
            )
            self.assertEqual((workspace / "kernel.cu").read_text("utf-8"), "// source")
            self.assertTrue((workspace / "nested" / "helper.py").is_file())
            self.assertTrue(evidence.is_file())
            self.assertFalse((fixtures / "compiler_evidence").exists())

    def test_run_never_reuses_a_stale_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            output.write_text('{"stale":true}', encoding="utf-8")

            with self.assertRaisesRegex(AssertionError, "fresh|missing"):
                _run([sys.executable, "-c", "pass"], output)

            self.assertFalse(output.exists())

    def test_run_rejects_symlink_output_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outside.json"
            output = root / "result.json"
            target.write_text('{"outside":true}', encoding="utf-8")
            try:
                output.symlink_to(target)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(AssertionError, "symlink"):
                _run([sys.executable, "-c", "pass"], output)

            self.assertEqual(target.read_text("utf-8"), '{"outside":true}')

    def test_runner_contract_is_fail_closed_and_uses_immutable_image(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")

        self.assertNotIn("|| true", source)
        self.assertGreaterEqual(source.count("assert_gpu_idle"), 3)
        self.assertIn("resolved_image_id", source)
        self.assertIn("requested_ref", source)
        self.assertIn('"$resolved_image_id"', source)
        self.assertIn("must be empty", source)

    def test_authorized_ncu_runner_is_explicit_ephemeral_and_does_not_change_host_policy(self) -> None:
        source = NCU_AUTHORIZED_RUNNER.read_text(encoding="utf-8")

        self.assertIn("CUDA_E2E_ALLOW_SYS_ADMIN", source)
        self.assertIn("--cap-drop ALL", source)
        self.assertIn("--cap-add SYS_ADMIN", source)
        self.assertIn("--rm", source)
        self.assertIn("--network none", source)
        self.assertIn("resolved_image_id", source)
        self.assertGreaterEqual(source.count("assert_gpu_idle"), 3)
        self.assertNotIn("--privileged", source)
        self.assertNotIn("sysctl", source)
        self.assertNotIn("RmProfilingAdminOnly", source)

    def test_runner_restricts_cutlass_to_the_dedicated_checkout(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("CUDA_E2E_ROOT", source)
        self.assertIn("CUTLASS_PATH", source)
        self.assertNotIn("/data/tcheng", source)
        self.assertIn("include/cutlass/cutlass.h", source)
        self.assertIn("include/cutlass/version.h", source)
        self.assertIn('expected_cutlass_version="4.6.1"', source)
        self.assertIn("vllm-opt", source)
        self.assertIn('-v "$repo_root:$repo_root:ro"', source)

    def test_runner_rejects_a_writable_artifact_mount_inside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repository = root / "artifacts" / "repo"
            runner = repository / "tests" / "gpu" / "sm120" / "remote" / "run_lane.sh"
            runner.parent.mkdir(parents=True)
            shutil.copy2(RUNNER, runner)
            cutlass = root / "cutlass"
            include = cutlass / "include" / "cutlass"
            include.mkdir(parents=True)
            (include / "cutlass.h").write_text("// fixture\n", encoding="utf-8")
            (include / "version.h").write_text(
                "#define CUTLASS_MAJOR 4\n"
                "#define CUTLASS_MINOR 6\n"
                "#define CUTLASS_PATCH 1\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_E2E_ROOT": str(root),
                    "CUDA_E2E_ARTIFACTS": str(repository / "writable"),
                    "CUTLASS_PATH": str(cutlass),
                }
            )

            completed = subprocess.run(
                [str(runner), "compat"],
                text=True,
                capture_output=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertIn("must not overlap", completed.stderr)


if __name__ == "__main__":
    unittest.main()
