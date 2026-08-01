from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "github.com/troycheng/cuda-kernel-optimizer"


class StandaloneProjectTests(unittest.TestCase):
    def test_public_version_and_release_notes_are_v1_4(self) -> None:
        self.assertEqual((ROOT / "VERSION").read_text("utf-8").strip(), "1.4.0")
        for name in ("README.md", "README.en.md"):
            text = (ROOT / name).read_text("utf-8")
            self.assertEqual(text.count("### V1.4.0"), 1)
            self.assertNotIn("开发中", text)
            self.assertNotIn("in development", text.lower())

    def test_readmes_install_from_the_standalone_repository(self) -> None:
        for name in ("README.md", "README.en.md"):
            self.assertIn(REPOSITORY, (ROOT / name).read_text("utf-8"))

    def test_origin_notice_preserves_provenance(self) -> None:
        notice = (ROOT / "NOTICE").read_text("utf-8")
        for marker in (
            "KernelFlow-ops/cuda-optimized-skill",
            "git.yukework.com/mlsys/cuda-optimized-skill",
            "github.com/troycheng/cuda-optimized-skill",
            "Acknowledgements",
            "Mark Liu",
            "MIT",
        ):
            self.assertIn(marker, notice)

    def test_installed_skill_carries_license_and_notice(self) -> None:
        skill = ROOT / "skills" / "cuda-kernel-optimizer"
        for name in ("LICENSE", "NOTICE"):
            self.assertEqual(
                (skill / name).read_text("utf-8"),
                (ROOT / name).read_text("utf-8"),
            )

    def test_public_files_do_not_expose_private_storage_paths(self) -> None:
        private_path = "/data/" + "tcheng"
        paths = [ROOT / name for name in ("README.md", "README.en.md", "CONTRIBUTING.md")]
        paths.extend((ROOT / "docs").rglob("*.md"))
        paths.extend((ROOT / "skills" / "cuda-kernel-optimizer").rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text("utf-8")
            except UnicodeDecodeError:
                continue
            self.assertNotIn(private_path, text, str(path))

    def test_templates_use_no_archived_repository_schema_identity(self) -> None:
        old_prefix = "https://github.com/troycheng/cuda-optimized-skill/"
        for path in (ROOT / "skills/cuda-kernel-optimizer/templates").glob("*.json"):
            value = json.loads(path.read_text("utf-8"))
            self.assertFalse(str(value.get("$id", "")).startswith(old_prefix), path.name)

    def test_public_tree_excludes_old_maintenance_tools(self) -> None:
        for path in (
            ROOT / "maintainers",
            ROOT / "tools" / "publish_dual_remote.py",
            ROOT / "tools" / "run_skill_eval.py",
            ROOT / "tests" / "evals",
        ):
            self.assertFalse(path.exists(), str(path))

    def test_community_files_are_present(self) -> None:
        for relative in (
            "CONTRIBUTING.md",
            "SECURITY.md",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            ".github/pull_request_template.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        self.assertIn(
            "python3 -m unittest discover -s tests",
            (ROOT / "CONTRIBUTING.md").read_text("utf-8"),
        )

    def test_ci_runs_current_release_gate_on_supported_python(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
        for marker in (
            '"3.10"',
            '"3.12"',
            "python -m unittest discover -s tests -p 'test_*.py'",
            "python -m compileall -q skills/cuda-kernel-optimizer/scripts tests",
            "python skills/cuda-kernel-optimizer/scripts/self_check.py",
            "timeout-minutes:",
            "contents: read",
        ):
            self.assertIn(marker, workflow)
        self.assertNotIn("skills/cuda-kernel-optimizer/tests", workflow)
        self.assertNotIn("fetch-depth: 0", workflow)


if __name__ == "__main__":
    unittest.main()
