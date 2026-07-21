from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEW_REPOSITORY = "github.com/troycheng/cuda-kernel-optimizer"
OLD_PUBLIC_REPOSITORY = "github.com/troycheng/cuda-optimized-skill"


class StandaloneProjectTests(unittest.TestCase):
    def test_public_version_starts_at_v1(self) -> None:
        self.assertEqual((ROOT / "VERSION").read_text("utf-8").strip(), "1.0.0")
        for name, heading in (
            ("README.md", "### V1.0.0"),
            ("README.zh-CN.md", "### V1.0.0"),
        ):
            text = (ROOT / name).read_text("utf-8")
            self.assertIn(heading, text)
            self.assertNotRegex(text, r"(?m)^### V(?:2|3)\.")

    def test_readmes_install_from_the_standalone_repository(self) -> None:
        for name in ("README.md", "README.zh-CN.md"):
            text = (ROOT / name).read_text("utf-8")
            self.assertIn(NEW_REPOSITORY, text)
            self.assertNotIn(OLD_PUBLIC_REPOSITORY, text)

    def test_origin_notice_preserves_provenance(self) -> None:
        notice = (ROOT / "NOTICE").read_text("utf-8")
        self.assertIn("KernelFlow-ops/cuda-optimized-skill", notice)
        self.assertIn("git.yukework.com/mlsys/cuda-optimized-skill", notice)
        self.assertIn("github.com/troycheng/cuda-optimized-skill", notice)
        self.assertIn("MIT", notice)
        self.assertIn("Acknowledgements", notice)
        self.assertIn("https://github.com/KernelFlow-ops", notice)
        self.assertIn("Mark Liu", notice)
        self.assertIn("https://github.com/mark-liu", notice)

    def test_public_tree_excludes_maintainer_history_and_dual_publisher(self) -> None:
        self.assertFalse((ROOT / "maintainers").exists())
        self.assertFalse((ROOT / "tools" / "publish_dual_remote.py").exists())
        self.assertFalse((ROOT / "tests" / "test_publish_dual_remote.py").exists())

    def test_community_files_are_present(self) -> None:
        for relative in (
            "CONTRIBUTING.md",
            "SECURITY.md",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            ".github/pull_request_template.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        contributing = (ROOT / "CONTRIBUTING.md").read_text("utf-8")
        self.assertIn("python3 -m unittest discover -s tests", contributing)
        security = (ROOT / "SECURITY.md").read_text("utf-8")
        self.assertIn("private vulnerability reporting", security.lower())

    def test_ci_runs_static_suite_on_supported_python_versions(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
        for marker in (
            '"3.10"',
            '"3.12"',
            "python -m unittest discover -s tests -p 'test_*.py'",
            "python -m unittest discover -s skills/cuda-kernel-optimizer/tests",
            "python -m compileall -q skills/cuda-kernel-optimizer/scripts tests",
            "python skills/cuda-kernel-optimizer/scripts/self_check.py",
            "timeout-minutes:",
            "contents: read",
        ):
            self.assertIn(marker, workflow)

    def test_validation_count_matches_the_release_gate(self) -> None:
        validation = (ROOT / "docs" / "validation.md").read_text("utf-8")
        self.assertIn("1,111 tests", validation)
        self.assertIn("1,102 passed", validation)
        self.assertIn("nine physical RTX 5090 opt-in tests were skipped", validation)


if __name__ == "__main__":
    unittest.main()
