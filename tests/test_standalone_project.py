from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "github.com/troycheng/cuda-kernel-optimizer"


class StandaloneProjectTests(unittest.TestCase):
    def test_public_version_and_release_notes_match_current_release(self) -> None:
        self.assertEqual((ROOT / "VERSION").read_text("utf-8").strip(), "1.6.1")
        for name in ("README.md", "README.en.md"):
            text = (ROOT / name).read_text("utf-8")
            self.assertEqual(text.count("### V1.6.1"), 1)
            self.assertEqual(text.count("### V1.5.0"), 1)
            self.assertEqual(text.count("### V1.4.2"), 1)
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

    def test_public_files_do_not_expose_private_storage_or_internal_hosts(self) -> None:
        private_markers = ("/data/" + "tcheng", "git." + "yukework.com")
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
            for marker in private_markers:
                self.assertNotIn(marker, text, str(path))

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
            ".github/evolution/case-snapshot.md",
            ".github/evolution/evaluation-definition.md",
            ".github/evolution/evaluation-result.md",
            ".github/evolution/release-decision.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        contributing = (ROOT / "CONTRIBUTING.md").read_text("utf-8")
        self.assertIn(
            "python3 -m unittest discover -s tests",
            contributing,
        )
        self.assertIn("Project evolution", contributing)
        pull_request_template = (ROOT / ".github/pull_request_template.md").read_text(
            "utf-8"
        )
        self.assertIn("Staged installation self-check passes", pull_request_template)
        self.assertIn("private material", pull_request_template.lower())
        self.assertIn("Evaluation Definition", pull_request_template)
        self.assertNotIn("Installed-skill tests pass", pull_request_template)

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
