from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEW_REPOSITORY = "github.com/troycheng/cuda-kernel-optimizer"
OLD_PUBLIC_REPOSITORY = "github.com/troycheng/cuda-optimized-skill"


class StandaloneProjectTests(unittest.TestCase):
    def test_public_version_uses_the_v1_release_line(self) -> None:
        self.assertEqual((ROOT / "VERSION").read_text("utf-8").strip(), "1.1.0")
        for name in ("README.md", "README.zh-CN.md"):
            text = (ROOT / name).read_text("utf-8")
            self.assertIn("### V1.1.0", text)
            self.assertIn("### V1.0.1", text)
            self.assertIn("### V1.0.0", text)
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

    def test_installed_skill_carries_license_and_notice(self) -> None:
        skill = ROOT / "skills" / "cuda-kernel-optimizer"
        for name in ("LICENSE", "NOTICE"):
            distributed = skill / name
            self.assertTrue(distributed.is_file(), name)
            self.assertEqual(
                distributed.read_text("utf-8"),
                (ROOT / name).read_text("utf-8"),
            )

    def test_public_files_do_not_expose_maintainer_storage_paths(self) -> None:
        paths = (
            ROOT / "skills" / "cuda-kernel-optimizer" / "references" / "compatibility.md",
            ROOT / "tests" / "gpu" / "sm120" / "README.md",
            ROOT / "tests" / "gpu" / "sm120" / "remote" / "run_lane.sh",
        )
        for path in paths:
            self.assertNotIn("/data/tcheng", path.read_text("utf-8"), str(path))

    def test_schema_identity_policy_keeps_only_versioned_legacy_ids(self) -> None:
        old_prefix = "https://github.com/troycheng/cuda-optimized-skill/"
        versioned_prefixes = (old_prefix + "schema/v", old_prefix + "schemas/v")
        legacy_ids = []
        for path in sorted((ROOT / "skills/cuda-kernel-optimizer/templates").glob("*.schema.json")):
            schema_id = json.loads(path.read_text("utf-8")).get("$id", "")
            if schema_id.startswith(old_prefix):
                self.assertTrue(schema_id.startswith(versioned_prefixes), path.name)
                legacy_ids.append(schema_id)
        self.assertTrue(legacy_ids)
        compatibility = (ROOT / "docs" / "compatibility.md").read_text("utf-8")
        self.assertIn("Schema identities", compatibility)
        self.assertIn("versioned pre-V1", compatibility)

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
        suite = unittest.defaultTestLoader.discover(
            str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT)
        )

        def iter_cases(node):
            for item in node:
                if isinstance(item, unittest.TestSuite):
                    yield from iter_cases(item)
                else:
                    yield item

        cases = list(iter_cases(suite))
        skipped = sum(
            bool(getattr(case.__class__, "__unittest_skip__", False))
            or bool(
                getattr(
                    getattr(case, case._testMethodName), "__unittest_skip__", False
                )
            )
            for case in cases
        )
        passed = len(cases) - skipped
        self.assertIn(f"{len(cases):,} tests", validation)
        self.assertIn(f"{passed:,} passed", validation)
        self.assertIn(
            f"{skipped:,} physical RTX 5090 opt-in tests were skipped", validation
        )


if __name__ == "__main__":
    unittest.main()
