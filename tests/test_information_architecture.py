from __future__ import annotations

import ast
import json
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"
FINAL_REFERENCES = {
    "compatibility.md",
    "environment_readiness.md",
    "knowledge",
    "ncu_metrics_guide.md",
    "nonstationary_serving_evidence.md",
    "optimizer_limits.md",
    "performance_iteration.md",
    "research_augmentation.md",
    "sass_signatures.json",
    "serving_evidence_protocol.md",
    "systems_and_ir_coverage.md",
    "version_stack_audit.md",
}


class InformationArchitectureTests(unittest.TestCase):
    def test_installed_skill_has_one_final_reference_and_example_surface(self) -> None:
        references = SKILL / "references"
        self.assertEqual({path.name for path in references.iterdir()}, FINAL_REFERENCES)
        self.assertEqual(
            {path.name for path in (SKILL / "examples").iterdir()},
            {"walkthrough.md"},
        )
        installed_prose = [SKILL / "SKILL.md", SKILL / "examples" / "walkthrough.md"]
        installed_prose.extend(
            path for path in references.iterdir() if path.is_file() and path.suffix == ".md"
        )
        for path in installed_prose:
            text = path.read_text(encoding="utf-8").lower()
            for legacy in (
                "controller",
                "orchestrator",
                "checkpoint",
                "direction admission",
                "evidence automation",
            ):
                self.assertNotIn(legacy, text, path)

    def test_current_installation_self_check_passes(self) -> None:
        checker_path = SKILL / "scripts" / "self_check.py"
        spec = importlib.util.spec_from_file_location("v14_current_self_check", checker_path)
        checker = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(checker)
        result = checker.check_installation(SKILL)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["gpu_checks_run"])
        self.assertFalse(result["network_checks_run"])

    def test_v14_production_surface_is_exact_and_smaller_than_task5_baseline(self) -> None:
        checker_path = SKILL / "scripts" / "self_check.py"
        spec = importlib.util.spec_from_file_location("v14_surface_check", checker_path)
        checker = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(checker)

        scripts = SKILL / "scripts"
        actual = {
            path.name
            for path in scripts.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix == ".py"
        }
        self.assertEqual(actual, set(checker.PRODUCTION_MODULES))
        self.assertEqual(len(actual), 17)
        line_count = sum(
            (scripts / name).read_text(encoding="utf-8").count("\n")
            for name in checker.PRODUCTION_MODULES
        )
        self.assertLess(line_count, 14_671)

        store = (scripts / "artifact_store.py").read_text(encoding="utf-8")
        for legacy_symbol in (
            "read_regular_with_optional_sibling",
            "read_regular_bundle",
            "publish_regular_bundle",
            "atomic_write_json",
            "atomic_write_jsonl",
            "write_paired_samples",
            "class ArtifactStore",
        ):
            self.assertNotIn(legacy_symbol, store)

        for name in checker.PRODUCTION_MODULES:
            if name == "_invocation_runtime.py":
                continue
            tree = ast.parse(
                (scripts / name).read_text(encoding="utf-8"),
                filename=name,
            )
            imported = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported.update(
                node.module.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertNotIn("subprocess", imported, name)

    def test_workload_evaluator_does_not_import_the_champion_tool(self) -> None:
        evaluator = (
            SKILL / "scripts" / "workload_evaluate.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('_load_sibling("champion.py"', evaluator)
        self.assertNotIn('"champion.py",', evaluator)

    def test_self_check_rejects_an_illegal_internal_dependency(self) -> None:
        checker_path = SKILL / "scripts" / "self_check.py"
        spec = importlib.util.spec_from_file_location("v14_dependency_check", checker_path)
        checker = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(checker)

        with tempfile.TemporaryDirectory() as directory:
            scripts = Path(directory)
            for name in checker.PRODUCTION_MODULES:
                (scripts / name).write_text("pass\n", encoding="utf-8")
            (scripts / "artifact_store.py").write_text(
                "import paired_stats\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "unexpected production dependency: artifact_store -> paired_stats",
            ):
                checker._validate_dependency_graph(scripts)

    def test_v14_self_check_rejects_an_extra_production_script(self) -> None:
        checker_path = SKILL / "scripts" / "self_check.py"
        spec = importlib.util.spec_from_file_location("v14_self_check", checker_path)
        checker = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "skill"
            scripts = fixture / "scripts"
            templates = fixture / "templates"
            agents = fixture / "agents"
            examples = fixture / "examples"
            references = fixture / "references"
            knowledge = fixture / "references" / "knowledge"
            playbooks = knowledge / "playbooks"
            scripts.mkdir(parents=True)
            templates.mkdir()
            agents.mkdir()
            examples.mkdir()
            playbooks.mkdir(parents=True)
            for name in checker.ROOT_FILES:
                (fixture / name).write_text("fixture\n", encoding="utf-8")
            for name in checker.AGENT_FILES:
                (agents / name).write_text("fixture\n", encoding="utf-8")
            for name in checker.EXAMPLE_FILES:
                (examples / name).write_text("fixture\n", encoding="utf-8")
            for name in checker.REFERENCE_FILES:
                (references / name).write_text("fixture\n", encoding="utf-8")
            for name in checker.PRODUCTION_MODULES:
                source = (
                    "from pathlib import Path\n"
                    "def _lock_root():\n"
                    "    root = Path.home() / '.cache' / 'self-check-fixture-locks'\n"
                    "    root.mkdir(parents=True, exist_ok=True)\n"
                    "    return root\n"
                    if name == "_invocation_runtime.py"
                    else "pass\n"
                )
                (scripts / name).write_text(source, encoding="utf-8")
            for name in checker.DRIVER_TEMPLATES:
                (templates / name).write_text("{}\n", encoding="utf-8")
            (knowledge / "sources.json").write_text(
                '{"sources":[{"id":"source-a","status":"verified",'
                '"summary_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}\n',
                encoding="utf-8",
            )
            (knowledge / "cards.json").write_text(
                '{"cards":[{"id":"card-a","source_ids":["source-a"],'
                '"playbook":"playbooks/a.md",'
                '"playbook_sha256":"87ae67dd711f2cf45b790fe2515e80dcf3d301265c6c016bd580d04aeba0fa52"}]}\n',
                encoding="utf-8",
            )
            (playbooks / "a.md").write_text("# playbook\n", encoding="utf-8")
            (scripts / "unexpected.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected production scripts"):
                checker.check_installation(fixture)

    def test_self_check_rejects_symlinked_knowledge_manifests(self) -> None:
        checker_path = SKILL / "scripts" / "self_check.py"
        spec = importlib.util.spec_from_file_location("v14_knowledge_check", checker_path)
        checker = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "references" / "knowledge"
            playbooks = knowledge / "playbooks"
            playbooks.mkdir(parents=True)
            playbook = playbooks / "a.md"
            playbook.write_text("# playbook\n", encoding="utf-8")
            (knowledge / "sources.json").write_text(
                '{"sources":[{"id":"source-a","status":"verified",'
                '"summary_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}\n',
                encoding="utf-8",
            )
            outside = root / "outside-cards.json"
            outside.write_text(
                '{"cards":[{"source_ids":["source-a"],'
                '"playbook":"playbooks/a.md",'
                '"playbook_sha256":"87ae67dd711f2cf45b790fe2515e80dcf3d301265c6c016bd580d04aeba0fa52"}]}\n',
                encoding="utf-8",
            )
            try:
                (knowledge / "cards.json").symlink_to(outside)
            except OSError:
                self.skipTest("symlinks are unavailable")

            with self.assertRaisesRegex(ValueError, "unsafe knowledge manifest"):
                checker._validate_knowledge(root)

    def test_docs_are_user_facing_and_history_is_not_shipped(self) -> None:
        self.assertFalse((ROOT / "docs" / "superpowers").exists())
        self.assertFalse((ROOT / "maintainers").exists())
        for name in (
            "environment-readiness.md",
            "validation.md",
            "case-studies.md",
            "knowledge-and-research.md",
        ):
            self.assertTrue((ROOT / "docs" / name).is_file(), name)

    def test_readmes_route_validation_and_case_studies_separately(self) -> None:
        english = (ROOT / "README.en.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("## Tested scope", english)
        self.assertNotIn("## 已测试范围", chinese)
        self.assertIn("## Results and acceptance", english)
        self.assertIn("## 结果与验收", chinese)
        self.assertNotIn("## 验证情况", chinese)
        for text in (english, chinese):
            self.assertIn("docs/validation.md", text)
            self.assertIn("docs/case-studies.md", text)

    def test_skill_is_a_small_router_with_on_demand_routes(self) -> None:
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 240)
        self.assertLessEqual(len(text.split()), 1800)
        for marker in (
            "scripts/readiness.py",
            "scripts/workload_evaluate.py",
            "scripts/profile_ncu.py",
            "scripts/profile_nsys.py",
            "scripts/profile_pytorch.py",
            "scripts/knowledge_query.py",
            "references/environment_readiness.md",
            "references/research_augmentation.md",
        ):
            self.assertIn(marker, text)

    def test_user_inputs_use_plain_consistent_terms(self) -> None:
        paths = (
            ROOT / "README.en.md",
            ROOT / "docs" / "getting-started.md",
            ROOT / "docs" / "environment-readiness.md",
            ROOT / "docs" / "workflows.md",
            ROOT / "docs" / "long-running-optimization.md",
            SKILL / "SKILL.md",
            SKILL / "references" / "environment_readiness.md",
        )
        texts = {
            path: " ".join(path.read_text(encoding="utf-8").split())
            for path in paths
        }
        for path, text in texts.items():
            self.assertNotIn("correctness reference", text.lower(), path)

        readme = texts[ROOT / "README.en.md"]
        self.assertIn(
            "Test workload (dataset, representative requests, or replay)",
            readme,
        )
        self.assertIn(
            "Correctness checks (expected outputs, tolerances, or accuracy criteria)",
            readme,
        )

    def test_offline_knowledge_has_freshness_and_primary_sources(self) -> None:
        manifest = json.loads(
            (SKILL / "references" / "knowledge" / "sources.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertRegex(manifest["as_of"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertIn("staleness_policy", manifest)
        sources = manifest["sources"]
        self.assertGreaterEqual(len(sources), 8)
        for source in sources:
            self.assertIn(
                source["source_kind"],
                {
                    "primary",
                    "official-documentation",
                    "official-api-documentation",
                    "reference-implementation",
                },
            )
            self.assertTrue(source["url"].startswith("https://"))
            self.assertIn("last_verified", source)

    def test_generated_python_artifacts_are_not_part_of_the_skill(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "*.pyc", "**/__pycache__/**"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
        self.assertEqual(tracked, [])


if __name__ == "__main__":
    unittest.main()
