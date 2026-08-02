from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PAGES = (
    "docs/index.md",
    "docs/getting-started.md",
    "docs/environment-readiness.md",
    "docs/workflows.md",
    "docs/long-running-optimization.md",
    "docs/evidence-and-safety.md",
    "docs/compatibility.md",
    "docs/validation.md",
    "docs/case-studies.md",
    "docs/knowledge-and-research.md",
    "docs/project-evolution.md",
    "docs/project-evolution.en.md",
    "docs/evolution-case-profiler-evidence-validation.md",
)


class PublicDocsTests(unittest.TestCase):
    def test_public_navigation_and_relative_links(self) -> None:
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        self.assertNotIn("blob/main", config)
        self.assertNotIn("Agent Protocol:", config)
        for page in (Path(item).name for item in PUBLIC_PAGES):
            self.assertIn(page, config)
        pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for relative in PUBLIC_PAGES:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            for target in pattern.findall(path.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                resolved = (path.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(resolved.exists(), f"{path}: {target}")

    def test_public_docs_use_only_the_v14_model(self) -> None:
        prose = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in PUBLIC_PAGES
        ).lower()
        for legacy in (
            "controller",
            "orchestrator",
            "planner",
            "checkpoint",
            "append-only ledger",
            "active_diagnosis",
            "direction admission",
            "run-level grant",
            "v2.5",
            "v3.1",
        ):
            self.assertNotIn(legacy, prose)
        for current in (
            "target",
            "variant",
            "experiment",
            "invocation",
            "champion",
            "chatgpt",
        ):
            self.assertIn(current, prose)

    def test_getting_started_and_readiness_define_user_inputs(self) -> None:
        started = (ROOT / "docs/getting-started.md").read_text(encoding="utf-8")
        readiness = (ROOT / "docs/environment-readiness.md").read_text(encoding="utf-8")
        for marker in (
            "test workload",
            "correctness checks",
            "stable benchmark",
            "target GPU",
            "minimum useful effect",
            "allowed modification scope",
            "backup outside the active skills directory",
        ):
            self.assertIn(marker, started)
        self.assertIn("must be supplied by the user", started)
        for marker in (
            "readiness.py check",
            "target.json",
            "original Variant",
            "command driver",
            "Host changes",
            "self-check",
        ):
            self.assertIn(marker, readiness)

    def test_workflow_has_one_model_and_explicit_operations(self) -> None:
        workflow = (ROOT / "docs/workflows.md").read_text(encoding="utf-8")
        for marker in (
            "one model-led workflow",
            "Target",
            "Variant",
            "Experiment",
            "Invocation",
            "Champion Selection",
            "Handoff",
            "check",
            "baseline",
            "screen",
            "target",
            "select",
            "final_audit",
        ):
            self.assertIn(marker, workflow)
        self.assertIn("do not store the current optimization stage", workflow)

    def test_long_run_page_explains_dynamic_investment_without_frequent_questions(self) -> None:
        text = (ROOT / "docs/long-running-optimization.md").read_text(encoding="utf-8")
        for marker in (
            "authorization boundary, not a target to consume",
            "at most one planned authorization question",
            "Unattended runs ask none",
            "removable-time ceiling",
            "operation timeouts",
            "terminal result",
            "heartbeat",
        ):
            self.assertIn(marker, text)

    def test_evidence_knowledge_and_validation_boundaries_are_explicit(self) -> None:
        evidence = (ROOT / "docs/evidence-and-safety.md").read_text(encoding="utf-8")
        knowledge = (ROOT / "docs/knowledge-and-research.md").read_text(encoding="utf-8")
        validation = (ROOT / "docs/validation.md").read_text(encoding="utf-8")
        for marker in ("Correctness before performance", "Paired measurement", "fail closed", "ERR_NVGPUCTRPERM"):
            self.assertIn(marker, evidence)
        for marker in ("empty result", "does not block", "digest-bound path", "External availability is optional"):
            self.assertIn(marker, knowledge)
        self.assertIn("exact 17-module production surface", validation)
        self.assertIn("does not embed a live test count", validation)
        self.assertIn("CUDA_V14_HANDOFF_ROOT", validation)
        self.assertNotIn("/data/triton-handoff", validation)
        self.assertIn("recomputed statistics remained inconclusive", validation)
        self.assertIn("historical `REJECT`, `REJECT`, and `STOP`", validation)

    def test_case_studies_do_not_publish_an_unreviewed_positive_example(self) -> None:
        text = (ROOT / "docs/case-studies.md").read_text(encoding="utf-8")
        self.assertIn("complete evidence", text)
        self.assertIn("not as a positive public performance case", text)
        self.assertNotRegex(text, r"\d+(?:\.\d+)?%")

    def test_project_evolution_routes_language_and_public_case(self) -> None:
        text = (ROOT / "docs/project-evolution.md").read_text(encoding="utf-8")
        english = (ROOT / "docs/project-evolution.en.md").read_text(encoding="utf-8")
        self.assertIn("project-evolution.en.md", text)
        self.assertIn("evolution-case-profiler-evidence-validation.md", text)
        for page in (text, english):
            self.assertNotIn("../.github", page)
            self.assertIn(
                "https://github.com/troycheng/cuda-kernel-optimizer/tree/main/.github/evolution",
                page,
            )

    def test_internal_history_is_not_public_documentation(self) -> None:
        self.assertFalse((ROOT / "maintainers").exists())
        self.assertFalse((ROOT / "docs" / "superpowers").exists())

    def test_docs_index_routes_to_the_installed_protocol_not_moving_main(self) -> None:
        index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
        self.assertNotIn("blob/main", index)
        self.assertIn("skills/cuda-kernel-optimizer/SKILL.md", index)


if __name__ == "__main__":
    unittest.main()
