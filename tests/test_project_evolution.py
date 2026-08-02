from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "case-snapshot.md",
    "evaluation-definition.md",
    "evaluation-result.md",
    "release-decision.md",
}


class ProjectEvolutionTests(unittest.TestCase):
    def test_public_guides_keep_the_six_concepts_and_v14_boundary(self) -> None:
        chinese = (ROOT / "docs/project-evolution.md").read_text("utf-8")
        english = (ROOT / "docs/project-evolution.en.md").read_text("utf-8")
        for marker in (
            "演进契约",
            "案例快照",
            "项目版本",
            "评测定义",
            "评测结果",
            "发布决定",
            "ChatGPT",
            "确定性工具",
            "不修改 V1.4 运行时",
        ):
            self.assertIn(marker, chinese)
        for marker in (
            "Evolution Contract",
            "Case Snapshot",
            "Project Revision",
            "Evaluation Definition",
            "Evaluation Result",
            "Release Decision",
            "ChatGPT",
            "deterministic tools",
            "does not modify the V1.4 runtime",
        ):
            self.assertIn(marker, english)
        self.assertIn("不自动合并", chinese)
        self.assertIn("does not automatically merge", english.lower())

    def test_templates_are_small_markdown_records_not_a_new_schema(self) -> None:
        root = ROOT / ".github/evolution"
        self.assertEqual({path.name for path in root.iterdir()}, TEMPLATES)
        joined = "\n".join(
            (root / name).read_text("utf-8") for name in sorted(TEMPLATES)
        )
        for marker in (
            "Private material",
            "Claim ceiling",
            "Repository revision",
            "Evaluator identity",
            "Actual result",
            "Human decision",
        ):
            self.assertIn(marker, joined)
        self.assertNotIn("next_action", joined)
        self.assertNotIn("current_state", joined)
        self.assertNotIn("knowledge_weight", joined)

    def test_public_case_is_narrow_replayable_and_not_a_performance_claim(self) -> None:
        text = (ROOT / "docs/evolution-case-profiler-evidence-validation.md").read_text(
            "utf-8"
        )
        for marker in (
            "5211e832b6d5055ed316fe6fc77efa57813f5934",
            "9a3ff596907fcab7dd9abf4615bb080a1a2c2222",
            "v1.4.0",
            "retrospective",
            "not a performance case",
            "test_candidate_collection_rejects_changed_evidence_payload",
        ):
            self.assertIn(marker, text)
        self.assertIsNone(re.search(r"\b\d+(?:\.\d+)?%\b", text))

    def test_public_materials_do_not_admit_private_evidence(self) -> None:
        paths = [
            ROOT / "docs/project-evolution.md",
            ROOT / "docs/project-evolution.en.md",
            ROOT / "docs/evolution-case-profiler-evidence-validation.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / ".github/pull_request_template.md",
        ]
        joined = "\n".join(path.read_text("utf-8") for path in paths)
        self.assertIn("private workload", joined.lower())
        self.assertIn("must still stand", joined.lower())
        self.assertIn("human", joined.lower())
        self.assertNotIn("private evidence queue", joined.lower())


if __name__ == "__main__":
    unittest.main()
