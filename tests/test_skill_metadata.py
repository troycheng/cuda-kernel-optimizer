from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "cuda-kernel-optimizer"
SKILL_MD = SKILL_DIR / "SKILL.md"

_PUBLIC_TOOLS = {
    "readiness.py",
    "workload_evaluate.py",
    "profile_ncu.py",
    "profile_nsys.py",
    "profile_pytorch.py",
    "compiler_evidence.py",
    "sass_check.py",
    "knowledge_query.py",
    "champion.py",
}
_LEGACY_CONTROL_WORDS = (
    r"\bcontroller\b",
    r"\borchestrate\w*\b",
    r"\bbudget\b",
    r"\bstate\b",
    r"decision\.json",
)


class SkillMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = SKILL_MD.read_text(encoding="utf-8")
        self.prose = " ".join(self.text.split())
        self.lower = self.prose.lower()

    def test_frontmatter_is_portable_and_trigger_focused(self) -> None:
        self.assertTrue(self.text.startswith("---\n"))
        frontmatter, _ = self.text[4:].split("\n---\n", 1)
        lines = [line for line in frontmatter.splitlines() if line.strip()]
        self.assertEqual([line.split(":", 1)[0] for line in lines], ["name", "description"])
        self.assertRegex(lines[0].split(":", 1)[1].strip(), r"^[a-z0-9-]+$")
        description = json.loads(lines[1].split(":", 1)[1].strip())
        self.assertTrue(description.startswith("Use when "))
        self.assertLessEqual(len(description), 1024)
        for trigger in ("CUDA", "CUTLASS", "Triton", "PyTorch", "vLLM", "NCU"):
            self.assertIn(trigger, description)

    def test_skill_is_a_compact_v14_router(self) -> None:
        self.assertLessEqual(len(self.text.split()), 1800)
        self.assertIn("## 按需读取", self.text)
        self.assertIn("ChatGPT", self.text)

    def test_chatgpt_is_the_only_optimization_decision_maker(self) -> None:
        self.assertIn("ChatGPT 负责优化判断", self.prose)
        for responsibility in ("瓶颈", "候选", "投入产出", "下一步"):
            self.assertIn(responsibility, self.prose)
        self.assertIn("工具不选择方向，不判断 ROI，也不生成下一步计划", self.prose)

    def test_router_mentions_only_v14_public_tools(self) -> None:
        routed = set(re.findall(r"(?:<skill>/)?scripts/([a-z_]+\.py)", self.text))
        self.assertEqual(routed, _PUBLIC_TOOLS)

    def test_original_tests_and_precision_validation_precede_measurement(self) -> None:
        self.assertIn("修改代码前先测原始业务基线", self.prose)
        self.assertIn("精度未通过时，不解释性能样本", self.prose)

    def test_no_legacy_control_plane_terms_remain(self) -> None:
        for pattern in _LEGACY_CONTROL_WORDS:
            self.assertNotRegex(self.lower, pattern)

    def test_profiler_and_knowledge_only_return_facts_and_empty_knowledge_does_not_block(self) -> None:
        self.assertIn("profiler 只返回观测事实", self.prose)
        self.assertIn("它不返回优化方向、ROI 或下一步", self.prose)
        self.assertIn("知识查询只返回匹配材料或空结果", self.prose)
        self.assertIn("空结果不会阻止源码分析", self.prose)

    def test_external_search_and_ai_are_optional_and_local_evidence_decides(self) -> None:
        self.assertIn("外部搜索和第三方 AI 质证是可选的研究手段", self.prose)
        self.assertIn("本地精度与测量证据决定", self.prose)

    def test_every_referenced_skill_path_exists(self) -> None:
        paths = set(
            re.findall(
                r"(?:<skill>/)?((?:scripts|references|examples)/[^`\s)]+\.(?:py|md|json))",
                self.text,
            )
        )
        self.assertGreaterEqual(len(paths), len(_PUBLIC_TOOLS))
        for relative in paths:
            self.assertTrue((SKILL_DIR / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
