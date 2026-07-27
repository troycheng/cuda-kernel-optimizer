from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_ZH = ROOT / "README.md"
README_EN = ROOT / "README.en.md"
README_ZH_COMPAT = ROOT / "README.zh-CN.md"

MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
EDGE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)[^-\n]*?\s*(-->|-\.->)\s*"
    r"([a-z][a-z0-9_]*)",
    re.MULTILINE,
)


def assert_in_order(testcase, text: str, markers: tuple[str, ...]) -> None:
    positions = [text.index(marker) for marker in markers]
    testcase.assertEqual(positions, sorted(positions))


class ReadmeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chinese = README_ZH.read_text(encoding="utf-8")
        self.english = README_EN.read_text(encoding="utf-8")
        self.compat = README_ZH_COMPAT.read_text(encoding="utf-8")

    def test_language_roles_and_full_readme_sizes(self) -> None:
        self.assertIn("## 这是什么", self.chinese)
        self.assertIn("## About", self.english)
        self.assertNotIn("## About", self.chinese)
        self.assertNotIn("## 这是什么", self.english)
        self.assertLessEqual(len(self.chinese.splitlines()), 200)
        self.assertLessEqual(len(self.english.splitlines()), 240)
        self.assertLessEqual(len(self.compat.splitlines()), 8)

    def test_hero_explains_the_product_and_switches_language(self) -> None:
        for text in (self.chinese, self.english):
            opening = text[: text.index("\n## ")]
            self.assertIn("asset/logo-wordmark-dark.svg", opening)
            self.assertIn("asset/logo-wordmark.svg", opening)
            self.assertIn('width="520"', opening)
            for marker in ("ChatGPT", "CUDA", "CUTLASS", "Triton"):
                self.assertIn(marker, text[: text.index("\n## ", text.index("\n## ") + 1)])
        self.assertIn('href="README.en.md"', self.chinese)
        self.assertIn('href="README.md"', self.english)

    def test_chinese_primary_readme_uses_reader_first_order(self) -> None:
        assert_in_order(
            self,
            self.chinese,
            (
                "## 这是什么",
                "## 它能帮你完成什么",
                "## 正式性能结论通常需要什么",
                "## 十分钟判断是否适合",
                "## 正式优化会怎样进行",
                "## V1.2 如何控制投入",
                "## 你会得到什么",
                "## 结论能到什么程度",
                "## 安装",
                "## 安全边界",
                "## 验证情况",
                "## 版本记录",
                "## 文档",
            ),
        )
        self.assertLess(
            self.chinese.index("## 十分钟判断是否适合"),
            self.chinese.index("## V1.2 如何控制投入"),
        )

    def test_first_use_path_is_concrete_and_ai_executed(self) -> None:
        for marker in (
            "最多用 10 分钟",
            "不要修改源码",
            "不声称获得提速",
            "github.com/troycheng/cuda-kernel-optimizer",
            "安装标签、commit 和目标目录",
        ):
            self.assertIn(marker, self.chinese)
        english = " ".join(self.english.split())
        for marker in (
            "10-minute fit check",
            "Do not edit source files",
            "does not claim a speedup",
            "github.com/troycheng/cuda-kernel-optimizer",
            "installed tag, commit, and destination",
        ):
            self.assertIn(marker, english)

    def test_full_readmes_cover_the_same_supported_scenarios(self) -> None:
        for marker in (
            "环境",
            "CUDA",
            "CUTLASS",
            "Triton",
            "完整、可重复的 workload",
            "serving KPI",
            "已有 NCU report",
        ):
            self.assertIn(marker, self.chinese)
        for marker in (
            "Environment readiness",
            "Kernel optimization",
            "Complete workload",
            "Serving validation",
            "Existing NCU report",
        ):
            self.assertIn(marker, self.english)

    def test_readmes_show_their_public_workflow(self) -> None:
        chinese = MERMAID.findall(self.chinese)
        english = MERMAID.findall(self.english)
        self.assertEqual(len(chinese), 1)
        self.assertEqual(len(english), 1)
        chinese_edges = set(EDGE.findall(chinese[0]))
        for edge in (
            ("baseline", "-->", "brief"),
            ("brief", "-->", "grant"),
            ("change", "-->", "stages"),
            ("stages", "-->", "pause"),
            ("pause", "-->", "grant"),
        ):
            self.assertIn(edge, chinese_edges)
        self.assertIn(("evaluation", "-->", "keep"), set(EDGE.findall(english[0])))
        self.assertIn(("evaluation", "-->", "restore"), set(EDGE.findall(english[0])))

    def test_readmes_explain_v1_2_adaptive_investment(self) -> None:
        chinese = "".join(self.chinese.split()).replace("`", "")
        for marker in (
            "收益上限",
            "最低成本的验证方式",
            "运行级授权",
            "它是边界，不是必须消耗完的预算",
            "用户等待和暂停时间不会占用授权",
            "REVIEW_REQUIRED",
        ):
            self.assertIn(marker, chinese)
        english = " ".join(self.english.split()).replace("`", "")
        for marker in (
            "run-level grant",
            "scope, risk, stage",
            "waiting does not consume it",
            "pauses for review",
        ):
            self.assertIn(marker, english)

    def test_readmes_explain_candidate_stages_and_recovery(self) -> None:
        for marker in (
            "ChangeSet",
            "静态检查",
            "最低正确性",
            "短版成对测试",
            "不重复执行或扣费",
            "补充授权后继续",
            "明确放弃",
        ):
            self.assertIn(marker, self.chinese)
        for marker in (
            "ChangeSet",
            "static",
            "correctness",
            "short paired",
            "Completed stages survive restarts",
            "grant pause preserves",
            "explicit abandonment",
        ):
            self.assertIn(marker, self.english)

    def test_real_workload_and_claim_boundaries_are_explicit(self) -> None:
        for marker in (
            "真实 workload",
            "不会自行下载或编造",
            "不能作为性能提升结论",
            "局部 kernel 变快不等于完整 workload 变快",
            "正确性和成对性能数据",
        ):
            self.assertIn(marker, self.chinese)
        english = " ".join(self.english.split())
        for marker in (
            "A real workload must be supplied by the user",
            "does not download or invent one",
            "does not claim a speedup",
            "correctness",
            "paired",
        ):
            self.assertIn(marker, english)

    def test_readiness_and_host_boundaries_are_explicit(self) -> None:
        for marker in (
            "编译、正确性、benchmark、GPU、profiler",
            "不自动修改",
            "self_check",
            "ERR_NVGPUCTRPERM",
            "外部搜索和第三方 AI",
        ):
            self.assertIn(marker, self.chinese)
        for marker in (
            "Automatic pre-baseline readiness",
            "never changes host-level settings automatically",
            "self_check",
            "ERR_NVGPUCTRPERM",
            "External search and multi-model challenge",
        ):
            self.assertIn(marker, self.english)

    def test_primary_readme_names_the_durable_outputs(self) -> None:
        for marker in (
            "summary.md",
            "active_diagnosis/initial_investment_brief.json",
            "active_diagnosis/performance_model.json",
            "decision.json",
            "原始成对样本",
            "证据完整性",
        ):
            self.assertIn(marker, self.chinese)
        for marker in (
            "summary.md",
            "performance_model.json",
            "investment_brief.json",
            "decision.json",
        ):
            self.assertIn(marker, self.english)

    def test_readmes_publish_the_same_v1_release_line(self) -> None:
        for text in (self.chinese, self.english):
            self.assertEqual(text.count("### V1.2.0"), 1)
            self.assertEqual(text.count("### V1.1.0"), 1)
            self.assertEqual(text.count("### V1.0.1"), 1)
            self.assertEqual(text.count("### V1.0.0"), 1)
            self.assertNotRegex(text, r"(?m)^### V(?:2|3)\.")

    def test_validation_and_case_studies_are_separate(self) -> None:
        for text in (self.chinese, self.english):
            self.assertIn("docs/validation.md", text)
            self.assertIn("docs/case-studies.md", text)
        validation = (ROOT / "docs/validation.md").read_text(encoding="utf-8")
        cases = (ROOT / "docs/case-studies.md").read_text(encoding="utf-8")
        for fact in ("18 of 18", "52.141", "ERR_NVGPUCTRPERM"):
            self.assertIn(fact, validation)
        for fact in ("60.4616%", "140"):
            self.assertIn(fact, cases)
        self.assertNotIn("60.4616%", validation)

    def test_readmes_route_to_public_documents(self) -> None:
        links = (
            "docs/getting-started.md",
            "docs/environment-readiness.md",
            "docs/workflows.md",
            "docs/evidence-and-safety.md",
            "docs/compatibility.md",
            "docs/validation.md",
            "docs/case-studies.md",
            "docs/knowledge-and-research.md",
            "docs/long-running-optimization.md",
            "skills/cuda-kernel-optimizer/SKILL.md",
            "skills/cuda-kernel-optimizer/examples/walkthrough.md",
            "tests/gpu/sm120/README.md",
            "LICENSE",
        )
        for text in (self.chinese, self.english):
            for marker in links:
                self.assertIn(marker, text)

    def test_readmes_are_not_internal_cli_or_marketing_guides(self) -> None:
        banned = (
            "python3 scripts/orchestrate.py",
            "python3 scripts/workload_controller.py",
            "python3 tools/publish_dual_remote.py",
            "--run-dir",
            "powerful",
            "seamless",
            "revolutionary",
            "comprehensive",
            "可信边界",
            "终局状态",
            "赋能",
            "无缝",
            "强大",
        )
        for text in (self.chinese, self.english):
            for marker in banned:
                candidate = text.lower() if marker.isascii() else text
                self.assertNotIn(marker, candidate)
            self.assertNotIn("```bash", text)

    def test_readmes_and_compatibility_page_link_to_each_other(self) -> None:
        self.assertIn("README.en.md", self.chinese)
        self.assertIn("README.md", self.english)
        self.assertIn("README.md", self.compat)
        self.assertIn("README.en.md", self.compat)

    def test_local_readme_links_resolve(self) -> None:
        markdown = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        html = re.compile(r'href="([^"]+)"')
        for path, text in (
            (README_ZH, self.chinese),
            (README_EN, self.english),
            (README_ZH_COMPAT, self.compat),
        ):
            for target in markdown.findall(text) + html.findall(text):
                if "://" in target or target.startswith("#"):
                    continue
                resolved = (path.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(resolved.exists(), f"missing README link: {target}")


if __name__ == "__main__":
    unittest.main()
