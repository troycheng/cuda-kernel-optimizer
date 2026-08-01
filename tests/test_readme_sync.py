from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_ZH = ROOT / "README.md"
README_EN = ROOT / "README.en.md"
MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
EDGE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)[^-\n]*?\s*(-->|-\.->)\s*"
    r"(?:\|[^|]+\|\s*)?([a-z][a-z0-9_]*)",
    re.MULTILINE,
)


class ReadmeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chinese = README_ZH.read_text(encoding="utf-8")
        self.english = README_EN.read_text(encoding="utf-8")

    def test_chinese_is_primary_and_english_matches_its_structure(self) -> None:
        self.assertEqual(
            re.findall(r"^## .+$", self.chinese, re.MULTILINE),
            [
                "## 项目定位",
                "## 能力范围",
                "## 使用准备",
                "## 快速开始",
                "## 优化模型",
                "## 设计演进",
                "## 结果与验收",
                "## 版本说明",
                "## 进一步阅读",
            ],
        )
        self.assertEqual(
            re.findall(r"^## .+$", self.english, re.MULTILINE),
            [
                "## Project scope",
                "## Capabilities",
                "## Prerequisites",
                "## Quick start",
                "## Optimization model",
                "## Design evolution",
                "## Results and acceptance",
                "## Release notes",
                "## Further reading",
            ],
        )
        self.assertIn('href="README.en.md"', self.chinese)
        self.assertIn('href="README.md"', self.english)
        self.assertLessEqual(len(self.chinese.splitlines()), 220)
        self.assertLessEqual(len(self.english.splitlines()), 220)

    def test_front_door_explains_the_product_before_its_internal_model(self) -> None:
        chinese = " ".join(self.chinese.split())
        english = " ".join(self.english.split())
        for marker in (
            "用户提供可运行的测试 workload",
            "完整执行路径",
            "基于 CUDA、CUTLASS 和 Triton 的 GPU kernel 与算子实现",
            "PyTorch 框架以及 vLLM、TensorRT-LLM 推理系统",
            "局部 kernel 提升不等于业务提速",
            "不承诺固定收益",
            "ChatGPT 显式选择",
        ):
            self.assertIn(marker, chinese)
        for marker in (
            "The user supplies a runnable test workload",
            "complete execution path",
            "GPU kernel and operator implementations built with CUDA, CUTLASS, or Triton",
            "the PyTorch framework and vLLM/TensorRT-LLM inference systems",
            "A faster kernel is not the same as a faster service",
            "does not promise a fixed speedup",
            "only after ChatGPT explicitly selects it",
        ):
            self.assertIn(marker, english)

    def test_design_is_explained_before_its_history_and_cost(self) -> None:
        chinese = " ".join(self.chinese.split())
        english = " ".join(self.english.split())
        for marker in (
            "ChatGPT 负责优化判断",
            "确定性工具负责执行",
            "证据记录负责衔接",
            "早期版本",
            "删除了上万行代码",
            "一部分投入形成了返工",
        ):
            self.assertIn(marker, chinese)
        for marker in (
            "ChatGPT owns optimization judgment",
            "Deterministic tools own execution",
            "Evidence records connect the work",
            "Earlier versions",
            "more than ten thousand lines",
            "Some investment became rework",
        ):
            self.assertIn(marker, english)
        self.assertLess(self.chinese.index("## 优化模型"), self.chinese.index("## 设计演进"))
        self.assertLess(self.english.index("## Optimization model"), self.english.index("## Design evolution"))
        self.assertIn("可交接、可审查", self.chinese)
        self.assertNotIn("可恢复", self.chinese)

    def test_opening_explains_product_use_and_result_boundary(self) -> None:
        chinese = " ".join(self.chinese.split())
        english = " ".join(self.english.split())
        for marker in ("ChatGPT", "GPU 性能优化 skill", "完整 workload", "精度"):
            self.assertIn(marker, chinese[:2500])
        for marker in ("ChatGPT", "GPU performance optimization skill", "complete workload", "correctness"):
            self.assertIn(marker, english[:2500])
        self.assertIn("不能声称完整业务提速", chinese)
        self.assertIn("cannot claim complete business speedup", english)

    def test_first_use_is_ai_executed_and_concrete(self) -> None:
        chinese = " ".join(self.chinese.split())
        english = " ".join(self.english.split())
        for marker in (
            "用户不需要手工运行",
            "最多用 10 分钟",
            "不要修改源码",
            "不声称获得提速",
            "安装标签、commit 和目标目录",
        ):
            self.assertIn(marker, chinese)
        for marker in (
            "do not need to run",
            "Spend at most 10 minutes",
            "Do not edit source",
            "Do not claim a speedup",
            "tag, commit, and destination",
        ):
            self.assertIn(marker, english)

    def test_inputs_use_plain_user_terms(self) -> None:
        for marker in (
            "测试 workload（数据集、代表性请求或 replay）",
            "精度校验（期望输出、容差或业务精度指标）",
            "最低有效收益",
            "不会自行下载或编造 workload",
        ):
            self.assertIn(marker, self.chinese)
        for marker in (
            "Test workload (dataset, representative requests, or replay)",
            "Correctness checks (expected outputs, tolerances, or accuracy criteria)",
            "Minimum useful effect",
            "does not download or invent a workload",
        ):
            self.assertIn(marker, self.english)
        self.assertNotIn("正确性 reference", self.chinese)
        self.assertNotIn("correctness reference", self.english.lower())

    def test_two_diagrams_show_the_same_v14_information_flow(self) -> None:
        chinese = MERMAID.findall(self.chinese)
        english = MERMAID.findall(self.english)
        self.assertEqual(len(chinese), 2)
        self.assertEqual(len(english), 2)
        expected = (
            {
                ("input", "-->", "ai"),
                ("ai", "-->", "tools"),
                ("tools", "-->", "evidence"),
                ("evidence", "-->", "ai"),
                ("ai", "-->", "outcome"),
            },
            {
                ("facts", "-->", "map"),
                ("source", "-->", "hypotheses"),
                ("map", "-->", "hypotheses"),
                ("objective", "-->", "decision"),
                ("hypotheses", "-->", "decision"),
                ("decision", "-->", "check"),
                ("check", "-->", "facts"),
                ("decision", "-->", "experiment"),
                ("decision", "-->", "stop"),
            },
        )
        for index, wanted in enumerate(expected):
            self.assertEqual(set(EDGE.findall(chinese[index])), wanted)
            self.assertEqual(set(EDGE.findall(english[index])), wanted)

    def test_readmes_name_v14_records_and_no_old_workflow(self) -> None:
        for text in (self.chinese, self.english):
            for marker in (
                "target.json",
                "experiments/<experiment-id>.json",
                "invocations/<invocation-id>/",
                "request.json",
                "events.jsonl",
                "result.json",
                "champion/",
                "handoff.md",
                "### V1.4.0",
            ):
                self.assertIn(marker, text)
            lowered = text.lower()
            for legacy in (
                "controller",
                "orchestrate.py",
                "checkpoint",
                "active_diagnosis",
                "decision.json",
                "run-level grant",
            ):
                self.assertNotIn(legacy, lowered)

    def test_host_profiler_and_knowledge_boundaries_match(self) -> None:
        chinese = " ".join(self.chinese.split())
        english = " ".join(self.english.split())
        self.assertIn("宿主机变化默认只给建议", chinese)
        self.assertIn("Profiler 不是固定阶段", chinese)
        self.assertIn("外部搜索与第三方 AI", chinese)
        self.assertIn("host changes", english.lower())
        self.assertIn("A profiler is not a mandatory stage", english)
        self.assertIn("External search and third-party AI", english)

    def test_compatibility_readme_is_only_a_language_pointer(self) -> None:
        compat = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(compat.splitlines()), 8)
        self.assertIn("README.md", compat)
        self.assertIn("README.en.md", compat)


if __name__ == "__main__":
    unittest.main()
