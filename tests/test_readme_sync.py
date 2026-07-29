from __future__ import annotations

import hashlib
import json
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


def claims_v1_3_release(text: str) -> bool:
    version = r"V1\.3(?:\.\d+)*"
    english = r"(?:released|published|current\s+release)"
    chinese = r"(?:已发布|正式发布|当前版本)"
    return bool(
        re.search(r"(?m)^#+\s+V1\.3\.\d+(?:\.\d+)*\b", text)
        or re.search(rf"(?is)\b{version}\b.{{0,80}}\b{english}\b", text)
        or re.search(rf"(?is)\b{english}\b.{{0,80}}\b{version}\b", text)
        or re.search(rf"(?s){version}.{{0,80}}{chinese}", text)
        or re.search(rf"(?s){chinese}.{{0,80}}{version}", text)
    )


class ReadmeSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chinese = README_ZH.read_text(encoding="utf-8")
        self.english = README_EN.read_text(encoding="utf-8")
        self.compat = README_ZH_COMPAT.read_text(encoding="utf-8")

    def test_language_roles_and_full_readme_sizes(self) -> None:
        self.assertIn("## 项目概述", self.chinese)
        self.assertIn("## Project overview", self.english)
        self.assertNotIn("## Project overview", self.chinese)
        self.assertNotIn("## 项目概述", self.english)
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
        self.assertEqual(
            re.findall(r"^## .+$", self.chinese, re.MULTILINE),
            [
                "## 项目概述",
                "## 核心能力",
                "## 快速开始",
                "## 工作流程",
                "## 结果与验收",
                "## 版本说明",
                "## 相关文档",
            ],
        )
        self.assertLess(
            self.chinese.index("## 快速开始"),
            self.chinese.index("## 工作流程"),
        )

    def test_english_readme_follows_chinese_information_architecture(self) -> None:
        self.assertEqual(
            re.findall(r"^## .+$", self.english, re.MULTILINE),
            [
                "## Project overview",
                "## Core capabilities",
                "## Quick start",
                "## Workflow",
                "## Results and acceptance",
                "## Release notes",
                "## Documentation",
            ],
        )
        self.assertEqual(
            re.findall(r"^### .+$", self.english, re.MULTILINE),
            [
                "### Installation",
                "### What to prepare",
                "### Run a 10-minute fit check",
                "### Start formal optimization",
                "### How optimization directions are formed",
                "### How candidate changes advance",
                "### V1.3.0",
                "### V1.2.0",
                "### V1.1.0",
                "### V1.0.1",
                "### V1.0.0",
            ],
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
            "build, correctness, benchmark, GPU, and profiler",
            "CUDA",
            "CUTLASS",
            "Triton",
            "complete, repeatable workload",
            "serving KPI",
            "existing NCU report",
        ):
            self.assertIn(marker, self.english)

    def test_readmes_show_their_public_workflow(self) -> None:
        chinese = MERMAID.findall(self.chinese)
        english = MERMAID.findall(self.english)
        self.assertEqual(len(chinese), 2)
        self.assertEqual(len(english), 2)
        diagnosis_edges = set(EDGE.findall(chinese[0]))
        for edge in (
            ("baseline", "-->", "execution"),
            ("profile", "-->", "execution"),
            ("execution", "-->", "accounting"),
            ("accounting", "-->", "hypotheses"),
            ("hypotheses", "-->", "falsifier"),
            ("falsifier", "-->", "evidence"),
            ("evidence", "-->", "execution"),
        ):
            self.assertIn(edge, diagnosis_edges)
        candidate_edges = set(EDGE.findall(chinese[1]))
        for edge in (
            ("direction", "-->", "candidate"),
            ("candidate", "-->", "gate"),
            ("pause", "-->", "gate"),
            ("stage", "-->", "result"),
            ("reject", "-->", "analysis"),
            ("keep", "-->", "finish"),
        ):
            self.assertIn(edge, candidate_edges)
        self.assertEqual(
            set(EDGE.findall(english[0])),
            diagnosis_edges,
        )
        self.assertEqual(
            set(EDGE.findall(english[1])),
            candidate_edges,
        )

    def test_readmes_explain_v1_2_adaptive_investment(self) -> None:
        chinese = "".join(self.chinese.split()).replace("`", "")
        for marker in (
            "收益上限",
            "最低成本",
            "运行级授权",
            "不会为了用完授权时间而继续实验",
            "等待和暂停不计入",
            "保存现场并暂停",
        ):
            self.assertIn(marker, chinese)
        english = " ".join(self.english.split()).replace("`", "")
        for marker in (
            "run-level grant",
            "scope, risk, stage",
            "waiting does not consume it",
            "does not continue experimenting just to spend the authorization",
            "saves the run state and pauses",
        ):
            self.assertIn(marker, english)

    def test_readmes_explain_candidate_stages_and_recovery(self) -> None:
        for marker in (
            "静态检查",
            "最低正确性",
            "短版成对测试",
            "不会重复运行已经完成的昂贵阶段",
            "补充授权后继续",
            "明确放弃",
        ):
            self.assertIn(marker, self.chinese)
        english = " ".join(self.english.lower().split())
        for marker in (
            "static review or isolated small test",
            "minimum correctness",
            "short paired screen",
            "will not rerun completed expensive stages",
            "resume after additional authorization",
            "explicitly abandons",
        ):
            self.assertIn(marker, english)

    def test_real_workload_and_claim_boundaries_are_explicit(self) -> None:
        for marker in (
            "真实 workload",
            "测试 workload（测试集或代表性请求）",
            "正确性校验标准",
            "不会自行下载或编造",
            "不能作为性能提升结论",
            "局部 kernel 变快不等于完整 workload 变快",
            "正确性和成对性能数据",
        ):
            self.assertIn(marker, self.chinese)
        self.assertNotIn("正确性 reference", self.chinese)
        english = " ".join(self.english.split())
        for marker in (
            "test workload (dataset, representative requests, or replay)",
            "correctness checks (expected outputs, tolerances, or accuracy criteria)",
            "does not download or invent one",
            "does not claim a speedup",
            "correctness",
            "paired",
        ):
            self.assertIn(marker, english)
        self.assertNotIn("correctness reference", english.lower())

    def test_readiness_and_host_boundaries_are_explicit(self) -> None:
        for marker in (
            "编译、正确性、benchmark、GPU、profiler",
            "不自动修改",
            "self_check",
            "ERR_NVGPUCTRPERM",
            "外部搜索和第三方 AI",
        ):
            self.assertIn(marker, self.chinese)
        english = " ".join(self.english.split())
        for marker in (
            "build, correctness, benchmark, GPU, and profiler",
            "does not modify them automatically",
            "self_check",
            "ERR_NVGPUCTRPERM",
            "External search and third-party AI",
        ):
            self.assertIn(marker, english)

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
        english = " ".join(self.english.lower().split())
        for marker in (
            "summary.md",
            "active_diagnosis/initial_investment_brief.json",
            "performance_model.json",
            "decision.json",
            "raw paired samples",
            "evidence-integrity records",
        ):
            self.assertIn(marker, english)

    def test_readmes_publish_v1_3(self) -> None:
        release_headings = (
            (self.chinese, "### V1.3.0"),
            (self.english, "### V1.3.0"),
        )
        for text, heading in release_headings:
            self.assertEqual(text.count(heading), 1)
            self.assertNotIn("开发中", text)
            self.assertNotIn("in development", text.lower())
            self.assertEqual(text.count("### V1.2.0"), 1)
            self.assertEqual(text.count("### V1.1.0"), 1)
            self.assertEqual(text.count("### V1.0.1"), 1)
            self.assertEqual(text.count("### V1.0.0"), 1)
            self.assertNotRegex(text, r"(?m)^### V(?:2|3)\.")

    def test_v1_3_release_is_backed_by_retained_controller_replays(self) -> None:
        package_suite = json.loads(
            (
                ROOT
                / "tests"
                / "fixtures"
                / "knowledge_replay"
                / "fresh_controller_cases.json"
            ).read_text(encoding="utf-8")
        )
        postfreeze_suite = json.loads(
            (
                ROOT
                / "tests"
                / "fixtures"
                / "knowledge_replay"
                / "postfreeze_controller_cases.json"
            ).read_text(encoding="utf-8")
        )
        package_regressions = sum(
            case["scoring_group"] == "triton"
            and case["replay_eligibility"]["status"] == "package_regression"
            for case in package_suite["cases"]
        )
        scoreable = [
            case
            for case in postfreeze_suite["cases"]
            if case["scoring_group"] == "triton"
            and case["replay_eligibility"]["status"] == "scoreable"
        ]
        source_manifests = {
            case["input_snapshot"]["archive_identity_facts"][
                "source_manifest_sha256"
            ]
            for case in scoreable
        }
        source_commits = {
            case["input_snapshot"]["archive_identity_facts"][
                "controller_source_identity"
            ]["source_repo_head"]
            for case in scoreable
        }
        package_case_numbers = {
            case["case_id"].split("-", 1)[0]
            for case in package_suite["cases"]
        }
        replay_case_numbers = {
            case["case_id"].split("-", 1)[0]
            for case in scoreable
        }
        self.assertEqual(
            postfreeze_suite["cases_sha256"],
            hashlib.sha256(
                json.dumps(
                    postfreeze_suite["cases"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest(),
        )
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        public_paths = (
            README_ZH,
            README_EN,
            README_ZH_COMPAT,
            ROOT / "docs" / "index.md",
            ROOT / "docs" / "getting-started.md",
            ROOT / "docs" / "environment-readiness.md",
            ROOT / "docs" / "workflows.md",
            ROOT / "docs" / "long-running-optimization.md",
            ROOT / "docs" / "evidence-and-safety.md",
            ROOT / "docs" / "compatibility.md",
            ROOT / "docs" / "validation.md",
            ROOT / "docs" / "case-studies.md",
            ROOT / "docs" / "knowledge-and-research.md",
            ROOT / "skills" / "cuda-kernel-optimizer" / "SKILL.md",
        )
        v1_3_release_claimed = version.startswith("1.3") or any(
            claims_v1_3_release(path.read_text(encoding="utf-8"))
            for path in public_paths
        )
        self.assertEqual(package_regressions, 6)
        self.assertEqual(len(scoreable), 6)
        self.assertEqual(len(source_manifests), 6)
        self.assertEqual(
            package_case_numbers,
            replay_case_numbers,
            "the V1.3 release set is an explicit retained-case regression",
        )
        self.assertEqual(
            source_commits,
            {"db5d19c8a03a6f8350294e582dd9f283259262f4"},
        )
        self.assertEqual(version, "1.3.0")
        self.assertTrue(
            v1_3_release_claimed,
            "the release must be stated only after six scoreable retained-case replays exist",
        )

    def test_readmes_explain_v1_3_evidence_bound_knowledge(self) -> None:
        for marker in (
            "当前 workload 的封存证据",
            "最多三个可证伪方向",
            "历史收益数字",
            "active_diagnosis/knowledge_context.json",
            "知识库没有匹配不会阻止模型提出方向",
        ):
            self.assertIn(marker, self.chinese)
        english = " ".join(self.english.split())
        for marker in (
            "sealed evidence from the current workload",
            "at most three falsifiable directions",
            "historical speedup numbers",
            "active_diagnosis/knowledge_context.json",
            "A missing knowledge match does not block a model-proposed direction",
        ):
            self.assertIn(marker, english)

    def test_v1_3_release_claim_detection_covers_public_wording(self) -> None:
        for claim in (
            "### V1.3.1",
            "V1.3 is now released.",
            "The current release is V1.3.2.",
            "V1.3 已正式发布。",
            "当前版本为 V1.3。",
        ):
            with self.subTest(claim=claim):
                self.assertTrue(claims_v1_3_release(claim))

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
            "## 正式性能结论通常需要什么",
            "## V1.2 如何控制投入",
            "## 结论能到什么程度",
            "## 安全边界",
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
