from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "cuda-kernel-optimizer"
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


class SkillEvalTests(unittest.TestCase):
    def test_v14_acceptance_is_covered_by_the_six_public_blackbox_groups(self) -> None:
        expected = {
            "test_v14_target_baseline.py",
            "test_v14_variant_experiment.py",
            "test_v14_screen_target.py",
            "test_v14_profiler_knowledge_structure.py",
            "test_v14_invocation_lifecycle.py",
            "test_v14_champion_audit.py",
        }
        self.assertEqual(
            {path.name for path in (ROOT / "tests").glob("test_v14_*.py")},
            expected,
        )

    def test_no_separate_model_planner_evaluation_workflow_remains(self) -> None:
        self.assertFalse((ROOT / "tools" / "run_skill_eval.py").exists())
        self.assertFalse((ROOT / "tests" / "evals").exists())

    def test_public_cli_exposes_only_explicit_v14_operations(self) -> None:
        operations = {
            "readiness.py": "{check}",
            "workload_evaluate.py": (
                "{baseline,experiment,screen,target,final_audit,status,cancel}"
            ),
            "profile_ncu.py": "{analyze,collect,status,cancel}",
            "profile_nsys.py": "{analyze,collect,status,cancel}",
            "profile_pytorch.py": "{analyze,collect,status,cancel}",
            "compiler_evidence.py": "{analyze,status,cancel}",
            "sass_check.py": "{analyze,status,cancel}",
            "knowledge_query.py": "{query}",
            "champion.py": "{show,select,restore-original}",
        }
        for script, choices in operations.items():
            completed = subprocess.run(
                [sys.executable, str(SCRIPTS / script), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, script)
            self.assertIn(choices, completed.stdout, script)

    def test_primary_priority_and_secondary_result_retention_are_distinct(self) -> None:
        skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        iteration = (SKILL / "references" / "performance_iteration.md").read_text(
            encoding="utf-8"
        )
        for marker in (
            "primary 指标决定候选排序和任务是否完成",
            "不能仅因调用次数少或总耗时占比低而否定",
            "不只看 primary",
            "将它纳入优化结果",
            "不能成为当前 Target 的 Champion",
            "不能据此停止主目标搜索",
            "局部结果存在不等于优化任务完成",
        ):
            self.assertIn(marker, skill)
        for marker in (
            "primary 决定研究方向和任务是否完成",
            "不是候选排序规则",
            "时间占比和 Amdahl 上限只适用于吞吐、均值",
            "不能单独作为删除改动的理由",
            "保留整体不负向的长尾",
            "不得把任务标记为优化完成",
        ):
            self.assertIn(marker, iteration)

    def test_post_use_feedback_is_actionable_and_opt_in_for_external_submission(self) -> None:
        skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        for marker in (
            "本次真实使用暴露",
            "观察到的行为、实际影响或成本、期望改动和最小证据",
            "没有可行动反馈时写 `none`",
            "只有用户明确授权时才向外部仓库提交反馈",
            "本身不等于 skill feedback",
        ):
            self.assertIn(marker, skill)

    def test_real_use_regressions_require_model_level_gates(self) -> None:
        skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        iteration = (SKILL / "references" / "performance_iteration.md").read_text(
            encoding="utf-8"
        )
        serving = (SKILL / "references" / "serving_evidence_protocol.md").read_text(
            encoding="utf-8"
        )
        systems = (SKILL / "references" / "systems_and_ir_coverage.md").read_text(
            encoding="utf-8"
        )
        for marker in (
            "明确分子、分母、数据来源",
            "当前候选族关闭不等于 Target 完成",
            "GPU 进程列表为空或显存占用很低都不能单独证明设备空闲",
            "字符串只证明文本存在",
        ):
            self.assertIn(marker, skill)
        for marker in (
            "elapsed/remaining budget",
            "跨 subsystem",
            "重新 tokenize 生成文本",
        ):
            self.assertIn(marker, iteration)
        self.assertIn("metric semantic audit", serving)
        self.assertIn("预 tokenized token ids", serving)
        self.assertIn("写 unproven", systems)

    def test_secondary_only_result_cannot_bypass_the_primary_verdict(self) -> None:
        skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        iteration = (SKILL / "references" / "performance_iteration.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("通过当前 Target 的 primary verdict", skill)
        self.assertIn("secondary-only 收益保留为局部结果", skill)
        self.assertIn("只有当前 Target 的 primary verdict 通过后", iteration)
        self.assertNotIn("选择理由只来自 secondary", skill)
        self.assertNotIn("选择理由只来自预先声明的 secondary", iteration)


if __name__ == "__main__":
    unittest.main()
