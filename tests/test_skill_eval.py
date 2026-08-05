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

    def test_business_positive_tail_result_is_not_discarded_by_throughput_coverage(self) -> None:
        skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        iteration = (SKILL / "references" / "performance_iteration.md").read_text(
            encoding="utf-8"
        )
        for marker in (
            "不能仅因调用次数少或总耗时占比低而否定",
            "不只看 primary",
            "就将它纳入优化结果并明确适用场景",
            "没有提升当前主指标不能单独作为丢弃理由",
        ):
            self.assertIn(marker, skill)
        for marker in (
            "时间占比和 Amdahl 上限只适用于吞吐、均值",
            "不能单独作为丢弃改动的理由",
            "保留整体不负向的长尾",
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


if __name__ == "__main__":
    unittest.main()
