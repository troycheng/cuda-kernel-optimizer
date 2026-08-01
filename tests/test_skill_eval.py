from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


if __name__ == "__main__":
    unittest.main()
