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
        self.assertIn("Route", self.text)
        self.assertIn("ChatGPT", self.text)

    def test_chatgpt_is_the_only_optimization_decision_maker(self) -> None:
        self.assertRegex(
            self.prose,
            r"ChatGPT.{0,80}(only|sole).{0,80}(optimization )?decision",
        )
        for responsibility in (
            "bottleneck",
            "direction",
            "candidate",
            "ROI",
            "next step",
        ):
            self.assertRegex(
                self.prose,
                rf"ChatGPT.{{0,260}}{re.escape(responsibility)}",
            )
        self.assertRegex(
            self.lower,
            r"tools?.{0,100}(do not|cannot).{0,120}(direction|roi|next step)",
        )

    def test_router_mentions_only_v14_public_tools(self) -> None:
        routed = set(re.findall(r"(?:<skill>/)?scripts/([a-z_]+\.py)", self.text))
        self.assertEqual(routed, _PUBLIC_TOOLS)

    def test_original_tests_and_precision_validation_precede_measurement(self) -> None:
        self.assertRegex(
            self.lower,
            r"original test.{0,120}(precision|correctness).{0,120}before.{0,120}"
            r"(benchmark|profil|measur)",
        )

    def test_no_legacy_control_plane_terms_remain(self) -> None:
        for pattern in _LEGACY_CONTROL_WORDS:
            self.assertNotRegex(self.lower, pattern)

    def test_profiler_and_knowledge_only_return_facts_and_empty_knowledge_does_not_block(self) -> None:
        self.assertRegex(
            self.lower,
            r"profiler.{0,160}(facts|observations).{0,180}(not|never).{0,100}(direction|roi|next step)",
        )
        self.assertRegex(
            self.lower,
            r"knowledge.{0,180}(facts|results).{0,180}(not|never).{0,100}(direction|roi|next step)",
        )
        self.assertRegex(
            self.lower,
            r"(empty|no) knowledge.{0,160}(does not|cannot|never).{0,80}block",
        )

    def test_external_search_and_ai_are_optional_and_local_evidence_decides(self) -> None:
        self.assertRegex(self.lower, r"(external )?(search|ai).{0,120}optional")
        self.assertRegex(self.lower, r"local evidence.{0,100}(decisive|decides|authoritative)")

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
