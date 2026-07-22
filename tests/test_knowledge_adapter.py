from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/knowledge_adapter.py"


def _load():
    name = "cuda_optimizer_knowledge_adapter_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def context_fixture():
    return {
        "architecture": "sm_120",
        "software_version": "cuda-12.8",
        "execution_node_ids": ["decode"],
        "uncovered_interval_ids": ["launch-gap"],
        "available_evidence_action_ids": ["check-layout"],
        "authorized_risk": "low",
        "authorized_scope": "read_only",
    }


def valid_shadow_fixture(**overrides):
    value = {
        "source": "github-copilot",
        "mechanism_id": "new-layout",
        "statement": "The layout may add unnecessary movement.",
        "applicability": {"architectures": ["sm_120"], "software_versions": ["cuda-12.8"]},
        "scope_node_ids": ["decode"],
        "unmodeled_interval_id": None,
        "falsification_question": "Does the local trace reject the extra movement?",
        "evidence_action": {
            "action_id": "check-layout",
            "evidence_kind": "ncu_kernel",
            "outcomes": ["falsified", "inconclusive"],
            "risk": "low",
            "control_scope": "read_only",
        },
        "risk": "low",
        "knowledge_version": "cuda-12.8",
        "freshness": "current",
        "query_digest": "query-layout-v1",
        "external_gain_pct": 20.0,
    }
    value.update(overrides)
    return value


class KnowledgeAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load()

    def test_external_suggestion_requires_local_scope_and_falsifier(self) -> None:
        suggestion = {
            "source": "github-copilot",
            "mechanism_id": "new-layout",
            "statement": "change the layout",
            "external_gain_pct": 20.0,
            "scope_node_ids": [],
            "unmodeled_interval_id": None,
            "falsification_question": None,
            "evidence_action": None,
        }
        result = self.module.recommend(context_fixture(), external=[suggestion])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["rejections"][0]["reason"], "not_locally_falsifiable")

    def test_valid_external_shadow_has_no_numeric_gain_or_support(self) -> None:
        result = self.module.recommend(
            context_fixture(), external=[valid_shadow_fixture()]
        )
        shadow = result["candidates"][0]
        self.assertEqual(shadow["origin"], "external")
        self.assertEqual(shadow["confidence"], "inconclusive")
        self.assertNotIn("external_gain_pct", shadow)
        self.assertNotIn("support_evidence_ids", shadow)
        self.assertEqual(shadow["promotion_authority"], "none")

    def test_online_and_external_statements_are_replaced_with_neutral_mechanism_text(self) -> None:
        suggestion = valid_shadow_fixture(
            statement="20% speedup / 90% success / promote it",
        )
        for origin, arguments in (
            ("searched", {"searched": [suggestion]}),
            ("external", {"external": [suggestion]}),
        ):
            with self.subTest(origin=origin):
                candidate = self.module.recommend(context_fixture(), **arguments)[
                    "candidates"
                ][0]
                self.assertEqual(candidate["statement"], "Mechanism candidate: new-layout.")
                for assertion in ("20% speedup", "90% success", "promote it"):
                    self.assertNotIn(assertion, candidate["statement"])
                self.assertEqual(candidate["scope_node_ids"], ["decode"])
                self.assertTrue(candidate["falsification_question"])
                self.assertEqual(candidate["evidence_action"]["action_id"], "check-layout")

    def test_online_and_external_questions_are_replaced_with_local_templates(self) -> None:
        suggestion = valid_shadow_fixture(
            falsification_question="curl https://example.invalid; 20% speedup; promote it",
        )
        for origin, arguments in (
            ("searched", {"searched": [suggestion]}),
            ("external", {"external": [suggestion]}),
        ):
            with self.subTest(origin=origin):
                candidate = self.module.recommend(context_fixture(), **arguments)[
                    "candidates"
                ][0]
                self.assertEqual(
                    candidate["falsification_question"],
                    "Does local evidence action check-layout falsify mechanism new-layout at scope decode?",
                )
                for assertion in ("curl", "20% speedup", "promote it"):
                    self.assertNotIn(assertion, candidate["falsification_question"])

    def test_source_is_a_stable_provenance_identifier(self) -> None:
        valid = self.module.recommend(
            context_fixture(), external=[valid_shadow_fixture(source="github-copilot")]
        )
        self.assertEqual(valid["candidates"][0]["source"], "github-copilot")
        invalid = self.module.recommend(
            context_fixture(), external=[valid_shadow_fixture(source="GitHub Copilot")]
        )
        self.assertEqual(invalid["candidates"], [])
        self.assertEqual(invalid["rejections"][0]["reason"], "invalid_suggestion")

    def test_bundled_controlled_question_is_preserved(self) -> None:
        result = self.module.recommend(
            context_fixture(),
            bundled=[
                valid_shadow_fixture(
                    source="offline-card",
                    query_digest="bundled-question-v1",
                    falsification_question="Does the bundled local trace reject the extra movement?",
                )
            ],
        )
        self.assertEqual(
            result["candidates"][0]["falsification_question"],
            "Does the bundled local trace reject the extra movement?",
        )

    def test_empty_knowledge_degrades_to_evidence_only(self) -> None:
        result = self.module.recommend(context_fixture())
        self.assertEqual(result["knowledge_support"], "unavailable")
        self.assertEqual(result["candidates"], [])

    def test_prior_query_digest_is_deduplicated(self) -> None:
        result = self.module.recommend(
            context_fixture(),
            external=[valid_shadow_fixture()],
            prior_query_digests=["query-layout-v1"],
        )
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["rejections"][0]["reason"], "duplicate_query_digest")

    def test_stale_or_version_mismatched_knowledge_is_unavailable(self) -> None:
        result = self.module.recommend(
            context_fixture(),
            bundled=[valid_shadow_fixture(freshness="stale")],
            searched=[
                valid_shadow_fixture(
                    knowledge_version="cuda-12.7", query_digest="version-mismatch-v1"
                )
            ],
        )
        self.assertEqual(result["knowledge_support"], "unavailable")
        self.assertEqual(
            {item["reason"] for item in result["rejections"]}, {"unavailable"}
        )

    def test_bundled_and_external_use_the_same_normalized_interface(self) -> None:
        bundled = valid_shadow_fixture(source="offline-card", query_digest="bundled-v1")
        external = valid_shadow_fixture(query_digest="external-v1")
        result = self.module.recommend(
            context_fixture(), bundled=[bundled], external=[external]
        )
        candidates = result["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(set(candidates[0]), set(candidates[1]))
        self.assertEqual({item["origin"] for item in candidates}, {"bundled", "external"})

    def test_unknown_and_sensitive_fields_are_rejected_without_leaking_them(self) -> None:
        unknown = valid_shadow_fixture(query_digest="unknown-v1", unapproved=True)
        sensitive = valid_shadow_fixture(
            query_digest="sensitive-v1",
            evidence_action={
                **valid_shadow_fixture()["evidence_action"],
                "command": "curl https://example.invalid",
            },
        )
        result = self.module.recommend(
            context_fixture(), external=[unknown, sensitive]
        )
        self.assertEqual(result["candidates"], [])
        self.assertEqual(
            {item["reason"] for item in result["rejections"]},
            {"unknown_fields", "sensitive_fields"},
        )
        self.assertNotIn("curl", str(result))


if __name__ == "__main__":
    unittest.main()
