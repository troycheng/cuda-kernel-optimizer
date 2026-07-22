from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

from tests.test_analysis_epoch import epoch_fixture
from tests.test_evidence_selector import catalog_fixture, policy_fixture, request_fixture
from tests.test_execution_map import evidence_catalog, map_fixture
from tests.test_hypothesis_space import hypothesis_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "cuda-kernel-optimizer" / "scripts"


def _load(filename: str, name: str):
    path = SCRIPTS / filename
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class DiagnosticDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.map_module = _load("execution_map.py", "decision_execution_map")
        self.model_module = _load("performance_model.py", "decision_performance_model")
        self.hypothesis_module = _load("hypothesis_space.py", "decision_hypothesis_space")
        self.selector_module = _load("evidence_selector.py", "decision_evidence_selector")
        self.module = _load("diagnostic_decision.py", "diagnostic_decision_test")
        self.epoch = epoch_fixture()
        self.execution_map = map_fixture(self.map_module)
        self.evidence = evidence_catalog()

    def hypotheses(self, value=None):
        return self.hypothesis_module.validate_hypothesis_set(
            hypothesis_fixture(self.hypothesis_module, self.map_module)
            if value is None
            else value,
            epoch=self.epoch,
            execution_map=self.execution_map,
            evidence_catalog=self.evidence,
        )

    def model(self, execution_map=None, *, minimum_effect_us=1.0, timings=()):
        return self.model_module.build_performance_model(
            self.execution_map if execution_map is None else execution_map,
            minimum_effect_us=minimum_effect_us,
            action_timings=list(timings),
        )

    def selection(self, hypothesis_result, *, policy=None, only_request=None):
        request = request_fixture()
        request["epoch_sha256"] = self.map_module.epoch_digest(self.epoch)
        request["hypothesis_set_sha256"] = hypothesis_result["hypothesis_set_sha256"]
        if only_request is not None:
            request["requests"] = [request["requests"][only_request]]
        return self.selector_module.select_evidence_request(
            request,
            epoch=self.epoch,
            execution_map=self.execution_map,
            hypothesis_result=hypothesis_result,
            evidence_catalog=self.evidence,
            action_catalog=catalog_fixture(),
            policy=policy_fixture() if policy is None else policy,
            request_history=[],
        )

    def decide(self, model, hypotheses, selection, *, external_review=None):
        return self.module.decide_next_step(
            model,
            hypotheses,
            selection,
            external_review=external_review,
        )

    def test_selected_discriminator_returns_measure_with_unknown_numeric_cost(self) -> None:
        hypotheses = self.hypotheses()
        result = self.decide(
            self.model(), hypotheses, self.selection(hypotheses)
        )

        self.assertEqual(result["decision"], "MEASURE")
        self.assertEqual(result["next_action"]["action_id"], "framework-targeted")
        self.assertEqual(result["cost"]["class"], "low")
        self.assertIsNone(result["cost"]["p50_seconds"])
        self.assertIsNone(result["cost"]["p90_seconds"])
        self.assertEqual(result["cost"]["basis"], "unavailable")
        self.assertEqual(result["next_checkpoint"], "after_selected_evidence")

    def test_supported_direction_returns_pursue(self) -> None:
        value = hypothesis_fixture(self.hypothesis_module, self.map_module)
        framework, kernel = value["hypotheses"]
        framework.update(
            {
                "confidence": "direction_supported",
                "support_evidence_ids": ["ev-cpu", "ev-gpu"],
                "missing_evidence_kinds": [],
            }
        )
        kernel.update(
            {
                "disposition": "rejected",
                "oppose_evidence_ids": ["ev-edge"],
                "missing_evidence_kinds": [],
            }
        )
        hypotheses = self.hypotheses(value)

        result = self.decide(
            self.model(), hypotheses, self.selection(hypotheses)
        )

        self.assertEqual(result["decision"], "PURSUE")
        self.assertEqual(result["primary_diagnosis"]["claim_layer"], "runtime")
        self.assertEqual(result["next_checkpoint"], "after_candidate_screen")

    def test_no_direction_above_minimum_effect_returns_stop(self) -> None:
        value = copy.deepcopy(self.execution_map)
        value["nodes"][0].update(
            {"duration_us": 0.4, "first_start_us": 0.0, "last_end_us": 0.4}
        )
        value["nodes"][1].update(
            {"duration_us": 0.4, "first_start_us": 999.6, "last_end_us": 1000.0}
        )
        hypotheses = self.hypotheses()

        result = self.decide(
            self.model(value), hypotheses, self.selection(hypotheses)
        )

        self.assertEqual(result["decision"], "STOP")
        self.assertEqual(result["terminal_reason"], "benefit_ceiling_below_minimum_effect")
        self.assertIsNone(result["next_action"])

    def test_overlapping_scoped_nodes_use_union_not_summed_duration(self) -> None:
        execution_map = copy.deepcopy(self.execution_map)
        for node in execution_map["nodes"]:
            node.update(
                {
                    "duration_us": 700.0,
                    "first_start_us": 0.0,
                    "last_end_us": 700.0,
                }
            )
        value = hypothesis_fixture(self.hypothesis_module, self.map_module)
        value["execution_map_sha256"] = self.map_module.execution_map_digest(
            execution_map,
            epoch=self.epoch,
            evidence_catalog=self.evidence,
        )
        value["hypotheses"] = [
            {
                "hypothesis_id": "h-overlapping-path",
                "kind": "mechanism",
                "scope_node_ids": ["cpu-launch", "gpu-kernel"],
                "statement": "One mechanism affects two fully overlapping nodes.",
                "mechanism": "shared_overlapping_path",
                "claim_layer": "runtime",
                "disposition": "active",
                "confidence": "direction_supported",
                "support_evidence_ids": ["ev-cpu", "ev-gpu"],
                "oppose_evidence_ids": [],
                "missing_evidence_kinds": [],
                "falsification_question": "Does either node remain after the shared mechanism is removed?",
            }
        ]
        value["relationships"] = []
        hypotheses = self.hypothesis_module.validate_hypothesis_set(
            value,
            epoch=self.epoch,
            execution_map=execution_map,
            evidence_catalog=self.evidence,
        )
        selection = {
            "status": "sufficient",
            "selected_request": None,
            "rejections": [],
            "missing_capability_ids": [],
            "gap_reason": "hypotheses_sufficiently_supported",
        }

        result = self.decide(
            self.model(execution_map, minimum_effect_us=800.0),
            hypotheses,
            selection,
        )

        self.assertEqual(result["benefit_ceiling"]["microseconds"], 700.0)
        self.assertEqual(result["decision"], "STOP")
        self.assertEqual(
            result["terminal_reason"], "benefit_ceiling_below_minimum_effect"
        )

    def test_high_value_action_outside_authorization_requires_review(self) -> None:
        hypotheses = self.hypotheses()
        policy = policy_fixture()
        policy["max_cost"] = "low"
        selection = self.selection(hypotheses, policy=policy, only_request=1)

        result = self.decide(self.model(), hypotheses, selection)

        self.assertEqual(selection["rejections"][0]["reason"], "cost_exceeds_policy")
        self.assertEqual(result["decision"], "REVIEW_REQUIRED")
        self.assertEqual(result["terminal_reason"], "valuable_action_outside_authorization")
        self.assertEqual(result["next_action"]["action_id"], "ncu-targeted")
        self.assertEqual(result["cost"]["class"], "high")
        self.assertEqual(result["next_checkpoint"], "after_authorization_decision")

    def test_no_active_hypothesis_returns_stop_not_another_round(self) -> None:
        value = hypothesis_fixture(self.hypothesis_module, self.map_module)
        for item in value["hypotheses"]:
            item.update(
                {
                    "disposition": "rejected",
                    "confidence": "inconclusive",
                    "oppose_evidence_ids": ["ev-edge"],
                    "missing_evidence_kinds": [],
                }
            )
        hypotheses = self.hypotheses(value)
        selection = {
            "status": "evidence_gap",
            "selected_request": None,
            "rejections": [],
            "missing_capability_ids": [],
            "gap_reason": "no_admissible_discriminator",
        }

        result = self.decide(self.model(), hypotheses, selection)

        self.assertEqual(result["decision"], "STOP")
        self.assertEqual(result["terminal_reason"], "no_active_hypothesis")
        self.assertNotIn("next round", str(result).lower())

    def test_identity_matched_history_is_used_for_selected_action_only(self) -> None:
        hypotheses = self.hypotheses()
        timings = [
            {
                "action_id": "framework-targeted",
                "identities": copy.deepcopy(self.execution_map["identities"]),
                "elapsed_seconds": value,
            }
            for value in (8.0, 12.0, 20.0)
        ]

        result = self.decide(
            self.model(timings=timings), hypotheses, self.selection(hypotheses)
        )

        self.assertEqual(result["cost"]["p50_seconds"], 12.0)
        self.assertEqual(result["cost"]["p90_seconds"], 20.0)
        self.assertEqual(result["cost"]["basis"], "identity_matched_history")

    def test_external_unavailability_or_disagreement_cannot_override_local_decision(self) -> None:
        hypotheses = self.hypotheses()
        model = self.model()
        selection = self.selection(hypotheses)
        local = self.decide(model, hypotheses, selection)
        unavailable = {
            "status": "unavailable",
            "providers_requested": ["google-ai-mode"],
            "providers_completed": [],
            "failed_providers": ["google-ai-mode"],
            "total_wait_seconds": 1.5,
            "reviews": [],
        }
        conflict = {
            "status": "completed",
            "providers_requested": ["glm", "kimi"],
            "providers_completed": ["glm", "kimi"],
            "failed_providers": [],
            "total_wait_seconds": 2.0,
            "reviews": [
                {"provider": "glm", "status": "completed", "response": {"verdict": "support"}},
                {
                    "provider": "kimi",
                    "status": "completed",
                    "response": {
                        "verdict": "challenge",
                        "concerns": ["The trace may mix warmup and steady state."],
                        "suggested_experiments": ["Repeat one bounded steady-state trace."],
                    },
                },
            ],
        }

        for external in (unavailable, conflict):
            with self.subTest(status=external["status"]):
                result = self.decide(
                    model, hypotheses, selection, external_review=external
                )
                self.assertEqual(result["decision"], local["decision"])
                self.assertTrue(result["external_challenge"]["advisory_only"])

        challenged = self.decide(
            model, hypotheses, selection, external_review=conflict
        )["external_challenge"]
        self.assertEqual(
            challenged["challenges"],
            [
                {
                    "provider": "kimi",
                    "concerns": ["The trace may mix warmup and steady state."],
                    "suggested_experiments": ["Repeat one bounded steady-state trace."],
                }
            ],
        )
        self.assertEqual(
            challenged["local_adjudication"]["status"], "continue_measurement"
        )

    def test_supported_local_direction_retains_unmapped_external_challenge(self) -> None:
        value = hypothesis_fixture(self.hypothesis_module, self.map_module)
        framework, kernel = value["hypotheses"]
        framework.update(
            {
                "confidence": "direction_supported",
                "support_evidence_ids": ["ev-cpu", "ev-gpu"],
                "missing_evidence_kinds": [],
            }
        )
        kernel.update(
            {
                "disposition": "rejected",
                "oppose_evidence_ids": ["ev-edge"],
                "missing_evidence_kinds": [],
            }
        )
        hypotheses = self.hypotheses(value)
        external = {
            "status": "completed",
            "providers_requested": ["google-ai-mode"],
            "providers_completed": ["google-ai-mode"],
            "failed_providers": [],
            "total_wait_seconds": 1.0,
            "reviews": [
                {
                    "provider": "google-ai-mode",
                    "status": "completed",
                    "response": {
                        "verdict": "challenge",
                        "concerns": ["Check whether launch gaps are causal."],
                        "suggested_experiments": [],
                    },
                }
            ],
        }

        result = self.decide(
            self.model(), hypotheses, self.selection(hypotheses), external_review=external
        )

        self.assertEqual(result["decision"], "PURSUE")
        adjudication = result["external_challenge"]["local_adjudication"]
        self.assertEqual(
            adjudication["status"], "retained_for_candidate_validation"
        )
        self.assertEqual(adjudication["evidence_ids"], [])


if __name__ == "__main__":
    unittest.main()
