from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "skills/cuda-kernel-optimizer/scripts/adaptive_investment.py"


def _load():
    name = "adaptive_investment_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def direction_fixture(
    direction_id: str = "z",
    *,
    lower: float = 0.0,
    upper: float = 4.0,
    status: str = "candidate",
    mechanism: str = "mechanism-z",
    stale: bool = False,
) -> dict:
    return {
        "direction_id": direction_id,
        "status": status,
        "mechanism": mechanism,
        "benefit": {
            "lower": lower,
            "upper": upper,
            "basis": "measured-bound",
            "identity_digest": "a" * 64,
            "stale": stale,
        },
    }


def evidence_action(
    action_id: str,
    *,
    direction: str,
    mechanism: str | None = None,
    kind: str = "check",
    cost: str = "low",
    perturbation: str = "low",
    risk: str = "none",
    p90_seconds: float = 1.0,
    outcomes: list[dict] | None = None,
) -> dict:
    return {
        "action_id": action_id,
        "kind": kind,
        "mechanism": mechanism or f"evidence-{action_id}",
        "cost": cost,
        "perturbation": perturbation,
        "risk": risk,
        "p90_seconds": p90_seconds,
        "target_direction_ids": [direction],
        "outcomes": outcomes
        if outcomes is not None
        else [
            {"outcome_id": f"{action_id}-supports", "supports": [direction], "opposes": []},
            {"outcome_id": f"{action_id}-opposes", "supports": [], "opposes": [direction]},
        ],
    }


def candidate_action(action_id: str, **kwargs) -> dict:
    return evidence_action(action_id, direction="z", **kwargs)


def authorization(*, max_seconds: float = 3600.0) -> dict:
    return {"max_seconds": max_seconds}


def spend(*, elapsed_seconds: float = 0.0) -> dict:
    return {"elapsed_seconds": elapsed_seconds}


class AdaptiveInvestmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load()

    def decide(self, directions, actions, *, auth=None, current_spend=None, minimum_effect=1.0):
        return self.module.decide_next_action(
            directions,
            actions,
            authorization=authorization() if auth is None else auth,
            spend=spend() if current_spend is None else current_spend,
            minimum_effect=minimum_effect,
        )

    def test_external_gain_cannot_change_local_bound(self):
        direction = direction_fixture(lower=0.0, upper=4.0)
        direction["external_claim_pct"] = 20.0
        result = self.decide(
            [direction], [evidence_action("check-z", direction="z")],
            auth=authorization(max_seconds=100), current_spend=spend(elapsed_seconds=0),
        )
        self.assertEqual(result["portfolio"][0]["benefit"]["upper"], 4.0)

    def test_cumulative_small_actions_cannot_bypass_authorization(self):
        result = self.decide(
            [direction_fixture(status="supported")],
            [candidate_action("next", p90_seconds=2.0)],
            auth=authorization(max_seconds=100.0), current_spend=spend(elapsed_seconds=99.0),
        )
        self.assertEqual(result["decision"], "REVIEW_REQUIRED")
        self.assertEqual(result["blocked_action"]["action_id"], "next")
        self.assertEqual(result["projected_spend"]["p90_seconds"], 101.0)

    def test_fifty_small_actions_cannot_bypass_cumulative_authorization(self):
        current_spend = spend()
        for index in range(1, 51):
            action_id = f"renamed-small-action-{index}"
            result = self.decide(
                [direction_fixture()],
                [evidence_action(action_id, direction="z", p90_seconds=1.0)],
                auth=authorization(max_seconds=49.5),
                current_spend=current_spend,
            )
            if index < 50:
                self.assertEqual(result["decision"], "ACT")
                self.assertEqual(result["selected_action"]["action_id"], action_id)
                current_spend = {
                    "elapsed_seconds": result["projected_spend"]["p90_seconds"]
                }
                continue

            self.assertEqual(current_spend["elapsed_seconds"], 49.0)
            self.assertEqual(result["decision"], "REVIEW_REQUIRED")
            self.assertEqual(result["blocked_action"]["action_id"], action_id)
            self.assertEqual(result["projected_spend"]["p90_seconds"], 50.0)

    def test_no_decision_changing_outcome_means_action_is_skipped(self):
        action = evidence_action("repeat", direction="z")
        action["outcomes"] = [
            {"outcome_id": "same-a", "supports": [], "opposes": []},
            {"outcome_id": "same-b", "supports": [], "opposes": []},
        ]
        result = self.decide([direction_fixture()], [action])
        self.assertEqual(result["decision"], "STOP")
        self.assertEqual(result["reason"], "no_decision_changing_action")

    def test_low_upper_direction_is_closed(self):
        result = self.decide(
            [direction_fixture(upper=0.99)],
            [evidence_action("check-z", direction="z")],
        )
        self.assertEqual(result["decision"], "STOP")
        self.assertEqual(result["portfolio"][0]["status"], "closed")
        self.assertEqual(result["portfolio"][0]["reason"], "below_minimum_effect")

    def test_exact_mechanism_repeat_direction_is_closed(self):
        result = self.decide(
            [
                direction_fixture("a", mechanism="same"),
                direction_fixture("b", mechanism="same"),
            ],
            [evidence_action("check-a", direction="a")],
        )
        portfolio = {item["direction_id"]: item for item in result["portfolio"]}
        self.assertEqual(portfolio["a"]["status"], "candidate")
        self.assertEqual(portfolio["b"]["status"], "closed")
        self.assertEqual(portfolio["b"]["reason"], "exact_mechanism_repeat")

    def test_stale_direction_only_allows_refresh(self):
        stale = direction_fixture(stale=True)
        result = self.decide(
            [stale],
            [
                evidence_action("check", direction="z", kind="check"),
                evidence_action("refresh", direction="z", kind="refresh"),
            ],
        )
        self.assertEqual(result["decision"], "ACT")
        self.assertEqual(result["selected_action"]["action_id"], "refresh")
        self.assertIn("check", result["skipped_actions"])

    def test_cheaper_action_covering_more_directions_dominates_expensive_action(self):
        directions = [
            direction_fixture("a", mechanism="mechanism-a"),
            direction_fixture("b", mechanism="mechanism-b"),
        ]
        result = self.decide(
            directions,
            [
                evidence_action(
                    "expensive",
                    direction="a",
                    mechanism="shared-evidence",
                    cost="high",
                    p90_seconds=10.0,
                ),
                {
                    **evidence_action(
                        "cheap",
                        direction="a",
                        mechanism="shared-evidence",
                        cost="low",
                        p90_seconds=1.0,
                    ),
                    "target_direction_ids": ["a", "b"],
                    "outcomes": [
                        {"outcome_id": "cheap-support", "supports": ["a", "b"], "opposes": []},
                        {"outcome_id": "cheap-oppose", "supports": [], "opposes": ["a", "b"]},
                    ],
                },
            ],
        )
        self.assertEqual(result["selected_action"]["action_id"], "cheap")
        self.assertIn("expensive", result["skipped_actions"])

    def test_cheaper_different_mechanism_evidence_does_not_dominate_fallback(self):
        result = self.decide(
            [direction_fixture()],
            [
                evidence_action("cheap-trace", direction="z", mechanism="trace", cost="low"),
                evidence_action("ncu-fallback", direction="z", mechanism="ncu", cost="high"),
            ],
        )
        self.assertEqual(result["selected_action"]["action_id"], "cheap-trace")
        self.assertNotIn("ncu-fallback", result["skipped_actions"])

    def test_different_mechanism_fallback_survives_failed_preferred_implementation(self):
        failed = evidence_action("preferred", direction="z", mechanism="ncu")
        failed["implementation_status"] = "failed"
        fallback = evidence_action("fallback", direction="z", mechanism="trace")
        result = self.decide([direction_fixture()], [failed, fallback])
        self.assertEqual(result["decision"], "ACT")
        self.assertEqual(result["selected_action"]["action_id"], "fallback")
        self.assertIn("preferred", result["skipped_actions"])

    def test_failed_same_mechanism_action_cannot_close_direction_before_fallback(self):
        failed = evidence_action("preferred", direction="z", mechanism="mechanism-z")
        failed["implementation_status"] = "failed"
        fallback = evidence_action("fallback", direction="z", mechanism="trace")
        result = self.decide([direction_fixture()], [failed, fallback])
        self.assertEqual(result["decision"], "ACT")
        self.assertEqual(result["selected_action"]["action_id"], "fallback")
        self.assertIn("preferred", result["skipped_actions"])

    def test_outcomes_cannot_reference_stale_non_target_direction(self):
        action = evidence_action("check-a", direction="a")
        action["outcomes"] = [
            {"outcome_id": "stale-b", "supports": ["b"], "opposes": []},
        ]
        with self.assertRaisesRegex(self.module.ValidationError, "target_direction_ids"):
            self.decide(
                [
                    direction_fixture("a", mechanism="mechanism-a"),
                    direction_fixture("b", mechanism="mechanism-b", stale=True),
                ],
                [action],
            )

    def test_outcomes_cannot_reference_closed_non_target_direction(self):
        action = evidence_action("check-a", direction="a")
        action["outcomes"] = [
            {"outcome_id": "closed-b", "supports": [], "opposes": ["b"]},
        ]
        with self.assertRaisesRegex(self.module.ValidationError, "target_direction_ids"):
            self.decide(
                [
                    direction_fixture("a", mechanism="mechanism-a"),
                    direction_fixture("b", mechanism="mechanism-b", upper=0.5),
                ],
                [action],
            )

    def test_same_checkpoint_replay_is_deterministic(self):
        directions = [
            direction_fixture("a", mechanism="mechanism-a"),
            direction_fixture("b", mechanism="mechanism-b"),
        ]
        actions = [
            evidence_action("b-action", direction="b"),
            evidence_action("a-action", direction="a"),
        ]
        first = self.decide(copy.deepcopy(directions), copy.deepcopy(actions))
        second = self.decide(copy.deepcopy(directions), copy.deepcopy(actions))
        self.assertEqual(first, second)
        self.assertEqual(first["next_checkpoint"], "a-action")


if __name__ == "__main__":
    unittest.main()
