from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from tests import test_diagnostic_knowledge as knowledge_helpers


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "skills/cuda-kernel-optimizer/references"
FIXTURE_PATH = (
    ROOT / "tests/fixtures/knowledge_coverage/cross_layer_cases.json"
)
GENERAL_MECHANISMS = {
    "global_memory_transactions",
    "redundant_dram_traffic",
    "memory_latency_hiding",
    "register_or_shared_pressure",
    "parallelism_or_wave_tail",
    "compute_pipeline_or_dtype",
    "synchronization_or_atomic_contention",
    "framework_launch_fragmentation",
    "host_device_transfer_serialization",
    "cpu_or_data_pipeline_starvation",
    "collective_wait_or_rank_skew",
    "serving_scheduling_or_request_path",
}
STACK_FAMILIES = {
    "cuda_kernel",
    "cutlass_cute",
    "triton",
    "pytorch",
    "serving",
    "nccl",
}
OFFICIAL_REPOSITORIES = {
    "NVIDIA/cuda-samples",
    "NVIDIA/cutlass",
    "triton-lang/triton",
    "pytorch/tutorials",
    "vllm-project/vllm",
    "NVIDIA/nccl-tests",
}
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class KnowledgeCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.cases = cls.fixture["cases"]
        cls.cards_document = json.loads(
            (REFERENCE_DIR / "diagnostic_cards.json").read_text(encoding="utf-8")
        )
        cls.cards_by_mechanism = {
            card["mechanism_key"]: card
            for card in cls.cards_document["cards"]
        }
        cls.actions_document = json.loads(
            (REFERENCE_DIR / "evidence_action_catalog.json").read_text(
                encoding="utf-8"
            )
        )
        cls.actions = {
            action["action_id"]: action
            for action in cls.actions_document["actions"]
        }
        evidence_module = knowledge_helpers._load_script(
            "diagnostic_evidence.py",
            "knowledge_coverage_evidence_contract",
        )
        cls.producer_contract = evidence_module.semantic_producer_contract()
        cls.hypothesis_module = knowledge_helpers._load_script(
            "hypothesis_space.py",
            "knowledge_coverage_hypothesis_keys",
        )

        cls.temporary = tempfile.TemporaryDirectory()
        cls.reference_dir = Path(cls.temporary.name) / "references"
        shutil.copytree(REFERENCE_DIR, cls.reference_dir)
        case_path = cls.reference_dir / "case_memory.json"
        case_memory = json.loads(case_path.read_text(encoding="utf-8"))
        case_memory["cases"] = []
        knowledge_helpers._write_json(case_path, case_memory)
        card_path = cls.reference_dir / "diagnostic_cards.json"
        cards = json.loads(card_path.read_text(encoding="utf-8"))
        for card in cards["cards"]:
            card["case_ids"] = []
        knowledge_helpers._write_json(card_path, cards)

        cls.module = knowledge_helpers._load()
        cls.module.REFERENCE_DIR = cls.reference_dir
        cls.module.CARDS_PATH = card_path
        cls.query_module = knowledge_helpers._load_script(
            "knowledge_query.py",
            "knowledge_coverage_frozen_query",
        )
        cls.query_module._load_diagnostic_knowledge = lambda: cls.module

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _compatible_actions(self, case: dict) -> list[str]:
        card = self.cards_by_mechanism[case["mechanism_key"]]
        return list(
            dict.fromkeys(
                [
                    case["action_id"],
                    *card["preferred_actions"],
                    card["cheapest_falsifier"]["action_id"],
                ]
            )
        )

    def _producer_action(self, case: dict, observation: dict) -> str:
        for action_id in self._compatible_actions(case):
            contract = self.producer_contract.get(action_id)
            if contract is None:
                continue
            evidence_kind = contract["evidence_kind"]
            tool_contract = self.module._ACTIVE_EVIDENCE_TOOLS.get(evidence_kind)
            if (
                observation["semantic_id"] in contract["derived_semantic_ids"]
                and tool_contract is not None
                and observation["tool"]["name"] in tool_contract[0]
            ):
                return action_id
        self.fail(
            f"{case['id']} semantic {observation['semantic_id']} has no "
            "compatible producer/action"
        )

    def _frozen(
        self,
        case: dict,
        observations: list[dict],
        *,
        variant: str,
        cheapest_action_available: bool = True,
    ) -> dict:
        frozen = knowledge_helpers._frozen_inputs()
        frozen["diagnostic_evidence"] = []
        producer_action = (
            self._producer_action(case, observations[0])
            if observations
            else None
        )
        contract_actions = set(frozen["contract_action_ids"])
        if cheapest_action_available:
            contract_actions.add(case["action_id"])
        else:
            contract_actions.discard(case["action_id"])
        if producer_action is not None and producer_action != case["action_id"]:
            contract_actions.add(producer_action)
        ready = set(frozen["ready_capability_ids"])
        for action_id in contract_actions:
            ready.update(self.actions[action_id]["required_capability_ids"])
        frozen["contract_action_ids"] = sorted(contract_actions)
        frozen["ready_capability_ids"] = sorted(ready)
        frozen["available_actions"] = sorted(
            action_id
            for action_id, action in self.actions.items()
            if action_id in contract_actions
            and action["control_scope"] == "read_only"
            and set(action["required_capability_ids"]).issubset(ready)
        )

        for observation in observations:
            tool = observation["tool"]
            if tool["name"] in {"ncu", "nsys"}:
                frozen["knowledge_identity"]["profiler_versions"][
                    tool["name"]
                ] = knowledge_helpers._identity_fact(tool["version"])
            elif tool["name"] == "pytorch":
                frozen["knowledge_identity"]["framework_versions"][
                    "pytorch"
                ] = knowledge_helpers._identity_fact(tool["version"])

        if observations:
            contract = self.producer_contract[producer_action]
            frozen["active_evidence_results"] = [
                {
                    "action_id": producer_action,
                    "evidence_kind": contract["evidence_kind"],
                    "adapter_implementation_sha256": (
                        knowledge_helpers.IMPLEMENTATION_SHA
                    ),
                    "result_sha256": hashlib.sha256(
                        f"{case['id']}:{variant}".encode("utf-8")
                    ).hexdigest(),
                    "status": "observed",
                    "observations": {
                        "semantic_observations": copy.deepcopy(observations)
                    },
                }
            ]
        else:
            frozen["active_evidence_results"] = []
        return frozen

    def _has_result(
        self,
        context: dict,
        collection: str,
        case: dict,
        reason: str,
    ) -> bool:
        card_id = self.cards_by_mechanism[case["mechanism_key"]]["id"]
        return any(
            item["card_id"] == card_id and item["reason"] == reason
            for item in context[collection]
        )

    def test_fixture_is_offline_pinned_and_covers_all_mechanisms_and_stacks(
        self,
    ) -> None:
        self.assertEqual(
            self.fixture["schema_version"],
            "cuda-optimizer/knowledge-coverage-fixture-v1",
        )
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(
            {case["mechanism_key"] for case in self.cases},
            GENERAL_MECHANISMS,
        )
        self.assertEqual(
            {case["stack_family"] for case in self.cases},
            STACK_FAMILIES,
        )
        self.assertEqual(len({case["id"] for case in self.cases}), 12)
        self.assertEqual(len({case["public_path"] for case in self.cases}), 6)
        self.assertEqual(len({case["source_commit"] for case in self.cases}), 6)
        self.assertEqual(len({case["content_sha256"] for case in self.cases}), 6)

        path_digests: dict[str, str] = {}
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                repo_commit, path = case["public_path"].split(":", 1)
                repository, pinned_commit = repo_commit.rsplit("@", 1)
                self.assertIn(repository, OFFICIAL_REPOSITORIES)
                self.assertTrue(path)
                self.assertEqual(pinned_commit, case["source_commit"])
                self.assertRegex(case["source_commit"], COMMIT)
                self.assertRegex(case["content_sha256"], SHA256)
                self.assertGreater(len(set(case["source_commit"])), 8)
                self.assertGreater(len(set(case["content_sha256"])), 8)
                self.assertIsNone(case["migrated_from"])
                self.assertEqual(
                    case["action_id"],
                    self.cards_by_mechanism[case["mechanism_key"]][
                        "cheapest_falsifier"
                    ]["action_id"],
                )
                self.assertEqual(
                    case["evidence_kind"],
                    self.producer_contract[case["action_id"]]["evidence_kind"],
                )
                self.assertEqual(case["semantic_observations"][0]["tool"], case["tool"])
                for field in (
                    "semantic_observations",
                    "counter_observations",
                    "invalidator_observations",
                ):
                    self.assertEqual(len(case[field]), 1)
                    self.assertEqual(case[field][0]["quality"], "validated")
                previous = path_digests.setdefault(
                    case["public_path"], case["content_sha256"]
                )
                self.assertEqual(previous, case["content_sha256"])

        serialized = json.dumps(self.fixture, sort_keys=True).lower()
        for forbidden in ("rtx", "5090", "winner", "speedup"):
            self.assertNotIn(forbidden, serialized)

    def test_empty_case_memory_admits_only_source_verified_general_cards(
        self,
    ) -> None:
        result = self.module.validate_knowledge_package(self.reference_dir)
        expected = {
            card["id"]
            for card in self.cards_document["cards"]
            if card["content_status"] == "source_verified"
            and card["observation_rules"]["positive"]
        }

        self.assertEqual(result["case_count"], 0)
        self.assertEqual(set(result["runtime_candidate_card_ids"]), expected)

    def test_nonempty_case_memory_remains_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference_dir = Path(temporary) / "references"
            shutil.copytree(REFERENCE_DIR, reference_dir)
            path = reference_dir / "case_memory.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cases"].append(copy.deepcopy(payload["cases"][0]))
            knowledge_helpers._write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "duplicate case memory"):
                self.module.validate_knowledge_package(reference_dir)

    def test_default_package_retains_5090_cases(self) -> None:
        package = knowledge_helpers._load().validate_knowledge_package(
            REFERENCE_DIR
        )
        cards = self.cards_document["cards"]
        retained = [
            card
            for card in cards
            if card["id"].startswith("diagnostic.triton.")
            and card["case_ids"]
        ]

        self.assertEqual(package["case_count"], 15)
        self.assertEqual(len(retained), 6)

    def test_every_source_verified_rule_has_a_compatible_producer_action(
        self,
    ) -> None:
        for card in self.cards_document["cards"]:
            if card["content_status"] != "source_verified":
                continue
            compatible_actions = set(card["preferred_actions"])
            compatible_actions.add(card["cheapest_falsifier"]["action_id"])
            for group in ("positive", "counter", "invalidators"):
                for rule in card["observation_rules"][group]:
                    producers = [
                        action_id
                        for action_id in compatible_actions
                        if action_id in self.producer_contract
                        and rule["semantic_id"]
                        in self.producer_contract[action_id][
                            "derived_semantic_ids"
                        ]
                    ]
                    with self.subTest(
                        card_id=card["id"],
                        group=group,
                        semantic_id=rule["semantic_id"],
                    ):
                        self.assertTrue(producers)

    def test_trusted_positive_routes_only_expected_mechanism(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                observations = case["semantic_observations"]
                context = self.module.build_knowledge_context(
                    self._frozen(case, observations, variant="positive"),
                    limit=3,
                )
                self.assertEqual(
                    [item["mechanism_key"] for item in context["candidates"]],
                    [
                        self.hypothesis_module.canonical_mechanism_key(
                            case["mechanism_key"]
                        )
                    ],
                )
                candidate = context["candidates"][0]
                expected_card = self.cards_by_mechanism[case["mechanism_key"]]
                self.assertEqual(candidate["card_id"], expected_card["id"])
                self.assertEqual(
                    [item["semantic_id"] for item in candidate["evidence"]["positive"]],
                    [observations[0]["semantic_id"]],
                )
                self.assertEqual(
                    candidate["cheapest_falsifier"]["action_id"],
                    case["action_id"],
                )
                self.assertEqual(candidate["confidence"], "inconclusive")
                self.assertEqual(candidate["promotion_authority"], "none")
                self.assertEqual(candidate["case_ids"], [])
                self.assertEqual(
                    candidate["case_support"],
                    {"exact": [], "analogous": []},
                )
                self.assertEqual(context["promotion_authority"], "none")

    def test_trusted_counter_never_becomes_positive_or_candidate(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                context = self.module.build_knowledge_context(
                    self._frozen(
                        case,
                        case["counter_observations"],
                        variant="counter",
                    ),
                    limit=3,
                )
                self.assertEqual(context["candidates"], [])
                self.assertTrue(
                    self._has_result(
                        context,
                        "explanations",
                        case,
                        "counter_observed",
                    )
                )

    def test_trusted_invalidator_directly_rejects(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                context = self.module.build_knowledge_context(
                    self._frozen(
                        case,
                        case["invalidator_observations"],
                        variant="invalidator",
                    ),
                    limit=3,
                )
                self.assertEqual(context["candidates"], [])
                self.assertTrue(
                    self._has_result(
                        context,
                        "rejections",
                        case,
                        "invalidator_observed",
                    )
                )

    def test_unavailable_cheapest_action_never_produces_candidate(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case["id"]):
                context = self.module.build_knowledge_context(
                    self._frozen(
                        case,
                        case["semantic_observations"],
                        variant="action-unavailable",
                        cheapest_action_available=False,
                    ),
                    limit=3,
                )
                self.assertEqual(context["candidates"], [])
                self.assertTrue(
                    self._has_result(
                        context,
                        "explanations",
                        case,
                        "read_only_action_unavailable",
                    )
                )

    def test_no_match_returns_explanations_without_space_claim(self) -> None:
        case = self.cases[0]
        context = self.module.build_knowledge_context(
            self._frozen(case, [], variant="no-match"),
            limit=3,
        )

        self.assertEqual(context["candidates"], [])
        self.assertTrue(context["explanations"])
        serialized = json.dumps(context, sort_keys=True).lower()
        self.assertNotIn("no optimization space", serialized)
        self.assertNotIn("no_optimization_space", serialized)

    def test_query_frozen_preserves_task4_trust_gate(self) -> None:
        case = self.cases[0]
        observation = copy.deepcopy(case["semantic_observations"][0])
        observation["quality"] = "heuristic"
        frozen = self._frozen(case, [observation], variant="untrusted-direct")

        context = self.query_module.query_frozen(frozen, limit=3)

        self.assertEqual(context["candidates"], [])
        self.assertTrue(
            self._has_result(
                context,
                "explanations",
                case,
                "quality_untrusted",
            )
        )


if __name__ == "__main__":
    unittest.main()
