from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

from tests.test_analysis_epoch import epoch_fixture
from tests.test_evidence_selector import (
    catalog_fixture,
    policy_fixture,
    request_fixture,
)
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


def _coverage(value: dict, layer: str, status: str = "observed") -> None:
    item = next(entry for entry in value["coverage"] if entry["layer"] == layer)
    item.update(
        {
            "status": status,
            "reason": None if status == "observed" else "not captured by fixture",
        }
    )


def _node(
    node_id: str,
    layer: str,
    lane: str,
    start: float,
    end: float,
    *,
    attribution: str = "explained",
) -> dict:
    return {
        "node_id": node_id,
        "layer": layer,
        "lane": lane,
        "kind": f"{layer}_work",
        "label": node_id,
        "duration_us": end - start,
        "occurrences": 1,
        "timing_status": "observed",
        "first_start_us": start,
        "last_end_us": end,
        "attribution_status": attribution,
        "evidence_ids": ["ev-gpu"],
    }


def scenario_maps(map_module) -> dict[str, dict]:
    kernel = map_fixture(map_module)
    kernel["nodes"][0].update(
        {"duration_us": 80.0, "last_end_us": 80.0, "occurrences": 1}
    )
    kernel["nodes"][1].update(
        {
            "duration_us": 900.0,
            "occurrences": 1,
            "first_start_us": 100.0,
            "last_end_us": 1000.0,
        }
    )

    framework = map_fixture(map_module)
    _coverage(framework, "framework")
    framework["nodes"].append(
        _node("framework-gap", "framework", "python-main", 0.0, 300.0)
    )
    framework["edges"].append(
        {
            "source": "framework-gap",
            "target": "gpu-kernel",
            "relation": "precedes",
            "overlap_us": None,
            "evidence_ids": ["ev-edge"],
        }
    )
    framework["hot_path"] = ["framework-gap", "gpu-kernel"]

    transfer = map_fixture(map_module)
    _coverage(transfer, "transfer")
    transfer["nodes"].append(
        _node("h2d-copy", "transfer", "copy-stream-0", 200.0, 400.0)
    )
    transfer["edges"].append(
        {
            "source": "gpu-kernel",
            "target": "h2d-copy",
            "relation": "overlaps",
            "overlap_us": 200.0,
            "evidence_ids": ["ev-edge"],
        }
    )

    unknown_idle = map_fixture(map_module)
    _coverage(unknown_idle, "idle")
    unknown_idle["nodes"].append(
        _node(
            "idle-gap",
            "idle",
            "gpu-0",
            900.0,
            950.0,
            attribution="unexplained",
        )
    )
    unknown_idle["uncovered_intervals"] = [
        {"start_us": 900.0, "end_us": 950.0, "reason": "unknown GPU idle"}
    ]
    unknown_idle["conclusion_level"] = "inconclusive"

    mixed = map_fixture(map_module)
    _coverage(mixed, "transfer")
    mixed["nodes"][0].update(
        {"duration_us": 400.0, "occurrences": 1, "last_end_us": 400.0}
    )
    mixed["nodes"][1].update(
        {
            "duration_us": 500.0,
            "occurrences": 1,
            "first_start_us": 300.0,
            "last_end_us": 800.0,
        }
    )
    mixed["nodes"].append(
        _node("mixed-copy", "transfer", "copy-stream-0", 650.0, 850.0)
    )
    mixed["edges"].append(
        {
            "source": "gpu-kernel",
            "target": "mixed-copy",
            "relation": "overlaps",
            "overlap_us": 150.0,
            "evidence_ids": ["ev-edge"],
        }
    )
    mixed["hot_path"] = ["cpu-launch", "gpu-kernel", "mixed-copy"]

    return {
        "kernel_hot_path": kernel,
        "framework_gap": framework,
        "transfer_overlap": transfer,
        "unknown_idle": unknown_idle,
        "mixed": mixed,
    }


class ActiveDiagnosisVerticalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.map_module = _load("execution_map.py", "vertical_execution_map")
        self.hypothesis_module = _load(
            "hypothesis_space.py", "vertical_hypothesis_space"
        )
        self.selector_module = _load(
            "evidence_selector.py", "vertical_evidence_selector"
        )
        self.model_module = _load(
            "performance_model.py", "vertical_performance_model"
        )
        self.decision_module = _load(
            "diagnostic_decision.py", "vertical_diagnostic_decision"
        )
        self.epoch = epoch_fixture()
        self.evidence = evidence_catalog()

    def test_five_cpu_scenarios_stay_compact_and_preserve_gaps(self) -> None:
        expected_unmodeled = {"unknown_idle"}
        for name, execution_map in scenario_maps(self.map_module).items():
            with self.subTest(name=name):
                result = self.map_module.validate_execution_map(
                    execution_map,
                    epoch=self.epoch,
                    evidence_catalog=self.evidence,
                )
                self.assertEqual(
                    result["requires_unmodeled_hypothesis"],
                    name in expected_unmodeled,
                )
                compact_bytes = len(
                    json.dumps(
                        result["execution_map"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                self.assertLess(compact_bytes, 64 * 1024)

    def test_execution_map_ablation_fails_closed(self) -> None:
        hypothesis = hypothesis_fixture(self.hypothesis_module, self.map_module)
        with self.assertRaisesRegex(ValueError, "execution_map"):
            self.hypothesis_module.validate_hypothesis_set(
                hypothesis,
                epoch=self.epoch,
                execution_map=None,
                evidence_catalog=self.evidence,
            )

    def test_relationship_ablation_loses_pair_discrimination(self) -> None:
        execution_map = map_fixture(self.map_module)
        hypothesis = hypothesis_fixture(self.hypothesis_module, self.map_module)
        admitted = self.hypothesis_module.validate_hypothesis_set(
            hypothesis,
            epoch=self.epoch,
            execution_map=execution_map,
            evidence_catalog=self.evidence,
        )
        request = request_fixture()
        request["epoch_sha256"] = self.map_module.epoch_digest(self.epoch)
        request["hypothesis_set_sha256"] = admitted["hypothesis_set_sha256"]
        request["requests"] = [request["requests"][0]]
        selected = self.selector_module.select_evidence_request(
            request,
            epoch=self.epoch,
            execution_map=execution_map,
            hypothesis_result=admitted,
            evidence_catalog=self.evidence,
            action_catalog=catalog_fixture(),
            policy=policy_fixture(),
            request_history=[],
        )
        self.assertEqual(
            selected["selected_request"]["discrimination"]["exclusive_pair_count"],
            1,
        )

        ablated_hypothesis = copy.deepcopy(hypothesis)
        ablated_hypothesis["relationships"] = []
        ablated = self.hypothesis_module.validate_hypothesis_set(
            ablated_hypothesis,
            epoch=self.epoch,
            execution_map=execution_map,
            evidence_catalog=self.evidence,
        )
        ablated_request = copy.deepcopy(request)
        ablated_request["hypothesis_set_sha256"] = ablated[
            "hypothesis_set_sha256"
        ]
        ablated_request["requests"][0]["exclusive_pairs"] = []
        selected_without_relationships = self.selector_module.select_evidence_request(
            ablated_request,
            epoch=self.epoch,
            execution_map=execution_map,
            hypothesis_result=ablated,
            evidence_catalog=self.evidence,
            action_catalog=catalog_fixture(),
            policy=policy_fixture(),
            request_history=[],
        )
        self.assertEqual(
            selected_without_relationships["selected_request"]["discrimination"][
                "exclusive_pair_count"
            ],
            0,
        )

    def test_request_history_ablation_reintroduces_duplicate_profile(self) -> None:
        execution_map = map_fixture(self.map_module)
        hypothesis = hypothesis_fixture(self.hypothesis_module, self.map_module)
        admitted = self.hypothesis_module.validate_hypothesis_set(
            hypothesis,
            epoch=self.epoch,
            execution_map=execution_map,
            evidence_catalog=self.evidence,
        )
        request = request_fixture()
        request["epoch_sha256"] = self.map_module.epoch_digest(self.epoch)
        request["hypothesis_set_sha256"] = admitted["hypothesis_set_sha256"]
        request["requests"] = [request["requests"][0]]

        def select(history):
            return self.selector_module.select_evidence_request(
                request,
                epoch=self.epoch,
                execution_map=execution_map,
                hypothesis_result=admitted,
                evidence_catalog=self.evidence,
                action_catalog=catalog_fixture(),
                policy=policy_fixture(),
                request_history=history,
            )

        first = select([])
        signature = first["selected_request"]["request_signature"]
        self.assertEqual(select([signature])["status"], "evidence_gap")
        self.assertEqual(select([])["status"], "selected")

    def test_four_controlled_mechanisms_choose_layer_appropriate_next_action(self) -> None:
        scenarios = []

        launch = map_fixture(self.map_module)
        launch["nodes"][0].update(
            {"duration_us": 700.0, "first_start_us": 0.0, "last_end_us": 700.0}
        )
        launch["nodes"][1].update(
            {"duration_us": 200.0, "first_start_us": 800.0, "last_end_us": 1000.0}
        )
        scenarios.append(
            (
                "launch_graph",
                launch,
                "cuda_graph_launch_batching",
                "runtime",
                ["cpu-launch"],
                "framework-targeted",
                "framework_trace",
            )
        )

        memory = map_fixture(self.map_module)
        memory["nodes"][0].update(
            {"duration_us": 50.0, "first_start_us": 0.0, "last_end_us": 50.0}
        )
        memory["nodes"][1].update(
            {"duration_us": 900.0, "first_start_us": 100.0, "last_end_us": 1000.0}
        )
        scenarios.append(
            (
                "memory_coalescing",
                memory,
                "memory_coalescing",
                "kernel",
                ["gpu-kernel"],
                "ncu-targeted",
                "ncu_kernel",
            )
        )

        compute = copy.deepcopy(memory)
        compute["map_id"] = "map-compute"
        compute["nodes"][1]["label"] = "gemm-mainloop"
        scenarios.append(
            (
                "compute_gemm",
                compute,
                "gemm_tile_occupancy",
                "kernel",
                ["gpu-kernel"],
                "ncu-targeted",
                "ncu_kernel",
            )
        )

        transfer = map_fixture(self.map_module)
        _coverage(transfer, "transfer")
        transfer["nodes"].append(
            _node("serialized-copy", "transfer", "copy-stream-0", 200.0, 800.0)
        )
        transfer["edges"].append(
            {
                "source": "gpu-kernel",
                "target": "serialized-copy",
                "relation": "precedes",
                "overlap_us": None,
                "evidence_ids": ["ev-edge"],
            }
        )
        transfer["hot_path"] = ["cpu-launch", "serialized-copy", "gpu-kernel"]
        scenarios.append(
            (
                "transfer_overlap",
                transfer,
                "async_transfer_overlap",
                "runtime",
                ["serialized-copy"],
                "direction-experiment",
                "direction_experiment",
            )
        )

        for (
            name,
            execution_map,
            mechanism,
            claim_layer,
            scope,
            action_id,
            evidence_kind,
        ) in scenarios:
            with self.subTest(name=name):
                execution_map = self.map_module.validate_execution_map(
                    execution_map,
                    epoch=self.epoch,
                    evidence_catalog=self.evidence,
                )["execution_map"]
                hypothesis = {
                    "schema_version": "cuda-optimizer/hypothesis-set-v1",
                    "set_id": f"hypotheses-{name}",
                    "epoch_id": self.epoch["epoch_id"],
                    "epoch_sha256": self.map_module.epoch_digest(self.epoch),
                    "execution_map_sha256": self.map_module.execution_map_digest(
                        execution_map,
                        epoch=self.epoch,
                        evidence_catalog=self.evidence,
                    ),
                    "hypotheses": [
                        {
                            "hypothesis_id": f"h-{name}",
                            "kind": "mechanism",
                            "scope_node_ids": scope,
                            "statement": f"Controlled {name} mechanism dominates the path.",
                            "mechanism": mechanism,
                            "claim_layer": claim_layer,
                            "disposition": "active",
                            "confidence": "inconclusive",
                            "support_evidence_ids": [],
                            "oppose_evidence_ids": [],
                            "missing_evidence_kinds": [evidence_kind],
                            "falsification_question": f"Can {evidence_kind} falsify {mechanism}?",
                        }
                    ],
                    "relationships": [],
                }
                hypothesis_result = self.hypothesis_module.validate_hypothesis_set(
                    hypothesis,
                    epoch=self.epoch,
                    execution_map=execution_map,
                    evidence_catalog=self.evidence,
                )
                action_catalog = catalog_fixture()
                if action_id == "direction-experiment":
                    action_catalog["actions"].append(
                        {
                            "action_id": action_id,
                            "evidence_kind": evidence_kind,
                            "required_capability_ids": ["workload.smoke"],
                            "cost": "medium",
                            "perturbation": "low",
                            "risk": "low",
                            "control_scope": "project_copy",
                            "repeatable": True,
                        }
                    )
                request = {
                    "schema_version": "cuda-optimizer/evidence-request-set-v1",
                    "request_set_id": f"requests-{name}",
                    "epoch_id": self.epoch["epoch_id"],
                    "epoch_sha256": self.map_module.epoch_digest(self.epoch),
                    "hypothesis_set_sha256": hypothesis_result["hypothesis_set_sha256"],
                    "requests": [
                        {
                            "request_id": f"request-{name}",
                            "action_id": action_id,
                            "question": f"Does {evidence_kind} falsify {mechanism}?",
                            "target_hypothesis_ids": [f"h-{name}"],
                            "exclusive_pairs": [],
                            "outcomes": [
                                {
                                    "outcome_id": "mechanism-present",
                                    "supports": [f"h-{name}"],
                                    "opposes": [],
                                },
                                {
                                    "outcome_id": "mechanism-absent",
                                    "supports": [],
                                    "opposes": [f"h-{name}"],
                                },
                            ],
                        }
                    ],
                }
                policy = policy_fixture()
                policy["available_capability_ids"].append("workload.smoke")
                selection = self.selector_module.select_evidence_request(
                    request,
                    epoch=self.epoch,
                    execution_map=execution_map,
                    hypothesis_result=hypothesis_result,
                    evidence_catalog=self.evidence,
                    action_catalog=action_catalog,
                    policy=policy,
                    request_history=[],
                )
                model = self.model_module.build_performance_model(
                    execution_map, minimum_effect_us=1.0
                )
                decision = self.decision_module.decide_next_step(
                    model, hypothesis_result, selection
                )

                self.assertEqual(decision["decision"], "MEASURE")
                self.assertEqual(decision["primary_diagnosis"]["mechanism"], mechanism)
                self.assertEqual(decision["primary_diagnosis"]["claim_layer"], claim_layer)
                self.assertEqual(decision["next_action"]["action_id"], action_id)
                self.assertTrue(decision["benefit_ceiling"]["qualifies"])
                if claim_layer != "kernel":
                    self.assertNotEqual(decision["next_action"]["action_id"], "ncu-targeted")


if __name__ == "__main__":
    unittest.main()
