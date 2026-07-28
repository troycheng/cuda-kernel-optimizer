from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.test_analysis_epoch import epoch_fixture
from tests.test_evidence_selector import (
    catalog_fixture,
    policy_fixture,
    request_fixture,
)
from tests.test_execution_map import evidence_catalog, map_fixture
from tests.test_hypothesis_space import hypothesis_fixture
from tests import test_workload_controller as workload_fixtures


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

    def test_explicit_controlled_spend_is_not_inflated_by_timing_history(
        self,
    ) -> None:
        authorization, spend = self.decision_module._investment_inputs(
            {
                "action_timing_estimates": {
                    "pytorch-operator-trace": {
                        "sample_count": 2,
                        "p50_seconds": 20.0,
                    }
                }
            },
            {"max_seconds": 30.0, "max_risk": "high"},
            {"elapsed_seconds": 7.0},
        )

        self.assertEqual(
            authorization,
            {"max_seconds": 30.0, "max_risk": "high"},
        )
        self.assertEqual(spend, {"elapsed_seconds": 7.0})

    def _controller_with_two_supported_directions(self, root: Path):
        helper = workload_fixtures.WorkloadRoundTests()
        helper.setUp()
        control, run_dir, project = helper._workspace(root)
        workload_fixtures._enable_v2_readiness(
            control,
            root,
            capability_ids=(
                "gpu-execute",
                "ncu.counter_access",
                "cuda.disassembler",
            ),
        )
        workload_fixtures._enable_active_diagnosis(control, root)
        ncu_adapter = project / "collect_ncu_evidence.py"
        ncu_adapter.write_text(
            "import json, os\n"
            "request = json.load(open(os.environ['CUDA_OPTIMIZER_EVIDENCE_REQUEST']))\n"
            "payload = {\n"
            " 'schema_version': 'cuda-optimizer/evidence-result-v1',\n"
            " 'request_signature': request['request_signature'],\n"
            " 'status': 'observed', 'outcome_id': 'kernel-present',\n"
            " 'observations': {'stall': 'memory'}, 'artifacts': []}\n"
            "open(os.environ['CUDA_OPTIMIZER_EVIDENCE_OUTPUT'], 'w').write(json.dumps(payload))\n",
            encoding="utf-8",
        )
        contract_path = Path(control["analysis_contract"])
        contract = json.loads(contract_path.read_text("utf-8"))
        contract["actions"].append(
            {
                "action_id": "ncu-targeted-kernel",
                "adapter_path": str(ncu_adapter),
                "adapter_sha256": hashlib.sha256(ncu_adapter.read_bytes()).hexdigest(),
                "argv": [sys.executable, str(ncu_adapter)],
                "timeout_seconds": 5,
                "cost_bound": {
                    "p50_seconds": 1.0,
                    "p90_seconds": 2.0,
                    "basis": "user_authorized_upper_bound",
                },
            }
        )
        sass_adapter = project / "collect_sass_evidence.py"
        sass_adapter.write_text(
            "import json, os\n"
            "request = json.load(open(os.environ['CUDA_OPTIMIZER_EVIDENCE_REQUEST']))\n"
            "payload = {\n"
            " 'schema_version': 'cuda-optimizer/evidence-result-v1',\n"
            " 'request_signature': request['request_signature'],\n"
            " 'status': 'observed', 'outcome_id': 'sass-refreshed',\n"
            " 'observations': {'execution_map_node_updates': [\n"
            "   {'node_id': 'gpu-kernel', 'duration_us': 850.0,\n"
            "    'first_start_us': 100.0, 'last_end_us': 950.0}\n"
            " ]}, 'artifacts': []}\n"
            "open(os.environ['CUDA_OPTIMIZER_EVIDENCE_OUTPUT'], 'w').write(json.dumps(payload))\n",
            encoding="utf-8",
        )
        contract["actions"].append(
            {
                "action_id": "compiler-sass-inspection",
                "adapter_path": str(sass_adapter),
                "adapter_sha256": hashlib.sha256(
                    sass_adapter.read_bytes()
                ).hexdigest(),
                "argv": [sys.executable, str(sass_adapter)],
                "timeout_seconds": 5,
                "cost_bound": {
                    "p50_seconds": 1.0,
                    "p90_seconds": 2.0,
                    "basis": "user_authorized_upper_bound",
                },
            }
        )
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        helper.controller.start_run(control, run_dir)
        helper._authorize_active_run(
            control,
            run_dir,
            "grant-vertical-supported-directions",
        )

        def bind(hypothesis, request):
            active = run_dir / "active_diagnosis"
            epoch = json.loads((active / "epoch.json").read_text("utf-8"))
            result = helper.controller._load_hypothesis_space_module().validate_hypothesis_set(
                hypothesis,
                epoch=epoch,
                execution_map=json.loads(
                    (active / "execution_map.json").read_text("utf-8")
                ),
                evidence_catalog=json.loads(
                    (active / "evidence_catalog.json").read_text("utf-8")
                ),
            )
            request["hypothesis_set_sha256"] = result["hypothesis_set_sha256"]
            return hypothesis, request

        hypothesis, request = helper._active_proposal(run_dir)
        hypothesis["relationships"] = []
        framework, kernel = hypothesis["hypotheses"]
        kernel.update(
            {
                "confidence": "plausible",
                "support_evidence_ids": ["ev-global-scan"],
            }
        )
        request["requests"] = [
            {
                "request_id": "req-framework-support",
                "action_id": "pytorch-operator-trace",
                "question": "Does the framework trace support launch overhead?",
                "target_hypothesis_ids": ["h-framework-gap"],
                "exclusive_pairs": [],
                "outcomes": [
                    {
                        "outcome_id": "gap-present",
                        "supports": ["h-framework-gap"],
                        "opposes": [],
                    },
                    {
                        "outcome_id": "gap-absent",
                        "supports": [],
                        "opposes": ["h-framework-gap"],
                    },
                ],
            }
        ]
        bind(hypothesis, request)
        helper.controller.register_active_diagnosis_proposal(
            control, run_dir, hypothesis, request
        )
        helper.controller.collect_active_diagnosis_evidence(control, run_dir)

        catalog = json.loads(
            (run_dir / "active_diagnosis" / "evidence_catalog.json").read_text(
                "utf-8"
            )
        )
        framework_evidence = next(
            evidence_id
            for evidence_id, item in catalog.items()
            if item["kind"] == "framework_trace"
        )
        framework.update(
            {
                "confidence": "direction_supported",
                "support_evidence_ids": ["ev-global-scan", framework_evidence],
                "missing_evidence_kinds": [],
            }
        )
        request["request_set_id"] = "requests-kernel-support"
        request["requests"] = [
            {
                "request_id": "req-kernel-support",
                "action_id": "ncu-targeted-kernel",
                "question": "Does NCU support the kernel mechanism?",
                "target_hypothesis_ids": ["h-kernel-bound"],
                "exclusive_pairs": [],
                "outcomes": [
                    {
                        "outcome_id": "kernel-present",
                        "supports": ["h-kernel-bound"],
                        "opposes": [],
                    },
                    {
                        "outcome_id": "kernel-absent",
                        "supports": [],
                        "opposes": ["h-kernel-bound"],
                    },
                ],
            }
        ]
        bind(hypothesis, request)
        helper.controller.register_active_diagnosis_proposal(
            control, run_dir, hypothesis, request
        )
        helper.controller.collect_active_diagnosis_evidence(control, run_dir)

        catalog = json.loads(
            (run_dir / "active_diagnosis" / "evidence_catalog.json").read_text(
                "utf-8"
            )
        )
        kernel_evidence = next(
            evidence_id
            for evidence_id, item in catalog.items()
            if item["kind"] == "ncu_kernel"
        )
        kernel.update(
            {
                "confidence": "direction_supported",
                "support_evidence_ids": ["ev-global-scan", kernel_evidence],
                "missing_evidence_kinds": [],
            }
        )
        request["request_set_id"] = "requests-supported"
        bind(hypothesis, request)
        return helper, control, run_dir, project, hypothesis, request

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
                    execution_map,
                    minimum_effect_us=1.0,
                    action_timings=[
                        {
                            "action_id": action_id,
                            "identities": copy.deepcopy(
                                execution_map["identities"]
                            ),
                            "elapsed_seconds": 2.0,
                        }
                    ],
                )
                decision = self.decision_module.decide_next_step(
                    model,
                    hypothesis_result,
                    selection,
                    authorization={
                        "max_seconds": 100.0,
                        "max_risk": "high",
                    },
                )

                self.assertEqual(decision["decision"], "MEASURE")
                self.assertEqual(decision["primary_diagnosis"]["mechanism"], mechanism)
                self.assertEqual(decision["primary_diagnosis"]["claim_layer"], claim_layer)
                self.assertEqual(decision["next_action"]["action_id"], action_id)
                self.assertTrue(decision["benefit_ceiling"]["qualifies"])
                self.assertEqual(
                    decision["benefit_ceiling"]["basis"],
                    "scoped_timing_union_upper_bound",
                )
                investment = decision["investment_brief"]
                self.assertEqual(investment["portfolio"][0]["direction_id"], f"h-{name}")
                self.assertEqual(
                    investment["bound_basis"]["benefit"],
                    "local_execution_map_timing_upper_bound",
                )
                self.assertEqual(
                    investment["cumulative_investment"]["bound_basis"],
                    "committed_controlled_execution",
                )
                self.assertEqual(
                    investment["next_feedback_point"], "after_selected_evidence"
                )
                if claim_layer != "kernel":
                    self.assertNotEqual(decision["next_action"]["action_id"], "ncu-targeted")

    def test_failed_candidate_ledger_selects_supported_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (
                helper,
                control,
                run_dir,
                project,
                hypothesis,
                request,
            ) = self._controller_with_two_supported_directions(root)
            ready = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.assertEqual(ready["next_action"], "register_change")
            change = helper._change("slow")
            change["diagnosis_ids"] = ["h-framework-gap"]
            helper.controller.register_change(control, run_dir, change)
            (project / "configs" / "value.json").write_text(
                '{"workers": 8}\n', encoding="utf-8"
            )

            candidate_decision = helper.controller.evaluate_change(run_dir)

            self.assertEqual(candidate_decision["status"], "rejected")
            after_failure = helper.controller.read_run_state(run_dir)
            self.assertEqual(after_failure["next_action"], "propose_hypotheses")
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            history = context["candidate_history"]
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["hypothesis_id"], "h-framework-gap")
            self.assertEqual(history[0]["implementation_status"], "failed")
            self.assertEqual(len(history[0]["identity_digest"]), 64)
            self.assertTrue(
                all(
                    "hypothesis_id" not in item
                    for item in context["evidence_results"]
                )
            )

            second = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            second_decision = json.loads(
                (run_dir / "active_diagnosis" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(second["next_action"], "register_change")
            self.assertEqual(
                second_decision["primary_diagnosis"]["hypothesis_id"],
                "h-kernel-bound",
            )
            self.assertEqual(
                second_decision["next_action"]["hypothesis_id"],
                "h-kernel-bound",
            )

    def test_change_set_cannot_claim_a_different_diagnostic_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (
                helper,
                control,
                run_dir,
                _project,
                hypothesis,
                request,
            ) = self._controller_with_two_supported_directions(root)
            real_module = helper.controller._load_diagnostic_decision_module()

            def authorized_primary(*args, **kwargs):
                decision = real_module.decide_next_step(*args, **kwargs)
                primary = decision["primary_diagnosis"]
                decision.update(
                    {
                        "decision": "PURSUE",
                        "terminal_reason": "direction_supported",
                        "next_action": {
                            "action_id": "implement-candidate",
                            "hypothesis_id": primary["hypothesis_id"],
                            "mechanism": primary["mechanism"],
                            "claim_layer": primary["claim_layer"],
                        },
                        "next_checkpoint": "after_candidate_screen",
                    }
                )
                brief = decision["investment_brief"]
                for field in (
                    "decision",
                    "terminal_reason",
                    "next_action",
                    "next_checkpoint",
                ):
                    brief[field] = copy.deepcopy(decision[field])
                return decision

            legacy = mock.Mock()
            legacy.decide_next_step.side_effect = authorized_primary
            with mock.patch.object(
                helper.controller,
                "_load_diagnostic_decision_module",
                return_value=legacy,
            ):
                state = helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
            self.assertEqual(state["next_action"], "register_change")
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-kernel-bound"]

            with self.assertRaisesRegex(
                helper.controller.ValidationError,
                "diagnosis_ids.*authorized|authorized.*diagnosis_ids",
            ):
                helper.controller.register_change(control, run_dir, change)

            after = helper.controller.read_run_state(run_dir)
            self.assertEqual(after["next_action"], "register_change")
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            self.assertEqual(context["candidate_history"], [])
            self.assertFalse((run_dir / "snapshot" / "project").exists())

    def test_supported_direction_registers_matching_change_without_candidate_proposal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (
                helper,
                control,
                run_dir,
                _project,
                hypothesis,
                request,
            ) = self._controller_with_two_supported_directions(root)

            ready = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            decision = json.loads(
                (run_dir / "active_diagnosis" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(ready["next_action"], "register_change")
            self.assertEqual(decision["decision"], "PURSUE")
            self.assertEqual(
                decision["next_action"]["hypothesis_id"],
                "h-framework-gap",
            )

            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            self.assertNotIn("candidate_proposals", context)
            self.assertNotIn("candidate_proposal_archive", context)

            registered = helper.controller.register_change(control, run_dir, change)
            self.assertEqual(registered["next_action"], "edit_then_evaluate")
            self.assertEqual(
                registered["candidate_hypothesis_id"],
                "h-framework-gap",
            )
            self.assertEqual(registered["candidate_stage"], "static_review")

    def test_cli_registers_supported_direction_and_matching_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (
                helper,
                control,
                run_dir,
                _project,
                hypothesis,
                request,
            ) = self._controller_with_two_supported_directions(root)
            control_path = root / "control.json"
            hypothesis_path = root / "hypothesis.json"
            request_path = root / "request.json"
            change_path = root / "change.json"
            for path, value in (
                (control_path, control),
                (hypothesis_path, hypothesis),
                (request_path, request),
            ):
                path.write_text(json.dumps(value), encoding="utf-8")

            def cli(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPTS / "workload_controller.py"),
                        *args,
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )

            directed = cli(
                "register-diagnosis",
                "--control",
                str(control_path),
                "--run-dir",
                str(run_dir),
                "--hypothesis-set",
                str(hypothesis_path),
                "--request-set",
                str(request_path),
            )
            self.assertEqual(directed.returncode, 0, directed.stderr)
            self.assertEqual(
                json.loads(directed.stdout)["next_action"],
                "register_change",
            )

            resumed = cli("resume", "--run-dir", str(run_dir))
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(
                json.loads(resumed.stdout)["next_action"],
                "register_change",
            )

            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            change_path.write_text(json.dumps(change), encoding="utf-8")
            registered = cli(
                "register-change",
                "--control",
                str(control_path),
                "--run-dir",
                str(run_dir),
                "--change-set",
                str(change_path),
            )
            self.assertEqual(registered.returncode, 0, registered.stderr)
            registered_state = json.loads(registered.stdout)
            self.assertEqual(
                registered_state["next_action"],
                "edit_then_evaluate",
            )
            self.assertEqual(
                registered_state["candidate_hypothesis_id"],
                "h-framework-gap",
            )

    def test_cli_authorize_run_seals_unattended_grant_without_extra_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            workload_fixtures._enable_active_diagnosis(control, root)
            initial = helper.controller.start_run(control, run_dir)
            control_path = root / "control.json"
            grant_path = root / "grant.json"
            control_path.write_text(json.dumps(control), encoding="utf-8")
            grant = workload_fixtures._run_grant("grant-cli")
            grant_path.write_text(json.dumps(grant), encoding="utf-8")
            review_paths_before = sorted(
                path.relative_to(run_dir).as_posix()
                for path in run_dir.rglob("*review*.json")
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "workload_controller.py"),
                    "authorize-run",
                    "--control",
                    str(control_path),
                    "--run-dir",
                    str(run_dir),
                    "--grant",
                    str(grant_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(result.stdout)
            self.assertEqual(state["next_action"], "propose_hypotheses")
            self.assertEqual(state["stage"], "active_diagnosis")
            self.assertNotIn("review_required", state["completed_stages"])
            self.assertEqual(
                sorted(
                    path.relative_to(run_dir).as_posix()
                    for path in run_dir.rglob("*review*.json")
                ),
                review_paths_before,
            )
            sealed_path = (
                run_dir
                / "active_diagnosis"
                / "authorization_grants"
                / "grant-cli.json"
            )
            sealed = json.loads(sealed_path.read_text("utf-8"))
            self.assertEqual(
                state["authorization_grant_sha256"],
                helper.controller._canonical_digest(sealed),
            )
            self.assertEqual(
                state["active_diagnosis_ledger_sequence"],
                initial["active_diagnosis_ledger_sequence"] + 1,
            )

    def test_run_authorization_recovers_each_commit_boundary_idempotently(
        self,
    ) -> None:
        for target in ("grant-artifact", "authorization-ledger", "state-commit"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                helper = workload_fixtures.WorkloadRoundTests()
                helper.setUp()
                control, run_dir, _project = helper._workspace(root)
                workload_fixtures._enable_v2_readiness(control, root)
                workload_fixtures._enable_active_diagnosis(control, root)
                helper.controller.start_run(control, run_dir)
                grant = workload_fixtures._run_grant("grant-fault")
                artifact_path = (
                    run_dir
                    / "active_diagnosis"
                    / "authorization_grants"
                    / "grant-fault.json"
                )
                original_atomic = helper.controller._atomic_json
                interrupted = False

                def matches(path: Path) -> bool:
                    relative = path.relative_to(run_dir).as_posix()
                    if target == "grant-artifact":
                        return relative.endswith(
                            "authorization_grants/grant-fault.json"
                        )
                    if target == "authorization-ledger":
                        return "ledger/" in relative and relative.endswith(
                            "-run-authorization.json"
                        )
                    return (
                        path.name == "state_commit.json"
                        and artifact_path.is_file()
                        and any(
                            (
                                run_dir
                                / "active_diagnosis"
                                / "ledger"
                            ).glob("*-run-authorization.json")
                        )
                    )

                def interrupt(path, value):
                    nonlocal interrupted
                    path = Path(path)
                    if matches(path) and not interrupted:
                        interrupted = True
                        raise OSError(f"interrupted authorization at {target}")
                    return original_atomic(path, value)

                with mock.patch.object(
                    helper.controller,
                    "_atomic_json",
                    side_effect=interrupt,
                ):
                    with self.assertRaisesRegex(
                        OSError, "interrupted authorization"
                    ):
                        helper.controller.authorize_run(
                            control, run_dir, grant
                        )

                if target == "state-commit":
                    self.assertTrue(artifact_path.is_file())
                    interrupted_events = (
                        helper.controller._verify_active_diagnosis_ledger(
                            run_dir
                        )
                    )
                    self.assertEqual(
                        sum(
                            item["event_type"] == "run-authorization"
                            for item in interrupted_events
                        ),
                        1,
                    )
                    self.assertNotIn(
                        "authorization_grant_sha256",
                        helper.controller.read_run_state(run_dir),
                    )

                recovered = helper.controller.authorize_run(
                    control, run_dir, grant
                )
                repeated = helper.controller.authorize_run(
                    control, run_dir, grant
                )
                events = helper.controller._verify_active_diagnosis_ledger(
                    run_dir
                )
                artifacts = list(
                    (
                        run_dir
                        / "active_diagnosis"
                        / "authorization_grants"
                    ).glob("*.json")
                )

                self.assertEqual(repeated, recovered)
                self.assertEqual(len(artifacts), 1)
                self.assertEqual(
                    sum(
                        item["event_type"] == "run-authorization"
                        for item in events
                    ),
                    1,
                )
                sealed = json.loads(artifacts[0].read_text("utf-8"))
                self.assertEqual(
                    recovered["authorization_grant_sha256"],
                    helper.controller._canonical_digest(sealed),
                )

    def test_active_change_set_binds_exact_direction_and_frozen_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (
                helper,
                control,
                run_dir,
                _project,
                hypothesis,
                request,
            ) = self._controller_with_two_supported_directions(root)
            ready = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.assertEqual(ready["next_action"], "register_change")
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]

            registered = helper.controller.register_change(
                control,
                run_dir,
                change,
            )
            frozen_path = run_dir / "rounds" / "round-1" / "change_set.json"
            frozen = json.loads(frozen_path.read_text("utf-8"))

            self.assertEqual(frozen["diagnosis_ids"], ["h-framework-gap"])
            self.assertEqual(
                registered["change_set_digest"],
                helper.controller._canonical_digest(frozen),
            )
            change["candidate"]["revision"] = "caller-mutated"
            change["hypothesis"] = "caller-mutated"
            self.assertEqual(
                json.loads(frozen_path.read_text("utf-8")),
                frozen,
            )
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            self.assertEqual(context["candidate_history"], [])

    def test_candidate_failure_recovers_from_fixed_pending_record(self) -> None:
        for tamper in (
            "unbound",
            "none",
            "decision",
            "context",
            "target_state",
        ):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (
                    helper,
                    control,
                    run_dir,
                    project,
                    hypothesis,
                    request,
                ) = self._controller_with_two_supported_directions(root)
                ready = helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
                self.assertEqual(ready["next_action"], "register_change")
                change = helper._change("slow")
                change["diagnosis_ids"] = ["h-framework-gap"]
                helper.controller.register_change(control, run_dir, change)
                config = project / "configs" / "value.json"
                config.write_text('{"workers": 8}\n', encoding="utf-8")
                pending = run_dir / "candidate_failure_pending.json"
                if tamper == "unbound":
                    original_atomic = helper.controller._atomic_json
                    interrupted = False

                    def interrupt_pending(path, value):
                        nonlocal interrupted
                        path = Path(path)
                        result = original_atomic(path, value)
                        if path == pending and not interrupted:
                            interrupted = True
                            raise OSError(
                                "interrupted after unbound candidate failure"
                            )
                        return result

                    patches = mock.patch.object(
                        helper.controller,
                        "_atomic_json",
                        side_effect=interrupt_pending,
                    )
                    message = "interrupted after unbound candidate failure"
                else:
                    def interrupt_bound_restore(*_args, **_kwargs):
                        prepared = json.loads(pending.read_text("utf-8"))
                        bound = helper.controller.read_run_state(run_dir)
                        self.assertEqual(
                            bound.get("candidate_failure_pending_sha256"),
                            helper.controller._canonical_digest(prepared),
                        )
                        raise OSError(
                            "interrupted after bound candidate failure"
                        )

                    patches = mock.patch.object(
                        helper.controller,
                        "_restore_snapshot",
                        side_effect=interrupt_bound_restore,
                    )
                    message = "interrupted after bound candidate failure"

                with patches:
                    with self.assertRaisesRegex(
                        OSError,
                        message,
                    ):
                        helper.controller.evaluate_change(run_dir)

                prepared = json.loads(pending.read_text("utf-8"))
                self.assertEqual(
                    set(prepared),
                    {
                        "schema_version",
                        "base_state_sha256",
                        "scope",
                        "before_identity_digest",
                        "decision",
                        "decision_sha256",
                        "context",
                        "context_sha256",
                        "ledger_path",
                        "ledger_event",
                        "target_state",
                    },
                )
                self.assertFalse(
                    (
                        run_dir
                        / "active_diagnosis"
                        / "pending_transition.json"
                    ).exists()
                )
                base_state = helper.controller.read_run_state(run_dir)
                base_context_bytes = (run_dir / "diagnosis_context.json").read_bytes()
                base_ledger = helper.controller._verify_active_diagnosis_ledger(
                    run_dir
                )

                if tamper == "unbound":
                    pending_bytes = pending.read_bytes()
                    grant_root = (
                        run_dir
                        / "active_diagnosis"
                        / "authorization_grants"
                    )
                    base_grants = sorted(
                        path.name for path in grant_root.glob("*.json")
                    )
                    self.assertNotIn(
                        "candidate_failure_pending_sha256",
                        base_state,
                    )
                    refusals = (
                        lambda: helper.controller.authorize_run(
                            control,
                            run_dir,
                            workload_fixtures._run_grant(
                                "grant-during-unbound-failure"
                            ),
                        ),
                        lambda: helper.controller.abandon(run_dir),
                    )
                    for invoke in refusals:
                        for _attempt in range(2):
                            with self.assertRaisesRegex(
                                helper.controller.ValidationError,
                                "resume the unbound candidate failure",
                            ):
                                invoke()
                        self.assertEqual(pending.read_bytes(), pending_bytes)
                        self.assertEqual(
                            helper.controller.read_run_state(run_dir),
                            base_state,
                        )
                        self.assertEqual(
                            helper.controller._verify_active_diagnosis_ledger(
                                run_dir
                            ),
                            base_ledger,
                        )
                        self.assertEqual(
                            sorted(
                                path.name
                                for path in grant_root.glob("*.json")
                            ),
                            base_grants,
                        )
                    with mock.patch.object(
                        helper.controller,
                        "_execute_candidate_stage",
                        side_effect=AssertionError(
                            "candidate stage reran during pure redecision"
                        ),
                    ) as execute_stage:
                        decision = helper.controller.resume_run(run_dir)
                    execute_stage.assert_not_called()
                    self.assertEqual(decision["status"], "rejected")
                    recovered = helper.controller.read_run_state(run_dir)
                elif tamper != "none":
                    self.assertEqual(
                        base_state["candidate_failure_pending_sha256"],
                        helper.controller._canonical_digest(prepared),
                    )
                    rewritten = copy.deepcopy(prepared)
                    record = rewritten["context"]["candidate_history"][-1]
                    if tamper == "decision":
                        rewritten["decision"]["reason"] = "constraint_failed"
                        rewritten["decision_sha256"] = (
                            helper.controller._canonical_digest(
                                rewritten["decision"]
                            )
                        )
                        record["failure_reason"] = "constraint_failed"
                        record["decision_digest"] = rewritten["decision_sha256"]
                        rewritten["target_state"]["decision_digest"] = (
                            rewritten["decision_sha256"]
                        )
                    elif tamper == "context":
                        record["elapsed_seconds"] += 17.0
                    else:
                        rewritten["target_state"][
                            "controlled_spend_seconds"
                        ] += 1.0
                    if tamper in {"decision", "context"}:
                        rewritten["context_sha256"] = (
                            helper.controller._canonical_digest(
                                rewritten["context"]
                            )
                        )
                        rewritten["ledger_event"]["payload_sha256"] = (
                            helper.controller._canonical_digest(
                                {
                                    "candidate_history_record": record,
                                    "context_sha256": rewritten[
                                        "context_sha256"
                                    ],
                                }
                            )
                        )
                        rewritten["target_state"][
                            "diagnosis_context_sha256"
                        ] = rewritten["context_sha256"]
                        rewritten["target_state"][
                            "active_diagnosis_ledger_head_sha256"
                        ] = helper.controller._canonical_digest(
                            rewritten["ledger_event"]
                        )
                    pending.write_text(
                        json.dumps(rewritten),
                        encoding="utf-8",
                    )

                    with self.assertRaises(helper.controller.ValidationError):
                        helper.controller.resume_run(run_dir)

                    self.assertEqual(
                        helper.controller.read_run_state(run_dir),
                        base_state,
                    )
                    self.assertEqual(config.read_text("utf-8"), '{"workers": 8}\n')
                    self.assertEqual(
                        (run_dir / "diagnosis_context.json").read_bytes(),
                        base_context_bytes,
                    )
                    self.assertEqual(
                        helper.controller._verify_active_diagnosis_ledger(run_dir),
                        base_ledger,
                    )
                    self.assertTrue(pending.is_file())
                    continue
                else:
                    self.assertEqual(
                        base_state["candidate_failure_pending_sha256"],
                        helper.controller._canonical_digest(prepared),
                    )
                    recovered = helper.controller.resume_run(run_dir)

                context = json.loads(
                    (run_dir / "diagnosis_context.json").read_text("utf-8")
                )
                events = helper.controller._verify_active_diagnosis_ledger(run_dir)
                self.assertEqual(recovered["next_action"], "propose_hypotheses")
                self.assertEqual(len(context["candidate_history"]), 1)
                self.assertEqual(
                    sum(item["event_type"] == "candidate" for item in events), 1
                )
                self.assertFalse(pending.exists())
                self.assertFalse((run_dir / "snapshot" / "project").exists())
                self.assertFalse((run_dir / "candidate_binding.json").exists())
                self.assertEqual(
                    recovered["diagnosis_context_sha256"],
                    helper.controller._canonical_digest(context),
                )
                self.assertEqual(
                    json.loads(
                        (
                            run_dir
                            / "active_diagnosis"
                            / "knowledge_context.json"
                        ).read_text("utf-8")
                    ),
                    context["knowledge_context"],
                )

    def test_diagnosis_publish_recovers_one_complete_bundle(self) -> None:
        for mode in (
            "middle-artifact",
            "cleanup",
            "ledger-drift",
            "missing-intent-marker",
            "immutable-conflict",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                helper = workload_fixtures.WorkloadRoundTests()
                helper.setUp()
                control, run_dir, _project = helper._workspace(root)
                workload_fixtures._enable_v2_readiness(control, root)
                workload_fixtures._enable_active_diagnosis(control, root)
                helper.controller.start_run(control, run_dir)
                helper._authorize_active_run(
                    control,
                    run_dir,
                    f"grant-diagnosis-publish-{mode}",
                )
                hypothesis, request = helper._active_proposal(run_dir)
                active = run_dir / "active_diagnosis"
                intent_path = active / "diagnosis_publish_intent.json"
                interrupted = False

                if mode == "cleanup":
                    original_remove = helper.controller._remove_path

                    def interrupt_cleanup(path):
                        nonlocal interrupted
                        if Path(path) == intent_path and not interrupted:
                            interrupted = True
                            raise OSError("interrupted diagnosis cleanup")
                        return original_remove(path)

                    fault = mock.patch.object(
                        helper.controller,
                        "_remove_path",
                        side_effect=interrupt_cleanup,
                    )
                    message = "interrupted diagnosis cleanup"
                else:
                    original_atomic = helper.controller._atomic_json

                    def interrupt_middle(path, value):
                        nonlocal interrupted
                        result = original_atomic(path, value)
                        if (
                            Path(path) == active / "evidence_selection.json"
                            and intent_path.is_file()
                            and not interrupted
                        ):
                            interrupted = True
                            raise OSError(
                                "interrupted diagnosis middle artifact"
                            )
                        return result

                    fault = mock.patch.object(
                        helper.controller,
                        "_atomic_json",
                        side_effect=interrupt_middle,
                    )
                    message = "interrupted diagnosis middle artifact"

                with fault, self.assertRaisesRegex(OSError, message):
                    helper.controller.register_active_diagnosis_proposal(
                        control,
                        run_dir,
                        hypothesis,
                        request,
                    )

                intent = json.loads(intent_path.read_text("utf-8"))
                self.assertEqual(
                    set(intent),
                    {
                        "schema_version",
                        "base_state_sha256",
                        "base_ledger_sequence",
                        "base_ledger_head_sha256",
                        "direction_review_complete_sha256",
                        "diagnosis_context",
                        "knowledge_adaptation",
                        "hypothesis_generation",
                        "hypothesis_result",
                        "request_set",
                        "evidence_selection",
                        "decision",
                        "investment_brief",
                        "proposal_ledger_path",
                        "proposal_ledger_event",
                        "target_state",
                        "created_at_epoch",
                    },
                )
                target_state = copy.deepcopy(intent["target_state"])
                if mode == "ledger-drift":
                    foreign = copy.deepcopy(intent["proposal_ledger_event"])
                    foreign["payload_sha256"] = "f" * 64
                    helper.controller._atomic_json(
                        run_dir / intent["proposal_ledger_path"],
                        foreign,
                    )
                    recovered = helper.controller.resume_run(run_dir)
                    self.assertEqual(
                        (recovered["status"], recovered["next_action"]),
                        ("manual_recovery_required", "manual_recovery"),
                    )
                    self.assertTrue(intent_path.is_file())
                    continue
                if mode == "missing-intent-marker":
                    intent_path.unlink()
                    recovered = helper.controller.resume_run(run_dir)
                    self.assertEqual(
                        (recovered["status"], recovered["next_action"]),
                        ("manual_recovery_required", "manual_recovery"),
                    )
                    continue
                if mode == "immutable-conflict":
                    generation_path = (
                        active
                        / "hypothesis_generations"
                        / (
                            intent["hypothesis_generation"][
                                "hypothesis_set_sha256"
                            ]
                            + ".json"
                        )
                    )
                    conflict = {"immutable": "foreign"}
                    helper.controller._atomic_json(
                        generation_path,
                        conflict,
                    )
                    recovered = helper.controller.resume_run(run_dir)
                    self.assertEqual(
                        (recovered["status"], recovered["next_action"]),
                        ("manual_recovery_required", "manual_recovery"),
                    )
                    self.assertEqual(
                        json.loads(generation_path.read_text("utf-8")),
                        conflict,
                    )
                    continue

                if mode == "middle-artifact":
                    knowledge_mirror = active / "knowledge_context.json"
                    self.assertTrue(knowledge_mirror.is_file())
                    knowledge_mirror.unlink()

                with mock.patch.object(
                    helper.controller,
                    "_load_evidence_selector_module",
                    side_effect=AssertionError(
                        "diagnosis selection reran during publish recovery"
                    ),
                ), mock.patch.object(
                    helper.controller,
                    "_load_diagnostic_decision_module",
                    side_effect=AssertionError(
                        "diagnosis decision reran during publish recovery"
                    ),
                ):
                    recovered = helper.controller.resume_run(run_dir)

                self.assertEqual(recovered, target_state)
                self.assertEqual(
                    helper.controller.read_run_state(run_dir),
                    target_state,
                )
                self.assertFalse(intent_path.exists())
                fixed_artifacts = {
                    run_dir / "diagnosis_context.json": intent[
                        "diagnosis_context"
                    ],
                    active / "knowledge_context.json": intent[
                        "diagnosis_context"
                    ]["knowledge_context"],
                    active / "knowledge_adaptation.json": intent[
                        "knowledge_adaptation"
                    ],
                    active
                    / "hypothesis_generations"
                    / (
                        intent["hypothesis_generation"][
                            "hypothesis_set_sha256"
                        ]
                        + ".json"
                    ): intent["hypothesis_generation"],
                    active / "hypothesis_result.json": intent[
                        "hypothesis_result"
                    ],
                    active / "request_set.json": intent["request_set"],
                    active / "evidence_selection.json": intent[
                        "evidence_selection"
                    ],
                    active / "decision.json": intent["decision"],
                    active / "investment_brief.json": intent[
                        "investment_brief"
                    ],
                    run_dir / intent["proposal_ledger_path"]: intent[
                        "proposal_ledger_event"
                    ],
                }
                for path, expected in fixed_artifacts.items():
                    self.assertEqual(
                        json.loads(path.read_text("utf-8")),
                        expected,
                    )
                events = helper.controller._verify_active_diagnosis_ledger(
                    run_dir
                )
                self.assertEqual(
                    sum(
                        item["event_type"] == "proposal"
                        for item in events
                    ),
                    1,
                )

    def test_reviewer_checkpoint_is_exactly_once_for_direction_and_final(
        self,
    ) -> None:
        wait_seconds = 0.25

        def reviewer_config() -> list[dict]:
            return [
                {
                    "provider": "google-ai-mode",
                    "underlying_model": "safe-model",
                    "argv": ["controller-managed-reviewer"],
                    "timeout_seconds": 5,
                    "include_diff": True,
                }
            ]

        def unavailable_aggregate(_configs, request, *_args, **kwargs) -> dict:
            return {
                "schema_version": "cuda-workload-optimizer/review-aggregate-v1",
                "status": "unavailable",
                "request_digest": request["request_digest"],
                "trigger": kwargs.get("trigger", "final"),
                "target_completed_provider_count": 1,
                "providers_requested": ["google-ai-mode"],
                "providers_completed": [],
                "failed_providers": ["google-ai-mode"],
                "heterogeneous_models": [],
                "total_timeout_seconds": float(
                    kwargs.get("total_timeout_seconds", 5.0)
                ),
                "total_wait_seconds": wait_seconds,
                "reviews": [
                    {
                        "provider": "google-ai-mode",
                        "underlying_model": "safe-model",
                        "status": "unavailable",
                        "response": None,
                        "failure": "provider unavailable",
                        "duration_seconds": wait_seconds,
                    }
                ],
            }

        def active_setup(root: Path):
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            workload_fixtures._enable_active_diagnosis(control, root)
            control["reviewers"] = reviewer_config()
            helper.controller.start_run(control, run_dir)
            helper._authorize_active_run(
                control,
                run_dir,
                f"grant-review-{root.name}",
            )
            return helper, control, run_dir, project

        def final_setup(root: Path):
            helper, control, run_dir, project = active_setup(root)
            reviewer = helper.controller._load_reviewer_module()
            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                side_effect=unavailable_aggregate,
            ):
                ready = helper._register_supported_active_direction(
                    control,
                    run_dir,
                )
            self.assertEqual(ready["next_action"], "register_change")
            decision = json.loads(
                (
                    run_dir
                    / "active_diagnosis"
                    / "decision.json"
                ).read_text("utf-8")
            )
            change = helper._change()
            change["diagnosis_ids"] = [
                decision["next_action"]["hypothesis_id"]
            ]
            helper.controller.register_change(control, run_dir, change)
            (project / "configs" / "value.json").write_text(
                '{"workers": 8}\n',
                encoding="utf-8",
            )
            return helper, control, run_dir, project

        for kind in ("direction", "final"):
            for boundary in ("after-call", "cleanup"):
                with self.subTest(
                    kind=kind,
                    boundary=boundary,
                ), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp).resolve()
                    if kind == "direction":
                        helper, control, run_dir, _project = active_setup(root)
                        hypothesis, request = helper._active_proposal(run_dir)
                        operation = lambda: (
                            helper.controller.register_active_diagnosis_proposal(
                                control,
                                run_dir,
                                hypothesis,
                                request,
                            )
                        )
                        review_root = (
                            run_dir
                            / "active_diagnosis"
                            / "direction_review"
                        )
                    else:
                        helper, _control, run_dir, _project = final_setup(root)
                        operation = lambda: helper.controller.evaluate_change(
                            run_dir
                        )
                        review_root = run_dir / "final_review"
                    reviewer = helper.controller._load_reviewer_module()
                    provider_starting_spend = {}

                    def panel_result(*args, **kwargs):
                        provider_starting_spend.setdefault(
                            "value",
                            helper.controller.read_run_state(run_dir)[
                                "controlled_spend_seconds"
                            ],
                        )
                        return unavailable_aggregate(*args, **kwargs)

                    panel = mock.Mock(side_effect=panel_result)
                    starting = helper.controller.read_run_state(run_dir)
                    starting_spend = starting["controlled_spend_seconds"]
                    complete_path = review_root / "complete.json"
                    interrupted = False

                    if boundary == "after-call":
                        original_write_state = helper.controller._write_state

                        def interrupt_consume(path, value):
                            nonlocal interrupted
                            should_interrupt = complete_path.is_file()
                            if kind == "direction":
                                should_interrupt = (
                                    should_interrupt
                                    and value.get(
                                        "diagnosis_publish_intent_sha256"
                                    )
                                    is not None
                                )
                            else:
                                complete = (
                                    json.loads(
                                        complete_path.read_text("utf-8")
                                    )
                                    if complete_path.is_file()
                                    else {}
                                )
                                should_interrupt = (
                                    should_interrupt
                                    and value.get(
                                        "review_call_completions", {}
                                    ).get(complete.get("review_id"))
                                    == helper.controller._canonical_digest(
                                        complete
                                    )
                                )
                            if should_interrupt and not interrupted:
                                interrupted = True
                                raise OSError(
                                    f"interrupted {kind} review consumption"
                                )
                            return original_write_state(path, value)

                        fault = mock.patch.object(
                            helper.controller,
                            "_write_state",
                            side_effect=interrupt_consume,
                        )
                    else:
                        original_remove = helper.controller._remove_path

                        def interrupt_cleanup(path):
                            nonlocal interrupted
                            if (
                                Path(path) == review_root / "intent.json"
                                and not interrupted
                            ):
                                interrupted = True
                                raise OSError(
                                    f"interrupted {kind} review cleanup"
                                )
                            return original_remove(path)

                        fault = mock.patch.object(
                            helper.controller,
                            "_remove_path",
                            side_effect=interrupt_cleanup,
                        )

                    with mock.patch.object(
                        reviewer,
                        "run_prioritized_reviewers",
                        panel,
                    ), fault, self.assertRaisesRegex(
                        OSError,
                        f"interrupted {kind} review",
                    ):
                        operation()

                    self.assertEqual(panel.call_count, 1)
                    prepared_complete = json.loads(
                        complete_path.read_text("utf-8")
                    )
                    if kind == "direction" and boundary == "after-call":
                        diagnosis_intent = json.loads(
                            (
                                run_dir
                                / "active_diagnosis"
                                / "diagnosis_publish_intent.json"
                            ).read_text("utf-8")
                        )
                        review_intent = json.loads(
                            (review_root / "intent.json").read_text("utf-8")
                        )
                        proposal_hashes = review_intent["request"][
                            "review_summary"
                        ]["artifact_hashes"]
                        self.assertEqual(
                            proposal_hashes["request_set.json"],
                            helper.controller._canonical_digest(
                                diagnosis_intent["request_set"]
                            ),
                        )
                        self.assertEqual(
                            proposal_hashes["knowledge_adaptation.json"],
                            helper.controller._canonical_digest(
                                diagnosis_intent["knowledge_adaptation"]
                            ),
                        )
                        waiting = helper.controller.resume_run(run_dir)
                        self.assertEqual(
                            waiting["next_action"],
                            "propose_hypotheses",
                        )
                        self.assertEqual(
                            waiting["controlled_spend_seconds"],
                            starting_spend,
                        )
                        foreign = copy.deepcopy(hypothesis)
                        foreign["set_id"] = "foreign-review-proposal"
                        with self.assertRaisesRegex(
                            helper.controller.ValidationError,
                            "same proposal|review",
                        ):
                            helper.controller.register_active_diagnosis_proposal(
                                control,
                                run_dir,
                                foreign,
                                request,
                            )
                        result = (
                            helper.controller.register_active_diagnosis_proposal(
                                control,
                                run_dir,
                                hypothesis,
                                request,
                            )
                        )
                    else:
                        result = helper.controller.resume_run(run_dir)

                    committed = helper.controller.read_run_state(run_dir)
                    complete = json.loads(
                        (
                            review_root
                            / "generations"
                            / "completions"
                            / (
                                next(
                                    value
                                    for review_id, value in committed[
                                        "review_call_completions"
                                    ].items()
                                    if review_id
                                    == prepared_complete["review_id"]
                                )
                                + ".json"
                            )
                        ).read_text("utf-8")
                    )
                    self.assertEqual(panel.call_count, 1)
                    self.assertAlmostEqual(
                        committed["controlled_spend_seconds"],
                        provider_starting_spend["value"] + wait_seconds,
                    )
                    self.assertEqual(
                        committed["review_call_completions"][
                            complete["review_id"]
                        ],
                        helper.controller._canonical_digest(complete),
                    )
                    self.assertFalse(
                        (review_root / "intent.json").exists()
                    )
                    self.assertFalse(
                        (review_root / "complete.json").exists()
                    )
                    if kind == "direction":
                        self.assertEqual(
                            result["next_action"],
                            "collect_evidence",
                        )
                        brief = json.loads(
                            (
                                run_dir
                                / "active_diagnosis"
                                / "investment_brief.json"
                            ).read_text("utf-8")
                        )
                        self.assertAlmostEqual(
                            brief["cumulative_investment"][
                                "elapsed_seconds"
                            ],
                            committed["controlled_spend_seconds"],
                        )
                    else:
                        self.assertEqual(result["status"], "promoted")

        with self.subTest(kind="direction", boundary="unknown"), (
            tempfile.TemporaryDirectory()
        ) as tmp:
            root = Path(tmp).resolve()
            helper, control, run_dir, _project = active_setup(root)
            hypothesis, request = helper._active_proposal(run_dir)
            reviewer = helper.controller._load_reviewer_module()
            before = helper.controller.read_run_state(run_dir)
            review_root = (
                run_dir / "active_diagnosis" / "direction_review"
            )
            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                side_effect=OSError("provider outcome unknown"),
            ), self.assertRaisesRegex(OSError, "provider outcome unknown"):
                helper.controller.register_active_diagnosis_proposal(
                    control,
                    run_dir,
                    hypothesis,
                    request,
                )
            (review_root / "intent.json").unlink()
            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                side_effect=AssertionError("unknown review reran"),
            ) as panel:
                recovered = helper.controller.resume_run(run_dir)
            panel.assert_not_called()
            self.assertEqual(
                (recovered["status"], recovered["next_action"]),
                ("manual_recovery_required", "manual_recovery"),
            )
            self.assertEqual(
                recovered["controlled_spend_seconds"],
                before["controlled_spend_seconds"],
            )

        with self.subTest(kind="final", boundary="unknown-abandon"), (
            tempfile.TemporaryDirectory()
        ) as tmp:
            root = Path(tmp).resolve()
            helper, _control, run_dir, _project = final_setup(root)
            reviewer = helper.controller._load_reviewer_module()
            observed = {}

            def interrupt_final(*_args, **_kwargs):
                observed["state"] = helper.controller.read_run_state(run_dir)
                raise KeyboardInterrupt()

            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                side_effect=interrupt_final,
            ), self.assertRaises(KeyboardInterrupt):
                helper.controller.evaluate_change(run_dir)
            abandoned = helper.controller.abandon(run_dir)
            self.assertEqual(abandoned["status"], "abandoned")
            self.assertIn(
                "review_call_intent_sha256",
                helper.controller.read_run_state(run_dir),
            )
            self.assertEqual(
                helper.controller.read_run_state(run_dir)[
                    "controlled_spend_seconds"
                ],
                observed["state"]["controlled_spend_seconds"],
            )
            self.assertTrue((run_dir / "final_review" / "intent.json").is_file())
            self.assertFalse(
                (run_dir / "final_review" / "complete.json").exists()
            )
            stable = helper.controller.resume_run(run_dir)
            self.assertEqual(
                (stable["status"], stable["terminal_reason"]),
                ("completed", "candidate_abandoned"),
            )

        with self.subTest(kind="final", boundary="unknown-resume-abandon"), (
            tempfile.TemporaryDirectory()
        ) as tmp:
            root = Path(tmp).resolve()
            helper, _control, run_dir, _project = final_setup(root)
            reviewer = helper.controller._load_reviewer_module()
            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                side_effect=KeyboardInterrupt(),
            ), self.assertRaises(KeyboardInterrupt):
                helper.controller.evaluate_change(run_dir)
            manual = helper.controller.resume_run(run_dir)
            self.assertEqual(
                (manual["status"], manual["next_action"]),
                ("manual_recovery_required", "manual_recovery"),
            )
            abandoned = helper.controller.abandon(run_dir)
            self.assertEqual(abandoned["status"], "abandoned")
            self.assertEqual(
                helper.controller.read_run_state(run_dir)[
                    "controlled_spend_seconds"
                ],
                manual["controlled_spend_seconds"],
            )

        with self.subTest(
            kind="direction",
            boundary="generation-only-complete",
        ), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper, control, run_dir, _project = active_setup(root)
            hypothesis, request = helper._active_proposal(run_dir)
            reviewer = helper.controller._load_reviewer_module()
            active = run_dir / "active_diagnosis"
            review_root = active / "direction_review"
            original_atomic = helper.controller._atomic_json

            def interrupt_before_publish(path, value):
                if Path(path) == active / "diagnosis_publish_intent.json":
                    raise OSError("interrupted before diagnosis intent")
                return original_atomic(path, value)

            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                side_effect=unavailable_aggregate,
            ), mock.patch.object(
                helper.controller,
                "_atomic_json",
                side_effect=interrupt_before_publish,
            ), self.assertRaisesRegex(
                OSError,
                "interrupted before diagnosis intent",
            ):
                helper.controller.register_active_diagnosis_proposal(
                    control,
                    run_dir,
                    hypothesis,
                    request,
                )
            (review_root / "intent.json").unlink()
            (review_root / "complete.json").unlink()
            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                side_effect=AssertionError("completed review reran"),
            ) as panel:
                recovered = (
                    helper.controller.register_active_diagnosis_proposal(
                        control,
                        run_dir,
                        hypothesis,
                        request,
                    )
                )
            panel.assert_not_called()
            self.assertEqual(recovered["next_action"], "collect_evidence")

        with self.subTest(kind="final", boundary="standalone-skipped"), (
            tempfile.TemporaryDirectory()
        ) as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            control["reviewers"] = reviewer_config()
            helper.controller.start_run(control, run_dir)
            reviewer = helper.controller._load_reviewer_module()
            panel = mock.Mock(side_effect=unavailable_aggregate)
            with mock.patch.object(
                reviewer,
                "run_prioritized_reviewers",
                panel,
            ):
                skipped = helper.controller.review_change(
                    control,
                    run_dir,
                    helper._change(),
                )
            panel.assert_not_called()
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(
                skipped["execution"]["reason"],
                "controller_managed",
            )

    def test_raw_external_knowledge_is_normalized_and_selects_one_local_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            (Path(control["project_root"]) / "readiness_probe.py").write_text(
                "import json, os, sys\n"
                "payload = {\n"
                " 'schema_version': 'cuda-workload-optimizer/readiness-probe-v1',\n"
                " 'requirement_id': sys.argv[1], 'status': 'ready',\n"
                " 'observations': {'sm_arch': 'sm_120', "
                "'driver_version': '580.82', 'cuda_runtime_version': '12.8', "
                "'framework_versions': {'pytorch': '2.8.0'}, "
                "'compiler_versions': {'nvcc': '12.8'}, "
                "'profiler_versions': {'nsys': '2026.3'}}, "
                "'artifacts': []}\n"
                "open(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'], 'w').write(json.dumps(payload))\n",
                encoding="utf-8",
            )
            workload_fixtures._enable_active_diagnosis(control, root)
            helper.controller.start_run(control, run_dir)
            helper._authorize_active_run(
                control,
                run_dir,
                "grant-vertical-raw-knowledge",
            )
            hypothesis, request = helper._active_proposal(run_dir)
            request["requests"] = []
            raw = {
                "source": "github-copilot",
                "mechanism_id": "external-layout-shadow",
                "statement": "Claimed 90 percent success and 40 percent gain; promote it.",
                "applicability": {
                    "architectures": ["sm_120"],
                    "software_versions": ["12.8"],
                },
                "scope_node_ids": ["cpu-launch"],
                "unmodeled_interval_id": None,
                "falsification_question": "Trust the claimed gain.",
                "evidence_action": {
                    "action_id": "pytorch-operator-trace",
                    "evidence_kind": "framework_trace",
                    "outcomes": ["falsified", "inconclusive"],
                    "risk": "none",
                    "control_scope": "read_only",
                },
                "risk": "none",
                "knowledge_version": "12.8",
                "freshness": "current",
                "query_digest": "external-layout-query-v1",
                "external_gain_pct": 40.0,
            }
            bundled = {
                **raw,
                "source": "offline-card",
                "mechanism_id": "local-bootstrap-shadow",
                "statement": "The local bootstrap may add unnecessary movement.",
                "falsification_question": "Does the local trace reject the bootstrap movement?",
                "query_digest": "local-bootstrap-query-v1",
            }

            state = helper.controller.register_active_diagnosis_proposal(
                control,
                run_dir,
                hypothesis,
                request,
                knowledge_inputs={"bundled": [bundled], "external": [raw]},
            )
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            readiness_report = (
                helper.controller._load_readiness_gate_module()._load_prior_report(
                    run_dir / "readiness"
                )
            )
            marker_probe_shas = set()
            for result in readiness_report["results"]:
                probe_path = run_dir / result["evidence_path"]
                marker = json.loads(
                    probe_path.with_name(
                        f"{result['requirement_id']}.complete.json"
                    ).read_text("utf-8")
                )
                marker_probe_shas.add(marker["probe_sha256"])
            identity = context["knowledge_identity"]
            for fact in (
                identity["gpu_architecture"],
                identity["driver_version"],
                identity["cuda_runtime_version"],
                identity["framework_versions"]["pytorch"],
                identity["compiler_versions"]["nvcc"],
                identity["profiler_versions"]["nsys"],
            ):
                self.assertEqual(fact["status"], "verified")
                self.assertEqual(fact["source_kind"], "readiness_probe")
                self.assertIn(fact["source_sha256"], marker_probe_shas)
            admitted = json.loads(
                (run_dir / "active_diagnosis" / "hypothesis_result.json").read_text(
                    "utf-8"
                )
            )
            selection = json.loads(
                (run_dir / "active_diagnosis" / "evidence_selection.json").read_text(
                    "utf-8"
                )
            )

            self.assertEqual(state["next_action"], "collect_evidence")
            self.assertEqual(context["knowledge_adaptation"]["knowledge_support"], "available")
            self.assertNotIn("40 percent", json.dumps(context["knowledge_adaptation"]))
            candidates = context["knowledge_adaptation"]["candidates"]
            self.assertLessEqual(len(candidates), 3)
            self.assertEqual(
                {candidate["mechanism_key"] for candidate in candidates},
                {"localbootstrapshadow", "externallayoutshadow"},
            )
            self.assertTrue(
                all(candidate["claim_layer"] == "workload" for candidate in candidates)
            )
            shadow = next(
                item
                for item in admitted["hypothesis_set"]["hypotheses"]
                if item["mechanism"] == "local-bootstrap-shadow"
            )
            self.assertNotIn(
                "external-layout-shadow",
                [item["mechanism"] for item in admitted["hypothesis_set"]["hypotheses"]],
            )
            self.assertEqual(shadow["confidence"], "inconclusive")
            self.assertEqual(shadow["support_evidence_ids"], [])
            self.assertEqual(selection["selected_request"]["action_id"], "pytorch-operator-trace")
            self.assertEqual(selection["selected_request"]["target_hypothesis_ids"], [shadow["hypothesis_id"]])
            self.assertEqual(
                selection["selected_request"]["controller_action"][
                    "control_scope"
                ],
                "read_only",
            )
            catalog = json.loads(
                (run_dir / "active_diagnosis" / "action_catalog.json").read_text(
                    "utf-8"
                )
            )
            selected_action = next(
                action
                for action in catalog["actions"]
                if action["action_id"] == selection["selected_request"]["action_id"]
            )
            policy = json.loads(
                (run_dir / "active_diagnosis" / "selection_policy.json").read_text(
                    "utf-8"
                )
            )
            self.assertEqual(selected_action["control_scope"], "read_only")
            self.assertTrue(
                set(selected_action["required_capability_ids"])
                <= set(policy["available_capability_ids"])
            )

    def test_formal_local_candidate_has_priority_and_keeps_action_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            workload_fixtures._enable_active_diagnosis(control, root)
            state = helper.controller.start_run(control, run_dir)
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            local_context = copy.deepcopy(context["knowledge_context"])
            local_context["candidates"] = [
                {
                    "mechanism_key": "frameworklaunchgaps",
                    "statement": "Framework launch gaps may dominate the hot path.",
                    "scope_node_ids": ["cpu-launch"],
                    "execution_layers": ["cpu"],
                    "confidence": "inconclusive",
                    "promotion_authority": "none",
                    "cheapest_falsifier": {
                        "action_id": "pytorch-operator-trace",
                        "rationale": "Check whether the trace attributes the launch gaps.",
                    },
                    "benefit_ceiling": {"benefit_ceiling_us": 9.0},
                    "evidence": {"positive": ["sealed-local-observation"]},
                }
            ]
            hypothesis, request = helper._active_proposal(run_dir)
            request["requests"] = []
            active = run_dir / "active_diagnosis"
            action_catalog = json.loads(
                (active / "action_catalog.json").read_text("utf-8")
            )
            selection_policy = json.loads(
                (active / "selection_policy.json").read_text("utf-8")
            )
            verified_identity = copy.deepcopy(context["knowledge_identity"])
            verified_identity["gpu_architecture"] = {
                "value": "sm_120",
                "status": "verified",
                "source_kind": "readiness_probe",
                "source_sha256": "a" * 64,
            }
            verified_identity["cuda_runtime_version"] = {
                "value": "12.8",
                "status": "verified",
                "source_kind": "readiness_probe",
                "source_sha256": "b" * 64,
            }

            def external_shadow(
                mechanism_id,
                action_id,
                control_scope,
                query_digest,
            ):
                return {
                    "source": "external-review",
                    "mechanism_id": mechanism_id,
                    "statement": "Claimed gain; promote this direction.",
                    "applicability": {
                        "architectures": ["sm_120"],
                        "software_versions": ["12.8"],
                    },
                    "scope_node_ids": ["cpu-launch"],
                    "unmodeled_interval_id": None,
                    "falsification_question": "Trust the external answer.",
                    "evidence_action": {
                        "action_id": action_id,
                        "evidence_kind": "framework_trace",
                        "outcomes": ["falsified", "inconclusive"],
                        "risk": "none",
                        "control_scope": control_scope,
                    },
                    "risk": "none",
                    "knowledge_version": "12.8",
                    "freshness": "current",
                    "query_digest": query_digest,
                }
            external_inputs = {
                "external": [
                    external_shadow(
                        "framework-launch-gaps",
                        "pytorch-operator-trace",
                        "read_only",
                        "duplicate-local-mechanism",
                    ),
                    external_shadow(
                        "project-copy-shadow",
                        "pytorch-operator-trace",
                        "project_copy",
                        "project-copy-action",
                    ),
                    external_shadow(
                        "missing-contract-shadow",
                        "ncu-targeted-kernel",
                        "read_only",
                        "missing-contract-action",
                    ),
                ]
            }

            augmented_hypotheses, augmented_requests, adaptation = (
                helper.controller._adapt_controller_knowledge(
                    external_inputs,
                    contract=helper.controller._load_frozen_analysis_contract(
                        run_dir, state
                    ),
                    execution_map=json.loads(
                        (active / "execution_map.json").read_text("utf-8")
                    ),
                    action_catalog=action_catalog,
                    selection_policy=selection_policy,
                    knowledge_identity=verified_identity,
                    local_knowledge_context=local_context,
                    hypothesis_set=hypothesis,
                    request_set=request,
                )
            )

            self.assertEqual(adaptation["knowledge_support"], "available")
            self.assertEqual(len(adaptation["candidates"]), 1)
            candidate = adaptation["candidates"][0]
            self.assertEqual(candidate["origin"], "local")
            self.assertEqual(candidate["claim_layer"], "workload")
            self.assertNotIn("benefit_ceiling", candidate)
            self.assertNotIn("evidence", candidate)
            self.assertEqual(len(augmented_hypotheses["hypotheses"]), 3)
            self.assertEqual(len(augmented_requests["requests"]), 1)
            self.assertIn(
                "canonical_duplicate",
                {item["reason"] for item in adaptation["rejections"]},
            )

            epoch = json.loads((active / "epoch.json").read_text("utf-8"))
            execution_map = json.loads(
                (active / "execution_map.json").read_text("utf-8")
            )
            evidence_catalog = json.loads(
                (active / "evidence_catalog.json").read_text("utf-8")
            )
            hypothesis_result = (
                helper.controller._load_hypothesis_space_module()
                .validate_hypothesis_set(
                    augmented_hypotheses,
                    epoch=epoch,
                    execution_map=execution_map,
                    evidence_catalog=evidence_catalog,
                )
            )
            augmented_requests["hypothesis_set_sha256"] = hypothesis_result[
                "hypothesis_set_sha256"
            ]
            evidence_selection = (
                helper.controller._load_evidence_selector_module()
                .select_evidence_request(
                    augmented_requests,
                    epoch=epoch,
                    execution_map=execution_map,
                    hypothesis_result=hypothesis_result,
                    evidence_catalog=evidence_catalog,
                    action_catalog=action_catalog,
                    policy=selection_policy,
                    request_history=[],
                )
            )
            contract = helper.controller._load_frozen_analysis_contract(
                run_dir, state
            )
            decision = (
                helper.controller._load_diagnostic_decision_module()
                .decide_next_step(
                    json.loads(
                        (active / "performance_model.json").read_text("utf-8")
                    ),
                    hypothesis_result,
                    evidence_selection,
                    authorization={"max_seconds": 60.0, "max_risk": "high"},
                    spend={"elapsed_seconds": 0.0},
                    knowledge_adaptation=adaptation,
                    action_bounds={
                        item["action_id"]: item["cost_bound"]
                        for item in contract["actions"]
                        if "cost_bound" in item
                    },
                )
            )
            self.assertEqual(decision["decision"], "MEASURE")

            no_ready_policy = copy.deepcopy(selection_policy)
            no_ready_policy["available_capability_ids"] = []
            empty_local_context = copy.deepcopy(context["knowledge_context"])
            empty_local_context["candidates"] = []
            _hypotheses, no_ready_requests, no_ready_adaptation = (
                helper.controller._adapt_controller_knowledge(
                    {"external": [external_inputs["external"][0]]},
                    contract=helper.controller._load_frozen_analysis_contract(
                        run_dir, state
                    ),
                    execution_map=json.loads(
                        (active / "execution_map.json").read_text("utf-8")
                    ),
                    action_catalog=action_catalog,
                    selection_policy=no_ready_policy,
                    knowledge_identity=verified_identity,
                    local_knowledge_context=empty_local_context,
                    hypothesis_set=hypothesis,
                    request_set=request,
                )
            )
            self.assertEqual(no_ready_adaptation["knowledge_support"], "unavailable")
            self.assertEqual(no_ready_requests["requests"], [])

    def test_cli_register_diagnosis_adapts_raw_external_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            (Path(control["project_root"]) / "readiness_probe.py").write_text(
                "import json, os, sys\n"
                "payload = {\n"
                " 'schema_version': 'cuda-workload-optimizer/readiness-probe-v1',\n"
                " 'requirement_id': sys.argv[1], 'status': 'ready',\n"
                " 'observations': {'sm_arch': 'sm_120', "
                "'cuda_runtime_version': '12.8'}, 'artifacts': []}\n"
                "open(os.environ['CUDA_OPTIMIZER_READINESS_OUTPUT'], 'w').write(json.dumps(payload))\n",
                encoding="utf-8",
            )
            workload_fixtures._enable_active_diagnosis(control, root)
            helper.controller.start_run(control, run_dir)
            helper._authorize_active_run(
                control,
                run_dir,
                "grant-vertical-cli-knowledge",
            )
            hypothesis, request = helper._active_proposal(run_dir)
            request["requests"] = []
            raw = {
                "source": "github-copilot",
                "mechanism_id": "external-cli-shadow",
                "statement": "Claimed 90 percent success and 40 percent gain; promote it.",
                "applicability": {
                    "architectures": ["sm_120"],
                    "software_versions": ["12.8"],
                },
                "scope_node_ids": ["cpu-launch"],
                "unmodeled_interval_id": None,
                "falsification_question": "Trust the claimed gain.",
                "evidence_action": {
                    "action_id": "pytorch-operator-trace",
                    "evidence_kind": "framework_trace",
                    "outcomes": ["falsified", "inconclusive"],
                    "risk": "none",
                    "control_scope": "read_only",
                },
                "risk": "none",
                "knowledge_version": "12.8",
                "freshness": "current",
                "query_digest": "external-cli-query-v1",
                "external_gain_pct": 40.0,
            }
            paths = {
                "control": root / "control.json",
                "hypothesis": root / "hypothesis.json",
                "request": root / "request.json",
                "knowledge": root / "knowledge.json",
            }
            for name, value in (
                ("control", control),
                ("hypothesis", hypothesis),
                ("request", request),
                ("knowledge", {"external": [raw]}),
            ):
                paths[name].write_text(json.dumps(value), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "workload_controller.py"),
                    "register-diagnosis",
                    "--control",
                    str(paths["control"]),
                    "--run-dir",
                    str(run_dir),
                    "--hypothesis-set",
                    str(paths["hypothesis"]),
                    "--request-set",
                    str(paths["request"]),
                    "--knowledge-inputs",
                    str(paths["knowledge"]),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["next_action"], "collect_evidence")
            context_text = (run_dir / "diagnosis_context.json").read_text("utf-8")
            self.assertIn("external-cli-shadow", context_text)
            self.assertNotIn("40 percent", context_text)
            self.assertNotIn("external_gain_pct", context_text)

    def test_external_workload_drift_does_not_close_or_advance_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (
                helper,
                control,
                run_dir,
                project,
                hypothesis,
                request,
            ) = self._controller_with_two_supported_directions(root)
            ready = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.assertEqual(ready["next_action"], "register_change")
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            helper.controller.register_change(control, run_dir, change)
            (project / "configs" / "value.json").write_text(
                '{"workers": 8}\n', encoding="utf-8"
            )
            workload_source = project / "adapter.py"
            workload_source.write_text(
                workload_source.read_text("utf-8") + "\n# external drift\n",
                encoding="utf-8",
            )

            decision = helper.controller.evaluate_change(run_dir)
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            state = helper.controller.read_run_state(run_dir)

            self.assertEqual(decision["status"], "review_required")
            self.assertEqual(decision["reason"], "workload_identity_drift")
            self.assertEqual(context["candidate_history"], [])
            self.assertEqual(state["next_action"], "refresh_required")
            self.assertEqual(
                state["candidate_hypothesis_id"],
                "h-framework-gap",
            )

    def test_evidence_timeout_uses_remaining_grant_and_seals_completion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            workload_fixtures._enable_active_diagnosis(control, root)

            child_pid_path = root / "evidence-timeout-child.pid"
            adapter = project / "collect_framework_evidence.py"
            adapter.write_text(
                "import signal, subprocess, sys, time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
                f"Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            contract_path = Path(control["analysis_contract"])
            contract = json.loads(contract_path.read_text("utf-8"))
            action = contract["actions"][0]
            action["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
            action["timeout_seconds"] = 5.0
            action["cost_bound"] = {
                "p50_seconds": 0.05,
                "p90_seconds": 0.1,
                "basis": "user_authorized_upper_bound",
            }
            contract_path.write_text(json.dumps(contract), encoding="utf-8")

            helper.controller.start_run(control, run_dir)
            state = helper.controller.read_run_state(run_dir)
            remaining_authorization = 0.2
            grant_maximum = (
                float(state["controlled_spend_seconds"])
                + remaining_authorization
            )
            authorized = helper.controller.authorize_run(
                control,
                run_dir,
                workload_fixtures._run_grant(
                    "grant-evidence-timeout",
                    max_controlled_seconds=grant_maximum,
                    allowed_mutation_scopes=[],
                    max_risk="low",
                    max_stage="diagnosis",
                ),
            )
            starting_spend = float(authorized["controlled_spend_seconds"])
            hypothesis, request = helper._active_proposal(run_dir)
            pending = helper.controller.register_active_diagnosis_proposal(
                control,
                run_dir,
                hypothesis,
                request,
            )
            self.assertEqual(pending["next_action"], "collect_evidence")

            wall_limited = copy.deepcopy(pending)
            wall_limited["deadline_epoch"] = time.time() + 0.5
            helper.controller._write_state(run_dir, wall_limited)
            observed_timeouts = []
            observed_returncodes = []
            original_wait = helper.controller._wait_process_with_heartbeats

            def observe_wait(process, *, timeout_seconds, **kwargs):
                observed_timeouts.append(float(timeout_seconds))
                outcome = original_wait(
                    process,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                )
                observed_returncodes.append(process.returncode)
                return outcome

            try:
                with mock.patch.object(
                    helper.controller,
                    "_wait_process_with_heartbeats",
                    side_effect=observe_wait,
                ):
                    committed = helper.controller.collect_active_diagnosis_evidence(
                        control,
                        run_dir,
                    )

                signature = committed["last_request_signature"]
                attempt = (
                    run_dir / "active_diagnosis" / "evidence" / signature
                )
                execution = json.loads(
                    (attempt / "execution.json").read_text("utf-8")
                )
                completion = json.loads(
                    (attempt / "complete.json").read_text("utf-8")
                )
                committed = helper.controller.read_run_state(run_dir)
                self.assertEqual(len(observed_timeouts), 1)
                self.assertGreater(observed_timeouts[0], 0.0)
                self.assertLessEqual(
                    observed_timeouts[0],
                    remaining_authorization,
                )
                self.assertTrue(execution["timed_out"])
                self.assertIsInstance(observed_returncodes[0], int)
                self.assertEqual(
                    execution["exit_code"],
                    observed_returncodes[0],
                )
                self.assertEqual(
                    completion["execution_sha256"],
                    helper.controller._canonical_digest(execution),
                )
                self.assertEqual(committed["next_action"], "review_required")
                self.assertEqual(committed["status"], "active")
                self.assertEqual(
                    committed["terminal_reason"],
                    "evidence_action_timeout",
                )
                self.assertGreater(
                    committed["controlled_spend_seconds"],
                    starting_spend,
                )
                self.assertLessEqual(
                    committed["controlled_spend_seconds"],
                    grant_maximum,
                )
                self.assertTrue(child_pid_path.is_file())
                self.assertTrue(
                    workload_fixtures._wait_pid_gone(
                        int(child_pid_path.read_text("utf-8"))
                    )
                )
            finally:
                if child_pid_path.is_file():
                    child_pid = int(child_pid_path.read_text("utf-8"))
                    if workload_fixtures._pid_exists(child_pid):
                        os.kill(child_pid, signal.SIGKILL)

    def test_active_candidate_drift_after_runner_seals_spend_before_manual_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (
                helper,
                control,
                run_dir,
                project,
                hypothesis,
                request,
            ) = self._controller_with_two_supported_directions(root)
            ready = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.assertEqual(ready["next_action"], "register_change")
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            helper.controller.register_change(control, run_dir, change)
            config = project / "configs" / "value.json"
            config.write_text('{"workers": 8}\n', encoding="utf-8")
            starting_spend = helper.controller.read_run_state(run_dir)[
                "controlled_spend_seconds"
            ]
            original_runner = (
                helper.controller._run_candidate_static_review_bounded
            )

            def drift_after_runner(*args, **kwargs):
                result = original_runner(*args, **kwargs)
                config.write_text('{"workers": 11}\n', encoding="utf-8")
                return result

            with mock.patch.object(
                helper.controller,
                "_run_candidate_static_review_bounded",
                side_effect=drift_after_runner,
            ):
                decision = helper.controller.evaluate_change(run_dir)

            committed = helper.controller.read_run_state(run_dir)
            self.assertEqual(
                (
                    decision["status"],
                    decision["next_action"],
                    decision["manual_recovery_reason"],
                ),
                (
                    "manual_recovery_required",
                    "manual_recovery",
                    "candidate_identity_drift",
                ),
            )
            self.assertEqual(decision, committed)
            self.assertEqual(
                json.loads(config.read_text("utf-8"))["workers"],
                11,
            )
            completion = committed["candidate_stage_completions"][
                "static_review"
            ]
            self.assertEqual(
                completion["result"]["reason"],
                "candidate_identity_drift",
            )
            self.assertGreater(completion["duration_seconds"], 0.0)
            self.assertGreater(
                committed["controlled_spend_seconds"],
                starting_spend,
            )
            self.assertFalse(
                (run_dir / "candidate_stage_intent.json").exists()
            )
            self.assertFalse(
                (run_dir / "candidate_stage_complete.json").exists()
            )

    def test_task5_seals_identity_refreshes_local_context_and_preserves_provenance(
        self,
    ) -> None:
        """Task5 uses only sealed readiness/evidence facts across the existing bundle."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(
                control,
                root,
                capability_ids=("gpu-execute", "pytorch.profiler"),
            )
            workload_fixtures._enable_active_diagnosis(control, root)
            helper.controller.start_run(control, run_dir)
            helper._authorize_active_run(control, run_dir, "grant-task5-context")

            active = run_dir / "active_diagnosis"
            context_path = run_dir / "diagnosis_context.json"
            context = json.loads(context_path.read_text("utf-8"))
            mirror = json.loads((active / "knowledge_context.json").read_text("utf-8"))

            identity = context["knowledge_identity"]
            self.assertEqual(identity["gpu_architecture"]["status"], "unknown")
            self.assertIsNone(identity["gpu_architecture"]["value"])
            self.assertNotEqual(
                identity["gpu_architecture"]["value"],
                "fixture-adapter",
            )
            self.assertEqual(context["knowledge_context"], mirror)
            before_input_sha = context["knowledge_context"]["input_sha256"]
            before_evidence_sha = context["knowledge_context"]["evidence_sha256"]
            before_model_sha = context["knowledge_context"]["performance_model_sha256"]

            hypothesis, request = helper._active_proposal(run_dir)
            helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            knowledge_module = (
                helper.controller._load_diagnostic_knowledge_module()
            )
            captured_frozen_inputs = []

            def capture_knowledge_inputs(frozen_inputs, *, limit):
                captured_frozen_inputs.append(copy.deepcopy(frozen_inputs))
                return knowledge_module.build_knowledge_context(
                    frozen_inputs,
                    limit=limit,
                )

            with mock.patch.object(
                helper.controller,
                "_load_diagnostic_knowledge_module",
            ) as load_knowledge:
                load_knowledge.return_value.build_knowledge_context.side_effect = (
                    capture_knowledge_inputs
                )
                helper.controller.collect_active_diagnosis_evidence(
                    control,
                    run_dir,
                )

            refreshed = json.loads(context_path.read_text("utf-8"))
            refreshed_mirror = json.loads(
                (active / "knowledge_context.json").read_text("utf-8")
            )
            self.assertEqual(refreshed["knowledge_context"], refreshed_mirror)
            self.assertNotEqual(
                before_input_sha,
                refreshed["knowledge_context"]["input_sha256"],
            )
            self.assertEqual(
                before_evidence_sha,
                refreshed["knowledge_context"]["evidence_sha256"],
            )
            self.assertNotEqual(
                before_model_sha,
                refreshed["knowledge_context"]["performance_model_sha256"],
            )
            summary = refreshed["evidence_results"][-1]
            frozen_with_evidence = next(
                item
                for item in reversed(captured_frozen_inputs)
                if item["active_evidence_results"]
            )
            self.assertEqual(
                len(frozen_with_evidence["active_evidence_results"]),
                1,
            )
            envelope = frozen_with_evidence["active_evidence_results"][0]
            catalog = json.loads(
                (active / "action_catalog.json").read_text("utf-8")
            )
            catalog_action = next(
                item
                for item in catalog["actions"]
                if item["action_id"] == summary["action_id"]
            )
            contract = json.loads(
                (active / "analysis_contract.json").read_text("utf-8")
            )
            contract_action = next(
                item
                for item in contract["actions"]
                if item["action_id"] == summary["action_id"]
            )
            result_path = run_dir / summary["result_path"]
            sealed_result = json.loads(result_path.read_text("utf-8"))
            self.assertEqual(envelope["action_id"], summary["action_id"])
            self.assertEqual(
                envelope["evidence_kind"],
                catalog_action["evidence_kind"],
            )
            self.assertEqual(
                envelope["adapter_implementation_sha256"],
                contract_action["adapter_sha256"],
            )
            self.assertEqual(
                envelope["result_sha256"],
                summary["result_sha256"],
            )
            self.assertEqual(
                envelope["observations"],
                sealed_result["observations"],
            )
            self.assertNotIn("stdout", json.dumps(envelope))
            mirror_path = active / "knowledge_context.json"
            mirror_path.unlink()
            state_before_repair = helper.controller.read_run_state(run_dir)
            ledger_before_repair = (
                helper.controller._verify_active_diagnosis_ledger(run_dir)
            )
            with mock.patch.object(
                helper.controller,
                "_load_diagnostic_knowledge_module",
                side_effect=AssertionError(
                    "state-bound mirror repair must not re-query the knowledge package"
                ),
            ):
                repaired_state = helper.controller.resume_run(run_dir)
            self.assertEqual(repaired_state, state_before_repair)
            self.assertEqual(
                helper.controller._verify_active_diagnosis_ledger(run_dir),
                ledger_before_repair,
            )
            self.assertEqual(
                json.loads(mirror_path.read_text("utf-8")),
                refreshed["knowledge_context"],
            )

            self.assertEqual(
                summary["result_sha256"],
                helper.controller._canonical_digest(
                    helper.controller.load_json_object(result_path)
                ),
            )
            tampered = json.loads(result_path.read_text("utf-8"))
            tampered["observations"] = {"semantic_observations": []}
            result_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(
                helper.controller.ValidationError, "digest|artifact|context"
            ):
                helper.controller.resume_run(run_dir)



if __name__ == "__main__":
    unittest.main()
