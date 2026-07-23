from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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
                    authorization={"max_seconds": 100.0},
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
                    "controller_elapsed_or_identity_matched_history",
                )
                self.assertEqual(
                    investment["next_feedback_point"], action_id
                )
                if claim_layer != "kernel":
                    self.assertNotEqual(decision["next_action"]["action_id"], "ncu-targeted")

    def test_controller_stale_bound_never_enters_unexecutable_collection(self) -> None:
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
            original_inputs = helper.controller._diagnostic_investment_inputs
            evidence_root = run_dir / "active_diagnosis" / "evidence"
            evidence_entries_before = {
                path.relative_to(evidence_root)
                for path in evidence_root.rglob("*")
            }

            def stale_inputs(*args, **kwargs):
                inputs = original_inputs(*args, **kwargs)
                inputs["candidate_history"] = [
                    {
                        "hypothesis_id": hypothesis_id,
                        "action_id": f"implement-{hypothesis_id}",
                        "implementation_status": "available",
                        "identity_digest": "0" * 64,
                        "elapsed_seconds": 2.0,
                    }
                    for hypothesis_id in (
                        "h-framework-gap",
                        "h-kernel-bound",
                    )
                ]
                return inputs

            with mock.patch.object(
                helper.controller,
                "_diagnostic_investment_inputs",
                side_effect=stale_inputs,
            ):
                state = helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )

            selection = json.loads(
                (run_dir / "active_diagnosis" / "evidence_selection.json").read_text(
                    "utf-8"
                )
            )
            decision = json.loads(
                (run_dir / "active_diagnosis" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(selection["status"], "sufficient")
            self.assertIsNone(selection["selected_request"])
            self.assertEqual(decision["decision"], "REVIEW_REQUIRED")
            self.assertEqual(
                decision["next_action"]["authorization_reason"],
                "refresh_action_unavailable",
            )
            self.assertEqual(state["next_action"], "review_required")
            self.assertEqual(helper.controller.resume_run(run_dir), state)
            self.assertEqual(
                {
                    path.relative_to(evidence_root)
                    for path in evidence_root.rglob("*")
                },
                evidence_entries_before,
            )

    def test_persisted_brief_clears_fallback_suppressed_by_unknown_primary(self) -> None:
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
            original_inputs = helper.controller._diagnostic_investment_inputs
            model = json.loads(
                (run_dir / "active_diagnosis" / "performance_model.json").read_text(
                    "utf-8"
                )
            )
            identity_digest = hashlib.sha256(
                json.dumps(
                    model["identities"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

            def mixed_inputs(*args, **kwargs):
                inputs = original_inputs(*args, **kwargs)
                inputs["candidate_proposals"] = [
                    {
                        "proposal_id": "proposal-kernel-bound",
                        "hypothesis_id": "h-kernel-bound",
                        "action_id": "implement-h-kernel-bound",
                        "identity_digest": identity_digest,
                        "p50_seconds": 1.0,
                        "p90_seconds": 2.0,
                        "basis": "user_authorized_upper_bound",
                        "freshness": "current",
                    }
                ]
                return inputs

            with mock.patch.object(
                helper.controller,
                "_diagnostic_investment_inputs",
                side_effect=mixed_inputs,
            ):
                state = helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )

            persisted = json.loads(
                (run_dir / "active_diagnosis" / "investment_brief.json").read_text(
                    "utf-8"
                )
            )
            self.assertEqual(state["next_action"], "review_required")
            self.assertIsNone(persisted["selected_action"])
            self.assertEqual(
                persisted["blocked_action"]["action_id"],
                "implement-h-framework-gap",
            )
            self.assertEqual(
                persisted["next_feedback_point"],
                "after_authorization_decision",
            )
            self.assertIn(
                "implement-h-kernel-bound", persisted["skipped_action_ids"]
            )

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
            first = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.assertEqual(first["next_action"], "review_required")
            change = helper._change("slow")
            change["diagnosis_ids"] = ["h-framework-gap"]
            first = helper.controller.seal_active_diagnosis_candidate_proposal(
                control,
                run_dir,
                {
                    "proposal_id": "proposal-framework-slow",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                },
            )
            self.assertEqual(first["next_action"], "register_change")
            helper.controller.register_change(control, run_dir, change)
            (project / "configs" / "value.json").write_text(
                '{"workers": 8}\n', encoding="utf-8"
            )

            candidate_decision = helper.controller.evaluate_change(run_dir)

            self.assertEqual(candidate_decision["status"], "rejected")
            after_failure = helper.controller.read_run_state(run_dir)
            self.assertEqual(after_failure["next_action"], "propose_hypotheses")
            self.assertNotIn("candidate_proposal_id", after_failure)
            self.assertNotIn("candidate_proposal_digest", after_failure)
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            history = context["candidate_history"]
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["hypothesis_id"], "h-framework-gap")
            self.assertEqual(history[0]["implementation_status"], "failed")
            self.assertEqual(len(history[0]["identity_digest"]), 64)
            self.assertEqual(context["candidate_proposals"], [])
            self.assertEqual(
                context["candidate_proposal_archive"][0]["proposal"]["proposal_id"],
                "proposal-framework-slow",
            )
            self.assertEqual(
                context["candidate_proposal_archive"][0]["archive_reason"],
                "candidate_failed",
            )
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
            self.assertEqual(second["next_action"], "review_required")
            self.assertEqual(
                second_decision["primary_diagnosis"]["hypothesis_id"],
                "h-kernel-bound",
            )
            self.assertEqual(
                second_decision["next_action"]["hypothesis_id"],
                "h-kernel-bound",
            )

    def test_failed_candidate_then_map_refresh_archives_old_proposal_and_requests_fallback_cost(self) -> None:
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
            helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            change = helper._change("slow")
            change["diagnosis_ids"] = ["h-framework-gap"]
            helper.controller.seal_active_diagnosis_candidate_proposal(
                control,
                run_dir,
                {
                    "proposal_id": "proposal-before-map-refresh",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                },
            )
            helper.controller.register_change(control, run_dir, change)
            (project / "configs" / "value.json").write_text(
                '{"workers": 8}\n', encoding="utf-8"
            )
            helper.controller.evaluate_change(run_dir)

            active = run_dir / "active_diagnosis"

            def bind(value: dict, requests: dict) -> None:
                epoch = json.loads((active / "epoch.json").read_text("utf-8"))
                execution_map = json.loads(
                    (active / "execution_map.json").read_text("utf-8")
                )
                catalog = json.loads(
                    (active / "evidence_catalog.json").read_text("utf-8")
                )
                value["execution_map_sha256"] = helper.controller._load_execution_map_module().execution_map_digest(
                    execution_map, epoch=epoch, evidence_catalog=catalog
                )
                admitted = helper.controller._load_hypothesis_space_module().validate_hypothesis_set(
                    value,
                    epoch=epoch,
                    execution_map=execution_map,
                    evidence_catalog=catalog,
                )
                requests["epoch_sha256"] = helper.controller._load_execution_map_module().epoch_digest(
                    epoch
                )
                requests["hypothesis_set_sha256"] = admitted[
                    "hypothesis_set_sha256"
                ]

            refresh_hypothesis = {
                "hypothesis_id": "h-map-refresh",
                "kind": "mechanism",
                "scope_node_ids": ["gpu-kernel"],
                "statement": "Fresh compiler timing may change the execution map.",
                "mechanism": "compiler_timing_refresh",
                "claim_layer": "kernel",
                "disposition": "active",
                "confidence": "plausible",
                "support_evidence_ids": ["ev-global-scan"],
                "oppose_evidence_ids": [],
                "missing_evidence_kinds": ["compiler_sass"],
                "falsification_question": "Does fresh SASS timing leave the map unchanged?",
            }
            hypothesis["hypotheses"].append(refresh_hypothesis)
            request["request_set_id"] = "requests-map-refresh"
            request["requests"] = [
                {
                    "request_id": "req-map-refresh",
                    "action_id": "compiler-sass-inspection",
                    "question": "Does fresh compiler evidence change GPU timing?",
                    "target_hypothesis_ids": ["h-map-refresh"],
                    "exclusive_pairs": [],
                    "outcomes": [
                        {
                            "outcome_id": "sass-refreshed",
                            "supports": [],
                            "opposes": ["h-map-refresh"],
                        },
                        {
                            "outcome_id": "sass-unchanged",
                            "supports": ["h-map-refresh"],
                            "opposes": [],
                        },
                    ],
                }
            ]
            bind(hypothesis, request)
            selected = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            self.assertEqual(selected["next_action"], "collect_evidence")
            before_map = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )["execution_map_sha256"]

            helper.controller.collect_active_diagnosis_evidence(control, run_dir)

            refreshed_context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            self.assertNotEqual(
                refreshed_context["execution_map_sha256"], before_map
            )
            refresh_evidence = next(
                evidence_id
                for evidence_id, item in json.loads(
                    (active / "evidence_catalog.json").read_text("utf-8")
                ).items()
                if item["kind"] == "compiler_sass"
            )
            refresh_hypothesis.update(
                {
                    "disposition": "rejected",
                    "confidence": "inconclusive",
                    "support_evidence_ids": [],
                    "oppose_evidence_ids": [refresh_evidence],
                    "missing_evidence_kinds": [],
                }
            )
            request["request_set_id"] = "requests-after-map-refresh"
            bind(hypothesis, request)

            next_state = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            next_decision = json.loads((active / "decision.json").read_text("utf-8"))
            final_context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            self.assertEqual(next_state["next_action"], "review_required")
            self.assertEqual(next_decision["terminal_reason"], "cost_unavailable")
            self.assertEqual(
                next_decision["primary_diagnosis"]["hypothesis_id"],
                "h-kernel-bound",
            )
            self.assertEqual(final_context["candidate_proposals"], [])
            archived = final_context["candidate_proposal_archive"]
            self.assertEqual(
                archived[0]["proposal"]["proposal_id"],
                "proposal-before-map-refresh",
            )

    def test_execution_map_refresh_archives_a_still_active_candidate_proposal(self) -> None:
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
            helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            helper.controller.seal_active_diagnosis_candidate_proposal(
                control,
                run_dir,
                {
                    "proposal_id": "proposal-before-direct-map-refresh",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                },
            )
            active = run_dir / "active_diagnosis"
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            epoch = json.loads((active / "epoch.json").read_text("utf-8"))
            execution_map = json.loads(
                (active / "execution_map.json").read_text("utf-8")
            )
            catalog = json.loads(
                (active / "evidence_catalog.json").read_text("utf-8")
            )
            policy = json.loads(
                (active / "selection_policy.json").read_text("utf-8")
            )
            gpu_node = next(
                item
                for item in execution_map["nodes"]
                if item["node_id"] == "gpu-kernel"
            )
            gpu_node["duration_us"] = float(gpu_node["duration_us"]) - 1.0

            refreshed = helper.controller._refresh_active_diagnosis_context(
                run_dir,
                context,
                epoch,
                execution_map,
                catalog,
                policy,
                {
                    "action_id": "compiler-sass-inspection",
                    "duration_seconds": 1.0,
                },
                1.0,
            )

            self.assertEqual(refreshed["candidate_proposals"], [])
            self.assertEqual(
                refreshed["candidate_proposal_archive"][0]["archive_reason"],
                "analysis_identity_changed",
            )
            self.assertEqual(
                refreshed["candidate_proposal_archive"][0]["proposal"][
                    "proposal_id"
                ],
                "proposal-before-direct-map-refresh",
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

    def test_no_mock_supported_direction_seals_proposal_then_registers_matching_change(self) -> None:
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

            paused = helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            initial = json.loads(
                (run_dir / "active_diagnosis" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(paused["next_action"], "review_required")
            self.assertEqual(initial["decision"], "REVIEW_REQUIRED")
            self.assertEqual(initial["terminal_reason"], "cost_unavailable")

            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            proposal = {
                "proposal_id": "proposal-framework-0001",
                "hypothesis_id": "h-framework-gap",
                "mutation_scope": "project",
                "risk": "low",
                "candidate_digest": helper.controller._canonical_digest(
                    change["candidate"]
                ),
                "change_set_digest": helper.controller._canonical_digest(change),
                "estimated_cost": {
                    "p50_seconds": 10.0,
                    "p90_seconds": 20.0,
                    "basis": "user_authorized_upper_bound",
                },
                "fresh_until_epoch": time.time() + 600.0,
            }
            authorized = helper.controller.seal_active_diagnosis_candidate_proposal(
                control, run_dir, proposal
            )
            decision = json.loads(
                (run_dir / "active_diagnosis" / "decision.json").read_text("utf-8")
            )
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )

            self.assertEqual(authorized["next_action"], "register_change")
            self.assertEqual(decision["decision"], "PURSUE")
            self.assertEqual(
                decision["next_action"]["proposal_id"], "proposal-framework-0001"
            )
            self.assertEqual(len(context["candidate_proposals"]), 1)
            sealed = context["candidate_proposals"][0]
            self.assertEqual(sealed["workload_source_hash"], authorized["workload_source_hash"])
            self.assertEqual(sealed["epoch_sha256"], context["epoch_sha256"])
            self.assertEqual(sealed["execution_map_sha256"], context["execution_map_sha256"])
            self.assertNotIn(sealed, context["candidate_history"])

            registered = helper.controller.register_change(control, run_dir, change)
            self.assertEqual(registered["next_action"], "edit_then_evaluate")
            self.assertEqual(
                registered["candidate_proposal_id"], "proposal-framework-0001"
            )

    def test_cli_seals_supported_candidate_resumes_and_registers_change(self) -> None:
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
            proposal_path = root / "proposal.json"
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

            paused = cli(
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
            self.assertEqual(paused.returncode, 0, paused.stderr)
            self.assertEqual(json.loads(paused.stdout)["next_action"], "review_required")

            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            proposal = {
                "proposal_id": "proposal-cli-framework-0001",
                "hypothesis_id": "h-framework-gap",
                "mutation_scope": "project",
                "risk": "low",
                "candidate_digest": helper.controller._canonical_digest(
                    change["candidate"]
                ),
                "change_set_digest": helper.controller._canonical_digest(change),
                "estimated_cost": {
                    "p50_seconds": 10.0,
                    "p90_seconds": 20.0,
                    "basis": "user_authorized_upper_bound",
                },
                "fresh_until_epoch": time.time() + 600.0,
            }
            change_path.write_text(json.dumps(change), encoding="utf-8")
            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

            rejected = copy.deepcopy(proposal)
            rejected["api_token"] = "must-not-leak-cli-secret"
            proposal_path.write_text(json.dumps(rejected), encoding="utf-8")
            sensitive = cli(
                "seal-candidate-proposal",
                "--control",
                str(control_path),
                "--run-dir",
                str(run_dir),
                "--proposal",
                str(proposal_path),
            )
            self.assertEqual(sensitive.returncode, 2)
            self.assertNotIn("must-not-leak-cli-secret", sensitive.stderr)
            self.assertNotIn("must-not-leak-cli-secret", sensitive.stdout)

            proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
            sealed = cli(
                "seal-candidate-proposal",
                "--control",
                str(control_path),
                "--run-dir",
                str(run_dir),
                "--proposal",
                str(proposal_path),
            )
            self.assertEqual(sealed.returncode, 0, sealed.stderr)
            self.assertEqual(json.loads(sealed.stdout)["next_action"], "register_change")

            resumed = cli("resume", "--run-dir", str(run_dir))
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(json.loads(resumed.stdout)["next_action"], "register_change")

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
            self.assertEqual(registered_state["next_action"], "edit_then_evaluate")
            self.assertEqual(
                registered_state["candidate_proposal_id"],
                "proposal-cli-framework-0001",
            )

    def test_active_change_set_requires_exact_ids_and_matching_sealed_digests(self) -> None:
        for mismatch in ("extra_diagnosis", "candidate_digest", "change_digest"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (
                    helper,
                    control,
                    run_dir,
                    _project,
                    hypothesis,
                    request,
                ) = self._controller_with_two_supported_directions(root)
                helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
                change = helper._change("fast")
                change["diagnosis_ids"] = ["h-framework-gap"]
                proposal = {
                    "proposal_id": "proposal-framework-0001",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                }
                helper.controller.seal_active_diagnosis_candidate_proposal(
                    control, run_dir, proposal
                )
                if mismatch == "extra_diagnosis":
                    change["diagnosis_ids"].append("h-kernel-bound")
                elif mismatch == "candidate_digest":
                    change["candidate"]["revision"] = "optimized-drift"
                else:
                    change["hypothesis"] = "different sealed change"

                with self.assertRaisesRegex(
                    helper.controller.ValidationError,
                    "exactly.*authorized|proposal.*digest|digest.*proposal",
                ):
                    helper.controller.register_change(control, run_dir, change)

                context = json.loads(
                    (run_dir / "diagnosis_context.json").read_text("utf-8")
                )
                self.assertEqual(context["candidate_history"], [])

    def test_expired_candidate_proposal_returns_to_review_and_can_be_resealed(self) -> None:
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
            helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]

            def proposal(proposal_id: str, fresh_until: float) -> dict:
                return {
                    "proposal_id": proposal_id,
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": fresh_until,
                }

            with mock.patch.object(helper.controller.time, "time", return_value=100.0):
                helper.controller.seal_active_diagnosis_candidate_proposal(
                    control, run_dir, proposal("proposal-expiring", 150.0)
                )
            with mock.patch.object(helper.controller.time, "time", return_value=200.0):
                stale = helper.controller.register_change(control, run_dir, change)

            self.assertEqual(stale["next_action"], "review_required")
            self.assertEqual(stale["terminal_reason"], "proposal_stale")
            stale_decision = json.loads(
                (run_dir / "active_diagnosis" / "decision.json").read_text("utf-8")
            )
            self.assertEqual(stale_decision["decision"], "REVIEW_REQUIRED")
            self.assertEqual(stale_decision["terminal_reason"], "proposal_stale")
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            self.assertEqual(context["candidate_proposals"], [])
            self.assertEqual(
                context["candidate_proposal_archive"][0]["archive_reason"],
                "proposal_stale",
            )

            with mock.patch.object(helper.controller.time, "time", return_value=250.0):
                resealed = helper.controller.seal_active_diagnosis_candidate_proposal(
                    control, run_dir, proposal("proposal-resealed", 500.0)
                )
                registered = helper.controller.register_change(
                    control, run_dir, change
                )

            self.assertEqual(resealed["next_action"], "register_change")
            self.assertEqual(registered["next_action"], "edit_then_evaluate")
            self.assertEqual(registered["candidate_proposal_id"], "proposal-resealed")

    def test_resume_detects_an_expired_candidate_proposal_without_a_change_set(self) -> None:
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
            helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            with mock.patch.object(helper.controller.time, "time", return_value=100.0):
                helper.controller.seal_active_diagnosis_candidate_proposal(
                    control,
                    run_dir,
                    {
                        "proposal_id": "proposal-expiring-on-resume",
                        "hypothesis_id": "h-framework-gap",
                        "mutation_scope": "project",
                        "risk": "low",
                        "candidate_digest": helper.controller._canonical_digest(
                            change["candidate"]
                        ),
                        "change_set_digest": helper.controller._canonical_digest(
                            change
                        ),
                        "estimated_cost": {
                            "p50_seconds": 10.0,
                            "p90_seconds": 20.0,
                            "basis": "user_authorized_upper_bound",
                        },
                        "fresh_until_epoch": 150.0,
                    },
                )

            with mock.patch.object(helper.controller.time, "time", return_value=200.0):
                stale = helper.controller.resume_run(run_dir)

            self.assertEqual(stale["next_action"], "review_required")
            self.assertEqual(stale["terminal_reason"], "proposal_stale")
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
            self.assertEqual(context["candidate_proposals"], [])
            self.assertEqual(
                context["candidate_proposal_archive"][0]["archive_reason"],
                "proposal_stale",
            )

    def test_candidate_proposal_seal_recovers_each_interrupted_transition_write(self) -> None:
        targets = (
            "diagnosis_context.json",
            "candidate_proposal.json",
            "decision.json",
            "investment_brief.json",
            "candidate-proposal-ledger",
            "state_commit.json",
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (
                    helper,
                    control,
                    run_dir,
                    _project,
                    hypothesis,
                    request,
                ) = self._controller_with_two_supported_directions(root)
                helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
                change = helper._change("fast")
                change["diagnosis_ids"] = ["h-framework-gap"]
                proposal = {
                    "proposal_id": "proposal-fault",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                }
                pending = run_dir / "active_diagnosis" / "pending_transition.json"
                original_atomic = helper.controller._atomic_json
                interrupted = False

                def matches(path: Path) -> bool:
                    relative = str(path.relative_to(run_dir))
                    if target == "candidate_proposal.json":
                        return relative.endswith(
                            "candidate_proposals/proposal-fault.json"
                        )
                    if target == "candidate-proposal-ledger":
                        return "ledger/" in relative and relative.endswith(
                            "-candidate-proposal.json"
                        )
                    if target == "state_commit.json":
                        return path.name == target and pending.is_file()
                    return path.name == target

                def interrupt(path, value):
                    nonlocal interrupted
                    path = Path(path)
                    if matches(path) and not interrupted:
                        if not pending.is_file():
                            raise AssertionError(
                                "recoverable intent was not persisted before writes"
                            )
                        interrupted = True
                        raise OSError(f"interrupted seal at {target}")
                    return original_atomic(path, value)

                with mock.patch.object(
                    helper.controller, "_atomic_json", side_effect=interrupt
                ):
                    with self.assertRaisesRegex(OSError, "interrupted seal"):
                        helper.controller.seal_active_diagnosis_candidate_proposal(
                            control, run_dir, proposal
                        )

                recovered = helper.controller.resume_run(run_dir)
                context = json.loads(
                    (run_dir / "diagnosis_context.json").read_text("utf-8")
                )
                events = helper.controller._verify_active_diagnosis_ledger(run_dir)
                self.assertEqual(recovered["next_action"], "register_change")
                self.assertEqual(
                    [item["proposal_id"] for item in context["candidate_proposals"]],
                    ["proposal-fault"],
                )
                self.assertEqual(
                    sum(item["event_type"] == "candidate-proposal" for item in events),
                    1,
                )
                self.assertFalse(pending.exists())
                self.assertEqual(
                    recovered["diagnosis_context_sha256"],
                    helper.controller._canonical_digest(context),
                )

    def test_pending_candidate_transition_rejects_tampering_even_with_a_rehashed_marker(self) -> None:
        for rehash_marker in (False, True):
            with self.subTest(rehash_marker=rehash_marker), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (
                    helper,
                    control,
                    run_dir,
                    _project,
                    hypothesis,
                    request,
                ) = self._controller_with_two_supported_directions(root)
                helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
                change = helper._change("fast")
                change["diagnosis_ids"] = ["h-framework-gap"]
                proposal = {
                    "proposal_id": "proposal-tamper",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                }
                original_atomic = helper.controller._atomic_json
                interrupted = False

                def interrupt(path, value):
                    nonlocal interrupted
                    path = Path(path)
                    if path.name == "diagnosis_context.json" and not interrupted:
                        interrupted = True
                        raise OSError("interrupt before transition outputs")
                    return original_atomic(path, value)

                with mock.patch.object(
                    helper.controller, "_atomic_json", side_effect=interrupt
                ):
                    with self.assertRaisesRegex(OSError, "interrupt before"):
                        helper.controller.seal_active_diagnosis_candidate_proposal(
                            control, run_dir, proposal
                        )

                active = run_dir / "active_diagnosis"
                pending_path = active / "pending_transition.json"
                marker_path = active / "pending_transition_commit.json"
                pending = json.loads(pending_path.read_text("utf-8"))
                pending["target_state"]["next_action"] = "done"
                pending_path.write_text(json.dumps(pending), encoding="utf-8")
                if rehash_marker:
                    marker = json.loads(marker_path.read_text("utf-8"))
                    marker["intent_sha256"] = helper.controller._canonical_digest(
                        pending
                    )
                    marker_path.write_text(json.dumps(marker), encoding="utf-8")

                with self.assertRaisesRegex(
                    helper.controller.ValidationError,
                    "transition.*(digest|state changes|target)",
                ):
                    helper.controller.resume_run(run_dir)
                state = helper.controller.read_run_state(run_dir)
                self.assertEqual(state["next_action"], "review_required")

    def test_candidate_transition_prepare_and_marker_interruptions_recover_safely(self) -> None:
        for target in ("pending_transition.json", "pending_transition_commit.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (
                    helper,
                    control,
                    run_dir,
                    _project,
                    hypothesis,
                    request,
                ) = self._controller_with_two_supported_directions(root)
                helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
                change = helper._change("fast")
                change["diagnosis_ids"] = ["h-framework-gap"]
                proposal = {
                    "proposal_id": f"proposal-{target.split('.')[0]}",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                }
                original_atomic = helper.controller._atomic_json
                interrupted = False

                def interrupt(path, value):
                    nonlocal interrupted
                    path = Path(path)
                    if path.name == target and not interrupted:
                        interrupted = True
                        raise OSError(f"interrupt transition metadata at {target}")
                    return original_atomic(path, value)

                with mock.patch.object(
                    helper.controller, "_atomic_json", side_effect=interrupt
                ):
                    with self.assertRaisesRegex(OSError, "interrupt transition"):
                        helper.controller.seal_active_diagnosis_candidate_proposal(
                            control, run_dir, proposal
                        )

                recovered = helper.controller.resume_run(run_dir)
                if target == "pending_transition.json":
                    self.assertEqual(recovered["next_action"], "review_required")
                    self.assertNotIn(
                        "active_diagnosis_pending_transition_sha256", recovered
                    )
                    recovered = helper.controller.seal_active_diagnosis_candidate_proposal(
                        control, run_dir, proposal
                    )
                self.assertEqual(recovered["next_action"], "register_change")
                self.assertFalse(
                    (run_dir / "active_diagnosis" / "pending_transition.json").exists()
                )
                self.assertFalse(
                    (
                        run_dir
                        / "active_diagnosis"
                        / "pending_transition_commit.json"
                    ).exists()
                )

    def test_candidate_failure_recovers_each_interrupted_transition_write(self) -> None:
        targets = (
            "decision.json",
            "diagnosis_context.json",
            "candidate-ledger",
            "state_commit.json",
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (
                    helper,
                    control,
                    run_dir,
                    project,
                    hypothesis,
                    request,
                ) = self._controller_with_two_supported_directions(root)
                helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
                change = helper._change("slow")
                change["diagnosis_ids"] = ["h-framework-gap"]
                helper.controller.seal_active_diagnosis_candidate_proposal(
                    control,
                    run_dir,
                    {
                        "proposal_id": "proposal-failure-fault",
                        "hypothesis_id": "h-framework-gap",
                        "mutation_scope": "project",
                        "risk": "low",
                        "candidate_digest": helper.controller._canonical_digest(
                            change["candidate"]
                        ),
                        "change_set_digest": helper.controller._canonical_digest(
                            change
                        ),
                        "estimated_cost": {
                            "p50_seconds": 10.0,
                            "p90_seconds": 20.0,
                            "basis": "user_authorized_upper_bound",
                        },
                        "fresh_until_epoch": time.time() + 600.0,
                    },
                )
                helper.controller.register_change(control, run_dir, change)
                (project / "configs" / "value.json").write_text(
                    '{"workers": 8}\n', encoding="utf-8"
                )
                pending = run_dir / "active_diagnosis" / "pending_transition.json"
                original_atomic = helper.controller._atomic_json
                interrupted = False

                def matches(path: Path) -> bool:
                    relative = str(path.relative_to(run_dir))
                    if target == "candidate-ledger":
                        return "ledger/" in relative and relative.endswith(
                            "-candidate.json"
                        )
                    if target == "state_commit.json":
                        return path.name == target and pending.is_file()
                    return path.name == target

                def interrupt(path, value):
                    nonlocal interrupted
                    path = Path(path)
                    if matches(path) and not interrupted:
                        if not pending.is_file():
                            raise AssertionError(
                                "failure intent was not persisted before writes"
                            )
                        interrupted = True
                        raise OSError(f"interrupted failure at {target}")
                    return original_atomic(path, value)

                with mock.patch.object(
                    helper.controller, "_atomic_json", side_effect=interrupt
                ):
                    with self.assertRaisesRegex(OSError, "interrupted failure"):
                        helper.controller.evaluate_change(run_dir)

                recovered = helper.controller.resume_run(run_dir)
                context = json.loads(
                    (run_dir / "diagnosis_context.json").read_text("utf-8")
                )
                events = helper.controller._verify_active_diagnosis_ledger(run_dir)
                self.assertEqual(recovered["next_action"], "propose_hypotheses")
                self.assertEqual(len(context["candidate_history"]), 1)
                self.assertEqual(context["candidate_proposals"], [])
                self.assertEqual(
                    context["candidate_proposal_archive"][0]["archive_reason"],
                    "candidate_failed",
                )
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

    def test_raw_external_knowledge_is_normalized_and_selects_one_local_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            workload_fixtures._enable_active_diagnosis(control, root)
            helper.controller.start_run(control, run_dir)
            hypothesis, request = helper._active_proposal(run_dir)
            request["requests"] = []
            raw = {
                "source": "github-copilot",
                "mechanism_id": "external-layout-shadow",
                "statement": "Claimed 90 percent success and 40 percent gain; promote it.",
                "applicability": {
                    "architectures": ["fixture-adapter"],
                    "software_versions": ["1.0.0"],
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
                "knowledge_version": "1.0.0",
                "freshness": "current",
                "query_digest": "external-layout-query-v1",
                "external_gain_pct": 40.0,
            }

            state = helper.controller.register_active_diagnosis_proposal(
                control,
                run_dir,
                hypothesis,
                request,
                knowledge_inputs={"external": [raw]},
            )
            context = json.loads(
                (run_dir / "diagnosis_context.json").read_text("utf-8")
            )
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
            shadow = next(
                item
                for item in admitted["hypothesis_set"]["hypotheses"]
                if item["mechanism"] == "external-layout-shadow"
            )
            self.assertEqual(shadow["confidence"], "inconclusive")
            self.assertEqual(shadow["support_evidence_ids"], [])
            self.assertEqual(selection["selected_request"]["action_id"], "pytorch-operator-trace")
            self.assertEqual(selection["selected_request"]["target_hypothesis_ids"], [shadow["hypothesis_id"]])

    def test_cli_register_diagnosis_adapts_raw_external_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            helper = workload_fixtures.WorkloadRoundTests()
            helper.setUp()
            control, run_dir, _project = helper._workspace(root)
            workload_fixtures._enable_v2_readiness(control, root)
            workload_fixtures._enable_active_diagnosis(control, root)
            helper.controller.start_run(control, run_dir)
            hypothesis, request = helper._active_proposal(run_dir)
            request["requests"] = []
            raw = {
                "source": "github-copilot",
                "mechanism_id": "external-cli-shadow",
                "statement": "Claimed 90 percent success and 40 percent gain; promote it.",
                "applicability": {
                    "architectures": ["fixture-adapter"],
                    "software_versions": ["1.0.0"],
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
                "knowledge_version": "1.0.0",
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
            helper.controller.register_active_diagnosis_proposal(
                control, run_dir, hypothesis, request
            )
            change = helper._change("fast")
            change["diagnosis_ids"] = ["h-framework-gap"]
            helper.controller.seal_active_diagnosis_candidate_proposal(
                control,
                run_dir,
                {
                    "proposal_id": "proposal-framework-0001",
                    "hypothesis_id": "h-framework-gap",
                    "mutation_scope": "project",
                    "risk": "low",
                    "candidate_digest": helper.controller._canonical_digest(
                        change["candidate"]
                    ),
                    "change_set_digest": helper.controller._canonical_digest(change),
                    "estimated_cost": {
                        "p50_seconds": 10.0,
                        "p90_seconds": 20.0,
                        "basis": "user_authorized_upper_bound",
                    },
                    "fresh_until_epoch": time.time() + 600.0,
                },
            )
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
            self.assertEqual(state["candidate_hypothesis_id"], "h-framework-gap")

    def test_active_review_replay_rejects_epoch_and_execution_map_drift(self) -> None:
        for target in ("epoch", "execution_map"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (
                    helper,
                    control,
                    run_dir,
                    project,
                    hypothesis,
                    request,
                ) = self._controller_with_two_supported_directions(root)
                helper.controller.register_active_diagnosis_proposal(
                    control, run_dir, hypothesis, request
                )
                change = helper._change("fast")
                change["diagnosis_ids"] = ["h-framework-gap"]
                change["candidate"]["estimated_cost"]["formal_paired"]["p90_seconds"] = 90.0
                helper.controller.seal_active_diagnosis_candidate_proposal(
                    control,
                    run_dir,
                    {
                        "proposal_id": "proposal-framework-review",
                        "hypothesis_id": "h-framework-gap",
                        "mutation_scope": "project",
                        "risk": "low",
                        "candidate_digest": helper.controller._canonical_digest(
                            change["candidate"]
                        ),
                        "change_set_digest": helper.controller._canonical_digest(
                            change
                        ),
                        "estimated_cost": {
                            "p50_seconds": 10.0,
                            "p90_seconds": 20.0,
                            "basis": "user_authorized_upper_bound",
                        },
                        "fresh_until_epoch": time.time() + 600.0,
                    },
                )
                helper.controller.register_change(control, run_dir, change)
                (project / "configs" / "value.json").write_text(
                    '{"workers": 8}\n', encoding="utf-8"
                )
                state = helper.controller.read_run_state(run_dir)
                started = time.time()
                state["optimization_started_at_epoch"] = started
                state["soft_target_epoch"] = started + 5.0
                state["deadline_epoch"] = started + 5.0
                helper.controller._write_state(run_dir, state)

                decision = helper.controller.evaluate_change(run_dir)
                self.assertEqual(decision["status"], "review_required")
                path = run_dir / "active_diagnosis" / f"{target}.json"
                payload = json.loads(path.read_text("utf-8"))
                payload["tampered_after_review"] = True
                path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(
                    helper.controller.ValidationError,
                    "review-required.*analysis epoch or map",
                ):
                    helper.controller.evaluate_change(run_dir)


if __name__ == "__main__":
    unittest.main()
