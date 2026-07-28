from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_analysis_epoch import epoch_fixture
from tests.test_execution_map import evidence_catalog, map_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py"
REFERENCE_DIR = ROOT / "skills/cuda-kernel-optimizer/references"
REPLAY_FIXTURE = ROOT / "tests/fixtures/knowledge_replay/decision_points.json"
FRESH_REPLAY_FIXTURE = (
    ROOT / "tests/fixtures/knowledge_replay/fresh_controller_cases.json"
)
IMPLEMENTATION_SHA = "1" * 64
REQUEST_SHA = "2" * 64
RESULT_SHA = "3" * 64


def _load():
    name = "cuda_optimizer_diagnostic_knowledge_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_script(filename: str, name: str):
    path = SCRIPT.with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _diagnostic_value(
    *,
    kind: str = "nsys_timeline",
    producer_id: str = "nsys-timeline-adapter",
    producer_version: str = "1.0.0",
    signals=None,
) -> dict:
    return {
        "kind": kind,
        "layer": "workload",
        "summary": "Validated detached diagnostic evidence.",
        "signals": list(signals or ["launch_gap_short_context"]),
        "producer": {
            "id": producer_id,
            "version": producer_version,
            "implementation_sha256": IMPLEMENTATION_SHA,
        },
        "adapter_request_sha256": REQUEST_SHA,
        "recorded_at": 100.0,
        "subject": {"target_sha256": "4" * 64},
        "result": {"artifact_sha256": "5" * 64, "events_total": 12},
    }


def _active_result(semantic_observations=None, **extra_observations) -> dict:
    observations = dict(extra_observations)
    if semantic_observations is not None:
        observations["semantic_observations"] = semantic_observations
    return {
        "action_id": "nsys-global-timeline",
        "evidence_kind": "nsys_timeline",
        "adapter_implementation_sha256": IMPLEMENTATION_SHA,
        "result_sha256": RESULT_SHA,
        "status": "observed",
        "observations": observations,
    }


def _semantic_observation(**updates) -> dict:
    value = {
        "semantic_id": "runtime.launch_gap_us",
        "status": "observed",
        "value": 12.5,
        "unit": "us",
        "scope": ["cpu-submit", "gpu-kernel"],
        "aggregation": "median",
        "tool": {"name": "nsys", "version": "2026.3"},
        "quality": "validated",
    }
    value.update(updates)
    return value


def _observation_rule(**updates) -> dict:
    value = {
        "semantic_id": "runtime.launch_gap_short_context",
        "statuses": ["present"],
        "scope_all": ["cpu-submit", "gpu-kernel"],
        "unit": "state",
        "aggregation": "presence",
        "comparison": None,
    }
    value.update(updates)
    return value


def _identity_fact(value: str) -> dict:
    return {
        "value": value,
        "status": "verified",
        "source_kind": "analysis_contract",
        "source_sha256": "6" * 64,
    }


def _frozen_inputs() -> dict:
    map_module = _load_script("execution_map.py", "knowledge_test_execution_map")
    model_module = _load_script("performance_model.py", "knowledge_test_performance")
    epoch = epoch_fixture()
    execution_map = map_fixture(map_module)
    performance_model = model_module.build_performance_model(
        execution_map,
        minimum_effect_us=10.0,
    )
    knowledge_identity = {
        "schema_version": "cuda-optimizer/knowledge-identity-v1",
        "gpu_architecture": _identity_fact("sm_120"),
        "driver_version": _identity_fact("580.65"),
        "cuda_runtime_version": _identity_fact("13.0"),
        "framework_versions": {"pytorch": _identity_fact("2.11.0")},
        "compiler_versions": {"triton": _identity_fact("3.6.0")},
        "profiler_versions": {"nsys": _identity_fact("2026.3")},
        "workload_contract_sha256": epoch["identities"][
            "workload_contract_sha256"
        ],
        "source_sha256": epoch["identities"]["source_sha256"],
        "environment_sha256": epoch["identities"]["environment_sha256"],
    }
    semantic = [
        _semantic_observation(
            semantic_id="runtime.gpu_idle_gap",
            status="present",
            value=True,
            unit="state",
            scope=["gpu-kernel"],
            aggregation="presence",
        ),
        _semantic_observation(
            semantic_id="kernel.memory_pressure",
            value=80.0,
            unit="percent",
            scope=["gpu-kernel"],
            aggregation="mean",
            tool={"name": "ncu", "version": "2026.2"},
        ),
        _semantic_observation(
            semantic_id="transfer.boundary_ambiguous",
            status="present",
            value=True,
            unit="state",
            scope=["transfer"],
            aggregation="presence",
        ),
    ]
    return {
        "knowledge_identity": knowledge_identity,
        "diagnosis": {
            "primary_category": "mixed",
            "ranked_categories": [
                {"category": "framework", "score": 100},
                {"category": "kernel", "score": 90},
                {"category": "transfer", "score": 80},
                {"category": "io", "score": 70},
                {"category": "cpu_data", "score": 60},
                {"category": "communication", "score": 50},
            ],
        },
        "analysis_epoch": epoch,
        "evidence_catalog": evidence_catalog(),
        "execution_map": execution_map,
        "performance_model": performance_model,
        "diagnostic_evidence": [_diagnostic_value()],
        "active_evidence_results": [_active_result(semantic)],
        "requested_claim": "workload",
        "ready_capability_ids": [
            "cuda.disassembler",
            "pytorch.profiler",
        ],
        "contract_action_ids": [
            "compiler-sass-inspection",
            "pytorch-operator-trace",
        ],
        "available_actions": [
            "compiler-sass-inspection",
            "pytorch-operator-trace",
        ],
        "closed_mechanism_keys": [],
        "candidate_history": [],
    }


def _prepare_valid_contract(reference_dir: Path) -> None:
    source_path = reference_dir / "knowledge_sources.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    for source in sources["sources"]:
        summary = f"Test summary for {source['title']}."
        source.update(
            locator="test section",
            summary=summary,
            summary_sha256=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            status="verified",
        )
    _write_json(source_path, sources)

    card_path = reference_dir / "diagnostic_cards.json"
    cards = json.loads(card_path.read_text(encoding="utf-8"))
    for card in cards["cards"]:
        card.update(
            mechanism_key=card["id"].removeprefix("diagnostic."),
            execution_layers=["gpu"],
            applies_when=["test applicability is observed"],
            required_evidence=["test evidence"],
            cheapest_falsifier={
                "action_id": "nsys-global-timeline",
                "rationale": "Test the mechanism against a bounded timeline.",
            },
            content_status="source_verified",
            case_ids=[],
        )
    _write_json(card_path, cards)

class DiagnosticKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load()

    def test_routes_a_small_framework_launch_context(self) -> None:
        diagnosis = {
            "primary_category": "framework",
            "ranked_categories": [{"category": "framework", "score": 100}],
        }
        execution_map = {
            "nodes": [
                {"node_id": "cpu", "layer": "cpu", "label": "cudaLaunchKernel"},
                {"node_id": "gpu", "layer": "gpu", "label": "decode_attention"},
            ]
        }
        result = self.module.route_cards(diagnosis, execution_map, limit=3)
        self.assertLessEqual(len(result["cards"]), 3)
        self.assertEqual(result["cards"][0]["id"], "diagnostic.framework.launch-gaps")
        self.assertEqual(result["promotion_authority"], "none")
        self.assertTrue(result["cards"][0]["distinguishing_question"])
        self.assertTrue(result["cards"][0]["counter_signals"])

    def test_inconclusive_diagnosis_routes_cross_layer_triage(self) -> None:
        result = self.module.route_cards(
            {"primary_category": None, "ranked_categories": []},
            {"nodes": []},
            limit=3,
        )
        self.assertEqual(result["cards"][0]["id"], "diagnostic.cross-layer.triage")

    def test_cards_are_hints_not_direction_evidence(self) -> None:
        result = self.module.route_cards(
            {
                "primary_category": "kernel",
                "ranked_categories": [{"category": "kernel", "score": 100}],
            },
            {"nodes": [{"node_id": "gpu", "layer": "gpu", "label": "gemm"}]},
            limit=3,
        )
        self.assertEqual(result["promotion_authority"], "none")
        self.assertTrue(all(card["status"] == "routing_only" for card in result["cards"]))

    def test_normalizes_validated_diagnostic_and_active_observations(self) -> None:
        observations = self.module.normalize_observations(
            diagnostic_evidence=[_diagnostic_value()],
            active_evidence_results=[
                _active_result([_semantic_observation()], stall=91)
            ],
        )
        self.assertEqual(
            set(observations[0]),
            {
                "semantic_id",
                "status",
                "value",
                "unit",
                "scope",
                "aggregation",
                "tool",
                "quality",
                "source_digest",
            },
        )
        self.assertEqual(
            [item["semantic_id"] for item in observations],
            sorted(item["semantic_id"] for item in observations),
        )
        active = next(
            item
            for item in observations
            if item["semantic_id"] == "runtime.launch_gap_us"
        )
        self.assertEqual(active["source_digest"], RESULT_SHA)
        self.assertFalse(any(item["semantic_id"] == "stall" for item in observations))

    def test_rejects_unknown_producer_version_and_active_adapter_identity(self) -> None:
        mutations = (
            (
                [_diagnostic_value(producer_id="unknown-adapter")],
                [],
                "producer",
            ),
            (
                [_diagnostic_value(producer_version="9.9.9")],
                [],
                "producer|version",
            ),
            (
                [],
                [_active_result([]) | {"adapter_implementation_sha256": "bad"}],
                "adapter.*identity|SHA",
            ),
        )
        for diagnostic, active, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                self.module.normalize_observations(
                    diagnostic_evidence=diagnostic,
                    active_evidence_results=active,
                )

    def test_rejects_incomplete_explicit_numeric_observations(self) -> None:
        mutations = (
            lambda item: item.pop("unit"),
            lambda item: item.pop("scope"),
            lambda item: item.pop("aggregation"),
            lambda item: item["tool"].pop("version"),
            lambda item: item.pop("quality"),
        )
        for mutation in mutations:
            item = _semantic_observation()
            mutation(item)
            with self.subTest(item=item), self.assertRaisesRegex(
                ValueError, "closed|missing|unit|scope|aggregation|version|quality"
            ):
                self.module.normalize_observations(
                    active_evidence_results=[_active_result([item])]
                )

    def test_rejects_same_source_conflicts_and_deduplicates_identical_values(self) -> None:
        first = _semantic_observation()
        duplicate = _semantic_observation()
        result = self.module.normalize_observations(
            active_evidence_results=[_active_result([first, duplicate])]
        )
        self.assertEqual(len(result), 1)
        conflicting = _semantic_observation(value=14.0)
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.module.normalize_observations(
                active_evidence_results=[_active_result([first, conflicting])]
            )

    def test_counter_permission_error_is_unavailable_not_kernel_counterevidence(self) -> None:
        denied = _semantic_observation(
            semantic_id="kernel.stall",
            status="unavailable",
            value="ERR_NVGPUCTRPERM",
            unit="state",
            aggregation="presence",
            tool={"name": "ncu", "version": "2026.2"},
        )
        envelope = _active_result([denied])
        envelope.update(
            action_id="ncu-targeted-kernel",
            evidence_kind="ncu_kernel",
        )
        result = self.module.normalize_observations(
            active_evidence_results=[envelope]
        )
        self.assertEqual(
            [(item["semantic_id"], item["status"]) for item in result],
            [("profile.counter_access", "unavailable")],
        )
        self.assertFalse(any(item["semantic_id"].startswith("kernel.") for item in result))

    def test_open_active_observations_are_not_guessed(self) -> None:
        result = self.module.normalize_observations(
            active_evidence_results=[
                _active_result(
                    None,
                    stall=91,
                    launch_gap_us=12.5,
                    execution_map_node_updates=[{"node_id": "gpu"}],
                )
            ]
        )
        self.assertEqual(result, [])

    def test_builds_bounded_identity_bound_context_from_frozen_inputs(self) -> None:
        frozen = _frozen_inputs()
        production = self.module.build_knowledge_context(frozen, limit=3)
        self.assertEqual(production["candidates"], [])
        self.assertTrue(production["explanations"])

        reference_dir = self._copied_references()
        identity_sha = _canonical_sha256(frozen["knowledge_identity"])
        case_path = reference_dir / "case_memory.json"
        cases = json.loads(case_path.read_text(encoding="utf-8"))
        exact = next(item for item in cases["cases"] if item["id"] == "R01")
        exact.update(
            replay_status="scoreable",
            identity_match="exact",
            knowledge_identity_sha256=identity_sha,
            content_status="locally_measured",
        )
        rejected = next(item for item in cases["cases"] if item["id"] == "X01")
        rejected["content_status"] = "replay_verified"
        _write_json(case_path, cases)

        action_path = reference_dir / "evidence_action_catalog.json"
        actions = json.loads(action_path.read_text(encoding="utf-8"))
        next(
            item
            for item in actions["actions"]
            if item["action_id"] == "pytorch-operator-trace"
        )["cost"] = "medium"
        _write_json(action_path, actions)

        card_path = reference_dir / "diagnostic_cards.json"
        cards = json.loads(card_path.read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in cards["cards"]}
        duplicate = by_id["diagnostic.cross-layer.triage"]
        duplicate["mechanism_key"] = "framework_launch_gaps"
        framework = by_id["diagnostic.framework.launch-gaps"]
        framework.update(content_status="locally_measured", case_ids=["R01"])
        framework["observation_rules"]["positive"].append(
            _observation_rule(
                semantic_id="framework.shape_fragmentation",
                statuses=["present"],
                scope_all=["framework"],
            )
        )
        framework["observation_rules"]["counter"].append(
            _observation_rule(
                semantic_id="runtime.gpu_idle_gap",
                scope_all=["gpu-kernel"],
            )
        )
        kernel = by_id["diagnostic.kernel.resource-or-memory"]
        kernel.update(content_status="replay_verified", case_ids=["X01"])
        kernel["observation_rules"]["positive"] = [
            _observation_rule(
                semantic_id="kernel.memory_pressure",
                statuses=["observed"],
                scope_all=["gpu-kernel"],
                unit="percent",
                aggregation="mean",
                comparison={"op": "gte", "value": 70.0},
            )
        ]
        cpu_data = by_id["diagnostic.cpu-data.starvation"]
        cpu_data.update(content_status="replay_verified", case_ids=["X01"])
        cpu_data["identity_constraints"]["cuda_runtime_version"] = ["13.0"]
        transfer = by_id["diagnostic.transfer.h2d"]
        transfer.update(content_status="replay_verified", case_ids=["X01"])
        transfer["observation_rules"]["invalidators"] = [
            _observation_rule(
                semantic_id="transfer.boundary_ambiguous",
                scope_all=["transfer"],
            )
        ]
        io_card = by_id["diagnostic.io.request-path"]
        io_card.update(content_status="replay_verified", case_ids=["X01"])
        communication = by_id["diagnostic.communication.collective"]
        communication["identity_constraints"]["gpu_architecture"] = ["sm_999"]
        communication["identity_constraints"]["framework_versions"] = {
            "pytorch": ["9.9.9"]
        }
        communication["requested_claims"] = ["workload"]
        _write_json(card_path, cards)

        old_reference_dir = self.module.REFERENCE_DIR
        old_cards_path = self.module.CARDS_PATH
        self.module.REFERENCE_DIR = reference_dir
        self.module.CARDS_PATH = card_path
        self.addCleanup(setattr, self.module, "REFERENCE_DIR", old_reference_dir)
        self.addCleanup(setattr, self.module, "CARDS_PATH", old_cards_path)

        context = self.module.build_knowledge_context(frozen, limit=3)
        timed = copy.deepcopy(frozen)
        timed["performance_model"]["action_timing_estimates"] = {
            "pytorch-operator-trace": {
                "sample_count": 4,
                "p50_seconds": 12.0,
                "p90_seconds": 18.0,
                "basis": "identity_matched_history",
            }
        }
        timed_context = self.module.build_knowledge_context(timed, limit=3)
        self.assertEqual(
            [item["card_id"] for item in timed_context["candidates"]],
            [item["card_id"] for item in context["candidates"]],
        )
        self.assertNotEqual(
            timed_context["performance_model_sha256"],
            context["performance_model_sha256"],
        )
        self.assertEqual(
            set(context),
            {
                "schema_version",
                "input_sha256",
                "knowledge_package",
                "knowledge_identity_sha256",
                "evidence_sha256",
                "performance_model_sha256",
                "categories",
                "candidates",
                "explanations",
                "rejections",
                "filtered_counts",
                "promotion_authority",
            },
        )
        self.assertEqual(context["promotion_authority"], "none")
        self.assertLessEqual(len(context["candidates"]), 3)
        self.assertLessEqual(
            len(
                json.dumps(
                    context,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ),
            12 * 1024,
        )
        candidate_ids = [item["card_id"] for item in context["candidates"]]
        self.assertEqual(
            candidate_ids,
            [
                "diagnostic.framework.launch-gaps",
                "diagnostic.kernel.resource-or-memory",
            ],
        )
        self.assertTrue(context["candidates"][0]["evidence"]["positive"])
        self.assertTrue(context["candidates"][0]["evidence"]["counter"])
        self.assertTrue(context["candidates"][0]["evidence"]["missing"])
        self.assertTrue(
            any(
                item["card_id"] == "diagnostic.transfer.h2d"
                and item["reason"] == "invalidator_observed"
                for item in context["rejections"]
            )
        )
        self.assertTrue(
            any(
                item["card_id"] == "diagnostic.io.request-path"
                and item["reason"] == "read_only_action_unavailable"
                for item in context["explanations"]
            )
        )
        self.assertTrue(
            any(
                item["card_id"] == "diagnostic.communication.collective"
                and item["reason"] == "identity_mismatch"
                and {
                    "gpu_architecture",
                    "framework_versions.pytorch",
                }.issubset(set(item["details"]))
                for item in context["rejections"]
            )
        )
        claim_mismatch = copy.deepcopy(frozen)
        claim_mismatch["requested_claim"] = "kernel"
        claim_context = self.module.build_knowledge_context(claim_mismatch, limit=3)
        self.assertTrue(
            any(
                item["card_id"] == "diagnostic.communication.collective"
                and item["reason"] == "task_mismatch"
                for item in claim_context["rejections"]
            )
        )
        forged_availability = copy.deepcopy(frozen)
        forged_availability["ready_capability_ids"].remove("pytorch.profiler")
        with self.assertRaisesRegex(
            ValueError, "available_actions must equal derived availability"
        ):
            self.module.build_knowledge_context(forged_availability, limit=3)
        self.assertGreaterEqual(context["filtered_counts"]["canonical_duplicate"], 1)
        self.assertGreaterEqual(context["filtered_counts"]["pareto_dominated"], 1)
        surviving_keys = [
            item["mechanism_key"]
            for field in ("candidates", "explanations")
            for item in context[field]
        ]
        self.assertEqual(len(surviving_keys), len(set(surviving_keys)))
        for candidate in context["candidates"]:
            self.assertTrue(
                {
                    "mechanism_key",
                    "statement",
                    "execution_layers",
                    "scope_node_ids",
                    "evidence_grade",
                    "evidence",
                    "benefit_ceiling",
                    "cheapest_falsifier",
                    "source_ids",
                    "case_ids",
                    "confidence",
                    "promotion_authority",
                }.issubset(candidate)
            )
            self.assertEqual(candidate["confidence"], "inconclusive")
            self.assertEqual(
                candidate["benefit_ceiling"]["performance_model_sha256"],
                context["performance_model_sha256"],
            )
            self.assertTrue(
                candidate["benefit_ceiling"]["qualifies_minimum_effect"]
            )
            self.assertEqual(candidate["promotion_authority"], "none")
            self.assertNotIn("speedup", json.dumps(candidate).lower())
            self.assertNotIn("historical", json.dumps(candidate).lower())

        repeated = self.module.build_knowledge_context(frozen, limit=3)
        self.assertEqual(
            [item["card_id"] for item in repeated["candidates"]],
            candidate_ids,
        )
        changed_input = copy.deepcopy(frozen)
        changed_input["diagnosis"]["ranked_categories"][0]["score"] = 101
        rebound = self.module.build_knowledge_context(changed_input, limit=3)
        self.assertNotEqual(rebound["input_sha256"], context["input_sha256"])

        closed = copy.deepcopy(frozen)
        closed["closed_mechanism_keys"] = ["framework-launch-gaps"]
        closed_context = self.module.build_knowledge_context(closed, limit=3)
        self.assertNotIn(
            "diagnostic.framework.launch-gaps",
            {item["card_id"] for item in closed_context["candidates"]},
        )

        unknown = copy.deepcopy(frozen)
        unknown["knowledge_identity"]["cuda_runtime_version"] = {
            "value": None,
            "status": "unknown",
            "source_kind": "unknown",
            "source_sha256": None,
        }
        unknown_context = self.module.build_knowledge_context(unknown, limit=3)
        self.assertEqual(unknown_context["candidates"], [])
        self.assertFalse(
            any(
                item["reason"] == "exact_case_rejection"
                for item in unknown_context["rejections"]
            )
        )
        self.assertTrue(
            any(
                item["card_id"] == "diagnostic.cpu-data.starvation"
                and item["reason"] == "identity_unverified"
                for item in unknown_context["explanations"]
            )
        )
        unknown_action = copy.deepcopy(frozen)
        unknown_action["available_actions"].append("unknown-action")
        with self.assertRaisesRegex(ValueError, "unknown available action"):
            self.module.build_knowledge_context(unknown_action, limit=3)

        history_bound = copy.deepcopy(frozen)
        history_bound["candidate_history"] = [
            {
                "hypothesis_id": "hyp-1",
                "action_id": "compiler-sass-inspection",
                "implementation_status": "failed",
                "identity_digest": "7" * 64,
                "elapsed_seconds": 1.5,
                "candidate_digest": "8" * 64,
                "decision_digest": "9" * 64,
                "failure_reason": "correctness_failed",
            }
        ]
        history_context = self.module.build_knowledge_context(history_bound, limit=3)
        self.assertNotEqual(
            history_context["input_sha256"],
            context["input_sha256"],
        )
        self.assertEqual(
            [item["card_id"] for item in history_context["candidates"]],
            candidate_ids,
        )

        below_mde = copy.deepcopy(frozen)
        model_module = _load_script(
            "performance_model.py", "knowledge_test_below_mde_model"
        )
        below_mde["performance_model"] = model_module.build_performance_model(
            below_mde["execution_map"],
            minimum_effect_us=1000.0,
        )
        below_context = self.module.build_knowledge_context(below_mde, limit=3)
        self.assertEqual(below_context["candidates"], [])
        self.assertTrue(
            any(
                item["reason"] == "below_minimum_effect"
                for item in below_context["explanations"]
            )
        )

        prior_package_sha = context["knowledge_package"]["sha256"]
        cards["cards"][1]["positive_signals"][0] = "display prose changed only"
        _write_json(card_path, cards)
        changed_package = self.module.build_knowledge_context(frozen, limit=3)
        self.assertNotEqual(
            changed_package["knowledge_package"]["sha256"],
            prior_package_sha,
        )
        self.assertEqual(
            [item["card_id"] for item in changed_package["candidates"]],
            candidate_ids,
        )

        trimmed = self.module.build_knowledge_context(
            frozen,
            limit=3,
            max_bytes=1800,
        )
        trimmed_bytes = json.dumps(
            trimmed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        self.assertLessEqual(len(trimmed_bytes), 1800)
        self.assertEqual(set(trimmed), set(context))
        self.assertLess(
            sum(
                len(trimmed[field])
                for field in ("candidates", "explanations", "rejections")
            ),
            sum(
                len(context[field])
                for field in ("candidates", "explanations", "rejections")
            ),
        )

    def test_validates_closed_knowledge_package(self) -> None:
        result = self.module.validate_knowledge_package(REFERENCE_DIR)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["case_count"], 15)
        self.assertEqual(result["card_count"], 13)

    def _mutated_references(self):
        temporary = tempfile.TemporaryDirectory()
        reference_dir = Path(temporary.name) / "references"
        shutil.copytree(REFERENCE_DIR, reference_dir)
        _prepare_valid_contract(reference_dir)
        self.addCleanup(temporary.cleanup)
        return reference_dir

    def _copied_references(self):
        temporary = tempfile.TemporaryDirectory()
        reference_dir = Path(temporary.name) / "references"
        shutil.copytree(REFERENCE_DIR, reference_dir)
        self.addCleanup(temporary.cleanup)
        return reference_dir

    def test_rejects_incomplete_mechanisms_and_non_read_only_falsifiers(self) -> None:
        for mutation, message in (
            (
                lambda card: card.pop("mechanism_key"),
                "mechanism_key",
            ),
            (
                lambda card: card["cheapest_falsifier"].update(
                    action_id="direction-experiment-project-copy"
                ),
                "read_only",
            ),
            (
                lambda card: card.pop("observation_rules", None),
                "observation_rules",
            ),
            (
                lambda card: card.pop("identity_constraints", None),
                "identity_constraints",
            ),
            (
                lambda card: card.update(
                    identity_constraints={
                        "match": "semver_range",
                        "gpu_architecture": [],
                        "driver_version": [],
                        "cuda_runtime_version": [],
                        "framework_versions": {},
                        "compiler_versions": {},
                        "profiler_versions": {},
                    }
                ),
                "exact_only",
            ),
            (
                lambda card: card.update(
                    observation_rules={
                        "positive": [
                            _observation_rule(
                                comparison={"op": "probability", "value": 0.9}
                            )
                        ],
                        "counter": [],
                        "invalidators": [],
                    }
                ),
                "comparison",
            ),
            (
                lambda card: card.update(
                    observation_rules={
                        "positive": [],
                        "counter": [
                            _observation_rule(statuses=["unavailable"])
                        ],
                        "invalidators": [],
                    }
                ),
                "unavailable",
            ),
            (
                lambda card: card.update(
                    observation_rules={
                        "positive": [
                            _observation_rule(),
                            _observation_rule(
                                scope_all=["gpu-kernel", "cpu-submit"]
                            ),
                        ],
                        "counter": [],
                        "invalidators": [],
                    }
                ),
                "duplicate",
            ),
            (
                lambda card: card.update(
                    observation_rules={
                        "positive": [_observation_rule()],
                        "counter": [
                            _observation_rule(
                                scope_all=["gpu-kernel", "cpu-submit"]
                            )
                        ],
                        "invalidators": [],
                    }
                ),
                "duplicate",
            ),
        ):
            with self.subTest(message=message):
                reference_dir = self._mutated_references()
                path = reference_dir / "diagnostic_cards.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutation(payload["cards"][0])
                _write_json(path, payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.module.validate_knowledge_package(reference_dir)

    def test_rejects_unknown_source_action_and_case_references(self) -> None:
        mutations = (
            (
                "diagnostic_cards.json",
                lambda payload: payload["cards"][0]["source_ids"].append("unknown-source"),
                "unknown source",
            ),
            (
                "diagnostic_cards.json",
                lambda payload: payload["cards"][0]["cheapest_falsifier"].update(
                    action_id="unknown-action"
                ),
                "unknown action",
            ),
            (
                "case_memory.json",
                lambda payload: payload["cases"][0].update(
                    replay_case_id="unknown-case"
                ),
                "case memory ids",
            ),
        )
        for filename, mutation, message in mutations:
            with self.subTest(message=message):
                reference_dir = self._mutated_references()
                path = reference_dir / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutation(payload)
                _write_json(path, payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.module.validate_knowledge_package(reference_dir)

    def test_rejects_incomplete_or_tampered_source_metadata(self) -> None:
        for mutation, message in (
            (
                lambda source: source.pop("locator"),
                "locator",
            ),
            (
                lambda source: source.update(summary="tampered summary"),
                "summary_sha256",
            ),
            (
                lambda source: source.update(last_verified="2026-99-99"),
                "ISO date",
            ),
        ):
            with self.subTest(message=message):
                reference_dir = self._mutated_references()
                path = reference_dir / "knowledge_sources.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutation(payload["sources"][0])
                _write_json(path, payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.module.validate_knowledge_package(reference_dir)

    def test_rejects_historical_benefit_and_unbacked_content_maturity(self) -> None:
        reference_dir = self._mutated_references()
        case_path = reference_dir / "case_memory.json"
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        payload["cases"][0]["replay_status"] = "scoreable"
        _write_json(case_path, payload)
        self.assertEqual(
            self.module.validate_knowledge_package(reference_dir)["status"],
            "passed",
        )

        replay_dir = self._mutated_references()
        replay_case_path = replay_dir / "case_memory.json"
        replay_cases = json.loads(replay_case_path.read_text(encoding="utf-8"))
        rejection = next(item for item in replay_cases["cases"] if item["id"] == "X01")
        rejection["content_status"] = "replay_verified"
        _write_json(replay_case_path, replay_cases)
        replay_card_path = replay_dir / "diagnostic_cards.json"
        replay_cards = json.loads(replay_card_path.read_text(encoding="utf-8"))
        replay_cards["cards"][0].update(
            content_status="replay_verified", case_ids=["X01"]
        )
        _write_json(replay_card_path, replay_cards)
        replay_result = self.module.validate_knowledge_package(replay_dir)
        self.assertIn(
            replay_cards["cards"][0]["id"],
            replay_result["runtime_candidate_card_ids"],
        )

        local_dir = self._mutated_references()
        local_case_path = local_dir / "case_memory.json"
        local_cases = json.loads(local_case_path.read_text(encoding="utf-8"))
        measured = next(item for item in local_cases["cases"] if item["id"] == "R01")
        measured.update(
            replay_status="scoreable",
            content_status="locally_measured",
            identity_match="exact",
            knowledge_identity_sha256=hashlib.sha256(
                b"test complete knowledge identity"
            ).hexdigest(),
        )
        _write_json(local_case_path, local_cases)
        local_card_path = local_dir / "diagnostic_cards.json"
        local_cards = json.loads(local_card_path.read_text(encoding="utf-8"))
        local_cards["cards"][0].update(
            content_status="locally_measured", case_ids=["R01"]
        )
        _write_json(local_card_path, local_cards)
        local_result = self.module.validate_knowledge_package(local_dir)
        self.assertIn(
            local_cards["cards"][0]["id"],
            local_result["runtime_candidate_card_ids"],
        )

        mutations = (
            (
                "case_memory.json",
                lambda payload: payload["cases"][0].update(speedup=1.25),
                "unknown fields",
            ),
            (
                "diagnostic_cards.json",
                lambda payload: payload["cards"][0].update(
                    content_status="replay_verified", case_ids=[]
                ),
                "replay_verified",
            ),
            (
                "diagnostic_cards.json",
                lambda payload: payload["cards"][0].update(
                    content_status="replay_verified", case_ids=["K01"]
                ),
                "replay_verified",
            ),
            (
                "diagnostic_cards.json",
                lambda payload: payload["cards"][0].update(
                    content_status="locally_measured", case_ids=["R01"]
                ),
                "identity digest",
            ),
        )
        for filename, mutation, message in mutations:
            with self.subTest(message=message):
                reference_dir = self._mutated_references()
                path = reference_dir / filename
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutation(payload)
                _write_json(path, payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.module.validate_knowledge_package(reference_dir)

    def test_case_memory_matches_task1_fixture_identity_status_and_digest(self) -> None:
        fixture = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))
        fresh_fixture = json.loads(
            FRESH_REPLAY_FIXTURE.read_text(encoding="utf-8")
        )
        wanted = {
            "R01",
            "R02",
            "R03",
            "R04",
            "R05",
            "R06",
            "X01",
            "K01",
            "K02",
            "R07-fresh-20260728",
            "R08-fresh-20260728",
            "R09-fresh-20260728",
            "R10-fresh-20260728",
            "R11-fresh-20260728",
            "R12-fresh-20260728",
        }
        replay_cases = {
            case["case_id"]: case
            for case in [*fixture["cases"], *fresh_fixture["cases"]]
            if case["case_id"] in wanted
        }
        memory = json.loads((REFERENCE_DIR / "case_memory.json").read_text(encoding="utf-8"))
        self.assertEqual({item["id"] for item in memory["cases"]}, wanted)
        for item in memory["cases"]:
            replay = replay_cases[item["replay_case_id"]]
            self.assertEqual(item["id"], replay["case_id"])
            self.assertEqual(item["replay_status"], replay["replay_eligibility"]["status"])
            self.assertEqual(item["replay_case_sha256"], _canonical_sha256(replay))
            self.assertEqual(
                item["predecision_evidence_sha256"],
                _canonical_sha256(replay["input_snapshot"]),
            )
            self.assertNotIn("speedup", item)
            self.assertNotIn("benefit", item)


if __name__ == "__main__":
    unittest.main()
