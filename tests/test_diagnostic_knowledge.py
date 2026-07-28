from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py"
REFERENCE_DIR = ROOT / "skills/cuda-kernel-optimizer/references"
REPLAY_FIXTURE = ROOT / "tests/fixtures/knowledge_replay/decision_points.json"
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

    def test_validates_closed_knowledge_package(self) -> None:
        result = self.module.validate_knowledge_package(REFERENCE_DIR)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["case_count"], 9)

    def _mutated_references(self):
        temporary = tempfile.TemporaryDirectory()
        reference_dir = Path(temporary.name) / "references"
        shutil.copytree(REFERENCE_DIR, reference_dir)
        _prepare_valid_contract(reference_dir)
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
        wanted = {"R01", "R02", "R03", "R04", "R05", "R06", "X01", "K01", "K02"}
        replay_cases = {
            case["case_id"]: case for case in fixture["cases"] if case["case_id"] in wanted
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
