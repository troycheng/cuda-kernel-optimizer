import copy
import json
import os
import tempfile
import unittest
from unittest import mock
import re
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

from tests import test_workload_controller as workload_fixtures
from tools import build_knowledge_replay


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge_replay"
FRESH_FIXTURE = FIXTURE_DIR / "fresh_controller_cases.json"
POSTFREEZE_FIXTURE = FIXTURE_DIR / "postfreeze_controller_cases.json"
POSTFREEZE_BASELINE = FIXTURE_DIR / "v1_2_postfreeze_baseline.json"
RUNTIME_INPUT_FIELDS = {
    "knowledge_identity",
    "diagnosis",
    "analysis_epoch",
    "evidence_catalog",
    "execution_map",
    "performance_model",
    "diagnostic_evidence",
    "active_evidence_results",
    "requested_claim",
    "ready_capability_ids",
    "contract_action_ids",
    "available_actions",
    "closed_mechanism_keys",
    "candidate_history",
}
EXPECTED_PARTIAL_REASONS = {
    "R01": {
        "aggregate_timing_only",
        "missing_execution_window",
        "missing_node_boundaries",
        "missing_execution_topology",
    },
    "R02": {
        "aggregate_timing_only",
        "missing_execution_window",
        "missing_node_boundaries",
        "missing_execution_topology",
    },
    "R03": {
        "missing_predecision_timing",
        "missing_execution_window",
        "missing_execution_topology",
    },
    "R04": {
        "aggregate_timing_only",
        "missing_execution_window",
        "missing_node_boundaries",
        "missing_execution_topology",
    },
    "R05": {
        "missing_predecision_timing",
        "missing_execution_window",
        "missing_execution_topology",
    },
    "R06": {
        "historical_delta_not_execution_interval",
        "label_timing_excluded",
        "missing_execution_window",
        "missing_execution_topology",
    },
    **{
        f"R{index:02d}": {
            "missing_controller_epoch",
            "missing_knowledge_identity",
            "missing_controller_execution_map",
            "missing_controller_performance_model",
            "label_not_machine_mapped",
        }
        for index in range(7, 13)
    },
}


def nested_keys(value):
    if isinstance(value, dict):
        result = set(value)
        for item in value.values():
            result.update(nested_keys(item))
        return result
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(nested_keys(item))
        return result
    return set()


class KnowledgeReplayTest(unittest.TestCase):
    def setUp(self):
        self.suite = json.loads((FIXTURE_DIR / "decision_points.json").read_text())
        self.baseline = json.loads((FIXTURE_DIR / "v1_2_baseline.json").read_text())

    def test_v1_2_baseline_declares_all_unavailable_triton_replays(self):
        baseline_path = FIXTURE_DIR / "v1_2_baseline.json"
        self.assertTrue(baseline_path.is_file(), "frozen V1.2 baseline fixture is missing")
        baseline = json.loads(baseline_path.read_text())["cases"]
        self.assertEqual(
            set(baseline),
            {f"R{index:02d}" for index in range(1, 13)},
        )

    def test_frozen_inputs_do_not_contain_future_labels(self):
        forbidden = {
            "accepted_mechanism_keys",
            "cheapest_valid_action_ids",
            "expected_terminal_decisions",
            "label_source_sha256",
            "promoted_mechanism_keys",
            "observed_diagnostic_decision",
            "candidate_outcome",
            "source_documents",
            "speedup",
            "verdict",
        }
        for case in self.suite["cases"]:
            self.assertTrue(forbidden.isdisjoint(nested_keys(case["input_snapshot"])))

    def test_frozen_suite_has_required_scoring_groups_and_rejections(self):
        cases = {case["case_id"]: case for case in self.suite["cases"]}
        self.assertEqual(
            {
                key
                for key, case in cases.items()
                if case["scoring_group"] == "triton"
            },
            {f"R{index:02d}" for index in range(1, 13)},
        )
        self.assertEqual({key for key, case in cases.items() if case["scoring_group"] == "public_kernel"}, {"K01", "K02"})
        self.assertTrue(all(cases[key]["label"]["label_status"] == "protocol_only" for key in ("K01", "K02")))
        self.assertEqual({key for key in cases if key.startswith("counterexample-")}, {"counterexample-version-mismatch", "counterexample-missing-evidence", "counterexample-duplicate-mechanism", "counterexample-unstable-benchmark"})
        self.assertEqual(cases["X01"]["scoring_group"], "rejection_only")
        self.assertIn("evidence_summaries", cases["X01"]["input_snapshot"])
        self.assertIn("historical_outcome", cases["X01"]["label"])

    def test_partial_replays_do_not_claim_a_runtime_contract(self):
        forbidden = {"knowledge_identity", "analysis_epoch", "execution_map", "performance_model", "accepted_mechanism_keys", "cheapest_valid_action_ids", "expected_terminal_decisions", "window", "duration_us", "first_start_us", "last_end_us", "coverage", "hot_path", "regime"}
        for case in self.suite["cases"]:
            if case["case_id"].startswith("R"):
                self.assertIn("replay_eligibility", case)
                self.assertEqual(case["replay_eligibility"]["status"], "partial")
                self.assertEqual(case["replay_eligibility"]["timing_provenance"], [])
                self.assertEqual(
                    set(case["replay_eligibility"]["reason_codes"]),
                    EXPECTED_PARTIAL_REASONS[case["case_id"]],
                )
                self.assertTrue(
                    forbidden.isdisjoint(nested_keys(case["input_snapshot"]))
                )
                self.assertIn("archive_identity_facts", case["input_snapshot"])
                self.assertIn("diagnosis", case["input_snapshot"])
                self.assertIn("read_only_actions", case["input_snapshot"])
                self.assertIn("evidence_summaries", case["input_snapshot"])
                self.assertEqual(case["input_snapshot"]["diagnosis"]["authority"], "none")
                self.assertEqual(case["label"]["historical_outcome"]["authority"], "archived_only")

    def test_remote_archive_review_does_not_upgrade_missing_controller_contracts(self):
        cases = {case["case_id"]: case for case in self.suite["cases"]}
        runtime_fields = {
            "knowledge_identity",
            "analysis_epoch",
            "execution_map",
            "performance_model",
        }
        for case_id in {f"R{index:02d}" for index in range(7, 13)}:
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                self.assertEqual(case["replay_eligibility"]["status"], "partial")
                self.assertTrue(
                    runtime_fields.isdisjoint(case["input_snapshot"])
                )
                self.assertNotIn(
                    "accepted_mechanism_keys",
                    case["label"],
                )

    def test_post_audit_cases_use_a_mount_independent_archive_identity(self):
        cases = {case["case_id"]: case for case in self.suite["cases"]}
        for case_id in {f"R{index:02d}" for index in range(7, 13)}:
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    cases[case_id]["input_snapshot"]["archive_identity_facts"][
                        "archive_case_directory"
                    ],
                    "loop30",
                )

    def test_no_triton_case_is_scoreable_without_controller_sealed_artifacts(self):
        self.assertFalse(
            any(
                case["scoring_group"] == "triton"
                and case["replay_eligibility"]["status"] == "scoreable"
                for case in self.suite["cases"]
            )
        )

    def test_unavailable_baseline_has_no_route_claims(self):
        forbidden = {"route_output_sha256", "ranked_card_ids", "action_sequence", "cost", "profiler_required", "terminal_decision", "scoring_denominator"}
        cases = {case["case_id"]: case for case in self.suite["cases"]}
        for case_id, item in self.baseline["cases"].items():
            self.assertIn("status", item)
            self.assertEqual(item["status"], "unavailable")
            self.assertEqual(
                item["reason_codes"],
                cases[case_id]["replay_eligibility"]["reason_codes"],
            )
            self.assertTrue(forbidden.isdisjoint(item))

    def test_v1_2_router_snapshot_tampering_fails_closed(self):
        snapshot = json.loads(
            build_knowledge_replay.V1_2_ROUTER_SNAPSHOT.read_text()
        )
        snapshot["cards"][2]["preferred_actions"][0] = (
            "compiler-sass-inspection"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1_2_router_snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with mock.patch.object(
                build_knowledge_replay,
                "V1_2_ROUTER_SNAPSHOT",
                path,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "snapshot provenance",
                ):
                    build_knowledge_replay._load_v1_2_router_snapshot()

    def test_fixture_and_baseline_digests_are_closed(self):
        self.assertEqual(self.suite["cases_sha256"], build_knowledge_replay.canonical_sha256(self.suite["cases"]))
        self.assertEqual(self.baseline["source_cases_sha256"], self.suite["cases_sha256"])
        self.assertIn("baseline_cases_sha256", self.baseline)
        self.assertEqual(
            self.baseline["baseline_cases_sha256"],
            build_knowledge_replay.canonical_sha256(self.baseline["cases"]),
        )

    @unittest.skipUnless(os.environ.get("CUDA_OPTIMIZER_TRITON_REPLAY_ROOT"), "archive root not supplied")
    def test_optional_archive_regeneration_matches_fixture(self):
        generated = build_knowledge_replay.build_suite(Path(os.environ["CUDA_OPTIMIZER_TRITON_REPLAY_ROOT"]))
        self.assertEqual(build_knowledge_replay.canonical_sha256(generated), build_knowledge_replay.canonical_sha256(self.suite))
        self.assertEqual(
            build_knowledge_replay.build_baseline(generated),
            self.baseline,
        )

    def test_archive_sources_exclude_invalid_uuid_and_fixed_window_throughput(self):
        text = json.dumps(self.suite, ensure_ascii=False).lower()
        self.assertNotIn("iter127", text)
        self.assertNotIn("fixed-window", text)
        self.assertNotIn("fixed_window", text)

    def test_builder_fails_closed_when_required_archive_file_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "required archive evidence is missing"):
                build_knowledge_replay.build_suite(Path(directory))

    def test_archive_summaries_are_relative_and_non_placeholder_sha256(self):
        digest = re.compile(r"[0-9a-f]{64}\Z")
        for case in self.suite["cases"]:
            if "evidence_summaries" not in case["input_snapshot"]:
                continue
            input_summaries = case["input_snapshot"]["evidence_summaries"]
            label_summaries = case["label"]["historical_outcome"]["source_refs"]
            self.assertIn(
                "source_manifest_sha256",
                case["input_snapshot"]["archive_identity_facts"],
            )
            self.assertEqual(
                case["input_snapshot"]["archive_identity_facts"][
                    "source_manifest_sha256"
                ],
                build_knowledge_replay.canonical_sha256(input_summaries),
            )
            self.assertEqual(
                case["label"]["label_source_sha256"],
                build_knowledge_replay.canonical_sha256(label_summaries),
            )
            for summary in [*input_summaries, *label_summaries]:
                self.assertFalse(summary["relative_path"].startswith("/"))
                self.assertRegex(summary["sha256"], digest)
                self.assertNotEqual(summary["sha256"], "0" * 64)
                self.assertEqual(summary["locator"], "whole_file")

    def _scoreable_controller_case(self, root: Path) -> tuple[dict, Path, Path]:
        helper = workload_fixtures.WorkloadRoundTests()
        helper.setUp()
        control, run_dir, _project = helper._workspace(root)
        workload_fixtures._enable_v2_readiness(control, root)
        workload_fixtures._enable_active_diagnosis(control, root)
        helper.controller.start_run(control, run_dir)
        helper._authorize_active_run(control, run_dir)
        hypothesis, request = helper._active_proposal(run_dir)
        helper.controller.register_active_diagnosis_proposal(
            control,
            run_dir,
            hypothesis,
            request,
        )
        source_files = {
            "workload_controller_sha256": (
                Path(__file__).parents[1]
                / "skills/cuda-kernel-optimizer/scripts/workload_controller.py"
            ),
            "evidence_selector_sha256": (
                Path(__file__).parents[1]
                / "skills/cuda-kernel-optimizer/scripts/evidence_selector.py"
            ),
            "diagnostic_knowledge_sha256": (
                Path(__file__).parents[1]
                / "skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py"
            ),
            "diagnostic_cards_sha256": (
                Path(__file__).parents[1]
                / "skills/cuda-kernel-optimizer/references/diagnostic_cards.json"
            ),
            "case_memory_sha256": (
                Path(__file__).parents[1]
                / "skills/cuda-kernel-optimizer/references/case_memory.json"
            ),
        }
        source_repo_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (run_dir / "source-state.json").write_text(
            json.dumps(
                {
                    "source_repo_head": source_repo_head,
                    **{
                        field: hashlib.sha256(path.read_bytes()).hexdigest()
                        for field, path in source_files.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        outcome_path = run_dir / "validation" / "outcome.json"
        outcome_path.parent.mkdir()
        outcome = {
            "accepted_mechanism_keys": ["frameworklaunchoverhead"],
            "bootstrap_95_benefit_ci_us": [1.5, 2.5],
            "cheapest_valid_action_ids": ["pytorch-operator-trace"],
            "correctness_passed": True,
            "mean_benefit_us": 2.0,
            "minimum_mechanism_effect_us": 1.0,
            "pair_count": 6,
            "validation_result": "confirmed_above_mechanism_threshold",
            "wins": 6,
        }
        outcome_path.write_text(
            json.dumps(outcome, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        label_path = run_dir / "validation" / "label.json"
        decision_path = run_dir / "active_diagnosis" / "decision.json"
        label_path.write_text(
            json.dumps(
                {
                    "schema_version": "cuda-optimizer/knowledge-replay-label-v2",
                    "case_id": "T01",
                    "diagnostic_decision_source": {
                        "relative_path": "active_diagnosis/decision.json",
                        "source_sha256": hashlib.sha256(
                            decision_path.read_bytes()
                        ).hexdigest(),
                    },
                    "candidate_validation_source": {
                        "relative_path": "validation/outcome.json",
                        "source_sha256": hashlib.sha256(
                            outcome_path.read_bytes()
                        ).hexdigest(),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        case = build_knowledge_replay.extract_scoreable_controller_case(
            run_dir,
            label_path,
        )
        return case, outcome_path, label_path

    def test_scoreable_controller_run_extracts_closed_inputs_and_v1_2_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            case, _outcome_path, _label_path = self._scoreable_controller_case(
                Path(directory).resolve()
            )

            self.assertEqual(case["case_id"], "T01")
            self.assertEqual(case["scoring_group"], "triton")
            self.assertEqual(case["replay_eligibility"]["status"], "scoreable")
            self.assertTrue(case["replay_eligibility"]["timing_provenance"])
            self.assertEqual(
                case["label"]["promoted_mechanism_keys"],
                ["frameworklaunchoverhead"],
            )
            self.assertEqual(
                case["label"]["observed_diagnostic_decision"],
                case["controller_decision"]["decision"],
            )
            self.assertEqual(
                case["label"]["candidate_outcome"]["status"],
                "promoted",
            )
            self.assertNotIn(
                "promoted_mechanism_keys",
                nested_keys(case["input_snapshot"]),
            )
            build_knowledge_replay.validate_scoreable_case(case)

            suite = {
                "schema_version": "cuda-optimizer/knowledge-replay-v1",
                "cases": [case],
                "cases_sha256": build_knowledge_replay.canonical_sha256([case]),
            }
            baseline = build_knowledge_replay.build_baseline(suite)["cases"]["T01"]
            self.assertTrue(baseline["valid_for_ranking_scoring"])
            self.assertTrue(baseline["valid_for_action_id_scoring"])
            self.assertFalse(baseline["valid_for_measured_cost_scoring"])
            self.assertFalse(baseline["valid_for_terminal_scoring"])
            self.assertEqual(
                baseline["diagnostic_terminal_decision"],
                {
                    "status": "unavailable",
                    "reason": "v1_2_controller_terminal_not_replayed",
                },
            )
            self.assertTrue(baseline["ranked_mechanism_keys"])
            self.assertTrue(baseline["next_actions"])
            self.assertRegex(baseline["route_output_sha256"], r"[0-9a-f]{64}\Z")

    def test_scoreable_case_rejects_unverifiable_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            case, _outcome_path, _label_path = self._scoreable_controller_case(
                Path(directory).resolve()
            )
            case["input_snapshot"]["archive_identity_facts"][
                "controller_source_identity"
            ]["source_repo_head"] = "1" * 40
            with self.assertRaisesRegex(ValueError, "Controller source commit"):
                build_knowledge_replay.validate_scoreable_case(case)

    def test_scoreable_case_rejects_embedded_label_evidence_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            case, _outcome_path, _label_path = self._scoreable_controller_case(
                Path(directory).resolve()
            )
            tampered = copy.deepcopy(case)
            tampered["label"]["source_documents"]["candidate_validation"][
                "raw_json"
            ] = "{}\n"
            with self.assertRaisesRegex(ValueError, "label evidence"):
                build_knowledge_replay.validate_scoreable_case(tampered)

    def test_scoreable_case_rejects_identity_map_or_model_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            case, _outcome_path, _label_path = self._scoreable_controller_case(
                Path(directory).resolve()
            )
            for field in ("knowledge_identity", "analysis_epoch", "execution_map"):
                tampered = copy.deepcopy(case)
                if field == "knowledge_identity":
                    tampered["input_snapshot"][field][
                        "workload_contract_sha256"
                    ] = "f" * 64
                elif field == "analysis_epoch":
                    tampered["input_snapshot"][field]["identities"][
                        "source_sha256"
                    ] = "f" * 64
                else:
                    tampered["input_snapshot"][field]["window"]["end_us"] += 1
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        build_knowledge_replay.validate_scoreable_case(tampered)

            tampered = copy.deepcopy(case)
            tampered["input_snapshot"]["performance_model"][
                "minimum_effect_us"
            ] += 1
            with self.assertRaises(ValueError):
                build_knowledge_replay.validate_scoreable_case(tampered)

            tampered = copy.deepcopy(case)
            tampered["input_snapshot"]["archive_identity_facts"][
                "source_manifest_sha256"
            ] = "f" * 64
            with self.assertRaisesRegex(ValueError, "source manifest"):
                build_knowledge_replay.validate_scoreable_case(tampered)

            tampered = copy.deepcopy(case)
            tampered["input_snapshot"]["archive_identity_facts"][
                "controller_source_identity"
            ]["workload_controller_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "controller source"):
                build_knowledge_replay.validate_scoreable_case(tampered)

            tampered = copy.deepcopy(case)
            tampered["replay_eligibility"]["timing_provenance"][0][
                "locator"
            ] = "/wrong"
            with self.assertRaisesRegex(ValueError, "locator"):
                build_knowledge_replay.validate_scoreable_case(tampered)

    def test_scoreable_extractor_requires_current_controller_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _case, _outcome_path, label_path = self._scoreable_controller_case(root)
            run_dir = root / "run"
            source_state = run_dir / "source-state.json"
            source_state.unlink()
            with self.assertRaisesRegex(ValueError, "source-state"):
                build_knowledge_replay.extract_scoreable_controller_case(
                    run_dir,
                    label_path,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _case, _outcome_path, label_path = self._scoreable_controller_case(root)
            run_dir = root / "run"
            source_state = json.loads(
                (run_dir / "source-state.json").read_text(encoding="utf-8")
            )
            source_state["diagnostic_cards_sha256"] = "f" * 64
            (run_dir / "source-state.json").write_text(
                json.dumps(source_state),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source-state"):
                build_knowledge_replay.extract_scoreable_controller_case(
                    run_dir,
                    label_path,
                )

    def test_scoreable_extractor_rejects_label_source_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _case, outcome_path, label_path = self._scoreable_controller_case(root)
            outcome = json.loads(outcome_path.read_text("utf-8"))
            outcome["mean_benefit_us"] = 99.0
            outcome_path.write_text(json.dumps(outcome), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source digest"):
                build_knowledge_replay.extract_scoreable_controller_case(
                    root / "run",
                    label_path,
                )


class FreshControllerReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads(FRESH_FIXTURE.read_text())
        path = (
            Path(__file__).parents[1]
            / "skills/cuda-kernel-optimizer/scripts/diagnostic_knowledge.py"
        )
        spec = importlib.util.spec_from_file_location(
            "cuda_optimizer_fresh_replay_knowledge",
            path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(path)
        cls.knowledge = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.knowledge
        spec.loader.exec_module(cls.knowledge)

    def _runtime_input(self, case):
        return {
            key: value
            for key, value in case["input_snapshot"].items()
            if key in RUNTIME_INPUT_FIELDS
        }

    def test_fresh_suite_has_six_controller_sealed_package_regressions(self):
        self.assertEqual(len(self.suite["cases"]), 6)
        self.assertEqual(
            self.suite["cases_sha256"],
            build_knowledge_replay.canonical_sha256(self.suite["cases"]),
        )
        for case in self.suite["cases"]:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(
                    case["replay_eligibility"]["status"],
                    "package_regression",
                )
                build_knowledge_replay.validate_package_regression_case(case)
                self.assertTrue(
                    case["input_snapshot"]["active_evidence_results"][0][
                        "observations"
                    ]["semantic_observations"]
                )

    def test_fresh_suite_keeps_labels_out_of_runtime_input(self):
        forbidden = {
            "accepted_mechanism_keys",
            "cheapest_valid_action_ids",
            "expected_terminal_decisions",
            "label_source_sha256",
            "promoted_mechanism_keys",
            "observed_diagnostic_decision",
            "candidate_outcome",
            "source_documents",
        }
        for case in self.suite["cases"]:
            self.assertTrue(
                forbidden.isdisjoint(nested_keys(case["input_snapshot"]))
            )

    def test_v1_3_seed_package_regression_is_not_a_release_gate(self):
        contexts = {
            case["case_id"]: self.knowledge.build_knowledge_context(
                self._runtime_input(case),
                limit=3,
            )
            for case in self.suite["cases"]
        }
        self.assertTrue(
            all(
                case["input_snapshot"]["knowledge_identity"][field][
                    "status"
                ]
                == "verified"
                for case in self.suite["cases"]
                for field in ("gpu_architecture", "cuda_runtime_version")
            )
        )
        self.assertEqual(
            {
                case["controller_decision"]["decision"]
                for case in self.suite["cases"]
            },
            {"PURSUE"},
        )
        self.assertTrue(
            all(
                context["promotion_authority"] == "none"
                and len(
                    json.dumps(
                        context,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
                <= 12 * 1024
                for context in contexts.values()
            )
        )
        for case_id in ("R11-fresh-20260728", "R12-fresh-20260728"):
            reasons = {
                item["reason"] for item in contexts[case_id]["rejections"]
            }
            self.assertIn("exact_case_rejection", reasons)


class PostFreezeControllerReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads(POSTFREEZE_FIXTURE.read_text())
        cls.baseline = json.loads(POSTFREEZE_BASELINE.read_text())
        scripts = (
            Path(__file__).parents[1]
            / "skills/cuda-kernel-optimizer/scripts"
        )

        def load(filename, name):
            path = scripts / filename
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

        cls.knowledge = load(
            "diagnostic_knowledge.py",
            "cuda_optimizer_postfreeze_replay_knowledge",
        )

    @staticmethod
    def _runtime_input(case):
        return {
            key: value
            for key, value in case["input_snapshot"].items()
            if key in RUNTIME_INPUT_FIELDS
        }

    def test_six_retained_cases_are_distinct_controller_replays(self):
        self.assertEqual(len(self.suite["cases"]), 6)
        self.assertEqual(
            self.suite["cases_sha256"],
            build_knowledge_replay.canonical_sha256(self.suite["cases"]),
        )
        manifests = set()
        epochs = set()
        decisions = set()
        for case in self.suite["cases"]:
            with self.subTest(case=case["case_id"]):
                build_knowledge_replay.validate_scoreable_case(case)
                self.assertEqual(
                    case["replay_eligibility"]["status"],
                    "scoreable",
                )
                source = case["input_snapshot"]["archive_identity_facts"][
                    "controller_source_identity"
                ]
                self.assertEqual(
                    source["source_repo_head"],
                    "db5d19c8a03a6f8350294e582dd9f283259262f4",
                )
                manifests.add(
                    case["input_snapshot"]["archive_identity_facts"][
                        "source_manifest_sha256"
                    ]
                )
                epochs.add(case["controller_decision"]["epoch_id"])
                decisions.add(
                    case["label"]["source_documents"][
                        "diagnostic_decision"
                    ]["source_sha256"]
                )
        self.assertEqual(len(manifests), 6)
        self.assertEqual(len(epochs), 6)
        self.assertEqual(len(decisions), 6)

    def test_postfreeze_labels_separate_diagnosis_from_candidate_outcome(self):
        for case in self.suite["cases"]:
            with self.subTest(case=case["case_id"]):
                decision = case["controller_decision"]
                primary = decision["primary_diagnosis"]
                catalog = case["input_snapshot"]["evidence_catalog"]
                support_kinds = {
                    catalog[evidence_id]["kind"]
                    for evidence_id in primary["support_evidence_ids"]
                }
                self.assertEqual(
                    case["label"]["observed_diagnostic_decision"],
                    decision["decision"],
                )
                self.assertEqual(decision["decision"], "PURSUE")
                self.assertEqual(
                    primary["confidence"],
                    "direction_supported",
                )
                self.assertGreaterEqual(len(support_kinds), 2)
                self.assertTrue(decision["benefit_ceiling"]["qualifies"])
                self.assertTrue(
                    decision["investment_brief"]["knowledge_adaptation"][
                        "advisory_only"
                    ]
                )
        outcomes = [
            case["label"]["candidate_outcome"]["status"]
            for case in self.suite["cases"]
        ]
        self.assertEqual(outcomes.count("promoted"), 4)
        self.assertEqual(outcomes.count("rejected"), 2)

    def test_postfreeze_v1_3_direction_and_action_metrics_improve_v1_2(self):
        promoted = [
            case
            for case in self.suite["cases"]
            if case["label"]["promoted_mechanism_keys"]
        ]
        self.assertEqual(len(promoted), 4)
        v1_3_top1 = 0
        v1_3_top3 = 0
        v1_3_valid_actions = 0
        v1_3_profiler_actions = 0
        v1_2_top1 = 0
        v1_2_top3 = 0
        v1_2_valid_actions = 0
        v1_2_profiler_actions = 0
        for case in promoted:
            context = self.knowledge.build_knowledge_context(
                self._runtime_input(case),
                limit=3,
            )
            ranked = [
                item["mechanism_key"] for item in context["candidates"]
            ]
            actions = [
                item["cheapest_falsifier"]["action_id"]
                for item in context["candidates"]
            ]
            expected_mechanisms = set(
                case["label"]["promoted_mechanism_keys"]
            )
            expected_actions = set(
                case["label"]["cheapest_valid_action_ids"]
            )
            v1_3_top1 += bool(ranked[:1] and ranked[0] in expected_mechanisms)
            v1_3_top3 += bool(expected_mechanisms & set(ranked[:3]))
            v1_3_valid_actions += bool(expected_actions & set(actions))
            v1_3_profiler_actions += sum(
                action.startswith("ncu") or action.startswith("nsys")
                for action in actions
            )

            baseline = self.baseline["cases"][case["case_id"]]
            self.assertTrue(baseline["valid_for_ranking_scoring"])
            self.assertTrue(baseline["valid_for_action_id_scoring"])
            self.assertFalse(baseline["valid_for_measured_cost_scoring"])
            self.assertFalse(baseline["valid_for_terminal_scoring"])
            baseline_ranked = baseline["ranked_mechanism_keys"]
            baseline_actions = baseline["next_actions"]
            v1_2_top1 += bool(
                baseline_ranked[:1]
                and baseline_ranked[0] in expected_mechanisms
            )
            v1_2_top3 += bool(
                expected_mechanisms & set(baseline_ranked[:3])
            )
            v1_2_valid_actions += bool(
                expected_actions
                & {item["action_id"] for item in baseline_actions}
            )
            v1_2_profiler_actions += sum(
                item["is_profiler"] for item in baseline_actions
            )

        self.assertEqual(
            (
                v1_3_top1,
                v1_3_top3,
                v1_3_valid_actions,
                v1_3_profiler_actions,
            ),
            (3, 3, 3, 0),
        )
        self.assertEqual(
            (
                v1_2_top1,
                v1_2_top3,
                v1_2_valid_actions,
                v1_2_profiler_actions,
            ),
            (0, 0, 0, 4),
        )

    def test_postfreeze_knowledge_stays_advisory_and_bounded(self):
        for case in self.suite["cases"]:
            with self.subTest(case=case["case_id"]):
                context = self.knowledge.build_knowledge_context(
                    self._runtime_input(case),
                    limit=3,
                )
                self.assertEqual(context["promotion_authority"], "none")
                self.assertTrue(
                    all(
                        item["confidence"] == "inconclusive"
                        and item["promotion_authority"] == "none"
                        for item in context["candidates"]
                    )
                )
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
        by_id = {case["case_id"]: case for case in self.suite["cases"]}
        for case_id in (
            "R11-postfreeze-20260729",
            "R12-postfreeze-20260729",
        ):
            context = self.knowledge.build_knowledge_context(
                self._runtime_input(by_id[case_id]),
                limit=3,
            )
            self.assertFalse(context["candidates"])
            self.assertIn(
                "exact_case_rejection",
                {item["reason"] for item in context["rejections"]},
            )

if __name__ == "__main__":
    unittest.main()
