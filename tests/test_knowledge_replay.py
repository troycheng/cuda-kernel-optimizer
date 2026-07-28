import json
import os
import tempfile
import unittest
import re
from pathlib import Path

from tools import build_knowledge_replay


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge_replay"
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
            {"R01", "R02", "R03", "R04", "R05", "R06"},
        )

    def test_frozen_inputs_do_not_contain_future_labels(self):
        forbidden = {
            "accepted_mechanism_keys",
            "cheapest_valid_action_ids",
            "expected_terminal_decisions",
            "label_source_sha256",
            "speedup",
            "verdict",
        }
        for case in self.suite["cases"]:
            self.assertTrue(forbidden.isdisjoint(nested_keys(case["input_snapshot"])))

    def test_frozen_suite_has_required_scoring_groups_and_rejections(self):
        cases = {case["case_id"]: case for case in self.suite["cases"]}
        self.assertEqual({key for key, case in cases.items() if case["scoring_group"] == "triton"}, {"R01", "R02", "R03", "R04", "R05", "R06"})
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


if __name__ == "__main__":
    unittest.main()
