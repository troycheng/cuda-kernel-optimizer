from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "skills"
    / "cuda-kernel-optimizer"
    / "scripts"
    / "diagnostic_evidence.py"
)
KNOWLEDGE_PATH = MODULE_PATH.with_name("diagnostic_knowledge.py")
CONTRACT_SHA = "a" * 64
ENVIRONMENT_SHA = "b" * 64
TARGET_SHA = "2" * 64
IMPLEMENTATION_SHA = "3" * 64
REQUEST_SHA = "4" * 64


def _load():
    spec = importlib.util.spec_from_file_location("cuda_diagnostic_evidence_v3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_knowledge():
    spec = importlib.util.spec_from_file_location(
        "cuda_diagnostic_knowledge_v13", KNOWLEDGE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _measurement(signals=None):
    return {
        "schema_version": "cuda-optimizer/diagnostic-measurement-v1",
        "subject": {"target_sha256": TARGET_SHA},
        "report": {"artifact_sha256": "5" * 64, "events_total": 12},
        "signals": list(signals or []),
        "checks": [{"name": "report_parse_complete", "passed": True}],
    }


class DiagnosticEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load()

    def test_controller_derives_and_revalidates_nsys_evidence(self) -> None:
        raw = (json.dumps(_measurement(["launch_gap_short_context"])) + "\n").encode()
        evidence = self.module.derive_diagnostic_evidence(
            raw,
            kind="nsys_timeline",
            producer_id="nsys-timeline-adapter",
            producer_version="1.0.0",
            implementation_sha256=IMPLEMENTATION_SHA,
            adapter_request_sha256=REQUEST_SHA,
            contract_sha256=CONTRACT_SHA,
            environment_sha256=ENVIRONMENT_SHA,
            recorded_at=100.0,
        )
        result = self.module.validate_diagnostic_evidence(
            evidence,
            expected_contract_sha256=CONTRACT_SHA,
            expected_environment_sha256=ENVIRONMENT_SHA,
        )

        self.assertEqual(result["kind"], "nsys_timeline")
        self.assertEqual(result["signals"], ["launch_gap_short_context"])
        self.assertEqual(result["producer"]["implementation_sha256"], IMPLEMENTATION_SHA)
        self.assertTrue(evidence.endswith(b"\n"))
        self.assertEqual(json.loads(evidence)["adapter_request_sha256"], REQUEST_SHA)

    def test_validated_pytorch_evidence_normalizes_to_framework_observation(self) -> None:
        raw = (
            json.dumps(_measurement(["framework_dispatch_overhead"])) + "\n"
        ).encode()
        evidence = self.module.derive_diagnostic_evidence(
            raw,
            kind="pytorch_profile",
            producer_id="pytorch-profile-adapter",
            producer_version="1.0.0",
            implementation_sha256=IMPLEMENTATION_SHA,
            adapter_request_sha256=REQUEST_SHA,
            contract_sha256=CONTRACT_SHA,
            environment_sha256=ENVIRONMENT_SHA,
            recorded_at=100.0,
        )
        validated = self.module.validate_diagnostic_evidence(
            evidence,
            expected_contract_sha256=CONTRACT_SHA,
            expected_environment_sha256=ENVIRONMENT_SHA,
        )

        observations = _load_knowledge().normalize_observations(
            diagnostic_evidence=[validated]
        )

        self.assertEqual(
            [(item["semantic_id"], item["status"]) for item in observations],
            [("framework.dispatch_overhead", "present")],
        )

    def test_kind_signal_vocabulary_and_raw_metadata_are_closed(self) -> None:
        invalid = [
            _measurement(["kv_gather_dram"]),
            _measurement(["launch_gap_short_context"]) | {"status": "PASS"},
        ]
        for index, measurement in enumerate(invalid):
            with self.subTest(index=index), self.assertRaisesRegex(
                ValueError, "signal|closed|measurement"
            ):
                self.module.derive_diagnostic_evidence(
                    (json.dumps(measurement) + "\n").encode(),
                    kind="nsys_timeline",
                    producer_id="nsys-timeline-adapter",
                    producer_version="1.0.0",
                    implementation_sha256=IMPLEMENTATION_SHA,
                    adapter_request_sha256=REQUEST_SHA,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    recorded_at=100.0,
                )

    def test_failed_check_and_duplicate_signal_fail_closed(self) -> None:
        failed = _measurement(["launch_gap_short_context"])
        failed["checks"][0]["passed"] = False
        duplicate = _measurement(
            ["launch_gap_short_context", "launch_gap_short_context"]
        )
        for measurement in (failed, duplicate):
            with self.assertRaisesRegex(ValueError, "check|duplicate|signal|PASS"):
                self.module.derive_diagnostic_evidence(
                    json.dumps(measurement).encode(),
                    kind="nsys_timeline",
                    producer_id="nsys-timeline-adapter",
                    producer_version="1.0.0",
                    implementation_sha256=IMPLEMENTATION_SHA,
                    adapter_request_sha256=REQUEST_SHA,
                    contract_sha256=CONTRACT_SHA,
                    environment_sha256=ENVIRONMENT_SHA,
                    recorded_at=100.0,
                )

    def test_supported_ncu_metrics_normalize_to_closed_validated_semantics(self) -> None:
        result = self.module.normalize_ncu_metrics(
            {
                "dram__throughput.avg.pct_of_peak_sustained_elapsed": (81.0, "%"),
                "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": (
                    37.0,
                    "%",
                ),
                "unknown__metric": (9.0, "%"),
            },
            tool_version="2026.2",
        )

        self.assertEqual(
            set(result),
            {"semantic_observations", "unmodeled_metrics", "mapping_version"},
        )
        self.assertEqual(result["mapping_version"], "ncu-semantic-v1")
        self.assertEqual(
            [item["semantic_id"] for item in result["semantic_observations"]],
            ["kernel.dram_throughput_pct", "kernel.long_scoreboard_pct"],
        )
        self.assertTrue(
            all(
                set(item)
                == {
                    "semantic_id",
                    "status",
                    "value",
                    "unit",
                    "scope",
                    "aggregation",
                    "tool",
                    "quality",
                }
                for item in result["semantic_observations"]
            )
        )
        self.assertTrue(
            all(
                item["status"] == "observed"
                and item["quality"] == "validated"
                and item["tool"] == {"name": "ncu", "version": "2026.2"}
                for item in result["semantic_observations"]
            )
        )
        self.assertEqual(
            result["unmodeled_metrics"],
            [{"metric_name": "unknown__metric", "reason": "unknown_metric"}],
        )

    def test_unknown_ncu_major_minor_is_unmodeled_without_semantics(self) -> None:
        result = self.module.normalize_ncu_metrics(
            {
                "dram__throughput.avg.pct_of_peak_sustained_elapsed": (81.0, "%"),
            },
            tool_version="2026.3",
        )

        self.assertEqual(result["semantic_observations"], [])
        self.assertEqual(
            result["unmodeled_metrics"],
            [
                {
                    "metric_name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "reason": "unsupported_tool_version",
                }
            ],
        )

    def test_invalid_ncu_values_units_and_duplicate_semantics_fail_closed(self) -> None:
        cases = [
            (
                {
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": (
                        float("nan"),
                        "%",
                    )
                },
                "non_finite_value",
            ),
            (
                {
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": (
                        81.0,
                        "",
                    )
                },
                "missing_unit",
            ),
            (
                {
                    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": (
                        37.0,
                        "%",
                    ),
                    "smsp__average_warp_latency_issue_stalled_long_scoreboard_per_warp_active.pct": (
                        41.0,
                        "%",
                    ),
                },
                "conflicting_semantic_values",
            ),
        ]

        for metrics, reason in cases:
            with self.subTest(reason=reason):
                result = self.module.normalize_ncu_metrics(
                    metrics, tool_version="2026.2"
                )
                self.assertEqual(result["semantic_observations"], [])
                self.assertTrue(result["unmodeled_metrics"])
                self.assertEqual(
                    {item["reason"] for item in result["unmodeled_metrics"]},
                    {reason},
                )


if __name__ == "__main__":
    unittest.main()
