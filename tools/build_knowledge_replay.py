#!/usr/bin/env python3
"""Freeze archive references without fabricating a runtime replay contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _reference(root: Path, relative_path: str) -> dict:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"required archive evidence is missing: {path}")
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "locator": "whole_file",
    }


def _references(root: Path, paths: tuple[str, ...]) -> list[dict]:
    records = [_reference(root, path) for path in paths]
    hashes = [record["sha256"] for record in records]
    if len(hashes) != len(set(hashes)) or any(value == "0" * 64 for value in hashes):
        raise ValueError("archive evidence has duplicate or placeholder SHA-256")
    return records


def _case(
    case_id: str,
    directory: Path,
    inputs: tuple[str, ...],
    labels: tuple[str, ...],
    reason_codes: list[str],
    diagnosis_text: str,
    outcome_text: str,
    *,
    scoring_group: str = "triton",
) -> dict:
    input_refs = _references(directory, inputs)
    label_refs = _references(directory, labels)
    return {
        "case_id": case_id,
        "scoring_group": scoring_group,
        "replay_eligibility": {
            "status": "partial" if scoring_group == "triton" else "rejection_only",
            "reason_codes": reason_codes,
            "timing_provenance": [],
        },
        "input_snapshot": {
            "archive_identity_facts": {
                "status": "incomplete",
                "archive_case_directory": directory.name,
                "source_manifest_sha256": canonical_sha256(input_refs),
                "unknown_fields": [
                    "knowledge_identity",
                    "analysis_epoch",
                    "execution_map",
                    "performance_model",
                ],
            },
            "diagnosis": {
                "status": "unavailable_for_runtime_replay",
                "authority": "none",
                "source_refs": input_refs,
                "note": diagnosis_text,
            },
            "read_only_actions": [
                {
                    "action_id": f"archive-protocol-{case_id.lower()}",
                    "kind": "archived_protocol_reference",
                    "availability": "historical_only",
                    "authority": "none",
                    "source_refs": input_refs,
                }
            ],
            "evidence_summaries": input_refs,
        },
        "label": {
            "historical_outcome": {
                "authority": "archived_only",
                "source_refs": label_refs,
                "note": outcome_text,
            },
            "label_source_sha256": canonical_sha256(label_refs),
        },
    }


def _extract_r01(root: Path) -> dict:
    return _case(
        "R01",
        root / "iter_156_nms_fp32_output",
        (
            "rewrite_manifest.json",
            "run_candidate_correctness.sh",
            "run_nsys.sh",
            "nsys_fp32_output_1000/mechanism_analysis.json",
        ),
        (
            "correctness_output_invariants.json",
            "correctness_semantic_iter149_vs_candidate.json",
            "correctness.sha256",
            "timing_confirmation_vs_iter149/analysis.json",
            "timing_confirmation_vs_iter149/evidence.sha256",
        ),
        [
            "aggregate_timing_only",
            "missing_execution_window",
            "missing_node_boundaries",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Later archive artifacts remain historical-only.",
    )


def _extract_r02(root: Path) -> dict:
    return _case(
        "R02",
        root / "iter_173_pdl_gap_audit",
        ("analysis.json", "run_nsys_v3.sh", "run_correctness_v3.sh"),
        ("DECISION.md",),
        [
            "aggregate_timing_only",
            "missing_execution_window",
            "missing_node_boundaries",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Decision text remains historical-only.",
    )


def _extract_r03(root: Path) -> dict:
    return _case(
        "R03",
        root / "iter_182_stack_upgrade_iter161",
        ("DESIGN.md", "layer_analysis.json"),
        (
            "correctness_attempt3/analysis.json",
            "exact_attribution_primary/analysis.json",
            "exact_attribution_primary/evidence.sha256",
            "exact_deployment_primary/analysis.json",
            "exact_deployment_primary/evidence.sha256",
            "closure.sha256",
            "DECISION.md",
        ),
        [
            "missing_predecision_timing",
            "missing_execution_window",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Correctness closure remains historical-only.",
    )


def _extract_r04(root: Path) -> dict:
    return _case(
        "R04",
        root / "iter_184_fastsort_map_fused",
        ("DESIGN.md", "run_candidate_correctness.sh", "run_nsys_mechanism_gate.sh"),
        ("analysis.json", "DECISION.md", "closure.sha256"),
        [
            "aggregate_timing_only",
            "missing_execution_window",
            "missing_node_boundaries",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Mechanism-gate closure remains historical-only.",
    )


def _extract_r05(root: Path) -> dict:
    return _case(
        "R05",
        root / "iter_186_iter161_vs_original_endpoint",
        ("DESIGN.md", "run_correctness.sh", "run_endpoint.sh"),
        (
            "results/correctness_decision.json",
            "results/fixed_analysis.json",
            "results/tail_analysis.json",
            "closure.sha256",
            "DECISION.md",
        ),
        [
            "missing_predecision_timing",
            "missing_execution_window",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Endpoint conclusion remains historical-only.",
    )


def _extract_r06(root: Path) -> dict:
    return _case(
        "R06",
        root / "iter_187_stack_upgrade_iter161_adapted",
        (
            "DESIGN.md",
            "run_correctness_screen.sh",
            "run_exact_forced2.sh",
            "run_endpoint_versions.sh",
        ),
        (
            "correctness_screen_attempt2/analysis.json",
            "exact_forced2_primary/analysis.json",
            "exact_forced2_primary/evidence.sha256",
            "endpoint_versions_primary/analysis.json",
            "endpoint_versions_primary/evidence.sha256",
            "closure.sha256",
            "DECISION.md",
        ),
        [
            "historical_delta_not_execution_interval",
            "label_timing_excluded",
            "missing_execution_window",
            "missing_execution_topology",
        ],
        "Archive has no extracted runtime diagnosis.",
        "Migration conclusion remains historical-only.",
    )


def _extract_x01(root: Path) -> dict:
    return _case(
        "X01",
        root / "iter_188_full_ensemble_original_vs_iter161",
        (
            "DESIGN.md",
            "readiness_input.json",
            "run_correctness.sh",
            "run_original_full_ensemble_perf.sh",
        ),
        (
            "correctness_attempt3_three_repeat/analysis.json",
            "correctness_attempt3_three_repeat/closure.sha256",
            "performance_attempt1_original_script/analysis.json",
            "performance_attempt1_original_script/closure.sha256",
            "closure.sha256",
            "RESULTS.md",
        ),
        [
            "correctness_gate_failed",
            "performance_result_not_promotable",
        ],
        "The complete ensemble gate is not available as a scoreable direction replay.",
        "Correctness and performance artifacts are retained only as rejection evidence.",
        scoring_group="rejection_only",
    )


def build_suite(root: Path) -> dict:
    cases = [
        extractor(root)
        for extractor in (
            _extract_r01,
            _extract_r02,
            _extract_r03,
            _extract_r04,
            _extract_r05,
            _extract_r06,
        )
    ]
    for case_id, relative_path, digest in (
        (
            "K01",
            "KernelBench/level1/19_ReLU.py",
            "cbbfc9409662168ee7a5d3e7f7a59bf56e0faf9d763197e7f6a41fb5942dd63a",
        ),
        (
            "K02",
            "KernelBench/level1/92_cumsum_exclusive.py",
            "ec1551c196130f5d7fae707f0750016c35b988e76aa3f1657da4347f463ced86",
        ),
    ):
        cases.append(
            {
                "case_id": case_id,
                "scoring_group": "public_kernel",
                "replay_eligibility": {
                    "status": "protocol_only",
                    "reason_codes": ["candidate_labels_not_frozen"],
                    "timing_provenance": [],
                },
                "input_snapshot": {
                    "repository": "ScalingIntelligence/KernelBench",
                    "commit": "423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
                    "relative_path": relative_path,
                    "source_sha256": digest,
                },
                "label": {"label_status": "protocol_only"},
            }
        )
    counterexamples = {
        "counterexample-version-mismatch": {
            "expected_behavior": "explanation_only",
        },
        "counterexample-missing-evidence": {
            "expected_decisions": ["MEASURE"],
        },
        "counterexample-duplicate-mechanism": {
            "expected_candidate_count": 1,
        },
        "counterexample-unstable-benchmark": {
            "expected_decisions": ["REVIEW_REQUIRED", "STOP"],
        },
    }
    for case_id, expected in counterexamples.items():
        cases.append(
            {
                "case_id": case_id,
                "scoring_group": "rejection_only",
                "replay_eligibility": {
                    "status": "rejection_only",
                    "reason_codes": [case_id.removeprefix("counterexample-")],
                    "timing_provenance": [],
                },
                "input_snapshot": {"counterexample": case_id},
                "label": expected,
            }
        )
    cases.append(_extract_x01(root))
    return {
        "schema_version": "cuda-optimizer/knowledge-replay-v1",
        "cases": cases,
        "cases_sha256": canonical_sha256(cases),
    }


def build_baseline(suite: dict) -> dict:
    cases = {
        case["case_id"]: {
            "status": "unavailable",
            "reason_codes": case["replay_eligibility"]["reason_codes"],
        }
        for case in suite["cases"]
        if case["scoring_group"] == "triton"
    }
    return {
        "schema_version": "cuda-optimizer/knowledge-replay-baseline-v1",
        "source_cases_sha256": suite["cases_sha256"],
        "baseline_cases_sha256": canonical_sha256(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-output", required=True, type=Path)
    args = parser.parse_args()
    suite = build_suite(args.archive_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    args.baseline_output.write_text(json.dumps(build_baseline(suite), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
