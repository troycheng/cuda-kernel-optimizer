"""Rerun one controlled scenario for Controller evidence-admission testing."""

from __future__ import annotations

import json
import os
from pathlib import Path

from diagnostic_scenarios import run_scenario


workspace = Path(__file__).resolve().parent
config = json.loads((workspace / "scenario-config.json").read_text("utf-8"))
request = json.loads(Path(os.environ["CUDA_OPTIMIZER_EVIDENCE_REQUEST"]).read_text("utf-8"))
measured = run_scenario(config["scenario"])
observation = measured["observation"]
baseline = float(observation["baseline_median_us"])
comparison = float(observation["comparison_median_us"])
expected = observation["expected_comparison"]
relation_holds = comparison < baseline if expected == "faster" else comparison > baseline
supported = bool(
    observation["correctness_passed"]
    and relation_holds
    and abs(baseline - comparison) >= 1.0
)
artifact = Path(os.environ["CUDA_OPTIMIZER_EVIDENCE_DIR"]) / "controlled-observation.json"
artifact.write_text(
    json.dumps(measured, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
Path(os.environ["CUDA_OPTIMIZER_EVIDENCE_OUTPUT"]).write_text(
    json.dumps(
        {
            "schema_version": "cuda-optimizer/evidence-result-v1",
            "request_signature": request["request_signature"],
            "status": "observed",
            "outcome_id": (
                "mechanism-supported" if supported else "alternative-supported"
            ),
            "observations": {
                "scenario": observation["name"],
                "baseline_median_us": baseline,
                "comparison_median_us": comparison,
                "effect_us": abs(baseline - comparison),
                "expected_comparison": expected,
                "correctness_passed": observation["correctness_passed"],
            },
            "artifacts": [{"path": str(artifact)}],
        }
    ),
    encoding="utf-8",
)
