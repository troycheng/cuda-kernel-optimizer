"""Run one controlled GPU scenario and emit a Controller map fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path

from diagnostic_scenarios import run_scenario


workspace = Path(__file__).resolve().parent
config = json.loads((workspace / "scenario-config.json").read_text("utf-8"))
measured = run_scenario(config["scenario"])
observation = measured["observation"]
window_us = max(
    float(observation["baseline_median_us"]),
    float(observation["comparison_median_us"]),
    1.0,
) * 1.25
primary_node = "gpu-kernel" if observation["claim_layer"] == "kernel" else "cpu-launch"
minor_duration = min(max(1.0, window_us * 0.05), window_us)

raw_path = Path(os.environ["CUDA_OPTIMIZER_RUN_DIR"]) / "active_diagnosis" / "global-scan-observation.json"
raw_path.parent.mkdir(parents=True, exist_ok=True)
raw_path.write_text(
    json.dumps(measured, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)

Path(os.environ["CUDA_OPTIMIZER_OUTPUT"]).write_text(
    json.dumps(
        {
            "schema_version": "cuda-workload-optimizer/probe-v1",
            "probe_id": "timeline",
            "kind": "timeline",
            "status": "ok",
            "metrics": {
                "gpu_busy_pct": 100.0,
                "cpu_busy_pct": 0.0,
                "data_wait_pct": 0.0,
            },
            "issues": [],
            "artifacts": [],
        }
    ),
    encoding="utf-8",
)

coverage = [
    {
        "layer": layer,
        "status": "observed" if layer in {"cpu", "gpu"} else "not_observed",
        "reason": None if layer in {"cpu", "gpu"} else "not present in controlled window",
    }
    for layer in (
        "cpu",
        "gpu",
        "framework",
        "transfer",
        "communication",
        "io",
        "synchronization",
        "idle",
    )
]
nodes = []
for node_id, layer, lane, kind, label in (
    ("cpu-launch", "cpu", "thread-0", "cuda_api", "controlled_host_path"),
    ("gpu-kernel", "gpu", "stream-0", "kernel", "controlled_gpu_path"),
):
    duration = (
        min(float(observation["comparison_median_us"]), window_us)
        if node_id == primary_node
        else minor_duration
    )
    nodes.append(
        {
            "node_id": node_id,
            "layer": layer,
            "lane": lane,
            "kind": kind,
            "label": label,
            "duration_us": duration,
            "occurrences": len(observation["samples_us"]),
            "timing_status": "observed",
            "first_start_us": 0.0,
            "last_end_us": duration,
            "attribution_status": "explained" if layer == "cpu" else "not_applicable",
            "evidence_ids": ["ev-global-scan"],
        }
    )

Path(os.environ["CUDA_OPTIMIZER_ACTIVE_DIAGNOSIS_OUTPUT"]).write_text(
    json.dumps(
        {
            "schema_version": "cuda-optimizer/global-scan-draft-v1",
            "regime": {
                "shape_distribution_sha256": "1" * 64,
                "dynamic_branch_sha256": "2" * 64,
                "execution_regime_sha256": "3" * 64,
            },
            "boundary_ambiguous": False,
            "window": {"start_us": 0.0, "end_us": window_us},
            "coverage": coverage,
            "nodes": nodes,
            "edges": [
                {
                    "source": "cpu-launch",
                    "target": "gpu-kernel",
                    "relation": "calls",
                    "overlap_us": None,
                    "evidence_ids": ["ev-global-scan"],
                }
            ],
            "hot_path": ["cpu-launch", "gpu-kernel"],
            "uncovered_intervals": [],
            "conclusion_level": "observed",
        }
    ),
    encoding="utf-8",
)
