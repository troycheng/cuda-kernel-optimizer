#!/usr/bin/env python3
"""Real Triton command driver for the opt-in V1.4 SM120 acceptance lane."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path


def _load_kernel(locator: str):
    path = Path(locator) / "kernel.py"
    spec = importlib.util.spec_from_file_location(
        f"v14_sm120_kernel_{path.stat().st_ino}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Triton variant")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment(torch, triton) -> dict:
    fields = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).splitlines()[0].split(",")
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu_uuids": [fields[0].strip()],
        "gpu_models": [properties.name],
        "gpu_architectures": [f"sm_{properties.major}{properties.minor}"],
        "driver_version": fields[1].strip(),
        "cuda_runtime_version": str(torch.version.cuda),
        "frameworks": {
            "pytorch": str(torch.__version__),
            "triton": str(triton.__version__),
        },
        "container": {
            "kind": "docker-image",
            "identity": os.environ["CUDA_E2E_IMAGE_ID"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    import torch
    import triton

    module = _load_kernel(request["variant"]["locator"])
    suite = json.loads(
        Path(request["test_suite"]["locator"]).read_text(encoding="utf-8")
    )
    case = next(
        item for item in suite["cases"] if item["id"] == request["case"]["id"]
    )
    reference = json.loads(
        Path(request["correctness"]["reference"]["locator"]).read_text(
            encoding="utf-8"
        )
    )
    if reference.get("expression") != "x*x+1":
        raise ValueError("unsupported correctness reference")
    tolerance = float(reference["atol"])
    state = module.setup(N=int(case["size"]), seed=int(case["seed"]))
    inputs = state["inputs"]
    module.run_kernel(**inputs)
    torch.cuda.synchronize()

    result = {
        "protocol_version": "cuda-kernel-optimizer/driver-result-v1",
        "request_digest": request["request_digest"],
        "target_id": request["target_id"],
        "execution_id": request["execution_id"],
        "variant_digest": request["variant"]["digest"],
        "role": request["role"],
        "mode": request["mode"],
        "case_id": request["case"].get("id"),
        "artifacts": [],
        "cleanup": {"status": "confirmed", "live_tasks": []},
        "driver_identity": request["driver_identity"],
        "environment": _environment(torch, triton),
    }
    if request["mode"] in {"correctness", "combined"}:
        expected = inputs["x"] * inputs["x"] + 1.0
        maximum_error = float((inputs["out"] - expected).abs().max().item())
        result["correctness"] = {
            "status": "passed" if maximum_error <= tolerance else "failed",
            "metrics": {
                "exact_match": 1.0 if maximum_error <= tolerance else 0.0
            },
        }
    if request["mode"] in {"measure", "combined"}:
        for _ in range(5):
            module.run_kernel(**inputs)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                module.run_kernel(**inputs)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) / 50.0)
        result["measurements"] = {
            "primary": {
                "name": "latency_ms",
                "unit": "ms",
                "samples": samples,
            },
            "constraints": [],
        }

    output = Path(request["output_path"])
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
