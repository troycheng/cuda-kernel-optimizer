#!/usr/bin/env python3
"""Single-method ablation for attribution.

For each method applied in the champion kernel, this script expects the agent
to have generated an ablated kernel (champion minus that one method) under
  iterv{i}/ablations/{method_id}/kernel.<ext>

This script benchmarks each ablated kernel and computes attribution:
  attribution(m) = ms_ablated(m) - ms_champion

Positive → the method helped (removing it slowed things down).
Zero/Negative → the method did not help or actually hurt.

Writes iterv{i}/attribution.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import artifact_store  # noqa: E402
from paired_benchmark import run_paired as _run_paired  # noqa: E402
from paired_stats import classify_pairs  # noqa: E402


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
_METHOD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _regular_identity(path: str | os.PathLike) -> dict:
    """Capture one no-follow file descriptor identity and its exact bytes."""
    directory_fd = None
    file_fd = None
    try:
        directory_fd, leaf, target = artifact_store._open_parent_directory(
            path, create=False
        )
        file_fd = os.open(
            leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd
        )
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"ablation evidence is not a regular file: {target}")
        current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError(f"ablation evidence changed while opening: {target}")
        chunks = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        final = os.fstat(file_fd)
        current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)
        if identity != (final.st_dev, final.st_ino, final.st_size) or (
            current.st_dev, current.st_ino, current.st_size
        ) != identity:
            raise ValueError(f"ablation evidence changed while reading: {target}")
        if len(payload) != metadata.st_size:
            raise ValueError(f"ablation evidence size changed while reading: {target}")
        return {
            "path": str(target),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": payload,
        }
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _bench_kernel(
    benchmark_py: str,
    kernel_path: str,
    ref_path: str,
    dims: dict,
    ptr_size: int,
    json_out: str,
    warmup: int = 5,
    repeat: int = 15,
) -> dict | None:
    """Run benchmark.py on a single kernel and return the parsed JSON result."""
    cmd = [
        sys.executable, benchmark_py, kernel_path,
        "--ref", ref_path,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", json_out,
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    output = Path(json_out)
    if output.is_symlink():
        output.unlink()
    elif output.exists():
        if not output.is_file():
            print(f"[ablate] benchmark output is not a regular file: {output}", file=sys.stderr)
            return None
        output.unlink()

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )
    except OSError as e:
        print(f"[ablate] benchmark failed: {e}", file=sys.stderr)
        return None

    if r.returncode != 0:
        return None
    if os.path.isfile(json_out):
        return _load_json(json_out)
    return None


def run(state_path: str, iteration: int, benchmark_py: str = None) -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    bench_py = benchmark_py or _BUNDLED_BENCHMARK

    # The published candidate bench is only a correctness precondition.  Method
    # contribution is measured later from same-run AB/BA pairs.
    champion_bench = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(champion_bench):
        sys.exit(f"Champion bench.json not found at {champion_bench}")
    try:
        champion_identity = _regular_identity(champion_bench)
        champion_bench = champion_identity["path"]
        champion_data = json.loads(champion_identity["bytes"].decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        sys.exit(f"Champion bench.json is unsafe or malformed: {error}")
    if champion_data.get("correctness", {}).get("passed") is not True:
        sys.exit("Selected candidate failed correctness validation")
    candidate_paths = [
        Path(iter_dir) / f"kernel{extension}" for extension in (".cu", ".py")
    ]
    candidate_identities = []
    for candidate_path in candidate_paths:
        try:
            candidate_identities.append(_regular_identity(candidate_path))
        except (OSError, ValueError):
            continue
    if len(candidate_identities) != 1:
        sys.exit("Iteration must contain exactly one selected candidate kernel")
    candidate_identity = candidate_identities[0]

    # Load methods
    methods_path = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_path):
        sys.exit(f"methods.json not found at {methods_path}")
    methods_data = _load_json(methods_path)
    methods_list = list(methods_data.get("methods", []))
    inherited_methods = methods_data.get("inherited_methods", [])
    if not isinstance(inherited_methods, list) or any(
        not isinstance(method_id, str) or not method_id.strip()
        for method_id in inherited_methods
    ):
        sys.exit("inherited_methods must be a string array")
    all_method_ids = [
        method.get("id") for method in methods_list if isinstance(method, dict)
    ] + inherited_methods
    if any(
        not isinstance(method_id, str) or _METHOD_ID.fullmatch(method_id) is None
        for method_id in all_method_ids
    ):
        sys.exit("method ids must be safe relative identifiers")
    seen_method_ids = {
        method.get("id") for method in methods_list if isinstance(method, dict)
    }
    methods_list.extend(
        {"id": method_id}
        for method_id in inherited_methods
        if method_id not in seen_method_ids
    )

    ref_file = state["ref_file"]
    dims = state.get("dims", {})
    ptr_size = state.get("ptr_size", 0)
    noise_threshold = state.get("noise_threshold_pct", 2.0)
    minimum_effect_us = state.get("minimum_effect_us", 1.0)
    if (
        isinstance(minimum_effect_us, bool)
        or not isinstance(minimum_effect_us, (int, float))
        or not math.isfinite(float(minimum_effect_us))
        or float(minimum_effect_us) <= 0.0
    ):
        raise ValueError("state minimum_effect_us must be a positive finite number")
    minimum_effect_us = float(minimum_effect_us)
    budget = state.get("budget") or {}
    blocks = budget.get("min_pairs", 20)
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2:
        raise ValueError("state budget.min_pairs must be an integer of at least 2")
    backend = state.get("backend", "auto")
    env = state.get("env") or {}
    arch = state.get("arch") or env.get("primary_sm_arch")
    if not arch:
        gpus = env.get("gpus") or []
        if gpus and isinstance(gpus[0], dict):
            arch = gpus[0].get("sm_arch")
    if not isinstance(arch, str) or not arch.strip():
        raise ValueError("state must provide arch or env.primary_sm_arch")
    nvcc = env.get("nvcc") or {}
    nvcc_bin = state.get("nvcc_bin") or nvcc.get("path") or "nvcc"
    if not isinstance(nvcc_bin, str) or not nvcc_bin.strip():
        raise ValueError("state nvcc_bin must be a non-empty string")
    seed = state.get("seed", 0)
    confidence = state.get("confidence", 0.95)
    bootstrap_samples = state.get("bootstrap_samples", 10000)
    max_temperature_delta_c = state.get("max_temperature_delta_c", 5)
    max_clock_delta_pct = state.get("max_clock_delta_pct", 5)
    input_hash = state.get("input_hash")
    if not isinstance(input_hash, str) or not input_hash.strip():
        raise ValueError("state input_hash must be non-empty")

    attributions = []
    ablation_dir = os.path.join(iter_dir, "ablations")
    paired_champion_ms = None

    for m in methods_list:
        mid = m.get("id", "unknown")
        # Safe method ids contain no path separators, so keeping the exact id is
        # both path-safe and one-to-one (unlike replacing dots with underscores).
        method_dir = os.path.join(ablation_dir, mid)

        # Find ablated kernel
        ablated_identities = []
        for ext in (".cu", ".py"):
            candidate = os.path.join(method_dir, f"kernel{ext}")
            try:
                identity = _regular_identity(candidate)
            except (OSError, ValueError):
                continue
            ablated_identities.append(identity)

        if not ablated_identities:
            # No ablated kernel provided — skip, assume neutral
            attributions.append({
                "method_id": mid,
                "ablated_kernel": None,
                "ablated_ms": None,
                "champion_ms": None,
                "attribution_ms": 0.0,
                "attribution_pct": 0.0,
                "contributed": False,
                "note": "no_ablated_kernel_provided",
            })
            continue
        if len(ablated_identities) != 1:
            sys.exit(f"Method {mid} must contain exactly one ablated kernel")
        kernel_before = ablated_identities[0]
        ablated_kernel = kernel_before["path"]

        # Benchmark ablated kernel
        ablated_json_out = os.path.join(method_dir, "bench.json")
        result = _bench_kernel(
            bench_py,
            ablated_kernel,
            ref_file,
            dims,
            ptr_size,
            ablated_json_out,
            1,
            1,
        )

        if result is None:
            attributions.append({
                "method_id": mid,
                "contributed": False,
                "note": "ablation_benchmark_failed",
            })
            continue

        if not result.get("correctness", {}).get("passed", False):
            try:
                kernel_after = _regular_identity(ablated_kernel)
                bench_identity = _regular_identity(ablated_json_out)
                if (
                    kernel_after != kernel_before
                    or json.loads(bench_identity["bytes"].decode("utf-8")) != result
                ):
                    raise ValueError("correctness ablation evidence changed")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                attributions.append({
                    "method_id": mid,
                    "contributed": False,
                    "note": f"unsafe_or_drifted_ablation_evidence: {error}",
                })
                continue
            attributions.append({
                "method_id": mid,
                "ablated_kernel": kernel_before["path"],
                "ablated_kernel_sha256": kernel_before["sha256"],
                "ablated_bench": bench_identity["path"],
                "ablated_bench_sha256": bench_identity["sha256"],
                "ablated_ms": None,
                "champion_ms": None,
                "attribution_ms": None,
                "attribution_pct": None,
                "contributed": False,
                "note": "ablated_kernel_failed_correctness",
            })
            continue

        try:
            stable_champion = _regular_identity(champion_bench)
            kernel_after = _regular_identity(ablated_kernel)
            bench_identity = _regular_identity(ablated_json_out)
            if (
                stable_champion != champion_identity
                or kernel_after != kernel_before
                or json.loads(bench_identity["bytes"].decode("utf-8")) != result
            ):
                raise ValueError("ablation evidence changed during collection")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            attributions.append({
                "method_id": mid,
                "contributed": False,
                "note": f"unsafe_or_drifted_ablation_evidence: {error}",
            })
            continue

        try:
            paired = _run_paired(
                candidate_identity["path"],
                ablated_kernel,
                backend=backend,
                dims=dims,
                ptr_size=ptr_size,
                arch=arch,
                nvcc_bin=nvcc_bin,
                seed=seed,
                blocks=blocks,
                warmup=1,
                max_temperature_delta_c=max_temperature_delta_c,
                max_clock_delta_pct=max_clock_delta_pct,
            )
            if _regular_identity(candidate_identity["path"]) != candidate_identity:
                raise ValueError("selected candidate changed during paired attribution")
            if _regular_identity(ablated_kernel) != kernel_before:
                raise ValueError("ablated kernel changed during paired attribution")
            raw_pairs = paired["pairs"]
            valid_pairs = [
                pair for pair in raw_pairs if pair.get("valid", True) is True
            ]
            if not valid_pairs:
                raise ValueError("paired attribution produced no valid pairs")
            champion_ms = statistics.median(
                float(pair["baseline"]) for pair in valid_pairs
            )
            ablated_ms = statistics.median(
                float(pair["candidate"]) for pair in valid_pairs
            )
            attr_ms = statistics.median(
                float(pair["candidate"]) - float(pair["baseline"])
                for pair in valid_pairs
            )
            effective_min_effect_pct = max(
                float(noise_threshold),
                minimum_effect_us / (champion_ms * 10.0),
            )
            statistics_payload = classify_pairs(
                raw_pairs,
                direction="higher",
                min_effect_pct=effective_min_effect_pct,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            paired_samples = artifact_store.write_paired_samples(
                os.path.join(method_dir, "paired_samples.jsonl"),
                raw_pairs,
                kind="kernel",
                input_hash=input_hash,
                iteration=iteration,
                candidate_id=mid,
                candidate_file=ablated_kernel,
                baseline_file=candidate_identity["path"],
                classifier_config={
                    "direction": "higher",
                    "min_effect_pct": effective_min_effect_pct,
                    "confidence": confidence,
                    "bootstrap_samples": bootstrap_samples,
                    "seed": seed,
                },
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            attributions.append({
                "method_id": mid,
                "contributed": False,
                "note": f"paired_attribution_failed: {type(error).__name__}",
            })
            continue

        if paired_champion_ms is None:
            paired_champion_ms = champion_ms
        attr_pct = statistics_payload["estimate_pct"]
        contributed = statistics_payload["status"] == "confirmed_win"

        attributions.append({
            "method_id": mid,
            "evidence_kind": "performance_attribution",
            "champion_bench": champion_bench,
            "champion_bench_sha256": champion_identity["sha256"],
            "ablated_kernel": kernel_before["path"],
            "ablated_kernel_sha256": kernel_before["sha256"],
            "ablated_bench": bench_identity["path"],
            "ablated_bench_sha256": bench_identity["sha256"],
            "ablated_ms": ablated_ms,
            "champion_ms": champion_ms,
            "attribution_ms": attr_ms,
            "attribution_pct": attr_pct,
            "contributed": contributed,
            "minimum_effect_us": minimum_effect_us,
            "effective_min_effect_pct": effective_min_effect_pct,
            "statistics": statistics_payload,
            "paired_samples": paired_samples,
        })

    try:
        if _regular_identity(champion_bench) != champion_identity:
            raise ValueError("champion bench identity drifted before attribution write")
        if _regular_identity(candidate_identity["path"]) != candidate_identity:
            raise ValueError("selected candidate changed before attribution write")
    except (OSError, ValueError) as error:
        sys.exit(f"Candidate attribution inputs changed before write: {error}")

    output = {
        "iter": iteration,
        "champion_ms": paired_champion_ms,
        "candidate_file": candidate_identity["path"],
        "candidate_sha256": candidate_identity["sha256"],
        "noise_threshold_pct": noise_threshold,
        "minimum_effect_us": minimum_effect_us,
        "attributions": attributions,
    }

    out_path = os.path.join(iter_dir, "attribution.json")
    artifact_store.atomic_write_json(out_path, output)

    print(json.dumps(output, indent=2))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--benchmark", default=None)
    args = p.parse_args()
    run(args.state, args.iter, args.benchmark)


if __name__ == "__main__":
    main()
