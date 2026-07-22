#!/usr/bin/env python3
"""Controlled real-GPU observations for the V1.1 diagnosis lane."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def _event_samples(function, *, warmup: int = 3, repeat: int = 7) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1_000.0)
    return values


def _wall_samples(function, *, warmup: int = 3, repeat: int = 7) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1_000.0)
    return values


def _observation(
    name: str,
    mechanism: str,
    claim_layer: str,
    baseline: list[float],
    comparison: list[float],
    *,
    correctness_passed: bool,
    **details,
) -> dict:
    return {
        "name": name,
        "mechanism": mechanism,
        "claim_layer": claim_layer,
        "baseline_median_us": statistics.median(baseline),
        "comparison_median_us": statistics.median(comparison),
        "samples_us": comparison,
        "correctness_passed": correctness_passed,
        **details,
    }


def launch_graph() -> dict:
    tensor = torch.zeros(1 << 18, device="cuda")
    operations = 32

    def eager():
        tensor.zero_()
        for _ in range(operations):
            tensor.add_(1.0)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        eager()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eager()

    eager_samples = _wall_samples(eager)
    graph_samples = _wall_samples(graph.replay)
    graph.replay()
    torch.cuda.synchronize()
    correct = bool(torch.all(tensor == float(operations)).item())
    return _observation(
        "launch_graph",
        "cuda_graph_launch_batching",
        "runtime",
        eager_samples,
        graph_samples,
        correctness_passed=correct,
        launches_per_replay=operations + 1,
    )


def memory_coalescing() -> dict:
    size = 8 * 1024 * 1024
    source = torch.linspace(0.0, 1.0, size, device="cuda")
    sequential = torch.arange(size, device="cuda", dtype=torch.int64)
    strided = (sequential * 131_071) % size
    output = torch.empty_like(source)

    def gather(indices):
        torch.index_select(source, 0, indices, out=output)

    sequential_samples = _event_samples(lambda: gather(sequential))
    strided_samples = _event_samples(lambda: gather(strided))
    gather(strided)
    torch.cuda.synchronize()
    correct = bool(
        torch.isclose(output.sum(), source.sum(), rtol=1e-5, atol=1e-2).item()
    )
    return _observation(
        "memory_coalescing",
        "memory_coalescing",
        "kernel",
        sequential_samples,
        strided_samples,
        correctness_passed=correct,
        bytes_read=size * source.element_size(),
    )


def compute_gemm() -> dict:
    size = 4096
    left = torch.randn((size, size), device="cuda", dtype=torch.float16)
    right = torch.randn((size, size), device="cuda", dtype=torch.float16)
    output = torch.empty((size, size), device="cuda", dtype=torch.float16)

    def gemm():
        torch.mm(left, right, out=output)

    samples = _event_samples(gemm)
    gemm()
    torch.cuda.synchronize()
    median_us = statistics.median(samples)
    observed_tflops = (2.0 * size**3) / (median_us * 1_000_000.0)
    return _observation(
        "compute_gemm",
        "gemm_tile_occupancy",
        "kernel",
        samples,
        samples,
        correctness_passed=bool(torch.isfinite(output).all().item()),
        observed_tflops=observed_tflops,
        shape=[size, size, size],
    )


def transfer_overlap() -> dict:
    elements = 4 * 1024 * 1024
    host = torch.ones(elements, dtype=torch.float32, pin_memory=True)
    device = torch.empty(elements, device="cuda", dtype=torch.float32)
    left = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    right = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    output = torch.empty_like(left)
    copy_stream = torch.cuda.Stream()

    def serial():
        device.copy_(host, non_blocking=True)
        torch.mm(left, right, out=output)

    def overlapped():
        with torch.cuda.stream(copy_stream):
            device.copy_(host, non_blocking=True)
        torch.mm(left, right, out=output)

    serial_samples = _wall_samples(serial)
    overlap_samples = _wall_samples(overlapped)
    overlapped()
    torch.cuda.synchronize()
    correct = bool(device[0].item() == 1.0 and torch.isfinite(output).all().item())
    return _observation(
        "transfer_overlap",
        "async_transfer_overlap",
        "runtime",
        serial_samples,
        overlap_samples,
        correctness_passed=correct,
        transfer_bytes=elements * host.element_size(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    capability = list(torch.cuda.get_device_capability())
    if capability != [12, 0]:
        raise SystemExit(f"expected SM120, got {capability}")
    payload = {
        "schema_version": "cuda-optimizer/sm120-diagnostic-scenarios-v1",
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": capability,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "scenarios": [
            launch_graph(),
            memory_coalescing(),
            compute_gemm(),
            transfer_overlap(),
        ],
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
