# MXFP6 SM120 exact dispatch and p99 TPOT

[简体中文](case-mxfp6-sm120-tail-latency.zh-CN.md)

This is a small, real-use field feedback example, not a reproducible benchmark. It records why a bounded optimization was retained and what the optimizer learned from that decision.

## Goal and change

The task optimized a custom MXFP6 CUTLASS extension used by a tensor-parallel vLLM service on two RTX 5090 GPUs. The primary goal was end-to-end token throughput, with TPOT and correctness retained as important service metrics.

The candidate reused two existing kernels by adding exact dispatches for `(40, 5120, 3072)` and `(48, 5120, 3072)`. It did not introduce a new kernel implementation. The code change was contributed upstream in [Nekofish-L/mxfp6_sm120#1](https://github.com/Nekofish-L/mxfp6_sm120/pull/1).

## Observed evidence

The two targeted kernel shapes improved as follows:

| Shape | Kernel improvement |
|---|---:|
| `(40, 5120, 3072)` | 17.84% |
| `(48, 5120, 3072)` | 19.34% |

Three paired serving runs then produced this bounded result:

| Metric | Observation |
|---|---|
| p99 TPOT improvement | 2.70%, 7.53%, and 3.05%; median 3.05% |
| Token throughput | median +0.32%; worst pair -0.41% |
| Throughput guardrail | worst pair remained inside the frozen 0.5% non-inferiority bound |
| Correctness and request success | unchanged |

Positive TPOT improvement means lower time per output token. The measurements did not establish a material end-to-end throughput improvement.

## Decision

The candidate was retained as a scoped tail-latency optimization: p99 TPOT improved in all three paired runs while throughput, correctness, and request success remained within their declared constraints. The result was not presented as a throughput win.

The important lesson was about result selection. Whole-workload time share can bound expected throughput or mean-latency gains, but it cannot by itself reject a repeatable p95 or p99 improvement on affected requests. An optimization that improves an important metric and keeps the other key metrics non-regressive is still a valid result when its scope is stated precisely.

## Project feedback

The previous instructions could discard this result because the targeted shapes had limited total-time coverage and the primary throughput metric barely moved. [Commit `16d4f96`](https://github.com/troycheng/cuda-kernel-optimizer/commit/16d4f96) changed the judgment rule to retain stable, non-regressive tail improvements. [Issue #6](https://github.com/troycheng/cuda-kernel-optimizer/issues/6) records the originating feedback.

The same run also exposed two separate workflow costs. Duplicate full-service invocations were tracked in [issue #7](https://github.com/troycheng/cuda-kernel-optimizer/issues/7), and the timing of source research for niche stacks was tracked in [issue #8](https://github.com/troycheng/cuda-kernel-optimizer/issues/8). Both were addressed in V1.5.0 and do not change the result above.

## Evidence boundary

The private multimodal workload and raw service artifacts are not published. The summarized measurements were authorized for this example. They support only the decision reported for this workload and environment and do not predict gains elsewhere.
