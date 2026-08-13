# MXFP6 SM120 MoE optimization from kernel to serving

[简体中文](case-mxfp6-sm120-moe-end-to-end.zh-CN.md)

This is an authorized real-use feedback example, not a reproducible benchmark. It records which result survived the complete-service objective, which attractive local result did not, and which optimizer decisions changed because of the case.

## Goal and evidence contract

The task optimized the MXFP6 path of a Qwen3.5-35B-A3B MoE service on two RTX 5090 GPUs with tensor parallelism. The primary metric was output-token throughput on a frozen 64-request real multimodal workload at concurrency 4. TPOT, request completion, server-reported token counts, layer correctness, and reference-output fidelity were retained as guardrails.

The measurements used an internally patched vLLM 0.25.1 environment. The executable original was the developer's MoE feature baseline; the public repository `main` at the start of the work did not contain the MoE implementation and could not run the same workload.

## Retained optimization

The retained kernel change specialized the batch-4 W2 path: it used aligned packed loads for the 12-byte FP6 weight segments, preserved a scalar fallback, and kept routed and shared-expert weights separate. The public CUDA Graph layer result reports exact output equality and this paired TP2 measurement:

| Scope | Reference | Candidate | Improvement |
|---|---:|---:|---:|
| Complete batch-4 MoE layer | 26.618 us | 24.647 us | 8.00% |

The full layer samples and measurement contract are published in the [B4 vector-load result manifest](https://github.com/Nekofish-L/mxfp6_sm120/blob/main/benchmarks/results/qwen35_moe_b4_vector.json).

On the frozen service workload, the final MXFP6 Champion improved over the executable developer baseline as follows:

| Metric | Change |
|---|---:|
| Output-token throughput | +4.092% |
| Mean TPOT | -3.498% |
| p99 TPOT | -3.275% |

The two time-ordered throughput pairs were +4.513% and +3.679%. Every formal block completed 64/64 requests with the same 203,631 prompt tokens and 59,193 completion tokens reported by the server.

## FP8 delivery comparison and fidelity

The same MXFP6 build was also compared with the official same-architecture FP8 checkpoint. This is a deployment-format comparison, not the optimizer's incremental gain over its original implementation. MXFP6 produced +13.581% output-token throughput, -10.085% mean TPOT, and -7.960% p99 TPOT in that environment. The full block results and identities are published in the [service result manifest](https://github.com/Nekofish-L/mxfp6_sm120/blob/main/benchmarks/results/qwen35_moe_service_tp2.json).

Both formats completed 742/742 cases in a paired reference-output evaluation. Full normalized-character similarity was 35.62% for FP8 and 35.05% for MXFP6, a difference of -0.57 percentage points with a paired 95% confidence interval of [-1.29, +0.15]. Answer-section similarity differed by -0.32 points with a 95% interval of [-1.62, +0.95]. No statistically significant reference-fidelity loss was detected, but this was not an audited business-accuracy evaluation.

## Decisions that mattered

### A local winner was not promoted without service coverage

A batch-2 extension improved its isolated TP2 layer by 11.03%. The measured service graph, however, contained no batch-2 W2 launches. It was recorded as a local result and excluded from the service Champion rather than being presented as additional end-to-end gain.

This was the practical value of requiring a kernel candidate to pass through complete-layer coverage and then the frozen service target. Kernel speedup alone did not determine the result.

### A promising path was closed narrowly

Several W1 load, staging, split-K, unroll, and pipeline implementations failed correctness or complete-layer gates. Those experiments closed the measured implementations, not every future W1 optimization. The same narrow wording was used for unsuccessful communication fusion and GDN experiments.

This kept negative evidence useful without turning one implementation failure into an unsupported statement that an entire mechanism had no remaining potential.

### Late source verification caused a measurable regression

During submission preparation, valid CUDA Programmatic Dependent Launch behavior was initially judged unsafe from an incomplete semantic model. Removing it reduced the service mean from 665.83 to 655.70 output tokens/s, about 1.52%. Review of the CUDA and CUTLASS primary documentation showed that the dependent grid's entry wait supplied the required completion and memory-visibility semantics, so the removal was rejected and PDL was restored.

The costly behavior was not specific to PDL: an unfamiliar external-stack fact was allowed to drive a rejection before the relevant primary source had been checked.

## Project feedback produced by the case

The case produced four scoped changes or backlog items instead of one broad workflow rewrite:

- [Issue #15](https://github.com/troycheng/cuda-kernel-optimizer/issues/15): bind derived-container identity, not only inherited image labels or package versions.
- [Issue #16](https://github.com/troycheng/cuda-kernel-optimizer/issues/16): use same-process pairing when small kernel-selection effects can be hidden by startup and cache variance.
- [Issue #17](https://github.com/troycheng/cuda-kernel-optimizer/issues/17): separate implementation parity from cross-checkpoint fidelity.
- [Issue #18](https://github.com/troycheng/cuda-kernel-optimizer/issues/18): verify relevant primary sources before an unfamiliar external-stack fact drives a safety, correctness, or terminal decision.

The optimizer instructions were also tightened so that promising failed candidates require evidence proportional to their potential impact and so that a rejection closes only the tested implementation and conditions.

## Evidence boundary

The private multimodal workload, raw traces, model artifacts, and internal runtime patches are not published. The summarized measurements were authorized for this example; public result manifests retain the shareable environment and metric identities. The case supports the decisions above only for the recorded SM120, TP2, model, and runtime conditions. It neither predicts gains elsewhere nor claims that all remaining MoE optimization space was exhausted.
