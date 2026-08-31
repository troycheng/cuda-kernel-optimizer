# Community projects and papers worth absorbing

Date: 2026-08-31  
Scope: research only; no production code or public workflow changes

## Conclusion

The useful external work points in one direction: improve the evidence available to ChatGPT before it commits to an implementation. It does not justify adding another optimizer, controller, planner, fixed profiling pipeline, or copied knowledge corpus.

Three improvements are worth carrying forward:

1. search maintained kernel catalogs and framework benchmark repositories before writing a non-trivial implementation;
2. evaluate the skill with cases reconstructed from real framework changes and production-shaped workloads, not only isolated operator exercises;
3. measure search quality by how quickly the skill establishes the right replacement boundary, rejects an invalid direction, or finds a reusable implementation.

The current V1.4 boundary remains appropriate: ChatGPT interprets evidence and decides what to do; deterministic tools perform one explicit operation and return facts.

## Findings

| Project or paper | What is worth absorbing | Current coverage | Recommendation |
|---|---|---|---|
| [GPU MODE Triton Index](https://github.com/gpu-mode/triton-index) and [Meta TritonBench](https://github.com/meta-pytorch/tritonbench) | Treat community capability discovery as a first-class evidence source. Triton Index catalogs released kernels; Meta TritonBench also exposes operator implementations from projects such as FlashAttention, FBGEMM, Liger Kernel, TileLang and ThunderKittens. Together they provide a much better starting point than a broad web query. | V1.5 already requires a bounded upstream check before non-trivial kernel, primitive or adapter work. It does not prescribe a fixed catalog, which is correct. | **Adopt as preferred live sources**, not as bundled data. Query them together with the target framework's source, releases, PRs and issues. Do not mirror their contents into the local knowledge base; availability, interfaces and maintenance status change too quickly. |
| [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) and [FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace) | Its Definition–Workload–Solution–Trace split keeps the mathematical contract, concrete workload, implementation and measured result distinct. The public dataset uses deployment-derived kernel workloads rather than invented shapes. | The skill already separates Target, Variant, Experiment and result, and requires a real test set. What it lacks is a portable public corpus of production-shaped replay cases. | **Adopt the case-source model; do not replace the artifact protocol.** Future public regression cases can reference a pinned FlashInfer definition/workload and keep the existing skill records as the source of truth. This adds evidence without creating a second schema inside the project. |
| [RealisticTritonBench](https://arxiv.org/html/2608.12004) | The benchmark constructs tasks from real Triton-changing pull requests, restores their engineering context, integrates generated code back into the original framework and runs end-to-end tests. This directly addresses the gap between an isolated fast kernel and an effective framework change. | Existing case studies are evidence-rich but few, and most validate known failures or historical decisions rather than sampling broader real-world integration work. | **Adopt as the strongest evaluation pattern.** Build a small, pinned replay set from real upstream PRs. Score whether the skill finds the true production boundary, reuses existing work, chooses the cheapest useful falsifier and reaches the same keep/reject conclusion. Do not import the full benchmark or turn its tasks into bundled knowledge cards. |
| [KernelBenchX](https://github.com/BonnieW05/KernelBenchX) and its [paper](https://arxiv.org/html/2605.04956v2) | It separates buildability, semantic correctness and efficiency, classifies 176 tasks by computational structure, and reports large cross-hardware variation. The paper also finds that iterative repair can improve correctness while reducing average speedup. | The skill already separates correctness and performance and binds results to hardware identity. It has no broad, category-balanced capability baseline for ChatGPT's kernel work. | **Use only as an external regression corpus.** A small stratified subset—fusion, reduction, quantization, irregular indexing and matrix operations—can expose systematic weaknesses. Do not transfer its eager-baseline speedups into production ROI, and do not equate passing the corpus with workload-level competence. |
| [KernelBrain](https://arxiv.org/abs/2608.02611) | Its most useful result is asymmetric measurement: many candidates receive cheap correctness and low-repeat screening; only a few receive expensive measurement. This is empirical support for spending effort according to remaining uncertainty and potential value. | The skill already requires the cheapest falsifier and prevents expensive work after a valid rejection. It deliberately removed automatic budget controllers. | **Adopt the principle, reject the architecture.** Evaluation should record time and GPU calls until a defensible decision, then compare that cost across skill versions. Do not add rungs, promotion fractions, a scheduler, agent memory, or automatic candidate mutation. ChatGPT can choose different evidence depth for different hypotheses without a persistent state machine. |
| [Mirage: A Multi-Level Superoptimizer for Tensor Programs](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) | Mirage searches across algebraic, kernel, block and thread-level transformations, prunes the search with abstract expressions, canonicalizes equivalent graphs, and validates semantic equivalence before performance selection. Its central lesson is that the useful search space is structured, not a flat list of tuning knobs. | The skill asks ChatGPT to compare subsystems and claim layers, but its offline mechanisms are mainly individual optimization ideas. | **Use as a reasoning model and an upstream capability to try, not a component to reimplement.** For a costly fusion or scheduling direction, first state the transformation level, preserved semantics and a structural impossibility check. If Mirage supports the exact fragment and environment, evaluating or reusing it is preferable to manually rebuilding its search. A local superoptimizer would violate the project's size and responsibility boundaries. |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | Its `fast_p` metric requires both correctness and a speedup threshold, and its levels range from operators to models. More importantly, its own repository describes itself as an evaluation toolkit rather than an agentic scaffold. | The skill already uses minimum effect thresholds and real workload verdicts. | **Reference, but do not integrate.** KernelBench is useful for a stable smoke/regression subset and for comparing model changes. RealisticTritonBench and FlashInfer-Bench are better sources for production-path evaluation, so a large KernelBench integration would duplicate effort. |

## What this changes in the project plan

### 1. Community reuse should be a live lookup, not a larger local database

The existing build-or-reuse rule is directionally complete. The improvement is operational: for Triton work, start with Triton Index and Meta TritonBench, then inspect the owning framework and likely kernel libraries. For serving work, search FlashInfer, vLLM, SGLang, TensorRT-LLM and the relevant communication library at the exact checked-out version.

The output should remain a short capability map:

- primitive already exists or not;
- framework integration already exists, is in progress, or is absent;
- exact production path and version in which it is usable;
- remaining work: reuse, backport, narrow adapter, measurement harness, or genuinely new implementation.

This is evidence prepared for the current decision. It is not a new persistent project object and does not require a new script.

### 2. The next important asset is a small decision-replay suite

The project already tests deterministic tools extensively. The larger unmeasured risk is whether ChatGPT chooses an appropriate direction with limited evidence. A useful replay suite should therefore test decisions, not only code generation.

Each replay case should preserve:

- the original framework change or production-shaped workload;
- the true replacement boundary and execution form;
- available community implementations at that historical version;
- the cheapest observation that should change the decision;
- the measured local and end-to-end outcome;
- the acceptable final conclusion and the claims that must not be made.

The initial set should stay small and structurally diverse. RealisticTritonBench supplies framework-integration candidates; FlashInfer Trace supplies concrete inference workloads; KernelBenchX supplies difficult correctness and hardware-portability categories. These external assets should be pinned by revision and referenced from replay metadata rather than copied into the installed skill.

The useful metrics are:

- correct direction or defensible stop decision;
- correct claim layer and ROI boundary;
- whether an existing community capability was found before implementation;
- elapsed time, GPU invocations and profiler calls until the decision became defensible;
- unnecessary implementation or expensive measurement started before that point.

This evaluates “find an effective direction sooner” without giving a deterministic runner authority to choose directions.

### 3. Knowledge remains bounded and query-driven

Kernel catalogs and benchmark corpora are too volatile and too large to bundle. The offline knowledge base should continue to contain stable technical contracts, falsifiers and carefully reviewed mechanism guidance. Current community availability and framework integration should be retrieved live when a decision depends on them.

If external access is unavailable, the skill can still use the pinned local source records and source tree. It should report that current community capability coverage was not verified; it should not compensate by treating an old copied catalog as current.

### 4. Cost-aware search belongs in evaluation, not in a Controller

KernelBrain supports a principle already present in the skill: weak candidates should not receive the same measurement effort as strong ones. The safe improvement is to measure whether ChatGPT reaches sufficient evidence early and stops unnecessary work. Encoding KernelBrain's rungs or promotion policy would revive the automatic control architecture removed in V1.4 and would be less capable of handling production-boundary and cross-layer evidence.

## Explicitly not recommended

- importing an autonomous kernel agent or its global scheduler;
- creating a project-wide registry that continuously crawls community repositories;
- copying KernelBench, FlashInfer Trace or Triton Index into the installed knowledge package;
- using isolated eager-kernel speedup as production ROI;
- treating profiler output, a benchmark category or a community ranking as a candidate decision;
- adding a new schema merely to reproduce the names used by another project.

## Query workflow

The research used Exa in two passes and reviewed `sources_reviewed: 80` search results across eight targeted queries:

1. LLM-generated CUDA/Triton kernel evaluation, correctness and hardware efficiency;
2. open GPU-kernel benchmark corpora and verifiers;
3. tensor-program superoptimization, equivalence and search-space pruning;
4. official implementations of Mirage, TileLang-like systems and related compilers;
5. cost-aware and multi-fidelity GPU/compiler autotuning;
6. active-learning and transfer approaches for reducing measurement cost;
7. production-derived inference kernel workloads and trace formats;
8. searchable catalogs of released CUDA/Triton implementations.

Thirteen promising primary URLs were then fetched directly. Secondary roundups and projects without runnable code, an original paper, or a maintained first-party repository were excluded. Overlapping benchmarks were retained only when they contributed a distinct mechanism: production integration, category-aware capability analysis, or correctness-plus-performance evaluation.

## Sources reviewed in depth

- [gpu-mode/triton-index](https://github.com/gpu-mode/triton-index)
- [meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench)
- [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench)
- [FlashInfer Trace schema](https://bench.flashinfer.ai/docs/flashinfer-trace)
- [RealisticTritonBench paper](https://arxiv.org/html/2608.12004)
- [KernelBenchX repository](https://github.com/BonnieW05/KernelBenchX)
- [KernelBenchX paper](https://arxiv.org/html/2605.04956v2)
- [KernelBrain paper](https://arxiv.org/abs/2608.02611)
- [Mirage OSDI 2025 paper page](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)
- [Mirage repository](https://github.com/mirage-project/mirage)
- [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench)
- [TritonBench repository](https://github.com/thunlp/TritonBench)
- [FlashInfer repository](https://github.com/flashinfer-ai/flashinfer)
