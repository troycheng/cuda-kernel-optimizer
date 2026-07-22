# Performance diagnosis and optimization decision engine

Status: approved for implementation planning on 2026-07-22.

## Purpose

The project needs a decision layer that helps an AI find a useful optimization
direction sooner. It must turn workload, source, profiler, and experiment
evidence into a maintained performance model, no more than three competing
mechanism hypotheses, and one justified next action or a terminal decision.

The engine is not a profiler report summarizer, a new Controller, or an
autotuner. Its value is measured by whether it reduces expensive profiling,
invalid candidates, and elapsed time before the first useful direction.

An original business baseline, a correctness reference, a stable benchmark,
and current workload, source, and environment identities remain prerequisites.
The engine must not build a performance conclusion on a benchmark that failed
the existing readiness or evidence checks.

## Outcome contract

Given a frozen workload contract and the evidence available in the current
analysis epoch, the engine must produce:

- an evidence-backed performance model;
- at most three competing mechanism hypotheses;
- the highest-value next evidence action, a candidate recommendation, a request
  for more authorization, or `STOP`;
- a calculable benefit ceiling and the evidence supporting it;
- a time or resource cost range with its basis, or `unknown`;
- the unresolved uncertainty and the next feedback checkpoint;
- an updated decision after every admitted measurement or experiment.

The only decision states are:

- `MEASURE`: collect one named item of evidence that distinguishes live
  hypotheses;
- `PURSUE`: the evidence supports entering candidate implementation;
- `REVIEW_REQUIRED`: a potentially valuable next action exceeds the current
  authorization or remains materially uncertain;
- `STOP`: no admissible direction has enough supported headroom to continue.

Budget expiry is not evidence that a direction lacks value. If a useful action
does not fit the current authorization, the engine returns `REVIEW_REQUIRED`
with a checkpoint and continuation estimate.

## Design invariants

1. Measured facts are produced by deterministic code, never by a model.
2. Every active hypothesis cites supporting evidence, opposing evidence, and a
   falsification question.
3. No more than three hypotheses may remain active at once.
4. Every requested action must resolve a named uncertainty or distinguish a
   named hypothesis pair.
5. NCU is not admitted until the target kernel, execution regime, and question
   are explicit. Full-set or workload-wide NCU collection is not a default.
6. A contradicted or duplicate mechanism is closed; renaming it does not admit
   it again.
7. The engine recomputes the model after new evidence. It does not preserve a
   stale diagnosis merely to continue an earlier plan.
8. External research and external models are advisory. Local evidence remains
   the only promotion authority.
9. Missing evidence lowers the supported claim. It is not replaced by a generic
   optimization recommendation.
10. No implementation change may exceed the file scope in this design without
    first revising and reviewing the design.

## Alternatives considered

### Deterministic expert system

A rule-only system is reproducible but becomes brittle when fusion, dynamic
shapes, compiler behavior, or cross-layer interactions violate a fixed
classification tree. It would encourage more policy files and schemas without
solving source-level mechanism reasoning.

### Model-led agent loop

A model can reason across profiler output and source, but a model-led loop can
invent measurements, overstate confidence, repeat mechanisms, and request
expensive evidence without a reproducible basis.

### Selected: hybrid decision core

Deterministic code owns identity, parsing, arithmetic, critical-path facts,
headroom, action cost, and evidence admission. The model owns source-aware
mechanism hypotheses and explanations. Deterministic policy validates the
proposal and chooses the next action or stop decision.

## Architecture

```text
Frozen workload, source, environment, baseline
                       |
           profiler and experiment artifacts
                       |
               deterministic facts
                       |
       execution map and performance model
                       |
     model-proposed competing mechanisms (<= 3)
                       |
       deterministic evidence/action selection
                       |
       investment brief and next decision
                       |
       new evidence or experiment outcome
                       +-----------------------> recompute
```

### Evidence normalization

The input layer normalizes existing Controller-owned artifacts. It records the
producer version, export schema, subject identity, coverage window, and artifact
digest. An observation that cannot be bound to the current epoch is unavailable
rather than approximate.

The initial implementation supports the current global-scan adapter contract,
Nsys-derived timeline facts, PyTorch profiler evidence, targeted NCU evidence,
and candidate outcomes already represented by the project. It does not create a
new universal profiler plugin system.

Nsys exports require version-aware handling. The `.nsys-rep` identity and the
export schema version are retained because exported SQLite schemas can change.
NCU metric requests are selected from the live question because metric sections
can require multiple replay passes and materially change collection cost.

### Deterministic performance model

The performance model derives:

- observed and missing execution layers;
- execution regimes and shape distributions;
- CPU, framework, CUDA API, kernel, transfer, communication, I/O, and idle
  intervals;
- overlap, serialization, synchronization, and unexplained gaps;
- the critical path and each observed component's reducible contribution;
- hot kernel groups by identity, shape, call count, and critical-path share;
- measured Roofline facts when all required inputs exist;
- a benefit ceiling for each scoped direction;
- contradictory observations and uncertainty boundaries;
- measured durations for build, profile, correctness, screening, and validation
  actions when available.

The model does not equate the longest accumulated kernel time with the best
optimization target. It distinguishes accumulated work from reducible
critical-path time.

### Mechanism hypotheses

The AI receives a compact performance context, a bounded set of matching
knowledge cards, relevant source excerpts, and prior mechanism outcomes. It may
propose at most three hypotheses across `kernel`, `framework`, `cpu_data`,
`transfer`, `communication`, `io`, `environment`, or `mixed` layers.

Each proposal contains:

- the mechanism and affected execution-map nodes;
- the source location or project surface it concerns;
- supporting and opposing evidence identifiers;
- the expected observable consequence if the mechanism is true;
- missing evidence and one cheapest falsification question;
- the expected claim layer;
- an implementation-risk class, not an invented elapsed time;
- rejection and promotion conditions.

The existing hypothesis validator rejects unbound evidence, identity drift,
duplicate mechanisms, unsupported confidence, and stale epoch references.

### Action selection

The selector ranks only actions that answer a live question. It considers:

- how many competing hypotheses the result can distinguish;
- whether the action is necessary for the supported claim layer;
- perturbation and evidence-integrity risk;
- measured or bounded execution cost;
- benefit ceiling and project minimum effect;
- already completed actions and prior mechanism failures;
- remaining authorization.

Static inspection and already-collected evidence are preferred over a new tool
run. A global profiler precedes targeted NCU. A candidate experiment is
admissible when it is cheaper and more discriminating than further profiling.

No universal kernel-share, kernel-duration, or profiler-overhead percentage is
introduced. Those thresholds are workload-dependent and are derived from the
project minimum effect, critical-path headroom, and measured action costs.

### Investment brief

The first brief is emitted after readiness, the original baseline, and the
first global analysis. It is updated before each materially more expensive
stage.

It contains:

- current primary diagnosis and uncertainty;
- supported benefit ceiling and calculation basis;
- the proposed next action and the question it answers;
- a P50/P90 range only when matching observed timing data exists;
- otherwise a qualitative cost class and `unknown` numeric duration;
- the forecast basis: current run, identity-matched history, or unavailable;
- `MEASURE`, `PURSUE`, `REVIEW_REQUIRED`, or `STOP`;
- the next feedback checkpoint.

The engine does not simulate work merely to improve a time estimate. Historical
timings are reusable only when the workload, source, environment, and action
identities match.

The existing hard ceiling remains a process-safety and authorization boundary
for this feature. Per-command timeouts continue to terminate process groups.
The feature does not broadly rewrite budget or resume semantics.

### Evidence update

After an evidence action or experiment, the engine rebuilds the affected
performance model and records whether each hypothesis is supported, opposed,
closed, or still unresolved. It detects bottleneck migration and outcome/model
contradictions. A contradiction creates a new uncertainty; it does not silently
preserve the previous direction.

The engine returns `STOP` when every direction is closed, below the project's
minimum effect, duplicated, unsupported, or more expensive to distinguish than
its supported headroom justifies.

## Optional external challenge

External challenge mode is `auto`. Provider selection uses this order:

1. Google AI Mode;
2. GLM / Zhipu Qingyan;
3. Kimi;
4. DeepSeek;
5. Gemini.

Google AI Mode and Gemini are distinct providers.

One highest-priority available provider is enough for an ordinary challenge.
A major direction choice uses the first two available providers in parallel. A
plateau or final review may use the first three available providers in
parallel. All providers share one 180-second wait bound.

The trigger points are limited to:

- initial selection among materially different high-impact directions;
- conflicting evidence, repeated failure of one mechanism, or a clear plateau;
- final review of a major retained change.

The evidence packet contains the objective, minimum effect, environment and
workload type, compact profiler observations, current hypotheses, failed
mechanisms, and unresolved questions. Proprietary source, raw inputs,
credentials, hostnames, and raw logs are excluded unless the user explicitly
authorizes them.

Each reviewer must state assumptions, contradicting evidence, and a falsifying
test. The local engine maps useful concerns back to local evidence, removes
unsupported or duplicate proposals, preserves disagreement, and selects any
new evidence action. It never uses model voting or provider priority as a
promotion rule.

Unavailable, unauthenticated, timed-out, or invalid providers are recorded and
the next provider is attempted within the shared wait bound. If none succeeds,
the local decision proceeds unchanged. The repository does not embed provider
credentials or vendor API clients. Interactive environments may use an
available browser or connector; non-interactive environments may use the
existing reviewer CLI hook.

## File scope

### New focused modules

- `skills/cuda-kernel-optimizer/scripts/performance_model.py`: deterministic
  performance-state and benefit-ceiling computation;
- `skills/cuda-kernel-optimizer/scripts/diagnostic_decision.py`: compose the
  validated model, hypotheses, evidence selection, investment brief, and final
  decision.

### Existing modules with bounded changes

- `workload_diagnosis.py`: replace coarse classification as the final answer
  with normalized facts consumed by the model;
- `execution_map.py`: expose required deterministic critical-path facts without
  changing epoch or identity ownership;
- `hypothesis_space.py`: retain at most three active hypotheses and mechanism
  deduplication;
- `evidence_selector.py`: select by live discrimination value and measured cost;
- `diagnostic_knowledge.py`: provide only evidence-matched mechanism context;
- `strategy_memory.py`: expose identity-matched timing and mechanism outcomes;
- `workload_reviewer.py`: reuse the existing parallel, redacted, advisory
  reviewer protocol for direction challenge;
- `workload_controller.py`: add narrow call sites after global diagnosis and
  after admitted evidence or experiment outcomes.

### Explicitly excluded

- no new Controller;
- no broad rewrite of `orchestrate.py`, `budget.py`, `run_control.py`, or the
  state machine;
- no new family of JSON schemas;
- no provider credentials or built-in vendor API integrations;
- no workload-wide or full-set NCU default;
- no automatic large source rewrite, general model training, or online service
  control;
- no unrelated documentation or repository reorganization.

## Verification

### CPU behavior tests

Tests must prove behavior, not only structural validation:

- a launch-bound global trace selects a framework/CPU direction and never NCU;
- a transfer-serialization trace selects the transfer path and never NCU;
- a kernel-dominated trace selects only the target kernel and a question-bound
  NCU action;
- a mixed or incomplete trace preserves competing hypotheses and requests the
  cheapest distinguishing evidence;
- contradictory evidence closes or downgrades the earlier hypothesis;
- a renamed duplicate mechanism is rejected;
- an upper bound below the contract minimum returns `STOP`;
- a high-value action outside current authorization returns
  `REVIEW_REQUIRED`, not `STOP`;
- cost estimates without identity-matched observations remain `unknown`;
- unavailable external reviewers preserve the local decision;
- conflicting external reviewers cannot override local evidence;
- external payloads redact protected material and share one 180-second bound.

### Physical RTX 5090 acceptance

Use causally controlled, real workloads for:

1. short kernels with CPU launch overhead and a CUDA Graph intervention;
2. coalesced versus deliberately inefficient memory access;
3. a compute-dominated GEMM regime;
4. serialized versus overlapped transfer or communication.

For all four scenarios:

- the primary bottleneck layer must match the controlled mechanism;
- the known useful mechanism must appear in the top three;
- non-kernel scenarios must launch no NCU action;
- kernel scenarios may profile only the selected target and question;
- counter-evidence must update or close the original direction;
- an investment brief must appear after the first global analysis;
- external-provider failure must not block local completion;
- no scenario may use more expensive actions than current main, and at least two
  scenarios must use fewer;
- a scenario with no qualifying direction must return `STOP`.

The comparison records time to first supported direction, number of expensive
profiler actions, number of invalid candidates before that direction, total GPU
profiling time, and the terminal decision. Schema tests remain necessary but do
not satisfy this acceptance lane.

## Implementation sequence

1. Add causally controlled fixtures and failing behavior tests.
2. Implement deterministic performance-model computation.
3. Implement the bounded hypothesis-to-action decision core.
4. Add the investment brief and evidence update loop.
5. Reuse the external reviewer protocol at the approved triggers.
6. Add narrow Controller integration points.
7. Run CPU/static verification.
8. Run the four physical RTX 5090 acceptance scenarios and compare against
   current main.
9. Review scope, complexity, public documentation, and release notes before any
   remote publication.

## Sources and design challenge

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/2025.4/UserGuide/index.html)
- [NVIDIA Nsight Systems 2026.1 release notes](https://docs.nvidia.com/nsight-systems/2026.1/ReleaseNotes/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
- [PyTorch Chrome trace export](https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.export_chrome_trace.html)

The architecture was independently challenged with Kimi, Gemini, and GLM. The
reviewers agreed on deterministic facts plus model-generated hypotheses and
warned about rule explosion, local optimization traps, fabricated precision,
and recursive profiling. Suggested universal thresholds for kernel share,
duration, and profiler-overhead ratio were rejected because the project
minimum effect, critical path, and measured action cost are the correct local
basis.
