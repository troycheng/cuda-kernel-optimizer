# Structured production ROI claim

This is a deterministic, CPU-only conformance case. It checks whether an
Experiment can reject an internally inconsistent or economically insufficient
production ROI claim before any workload operation starts. It makes no GPU
speed or general optimization-quality claim.

## Case Snapshot

A real optimization attempt fused Q/K RMSNorm, partial interleaved MRoPE and a
gate operation. An eager CUDA Graph microbenchmark suggested roughly
49.1 to 4.6 microseconds per layer, but the production reference used Inductor
lowering and a full CUDA Graph. Its actual boundary cost was 2.899 microseconds
per layer; the bounded prototype measured 2.316 microseconds per layer.

The production graph contained ten occurrences in a 4314-microsecond replay.
Completely removing the reference boundary therefore had an ideal throughput
ceiling of about 0.677%. Reaching the 0.5% Target threshold required a candidate
below about 0.753 microseconds per layer. The measured prototype could save only
5.83 microseconds per replay, giving an ideal throughput ceiling of about
0.135%. A later directional service screen showed no stable positive signal.

The required behavior is narrow: Experiment creation must preserve the
production execution form and Candidate scope, recalculate the bound, and
reject the 0.135% opportunity before a workload invocation can be created. It
must not treat a self-declared evidence relationship as proof that a measurement
really came from production.

### Public challenge view

The public fixture covers five invariants:

- the production numbers above are below a 0.5% Target threshold;
- eager evidence cannot be labelled as the Inductor production boundary when
  its declared execution form says otherwise;
- a W2-only Candidate cannot include a W1 or dense timing pool;
- a justified conservative reference upper bound remains usable;
- full-removal and measured-prototype bounds remain distinct.

The tool validates explicit identity fields, Candidate scope, pool overlap,
units, hashes and arithmetic. ChatGPT still decides whether evidence is truly
production-equivalent and whether a conservative-bound rationale is credible.

### Audit provenance and environment

The case is a sanitized reconstruction of a Qwen3.5-35B-A3B MXFP6 TP2 decode
experiment on two RTX 5090 GPUs. The candidate mechanism was associated with
public vLLM PR #52676. The conformance evaluation itself uses the repository's
temporary CPU-only fake driver on macOS with Python 3.9.6; it uses no GPU,
network, service or profiler.

### Private material

Runtime image digests, checkpoint paths, raw requests, internal hosts and raw
trace locations remain private and are not required by the public fixture.

## Evaluation Definition

This definition is frozen before the recorded evaluation result.

### Project revision and intended axis

- Project revision: `c6411bb33b2eb437ac14dae6e857ec5c40f6572d`
- Production evaluator digest: `256469b542e02e9941c05190694ae57cdc4d6c39469d5c53cb0302f2569fdc19`
- Focused test digest: `f8388258e8afde776ca2053935f694f0e241a177ac2e544603350949ca0ac5c7`
- Shared fixture digest: `8778f4dcc8608d4d40516aeff18f55d3658af3790a35e7b660290ba2fe7885c3`

The only intended behavior change is the required structured
`opportunity_claim` in Experiment creation. No public operation, runner,
Controller, state machine, profiler stage or automatic direction decision is
added. The unused formal-design validator is removed; the paired-order helper
used by the evaluator remains.

### Workload, correctness and resources

Run the five tests in `tests.test_opportunity_claim` once, followed by the full
unit suite, Python compile check, installation self-check, skill quick
validation and `git diff --check`. The fixture is deterministic and CPU-only.
No GPU, network, service, profiler or external model is authorized or needed.

The five focused outcomes are valid only when they respectively:

- reject the measured 0.135% opportunity under a 0.5% Target;
- calculate approximately 0.677% for full removal, 0.753 microseconds for the
  required candidate time, and 0.135% for the measured prototype;
- reject a declared eager/Inductor execution-form mismatch;
- reject a timing pool outside Candidate component scope;
- accept a justified conservative reference upper bound.

Any exception outside the expected rejection envelope, changed evaluator,
failed command, test mutation or unexpected external dependency makes the
trial invalid. No retries are planned.

### Claim ceiling

The evaluation can establish only that the revision freezes a structured ROI
claim, detects the listed explicit contradictions, recalculates the recorded
time-contribution bounds and prevents a below-threshold Experiment from being
created. It cannot prove that a human- or model-authored evidence relationship
is semantically true, that a hash names an available artifact, that an external
probe stayed within a declared cost, or that the change improves GPU
optimization quality in general.

## Evaluation Result

The definition was frozen in commit
`9bd8f3a76a9db66963e9943b4a47127801554e74`. Evaluation ran once on macOS
26.6.2 with Python 3.9.6.

- The five focused tests passed in 2.505 seconds. They observed the expected
  0.677% full-removal ceiling, 0.753-microsecond required candidate time and
  0.135% measured-prototype ceiling, and produced all four declared accept or
  reject outcomes.
- The full unit suite passed 262 tests and skipped two in 43.988 seconds.
- Python compilation, installation self-check, skill quick validation and
  `git diff --check` all returned code 0.
- Self-check ran no GPU or network checks. No trial was retried, interrupted or
  left unrun.
- Production Python decreased from 14,665 to 14,659 lines. The public evaluator
  operation set remained unchanged.

The result supports only the claim ceiling defined above. In particular, the
tool freezes model-authored evidence relationships and rejects explicit
contradictions; it does not independently certify production semantics or
external probe cost.

## Release Decision

No release decision is recorded. Merge, release, installation and remote
publication remain maintainer actions.
