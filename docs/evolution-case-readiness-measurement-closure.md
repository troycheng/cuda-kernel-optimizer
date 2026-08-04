# Readiness measurement closure and system-gate behavior

This is a prospective conformance and behavioral case. It checks bounded
protocol validation and model instructions. It is not a GPU performance case
and makes no workload speed, hardware, or general optimization-quality claim.

## Case Snapshot

An optimization skill must reject an incomplete measurement contract before a
baseline or Candidate is created. For the current V1.4 Target contract, a new
optimization Target must use one combined readiness probe that returns both
correctness and measurements. The returned metric names, units, constraint set,
and sample counts must exactly match the request.

The skill must also keep local profiler evidence at its supported claim layer.
Before selecting a Candidate, and again before a formal target comparison, it
must check Target applicability, phase, coverage, system attribution, and the
idealized end-to-end upper bound. A formal shared-host measurement without a
continuous, time-aligned resource observation is inconclusive for performance.

### Public challenge view

The deterministic challenge uses the repository's temporary CPU-only fake
driver and covers:

- a separate optimization driver and a correctness-only smoke;
- an undeclared constraint in a combined smoke result;
- fewer samples than requested by readiness or baseline;
- a failed command with long stdout and stderr;
- one valid combined readiness result.

Two clean model sessions receive only synthetic facts:

1. a steady-state Target and a profiler report from a different concurrency and
   mixed phase, plus a faster single shape whose Target coverage is unknown;
2. a screened Candidate, a new profiler fact from a different phase, and a
   formal shared-host run that has only one pre-run resource snapshot.

No historical answer, private workload, or production trace is part of either
challenge.

### Audit provenance and private material

A private optimization task motivated the scope, but it is not public proof.
No private workload, internal trace, weights, proprietary source, model path,
credential, host address, or business data is attached or required. The public
fake-driver tests and synthetic prompts stand independently.

### Environment and authorization

The deterministic evaluation is CPU-only on Darwin arm64 with Python 3.9.6.
It uses no GPU and no network. The behavioral evaluation uses read-only clean
sessions and does not authorize repository, service, or artifact writes.

## Project Revision

- Original: `27a2e1507b865796ea25666fab367b5592e89e3d`
- Candidate: `726ecb7847e5b9ce30b1eea81d408ea057eee043`

The candidate changes only the existing V1.4 readiness/evaluator seams, driver
documentation, model instructions, and their public regression tests. It adds
no public operation, protocol version, required field, persistent runtime
object, automatic next action, or performance claim.

## Evaluation Definition

This definition is frozen before the evaluation result is produced.

### Comparison arms and intended axes

The deterministic comparison runs the focused evaluator from Candidate against
both Original and Candidate production code. The only intended axis is the
production revision. Test source, Python, fixture inputs, commands, and host are
held equal.

The behavioral comparison uses separate clean sessions for Original and
Candidate. Both arms use the same model family, reasoning effort, raw prompt,
authorization, and lack of historical answer. The only intended axis is the
skill and directly referenced documentation.

### Deterministic evaluator

Use these Candidate test files unchanged in both isolated worktrees:

- `tests/test_readiness.py`
- `tests/test_v14_target_baseline.py`
- `tests/v14_support.py`

Run these focused tests:

```text
tests.test_readiness.ReadinessPublicSurfaceTests.test_optimization_rejects_separate_driver_before_execution
tests.test_readiness.ReadinessPublicSurfaceTests.test_combined_smoke_rejects_an_undeclared_constraint
tests.test_readiness.ReadinessPublicSurfaceTests.test_combined_smoke_requires_two_measurement_samples
tests.test_readiness.ReadinessPublicSurfaceTests.test_smoke_command_failure_bounds_streams_before_the_error_envelope
tests.test_readiness.ReadinessPublicSurfaceTests.test_combined_readiness_accepts_a_closed_two_sample_result
tests.test_v14_target_baseline.TargetAndBaselineTests.test_baseline_rejects_sample_count_mismatch_before_aggregation
tests.test_v14_target_baseline.TargetAndBaselineTests.test_baseline_preserves_exact_constraint_contract_failure
```

Then run the Candidate full unit suite, compile check, self-check, skill quick
validation, and `git diff --check`.

### Behavioral evaluator

Use `gpt-5.6-terra` with high reasoning effort in four new read-only sessions:
one Original and one Candidate session for each public challenge. Runtime build
identity is not exposed by the harness and must be recorded as unpinnable. Each
session reads only its assigned `SKILL.md` and directly routed references; it
must not read tests, repository diff, this definition, or another arm's answer.

The profile-applicability trial is valid when the answer states the strongest
supported claim, decides whether to create a kernel Candidate or formal
Experiment, and identifies the cheapest next observation. It passes only if it
does not promote mismatched phase/concurrency evidence or unknown shape
coverage into a workload or serving claim.

The formal-resource trial is valid when the answer decides whether to run or
accept the formal target result, handles a newly introduced profiler fact, and
leaves a terminal handoff. It passes only if missing continuous resource
evidence makes performance inconclusive, correctness remains independently
usable, no Champion is selected, and the recoverable version and uncovered
risks are explicit.

### Outcomes, budget, and claim ceiling

A trial is valid only if it uses the bound arm and evaluator. Command failure,
session leakage, wrong revision, or changed prompt makes the trial invalid and
requires a fresh session. No GPU or network call is allowed. One execution per
deterministic arm and one clean session per behavioral arm are authorized.

The strongest supported claim is that Candidate rejects the listed public
invalid inputs, accepts the listed valid input, and makes more complete bounded
decisions on the two public synthetic scenarios than Original. The evaluation
cannot establish GPU speed, production correctness, general model reliability,
all driver compatibility, or improvement across CUDA workloads.

## Evaluation Result

Pending independent evaluation.

## Release Decision

Pending human maintainer review. An evaluation result will not select a release.
