# Getting Started

## Install with ChatGPT

The reader does not run installation commands or project scripts by hand. Send
this request in a ChatGPT coding session:

> Install `skills/cuda-kernel-optimizer` from [troycheng/cuda-kernel-optimizer](https://github.com/troycheng/cuda-kernel-optimizer) at its latest published release tag. Install only that skill into the active skills directory, run its CPU/static `self_check`, and report the installed tag, commit, and destination. Do not use `main` unless I ask.

Use the moving `main` branch only when deliberately testing unreleased changes.
Start a new session after installation so the agent reloads the skill.

## Run a 10-minute fit check

Use this read-only check before selecting a formal optimization budget:

> Use cuda-kernel-optimizer for a read-only fit check of this project. Spend at most 10 minutes. Do not edit source files, install packages, or change host settings. Confirm the runnable target, correctness reference, benchmark, target GPU, and profiler access. Report the supported claim layer, blockers, missing evidence, and the first lowest-cost action. Do not claim a speedup.

The check does not claim a speedup. It answers whether optimization can start,
what result the current setup could support, and what must be prepared first.

## Prepare the task

Provide as much of the following as currently exists. The skill first reports a
claim ceiling and helps prepare missing foundations before formal optimization:

1. A **runnable target**: kernel code, a complete workload, or an existing
   `.ncu-rep`.
2. A **correctness reference**: reference implementation, tests, validator, or
   comparable expected output.
3. The **test environment**: target GPU, driver, toolchain, dependencies, and
   access boundaries.
4. A **performance goal**: latency, throughput, memory, cost, or another primary
   KPI, including its direction and threshold.
5. **Constraints**: accuracy, checksums, output quality, memory limits, and any
   per-case requirements.
6. The **allowed modification scope**: project paths and isolated environment
   locations that may change.

A real workload must be supplied by the user. The skill does not download,
invent, or replace it with a microbenchmark. Without one, the strongest possible
result is a kernel-level claim.

If the runnable target, correctness reference, or stable benchmark is missing,
start with [Environment readiness](environment-readiness.md). Source-only work
may produce useful hypotheses, but not a performance result.

## What the AI does in a formal run

1. Freeze the workload, objective, constraints, environment, allowed paths, and
   measurement policy.
2. Run the project's original business baseline before testing a candidate.
3. Evaluate each candidate from the cheapest falsifier through correctness,
   short paired timing, profiling when needed, and formal workload evidence.
4. Keep a change only when its declared claim passes; otherwise restore the
   previous implementation and record the stop reason.
5. Report progress during long work and finish with the exact run directory.

## Choose a budget

| Budget | Maximum wall time | Use it for |
|---|---:|---|
| `quick` | 45 minutes | Check an idea and narrow the candidate set |
| `balanced` | 3 hours | Default search and validation depth |
| `thorough` | 10 hours | Broader exploration and deeper evidence |

These are ceilings. A task may stop earlier when it has a conclusive result, no
eligible candidate remains, or required evidence is unavailable.

Each budget also supplies a default stability policy: confidence and power
targets, bootstrap count, minimum valid calibration pairs, and a recurring
audit cadence. The frozen workload contract records the effective values, so a
long run cannot silently relax them later. Users may override these values
before the contract is frozen.

## First request

> Use cuda-kernel-optimizer to optimize the Triton kernel in this directory. Confirm the runnable reference, inputs, performance goal, constraints, allowed files, and target environment before profiling. Use the balanced budget and keep a change only when correctness and paired performance both pass.

Next, select the matching [workflow](workflows.md) and review the
[evidence and safety boundaries](evidence-and-safety.md). For work that may run
for many iterations, also review the
[long-running optimization loop](long-running-optimization.md).

## Inspect the result

Start with `<run-dir>/summary.md`. It states the terminal result, elapsed time,
stop reason, retained or restored change, strongest supported claim, and missing
evidence. Use `<run-dir>/itervN/decision.json` for the machine-readable
correctness, performance, constraint, and evidence-integrity decision. Raw
paired samples and manifests remain in the same run directory for audit.

A change is **merge-ready** only when the declared workload objective,
correctness reference, constraints, and evidence integrity all pass. A
kernel-only improvement is not merge-ready for an end-to-end workload claim.
