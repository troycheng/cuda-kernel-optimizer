# Getting started

## Install with ChatGPT

In an environment that supports Skills CLI, install the current release directly:

```bash
npx skills add https://github.com/troycheng/cuda-kernel-optimizer/tree/v1.5.1/skills/cuda-kernel-optimizer --skill cuda-kernel-optimizer
```

The user does not run repository scripts manually. Send this request in a ChatGPT coding session:

> Install `skills/cuda-kernel-optimizer` from the latest published release tag of [troycheng/cuda-kernel-optimizer](https://github.com/troycheng/cuda-kernel-optimizer). Install only that skill into the active skills directory. When replacing an existing version, keep its backup outside the active skills directory so only one skill with this name is loaded. Run the CPU/static self-check and report the tag, commit, and destination. Do not use the moving `main` branch unless I ask.

Open a new session after installation so the skill instructions reload.

## Prepare the task

Provide:

1. a runnable target: source, binary, deployment, or an existing exported profiler artifact;
2. a test workload: dataset, representative requests, or replay;
3. correctness checks: expected outputs, tolerances, tests, or accuracy criteria;
4. a stable benchmark or service metric;
5. the target GPU, toolchain, framework, and container identity;
6. the performance objective, minimum useful effect, and constraints;
7. the allowed modification scope and host-change boundary.

The test workload must be supplied by the user. The skill does not download, invent, or silently replace it with a microbenchmark.

## Run a short fit check

> Use cuda-kernel-optimizer for a read-only fit check of this project. Spend at most 10 minutes. Do not edit source, install dependencies, or change host settings. Confirm the workload, correctness checks, benchmark, target GPU, and profiler access. Report blockers, the strongest currently supportable claim, and the lowest-cost next step. Do not claim a speedup.

This check answers whether the project can be measured, what is missing, and whether formal optimization is worth starting.

## Start a formal run

> Use cuda-kernel-optimizer to optimize this Triton workload. Use my workload and correctness checks as authoritative. Optimize end-to-end latency with a 0.5% minimum useful effect. Modify only the specified directory and leave host settings unchanged. Run the original baseline first, then explain the main bottleneck, lowest-cost falsifier, and expected investment before implementing a candidate.

ChatGPT freezes a Target, measures original, creates one explicit Experiment for each candidate, and invokes only the operations needed to test the current hypothesis. It reports progress for long operations and finishes with the artifact directory and terminal reason.

See [Preparing a workload and environment](environment-readiness.md) if the foundation is incomplete, and [Optimization workflow](workflows.md) for the durable records.
