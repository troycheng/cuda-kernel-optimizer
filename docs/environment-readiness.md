# Preparing a workload and test environment

Readiness establishes what the current inputs can prove before code changes or expensive profiling begin. ChatGPT runs it; the user supplies the real target and authorizes any environment changes.

| Available foundation | Strongest normal outcome |
|---|---|
| Source only | Static analysis and a preparation plan |
| Source, build, and correctness checks | Correctness and compiler observations |
| Stable kernel workload | Kernel performance result |
| Representative complete workload | End-to-end workload result |
| Controlled service experiment | Serving result |
| Existing exported profiler artifact | Read-only diagnostic facts within that artifact |

## Required for optimization

- original source, binary, or deployment;
- user-provided workload and case identities;
- expected outputs, tolerances, or accuracy criteria;
- a command driver implementing the installed request/result protocol;
- primary metric, minimum useful effect, constraints, and paired-sampling requirements;
- target GPU and required tool identities;
- a low-cost correctness smoke test;
- bounded file scanning and operation timeouts.

`readiness.py check` freezes these values in `target.json`, stores the original Variant by content digest, and runs the smoke test. A failed original correctness check stops performance work.

## Missing tools and repairs

ChatGPT may install ordinary analysis packages in an isolated environment when the user has authorized it. Driver, GPU permission, clocks, power, services, and container-runtime changes remain recommendations by default. Environment repair is reported separately from performance iterations.

Host changes are never inferred from permission to optimize project code.

The installed self-check verifies the final package surface, dependency graph, driver templates, offline knowledge closure, and runtime lock directory. It does not prove that the target GPU or workload is ready.

For the exact installed protocol, see the [environment readiness reference](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.6.0/skills/cuda-kernel-optimizer/references/environment_readiness.md).
