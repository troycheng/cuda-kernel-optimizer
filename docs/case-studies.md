# Case studies

This page separates independently reviewable performance cases from authorized real-use feedback. Their numbers apply only to the recorded workload and environment and are not expected gains for a new project.

## Independently reviewable performance cases

A performance case is published only when the original workload, correctness checks, environment identity, raw measurements, and terminal decision remain available.

The V1.4 branch currently uses retained Triton material for regression validation, not as a positive public performance case. A new case will be added only after its complete evidence can be reviewed independently.

## Real-use feedback examples

These examples report authorized field evidence whose private workload or raw artifacts are not published. They can explain a project decision and its limits, but they are not reproducible benchmarks or formal knowledge.

- [MXFP6 SM120 exact dispatch and p99 TPOT](case-mxfp6-sm120-tail-latency.md) ([简体中文](case-mxfp6-sm120-tail-latency.zh-CN.md)): a small real serving example where two shape-level kernel gains produced a stable p99 TPOT improvement without a material throughput gain.

See [Validation records](validation.md) for project checks.
