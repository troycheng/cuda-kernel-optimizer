# Validation status

This page describes where the project itself has been exercised. It does not
predict the speedup of a new workload.

## Automated checks

The local CPU/static suite ran 1,179 tests on 2026-07-22: 1,169 passed, 10 physical RTX 5090 opt-in tests were skipped, and none failed. It covers input
validation, state recovery, evidence binding, shared-host guards, timeouts, restoration, capability retrieval,
stability calibration, audit cadence, performance-model accounting, bounded
hypothesis admission, targeted evidence selection, and deterministic decision logic. Pre-V1
protocol generation 3.1 added
closed-loop adapter execution, outcome-bound support/opposition, cross-round request
history, content-based project identity, frozen launcher identity, result and artifact
tamper detection, interruption handling, concurrent start/resume, readiness-capability
replay, and project-copy direction experiments. These checks do not
validate the reader's CUDA environment.

## Physical GPU lane

The V1.1 lane passed 24 of 24 checks in 98.813 seconds on a physical RTX 5090
on 2026-07-22. It used immutable compatibility image
`sha256:b810841fe8962f6f65bb48a693773696be778653d48c7903dc65471ca37188a2`.
Four controlled workloads exercised CUDA Graph launch batching, coalesced versus
strided memory access, 4096x4096 FP16 GEMM, and pinned-memory transfer overlap.
Each real observation then passed through the production performance model,
hypothesis admission, evidence selector, and decision logic. All four reached the
expected supported direction in 2.3--2.9 milliseconds without an expensive
profiler action.
The measurements confirm that these mechanisms and the target path execute; they
do not predict a new workload's speedup. The complete lane also replayed the
existing readiness, active-diagnosis, paired-measurement, restoration, and
promotion paths. The default unprivileged lane returned `ERR_NVGPUCTRPERM`.
A separate, explicitly authorized disposable container added only `SYS_ADMIN`,
completed a nine-pass NCU smoke profile, and was removed afterward. The host
kept `RmProfilingAdminOnly: 1`; no driver, package, or host setting was changed.

The final protocol-generation 3.1 completion lane passed 20 of 20 checks in 58.876 seconds on a physical
RTX 5090 on 2026-07-20. It used immutable image
`sha256:a2d9d89bc4394eab3fadc62c6b5b3f739b6494c1f64c56f5ba5e6c008252a0e5`.
The new active-diagnosis test executed a real PyTorch CPU/CUDA profile action,
sealed its 14,341-byte Chrome trace and observed outcome, bound the outcome's
support/opposition effects, and returned to the next hypothesis round. No host
setting or package was changed.

The protocol-generation 3.1 readiness lane completed 18 of 18 checks in 52.141 seconds
on 2026-07-20 with the same immutable image listed below. Readiness itself took
8.793 seconds; the first baseline artifact appeared 9.297 seconds after the run
started. CUDA 13.3 target compilation, SM120 execution and SASS, Compute
Sanitizer, and the Triton correctness/KPI smoke passed. Nsys was absent and was
recorded as diagnostic degradation. A real NCU target range returned
`ERR_NVGPUCTRPERM`, so the report requested user action without changing host
policy. All required capabilities passed and the baseline ran afterward.

The historical protocol-generation 3.0 Controller produced its first baseline artifact in about
0.014 seconds because it had no readiness stage. The added 9-second startup cost
is not evidence that protocol generation 3.1 finds a useful direction faster. That claim still
requires a long user workload showing fewer tool repairs, repeated probes, and
unproductive profiling rounds.

The V3 RTX 5090 lane completed 15 of 15 checks in 34.307 seconds using immutable
container image
`sha256:a2d9d89bc4394eab3fadc62c6b5b3f739b6494c1f64c56f5ba5e6c008252a0e5`.
Its new long-run test measured eight real identical-kernel pairs. The observed
noise median was 34.153%, the upper confidence bound was 36.712%, and the
minimum detectable effect was 40.193%, above the frozen 0.5% practical effect.
The Controller therefore stayed in `CALIBRATING` instead of admitting an
optimization claim. Target-side NCU collection returned `ERR_NVGPUCTRPERM`;
the workflow reported the permission boundary and did not change the driver or
counter policy.

Exact commands and opt-in requirements are maintained in the
[RTX 5090 test guide](../tests/gpu/sm120/README.md). Toolchain and architecture
rules are listed in [Compatibility](compatibility.md).

## What these checks mean

They show that the project workflow, evidence files, and failure paths behaved
as recorded in those environments. They do not show that every CUDA, CUTLASS,
Triton, framework, or serving workload is supported, and they are not a general
performance guarantee.

Workload-specific results are kept separately in [Case studies](case-studies.md).
