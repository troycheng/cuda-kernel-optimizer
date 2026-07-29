# Validation status

This page describes where the project itself has been exercised. It does not
predict the speedup of a new workload.

## Automated checks

Current branch validation on 2026-07-29 covered 1,360 tests: 1,349 passed and
11 physical RTX 5090 opt-in tests were skipped. Ten require a GPU; the remaining
read-only replay test requires the 5090 archive to be mounted locally. The suite covers input validation, state recovery,
evidence binding, shared-host guards, timeouts, restoration, capability retrieval,
stability calibration, audit cadence, performance-model accounting, bounded
hypothesis admission, targeted evidence selection, and deterministic decision logic. Pre-V1
protocol generation 3.1 added
closed-loop adapter execution, outcome-bound support/opposition, cross-round request
history, content-based project identity, frozen launcher identity, result and artifact
tamper detection, interruption handling, concurrent start/resume, readiness-capability
replay, and project-copy direction experiments. These checks do not
validate the reader's CUDA environment.

## Physical GPU lane

The V1.3 release regression used six retained Triton decision points on a
physical RTX 5090 with container image
`ngc.nju.edu.cn/nvidia/tritonserver@sha256:07c340d3b2de4139ca196ff014ded951bbfec475394c3916aa577c6aac15b308`.
The six retained-case Controller replays were sealed after the source was
frozen at `db5d19c`. Their source manifests, epochs, and decision artifacts are
distinct. The current knowledge route ranked 3 of 4 promoted mechanisms at
Top-1 and Top-3, supplied the valid lowest-cost action in 3 of 4, and proposed
0 profiler actions. The frozen V1.2 route scored 0 of 4 on the first three
measures and proposed 4 profiler actions. V1.2 has neither measured action
duration nor a replayed diagnostic terminal decision for these cases, so the
release makes no time or terminal comparison.

All six observed diagnosis decisions were `PURSUE`. Each was backed by at least
two kinds of local evidence and a benefit ceiling above 1 us. Later candidate
validation promoted four mechanisms and rejected two below the 1 us threshold.
The two rejections are candidate outcomes; they are not rewritten as diagnosis
stage `STOP`. In R10, the knowledge package returned no candidate, but the model
still proposed auxiliary-stream overlap and the Controller admitted it only
after local direction-experiment and Nsys evidence.

These are known cases that also appear in the bundled knowledge package. They
prove retained-case regression, provenance, and stage behavior; they do not prove generalization to a new workload.

A separate Iter0 serving run exercised the unfamiliar-profile path. Three
complete-service baseline blocks produced a median 343.549 QPS with 0.615% CV.
Nsys identified `ParallelNMSSelect<half>` as the largest observed kernel-time
contributor. Deferred output materialization measured +0.084% in the short
pair, and native warp minimum reduction measured +0.043% across two formal
pairs. Both were below the 0.5% service threshold and were rejected before NCU
or promotion.

After the bootstrap fixes, a clean Controller run preserved the unresolved
execution-map scope and admitted the model-proposed
`parallel_nms_select_single_warp_latency` direction even though the knowledge
package had no matching candidate. NCU capability was unavailable, so the run
returned `REVIEW_REQUIRED` without starting NCU. This run verifies that raw
evidence can start a model direction, and that unavailable tooling stops safely;
it does not show that V1.3 found a useful new-workload optimization. No driver,
clock, service, or host policy was changed.

The V1.1 lane passed 24 of 24 checks in 134.726 seconds on a physical RTX 5090
on 2026-07-22. It used immutable compatibility image
`sha256:b810841fe8962f6f65bb48a693773696be778653d48c7903dc65471ca37188a2`.
Four controlled workloads exercised CUDA Graph launch batching, coalesced versus
strided memory access, TF32-disabled versus TF32-enabled 4096x4096 GEMM, and
pinned-memory transfer overlap. For each known fixture hypothesis, the Controller
ran a global measurement, admitted a separate project-copy direction experiment,
reran the GPU workload, sealed the new observation, and only then allowed
`direction_supported` and `PURSUE`. The four paths completed in 9.221--9.659
seconds without a high-cost profiler action.
After the final closed-scope history hardening, the four Controller scenarios
were rerun against the same immutable image and passed in 44.308 seconds.

This is a Controller evidence-admission test. The hypotheses and benchmark-derived
map fixtures are supplied by the test, so the result does not show that an AI can
infer the mechanism from an unfamiliar profile. It also does not predict a new
workload's speedup. The complete lane replayed the
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
