# Validation records

This page describes what the project itself checks. It does not predict the speedup or direction-finding accuracy of a new workload.

## CPU/static release gate

The release gate covers:

- the exact 17-module production surface and acyclic dependency direction;
- Target, original baseline, Candidate, Experiment, screen, target, Champion, and final-audit black-box behavior;
- Invocation deduplication, resource locks, heartbeats, timeout, cancellation, worker loss, and cleanup;
- the single Driver V2 path, closed evidence bundles, and isolated or same-process acquisition contracts;
- correctness invalidating performance interpretation and preventing unnecessary later evidence calls;
- NCU, Nsys, PyTorch Profiler, compiler, SASS, and execution-map known-format parsing;
- fail-closed behavior for unknown interpretation-critical fields, versions, units, identities, paths, and digests, with non-critical extensions retained as unmodeled material;
- bounded, identity-filtered offline knowledge queries and empty-result behavior;
- installable package structure, README links, metadata, license, and self-check.

The documentation intentionally does not embed a live test count. The current count belongs to the CI run attached to a commit; copying it into prose creates stale release claims.

## Physical RTX 5090 lane

Physical tests are opt-in and run on the configured SM120 host. They validate Linux process behavior, CUDA/Triton execution, profiler collection where permissions allow, and replay of retained Triton evidence. They do not modify host driver, counter policy, clocks, power, or services.

Unprivileged NCU collection may return `ERR_NVGPUCTRPERM`. This is an expected capability result: the invocation must retain the tool identity and error, produce no fabricated counter observations, and leave host policy unchanged.

V1.4 release validation uses the read-only material supplied through `CUDA_V14_HANDOFF_ROOT` as an immutable regression source. The replay checks current statistical interpretation and archived decision fields separately without rewriting the source. It is not evidence that the current statistic independently validates a historical decision, or that the skill will discover the same mechanisms on an unseen workload.

The 2026-08-01 release run completed on an idle RTX 5090 using immutable image `sha256:b810841fe8962f6f65bb48a693773696be778653d48c7903dc65471ca37188a2`. Five acceptance checks passed in 47.858 seconds. The real Triton path retained Target, Experiment, four Invocation results, Champion selection, and final-audit records. Its one-pair diagnostic screen remained inconclusive but showed a large valid signal; a separately requested formal comparison and final audit both passed. In the retained Iter0 replay, recomputed statistics remained inconclusive while both point estimates stayed below the configured threshold; the archive separately retained its historical `REJECT`, `REJECT`, and `STOP` fields. Digest checks confirmed that the archived inputs were unchanged. The selected GPU had no compute process and no validation mount remained after the run.

Exact commands and prerequisites are maintained in the [RTX 5090 test guide](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.4.2/tests/gpu/sm120/README.md). Workload-specific outcomes belong in [Case studies](case-studies.md).
