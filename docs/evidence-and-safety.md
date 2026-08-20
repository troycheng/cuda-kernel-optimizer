# Evidence and safety

## Correctness before performance

Every driver call records correctness and measurements together, but performance samples are interpreted only after the Target gate and any Experiment-specific invariant gates pass. A failed or missing gate invalidates the affected performance result and prevents later expensive evidence calls. A final audit may continue only far enough to establish whether the original can be restored safely.

## Paired measurement

Original, Candidate, and Champion comparisons use the same frozen workload, runtime identity, comparison contract, and sampling design. The contract states what relationship is being tested, which state is shared or rebuilt, and whether subjects run in isolated processes or the same process. Raw ordered samples remain available. Point estimates are reported with uncertainty and constraints; a positive point estimate alone is not a passing result.

Shared-host or serving measurements also retain queue, clock, temperature, cache, concurrency, and traffic observations when available. Predeclared contamination rules determine which pairs are comparable.

## Immutable identity

Target, Variant, Experiment, tool executable, profiler artifact, and result identities are content-bound. A source, binary, report, tool, or runtime change invalidates the affected comparison. Container evidence records the final runtime identity plus confirmed base, overlay, and component identities. Incomplete lineage narrows attribution to that frozen runtime instead of guessing an upstream source. Symlinks, path traversal, mutable sibling files, and unknown report dialects fail closed.

## Profiler limits

Profiler output is diagnostic evidence, not promotion evidence by itself. NCU, Nsys, and PyTorch parsers accept only tested versions and interpretation-critical fields; non-critical extensions are retained as unmodeled material. `ERR_NVGPUCTRPERM` records unavailable hardware counters without changing host policy.

## Opportunity estimates

ROI is a derived evidence claim. Its timing, coverage, and cost inputs must apply to the production boundary the Candidate actually replaces, including the selected lowering, graph, dispatch, fallback, and overlap behavior. Evidence from another execution form or another component remains diagnostic unless it is shown to be a conservative upper bound. When a formal workload result contradicts the estimate, correct the failed assumptions before ranking another Candidate.

## Modification boundary

The skill changes only user-authorized project files and isolated environments. Host drivers, permissions, clocks, power, services, containers, and system configuration remain recommendations unless separately authorized.

The CPU/static self-check validates the installed package, not the user's GPU environment. Target-side readiness and actual workload execution are still required.
