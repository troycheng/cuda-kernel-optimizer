# Evidence and safety

## Correctness before performance

Every Candidate is checked against the Target's expected outputs, tolerances, or accuracy criteria before its performance samples are interpreted. A failed or missing correctness result blocks profiling and formal measurement for that candidate.

## Paired measurement

Original, Candidate, and Champion comparisons use the same frozen workload, environment identity, and sampling design. Raw ordered samples remain available. Point estimates are reported with uncertainty and constraints; a positive point estimate alone is not a passing result.

Shared-host or serving measurements also retain queue, clock, temperature, cache, concurrency, and traffic observations when available. Predeclared contamination rules determine which pairs are comparable.

## Immutable identity

Target, Variant, Experiment, tool executable, profiler artifact, and result identities are content-bound. A source, binary, report, tool, or environment change invalidates the affected comparison. Symlinks, path traversal, mutable sibling files, and unknown report dialects fail closed.

## Profiler limits

Profiler output is diagnostic evidence, not promotion evidence by itself. NCU, Nsys, and PyTorch parsers accept only tested versions and interpretation-critical fields; non-critical extensions are retained as unmodeled material. `ERR_NVGPUCTRPERM` records unavailable hardware counters without changing host policy.

## Modification boundary

The skill changes only user-authorized project files and isolated environments. Host drivers, permissions, clocks, power, services, containers, and system configuration remain recommendations unless separately authorized.

The CPU/static self-check validates the installed package, not the user's GPU environment. Target-side readiness and actual workload execution are still required.
