# Structured production ROI claim

This is a deterministic, CPU-only conformance case. It checks whether an
Experiment can reject an internally inconsistent or economically insufficient
production ROI claim before any workload operation starts. It makes no GPU
speed or general optimization-quality claim.

## Case Snapshot

A real optimization attempt fused Q/K RMSNorm, partial interleaved MRoPE and a
gate operation. An eager CUDA Graph microbenchmark suggested roughly
49.1 to 4.6 microseconds per layer, but the production reference used Inductor
lowering and a full CUDA Graph. Its actual boundary cost was 2.899 microseconds
per layer; the bounded prototype measured 2.316 microseconds per layer.

The production graph contained ten occurrences in a 4314-microsecond replay.
Completely removing the reference boundary therefore had an ideal throughput
ceiling of about 0.677%. Reaching the 0.5% Target threshold required a candidate
below about 0.753 microseconds per layer. The measured prototype could save only
5.83 microseconds per replay, giving an ideal throughput ceiling of about
0.135%. A later directional service screen showed no stable positive signal.

The required behavior is narrow: Experiment creation must preserve the
production execution form and Candidate scope, recalculate the bound, and
reject the 0.135% opportunity before a workload invocation can be created. It
must not treat a self-declared evidence relationship as proof that a measurement
really came from production.

### Public challenge view

The public fixture covers five invariants:

- the production numbers above are below a 0.5% Target threshold;
- eager evidence cannot be labelled as the Inductor production boundary when
  its declared execution form says otherwise;
- a W2-only Candidate cannot include a W1 or dense timing pool;
- a justified conservative reference upper bound remains usable;
- full-removal and measured-prototype bounds remain distinct.

The tool validates explicit identity fields, Candidate scope, pool overlap,
units, hashes and arithmetic. ChatGPT still decides whether evidence is truly
production-equivalent and whether a conservative-bound rationale is credible.

### Audit provenance and environment

The case is a sanitized reconstruction of a Qwen3.5-35B-A3B MXFP6 TP2 decode
experiment on two RTX 5090 GPUs. The candidate mechanism was associated with
public vLLM PR #52676. The conformance evaluation itself uses the repository's
temporary CPU-only fake driver on macOS with Python 3.9.6; it uses no GPU,
network, service or profiler.

### Private material

Runtime image digests, checkpoint paths, raw requests, internal hosts and raw
trace locations remain private and are not required by the public fixture.
