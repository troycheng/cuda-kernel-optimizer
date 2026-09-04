# Triton OCR serving knowledge contracts

This is a prospective knowledge-scope conformance case. It checks whether the
bundled knowledge can return bounded, source-reviewed Triton serving contracts
that are useful before a stable OCR model and profile exist. It is not a GPU
performance case and makes no OCR speed, correctness, compatibility, or
configuration-optimality claim.

## Case Snapshot

### Original request and required outcome

The project is used for inference optimization work that includes OCR services
running on NVIDIA Triton Inference Server. While the model structure is still
unstable and a representative profile cannot yet be collected, the immediately
useful project change is to make stable serving contracts queryable without
inventing model-specific advice.

The required outcome is a small knowledge-only increment covering Triton
measurement semantics, scheduling and dataflow boundaries, cache eligibility,
TensorRT shape and graph constraints, ONNX Runtime TensorRT fallback, and
encoded-image batching. The change must preserve source identity, version
sensitivity, claim-layer limits, and explicit nonclaims.

### Public challenge view

At Original revision `2dc7514`, the bundled registry contains 83 cards and 29
sources. It contains no dedicated card matching OCR, dynamic batching, instance
groups, ensemble models, response cache, Model Analyzer, Performance Analyzer,
DALI, or ONNX Runtime. Existing uses of “Triton” primarily refer to Triton
Language or narrow historical optimization cases rather than the inference
server's general serving contracts.

The public challenge asks deterministic knowledge queries for:

- Triton request, inference, execution, queue, and compute metric semantics;
- dynamic batching and model-instance interaction boundaries;
- ensemble dataflow and response-cache eligibility;
- TensorRT dynamic-shape profile and CUDA Graph identity constraints;
- ONNX Runtime TensorRT execution-provider fallback;
- variable-length encoded-image batching with the Triton DALI backend.

Each answer must remain attached to an official source, expose its nonclaims,
and avoid a compatibility or performance conclusion when the request does not
provide a bounded framework version or representative workload.

### Audit provenance

The request arose from an internal OCR optimization workstream, but the public
problem is independently visible in Original's committed registries and can be
evaluated with public NVIDIA documentation and deterministic CPU-only tests.
No internal result is needed to prove the knowledge gap or assess the change.

### Environment and authorization

Repository edits, CPU-only tests, and local Git commits are authorized. GPU
profiling, service changes, model conversion, production traffic, publishing,
and remote pushes are outside this case. The target Triton, backend, TensorRT,
CUDA, GPU, OCR model, input-shape distribution, concurrency, and service SLO
identities are currently unknown.

### Private material

The internal service topology, model files, weights, images, traces, traffic
distribution, metrics, host identities, and prior optimization results remain
local and are not attached or used as public proof.

### Safe public derivative

The safe derivative is limited to source-reviewed technical-contract cards,
their public source records, deterministic query tests, and this evolution
record. It contains no private artifact, heuristic OCR recommendation, target
configuration, performance number, or automatically selected next action.

## Evaluation Definition

This definition is frozen before the candidate content is added or its result
is produced.

### Repository revision or evaluation arms

- Original: `2dc7514` (`v1.5.1`, current `main` before this case).
- Candidate: the later local commit containing only the declared public source
  records, technical-contract cards, deterministic query tests, registry date
  updates, and this evolution record.

The comparison uses different repository revisions. No runtime service or GPU
condition is an evaluation arm.

### Only intended comparison axis

The only intended axis is whether the declared serving contracts are present
and queryable in Candidate. Knowledge-query implementation, scoring,
serialization, schemas, public operations, workflow instructions, and all
unrelated cards and sources must remain unchanged.

### Evaluator identity

Candidate adds table-driven cases to the existing
`tests/test_knowledge_query.py` evaluator. The tests bind exact mechanism keys,
source references, content kind, status, nonclaims, claim-layer behavior, and
unbounded-version behavior. Existing registry-integrity, empty-query, and
information-architecture tests remain part of the evaluator.

After Candidate is fixed, the evaluator files must not change during the
assessment. The Evaluation Result will bind their commit identity.

### Environment and model identity

The evaluation is CPU-only on the current macOS host using the repository's
Python interpreter and standard library. The actual OS and Python identities
will be recorded with the result. It uses no GPU, driver, CUDA runtime, model,
profiler, Triton process, network request, or external service.

### Workload, correctness, budget, and repetitions

Run each of the following once on Candidate:

```text
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q skills/cuda-kernel-optimizer/scripts tests
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/cuda-kernel-optimizer
git diff --check
```

The workload is the public deterministic test suite. Correctness requires all
commands to return zero and the focused query assertions to preserve source,
scope, and nonclaim boundaries. No GPU or network operation is authorized.

### Expected outcome envelope

- Valid and supported: all commands return zero and every declared contract is
  returned with the expected source and boundary behavior.
- Valid rejection: a focused assertion fails because Candidate returns a wrong
  or over-broad contract.
- Invalid: the bound evaluator changes after Candidate is fixed, a required
  tool is unavailable, the wrong revision is tested, or a command uses GPU,
  network, or private material.
- Interrupted: an authorized command does not terminate; record it separately
  and do not infer success.
- Inconclusive: structural validation passes but the declared query behavior is
  not fully exercised.

### Claim ceiling

At most, the result can establish that Candidate structurally validates and
returns the declared source-reviewed contracts with the tested claim-layer,
version, source, and nonclaim boundaries. It cannot establish OCR correctness,
GPU performance, production applicability, framework-version compatibility,
configuration optimality, source completeness, or general model quality.

### Private material

The evaluator requires no private material. Private OCR experience explains
why this gap matters but is non-proving context and is not admitted into the
knowledge registry.

## Evaluation Result

The Evaluation Definition was frozen in `002b855` before Candidate content or
results were produced. Candidate and its unchanged evaluator were then fixed in
`1f51c42822d46594fa1817129062cb87aedc69b8`.

### Bound definition and actual identities

- Original: `2dc7514` (`v1.5.1`).
- Frozen definition: `002b855`.
- Candidate and evaluator: `1f51c42822d46594fa1817129062cb87aedc69b8`.
- Environment: Darwin 25.6.0 arm64, Python 3.9.6.
- Workload: the repository's deterministic CPU-only test and validation suite.
- GPU, network, model, profiler, and private-material use: none.

Candidate changed only the two knowledge registries and
`tests/test_knowledge_query.py`. It added nine reviewed public sources and nine
`source_reviewed` technical-contract cards. The registry now contains 92 cards
and 38 sources; `cards.json` is 187,813 bytes, below the existing 200,000-byte
limit. No query implementation, workflow instruction, schema, public operation,
or automatic decision changed.

### Actual result

- `python3 -m unittest discover -s tests -p 'test_*.py'` returned 0. All 260
  tests passed in 45.126 seconds; 2 were skipped.
- `python3 -m compileall -q skills/cuda-kernel-optimizer/scripts tests`
  returned 0 in 0.04 seconds.
- `python3 skills/cuda-kernel-optimizer/scripts/self_check.py` returned 0 in
  0.13 seconds with status `passed`, `gpu_checks_run: false`, and
  `network_checks_run: false`.
- `python3 /Users/tcheng/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/cuda-kernel-optimizer`
  returned 0 in 0.03 seconds and reported `Skill is valid!`.
- `git diff --check` returned 0.

The focused table-driven query evaluator returned each declared mechanism with
its expected public source, technical-contract status, nonclaim text, and a
`related` rather than `compatible` relation when component versions were not
enumerated by the reviewed material. A separate focused assertion returned an
`incompatible` relation when the serving-only dynamic-batching contract was
queried at the kernel claim layer.

### Validity, cost, terminal reason, and uncertainty

Every declared trial ran once and was valid. No trial failed, was interrupted,
or remained unrun. Evaluation used zero GPU time, no network access, and no
expensive external operation. It ended because every frozen command returned
zero and every declared focused behavior was covered by the full suite.

The result is structural and deterministic. It does not test the current
documentation against a deployed Triton version, execute a model, measure a
service, or show that the nine contracts are the complete knowledge needed for
the later OCR profile.

### Supported claim scope

The result supports only that Candidate structurally validates and can return
the nine declared source-reviewed contracts with the tested source, nonclaim,
claim-layer, and unbounded-version behavior. It does not establish OCR
correctness, GPU performance, production applicability, framework-version
compatibility, configuration optimality, source completeness, or general
optimization quality.

No private material was uploaded, evaluated, or used as public proof.

## Release Decision

The maintainer authorized publication in `v1.6.0` on 2026-09-04. The accepted
scope is limited to the source-reviewed contracts and query behavior established
above; the release makes no OCR correctness, compatibility, or GPU performance
claim. The release carriers are the [GitHub v1.6.0 release](https://github.com/troycheng/cuda-kernel-optimizer/releases/tag/v1.6.0)
and the internal `v1.6.0` tag.
