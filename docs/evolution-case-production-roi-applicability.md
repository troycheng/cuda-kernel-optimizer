# Production ROI Applicability

## Case Snapshot

GitHub [Issue #19](https://github.com/troycheng/cuda-kernel-optimizer/issues/19) records two forms of the same failure. A narrow Candidate inherited savings from components it did not change, and an eager CUDA-Graph proxy was later treated as the reference cost of an Inductor-lowered production region. The latter changed the apparent reference from 49.1 us per layer to the applicable 2.899 us. With a 2.316 us Candidate, ten occurrences, and a 4.314 ms replay, the production opportunity was about +0.135%, below the +0.5% Target.

The required behavior is not a new production-path check. ROI must follow the existing V1.5 rule that a claim can use only evidence applicable to that claim. Public issue data and synthetic scenarios are sufficient to evaluate the behavior; no private workload, trace, source, address, credential, or model weight is included.

## Project Revision

The revision treats ROI as a derived evidence claim. It binds opportunity inputs to the Candidate's actual replacement boundary and execution form, limits savings to Candidate scope and critical-path exposure, preserves exact measurements and justified conservative upper bounds, and requires prediction-error reconciliation before another Candidate is ranked.

The revision changes instructions, references, public documentation, and behavior tests. It adds no production module, public operation, protocol object, automatic decision path, fixed profiler stage, or GPU performance claim.

## Evaluation Definition

This definition is committed with the fixed Candidate content and before the reported evaluation result.

- **Repository arms:** `origin/main` is the Original; the commit containing this definition is the Candidate.
- **Only intended axis:** how ChatGPT establishes and corrects a Candidate's production ROI claim. Existing execution tools and schemas remain unchanged.
- **Deterministic evaluator:** the repository's complete CPU/static test suite, bytecode compilation, package self-check, and diff validation.
- **Behavior evaluator:** one fresh read-only `gpt-5.4` session, using a clean temporary `CODEX_HOME` and the Candidate skill's absolute paths, over five public scenarios: eager versus lowered timing, W2-only scope, dispatch mismatch, a conservative production bound, and a prediction-versus-target conflict. The model is pinned to a version supported by the ChatGPT-account Codex CLI; no previously installed copy of the skill may be loaded.
- **Environment:** macOS arm64, Python 3, ChatGPT bundled Codex CLI 0.148.0-alpha.15; no GPU or network evidence is required.
- **Budget:** one complete deterministic run and one behavior session. Repetition is unnecessary for deterministic checks; the model result is a bounded behavior observation, not a statistical capability estimate.
- **Valid outcome:** all deterministic checks pass and the behavior session applies the five required invariants without inventing missing measurements.
- **Failed outcome:** any deterministic failure, use of mismatched evidence in production ROI, inherited out-of-scope savings, or switching Candidate before reconciling a material prediction error.
- **Claim ceiling:** the result can support only that this revision states and exhibits the required behavior in the five scenarios. It cannot establish GPU speedup, universal model compliance, or correctness on an undisclosed workload.
- **Private material:** none is required or admitted as evidence.

## Evaluation Result

Pending execution against the committed Candidate.

## Release Decision

Pending the recorded evaluation result and maintainer review.
