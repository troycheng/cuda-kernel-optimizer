# Improving the project through real use

[简体中文](project-evolution.md)

## What this method solves

Real optimization work exposes problems in the project itself: an instruction may be unclear, a tool may accept invalid evidence, knowledge may be scoped too broadly, or an operation may cost more time than its value warrants. One case can reveal a problem, but it cannot automatically establish a general rule.

The project uses a simple process. After the task, a person decides whether the problem is safe and useful to preserve. A bounded Project Revision is then evaluated under conditions fixed before the result exists. The Evaluation Result records facts, and a maintainer makes the Release Decision.

## What it does not do

- The skill does not rewrite itself during optimization.
- The project does not collect or upload a private workload, internal trace, weights, proprietary source, or business data.
- A case does not enter bundled knowledge automatically.
- An Evaluation Result does not automatically merge or release a change.
- This maintenance process does not modify the V1.4 runtime.

V1.4 keeps the same responsibility split: ChatGPT makes optimization decisions, while deterministic tools perform one explicit operation and record facts. Project evolution starts after an optimization task ends and does not control its internal analysis or experiments.

## The six elements

| Element | Purpose | Repository representation |
|---|---|---|
| Evolution Contract | Holds boundaries that one evaluation cannot redefine | The approved [full design](maintenance/evidence-driven-self-evolution-design.md) |
| Case Snapshot | Preserves the original problem without leaking the historical answer into the challenge | `case-snapshot.md` |
| Project Revision | Identifies exactly which repository content changed | Git commit or tag |
| Evaluation Definition | Fixes comparison arms, environment, method, and claim ceiling before results exist | `evaluation-definition.md` |
| Evaluation Result | Records actual identities, trials, measurements, costs, and uncertainty | `evaluation-result.md` |
| Release Decision | Records a human decision to accept, reject, narrow, withdraw, or roll back | `release-decision.md` and the actual release carrier |

These elements are evidence records, not another execution system. None of them invokes the next one.

## Contribution flow

Start by locating the first broken link in the decision chain, not by deciding in advance to add knowledge, rules, or tooling. Record the decisive facts available before the incorrect decision, the knowledge or evidence actually used, and whether the correct conclusion was already derivable. Missing material is a knowledge gap; available material that was not retrieved is a routing problem; material that was used but led to the wrong conclusion is a reasoning problem. Invalid measurement, incorrect deterministic tool behavior, and genuine ambiguity remain separate cases. Record only the identities and summaries that affected the decision, not the complete context.

```text
A real task reveals a project problem
  → a person decides whether to create a safe public case
  → a bounded repository change is proposed
  → the Evaluation Definition is committed before results
  → an independent evaluation records facts
  → a human reviews the evidence and decides whether to release
```

Small, bounded code fixes can still use the normal pull request path. The full project-evolution material is intended for changes to ChatGPT behavior, knowledge scope, evaluation semantics, or public performance claims.

## Suitable contributions

- A public minimal reproduction of a deterministic tool defect;
- a compatibility boundary for a GPU, CUDA version, framework, or profiler;
- independently runnable tests, code, or documentation corrections;
- a performance case with a public workload, correctness checks, original baseline, and environment identity;
- a public synthetic case derived from private experience.

Once published, a case remains useful as an example and regression, but it is no longer a holdout. Repeated submissions from the same controlling source do not become independent evidence.

## Contributions derived from private work

Private material remains in the contributor's environment. A contributor may explain the kind of environment that revealed the issue and submit a public reproduction, test, code fix, or scope correction. The public change must still stand if the private validation statement is removed.

That statement can explain why the change was proposed. It cannot prove production performance, correctness, or generality. Community code and attachments are not run automatically on a maintainer's GPU.

## Required evidence

Deterministic tools and protocols use conformance evaluation. A real reproduction, valid and invalid inputs, fail-closed behavior, and a focused regression are normally enough for a narrowly scoped fix.

Instructions, knowledge, and ChatGPT decisions use behavioral evaluation. Keep the model, environment, and authorization fixed; use independent clean sessions, repeat when practical, and include a counterexample or holdout. Grade the semantic outcome rather than an exact historical command sequence.

Local evidence supports a local claim. A tool fix does not prove a performance gain, and one real task does not establish generality across GPU workloads.

## Templates and example

The four templates live in the repository's [`.github/evolution`](https://github.com/troycheng/cuda-kernel-optimizer/tree/main/.github/evolution) directory: `case-snapshot.md`, `evaluation-definition.md`, `evaluation-result.md`, and `release-decision.md`. The contribution surface follows the main repository and is not installed into a user's project with the skill.

The first public example is the [profiler evidence-object validation fix](evolution-case-profiler-evidence-validation.md). It shows a deterministic conformance replay and the limits of retrospective evidence.
