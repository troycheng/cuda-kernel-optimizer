# V1.2 Adaptive Investment Control

Status: approved for implementation on 2026-07-22.

## Purpose

V1.2 changes how the Controller spends time after V1.1 has produced a supported
direction. It decides whether the next candidate stage is worth starting and
authorized to run. It does not improve hypothesis generation, predict a
candidate's probability of success, or claim that an unfamiliar workload's
best direction will be found.

The feature must prevent an inconclusive or expired run from being reported as
a performance failure. It must also avoid repeated user prompts: one normal
authorization request at most, and no routine prompt in unattended operation.

## Existing inputs

The implementation consumes existing artifacts and contracts:

- the user's selected budget or custom automatic-run allowance;
- the frozen workload objective and minimum effect;
- `performance_model.json`, `decision.json`, and `investment_brief.json` from
  active diagnosis;
- the candidate declaration and its stage cost bounds;
- measured correctness, short paired, profiler, and formal paired outcomes.

No new ROI model or probability score is introduced. Before a candidate exists,
the supported benefit ceiling is an upper bound, not an expected gain. After a
short paired screen, measured confidence bounds become the performance signal.

## Decision boundary

The Controller continues to expose the existing diagnosis decisions:
`MEASURE`, `PURSUE`, `REVIEW_REQUIRED`, and `STOP`. Candidate evaluation also
returns `PROMOTE` after every required stage passes.

At each candidate-stage boundary the Controller applies these rules in order:

1. A prior failed stage returns `STOP` for that candidate and blocks every later
   stage.
2. A short paired upper bound below the frozen minimum effect returns `STOP` for
   that candidate.
3. A profiler runs only when it answers a live uncertainty; otherwise it is
   explicitly `not_applicable`.
4. If the next stage has no bounded cost or cannot fit inside the remaining
   automatic-run allowance, the Controller returns `REVIEW_REQUIRED` before
   launching it.
5. Otherwise the Controller starts exactly the next stage and reevaluates after
   its outcome.
6. After all required stages pass, the candidate is eligible for promotion.

The soft target remains guidance. Reaching it does not change a direction or
candidate decision. The automatic-run allowance is not a target to consume.

## Time and authorization semantics

Per-command timeouts remain hard safety controls and continue to terminate the
whole process group. The run deadline remains a finite automatic-authorization
boundary. Normal admission requires the declared worst-case stage cost to fit
inside the remaining allowance, so a stage should not normally be interrupted
by the run boundary.

An expired allowance or a stage that does not fit produces
`REVIEW_REQUIRED`, not `STOP`, `budget_expired`, or a rejected performance
verdict. The Controller restores an unverified candidate snapshot for safety,
preserves the candidate diff and evidence, records the next blocked stage, and
returns a stable review-required result on repeated status or evaluation calls.
Continuation requires a newly authorized run; V1.2 does not add an in-place
deadline mutation API.

Advisory external review is not a promotion authority. Provider failure or the
end of its allotted wait does not turn a locally confirmed workload result into
a rejection.

## Interaction policy

The Controller emits one review-required terminal state containing all known
authorization issues, the blocked next stage, elapsed time, and skipped
expensive stages. The skill reports that state once in normal operation. In an
unattended run it stops cleanly and leaves a continuation report without asking a
question.

Routine correctness failures, profiler degradation, external-review failure,
and candidate rejection are results, not authorization questions.

## Scope

Implementation is limited to:

- `skills/cuda-kernel-optimizer/scripts/budget.py`: distinguish evidence stops
  from authorization admission and expose the blocked next stage;
- `skills/cuda-kernel-optimizer/scripts/workload_controller.py`: preserve
  `REVIEW_REQUIRED` semantics, restore unverified changes safely, and keep the
  result idempotent;
- the existing budget, workload-controller, and installed-skill time-gate
  tests;
- concise skill and workflow wording needed to describe the changed behavior.

V1.2 does not rebuild V1.1, add a knowledge base, add online learning, add a new
schema family, rewrite `orchestrate.py`, or expand external-provider support.

## Failure handling

- Static, correctness, short-screen, profiler, formal, or service evidence
  failures remain candidate `STOP` results and restore the frozen snapshot.
- Authorization exhaustion returns `REVIEW_REQUIRED` and restores the frozen
  snapshot without calling the candidate a performance failure.
- Snapshot restoration failure returns `manual_recovery_required` and never
  deletes or promotes the uncertain working copy.
- Missing, stale, or non-finite evidence fails closed.
- A command timeout still kills its process group and records a terminal reason.

## Acceptance

Automated tests must prove:

- static falsification launches no GPU benchmark;
- correctness failure launches no profiler;
- a short-screen upper bound below the threshold launches no profiler or formal
  paired test;
- a profiler with no live question is not executed;
- a next stage that cannot fit returns `REVIEW_REQUIRED`, names the stage, and
  does not launch it;
- an already expired automatic allowance returns `REVIEW_REQUIRED`, not a
  rejected performance result;
- repeated evaluation of a review-required run returns the same decision and
  does not execute more work or create another authorization request;
- conclusive results exit before the allowance is consumed;
- output retains `elapsed_seconds`, `stop_reason`, and
  `skipped_expensive_stages`;
- process-group timeout behavior and all existing V1.1 evidence gates remain
  unchanged.

The release claim is limited to adaptive stage admission and reduced needless
execution. Finding the correct direction on an unfamiliar real workload remains
an unproven capability boundary until separate real-case evidence exists.
