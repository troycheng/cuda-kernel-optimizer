# V1.2 Evidence-Bounded Adaptive Investment Control

Status: approved for implementation on 2026-07-22.

## Goal

V1.2 turns the V1.1 diagnosis output into a closed investment loop. The
Controller repeatedly chooses the cheapest next action that can materially
change an optimization decision, assimilates the result, and recalculates the
remaining directions before starting more work.

The Controller must spend different amounts of effort on different directions:

- close a direction when local evidence proves its benefit ceiling is below the
  frozen minimum effect;
- give an uncertain high-upside direction one bounded discriminating check
  rather than killing it because of one noisy sample;
- invest further when local evidence becomes stronger;
- stop launching new work when the next action is not worth its cost or exceeds
  the user's authorization;
- never treat an authorization boundary as a performance failure.

V1.2 does not promise that an unfamiliar workload's globally best optimization
will be found. It makes the use of available evidence, time, GPU resources, and
user attention explicit and testable.

## Version boundary

V1.2 implements:

- an evidence-bounded direction portfolio;
- deterministic next-action selection;
- cumulative authorization checks before action launch;
- candidate-stage reevaluation after each result;
- one stable adapter for bundled, searched, and externally proposed knowledge;
- run-scoped online augmentation and GitHub Copilot provider support;
- integration with the existing V1.1 diagnosis and workload Controller.

V1.3 will concentrate on knowledge quality: broader mechanism coverage, better
retrieval, source refresh, real-case calibration, and reviewed knowledge
ingestion. V1.2 must not expand the bundled method catalog as a substitute for
the Controller.

## Existing foundations

V1.2 reuses rather than replaces:

- the frozen workload objective and minimum practical effect;
- the execution map and benefit ceilings;
- at most three competing hypotheses and their relationships;
- `inconclusive`, `plausible`, and `direction_supported` evidence states;
- evidence actions with Controller-owned cost, perturbation, risk, capability,
  and outcome declarations;
- identity-matched P50 and P90 action timings;
- paired confidence intervals and workload/service promotion gates;
- mechanism fingerprints, candidate history, and evidence lineage;
- the existing `MEASURE`, `PURSUE`, `REVIEW_REQUIRED`, and `STOP` decisions.

No success probability or historical speedup is converted into a current
workload expectation.

## Control loop

At every checkpoint the Controller performs the same sequence:

1. Validate that the objective, execution map, evidence, timings, and candidate
   outcomes belong to the current identities and analysis epoch.
2. Rebuild the live direction portfolio.
3. Generate locally falsifiable next actions for all live directions.
4. Remove inadmissible actions and actions whose declared outcomes cannot
   change a direction or run decision.
5. Apply cumulative authorization before launch.
6. Select one deterministic next action.
7. Execute it under a per-command timeout.
8. Bind its result to the current identities and ledger.
9. Return to step 1 before any later action starts.

Only external search and independent advisory reviewers may run concurrently.
GPU measurements and repository mutations remain serialized so their evidence
is attributable to one state.

## Direction portfolio

Each live direction carries:

- stable hypothesis and mechanism identity;
- claim layer and scoped execution-map nodes or uncovered interval;
- evidence state and supporting, opposing, and missing evidence;
- local benefit lower and upper bounds;
- the basis and identity binding for those bounds;
- the next uncertainty whose resolution can change a decision;
- eligible evidence or candidate actions;
- known dependencies, exclusions, and duplicate mechanism fingerprint;
- action cost range, perturbation, risk, and mutation scope;
- lifecycle state: `live`, `supported`, `falsified`, `stale`, or `closed`.

Directions are not evicted merely because another direction has a larger
ceiling. An alternative remains live until it is locally falsified, its bound
falls below the minimum effect, it is the same mechanism under a different
name, its evidence becomes stale, or no admissible action can resolve it.

## Benefit bounds

Before a candidate is measured:

- lower bound is zero;
- upper bound is the locally observed removable-time ceiling for the direction's
  scoped execution-map nodes;
- the upper bound is labelled a ceiling, never an expected gain;
- external benchmarks and method `typical_speedup` values cannot alter it.

After a representative short paired screen, the existing paired-statistics
confidence interval becomes the candidate's measured bound:

- upper bound below the frozen minimum effect closes the candidate;
- lower bound at or above the frozen minimum effect admits formal validation;
- an interval that still crosses the minimum effect is inconclusive, not a
  failure;
- one further bounded paired block may run automatically only when the frozen
  experiment design says it can narrow the interval, the sample cap has not
  been reached, and its P90 cost remains authorized;
- when no admissible measurement can resolve an inconclusive interval, the run
  ends with an explicit inconclusive terminal reason. It does not claim that
  the mechanism is ineffective.

Short screening must use a representative case already frozen in the workload
contract. It may reduce repetitions but must not silently substitute a smaller
shape, mock distribution, or different serving state that changes the
mechanism being tested.

Benefit bounds are bound to the execution-map and workload identities. A
dispatch change, relevant source change, workload change, or observed path
change marks the prior bounds stale. The Controller must refresh the execution
map before using stale bounds for another expensive action. It does not re-run a
global profile after every candidate unless identity/path evidence requires it.

## Decision-changing actions

An action is decision-changing only when its declared outcomes can produce at
least two different valid post-action states, for example:

- support versus oppose a live hypothesis;
- distinguish an exclusive pair;
- move a candidate from inconclusive to reject or formal validation;
- establish that a direction's ceiling is below the objective;
- invalidate or refresh an identity-bound result.

An action that only repeats already equivalent evidence, cannot affect a live
direction, or has no falsifying outcome is rejected before launch.

Evidence actions use deterministic dominance. Action A dominates B only when A
has no greater cost, perturbation, or risk; covers at least the same
decision-changing targets; provides no less independent evidence; and is
strictly better in at least one of those dimensions. Stable IDs break exact
ties. No model-provided information score is accepted.

Within the authorized action set, discriminating read-only checks run before
repository mutation. A candidate implementation is eligible only for a current
`direction_supported` hypothesis. If several implementation directions remain,
the Controller retains all non-dominated alternatives, tries the highest local
benefit ceiling first, then lower measured P90 cost, then stable identity. A
failed first implementation does not erase a distinct fallback direction.

## Authorization and time

The user's budget is an authorization envelope, not a target to consume and
not a performance verdict. The Controller tracks:

- cumulative measured elapsed time;
- cumulative GPU time where available;
- candidates and evidence actions already consumed;
- P50/P90 cost of the next action;
- risk, perturbation, and mutation scope;
- unresolved directions and their local benefit ceilings.

Before launch, the action's cumulative projected P90 cost must fit the current
authorization. This prevents a sequence of individually small actions from
silently consuming the entire allowance.

The outcomes are:

- `MEASURE`: execute one authorized decision-changing evidence action;
- `PURSUE`: execute one authorized candidate or validation action;
- `REVIEW_REQUIRED`: useful work remains, but the next material action exceeds
  authorized cumulative time, GPU, risk, perturbation, or mutation scope;
- `STOP`: no live qualifying direction or no admissible action remains;
- candidate promotion continues to use the existing local correctness and
  formal workload/service evidence.

When authorization is exhausted, no new action starts. The current diff,
evidence, portfolio, blocked action, projected cost, and continuation reason are
preserved. Authorization exhaustion is not recorded as `budget_expired`
performance failure.

Per-command timeouts remain hard safety controls and terminate the command's
whole process group. They protect against hung tools; they do not decide whether
an optimization direction is valuable.

Normal operation reports one aggregated `REVIEW_REQUIRED` state. Unattended
operation asks no question and stops with the same continuation artifact.

## Knowledge adapter

V1.2 defines one internal interface rather than a new schema family:

```python
recommend(context) -> list[KnowledgeCandidate]
```

The compact context includes architecture and software identity, workload
objective, observed execution layers, active hypotheses, missing evidence,
available tools, and authorized risk/scope.

A normalized knowledge candidate may provide:

- mechanism identity and applicability prerequisites;
- scoped execution nodes or an explicit unmodeled interval;
- competing explanations and a falsification question;
- a locally executable evidence action with declared outcomes;
- compatibility, side effects, invalidators, source, version, and freshness.

Knowledge can expand the hypothesis/action set. It cannot set local confidence,
change benefit bounds, promote a candidate, override failed correctness or
paired evidence, or authorize host changes.

An external suggestion enters as a shadow hypothesis. It is admitted only when
it can be mapped to observed nodes or an uncovered interval and to a local
falsifying action. Numeric gains supplied by an external source are discarded.
An unmapped suggestion remains advisory and cannot launch implementation.

Missing, stale, or unavailable knowledge degrades to the evidence-only path and
is recorded as `knowledge_support: unavailable`; it does not block optimization.

## Online search and external AI

When local knowledge is missing, stale, version-mismatched, or leaves a material
direction unresolved, V1.2 may retrieve current primary sources and request
independent model challenges. This produces run-scoped temporary knowledge, not
an automatic update to the bundled catalog.

Source order is official documentation, release notes and source repositories;
original papers; project issues and maintainer discussions; other technical
material; then uncited model statements.

External calls are triggered only for initial material direction selection, a
version/architecture knowledge gap, a clear plateau, or final review. Requests
are sanitized, digest-bound, deduplicated, executed independently and in
parallel, and bounded by a 180-second aggregate wait. Provider failure degrades
locally.

Default provider priority is:

1. Google AI Mode;
2. GitHub Copilot for repository/code questions;
3. GLM;
4. Kimi;
5. DeepSeek;
6. Gemini.

The review record distinguishes provider surface from the underlying model when
known. Copilot's value may be repository context rather than model diversity;
two surfaces using the same underlying model do not count as heterogeneous
reviewers.

External output is advisory. It may create a locally falsifiable shadow
hypothesis but cannot alter local benefit bounds or be counted as support
evidence. Full response, failed providers, model metadata when known, and total
wait are retained.

## Failure and recovery

- A failed correctness stage blocks every performance and profiler stage for
  that candidate.
- A short paired upper bound below the minimum effect blocks formal validation.
- A noisy interval crossing the threshold remains inconclusive until its
  declared measurement cap or authorization is reached.
- Evidence invalidated by identity or dependency change is marked stale rather
  than permanently killing the direction.
- Snapshot restoration failure remains `manual_recovery_required`.
- Repeated evaluation of the same checkpoint is idempotent and launches no new
  action.
- External provider failure or disagreement cannot reject a locally supported
  direction or promote an unsupported one.

## External design challenge

Google AI Mode, GitHub Copilot, and GLM challenged the design before this
specification was frozen. Accepted concerns are cumulative small-action budget
leakage, premature pruning after noisy evidence, stale bounds after path changes,
hidden evidence dependencies, representative short-screen identity, and
external knowledge contaminating local bounds.

Rejected suggestions include arbitrary fixed contraction percentages, fixed
cycle counts, mean-plus-two-standard-deviation intervals for tiny samples,
invented ten-percent bounds for external ideas, parallel GPU mutations, and
deferring formal workload validation or the online adapter beyond V1.2. Those
suggestions lacked local evidence or contradicted the approved scope.

## Acceptance tests

Automated tests must prove:

1. A locally falsified or below-threshold direction launches no later GPU stage.
2. A noisy short interval that still includes a qualifying effect is not killed
   after one sample; only a declared bounded follow-up may run.
3. A short-screen upper bound below the minimum effect launches no formal test.
4. A positive lower bound admits formal validation but does not itself promote.
5. A non-representative short-screen identity is rejected before measurement.
6. An action whose outcomes cannot change a decision is never launched.
7. The cheapest non-dominated discriminator is selected without a model score.
8. A distinct fallback direction survives another direction's implementation
   failure.
9. Fifty individually small actions cannot bypass cumulative authorization.
10. An action that would exceed cumulative P90 authorization returns
    `REVIEW_REQUIRED` before launch and names the blocked action and excess.
11. Authorization exhaustion is not reported as a performance rejection.
12. Identity/path change invalidates old bounds and blocks expensive reuse until
    refresh.
13. An external `+20%` claim cannot change a locally computed benefit bound.
14. An external suggestion without a node/interval and falsifying action cannot
    launch implementation.
15. A valid shadow hypothesis can request one local check but cannot become
    `direction_supported` from external text.
16. Search/reviewer requests are deduplicated, sanitized, bounded to 180 seconds,
    and provider failures fall back locally.
17. GitHub Copilot is ordered after Google AI Mode and records its underlying
    model as known, `auto`, or `unknown`.
18. Repeated checkpoint evaluation is idempotent.
19. Output contains elapsed time, cumulative spend, benefit bounds and basis,
    selected or blocked action, decision reason, skipped expensive actions, and
    the next feedback point.
20. When every non-duplicate direction is closed, the Controller returns `STOP`
    instead of proposing another round.

## Implementation constraint

Keep the decision logic in small pure modules with direct unit tests. The large
`workload_controller.py` may adapt inputs, execute the selected action, and
persist results, but it must not reimplement portfolio ranking or authorization
rules. Add no new JSON schema family and do not expand the bundled method
catalog in V1.2.
