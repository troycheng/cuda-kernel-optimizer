# V1.2 Adaptive Investment Control Implementation Plan

> **For AI agent workers:** Required sub-skill: use superpowers:executing-plans to implement this plan task by task. Track every step with the checkboxes below.

**Goal:** Make candidate evaluation stop unnecessary work early and return `REVIEW_REQUIRED`, rather than a false performance rejection, when the next bounded stage does not fit the remaining automatic authorization.

**Architecture:** Keep V1.1 diagnosis unchanged. Extend the existing `CandidateGate` to distinguish evidence failure from authorization admission, then teach `workload_controller.py` to persist one idempotent review-required terminal state while restoring an unverified candidate. Reuse current decision artifacts and tests; add no new schema family or deadline-extension API.

**Technical stack:** Python 3 standard library, `unittest`, existing JSON artifacts and process-group runner.

---

## File structure

- Modify `skills/cuda-kernel-optimizer/scripts/budget.py`: return a bounded
  `REVIEW_REQUIRED` admission result with the blocked stage.
- Modify `skills/cuda-kernel-optimizer/scripts/workload_controller.py`: persist
  review-required candidate outcomes, restore the snapshot, and make repeated
  evaluation idempotent.
- Modify `skills/cuda-kernel-optimizer/tests/test_time_gates.py`: installed-skill
  contract tests for admission and skipped stages.
- Modify `tests/test_budget.py`: focused unit test for the gate result shape.
- Modify `tests/test_workload_controller.py`: integration tests for expired and
  insufficient authorization, rollback, and repeated evaluation.
- Modify `skills/cuda-kernel-optimizer/SKILL.md`: document the exact time and
  interaction semantics after the implementation is verified.
- Modify `docs/long-running-optimization.md`: explain the review-required
  terminal state without claiming improved direction discovery.

### Task 1: Separate candidate evidence failure from authorization admission

**Files:**
- Modify: `skills/cuda-kernel-optimizer/tests/test_time_gates.py`
- Modify: `tests/test_budget.py`
- Modify: `skills/cuda-kernel-optimizer/scripts/budget.py`

- [ ] **Step 1: Add a failing installed-skill admission test**

Add a test that gives the gate enough time for static review and correctness,
but not enough for the declared short paired stage:

```python
def test_next_stage_outside_authorization_requests_review_without_starting_it(self):
    self.contract["hard_ceiling_seconds"] = 6.5
    calls = []
    result = self._gate().run(
        {
            "static_review": self.clock.action(
                calls, "static_review", {"status": "passed"}
            ),
            "build_correctness": self.clock.action(
                calls, "build_correctness", {"status": "passed"}
            ),
            "short_paired": self.clock.action(
                calls,
                "short_paired",
                {"status": "passed", "upper_bound": 2.0},
            ),
        }
    )
    self.assertEqual(calls, ["static_review", "build_correctness"])
    self.assertEqual(result["decision"], "REVIEW_REQUIRED")
    self.assertEqual(result["stop_reason"], "automatic_authorization_insufficient")
    self.assertEqual(result["next_stage"], "short_paired")
```

- [ ] **Step 2: Add the focused unit assertion**

In `tests/test_budget.py`, construct a short authorization window and assert
that the blocked action is not called and `skipped_expensive_stages` begins with
the blocked stage.

- [ ] **Step 3: Run the two new tests and verify failure**

Run:

```bash
python3 skills/cuda-kernel-optimizer/tests/test_time_gates.py \
  TimeGateTests.test_next_stage_outside_authorization_requests_review_without_starting_it
python3 -m unittest \
  tests.test_budget.CandidateGateTests.test_insufficient_authorization_returns_review_required
```

Expected: failures because `CandidateGate` currently returns `STOP` with
`hard_ceiling_admission_failed` and has no `next_stage` field.

- [ ] **Step 4: Implement the minimum gate change**

Extend `CandidateGate._result` with an optional `next_stage`. On an admission
failure before a stage, return:

```python
return self._result(
    started_at=started,
    decision="REVIEW_REQUIRED",
    stop_reason="automatic_authorization_insufficient",
    completed=completed,
    next_stage=stage,
)
```

Do not change evidence-failure `STOP` results, threshold checks, stage order, or
process-group timeout code.

- [ ] **Step 5: Run the complete budget and time-gate tests**

Run:

```bash
python3 -m unittest tests.test_budget
python3 skills/cuda-kernel-optimizer/tests/test_time_gates.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add skills/cuda-kernel-optimizer/scripts/budget.py \
  skills/cuda-kernel-optimizer/tests/test_time_gates.py tests/test_budget.py
git commit -m "feat: distinguish candidate authorization admission"
```

### Task 2: Persist an idempotent review-required candidate result

**Files:**
- Modify: `tests/test_workload_controller.py`
- Modify: `skills/cuda-kernel-optimizer/scripts/workload_controller.py`

- [ ] **Step 1: Replace the expired-budget rejection expectation**

Change the existing expired-after-edit test to require:

```python
self.assertEqual(decision["status"], "review_required")
self.assertEqual(decision["reason"], "automatic_authorization_expired")
self.assertTrue(decision["rolled_back"])
self.assertEqual(config.read_text("utf-8"), original)
self.assertEqual(
    self.controller.read_run_state(run_dir)["next_action"],
    "review_required",
)
```

- [ ] **Step 2: Add a next-stage admission integration test**

Set the frozen deadline so static review is allowed but the declared candidate
workload stage is not. Patch `evaluate_pairs`, call `evaluate_change`, and
assert:

```python
evaluator.evaluate_pairs.assert_not_called()
self.assertEqual(decision["status"], "review_required")
self.assertEqual(decision["next_stage"], "build_correctness")
self.assertIn("build_correctness", decision["skipped_expensive_stages"])
```

- [ ] **Step 3: Add an idempotency assertion**

Call `evaluate_change` a second time after the review-required result and assert
that it returns the exact same decision without calling workload evaluation or
review again.

- [ ] **Step 4: Run the new Controller tests and verify failure**

Run:

```bash
python3 -m unittest \
  tests.test_workload_controller.WorkloadRoundTests.test_expired_budget_after_edit_requests_review_and_restores_candidate \
  tests.test_workload_controller.WorkloadRoundTests.test_stage_outside_authorization_requests_review_before_launch \
  tests.test_workload_controller.WorkloadRoundTests.test_review_required_evaluation_is_idempotent
```

Expected: failures because expiration and gate admission currently call
`_finish_rejected` and mark the run completed.

- [ ] **Step 5: Implement `_finish_review_required`**

Add a bounded helper next to `_finish_rejected`. It must:

- restore the frozen snapshot;
- preserve `candidate.diff`, `time_gate.json`, and completed evidence;
- write the existing decision schema with `status="review_required"`,
  `rolled_back=True`, the reason, `next_stage`, elapsed time, stop reason, and
  skipped stages;
- write state with `status="review_required"`, `stage="decision"`, and
  `next_action="review_required"`;
- reuse the current manual-recovery behavior if snapshot restoration fails.

- [ ] **Step 6: Route both authorization paths to the helper**

In `_evaluate_change_unlocked`:

- treat an already persisted `review_required` decision like a completed
  decision for idempotent reads;
- map an already expired deadline to
  `automatic_authorization_expired`;
- map `CandidateGate`'s `REVIEW_REQUIRED` result to
  `_finish_review_required`;
- keep every evidence `STOP` routed to `_finish_rejected`;
- do not translate authorization admission into `budget_expired`.

Do not add an in-place authorization extension command.

- [ ] **Step 7: Keep advisory review from changing a local verdict**

Remove the post-review `budget_expired` rejection. `review_change` already has a
bounded deadline; its timeout or unavailability remains advisory and the local
formal paired verdict stays authoritative.

- [ ] **Step 8: Run the Controller time and candidate tests**

Run:

```bash
python3 -m unittest tests.test_workload_controller
```

Expected: all tests pass, including rollback and process-group timeout tests.

- [ ] **Step 9: Commit Task 2**

```bash
git add skills/cuda-kernel-optimizer/scripts/workload_controller.py \
  tests/test_workload_controller.py
git commit -m "feat: pause candidates outside automatic authorization"
```

### Task 3: Document the bounded behavior without expanding claims

**Files:**
- Modify: `skills/cuda-kernel-optimizer/SKILL.md`
- Modify: `docs/long-running-optimization.md`

- [ ] **Step 1: Update the skill contract**

Replace wording that makes the hard ceiling a performance outcome. State that
the Controller checks the next stage before launch, authorization exhaustion is
`REVIEW_REQUIRED`, normal operation reports one aggregated request, and
unattended operation stops without prompting.

- [ ] **Step 2: Update the long-run user documentation**

Describe the three distinct results:

- candidate `STOP`: evidence rejected the candidate;
- `REVIEW_REQUIRED`: more automatic time, GPU, risk, or scope is required;
- command timeout: the process group was terminated for safety.

Explicitly state that V1.2 does not prove better hypothesis generation.

- [ ] **Step 3: Run documentation and diff checks**

Run:

```bash
python3 -m unittest tests.test_skill_metadata tests.test_skill_eval
git diff --check
```

Expected: tests pass and `git diff --check` produces no output.

- [ ] **Step 4: Commit Task 3**

```bash
git add skills/cuda-kernel-optimizer/SKILL.md docs/long-running-optimization.md
git commit -m "docs: explain adaptive investment boundaries"
```

### Task 4: Full regression and release-claim audit

**Files:**
- Modify only if a test exposes an in-scope regression.

- [ ] **Step 1: Run the installed skill suite**

```bash
python3 -m unittest discover -s skills/cuda-kernel-optimizer/tests -p 'test_*.py'
```

Expected: all tests pass.

- [ ] **Step 2: Run the full local suite**

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected: all CPU/static tests pass; physical GPU opt-in tests may be skipped.

- [ ] **Step 3: Verify repository state**

```bash
git diff --check
git status --short
git log --oneline --decorate -5
```

Expected: no uncommitted implementation changes and the three feature commits
appear after the design and plan commits.

- [ ] **Step 4: Audit the claim boundary**

Search changed documentation and code for claims that V1.2 finds a better
direction, predicts success probability, or has been validated on an unfamiliar
business workload. Remove any such claim unless separate evidence exists.

- [ ] **Step 5: Request code review**

Review the complete branch diff against
`docs/superpowers/specs/2026-07-22-adaptive-investment-control-design.md`, then
fix only confirmed in-scope findings and rerun the affected tests.
