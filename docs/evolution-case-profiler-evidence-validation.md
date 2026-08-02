# Profiler evidence-object validation

This is a retrospective conformance case, not a performance case. It checks one deterministic validation change and makes no GPU speed, workload, or general capability claim.

## Case Snapshot

Candidate profiler collection must use a candidate Variant and a correctness evidence object whose manifest and payload still match the immutable object identity. Invalid inputs must be rejected before a profiler command starts.

The case uses only public repository history. It contains no private workload, trace, weights, source, GPU environment, or internal address.

## Project Revision

- Original: `5211e832b6d5055ed316fe6fc77efa57813f5934`
- Revision: `9a3ff596907fcab7dd9abf4615bb080a1a2c2222`
- Historical release carrier: `v1.4.0`

The revision added two relevant checks: an Experiment candidate must have the `candidate` role, and an evidence object must be materialized through the object store so that its payload is verified.

## Evaluation Definition

The definition was frozen in commit `9c8cdaf6f24dbeccc47526288eae95bc879ff4f6` before this replay. Both repository revisions used the same Python process and the test file from revision `9a3ff596907fcab7dd9abf4615bb080a1a2c2222`.

The focused tests were:

- `test_candidate_collection_rejects_non_candidate_experiment_role`
- `test_candidate_collection_rejects_changed_evidence_payload`

No GPU or network access was used. The evaluator changed only the test file in the original detached worktree; its production source remained at the original commit.

## Evaluation Result

- Original: both focused tests failed because `ValueError` was not raised.
- Revision: both focused tests passed.
- Revision regression: all 10 tests in `tests.test_workload_adapter` passed.
- Python: 3.9.6.

The result was recorded in commit `7c22f783e8f552943c3ddb1c64a82b41c455358d`. The complete commands, observed return codes, cleanup result, and evidence limits are in the [maintenance record](maintenance/evolution-pilot-profiler-evidence.md).

## Release Decision and claim limit

The code change was already included in `v1.4.0`; this replay does not create a new release. It supports only the claim that these two public invalid inputs are rejected by the revision and were not rejected by the original.

Because the evaluation is retrospective, it does not show that the historical fix was preregistered. It also does not establish performance, coverage of every evidence-object shape, behavior under a different model, or general optimization quality.
