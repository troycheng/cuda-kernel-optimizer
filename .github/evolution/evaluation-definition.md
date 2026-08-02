# Evaluation Definition

Commit this definition after the candidate content is fixed and before any evaluation result is produced. Replace the guidance with exact identities and conditions.

## Repository revision or evaluation arms

Identify each Git commit or content digest. State whether the comparison uses different revisions or one revision under different external conditions.

## Only intended comparison axis

Name the single intended difference. List every condition that must remain equal or paired.

## Evaluator identity

Bind the test, protocol, validator, fixture, and holdout by commit or digest. The candidate must not change this evaluator during the same assessment.

## Environment and model identity

Record the requested and actual model constraints, harness, OS, Python, GPU, driver, CUDA, framework, and profiler identities that matter. Mark unpinnable values explicitly.

## Workload, correctness, budget, and repetitions

Identify public workload and correctness material, or explain why the evaluation is deterministic and CPU-only. State the authorized resources and repeat count.

## Expected outcome envelope

Define valid, failed, interrupted, canceled, and inconclusive outcomes. Do not prescribe one optimization path when several paths could satisfy the task.

## Claim ceiling

State the strongest claim this evaluation could support before seeing its result. Local evidence must not become a general performance claim.

## Private material

Confirm that the evaluator does not require private material. If local private validation inspired the change, it remains non-proving context.
