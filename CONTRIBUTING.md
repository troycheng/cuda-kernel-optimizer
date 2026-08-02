# Contributing

Open an issue before changing the V1.4 core model or adding a public operation.
Small fixes may go directly to a focused pull request.

Keep changes bounded, explain the failure or performance claim they address,
and add a regression test for changed behavior. Do not report a speedup without
the workload, baseline, correctness result, measurement method, and relevant
environment identity.

## Project evolution

Small, bounded fixes do not need a project-evolution record. Use the
[project-evolution process](docs/project-evolution.en.md) when a change alters
ChatGPT behavior, bundled knowledge scope, evaluation semantics, or a public
performance claim.

For those changes:

- create a [Case Snapshot](.github/evolution/case-snapshot.md) from public,
  independently reviewable material;
- commit the [Evaluation Definition](.github/evolution/evaluation-definition.md)
  before producing the reported result;
- record facts and limits in the
  [Evaluation Result](.github/evolution/evaluation-result.md);
- leave the final [Release Decision](.github/evolution/release-decision.md) to
  a human maintainer.

Private experience may explain why a pull request was proposed, but the public
change must still stand when that statement is removed. Do not submit a private
workload, internal trace, weights, proprietary source, business image,
credential, or internal address. External code and attachments are reviewed
before any maintainer decides whether to run them; they are never executed
automatically on a self-hosted GPU.

An evaluation result does not admit knowledge, merge a pull request, or publish
a release. Its claim must remain no broader than the recorded public evidence.

Run the local checks before opening a pull request:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q skills/cuda-kernel-optimizer/scripts tests
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
```

Physical GPU tests are opt-in and are not required for documentation-only or
CPU-only changes. Host-level driver, counter, clock, power, service, and system
settings must remain recommendations unless a maintainer has explicitly approved
that environment change.
