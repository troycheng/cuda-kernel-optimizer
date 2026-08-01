# Contributing

Open an issue before changing the V1.4 core model or adding a public operation.
Small fixes may go directly to a focused pull request.

Keep changes bounded, explain the failure or performance claim they address,
and add a regression test for changed behavior. Do not report a speedup without
the workload, baseline, correctness result, measurement method, and relevant
environment identity.

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
