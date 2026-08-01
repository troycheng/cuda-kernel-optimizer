# RTX 5090 / SM120 opt-in acceptance

This lane validates the V1.4 public operation model on a physical RTX 5090. It is skipped unless `CUDA_SM120_E2E=1` is set.

## What it proves

- a real Triton command driver can pass readiness and freeze an exact SM120 Target;
- original baseline measurement precedes Candidate creation;
- a deliberately faster, correctness-preserving Candidate produces a valid one-pair `screen` signal, after which an explicit formal `target` comparison confirms the win;
- only an explicit `champion select` records the Candidate as current best;
- `final_audit` rechecks original against the selected Champion;
- retained Iter0 service evidence is read without modification; current paired statistics place both point estimates below the configured threshold but classify both results as `inconclusive`, while the archive retains its historical `REJECT`, `REJECT`, and `STOP` fields;
- the normal lane does not add profiling capabilities or modify driver, clock, service, power, or counter policy.

The retained replay checks current statistical interpretation and historical decision fields separately; it does not use the current statistic to retroactively validate the archived decisions. The real Triton fixture checks the V1.4 execution and record path, not mechanism discovery or expected speedup for another project.

The one-pair diagnostic screen is expected to remain `inconclusive`: it is too small to confirm a win. The acceptance test explicitly starts `target` only after checking that the valid observed signal exceeds the configured minimum effect. This models a ChatGPT decision between two separate operations; the runner does not promote the Candidate or start the next stage itself.

## Local CPU helper checks

```bash
python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v
```

Physical tests are reported as skipped without opt-in.

## Isolated container lane

`remote/run_lane.sh` accepts `current` or `compat`. It requires:

- `CUDA_E2E_ROOT`: an isolated root containing this repository;
- `CUDA_E2E_ARTIFACTS`: a fresh directory below `$CUDA_E2E_ROOT/artifacts`;
- `CUDA_E2E_GPU`: an idle GPU index;
- `CUDA_V14_HANDOFF_ROOT`: the retained Iter0 replay root containing `blind-run-summary.json` and both Candidate results.

Example:

```bash
CUDA_E2E_GPU=2 \
CUDA_E2E_ROOT=/srv/cuda-skill-v14 \
CUDA_E2E_ARTIFACTS=/srv/cuda-skill-v14/artifacts/acceptance \
CUDA_V14_HANDOFF_ROOT=/srv/replay/iter0 \
tests/gpu/sm120/remote/run_lane.sh compat
```

The runner resolves the image reference to one immutable image ID, checks the selected GPU for compute processes, drops Linux capabilities, disables networking, and mounts the repository and retained replay read-only. Only the artifact directory is writable.

Expected V1.4 artifacts include:

```text
artifacts/acceptance/
├── container-image.json
└── v14-real-triton/artifacts/
    ├── target.json
    ├── experiments/<experiment-id>.json
    ├── invocations/<invocation-id>/
    │   ├── request.json
    │   ├── events.jsonl
    │   └── result.json
    └── champion/
        ├── current.json
        └── selections/<selection-id>.json
```

## Optional NCU permission smoke

The normal lane accepts `ERR_NVGPUCTRPERM` as a recorded capability limit. When the operator separately authorizes a disposable `SYS_ADMIN` container, `remote/run_ncu_authorized_smoke.sh` may be used on an idle GPU. It drops all other capabilities, disables networking, removes the container on exit, and does not change the host driver or `RmProfilingAdminOnly` policy.
