---
name: cuda-kernel-optimizer
description: "Use when optimizing, tuning, diagnosing, or profiling CUDA, CUTLASS, Triton, PyTorch, vLLM, TensorRT-LLM, or another GPU workload; when assessing an NCU, Nsys, or PyTorch Profiler report; or when the test workload, correctness checks, measurement path, or target environment is incomplete."
---

# CUDA Kernel and Workload Optimizer

ChatGPT is the sole optimization decision maker. It identifies the bottleneck,
chooses the direction and candidate, judges ROI, and decides the next step.
The bundled tools perform deterministic checks, measurements, parsing, and
record keeping. Tools do not choose a direction, judge ROI, or propose the next
step.

Optimize the user's complete objective, not an isolated kernel metric. A kernel
measurement supports a kernel claim. A workload or serving claim requires the
user's original test workload and correctness or precision validation before
any benchmark, profiler run, or performance measurement. Never invent,
download, or silently substitute a test workload.

## Route

Load only the references needed for the current problem.

| Need | Tool or reference |
|---|---|
| Freeze the target and verify that the test workload, precision checks, driver, environment, and low-cost smoke test are usable | `scripts/readiness.py`; `references/environment_readiness.md` |
| Create a candidate, run the cheapest falsifier, check correctness, compare performance, or audit the selected result | `scripts/workload_evaluate.py`; `references/performance_iteration.md` |
| Parse or collect Nsight Compute facts | `scripts/profile_ncu.py`; `references/ncu_metrics_guide.md` |
| Parse or collect Nsight Systems facts | `scripts/profile_nsys.py` |
| Parse or collect PyTorch Profiler facts | `scripts/profile_pytorch.py` |
| Inspect frozen compiler stages or explicit binary output | `scripts/compiler_evidence.py` and `scripts/sass_check.py` |
| Query the bundled offline knowledge | `scripts/knowledge_query.py` |
| Inspect, select, or restore the recorded best variant | `scripts/champion.py` |
| Validate serving evidence | `references/serving_evidence_protocol.md` and `references/nonstationary_serving_evidence.md` |
| Check current primary sources or request an external challenge | `references/research_augmentation.md` |

Run each tool with `--help` before constructing its closed JSON request. Do not
load source files merely to discover CLI arguments.

## Optimization workflow

1. Confirm the user's objective, minimum useful effect, allowed files, risk,
   time, GPU use, and host-change boundary.
2. Run readiness. An optimization target must freeze the original variant,
   original test workload, correctness or precision rule, command driver,
   objective, environment identity, and validity requirements. A diagnostic
   target freezes existing reports and does not pretend that a workload is
   available.
3. Measure the original baseline before changing code. If correctness fails,
   do not accept performance samples.
4. Analyze source and existing observations. Consider kernel, launch,
   framework, CPU/data, transfer, communication, I/O, serving, and environment
   causes. ChatGPT keeps competing hypotheses and chooses the cheapest
   observation that can disprove the leading one.
5. Freeze one candidate as an Experiment before executing it. Run its declared
   falsifier, then correctness, then a short paired screen. Start a profiler
   only when it answers a specific unresolved question. Start formal paired
   measurement only after the earlier evidence remains valid.
6. Compare the measured effect and uncertainty with the user's minimum useful
   effect and the expected cost of the next action. Stop early when further
   work is not worthwhile. Continue a promising direction only within the
   user's allowed scope.
7. ChatGPT may explicitly select a candidate after a valid formal result. Run a
   final audit against the original variant before making the strongest
   workload or serving claim.

Each long command has its own timeout and process-group cleanup. A timeout is a
safety boundary, not a target duration. If the user allows unattended work,
ChatGPT continues only while the evidence, expected benefit, cost, and allowed
scope still justify the next action.

## Evidence rules

- Candidate and reference content must be immutable and explicitly bound to
  every result.
- Correctness failure blocks performance interpretation.
- Compare variants with paired samples from the same frozen test workload and
  environment identity.
- A profiler returns facts or observations. It never returns an optimization
  direction, ROI judgment, or next step.
- Knowledge lookup returns advisory facts or empty results. It never chooses a
  direction, judges ROI, or names the next step. Empty knowledge does not block
  source analysis, profiling, or a model-proposed hypothesis.
- Unknown profiler versions, fields, or units fail closed. Do not infer values
  from a nearby tool version or GPU architecture.
- `ERR_NVGPUCTRPERM` means NCU counters are unavailable under current host
  permissions. Record the limitation and use another valid evidence source;
  recommend host permission changes rather than applying them.
- Keep raw reports, paired samples, environment identity, rejected candidates,
  and the terminal reason needed to review the conclusion.

## External research

External search and external AI review are optional. Use them for primary-source
verification, difficult direction selection, a plateau, or final review. Redact
private material and preserve disagreement. Network failure does not stop local
work. Local evidence remains authoritative; correctness and measurement decide
whether a candidate is accepted.

This skill does not provide an OS sandbox. Modify only the files and isolated
environments authorized by the user. Host, driver, clocks, power, services,
container runtime, and GPU permission changes remain recommendations unless the
user explicitly authorizes them.
