# CUDA Kernel Optimizer

`cuda-kernel-optimizer` helps ChatGPT optimize GPU workloads with a user-provided test workload, correctness checks, and paired measurements. ChatGPT makes optimization decisions; installed tools execute one explicit operation and record immutable facts.

Start with the [Chinese README](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.5.0/README.md) or [English README](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.5.0/README.en.md).

Install the current release in an environment that supports Skills CLI:

```bash
npx skills add https://github.com/troycheng/cuda-kernel-optimizer/tree/v1.5.0/skills/cuda-kernel-optimizer --skill cuda-kernel-optimizer
```

## Documentation

- [Getting started](getting-started.md): install the skill, prepare inputs, and run a short fit check.
- [Preparing a workload and environment](environment-readiness.md): establish the strongest claim the available setup can support.
- [Optimization workflow](workflows.md): understand Target, Experiment, Invocation, and Champion.
- [Long-running optimization](long-running-optimization.md): keep extended work efficient, observable, and within user authorization.
- [Evidence and safety](evidence-and-safety.md): review correctness, measurement, identity, and host boundaries.
- [Compatibility](compatibility.md): check platform and profiler requirements.
- [Validation records](validation.md): see what the project itself has tested.
- [Case studies](case-studies.md): review historical workload outcomes without treating them as predictions.
- [Knowledge and research](knowledge-and-research.md): understand offline knowledge, current-source search, and external challenge.
- [Project evolution](project-evolution.md): learn how real use becomes a bounded, evidence-backed project change.
- [Project evolution in English](project-evolution.en.md): read the English contribution guide.
- [Evolution case](evolution-case-profiler-evidence-validation.md): review a deterministic profiler-validation fix, not a performance example.
- AI execution protocol: read `skills/cuda-kernel-optimizer/SKILL.md` from the installed release, so the instructions match the installed code.

A result is usable only when its claim matches the supplied evidence. Source inspection supports a hypothesis; a stable kernel test supports a kernel claim; a representative workload supports an end-to-end claim; a controlled serving experiment supports a serving claim.
