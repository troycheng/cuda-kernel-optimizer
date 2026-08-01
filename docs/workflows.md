# Optimization workflow

V1.4 has one model-led workflow. ChatGPT decides what to investigate and whether more work is worthwhile. Public tools execute one explicit operation and write facts; they do not call one another to plan the run.

## Durable model

| Record | Meaning | Created by |
|---|---|---|
| Target | Frozen objective, original Variant, workload, correctness, driver, environment, and validity requirements | readiness `check` |
| Variant | Immutable original or Candidate content | readiness or Experiment creation |
| Experiment | One candidate mechanism, mutation, falsifier, measurement design, and acceptance boundary | workload `experiment` |
| Invocation | One operation request, event stream, result, timeout, and cleanup outcome | the invoked tool |
| Champion Selection | Explicit record that a valid formal result became the current best Variant | champion `select` |
| Handoff | User-facing conclusion, retained changes, rejected directions, evidence gaps, and terminal reason | ChatGPT when pausing or finishing |

These machine records describe facts and explicit selections. The Handoff summarizes them for the user and later ChatGPT sessions; tools never read it. They do not store the current optimization stage or a machine-generated next action.

## Typical path

1. `check`: freeze the Target and exercise the original correctness smoke.
2. `baseline`: establish original performance before any candidate exists.
3. ChatGPT analyzes source and observations, then chooses the lowest-cost evidence that can distinguish the leading hypotheses.
4. `experiment`: freeze one Candidate and its measurement design.
5. `screen`: run correctness and the declared short paired screen. Independent low-cost falsifiers are completed before the Experiment is created; a conclusive failure blocks later expensive work.
6. A specific profiler `analyze` or `collect`: only when it answers an explicit unresolved question.
7. `target`: perform formal paired comparison with original or the current Champion.
8. `select`: explicitly record a passing Candidate as Champion.
9. `final_audit`: compare original with the current Champion before the strongest workload or serving claim.

`status` and `cancel` inspect or stop an Invocation. They do not resume an optimization plan; ChatGPT reads the completed evidence and decides what to do next.

## Different claim layers

- Kernel work uses runnable correctness checks and a stable kernel workload.
- Complete-workload work includes framework, CPU, transfer, communication, I/O, and environment effects.
- Serving work adds deployment identity, route coverage, traffic strata, queue/cache state, and environmental guards.
- Existing profiler artifacts support read-only diagnosis only within their recorded identity and field coverage.

See [Evidence and safety](evidence-and-safety.md) before adopting a result.
