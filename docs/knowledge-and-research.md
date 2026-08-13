# Knowledge, research, and external challenge

The skill uses three sources of information:

1. local facts: source, compiler output, profiler observations, raw samples, and workload KPIs;
2. bundled knowledge: technical contracts, optimization heuristics, practice cases, primary-source records, and detailed playbooks loaded only when matched;
3. external research: current primary documentation and optional independent AI challenge.

Only local correctness and measurement decide whether a Candidate passes.

## Offline knowledge

`knowledge_query.py` finds bounded material by observed phenomena or explicit mechanism keys, then reports whether each material is compatible with the supplied GPU, CUDA, framework, and claim-layer identity. Explicitly requested but incompatible material is returned with field-level mismatch reasons instead of disappearing. A contract with an unbounded component version is marked `related`, not `compatible`. A matched detailed playbook is returned only as a digest-bound path so ChatGPT can load it on demand.

Technical contracts report a primary-source proposition, applicability, unsupported inferences, and the decisions that depend on it. Heuristics can suggest a candidate or falsifying observation; cases remain bound to their recorded environment. The query never decides whether a premise is resolved, whether a mechanism is supported, or what to do next. An empty result is successful and does not block source analysis, profiling, or a new ChatGPT hypothesis. Historical gain numbers are not transferred to the current Target.

Sources record title, exact section locator, version, URL, review date, status, and summary digest. `reviewed` means the bundled statement was compared with that source; the digest detects later local edits but does not authenticate a remote page. A mismatch describes the bundled material, not platform support; ChatGPT must use matching primary documentation, implementation source, or local evidence before making a high-cost negative decision.

## External search

Use current official documentation, source repositories, specifications, and papers for version-sensitive claims. Record which local observation each source is meant to explain. Search results remain advisory until verified in the current environment.

## Independent AI challenge

External models are most useful for a major direction choice, an unexplained plateau, repeated rejection of one mechanism, or final review. Send a small redacted evidence packet and ask for counterarguments, the cheapest falsifier, possible measurement confounders, and primary sources.

External availability is optional. A request without a complete answer is recorded as failed review, not successful coverage. Preserve disagreement rather than turning answers into a vote. Private source, credentials, raw inputs, hostnames, and business data are not sent without explicit approval.

See the installed [research reference](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.4.2/skills/cuda-kernel-optimizer/references/research_augmentation.md) for provider order and review boundaries.
