# Knowledge, research, and external challenge

The skill uses three sources of information:

1. local facts: source, compiler output, profiler observations, raw samples, and workload KPIs;
2. bundled knowledge: identity-bound mechanism cards, primary-source records, compatibility notes, and detailed playbooks loaded only when matched;
3. external research: current primary documentation and optional independent AI challenge.

Only local correctness and measurement decide whether a Candidate passes.

## Offline knowledge

`knowledge_query.py` filters cards by exact GPU architecture, CUDA version, framework versions, claim layer, observed phenomena, and optional mechanism keys. It limits result count and UTF-8 context size. A matched detailed playbook is returned only as a digest-bound path so ChatGPT can load it on demand.

The query returns facts and falsification guidance, not an optimization direction or next action. An empty result is successful and does not block source analysis, profiling, or a new ChatGPT hypothesis. Historical gain numbers are not transferred to the current Target.

Sources record title, version, URL, verification date, status, and summary digest. Cards may cite only known verified sources. Unknown identities and mismatched versions fail closed rather than inheriting from a nearby architecture.

## External search

Use current official documentation, source repositories, specifications, and papers for version-sensitive claims. Record which local observation each source is meant to explain. Search results remain advisory until verified in the current environment.

## Independent AI challenge

External models are most useful for a major direction choice, an unexplained plateau, repeated rejection of one mechanism, or final review. Send a small redacted evidence packet and ask for counterarguments, the cheapest falsifier, possible measurement confounders, and primary sources.

External availability is optional. A request without a complete answer is recorded as failed review, not successful coverage. Preserve disagreement rather than turning answers into a vote. Private source, credentials, raw inputs, hostnames, and business data are not sent without explicit approval.

See the installed [research reference](https://github.com/troycheng/cuda-kernel-optimizer/blob/v1.4.1/skills/cuda-kernel-optimizer/references/research_augmentation.md) for provider order and review boundaries.
