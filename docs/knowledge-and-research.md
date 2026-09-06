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

For layout, synchronization or architecture-specific implementation questions, [KernelWiki](https://github.com/mit-han-lab/KernelWiki) can help locate upstream code. Read only relevant pages, then verify the owning source revision and license. Its SM90/SM100 focus is not evidence of SM120 or distributed-system support. The skill does not mirror the corpus or install a second optimizer; offline work continues with bundled material and deployed source.

External research serves two different decisions. Capability discovery checks whether an equivalent primitive already exists, whether the target framework and deployed version integrate it, and whether production-path evidence is still missing. Mechanism research checks version-sensitive semantics, competing explanations, and counterexamples. Maintained catalogs can help discover implementations, but availability and compatibility must be verified in the owning source repository, release, or change record.

Stop when the checked scope is sufficient to choose between reuse, a minimal backport, a narrow adapter, a measurement harness, or genuinely new implementation. Record only decision-changing, versioned facts in the existing Experiment premises; do not mirror a volatile community catalog into bundled knowledge. Code produced by an external optimizer remains an ordinary Candidate and must pass the same correctness and workload evaluation. If current upstream access is unavailable, continue from the deployed source and local evidence while stating the freshness limit instead of claiming that no equivalent implementation exists.

## Independent AI challenge

External models are most useful for a major direction choice, an unexplained plateau, repeated rejection of one mechanism, or final review. Send a small redacted evidence packet and ask for counterarguments, the cheapest falsifier, possible measurement confounders, and primary sources.

External availability is optional. A request without a complete answer is recorded as failed review, not successful coverage. Preserve disagreement rather than turning answers into a vote. Private source, credentials, raw inputs, hostnames, and business data are not sent without explicit approval.

See the installed [research reference](../skills/cuda-kernel-optimizer/references/research_augmentation.md) for source selection and review boundaries.
