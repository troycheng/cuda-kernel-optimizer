# Knowledge, search, and independent challenge

The optimizer uses three evidence layers:

1. **Local facts** — source, environment probes, compiler output, profiler data,
   raw benchmark samples, and workload KPIs. These decide whether a change wins.
2. **Bundled knowledge** — architecture capabilities, diagnostic mechanism cards,
   profiler guidance, compatibility notes, and identity-bound case memory.
3. **External research** — current primary documentation and optional independent
   model critiques. These may suggest an explanation or check, but remain advisory.

## Runtime knowledge context

During active diagnosis, the Controller builds
`active_diagnosis/knowledge_context.json` from the current:

- environment and workload identity;
- execution map and performance model;
- sealed semantic observations from diagnostic and active evidence;
- ready capabilities and allowed read-only actions;
- closed mechanisms and prior candidate history.

The engine applies exact identity, claim-layer, capability, action, and
observation rules before ranking. It returns at most three candidates, together
with explanations and explicit rejections. Each candidate names the cheapest
available check that could falsify it. The context always has
`promotion_authority: none`: it cannot authorize code changes, create a benefit
fact, or promote a candidate.

An empty knowledge result is not a finding that the workload has no useful
direction. If a raw profile identifies a qualifying execution layer but lacks a
mechanism-level observation, the Controller may allow one low-cost, read-only
measurement route. A neutral result remains neutral. The model may then propose
a concrete mechanism from the sealed profile, execution map, and source code.
A missing knowledge match does not block that proposal; the normal evidence and
scope checks still apply. If the execution map remains incomplete, its
unexplained scope is preserved rather than treated as resolved.

Historical speedup numbers are not runtime inputs. A locally measured case may
support or reject a mechanism only under its exact identity. A previous
identity-bound rejection prevents the same mechanism from consuming another
round, while an analogous case can only appear as an explanation.

Corrupted source records, invalid sealed evidence, or an identity mismatch
claimed as an exact local match fail closed. Missing identity, unavailable
tools, unsupported versions, absent semantic observations, or the lack of an
allowed read-only falsifier can remove a knowledge candidate, but do not forbid
the model from proposing a separately falsifiable direction. The engine does
not infer units from text, convert incompatible metrics, or treat a profiler
permission error as evidence against a mechanism.

The Controller-owned file is the authoritative runtime value.
`knowledge_query.py --frozen-input` is an inspection path for the same closed
input. The architecture-only CLI mode remains a reference catalog and is not a
substitute for current workload evidence.

## Offline package

The repository carries compact, machine-queryable knowledge rather than copies
of vendor manuals:

- source records identify a primary source, version, locator, verification date,
  and content digest;
- mechanism cards bind positive, counter, and invalidating observations to an
  existing read-only evidence action;
- case memory stores evidence and outcome identity, without transferable gain
  estimates.

A fixed UTF-8 byte limit keeps the complete catalog out of the model context.
Unknown architectures and version mismatches fail closed until local probes or
updated primary sources resolve them. Package validation checks cross-file
references, source status, action availability, and case identity without
network or GPU access.

## External search

When network access and policy allow it, the agent searches primary vendor
documentation, source repositories, specifications, and papers for
version-specific questions. It records the source and the local observation the
source is intended to explain. Search results do not bypass the local evidence
or action gates.

## Independent model challenge

External models are most useful for major direction choices, unexplained
plateaus, repeated failure of one mechanism, or review of a new compatibility
claim. They receive a small, redacted evidence packet and answer independently.
Their assumptions and proposed falsification tests are recorded; disagreement
is preserved rather than converted into a vote.

Private source, credentials, raw logs, inputs, and hostnames are not sent
outside the environment without explicit approval. External systems never run
the target, modify the repository, change the host, or promote a candidate. If
search or model providers are unavailable, the offline workflow continues.
