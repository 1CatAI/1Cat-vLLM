# Phase E: maintained architecture and development workflow

Baseline: main `18784e02741b8610dbca8192d2045fdeec932600`, after D1–D6.
E changes documentation, offline tools and CI only. PLE follow-up, DDTree,
INT8 implementation and runtime/performance changes remain deferred.

| Delivery | Scope | Status |
| --- | --- | --- |
| E1 | Ownership map and attention/KV/MoE contracts; rework useful #1064 documentation from merged main | Merged in #1064 (`7ab8b4477`) |
| E2 | Source-derived reference, drift/link checks and correct source URLs | Implemented; validation below |
| E3 | Developer entry, coverage template and lightweight CI | Planned |

Historical A–D reports and the migration control log retain their paths.
Current facts belong to component contracts, source tables to the generated
reference, and measurements to their original report. Historical constraints
are not rewritten as current qualification.

## E1 validation

Review covers model/platform/runner boundaries, policy ownership, native
capabilities, resource lifetimes and compatible imports/counters against merged
source. Former #1064 traits assumptions are removed; INT8-G64 remains a proposal.
No runtime source, default, ABI or test oracle changes. Applicable Markdown,
pre-commit and local links are checked before merge; GPU/model tests are not
applicable to this documentation delivery.

D recorded five PLE regression failures. Later CPU diagnosis reproduced them on
C and passed the five original assertions with fixture-only adjustments, which
are not merged. PLE is deferred, not fixed or GPU-qualified by E.

E1 checks: all applicable pre-commit hooks passed, including existing runtime
parameter ownership and metadata gates. All local links in the six maintained
Markdown files resolve. Source scope is documentation only; no GPU allocation,
model loading or numerical/performance claim.

## E2 validation

The offline generator derives reachable configuration ownership with the D
reader and shares literal MoE binding extraction with the B inventory. It reads
KV and route declarations without importing vLLM, Torch, Triton or native code.
Its default/`--check` mode does not write; unknown declaration forms fail with an
error. It checks only the explicitly maintained E documents, including source
line references and local heading anchors. Existing environment documentation
and runtime selectors remain independent authoritative sources.

Fifteen focused CPU cases pass in a minimal documentation environment, including
read-only CLI behavior, deterministic output, declaration drift, unsupported
expressions, missing targets, duplicate-heading anchors and poisoned imports /
environment getters. Source URLs follow the configured repository and preserve
explicit upstream links. The documentation-only Torch mock now shares a real
Python `Module` base across both import styles, fixing the baseline CLI-doc
metaclass conflict without touching PLE or other runtime modules.

`API_AUTONAV_EXCLUDE=vllm` MkDocs build passed. Unrelated existing navigation and
missing-anchor messages (including `api/vllm`, pooling scoring and serving pages)
remain in the build log; they are outside E's maintained-document scope. No GPU,
model or runtime numerical tests are claimed for this tool/documentation change.
