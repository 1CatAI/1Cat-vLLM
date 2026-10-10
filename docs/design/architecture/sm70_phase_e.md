# Phase E: maintained architecture and development workflow

Baseline: main `18784e02741b8610dbca8192d2045fdeec932600`, after D1–D6.
E changes documentation, offline tools and CI only. PLE follow-up, DDTree,
INT8 implementation and runtime/performance changes remain deferred.

| Delivery | Scope | Status |
| --- | --- | --- |
| E1 | Ownership map and attention/KV/MoE contracts; rework useful #1064 documentation from merged main | In review in #1064 |
| E2 | Source-derived reference, drift/link checks and correct source URLs | Planned |
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
