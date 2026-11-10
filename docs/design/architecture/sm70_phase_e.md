# Phase E: maintained architecture and development workflow

Baseline: main `18784e02741b8610dbca8192d2045fdeec932600`, after D1–D6.
E changes documentation, offline tools and CI only. PLE follow-up, DDTree,
INT8 implementation and runtime/performance changes remain deferred.

| Delivery | Scope | Status |
| --- | --- | --- |
| E1 | Ownership map and attention/KV/MoE contracts; rework useful #1064 documentation from merged main | Merged in #1064 (`7ab8b4477`) |
| E2 | Source-derived reference, drift/link checks and correct source URLs | Merged in #1155 (`bef522e3c`) |
| E3 | Developer entry, coverage template and lightweight CI | This delivery, based on merged E2 main |

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

## E3 validation and maintained workflow

The [development guide](../../contributing/1cat-development.md) supplies six
recipes: codec, path/provider, model qualification, parameter, state/workspace
and native binding. Each links the existing owner and extension point, describes
compatibility/fallback duties, and names a minimal validation command. The tool
index distinguishes source declarations, selected-plan predictions, initialized
policy and observed native execution. The PR template records those evidence
levels and unverified conditions; documentation-only work can report N/A for GPU
tests with its reason.

Seventeen focused CPU tests pass, including filter coverage for the reader's
inputs and every maintained document. The same file scope is used by the two
new local hooks and the filtered `architecture-docs` workflow. The all-files CI
job delegates only those two hooks to that workflow. Existing ownership,
deprecation and layering checks retain their constraints and exclusions.

Both new hooks passed in their isolated pre-commit environments. The existing
layering check passed. The final API-disabled MkDocs build passed; the rendered
developer guide's six task anchors, architecture/reference navigation and 1Cat
source URLs were checked. Existing missing API cross-references, unavailable
submodule-source links in historical B reports, and older navigation/anchor
warnings are retained separately; none establishes a new runtime failure.

## Closure and retained boundaries

| Before E | Maintained result |
| --- | --- |
| Architecture entry predates B–D; development rules are scattered | Current ownership/navigation with six task recipes and links to the existing tools |
| Attention contract exists; KV and MoE contracts are missing | Three component contracts describe stages, numerical/resource boundaries, fallback and minimal checks |
| Source declarations need manual comparison with documentation | Deterministic source-derived reference with read-only drift, local-link and anchor checks |
| Source URLs use a hardcoded upstream repository | Site repository selects 1Cat source links; intentional upstream links remain |
| Review evidence mixes declared, selected and observed paths | Reference, tool index and PR coverage table identify the evidence type and gaps |

E adds documentation and offline validation only. It does not change inference
paths, defaults, numerical semantics, custom-op schemas or native ABI, and makes
no new performance claim. Runtime flow/coupling counts are therefore not an E
success metric. Historical reports and the migration log remain at their original
paths. Environment parameters continue to use the existing generated reference;
there is no additional runtime selector, configuration registry or parameter
parser.

PLE test adaptation and DDTree remain deferred. INT8-G64 remains an unimplemented
design. Full API documentation generation, external/historical link cleanup,
unsupported hardware topologies and model/35B throughput are outside this
delivery's validation. The documentation mock fix affects only no-Torch CLI
reference generation and does not resolve the deferred PLE runtime tests.
