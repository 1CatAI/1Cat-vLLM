# Flash-V100 Phase A3 execution record

Baseline: #1060, `8c96e32e56c09d4a3e3112cb5d1a367571f69476`.
Updated scope (2026-10-08): complete A3 autonomously. The user subsequently
authorized the executor to merge after self-review and every required gate.
No merge is authorized by a passing smoke alone. GPU validation uses the
explicitly authorized `dx.1catai.com:54633`, with the whole-group and per-GPU
locks, `VLLM_NO_USAGE_STATS=1`, and task-owned artifacts/dependencies.

| Step | PR | Status | Metrics | GPU validation | Open items |
| --- | --- | --- | --- | --- | --- |
| 0: scope and codec ownership | — | Decision communicated; #1028 rebased locally | Baseline measured | Runtime parity belongs to 1c | Retest each subsequent step |
| 1a: immutable CPU trace and owner guard | #1071 | Gates passed; ready | Production unchanged | 1667 passed / 7 inherited failures; no changed outcomes | Merge with prerequisite stack |
| 1b: patch efficacy + dependency ratchet | #1072 | Gates passed; ready | 14 cycles / 32 forbidden edges frozen | 1668 passed / same 7 failures; 41 patch names consumed | Merge with prerequisite stack |
| 1c: route/token/output parity tools | #1073 | Draft; model records pending | Production unchanged | 12 native cases exact; host baseline unit tests 135 passed | Greedy model records and full outcome map |
| 2a: dynamic environment boundary | #1075 | Draft; focused CPU and rebase passed | Outside-config reads 119 → 41; env ratchet 334 → 306 | Pending complete outcome map | Prerequisite model and outcome gates |
| 2b: frozen construction policy | — | Focused CPU and pre-commit passed | Remaining 41 → 0; 41 immutable fields; env ratchet 306 → 284 | Pending complete outcome map | Commit/rebase and prerequisite gates |
| 3: owned workspaces | — | Not started | — | Required | Step 2 gates |
| 4: decode executor and ordered candidates | — | Not started | — | Required | Step 3 gates |
| 5a: per-sequence prefill candidates | — | Not started | — | Required | Step 4 gates |
| 5b: batch prefill candidates | — | Not started | — | Required | Step 5a gates |
| 5c: debug observer | — | Not started | — | Required | Step 5b gates |
| 6: registered speculative features | — | Not started | — | Required | Step 5c gates |
| 7: final boundaries, flags, docs, ratchet | — | Not started | — | Final greedy evidence required | Step 6 gates |

## Step 0 decisions

- #1061, #1063, #1064, #1065 and #1066 are frozen. No work is stacked on them.
- `vllm/v1/attention/kv_codecs.py` (#1049) is the sole Python codec API, by
  the user's explicit decision. #1048's useful CUDA readers and storage
  accounting will be adapted to it later, preserving rounding and scale
  semantics. This does not introduce a second Python registry. The decision
  was [communicated on #1048](https://github.com/1CatAI/1Cat-vLLM/pull/1048#issuecomment-6060298392);
  no reply or owner agreement is claimed.
- #1028 (`17fa23e576fd12d2b3ce527558539058ab3024e6`) was replayed onto #1060
  in the owned `agent/v100-arch-a3-1028-integration-20261008` worktree. Three
  historical replay conflicts required preserving both host-KV and SM70
  additions. The final validation commit `dca435d4398168e1c846f88a97b7c14c8bf48a91`
  has exactly the clean merge-tree result `6f1c0f15840aebb104a2b5f3be21c6feab3e1096`.
  This temporary branch is not published or merged into #1028. Its host-KV,
  QSA-cache and KV-cache tests returned 37 passed / 98 skipped on CPU;
  GPU runtime parity remains a separate Step 1c requirement.
- #1048 advanced to `8794f126271360e7eae9eb281e776e742aefa12d` during this
  work. Its current diff overlaps the #1060 stack in five files: the migration
  control document, `sm70_qwen38_qk_rope.py`, the compatibility module, package
  initializer and metadata. Step 1 has no production-file overlap. This is a
  fresh comparison, not the earlier review's nine-file snapshot.
- Step 1 changes tools, tests and documentation only. Existing `qsa.py`
  direct-native calls, route names and public KV-layout signatures are frozen.

## Metric definitions

The Step 1b companion `python -m tools.sm70.flash_v100_audit` measures the complete package, including
function-local imports and speculative modules. Function length includes the
signature and docstring. Private references count attributes accessed through
an imported package module; cycles count directed simple cycles. Model hits
count occurrences in AST-normalized code outside `spec/`, excluding the
`ROUTE_SPECS` assignment. Environment reads include typed `envs` attributes and
raw `os` reads; captured versus dynamic classification records the current
read location, not a proposed behavior change.

| Metric | Reviewer estimate | Measured #1060 | Step 1 |
| --- | ---: | ---: | ---: |
| `forward` lines | 597 | 597 | unchanged |
| Largest function | 977 | 977 | unchanged |
| Cross-module private references | 380 | 390 | unchanged |
| Import cycles | present | 14 | unchanged |
| Model-name occurrences outside spec | ~150 | 169 | unchanged |
| Environment reads outside config | ~120 | 119 | unchanged |
| `state.py` boolean flags | 30 | 29 | unchanged |
| Repository ratchet model/platform/env | 2325/3964/~334 | 2325/3964/334 | unchanged |

The differences from the reviewer's estimates are measurement definitions,
not reductions. Step 1 is the explicitly requested production-code-free
safety net; it cannot claim a coupling reduction. Step 2 onward must show
actual reductions and preserve the immutable behavioral golden.

## Step 1 safety-net evidence

The CPU recorder executes real admission, layout, metadata helpers and
`forward`; only device allocation and native/Triton boundaries are replaced.
The matrix contains 768 Cartesian cases plus 45 targeted cases (813 total).
It covers competing candidates, native attempts returning `None`, gather
side effects, grouped verification, shared/legacy native ABI, post-construction
environment changes and persistent-buffer identities across two calls.
Zero-filled recorder outputs prove control flow and write destinations only.

`coverage run --branch` on #1060 hit every return in `forward` and
`_flash_v100_prefill_with_prefix`, and all three decode-cache reset sites.
The required-site missing list is empty. Overall statement/branch coverage
is lower because diagnostic and error paths remain; the checked-in coverage
fixture reports those totals without claiming complete branch coverage.

The strict shim rejects ownerless writes/deletes. The Step 1b opt-in pytest audit
observes real production calls without wrapping replacement identities;
ordinary functions must be called, while flags, exported classes and cached
objects must be consumed. The companion fixes four existing tests to exercise production paths
instead of relying only on export identity or overwritten cache state.
Dependency ceilings include existing forbidden edges as well as cycles;
new violations are rejected before the subsequent extraction steps.

Retained artifacts are under `/home/ymzx/arch-ws/tmp/a3-*` locally and
`~/arch-ws/architecture-gpu-54633-20261008/a3-step1/` on the authorized host.
The remote parent was verified against every tracked #1060 source hash before
building Flash-V100 with CUDA 12.8 / Torch 2.10.0+cu128. The task-owned native
extension is shared by parent/candidate; unchanged vLLM/FA2 extensions reuse
the declared runtime. This is source-pinned refactor validation, not evidence
of a newly built distributable wheel. Full parent/candidate results are pending.

The initial combined safety-net attempt repeatedly exposed unused shim patches.
It was returned to #1060 and divided into owned trace and audit PRs under the
plan's retry/split rule; neither permits starting Step 2 alone. The immutable
golden was retained. No production-code or numerical repair was folded in.

Trace PR local gates: `pytest -q --import-mode=importlib` on
`test_flash_v100_trace_golden.py` and `test_flash_v100_compatibility.py`
returned 16 passed (813 trace cases grouped into seven batches).
Pre-commit, including mypy, and `tools/pre_commit/check_layering.py` passed.
The first remote run stopped at a network configuration lookup after 1360
passes and the documented shared-ABI failure; it is not accepted as a full
run. The replacement uses the same task-owned offline model-config cache
for parent, trace and audit trees, with all original 54 files plus required
compatibility/composition suites and shared-ABI tests (60 files total).

Step 1b adds the actual import-edge/cycle ratchet and a per-name shim-use
report. Function replacement requires a production call, including callable
objects; reading its identity cannot satisfy the gate. State/export patches
require production reads. The audit retains exact patched test IDs and
call/read locations. No production import edges or environment reads change.

The final strict CPU run returned 219 passed / 1 skipped / 28 deselected,
with 40 patched names consumed (16 by calls, 24 by state/export reads).
The deselected metadata-builder cases require the GPU run; they are not
claimed as passing CPU tests. Same-code closures are attributed using the
caller's observed lookup, so one alias cannot satisfy another name's gate.
Current package ceilings are 14 cycles and 32 forbidden import edges.
The generated fixture includes all 119 env reads and their current owners;
39 reads in the implementation constructor and two in the metadata builder
are captured; the other 78 reads are classified as dynamic by current location.

## Completed Step 1a/1b gates

The complete 60-file run returned 1656 passed / 7 failed at #1060, 1667 / 7
at #1071, and 1668 / 7 at #1072. Both comparisons have an empty changed-outcome
map and all new tests pass. The seven failures are documented baseline issues.
The final strict GPU report consumed 41 names with no unused patches.
Evidence: local `a3-outcome-parity.json`, `a3-gpu-shim.json`; remote
`a3-step1/logs/outcome-parity.json`, `candidate-files-verified.log`.

PR #1028 was rebased independently onto each safety-net commit. For #1071 the
validation head is `ee51e3c447c8ff369a996161094cdaa9556456ee`, for #1072 it is
`f0b12ea9da153e339cf8cc3e20725af19b9582f0`. Both trees exactly match their
respective clean merge results; each host-KV/QSA CPU run again reports
37 passed / 98 skipped. GitHub pre-commit checks are green for both PRs.

## Step 1c tools

`route_parity.py` records fixed prompts, greedy token IDs, startup/request
route counters, native hashes and actual host-FP8 QSA epochs/statistics.
`op_parity.py` runs native forward with eager/graph fixtures and requires
finite outputs with maximum absolute error zero. The separate performance
gate requires both XQA and the 75T prefill workload and rejects changes
outside +/-2%; this is not a claim of 75 TFLOP/s throughput.
Both tools reject mismatched workload/runtime contracts or empty evidence.
Model/tokenizer files are hash-verified before generation. The small Qwen
fixture is Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`;
weights live only in task-owned remote artifacts. No GPU parity or final A3
completion is claimed until the pending records and comparisons finish.

## Step 1c initial GPU evidence

On 54633, all 12 attention cases at source `2b4532127c4c90e93630225beafcd0ecd32882d2`
match exact #1060: finite outputs, zero maximum absolute error, equal route
counts and stable current-owner buffer pointers, including metadata refresh.
The two XQA graph cases and eager 8192-token architecture prefill differ by
0%, +0.0273% and -0.00773% respectively. Other timings are recorded but are
not substituted for those designated performance gates.
The #1028 baseline GPU suites also pass all 135 cases. These are operator
and integration-unit results, not model-token parity or final A3 completion.
Artifacts: `a3-step1c/logs/op-compare.log`, `host-unit-parent.log` on 54633.

## Step 2 retry split and environment inventory

The initial combined prototype preserved all 813 golden traces and passed
220 focused CPU tests, including immutable-snapshot checks. The work encountered
a two-line size-ceiling increase, two old AST-normalization adaptations, a
test AST type annotation, and six GPU-only fixtures selected in an early CPU
run. Under the requested retry/split rule it was returned to the exact parent
and split into dynamic-read centralization (2a) and constructor capture (2b).
No golden was regenerated and no acceptance ceiling was relaxed.

Step 2a centralizes the 78 previously inventoried non-constructor reads plus
two environment-membership predicates missed by the original inventory.
The 39 implementation and two metadata-constructor reads remain untouched
for 2b. The JSON `environment` table in
`tests/v1/attention/fixtures/flash_v100_dependency_baseline.json` records every
call site, expression, owner, capture/dynamic classification and `via_config`.
The `config.py` implementation row is separate from the 121 policy call sites.
Dynamic reads preserve short-circuit order, raw-string defaults, registered
getter parsing and the existing environment cache. No new cache is added.
The audit recognizes the mediated call sites so centralization cannot erase
them from the inventory. The package ceiling is reduced to 41 direct reads
outside config; repository model/platform/env ceilings become 2325/3964/306.

The owned 2a worktree is `v100-a3-config-20261008-153907`, based on #1073
`ec40c95f506ae5a95e23fe3334ffe4e1a5b2a4ae`. GPU model jobs remained queued under
the group/per-card leases while this CPU work proceeded. Both 2a and 2b stay
Draft until their prerequisite and complete outcome-map gates pass.

The reduced 2a scope passes the original focused command: 219 passed,
1 skipped and 28 GPU-only cases deselected. All 813 immutable golden cases
match; strict patch-use validation passes. Pre-commit including mypy and
layering passes. Logs are `a3-config-dynamic-{strict,precommit}.log` and
`a3-config-dynamic-shim.json` in the local task artifact directory.

## Step 2b frozen snapshot

`V100AttnConfig` now owns 41 resolved policy fields. The implementation's
39 environment reads and metadata builder's two reads use the same boundary,
at their original execution points. Construction first performs the existing
validation and short-circuit evaluation, then transfers the scalar fields into
a frozen snapshot without re-reading environment variables. Native operators,
geometry inherited from Triton and mutable buffers retain their existing owners
until their dedicated extraction steps.

`ConfigField` is a temporary compatibility descriptor for existing callers and
partial test fixtures. A completed implementation stores each policy value only
in its snapshot. A legacy assignment creates a replacement snapshot; retained
snapshots remain immutable. It is not an executor, hook, registry or mixin.
Steps 4/5 will consume the snapshot directly instead of passing an Impl object.
The test checks real construction, immutability, absence of duplicate scalar
storage, post-construction environment changes and legacy snapshot replacement.

All 813 immutable traces remain identical. The focused strict command returns
220 passed / 1 skipped / 28 GPU-only deselections; the extra passing case is the
snapshot contract. Pre-commit including mypy passes. Direct reads outside config
are now zero and the repository model/platform/env ceiling is 2325/3964/284.
Logs: `a3-frozen-config-{strict,precommit}.log`, `a3-frozen-config-shim.json`.
Constructor environment wiring and frozen-state ownership are separate commits.

For #1075, pinned #1028 was rebased to validation head
`0ab8ad125a7d339c402a78c5afbf6ebafad93fb4`; tree
`9ea84629e4c0af8fbd757d62714705ff4713f677` exactly matches the clean merge.
Its CPU suites return 37 passed / 98 skipped. This check will be repeated for
2b before readiness. All pending GPU gates remain pending; no merge is claimed.
