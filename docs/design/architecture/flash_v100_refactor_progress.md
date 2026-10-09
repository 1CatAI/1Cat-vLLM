# Flash-V100 Phase A3 execution record

Baseline: #1060, `8c96e32e56c09d4a3e3112cb5d1a367571f69476`.
Updated scope (2026-10-09): the user explicitly authorized self-review and
merging the completed work to main with proportionate acceptance gates. DDTree
is deferred for this campaign; #1089 is excluded. The earlier requirement to
wait for every slice's host/spec/GPU matrix no longer blocks merging. Historical
failed and pending evidence below is retained as history, not relabeled as a pass.
GPU validation uses the authorized `dx.1catai.com:54633` and task-owned locks,
artifacts and dependencies. Fifty idle host/spec retry queues were stopped under
this updated policy; their deferred records replace no test completion markers.

Current merge decision: accept the completed stack through #1110 after the
FlashInfer-SM70 compatibility fix described below. Keep the still-open A3
structural targets as follow-up work, rather than accumulating more draft PRs.
The final 153-line forward, 315-line maximum function, 269 cross-module private
references, 83 outside-Spec model terms and 20 state flags are **not** full A3
completion. Import cycles and environment reads outside config are already zero.

| Step | PR | Status | Metrics | GPU validation | Open items |
| --- | --- | --- | --- | --- | --- |
| 0: scope and codec ownership | — | Decision communicated; #1028 rebased locally | Baseline measured | Runtime parity belongs to 1c | Retest each subsequent step |
| 1a: immutable CPU trace and owner guard | #1071 | Merged to main | Production unchanged | 1667 passed / 7 inherited failures; no changed outcomes | Merge with prerequisite stack |
| 1b: patch efficacy + dependency ratchet | #1072 | Merged to main | 14 cycles / 32 forbidden edges frozen | 1668 passed / same 7 failures; 41 patch names consumed | Merge with prerequisite stack |
| 1c: route/token/output parity tools | #1073 | Merged to main | Production unchanged | 12 native cases and 4 Qwen contracts exact; 1684 passes / same 7 failures | Deferred; not a merge gate |
| 2a: dynamic environment boundary | #1075 | Merged to main | Outside-config reads 119 → 41; env ratchet 334 → 306 | 1684 passes / same 7 failures; exact old outcome map | Deferred; not a merge gate |
| 2b: frozen construction policy | #1076 | Merged to main | Remaining 41 → 0; 41 immutable fields; env ratchet 306 → 284 | 1685 passes / same 7 failures; one new pass | Deferred; not a merge gate |
| 3a: per-layer decode cache | #1077 | Merged to main | Private references 390 → 387 | 1686 passes / same 7 failures; 12 native outputs exact; 4 Qwen contracts exact | Deferred; not a merge gate |
| 3b: step plan and persistent metadata buffers | #1079 | Merged to main | Private references 387 → 380 | 1687 passes / same 7 failures; 12 outputs exact; 4 Qwen contracts exact | Deferred; not a merge gate |
| 4a: explicit decode executor dependencies | #1080 | Merged to main | Private references 380 → 374; cycles 14 → 13 | 1689 passes / same 7 failures; 12 outputs exact; 4 Qwen contracts exact | Deferred; not a merge gate |
| 4b: native decode candidates | #1081 | Merged to main | Private references 374 → 370 | 1691 passes / same 7 failures; 12 outputs exact; named timings and 4 Qwen contracts pass | Deferred; not a merge gate |
| 4c: outer decode dispatch candidates | #1083 | Merged to main | Forward 597 → 402; private 370 → 358 | 1691 passes / same 7 failures; 12 outputs exact; 4 Qwen contracts pass | Deferred; not a merge gate |
| 1c follow-up: immutable requested workload | #1084 | Merged to main | Production unchanged | 1692 passes / same 7 failures; no changed old outcomes | Deferred; not a merge gate |
| 5a: per-sequence prefill candidates | #1085 | Merged to main | Largest function 977 → 529; private 358 → 348 | Queued after prerequisites | Follow-up GPU evidence |
| 5b: batch prefill candidates | #1086 | Merged to main | Largest function 529 → 414; private 348 → 347 | Follow-up | Follow-up GPU evidence |
| 5c: debug observer | #1088 | Merged to main | Largest function 414 → 402 | Queued | Follow-up GPU evidence |
| 6a: verifier ownership | #1090 | Merged to main | Cycles 13 → 11; model terms 169 → 158 | Queued | Follow-up GPU evidence |
| 6b: metadata builder ownership | #1093 | CPU/golden/strict/rebase passed | Cycles 11 → 4; private 347 → 341 | Queued | Follow-up GPU evidence |
| 6c: attention policy ownership | #1095 | CPU/golden/strict/rebase passed | Cycles 4 → 3; private 341 → 333; model terms 158 → 154 | Queued | Follow-up GPU evidence |
| 6d: owned per-request metadata packet | #1096 | CPU/golden/strict/rebase passed | Private 333 → 332; final metadata mixin removed | Queued | Follow-up GPU evidence |
| 6e: registered speculative features | #1097 | CPU/golden/strict/rebase passed | Private 332 → 330 | Queued | Follow-up GPU evidence |
| 6f: complete prefill execution ownership | #1101 | CPU/golden/strict/rebase passed | Private 330 → 318; cycles 3 → 2; model terms 154 → 151 | Queued | Follow-up GPU evidence |
| 6g: owned comparison diagnostics | #1103 | CPU/golden/strict/rebase passed | Private 318 → 309; cycles 2 → 1 | Queued | Follow-up GPU evidence |
| 6h: shared allocation ownership | #1104 | CPU/golden/strict/rebase passed | Private 309 → 308; final cycle 1 → 0 | Queued | Follow-up GPU evidence |
| 6i: outer prefill dispatch | #1105 | CPU/golden/strict/rebase passed | Forward 400 → 153; largest function 400 → 318 | Queued | Follow-up GPU evidence |
| 6j: tree visibility feature ownership | #1107 | CPU/golden/strict/rebase passed | Private 308 → 303; model terms 151 → 137 | Queued | Follow-up GPU evidence |
| 6k: feature contract ownership | #1108 | CPU/golden/strict/rebase passed | Private 303 → 301; model terms 137 → 126 | Four queues staged | Follow-up GPU evidence |
| 6l: dynamic feature policies | #1109 | CPU/golden/strict/rebase passed | Private 301 → 287; model terms 126 → 83 | Four queues staged | Follow-up GPU evidence |
| 6m: decode one-shot logging | #1110 | Full CPU/golden/strict and logger tests passed | State flags 29 → 20; private 287 → 269 | Follow-up | Follow-up GPU evidence |
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

## Step 3 retry split

The combined workspace prototype is preserved, unpublished, at
`1becac8d7` in `v100-a3-workspace-20261008-162629`. Its final CPU check passes
238 tests / 1 skip / 28 GPU deselections, including all 813 immutable traces.
During implementation it encountered a missing moved import, a moved constant
reference, and new import cycles from workspace back through KV layout; hooks
also corrected formatting. Under the requested retry rule, work returned to
exact parent #1076 and was split into smaller rollback scopes. No trace,
calculation hash, dependency ceiling or GPU requirement was relaxed.

Step 3a owns only the per-layer decode cache. `DecodeCache` holds its tensors,
length and capacity and receives the extraction callable explicitly, so
workspace imports neither Impl nor KV layout/metadata. All three original
prefill invalidation sites remain in place. Cache arithmetic is compared to the
original method hashes after normalizing explicit receiver/field names; the
trace observer recognizes the actual invalidate frame and source-site ordinal.
The capacity test checks reuse, geometric growth, prefix preservation,
invalidation and instance isolation. Private references drop 390 to 387;
forward/maximum length, 14 cycles, env and model ceilings remain unchanged.
The mechanical extraction (`69d6b8595`) and ownership changes are separate.

Step 3b will move the per-step mixed-row plan and builder-persistent metadata
buffers onto 3a, retaining the tested prototype's lifetime and copy semantics.
It will have a separate outcome map and GPU gate. Neither scope is complete
based on CPU evidence alone.

Step 1c's four small Qwen contracts (FP16/E4M3 × eager/graph) compare exactly,
including all three prompts and chunked prefill. The complete regression map
has 1684 passes and the same seven inherited failures, zero changed old outcomes,
and all 16 new tool cases passing. Host-FP8 and DFlash2/DDTree real-model
records remain pending. Raw evidence is under `a3-step1c/{artifacts,logs}` on
54633. Failed IPC attempts are retained; short TMPDIR/RPC paths now pass the
shared-memory IPC preflight.

The rebuilt 3a scope passes 237 focused tests / 1 skip / 28 GPU deselections,
including all 813 immutable traces, strict shim use and parity-tool controls.
Pre-commit including mypy/layering passes. Evidence:
`a3-decode-cache-{strict,precommit}.log` and `a3-decode-cache-shim.json` in the
local task artifact directory. GPU and #1028 checks remain separate gates.

## Step 3a integration and Step 3b ownership

Step 3a is Draft PR #1077 at `f08a7711ad0e4931c32cb10f5ec2470e429d9505`.
Pinned #1028 was rebased onto it: head
`04f6758c1e50b20743ea0843d946e52766a83410`, tree
`ceecc15881a683f87a0a61e702ac8d405978b51d`, exactly equal to the clean merge-tree.
Host-KV/QSA CPU tests pass 37 with 98 GPU-only skips. Its source-verified
candidate, host integration and queued runners are in `a3-step3a` on 54633;
all GPU and prerequisite gates remain required before promotion.

Step 3b is rebuilt on that published head. `MixedDecodeRowsPlan` retains
per-step sharing across attention layers, the original metadata cache key and
lazy authoritative device-length gathers. `MetadataWorkspace` owns persistent
`DraftBuffers` and `SmallQueryBuffers` for each builder; capacity is explicit
and captured allocations never resize. Builder adapters retain the original
policy/evaluation points and supply plain values to workspace. The proposer
reads the actual workspace shape. Workspace receives neither Impl nor builder.

Mechanical extraction (`e3859c1d0`) is separate from ownership wiring.
The original metadata calculation hashes are unchanged; the comparison adapter
inlines delegation and normalizes explicit receiver names. Trace and GPU parity
observers enumerate the current nested owners on every observation, retaining
baseline labels without replacing the actual pointers. Focused tests cover
capacity refusal, alias retention and refreshed copy contents. Private module
references fall from 387 to 380, with no new cycles or forbidden edges.

PR #1048 now points at `860c126c15b244601faca9a66cc651c17cbe7234`.
Its Step 3b overlap is metadata; future readers/accounting adaptation must use
PR #1049's sole Python codec API. Neither its branch nor QSA native calls is changed.

The rebuilt 3b scope passes **238 tests / 1 skip / 28 GPU deselections**,
including the immutable 813-case trace, strict shim use, original calculation
hashes, capacity/pointer contracts and parity-tool controls. Evidence:
`a3-metadata-workspace-strict.log` and `a3-metadata-workspace-shim.json`.
GPU route/token/output/performance and complete outcome maps are pending.

Step 3b's pinned #1028 rebase at code head `4259f29bd` passed 37 host-KV/QSA
CPU tests with 98 GPU skips; its integration tree matches the clean merge-tree.
The subsequent full Step 1c Flash-Next baseline exposed an older `_C` HC ABI
while warming the #1028 integration. A source-matching integration `_C` build is
running under `a3-native-abi` on 54633, without changing the shared runtime.
Host model parity remains failed/pending, with failed logs retained. See the
known-issues document; no model gate or merge is claimed from the CPU results.

## Step 4a decode dependency boundary

Step 3b is Draft PR #1079 at `7b061e4e168f19820e3f400e454baf9c567148d4`.
Its final pinned PR #1028 rebase has head
`b13f5027c1321c791c25163cafe77663f517f1c5`, tree
`5c76785cc4ca73452e574dbb254d72ccc0190670`, equal to the clean merge-tree,
and 37 CPU passes / 98 GPU skips. Verified snapshots and dependent GPU queues
are under `a3-step3b` on 54633. Model fixtures remain identical to Step 1c.

Step 4 is split into executor ownership and the selection loop so each boundary
has an independent rollback scope. The unchanged calculations were grouped
in commit `d6efe5a69`. The initial class receiver annotations conflicted with
mypy's bound-method rules; the extraction retains an untyped receiver only
until ownership wiring. The final executor imports and receives no Impl.
It accepts `DecodeConfig` (frozen policy plus geometry), explicit native ABI
callables and `V100Workspace`. Legacy entry points delegate through a narrow
adapter that preserves post-construction operator replacement and partial
legacy fixtures. Diagnostic callbacks are explicit; their migration to event
subscribers belongs to Step 5c. They do not change debug capture behavior.

The original window/codec calculations are repeated locally with explicit
inputs pending shared planning in Step 4b; no feature policy is reimplemented.
The existing feature predicate is supplied as a callable. Original method hashes
are preserved by normalizing actual dependency paths and typed delegates;
no hash or trace fixture is regenerated. The new standalone executor test
injects separate scalar/XQA operators and verifies the output, selection and
shape/scale hints without invoking the backend's operators.

The immutable 813-case trace and calculation suite pass. Private module references
drop 380 to 374, cycles 14 to 13; no new forbidden edge or cycle is introduced.
The reduced dependency ceiling is locked. Full strict CPU, final rebase,
complete GPU outcome maps and model/performance gates remain required.

Step 4a's strict focused suite passes **240 tests / 1 skip / 28 GPU deselections**,
including all 813 traces, the two standalone executor cases and 40 consumed
legacy patch names. Original calculation hashes remain fixed. Evidence is
`a3-decode-executor-strict.log` and `a3-decode-executor-shim.json` in the local
task artifact directory. The owned adapter's type narrowing and AST assertion
were corrected; the final mypy/layering run passes. The diagnostic callbacks
retain their old state owner pending Step 5c, rather than claiming that debug
state has already been decoupled.

## Step 4b native selection after reducing scope

Step 4a is Draft PR #1080 at `4b72df3b10418d5d10190e3b80f4d8183c36a923`.
Its pinned PR #1028 integration is `ee202528b9d1843553d8cbe0207f2b1bb16632d0`,
tree `09e77321a1534e5666d99f583b645daf2aebd3b3`, matching the clean merge-tree.
Host-KV/QSA CPU tests pass 37 / 98 GPU skips. Its verified snapshots and
GPU queues are under `a3-step4a` on 54633. PR #1048 advanced to
`847bb3d8eb66b395f3e7910c3ac66953eb6bd619`; it has no file overlap with 4a.

The combined dispatch prototype passed all 813 traces but encountered three
check failures: the extracted layer-name annotation was narrower than the
existing metadata type, the expanded AST adapter lacked its decode import,
and it initially treated the new outer delegate as an old named delegate.
Under the retry rule, work returned to exact parent PR #1080 and was split.
The unpublished prototype remains in `v100-a3-decode-dispatch-20261008-180444`
at move commit `677d5f068` plus saved patch/plan artifacts; it is not pushed.
No golden, original calculation hash or acceptance gate was relaxed.

Reduced Step 4b introduces real XQA and scalar candidates behind the same
existing structural admission. Their run methods execute the original native
calculations. `plan/routing.execute` consumes candidates in order, calls
admit before run, and continues only on None. Its recorder preserves each
observation's exact position around native calls. A temporary observation
adapter supports private legacy calls outside forward. The generic driver
is shared by both actual decode implementations, not an empty registry.
Step 4c will move the outer diagnostic/fallback choices separately.

Mechanical extraction `8e72dc14a` retains the calculations before ownership
wiring. The focused trace/calculation/executor/selection suite passes 19 tests,
including all 813 traces. The two new cases prove that preparation and route
observations survive decline, rejected candidates do no work, falsey results
complete, and later candidates are not evaluated after success. Original
calculation hashes remain fixed. Complete strict CPU, rebase and GPU gates
remain pending.

Step 2a's complete GPU regression reports 1684 passes and the same seven
inherited failures. Its 12 native cases have max-abs zero. Step 2b is running;
full parent/candidate outcome maps are emitted after both finish. DDTree eager
timing in this diagnostic run is not a designated XQA/prefill performance gate.

Reduced Step 4b passes **242 tests / 1 skip / 28 GPU deselections** under the
strict patch-use plugin. The immutable 813-case trace and original calculation
hashes pass, and all consumed patch names retain real call/read evidence.
Pre-commit including mypy/layering passes. Private references fall 374 to 370;
cycles remain 13 and no new forbidden edge is added. Evidence:
`a3-decode-candidates-{strict,final-precommit}.log` and
`a3-decode-candidates-shim.json`. GPU and final rebase evidence remain separate.

## Step 4c outer decode dispatch

Step 4b is Draft PR #1081 at `2b6fc00a6e614dd3dbda491a8999e2d8d88c9a34`.
Its pinned PR #1028 integration is `b0efeed09fbdb2046d6e81690401644b69adad28`,
tree `1e6ec72c42585c55e2444471fdce14ca88074cb0`, exactly the clean merge-tree,
with 37 CPU passes / 98 GPU skips. Its source-verified snapshots and GPU queues
are under `a3-step4b` on 54633. A transient SSH interruption was retried before
staging; hashes were verified before its dependent runners were started.

Step 4c extracts the outer block mechanically in `55905d068`, then replaces it
with six real ordered candidates: unavailable decode, paged-prefill bridge,
dense cache, dense reference, scalar-disabled fallback and native paged decode.
Each admit is pure and retains its original short-circuit expression. Runs use
the existing executor/config/ops/workspace boundary. Native XQA/scalar selection
remains inside Step 4b's loop. No executor imports or receives Impl; explicit
Triton/diagnostic callables retain compatibility until debug event extraction.

The trace observes actual candidate policy reads at their new owner, without
fabricating decisions or updating any golden. The original forward AST hash
also remains unchanged after expanding the actual declared candidate order,
predicates and bodies and normalizing explicit request/dependency paths. The
adapter validates the real delegation and selection expressions; it does not
substitute saved branch bodies. The generic loop's decline/order tests remain.

The focused and full strict checks pass: **242 tests / 1 skip / 28 GPU
exclusions**, including all 813 immutable traces and the original calculation
hashes. Pre-commit with mypy/layering passes. Evidence is in
`a3-outer-decode-{owned-check,strict,precommit}.log` and
`a3-outer-decode-shim.json`. Forward shrinks 597 to 402 lines and private module
references 370 to 358. Cycles remain 13, maximum function 977, model hits 169,
outside-config environment reads 0, state flags 29; no new forbidden edge or
cycle appears. The lower ceilings are locked. GPU and final rebase remain gates.

Step 2's complete GPU maps are now final: dynamic 1684 passes / 7 inherited
failures versus the same parent outcomes; frozen 1685 passes / the same 7,
with only the new frozen-policy test added and passing. Both 12-case native
comparisons have max-abs zero. The run ended with exit 0 and is recorded in
`a3-step2/logs/outcome-parity.json` on 54633, copied locally as
`a3-config-gpu-outcome-parity.json`. PRs #1075/#1076 have updated evidence and
remain Draft while Step 1c model prerequisites are pending.

## Parity recorder configuration snapshot

PR #1083 is `15d8de1229c8df35f2df0c81e0213b66278a1cf5`. Its actual pinned
PR #1028 rebase is `256bb7e0e63d72fc5ff7f75e419845536787600b`, tree
`e4091b3a41d4e6146c73fc5853c4bd5b29319985`, identical to the clean merge-tree;
37 CPU tests pass and 98 GPU cases are skipped in that local check.

The DFlash2 baseline completed graph warmup and generation but failed while
saving its parity JSON: engine initialization enriched the nested speculative
options with a non-JSON `ModelConfig`. The recorder had retained a reference
to that mutable dictionary. No successful token artifact was produced, so
this run is not parity evidence. Fix the tool by copying the requested JSON
options before constructing the engine. The engine still receives exactly the
same options; no serializer fallback, removed field or production change is
used. The new CPU regression mutates both speculative and graph subcontainers
and checks the actual saved contract, tokens and native provenance.

The focused tool suite passes 17 tests. Deploy this small repair as versioned
`a3-parity-tools-v2` on 54633, separately from immutable backend snapshots.
Each result records the actual harness hashes and backend source SHA. Both
host/spec comparison arms use the same repaired tool. Preserve failed logs
and rerun generation; tokens from an exited worker cannot be reconstructed.

Step 3a's full GPU outcome map now reports 1686 passes / the same seven
inherited failures, with only its new cache test added. All old outcomes match.
Its 12 native cases and designated XQA/prefill timing gates pass; see
`a3-step3a/logs/{regression-parity.json,op-compare.log}`. Full model gates remain
pending and PR #1077 remains Draft.

The recorder repair passes the complete strict suite: **243 passed / 1 skipped /
28 GPU exclusions**, including all 813 unchanged golden traces. Pre-commit,
mypy and layering checks pass. Logs are `a3-parity-snapshot-{strict,precommit}.log`
and `a3-parity-snapshot-shim.json` in the local task evidence directory.
The versioned remote harness passes SHA256 and import-location verification;
host/spec runners are updated atomically and the failed DFlash baseline is
queued again. Step 4c's immutable source and host-integration snapshots and four
dependent GPU queues are staged under `a3-step4c`; no earlier source snapshot
is overwritten by the tool repair.

## Step 5a sequence candidates

The preceding tool repair is PR #1084 at
`ec47ee44cdbd590f1856cc4a170c6c7afac09b3b`. Its actual pinned PR #1028 rebase
is `57ccb8c40eeb5ae4bcb373b56e357601891fd498`, tree
`5eac54b5afd52124bcbd11e040bb070db7dec37b`, with 37 passes / 98 GPU skips.
The full outcome-map runner is queued under `a3-snapshot` on 54633.

Mechanical extraction `45c0fda62` separates the original 471-line sequence
block before ownership changes. Seven real candidates now execute through the
shared admission/attempt/decline loop: BFLA, FA2, contiguous BHMD, contiguous
dense, FP8 bridge, split-KV and paged. `PrefillExecutor(config, ops, workspace)`
neither imports nor receives Impl. Native functions and temporary policy/debug
callbacks are explicit dependencies; remaining batch/debug ownership belongs
to Steps 5b/5c. Actual mask/view/gather/native preparation occurs in candidate
run methods and retains side effects when a candidate declines.

The first four candidates preserve the unconditional split-KV then FP8-bridge
policy reads after successful preparation but before final native execution.
The fallback generator performs those reads at the same position when all four
decline. Contiguous policy is evaluated once. BHMD retains its destination copy
and early debug skip. A declining FP8 bridge directly executes the original
paged fallback, bypassing split-KV even if that predicate was true.

All 813 immutable traces pass. The original full calculation hashes also pass:
the test projects the actual candidate order, admission/decline checks,
preparation calls and logging callbacks back into the existing AST comparison.
No saved branch body or changed golden is substituted. Ten new independent
operator-injection cases cover every winner plus mask, FA2 and bridge declines,
policy timing and destination identity. Source evidence is
`a3-prefill-{trace-2,composition-1,injection}.log` locally. The first trace check
reported the newly introduced private helper as an extra call; that helper now
has a public orchestration name, while all original observed decisions remain.

Current ceilings are 402 / 529 / 348 / 13 / 169 / 0 / 29 for forward, largest
function, private references, cycles, outside-spec model terms, outside-config
environment reads and state flags. No new forbidden dependency edge appears.
Both largest-function and private-reference ceilings are tightened. The first
mypy run found generator-variable inference and AST type-narrowing issues;
these are corrected without changing calculations or the trace contract.
GPU locks are currently occupied by another task; all owned queues keep waiting
and all unverified PRs stay Draft. No other task's processes are stopped.

The final unchanged-source strict run passes **253 tests / 1 skip / 28 GPU
exclusions**, with consumed shim patches and unchanged golden/calculation
oracles. Pre-commit including mypy/layering passes. Evidence:
`a3-prefill-candidates-strict-final.log`,
`a3-prefill-candidates-shim-final.json`, and
`a3-prefill-candidates-precommit-2.log`. A preceding strict run read source while
the type-only corrections were being applied and is retained as an invalid
mixed-revision check; its lone calculation-source assertion failure is not
accepted as final evidence. The final focused 25-test suite also passes.
The four existing Qwen baseline JSON contracts were checked against their
requested engine JSON and match exactly, so the recorder snapshot repair does
not invalidate those completed baselines.

## Step 5b batch candidates

PR #1085 is `fb0cfe6697d352d372bb85b07c3f1b518b969afe`. Its actual pinned
PR #1028 integration is `a218b5b1d70aff3b041bbac041a23fc717bb5f12`, tree
`275d756f178d1577339f9519c718834cd4e8a8fc`, matching the clean merge-tree;
37 CPU tests pass and 98 GPU cases are skipped locally. Its verified source,
host integration and four dependent GPU queues are under `a3-step5a` on 54633.

Mechanical extraction `e51840dd7` precedes ownership changes. The same explicit
PrefillExecutor now selects noncausal batch, tree verification, small-query
decode and mixed decode rows in their original order. A partial selection
returns no completed result when every candidate declines, allowing ordinary
sequence work to proceed. Complete batch results are distinct from partial
row sets, including an explicitly completed callback returning None. The
required selection wrapper still raises when no candidate completes and still
accepts falsey non-None results.

All 813 immutable traces and the original calculation hashes pass. The source
oracle projects actual batch admission/run statements and validates both the
driver and legacy request arguments; no old branch body or new golden is used.
Six independent batch cases cover all four winners, remaining sequence rows
and the terminal-None distinction. One additional driver case proves preparation
survives a complete decline, while required dispatch still rejects it. The
focused executor/driver/calculation run reports 22 passes; combined with the
trace check, evidence is in `a3-prefill-batch-{trace,composition,injection}.log`.

The metric ceiling becomes 402 / 414 / 347 / 13 / 169 / 0 / 29. Existing
policy, profiling and speculative callbacks are still explicit temporary
dependencies; their extraction remains assigned to Steps 5c/6. No new import
cycle or forbidden edge appears. GPU model and complete outcome gates remain
pending; the prior DFlash2 baseline is now capturing FULL graphs with the fixed
recorder, and the host integration's private native build is still progressing.

The complete strict run passes **260 tests / 1 skip / 28 GPU exclusions** with
all consumed shim patches. Pre-commit including mypy/layering passes after a
test-local empty-list type annotation was added. Evidence:
`a3-prefill-batch-strict.log`, `a3-prefill-batch-shim.json` and
`a3-prefill-batch-precommit-final.log`. Production code was held fixed throughout
this full run.

The fixed recorder has saved the DFlash2 baseline on 54633 as
`a3-step1c/artifacts/dflash-parent.json`, copied locally as
`a3-dflash-parent-v2.json`. FULL graph capture completed; all four workers hit
`prefill_smallq_fp16_grouped_fp32`. Prompts have 25/21 input tokens and 2/64
output tokens: Paris stops naturally; the Chinese Rayleigh explanation reaches
the requested limit. This is a baseline route/token record, not a finished
comparison or long-output quality claim. The recorded harness SHA256 is
`e38d6bebda6a495703fba53d326caec5e553a03f859a192eca1a32b00ad3eebf`.
DDTree, candidate and host-FP8 model gates remain pending.

## Step 5c prefix debug subscriptions

PR #1086 is `e21545418438f4ea49d2e58da0cd5502fc6f7cf2`. Its actual pinned
PR #1028 integration is `bbeb4d9ab05d2fc576b2e83462fb6db84a0cd36b`, tree
`b42901b569aaf40cb5af746f44262a36f895e91e`, matching the clean merge-tree.
The three host-integration CPU suites pass 37 tests with 98 GPU skips. Its
source and four dependent GPU queues are staged and verified under `a3-step5b`.

Mechanical extraction `a527f7442` precedes ownership changes. The original
221-line prefix diagnostic calculation now runs in two ordered subscribers:
reference gathering/comparison, then dumps/reporting. The prefill path emits
an explicit event with tensors, geometry, configuration and narrow reference
callbacks. Neither subscriber takes an Impl receiver. Synchronous subscription
order, failures, original diagnostic guards, output layouts, slot mapping and
process-shared flags remain unchanged. Existing decode comparison methods are
still legacy methods and must be addressed before the final dependency gate.

All 813 immutable traces and original calculation hashes pass. The source
oracle expands the actual subscriptions, event arguments and reference handoff;
it does not substitute a saved calculation body. Six direct diagnostic tests
exercise real CPU cache extraction, draft/dense reference dispatch, valid and
invalid slots, NaN dumps, shared one-shot flags and subscriber error propagation.
Evidence: `a3-prefill-debug-focused.log` and `a3-prefill-debug-injection.log`.
The current ceiling is 402 / 402 / 347 / 13 / 169 / 0 / 29; no new forbidden
edge or cycle appears. The largest-function ceiling is tightened.

The private #1028 native build is complete. Its `_C.abi3.so` SHA256 is
`0e17f8c1320bd5c54bc1932c998e7463dc879f50a927f5f5fed84788f1ee3458`.
An isolated import confirms nine `sm70_hc_ll_down_out` arguments including
`optimized_loads=True`; host model parity remains queued. DDTree's frozen
baseline now has a recorded proposer/model slot-mapping type failure, documented
in `flash_v100_known_issues.md`. The DFlash comparison is queued independently;
its completion marker cannot mark the combined speculative gate complete.

The fixed-source strict run passes **266 tests / 1 skip / 28 GPU exclusions**,
including all consumed shim patches. Evidence: `a3-prefill-debug-strict.log`
and `a3-prefill-debug-shim.json`. All code/type/layering hooks pass; the first
pre-commit invocation only reformatted an extra Markdown blank line.

## Step 6a verifier ownership

PR #1088 is `27ecbdccfe4d1e4b11a65b1f2cd39619e97df745`. Its actual pinned
PR #1028 integration is `30d27b609bcd983d45aa5bfe8e4439c377959423`, tree
`1e6b737b5a178fc25f71cde2d7edb4c5be9f58ec`, equal to the clean merge-tree.
The integration passes 37 CPU tests with 98 GPU skips. Four dependent GPU
queues and hash-verified sources are staged under `a3-step5c` on 54633.

Mechanical grouping `c06fb8c15` precedes explicit ownership. VerificationExecutor
receives frozen VerificationConfig and explicit VerificationOps; its calculation
module no longer imports Impl. Grouped-kernel admission receives only its five
required fields/callables. Tree correction receives the three scalar policy
fields it consumes through the frozen configuration. Internal legacy test
injections remain explicit optional operators; a falsey callable is retained.
Temporary typed adapter functions preserve existing callers and method signatures.
They do not claim completion of speculative feature registration or removal of
all legacy adapters, which belongs to the following Step 6/7 work.

All 813 immutable traces and original calculation hashes pass (15 focused
checks). The source oracle validates every adapter argument and projects actual
executor dependencies back to the original calculation. Eight independent
executor cases and seven existing attention-hook tests pass: grouped FP16,
E4M3, XQA and scalar order, native input contract, persistent metadata pointers,
falsey injected operators and declared causality. Evidence:
`a3-spec-verifier-focused.log` and `a3-spec-verifier-injection.log`.
The ceiling is 402 / 402 / 347 / 11 / 158 / 0 / 29; cycles and model-name
ceilings are tightened and no new forbidden edge appears. Moving the old static
calculation aliases exposed mypy descriptor inference; typed function aliases
fixed the intermediate mechanical commit before the ownership change.

The first complete strict run exposed three historical policy cases calling an
unbound Impl method on SimpleNamespace. Those same cases now inject the actual
VerificationExecutor, preserving their test IDs, inputs, native/route assertions
and strict shim patch consumption. Ordinary non-speculative contract validation
also bypasses executor construction, retaining the lightweight per-token guard.
The final focused executor/calculation/three-policy-case run passes 14 tests.
Evidence: `a3-spec-verifier-focused-final.log`. The earlier strict failure is
retained in `a3-spec-verifier-strict.log` and is not a passing final gate.

Step 1c DFlash2 parity now passes: both fixed greedy requests match exactly,
including all four workers' route records. The shared versioned recorder returns
`equal: true, requests: 2`. Evidence: `a3-step1c/logs/dflash-compare.log` and
`dflash.done` on 54633, with both JSON artifacts copied locally. DDTree remains
separately blocked on its original baseline interface bug; the isolated fix is
Draft PR #1089, `6f22879a12a2a51e26656797ef1fa9b6c3ba2fa1`. It is not part of
A3. Applying it symmetrically to DDTree's model comparison requires the pending
baseline clarification. Host-FP8's rebuilt-binary parent unit suite passes all
135 cases; its full model is currently loading under the declared contract.

The final fixed-source strict run passes **274 tests / 1 skip / 28 GPU
exclusions**, with all original shim patches consumed. Evidence:
`a3-spec-verifier-strict-final.log` and `a3-spec-verifier-shim-final.json`.
Pre-commit including mypy/layering passes (`a3-spec-verifier-precommit-ready.log`).
No production files changed while this accepted full run was executing.

## Step 6b metadata builder ownership

PR #1090 is `f917a5f35df0c652c90896bf7f053d73d62022ef`. Its actual pinned
PR #1028 integration is `674785c0e9495fd9f6feacfaa51bdf490eb1ee28`, tree
`ffe9ec852916153f1ecb0ba228fcffb53ab7b0a0`, matching the clean merge-tree.
The integration passes 37 CPU tests with 98 GPU skips; four hash-verified GPU
queues are staged under `a3-step6a` on authorized 54633.

Mechanical extraction `9a842373f` precedes builder ownership. The common
builder no longer inherits the speculative method mixin. SpecMetadataState
owns its configuration and MetadataWorkspace, receives immutable inputs and
five narrow common callbacks, and never receives/imports the common builder.
The old MetadataHooks object remains only as a compatibility adapter. Attention
and metadata field mixins still exist; feature registration is subsequent work.

The original builder identity is passed explicitly: grouped metadata prepared
by the proposer continues to validate against that identity, not the new state
object's identity. Persistent draft and small-query buffers retain addresses
across refreshes. Legacy configuration writes replace immutable inputs while
legacy state reads/writes reach the single owner. Existing ordering/capture
tests inject owned callbacks and retain their original assertions and test IDs.

All 813 immutable traces and 14 original metadata calculation hashes pass.
Six independent ownership cases cover replay pointers, prepared metadata
identity, two capacity failures before publication, compatibility writes and
base-build failure propagation. The focused suite passes 24 tests; evidence:
`a3-spec-metadata-owner-golden.log`. The dependency ceiling is now
402 / 402 / 341 / 4 / 158 / 0 / 29, without new forbidden edges.

Step 1c host-FP8's rebuilt native parent and candidate unit suites each pass
135 tests. The parent full model has saved its two greedy records; candidate
execution is underway. Full host parity is not yet claimed. DFlash2 parity
passes; the original DDTree baseline failure and separate fix #1089 remain
recorded, with baseline clarification pending.

The fixed-source strict suite passes **280 tests / 1 skip / 28 GPU exclusions**,
including actual consumed shim patches. Evidence:
`a3-spec-metadata-owner-strict.log` and `a3-spec-metadata-owner-shim.json`.
Pre-commit including mypy/layering passes; no production or oracle file changed
during the accepted full run. Four GPU queues and #1028 replay are next.

## Step 6c attention policy ownership

PR #1093 is `21b38c8d0436dec88ed5cfe4aed46eacfbc9c3cf`. Actual pinned
PR #1028 replay produces `1fc3f8c1711d304292ce47ea12c994574b949c0d`, tree
`980392578b6f0a01eff4eb8814fdbdd95e2597a5`, identical to clean merge-tree.
Its three integration suites pass 37 CPU cases with 98 GPU skips. Four queues
and verified source snapshots are staged under `a3-step6b` on 54633.

Mechanical extraction `e5feb51a5` precedes ownership. SpecAttentionState owns
construction policy and receives native ABI values, a keyword probe and native
operators. It neither imports nor receives Impl. The attention mixin and
single-provider AttentionHooks are removed. Fallback dispatch takes common
policy; contract validation takes a callback. Legacy methods bind at the common
assembly boundary, preserving bound/unbound calls and instance overrides;
VerificationExecutor no longer contains adapters receiving an Impl receiver.
Ordinary validation retains its lightweight guard without executor construction.

All 813 immutable traces and original calculation hashes pass. The recorder
observes the real verifier predicate, retaining the frozen canonical event name.
The source oracle checks original method-to-executor bindings and actual narrow
callback/policy arguments. Six direct owner tests cover layer-local state,
short-circuit configuration reads with/without an operator, native ABI injection,
prefill keyword wrapping, the allocation-free ordinary guard and bound/unbound
compatibility arguments. Focused suites pass 30 + 6 tests.

The first focused run failed only the dependency ratchet: mechanical extraction
introduced a policy-to-ops import and a second assembly-to-policy edge. Native
values/probes are now injected, and assembly uses its existing feature boundary.
The passing run retains the original limits; no new forbidden edge is allowed.
Evidence: `a3-spec-attention-owner-focused-final.log` and
`a3-spec-attention-owner-injection.log`. Current metrics are
400 / 400 / 333 / 3 / 154 / 0 / 29; remaining metadata field mixin and per-method
feature registration are not complete.

Step 1c host-FP8's first complete comparison fails greedy token identity.
Both model arms complete and pass 135 native host unit cases each. France is
identical; the 64-token Chinese answer first diverges at zero-based token 21.
Both arms have identical 2,705 production source hashes, native-library hashes,
workload, recorder and GPU state. Only declared private cache/IPC paths differ.
The failed gate is retained, and an unchanged-parent repeat is queued to test
baseline reproducibility. No host completion marker or merge approval is inferred.

The complete fixed-source strict suite passes **286 tests / 1 skip / 28 GPU
exclusions**, with all shim replacements consumed by real calls/reads. Evidence:
`a3-spec-attention-owner-strict.log` and `a3-spec-attention-owner-shim.json`.
Pre-commit/mypy/layering passes (`a3-spec-attention-owner-precommit-final.log`).
No production/source-oracle file changed during the accepted strict run.

## Step 6d per-request metadata packet

PR #1095 is `224119021840522a1860661e8ae2a5f192109053`. Actual pinned
PR #1028 replay is `3cfa92828870ee90b0fa80b3962d615e112b13c6`, tree
`94070b3149116d9740aa48b8f07d0fece6c8982f`, equal to the clean merge-tree.
Three integration suites pass 37 CPU tests with 98 GPU skips. Four GPU queues
and SHA256-verified source snapshots are staged under `a3-step6c` on 54633.

Mechanical field grouping `1fed2deb4` precedes ownership. The remaining metadata
field mixin is removed. Common metadata owns a SpecMetadataPacket via spec_state;
old field reads/writes/deletes forward to that packet. The Triton builder's exact
metadata class is adopted as the existing Flash subtype in place, preserving
object identity and all tensor addresses. Other external metadata types retain
their legacy view. Shallow copies own independent packet containers with shared
tensors, and packets have no reference back to the metadata object.

The tree attachment operation is now a public owned API, with its old name
retained as a compatibility alias. All 813 immutable traces and the 14 metadata
calculation hashes pass. The source oracle maps the public operation's actual
body to its original name without changing the fixture. Six direct packet tests
cover in-place adoption, legacy access/deletion, shallow-copy isolation, prompt
object release, and tree capture's authoritative values and persistent pointers.
Focused suites pass 24 + 6 tests; evidence: `a3-spec-features-focused.log` and
`a3-spec-features-packet.log`. The ceiling is 400 / 400 / 332 / 3 / 154 / 0 / 29,
with no new forbidden edge. Per-method SpecFeature registration is still open.

Step 3b's authorized 54633 regression is complete: 1687 passes and the same
seven inherited failures, with exactly one new passing workspace case and no
changed existing outcomes. All 12 native operator cases have max-abs 0.
Designated timing deltas are FP16 XQA graph 0%, E4M3 XQA graph +0.0127824%,
and 75T-role prefill -0.0505210%, all within 2%. DDTree eager timing is recorded
but has no declared performance role; no broader timing acceptance is claimed.
Evidence: `a3-step3b/logs/regression-parity.json`, `op-compare.log` and
`regression.done` (2026-10-09 05:32:16 +08:00). Model gates remain pending.

The fixed-source strict suite passes **292 tests / 1 skip / 28 GPU exclusions**,
with real shim consumption (`a3-spec-features-strict.log`,
`a3-spec-features-shim.json`). Pre-commit, mypy and layering all pass
(`a3-spec-features-precommit.log`). Production and source-oracle files remained
unchanged during the accepted run. GPU completion and feature registration are
still pending.

## Step 6e method-specific verification providers

PR #1096 is `f92fbed160516cfa6da47680505d3d35826da1e5`. Pinned #1028
replay is `3f5cf2a6190460ccd6f2194fde63c2b5e5b3409c`, tree
`0c81e04abe81e1800775ce051e33f9a984abf341`, equal to clean merge-tree;
37 CPU integration cases pass with 98 GPU skips. Four source-verified queues
are staged as `a3-step6d` on 54633.

Mechanical extraction `59f4c9c9e` precedes registration. A proposer-side
SpecFeature protocol and immutable method registry select independent per-builder
DFlash2Feature, DDTreeFeature or MTPFeature instances. The tree provider preserves
verification suppression, the parallel provider consumes prepared metadata, and
the linear provider expands small queries. Explicit tree/prepared inputs retain
their precedence even when supplied with another configured method. Unknown or
absent methods retain the original linear fallback; initialization/config reads
remain before method selection.

The 16-case method/payload matrix verifies actual preparation calls, order and
lazy capacity reads. Two registry cases check distinct provider instances,
unknown methods, immutable registrations and external provider injection.
All 813 golden traces and 14 calculation hashes remain unchanged; the metadata
oracle expands the actual complete tree-provider body, and the independent
matrix covers all registered providers. The first focused run caught an
accidental abbreviated prepared-method name in the extracted tree provider;
that production call was corrected without changing fixtures or expectations.
The accepted focused suite has 42 passes (`a3-feature-registration-focused-final.log`).
Public metadata calculation APIs replace two cross-module private references,
locking the ceiling at 400 / 400 / 330 / 3 / 154 / 0 / 29.

Host-FP8's unchanged-parent repeat has now failed to reproduce the original
parent Chinese output at the same token 21 (96378 versus 99505). France is
identical. This diagnoses baseline non-reproducibility for this workload,
not its numerical cause and not candidate acceptance. The original and repeat
artifacts remain separate; no host completion marker is written. See
`a3-step1c/logs/host-parent-repeat-report.json`. DDTree baseline clarification
and the complete model gates remain open.

The fixed-source complete strict suite passes **310 tests / 1 skip / 28 GPU
exclusions**, including actual shim consumption (`a3-feature-registration-strict.log`
and `a3-feature-registration-shim.json`). No production/source-oracle file
changed during the run. Subsequent lint corrections only wrap a dictionary value
and annotate the new test's mixed event list; the affected tests are rerun.

## Step 6f complete prefill execution ownership

PR #1097 is `bad1ad1d0cbce85a3744202d009e4cb92b87e19d`. Actual #1028
replay is `69cbc10d8e10d5e19a1e247d1c07d633d552d43c`, tree
`f2228c98f0bf5700774d6ba1ba68d6296ce56144`, equal to clean merge-tree;
37 CPU integration tests pass with 98 GPU skips. Four source-verified GPU queues
are staged as `a3-step6e`. Pinned #1028/#1048 heads remain unchanged.

Prefill calculation functions now receive an owned PrefillExecutor configured
with immutable policy/geometry, explicit native operators and narrow callbacks,
and the existing workspace. Common assembly retains the external method
facades and instance overrides, including falsey callables. The candidate
executor uses this owner; prefill neither imports nor receives Impl. No
calculation body is moved or rewritten in this slice; ownership adapters and
dependency construction are the change.

The initial golden run caught one omitted grouped native operator dependency,
previously retrieved with getattr in mixed-row preparation. The operator is
now injected and the unchanged golden passes. The 38-case focused suite covers
all 813 traces and original method hashes (`a3-prefill-owner-focused-final.log`).
Five boundary cases exercise real bound/unbound profile calls on the owner,
falsey overrides, native refresh/feature overrides and policy snapshot isolation
(`a3-prefill-owner-boundary.log`). Metrics tighten to
400 / 400 / 318 / 2 / 151 / 0 / 29 without new forbidden edges.

Step 4a GPU regression/native gates finish on 54633: 1689 passes / the same
seven inherited failures, exactly two new passing cases and no changed old
outcomes. All 12 attention outputs have max-abs 0. Designated timing deltas are
FP16 XQA graph 0%, E4M3 XQA graph +0.0236434%, and 75T-role prefill +0.778561%,
all within 2%. Other timings are recorded without broadening their acceptance
role. Evidence: `a3-step4a/logs/{regression-parity.json,op-compare.log,regression.done}`
(completed 2026-10-09 06:09:34 +08:00). Complete model gates remain pending.

The complete fixed-source strict suite passes **315 tests / 1 skip / 28 GPU
exclusions**, including real shim use (`a3-prefill-owner-strict.log` and
`a3-prefill-owner-shim.json`). Pre-commit/mypy/layering pass
(`a3-prefill-owner-precommit-final.log`). No production/source-oracle file
changed during the accepted run.

Step 3a's small Qwen model gate is also complete: FP16/E4M3 crossed with
eager/graph, three requests each, all token/route records equal. The four
`a3-step3a/logs/qwen-*-compare.log` files report `equal=True, requests=3`;
`qwen.done` is dated 2026-10-09 05:16:38 +08:00. Host/spec gates still remain.

## Step 6g comparison diagnostics ownership

PR #1101 is `785d84b7fd62bd7e675dbc517d3b9fa016baf80c`. Actual #1028
replay is `f0dd2d11a4c1d4ebd061b2dc8256db43bd09c701`, tree
`b8b9fa02f3963d0b33eae6d54c1f831954bb62ef`, equal to clean merge-tree;
37 CPU integration tests pass with 98 GPU skips. Four source-verified durable
queues are staged as `a3-step6f` on 54633.

ComparisonExecutor receives policy, scalar geometry, explicit native/reference
operators and per-layer ComparisonState. No comparison calculation imports or
receives Impl. Legacy counter reads/writes reach the owned state; enabled
partially initialized objects retain their missing-counter error. Common
assembly binds the original class-cell super().forward as a narrow reference
callback, preserving export-monkeypatch behavior and per-instance overrides,
including static helpers. Calculation bodies stay in place; the old super call
becomes the injected reference callback, already covered by the original AST
projection. No mechanical code move is included in this slice.

All 813 traces and original calculation hashes pass in 21 focused cases
(`a3-debug-owner-focused.log`). Six direct tests verify layer-local quotas shared
across short-lived executors, legacy counter resets, partial initialization,
capture skipping after quota reservation, reference failure propagation, real
BHMD JSON output and static helper overrides (`a3-debug-owner-boundary.log`).
The ceiling tightens to 400 / 400 / 309 / 1 / 151 / 0 / 29 without new forbidden
edges. The remaining import cycle is KV gather versus dense allocation; complete
debug event dispatch and the other final A3 gates are still open.

The complete fixed-source strict suite passes **321 tests / 1 skip / 28 GPU
exclusions**, with real shim consumption (`a3-debug-owner-strict.log`,
`a3-debug-owner-shim.json`). Pre-commit, mypy and layering pass
(`a3-debug-owner-precommit.log`). Production and source-oracle files remain
unchanged throughout the accepted strict run.

## Step 6h shared allocation ownership

PR #1103 is `d75b5f17297fd5330a836274a7f4af176e62219e`. Actual #1028
replay is `d1519318e057d75644fa2ef35a47cbaab9090b0f`, tree
`e208f1bbe0efc2ddf9c1e8d8849280d86beef783`, equal to clean merge-tree;
37 CPU integration tests pass with 98 GPU skips. Four source-verified durable
queues are staged as `a3-step6g` on 54633.

Growing allocation belongs to workspace. KV gather and dense prefill both
consume its public allocation API, removing the last package import cycle.
A separate mechanical commit preserves the original allocation body before
the public-name rewrite. Legacy shim names resolve to the actual module and
attribute, so a patch through the old private name reaches both real consumers;
no stale function-value alias is retained. Strict patch auditing follows this
canonical binding while still requiring callable invocation from production.

The nine new cases cover initial success, OOM retry with CPU/CUDA cache-release
ordering, second OOM decline, non-OOM error propagation, real gather/bridge
allocation through the legacy patch, cache pointer reuse and alias deletion
and restoration. The initial 19-case focused selection passed its tests but
failed the suite-wide patch-consumption gate for a pre-existing dense-prefill
patch; it is retained as a failed run, not accepted as validation. The complete
strict suite is the gate. Metrics are 400 / 400 / 308 / 0 / 151 / 0 / 29.

Step 3b Qwen model validation completes on 54633 at 2026-10-09 06:35 +08:00:
FP16/E4M3 crossed with eager/graph, three requests each, all four comparisons
report equal route/token records. Evidence is `a3-step3b/logs/qwen-*-compare.log`
and `qwen.done`. Host/spec baseline issues still block complete model gates.

The complete fixed-source strict suite passes **330 tests / 1 skip / 28 GPU
exclusions**, with unchanged 813 golden traces and original calculation hashes.
Every audited patch is consumed; the legacy allocator patch reaches both real
production consumers. Evidence: `a3-allocation-owner-strict.log` and
`a3-allocation-owner-shim.json`. Production/source-oracle files stay fixed
throughout the accepted run.

## Step 6i outer prefill dispatch ownership

PR #1104 is `73b8fba61904aa631132b983d21083b026032102`. Actual #1028
replay is `55c52e03db41716cb192a1bea10f474e8f9cc6f1`, tree
`7c1827808dceb04f27308f2a6bbeca5972e4c035`, equal to clean merge-tree;
37 CPU integration tests pass with 98 GPU skips. Four verified durable queues
are staged as `a3-step6h` on 54633. Pre-commit/mypy/layering passed for #1104.

The common forward now delegates prefill dispatch to PrefillExecutor, which
owns policy, native/callback dependencies and workspace. A mechanical commit
extracts the unchanged branch body first. The ownership commit transfers it
with explicit Triton reference, comparison, small-query admission and capture
feature callbacks; no Impl receiver/import is added to prefill. Per-instance
calculation overrides still resolve when the owner is assembled.

All three decode-cache invalidations remain at their original branch positions
inside the executed prefill forward. Capture's two early returns still preserve
resident cache state. The trace observer reads real owned policy lookups and
reset calls; the calculation oracle expands the actual helper/executor body,
checks its arguments and binding, then compares the unchanged original hashes.
The initial 20-case focused suite passes all 813 golden traces and calculation
hashes (`a3-prefill-dispatch-focused.log`). Metrics tighten to
153 / 318 / 308 / 0 / 151 / 0 / 29, with no new forbidden edges. The final
150/200 line targets and remaining Spec/private/logging gates are still open.

Seven direct dispatch cases verify invalidation-before-execution at all three
reset sites, completed None results without a fallback attempt, both capture
early returns preserving resident cache, the original Triton super binding,
and refresh/isolation of feature callback overrides. Evidence:
`a3-prefill-dispatch-boundary.log` (7 passed). Pre-commit removed two unused
imports and required an explicit AST type assertion; the final pre-commit,
mypy and layering checks pass (`a3-prefill-dispatch-precommit-final.log`).
The fixed-source strict suite then passes **337 tests / 1 skip / 28 GPU
exclusions**, including unchanged traces/hashes and real shim consumption
(`a3-prefill-dispatch-strict.log`, `a3-prefill-dispatch-shim.json`).

Step 4b's first full GPU regression stopped with 1690 passes / eight failures:
the seven inherited failures plus the literal-accounting source inventory.
That checker recognized only attribute `_record_route` calls, missing the
three typed injected decode calls (41 versus the required 44). Production route
names are unchanged: all 14 affected published PR source trees have exactly
the same 44 static accounting names as #1060 when injected calls are included.
The original failed snapshot/logs remain intact. A test-only correction keeps
the 44-name requirement and validates the injected parameter's RecordRoute
type; corrected-head GPU regression is required before the queue proceeds.

## Step 6j tree visibility feature ownership

PR #1105 is `4d653b7ec13ad0c38959fc52db1199c85b8b0919`; its actual
the #1028 replay is `c0edaf9c64e1eb0a77a400f3ff25b2fdd2821769`, clean tree
`ef848e6e438b3e340a61a9a19e4bad8943978c9c`. The integration CPU suites
pass 37 tests / 98 GPU skips. Four verified queues are staged as `a3-step6i`.

Five tree visibility/metadata-contract calculations move mechanically into
spec/tree_masks.py, then expose public feature APIs. Common verification and
batch prefill receive explicit callbacks from common assembly. Generic masks
no longer owns or exports those five algorithms; old facade names resolve to
the live Spec bindings. No whole Impl receiver or new forbidden import edge
is introduced. The remaining parent-CPU-cache adapter is still in generic masks
and remains tracked for the next boundary work.

The focused suite passes 44 cases, including all 813 immutable golden traces
and original calculation hashes (`a3-tree-masks-focused.log`). Six direct cases
check sibling exclusion/window visibility, capture restoration without host
comparison, mixed linear/tree parent-copy behavior and no capture allocation,
and real batch-admission use of a legacy patch. The latter passes strict shim
consumption (`a3-tree-masks-boundary.log`, `a3-tree-masks-boundary-shim.json`).
Maximum-function measurement now considers every AST function, including names
repeated within one file; the diagnostic name map can no longer hide a larger
function by overwriting its key. The measured maximum remains 318. Metrics
reduce to 153 / 318 / 303 / 0 / 137 / 0 / 29.

## Corrected route inventory and preserved GPU evidence

The test-only correction `26db013963221a67d083a76bd28ede9765906067`
was propagated through ordinary merges into the existing Draft stack; no
published history was rewritten. Every changed tree differs only in
`test_flash_v100_routes.py`. Each updated head was actually replayed with the
pinned #1028, matched clean merge-tree, and passed 37 integration CPU tests
with 98 GPU skips. The exact 44 original accounting names remain unchanged.
All tracked source files in the new GPU candidate/host snapshots were checked
against manifests generated from the corresponding Git commits. New roots
use the -v2 suffix; only the obsolete owned waiting queues were retired.
Original snapshots, the Step 4b eight-failure log and all baseline failures
remain intact. No old failure was turned into a passing marker.

| PR | Corrected head | #1028 replay | GPU root |
| --- | --- | --- | --- |
| #1081 | `26db013963221a67d083a76bd28ede9765906067` | `fea00216224b081965cb13e2429eb2fd16c8abc5` | `a3-step4b-v2` |
| #1083 | `43f17967333b3f6b309533c1ad0e75a0c0b5f3ec` | `cd88482803dc07e3ae7feeca805df47350fb3717` | `a3-step4c-v2` |
| #1084 | `dd655ad13324b23836b65f6da662ce4230bbfec0` | `672bac94a9d509840b813023daa8ba9b83bbbd88` | `a3-snapshot-v2` |
| #1085 | `6a3615c0a63403ca808f16dbe3d3030a286d9d48` | `2a1d9701568ecd7cdf9aac9de197a508c0fcad0e` | `a3-step5a-v2` |
| #1086 | `b020db66b6165b7833b4bed58c2b540fd18132be` | `15e3f84becea3a211401751fa22acc58bf30979a` | `a3-step5b-v2` |
| #1088 | `aa175e1d079165da40ae06ab63d169b06cd3e899` | `585d607f998eda96c93dee1200aae1a0ef9b7ba0` | `a3-step5c-v2` |
| #1090 | `33d85a9c9e926ccc151872637418d29668e360db` | `b1c0988d1bc45b44ed027b0a0375aaad1e2c0d90` | `a3-step6a-v2` |
| #1093 | `94d0b247497d4c6621ebae8725d82325df4e6271` | `79c442a222f4e34cfad88775f966fc80b793aecf` | `a3-step6b-v2` |
| #1095 | `7ae8924e9936dd45a0935c722e444ee85d5ac319` | `24e614aaeac254781a318256c57d490c49934fb4` | `a3-step6c-v2` |
| #1096 | `541f156de2ce2f1b286565230047366557d1fec7` | `3c9112ee6ef2cda6943ee508b1ee11636db66fdb` | `a3-step6d-v2` |
| #1097 | `668bb6972f14f878276afdfa99829d369dea4ad3` | `7a7afa7bf7dd021d3bf5e49131ce2501592ffa0a` | `a3-step6e-v2` |
| #1101 | `ee9ae8a03373614261528c242e2ebd24803268ee` | `3cdb59e051f3913a4daabb1a2c79ed3749fbec0a` | `a3-step6f-v2` |
| #1103 | `0fc3e7fe66f65ddd35840a929615bf785c65069a` | `1542ef8d20ca37d64cbb773226576998570d864a` | `a3-step6g-v2` |
| #1104 | `5bbcf47ccb99489edd4d81790356c8d836304a6a` | `16d4df3ef003b368505d6fee13e80899ec008495` | `a3-step6h-v2` |

Step 4a Qwen model validation completes at 2026-10-09 07:16 +08:00:
all four FP16/E4M3 × eager/graph contracts report equal route/token records
for three requests each (`a3-step4a/logs/qwen-*-compare.log`, `qwen.done`).
Host/spec baseline failures remain separate; no complete model gate is claimed.

The complete fixed-source strict suite passes **344 tests / 1 skip / 28 GPU
exclusions** (`a3-tree-masks-strict.log`, `a3-tree-masks-shim.json`): six new
boundary cases plus the already existing 44-name inventory now included in
this CPU selection. All golden/calculation oracles and real patch consumption
pass. Pre-commit, mypy and layering pass (`a3-tree-masks-precommit-final.log`).
No production/source-oracle file changes during the accepted strict run.

## Step 6k declared feature contracts

PR #1107 is `4d892123e96e4dcca8f9d40ee012cbf6b225f4d2`. Its actual
integration with #1028 is `1c9f2fe07fc654ccfc7c5c7e6bdd574b1266eb33`, tree
`9817ffd2415f7dcf3dbfbf2661753e28d8985258`, equal to clean merge-tree.
The integration CPU suites pass 37 tests / 98 GPU skips. Four source-verified
queues are staged as `a3-step6j` on 54633.

Declared causality/window validation and its process-shared signature set now
belong to spec/contracts.py. Common assembly and VerificationOps inject the
validation callback; neither the common verifier nor shared state keeps the
model-specific contract body/record. The legacy facade resolves both the
validator and observation-set aliases to the live Spec owner. The fast path
still performs contract checks without constructing a verifier.

The first focused run passes its golden cases but fails the original contract
hash because the AST projection left a qualified state name where the existing
normalizer expects an unqualified global name. Only that projection was
corrected; the immutable hash and production body were not changed. The next
8-case run passes all calculation hashes and the five new boundaries under
strict shim consumption (`a3-spec-contract-boundary.log`,
`a3-spec-contract-boundary-shim.json`). Cases verify process-wide deduplication
across two instances/layers, the unchanged layer/causal/window/RoPE signature,
real legacy validator/set consumption, and causal/window failures before
observation or premature window evaluation. Metrics reduce to
153 / 318 / 301 / 0 / 126 / 0 / 29 without new forbidden edges.

The corrected Step 4b GPU run completes at 2026-10-09 07:31:45 +08:00:
1691 passes / the same seven inherited failures, two new passing cases and
no changed old outcomes. All 12 native outputs have max-abs 0. Designated
timing deltas are FP16 XQA graph +0.0168462%, E4M3 XQA graph +0.00916205%,
and 75T-role eager prefill +0.183613%, within 2%. DDTree eager records
+5.36524%; it is not one of the designated timing cases and no broad speed
acceptance is claimed. Evidence is under `a3-step4b-v2/logs/`:
`regression-parity.json`, `op-compare.log` and `regression.done`.
Model gates remain pending; the original failed Step 4b run stays preserved.

The complete fixed-source strict suite passes **349 tests / 1 skip / 28 GPU
exclusions** (`a3-spec-contract-strict.log`, `a3-spec-contract-shim.json`),
including the unchanged 813 golden traces, original calculation hashes and
real legacy patch consumption. Pre-commit, mypy and layering pass
(`a3-spec-contract-precommit.log`). No production or source-oracle files
changed during the accepted run.

## Step 6l dynamic feature policies

PR #1108 is `4c67986baef2a350a3dc06800ee85fa6356871ec`. Its actual
integration with #1028 is `3a8be3d1d9658500ceff7659da9fc533af625d25`, tree
`d9576079a24360a861f7e9758bb4edddfe480152`, equal to clean merge-tree.
Integration CPU tests pass 37 cases / 98 GPU skips. Four verified queues
are staged as `a3-step6k` on 54633; model gates remain pending.

Eight feature policies now live in spec/policy.py. Verification and prefill
receive individual callables; feature metadata preparation uses the same
owner directly. Common routing/debug no longer exports feature policy bodies.
The public decode partition-domain constant remains shared, and legacy names
resolve to live owner bindings. There is no configuration snapshot or new
registry: environment reads still occur on the original decision/emission
paths after executor construction. Callback injection does not receive Impl.

The focused suite passes 23 cases, including all 813 unchanged golden traces
and original calculation hashes (`a3-spec-policy-focused.log`). Metrics reduce
to 153 / 315 / 287 / 0 / 83 / 0 / 29; no new forbidden edge is introduced.
The maximum-function reduction is only shorter injected-call spelling; further
responsibility splits are still required to reach 200 lines.

Corrected Step 4c GPU validation completes at 2026-10-09 07:47:46 +08:00:
1691 passes / the same seven inherited failures, no changed old outcomes and
no added cases. All 12 native outputs have max-abs 0. Designated timing deltas
are FP16 XQA graph +0.0115837%, E4M3 XQA graph +0.0417382%, and 75T-role
eager prefill +0.708070%, all within 2%. DDTree eager records +11.1154%; it
is outside the three designated timing gates, with no broad speed claim.
Evidence remains in `a3-step4c-v2/logs/regression-parity.json`,
`op-compare.log` and `regression.done`; complete model gates remain pending.

Seven direct boundaries pass with strict shim consumption
(`a3-spec-policy-boundary.log`): post-construction switch changes, lazy disabled
partition parsing, unchanged error cause/domain, trace destination/payload
precedence and nonfatal write failure, and an old partition patch consumed by
an actual XQA operator call. The first complete strict run reports 355 passes
and one failure: two metadata calculation hashes still see the old profile
helper name. The projection now maps the actual Spec-owned helper back to its
original dependency name; frozen hashes and metadata calculations are intact.
The new optional prefill operator field is appended to preserve existing
positional construction. A fresh complete strict run is required after both
changes. Initial mechanical-import lint failures were corrected with temporary
lazy delegates; the separate ownership commit removes those delegates.

The corrected fixed-source full strict run passes **356 tests / 1 skip / 28
GPU exclusions** (`a3-spec-policy-strict-final.log`, `a3-spec-policy-shim.json`).
All immutable trace, calculation, metadata and real patch-consumption gates
pass. Pre-commit/mypy/layering pass (`a3-spec-policy-precommit-final.log`).
An independent AST comparison confirms all eight policy bodies exactly match
the parent after only explicit function/constant-owner renames. No production
or source-oracle file changes during the accepted run.

Step 4b Qwen model validation completes at 2026-10-09 07:50:55 +08:00:
all four FP16/E4M3 × eager/graph contracts report equal routes/tokens for
three requests each (`a3-step4b-v2/logs/qwen-*-compare.log`, `qwen.done`).
Host/spec baseline failures remain unresolved; no complete model gate claimed.
The current upstream heads for #1028 and #1048 were rechecked and still match
the pinned replay/reference SHAs. No frozen follow-up PR was modified.

## Step 6m process-wide decode log events

PR #1109 is `78df907297dfc8d396ad5e458952d6f56cd8042a`. Its actual
integration with #1028 is `7f4d3e6a61ae9d5351e5d8214a32f6298f833cf8`, tree
`a252bcd391af331dd9d507c9e1d51e603fe0b5c9`, equal to clean merge-tree.
Integration tests pass 37 cases / 98 GPU skips. Four verified queues are
staged as `a3-step6l` on 54633. No complete model gate or main merge claimed.

Nine decode log flags now use explicit process-wide logger keys. Existing
info_once/warning_once behavior remains unchanged when no key is supplied;
explicit namespaced keys deduplicate across changing messages, arguments and
logger instances without LRU expiry. The original outer guards remain, so
already-observed events do not reevaluate guarded log arguments. Observation
is recorded only after a successful logging call, at the original assignment
site even for an injected logger. Log levels, messages and process scope stay
unchanged. This is a rewrite without a mechanical body move.

Legacy flag names are live module views of the actual logger keys. The strict
patch auditor retains custom module setters and credits only a production
read of the matching logger key; patch setup/property reads are not evidence.
The focused 15-case strict run passes all immutable golden and calculation
hashes (`a3-decode-once-focused.log`, `a3-decode-once-focused-shim.json`). Its
flag consumption points are the real dense-cache/reference methods. Five
boundary cases pass, covering process granularity, changed payload/logger,
non-expiring event keys, scope decline, emission errors, unchanged unkeyed
behavior, and reset controls across two real attention instances
(`a3-decode-once-boundary.log`). All 22 existing logger tests pass
(`a3-decode-once-logger.log`). Metrics reduce to
153 / 315 / 269 / 0 / 83 / 0 / 20; full strict and GPU gates remain required.

The complete fixed-source strict suite passes **362 tests / 1 skip / 28 GPU
exclusions** (`a3-decode-once-strict.log`, `a3-decode-once-shim.json`). All
813 golden traces, original calculation/metadata hashes and actual patch
consumption pass. The sixth boundary launches an isolated strict auditor
and proves that a flag patched/read only by the test is still rejected as
unconsumed (`a3-decode-once-auditor-negative.log`). Pre-commit/mypy/layering
pass (`a3-decode-once-precommit-final.log`); the method-binding map needed an
explicit Callable type after optional keyed signatures diverged. No production
or source-oracle file changes during the accepted strict run.

Step 4c Qwen validation completes at 2026-10-09 08:10:48 +08:00: all four
FP16/E4M3 × eager/graph contracts agree on route/token records for three
requests each (`a3-step4c-v2/logs/qwen-*-compare.log`, `qwen.done`). The
recorder follow-up completes GPU regression at 08:07:39 +08:00 with 1692
passes / the same seven failures, one new pass and no changed old outcomes
(`a3-snapshot-v2/logs/regression-parity.json`, `regression.done`). Both PR
bodies reflect these results; host/spec baseline failures remain unresolved.

## Merge acceptance and FlashInfer-SM70 compatibility (2026-10-09)

The merge review found CI errors hidden by changed-file local mypy checks:
FlashInfer-SM70 still passed feature metadata positionally to the newly owned
builder, while its paged-route observer relied on a dynamically attached parent
method. Restore positional forwarding through the assembly boundary, declare the
parent paged-call adapter explicitly, and inject the subclass observer into the
prefill executor. The adapter invokes the owner's default operation so a subclass
`super()` call cannot recurse into its own injected override. No native kernels,
route names, candidate ordering or speculative algorithms change.

Three new tests reach the real parent adapter for positional/keyword metadata
and the real owned prefill candidate callback. The earlier FlashInfer test
stubbed the parent build/forward, which concealed this regression. The targeted
suite passed 25 cases. CI-style mypy for Python 3.12 checks all 61 source files
changed since the original main and passes. The strict trace suite and final
integration result are recorded below after completion.

Previously recorded evidence includes 813 unchanged golden traces, strict shim
consumption, 362 focused CPU passes, pinned #1028 replay (37 passes / 98 GPU
skips), exact native outputs and four exact Qwen FP16/E4M3 eager/graph contracts
through Step 5a. Step 5a's GPU regression has 1702 passes and the same seven known
failures. Later per-slice GPU matrices are follow-up evidence; no unrun result is
claimed. Host-FP8's original same-source A/A greedy mismatch remains a known
issue, not a refactor pass or an established regression. DDTree and its separate
fix #1089 remain deferred. Frozen #1061/#1063/#1064/#1065/#1066 stay excluded.

Merging to main preserves the source commits and their review history. Another
maintainer merged #956 during this delivery; its two Mamba cache-manager files
are preserved in the clean integration result, without modifying that work.

Compatibility validation: the full strict run on the repaired production source
returned 361 passes, one skip and one source-inventory failure: the new typed
adapter was counted twice as the original calculation. The oracle now verifies
the adapter's complete delegation separately and still hashes the unchanged
executor calculation against the immutable fixture. The corrected source-oracle,
FlashInfer and owned-prefill suites pass all 28 cases with strict shim consumption.
No golden or stored calculation hash was changed. CI-style mypy (all 61 changed
source files) and pre-commit pass. Final package metrics remain
153 / 315 / 269 / 0 / 83 / 0 / 20.

Main delivery: 24 campaign PRs through #1090 have been merged individually and
verified against their clean integration trees. The remaining 12 reviewed PRs
are delivered together with the compatibility correction in
[#1113](https://github.com/1CatAI/1Cat-vLLM/pull/1113), preserving their commits.
The integrated source at `93226ec38381f9811644c02b826498aa5d7f139a` passed
**70 focused CPU tests** with strict shim consumption. This includes the real
FlashInfer adapters and the concurrent main changes in #956/#1005/#1111/#1112.
Those changes were retained, not authored or independently promoted by A3.
In particular, #1112 superseded the separate #1089 proposal; DDTree remains
outside this campaign's current acceptance and development scope.

The final pinned #1028 replay is
`17089eef4539a53f2e9231775f769a79bb581413`, tree
`f3f5a36e8dd4d1823ed587c0f81ce50f3ec6541b`, equal to the clean merge result,
with **37 passes / 98 GPU skips** and unchanged native sources. Local evidence:
`a3-main-integration-tests.log`, `a3-main-integration-1028.json` and
`a3-main-integration-1028-cpu.log`. An additional 37 idle per-slice GPU retry
queues were retired; an already-active regression job was left to finish.
No deferred queue was given a passing completion marker.
