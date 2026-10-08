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
| 1c: route/token/output parity tools | #1073 | Draft; host/spec model records pending | Production unchanged | 12 native cases and 4 Qwen contracts exact; 1684 passes / same 7 failures | Host/spec model gates |
| 2a: dynamic environment boundary | #1075 | Draft; focused CPU and rebase passed | Outside-config reads 119 → 41; env ratchet 334 → 306 | 1684 passes / same 7 failures; exact old outcome map | Prerequisite model gates |
| 2b: frozen construction policy | #1076 | Draft; CPU and rebase passed | Remaining 41 → 0; 41 immutable fields; env ratchet 306 → 284 | 1685 passes / same 7 failures; one new pass | Prerequisite model gates |
| 3a: per-layer decode cache | #1077 | Draft; CPU/golden/rebase passed | Private references 390 → 387 | Pending | CPU/rebase and GPU prerequisites |
| 3b: step plan and persistent metadata buffers | #1079 | Draft; CPU/golden/rebase/native gates passed | Private references 387 → 380 | 1687 passes / same 7 failures; 12 outputs exact; named timing gates passed | Model gates pending |
| 4a: explicit decode executor dependencies | #1080 | CPU/golden/strict passed | Private references 380 → 374; cycles 14 → 13 | Queued after Step 3 | Required GPU gates |
| 4b: native decode candidates | #1081 | CPU/golden/strict passed | Private references 374 → 370 | Required | Parent and GPU gates |
| 4c: outer decode dispatch candidates | #1083 | CPU/golden/strict/rebase passed | Forward 597 → 402; private 370 → 358 | Required | GPU gates |
| 1c follow-up: immutable requested workload | #1084 | CPU/golden/strict/rebase passed | Production unchanged | DFlash serialization failure retained; rerun queued | Host/spec token records |
| 5a: per-sequence prefill candidates | #1085 | CPU/golden/strict/rebase passed | Largest function 977 → 529; private 358 → 348 | Queued after prerequisites | GPU gates |
| 5b: batch prefill candidates | #1086 | CPU/golden/strict/rebase passed | Largest function 529 → 414; private 348 → 347 | Required | Rebase/GPU gates |
| 5c: debug observer | #1088 | CPU/golden/strict/rebase passed | Largest function 414 → 402 | Required | Strict/rebase/GPU gates |
| 6a: verifier ownership | #1090 | CPU/golden/strict/rebase passed | Cycles 13 → 11; model terms 169 → 158 | Required | Strict/rebase/GPU gates |
| 6b: metadata builder ownership | #1093 | CPU/golden/strict/rebase passed | Cycles 11 → 4; private 347 → 341 | Required | Strict/rebase/GPU gates |
| 6c: attention policy ownership | #1095 | CPU/golden/strict/rebase passed | Cycles 4 → 3; private 341 → 333; model terms 158 → 154 | Required | Strict/rebase/GPU gates |
| 6d: owned per-request metadata packet | #1096 | CPU/golden/strict/rebase passed | Private 333 → 332; final metadata mixin removed | Required | Strict/rebase/GPU gates |
| 6e: registered speculative features | — | CPU/golden passed | Private 332 → 330 | Required | Strict/rebase/GPU gates |
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
