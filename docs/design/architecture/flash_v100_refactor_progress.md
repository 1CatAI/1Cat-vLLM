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
| 3b: step plan and persistent metadata buffers | #1079 | Draft; CPU/golden/rebase passed | Private references 387 → 380 | Pending | Separate outcome and GPU gates |
| 4a: explicit decode executor dependencies | #1080 | CPU/golden/strict passed | Private references 380 → 374; cycles 14 → 13 | Queued after Step 3 | Required GPU gates |
| 4b: native decode candidates | #1081 | CPU/golden/strict passed | Private references 374 → 370 | Required | Parent and GPU gates |
| 4c: outer decode dispatch candidates | — | CPU/golden/strict passed | Forward 597 → 402; private 370 → 358 | Required | Parent and GPU gates |
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
