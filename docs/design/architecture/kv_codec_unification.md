# KV codec unification: implementation and qualification

This implements the storage and first operator slice of the
[KV acceleration audit](https://github.com/1CatAI/1Cat-vLLM/pull/1178).
The full acceleration migration is **not qualified or complete**. INT8 is
explicitly selected; the default cache format and FP8 numerical policies remain.
The follow-up XQA slice changes algorithm preference as described below.

## Storage contract

`--kv-cache-dtype=int8` identifies version 1 of signed, symmetric token/head
quantization. K and V independently use FP32
`s = max(absmax(x) / 127, 1e-6)` and
`q = clamp(round_even(x / s), -128, 127)`. The writer uses round-to-nearest-even
division and rounding. An invalid input row has a NaN scale so that readers
do not silently turn it into valid data. Reference reconstruction saturates
finite rows at the destination floating-point boundary.

Each physical kernel page contains four sixteen-byte-aligned regions: K
payload, V payload, K scales and V scales. They share the runner allocation
and lifetime. `KVPageLayout` accounts for all four regions; `KVCacheView`
borrows their actual addresses and byte strides. Binding never initializes
scales, including after prefix restore or address rebinding. Kernel pages
must be multiples of 16 tokens and divide scheduler blocks; supplied hybrid
padding must permit aligned subdivision. Unsupported padding is rejected
without changing the allocator's common page budget. Packed V2 members retain
their existing interleaving owner.

`int8_per_token_head` remains a distinct legacy codec: truncation, reciprocal
multiplication and inline `D + 4` storage. It is never aliased to the new format.
The cache configuration hash includes the new codec's encoding identity.
Transfer peers must also agree on layout and encoding version before transfer
support can be qualified.

## Operator and native contracts

The existing Triton writer and unified attention implement both INT8 codecs.
Their page traversal, masks, softmax and reduction remain shared. The native
scalar kernel uses compile-time `KVCodecTraits` for FP16, E4M3, E5M2 and INT8.
INT8 K scale participates per key row in QK; V scale participates per token
in PV. Old FP8 operation order is preserved. Scalar INT8 uses FP32 partition
outputs; there is no second INT8 attention algorithm or workspace owner.

Native codec support is read from the loaded binary, independently of Python
function signatures. The existing scalar ABI remains available. The new
`decode_paged_kv_fwd` ABI receives both scale planes; `decode_paged_kv_abi`
and `decode_paged_kv_codecs` advertise its contract. An old binary cannot enter
INT8 scalar through an unchecked fallback. Every fallback must satisfy its
own geometry, codec and native capability checks.

The shared CUDA reader originates in Draft #1061; its FP16/E4M3/E5M2 scalar,
packed and address calculations are checked against frozen legacy PTX.
Triton row encoding reuses the applicable extraction from Draft #1048 without
introducing its parallel registry. Main vLLM and Flash-V100 source manifests
ship the same canonical headers.

## Coverage and remaining owners

Implementation, actual native execution and measurement are separate claims.
Registration and compilation prove neither GPU correctness nor latency.
`tools.generate_architecture_reference --json` emits the declared route/codec
matrix, including missing entries; it does not relabel a missing binding as
supported.

| Component | Implemented in this slice | Remaining acceptance / owner |
| --- | --- | --- |
| Codec / allocation / views | Identities, planar layout, V1/V2 bindings, actual scale strides | Engine initialization, hybrid cache, prefix/offload and multi-engine GPU state tests / runners |
| Triton writer / attention | Common encoder and existing unified attention; component GPU checks pass | Engine-level state and task qualification / Triton backend |
| Native scalar | Same partition kernel and reduction, scale ABI, qualified fallback admission; component GPU checks pass | Latency and engine-level qualification / Flash-V100 |
| XQA / long / compact | Standard XQA bound to row scales; component GPU checks pass | Other tuning variants, compact paths and latency / Flash-V100 |
| Grouped / mixed / MTP / DFlash | Ordinary grouped FP32 shares native codec bindings; component GPU checks pass | Engine bindings, tree/sparse masks and commit/rollback / verifier owners |
| Prefill / prefix / Split-D / split-KV | FP16/FP8 reader extraction | Native INT8 binding and bounded dense conversion / prefill owners |
| Norm / RoPE fusion | No new format admitted | Common writer binding and byte compatibility / fused writers |
| QSA / host history / hot cache | No new format admitted | Representation and transport adapters / QSA owners |
| MLA / indexer | No new format admitted | Group-axis and unquantized-field codecs / MLA owners |
| KV transfer | Whole-allocation byte-copy contract tested on CPU | Peer identity, connector adapters and execution / transfer owners |

The ordinary prototype uses equal K/V head dimensions. The public Triton
backend accepts the format; Flash-V100's full backend gate continues to reject
it until prefill, grouped and writer paths are bound. The native scalar
operator uses its common public function. BF16 compute, unequal K/V dimensions,
model task quality and all-family acceleration are not claims of this slice.

## Frozen validation gates

CPU coverage checks reference rounding including ties, finite zero/extreme
rows, invalid propagation, allocation accounting, block subdivisions, packed
members, non-contiguous page restore and view rebinding. Native fallback and
old/new ABI tests distinguish advertised capability from accepted keywords.
The 547-test affected suite passes. After additional ABI, identity and padding
cases, the 516-test affected subset passes. The offline tool suite passes 17
tests, including legacy-reader PTX parity with CUDA 12.0. The actual Triton
writer compiles for SM70 in both rounding modes (2 tests).

At the foundation commit, GPU tests had not run; the XQA follow-up below records
their completed component run. They use non-contiguous pages and
lengths 15/16/17, 255/256/257 and 1023/1024/1025 tokens. Writer graph replay
changes slots and inputs while preserving untouched bytes. Attention is
compared with FP64 arithmetic over the independently encoded cache, with
`atol=0.004, rtol=0.015` frozen before candidate runs. This is an operator
arithmetic bound, not a cache reconstruction or model-quality bound.

The prior audit reproduced one existing grouped-codec AST-fixture failure in
`test_native_loading_workspace_and_scalar_long_calculations_match_parent`.
Its five stale snapshots belong to changes `495fb442a3` and `c8efae47b1`; they
are not regenerated to disguise a new regression.

No FP8 calibration, model-quality or latency comparison is claimed. These
require independently initialized/calibrated FP8 controls, frozen representative
tasks and same-condition repeated timing before candidate measurement.
35B-A3B AWQ and FP8 pure decode remain mandatory against the existing 0.0.3
baseline. Short route-hit runs do not satisfy that gate.

## Build and evidence record

- Integration base: `onecat/main`, `be3722bc095b9792bf9833068595f0b1ef70ad37`.
- Owned branch: `codex/v100-kv-unification-20261011-070157`.
- CPU runtime: Python 3.12, Torch 2.10.0+cu128, Triton 3.6; no GPU execution.
- Native build: CUDA 12.8.93, Torch 2.10.0+cu128, SM70, normal
  `setup.py bdist_wheel`, `MAX_JOBS=2`, owned source/build/cache paths.
- Flash-V100 wheel SHA256:
  `8e97d8169c2143683f25400fc7957fb9c9390c5bce2300f5433a3bf828c4da69`.
- Native extension SHA256:
  `c00dc2ccbd900c14da07b58dadd549bf40fd1d2f41e383ba8da0967bad4ba285`.
- Extracted wheel imports its public package in a fresh process without
  `LD_PRELOAD` or `LD_LIBRARY_PATH` and advertises the INT8 ABI. ELF inspection
  finds no private dependency; loaded dependencies are standard Torch/CUDA
  libraries. This is a load test, not GPU execution.
- Retained build/load evidence: task directory
  `/home/ymzx/kv-unification-20261011-070157` on the validation host;
  `build.log`, `artifact-check.json`, `artifacts/` and `source/`.
- CPU logs: `/tmp/1cat-kv-unification-{cpu,contracts,tools}.log`.

During the foundation slice, the V100 host could not initialize GPUs: kernel driver 580.173.02 and
user-space NVML 580.178 disagree. The user authorized continuing development
while the host is repaired. No GPU service, driver or unrelated process was
changed. The first CUDA build rejected `__half2_as_uint`; a portable union
bit representation replaces it and the normal rebuild passes. Main vLLM's
full CUDA wheel, GPU correctness, remaining bindings and performance
qualification are still outstanding.

## XQA binding and scalar priority follow-up

The follow-up branch `codex/v100-kv-xqa-bindings-20261011-075422` starts from
foundation commit `9630019528d54e3d07ef891f0d8254aa8ffcd8a4` (Draft #1179).
It binds both INT8 codecs to the existing XQA kernel, through the same native
codec visitor, scale view and Python workspace planner. It adds no INT8-only
attention kernel, scheduler, model name or weight-format gate.

XQA loads signed bytes as exact half values, applies K row scales after QK,
and applies V row scales to FP32 probability tiles after accumulating the
unscaled softmax sum. FP32 partition outputs use the existing reduction.
Inline legacy rows can be only four-byte aligned, so their packed reader uses
two aligned word loads. Existing FP16/FP8 packed-reader PTX remains unchanged.
INT8 invalid-row markers propagate through the softmax clamp; dedicated GPU
cases check both K and V scales and isolation of the other KV head.

The versioned `decode_paged_xqa_kv_fwd` binding advertises codec support from the
loaded binary. Scalar and XQA share `bind_decode_kv` for entry-point selection,
scale validation and accumulator storage. Missing INT8 support is rejected
before workspace allocation or capture. Legacy public signatures remain.

The requested XQA-first policy removes the default G4, E5M2 and small-query
length preferences for scalar, including the native E5M2 short-context scalar
launch. Explicit old threshold overrides remain available. Disabling scalar
now leaves XQA eligible; fallback is considered only after actual XQA admission.
Native loading and the initialized decode-availability flag also treat XQA
independently: a missing or incompatible scalar binding cannot disable a usable
XQA binding. The CPU matrix exercises enabled/disabled and present/missing
combinations for both operators, including explicit rejection/fallback when
neither is usable. These initialization, executor and route checks pass
**523 tests** (`/tmp/1cat-kv-xqa-availability.log`).
Existing layout, mask, E4M3 numerical qualification and native-availability
guards remain. Compact scalar tails are reached only through scalar fallback.
This policy is a Draft candidate, not a performance-qualified release default.

Correctness implementation range for the new XQA binding: FP16 queries, D=256,
GQA=4/6/8, ordinary full causal decode, arbitrary positive-stride K/V and FP32
scale views from the declared codecs, and partitions 256/512/1024 tokens.
The new planar allocator supplies pages in multiples of 16 tokens. Existing
FP8-specific 64/128/896/1664 tuning, compact-tail, staged-PV, grouped verifier,
prefill and full-backend INT8 admission still require their own binding work;
this slice does not relabel those gaps as accelerated support.

Initial GPU evidence is now available after the validation host recovered:
CUDA 12.8.93, Torch 2.10.0+cu128, V100-SXM2-32GB GPU 0, no model or service.
The normal Flash-V100 wheel ran both INT8 codecs at GQA=4/6 and 17/257 tokens,
with shuffled pages and changed-length graph replay (eight output comparisons).
The frozen arithmetic limits passed. CUDA profiler events identify
`flash_attention_decode_xqa_tc_partition_kernel_256_wide`, independently of
Python route names. This used the public package without preloading Torch,
private libraries or import-time registration patches. The first trace check
mistakenly expected the launch-helper name; correcting the expected kernel name
required no kernel or tolerance change.

Retained evidence on the validation host:
`/home/ymzx/kv-xqa-bindings-20261011-075422/iterations/first/` contains the initial
wheel and `native-probe.json`; `native-probe.py` records the exact probe.
The final wheel repeats these eight output checks and passes four additional
invalid-scale/head-isolation checks. `invalid-before.json` reproduces the K-scale
NaN suppression in the first wheel; `invalid-after.json` records the correction.
Final wheel SHA256:
`437cac31277e2f907e487df7d268045db5d03212c4df092ebb62247cebc44fec`.
Native SHA256:
`f95d934970e42b83b96c562718906897bfa8a4ca78dd4dae02dc7773a6a759e6`.
`artifact-check.json` records public-package-first loading, both codec ABIs and
standard Torch/CUDA dependencies, without private library overrides.
The task's normal main-package CMake build and final-wheel component suite are
tracked separately. The initial combined test collection failed because the
owned main source had no `_C` extension; no private extension was substituted.
The normal CMake `_C` build and component installation subsequently completed
on SM70. Installed `_C.abi3.so` SHA256:
`d7660d3d6d477988a3deebb5fdcac8c9dafe23ade34ffc517690376c9436fe0a`.
This builds the main native component, not the full main-package release wheel.

The exact committed `test_int8_kv_codec.py` and
`test_flash_v100_bound_runtime.py` suites now pass **104 tests** on GPU 0,
using the final Flash-V100 wheel and that normally installed main component.
The tests cover both INT8 writers at D=32/64/80/128/256; Triton and scalar
attention at D=128; XQA at D=256; the page/partition boundaries listed above;
GQA=4/6/8, two KV heads, one/two decode rows and partitions 256/512/1024;
changed queries/pages/lengths/scales at graph replay; negative write slots;
invalid-scale/head isolation; and configured FP16/E4M3/E5M2/INT8 calls,
environment changes and independent workspace/capture owners.
`gpu-components.log` retains the result. Eight additional native scalar
comparisons at D=256, GQA=4/6 and 17/257 tokens also pass
(`scalar-probe.json`). No model or inference service was launched.

CPU evidence: the final 529-test route/binding/executor/lifecycle suite passes,
including scalar disabled and explicit-scale forwarding. Four legacy-reader
compilation checks also pass. The historical whole-method AST test
already fails on eight unrelated entries at audit commit `ff85ba8341ba`.
These entries remain unchanged. Three intentional KV/priority changes are
covered by executable contracts instead of the old implementation hashes.
Logs are `/tmp/1cat-kv-xqa-{routing,executor,composition,reader-parity}.log`;
`composition-{base,head}.log` retains the baseline comparison.

An exploratory operator comparison freezes its generated Gaussian inputs and
independently calibrates FP8 E4M3 K/V tensor scales before candidate execution.
At GQA6/Hkv2/D256 and 257/4097/32769 tokens, INT8 reconstruction MSE is about
14.5 times lower on those fixtures; attention arithmetic meets the frozen
operator bound. This is not representative model-task evidence. Initial FP8
timing repetitions drift by roughly 9–13% at two lengths, so the observed INT8
latency advantage is diagnostic only and does not pass the repeatability gate.
The exact probe, fixture hashes, protocol and raw samples are retained in
`decode-comparison.py`, `decode-comparison-protocol.json` and
`decode-comparison.json` in the same task directory.

No representative task-quality result, latency parity or 35B pure-decode result
is claimed here. The broad acceptance matrix above remains open until those
measurements and remaining native bindings are complete.

## Shared grouped FP32 bindings

The next slice binds FP16, E4M3, E5M2 and both INT8 storage identities to the
existing compensated grouped attention kernel. It keeps the old E4M3 public
entry point as a compatibility wrapper around the common implementation.
There is no INT8-only grouped algorithm, row planner or workspace owner.
The new public entry is `flash_attn_grouped_fp32_paged`; its native
`grouped_fp32_paged_kv_abi=1` advertises the actual installed codec bindings.

Grouped attention applies K scales to each key column after QK, and V scales
to probabilities after accumulating the unscaled softmax sum. The existing
compensated probability product and FP32 partition merge remain shared.
V scales are applied only after the causal-row mask, preventing an invalid
future token from contaminating an earlier row through zero times NaN.
Invalid K rows keep a live NaN partition rather than being dropped as empty.
Zero-length rows and unrelated requests/heads retain their exact state contract.

The public binding helper is now `bind_kv`: current native ABIs receive scale
arguments for every codec, including `None` for tensor-scaled/unquantized
storage. Only old binaries use the legacy signatures. The scalar, XQA and
grouped native instances are generated through the same codec visitor.
Storage dtype, vector alignment and scale granularity come from the codec
contract; row scaling does not imply signed-byte storage. Grouped FP32
declares its own FP32 workspace requirement. Runtime owners copy each
algorithm's versioned capability metadata without another algorithm list.

The implemented ordinary grouped envelope is FP16 Q/output, D=256, GQA6,
q2..8 for one request or request-major q8 for 2..16 requests, any positive
local KV-head count, pages in multiples of 16 tokens, and at most **266240
tokens per request's block-table capacity**. Row lengths are authoritative;
indices and positive lengths must remain within that capacity. Payload
vector alignment is 16 bytes for FP16, 8 for FP8, and 4 for signed INT8.
FP8 explicit-row groups retain the unpaired reader for valid eight-byte
strides, including an eight-byte-offset head inside a multi-head cache.

Correctness measurements use V100 GPU 0, CUDA 12.8.93 and Torch 2.10.0+cu128:
the generated codec matrix tests shuffled pages, q2/q5/q8, B1/B2/B16, one/two
KV heads, page16/page32, 17/129/255/257/1025/4097-token cases and graph replay
with changed lengths, pages, queries and scales. Each declared codec is also
checked at the 266240-token boundary and neighboring lengths. Five codec
instances have observed grouped CUDA kernel events, independently of route
names. Separate cases cover causal NaN isolation and two runtime owners with
different codecs, independent captures and release.

The frozen grouped arithmetic limit is the existing relative-L2 bound:
`1.02 * FP16_output_rounding_floor + 2e-6` against FP64 over the same
quantized cache. It is not the decode bound or a model-task quality gate.
The final normal wheel passes 248 component checks: 43 new grouped cases,
96 existing E4M3/grouped/QSA regressions, 104 writer/Triton/decode/runtime
cases and five E4M3 scalar cases. A further 16 workspace checks pass, including
graph address retention and switching between inference and ordinary calls.
The 22 CPU binding cases and four legacy-reader PTX/source-packaging checks pass.
Wheel SHA256:
`ff681b7a369cd7e6ade8a7c931b3cbc7b698616f550a92c87a872d086947c36a`.
Native SHA256:
`388aa79c46aae6e3b38377cb680b77ca20fefa36c691e57df14faaf73d5016f5`.
Fresh-process public-package loading succeeds without private library overrides.

The first combined run exposed cached decode scratch allocated inside inference
mode and later mutated outside that scope. Allocation now creates ordinary
tensors regardless of the caller's scope, retaining the same cache keys and
capture addresses. The retained earlier wheel reproduces the failure for FP16
and FP32 scratch. The first added regression test mistakenly crossed two
partition-capacity buckets; correcting it to reuse one bucket needs no runtime
change. The old E4M3 precision-revision error remains checked before binding.

Retained evidence is in
`/home/ymzx/kv-grouped-bindings-20261011-084138` on the validation host:
`test_sm70_grouped_kv_codecs.py`, `gpu-codecs.log`,
`gpu-grouped-{owners,capacity}.log`, `gpu-legacy-regressions.log`,
`gpu-bound-runtime.log`, `artifact-check.json` and normal wheel build logs.
`gpu-final-suite.log` preserves the initial combined failure;
`gpu-workspace-suite.log` records the final wheel's 248 passing component cases
and 14 workspace cases before correcting the test's capacity buckets;
`gpu-workspace-final.log` records all 16 corrected workspace cases passing.
`workspace-before-reproducer.log` retains the same corrected test failing with
the previous wheel. `build-workspace.log` records final packaging.
CPU evidence is `/tmp/1cat-kv-grouped-contracts.log`.

This qualifies component arithmetic/state for the measured cases. The old
E4M3 engine wrapper uses the shared implementation; FP16/E5M2/INT8's new
generic grouped entry is not newly selected by the engine in this slice.
Full-backend INT8 admission, sparse/tree INT8 bindings, prefill, fused writes,
transport, task quality and latency gates remain open. No grouped speedup or
end-to-end engine acceptance is claimed.

## Requested 27B NVFP4 / INT8 / DFlash2 service check

The 2026-10-11 service request uses QUASAR Qwen3.8-27B NVFP4, TP4 on four
V100-SXM2-32GB GPUs, FP16 compute, planar `int8` target KV, FP16 draft KV,
DFlash2 with seven probabilistic drafts, prefix caching and CUDA graphs.
The context capacity is **262144 tokens**, not a requested prompt length.
The requested first measurements are ordinary 4096/32768-token inputs, cold
and prefix-hit requests, with prefill and target verification reported
separately. The retained launch uses 8192 batched tokens, four sequences,
0.80 memory utilization, 2048-token KV blocks and 8192-token Mamba blocks.

At source `5944c302ff5536c3b6bee41cdae562a57581e758`, this is **not yet a
runnable full Flash-V100 INT8 service**. The GPU driver is functional
(580.178.04); the remaining issue is engine integration, not GPU access.
The earlier 264 passing component checks do not establish model execution.

The first CLI launch exposed the component-only build's missing
`_C_stable_libtorch` library during DFlash2 architecture inspection
(`_C.silu_and_mul` was absent). The normal CMake targets
`_C_stable_libtorch`, `_vllm_fa2_C`, `_moe_C` and `cumem_allocator` were then
built and installed into the owned source runtime. No private library was
copied or preloaded. This extends the normal component build; it is still
not a full main-package release wheel.

With those additional libraries installed, the same 264 GPU component and
workspace checks pass again, with no skips or failures (24.17 seconds).
The report and JUnit output are in the `components-recheck` artifact subdirectory.

After installing the engine dependencies, the actual TP4 engine configuration
constructs successfully and its native capability probe has no load error.
DFlash2/GDN switches, Q8000 prefill and CUDA graphs are configured, but
grouped, long-context and compact-tail profile entries still report
`kv_dtype` for INT8. Those configured switches are not proof of execution.
Fresh backend initialization loads the native operators and admits FP16,
E4M3 and E5M2. INT8 has native decode operators but its full-backend admission
is false; a forward call is explicitly rejected before attention execution.
The engine grouped binding table still contains only FP16/E4M3, and the
prefix-prefill bridge still accepts only E4M3/E5M2.

Consequently there are **no model prefill, verifier-round or decode latency
measurements** from this attempt. Do not promote the codec by deleting the
backend guard. The next execution prerequisite is to connect the borrowed KV
view and row scales to the common prefill/verification flows and their native
bindings, retaining the mask, state and fallback contracts above.

Retained artifacts on the validation host are under
`/home/ymzx/kv-int8-dflash2-20261011-1157`: `manifest.json`, `server.log`,
normal component build/install logs, `configured-acceleration.json`,
`backend-capabilities.json` and their probe scripts/logs. A diagnostic harness
mistakenly exposed one GPU for a TP4 configuration check; that failure is
retained as `harness-one-gpu-config-preflight.log`. Restoring the declared
four-GPU visibility makes configuration succeed without a source change.

## Engine integration and repeated prefill/decode audit

The subsequent [path-by-path inventory](kv_fastpath_inventory.md) pins source
`19d43870059472a794b86d416c6de73e30003f74` and separates native variants,
engine reachability, existing FP16-only boundaries and missing INT8 bindings.
It also records the CUDA cache writer, GLM sparse MLA and V2 implicit draft
default gaps that are not visible in a top-level Flash-V100 route name.

This next slice is based on `dea7beec35eb1f281864cbd6a03fb6bb0066b061`.
The audit follows both engine entry → selector → native binding → fallback
and native FP8 implementation → public wrapper → engine caller. An advertised
codec, a selected route and an executed kernel are separate evidence. Historical
FP8 names remain compatibility names, not a complete capability inventory.

The engine borrows one `KVCacheView` for decode, prefill, mixed batches and
verification. The view supplies the physical payload and scale strides to the
existing algorithms. Shared grouped, FA2 long-context and compact-tail bindings
use the same codec visitor as scalar/XQA. Norm/RoPE cache writers use the common
encoder. No model name, weight format or TP whitelist was added for INT8.
Prefill partial bindings retain their native capabilities when wrapped with
engine policy. Both candidate and final scalar fallback check the installed
binary; a missing INT8 binding cannot silently interpret signed bytes as FP8.

The public package now exposes native codec bindings for ordinary paged prefill,
BHMD, split-KV, BFLA and WMMA decode. INT8 K row scales are applied after QK;
V row scales multiply probabilities after the unscaled softmax denominator.
The original FP8 arithmetic remains unchanged. The existing split-KV merge
grid also needed its own query tile size: reusing the partial kernel's larger
tile count left rows unwritten beyond a tile boundary.

The first INT8 prefill candidate converted resident KV to FP16 before the old
FP16 algorithm. This failed the frozen `atol=0.004, rtol=0.015` output gate:
maximum errors were about 0.047/0.064, and 0.5–0.7% of elements failed. A paired
diagnostic reproduced the candidate exactly using the existing FP16 entry;
early scale multiplication and FP16 rounding were responsible. The failed
candidate was replaced by native codec reads, without widening the bound.
Resident INT8's dense bridge remains disabled pending a precision-preserving
adapter. Its absence is still an acceleration coverage gap, even though native
paged prefill works. Uncached prefill can use the original dense Q/K/V tensors.

Repeated audit findings and responsibility:

| Family / boundary | This slice | Remaining closure requirement |
| --- | --- | --- |
| Ordinary scalar/XQA and mixed decode | Shared payload/scale view, actual ABI admission, XQA before scalar | Measure the engine; inspect native events because an XQA wrapper can launch scalar internally |
| XQA wave partitions, batch/context, dual CTA, page reuse | Basic codec bindings exist | `flash_decode_paged.cu` and `routing.py` retain E4M3/E5M2 tuning predicates; INT8 lacks parts of those plans. LUT conversion is format-specific; partition/reduction algorithms should be shared |
| Grouped q2..8 / request-major q8 | Common binding and explicit row lengths | GPU engine/state evidence; legacy E5M2 q16 verifier still needs common binding |
| Long-context grouped / compact tail | Common FA2 binding, engine-owned scratch, codec-aware V2 graph admission | New artifact GPU evidence, valid range interiors/boundaries and graph replay; scalar stays a fallback |
| Native paged prefill / split-KV / BHMD / WMMA | Shared codec implementation | Broader engine execution and paired latency; native BFLA all-ones masks do not qualify sparse-mask generation |
| Resident prefix dense / Split-D / v37 / 79t / splitkv3 | FP8 paths retained; common bounded conversion exists | `dense_prefill.py` plus FA2 dense providers need a numerically qualified row-scale adapter; deleting their FP16 guards is insufficient |
| BFLA and anchored windows | Existing format restrictions retained | BFLA mask pooling and native mask tile geometry need a common contract; anchored windows were FP16-only for FP8 as well |
| QSA page4, host history and hot cache | Existing FP16/E4M3 paths preserved | `models/qwen4_exp/nvidia/qsa.py`, host KV and sparse-page4 native entry need representation-aware scales, indexing and transfer validation |
| MLA / compressed sparse / indexer | Existing latent, RoPE and grouped scale layouts preserved | DeepSeek sparse/indexer writers and readers need their own codec contracts; ordinary token/head INT8 is not interchangeable with their packed G64/exponent-scale representation |
| Fused Norm/RoPE/KV | Common encoder and address rules; 10 component GPU byte/replay cases pass | Engine execution and existing model-side fusion qualification, which still contains old TP/weight/model gates |
| Draft/target defaults and global predicates | INT8 included in common quantized-KV and token/head predicates; V1 DFlash/DDTree implicit draft retains FP16 | V2 DFlash implicit default still recognizes only FP8; explicit mixed-codec engines and draft acceptance/rollback quality remain unmeasured |
| CUDA Graph / prefix / sleep / offload / transfer | Data and scales share physical allocation; view rebinding is read-only | End-to-end restore, reorder, slot reuse, transport identity and V1 graph tuning still require execution evidence |
| Diagnostics | Actual native codec manifests are reported alongside historical names | Configured capability is not observed execution; task latency and quality cannot be inferred from this report |

The second pass found that `utils/torch_utils.py` had omitted new `int8` from
both quantized-KV predicates although the runner's quantization-mode enum
included it. This affected common consumers such as post-sleep initialization.
Both predicates now agree with every declared codec/alias, with a generated
codec test. The old DDTree implicit draft rule also recognized only FP8;
row-scaled target caches now retain the same FP16 draft default. Explicit
draft cache choices are preserved.

Current measured correctness scope: V100-SXM2-32GB GPU 0, driver 580.178.04,
CUDA 12.8.93, Torch 2.10.0+cu128, normal Flash-V100 wheel build and fresh process.
Before the candidate, 15 FP16/E4M3/E5M2 public prefill cases passed. The native
candidate passes **55 GPU checks**: 15 byte-exact conversion/replay cases,
25 paged/BHMD/split-KV/BFLA/WMMA cases across five codec identities, and 15
geometry/window/split-merge cases. These include shuffled physical pages,
changed graph lengths, D64/D128/D256, one/two KV heads, q1/q8/q17/q33,
noncausal window attention and a 33-query split-KV merge with NaN-initialized
output. INT8 prefill keeps the frozen output bounds above. These measurements
do not establish task quality, engine throughput or full sparse-mask support.

Wheel SHA256:
`6ecd5ac9ba251cde4e76b2975ba00e70ea5bb2693ebfdeb0608ac1c06eeb0648`.
Evidence under `/home/ymzx/kv-engine-bindings-20261011-123817` on the validation
host includes `build-public-r5.log`, `gpu-prefill-fp8-baseline-r3.log`,
`gpu-prefill-r5.log`, the rejected `gpu-prefill-r3.log`, and
`prefill-r3-numerical-diagnosis.log`. Separately, `gpu-fused-r6.log` records
**10 passed** for the target partial-RoPE and draft NeoX-RoPE writers across
five codec identities, including byte compatibility and changed-input graph
replay. FA2 long/tail and engine GPU qualification remain pending; these fused
component results do not establish model-side route execution.
The normal main-package build also finished `_C`, `_C_stable_libtorch`,
`_vllm_fa2_C`, `_moe_C` and `cumem_allocator` in the owned build directory.
Build completion is not execution evidence; no model service was started during
the path-inventory audit.

CPU checks include 45 acceleration-report cases and 33 targeted codec/default
cases; 518 route/engine/declaration cases and 28 ownership/size regressions pass.
A broader DDTree configuration run has four pre-existing failures
(two missing observability fixture fields, two old async-policy expectations),
reproduced unchanged on the parent worktree. The historical scalar-tail AST
snapshot also predates this slice and does not describe the current parent.
The broader trace/metadata check reproduces eight failures on the parent in
this CPU environment (Triton stub APIs and the old scalar-first trace). Its
one new function-size regression was fixed without increasing the ratchet.
Retain these baseline failures separately from new numerical/state regressions.
No 27B service timing, representative task-quality gate, latency parity or
35B AWQ/FP8 decode acceptance is claimed at this checkpoint.

## Shared native writer and XQA execution plans (2026-10-12)

This follow-up starts from Draft #1182 at `9a787d03e6`. Ordinary CUDA cache
publication now accepts the codec view's two FP32 scale planes in the existing
`reshape_and_cache_flash` entry. Its eight-argument compatibility call remains
valid. ABI capability is checked once per backend instance; an older binary
uses the existing, qualified Triton row writer. Decode admission no longer
requires an unrelated converter or paged-prefill binary.

Native and Python XQA planning use common geometry and aligned byte-storage
requirements. Wave/merged-wave, device-selected partitions, dual CTAs,
batch/context routing, partition page-ID reuse and paired byte reads admit
all compatible ordinary byte codecs. FP8 decoding/LUTs and partial types remain
codec traits. Plan revision 2 removes the native E5M2 short-context scalar
launch; a qualified XQA request does not enter scalar even with an explicitly
large historical scalar threshold. Old binary policy checks remain intact.
V1/V2 now share the implicit DFlash draft-KV resolver; explicit choices win.
Graph buckets and policy hashing include the common byte-codec tuning inputs.

GPU qualification uses V100/SM70, CUDA 12.8.93, Torch 2.10.0+cu128,
driver 580.178.04 and the task's own normal CMake components and Flash-V100
wheel. It does not load sidecar kernels or preloaded DSOs. Results:

- Writer: **21 passed**, including signed INT8, legacy INT8, both FP8 codecs
  and FP16; D64/80/128/256, multiple heads, non-contiguous source rows,
  non-contiguous slots, untouched physical bytes, invalid rows and replay.
- XQA plans: **44 passed**, including GQA4/6/8, B1/2/4/5, Hkv1/2,
  page16/32/784/800/1616 (including Hkv1 five-row planning), wave and merged-wave threshold neighbors,
  non-contiguous pages and changed graph inputs. The short INT8 test records
  the CUDA XQA partition kernel and rejects a scalar partition launch.
- Affected conversion/grouped/long/tail/engine-component regression:
  **90 passed**, including graph replay. This is a backend component run,
  not a complete model-engine or task-quality acceptance.
- CPU route/admission/hash subset: **545 passed**. Graph/lifecycle/workspace
  subset: **28 passed, 2 GPU-dependent skipped**. The four shared draft-default
  cases pass. These do not substitute for the GPU or model gates.

Writer revision 3 initially failed on legacy inline rows with a misaligned
vector store. Revision 4 fixed addressing but exposed three byte mismatches:
Triton's historical `div.full`/reciprocal and CUDA RN division differ at
truncation boundaries. Revision 5 preserves that legacy operation sequence;
the new codec retains RN division and nearest-even rounding. Both failures
remain in the task logs; numerical limits were not loosened.

Retained logs are `gpu-writer-r5.log`, `gpu-xqa-only-r3.log` and
`gpu-engine-regression-r5.log` and `gpu-shared-boundaries-r5.log` in the
validation task directory. The latter adds the five Hkv1/page1616 codec
cases and a positive slot-capacity boundary. Flash-V100 R3
wheel SHA256 is
`b71dec051db2e87b8e3aaaace64b61942ff12678ba696f8cf54d560014be06a3`.
The main components compile from source, including `_C`, `_C_stable_libtorch`,
`_vllm_fa2_C`, `_moe_C` and the allocator. A final normally installed main wheel
and release-artifact acceptance are still required.

**The full migration remains incomplete and experimental.** Legacy grouped
q16/two-pass/staging, the independently scaled resident-prefix dense adapter,
QSA/history, MLA/indexer and plane-aware transfer remain open responsibilities.
The above GPU evidence closes component correctness cases, not all latency
or state-semantics gates. The 27B NVFP4/TP4 FP8 control is independently
initialized with scale calculation, DFlash2 seven drafts, FP16 draft storage,
prefix caching and graphs, and capacity 262144 tokens. Its latency corpus is
frozen before candidate model runs. The initial reset-prefix request received
HTTP 404 because development API routes were disabled; no model request ran.
The control is restarted with the explicit normal read-only route/scale
extension and development API. Calibration quality and cold/hot repeatability
remain unaccepted; no speed or quality improvement is claimed.

The live scale audit found all Q/K/V values at 1.0 on every TP rank,
with no pending calculation: the existing hybrid-model guard disables
`calculate_kv_scales` at configuration initialization. This control is
**not an independently calibrated precision baseline**. Preserve the guard;
unit scales or a short request cannot prove INT8 precision improvement.
A separately qualified calibration/checkpoint control remains required.

## Shared grouped execution plans (2026-10-12)

Owned follow-up to the shared writer/XQA stage (`c24a1ef562`). The existing
Tensor Core grouped kernel now accepts q9..16 through the codec visitor and
supports both one-pass and two-pass FP32 plans. Both passes use compensated QK;
the probability residual retains the established FP32 numerator/max/sum and
relative-L2 contract. Query packing, masks, partitioning and reduction are shared.

The plan derives fixed-stride page1648/page3296 admission and paired reads
from payload geometry, with scale addresses kept independent. Split-local page
IDs are staged with the same capacity fallback for every codec. Valid eight-byte
views and the legacy inline layout retain their qualified generic reader instead
of truncating offsets to sixteen-byte boundaries. Original native entry points
and stale-binary q8 capability remain available; q16/two-pass require the new
actual plan entry. Public metadata propagates the actual maximum query rows to
the engine admission, without a model/weight/TP allowlist.

V100/SM70, CUDA 12.8.93, Torch 2.10.0+cu128 and driver 580.178.04 results:

- New grouped schedules: **50 GPU cases passed**, covering five codec identities,
  one/two-pass, q9/q16, Hkv1/2, B1 and B2 q8, non-contiguous physical pages,
  zero-length rows and changed-input graph replay. The numerical bound remains
  `relative L2 <= 1.02 * FP16 rounding floor + 2e-6`.
- Address/native-execution boundaries: **5 GPU cases passed**, including four
  byte-codec eight-byte offsets and a profiler-confirmed q16 INT8 native
  page1648 grouped instance with no decode kernel. The first test invocation
  omitted required output/scale arguments; the corrected five cases were rerun
  separately. No kernel or numerical gate changed to fix those test calls.
- Existing grouped E5M2/E4M3 entry regressions: **94 GPU cases passed**, preserving
  original bitwise per-request/fixed-layout checks and precision-qualified tests.
- Codec/max-row admission: **51 CPU cases passed**. The wider verifier subset
  passes 68 cases and reproduces two inherited policy-reader fixture failures;
  the historical long-helper AST snapshot remains an inherited failure.

The normal Flash-V100 R2 wheel SHA256 is
`5c1641aa08802e48ec2cae8c57074fc9ec8d539abadc1347ef84be49f3a8a955`.
Unchanged main-package native components are installed with their normal CMake
component install; this development runtime is not final installed-release
acceptance. Retained artifacts: `gpu-grouped-plans-r2.log`,
`gpu-grouped-boundaries-r2b.log` and `gpu-grouped-legacy-r2.log`.

**S03 still requires complete engine/state and latency qualification.** These
component results do not close the resident-prefix dense adapter, QSA/history,
MLA/indexer or transfer responsibilities. The benchmark-only range collector
runs on a separate eager FP16 reference, excludes invalid writer slots, exports
TP-local Q/K/V ranges and never mutates captured scales. Its frozen calibration
inputs and normally loaded checkpoint scales must qualify the independent FP8
control before any precision-improvement claim.

## Resident-prefix payloads and late scale operands

This slice starts from `adc90287802d8fc8dcfb87888dba01121f2fab62`.
The ordinary INT8 adapter expands signed bytes exactly to Half payloads and
retains K/V scales in independent FP32 views. It does not reconstruct scaled
Half KV before attention. The existing bridge workspace owns both planes;
its capacity growth, capture preparation and allocation-failure behavior remain
shared with FP8. Existing FP8 consumers retain their original arithmetic order.

Dense, paged and split-KV3 Split-D now include one source core. Their row-scale
specialization applies K scales to FP32 QK before masking and softmax, and V
scales to the numerator while preserving the unscaled denominator. v37 uses
the same operand contract, tail core and workspace owner. Original six-argument
native entries remain valid; optional planes require the loaded binary's ABI 1
probe. Missing capabilities cannot send raw payloads to an old Half consumer.
This is an operand binding, not another INT8 executor.

A single rounded Half probability product failed the frozen attention gate:
five of 589824 checked elements failed in a 384-query INT8 case, with maximum
absolute error 0.00838. The candidate was rejected. The admitted specialization
retains compensated high/residual probability products and FP32 partials;
the `atol=0.004, rtol=0.015` gate was not loosened. Legacy no-plane arithmetic
is unchanged.

Component evidence on the same V100 software stack:

- **97 GPU cases passed**: 46 shared dense/paged/split-KV3 cases, 26 v37
  late-scale cases and 25 existing v37 regressions. Five codec identities,
  strided planes, shuffled pages, boundary padding and changed-scale graph
  replay are included. Small-query checks cover every output row; larger
  geometries check selected rows and every output for non-finite values.
- **9 GPU implementation-stage cases passed** for both INT8 identities,
  including 1568-query tail padding and generic Split-D. The profiler observes
  the native unscaled bridge and Split-D CUDA kernel, separately from the
  selected route. These tests instantiate the attention implementation without
  loading a model; they are not complete serving or quality acceptance.
- **613 existing CPU checks passed**, plus **14 new capability/fallback checks**.
  The scale contract is read from an actual binary probe rather than inferred
  from a Python signature. If both qualified consumers decline, the bridge
  returns to the original cache view without invoking an old Half consumer.
- The public R2 normal wheel has SHA256
  `9f56c552a68bcb16d4d64a5e30dccf9130681e96f146ae719a6beab5e77a9679`.
  FA2 is built and installed through its normal component target with the
  tracked dependency patch. A complete normally installed main-package artifact
  remains required for final acceptance.

**P05 remains open for 79t and the remaining FP8 conversion-reachable optimized
variants**, followed by full serving/state, task-quality and latency gates.
Its generic fallback does not close a missing 79t binding. Other audit families
retain their separate closure requirements above.

The latency observer now reads the transferred engine policy and initialized
diagnostics owner without mutating it or reparsing ambient defaults. Route
counts describe host dispatch/capture, not graph replay. The calibrated FP8
control verifies all 192 Q/K/V scale values after capture against its independent
checkpoint before comparison. Its prefix protocol explicitly uses a 2048-token
hybrid alignment: repeated 4096/32768-token requests actually reuse 2048/30720
tokens. The earlier requests that recomputed all 4096 tokens are excluded from
prefix-hit evidence. Capacity is still 262144 tokens. These reference runs
establish the comparison setup, not INT8 task quality or a speedup claim.

## 79t shared row-scale binding (2026-10-12)

This slice starts from `eaa70863f8` (Draft #1186). Q8000 and Q8192
use the same 79t host schedule, triangular tail tasks, request/head ordering,
streams, state merge and engine-owned score workspace. Optional FP32 K/V
planes bind compile-time operand specializations; the old six-argument native
entries wrap the common implementation with absent planes. The old Half
arithmetic remains bit-exact in paired GPU checks. A loaded binary must
advertise late-scale ABI 1 and expose the corresponding entry before row-scale
operands are admitted. Missing unrelated dense/paged/split-KV capabilities no
longer hide an otherwise qualified 79t entry.

K scales participate in FP32 QK before mask/softmax, and V scales participate
in PV without changing the denominator. Row-scale QK scores and partial
numerators remain FP32. Compensated high/residual probability products share
one tiled PV traversal. FP32 scores are lazily allocated by the existing
owner and included in engine memory profiling before KV budgeting; scale-aware
tail metadata shares the existing immutable capture-lifetime map. No private
executor, model/weight/TP list or second resource owner was introduced.

Frozen numerical and performance decisions:

- Absolute Half QK scores failed 80 of 6144 checked elements (maximum absolute
  error 0.07443). A paired Half-score oracle reproduced the failure; storing
  FP32 QK scores fixed it. `atol=0.004, rtol=0.015` remains unchanged.
- A finite value-scale adversary exposed a missed prefix peak; row-scale
  prefix maxima now use the complete FP32 score traversal. Prefix/tail
  finite-value cases, changed-scale graph replay and all 8192 query rows are
  checked without relaxing the error bound.
- Fusing compensated PV and reducing its tile removed a duplicated PV
  traversal. The retained four-shape microbenchmark still costs approximately
  **1.58–1.67 times** the original 79t computation over reconstructed Half
  inputs. This is an isolated algorithm measurement, **not a calibrated FP8
  comparison or an end-to-end result**; its performance gate remains failed.
- A later centered-Half score experiment passed numerical checks but increased
  that ratio to **1.84–1.93**. It was rejected and removed. Choosing v37 instead
  does not close the missing 79t performance qualification.

GPU validation passes **30 component cases** (19 late-scale and 11 existing
79t stability cases) and **15 implementation-stage prefix cases** in separate
fresh processes. They include both INT8 identities, Q8000/Q8192, 8001/8096/8191
padding, independent strided planes, all query rows, prefix/tail finite-value
adversaries, engine-owner preparation, release and changed-scale graph replay.
Both the native scaled QK/tensor-core PV and the bridge/Split-D kernels are
observed by their profiler cases. A combined-process run passed 44 cases but
did not observe the bridge event in its second profiler; the same prefix suite
passed all 15 cases in a fresh process. That observation is retained rather
than relabeled as a complete combined profiler run.

CPU capability/owner/selection and artifact checks pass (64). The parent and candidate also
share 16 failures in the broader policy fixture: ten fixtures lack CPU device
configuration and six retain older scalar/model-selection expectations. They
are recorded separately from the affected passing checks.

The normal FA2 component was built and installed by its existing CMake target.
The retained R9 library SHA256 is
`5e52d9f7e0f53b967688a4460345fcb8b2a2fb9f32f731b11063ace89797c5c1`.
The public Flash-V100 wheel remains the preceding normal R2 artifact.
Raw GPU tests, rejected variants and alternating cost samples are retained in
`/home/ymzx/kv-79t-scales-20261012-031000/`. A complete normally installed
main-package artifact, all remaining P05 variants, full serving/state, quality
and latency gates remain required. This Draft stage does not complete P05
or the full 43-item migration.

## Non-integral shared physical pages (2026-10-12)

The first 27B TP4 INT8-target/FP16-draft serving startup from `21442955a6`
failed in `unify_kv_cache_spec_page_size`, before cache allocation or inference.
A 2048-token ordinary INT8 page occupies 1064960 bytes including its FP32
scale planes. The corresponding 1024-token draft page occupies 1048576 bytes.
Their 65:64 ratio cannot be represented by the previous integer block-size
multiplier. This is a physical allocation issue, not an algorithm or model
admission restriction.

The common planner now uses existing attention/state `page_size_padded` for
non-integral page ratios and retains each logical token/checkpoint block size.
Integral ratios retain their previous block-size behavior. Specs without an
effective padding contract still reject before allocating storage. The codec
view remains the authority for region offsets and byte strides; there is no
format string or model-specific exception in the new planner rule.

CPU checks cover 65:64 ordinary/draft/state page ratios, immutable input specs,
idempotent planning, existing rounded FP8 layouts, and V1/V2 runner-owned
views. Cache restore checks retain payload, scales and untouched page padding.
The failed startup is retained in
`/home/ymzx/kv-79t-scales-20261012-031000/int8-engine-r1/`.
Successful serving, graph/state interactions and paired end-to-end quality and
latency remain separate gates; this allocation fix does not close those gates.

The retry passed planning and compilation, then exposed a separate owner
binding gap: V2 passed the target's `int8` spelling to the FP16 draft carrier
shape API, adding scale bytes absent from the draft spec. V1 had the same
shape call. Both runners now derive the compatibility shape/block-dimension
spelling from each spec's declared quantization mode. This does not choose a
reader codec or change algorithm policy. Mixed target/draft tests now keep the
spec fixed while varying the outer `auto`, FP8 and INT8 settings, and check
both runner views and restore behavior.

## Independent grouped indexer codecs (2026-10-12)

`AttentionConfig.indexer_cache_dtype=auto|fp8|int8` controls indexer storage
independently of attention KV. `auto` preserves FP8 and the explicit legacy
FP4 option; an explicit FP8/INT8 choice conflicts with that FP4 option. The
effective choice participates in the attention configuration hash.

Indexer codecs bind the existing E4M3 and RNE INT8 formats to the indexer
representation, last-axis groups of 128 and FP32 scales. Ordinary and grouped
cache identities differ. Their borrowed view describes page payload and
scale regions, actual byte strides and physical padding. One CUDA writer and
the existing Triton scoring, gather/cuBLAS and KPool algorithms take the codec;
there is no second INT8 indexer executor. Old FP8 native entry points remain;
INT8 requires the actual native codec ABI before writing any cache byte.

The SM70 component build passes 16 native-writer cases, 50 scorer/gather/cuBLAS
cases and 56 KPool cases generated for both compatible codecs. Coverage includes
128/256-dimensional keys with 128-element groups, FP16/BF16/FP32 inputs,
non-contiguous row strides, physical padding, independent scales, rounding ties,
invalid group propagation, changed-input graph replay, non-contiguous pages,
top-k sets, pool completion and tail state. A CUDA profiler case observes the
native codec writer kernel. The native component was rebuilt serially after
rejecting two overlapping build attempts; artifacts from those attempts do not
qualify this stage.

Three initial INT8 factored-score cases exceeded the existing `atol=0.01,
rtol=0.01` contract. Rounding the combined weighted query and decoded keys to
FP16 caused cancellation error. The shared factored computation now retains
FP32 intermediates for both codecs; all 50 cases pass the unchanged limits.
The default per-head ReLU algorithm retains its existing computation.

The DeepSeek indexer compressor forwards the same codec to the shared encoder,
but its fused execution remains unqualified. Representative sparse-engine
quality, state-lifecycle, latency and a complete normally installed wheel are
also pending. This stage does not qualify ordinary MLA, QSA or history INT8.
