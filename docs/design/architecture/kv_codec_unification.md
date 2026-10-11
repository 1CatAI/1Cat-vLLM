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
