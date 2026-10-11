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
| Triton writer / attention | Common encoder and existing unified attention | GPU execution, capture/replay and numerical limits / Triton backend |
| Native scalar | Same partition kernel and reduction, scale ABI, qualified fallback admission | GPU correctness and latency / Flash-V100 |
| XQA / long / compact | Common FP16/FP8 reader extracted | Row-scale binding, tuning and execution / Flash-V100 |
| Grouped / mixed / MTP / DFlash | Common reader extracted where shared | Row-scale binding, masks, commit/rollback and execution / verifier owners |
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

GPU tests are present but have **not run**. They use non-contiguous pages and
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

CPU evidence: the final 529-test route/binding/executor/lifecycle suite passes,
including scalar disabled and explicit-scale forwarding. Four legacy-reader
compilation checks also pass. The historical whole-method AST test
already fails on eight unrelated entries at audit commit `ff85ba8341ba`.
These entries remain unchanged. Three intentional KV/priority changes are
covered by executable contracts instead of the old implementation hashes.
Logs are `/tmp/1cat-kv-xqa-{routing,executor,composition,reader-parity}.log`;
`composition-{base,head}.log` retains the baseline comparison.

No independently calibrated FP8 accuracy comparison, task-quality result,
latency parity or 35B pure-decode result is claimed here. The broad acceptance
matrix above remains open until those measurements and remaining native
bindings are complete.
