# KV codec unification: implementation and qualification

This implements the storage and first operator slice of the
[KV acceleration audit](https://github.com/1CatAI/1Cat-vLLM/pull/1178).
The full acceleration migration is **not qualified or complete**. INT8 is
explicitly selected; existing defaults and FP8 numerical policies remain.

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

The V100 host cannot currently initialize GPUs: kernel driver 580.173.02 and
user-space NVML 580.178 disagree. The user authorized continuing development
while the host is repaired. No GPU service, driver or unrelated process was
changed. The first CUDA build rejected `__half2_as_uint`; a portable union
bit representation replaces it and the normal rebuild passes. Main vLLM's
full CUDA wheel, GPU correctness, remaining bindings and performance
qualification are still outstanding.
