# Decouple SM70 attention schedules from KV storage

Status: staged FP16/E4M3 refactor, **not completion or INT8 admission**.
Integration line: `onecat/main`; immutable baseline:
`c4f6245f841466782752a8c3283e4727565cf17a`.

## Rechecked problem

At this baseline, `flash_attn_v100.py` has 9870 lines and
`flash_decode_paged.cu` has 6365 lines. There are 44 literal route names, but
52 `_record_route` call sites, including eight nonliteral expressions. The
backend reads 89 distinct environment names, not 78 under the explicit AST
definition used here. There are 134 exact `kv_cache_dtype` tokens and 26
if/conditional/assert predicates mentioning cache dtype; the reported 160
branches cannot be reproduced under that definition. These numbers measure
different things and must not be substituted for one another.

The repository has 210 tracked source files spelling `kv_cache_dtype`, including
73 package files, 15 native files, 76 tests and 39 benchmarks. The broad search
also examines `fp8`, `e4m3`, `_record_route`, and `uint8`: its 1361 matching files
include weight quantization, other architectures and unrelated uint8 uses.
See the [source inventory](sm70_kv_source_inventory.md),
[path matrix](sm70_kv_path_matrix.md), and
[environment inventory](sm70_kv_environment_inventory.md).

The split is visible in source: GQA6 FP16 page-784 plans a p256 envelope for
device selection of p256/p1024, E4M3 plans p64 for p64/p256/wave selection, and
E5M2 plans p256. Scalar/XQA/grouped kernels and reduction paths have additional
format gates. Two copies of `fp8_kv_utils.cuh` implement almost the same scalar
conversion. Format-dependent route names are observable compatibility counters.

The supplied 54304 decode measurements (1261 versus 599 µs at single-card
128K, approximately 28% E4M3 reduction time, approximately 200 GB/s partition
throughput) and the acceptance delta of −2.28 percentage points are the campaign's
reported motivation. They have **not** been independently reproduced on this
host. Historical microbenchmark files are available locally but do not provide
a complete source/wheel/device provenance contract. Do not present them as the
new baseline or an accepted same-host comparison. The reported 8192-chunk 75T
prefill route is a required regression gate.

## Boundaries and interfaces

Use four distinct concerns:

1. A codec owns the payload representation, scale representation and location,
   encoding/decoding, rounding and physical byte accounting.
2. A cache view owns page tables, payload/scales, strides and offsets. It carries
   the codec identity; a uint8 tensor alone never identifies its format.
3. A schedule owns phase, query lengths, GQA, head dimensions, context buckets,
   logical page geometry, partition size and reduction algorithm.
4. A capability table owns admission of a schedule × codec pair. Shape gates
   remain explicit and are checked before graph capture/dispatch.

Proposed CUDA boundary: `KVReader<Format>` and `KVWriter<Format>` with scalar and
tile operations. Tile readers receive both payload and scale view/indices, and
emit the accumulator kernel's declared staging type. Tensor/Core math and
partition/reduction code must not parse dtype strings. Instantiate the same
attention kernel with different codecs, rather than copy a whole kernel for
INT8. Storage-conversion optimizations belong to the codec; exceptional compute
kernels are separate schedule implementations with explicit capabilities.

The first implementation introduces the reader boundary in
`flash-attention-v100/kernel/kv_codec.cuh`, shared by the vendored extension and
the installed grouped/scalar sources. Compatibility wrappers keep old call
sites and launch ABIs. Scalar FP16, E4M3, E4M3 bit conversion and E5M2 arithmetic
retain their operation order; unsupported reader template IDs fail at compile
time. The original codec lives in the vendored directory so that its standalone
source distribution can build without a parent checkout. Both source manifests
include the header.

Python conversion helpers move to `flash_v100/codec.py`; metadata helpers move
to `flash_v100/metadata.py`. The original backend reexports their names. The
remaining implementation/builder, routing, writers, host ownership and
configuration are **not yet migrated**. Dense page accounting and the existing
Triton inline scale views now share a storage descriptor as described below. This is an independently revertible
first review scope, not an uncalled registry claiming the final architecture.

## Scheduling and compatibility

The eventual planner accepts a format-free `AttentionProblem` and produces an
`AttentionSchedule`. A separate admission step injects `KVCacheView.codec`.
Device sequence lengths still choose among captured shape-compatible plans;
planning never reads `.item()` during graph capture or allocates new workspaces
on replay. Workspace capacity and active length remain distinct.

Codec-independent scheduling is the final contract, but replacing E4M3's
existing partitions during mechanical extraction would change accumulation
order. First prove the old FP16/E4M3 behavior on the extracted interfaces. Then
introduce shared scheduling as an explicit numerical/performance change with
its own rollback scope. Do not hide that change in code movement.

Keep old route counters during extraction. Later count the neutral schedule ID,
codec ID, rejection reason and selected fallback. Missing support must warn once
per engine/capability reason and increment a rejection counter on every event.
Logging should identify the requested phase/shape/codec and actual fallback;
metadata-only counters must not count as executed attention. A strict
qualification mode turns unsupported accelerated pairs into an error. It is
not a new production default before compatibility testing.

The JSON matrix is a source-coverage ledger. `native`, `bridge`, and `reference`
describe implementation paths, **not admission evidence**. `review` means a
shape/format cell needs further audit. INT8 cells stay `pending` or
`upstream_only` until real acceleration and all gates pass. Tests cover all
literal and nonliteral backend route sites; additional QSA/host/write/allocator
consumers are listed explicitly. A later admission test must reject any FP8
path whose corresponding INT8 cell has not reached qualification.

## Writers, scales and allocation

Mainline already provides `int8_per_token_head`: a signed int8 payload, dynamic
FP32 scales per token/head, separate scale tensors budgeted from the KV
allocation, and Triton read/write support. This is an existing storage candidate,
not evidence that Flash-V100/QSA or fused writes support it. Reusing it avoids a
second incompatible allocation protocol; final selection depends on real KV
and attention errors and SM70 cost.

Before switching writers, preserve each current rounding contract:

- `reshape_and_cache_flash` promotes FP16 to FP32 before calibrated E4M3 scaling.
- The Python dense fallback converts the stored FP8 to FP16 **before** scaling.
- Native readers stage/scale according to their current kernel arithmetic.

These contracts are intentionally different today; changing their rounding is
a separate quality decision. One writer interface must eventually cover native
and Triton reshape/cache, DFlash fused QK RoPE, QSA prep/QSA writes and restore.
Transport normally copies encoded bytes and scales rather than requantizing.
Prefix hashes, cache sharing/restore and graph workspaces must include codec and
scale layout identity. An indexer side cache is not automatically the target KV
cache format.

Represent storage cost as payload bytes per logical token **plus** scale bytes
per token/page and fixed per-layer scale overhead. Per-tensor FP8 scales cannot
be honestly described as a constant integral per-token addition. For token-head
INT8 with symmetric K/V head dimension D, the existing layout requires
`2 * Hkv * D` payload bytes and `2 * Hkv * 4` scale bytes per token. FP16 requires
`2 * Hkv * D * 2` payload bytes. Unequal K/V dimensions, MLA packed layouts and
TurboQuant must have their own codec accounting, not inherit this formula.

Make the allocator, Mamba/page alignment, prefix restore and host hot-cache
planner consume that accounting. Logical token page size and physical byte size
are separate fields: FP16 and INT8 cannot occupy identical bytes for an identical
number of tokens. Hybrid groups need exact divisibility/alignment proofs and
padding accounting; forcing all formats to the historical FP16 page geometry
without measuring is not justified. Include scale storage in host/CPU budget.

## Dense storage descriptor migration

`vllm/v1/kv_cache_codec.py` owns the existing `KVQuantMode` enumeration and
helpers, reexported from `kv_cache_interface.py` to preserve callers. Its
`KVCacheCodec` describes payload dtype, quantization granularity, payload/token
bytes, inline scale/token bytes, head padding, and FP32 scale-view offsets.
This is the storage portion of the eventual codec, not a claim that writers,
readers and every consumer have completed migration.

`AttentionSpec`, `FullAttentionSpec` and `SlidingWindowSpec` consume its payload
and scale accounting. The existing allocator/page-unification/Mamba alignment
code therefore receives the same sizes through these specs. Triton cache shapes
and `_ensure_scale_caches` consume the same descriptor for head padding and
scale strides. Inspection corrected the earlier interpretation: these scales
are physically **inline after each head**, not separate allocation tails;
kernels receive separate typed strided views. Per-tensor FP8 scales remain
outside the token pool. Unequal K/V dimensions and NVFP4 retain their existing
formulas; specialized MLA, circular QSA and TurboQuant payload overrides retain
their own contracts. Host hot-cache transport and QSA codec migration are pending.

The decision follows the existing mainline-compatible storage protocol and
FlashInfer's explicit layout/stride boundary, avoiding an additional INT8
allocation protocol. No format/default or numerical kernel changes occur.
`tools/kv_codec/verify_storage.py` compares the immutable original source's
payload and page sizes over 2646 configurations on CPU. All match; 17 tests
cover unequal K/V, NVFP4, inline scale aliasing, page padding, specialized-layout
preservation and compatibility reexports. GPU cache-write/graph and model gates
remain required before promotion.

## Offline comparison tool

`tools/kv_codec/evaluate.py` consumes a version-1 JSON manifest with `samples`.
Each entry must identify `model`, `layer`, `rank`, `tp`, `role`, `source_sha`,
`wheel_sha256`, `request_tokens_sha256`, relative `tensor_path`,
`request_origin: "real"`, and `capture_stage: "post_rope_pre_quantization"`.
Tensor archives contain FP16 `q`, `k`, `v` in token/head/dimension order,
Boolean `allowed` (query/key or query/query-head/key), `attention_scale`,
`k_scale`, `v_scale`, and int64 `request_token_ids`, `query_positions`,
`key_positions`. The tool verifies request token hashes and positional bounds;
QSA compression/indexer/window selections must be reflected in the actual mask.
It refuses encoded K/V as an FP16 oracle and empty masked attention rows.

Run `.venv/bin/python tools/kv_codec/evaluate.py --manifest DATA/manifest.json
--out DATA/comparison.json`, with both paths on the task data disk. If using
`--device cuda`, acquire the normal GPU locks first. The tool compares nine
schemes: FP16, E4M3 with captured layer scales, signed token/head INT8 with
FP32/FP16 scales, affine token/head u8 with FP32 scale/minimum, feature groups
32/64 with FP16 scales, and channel-wise K over token groups 32/64 with
FP32 scales plus token/head V. Partial groups include padded storage in the
reported bytes. FP16 scale candidates quantize against the stored rounded scale.
Results retain each sample separately, K/V and attention errors, clipping and
physical storage cost. Candidates stream one pair at a time; percentiles use
order statistics so long arrays exceed neither the `torch.quantile` size limit
nor the memory cost of retaining every reconstructed format. The reference is FP32 masked attention over captured
FP16 tensors, not a claim of bitwise native-kernel output. Eight numerical
self-tests pass; they are synthetic tool checks and **not format-selection data**.
No real three-model dataset has been collected yet; no default is selected.

### Constraints for the INT8 tile reader

The existing token/head INT8 head spans `D + 4` bytes. At D256, successive
heads start 260 bytes apart: every second start is only 4-byte aligned. The
current XQA FP8 loader reinterprets the payload as `uint64_t` and divides
the element offset by eight. Reusing that loader for inline-scale INT8 would
both misalign loads and truncate offsets. Keep the existing protocol as a
candidate, but implement its packed reader with alignment-safe loads (for
example two 32-bit loads); do not change its budget to an 8-byte padded layout
without treating that as a separate format/layout decision. A smaller FP16
inline scale would change alignment again and is not the existing protocol.

Per-token scales also cannot reuse global-scale epilogues blindly. Moving a
small token-specific V scale into FP16 softmax probabilities can underflow at
128K/256K, whereas the current global V scale can be applied after the PV
accumulation. The reader interface must own how a tile is staged and which
scale remains for QK/output epilogues, preserving the existing FP16/E4M3
arithmetic. Evaluate FP16 staging of dequantized INT8 V against the captured
reference before optimizing it. Do not declare template instantiation alone
as proof that a format is supported by the path.

## Research and decisions

| Reference inspected | Adopt / evaluate | Avoid / reason |
|---|---|---|
| [vLLM backend interface](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py), local `KVQuantMode`/allocator/Triton implementation at the baseline | Keep backend capability declarations and reuse the existing token-head storage candidate | Do not rely on a backend dtype whitelist as proof that every accelerated route exists |
| [SGLang attention interface](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/base_attn_backend.py) and [memory pool](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py) | Separate storage ownership from attention execution | Do not import its whole allocator or assume an SM80 conversion works on SM70 |
| [FlashInfer typed paged cache](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh) | Typed payload/scale views and explicit strides, specialize conversion at compile time | Its current kernels are not an SM70 drop-in; preserve local page/graph behavior |
| [TensorRT-LLM XQA runner](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h) and [attention/INT8 KV documentation](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html) | Separate support checks and kernel parameters; compare per-tensor INT8 as a calibration candidate | Do not assume newer-architecture XQA is runnable on Volta or that tensor-wide scales preserve this model's quality |
| [TurboMind quantization](https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/attention/quantization.h), vendored file at baseline | Evaluate its packed u8→f16 byte-permutation conversion, parameterized conversion and scale/zero-point handling | Do not paste its offset-u8 conversion into a signed-int8 reader without matching zero point and rounding |
| [ninfer small-t INT8 Volta v2](https://github.com/huangserva/ninfer-v100-tpx/blob/9913bc229e77aa1bd3afb82ac64ff98b071062a4/src/ops/softmax_attention/dense/causal_cache/small_t_i8_volta_v2.cuh), `prompt_i8`, Apache-2.0 | Benchmark its shared KV tile and FP32 PV accumulation; study FP16 token scales and reduction work | It uses BF16 query inputs and a different cache/page protocol: match contracts before attributing a speed gap |
| [llama.cpp fattn-vec](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-vec.cuh) / q8_0 | Include group-32 FP16 scale overhead and conversion in candidate comparison | Do not assume its quantized block layout is interchangeable with vLLM paged KV |
| [KIVI](https://arxiv.org/abs/2402.02750) | Compare channel-grouped K and token-grouped V, including asymmetric ranges | Channel-grouped K needs a defined policy for partial groups and changing extrema; 2-bit findings alone do not choose an 8-bit format |
| [KVQuant](https://arxiv.org/abs/2401.18079) | Measure channel outliers and layer sensitivity | Pre-RoPE storage/outlier side channels add reader and prefix-position state; postpone unless real INT8 error requires them |
| [QServe](https://arxiv.org/abs/2405.04532) | Treat dequantization cost and quantization granularity as a joint decision | Its KV4/system results do not establish Volta INT8 quality or speed |
| Local `turboquant_k8v4` and `flash_decode_turboquant.cu` | Preserve packed-layout accounting and route compatibility | K8V4 is a distinct codec; do not call it symmetric INT8 or silently replace its value format |

No external kernel implementation is copied in the first extraction. The
shared reader is existing repository code. Online main references are research
references; pin the exact upstream implementation commit before importing code
and retain its license and attribution.

## Data selection and acceptance

Capture FP16 Q, post-RoPE K and V **before quantization** from real Flash-Next,
27B NVFP4 and 27B GGUF requests, for multiple layers and context lengths. Retain
request/token IDs, target/draft role, layer, rank/TP/head topology, query offsets,
valid lengths, masks/window/indexer selections, source/wheel hash, sampling and
MTP state. Do not reuse already quantized KV as the FP16 oracle. Avoid unbounded
tensor dumps and keep them on the task data disk.

Offline candidates: FP16; current calibrated E4M3; symmetric token/head INT8 with
FP32 versus FP16 scales; symmetric channel groups of 32/64 elements; asymmetric
token/head INT8; channel-grouped K plus token-head V. Report K/V separately:
RMSE, max/percentile absolute errors and outlier distribution; attention output
RMSE/max error against the same masked FP16 attention. Keep each request/layer
separate before aggregation. Account for payload, scale and zero-point bytes.
FP16-scale candidates quantize with the **stored** scale to expose rounding or
underflow. Do not choose a runtime default from synthetic random tensors.

For each format/path/shape, the promotion order is:

1. Element/reference attention comparisons, and bitwise old/new FP16/E4M3
   extraction parity, including cache write/restore and graph replay.
2. Exact-shape single-operator A/B, cold L2/CUDA Graph where required; report
   partition and reduction separately. INT8 q=1 must match or beat ninfer INT8;
   other INT8 paths must match or beat corresponding E4M3 paths.
3. Model tests with executed route assertions, including 8192-chunk 75T/FA2,
   TP2/TP4, actual prefix restore, host offload and target/draft combinations.

Then run FP16/E4M3/INT8 teacher-forcing full-vocabulary KL and top-1 agreement,
paired acceptance on the existing eight natural prompts with a 95% interval,
128K/256K retrieval, natural EOS and C4 health. Respect correlation when forming
the interval: prompt/seed pairing must not be replaced by treating every
accepted token as an independent sample. Whole C1/C4 A/B uses one host, one wheel
and a fixed model/weight/TP/GPU/length/sampling/MTP/graph contract; separate TTFT,
prefill, pure decode and emitted-token round cost. Existing 35B AWQ/FP8 migration
acceptance remains required; no short smoke substitutes for it.

## First extraction evidence and next gate

Retained raw artifacts and commands are in the task worklog. The source-preservation
fixture freezes the original top-level function/class ASTs, including every
dispatch method, and checks relocated reexports, matrix coverage and manifests.
It is an extraction guard: intentional later changes require a documented gate
and an explicit fixture update, not automatic regeneration on test failure.

The native reader probe compares all 65536 FP16 encodings and every E4M3/E5M2
byte, including NaNs/zeros/subnormals, four scales and float/half readers. Its
normalized SM70 PTX and GPU output bytes match the original. This proves the
reader extraction, **not whole-attention or model performance**.

Before this scope is promoted, finish isolated metadata oracles, clean extension
builds and actual attention parity/timings, then a final-wheel route/model gate.
Afterwards migrate one writer family, codec byte accounting and scheduling
module at a time. INT8 kernel admission and default selection wait for the full
FP16/E4M3 refactor gate and real-request format-selection data. No new format or
E4M3 partition change is enabled in this first scope.

Full first-extraction measurements, component wheel hashes and unresolved gates
are retained in [the result ledger](sm70_kv_first_extraction_results.json). The
48 direct-operator outputs match bitwise; maximum FP32 reference error is
6.50e-6. The two initial long FP16 timing outliers were rechecked in ABBA order
with 101 replays per shape: +0.15% single-row and −3.26% batch. This is a
no-regression screen, not a statistical equivalence bound or a speedup claim.
The paired source-policy suite has identical 186 passes / 11 failures in each
arm; all failures require the absent built SM70 FA2 library. They remain pending
for the final installed vLLM wheel. Pre-commit and 16 source/build guards pass.
