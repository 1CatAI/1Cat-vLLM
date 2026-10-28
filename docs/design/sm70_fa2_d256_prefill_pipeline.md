# SM70 FA2 D256 Split-D Prefill

## Status

Qwen3.6-27B full-attention prefill now uses a dedicated Volta D256 kernel for
both the first dense chunk and later chunks in the standard interleaved paged
KV cache. The direct paged operator reads that layout without a repack or
temporary output. For eligible long, single-sequence prefix chunks, the
dispatcher now gathers logical pages into a reusable exact-dense K/V workspace
and calls the faster exact dense operator.

The route is enabled by default. Set
`VLLM_FLASH_V100_FA2_D256_PREFILL=0` to disable it. Dispatch is exact-only:
unsupported shapes return to the established `FLASH_ATTN_V100` path rather
than entering an unvalidated generic FA2 fallback. Decode, MTP small-query
attention, FP8 KV cache, sliding-window attention, and non-causal attention
are unchanged.

Long-prefix gathering is also enabled by default. Set
`VLLM_FLASH_V100_PREFILL_GATHER_DENSE=0` to retain direct paged attention.
Its default thresholds are `Q>=4096` and `KV>=8192`.
For the admitted Qwen TP4 shape, gathered dense attention also uses the
default-on three-way KV partition at `KV>=32768`; set
`VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3=0` to retain unsplit exact dense.

## Accepted Contract

The full-model acceptance workload is:

- model: `/home/ymzx/models/Qwen3.6-27B-AWQ`;
- GPU: 4x V100-32GB, TP4, CUDA 12.8, PyTorch 2.10.0+cu128;
- FP16 activations and KV, TurboMind AWQ, no MTP;
- `FLASH_ATTN_V100`, CUDA graph `FULL_DECODE_ONLY`, not eager;
- input/output: 8192/16, chunk size 4096, max model length 16384;
- one warmup and two measured requests;
- separate official-sampling quality runs at 1K/64 and 8K/128.

The operator requires SM70, FP16, causal full attention, head dimension 256,
unit inner strides, query length at least 1024, and query/KV lengths divisible
by 64. The paged route currently accepts one sequence per call and preserves
the runtime block table. Gathering further requires one sequence, `Q>=4096`,
`KV>=8192`, `KV>Q`, no graph capture, and lengths divisible by 64. All other
cases retain direct paged attention. Three-way partitioning further requires
batch 1, `Q=4096`, `Hq=6`, `Hkv=1`, and `KV>=32768`.

## Kernel Design

The external source is pinned to
`zhinianqin/flash-attention-v100@c2eda5e6115b98c3ba4bfd181570668742eece22`.
The vendored patch adds exact dense and paged Torch operators. The final
operator ABI includes `n32`; missing exact N32 operators are treated as a
stale build and cause a safe fallback. This prevents an older,
quality-rejected N64 binary from being selected from a local cache.

The accepted specialization uses:

- `BLOCK_M=64`, `BLOCK_N=32`, and four D64 chunks;
- eight warps arranged as four two-warp groups;
- eight distinct Q rows per warp for QK, eliminating duplicated QK work;
- a shared N32 P tile per warp pair;
- D128 output ownership per warp for PV;
- the standard FA2 N32 online-softmax order;
- conflict-aware Q/K/V layouts and software register prefetch;
- one paged-KV address resolution per K/V N32 tile, with D64 pointers
  derived by constant column offsets;
- a Volta `TT` PV tensor-core mapping that consumes V as K-by-D directly;
- two alternating PV register fragments so the next V shared load overlaps
  the current phase's HMMA stream;
- 128-bit conflict-aware V stores and paired 64-bit V loads, removing the
  previous transpose-path scalar loads and most operand permutation work;
- FP32 score, online-softmax, and output accumulation;
- 45,568 bytes of dynamic shared memory.

The N32 order is a quality requirement, not a shape-tuning choice. An earlier
N64 variant merged two N32 halves before softmax. Its standalone relative L2
error was only about `1.27e-4`, but repeated full-attention layers amplified
that error enough to change official sampled tokens. N32 reduces the operator
relative L2 error to about `4.6e-6` at 4K and restores model token parity.

Dense and paged kernels use independent Q, K, and V strides. A real TP rank
has:

```text
Q: shape (4096, 6, 256), stride (1536, 256, 1)
K: shape (4096, 1, 256), stride (256, 256, 1)
V: shape (4096, 1, 256), stride (3584, 256, 1)
```

The paged kernel reads the normal
`[block, K/V, page, head, dim]` allocation directly, including page size 784.
No KV-major allocation change is retained.

The final causal SM70 specialization uses 255 registers/thread with no spill
or local stack. It remains at one CTA per SM. The gain comes from useful D256
parallelism, lower operand-movement cost, and software scheduling, not higher
occupancy.

## Operator Results

Measurements use Qwen TP4 `Hq=6`, `Hkv=1`, D256, FP16, and causal attention.
`Causal TFLOP/s` counts attended pairs. `Full-square logical TFLOP/s` counts
the masked upper triangle and is included only to compare with tools that use
that convention; it is not hardware-executed work.

| Q=KV | Generic FA2 | N32 Split-D | Speedup | Causal TFLOP/s | Full-square logical |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.340 ms | 0.213 ms | 1.60x | 15.1 | 30.2 |
| 4096 | 2.190 ms | 1.779 ms | 1.23x | 29.0 | 57.9 |
| 8192 | 7.043 ms | 5.425 ms | 1.30x | 38.0 | 76.0 |

The 4K and 8K logical rates exceed the original 49-TOPS comparison target.
The physically meaningful causal rates are reported separately above.

The release comparison below uses the real chunked-prefill shape (`Q=4096`),
Qwen TP4 heads (`Hq=6`, `Hkv=1`), page size 784, and a fixed 1312 MHz V100
application clock. `Current 1Cat` is the Flash-V100 extension shipped in
1Cat-vLLM v1.2.2 at source revision `644d8a7cd0`, not an earlier kernel from
this research branch. Its extension SHA256 is
`8582b5b1a72d5ebfd9a35417f267298845195a6846285e26fff7ad9a5905f771`.
Each entry is the median of two alternating runs after six warmups; both
variants were measured at exactly the same clock.

| KV length | Current 1Cat v1.2.2 | Final Split-D TT | Latency reduction | Speedup |
|---:|---:|---:|---:|---:|
| 8K | 11.1255 ms | 5.0542 ms | 54.57% | 2.20x |
| 64K | 87.6001 ms | 50.4504 ms | 42.41% | 1.74x |
| 128K | 174.9729 ms | 103.8420 ms | 40.65% | 1.68x |
| 256K | 349.6960 ms | 210.7364 ms | 39.74% | 1.66x |

A separate production-clock cross-check at up to 1530 MHz measured 2.20x,
1.69x, 1.75x, and 1.64x respectively. The fixed-clock table is authoritative
because the long runs otherwise trigger clock drift.

The final clean build is bitwise identical to the previously accepted exact
N32 operator at all four lengths. Against the current 1Cat extension, relative
L2 error is `3.79e-4` at 8K and remains below `4.0e-4` through 256K.

NCU isolates the final TT operand path from the immediately preceding
address-reuse candidate. At 8K and 1312 MHz, kernel time falls from 5.18 ms to
5.06 ms while LSU thread instructions fall from 187.1M to 132.5M and shared
load wavefronts fall from 268.0M to 230.0M. Static SASS shrinks from 1,875 to
1,767 instructions: `LDS.U16` falls from 128 to zero and `PRMT` from 68 to
four. Tensor active reaches 32.06%. Shared-store conflicts increase, but their
cost is smaller than the removed transpose/load/permutation chain.

## Long-Prefix Gather-to-Exact-Dense

The production KV allocation is interleaved as
`[physical_page, K/V, 784, Hkv, D]`; unbinding K and V therefore produces
non-contiguous views. The accepted path uses the runtime block table to
`index_select` both views into per-device, per-stream workspaces, slices the
last page to the exact sequence length, and invokes the same exact N32 dense
operator. Random physical page order is part of the test, and all outputs are
bitwise equal to direct paged attention.

The following single-rank microbenchmark uses the validated production binary,
`Q=4096`, `Hq=6`, `Hkv=1`, D256, FP16, page size 784, and randomized physical
page order:

| KV length | Direct paged | Gather | Gather + exact dense | Net reduction |
|---:|---:|---:|---:|---:|
| 8K | 5.6740 ms | 0.0635 ms | 5.0606 ms | 10.81% |
| 64K | 50.4571 ms | 0.1654 ms | 46.6877 ms | 7.47% |
| 128K | 103.8479 ms | 0.2985 ms | 95.0461 ms | 8.48% |
| 256K | 210.6481 ms | 0.5868 ms | 192.8289 ms | 8.46% |

This is a 7.47-10.81% **attention-operator** reduction, not a 6% whole-model
claim. The controlled Qwen3.6-27B-AWQ TP4 full-model A/B is:

| Input | Direct paged prefill | Gathered prefill | Saved | Reduction | Quality/route gate |
|---:|---:|---:|---:|---:|---|
| 8K | 2.584815 s | 2.567641 s | 17.174 ms | 0.66% | exact tokens; 48 hits/rank |
| 64K | 26.319437 s | 25.860738 s | 458.699 ms | 1.74% | exact tokens; 480 hits/rank |

The 8K run uses one warmup and two measured requests. The 64K run uses one
warmup and one measured request. Decode TPOT is unchanged within 0.04%; the
gate does not select decode. At 256K, geometric workspace growth can retain up
to about 269.5 MiB per active stream and TP rank for `Hkv=1`, D256, FP16 K/V.
If allocation fails, dispatch logs once and falls back to direct paged
attention instead of failing the request.

## Accepted Exact-Dense Split-KV3

For the Qwen3.6 TP4 chunk shape (`Q=4096`, `Hq=6`, `Hkv=1`, D256), the exact
dense kernel launches 384 CTAs on 72 SMs. One-CTA-per-SM residency requires
six waves and leaves 48 slots unused in the final wave. The accepted default
for this shape partitions each CTA's visible KV range into three independent
segments,
launches 1,152 CTAs (exactly 16 waves), and combines FP32 partial
numerator/max/sum state in a separate kernel. It does not quantize or truncate
K/V and retains the accepted N32 body inside each partition.

Clean paired single-rank measurements at 1312 MHz are:

| KV length | Exact dense | Split-KV3 | Throughput gain | Decision |
|---:|---:|---:|---:|---|
| 8K | 4.6572 ms | 4.5117 ms | 3.22% | keep the original route |
| 32K | 22.5864 ms | 21.0775 ms | 7.16% | admit |
| 64K | 46.4625 ms | 42.7131 ms | 8.78% | admit |
| 128K | 94.7456 ms | 87.1639 ms | 8.70% | admit |

At 64K, the candidate differs from the accepted FP16 reduction tree by at
most `3.0518e-5`. Against an FP32 reference over 64 sampled query rows, its
relative L2 error is `2.8796e-4`, slightly lower than the accepted dense
kernel's `2.8850e-4`. The same check passes at 128K. PTXAS reports 253
registers/thread for the partition kernel and 26 for the merge, with zero
spills for both. The reusable FP32 workspace is about 75.6 MiB per active
stream and rank for the admitted shape; OOM falls back to exact dense.

The exclusive-GPU Qwen3.6-27B-AWQ TP4 full-model gate measured
`25.860738 -> 25.514792 s` at 64K, saving 345.946 ms (1.34%) after one warmup.
Every rank reported 288 split-KV3 kernel hits, and all 16 deterministic output
token IDs, text, and the token hash matched the gathered exact-dense baseline.
Decode TPOT remained unchanged at about 21.35 ms.

`VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3` is therefore enabled by default. The
dispatch remains evidence-bounded to batch 1, `Q=4096`, `Hq=6`, `Hkv=1`,
D256, FP16 dense K/V, and `KV>=32768`; other shapes continue to use exact
dense. Set the variable to `0` to disable it.

### Split-KV for short prefix chunks

A prefix-cache hit leaves a short uncached chunk over a long prefix: the rest
of a conversation turn, or the few tokens after a snapshot boundary. Such a
chunk has few 64-row query tiles (Q=1088 gives 17 x 6 heads = 102 CTAs on 80
SMs; Q=64 gives 6), so the unsplit kernel left most of the GPU idle, and
chunks below Q=1024 fell back to the paged BM32 kernel. The split-KV3 kernel
now takes a runtime split count
(`sm70_d256_splitd_n32_dense_splitkv_fwd`, patch
`sm70_flash_attn_d256_splitkv_n.patch`); the three-split entry and its merge
are unchanged and bit-identical. Prefix chunks below the 79T Q4096 range pick
the split count that minimizes CTA waves per split plus a small per-split
merge cost, bounded by a shared 24 MiB FP32 partial workspace (4096 query rows)
reserved during memory profiling and by at least 1024 prefix keys per split.

The E4M3 bridge also pads chunks it previously could not admit. Unaligned
sequence lengths get zero keys and zero queries appended past the end of the
sequence, which causal masking hides from every real query; leading zero
queries still align the chunk to 64 rows. Chunks shorter than 32 tokens enter
the bridge once the sequence reaches 8192 keys.

Measured on V100 at 110K keys (attention per layer): Q=64 13.5 -> 1.1 ms,
Q=128 12.2 -> 1.9 ms, Q=512 11.7 -> 7.3 ms, Q=1088 24.0 -> 15.5 ms. For
Qwen3.8-27B NVFP4 TP4 with a 109K cached prefix, TTFT for a 1.1K-token turn
fell 1.39 -> 1.09 s, and turns whose last step had fewer than 32 tokens fell
1.9 -> 0.63 s. `VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3=0` disables both
split routes.

The subsequent exhaustive SM70 ownership and instruction-scheduling study is
recorded in [SM70 HMMA Pipeline Search](sm70_hmma_pipeline_search.md). It
admits a register-double-buffered PV schedule and closes the minimum-shuffle,
second-K-lookahead, and conditional-rescale variants with wall-time evidence.

An adjacent real-model TP4 route check used Qwen3.6-27B-AWQ, 8K input,
16-token deterministic output, chunk size 4096, FP16 KV, FlashAttentionV100,
and non-eager FULL_DECODE_ONLY graphs. With an unrelated resident Ray service
left untouched and a 2 GiB KV reservation, the previous binary measured
2586.58 ms prefill and the candidate measured 2581.31 ms, a 5.27 ms (0.20%)
reduction. Every rank reported 48
`prefill_prefix_paged_splitd_d256` calls and output token hashes matched. The
resident service slowed both prefill and decode relative to the clean historic
baseline, so this result proves route translation but is not a replacement
release baseline.

## Post-Dense-Projection Trace And Closed P Swizzle

After enabling gathered exact-dense attention, split-KV3, and the exact FP16
AWQ prefill projections, an Nsight Systems capture of the 64K TP4 request
measured 21.807 seconds of critical-rank GPU timeline. The matching unprofiled
prefill result is 20.988843 seconds. The largest device-0 kernel buckets are:

| Bucket | Kernel time | Share of summed kernel time |
|---|---:|---:|
| exact FP16 projection GEMMs | 9.912 s | 47.4% |
| D256 full attention | 5.767 s | 27.6% |
| NCCL all-reduce | 1.412 s | 6.7% |
| vectorized elementwise kernels | 0.847 s | 4.0% |
| TurboMind FP16 GEMM | 0.771 s | 3.7% |
| FlashQLA GDN | 0.740 s | 3.5% |
| RMSNorm | 0.517 s | 2.5% |

The full-attention time grows nearly linearly across the 16 prefill chunks:
the per-layer kernel is 1.701 ms at 4K, 21.113 ms when split-KV3 first selects
at 32K, and 42.522 ms at 64K. The final chunk sustains about 37.6 causal
TFLOP/s. KV gather is not a bottleneck.

A follow-up applied the previously useful conflict-free P layout to the final
TT-PV split-KV3 body. It preserves the exact output hashes and lowers the
partition kernel from 253 to 251 registers/thread with zero spill, but the
required accumulator-to-layout warp shuffles are too expensive:

| KV length | Accepted split-KV3 | TT-PV + P swizzle | Regression |
|---:|---:|---:|---:|
| 32K | 21.0330 ms | 27.1365 ms | 29.0% |
| 64K | 42.6716 ms | 55.8776 ms | 30.9% |

This combination is rejected. Do not retry P bank-conflict removal by adding
shuffle-based accumulator repacking to the TT-PV body. A successor must
change the native QK accumulator or PV operand ownership so the desired
shared address is produced without a conversion stream.

## Numerical And Quality Gates

- Exact Split-D versus generic FA2 has max absolute error `2.4414e-4` and
  relative L2 error `4.58e-6` at 4K and `5.79e-6` at 8K.
- Real non-contiguous V and a contiguous copy are bitwise identical.
- A randomized non-sequential paged block table is bitwise identical to the
  corresponding dense exact operator.
- The 1K/64 official Qwen sample produces 64/64 identical token IDs against
  both original Flash-V100 and generic FA2.
- The 8K/128 official Qwen sample produces 128/128 identical token IDs and
  identical text against original Flash-V100.
- The deterministic warmup and both measured requests retain token hash
  `220e51bcf45e69de1a35817c9501aadfcae784ed195448b3bf37b05d4aa815a2`.
- Every TP rank reports 48 dense and 48 paged exact route hits in the stable
  warmup-plus-two-repeat benchmark.

Prompt logprobs are not bitwise stable across FP16 attention implementations;
generic FA2 also differs from the original backend. Therefore promotion uses
operator error bounds plus official sampled token parity, rather than treating
the original backend's prompt perplexity as an absolute numerical oracle.

## Full-Model Result

The final controlled A/B uses Qwen3.6-27B-AWQ, TP4, FP16 KV, chunk size 4096,
16 generated tokens, no prefix cache, and non-eager compile graphs. Both
variants use the same source and runtime configuration; only
`VLLM_FLASH_V100_FA2_D256_PREFILL` changes. Four V100s are fixed at 1312 MHz.
Each value is the mean of two requests after a per-length warmup.

| Input | Current 1Cat prefill | Final TT prefill | Latency reduction | Current tok/s | Final tok/s | Throughput gain |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 2.5722 s | 2.4440 s | 4.99% | 3184.8 | 3351.9 | 5.25% |
| 64K | 30.2812 s | 25.5058 s | 15.77% | 2164.2 | 2569.5 | 18.72% |
| 128K | 83.1879 s | 64.5516 s | 22.40% | 1575.6 | 2030.5 | 28.87% |
| 256K | 256.6722 s | 183.9463 s | 28.33% | 1021.1 | 1424.8 | 39.54% |

The 256K row uses 262,080 input tokens so the final 4,032-token chunk remains
divisible by 64 and the 16-token generation stays within the model's exact
262,144-token limit. All measured requests preserve the current 1Cat output
token hash at every length.

The corresponding no-MTP decode gate uses the model's official sampling
parameters (`temperature=1.0`, `top_k=20`, `top_p=0.95`) and 256 generated
tokens. TPOT excludes prefill and the first generated token. The 256K row uses
261,824 input tokens, leaving 64 tokens below the exact model limit.

| Input | Current 1Cat TPOT | Final TT TPOT | Current tok/s | Final tok/s | Throughput delta |
|---:|---:|---:|---:|---:|---:|
| 8K | 15.034 ms | 15.037 ms | 66.52 | 66.50 | -0.02% |
| 64K | 18.999 ms | 18.985 ms | 52.63 | 52.67 | +0.08% |
| 128K | 24.072 ms | 24.062 ms | 41.54 | 41.56 | +0.04% |
| 256K | 33.766 ms | 33.760 ms | 29.62 | 29.62 | +0.02% |

All four 256-token output hashes match. The differences are measurement noise:
the new D256 operator is gated to prefill queries of at least 1,024 tokens,
while q=1 decode continues to use the existing paged XQA path. Relative to
8K, final decode throughput falls 20.8% at 64K, 37.5% at 128K, and 55.5% at
256K. Reducing that decay requires a separate decode-attention change.

An earlier no-compile 8K route-development run measured:

| Route | Mean prefill | Delta from original | Deterministic tokens |
|---|---:|---:|---:|
| Original Flash-V100 | 2.7445 s | reference | exact |
| Generic SM70 FA2 | 2.4084 s | -12.25% | exact |
| N32 Split-D direct output | 2.3355 s | **-14.90%** | exact |

The accepted route saves 408.93 ms against the original implementation and
72.85 ms against generic FA2. Its mean differs by only 0.73 ms from the faster
but quality-rejected N64 experiment.

The independent 8K/128 official-sampling run measured prefill
`3.1962 -> 2.8814 s` (-9.85%) and steady decode
`60.04 -> 60.27 tok/s`, with exact 128-token parity.

## Bottleneck After Attention

The retained Nsight Systems trace was captured before the N32 numerical-order
fix, so its exact attention time must not be reused as a current kernel timing.
It remains valid for category prioritization because N32 changes full-model
prefill by less than 0.1% relative to that run:

| Category | TP wall-equivalent time | Share |
|---|---:|---:|
| TurboMind AWQ GEMM | 1577.127 ms | 68.17% |
| Norm, activation, elementwise | 250.836 ms | 10.84% |
| TP NCCL collectives | 170.101 ms | 7.35% |
| GDN / linear attention | 111.975 ms | 4.84% |
| Split-D full attention | 96.850 ms | 4.19% |

The next end-to-end bottleneck is TurboMind AWQ GEMM, not D256 attention. A
representative `4096x8704x5120` gate/up kernel reaches 58.87 tensor TOPS but
is limited to 12.5% occupancy by 248 registers/thread and 65.55 KiB shared
memory. The retained category trace predates long-prefix gathering; a new 64K
trace is required before changing another attention structure.

## Evidence

- Final CUDA gate:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/splitd_n32_final_cuda_gate.json`
- Stable full-model result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_8k_splitd_n32_direct_out.json`
- Final fixed-clock full-model comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_prefill_current_vs_splitd_tt_fixed1312.json`
- Final fixed-clock decode comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_decode_current_vs_splitd_tt_fixed1312.json`
- Current and final full-model raw results:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_prefill_current_1cat_fixed1312.json`
  and
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_prefill_splitd_tt_fixed1312.json`
- Current and final decode raw results:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_decode_current_1cat_fixed1312.json`
  and
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/fullmodel_decode_splitd_tt_fixed1312.json`
- Official 8K quality result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/splitd_n32_official_i8k_o128.json`
- Final N32 ABI default-on 1K quality result:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/splitd_n32_abi_default_official_i1k_o64_logprobs.json`
- Official 8K comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/official_quality/compare_baseline_splitd_n32_i8k_o128.json`
- Nsight Systems report:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/profiles/qwen36_27b_awq_tp4_i8k_splitd_both_exact_prefill.nsys-rep`
- Fixed-clock current-1Cat comparison:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/current_1cat_v122_vs_tt_vload_fixed1312.json`
- Final clean-build latency and quality gates:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/tt_final_clean_gate.jsonl`
  and
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/qwen/tt_final_clean_quality.json`
- Validated runtime binary SHA256:
  `91bd8ec125459411da57d5f6d111e6760573a717d3c8ab0f2161752dc6cdb084`
- Long-prefix gather A/B results and logs:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/gather-dense/results/`
- Split-KV3 clean microbenchmark and FP32-reference gates:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/splitkv3_n32_gate_32k_128k_clean.json`
  and
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/splitkv3_n32_quality_ref_q4096_kv65536.json`
- Split-KV3 production patch-check binary SHA256:
  `84571628436990f433572d073c55a85d4910a7db83eb883fcbecceb133b154b4`
- Split-KV3 exclusive 64K full-model gate:
  `/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/splitkv3-fullmodel/results/candidate_decode_harness_i65536.json`
- Independent clean-build binary SHA256:
  `91bd8ec125459411da57d5f6d111e6760573a717d3c8ab0f2161752dc6cdb084`
- Vendored patch SHA256:
  `7fc34a1fa9d25d7f1c6c1b77382717c4b0f9aba252b486eeb346f3f8cbe4826b`

## Closed Paths

- N64 combined-softmax Split-D is rejected: fast standalone, but official
  sampled tokens diverge because its FP16 online-softmax order differs.
- D256 ports that keep all output columns in one warp/CTA exceed practical
  Volta instruction-cost limits. A full-head, single-barrier prototype
  reduced barrier stall to 0.75%, long-scoreboard to 2.52%, and shared-load
  conflicts to three, but increased LSU instructions by 37% and integer
  instructions by 80%; its 4.707 ms 8K result loses to address reuse.
- Split-D warp-pair register-P exchange is rejected. Three bitwise variants
  ranged from 5.495 ms to 10.664 ms at 8K. The best eliminated all P shared
  conflicts without changing Tensor instructions, but raised LSU instructions
  by 16.5% and integer instructions by 26.6%; shuffle/mailbox cost exceeded
  the conflict reduction.
- A KV-major cache allocation did not improve the full model and expanded the
  cache-management blast radius, so it was removed.
- Chunk size 8192 was 1.50% slower end to end than 4096 in the earlier A/B.
- Full-D256 tile residency is closed: the spill-free hybrid still regressed
  64K by 14.97% because integer and LSU instructions rose by 53.7% and 18.6%.
- Further attention-only work has a small current-model ceiling. Future
  prefill work should first target AWQ GEMM and elementwise overhead.
