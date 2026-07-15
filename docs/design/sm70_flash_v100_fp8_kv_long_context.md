# SM70 Flash-V100 FP8 KV Long-Context Optimization

Date: 2026-07-15

## Scope And Status

This document owns the FP8 E5M2 KV-cache long-context path for
Qwen3.6-27B-AWQ on TP2 V100. The accepted runtime uses TurboMind AWQ,
`FLASH_ATTN_V100`, prefix caching, Mamba align mode, CUDA graphs, and no eager
execution.

Both identified route gaps are now implemented and default enabled:

- prefill: one-pass E5M2 expansion into a shared FP16 page-784 workspace;
- decode: vectorized E5M2 XQA with CUDA-graph capture support.

The prefill BM32 route supports a final partial query tile. This is required
for MTP4 align mode, whose scheduler chunk is 1616 tokens (`50*32+16`), rather
than the no-MTP 1568-token chunk.

The changes preserve the old FP8 model output hashes for the measured 16K and
64K official-sampling cases. FP8 quantization quality relative to FP16 remains
a separate model-level acceptance question; these kernels add no observed
quality regression relative to the prior FP8 route.

## Original Failure

The matched historical no-MTP FP8 baseline regressed increasingly with
context length:

| Context | FP16 prefill | Old FP8 prefill | FP16 TPOT | Old FP8 TPOT |
|---:|---:|---:|---:|---:|
| 16K | 10.098 s | 14.680 s | 19.532 ms | 21.762 ms |
| 64K | 60.328 s | 127.842 s | 26.014 ms | 32.768 ms |
| 128K | 169.206 s | 450.290 s | 35.218 ms | 47.119 ms |

This was not an intrinsic FP8 bandwidth penalty.

## Root Cause

### Prefill

The TP2 full-attention shape is `Hq=12,Hkv=2,D=256`, with scheduler chunk
`M=1568` and Mamba-aligned KV pages of 1568 tokens.

Two route losses multiplied together:

1. Page 1568 did not select the FP16 page-784 BM32 phase kernel. At 128K,
   FP16 page 784 took 115.175 ms per layer while FP16 page 1568 took
   264.380 ms.
2. The generic FP8 kernel performed page lookup and E5M2 expansion inside
   every query CTA. At the same page size, FP8 page 1568 took 565.320 ms.

NCU showed that this was not HBM saturation. The generic FP8 prefill kernel
used 96.64 KiB dynamic shared memory, admitted one CTA per SM, reached only
25% occupancy, and had about 4% tensor activity. It repeatedly converted the
same KV values for the many query CTAs.

### Decode

E5M2 conversion itself was already cheap. At 64K, FP8 scalar decode was
slightly faster than FP16 scalar decode. The regression came from the Python
and C++ XQA gates accepting only FP16 KV.

The scalar GQA kernel launched one CTA per query head and partition. For
`q_per_kv=6`, that launched 768 CTAs and reread each KV head six times. XQA
launches one CTA per KV head and partition, or 128 CTAs for the same 64K case.

## Correctness Defect Found During Bring-Up

The first bridge experiment reached about 4.8x prefill speedup but failed the
numeric gate when `Hkv>1`. The BM32 PV path used a hard-coded V token stride of
256. That was valid only for `Hkv=1`; TP2 requires a stride of
`Hkv*D = 512`.

The fix passes the live `v_token_stride` to both PV WMMA loads. BM32 is now
bitwise equal to the prior accurate path for `Hkv=1,2,3`, causal attention,
reverse page tables, and tail pages. This was a latent FP16 BM32 defect, not an
FP8 quantization error.

## Implemented Prefill Path

`fp8_e5m2_paged_kv_to_fp16` performs one vectorized gather/expansion pass:

- reads two E5M2 values at a time and expands them exactly into `half2`;
- resolves non-contiguous physical pages through the live block table;
- supports partial tail pages and non-unit K/V scales;
- writes into preallocated FP16 pages of 784 tokens;
- performs no allocation or host synchronization inside the CUDA op.

The downstream BM32 kernel uses a ceil-divided query grid. Full 32-row CTAs
retain the original unguarded load/store path; only the final partial CTA
zero-fills missing query rows and guards output/LSE writes. This preserves the
fast schedule while making `M=1616` a first-class route instead of sending the
first 1600 rows through the generic FP8 kernel.

The backend workspace is shared across attention layers and keyed by GPU,
CUDA stream, `Hkv`, and head dimension. At 256K and `Hkv=2,D=256`, one K/V
workspace costs about 512 MiB per rank. It is reused serially by all full
attention layers, rather than allocating one copy per layer. FP8 still saves
substantially more memory across the model than this single workspace costs.

Prefill microbenchmark, `M=1568,Hq=12,Hkv=2,D=256`:

| Context | Old FP8 page-1568 | Bridge total | Expansion only | Speedup |
|---:|---:|---:|---:|---:|
| 16K | 66.378 ms | 14.670 ms | 0.108 ms | 4.52x |
| 64K | 280.102 ms | 57.985 ms | 0.386 ms | 4.83x |
| 128K | 565.159 ms | 115.826 ms | 0.764 ms | 4.88x |

The bridge output and final attention output are bitwise equal to the old FP8
path for unit and non-unit scales.

MTP4-aligned microbenchmark, `M=1616`, page 1616:

| Context | Old FP8 | Bridge total | Expansion only | Speedup |
|---:|---:|---:|---:|---:|
| 16K | 61.500 ms | 15.904 ms | 0.108 ms | 3.87x |
| 64K | 255.623 ms | 62.905 ms | 0.387 ms | 4.06x |

The 16-row partial CTA is bitwise equal to the prior accurate FP8 path in the
targeted CUDA regression.

## Implemented Decode Path

The XQA panel loader now supports FP8 E5M2:

- one aligned 8-byte global load reads eight E5M2 values;
- register bit expansion produces eight FP16 values for shared-memory staging;
- the existing HMMA/XQA schedule then serves six query heads from one KV read;
- `k_scale` is applied with the score scale and `v_scale` with partition output,
  matching the scalar path's scale placement.

Non-graph execution uses scalar below 8K and XQA at or above 8K. CUDA-graph
capture is explicitly marked by the metadata builder and fixes the captured
FP8 route to XQA; a graph cannot switch kernels later as context grows.

Decode microbenchmark, page 1568:

| Context | FP8 scalar | FP8 XQA | Speedup |
|---:|---:|---:|---:|
| 4K | 0.137 ms | 0.136 ms | 1.00x |
| 8K | 0.204 ms | 0.153 ms | 1.34x |
| 16K | 0.286 ms | 0.227 ms | 1.26x |
| 64K | 0.841 ms | 0.577 ms | 1.46x |
| 128K | 1.437 ms | 1.053 ms | 1.37x |

The maximum scalar-versus-XQA error was at most `6.11e-5` in the tested
contexts and at most `3.05e-5` for non-unit scales.

### NCU Proof At 64K

| Metric | FP8 scalar | FP8 XQA |
|---|---:|---:|
| Partition kernel | 844.32 us | 534.18 us |
| Grid size | 768 CTAs | 128 CTAs |
| Registers/thread | 44 | 186 |
| Dynamic/shared memory | 12.88 KiB static | 43.26 KiB dynamic |
| Achieved occupancy | 53.50% | 12.49% |
| DRAM throughput | 7.24% | 11.59% |
| L2 hit rate | 87.21% | 4.43% |
| Memory throughput | 81.94 GB/s | 130.93 GB/s |

The lower XQA L2 hit rate is expected: scalar repeatedly rereads the same KV
for six query heads, while XQA removes those cache hits by removing the
redundant requests. XQA remains register/shared-memory limited and does not
saturate DRAM.

## Full-Model Result

Matched contract: Qwen3.6-27B-AWQ, TP2 V100, FP8 E5M2 KV, TurboMind AWQ,
Flash-V100, `max_model_len=262144`, `max_num_batched_tokens=2048`, prefix
caching, Mamba align, `FULL_AND_PIECEWISE`, no eager, 256 forced output tokens,
and official `temperature=1.0,top_p=0.95,top_k=20,seed=20260620` sampling.

| Context | Old FP8 prefill | New prefill | Prefill delta | Old FP8 TPOT | New TPOT | Decode tok/s delta |
|---:|---:|---:|---:|---:|---:|---:|
| 16K | 14.680 s | 10.317 s | -29.72% | 21.762 ms | 19.715 ms | +10.39% |
| 64K | 127.842 s | 62.964 s | -50.75% | 32.768 ms | 26.910 ms | +21.77% |

The bridge-only intermediate TPOT was 20.721 ms at 16K and 30.625 ms at 64K.
The remaining drop to 19.715/26.910 ms is the CUDA-graph FP8 XQA propagation.
Both final 256-token hashes exactly match the old FP8 baseline:

- 16K: `bbf3175c6946c021376d51d26ecc7c21ae8d3e22baed664b475508534cb8b587`
- 64K: `100b97590c707aecd4398eba34e12e2087f5b793ef8e0073492393c08ba04892`

### MTP4 Propagation

Matched MTP4 adds `num_speculative_tokens=4`, probabilistic draft sampling,
local argmax reduction, and an explicit Flash-V100 drafter. The old FP8 and
new FP8 runs use the same TP2 GPU pair and all other fields from the contract
above.

| Context | FP16 prefill control | Old FP8 prefill | New FP8 prefill | Prefill vs old FP8 | FP16 TPOT | Old FP8 TPOT | New FP8 TPOT | Acceptance length |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16K | 11.367 s | 13.977 s | 9.972 s | -28.66% | 10.056 ms | 10.104 ms | 10.072 ms | 4.8679 |
| 64K | 79.301 s | 123.412 s | 56.441 s | -54.27% | 17.832 ms | 18.179 ms | 18.020 ms | 4.8113 |

The new MTP4 output hashes equal both the old FP8 run and the matched no-MTP
run at 16K/64K. Acceptance length is unchanged. FP8 decode is now within about
1.1% of the FP16 MTP4 TPOT control at 64K; most of the prior visible MTP4 FP8
regression was prefill, because speculative acceptance already hid much of the
old scalar-decode cost.

## TP4 And Cross-Quantization Coverage

The TP4 acceptance contract matches the TP2 contract above except for four
V100s and `tensor_parallel_size=4`. Every full-model row below uses TurboMind,
Flash-V100, non-eager CUDA graphs, prefix caching, Mamba align,
`max_model_len=262144`, `max_num_batched_tokens=2048`, one request, 256 forced
output tokens, and official `temperature=1.0,top_p=0.95,top_k=20` sampling.

### TP4 Operator Shapes

The per-rank full-attention shapes differ enough that TP2-only evidence is not
sufficient:

| Model | TP4 shape | No-MTP page | MTP4 page | Production decode decision |
|---|---|---:|---:|---|
| Qwen3.6-27B | `Hq=6,Hkv=1,D=256,G=6` | 1568 | 1616 | FP8 XQA at long context |
| Qwen3.6-35B-A3B | `Hq=4,Hkv=1,D=256,G=4` | 1056 | 1088 | FP8 scalar |

The TP4 microbenchmarks are:

| Shape | Context | Old FP8 prefill | Bridge | Prefill speedup | FP8 scalar decode | FP8 XQA decode |
|---|---:|---:|---:|---:|---:|---:|
| 27B `Hq6/Hkv1` | 16K | 39.924 ms | 8.362 ms | 4.77x | 0.208 ms | 0.155 ms |
| 27B `Hq6/Hkv1` | 64K | 159.522 ms | 32.966 ms | 4.84x | 0.557 ms | 0.338 ms |
| 35B `Hq4/Hkv1` | 16K | 24.339 ms | 6.397 ms | 3.80x | 0.166 ms | 0.171 ms |
| 35B `Hq4/Hkv1` | 64K | 96.449 ms | 25.289 ms | 3.81x | 0.366 ms | 0.339 ms |

FP8 G4 XQA was not promoted. It loses at 16K, saved only 0.027 ms in the 64K
operator microbenchmark, and produced no full-model gain. Production therefore
keeps 35B TP4 FP8 decode on the scalar route while retaining the bridge for
prefill. The G4 XQA kernel remains covered by a scalar-versus-XQA numerical
regression, but is not a production route. Sampled hashes varied across both
XQA and fresh scalar processes, so that variation is not attributed to XQA.

### TP4 Full-Model No-MTP Matrix

The table compares current FP8 E5M2 KV against the matched FP16 KV control. A
positive TPOT delta is residual FP8 decode overhead; a negative prefill delta
is a bridge win over the FP16 control, not merely over the old generic FP8
path.

| Weight route | Context | FP16 prefill | FP8 prefill | Prefill delta | FP16 TPOT | FP8 TPOT | TPOT delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 27B AWQ | 16K | 5.064 s | 5.198 s | +2.65% | 13.706 ms | 13.924 ms | +1.59% |
| 27B AWQ | 64K | 30.213 s | 31.116 s | +2.99% | 17.283 ms | 17.458 ms | +1.02% |
| 35B-A3B AWQ | 16K | 2.101 s | 2.603 s | +23.87% | 9.074 ms | 9.322 ms | +2.73% |
| 35B-A3B AWQ | 64K | 14.111 s | 10.917 s | -22.63% | 11.328 ms | 11.459 ms | +1.16% |
| 35B-A3B FP8 weights | 16K | 2.024 s | 1.981 s | -2.10% | 9.781 ms | 9.961 ms | +1.84% |
| 35B-A3B FP8 weights | 64K | 14.618 s | 10.120 s | -30.77% | 12.043 ms | 12.101 ms | +0.49% |
| 27B NVFP4 | 16K | 6.167 s | 6.146 s | -0.35% | 14.337 ms | 14.504 ms | +1.16% |
| 27B NVFP4 | 64K | 35.147 s | 33.961 s | -3.37% | 17.890 ms | 18.050 ms | +0.90% |

The 27B NVFP4 logs contain `SM70 compressed-tensors NVFP4 TurboMind W4A16
dense path enabled`; no Marlin route is present. Compressed-tensors models
without a checkpoint `kv_cache_scheme` now use the same explicit unit-scale
E5M2 KV override as other compatible quantized checkpoints. An explicit
checkpoint KV scale scheme remains rejected.

The 27B AWQ and NVFP4 FP16/FP8 pairs have identical 16K and 64K output hashes.
The 35B sampled hashes can differ between FP16 and FP8 KV because the cache
dtype itself changes attention arithmetic. The relevant implementation gate is
stronger and more local: page 1056 and page 1088 bridge outputs are bitwise
equal to the previous accurate FP8 path for the tested full and partial query
chunks.

### TP4 MTP4 Pair

TP4 MTP originally failed before model execution because the default resolver
hard-coded the TP2 ranking artifact. The resolver now selects the ranking asset
from runtime tensor-parallel size, and all four ranks load
`sm70_mtp_dynamic_vocab_qwen36_27b_tp4.pt`.

| Context | FP16 prefill | FP8 prefill | Prefill delta | FP16 TPOT | FP8 TPOT | TPOT delta | FP16 AL | FP8 AL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16K | 6.189 s | 5.433 s | -12.21% | 7.393 ms | 7.577 ms | +2.50% | 4.8679 | 4.6364 |
| 64K | 44.639 s | 31.742 s | -28.89% | 11.421 ms | 11.390 ms | -0.27% | 4.8113 | 4.8113 |

At 64K, FP8 KV no longer adds MTP decode decay and cuts prefill by 28.89%.
The single 16K sample has a 4.76% acceptance-length difference and a different
sampled hash, so it is performance/route evidence, not a natural-prompt MTP
quality promotion. The 64K sample has identical acceptance length and output
hash. A natural-prompt multi-case quality gate is still required before making
a new 35B or FP8-MTP serving recommendation.

There is no valid 27B NVFP4 MTP body in the available checkpoint, and there is
no local 35B NVFP4 checkpoint. Those combinations are recorded as unavailable,
not silently replaced by AWQ. Full-model 35B MTP is also deferred until a
35B-specific dynamic-vocabulary ranking is generated; the 35B MTP page-1088
attention shape is covered by the bitwise operator regression.

## Runtime Controls

- `VLLM_FLASH_V100_FP8_PREFILL_BRIDGE=0`: disable the prefill bridge.
- `VLLM_FLASH_V100_DECODE_USE_XQA=0`: disable all XQA decode.
- `VLLM_FLASH_V100_DECODE_FP8_XQA_MIN_SEQ_LEN=8192`: non-graph FP8 XQA
  crossover threshold.

The bridge and FP8 XQA are enabled by default under their strict shape and
dtype gates. FP8 E4M3 is not routed through either E5M2 implementation.

## Tests And Artifacts

Tests:

- `tests/kernels/attention/test_sm70_flash_v100_varlen_layout.py`
- `tests/quantization/test_fp8.py`
- `tests/v1/attention/test_sm70_flash_v100_policy.py`
- `tests/v1/spec_decode/test_static_draft_vocab.py`

Primary artifacts:

- `bench_results/flash_v100_fp8_kv_20260715/baseline.json`
- `bench_results/flash_v100_fp8_kv_20260715/bridge_v2_stride_fixed.json`
- `bench_results/flash_v100_fp8_kv_20260715/fp8_xqa_v1.json`
- `bench_results/flash_v100_fp8_kv_20260715/mtp_m1616_tail_bridge_micro.json`
- `bench_results/flash_v100_fp8_kv_20260715/ncu_decode_fp8_xqa_n64k.ncu-rep`
- `bench_results/flash_v100_fp8_kv_20260715/full_model_candidate_i16k_i64k.json`
- `bench_results/flash_v100_fp8_kv_20260715/full_model_candidate_graph_xqa_v2_i16k_i64k.json`
- `bench_results/flash_v100_fp8_kv_20260715/full_model_mtp4_graph_xqa_tail_v2_i16k_i64k.json`
- `bench_results/flash_v100_fp8_kv_20260715/tp4_27b_awq_fp8kv_i16k_i64k.json`
- `bench_results/flash_v100_fp8_kv_20260715/tp4_27b_awq_mtp4_fp8kv_i16k_i64k_tp4asset.json`
- `bench_results/flash_v100_fp8_kv_20260715/tp4_35b_awq_fp8kv_safe_g4scalar_i16k_i64k.json`
- `bench_results/flash_v100_fp8_kv_20260715/tp4_35b_fp8weights_fp8kv_safe_i16k_i64k.json`
- `bench_results/flash_v100_fp8_kv_20260715/tp4_27b_nvfp4_turbomind_fp8kv_i16k_i64k.json`
- `bench_results/mtp_flash_drafter_20260715/long_context_tp2/no_mtp_fp8kv_mbt2048_g01.json`

## Closed And Remaining Work

Do not retry scalar E5M2 conversion tuning as the primary solution. Conversion
was not the decode bottleneck, and prefill needed conversion reuse rather than
a cheaper repeated conversion.

The remaining XQA kernel has 186 registers/thread, one CTA/SM, and 73.3% cycles
with no eligible warp in NCU. Reducing that resource footprint is a valid next
kernel project, but the first-order FP8 long-context regression is closed.
Before further work, measure its share of final TPOT; do not optimize the
534-us partition kernel as though it were the entire model token latency.

The 256K full-model point remains unmeasured after this change because the
matched run is expensive. The 128K operator microbenchmark is complete, and a
future 256K full-model run should be a confirmation gate rather than another
route-discovery experiment.
