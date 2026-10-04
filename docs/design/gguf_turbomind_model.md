# TurboMind GGUF dense projection integration

GGUF linear layers prepare independent canonical projections during loading.
Each projection retains its source codec and chooses the existing affine,
LUT4 or lattice mixed-precision kernel. The model scheduler, attention and
CUDA graph lifecycle retain their existing contracts. Mixed fused projections
are evaluated in their logical order and concatenated without treating one
packed format as another.

The first model workload is Qwen3.8-27B UD-Q4_K_M with TP4. Its mixed FFN
weights require affine, LUT4 and lattice preparation. Four FFN projections
use IQ3_S despite the checkpoint's Q4_K_M name. Output tails zero-pad canonical rows to the converter pack size and crop
the result back to its logical width. Unsupported canonical coefficients
retain the packaged fallback and report their rejection reason. GDN input layout transforms compose with canonical
storage. Embedding and packed-row PLE preparation are separate integration
scopes. Flash-Next stays TP4.

## Validation

The underlying operators have official dequantization oracles and matched
real-shape timings in the family design documents. This layer additionally
requires mixed projection and layout checks, an ordinary installed-wheel
route check, greedy/logit distribution comparisons and model quality checks.
Model timings must separate prefill from steady decode and report C1/C4/C8/C16
and 8K/32K prompts. Operator timings are not model throughput evidence.

## Layer checks

Fifteen GPU checks pass on V100 32GB, CUDA 12.8 and Torch 2.10.0+cu128.
Independent affine, LUT4 and all seven lattice formats match official
dequantization with FP32 accumulation (relative L2 below 0.003), including
graph replay and full-graph tracing. Mixed affine/LUT4/lattice/FP16 projections
retain their logical order when loaded in a different file order; GDN input
tiling and bias compose with those projections. Preparation preserves shared
source parameters and reports incomplete output packs explicitly.

All 13 existing Qwen3.5 adapter tests also pass. The first layer check uses
the normal main operator artifact with SHA256
`4910c47ab1aaed253001d5950bf44dd40a350b2b087202a8ea2b13f2c5457782`.

## Installed artifact and model correctness

An ordinary wheel in a fresh runtime passes all 15 layer GPU checks with
210 compatible dependencies. The source and installed canonical extension
are the normal main artifact, SHA256
`5cd0fa29e533f92644e012c57fe7b439293bf360e8988b8d73d7bbef54839f6a`.
Wheel `1cat_vllm-1.5.2.dev416+gba6295209.precompiled-cp312-cp312-linux_x86_64.whl`
has SHA256 `b25b8acbcb370df4a0e07a3316720aca4edb39cf746e9f7d625e8f1f28c81c47`.
No private extension or Python-path override is required.

The Qwen3.5-0.8B TP4 regression check keeps 64/64 English and arithmetic
tokens, with first-logit RMSE reduced to 0.109217/0.118177/0.144899 across
the three fixed prompts. Chinese still diverges after token 23; this issue
is not resolved. Embedded chat results remain `Paris`, `4`, and `你好`.

Qwen3.8-27B UD-Q4_K_M completes installed-wheel TP4 inference with FP16,
maxlen 2048, maxbatch 256, maxseqs 4, memory utilization 0.3, eager and no MTP.
Worker logs select the three canonical families and Flash-V100/FlashQLA.
The four fixed GGUF/HF chat tokenizations match exactly. Greedy output bodies
match pinned CPU llama.cpp for `Paris`, arithmetic, translation and a
29-token Chinese explanation of lunar phases; all stop normally.

| Prompt | First-logit RMSE | Relative L2 | Reference-to-actual KL | Top-20 overlap |
| --- | ---: | ---: | ---: | ---: |
| Paris | 0.114157 | 0.036342 | 0.00006944 | 18/20 |
| Arithmetic | 0.175268 | 0.065481 | 0.00003251 | 18/20 |
| Translation | 0.127382 | 0.046972 | 0.00005601 | 20/20 |
| Lunar phases | 0.077471 | 0.036478 | 0.00374037 | 19/20 |

All four top-1 logits match. These short checks establish model operation;
the common quality set and C1/C4/C8/C16 plus 8K/32K performance comparisons
are still pending. The 27B IQ3_S gate/down TP4 shapes additionally require
prefill crossover measurements; the existing Flash-Next calibration does
not establish their crossover.

## 27B IQ3_S operator crossovers

Real `blk.11.ffn_gate.weight` and `blk.14.ffn_down.weight` use IQ3_S.
TP4 yields gate N=4352/K=5120 and down N=5120/K=4352. The existing shared
FP16 scratch fits both shapes. Canonical dequantization retains FP16 weight
reconstruction and explicit FP32 cuBLAS accumulation; no activation quantization
or reduced-precision reduction is introduced. The full M sweep below uses
V100 32GB, CUDA 12.8, Torch 2.10.0+cu128, 100 ms warmup per route and 100
CUDA graph timing iterations. All columns are microseconds.

Small outputs capture eight invocations per replay; outputs above ten million
elements capture one. Each route returns its output so this rule applies
consistently. Raw DQ uses the original GGUF reader/operator; canonical DQ uses
the expanded FP16 coefficients. AWQ is a same-shape comparator, not a claim
that both checkpoints have identical quantized values.

### Gate

| M | TM fused | Canonical DQ+FP32 BLAS | AWQ | llama MMVQ/MMQ | Raw DQ+cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.15 | 195.44 | 27.46 | 30.57 | 343.33 |
| 2 | 31.80 | 172.44 | 27.54 | 32.48 | 370.08 |
| 4 | 32.03 | 173.20 | 27.72 | 40.69 | 365.12 |
| 8 | 33.11 | 174.38 | 28.56 | 67.35 | 359.68 |
| 16 | 39.55 | 175.36 | 35.56 | 77.21 | 357.00 |
| 32 | 55.80 | 174.74 | 45.28 | 101.26 | 358.70 |
| 64 | 92.89 | 191.95 | 75.37 | 141.96 | 399.98 |
| 128 | 130.69 | 230.94 | 111.50 | 236.39 | 430.35 |
| 512 | 497.03 | 367.42 | 463.58 | 780.07 | 568.65 |
| 2048 | 1660.44 | 1228.45 | 1388.76 | 2981.29 | 1471.85 |
| 8192 | 6371.06 | 4221.96 | 5191.27 | 11812.26 | 4819.49 |

### Down

| M | TM fused | Canonical DQ+FP32 BLAS | AWQ | llama MMVQ/MMQ | Raw DQ+cuBLAS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.37 | 199.41 | 40.89 | 31.92 | 344.19 |
| 2 | 30.82 | 173.32 | 40.97 | 35.40 | 355.00 |
| 4 | 31.04 | 174.85 | 41.00 | 44.49 | 355.51 |
| 8 | 34.12 | 177.58 | 25.84 | 64.20 | 356.65 |
| 16 | 37.89 | 182.43 | 31.61 | 71.78 | 358.35 |
| 32 | 50.09 | 189.61 | 40.01 | 96.12 | 354.47 |
| 64 | 83.20 | 181.79 | 64.99 | 134.18 | 397.91 |
| 128 | 116.04 | 226.38 | 103.10 | 218.82 | 430.23 |
| 512 | 378.01 | 353.93 | 303.20 | 729.35 | 548.57 |
| 2048 | 1522.90 | 1110.55 | 1230.97 | 2897.73 | 1395.33 |
| 8192 | 6445.07 | 4151.62 | 5069.11 | 11497.36 | 4746.95 |

At M=128, fusion remains faster than canonical DQ. At M=512 and above,
canonical DQ is faster in both measured shapes, so capability bands admit
those IQ3_S descriptors for M>=512. Other descriptors retain their existing
calibration or fused fallback. M=8192 saves about 34–36% relative to fusion,
and is faster than the same-shape AWQ comparator.

Gate/down expanded-scale relative L2 is 0.0002040/0.0002030; max absolute
weight error is 0.00005722/0.00005007. Output relative L2 is approximately
0.000404. The benchmark core fingerprint is
`5cd0fa29e533f92644e012c57fe7b439293bf360e8988b8d73d7bbef54839f6a`.
The complete route sweep precedes model throughput measurement.

## Small projection row packs and FP16 cache

TP4 GDN alpha/beta projections have logical N=12/K=5120. All canonical
families now pad incomplete N32 row packs with zero coefficients and crop
outputs to the logical width. Integer/index/sign payloads and real rows are
unchanged. Mixed projections and source parameter sharing retain their
existing contracts.

For the real Q8_0 alpha projection, the full M sweep shows an inexpensive
FP16 weight cache beats padded integer GEMM at M=1 and M>=32, while
cuBLAS transpose algorithms regress at M=2–16. The framework admits the
measured descriptor at M=1 and M=32–8192; other M retains packed MMA.
Unmeasured descriptors report `small_projection_cache_shape_has_no_calibration`.
Cache admission requires FP16 activations and both FP16 reduced reductions
and FP16 accumulation disabled, otherwise it reports `requires_fp32_matmul_policy`.
This uses the normal worker precision policy and adds no environment variable.

Real-weight graph timings (us), V100 32GB/CUDA12.8/Torch2.10, 100 ms warmup
and 100 iterations, eight calls per replay:

| M | N32 packed MMA | Existing Q8 activation route | Cached FP16 |
| ---: | ---: | ---: | ---: |
| 1 | 19.04 | 8.39 | 4.47 |
| 2 | 19.58 | 8.35 | 76.50 |
| 4 | 16.87 | 7.28 | 72.53 |
| 8 | 17.44 | 38.62 | 74.51 |
| 16 | 19.58 | 39.07 | 76.85 |
| 32 | 18.10 | 39.88 | 7.59 |
| 64 | 65.31 | 41.08 | 6.46 |
| 128 | 111.59 | 43.58 | 7.38 |
| 512 | 111.41 | 72.61 | 14.34 |
| 2048 | 105.19 | 185.57 | 43.24 |
| 8192 | 332.61 | 698.26 | 155.16 |

A K-major cache padded to N16 was also measured: M=2–16 remained slower
than packed MMA (29–30 us), while larger M changed only modestly. Retain
one simple N-major FP16 cache. The normalized FP16 and cache output relative
L2 is about 0.00020–0.00029, versus 0.0047–0.0088 for the existing activation
quantization route in this sweep. No activation precision is reduced.

## Installed graph model measurement

The dense integration runs with the normal packaged CUDA extensions. Benchmark
helper imports restore the caller's module search path before spawning workers;
otherwise their standalone source setup can shadow the installed Flash-V100
package. The previous failed graph startup came from that source-path shadowing,
not missing wheel extensions. Both Flash-V100 extensions import from the installed
package, and worker logs confirm Flash-V100 attention and FlashQLA GDN dispatch.

Qwen3.8-27B UD-Q4_K_M, V100 SXM2 32GB x4, TP4, CUDA 12.8, Torch 2.10,
FP16 activations/KV, no MTP, prefix caching disabled, maximum length 33024,
batch budget 8192, maximum 16 sequences, GPU memory utilization 0.7,
FULL_AND_PIECEWISE graphs with captures 1/2/4/8/16:

| Concurrent requests | Aggregate pure decode (tok/s) | Mean step (ms) |
| ---: | ---: | ---: |
| 1 | 47.51 | 21.05 |
| 4 | 178.38 | 22.42 |
| 8 | 319.63 | 25.03 |
| 16 | 555.05 | 28.83 |

Each point averages two fixed-length runs after warmup: 1024 input tokens,
128 generated tokens, greedy/ignore-EOS synthetic timing, complete atomic cohorts.
Pure decode excludes prefill and the first/last eight eligible engine intervals.
Separate natural greedy checks retain EOS: all four complete normally, including
the Chinese lunar explanation matching the earlier llama.cpp reference.

| Input tokens | Prefill (s) | TTFT (s) |
| ---: | ---: | ---: |
| 8192 | 2.5154 | 2.5186 |
| 32768 | 10.6431 | 10.6538 |

Prefill averages two runs with one generated token after warmup. These results
exclude startup: weight preparation took about 237 seconds and initial graph
capture about 240 seconds. The packaged core fingerprint remains
`5cd0fa29e533f92644e012c57fe7b439293bf360e8988b8d73d7bbef54839f6a`.
The same workload on the Qwen3.8-27B QUASAR NVFP4 checkpoint gives:

| Concurrent requests | GGUF decode (tok/s) | NVFP4 decode (tok/s) | GGUF/NVFP4 |
| ---: | ---: | ---: | ---: |
| 1 | 47.51 | 66.75 | 0.712 |
| 4 | 178.38 | 251.26 | 0.710 |
| 8 | 319.63 | 470.88 | 0.679 |
| 16 | 555.05 | 830.51 | 0.668 |

NVFP4 prefill is 2.5089 s at 8K and 10.4705 s at 32K, so GGUF is
approximately 0.3%/1.6% slower. Prefill is close to the native path; decode
throughput still trails by 29–33%. Both checkpoints complete the four natural
text checks, but this does not substitute for the common quality set, which
remains pending. Investigate the C4 graph kernel breakdown before choosing a
decode optimization. Trace instrumentation is separate from accepted unprofiled
throughput and does not change the kernel precision policy.

## Compatible projection coalescing

C4 graph-node traces contain 127 complete replays across four TP ranks, using
125 middle steps for aggregate analysis. Graph-node coverage is 97.5% for GGUF
and 96.5% for NVFP4. The leading TurboMind kernels account for approximately
15.15/9.00 ms of rank-average GPU service and 590/256 launches per step in
GGUF/NVFP4. Attention and GDN decode service are approximately 1.00/0.67 ms in
both. GPU service sums describe work and cannot be added into wall critical
path; unprofiled C4 step time remains 22.424/15.920 ms. The generic parser labels
FP4 GEMM as MoE, but this model is dense; the comparison above uses actual
TurboMind kernel names rather than that generic label.

Coalesce adjacent shards with identical source type, packed input width, dtype
and device before canonical preparation. Keep mixed types as independent
projections and preserve the original row order. Startup reports retain each
source output size. This reduces repeated launches without adding a decoder,
changing activations, or storing duplicate prepared copies.

Real TP4 projection sweeps below use FP16 activation/reconstruction and FP32
accumulation, 100 ms warmup and 100 graph iterations. QKV spans and gate rows
come from the named source tensors; numerical reference uses official source
dequantization. The alpha/beta cache descriptor N=24/K=5120 admits all measured
M=1..8192, with the existing FP32 matmul policy guard. Its N=12 predecessor
retains its narrower calibration. Large merged affine shapes use canonical
DQ+FP32 cuBLAS only after their measured crossover; direct fusion alone can
regress prefill.

| Bits/group | K | N | Minimum M for DQ+FP32 BLAS |
| ---: | ---: | ---: | ---: |
| 4/32 | 5120 | 2560 | 2048 |
| 4/32 | 5120 | 4096 | 512 |
| 4/32 | 5120 | 8704 | 512 |
| 5/32 | 5120 | 2560 | 2048 |
| 5/32 | 5120 | 4096 | 512 |

All other measured merged descriptors below retain fused MMA. Output relative
L2 versus the official FP16 weight oracle stays below 0.00081 in these sweeps.
No lower activation or accumulation precision is introduced. Model throughput
and quality after coalescing still need installed-wheel validation.

### gdn_qkvz (N=4096, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 93.92 | 32.71 | 182.14 | 32.71 |
| 2 | 94.87 | 32.85 | 162.10 | 32.85 |
| 4 | 97.14 | 33.11 | 189.99 | 33.11 |
| 8 | 103.15 | 35.85 | 162.85 | 35.85 |
| 16 | 105.25 | 36.73 | 191.26 | 36.73 |
| 32 | 117.41 | 45.77 | 177.72 | 45.77 |
| 64 | 214.02 | 76.40 | 198.46 | 76.40 |
| 128 | 407.59 | 118.78 | 244.10 | 118.78 |
| 512 | 555.17 | 423.55 | 395.60 | 395.60 |
| 2048 | 1600.88 | 1763.39 | 1422.14 | 1422.14 |
| 8192 | 4938.86 | 6117.05 | 5007.00 | 5007.00 |

### gdn_alpha_beta (N=24, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.94 | 19.01 | 4.61 | 4.61 |
| 2 | 40.16 | 20.14 | 5.34 | 5.34 |
| 4 | 35.45 | 18.53 | 5.39 | 5.39 |
| 8 | 36.99 | 19.69 | 5.59 | 5.59 |
| 16 | 40.34 | 20.35 | 6.08 | 6.08 |
| 32 | 16.54 | 19.81 | 6.41 | 6.41 |
| 64 | 14.39 | 66.35 | 6.83 | 6.83 |
| 128 | 16.27 | 112.04 | 6.89 | 6.89 |
| 512 | 30.72 | 111.75 | 12.23 | 12.23 |
| 2048 | 89.67 | 105.15 | 40.67 | 40.67 |
| 8192 | 308.03 | 328.55 | 142.64 | 142.64 |

### ffn_gate_up (N=8704, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 61.39 | 43.33 | — | 43.33 |
| 2 | 62.41 | 45.28 | — | 45.28 |
| 4 | 59.59 | 45.52 | — | 45.52 |
| 8 | 66.33 | 46.86 | — | 46.86 |
| 16 | 67.87 | 53.61 | — | 53.61 |
| 32 | 96.13 | 78.26 | — | 78.26 |
| 64 | 177.52 | 137.50 | — | 137.50 |
| 128 | 250.05 | 221.90 | — | 221.90 |
| 512 | 987.94 | 837.47 | — | 837.47 |
| 2048 | 3340.85 | 2874.37 | — | 2874.37 |
| 8192 | 12178.69 | 11885.67 | — | 11885.67 |

### qkvz_q4k (N=4096, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 67.82 | 26.46 | 175.95 | 26.46 |
| 2 | 68.35 | 26.49 | 169.99 | 26.49 |
| 4 | 69.01 | 26.51 | 157.46 | 26.51 |
| 8 | 73.64 | 27.29 | 170.84 | 27.29 |
| 16 | 81.14 | 32.02 | 158.91 | 32.02 |
| 32 | 92.52 | 40.93 | 186.33 | 40.93 |
| 64 | 172.24 | 57.15 | 161.74 | 57.15 |
| 128 | 365.80 | 99.53 | 253.55 | 99.53 |
| 512 | 477.57 | 377.97 | 367.84 | 367.84 |
| 2048 | 1583.04 | 1538.95 | 1446.89 | 1446.89 |
| 8192 | 5800.51 | 5122.84 | 5000.23 | 5000.23 |

### qkv_q4k (N=2560, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 53.07 | 21.65 | 88.10 | 21.65 |
| 2 | 51.02 | 21.10 | 97.08 | 21.10 |
| 4 | 51.26 | 21.14 | 98.54 | 21.14 |
| 8 | 54.53 | 21.87 | 100.36 | 21.87 |
| 16 | 58.85 | 26.42 | 103.73 | 26.42 |
| 32 | 65.77 | 33.27 | 101.62 | 33.27 |
| 64 | 130.66 | 48.57 | 107.71 | 48.57 |
| 128 | 279.76 | 93.58 | 201.09 | 93.58 |
| 512 | 337.12 | 200.79 | 302.34 | 200.79 |
| 2048 | 996.15 | 756.03 | 615.67 | 615.67 |
| 8192 | 3850.63 | 3169.34 | 2425.79 | 2425.79 |

### qkv_q5k (N=2560, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 70.91 | 27.35 | 90.36 | 27.35 |
| 2 | 70.50 | 27.32 | 99.28 | 27.32 |
| 4 | 70.61 | 27.29 | 100.87 | 27.29 |
| 8 | 75.90 | 28.87 | 102.59 | 28.87 |
| 16 | 77.37 | 29.89 | 105.93 | 29.89 |
| 32 | 82.33 | 36.62 | 103.54 | 36.62 |
| 64 | 164.14 | 56.20 | 110.11 | 56.20 |
| 128 | 304.87 | 102.04 | 202.86 | 102.04 |
| 512 | 381.66 | 235.98 | 304.12 | 235.98 |
| 2048 | 1030.77 | 912.94 | 613.81 | 613.81 |
| 8192 | 3492.54 | 3884.11 | 2426.44 | 2426.44 |

### qkvz_lut4 (N=4096, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 87.85 | 28.36 | — | 28.36 |
| 2 | 88.85 | 28.35 | — | 28.35 |
| 4 | 88.57 | 28.52 | — | 28.52 |
| 8 | 94.95 | 31.21 | — | 31.21 |
| 16 | 99.05 | 32.82 | — | 32.82 |
| 32 | 107.19 | 42.77 | — | 42.77 |
| 64 | 215.65 | 74.58 | — | 74.58 |
| 128 | 381.45 | 108.16 | — | 108.16 |
| 512 | 506.05 | 408.25 | — | 408.25 |
| 2048 | 1711.99 | 1665.93 | — | 1665.93 |
| 8192 | 6203.41 | 5585.90 | — | 5585.90 |

### ffn_gate_up_q4k (N=8704, K=5120)

| M | Separate (us) | Fused MMA (us) | FP16 cache or DQ+FP32 BLAS (us) | Selected (us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 57.68 | 43.54 | 392.09 | 43.54 |
| 2 | 57.82 | 43.88 | 309.55 | 43.88 |
| 4 | 57.59 | 43.97 | 314.80 | 43.97 |
| 8 | 59.25 | 44.73 | 317.70 | 44.73 |
| 16 | 68.68 | 50.55 | 330.13 | 50.55 |
| 32 | 88.54 | 65.77 | 346.15 | 65.77 |
| 64 | 142.66 | 111.25 | 370.55 | 111.25 |
| 128 | 241.31 | 214.21 | 350.21 | 214.21 |
| 512 | 757.39 | 781.03 | 734.51 | 734.51 |
| 2048 | 2534.52 | 2628.16 | 2228.99 | 2228.99 |
| 8192 | 8798.97 | 10979.79 | 8413.24 | 8413.24 |

## Installed model result after coalescing

The updated ordinary wheel passes all 12 targeted coalescing, cache and
crossover GPU checks in a fresh environment. Source Python, wheel members and
installed Python match exactly; the CUDA core fingerprint is unchanged. Reuse
the matched native NVFP4 control because its implementation and runtime are
unchanged. Model workload and sampling remain the same as the preceding
unprofiled measurement.

| Concurrency | Previous GGUF (tok/s) | Coalesced GGUF (tok/s) | Native NVFP4 (tok/s) | GGUF improvement | Gap to native |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.51 | 58.88 | 66.75 | +24.0% | -11.8% |
| 4 | 178.38 | 216.73 | 251.26 | +21.5% | -13.7% |
| 8 | 319.63 | 396.87 | 470.88 | +24.2% | -15.7% |
| 16 | 555.05 | 671.47 | 830.51 | +21.0% | -19.1% |

| Input tokens | Previous GGUF prefill (s) | Coalesced GGUF (s) | Native NVFP4 (s) |
| ---: | ---: | ---: | ---: |
| 8192 | 2.5154 | 2.4228 | 2.5089 |
| 32768 | 10.6431 | 10.2626 | 10.4705 |

All four natural greedy token sequences and finish reasons match the preceding
GGUF measurement and its llama.cpp reference. Prefill improves rather than
regressing, and decode throughput improves by 21–24%. Decode remains 12–19%
below the native NVFP4 control, so this is an improvement rather than a claim
of complete performance parity. The frozen common quality set is running
separately; its scores are still pending.

Updated wheel fingerprint:
`ff173a2e75d44de5fb14fe9507963f2ab0d8d0c0596770d1d6e96a7238d8aa6e`.
