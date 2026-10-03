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
