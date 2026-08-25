# SM70 Qwen3.8 FP8 Prefill Decay

Date: 2026-08-15

## Contract

- Source base: `87ac589295ba64399695ef2237c37cffc0d8b71b`.
- Model: Qwen3.8-27B-FP8 local checkpoint.
- Model weights: FP8 E4M3. This is independent from the KV-cache dtype.
- FP8 KV cache on SM70/Flash-V100: FP8 E5M2. The `fp8` CLI shorthand
  resolves to E5M2 on this route; explicit E4M3 remains explicit E4M3.
- Hardware: TP4 on V100-SXM2-16GB GPUs 0-3.
- Runtime: Python 3.12, Torch 2.10.0+cu128, `FLASH_ATTN_V100`, prefix
  caching with Mamba align, CUDA graphs, and no MTP. The chunk baseline and
  128K trace use FP16 KV; the later regression and fixed-route sections use
  the explicitly stated FP8 KV dtype.
- Sampling: temperature 1.0, top-p 0.95, top-k 20, seed 20260815.

## Chunk Baseline

Every request resets the prefix cache and emits 32 tokens. Output hashes are
stable and identical between the two chunk configurations.

| Input | Chunk 15680 | Chunk 4096 | Chunk 15680 gain |
|---:|---:|---:|---:|
| 4K | 4160.0 tok/s | 4146.1 tok/s | 0.33% |
| 16K | 4290.1 tok/s | 3841.2 tok/s | 11.69% |
| 64K | 3420.2 tok/s | 2778.9 tok/s | 23.07% |
| 128K | 2702.9 tok/s | 2020.5 tok/s | 33.77% |

Chunk 15680 is retained. The greater-than-4K peak is present on the current
machine; the unresolved issue is long-context decay, not a missing dispatch.

## 128K Critical-Rank Trace

The profiled prefill wall is 49.821 seconds versus 48.492 seconds unprofiled.
Critical-rank kernel attribution is:

| Category | Kernel time | Profiled wall share |
|---|---:|---:|
| D256 exact-dense attention | 17.710 s | 35.55% |
| D256 direct-paged attention | 3.099 s | 6.22% |
| FP8 exact-dense projections | 12.787 s | 25.67% |
| TurboMind FP8 projections | 5.551 s | 11.14% |
| TP communication | 4.895 s | 9.82% |
| GDN / linear attention | 1.640 s | 3.29% |
| Other FP16 GEMM | 1.547 s | 3.11% |
| Norm / elementwise / sampling | 1.240 s | 2.49% |
| KV cache and gather | 0.042 s | 0.08% |
| Host and unattributed residual | 1.309 s | 2.63% |

The 128 exact-dense calls are eight full chunks times 16 full-attention
layers. Mean per-layer latency grows from 18.50 ms in the first group to
256.59 ms in the eighth group.

## Exact-Shape NCU

The isolated shape is `Q=15680, KV=125440, Hq=6, Hkv=1, D=256`. The
unprofiled accepted operator is 242.954 ms median, 46.63 causal TFLOP/s, and
has output SHA256
`71653010ebcad1ff38a231ad92cc970fe47491f14d570f29b81f963c3b794862`.

- 253-254 registers/thread, 45.57 KiB dynamic shared memory, one CTA/SM,
  12.5% occupancy, and 18.38 waves.
- Tensor pipe active: 38.67%; issue active: 36.75%.
- Schedulers have no eligible warp for 62.14% of cycles.
- DRAM throughput is only 9.06% (81.65 GB/s); L2 hit rate is 88.51%.
- MIO throttle is the largest per-issue stall. Eight P-operand `LDS.128`
  instructions each account for about 172.9 million excessive shared
  wavefronts.
- The dominant long-scoreboard PC is the next-K shared publication waiting
  for its global register load. Earlier publication is already a closed
  regression because it extends the fragment across the register-dense loop.

This is a CTA-local shared-memory and dependency problem, not an HBM or grid
parallelism limit.

## Rejected Pre-Fix FP8-Alias Run

The same TP4, no-MTP contract was rerun with `kv_cache_dtype=fp8` and
`max_num_batched_tokens=8192`. At that point, the generic `fp8` alias was
incorrectly left as E4M3. These numbers describe the bug and are not an E5M2
baseline.

| Input | Prefill | FP16 KV/chunk 15680 | Relative | Decode |
|---:|---:|---:|---:|---:|
| 4K | 3529.8 tok/s | 4160.0 tok/s | -15.1% | 53.57 tok/s |
| 16K | 2479.0 tok/s | 4290.1 tok/s | -42.2% | 52.27 tok/s |
| 64K | 859.3 tok/s | 3420.2 tok/s | -74.9% | 32.55 tok/s |
| 128K | 460.3 tok/s | 2702.9 tok/s | -83.0% | 22.91 tok/s |

The configuration provides 685,491 KV tokens and 2.61x nominal 256K request
capacity, versus 294,431 tokens and 1.12x for FP16 KV/chunk 15680. The severe
regression came from selecting the wrong KV format and therefore missing every
SM70 E5M2 fast path:

- E4M3 prefix chunks cannot enter the D256 exact-dense path because that path
  currently requires FP16 K/V cache tensors.
- The existing one-pass FP8-to-FP16 bridge supports E5M2 only. E4M3 therefore
  uses direct paged prefill with software dequantization in the attention loop.
- Decode selects the scalar-paged E4M3 route instead of XQA, causing additional
  long-context decay.
- Model E4M3 weights do not imply an E4M3 KV cache. Weight quantization must
  never be used as a KV dispatch condition.

The fix resolves the SM70/Flash-V100 `fp8` alias to E5M2 while preserving an
explicit `fp8_e4m3` request. E5M2 then enters the exact-dense prefill bridge,
native cache writer, and scalar/XQA decode routes. Current E5M2 decode has a
lower long-context slope than FP16 and is faster at 128K and 256K; the final
matched prefill/decode table is recorded after the full acceptance sweep.

## Rejected Candidate

The first candidate reused the conflict-free native PV matrix-A layout but
kept direct logical stores from the QK accumulator, avoiding the previously
rejected shuffle/repack function.

- PTXAS: 253 registers/thread, zero stack, zero spill, unchanged shared size.
- Quality: exact output hash.
- Wall: 242.954 ms control to 254.450 ms candidate, a 4.73% regression.

The extra address-generation/scalar-publication issue cost exceeds the saved
P-load replay. Do not retry this layout spelling or the earlier
shuffle/repack form.

## 2026-08-24 Exact-D256 K-Stage Ping-Pong

The accepted exact-N32 arithmetic order is unchanged. The candidate alternates
K D64 panels between two non-overlapping shared-memory stages. The second K
stage borrows bytes from the later V/P allocation: K is dead before V is
loaded and P is materialized, so those lifetimes do not overlap. A next-K
fragment is published to the stage that the current QK phase is not reading,
so the pre-overwrite full-CTA barrier is no longer required. The visibility
barrier before the next QK phase remains.

The K layout's logical size is 2048 half elements, but its padded physical
`cosize` is 2188. The final stage stride is therefore 2240 half elements: it
is larger than the physical span, 16-byte aligned for `STS.128`, and keeps
both stages on the same 128-byte shared-bank phase. The final lifetime-aliased
layout leaves dynamic shared memory at 45,568 bytes. Dense/split-KV3 kernels
remain at 254/253 registers per thread with zero stack or spill.

SASS for the exact dense nonpaged specialization changes as follows:

- static `BAR.SYNC.DEFER_BLOCKING` count: 15 to 12;
- HMMA step counts: 128 each, unchanged;
- `LDG`, `LDS`, `STS`, `SHFL`, and `MUFU` counts: unchanged;
- `FADD` and `IADD3` static counts: 15 to 12 and 71 to 68;
- full SASS from the clean repository-patch build is byte-identical to the
  measured lifetime-alias artifact.

The initial fully-disjoint v5 stage layout was screened across the length
curve. Same-build, separate-process A/B/A CUDA-event measurements use FP16,
`Hq=6`, `Hkv=1`, D256, causal attention, five warmups, five queued calls per
sample, and seven samples. The control column is the mean of the two bracketing
control medians except for the first 8K chunk, where the first process was a
visible clock-ramp outlier and the hot second control is reported.

| Q | KV | Control | Ping-pong | Latency change | Exact |
|---:|---:|---:|---:|---:|---:|
| 8192 | 8192 | 4.8259 ms | 4.8302 ms | +0.09% | bitwise |
| 8192 | 32768 | 32.5750 ms | 32.2132 ms | -1.11% | bitwise |
| 8192 | 65536 | 68.2283 ms | 67.7216 ms | -0.74% | bitwise |
| 8192 | 131072 | 142.3325 ms | 140.9176 ms | -0.99% | bitwise |
| 15680 | 125440 | 251.6402 ms | 250.2695 ms | -0.54% | bitwise |

A second 128K run from the clean v5 build measured
`142.1020 -> 141.5512 ms` (`-0.39%`) and was again bitwise exact. The final
lifetime-alias layout was then compared with v5 in both execution orders,
each bracketed by fresh controls. Across the two orders, pooled control, v5,
and lifetime-alias medians are `142.4068`, `141.8076`, and `141.7669 ms`:
`-0.42%` for v5 and `-0.45%` for the final layout. The `0.03%` difference
between candidates is noise-sized; the final layout is selected because it
does not enlarge shared memory or alter the original V/P addresses.

Direct paged and split-KV3 gates at `Q4096/KV32768` are bitwise equal over
6,291,456 elements each; the dense gate covers 12,582,912 elements. The gain
is small but length-directed: the hot first chunk is neutral while long-prefix
calls consistently improve.

Closed intermediate variants must not be repeated:

- 2048-half stage spacing overlaps the 2188-half K physical span and changes
  almost every output element;
- 2188-half spacing removes overlap but misaligns the second stage for
  `STS.128` and raises a CUDA misaligned-address fault;
- 2192-half spacing is exact but loses address/bank-phase efficiency;
- 2304-half spacing is exact but is slower than the minimal same-phase 2240
  spacing;
- allocating the two 2240-half stages as fully disjoint storage is exact and
  performance-equivalent, but unnecessarily grows shared memory to 46,336
  bytes; the lifetime-alias form retains the original 45,568-byte envelope.

Nsight Compute recognized the exact mangled kernel but the host driver denied
performance-counter access with `ERR_NVGPUCTRPERM`. No privilege or machine
security setting was changed. CUDA-event timing and static SASS are therefore
the timing and structural authorities for this experiment.

The matched end-to-end TP4 128K A/B/A gate also passes. It fixes Qwen3.8-27B
FP8 weights, 131,072 input tokens, 256 sampled output tokens, E5M2 KV, chunk
8192, no MTP, prefix caching, Mamba align, Flash-V100, CUDA graphs, and
`temperature=1.0/top_p=0.95/top_k=20/seed=20260824`. The control is the mean
of the two bracketing baseline runs.

| Metric | Bracketed control | Lifetime alias | Change |
|---|---:|---:|---:|
| prefill | 47.9233 s | 47.6687 s | -0.531% |
| prefill throughput | 2735.0 tok/s | 2749.6 tok/s | +0.534% |
| TTFT | 47.9498 s | 47.6937 s | -0.534% |
| pure decode | 5.1557 s | 5.1547 s | neutral (+0.020% speed) |

All three runs produce the same 256-token output hash
`2db7ef09...503325a`. Every rank reports the same route counts, including 32
exact dense D256 calls, 544 prefix-paged calls, and 544 E5M2 bridge calls.
This promotes the small dependency-chain win, but it also establishes that
barrier removal alone is not a double-digit long-context solution.

## 2026-08-24 Q8000 Split-KV3 Tail-Wave Experiment

The matched runtime does not present `Q=8192` to long prefix-attention calls.
Although `max_num_batched_tokens` is 8192, Mamba/attention page alignment makes
the repeated full chunks `Q=8000`; a 40K diagnostic observed
`Q=(1,8000,6,256)` and `KV=(1,40000,1,256)`. The first Q8192 experiment was
therefore a useful operator screen but had zero end-to-end route hits. Its
first control measurement was also invalidated by a second TP4 job starting on
GPUs 4-7 between warmup and measurement (`47.90 -> 64.45 s`). The uncontended
candidate/control-B pair used the same unsplit route and differed by only
0.15%, confirming that run did not measure split-KV3.

The corrected candidate admits exactly `Q=(1,8000,6,256)` behind the
default-off
`VLLM_FLASH_V100_PREFILL_DENSE_SPLITKV3_Q8000_EXPERIMENTAL` gate. It leaves
the existing production Q4096 policy unchanged and does not admit Q8192 or
other nearby shapes. Three independent KV partitions turn 750 Q tiles into
2250 CTAs at Q8000, avoiding the short last wave of the 72-SM V100. Bracketed
CUDA-event operator results from the final K-stage binary are:

| Q | KV | Unsplit | Split-KV3 | Throughput change | Causal TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 8000 | 40000 | 40.1736 ms | 38.6519 ms | +3.94% | 44.05 -> 45.78 |
| 8000 | 64000 | 66.5214 ms | 64.6753 ms | +2.85% | 44.33 -> 45.60 |
| 8000 | 96000 | 102.8820 ms | 99.4545 ms | +3.45% | 43.95 -> 45.47 |
| 8000 | 128000 | 138.9150 ms | 134.3961 ms | +3.36% | 43.87 -> 45.35 |

The three FP32 partial-output buffers plus max/sum state require 141.72 MiB
per rank. Outputs are repeat-stable but not bitwise equal to unsplit because
the partition merge changes reduction order. Across the four shapes, maximum
absolute difference is `1.526e-5` to `3.052e-5`; mean absolute difference is
`7.51e-7` to `1.38e-6`.

The corrected full-model A/B/A uses the same TP4 128K official-sampling
contract as the accepted K-stage endpoint. Control A/candidate/control B
prefill times are `47.5940/47.1074/47.6612 s`. The bracketed control is
`47.6276 s` or `2752.0 tok/s`; split-KV3 is `47.1074 s` or `2782.4 tok/s`,
giving `-1.092%` latency and `+1.104%` throughput. TTFT improves 1.078%.
Decode remains neutral: `5.1533 -> 5.1617 s` and
`49.483 -> 49.402 tok/s`. Every rank reports 384 split-KV3 kernel hits, which
matches 12 eligible long chunks times 16 full-attention layers times two
requests. KV capacity remains 2,424,439 tokens per rank and no OOM fallback
occurs.

Quality does not yet justify a default-on promotion. Every official-sampling
lane is internally stable between warmup and measurement, and the candidate
is token-for-token identical to control B for both 256-token outputs. However,
identical unsplit control A selects a different stable sampled stream despite
the same official seed. The candidate therefore shows no unique output
divergence, but this cross-process sampling instability cannot prove
equivalence for a non-bitwise route.

A follow-up deterministic greedy gate used three exact-128K natural-text
requests: two identical Chinese prompts and one Python task. Both control and
candidate reproduce the duplicate prompt exactly within their own process,
and all outputs are coherent; both Python outputs stop normally and return the
same correct `add(a, b)` implementation. The two routes are not token-for-token
equal, though. The Chinese outputs first differ at token 7 and align for
126/128 tokens; the Python outputs first differ at token 20, with all 67
control tokens aligned inside the 70-token candidate stream. Candidate prefill
times are `47.1360/47.2292/47.3380 s` versus control
`47.6704/47.8330/47.9274 s`. This unbracketed quality run is directionally
consistent with the formal A/B/A but is not a replacement performance claim.
All four candidate ranks report exactly 576 split-KV3 hits, matching three
requests times 12 eligible chunks times 16 full-attention layers; controls
report zero.

The semantic smoke passes stability and basic task correctness, but strict
greedy identity fails. Keep Q8000 split-KV3 explicit and default-off until a
fixed-text logprob/perplexity or dataset-level quality gate establishes that
the classified Type-B reduction-order drift does not reduce model quality.

## 2026-08-25 Q8000/KV40K..128K GQA-packed Attention Architecture

This experiment freezes the final long-prefill call at causal FP16
`Q=8000`, `KV=128000`, `Hq=6`, `Hkv=1`, and `D=256` on one
V100-SXM2-32GB. Its metric is 6.094872576 useful causal TFLOPs divided by the
complete Attention elapsed time. It is neither whole-model TOPS nor prompt
tokens/s; the acceptance gate is at most 101.581 ms, or at least 60.0 useful
causal TFLOP/s.

Tile and barrier tuning of the fused BM64/BN32 kernel was already exhausted:
the long-shape NCU record showed one CTA/SM, 12.5% occupancy, 62.14% of cycles
without an eligible warp, and only 9.06% DRAM throughput. The replacement is a
different scheduling and dataflow architecture:

- split a fully visible 32K..120K prefix from the exact causal 8K tail;
- pack all six GQA query heads into GEMM M=48,000;
- run wide SM70 Tensor-Core QK GEMMs that emit FP16 scores plus row max/sum;
- transform scores into normalized probabilities during the PV
  global-to-shared load, avoiding a materialized probability matrix;
- fold one reusable FP16 PV partial into a FP32 online prefix
  numerator/max/sum state;
- export max/sum from the accepted exact 8K tail and merge the two online
  softmax states.

The retained score block is BN8192. Final frozen-extension Torch A/B/A for the
score-cache layout measures `140.00213 -> 100.76979 ms`, or
`43.53414 -> 60.48313` useful causal TFLOP/s: 28.0227% lower latency and a
1.38933x speedup. All 12,288,000 outputs are finite; max/mean absolute error
is `2.2888e-4 / 2.5430e-5` and relative L2 is `6.8629e-3` versus the exact
FP32-accumulator route. BN7168 reaches only 58.42057 TFLOP/s and BN8000 only
59.92825 TFLOP/s, so neither is promoted or rounded into a pass.

The same bounded workspace was then generalized across the twelve observed
8K-chunk shapes. Each row is a strict A/B bracket against the active split-KV3
control:

| KV tokens | split-KV3 ms | architecture ms | latency change | useful TFLOP/s |
|---:|---:|---:|---:|---:|
| 40,000 | 38.41638 | 29.81530 | -22.3891% | 59.34862 |
| 48,000 | 47.25811 | 36.38477 | -23.0084% | 59.44005 |
| 56,000 | 56.53999 | 42.68715 | -24.5010% | 59.87584 |
| 64,000 | 65.17897 | 49.14552 | -24.5991% | 60.00842 |
| 72,000 | 74.00670 | 55.44960 | -25.0749% | 60.27745 |
| 80,000 | 82.74432 | 61.77792 | -25.3388% | 60.46783 |
| 88,000 | 91.98507 | 68.37111 | -25.6715% | 60.38797 |
| 96,000 | 100.88397 | 74.72862 | -25.9262% | 60.51241 |
| 104,000 | 109.80642 | 81.09448 | -26.1478% | 60.61108 |
| 112,000 | 118.42901 | 87.71959 | -25.9307% | 60.51602 |
| 120,000 | 127.35232 | 93.99757 | -26.1909% | 60.65749 |
| 128,000 | 135.93617 | 100.34193 | -26.1845% | 60.74103 |

All 147,456,000 candidate output elements are finite. Across the family, the
worst max absolute difference is `5.0354e-4` and worst relative L2 is
`6.8983e-3` versus the exact FP32-accumulator route. The three shortest points
remain slightly below 60 because the 43-TFLOP/s exact causal tail is a larger
fixed fraction, but they are still 1.288x..1.325x faster than split-KV3.

The architecture trace is compute-dense rather than launch-starved: 49 timed
kernels occupy 95.630493 ms of a 95.809850-ms GPU span, or 99.81%. Prefix QK
accounts for 54.22% and prefix PV 39.74%; the exact tail is 4.89%. The next
compute ceiling is therefore QK/PV instruction efficiency, not additional
host launch fusion.

Real-model integration exposed a separate memory contract. Retaining all
partials or the entire workspace passes the operator gate but leaves too
little memory for the following 80-MiB TP all-reduce. The final layout retains
only the 750-MiB FP16 score matrix per device and packs about 100 MiB of other
state into one transient byte slab. Before first use it requires enough
driver-visible free memory for both allocations plus 128 MiB of downstream
headroom; otherwise it raises a typed OOM before touching the caching allocator
and falls back to the exact route.

The route is default-on for its exact engine contract. The legacy
`VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL=0` setting is the
rollback, and the route rejects CUDA graph capture. It admits only `Q=8000`,
`KV=40000..128000` in 8000-token steps, `Hq=6/Hkv=1/D=256`, FP16, causal
attention, and scale 1/16. The earlier
endpoint-only `.875` TP4
Qwen3.8-27B-FP8 control/candidate/control prefill times are
`46.45658 / 45.47797 / 46.11195 s`. Relative to the `46.28426-s` bracketed
control, the candidate lowers latency by 1.7421% and raises prompt throughput
from 2831.89 to 2882.10 token/s (1.7729%). Every candidate rank logs 16
architecture hits, with no architecture OOM/fallback, and all three runs emit
the same 32 output token IDs and SHA256
`df4fee7f5f0126fe6b391fe77b4fc19667831de5ef55fd69c28c2f52a3d7086e`.

The endpoint-only `.88` run also succeeds at 45.53427-s prefill with the same
hash and 16 hits/rank. Its memory profiler left enough headroom, so the route
passed rather than exercising the fallback branch.

The widened TP4 Qwen3.8-27B-FP8 gate brackets the candidate with `46.11195-s`
and `46.09222-s` controls. Relative to their `46.10208-s` mean, the
`41.51191-s` candidate lowers prefill latency by 9.9565% and raises prompt
throughput from 2843.08 to 3157.46 token/s (11.0575%). Every rank logs exactly
192 family-route hits with no architecture OOM/fallback. Both controls and the
candidate emit the same 32 token IDs and SHA256
`df4fee7f5f0126fe6b391fe77b4fc19667831de5ef55fd69c28c2f52a3d7086e`.

The 2026-08-26 merge audit rebuilt the four-patch vendored FA2 stack from its
locked clean baseline with CUDA 12.8 and `sm_70`. The production QK/PV kernels
again compiled at 254/119 registers with zero spill. On the rebuilt extension,
the 40K endpoint measured `39.29395 -> 30.27579 ms` (`1.29787x`) with all
outputs finite, max/mean absolute difference `5.4932e-4 / 4.4392e-5`, and
relative L2 `6.6098e-3`. The 128K endpoint measured
`136.07628 -> 100.20284 ms` (`1.35801x`) with all outputs finite,
max/mean absolute difference `2.2888e-4 / 2.5484e-5`, and relative L2
`6.9166e-3`. Both points pass the FP16 merge envelope of max absolute error
at most `1e-3` and relative L2 at most `1e-2`; greedy or bitwise identity is
not required. The complete Flash-V100 policy suite passes 112 tests on a real
V100, including a direct architecture-OOM-to-dense-fallback test.

## Artifacts

- D256 GQA architecture artifacts are retained outside Git; their private
  filesystem location is intentionally not committed.
- Final operator A/B/A:
  `results/torch-architecture-scorecache-final-v1-aba.json` under that task
  root.
- Shape-family operator A/B:
  `results/torch-architecture-shapefamily-v1-aba.json` under that task root.
- Shape-family TP4 model gate:
  `results/tp4-128k-architecture-shapefamily-{candidate,control-after}-0875.json`,
  bracketed with the endpoint gate's
  `results/tp4-128k-architecture-scorecache-final-control-b-0875.json`.
- Final TP4 `.875` model gate:
  `results/tp4-128k-architecture-scorecache-final-{control-a,candidate,control-b}-0875.json`.
- Final high-memory `.88` route-success run:
  `results/tp4-128k-architecture-scorecache-final-preflight-fallback-088.json`;
  despite the retained filename, the preflight passed and no fallback ran.
- K-stage task root:
  `/data/minimax-h3/task-cache/qwen38-fp8-128k-flashattention-20260824`.
- Final clean FA2 binary:
  `build/formal-v6-final-cmake/_vllm_fa2_C.abi3.so` under that task root.
- Endpoint A/B/A JSON:
  `results/tp4-128k-{baseline-a-v6,candidate-v6,baseline-b-v6}.json`.
- Q8000 operator sweep:
  `results/splitkv3-q8000-kv{40000,64000,96000,128000}-v6.json`.
- Q8000 endpoint A/B/A:
  `results/tp4-128k-splitkv3-q8000-{control-a,candidate,control-b}.json`.
- Q8000 deterministic quality gate:
  `results/tp4-128k-splitkv3-q8000-quality-{control,candidate}.json` and the
  matching files under `logs/`.
- Shape diagnostic and the zero-hit Q8192 negative run:
  `logs/tp4-splitkv3-shape-diag.log` and
  `results/tp4-128k-splitkv3-{control-a,candidate,control-b}.json`.
- Root: task-local `qwen38-fp8-prefill-decay-20260815` artifact directory.
- Nsight Systems:
  `profiles/qwen38-fp8-tp4-i128k-chunk15680-r2.nsys-rep`.
- Nsight Compute:
  `profiles/ncu-d256-exact-q15680-kv125440-baseline.ncu-rep`.
- Candidate binary:
  `experiments/p-scalar-native-build-r3/_vllm_fa2_C.abi3.so`.
- FP8 KV/chunk 8192 result:
  `results/fp8kv-chunk8192-tp4.json`.

## 2026-08-16 source-worktree FA2 route audit

A later Qwen3.8 MTP-prefill investigation initially measured only 2798.6
tok/s without MTP and 2702.6 tok/s with MTP4 at 64K. Those values are invalid
as optimized-route baselines. The source worktree shadowed the installed vLLM
package but did not contain `_vllm_fa2_C.abi3.so`; the optional operator
loader swallowed the resulting import error and silently disabled the exact
SM70 D256 prefill route.

Restoring the same vendored FA2 binary used by the accepted build made the
required dense, paged, and split-KV3 operators visible. Under the matched TP4,
E5M2 KV, chunk-8192, prefix-cache/Mamba-align, CUDA-graph, official-sampling,
64K-input/256-output contract, the corrected results are:

| Mode | Prefill | Throughput | Relative to retained table |
|---|---:|---:|---:|
| no-MTP | 18.743949 s | 3496.4 tok/s | -3.7% vs. 3630.6 |
| MTP4 | 22.308881 s | 2937.7 tok/s | -0.4% vs. 2950.5 |

The no-MTP worker recorded 288 hits on
`prefill_prefix_fp8_bridge_exact_dense_d256_tailpad`. Both modes emitted 256
coherent tokens with stable warmup/measurement hashes. The remaining matched
MTP prefill overhead is 3.564932 seconds (+19.0% latency, -16.0% throughput),
which is now the optimization target. Raw results and logs are retained under
`/data/minimax-h3/task-cache/qwen38-mtp-fp8kv-prefill-20260816/`.

Synchronized interval profiling attributes 21.281 seconds to the MTP target
forward and 0.864 seconds to the four-step drafter GPU timeline. The target
forward is therefore the dominant prefill cost; removing drafter sampling or
the three dependent draft loops cannot recover the full MTP penalty.

A state-only prefill candidate kept the first draft forward/KV update but
skipped prefill draft sampling and the three dependent loops. It was rejected:
prefill changed from 22.308881 to 22.330927 seconds (+0.10%), while accepted
length fell from 3.779 to 3.253 and decode throughput fell from 60.86 to 58.79
tok/s. The 256-token output remained coherent, but changing the draft RNG
sequence made this route unsuitable even apart from the lack of speedup. The
candidate was removed; its raw result is
`results/mtp4-e5m2-64k-o256-prefill-state-only-r1.json` under the artifact root.
