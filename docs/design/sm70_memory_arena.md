# SM70 graph scratch and DFlash peak memory

The default path shares decode scratch across row capacities, uses 8192-token
native prefill score blocks (previously 16384), and stores DFlash auxiliary
snapshots in the dtype already consumed by the draft projection. FP16 attention
operands and FP32 accumulation are preserved. There is no new TP count, batch=1,
context-length, or model-quantization gate.

## Storage and rollback

- `VLLM_FLASH_V100_SHARE_DECODE_WORKSPACE=0` restores independent row buffers.
  Sharing defaults on. Device, stream, head count, head dimension, partition
  size, partial dtype, and rounded partition capacity remain separate. Capture
  normally proceeds largest-first. Captured generations remain alive after
  growth, including warmup allocations subsequently referenced by a graph.
  Reuse assumes serial execution on the owning stream; other streams have
  independent storage.
- `VLLM_FLASH_V100_PREFILL_SCORE_BLOCK_TOKENS=16384` restores the old score
  capacity. The default is 8192; set it before worker startup. Reducing the block
  changes online-softmax merge order, so this part is not bitwise equivalent.
  Rebuild the source extension and restart workers; no wheel is required.
- `VLLM_DFLASH_COMPACT_AUX_HIDDEN=0` restores concatenate-then-cast and original
  auxiliary snapshot storage. Compaction defaults on. The original-precision
  hidden/residual addition still executes; only its retained copy is narrowed,
  and only when the loaded draft projection consumes a smaller dtype. Target
  hidden/residual tensors remain unchanged. Concatenation writes directly into
  the projection dtype. Other drafters retain their existing interface.

PR 661 concerns capture-time growth retention; this change primarily shares
row capacities and also covers warmup allocations later used in capture.
PR 660's prefill bridge growth policy is outside this change. Neither was
imported wholesale.

## Reproduction contract

Integration base: `4f8e5e674a1afa4be6e712fd735386a875385a5b`.
Final implementation: `c911a7a5fb75e162bde3dc1b1dd078dade8457e8`.
V100 SXM2 32GB, 185 W/card, CUDA 12.8, Torch 2.10.0+cu128; own source build,
no preload or external kernel overlay. Native manifests, mapped libraries,
commands, allocator traces and per-request records are retained in the local
handoff. The six initial libraries retain identical SHA256 hashes after three
additional source-built targets are installed for the final runs.

Model: `Qwen3.8-27B-QUASAR-NVFP4-d8e6fbfa`, FP16 activations. DFlash2 revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, seven draft tokens, probabilistic
sampling. Target KV is E4M3; draft KV is FP16. Flash-V100,
FULL_AND_PIECEWISE CUDA Graph, max length 262144, chunk 8192, max sequences 32,
GUM 0.9, attention block 2048, Mamba block 8192, align mode, text-only.
All serving tests keep Graph enabled.

Control sets all three rollback values above; candidate leaves their defaults
unset. TP2 timings use GPU0/1. The TP2 candidate quality cohort uses GPU2/3 with
otherwise matching settings. The initial control imported `7daa5b4999f37b61789fa34526fa6e0acd6724e0` with all
three changes disabled; candidate quality imported `a746793feb`. Later snapshot
edits only affect the enabled path. The final change after that quality cohort
is workspace capacity grouping, covered by exact mixed-context Graph tests.
The C4 control has the pending capacity-grouping patch on `a746793feb`, with
sharing disabled; its exact source diff is retained. Final TP4 arms use the
same `c911a7a5fb` source and nine-library manifest.

`vllm bench serve` uses random prompts, temperature 0.7, top-p 0.8, top-k 20,
256 generated tokens, fixed request seed 20260922. Prefix cache is reset after
warmup. Short tests use 2048 input tokens and 8/16/64 prompts for offered
C1/C8/C32. The long C1 tests use one request. Synthetic speed tests ignore EOS;
quality requests use natural EOS with a 65536-token ceiling. Report official
TPOT and full-request throughput separately; neither C/TPOT nor full-request
TPS is a measured pure-decode rate.

## Memory findings

The baseline TP2 peak includes an 800 MiB FP32 concatenation, a 400 MiB cast
projection input, and five 160 MiB auxiliary snapshots at an 8192-token chunk.
Direct concatenation eliminates the 800 MiB intermediate with exact projection
outputs. This alone exposes a different target-forward peak, so narrowing the
retained snapshots also removes 400 MiB. Neither saving should be added blindly
to the final peak reduction.

The first row-sharing implementation multiplied a prior long-context capacity
by a later short-request batch, retaining a 672 MiB buffer on an additional
stream. The final partition-capacity grouping avoids that cross-product.
The failed intermediate measurement is retained; it is not the default result.

TP2 per-card measured allocations after long requests and concurrent serving:

| Measurement, GiB/card | Control | Final candidate |
|---|---:|---:|
| Model-load active allocation | 12.370 | 12.370 |
| Persistent warmup allocation above model load | 1.910 | 1.160 |
| Actual Graph capture device-memory increment | 2.947 | 1.678 |
| Decode scratch after concurrent requests | 1.996 | 0.787 |
| Request temporary peak above end allocation | 2.110 | 1.674 |
| Active allocation excluding KV pool | 17.117 | 15.146 |
| Allocator reserved but inactive | 3.614 | 2.862 |

These rows overlap; do not sum them. The non-KV active allocation decreases by
1.971 GiB/card, and the temporary increment by 0.436 GiB/card. At fixed GUM,
freed space can become KV capacity. The initial matched pair allocates
9.094 -> 10.719 GiB/card to KV; warm-cache startup reports budget
9.72 -> 11.34 GiB. The final warm candidate's actual pool is 11.313 GiB.
Do not credit the extra cold/warm budget difference to this optimization.
Final device usage is 30.380 GiB/card, including the larger KV pool, versus
30.884 GiB in the initial control.

TP4, using the same final source and nine-library set in both arms, gives the
following per-card measurements after concurrent requests:

| Measurement, GiB/card | Control | Candidate |
|---|---:|---:|
| Model-load active allocation | 6.675 | 6.675 |
| Persistent warmup allocation above model load | 1.911 | 1.161 |
| Actual Graph capture device-memory increment | 1.801 | 1.162 |
| Decode scratch after concurrent requests | 1.076 | 0.405 |
| Request temporary peak above end allocation | 2.183 | 1.343 |
| Active allocation excluding KV pool | 10.225 | 8.794 |
| Allocator reserved but inactive | 3.560 | 2.554 |

Non-KV active allocation decreases by 1.431 GiB/card. At fixed GUM, the KV
pool increases from 15.016 to 17.406 GiB/card; total device usage after
concurrent requests is 30.045 -> 29.998 GiB/card. Profile and Graph capture
measurements are different quantities and should not be added to the peak.

TP1 on a 32GB V100 still fails admission with DFlash2, chunk8192, C32 and GUM0.9:
model-load active23.337 GiB, profile live24.496 GiB, temporary peak2.254 GiB,
KV budget-1.939 GiB. Its persistent workspace is reduced by0.750 GiB; this
is not a successful single-card serving claim. No physical16GB GPU was tested.

## TP2 serving results

| Workload | Full output TPS, off/on | Median TTFT seconds, off/on | Median TPOT ms, off/on |
|---|---:|---:|---:|
| 32K input, C1 | 13.016 / 13.023 | 18.051 / 18.090 | 6.333 / 6.146 |
| 2K input, C1 | 104.801 / 104.944 | 1.055 / 1.057 | 4.989 / 4.967 |
| 2K input, C8 | 105.031 / 114.084 | 8.369 / 5.049 | 28.400 / 29.594 |
| 2K input, C32 | 102.867 / 108.482 | 67.603 / 61.211 | 34.407 / 37.048 |

All requests complete, with zero measured prefix hits and zero preemptions.
Peak resident requests increase from5 to6 for offered C8/C32. Their aggregate
throughput improves8.62%/5.46%, but per-request TPOT grows4.20%/7.68%: these
latencies are not all within3%. Median complete-request latency improves6.49%/
6.47%. For a matched resident count, three C4 repeats give average full TPS
120.048 -> 120.575 (+0.44%) and mean TPOT22.620 -> 22.517 ms (-0.45%), with
identical mean accepted draft length5.367. The fixed-residency speed gate passes.

Cold256000-token retrieval returns the same correct16-token answer and natural
EOS: TTFT245.024 -> 247.637 seconds (+1.07%). This TP2 result is not the older
TP4 2.4K-2.5K prefill baseline.

## TP4 serving results

| Workload | Full output TPS, off/on | Median TTFT seconds, off/on | Median TPOT ms, off/on |
|---|---:|---:|---:|
| 32K input, C1 | 24.829 / 24.941 | 9.274 / 9.231 | 4.062 / 4.045 |
| 256K input, C1 | 2.013 / 2.001 | 124.598 / 125.320 | 10.125 / 10.270 |
| 2K input, C1 | 158.864 / 158.283 | 0.560 / 0.565 | 3.949 / 3.942 |
| 2K input, C8 | 189.948 / 193.439 | 1.289 / 1.294 | 22.577 / 22.628 |
| 2K input, C32 | 213.822 / 221.764 | 17.134 / 13.588 | 53.461 / 64.454 |

The 256K benchmark has zero prefix hits and no preemptions in either arm.
Its TTFT and TPOT increase 0.58% and 1.44%, within the 3% matched-speed gate.
The short C8 workload admits eight resident requests in both arms: full output
TPS improves 1.84%, and median TPOT increases 0.22%. At offered C32, resident
requests increase from 18 to 21 and full output TPS improves 3.71%, while
median per-request TPOT increases 20.56%. Larger simultaneous admission changes
the per-request latency; the C32 result is not a 3%-latency claim. All benchmark
requests complete, with zero measured prefix hits and preemptions. Cold 256K
retrieval returns the same correct 16-token answer and natural EOS in both
arms, TTFT 123.197 -> 124.357 seconds (+0.94%).

## Numerical and quality checks

- 31 decode arena/planner checks pass, including FP16/E4M3 KV, local heads
  6/12/24, different streams, warmup-buffer capture, growth retention, and
  alternating short/long-context Graph replay against independent buffers.
- 25 compact projection/snapshot checks pass, including compiled/Graph execution,
  non-contiguous inputs, FP16/FP32 input and projection dtypes, unchanged target
  tensors, and exact projection equality. Fifteen DFlash loading/contract checks
  and the context-pipeline GPU check pass.
- Q8192/Q8000 at256K show operator latency increases1.92%/2.22% for8K versus16K
  score blocks. FP32 accumulation is retained; biased/periodic stress cases pass
  the FP32 oracle and exact replay checks.
- TP2 MBPP sanitized test sample: off24/32, on24/32, zero truncation in both;
  longest complete answers41432/38113 tokens. Index18 changes fail->pass;
  index144 changes pass->fail. These are retained logical-code errors. Equal
  totals from32 samples do not establish statistical equivalence. Dataset SHA256:
  `e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850`.
- Four serial MBPP tasks with a 32768-token unrelated prefix pass in both TP4
  arms (4/4), with natural EOS and no truncation. This tests answer quality
  after long prefill; it does not substitute for a broad long-context benchmark.

The full-FP32 75T and separate 35B migration performance targets remain open.
