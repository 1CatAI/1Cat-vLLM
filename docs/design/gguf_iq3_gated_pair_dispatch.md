# IQ3_S gated-pair dispatch on SM70

Mixed GGUF gate/up projections launch separate canonical GEMMs and concatenate
their outputs before SiLU/multiply. Latest main already coalesces adjacent
same-type projections into one canonical GEMM; an IQ3_S pair therefore retains
that N8704 fallback, rather than splitting it back into two N4352 GEMMs.
The calibrated IQ3_S
pair reads the source-sized lossless records and performs FP32 MMA,
FP32 split-K reduction, and the existing FP16 projection/activation rounding
in one launch.

Only `(types=(IQ3_S, IQ3_S), M=8, N=4352, K=5120)` is admitted.
Other row counts retain the canonical projection operators. The row-count
choice stays inside an opaque operator, including prefill workspace lookup,
so range compilation cannot freeze the prefill branch. Layout-transformed
layers, mixed types, unknown dimensions, missing operators and disabled
kernel policy retain their existing routes. Preparation reports explain
these conditions. No qkvz route is enabled by this change.

The model uses its existing fused-SiLU linear interface. Loading prepares
source-byte records before releasing checkpoint storage and retains
canonical storage for the other M ranges. Records do not expand indices,
signs or the original two scale levels. This adds 19,148,800 bytes per
admitted layer per TP4 rank; eight layers total 153,190,400 bytes. The
codebook and raw decoder retain their llama.cpp/TurboMind provenance and
licenses.

## Numerical and graph checks

The CPU permutation independently reconstructs every original source byte.
Random blocks, disabled/missing capability conditions, mixed types,
uncalibrated dimensions, FP16 policy and actual-M fallback are tested.
A single dynamic compilation covers M=512/8/16/512 and keeps exactly one
opaque gated-pair node. The existing runtime projection tests are rerun.
Thirteen targeted CPU tests passed with Torch 2.10 / CUDA 12.8, including the
coalesced canonical fallback, M32, runtime projection and IQ4 layout checks.

The public nibble-book decoder uses exact signed lookup and half unpacking;
the existing raw lattice device API remains intact. The original d and small scale
remain separate. The reference forms GGUF weights using the official reader,
and forms projection products with FP32 GEMM. Projection results and SiLU
round to FP16 as in the existing path; the dot products and reduction remain
FP32. A new packaged-operator benchmark checks native M8 and canonical
M1/M16/M32/M512 before comparing cold-L2 graph medians in ABBA order.

## Research measurements

Frozen micro shape: TP4 rank0 layer6, M8/N4352/K5120, actual model weights,
FP16 activations, FP32 accumulation, 16MiB L2 eviction before each event,
84 timed samples, V100 SXM2 32GB, SM/memory 1290/877MHz, 300W,
Torch 2.10 / CUDA 12.8. Source-byte bandwidth is a workload ratio, not the
NCU DRAM counter. The following operators were research builds; packaged
operator measurements are required before promotion.

| Variant | Graph median | Source bandwidth | Resources |
| --- | ---: | ---: | --- |
| Original signed HFMA pair | 62.464us, earlier ABBA | 306.56GB/s | 62.625 instructions/K16, 50 registers, 32KiB shared, no spills |
| One 16KiB signed book | 62.464us, both ABBA controls | 306.56GB/s | 78.125 instructions/K16, 51 registers, no spills |
| Two 16KiB books, thread parity selects copy | 62.464us, both ABBA candidates | 306.56GB/s | 82.125 instructions/K16, 52 registers, 32KiB shared, no spills |
| Same-round signed HFMA control | 63.488us | 301.61GB/s | Unchanged decoder |
| Same-round NVFP4 pair, split8/16 | 51.200us each | 489.60GB/s | 25,067,520 source bytes |

Duplicating the book was numerically bitwise equal to the retained control
but gave no timing improvement. It is not selected. Read-only-cache,
texture and bank-permutation variants are already rejected and are not
repeated. Register prefetch/interleaving experiments are also closed.

A subsequent matched shared-activation prototype loads A once per CTA into
padded shared rows, reused by gate and up. Its ABBA medians are
63.488 / 50.176 / 50.176 / 63.488us (non-staged / staged / staged / non-staged),
with the original single-book control at 62.464us and NVFP4 at
51.200/52.224us for split8/16. SM/memory remain 1290/877MHz. Output is
bitwise equal to the retained IQ3 control; official-reference relative L2
is 0.000495 and maximum absolute error 0.00390625, unchanged. The staged
kernel uses 48 registers, 33,792 shared bytes, and no spills. Static loop
instructions increase to 82.375/K16 while global-load sites decrease from
22 to 7. This validates activation sharing rather than instruction-count
reduction. The packaged candidate uses this measured N32 shared-A version.

Wider N64 tasks with partition2 last-CTA reduction were also tested in one
matched sweep. Grid80 gave 58.368us and grid160 gave 53.248us, repeated
with unchanged results against N32 controls of 50.176us. The candidates
used 62 registers, no spills and 41,472 shared bytes. The changed FP32
reduction tree produced relative L2 7.153e-6 versus the retained output;
all counters reset correctly across the 84-sample graph replay. These
variants are slower and are not admitted. No separate reduction launch was
introduced.

The original 76–81us canonical comparison used separate projections. The
packaged comparison measures latest main’s coalesced N8704 canonical GEMM
plus SiLU; it must pass independently before the native pair is admitted.

Only eight layers have two IQ3_S FFN projections. Their source byte share is
12.765% of all gate/up pairs. The primary-machine packaged comparison below
saves 20.48–21.50us per eligible layer, approximately 0.164–0.172ms across
those eight layers. This is a projection estimate, not a measured full-round
saving. The 40 mixed-type pairs account for
62.286% of pair bytes; see the complete
[source inventory](gguf_qwen38_iq3s_source_inventory.md).

## Packaged operator and latest-main baseline

Normal source-built wheel `1.5.2.dev13+g1997e1bb24.precompiled`, Torch 2.10,
CUDA 12.8, passes the installed-operator check on a V100 SXM2 32GB.
Both installed native modules match their wheel members. No private library
or source overlay is used. M1/M16/M32/M512 fallback outputs are bitwise equal
to the existing coalesced canonical path. Against official FP32 GGUF weights
and FP32 GEMM, native M8 relative L2 is 0.000497/0.000516 on two independent
inputs; maximum absolute errors are 0.00390625/0.0078125. Canonical fallback
relative L2 is 0.000615–0.000628.

This secondary machine ran at SM/memory 1530/877MHz and 300W, whereas the
research table used 1290/877MHz. Do not compare the two timing tables across
clocks. Within its single ABBA session, the packaged canonical N8704 GEMM
plus SiLU took 62.464us in both arms, versus 45.056us in both native arms.
Native source-payload bandwidth is 425.0GB/s. This is installed-package
validation on the secondary machine.

The same normal wheel also passes the primary-machine installed-operator
check at steady SM/memory 1290/877MHz, 300W. Every ABBA arm records the
same clocks. Canonical N8704 GEMM plus SiLU takes 70.656/69.632us; the native
pair takes 49.152us in both arms. Native source/packed payload is 19,148,800
bytes and payload bandwidth is 389.6GB/s. Canonical packed codes plus stats
occupy 22,282,240 bytes, corresponding to 315.4/320.0GB/s. These payload
ratios are distinct from NCU physical traffic. Native M8 relative L2 and
maximum errors match the secondary-machine values above; other tested
row counts again equal the canonical fallback bitwise. The retained
same-clock NVFP4 research pair measurement is 51.200/52.224us with
25,067,520 source bytes; it is a separate research run, rather than a third
arm of this installed-package session.

Latest-main baseline source `0e359c87d315931b89b7d5a774927f212d57f3b1`
uses its normal `dev11+g0e359c87d3` wheel on four fully connected V100s,
TP4, 300W, steady SM/memory 1290/877MHz, FP16 KV/operands and FP32 SSM.
The target and Q8_0 DFlash2 checkpoints match the recorded hashes. Maximum
length is 262144, batched tokens 1024, maximum sequences four, prefix cache
off, Flash-V100, CUDA graph, seven probabilistic draft proposals, temperature
0.7/top-p0.9/top-k20/seed123, thinking off. Eight fixed 600-token speed
requests per input length explicitly ignore EOS; the separate natural C4
smoke has a 96-token limit. Prompt token IDs and sampling equal the previous
baseline. First twenty rounds are excluded per prompt.

| Input | Full-round mean | Emitted tokens/round | Emitted 95% prompt-bootstrap interval | TTFT |
| --- | ---: | ---: | --- | ---: |
| 1024 | 34.173ms | 2.962 | 2.846–3.091 | 330.996ms |
| 8192 | 35.292ms | 2.974 | 2.790–3.167 | 2669.433ms |

Intervals use 10,000 whole-prompt resamples. Counter-based bonus-plus-accepted
means are 2.959 and 2.958; they are distinct from observed emitted batches.
TTFT includes prefill and scheduling; pure prefill is not measured separately.
C4 completes four nonempty reasonable answers in 4.688s; all reach the chosen
96-token limit. It is a smoke, not a matched concurrency throughput claim.
The previous round means were 34.940/35.398ms. No 25ms prediction is
presented as a measured result.

## Latest-main graph-boundary ledger

The single new node trace uses the historical trace contract: maximum length
32768, input 1024, output 64, TP4 and the same normal baseline wheel. It is
separate from the unprofiled 262144-context contract above. Target launches
are linked to GPU nodes by process and correlation ID. Each round starts at
one target graph's first GPU node and ends at the next target graph's first
node; host NVTX submission intervals are not used as GPU boundaries.
Four ranks contain fourteen retained steady rounds each.

Rank0's mean full round is 36.259ms, its target graph envelope is 32.379ms,
kernel busy union is 33.977ms and uncovered residual is 2.283ms. Taking the
maximum rank at each replay ordinal gives 36.294ms. These profiled times do
not replace unprofiled end-to-end measurements. Kernel service categories
below overlap and must not be added into another wall-time estimate.

| Rank0 category | Calls/round | Service/round |
| --- | ---: | ---: |
| TurboMind projections | 379 | 14.644ms |
| Single-CTA CUTLASS floating shards | 96 | 11.525ms |
| Communication/reduction | 152 | 1.852ms |
| Attention | 37 | 1.552ms |
| Copy/concatenation | 212 | 0.912ms |
| cuBLAS FP16 | 1 | 0.013ms |

There are no whole-weight lattice/affine dequantization or raw Q4_K MMQ
calls in the retained rounds. The one remaining s884gemm is a small 13us
call, rather than the previous prefill projection path. Target attention
selects sixteen grouped FP16 verifier partial kernels, totaling 0.881ms.
LM head's two canonical GEMMs total 0.546ms. The remaining 96 single-CTA
calls confirm that BF16 a/b shards in mixed qkvz projections still require
independent floating-shard admission: converting an entirely floating
projection does not cover a floating shard inside a quantized projection.
Their mean service is 120.1us per call. Reuse of the existing FP32 row GEMV
must pass its own actual-weight microbenchmark before admission; this ledger
does not claim the proposed saving has been achieved.

## Activation traffic

A separate SourceCounters capture compared the signed HFMA and NVFP4
kernels, using the same M8 real-weight shape. SASS load addresses identify
activation and weight/codebook operations. Counters and profiler service
latencies below must not be substituted for unprofiled graph medians.

| Counter | IQ3_S | NVFP4 |
| --- | ---: | ---: |
| Actual total L1 global read bytes | 68,497,152 | 69,629,024 |
| Actual total L2 read sectors x32 | 31,000,096 | 31,630,048 |
| Actual DRAM read bytes | 19,466,304 | 25,149,664 |
| Activation source-correlated L1 tag requests | 1,392,640 | 1,392,640 |
| Weight/codebook source-correlated L1 tag requests | 192,576 | 261,120 |
| Activation theoretical L2 sector bytes | 44,564,480 | 44,564,480 |
| Weight/codebook theoretical L2 sector bytes | 23,814,144 | 25,067,520 |
| Profiled kernel service | 59.648us | 47.520us |

Activation tag requests are 7.23 times the IQ3_S weight/codebook requests.
The 44.56MB source-correlated value is theoretical sector demand before
cache hits; it does not mean measured activation-only L2 bytes. The
hardware counter provides total L2 bytes, without separating A and B.
The repeated activation demand justifies CTA-local staging and gate/up
reuse. The matched shared-A result is recorded above. Wider-N changes require
their own correctness, resource and matched graph comparison before replacing
that candidate.

## Reproduction

`benchmark_gguf_iq3_gated.py --model MODEL.gguf --output RESULT.json` uses
only the installed operator, records clocks and numerical errors, and
checks runtime canonical fallbacks. Acquire the shared GPU lock and the
selected device lock before running it. The complete source inventory is
reproducible with `benchmark_gguf_quantization_inventory.py`.

Latest-main end-to-end, graph-boundary ledger and both machines' packaged
operator checks are recorded above. The native-route natural termination
check is pending. No end-to-end gain is claimed from isolated operator timings.
