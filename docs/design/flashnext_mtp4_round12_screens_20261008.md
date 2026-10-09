# Flash-Next round12 operator screens, 2026-10-08

The acceptance objective remains C1 <=12 ms/round, correct target output,
retained acceptance and no C4 regression. The measured host-E4M3 baseline is
18.320 ms/round; no new endpoint improvement is established by these screens.
The [contributor round ledger](https://github.com/1CatAI/1Cat-vLLM/blob/81259b8109c99d8823ba5e731d60da2412814d0d/docs/design/flashnext_mtp4_latency_20261008.md) records the
workload, graph composition and overlapping service totals.

## Measurement contract

Use four V100-SXM2-32GB GPUs with the two-hop TP4 topology, CUDA 12.8,
Torch 2.10.0+cu128 and installed wheel `1.5.2.dev1190+gc1ebacb67`. The core
SHA256 is `784d1447f4f5f5593fa77db525e6b40e841366d654202aab5a56beec67398a0c`.
Application clocks are 1290/877 MHz; observed SM burst clocks are recorded
per case. Clocks are not changed. Do not compare different runs as identical
clock conditions.

HC screens use eight real weight pairs, changed activations on every rank,
changed-input graph replays and ABBA graph timing. Report the maximum rank
median for the complete tested chain. Expert screens use real TP4
N160/K2560 gate/up weights, top-10 routing, M1/M5/M20, changed inputs/routes,
and same-process graph events with L2 eviction. Candidate private extensions
are research-only; none is a production dependency or selected model route.

## Rejected and bounded candidates

| Candidate | Control / candidate, us | Decision |
| --- | ---: | --- |
| Compute each HC RMS coefficient once | 29.972 / 30.724 | Publication/wait overhead exceeds reuse |
| Hierarchical HC CTA-arrival counters | 29.260 / 29.872 | No improvement of the complete chain |
| Spread LoRA reception across 80 HC CTAs | 29.284 / 29.236 | Insufficient gain; retain current mapping |
| Overlap Q6_K output projection and HC | 38.000 / 43.644 | Resource competition and stream joins regress |
| Coarse host resolve, M5/hot8192 | 30.528 / 30.112 | Too small; hot64 misses regress 484.768 / 524.064 |
| Pointer-selected host QSA reader, M5/hot8192 | 90.112 / 87.552 | Cold path regresses 731.584 / 913.216 |
| Replicated IQ codebook, M5 IQ3_XXS | 45.056 / 48.128 | Rejected |
| Replicated IQ codebook, M5 IQ3_S | 47.104 / 56.320 | Rejected; vector initialization does not recover it |
| Exact expanded-u4 IQ3_S, M5 | 49.152 / 45.056 | +23.6% weight storage; M20 regresses |
| Original IQ3_S halfword planes, M5 | 49.152 / 55.296 | Rejected despite lossless byte layout |
| IQ3_S group32 field packets, M5 | 49.152 / 41.984 | Positive isolated candidate, not an endpoint claim |
| Compact IQ3_S field planes, M5 | 49.152 / 45.056 | Smaller benefit with +1.8% storage |

All tested HC outputs and changed-input graph checks are byte-identical.
The packet layout retains original codes, signs and both scale levels, with
source-byte inversion checks. Its M5 decoded Q8 intermediate error is zero.
M20 field packets improve 131.072 to 115.712 us; decoded error for that schedule
is zero. M1 is slower at the M5 schedule and needs independent admission.

The field packet expands each IQ3_S superblock from 110 to 128 bytes. Its
458.5 GB/s expanded-payload rate is **394.1 GB/s of original payload**;
it does not prove the original-weight 450 GB/s target. This file contains
10 IQ3_S gate/up layers, 17 IQ3_XXS, 20 IQ2_S and one IQ4_XS. Thus the tested
IQ3_S saving estimates only 0.072 ms of isolated service per round, not a
48-layer or model-level speedup. Expanding these ten layers alone adds
1.10 GiB globally (0.275 GiB/rank). The packet decoder is not selected.

The paired-output-tile and next-packet-prefetch variants preserve decoded
M5 outputs but do not increase the best M5 packet saving consistently.
Different burst clocks also preclude attributing cross-run improvements to
these variants. No model restart is justified by summing these micro deltas.

## Existing GDN model result

The earlier same-wheel native-GDN pair is already complete: C1
17.406 / 17.613 ms and C4 45.025 / 46.229 ms. C1 token IDs agree, while C4
and natural prompt IDs differ. The paired acceptance delta is -0.150
percentage points, with 95% interval [-1.829, +1.464]. This interval does not
establish equivalence. This candidate was not promoted in that comparison. Its isolated M5 gain cannot
be used as a projected model win or trigger an identical model rerun.

## QSA output ownership

The trace contains 48 Half-Fill kernels: 36 follow GDN projection splitting,
and 12 initialize QSA outputs. Direct host QSA writes every active output,
including invalid requests and empty selections; graph padding still needs
explicit zero initialization. Initialize only padding on that path, and retain
the previous behavior for other backends and an entirely empty batch.

Twelve CPU tests pass. Eight CUDA cases cover M5/M20, hot64/hot8192, gate
on/off, invalid requests and NaN-prefilled output: the direct writer agrees
byte-for-byte with the zero-initialized reference, and invalid rows are zero.
The optimization is bounded by the 12 QSA fills, roughly 0.03 ms of profiled
service. No new C1, C4 or acceptance improvement is claimed for it.

## Remaining endpoint gap

The typical target envelope is 13.94 ms and the four-step draft envelope
3.40 ms. Current host/cache, fill and arrival-skew candidates cannot redeem
the approximately 6-ms target reduction needed with unchanged draft cost.
HC readiness changes tested here are rejected; positive expert layout results
remain small and carry a memory cost. Further work needs a larger structural
change with an isolated correctness and critical-path screen before another
same-wheel model A/B. Preserve the negative results and unchanged mean/tails.

## Additional readiness and QSA screens

Four-stream HC partial transposition is distinct from the earlier compact
partial layout. Eight real weight pairs, four ranks and changed-input graph
checks are byte-identical; max-rank median is 30.071 / 30.888 us. Amortizing
watchdog clock reads every 64 polls retains bounded waiting and exact outputs,
but regresses 28.928 / 29.696 us. Neither is selected.

D64 QSA merge tiling retains FP32 arithmetic. M5/hot8192 complete host resolve
plus attention measures 89.600 / 88.704 us, M20 392.640 / 393.408 us, and the
undersized M5 hot cache regresses 1387.520 / 1707.456 us. Query output max
absolute difference is 1.53e-5; the FP32 reduction layout is not byte-invariant.
The timing is a cache-plus-attention screen, not an isolated merge result.
Retain the old merge.

The exact-pair HC screen avoids an explicit MoE concatenation and measures
33.300 / 29.896 us. Its upper bound across 48 boundaries is 0.163 ms; any
integration must pass both tensors explicitly through compiler-visible
ownership, rather than the previously rejected hidden Tensor registry.

## Native IQ3_S hardware counters

Nsight Compute 2022.4.1 profiles the installed native M5/N160/K2560 gate/up
operator on a real TP4 expert shard: 50 routes, 47 unique experts, 250 CTAs,
512 threads each. This counter replay is not a new performance baseline.
Measured duration is 48.736 us at profiler clocks 1.14 GHz SM / 782 MHz HBM.

Registers/thread are 56, theoretical occupancy 50%, achieved occupancy 43.49%.
There are 51.51% cycles with no eligible warp and 1.18 eligible warps per
scheduler. L1/TEX scoreboard waits account for 31.0% of issued-instruction
spacing. Global accesses have 2,216,219 excessive sectors out of 2,899,219
(76%); shared accesses have 742,358 excessive wavefronts out of 1,576,608 (47%).
DRAM throughput is 371.48 GB/s, 46.37% of profiler-clock peak; ALU utilization
is 44% of active cycles. The next decoder experiment should address access
amplification and latency hiding, rather than assume integer issue saturation.

Lossless compact field planes already coalesce source fields and use 40
registers/thread at eight warps. Screen their per-specialization shared-memory
carveout next: separate kernel symbols are required so CUDA graph A/B does not
mutate a common kernel's launch policy after capture. This experiment remains
research-only until isolated and model-level gates qualify it.

The IQ3_S carveout screen subsequently completed. With distinct graph-safe
kernel symbols for default/33/66/100-percent preferences, M5 control is
48.128 us and every eight-warp compact candidate measures 44.032 us. M1
regresses 19.456 / 27.648 us; M20 default is 123.904 / 115.712 us, while the
33-percent preference regresses to 123.904 us. Decoded Q8 outputs agree exactly
at M5; M20 has three differing sum bytes but identical decoded values.
Carveout adds no M5 benefit beyond the existing compact layout, so it is not
selected or propagated to additional formats.

## Validated changes in the host-KV prototype

The following changes were measured in the contributor host-KV prototype
associated with #1028 and #1039. Recording them here does not enable them
in main.

QSA preparation reuses the existing native operator with 32 local staging
slots, then invokes the unchanged authoritative history writer. Fresh
installed production-helper tests cover M1/M5/M20, FP16/FP32 cosine caches
and two changed-input/position graph replays. With the representative FP16
cosine cache, reference / native-helper timings are 16.384 / 11.264 us (M1),
18.432 / 13.312 us (M5) and 18.944 / 13.312 us (M20). Key/value, gate, history
codes and FP32 history scales agree exactly. M20 query maximum error is
0.00012207, relative L2 5.42e-7; M1/M5 queries are byte-identical.

Host H6/D256 SM70 partials use two warps through 32 rows. Separate-state
four-request M20 tests cover contexts 128/512/1024/8192 and a 64-token hot
cache. Outputs are bitwise equal, including changed selections and invalid
rows. Full resolve-plus-attention timing pairs for contexts 128/512/1024 are
67.968 / 67.008, 100.096 / 89.024 and 176.896 / 110.592 us. The 8192-token
synthetic result must not be substituted for the short-prefix C4 endpoint.

Twenty-five CPU contract tests pass. A normal source-complete Python wheel
retains all 16 native libraries and the Rust executable byte-for-byte. These
changes have an isolated operator basis; preparation remains opt-in. No new
C1/C4 endpoint, acceptance equivalence or 12-ms achievement is claimed.
