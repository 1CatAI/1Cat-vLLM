# Flash-Next round12 operator screens, 2026-10-08

The acceptance objective remains C1 <=12 ms/round, correct target output,
retained acceptance and no C4 regression. The measured host-E4M3 baseline is
18.320 ms/round; no new endpoint improvement is established by these screens.
The [current round ledger](flashnext_mtp4_latency_20261008.md) records the
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
establish equivalence. Keep this route disabled. Its isolated M5 gain cannot
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
