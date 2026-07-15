# SM70 AWQ Small-N HMMA Operator

## Status

This document records the quality-preserving small-N TurboMind HMMA path for
Qwen3.6-27B-AWQ target verification on V100/SM70. It is separate from the
exact-M5 scalar GEMV experiment because this path retains the production HMMA
math, FP32 accumulator, K order, and split-K contract.

Accepted microbenchmark result on 2026-07-11:

- all 236 real rank-local AWQ calls: `11.3419 ms -> 9.0439 ms`;
- delta from the fixed TurboMind baseline: `-2.2980 ms`, `-20.26%`, or
  `1.254x` throughput;
- delta from the first accepted N64 plus A-swizzle route:
  `9.7679 ms -> 9.0439 ms`, `-0.7240 ms`, or `-7.41%`;
- all 236 output tensors are bitwise equal to the archived fixed TurboMind
  baseline;
- the default-on TP2 fast selector uses conflict-free N64 plus a two-register
  global-memory lookahead only for gate/up, qkv/z, and attention output
  projections;
- MLP down remains on its existing `8x256x64`, split-7 tactic.

This route is TurboMind only. It does not use Marlin and does not use eager
execution.

## Acceptance Contract

The target verifier has `M=5`, AWQ group size 128, TP2, and the following
rank-local call mix:

| family | descriptor shape | calls/rank | accepted tactic |
|---|---:|---:|---:|
| merged MLP gate/up | `5x17408x5120` | 63 | `8x64x64`, split 1, lookahead 2 |
| MLP down | `5x5120x8704` | 63 | existing `8x256x64`, split 7 |
| linear-attention qkv/z | `5x8192x5120` | 47 | `8x64x64`, split 2, lookahead 2 |
| all attention output projections | `5x5120x3072` | 63 | `8x64x64`, split 3, lookahead 2 |

The route must preserve:

- `CTA_K=64` and `TG_K=1`;
- the fixed split count and scheduler swizzle for every descriptor;
- `mma.sync.m8n8k4.f32.f16.f16.f32` accumulation;
- FP32 epilogue partials and final FP16 conversion;
- prepared AWQ weight and scale/zero semantics.

Changing `TG_K`, split count, or K tile order is not part of this optimization.
Those changes previously caused MTP acceptance loss.

## Implementation

The implementation is limited to five ownership points:

- `arch/operand_sm70_s884.h`: opt-in swizzled A operand;
- `mainloop_sm70.h`: opt-in two-register global-memory lookahead;
- `arch/config_sm70_s884.h`: opt-in AWQ configuration and lookahead parameter;
- `kernel/sm70_884_4.cu`: new `CTA_N=32/64` registry entries;
- `gemm.cu`: exact `M=5` shape selector rules.

The registry entries are:

```cpp
CS::Type<8, 32, 64, 1, 1, 1, D, S, 2, true, 1, 128>
CS::Type<8, 64, 64, 1, 2, 1, D, S, 2, true, 1, 128, -1, -1, 2>
```

They launch 32 and 64 threads per CTA. For gate/up, the output grid changes
from 68 `CTA_N=256` CTAs to 544 `CTA_N=32` or 272 `CTA_N=64` CTAs. N64 is the
accepted tactic: 272 CTAs are enough to fill 72 SMs while retaining two warps
per CTA and avoiding the extra scheduling and A-staging overhead of N32.
Only the N64 entry sets `GmemLookahead=2`; every existing registry entry keeps
the original lookahead-one mainloop.

The environment override remains available for isolated A/B tests:

```text
VLLM_SM70_AWQ_TP2_FAST_TARGETS=
  descriptor|cta_m x cta_n x cta_k:splits:swizzle:require_mgroup
```

`VLLM_SM70_AWQ_TP2_FAST_SELECTOR=0` disables the rules and restores normal
dispatch. The selector is default-on when the variable is unset.

## Conflict-Free A Layout

The stock A tile is `8x64` FP16 in row-major shared memory. Its row stride is
64 half values, or 128 bytes, so all eight row starts map to the same four-bank
group. Every K8 step issues an `LDS.U.128`; NCU attributed essentially all
baseline shared-load conflicts to these A loads.

The opt-in layout is:

```cpp
SmemLayoutV2<8, 64, 8, 64, Swizzle<3, 3, 3>>
```

For half-element index `p = 64*m + k`, with `k = 8*q + r`, it maps to:

```text
p' = p xor ((p & 0x1c0) >> 3)
   = 64*m + 8*(q xor m) + r
```

The low three bits are unchanged, so each eight-half load stays contiguous and
16-byte aligned. The row id is XORed into the vector-group bits, distributing
the eight A rows across disjoint bank groups. Lanes that need the same HMMA A
fragment are then served by native shared-memory broadcast without extra SHFL
instructions.

This is why a separate warp-shuffle activation broadcast was not retained:
after the physical bank mapping is fixed, the hardware broadcast already
coalesces identical addresses. Adding four 32-bit SHFL operations after one
LDS would add issue and dependency cost without removing HBM traffic.

## Two-Register Global Lookahead

The first N64 route still started the global load for K tile `i+1` while the
HMMA loop for tile `i` was already running. At the end of the loop, the
register-to-shared `STS.128` could therefore reach the next weight fragment
before its streaming `LDG` had completed. In the accepted NCU report this
appeared as 559 not-issued long-scoreboard samples.

`GmemLookahead=2` keeps the shared pipeline at two stages but adds a second set
of global-load register fragments. Its steady-state schedule is:

```text
bootstrap: fetch/store tile 0, fetch tile 1 into slot A
compute tile i:     store ready tile i+1, fetch tile i+2 into slot B
compute tile i + 1: store ready tile i+2, fetch tile i+3 into slot A
```

The next tile is now fetched one complete K64 compute stage earlier. The two
register slots alternate without changing the order of shared loads, AWQ
dequantization, HMMA instructions, FP32 accumulation, or epilogue reduction.
`GroupSizeV=128` spans exactly two K64 tiles: metadata is fetched on the even
tile and the odd tile reuses the shared metadata, so the two-slot cadence does
not change scale/zero semantics.

This is deliberately not a third shared-memory stage. Shared memory remains
6.69 KiB/CTA, while code generation drops from 122 to 120 registers/thread.
The paired outer loop also removes loop and iterator instructions; NCU reports
11.782 M executed instructions instead of 13.049 M with identical tensor math
and effectively unchanged DRAM bytes.

## Microbenchmark Results

Environment:

```text
GPU: Tesla PG503-216 / V100 / SM70, physical GPU 2
Torch: 2.10.0+cu128
model: /home/ymzx/models/Qwen3.6-27B-AWQ
TP: 2, rank 0, M=5, group size 128
baseline: _C.baseline_fixed.abi3.so
timing: 50 warmups, 200 samples, 20 queued calls/sample
```

Representative shape results for the final default selector:

| family | fixed baseline | N64 + A swizzle | final lookahead 2 | final vs N64 | exact |
|---|---:|---:|---:|---:|---:|
| gate/up | 88.620 us | 71.088 us | 63.960 us | -10.03% | yes |
| MLP down | 39.550 us | 39.547 us | 39.629 us | +0.21% | yes |
| qkv/z | 46.362 us | 38.450 us | 34.430 us | -10.46% | yes |
| linear-attention out | 24.644 us | 20.490 us | 19.108 us | -6.75% | yes |
| full-attention out | 24.077 us | 20.676 us | 19.094 us | -7.65% | yes |

All-real aggregate:

| route | mean | p50 | p90 | output comparison |
|---|---:|---:|---:|---:|
| fixed TurboMind | 11.3419 ms | 11.3388 | 11.3603 | reference |
| N64 + A swizzle rerun | 9.7679 ms | 9.7674 | 9.7853 | 236/236 bitwise exact |
| unrestricted lookahead-2 selector | 9.0439 ms | 9.0429 | 9.0614 | M=5 exact; prefill unsafe |
| final exact-M5 + cache guard | 9.0476 ms | 9.0435 | 9.0839 | M=5 and prefill exact |

This aggregate queues the actual 236 checkpoint-layer calls in runtime order.
It is the first accepted microbenchmark result below the phase-one 10 ms AWQ
GEMM target.

## NCU Before And After

NCU report duration is used only for attribution because kernel replay changes
absolute timing.

| gate/up metric | fixed `8x256x64` | N64 + A swizzle | final lookahead 2 |
|---|---:|---:|---:|
| CTA grid | 68 | 272 | 272 |
| block threads | 128 | 64 | 64 |
| NCU duration | 99.456 us | 79.360 us | 73.696 us |
| DRAM throughput | 481.3 GB/s | 603.1 GB/s | 649.6 GB/s |
| DRAM peak | 43.24% | 54.14% | 58.37% |
| tensor-pipe active | - | 20.49% | 21.52% |
| registers/thread | 159 | 122 | 120 |
| dynamic shared memory | 20.51 KiB | 6.69 KiB | 6.69 KiB |
| achieved occupancy | 6.25% | 11.68% | 11.66% |
| scheduler no-eligible | 65.29% | 51.86% | 53.35% |
| executed instructions | - | 13.049 M | 11.782 M |
| sampled long-scoreboard, not issued | - | 559 | 334 |
| excessive shared wavefronts | about 2.10 M | 4,352 | 4,352 |

The remaining 4,352 excessive wavefronts are the small epilogue-store
residual. The dominant A `LDS.U.128` sites stay at ideal wavefront count after
the swizzle. The lookahead leaves occupancy and shared traffic unchanged, but
raises delivered DRAM throughput by 7.71%, lowers instruction count by 9.72%,
and lowers NCU duration by 7.14%. This is latency hiding and control-overhead
removal, not a precision or split-K change.

NCU PC sampling is stochastic. A second report with normalized SASS identical
to the final binary measured 72.672 us, 658.6 GB/s, and 232 long-scoreboard
samples. The final-binary report is used in the table; both replicas show the
same wall-time and bandwidth direction. They bound the long-scoreboard
reduction at 40-58% rather than supporting an exact sample-count claim.

Not every utilization percentage improves. Achieved occupancy remains about
11.7%, and no-eligible rises by 1.49 percentage points in the final sample.
The kernel is shorter and performs the same DRAM work faster, so the decisive
signals are elapsed time, bytes/second, instruction count, and exact output.

## Closed Experiments

These paths were measured and must not be repeated without a new mechanism:

| experiment | gate/up mean | decision |
|---|---:|---|
| fixed baseline | 88.311 us | reference for this sequence |
| N64 without A swizzle | 87.345 us | only 1.1%; insufficient |
| N32 without A swizzle | 82.757 us | faster but 4.19 M excessive wavefronts |
| N32 with A swizzle | 74.353 us | valid, slower than N64 |
| N64 with A swizzle | 71.291 us | accepted |
| move existing fetch earlier only | 69.072 vs 69.156 us A/B | noise; no structural gain |
| N64 two-register lookahead | 63.960 us | accepted final route |
| force gate/up split 2 | 59.029 us | 530 one-ULP differences; rejected |
| force N64 on MLP down | 40.626 us | regresses baseline 39.550 us |
| two-slot packed/converted fragment mainloop | 83.715 us | closed |
| one shared V metadata fragment per K128 group | 83.974 us | closed |
| V metadata reused for two K8 steps | 94.128 us | closed |
| force A-first compiler memory barrier | 73.568 us NCU | slower than 72.672 us control |

The compact mainloop reduced NCU registers from 122 to 107 but increased
executed instructions from 13.05 M to 14.08 M, did not increase theoretical
occupancy, and regressed NCU duration from 79.36 us to 91.10 us. The compiler's
original fully unrolled arrays provide useful independent dequant/HMMA work;
manual ping-pong serialized the load-convert-use chain.

V metadata loads are scalar, conflict-free shared loads. Keeping independent
copies costs registers but exposes dequantization ILP. Reusing one or four
copies lengthened dependencies and was slower. Register reduction is only
useful if it crosses a residency allocation boundary without destroying the
software pipeline.

Changing only gate/up from split 1 to split 2 reached 59.029 us, but changed
FP32 reduction associativity: 530 of 87,040 elements differed by one FP16 ULP.
It therefore fails the bitwise operator contract even though it had no sign
flips. Do not count this result toward the accepted route.

The first lookahead implementation put both compile-time branches behind a
generic lambda. That changed code generation for the default lookahead-one
path and regressed MLP down from about 39.55 to 44.43 us. The final code keeps
the old branch line-for-line and contains the paired lambda only in the
lookahead-two specialization. This compiler sensitivity is a required A/B
check for future mainloop refactors.

An empty inline-assembly memory clobber was also tested to force the A load
ahead of B. It changed compiler scheduling but regressed NCU duration from
72.672 to 73.568 us and increased sampled long-scoreboard stalls from 232 to
285. It was removed.

## 2026-07-11 Utilization Follow-Up

The accepted lookahead-2 route was used as the control for a second round of
dependency-hiding and MLP-down experiments. All runnable candidates remained
bitwise exact, but none passed the all-real performance gate.

| experiment | decisive result | decision |
|---|---|---|
| third A slot with two B/U/V slots | 177 registers/thread, 64-byte stack; representative `9.447 vs 8.951 ms` | reject: loses residency and wall time |
| reuse an A register after its shared store | 120 registers/thread; representative `9.179 vs 8.999 ms` | reject: dependency schedule is slower |
| store B/U/V before A | gate/up ABBA `63.120 vs 63.066 us` | reject: noise-to-regression |
| N32 A-swizzle lookahead 2 | gate/up `78.283 us` | reject: slower than accepted N64 |
| issue final-K HMMA before store/barrier | gate/up `65.168 vs 62.651 us` | reject: longer critical path |
| CTA_N=96 MLP down | generated a 10-byte `Array<half, 5>` access rejected by the SM70 iterator | reject: invalid vector shape |
| CTA_N=128 A-swizzle lookahead 2 | repeated-weight down `36.475 vs 40.133 us`, but two all-real controls regressed `9.053 -> 9.083 ms` and `9.076 -> 9.113 ms` | reject: hot-cache false positive |
| CTA_N=128 stock lookahead 1 | all-real `9.597 ms` | reject |
| CTA_N=256 A-swizzle MLP down | down `47.036 us`, 163 registers/thread, versus the usual `39-40 us` control range | reject: four-warp layout regression |

The CTA_N=128 experiment is an important methodology result. Repeating one
weight tensor made both its isolated down kernel and a short representative
mix look faster, but the complete 236-call sequence streams different weights
and reversed the result. A candidate is no longer accepted from a repeated
single-weight microbenchmark; it must win an interleaved same-GPU control on
all 236 real rank-local weights before NCU or serving validation.

After removing every rejected specialization, a rebuilt source-tree check
measured `9.0721 ms` over all 236 calls with 236/236 bitwise-exact outputs. The
small difference from the archived `9.0439 ms` result is within the observed
same-session thermal/run variance. The archived lookahead-2 binary remains the
NCU/micro reference, but the exact-M5 build below supersedes it for deployment.

## Strict Kernel-To-Endpoint Validation

The first endpoint check compared the optimized binary to a historical run
with a different generated sequence. That comparison showed almost no
throughput change and was not capable of measuring the kernel contribution.
A current-source registration-only A/B resolved the discrepancy.

The fixed binary removed only the two A-swizzled N32/N64 registrations. Its
all-real M=5 result was `11.3379 ms`. Restoring the unrestricted registrations
measured `9.0521 ms` on rank 0 and `9.0722 ms` on rank 1; all 236 tensors and
10,634,240 elements per rank were bitwise exact.

The unrestricted binary was nevertheless not endpoint-exact. The new CTA-M8
kernels were feasible for non-verifier shapes and changed 2,072 of 606,208
elements in a 74-token linear-attention QKV/Z prefill projection, with maximum
absolute error `0.00048828125`. This changed the dynamic-vocabulary bootstrap
and the later seeded probabilistic trajectory. M=1 and M=5 checks alone did
not expose the problem.

The accepted implementation wraps both registrations in
`ExactMKernelImpl<Gemm, 5>`. `Gemm::Dispatch` also revalidates exact and
lower-bound cache hits with `kernel->is_feasible()` so imported or reused
dispatch entries cannot bypass the exact-M contract. The final checks are:

| gate | fixed | final | result |
|---|---:|---:|---|
| M=5, 236 real calls | 11.3379 ms | 9.0476 ms | `-20.201%`, all elements exact |
| M=74 representative prefill | reference | 5/5 exact | no leakage |
| warm endpoint throughput | 95.9467 tok/s | 100.5474 tok/s | `+4.795%` |
| MTP round time | 41.3273 ms | 39.4363 ms | `-1.8910 ms` |

Both endpoint arms naturally stopped after 5,473 tokens and had the same
response hash, acceptance length `3.965217`, draft acceptance `74.1304%`,
accepted/draft/round counts, and per-position counts. Both passed the quality
gate. The endpoint realizes `82.6%` of the `2.2903 ms` microbenchmark saving;
the remaining about `0.40 ms` is graph critical-path overlap and scheduling
composition.

The strict ledger is
`bench_results/awq_m5_smalln_hmma_20260711/e2e_strict_current_source_ab/summary.md`.

## Reproduction

Final all-real run, using the exact-M5 selector and cache-guarded library:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
uv run --no-project \
  --python /home/ymzx/miniconda3/envs/vllm-0.0.5-t210/bin/python \
  python benchmarks/benchmark_sm70_awq_verifier_micro.py \
  --model /home/ymzx/models/Qwen3.6-27B-AWQ \
  --device cuda:0 --m 5 --tp-size 2 --tp-rank 0 \
  --all-real-layers --skip-case-timing \
  --aggregate-warmup 5 --aggregate-iters 30 \
  --op-library \
    bench_results/awq_m5_smalln_hmma_20260711/e2e_strict_current_source_ab/_C.optimized_m5_only_cache_guard.abi3.so \
  --reference-outputs \
    bench_results/awq_m5_trt_gemv_20260711/real236_fixed_baseline_gpu2_outputs.pt
```

Accepted binary SHA-256:

```text
7643aac62ad000f2e9d4ab17ac712387ae31685babb2df9ef909fb0493af6141
```

Primary artifacts are under:

```text
bench_results/awq_m5_smalln_hmma_20260711/
```

## Remaining Bottleneck

The gate/up kernel is faster but is not at a V100 hardware ceiling:

- DRAM is 649.6 GB/s, 58.37% of the NCU peak, so this is not a saturated HBM
  roofline;
- occupancy is still 11.66% with 7.46 active warps/SM, and NCU classifies the
  272-CTA grid as only 0.5 full waves relative to the register-limited block
  capacity;
- wait, long-scoreboard, and no-instruction are the three largest final
  not-issued sample classes;
- 260 of the final report's 334 long-scoreboard samples land on
  `STS.128 [R51], R44`, whose producer is the cached activation
  `LDG.E.128.CONSTANT.SYS R44`; the steady-state streaming-weight stores are
  now mostly ready;
- shared conflicts are no longer a material target, and a plain N32 increase
  in CTA count was already slower.

The asymmetric deeper-A idea is now closed in its straightforward forms. A
third A slot compiled to 177 registers/thread plus local stack traffic, while
register rollover, store reordering, and final-K scheduling all lost wall
time. The sampled A-load stall is therefore a symptom of the current schedule,
not an independently removable delay at unchanged resource cost.

MLP down remains about 2.50 ms of the representative weighted sum. Its older
NCU report shows 1.06 M excessive shared wavefronts, only 478.6 GB/s DRAM
throughput, 159 registers/thread, and 11.41% achieved occupancy. However,
mechanically applying the two-warp N64 A swizzle to its four-warp N256 kernel
increased runtime to 47.036 us and registers to 163. CTA_N=128 only won when
the same weight stayed hot in cache and lost on the real 236-weight stream.

The next valid structural route is a dedicated four-warp N256 operand layout
with warp-private A storage, or a specialized CTA that contains two
independent N32 warp groups and removes their CTA-wide synchronization. Either
route requires a new operand/mainloop contract; it is not another selector or
swizzle parameter change. Split-7 reduction order, K order, FP32 accumulation,
and bitwise output remain fixed. All-real interleaved A/B is the first
performance gate for either design.

### Rejected Direct Four-Warp A Swizzle (2026-07-12)

The existing `Operand_A_Swizzle_8x64` was reused exactly for the TP4 row
`8x256x64` kernel. It is a valid bank-conflict fix: NCU reduces excessive
shared wavefronts from `211,200 (47%)` to `15,360 (6%)`, and the selected
real-layer output is bitwise exact. It is not an optimization by itself.

The layout raises registers/thread `159 -> 163`, executed instructions
`1,768,976 -> 1,860,480`, and NCU duration `30.56 -> 31.90 us`, while
no-eligible cycles remain approximately `81%`. Its XOR address arithmetic
increases instruction-fetch pressure enough to offset the eliminated A-load
conflicts. The temporary diagnostic registration has been removed. Do not
retry the same whole-tile swizzle; a future layout must retain the generic
route's register count or eliminate a larger synchronization/dependency cost.

### Rejected Eight-Warp N32 Decomposition (2026-07-12)

Changing only `MMA_Map` from four N64 warp groups to eight N32 groups keeps the
same `8x256x64`, split-12 work partition and is bitwise exact. It improves the
local NCU counters (`159 -> 115` registers/thread, `16.03% -> 22.63%`
occupancy, and `81.29% -> 73.45%` no-eligible cycles), but increases total
executed instructions `1.769M -> 2.151M`. Alternating CUDA-event pairs were
`-0.45%` and `+0.44%`, so there is no repeatable whole-operator gain.

The current mainloop replicates staging/control work for each new group. Do
not retry a bare `TG_N=8` registration. A future warp-private or independent
group implementation must first assign shared A/B/V loads to non-overlapping
warps and prove that it removes, rather than duplicates, the extra work.

## 2026-07-12 TP4 M=1 Gate/Up CTA Fill

TP4 turns the merged gate/up descriptor into `M=1,N=8704,K=5120`. The old
`8x256x64`, split-2 tactic created only 68 CTAs, below the 72 V100 SMs. This is
a different scaling failure from the accepted M=5 verifier path above.

The accepted TP4 tactic is `8x64x64`, split 2, swizzle 4, with the existing
conflict-free A layout and lookahead-two mainloop. It is registered through
`ExactMnkKernelImpl<..., 1, 8704, 5120>` and selected only for the exact AWQ
descriptor. It cannot become feasible for prefill, M=5 verification, or the
TP4 row-projection shapes.

| metric | TP4 old | TP4 N64 candidate |
|---|---:|---:|
| CTA count | 68 | 288 |
| registers/thread | 159 | 120 |
| dynamic shared memory | 20.51 KiB | 6.69 KiB |
| achieved occupancy | 6.24% | 11.51% |
| DRAM peak throughput | 37.35% | 48.67% |
| SM throughput | 27.23% | 37.04% |
| rank-0 236-call real-weight aggregate | 7.1955 ms | 6.4429 ms |

Rank 0 and rank 1 each pass all 236 real calls and 1,385,984 output elements
with zero difference from their fixed route. A full same-binary no-MTP TP4
CUDA-graph A/B also preserves every generated token ID and text over three
512-to-256 greedy repeats while improving `70.7158 -> 74.2807 tok/s` and
`14.1411 -> 13.4624 ms` TPOT (`+5.04%`, `-4.80%`). This endpoint lane uses
`VLLM_USE_AOT_COMPILE=0`; it is a graph decode performance proof, not a
replacement for the official-sampling MTP quality lane.

The strict ledger, exactness artifacts, NCU report, and rejected candidates are
in `bench_results/tp4_awq_m1_cta_fill_20260712/summary.md`.

## 2026-07-12 TP4 M=1 Row Projection K64

The TP4 row descriptor `M=1,N=5120,K=1536` has enough baseline CTAs but
spends most issue gaps at CTA barriers. Its accepted tactic is therefore
different from the gate/up N64 fill: `8x256x64`, `MMA=1x4x1`, split 12,
swizzle 4. The selector pins `s884_1x4x1` to avoid the unverified `TG_K=2`
sibling.

The larger K64 tile preserves the 12-way reduction contract while reducing
staging/control phases. All 236 real calls on both ranks are exact. Paired
all-real rank-0 timing is `6.4509 -> 6.1731 ms` (`-4.31%`) on top of the
accepted gate/up route. The no-MTP TP4 graph result becomes
`74.2807 -> 75.9375 tok/s` and `13.4624 -> 13.1687 ms` TPOT, with identical
three-repeat token IDs/text.

This is not an occupancy win: NCU reports `24.13% -> 16.08%` achieved
occupancy and a shared-load conflict regression from 4.1-way to 4.7-way. The
useful change is executed instructions `2.50M -> 1.78M` and the leading stall
shifting from CTA barrier to instruction fetch. The direct four-warp A swizzle
and bare eight-warp decomposition were later rejected: the next valid row
mechanism must give A/B/V staging explicit non-overlapping warp ownership, not
change a selector, split count, or shared-memory XOR alone.

Full evidence and the cache/selector constraint are in
`bench_results/tp4_awq_m1_row_20260712/summary.md`.

## Rejected TP4 Gate/Up M1 Live-A HMMA (2026-07-12)

The TP4 gate/up descriptor is `M=1,N=8704,K=5120` and uses the accepted
`8x64x64`, split-2, two-warp route. Its `(16,9,2)` launch grid has 288 physical
CTAs but only 272 useful N tiles. The N32/N64/N128 alternatives have roughly
the same total useful warp supply, so another CTA-size sweep cannot solve its
underfill.

The diagnostic retained HMMA, K64 order, group-128 scale cadence, two-way
split-K FP32 reduction, and the accepted B/U/V pipeline. It replaced invalid
M8-row A shared reads with explicit zero fragments while keeping the live m=0
A shared broadcast. It passed a real-layer bitwise comparison (`8,704 / 8,704`
elements), so the rejection is strictly performance-based.

The first gate failed decisively: CUDA-event time regressed `35.30 -> 41.42 us`
(`+17.3%`). NCU attributes the regression to register and code growth, not to
memory bandwidth: registers/thread `120 -> 168`, theoretical occupancy
`25% -> 18.75%`, executed instructions `5.919M -> 6.341M`, DRAM throughput
`48.67% -> 40.75%`, and no-eligible cycles `58.45% -> 60.55%`. The temporary
mainloop, config, and diagnostic registration were removed. Do not revive this
copy-and-specialize structure; a future M1 route must lower, not expand, the
live register footprint before it can be benchmarked again.

## 2026-07-12 TP4 M=1 QKVZ CTA Fill

The TP4 linear-attention QKVZ descriptor is `M=1,N=4096,K=5120`. It was still
using `8x256x64`, split 4: only 64 CTAs with four warps each. The accepted
exact-MNK route is `8x64x64`, split 4, swizzle 4, with the existing
conflict-free A layout and lookahead-two mainloop. It preserves K64 order,
four-way serial FP32 split-K reduction, and prepared AWQ layout.

| metric | prior `8x256x64` | exact N64 |
|---|---:|---:|
| one real QKVZ layer CUDA-event | 27.896 us | 22.729 us |
| weighted 47-call representative | 1.3111 ms | 1.0683 ms |
| rank-0 all-real 236-call aggregate | 6.1747 ms | 5.9398 ms |
| rank-1 all-real 236-call aggregate | 6.1540 ms | 5.9414 ms |

All 236 real checkpoint tensors on both ranks are bitwise equal. NCU attributes
the win to TP4 resource fill: physical grid `64 -> 256` CTAs,
registers/thread `159 -> 120`, register-limited resident blocks/SM `3 -> 8`,
active warps `6.23% -> 9.98%`, DRAM `27.24% -> 32.18%`, SM throughput
`20.51% -> 25.13%`, and NCU duration `37.920 -> 32.416 us`.

Do not inherit this rule for MLP-down `M=1,N=5120,K=4352`. Exact N64, N128,
and N256/swizzle4 all regress (`25.71 -> 27.99`, `28.41`, and `30.03 us`),
because the split-7 epilogue cost dominates. Artifacts are in
`bench_results/tp4_awq_m1_cta_fill_20260712/`.

### QKVZ Same-Binary No-MTP Endpoint A/B (2026-07-12)

The QKVZ result is also visible in the full TP4 CUDA graph, not only in the
isolated operator timing. Both arms use Qwen3.6-27B-AWQ, TP4, TurboMind AWQ,
custom all-reduce, `FLASH_ATTN_V100`, input/output `512/256`, greedy
`temperature=0/top_p=1/top_k=-1/ignore_eos`, and `VLLM_USE_AOT_COMPILE=0`.
They differ only in `VLLM_SM70_AWQ_TP4_QKV_CTA64`:

| QKVZ N64 selector | steady decode | TPOT | output TPS |
|---|---:|---:|---:|
| off, prior `8x256x64` tactic | 76.0039 tok/s | 13.1572 ms | 72.7278 tok/s |
| on, exact `8x64x64`, split 4, swizzle 4 | 77.3787 tok/s | 12.9235 ms | 73.9729 tok/s |
| delta | +1.3747 tok/s, +1.81% | -0.2338 ms, -1.78% | +1.71% |

All five repeats have identical token IDs and text between arms. The candidate
startup and graph-capture trace explicitly selects `8x64x64`, split 4,
swizzle 4. The `0.2338 ms` graph saving matches the two-rank all-real
operator saving (`0.21-0.24 ms`), so the endpoint result is not inferred from
a synthetic microbenchmark. The selector remains on by default; the
environment variable is an A/B diagnostic gate only. Artifacts are
`bench_results/tp4_awq_m1_cta_fill_20260712/endpoint_qkv_cta64/`.

## Rejected TP4 M=1 Three-Slot Global Lookahead (2026-07-12)

The source-level NCU `LDG -> STS.128` long-scoreboard samples motivated an
exact TP4 gate/up experiment with a third complete A/B/U/V global-load
fragment slot. It targeted only `M=1,N=8704,K=5120` through an explicit
`_gmem_l3` selector name and retained K64 order, group-128 metadata cadence,
shared-store order, HMMA accumulation, split-2 reduction, and FP16 conversion.
The real output is bitwise exact (`8,704 / 8,704`).

The resource cost is decisive: `120 -> 165` registers/thread, with the same
64-byte stack, zero local memory, and 6,688-byte dynamic shared memory. The
first GPU-1 screen was non-positive, but the final clean-control check used
physical V100 GPU 2. Clean two-slot builds average `34.2705 us`; the exact
three-slot candidate averages `46.0676 us` (`+34.4%`). Merely compiling the
generic three-slot branch changes the two-slot control to `46.2771 us`
(`+35.0%`) despite its unchanged 120-register resource report. The candidate
is only `-0.45%` versus that polluted control, so it is not a valid win. No
all-real, NCU, or endpoint work was run.

The specialization was removed rather than retained as a dormant selector.
Do not repeat generic `GmemLookahead=3` for TP4 gate/up. A successor needs to
keep clean two-slot code generation unchanged, provide a new
register/occupancy argument with at most 128 registers/thread or proof of a
new occupancy boundary, and show a repeatable at-least-2% real-weight win
against the clean control before profiling. Evidence:
`bench_results/tp4_awq_m1_cta_fill_20260712/gmem_l3/summary.md`.

## Rejected TP4 MLP-down Split-7 Deferred Reduction (2026-07-12)

The TP4 MLP-down descriptor is `M=1,N=5120,K=4352`; the real production route
is `8x256x64`, `mma=1x4x1`, split 7, swizzle 0. The serial split-K semaphore
chain looks removable in isolation: a correct synthetic split-7 reducer improves
`10.4850 -> 5.5552 us` (`-47.0%`). That result is not a valid GEMM predictor.

The exact candidate wrote each split's FP32 accumulator plane and used a second
same-stream reducer to replay `z=0..6` addition order. It is bitwise exact on a
real layer (`5120 / 5120` equal FP16 values), but CUDA-event timing on physical
V100 GPU2 rejects it: production is `24.375 us` per call (`1.5356 ms` for 63
down-proj calls) and stage plus reducer is `38.7448 us` (`2.4409 ms`, `+58.9%`).
The complete seven-plane global write/read round trip and the extra launch cost
more than the serial-chain removal saves.

NCU is useful only for resource attribution here: the 20-CTA reducer has
`0.03` waves/SM and 12.4% achieved occupancy, while the stage kernel remains
at 158 registers/thread versus 159 for baseline. Nsight timing is excluded from
the gate because profiling perturbs this synchronization-sensitive route.

The registration, selector, and epilogue templates were removed. Do not repeat
generic plane materialization plus external or last-arrival reduction unless a
new exact design eliminates the full partial-plane round trip and first clears a
2% real-layer CUDA-event win. Evidence:
`bench_results/tp4_awq_m1_cta_fill_20260712/split7_deferred/summary.md`.

## Closed TP4 MLP-down Scheduling and TG_N Screens (2026-07-12)

The remaining `8x256x64`, split-7 route was screened without changing the
seven K=128 partial boundaries. Swizzle 1 and swizzle 2 are bitwise exact but
fail a long CUDA-event gate: swizzle 0 is `24.0165 us`, while swizzle 1 is
`24.5618 us` and swizzle 2 is `24.2871 us` (`+1.13%`). Short samples made
swizzle 2 look favorable due to split-K tail variance; it is not a default win.

An explicit `TG=1x8x1` candidate also preserved CTA `8x256x64`, K64, split 7,
and all output bits. It reduced resources from the baseline 159 registers to
115 with zero local memory, but changes each warp from an N64 tile to N32. The
real-layer CUDA-event result is `24.1195 -> 28.3348 us` (`+17.5%`). The extra
per-warp HMMA/shared/control work overwhelms the expected residency gain; the
registration was removed. Do not retry bare `TG_N=8`. Evidence:
`bench_results/tp4_awq_m1_cta_fill_20260712/split7_split_sweep/summary.md` and
`bench_results/tp4_awq_m1_cta_fill_20260712/tgn8/summary.md`.

## TP4 MTP4 M=5 Exact Selector Bridge (2026-07-12)

The TP4 MTP verifier has four AWQ shape families. Only the two column-major
families can safely inherit the current no-MTP small-N tactics:

- `5x8704x5120` gate/up uses `8x64x64`, split 2, swizzle 4;
- `5x4096x5120` QKVZ uses `8x64x64`, split 4, swizzle 4.

Both are default-on in `gemm.cu` and can be disabled only for same-binary
diagnosis with `VLLM_SM70_AWQ_MTP_M5_FAST_SELECTOR=0`. The legacy global
fast-selector gate remains separate. The `5x5120x1536` row candidate was
rejected: it was faster but changed 3,836 FP16 values on the rank-2 all-real
suite, so it must not be revived as a serving optimization.

The accepted paths are bitwise exact for all 236 real checkpoint calls on each
of TP0--TP3 (6,929,920 FP16 values per shard, maximum error and sign changes
zero). Final same-binary all-rank timing on V100 GPU2 gives a maximum-rank
M=5 AWQ verifier aggregate of `6.8065 -> 5.8273 ms` (`-0.9792 ms`, `-14.39%`).
The four rank means are within 0.003 ms, so this is not hidden by a slow shard.

The current TP4 MTP event profile measures target-forward p50 at `18.433 ms`.
Thus the complete exact AWQ GEMM component is about 31.6% of target forward;
the remaining target-forward work must be decomposed before chasing draft
sampling. The latest artifacts and endpoint caveat are recorded in
`bench_results/mtp4_current_binary_20260712/latest_mtp4_latency_analysis.md`.
