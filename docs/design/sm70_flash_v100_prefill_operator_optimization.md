# SM70 Flash-V100 Long-Prefill Operator Optimization

## Scope And Acceptance Shape

This document is the experiment ledger for the exact paged Flash-V100
full-attention operator used by Qwen3.6-27B-AWQ TP4 prefill. It is separate
from the end-to-end long-context ledger so rejected kernel designs are not
repeated.

The acceptance microbenchmark is:

- V100 SM70, one TP rank
- `M=1024`, `N=65536`, `Hq=6`, `Hkv=1`, `D=256`
- fp16 KV cache, page size `784`
- `BLOCK_M=16`, `BLOCK_N=128`, 16 warps/CTA
- output accumulator stride `268`
- exact causal output, no changed softmax or FP32 accumulation order

The accepted starting point is approximately `40.22 ms/layer`. The first
pipeline gate was `<=35.0 ms/layer`, bitwise equality on normal, tail, and
reversed block tables, 64 or fewer registers per thread, two CTAs/SM, and no
new local-memory traffic. The gate was reached on 2026-07-15 by the
cross-block steady/drain pipeline plus conflict-free Q/P layouts:
`39.064 -> 34.249 ms` against the matching early-store baseline and about
`-14.87%` from the original `40.23 ms` operator.

## Accepted Starting Point

The `sO=268` layout reduced excessive shared wavefronts by 38.6% and improved
the original operator by 8.67%. The later early probability-store candidate
writes each rounded fp16 probability directly to the separate D256 `sP`
buffer before the row-sum reduction. It does not change exponentiation,
rounding, online-softmax state, output scaling, or HMMA accumulation order.

| route | stack frame | alternating A/B result | status |
|---|---:|---:|---|
| `sO=268` baseline | 144 B | about `40.23 ms` | accepted default |
| early `sP` store | 48 B | `40.230 -> 39.185 ms`, `-2.596%`, 200/200 wins | accepted prerequisite |
| cross-block steady/drain | 1024 B | `39.153 -> 38.222 ms`, `-2.377%`, 40/40 wins | accepted prerequisite |
| cross-block plus Q/P swizzle | 1024 B | `39.064 -> 34.249 ms`, `-12.326%`, 200/200 wins | current accepted candidate |

The early-store tail/reversed-table gate at `M=1024,N=65521` is bitwise equal
and improves `40.245 -> 39.194 ms` (`-2.612%`, 20/20 wins).

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/p_early_store_ab200_m1024_n65536.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/p_early_store_reverse_tail_ab20_m1024_n65521.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/cross_block_panel0_steady_drain_ab40_m1024_n65536.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/cross_block_swizzled_qp_ab200_m1024_n65536.json`

## NCU Phase Decomposition

The full NCU report for early-store retains 64 registers, 36.35 KB dynamic
shared memory, two CTAs/SM, and 47.14% achieved occupancy. DRAM throughput is
only 0.53% and L2 hit rate is 99.23%; this is not an HBM-bandwidth ceiling.
Tensor-pipe active is 9.29%, while the scheduler has no eligible warp in
66.14% of cycles.

PC sampling separates the steady kernel into the following static SASS
regions. Sample share is an attribution signal, not a sum of independent wall
timers.

| phase | sample share | long scoreboard | barrier | short scoreboard | MIO throttle |
|---|---:|---:|---:|---:|---:|
| setup/page | 0.02% | 97 | 37 | 17 | 11 |
| QK | 30.31% | 166,885 | 314,143 | 24,938 | 9,324 |
| softmax/output scale | 27.74% | 0 | 48,366 | 178,402 | 51,766 |
| PV | 40.85% | 405,769 | 107,672 | 43,079 | 84,294 |
| outer/final | 1.07% | 17,944 | 0 | 32 | 78 |

Of all long-scoreboard samples, 94.81% land on HMMA step 0. Within those
HMMA samples, PV `ROW/ROW` contributes 71.97% and QK `ROW/COL` contributes
28.03%. QK maps only eight score tiles, so eight of the CTA's sixteen warps
wait at the QK completion barrier. PV issues direct global V fragment loads
immediately before HMMA; the first HMMA group in each WMMA operation carries
most of that dependency wait.

Early-store removes `14.06M` local loads and `12.50M` local stores relative to
the `sO=268` baseline. It does not improve the HMMA dependency chain: HMMA
long-scoreboard samples remain approximately `560K`. The next speedup must
therefore overlap independent QK/PV work rather than further shorten the
softmax temporary lifetime.

Artifact:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_p_early_store_m1024_n65536.ncu-rep`

## Experiment Ledger

Every result below uses same-process alternating timing and an exact output
comparison. Rejected implementations are removed from the active source unless
the same code has a separately useful opt-in route.

| candidate | exact | result versus its matching baseline | decision |
|---|---|---:|---|
| 8 producer / 8 consumer QK shared double buffer | yes | `40.221 -> 53.720 ms`, `+33.56%` | reject; barrier and global-to-shared round trip dominate |
| 8 producer / 8 consumer PV shared double buffer | yes | `40.212 -> 42.750 ms`, `+6.31%` | reject |
| warp-local QK double-B lookahead | yes | about `-0.56%`, 40/40 wins | retain opt-in; does not stack with early-store |
| warp-local PV B lookahead | yes | `+1.26%` | reject; longer fragment lifetime |
| double A+B lookahead | yes | QK unchanged; PV `+4.51%` | reject; stack/liveness increase |
| QK triple-B lookahead | yes | `-0.59%` | reject; no gain over double-B |
| coarse PV shared stage, pitch 128 | yes | `+10.47%` | reject |
| coarse PV shared stage, pitch 136 | yes | `+3.70%` | reject |
| coarse PV shared stage, pitch 152 | yes | `+3.60%` | reject; lower conflicts do not repay staging cost |
| early `sP` store | yes | `-2.596%`, 200/200 wins | current best |
| early-store plus QK double-B | yes | `-2.583%` versus original | reject as a combination; no gain beyond early-store |
| QK-idle-warp V L1 prefetch | yes | `39.219 -> 39.408 ms`, `+0.482%`, 0/20 wins | reject and remove |
| cross-block loop with runtime final-block predicate | yes | local loads `16.04M` | reject; loop-carried state spills |
| cross-block compile-time steady loop plus generic final drain | yes | `39.153 -> 38.222 ms`, `-2.377%` | accept; local loads reduced to `3.58M` |
| spread eight next-QK tiles over four PV panels (`2+2+2+2`) | yes | `+5.39%` | reject; QK parallelism and barrier cost dominate |
| true 8 producer / 8 consumer named-barrier schedule | yes | `+4.84%` | reject; barrier `9.31%`, tensor active `8.89%` |
| direct global Q WMMA load | yes | `+28.09%` | reject; about `1.99B` global sectors |
| paired QK accumulator reuse | yes | only `-1.68%` versus baseline | reject; local loads `105M`, stores `25.1M` |
| compact conflict-free Q layout | yes | `39.240 -> 36.446 ms`, `-7.122%` | accept |
| Q duplicate-lane shuffle | yes | only `-3.97%` versus baseline | reject; shuffle cost exceeds shared-load saving |
| compact conflict-free Q and P layouts | yes | `39.064 -> 34.249 ms`, `-12.326%`, 200/200 wins | accept; first-stage target reached |
| K/V global-load-first source reordering | yes | `49.654 -> 43.575 ms`, `-12.244%` at locked 1200 MHz | reject; no incremental gain over Q/P swizzle |
| direct vector construction of col-major WMMA B fragment | yes | `43.584 -> 43.868 ms`, `+0.652%`, 0/100 wins at locked 1200 MHz | reject and remove; same global bytes, no duplicate-lane elimination |
| half-lane col-major B loads plus eight fragment shuffles per K16 | yes | QK panel `185.536 -> 199.296 us`, `+7.416%`, 0/100 wins | reject before NCU/integration; 128 added SHFL per warp chain |
| raw-HMMA QK, best B-reuse shape | yes | `212.352 -> 251.008 us`, `+18.204%`, 0/100 wins | reject; splitting native M16 adds serial HMMA chains and/or B rereads |
| raw-HMMA PV, single/staged | yes | `34.432 -> 43.904 us`, `+27.509%`, 0/100 wins | reject; native row/row remains faster |
| native BM32 QK, B reused by top/bottom M16 | yes | `176.704 -> 145.152 us`, `-17.856%`, 100/100 wins | accept at panel gate; integrate only through a low-smem full pipeline |
| native BM32 PV, V reused by top/bottom M16 | yes | `122.240 -> 108.928 us`, `-10.890%`, 100/100 wins | accept at panel gate; integrate only with register-resident output |

The L1-prefetch rejection is counter-based, not only timing-based. L1 global
load hit sectors rose by `15.67M` and misses fell by `12.89M`, but PV HMMA
long-scoreboard samples rose `403,052 -> 462,351`. Local loads rose
`5.11M -> 9.79M`, and the named midpoint barrier merely split the existing QK
wait. This route must not be retried by changing prefetch volume or panel
order.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/v_l1_prefetch_ab20_m1024_n65536.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_v_l1_prefetch_m1024_n65536.ncu-rep`

## Accepted Cross-Block Pipeline And Layout

The accepted kernel overlaps QK for block `n+1` with the first 32-token PV
panel of block `n`. A compile-time steady-state loop processes
`num_n_tiles - 1` blocks; the existing generic implementation drains the last
block. This split matters: keeping a runtime `has_next` value live through the
hot loop caused local-memory traffic (`16.04M` local loads), while the
steady/drain form reduces it to `3.58M` local loads and `0.074M` local stores.

The initial cross-block implementation was still limited by shared-memory
replay. A dedicated SM70 fragment probe established the exact matrix-A lane
mapping and proved that the following compact layout produces all 32 by 8
WMMA fragment words bit-for-bit:

```text
row  = (lane & 3) + 4*((lane >> 4) & 1) + 8*((lane >> 2) & 1)
slot = (row & 3) | ((row & 8) >> 1) | ((row & 4) << 1)
```

Q and each 16x32 probability panel use two eight-half planes with this row-bit
swap. Two `ld.shared.v4.u32` instructions construct the existing WMMA
matrix-A fragment directly; HMMA operands, softmax, fp16 probability rounding,
FP32 accumulator order, and final output order are unchanged.

Formal gates:

| case | matching baseline | Q/P swizzle | delta | exactness |
|---|---:|---:|---:|---:|
| `M1024,N65536`, 200 alternating pairs | `39.064 ms` | `34.249 ms` | `-12.326%` | 200/200 bitwise |
| reversed table, tail `N=65521` | `39.152 ms` | `34.332 ms` | `-12.311%` | 20/20 bitwise |
| `N=16K` | `9.716 ms` | `8.558 ms` | `-11.915%` | 20/20 bitwise |
| `N=32K` | `19.526 ms` | `17.136 ms` | `-12.243%` | 20/20 bitwise |
| `N=128K` | `78.347 ms` | `68.686 ms` | `-12.331%` | 20/20 bitwise |

Resource gates remain valid: 64 registers/thread, 44.93 KB dynamic shared
memory, two CTAs/SM, and unchanged local-memory instructions. Excessive shared
wavefronts fall from `637.8M` to `275.5M` (`-56.8%`). The NCU replay duration
falls to about `39.98 ms`; the unprofiled CUDA-event result is the accepted
`34.249 ms` number.

Artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/wmma_matrix_a_fragment_stride16.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/wmma_matrix_a_swizzled_fragment_compare.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/cross_block_swizzled_qp_ab200_m1024_n65536.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/cross_block_swizzled_qp_reverse_tail_ab20_m1024_n65521.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/cross_block_swizzled_qp_scaling_16k_128k_ab20.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_cross_block_swizzled_qp_full.ncu-rep`

## Full-Model Propagation Gate

Uncontrolled 128K runs are too noisy for a promotion decision: two candidate
runs differed by about 6.8 seconds. The auditable endpoint gate therefore
locks all four V100 SM clocks at 1200 MHz and compares the same binary,
sampling parameters, prompt hash, TP4 topology, CUDA-graph policy, and
`max_num_batched_tokens=1024`. The baseline keeps the accepted early-store
route (`PV=1,QK=0`); the candidate changes only `QK=1` so both swizzles become
active.

| 64K TP4 metric | baseline | candidate | delta |
|---|---:|---:|---:|
| prefill time | `47.9785 s` | `44.8346 s` | `-3.1439 s`, `-6.5526%` |
| steady decode TPOT | `19.0642 ms` | `19.2086 ms` | decode-neutral noise |
| route hits per rank | 1008 | 1008 | `prefill_prefix_flash` |
| output tokens | 64 | 64 | all token IDs equal |

This proves that the operator gain reaches the real model. It does not imply a
6.55% decode gain; this workstream changes chunked prefill only.

The exact page-784 route is now default-enabled. Setting either
`VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_QK=0` or
`VLLM_FLASH_V100_PREFILL_D256_SW_PIPELINE_PV=0` restores the corresponding
diagnostic baseline. Other page sizes remain unchanged. A default-on smoke at
the locked 1200 MHz clock measures `43.600 ms` for the acceptance micro shape,
matching the explicit candidate lane.

Artifacts:

- `bench_results/nomtp_long_context_20260715/prefill_pipeline_baseline_pvonly_64k_m1024_tp4_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_pipeline_swizzled_qp_64k_m1024_tp4_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_pipeline_64k_clock1200_compare.json`

## Residual Bottleneck And Next Gate

After Q/P swizzling, NCU reports 48.10% achieved occupancy, 36.23% issue-active
cycles, and 63.77% cycles with no eligible warp. Tensor-pipe active improves
only to 10.61%. Of all long-scoreboard samples, 94.73% remain on HMMA step 0:
PV `ROW/ROW` contributes 61.61% and QK `ROW/COL` 33.12%. Earlier revisions of
this document had those phase labels reversed. The PV producer group has four
`LDG.E.64.SYS` instructions; QK uses two `LDG.E.128.SYS` instructions per K16.
For QK, duplicate fragment lanes increase thread-side materialization to
1024 bytes for a logical 512-byte tile, but source counters show only 16 L1
tag/L2 theoretical sectors per E128 instruction. The duplicate lanes are
already coalesced at that level, so they are not the primary memory-transaction
bottleneck.

The SM70 matrix-B probe is now complete. For col-major B, each lane owns one
logical 16-element column and lane `lane ^ 4` owns a duplicate fragment.
Loading that column with two `ld.global.v4.u32` instructions constructs all
eight fragment words bit-for-bit. This is a mapping result, not a performance
result: every duplicate partner still performs the same loads, so the warp
still requests 1024 bytes for a logical 512-byte tile.

The corresponding full-kernel experiment is closed. An initial cache-strong
variant compiled to `LDG.E.128.STRONG.CTA` and lost `0.663%`. Restoring the
native weak/system scope generated `LDG.E.128.SYS` but still lost
`43.584 -> 43.868 ms` (`+0.652%`, 0/100 alternating wins). Both variants are
bitwise exact and remain at 64 registers/thread; neither reduces requested
bytes or HMMA operand wait. The main-kernel template and environment switch
were removed. The standalone fragment probe remains as evidence.

A half-lane loader was then tested at the required QK panel boundary instead
of in the full attention kernel. Only lanes with `(lane & 4) == 0` perform the
two vector loads; all lanes reconstruct eight words from source
`lane & ~4`. The complete `BM16,BN128,K256` output is bitwise equal across
2,097,152 FP32 words and stays spill-free (43 registers for the candidate),
but SASS adds 128 data shuffles to each warp's K loop. At locked 1200 MHz it
regresses `185.536 -> 199.296 us` (`+7.416%`) and loses all 100 alternating
pairs. This proves that reducing participating load lanes does not repay the
Volta shuffle dependency chain; NCU and integration were correctly skipped.

The operand/order probe for an explicit Volta raw-HMMA `BM8,BN32` atom now
passes its correctness gate. It compares two native `m16n16k16` N tiles with
two raw M8 fragments over a complete `16x32` output, nonzero FP32 C, QK
row/col, PV row/row, and random, alternating-sign, and exponent-span inputs.
Across all 24 K4 permutations, only `[0,1,2,3]` matches every case bit-for-bit
over all 512 FP32 words. This establishes the required native accumulation
order; it does not establish a speedup.

The raw-HMMA branch is closed. The canonical K4 order is correct, but the best
QK variant still regresses `212.352 -> 251.008 us` despite 64 registers and no
spill. The PV variants regress `34.432 -> 43.904 us`. Raw `m8n8k4` divides a
native M16 operation into additional serial chains and, for most shapes,
rereads B. More register scheduling cannot remove that structural cost.

The measured opportunity is instead native BM32 reuse. The QK candidate keeps
the native `m16n16k16` chain and loads each K fragment once before applying it
to top and bottom M16 query panels. It improves `176.704 -> 145.152 us`
(`17.856%`) with 100/100 alternating wins and 4,194,304 bitwise-equal FP32
words. NCU measures `165.760 -> 138.752 us`; L1 load requests fall about
47.1%, L2 sectors 35.3%, total instructions 21.8%, and long-scoreboard issue
share `62.32% -> 52.70%`.

The PV candidate similarly loads each V fragment once for top and bottom M16.
At 1024 groups it improves `122.240 -> 108.928 us` (`10.890%`) with 100/100
wins and 8,388,608 bitwise-equal FP32 words. Both native candidates compile at
64 registers/thread with no local memory. Their 256-thread panel kernels reach
four CTAs/SM, so the gain is load and instruction reuse rather than a change in
resident warp count.

The active gate is now a complete BM32 QK-softmax-PV pipeline, not another
isolated instruction experiment. It must meet all of the following before the
main kernel changes:

1. preserve the exact native M16 QK and PV accumulation order, per-row
   softmax reduction, fp16 probability rounding, and online rescale order;
2. use eight QK producer warps and eight persistent PV consumer warps, with
   top/bottom B and V reuse proven by the accepted panels;
3. eliminate shared `sO`, keep total shared memory at or below 48 KB, compile
   at no more than 64 registers/thread with no `LDL/STL`, and retain two
   CTAs/SM;
4. use a bounded score lifetime, currently designed as three M16x128 FP32
   half-buffers, so next-block QK can overlap current softmax/PV without a
   second full score matrix;
5. beat a matching complete BM16 QK-softmax-PV microbaseline by at least 2%
   before integration;
6. after integration, beat the accepted `M1024,N65536` path by at least 2% at
   locked 1200 MHz while preserving every output bit.

### Complete-pipeline resource rejection

The first complete BM32 prototype used one 512-thread CTA, eight QK producer
warps, eight PV consumer warps, three M16x128 FP32 score half-buffers, two
top/bottom P ping-pong panels, and four register-resident PV accumulators per
consumer warp. Its score/P lifetime and barrier counts are race-free, and its
45,456-byte shared footprint would permit two CTAs on a 96 KB V100 SM.

It fails the register gate before correctness or timing:

| build | registers/thread | stack/thread | shared/CTA | decision |
|---|---:|---:|---:|---|
| natural, one-CTA launch bound | 128 | 8 B | 45,456 B | reject; only one CTA can reside |
| `__launch_bounds__(512,2)` | 64 | 480 B | 45,456 B | reject; 299 static `LDL/STL` instructions |
| inline `bar.sync 0,512` diagnostic | 64 | 480 B | 45,456 B | reject; barrier spelling does not remove spill |
| shared-epoch producer/consumer | 64 | 96 B | 45,600 B | reject; zero-barrier protocol still spills PV state |

The hard two-CTA register budget is
`65,536 / (512 threads * 2 CTAs) = 64 registers/thread`. The four persistent
PV accumulator fragments already consume 32 FP32 registers per PV thread.
The PV update peak also needs matrix-A/B fragments, HMMA rename registers,
two P addresses, 64-bit V addressing, row scale, and loop state. The QK role's
unused registers cannot be reassigned to PV threads because allocation is
uniform for the entire kernel. This explains why the isolated one-panel PV
candidate reaches 64 registers without spill while the complete multi-panel
pipeline does not.

The shared-epoch follow-up removed every hot-loop CTA barrier and used monotonic
P-ready/P-consumed counters. It reduced the forced stack from 480 to 96 bytes,
but a natural one-CTA build still required 84 registers/thread. Replacing the
counter protocol with empty synchronization stubs left the same 96-byte stack,
and separating QK/PV into noinline device functions left the PV callee spilling.
The protocol is therefore not the remaining cause: four persistent accumulators
plus the native PV update have a live set above the 64-register two-CTA limit.

The D128 fallback then launched two CTA/group, each computing M32 but only one
D128 output half. It reached 64 registers, zero stack/spill, 37,408 bytes shared,
and two CTAs/SM. Random and alternating inputs at one, two, and four KV blocks
were bitwise equal, and Compute Sanitizer reported zero errors. It nevertheless
regressed `134.912 -> 164.352 us` (`+21.82%`) with 0/100 wins. Repeating QK and
softmax in the two D-half CTAs costs more than the V-load reuse saves. This
structure is closed.

### Accepted complete phase-reuse microkernel

The passing structure keeps one CTA/group and removes next-block overlap. Eight
warps perform QK with one K fragment reused by top/bottom M16. All sixteen warps
then process matching top/bottom softmax rows and each owns one D16 with only
two persistent PV accumulators. One V fragment is reused by top/bottom M16.
This reduces the live state enough to keep the result in registers while also
removing the shared output buffer.

Formal locked-1200-MHz gates, `groups=144`, 100 alternating pairs:

| input | KV blocks | BM16 complete baseline | BM32 phase | speedup | exact |
|---|---:|---:|---:|---:|---|
| random | 1 | 77.952 us | 58.750 us | 24.63% | bitwise |
| random | 2 | 140.544 us | 92.288 us | 34.34% | bitwise |
| random | 4 | 264.576 us | 159.104 us | 39.86% | bitwise |
| alternating | 1 | 78.208 us | 58.880 us | 24.71% | bitwise |
| alternating | 2 | 140.544 us | 92.288 us | 34.34% | bitwise |
| alternating | 4 | 264.576 us | 159.104 us | 39.86% | bitwise |

The candidate is 64 registers/thread, zero stack/local/spill, 35,216 bytes
shared, two CTAs/SM, and has no SASS `LDL/STL`. All six cases win 100/100 pairs;
Compute Sanitizer reports zero errors.

Minimal NCU on the four-block case confirms the mechanism:

| metric | BM16 baseline | BM32 phase | delta |
|---|---:|---:|---:|
| profiled duration | 245.024 us | 148.416 us | -39.43% |
| L1 global-load requests | 889,344 | 446,976 | -49.74% |
| L1 global-load sectors | 7,135,619 | 3,603,288 | -49.50% |
| L2 read sectors | 4,873,828 | 2,466,835 | -49.39% |
| executed instructions | 19,261,440 | 16,625,664 | -13.68% |
| tensor instructions | 4,718,592 | 4,718,592 | unchanged |
| eligible warps/cycle | 0.479 | 0.763 | +59.36% |

The candidate performs the same tensor work while almost exactly halving the
K/V request and sector counts. This is the required bottleneck-directed
evidence. The microbaseline is serial BM16, so the 39.86% number is not yet a
claim against the accepted cross-block production kernel. Promotion now
requires a default-off page-784 integration and a direct same-process A/B
against that accepted path.

Do not retry direct per-lane vector B loads, source-level K/V-load reordering,
generic extra lookahead, or the 8+8 named-barrier schedule. They have already
failed their wall-time gates.

Additional artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/wmma_matrix_ab_compact_b_fragment_probe.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/compact_b_qk_ab100_m1024_n65536_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/compact_b_qk_weak_ab100_m1024_n65536_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/qk_b_load_shuffle_ab100_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/raw_hmma_full_16x32_order_probe.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/raw_hmma_qk_panel_candidates_ab100_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/raw_hmma_pv_panel_ab100_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/native_bm32_qk_reuse_ab100_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/native_bm32_pv_reuse_ab100_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_native_bm32_qk_baseline_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_native_bm32_qk_candidate_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_native_bm32_pv_baseline_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_native_bm32_pv_candidate_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/native_bm32_full_phase_aggregate_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_native_bm32_full_phase_baseline_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/ncu_native_bm32_full_phase_candidate_clock1200.ncu-rep`

### Production page-784 integration and promotion

The phase-reuse structure is now integrated as a dedicated paged-attention
kernel. It is intentionally narrow: SM70 FP16 KV, `D=256`, page size `784`,
`M >= 32`, `M % 32 == 0`, no BFLA mask, and no sliding window. Every other
shape falls through to the previous low-SMEM kernel. Set
`VLLM_FLASH_V100_PREFILL_D256_BM32_PHASE=0` to force the previous path for
rollback or A/B measurement.

The production kernel uses one 512-thread CTA per M32 group. Eight QK warps
reuse each K fragment for the top and bottom M16 panels; all 16 warps then own
one D16 slice and reuse each V fragment for the same two panels. QK warps
temporarily spill their persistent output fragments into the future score
tile, not local memory, while QK is live. This keeps the final kernel at 64
registers/thread, zero stack/local/spill, 35,408 bytes shared memory, and two
resident CTAs/SM. K0 and K16 use independent paged-cache pointers because a
784-token page can split a 32-token panel across noncontiguous physical pages.

Locked-1200-MHz direct A/B against the accepted BM16 cross-block Q/P-swizzled
kernel, `M=1024,Hq=6,Hkv=1,D=256`:

| KV length / layout | accepted kernel | BM32 phase | delta | paired wins | exact |
|---|---:|---:|---:|---:|---|
| 16K contiguous | 10.847 ms | 8.787 ms | -18.99% | 20/20 | bitwise |
| 32K contiguous | 21.790 ms | 17.440 ms | -19.96% | 20/20 | bitwise |
| 64K contiguous | 43.605 ms | 34.759 ms | -20.29% | 100/100 | bitwise |
| 64K reverse/tail (`N=65521`) | 43.600 ms | 34.755 ms | -20.29% | 100/100 | bitwise |
| 128K contiguous | 87.210 ms | 69.404 ms | -20.42% | 20/20 | bitwise |

Random and alternating micro inputs at one, two, and four KV panels are also
bitwise exact, and Compute Sanitizer reports zero errors. Minimal NCU shows the
same tensor-instruction count, approximately half the L1 global-load requests
and L2 read sectors, and `eligible warps/cycle` rising `0.479 -> 0.763`. The
speedup therefore comes from K/V operand reuse and a smaller instruction/load
stream; it does not change datatype, softmax, FP16 probability rounding, HMMA
count, or FP32 accumulation order.

The full-model propagation gate passes on Qwen3.6-27B-AWQ TP4, Flash-V100,
CUDA graph, no eager, no MTP, `max_model_len=65536`, prefill chunk 1024,
`input_len=65472`, and fixed 1200 MHz:

| metric | accepted Q/P path | BM32 phase | delta |
|---|---:|---:|---:|
| prefill | 44.8346 s | 40.4681 s | -9.74% |
| TTFT | 44.8832 s | 40.6789 s | -9.37% |
| total generation | 46.0961 s | 41.9270 s | -9.04% |
| no-MTP TPOT | 19.209 ms | 19.009 ms | -1.04% |

All 64 generated token IDs are identical under the same official sampling
parameters. The output token-vector SHA-256 is
`7ef979a01991cb9fe8d5276ff7e0db1ae489782674769293c7f096cfcc72d0e8`
for both runs. The phase route is consequently default-enabled only under the
strict dispatch conditions above; explicit `=0` remains the rollback.

Production artifacts:

- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/integrated_bm32_phase_ab100_m1024_n65536_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/integrated_bm32_phase_reverse_tail_ab100_m1024_n65521_clock1200.json`
- `bench_results/nomtp_long_context_20260713/current_tp4/q1_separate_sp/prefill_pipeline_20260714/integrated_bm32_phase_scaling_16k_128k_ab20_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_bm32_phase_64k_m1024_tp4_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_bm32_phase_vs_accepted_qp_64k_compare.json`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_phase_default_on_ab20_m1024_n65536_clock1200.json`

Further work is not allowed to start from another generic prefetch or staging
variant. It must first show a residual production-kernel bottleneck in NCU and
state how the proposed schedule preserves the 64-register/two-CTA gate. The
full four-accumulator 8+8 producer/consumer kernel and the D128 two-CTA split
are closed: the former spills above the 64-register budget, while the latter
repeats QK/softmax and regresses 21.82% despite reaching two CTAs/SM.

### Promoted-kernel NCU and all-P barrier elimination

NCU was repeated on the production BM32 phase kernel after promotion; the old
kernel's bottleneck attribution is no longer authoritative. At
`M1024,N65536,Hq6,Hkv1,D256,page784`, fixed 1200 MHz, the promoted kernel has:

| metric | production BM32 phase |
|---|---:|
| NCU duration | 31.871 ms |
| DRAM / L2 throughput | 0.56% / 12.26% |
| L1/TEX throughput | 86.95% |
| SM throughput | 25.37% |
| achieved occupancy | 45.03% |
| no eligible scheduler cycles | 62.96% |
| eligible warps/scheduler | 0.853 |
| barrier / short-SB / MIO / long-SB | 5.51 / 3.53 / 2.99 / 2.89 cycles per issue |

PC sampling attributes 27.2% of samples to barriers. Of the barrier samples,
65.0% are at the post-QK rendezvous where only warps 0-7 computed QK and warps
8-15 waited. The remaining repeated barrier clusters are the four P32 panels:
each panel first publishes online-softmax P, then prevents the shared P buffer
from being overwritten until all PV warps consume it.

SourceCounters also records 184.20M excessive shared wavefronts, 23% of
807.28M total shared wavefronts. The temporary PV-accumulator save/restore and
the final QK score stores use 8-way row-major mappings; the row-owned P stores
use 4-way mappings. These are separate optimization targets. A replacement
scratch layout can remove only the temporary accumulator save/restore portion;
it must not claim that final row-major QK score stores disappear.

The first response to this new profile is the all-P schedule, not another K/V
prefetch. It retains all four top/bottom P32 panels and all four row-exp-diff
vectors in shared memory, performs online softmax in its original order with a
warp visibility barrier between panels, then performs PV in its original
order. This removes shared-buffer-reuse CTA barriers without changing any
score, exponential, FP16 probability rounding, HMMA, or FP32 accumulator
order. A review found and fixed a block-index race by restoring a CTA barrier
before thread 0 advances the shared block state.

The race-fixed all-P microkernel is 64 registers/thread, zero stack/local/spill,
41,744 bytes shared, and two CTAs/SM. Incremental fixed-clock timing against the
accepted one-P phase microkernel is:

| input | KV blocks | one-P phase | all-P | incremental delta | exact |
|---|---:|---:|---:|---:|---|
| random | 1 | 58.750 us | 58.112 us | -1.09% | bitwise |
| random | 2 | 92.288 us | 89.984 us | -2.50% | bitwise |
| random | 4 | 159.104 us | 153.600 us | -3.46% | bitwise |
| alternating | 1 | 58.880 us | 58.112 us | -1.30% | bitwise |
| alternating | 2 | 92.288 us | 89.984 us | -2.50% | bitwise |
| alternating | 4 | 159.104 us | 153.472 us | -3.54% | bitwise |

All cases win 100/100 pairs. Same-metric NCU at four KV blocks reports
`148.77 -> 144.80 us` (`-2.67%`), barrier stall `6.39 -> 5.51` cycles/issue,
unchanged tensor instructions, and a 2.36% increase in total instructions.
MIO throttle rises `1.58 -> 2.18`, so retaining more P is accepted as a bounded
barrier trade, not presented as a complete shared-memory fix.

The production page-784 kernel passes direct A/B and edge gates:

| KV length / layout | one-P phase | all-P | delta | paired wins | exact |
|---|---:|---:|---:|---:|---|
| 16K contiguous | 8.777 ms | 8.438 ms | -3.86% | 20/20 | bitwise |
| 32K contiguous | 17.450 ms | 16.791 ms | -3.78% | 20/20 | bitwise |
| 64K contiguous | 34.755 ms | 33.482 ms | -3.66% | 100/100 | bitwise |
| 64K reverse/tail (`N=65521`) | 34.743 ms | 33.491 ms | -3.60% | 100/100 | bitwise |
| 128K contiguous | 69.379 ms | 66.843 ms | -3.66% | 20/20 | bitwise |

Compute Sanitizer reports zero errors on reverse-page-table `M32,N913`. The
Qwen3.6-27B-AWQ TP4 64K propagation gate changes prefill
`40.4681 -> 39.6948 s` (`-1.91%`), TTFT `40.6789 -> 39.8375 s` (`-2.07%`),
and total generation `41.9270 -> 41.0748 s` (`-2.03%`). All 64 output token
IDs match; no-MTP TPOT is unchanged within 0.01 ms. The all-P route is now
default-on inside the already strict BM32 phase dispatch. Set
`VLLM_FLASH_V100_PREFILL_D256_BM32_ALL_P=0` to recover the one-P phase path.

All-P artifacts:

- `bench_results/nomtp_long_context_20260715/native_bm32_full_phase_allp_racefixed_aggregate_clock1200.json`
- `bench_results/nomtp_long_context_20260715/ncu_native_bm32_phase_old_nblocks4_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/ncu_native_bm32_phase_allp_nblocks4_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_allp_ab100_m1024_n65536_clock1200.json`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_allp_reverse_tail_ab100_m1024_n65521_clock1200.json`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_allp_scaling_16k_128k_ab20_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_bm32_allp_64k_m1024_tp4_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_bm32_allp_vs_phase_64k_compare.json`

The next allowed microexperiment is the source-derived conflict-reduced
temporary accumulator scratch. It must reuse the existing score allocation,
preserve the four 64-bit stores and loads per fragment, remain at 64 registers
with no local instructions, and prove reduced SourceCounters wavefronts before
production integration. Cross-block QK/PV overlap remains P1 until this lower
risk shared bottleneck is closed.

### Pair-slab accumulator scratch promotion

The temporary-accumulator target is now closed and promoted. The old WMMA
row-major scratch maps each 64-bit store/load instruction onto only four bank
pairs and produces 8-way replay. The replacement groups QK warps into pairs.
Each pair owns a disjoint 32-column score slab; the two warps use disjoint rows
inside that slab, and lane `l` maps to column `2*(l % 16)`. This reaches the
minimum 2-way replay for a 32-lane 64-bit shared instruction without allocating
new shared memory. A 64-thread named barrier delays final row-major QK score
stores until both warps have reloaded their persistent PV accumulators. A tail
with only one active warp skips that pair barrier because no partner can
overwrite its scratch.

The standalone A/B changes no HMMA, LDG, LDS, or STS instruction count; the
candidate adds exactly one static `BAR` instruction. Both paths use 64
registers/thread, zero stack/local/spill, 41,744 bytes shared, and two CTAs/SM.
At fixed 1200 MHz, all random/alternating cases are packed-FP16 bitwise equal:

| input | KV blocks | row-major scratch | pair scratch | delta | paired wins |
|---|---:|---:|---:|---:|---:|
| random | 1 | 59.136 us | 57.600 us | -2.60% | 100/100 |
| random | 2 | 91.520 us | 88.960 us | -2.80% | 100/100 |
| random | 4 | 156.160 us | 151.932 us | -2.71% | 100/100 |
| alternating | 1 | 59.136 us | 57.600 us | -2.60% | 98/100 |
| alternating | 2 | 91.522 us | 88.960 us | -2.80% | 100/100 |
| alternating | 4 | 156.160 us | 151.936 us | -2.70% | 100/100 |

Standalone NCU at four blocks reduces shared bank conflicts
`1,227,501 -> 809,752` (`-34.0%`) and duration `148.13 -> 144.16 us`
(`-2.68%`). Barrier stall does not increase (`23.14% -> 22.87%`) despite the
pair handoff. Tensor instructions remain unchanged.

The production page-784 result is larger and remains stable with context:

| KV length / layout | all-P baseline | pair scratch | delta | exact |
|---|---:|---:|---:|---:|
| 16K contiguous | 8.440 ms | 7.999 ms | -5.22% | bitwise |
| 32K contiguous | 16.744 ms | 15.871 ms | -5.21% | bitwise |
| 64K contiguous | 33.411 ms | 31.717 ms | -5.07% | bitwise |
| 64K reverse tail, `N=65521` | 33.426 ms | 31.694 ms | -5.18% | bitwise |
| 64K reverse unpaired tail, `N=65505` | 33.435 ms | 31.691 ms | -5.22% | bitwise |
| 128K contiguous | 66.737 ms | 63.333 ms | -5.10% | bitwise |

Production NCU at `M1024,N65536,Hq6,Hkv1,D256,page784` records
`30.72 -> 29.15 ms` (`-5.11%`) and shared bank conflicts
`188.69M -> 117.94M` (`-37.50%`). Tensor instructions are exactly unchanged;
barrier stall falls `21.70% -> 20.68%`. Long- and short-scoreboard percentage
shares rise because the shared-replay portion became shorter; they are the
residual dependency target, not evidence that HBM became saturated.

Racecheck reports the same 16 conservative shared-score warnings on the old
and new kernels at the deliberately unpaired `M32,N97` tail, with zero errors.
The candidate adds no racecheck category or warning count. Normal, reverse,
full-pair, and unpaired-tail output comparisons are bitwise equal.

The Qwen3.6-27B-AWQ TP4 64K full-model gate uses Flash-V100 CUDA graphs, no
eager, no MTP, chunk size 1024, and the official sampling parameters:

| metric | all-P baseline | pair scratch | delta |
|---|---:|---:|---:|
| prefill | 39.6948 s | 38.7454 s | -2.39% |
| TTFT | 39.8375 s | 38.7997 s | -2.61% |
| total generation | 41.0748 s | 40.0010 s | -2.61% |
| no-MTP TPOT | 19.001 ms | 18.993 ms | neutral |

All 64 sampled output token IDs match. The cumulative 64K full-model prefill
gain from the 47.9785 s early-store reference is now 19.24%, still below the
30% project target. The pair scratch is default-on inside the existing strict
BM32 all-P dispatch. Set
`VLLM_FLASH_V100_PREFILL_D256_BM32_PAIR_SCRATCH=0` to restore row-major
scratch.

Artifacts:

- `bench_results/nomtp_long_context_20260715/native_bm32_allp_pair_scratch_aggregate_clock1200.json`
- `bench_results/nomtp_long_context_20260715/ncu_native_bm32_allp_pair_scratch_baseline_nblocks4_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/ncu_native_bm32_allp_pair_scratch_candidate_nblocks4_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_pair_scratch_ab100_m1024_n65536_clock1200.json`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_pair_scratch_reverse_tail_ab40_m1024_clock1200.json`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_pair_scratch_scaling_16k_128k_ab20_clock1200.json`
- `bench_results/nomtp_long_context_20260715/ncu_integrated_bm32_pair_scratch_baseline_m1024_n65536_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/ncu_integrated_bm32_pair_scratch_candidate_m1024_n65536_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/prefill_bm32_pair_scratch_64k_m1024_tp4_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_bm32_pair_scratch_vs_allp_64k_compare.json`
- `bench_results/nomtp_long_context_20260715/integrated_bm32_pair_scratch_default_on_m1024_n65536_clock1200.json`

The next allowed software-pipeline experiment is narrower than the rejected
8-producer/8-consumer shared staging. It must keep direct global-to-WMMA
fragments and all 16 PV accumulators: after P for block `n` is complete,
warps 0-7 may compute QK for `n+1` while warps 8-15 compute the upper D128 of
PV for `n`, then warps 0-7 finish the lower D128. A prologue/steady/drain
microkernel must remain at 64 registers, zero local spill, and two CTAs/SM;
both wall time and matched HMMA long-scoreboard must improve. Do not retry K/V
shared double buffering, persistent eight-consumer PV, or another pitch-only
layout.

### Direct-fragment cross-block schedule rejection

The proposed prologue/steady/drain schedule has now been isolated and rejected
before production integration. With the production PTXAS optimization level,
the baseline remains at 64 registers/thread, zero stack/spill, 41,744 bytes
shared, and two CTAs/SM. The cross-block candidate still reports 64 registers,
but allocates a 24-byte stack frame and emits 20-byte spill stores plus 20-byte
spill loads per thread. It therefore fails the resource gate before timing.

A diagnostic-only `ptxas -O1` build removes those spills from both variants and
keeps packed-FP16 output bitwise exact for random and alternating inputs. Even
under that non-production concession, the schedule loses consistently:

| input | KV blocks | baseline | cross-block | delta | candidate wins |
|---|---:|---:|---:|---:|---:|
| random | 2 | 90.368 us | 91.136 us | +0.85% | 1/100 |
| random | 4 | 153.984 us | 155.136 us | +0.75% | 2/100 |
| alternating | 2 | 90.368 us | 91.136 us | +0.85% | 2/100 |
| alternating | 4 | 153.984 us | 155.136 us | +0.75% | 2/100 |

NCU was intentionally skipped because the wall-time gate failed. This closes
the direct-fragment D128 split in its current form: moving half of PV to eight
warps does not create useful overlap on SM70, and the longer live ranges cause
production spills. Do not tune its barriers or integrate it into the paged
kernel.

Artifacts:

- `bench_results/nomtp_long_context_20260715/native_bm32_allp_crossblock_default_ptxas_smoke_clock1200.json`
- `bench_results/nomtp_long_context_20260715/native_bm32_allp_crossblock_diagnostic_ptxas_o1_ab100_clock1200.json`

The next implementation is not selected yet. First collect SourceCounters and
PC sampling on the promoted pair-scratch production kernel, then attribute the
remaining 117.94M shared conflicts and HMMA scoreboard samples to exact source
operations. A new pipeline is allowed only if that profile identifies an
independent producer/consumer window without increasing the 64-register live
set or adding full-CTA barriers.

### Pair-scratch residual PC attribution

The required production SourceCounters profile is complete for
`M1024,N65536,Hq6,Hkv1,D256,page784` and template `<causal=1,allP=1,
pairScratch=1>`. SourceCounters records 113.42M excessive shared wavefronts,
down 38.43% from the 184.20M pre-pair profile. The remaining accesses are now
fully classified:

| source operation | excessive wavefronts | share | interpretation |
|---|---:|---:|---|
| 8 final QK `STS.64` | 37,748,736 | 33.28% | row-major FP32 score stores, 8-way |
| 16 probability `STS` | 75,497,472 | 66.57% | four all-P panels, 4-way |
| query staging `STS.128` | 172,032 | 0.15% | one-time staging, 32-way but negligible |

K/V global loads are not the source of the residual shared conflicts. A layout
change that touches only temporary accumulator scratch is therefore exhausted;
the next shared target must change final QK or probability production.

PC sampling records 193,427 barrier samples. Their dominant locations are:

| rendezvous | samples | share |
|---|---:|---:|
| QK completion before score scaling | 150,111 | 77.61% |
| page-pointer publication before QK | 13,809 | 7.14% |
| PV completion before next block | 13,301 | 6.88% |
| all-P publication before PV | 8,465 | 4.38% |
| pair-scratch named barrier | 3,527 | 1.82% |
| block-index handoff | 1,261 | 0.65% |

The pair handoff is not the new bottleneck. The largest idle window is the
existing QK full-CTA barrier: only warps 0-7 calculate both top and bottom QK
tiles while warps 8-15 wait. HMMA long-scoreboard attribution supports the
same priority. Of 218,014 HMMA long-scoreboard samples, QK contributes 66,876
(`30.68%`) and PV contributes 151,138 (`69.32%`); another 3,999 non-HMMA
samples are only 1.80% of the total long-scoreboard set.

This authorizes one exact microbenchmark, not production integration. It pairs
all 16 warps by N16 tile so one warp computes top QK and the other bottom QK.
Each pair must load K only once. Two 4KB K stages may alias the all-P
`probability_top` and `probability_bottom` storage before softmax produces P,
so a double-buffered software pipeline adds no shared footprint. The candidate
must preserve the 41,936-byte allocation, 64-register/two-CTA limit, FP32 QK
bits, and persistent PV accumulator order. It must reduce the QK rendezvous
through useful work, not merely replace one full-CTA barrier with many equally
expensive pair barriers.

Artifact:

- `bench_results/nomtp_long_context_20260715/ncu_integrated_bm32_pair_scratch_source_m1024_n65536_clock1200.ncu-rep`

### Sixteen-warp paired-QK rejection

The profile-authorized 16-warp paired-QK sketch is closed at the first gate.
It differs from the earlier dedicated 8-producer/8-consumer staging attempt:
the producer warp also computes top QK, its pair computes bottom QK, and all 16
warps execute HMMA while each pair performs only one global K load. The two K
stages fit by aliasing the 8KB all-P probability allocation, so the shared
footprint stays at 41,744 bytes in the faithful micro.

Production `ptxas -O3` nevertheless emits local state before any execution:

| path | registers | shared | stack | spill store/load |
|---|---:|---:|---:|---:|
| pair-scratch baseline | 64 | 41,744 B | 0 B | 0 B / 0 B |
| 16-warp paired QK | 64 | 41,744 B | 8 B | 12 B / 8 B |

The candidate therefore fails the zero-local-memory/two-CTA gate. Exactness,
formal timing, and NCU were intentionally skipped.

Review also found a structural lifetime failure in the proposed 16KB QK
result scratch. The query tile is reused for every KV block, so it is not dead
after one QK phase. Aliasing QK results onto `shared.query` would require
restaging 16KB of Q and another full-CTA publication barrier for every
128-token block. This invalidates the intended overlap even if the small PTXAS
spill were later removed. Do not continue this form by forcing register caps,
using `ptxas -O1`, or adding per-block Q restaging.

Artifact:

- `bench_results/nomtp_long_context_20260715/native_bm32_qk_dualwarp_pipeline_ptxas_rejection_clock1200.json`

The next measured target is the 75.50M probability-store excessive wavefronts
(`66.57%` of the residual shared replay). A candidate must change the four
8-column groups' bank mapping without changing the 8KB probability capacity,
FP16 rounding point, or two-load WMMA matrix-A reconstruction. This is a
layout gate, not another K/V staging attempt.

### Probability group-rotation rejection

The same-capacity P group-rotation layout is closed at the production PTXAS
gate. It rotates the 16-row slot independently for each 8-column group, which
is a valid static bank mapping: the eight active half2 stores map from four-way
same-bank groups to distinct banks, while each PV matrix-A fragment still uses
two shared vector loads. It also leaves HMMA, global loads, shared capacity,
and all-P ordering unchanged.

The additional group-dependent address construction is not free at the
current 64-register boundary:

| path | registers | shared | stack | spill store/load | local SASS |
|---|---:|---:|---:|---:|---:|
| pair-scratch baseline | 64 | 41,744 B | 0 B | 0 B / 0 B | 0 / 0 |
| P group rotation | 64 | 41,744 B | 8 B | 8 B / 8 B | 1 LDL / 1 STL |

An algebraically reduced XOR/template form was also checked only at compile
time and made the spill larger; it is not a separate candidate. The harness
now rejects PTXAS/SASS resource failure before any kernel launch. Exactness,
formal timing, and NCU were therefore not run. Do not retry per-group row-slot
rotation under another spelling or diagnostic compiler flag.

Artifact:

- `bench_results/nomtp_long_context_20260715/native_bm32_allp_p_group_skew_smoke_clock1200.json`

One distinct P layout remains eligible: a 136-half physical pitch for each
16x8 group. The 272-byte group stride shifts each group by four banks and lets
the second PV vector-load address derive from the first with one constant add.
It costs 24 half per panel, or 384 bytes across the eight all-P top/bottom
panels. This is allowed only if 42,128 bytes still yields two CTAs/SM, zero
local traffic, exact output, and a formal wall-time win.

### Probability 136-half group-pad rejection

The padded layout passes every resource and exactness gate. Candidate shared
memory is 42,128 bytes versus 41,744 bytes, while both paths retain 64
registers/thread, zero stack/spill/local, two CTAs/SM, and identical HMMA, LDG,
LDS, STS, and BAR instruction counts. Random and alternating outputs are
packed-uint32 bitwise equal at one, two, and four KV blocks.

The bank fix is real but too small at wall time:

| input | KV blocks | baseline | 136-half pad | speedup | candidate wins |
|---|---:|---:|---:|---:|---:|
| random | 1 | 57.600 us | 57.216 us | 0.67% | 91/100 |
| random | 2 | 89.088 us | 88.064 us | 1.15% | 97/100 |
| random | 4 | 151.808 us | 149.632 us | 1.43% | 100/100 |
| alternating | 1 | 57.600 us | 57.216 us | 0.67% | 91/100 |
| alternating | 2 | 88.960 us | 87.936 us | 1.15% | 99/100 |
| alternating | 4 | 151.808 us | 149.632 us | 1.43% | 100/100 |

It fails the mandatory 2% all-case gate, so NCU and production integration
were skipped. This closes probability-only bank-layout work; do not combine
several sub-gate layouts merely to cross the threshold without an independent
dominant-stall target.

Artifacts:

- `bench_results/nomtp_long_context_20260715/native_bm32_allp_p_group_pad_smoke_clock1200.json`
- `bench_results/nomtp_long_context_20260715/native_bm32_allp_p_group_pad_ab100_clock1200.json`

The next dominant target returns to PV HMMA operand readiness. Production SASS
shows that the second K16 V fragment in each P32 panel is scheduled immediately
before its first HMMA step, leaving only one instruction of dependency
distance; the corresponding PCs carry 17K-24K long-scoreboard samples each.
The first allowed experiment is a zero-SASS compiler fence that keeps this V
load ahead of the existing independent P shared loads. It may not allocate a
second B fragment, add prefetch traffic, or change any synchronization.

### PV second-K16 compiler-fence rejection

The zero-instruction compiler fence is closed. The candidate adds only
`asm volatile("" ::: "memory")` after loading the second K16 V fragment. Both
paths compile to 64 registers/thread, 41,744 bytes of shared memory, zero
stack/spill/local traffic, and two CTAs/SM. Their static instruction counts
are also identical: 768 HMMA, 74 LDG, 160 LDS, and 6 BAR per CTA. Random and
alternating outputs are packed-uint32 bitwise equal in every case.

At a locked 1200 MHz, 100 alternating A/B pairs show no measurable scheduling
change:

| input | KV blocks | baseline | compiler fence | speedup |
|---|---:|---:|---:|---:|
| random | 1 | 54.144 us | 54.208 us | -0.118% |
| random | 2 | 83.456 us | 83.328 us | +0.153% |
| random | 4 | 128.128 us | 128.320 us | -0.150% |
| alternating | 1 | 48.896 us | 48.896 us | 0.000% |
| alternating | 2 | 75.520 us | 75.520 us | 0.000% |
| alternating | 4 | 127.616 us | 127.616 us | 0.000% |

The pair wins and losses are mixed in every case. Source-level ordering is
therefore insufficient; any follow-up must prove that the emitted SASS order
actually changed before timing it.

Artifact:

- `bench_results/nomtp_long_context_20260715/native_bm32_allp_pv_load_fence_ab100_clock1200.json`

### FlashAttention-2-style M64 rejection

An exact M64/BN128 complete-phase micro was compared against two accepted
BM32 `ALL_P=true, PAIR_SCRATCH=true` CTAs for the same 64 query rows. The M64
CTA uses all 16 warps for QK, keeps four M16 output panels per PV warp, and
reuses each V fragment across all four panels. Q remains persistent across KV
blocks; the first proof deliberately keeps P separate instead of relying on
an unsafe Q/P alias.

The implementation passes the hard gates. BM32 uses 64 registers/thread,
41,744 bytes of static shared memory, zero stack/spill/local traffic, and two
CTAs/SM. M64 uses 128 registers/thread, 83,472 bytes of dynamic shared memory,
zero stack/spill/local traffic, and one CTA/SM. All random and alternating
outputs through 64 KV blocks are packed-uint32 bitwise equal.

The fixed-frequency wall-time result is uniformly negative:

| input | KV blocks | 2x BM32 | M64 | speedup | M64 wins |
|---|---:|---:|---:|---:|---:|
| random | 1 | 55.808 us | 55.936 us | -0.229% | 30/100 |
| alternating | 1 | 55.808 us | 55.936 us | -0.229% | 39/100 |
| random | 2 | 85.760 us | 87.296 us | -1.791% | 2/100 |
| alternating | 2 | 85.760 us | 87.168 us | -1.642% | 0/100 |
| random | 4 | 146.944 us | 149.760 us | -1.916% | 0/100 |
| alternating | 4 | 146.944 us | 149.632 us | -1.829% | 0/100 |
| random | 64 | 1,976.576 us | 2,016.896 us | -2.040% | 0/100 |
| alternating | 64 | 1,976.000 us | 2,015.488 us | -1.998% | 0/100 |

NCU at 64 KV blocks explains the regression. M64 reduces L1TEX global-load
sectors from 56.36M to 47.16M (`-16.3%`) and barrier stall from 24.82% to
8.12%, while tensor instructions stay identical at 75.50M. However, active
warps per scheduler fall from 7.98 to 4.00, eligible warps fall from 0.905 to
0.514, and long-scoreboard stall rises from 14.61% to 24.78%. NCU duration
therefore increases from 1.828 ms to 1.874 ms (`+2.49%`). The extra V reuse
does not repay the loss of the second resident CTA and half of the latency
hiding.

Do not integrate or retry this 512-thread, one-CTA M64 form with only shared
layout spelling changes. A distinct M64 continuation must first restore more
than 16 resident warps without increasing V loads back to the BM32 level.

Artifacts:

- `bench_results/nomtp_long_context_20260715/native_bm64_allp_ab100_clock1200.json`
- `bench_results/nomtp_long_context_20260715/ncu_native_bm64_bm32_baseline_nblocks64_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/ncu_native_bm64_candidate_nblocks64_clock1200.ncu-rep`

### Native HMMA.884 SASS schedule rejection

The SASS-level feasibility probe succeeds technically but fails the wall-time
gate. A native Volta `m16n16k16` WMMA operation expands into 16 `HMMA.884`
instructions: four operand groups, each with `STEP0..3`. In the isolated
two-WMMA chain, B0 register `R10` reaches its last B-operand use at `0x02a0`.
The proven native-cubin candidate swaps complete 16-byte instruction bundles
to move the next B1 `LDG.E.64.SYS R10, [R30+0x200]` from `0x0350` to `0x0320`,
between group 2 `STEP1` and `STEP2`; the independent A1 load moves to
`0x0350`. `cuobjdump` confirms that the actual SASS order changed.

The candidate retains one B fragment, 38 registers/thread, zero shared/local
memory, 128 threads/CTA, and identical static counts of 32 HMMA and 12 LDG.
All 4,194,304 FP32 output words are bitwise equal. Fixed-frequency A/B does
not show a material gain:

| native bundle candidate | baseline | candidate | speedup | pair result |
|---|---:|---:|---:|---:|
| preserve moved LDG `wait=0x20` | 68.608 us | 68.480 us | +0.187% | 54 / 36 / 10 |
| clear moved LDG wait mask | 68.608 us | 68.608 us | 0.000% | 42 / 47 / 11 |

Each row uses 16,384 groups, 100 alternating rounds, eight launches/sample,
and strict 1200 MHz checks before and after execution. The sub-percent result
is below the 2% isolated gate and is not eligible for production-kernel work.

TuringAs revision `f7c1a74ebe6bc2ac51920b0fee29e27767f81724` successfully
assembles the isolated LDG/HMMA core, and its instruction bits match the
native cubin. A strict full native round trip is not supported because its
predicate parser rejects the native constant-false `@!PT SHFL.IDX` form.
Removing that instruction and supplying missing arguments produces a
loadable cubin, but the result loses native WMMA/thread metadata and changes
the resource contract to 39 registers and 1024 max threads. It is therefore
not a resource-equivalent performance candidate. The accepted proof mechanism
for this experiment is the native 16-byte bundle patch, not a TuringAs full
round trip.

Do not patch this single B1 load into the production kernel. A future SASS
experiment must target several independently verified hot load/HMMA windows
and first demonstrate an aggregate gain above 2% in the isolated micro.

Artifacts:

- `bench_results/nomtp_long_context_20260715/hmma884_native_bundle_swap_wait20_ab100_clock1200.json`
- `bench_results/nomtp_long_context_20260715/hmma884_native_bundle_swap_wait0_ab100_clock1200.json`

### Current production roof and CTA-wave quantization

The promoted `BM32 + ALL_P + PAIR_SCRATCH` kernel was re-profiled on the
acceptance shape `M1024,N65536,Hq6,Hkv1,D256,page784`. The complete SOL report
and the existing source-attribution report agree that this is neither an HBM
bandwidth ceiling nor a tensor-compute ceiling:

| metric | current production kernel |
|---|---:|
| registers / static shared | 64/thread / 41.94 KB/CTA |
| theoretical / achieved occupancy | 50.00% / 47.19% |
| SM compute throughput | 26.74% |
| tensor active, elapsed / SM-active cycles | 14.62% / 21.52% |
| L1/TEX throughput | 92.25% |
| L1 / L2 hit rate | 33.65% / 99.29% |
| L2 / DRAM throughput | 21.40% / 0.47% |
| eligible warps / scheduler | 0.74 |
| cycles with no eligible warp | 60.92% |
| issue slots busy | 39.36% |

`L1/TEX=92.25%` is request/shared-wavefront pressure, not HBM traffic. The
remaining per-CTA latency is a mix of HMMA operand readiness, shared
dependencies, and barriers. Existing PC sampling attributes HMMA
long-scoreboard samples `69.32%` to PV and `30.68%` to QK; `77.61%` of barrier
samples remain at QK completion. This explains why tensor cores are underfed
despite the high L1/TEX number.

The complete report also exposed a separate launch-level bottleneck. There
are 192 CTAs, while 72 SMs at two resident CTAs/SM accept 144 CTAs per wave.
The launch is therefore one full wave plus a 48-CTA tail. A same-process M
sweep under identical GPU conditions proves that the tail costs almost a
complete wave:

| M | CTAs | wave shape | median | effective tensor work |
|---:|---:|---:|---:|---:|
| 512 | 96 | partial one-wave launch | 16.041 ms | 12.80 TF/s |
| 768 | 144 | one full wave | 16.029 ms | 19.18 TF/s |
| 1024 | 192 | 144 + 48 tail | 31.964 ms | 12.80 TF/s |
| 1152 | 216 | 144 + 72 tail | 31.993 ms | 14.37 TF/s |
| 1280 | 240 | 144 + 96 tail | 32.062 ms | 15.92 TF/s |
| 1536 | 288 | two full waves | 31.978 ms | 19.11 TF/s |

The current `M=1024` shape consequently loses about one third of attainable
throughput to CTA-wave quantization before any further per-CTA tuning. This is
larger than the remaining isolated bank-layout opportunities.

The old split-KV switch is not a solution as implemented. It instantiates the
legacy BM16 generic partial kernel rather than the promoted BM32 phase kernel.
At `M1024,N65536`, the current unsplit BM32 path is `31.964 ms`; legacy
split-KV is `56.294 ms` with two partitions and `51.901 ms` with four. These
measurements reject enabling the old switch, not KV partitioning itself.

The next authorized microimplementation is a **BM32-native three-way
split-KV** path. Three partitions produce `192 * 3 = 576` partial CTAs, exactly
four full 144-CTA waves. With each CTA processing approximately one third of
N, the measured full-wave rate predicts about `21.3 ms` for partial attention
plus merge overhead, versus `31.964 ms` now. The acceptance gate is:

1. reuse the exact promoted BM32 QK/softmax/PV body and its 64-register,
   41.94-KB, two-CTA resource contract;
2. preallocate graph-safe partial output/max/sum workspaces;
3. use three 128-token-aligned KV partitions for this 192-CTA shape;
4. reach `<=24.0 ms` combined partial plus merge (`>=25%` wall-time gain);
5. report merge-order numerical drift and pass token/logprob quality gates
   before any default-on integration.

Artifacts:

- `bench_results/nomtp_long_context_20260715/ncu_integrated_bm32_pair_scratch_full_m1024_n65536_clock1200.ncu-rep`
- `bench_results/nomtp_long_context_20260715/bm32_pair_scratch_wave_sweep_m512_1536_n65536_clock1200.json`
- `bench_results/nomtp_long_context_20260715/splitkv_current_m1024_n65536_split32768_clock1200.json`
- `bench_results/nomtp_long_context_20260715/splitkv_current_m1024_n65536_split16384_clock1200.json`

### Tensor-peak calibration for the installed PG503 cards

The installed `Tesla PG503-216` is a 72-SM GV100 with a reported maximum SM
clock of 1530 MHz. It is not the 80-SM configuration behind the commonly
quoted 125.3-TF/s V100 number. Its unthrottled FP16 Tensor Core ceiling is:

`72 SM * 8 TC/SM * 64 FMA/TC/cycle * 2 FLOP/FMA * 1.53 GHz = 112.80 TF/s`.

Large square FP16 GEMMs through PyTorch/cuBLAS reached a best `83.45 TF/s` at
`N=8192`. NCU confirms that this is a Volta
`tensorop_f16_s884gemm` kernel, not a CUDA-core fallback: Tensor-pipe active is
90.04% and SM compute throughput is 88.42%. Under sustained GEMM, however, the
requested 1530-MHz clock falls to 1260-1275 MHz at 289-299 W with the 300-W
software power cap active. The clock-adjusted Tensor ceiling is therefore
about 93.4 TF/s, and the measured 83-84 TF/s is approximately 89-90% of the
practical ceiling. This hardware cannot sustain 125 TF/s.

The prefill benchmark's `effective_tflops` already includes Tensor Core work.
It counts QK and PV, each as one multiply-add, using
`4 * M * effective_keys * Hq * D`. For `M1024,N65536,Hq6,D256`, the kernel
executes 805,306,368 HMMA.884 warp instructions. At 512 FLOP per instruction,
that is exactly 412,316,860,416 FLOP, equal to the unmasked
`4 * M * N * Hq * D` count. The reported effective count is 0.78% smaller only
because causally masked work is excluded. Consequently, current prefill at
12.8 TF/s uses roughly 15% of the measured sustainable Tensor-GEMM ceiling;
the full-wave M768 case at 19.18 TF/s uses roughly 23%.

Artifact:

- `bench_results/nomtp_long_context_20260715/ncu_v100_cublas_fp16_gemm_n8192_clock1530.ncu-rep`

### Instantaneous-utilization measurement contract

All operator TF/s values in this document are per GPU and per TP rank. The
`M768,N65536,Hq6,Hkv1` result of 19.18 TF/s is one V100's useful attention
rate. Four perfectly aligned TP4 ranks would sum to 76.7 TF/s, but aggregate
TP4 throughput must be measured from aligned rank timelines rather than
inferred by multiplying one-rank averages.

`benchmark_flash_v100_prefill_decay.py` now retains every timed launch in
`samples_ms` and derives per-launch effective-TF/s samples plus min, p50, p90,
p99, and max. Console summaries show `launch[min/p50/p90/max]`. Mean or median
alone is no longer sufficient promotion evidence.

A 100-launch sample on `N65536` demonstrates why. `M768` is narrow at roughly
22.01-24.44 TF/s, but `M1024` is bimodal: 15 launches are about 20.3 ms
(`~20.1 TF/s`) while most are about 25.1 ms (`~16.3 TF/s`). Post-launch NVML
sampling shows 1530 MHz and no throttle for both clusters; the faster cluster
draws more power. This is not explained by average clock or thermal throttling
and must remain visible while CTA-tail scheduling and wrapper allocation are
separated.

SM70 tool limits must be stated explicitly. Nsight Compute PM Sampling starts
at CC 7.5, and Nsight Systems reports the installed SM70 devices as an
unsupported architecture for GPU Metrics. Therefore the accepted observation
stack is:

1. raw CUDA-event samples for launch-to-launch instantaneous latency/TF/s;
2. synchronized per-rank NVTX/CUDA timelines for TP4 aggregation;
3. NCU aggregate counters plus PC Sampling for hot SASS/stall attribution;
4. profile-only device `clock64()` phase stamps when QK, softmax, PV, and
   rendezvous timing inside one kernel is required.

Do not describe an NCU whole-kernel average as an instantaneous Tensor, SM, or
memory-utilization curve on V100.

Artifact:

- `bench_results/nomtp_long_context_20260715/prefill_launch_instantaneous_m768_m1024_n65536.json`
