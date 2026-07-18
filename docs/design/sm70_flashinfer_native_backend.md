# Native FlashInfer-Shaped Backend for SM70

## Purpose

This is the control document for the independent `FLASHINFER_SM70` attention
backend.  The route ports FlashInfer's planner, work decomposition, state
ownership, and short dependency graph to Volta.  It does not enable upstream
SM75+ kernels by weakening an architecture check, and it does not replace the
accepted `FLASH_ATTN_V100` baseline until every promotion gate passes.

The upstream reference is FlashInfer `v0.6.13`, commit
`57ba7eeb7ea3003a2d6ad5d9a057c4f952709bac`.  The measured reference kernel is
the RTX 5090 `BM64/BN64`, four-warp, causal D256 prefill kernel recorded in:

- `bench_results/prefill_wave_chunk_20260715/ncu_flashinfer_prefill_q8096_kv121440.ncu-rep`

## Route Isolation

`AttentionBackendEnum.FLASHINFER_SM70` is an explicit-only backend.  It has a
separate backend, metadata-builder, and implementation class.  The validated
single-request prefix-prefill shape now uses a fixed SM70 entry that directly
launches the accepted causal BM32 `ALL_P + PAIR_SCRATCH` specialization.
Unsupported, graph-captured, decode, multi-request, and split-KV shapes still
delegate explicitly to the accepted SM70 implementation and cannot be counted
as `FLASHINFER_SM70` kernel gains.

The accepted `FLASH_ATTN_V100` priority and default behavior must not change
during development.  A failed candidate is removed from the new route or kept
behind an explicit diagnostic switch; it is never allowed to silently fall
through and be reported as a FlashInfer-SM70 result.

## Locked Comparison

The first endpoint is Qwen3.6-27B-AWQ, TP4, TurboMind AWQ, FP16 KV cache,
page size 784, CUDA graphs, no eager, no MTP, one request, official sampling,
and `max_num_batched_tokens=8096`.

| metric | V100 accepted route | RTX 5090 reference |
|---|---:|---:|
| 64K prefill | 33.0984 s | 21.518 s |
| prefill throughput | 1,978.1 tok/s | 3,045.7 tok/s |
| TPOT | 19.114 ms | 14.366 ms |
| decode throughput | 52.32 tok/s | 69.61 tok/s |

The exact prefill operator comparison is `M=8096`, `N=121440`, causal D256.
V100 runs one TP rank (`Hq=6,Hkv=1`); the 5090 reference runs the full head set
(`Hq=24,Hkv=4`).

| metric | V100 BM32 | RTX 5090 FlashInfer |
|---|---:|---:|
| wall | 325.886 ms | 144.269 ms |
| useful rate | 17.92 TF/s | 161.90 TF/s |
| threads/CTA | 512 | 128 |
| registers/thread | 64 | 254 |
| resident CTA/SM | 2 | 1 |
| tensor active | 20.97% | 35.20% |
| eligible warps/scheduler | 0.738 | 0.123 |
| no-eligible cycles | 60.86% | 87.66% |
| long-scoreboard cycles/issue | 4.684 | 0.096 |
| barrier cycles/issue | 4.393 | 0.089 |
| MIO-throttle cycles/issue | 1.982 | 0.105 |

The target is not to copy the 5090 occupancy number.  The target is to move
SM70 dependency stalls toward the reference: useful tensor issue must rise
while HMMA operand wait, full-CTA barriers, and shared-memory replay fall.

## Upstream Structure to Preserve

The following FlashInfer mechanisms are architecture-independent and should
be ported rather than reinvented:

1. GQA rows are packed as `(query token, query head within KV group)`.
2. Each query warp owns its online-softmax state and output for the complete
   KV traversal.
3. The planner derives CTA tiles and split-KV work from request lengths and an
   SM-count work budget.
4. Split-K temporary state and planning metadata are preallocated and safe for
   CUDA-graph replay.
5. K and V reuse one repack buffer at different pipeline phases.
6. QK logits are transformed in place into probabilities instead of keeping a
   second FP32 probability matrix.
7. Prologue, steady-state, and drain are separate compile-time schedules so a
   runtime tail predicate does not extend hot-loop live ranges.

## SM70 Compatibility Layer

The modern implementation must remain untouched.  SM70 receives explicit
specializations for the following operations:

| FlashInfer primitive | Modern path | SM70 optimal replacement |
|---|---|---|
| prefill operand feed | `cp.async` / `LDGSTS` | direct global WMMA-B load plus top/bottom M16 operand reuse; a bounded 512-byte stage is allowed only if SASS and wall time beat direct load |
| decode operand feed | `cp.async` / `LDGSTS` | early `LDG.E.128` into a four-GPR payload, current-panel compute, then delayed `STS.128` into the alternate stage |
| matrix shared load | `ldmatrix` | conflict-reduced `ld.shared.v4.u32` that constructs the exact Volta fragment lane map |
| matrix multiply | `mma.m16n8k16` | native Volta WMMA/HMMA.884 `m16n16k16` with one K/V fragment reused across two M16 query panels |
| async groups | commit/wait groups | prologue/steady/drain software schedule and participant-counted named barriers; full-CTA barriers only for true CTA-wide lifetime handoff |
| TMA/persistent scheduling | hardware descriptors | FlashInfer host planner, precomputed page-pointer metadata, graph-stable workspace, and SM-count partition scheduling |
| warp-group register control | `setmaxnreg`/warp specialization | tile and role decomposition that satisfies Volta's uniform per-thread allocation naturally; forced register caps are forbidden |
| K/V FP8 repack | native FP8 path | vector FP8 load and fused FP16 conversion immediately before the accepted FP16 WMMA path, promoted only after FP16 arithmetic parity |

These replacements are workload-specific.  In particular, a generic
`uint4 -> shared -> barrier` emulation of `cp.async` is not accepted as the
prefill solution: previous coarse K/V staging increased wall time.  Decode has
a different measured dependency window and may use early-load/delayed-store.
No helper may emulate an unsupported instruction by adding an unbounded
register live range.  PTXAS and SASS gates are part of compilation, not an
after-the-fact profiler check.

## Register and Shared-Memory Budget

The measured SM120 kernel uses 254 registers/thread with no `LDL/STL`.  Its
source-level persistent state is approximately:

| state | logical 32-bit registers/thread |
|---|---:|
| D256 FP32 output accumulator | 128 |
| QK score fragment | 32 |
| online-softmax max/sum | 4 |
| page/KV offsets | about 8 |
| persistent lower bound | about 172 |

The modern path then fits operand fragments, addresses, scales, and control in
the remaining registers.  A literal SM70 translation does not fit: Volta's
operand fragments require about eight more transient registers, and the
synchronous 128-bit copy needs payload registers that `cp.async` avoids.

Consequently, `natural register demand >255` is a hard rejection for the
literal four-warp translation.  The earlier estimate of roughly 300 registers
is not treated as a measured value; spill bytes cannot recover the compiler's
unbounded natural allocation.

The first viable SM70 structure uses eight warps, with two D128 owners for each
M16 query panel.  This cuts persistent output state to 64 FP32 registers per
warp while retaining one logical owner group across the KV traversal.  During
QK, the four partner warps are not idle: they cooperatively prefetch one V16
tile into bounded vector payloads.  After the QK/K-stage handoff they publish
that V tile into the same 8KB shared stage and all eight warps execute PV.
This is a Volta-specific mapping of FlashInfer ownership, not a return to the
old 512-thread phase kernel.

Shared-memory gates on V100 are:

- at most 49,152 bytes/CTA when two resident CTAs are required;
- at most 98,304 bytes/CTA for a one-CTA diagnostic candidate;
- zero local-memory load/store instructions;
- no forced `maxrregcount` and no performance result from a diagnostic PTXAS
  optimization level.

## Prefill P0 Structural Trial

The first executable kernel is a true paged `BM64/BN32-or-64`, D256 micro with
the same GQA packing as FlashInfer.  The planned SM70 mapping is:

1. Eight warps, grouped as two warps per M16 query panel.
2. Each warp keeps one D128 half of output in registers: 64 logical FP32 GPRs.
3. One warp in each pair computes QK and publishes rounded FP16 probability;
   the partner does not repeat QK or softmax.
4. A K16-by-D256 tile is loaded once into an 8KB shared stage and reused by
   four QK warps.  This replaces the modern `ldmatrix` path with the proven
   conflict-reduced Volta WMMA-B mapping.
5. While QK consumes K, the four partner warps divide the V tile's global
   vectors and issue early loads into bounded payload registers.  After K is
   dead they publish V into the same stage, avoiding a second 8KB allocation.
6. Q uses 32KB shared memory and P uses about 2KB for one physical N16 tile;
   row state and alignment must keep the total below 48KB.
7. The compiler gate is at most 128 registers/thread.  At 256 threads this
   leaves exactly 32,768 registers/CTA and permits two CTAs/SM; 129 registers
   is a structural rejection, not a tuning point.
8. A logical N32 reference panel is retained as two ordered physical N16
   stages when bitwise parity requires the original online-softmax boundary.
9. Arithmetic order, online-softmax order, FP16 probability rounding, and FP32
   accumulation order remain identical to the accepted reference.

The first micro gate is the real TP4-rank shape, not an isolated MMA panel:

- `M=8096,N=121440,Hq=6,Hkv=1,D=256,page=784`;
- normal, reverse, and tail page tables;
- bitwise output at the first promotion gate;
- no PTXAS spill and no SASS `LDL/STL`;
- shared memory at most 48 KiB if two-CTA residency is claimed;
- at least 10% wall reduction before NCU;
- after NCU: tensor active at least 30%, no-eligible at most 45%, and both
  long-scoreboard and barrier cycles/issue below the accepted kernel.

An eight-warp candidate that only changes occupancy, repeats K/V reads, or
adds shared round trips is rejected even if its register count is lower.

The measured outcome of this trial is recorded in the closed-path section
below.  The full 4/8/12/16-warp M64 matrix is now closed.

## Decode P0

Decode remains a separate kernel because its target is KV bandwidth and
partition scheduling rather than M64 tensor reuse.  It starts from the
accepted G6 p256 partition and preserves ascending reduction order.

1. Prefetch the next K/V panel into bounded vector payload registers.
2. Execute current QK/PV before delayed publication to the alternate shared
   stage.
3. Replace full-CTA stage synchronization with named barriers.
4. Derive active partitions from SM count using the FlashInfer XQA planner.
5. After arithmetic promotion, let the last completing CTA perform the
   existing ordered merge through graph-stable semaphore state.

The 128K reference is 0.37682 ms for partition plus merge.  Promotion requires
at most 0.301 ms, bitwise output, DRAM throughput at least 60%, no race, and an
endpoint TPOT win.

## Closed Paths

Do not repeat the following under the new backend name:

- 512-thread M64 with 83 KB shared memory: fewer barriers but lower eligible
  warps and a 2% long-shape regression;
- generic 8-producer/8-consumer shared K/V staging;
- 16-warp paired-QK staging that spills at the 64-register boundary;
- two D128 CTAs that repeat QK and softmax;
- probability-only pitch or group-rotation tuning;
- source-only prefetch fences that do not change SASS;
- raw HMMA.884 decomposition that replaces one native WMMA operation with
  longer serial instruction chains;
- direct capability-gate changes to run the SM75+ FlashInfer kernel on SM70.

### Eight-warp BM64 staged-K/V rejection

The first Volta-specific mapping passed every feasibility gate before timing:
eight warps, four M16 query-owner pairs, D128 output per warp, a reusable 8KB
K/V stage, bounded 16-GPR V payloads, 128 registers/thread, 43,776B shared,
two CTAs/SM, zero stack/spill/`LDL`/`STL`, and packed-FP16 bitwise equality for
random and alternating inputs at one, two, and four KV blocks.

It nevertheless fails the wall-time gate decisively against the exact two-BM32
`ALL_P + PAIR_SCRATCH` baseline:

| input | KV blocks | two-BM32 baseline | eight-warp BM64 | delta | wins |
|---|---:|---:|---:|---:|---:|
| random | 1 | 56.576 us | 69.760 us | +23.30% | 0/100 |
| alternating | 1 | 56.832 us | 70.144 us | +23.42% | 0/100 |
| random | 2 | 86.144 us | 116.480 us | +35.22% | 0/100 |
| alternating | 2 | 86.656 us | 117.120 us | +35.16% | 0/100 |
| random | 4 | 147.712 us | 210.560 us | +42.55% | 0/100 |
| alternating | 4 | 147.522 us | 210.432 us | +42.64% | 0/100 |

Static SASS preserves 768 HMMA instructions and reduces LDG `74 -> 21` and
STS `70 -> 54`, but increases LDS `160 -> 200` and BAR `6 -> 12`.  More
importantly, resident warps fall `32 -> 16`.  The saved global operand loads do
not repay shared staging, extra synchronization, and half the latency-hiding
pool; the regression growing with KV blocks confirms a steady-state structural
loss rather than launch noise.

NCU is intentionally skipped because the wall gate failed.  This closes the
eight-warp BM64 shared-stage structure, including spelling changes to its
barriers or stage layout.  Full-grid checks remove the only remaining doubt:
at 144 groups the one/two/four-block regressions are 11.3%, 19.7%, and 22.9%;
at 288 groups they are 11.9%, 19.3%, and 23.6%.  The result therefore remains
negative after both resident CTA slots are populated and is representative of
the wide M8096 grid.

The distinct 12-warp direct-K/V owner-local form also fails before timing.
With `__launch_bounds__(384,2)`, PTXAS is forced to 80 registers but emits an
816-byte stack plus 1,512-byte spill stores and 1,992-byte spill loads per
thread; SASS contains 453 `LDL` and 237 `STL`.  It cannot satisfy the two-CTA
resource envelope.  The older 512-thread M64 form is spill-free but uses
83,472B shared, permits one CTA/SM, and loses about 2% at long shape.  The
literal four-warp mapping exceeds the 255-register limit.  These results close
the 4/8/12/16-warp M64 family for the current arithmetic contract.  No new
M64 successor is selected.

Artifacts:

- `bench_results/flashinfer_sm70_20260715/bm64_pipeline_smoke_clock1200.json`
- `bench_results/flashinfer_sm70_20260715/bm64_pipeline_ab100_clock1200.json`
- `bench_results/flashinfer_sm70_20260715/bm64_pipeline_groups144_ab100_clock1200.json`
- `bench_results/flashinfer_sm70_20260715/bm64_pipeline_groups288_ab50_clock1200.json`
- `bench_results/flashinfer_sm70_20260715/bm64_direct_groups144_ab50_clock1200.json`

## Active Prefill Decision

For SM70, the optimal replacement for FlashInfer's modern four-warp M64 body
is the accepted native BM32 `ALL_P + PAIR_SCRATCH` body, not an emulation of
the modern tile.  It retains 32 resident warps, native Volta WMMA ordering,
direct K/V fragment loads, top/bottom operand reuse, zero local traffic, and
the strongest measured wall time.

The independent backend now generates FlashInfer-style per-request
`request_indices`, `qo_tile_indices`, `kv_tile_indices`, `merge_indptr`, and
`o_indptr`, including resource-derived split decisions, graph padding, and
16-byte-aligned integer/float workspace sizes.  Exact CPU lengths and
upper-bound plans are distinguished.  The description is attached to metadata
but the graph-stable GPU workspace is not allocated yet.

Kernel selection is architecture-specific: the strict SM70 prefix shape
dispatches the BM32 body while modern GPUs retain upstream BM64.  This is the
required "optimal unsupported-primitive replacement" policy; a slower shape is
not accepted merely because it resembles upstream source more closely.

The fixed entry is plumbing, not a speedup: its operator p50 is intentionally
neutral to the same BM32 body.  The next gate remains microbenchmark-only at
the real `M8096,N121440,Hq6,Hkv1,D256,page784` shape.  A candidate must reduce
p50 wall by at least 2% and win at least 95/100 paired rounds before any Qwen
endpoint run is allowed.  A candidate that preserves reduction order must be
bitwise exact.  Split-KV is the explicit exception because it changes the FP32
reduction tree: it must instead pass output max/p99 absolute-error bounds,
LSE bounds, fully-masked causal cases, and the later model-quality gate.  The
comparison must include the inherited zero-copy contiguous-dense route when
the page table permits it, so the fixed paged entry cannot displace a faster
existing path unnoticed.

## Fixed BM32 Entry Promotion

`flash_attn_prefill_paged_d256_bm32_allp_pair_scratch` is a dedicated pybind
entry.  It directly instantiates
`flash_attention_forward_paged_d256_bm32_phase_kernel<true,true,true>` and does
not read the generic BM32 environment switches.  It rejects non-FP16 KV,
non-D256, non-page-784, `M < 32`, invalid GQA/layout, non-SM70, and unvalidated
CUDA-graph capture instead of selecting another kernel.

The formal CPython 3.12 build records 64 registers/thread, zero stack/local
spill, 41,936 bytes shared memory, two resident CTAs/SM, 768 HMMA instructions,
and zero `LDL/STL`.  Normal, reverse, and tail page tables are bitwise equal to
the accepted generic entry even when all four generic BM32 switches are set to
zero for the fixed call.

The first 100-pair fixed-entry artifact is invalid as absolute performance
evidence.  An unrelated TP4 process overlapped the run and inflated the
`M1024,N65536,Hq6,Hkv1,D256,page784` p50 from the accepted approximately
`31.7 ms` to `74.7 ms`.  The fixed/generic delta was still neutral, but neither
its absolute time nor its win count is a promotion result.  A clean 20-pair
calibration at 1200 MHz restored generic/fixed p50 to
`31.8915/31.8336 ms` for normal pages, `31.8858/31.8403 ms` for reverse pages,
and `31.8956/31.8351 ms` for the tail case.  This confirms that the pybind
entry is neutral plumbing; it does not satisfy the acceleration gate.

`FlashInferSM70Impl.forward` now bypasses the parent dispatcher only for exact,
single-request, non-graph, causal prefix prefill with FP16 page-784 KV, D256,
`M >= 32`, no BFLA/DDTree/sliding window, and no requested Triton or split-KV
route.  A fixed-entry error propagates without fallback.  Runtime proof records
the fixed pybind and `<true,true,true>` specialization only after a successful
launch; all delegated shapes retain `kernel_promoted=false`.

## Exact-Shape Clock And Process Gate

The real acceptance baseline was re-established at
`M8096,N121440,Hq6,Hkv1,D256,page784`.  A reset graphics-clock lock produced a
false `255.6-257.0 ms` result while the device actually ran at
`1515-1530 MHz`.  The ratio is exactly explained by frequency:
`324.1 ms * 1200/1530 = 254.2 ms`.  This is not a kernel improvement.

After relocking GPU0 to 1200 MHz, the paged BM32 pair-scratch path measures
`324.109 ms` p50 over 20 launches.  One-second `nvidia-smi dmon` samples show
1200 MHz throughout the load.  This matches the earlier `325.886 ms` baseline
within 0.55%.  Future promotion artifacts must fail if another GPU0 compute
process appears or if any load-period clock sample differs from 1200 MHz;
before/after clock checks alone are insufficient.

The physically contiguous dense path is not the fastest valid baseline at
this shape.  With Hkv=1 its K/V tensors are genuine zero-copy views and its
output is bitwise equal to paged attention, but a clean calibration measured
approximately `548 ms` versus `256 ms` paged at the same temporarily unlocked
1530-MHz condition.  It is therefore structurally more than 2x slower and is
excluded from candidate promotion while remaining covered by route tests.

## Three-Way Split-KV Feasibility Gate

The first algorithmic candidate after the neutral fixed entry is the existing
BM32-native three-way split-KV micro.  Its partial kernel compiles at
64 registers/thread, 41,744 bytes shared, zero spill/local traffic, and two
CTAs/SM; its merge kernel uses 32 registers/thread and no shared memory.
These static results only prove resource feasibility, not paged correctness or
speed.

At the exact grid, unsplit has `1518 = 10*144 + 78` CTAs and consumes eleven
waves.  Three-way partial has `4554 = 31*144 + 90` CTAs and consumes 32
one-third-length waves.  Against the locked `324.109 ms` baseline, the ideal
lower bound is `314.288 ms`, only 3.03% faster.  The combined partial plus
merge p50 must therefore be at most `317.627 ms` to pass the 2% gate, leaving
only 3.339 ms above the ideal wave model for partition and merge overhead.

The first run uses 759 synthetic M64 groups so its baseline grid is exactly
1518 CTAs.  It is a fast rejection test only.  A passing result must still be
reimplemented and remeasured with the production paged kernel at exact
`M8096,N121440`, causal masking, page size 784, the 96-token tail, normal and
reverse page tables, softmax scale 1/16, and LSE comparison.  Split-KV changes
the FP32 reduction order and cannot claim bitwise parity.  Therefore a speed
pass is feasibility evidence only under the current bitwise promotion rule;
it does not authorize endpoint timing.  Promotion would require an explicit
exactness-policy decision backed by strict numerical bounds and the complete
model-quality gate.

### Production exact split-KV result

The synthetic feasibility screen passed before the production port.  At 759
groups and 1200 MHz, random input measured `313.220 -> 304.859 ms`
(`-2.669%`, 100/100 wins), while alternating input measured
`313.137 -> 304.775 ms` (`-2.671%`, 20/20 wins).  This established that the
CTA-wave decomposition was viable, but it did not exercise paged addressing,
causal masking, or the production softmax body.

The production implementation uses a sibling global entry, leaving the
accepted three-boolean kernel symbol and argument ABI intact.  Its partial
kernel has 64 registers/thread, 41,952B shared memory, zero stack/local
traffic, zero `LDL/STL`, and two resident CTAs/SM.  The merge has 32
registers/thread and no shared/local traffic.  Workspace is fixed at
`[B,H,3,M,D]` FP32 numerator plus `[B,H,3,M]` FP32 max and sum, or about
143.4 MiB total at the exact shape.

The first safe production form was correct but failed the speed gate:

| page table | unsplit p50 | safe split p50 | delta | wins |
|---|---:|---:|---:|---:|
| normal | 322.715 ms | 318.086 ms | -1.434% | 100/100 |
| reverse | 322.856 ms | 318.487 ms | -1.353% | 100/100 |
| tail | 322.902 ms | 318.536 ms | -1.352% | 100/100 |

Nsight Systems split the safe candidate into about 317.702 ms partial and
0.538 ms merge, so merge optimization alone could not recover the missing
percentage point.  A row-max sentinel attempt moved the causal-empty test out
of integer address logic, but expanded the partial SASS from 3,432 to 3,504
instructions and regressed it to 328.876 ms versus 322.370 ms unsplit in the
same profile.  It was reverted and must not be retried by changing the
sentinel spelling.

The accepted micro candidate uses a host-length fast-visible specialization.
It is restricted to `B=1`, receives the planner's scalar `actual_n`, and uses
64-bit host arithmetic.  For `T=ceil(N/128)`, the fast body is selected only
when `T>=3` and
`floor(2*T/3)*128 <= N-M`.  This proves that every non-empty partition has a
visible prefix for the earliest causal query; it does not claim that all later
panels are visible, and it does not remove causal masking.  Shapes outside the
condition retain the safe helper.  At `N=121440,M=8096`, `T=949`, the final
split begins at 80,896 and the earliest cutoff is 113,344, so the fast body is
valid.  Its SASS falls to 2,688 instructions, 47 branches, and 69 predicate
instructions while retaining 768 HMMA and zero local-memory instructions.

Formal 100-pair results at locked 1200 MHz are:

| page table | unsplit p50 | fast split p50 | delta | wins | split p90/p99 |
|---|---:|---:|---:|---:|---:|
| normal | 322.736 ms | 315.723 ms | -2.173% | 100/100 | 315.780/316.276 ms |
| reverse | 322.681 ms | 315.685 ms | -2.168% | 100/100 | 315.746/316.259 ms |
| tail | 322.710 ms | 315.688 ms | -2.176% | 100/100 | 315.757/316.292 ms |

Across all page tables, output max/p99 absolute error is
`1.5259e-5/7.6294e-6`; LSE max is `3.8147e-6` to `4.7684e-6` and LSE p99 is
`2.8610e-6`.  The dedicated `N=513,M=512` safe fallback test verifies that
fully masked partitions write exactly zero numerator/sum.  The equality
boundary `N=785,M=273` covers the 784-token page crossing, reverse pages, a
17-token KV tail, and a partial BM32 query tile.

Nsight Systems confirms the exact route launches
`splitkv3_partial_kernel<false>`: 316.302 ms partial plus 0.561 ms merge,
versus 324.415 ms unsplit under profiling.  The formal run contains 595
background samples, all at 1200 MHz, with no external compute PID or monitor
violation.

### Validated chunk window and endpoint result

The production route is not enabled from the first mathematically safe
length.  A locked-clock normal-page screen found that the split overhead does
not cross the 2% wall-time gate until the tenth 8,096-token chunk:

| host N | unsplit p50 | split p50 | delta | wins | decision |
|---:|---:|---:|---:|---:|---|
| 24,288 | 64.626 ms | 63.988 ms | -0.988% | 20/20 | reject |
| 32,384 | 86.145 ms | 84.958 ms | -1.377% | 20/20 | reject |
| 64,768 | 172.125 ms | 168.803 ms | -1.930% | 20/20 | reject |
| 72,864 | 193.661 ms | 189.833 ms | -1.977% | 20/20 | reject |
| 80,960 | 215.178 ms | 210.812 ms | -2.029% | 20/20 | promote boundary |
| 89,056 | 236.599 ms | 231.721 ms | -2.062% | 20/20 | promote |
| 97,152 | 258.158 ms | 252.748 ms | -2.096% | 20/20 | promote |
| 105,248 | 279.783 ms | 273.795 ms | -2.140% | 20/20 | promote |
| 113,344 | 301.242 ms | 294.769 ms | -2.149% | 20/20 | promote |

The weakest promoted length, `N=80960`, then passed the full 100-pair gate:
normal `215.173 -> 210.797 ms` (`-2.034%`), reverse
`215.142 -> 210.793 ms` (`-2.021%`), and tail
`215.240 -> 210.905 ms` (`-2.014%`), all 100/100 wins.  Its output maximum
absolute error is `1.5259e-5`, LSE maximum is `3.8147e-6`, and all 398 monitor
samples passed at 1200 MHz.  Consequently the backend accepts only
`N={80960,89056,97152,105248,113344,121440}` for `M=8096,Hq=6,Hkv=1`;
it does not infer a wider range from the fast-visible predicate.

The wrapper caches one approximately 143.4 MiB FP32 workspace per device and
stream.  The explicit `FLASHINFER_SM70` backend records host `actual_n`, the
executed `<false>+merge` kernel, and fails closed when the split entry is
missing.  Backend/planner tests pass 32/32 and the real SM70 fixed/split
operator tests pass 6/6.

Matched Qwen3.6-27B-AWQ endpoint runs used TP4, TurboMind AWQ, FP16 KV,
`max_model_len=131072`, chunk 8,096, CUDA graphs enabled, input 121,440,
64 generated tokens, and the official `temperature=1.0,top_p=0.95,top_k=20`
sampling parameters.  Startup and graph capture are excluded from request
metrics:

| run | backend | prefill | TTFT | decode TPOT |
|---|---|---:|---:|---:|
| baseline 1 | FLASH_ATTN_V100 | 79.638 s | 79.726 s | 23.445 ms |
| candidate 1 | FLASHINFER_SM70 | 78.247 s | 78.283 s | 22.245 ms |
| baseline 2 | FLASH_ATTN_V100 | 78.802 s | 78.961 s | 22.238 ms |
| candidate 2 | FLASHINFER_SM70 | 78.205 s | 78.241 s | 22.379 ms |

Using the faster baseline 2 as the conservative reference, the two candidate
runs save 554-596 ms, or 0.704%-0.757% of prefill.  This matches the operator
prediction: the six promoted chunks save about 34.1 ms per full-attention
layer, or about 546 ms across the model's 16 full-attention layers.  Decode is
unchanged within run noise.  Every TP rank reports exactly 96 split calls
(`6 chunks * 16 layers`), 128 fixed-entry calls, and 16 first-chunk dense calls.

All four endpoint runs produce identical 64-token sequences.  Baseline
self-noise reaches `0.00131` on chosen-token logprob and `1.109` in the low-rank
top-20 tail; baseline 2 versus candidate 2 is `0.00128` and `0.289`
respectively.

A separate production-memory quality A/B dumped the complete 248,320-element
sampler logits for 16 generated steps on all four TP ranks.  All 64 shapes and
argmax values match, with zero argmax mismatches and maximum absolute logits
difference `0.12109375` against the pre-registered `0.125` bound.  The 16
sampled tokens are identical; selected-token logprob max difference is
`0.001277 < 0.02`, and top-20 max difference is `0.03125 < 0.125`.

Full 121,440-token prompt-logprob collection is not feasible on these 32-GiB
V100s with an 8,096 chunk: each chunk requires a 3.75-GiB full-vocabulary
all-gather and the prompt results remain resident.  It OOMs at memory
utilization 0.9 and 0.75; `expandable_segments` cannot be used because it
breaks custom-all-reduce CUDA IPC graph registration.  Therefore the
repository's aggregate model gate remains `B-pending` only for missing prompt
logprob/perplexity evidence.  Operator-wide numeric checks plus output and
sampler-logits gates pass, but the backend remains explicit-only until a
bounded selective prompt-logprob observer or a broader long-context quality
matrix closes that last evidence gap.

Artifacts:

- `bench_results/flashinfer_sm70_20260716/native_bm32_splitkv_realshape_ab100_random_g759_nb949_clock1200.json`
- `bench_results/flashinfer_sm70_20260716/splitkv3_exact_ab100_m8096_n121440_clock1200.json`
- `bench_results/flashinfer_sm70_20260716/splitkv3_fast_visible_ab100_m8096_n121440_clock1200.json`
- `bench_results/flashinfer_sm70_20260716/splitkv3_fast_visible_ab100_m8096_n80960_clock1200.json`
- `bench_results/flashinfer_sm70_20260716/splitkv3_fast_visible_exact_correctness_nsys.nsys-rep`
- `bench_results/flashinfer_sm70_20260716/model_splitkv3_endpoint/`
- `bench_results/flashinfer_sm70_20260716/model_splitkv3_quality/compare_sampler_only_baseline_vs_candidate.json`

## Promotion Sequence

| stage | required evidence | status |
|---|---|---|
| independent backend registry and graph plumbing | targeted unit tests | implemented; explicit route, graph fixed entry still closed |
| resource-driven SM70 planner | per-request indices, resource/split matrix, workspace sizing | host description implemented; GPU workspace pending |
| SM70 instruction compatibility micro | K256 register-resident exact fragment mapping, SASS, no spill | passed |
| paged prefill M64 trial | exactness, resource, full-grid wall gate | rejected; BM32 selected |
| fixed BM32 pybind entry | exactness, no fallback, resource/SASS, operator A/B | passed |
| paged decode micro | exactness, wall gate, NCU gate | pending |
| Python implementation integration | cached workspace, real split entry, runtime proof, fail-closed route | passed; 32 backend/planner and 6 SM70 operator tests |
| exact M8096 micro acceleration | strict numeric bounds, >=2% p50, >=95/100 wins against fastest valid baseline | passed for the explicit six-length promotion set |
| endpoint integration | fixed clocks, route counts, output IDs, TPOT/TTFT | speed/route passed at +0.704% to +0.757% prefill; sampler/output quality passed, full prompt metric pending |
| default promotion | full quality and long-context matrix | pending |

Every failed experiment is recorded here with its exact shape, resource
report, wall result, and decision before the next design starts.

Primitive compatibility evidence:

- `bench_results/flashinfer_sm70_20260715/volta_mma_k256_smoke_clock1200.json`
- `bench_results/flashinfer_sm70_20260715/fixed_entry_operator_ab100_m1024_n65536_clock1200.json`
