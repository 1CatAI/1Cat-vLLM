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
separate backend, metadata-builder, and implementation class.  During the
bring-up phase it deliberately delegates to the accepted SM70 implementation;
this proves serving and CUDA-graph plumbing without claiming a kernel gain.
Individual FlashInfer-shaped kernels replace that delegation only after
microbenchmark promotion.

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

The independent backend still ports FlashInfer's planner, GQA-aware work
description, graph-stable workspace, split/merge metadata, and route
instrumentation.  Kernel selection is architecture-specific: SM70 dispatches
the BM32 body while modern GPUs retain upstream BM64.  This is the required
"optimal unsupported-primitive replacement" policy; a slower shape is not
accepted merely because it resembles upstream source more closely.

The next implementation step is planner/backend hookup around this selected
body.  It must first reproduce the accepted kernel wall time exactly under the
new backend name, with explicit route proof and no fallback.  Only then is a
new residual NCU target admitted.

## Promotion Sequence

| stage | required evidence | status |
|---|---|---|
| independent backend registry and graph plumbing | targeted unit tests | implemented; delegates to baseline |
| resource-driven SM70 planner | register/shared occupancy and split-KV unit matrix | implemented; kernel hookup pending |
| SM70 instruction compatibility micro | K256 register-resident exact fragment mapping, SASS, no spill | passed |
| paged prefill M64 trial | exactness, resource, full-grid wall gate | rejected; BM32 selected |
| paged decode micro | exactness, wall gate, NCU gate | pending |
| per-layer integration | same-process A/B against accepted route | pending |
| endpoint integration | fixed clocks, route counts, output IDs, TPOT/TTFT | pending |
| default promotion | full quality and long-context matrix | pending |

Every failed experiment is recorded here with its exact shape, resource
report, wall result, and decision before the next design starts.

Primitive compatibility evidence:

- `bench_results/flashinfer_sm70_20260715/volta_mma_k256_smoke_clock1200.json`
