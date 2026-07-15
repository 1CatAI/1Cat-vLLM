# SM70 Versus SM120 Long-Context Attention Parity

## Scope

This document compares the current V100 path with the RTX 5090 only to find
portable optimization ideas. It does not optimize the 5090 deployment.

The locked V100 endpoint is Qwen3.6-27B-AWQ, TP4, TurboMind AWQ,
Flash-V100, FP16 KV cache, 784-token hybrid pages, CUDA graphs, no eager, no
MTP, one request, official sampling, and `max_num_batched_tokens=8096`.
The reference 5090 endpoint is the reported single-GPU FlashInfer 0.6.13
service with FP8 KV cache and the same 8096-token scheduler chunk.

The machines do not have identical TP or KV dtypes. Operator comparisons
therefore report both wall time and useful attention FLOP/s; endpoint results
remain the final authority.

## Measurement Integrity

The first 8096-token micro run in this investigation used the base Python 3.13
environment. It loaded the stale July 13
`flash_attn_v100_cuda.cpython-313-x86_64-linux-gnu.so`, which did not contain
the promoted BM32 phase kernel and dispatched the old generic BM16 kernel.
The resulting `26.629 s` 16-layer 64K estimate and `737.8 ms` M8096/N121440
measurement are invalid as current baselines.

All accepted results below use Python 3.12 from
`vllm-0.0.5-t210`, Torch 2.10.0+cu128, and the July 15
`flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so`. The microbenchmark now
records the Python executable, version, and loaded extension path in every
result header.

## Locked 8096 Endpoint Baseline

The V100 run uses 65,472 prompt tokens and 64 forced output tokens. The old
and new runs have equal prompt hashes and exactly equal 64 output token IDs.

| endpoint | chunk | 64K prefill | prefill tok/s | TPOT | decode tok/s |
|---|---:|---:|---:|---:|---:|
| V100 accepted prior run | 1,024 | 38.7454 s | 1,689.8 | 18.993 ms | 52.65 |
| **V100 locked baseline** | **8,096** | **33.0984 s** | **1,978.1** | **19.114 ms** | **52.32** |
| RTX 5090 reference | 8,096 | 21.518 s | 3,045.7 | 14.366 ms | 69.61 |

Changing only the V100 scheduler chunk from 1024 to 8096 reduces prefill
latency by 14.57% and raises prefill throughput by 17.06%. Decode changes by
0.12 ms, which is neutral at this one-run precision. The remaining 64K gap is
1.54x in prefill throughput and 1.33x in decode latency.

The route summary proves 16 dense Flash-V100 calls, 128 paged-prefix
Flash-V100 calls, and 48 XQA decode calls per rank. There is no Triton or eager
fallback.

## Exact Prefill Operator Comparison

The comparable long-prefix point is `M=8096`, `N=121440`, causal D256.
V100 measures one TP rank (`Hq=6,Hkv=1`, FP16 KV); 5090 measures the full
model (`Hq=24,Hkv=4`, FP8 KV).

| device/path | useful work | low-overhead wall | useful rate |
|---|---:|---:|---:|
| V100 BM32 pair-scratch, one TP rank | 5.839 TFLOP | 325.886 ms | 17.92 TF/s |
| RTX 5090 FlashInfer | 23.357 TFLOP | 144.269 ms | 161.90 TF/s |

TP4 executes the four V100 ranks concurrently, so model-level attention wall
time is still 325.886 ms, not four times that value. The 5090 handles four
times the per-GPU work in 0.443x the wall time. The current V100 model-level
attention wall is therefore 2.26x the 5090 reference.

### NCU comparison

NCU replay inflates the 5090 wall time, so CUDA-event/Nsight Systems values
above remain timing authority. The counters still provide a useful structural
comparison.

| item | V100 BM32 | RTX 5090 FlashInfer |
|---|---:|---:|
| grid / block | 1,518 / 512 threads | 3,036 / 128 threads |
| registers/thread | 64 | 254 |
| shared memory/CTA | 41.94 KB | 99.36 KB including driver allocation |
| resident CTA/SM | 2 | 1 |
| achieved occupancy | 49.96% | 8.33% |
| CTA waves/SM | 10.54 | 17.86 |
| SM throughput | 37.53% | 35.20% |
| Tensor-pipe active, elapsed | 20.97% | 35.20% |
| DRAM throughput | 1.00% | 0.19% |
| L1/TEX throughput | 94.01% | 31.40% |
| L2 throughput / hit rate | 24.69% / 99.59% | 17.08% / 99.65% |
| active / eligible warps per scheduler | 7.99 / 0.738 | 1.00 / 0.123 |
| no-eligible cycles | 60.86% | 87.66% |
| long-scoreboard cycles/issue | 4.684 | 0.096 |
| short-scoreboard cycles/issue | 4.638 | 0.942 |
| barrier cycles/issue | 4.393 | 0.089 |
| MIO-throttle cycles/issue | 1.982 | 0.105 |
| math-pipe / wait cycles/issue | 0.263 / 1.929 | 2.751 / 2.973 |

Low occupancy is not itself the 5090 advantage. FlashInfer keeps only one
warp per scheduler resident, but its dependency stalls are tiny and its main
wait has moved to math-pipe execution. V100 has eight resident warps per
scheduler and still cannot find an eligible warp for 60.86% of cycles. Its
remaining prefill bottleneck is the synchronous L1/shared/HMMA operand path,
not HBM capacity or insufficient grid size.

At M8096 the V100 launch is ten full 144-CTA waves plus a 78-CTA tail. Even
perfect wave alignment can remove at most about 4.4% at this shape. It is a
secondary optimization, not the parity route.

### Why resident warps do not become useful work

The V100 kernel is not generally idle. It is spending most cycles moving and
replaying operands through the on-chip path instead of issuing useful HMMA.
The source-counter report attributes the not-issued samples as follows:

| not-issued cause | share | dominant code region |
|---|---:|---|
| long scoreboard | 27.55% | HMMA waiting for K/V operands |
| CTA barrier | 24.42% | phase hand-off after imbalanced QK work |
| short scoreboard | 24.21% | shared score/probability and HMMA dependencies |
| MIO throttle | 13.39% | shared-memory instruction pressure |
| all other causes | 10.43% | math, wait, control, and minor stalls |

These four causes account for 89.57% of all not-issued samples. The kernel has
7.99 active warps per scheduler, but only 0.74 eligible warps and 39.15% issue
slots busy. Its 94.01% L1/TEX throughput is therefore not evidence of useful
KV bandwidth. NCU reports 2.5-way shared-store conflicts, with excessive
wavefronts making up 53.72% of shared-store wavefronts. This is an on-chip
operand-feed bottleneck.

The source structure explains the counters:

- the CTA has 16 warps, but only the first eight execute QK;
- K and V matrix fragments are loaded from their cache pointers immediately
  before the consuming HMMA, without a cooperative KV shared-memory pipeline;
- QK scores are materialized in a 16KB shared array, converted into shared
  FP16 probabilities, and consumed again by PV;
- every 128-token step uses full-CTA phase barriers, so the resident warps
  become blocked together rather than hiding one another's latency.

The current 128K G6 decode kernel shows the same failure in a bandwidth-shaped
workload. It has 2.97 active but only 0.54 eligible warps per scheduler, 65.66%
no-eligible cycles, and 62.00% of not-issued samples in long scoreboard. The
top PCs are `STS.128` instructions waiting for preceding global KV loads. It
therefore reaches only 37.03% DRAM throughput even though the KV scan should be
bandwidth bound.

## What FlashInfer Does Differently

The measured 5090 prefill kernel is causal `BM64/BN64`, four warps, and uses
FP8 KV repacking into a 128-byte-swizzled FP16 shared layout. FlashInfer's
kernel source pipelines global-to-shared copies with `cp.async`, then waits on
the asynchronous groups immediately before QK and PV consumption. Its planner
selects query/KV tiles and split-KV work from the request lengths and an
SM-derived CTA budget. The graph-safe wrappers preallocate split-K temporary
state and reuse planning metadata across layers.

Relevant upstream material:

- [FlashInfer paper](https://arxiv.org/abs/2501.01005)
- [FlashInfer 0.6.13 prefill kernel](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/include/flashinfer/attention/prefill.cuh)
- [FlashInfer 0.6.13 scheduler](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/include/flashinfer/attention/scheduler.cuh)
- [FlashInfer 0.6.13 XQA scheduler](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/csrc/xqa/mha.cu)
- [FlashInfer 0.6.13 async-copy wrapper](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/include/flashinfer/cp_async.cuh)
- [FlashInfer attention wrappers](https://docs.flashinfer.ai/api/attention.html)

The complete backend cannot be enabled on V100 by changing one capability
check. The local vLLM backend requires SM75 or newer, and the FA2 kernel uses
`cp.async`/`ldmatrix`-era layouts that SM70 cannot execute. The portable part
is the planning, work decomposition, graph-safe workspace, and short
dependency structure. Those ideas must be reimplemented with Volta HMMA.884,
ordinary LDG/STS, registers, and named barriers.

## Closed Prefill Paths

The following results remain closed and must not be repeated under a new
name:

- the 512-thread M64 kernel lost about 2% because it dropped to one CTA/SM;
- the 8-producer/8-consumer shared K/V pipeline spilled or regressed;
- the 16-warp paired-QK pipeline spilled despite aliasing the all-P scratch;
- probability rotation spilled, while padded probability layouts stayed
  below the mandatory 2% wall gate;
- moving one existing V load with a compiler fence did not change scheduling;
- the single HMMA bundle swap did not produce an operator win.

FlashInfer-style split-KV is also not the first prefill action at M8096. The
current launch already has 1,518 CTAs and 10.54 waves/SM; splitting KV adds
partials and a merge without fixing the per-CTA dependency chain.

## Directly Portable FlashInfer Structure

The exact FlashInfer 0.6.13 prefill kernel used by the 5090 does not assign
warps to disjoint N slices and then exchange scores through shared memory. It
packs GQA rows as `(token, query-head-within-KV-group)`, launches over KV heads,
and assigns each of four warps an M16 query panel. With `NUM_WARPS_KV=1`, each
warp owns its QK fragments, online-softmax state, and output fragments through
the whole KV traversal. K and V use 128-byte-swizzled shared layouts.

Its two asynchronous copy groups implement this repeating schedule:

1. wait for current K while current V may remain in flight;
2. execute current QK and issue the next K copy;
3. wait for current V while next K may remain in flight;
4. execute current PV and issue the next V copy.

SM70 cannot execute `cp.async` or the newer matrix-load path. The same
dependency graph is portable: issue `LDG.E.128` into prefetch registers well
before use, execute independent HMMA, then publish with `STS.128` after the
old shared stage is dead. Named barriers should protect only the stage being
handed off. GQA packing, query-panel ownership, register-resident state,
SM-count work planning, and graph-safe workspace reuse are directly portable.
The 128-byte bank-period idea is portable, but FlashInfer's exact swizzle is
not: SM70 needs an explicit LDS/HMMA fragment mapping instead of `ldmatrix`.

### P0: FlashInfer-shaped M64 prefill microkernel

This supersedes the compact eight-warp BM32 sketch as the first structural
microbenchmark. The prior 512-thread M64 rejection does not close this route:
that kernel retained N-sliced QK, shared score/probability round trips, and
full-CTA phase barriers.

1. Pack the six TP4 GQA query heads and launch `grid.z=Hkv=1`, matching
   FlashInfer's logical work mapping.
2. Use `M64` with four query-owner warps. Each warp owns one M16 panel through
   QK, online softmax, and PV, so no cross-warp softmax exchange is required.
3. Keep the complete D256 output in registers. The hard gate is at most 255
   registers/thread, zero local traffic, and a 128-thread CTA.
4. Use 32KB shared Q plus one 8KB K and one 8KB V microstage. This exactly
   fits 48KB and allows two CTAs/SM on V100.
5. Preserve each logical N32 arithmetic panel with two N16 physical stages.
   This retains token order, FP16 probability rounding, and HMMA accumulation
   order while allowing an early-LDG/delayed-STS software pipeline.
6. Implement a same-capacity, 128-byte-period Volta K/V swizzle with explicit
   LDS/HMMA fragment loads. Do not copy the `ldmatrix` lane map. Reject the
   candidate if shared conflicts, long scoreboard, or barrier stalls are
   merely moved elsewhere.

The live V100 reports 65,536 registers and 98,304B shared memory per SM. With
register allocation rounded to 8,192 registers/warp, four 255-register warps
consume 32,768 registers; 48KB shared consumes 49,152B. Two such CTAs fit
exactly by both limits, yielding eight resident warps (12.5% occupancy). This
is a verified resource envelope, not an assumed occupancy target.

The first gate remains bitwise equality on sequential, reversed, and tail page
tables and `M8096,N121440 <=293.3 ms`. Structural promotion requires
`<=260.7 ms` (20% faster), Tensor-pipe active at least 30%, at most 40%
no-eligible cycles, and no spills. Compact eight-warp BM32 remains a fallback
only if M64 cannot satisfy the register/shared gates.

### P0: FlashInfer-shaped G6 decode pipeline

The current FP16 XQA kernel is already split-KV and graph-safe, but at 128K it
uses 168 registers, 43.26KB dynamic shared memory, reaches only 37.03% DRAM
and 30.51% SM throughput, and has 0.54 eligible warps/scheduler. FlashInfer's
long decode uses an SM-sized split grid and a persistent merge, but its
hardware pipeline cannot be copied directly.

Current G6 already removes six redundant global KV reads, but it still stages
each panel as load, immediate `STS.128`, full barrier, consume. QK uses only
four of six warps, while PV uses six scalar-FMA warps. The first SM70 micro must
therefore copy FlashInfer's dependency graph rather than only reduce the thread
count:

1. issue the next panel's global K/V loads into prefetch registers before the
   current panel's compute;
2. delay `STS.128` until the corresponding shared buffer is dead, with named
   stage barriers rather than full-CTA barriers;
3. keep G6 query and online-softmax state resident and retain the accepted p256
   partition boundaries for the first bitwise gate;
4. derive active subsequence work from SM count, as FlashInfer XQA does, but
   initially map that work onto the same ascending p256 partials;
5. after the partition kernel passes, let the last completing CTA perform the
   existing ascending-partition reduction through a semaphore/workspace path,
   removing the separate merge launch without changing reduction order.

Keep the accepted p256 partition and merge order for the first exactness gate.
The 128K partition-plus-merge reference is 0.37682 ms; promotion requires
`<=0.301 ms` (20% gain), bitwise output, zero races, DRAM throughput at least
60%, and an endpoint TPOT win. FP8 KV integration is a separate workstream and
must consume this kernel only after the FP16 arithmetic gate passes.

## Expected Endpoint Mapping

At 64K the measured Flash-V100 attention-only estimate is 12.811 s out of the
33.098 s endpoint prefill. A 10% attention gain saves at most 1.28 s and moves
the endpoint to about 31.82 s. A 30% gain moves it to about 29.25 s. Matching
the observed 2.26x 5090 attention wall would move V100 to about 25.96 s; the
remaining 4.44 s gap to the 5090 endpoint would then be non-attention model
work. Attention is the first target, but it cannot close the entire prefill
gap by itself.

At 64K the current V100 decode gap to 5090 is 4.75 ms/token. Existing graph
decomposition attributes almost all long-context growth to q=1 attention, so
decode work should remain on XQA/partition compute and KV feeding rather than
AWQ GEMM or communication. The first endpoint gate is below 17 ms TPOT at 64K
with unchanged official-sampling output.

## Artifacts

- `bench_results/prefill_wave_chunk_20260715/chunk8096_prompt65536_clock1200_py312.json`
- `bench_results/prefill_wave_chunk_20260715/v100_paged_m8096_n121440_clock1200_py312.json`
- `bench_results/prefill_wave_chunk_20260715/ncu_v100_bm32_m8096_n121440_clock1200_py312.ncu-rep`
- `bench_results/prefill_wave_chunk_20260715/ncu_flashinfer_prefill_q8096_kv121440.ncu-rep`
- `bench_results/prefill_wave_chunk_20260715/model_qwen36_27b_awq_tp4_i65472_o64_chunk8096_clock1200.json`
- `bench_results/nomtp_long_context_20260715/prefill_bm32_pair_scratch_64k_m1024_tp4_clock1200.json`
