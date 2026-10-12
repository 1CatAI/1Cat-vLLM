# SM70 collective-to-projection overlap and activation-residency screens

This benchmark tests whether the next projection can begin reading weights while the existing TP4 all-reduce + residual + Gemma RMSNorm is running. The production collective and original projection kernel remain unchanged. Candidates use the same IQ3_S decoder, FP16 coefficients, FP32 accumulation order and SiLU epilogue. There are no serving or wheel changes.

## Purpose and upper bound

The latest diagnostic target trace attributes 1.304 ms to 130 communication/norm calls. Hiding only the first two projection packets can cover at most a fraction of that service time; the initial optimistic overlap estimate was 0.4–0.7 ms across admitted shapes. Neither that estimate nor kernel service sums constitute complete-round savings.

Two streams fork after clearing a task-owned readiness flag. The auxiliary stream runs the shipped collective and publishes readiness; the projection reads two weight packets before waiting, then uses coherent L2 activation loads. Streams join before the next chain stage. The fixed 68-consumer-CTA grid leaves at least 12 of 80 SMs free. The shipped 40-CTA norm has 56 registers/thread and 28 shared bytes, permitting nine norm blocks per free SM. This avoids an unsafe all-SM consumer rendezvous. No private custom-allreduce C++ ABI is accessed.

The complete-activation-cache alternative stores all eight normalized rows once per CTA with a padded 641-vector shared stride. The K loop reads shared memory directly and does not repeatedly stage activation packets. It requires 85,120 dynamic shared bytes. A final screen eliminates conditions that are known true for this exact, fully populated pair shape and clamps the unused last prefetch to a valid packet.

## Test plan

Base `c4f6245f841466782752a8c3283e4727565cf17a`, four V100-SXM2-32GB ranks with full NV2 connectivity; CUDA 12.8, Torch 2.10.0+cu128. Both arms use the same normal wheel `1.5.2.dev0+g2b00cc8a38.cu128` for the unchanged collective and one research JIT artifact for projections. The JIT artifact is not a production extension.

Real IQ3_S gate/up shards from layers 6, 23, 24 and 51, M=8, K=5120, N=4352 per projection per rank. Four independent weight pairs exceed L2. Eager checks at three activation amplitudes compare norm, FP32 residual and output bits. Twenty changed-input graph replays check the same three outputs on all ranks. Timings use same-process ABBA or symmetric three-arm ordering.

Power limit was observed at 185 W per GPU and was not changed. Clock samples are retained: the later cache screening windows varied mainly around 1.2–1.4 GHz SM with 877 MHz memory. Five seconds of sustained graph warmup did not establish a constant 1530 MHz clock. Use only paired differences from each run; do not subtract absolute values from other runs or machines.

## Test results

All four ranks passed bitwise eager and changed-input graph checks, including FP32 residuals.

| Screen | Four-pair control us | Candidate us | Decision |
|---|---:|---:|---|
| Two-stream overlap | 174.588–174.594 | 306.073–306.120 | Reject |
| Three-arm isolation, original | 173.870–173.878 | — | Control |
| Same ready kernel, serial | same run | 229.833–229.848 | Reader/cache/synchronization cost |
| Same ready kernel, concurrent | same run | 304.400–304.451 | Additional overlap overhead |
| Whole activation tile, sustained warmup | 192.550–192.556 | 208.347–208.350 | Reject |
| Unconditional whole-tile prefetch | 194.913–194.917 | 196.076–196.083 | No stable positive gain |

Register counts are 97 original, 118 ready reader, 78 complete cache and 64 unconditional cache, all without stack/local spills. Lower register use alone did not improve the chain.

The short graph-node trace confirms only limited projection/norm overlap. After excluding the first replay of each arm, which contains cross-rank profiler-start skew, mean profiled kernel service was:

| Kernel | Original us | Concurrent ready us | Serial ready us |
|---|---:|---:|---:|
| AR + residual + norm | 7.041 | 13.723 | 9.971 |
| gate/up | 33.477 | 46.950 | 41.950 |

These are diagnostic service durations, not end-to-end results.

Privileged NCU worked through sudo while retaining key SSH. Only the task's profiler process was elevated; no driver or sudo policy changes were made. Complete-cache NCU showed 7,745,744 → 7,585,264 dynamic warp instructions (about 2% fewer), barrier stall/issue ratio 0.914 → 0.398, but long-scoreboard stall/issue ratio 0.205 → 2.140. Source correlation attributed 769 additional-wait samples to the decode entrance at SASS offset 0x1620 and substantial samples to activation staging. Removing the known-valid prefetch conditions recovered most of the cache regression but did not produce a reliable speedup. Instruction reduction and overlap are insufficient unless memory waits actually stay off the critical path.

## Decision

Stop these variants and keep the current serving path. Do not perform a model run or claim acceptance/C4 improvements for rejected chains. No complete-round speedup has been accepted; the 12 ms objective remains unmet. The next investigation addresses the draft/verification boundary and graph tail rather than another projection tile or prefetch variation.

Compact records and profiler counters are in [data/gguf_norm_next_projection_54633_20261008.json](data/gguf_norm_next_projection_54633_20261008.json). Generated CUDA, binaries, clocks, lock records and raw profiles are retained outside Git.

## References and transfer limits

[CUDA graph capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) requires captured auxiliary streams to rejoin the origin stream. [PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#cache-operators) distinguish coherent L2 access from non-coherent per-SM L1 caching; the overlapping reader does not use a read-only cache while its activation is being written.

[TokenWeave](https://arxiv.org/html/2505.11329v1) motivates wave-aware communication overlap and warns that small splits can increase overhead. Its Hopper multimem implementation is not available on SM70. Splitting an eight-token GDN verification block also introduces recurrent prefix dependencies and repeated weight reads, so its large-batch results cannot be transferred directly. These sources informed the scheduling screen; no external kernel code was copied.
