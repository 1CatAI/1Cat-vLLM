# GGUF verifier execution boundaries on SM70

The complete-round objective is 12 ms for Qwen3.8-27B IQ3_S with a Q8_0
DFlash2 draft, TP4, E4M3 target KV, FP16 draft KV and FP32 GDN state.
No complete-round improvement is claimed here until the packaged-wheel
comparison, acceptance check and C4 comparison finish. The previously
recorded 14.307 ms is a different host's reference, not a baseline to subtract
from the new host's latency. Every model comparison uses one wheel and
changes only `sm70_gguf.projection_collective_pipeline`.

## Payload and synchronization ledger

The prior resident-operator measurement omitted 19 fallback banks from its
byte total. Recovering their recorded storage footprints adds 181,729,280
bytes. The projection sources and loader layouts have not changed between
that record and integration base `c4f6245f84`.

The peak HBM bound is 898.048 GB/s at the observed 877 MHz memory clock.
These are minimum payload reads, not measured DRAM traffic; activation,
paged-layout, cache, decode and dependency costs are additional.

| Work, per card per round | Payload bytes | Peak HBM floor, ms |
| --- | ---: | ---: |
| Target qkvz + a/b | 476,364,800 | 0.530 |
| Target gate/up | 1,345,848,352 | 1.499 |
| Target down | 723,568,334 | 0.806 |
| Target GDN output | 188,743,680 | 0.210 |
| Target full-attention q/k/v | 141,117,440 | 0.157 |
| Target full-attention o | 61,194,240 | 0.068 |
| Target projections total | 2,936,836,846 | 3.270 |
| Target vocabulary projection | 198,656,000 | 0.221 |
| Draft shared vocabulary projection | 198,656,000 | 0.221 |
| Draft FC, query projections, convolution and selector | 620,298,240 | 0.691 |
| Draft dense context K/V | 26,214,400 | 0.029 |
| GDN initial state + eight FP32 snapshots | 339,738,624 | 0.378 |
| Target unique KV reads at 1K | 8,388,608 | 0.009 |
| Draft unique KV reads at 1K | 5,242,880 | 0.006 |

There are 128 target-layer reductions of 81,920 bytes. Three remote peers
receive 245,760 bytes per card per reduction. Three NV2 pairs provide an
optimistic aggregate one-way 150 GB/s, giving 0.210 ms of serialization for
128 reductions. Protocol latency, waiting and norm work are excluded. The
prior diagnostic trace's 130 communication/reduction nodes include work
besides these layer reductions; its 1.569 ms service sum is not removable
communication time.

A prior NCU counter records 1,513,280 executed warp instructions for IQ3_S
GDN out at M8/N5120/K1536. Four issue slots per SM and 80 SMs at 1290 MHz
imply a 3.666 us ideal issue floor for that one kernel. Other formats and
work categories still need their own counters. Unknown instruction and
synchronization bounds remain explicitly null in the evidence; weight-byte
bounds alone do not complete the round's lower-bound ledger.

Generate the reconciled ledger with `benchmark_gguf_round_lower_bounds.py`.
The two vocabulary projections are dependency-separated and read different
hidden states. The draft file's target feature taps are 6, 20, 34, 48 and 62;
its whole forward cannot run before the final tap becomes available.

## Whole-chain screens

Use actual four-rank shards, eight rotating banks exceeding L2, CUDA Graph,
ABBA order and recorded 1290/877 MHz clocks. These are operator-chain
measurements, not model rounds. Numerical references preserve FP16 operands,
FP32 accumulation, peer sum order and the exact GGUF RMS epsilon.

A first cooperative whole-MLP implementation concatenated two complete
stages with a grid barrier. Current gate/up then down takes about 59 us;
the prototype takes 97--102 us. All intermediate and final output bits
match, including changed-input graph replays. This implementation is
rejected. Its SASS also shows parameter copies to thread-local storage and
121--126 registers, against the original kernels' 95--117 registers. A
persistent launch alone does not remove the dependency or guarantee speed.

The projection/collective prototype publishes each down-output tile while
other CTAs are still computing. Eighty cooperatively resident CTAs prevent
consumer polling from blocking an unscheduled producer. The original five
norm parts and 128-thread warp/sum topology are retained within the larger
CTA. All four ranks match projection, FP32 residual and normalized output
bits across 20 changed-input graph replays per format.

| Actual down shard | Separate down + AR/norm, us | Tile pipeline, us |
| --- | ---: | ---: |
| IQ3_S, layer 6 | 29.522 | 27.101 |
| Q4_K, layer 51 | 32.893 | 30.488 |

These prototype timings precede reuse of the production packet buffer and
omission of unused IQ3 tables for affine/LUT-only projections. The native
operator must be rechecked with its actual packaged layout. Extrapolating
2.4 us over 64 layers gives about 0.15 ms, not a multi-millisecond model
improvement. Only measured, eligible layers can contribute to the final
model delta.

Ten-part and four-warp tree variants changed FP32 variance summation order,
producing one-ULP norm differences. They are rejected in favor of the original
five-part, sequential-warp sum. Read-only kernel descriptors use
`__grid_constant__` and device references to avoid per-thread parameter copies;
that removes the pipeline's local stack frame without changing arithmetic.

## Dispatch and validation

The native operator is part of the normal CMake `_C` extension. It reuses
the existing TP4 push/norm packets and preserves their generation protocol.
No new weight representation or communication-buffer allocation is needed.
Admission covers M8/K4352/N5120, KW4/TN2/split1, one Q4_K, IQ4_XS or IQ3_S
segment, FP16 input, FP32 residual/norm and full NVLink TP4. Other rows and
formats use the original projection followed by the original collective
and norm. Compile-time matching deliberately does not specialize on M.

The graph rewrite retains functionalized scratch outputs and rejects an
additional projection consumer, such as an auxiliary hidden-state tap.
An additional graph screen alternates control and candidate on the same
production packet buffer. Projection, residual and norm remain bitwise
identical for all four ranks and three formats over 20 changed-input replays.
At 1530/877 MHz, rank 0 measures 29.738 to 26.812 us for IQ4_XS and 31.776
to 29.124 us for Q4_K; these numbers are compared only within this new run,
not against the earlier 1290 MHz screen.

Thirty-nine CPU fake-tensor cases cover direct, functionalized and v2 graphs,
M1/M8/M32, metadata epsilon and nonqualified consumers/shapes. Packaged GPU
and model results are pending; the route is not approved for integration yet.

## External implementations

[Mirage's persistent task runtime](https://github.com/mirage-project/mirage/blob/main/include/mirage/persistent_kernel/runtime_header.h)
uses explicit tasks and events rather than simply concatenating operators.
Its default shared-memory budgets exceed SM70's 96 KiB and cannot be used
unchanged. [NanoFlow](https://arxiv.org/abs/2408.12757) studies overlapping
work with different resource demands. These motivate tile readiness and
communication overlap; no external kernel implementation is copied here.

[NVIDIA's Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/volta-tuning-guide/index.html)
provides the scheduler, memory and residency constraints used above.
[NVIDIA's kernel-parameter description](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)
explains `__grid_constant__` on Volta and later GPUs. Measurements, hashes and
explicitly unresolved bounds are in
[data/gguf_structural_pipeline_20261008.json](data/gguf_structural_pipeline_20261008.json).
