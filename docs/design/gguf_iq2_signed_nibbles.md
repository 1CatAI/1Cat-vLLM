# Exact IQ2 signed nibbles for M8 projections

The IQ2_XXS/XS/S fallback gate/up and down projections decode a lattice grid
inside the native reader. On actual Qwen3.8-27B TP4 rank0 weights this costs
50–53us for gate/up and 28–30us for down at M8. The magnitude alphabet is
only `{8,25,43}`. Expand each signed weight to one of six nibble values at
load time, then use the shared-activation DMV skeleton and a register lookup.
The IQ2 grids and formulas follow gguf-py/llama.cpp (MIT).

The resident plane uses four bits/weight plus eight metadata bytes per K128
row: the original FP16 `d` and eight local scale nibbles. Group32 local
scales are duplicated for the group16 decoder. `grid * ((0.5+nibble)/4)`
is exactly representable in FP16; multiplying by original `d` rounds only
at the final reconstructed FP16 weight. Accumulation remains FP32. There
is no conversion of the combined coefficient to FP16 in the M8 reader.

Other M values restore the original canonical packets, not a requantized
weight. An inverse magnitude table has `3^8` entries: it reconstructs the
original grid index, while nibble signs reconstruct the original sign mask.
The inverse tables are shared across layers and used only for restoration.
Their total size for three types is39366 bytes. M8 requires no shared grid.
Admission is limited to measured TP4 gate/up and down geometries; other
roles retain their existing paths. `sm70_gguf.iq2_signed_nibbles` defaults
on and contributes to the graph hash for an independent control arm.

## Measurements

Real rank0 blocks are measured at M8 with cold rotating weight banks,
graph replay and balanced ABBA readings at 1290MHz SM/877MHz memory.
These measurements are distinct from model savings. The ordinary wheel
from source `cfd4f5db47` passes 129 GPU/codec checks. All twelve real mixed
pairs and seven down projections match the existing native output bits
at three activation amplitudes. Their weighted sum drops 0.8403→0.7257ms,
which is 0.1146ms of isolated operator saving.

| Projection | Types | Existing us | Nibble us | Nibble weight MB | Weight GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| gate/up | IQ2_XS / IQ2_XXS | 50.287 | 41.620 | 25.068 | 602.3 |
| gate/up | IQ2_S / IQ3_S | 52.969 | 50.556 | 22.282 | 440.7 |
| down | IQ2_XS | 29.052 | 23.727 | 12.534 | 528.2 |
| down | IQ2_S | 29.755 | 23.712 | 12.534 | 528.6 |

The model route uses KW8/TN2/split1. Pairs preserve the existing eight
contiguous K reductions and FP16 SiLU activation boundary; down preserves
eight interleaved K reductions. Mixed IQ3 operands retain the original
scale and final FP16 reconstruction. Dot products accumulate in FP32.
IQ2 mixed with IQ4_XS remains excluded until its original scale precision
is preserved. The nineteen measured projections contain no such pair.

Official dequantization to FP16 matches every expanded IQ2 weight exactly.
CPU codec checks cover negative `d` and FP16 subnormals. GPU checks cover
canonical packet restoration, changed-input graph replay, mixed-type pairs,
bitwise native M8 output, and bitwise fallback output at M1/2/4/16/32/512.

The first version saved 0.239ms in isolated operators but changed reduction
and activation rounding. Its same-wheel model comparison saved 0.148ms/round
at 1K and 0.167ms at 8K, while emitted-token latency worsened
5.373→5.597ms and 5.651→5.734ms because tokens per round declined.
That version is not promoted. The arithmetic-compatible revision above
is undergoing a new same-wheel model comparison; it has no qualified
model-speed claim yet. Both versions' measurements remain in the data file.

The complete thirty-tensor IQ2 inventory would add46.797MiB/card relative
to canonical storage alone. This phase admits twenty-three gate/up/down
source tensors; mixed companions and release of old banks also affect the
actual model budget. The admitted IQ2 tensors add38.516MiB relative to
canonical alone; record measured resident memory when integrating.

## Current complete-round ledger

The qualified unprofiled reference is16.176ms at1K and16.958ms at8K on four
V100-SXM2-32GB cards with full NVLink,1290/877MHz, CUDA12.8 and Torch2.10.0+cu128.
Projection planes, collective/norm fusion and page832 draft split attention
are active. The following single diagnostic trace predates the draft split
dispatch correction and includes projection and collective/norm integration.
Trace envelopes are separate from unprofiled complete-round latency.

Rank0 target-start to next target-start is18.142ms under node tracing:
target graph14.270ms, tail3.872ms. Target projection service is8.492ms:

| Target role | Calls/round | Service ms | us/call | Weight GB/s |
| --- | ---: | ---: | ---: | ---: |
| Plane gate/up | 50 | 2.308 | 46.155 | 451.4 |
| Native fallback gate/up | 14 | 0.875 | 62.512 | 236.3 |
| down | 64 | 1.807 | 28.238 | 378.3 |
| qkvz+a/b | 48 | 1.638 | 34.122 | 290.9 |
| GDN out | 48 | 0.976 | 20.337 | 193.4 |
| Attention q/k/v | 16 | 0.582 | 36.395 | 241.3 |
| Attention o | 16 | 0.306 | 19.125 | 200.0 |

Tail service includes draft GEMM0.995ms, draft attention0.525ms, target
head0.272ms, draft shared head0.273ms, sampling/sorting0.169ms and
communication0.201ms. Its remaining0.817ms is recorded as other tail work.
The target graph has1.177ms of inter-kernel gaps; the tail has0.619ms.
These service and gap values are a profile attribution, not additive
predictions for an unprofiled speedup. The goal of12ms has not been reached.

The draft attention dispatch issue is independent: actual page832 inputs
pass the layout guards, but an unrelated missing native BMHD symbol gated
the packaged Triton split path. Both earlier policy arms used the general
paged kernel. The corrected route passes a same-wheel sixteen-prompt model
control: complete rounds improve16.607→16.176ms at1K and17.700→16.958ms at8K.
Mean KL is5.84e-6 and top-1 agreement is100% over128 matched logit rows.
Natural EOS and the four-request health check pass. These are independent
draft-side savings; they are not attributed to IQ2 expansion.

Retain rejected controls: two copies of the IQ3 shared grid, common mixed
loops, folded coefficients, shared reduction swizzles and a shorter K loop
did not improve the full215-projection graph. The shorter loop gives
5.689→5.742ms with bitwise equal output, and is not admitted. A dynamic
barrier allocation is not the occupancy limit here: Volta exposes64 block
barriers per SM, so sixteen allocated barriers allow four blocks while
register and warp limits are tighter. Equal-byte group-major planes improve
the215-projection graph only5.698→5.643ms. A smaller register fragment,
additional split-K and a specialized pair kernel do not improve the complete
graph. None of these controls is admitted.

A same-input projection-only graph quantifies instrumentation overhead:
GDN out averages11.24us without profiling and13.23us with graph-node tracing.
The model trace still takes20.34us. The difference cannot be claimed as an
end-to-end opportunity until its execution context is reproduced. Controls
changing shared-memory/cache preferences and touching large page ranges do
not reproduce that full difference.
