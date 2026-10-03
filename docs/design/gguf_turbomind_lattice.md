# Canonical GGUF lattice codebooks on SM70

IQ1_S/M, IQ2_XXS/XS/S and IQ3_XXS/S preserve codebook indices, full sign
masks and IQ1 delta polarity. Nested scale products expand into FP16.
The official MIT-licensed codebooks retain their original order and
are stored as exact `value + 128` bytes. Shared-memory lookup restores
FP16 pairs with the 1024 mantissa trick; sign bits and IQ1 deltas are
restored before FP16 scaling and FP32 mma884 accumulation.

## Canonical storage

All formats share a two-bit operand carrier. Each eight weights occupy a
16-bit packet after the standard TurboMind operand converter. This packet
contains indices and signs rather than affine integer weight values.

| Family | Packet | Group | Metadata | Total bits per weight |
| --- | --- | --- | --- | --- |
| IQ1_S | Full index and delta polarity | 32 | FP16 scale | 2.5 |
| IQ1_M | Full index and delta polarity | 16 | FP16 scale | 3 |
| IQ2_XXS | Low index byte and eight signs | 32 | FP16 scale and high index bits, uint32 | 3 |
| IQ2_XS/S | Low index byte and eight signs | 16 | FP16 scale and high index bits, uint32 | 4 |
| IQ3_XXS/S | Two low index bytes | 32 | FP16 scale, 32 signs and high indices, uint64 | 4 |

The layouts intentionally reuse SM70 operand conversion and metadata loading.
They occupy more memory than original GGUF blocks; model integration must
account for that difference before selecting resident representations.
For group32 metadata, CTA K is a multiple of 32 so sign/index positions
remain aligned across tiles. IQ1/2/3 use one decoder family with compile-time
codebook and metadata specializations, not separate GEMM implementations.

A CTA initializes its codebook once in shared memory. Existing affine/LUT
kernels retain their original shared storage size and decoding. Dense and
grouped descriptors have separate source-format tags and tuning keys.

## CPU correctness and real tensors

22 checks compare all seven formats against official reconstruction, including
exact power-of-two scales, decimal coefficient rounding, sign/parity expansion,
IQ1 delta polarity, TP4 slices across source blocks and bit-carrier round trips.

The real tensors below come from Flash-Next IQ3_XXS. Expert rows measure expert
0 only. Transcode times are single CPU measurements, include coefficient
conversion and exclude reader/header parsing; they are not GPU throughput.

| Tensor | Type | N | K | Transcode ms | Max absolute reconstruction error | Relative L2 |
| --- | --- | --- | --- | --- | --- | --- |
| `blk.0.ffn_gate_exps.weight` | IQ2_XS | 640 | 2560 | 10.25 | 0.000056386 | 0.00020640 |
| `blk.35.ffn_gate_exps.weight` | IQ3_XXS | 640 | 2560 | 3.10 | 0.000236511 | 0.00021106 |
| `blk.0.attn_gate.weight` | IQ3_S | 6144 | 2560 | 26.06 | 0.000171661 | 0.00020825 |

Expert gate/up TP4 slices retain N=160/K=2560. The dense gate retains
N=1536/K=2560. No expert-parallel substitution is required.

The first compile rejected external-linkage CUDA inline codebook variables
under whole-program compilation. Internal-linkage device arrays fix that
build contract. Seven dense GPU checks first passed all M values and graph replay. The
initial grouped fixture retained a LUT4 field name; fixing that test harness
yielded 21 passing GPU checks, including grouped, empty experts and full-graph
tracing. No decoder change was needed for that harness failure. Installed-wheel
validation remains pending; no model-level performance conclusion is drawn.

The dense reference harness now distinguishes the explicit MMQ candidate from
llama.cpp's preferred-dispatch policy. On Volta that policy favors BLAS at
large M, so the original capability-based sweeps omitted MMQ there. New
sweeps record when explicit MMQ is measured despite that policy. Earlier
missing MMQ columns are not evidence that the implementation lacks the format.

## Initial TP4 operator measurements

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, FP16 activations and
FP32 MMA accumulation. Routes use 100 ms warmup and 20 timed iterations.
Graph timing captures eight invocations for outputs with at most ten million
elements and one otherwise. Times are microseconds. AWQ uses the same
dimensions with valid group128 storage; it is a speed comparison, not
checkpoint quality equivalence.

### IQ2_XS: N=160, K=2560

| M | GGUF | AWQ | MMVQ | Explicit MMQ | DQ + cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 1 | 21.20 | 14.04 | 7.72 | unavailable | 13.52 |
| 2 | 20.36 | 13.76 | 8.18 | unavailable | 18.57 |
| 4 | 20.35 | 14.21 | 8.68 | unavailable | 20.17 |
| 8 | 21.54 | 17.44 | 11.21 | 22.12 | 14.96 |
| 16 | 21.86 | 22.34 | unavailable | 24.42 | 15.26 |
| 32 | 21.69 | 32.72 | unavailable | 28.80 | 16.22 |
| 64 | 30.36 | 33.15 | unavailable | 37.17 | 17.48 |
| 128 | 66.66 | 26.02 | unavailable | 64.42 | 21.48 |
| 512 | 67.06 | 26.39 | unavailable | 86.12 | 47.39 |
| 2048 | 83.13 | 71.81 | unavailable | 234.47 | 85.96 |
| 8192 | 265.94 | 240.84 | unavailable | 809.08 | 164.35 |

### IQ3_XXS: N=160, K=2560

| M | GGUF | AWQ | MMVQ | Explicit MMQ | DQ + cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 1 | 17.28 | 14.03 | 7.78 | unavailable | 13.79 |
| 2 | 16.36 | 14.32 | 8.03 | unavailable | 18.83 |
| 4 | 16.35 | 14.44 | 8.88 | unavailable | 20.47 |
| 8 | 17.73 | 17.21 | 10.85 | 21.63 | 15.17 |
| 16 | 17.56 | 22.66 | unavailable | 23.67 | 15.49 |
| 32 | 17.89 | 33.18 | unavailable | 28.09 | 16.48 |
| 64 | 28.68 | 32.61 | unavailable | 36.37 | 17.73 |
| 128 | 69.84 | 25.93 | unavailable | 62.48 | 21.50 |
| 512 | 70.54 | 26.51 | unavailable | 84.56 | 47.64 |
| 2048 | 85.48 | 71.08 | unavailable | 226.62 | 85.81 |
| 8192 | 276.40 | 239.16 | unavailable | 778.04 | 164.47 |

### IQ3_S: N=1536, K=2560

| M | GGUF | AWQ | MMVQ | Explicit MMQ | DQ + cuBLAS |
| --- | --- | --- | --- | --- | --- |
| 1 | 20.91 | 15.00 | 11.60 | unavailable | 67.66 |
| 2 | 19.88 | 15.04 | 12.24 | unavailable | 68.77 |
| 4 | 20.17 | 15.23 | 14.39 | unavailable | 69.06 |
| 8 | 20.84 | 15.96 | 18.66 | 23.25 | 69.49 |
| 16 | 22.52 | 17.99 | unavailable | 26.32 | 69.86 |
| 32 | 25.86 | 22.69 | unavailable | 33.07 | 73.46 |
| 64 | 36.83 | 28.29 | unavailable | 46.87 | 77.50 |
| 128 | 45.10 | 46.50 | unavailable | 87.88 | 92.60 |
| 512 | 97.80 | 86.21 | unavailable | 188.82 | 136.02 |
| 2048 | 397.75 | 356.80 | unavailable | 644.50 | 350.99 |
| 8192 | 1217.33 | 953.96 | unavailable | 2488.83 | 916.33 |

### IQ2_XS grouped

Four distinct experts, N=160/K=2560, sorted rows with one expert assignment
per row. Router, sorting and the full FFN are excluded. The reference DQ route
sorts IDs on the host and therefore reports eager timing.

| M | GGUF | AWQ | MoE MMVQ | MoE MMQ | DQ + cuBLAS eager |
| --- | --- | --- | --- | --- | --- |
| 1 | 26.93 | 15.26 | 9.00 | unavailable | 91.49 |
| 2 | 23.55 | 15.32 | 10.46 | unavailable | 129.59 |
| 4 | 23.82 | 16.14 | 11.19 | unavailable | 206.34 |
| 8 | 23.86 | 18.44 | 12.72 | 26.64 | 200.65 |
| 16 | 25.96 | 16.22 | 23.10 | 29.82 | 192.46 |
| 32 | 26.62 | 17.43 | 41.43 | 36.11 | 213.15 |
| 64 | 36.46 | 21.71 | 79.65 | 50.57 | 241.51 |
| 128 | 33.21 | 28.08 | 166.34 | 92.88 | 245.35 |
| 512 | 42.88 | 28.17 | 764.49 | 231.76 | 249.29 |
| 2048 | 117.79 | 71.72 | 3061.01 | 830.68 | 329.88 |
| 8192 | 405.49 | 228.68 | 12186.64 | 2228.52 | 668.57 |

### IQ3_XXS grouped

Four distinct experts, N=160/K=2560, sorted rows with one expert assignment
per row. Router, sorting and the full FFN are excluded. The reference DQ route
sorts IDs on the host and therefore reports eager timing.

| M | GGUF | AWQ | MoE MMVQ | MoE MMQ | DQ + cuBLAS eager |
| --- | --- | --- | --- | --- | --- |
| 1 | 21.64 | 15.24 | 8.97 | unavailable | 94.98 |
| 2 | 20.75 | 15.29 | 10.39 | unavailable | 132.45 |
| 4 | 21.00 | 16.10 | 11.07 | unavailable | 202.19 |
| 8 | 21.11 | 18.43 | 12.38 | 26.38 | 204.54 |
| 16 | 24.77 | 16.23 | 22.46 | 29.39 | 201.73 |
| 32 | 25.63 | 17.42 | 40.66 | 35.80 | 204.49 |
| 64 | 38.94 | 21.69 | 78.88 | 49.89 | 259.48 |
| 128 | 29.73 | 28.08 | 163.14 | 91.26 | 260.25 |
| 512 | 40.18 | 28.19 | 760.22 | 225.26 | 255.44 |
| 2048 | 100.74 | 71.62 | 3011.67 | 799.80 | 330.50 |
| 8192 | 361.99 | 228.75 | 12000.56 | 2195.53 | 661.86 |

The initial fused path has substantial intermediate-M and grouped gaps.
For example, grouped M=8192 costs 405.49 us (IQ2_XS) and 361.99 us
(IQ3_XXS), versus AWQ about 228.7 us. Dense IQ3_S costs 1217.33 us
versus AWQ 953.96 us. These results require decoder and prefill work before
model integration. Static SASS for the IQ3_S CTA128/N256/K32 decoder shows
repeated 16-bit shared lookup loads. GPU hardware counters remain unavailable
under the previously recorded driver permissions, so static counts are not
claimed as runtime attribution.
