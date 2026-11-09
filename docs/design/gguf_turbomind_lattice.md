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

## Word lookup and canonical FP32 prefill

Aligned word lookup reads a whole four/eight-byte grid row once and restores
FP16 pairs in registers. Matched IQ3_S CTA128/N256/K32 static SASS changes
40 table halfword loads into 20 word loads, while MMA instruction counts
remain unchanged. Dense and grouped timing improves about 4–8%, leaving
substantial grouped gaps. These static counts are not hardware-counter data.

Canonical lattice dequantization reuses that decoder, writes transient
FP16 `[K,N]` scratch, and calls cuBLAS with explicit FP32 accumulation and
reductions. Output and scratch are caller-owned and graph-safe. All seven
formats reconstruct canonical FP16 weights elementwise, match the FP32 GEMM
oracle, and pass graph replay/full-graph tracing. The combined suite passes
92 GPU checks with one non-SM70 skip. Framework reporting declares calibrated M intervals. Unmeasured local
shapes retain fused MMA with `local_shape_has_no_prefill_calibration`.

| Type | N | K | M | Word fused | Canonical DQ + FP32 GEMM | AWQ |
| --- | --- | --- | --- | --- | --- | --- |
| IQ2_XS | 160 | 2560 | 1 | 21.22 | 16.03 | 13.67 |
| IQ2_XS | 160 | 2560 | 128 | 64.08 | 14.60 | 25.78 |
| IQ2_XS | 160 | 2560 | 512 | 64.90 | 25.25 | 26.25 |
| IQ2_XS | 160 | 2560 | 2048 | 77.60 | 80.09 | 70.71 |
| IQ2_XS | 160 | 2560 | 8192 | 259.29 | 160.01 | 240.61 |
| IQ3_XXS | 160 | 2560 | 1 | 16.77 | 13.46 | 13.28 |
| IQ3_XXS | 160 | 2560 | 128 | 68.12 | 12.00 | 25.97 |
| IQ3_XXS | 160 | 2560 | 512 | 68.42 | 20.76 | 26.18 |
| IQ3_XXS | 160 | 2560 | 2048 | 81.07 | 77.16 | 71.47 |
| IQ3_XXS | 160 | 2560 | 8192 | 265.52 | 157.15 | 240.87 |
| IQ3_S | 1536 | 2560 | 1 | 20.08 | 34.84 | 15.25 |
| IQ3_S | 1536 | 2560 | 128 | 43.25 | 46.83 | 46.64 |
| IQ3_S | 1536 | 2560 | 512 | 91.07 | 124.56 | 85.81 |
| IQ3_S | 1536 | 2560 | 2048 | 379.42 | 282.15 | 358.34 |
| IQ3_S | 1536 | 2560 | 8192 | 1155.28 | 769.95 | 960.82 |

Canonical DQ at M=8192 costs 769.95 us for IQ3_S, versus AWQ 960.82 us,
with unchanged FP16 activations and FP32 accumulation. Small expert shapes
have a nonmonotonic crossover: at M=128/512 DQ wins clearly, but at M=2048
its difference from fused MMA is small. Intermediate and boundary points
must be measured before setting default bands.

Grouped M=8192 improves from 405.49 to 385.47 us (IQ2_XS) and from 361.99
to 342.75 us (IQ3_XXS), while AWQ remains about 228.4 us. Grouped continues
to use the fused operator; dense DQ timing does not claim grouped performance.

## Calibrated dense dispatch

The complete TP4 sweep adds M=256/1024/4096 to the standard M points.
IQ2_XS and IQ3_XXS at N=160/K=2560 select canonical DQ + FP32 GEMM for
M=8–1024 and M>=4096. M=2048 remains fused: the IQ2_XS DQ path is slightly
slower, and the IQ3_XXS difference is small. IQ3_S at N=1536/K=2560 selects
DQ for M>=2048. No interpolation changes unmeasured descriptors.

Scratch reuses the affine prefill pool, bounded at 178,257,920 bytes per
device, shared across layers and allocated before graph capture. Its views
retain the pool lifetime. Missing operators, unmeasured shapes and allocation
failure have explicit capability reasons and retain fused MMA. Sequential
layer execution on the device stream is required for shared scratch reuse.
FP16 activations, FP16 reconstructed weights and explicit FP32 cuBLAS
accumulation/reductions remain unchanged.

| Type | M | Fused us | Canonical DQ us | AWQ us | Default |
| --- | --- | --- | --- | --- | --- |
| IQ2_XS | 1 | 19.76 | 15.97 | 13.77 | Fused |
| IQ2_XS | 2 | 19.73 | 22.31 | 14.02 | Fused |
| IQ2_XS | 4 | 19.86 | 22.32 | 14.14 | Fused |
| IQ2_XS | 8 | 21.38 | 13.69 | 17.14 | DQ |
| IQ2_XS | 16 | 20.86 | 12.83 | 22.16 | DQ |
| IQ2_XS | 32 | 21.25 | 12.90 | 32.97 | DQ |
| IQ2_XS | 64 | 29.00 | 13.38 | 32.92 | DQ |
| IQ2_XS | 128 | 64.17 | 14.62 | 25.90 | DQ |
| IQ2_XS | 256 | 64.78 | 17.18 | 25.89 | DQ |
| IQ2_XS | 512 | 64.49 | 25.31 | 26.27 | DQ |
| IQ2_XS | 1024 | 68.79 | 39.64 | 45.32 | DQ |
| IQ2_XS | 2048 | 78.81 | 80.45 | 71.56 | Fused |
| IQ2_XS | 4096 | 128.85 | 100.90 | 117.35 | DQ |
| IQ2_XS | 8192 | 257.38 | 161.32 | 239.87 | DQ |
| IQ3_XXS | 1 | 16.82 | 13.44 | 12.74 | Fused |
| IQ3_XXS | 2 | 14.82 | 19.72 | 13.09 | Fused |
| IQ3_XXS | 4 | 15.14 | 19.71 | 13.28 | Fused |
| IQ3_XXS | 8 | 16.35 | 11.10 | 17.03 | DQ |
| IQ3_XXS | 16 | 16.32 | 10.22 | 22.23 | DQ |
| IQ3_XXS | 32 | 16.25 | 10.52 | 32.76 | DQ |
| IQ3_XXS | 64 | 24.60 | 10.84 | 32.93 | DQ |
| IQ3_XXS | 128 | 67.47 | 11.96 | 25.78 | DQ |
| IQ3_XXS | 256 | 67.99 | 14.60 | 26.03 | DQ |
| IQ3_XXS | 512 | 68.19 | 20.63 | 26.14 | DQ |
| IQ3_XXS | 1024 | 68.26 | 35.97 | 45.51 | DQ |
| IQ3_XXS | 2048 | 80.04 | 75.99 | 71.07 | Fused |
| IQ3_XXS | 4096 | 132.36 | 97.93 | 117.56 | DQ |
| IQ3_XXS | 8192 | 263.33 | 157.02 | 239.78 | DQ |
| IQ3_S | 1 | 20.19 | 35.46 | 15.07 | Fused |
| IQ3_S | 2 | 19.35 | 33.27 | 15.14 | Fused |
| IQ3_S | 4 | 19.42 | 33.52 | 15.23 | Fused |
| IQ3_S | 8 | 20.33 | 33.86 | 16.03 | Fused |
| IQ3_S | 16 | 21.87 | 34.46 | 18.17 | Fused |
| IQ3_S | 32 | 24.82 | 36.42 | 22.78 | Fused |
| IQ3_S | 64 | 29.29 | 39.98 | 28.22 | Fused |
| IQ3_S | 128 | 43.30 | 46.78 | 46.54 | Fused |
| IQ3_S | 256 | 73.64 | 67.83 | 51.04 | Fused |
| IQ3_S | 512 | 90.94 | 124.18 | 86.36 | Fused |
| IQ3_S | 1024 | 193.98 | 200.09 | 167.68 | Fused |
| IQ3_S | 2048 | 380.63 | 284.38 | 357.31 | DQ |
| IQ3_S | 4096 | 631.48 | 449.40 | 526.80 | DQ |
| IQ3_S | 8192 | 1173.04 | 775.53 | 962.25 | DQ |

A narrow grouped tile experiment passed all 21 lattice GPU checks but had
format-dependent results. IQ2_XS grouped M=8192 improved from 385.47 to
321.14 us; AWQ was 228.58 us. IQ3_XXS regressed from 342.75 to 487.37 us
at the same point, so its new narrow candidates were rejected. Only the
IQ2_XS grouped candidates remain, restricted to N<=256 and M<=64 or M>=512. At M=128 the new IQ2_XS
candidate costs 36.17 us versus the previous 32.12 us, so intermediate M
retains the existing candidates. The final candidate
set passes 95 GPU checks with one non-SM70 skip, including interval
boundaries, shared scratch and graph tracing. The ordinary wheel also passes 56 CPU and 95 GPU checks with one
non-SM70 skip in a separate installed environment. Grouped performance still
needs work before model integration; dense DQ results do not establish full
MoE or model throughput.

### Grouped sweep before the intermediate-M exclusion

The table uses the same four distinct experts and all standard M values.
Only IQ2_XS received new narrow candidates. The M=128 regression motivates
excluding them for M=65–511. The installed wheel restores M=128 to
31.71 us, versus AWQ 28.10 us.

| Type | M | GGUF us | AWQ us | MoE MMVQ us | MoE MMQ us |
| --- | --- | --- | --- | --- | --- |
| IQ2_XS | 1 | 23.72 | 15.18 | 8.99 | unavailable |
| IQ2_XS | 2 | 22.83 | 15.36 | 10.81 | unavailable |
| IQ2_XS | 4 | 22.91 | 16.33 | 12.19 | unavailable |
| IQ2_XS | 8 | 23.03 | 18.42 | 12.35 | 26.67 |
| IQ2_XS | 16 | 23.39 | 16.28 | 22.18 | 29.82 |
| IQ2_XS | 32 | 25.38 | 17.45 | 40.76 | 36.07 |
| IQ2_XS | 64 | 27.52 | 21.68 | 78.57 | 50.66 |
| IQ2_XS | 128 | 36.17 | 28.06 | 164.86 | 92.55 |
| IQ2_XS | 512 | 40.15 | 28.20 | 765.42 | 232.31 |
| IQ2_XS | 2048 | 93.07 | 71.99 | 3032.98 | 829.88 |
| IQ2_XS | 8192 | 322.27 | 228.83 | 12116.46 | 2228.01 |
| IQ3_XXS | 1 | 21.11 | 16.81 | 9.75 | unavailable |
| IQ3_XXS | 2 | 20.33 | 15.40 | 10.58 | unavailable |
| IQ3_XXS | 4 | 20.38 | 16.35 | 11.35 | unavailable |
| IQ3_XXS | 8 | 20.49 | 18.39 | 12.42 | 26.43 |
| IQ3_XXS | 16 | 23.97 | 16.61 | 22.16 | 29.66 |
| IQ3_XXS | 32 | 24.72 | 17.46 | 40.50 | 35.83 |
| IQ3_XXS | 64 | 37.61 | 21.91 | 78.62 | 50.07 |
| IQ3_XXS | 128 | 28.15 | 28.36 | 164.35 | 91.94 |
| IQ3_XXS | 512 | 36.97 | 28.22 | 758.71 | 226.34 |
| IQ3_XXS | 2048 | 94.59 | 71.97 | 3024.26 | 800.68 |
| IQ3_XXS | 8192 | 342.03 | 228.77 | 11967.08 | 2195.65 |

## Installed-wheel validation

The ordinary SM70 wheel contains the normal `_C` and `_C_gguf` components.
The source-built, packaged and installed `_C` hashes match and contain no
RPATH/RUNPATH. Python 3.12.3, Torch 2.10.0+cu128, GGUF 0.19.0,
Transformers 5.18.0, XGrammar 0.2.0 and Tilelang 0.1.10 were used.

Wheel SHA256:
`da76d05691d0226bdd6654dc6809d943dc6049b043ce899226f5d5284c680600`.
Core SHA256:
`5ce398153efd6a51b1b8ed686c6f9f075c52214bc3dfbf6b4a89b0dfa183a90f`.

| Operator | M | Installed us | AWQ us |
| --- | --- | --- | --- |
| IQ2_XS grouped, four experts | 64 | 29.38 | 21.75 |
| IQ2_XS grouped, four experts | 128 | 31.71 | 28.10 |
| IQ2_XS grouped, four experts | 8192 | 322.28 | 228.11 |
| IQ3_XXS dense, canonical DQ | 128 | 12.00 | 25.94 |
| IQ3_XXS dense, canonical DQ | 8192 | 159.60 | 239.22 |
| IQ3_S dense, fused | 128 | 43.65 | 46.55 |
| IQ3_S dense, canonical DQ | 8192 | 771.64 | 949.15 |

No model adapter, tokenizer, routing algorithm or PLE integration is changed
by this operator layer. Full-model quality and throughput remain separate
work. Grouped still has a material gap at large M and requires further
operator work before connecting Flash-Next.
