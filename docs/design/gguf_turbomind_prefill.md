# Canonical GGUF dequantization and FP32-accumulating prefill

The affine dequantization operator reads TurboMind's packed U2/U4/U8 and
bit-plane carriers directly. It reuses the register decoders from mma884 and
writes a transient row-major `[K,N]` FP16 workspace. It retains no second copy
of the quantized weights or persistent FP16 copy of every layer.

The GEMM operator consumes that workspace with explicit `CUBLAS_COMPUTE_32F`
and `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`, restoring the handle's
previous math mode after the call. Activations and reconstructed weights stay
FP16; accumulation and reductions stay FP32. Output and scratch tensors are
caller-owned, so allocation can occur before graph capture.

The mixed-precision kernel lifecycle declares this prefill candidate alongside
the fused affine operator. Startup reporting includes family, source format,
M range, graph support and an explicit missing-operator reason. Calibrated physical layouts select the prefill candidate by M range;
unmeasured layouts retain fused MMA and report the missing calibration.
Model loading remains a separate layer.

## Correctness

Eight GPU checks pass: all seven physical affine storage contracts reconstruct
their canonical FP16 weights exactly, including negative scales and zero-scale
constant blocks. Dense GEMM matches the FP32 oracle, CUDA graph replay and
full-graph tracing pass, and the candidate appears in startup reporting.

The first run passed numerical and graph checks but exceeded Dynamo's shared
code-object compilation limit across parameterized cases. Independent cache
reset between contracts resolved the test-harness failure; no operator change
was needed. CPU import/transcode checks also pass (23 tests).

## First real-shape results

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128; 100 ms warmup and 20 timed
iterations. Small outputs capture eight operator invocations per replay;
larger outputs use one to bound graph-pool memory. Times below are microseconds
and include canonical dequantization and GEMM. AWQ has the same N/K with valid
group128 storage, and is a performance comparison rather than a quality-equivalent
checkpoint. Shapes come from Qwen3.8-27B UD-Q4_K_M and are full projections.

| Type | N | K | M | Fused GGUF | Canonical DQ + FP32 GEMM | AWQ |
| --- | --- | --- | --- | --- | --- | --- |
| Q3_K | 17408 | 5120 | 512 | 1540.09 | 1473.93 | 1374.66 |
| Q3_K | 17408 | 5120 | 2048 | 5647.31 | 4446.16 | 5026.71 |
| Q3_K | 17408 | 5120 | 8192 | 22910.26 | 16442.73 | 20453.79 |
| Q5_K | 6144 | 5120 | 512 | 637.40 | 656.46 | 543.62 |
| Q5_K | 6144 | 5120 | 2048 | 2126.69 | 1598.21 | 1825.43 |
| Q5_K | 6144 | 5120 | 8192 | 9253.73 | 5987.02 | 7445.56 |
| Q6_K | 5120 | 6144 | 512 | 484.55 | 511.97 | 428.27 |
| Q6_K | 5120 | 6144 | 2048 | 1949.49 | 1568.26 | 1743.82 |
| Q6_K | 5120 | 6144 | 8192 | 8108.29 | 5830.91 | 7070.93 |

At M=2048–8192 this candidate is faster than both fused GGUF and same-shape
AWQ for all three projections. M=512 differs by shape: the wide Q3 projection
improves, while Q5/Q6 retain a faster fused path. Thus a single unmeasured
prefill cutoff is insufficient. TP4 projection measurements and default selection follow below.
Additional affine types and installed-wheel validation remain separate gates.
Grouped MoE retains the graph-compatible TurboMind grouped route; these dense
measurements do not claim MoE FFN or model throughput.

## TP4 projection sweep and default selection

All four local projections were measured at M=1, 2, 4, 8, 16, 32, 64,
128, 512, 2048 and 8192. The table shows the prefill rows in microseconds;
all three routes use the same repeated-invocation CUDA graph protocol.

| Type | N | K | M | Fused GGUF | Canonical DQ + FP32 GEMM | AWQ |
| --- | --- | --- | --- | --- | --- | --- |
| Q3_K | 4352 | 5120 | 512 | 478.10 | 367.58 | 466.73 |
| Q3_K | 4352 | 5120 | 2048 | 1559.58 | 1224.03 | 1401.68 |
| Q3_K | 4352 | 5120 | 8192 | 5865.98 | 4172.75 | 5318.14 |
| Q5_K | 1536 | 5120 | 512 | 166.62 | 183.62 | 152.25 |
| Q5_K | 1536 | 5120 | 2048 | 663.12 | 560.05 | 565.48 |
| Q5_K | 1536 | 5120 | 8192 | 2280.70 | 1427.20 | 1892.04 |
| Q4_K | 4352 | 5120 | 512 | 454.07 | 370.55 | 465.79 |
| Q4_K | 4352 | 5120 | 2048 | 1545.34 | 1240.84 | 1403.08 |
| Q4_K | 4352 | 5120 | 8192 | 5480.45 | 4229.84 | 5300.79 |
| Q6_K | 5120 | 1536 | 512 | 126.41 | 128.60 | 112.05 |
| Q6_K | 5120 | 1536 | 2048 | 509.49 | 403.56 | 454.81 |
| Q6_K | 5120 | 1536 | 8192 | 2129.00 | 1514.65 | 1891.64 |

Q3_K TP4 and Q4_K TP4 select canonical DQ at M>=512. Q5_K and
Q6_K full/TP4 layouts select it at M>=2048. The full Q3_K projection
also selects it at M>=512. These decisions apply to identical physical
bit width, group size and local dimensions, independent of index values;
other layouts retain fused TurboMind until measured.

Preparation allocates one shared, bounded 178257920-byte FP16 workspace per
device for these calibrated layouts, with layers retaining views. It stores
transient dequantization, not persistent per-layer FP16 weights. Allocation
happens before graph capture. Missing operators, uncalibrated dimensions and
allocation failures each appear as capability reasons; failure retains fused
MMA. No runtime environment variables or lower accumulation precision are
introduced. Grouped MoE continues to use the native grouped operator.
