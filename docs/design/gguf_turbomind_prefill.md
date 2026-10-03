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
M range, graph support and an explicit missing-operator reason. Default routing
is being selected from measurements; declaring a candidate does not change the
model loading path.

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
prefill cutoff is insufficient. TP4 projections, additional affine types,
installed-wheel validation and the final default selection remain pending.
Grouped MoE retains the graph-compatible TurboMind grouped route; these dense
measurements do not claim MoE FFN or model throughput.
