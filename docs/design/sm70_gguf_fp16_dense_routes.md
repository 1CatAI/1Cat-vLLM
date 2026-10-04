# Floating GGUF projections and SM70 dense routes

GGUF is a checkpoint format, not an operator capability. A complete floating
projection should use the same dense linear lifecycle as a safetensors weight.
The GGUF loader previously retained `GGUFLinearMethod` even when every shard
had already become an ordinary FP16 matrix. This prevented the Qwen4Exp
post-load preparation from selecting its row GEMV and independent a/b route.

After resolving the descriptors, loading now restores a dense method only when
all shards are floating, use the destination dtype, and form the complete
logical matrix. Mixed quantized projections and projections with input layout
transforms retain their GGUF method. Shards concatenate in logical projection
order. A single contiguous matrix shares its existing storage.

The loader then lets the model prepare newly discovered dense projections,
before attention processing and graph capture. Existing Qwen4Exp methods and
their already packed weights are retained.

The small GDN a/b projection uses the existing FP32 row reduction at M=2..16,
independently of the quantized QKV projection. Other dense projection tuning
remains separate. The packed router also accumulates and reduces entirely in
FP32; its admission no longer requires cuBLAS's FP16 partial-reduction switch.
The shared-expert kernel's FP16 partial policy is unchanged.

## GDN a/b operator measurements

Source `9d7a11620e`, packaged runtime `1.5.2.dev464+g9d7a11620.precompiled`;
Torch 2.10 CUDA 12.8, Python 3.12.3, driver 580.173.02. GPU: one V100 SXM2
32 GB. The ordinary packaged extension is used, with no external kernel DSO.

The benchmark reads all 36 GDN layers from Flash-Next GSQ-RCO IQ3_S. It takes
TP4 rank 0's 12 rows from each original BF16 a/b tensor and uses the normal
FP16 load conversion. The old path runs the two separate dense projections
and concatenates their outputs. The new path runs one complete 24-row matrix.
Activations are synthetic FP16. Correctness uses FP32 matmul of these loaded
weights. The benchmark asserts that the admitted route never calls dense
GEMM during its route check.

Graph timing uses five samples of 100 replays, after warmup. The table is the
median service time of the 36-projection chain, without model attention,
experts, scheduling or communication. The small weight set can fit in cache;
these are operator measurements, not a model throughput result.

| M | Old chain (µs) | New chain (µs) | Saved per chain (µs) | New per layer (µs) | Maximum absolute error | Maximum relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 313.805 | 69.581 | 244.224 | 1.933 | 0 | 0 |
| 5 | 2756.188 | 87.695 | 2668.493 | 2.436 | 7.63e-6 | 7.64e-6 |
| 10 | 2806.067 | 103.885 | 2702.182 | 2.886 | 1.22e-4 | 5.92e-5 |
| 16 | 2895.462 | 113.490 | 2781.972 | 3.152 | 1.22e-4 | 4.85e-5 |
| 20 | 670.863 | 254.853 | 416.010 | 7.079 | 1.22e-4 | 4.42e-5 |

M=20 retains dense GEMM; its improvement comes from eliminating the separate
shards and concatenation. The M=5 operator chain suggests about 2.67 ms less
work per target verification round. End-to-end C1/C4 must measure the actual
round and acceptance change separately.

Reproduce with `benchmarks/kernels/benchmark_gguf_fp16_ba.py MODEL.gguf
--rank 0 --output RESULT.json`. Use `--projection router` for the independent
router projection comparison.

## Router operator measurements

Source `8a649367f0`, packaged runtime `1.5.2.dev466+g8a649367f.precompiled`;
hardware and dependencies match the a/b comparison. All 48 real BF16 router
matrices are loaded as FP16. Their working set exceeds V100's L2 cache.
The old path uses dense GEMM; the candidate selects the existing row GEMV
at M=1 and the packed FP32 router at M=5/10. Route checks reject dense
fallback in those admitted cases.

| M | Old chain (µs) | New chain (µs) | Saved per chain (µs) | Maximum absolute error | Maximum relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 390.472 | 244.654 | 145.818 | 2.44e-4 | 4.45e-5 |
| 5 | 539.628 | 336.169 | 203.459 | 2.44e-4 | 3.69e-5 |
| 10 | 543.222 | 371.077 | 172.145 | 2.44e-4 | 3.59e-5 |

M=16/20 retain dense GEMM, with about 0.2% timing variation between arms.
The M=5 chain suggests another 0.20 ms less projection work per verifier
round. This comparison does not alter or measure router top-k selection.

## HC and remaining checks

The existing trace already contains 48 target router top-k calls per round.
HC and router dense methods were prepared at model construction; their missing
fast kernels have additional runtime causes. In particular, batched HC's IPC
collective requires a fully connected TP4 NVLink topology. Dense restoration
does not satisfy that hardware requirement. Local HC compute with the
admitted TP collective is a separate change.

Loading and route preparation have focused CPU regressions, including mixed
format rejection, layout rejection, shard order, storage sharing and router
FP32 admission. The combined HC/dense model C1/C4 comparison remains pending.
No end-to-end speedup is claimed here.
