# SM70 DFlash2 small-M projections

The TP4 DFlash2 BF16 emulation path can prepare an additional packed layout
of its loaded FP16 projection weights. At eight query rows, supported shapes
use a dedicated Volta kernel with FP32 accumulation and an ordered local
reduction. Other row counts use the existing dense matrix product.

The packed layout preserves every loaded weight bit. Dispatch requires
SM70, the DFlash2 projection marker, contiguous FP16 matrices, no bias, and
one of the measured projection shapes. Batch-invariant execution uses its
existing implementation.

Operator validation compares changing graph inputs against dense FP32
matrix multiplication and checks the original weight storage and dynamic
prefill dispatch. Model admission additionally requires fixed-prefix
numerical comparisons against a dense FP32/FP64 reference and matched
acceptance measurements.

The serving workload uses CUDA 12.8, Torch 2.10, four V100-SXM2-32GB GPUs at
300 W, TP4, original Qwen3.8-27B NVFP4 with FP8 head, DFlash2 draft7, maximum
length 262144, memory utilization 0.8, FP8 E4M3 target KV storage, and
1024/8192-token inputs.
