# SM70 TP4 push allreduce and Gemma RMSNorm

The compiler can fuse a TP4 allreduce followed by the DFlash2 Gemma
residual normalization boundary on a fully connected NVLink group of four
SM70 devices. No additional configuration flag is required.

For eight FP16 rows with hidden size 5120, FP32 residuals, and FP16 or FP32
normalization weights, one kernel pushes the peer contributions, adds the
residual, and normalizes the result. Each row uses five CTAs. The partial
variance and inverse RMS use local device flags; peer payloads occupy a
separate channel from ordinary allreduce and HC collectives. Unsupported
shapes, weight types, and communicators use the existing collective and
normalization kernels.

The residual preserves the original FP32 addition and rank reduction order.
The variance reduction order changes, so kernel comparisons use a dense
FP64 reference and model admission also requires numerical and acceptance
checks.

Validation uses CUDA 12.8, Torch 2.10, four V100-SXM2-32GB devices at 300 W,
TP4, Qwen3.8-27B NVFP4 with its original FP8 LM head, and DFlash2 draft7.
Model measurements keep the maximum length at 262144, GPU memory utilization
at 0.8, FP8 E4M3 target KV storage, and single-request 1024/8192-token inputs.
