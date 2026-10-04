# FP32 logits for the SM70 channel-FP8 head

For the supported DFlash2 channel-FP8 LM head, SM70 TurboMind writes FP32
logits for verification widths from one through eight. This avoids the final
FP16 rounding that can make nearby vocabulary scores equal. The checkpoint
weights remain per-channel FP8.

Dispatch uses the prepared head's capabilities and shape: FP16 activations,
hidden size 5120, and a 62080-column vocabulary shard. Other shapes retain
their existing implementation.

Numerical validation uses fixed teacher-forcing prefixes and an independent
FP64 dense multiplication of the original FP8 codes and BF16 channel scales.
The gate requires mean KL at most 0.001, p99 at most 0.01, maximum at most
0.05, top-1 agreement of at least 99%, and maximum logit error of 0.5.
Acceptance and timing use eight fixed-seed 600-token prompts at each input
length, together with a concurrency smoke.

The model workload uses CUDA 12.8, Torch 2.10, four V100-SXM2-32GB GPUs at
300 W, TP4, original Qwen3.8-27B NVFP4, DFlash2 draft7, maximum length 262144,
memory utilization 0.8, FP8 E4M3 target KV storage, and 1024/8192-token inputs.
