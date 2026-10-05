# Shared GGUF dp4a path for small batches

For M at most20, quantize each activation row once to standard Q8_1 blocks:
32 signed int8 values, FP16 scale and FP16 sum. Integer-codebook readers feed
SM70 dp4a directly, with original weight and activation scales applied after
the integer dot. FP32 accumulation and the final FP16 projection boundary
remain unchanged. Activations incur the explicitly authorized int8 rounding.

The common header under TurboMind is intended for both dense and routed GGUF
projections. The first reader is IQ3_S. A routed gate/up launch reads routing
IDs directly and forms SiLU times up without activation gathering. A debug
mode exposes the separate projection boundary for the official-weight/Q8_1
oracle. It is not selected by model dispatch until calibrated and qualified.

IQ3_XXS and IQ2_S, dense formats and fused down reduction follow only after
the IQ3_S M5 expert benchmark reaches450GB/s. Every bandwidth report must give
actual weight storage and distinguish unique active-expert bytes from repeated
route reads. Gate/up-only and activation-quantization-inclusive timings are
reported separately, with kernel counts. Whole-model gains require separate
unprofiled C1/C4 measurements.

References: [llama.cpp integer dot](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh),
[Q8_1 quantization](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/quantize.cu),
[Strix rows/signs/prefetch](https://github.com/halo-box/strix-llama.cpp/pull/106),
[shared-expert fusion](https://github.com/ggml-org/llama.cpp/pull/29184).
The original MIT licenses are packaged with the GGUF implementation. Strix
wave tuning targets AMD and needs independent Volta measurements.
