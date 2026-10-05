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

## First IQ3_S measurements

The initial whole-wheel implementation uses layer17's original IQ3_S rows,
512 experts, TP4 local N160 and K2560. Random top10 routing touches49 unique
experts at M5 and168 at M20. Activation inputs are synthetic FP16 normal
samples with standard deviation0.125. Measurements use CUDA Graph replay,
eight alternating candidate/control epochs and100 replays per epoch.
The native control is the current original-block FP32 grouped vector dot;
its activation gather and separate SiLU are prepared outside the timed region.
The dp4a candidate includes direct routing and the FP16 SiLU/multiply boundary.

| M | Native gate/up | dp4a fused | Q8_1 plus fused | Unique-source bandwidth |
|---|---:|---:|---:|---:|
|5|73.82us|42.85us|44.06us|403.94GB/s|
|20|210.37us|120.90us|121.99us|490.90GB/s|

M5 runs at1290MHz core and877MHz memory. M20 boosts to1530MHz in later
epochs, so these rows are not a controlled clock comparison. The M5 result
does **not** meet the450GB/s expansion gate. Its49 active experts contain
17,310,720 bytes: the gate requires at most38.47us for this routing sample.
One M5 combined epoch reached241us; it remains in the raw samples.
Standalone quantization replay is host-starved and is not a reliable measure
of its GPU cost; combined minus fused is approximately1.2us.

The actual NVFP4 TP4 weights, prepared by the production converter, measure
41.25us for plan plus W13 at M5. This is a separately quantized weight
representation, not an output oracle. That comparator supports at most160
routes and is explicitly omitted at M20 (200 routes).

Nine focused GPU tests pass, including changed-input graph replay. On real
IQ3_S rows the relative L2 error against official GGUF dequantization with
the same Q8_1 activations is0.00020–0.00021; against the FP16-activation
control it is0.0053–0.0054. These are projection measurements, not model
KL, top1, acceptance or end-to-end latency evidence. No model dispatch is
enabled by this initial result.
