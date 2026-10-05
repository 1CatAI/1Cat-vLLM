# Routed Q8 intermediates for GGUF experts

The existing integer expert path quantizes each token once for gate/up, but
its down output CTAs independently quantize the same routed FP16 intermediate.
For a local output width of 2560, this repeats the intermediate encoder in
80 CTAs. The proposed gate/up epilogue writes standard group32 Q8_1 blocks
once per routed row. Down copies these blocks into shared memory and retains
its existing integer dot, FP16 down boundary and FP32 weighted reduction.
Both paths issue three kernels per layer, including input activation encoding.

Gate/up keeps its FP16 gate/up, SiLU and multiplication boundaries before
encoding. A CTA owns 32 contiguous output rows, so its first warp can encode
one complete group without communication between CTAs. The 16-lane K
partition retains the existing reduction tree. Eight- and four-lane variants
are benchmark candidates; their FP32 summation trees require separate checks.
The unified expert capability policy admits the qualified 16-lane path at
M5/M20. Other batches keep the FP16 intermediate.

The operator accepts either the existing FP16 intermediate or contiguous
Q8_1 blocks. Existing callers keep the FP16 path. The decoder and activation
encoder remain shared with other GGUF integer operators; no independent
codebook or rounding implementation is introduced.

The focused tests compare the 16-lane output bytes with the retained FP16
intermediate and existing encoder, check TP4 down outputs against official
weights, and replay changed inputs and routes. The microbenchmark reads
actual TP4 GGUF banks and measures M5/M20, individual stages and the complete
three-launch pipeline. Every timed operation follows a 32 MiB cache eviction;
reported graph event times include the common event boundary overhead.
Synthetic top10 routing is explicitly labeled and cannot establish the real
model's routing distribution. Whole-model speed and quality remain required.

At the initial compiler checkpoint, IQ3_S gate/up uses 56 registers in the
FP16 and 16/8-lane Q8 variants, and 40 registers in the four-lane variant.
Q2_0 down uses 40 registers with repeated quantization and 38 when reading Q8.
All inspected variants have zero stack and local-memory allocation. These
resource counts establish no speed benefit; GPU timing is reported below.

## Cold-cache qualification

The source-complete SM70 wheel passes 73 GPU checks, including exact 16-lane
Q8 byte comparisons, official weight references and changed-input graph replay.
Nineteen CPU capability checks cover the operator protocol and rejection reasons.
Torch 2.10, CUDA 12.8 and a V100 at observed 1530/877 MHz are used. Every
operation follows a 32 MiB eviction; top10 routing is seeded synthetic routing.

| Gate / down | M5 old / Q8 pipeline, us | M20 old / Q8 pipeline, us |
|---|---:|---:|
|IQ3_S / IQ4_NL|76.80 / 67.58|200.70 / 182.02|
|IQ3_XXS / IQ4_NL|72.70 / 63.49|184.83 / 171.01|
|IQ2_S / Q2_0|74.75 / 63.49|191.49 / 159.74|

All pipelines issue three kernels. The 16-lane schedule wins; eight and four
lanes are not admitted. IQ3_S M5 gate/up remains about 45.06 us, or 384 GB/s
using unique original-weight bytes. This does not meet the 450 GB/s cold-cache
target. The gain comes mainly from avoiding repeated down input quantization.
Model quality and round latency must be measured separately.

An exact XOR/add replacement for packed sign subtraction passes 73 GPU
checks and exhaustive CPU checks of every codebook/sign combination. Two
paired cold repeats retain 45.06 us M5 and 117.76 us M20 Q8 gate/up times.
It is reverted because it provides no measured throughput benefit.
