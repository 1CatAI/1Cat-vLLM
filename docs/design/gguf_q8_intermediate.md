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
No model dispatcher selects the candidate before qualification.

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
resource counts establish no speed benefit; GPU timing is still pending.
