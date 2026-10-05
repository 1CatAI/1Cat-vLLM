# Source-sized IQ3_S/IQ4_XS gated pairs

Forty layers in the Qwen3.8-27B IQ3_S checkpoint have different gate and up
formats. Eleven use IQ3_S and IQ4_XS, accounting for 31.509% of mixed-pair
source bytes. A joint operator can share activations and fuse the gated
epilogue while retaining each projection's original decoding formula.

`gguf_native_pair_sm70_out` instantiates the shared-activation M8/N32
skeleton with two independent readers. Both type orientations are compiled.
The IQ3_S reader reuses the signed nibble-book decoder and retains both
original scale levels. IQ4_XS uses the existing LUT4 transform, restores its
interleaved lane order, multiplies original scale metadata in FP32 and rounds
only the final weight operand to FP16. Neither reader expands checkpoint
storage. Their N32 macro records contain 110 or 136 bytes per K256 block,
respectively. Read cursors advance in K128 steps and cache metadata across
the two halves of a block, including odd starting halves.

All CTA threads load activations into padded shared rows once per K128 step.
Gate and up reuse the same activation tile. Dot products and the eight-way
split-K reduction remain FP32. The final FP16 projection rounding and gated
SiLU/multiply match the retained pair epilogue. The operator requires SM70,
FP16 operands, M8, N divisible by 32, and K divisible by 1024; source byte
counts are checked independently for both projections.

## Verification boundary

The independent CPU cursor oracle checks 10,240 actual K128 records across
layer39 and layer42, all eight split-K starting positions, and both format
orientations. Original metadata, signed indices and nibble packets match
without errors. The previously checked IQ4_XS device operand matches official
dequantization bitwise, including extreme-scale stress cases. These checks
do not establish correctness or speed of the mixed GEMM.

The normal CMake extension registers the operator; no private extension is
required. `benchmark_gguf_native_pair.py` checks actual TP4 N4352/K5120
weight slices against official FP32 dequantization and FP32 GEMM on three
inputs. It then compares canonical and native pairs in cold-L2 ABBA graph
replay, recording SM/memory clocks for each arm. Source-payload bandwidth
is a workload ratio, not an NCU memory counter. Acquire the shared GPU lock
and selected device lock before running it.

GPU GEMM numerical checks and same-session timing are pending. No model
shape is admitted by this operator change. Mixed projections retain their
canonical route until an actual-weight comparison shows a faster shape;
slower or unmeasured shapes must retain canonical dispatch.
