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

GPU GEMM checks now pass in a normal source-complete wheel with Torch 2.10,
CUDA 12.8 and V100 SXM2 32GB. Three independent M8 inputs per orientation
give native relative L2 0.000497–0.000534 against official FP32 weights and
FP32 GEMM, versus 0.000598–0.000623 for canonical. Native maximum absolute
error is 0.00390625–0.0078125; all results are finite.

| Actual TP4 slice | Clock in every timed arm | Canonical ABBA arms | Native ABBA arms | Native source bandwidth |
| --- | --- | --- | --- | --- |
| Layer39, IQ4_XS/IQ3_S | 1530/877MHz | 74.752/74.752us | 54.272/54.272us | 394.5GB/s |
| Layer42, IQ3_S/IQ4_XS | 1530/877MHz | 75.776/75.776us | 54.272/54.784us | 390.8–394.5GB/s |

Each pair contains 21,411,840 source bytes. Cold-L2 eviction is 16MiB,
with 84 graph samples per arm and 300W power limit. The first layer42 run
changed SM clocks during its first canonical arm; that arm is excluded,
and the table uses one focused repeat with stable clocks. These are
secondary-machine operator measurements, not primary-machine or end-to-end
results. Across eleven eligible layers, 20.48–21.50us per layer suggests
about 0.225–0.237ms of projection saving. No full-round gain is claimed.

Capability declarations admit only both measured orientations at
M8/N4352/K5120, FP16 operands and SM70. Source records are retained only
after that admission, adding 235,530,240 bytes per rank across eleven layers.
Actual M is resolved inside an opaque operator; other M reuse the existing
canonical mixed-projection policies, including M5/M20 direct output and
prefill workspace resolution. Disabled policy, missing operator, transformed
layouts, other types and uncalibrated shapes retain canonical dispatch with
reported reasons. The loader and fused-SiLU interface use the same capability.

Five new CPU dispatch/graph checks and the adjacent canonical regressions
pass. Normal installed wheel `dev16+ga5e43e667d` also passes preparation and
opaque-operator GPU graph checks for both orientations, covering
M512/8/1/5/16/20/32/8. M8 selects the native pair; other M match canonical
outputs bitwise, and every CUDA graph replay matches its eager output.
Both native modules match the wheel members, with only standard libraries
resolved and no source override.

The installed opaque path's layer39 ABBA at steady 1530/877MHz is
canonical 74.752/73.728us versus native 55.296/55.296us. Layer42 changes
clocks between arms in that installed wiring run; its timing is not treated
as a stable ABBA result. The focused stable raw-operator comparison above
establishes its admission, and the installed run establishes its numerical
and graph behavior. Natural model output and primary-machine operator
comparison remain pending before promotion.
