# Normalized integer GGUF dense projections on SM70

The dense path shares the Q8_1 activation layout with the routed expert
operators. It supports three representations: signed integer bytes with
original group16 scale levels, unsigned U4 with affine scale/min levels,
and U4 integer lookup with group32 scales. FP32 dot accumulation and final
FP16 output boundaries remain explicit. No expanded FP16 coefficient is
introduced by normalization.

Q6_K expands its six-bit signed codes losslessly to int8, removing register
decode arithmetic. Q4_K retains unsigned nibbles and both original scale/min
levels. IQ4_XS retains nibble indices and the signed six-bit subgroup scale,
using TurboMind's integer IQ4 lookup. Original base scales remain FP16 values
from the file. Duplicating these original values per subgroup introduces no
rounding and permits load-time head-order permutations without input copies.

Twelve CPU tests reconstruct K1536/K2560 weights elementwise identically to the
official GGUF reader, invert normalized packet order, and restore GDN
head order at load time. Expanded original subgroup scales are permuted
with the integer packets, so the runtime does not reorder activation inputs. Storage expansion,
especially signed Q6 bytes and duplicated scale levels, must be included in
model memory accounting; retaining both canonical and normalized weights can
exhaust the small TP4 V100 headroom.

Each CTA serves up to five tokens with one decoded weight group. Four warps
partition K; split-K can increase the grid for skinny outputs. A cooperative
variant reduces FP32 partials in the same launch when the grid fits resident
block capacity. Otherwise the explicit projection/reduction pair remains
available for measurement. Split1 writes the output directly. Output views
are supported, and an optional epilogue keeps FP16 gate/up, SiLU and multiply
rounding for fused gated projections. All variants support CUDA Graph capture.

The Q4 affine dot follows standard Q8_1 semantics: its correction uses the
rounded original activation sum, not the sum reconstructed from int8 codes.
The independent oracle compares official weights with decoded Q8 activations
and applies that sum correction explicitly.

The benchmark reads actual TP4 model tensor dimensions, compares normalized
integer dots with the current TurboMind preparation, alternates controls,
reports core/memory clocks and includes activation encoding separately.
Bandwidth counts actual normalized storage, rather than original GGUF file
bytes. GPU correctness, bandwidth and the chosen split-K schedules are still
pending the normal complete-extension build. No model dispatcher selects
these operators yet. Model integration must preserve mixed-projection order,
share QKV/Z encoding, preserve head layout, and avoid duplicated weight banks.

For the three supported types in the Flash-Next tensor inventory, normalized
storage increases an estimated 81.6 MiB per TP4 rank: Q6_K contributes
73.9 MiB, Q4_K 5.3 MiB and IQ4_XS 2.4 MiB, excluding padding. These are
storage estimates, not a measured model allocation. Keeping both complete
layouts would add about 571 MiB, exceeding the observed tightest rank's
headroom. Integration therefore needs one resident weight representation and
a shared FP16 prefill workspace, with actual memory validation before adoption.
