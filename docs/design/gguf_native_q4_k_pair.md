# Original Q4_K reader for shared-activation pairs

The next three IQ4_XS/Q4_K mixed pairs reuse the existing M8/N32 shared-A
kernel with an independent original-byte affine reader. The four K128
packet planes and original K256 metadata come from `gguf_q4_k_records.py`.
Metadata is cached across both halves, including odd split-K starts.

The reader restores the original six-bit scale/min fields and nibble order.
It multiplies original FP16 d/dmin and integer coefficients in FP32, then
converts only the final operand to FP16. For finite FP16 metadata,
`d * scale * q` has at most 21 significant bits and is exact in FP32;
FMA therefore preserves the official separate multiply/subtract rounding.
Dot products, split-K reduction and the gated epilogue are unchanged.

The original field interpretation follows llama.cpp's `get_scale_min_k4`;
the existing MIT license remains in the vendored source. No llama.cpp GEMM
or scheduling kernel is added. The raw operator compiles both orientations,
while model capability admission remains unchanged until actual-weight
numerical and cold-L2 graph comparisons pass.

An independent CPU cursor oracle covers all eight split-K starts, both
N32 tiles, all columns and sixteen fragments per K128 record at K5120.
It checks cached original metadata, scales/mins and unpacked nibbles.
Normal CUDA 12.8 SM70 extension and complete wheel build pass. Both
orientations use 64 registers, 33,792 shared bytes and no local memory or
stack. Actual layer 37/layer 58/layer 62 FP32-GEMM checks against official
GGUF dequantization pass at relative L2 0.000501–0.000528. Runtime
M512/8/1/5/16/20/32/8 and CUDA graph replay checks pass, with other M
retaining bitwise canonical outputs.

The first benchmark input list included IQ3_S/Q4_K rather than the
intended IQ4_XS/Q4_K orientation. Type validation rejected it before
projection preparation. The corrected source inventory selects layers 37,
58 and 62, and the initial failure is retained.

At fixed 1290/877MHz, 300W, 16MiB cold-L2 graph ABBA with 84 samples,
canonical is 78.848us for all three cases. The original native reader
measures 80.896us for layer 37,79.872us for layer 58 and79.872–80.896us
for layer 62. Source payload is 24,371,200 bytes, or 301.3–305.1GB/s for
the native operator. These slower paths retain canonical model dispatch.

A secondary 1530MHz run is equal or only marginally faster; layer 62 changes
clocks in the final canonical arm, and that arm is excluded. The primary
comparison above determines the current admission decision.

A one-node NCU capture at 1290/877MHz measures 13,492,832 warp instructions
and 46.71% issue-active utilization. PRMT,LOP3 and conversion instructions
are substantial; L1 global reads total 35,771,008 bytes, L2 reads total
34,851,840 bytes and DRAM reads total 24,871,264 bytes. Excess shared
wavefronts are 17,408, so this is unlike the IQ3_S codebook conflict.
Profiled 81.856us service is not an unprofiled speed result.

The revised IQ4_XS reader initializes a 64-byte shared FP32 table from
TurboMind's existing `iq_values` helper. All 16 integers are exactly
representable in FP16 and FP32. Direct lookup removes unity-scale Half
conversion and promotion; original FP32 scaling and final operand rounding
remain unchanged. A CPU oracle exhausts 65,536 four-nibble packets against
the canonical constants and official GGUF codebook. The normal CUDA 12.8 build and complete wheel pass. Fifteen actual-weight
outputs spanning layers 37/58/62 and already admitted layers 39/42, with
three activation seeds each, are bitwise identical to the previous reader.
No additional model admission is enabled while the primary performance
comparison is pending.

A second one-node NCU capture at the same 1290/877MHz and 300W measures
11,921,488 warp instructions, 11.6% fewer, and 41.96% issue-active
utilization. PRMT drops from 1,577,600 to 707,608 instructions; packed
Half arithmetic is replaced by shared FP32 lookup. L1 reads are
35,940,864 bytes, L2 reads 34,964,864 bytes and DRAM reads 24,870,944
bytes, with 17,408 excess shared wavefronts. Profiled 79.936us service
is recorded separately from unprofiled graph timing.

In secondary matched graph runs at stable 1530/877MHz, layers 37 and 58
measure 66.560–67.584us native versus 72.704us canonical, or
360.6–366.2GB/s of original payload. Layer 62 changes SM clock during
the comparison and is excluded. Existing IQ3_S/IQ4_XS orientations in
layers 39 and 42 measure 53.248us native versus 80.896–81.920us
canonical. Numerical, runtime-M and graph fallback checks pass. These
secondary measurements do not replace the primary fixed-clock comparison.

The revised reader passes the primary 1290/877MHz, 300W comparison:
layer 37 measures 76.800us native versus 77.824us canonical, layer 58
75.776–76.800us versus 78.848us, and layer 62 76.800us versus 78.848us.
Native source bandwidth is 317.3–321.6GB/s. Both Q4_K/IQ4_XS orientations
are admitted only for SM70 FP16 M8/N4352/K5120 gate/up projections; all
other shapes retain canonical fallback. The three layers save an estimated
5.12–6.14us per M8 round, not an end-to-end measurement. Source-sized
packing preserves 73,113,600 bytes of original weights per TP4 rank.
