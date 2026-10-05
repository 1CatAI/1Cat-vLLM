# Shared-activation QKVZ and floating B/A

A GDN input projection contains Q, K, V, Z and two floating B/A shards.
The canonical path coalesces compatible quantized shards, then projects
floating shards separately and concatenates their outputs. The one-launch
prototype schedules each source's N64 tiles in one grid and writes the
logical Q/K/V/Z/B/A output directly.

The prototype reuses the measured single-projection shared-activation body,
source-sized native readers, existing canonical U2G16/U4G32 streams and
ordered FP32 split-K reduction. Q2_K and Q4_K use prepared canonical streams;
no original superblock tuning is introduced. Floating B/A uses lane-interleaved
FP16 packets padded to N64. No scale rounding or activation quantization is
added. Only the twenty-four live floating columns are written.

The operator contract is TP4 GDN M8/K5120, source widths 512/512/1536/1536
and B/A widths 12/12. Source descriptors are passed as kernel arguments;
there is no per-call metadata copy, separate split-K reduction or output cat.
Scratch is private to the operator instance, and completion tickets reset
inside the last CTA before graph replay finishes.

The focused benchmark restores the adapter's GDN head order before selecting
TP row shards. It compares three seeded FP32 official dequantization oracles,
checks floating outputs separately, verifies one thousand bitwise graph
replays per input and then measures cold-L2 graph ABBA against the actual
coalesced canonical projection. It records clocks and both source and loaded
candidate byte counts. Compile, numerical and performance results are pending;
model dispatch remains canonical until shape-specific GPU comparisons pass.

## Initial actual-weight comparison

The complete CUDA 12.8 SM70 wheel builds and its installed native libraries
match the wheel hashes. The joint kernel uses 64 registers, 33,793 shared
bytes, eight stack bytes and zero local memory. The extracted single-matrix
body retains 62–64 registers and zero stack/local memory.

At 1290/877 MHz and 300 W, M8 actual TP4 rank-zero shards, 16 MiB cold-L2
CUDA Graph ABBA gives:

| QKV type | Z type | Layer | Joint µs | Coalesced canonical µs | Loaded joint bytes | Effective GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| IQ3_S | IQ3_XXS | 1 | 39.936 | 76.800 | 9,297,920 | 232.82 |
| IQ4_XS | IQ4_XS | 0 | 45.056 | 48.128–51.200 | 11,796,480 | 261.82 |
| IQ3_S | Q2_K | 9 | 38.912 | 73.728 | 10,219,520 | 262.63 |
| IQ3_XXS | Q4_K | 20 | 41.984 | 66.560 | 10,588,160 | 252.20 |

All four pass three official FP32 numerical oracles and one thousand
bitwise graph replays per input. Maximum relative L2 is 0.0005201; floating
B/A is checked separately. The same-clock checkpoint-line fused FP8 QKVZ/B/A
reference is 35.840 µs. These are operator comparisons, not model timings.

Model preparation aliases the existing canonical code tiles and scale/min
columns for Q2_K/Q4_K, including the original coalesced scale row stride.
It retains only source-sized native records and padded floating B/A storage
in addition to the canonical fallback. Actual M is selected inside an opaque
operation with explicit mutable reduction workspace. Only measured complete
source combinations can enter; every other M delegates to canonical.
Installed model-wiring comparisons and the remaining real source combinations
are pending.

The remaining fifteen real QKV/Z combinations also pass. Across all nineteen
combinations representing all forty-eight GDN layers, joint latency is
38.912–45.056 µs versus coalesced canonical 48.128–81.920 µs, at identical
1290/877 MHz clocks. Maximum relative L2 over all fifty-seven inputs is
0.0006698. Every combination wins both comparison arms. The
[complete operator record](data/gguf_qkvz_tp4_m8_20261006.json) includes source
and loaded byte counts, bandwidth, clocks, layer multiplicity and each
numerical/ABBA result. Multiplying each saving by its layer count gives an
estimated 1.262–1.276 ms per M8 round. This has not been measured end to end.

Model capability declarations cover these nineteen complete source tuples,
FP16 activations, SM70, M8 and the exact combined TP4 shape. Other sources,
shapes, M and disabled policy retain canonical with an explicit admission
reason. Coalesced affine code views share their existing storage; scale/min
views retain their original row stride. CPU tests cover this aliasing,
capability gates, runtime-M fallback and a single opaque dynamic export.
Installed compiled-layer checks remain pending.
