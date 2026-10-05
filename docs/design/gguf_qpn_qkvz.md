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
