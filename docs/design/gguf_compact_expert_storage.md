# Original-byte IQ3_XXS expert storage

Calibrated SM70 TP4 IQ3_XXS gate/up banks use N160/K2560/E512. Load their
original GGUF blocks into the equal-byte packet layout from #897, skip canonical
transcoding and extra original-row retention, and run the grouped operator.
The packet layout preserves all source bits, including seven-bit signs and
nested scales; FP16 operands match official dequantization and MMA accumulates
in FP32.

Each bank has one 80,281,600-byte uint8 buffer (76.5625 MiB). Canonical carriers
alone use 104,857,600 bytes (100 MiB), and retained original aligned rows add
80,609,280 bytes (76.875 MiB). Pointer tables add further storage. Compact
storage requires neither canonical metadata/pointers nor a second row bank.
The admission report records layout, bytes and the reason when geometry,
hardware, dtype or operators prevent the original-byte path.

The storage factory admits only measured IQ3_XXS N160/K2560/E512 on SM70 with
FP16 operands. Gate/up loading uses the existing TP partition, and other
geometries use their existing preparation. Routing, activation and down
projection continue through the same MoE structure. No new environment variable
is introduced.

## Validation

The normal whole wheel passes 30 CPU storage/dispatch checks. A complete bank
contains exactly 80,281,600 bytes, canonical preparation is not called, and no
stats, pointers or additional original-row bank is retained. Its native core
hash matches the source-complete operator wheel. A temporary missing `patchelf`
path during packaging is corrected by using the runtime's normal tools.

A full CUDA graph test uses different expert weights, changed input and changed
routing to compare the bank's output with official dequantization and FP32 dots.
The first job times out before acquiring the shared GPU lock; this validation
remains pending, as do mixed-projection regressions and model quality.

Independent operator full graphs from #897 use actual Flash-Next IQ3_XXS
weights, E512/top-10/TP4 and distinct banks exceeding twice V100 L2. Grouped
C1/5/8/16/512 times are 30.38/37.78/53.63/80.36/325.79 µs versus canonical
72.10/94.40/112.48/140.87/528.68 µs, with official-weight relative L2 around
0.0202–0.0209%. These exclude sorting and the remaining FFN.

These canonical grouped controls are not the complete model baseline. The
current model also selects canonical vectors and original-block joint gate/up
for specific small batches. A matched gate/up pair comparison against those
selected paths is required before promotion. Compact banks currently dispatch
two grouped calls; a joint original-byte packet operator may be needed to retain
small-batch latency. Projection numbers are not model step-time savings.

Model promotion requires the new bank graph gate, full model numerical/quality
checks and matched C1/C4/C8/C16 and 8K/32K prefill with TP4 and full CUDA graphs.
The implementation remains a separate dependent PR until these checks pass.
