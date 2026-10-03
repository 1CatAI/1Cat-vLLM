# GGUF lattice grouped GEMM dispatch

## Rejected FP16 shared-codebook expansion

IQ3_XXS and IQ3_S codebooks were expanded into FP16 during CTA initialization,
then read as aligned 64-bit rows. Every value converts exactly: IQ3_XXS has
1024 entries in [4,62], IQ3_S has 2048 entries in [1,15]. Shared storage doubles
to 2/4 KiB. The experiment preserves FP16 activations, canonical scales and
FP32 accumulation, and passes 31 lattice/prefill GPU checks.

Measurements use actual Flash-Next TP4 tensors, four distinct experts,
V100-SXM2-32GB, CUDA 12.8 and Torch 2.10.0+cu128. Graph timing uses 100 ms
warmup and 20 iterations, matching the previous byte-table sweep.

| IQ3_XXS grouped M | Byte table us | FP16 table us | AWQ us |
| --- | --- | --- | --- |
| 1 | 21.11 | 22.22 | 15.28 |
| 2 | 20.33 | 19.50 | 15.38 |
| 4 | 20.38 | 19.35 | 16.15 |
| 8 | 20.49 | 19.48 | 18.42 |
| 16 | 23.97 | 21.77 | 16.29 |
| 32 | 24.72 | 22.56 | 17.44 |
| 64 | 37.61 | 32.84 | 21.71 |
| 128 | 28.15 | 28.42 | 28.12 |
| 512 | 36.97 | 37.93 | 28.20 |
| 2048 | 94.59 | 96.77 | 71.87 |
| 8192 | 342.03 | 354.67 | 228.72 |

M=16–64 improves about 9–13%, while large M regresses slightly. IQ3_S dense
M=8192 changes from 1173.04 to 1131.62 us, while its calibrated canonical DQ
route costs 774.55 us. This does not resolve the grouped gap, so the byte-table
decoder remains the default. Hardware counters are unavailable; increased
shared-table traffic is a hypothesis rather than a measured explanation.

## Grouped dispatch cache

Static resource usage for an available IQ3_XXS CTA128/N128/K32 kernel reports
255 registers and an 80-byte stack frame. Runtime trace subsequently showed
that default dispatch actually selects CTA32/N128/K32 with 127 registers and
K split into 2–3 partitions. Static data alone did not identify the active
bottleneck. A CTA64/N128/K64, eight-warp candidate compiled with 149 registers
for IQ3_XXS and 161 for IQ2_XS, but measured selection retained existing tiles.
The extra candidates were removed.

Grouped operators previously used only heuristic dispatch. They now measure
M=512–8192 before capture and reuse the existing per-device cache, with distinct
source-format keys. An uncached captured descriptor retains heuristic dispatch;
measurement is performed only outside capture. Precision and canonical weight
storage remain unchanged. Existing lattice/prefill checks pass all 31 cases.

The synthetic route trace selects IQ2_XS CTA128/N128/K16 at M=8192 with one
K partition and swizzle 1, while IQ3_XXS selects CTA32/N128/K32 with three
partitions and swizzle 1. These traces describe the controlled input distribution;
actual checkpoint timing remains the evidence for performance.

| Type | M | Previous grouped us | Measured grouped us | AWQ us |
| --- | --- | --- | --- | --- |
| IQ2_XS | 1 | 23.72 | 22.17 | 15.33 |
| IQ2_XS | 2 | 22.83 | 22.82 | 15.32 |
| IQ2_XS | 4 | 22.91 | 22.89 | 16.34 |
| IQ2_XS | 8 | 23.03 | 23.29 | 18.37 |
| IQ2_XS | 16 | 23.39 | 23.62 | 16.25 |
| IQ2_XS | 32 | 25.38 | 25.77 | 17.42 |
| IQ2_XS | 64 | 27.52 | 27.60 | 21.68 |
| IQ2_XS | 128 | 36.17 | 31.78 | 28.09 |
| IQ2_XS | 512 | 40.15 | 40.70 | 28.13 |
| IQ2_XS | 2048 | 93.07 | 93.23 | 71.81 |
| IQ2_XS | 8192 | 322.27 | 207.61 | 228.37 |
| IQ3_XXS | 1 | 21.11 | 21.20 | 15.26 |
| IQ3_XXS | 2 | 20.33 | 20.40 | 15.37 |
| IQ3_XXS | 4 | 20.38 | 20.45 | 16.17 |
| IQ3_XXS | 8 | 20.49 | 20.61 | 18.46 |
| IQ3_XXS | 16 | 23.97 | 24.10 | 16.34 |
| IQ3_XXS | 32 | 24.72 | 24.80 | 17.47 |
| IQ3_XXS | 64 | 37.61 | 37.47 | 21.68 |
| IQ3_XXS | 128 | 28.15 | 27.93 | 28.07 |
| IQ3_XXS | 512 | 36.97 | 36.58 | 28.17 |
| IQ3_XXS | 2048 | 94.59 | 85.48 | 71.88 |
| IQ3_XXS | 8192 | 342.03 | 301.95 | 228.29 |

At M=8192, IQ2_XS improves from 322.27 to 207.61 us, versus AWQ 228.37 us.
IQ3_XXS improves from 342.03 to 301.95 us, versus AWQ 228.29 us, leaving a
material gap. Router/sorting/full FFN work is excluded. The cold-graph check passes: capture before tuning, eager measurement and
replay of the earlier graph all match the FP32 oracle. The final candidate
set, larger expert counts and installed-wheel validation remain pending. These operator results do not establish model throughput.
