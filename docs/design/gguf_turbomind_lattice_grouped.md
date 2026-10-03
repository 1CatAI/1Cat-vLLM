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
replay of the earlier graph all match the FP32 oracle. The ordinary installed wheel passes 96 GPU checks with one non-SM70 skip,
including the final candidate set and cold-graph behavior. Complete dependency
checking passes; source-built, packaged and installed core hashes match, with
no RPATH/RUNPATH. These operator results do not establish model throughput.

## Installed wheel and 512-expert measurements

The installed environment uses Python 3.12.3, Torch 2.10.0+cu128,
Transformers 5.18.0, GGUF 0.19.0, XGrammar 0.2.0 and Tilelang 0.1.10.
Wheel SHA256:
`cf7e4c6a9bdc77f8de7f9ccadf764269b74d1302b2609c8e0c420ec5c91e67ca`.
Core SHA256:
`20ac310a9a80ac4075719cfd1c75a9e70f9649eb714ff21b7be12e584fc2a2cf`.

All rows retain TP4 N=160/K=2560. M counts sorted expert rows, with one
assignment per row. Router/top-k expansion and the complete FFN are excluded.
The 512-expert sweep uses all distinct checkpoint expert matrices. At M=128,
384 experts are empty; at M=512 each expert receives one row; at M=8192 each
receives sixteen rows. These are operator distributions rather than model
request-concurrency measurements.

| Type | Experts | M | GGUF us | AWQ us | MMVQ us | MMQ us | Output relative L2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IQ2_XS | 4 | 512 | 40.79 | 28.06 | 763.48 | 230.44 | 0.000389 |
| IQ2_XS | 4 | 8192 | 208.01 | 228.46 | 12104.10 | 2225.70 | 0.000390 |
| IQ2_XS | 512 | 128 | 113.79 | 72.77 | 201.82 | 827.92 | 0.000393 |
| IQ2_XS | 512 | 512 | 274.84 | 186.48 | 801.48 | 2726.57 | 0.000389 |
| IQ2_XS | 512 | 8192 | 448.17 | 351.68 | 12217.66 | 7742.73 | 0.000389 |
| IQ3_XXS | 4 | 512 | 36.64 | 28.08 | 745.45 | 223.69 | 0.000407 |
| IQ3_XXS | 4 | 8192 | 300.32 | 228.64 | 11645.76 | 2192.37 | 0.000407 |
| IQ3_XXS | 512 | 128 | 152.01 | 72.45 | 197.82 | 763.89 | 0.000404 |
| IQ3_XXS | 512 | 512 | 307.14 | 186.23 | 783.05 | 2509.22 | 0.000407 |
| IQ3_XXS | 512 | 8192 | 562.61 | 351.40 | 12008.97 | 7532.20 | 0.000407 |

The E4 M=8192 improvement reproduces after installation: IQ2_XS 208.01 us
versus AWQ 228.46 us, and IQ3_XXS 300.32 us versus AWQ 228.64 us.
Larger expert counts expose another gap. At E512/M=8192, IQ2_XS costs
448.17 us versus AWQ 351.68 us, and IQ3_XXS costs 562.61 us versus
351.40 us. E512/M=128 is 113.79/152.01 us versus AWQ about 72.6 us.
These distributions require grouped decode/batch work before connecting
Flash-Next; the E4 prefill improvement does not establish parity for all MoE
workloads. No lower-precision activation or accumulator mode was introduced.
