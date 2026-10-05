# Qwen3.8-27B original-byte mixed gated-pair coverage

The [64-layer source inventory](gguf_qwen38_iq3s_source_inventory.md)
contains40 mixed gate/up pairs in19 orientations. This source tree admits
all19 measured orientations for SM70 FP16 M8/N4352/K5120, covering all
40 mixed layers and their2,989,998,080 original payload bytes. Including
the eight pure IQ3_S pairs,48/64 layers use a joint original-byte gated
pair, covering3,602,759,680 bytes of gate/up payload. Twelve pure IQ3_XXS
and four pure IQ4_XS pairs remain canonical pending their own matched
measurements. qkvz remains canonical; its raw prototype was slower.

Admission is capability-based, default-on and restricted to the measured
shape. Missing operators, other architectures/dtypes/dimensions and
non-M8 use canonical dispatch with recorded reasons. Installed capability
declarations match all19 actual mixed orientations. Real-weight official
FP32 GEMM and changed-M graph checks cover every mixed combination.
The last eight layers use complete audited CUDA12.8 SM70 wheels with
Torch2.10.0+cu128 and Python3.12. No private native overlay is required.

## Last eight mixed layers

All numbers are actual per-rank TP4 weights, M8/N4352/K5120,16MiB cold-L2
CUDA graph ABBA, V100-SXM2-32GB,300W. Payload bandwidth counts original
weights only; it is not a hardware DRAM counter. Native accumulation and
reduction stay FP32. Different clocks/topologies are reported separately.

Primary full-NV2 host,1290/877MHz:

| Layers | Gate/up | Original bytes/rank | Canonical us | Native us | Payload GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 11 | IQ2_XS/IQ3_XXS | 14970880 | 88.064 | 57.344 | 261.1 |
| 16 | IQ2_XS/IQ3_XXS | 14970880 | 88.064 | 57.344–58.368 | 256.5–261.1 |
| 12 | IQ2_S/IQ2_XS | 13578240 | 93.184 | 58.368–59.392 | 228.6–232.6 |

These three layers save94.208–96.256us of isolated operator time per
round/rank. The other five primary fixed-clock comparisons remain queued.

Secondary ring-NVLink host,1530/877MHz:

| Layers | Gate/up | Original bytes/rank | Canonical us | Native us | Payload GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 11 | IQ2_XS/IQ3_XXS | 14970880 | 77.824 | 50.176 | 298.4 |
| 16 | IQ2_XS/IQ3_XXS | 14970880 | 77.824 | 51.200 | 292.4 |
| 12 | IQ2_S/IQ2_XS | 13578240 | 82.944 | 51.200–52.224 | 260.0–265.2 |
| 0/1 | IQ2_XS/IQ2_XXS | 12185600 | 76.800 | 48.128 | 253.2 |
| 14 | IQ2_XXS/IQ2_S | 12881920 | 78.848 | 52.224 | 246.7 |
| 28 | Q2_K/IQ3_S | 16885760 | 73.728 | 58.368 | 289.3 |
| 13 | IQ1_M/IQ2_S | 12011520 | 90.112 | 64.512 | 186.2 |

Every row has three official real-weight numerical checks and runtime
M512/8/1/5/16/20/32/8 graph checks. Non-M8 outputs remain bitwise
canonical. Installed prepared-layer checks repeat numerical/dispatch/graph
validation without another speed run. All shipped native libraries are
bitwise unchanged between each new prototype and model-wiring wheel;
the latest-main IQ2_XS rebuild additionally matches qualified pair SASS.
Source metadata remains unrounded and no new precision reduction exists.

These isolated savings do not establish end-to-end latency. The one
[latest-main end-to-end/trace baseline](gguf_iq3_gated_pair_dispatch.md)
remains34.173/35.292ms per round at1K/8K; no new full trace or speed run
is added here. The preceding Q4_K/IQ4_XS change supplies the last full-model
natural-EOS health check. The shared-A positive result and duplicate-book,
wider-N negative results remain in the inventory/dispatch documents.
