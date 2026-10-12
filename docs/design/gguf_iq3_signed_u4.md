# Exact signed-u4 IQ3_S screening on SM70

This research benchmark expands signed IQ3_S grid values into `v = 2q - 15`, retaining the existing FP16 group coefficients and FP32 MMA accumulation. The unchanged DMV11 kernel is the control. No serving dispatch, storage replacement, or production extension changes are included.

## Test plan

Four V100-SXM2-32GB ranks, TP4, full NV2 connectivity, CUDA 12.8, Torch 2.10.0+cu128; base `c4f6245f841466782752a8c3283e4727565cf17a`. Real 27B IQ3_S weight shards, M=8, CUDA graphs, independent weight banks exceeding L2, same-process ABBA. The normal control wheel is `1.5.2.dev0+g2b00cc8a38.cu128`; both arms use one research JIT module generated from this tree. The module is not a production artifact.

The benchmark checks decoded weight bits, signed-zero scale cases, official GGUF dequantization error, and changed-input graph replay output bits on all four ranks. Singles preserve the GDN three-plane head layout when slicing K. Timing-window clock samples are retained; post-run clocks range from 1522 to 1530 MHz SM and 877 MHz memory. A post-run reading does not establish a constant clock for the entire timing window.

## Test result

All decoded-weight and graph-output comparisons passed bitwise. Four independent pure-IQ3_S gate/up pairs measured 151.112–163.465 us in the control and 147.169–148.560 us after expansion. Per-call singles measured:

| Projection | Original us | Signed-u4 us |
|---|---:|---:|
| down | 22.955–23.709 | 20.845–21.170 |
| GDN out | 12.230–12.458 | 11.787–11.983 |
| attention o | 12.441–12.805 | 12.064–12.372 |

The proven coverage is eight pure IQ3_S gate/up layers, 22 IQ3_S down layers, 22 IQ3_S GDN out layers, and ten IQ3_S attention o layers. Layer-weighted isolated savings are below 0.1 ms per round. This is an estimate, not model latency evidence. Mixed pairs and qkv have not been screened.

The expanded payload is 1.285714 times the existing native packed payload. Expanding all IQ3_S block matrices would add 264,929,280 bytes (252.656 MiB) per TP4 rank before allocation padding. Keeping a second fallback copy would add further memory and is not justified by these small measured gains. Existing canonical metadata already contains expanded FP16 coefficients; inverse-codebook restoration is feasible with an 8192-byte CPU table, but has not been implemented or proven here.

## Decision

Defer production integration. The experiment removes shared grid lookup while preserving numerical behavior, but the measured savings are too small to advance the complete-round target materially. Do not spend further iterations on tile or packet variants of this format. The next screen addresses collective/normalization and projection serialization.

Compact raw records and checksums are in [data/gguf_iq3_signed_u4_54633_20261008.json](data/gguf_iq3_signed_u4_54633_20261008.json). Large generated CUDA, research binaries, timing clocks and lock records are retained outside Git. No model-level acceptance, C4, or end-to-end gain is claimed.
