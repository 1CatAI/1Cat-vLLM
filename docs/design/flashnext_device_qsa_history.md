# Direct device QSA history on SM70

The device-history reference stores target K/V as per-vector E4M3 bytes with
FP32 scales, and draft K/V as FP16. Previously both passed through protected
hot-page resolution and miss staging before attention. Direct device reads
remove that ownership, resolution and staging dependency when the authoritative
history is already on the GPU. Host history keeps the protected reader.

The native path groups six query heads per local KV head, uses Volta mma884
for FP16 QK operands with FP32 accumulation, then retains FP32 probabilities,
PV FMAs and split merging. E4M3 decoding includes signs, subnormals and NaNs;
per-vector scaling rounds to the same FP16 operands as the protected reader.
The attention output rounds to FP16 before its FP32 sigmoid gate and final
FP16 product. There is no additional activation or probability quantization.

The two kernels cover M1..20, H6, D256 and selection widths up to 4096. A
single per-device, per-width workspace is shared by serial target and draft
QSA owners and allocated before graph capture. Other shapes keep the existing
reader. `sm70_qsa_device_history` controls the capability; startup reports
explain disabled, host-resident, unsupported hardware and missing-extension
cases. Runtime guards explain incompatible geometry or metadata. The module
is built and installed through normal CMake and wheel registration.

## Isolated measurements

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, one card, same-process graph
ABBA. Twelve independent target histories and one shared draft history use
causal selections, 512 selected page4 groups and an open tail. Long-context
queries share 350 selected pages per request; these are synthetic selectors,
not replayed model activations. Query standard deviation is one during timing.
Each of four draft calls reuses the same FP16 history. Workload M5/M1/M1/M1
and M20/M4/M4/M4 matches C1/C4 attention row counts.

| Chain | Protected reader us | Direct reader us | Difference us |
| --- | ---: | ---: | ---: |
| Target twelve M5 calls, context 8448 | 841.168 | 489.296 | -351.872 |
| Target twelve M20 calls, context 728 | 1287.296 | 953.984 | -333.312 |
| Target M5 plus four draft calls | 1080.464 | 565.024 | -515.440 |
| Target M20 plus four draft calls | 1537.904 | 1090.304 | -447.600 |

The complete attention chain drops from 64 to 32 kernels. These are isolated
chain measurements, not endpoint ms/round. Each candidate passes an FP32
attention oracle on exactly decoded FP16 K/V, two changed-query graph replays,
invalid-request and negative-position masks, and an isolated query-scale-eight
case. Maximum absolute initial errors are 3.052e-5 for M5 and 6.104e-5 for
M20. The protected reader's FP16 probability boundary produces larger oracle
error in these samples. Model teacher-forcing, acceptance and same-wheel C1/C4
measurements remain required before promotion.

A probability-decomposition variant replaces scalar PV with three tensor-core
products and an FP32 residual. Independent power-of-two scaling preserves
probability bits, including values not representable by a normal half. CPU
reconstruction is exact on one million random values and exponent boundaries;
isolated GPU numerical checks pass. It is slower: M5 target chain
825.952→686.336 us versus about 489 us for scalar PV, and M20
1286.000→1723.952 us. It is rejected and has no production dispatch.

## Sources and applicability

[FlashInfer split-KV](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
shows how KV splits fill otherwise idle SMs for small query batches; the
native path retains enough splits for the 80-SM V100 without host scheduling.
[HiSparse](https://arxiv.org/html/2608.07009v1) emphasizes that resolution and
placement costs lie directly before attention. Here the authoritative history
is already device-resident, so resolution can be removed entirely; the host
placement experiment remains separate.
[BitDecoding](https://arxiv.org/html/2503.18773v1) separates quantized layouts,
cooperative decoding and tensor-core computation. Its tested SM80/89/90
mechanisms are not copied onto SM70: this implementation uses ordinary loads,
shared memory and mma884, with no TMA, cp.async or WGMMA.
