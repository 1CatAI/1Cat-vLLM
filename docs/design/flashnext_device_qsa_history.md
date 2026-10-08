# Direct device QSA history on SM70

The device-history reference stores target K/V as per-vector E4M3 bytes with
FP32 scales, and draft K/V as FP16. Previously both passed through protected
hot-page resolution and miss staging before attention. Direct device reads
remove that ownership, resolution and staging dependency when the authoritative
history is already on the GPU. Host history keeps the protected reader.

The direct reader groups six query heads per local KV head and decodes
per-vector E4M3 scales into the same FP16 K/V operands as the protected
reader. It retains the protected reader's tile/split assignment, FP32
accumulation and softmax, FP16 probability operands for PV, FP32 merging and
FP16 attention output before the FP32 sigmoid gate. Only placement and
ownership dependencies are removed. This arithmetic-preserving revision
requires its own isolated and model qualification.

The two attention kernels cover M1..20, H6, D256 and selection widths up to 4096. A
single per-device, per-width workspace is shared by serial target and draft
QSA owners and allocated before graph capture. Other shapes keep the existing
reader. `sm70_qsa_device_history` controls the capability; startup reports
explain disabled, host-resident, unsupported hardware and missing-extension
cases. Runtime guards explain incompatible geometry or metadata. The module
is built and installed through normal CMake and wheel registration.

## Initial FP32-probability experiment

The initial native experiment used Volta mma884 for FP16 QK operands with
FP32 accumulation, then retained FP32 probabilities and scalar PV FMAs. It
changed the protected reader's FP16 probability boundary despite using the
same decoded K/V values. The measurements below describe that experiment,
not the arithmetic-preserving revision.

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

The clean wheel passes nine GPU cases, including all E4M3 encodings, rewritten
history and aliased pages. Its 16 previous native modules retain their exact
hashes. On a full TP4 NV2 V100 mesh, the capability-disabled model control
measures C1 19.809 ms/round at I8192/O256 and C4 43.113 ms/round at I128/O600.
The C1 probe emits 4.886 tokens/round. Sixty-four repeated, aligned
teacher-forcing positions have identical logits and top-1 decisions within
that engine. The capability-enabled initial experiment measures C1 19.103
and C4 42.244 ms/round, saving 0.706 and 0.869 ms respectively. C1 emits the
same 256 token IDs and 4.886 tokens/round.

The initial experiment fails the model qualification gate. Across 64 aligned
teacher positions, mean KL is 0.000859, maximum KL 0.011318 and top-1 agrees
at 63 positions. Repeated candidate captures are bit-exact at all 64 positions.
Eight 600-token prompt clusters have mean acceptance 43.445% versus 44.990%
for the control. The paired difference is -1.544 percentage points, with a
95% prompt-bootstrap interval [-4.348, +0.675]. An interval spanning zero
does not establish unchanged acceptance. Natural outputs diverge after 3–164
tokens, and none of the four C4 streams is identical. These speed results are
not admitted as a qualified improvement.

The next revision loads device history inside the existing attention kernel,
without protected page resolution or miss staging. It uses the same compact
page4 count, split assignment, online softmax and FP16 PV boundary. New
isolated tests compare directly with the protected reader in addition to the
FP32 oracle, including long and short contexts, E4M3 and FP16 histories and
changed-input graph replay. GPU qualification for this revision is pending.

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
