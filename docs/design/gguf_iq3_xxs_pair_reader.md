# IQ3_XXS reader for shared-activation pairs

The next seven IQ3_S/IQ3_XXS gate/up pairs reuse the two-reader M8 kernel
from the IQ3_S/IQ4_XS path and the source-sized records described in
`gguf_iq3_xxs_pair_records.md`. The IQ3_XXS reader loads three aligned
16-byte packets per K128 record, caches original d across K256 halves,
and advances pointers without recalculating block addresses in the loop.
It retains the original seven sign bits and four-bit small scale per K32.

Operand formation calls `LatticeRawDecoder<18>::table_fragment<half>`.
The existing original-block decoder owns the FP16 operand proof and scale
formula. This reader adds no quantization formula, scale expansion or
rounding policy. Dot products and split-K reduction retain FP32 arithmetic
in the existing shared-activation skeleton.

The raw operator builds both IQ3_XXS/IQ3_S orientations alongside the two
existing IQ3_S/IQ4_XS orientations. Byte checks distinguish 98/110/136-byte
blocks. Host launch code is shared; the existing two kernel instantiations
retain their implementation. Actual-weight GPU checks and the matched cold-L2 graph comparison pass
in both orientations. Model capability declarations admit only
M8/N4352/K5120 on SM70 with FP16 operands; all other descriptors retain
canonical dispatch. Source-sized IQ3_XXS records are prepared alongside
the existing independent IQ3_S reader.

Ten CPU record/cursor checks pass. The new cursor oracle covers all eight
split-K starts for K5120, including odd starts, two N32 tiles, all columns
and all sixteen octets per K128 record. Reconstructed index bytes, sign/scale
words and cached original d match the source. An initial oracle failure was
caused by using big-endian integer parsing in the test; specifying the
original little-endian byte order fixes the oracle without changing the
reader. The normal CUDA 12.8 SM70 extension and complete wheel build pass. Both
IQ3_XXS orientations use 63 registers, 34,816 shared bytes, zero stack and
zero local memory. Existing IQ3_S/IQ4_XS instantiations retain 64 registers
and 33,792 shared bytes. The installed wheel imports both normal extensions
and passes native-member hashes and dependency checks without private DSOs.
Actual layer2 and layer38 TP4 slices pass official-dequantization FP32-GEMM
checks across three seeds: native relative L2 is 0.000503–0.000536 and
0.000507–0.000516, respectively. Runtime M512/8/1/5/16/20/32/8 and graph
replay checks pass; other M retain bitwise canonical results.

Cold-L2 graph ABBA uses 16 MiB eviction before each event, 84 samples,
V100 SXM2 32GB, fixed SM/memory 1290/877MHz and 300W, Torch 2.10 cu128
and CUDA 12.8. Both orientations read 18,104,320 source bytes per pair.

| Gate/up types | Canonical ABBA arms | Native ABBA arms | Source bandwidth |
| --- | --- | --- | --- |
| IQ3_XXS/IQ3_S | 86.016/86.016us | 57.344/57.344us | 315.7GB/s |
| IQ3_S/IQ3_XXS | 87.040/87.040us | 58.368/58.368us | 310.2GB/s |

Seven layers save an estimated 0.201ms of projection service per round,
covering another 16.954% of mixed-pair source bytes. Together with the
IQ3_S/IQ4_XS pair, 18/40 mixed layers cover 48.463% of mixed-pair bytes.
This is a weighted microbenchmark estimate, not an end-to-end result.
The first GPU attempt failed while the filesystem was full; its controller
and logs are retained. Duplicate archived artifacts were retired before
repeating the operator checks. Model integration checks remain pending.

## Shared-activation hardware counters

A separate one-node CUDA graph capture profiles the already admitted pure
IQ3_S gate/up operator from the normal installed wheel. It uses actual
layer6 TP4 weights, M8/N4352/K5120, a 16MiB flush before the node,
fixed SM/memory 1290/877MHz and 300W. No private extension is loaded.
Ordinary-user counter permission failed; an authorized privileged retry
passes under the same GPU leases. The failed capture is retained.

| Actual total counter | Earlier signed pair | Shared-A pair |
| --- | ---: | ---: |
| L1 global read bytes | 68,497,152 | 33,221,120 |
| L2 read sectors times 32 | 31,000,096 | 31,044,928 |
| DRAM read bytes | 19,466,304 | 19,457,728 |

L1 read demand falls 51.5%, while measured total L2 traffic is essentially
unchanged. These totals do not identify activation-only L2 bytes. The
profiled 47.360us service and 7,837,952 warp instructions are counter evidence,
not an unprofiled speed result. Excess shared wavefronts remain 911,091;
shared activation reuse does not remove the codebook conflict. The earlier
two-copy parity lookup was measured equal in latency and remains rejected.
