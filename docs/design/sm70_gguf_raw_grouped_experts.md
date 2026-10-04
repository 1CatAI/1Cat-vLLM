# Original-block grouped GGUF expert vectors

Small verification batches visit many experts with approximately one token
per expert. Padding those intervals to an M=8 tensor-core tile wastes work.
The grouped vector operator covers active experts and output rows directly,
and computes gate and up in one launch. It reuses the IQ2_S and IQ3_S original
block decoders from #897, including their shared codebooks and original FP32
scale formulas. Activations and final outputs are FP16; products and reductions
accumulate in FP32.

Sorted route IDs and expert offsets are prepared before the operator. The
grid covers route slots and pairs of output rows; repeated expert slots
return before codebook initialization. One warp computes one output row.
Gate and up occupy two warps each. No expert interval is padded to M=8.
The prepared-routing contract requires distinct top-k experts per token.

## Initial operator screen

Hardware: V100 SXM2 32 GB, driver 580.173.02. Runtime: Torch 2.10 CUDA 12.8,
Python 3.12.3, ordinary package `1.5.2.dev541` from `9cdae2022a`. The packaged
core SHA256 is
`12447b4a01c54143133ea0a69c2f8d1d65bd98573d1239aad183a53a4bf1f047`.
The whole wheel SHA256 is
`cd74a0b74d1779c71532b65833a329a2f1c51aff3644f07c7c7db5905ff8d19f`.
Fresh-process import selects the packaged implementation without a private
kernel DSO or preload.

Eight GPU tests pass against official FP32 dequantization and matmul. Outputs
also match the existing raw single-row operator bitwise. Cases cover IQ2_S
and IQ3_S, M=1/5/20/32, odd output row counts, empty experts, singleton and
maximum-length expert intervals, changed inputs and bitwise CUDA graph replay.

The benchmark reads Flash-Next IQ3_S's actual gate/up tensors: layer 1 is
IQ2_S and layer 17 is IQ3_S. Each TP4 rank has local expert shape
`[512, 160, 2560]`. Synthetic top-10 routes produce 10, 48 and 165 active
experts at M=1, 5 and 20. Six replicas of the same real weights rotate
addresses to keep the active weight working set above twice V100 L2,
including M=1. CUDA graph timings normalize the complete gate/up pair.
The reference selects the current canonical grouped GEMM for these types.

| Type | M | Canonical gate/up (µs) | Initial joint raw gate/up (µs) | Saving (µs) |
| --- | ---: | ---: | ---: | ---: |
| IQ2_S | 1 | 120.410 | 16.854 | 103.555 |
| IQ2_S | 5 | 158.831 | 129.508 | 29.324 |
| IQ2_S | 20 | 213.570 | 736.836 | -523.266 |
| IQ3_S | 1 | 145.316 | 14.889 | 130.427 |
| IQ3_S | 5 | 186.159 | 116.534 | 69.625 |
| IQ3_S | 20 | 287.960 | 740.592 | -452.633 |

At M=5, IQ2_S maximum projection relative L2 improves from 3.65e-4 to 2.09e-4;
IQ3_S improves from 3.65e-4 to 2.07e-4. Both projections remain finite at
every point. These are operator results with synthetic activations, not
model-quality or complete-round measurements.

The initial layout keeps a power-of-two number of FP32 accumulators based on
the complete batch size. It uses 80–84 registers per thread at M=5 and
108–114 at M=20, with no local-memory spill. Most expert intervals still have
one token. Its M=20 regression rejects model integration and default dispatch.

## Token-pair follow-up

A follow-up bounds the register tile to two tokens and loops over larger
expert intervals. Singleton intervals read each block once; pairs share the
same decoded block. Longer intervals reuse the shared codebook and reread
weight blocks for their next pair. M=1 retains its single-token tile.
This preserves the FP32 reduction order of each token while bounding
register pressure. Source `a28fda2095`, ordinary package `1.5.2.dev542`, passes
the same eight GPU checks, including bitwise raw-vector comparison and graph
replay. Its core SHA256 is
`f256c98d99cff4b9760c305f94207d6befe61e199476c6d1a715b503429d4ea1`.
The token-pair variants use 48 registers for IQ2_S and 56 for IQ3_S, with
no local-memory spill.

| Type | M | Canonical gate/up (µs) | Token-pair raw gate/up (µs) | Saving (µs) |
| --- | ---: | ---: | ---: | ---: |
| IQ2_S | 1 | 121.858 | 17.011 | 104.847 |
| IQ2_S | 5 | 159.891 | 72.013 | 87.878 |
| IQ2_S | 20 | 213.280 | 239.777 | -26.497 |
| IQ3_S | 1 | 142.715 | 15.207 | 127.508 |
| IQ3_S | 5 | 186.012 | 67.478 | 118.534 |
| IQ3_S | 20 | 289.946 | 223.569 | 66.377 |

Projection errors are unchanged from the initial implementation. Using the
20 IQ2_S and 10 IQ3_S layers suggests 2.943 ms less gate/up service at M=5.
This remains a layer extrapolation; acceptance and full-round latency are
unmeasured. IQ2_S at M=20 still rejects default vector dispatch.

A further screen doubles output rows per CTA for M greater than one,
amortizing codebook initialization across eight output rows. Source
`93a23db`, packaged with documentation in `1.5.2.dev545`, passes all eight
GPU checks but provides no material improvement. The row change is reverted.

| Type | M | Canonical gate/up (µs) | Larger row tile (µs) |
| --- | ---: | ---: | ---: |
| IQ2_S | 5 | 159.152 | 72.079 |
| IQ2_S | 20 | 213.595 | 234.520 |
| IQ3_S | 5 | 186.065 | 68.976 |
| IQ3_S | 20 | 288.036 | 220.812 |

The token-pair implementation is retained. The central kernel capability
declaration admits measured original batches M=1/5 for IQ2_S and M=1/5/20
for IQ3_S. IQ2_S M=20 reports
`measured_slower_than_canonical_grouped_gemm`; unmeasured batches and shapes
retain canonical scheduling. Seven CPU capability tests pass. The rejection
uses original token count, before top-k routing expands it.

The current ordinary package is `1.5.2.dev548` from `4331761a4f`. Its core
SHA256 is
`efc4cb077ae69c45848c187a048b09683f71a3a007c8fdc2623114f0834941c7`;
the whole wheel SHA256 is
`eeeca72f392165b292e0fd1bc70a39b15d8247ac1a1b40dd142b266d41ee1ee1`.
Native source matches the tested token-pair revision. Fresh-process package
import and dependency checks pass without private library overrides.
No model restart follows these tile screens.

The checkpoint contains IQ3_XXS in 17 gate/up layers, IQ2_S in 20, IQ3_S in
10 and IQ4_XS in one. This operator currently covers the 30 IQ2_S/IQ3_S
layers. Other formats and down projections retain their existing operators;
no model dispatch has been changed by this operator screen.

Reproduce under an exclusive GPU lease:

```bash
python -m pytest tests/kernels/quantization/test_gguf_raw_grouped.py
python benchmarks/kernels/benchmark_gguf_raw_grouped.py MODEL.gguf \
  --layer 1 --rank 0 --m 1 5 20 --output IQ2_S.json
python benchmarks/kernels/benchmark_gguf_raw_grouped.py MODEL.gguf \
  --layer 17 --rank 0 --m 1 5 20 --output IQ3_S.json
```

## IQ3_XXS and counter follow-up

IQ3_XXS extends the same decoder family with its original four-bit scale and
seven-bit sign index; the eighth sign is parity. Its shared grid is the
existing TurboMind codebook. Twelve packaged GPU checks pass for the three
formats at M=1/5/20/32, including bitwise FP32 dequantization against the
official reader, bitwise raw-vector projection and graph replay.
Source `9ac70e3988`, ordinary package `1.5.2.dev551`, core SHA256
`c91aa91eedd66dced71b1eb3af7be84a66fde37f9128ddc26969c31c91fd6f35`.

Layer 0 provides the real IQ3_XXS gate/up pair. The same six-bank cold sweep
selects the existing canonical vector at M=1/5 and grouped GEMM at M=20.

| Type | M | Canonical gate/up (µs) | Joint original-block gate/up (µs) |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 1 | 45.394 | 14.170 |
| IQ3_XXS | 5 | 75.175 | 62.690 |
| IQ3_XXS | 20 | 286.525 | 208.448 |

At M=5, maximum projection relative L2 decreases from 3.65e-4 to 2.09e-4.
Across 17 corresponding layers, the M=5 saving projects only 0.212 ms;
M=20 projects 1.327 ms. Retaining both original and canonical banks adds
approximately 2.55 GiB per TP4 rank for these layers. Default model dispatch
has not admitted this storage tradeoff; unsupported batches retain canonical
storage and operators.

An isolated Nsight Compute sample of IQ2_S M=5 reports 147.79 GB/s memory
throughput, 16.64% DRAM throughput, 62.19% SM throughput, theoretical occupancy
62.50% and achieved occupancy 58.48%. The issue-stall rule attributes 6.9
cycles, 47.5% of warp issue cycles, to L1TEX scoreboard dependencies. Its
87.584 µs duration is instrumented and must not replace the graph benchmark.
This motivates testing the existing raw vector's next-block software prefetch
inside grouped projection. FP32 FMA/reduction order and stored blocks remain
unchanged; that candidate still needs an unprofiled speed screen.

The prefetch screen from `27db2cdd6b`, ordinary package `1.5.2.dev552`,
passes the same twelve GPU checks, including bitwise FP32 dequantization and
raw-vector output. The whole wheel SHA256 is
`2533d7abe0e0ad0780447db1cf256614d56ec7b4d53f79cfb4b9169675f16617`;
its core SHA256 is
`8471aa2ea122cefd892d9b57851be433feb40dbc54234c1c69f6483aa3d0b53e`.

| Type | M | Canonical gate/up (µs) | Prefetched original-block gate/up (µs) |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 1 | 46.501 | 14.688 |
| IQ3_XXS | 5 | 75.140 | 61.533 |
| IQ3_XXS | 20 | 287.181 | 204.927 |
| IQ2_S | 1 | 121.934 | 16.709 |
| IQ2_S | 5 | 159.033 | 71.027 |
| IQ2_S | 20 | 213.728 | 235.095 |
| IQ3_S | 1 | 147.138 | 14.819 |
| IQ3_S | 5 | 186.113 | 62.830 |
| IQ3_S | 20 | 287.981 | 209.994 |

The prefetch is retained. IQ2_S M=20 still rejects vector dispatch. The
30 IQ2_S/IQ3_S layers project 2.993 ms less M=5 gate/up service. These are
operator results with synthetic routing, separate from model throughput.
The kernel capability also admits measured IQ3_XXS M=1/5/20 when an original
bank is retained; missing storage reports `original_expert_bank_not_retained`.
Nine CPU capability tests cover the measured points and fallback reasons.

## Decoder interface synchronization

Source `39de67b1dd` synchronizes the lattice decoder dependency at
`7418b9af4f` and retains the joint grouped operator alongside compact prefill.
The removed vector scale option does not alter the grouped FP32 arithmetic.
The ordinary source-containing package `1.5.2.dev816+g39de67b1d` passes
16 GPU checks: twelve joint grouped cases and four vector/dequantization
cases. All 18 raw/grouped-down CPU capability checks pass, as do 213 package
dependency checks. Fresh import resolves the shipped core and standard
Torch/CUDA libraries without preload or a private build dependency.

Whole wheel SHA256:
`09d298f5a2434c67d5b37d1c6fcae578329d53c4252206f3887d0b4658d822d3`.
Core SHA256:
`9908f739bba208ba7b5b8210956e0a7c8ccf8cd8bd213d2ab42c44f42290fb03`.
This interface synchronization adds no new speed measurement; the cold-bank
operator results above and the separately documented model composition retain
their original source and artifact provenance.
