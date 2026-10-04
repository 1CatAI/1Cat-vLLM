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

A further operator screen doubles output rows per CTA for M greater than one,
amortizing codebook initialization across eight output rows. The M=1 row tile
is retained. That revision is pending correctness and speed measurements.

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
