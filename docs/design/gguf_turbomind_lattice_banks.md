# Replicated IQ3 grouped-vector codebook

The canonical IQ3_XXS decoder reads two four-byte shared codebook rows per
fragment. This candidate interleaves eight identical copies of each row and
selects a replica using the low three lane bits. The physical shared address
separates lane groups across banks; indices, signs, scales and reconstructed
values remain identical. Initialization retains aligned word copies and reads
the original row for every replica.

IQ3_XXS grouped-vector shared codebook storage grows from 1 KiB to 8 KiB.
GPU weight storage is unchanged. GEMM, dequantization, IQ3_S and other formats
retain their original tables. The common fragment decoder accepts an internal
replica count, defaulting to one; only IQ3_XXS grouped vector requests eight.
FP16 activations/reconstruction and FP32 accumulation remain unchanged.
Descriptor and cache keys remain unchanged because each existing kernel has
one decoder implementation.

Runtime tracing places the IQ3 grouped gap in the mainloop, but hardware
bank-conflict counters are unavailable. Reduced shared-bank contention is a
hypothesis, not an established runtime cause. Matched measurements justify
replication only for the vector operator. No dispatch threshold or environment
variable is added.

## Initial measurements and narrowed scope

The shared-table candidate passes all 40 lattice GPU checks. Real IQ3_XXS
TP4 N=160/K=2560, V100 32GB, CUDA 12.8, Torch 2.10.0+cu128, 100 ms
warmup and 100 graph timing iterations show different outcomes by operator.
E512 vector M=128/512 improves from 76.45/260.25 to 69.01/242.04 us,
versus AWQ 72.56/185.95 us. E4 vector M=8 changes from 15.29 to 15.06 us.
GEMM regresses: E512/M=8192 changes from 555.25 to 573.27 us, while
E4/M=8192 changes from 296.43 to 307.93 us.

The final implementation limits replication to IQ3_XXS grouped vector decode.
GEMM and dequantization retain the single shared table. The original
all-operator candidate core is
`c0ee0a355cdff92ec8cfc4d080242ef3fb4efcc353e74554a092d4827934b8cb`.

## Final operator checks

All 40 GPU tests pass after limiting replication to grouped vector. These
cover the seven lattice formats, FP32 matrix/vector oracles, empty experts,
graph capture/replay, tracing and measured capability selection. All 256
official IQ3_XXS rows and 32 lane addresses reconstruct the original codebook
words exactly. Integer payloads and coefficients are unchanged.

Matched real-weight graph timings are below, in microseconds. The workload
is one TP4 expert projection, N=160/K=2560, using distinct checkpoint experts
and one sorted assignment per row. M is the number of assigned rows, not
request concurrency. Routing, sorting and the complete FFN are excluded.
Each route uses 100 ms warmup and 100 timing iterations; every listed graph
captures eight invocations. AWQ uses the same dimensions and assignments.

| Experts | M | Framework selection | Direct GEMM | Direct vector | AWQ | llama MMVQ | llama MMQ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 17.11 | 50.69 | 17.09 | 19.52 | 9.42 | Unsupported |
| 512 | 128 | 69.50 | 134.37 | 69.05 | 72.49 | 204.05 | 765.25 |
| 512 | 512 | 241.92 | 303.57 | 240.92 | 186.01 | 812.80 | 2510.87 |
| 512 | 8192 | 554.86 | 555.19 | 3538.13 | 351.29 | 12239.97 | 7532.06 |
| 4 | 8 | 15.05 | 22.25 | 15.05 | 18.47 | 12.53 | 26.33 |
| 4 | 16 | 23.09 | 22.96 | 25.81 | 16.19 | 21.86 | 29.34 |
| 4 | 8192 | 298.51 | 296.35 | 16664.38 | 227.74 | 12023.62 | 2194.33 |

E512 M=128/512 vector gains are approximately 10%/7% relative to the
aligned-word baseline. E512 M=8192 GEMM returns to 555 us and E4 M=8192
to 296 us. The framework keeps its existing measured intervals: vector for
E512 M<=128 and M=512, and E4 M<=8; otherwise GEMM. M=1 llama MMVQ
remains a faster fallback candidate. M=512 and M=8192 still trail AWQ by
about 30% and 58%; this change does not close the large grouped workload gap.

Output relative L2 versus official dequantization and FP32 accumulation is
0.000393–0.000408. Source extension fingerprint:
`52231cc48200b9218ad87633db4d12fc15a038373fdc692f08002df522cec8e4`.
Clean installed-wheel validation is recorded separately below.
