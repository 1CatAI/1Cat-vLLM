# SM70 MTP row-strided QKV

The existing mixed-QKV recurrent kernel reads feature-contiguous views using
their physical row stride. Feature-strided views retain the contiguous
fallback. Logical width validation stays independent of row storage.
Kernel schedule, FP32 arithmetic and state storage are unchanged.

The primary GDN core admits measured row-strided M5/M20 speculative batches
with the existing local H4/HV12, K128/V128 and FP32-state contract. Existing
contiguous batches keep their route; unmeasured row-strided batches and
unsupported layouts keep the original Q/K/V rearrange. M1 behavior is unchanged.

## Validation

The ordinary source-containing package passes 18 GPU cases; six unmeasured
row-strided fixture points are skipped. Tests compare the original separated
Q/K/V loader with mixed-QKV inputs, then exercise convolution plus recurrence
using both cache orientations, nonmonotonic state slots, changed accepted
selectors, changing activations/gating and ten graph replays. FP16 output,
FP16 convolution cache and every FP32 SSM snapshot match bitwise.

Python 3.12.3, Torch 2.10.0+cu128, CUDA 12.8.93, V100-SXM2-32GB. The
benchmark uses 36 distinct state banks, real local QKV geometry, FP16 input,
FP32 state and five samples of 100 warmed graph replays.

| Rows | Pack and recurrence (us) | Row-strided recurrence (us) | Saved (us) |
| --- | ---: | ---: | ---: |
| 5 | 964.454 | 470.385 | 494.070 |
| 20 | 1563.167 | 1018.276 | 544.891 |

These are operator-chain savings, not complete model-round gains.
Run `benchmarks/kernels/benchmark_sm70_mtp_strided_qkv.py --output result.json`.
Operator timing source: `0789948eb5`; model-state tests use
`1.5.2.dev740+g0ae5c3970`. No private kernel DSO or preload is required.
