# Original-block joint GGUF expert projections

Small MTP verification batches visit many experts with approximately one
routed token per expert. Compute gate and up in one launch without padding
expert intervals to M=8. The grid spans active expert slots and output rows;
repeated slots return before shared codebook initialization. One warp handles
one output row, and token pairs reuse decoded blocks. Activations and outputs
remain FP16; scale reconstruction, products and reductions are FP32.

The decoder interface reuses #897's original-block lattice formulas and MIT
codebooks. Its provenance points to llama.cpp's exact source revision in the
shared header. This scope includes only joint expert projection and official
reference dequantization, without dense projection tuning. Normal CMake and
core registration ship both operators. Model bank retention and opaque
original-batch dispatch are implemented separately in #939.

## Qualification and real-weight screen

Source `f7998a10b0558a3770b74a4a2609acb78f11b852`, ordinary package
`1.5.2.dev757+gf7998a10b`, V100 SXM2 32 GB, driver 580.173.02,
Torch 2.10 CUDA 12.8 and Python 3.12.3. Nine installed CPU capability checks,
12 GPU official-reference/graph checks and 213 package dependency checks pass.
Official FP32 dequantization matches elementwise with zero tolerance. GPU
cases cover IQ3_XXS/IQ3_S/IQ2_S, M1/5/20/32, odd output row counts, empty,
singleton and maximal expert intervals, changed activations and bitwise
repeated graph replay.

Full native source builds into the normal core. Fresh-process import resolves
the packaged core without RPATH or private library overrides. Core SHA256:
`f1cf194ddd056195d690db8396ba81b7d73ea90cb8505981170f2fb25fba9b5e`.
Whole wheel SHA256:
`10ccf8617c07b25ed4bf458e7d762d3321170e8a976a17ac3046dadc83cc6c7b`.
Later synchronization imports only merged Python/diagnostic changes; native
source remains identical to this artifact.

The benchmark reads real Flash-Next IQ3_S gate/up tensors at layers 0, 1 and
17. Each TP4 rank has local shape [512,160,2560]. Synthetic top-10 routes
produce 10/48/165 active experts for M1/5/20. Six address-rotated copies of
real weights keep the working set above twice V100 L2. CUDA graph timings
normalize the complete gate/up pair, with 100 measured replays. The control
selects canonical vector or grouped GEMM through existing capabilities.

| Type | Original M | Canonical pair µs | Joint original pair µs | Saving µs |
| --- | ---: | ---: | ---: | ---: |
| IQ3_XXS | 1 | 47.383 | 14.748 | 32.635 |
| IQ3_XXS | 5 | 75.269 | 61.417 | 13.852 |
| IQ3_XXS | 20 | 286.732 | 205.005 | 81.726 |
| IQ2_S | 1 | 120.324 | 16.723 | 103.601 |
| IQ2_S | 5 | 158.807 | 71.001 | 87.806 |
| IQ2_S | 20 | 213.025 | 234.923 | -21.898 |
| IQ3_S | 1 | 142.660 | 14.931 | 127.729 |
| IQ3_S | 5 | 186.049 | 62.956 | 123.094 |
| IQ3_S | 20 | 288.016 | 209.860 | 78.155 |

Maximum raw projection absolute error is 0.00097394; maximum relative L2
is 0.00021866 against official FP32 dequantization and FP32 matmul. The
canonical projection's relative L2 is approximately 3.4–3.8e-4. All outputs
are finite. These values include FP16 final-output rounding.

Across 20 IQ2_S and 10 IQ3_S layers, M5 projects 2.987 ms less
operator service. This is a layer extrapolation with synthetic routes,
not an isolated full-model latency claim. The independently measured model
composition and accepted-output trajectories are documented separately.

Capabilities admit IQ2_S original M1/5, IQ3_S M1/5/20 and IQ3_XXS M1/5/20
when retained storage exists. IQ2_S M20 is slower and reports
`measured_slower_than_canonical_grouped_gemm`. Missing original storage,
unmeasured batches, incompatible layouts and unavailable SM70 operators
report explicit reasons and retain canonical scheduling. The default model
retains only IQ2_S/IQ3_S: IQ3_XXS adds about 2.55 GiB per rank for modest M5
savings, so its operator support does not force that storage tradeoff.

Reproduce while holding the shared GPU lock:

```bash
python -m pytest tests/kernels/quantization/test_gguf_raw_grouped.py
python benchmarks/kernels/benchmark_gguf_raw_grouped.py MODEL.gguf \
  --layer 1 --rank 0 --m 1 5 20 --iterations 100 --weight-banks 6 \
  --output IQ2_S.json
```
