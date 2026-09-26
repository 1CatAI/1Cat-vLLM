# SM70 long-prefill range repair

The sampled maximum in the Q8000/Q8192 D256 GQA endpoint could miss isolated
large logits. Its exponent guard then clipped unequal logits to the same
weight. FP32 QK and PV accumulators cannot recover the resulting distribution.
A retained 152000-key model capture had 20.0106% relative error in the worst
selected query/head row, despite all outputs being finite.

## Implementation

- Keep the stride-8 prefix maximum. Detect every score exceeding the exponent
  range during the existing complete PV traversal.
- Mark the corresponding 128-row query tile on device. Before committing its
  partial result to the online accumulator, recompute that tile with a complete
  maximum, overwriting both its numerator and denominator.
- Reset flags for every prefix block. GPU kernels skip unmarked tiles; no host
  readback or graph-dependent host control flow is added.
- Always use complete maxima for the causal tail, whose numerator is FP16.
  Remove the extra positive score margin and the factor-64 value headroom in
  the FP32-prefix configuration. After value scaling, the magnitude of each
  tail value is at most approximately one and each tail weight is at most one;
  the exact-arithmetic numerator bound is 8192, below the FP16 finite maximum.
  Rounding error and FP16 score/probability storage remain separate concerns.

This does not introduce a new GPU-count, model-quantization, or batch-size
dispatch restriction. The existing shape checks remain unchanged. Range-check
and recomputation are established ideas, including FlashDecoding++ (MLSys
2024); this change applies them to the materialized SM70 prefix PV pipeline.

## Validation recorded on September 26, 2026

Environment: V100-SXM2-32GB, 300 W, Torch 2.10.0+cu128, CUDA 12.8. Every timed
operator executes a CUDA Graph replay. Native extensions are built normally
from this worktree, with no private DSO overlay and no wheel build.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
  .venv/bin/python -m pytest --confcutdir=tests/kernels/attention \
  tests/kernels/attention/test_sm70_79t_stability.py -q
```

The 17 tests cover both query sizes, prefix/tail outliers, dense/single-row
outliers, biased values, and repeated graph input changes including a return
to benign inputs. All pass. The unequal-peak regression fails with the retained
main-equivalent extension.

Seven retained model inputs, with 138 selected queries and all keys, heads,
and output features, pass finite-output checks against a same-input FP64
reference. On the earlier 152K outlier, aggregate relative L2 decreases from
0.98146% to 0.05397%; maximum selected-row error decreases from 20.0106% to
0.20857%. This is attention arithmetic error, not a task failure rate.

Q8192, KV256000, one local KV head, GQA6, D256, compact score block8192:

| Variant | Complete graph endpoint | Useful TFLOP/s |
| --- | ---: | ---: |
| Retained sampled endpoint | 180.065 ms | 70.412 |
| Initial always-complete-max repair | 210.207 ms | 60.316 |
| Adaptive repair | 183.430 ms | 69.121 |
| Same-binary exact control | 282.103 ms | 44.944 |

The final candidate/control run uses 50 ABBA trials and has median paired
speedup 1.539 with bootstrap 95% interval [1.537, 1.541]. Old/new candidate
measurements use the same GPU in separate processes, so their approximately
1.87% latency difference has no paired confidence interval. A separate
24576-score-block experiment reaches 71.358 TFLOP/s at 2.648 GiB first-call
allocation versus 1.148 GiB for the default compact configuration. It is not a
new default. No result here qualifies 75 TFLOP/s with both products accumulating
in FP32.

Raw receipts: `adaptive-margin0-paired-gpu1.json`,
`adaptive-margin0-captured-graph.json`, `adaptive-margin0-b24576-gpu1.json`,
`main-sampled-gpu1.json`, and `fullmax-unprofiled.json` in the companion paper
artifact. They retain input seeds, native/source hashes, trials and references.
The first always-complete-max variant was rejected for its measured slowdown.

## Promotion gate

Kernel regressions alone do not qualify model quality. A same-binary,
attention-only ON/OFF audit with TP4, NVFP4 27B, DFlash2, E4M3 target KV,
262144 context capacity and CUDA Graph enabled is still being collected.
Dataset responses are allowed natural EOS with a 65536-token output budget;
long prompts are used to exercise this endpoint. Keep this PR in Draft until
those results and matched long-context latency are reviewed. These tests do
not establish universal task-level non-inferiority.
