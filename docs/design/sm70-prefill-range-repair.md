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

The 19 GPU tests cover both query sizes, prefix/tail outliers, dense/single-row
outliers, biased values, and repeated graph input changes including a return
to benign inputs. All pass. The unequal-peak regression fails with the retained
main-equivalent extension.

The first long-prompt C4 serving benchmark exposed another boundary bug: a
mixed prefill/decode scheduler step produced Q8160/KV8160, but query-only
padding dispatched it to the native Q8192 kernel. The runtime rejected KV8160.
The wrapper now pads only future K/V positions and moves the real query slice
back by that padding amount. Its last visible key remains KV - Q + i, and
padded keys never enter a real query's causal domain. This preserves the fast
path instead of imposing a new request-concurrency restriction. Two of the
19 GPU graph tests compare these cases with unpadded FP64 attention; six CPU
policy/padding tests also pass, including an independent uniform-attention
prefix-mean oracle. The failed C4 benchmark is retained and is not a valid
throughput measurement. After correction, both ON/OFF C1 and C4 benchmark
waves complete with 32768 input and 256 output tokens per request.

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

## Completed model and public-control audit

The serving source is `9faa014636`; both conditions use the same native hash,
launch arguments and environment except the long-prefill attention switch.
Four V100s run the retained Qwen3.8-27B-QUASAR-NVFP4 checkpoint, FP16 target
computation, E4M3 target KV and DFlash2 with seven speculative tokens and FP16
draft KV. Context capacity is 262144, chunk budget 8192, maximum sequences 32,
and GPU memory utilization 0.85. Workers report FULL_AND_PIECEWISE and
enforce_eager=false. Variable-length prefill is dispatched outside attention
capture regions; this is not a whole-request prefill graph claim.

Each task has a deterministic 32768-token WikiText background. This activates
the long-prefill route but modifies the standard short-prompt benchmark.
Actual prompts have 32890-33141 tokens. Sampling is temperature 1.0, top-p
0.95, top-k 20, thinking enabled, concurrency four and max output 65536.
The same 32 items and request hashes are used on both sides of each suite.

| Suite | OFF pass | ON pass | OFF-only / ON-only |
| --- | ---: | ---: | ---: |
| GSM8K | 27/32 | 31/32 | 0 / 4 |
| MATH-500 | 32/32 | 32/32 | 0 / 0 |
| MBPP sanitized | 25/32 | 25/32 | 0 / 0 |
| Total | 84/96 | 88/96 | 0 / 4 |

All 192 responses end naturally; none reaches the output limit. The longest
has 25192 output tokens. No scorer exceptions occur. MBPP has the same seven
assertion failures on both paths, with no discordant pass/fail outcomes.
Both GSM8K responses for item830 express 18 hours 48 minutes, equivalent to
1128 minutes, but the retained integer extractor selects 48. Primary scores
are not changed after manual inspection. GSM8K's exact paired McNemar p-value
is 0.125; a small single-seed audit does not establish non-inferiority or an
accuracy improvement.

The specialized route count increases on all four ON ranks in each suite
and remains zero OFF. All retained cache snapshots have zero internal and
external prefix hits and zero preemptions. Counts establish prefix use, not
specialization of every final task token: shorter terminal chunks can use
the fallback. Cold four-integer retrieval succeeds at all four lengths
(32768, 131072, 256000, 262016) on both paths. At 256000 input tokens, TTFT is
103.795883 s ON (2466 input tok/s) and 124.822676 s OFF (2051 input tok/s).
These are single cold request observations, not pure decode measurements.

An independent paired CUDA Graph control forces PyTorch 2.10 efficient SDPA
with matching lower-right causal bias and stride-zero shared-KV expansion.
No dense mask, repeated KV or extra output copy is materialized. Across 50
ABBA trials, Q8192/KV128000 medians are 89.707 ms candidate and 414.028 ms
SDPA; KV256000 medians are 183.043 and 890.064 ms. Paired ratios are 4.610
and 4.877. FP64 same-input and changed-input checks pass. These are separate
paired runs, not timings mixed with the stronger internal exact baseline.

Receipts include `external-sdpa-paired-gpu1.json`, full task response objects,
per-item outcomes, model/dataset/source/native hashes, and serving counters
in the precision-revision paper artifact. The measured implementation remains
`9faa014636`; this post-validation documentation changes no runtime source.

## Review scope

The declared numerical and sampled model checks are complete. Keep the PR in
Draft for independent review of synchronization, the finite-score range
contract and the statistical limits. Mixed storage retains rounding error;
many flagged tiles can raise repair cost. Neither these tests nor the public
control qualify universal task parity or a dual-FP32 75 TFLOP/s endpoint.
