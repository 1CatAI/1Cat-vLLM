# SM70 DeepSeek V4 Sparse MLA QK D-Split

## Scope

- Dependency: SM70 sparse MLA split-K PR #163
- Task branch: `agent/v100-dsv4-sparse-mla-dsplit-20260802-193030`
- Hardware: 8 x V100-SXM2-32GB, TP8
- Decode shape: q=1, eight query heads/rank, packed `fp8_ds_mla` KV
- Runtime gate: `VLLM_SM70_DSV4_SPARSE_MLA_QK_DSPLIT=1`

The route remains default-off until the full-model speed, deterministic output,
official-sampling quality, and graph replay gates pass.

## Measured Bottleneck

After the first split-K implementation, one C4 stage-1 launch still used only
40 CTAs on 80 SMs. Exact-shape NCU measured 255 registers/thread, 33.28 KiB
dynamic shared memory/CTA, 6.25% achieved occupancy, and 95.9% scheduler cycles
without an eligible warp. DRAM throughput was only 12.93%, so this was exposed
QK/dequant latency under resource pressure rather than an HBM limit.

## Implementation

The accepted candidate partitions the 512-dimensional QK dot product into
eight 64-dimensional tiles:

1. 320 QK CTAs compute FP32 score partials for the C4 shape.
2. A fixed-order reducer sums the eight dimension tiles, applies the scale and
   softmax, and stores FP32 max/sum plus FP16 probabilities.
3. PV is independently partitioned into 64-dimensional output tiles.
4. The existing FP32 split-K reducer combines partial states and writes FP16.
5. All scratch is obtained from the graph-safe worker workspace; the hot path
   performs no allocation or host synchronization.

## Microbenchmark Gate

CUDA Graph, seed 4111, 1000 iterations:

| Shape | Existing split-K | QK D-split | Reduction | Max abs error |
|---|---:|---:|---:|---:|
| C4, main 128 + extra 320 | 0.08565 ms | 0.03061 ms | 64.3% | 7.63e-6 |
| C128, main 128 + extra 10 | 0.06286 ms | 0.02269 ms | 63.9% | 3.05e-5 |
| SWA-only, main 128 | 0.06278 ms | 0.02153 ms | 65.7% | 3.05e-5 |

Three additional C4 seeds remained finite, with max error from 7.63e-6 to
1.53e-5 and graph means from 0.03117 to 0.03357 ms.

Direct old-route versus candidate checks tighten the numerical evidence:

| Shape | Different FP16 elements | Max abs error |
|---|---:|---:|
| C4 | 0 | 0 |
| C128 | 1 | 3.8147e-6 |
| SWA-only | 1 | 1.9073e-6 |

Compute Sanitizer reports zero errors for the C4 candidate.

The C4 QK tile NCU comparison is:

| Metric | Existing split-K | QK D-split |
|---|---:|---:|
| Grid | 40 CTAs | 320 CTAs |
| Registers/thread | 255 | 95 |
| Dynamic shared memory/CTA | 33.28 KiB | 4.10 KiB |
| Achieved occupancy | 6.25% | 21.65% |
| No eligible warp cycles | 95.9% | 82.63% |

NSYS attributes about 12.37/2.12/6.83/4.42 us to QK tiles, QK reduction,
PV tiles, and final reduction respectively. These values are kernel service,
not an end-to-end claim.

## Full-Model Transfer

The accepted performance run uses the same 1024-input/256-output, TP8,
official `temperature=1.0`/`top_p=1.0`, FP8 MLA KV, CUDA Graph, no-MTP
contract as the preceding stacked baseline. It also includes compact/direct
MXFP4 and the graph-skew-safe hierarchical all-reduce from PR #170.

| Metric | Existing split-K | QK D-split | Change |
|---|---:|---:|---:|
| TPOT, three-run mean | 23.460869 ms | 20.765477 ms | -2.695392 ms (-11.49%) |
| Decode throughput | 42.6242 tok/s | 48.1569 tok/s | +12.98% |
| Candidate TPOT samples | - | 20.767631 / 20.758487 / 20.770312 ms | 0.006200 ms stdev |
| Completed output | 256 tokens | 256/256 tokens in all three runs | pass |

The graph-node trace was taken with the serialized safety version of PR #170,
so its absolute traced TPOT is 23.258 ms and is not the accepted endpoint
speed. On the critical rank, sparse-MLA service is 1.350 ms/token:

| Sparse phase | Service per token |
|---|---:|
| QK dimension tiles | 0.592 ms |
| QK reduction/softmax | 0.124 ms |
| PV output tiles | 0.459 ms |
| Final split-K reduction | 0.174 ms |

The preceding split-K trace was about 3.48 ms/token, so the measured service
reduction is about 2.13 ms/token. Category service can overlap and is not an
additive wall-clock decomposition; the unprofiled table above is the endpoint
claim.

The first combined run exposed an existing one-slot race in hierarchical
all-reduce: two short requests completed, then a formal request stopped after
22 tokens. Disabling custom all-reduce completed 128/128 tokens at 23.198
ms/token, isolating the issue from sparse arithmetic. PR #170 now alternates
signal/partial slots and performs clique consumption acknowledgement in
parallel with pair exchange. Its 8,700-collective graph stress and the three
full endpoint runs above all complete. The QK route therefore depends on that
skew-safe PR #170 revision when both gates are enabled.

Artifacts:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-sparse-mla-dsplit-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-sparse-mla-qk-dsplit-fullmodel-20260803/
```

## Rejected Variants

| Variant | Result | Decision |
|---|---:|---|
| Score-only split followed by PV tiles | 0.10887 ms C4 | Reject; QK retained the 255-register bottleneck |
| 32-dimensional QK tiles | 0.02944 ms C4 | Reject; output contained NaN |
| 128-dimensional QK tiles | 0.03716 ms C4 | Reject; slower than 64D |
| `BLOCK_H=4` | 0.04340-0.04503 ms C4 | Reject; duplicated KV work |
| Two or eight stage-1 warps | 0.03635/0.03743 ms C4 | Reject; four warps is faster |

## Remaining Gates

1. Complete the independent official-sampling text-health and model quality
   suite; technical-prompt completion alone is not a quality pass.
2. Sweep long-context and concurrent decode before changing the default.
3. Validate on a second V100 host and matching two-clique TP8 topology.
