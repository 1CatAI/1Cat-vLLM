# SM70 projection and GDN screens

These screens are research-only. They do not change production dispatch or
establish end-to-end performance. Runtime: Torch 2.10, CUDA 12.8, V100-SXM2 at
300 W, cold 32 MiB L2 eviction, CUDA graph timing, original checkpoint weights
with rank-zero TP4 shards. Paired configurations retain the same grid and
arithmetic unless explicitly noted.

| Candidate | Control → candidate | Decision |
|---|---|---|
| FP4 shared per-lane lookup, gate/up | 47.104 → 68.608 µs | Reject |
| FP4 shared per-lane lookup, down | 27.648 → 33.792 µs | Reject |
| TP4 serial row-local norm | Critical median 17.408 → 28.672 µs | Reject |
| TP4 parallel row-local norm | All four rank medians regress | Reject |
| GDN gated-norm epilogue, complete layer | Mean 148.839 → 150.180 µs | Reject |

The lookup decoder reuses the original 16 KiB partial-reduction scratch. Its
gate/up register count increases from 50 to 56; down retains 48. At four input
amplitudes, output values compare equal. This check does not establish
signed-zero bit-pattern equality. Decode-only graphs also regress. Do not
promote the candidate on the basis of an isolated arithmetic estimate.

Both TP4 row-local norm variants preserve the original five CUB128 reductions
and their ordered final sum. All four ranks are bitwise at four amplitudes.
The serial variant raises registers from 56 to 70 without local spills. The
parallel 640-thread variant changes rank medians from
17.920/19.456/15.360/19.456 to 20.480/21.504/16.896/21.504 µs. Both retain one
compute kernel; their eviction-inclusive graph node count is 2 → 2.

The GDN epilogue retains BV2, one warp per delta CTA, and all eight FP32 state
snapshots. It changes the existing convolution initializer to a signalling-NaN
payload sentinel; only the first value tile per head waits for that head's
completed payload. There is no additional initialization launch. Snapshot and
core bit patterns match the control, and the final real-weight layer has zero
maximum difference in this screen. The candidate uses 64 registers. Unchanged
TP collectives are excluded from both timings. This restricted finite-input
protocol has not passed a production numerical gate. Its negative timing
precludes integration, without repeating teacher-forcing or model acceptance.

`benchmark_sm70_qpn2_publish_norm.py --eighty-block-screen` additionally prepares
an 80-CTA, 64-thread norm comparison. It splits each row into ten parts instead
of five and uses independent metadata/payload storage. Its norm reduction
order changes. A tolerance check is only a screen; model KL/top-1 admission would be
required for promotion. The measured result below does not justify promotion.

The 80-CTA tolerance screen was run on all four 9609 V100s. Independent cold
graph rank medians are 17.408/16.384/15.360/14.336 →
16.384/16.384/15.360/14.336 µs. A burst of 64 calls in one graph, with just
one leading eviction, reduces critical median per call from 8.000 to
7.272 µs. The latter amortizes CPU launch skew but is not a cold invocation
or a model-layer measurement. Even the optimistic 122-call estimate is only
0.09 ms; the changed reduction order is not admitted. Both graphs retain one
compute node per norm.

An additional decoder control folds the exact power-of-two 2^14 into each
already-rounded FP16 group scale. It changes neither activation packing,
weight addresses, the one-byte scale layout, grid, nor MMA/reduction order.
Its wrapper is restricted to abs(global inverse scale) ≤ 3/512. All 168
original checkpoint MLP global scales meet this conservative bound. An
exhaustive 557,056-case GPU bit check covers 16 FP4 codes and both signs of
all FP16 scales with magnitude below four. Real-weight complete output bits
also match at four input amplitudes. The paired cold-graph full kernel is
45.056 → 45.056 µs for gate/up and 25.600 → 25.600 µs for down on 9609.
Decode-only gate/up improves by one event-clock bin, 45.056 → 44.032 µs;
complete-kernel timing does not improve. Reject this fold alone rather than
claiming a model benefit or repeating expensive end-to-end gates.
