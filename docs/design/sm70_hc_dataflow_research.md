# SM70 HC dataflow prototype

This research operator combines HC residual update, grouped RMSNorm, down
projection, SiLU, up projection and gate mixing in one cooperative launch.
It is not registered in production dispatch and has no claimed model speedup.

## Schedule and precision

Norm, down and up/mix have fixed CTA owners. Down owners cache their two weight
rows before waiting for normalized input. Up owners cache eight output columns
across every HC branch before waiting for the gathered low-rank vector.
Communication is in the producer epilogues: write directly into each peer's
result buffer, then publish a system-release generation. Consumer lanes poll
only their prerequisite flags. No `grid.sync` is used.

Cooperative residency is checked from actual kernel occupancy and shared-memory
requirements. For H2560/HC4/R320/TP4, there are four norm owners, 41 down owners
and 80 up owners: 125 CTAs. The weight cache plus scratch requires 40,992 bytes
per CTA. The initial build uses 40 registers and no spills. These dimensions
describe the measured case, not a model/TP dispatch predicate.

The low-rank and injection rows are evenly distributed across peer ranks.
Logical layout divisibility, direct peer access and resident capacity are
required. M1 and M5 have template instances; other positive batch widths use
the same algorithm with a runtime loop, not another implementation. Batch
width does not increase the number of resident roles. Each row has independent
storage and generation checks survive changing graph widths.

Inputs, weights, materialized residual/normalization, projection boundaries and
outputs are FP16. Dot products, norms, sigmoid arguments and weighted mixing
use FP32. This dense HC reader currently accepts FP16 only; it does not change
the separate expert prototype's FP16/NVFP4 reader interface.

This is a standalone HC segment. It has two peer publication stages and does
not yet consume a preceding TP all-reduce or implement its reduction epilogue.

## Validation

An independent FP64 oracle preserves each FP16 boundary. Four-GPU tests cover
TP1/TP2/TP4 at H64 and TP4 at H2560, with widths 1/5/7/5/1/7/1/5 sharing the same
workspaces and counters. Residuals and injection change between replays and
unused output rows remain poisoned. All tests pass. The largest observed
absolute error across residual, normalization, low-rank, injection and final
output is 0.001953125. This local tolerance is not the model distribution gate.

Candidate-only C1 timing uses 100 dependent HC nodes per graph and five repeats.
It excludes trace instrumentation and CPU replay loops. Initial medians:

| TP | H | HC | R | C1 microseconds |
|---:|---:|---:|---:|---:|
| 1 | 64 | 4 | 32 | 24.300 |
| 2 | 64 | 4 | 32 | 26.450 |
| 4 | 64 | 4 | 32 | 25.948 |
| 4 | 2560 | 4 | 320 | 77.005 |

The actual-size case is far over the 10-us HC budget. These synthetic isolated
timings are not a matched comparison against the in-model path. The prototype
is not admitted; no full-model performance or quality run is justified yet.
An optional per-CTA globaltimer trace separates weight prefetch, prerequisite
waiting, computation/push and the final peer wait to localize this negative
result. Trace timestamps are diagnostic, with the V100 timer's measured
1.024-us resolution; they do not fill the model's missing DRAM-byte counters.

## Reader and norm follow-up

The scalar reader spent 36.864 us (median CTA duration) prefetching down weights.
128-bit read-only loads reduce that to roughly 7–8 us and up prefetch to 3.072 us.
Capability checks use alignment and row layout; incompatible views keep the
scalar reader within the same algorithm. C1 median drops to 53.156 us. Replays
and the FP64 boundary reference still pass.

Combine/norm initially reloaded materialized residuals from global memory.
Vector input/output and affine loads now preserve those FP16 values in shared
memory until normalization finishes. This keeps both FP16 boundaries and FP32
math. The actual-size C1 median becomes 44.339 us; all TP and width checks pass.
The build uses 47 registers with no spills.

Steady timing comes from the last node of a 100-node graph: single-node
cross-GPU replays include host launch skew. Median CTA phases after the norm
change are as follows. Norm's second column measures its own computation,
whereas down/up measure prerequisite waiting. Rows overlap and cannot be added
as a layer budget.

| Role | Prefetch | Input wait / norm | Compute and push | Peer wait |
|---|---:|---:|---:|---:|
| Norm | 0.000 | 6.144 | 0.000 | 0.000 |
| Down | 7.168 | 0.000 | 9.216 | 0.000 |
| Up | 3.072 | 20.480 | 10.240 | 5.120 |

An ordinary/cooperative dispatch probe uses the same argument struct, 125 CTAs,
128 threads, 40,992-byte dynamic shared memory, 1,000 graph nodes and alternating
arms. Five-run medians are 1.930 and 1.927 us per node, respectively. The large
remaining HC cost is not explained by cooperative dispatch itself.

Polling now uses volatile loads followed by one acquire after readiness, and
local norm flags keep GPU scope. Peer publication retains system ordering.
That build measures 38.543 us, with all boundary and replay checks passing.

The final build reuses the existing production HC FP32 sigmoid primitive,
including its exponent and full-precision division instructions. It measures
44.616 us; the four toy TP cases and actual-size replay checks pass, with the
same maximum boundary error of 0.001953125. The 38.543-us result belongs to the
previous sigmoid implementation and is not the final candidate's timing.

Final steady CTA phase medians are 7.168-us down prefetch, 1.024-us down input
wait, 9.216-us down compute/push; and 2.048-us up prefetch, 19.456-us up input
wait, 10.240-us up compute/push, 5.120-us peer wait. Norm computation measures
4.096 us. These overlapping phases localize the current dependency chain; they
are not additive layer attribution or actual DRAM counters.

The prototype remains above the HC budget and is parked. None of these
research results admits a default model path or demonstrates a full-model
improvement. A subsequent design must shorten the measured down-to-up chain
and incorporate the preceding reduction, rather than repeat this standalone
fixed-role launch unchanged.

Standalone GPU scripts keep a raw flock descriptor until process exit, so CUDA
teardown remains inside the ownership window.
