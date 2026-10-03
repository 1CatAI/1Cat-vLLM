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
