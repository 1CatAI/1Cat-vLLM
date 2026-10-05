# Canonical streams on the small-M QPN skeleton

When original superblock decoding loses to canonical GEMM, reuse the streams
already prepared by TurboMind. A group32 U4 affine reader already exists;
add a group32 IQ4 reader using the same packet layout and lookup transform as
canonical GEMM. The single-matrix and gated-pair candidates share the measured
QPN activation staging, FP32 accumulation and reduction bodies.

No new coefficient conversion or expanded lattice allocation is needed.
Canonical stats retain their existing precision and coalesced row stride.
The benchmark compares actual TP4 down and pure-type gate/up projections
using official reconstruction, three seeds, stable graph replay and cold-L2
ABBA timing. Candidates have no model admission until their real shapes win.
