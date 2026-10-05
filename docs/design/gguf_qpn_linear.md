# Shared-activation GGUF single projections on SM70

The single-matrix operator reuses the gated pair's source readers and
activation staging. The first N32 variant exposes 160 CTAs for the TP4
N5120/K4352 down projection, but only 256 threads per CTA. Nsight Compute
reports 64 registers/thread, 33.79KiB shared memory, 25% theoretical
occupancy and 39.18% L1/TEX wait contribution. It passes numerical checks
but gives small gains and regressions for several formats.

The N64 variant uses 512 threads to compute two N32 subtiles from the
same staged activation. Two global K partitions keep the grid at 160
CTAs. Every writer publishes FP32 partials before a completion ticket;
the last CTA reduces partitions in a fixed order and resets its counter.
There is no separate reduction launch. Counters start at zero and scratch
belongs to each prepared layer. Concurrent invocations need separate
scratch; ordinary worker graph replay uses the layer's sequential stream.

Nine existing original-byte readers are instantiated. The canonical
variant consumes prepared U2/group16 or U4/group32 code/stat streams and
uses the existing TurboMind operand transform. It does not unfold source
superblocks again or allocate a second canonical weight stream.

## Precision and numerical checks

Original readers retain source scale levels and their existing arithmetic.
MMA and both reductions accumulate in FP32; only final operands and
outputs use the established FP16 rounding. The original-format K256 unit
basis tests recover every operand in the first 64 real rows exactly against
official GGUF dequantization followed by FP16 conversion. Three independent
M8 inputs compare the complete N5120/K4352 matrix to official FP32 weights
and FP32 GEMM. Worst native relative L2 is 0.0003254; all outputs are finite.

Canonical-stream basis tests recover the existing canonical FP16 operands
bitwise. Their expanded FP16 coefficients retain the pre-existing
transcoding error against original GGUF; this route adds no coefficient
rounding. Both prototype families pass repeat graph equality and counter
reset checks, including the short K256 uneven/empty-warp case.

## Cold graph measurements

Real rank0 TP4 down weights, V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128,
Python 3.12.14, complete normal packages `dev43+g05ef35e37c` and
`dev44+gfd8eea3cb1`. All ABBA arms record 1290/877MHz and 300W.
Each arm has 84 graph samples; 16MiB L2 eviction is outside the timed
interval. Effective bandwidth counts weight-stream bytes only.

| Source | Layer | Original bytes | Canonical bytes | Canonical ABBA us | N64 native ABBA us | Native GB/s |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| IQ2_S | 0 | 7137280 | 11141120 | 47.104 / 46.080 | 38.912 / 38.912 | 183.4 |
| IQ3_S | 1 | 9574400 | 11141120 | 41.984 / 41.984 | 35.840 / 35.840 | 267.1 |
| IQ3_XXS | 4 | 8529920 | 11141120 | 40.960 / 40.960 | 37.888 / 37.888 | 225.1 |
| IQ4_XS | 5 | 11837440 | 12533760 | 38.912 / 38.912 | 40.960 / 39.936 | 296.4 |
| IQ2_XS | 11 | 6440960 | 11141120 | 43.008 / 43.008 | 35.840 / 35.840 | 179.7 |
| Q2_K | 16 | 7311360 | 11141120 | 36.864 / 36.864 | 52.224 / 52.224 | 140.0 |
| Q4_K | 28 | 12533760 | 13926400 | 34.816 / 34.816 | 57.344 / 57.344 | 218.6 |

| Prepared stream | Generic canonical ABBA us | QPN canonical-stream ABBA us | Decision |
| --- | --- | --- | --- |
| Q2_K | 37.888 / 37.888 | 34.816 / 34.816 | QPN wins |
| Q4_K | 34.816 / 34.816 | 38.912 / 38.912 | Keep generic canonical |

The same-shape NVFP4 QPN2 down operator reads 12533760 bytes and measures
26.624us, or 470.8GB/s. The native winners remain slower than that
reference, so these measurements establish incremental admission rather
than parity. Offline resources show 62–64 registers/thread and zero stack
or local spills for every N64 variant; register spilling does not explain
the Q4_K regression. Do not resume raw Q2_K/Q4_K superblock tuning.

Run the package benchmark under the GPU leases:

```bash
python benchmarks/kernels/benchmark_gguf_native_linear.py \
  --model TARGET.gguf --nvfp4-model NVFP4_DIR --n64 --output down.json
python benchmarks/kernels/benchmark_gguf_native_linear.py \
  --model TARGET.gguf --canonical-qpn --output canonical-down.json
```

## Model dispatch boundary

Capabilities admit only M8/N5120/K4352, FP16 activation and SM70.
Original records are retained for IQ2_S, IQ2_XS, IQ3_S and IQ3_XXS down
projections. Q2_K uses its existing canonical streams. IQ4_XS and Q4_K
keep generic canonical dispatch, reporting the measured regression.
Other M use the exact existing single-projection policy through the same
opaque runtime-M operator, without adding a concatenation. Startup reports
include the selected operator, calibrated interval and fallback reason.

The four native formats cover 36 layers; Q2_K covers one more. Retained
original records add 318218240 bytes per rank. The 37 layer workspaces and
counters add 12136000 bytes. Q2_K adds no weight copy. These are explicit
buffer increments, excluding allocator reservation and graph pools.

Installed-package CPU checks cover admission, disabled/incompatible
devices and dtypes, canonical fallback and dynamic export. Compilation from
a prefill example retains one opaque operator for dynamic M1–8192. The
compiled GPU layer gate checks M512/8/1/5/16/20/32/8, eager/compiled and
graph equality, other-M bitwise canonical results and cold-graph ABBA.

Compiled native results preserve the raw operator gains for all four
original formats. The fifth Q2_K instance initially hits the test harness's
shared forward-code recompile limit. Resetting the compiler between independent
instances and retrying only that case completes the gate: compiled Q2_K is
34.816us versus canonical 36.864us in both ABBA arms at 1290/877MHz.
All eight M values pass graph equality and other-M bitwise canonical checks.
The complete integrated package also passes 75 adjacent CPU checks.

About 0.21ms per round is estimated from native per-layer deltas and the
actual layer counts; no full-model gain is measured here. The full 16-prompt
1K/8K and C4 run follows the merged projection batch, with complete round
time, emitted tokens per round and time per emitted token reported together.
