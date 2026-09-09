# FlashInfer feeding: reject communication overlap after error amplification

The investigation baseline is **63.565580 seconds for 20 updates**, or
**54.491025 useful TFLOPS/card**, with the contract and quality evidence in
[FLASHINFER_SILU.md](FLASHINFER_SILU.md). Its exact-output successor takes
62.434779 seconds; see [FLASHINFER_V4.md](FLASHINFER_V4.md). The below-50-second
and >80-TFLOPS/card targets remain open. Subsequent optimization prioritizes
bitwise video/audio latent equality to this baseline, as requested by the user.

## Fresh bottleneck measurement

An Nsight Systems capture of source `5b4f0bba31`, with the retained FlashInfer
and TurboMind binaries, measures the first two updates of the existing
21-position schedule. It retains 1344x768, 39 frames, 24 FPS, seed 42, TP4
GPU0-3, INT8 ConvRot, the 200-weight cache within 10 GiB, FP32 residual
sharding, and video/audio shifts 12/3. This is a diagnostic prefix, not a
two-step generation schedule or full-denoise acceptance measurement.

Rank 0's synchronized denoise interval is 6.359245 seconds. The categories
below are exclusive GPU wall intervals from that same capture, including
the final four-rank synchronization boundary; they are not summed durations
from overlapping streams.

| Category | Two-update seconds | Interval share |
| --- | ---: | ---: |
| Independent FlashInfer attention | 2.589364 | 40.72% |
| GEMM | 2.525066 | 39.71% |
| TP communication | 0.839831 | 13.21% |
| Other kernels | 0.237261 | 3.73% |
| ConvRot | 0.122461 | 1.93% |
| Copies | 0.012780 | 0.20% |
| No recorded GPU activity | 0.032481 | 0.51% |

Thus host starvation is not the main observed problem. The earlier hardware
counters for this unchanged attention binary still identify shared operand
traffic, MIO pressure, and limited eligible warps; see
[FLASHINFER_VTRANSPOSE.md](FLASHINFER_VTRANSPOSE.md). The generic NCCL kernel
name containing `RING_LL` does not establish the selected protocol. No NCCL
protocol, power limit, clock, or ECC setting was changed in this experiment.

## Projection/communication experiment

The artifact prototype computes a row chunk on the main stream while the
preceding chunk undergoes actual PyNccl FP32 reduce-scatter on a separate
high-priority stream. It keeps the existing input scaling, 256-channel
ConvRot, column-major FP16 weights, and FP32 GEMM accumulation. Physical
M=12352 is packed as four rank partitions of 3088 rows. A short remainder is
folded into the last chunk; no extra model rows are computed or counted.

A checked cuBLASLt plan clone preserves the complete baseline algorithm and
uses `cublasLtMatmulAlgoCheck` to validate smaller row counts with zero
workspace. This is necessary because some valid small shapes omit that
algorithm from the first 32 heuristic entries. No operator replacement or
extra arithmetic is credited as useful work.

| Projection + reduce-scatter | Full projection | Best overlap | Local chunk |
| --- | ---: | ---: | ---: |
| N5376/K1792 | 5.285888 ms | 4.226048 ms | 512 rows |
| N5376/K3584 | 8.368128 ms | 7.529472 ms | 1024 rows |

These are maximum-rank medians from seven alternating trials after fixed
warmup, not full-model speedups. Reassembled GEMM outputs are bitwise equal.
After FP32 reduce-scatter the relative L2 error is about 3.3-3.6e-8, with
maximum absolute errors 6.103516e-5 and 9.765625e-4 respectively. Changing
collective payload size changes the floating-point summation order.

The model's first two updates improve from 6.325525 to 6.212453 seconds
(1.79%). Each rank executes 200 overlapping projections and 900 chunks;
useful work remains 346,375,340,509,184 FLOPs/rank. Video/audio latent relative
L2 is initially 1.498031e-5 / 1.330002e-5.

One unprofiled complete short schedule then takes **62.321609 seconds**, or
55.578693 useful TFLOPS/card: only 1.243970 seconds saved. Full-denoise useful
work remains 3,463,753,579,661,312 FLOPs/rank. At the end of sampling, however,
video/audio latent relative L2 grows to **0.019131655 / 0.008072156**, with
maximum absolute errors 0.947431 / 0.015767. The resulting MP4 passes automatic
decode/frame/dimension/audio/black/static checks. This does not establish
human quality acceptance or justify treating numerical drift as harmless.

Following the user's request to minimize growing error, this route is
**not retained**. The uninstalled integration patch and prototype are archived;
all corresponding source edits were reverted, its native build was cancelled,
and neither production binary was replaced. Sampling, TP sum order, and
default runtime behavior therefore remain at the retained baseline.

## Failed paths and artifacts

- The first micro harness incorrectly required the fixed Lt algorithm to
  appear in every small shape's heuristic list; it stopped before meaningful
  timings. Later controls explicitly report unsupported shapes and validate
  cloned plans rather than guessing compatibility.
- An initial probe extension reused the native C++ pybind type name, causing
  registration failure and an unwanted fallback JIT attempt. A distinct C++
  type, explicit native preloading, and a task-owned extension cache fix the
  harness. The failed attempt has no valid timing or installed binary.
- The first model harness omitted `stage_durations` before warmup. All ranks
  failed before candidate execution; the corrected version initializes it
  before each warmup. No failed run is included in the reported timings.
- Keeping softmax maximum/denominator in registers passes all 13 short/tail/
  N12323 bitwise controls with 128 registers and no spills, but changes the
  isolated median only from 24.543232 to 24.503296 ms (0.16%). This is too small
  to retain or justify a video run.

Raw evidence is under the campaign artifact directory `feeding-overlap/`:
`manifest.json`, `HANDOFF.json`, `steps-profile.nsys-rep`, `steps-profile.sqlite`,
`steps-breakdown.json/.csv`, `projection-overlap-v5-results.json`,
`outputs/model-probe-v2/results.json`, and
`outputs/quality-prototype/FLASHINFER_SM70/`. The latter video is an explicitly
rejected experiment, not a replacement for the retained quality artifact.
Scripts, full arguments, queue/lease records, source and binary snapshots,
failed logs, and the unpromoted patch are preserved outside Git.

The subsequent exact-output V-store improvement and its full-model evidence
are in [FLASHINFER_V4.md](FLASHINFER_V4.md). It changes neither arithmetic,
softmax grouping, TP reduction, nor sampling.
