# Local ConvRot before TP gather: preserve exact outputs

The unchanged **1344x768, 39-frame, 24-FPS, seed-42, TP4 GPU0-3** workload now
completes 20 denoise updates in **61.538397 seconds**, versus 62.434779 seconds
in [FLASHINFER_V4.md](FLASHINFER_V4.md). This saves **0.896382 seconds (1.44%)**,
giving **3.076920 seconds/update and 56.286054 useful TFLOPS/card**. Final
video/audio latents are bitwise equal on all four ranks, and the freshly
decoded MP4 is byte-identical to the preceding retained output.

The below-50-second and >80-TFLOPS/card targets remain incomplete. This is one
unprofiled development measurement after warmup, not formal three-run
acceptance. The underlying video's human quality review remains pending;
optional residual sequence parallelism still defaults off.

## Remove duplicated rotation

Previously, each rank gathered all FP16 residual rows and then applied the
same 256-channel ConvRot before its QKV and gate/up projections. The rotation
is independent for each row: compute it on that row's owner first, then
all-gather the exact resulting FP16 bits. This removes three redundant copies
of those rotations without changing any matrix multiply or TP summation.

H3-specific column-linear subclasses accept an explicit `input_is_rotated`
argument. They keep the normal module call and complete gathered input shape,
so forward hooks and useful-FLOP accounting remain intact. The special path
requires bias-free INT8 ConvRot with group size 256 and CUDA FP16 inputs.
CPU, unquantized, ordinary column calls and unsupported pre-rotation cases
retain the original path. No mutable tensor attributes, stream scheduling,
new deployment flag, cache budget change or communication algorithm is used.

The existing FP16 normalization boundary, signed INT8 codes/scales, FP32
ConvRot intermediates, GEMM accumulation and residual reduce-scatter remain
unchanged. Both native CUDA binaries are unchanged from the preceding build.

## Select from measured alternatives

Seven alternating microbenchmark trials after fixed warmup measure the
maximum rank time at physical M12352, K5376. All candidates preserve output
bits on every rank:

| Gather, rotation and projection | Original | Local rotation | Best overlap |
| --- | ---: | ---: | ---: |
| QKV N5376 | 9.262080 ms | 8.915968 ms | 8.767488 ms |
| Gate/up N7168 | 11.950080 ms | 11.578368 ms | 11.716608 ms |

The overlap prototype uses checked clones of the baseline zero-workspace
cuBLASLt algorithm, copies FP16 chunks with real PyNccl all-gather and restores
the original row order. It does not perform cross-rank arithmetic. Nevertheless,
the model's first two updates select the simpler implementation:

| First two updates of the existing 21-position schedule | Seconds |
| --- | ---: |
| Original implementation | 6.203683 |
| Local rotation only | 6.136137 |
| Local rotation plus QKV gather/projection overlap | 6.149361 |

Every final video/audio prefix latent is bitwise equal. Each rank still
performs 346,375,340,509,184 useful FLOPs. The extra stream, chunk copies and
plan-cloning path are **not integrated** because they do not improve this
model comparison. This is a diagnostic schedule prefix, not two-step video
generation or a projected full-run acceptance result.

## Current bottleneck

A fresh Nsight Systems trace of the retained implementation measures a
synchronized two-update interval of **6.170193 seconds** on rank 0:

| Exclusive GPU wall category | Seconds | Interval share |
| --- | ---: | ---: |
| GEMM | 2.535715 | 41.10% |
| Independent FlashInfer attention | 2.456298 | 39.81% |
| TP communication | 0.833928 | 13.52% |
| Other kernels | 0.234069 | 3.79% |
| ConvRot | 0.062709 | 1.02% |
| Copies | 0.012582 | 0.20% |
| No recorded GPU activity | 0.034893 | 0.57% |

The unchanged kernel's earlier ConvRot service was 0.122461 seconds in the
pre-V4 trace. The new approximately halved work matches the local-rotation
change; these separate traces are not a fixed-clock full-run A/B experiment.
The current dominant work remains GEMM, attention and communication. Further
tiny rotation optimizations cannot close the **11.538397-second** gap to 50
seconds. >80 TFLOPS requires less than **43.296920 seconds** for this workload.
The unchanged attention binary's NCU operand/issue counters remain in
[FLASHINFER_V4.md](FLASHINFER_V4.md); no new NCU capture is claimed here.

Two additional exact-output attention designs were tested without changing
the native binary:

- Q64/K64 with aliased K/V storage targets two resident CTAs. The initial,
  delayed-V-load and unroll-1 builds each use 128 registers and 16 bytes of
  spill stores/loads. The unroll-1 control passes all 13 short/tail/N12323
  bitwise checks but regresses **22.945791 ->25.735168 ms** at 1470-1477 MHz.
  It is rejected without a video run.
- The previously validated Q32/N64 warp design was updated to the current
  68-half P stride and four-row V stores, allowing K/V fragments to serve two
  query fragments. It uses 168 registers without spills and passes all 13
  bitwise controls, but regresses **22.504448 ->23.595009 ms** at 1507-1522 MHz.
  The changed layouts do not make that reuse strategy faster. It is rejected.

## Full schedule, validation and resources

The full run retains Comfy INT8 ConvRot, the exact 200-weight cache within
10 GiB, prompt-verified cached TP4 text conditioning, 21 sigma positions,
video/audio shifts 12/3 and one single-update warmup. There is no LoRA,
approximate activation caching or step skipping. Useful work remains
**3,463,753,579,661,312 FLOPs/rank**; moving auxiliary rotations does not add
to the numerator. Denoise timing includes computation, communication and
four-rank synchronization.

- Affected suite: **123 passed, 2 skipped, 1 deselected**.
- The new INT8 TP4 block regression passes on every rank for valid lengths
  33/131, with and without the exact FP16 weight cache. It checks local-row
  rotation dispatch, bitwise block outputs, unchanged FLOP hooks and FP32
  residuals above FP16's finite range. The existing unquantized TP4 block
  regression also passes on every rank.
- The initial combined two-module torchrun hung after the first pass markers
  and was stopped. It is not counted as a successful run. Each distributed
  module then passes in its own torchrun process; the passing main suite was
  not repeated. The raw log does not establish the precise hang cause.
- Full video/audio latent maximum error and relative L2 are **zero** before
  fresh VAE decoding. All automatic media checks pass; MP4 SHA256 remains
  `5f1e58e555d613ed28e3793d02022b914fc62486190b9bfcf6300648a1974fb2`.
- Prior native attention sanitizer results remain applicable because neither
  CUDA binary changed. CPU checks and Ruff/mypy/SPDX/pre-commit cover the new
  Python path; no new full-model CUDA Graph acceptance is claimed.

Torch DiT peak remains **15.197946 GiB/card**; NVML peak is
**17.226074 GiB/card**. Median clocks are 1380/1455/1462/1455 MHz, median powers
284.480/267.001/272.884/272.042 W and maximum temperatures 57/60/57/63 C.
Utilization medians are 100%, with throttle masks 0/4. No clock, power, ECC or
NCCL protocol settings change. Separate VAE decode takes 6.684605 seconds.
Profiler execution is separate from the full unprofiled timing run.

## Reproduction and rollback

Use the pinned Python 3.12.13, Torch 2.10.0+cu128, CUDA Toolkit 12.8.93,
Transformers 5.15.1 and checkpoint revisions from the preceding documents.
The same `FLASHINFER_SM70`, `--residual-sequence-parallel`, measured FP16 cache
list and `--num-frames 39 --num-inference-steps 21` configuration selects this
path. No new CLI option or CUDA rebuild is required. Reproduce the preceding
code at `53780abef0e9ef190aa55a1f61d8fd5fbd390aa1` in a separate worktree.

Run the affected suite as in [FLASHINFER_V4.md](FLASHINFER_V4.md), and launch
`tests/video/test_h3_local_rotation.py -k gpu` and
`tests/video/test_h3_residual_parallel.py -k gpu` in **separate**
`torchrun --standalone --nproc-per-node=4 -m pytest -q` processes on a leased
GPU0-3 group. Keep the virtual environment and explicit task-owned caches.

Raw evidence is under the campaign's `feeding-gather/`: `HANDOFF.json`,
immutable micro/model scripts and job records, `gather-results.json`,
`outputs/model-probe/results.json`, native test logs,
`outputs/native39-20steps/FLASHINFER_SM70/` with MP4/WAV/latents/screenshots and
NVML curves, quality/resource summaries, `steps-profile.nsys-rep/.sqlite`,
`steps-breakdown.json/.csv` and rejected attention builds/controls. Prototypes
target their recorded base source; do not apply their monkeypatches to the
new native implementation. No weights or build products are committed.
