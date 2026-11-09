# FlashInfer operand feeding: eliminate probability-tile bank conflicts

The unchanged 39-frame, 1344x768, 24-FPS, seed-42, TP4 GPU0–3 workload now
completes 20 denoise updates in **65.804661 seconds**, versus 66.863312 seconds
with the previous FlashInfer operator. This is a 1.58% full-denoise improvement,
or **52.636903 useful TFLOPS per card**. The below-50-second and >80-TFLOPS/card
targets remain incomplete.

This comparison enables the existing optional FP32 residual sharding and the
same 200-weight exact cache within 10 GiB. It uses 21 sigma positions and
video/audio shifts 12/3, verified cached TP4 text conditioning, one single-update
warmup and one unprofiled full short schedule. Frames, steps and arithmetic
precision are unchanged. It is not end-to-end or the formal three-run result.

## Located bottleneck and implementation

Instruction-correlated counters from the preceding operator identify two
different costs:

- Twelve QK shared operand loads account for 170,430 MIO-throttle samples.
  These loads already have zero excessive shared wavefronts; their issue/operand
  volume, rather than bank conflicts, remains a bottleneck.
- Eight probability-fragment loads each require twice their ideal wavefronts,
  while paired probability stores have four-way conflicts. Together they
  account for almost all of the preceding 235,879,168 excessive wavefronts.

The probability tile now uses a **68-half row stride** instead of 72 halves.
That stride distributes accumulator pair stores across distinct banks. Odd
rows are 8-byte aligned, so four explicit 64-bit loads populate each WMMA A
fragment, replacing its two conflicting 128-bit loads. The FP16 values and
fragment coordinates are unchanged. No softmax formula, reduction order,
FP32 accumulation, Tensor Core operation or attention scale changes.

The dead probability storage still exactly accommodates the next V tile's
transpose scratch. The kernel retains Q128/K64, 512 threads, 128 registers,
one CTA/SM and no spills. Dynamic shared memory falls from 90,112 to 89,088
bytes. Attention remains an independent FlashInfer-SM70 operator.

## Measurements and quality

| Unprofiled operator control | Previous median ms | Native median ms |
| --- | ---: | ---: |
| S12323/H14/D128, five alternating pairs | 25.292801 | 24.263680 |
| S73483/H14/D128, three alternating pairs | 883.228699 | 851.920898 |

Short-control clocks are 1522–1530 MHz; long-control clocks are 1507 MHz for
both operators. All 13 short/tail lengths and the long reference check pass.
All operator outputs are bitwise equal. Separate prototype controls showed
25.318399 -> 24.342527 ms at matching 1522–1530 MHz clocks.

Video and audio latents from the full native 20-update run are bitwise equal
to the previous residual-sharding source, with zero max error and relative L2.
Fresh VAE decode takes 5.682266 seconds and the MP4 is identical:
`5f1e58e555d613ed28e3793d02022b914fc62486190b9bfcf6300648a1974fb2`.
All automatic video/audio checks pass. This establishes no additional output
regression for the fixed test; the preceding residual-sharding route's human
five-axis review remains pending, and that option still defaults off.

Useful work remains 3,463,753,579,661,312 FLOPs/rank; padding, conversion and
rotation do not increase the numerator. Each card's Torch DiT allocation
peak is 15.279062 GiB. NVML peak is 17.439270 GiB on GPU0 and 17.433105 GiB
on GPUs1–3. These are denoise measurements, not whole-pipeline acceptance.

| NVML median during native run | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | ---: | ---: | ---: | ---: |
| GPU utilization (%) | 100 | 100 | 100 | 100 |
| SM clock (MHz) | 1380 | 1485 | 1492 | 1485 |
| Power (W) | 280.012 | 271.092 | 276.024 | 274.704 |
| Maximum temperature (C) | 55 | 59 | 56 | 63 |

Throttle masks are 0/4, with no clock or power setting changes. GPU busy
percentage is not a measure of effective TFLOPS.

## Fresh hardware counters

Separate Nsight Compute 2022.4.1 captures use GPU1, the same S12323/H14/D128
shape and 24 replay passes. The new capture verifies native binary SHA256
`caedce12dda69f65ed6c1be41680bf3c7f4f2841fbd3a7913e9350abb2a9c283`.

| Diagnostic | Previous operator | New operator |
| --- | ---: | ---: |
| Excessive shared wavefronts | 235,879,168 | 1,042,944 |
| Total shared wavefronts | 1,931,106,352 | 1,696,270,128 |
| Tensor pipe active (%) | 34.905979 | 36.328437 |
| Eligible warps / scheduler | 1.006326 | 1.054176 |
| Issue active (%) | 49.912548 | 51.801382 |
| Profiler duration (ms) | 29.819872 | 28.652480 |

The remaining excessive wavefronts come from initial V staging; the repeating
probability loads/stores have none. Total PC samples decrease from 1,494,689
to 1,436,187. Short-scoreboard samples fall from 261,126 to 184,285; barrier
samples fall from 112,292 to 104,231. MIO-throttle samples increase from
296,681 to 323,365, consistent with narrower load instructions adding issue
pressure while removing memory conflicts. These samples are not exclusive
wall-time categories, and profiler durations are not unprofiled acceptance.

The remaining 15.804661-second gap requires more than further padding changes.
Conflict-free operand traffic, readiness and instruction issue still limit
Tensor Core feeding. The preceding two-update NSYS split remains useful
context, but its percentages are not claimed as a new trace of this operator.

## Rejected paths and test-harness correction

- QK unroll-1: 25.322496 -> 25.636864 ms, 128 registers, no spills, same bits.
- QK compiler scheduling fence: 25.305088 -> 25.450497 ms, 128 registers,
  no spills, same bits.
- Skipping multiplication when every row's softmax rescale equals one causes
  16-byte register spills; no GPU benchmark was run.
- Explicit P register caching with delayed K/V prefetch: 25.096191 ->
  26.096640 ms, same bits. Both binaries have 52 static LDS128 and 320 static
  HMMA instructions: the compiler had already reused P loads. The later global
  prefetch hurts instead of reducing operand traffic.
- A Q32/N64 warp tile reuses K/V across two query fragments, uses 168 registers
  and 256 threads, but regresses from 25.085953 to 25.901056 ms with matching
  1530-MHz clocks. All 13 reference checks pass with identical bits. Initial
  prototypes had a one-key-fragment compile guard and a hard-coded 32-column
  output stride; those failures are retained and corrected before measurement.
- Q96/K96 reduces softmax iterations, uses 160 registers and 97,792 shared
  bytes with no spills, but regresses from 25.087999 to 25.882624 ms at matching
  1530-MHz clocks. All 13 reference checks pass; rounding differs because the
  key grouping changes. It is rejected without a full-video run.

The first native A/B harness loaded both libraries under `_h3_flashinfer_C`.
Python reused one module and one forward function, so that short/long control
was **invalid**; both raw result files explicitly record this. Qualified names
`baseline._h3_flashinfer_C` and `candidate._h3_flashinfer_C` now isolate the
libraries, and the harness asserts distinct modules/functions and the resolved
candidate path. The table above uses only the corrected controls. Earlier
differently named prototypes and single-library native model/tests/profiler
runs are unaffected.

## Validation and reproduction

The existing affected suite passes **117 tests**, with one distributed test
skipped and one parallel-backend case deselected. Compute Sanitizer memcheck,
racecheck and synccheck each pass **12** tail, storage-offset, independent-query
and graph-replay cases, with zero errors/hazards. No new test merely mirrors the
storage layout; existing numerical and memory-boundary tests cover it.

Use the unchanged commands and fixed Python3.12/Torch2.10+cu128/CUDA12.8,
Transformers5.15.1, checkpoint revisions and 200-weight cache list in
[FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md) and
[FLASHINFER_TO50.md](FLASHINFER_TO50.md). Rebuild the FlashInfer extension after
updating source. Keep explicit `FLASHINFER_SM70` and
`--residual-sequence-parallel` to reproduce this timing. Roll back only the
operator with parent `32ccd89339e0b9d4d8f1fdcd1ecafeec1831098d` in a separate
worktree and rebuild its extension; omit the residual flag for its independent
rollback.

Raw artifacts are in the campaign's `feeding-operands/`: `HANDOFF.json`,
`manifest.json`, source-correlated CSVs, `native_stride68_verified*-results.json`,
the invalid import-audit controls, sanitizer/suite logs, `quality-summary.json`,
`nvml-summary.json`, `attention-stride68.ncu-rep` and
`outputs/native39-20steps/FLASHINFER_SM70/` (MP4/WAV, latents, screenshots and
NVML PNG/SVG curves). Build products and weights are not committed.
