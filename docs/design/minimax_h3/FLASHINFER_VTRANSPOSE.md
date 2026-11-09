# FlashInfer warp transpose: remove a shared staging round trip

The unchanged 1344x768, 39-frame, 24-FPS, seed-42, TP4 GPU0–3 development
workload completes **20 denoise updates in 64.920336 seconds**, or
**3.246017 seconds/update and 53.353907 useful TFLOPS/card**. The preceding
probability-layout operator takes 65.804661 seconds. This additional 1.34%
gain preserves video/audio latents and the freshly decoded MP4 bitwise.

The run enables optional FP32 residual sharding and the exact 200-weight cache
within 10 GiB. It uses verified cached TP4 text conditioning, 21 sigma positions,
video/audio shifts 12/3, one single-update warmup and one unprofiled full short
schedule. It is not an end-to-end or three-run acceptance measurement.
The below-50-second and >80 useful TFLOPS/card targets remain incomplete;
residual sharding still defaults off pending human quality review.

## Implementation and controls

Each warp prefetches an 8-row by 32-column K/V tile in 128-bit vectors.
Adjacent K/V rows belong to lanes differing in bit 2. A warp shuffle exchanges
each exact FP16 pair: the even-row lane writes even columns, and its partner
writes odd columns into the transposed V tile. This removes the store/load
through dead probability storage and one CTA barrier per subsequent key tile.
CTA barriers still protect K/V consumption and the next iteration's reads.
Unaligned-input fallback, tail zeroing, attention scale, softmax, FP16 conversion
and FP32 HMMA accumulation retain their numerical contracts.

The native kernel uses **125 registers, zero spills, 89,088 shared bytes and
512 threads**. Shared memory still permits only one CTA/SM. FlashInfer remains
an independent SM70 D128 operator. Its native binary SHA256 is
`23af8c6b5cdf8a4d1ce9b319385fd840ad94f23e865698d8199837272855b116`.

| Native operator control | Probability-layout median ms | Warp transpose median ms |
| --- | ---: | ---: |
| S12323/H14/D128, five alternating pairs | 24.342527 | 23.553024 |
| S73483/H14/D128, three alternating pairs | 852.479980 | 828.352539 |

Short-control clocks are 1522 MHz for both operators. Long-control clocks are
1507 MHz for the baseline and 1492 MHz for the candidate. All 13 short/tail
lengths and the long reference pass with bitwise equality. Qualified module
names and immutable binary hashes verify distinct implementations are loaded.

## Complete-denoise resources and quality

Each rank's useful work remains 3,463,753,579,661,312 FLOPs, excluding padding,
rotation and conversion. Torch peak allocation is 15.279062 GiB/card; NVML
peak is 17.433105 GiB/card. Fresh VAE decode takes 5.619666 seconds separately.

| Native denoise NVML diagnostic | GPU0 | GPU1 | GPU2 | GPU3 |
| --- | ---: | ---: | ---: | ---: |
| Median GPU utilization (%) | 100 | 100 | 100 | 100 |
| Median SM clock (MHz) | 1380 | 1477 | 1485 | 1485 |
| Median power (W) | 280.298 | 269.487 | 273.887 | 275.178 |
| Maximum temperature (C) | 55 | 59 | 55 | 62 |

Throttle masks are 0/4. The queried power limit, default and maximum are all
300 W; no power, clock or ECC setting changes. Busy percentage does not
establish Tensor Core saturation or useful throughput.

Video/audio latent max error and relative L2 are zero against the preceding
probability-layout build. All automatic frame/FPS/dimension/audio/finite/
black/static checks pass. The fresh MP4 SHA256 remains
`5f1e58e555d613ed28e3793d02022b914fc62486190b9bfcf6300648a1974fb2`.
This establishes no additional regression on the fixed residual-sharding sample.
Its difference from replicated residuals and pending human five-axis review
remain documented in [FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md).

## Hardware counters and remaining bottleneck

Separate NCU 2022.4.1 captures use GPU1, S12323/H14/D128 and 24 replay passes.
Profiler timings and sampled stalls are diagnostic, not acceptance wall time.

| Diagnostic | Probability layout | Warp transpose |
| --- | ---: | ---: |
| Tensor pipe active (%) | 36.328437 | 37.57 |
| Profiler duration (ms) | 28.652480 | 27.575648 |
| Eligible warps / scheduler | 1.054176 | 1.01 |
| Issue active (%) | 51.801382 | 47.65 |
| Total PC samples | 1,436,187 | 1,391,829 |
| Short-scoreboard samples | 184,285 | 119,229 |
| MIO-throttle samples | 323,365 | 389,527 |
| Barrier samples | 104,231 | 123,204 |
| Ideal shared wavefronts | 1,695,227,184 | 1,632,650,544 |
| Excessive shared wavefronts | 1,042,944 | 147,055,104 |

Removing staging reduces required accesses and short-scoreboard samples, but
direct V stores introduce four-way conflicts. Total shared wavefronts increase
to 1,779,705,648; MIO remains the next operand-feeding bottleneck. Fewer barrier
instructions do not imply fewer sampled barrier stalls. DRAM throughput is
1.26% and L2 hit rate 97.13%; bulk HBM bandwidth is not the limiting resource.

The gap to 50 seconds is 14.920336 seconds. The >80-TFLOPS criterion is stricter:
this workload needs less than 43.296920 seconds. Further work must reduce
operand/issue pressure and assess TP communication overlap. The preceding
two-update NSYS split is context, not a fresh trace of this build.

## Validation, rejected paths and reproduction

- Affected video suite: **117 passed, 1 skipped, 1 deselected**. The unchanged
  separate TP4 residual-block test retains its preceding four-rank result.
- Memcheck, racecheck and synccheck: **12 cases each**, zero errors/hazards,
  covering tails, storage offsets, query groups and graph replay.
- A 16x16 warp tile regresses 24.570881 -> 24.941568 ms at 1507 MHz despite
  bitwise equality, 125 registers and no spills.
- A 4x64 tile reaches 24.366079 ms versus 24.566784 ms at 1507–1515 MHz and
  128 registers. Its gain is smaller than 8x32; neither alternative gets a
  full-video run.
- Initial alternative-tile generation also replaced an unrelated Q fragment
  lane mask and failed the oracle. Corrected versions limit edits to K/V staging.
  Failed artifacts are retained and were never installed.
- A follow-up V row permutation and row-dependent K-coordinate XOR targets
  the new store conflicts, retaining 128-bit fragment loads. Its first build
  spills 24 bytes and is not benchmarked. A compiler barrier limiting address
  lifetime removes spills at 128 registers and passes all 13 reference lengths
  bitwise, but improves the operator only 23.781376 -> 23.501823 ms at matched
  1507–1515 MHz. This 1.18% operator result remains experimental; it is not
  installed or credited as a full-denoise improvement, and no additional video
  is generated for it. Both variants remain under `feeding-operands/vwarp_swizzle*`.

Use the environment, fixed revisions, cache list and commands from
[FLASHINFER_TO50.md](FLASHINFER_TO50.md), with explicit `FLASHINFER_SM70` and
`--residual-sequence-parallel` from [FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md).
Rebuild the native extension after updating source. Reproduce the preceding
operator with `8da9bbfe2544b68841ca03782eac243e68c502ce` in a separate worktree
and rebuild; omit the residual flag for its independent rollback.

Raw evidence is in the campaign's `feeding-vtranspose/`: manifests, immutable
baseline, native controls, test/sanitizer logs, video/audio/latents, quality
summary, NVML PNG/SVG curves and NCU raw/source-counter exports. Prototype
and failed paths are under `feeding-operands/vwarp*`. No weights or build
products are committed. Dependency synchronization through `f825756607` imports
the parallel FlashAttention changes without changing the FlashInfer path.
