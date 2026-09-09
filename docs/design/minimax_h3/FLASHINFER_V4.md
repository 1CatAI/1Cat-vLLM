# FlashInfer four-row V transpose: preserve exact outputs

Complete denoise for the unchanged **1344x768, 39-frame, 24-FPS, seed-42,
TP4 GPU0-3** development workload improves from **63.565580 to 62.434779
seconds** for 20 actual updates. This saves **1.130801 seconds (1.78%)**,
giving **3.121739 seconds/update and 55.477950 useful TFLOPS/card**. Final
video/audio latents are bitwise equal on all four ranks, and freshly decoded
MP4 bytes match the preceding baseline.

The below-50-second and >80-useful-TFLOPS/card targets remain incomplete.
These are single unprofiled development measurements, not the formal
three-run acceptance. This change introduces no additional numerical error
relative to [FLASHINFER_SILU.md](FLASHINFER_SILU.md); human acceptance of the
underlying residual-sharded video's quality remains pending.

## Change and evidence

The previous independent FlashInfer D128 kernel writes four 32-bit half
pairs per lane when staging the next V tile. An additional exact lane exchange
transposes four rows at a time, permitting two aligned 64-bit stores. It
reduces shared-store instructions and bank conflicts while preserving every
FP16 bit. Arithmetic, QK/PV accumulation, softmax grouping, FP16 probability
rounding, TP collectives and sampling are unchanged. There is no additional
shared scratch buffer, barrier, global workspace or persistent weight cache.

The final native build uses **127 registers/thread, no spills, 512 threads and
89,088 shared bytes/block**. The kernel is implemented and dispatched inside
FlashInfer-SM70; no parallel FlashAttention operator is substituted.

Qualified imports and distinct module/function identities verify the paired
native controls, all with bitwise output equality:

| Native attention control on GPU 1 | Previous | Four-row stores |
| --- | ---: | ---: |
| N12323/H14/D128, five alternating samples | 23.865343 ms | 22.538240 ms |
| N73483/H14/D128, three alternating samples | 829.774841 ms | 795.958252 ms |

Short-control SM clocks are 1492-1507 MHz for both implementations. The long
control records 1485-1492 MHz for the baseline and 1470 MHz for the candidate;
do not present that pair as a fixed-clock comparison. The earlier prototype
on GPU 0 separately measures 25.269247 -> 23.933952 ms at 1417-1425 MHz.
Operator rates are not full-denoise effective throughput.

A separate Nsight Compute capture of the final binary confirms **39.65%
Tensor pipe activity**, versus 37.57% for its predecessor. Eligible warps rise
from 1.01 to 1.05 per scheduler, and issue activity from 47.65% to 49.08%.
Excessive shared wavefronts decrease from 147,055,104 to **134,539,776**;
MIO-throttle samples decrease from 389,527 to 370,668, while all PC samples
decrease from 1,391,829 to 1,317,635. Raw sample counts are not stall-time
fractions. MIO pressure remains substantial; the change does not eliminate
the operand bottleneck. DRAM throughput is 1.32% and L2 hit rate is 97.18%.
This profiled kernel capture is separate from unprofiled denoise timing.

## Full short schedule and quality

The run keeps Comfy INT8 ConvRot, the exact 200-weight FP16 cache within
10 GiB, optional FP32 residual sequence sharding, verified cached TP4 text
conditioning, 21 sigma positions, and video/audio shifts 12/3. One
single-update warmup precedes the complete 20-update schedule. There is no
LoRA, step skipping, approximate activation caching or communication overlap.

Useful work remains **3,463,753,579,661,312 FLOPs/rank**; padding, transpose
instructions and redundant copies do not increase it. The denominator
includes synchronized full-denoise computation, communication and waits.
The remaining gaps are **12.434779 seconds to 50 seconds**, and achieving
>80 TFLOPS with this work count requires **less than 43.296920 seconds**.

The validation harness asserts exact baseline video/audio latent equality
on every rank before VAE decoding. Both maximum absolute error and relative
L2 are **zero**. Fresh decode passes all automatic frame/dimension/FPS/audio/
finite/black/static checks and preserves MP4 SHA256
`5f1e58e555d613ed28e3793d02022b914fc62486190b9bfcf6300648a1974fb2`.
This is a fresh decode, not a reused prior video. Separate VAE decode takes
5.167203 seconds. The unchanged video's human temporal/audio review and
five-axis score remain pending; residual sequence parallelism still defaults
off.

Torch DiT peak is **15.197946 GiB/card**, NVML peak **17.351074 GiB/card**.
Median SM clocks are 1365/1455/1462/1455 MHz, powers are
278.348/267.206/273.597/270.673 W, and maximum temperatures are 56/61/57/64 C.
GPU utilization medians are 100%; throttle reason masks are 0/4. Power limits
already equal the supported 300-W maximum, and no settings were changed.

## Validation and rejected alternatives

- Affected video suite: **122 passed, 1 skipped, 1 deselected**. The unchanged
  separate TP4 residual-block regression retains its preceding four-rank
  result; this edit changes only the attention's V staging.
- Native short/tail/long controls pass bitwise. CUDA 12.8 memcheck, racecheck
  and synccheck each pass **12 cases** with zero errors/hazards, including
  storage offsets, independent query groups and graph replay.
- Eight-row 128-bit stores introduce one four-byte spill load/store at 128
  registers. Writing all K vectors before V staging does not eliminate it.
  Neither version is installed or credited with a speed result.
- The softmax-register-state and projection/communication-overlap experiments
  are documented in [FLASHINFER_OVERLAP.md](FLASHINFER_OVERLAP.md). In
  particular, the overlap prototype's small initial reduction error grows
  during sampling. It is rejected in favor of exact-output optimization.

## Reproduction, provenance and rollback

Use the environment, pinned checkpoint revisions and full commands in
[FLASHINFER_TO50.md](FLASHINFER_TO50.md) and
[FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md): Python 3.12.13, Torch
2.10.0+cu128, CUDA Toolkit 12.8.93, Transformers 5.15.1, explicit
`FLASHINFER_SM70` and `--residual-sequence-parallel`. The short workload uses
`--num-frames 39 --num-inference-steps 21`; rebuild the native FlashInfer
extension after updating source. Reproduce the prior implementation at
`5b4f0bba31f532057cf288787a6c8dafa5ec471f` in a separate worktree/build. Omit
the residual flag for its independent rollback.

The final attention binary SHA256 is
`dd7eff129614c25e18a2e2e7fad2e63991f1030f6d94716e6acde4b42c65f326`.
The unchanged W8 binary SHA256 is
`39988836d12f0e327d7f12b5e2a079202826d108b9fb89e9ec353d5cd92bd108`.
Artifact scripts install the new attention binary atomically after building.

Raw evidence is under the campaign's `feeding-overlap/`: `HANDOFF.json`,
source/binary snapshots, immutable candidate scripts, native short/long
results, suite/sanitizer logs, `quality-summary.json`, `nvml-summary.json`, and
`outputs/native-v4-20steps/FLASHINFER_SM70/` with MP4/WAV/latents, screenshots,
run metadata and NVML PNG/SVG curves. Rejected prototype media is in a
separate directory and is not a retained quality result. No weights or build
products are committed.

`attention-v4.ncu-rep`, raw metric CSV, details and summary retain the final
hardware counters. The profiler used a bounded GPU0-3 reservation in the
cooperating H3 queue after all seven native validation jobs completed. Its
initial worker-child exit message did not invalidate the capture: the target
kernel completed 24 passes, reported finite output and the expected binary
hash, and NCU exited successfully. Task-owned GPU reservations are released.

The follow-up in
[FLASHINFER_LOCAL_ROTATION.md](FLASHINFER_LOCAL_ROTATION.md) moves ConvRot
before the gather and preserves exact outputs. Its gather/projection overlap
control adds no model benefit and is not integrated.
