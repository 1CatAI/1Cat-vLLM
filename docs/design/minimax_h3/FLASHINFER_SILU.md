# FlashInfer integration with shared fused MLP preparation

The unchanged **1344x768, 39-frame, 24-FPS, seed-42, TP4 GPU0–3** development
workload completes **20 denoise updates in 63.565580 seconds**, or
**3.178279 seconds/update and 54.491025 useful TFLOPS/card**. This integrates
dependency `9764b6c202`'s shared MLP fusion with the retained independent
FlashInfer warp-transpose kernel. The 64.920336-second attention-only result,
native operator controls and hardware counters remain in
[FLASHINFER_VTRANSPOSE.md](FLASHINFER_VTRANSPOSE.md).

The fusion saves another 1.354756 seconds (2.09%). Across probability-layout,
warp-transpose and shared MLP changes, denoise decreases from 66.863312 to
63.565580 seconds, **4.93% less time**. Video/audio latents and fresh MP4 remain
bitwise equal. These are single unprofiled development measurements; the formal
three-run, human quality, <50-second and >80-TFLOPS/card gates remain open.

## Integration and fixed contract

The shared fusion evaluates SiLU/product in FP32 and prepares FP16 rows with
the existing power-of-two scale, avoiding a large FP32 temporary write/read.
Its H3-only row-parallel subclass restores scale in FP32 before reduction and
retains module/FLOP hooks. Our residual-sharding path disables the row module's
ordinary reduction and performs its existing FP32 reduce-scatter afterward.
The gate/up buffer is released before projection. CPU, unquantized and
FP32-input MLPs retain their previous path.

The native FlashInfer and TurboMind binaries are unchanged from the preceding
run. Both SHA256 values, merged source hashes and task-local Triton cache
hashes are retained in the run manifest. No parallel FlashAttention operator
is selected; all tests/generation explicitly use `FLASHINFER_SM70`.

The run retains optional FP32 residual sharding, the exact 200-weight cache
within 10 GiB, verified cached TP4 text conditioning, Comfy INT8 ConvRot,
21 sigma positions and video/audio shifts 12/3. One single-update warmup
precedes the complete short schedule. This is not end-to-end timing.
Useful work remains 3,463,753,579,661,312 FLOPs/rank; preparation and padding
do not increase the numerator. Residual sharding still defaults off.

## Validation and resources

- Merged affected suite: **122 passed, 1 skipped, 1 deselected**.
- The separate TP4 residual-block regression passes on all four ranks at valid
  lengths 32/33/131, with changed padding and FP32 residuals above FP16 range.
- Video/audio latent max error and relative L2 are zero against the retained
  64.920336-second build. Fresh decode passes every automatic media check and
  preserves MP4 SHA256
  `5f1e58e555d613ed28e3793d02022b914fc62486190b9bfcf6300648a1974fb2`.
- The native attention's preceding three 12-case sanitizer results remain
  applicable: neither CUDA binary changed in this dependency integration.
- Source/document pre-commit checks pass. An initial additional TP4 launcher
  used a selector matching no tests; its exit-5 result is not counted. The
  corrected selector passes on all ranks without repeating the 122-test suite.

Torch DiT peak allocation falls to **15.197946 GiB/card**; NVML peak is
**17.351074 GiB/card**. Fresh VAE decode takes **5.308711 seconds** separately.
Median SM clocks are 1372/1477/1485/1470 MHz and median powers are
278.586/268.245/271.750/273.940 W. Maximum temperatures are 55/59/56/63 C;
utilization medians remain 100%, with throttle masks 0/4. No power/clock/ECC
settings change. None of these busy percentages establish useful TFLOPS.

The unchanged attention binary's separate NCU capture still shows 37.57%
Tensor pipe activity and significant MIO pressure. The current remaining
gap is **13.565580 seconds to 50 seconds**, and >80 useful TFLOPS requires
less than 43.296920 seconds. Operand traffic, register/occupancy limits and TP
communication overlap remain the next optimization work. The earlier NSYS
percentages are not presented as a fresh trace of the fused complete model.

## Reproduction and rollback

Use the unchanged environment, pinned revisions, cache list and commands in
[FLASHINFER_TO50.md](FLASHINFER_TO50.md) and
[FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md), with explicit
`FLASHINFER_SM70` and `--residual-sequence-parallel`. Reproduce the preceding
unfused implementation using `572dc56ef9` in a separate worktree. Omit the
residual flag for its independent rollback. Human quality review of residual
sharding relative to replicated residuals remains pending.

Raw final evidence is in the campaign's `feeding-silu/`: manifest, source and
Triton cache hashes, exact commands, suite and corrected TP4 logs, full schedule,
latents, fresh MP4/WAV, quality summary and NVML PNG/SVG curves. `feeding-vtranspose/`
retains the unchanged native binary's operator/sanitizer/NCU evidence.
