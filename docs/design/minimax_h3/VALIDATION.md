# H3 reproducible validation

This draft has not passed the full-video quality or per-GPU 80 TFLOPS gates.
Component numerics and short cached-text generation are working. FlashAttention-V100
is the user-selected mainline and the default denoiser. Keep profiler
diagnostics separate from the three formal timing runs.

The current short-development target is **under 50 seconds for 20 actual DiT
updates**, retaining 1344x768, 39 frames/24 FPS, seed42, INT8 ConvRot FL2VA,
TP4 GPU0-3 and output quality, without a substantial memory increase.

## Latest: local ConvRot, 58.19 seconds

With `--residual-sequence-parallel`, normalize and rotate each rank's local
FP16 rows before gathering, then project without repeating the rotation. This
preserves the gathered bits, original ConvRot256 operator, INT8 scales, GEMMs
and useful-FLOP hooks. It adds no persistent FP16 weights or CUDA workspace.
The installed H3 CUDA binaries are unchanged.

The complete 20-update workload now takes **58.190385 s**, or **2.909519 s/update
and 59.524500 useful model TFLOPS/card**, versus the earlier 58.794562 s sharded
route (1.03% less time). Each rank's useful FLOPs remain
3,463,753,579,661,312. Peak denoise Torch allocation is unchanged at
**6.195001125 GiB/card**; NVML peak device-used memory is 8.046386719 GiB/card.
This is one unprofiled development measurement after a one-call warmup with
prompt-verified cached text, not a formal median or end-to-end measurement.
Three paired two-update probes give medians 5.878070 ->5.813375 s (1.10%).

Both complete video/audio latents equal the earlier 58.79-second run bitwise,
are finite and match across ranks. Its previously checked decode is reused
after latent and MP4-hash checks: **no new VAE decode was run**. Automatic
validity is preserved; the previous human audiovisual review remains pending.
The flag still defaults to false; omit it for the 62.804019-second default.

Retained source: `ca5b6a855b` on the native FA residual branch. This reuses a
read-only working-tree snapshot of the independent FI task, based on
`53780abef0e9ef190aa55a1f61d8fd5fbd390aa1`; the source was uncommitted at capture.
Patch SHA256 is
`ce41592338adeef0388c7b5b826351d378c468bfa58e7df3def662cf7056dc4b`.
The new regression uses native FlashAttention; there is no FI delegation.
Parent `2a560b0a7fa0` rolls back only local rotation.

Validation: 62 targeted tests pass with one GPU case deselected. One separate
four-rank GPU test passes on every rank, checking cached/uncached INT8, two
valid/padded lengths, consecutive blocks, rank-dependent weights, FP32 values
above 65504, exact outputs and unchanged FLOP hooks. Pre-commit passes. Exact
commands are in `flashattention-register50/jobs/13-local-rotation-tests.done.json`
and `14-local-rotation-tp4.done.json`. Distributed GPU modules run in separate
pytest invocations because the repository fixture tears down distributed state
between tests. Prior sanitizer results cover unchanged operators only.

Eight additional attention candidates match the native output across 12 tested
lengths but show no stable paired speedup. Query-window projection overlap and
a TP4 block pipeline also lose. The latter's raw counter includes 29 padding
rows and is explicitly invalid for acceptance accounting. None is installed.
See CONTROL.md and `flashattention-register50/failed-paths.json`; do not repeat
unchanged candidates. GEMM/attention remain the major optimization target.

Latest report, exact reproduction wrapper, raw timing, source hashes and NVML
curves: `flashattention-register50/`. Output:
`outputs/quality39-int8-flashattn-localrot-20steps/FLASH_ATTN_V100/`.
MP4 SHA256 remains
`f00ba75587f58e2a63a105647cb634eebd60b154ab7b3551f385ca2df96e5243`.
**Under 50 seconds, standalone attention 60 TFLOPS, formal per-card 80 TFLOPS
and human quality acceptance remain incomplete.**

## Previous residual-sharding and wide-attention evidence

The initial explicit `--residual-sequence-parallel` option completed this workload
in **58.794562 s**, or **2.939728 s/update and 58.912822 useful model
TFLOPS/card**. Peak denoise Torch allocation is **6.195001125 GiB/card** and
NVML peak device-used memory is 8.171386719 GiB/card, with zero persistent
FP16 cache and zero Lt workspace. This is a 6.38% time reduction from the
default 62.804019 s route. The option defaults to false and currently requires
TP4 FL2VA INT8 without adapters, using either native SM70 attention backend.
It preserves FP32 residuals and gathers FP16 values only at the original
normalization boundary. Omit the flag for rollback.

Measured implementation commit: `59df80f5aae63e36a428a095d6f43570cfbc6547`.
Integration with current main `e7fa44deb2` preserves byte-identical DiT,
activation, quantization, attention, residency and weight-cache source, and
an AST-identical `diffuse` method. The separate CPU compatibility check has
78 passes, 30 GPU skips and one deselection. It does not replace the earlier
actual four-rank residual test, which passed on every rank for both native
backends. The earlier targeted numerics/service/configuration run has 70
passes; these overlapping suites are not summed as distinct tests.

A fresh decode passes all automatic media checks. FP32 collective reduction
order changes yield video/audio latent relative RMS deltas of 3.87465% and
1.20270% from the default. Decoded-video SSIM is 0.983989; PSNR is38.238221 dB.
These are auxiliary comparisons, not human quality thresholds. Eight sampled
frames show stable count, color and background; full audiovisual review is
pending. New MP4 SHA256:
`f00ba75587f58e2a63a105647cb634eebd60b154ab7b3551f385ca2df96e5243`.
Evidence is in `flashattention-pipeline50/` and fresh media in
`outputs/quality39-int8-flashattn-residual-20steps/FLASH_ATTN_V100/`.

Separate two-update traces attribute the residual-sharding gain to lower
normalization/gating work (0.440786 ->0.237076 s of other GPU kernels) and
collective data movement (1.076831 ->0.845654 s). GEMM and attention still
account for about78% of the new trace. Copies are only0.012507 s. Eight paired
attention candidates failed to establish a gain and remain uninstalled.
A zero-cache projection-overlap prototype improves a two-update paired median
by2.20%, but is not integrated or claimed as a complete20/video result.
See CONTROL.md for NCU/SASS evidence and negative experiments.

Without residual sharding, the native wide-QK plus fused MLP route measures
**62.804019 s**, or
**3.140201 s/update and 55.151783 useful model TFLOPS/card**, compared with
69.062335 s previously (9.06% reduction). Cache and Lt workspace are zero.
Peak Torch denoise allocation is **6.566735744 GiB/card**, down from
6.647851467; NVML peak device-used memory is 8.419433594 GiB/card.

The main gain reuses Q operands across a wider K tile and fixes softmax's
incorrect inferred warp count. A second change fuses FP32 SiLU/product with
power-of-two FP16 input preparation and releases the gate buffer before GEMM.
The native B1/N12323/H14/D128 attention paired control is 25.572351 ->20.110336
ms (42.565744 ->54.126701 useful TFLOPS). The **60 TFLOPS attention** milestone
requires <=18.141769 ms. A larger-Q prototype improves only 1.48% against the
new native path and is not enabled. Preserve the arithmetic/data-movement
focus identified in the earlier NCU/SASS evidence in CONTROL.md.

Wide attention changes online reduction and FP16 probability rounding; its
64.284648-second native run was freshly decoded. Video/audio latent relative
RMS deltas against the previous K64 implementation are 10.5741% /1.8484%.
Automatic media checks pass, and eight sampled frames show no obvious count,
color or geometry regression. Human audiovisual scoring is still pending.
The subsequent MLP fusion (62.852804 s) and buffer release (62.804019 s) are
bitwise equal to the wide-attention latents; their decoder reuse is explicit.
The output MP4 SHA256 is
`97b9196c49a8e1bf61cb8368f4db7d7013e99bc107650945dbe92b4b59b0184a`.

There are **103 passing targeted tests** for attention, fused activation,
INT8 layout/Lt and model numerics. After repository formatting and rebuilding,
the resulting SASS is byte-for-byte identical and the 41 affected attention
and activation tests pass again; this is a repeated subset, not 144 distinct
tests. Isolated CUDA12.8 memcheck, racecheck and synccheck each pass 25
attention cases with zero errors/hazards. Ordinary pytest covers graph
replay; isolated sanitizers do not capture graphs. N73483 uses sampled FP32
reference rows, avoiding a full quadratic allocation. Sixteen lengths retain
bitwise K64 primitive rollback. Two earlier same-module-name control probes
are explicitly invalidated because Python reused the candidate extension;
only `native-verified-results.json` is valid native paired-control evidence.

These are separate single unprofiled development runs after a one-call warmup,
with prompt-verified cached text. Do not pool different builds into a formal
median or present this as end-to-end / three-run 243-frame acceptance.
Evidence and NVML plots are in `flashattention-qk50/`; media and raw rank/NVML
reports are in
`outputs/quality39-int8-flashattn-native-wide-silu-release-20steps/FLASH_ATTN_V100/`.
**Attention 60 TFLOPS, under-50-second denoise, formal >80 TFLOPS/card and human
quality gates remain incomplete.**

## Short development checks

Use the native entrypoint with the same spatial canvas and a short temporal
extent while developing kernels or debugging components:

```bash
vllm video generate --model /path/to/MiniMax-H3 \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --tensor-parallel-size 4 --attention-backend FLASH_ATTN_V100 \
  --int8-weight-layout column --fp16-weight-cache-gib 0 \
  --num-frames 39 --num-inference-steps 21 --output-dir /path/to/development
```

This produces 1.625 seconds and performs 20 DiT forwards. It is the short
quality/development workload; the sigma-point convention needs 21 positions
for 20 actual updates. One- or two-forward runs remain numerical/execution
diagnostics and must not be used to judge normal model image quality. The streaming VAE requires at least 22 frames.
Keep the requested 50 sigma positions when evaluating final quality. Do not run
the full acceptance runner for routine development.

## Fixed workload

The acceptance helper uses the Chinese paper-boat/duck prompt in
`H3Request`, 1344x768, 243 frames, 24 FPS, seed 42 and 50 sigma positions.
The checkpoint schedule performs 49 DiT calls. All four ranks must report
their actual completed call counts.

After installing 1Cat with `.[video]` and building the H3 extensions:

```bash
python tools/minimax_h3/fixtures.py --model-root /path/to/MiniMax-H3 \
  --output-dir /path/to/reference-fixtures
python tools/minimax_h3/benchmark.py --model /path/to/MiniMax-H3 \
  --transformer-path /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --attention-backend FLASH_ATTN_V100 --output-dir /path/to/acceptance
```

The runner performs one warmup and three unprofiled requests in one engine.
The warmup's automatic media checks must pass before formal timing starts.
Add `--residual-sequence-parallel` to evaluate the experimental route;
this records the option in every run config and rejects mixed configurations.
The full 243-frame benchmark has not been run for this option.
Use `FLASHINFER_SM70` for an explicit comparison or rollback. A nonzero FP16
weight-cache budget requires an explicit measured list of `--fp16-cache-layer`
arguments. The default budget is zero.

## Reports and interpretation

```bash
uv pip install --python .venv/bin/python matplotlib
python tools/minimax_h3/report.py /path/to/acceptance/run-1
```

Each request retains `video.mp4`, the original waveform, sampled screenshots,
`run.json`, and `nvml.jsonl`. The report helper exports rank throughput and phase
times as CSV, and six NVML curves as PNG/SVG. Phase times include overlapping
aggregate and component fields; do not sum all fields indiscriminately.

Useful FLOPs count the actual local model matrices and effective attention
lengths. Padding, Hadamard rotations, weight dequantization and extra output
rows do not inflate the numerator. Each rank uses the slowest rank's full
denoise wall time. Formal evaluation rejects mismatched requests/configurations,
profiled timing, duplicate ranks and incomplete schedules.

The performance gate requires all four per-rank medians to be strictly above
80 TFLOPS and the three denoise times' population coefficient of variation to
be at most 5%. Memory is a separate 30 GiB gate. NVML utilization is never
interpreted as Tensor Core activity. Retain Nsight Compute counter evidence and
Nsight Systems communication/stall traces separately.

The evaluator leaves overall `accepted` false. Final review must score prompt
following, subject consistency, temporal continuity, detail and audio at least
4/5 each and reject flicker, deformation or audio/video faults.

## Evidence completed so far

- Both fixed INT8 partition files are downloaded and SHA256 manifests saved.
  The original BF16 DiT partitions and shared components are downloaded, and
  their 98-file checksum manifest is complete.
- Real TP4 text-encoder outputs are finite and bitwise identical between ranks.
- Real 50-block INT8 DiT single-forward checks pass after preserving large
  condition projections, residuals, gated products and row-projection outputs
  in FP32. The matrix inputs/weights remain FP16 for Tensor Core execution.
- The current numerical/service suite passes 47 tests, including short clips,
  the optimized FlashInfer kernel, prefetch tails and unaligned storage views.
  Seven acceptance-contract tests pass, including excluded timing.
- A 256x256, 107-frame, one-DiT-call route exported finite video and audio and
  passed the automatic media checks. It is not a quality/performance baseline.
- The earlier full primary video was interrupted before completion and is
  retained as partial diagnostics only. Development now uses 39-frame clips.
  Cached-text INT8 TP4 denoise (two forwards after warmup) measured 11.29494 s
  with Flash-V100 and 10.55566 s with the FlashInfer candidate: 30.6664 versus
  32.8142 useful TFLOPS on each rank. Both produced finite, rank-identical
  latents. These are development results, not full-schedule acceptance.
- Original BF16 FL2VA text-to-video completed 20 updates at 39 frames through
  Flash-V100 with FP16/FP32 computation. All automatic checks pass; denoise took
  110.569167 s. Other original-checkpoint backend/partition combinations, mixed
  references, extra seeds, full quality and formal three-run timing remain pending.
- Standard CMake configuration, build and component installation of both H3
  extensions pass. Complete release-wheel validation remains pending.

The earlier two-forward short export contains 39 decoded frames and valid
audio, but its image has clear ghosting and grid artifacts. It is not quality
accepted. Other GPU workers were present during that export run, so its JSON
marks timing invalid. Both the evaluator and report helper reject explicitly
excluded timing. The earlier isolated candidate comparison is retained separately.

A subsequent matched 20-forward comparison completed on exclusively leased
GPUs 0–3. Both backends pass automatic media checks, and inspected frames no
longer show the severe two-forward artifacts. Flash-V100 took 113.459387 s
(5.672969 s/call); FlashInfer took 105.642574 s (5.282129 s/call). Both used the
same INT8 checkpoint, prompt, seed, 39 frames, cached text and shift rules.
Full primary acceptance and human audiovisual review remain pending.

The subsequent FlashInfer V-layout/KV-prefetch optimization completes the same
20-update short request in 90.880784 s (4.544039 s/update), with 38.113157 useful
TFLOPS on each rank. Video/audio latents and the exported MP4 are bitwise
identical to the previous FlashInfer result; automatic media checks pass.
CUDA 12.8 memcheck/synccheck each pass the six new tail/unaligned cases with
zero errors. The standard CMake FlashInfer component builds successfully.
The 6.647851 GiB Torch peak is measured during DiT only and must not be treated
as the full pipeline's memory gate. This single short run does not satisfy the
primary three-run timing contract or the >80 TFLOPS threshold.

The profiler-only run uses the first two updates of the unchanged 20-update
schedule. Its rank-0 trace attributes about 52% of the synchronized
span to attention, 25% to GEMM and 10% to communication before this optimization.
Explicit wall coverage and kernel service are retained separately. Profiling
results never replace unprofiled timing.

The selected cooperative-transpose kernel completes the same 20-update denoise
in 87.824099 s (4.391205 s/update), with 39.439671 useful TFLOPS on each rank.
Compared with the original 105.642574 s FlashInfer baseline, throughput improves
by 20.29%. Both final video and audio latent tensors are bitwise identical to
the previously decoded prefetch result. Only after checking both tensors and
the MP4 hash, this run reuses that validated decoder output. Its metadata marks
`short_clip_denoise_quality_with_reused_decode`; it does not provide a new VAE
or end-to-end timing measurement. Text embeddings are also cached and verified.

The final binary passes all 47 video tests and the ordinary CMake component
build. CUDA 12.8 memcheck, racecheck and synccheck each pass all six new
tail/unaligned cases without errors or hazards. Matched-shape Nsight Compute
reports 25.091% tensor-pipe activity, up from 16.976% before prefetch. This
counter is diagnostic, not effective model throughput. The 39-frame result
remains below 80 TFLOPS/card; primary-load quality, three formal timing runs
and the full-pipeline memory gate remain incomplete.

Fused FP32 input preparation and warp-shuffle ConvRot subsequently reduce the
same 20-update denoise to 86.200372 s (4.310019 s/update), or 40.182583 useful
TFLOPS on every rank. Final video/audio latents remain bitwise identical;
the run retains explicit cached-text/decoder-reuse labels. The video suite
passes 58 tests, with a subsequently added nonfinite-input test passing
separately. All eleven new rotation/scaling cases pass memcheck, racecheck
and synccheck, and the W8A16 CMake component builds.

A recovered ComfyUI V100 attention source was also audited. On aligned D128
controls it is about 1.9 times slower than current FlashInfer, and its raw
unaligned path has query-tail synchronization and numerical failures. Those
controls do not replace H3 quality or performance acceptance. The mainline
and >80 TFLOPS/card target remain unchanged; see CONTROL.md for evidence.
