# Studio dense H3 fast-path integration

This integration combines mainline `24220ca0eb` (native media progress,
UUID-preserving GPU selection and reclaimable disk-backed host weights) with
shared dense-kernel commit `9f220dc2fc` from PR #581. It is an integration and
packaging scope; it does not reimplement the kernel campaign in #571/#578/#581.

## Deployment contract

Studio imports the native `video.fastpath.studio_capabilities()` descriptor
without allocating a CUDA context. The descriptor requires packaged projection,
query-tiled FlashAttention-V100 and exact row-reduction binaries. Old runtimes
retain their existing arguments. No request sampler, checkpoint, adapter scale,
resolution or frame count is changed by this profile.

For the supported dedicated TP4 V100 group, Studio's fast mode selects:

- FlashAttention-V100 query tile 128 and column-major floating projections;
- FP32 residual sharding, calibrated peer rows with a 4 GiB setup/buffer budget;
- pageable host masters and shared immutable VAE storage;
- the existing automatic disk-backed DiT/text storage on hosts below 128 GiB.

Native CLI defaults remain compatible. Studio services may explicitly select
`execution_mode=standard` for a control or rollback. Other attention backends,
GPU families, TP sizes and missing/old kernel packages keep the ordinary path.
Peer setup additionally agrees across ranks on actual available memory and
peer accessibility. Unsupported or oversized requests use the original FP32
all-reduce and record the reason. There is no sparse attention, approximate
cross-step caching, changed quantization or shortened generation.

The exact-reduction extension is included in source builds and precompiled
wheel reuse. First generation must not require a separate CUDA JIT build.
The UI's loading and GPU-completion-based progress callbacks remain intact;
merged work accounting records actual denoiser calls separately from sigma
positions. Shared VAE, per-layer staging and mmap host tests remain separate.

## Observed production baseline

Four V100 SXM2 32 GB cards, original Studio native source
`48b375b84d`, W8A16 ConvRot FL2VA and
`minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`, seed 42,
1344x768, 147 requested frames, 24 FPS, five sigma points / four DiT calls:

| Stage | Recorded seconds |
| --- | ---: |
| Input encoding | 22.146 |
| DiT staging/cache preparation | 9.201 |
| Complete denoise | 109.835 |
| VAE decode | 17.995 |
| MP4 packaging | 25.207 |
| Output validation | 3.622 |
| Native complete request | 188.941 |

These are the recorded request values, not a new matched benchmark. Stage
values are slowest-rank durations and overlap with broader parent spans; do
not add both `denoise` and `denoise_including_staging`. Historical 39-frame
20-step cached-text denoise runs are not comparable to 147/360-frame complete
requests. Keep model/adapter hashes, power, frames and warmup state fixed in
any new comparison; report both denoise and complete request.
The machine currently reports a 300 W limit; the old native telemetry files
were removed after Studio imported the outputs, so that observation does not
establish the historical request's power or clock conditions.

## Acceptance status

Four packaged SM70 extension targets compile with Torch 2.10.0+cu128 and
CUDA 12.8. Offline integration/ABI checks pass. Remote same-contract GPU speed,
complete output preservation and frontend generation remain pending: the first
control task was cancelled as the machine switched to an active chat model.
Do not treat inherited development-host performance as a measured Studio gain
or promote this integration until the allocated-GPU control is complete.
