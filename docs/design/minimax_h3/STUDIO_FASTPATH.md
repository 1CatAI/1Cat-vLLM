# Studio H3 execution profiles

This integration combines mainline `24220ca0eb` (native media progress,
UUID-preserving GPU selection and reclaimable disk-backed host weights) with
the H3/VSA integration at `c2ba9e2929` from PR #583 (including #581). It is an integration and
packaging scope; it does not reimplement the kernel campaign in #571/#578/#581.

## Deployment contract

Studio imports the native `video.fastpath.studio_capabilities()` descriptor
without allocating a CUDA context. The descriptor requires packaged projection,
query-tiled FlashAttention-V100 and exact row-reduction binaries. Old runtimes
retain their existing arguments. No request sampler, checkpoint, adapter scale,
resolution or frame count is changed by this profile.

For the supported dedicated TP4 V100 group, the dense execution profile selects:

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
positions. Mainline host-storage tests cover shared VAE, per-layer staging and mmap.

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

## Explicit experimental Fast VSA

The capability descriptor now separately advertises FastH3 and the packaged
`_sm70_sparse_attention_C` operator. `vsa_available` describes executable code,
not quality qualification. `experimental=true`, `quality_status=not_accepted`
and `tasks=[t2va]` are explicit; an old binary cannot enable the Studio switch.

Studio's user-facing Fast switch belongs to the FastH3 Preview v1 Data-Free
workflow. Off selects its Dense adapter and FA query tile 128; on selects its
VSA Data-Free adapter, FASTVIDEO_VSA, top-k 64 and query tile 64. The shared
column/host/residual optimizations remain active. This switch is separate from
`execution_mode`, which controls the existing lossless dense execution profile.
The FastH3 model requires original floating FL2VA weights and cannot consume
INT8 ConvRot or LightX2V adapters. Runtime INT8 fusion is not implemented by
PR #583. Standard INT8 Studio recipes retain their actual adapters and schedules.

The original artifact stores BF16 tensors; native SM70 execution uses FP16.
The explicit adapter is `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`,
`vsa-datafree/adapter_model.safetensors`, SHA256
`42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`.
Both the original checkpoint and adapters are available through ModelScope.
FastH3 uses four API intervals, unlike LightX2V's five sigma positions.

The existing #583 matched campaign gives median denoise 52.873835 / 30.990756
seconds (Dense / VSA), a 1.706x ratio and 41.39% less denoise time. Complete
request medians are 87.426512 / 68.157817 seconds, 1.283x and 22.04% less time.
Those are the retained development-host measurements, not a new remote Studio
benchmark. See VSA_QUALITY_SPEED.md for the failed independent FP32 quality
comparison and exact workload/source/binary evidence. Do not claim half the
complete generation time or combine the fast kernel's timing with the slower
FP32 diagnostic's quality pass. New remote frontend/media acceptance remains
pending; no new quality or AUTO qualification is introduced.
