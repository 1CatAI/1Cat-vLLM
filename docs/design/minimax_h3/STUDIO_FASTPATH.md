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

All five packaged H3/SM70 extension targets compile locally and on the remote
V100 machine with Torch 2.10.0+cu128 and CUDA 12.8. Focused CPU tests pass;
remote validation additionally passes nine sparse-kernel GPU tests and three
Studio-capability checks. The real Studio frontend campaign below completes
both explicit modes. This validates executable integration and output media;
it does not qualify VSA numerical quality or change the native AUTO default.

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
FP32 diagnostic's quality pass. The remote campaign below is separate from that development-host evidence;
no new quality or AUTO qualification is introduced.

## Remote Studio campaign (2026-09-10)

Native executable source `5e50ef0df7`, Studio inference orchestration
`10557b383a`, frontend regression `7f76ae57f7`. Four V100 SXM2 32 GB cards,
TP4, 300 W limits, dynamic clocks; Python 3.12.13, Torch 2.10.0+cu128,
CUDA 12.8, 62 GiB host RAM, disk-backed DiT/text masters and shared VAEs.
No GPU power/clock writes were performed. Original FL2VA checkpoint and both
official Data-Free adapters were downloaded from ModelScope and SHA256-checked.

The UI submits the same paper-boat/yellow-duck prompt, seed 42, 1280x736,
120 requested frames at 24 FPS, four denoiser intervals. Each mode uses its
corresponding official adapter, one excluded cold warmup and three measurements.
Actual output is 124 aligned video frames with audio (about 5.17 seconds).

| Median of three warm UI submissions | Dense / FA | Fast / VSA | Ratio |
| --- | ---: | ---: | ---: |
| Complete denoise | 51.578823 s | 29.007658 s | 1.778x |
| Complete native generation | 122.276137 s | 97.195463 s | 1.258x |
| Studio submission through result save | 124.251130 s | 99.709044 s | 1.246x |

Measured native request seconds: Dense 122.797036 / 117.617351 / 122.276137;
VSA 102.989798 / 97.195463 / 95.907317. All four ranks reported the requested
FA/VSA backend and four DiT calls. Peer residual communication stayed on its
measured route without a native all-reduce fallback. Warm outputs were bitwise
repeatable within each mode. This is not Dense/VSA output equivalence or a
comparison with INT8/LightX2V; VSA still fails the inherited FP32 quality gate.

Native job IDs (measured, in order):

- Dense: `video_16f1c29d6ce7428886270e2368f59c98`,
  `video_2e48e29f0ebc4e5e8a4be678bec41d9b`,
  `video_7c36a88bdc7640b4864d62a9136b8c9e`.
- VSA: `video_febaf428e1bd4b47b05e7d0e54b043ce`,
  `video_1f94aaa8e9114f389d0cc833f2d9a9f0`,
  `video_09c73ec3b05343d498fa823dc3009e94`.

The retained `live-browser/results.json` report has SHA256
`88c62a1a2378099a892a42f1585a4ab253500395a30e65fb55a22e35fde69ee7`.
All eight outputs pass browser playback/download, full FFmpeg decode and
frame/dimension/audio checks. Screenshots and frame inspection show playable
outputs, not numerical quality qualification. Studio retains elapsed time,
restores the generation button, and references the same saved asset on canvas.
Cold adapter switching/loading is not included in these warm medians; the
remote VSA preparation took 944.26 seconds. The workbench exposes preparation
and generation separately rather than treating loading as denoising speed.
