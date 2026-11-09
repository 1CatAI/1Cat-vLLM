# Native creative jobs

This change adds a native single-device Z-Image FP16 sampler and asynchronous
image jobs alongside the existing H3 video engine. Studio remains responsible
for authenticated asset storage and model lifecycle confirmation.

- Image CLI: `vllm image --model /local/model --checkpoint z-image-turbo --output-dir /local/output`.
- Image jobs: `POST /v1/images/jobs`, `GET /v1/images/jobs/{id}`, content at
  `/v1/images/jobs/{id}/content`. Synchronous `/v1/images/generations` remains
  compatible with URL and base64 results. Queued cancellation is supported;
  running jobs finish and retain outputs.
- H3 jobs now report `stage`, `stage_started_at`, `updated_at`, `stage_progress`
  and `denoise_progress`. Counts are sampler iterations, not output counts.
  Existing top-level video `progress` retains its legacy output-count meaning.
- `POST /v1/videos/{id}/cancel` never deletes an already completed output.
- Progress crosses the existing worker pipe on rank zero. Reporting introduces
  no tensor transfer, device synchronization, or additional collective.

## Recipes and precision

Z-Image's tokenizer, Qwen encoder, diffusion transformer, scheduler and VAE are
loaded exclusively from a local directory downloaded and verified by ModelScope.
No remote model code or automatic Hub fallback is enabled. The transformer and
text encoder use FP16 weights; latents, classifier-free guidance and VAE decoding
use FP32. Projection accumulators and SwiGLU gating retain FP32 range before
normalization. Base additionally keeps attention, AdaLN scaling and residuals in
FP32 because its learned modulation exceeds FP16 range. It retains FP16 weights
and fits one V100 32 GB, but is slower than Turbo. No outlier clamping is used.
Turbo uses eight nonzero sampler updates; base uses fifty with CFG 4.
Component implementations are pinned to Diffusers 0.40.0 and Transformers 5.15.1.
The reference sampler is Tongyi-MAI/Z-Image commit
26f23eda626ffadda020b04ff79488e1d72004cd (Apache-2.0).

Image editing is deliberately outside this capability. The model registry must
not advertise image editing, unsupported output sizes or unverified GPU quality.

## Acceptance — 2026-09-09

53 CPU media tests pass, covering protocol compatibility, real step counting,
queued/running cancellation, restart idempotency, FP16 outlier handling, corrupt
saved records and disk failure without a stalled queue. Pre-commit and CI pass.

Actual Studio frontend runs on V100 SXM2 32 GB / PyTorch 2.10 / CUDA 12.8 produced:

- Z-Image Turbo and Base: 1024 × 1024 PNG, seed 42, respectively 8 and 50 updates.
- H3 Turbo 4: text, first/last-frame and image-reference videos, each 1344 × 768,
  107 frames at 24 fps (4.458333 s), with synchronous 32 kHz stereo audio.
- The image run uses one explicitly selected V100; H3 uses four. Display GPU is
  excluded. ABI-matching SM70 extensions from `b6d91d61ff` are reused; this PR
  changes no CUDA/C++ source and makes no power-policy changes.

Z-Image Base now retains FP32 attention/modulation/residual range after measured
FP16 overflow in layer 25 Q/K/V and later AdaLN scaling. It fits one 32 GB V100,
but its 50-update original recipe takes roughly ten minutes in this setup.
Turbo is the default for interactive creation. Images were visually checked;
video playback/download and keyframe/reference consistency were checked in UI.

Progress parity with fixed inputs/recipes/seeds:

| Output | Enabled/disabled reporting |
| --- | --- |
| Z-Image Turbo PNG | Byte-identical (`a08ca43be383426e…`) |
| Z-Image Base PNG | Byte-identical (`1d9db6d2e332c65a…`) |
| Warm H3 keyframe MP4 | Byte-identical (`fb9dc41853af7778…`), including raw audio |

First-run H3 audio has a maximum 5.44e-7 sample difference from warm runs, also
present with reporting enabled in both runs; enabled/disabled warm outputs
match exactly. Video frames match in all three cases. These are correctness
checks, not a throughput benchmark or exhaustive validation of every allowed
output dimension/duration.

Weights and component code came from the checksum-verified ModelScope catalog:
`Tongyi-MAI/Z-Image-Turbo`, `Tongyi-MAI/Z-Image`, `MiniMax/MiniMax-H3`,
`Comfy-Org/MiniMax-H3`, and `lightx2v/Minimax-h3-Turbo`. The original model revision
and complete manifest hashes are retained in deployment acceptance records.
No Hub fallback is used by the native image path. Image editing remains disabled.
