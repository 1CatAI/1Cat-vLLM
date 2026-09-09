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

## Acceptance status

CPU protocol and cancellation tests are included. V100 generation quality,
fixed-seed progress on/off parity and Studio frontend acceptance are pending.
This document does not establish model or workflow availability. Record source
and model hashes, explicit GPU scope, dimensions, sampler recipe, output hashes,
quality review, elapsed time and UI run IDs before promoting this integration.
