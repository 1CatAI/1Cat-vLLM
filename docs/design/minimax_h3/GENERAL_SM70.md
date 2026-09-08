# General SM70 H3 acceleration campaign

Integration: `onecat/main`, base `4f19ef7a20db60bb0685e599bd3f4dd156202eed`.
Owned branch: `codex/v100-h3-general-sm70-20260908-165458`.
Raw evidence: `/data/minimax-h3/sm70-general-20260909/`.

## Accepted objective

Accelerate every supported H3 generation task across original floating-point
weights, W8A16, all eight LightX2V four/eight-step adapters, FlashGen, FastH3
Dense/VSA, TeaCache and Cache-DiT. Keep official task, adapter and sigma
contracts. Extract reusable SM70 DiT operators; a second complete model is
outside this campaign. Frontends continue to call the native vLLM API.

Use hardware, tensor layout/dtype, precision, adapter and collective capabilities
for dispatch. Preserve FP32 accumulation, residuals and wide-range scaling.
ConvRot applies only to matching rotated weights. LoRA receives the unrotated
activation and joins TP partial sums before reduction. Extend residual sequence
sharding to original weights, Ref2VA, adapters and TP2/TP4; TP1 remains ordinary
execution. AUTO may select only qualified routes.

Primary performance cases are the original 720p five-second four-step sample,
1344x768/243-frame official workload, and the 15-second boundary on TP4.
Count actual useful rank-local model work over the slowest rank's complete
denoise wall time, excluding padding, redundant work and skipped operations.
Sparse attention counts selected pairs and its actual compression projections;
cache savings are reported separately from arithmetic throughput. Record
actual denoiser calls, executed blocks and per-step/stage timing.

Acceptance requires one full warmup then three unprofiled measurements,
each rank's median above 80 TFLOP/s and denoise CV <= 5%, plus measured request
latency and memory. Smaller shapes and TP1/TP2 require compatibility checks.
The old 43.914752 TFLOP/s/card sample had 71.315842 s denoise and no warmup;
its equal-work 80 TFLOP/s budget is 39.147719 s. The old 39-frame/20-step
FA/FI diagnostics are not matching speed baselines.

Quality compares each variant against the same weights, initial noise and
official algorithm, including full sampling. Default numerical gates are
video/audio final-latent relative L2 <= 0.01, pre-encoding video PSNR >= 40 dB
and SSIM >= 0.99, audio spectral cosine >= 0.99 and RMS ratio in [0.99, 1.01].
Complete audiovisual and reference-consistency review remains a separate gate.
Unaccepted drift is not a new oracle; thresholds must not be relaxed to pass.

## Progress and required evidence

| Requirement | Implementation | Validation / evidence |
| --- | --- | --- |
| Common FP16 input/GEMM, dense column-major path | Shared operator and explicit dense layout | GPU operator checks pass; full original-weight generation pending |
| Prepared LoRA input and collective ordering | Implemented, including explicit original basis | GPU prepared/normal results bitwise equal; TP2/TP4 block comparisons pass |
| General residual sequence sharding | TP2/TP4, original/INT8, matching adapters; TP1 no-op | Both backends and wide residual/padding block checks pass; full Ref2VA pending |
| FA and FI kernel optimization | Existing narrower routes | Matched baselines pending |
| All dense task/weight/adapter combinations | Partial mainline support | Full matrix pending |
| FastH3 VSA on SM70 | Pending | Pending |
| TeaCache and Cache-DiT | Pending | Pending |
| AUTO and native variant APIs | Pending | Pending |
| Workflow-specific performance accounting | Fixed primary case only | Pending |
| Non-H3 DiT operator reuse | Shared GEMM/input preparation | Two non-H3 GEMM shapes pass; Attention reuse pending |
| >80 TFLOP/s/card, full quality, memory | Not achieved | No qualifying results |
| Draft PRs, matrix report and playable samples | Pending | Pending |

## Development record

- Baseline and active PRs inspected; no overlapping H3 PR is open. Both prior
  attention directions and NVENC are merged at the declared base.
- All eight V100s were idle at preflight. Every GPU launch must acquire the
  native GPU lease and recheck actual processes; idle observations do not
  reserve a device.
- The attachment and complete accepted plan were read. No previous execution
  turn existed: the preceding turn produced a plan, not implementation evidence.
- Next: rebuild owned baseline extensions, prove the current numerical route,
  then implement shared prepared linear execution and residual sharding.

### First implementation checkpoint

- Fresh SM70 extensions built from the integration base; immutable paths and
  SHA256 recorded in `baseline-binaries.json`. The baseline runtime is a clean
  `git archive` of that SHA. Generated extension aliases are confined to the
  artifact bootstrap, preserving package ABI names without copying stale H3
  binaries from another task.
- `prepared-linear-v2.log`: 28 GPU tests passed (shared GEMM, scaling, LoRA,
  activation, column-major plans). Prepared LoRA equals normal execution
  bitwise for original FP16 and W8A16, scales 0/0.75/-0.5, including explicit
  unrotated inputs beside rotated base operands.
- `candidate-tp2.log` and `candidate-tp4.log`: the complete distributed block
  case passes on all respective ranks. Cases include both attention backends,
  original and INT8 adapted blocks, consecutive blocks, padding and residuals
  exceeding FP16 range. This is not complete-model quality evidence.
- `core-regressions-v2.log`: 133 CPU adapter/config/API/conversion checks passed,
  one skipped and eight GPU cases deselected. Earlier export/import failures
  were missing video-extra dependencies in the new isolated environment.
  The environment now matches the retained native model-component versions;
  Torch remains 2.10.0+cu128. Both dynamic VAE classes import successfully.
- Failed setup paths retained: combining two distributed fixture lifetimes in
  one torchrun produced a Gloo rendezvous error after the first case passed;
  separate torchrun invocations resolve it. The first prepared run caught a
  removed compatibility export; the alias is restored and all 28 tests pass.
- The first full baseline stopped during VAE construction because the new
  environment lacked `diffusers`; no generated result or speed is claimed.
  Align the video dependencies with the retained native environment before retry.
- Active LoRA retains the normal unrotated gather in residual sharding. Its
  prepared row projections are enabled, and explicit dual-basis column input
  is supported, but reducing adapter input/gather overhead remains work.
- Original checkpoint loading rejects non-finite FP16 conversion, while keeping
  wide-range FP32 parameters intact. Tests cover BF16 overflow, infinity and NaN.
- Current full baseline: `baseline-720p-v2`, frozen integration Python source,
  fresh owned kernels, FA backend, no residual sharding, four-step v1.2 adapter,
  no FP16 weight cache, full warmup then one captured quality/timing request.
  This is a baseline acquisition, not the formal three-run acceptance.
