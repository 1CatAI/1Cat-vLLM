# Native request-scoped H3 caches

Caching is disabled by default. The native Python, CLI and HTTP interfaces can
explicitly select the official FL2VA TeaCache policy or Cache-DiT. This is
algorithm integration, not qualification for AUTO or >80 useful TFLOP/s/card.
Full independent-model quality, complete workflow performance and human review
remain pending.

## Configuration and request semantics

`H3Config(cache_backend="tea_cache", cache_config={"rel_l1_thresh": 0.17})`
uses the official H3 FL2VA polynomial. Ref2VA-only deployments reject this
uncalibrated policy. `H3Config(cache_backend="cache_dit")` uses the official
Fn=1, Bn=0, warmup=4, residual threshold=0.24, maximum continuous reuse=3
profile. Both configurations are also available through `--cache-backend`
and JSON `--cache-config`. The exposed Cache-DiT options are Fn/Bn block
counts, warmup steps, total cached-step limit, residual threshold and the
continuous cached-step limit. Optional `enable_taylorseer` and
`taylorseer_order` select the pinned package's official forecasting calibrator.
`scm_steps_mask_policy` accepts `slow`, `medium`, `fast` or `ultra`;
`scm_steps_policy` accepts `dynamic` or `static`. TaylorSeer is off and SCM is
unset by default.

Omitted request quality follows the selected deployment policy. Explicit
`quality="lossless"` disables request caches. `quality="high"` selects the
official conservative Cache-DiT profile: Fn=1, Bn=0, warmup=4, threshold=0.04,
continuous reuse=1, without TaylorSeer or SCM. A TeaCache deployment rejects
`high` because the two backends are mutually exclusive.

SCM masks come from the pinned package and use actual sampling intervals,
including four/eight calls for LightX2V's five/nine sigma points. As in Omni,
unsupported short schedules (1/2/3/5/7 calls) use ordinary cache refresh without
a predefined mask. Every request installs a fresh calibrator and refreshes
the mask. Repeated force-refresh hints can intentionally prevent cache reuse:
refreshing every third call resets the eight-call medium mask before it reaches
its first reuse slot. This must not be reported as an implementation failure
or credited as skipped work.

The existing video API accepts `extra_params.force_refresh_step_hint` and
`force_refresh_step_policy` (`once` or `repeat`) for an active Cache-DiT
request. CLI equivalents use dashed option names. Hints are positive and
1-based; the worker validates their upper bound against the actual sigma
intervals. This preserves LightX2V's 5/9 sigma points and FlashGen/FastH3's
four-interval conventions. Invalid inactive-cache hints are rejected before
worker dispatch.

Each denoise request creates fresh cache state and tears it down in a `finally`
block, including failed sampling. A model rejects overlapping cache contexts.
No residual can survive into a different request, partition, weight set,
adapter scale or schedule. The native service already serializes requests.
Cached state remains local to each TP rank. TeaCache synchronizes the compute
vote across TP; Cache-DiT's official implementation reduces its residual
statistics across the participating ranks.

## Implementation and accounting

TeaCache follows the extractor, polynomial and residual update in Omni commit
`7be014bce6374f06c95b703763bdbac4c6198f31`. It caches the main block stack's
FP32 residual; token refinement and final output projections still run. The
modulation probe uses the matching local residual rows and global modality
indices, including padding and TP2/TP4 sharding.

Cache-DiT uses the exact `cache-dit==1.5.0` distribution pinned by Omni, with
its Pattern_3 block adapter and refresh implementation. Wrapper classes are
isolated per request to avoid the package's process-wide cached marker.
The locally installed distribution reports internal `__version__` as
`1.3.13.dev9`; the distribution version and source provenance are authoritative.
Diffusers 0.40.0 also requires a Hugging Face Hub build providing
`get_cached_repo_tree`; this worktree uses Hub 1.30.0.

Actual matrix and attention calls provide effective FLOPs. Each step records
executed block identities, skipped blocks, cache hits and decision-probe FLOPs.
The extra TeaCache modulation projection is excluded from effective model
FLOPs. Sparse selection savings are counted only for blocks that actually
execute. Skipped cache blocks never become credited compute. The acceptance
evaluator checks block identities, policy-compatible skip patterns and
per-step/per-layer totals, while retaining the full warmup and three unprofiled
request requirements.

## Validation record

Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8.93, V100 SXM2 32GB.
Evidence: `/data/minimax-h3/sm70-general-20260909/`.

- `cache-cpu-v1.log`: 18 tests pass, including the official polynomial and
  residual update, refresh arguments, repeated real Cache-DiT contexts,
  once/repeat hints, teardown and overlapping-request rejection.
- `cache-integration-cpu-v1.log`: 120 affected CPU checks pass, two GPU cases
  deselected.
- `cache-accounting-cpu-v2.log`: 105 checks pass, including rejection of
  invented hits, block identities, totals and decision overhead.
- `cache-gpu-tp1-v2.log`, `cache-gpu-tp2.log` and `cache-gpu-tp4.log`: real small H3 forward checks
  pass, with original FP16 weights, FP32 residuals and non-aligned valid rows.
  Per-step executed block counts are 6/6/6/6 for forced compute,
  6/2/2/6 for TeaCache reuse and 6/3/3/6 for Cache-DiT. Repeated requests
  reproduce the same outputs and patterns. These are operator/integration
  checks, not full-video quality or performance results.

The first TP1 test setup omitted the small model's final AdaLN dimension;
`cache-gpu-tp1.log` records that constructor rejection. The corrected test
sets both main and final AdaLN dimensions. No production algorithm was changed
in response to that fixture error.

## Full native request lifecycle

Source `d09e52dd302e2bed510cd036f1d338e5b8325c77` completed three sequential
requests per backend in one engine: cached, explicit lossless, cached. Both use
original FP16 execution weights, FP32 residuals, column-major weights, TP4 on
GPUs 0–3, pageable host masters, 256x448 output and 49 sampling intervals.
The requested four seconds resolves to 107 internal frames. These runs test
full-model lifecycle compatibility; they are not the primary resolution or
the required warmup-plus-three performance measurements.

| Backend | Cache hits per request, on every rank | Denoise seconds | Peak allocation bytes/card |
| --- | --- | --- | --- |
| TeaCache, threshold 0.17 | 5 / 0 / 5 | 45.601157 / 45.854488 / 41.358189 | 17,144,250,368 |
| Cache-DiT, official defaults | 34 / 0 / 34 | 16.450964 / 45.956182 / 15.157820 | 17,143,863,296 |

All six requests pass basic media validity and strict executed-work checks.
For each backend, the first and third requests produce bitwise identical
video/audio latents and RGB frames (PSNR infinity, SSIM 1). Decoded PCM has
small repeat-run differences: relative L2 1.684731e-6 for TeaCache and
2.066048e-6 for Cache-DiT; spectral cosine exceeds 0.999999999998 and RMS
ratio differs from 1 by less than 7e-9. All declared numerical repeatability
gates pass. This comparison checks fresh-request reproducibility, not whether
approximate cached outputs match lossless sampling or the official model.

Evidence: `teacache-native-summary.json`, `cachedit-native-summary.json`,
`teacache-repeat-quality.json` and `cachedit-repeat-quality.json` under the
evidence root. Raw outputs live under
`/home/ymzx/h3-sm70-artifacts-20260909/runs/` in the corresponding
`teacache-native-lifecycle` and `cachedit-native-lifecycle` directories.
Independent official quality, human review and >80 useful TFLOP/s/card
acceptance remain incomplete; no automatic policy is qualified.

## TaylorSeer and SCM validation

Source `f5e4340d4a7d35432fbcd85380c4c11d4514865f` adds the official optional
calibrator and predefined step masks without changing their defaults.
`cache-scm-cpu-v2.log` has 42 passing policy/lifecycle checks. The first
expanded run additionally passes 87 affected acceptance/service/API checks;
its three failures were tests incorrectly demanding cache hits when a repeated
refresh resets SCM before any reuse slot. The corrected tests assert the
official all-compute pattern, with no production algorithm change for that
fixture correction.

`cache-scm-gpu-tp{1,2,4}.log` passes real small H3 forwards on each TP size.
TaylorSeer order 2 with both dynamic and static fast SCM preserves repeated
outputs and records 6/6/3/6 executed blocks for four actual sampling calls.
The original forced-compute, TeaCache and Cache-DiT checks pass in the same
leased runs. All commit hooks pass.

Full original-weight TP4 cached/lossless/cached generation also completes with
TaylorSeer order 1, fast dynamic SCM and shared pageable VAE masters. The
256x448/107-frame/49-interval lifecycle records 23/0/23 hits on every rank;
denoise takes 27.017239 / 45.870016 / 24.999759 seconds. Peak allocation is
17,209,923,584 bytes/card for cached requests. Every request passes media and
strict actual-work validation. The first and third requests have bitwise
identical video/audio latents and all RGB frames; PCM relative L2 is
1.734748e-6, spectral cosine 0.9999999999992253 and RMS ratio 0.999999975445.
All declared repeatability gates pass (`cache-scm-repeat-quality.json`).

`cache-scm-native-summary.json` retains stages and request records. These
checks establish optional-policy execution and request isolation. They do not
qualify approximate outputs against the independent official model, compare
algorithm quality with lossless sampling or satisfy primary performance gates.

## Shared operators and layer residency integration

This branch inherits the shared dense attention interface, exact projection
epilogues and optional layer staging through the VSA/kernel dependency stack.
Sparse selection and request cache policy remain separate operations. The
combined CPU integration passes 126 checks (`cache-shared-integration-cpu.log`).

The small real-H3 distributed cache test now accepts `--layer-offload`.
TP1, TP2 and TP4 each pass all five forced-compute, TeaCache, Cache-DiT and
dynamic/static TaylorSeer-SCM cases across two consecutive requests. It stages
actual DiT blocks while keeping the first normalization/AdaLN probe weights
resident; outputs match the resident control within the existing 1e-5 gate,
executed-block counts agree across ranks and repeats, and every parameter
returns to its CPU master after each request. Cache state is removed on exit.

Evidence is `cache-layer-tp{1,2,4}-gpu.log` and `cache-layer-checks.json` in the
campaign root. These are operator/request-lifecycle integration checks; they
do not extend the recorded full-model quality or performance acceptance.

The same distributed fixture also accepts `--backend FLASHINFER_SM70`.
With the register-probability FI build inherited from kernel source `f8b85c681a`,
TP1/TP2/TP4 again pass all five policies, two consecutive requests and actual
layer staging. Backend selection is recorded in each result line. Counts agree
across ranks and outputs meet the unchanged resident-control gate. Evidence:
`cache-register-gpu.json`, `cache-register-tp{1,2,4}-gpu.log`, and the immutable
`shared-register-variants-binaries.json` manifest. This verifies cache/layer
integration on both explicit dense backends; it does not qualify cached model
quality or >80 throughput. The associated CPU integration passes 103 checks,
with seven explicitly device-masked GPU skips.

## Explicit peer reduction with cache and layer staging

The fixture at `567c060fa8` adds `--peer-reduction`, using the native pipeline
reduction owner rather than replacing a block forward. TP4 passes both
`FLASH_ATTN_V100` and `FLASHINFER_SM70` with actual layer offload, five cache
policies and two consecutive requests. Output and executed-block checks are
unchanged; every rank records peer calls, no native fallbacks and matching
communication counts. The owner is closed before distributed teardown.

The native API dependency merge `f7dbba3721` passes 110 targeted CPU checks
with one device-masked GPU skip. GPU evidence is `cache-peer-gpu.json` and
`cache-peer-{fa,fi}.log` under the campaign root. These checks establish cache,
staging and communication integration; full cached-model quality and primary
performance qualification remain incomplete.
