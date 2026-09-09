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
continuous cached-step limit; TaylorSeer and SCM profiles are not exposed yet.

Omitted request quality follows the selected deployment policy. Explicit
`quality="lossless"` disables request caches. `quality="high"` selects the
official conservative Cache-DiT profile: Fn=1, Bn=0, warmup=4, threshold=0.04,
continuous reuse=1, without TaylorSeer or SCM. A TeaCache deployment rejects
`high` because the two backends are mutually exclusive.

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
