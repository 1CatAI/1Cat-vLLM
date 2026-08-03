# SM70 DeepSeek V4 MXFP4 Grouped Prefill

## Scope

This WIP routes populated prefill experts through one TurboMind grouped MoE
launch instead of the per-expert loop. The route is opt-in through
`VLLM_SM70_MXFP4_MOE_GROUPED_PREFILL=1` and is based on the accepted active
expert branch `2872c6a3af4897fc59855b8a59310fa3789ab5fc`.

The branch also strengthens the CUDA Graph dynamic-route contract benchmark
and adds focused operator and streaming endpoint benchmark drivers.

## Current Gate

Python compile, Ruff, Ruff format, and `git diff --check` pass. The C++ route
has not yet been rebuilt and run on the TP8 V100 host, so there is no numerical,
prefill, or endpoint performance claim.

## Status

Keep the route disabled and the PR in Draft. Promotion requires a grouped vs.
legacy operator exactness comparison, repeatability across routing patterns,
and a matched 1K prefill endpoint A/B with decode held constant.
