# SM70 DeepSeek V4 Compressor State-Save Fusion

## Scope

This draft folds the current-token compressor state write into the existing
Triton compress/quant/cache kernel. The route is restricted to NVIDIA SM70,
batch-one decode, full CUDA Graph capture, and FP8 KV cache. It remains opt-in
through `VLLM_SM70_DSV4_FUSED_COMPRESSOR_SAVE=1`.

Integration base: `c6c72ed0f4ff420d03312b5bc1b5ad4712e9d78f`.

## Microbenchmark Gate

Command:

```bash
python benchmarks/benchmark_sm70_deepseek_v4_compressor_save.py \
  --output-json result.json --replays 1000 --repeats 5
```

The six C4/C128 boundary and non-boundary cases were bitwise exact for both
state cache and output KV cache. The weighted median projection was only
`0.020449 ms/token`, so this is not an accepted endpoint speedup.

Raw artifact:

`/home/fudanwl/v100-worktrees/runs/dsv4-compressor-save-fusion-micro-20260803/result.json`

The targeted FP8 Triton cases passed (`6 passed`). The six MXFP4 cases still
fail on SM70 because Triton emits unsupported `cvt.e2m1x2` PTX; this candidate
does not enable fused save for MXFP4 cache.

## Status

Keep the route disabled and the PR in Draft. Before promotion it needs a
matched 1K/256 endpoint A/B and an output-quality gate. Given the current
microbenchmark projection, do not prioritize that endpoint run unless a later
trace shows a larger state-save cost.
