# Measuring model startup and compilation-cache reuse

Startup includes Python and worker initialization, weight loading, model
compilation, memory profiling, CUDA graph capture, and warmup. The weight-loader
timer alone does not measure the wait before a user can send a request.

`benchmarks/benchmark_startup_cache.py` measures fresh server processes from
process launch until the loopback health endpoint succeeds. It retains a log
and request responses for each phase:

1. `uncached`: `VLLM_DISABLE_COMPILE_CACHE=1`.
2. `cache-fill`: `VLLM_DISABLE_COMPILE_CACHE=0`, allowing artifacts to be saved.
3. `cache-hit`: another process using the same caches, command, and requests.

The last phase must log an AOT artifact load. All phases must produce identical
deterministic response choices without hitting the output limit. Failed gates
remain in `results.json`; the benchmark stops its own server process group.
The initial phase starts with empty task caches. Source installations that need
to JIT-build native extensions therefore include that cost in their first run.
Prebuilt wheels should ship those extensions; prepare a source build before
comparing model startup independently of installation work.

## Run a controlled comparison

Reserve the GPUs before running. Use an unused loopback port and a new output
directory. Keep the model files, runtime, GPU set, TP size, context length,
batch limits, KV dtype, speculative configuration, and graph settings unchanged.
Do not compare a reduced-context or eager launch with a full production launch.

For example, save this deterministic request as `startup-request.json`:

```json
{
  "model": "startup-check",
  "messages": [{"role": "user", "content": "Compute 17 times 23."}],
  "temperature": 0,
  "top_p": 1,
  "seed": 42,
  "max_tokens": 256,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
.venv/bin/python benchmarks/benchmark_startup_cache.py \
  --output-dir startup-results --port 18552 \
  --request startup-request.json \
  -- .venv/bin/python -m vllm.entrypoints.cli.main serve /models/target \
  --host 127.0.0.1 --port 18552 --served-model-name startup-check \
  --dtype half --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 --kv-cache-dtype fp8_e5m2 \
  --max-model-len 262144 --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --enable-prefix-caching --mamba-cache-mode align
```

Add the actual production speculative configuration when measuring DFlash2.
Pass multiple `--request` files to cover text, code, and tool calling. A short
answer is only a startup smoke: run the model's normal sampling and relevant
long-output/context quality gates before enabling a cache in production.

## Diagnose misses and late startup failures

- A log saying `vLLM's torch.compile cache is disabled` means that persistent
  Inductor/Triton directories alone do not enable vLLM AOT artifact reload.
- The SM70 compile-graph policy historically disables that reload because of
  output drift. [PR #536](https://github.com/1CatAI/1Cat-vLLM/pull/536) adds
  unregistered `VLLM_*` kernel switches to the cache key.
  [PR #621](https://github.com/1CatAI/1Cat-vLLM/pull/621) proposes restoring the
  default and carries a Torch 2.10 serialization backport. Review its prerequisites
  rather than treating a cache environment variable as a correctness fix.
- AOT loads that immediately recompile may be missing the serialized Triton
  kernel table on Torch 2.10. Confirm actual artifact-load logs in a fresh process.
- A successful artifact load followed by an invalid-pointer error can mean a
  process-local scratch or weight address was serialized as an integer. The
  compressed-tensors channel-FP8 QPN8 route resolves its shared prefill workspace
  inside an opaque operator so the graph can be reused in a new process. Mixed
  NVFP4/FP8 checkpoints exercise this route even when their name says NVFP4.
- CUDA graph capture must still run in the new CUDA process; a compilation-cache
  hit does not eliminate that phase.
- Imported Python environments need their executable directories on `PATH`.
  Confirm Ninja and the CUDA toolkit can be found in the service environment,
  not just in an interactive shell. A missing extension reported by one TP rank
  can hide another rank's primary compiler error.
- Keep runtime installations and caches in persistent locations. Do not depend
  on another task's temporary source checkout, compiler directory, or kernel DSO.
