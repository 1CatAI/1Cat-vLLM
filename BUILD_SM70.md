# Building and serving 1Cat-vLLM on NVIDIA V100 (sm_70)

Validated recipe for building this fork from source for Volta (V100-32GB) and
serving with the compiled `FLASH_ATTN_V100` backend. All numbers below were
measured on a single V100-SXM2-32GB.

## Build environment

| requirement | value | why |
|---|---|---|
| Python | 3.12 | tested combination |
| torch | **2.10.0+cu128** | the fork pins `torch==2.10.0` (`requirements/cuda.txt`); the stable-ABI op library `vllm._C_stable_libtorch` requires torch >= 2.10. The cu126 wheel index tops out at torch 2.9.1, so cu128 is the correct CUDA variant. torch 2.10.0+cu128 still ships sm_70 kernels — verify with `python -c "import torch; print(torch._C._cuda_getArchFlags())"` (must include `sm_70`). Do **not** use a CUDA 13 (cu130) torch: sm_70 is dropped there. |
| CUDA toolkit (nvcc) | 12.6+ works | minor-version mismatch vs. torch's cu128 runtime is fine for compilation |
| `TORCH_CUDA_ARCH_LIST` | `7.0` | compile only sm_70; keeps the build to ~13 min on a 24-core host |

Do **not** set `VLLM_SKIP_FLASH_ATTN_V100_BUNDLE` — the bundled
`flash-attention-v100` extension is what provides the Volta attention kernels
(XQA decode, bhmd, paged prefill bhmd/bfla/splitkv, wmma decode, turboquant
decode). A wheel built without the bundle, or installed next to a stale
`flash_attn_v100` package, makes the strict `FLASH_ATTN_V100` backend refuse
to run ("required Flash op is unavailable").

```bash
export TORCH_CUDA_ARCH_LIST=7.0
export FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0
export MAX_JOBS=<cores>
pip wheel --no-build-isolation --no-deps -w dist .
pip install dist/1cat_vllm-*.whl
```

If the build environment has torch < 2.10, `setup.py` now fails loudly instead
of producing a wheel whose `_C` namespace is missing every generic op
(`silu_and_mul`, `rms_norm`, `rotary_embedding`, all of `_C_cache_ops`, ...) —
such a wheel imports cleanly but cannot load any model.

## Serving on V100

```bash
PYTHONNOUSERSITE=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<numeric index> \
python -m vllm.entrypoints.openai.api_server \
  --model <model> --dtype float16 \
  --max-model-len 8192 --gpu-memory-utilization 0.85
```

- `--dtype float16` always — Volta has no bf16.
- `PYTHONNOUSERSITE=1` if your user site-packages contains a different torch:
  a shadowing torch causes `vllm/_C.abi3.so: undefined symbol ...` at import.
- Prefer numeric `CUDA_VISIBLE_DEVICES` under `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
  UUID-form entries (`GPU-<uuid>`) are supported as of this branch (they used
  to crash device resolution with
  `ValueError: invalid literal for int() ...`); map UUID to index per boot
  with:
  `nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | sort | cat -n`
- Do **not** pass `--enforce-eager` in production. CUDA graph capture works on
  sm_70 (the fork auto-selects `mode=VLLM_COMPILE`,
  `cudagraph_mode=FULL_AND_PIECEWISE`) and is the single biggest win:

| mode | decode t/s (greedy, 128 tok, single stream) | prefill t/s (~5.3k tok) |
|---|---|---|
| eager (`--enforce-eager`) | 51.6 | 3836 |
| CUDA graphs (default) | **132.8 (2.56x)** | 3796 |

Model: Qwen/Qwen3-4B-AWQ, single V100-SXM2-32GB, `float16`, 8192 context.
Greedy output is bit-identical between eager and graphs.

- XQA decode gating: for `q_per_kv == 4` models (e.g. Qwen3-4B: 32 Q / 8 KV
  heads) the XQA route engages only when the decode sequence-length hint is
  >= 32768 (`VLLM_FLASH_V100_DECODE_XQA_Q4_MIN_SEQ_LEN`); shorter contexts use
  the scalar paged flash route. XQA is unconditional for `q_per_kv` 6 or 8.
- First requests warm up JIT/compiled kernels — warm before benchmarking.
- `VLLM_FLASH_V100_ROUTE_SUMMARY=1` prints per-route counters at exit.

---
Findings from PLI Labs V100 serving research (proprietarylegal.ai).
