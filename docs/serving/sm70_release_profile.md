# SM70 release profile and acceleration status

The SM70 wheel includes `serve_qwen38_27b_nvfp4_v100.sh` and the packaged
`qwen38_27b_nvfp4_dflash2` profile. The launcher reads the profile from its
installed Python environment. Extra command-line options override its values.

```bash
python -m pip install ./1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl
serve_qwen38_27b_nvfp4_v100.sh /path/to/target
python -m vllm.sm70_profiles show qwen38_27b_nvfp4_dflash2 --json
```

This recipe targets four peer-connected V100-SXM2 32GB GPUs, TP4, CUDA 12.8,
Torch 2.10.0+cu128, FP16 target/draft compute, FLASH_ATTN_V100, a 262144-token
context, an 8192-token prefill budget, four sequences, 0.80 memory utilization,
2048-token KV blocks and 8192-token mamba blocks. Model weights are downloaded
separately; pip installs the declared Python and CUDA runtime dependencies.
No source overlay or private native extension is required.

The release owner selected E4M3 KV for the profile and the main qualification
path. It admits the grouped, long-context and scalar-tail capabilities.
Performance and output-quality qualification remain release gates.
An explicit `--kv-cache-dtype fp8_e5m2` override reports `kv_dtype` for these
E4M3 paths and is not the profile's accelerated long-context baseline.

The draft uses a fixed Hugging Face revision. The profile also records the
corresponding ModelScope commit and configuration/weight SHA256 values: commit
IDs belong to different providers, while the two checkpoint payloads match.

At configuration completion, before graph capture, the engine logs configured
SM70 route capabilities and final switch values. Disabled expected paths emit
warnings with a reason. `GET /v1/sm70/acceleration` returns the same report and
uses the server's existing API-key authentication. A configured capability is
not proof that a particular request executed a kernel; retain worker route
logs and the graceful-shutdown route summary for performance qualification.
Worker shutdown explicitly flushes route counters, because multiprocessing
workers can exit without invoking Python's `atexit` hooks. Counters cover Python
dispatch and graph capture; CUDA graph replay does not increment them.

For a selected, qualified profile, set
`VLLM_SM70_REQUIRE_PROFILE_ACCELERATION=1` to refuse startup when an expected
path is disabled. It remains off by default. On other platforms paths are
reported as `not_applicable`.

Compile-cache status is reported separately. The existing SM70 quality policy
can disable AOT cache reload; this change does not alter that policy or any
acceleration default.

The report's `expected_acceleration` lists the paths required by this recipe.
Use that list when computing enabled/total counts. Flash-Next FP16 MoE decode
is a different model contract and is `not_applicable` to this dense 27B recipe.
`compile_cache_disabled` describes cache state; it does not mean the user
disabled decode acceleration. The SM70 quality policy can select that state
automatically, so repeated startup may still spend time compiling.

| Override | Expected capability affected |
| --- | --- |
| `--kv-cache-dtype fp8_e5m2` | E4M3 grouped FP32, long-context and scalar tail: `kv_dtype` |
| `--max-num-batched-tokens 4096` | Q8000 prefill: `budget<8000` |
| `--block-size 16` | Compact scalar tail: `page_size` |
| DFlash2 with five speculative tokens | Qualified verifier defaults: `num_speculative_tokens=5≠7` |
| No speculative config | DFlash2 verifier: `method=None≠dflash` |
| `--enforce-eager` | Compile/graph capability: `user_override` |

Some operators select their default inside the quantization layer rather than
in the global environment table. In particular, NVFP4 QPN2 decode and bounded
prefill are automatic for the compatible DFlash2 contract even when no
`VLLM_SM70_NVFP4_QPN2*` variables were exported. Larger live shapes retain
TurboMind. QPN4's target-only/single-sequence contract is a separate route;
enabling every experimental switch is not a supported serving recipe.

`benchmarks/benchmark_sm70_openai_stream.py` counts returned token IDs when
reporting TPOT. DFlash2 can emit multiple tokens in one response chunk, so chunk
intervals are reported separately. Token latency describes availability at the
HTTP client: co-emitted tokens share an arrival time, and its percentiles do not
measure individual GPU rounds. The client also supports 32K and longer input
construction and an explicit `--top-k` sampling option.
