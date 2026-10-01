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

The initial profile keeps E5M2 KV. It therefore reports `kv_dtype` for the
E4M3 grouped, long-context and scalar-tail paths. The final release KV default
must be selected after the E4M3/E5M2 quality and performance comparison.
To test the E4M3 candidate, append `--kv-cache-dtype fp8_e4m3`.

The draft uses a fixed Hugging Face revision. The profile also records the
corresponding ModelScope commit and configuration/weight SHA256 values: commit
IDs belong to different providers, while the two checkpoint payloads match.

At configuration completion, before graph capture, the engine logs configured
SM70 route capabilities and final switch values. Disabled expected paths emit
warnings with a reason. `GET /v1/sm70/acceleration` returns the same report and
uses the server's existing API-key authentication. A configured capability is
not proof that a particular request executed a kernel; retain worker route
logs and the graceful-shutdown route summary for performance qualification.

For a selected, qualified profile, set
`VLLM_SM70_REQUIRE_PROFILE_ACCELERATION=1` to refuse startup when an expected
path is disabled. It remains off by default. On other platforms paths are
reported as `not_applicable`.

Compile-cache status is reported separately. The existing SM70 quality policy
can disable AOT cache reload; this change does not alter that policy or any
acceleration default.
