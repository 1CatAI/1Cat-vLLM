# DFlash2 E4M3 attention with FP32 state

The SM70 Flash-V100 `fp8` KV alias now resolves to `fp8_e4m3`. Explicit
`fp8_e5m2` remains available for reproducing older deployments. Explicit
formats, checkpoint-resolved formats and non-SM70 backends are preserved.
This changes the FP8 alias, not the general `auto` cache policy or model weights.

For E4M3 DFlash2 target verification, the backend no longer selects the legacy
grouped entry that stores normalized attention partials in FP16. Its advertised
E4M3 support only proves byte-format compatibility. The backend routes compatible
single-request q2–8/H6/Hkv1/D256 input through the repaired FP32 entry, using
the metadata builder's live per-query lengths, including zero padding.

The repaired computation retains compensated QK accumulation, compensated
probabilities, tile-local FP32 PV accumulation, FP32 unnormalized partition
numerators, separate FP32 max/sum, and FP32 combination. KV storage remains
E4M3; Tensor Core operands and the final activation remain FP16. This is not a
full-FP32 model and does not remove FP8 quantization error.

Precision capability revision **4** adds page sizes **1728/3456** to the existing
800/848/1616/1648/3296 runtime-stride implementation. The arithmetic is unchanged
from repaired revision 3. It also adds FP32 scalar E4M3 partial storage and
reduction. The wrapper requires revision 4 so a new page or FP32 workspace
cannot reach an incompatible native binary. Rebuild the extension and restart
workers together with this Python update.

Unsupported grouped shapes, independent-request batches, q16 and explicit grouped
rollback use the scalar path with FP32 accumulation and the new FP32 split
workspace. Scalar and XQA workspaces are separated by partial dtype in the cache.
They do not re-enter the legacy E4M3 FP16-partial verifier. If the native binary
lacks revision 4, E4M3 scalar calls raise a rebuild error instead of silently
storing half partials. A DFlash2 target q1 also uses FP32 scalar state instead of
the half-partial XQA wave route. Ordinary non-DFlash XQA dispatch is unchanged.
The low-level legacy operator remains available for numerical A/B tests and
explicit E5M2 compatibility; it is not the E4M3 serving policy.

The Qwen3.8 DFlash2 configuration also enables
`VLLM_SM70_DFLASH2_FP32_LOGITS=1` by default so candidate rerank and dense fallback
retain FP32 logits. This is model-scoped configuration; the global environment
default remains off for unrelated models. Explicit environment overrides are
preserved. This does not enable MTP or change the sampling distribution settings.

## Observability and reproduction

A compatible worker reports `E4M3 grouped FP32 route selected` with its page,
query rows and FP32 state representation. Route summaries include
`prefill_smallq_e4m3_grouped_fp32` and `fp8_kv_decode_grouped_fp32`.
Check these alongside the resolved KV dtype and FP32-logit preparation; an image
name or requested flag is insufficient evidence of the numerical route.

With an isolated Python 3.12 runtime, CUDA 12.8, Torch 2.10+cu128 and idle V100s:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest -q \
  tests/v1/attention/test_sm70_flash_v100_policy.py \
  tests/v1/attention/test_sm70_e4m3_grouped.py \
  tests/v1/spec_decode/test_dflash2.py \
  tests/kernels/attention/test_sm70_grouped_e4m3_fp32.py

CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/benchmark_sm70_dflash2_fp32_attention.py --output operator.json
```

Point `PYTHONPATH` at the task's Python package and rebuilt extension. The
operator benchmark holds E4M3 bytes, query, causal visibility, KV scales and
native library fixed. It reports 20 warmups, 25 ABBA blocks (50 samples per
variant), latency percentiles, PyTorch FP32/FP64 arithmetic error and the FP16
output-rounding floor. It separates avoidable attention error from cache
quantization, and is not a model-throughput benchmark.

Model acceptance must be checked with request-level draft/accepted-token
counter deltas. A rolling logger's mean acceptance length does not provide a
matched request comparison. FP32 is the chosen arithmetic contract, but lower
operator L2 alone does not establish model-quality improvement or guarantee
identical sampled tokens.

## Rebuilt operator results, 2026-09-08

The initial routing build passed **420 tests** before the scalar fallback was
extended. The native q8 tests include 8K/64K/128K/256K, newly admitted
1728/3456 pages, relocated pages, non-unit scales and CUDA Graph replay with
changed/zero row lengths. Numerical assertions bound error relative to the
FP16 output-rounding floor, rather than only using an aggregate loose tolerance.
The final scalar/q1 policy follow-up passes **338 checks**, including all five
new scalar tests. The separate native/planner run passed 102 checks and exposed
an incorrect test assertion that eager and graph streams must share a workspace;
the corrected tests check reuse within one stream and FP32 buffers in both.
This was a test expectation error, not an attention numerical failure.

Physical GPU2, V100-SXM2-32GB, Torch 2.10.0+cu128, CUDA 12.8.93/GCC12,
q8/page3296, one rebuilt library and paired graph samples:

| Context | Legacy ms | FP32 ms | Legacy relative L2 | FP32 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 0.0778 | 0.1034 | 0.00035316 | 0.00020792 |
| 65536 | 0.2714 | 0.4209 | 0.00034410 | 0.00020785 |
| 131072 | 0.4977 | 0.7916 | 0.00035442 | 0.00020903 |
| 262144 | 0.9462 | 1.5206 | 0.00033226 | 0.00020327 |

L2 uses PyTorch FP32 attention over identical quantized KV. The companion FP64
reference gives an FP16 rounding floor of 0.000203267 at 256K. The operator
has about 39% lower L2 and 61% higher latency at that length; this is a deliberate
precision choice, not a speedup. It does not establish a model acceptance gain.
No cross-GPU absolute timing is combined.

Native library SHA256:
`d2f70b502985af14fe816b379ffa319ca4887191dfab311d62836ee121a41ef3`.
Raw artifacts are indexed under `dflash2-e4m3-fp32-default-20260908`, including
`operator-r2.json` with all samples and both numerical references.
