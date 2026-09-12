# Qwen3.8 Flash-Next SM70 default-path audit

The historical approximately 98 tok/s single-request baseline was measured
with FP16 activations/KV, native FP32 SSM state, NVFP4 expert weights,
TP4/PP1, 262144 context capacity, an 8192-token prefill chunk and no MTP or
prefix cache. It used explicit optimization switches.

This change lets ordinary model/capacity arguments select the missing parts
of that route. End-to-end validation remains pending; this document does not
claim a new speed result or that configuration tests reproduce throughput.

The initial audit found checkpoint-FP16 GEMV, fused GDN input and fused HC
disabled by default. This also prevents the dependent auto dual-compile and
hybrid PLE route. Most MoE, router, QSA and TP4 push optimizations are already
enabled. Shared-expert overlap and MoE add/reduce also required opt-in.

## Model-scoped defaults

The following five switches now default to 1 only for the matching
Qwen3.8 Flash-Next architecture and dimensions, checkpoint NVFP4
(`modelopt_fp4`), FP16 activations/KV, native FP32 SSM, all-SM70 local TP4/PP1,
DP1, no MTP, no LoRA, no expert parallelism and no dual-batch overlap:

- `VLLM_SM70_QWEN38_FP16_GEMV`
- `VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16`
- `VLLM_SM70_QWEN38_FUSED_HC_FP16`
- `VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP`
- `VLLM_SM70_MOE_ADD_ALLREDUCE`

Existing explicit values are preserved, including 0. Other models, hardware,
quantizations and speculative configurations do not receive these new
defaults. Enforce-eager/no-compile-decode-graph requests also skip this
auto-selection. No kernel arithmetic or precision is changed by this PR.

The existing dependent selectors can then enable dual compilation and hybrid
PLE automatically. Prefill uses asynchronous disk-mmap lookup; decode uses
local pinned-UVA lookup. **Hybrid PLE still needs substantial host RAM** and
must not be described as a disk-only, low-RAM mode.

## Configuration verification

Using the real checkpoint metadata and ordinary engine arguments, with no
`VLLM_*` launch variables and without loading weights or initializing CUDA:

| Resolved setting | Before this change | With this change |
|---|---|---|
| Five switches above | Off | On |
| Qwen3.8 dual compilation | Off | On |
| Hybrid PLE / CPU and disk offload | Off | On |
| Model Runner V2 | On | On |
| Full + piecewise graphs, native RMSNorm priority | Selected | Selected |
| SSM cache | FP32 | FP32 |

The configuration regression suite passed 23 tests; the benchmark contract
and correctness-gate suite passed 12 tests. These establish selection and
failure-handling behavior, not output equivalence or performance.

## Default-route reproduction

Build the runtime from the current checkout as described in
[the baseline reproduction guide](sm70_qwen38_reproducible_baseline.md).
The same public benchmark accepts `--use-defaults`:

```bash
python benchmarks/benchmark_sm70_qwen38_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --runtime-dir /path/to/fresh-runtime \
  --output /path/to/results/defaults.json \
  --reference-json /path/to/accepted-reference.json \
  --repeats 2 --long-context --use-defaults
```

This mode clears inherited `VLLM_*` switches and does not inject the explicit
baseline optimization map, quantization override, attention-backend override,
or graph/kernel configuration. Native library paths and isolated build caches
remain explicit reproduction plumbing. A compatible native vLLM installation
is still required; source-overlay users can supply
`--native-extension-dir /path/to/native-vllm`.

The workload still specifies TP4, FP16 activations/KV, 262144 context capacity,
8192-token prefill chunks, one text-only request, 0.90 GPU memory utilization,
no prefix cache and no speculative decoding. Default speed must be checked with that
same workload, not inferred from an unrelated serving concurrency or prompt.

The driver retains natural-output health checks, token-for-token comparison
with the accepted reference, per-worker runtime hashes and FP32 SSM checks,
and shutdown after the test. It reports launch settings and resolved worker
settings so an inherited fast-path flag cannot silently count as a default.
The long-context option adds 261631+513 and 262143+1 boundary cases.
