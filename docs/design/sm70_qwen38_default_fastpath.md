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

## SM70 concurrency switch audit (TP4, Qwen3.8-27B-NVFP4 dense)

Every concurrency switch below was measured rather than assumed, because the
defaults live in three different places and only one of them is `envs.py`.

Protocol: 4 x V100-SXM2-32GB, TP4, `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`,
`fp8_e4m3` KV, 32768 context, `max_num_batched_tokens=8192`,
`max_num_seqs=16`, prefix caching on, fixed 256-token outputs with
`ignore_eos` and `min_tokens`, identical compile cache, one server per arm.
Reported as aggregate tok/s at C=1/4/8/16.

### Measured effects

| Switch | C=1 | C=4 | C=8 | C=16 | Verdict |
| --- | --- | --- | --- | --- | --- |
| `VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY` | — | — | — | **+3.4%** | default on |
| `VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES` | — | — | — | **+1.0%** | default on |
| both, against both off | +7.8%* | 0% | -0.9%* | **+4.3%** | merged |
| `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH` | **-29.2%** | **-17.6%** | **-18.8%** | **-12.7%** | keep off |
| `VLLM_SM70_NVFP4_MOE_GROUPED_DECODE` | n/a | n/a | n/a | n/a | inapplicable |

`*` the C=1 and C=8 deltas are marked because their off/on ranges overlap.
Only the C=16 effect is separable from run-to-run spread: off measured
807.5/812.7/813.8/820.6, on measured 830.8/848.3/849.2/855.4/857.6 over the
campaign, and those two intervals do not intersect.

### Why the switches were invisible

`csrc/custom_all_reduce.cuh` reads the two push all-reduce switches with
`std::getenv` at kernel-launch time. The declarations in `vllm/envs.py` were
never consumed by anything, so enabling the paths required exporting the
variables by hand and no default deployment ever ran them. `envs.py` now
publishes its resolved values into `os.environ`, so the native path follows the
declaration. Any future C++/Python switch pair needs the same treatment.

### Scope limits

- `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH` was measured on the dense 27B path only.
  The -13% to -29% regression is large enough that it should not be assumed
  neutral elsewhere, but it has not been measured on MoE or speculative paths.
- `VLLM_SM70_NVFP4_MOE_GROUPED_DECODE` gates on an exact
  `(num_experts, hidden, intermediate, top_k) == (512, 2560, 160, 10)` shape.
  This checkpoint has no experts at all, so the switch cannot fire; no
  conclusion is drawn about its effect on checkpoints that do match.
- AWQ, FP8-MoE and DFlash2 switches were not exercised: this checkpoint is
  neither AWQ nor MoE, and the runs carried no speculative config.
- C=1 spread across the campaign reached 22% (57.9-70.8) for one and the same
  configuration, which is wider than the effect being measured. Any default
  decision that rests on C=1 needs at least ten repeats per arm first.

### Not run

The closed-loop numerical and mixed-size graph replay gates referenced by the
original opt-in restriction on the two push all-reduce switches. The evidence
here is throughput only.
