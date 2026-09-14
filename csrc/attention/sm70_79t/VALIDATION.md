# SM70 Q8000 integration validation (2026-09-14)

## Contract

Base `7217bb5d4f3866f87bf6a961204c894af3b03261`, implementation
`35c7342560`; normal source-built FA2/vLLM/Flash-V100/FlashQLA extensions.
Python 3.12, Torch 2.10.0+cu128, CUDA 12.8, four V100-SXM2-32GB GPUs
(physical 4–7 in PCI order), TP4. Model: Qwen3.8-27B-FP8
(`Qwen3_5ForConditionalGeneration`, 64 layers, 16 full-attention layers,
24 query heads / 4 KV heads / D256). FP8 weights, FP16 compute,
E4M3 KV, TurboMind quantization backend, FLASH_ATTN_V100 attention.

Max length 262144, chunk size 8000, one sequence, memory utilization 0.85,
FP16 Mamba cache/state, eager, graphs disabled, MTP off, prefix cache off.
Greedy sampling (temperature 0, top_p 1, top_k -1), max output 32, EOS
respected, thinking disabled. Engine initialization and short warmup are
outside timing. The same exact-token natural-language prompt requests a
marker placed at the beginning and the largest Solar System planet.

Run `benchmarks/benchmark_sm70_79t_cold.py --model "$MODEL" --lengths
16000 128000 256000 --output-len 32 --out "$RESULT"` with the runtime
flags in README, plus `VLLM_SM70_QUANT_BACKEND=turbomind`,
`VLLM_FLASH_V100_PREFILL_USE_TRITON=0`, `VLLM_FLASH_V100_FP8_PREFILL_BRIDGE=1`,
`VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1`, `VLLM_USE_AOT_COMPILE=0`, and the
local RPC setting documented in README. Set V37=0 for the architecture,
V37=1 for the control. Caches are owned by the task. No private library
paths or preload overrides are used.

## Corrected architecture cold requests

| Prompt tokens | TTFT (s) | Prompt tokens / TTFT | Full wall (s) | Subsequent decode (tok/s) |
| --- | --- | --- | --- | --- |
| 16000 | 3.36081 | 4760.75 | 4.69272 | 11.2623 |
| 128000 | 40.09957 | 3192.05 | 41.51768 | 10.5775 |
| 256000 | 111.37289 | 2298.58 | 112.75011 | 10.8916 |

These are single unprofiled requests, not confidence intervals. TTFT includes
prefill and first-token overhead; it is not isolated GPU prefill time.
Subsequent decode uses 15 intervals after the first token. Each request
emitted the same 16 tokens, including EOS, answering both questions correctly.
This is a scoped retrieval/text-health check, not broad model equivalence.
Cached tokens are zero for all cases. Every TP rank reports respectively
16/240/496 architecture calls and matching E4M3 bridge calls. The resolved
FA2 module is the normal extension in the owned source tree.

The previous 256000-token 1032.23-second report used a different runtime
and failed to select the intended architecture/bridge. It is diagnostic
context, not a matched benchmark baseline. The previous 62.73-second
request hit prefix cache. Historical ~2460 tok/s used 261888 tokens and
E5M2 KV, so it is also not a matched comparison.

## Numerical gates and rejected variants

- 18 route/bridge policy tests pass.
- Two CUDA regressions (KV16000 and KV128000, large scores, values biased
  by +8) pass against sampled full-KV FP32 attention, rtol 0.01 / atol 0.03.
- Captured real Q/K/V from all four ranks now produce finite output;
  sampled relative L2 errors are 0.002482, 0.001103, 0.001107, 0.001025.
- Raw historical recipe reached 81.50/80.45 TFLOPS at KV128K/256K on
  zero-mean random inputs, but overflowed on real model inputs and emitted
  invalid -1 token IDs. Those speeds are not qualified model results.
- Adding row maxima and value scaling fixed overflow, but retaining FP16
  MMA accumulation failed the biased-value oracle by 2–4%. Rejected.
- FP32 MMA without a register bound produced 150-register/512-thread
  kernels exceeding V100 register capacity. The corrected recipe uses a
  128-register cap and checks launches.
- Corrected operator median: 112.628 ms / 54.115 TFLOPS at KV128K;
  229.575 ms / 53.953 TFLOPS at KV256K. Sampled relative L2 about 0.0023.
  Ten timed CUDA-event samples after three warmups; useful causal FLOPs
  follow `4*Hq*D*(Q*(KV-Q)+Q*(Q+1)/2)`.

The build option remains off by default. This is a stable integration of
that architecture, not a claim that the raw 79T recipe is model-safe.

## Artifact identity and checks

End-to-end candidate FA2 SHA256:
`be632ddc94777cd43798ea68d5f11b0cbd23b8bbeefd71f89a8505f4c4d8e855`.
After formatting only, the normal installed artifact is
`ed4e274c62d5e1be0439934bcdd36208304bc82e2bcc689cea2dab4f002c4f24`;
the two CUDA regression tests pass again. No performance claim is inferred
from formatting. ELF dependencies are standard Torch/CUDA/cuBLAS/system
libraries; there is no task-cache DSO dependency.

Applicable commit hooks pass except the skipped pre-existing mypy-local
error on `flash_attn_v100.paged_kv_utils` in the unchanged backend import.
Raw logs, full route counters/token IDs and captured tensors are retained
in the task-local `.artifacts` directory and are deliberately not committed.

## Promotion decision

Keep the new route experimental and opt-in. Correcting the old dispatch
recovers the anomalous end-to-end latency, but does not establish a speedup
over a correctly selected v37 route. The raw 79T recipe cannot replace v37
on the basis of random-input TFLOPS. Further optimization must preserve the
new numerical regression and the matched 256K quality gate.

## Matched v37 control

| Prompt tokens | TTFT (s) | Prompt tokens / TTFT | Full wall (s) | Subsequent decode (tok/s) |
| --- | --- | --- | --- | --- |
| 16000 | 3.32942 | 4805.64 | 4.80879 | 10.1395 |
| 128000 | 38.75308 | 3302.96 | 39.97518 | 12.2742 |
| 256000 | 105.64800 | 2423.14 | 106.83766 | 12.6088 |

The corrected architecture has 5.42% longer 256K TTFT than v37 in this single matched run. It is not a prefill speedup over v37. All three prompt hashes and output token sequences match, with zero cached tokens. The healthy v37 route remains the performance baseline.
