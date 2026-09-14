# SM70 Q8000 integration validation (2026-09-14)

## Contract

Base `7217bb5d4f3866f87bf6a961204c894af3b03261`; optimized kernel
`7e3939f3e6`. The implementation is built into the normal
`vllm.vllm_flash_attn._vllm_fa2_C` extension. It has no private DSO or
preload dependency and remains experimental and opt-in.

The test host uses Python 3.12, Torch 2.10.0+cu128, CUDA 12.8, and four
V100-SXM2-32GB GPUs (physical 4-7 in PCI order) at TP4. The model is
Qwen3.8-27B-FP8 (`Qwen3_5ForConditionalGeneration`, 64 layers, 16
full-attention layers, 24 query heads / 4 KV heads / D256). Weights are FP8,
compute is FP16, KV storage is E4M3, the quantization backend is TurboMind,
and the attention backend is FLASH_ATTN_V100.

End-to-end runs use max length 262144, chunk size 8000, one sequence, memory
utilization 0.85, FP16 Mamba state/cache, eager execution, CUDA graphs off,
MTP off, and prefix caching off. Sampling is greedy with a maximum of 32
output tokens and EOS respected. Engine initialization and a short warmup
are outside TTFT. A deterministic natural-language prompt places a marker
at its beginning, then asks for that marker and the largest Solar System
planet.

Run `benchmarks/benchmark_sm70_79t_cold.py --model "$MODEL" --lengths 16000
128000 256000 --output-len 32 --out "$RESULT"` with the runtime flags in
README, plus `VLLM_SM70_QUANT_BACKEND=turbomind`,
`VLLM_FLASH_V100_PREFILL_USE_TRITON=0`,
`VLLM_FLASH_V100_FP8_PREFILL_BRIDGE=1`,
`VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1`, and `VLLM_USE_AOT_COMPILE=0`.
Set V37=0 for the architecture route and V37=1 for the matched control.

## Failure mechanism and final guard

The raw historical recipe assumes that unshifted exponentials and the FP16
PV numerator fit in FP16. They do on zero-mean random tensors, but they do
not on the model tensors. All captured Q/K/V inputs were finite while the
raw output contained infinities and the model emitted invalid token ID -1.

The first short-request failure occurred on rank 3, attention call 4. A
later candidate with 16x value headroom remained finite through KV144K, then
overflowed independently on ranks 2 and 3 at KV152K. Sparse score maxima
missed by the sampled shift were 15.31-16.00 above the sample. The largest
observed FP16 PV partial was 75310, beyond the FP16 finite limit 65504.

The qualified recipe retains FP16 tensor-core PV and applies four guards:

- sample one score in eight, add a 4.0 shift margin, and cap positive
  exponent input at 10.0;
- subtract a per-dimension V center when the first-4096-token mean magnitude
  is at least 0.05;
- scan all residual V values and scale them by an exact power of two with
  64x additional headroom;
- keep block masses and the online prefix/tail merge in FP32, then restore
  the V center after normalization.

The 64x headroom makes all three captured failure tensors finite. It is a
model-qualified bound, not a proof over arbitrary FP16 inputs.

## Operator result

The following medians use the final source-built artifact, 30 warmups and
100 CUDA-event samples on one V100. Useful causal FLOPs are
`4*Hq*D*(Q*(KV-Q)+Q*(Q+1)/2)`.

| KV tokens | Median (ms) | p10 (ms) | p90 (ms) | TFLOP/s | Relative L2 | Worst-row relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128000 | 80.6114 | 80.2029 | 80.8609 | **75.6081** | 0.007129 | 0.008302 |
| 256000 | 164.7010 | 163.6619 | 214.4517 | **75.2050** | 0.007052 | 0.008119 |

Both medians clear the 75-TFLOP/s target. The 256K p90 includes transient
host/GPU interference; the median and p10 remain consistent with the prior
100-sample run at 75.27 TFLOP/s.

## Cold end-to-end result

| Prompt tokens | TTFT (s) | Prompt tokens / TTFT | Full wall (s) | Subsequent decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 16000 | 3.31870 | 4821.17 | 4.62378 | 11.4937 |
| 128000 | 36.07453 | 3548.21 | 37.52012 | 10.3765 |
| 256000 | 94.58807 | **2706.47** | 95.77519 | 12.6356 |

These are single unprofiled cold requests, not confidence intervals. The
256K row is a final rerun of the formatted artifact; the 16K/128K rows come
from the immediately preceding formatting-equivalent build. TTFT includes
prefill and first-token overhead. Subsequent decode uses the 15 intervals
after the first token. `cached_tokens` is zero for every request. Every TP
rank reports respectively 16/240/496 architecture calls and the same number
of E4M3 bridge calls.

All lengths emit the same 16 tokens, including EOS:

```text
119920 96919 95761 12512 96143 97460 115783 10119
3709 145551 99960 114931 95761 147482 1710 248046
```

The decoded output is `校验词是「海蓝石榴」，太阳系最大的行星是木星。`.
Both retrieval and knowledge checks pass. The token sequence also matches
the stable FP32 implementation and matched v37 control. This is a scoped
long-context text-health gate, not broad model equivalence.

## Matched comparison

| Route | 16K TTFT / tok/s | 128K TTFT / tok/s | 256K TTFT / tok/s |
| --- | ---: | ---: | ---: |
| Optimized guarded FP16 PV | 3.3187 / 4821.17 | 36.0745 / 3548.21 | **94.5881 / 2706.47** |
| Stable FP32 PV | 3.3608 / 4760.75 | 40.0996 / 3192.05 | 111.3729 / 2298.58 |
| v37 control | 3.3294 / 4805.64 | 38.7531 / 3302.96 | 105.6480 / 2423.14 |

At 256K, the optimized route has 10.47% lower TTFT and 11.69% higher prompt
throughput than the matched v37 control. It has 15.07% lower TTFT and 17.75%
higher prompt throughput than the stable FP32 implementation.

The earlier 1032.23-second report did not select the intended architecture
and bridge. The optimized route reduces that wall-clock anomaly by about
10.8x. The reported 62.73-second second request hit prefix cache and is not
a cold-prefill result.

## Broader NVFP4 plus DFlash2 serving sample

A separate serving-quality run checked whether the guarded attention build
remains healthy in the Qwen3.8-27B-NVFP4 plus DFlash2 stack. This is a
compatibility and output-health result, not a matched comparison with the FP8
target-only cold-prefill contract above. It used TP4, E4M3 target KV,
Flash-V100 for target and draft attention, seven probabilistic draft tokens,
four concurrent requests, max length 262144, and a 65536-token output cap.
Sampling used temperature 0.6, top-p 0.95, top-k 20, seed 0, and xhigh
reasoning. The selected 96-case corpus SHA256 is
`46fcb5e990bfeb01069b9d676f87285e5672edcb8557eeada98d0a35d8b9af1e`.

The run was stopped after 78 complete cases at the operator's request. The 18
unstarted or interrupted AIME cases are excluded from every score:

| Suite | Complete / selected | Raw pass | Output tokens | Suite wall | Aggregate output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro, category-balanced | 32 / 32 | **29 / 32** | 134370 | 1044.38 s | **128.66** |
| MBPP sanitized | 32 / 32 | **31 / 32** | 101776 | 522.75 s | **194.69** |
| AIME 2024/2025 | 14 / 32 | **13 / 14** | 183238 | interrupted | not reported |
| Total complete | 78 / 96 | **73 / 78** | 419384 | mixed | not reported |

All 78 complete requests reached natural EOS, returned a nonempty final
answer, and contained no replacement characters. None hit the output cap.
The service log contains no NaN, Inf, overflow, traceback, CUDA error, or dead
engine/worker report. The three MMLU-Pro misses and one completed AIME miss
are coherent wrong answers rather than malformed output. The only raw MBPP
failure is task 229 (`mbpp:102`): its prose requires stable order within both
sign groups, while its first public assertion moves the value 2 behind values
4, 5, and 6. The model follows the prose, so this is retained as a raw failure
but classified as a dirty evaluation case.

Across all active ten-second service windows, including smoke cases and the
four interrupted requests, median aggregate generation throughput was 141.4
tok/s and p90 was 218.72 tok/s. Median DFlash2 acceptance length was 4.08 and
p90 was 4.62; median draft-token acceptance was 44.0%. The completed MMLU-Pro,
MBPP, and AIME output lengths reached 46458, 20699, and 30371 tokens. These
long natural-EOS traces show that the stack remains coherent deep into decode,
but they also expose costly xhigh overthinking tails.

The selected dataset prompts contain only 125-713 tokens. Setting max length
to 262144 verifies service capacity and long-decode compatibility; it does not
exercise the Q8000 architecture prefill route or constitute another near-256K
cold-prefill measurement. The 256K speed claim remains the matched target-only
result in the preceding section. A combined NVFP4 plus DFlash2 near-256K cold
request would be a separate acceptance item.

## Numerical gates and rejected variants

- 18 route/bridge policy tests pass.
- Three SM70 CUDA regressions pass. Two cover KV16K/KV128K large random
  scores with values biased by +8. The third places correlated score/value
  spikes at a fixed nonzero residue to reproduce numerator growth missed by
  sparse max sampling. All compare sampled rows with full-KV FP32 attention.
- The three real failure captures are finite after the final guard. Sampled
  relative L2 is 0.010756, 0.014670, and 0.006221; worst-row relative L2 is
  0.051346, 0.039120, and 0.013341 respectively.
- The raw recipe reached about 81.5/80.5 TFLOP/s at KV128K/256K on random
  inputs but emitted invalid model output. It is rejected.
- Exact row maxima plus FP32 PV reached only about 54.1/54.0 TFLOP/s. It is
  the stable diagnostic baseline, not the optimized endpoint.
- 4x value headroom failed the 16K model request. 16x passed 128K but failed
  at KV152K during the 256K request. Both are rejected.
- Increasing the score margin from 4 to 6 kept output finite but raised the
  three captured relative-L2 errors from 1.08%/1.47%/0.62% to
  3.91%/2.93%/1.25% because more weights lost FP16 dynamic range. It is
  rejected.

## Artifact identity and promotion decision

Final formatted source-built FA2 SHA256:
`c598bcf9ae0c866a7ba426f3c364a851ef205535fe3d4a8949658f935fcb6796`.
ELF dependencies are standard Torch/CUDA/cuBLAS/system libraries. Build,
pre-commit, route-policy tests, CUDA numerical tests, operator benchmark,
and cold model runs all use the owned worktree. Raw logs and captured model
tensors remain in task-local `.artifacts` and are deliberately not committed.

Keep this route experimental and opt-in. It meets the current 75-TFLOP/s,
finite-output, and scoped end-to-end quality targets. Promotion beyond this
prompt and model requires a broader long-context quality corpus or perplexity
comparison because the sampled score shift and FP16 PV path are approximate.
