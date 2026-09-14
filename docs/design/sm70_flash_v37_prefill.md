# FlashAttention-V100 v37 prefill integration

## Scope

We integrate the v37 FP32-accumulating long-prefill route and an explicit
E4M3-to-FP16 bridge into the SM70 FA2 extension. We retain ordinary decode,
model weights, sampling, SSM arithmetic, and speculative-decoding behavior.
No speculative configuration is used in the model validation below.

The implementation and admission contract are described in
[the kernel README](../../csrc/attention/sm70_v37/README.md). This change
does not rename the global `fp8` encoding or turn an E5M2 byte cache into E4M3.
The new E4M3 bridge is selected only for explicit `fp8_e4m3` storage.

## Why this is more than copying the private kernel

The private endpoint accepted Q8000. The current chunked-prefill scheduler
uses a different family of query lengths, so installing that endpoint alone
does not prove that a model benefits. We replace fixed packed-row strides
with the actual query/head extent and test the admitted alignment family.
The persistent score allocation is bounded by the maximum admitted query
extent. We retain the causal offset when prepending query padding.

The v37 symbol is separate from the legacy GQA symbol. Missing new native
code triggers a warning and an exact fallback, never a false v37 route hit.
We also join the private tail on exception before releasing its inputs and
temporary state. The legacy architecture remains available for rollback.

Set `VLLM_FLASH_V100_PREFILL_D256_GQA_V37=1` before starting workers to select
this v37 operator as a rollback or matched control. The default value of 0
selects the qualified Q8000 architecture operator on its narrow shape family;
other shapes retain their existing fallback. The E4M3 bridge is independent of
this compute-kernel selection. Runtime environment mutation in an already
initialized engine is not a rollback mechanism.

### Preserve the E4M3 decode launch contract

For the measured no-MTP TP4 E4M3/page1568 route, set these existing switches
before starting the engine or any worker:

```bash
export VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1
export VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO=1
export VLLM_FLASH_V100_XQA_E4M3_G6_WAVE_PARTITIONS=1
export VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH=1
```

Use explicit `kv_cache_dtype="fp8_e4m3"`, `speculative_config=None`, TP4,
`attention_backend="FLASH_ATTN_V100"`, `max_model_len=262144`,
`max_num_batched_tokens=8192`, `max_num_seqs=1` and
`gpu_memory_utilization=0.8`. The measured activation
and convolution state are FP16; the resolved SSM state is FP32. Keep prefix
caching off when reproducing the reported prefill timing.

The wave switches remain opt-in and obey their existing native shape/layout
gates. This PR does not globally enable them or reinterpret the `fp8` alias.
The `decode_xqa_p64_page1568` counter records a planning hint, not the final
device-selected partition. To verify the native route, additionally set
`VLLM_FLASH_V100_XQA_E4M3_G6_P64_P256_AUTO_TRACE=1` and check both the native
`merged_long=1, converter=shared-lut` message and the worker's long-context
CUDA Graph dispatch message. Setting a switch without seeing the relevant
execution is not a performance qualification.

## Operator evidence

The fixed-shape port matches the private v37 output bitwise on 12 complete
real-operand replays: three attention layers and KV lengths 16K, 64K, 128K,
and 256K. After replacing fixed row strides, the layer-63 Q8000 replay still
matches bitwise at all four lengths. Cropped operands are operator tests,
not fresh model executions.

On one physical V100, 20 warmups and 100 ABBA pairs compare the dynamic port
with the retained private v37 at Q8000/Hq6/Hkv1/D256/FP16/causal:

| KV tokens | Private v37 median ms | Port median ms | Median paired speedup |
| ---: | ---: | ---: | ---: |
| 128000 | 99.460 | 99.601 | 0.99877 |
| 256000 | 201.507 | 201.743 | 0.99948 |

These are complete dense attention endpoints, including prefix, causal
tail, state merging and synchronization. They exclude paged gathering and
FP8 conversion. The speedup is the median of paired ratios, not the ratio
of the two reported marginal medians. This is a port-regression comparison,
not a new speedup over an old 18-TFLOP/s or 60-TFLOP/s implementation.

A separate retained all-row FP64 audit on identical E4M3-representable KV
values found the following relative L2 errors for layer-63 Q8000 operands:

| KV tokens | Native paged error | Private v37 error |
| ---: | ---: | ---: |
| 64000 | 0.15823% | 0.02788% |
| 128000 | 0.26685% | 0.02679% |
| 256000 | 0.44761% | 0.02581% |

This isolates attention arithmetic from KV quantization. It does not show
that E4M3 has no quantization loss, nor that all model tokens must match an
unquantized reference. A constant-V result of exactly one on one layer is
also not a universal guarantee: other retained layers differ by one FP16
rounding step.

## Model comparison and interpretation

The initial source-overlay comparison fixes Qwen3.8-27B-FP8, TP4 on four
V100s, FP8 weights, explicit E4M3 KV, FP16 activations/conv state, FP32 SSM
state, no MTP, chunk budget 8192, max length 262144, prefix caching off and
CUDA graphs on. Every worker reports the actual KV page size as 1568.
The control uses the old main E4M3 direct-paged route; the candidate adds
the E4M3 bridge and v37. All other native libraries and settings are shared.

The initial post-warmup results are single observations per context, not a
publication-grade repeated timing study:

| Prompt tokens | Control prefill s | Candidate prefill s | Control decode tok/s | Candidate decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1.709 | 1.700 | 59.433 | 59.268 |
| 65536 | 76.328 | 16.825 | 37.650 | 37.448 |
| 128000 | 272.040 | 39.775 | 26.688 | 26.508 |
| 256000 | 1049.836 | 106.891 | 16.790 | 16.763 |

Prefill uses the engine's scheduled-to-first-token interval. Decode excludes
the first token and uses the interval between the first and last generated
tokens. The timing request forces 64 tokens and requests five logprobs;
retrieval quality is scored only before the first EOS. Natural-EOS text
checks are recorded separately. The initial retrieval answers match at all
four lengths, and every rank records 848 v37/E4M3-bridge prefill executions.

The large model prefill ratios combine two changes: enabling a previously
missing E4M3 bridge and selecting the v37 kernel. They must not be described
as v37-only kernel gains, reused as PR122/E5M2 baselines, or substituted into
the paper's earlier 60-TFLOP/s comparison without a matching contract.
Both initial runs omitted the existing E4M3 B1 long-context auto/wave
partition flags. Their similar decode rates therefore establish neither
historical performance parity nor decode-speed admission. The earlier
PR285 result of 50.376 tok/s at final context 262144 used this long-wave
route, with NVFP4 rather than the FP8 model weights used here. Promotion is
paused while a same-contract FP8 comparison isolates the missing dispatch.

## Final runtime audit: model-parity hold

PR [548](https://github.com/1CatAI/1Cat-vLLM/pull/548) remains in Draft.
Operator precision, port latency and memory-safety checks pass, but the
strict natural-output token-parity gate does not. Do not describe this as
a completed model-quality promotion.

The final attention components are the CMake-built FA2 target and a clean
native Flash rebuild, both CUDA12.8/GCC12. Shared core/stable dependencies
remain pinned; this is not a complete newly built wheel. The matched model
uses the explicit launch contract above, Torch2.10.0+cu128, FP8 weights,
deterministic temperature0 and top-5 logprob recording. Timing requests
generate64 tokens; natural-quality requests respect EOS with a512-token cap.

The first full comparison also changed native dependencies and found a
near-tie wording divergence at reasoning token279. The prompt has72 input
tokens, with no prefix, so that request cannot enter the v37 operator.
Both reasoning answers give the correct9 red/13 blue counts and explain
why6.5 whole balls cannot be moved. This observation is retained, not
relabelled as a passing strict-parity result or proof of degraded semantics.

A third run holds the native Flash and paged-helper binaries identical and
compares retained FA2/JIT-v37 against the parent-owned FA2/v37 port. This is
the relevant single-library integration comparison:

| Input tokens | Retained / release prefill s | Retained / release decode tok/s |
| ---: | ---: | ---: |
| 8192 | 1.705 / 1.699 | 58.918 / 59.001 |
| 65536 | 16.811 / 16.808 | 55.623 / 55.493 |
| 128000 | 39.747 / 39.779 | 50.447 / 50.505 |
| 256000 | 106.985 / 107.000 | 43.136 / 43.109 |
| 262080 +64 output | 110.551 / 110.560 | 42.904 / 42.941 |

These remain single post-warmup model observations, not statistical paired
model measurements. The exact262144 final-context request completes without
non-finite recorded logprobs or a corruption flag. All timing-request
natural prefixes match; forced post-EOS tokens are excluded from quality.

| Natural-EOS task | Retained / release output tokens | Exact token parity |
| --- | ---: | --- |
| Arithmetic | 4 / 4 | yes |
| Chinese explanation | 72 / 72 | yes |
| Reasoning | 394 / 394 | yes |
| Python function | 129 / 129 | yes |
| 128000-input summary | 180 / 180 | yes |
| 256000-input summary | 157 / 158 | **no** |

The matched-native256K summary first differs at token109, changing a phrase
equivalent to "this unique phrase" versus "this verification phrase".
Both summaries are coherent and retrieve the required phrase, but they do
not satisfy the strict identity gate. Common top-5 logprobs are not bitwise
equal either; their maximum difference is0.203125 on the matched reasoning
stream. This is not full-vocabulary KL or a perplexity result.

Further isolation finds bitwise-identical v37 outputs on17 real-derived
dynamic shapes, including Q64/384/1600/8192 with KV16K–256K. Queries beyond
the8000-row capture repeat recorded rows; these are derived operator tests,
not live model activation captures. The eight native D256 dense-prefill
kernel instruction dumps match, and six real no-MTP decode replays have
identical numerical metrics across native builds. These negative findings
do not establish the cause of the model-level divergence. Resolving it,
including possible independent-process variability, remains a merge gate.

Post-rebase checks:141 CPU tests passed/1 GPU-only skip;33 GPU tests passed
(25 v37 and8 wave-route cases). The earlier CMake/OOM suite passed26 tests;
Compute Sanitizer reported zero errors. The reviewed Python output passes
its three emitted assertions and six additional cases. All model workers
assert no speculative configuration and no drafter. Only GPUs0–3 were used.

## Rejected paths and remaining admission work

An initial variable-shape build retained a fixed 48000-row PV-statistic
stride and failed with an illegal access. The stride is now dynamic.
Q1664/KV4848, which has only 16-token KV alignment, failed the FP64 test with
relative L2 0.151814. We reject that shape before launch and retain the
32-token KV requirement; it is not a passed numerical result.

The source overlay is not a complete rebuilt vLLM wheel. The final compiled
FA2 target, clean native Flash dependency, stream/memory-safety checks,
short-query latency screening, and natural long-output comparison must be
qualified before promotion. No result above by itself establishes universal
model-quality equivalence or authorizes an unrelated MTP change.
