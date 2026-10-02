# Flash-Next MTP4 evaluation with a 256K configured context

This frozen evaluation follows PR #703 and records speed, MTP acceptance,
and accuracy on fixed 64-example subsets of GSM8K, MATH-500, and HumanEval
with a 262,144-token configuration. It is a development sample, not a
full-dataset leaderboard score or a qualification of current main.

## Frozen runtime

- MTP integration source: `db292f9a49318459f064075bfdbbed438b4e77a3` (PR #703).
  The target-only control additionally includes the compilation guard fix
  `6337a96bee052bf26f04a49dd0ca0f1ae5217a46` described below.
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, checkpoint NVFP4 experts,
  source-built native extensions, no wheel build or private kernel overlay.
- Hardware: TP4 on four NVIDIA V100-SXM2-32GB GPUs. Python 3.12.13,
  Torch 2.10.0+cu128, CUDA 12.8.93, driver 580.173.02.
- FP16 activations, `kv_cache_dtype=auto` (FP16 main QSA cache),
  FP16 Mamba convolution cache and FP32 Mamba SSM state; cache mode `align`.
- `max_model_len=262144`, `max_num_seqs=1`,
  `max_num_batched_tokens=2048`, `gpu_memory_utilization=0.97`.
- PLE table entirely in pinned host memory, approximately 11.92 GiB per rank.
- V2 model runner, FULL_AND_PIECEWISE CUDA graphs, prefix caching enabled
  but explicitly reset before every measured request. No profiler attached.
- Target verification uses batch decode (five tokens for one MTP4 request);
  four draft steps use greedy draft sampling. All PR #703 acceleration switches are enabled.
- No-MTP controls retain the same common acceleration switches, GPU set,
  precision, context cap, prefill batch size, memory fraction, and prompts.
  `VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1` disables automatic speculation.

The 2048-token prefill batch reduces temporary memory and the estimated graph
reserve sufficiently to admit the requested context. The measured engine reports
5.18 GiB available KV cache, capacity 364,130 tokens, and 1.39 theoretical 256K requests;
the configured scheduler still permits only one concurrent request. Merely
changing the old 8192-token prefill configuration's length cap was not used as
evidence of actual 256K support.

**The automatic 5.18-GiB KV budget did not survive the actual 256K request.**
After completing all 192 questions and the 128K speed probe, draft QSA indexer
prefill failed at approximately 225K computed tokens: a 64-MiB FP32 cuBLAS
score tile could not be allocated, with only 40.12 MiB free. This was temporary
workspace pressure, not exhaustion of usable KV blocks (reported KV usage
was about 64%). The follow-up explicitly sets `kv_cache_memory_bytes=4294967296`
and keeps precision, kernels, context limit, and prefill batch size fixed.
With this override, the explicit byte budget determines KV allocation rather
than `gpu_memory_utilization`. The full 192-question scores retain their
original contract; only the fixed first eight questions per dataset are
replayed to audit the changed capacity and compare against no-MTP.

Native extension SHA256:

```text
_C.abi3.so
790ca7b49e83c2289c98b167ca070146d10643f63dab653ee6a54e95f56a175f
_C_stable_libtorch.abi3.so
ea479867dce18c14b91c16045b62ef1ab20634a211069c3d89ace7d252ddda63
```

Enabled measurement switches, in addition to the common SM70 platform defaults:

```text
VLLM_SM70_QWEN38_GDN_INPUT_BATCH=1
VLLM_SM70_NVFP4_MOE_GROUPED_MTP5=1
VLLM_SM70_RMSNORM_GATED_EXACT=1
VLLM_SM70_MTP_MOE_FP16_EXACT=1
VLLM_SM70_FUSED_SIGMOID_MIXED_QKV=1
VLLM_SM70_MTP_HC_BATCH=1
VLLM_SM70_MTP_HC_COOPERATIVE=1
VLLM_SM70_MTP_ROUTER_BATCH=1
VLLM_SM70_MTP_HC_FULL_UNROLL=1
VLLM_SM70_MTP_ROUTER_TOP16=1
VLLM_SM70_MTP_SHARED_BATCH=1
VLLM_SM70_MTP_PLE_CONV=1
VLLM_SM70_QSA_MTP_TOPK=1
VLLM_QWEN4EXP_PLE_HOST_GIB=12.0
```

## Sampling and scoring

For each dataset independently, shuffle its full row index list with
`random.Random(42)` and take the first 64. Run one completion per question,
interleaving GSM8K, MATH-500, and HumanEval. Request seed is `20260927 + ordinal`.
The checkpoint chat template has thinking enabled and its default `xhigh`
reasoning effort. Target sampling uses the checkpoint defaults:
temperature 1.0, top-p 0.95, top-k 20, normal EOS. Output limits are 4096 for
GSM8K/HumanEval and 8192 for MATH-500; truncations are reported explicitly.
After the primary run, every truncated example receives one supplemental
continuation: prefill the original prompt plus its saved generated tokens,
then allow another original-budget number of tokens. Sampling settings and
the request seed are retained, but the sampler RNG restarts. This avoids
regenerating the saved reasoning; it is a diagnostic continuation rather
than a fresh larger-budget pass@1 sample. These results never replace the
primary sample scores.

| Dataset | Population | Sample | Actual prompt tokens, min–max |
| --- | ---: | ---: | ---: |
| GSM8K test | 1319 | 64 | 93–216 |
| MATH-500 test | 500 | 64 | 85–344 |
| HumanEval | 164 | 64 | 118–411 |

The dataset questions are short; configuring 256K does not turn them into
256K-input tests. Separate long-context probes exercise that capacity.

- GSM8K: final numerical answer exact match, with mathematical parsing for
  equivalent formatted answers.
- MATH-500: final boxed answer equivalence with `math-verify==0.9.0`; use only
  the final response after `</think>`, not intermediate reasoning matches.
- HumanEval: chat-adapted function generation, one completion per task,
  execution of the provided tests in a Landlock/seccomp sandbox using the
  isolated Python environment. Preserve imports from the original task prompt.
  No network access; 5-second CPU and 8-second wall limits per candidate.
- The 64 HumanEval canonical solutions and 64 MATH-500 canonical solutions
  pass the same graders. Wrong-answer and sandbox denial controls also pass.

Pure decode throughput is `sum(output_tokens - 1) / sum(last_token - first_token)`.
TTFT is reported separately. Acceptance is accepted draft tokens divided by
proposed draft tokens; mean acceptance length is `1 + accepted / draft_rounds`.
Endpoint decode time divided by draft rounds includes scheduler/sampling work
and is not a CUDA kernel trace or the target-only verifier duration.

## Results

The primary 192-question sample completed before the later long-context OOM.

| Dataset | Correct / 64 | Accuracy | Truncated | Pure decode, tokens/s | Draft acceptance | Mean tokens / round | Mean complete round, ms | Median TTFT, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 61/64 | 95.31% | 1 | 127.262 | 49.13% | 2.9654 | 23.1924 | 0.2080 |
| math500 | 61/64 | 95.31% | 2 | 123.640 | 46.40% | 2.8560 | 23.0695 | 0.2060 |
| humaneval | 56/64 | 87.50% | 8 | 113.903 | 40.84% | 2.6337 | 23.0887 | 0.2092 |

All 196,377 generated tokens, including reasoning and truncated generations,
are retained. The eight HumanEval failures all lack a completed final answer
within 4096 tokens. GSM8K has one truncation and MATH-500 has two.

The three non-truncated mismatches were inspected:

- `gsm8k/187`: model assumes monthly compound interest (106.12); the reference
  expects simple interest (106). The strict mismatch is retained.
- `gsm8k/454`: the question states two people each eat four apples daily for
  thirty days. The model answers 240; the reference inconsistently computes
  `(4+1)*30=150`. The strict score still records a mismatch; this reference
  inconsistency must not be mistaken for a kernel regression.
- `math500/test/intermediate_algebra/662.json`: a real algebra error. The
  model retains an extra factor 13 in one rational denominator and obtains
  four incorrect roots rather than `1 ± sqrt(19)`.

The average complete round remains above 20 ms under natural sampling. These
endpoint measurements do not constitute a new internal kernel trace.

### Explicit 4-GiB budget and quality follow-ups

The explicit budget admits 281,030 tokens (1.07 configured 256K requests).
The no-MTP engine admits 319,297 tokens at the same byte budget because it
does not allocate the draft cache. A complete 261631-input + 513-output MTP
request reaches exactly 262144 tokens without OOM. The same fixed first-eight-per-dataset sample scores 8/8 GSM8K,
8/8 MATH-500, and 6/8 HumanEval; the two HumanEval failures are truncations.

This is **not a bitwise equivalence result**: only 4/24 sampled output-token
sequences and 3/24 acceptance records match the original automatic-KV run.
The fixed greedy 8K emitted sequence is identical, but its acceptance changes
from 177/1340 over 335 rounds to 184/1316 over 329 rounds. KV capacity and
request history differ; this evaluation does not isolate the reason for the
sampled-output/acceptance differences. Do not transfer the 192-example scores
to the 4-GiB contract as though all examples were rerun.

All eleven originally truncated examples receive one saved-prefix continuation.
Three then pass, seven still truncate at the doubled total budget, and one
returns code that fails its tests. These are diagnostic continuations, not
replacement pass@1 samples:

| Original task | Total generated tokens | Outcome |
| --- | ---: | --- |
| `humaneval/HumanEval/145` | 8192 | Still truncated |
| `humaneval/HumanEval/137` | 8192 | Still truncated |
| `humaneval/HumanEval/132` | 8192 | Still truncated |
| `humaneval/HumanEval/76` | 8192 | Still truncated |
| `humaneval/HumanEval/32` | 8192 | Still truncated |
| `humaneval/HumanEval/75` | 6435 | Pass |
| `math500/test/geometry/880.json` | 16384 | Still truncated |
| `humaneval/HumanEval/99` | 6402 | Test failure |
| `gsm8k/1176` | 8192 | Still truncated |
| `math500/test/geometry/965.json` | 11185 | Pass |
| `humaneval/HumanEval/140` | 6709 | Pass |

HumanEval/99 fails because the generated function sets `Decimal` context
precision to zero, which raises `ValueError`. The other unresolved failures
show unfinished reasoning, not merely an answer-extraction issue. There are
also primary MATH responses that naturally finish above 4096 output tokens;
the observed truncations do not establish a universal 4096-token stopping bug.

### Matched 4-GiB target-only control

Both engines complete the same first eight questions per dataset with all
common accelerators enabled. The no-MTP engine includes the Python compile
guard fix described below; native binaries are identical. Each mode uses
normal EOS, so generated text and output lengths can differ. These are observed
pure-decode throughput ratios, not fixed-output request-latency speedups.

| Dataset | MTP4 tokens/s | No-MTP tokens/s | Ratio | MTP4 correct / 8 | No-MTP correct / 8 | Truncations MTP4 / off | MTP4 acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 162.906 | 95.317 | 1.709x | 8/8 | 8/8 | 0 / 0 | 68.83% |
| math500 | 138.098 | 94.570 | 1.460x | 8/8 | 8/8 | 0 / 0 | 52.98% |
| humaneval | 111.977 | 94.740 | 1.182x | 6/8 | 7/8 | 2 / 1 | 38.34% |

HumanEval has one additional truncated response with MTP in this small sample.
The sample does not establish statistical quality equivalence, and none of
its 24 sampled output sequences is bitwise equal between modes. Output totals
(MTP/off) are 1512/1989, 7496/14531, and 13870/10978 respectively. Retain these
differences when interpreting the aggregate throughput ratios.

Three deliberately selected failure controls are separate from the 24-example
subset. With no MTP, GSM8K/187 uses simple interest and passes; GSM8K/454 still
answers 240 against the inconsistent reference 150; the selected MATH algebra
case reaches 8192 tokens without finishing its reasoning. These controls do
not support attributing every primary mismatch to the MTP kernels.

## Long-context checks

The deterministic speed sweep uses exactly 8192, 131072, and 261631 prompt
tokens, with 513 output tokens, temperature 0, and `ignore_eos=True`. The last
case reaches exactly 262144 total tokens. These are length stress tests,
separate from the normal-EOS accuracy results.

| Input + output tokens | MTP4 decode tokens/s | No-MTP decode tokens/s | Ratio | MTP4 TTFT, s | No-MTP TTFT, s | MTP4 acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 + 513 | 72.250 | 97.380 | 0.742x | 2.520 | 3.302 | 13.98% |
| 131072 + 513 | 196.362 | 83.014 | 2.365x | 40.148 | 38.404 | 100.00% |
| 261631 + 513 | 100.357 | 72.978 | 1.375x | 86.780 | 77.555 | 100.00% |

All three forced-length output-token sequences match between modes. The 8K
control recovers the historical approximately 97 tokens/s target-only speed.
MTP loses on that fixture because acceptance is only 13.98%; it does not
guarantee a speedup. TTFT includes first-use shape compilation where applicable
(the no-MTP 8K log records QSA JIT compilation), so these one-request TTFT
values are not a warmed-prefill A/B. Decode and TTFT remain separate.

Natural-EOS retrieval probes place three distinct vault codes near 10%, 50%,
and 90% of documents with exactly 131072 and 261888 prompt tokens. They request
the three codes with thinking disabled, temperature 0, and a 256-token limit.
Both MTP probes return all three codes in the requested order and stop after
26 output tokens. They are narrow retrieval checks, not a comprehensive
long-context benchmark.

| MTP retrieval input tokens | Codes correct | Output tokens | TTFT, s |
| --- | ---: | ---: | ---: |
| 131072 | 3/3 | 26 | 40.927 |
| 261888 | 3/3 | 26 | 92.748 |

No target-only retrieval rerun was needed for this MTP capacity/quality check.

The forced-length speed outputs are not natural answers: the 8K fixture emits
repeated special tokens after EOS, and the long fixtures repeat benchmark
text. The 100% acceptance on these long repeated sequences must not be
generalized to natural tasks; use the normal-EOS dataset table for that purpose.

The [per-case CSV](sm70_flash_next_mtp4_256k_eval_20260927.csv) contains all
264 completed measurements, with separate mode labels for the automatic-KV
primary run, the 4-GiB MTP follow-up, and the 4-GiB no-MTP control. Diagnostic
continuations and selected failures retain separate dataset labels.

## Evidence and failed attempts

The companion CSV retains all 264 measurements with mode labels, fixed case
identities, prompt/output hashes, and separate TTFT/decode/acceptance metrics.
Original source and native-extension identities are recorded above.

The first engine attempt (`mtp256k_a`) passed memory admission and graph
capture, then exited before any dataset evaluation: `LLM` had mutated the
speculation configuration with a `ModelConfig`, and the harness attempted to
serialize that object. The harness now gives `LLM` a deep copy and preserves
an immutable JSON contract. This failed attempt contributes no performance
or accuracy samples. Preparation also required extracting `input_ids` from
the installed tokenizer's `BatchEncoding`. The sandbox allowlist includes
both the owned virtual environment and its base interpreter, verified before
executing generated candidates. None of these harness fixes changes kernels
or model precision.

The first target-only attempt (`nomtp256k_a`) failed before any generation.
With the exact gated RMSNorm accelerator enabled, Dynamo reached a C-backed
cached device-capability query through `RMSNormGated.forward_native` during
full-graph compilation. Fix `6337a96bee` caches the worker's SM70 capability in
the constructor and keeps one-time logging outside tracing. No arithmetic,
native kernel, shape admission, or precision flag changes. The added full-graph
regression fails before the fix; all 25 gated RMSNorm GPU tests pass afterward,
including six dynamic-shape compile cases. Changed-file pre-commit also passes.

The target-only control retains the native extensions, harness, hardware, and
engine settings of the 4-GiB MTP run, with speculation disabled. The Python
compilation guard is a disclosed source difference, so this is not an
identical-source A/B. Current main uses the later device-specific compilation
guard from [PR #704](https://github.com/1CatAI/1Cat-vLLM/pull/704).
