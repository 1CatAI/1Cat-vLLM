<!-- markdownlint-disable MD041 -->

<p align="center">
  <img src="./assets/1cat-llm-logo.png" alt="1Cat-LLM logo" width="960">
</p>

# 1Cat-LLM

## Make Volta Fast Again

### Modern LLM inference for NVIDIA Tesla V100 / SM70

> **1Cat-vLLM is becoming 1Cat-LLM.**
>
> **Broader support. More efficient kernels. More models to choose from.**

We started by making modern models fast on V100. The next chapter carries that work further: broader model and runtime support, more efficient execution, and more choice in how models are packaged and quantized.

New **[GGUF support](#gguf-support)** brings a wider choice of community-quantized checkpoints for supported architectures, with more options for model size, memory use and quality. The new name reflects that direction, with Volta engineering at its foundation.

**The rename is planned.** Repository links, release filenames and installation commands below retain their current names during the transition.

> Recommended checkpoints:
>
> Target: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`
>
> MoE: `RadixArk/Qwen3.8-Flash-Next-NVFP4`
>
> DFlash2 draft: `incoai/Qwen3.8-27B-DFlash2`

<strong>4× Tesla V100 16GB · Qwen3.8-27B-NVFP4 + DFlash2 · ≈260 tok/s</strong>

> Tesla V100 was released in 2017.
>
> Its Tensor Cores did not suddenly become useless.
>
> **The software stack simply stopped being optimized seriously for SM70.**

1Cat-LLM grows out of 1Cat-vLLM, a vLLM engineering fork that treats **NVIDIA Volta / SM70 / Tesla V100** as a first-class optimization target.

We are not satisfied with:

> “The latest model can start on V100.”

Our goal is:

> **Make modern models actually run fast on V100.**

The recorded four-card Tesla V100 16GB demo runs **Qwen3.8-27B-NVFP4 + DFlash2** through 1Cat-vLLM at roughly:

# ≈260 tokens/s

Demo: [4× V100 running Qwen3.8-27B-NVFP4-DFlash2](https://www.bilibili.com/video/BV1kstb6dEaF/)

> **≈260 tok/s is a real-machine demo headline, not a universal fixed decode rate.**
>
> Every benchmark below retains its own hardware, model, context length, batch size, KV dtype, sampling policy, and speculative-decoding contract. Attention TFLOP/s, prefill tok/s, target-only decode tok/s, and speculative decode tok/s are not interchangeable metrics.

**Latest published release: [v1.5.1](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1), October 3, 2026.** The release measurements below use **4× V100-SXM2 32GB**; they are a separate contract from the 16GB demo.

This README also covers merged development work through [main `b379775d`](https://github.com/1CatAI/1Cat-vLLM/commit/b379775dae6dc1cdda69b70777c6992ebebcb6db), October 5, 2026. The newer GGUF and mixed-load changes require a corresponding source/development build; they are not included merely by installing the v1.5.1 release wheel.

---

# 📊 Performance First

SM70 Flash-V100 resolves `--kv-cache-dtype fp8` to E4M3; the 27B release profile explicitly selects `fp8_e4m3`. DFlash2 verification retains FP32 attention state and FP32 logits on its admitted paths.

Keep the Python package and Flash-V100 extension from the same build. The original E4M3 repair requires precision revision 4; current multi-head and request-major routes require newer native capabilities. See [the precision contract](docs/design/sm70_dflash2_fp32_defaults.md) and [concurrent long-context validation](docs/design/sm70_dflash2_long_batch_20260927.md).
Historical E5M2/FP16-partial performance results below keep their original
configuration and are not speed claims for these precision defaults.

## v1.5.1: Measured Release-to-Release Decode

| Input / workload | v1.5.0 | v1.5.1 | Comparison |
|---|---:|---:|---|
| 4K · single request | 103.87 tok/s | **229.34 tok/s** | **+120.80%** |
| 32K · single request | 49.04 tok/s | **201.18 tok/s** | **+310.24%** |
| 128K · single request | No completed baseline | **171.43 tok/s** | No speedup calculated |
| 32K · C4 common decode window | Not measured under the same contract | **445.45 aggregate tok/s** | New-version result only |

Speedup percentages are calculated from the displayed throughput values.

**Single-request contract:** official wheels, the same 4× V100-SXM2-32GB / TP4, QUASAR 27B NVFP4 target, DFlash2 q7, FP16 compute/draft KV, E4M3 target KV, CUDA 12.8 / Torch 2.10, prefix caching, asynchronous scheduling and CUDA Graphs. Both versions use the 1.5.1 launch settings: 262144 maximum context, 8192-token prefill budget, four sequence slots, memory utilization 0.80, KV/Mamba blocks 2048/8192. Sampling is T=1, top-p=.95, top-k=20, seed=0, up to 256 output tokens with normal EOS. Each value is the median of three warm requests after one cold request, from one successful startup per version.

Pure decode excludes TTFT/prefill. At 32K, the old wheel reused 28,672 prefix tokens while the new wheel recomputed the prompt; both decoded with the same 32,768-token input context. This is a comparison under one shared deployment recipe, not each version's best possible tuning. The old 128K request did not finish within the 180-second test timeout.

**C4 contract:** fixed 32K synthetic inputs, 256 forced output tokens per request, T=.7 / top-p=.8 / top-k=20, fixed per-request seeds, one warmup cohort and three measured cohorts from one startup. Throughput counts returned token IDs only while all four requests are decoding. Forced-length timing is separate from natural-EOS quality testing.

Evidence and complete settings: [v1.5.1 release notes](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1). These release checks include output-health spot checks, not a new broad dataset-quality campaign.

## Long-Context Attention: 17.92 → 47.1 → ≈60.8 → ≈71 TFLOP/s

| Stage | Evidence | Useful causal Attention compute | Notes |
|---|---|---:|---|
| Previous production path | v1.2.2-era baseline | **17.92 TFLOP/s** | V100 long-prefix Attention baseline |
| D256 Split-D / N32 | [v1.3.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.3.0) | **46.63–47.1 TFLOP/s** | ≈2.6× over the previous production path |
| GQA-packed wide QK/PV | [PR #286](https://github.com/1CatAI/1Cat-vLLM/pull/286) · historical measurement | **≈60.8 TFLOP/s** | 6 GQA heads packed into wider Tensor-Core GEMMs |
| FP32 QK/PV · wider local head layouts | [PR #666](https://github.com/1CatAI/1Cat-vLLM/pull/666) · Q8192/KV262144 | **69.982–71.111 TFLOP/s** | Single-V100 operator replay for TP1/TP2/TP4 local layouts, including head-group copies |
| Earlier experimental ceiling | [PR #315](https://github.com/1CatAI/1Cat-vLLM/pull/315) | **≈79 TFLOP/s** | Historical candidate; **not a Release/default quality claim** |

From **17.92 → ≈60.8 TFLOP/s**, representative long-context V100 Attention useful compute improved by roughly **3.4×** on the same generation of hardware.

These figures count useful causal QK/PV work, not whole-model TOPS.

The later FP32 QK/PV record is **71.111 / 70.221 / 69.982 TFLOP/s** for TP4 / TP2 / TP1 local head layouts. These are operator measurements on one V100-SXM2-32GB, not multi-GPU model throughput. Earlier 75–77T records predate the complete QK FP32 repair; they are not the current precision baseline. Power and clock policy also matter: the same-source 185W audit recorded 53.816T. See [shape coverage](docs/design/sm70_tp_shape_coverage.md) and [quality limits](docs/design/sm70_tp_quality_audit.md).

---

# 🚀 Real Model Benchmarks

The table below preserves **complete-model / API / pure-decode / speculative-decode** measurements from their original PRs. They are historical evidence under separate contracts; the current published-wheel comparison is above.

| Model | Hardware / Runtime | Workload | Measured result | Evidence / Status |
|---|---|---|---:|---|
| **Qwen3.6-27B-AWQ + MTP4** | 4× V100 · TP4 · E5M2 KV · Flash-V100 · CUDA Graph | 64K decode | **100.564 tok/s** | v1.2.2 Release · AL 4.981 / 99.52% |
| **Qwen3.6-27B-AWQ + MTP4** | same | 128K decode | **85.258 tok/s** | v1.2.2 Release · +87.64% vs no-MTP |
| **Qwen3.6-27B-AWQ + MTP4** | same · max 256K | 261,888 context decode | **49.772 tok/s** | v1.2.2 Release · AL 5.000 / 100% |
| **Qwen3.6-35B-A3B NVFP4** | 4× V100 · TP4 · mixed FP8 + W4A16_NVFP4 | 4096 / 1024 · no-MTP | **116.99 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) |
| **Qwen3.6-35B-A3B NVFP4 + MTP4** | same | matched MTP4 run | **174.76 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) · 1.49× no-MTP |
| **Qwen3.8-27B-NVFP4** | 4× V100 · TP4 · E4M3 KV · full CUDA Graph · no-MTP | exact 128K decode | **61.834 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · measured |
| **Qwen3.8-27B-NVFP4** | same | exact 256K decode | **50.376 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · measured, not projected |
| **Qwen3.8-27B-FP8** | 4× V100 · TP4 · E5M2 KV · no-MTP | 128K decode | **50.68 tok/s** | [#212](https://github.com/1CatAI/1Cat-vLLM/pull/212) release-path sweep |
| **Qwen3.8-27B-FP8** | same | 256K decode | **41.11 tok/s** | [#212](https://github.com/1CatAI/1Cat-vLLM/pull/212) release-path sweep |
| **Qwen3.8 Flash-Next-NVFP4** | 4× V100 · TP4 · V2 · full CUDA Graph · no-MTP | 8K / 512 pure decode | **80.732 tok/s** | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · quality-audited |
| **Qwen3.8 Flash-Next-NVFP4 + MTP4** | 4× V100 · TP4 · V2 | final cold-JIT gate | **138.26 tok/s** | [#389](https://github.com/1CatAI/1Cat-vLLM/pull/389) · AL 4.943 / 98.57% |
| **Qwen3.8-27B-NVFP4 + DFlash2** | 4× V100 · TP4 · production API | historical web prompt · 512 output | **206.06 tok/s streaming decode** | [#422](https://github.com/1CatAI/1Cat-vLLM/pull/422) · 17.463 ms/round · 3.599 emitted/round |
| **Qwen3.8-27B-NVFP4 + DFlash2** | 4× V100 · TP4 · practical API | MBPP item 28 · natural EOS | **251.60 tok/s** | [#288](https://github.com/1CatAI/1Cat-vLLM/pull/288) · AL 4.686 · EvalPlus 1/1 |
| **Qwen3.8 DFlash2 + adaptive lookup q16** | 4× V100 · TP4 · opt-in lookup augmentation | repeated-context sample | **316.27 tok/s** | [#366](https://github.com/1CatAI/1Cat-vLLM/pull/366) · 3.162 ms TPOT · special opt-in contract |
| **DeepSeek-V4-Flash** | **8× V100** · TP8 · FP8 dense + MXFP4 experts · CUDA Graph · no-spec | 1024 / 256 | **15.357 ms TPOT ≈ 65.1 tok/s** | [#181](https://github.com/1CatAI/1Cat-vLLM/pull/181) · accepted no-MTP baseline |
| **DeepSeek-V4-Flash** | **8× V100** · PP2×TP4 · no-DSpark | combined quality-checked endpoint | **73.613–73.646 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) |
| **DeepSeek-V4-Flash** | same PP2×TP4 strict control | dataset-quality pair | **73.539 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) · GSM8K 64/64 · HumanEval 29/32 |
| **GLM-5.3-Flash-NVFP4** | 8× V100 · TP4/PP2 · E4M3 KV · no-MTP | 1K / 256 decode | **53.016 tok/s** | [#402](https://github.com/1CatAI/1Cat-vLLM/pull/402) · merged historical quality audit |

---

# 🧪 Dataset / Quality × Throughput Benchmarks

Raw `tok/s` alone can turn optimization into a benchmark game. 1Cat-LLM therefore records **real model throughput, dataset score, natural-stop health, output validity, and speculative acceptance** together.

The retained DFlash2 coding/PPL gates below use their recorded historical configuration, including E5M2 KV where specified. They do not stand in for a new quality evaluation of the E4M3 release wheel, current main, or GGUF models.

## Qwen3.8-27B-NVFP4 + DFlash2 — Practical 16K coding gate

Contract:

- 4× V100, TP4
- NVFP4 target
- official BF16 DFlash2 drafter
- FP8 E5M2 target KV
- FlashAttention-V100
- full CUDA Graph
- prefix cache
- Mamba align
- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `xhigh` reasoning
- 16K natural-EOS output cap
- three predeclared sampling seeds

| Dataset | Samples | Base score | Plus score | Natural stop | Aggregate output throughput | Mean steady decode | Acceptance pooled / request |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MBPP / EvalPlus** | 96 · 93 scored | **89/93** | **80/93** | 95/96 | **213.539 tok/s** | **236.902 tok/s** | 4.061 / 4.318 |
| **HumanEval / EvalPlus** | 96 | **94/96** | **92/96** | 91/96 | **208.978 tok/s** | **245.645 tok/s** | 3.972 / 4.476 |

Evidence: [PR #346](https://github.com/1CatAI/1Cat-vLLM/pull/346) and `docs/design/sm70_dflash2_quality_audit.md`.

The first seed matches the historical target-only / no-DFlash request-seed contract. Across MBPP + HumanEval, both routes score:

```text
Base : 62 / 63
Plus : 59 / 63
```

So the 200+ tok/s speculative path is not obtained by removing the task-quality gate.

### About length-capped failures

Some coding failures are caused by **long reasoning exhausting the 16K output budget**, rather than by an invalid final solution.

Across the retained MBPP + HumanEval campaign, **6 of 192 outputs reached the 16K cap**, and **3 of those were still extractable and correct**.

For this reason, the README separates:

- executable task score,
- natural-stop rate,
- output-cap failures,
- and throughput.

It does not treat every capped sample as proof of a model-capability regression.

## Optional precise-coding profile

Using the same DFlash2 engine and the same middle seed, only client-side sampling is changed to the model's precise-coding profile:

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
```

| Dataset | Temperature 1.0 | Temperature 0.6 |
|---|---:|---:|
| MBPP | Base 27/31 · Plus 24/31 | **Base 29/31 · Plus 27/31** |
| HumanEval | Base 31/32 · Plus 31/32 | **Base 31/32 · Plus 31/32** |

Across 80 requests:

```text
Mean steady decode:
233.187 → 244.520 tok/s

Output-token / decode-time throughput:
195.817 → 201.852 tok/s

Request-mean acceptance:
4.27345 → 4.51356
```

Natural stops move from 72/80 to 70/80, so this remains an **optional precise-coding profile**, not a forced global default.

---

## Other full-model quality gates

| Model / Route | Dataset / Quality | Real throughput under the recorded contract | Status |
|---|---|---:|---|
| **Qwen3.8 Flash-Next-NVFP4 · no-MTP** | GSM8K **15/16 raw · 15/16 strict** · 16/16 natural stop | **80.935 tok/s weighted pure decode** | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · merged / quality-audited |
| **Qwen3.8 Flash-Next-NVFP4 · MTP4** | HumanEval8 **8/8 semantic executions** | **150.17 tok/s weighted pure decode** | [#398](https://github.com/1CatAI/1Cat-vLLM/pull/398) · merged; bounded historical gate |
| **Qwen3.6-35B-A3B NVFP4 + MTP4** | GSM8K **122/128 (95.3125%)** · 0 invalid · 0 repetitive | matched MTP run **174.76 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) · merged |
| **Qwen3.6-35B-A3B NVFP4 + MTP4** | ShareGPT16 final-SHA workload | **120.096 tok/s pure decode** · **97.678 E2E output tok/s** · 241.973 prefill tok/s | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) · merged |
| **DeepSeek-V4-Flash · PP2×TP4** | GSM8K **64/64** · HumanEval **29/32** · LongBench **44.740** | **73.539 tok/s median** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) · strict quality control |
| **DeepSeek-V4-Flash · PP2×TP4** | Combined route endpoint: GSM8K **62/64** · 0 invalid · coherent output | **73.613–73.646 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) |
| **GLM-5.3-Flash-NVFP4 · no-MTP** | Max reasoning: **6/8** tasks finish within 4096 output tokens; targeted low-reasoning code rerun **2/2** AST + execution | **53.016 tok/s decode** · **266.040 tok/s 1K prefill** | [#402](https://github.com/1CatAI/1Cat-vLLM/pull/402) · merged; historical quality matrix |

---

## Tool Calling / Structured Output

Modern inference serving must do more than generate prose. DFlash2 + adaptive lookup was also tested against tool and structured-output workloads.

| Gate | Result |
|---|---:|
| **BFCL** | **29/32** |
| **ToolACE** | **12/12** |
| **NexusRaven** | **13/16** |
| **Strict JSON Schema** | **7/8** |
| Structured B1 | **12/12** |
| Structured B4 | **12/12** |
| Long prefix-state isolation | **5/5** |

These quality results match the target-only / q7 reference in the retained audit.

Runtime examples from the same development line:

```text
Ordinary q7 → adaptive q8:
168.52 → 170.98 tok/s

Repeated-context q16:
316.27 tok/s
3.162 ms TPOT
```

The q16 number is a **special repeated-context lookup-hit contract**. It is not presented as the expected throughput of every tool-calling request.

Evidence: [PR #366](https://github.com/1CatAI/1Cat-vLLM/pull/366).

---

## Distribution / PPL gate

DFlash2 is also checked at the target-distribution level.

Eight fixed WikiText 2,048-token segments, **16,376 scored prompt tokens**:

```text
Target-only PPL : 5.4993116
DFlash2 PPL     : 5.4993622
Absolute delta  : +0.0000506
Relative delta  : +0.00092%
Max segment Δ   : 0.0062143
```

The purpose of this gate is to detect cases where benchmark answers still look acceptable while speculative verification has systematically shifted the target distribution.

---

# ⚡ Why DFlash2 Can Reach 200+ tok/s

The repository contains multiple **real full-model DFlash2** throughput records:

- production web prompt: **206.06 tok/s streaming decode**, 512 output tokens, 17.463 ms/engine round;
- high-acceptance MBPP request: **251.60 tok/s**, acceptance length **4.686**, 328-token natural EOS, EvalPlus Base/Plus **1/1**;
- adaptive lookup q16 repeated-context workload: **316.27 tok/s**, explicitly a special opt-in repeated-context contract;
- the README headline remains **≈260 tok/s** from the real-machine demo.

> **206, 251, 260, and 316 tok/s are not the same benchmark.**
>
> DFlash2 throughput depends strongly on acceptance length, prompt repetition, q8/q16 verification width, context length, and task type.

---

# 📏 Long Context Means More Than “It Fits in 256K”

For Qwen3.8-27B-NVFP4, [PR #285](https://github.com/1CatAI/1Cat-vLLM/pull/285) reports real TP4 full-model long-context decode:

```text
128K : 40.561 → 61.834 tok/s
256K : 27.456 → 50.376 tok/s
```

**50.376 tok/s at 256K is the measured endpoint result.**

The PR also contains a decomposition-based projection of 52.216 tok/s, but this README intentionally uses the measured 50.376 tok/s result.

DeepSeek-V4 should also be judged by later full-model results rather than an early bring-up checkpoint:

```text
TP8 no-spec:
15.357 ms TPOT ≈ 65.1 tok/s

PP2×TP4 quality-checked endpoint:
73.613–73.646 tok/s
```

---

# 🔬 Selected Merged PR Benchmarks

| Area | PR / Contract | Control | 1Cat result | Gain |
|---|---|---:|---:|---:|
| D256 long-prefill Attention | [#198](https://github.com/1CatAI/1Cat-vLLM/pull/198) · Q4096/KV64K · Hq6/Hkv1/D256 | 87.6001 ms | **50.4504 ms** | **1.74×** |
| D256 long-prefill Attention | #198 · Q4096/KV8K | 11.1255 ms | **5.0542 ms** | **2.20×** |
| 128-bit E5M2 XQA load | [#268](https://github.com/1CatAI/1Cat-vLLM/pull/268) · B16/17.8K operator | 0.743424 ms | **0.602112 ms** | **1.235×** |
| 128-bit E5M2 XQA load | #268 · ragged B16/32K operator | 1.171296 ms | **0.925808 ms** | **1.265×** |
| Batched long decode | #268 · B16/16K full-model pure decode | 529.071 tok/s | **570.982 tok/s** | **+7.92%** |
| Long-context decode routing | [#206](https://github.com/1CatAI/1Cat-vLLM/pull/206) · 128K TP4 | 40.8208 tok/s | **48.5431 tok/s** | **+18.92%** |
| Long-context decode routing | #206 · 180K TP4 | 36.1387 tok/s | **42.5501 tok/s** | **+17.74%** |
| E4M3 XQA long decode | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · exact 128K | 40.561 tok/s | **61.834 tok/s** | **+52.45%** |
| E4M3 XQA long decode | #285 · exact 256K | 27.456 tok/s | **50.376 tok/s** | **+83.48%** |
| Grouped QSA Page4 | [#387](https://github.com/1CatAI/1Cat-vLLM/pull/387) · per-layer/rank | 55.151 ms | **9.632 ms** | **5.518×** |
| QSA full-model prefill | #387 · 64K | 4,446.64 tok/s | **5,777.43 tok/s** | **+29.93%** |
| Indexed NVFP4 MoE prefill | [#390](https://github.com/1CatAI/1Cat-vLLM/pull/390) · 64K | 5,777.43 tok/s | **6,241.48 tok/s** | **+8.03%** |
| Exact target-only decode | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · 8K/512 · no-MTP | 65.864 tok/s | **80.732 tok/s** | **+22.57%** |
| DFlash2 NVFP4 prefill | [#417](https://github.com/1CatAI/1Cat-vLLM/pull/417) · 32K/64K | retained pre-closure | **4069.25 / 3566.94 prefill tok/s** | **+30.1% / +37.7%** |
| DeepSeek-V4 sparse MLA | [#163](https://github.com/1CatAI/1Cat-vLLM/pull/163) · sparse MLA GPU service | 46.920 ms/token | **4.392 ms/token** | **-90.64%** |
| DeepSeek-V4 TP8 no-spec decode | [#181](https://github.com/1CatAI/1Cat-vLLM/pull/181) · 8×V100 · 1024/256 | 19.342 ms TPOT false-4K graph | **15.357 ms TPOT ≈65.1 tok/s** | **~20.6% lower TPOT** |
| DeepSeek-V4 PP2×TP4 full model | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) · 8×V100 · no-DSpark | — | **73.613–73.646 tok/s** | quality-checked endpoint |
| DFlash2 concurrent long decode | [#697](https://github.com/1CatAI/1Cat-vLLM/pull/697) · 32K/256 · C4 / C8 | 326.832 / 492.025 tok/s | **429.134 / 624.863 tok/s** | **+31.30% / +27.00%** |
| Flash-Next prefix-cache prefill | [#754](https://github.com/1CatAI/1Cat-vLLM/pull/754) · 32K · zero prefix hits | 3,046 tok/s | **5,068 tok/s** | **+66.38%** |
| Flash-Next NVFP4 MTP4 defaults | [#796](https://github.com/1CatAI/1Cat-vLLM/pull/796) · 31,744/256 · C1 | 79.832 tok/s | **92.199 tok/s** | **+15.49%** |
| Original-byte IQ3_XXS/IQ3_S pair | [#966](https://github.com/1CatAI/1Cat-vLLM/pull/966) · M8/N4352/K5120 · operator only | 86.016 μs | **57.344 μs** | **1.50×** |

The added rows retain their own baselines. #697 uses 4× V100-SXM2-32GB / TP4, NVFP4 + DFlash2 q7, E4M3 target KV and FP16 draft KV, with three warm simultaneous-request decode windows; acceptance also changes. #754 uses Flash-Next NVFP4 / MTP4, FP16 KV, prefix caching enabled and matched no-hit requests. #796 uses the same FP16-KV MTP4 recipe in both arms, with **21/21 exact paired quality outputs**; its C4 change is only +1.05%, and additional packed weights cost about 1.25 GiB/rank. #966 is a fixed-clock single-GPU cold-L2 operator comparison, not a whole-model speedup. These gains cannot be added together or substituted for the release-wheel comparison.

---

# 🧠 128-bit Loads: Not a Cosmetic Vectorization Change

[PR #268](https://github.com/1CatAI/1Cat-vLLM/pull/268) does more than replace a narrow type with a wider C++ type.

Inside real paged-KV partitions, it:

- reuses the Page ID;
- merges two `half8` conversion groups;
- issues one aligned 128-bit cache load;
- keeps softmax, PV, partition boundaries, and reduction order unchanged.

NCU evidence:

```text
L1 global-load requests:
656,443 → 383,814
-41.53%

Executed warp instructions:
97,998,831 → 83,583,696
-14.71%

Long-scoreboard stall:
39.14% → 30.10%

Eligible warps / scheduler:
0.55 → 0.65

B16 / 17.8K kernel duration:
648.352 → 499.520 μs
-22.95%
```

DRAM bytes stay nearly unchanged.

The gain comes from **fewer fragmented loads, lower address/dependency pressure, and a more continuous operand feed**, not from magically reducing the model size.

---

# ✅ Correctness / Quality Gates

1Cat-LLM does not treat a good-looking TPS number as sufficient evidence.

Representative gates include:

- [#198](https://github.com/1CatAI/1Cat-vLLM/pull/198): 64K full-model A/B/A 64-token IDs, text, and SHA256 match; random paged-KV, gathered-dense, and Split-KV3 have separate numerical gates.
- [#268](https://github.com/1CatAI/1Cat-vLLM/pull/268): uniform/ragged B4/B8/B12/B16, page256/page800, 12K–32K operator A/B is **bitwise exact**.
- [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285): 128K and 256K E4M3 XQA endpoints both emit the complete 64 tokens and preserve their matching control streams.
- [#346](https://github.com/1CatAI/1Cat-vLLM/pull/346): structured API **24/24**, long alternating-prefix state **5/5**, multi-seed MBPP/HumanEval quality gates, and target-only/DFlash2 WikiText PPL **5.4993116 / 5.4993622**.
- [#387](https://github.com/1CatAI/1Cat-vLLM/pull/387): grouped QSA replay is deterministic; arithmetic, Chinese-language, and performance-case token hashes match the retained baseline.
- [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415): GSM8K **15/16 strict**, natural stop **16/16**, zero capped outputs, zero structurally invalid outputs.
- [#427](https://github.com/1CatAI/1Cat-vLLM/pull/427): 1.5.0 RC isolated install passes `/v1/models`, `/metrics`, normal chat, streaming/non-streaming tool calls, JSON Schema, and repeated-prefix checks; a 10,017-token prefix moves from **2.642 s cold → 0.164 s cached**.

---

# 🔥 FlashAttention-V100

## We are not just “making FlashAttention compile on V100.”

## We are rebuilding the dataflow for Volta

FlashAttention is fundamentally an **IO and scheduling problem**:

- reduce HBM round trips;
- keep Q/K/V and intermediate state on-chip as long as possible;
- increase reuse;
- reduce materialization;
- reduce barriers;
- continuously feed Tensor Cores.

Modern FlashAttention implementations are designed around Ampere, Hopper, and newer GPUs.

Tesla V100 is SM70.

It does not have:

- Ampere `cp.async`;
- Turing/Ampere-style `ldmatrix` data paths available to newer Tensor-Core kernels;
- Hopper TMA;
- native FP8 Tensor Cores;
- Blackwell FP4 Tensor Cores.

A direct compatibility port may run, but it often leaves the GPU underfed.

That is why 1Cat-LLM rebuilds the execution path around the capabilities Volta actually has.

---

# ⚙️ Software-Reconstructed Async / Matrix Feed on SM70

We do **not** claim that V100 executes `cp.async` or `ldmatrix`.

Instead, 1Cat-LLM reconstructs the **design goals behind those mechanisms** using:

```text
LDG
STS
LDS
register prefetch
double buffering
Shared Memory swizzle
explicit HMMA fragment mapping
cross-tile / cross-stage software pipelining
```

The objective is the same:

```text
overlap memory movement with compute
        ↓
increase on-chip reuse
        ↓
shorten dependency chains
        ↓
reduce barriers and replay
        ↓
keep HMMA continuously fed
```

Representative techniques include:

- register prefetch and double buffering;
- overlap next-K tile loading with current QK compute;
- pre-stage PV operands while HMMA is still executing;
- phase-swizzled Shared Memory layouts;
- 128-bit vectorized access;
- explicit QK/TN and PV/TT HMMA fragment ownership;
- software scheduling across tile and stage boundaries.

> **We do not emulate a `cp.async` instruction.**
>
> **We rebuild the memory/computation overlap that modern hardware instructions are designed to provide.**

---

# Layer 1 — Move KV Cache Correctly, Wide, and Once

Paged KV maps logical tokens onto physical pages.

A naive SM70 path repeatedly:

```text
load Page ID
calculate address
load narrow FP8 fragment
convert
repeat
```

That wastes cycles on address work, dependency waits, and scalar memory traffic.

The 128-bit XQA work in [#268](https://github.com/1CatAI/1Cat-vLLM/pull/268) reuses page metadata and performs paired aligned loads.

Representative full-model batch results include:

```text
B16 / 16K:
529.071 → 570.982 tok/s
+7.92%
```

The corresponding operator gain reaches roughly **21%–26.5%** on representative long-context XQA shapes.

---

## Layer 2 — Rewrite D=256 Attention as a Volta-Native Pipeline

After reducing data-movement overhead, the Attention body itself is restructured.

Key components include:

## D256 Split-D

Split D=256 into four D64 slices.

Paired warps share QK probability work while increasing PV parallelism without recomputing the same QK work.

## N32 Online Softmax

Retain causal online-softmax and FP32 accumulation contracts without materializing a full score matrix.

## K-stage Ping-Pong

Alternate K/D64 panels across Shared Memory stages to reduce barrier and wait pressure.

## Split-KV3

Split long-prefix KV work into three partitions where useful, then merge FP32 partial state.

## GQA Multi-Head Packing

Pack six GQA query heads into wider Tensor-Core work.

## Wide QK / PV

Turn many fragmented small Tensor-Core operations into larger, more regular QK/PV GEMM-style work.

## Prefix / Causal-Tail Separation

Schedule the fully visible long prefix separately from the exact causal tail and merge the online-softmax state.

This optimization family evolved through [PR #198](https://github.com/1CatAI/1Cat-vLLM/pull/198), later D256 / Split-KV3 work, [v1.3.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.3.0), and [PR #286](https://github.com/1CatAI/1Cat-vLLM/pull/286).

The result:

```text
17.92 TFLOP/s
    ↓
46.63–47.1 TFLOP/s
    ↓
≈60.8 TFLOP/s
```

Same GPU generation. Same Tensor Cores.

The software stopped wasting them.

> ≈79 TFLOP/s is retained as an experimental research ceiling, not as the default production quality claim.

Later work extends this dataflow to **Q8000/Q8192, multiple requests and TP1/TP2/TP4 local head layouts**, with FP32 QK/PV accumulation. The qualified full-FP32 operator record reaches roughly **70–71 TFLOP/s** under the recorded conditions. Subsequent fixes cover short-chunk causal visibility, missed score maxima, overflowing score tiles and tail-intermediate range: [#820](https://github.com/1CatAI/1Cat-vLLM/pull/820), [#850](https://github.com/1CatAI/1Cat-vLLM/pull/850), [#851](https://github.com/1CatAI/1Cat-vLLM/pull/851), [#875](https://github.com/1CatAI/1Cat-vLLM/pull/875). Historical peak numbers do not replace these numerical checks.

---

# Layer 3 — Sparse Attention Must Also Be Native to V100

Qwen3.8 Flash Next QSA requires more than “select fewer tokens.”

The runtime must also handle:

- sparse block selection;
- physical-page mapping;
- Page4 K/V reuse;
- exact per-row masks;
- final QK/PV computation.

[PR #387](https://github.com/1CatAI/1Cat-vLLM/pull/387) groups eight adjacent query rows so overlapping Page4 K/V blocks are loaded once while preserving an exact 4-bit mask per row.

It then uses Volta WMMA directly for QK and PV.

Representative results:

```text
Old QSA path:
55.151 ms/layer/rank

Grouped Page4:
9.632 ms/layer/rank
+0.362 ms planner

Attention speedup:
5.518×
```

Full-model pure-prefill improvements:

```text
32K  : +32.36%
64K  : +29.93%
131K : +32.69%
```

---

# 🧩 Profiling-Driven Optimization

1Cat-LLM does not stop when one kernel becomes fast.

When QSA was accelerated, profiling showed the next hotspot had moved into NVFP4 MoE prefill.

[PR #390](https://github.com/1CatAI/1Cat-vLLM/pull/390) then removed the `[tokens × topK, hidden]` input-expansion bottleneck by using indexed W13 execution.

Representative results:

```text
8K operator chain:
6.026752 → 4.235264 ms
1.423×

Full-model pure prefill:
32K  : 5998.65 → 6507.10 tok/s
64K  : 5777.43 → 6241.48 tok/s
131K : 5450.92 → 5871.47 tok/s
```

[PR #393](https://github.com/1CatAI/1Cat-vLLM/pull/393) then fused exact FP16 SwiGLU, split the N320 W13 tail into N256+N64, and removed wasted tail-tile work.

This is the optimization philosophy of the project:

> **Profile the real model, move the bottleneck, profile again.**

---

# 🎯 Target-Only Decode Before Speculative Decoding

Before relying on DFlash2 or MTP, the target model itself must be fast.

[PR #415](https://github.com/1CatAI/1Cat-vLLM/pull/415) reports Qwen3.8-Flash-Next-NVFP4 on 4× V100:

```text
8K input / 512 output
no MTP
full CUDA Graph

Control:
65.864 tok/s
15.183 ms TPOT

Candidate:
80.732 tok/s
12.387 ms TPOT
```

That is **target-only throughput**.

The route also passes:

```text
GSM8K: 15/16 strict
Natural stop: 16/16
Weighted natural-output decode: 80.935 tok/s
```

---

# ⚡ DFlash2 on SM70

Traditional autoregressive decode requires one target-model pass per emitted token.

DFlash2 changes the execution model.

A block-diffusion draft model proposes several future tokens and the target verifies them together.

The effective service loop becomes:

```text
draft several candidates
        ↓
target verifies a block
        ↓
accept multiple tokens
        ↓
advance by more than one token per target round
```

For Qwen3.8 DFlash2, the release-oriented SM70 stack also optimizes:

- draft Attention;
- selector;
- grouped verifier;
- GDN metadata;
- sparse rejection;
- NVFP4/QPN paths;
- sampling;
- CUDA Graph;
- prefix state;
- Mamba align;
- tool / structured-output state.

The draft Attention itself uses:

```text
FLASH_ATTN_V100
```

rather than falling back to an unrelated generic path.

---

# DFlash2 Long-Context Decay

Long context must not make speculative verification cost grow unnecessarily.

[PR #328](https://github.com/1CatAI/1Cat-vLLM/pull/328) changes the non-anchored paged-prefill loop so it begins at the first sliding-window tile actually used by the draft.

At 256K:

```text
Draft attention:
0.422912 → 0.246784 ms/layer

Five-layer projection:
2.114560 → 1.233920 ms
```

Candidate medians:

```text
32K  : 0.252928 ms
128K : 0.243712 ms
256K : 0.246784 ms
```

The post-32K context slope is nearly eliminated for that draft-attention component.

---

# 🧵 Concurrency Must Survive New Prefill

Four decoders running alone are one workload. Four decoders sharing the GPU with new long prompts are another.

The newer SM70 path combines batched FP4/FP8 projections, joint draft attention, shared GDN metadata, long-context verification and sampling workspace reuse. [#697](https://github.com/1CatAI/1Cat-vLLM/pull/697), [#706](https://github.com/1CatAI/1Cat-vLLM/pull/706) and [#708](https://github.com/1CatAI/1Cat-vLLM/pull/708) retain separate operator and full-model comparisons.

[PR #860](https://github.com/1CatAI/1Cat-vLLM/pull/860) adds an adaptive mixed-prefill budget while preserving resident decode capacity and hybrid prefix replay. On its recorded **27B QUASAR NVFP4 + DFlash2 / 4× V100-SXM2-32GB / TP4 / E4M3 / 256K-capacity** deployment, two sustained decoders share the GPU with two incoming 32K prompts:

| Metric during incoming prefill | Control | Candidate |
|---|---:|---:|
| Resident output | 6.88 tok/s | **30.34 tok/s** |
| Longest output-update gap | 2.873 s | **0.310 s** |
| First / second incoming TTFT | 14.287 / 20.631 s | **17.994 / 35.491 s** |

These are medians of two warm measured runs after warmup, using the PR's two-NVLink-pair topology. Resident EOS is disabled to sustain the load; incoming requests complete naturally. **Shorter decoder stalls trade against longer incoming TTFT.** The 250 ms step target is a soft scheduler target, not a latency guarantee. This is merged development work after the v1.5.1 artifact.

Recent work also parallelizes compatible small-query draft attention ([#944](https://github.com/1CatAI/1Cat-vLLM/pull/944)) and reduces synchronization in TP4 all-reduce plus RMSNorm ([#957](https://github.com/1CatAI/1Cat-vLLM/pull/957)). Each keeps its measured precision, shape and topology limits; their small single-request gains are not advertised as a universal concurrency multiplier.

---

# 🔢 Quantization / Operator Stack

V100 predates many of the formats used by current LLM checkpoints.

1Cat-LLM therefore treats quantization support as an **operator-design problem**, not only a loader problem.

Current SM70 work includes:

- AWQ / W4A16;
- TurboMind SM70 kernels;
- compressed-tensors;
- FP8 E4M3 / E5M2 KV storage;
- ModelOpt NVFP4;
- MXFP4;
- standalone GGUF metadata, tokenizer and architecture adapters;
- original-block GGUF affine / codebook operators with capability-based fallback;
- Quark W4A16 INT4 / UINT4;
- QPN8;
- QPN4;
- QPN2;
- grouped MoE;
- exact-shape decode GEMV;
- custom SM70 sampling paths.

The goal is not:

> “The dtype parses.”

The goal is:

> **The quantized format becomes a usable high-performance serving path on Volta.**

---

<a id="gguf-support"></a>

# 🗂️ GGUF: Keep the Checkpoint, Rebuild the Execution

More model choice starts with more usable checkpoints.

GGUF brings community quantizations into the supported model paths, giving users more ways to balance model size, memory use and quality. 1Cat-LLM connects that choice to native kernels and the serving stack, with coverage and measurements recorded below.

Current main can read standalone GGUF metadata, Qwen BPE tokenizers and embedded chat templates. Native adapters cover Qwen3.5 dense, Qwen3.5 MoE, Qwen4Exp / Flash-Next and DFlash2 drafts, with tensor-parallel storage and mixed tensor types handled explicitly: [#809](https://github.com/1CatAI/1Cat-vLLM/pull/809), [#876](https://github.com/1CatAI/1Cat-vLLM/pull/876), [#888](https://github.com/1CatAI/1Cat-vLLM/pull/888), [#895](https://github.com/1CatAI/1Cat-vLLM/pull/895).

The operator work keeps original quantization information through layout preparation, selects measured TurboMind SM70 routes and retains packaged native/reference fallbacks. Compatible IQ3_S, IQ4_XS and IQ3_XXS gate/up pairs can share activations without replacing the checkpoint with a newly quantized model.

[PR #966](https://github.com/1CatAI/1Cat-vLLM/pull/966) admits seven additional IQ3_XXS/IQ3_S mixed pairs at M8/N4352/K5120, with FP32 dot products and split-K reduction. The combined mixed-pair coverage is **18/40 layers and 48.463% of mixed-pair source bytes** in that measured model. Other shapes retain canonical dispatch. The Q4_K records in [#968](https://github.com/1CatAI/1Cat-vLLM/pull/968) prepare lossless storage only; they do not add a new accelerated model route.

One retained **Flash-Next GSQ-RCO IQ3_S + FP16 MTP4** development composition reports:

| Workload | Full round | Measured decode |
|---|---:|---:|
| C1 · 8192 input / 256 output | **26.072 ms** | **92.267 tok/s** |
| C4 · 128 input / 1024 output per request | **52.059 ms** | **280.441 aggregate tok/s** |

Evidence: [#940](https://github.com/1CatAI/1Cat-vLLM/pull/940) and [the complete workload record](docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md). These use a named development build on **4× V100-SXM2-32GB / TP4**, direct NVLink ring edges, FP16 activation/KV, FP32 SSM state and FULL decode graphs. Fixed-length greedy timing ignores EOS and is repeated three times; four separate natural-EOS prompts match the original outputs. Longer timing trajectories differ across versions, and the **15–17 ms round goal remains open**.

Adapter support, kernel admission and model-quality qualification are separate milestones. The dense and DFlash2 adapters have recorded loading/generation checks; the MoE adapter PR alone does not establish broad 35B quality or throughput. Consult the [GGUF controls](docs/design/gguf_v100_control.md) and each model's recorded gates. These additions postdate the v1.5.1 release wheel.

---

# Qwen3.6-35B-A3B NVFP4

[PR #270](https://github.com/1CatAI/1Cat-vLLM/pull/270) adds an exact SM70 route for mixed ModelOpt NVFP4 checkpoints.

Highlights:

- FP8 dense projections;
- W4A16_NVFP4 routed/shared experts;
- grouped TurboMind MoE;
- duplicate expert-slot preservation;
- mixed-precision GDN routing;
- MTP cold-start warmup.

Matched no-MTP:

```text
AWQ:
prefill 0.3813 s
decode 113.71 tok/s

NVFP4:
prefill 0.4216 s
decode 116.99 tok/s
```

MTP4:

```text
174.76 tok/s
1.49× NVFP4 no-MTP
```

Quality:

```text
GSM8K:
122/128
95.3125%

invalid outputs:
0

repetitive records:
0
```

---

# DeepSeek-V4 on V100

DeepSeek-V4 work extends beyond a single sparse-attention kernel.

The SM70 stack includes work around:

- sparse MLA;
- FP8 dense projections;
- MXFP4 experts;
- grouped MoE;
- Indexer;
- KPool;
- Q normalization / RoPE / KV insertion;
- custom TP4 all-reduce;
- PP2×TP4 execution;
- exact GEMV hot paths.

Representative results:

```text
TP8 no-spec:
≈65.1 tok/s

PP2×TP4 strict quality control:
73.539 tok/s

PP2×TP4 combined endpoint:
73.613–73.646 tok/s
```

Strict quality control:

```text
GSM8K    : 64/64
HumanEval: 29/32
LongBench: 44.740
```

---

# GLM-5.3 on V100

The current GLM-5.3 SM70 path uses:

- ModelOpt NVFP4 MoE;
- FP16 non-expert weights;
- FP8 E4M3 KV;
- TP4 / PP2;
- sparse MLA;
- exact KDA GEMV;
- fused KDA f/g;
- mHC;
- custom all-reduce;
- full decode CUDA Graph.

Retained stability result:

```text
Decode:
53.013085
53.018516
53.017527 tok/s

Mean:
53.016376 tok/s

Mean TPOT:
18.862097 ms
```

1K prefill:

```text
266.039984 tok/s
```

The quality audit also records a reasoning-mode caveat: Max reasoning can exhaust the output budget on concise code tasks, while the targeted low-reasoning rerun completes and passes both AST and external execution checks.

The figures above remain the historical no-MTP audit. Later merged work integrates GLM DFlash2 with the current scheduler, TP8 verifier and PP2 KV-capacity handling: [#501](https://github.com/1CatAI/1Cat-vLLM/pull/501), [#503](https://github.com/1CatAI/1Cat-vLLM/pull/503). Those changes do not turn the 53.016 tok/s no-MTP record into a DFlash2 benchmark.

---

# 🎬 Native Image / Video Workflows

The repository also contains native **MiniMax H3 audio/video generation** and **Z-Image Turbo / Base image jobs**. H3 work covers SM70 attention, tensor-parallel execution, weight residency, export and actual lifecycle/denoise progress. Z-Image adds persistent asynchronous jobs, synchronous compatibility, cancellation and restart handling: [#557](https://github.com/1CatAI/1Cat-vLLM/pull/557), [#583](https://github.com/1CatAI/1Cat-vLLM/pull/583), [#585](https://github.com/1CatAI/1Cat-vLLM/pull/585).

The recorded V100 workflow checks include 1024-square Turbo/Base images and H3 text/keyframe/reference-image video at 1344×768, 107 frames and 24 fps with stereo audio. These are bounded workflow checks, not qualification of every duration, model variant or final release-wheel configuration. See [native creative jobs](docs/design/native_media/CREATIVE_JOBS.md) for the tested recipes and limits.

---

# 🧠 What We Mean by “Make Volta Fast Again”

We do **not** claim V100 has the same theoretical peak as A100, H100, or Blackwell.

The point is different.

A large amount of modern inference software simply does not seriously optimize for SM70 anymore.

That creates two gaps:

```text
hardware-generation gap
+
software-neglect gap
```

1Cat-LLM works on the second gap.

When representative Attention useful compute moves from:

```text
17.92 TFLOP/s
```

to:

```text
46–47 TFLOP/s
```

and then to:

```text
≈60.8 TFLOP/s
```

while real 27B 256K decode still reaches:

```text
50.376 tok/s
```

the conclusion is not that V100 “became A100.”

The conclusion is:

> **Software stopped wasting V100.**

---

# 📦 Installation

Install the published [v1.5.1 release](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1) into a clean environment. Download its wheel and `SHA256SUMS` from the same release.

Published binary requirements:

```text
Linux x86_64 · GLIBC >= 2.38 · GLIBCXX >= 3.4.21
Python 3.12
PyTorch 2.10.0 + CUDA 12.8 runtime dependencies
CUDA 12.8 Toolkit + C++ compiler + ninja for remaining TileLang JIT
SM70 / Tesla V100
```

The artifact is **`1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl`**. It has not passed the manylinux container gate. Its standard CUDA Toolkit/compiler requirements remain even though the wheel bundles the SM70 extensions, Flash-V100, FlashQLA and launchers. The release was tested with driver 580.173.02 and recommends 570.124.06 or newer.

With `uv` installed:

```bash
sha256sum --ignore-missing -c SHA256SUMS
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  ./1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl
source .venv/bin/activate
```

The installer resolves declared Python/Torch dependencies. Download target and draft weights separately. The release profile needs no private developer `.so`, source overlay or performance environment overrides; retain the standard Toolkit/JIT setup above.

Verification:

```bash
python - <<'PY'
import sys
import torch
import vllm
import flash_attn_v100
from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils
from flash_attn_v100 import flash_attn_grouped_verify_max_query_tokens

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("vLLM:", vllm.__version__)
print("flash_attn_v100:", flash_attn_v100.__version__)
print("DFlash2 grouped verify max Q:", flash_attn_grouped_verify_max_query_tokens())
print("FlashAttention-V100: OK")
PY
```

This checks imports and visible hardware. Validate actual startup and requests using your model, topology and memory budget; it is not a performance test.

---

# ▶ Qwen3.8-27B-NVFP4 + DFlash2

## Example TP4 + E4M3 serving command

The release wheel installs the V100 launcher and its versioned profile. Supply the target checkpoint path:

```bash
serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-QUASAR-NVFP4
```

The profile pins TP4, FP16 compute/draft KV, FP8 E4M3 target KV, 256K context,
`--max-num-batched-tokens 8192`, `--max-num-seqs 4`, and the 2048/8192 KV and
Mamba block sizes. Append normal `vllm serve` options to override a release
default. This profile is validated for four peer-connected V100-SXM2 32GB
GPUs; other hardware and concurrency levels need a separate memory and speed
check.

The first launch downloads the pinned `incoai/Qwen3.8-27B-DFlash2` revision (about 3.85 GB of draft weights). To use an existing local draft:

```bash
serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-QUASAR-NVFP4 \
  --draft /models/Qwen3.8-27B-DFlash2
```

For the validated Qwen3.8 DFlash2 contract, runtime policy resolves the checkpoint-native draft geometry and the SM70 draft Attention backend.

Representative automatic values:

```text
official draft block size = 8
draft width               = 7
selector Top-K            = 16
release target KV         = FP8 E4M3
release draft KV          = FP16
draft attention backend   = FLASH_ATTN_V100
verification fast paths   = automatic
```

Operators select compatible paths from the actual dtype, local head/projection layout, shape and execution policy. E4M3 grouped FP32 and long-context verification are part of the release recipe. Explicit E5M2 remains available for historical reproduction, but does not select the E4M3-specific acceleration paths. A changed TP degree, KV format or service capacity needs its own memory, speed and quality check.

Inspect the installed configuration and the running server's report:

```bash
python -m vllm.sm70_profiles show qwen38_27b_nvfp4_dflash2 --json
curl --fail http://127.0.0.1:8000/v1/sm70/acceleration
```

Supply the server's API-key authentication if enabled. The report distinguishes configured capabilities, and current main also reports loaded kernel instances and fallback reasons ([#802](https://github.com/1CatAI/1Cat-vLLM/pull/802)). A configured or prepared path is not proof that every request executed it; pair the report with worker route logs. See [the release profile guide](docs/serving/sm70_release_profile.md).

---

## Historical DFlash2 release-path measurements

These retain the original pre-1.5.1 PR contracts. Use the release-to-release table above for the published v1.5.1 comparison.

| Contract | Result |
|---|---:|
| Complete DFlash2 round | **≈17.38 ms** |
| 32K cold prefill | **≈4,039–4,069 tok/s** |
| 64K pure prefill | **≈3,567 tok/s** |
| 32K vs retained pre-closure DFlash2 | **+30.1%** |
| 64K vs retained pre-closure DFlash2 | **+37.7%** |
| Historical web-prompt streaming decode | **206.06 tok/s** |
| High-acceptance MBPP request | **251.60 tok/s** |
| Adaptive lookup q16 repeated-context | **316.27 tok/s** |
| Structured API | **24/24 pass** |
| Long alternating-prefix state | **5/5 pass** |
| Target-only / DFlash2 WikiText PPL | **5.4993116 / 5.4993622** |

---

# ▶ Flash-Next NVFP4 + MTP4

The same release includes a separate Flash-Next launcher:

```bash
serve_flash_next_nvfp4_v100.sh /models/Qwen3.8-Flash-Next-NVFP4
```

Its defaults are **TP4, FP16 compute/KV, MTP4, 131072 context, 8192-token prefill, one sequence, memory utilization 0.90, prefix caching and synchronous scheduling**. It targets four peer-connected V100-SXM2-32GB GPUs and automatically selects compatible FP16 acceleration and hybrid PLE placement.

The release test machine had ample host RAM: Flash-Next PLE tables alone total approximately **47.68 GiB**, before model mappings, caches and process memory. Low-RAM operation was not qualified by that release test. Later main includes capability-based device/pinned-host/disk PLE placement and packed GGUF lookup; these have their own storage and model-validation conditions ([#806](https://github.com/1CatAI/1Cat-vLLM/pull/806), [#877](https://github.com/1CatAI/1Cat-vLLM/pull/877), [#885](https://github.com/1CatAI/1Cat-vLLM/pull/885)).

---

# 🔨 Build From Source

Clone:

```bash
git clone https://github.com/1CatAI/1Cat-vLLM.git
cd 1Cat-vLLM
```

Set the bundled CUDA extensions' SM70 build targets:

```bash
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_CUDA_ARCHITECTURES=70
```

Use this fork's source checkout with CUDA 12.8 and its [CUDA build dependencies](requirements/build/cuda.txt). The [source-build procedure](docs/getting_started/installation/gpu.cuda.inc.md) describes editable builds; its upstream prebuilt-wheel examples are not the 1Cat SM70 release. Keep the bundled Flash-V100 and other native extensions matched to the source revision.

The two variables above are build-time inputs for a source build only. They
are already fixed in the SM70 release-wheel build and are not needed after
installing that wheel.

When building with CUDA 12.8 for SM70, dependency metadata selects the CUDA
12.8 PyTorch wheels matching the build interpreter and Linux architecture.
The published SM70 release wheel still requires Python 3.12 on x86_64;
building for another interpreter does not make that release wheel compatible.

Because this project contains custom CUDA extensions, make sure the active compiler/toolkit matches the PyTorch CUDA ABI used by your environment.

---

# 📐 Benchmarking Policy

1Cat-LLM intentionally separates:

```text
kernel latency
operator throughput
Attention useful TFLOP/s
prefill tok/s
target-only pure decode
speculative pure decode
streaming decode
endpoint throughput
task-quality score
PPL / distribution checks
```

A benchmark claim is most useful when it retains:

- exact model/checkpoint;
- GPU type/count;
- TP/PP topology;
- context and output length;
- batch size;
- KV dtype;
- quantization route;
- CUDA Graph mode;
- prefix-cache state;
- sampling contract;
- speculative method;
- acceptance length;
- quality result;
- whether the result is measured or projected.

This README follows that policy wherever the underlying PR retained enough information.

---

# 🛡️ Promotion Policy

A fast path is not promoted solely because a microbenchmark is faster.

Depending on the arithmetic change, promotion may require:

- bitwise operator equality;
- bounded numerical error;
- CUDA Graph replay stability;
- same-contract endpoint speed;
- dataset quality;
- natural-stop / output-health checks;
- PPL / logprob distribution checks;
- explicit rollback;
- structural/runtime admission rather than hard-coded model identity.

An open or Draft PR is not a merged feature. A merged PR may also retain rejected experiments alongside its accepted implementation; merge status does not qualify every number in its history.

The early ≈79 TFLOP/s Attention candidate is a good example: its historical 256K quality failure does not qualify that arithmetic as a production default. Later repaired FP32 routes were merged separately. The original 79T number remains historical; the later full-FP32 measurements keep their own precision and validation contracts.

## Open Follow-ups at This Snapshot

| PR | Reported scope | Status |
|---|---|---|
| [#956](https://github.com/1CatAI/1Cat-vLLM/pull/956) | Superseded Mamba state-block retention with align mode, async scheduling and prefill chunks spanning multiple state blocks | Open; pending integration |
| [#779](https://github.com/1CatAI/1Cat-vLLM/pull/779) | Bound TurboQuant SDPA continuation scratch; retain numerical and model-quality gates | Draft; pending integration |
| [#715](https://github.com/1CatAI/1Cat-vLLM/pull/715) | DeepSeek-V4 FP16 `hc_head` range handling for attention-sink rows | Open; pending integration |

These reports remain separate from the merged improvements above. Check the linked PR for the affected configuration, evidence and current resolution before relying on that fix.

---

# 🧱 Runtime, Not Just Kernels

1Cat-LLM includes work across the whole serving path:

- FlashAttention-V100;
- paged KV utilities;
- FP8 KV bridges;
- QSA sparse Attention;
- FlashQLA / GDN;
- TurboMind SM70 quantized kernels;
- grouped MoE;
- MTP;
- DFlash2;
- CUDA Graph;
- prefix cache;
- hybrid Mamba state;
- custom all-reduce;
- sampling;
- tool calling;
- reasoning parser;
- structured output;
- standalone GGUF loading and mixed-format projection preparation;
- capability-based kernel selection and loaded-kernel diagnostics;
- mixed-prefill scheduling and hybrid prefix replay;
- device / pinned-host / disk PLE storage;
- native image / audio / video jobs;
- wheel / RPATH / ABI packaging.

A fast kernel is only useful if the full model and serving API can use it correctly.

---

# 💾 More Room for the Model and Its Context

Recent work also reduces resident weights, scale copies, graph scratch and loading peaks. Representative PR measurements, **per GPU**, include:

| Change | Recorded result | Scope |
|---|---|---|
| Shared weights / codes / scales | Model allocation **10.07 → 6.47 GiB**; available KV budget **10.57 → 12.68 GiB** | [#671](https://github.com/1CatAI/1Cat-vLLM/pull/671) · QUASAR NVFP4 / TP4 / DFlash2 q7 / E4M3 target KV |
| Shared graph scratch / draft snapshots | Non-KV active memory: TP2 **17.117 → 15.146 GiB**, TP4 **10.225 → 8.794 GiB** | [#677](https://github.com/1CatAI/1Cat-vLLM/pull/677) · recorded default ON/OFF comparison |
| Reused compiled subgraphs | First startup **271.26 s**, matching-config restart **95.30 s** | [v1.5.1](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1) · two observations within the new version |

The memory records use V100-SXM2-32GB, CUDA 12.8 / Torch 2.10, FP16 compute/draft KV, 262144 context, 8192-token prefill and CUDA Graphs. #671 uses 32 sequence slots / memory .85; #677 uses 32 slots / memory .90. The savings overlap and must not be added. Automatic KV allocation can consume the recovered space, so total NVML usage need not fall by the same amount. Restart caching still loads weights and captures CUDA graphs; it does not make startup free.

Local tensor geometry now also admits qualified **TP1 / TP2** paths. [#666](https://github.com/1CatAI/1Cat-vLLM/pull/666) records target-only 27B NVFP4 checks at **TP1 64K** and **TP2 256K** on 32GB cards. These are bounded capability checks, not a claim that the TP4 DFlash2 release recipe fits unchanged on one card. Higher client concurrency can queue behind the actual resident capacity.

---

# 🧭 Project Direction

1Cat-LLM focuses on a simple question:

> **How much modern LLM inference performance is still hidden inside Volta if the software stack is redesigned instead of abandoned?**

Current directions include:

- further long-context Attention work;
- lower DFlash2 verifier cost;
- higher-acceptance speculative execution;
- sparse Attention;
- modern quantization formats on SM70;
- fused decode hot paths;
- MoE routing and grouped GEMM;
- broader GGUF model-quality and serving qualification;
- mixed-load latency, cache reuse and memory headroom;
- multi-model SM70 support;
- stable wheel/release packaging.

---

# 💬 WeChat Community

Join the **1Cat-LLM Open-Source Community** by adding WeChat ID **`YM_isi`** to request the latest group invitation.

> The previously displayed Group 8 QR code expired on **September 20, 2026**. Use the contact above for a current invitation.

---

# ❤️ Acknowledgements

1Cat-LLM builds on the work of the broader open-source inference ecosystem, including vLLM, NVIDIA CUDA, FlashAttention, CUTLASS/TurboMind-related kernels, model authors, quantization projects, and contributors whose work is referenced in individual PRs and source files.

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)
- [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
- [marlin_v100](https://github.com/zhinianqin/marlin_v100)
- [v100-skinny](https://github.com/dnv2003/v100-skinny) — QPN quadpair-N `m8n8k4` decode layout behind the SM70 QPN2 / QPN4 / QPN8 and MXFP4-QPN kernels (MIT; notice retained in `csrc/sm70_turbomind/ops/LICENSE.v100-skinny`)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and the vllm-gguf-plugin bridge — pinned GGUF source provenance is recorded in the [source manifest](csrc/quantization/gguf_upstream/source_manifest.json), with notices retained in the source and wheel.

Special thanks to [@yangzhuxinyzx](https://github.com/yangzhuxinyzx) and [@1CatTCat](https://github.com/1CatTCat) for their outstanding contributions to the continued evolution and performance breakthroughs of **1Cat-LLM**.

Where external implementations or algorithms are adapted, provenance and license information should be preserved in the corresponding source and PR history.

---

# License

Please refer to the repository license and the licenses of bundled or adapted third-party components.
