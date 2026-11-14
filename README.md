<!-- markdownlint-disable MD041 -->

<p align="center">
  <img src="./assets/1cat-llm-logo.png" alt="1Cat-LLM logo" width="960">
</p>

# 1Cat-LLM

**English** · [简体中文](README.zh-CN.md)

## Make Volta Fast Again

### Modern LLM inference for NVIDIA Tesla V100 / SM70

> **1Cat-vLLM is becoming 1Cat-LLM.**
>
> **Broader support. More efficient kernels. More models to choose from.**

We started by making modern models fast on V100. The next chapter carries that work further: broader model and runtime support, more efficient execution, and more choice in how models are packaged and quantized.

New **[GGUF support](#gguf-support)** brings a wider choice of community-quantized checkpoints for supported architectures, with more options for model size, memory use and quality. The new name reflects that direction, with Volta engineering at its foundation.

**The rename is planned.** Repository links, release filenames and installation commands below retain their current names during the transition.

**Published release · [v1.5.3](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · November 14, 2026**

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

# Demo and v1.5.3 Benchmarks

**892 aggregate tok/s measured at 8K with eight concurrent requests** using Qwen3.8-27B NVFP4 with native MTP4. The v1.5.3 tests use **4× V100-SXM2 32GB** (NVLink, TP4) at 877 MHz memory and a 300 W limit. [v1.5.3 benchmark report](http://156.226.174.161:48152/1.5.3/)

| v1.5.3 workload · temperature 0 | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| 8K · C8 · measured aggregate tok/s | 479 | 892 | 408 |
| 8K · C8 · estimated aggregate tok/s at 70% draft acceptance | 1,120 | 1,072 | 584 |
| 64K · C4 · measured ms per decode step | 32.68 | 24.20 | 34.03 |
| HumanEval · C1 · measured tok/s | 545 | 370 | 276 |

HumanEval uses 20 prompts with natural stopping. C1, C4 and C8 denote one, four and eight concurrent requests.

The Bilibili walkthrough shows the earlier **v1.5.2** release. [Watch the demo](https://www.bilibili.com/video/BV1ouWy67Esz/)

The published wheel and Git tag resolve to [`cb97f379b`](https://github.com/1CatAI/1Cat-vLLM/commit/cb97f379b67eb9f5703709db961d9dabcdecbcff). See the [English release notes](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/RELEASE_NOTES.en.md), [中文发布说明](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/RELEASE_NOTES.zh-CN.md), and [release PR index](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/PR_INDEX.md).

---

# 📊 Performance First

SM70 Flash-V100 resolves `--kv-cache-dtype fp8` to E4M3; the 27B release profile explicitly selects `fp8_e4m3`. DFlash2 verification retains FP32 attention state and FP32 logits on its admitted paths.

Keep the Python package and Flash-V100 extension from the same build. The original E4M3 repair requires precision revision 4; current multi-head and request-major routes require newer native capabilities. See [the precision contract](docs/design/sm70_dflash2_fp32_defaults.md) and [concurrent long-context validation](docs/design/sm70_dflash2_long_batch_20260927.md).
Historical E5M2/FP16-partial performance results below keep their original
configuration and are not speed claims for these precision defaults.

## v1.5.3 · Published-wheel Decode

**Hardware and build.** The release used 4× V100-SXM2 32GB (NVLink, TP4), 62 GB reported host RAM, ECC on, 877 MHz memory, up to 1530 MHz core, and a 300 W limit. The wheel is built from `cb97f379b` for CUDA 12.8. The three profiles use FP16 compute, up to eight scheduled sequences, a 4096-token prefill budget, and a 262144-token service limit. The 27B runs use the **Unsloth** NVFP4 checkpoint. Flash-Next uses NVFP4 with FP16 attention KV and host-backed FP8 E4M3 QSA history. See the launch profiles below for the KV and draft settings. The report identifies the 27B target family and pins the DFlash2 draft in the launch recipe. It does not publish the exact evaluated target revisions, synthetic prompt IDs, run repetitions, output cap, or complete non-greedy sampling settings. [Release notes](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [Detailed benchmark report](http://156.226.174.161:48152/1.5.3/#data)

**Measured synthetic decode.** Cells show **ms per engine step / reported aggregate tok/s** at temperature 0. This workload uses synthetic filler and a short question. Timings cover the decode window where all requests are generating and exclude prefill. Aggregate throughput is copied from the report's measured column. Its per-request column is a separate measurement and multiplying it by concurrency can give a different total. The report does not give raw event logs to reconcile those windows.

| Input | Concurrent requests | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- | --- |
| 8K | 1 | 13.87 / 243 | 12.99 / 229 | 16.93 / 141 |
| 8K | 2 | 17.33 / 338 | 15.23 / 329 | 25.92 / 180 |
| 8K | 4 | 23.80 / 461 | 18.93 / 561 | 32.13 / 228 |
| 8K | 8 | 42.23 / 479 | 28.39 / 892 | 51.90 / 408 |
| 64K | 1 | 16.08 / 204 | 14.72 / 204 | 17.76 / 136 |
| 64K | 2 | 21.68 / 264 | 17.70 / 315 | 27.85 / 181 |
| 64K | 4 | 32.68 / 304 | 24.20 / 552 | 34.03 / 265 |
| 64K | 8 | 59.76 / 322 | 38.61 / 720 | — |

**Normalized estimates at 70% draft acceptance.** Cells show **per-request / aggregate tok/s** (C1 shows one value). The report divides 5.9 emitted tokens per DFlash2 q7 step or 3.8 per MTP4 step by measured step time, rounds the per-request rate, and multiplies by concurrency. These estimates assume the same acceptance across workloads and exclude prefill. A dash denotes a missing measurement. The report has no 64K temperature-1.0 rows or Flash-Next 64K C8 rows.

| Input | Profile | Temperature | C1 | C2 | C4 | C8 |
| --- | --- | --- | --- | --- | --- | --- |
| 8K | 27B DFlash2 | 0 | 426 | 340 / 680 | 248 / 992 | 140 / 1120 |
| 8K | 27B DFlash2 | 0.6 | 403 | 325 / 650 | 241 / 964 | 138 / 1104 |
| 8K | 27B DFlash2 | 1 | 401 | 324 / 648 | 241 / 964 | 138 / 1104 |
| 8K | 27B MTP4 | 0 | 293 | 250 / 500 | 201 / 804 | 134 / 1072 |
| 8K | 27B MTP4 | 0.6 | 242 | 212 / 424 | 175 / 700 | 120 / 960 |
| 8K | 27B MTP4 | 1 | 243 | 212 / 424 | 175 / 700 | 122 / 976 |
| 8K | Flash-Next MTP4 | 0 | 224 | 147 / 294 | 118 / 472 | 73 / 584 |
| 8K | Flash-Next MTP4 | 0.6 | 186 | 133 / 266 | 111 / 444 | 75 / 600 |
| 8K | Flash-Next MTP4 | 1 | 186 | 133 / 266 | 112 / 448 | 72 / 576 |
| 64K | 27B DFlash2 | 0 | 367 | 272 / 544 | 181 / 724 | 99 / 792 |
| 64K | 27B DFlash2 | 0.6 | 350 | 262 / 524 | 178 / 712 | 98 / 784 |
| 64K | 27B MTP4 | 0 | 258 | 215 / 430 | 157 / 628 | 98 / 784 |
| 64K | 27B MTP4 | 0.6 | 221 | 189 / 378 | 146 / 584 | 94 / 752 |
| 64K | Flash-Next MTP4 | 0 | 214 | 136 / 272 | 112 / 448 | — |
| 64K | Flash-Next MTP4 | 0.6 | 180 | 127 / 254 | 103 / 412 | — |

Stock-clock v1.5.3 timings and the previous release's elevated-clock results use different hardware settings. The full report contains measured per-request rates, p90 step times, emitted tokens per step and TTFT for each available row. [Release notes](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [Detailed benchmark report](http://156.226.174.161:48152/1.5.3/#data)

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

## v1.5.3 · Six Real Prompt Sets

**Measured single-request tok/s / draft acceptance**, temperature 0, 20 prompts per set with natural stopping. The report also publishes C4 and temperature-0.6 throughput. C4 dataset timing includes requests ending at different times. The table reports generation speed on these prompts and accuracy is reported in the quality section below.

| Prompt set | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| HumanEval | 545 / 84% | 370 / 90% | 276 / 91% |
| MBPP | 486 / 80% | 340 / 85% | 261 / 84% |
| MATH500 | 473 / 76% | 336 / 82% | 255 / 82% |
| GSM8K | 464 / 78% | 335 / 84% | 251 / 81% |
| MT-Bench | 336 / 42% | 289 / 59% | 226 / 60% |
| Chinese prose | 189 / 22% | 224 / 46% | 168 / 44% |

Measured throughput depends on prompt mix. In the same release, DFlash2 led on the HumanEval subset while native MTP4 led on the Chinese prose subset. [Release notes](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [Detailed benchmark report](http://156.226.174.161:48152/1.5.3/#data)

## Historical Model Measurements

The table below preserves **complete-model / API / pure-decode / speculative-decode** measurements from their original PRs. They are historical evidence under separate contracts; the v1.5.3 wheel measurements are above.

| Model | Hardware / Runtime | Workload | Measured result | Evidence / Status |
|---|---|---|---:|---|
| **Qwen3.6-27B-AWQ + MTP4** | 4× V100 · TP4 · E5M2 KV · Flash-V100 · CUDA Graph | 64K decode | **100.564 tok/s** | [v1.2.2 release](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2) · AL 4.981 / 99.52% |
| **Qwen3.6-27B-AWQ + MTP4** | same | 128K decode | **85.258 tok/s** | [v1.2.2 release](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2) · +87.64% vs no-MTP |
| **Qwen3.6-27B-AWQ + MTP4** | same · max 256K | 261,888 context decode | **49.772 tok/s** | [v1.2.2 release](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2) · AL 5.000 / 100% |
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
| **DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XXS + DSpark5** | **8× V100** · PP2×TP4 · MRV2 · FULL graphs | 32K prompt / 1024 · code · JSON · legal · prose | **200.5 · 206.6 · 116.5 · 90.1 tok/s decode** · 1,900 tok/s prefill (2,200 at 128K) | [#1380](https://github.com/1CatAI/1Cat-vLLM/pull/1380) · merged documentation for a community result · [raw data](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/measurements.md) |
| **GLM-5.3-Flash-NVFP4** | 8× V100 · TP4/PP2 · E4M3 KV · no-MTP | 1K / 256 decode | **53.016 tok/s** | [#402](https://github.com/1CatAI/1Cat-vLLM/pull/402) · merged historical quality audit |

---

# 🧪 Dataset / Quality × Throughput Benchmarks

Raw `tok/s` alone can turn optimization into a benchmark game. 1Cat-LLM therefore records **real model throughput, dataset score, natural-stop health, output validity, and speculative acceptance** together.

## v1.5.3 · Greedy Quality and Output Health

The maintainers evaluated the same questions with greedy decoding against the original 27B BF16 model. The 40 coding cases use unit-test pass counts. Deltas and 95% confidence intervals are in percentage points versus 27B BF16, as reported. The public report provides aggregates and does not include per-item responses or the interval calculation.

| Evaluation | 27B BF16 | 27B NVFP4 | Flash-Next NVFP4 |
| --- | --- | --- | --- |
| GSM8K · 300 | 285/300 (95.0%) | 290/300 (96.7%) · +1.7 [0.0, +3.7] | 289/300 (96.3%) · +1.3 [−1.3, +4.0] |
| MMLU · 400 | 336/400 (84.0%) | 331/400 (82.8%) · −1.3 [−3.3, +0.8] | 349/400 (87.3%) · +3.3 [+0.5, +6.3] |
| MBPP + HumanEval · 20 each | 21/40 (52.5%) | 21/40 (52.5%) · 0.0 | 20/40 (50.0%) · −2.5 [−7.5, 0.0] |

- When 27B NVFP4 diverged from the 27B BF16 token choice in the reported audit, the BF16 preference margin had a median of **0.20 logprob** and the NVFP4 choice stayed in the BF16 top five. Flash-Next is a different model and the report compares it by task score.
- The evaluated outputs recorded **zero garbled outputs or repetition loops**. Tested cached and cold outputs matched character for character. The report gives one aggregate 27B NVFP4 quality column and does not give separate DFlash2 and MTP4 scores.
- Fresh Docker runs covered 45 API checks for 27B DFlash2 (including image inputs) and 38 for Flash-Next. In the separate 20-minute load tests, completion/cancellation counts were **2864/544** and **958/196**, respectively, with zero reported errors. [Release validation](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3)

The retained DFlash2 coding/PPL gates below use their recorded historical configuration, including E5M2 KV where specified. They record a separate historical quality protocol.

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
| **DeepSeek-V4.1-Flash-1017 · PP2×TP4 · IQ2_XXS** | MMLU-Pro **79.9 %** (700) · HumanEval+ **147/164** plus · LegalBench **0.716** (22 tasks) · top-1 vs original **89.0 %** | **182.8 tok/s code decode at 128K** | [#1380](https://github.com/1CatAI/1Cat-vLLM/pull/1380) · historical community evaluation |
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
- the v1.5.3 release measures **545 tok/s** on 20 HumanEval prompts at temperature 0 with natural stopping. [v1.5.3 benchmark report](http://156.226.174.161:48152/1.5.3/#data)

DFlash2 throughput depends on draft acceptance, prompt repetition, verification width, context length and task type.

---

# 📏 Long Context Means More Than “It Fits in 256K”

## v1.5.3 · Single-request Long Decode and Prefix Reuse

**Measured step time (ms) / normalized decode tok/s** at temperature 0. The throughput column uses the same 70% acceptance assumption as above.

| Input | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| 1K | 13.54 / 436 | 12.62 / 301 | 16.47 / 231 |
| 8K | 13.87 / 426 | 12.99 / 293 | 16.93 / 224 |
| 32K | 14.77 / 399 | 13.76 / 276 | 17.05 / 223 |
| 64K | 16.08 / 367 | 14.72 / 258 | 17.76 / 214 |
| 128K | 18.51 / 319 | 16.29 / 233 | 19.08 / 199 |
| 250K | 23.23 / 254 | 19.59 / 194 | 19.71 / 193 |

**Measured TTFT in seconds** for cold input / full prefix hit / 200-token follow-up. The follow-up appends 200 input tokens to an existing conversation and reuses cached input and generated history.

| Input | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| 1K | 0.57 / 0.29 / 0.34 | 0.56 / 0.28 / 0.33 | 0.52 / 0.29 / 0.31 |
| 8K | 1.81 / 0.49 / 0.54 | 1.81 / 0.48 / 0.56 | 1.50 / 0.40 / 0.53 |
| 32K | 7.85 / 0.51 / 0.57 | 7.95 / 0.48 / 0.56 | 5.99 / 0.45 / 0.48 |
| 64K | 17.37 / 0.43 / 0.61 | 17.64 / 0.40 / 0.61 | 12.49 / 0.52 / 0.51 |
| 128K | 41.50 / 1.05 / 1.14 | 42.51 / 1.02 / 1.14 | 26.54 / 0.86 / 0.91 |
| 250K | 106.68 / 1.58 / 1.75 | 110.56 / 1.60 / 1.74 | 61.40 / 2.79 / 4.58 |

The 262144-token service limit is shared by prompt and generation. KV capacity determines how many long requests can be resident at the same time. [Release notes](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [Detailed benchmark report](http://156.226.174.161:48152/1.5.3/#data)

## Historical Long-context Results

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
| DFlash2 q8 long verification | [#1505](https://github.com/1CatAI/1Cat-vLLM/pull/1505) · 64K / C4 · stock-clock pre-release A/B | 37.14 ms/step | **32.64 ms/step** | **12.1% lower** |
| Native MTP4 · 27B QUASAR NVFP4 | [#1491](https://github.com/1CatAI/1Cat-vLLM/pull/1491) · source A/B · V100 TP4 · E4M3 KV · 65,536 input / 256 output · C1, cold prefix, median of 3 | 18.82 ms/step | **14.34 ms/step** | **23.8% lower** |
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

These are medians of two warm measured runs after warmup, using the PR's two-NVLink-pair topology. Resident EOS is disabled to sustain the load; incoming requests complete naturally. **Shorter decoder stalls trade against longer incoming TTFT.** The 250 ms step target is a soft scheduler target, not a latency guarantee. This is a historical merged PR run with its own scheduler recipe. The v1.5.3 test profile sets `--mixed-prefill-step-latency-ms 0`.

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

Adapter support, kernel admission and model-quality qualification are separate milestones. The dense and DFlash2 adapters have recorded loading/generation checks; the MoE adapter PR alone does not establish broad 35B quality or throughput. Consult the [GGUF controls](docs/design/gguf_v100_control.md) and each model's recorded gates. Pair the tagged runtime with the model-specific GGUF configuration and quality evidence.

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

## DeepSeek-V4.1-Flash-1017 · Native PP4×TP4

[#1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482) validates the original FP8 32×32 dense / MXFP4 expert checkpoint with Engram and DSpark5 on **four Dell C4140 nodes**, each with four V100-SXM2 32GB GPUs and connected by InfiniBand. The release notes record a 1053 MHz GPU memory clock for this cluster. The source-built configuration allocates a **1,048,576-token** service window. The longest reported timed request is **512K**, with 23.65 ms per decode round and about 133 s TTFT. Native PP4 re-prefills requests because it runs with prefix caching disabled.

| 1K prompt, temperature 0 | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Measured ms per decode round | 19.3 | 19.7 | 24.8 | 35.1 | 49.5 | 77.5 | 135.7 |
| Release report estimate, aggregate tok/s | 202 | 396 | 628 | 888 | 1264 | 1600 | 1856 |

The estimate uses approximately 3.9 emitted tokens per DSpark round, with observed acceptance varying by case. The PR independently reports **1809 aggregate tok/s** at C64 / 1K and **249 tok/s** for a single HumanEval request. The PR also reports **297/300 GSM8K**, **349/400 MMLU**, **9/9 needle retrieval** across 16K/128K/512K, and **722 requests / zero errors** in its 20-minute soak. Concurrent and single-request outputs can differ at the token level on this configuration. These results come from the source-build PR validation. [PR measurements](https://github.com/1CatAI/1Cat-vLLM/pull/1482) · [Release launch report](http://156.226.174.161:48152/1.5.3/#launch)

Build the native extension and prepare the checkpoint, Engram data and DSpark draft on each node as documented in the [C4140 handoff](benchmarks/deepseek_v41/native/HANDOFF.md). Copy the long-context YAML from the release report. Before launch, set the actual model paths, master address, node IPs, NCCL interfaces and CPU affinity for your cluster. The tagged [run_pp4.sh](benchmarks/deepseek_v41/native/run_pp4.sh) contains C4140-specific paths and addresses. Run the following on each node after adapting its launcher, starting ranks 3, 2, 1 and then 0 (the API node).

```bash
export V41_CONFIG="$HOME/v41/serve-native-pp4-long.yaml"
export VLLM_V41_BATCH=1 VLLM_V41_WINDOW_RING=0 VLLM_V41_PP_MICRO=1 VLLM_V41_PP_AHEAD=1
export VLLM_V41_LAZY_OUTPUT_IDS=1 VLLM_V41_PROPOSE_GRAPH=1 VLLM_V41_INDEX_SHARD=1
export VLLM_V41_PREFILL_PIPELINE=1 VLLM_V41_HC_GEMM=1 VLLM_V41_LIVE_TABLE=1
export VLLM_V41_FAST_READD=1 VLLM_V41_TOPP_NO_FENCE=1
# Set RANK to the local node's PP rank (0 through 3).
"$HOME/v41/run_pp4.sh" "${RANK:?Set the local pipeline rank}"
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

The [v1.5.3 release](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) ships a **Linux x86_64, CPython 3.12, CUDA 12.8** wheel and a Docker build bundle. Use an NVIDIA driver compatible with CUDA 12.8. The wheel pins PyTorch 2.10.0+cu128 and bundles the SM70 extensions. Use a CUDA 12.8 Toolkit with `nvcc` on `PATH` and a C++ compiler for JIT compilation. The FlashQLA GDN prefill route can use its bundled precompiled kernel when TileLang cannot find `nvcc`. The Docker build uses Ubuntu 24.04 and includes both compilers. [GDN fallback #1400](https://github.com/1CatAI/1Cat-vLLM/pull/1400)

The wheel's bundled native libraries reference **GLIBC 2.38** and **GLIBCXX 3.4.30**. Use a host that provides those versions or build the Ubuntu 24.04 release image.

## Hardware and Checkpoints

| Profile or path | Hardware evidence | Checkpoint and scope |
| --- | --- | --- |
| 27B DFlash2 q7 | 4× V100-SXM2 32GB, NVLink, TP4 | [Unsloth Qwen3.8-27B NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) and [pinned DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/dedf8df68adfb1afeaf7b7480c0a0243108177b4) · release-wheel measurements |
| Native 27B MTP4 | Same 4×32GB release setup | Unsloth 27B NVFP4 with built-in MTP · same-wheel measurements. QUASAR NVFP4 and FP8 at TP2/TP4 have separate [PR tests](https://github.com/1CatAI/1Cat-vLLM/pull/1491) |
| Flash-Next MTP4 | 4× V100-SXM2 32GB, NVLink, TP4 | [RadixArk Flash-Next NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) is a project-listed checkpoint. The release recipe uses an MTP checkpoint directory and host-backed QSA KV |
| DeepSeek-V4.1-Flash-1017 | 4× Dell C4140, 16× V100-SXM2 32GB in total, PP4×TP4 | [Original checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), native FP8/MXFP4, DSpark5 · source-build test in [#1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482) |
| Other model paths | Model- and topology-specific evidence | See the Qwen3.6 AWQ/NVFP4, Qwen3.8 FP8, GGUF, DeepSeek-V4 and GLM-5.3 sections. Turing SM75 has individual merged paths such as [GDN prefill](https://github.com/1CatAI/1Cat-vLLM/pull/1228) and an [FA2 build change](https://github.com/1CatAI/1Cat-vLLM/pull/1346). The published release matrix covers SM70 |

The release recommends **at least 64 GB host RAM**. The v1.5.3 benchmark report records these startup and memory observations on its TP4 test host. The TP2 results come from separate source builds.

| Measured resource | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | ---: | ---: | ---: |
| Per-GPU VRAM after start / peak | 27.2 / 28.6 GiB | 26.3 / 26.3 GiB | 29.9 / 30.6 GiB |
| Reported KV capacity | 1,308,270 tokens | 1,598,295 tokens | 378,652 tokens with history in host RAM |
| Host launch with a warm compile cache | 125 s | 114 s | 342 s |
| Docker empty-cache launch / restart | 323 / 122 s | — | 624 / 343 s |

[Release resource and startup measurements](http://156.226.174.161:48152/1.5.3/#data)

## Wheel

With `uv` and `curl` installed, download the wheel and checksum file from the same release.

```bash
release_base=https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3
wheel=1cat_vllm-1.5.3+cu128-cp312-cp312-linux_x86_64.whl
curl -fL -o "$wheel" "$release_base/1cat_vllm-1.5.3%2Bcu128-cp312-cp312-linux_x86_64.whl" &&
  curl -fLO "$release_base/SHA256SUMS" &&
  sha256sum --ignore-missing -c SHA256SUMS &&
  uv venv --python 3.12 .venv &&
  uv pip install --python .venv/bin/python "./$wheel" &&
  source .venv/bin/activate
```

Check the installed build and visible GPU before starting a model.

```bash
python - <<'PY'
from importlib.metadata import version

import flash_attn_v100
import torch

print(version("1cat-vllm"), torch.__version__, torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

The wheel SHA256 is `0558021651c41ab111609d121cef029c09224de4b6e6f8fd362754dfdd84990f`. The Docker bundle SHA256 is `2c8e43157735964eda8b7691f6cddf61efa06e1715b3dd6438eb252067ee3db1`. [Published checksums](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/SHA256SUMS)

## Docker and 1Cat Studio

Docker requires a CUDA-compatible host driver and the NVIDIA Container Toolkit. The build contacts Ubuntu, PyPI, PyTorch and NVIDIA package sources. Mount model directories under `/models` using the names in the launch commands below. The Docker preset expects two directories for its selected model, including `Qwen3.8-27B-DFlash2` for `MODEL=27b` or `Qwen3.8-Flash-Next-mtp` for `MODEL=flashnext`. The bundle's `README-docker.en.md` documents mirrors and environment options.

```bash
archive=1cat-vllm-1.5.3-docker.tar.gz
curl -fLO "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/$archive" &&
  curl -fLO https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/SHA256SUMS &&
  sha256sum --ignore-missing -c SHA256SUMS &&
  tar -xzf "$archive" &&
  cd 1cat-vllm-1.5.3-docker &&
  sha256sum -c SHA256SUMS &&
  docker build -t 1cat/vllm-v100:1.5.3 .
```

```bash
docker run -d --runtime=nvidia --gpus all --network host --ipc=host \
  --ulimit memlock=-1 -e MODEL=27b -e HOST=127.0.0.1 -e PORT=8000 \
  -v /data/models:/models:ro -v vllm-cache-1.5.3:/cache \
  1cat/vllm-v100:1.5.3
```

Select `MODEL=flashnext` for the Flash-Next preset. The released Docker entrypoint exposes `VLLM_API_KEY`, `MAX_NUM_SEQS` and `MAX_MODEL_LEN` as environment options. In 1Cat Studio, install the **1.5.3** runtime and select it in the model configuration. For rollback, select the prior Studio runtime, reinstall its verified wheel in a separate environment, or restart with the prior Docker image and its matching cache volume. [Release deployment and rollback](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3)

---

# ▶ Qwen3.8-27B-NVFP4 + DFlash2

## Example TP4 + E4M3 Serving Command

The explicit launch below follows the published DFlash2 profile, including image admission and the thinking default. Download the target and pinned draft to the two local model directories. The target uses E4M3 KV, and the draft uses `auto` with FP16 compute. DFlash2 proposes seven tokens and uses probabilistic draft sampling.

```bash
vllm serve /models/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b-dflash2 \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 4 --dtype half \
  --kv-cache-dtype fp8_e4m3 --attention-backend FLASH_ATTN_V100 \
  --max-model-len 262144 --max-num-batched-tokens 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.83 --enable-prefix-caching --partial-prefix-cache \
  --mamba-cache-mode align --block-size 2048 --mamba-block-size 2048 \
  --prefix-cache-retention-interval 8192 --mixed-prefill-step-latency-ms 0 \
  --enable-prompt-tokens-details --limit-mm-per-prompt '{"video":0,"audio":0}' \
  --trust-remote-code --seed 0 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --speculative-config '{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","num_speculative_tokens":7,"kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic","enforce_eager":false}'
```

The wheel's bundled helper uses the same core TP, KV, context and DFlash2 settings. It downloads the pinned draft when `--draft` is omitted. Its packaged defaults include a text-only multimodal limit and thinking disabled. Use the explicit command above to reproduce the release example's frontend settings. [Profile source](vllm/sm70_profiles/qwen38_27b_nvfp4_dflash2.json) · [Launcher source](scripts/serve_qwen38_27b_nvfp4_v100.sh)

```bash
serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-NVFP4 --port 8000
python -m vllm.sm70_profiles show qwen38_27b_nvfp4_dflash2 --json
curl --fail http://127.0.0.1:8000/v1/sm70/acceleration
```

The acceleration endpoint reports configured capabilities, loaded kernel instances and fallback reasons where available. Match runtime route logs to the request shape when checking which path ran. [Acceleration diagnostics](vllm/sm70_profiles/acceleration.py)

---

# ▶ Qwen3.8-27B-NVFP4 + Native MTP4

Use a 27B NVFP4 checkpoint with its MTP head. The release command selects four greedy draft tokens using the target checkpoint and takes the default hybrid prefix geometry from the runtime. The shared accelerated verifier selects a compatible route by local shard layout and dtype. [MTP runtime](https://github.com/1CatAI/1Cat-vLLM/pull/1491) · [Prefix defaults](https://github.com/1CatAI/1Cat-vLLM/pull/1492)

```bash
vllm serve /models/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b-mtp \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 4 --dtype half \
  --kv-cache-dtype fp8_e4m3 --attention-backend FLASH_ATTN_V100 \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.83 --enable-prefix-caching \
  --mixed-prefill-step-latency-ms 0 --trust-remote-code \
  --limit-mm-per-prompt '{"video":0,"audio":0}' --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

---

# ▶ Flash-Next NVFP4 + MTP4

The wheel launcher sets `--max-num-seqs 8`, FP16 attention KV, FP8 E4M3 QSA history in pinned host RAM and FP32 Mamba SSM state. It uses greedy MTP4 drafts. The example uses the target's own MTP layers, which are present in the RadixArk checkpoint. For the separate-draft release recipe, add `--draft /models/Qwen3.8-Flash-Next-mtp` and supply that compatible checkpoint directory. The release names this directory without identifying a public revision for it. [Launcher source](scripts/serve_flash_next_nvfp4_v100.sh) · [Checkpoint card](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) · [Release launch profiles](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3)

```bash
serve_flash_next_nvfp4_v100.sh /models/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen3.8-flash-next-nvfp4 --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --trust-remote-code --seed 0 --default-chat-template-kwargs '{"enable_thinking":false}'
```

## OpenAI and Anthropic API Examples

Start one profile on port 8000. Set `model` to the `--served-model-name` used at launch. These examples target the 27B MTP4 command. When the server has `--api-key` configured, include `-H "Authorization: Bearer $VLLM_API_KEY"` and set the client-side variable to that key.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail-with-body -sN http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-mtp","messages":[{"role":"user","content":"Explain prefix caching in two sentences."}],"max_tokens":256,"temperature":0,"stream":true}'
```

```bash
curl --fail-with-body -sN http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen3.8-27b-mtp","messages":[{"role":"user","content":"Explain prefix caching in two sentences."}],"max_tokens":256,"temperature":0,"stream":true}'
```

The Anthropic-compatible Messages API includes `POST /v1/messages/count_tokens` and `HEAD` on both Messages paths for liveness. `GET /health` checks engine readiness. The release also updates `message_start`, measured cache usage, streaming tool calls and reasoning-token accounting. [Anthropic routes](vllm/entrypoints/anthropic/api_router.py) · [Cache accounting #1436](https://github.com/1CatAI/1Cat-vLLM/pull/1436) · [SSE #1440](https://github.com/1CatAI/1Cat-vLLM/pull/1440) · [Tool and reasoning usage #1443](https://github.com/1CatAI/1Cat-vLLM/pull/1443)

---

# 🔨 Build From Source

On Linux x86_64 with the CUDA 12.8 Toolkit, matching C++ build tools and `uv`, check out the release tag and target SM70. Use the same source revision for the Python runtime and bundled native extensions.

```bash
git clone --branch v1.5.3 https://github.com/1CatAI/1Cat-vLLM.git
cd 1Cat-vLLM
uv venv --python 3.12 .venv
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_CUDA_ARCHITECTURES=70
uv pip install -e . --torch-backend=cu128
```

The architecture variables are build-time inputs. See the [CUDA build requirements](requirements/build/cuda.txt) and [source installation guide](docs/getting_started/installation/gpu.cuda.inc.md) for the existing-Torch workflow and build dependencies.

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

## Benchmark and Regression Tools

- [DFlash2 concurrent API benchmark](benchmarks/benchmark_dflash2_concurrent_rounds.py) captures a shared streaming window at a fixed output length, using 8192- or 32768-token prompt fixtures. It exposes temperature, top-p, top-k and seed settings.
- [Flash-Next MTP4 round benchmark](benchmarks/benchmark_sm70_mtp4_round.py) measures complete rounds at a fixed 8K input and supports natural-prompt fixtures.
- [Qwen quality runner](benchmarks/benchmark_sm70_qwen38_quality.py) supports explicit case manifests, timing and output-health checks. [DeepSeek native client](benchmarks/deepseek_v41/native/bench_client.py) uses server counters for emitted and accepted tokens per round.

The v1.5.3 release report publishes summary tables. The tagged benchmark tools have their own inputs and timing windows. The report does not include the exact fixture manifest or a complete replay command for its published matrix.

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

## v1.5.3 Runtime Changes

- **Native MTP and grouped verification.** Shared TP2/TP4 dispatch covers qualified NVFP4 and FP8 Qwen shards. MTP4 q5 uses the newer grouped path. DFlash2 q6–q8 keeps the 48-row E4M3 kernel for eligible shapes. [#1472](https://github.com/1CatAI/1Cat-vLLM/pull/1472) · [#1478](https://github.com/1CatAI/1Cat-vLLM/pull/1478) · [#1491](https://github.com/1CatAI/1Cat-vLLM/pull/1491) · [#1505](https://github.com/1CatAI/1Cat-vLLM/pull/1505)
- **Hybrid prefix cache.** The default geometry uses 2048-token KV/Mamba blocks and partial prefix snapshots on admitted paths. Periodic retention defaults to 8192 where supported and is disabled with authoritative QSA host KV to preserve tail checkpoints. Explicit cache choices take precedence. Pool-capacity checks cover snapshot and hit paths. [#1401](https://github.com/1CatAI/1Cat-vLLM/pull/1401) · [#1492](https://github.com/1CatAI/1Cat-vLLM/pull/1492) · [#1500](https://github.com/1CatAI/1Cat-vLLM/pull/1500)
- **Startup and request stability.** MRV2 warmup respects bounded host-KV state pools, image encoder state survives until speculative tokens are confirmed, and CUDA capability checks are traceable during compilation. [#1493](https://github.com/1CatAI/1Cat-vLLM/pull/1493) · [#1494](https://github.com/1CatAI/1Cat-vLLM/pull/1494) · [#1498](https://github.com/1CatAI/1Cat-vLLM/pull/1498)

## Known Limitations

- **Flash-Next startup and short prefixes.** Set `--max-num-seqs 8` with the 256K recipe. Bare `vllm serve` may select 256 sequence slots and exhaust the Mamba state pool. The preset reuses prefix blocks at a 2048-token granularity and 250K follow-up TTFT reached **4.58 s** in the release test.
- **Concurrency and capacity.** Eight sequence slots is the configured limit of the three single-node release profiles. The 27B DFlash2 profile reserves activation memory for images and reports KV capacity equivalent to about five 262144-token contexts. Full-length requests are also constrained by recurrent state and the live prompt-plus-output budget. Flash-Next 64K C8 is absent from the published benchmark matrix. Long incoming prefills can briefly delay active decode.
- **Non-greedy sampling.** At 8K C1 the measured step cost rises by about 2.7 ms for 27B MTP4 and 3.5 ms for Flash-Next at temperature 0.6 versus greedy. A fresh engine can take roughly three extra seconds on its first nonzero-temperature request while the sampling kernel compiles.
- **Hardware and precision.** The release matrix covers the 32GB V100 topology above. Checkpoints without compatible MTP weights need another decode configuration. E4M3, FP16 and E5M2 KV paths have different admission and memory contracts.

[Released limitations](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [Measured TTFT and decode](http://156.226.174.161:48152/1.5.3/#data)

## Open Follow-ups at This Snapshot

| PR | Reported scope |
| --- | --- |
| [#1508](https://github.com/1CatAI/1Cat-vLLM/pull/1508) · [#1507](https://github.com/1CatAI/1Cat-vLLM/pull/1507) | Engine FP8 policy for fused GDN verification and bias in SM70 Marlin |
| [#1506](https://github.com/1CatAI/1Cat-vLLM/pull/1506) · [#1504](https://github.com/1CatAI/1Cat-vLLM/pull/1504) | SM70 regression test maintenance and compile-safe capability cleanup |
| [#1503](https://github.com/1CatAI/1Cat-vLLM/pull/1503) · [#1502](https://github.com/1CatAI/1Cat-vLLM/pull/1502) · [#1501](https://github.com/1CatAI/1Cat-vLLM/pull/1501) | Triton sampling cutoffs, cold-warmup CUDA stack, compiled Flash-Next MTP context |
| [#1497](https://github.com/1CatAI/1Cat-vLLM/pull/1497) · [#1495](https://github.com/1CatAI/1Cat-vLLM/pull/1495) · [#1486](https://github.com/1CatAI/1Cat-vLLM/pull/1486) | Cache-drift recovery, alternative tool parsing and tiered KV retention |
| [#1488](https://github.com/1CatAI/1Cat-vLLM/pull/1488) · [#1271](https://github.com/1CatAI/1Cat-vLLM/pull/1271) | Another community 8×V100 DeepSeek profile and Gemma-4 D=512 support |

These proposals were open at the November 14, 2026 audit snapshot. The release index lists merged work. [Open PRs](https://github.com/1CatAI/1Cat-vLLM/pulls) · [Release PR index](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/PR_INDEX.md)

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
| Reused compiled subgraphs | First startup **271.26 s**, matching-config restart **95.30 s** | [v1.5.1](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1) · two historical observations within v1.5.1 |

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

![1Cat-vLLM WeChat Group 8 QR code](docs/assets/wechat-group-qr-8-20261005.png)

---

# ❤️ Acknowledgements

1Cat-LLM builds on the work of the broader open-source inference ecosystem, including vLLM, NVIDIA CUDA, FlashAttention, CUTLASS/TurboMind-related kernels, model authors, quantization projects, and contributors whose work is referenced in individual PRs and source files.

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)
- [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
- [marlin_v100](https://github.com/zhinianqin/marlin_v100)
- [v100-skinny](https://github.com/dnv2003/v100-skinny) — QPN quadpair-N `m8n8k4` decode layout behind the SM70 QPN2 / QPN4 / QPN8 and MXFP4-QPN kernels (MIT; notice retained in `csrc/sm70_turbomind/ops/LICENSE.v100-skinny`)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and the vllm-gguf-plugin bridge — pinned GGUF source provenance is recorded in the [source manifest](csrc/quantization/gguf_upstream/source_manifest.json), with notices retained in the source and wheel.

Special thanks to [@yangzhuxinyzx](https://github.com/yangzhuxinyzx) and `@1CatTCat` for their outstanding contributions to the continued evolution and performance breakthroughs of **1Cat-LLM**.

Where external implementations or algorithms are adapted, provenance and license information should be preserved in the corresponding source and PR history.

---

# License

Please refer to the repository license and the licenses of bundled or adapted third-party components.
