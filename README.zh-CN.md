<!-- markdownlint-disable MD041 -->

<p align="center">
  <img src="./assets/1cat-llm-logo.png" alt="1Cat-LLM 标志" width="960">
</p>

# 1Cat-LLM

[English](README.md) · **简体中文**

## 让 Volta 重获高速推理能力

### 面向 NVIDIA Tesla V100 / SM70 的现代大模型推理

> **1Cat-vLLM 正在向 1Cat-LLM 演进。**
>
> **扩大模型与运行时覆盖，提升算子效率，提供更多量化选择。**

项目从加速 V100 上的现代模型起步，下一阶段会扩展模型和运行时支持，并改进执行效率与权重打包、量化方式。

新增的 **[GGUF 支持](#gguf-support)** 为受支持的架构引入了更多社区量化模型，便于按模型大小、内存占用和质量选型。新名称延续项目在 Volta 上的工程积累。

**更名仍在计划中。** 过渡期间，仓库链接、发布文件名和安装命令沿用现有名称。

**正式发布 · [v1.5.3](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · 2026 年 11 月 14 日**

> Tesla V100 发布于 2017 年。
>
> 它的 Tensor Core 仍可用于现代大模型推理。
>
> **持续针对 SM70 优化的软件栈能发挥这些算力。**

1Cat-LLM 源自 1Cat-vLLM，是一个以 **NVIDIA Volta / SM70 / Tesla V100** 为主要优化目标的 vLLM 工程分支。

项目从让新模型在 V100 上启动做起。

> “最新模型可以在 V100 上运行。”

项目继续优化完整模型的服务性能。

> **让现代模型在 V100 上跑得快。**

# 演示与 v1.5.3 性能数据

Qwen3.8-27B NVFP4 使用原生 MTP4，在 **8K、8 并发**下实测总吞吐为 **892 tok/s**。v1.5.3 测试使用 **4× V100-SXM2 32GB**（NVLink、TP4），显存频率 877 MHz，功耗上限 300 W。 [v1.5.3 测试报告](http://156.226.174.161:48152/1.5.3/)

| v1.5.3 负载 · 温度 0 | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| 8K · C8 · 实测总吞吐（tok/s） | 479 | 892 | 408 |
| 8K · C8 · 按 70% 草稿接受率折算的总吞吐（tok/s） | 1,120 | 1,072 | 584 |
| 64K · C4 · 实测解码轮耗时（ms） | 32.68 | 24.20 | 34.03 |
| HumanEval · C1 · 实测吞吐（tok/s） | 545 | 370 | 276 |

HumanEval 使用 20 道题，允许输出自然结束。C1、C4、C8 分别表示 1、4、8 个并发请求。

Bilibili 视频展示较早的 **v1.5.2**。 [观看演示](https://www.bilibili.com/video/BV1ouWy67Esz/)

发布 wheel 和 Git 标签均对应 [`cb97f379b`](https://github.com/1CatAI/1Cat-vLLM/commit/cb97f379b67eb9f5703709db961d9dabcdecbcff)。 [英文发布说明](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/RELEASE_NOTES.en.md) · [中文发布说明](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/RELEASE_NOTES.zh-CN.md) · [发布 PR 索引](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/PR_INDEX.md)

---

# 📊 性能优先

SM70 Flash-V100 将 `--kv-cache-dtype fp8` 解析为 E4M3，27B 发布配置显式选用 `fp8_e4m3`。满足条件时，DFlash2 校验使用 FP32 Attention 状态和 FP32 logits。

Python 包和 Flash-V100 扩展应来自同一次构建。最初的 E4M3 修复要求精度修订版本 4，当前多头及按请求组织的路径还要求更新的原生能力。 [精度约定](docs/design/sm70_dflash2_fp32_defaults.md) · [并发长上下文验证](docs/design/sm70_dflash2_long_batch_20260927.md)
下文涉及 E5M2 或 FP16 部分累加的历史性能结果均沿用原始配置。
当前精度默认值对应的测量结果见 v1.5.3 表格。

## v1.5.3 · 发布 wheel 解码测试

**硬件与构建。** 测试机为 4× V100-SXM2 32GB（NVLink、TP4），报告主机内存 62 GB，开启 ECC，显存 877 MHz，核心最高 1530 MHz，功耗上限 300 W。wheel 基于 `cb97f379b` 构建，使用 CUDA 12.8。三套配置均用 FP16 计算，最多 8 个调度序列，prefill 预算为 4096 token，服务上下文上限为 262144 token。27B 使用 **Unsloth** NVFP4 权重，Flash-Next 使用 NVFP4、FP16 Attention KV 和主机内存中的 FP8 E4M3 QSA 历史。KV 和草稿设置见下文启动命令。报告标明了 27B 目标模型系列，启动配置固定了 DFlash2 草稿版本。目标模型的精确修订、合成输入 ID、重复次数、输出上限和完整的非贪心采样参数未公开。 [发布说明](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [完整测试明细](http://156.226.174.161:48152/1.5.3/#data)

**合成输入解码实测。** 单元格为温度 0 时的 **每轮耗时（ms）/ 报告中的总吞吐（tok/s）**。负载由合成长文本和短问题组成，只计所有请求同时处于解码的窗口，排除 prefill。总吞吐直接取自报告的实测列，单路吞吐采用独立统计，乘以并发数可能得到不同的总量。报告未提供底层事件日志来对齐两种统计窗口。

| 输入长度 | 并发请求 | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- | --- |
| 8K | 1 | 13.87 / 243 | 12.99 / 229 | 16.93 / 141 |
| 8K | 2 | 17.33 / 338 | 15.23 / 329 | 25.92 / 180 |
| 8K | 4 | 23.80 / 461 | 18.93 / 561 | 32.13 / 228 |
| 8K | 8 | 42.23 / 479 | 28.39 / 892 | 51.90 / 408 |
| 64K | 1 | 16.08 / 204 | 14.72 / 204 | 17.76 / 136 |
| 64K | 2 | 21.68 / 264 | 17.70 / 315 | 27.85 / 181 |
| 64K | 4 | 32.68 / 304 | 24.20 / 552 | 34.03 / 265 |
| 64K | 8 | 59.76 / 322 | 38.61 / 720 | — |

**按 70% 草稿接受率折算。** 单元格为**单路 / 合计 tok/s**，C1 只列一个值。报告假定 DFlash2 q7 每轮产出 5.9 个 token、MTP4 每轮产出 3.8 个 token，各自除以实测轮耗时，单路取整后乘以并发数。折算对所有负载使用同一接受率，排除 prefill。破折号表示缺测。报告未列出 64K 温度 1.0 的数据和 Flash-Next 64K C8 数据。

| 输入长度 | 配置 | 温度 | C1 | C2 | C4 | C8 |
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

v1.5.3 使用默认显存频率和功耗上限，上一版采用了更高的硬件设置。完整报告列出各档的实测单路吞吐、p90 轮耗时、每轮产出 token 数和 TTFT。 [发布说明](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [完整测试明细](http://156.226.174.161:48152/1.5.3/#data)

## 长上下文 Attention：17.92 → 47.1 → ≈60.8 → ≈71 TFLOP/s

| 阶段 | 来源 | 因果 Attention 有效算力 | 备注 |
|---|---|---:|---|
| 此前生产路径 | v1.2.2 时期基线 | **17.92 TFLOP/s** | V100 长前缀 Attention 基线 |
| D256 Split-D / N32 | [v1.3.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.3.0) | **46.63–47.1 TFLOP/s** | ≈2.6× 相对于此前生产路径 |
| GQA 打包的宽 QK/PV | [PR #286](https://github.com/1CatAI/1Cat-vLLM/pull/286) · 历史测量 | **≈60.8 TFLOP/s** | 将 6 个 GQA 头打包成更宽的 Tensor Core GEMM |
| FP32 QK/PV · 更宽的本地头布局 | [PR #666](https://github.com/1CatAI/1Cat-vLLM/pull/666) · Q8192/KV262144 | **69.982–71.111 TFLOP/s** | 单张 V100 上重放 TP1/TP2/TP4 本地布局，包含头分组复制 |
| 早期实验上限 | [PR #315](https://github.com/1CatAI/1Cat-vLLM/pull/315) | **≈79 TFLOP/s** | 历史候选路径，生产默认配置另有质量验证 |

在同代 GPU 上，典型长上下文 Attention 的有效算力由 **17.92 增至约 60.8 TFLOP/s**，约为原来的 **3.4×**。

这里统计因果 Attention 中 QK/PV 的有效算力，整模型吞吐见后文。

TP4、TP2、TP1 本地头布局的 FP32 QK/PV 记录分别为 **71.111 / 70.221 / 69.982 TFLOP/s**，均为单张 V100-SXM2 32GB 的算子结果。早期 75–77T 数据测于 QK FP32 修复完成前。同源码的 185W 审计测得 53.816T。 [形状覆盖](docs/design/sm70_tp_shape_coverage.md) · [质量限制](docs/design/sm70_tp_quality_audit.md)

---

# 🚀 真实模型基准

## v1.5.3 · 六组真实题目

表中为**单请求实测 tok/s / 草稿接受率**。温度为 0，每组 20 道题，允许自然结束。报告另有 C4 和温度 0.6 的吞吐数据。C4 的数据集计时包括各请求先后完成的时段。下表列生成速度，正确率见质量章节。

| 题目集 | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| HumanEval | 545 / 84% | 370 / 90% | 276 / 91% |
| MBPP | 486 / 80% | 340 / 85% | 261 / 84% |
| MATH500 | 473 / 76% | 336 / 82% | 255 / 82% |
| GSM8K | 464 / 78% | 335 / 84% | 251 / 81% |
| MT-Bench | 336 / 42% | 289 / 59% | 226 / 60% |
| 中文写作 | 189 / 22% | 224 / 46% | 168 / 44% |

题目类型会改变吞吐。在同一发行版中，DFlash2 在 HumanEval 子集上更快，原生 MTP4 在中文写作子集上更快。 [发布说明](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [完整测试明细](http://156.226.174.161:48152/1.5.3/#data)

## 历史整模型实测

下表保留原始 PR 中的整模型、API、纯解码和投机解码记录，各行沿用自己的测试条件。v1.5.3 发布 wheel 的结果见上文。

| 模型 | 硬件与运行时 | 负载 | 实测结果 | 来源与状态 |
|---|---|---|---:|---|
| **Qwen3.6-27B-AWQ + MTP4** | 4× V100 · TP4 · E5M2 KV · Flash-V100 · CUDA Graph | 64K 解码 | **100.564 tok/s** | [v1.2.2 发布](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2) · AL 4.981 / 99.52% |
| **Qwen3.6-27B-AWQ + MTP4** | 同上 | 128K 解码 | **85.258 tok/s** | [v1.2.2 发布](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2) · +87.64% 相对无 MTP |
| **Qwen3.6-27B-AWQ + MTP4** | 同上 · 上限 256K | 261,888-token 上下文解码 | **49.772 tok/s** | [v1.2.2 发布](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.2.2) · AL 5.000 / 100% |
| **Qwen3.6-35B-A3B NVFP4** | 4× V100 · TP4 · 混合 FP8 + W4A16_NVFP4 | 4096 / 1024 · 无 MTP | **116.99 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) |
| **Qwen3.6-35B-A3B NVFP4 + MTP4** | 同上 | 匹配条件下的 MTP4 | **174.76 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) · 1.49× 无 MTP |
| **Qwen3.8-27B-NVFP4** | 4× V100 · TP4 · E4M3 KV · 全量 CUDA Graph · 无 MTP | 固定 128K 解码 | **61.834 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · measured |
| **Qwen3.8-27B-NVFP4** | 同上 | 固定 256K 解码 | **50.376 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · measured, not projected |
| **Qwen3.8-27B-FP8** | 4× V100 · TP4 · E5M2 KV · 无 MTP | 128K 解码 | **50.68 tok/s** | [#212](https://github.com/1CatAI/1Cat-vLLM/pull/212) 发布路径扫描测试 |
| **Qwen3.8-27B-FP8** | 同上 | 256K 解码 | **41.11 tok/s** | [#212](https://github.com/1CatAI/1Cat-vLLM/pull/212) 发布路径扫描测试 |
| **Qwen3.8 Flash-Next-NVFP4** | 4× V100 · TP4 · V2 · 全量 CUDA Graph · 无 MTP | 8K / 512 纯解码 | **80.732 tok/s** | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · 已审计质量 |
| **Qwen3.8 Flash-Next-NVFP4 + MTP4** | 4× V100 · TP4 · V2 | 最终冷 JIT 验收 | **138.26 tok/s** | [#389](https://github.com/1CatAI/1Cat-vLLM/pull/389) · AL 4.943 / 98.57% |
| **Qwen3.8-27B-NVFP4 + DFlash2** | 4× V100 · TP4 · 生产 API | 历史网页提示词 · 512 output | **206.06 tok/s 流式解码** | [#422](https://github.com/1CatAI/1Cat-vLLM/pull/422) · 17.463 ms/round · 3.599 产出 token/轮 |
| **Qwen3.8-27B-NVFP4 + DFlash2** | 4× V100 · TP4 · 实际 API | MBPP 第 28 题 · 自然 EOS | **251.60 tok/s** | [#288](https://github.com/1CatAI/1Cat-vLLM/pull/288) · AL 4.686 · EvalPlus 1/1 |
| **Qwen3.8 DFlash2 + 自适应查找 q16** | 4× V100 · TP4 · 显式启用查找增强 | 重复上下文样本 | **316.27 tok/s** | [#366](https://github.com/1CatAI/1Cat-vLLM/pull/366) · 3.162 ms TPOT · 特殊显式启用配置 |
| **DeepSeek-V4-Flash** | **8× V100** · TP8 · FP8 dense + MXFP4 experts · CUDA Graph · 无投机解码 | 1024 / 256 | **15.357 ms TPOT ≈ 65.1 tok/s** | [#181](https://github.com/1CatAI/1Cat-vLLM/pull/181) · 已验收的无 MTP 基线 |
| **DeepSeek-V4-Flash** | **8× V100** · PP2×TP4 · 无 DSpark | 通过质量检查的组合端点 | **73.613–73.646 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) |
| **DeepSeek-V4-Flash** | 同上 · PP2×TP4 严格对照 | 数据集质量配对测试 | **73.539 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) · GSM8K 64/64 · HumanEval 29/32 |
| **DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XXS + DSpark5** | **8× V100** · PP2×TP4 · MRV2 · FULL 计算图 | 32K 提示词 / 1024 · 代码、JSON、法律、写作 | **200.5 · 206.6 · 116.5 · 90.1 tok/s 解码** · 1,900 tok/s prefill (2,200 在 128K 时) | [#1380](https://github.com/1CatAI/1Cat-vLLM/pull/1380) · 已合并的社区结果说明 · [原始数据](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/docs/measurements.md) |
| **GLM-5.3-Flash-NVFP4** | 8× V100 · TP4/PP2 · E4M3 KV · 无 MTP | 1K / 256 解码 | **53.016 tok/s** | [#402](https://github.com/1CatAI/1Cat-vLLM/pull/402) · 已合并的历史质量审计 |

---

# 🧪 数据集质量与吞吐评测

1Cat-LLM 同时记录**整模型吞吐、数据集得分、自然结束率、输出有效性和投机草稿接受率**。

## v1.5.3 · 贪心解码质量与输出检查

维护者使用同一批题目和贪心解码，对照 27B 原版 BF16。40 道代码题按单元测试通过数计分。变化量及 95% 置信区间以 BF16 为基准，单位为百分点，取自发布报告。公开报告提供汇总数据，未附逐题输出或置信区间的计算过程。

| 评测 | 27B BF16 | 27B NVFP4 | Flash-Next NVFP4 |
| --- | --- | --- | --- |
| GSM8K · 300 | 285/300 (95.0%) | 290/300 (96.7%) · +1.7 [0.0, +3.7] | 289/300 (96.3%) · +1.3 [−1.3, +4.0] |
| MMLU · 400 | 336/400 (84.0%) | 331/400 (82.8%) · −1.3 [−3.3, +0.8] | 349/400 (87.3%) · +3.3 [+0.5, +6.3] |
| MBPP + HumanEval · 各 20 道 | 21/40 (52.5%) | 21/40 (52.5%) · 0.0 | 20/40 (50.0%) · −2.5 [−7.5, 0.0] |

- 在 27B NVFP4 与 27B BF16 的 token 选择发生分歧处，BF16 对自身选择的优势中位数为 **0.20 logprob**，NVFP4 所选 token 均在 BF16 的前五位。Flash-Next 属于不同模型，报告仅比较其任务得分。
- 评测输出记录为 **0 次乱码、0 次重复循环**。测试覆盖的缓存命中和冷算输出逐字符一致。报告为 27B NVFP4 给出一列汇总质量结果，未分别列出 DFlash2 和 MTP4 的得分。
- 干净 Docker 环境分别覆盖 27B DFlash2 的 45 项 API 检查（含图片输入）和 Flash-Next 的 38 项检查。独立进行的 20 分钟压力测试中，两者完成/取消分别为 **2864/544** 和 **958/196**，均记录为零错误。 [发布验证](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3)

下文的 DFlash2 编程与 PPL 验收沿用当时的测试设置，部分测试使用 E5M2 KV。

## Qwen3.8-27B-NVFP4 + DFlash2 · 16K 编程质量验收

测试配置如下。

- 4× V100, TP4
- NVFP4 目标模型
- 官方 BF16 DFlash2 草稿模型
- 目标模型使用 FP8 E5M2 KV
- FlashAttention-V100
- 全量 CUDA Graph
- 前缀缓存
- Mamba align
- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- `xhigh` 推理强度
- 16K 输出上限，允许自然 EOS
- 三个预先指定的采样种子

| 数据集 | 样本 | Base 得分 | Plus 得分 | 自然结束 | 总输出吞吐 | 稳态解码均值 | 接受长度：汇总 / 按请求 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MBPP / EvalPlus** | 96 · 93 条计入评分 | **89/93** | **80/93** | 95/96 | **213.539 tok/s** | **236.902 tok/s** | 4.061 / 4.318 |
| **HumanEval / EvalPlus** | 96 | **94/96** | **92/96** | 91/96 | **208.978 tok/s** | **245.645 tok/s** | 3.972 / 4.476 |

参见 `docs/design/sm70_dflash2_quality_audit.md`。 [PR #346](https://github.com/1CatAI/1Cat-vLLM/pull/346)

第一个种子沿用历史目标模型直解配置的请求种子。两条路径在 MBPP 与 HumanEval 上的得分如下。

```text
Base : 62 / 63
Plus : 59 / 63
```

达到 200+ tok/s 的投机路径也完成了任务质量验收。

### 输出触及长度上限

有些代码题在长推理耗尽 **16K 输出预算**时停止，需单独检查最终代码是否完整、可执行。

在保留的 MBPP 和 HumanEval 测试中，**192 条输出中有 6 条触及 16K 上限**，其中 **3 条仍可提取出正确答案**。

README 分别记录以下指标。

- 可执行任务得分
- 自然结束率
- 因输出达到上限而失败的案例
- 吞吐

触及输出上限的样本需要结合最终代码判断。

## 可选的精确编程配置

使用同一 DFlash2 引擎和三个种子中的中间值，只将客户端采样改为模型的精确编程配置。

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
```

| 数据集 | 温度 1.0 | 温度 0.6 |
|---|---:|---:|
| MBPP | Base 27/31 · Plus 24/31 | **Base 29/31 · Plus 27/31** |
| HumanEval | Base 31/32 · Plus 31/32 | **Base 31/32 · Plus 31/32** |

80 个请求的结果如下。

```text
稳态解码均值：
233.187 → 244.520 tok/s

按输出 token 与解码时间计算的吞吐：
195.817 → 201.852 tok/s

按请求平均的接受长度：
4.27345 → 4.51356
```

自然结束由 72/80 降至 70/80，因此**精确编程配置需显式启用**。

---

## 其他整模型质量验收

| 模型与路径 | 数据集与质量 | 记录条件下的实测吞吐 | 状态 |
|---|---|---:|---|
| **Qwen3.8 Flash-Next-NVFP4 · 无 MTP** | GSM8K **15/16 raw · 15/16 严格评分** · 16/16 自然结束 | **80.935 tok/s 加权纯解码** | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · 已合并，已审计质量 |
| **Qwen3.8 Flash-Next-NVFP4 · MTP4** | HumanEval8 **8/8 条语义执行通过** | **150.17 tok/s 加权纯解码** | [#398](https://github.com/1CatAI/1Cat-vLLM/pull/398) · 已合并，历史有限验收 |
| **Qwen3.6-35B-A3B NVFP4 + MTP4** | GSM8K **122/128 (95.3125%)** · 0 条无效 · 0 条重复 | 匹配条件下的 MTP 运行 **174.76 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) · 已合并 |
| **Qwen3.6-35B-A3B NVFP4 + MTP4** | ShareGPT16 最终提交负载 | **120.096 tok/s 纯解码** · **97.678 端到端（E2E）输出 tok/s** · 241.973 prefill 吞吐（tok/s） | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) · 已合并 |
| **DeepSeek-V4-Flash · PP2×TP4** | GSM8K **64/64** · HumanEval **29/32** · LongBench **44.740** | **73.539 tok/s median** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) · 严格质量对照 |
| **DeepSeek-V4-Flash · PP2×TP4** | 组合路径端点： GSM8K **62/64** · 0 条无效，输出连贯 | **73.613–73.646 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) |
| **DeepSeek-V4.1-Flash-1017 · PP2×TP4 · IQ2_XXS** | MMLU-Pro **79.9 %** (700) · HumanEval+ **147/164** plus · LegalBench **0.716** (22 个任务) · 原版 top-1 一致率 **89.0 %** | **182.8 tok/s 128K 代码解码** | [#1380](https://github.com/1CatAI/1Cat-vLLM/pull/1380) · 历史社区评测 |
| **GLM-5.3-Flash-NVFP4 · 无 MTP** | Max 推理： **6/8** 个任务在 4096 输出 token 内完成， 低推理强度的针对性代码复测 **2/2** AST 与执行 | **53.016 tok/s 解码** · **266.040 tok/s 1K prefill** | [#402](https://github.com/1CatAI/1Cat-vLLM/pull/402) · 已合并，历史质量矩阵 |

---

## 工具调用与结构化输出

DFlash2 与自适应查找的组合也接受了工具调用和结构化输出测试。

| 验收项 | 结果 |
|---|---:|
| **BFCL** | **29/32** |
| **ToolACE** | **12/12** |
| **NexusRaven** | **13/16** |
| **严格 JSON Schema** | **7/8** |
| 结构化输出 B1 | **12/12** |
| 结构化输出 B4 | **12/12** |
| 长前缀状态隔离 | **5/5** |

质量结果与此次审计中的目标模型直解和 q7 对照一致。

同一阶段的运行时测试如下。

```text
普通 q7 → 自适应 q8：
168.52 → 170.98 tok/s

重复上下文 q16：
316.27 tok/s
3.162 ms TPOT
```

q16 的数据来自重复上下文且命中查找的特定负载。

参见 [PR #366](https://github.com/1CatAI/1Cat-vLLM/pull/366)。

---

## 分布与 PPL 验收

DFlash2 还会核查目标模型输出分布。

使用 WikiText 的 8 个固定片段，每段 2,048 token，共 **16,376 个受评分的输入 token**。

```text
目标模型直解 PPL : 5.4993116
DFlash2 PPL     : 5.4993622
绝对变化  : +0.0000506
相对变化  : +0.00092%
单片段最大差值   : 0.0062143
```

这项验收用于识别任务答案看起来正常、投机校验却已系统性改变目标分布的情况。

---

# ⚡ DFlash2 如何达到 200+ tok/s

仓库中有多组 **DFlash2 整模型**吞吐记录。

- 生产环境网页提示词：流式解码 **206.06 tok/s**，输出 512 token，平均每轮 17.463 ms。
- 高接受率 MBPP 请求：**251.60 tok/s**，接受长度 **4.686**，328 token 时自然 EOS，EvalPlus Base/Plus **1/1**。
- 自适应查找 q16：在显式启用且上下文重复的负载中达到 **316.27 tok/s**。
- v1.5.3 在 20 道 HumanEval 题上实测 **545 tok/s**，温度为 0，允许自然结束。 [v1.5.3 测试报告](http://156.226.174.161:48152/1.5.3/#data)

DFlash2 吞吐与草稿接受率、上下文重复程度、校验宽度、上下文长度及任务类型有关。

---

# 📏 256K 上下文的速度与缓存表现

## v1.5.3 · 长上下文单请求与前缀复用

温度为 0。表中为**实测每轮耗时（ms）/ 解码 tok/s**，吞吐按 70% 草稿接受率折算。

| 输入长度 | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| 1K | 13.54 / 436 | 12.62 / 301 | 16.47 / 231 |
| 8K | 13.87 / 426 | 12.99 / 293 | 16.93 / 224 |
| 32K | 14.77 / 399 | 13.76 / 276 | 17.05 / 223 |
| 64K | 16.08 / 367 | 14.72 / 258 | 17.76 / 214 |
| 128K | 18.51 / 319 | 16.29 / 233 | 19.08 / 199 |
| 250K | 23.23 / 254 | 19.59 / 194 | 19.71 / 193 |

下表为**实测首字延迟（TTFT，秒）**，依次记录冷输入、完全命中前缀缓存、在已有对话后追加 200 个输入 token。追加一轮时复用缓存中的输入和生成历史。

| 输入长度 | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | --- | --- | --- |
| 1K | 0.57 / 0.29 / 0.34 | 0.56 / 0.28 / 0.33 | 0.52 / 0.29 / 0.31 |
| 8K | 1.81 / 0.49 / 0.54 | 1.81 / 0.48 / 0.56 | 1.50 / 0.40 / 0.53 |
| 32K | 7.85 / 0.51 / 0.57 | 7.95 / 0.48 / 0.56 | 5.99 / 0.45 / 0.48 |
| 64K | 17.37 / 0.43 / 0.61 | 17.64 / 0.40 / 0.61 | 12.49 / 0.52 / 0.51 |
| 128K | 41.50 / 1.05 / 1.14 | 42.51 / 1.02 / 1.14 | 26.54 / 0.86 / 0.91 |
| 250K | 106.68 / 1.58 / 1.75 | 110.56 / 1.60 / 1.74 | 61.40 / 2.79 / 4.58 |

提示词和输出共用 262144-token 上限。KV 容量决定能同时驻留多少个长请求。 [发布说明](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [完整测试明细](http://156.226.174.161:48152/1.5.3/#data)

## 历史长上下文结果

[PR #285](https://github.com/1CatAI/1Cat-vLLM/pull/285) 在 TP4 上测试了 Qwen3.8-27B-NVFP4 的整模型长上下文解码。

```text
128K : 40.561 → 61.834 tok/s
256K : 27.456 → 50.376 tok/s
```

**256K 时的端到端实测为 50.376 tok/s。**

同一 PR 还给出了基于分解分析的 52.216 tok/s 推算值。

DeepSeek-V4 在完成集成后的整模型测试结果如下。

```text
TP8 无投机解码:
15.357 ms TPOT ≈ 65.1 tok/s

PP2×TP4 已通过质量验证的端点:
73.613–73.646 tok/s
```

---

# 🔬 已合并 PR 的精选基准

| 领域 | PR 与测试条件 | 对照 | 1Cat 结果 | 提升 |
|---|---|---:|---:|---:|
| DFlash2 q8 长上下文校验 | [#1505](https://github.com/1CatAI/1Cat-vLLM/pull/1505) · 64K / C4 · 发布前默认时钟 A/B | 37.14 ms/step | **32.64 ms/step** | **下降 12.1%** |
| 原生 MTP4 · 27B QUASAR NVFP4 | [#1491](https://github.com/1CatAI/1Cat-vLLM/pull/1491) · 源码版 A/B · V100 TP4 · E4M3 KV · 65,536 输入 / 256 输出 · C1，冷前缀，3 次中位数 | 18.82 ms/step | **14.34 ms/step** | **下降 23.8%** |
| D256 长输入 prefill Attention | [#198](https://github.com/1CatAI/1Cat-vLLM/pull/198) · Q4096/KV64K · Hq6/Hkv1/D256 | 87.6001 ms | **50.4504 ms** | **1.74×** |
| D256 长输入 prefill Attention | #198 · Q4096/KV8K | 11.1255 ms | **5.0542 ms** | **2.20×** |
| 128 位 E5M2 XQA 加载 | [#268](https://github.com/1CatAI/1Cat-vLLM/pull/268) · B16/17.8K 算子 | 0.743424 ms | **0.602112 ms** | **1.235×** |
| 128 位 E5M2 XQA 加载 | #268 · 不规则 B16/32K 算子 | 1.171296 ms | **0.925808 ms** | **1.265×** |
| 批量长上下文解码 | #268 · B16/16K 整模型纯解码 | 529.071 tok/s | **570.982 tok/s** | **+7.92%** |
| 长上下文解码路由 | [#206](https://github.com/1CatAI/1Cat-vLLM/pull/206) · 128K TP4 | 40.8208 tok/s | **48.5431 tok/s** | **+18.92%** |
| 长上下文解码路由 | #206 · 180K TP4 | 36.1387 tok/s | **42.5501 tok/s** | **+17.74%** |
| E4M3 XQA 长上下文解码 | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) · 固定 128K | 40.561 tok/s | **61.834 tok/s** | **+52.45%** |
| E4M3 XQA 长上下文解码 | #285 · 固定 256K | 27.456 tok/s | **50.376 tok/s** | **+83.48%** |
| 分组 QSA Page4 | [#387](https://github.com/1CatAI/1Cat-vLLM/pull/387) · 每层、每 rank | 55.151 ms | **9.632 ms** | **5.518×** |
| QSA 整模型 prefill | #387 · 64K | 4,446.64 tok/s | **5,777.43 tok/s** | **+29.93%** |
| 索引式 NVFP4 MoE prefill | [#390](https://github.com/1CatAI/1Cat-vLLM/pull/390) · 64K | 5,777.43 tok/s | **6,241.48 tok/s** | **+8.03%** |
| 精确的目标模型直解 | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) · 8K/512 · 无 MTP | 65.864 tok/s | **80.732 tok/s** | **+22.57%** |
| DFlash2 NVFP4 prefill | [#417](https://github.com/1CatAI/1Cat-vLLM/pull/417) · 32K/64K | 上一阶段记录 | **4069.25 / 3566.94 prefill 吞吐（tok/s）** | **+30.1% / +37.7%** |
| DeepSeek-V4 稀疏 MLA | [#163](https://github.com/1CatAI/1Cat-vLLM/pull/163) · 稀疏 MLA GPU 服务 | 46.920 ms/token | **4.392 ms/token** | **-90.64%** |
| DeepSeek-V4 TP8 无投机解码 decode | [#181](https://github.com/1CatAI/1Cat-vLLM/pull/181) · 8×V100 · 1024/256 | 19.342 ms TPOT 旧的 4K 图配置 | **15.357 ms TPOT ≈65.1 tok/s** | **TPOT 下降约 20.6%** |
| DeepSeek-V4 PP2×TP4 整模型 | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) · 8×V100 · 无 DSpark | — | **73.613–73.646 tok/s** | 已通过质量验证的端点 |
| DFlash2 并发长上下文解码 | [#697](https://github.com/1CatAI/1Cat-vLLM/pull/697) · 32K/256 · C4 / C8 | 326.832 / 492.025 tok/s | **429.134 / 624.863 tok/s** | **+31.30% / +27.00%** |
| Flash-Next 前缀缓存 prefill | [#754](https://github.com/1CatAI/1Cat-vLLM/pull/754) · 32K · 无前缀命中 | 3,046 tok/s | **5,068 tok/s** | **+66.38%** |
| Flash-Next NVFP4 MTP4 默认设置 | [#796](https://github.com/1CatAI/1Cat-vLLM/pull/796) · 31,744/256 · C1 | 79.832 tok/s | **92.199 tok/s** | **+15.49%** |
| 原始字节 IQ3_XXS/IQ3_S 配对 | [#966](https://github.com/1CatAI/1Cat-vLLM/pull/966) · M8/N4352/K5120 · 仅算子 | 86.016 μs | **57.344 μs** | **1.50×** |

各行沿用自己的对照基线。#697 使用 4× V100-SXM2 32GB、TP4、NVFP4 + DFlash2 q7、E4M3 目标 KV 和 FP16 草稿 KV，测量三个预热后的同时解码窗口，草稿接受率也有变化。#754 使用 Flash-Next NVFP4 / MTP4、FP16 KV，在开启缓存的情况下匹配未命中请求。#796 两组使用同样的 FP16 KV MTP4 配置，配对质量输出 **21/21 完全一致**，C4 提升 +1.05%，额外打包权重每 rank 约 1.25 GiB。#966 为单卡、固定时钟、冷 L2 的算子对照。发布 wheel 的性能以其独立测试为准。

---

# 🧠 128 位加载与访存效率

[PR #268](https://github.com/1CatAI/1Cat-vLLM/pull/268) 改进了页式 KV 分区内的访问方式。

实现包括以下改动。

- 复用 Page ID
- 合并两组 `half8` 转换
- 发起一次对齐的 128 位缓存加载
- 沿用 softmax、PV、分区边界与归约顺序

Nsight Compute（NCU）测量如下。

```text
L1 全局加载请求:
656,443 → 383,814
-41.53%

已执行 warp 指令数:
97,998,831 → 83,583,696
-14.71%

长 scoreboard 等待:
39.14% → 30.10%

每调度器就绪 warp:
0.55 → 0.65

B16 / 17.8K 算子耗时:
648.352 → 499.520 μs
-22.95%
```

DRAM 传输字节数基本持平。

提升来自**更少的碎片化加载、更低的地址计算和依赖开销，以及更连续的操作数供给**。

---

# ✅ 正确性与质量验收

1Cat-LLM 将吞吐与正确性结果一同记录。

代表性用例如下。

- 64K 整模型 A/B/A 的 64 个 token ID、文本及 SHA256 匹配，随机 paged KV、gathered-dense 和 Split-KV3 各有独立的数值验收。 [#198](https://github.com/1CatAI/1Cat-vLLM/pull/198)
- 均匀及不规则 B4/B8/B12/B16、page256/page800、12K–32K 的算子 A/B **逐位一致**。 [#268](https://github.com/1CatAI/1Cat-vLLM/pull/268)
- 128K 和 256K E4M3 XQA 的端点均完整生成 64 token，与相应对照流一致。 [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285)
- 结构化 API **24/24**、长交替前缀状态 **5/5**，覆盖多个种子的 MBPP/HumanEval，目标直解与 DFlash2 的 WikiText PPL 为 **5.4993116 / 5.4993622**。 [#346](https://github.com/1CatAI/1Cat-vLLM/pull/346)
- grouped QSA 重放确定，算术、中文和性能用例的 token 哈希与对照匹配。 [#387](https://github.com/1CatAI/1Cat-vLLM/pull/387)
- GSM8K 严格评分 **15/16**，自然结束 **16/16**，0 条触及长度上限，0 条结构无效。 [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415)
- 1.5.0 RC 隔离安装通过 `/v1/models`、`/metrics`、普通聊天、流式和非流式工具调用、JSON Schema 和重复前缀检查。10,017-token 前缀延迟从 **冷算 2.642 s 降至命中 0.164 s**。 [#427](https://github.com/1CatAI/1Cat-vLLM/pull/427)

---

# 🔥 FlashAttention-V100

## 让 FlashAttention 在 V100 上高效运行

## 为 Volta 重构数据流

FlashAttention 的优化重点是 **I/O 和调度**。

- 减少 HBM 往返
- 尽量让 Q/K/V 和中间状态保留在片上
- 提高复用率
- 减少中间结果的显式存储
- 减少同步屏障
- 持续向 Tensor Core 提供操作数

现代 FlashAttention 实现主要面向 Ampere、Hopper 及更新的 GPU。

Tesla V100 的计算架构是 SM70。

以下硬件能力来自后续架构。

- Ampere 的 `cp.async`
- Turing/Ampere 风格的 `ldmatrix` 数据路径
- Hopper 的 TMA
- 原生 FP8 Tensor Core
- Blackwell 的 FP4 Tensor Core

直接移植可以实现兼容，但数据供给往往跟不上 Tensor Core。

1Cat-LLM 围绕 Volta 现有的硬件能力重构执行路径。

---

# ⚙️ SM70 上的软件流水线与矩阵供给

V100 的实现使用 Volta 可用的指令与存储层次。

以下技术用于重叠访存和计算。

```text
LDG
STS
LDS
寄存器预取
双缓冲
共享内存重排
显式 HMMA fragment 映射
跨 tile、跨阶段软件流水线
```

流水线的目标如下。

```text
重叠访存与计算
        ↓
提高片上复用
        ↓
缩短依赖链
        ↓
减少屏障与重放
        ↓
持续为 HMMA 提供操作数
```

代表性做法包括以下几项。

- 寄存器预取与双缓冲
- 在当前 QK 计算期间加载下一块 K
- 在 HMMA 执行时预先准备 PV 操作数
- 按阶段重排共享内存布局
- 128 位向量化访存
- 明确分配 QK/TN 与 PV/TT 的 HMMA fragment
- 跨 tile 和阶段进行软件调度

> **该路径使用 Volta 自有指令。**
>
> **通过软件调度重叠访存与计算。**

---

# 第一层：高效搬运 KV Cache

页式 KV 将逻辑 token 映射到物理页。

朴素的 SM70 路径会重复执行以下操作。

```text
加载 Page ID
计算地址
加载较窄的 FP8 fragment
转换
重复
```

重复寻址、依赖等待和标量访存都会占用周期。

[#268](https://github.com/1CatAI/1Cat-vLLM/pull/268) 的 128 位 XQA 复用页元数据，并成对执行对齐加载。

代表性整模型批量测试如下。

```text
B16 / 16K:
529.071 → 570.982 tok/s
+7.92%
```

典型长上下文 XQA 形状的对应算子提升约为 **21%–26.5%**。

---

## 第二层：将 D=256 Attention 改为 Volta 原生流水线

降低数据搬运开销后，进一步重构 Attention 的计算主体。

主要组成如下。

## D256 Split-D

将 D=256 切分为四个 D64 片段。

成对 warp 共享 QK 概率计算，提高 PV 并行度并避免重复计算 QK。

## N32 在线 Softmax

保留因果在线 softmax 和 FP32 累加，无须生成完整分数矩阵。

## K 阶段双缓冲

在共享内存阶段间交替使用 K/D64 面板，减少屏障与等待。

## Split-KV3

在适合的形状上将长前缀 KV 分成三个分区，再合并 FP32 部分状态。

## GQA 多头打包

将六个 GQA query 头打包成更宽的 Tensor Core 计算。

## 加宽 QK / PV

将零散的小型 Tensor Core 操作合并为更大、更规则的 QK/PV GEMM 式计算。

## 分离前缀与因果尾部

单独调度完全可见的长前缀和精确的因果尾部，再合并在线 softmax 状态。

这组优化从 [PR #198](https://github.com/1CatAI/1Cat-vLLM/pull/198) 和后续 D256 / Split-KV3 工作演进到 [v1.3.0](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.3.0)、[PR #286](https://github.com/1CatAI/1Cat-vLLM/pull/286)。

测量结果如下。

```text
17.92 TFLOP/s
    ↓
46.63–47.1 TFLOP/s
    ↓
≈60.8 TFLOP/s
```

硬件仍是同代 GPU，使用相同的 Tensor Core。

提升来自数据流和调度改进。

> 约 79 TFLOP/s 是实验性上限，生产默认路径有独立的精度验收。

后续工作将该数据流扩展到 **Q8000/Q8192、多请求及 TP1/TP2/TP4 本地头布局**，使用 FP32 QK/PV 累加。完整 FP32 的算子结果在记录条件下约为 **70–71 TFLOP/s**。后续修复覆盖短 chunk 的因果可见范围、分数峰值漏计、分数 tile 溢出及尾部中间值范围。 [#820](https://github.com/1CatAI/1Cat-vLLM/pull/820) · [#850](https://github.com/1CatAI/1Cat-vLLM/pull/850) · [#851](https://github.com/1CatAI/1Cat-vLLM/pull/851) · [#875](https://github.com/1CatAI/1Cat-vLLM/pull/875)

---

# 第三层：V100 原生稀疏 Attention

Qwen3.8 Flash-Next 的 QSA 同时涉及计算和访存。

运行时需要完成以下工作。

- 稀疏块选择
- 物理页映射
- Page4 K/V 复用
- 精确的逐行掩码
- 最终 QK/PV 计算

[PR #387](https://github.com/1CatAI/1Cat-vLLM/pull/387) 将相邻的八行 query 分组，同一 Page4 K/V 块只加载一次，每行保留精确的 4 位掩码。

QK 和 PV 直接使用 Volta WMMA。

代表性结果如下。

```text
旧 QSA 路径：
55.151 ms/层/rank

分组 Page4：
9.632 ms/层/rank
+0.362 ms 规划开销

Attention 加速：
5.518×
```

整模型纯 prefill 的提升如下。

```text
32K  : +32.36%
64K  : +29.93%
131K : +32.69%
```

---

# 🧩 依据性能剖析迭代优化

算子加速后，项目继续用整模型剖析定位瓶颈。

QSA 加速后，性能分析显示瓶颈转移到了 NVFP4 MoE prefill。

[PR #390](https://github.com/1CatAI/1Cat-vLLM/pull/390) 通过索引式 W13 执行去除 `[tokens × topK, hidden]` 输入展开瓶颈。

代表性结果如下。

```text
8K 算子链：
6.026752 → 4.235264 ms
1.423×

整模型纯 prefill：
32K  : 5998.65 → 6507.10 tok/s
64K  : 5777.43 → 6241.48 tok/s
131K : 5450.92 → 5871.47 tok/s
```

[PR #393](https://github.com/1CatAI/1Cat-vLLM/pull/393) 融合精确 FP16 SwiGLU，将 N320 W13 尾部拆为 N256+N64，省去无效的尾部 tile 计算。

项目的优化流程始于应用负载。

> **剖析整模型，优化当前瓶颈，再测量整模型。**

---

# 🎯 目标模型直解基准

目标模型直解是评估 DFlash2 和 MTP 加速之前的基础。

[PR #415](https://github.com/1CatAI/1Cat-vLLM/pull/415) 记录了 Qwen3.8-Flash-Next-NVFP4 在 4× V100 上的结果。

```text
8K 输入 / 512 输出
无 MTP
全量 CUDA Graph

对照:
65.864 tok/s
15.183 ms TPOT

候选方案:
80.732 tok/s
12.387 ms TPOT
```

以上为**目标模型直解吞吐**。

该路径还通过以下检查。

```text
GSM8K: 15/16 严格评分
自然结束: 16/16
自然结束样本加权解码: 80.935 tok/s
```

---

# ⚡ SM70 上的 DFlash2

传统自回归解码每产出一个 token 就要执行一次目标模型前向。

DFlash2 将多个 token 放在同一轮处理。

块扩散草稿模型提出多个后续 token，由目标模型批量校验。

服务循环如下。

```text
生成多个草稿候选
        ↓
目标模型校验一块
        ↓
接受多个 token
        ↓
目标模型每轮推进多个 token
```

面向发布的 Qwen3.8 DFlash2 SM70 实现还优化了以下部分。

- 草稿 Attention
- 候选选择器
- 分组校验
- GDN 元数据
- 稀疏拒绝采样
- NVFP4/QPN 路径
- 采样
- CUDA Graph;
- 前缀状态
- Mamba align;
- 工具调用与结构化输出状态

草稿 Attention 使用以下路径。

```text
FLASH_ATTN_V100
```

这些路径与草稿模型的窗口和页布局匹配。

---

# DFlash2 长上下文衰减

长上下文会放大草稿 Attention 中无效的历史扫描开销。

[PR #328](https://github.com/1CatAI/1Cat-vLLM/pull/328) 将无锚点的分页 prefill 循环起点移到草稿滑动窗口实际需要的第一个 tile。

256K 时的结果如下。

```text
草稿 Attention：
0.422912 → 0.246784 ms/层

按五层推算：
2.114560 → 1.233920 ms
```

候选方案的中位数如下。

```text
32K  : 0.252928 ms
128K : 0.243712 ms
256K : 0.246784 ms
```

这一草稿 Attention 组件在 32K 之后的耗时几乎不再随上下文增长而上升。

---

# 🧵 新请求 prefill 对并发解码的影响

活跃解码请求会与新到达的长提示词共享 GPU。

更新后的 SM70 路径结合批量 FP4/FP8 投影、联合草稿 Attention、共享 GDN 元数据、长上下文校验和采样工作区复用。算子和整模型各有独立的对照。 [#697](https://github.com/1CatAI/1Cat-vLLM/pull/697) · [#706](https://github.com/1CatAI/1Cat-vLLM/pull/706) · [#708](https://github.com/1CatAI/1Cat-vLLM/pull/708)

[PR #860](https://github.com/1CatAI/1Cat-vLLM/pull/860) 增加自适应混合 prefill 预算，保留常驻解码容量及混合前缀重放。测试部署使用 **27B QUASAR NVFP4 + DFlash2 / 4× V100-SXM2 32GB / TP4 / E4M3 / 256K 容量**，两个持续解码请求与两个新 32K 提示词共享 GPU。

| 新请求 prefill 期间的指标 | 对照 | 候选方案 |
|---|---:|---:|
| 常驻请求输出 | 6.88 tok/s | **30.34 tok/s** |
| 最长输出更新间隔 | 2.873 s | **0.310 s** |
| 第一个 / 第二个新请求首字延迟 | 14.287 / 20.631 s | **17.994 / 35.491 s** |

结果是两次预热后测量的中位数，使用 PR 中两对 NVLink 的拓扑。常驻请求关闭 EOS 以维持负载，新请求自然结束。**减少解码停顿会增加新请求的首字延迟。** 250 ms 是调度器的软目标。该历史 PR 使用自己的调度配置，v1.5.3 测试设置为 `--mixed-prefill-step-latency-ms 0`。

后续工作让满足条件的小 query 草稿 Attention 并行，并减少 TP4 all-reduce 加 RMSNorm 的同步。单请求收益以各自的精度、形状和拓扑为准。 [#944](https://github.com/1CatAI/1Cat-vLLM/pull/944) · [#957](https://github.com/1CatAI/1Cat-vLLM/pull/957)

---

# 🔢 量化与算子栈

许多现代大模型的权重格式晚于 V100。

1Cat-LLM 为这些格式实现加载、内存布局和原生算子。

SM70 相关工作包括以下部分。

- AWQ / W4A16
- TurboMind SM70 算子
- compressed-tensors
- FP8 E4M3 / E5M2 KV 存储
- ModelOpt NVFP4;
- MXFP4;
- 独立的 GGUF 元数据、分词器和架构适配器
- 使用原始 GGUF 量化块的仿射与码本算子，按设备和形状回退
- Quark W4A16 INT4 / UINT4;
- QPN8;
- QPN4;
- QPN2;
- 分组 MoE
- 针对具体形状的精确解码 GEMV
- SM70 自定义采样路径

权重格式需贯通加载和运行时。

> 支持解析该数据类型。

服务端目标如下。

> **让量化格式在 Volta 上对应可用的高性能推理路径。**

---

<a id="gguf-support"></a>

# 🗂️ GGUF：保留权重格式，重构执行路径

社区量化为部署提供了更多权重选择。

GGUF 将社区量化引入受支持的模型路径，便于在模型大小、内存和质量间取舍。1Cat-LLM 将 GGUF 接到原生算子及推理服务。

当前主分支可读取独立 GGUF 元数据、Qwen BPE 分词器及内嵌聊天模板。原生适配器覆盖 Qwen3.5 dense、Qwen3.5 MoE、Qwen4Exp / Flash-Next 与 DFlash2 草稿模型，并显式处理张量并行存储及混合张量类型。 [#809](https://github.com/1CatAI/1Cat-vLLM/pull/809) · [#876](https://github.com/1CatAI/1Cat-vLLM/pull/876) · [#888](https://github.com/1CatAI/1Cat-vLLM/pull/888) · [#895](https://github.com/1CatAI/1Cat-vLLM/pull/895)

算子在布局准备时保留原有量化信息，选择有测试依据的 TurboMind SM70 路径，并提供内置原生或参考实现作为回退。兼容的 IQ3_S、IQ4_XS、IQ3_XXS gate/up 配对可共享激活，沿用原始 checkpoint。

[PR #966](https://github.com/1CatAI/1Cat-vLLM/pull/966) 在 M8/N4352/K5120 下增加七组 IQ3_XXS/IQ3_S 混合配对，使用 FP32 点积和 split-K 归约。在该测试模型中，混合配对共覆盖 **18/40 层、48.463% 的混合配对源字节**。其他形状沿用常规分派。 [#968](https://github.com/1CatAI/1Cat-vLLM/pull/968) 的 Q4_K 记录覆盖无损存储准备。

**Flash-Next GSQ-RCO IQ3_S + FP16 MTP4** 的一组开发版结果如下。

| 负载 | 完整轮耗时 | 解码实测 |
|---|---:|---:|
| C1 · 8192 输入 / 256 输出 | **26.072 ms** | **92.267 tok/s** |
| C4 · 每请求 128 输入 / 1024 输出 | **52.059 ms** | **280.441 总计 tok/s** |

测试使用特定开发构建，部署为 **4× V100-SXM2 32GB / TP4**，使用 NVLink 环的直连边、FP16 激活和 KV、FP32 SSM 状态及 FULL 解码图。定长贪心计时忽略 EOS，重复三次。另有四条自然 EOS 的输出与原结果一致。更长的计时轨迹随版本变化，**15–17 ms 的单轮目标仍待达成**。 [#940](https://github.com/1CatAI/1Cat-vLLM/pull/940) · [完整测试记录](docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md)

适配器支持、算子准入与模型质量需各自验证。dense 和 DFlash2 适配器已有加载及生成检查。35B MoE 仍需更广的质量与吞吐验证。部署时结合标签对应的运行时、模型特定配置与质量证据。 [GGUF 验收规则](docs/design/gguf_v100_control.md)

---

# Qwen3.6-35B-A3B NVFP4

[PR #270](https://github.com/1CatAI/1Cat-vLLM/pull/270) 为混合 ModelOpt NVFP4 权重实现精确的 SM70 路径。

主要改动如下。

- FP8 稠密投影
- W4A16_NVFP4 路由与共享专家
- 分组 TurboMind MoE
- 保留重复专家槽位
- 混合精度 GDN 路由
- MTP 冷启动预热

同口径的无 MTP 对照如下。

```text
AWQ:
prefill 0.3813 s
解码 113.71 tok/s

NVFP4:
prefill 0.4216 s
解码 116.99 tok/s
```

MTP4 结果如下。

```text
174.76 tok/s
1.49× NVFP4 无 MTP
```

质量结果如下。

```text
GSM8K:
122/128
95.3125%

无效输出：
0

重复输出：
0
```

---

# DeepSeek-V4 在 V100 上运行

DeepSeek-V4 的优化覆盖稀疏 Attention 和整条推理链路。

SM70 相关工作主要包括以下部分。

- 稀疏 MLA
- FP8 稠密投影
- MXFP4 专家
- 分组 MoE
- Indexer;
- KPool;
- Q 归一化、RoPE、KV 写入
- 自定义 TP4 all-reduce
- PP2×TP4 执行
- 精确 GEMV 热路径

代表性结果如下。

```text
TP8 无投机解码:
≈65.1 tok/s

PP2×TP4 严格质量对照:
73.539 tok/s

PP2×TP4 组合端点：
73.613–73.646 tok/s
```

严格质量对照如下。

```text
GSM8K    : 64/64
HumanEval: 29/32
LongBench: 44.740
```

## DeepSeek-V4.1-Flash-1017 · 原生 PP4×TP4

[#1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482) 在 **4 台 Dell C4140** 上验证了原版 FP8 32×32 稠密权重、MXFP4 专家、Engram 与 DSpark5。每台配 4 张 V100-SXM2 32GB，节点间使用 InfiniBand。发布说明记录显存频率为 1053 MHz。源码构建配置的服务窗口为 **1,048,576 token**，公开报告中最长计时请求为 **512K**，每轮解码 23.65 ms，TTFT 约 133 s。原生 PP4 关闭前缀缓存，各请求重新 prefill。

| 1K 提示词，温度 0 | C1 | C2 | C4 | C8 | C16 | C32 | C64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 实测解码轮耗时（ms） | 19.3 | 19.7 | 24.8 | 35.1 | 49.5 | 77.5 | 135.7 |
| 发布报告推算总吞吐（tok/s） | 202 | 396 | 628 | 888 | 1264 | 1600 | 1856 |

表中换算使用约 3.9 个产出 token/DSpark 轮，各档的实际接受率有所不同。PR 另报告 1K、C64 为 **1809 tok/s** 总吞吐，单请求 HumanEval 为 **249 tok/s**。源码构建的测试还记录 **GSM8K 297/300**、**MMLU 349/400**、16K/128K/512K 的大海捞针 **9/9**，以及 20 分钟压力测试 **722 个请求、0 错误**。并发与单请求的 token 输出可能不同。 [PR 测试](https://github.com/1CatAI/1Cat-vLLM/pull/1482) · [发布启动报告](http://156.226.174.161:48152/1.5.3/#launch)

按 [C4140 handoff](benchmarks/deepseek_v41/native/HANDOFF.md) 在每个节点构建原生扩展，准备 checkpoint、Engram 数据和 DSpark 草稿。长上下文 YAML 见发布报告。启动前按集群实际情况设置模型路径、主节点地址、节点 IP、NCCL 网卡和 CPU 亲和性。标签中的 [run_pp4.sh](benchmarks/deepseek_v41/native/run_pp4.sh) 包含 C4140 专用路径与地址。适配脚本后，依次启动 rank 3、2、1、0，rank 0 为 API 节点。

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

# GLM-5.3 在 V100 上运行

GLM-5.3 的 SM70 路径使用以下组件。

- ModelOpt NVFP4 MoE;
- FP16 非专家权重
- FP8 E4M3 KV;
- TP4 / PP2;
- 稀疏 MLA
- 精确 KDA GEMV
- 融合 KDA f/g
- mHC;
- 自定义 all-reduce
- 全量解码 CUDA Graph

历史稳定性结果如下。

```text
解码：
53.013085
53.018516
53.017527 tok/s

均值：
53.016376 tok/s

平均 TPOT：
18.862097 ms
```

1K prefill 结果如下。

```text
266.039984 tok/s
```

质量审计发现，Max 推理模式可能在简短代码题上耗尽输出预算，针对性降低推理强度后可完成任务，并通过 AST 和外部执行检查。

上述 **53.016 tok/s** 来自历史无 MTP 测试。后续合并将 GLM DFlash2 接入调度器、TP8 校验和 PP2 KV 容量管理。 [#501](https://github.com/1CatAI/1Cat-vLLM/pull/501) · [#503](https://github.com/1CatAI/1Cat-vLLM/pull/503)

---

# 🎬 原生图像与视频工作流

仓库还提供原生 **MiniMax H3 音视频生成**和 **Z-Image Turbo / Base 图像任务**。H3 覆盖 SM70 Attention、张量并行、权重驻留、导出、任务状态和去噪进度。Z-Image 提供持久化异步任务、同步接口、取消和重启处理。 [#557](https://github.com/1CatAI/1Cat-vLLM/pull/557) · [#583](https://github.com/1CatAI/1Cat-vLLM/pull/583) · [#585](https://github.com/1CatAI/1Cat-vLLM/pull/585)

V100 工作流验证包括 1024×1024 的 Turbo/Base 图像，以及 H3 文本、关键帧、参考图像三种输入生成的 1344×768 视频（107 帧、24 fps、双声道音频）。覆盖范围以记录中的配方、模型和时长为准。 [原生创作任务](docs/design/native_media/CREATIVE_JOBS.md)

---

# 🧠 Volta 的实际推理性能

V100、A100、H100 和 Blackwell 的理论峰值各不相同。

实际推理速度还取决于软件栈。

许多现代推理软件已减少对 SM70 的专项优化。

两类差距分别来自硬件和软件。

```text
硬件代际差距
+
软件优化差距
```

1Cat-LLM 主要缩小软件一侧的差距。

典型 Attention 有效算力起点为

```text
17.92 TFLOP/s
```

随后增至

```text
46–47 TFLOP/s
```

再增至

```text
≈60.8 TFLOP/s
```

同一代硬件上的 27B 整模型 256K 解码实测达到

```text
50.376 tok/s
```

这些测量沿用 V100 硬件。

性能提升来自算子和运行时的改进。

> **让 Volta 的算力用于实际推理负载。**

---

# 📦 安装

[v1.5.3 release](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) 提供 **Linux x86_64、CPython 3.12、CUDA 12.8** wheel 和 Docker 构建包。宿主机需要兼容 CUDA 12.8 的 NVIDIA 驱动。wheel 固定 PyTorch 2.10.0+cu128 并自带 SM70 扩展。JIT 所需的环境包括 CUDA 12.8 Toolkit、`PATH` 上的 `nvcc` 及 C++ 编译器。TileLang 找不到 `nvcc` 时，FlashQLA GDN prefill 可回退到内置预编译算子。Docker 基于 Ubuntu 24.04，并包含两种编译器。 [GDN 回退 #1400](https://github.com/1CatAI/1Cat-vLLM/pull/1400)

wheel 内置原生库引用 **GLIBC 2.38** 和 **GLIBCXX 3.4.30**。宿主环境应提供这些版本，或者使用 Ubuntu 24.04 发布镜像构建。

## 硬件与模型权重

| 配置或路径 | 已验证硬件 | 权重与覆盖范围 |
| --- | --- | --- |
| 27B DFlash2 q7 | 4× V100-SXM2 32GB, NVLink, TP4 | [Unsloth Qwen3.8-27B NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 与 [固定版本 DFlash2 草稿](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/tree/dedf8df68adfb1afeaf7b7480c0a0243108177b4) · 发布 wheel 实测 |
| 27B 原生 MTP4 | 同上 · 4×32GB 发布配置 | 自带 MTP 的 Unsloth 27B NVFP4，在相同 wheel 上实测。QUASAR NVFP4 和 FP8 在 TP2/TP4 上有独立的 [PR 测试](https://github.com/1CatAI/1Cat-vLLM/pull/1491) |
| Flash-Next MTP4 | 4× V100-SXM2 32GB, NVLink, TP4 | [RadixArk Flash-Next NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) 是项目列出的权重。发布配置使用 MTP 草稿权重目录和主机内存 QSA KV |
| DeepSeek-V4.1-Flash-1017 | 4× Dell C4140, 共 16× V100-SXM2 32GB, PP4×TP4 | [原版权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), 原生 FP8/MXFP4, DSpark5 · 源码构建实测见 [#1482](https://github.com/1CatAI/1Cat-vLLM/pull/1482) |
| 其他模型路径 | 按模型与拓扑分别验证 | 见 Qwen3.6 AWQ/NVFP4、Qwen3.8 FP8、GGUF、DeepSeek-V4 和 GLM-5.3 章节。Turing SM75 有部分已合并路径，例如 [GDN prefill](https://github.com/1CatAI/1Cat-vLLM/pull/1228) 与 [FA2 构建改动](https://github.com/1CatAI/1Cat-vLLM/pull/1346). 发布测试覆盖 SM70 |

发布说明建议主机内存至少 **64 GB**。v1.5.3 的启动和内存数据在 TP4 测试机上测得，TP2 结果来自独立的源码构建。

| 实测资源指标 | 27B DFlash2 | 27B MTP4 | Flash-Next MTP4 |
| --- | ---: | ---: | ---: |
| 每卡启动后 / 峰值显存 | 27.2 / 28.6 GiB | 26.3 / 26.3 GiB | 29.9 / 30.6 GiB |
| 报告中的 KV 容量 | 1,308,270 token | 1,598,295 token | 378,652 token，历史在主机内存 |
| 宿主机启用已预热编译缓存时的启动耗时 | 125 s | 114 s | 342 s |
| Docker 空缓存启动 / 重启 | 323 / 122 s | — | 624 / 343 s |

[发布版资源和启动时间测量](http://156.226.174.161:48152/1.5.3/#data)

## 安装 wheel

安装 `uv` 和 `curl` 后，从同一个 release 下载 wheel 和校验文件。

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

启动模型前检查安装版本和可见 GPU。

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

wheel 的 SHA256 为 `0558021651c41ab111609d121cef029c09224de4b6e6f8fd362754dfdd84990f`，Docker 构建包的 SHA256 为 `2c8e43157735964eda8b7691f6cddf61efa06e1715b3dd6438eb252067ee3db1`。 [发布校验文件](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/SHA256SUMS)

## Docker 与 1Cat Studio

Docker 需要支持 CUDA 的宿主机驱动和 NVIDIA Container Toolkit。构建时会访问 Ubuntu、PyPI、PyTorch 和 NVIDIA 软件仓库。按下文命令使用的目录名，将模型目录挂载到 `/models` 下。Docker 预设要求为所选模型提供两个目录，`MODEL=27b` 对应的草稿目录为 `Qwen3.8-27B-DFlash2`，`MODEL=flashnext` 对应 `Qwen3.8-Flash-Next-mtp`。构建包内的 `README-docker.en.md` 记录了镜像源和环境选项。

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

Flash-Next 使用 `MODEL=flashnext`。Docker 启动脚本支持 `VLLM_API_KEY`、`MAX_NUM_SEQS` 和 `MAX_MODEL_LEN` 环境选项。使用 1Cat Studio 时，安装 **1.5.3** 运行时并在模型配置中选中。回滚可切换到旧版 Studio 运行时，在单独环境重装已校验的旧 wheel，或使用旧版 Docker 镜像及其对应缓存卷重启。 [发布部署与回滚](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3)

---

# ▶ Qwen3.8-27B-NVFP4 + DFlash2

## TP4 + E4M3 启动示例

以下命令采用发布版 DFlash2 配置，包括图片输入和 thinking 默认值。下载目标权重和固定版本的草稿权重，分别放入两个本地目录。目标模型使用 E4M3 KV，草稿使用 `auto` 与 FP16 计算。DFlash2 每轮提议七个 token，使用概率草稿采样。

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

wheel 自带脚本使用相同的 TP、KV、上下文和 DFlash2 核心配置。省略 `--draft` 时会下载固定版本草稿。脚本默认只接受文本，并关闭 thinking。要复现发布示例的前端设置，请使用上面的显式命令。 [配置源码](vllm/sm70_profiles/qwen38_27b_nvfp4_dflash2.json) · [启动脚本源码](scripts/serve_qwen38_27b_nvfp4_v100.sh)

```bash
serve_qwen38_27b_nvfp4_v100.sh /models/Qwen3.8-27B-NVFP4 --port 8000
python -m vllm.sm70_profiles show qwen38_27b_nvfp4_dflash2 --json
curl --fail http://127.0.0.1:8000/v1/sm70/acceleration
```

加速诊断接口返回已配置的能力、可用时的已加载算子实例及回退原因。核查具体请求走的路径时，可对照请求形状和运行时路由日志。 [加速诊断实现](vllm/sm70_profiles/acceleration.py)

---

# ▶ Qwen3.8-27B-NVFP4 + 原生 MTP4

使用包含 MTP 头的 27B NVFP4 权重。发布命令从目标权重生成四个贪心草稿 token，混合前缀缓存布局使用运行时默认值。共享加速校验器按本地分片布局与数据类型选择兼容路径。 [MTP 运行时](https://github.com/1CatAI/1Cat-vLLM/pull/1491) · [前缀缓存默认值](https://github.com/1CatAI/1Cat-vLLM/pull/1492)

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

wheel 自带脚本设置 `--max-num-seqs 8`、FP16 Attention KV、锁页主机内存中的 FP8 E4M3 QSA 历史以及 FP32 Mamba SSM 状态，使用贪心 MTP4 草稿。示例采用目标自带的 MTP 层，RadixArk 权重包含这些层。要按发布预设使用独立草稿，添加 `--draft /models/Qwen3.8-Flash-Next-mtp` 并提供兼容的权重目录。发布说明给出了目录名，未标注其公开修订版本。 [启动脚本源码](scripts/serve_flash_next_nvfp4_v100.sh) · [模型卡](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) · [发布启动配置](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3)

```bash
serve_flash_next_nvfp4_v100.sh /models/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen3.8-flash-next-nvfp4 --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --trust-remote-code --seed 0 --default-chat-template-kwargs '{"enable_thinking":false}'
```

## OpenAI 与 Anthropic API 示例

在 8000 端口启动一种配置。`model` 应与启动时的 `--served-model-name` 一致。以下示例使用 27B MTP4。服务端设置了 `--api-key` 时，请将客户端环境变量设为对应密钥，并添加 `-H "Authorization: Bearer $VLLM_API_KEY"`。

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

Anthropic 兼容的 Messages API 支持 `POST /v1/messages/count_tokens`，两个 Messages 路径都支持 `HEAD` 存活探测。`GET /health` 检查引擎就绪状态。本次发布还更新了 `message_start`、实测缓存用量、流式工具调用和推理 token 统计。 [Anthropic 路由](vllm/entrypoints/anthropic/api_router.py) · [缓存计量 #1436](https://github.com/1CatAI/1Cat-vLLM/pull/1436) · [SSE #1440](https://github.com/1CatAI/1Cat-vLLM/pull/1440) · [工具与推理用量 #1443](https://github.com/1CatAI/1Cat-vLLM/pull/1443)

---

# 🔨 从源码构建

在 Linux x86_64 上准备 CUDA 12.8 Toolkit、匹配的 C++ 构建工具及 `uv`，检出发布标签并指定 SM70。Python 运行时和内置原生扩展应使用同一源码修订。

```bash
git clone --branch v1.5.3 https://github.com/1CatAI/1Cat-vLLM.git
cd 1Cat-vLLM
uv venv --python 3.12 .venv
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_CUDA_ARCHITECTURES=70
uv pip install -e . --torch-backend=cu128
```

架构变量仅用于构建。已有 PyTorch 的构建方式及构建依赖见以下文档。 [CUDA 构建依赖](requirements/build/cuda.txt) · [源码安装指南](docs/getting_started/installation/gpu.cuda.inc.md)

---

# 📐 基准测试口径

1Cat-LLM 分别记录以下指标。

```text
算子延迟
算子吞吐
Attention 有效算力（TFLOP/s）
prefill 吞吐（tok/s）
目标模型直解
纯投机解码
流式解码
端到端吞吐
任务质量得分
PPL 与分布检查
```

对比时应保留以下测试条件。

- 精确的模型或 checkpoint
- GPU 型号和数量
- TP/PP 拓扑
- 上下文与输出长度
- 批大小
- KV 数据类型
- 量化路径
- CUDA Graph 模式
- 前缀缓存状态
- 采样参数
- 投机解码方法
- 草稿接受长度
- 质量结果
- 结果类型（实测或推算）

## 基准和回归工具

- [DFlash2 并发 API 基准](benchmarks/benchmark_dflash2_concurrent_rounds.py)使用 8192 或 32768-token 输入样本，在固定输出长度下截取共同流式窗口。可设置温度、top-p、top-k 和种子。
- [Flash-Next MTP4 轮耗时基准](benchmarks/benchmark_sm70_mtp4_round.py)以固定 8K 输入测量完整轮次，支持真实提示词样本。
- [Qwen 质量测试脚本](benchmarks/benchmark_sm70_qwen38_quality.py)支持显式用例清单、性能计时和输出健康检查。 [DeepSeek 原生客户端](benchmarks/deepseek_v41/native/bench_client.py)使用服务端计数器计算每轮产出和接受的 token 数。

v1.5.3 发布报告提供汇总表，标签中的脚本各有自己的输入和计时窗口。公开报告未附本轮完整样本清单或全矩阵重放命令。

以上参数按原始 PR 可追溯程度记录。

---

# 🛡️ 快速路径验收规则

微基准提速后，还需要在数值和整模型层面验收。

按算术变化，启用新路径可能需要通过以下检查。

- 算子结果逐位一致
- 数值误差有界
- CUDA Graph 重放稳定性
- 同口径 API 端到端速度
- 数据集质量
- 自然结束与输出健康检查
- PPL 与 logprob 分布检查
- 明确的回滚方式
- 按结构和运行时能力判断是否启用

开放和草稿 PR 仍在审查。已合并 PR 的历史中也可能保留被淘汰的实验，采用记录时应确认对应实现和验收结果。

早期约 79 TFLOP/s 的 Attention 候选在 256K 上曾未通过质量验收。之后单独合并了修复后的 FP32 路径。79T 属于历史实验记录，完整 FP32 结果沿用各自的精度和验证条件。

## v1.5.3 运行时改动

- **原生 MTP 与分组校验。** 共享 TP2/TP4 分派覆盖符合条件的 NVFP4 和 FP8 Qwen 分片。MTP4 q5 使用更新的分组路径，满足形状条件的 DFlash2 q6–q8 继续使用 48 行 E4M3 算子。 [#1472](https://github.com/1CatAI/1Cat-vLLM/pull/1472) · [#1478](https://github.com/1CatAI/1Cat-vLLM/pull/1478) · [#1491](https://github.com/1CatAI/1Cat-vLLM/pull/1491) · [#1505](https://github.com/1CatAI/1Cat-vLLM/pull/1505)
- **混合前缀缓存。** 符合条件的路径默认使用 2048-token KV/Mamba 块和部分前缀快照。支持周期保留时默认间隔为 8192。以主机 QSA KV 为准的路径会关闭默认周期保留，以保留尾部检查点。显式缓存设置优先，创建快照与命中时都检查池容量。 [#1401](https://github.com/1CatAI/1Cat-vLLM/pull/1401) · [#1492](https://github.com/1CatAI/1Cat-vLLM/pull/1492) · [#1500](https://github.com/1CatAI/1Cat-vLLM/pull/1500)
- **启动与请求稳定性。** MRV2 预热按主机 KV 状态池容量分配，图片编码器状态保留到投机 token 确认，CUDA 设备能力检查可在编译时跟踪。 [#1493](https://github.com/1CatAI/1Cat-vLLM/pull/1493) · [#1494](https://github.com/1CatAI/1Cat-vLLM/pull/1494) · [#1498](https://github.com/1CatAI/1Cat-vLLM/pull/1498)

## 已知限制

- **Flash-Next 启动与短前缀。** 256K 配置使用 `--max-num-seqs 8`。直接运行 `vllm serve` 可能默认启用 256 个序列槽位，超出 Mamba 状态池容量。预设以 2048 token 为单位复用前缀块，发布测试中 250K 追问的 TTFT 达 **4.58 s**。
- **并发与容量。** 三个单节点发布配置均设为 8 个序列槽位。27B DFlash2 为图片激活预留显存，报告中的 KV 容量约相当于五个 262144-token 上下文。满长请求还受循环状态及当前输入和输出预算约束。Flash-Next 未发布 64K C8 数据。新到达的长 prefill 可能短暂拖慢正在解码的请求。
- **非贪心采样。** 8K C1 从贪心切换到温度 0.6 时，27B MTP4、Flash-Next 每轮实测分别增加约 2.7 ms、3.5 ms。新启动引擎的首个非零温度请求还可能因编译采样算子多耗时约 3 秒。
- **硬件与精度。** 发布测试覆盖上述 32GB V100 拓扑。没有兼容 MTP 权重的模型需选用其他解码配置。E4M3、FP16 和 E5M2 KV 路径的准入和内存需求不同。

[发布说明中的限制](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.3) · [实测 TTFT 与解码数据](http://156.226.174.161:48152/1.5.3/#data)

## 快照时仍开放的后续 PR

| PR | 涉及范围 |
| --- | --- |
| [#1508](https://github.com/1CatAI/1Cat-vLLM/pull/1508) · [#1507](https://github.com/1CatAI/1Cat-vLLM/pull/1507) | 融合 GDN 校验的引擎 FP8 策略及 SM70 Marlin bias |
| [#1506](https://github.com/1CatAI/1Cat-vLLM/pull/1506) · [#1504](https://github.com/1CatAI/1Cat-vLLM/pull/1504) | SM70 回归测试维护与编译期能力检查清理 |
| [#1503](https://github.com/1CatAI/1Cat-vLLM/pull/1503) · [#1502](https://github.com/1CatAI/1Cat-vLLM/pull/1502) · [#1501](https://github.com/1CatAI/1Cat-vLLM/pull/1501) | Triton 采样截断、冷预热 CUDA 栈、Flash-Next MTP 编译上下文 |
| [#1497](https://github.com/1CatAI/1Cat-vLLM/pull/1497) · [#1495](https://github.com/1CatAI/1Cat-vLLM/pull/1495) · [#1486](https://github.com/1CatAI/1Cat-vLLM/pull/1486) | 缓存漂移恢复、备用工具解析器、分层 KV 保留 |
| [#1488](https://github.com/1CatAI/1Cat-vLLM/pull/1488) · [#1271](https://github.com/1CatAI/1Cat-vLLM/pull/1271) | 另一套社区 8×V100 DeepSeek 配置及 Gemma-4 D=512 支持 |

这些提案在 2026 年 11 月 14 日的审阅快照中仍为开放状态。已合并工作见发布索引。 [开放 PR](https://github.com/1CatAI/1Cat-vLLM/pulls) · [发布 PR 索引](https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.5.3/PR_INDEX.md)

---

# 🧱 推理运行时

1Cat-LLM 的工作覆盖以下服务链路。

- FlashAttention-V100;
- 页式 KV 工具
- FP8 KV 转接
- QSA 稀疏 Attention
- FlashQLA / GDN;
- TurboMind SM70 量化算子
- 分组 MoE
- MTP;
- DFlash2;
- CUDA Graph;
- 前缀缓存
- 混合架构 Mamba 状态
- 自定义 all-reduce
- 采样
- 工具调用
- 推理解析器
- 结构化输出
- 独立 GGUF 加载与混合格式投影准备
- 按能力选择算子并诊断已加载的算子
- 混合 prefill 调度与混合前缀重放
- GPU、锁页主机内存和磁盘 PLE 存储
- 原生图像、音频和视频任务
- wheel、RPATH 和 ABI 打包

算子需在整模型和服务 API 中正确执行，才能形成实际收益。

---

# 💾 模型权重与上下文的内存开销

近期工作减少了常驻权重、scale 副本、计算图临时空间和加载峰值。以下为代表性 PR 的**每卡**测量。

| 改动 | 测量记录 | 范围 |
|---|---|---|
| 共享权重、量化编码和 scale | 模型分配显存 **10.07 → 6.47 GiB**， 可用 KV 预算 **10.57 → 12.68 GiB** | [#671](https://github.com/1CatAI/1Cat-vLLM/pull/671) · QUASAR NVFP4 / TP4 / DFlash2 q7 / E4M3 目标模型 KV |
| 共享图临时缓冲区与草稿快照 | 非 KV 活跃内存: TP2 **17.117 → 15.146 GiB**, TP4 **10.225 → 8.794 GiB** | [#677](https://github.com/1CatAI/1Cat-vLLM/pull/677) · 默认启用与关闭的实测对照 |
| 复用已编译子图 | 首次启动 **271.26 s**, 相同配置重启 **95.30 s** | [v1.5.1](https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.1) · v1.5.1 内的两次历史测量 |

内存测试使用 V100-SXM2 32GB、CUDA 12.8 / Torch 2.10、FP16 计算和草稿 KV、262144 上下文、8192-token prefill 及 CUDA Graph。#671 使用 32 序列槽位、显存利用率 0.85，#677 均为 32 槽位、利用率 0.90。两组节省量有重叠。自动 KV 分配可能使用省出的空间，NVML 总占用未必同比下降。重启仍需加载权重并捕获 CUDA Graph。

本地张量布局也支持部分 **TP1 / TP2** 路径。 [#666](https://github.com/1CatAI/1Cat-vLLM/pull/666) 记录 32GB V100 上的 27B NVFP4 目标直解，**TP1 为 64K，TP2 为 256K**。单卡和双卡配置应按各自的内存约束选型，客户端并发超过可驻留容量时可排队。

---

# 🧭 项目方向

1Cat-LLM 关注一个问题。

> **重构软件栈后，Volta 还能释放多少现代推理性能？**

目前关注以下方向。

- 进一步优化长上下文 Attention
- 降低 DFlash2 校验成本
- 提高投机草稿接受率
- 稀疏 Attention
- SM70 上的现代量化格式
- 融合解码热路径
- MoE 路由与分组 GEMM
- 扩大 GGUF 模型质量与服务验证
- 改善混合负载延迟、缓存复用和内存余量
- 扩大 SM70 模型支持
- 稳定 wheel 和发布打包

---

# 💬 微信社区

添加微信号 **`YM_isi`**，申请加入 **1Cat-LLM 开源社区**的最新交流群。

![1Cat-vLLM 微信 8 群二维码](docs/assets/wechat-group-qr-8-20261005.png)

---

# ❤️ 致谢

1Cat-LLM 建立在开源推理生态的工作之上，包括 vLLM、NVIDIA CUDA、FlashAttention、CUTLASS/TurboMind 相关算子、模型作者、量化项目和其他贡献者。各 PR 与源码文件记录了具体来源。

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)
- [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
- [marlin_v100](https://github.com/zhinianqin/marlin_v100)
- [v100-skinny](https://github.com/dnv2003/v100-skinny) 提供 SM70 QPN2 / QPN4 / QPN8 和 MXFP4-QPN 算子所用的 QPN quadpair-N `m8n8k4` 解码布局。MIT 声明见 `csrc/sm70_turbomind/ops/LICENSE.v100-skinny`。
- [llama.cpp](https://github.com/ggml-org/llama.cpp) 和 vllm-gguf-plugin 桥接。固定的 GGUF 上游版本见 [来源清单](csrc/quantization/gguf_upstream/source_manifest.json)，相关声明随源码和 wheel 保留。

特别感谢 `@1CatTCat` 和 [@yangzhuxinyzx](https://github.com/yangzhuxinyzx) 对 **1Cat-LLM** 持续演进与性能优化的贡献。

外部实现或算法的来源和许可证保存在相应源码及 PR 历史中。

---

# 许可证

请参阅仓库许可证及随附或改编的第三方组件许可证。
