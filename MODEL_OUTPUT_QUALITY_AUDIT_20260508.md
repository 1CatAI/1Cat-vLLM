# Qwen3.5/V100 模型输出质量全链路审计

## 最终准入结果：FA2 + fp8_e5m2 KV

审计时间：2026-05-08 15:27-15:41
实测模型：`/home/ymzx/models/Qwen3.6-35B-A3B-AWQ`
实测硬件：4 x V100/SM70，`CUDA_VISIBLE_DEVICES=1,2,3,4`
实测后端：`FLASH_ATTN_V100`，TP=4，FP16，`kv_cache_dtype=fp8_e5m2`

准入结论：通过。没有发现 FA2 fp8_e5m2 造成的模型输出质量退化、特殊 token 泄漏、工具调用格式错误、NaN/Inf logprob、长上下文 needle 丢失或 256k 稳定性问题。

| 审计项 | 覆盖范围 | 结果 |
| --- | --- | --- |
| 算子质量 | dense/paged decode/prefill、GQA、HD256、模拟 logits、backward support surface | 29 / 29 通过 |
| 35B 端到端 32k | 短指令、JSON、干扰项、代码推理、官方 XML tool call、8k/32k 三深度 needle | 11 / 11 通过 |
| 35B 端到端 256k | 262100-token prompt，中部 needle，chunked prefill + decode | 6 / 6 通过 |
| 工具解析单测 | Qwen3Coder XML parser、`anyOf` 参数、重复同名工具、`$defs` schema 不原地修改 | 3 / 3 通过 |

关键数据：

- `fp8_e5m2` operator decode/prefill 最大误差：`max_abs <= 0.0009765625`
- 32k 审计：`failed_cases=[]`，`nonfinite_cases=[]`
- 256k 审计：最终 `prompt_tokens=262100`，输出 `MQA262144-50-V100`
- 256k needle latency：`121.943 s`
- 工具解析 pytest：`3 passed`

复现结果文件：

- `/tmp/fa2_v100_quality_audit_margin_20260508.json`
- `/tmp/qwen36_35b_fa2_e5m2_model_output_quality_32k_20260508.json`
- `/tmp/qwen36_35b_fa2_e5m2_model_output_quality_256k_20260508.json`

注意：下面保留的是同日早些时候的模型输出质量审计记录，部分路径仍写作 Qwen3.5/FP16。最终准入以上面的 `Qwen3.6-35B-A3B-AWQ + fp8_e5m2` 结果为准。

审计日期：2026-05-08
目标源码：`/home/ymzx/桌面/1cat-vllm/1Cat-vLLM-0.0.3/vllm`
实测模型：`/home/ymzx/models/Qwen3.5-35B-A3B-AWQ`
实测硬件：4 x V100/SM70，`CUDA_VISIBLE_DEVICES=1,2,3,4`
实测后端：`FLASH_ATTN_V100`，TP=4，FP16，`max_model_len=32768`

## 结论

当前 FA2/V100 路径没有在已测用例中表现出模型输出质量退化：

- 新增模型端到端质量哨兵：8/8 通过，`nonfinite_logprob=0`。
- 质量覆盖面包括：官方 Qwen3.5 chat template、`enable_thinking=False`、特殊 token 泄漏、JSON 格式、中文精确指令、干扰项抵抗、官方工具模板 XML tool call、32k 长上下文 needle 检索。
- 之前已完成的 FA2 vs Triton 质量审计显示：32k FA2/Triton 均 10/11，通过用例 token 完全一致；256k FA2 needle 3/3 通过。

同时发现并修复了两个会影响“模型输出看起来变差”的外围解析风险：

1. 工具 schema `$defs` 提取会修改原始 request tool 对象。已修复为深拷贝后提取，避免复用工具 schema 时 `$ref` 失效。
2. Qwen3/Qwen3.5 XML tool parser 在重复同名工具、`anyOf` 参数、畸形函数头、流式多参数同 delta、最后参数和 `</function>` 同 delta 到达时有解析风险。已修复并补充回归用例。

## 实测结果

输出文件：

- `model_output_quality_audit_20260508_FA2.json`

汇总：

```json
{
  "passed": 8,
  "total": 8,
  "pass_rate": 1.0,
  "failed_cases": [],
  "nonfinite_cases": [],
  "median_latency_sec": 1.1212229335214943
}
```

用例结果：

| 用例 | 覆盖点 | 结果 |
| --- | --- | --- |
| `exact_zh_phrase_no_leak` | 禁用 thinking、特殊 token 泄漏 | 通过 |
| `json_contract_stable` | JSON-only 格式约束 | 通过 |
| `distractor_resistance` | 干扰项抵抗 | 通过 |
| `simple_code_reasoning` | 简单推理/代码理解 | 通过 |
| `official_tool_template_xml_call` | 官方 Qwen3.5 工具模板 XML tool call | 通过 |
| `needle_len32768_depth0.10` | 32k 长上下文前部 needle | 通过 |
| `needle_len32768_depth0.50` | 32k 长上下文中部 needle | 通过 |
| `needle_len32768_depth0.90` | 32k 长上下文尾部 needle | 通过 |

复现命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
VLLM_SM70_ENABLE_LM_HEAD_FASTPATH=1 \
/home/ymzx/miniconda3/envs/1cat-vllm-0.0.3/bin/python \
/home/ymzx/桌面/1cat-vllm/1Cat-vLLM-0.0.3/vllm/tools/model_output_quality_audit.py \
  --disable-custom-all-reduce \
  --disable-mm \
  --disable-thinking \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --output /home/ymzx/桌面/1cat-vllm/1Cat-vLLM-0.0.3/vllm/model_output_quality_audit_20260508_FA2.json
```

## 已修复项

### 1. 工具 schema `$defs` 原地修改

风险：`get_json_schema_from_tools()` 之前通过 `params.pop("$defs", {})` 提取 definitions。`params` 来自请求对象本身，会把用户传入的工具 schema 改掉。对包含 `$defs/$ref` 的工具 schema，第一次处理后请求对象缺少 `$defs`，后续结构化输出、工具解析或 request 复用可能异常。

修复：

- `vllm/tool_parsers/utils.py` 对参数 schema 先 `deepcopy`，再移除嵌套 `$defs`。
- definitions 提取改为 `params.get("$defs", {})`，不修改原对象。
- 新增 `tests/tool_parsers/test_tool_parser_utils.py`。

### 2. Qwen3.5 XML 工具解析

风险：

- 同一函数连续调用会被按函数名去重，导致第二个调用丢失。
- `anyOf` 参数没有顶层 `type` 时被当成字符串，JSON 参数会被错误转义。
- malformed `<function=...` 会触发异常并让整段工具解析失败。
- 流式输出中一个 delta 带多个参数时，只处理第一个参数。
- 最后一个参数和 `</function>` 同 delta 到达时，可能先关闭 JSON，导致参数丢失。

修复：

- `vllm/tool_parsers/qwen3coder_tool_parser.py` 保留重复同名工具调用。
- `anyOf` 参数按 object 尝试 JSON 解析。
- malformed function header 安全忽略。
- 流式解析一次处理所有完整参数，并在参数后再处理 function close。
- 新增 `tests/tool_parsers/test_qwen3coder_parser_regression.py`。

本地验证：

- `ruff check` 通过。
- `py_compile` 通过。
- 源码级 parser/schema 直接验证通过。

## 配置审计结论

### 官方 Qwen3.5 模板

本地 tokenizer 官方模板行为：

- 不传 `enable_thinking` 时默认进入 `<think>`。
- `enable_thinking=False` 时 assistant 开头会插入空的 `<think>\n\n</think>\n\n`。
- 工具模板使用 Qwen XML 格式：`<tool_call><function=...><parameter=...>`。

建议：

- 非思考服务默认显式设置：`--default-chat-template-kwargs '{"enable_thinking": false}'`。
- 不要继续使用 Qwen3-Coder 魔改模板覆盖 Qwen3.5；默认用模型 tokenizer 自带官方模板。
- `--trust-request-chat-template` 保持默认 `False`，避免请求侧替换模板造成质量/安全漂移。

### 采样默认值

模型 `generation_config.json` 当前是采样默认：

- `do_sample=true`
- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`

这不是 FA2 质量问题，但会让线上输出不稳定。质量回归、工具调用、结构化输出建议显式传：

- `temperature=0`
- `top_p=1`
- 固定 `seed`

### 精度和量化

Qwen3.5 配置声明 `torch_dtype=bfloat16`，但 V100 不支持 BF16，运行日志显示会 cast 到 FP16。这是 V100 必然约束，不是 FA2 特有问题。若要判断“智商”相对原模型是否下降，需要另做 AWQ/FP16 vs BF16 的同题评测；本次只能确认当前 V100/AWQ/FP16 路径没有新增可见输出质量异常。

### 长上下文

本次 32k 三深度 needle 通过。此前 256k FA2 needle 已 3/3 通过，说明长上下文 prefill/chunked path 在 needle 类客观检索上未见质量失败。

### 仍需跟踪的日志风险

- `fla/ops/utils.py` 出现 `seq_len < num_heads` 的格式警告。当前短输出和长上下文质量均通过，因此更像 decode 小批次触发的误报，但建议补一个 GDN/linear attention 的张量布局单元测试后再决定是否降级/屏蔽警告。
- AWQ MoE 加载时出现 `experts.w2_weight/w13_weight not found, skip loading`，随后日志显示 `SM70 MoE: batched GEMM enabled`。这大概率是 AWQ packed 参数路径的预期提示，但建议增加“缺失非 packed 参数即失败”的加载审计，避免真漏权重被日志淹没。

## 下一步建议

1. 用 OpenAI server 真实接口补一轮 streaming tool call 测试，覆盖同名工具多次调用、多参数同 delta、`anyOf` 参数。
2. 把 `model_output_quality_audit.py` 加入回归清单，至少每次 FA2/Qwen3.5 改动跑短用例 + 32k needle。
3. 256k 全链路质量可复用现有 `fa2_model_quality_audit.py`，每次大改跑 0.10/0.50/0.90 三个 needle 深度。
4. 如果要证明“AWQ 量化没有降智”，需要同题对照更高精度基线；这不属于 FA2 后端回归，但属于模型部署质量验收。
