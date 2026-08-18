# opt21-25 优化包补丁（v1.3.0 重放）

> 适用于 1Cat-vLLM **v1.3.0**（`1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl`）的完整重放补丁集
> 基准：安装后的原始 `site-packages/vllm`（git 基线 `v130-replay-base`）

## ⚠️⚠️ 重要区分：单独 patch 与总 patch（请务必注意）

本 PR 提供 **9 个补丁文件**：

- **08-opt21-25-all.patch** = **总补丁**（一次应用全部优化，160KB）
  - 适用：全新部署 / 想一次到位
  - 应用：`git apply 08-opt21-25-all.patch`
- **01 ~ 07 单独 patch** = **逐项应用**（可按需选择部分优化项）
  - 适用：已有部分优化 / 只想启用特定项
  - 应用：`git apply 0X-*.patch`（**必须按编号顺序**，02b 依赖 02a）

> ⚠️ **不要同时应用总补丁和单独补丁**（会冲突）——二选一。
> ⚠️ 单独补丁必须按 01→02a→02b→03→04→05→06→07 顺序应用（有依赖关系）。
> ⚠️ 所有补丁以 `site-packages/` 为根（路径形如 `a/vllm/...`），在 `<site-packages>/vllm` 目录下执行 `git apply`。

---

## 补丁清单与说明

### 01-opt21+25-base.patch（opt21 基础项 + opt25 Anthropic 协议，12 文件）

**opt21 基础项**（基础设施修复）：
| 优化项 | 作用 | 解决什么问题 | 使用 |
|---|---|---|---|
| HTTP keepalive 5s→600s | 长连接不再被超时断开 | 长请求（长上下文生成/工具调用等待）期间前端/网关中断连接 | 默认生效（`VLLM_HTTP_TIMEOUT_KEEP_ALIVE` 可调） |
| Cache 迁移到 /tmp | 编译缓存/元数据缓存移到 tmpfs | 家目录磁盘膨胀写满 | 默认生效；`VLLM_PERSISTENT_CACHE=1` 回退家目录 |
| 启动日志持久化 | 启动期日志落盘 `/tmp/log/vllm-starting-log.txt` | 启动信息不可回溯 | 自动 |
| STREAM_KEEPALIVE 双层保活 | 引擎 15s + API 120s 双层心跳 | 长任务 SSE 连接被中间层断开 | 自动 |
| API 能力发现 | `/v1/models` 上报 context_window/max_output_tokens | 前端无法感知模型上下文能力 | 自动 |
| Auto Dynamic NTK（v130 缺失已补） | max_model_len 超限时自动升级 RoPE 缩放 | 长上下文位置编码越界 nan | 见 06 补丁（已默认启用） |
| MTP 安全检测（v130 缺失已补） | 无 MTP 层模型自动禁用投机 | 无 MTP 层配置 MTP 启动崩溃 | 自动 |

**opt25 Anthropic 协议**：
| 优化项 | 作用 | 解决什么问题 |
|---|---|---|
| 响应 ID `msg_<random_uuid>` | 统一消息 ID 格式 | CC-HAHA 会话恢复失败 |
| SSE type/role 补全 | 流式消息带上 type/role | SDK 无法识别消息静默丢弃 |
| HEAD 路由（全局 + /v1/messages） | 探针请求返回 200 | CC-HAHA 启动 HEAD 探测 404 |

**实测**：v1.3.0 + 535 驱动 + 4×V100，引擎就绪 260-350s；工具调用 stop_reason=tool_use 正常；NTK yarn 自动升级 factor=2.0。

### 02a-opt22-dualformat.patch（opt22 双格式工具调用，5 文件）

**作用**：qwen3-coder/qwen3-xml 双格式（XML/JSON）完整支持——fix 体系（0/1/2/3/4）+ JSON 流式 + 换行转义 + Edit 闭合引号 + 前缀截取 + 格式检测。

**解决什么问题**：
- JSON 流式工具调用参数丢失（换行未转义/闭合引号丢失）
- 前端无法任选 XML/JSON 格式
- 长文（Write/Edit content）参数不完整

**使用**：`VLLM_QWEN3X_TOOL_FIX`（1 推荐/2 强制 JSON/3 强制 XML；见 02b 已默认启用 1）；配套模板 `chat_template-fix.jinja`。

**实测**：fix=1/2/3 全矩阵通过（XML-Write 多行参数完整 / JSON-Edit 三参完整 / 纯正文正常）；4 并发工具调用 2.7s。

### 02b-opt22-fix-default.patch（opt22 fix=1 默认启用，1 文件）

**作用**：`tool-call-parser qwen3_coder/qwen3_xml` 启动参数即默认启用 fix=1（环境变量未设置时）——无需显式设置 `VLLM_QWEN3X_TOOL_FIX`。

**解决什么问题**：双格式修复默认不生效，需手动设置环境变量。

**实测**：无环境变量 + qwen3_coder parser → 双格式工具正常。

### 03-opt24-drafter.patch（opt24 Drafter 上下文对齐，1 文件）

**作用**：投机解码 drafter 的 `max_position_embeddings` 自动对齐 target `max_model_len`。

**解决什么问题**：长上下文投机解码 drafter 位置编码越界/不生效。

**实测**：v1.3.0 引擎 MTP 模式正常（cudagraph MTP 尺寸自动化 v1.3.0 已内置，无需重放）。

### 04-opt23.patch（opt23 并发 + MTP 完整支持，7 文件）

**并发部分**：
| 优化项 | 作用 | 解决什么问题 |
|---|---|---|
| long_prefill 自动配置 | max_seqs≥4 自动设 25% 阈值 | 长 prefill 阻塞 decode |
| 65/35 budget | prefill/decode 预算拆分 | decode 吞吐受限 |
| DDTree 膨胀封顶 | 树投机 draft 受 decode headroom 约束 | DDTree 膨胀失控 |
| GPU-LRU 多并发 | 1536 共享 tail 池 + prefill 并集注入 | 多并发 MTP tail 竞争 |

**MTP 完整支持（0/1/2/3/4）**：
| 档位 | 行为 |
|---|---|
| MTP=0 | 关闭投机（`--spec-tokens 0`，等价不传参） |
| MTP=1/2/3/4 | **fused 优先**（共用参数化 kernel）；fused 不可用自动降级 non-fused |

**使用**：多并发需 `VLLM_SM70_MTP_GPU_LRU_MULTI_CONCURRENT=1` + `--max-num-seqs 8`；MTP 档位用 `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`。

**实测**：8 并发纯文本全对（6.1s MTP4 / 10.6s 多并发）/ 4 并发工具调用 2.7s / GPU-LRU 激活 per_rank_tail=384 total=1536 / MTP=0 关闭投机正常。

### 05-opt21-kvwarm.patch（opt21 KV 缓存时限优化，1 文件）

**作用**：Warm Block 时间倒排回收——释放的带 hash block 进 warm 队列（保留 hash + 时间戳），同前缀请求直接复用，压力不足时冷→热两阶段回收。

**解决什么问题**：CC-HAHA 多轮对话重载 system prompt/历史前缀每次重算 prefill。

**使用**：默认生效；`VLLM_KV_RETENTION_P25~P80` 调保留时间（12h/6h/3h/1h/30m/即刻）。

**实测**：单元测试 6 项全通过（free 分流/touch 无崩溃/压力回收/reset 排空）；引擎同前缀多轮无崩溃；prefix cache 命中率 98.6%。

### 06-opt21-ntk-default.patch（opt21 NTK 默认启用，1 文件）

**作用**：Auto Dynamic NTK 默认启用（无需设置 `VLLM_ALLOW_LONG_MAX_MODEL_LEN`）；524288 硬拒绝保留。

**解决什么问题**：长上下文需显式环境变量才启用 RoPE 缩放，漏设则越界 nan。

**实测**：引擎日志 `Auto-upgraded rope_type to 'yarn' factor=2.0`（无环境变量）。

### 07-opt21-cache-read.patch（opt21 anthropic cache 数据填充，1 文件）

**作用**：`/v1/messages`（Anthropic 端点）usage 补 `cache_read_input_tokens`（3 处：非流式 + 流式 message_start/message_delta）。

**解决什么问题**：前端无法获取真实缓存命中数据（此前需固定常量修正，百分比恒 96.80%）。

**实测**：长 prompt 同前缀二次请求 `cache_read=1664`（真实值）。

---

## 应用方式

```bash
# 总补丁（推荐全新部署）
cd <site-packages>/vllm && git apply <path>/08-opt21-25-all.patch

# 或单独补丁（按顺序）
git apply <path>/01-opt21+25-base.patch
git apply <path>/02a-opt22-dualformat.patch
git apply <path>/02b-opt22-fix-default.patch
# ...

# 回滚（vllm 目录需 git 仓库）
git reset --hard v130-replay-base
```

## 环境备注

- 引擎验证环境：4×V100（SM70）+ 535 驱动 + vLLM v1.3.0 + MTP3/4 + 8 并发 + GPU-LRU
- 启动参数兼容：`--max-num-seqs 1|8`、`--speculative-config '{"method":"mtp","num_speculative_tokens":0|1|2|3|4}'`
- 完整重放记录：`1cat-vllm-v130/OPT21-25-重放分析-v130.md`
