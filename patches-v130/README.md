# opt21-25 补丁包（v1.3.0 重放）

> 基准：`1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl` 安装后的原始 `site-packages/vllm`（git 基线 `v130-replay-base` / commit `0634460`）
> 生成：2026-08-18（含 MTP 完整支持增量重放），取自 zxvllm120 环境 site-packages/vllm 的 git 提交历史

## 应用方式

所有 patch 均以 `site-packages/` 为根（路径形如 `a/vllm/...`）。统一使用 `git apply`：

```bash
cd <site-packages>/vllm          # vllm 目录需已 git init（或 cd site-packages 后 git init）
git apply <patches>/0X-*.patch   # 按序号顺序应用
# 或整体应用
git apply <patches>/08-opt21-25-all.patch
```

部分 patch 为 `git format-patch` 格式（含 commit 信息），`git apply` 兼容；如需保留提交信息可用 `git am`。

## 补丁清单（按应用顺序）

| # | 文件 | 内容 | 对应 commit |
|---|---|---|---|
| 01 | `opt21+25-base.patch` | **opt21 基础项**（keepalive 5→600 / Cache 迁移 /tmp / 启动日志持久化 / STREAM_KEEPALIVE 双层保活 / API 能力发现 / Auto Dynamic NTK / MTP 安全检测）+ **opt25 Anthropic 协议**（msg_id / SSE type-role / HEAD 路由）——12 文件 | `91ceff5` |
| 02a | `opt22-dualformat.patch` | **opt22 双格式工具调用**（fix 体系：qwen3coder 1943 行 + qwen3xml 1680 行整体替换 / envs fix 变量 / openai+anthropic serving 注入）——5 文件 | `6d16aa0` |
| 02b | `opt22-fix-default.patch` | **opt22 fix=1 默认启用**（parser 绑定：cli_args 检测 tool_call_parser qwen3_coder/qwen3_xml 且环境变量未设置 → 自动 VLLM_QWEN3X_TOOL_FIX=1）——1 文件 | `7c53ee0` 部分 |
| 03 | `opt24-drafter.patch` | **opt24 Drafter 上下文对齐**（原 opt26：Auto-extend drafter max_position_embeddings）——1 文件（cudagraph MTP 尺寸自动化 v1.3.0 已内置） | `470f8cb` |
| 04 | `opt23.patch` | **opt23 并发 + MTP 完整支持**：并发（long_prefill 自动配置 / 65-35 budget / DDTree 封顶 / GPU-LRU 多并发 1536 tail / prefill 并集注入）+ **MTP 0/1/2/3/4 完整支持**（fused 优先 MTP1/2/3/4 共用参数化 kernel，kernel 不可用自动降级 non-fused；MTP=0 关闭机制：Field ge=0 + __post_init__ self-disable + _verify 只拒负 + arg_utils 跳过 auto-MTP/返回 None）——7 文件 | `c0161e3` + `7c53ee0` 部分 |
| 05 | `opt21-kvwarm.patch` | **opt21 KV 缓存时限优化**（Warm Block 时间倒排回收，原 opt22 独立设计重新纳入）——1 文件（block_pool.py） | `0124dad` |
| 06 | `opt21-ntk-default.patch` | **opt21 NTK 默认启用**（Auto Dynamic NTK 去掉 VLLM_ALLOW_LONG_MAX_MODEL_LEN 条件——默认启用，524288 硬拒绝保留）——1 文件 | `7c53ee0` 部分 |
| 07 | `opt21-cache-read.patch` | **opt21 anthropic cache_read_input_tokens 填充**（usage 3 处补真实缓存命中数据）——1 文件 | `042b658` |
| 08 | `opt21-25-all.patch` | **整体补丁**（01-07 全部） | `0634460..HEAD` |

## MTP 档位支持（opt23）

| 档位 | 行为 |
|---|---|
| MTP=0 | 关闭投机（`--spec-tokens 0` → speculative_config=None；等价于不传参） |
| MTP=1/2/3/4 | **fused 优先**（共用参数化 kernel）；fused kernel 不可用 → **自动降级 non-fused 标准 dynamic-vocab 路径** |
| 不传 | auto-MTP（`VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS`）或显式配置 |

## 注意事项

1. **02b 依赖 02a**（fix 绑定依赖 fix 体系变量）；04 含 MTP=0 机制（依赖 01 的 MTP 安全检测同文件——顺序无冲突）
2. **环境变量**（全部可选——均有默认）：
   - fix：默认 1（qwen3 parser 启用即生效）；显式设 `VLLM_QWEN3X_TOOL_FIX` 覆盖（0/1/2/3/4）
   - NTK：默认启用（无需 `VLLM_ALLOW_LONG_MAX_MODEL_LEN`）
   - opt23 多并发：`VLLM_SM70_MTP_GPU_LRU_MULTI_CONCURRENT=1` + `--max-num-seqs 8`
   - KV warm：`VLLM_KV_RETENTION_P25~P80` 调保留时间
3. **启动参数**（v1.3.0 兼容）：`--max-num-seqs 1|8`、`--speculative-config '{"method":"mtp","num_speculative_tokens":0|1|2|3|4}'`
4. **验证**：应用后 `python -m py_compile` 各文件 + 启动引擎跑工具矩阵（参考 `OPT21-25-重放分析-v130.md`）

## 回滚

```bash
cd <site-packages>/vllm
git reset --hard v130-replay-base   # 恢复 v1.3.0 原始状态
```
