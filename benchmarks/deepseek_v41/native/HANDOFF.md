# DeepSeek-V4.1-Flash C4140 交接

更新于 2026-11-08。维护对象为原版 checkpoint 的 PP4 × TP4 路线，16 张 V100，单请求、greedy、固定 DSpark5。

## 交接状态

交付默认配置的最终测试为 **20.454 ms/轮**，重复 prefill 后复测为 **20.436 ms/轮**，均为 **3.704 token/轮**，吞吐 **179.449–179.602 token/s**。相对 21.683 ms 接手基线，轮延迟下降约 5.7%。四节点重新编译、算子回归、7K 真实文本及重复 prefill 已完成，服务以该配置运行。

交付默认关闭 prefill 提前接收，14,253-token TTFT 为 **2.93–2.95 秒**。提前接收开启时的非法访存根因仍待定位。私有 PR 使用草稿状态，完整静态检查的遗留失败列于下文。

用户已接受约 20 ms 的阶段成果，本次工作收口为源码和部署交接。后续迭代沿用 DSpark5。

源码基线为 `7257b3c1f254dc09edd5ab94b577546f9a703a94`，Dev-1 工作分支为 `claude/ds41-native-pp4`。

## 快速接手

登录 Dev-1（外部访问方式见内部运维文档）。

~~~bash
cd ~/v41
cat HANDOFF.md
git -C src status --short
git -C src log -1 --oneline
~~~

API 地址为 `http://10.56.0.11:18000`，模型名为 `deepseek-v4.1-flash`。节点之间可通过 IB 地址登录 Dev-1。

~~~bash
cd ~/v41
(setsid nohup cluster/bench.sh "" handoff_recheck prefill > logs/handoff_recheck.log 2>&1 < /dev/null &)
tail -n 30 logs/handoff_recheck.log
# 上一项完成后执行真实长文本检查
~/v41env/bin/python -u long_check.py handoff_recheck
~~~

`bench.sh` 重启四台节点，预热 64 token，再执行 2 轮 × 4 个 prompt × 256 token 的 greedy decode。Dev-3 从 SATA 重载权重约需 4.5 分钟，完整流程约 8 分钟。

## 机器与部署

| 节点 | PP 工作 | IB 地址 | 跳板机内侧地址 | 系统可见内存，十进制 GB |
| --- | --- | --- | --- | --- |
| Dev-1 | layers 0–7、Engram 第 1 层、API | 10.56.0.11 | 192.168.1.187 | 201.3 GB |
| Dev-2 | layers 8–19、Engram 第 14 层 | 10.56.0.12 | 192.168.1.207 | 201.3 GB |
| Dev-3 | layers 20–31 | 10.56.0.13 | 192.168.1.223 | 100.8 GB |
| Dev-4 | layers 32–39、target head、DSpark | 10.56.0.14 | 192.168.1.14 | 100.8 GB |

集群经跳板机访问表中的 LAN 地址，外部入口和账号见内部运维文档。节点间使用 IB 地址免密登录。

每节点为 Dell C4140、双 Xeon Gold 6138、4 × V100-SXM2-32GB、节点内 NVLink。IB 网卡为 `ibp216s0`，HCA 为 `mlx4_0:1`。驱动为 580.178.04，CUDA 为 12.8。

测量时应用 SM 时钟为 1530 MHz。Dev-4 物理 GPU0 的实际 HBM 为 918 MHz，应用时钟字段为 877 MHz，其余卡实际 HBM 为 877 MHz。此次软件优化沿用接手时的硬件状态。排查时 Dev-4 四张卡的 volatile corrected/uncorrected ECC 均为 0。时钟差异属于复现条件，非法访存的硬件归因尚无结论。

| 路径 | 用途 |
| --- | --- |
| `~/v41/src` | Dev-1 Git checkout，服务实际运行的源码 |
| `~/v41env` | torch 2.10.0+cu128、triton 3.6.0、1Cat-vLLM 1.5.2 wheel |
| `~/v41/run_pp4.sh` | 四节点 launcher，默认开关的部署入口 |
| `~/v41/serve-native-pp4.yaml` | TP4、PP4、单请求、16K 上下文、DSpark5 |
| `~/v41/cluster/` | 同步、构建、启停、测试、采样 |
| `~/v41/logs/` | 基线、候选、稳定性、构建和故障日志 |
| `~/v41/prof/` | 各节点自己的 nsys/sqlite |
| `~/models/ds41-1017` | 本节点 target 权重与 Engram 数据 |
| `~/models/ds41-1017-dspark` | DSpark 模型，权重位于 Dev-4 |
| `~/v41/engram_lock.py` | Dev-1/2 常驻的 Engram 内存锁定进程 |
| `benchmarks/deepseek_v41/native/` | 本路线的源码内工具与本文 |

旧 `benchmarks/deepseek_v41/HANDOFF.md` 描述另一条早期 GGUF 路线。本集群以本文及 `native/run_pp4.sh`、`native/serve-native-pp4.yaml` 为入口。

## 最终配置

`num_speculative_tokens=5`，FULL CUDA Graph capture sizes 为 1–6，`max-num-seqs=1`，`max-num-batched-tokens=1024`，prefix caching 关闭。launcher 的 `--async-scheduling` 覆盖 YAML 中的同步默认项。

五个新开关已注册到 `vllm/envs.py`，注册默认值为 `"0"`，集群 launcher 默认开启。

| 开关 | 作用 |
| --- | --- |
| `VLLM_V41_ATTN_RING_POST` | decode attention combine、逆 RoPE、FP8 QDQ、ring 写入融合 |
| `VLLM_V41_PREP_REGS` | register-resident attention prep |
| `VLLM_V41_GREEDY_VERIFY` | TP-local top-1、紧凑候选交换和单请求 greedy verification |
| `VLLM_V41_ROUTE_MATCH` | 按首次出现分组的 expert slot plan |
| `VLLM_V41_W13_SWIGLU` | K5120 MXFP4 gate/up 与 clipped SwiGLU 融合 |

集群默认设置 `VLLM_V41_PP_RECV_EARLY=0`，prefill 接收使用主 stream。`VLLM_V41_PP_SEND_DEFER=1` 沿用当前配置。

已有主要开关包括 `VLLM_V41_NATIVE=1`、`VLLM_PP_LAYER_PARTITION=8,12,12,8`、`VLLM_V41_LL_AR=1`、`VLLM_V41_H16=1`、`VLLM_V41_HC_SIDE=1`、`VLLM_V41_SHARED_STREAM=1`、`VLLM_V41_MARKOV_SHARD=1`、`VLLM_V41_PP_TOKEN_NCCL=1`。

greedy verification 仅用于匹配条件的单请求 DSpark 路径。随机采样、logprobs、penalties、structured output、采样诊断、扩展或截断词表等调用走已有 sampler / logits 路径。新增 head 实现使用 eager kernel launches。

## 性能与正确性

### 端到端结果

`ms_round` 来自预热后的 decode 墙钟时间除以投机轮数。以下各批 DSpark 深度均为 5。

| 批次 | ms/轮 | token/轮 | token/s | 结论 |
| --- | --- | --- | --- | --- |
| 接手基线 | 21.683 | 3.704 | 约 169 | 同机复测 |
| 基线复查 | 21.723 | 3.704 | 约 169 | 同机复测 |
| 01 | 21.200 | 3.704 | 173.151 | attention / router 首批优化 |
| 02b | 20.924 | 3.704 | 175.421 | eager TP-local greedy head |
| 03 | 20.564 | 3.704 | 178.496 | prep 与 head 路径整合 |
| fixed10 | 20.545 | 3.704 | 178.629 | 固定 DSpark5 回归 |
| fused11 | 20.285 | 3.704 | 180.911 | 首次完整 decode、prefill、7K 文本通过 |
| restored18 | 20.290 | 3.704 | 180.898 | 常规测试通过，追加 prefill 非法访存 |
| nohead19 | 20.576 | 3.704 | 178.403 | 关闭 greedy head 后仍非法访存 |
| noearly21 | 20.481 | 3.704 | 179.191 | 主 stream prefill 接收，四轮长度 sweep 通过 |
| final22 默认配置 | 20.454 | 3.704 | 179.449 | 重建后完整测试、四轮 prefill sweep 通过 |
| final22 压测后复测 | 20.436 | 3.704 | 179.602 | 同一服务重复 decode 验证通过 |

fused11 的 14,253-token prefill TTFT 为 2.72 秒。restored18 首轮为 2.73 秒。单轮通过记录需结合以下稳定性记录解释。

### 稳定性

关闭 `PP_RECV_EARLY` 后，noearly21 的初始 prefill 和追加三轮 489 / 1,948 / 3,894 / 7,785 / 14,253-token sweep 均通过，14K TTFT 为 2.92–2.95 秒。7,078-token 真实文本检查通过。该批首次长文本调用漏传结果标签，打印完整文本后因客户端 `IndexError` 退出，补齐标签后在同一服务上完成追加检查。

以上结果支持将主 stream 接收作为交付默认配置。后续优先排查提前接收路径的跨 stream 分配、使用与回收顺序。具体故障位置及其与其它 stream 的关系尚未确定。重复测试通过的边界为本文件列出的单请求长度和次数。

final22 在重建后使用 launcher 默认值，初始和追加三轮 prefill 均通过。两次独立启动合计通过 8 轮五种长度的 sweep。final22 的 7K 文本通过，追加 sweep 后 decode 再测为 20.436 ms/轮。完整输出为 `~/v41/logs/codex_final_22.log`。

此前非法访存集中在 Dev-4 TP0，异步调用栈曾落在 attention QDQ 的 Triton launch，该位置仅作为排查线索。历史故障按节点保存在 `~/v41/logs/codex_failure_17c_rankR.log`、`codex_failure_restored18_rankR.log`、`codex_failure_nohead19_rankR.log`，R 为本节点 PP rank。`codex_debug_20.log` 记录了 `CUDA_LAUNCH_BLOCKING=1` 下三轮 prefill 和长文本通过，诊断延迟包含同步开销。

### 算子回归

最终回归命令与输出见 `~/v41/logs/codex_final_checks.log`。

| 检查 | 覆盖 |
| --- | --- |
| `decode_kernel_check.py` | 806 个路由 / gate / decode 案例，含两种分组 |
| `moe_kernel_check.py` | 67 个基线案例，分组等价检查扩展到 109 个 |
| `attention_post_check.py` | 112 个逐位一致案例 |
| `attention_prep_check.py` | 87 个逐位案例及 2,160,640 个有限 FP32 编码值 |
| `greedy_head_check.py` | 137 项，含 TP4 合并、tie、padding、变化输入和 fallback guards |
| `moe_swiglu_check.py` | 87 个逐位案例，含正常 MoE apply 和变化 graph 输入 |

冻结基线摘要位于 `native/references/`，来自 `7257b3c1f` 的确定性合成输入。分组检查同时核对 slot 覆盖、专家归属及规范化输出。

~~~bash
cd ~/v41
cluster/stop.sh
cd src
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD
P=benchmarks/deepseek_v41/native
~/v41env/bin/python "$P/decode_kernel_check.py" --reference "$P/references/decode_7257b3c.json"
~/v41env/bin/python "$P/decode_kernel_check.py" --reference "$P/references/decode_7257b3c.json" --match-groups
~/v41env/bin/python "$P/moe_kernel_check.py" --reference "$P/references/moe_7257b3c.json" --match-groups
~/v41env/bin/python "$P/attention_post_check.py"
~/v41env/bin/python "$P/attention_prep_check.py" --reference "$P/references/attention_prep_7257b3c.json"
~/v41env/bin/python "$P/greedy_head_check.py"
~/v41env/bin/python "$P/moe_swiglu_check.py"
~~~

微测前停服务。延迟测试每次操作前读取至少 64 MB 冲刷 L2，并使用 CUDA Graph。`q8ks_bench.py` 已补充该条件。profile 分项用于定位，端到端验收采用普通运行结果。

## 源码地图

| 文件 | 此次主要改动 |
| --- | --- |
| `v41k/v41sk.cu` | warp router、gate score 并行化、attention epilogue、fixed-K MoE、W13/SwiGLU |
| `vllm/models/deepseek_v41/attention_ops.py` | 融合 attention dispatch 与 fallback |
| `attn_fuse.py` | register-resident prep |
| `quant_rope.py` | power-of-two reciprocal 与 FP8 编码 |
| `vllm/models/deepseek_v4/common/ops/fp8_software.py` | 有限值 E4M3FN RNE bitcast 编码 |
| `native.py` | 路由、MoE dispatch 与 fused activation |
| `greedy_head.py` | TP-local top-1、greedy verify 和适用条件 |
| `model.py` | attention / target-head 接入 |
| `vllm/v1/worker/gpu_model_runner.py` | sampler 调用与 logits 跳过条件 |
| `native/cluster/build.sh` | 汇总所有节点实际构建退出码 |
| `native/cluster/bench.sh` | 传播预热和测试失败，修正省略 prefill 的退出码 |
| `native/decode_ledger.py` | 在 prefill 暂停点截断 decode 统计 |

## 当前瓶颈

采用 `codex_optimized16_stageR.sqlite`，各段 TP0、44 个 decode 轮次。nsys 采样轮期约 21.37 ms，包含采样开销。

| PP 段 | GPU 活跃区间并集 |
| --- | --- |
| 0 | 3.363 ms |
| 1 | 4.966 ms |
| 2 | 4.641 ms |
| 3 | 5.817 ms |
| 合计 | 18.787 ms |

单请求下四个 PP 段依次执行。各段活跃区间内包含并行 stream，按算子类别相加会重复计入重叠部分。

- MoE GEMM 合计约 7.38 ms。target W13/SwiGLU 约 117–124 µs/层，W2 约 56–59 µs/层。
- 稠密 qkv 约 16 µs，wqb、woa、wob 各约 19 µs。
- 共享专家 gate/up 约 28–31 µs，mHC mixes 约 50–53 µs，多数与主 stream 重叠。
- decode attention split 约 11 µs，combine/post 约 6.5 µs，prep 约 5.8 µs。
- Dev-4 target MoE 后收尾约 2.85 ms，含 target head、采样、draft 模型及 draft head。
- target head 约 393 µs，draft head 约 389 µs。draft main projection 约 98 µs，Markov block 约 28 µs × 5。

后续可研究小 M dense FP8、MoE 权重带宽及 Dev-4 收尾。当前交付保留固定 DSpark5 的已测路径。

## 已撤回或未采用的实验

| 实验 | 结果 |
| --- | --- |
| target anchor 独立 attention | 单算子数值通过，端到端 token/轮降至 3.590，撤回 |
| PP 边界 LL broadcast | decode 20.254 ms、3.704，prefill 非法访存，撤回 |
| target head 私有 CUDA Graph | 较早批次长 prefill 非法访存，交付采用 eager head |
| confidence scheduling | 用户指定固定 DSpark5 后撤回 |
| attention split parts 2/4/8 | 数值通过，收益不足 |
| 8-head Volta MMA attention | 数值通过，实测较慢 |
| fixed-shape dense FP8、N16/N8 | 冷缓存未获得持续收益 |
| MXFP4 uint4 成对读取 | 实测较慢 |
| 跨节点 TP8 | 通信测试较慢，沿用 PP4 × TP4 |

相关 patch / log 保存在 `~/v41/logs/codex_*`。restored18 和 nohead19 也发生故障，PP 广播和 greedy head 的撤回记录无法单独证明根因。

## 同步、构建与回滚

源码编辑在 Dev-1 的 `~/v41/src` 完成。修改后同步，修改 CUDA 后构建四个节点。

~~~bash
cd ~/v41
cluster/sync.sh
(setsid nohup cluster/build.sh > logs/build_all.log 2>&1 < /dev/null &)
~~~

`sync.sh` 默认依据 Git 工作区改动选文件。已经提交的源码或恢复为 HEAD 的文件需显式传路径。删除试验文件可传 `--delete-missing-args` 和文件路径。

源码内 launcher / 工具复制到部署目录后生效。

~~~bash
cd ~/v41/src
P=benchmarks/deepseek_v41/native
cp "$P/run_pp4.sh" "$P/run_pp4_nsys.sh" "$P/serve-native-pp4.yaml" ~/v41/
cp "$P/bench_client.py" "$P/long_check.py" "$P/prefill_bench.py" ~/v41/
cp "$P/profile_both.py" "$P/decode_ledger.py" "$P/prefill_ledger.py" ~/v41/
cp "$P/cluster/"*.sh ~/v41/cluster/
~~~

完整回退时，停服务、保存当前源码提交，再 checkout `7257b3c1f`。将该提交的完整源码和 launcher 同步到各节点，重新构建，再跑端到端测试。仅关闭五个新开关仍保留此次无开关的 router、gate 和 FP8 编码优化。

## 静态检查与 PR 审阅边界

Ruff check/format、SPDX、禁止导入检查、ShellCheck、环境元数据、生成文档和 runtime configuration ownership 检查通过。集群脚本另通过 8 项失败传播检查及 8 个 shell 语法检查。CUDA 改动行通过 clang-format，完整格式化后的内容与 fused11 的已测 CUDA 源码一致。

早期提交遗留的 typos、mypy、torch.cuda 调用、开关注册和 layering 问题，在合入公开 main 时（PR #1482）已处理：V4.1 开关全部在 `vllm/envs.py` 注册并经 `vllm.envs` 读取，`torch.cuda` 调用改为 `torch.accelerator`，typos 与 clang-format 按 CI 修正，layering 基线只记录减少。

## 操作约束

- 代码在公开仓库 `1CatAI/1Cat-vLLM` main，集群 `~/v41/src` 为部署副本。
- Python 使用 `~/v41env/bin/python`，依赖通过 `~/.local/bin/uv` 管理。
- 每批验收要求 token/轮至少 3.70，并执行 7K 真实文本和重复 prefill。
- 微测前停服务，单卡计算测试设置 `CUDA_VISIBLE_DEVICES=0`。
- 硬件、时钟、网卡调整需要用户明确批准。
- NAS `/vol1/models` 和 10-22 客户机器属于受保护范围。
- 长任务使用 detached job 并检查日志。交接仅保存无密码的访问路径。
