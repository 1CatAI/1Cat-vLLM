# Flash-Next prefill 优化交接（9683，4×V100 16 GB）

更新：2026-10-13。分支 `agent/flashnext-16gb-fast-20261011`。本目录下的脚本都是这轮工作用过的测量工具。

## 1. 目标与约束

- **目标**：32K prompt 的 prefill 达到 ≥5,000 tok/s，即 34.7K token 的 TTFT ≤ 6.9 s。充分利用显卡、降低延迟。
- **当前**：7.50 s（4,616 tok/s），还差约 0.6 s。
- **部署形态不能动**：
    - 模型：Flash-Next IQ3_S + MTP4，256K 上下文，开启 prefix caching 和 tool calling。
    - 内存分布：QSA KV 放主机内存，PLE 放磁盘；路由专家 80% 在 GPU、**20% 在主机内存（必须保留）**。
    - chunk 8192；thinking 开启；采样 T0.6 / top_p 0.95 / top_k 20。
- **用户的工作要求**：
    - 先看延迟分布再动手，不要盲调。
    - 小改动先用单测或微基准验证，**攒够收益再做一次端到端验收**，不要每个小改动都跑端到端。
    - 工具调用的 timeout 不超过 120 s。长任务用 `setsid nohup` 脱离会话后轮询。
    - 启动时间保持约 5.5 分钟。
    - KV 要给其他显存用户让路。暂不做 piecewise graph。
    - 不在共享 venv 里安装包。改系统设置前先问用户。
    - 不碰其他 agent 的进程。遵守 GPU 锁协议（见下）。
    - 避免主机 OOM：memguard 必须开着。
    - strata 服务可以停，不能 disable。
    - 不使用、也不复述任何密码。

## 2. 环境与操作

- **登录**：`ssh test-9683`。工作目录 `/home/test/flashnext-256k`，代码在 `claude-src/`。部署方式是把本地 worktree 的文件 rsync 到 `claude-src/` 对应路径。目前远端与本分支一致。
- **服务**：
    - 启动：`cache-serve.sh`，用 `setsid nohup ./cache-serve.sh &` 启动。
    - 停止：`claude-stop.sh`，按进程组停。
    - 内存保护：`memguard.sh "[c]li.main serve"` 必须同时运行。
    - 端口：API 8001；代理 8011（`api_proxy.py`）经隧道 9537 对外。
    - 日志：`logs/cache-serve.log`。每次启动写一行 `=== start`，启动完成看 `Application startup complete`，约 5.5 分钟。
    - `cache-serve.sh` 现在导出了 `VLLM_BALANCED_PREFILL_CHUNKS=1`。
- **GPU 锁**：`claude-py.sh` 和 `claude-test.sh` 会拿 `/tmp/gpu*.lock`。服务启动时也拿这些锁，所以跑单测或微基准前要先停服务（显存已被占满）。
- **跑脚本**：
    - `claude-py.sh <log> script.py args`：在 4 卡上跑 python。
    - `claude-test.sh <log> tests/...`：在 GPU0 上跑 pytest。
- **编译**：
    - `claude-build.sh`：增量编译并安装，`TARGET=_C` 或 `_sm70_qsa_device_C`。
    - `claude-build-noinst.sh`：只编译不安装，服务运行时也能编，用 `QSA_SO=cache/claude-build/...so` 加载测试。
    - 不要在服务运行时覆盖安装已加载的 .so。
    - `csrc/gguf_expert_cache.cpp` 是 JIT 扩展（`torch.utils.cpp_extension.load`），改动后下次启动会自动重编。
- **nsys 剖析**：
  1. `PROF='{"profiler":"cuda"}' setsid nohup ./claude-serve-nsys.sh &` 启动带 nsys 的服务。
  2. 先预热两次请求。
  3. `curl -X POST localhost:8001/start_profile` → `python profile_request.py 32000` → `curl -X POST localhost:8001/stop_profile`。
  4. `nsys export --type sqlite logs/nsys12k.nsys-rep`。
  5. 分析：`python nsys_breakdown.py x.sqlite 0`。
    - nsys 开销约 3%。不要用 torch profiler，它会把主机侧时间放大约 25%。
- **磁盘**：根盘曾被写满，原因是 `cache/claude-run/vllm/torch_compile_cache`。磁盘满会让 stop_profile 静默杀掉 worker、启动脚本静默退出。
    - 启动前先 `df -h /` 看一眼。
    - 清理超过 3 小时的编译缓存。
    - nsys 报告不要留在远端。
- **测试需要离线 HF 配置**：scheduler 单测要 `facebook/opt-125m` 的配置，但 9683 连不上 HF。用伪造的离线缓存：`HF_HOME=/tmp/claude_hf HF_HUB_OFFLINE=1`。这个目录只放了一个 config.json，所以 max_model_len 上限是 2048。

## 3. 验收方法

- **TTFT**：`python ttft.py 32000`，实际约 34.7K token，分 5 个 chunk。重启后第一次请求偏慢 1 s 以上，取第 2、3 次。
- **长上下文检索**：`python needle.py`，必须 2/2。
- **Decode**：`bench_round.py`（远端根目录），正常是 15–20 ms/round。
- **QSA 真实数据微基准**：`qsa_dump_bench.py logs/qsa_dump/qsa_prefill_0.pt run_rows 12`。
    - 数据来自服务设 `VLLM_QSA_DUMP_DIR=<dir>` 时由 `qsa.py:_dump_prefill_inputs` 导出（TP0，≥8000 行，ctx ≥24K，层 0/5/11）。
    - 每份约 155 MB。

## 4. 本轮成果（32K TTFT 8.08 s → 7.50 s）

| 改动 | 32K TTFT | 说明 |
|---|---|---|
| 起点：QSA rows_fp8 kernel | 8.08 s | 每层每 chunk 的 QSA 从 22 ms 降到 12.5 ms，见 commit e756ed979 |
| 冷专家改 DMA 暂存 | 7.80 s | 见下 |
| 均分 chunk | 7.62 s | `VLLM_BALANCED_PREFILL_CHUNKS=1` |
| KV zeroer 跳过主机 KV（V2 runner） | **7.50 s** | 每步少写约 160 MB 零，约 17 ms |

其他长度：17K 4.09 s，8.7K 2.25 s，62 token 1.03 s。

### 冷专家 DMA（`gguf_expert_cache.py: TurboMindPrefill`）

- 原来放在主机内存的冷专家由 GPU 线程以 zero-copy 方式直接读（8 个 CTA）。在服务里，这会让并发的主 stream kernel 变慢最多 2 倍：HC down 从 0.59 ms 变成 1.2 ms，32K 一次请求多出约 1.3 s。这个现象在单独的微基准里复现不出来。
- 现在的做法：lane 的 `residency()` 返回冷专家所占主机 slot 的范围，copy engine 把这段 `cudaMemcpyAsync` 到设备上的暂存区（`self.stage`，约 68 MB/GPU），再用 64 CTA 从设备内存转码。
- `VLLM_TM_COLD_DMA=0` 可切回旧路径做对比。

### 均分 chunk（`scheduler.py: _balanced_prefill_chunk`）

- 每个 prefill 步都要把冷专家从 PCIe 搬一遍，约 0.93 s，与 token 数无关。所以短的尾 chunk 也要付一整步的代价。
- 现在把长 prompt 按 816 token（注意力 block 大小）对齐后均分。34.7K 由 8160×4 + 2083 变成 7344×4 + 5347。

### 不要再试的方向

- 强制 `NCCL_PROTO=Simple`：没有收益。decode 的 all-reduce 走 custom AR，只有 prefill 走 NCCL。
- 冷专家转码改用 2 个 CTA：TTFT 变成 12 s，PCIe 吃不满。
- 第二套 TurboMind bank（`SLOTS=2`）：显存不够，缓存分配器开始抖动，TTFT 变成 13–18 s。
- 给 KV 设上限：KV 本来就是固定的 state pool，设上限省不出显存。
- QSA：
    - 用 atomics 做融合 merge：更慢。
    - tile-union：行与行之间选中的 block 几乎不重合。
    - 每 SM 3 个 CTA：慢 2 倍。
    - 用 Triton 写融合 indexer：这版 Triton 在 sm_70 上不生成 tensor core MMA，比现有路径慢 60 倍。
- ncu 在 9683 上不能用：驱动不兼容，也没有计数器权限。只能靠 nsys 加消融实验。

## 5. 当前延迟分布

数据：32K、4×8K，GPU0 关键路径，约 6.4 s。已扣除修好的 zero。

| 模块 | 时间 | 占比 | 构成 |
|---|---|---|---|
| MoE 路由专家 | 1.36 s | 21% | gate\|up 0.76，down 0.43，unroute 0.11，路由/排序 0.06 |
| HC + 跨卡通信 | 1.39 s | 22% | AllGather 0.35，AllReduce 0.42，combine+norm 0.32，HC down 0.18，小 kernel 0.12 |
| QSA 稀疏注意力 | 0.90 s | 14% | rows_fp8 0.57；draft fp16 partial 0.08，top-k 0.08，`_write` 0.07，relu_headsum 0.06，其他 |
| TurboMind dense 投影 | 0.87 s | 14% | 已经约 62 TFLOPS |
| GPU 空闲 | 0.82 s | 13% | 主机来不及下发 0.15；等事件 0.67，分散在 MoE 路由、HC、restore、QSA 之间 |
| 其他 cuBLAS / GDN / elementwise | 0.91 s | 14% | |

另外有两点：

- 冷专家 PCIe 每步约 0.93 s，在独立 stream 上与计算并行，满 8K 的步里被完全掩盖。
- 请求处理在 GPU 之外还有约 0.3–0.5 s。

## 6. 建议的下一步（按性价比排序，括号里是预估收益）

1. **QSA 剩余项（约 −0.2 s）**：
   - draft 模型的 fp16 attention 目前走 `run_tiled` 加 partial/merge，改成 rows kernel 的 fp16 版本，见 `qsa.py` 里 `host_kv_forward` 的 staged 分支。
   - 手写 CUDA 融合 indexer：mma.m8n8k4，4 heads × 128，relu + head-sum + visible mask，输出 fp32 logits。替换 `ops/qsa.py:_qsa_mqa_cublas` 里的 `torch.mm` 和 `_qsa_relu_headsum_visible_kernel`，验收要求 top-512 选择一致。
   - 把 `HostQSAKV.write` 挪到 side stream。
2. **HC 小 kernel 融合（约 −0.1 到 −0.15 s）**：把 `_hc_combine_norm_kernel`、gate_mix、assemble、packet_silu 合并，代码在 `ops/hc.py` 和 `sm70_fp16_hc.py`。
3. **MoE 路由里的同步与小 kernel（约 −0.05 s）**：`_sub_bounds`、cub scan/sort、`fill_reverse_indices`，以及 `topkGating` 之后的 D2H memcpy。
4. **更大的方向**，单靠前三项到不了 6.9 s：
   - 通信与计算重叠：AllReduce 有一半是在等其他卡，可以按 token 切开做流水。
   - MoE grouped GEMM：现在约 30–35 TFLOPS，每个专家只有约 160 行。

## 7. 坑

- 服务用的是 **V2 runner**（`vllm/v1/worker/gpu/model_runner.py`）。改 runner 时 V1、V2 都要改，并用日志确认改动生效。
- 微基准在空闲 GPU 上测出来的结果，不能代表服务里的并发表现，最终要以 nsys 为准。`nsys_side_stream_excess.py` 用来找"side stream 拖慢主 stream"这类问题。
- `pkill -f <模式>`：如果模式也出现在你自己的 ssh 命令行里，会把自己的会话杀掉，ssh 返回 255。
- `HostQSAKV.write_staged` 的 `Blocks` 是 constexpr，必须固定为 `self.blocks`。每步传不同的值会让 Triton 每步重编，TTFT 多出约 0.7 s。
- `cudaHostRegister` 注册的内存，在 torch 看来 `is_pinned()` 是 False，`.copy_(non_blocking=True)` 会退化成同步拷贝。要异步 H2D 用原生的 `copy_h2d`。

## 8. 9683 MTP HC 显存诊断（2026-10-15）

主模型的 HC 行分片不能代替草稿模型的显存验证。启用 MTP 时必须保留
已验收的 FP16 GEMM reduction 策略；旧主模型 admission 的 reduction=False
条件曾使配置请求了行分片，却没有实际执行。移除该条件后，主模型路径
实际命中，但 8192 行的草稿 `mtp.layers.48` 仍在 HC combine/norm 申请
两个 160 MiB 输出；分配上限 384/512 MiB 留余量的整模型试验仍然 OOM。

候选 `kernel_config.layer_execution.hc_mtp_prefill_sequence_parallel` 默认
关闭，独立于主模型行分片配置。其加速范围是 SM70、FP16、HC4/H2560/L320、
sharded HC、Q8_0 MTP experts、TP4/PP1/DCP1/PCP1、C1、BACKED 动态形状，
无 LoRA/诊断，实际 query 行数 4096..16384，非 CUDA Graph prefill。
范围外继续原草稿模型；MTP 的小形状 decode 编译和 FULL Graph 不受该开关
影响。输入 target hidden 只借用，不原地修改；草稿 attention/expert 保留
原归约，HC 行分片复用现有投影/通信/归一化实现，最终恢复原 token 顺序和
两路完整输出，padding 行不暴露给 proposer。

启用该候选时，普通草稿回退模型使用独立的 UNBACKED token 维度编译配置，
并声明 ids、position、hidden、embedding 的行数关系。大 prefill 由新 view
处理后，原回退可能首次在 M1 编译；BACKED/AOT 关闭 guard 会将这一维度
固化，后续 M6 的投影/position 错位。最小配对回归中，旧 BACKED 路径在
M6 仍返回 M1 输出；UNBACKED 在首次 M1、随后 M6/M16/M2/M1/M64 及 AOT
缓存重新加载时保留正确行数。只有完整满足上述 admission 的模型复制
这一配置；主模型、decode view 和未加速模型保留原配置和计算策略。

`hc_mtp_sequence_runtime.py` 验证真实 MTP/decoder/HC adapter 的数值、行边界、
编译、变输入 graph replay、target hidden 不变和输出所有权；合成 producer
测试不能替代整模型质量或性能。候选尚需整模型 cold prefill、长文任务、
显存余量与验证成本验收。不要把验证 kernel 总和、MTP 轮间隔或 tok/s 当成
验证成本：长文使用 TP4 最早 target graph GPU 开始至最晚状态提交的完整
wall time，同时记录尾部和不完整轮次。

主模型的 PLE 还会同时保留 full-row hidden、key、gated value、归一化输入
及重复卷积输出。HC 行分片路径现在将 PLE 查表/卸载 ABI 保留为完整行，
查表的 hidden 参数只提供执行依赖，直接借用当前 TP 的局部行；显式传入
完整 query 长度供 IQ4_NL 解码，保留 packet/FP16 结果的完整行和未使用
尾部契约，省去 PLE 前 full-hidden all-gather 与再次分片。embedding
在投影和逐 token 门控前分片；只将归一化后的序列恢复为完整行供原卷积
和状态提交使用。卷积原生家族已返回独立结果，新 Python custom op 复用
该结果，避免另一个完整输出的清零/拷贝；不足完整行的结果保留旧零尾部
契约。未分片调用保持原输出缓冲 API。

TP4 `ple_sequence_rows.py` 使用真实 PLE forward、GroupedNorm 和原生
SM70 H10240/K4/dilation3/state9 卷积。4096/4097/7344/7345/8192/16384
行的 eager 配对输出/状态最大行 L2 <0.00050（预设上限0.001）；7344/7345
行同编译配置、局部依赖输入的配对最大行 L2 <0.00014，并通过变输入 graph replay、padding、
未使用状态槽和额外状态列检查。编译后 PLE live tensor 峰值增量分别从
466.23/466.84 MiB 降至295.86/332.32 MiB。微基准不替代整模型物理余量
或性能验收。旧编译路径本身与 eager 的输出距离可到0.0066，故编译配对使用
同配置编译原路径作控制；该对照差异保留，不放宽0.001配对限值。

### 普通 Worker 的卷积峰值回归

上述显存探针在 prefill 步末调用 `cudaMemGetInfo` 和 allocator 查询；其
执行节奏不能证明普通 Worker 的临时张量生命周期。`4881b5bed5` 在普通
Worker 首次 cold 32K 的第三个7344行 chunk 仍 OOM：PLE 卷积申请完整
144 MiB 输出时，Torch 已用13.92 GiB、保留但未分配317.66 MiB、物理
剩余641.94 MiB，分配器上限14.36 GiB。该部署不具备上线条件。

候选继续复用原 PLE Triton 卷积家族：完整输入和每个请求的因果历史、
state commit 保持不变，仅向当前 TP owner 的输出行写入结果。7344行的
输出从每卡144 MiB 变为36 MiB；不扩大混合 decode/spec 的加速范围。
这些形状继续原分发后取局部结果；未分片调用和旧 custom-op ABI 保留。
新返回值有独立存储，padding 为零；空输出仍提交全部有效请求状态。

22项针对性检查和132项相关回归已通过。四卡六种大形状及变输入 graph
replay 通过原0.001数值限值，7344/7345行 PLE live tensor 峰值增量约
188.28/188.86 MiB。整模型验收必须用普通 Worker、不加载剖析器或显存
探针重新完成；这些微基准不代表最终物理余量、验证成本或 API 已可用。
