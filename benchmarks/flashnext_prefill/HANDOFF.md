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
