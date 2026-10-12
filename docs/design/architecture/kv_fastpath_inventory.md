# SM70 KV 快路径逐项审计

本表审计源码 `19d43870059472a794b86d416c6de73e30003f74`，日期为
2026-10-11。集成基线为 `onecat/main` 的
`be3722bc095b9792bf9833068595f0b1ef70ad37`；当前实施分支基于 grouped
阶段的 `dea7beec35eb1f281864cbd6a03fb6bb0066b061`。
实现与历史证据见 [KV 统一工作记录](kv_codec_unification.md)。

**目前不能宣称 INT8 已完整复用 FP8 的全部加速。** 普通分页读写、基础
XQA、grouped 和分页 prefill 已有公共绑定，但同一个入口内部仍有格式专用
调优，resident-prefix dense 转换链尚未通过 INT8 数值资格，稀疏表示及部分
引擎生命周期也未闭合。本轮只审计并记录，不调整运行策略或放宽数值门槛。

## 2026-10-12 实施进展

此表继续保留 43 项责任清单。新增公共 writer 和 XQA 计划已通过组件检查，
完整质量、延迟及引擎生命周期仍未验收；未关闭项不会被通用回退抵消。
具体版本、失败及构建记录见 [统一工作记录](kv_codec_unification.md#shared-native-writer-and-xqa-execution-plans-2026-10-12)。

## 审计方法与口径

交叉检查引擎入口 → 选择器 → Python 包装 → 原生实例 → 兜底，以及原生
注册 → 正常构建 → Python 调用方 → 引擎可达性。源码中共有 **60 个静态
Flash-V100 路由名称**；Flash-V100 公共扩展有 **35 个不带 `_configured` 后缀的注册
入口**，其中包含兼容入口、诊断、版本查询和 backward。它们都不是独立快路径
数量，不能用命中总数计算完成百分比。

本表覆盖当前 SM70 引擎中的普通 attention、推测验证、写缓存、QSA、MLA、
indexer 及其图和搬运边界。ROCm、SM80+ 专用后端、训练 backward、未纳入正常
构建的历史源码不计入 V100 快路径完成分母；不据此宣称其他平台支持 INT8。
`q` 为每请求 query token 数，`B` 为请求数，`D` 为 head dimension，GQA 为
query/KV head 比；page 和 partition 大小均以 **token** 为单位。

- **已绑定**：源码存在原生实例、scale 传递及调用方，仍须按形状区分 GPU 证据。
- **缺接入**：算法或其有效调优分支仍排除 INT8，通用回退不抵消此缺口。
- **缺资格**：实现存在，但数值、状态或性能证据不足；不等于已经合格。
- **原有边界**：FP8 原本也没有该能力，不能伪报为遗漏的 FP8 路径。
- **独立表示**：潜变量、RoPE、压缩或旋转有真实数学差异，需要表示级 codec。

静态路由的完整名字及声明保留在[生成的参考表](runtime_reference.md)，本表按
算法和内部变体归类。`_configured` 包装、版本/ABI 查询、launch counter、
capture/metadata observer 不重复算作算法。`decode_dense_cache`、
`decode_dense_reference`、QK-score 诊断与显式 decode-as-prefill 用于调试或
对照，不当作默认生产快路径。按[正常构建清单][public-build]核对，
`fused_mha_forward_paged_full.cu` 等未编入当前扩展的历史文件也不计入。

## Decode 与原生 XQA 内部分支

主要源码：[引擎 decode][decode]、[公共选择器][routing]、[Python XQA 包装][interface]、
[原生 decode/XQA][native-decode]、[codec traits][traits]。

| 编号 | 实际算法或优化 | FP16 / FP8 现状与主要边界 | INT8 状态及责任位置 |
| --- | --- | --- | --- |
| D01 | 基础 XQA，公共 partition 与 reduction | D256、GQA4/6/8，partition 256/512/1024；E4M3 另有 GQA6 和批次资格 | 已绑定。`decode_paged_xqa_kv_fwd` 经同一 codec visitor 实例化；已有组件 GPU 证据，完整引擎/延迟资格未完成 |
| D02 | padded shared memory、通用 dual-CTA、split reduction、block4/16/784 索引；page784 auto/sawtooth | 原生通用分支按几何和策略组合，不全部属于 E5M2 专用分支 | 公共计划及 INT8 scale view 已接入；44 个 XQA GPU 用例覆盖非连续页、页几何和变输入 replay，仍缺主要变体独立延迟及完整引擎证据。 |
| D03 | E4M3 B1 p64/p128，p64→p256，p512/p896/p1664 wave；merged-wave launch/reducer | `KV_DTYPE == FP8_E4M3` 分支；D256/GQA6，页对齐与阈值检查 | 已接入公共量化 codec 模板：p64/p128、动态 wave 与 merged-wave。四种 byte codec 的阈值相邻点及 replay 通过；未完成同条件性能验收。 |
| D04 | E4M3 batch/context dual-CTA、split reduction、page800/page1568 固定布局 | E4M3 batch policy、uint8 和 stride 条件控制 | 已改为 byte codec、布局及 FP32 partial 条件，page800 固定布局与 B4 GPU/replay 通过；其他容量和性能范围仍须分别验证。 |
| D05 | E5M2 B1 p256/p1024 动态分区、one/two CTA、partition page-ID 复用、paired load | 原生 E5M2、Hkv1、16 对齐页、未显式覆盖 partition 的分支 | 动态分区、one/two CTA、页索引及 paired-byte reader 采用公共 codec/实际对齐；非连续页与分区图测试通过。未对齐旧布局使用公共合法读者，不伪装对齐。成本资格未闭合。 |
| D06 | E5M2 批次/context 计划和 batch-wide load | B≥4、GQA6、Hkv1、page16 或对齐大页，带 context hint | Python/native/graph 计划已接入普通 byte codec；B4/page16 和 page800 组件 GPU/replay 通过，完整引擎及批次成本待验收。 |
| D07 | E5M2 MTP5/page1616 dual-CTA | 五行、GQA6、page1616 的旧优化 | 原生 MTP5/page1616 已按五行、GQA6、对齐及资源条件公共化；Hkv1 五行及 Hkv2 的五 codec GPU/replay 通过，性能范围继续单列。 |
| D08 | staged QK/softmax + PV、G6 QK software pipeline | 原来只读 FP16；staged PV 还要求 p256、padded shared memory、dual-CTA | 原有边界。不能算 FP8 已有而 INT8 漏掉；后续可评估公共化 |
| D09 | scalar paged + partition merge | 一般 head dimension/window 的兜底；E4M3 另有 opt-in scalar fast 变体 | 公共 scalar 兜底保留能力检查；新二进制修复内部 E5M2 scalar 优先分支。scalar fast 实例已公共化，但不用于关闭未通过的 XQA 或性能项。 |
| D10 | compact scalar tail | q1、D256/GQA6、long-context 图 descriptor，当前原生 page≥1024；p1024、FP32 state | 已有 INT8 绑定，完整 GPU/图资格未完成。它在 scalar 调用内尝试，不能绕过已合格 XQA 的优先选择 |

`decode.py` 当前顺序为先检查可用且语义兼容的 XQA，再检查 scalar 的实际绑定。
新原生执行计划 revision 2 删除了 E5M2 包装内的短上下文 scalar 优先节点；
旧二进制保留旧能力和策略检查，不继承新覆盖声明。D≠256、当前
XQA 不支持的 GQA/window 等仍须保留合法兜底。E4M3 DFlash target 的旧数值
资格拒绝位于 `spec/attention.py:reject_xqa`，需要独立记录，不能因降低 scalar
优先级而直接删除。

## Prefill 与缓存前缀

主要源码：[批次/序列候选][prefill-candidates]、[prefill 执行][prefill]、
[dense/FA2 选择][dense]、[原生分页 prefill][native-prefill]。

| 编号 | 算法或优化 | 实际边界 | INT8 状态 |
| --- | --- | --- | --- |
| P01 | 无前缀 dense Flash、BMHD/BHMD | 消费当前原始 Q/K/V，尚不读取 resident KV | 不因选择 INT8 存储而必然失效；仍按原始张量/几何准入。不能和缓存前缀路径混为一项 |
| P02 | 普通 paged prefill、带前缀/chunked prefill | 公共 native 支持 D16/32/64/128/256；引擎声明 D64/128/256；保留 causal/window 语义 | 已绑定，五种 codec 身份的组件 GPU 用例通过；更大引擎组合未验收 |
| P03 | 分页 split-KV + merge | D256、q>1、page 为 16 的倍数，切分/容量 hint | 已绑定。INT8/E4M3 使用公共常规 split kernel，E5M2 原来的常规 split 实现相同；FP16 low-smem 子变体见 P06 |
| P04 | BHMD paged、WMMA q1 decode | 同一分页计算，用不同输入输出布局；q1 可作 decode-as-prefill | 已绑定并有组件 GPU 证据；不是默认 XQA 的替代品，也不代表 WMMA 更快 |
| P05 | dense / paged Split-D；contiguous/gather；v37；79t Q8000、Q8192、fringe/tail padding；dense splitkv3 | FA2 dense 算子消费 FP16。FP8 resident KV 经有界 bridge 后复用；Q8000/Q8192、对齐及 scratch 约束各自保留 | Split-D、splitkv3、v37 与 79t 已有保留独立 FP32 scales 的公共原生绑定及 GPU 数值证据；79t 包含 Q8000/Q8192、padding、图重放和引擎前缀调用。**P05 尚未关闭**：79t 延迟仍未达标，其他 FP8 转换可达的优化变体及完整模型验收未完成。早乘 scale 再转 FP16 的失败方案未采用；P01 原始 Q/K/V 仍按自身语义准入 |
| P06 | D256 low-smem、BM32 phase、ALL_P、pair scratch、contiguous load、output stride268、QK/PV software pipeline；固定 FlashInfer-SM70 page784 BM32/splitkv3 | 直接入口原本 FP16-only；一般 low-smem 可由 FP8 bridge 的 FP16 输出复用，固定 FlashInfer 入口本身只准入 FP16 cache | 直接 INT8 未实例化。FP8 bridge 可获得的子集归入 P05 缺口；FP8 从未准入的固定入口单列原有边界。普通 split-KV 通过不能冒充这些变体通过 |
| P07 | BFLA 稀疏 prefill | mask 构造器要求 FP16；消费 block mask 的 native 与 mask 生成是两项能力 | INT8 native 已绑定并通过全一 mask 测试；真实稀疏 mask 生成及引擎路径未打通。旧 FP8 也未直接进入该 mask 构造器 |
| P08 | prefix-anchored window | native 明确要求 FP16、causal，不能和一般 sliding window/BFLA 同时启用 | 原有 FP16-only 边界；不是普通 window 用例通过就已覆盖 |

P05 的失败证据必须保留：冻结的 `atol=0.004, rtol=0.015` 下，早期 INT8→FP16
bridge 出现约 0.047/0.064 最大误差；配对诊断定位到 scale 提前应用与 FP16
舍入。需要保持 row-scale 数值顺序的公共适配，不能通过移除 FP16 gate、放宽
误差界或复制整个 dense executor 来宣布接通。

## Grouped、mixed 和推测解码

| 编号 | 算法 | 当前覆盖 | 未闭合项 |
| --- | --- | --- | --- |
| S01 | compensated grouped FP32 | 公共 FP16/E4M3/E5M2/新旧 INT8；新计划支持 D256/GQA6、B1 q2..16，request-major q8 B2..16、16 对齐页；组件容量上限 266240 token | 新旧计划按实际二进制区分；q16 与变输入 replay 的组件证据已补齐。当前引擎调用、混合请求顺序、MTP 提交/回滚与接受率仍缺完整证据 |
| S02 | FA2 long-context grouped | 同一 80-split/FP32 state 算法，新普通 codec 绑定；B1 q2..8、request-major q8，descriptor 上限受服务窗口及 262144-token native 能力约束 | 代码已接入；INT8 long/tail GPU 和完整图生命周期资格未结束。V2 long-batch 准入已用 native codec 能力，不能推导 V1 其他图策略也已统一 |
| S03 | grouped verifier：q8/q16、两/三 head group、one/two-pass、page-ID staging | 旧 E5M2/E4M3 入口保留兼容；新公共计划沿用同一 Tensor Core 计算、FP32 状态和归约，按视图启用固定 stride、暂存页索引及成对读取 | 五种 codec 的 q9/q16、单/双遍、Hkv1/2、page1648/3296 和图重放共 50 个用例通过；另 5 个用例验证 8 字节合法偏移及实际 q16 INT8 原生实例。旧入口 94 个回归通过。完整引擎/state/性能资格仍未闭合。FP16 page832/短 context 特化仍单列原有边界 |
| S04 | mixed decode rows / small-query 行展开 | 公共 grouped → XQA → 合法 scalar，复用 persistent 表和实际 row lengths | 已接 view/scale；未建立所有批次/树 mask/state 的 GPU 覆盖。不能用单请求 q8 推导所有混合验证形状 |
| S05 | DFlash2 draft noncausal direct BMHD / window-split | D128/GQA4、q8、B1..4、指定页大小；单请求 window split 与并发 WMMA 为不同候选 | 旧优化只支持 FP16 draft KV。用户要求的 INT8 target + FP16 draft 应保留它；显式 INT8 draft 只能走已支持的公共分页候选，不能宣称复用该专用优化 |
| S06 | DDTree ancestor-mask Triton / dense correction | Triton cache 要求 FP16/BF16；dense correction 有量化 cache gather/reference | 原有量化树快路径缺口；view-aware gather 不等于 INT8 tree 数值与图资格。V1/V2 默认差异见 L02 |

入口与边界分别位于 [verifier][verifier]、[grouped contract][grouped]、
[long wrapper][long]、[compact tail][tail]、[DDTree kernel][tree]。

## 写缓存、转换和融合

| 编号 | 路径 | 当前判断 |
| --- | --- | --- |
| W01 | Triton 普通动态量化写入、统一 attention 读者 | 已接公共 codec/view，新 INT8 RNE planar 与旧 INT8 truncate inline 身份不同；writer/scalar/XQA 组件已有 GPU 证据。不能让二者按相同字节宽度互读 |
| W02 | `_C_cache_ops.reshape_and_cache_flash` 原生 CUDA 写入 | 已接现有 CUDA writer，可选 FP32 scale-plane ABI 保留旧八参数调用。21 个 GPU 字节/异常/槽位/replay 用例通过；旧 inline misalignment 和历史 truncation 边界均已修复并记录。性能仍待验收。 |
| W03 | 目标 Norm/partial RoPE/KV、draft Norm/NeoX RoPE/KV 融合 | 两种融合均接公共 encoder/view，五种 codec 共 10 个 GPU 字节兼容及变输入 replay 用例通过；目标 D256/GQA6、draft D128/GQA4 的几何/模型侧资格仍存在，未证明所有模型自动调用 |
| W04 | paged KV→FP16 bridge、contiguous/gather、尾部填零 | 公共转换已有 15 个组件 GPU 用例；转换字节/读写正确性与 P05 的下游 attention 数值资格分开。旧无 codec 的 `paged_kv_utils` 不得直接读取 planar INT8 |
| W05 | MLA QNorm/RoPE/插入与 indexer quant/cache | 公共 mixed MLA／indexer codec/view 已接现有 QNorm/RoPE、C4/C128 compressor；10 项新增 GPU 融合检查通过，保持旧 FP8 缩放顺序和模型侧数值资格。引擎命中、质量和性能待验收，见 [mixed MLA 记录](kv_mla_mixed_20261012.md) |

来源：[统一 writer 分派][triton-backend]、[CUDA cache writer][cache-native]、
[native 格式宏][cache-dispatch]、[目标融合][target-fused]、
[draft 融合][draft-fused]、[缓存转换与 gather][layout]。
W02 是缺原生绑定的事实，不是已经测得 Triton 比 CUDA 慢；接入与速度要分别验收。

## QSA、MLA、indexer 及其他表示

| 编号 | 算法族及必须分开的子路径 | INT8 判断 |
| --- | --- | --- |
| R01 | QSA dense prefix、稀疏 paged split-K/merge、grouped page4 planner/attention、尾行 XQA、按请求拆分 | 公共 view/planner、原生 XQA/grouped 与 Triton split-K 已接 INT8；主 owner/backend 和 writer 已接入。79 项 GPU 组件检查通过，并确认实际 codec-3 原生执行；小页/padding、尾行和变输入图均覆盖。完整 QSA 引擎、质量、延迟及生命周期仍开放，不以这些组件结果关闭 R01 |
| R02 | QSA 压缩键 scorer、cuBLAS gather/scoring、shared-key、lexicographic top-k、physical-index resolution、DCP | 独立压缩表示和 top-k 算法保留公共语义；两 rank 的 FP16/E4M3/INT8 output/LSE 组合通过。新增 mask 用例发现并修复原有 grouped planner 的无效首索引/尾行问题，门槛不变。FP16-only resolved-index 特化保留原有范围；全链引擎、压缩及索引状态资格仍未关闭 |
| R03 | host history 写入/保护、FP16 hot cache、miss gather、直接 host/device history attention | FP16／动态 E4M3／RNE INT8 共用 codec/view、写入和读取；正常 CMake 原生实例 0/1/3 已绑定，144 项 GPU 组件检查通过，其中追踪确认 `partial<3>`。hot cache 保持 FP16。实际完整引擎、独立 scale owner 的 offload/sleep/transfer、质量和性能仍待验收，不能以组件通过关闭 L04/L05 |
| R04 | DeepSeek sparse MLA：paged、split-K、QK-D split、gathered/BMM prefill；压缩与 SWA | 448 latent／64 BF16 RoPE 的公共 G64 view 已接原有执行器、融合 writer、压缩和 SWA 分配。FP8 584-byte/UE8M0，INT8 608-byte/FP32 scales；40 项新增 GPU 读写/算法及 10 项融合检查通过。完整安装、代表性引擎质量、状态生命周期和性能仍开放，见 [mixed MLA 记录](kv_mla_mixed_20261012.md) |
| R05 | GLM NoPE latent MLA：直接 paged、B1 gather+GEMM、M2..8 batched GEMM、较大行数 paged | 公共 G64 view/encoder 已接入同一执行器；FP8 保留 520-byte/UE8M0，INT8 为 544-byte/FP32 scales。24 项新增 GPU 检查及既有 FP8 回归通过；模型质量、完整安装和延迟待验收，见 [GLM MLA 记录](kv_mla_latent_20261012.md) |
| R06 | MLA indexer：paged score、Tensor Core scorer、gather/cuBLAS、KPool 压缩及 quant/cache | 公共 G128 codec/view、CUDA writer、Triton scorer、gather/cuBLAS 和 KPool 已接 E4M3/INT8；122 项 SM70 GPU 组件测试通过。DeepSeek C4/C128 融合 compressor 另有 4 项 GPU 检查。独立 `AttentionConfig.indexer_cache_dtype`，`auto` 保留旧选择；原生 ABI 缺失在写入前拒绝。代表性引擎、质量与性能仍开放，不以组件测试关闭全部验收 |
| R07 | 通用 Triton MLA prefill/decode | 当前 backend 格式声明只有 FP16/BF16/FP8，没有普通 INT8；普通 Triton attention 的 INT8 支持不能代替 MLA 支持 |
| R08 | TurboQuant store/decode | 旋转、低比特编码与查询变换为独立数学表示；不是一个尚漏 INT8 绑定的 FP8 快路径。可共享资源和准入框架，不能用 INT8 名称抹掉算法区别 |

来源：[QSA backend][qsa]、[QSA kernels/planner][qsa-ops]、[host history][host]、
[device history][device-history]、[DeepSeek sparse][ds-sparse]、
[GLM sparse][glm-sparse]、[GLM codec][glm-codec]、[indexer][indexer]、
[KPool][kpool]、[Triton MLA][triton-mla]。

## 图、启动与缓存生命周期

| 编号 | 边界 | 审计结论与责任位置 |
| --- | --- | --- |
| L01 | CUDA Graph context buckets / batch-context / wave routing | 普通 byte codec 复用图分桶、wave、batch/context 计划及有效策略哈希；组件 replay 通过。模型图状态/容量/生命周期及延迟仍未验收。 |
| L02 | V1/V2 DFlash 隐式 draft KV 默认 | V1/V2 共用隐式 draft KV 解析函数；显式配置优先。四种 target codec 的默认/显式/非 SM70 CPU 检查通过；引擎边界仍待验证。 |
| L03 | cold/warm/AOT 与融合资格 | [runtime defaults][defaults] 的旧 NVFP4/TP4/E4M3 条件把 AOT reload 设为关闭，INT8 不继承；[模型侧 Norm/RoPE 接入][qwen-next] 仍有 TP/权重/模型资格。需要分离算子结构准入和已有质量资格，不能机械删除保护 |
| L04 | prefix 恢复、槽位复用、重排、sleep/offload、MTP commit/rollback、多引擎 | INT8 payload/scale 同物理分配，V1/V2 allocator 已有 view；CPU/offload 有整页 byte-copy 路线，但实际引擎恢复/隔离未验收。`cache_dtype`/编码身份、padding、块拆分都需覆盖 |
| L05 | KV transfer：相同页复制、异构 TP/块大小、permute | [NIXL worker][nixl] 按连续半页或 head chunk 建描述；planar 页的 payload/scale 区不能靠除以 TP 或 block ratio 推导。当前没有 plane-aware 描述；handshake 有 dtype 字符串及通用 cache layout，但未使用 KVCodec 编码版本/布局身份。普通整页搬运与异构变换分别验收，不能因同分配就宣称 connector 全支持 |
| L06 | 后端能力、缺二进制与通用兜底 | 已解除整个后端必须同时具备 converter/prefill 的耦合；各算法依实际二进制 codec 能力准入。decode-only 能力缺口用例通过；实际回退/原生/性能范围仍分别验收。 |

`prefill_prefix_kv_bridge` 已在路由表中声明，但 P05 的数值 gate 使 INT8 当前
不能选中它，这是“声明不等于接入”的具体例子。类似地，BFLA 的已声明 native
codec 不能证明 mask 生成完整，long binding 不能证明图 variant 已生成。

FlashQLA/GDN、TurboMind linear/MoE、投影融合和采样等不直接消费普通 paged KV。
本轮核对的这些选择器未发现按普通 `kv_cache_dtype` 为 INT8 新建执行链；GDN
自己的 `mamba_cache_dtype` 是独立状态契约。不能把权重 FP8 的优化名称当成 KV
FP8 依赖，也不能由此推导端到端性能不变。L02/L03 的启动策略和共享资源成本
仍可能影响整条加速链。

## 证据范围与后续顺序

本轮审计没有新增模型测速。以前阶段的 XQA/grouped 组件证据，以及当前阶段
55 个 public prefill/conversion GPU 用例，见工作记录。本轮核对到融合测试
`gpu-fused-r6.log` **10 passed**，覆盖两种融合、五种 codec 的写入兼容及图重放；
这不是模型侧融合实际命中证明。FA2 long/tail 和完整引擎测试仍单列待验证。

结构审计 `python -m tools.sm70.flash_v100_audit` 通过：import cycles=0，
model names outside spec=0，env reads outside config=0，forward=102 行。
这是该工具覆盖的组件边界，不能否定 L03 的既有模型适配资格。原始搜索与注册
清单保留在本机 `/tmp/kv-fastpath-{source-inventory.json,source-audit-r1.log,
global-format-gates.txt,native-gates-audit.txt,qsa-audit.txt}`。

下一步按可复用的责任组件推进：

1. 闭合 W02、L02/L06 等明确绑定/入口差异，完成已接 long/tail/engine 的 GPU 资格。
2. 用一个公共 row-scale 契约打通 P05，再将 D03..D07 与 L01 的分区/图计划公共化；
   不复制执行器，不要求保留不适合 INT8 成本的同一个调优数值。
3. 闭合 S03 q16 与 sparse/history/MLA/indexer 的表示级绑定，再验证 L04/L05 状态搬运。
4. 在真实算子命中和质量合格后，执行已冻结的 27B NVFP4/INT8 target/FP16 draft、
   DFlash2、TP4、prefix caching、262144 最大上下文的常见长度测试，分开报告
   prefill、验证轮和纯 decode；35B AWQ/FP8 同条件 decode 门槛继续保留。

不能以一条通用路径可运行、一次短请求或总路由命中来关闭上述缺口。

[decode]: ../../../vllm/v1/attention/backends/flash_v100/decode.py
[public-build]: ../../../flash-attention-v100/setup.py
[routing]: ../../../vllm/v1/attention/backends/flash_v100/routing.py
[interface]: ../../../flash-attention-v100/flash_attn_v100/flash_attn_interface.py
[native-decode]: ../../../flash-attention-v100/kernel/flash_decode_paged.cu
[traits]: ../../../flash-attention-v100/kernel/kv_codec_traits.cuh
[prefill-candidates]: ../../../vllm/v1/attention/backends/flash_v100/prefill_candidates.py
[prefill]: ../../../vllm/v1/attention/backends/flash_v100/prefill.py
[dense]: ../../../vllm/v1/attention/backends/flash_v100/dense_prefill.py
[native-prefill]: ../../../flash-attention-v100/kernel/fused_mha_forward_paged.cu
[verifier]: ../../../vllm/v1/attention/backends/flash_v100/spec/verifier.py
[grouped]: ../../../vllm/v1/attention/ops/sm70_grouped.py
[long]: ../../../vllm/v1/attention/ops/sm70_grouped_long.py
[tail]: ../../../vllm/v1/attention/ops/sm70_grouped_scalar.py
[tree]: ../../../vllm/v1/attention/backends/ddtree_branch_triton.py
[triton-backend]: ../../../vllm/v1/attention/backends/triton_attn.py
[cache-native]: ../../../csrc/libtorch_stable/cache_kernels.cu
[cache-dispatch]: ../../../csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh
[target-fused]: ../../../vllm/model_executor/layers/attention/sm70_qwen38_qk_rope.py
[draft-fused]: ../../../vllm/model_executor/layers/attention/sm70_dflash2_qk_rope.py
[layout]: ../../../vllm/v1/attention/backends/flash_v100/kv_layout.py
[qsa]: ../../../vllm/models/qwen4_exp/nvidia/qsa.py
[qsa-ops]: ../../../vllm/models/qwen4_exp/nvidia/ops/qsa.py
[host]: ../../../vllm/models/qwen4_exp/nvidia/ops/host_kv.py
[device-history]: ../../../vllm/models/qwen4_exp/nvidia/ops/device_kv_attention.py
[ds-sparse]: ../../../vllm/models/deepseek_v4/sm70/sparse.py
[glm-sparse]: ../../../vllm/models/glm5next/sm70/sparse.py
[glm-codec]: ../../../vllm/models/glm5next/sm70/fp8_kv.py
[indexer]: ../../../vllm/models/deepseek_v4/sm70/indexer.py
[kpool]: ../../../vllm/model_executor/layers/sparse_attn_indexer_kpool.py
[triton-mla]: ../../../vllm/v1/attention/backends/mla/triton_mla.py
[defaults]: ../../../vllm/model_executor/models/runtime_defaults.py
[qwen-next]: ../../../vllm/model_executor/models/qwen3_next.py
[nixl]: ../../../vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py
