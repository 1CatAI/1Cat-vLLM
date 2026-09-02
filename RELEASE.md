# 1Cat-vLLM 1.5.0

1Cat-vLLM 1.5.0 is the largest SM70/V100 release since the project moved to
the current vLLM base. It turns the Qwen3.8 DFlash2 work into a practical
serving path, substantially improves long-context decode and prefill, adds
modern hybrid-model and NVFP4 support, and closes the wheel/API reliability
gaps required for normal deployment.

This release supersedes 1.3.0. The audited release range contains 126 merged
pull requests and 400 commits after `v1.3.0`.

> The performance figures below are measured results under the recorded model,
> GPU count, topology, context, sampling, KV dtype, and speculative-decoding
> contract. They are not interchangeable universal throughput claims.

## Highlights

### Production DFlash2 on V100

1.5.0 backports the official MRV2 DFlash dependency closure and DFlash2 model
while preserving the existing Eagle, MTP, and DDTree implementations
([#252](https://github.com/1CatAI/1Cat-vLLM/pull/252),
[#253](https://github.com/1CatAI/1Cat-vLLM/pull/253)). The SM70 path adds:

- independent target and draft KV dtypes;
- checkpoint-owned seven-token drafts and selector top-K 16;
- Qwen3.8 hidden-state capture at the correct five layer boundaries;
- non-causal sliding-window draft attention through Flash-V100;
- target and draft CUDA Graphs;
- exact probabilistic rejection sampling;
- TP4-sharded context projection and compact LM-head selection;
- exact grouped q8 verification over FP8 E5M2 KV;
- prefix-cache and Mamba-align state recovery;
- automatic, capability-based fast-path selection instead of one global
  all-or-nothing gate.

An omitted per-layer RMSNorm dependency initially reduced acceptance. After
repair, the fixed 64-prompt GSM8K contract reaches request-mean acceptance
length **5.3888** and pooled acceptance length **4.5644**, close to the public
official request mean of 5.46. This confirms that target FP8 quantization was
not the cause of the earlier acceptance gap.

The practical release path now measures approximately **17.38 ms per complete
DFlash2 round** without requiring the whole server to be configured as
`max_num_seqs=1`. Each verifier operator checks its own dtype, TP shape, KV
format, and live batch shape, then falls back independently when its local
contract is not met
([#420](https://github.com/1CatAI/1Cat-vLLM/pull/420),
[#436](https://github.com/1CatAI/1Cat-vLLM/pull/436)).

Representative full-model results on four V100 GPUs are:

| Workload | Result | Contract / status |
| --- | ---: | --- |
| Historical web prompt, 512 output tokens | **206.06 tok/s** | 17.463 ms/round, 3.599 emitted/round |
| High-acceptance MBPP request | **251.60 tok/s** | AL 4.686, natural EOS, EvalPlus Base/Plus 1/1 |
| Repeated-context adaptive lookup q16 | **316.27 tok/s** | Opt-in lookup-hit workload, not an ordinary-request default |
| 32K cold prefill | **4,039-4,069 tok/s** | QPN2-packed prefill, E5M2 target KV |
| 64K pure prefill | **3,567 tok/s** | Same DFlash2 serving stack |

The lookup-augmented route keeps the checkpoint-native seven DFlash2 drafts
for ordinary requests and expands target verification only after sustained
context-copy hits. Normal, low-confidence, and structured-output requests stay
on q8. The q16 result is therefore a specialized opt-in capability rather than
the expected speed of every request
([#355](https://github.com/1CatAI/1Cat-vLLM/pull/355),
[#366](https://github.com/1CatAI/1Cat-vLLM/pull/366)).

The recommended fully quantized target checkpoint for 1.5.0 is
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`. Its fresh-wheel TP4 release gate reaches
**17.645 ms per warmed complete round**, **4,038.5 tok/s at 32K prefill**, and
**3,596.5 tok/s at 64K prefill**. This checkpoint has a distinct all-NVFP4
layout, so these results are reported separately from the earlier mixed-NVFP4
206 tok/s workload above
([#445](https://github.com/1CatAI/1Cat-vLLM/pull/445)).

### Qwen3.8 target-only, MTP, and long-context performance

The target model was optimized independently of speculative decoding. This is
important because a faster drafter cannot compensate for an inefficient
target verifier.

| Model / route | Workload | Measured result | Evidence |
| --- | --- | ---: | --- |
| Qwen3.8-27B-NVFP4, no spec | exact 128K decode | **61.834 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) |
| Qwen3.8-27B-NVFP4, no spec | exact 256K decode | **50.376 tok/s** | [#285](https://github.com/1CatAI/1Cat-vLLM/pull/285) |
| Qwen3.8 Flash-Next-NVFP4, no MTP | 8K/512 pure decode | **80.732 tok/s** | [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415) |
| Qwen3.8 Flash-Next-NVFP4 + MTP4 | final cold-JIT gate | **138.26 tok/s** | [#389](https://github.com/1CatAI/1Cat-vLLM/pull/389) |
| Qwen3.6-35B-A3B NVFP4, no MTP | 4096/1024 decode | **116.99 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) |
| Qwen3.6-35B-A3B NVFP4 + MTP4 | matched MTP4 decode | **174.76 tok/s** | [#270](https://github.com/1CatAI/1Cat-vLLM/pull/270) |
| DeepSeek-V4-Flash PP2 x TP4 | strict quality control | **73.539 tok/s** | [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344) |

For Qwen3.8 Flash-Next, grouped Page4 QSA, indexed NVFP4 MoE input, fused
SwiGLU, and grouped-GEMM prefill raise pure-prefill throughput to:

| Prompt length | Pure prefill |
| ---: | ---: |
| 8K | **7,056.26 tok/s** |
| 32K | **6,603.30 tok/s** |
| 64K | **6,327.43 tok/s** |
| 131K | **5,927.33 tok/s** |
| 256K | **5,248.63 tok/s** |

The key steps are documented in
[#378](https://github.com/1CatAI/1Cat-vLLM/pull/378),
[#387](https://github.com/1CatAI/1Cat-vLLM/pull/387),
[#390](https://github.com/1CatAI/1Cat-vLLM/pull/390), and
[#393](https://github.com/1CatAI/1Cat-vLLM/pull/393). These are Flash-Next
target-prefill results; they must not be relabeled as DFlash2 prefill results.

### Flash-V100 long-context improvements

1.5.0 continues the Volta-native attention work from 1.3.0:

- batched FP8 E5M2 XQA reuses page IDs and widens paired cache loads to 128
  bits, improving B16/16K full-model pure decode by **7.92%** while preserving
  bitwise operator output
  ([#268](https://github.com/1CatAI/1Cat-vLLM/pull/268));
- long E4M3 XQA routing improves Qwen3.8 NVFP4 decode by **52.45% at 128K**
  and **83.48% at 256K**
  ([#285](https://github.com/1CatAI/1Cat-vLLM/pull/285));
- the default long-prefill GQA architecture packs six heads into wider
  Tensor-Core work and reaches approximately **60.8 useful causal TFLOP/s**,
  versus 46.6-47.1 TFLOP/s in the 1.3.0 path
  ([#286](https://github.com/1CatAI/1Cat-vLLM/pull/286));
- DFlash2 draft sliding-window attention skips unused prefix tiles, removing
  its post-32K context slope
  ([#328](https://github.com/1CatAI/1Cat-vLLM/pull/328));
- the grouped q8 verifier reduces operator latency by about 20-21% from 32K
  through 256K
  ([#284](https://github.com/1CatAI/1Cat-vLLM/pull/284)).

### Modern model and quantization support

- **ModelOpt NVFP4 on exact SM70.** ModelOpt W4A16/W4A4 weights now use proven
  TurboMind, Marlin, or explicit-emulation backends and fail closed otherwise.
  Scale provenance is preserved without magnitude heuristics
  ([#228](https://github.com/1CatAI/1Cat-vLLM/pull/228)).
- **QUASAR Qwen3.8-27B full NVFP4.** Adds the fully quantized
  `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` target to the DFlash2 production path.
  The SM70 implementation separates the checkpoint's logical TP-local GDN
  output width of 4,120 from its zero-padded physical execution width of 4,128,
  enables quality-audited QPN2 execution for compatible projections and packed
  prefill, and gives MRV2 its own K+1 draft buffers and slot mappings
  ([#445](https://github.com/1CatAI/1Cat-vLLM/pull/445)).
- **Qwen3.6-35B-A3B mixed FP8/NVFP4 MoE.** Adds graph-safe grouped NVFP4 MoE,
  compact active-expert paths, duplicate-slot correctness, and cold-start MTP
  warmup. The retained GSM8K result is 122/128 with zero invalid or repetitive
  outputs ([#270](https://github.com/1CatAI/1Cat-vLLM/pull/270)).
- **Qwen3.8 Flash-Next-NVFP4.** Adds PLE offload, QSA sparse attention,
  NVFP4 MoE, MTP4, and quality-audited target-only decode
  ([#338](https://github.com/1CatAI/1Cat-vLLM/pull/338),
  [#345](https://github.com/1CatAI/1Cat-vLLM/pull/345),
  [#389](https://github.com/1CatAI/1Cat-vLLM/pull/389),
  [#415](https://github.com/1CatAI/1Cat-vLLM/pull/415)).
- **GLM-5.3-Flash-NVFP4.** Introduces the TP4/PP2 SM70 model path with packed
  E4M3 KV, NVFP4 MoE, KPool/indexer, sparse MLA, mHC, and KDA support. Treat
  this as a newer support lane and retain the documented topology/quality
  gates ([#341](https://github.com/1CatAI/1Cat-vLLM/pull/341)).
- **DeepSeek-V4-Flash.** Corrects YaRN, indexer layout, sparse/hybrid cache
  behavior, PP2 x TP4 memory lifetime, and quality-gates aggressive FP8/FP13
  routes instead of enabling them only from operator speed
  ([#250](https://github.com/1CatAI/1Cat-vLLM/pull/250),
  [#251](https://github.com/1CatAI/1Cat-vLLM/pull/251),
  [#283](https://github.com/1CatAI/1Cat-vLLM/pull/283),
  [#344](https://github.com/1CatAI/1Cat-vLLM/pull/344)).
- **Multimodal capacity and safety.** Adds explicit DeepSeek-OCR resolution
  modes, generic UVA multimodal-tower offload, and bounded Gemma4 tower
  attention memory
  ([#293](https://github.com/1CatAI/1Cat-vLLM/pull/293),
  [#297](https://github.com/1CatAI/1Cat-vLLM/pull/297),
  [#310](https://github.com/1CatAI/1Cat-vLLM/pull/310)).
- **Quark W4A16.** Repairs INT4/UINT4 checkpoint support
  ([#401](https://github.com/1CatAI/1Cat-vLLM/pull/401)).

## Quality and serving validation

### Coding and distribution gates

The practical Qwen3.8-27B-NVFP4 + DFlash2 quality campaign uses official
`temperature=1.0`, `top_p=0.95`, `top_k=20`, xhigh reasoning, natural EOS,
and a 16K output cap:

| Dataset | Requests / scored | Primary score | Additional score | Note |
| --- | ---: | ---: | ---: | --- |
| MBPP / EvalPlus | 96 / 93 | Base **89/93** | Plus **80/93** | 95/96 natural stops |
| HumanEval / EvalPlus | 96 / 96 | Base **94/96** | Plus **92/96** | 91/96 natural stops |
| LiveCodeBench | 48 / 48 | **33/48** | - | Three seeds, exactly 11/16 each |

On the first matched seed, target-only and DFlash2 both score Base 62/63 and
Plus 59/63 across MBPP and HumanEval. Eight fixed WikiText segments produce:

```text
Target-only PPL : 5.4993116
DFlash2 PPL     : 5.4993622
Absolute delta  : 0.0000506
```

The quality audit and exact sampling contract are retained in
[#346](https://github.com/1CatAI/1Cat-vLLM/pull/346).

The recommended all-NVFP4 QUASAR checkpoint has an additional matched gate.
At seed `20260925`, target-only and DFlash2 both score MBPP 32/32, EvalPlus
Base 31/31, and Plus 28/31. Across two predeclared seeds there is only one
paired discordance per score class, while the maximum matched WikiText PPL
difference is 0.00851. Structured-output B1/B2/B4, a 96-request mixed B4
stress, tool-chain, nested/escaped JSON, stream parity, and repeated-prefix
state tests also pass
([#445](https://github.com/1CatAI/1Cat-vLLM/pull/445)).

### Tool calling, structured output, and prefix state

The DFlash2 serving path was tested beyond plain chat:

| Gate | Result |
| --- | ---: |
| Structured API B1/B4 | **24/24** |
| Long alternating-prefix state | **5/5** |
| BFCL | **29/32** |
| ToolACE | **12/12** |
| NexusRaven | **13/16** |
| Strict JSON Schema | **7/8** |

The accelerated results match their retained target-only/q7 references.
Parser and state fixes cover reasoning, streaming tool calls, JSON Schema,
prefix hits, and lookup q15 burst boundaries. The release also normalizes
malformed historical tool-call arguments instead of crashing a conversation
([#346](https://github.com/1CatAI/1Cat-vLLM/pull/346),
[#366](https://github.com/1CatAI/1Cat-vLLM/pull/366),
[#432](https://github.com/1CatAI/1Cat-vLLM/pull/432)).

## Reliability and correctness fixes

### Hybrid KV, prefix cache, and CPU offload

- Native CPU KV offload now supports Mamba/GDN aligned external hits, preserves
  MTP accepted-token ownership, and can opt into VMM-safe transfers with
  expandable segments
  ([#247](https://github.com/1CatAI/1Cat-vLLM/pull/247),
  [#248](https://github.com/1CatAI/1Cat-vLLM/pull/248),
  [#249](https://github.com/1CatAI/1Cat-vLLM/pull/249)).
- MRV2 zeroes newly reused KV blocks before attention, including interleaved
  multi-pool layouts
  ([#276](https://github.com/1CatAI/1Cat-vLLM/pull/276),
  [#280](https://github.com/1CatAI/1Cat-vLLM/pull/280)).
- Prefix-cache eviction tracks in-flight boundaries instead of invalidating
  active work ([#296](https://github.com/1CatAI/1Cat-vLLM/pull/296)).
- Compressed attention storage and QSA block tables consistently use physical
  page units ([#373](https://github.com/1CatAI/1Cat-vLLM/pull/373),
  [#381](https://github.com/1CatAI/1Cat-vLLM/pull/381)).

### Runtime and API stability

- Multimodal cache eviction no longer hangs on stale LRU-order keys
  ([#261](https://github.com/1CatAI/1Cat-vLLM/pull/261)).
- Custom all-reduce CUDA Graph capture temporarily disables incompatible VMM
  allocations and restores the complete allocator configuration afterward
  ([#262](https://github.com/1CatAI/1Cat-vLLM/pull/262)).
- Safetensors index loading honors tensor-to-file assignments instead of
  accepting unrelated shards
  ([#407](https://github.com/1CatAI/1Cat-vLLM/pull/407)).
- KV zeroing avoids H2D staging races on first use
  ([#406](https://github.com/1CatAI/1Cat-vLLM/pull/406)).
- DFlash q16 checks the native grouped-verifier capability before dispatch;
  older q8-only extensions fall back safely instead of terminating EngineCore
  ([#422](https://github.com/1CatAI/1Cat-vLLM/pull/422)).
- API failures drain sibling multimodal work, and in-process shutdown releases
  target/draft models, KV/state tensors, CUDA Graphs, and SM70 global
  workspaces ([#432](https://github.com/1CatAI/1Cat-vLLM/pull/432)).

## Packaging and installation

The recommended 1.5.0 build target is:

```text
Python 3.12
CUDA 12.8
PyTorch 2.10
NVIDIA SM70 / Tesla V100
```

The SM70 wheel contains the vLLM CUDA extensions, FlashAttention-V100,
paged-KV utilities, TurboMind SM70 kernels, the compact sampler, and the
precompiled FlashQLA extension. A supported wheel install does not require the
project source tree or runtime NVCC compilation.

Wheel assembly now:

- fails closed when `patchelf` cannot remove build-host RPATHs;
- emits the compact sampler with the CPython stable ABI;
- resets only the pinned vendored FlashAttention checkout before applying
  patches, making incremental builds repeatable;
- pins a compatible FastAPI metrics stack;
- validates all native-library imports and grouped-verifier capabilities in an
  isolated environment.

The final 1.5.0 release-candidate audit contained ten native libraries with no
RPATH/RUNPATH and passed `/v1/models`, `/metrics`, plain chat, streaming and
non-streaming tool calls, JSON Schema, and repeated-prefix requests. A repeated
10,017-token prefix moved from **2.642 s cold to 0.164 s cached**
([#319](https://github.com/1CatAI/1Cat-vLLM/pull/319),
[#320](https://github.com/1CatAI/1Cat-vLLM/pull/320),
[#421](https://github.com/1CatAI/1Cat-vLLM/pull/421),
[#427](https://github.com/1CatAI/1Cat-vLLM/pull/427)).

Install the published wheel with:

```bash
pip install ./1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl
```

### Recommended QUASAR NVFP4 + DFlash2 serving example

Download the recommended fully quantized target checkpoint from ModelScope:

```bash
modelscope download \
  --model QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --local_dir /data/models/Qwen3.8-27B-QUASAR-NVFP4
```

The following is the tested four-V100 production profile:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /data/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --served-model-name qwen3.8-27b-dflash2 \
  --trust-remote-code \
  --dtype half \
  --tensor-parallel-size 4 \
  --attention-backend FLASH_ATTN_V100 \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","kv_cache_dtype":"auto","draft_sample_method":"probabilistic"}' \
  --host 0.0.0.0 \
  --port 8000
```

The official draft checkpoint declares block size 8 and selector top-K 16, so
the runtime resolves seven speculative tokens automatically when the width is
omitted. TP4, E5M2 target KV, a 4,096-token scheduler budget, and four resident
sequences are the release-validation profile, not global DFlash2 requirements.
TP degree, KV dtype, scheduler capacity, and prefill chunk size do not globally
disable DFlash2; individual optimized operators still retain their own exact
local admission contracts.

## Upgrade notes and known limitations

- DFlash2 performance depends strongly on acceptance length, context,
  sampling, prefix reuse, and q8/q16 verification width. The 206, 251, and 316
  tok/s results above are different workloads.
- Adaptive lookup q16 and drafter-free chaining remain specialized opt-in
  paths. Ordinary traffic remains on the checkpoint-native q7/q8 contract.
- SM70 Flash-V100 disables saved AOT graph-cache reload by default because a
  cached artifact can cause deterministic token drift. This protects output
  quality but increases cold-start compilation time; warm the service before
  production traffic.
- Reasoning tokens and final content share the request output budget. Short
  JSON/tool tasks should either allocate enough tokens or disable thinking for
  that request.
- Several aggressive QPN8, FP13, and prescaled routes remain default-off after
  model-level quality gates rejected small speed wins. Do not enable research
  toggles solely from an operator microbenchmark.
- GLM-5.3-Flash and other newer hybrid-model lanes have narrower validated
  topology and checkpoint contracts than the mature Qwen3.6/Qwen3.8 paths.
- Source/editable builds may invoke explicit FlashQLA JIT fallback and require
  a compiler/toolkit matching the PyTorch CUDA ABI. The published SM70 wheel
  uses the precompiled extension.

## Selected pull requests

### DFlash and speculative decoding

[#252](https://github.com/1CatAI/1Cat-vLLM/pull/252),
[#253](https://github.com/1CatAI/1Cat-vLLM/pull/253),
[#256](https://github.com/1CatAI/1Cat-vLLM/pull/256),
[#257](https://github.com/1CatAI/1Cat-vLLM/pull/257),
[#284](https://github.com/1CatAI/1Cat-vLLM/pull/284),
[#288](https://github.com/1CatAI/1Cat-vLLM/pull/288),
[#328](https://github.com/1CatAI/1Cat-vLLM/pull/328),
[#346](https://github.com/1CatAI/1Cat-vLLM/pull/346),
[#355](https://github.com/1CatAI/1Cat-vLLM/pull/355),
[#366](https://github.com/1CatAI/1Cat-vLLM/pull/366),
[#417](https://github.com/1CatAI/1Cat-vLLM/pull/417),
[#420](https://github.com/1CatAI/1Cat-vLLM/pull/420),
[#422](https://github.com/1CatAI/1Cat-vLLM/pull/422),
[#426](https://github.com/1CatAI/1Cat-vLLM/pull/426),
[#436](https://github.com/1CatAI/1Cat-vLLM/pull/436), and
[#445](https://github.com/1CatAI/1Cat-vLLM/pull/445).

### Attention, quantization, and model performance

[#228](https://github.com/1CatAI/1Cat-vLLM/pull/228),
[#268](https://github.com/1CatAI/1Cat-vLLM/pull/268),
[#270](https://github.com/1CatAI/1Cat-vLLM/pull/270),
[#275](https://github.com/1CatAI/1Cat-vLLM/pull/275),
[#281](https://github.com/1CatAI/1Cat-vLLM/pull/281),
[#285](https://github.com/1CatAI/1Cat-vLLM/pull/285),
[#286](https://github.com/1CatAI/1Cat-vLLM/pull/286),
[#315](https://github.com/1CatAI/1Cat-vLLM/pull/315),
[#338](https://github.com/1CatAI/1Cat-vLLM/pull/338),
[#341](https://github.com/1CatAI/1Cat-vLLM/pull/341),
[#344](https://github.com/1CatAI/1Cat-vLLM/pull/344),
[#378](https://github.com/1CatAI/1Cat-vLLM/pull/378),
[#387](https://github.com/1CatAI/1Cat-vLLM/pull/387),
[#389](https://github.com/1CatAI/1Cat-vLLM/pull/389),
[#390](https://github.com/1CatAI/1Cat-vLLM/pull/390),
[#393](https://github.com/1CatAI/1Cat-vLLM/pull/393), and
[#415](https://github.com/1CatAI/1Cat-vLLM/pull/415).

### Serving, correctness, and packaging

[#247](https://github.com/1CatAI/1Cat-vLLM/pull/247),
[#248](https://github.com/1CatAI/1Cat-vLLM/pull/248),
[#249](https://github.com/1CatAI/1Cat-vLLM/pull/249),
[#261](https://github.com/1CatAI/1Cat-vLLM/pull/261),
[#262](https://github.com/1CatAI/1Cat-vLLM/pull/262),
[#276](https://github.com/1CatAI/1Cat-vLLM/pull/276),
[#280](https://github.com/1CatAI/1Cat-vLLM/pull/280),
[#296](https://github.com/1CatAI/1Cat-vLLM/pull/296),
[#297](https://github.com/1CatAI/1Cat-vLLM/pull/297),
[#319](https://github.com/1CatAI/1Cat-vLLM/pull/319),
[#320](https://github.com/1CatAI/1Cat-vLLM/pull/320),
[#373](https://github.com/1CatAI/1Cat-vLLM/pull/373),
[#381](https://github.com/1CatAI/1Cat-vLLM/pull/381),
[#406](https://github.com/1CatAI/1Cat-vLLM/pull/406),
[#407](https://github.com/1CatAI/1Cat-vLLM/pull/407),
[#421](https://github.com/1CatAI/1Cat-vLLM/pull/421),
[#427](https://github.com/1CatAI/1Cat-vLLM/pull/427), and
[#432](https://github.com/1CatAI/1Cat-vLLM/pull/432).

Full comparison:
[v1.3.0...v1.5.0](https://github.com/1CatAI/1Cat-vLLM/compare/v1.3.0...v1.5.0).

## Contributors

Thank you to everyone who contributed code, testing, hardware time, bug
reports, and review during this release cycle, including
[@yangzhuxinyzx](https://github.com/yangzhuxinyzx),
[@Leonccaa](https://github.com/Leonccaa),
[@andrewleech](https://github.com/andrewleech),
[@dg1kjd](https://github.com/dg1kjd),
[@lwh9346](https://github.com/lwh9346),
[@113636xfh](https://github.com/113636xfh),
[@kkobold](https://github.com/kkobold),
[@rivetphilbot](https://github.com/rivetphilbot), and
[@wfhe](https://github.com/wfhe), as well as upstream vLLM and the broader
open-source inference community.
