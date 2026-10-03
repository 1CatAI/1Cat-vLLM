# GGUF on V100: implementation and acceptance control

## Scope and integration

Target: extensible standalone GGUF loading on four SM70 V100s, then recover
prefill and C1/C4/C8/C16 throughput against matching native quantization paths.
Integration is `onecat/main`; never change versioned source snapshots for this
campaign. Kernel admission and startup reporting must use KernelConfig and
`vllm/sm70_profiles/acceleration.py`. No new environment switches.

A model generation smoke is not a logits, quality, or throughput acceptance.
Every performance comparison must freeze checkpoint and quantization, GPU
UUIDs/topology, TP/EP, CUDA/Torch/build SHA, context/input/output length,
sampling, MTP, attention backend, graphs, and KV format. Separate prefill,
TTFT, steady decode, and emitted-token throughput. The existing 35B AWQ/FP8
migration gates remain mandatory and are not replaced by a GGUF smoke.

## Ordered review scopes

1. Standalone metadata/tokenizer and qwen35 dense adapter. Preserve GGUF
   vocabulary/token types/chat template; explicit HF config/tokenizer wins.
   Restore GDN negative exp(A_log), convolution dimension, norm offsets,
   tiled value heads, and correctly shard the tiled out projection. Reject
   unknown tensors, missing shards, duplicate tensors, and invalid split counts.
   Mixed FP16/quantized projection shards retain distinct storage and dispatch.
2. qwen35moe adapter: stacked expert types, mixed gate/up expert quantization,
   MTP nextn adapter; verify full logits and greedy output against llama.cpp.
3. qwen4exp adapter: HC, QSA/indexer, and PLE hash constants/table. Use llama.cpp
   conversion/qwen4exp.py as format oracle. Preserve 64-bit hash constants;
   PLE layer IDs differ between GGUF (zero-based) and HF (one-based).
4. llama-family, glm4moe, deepseek2 (including MLA kv_b split), gpt-oss and
   minimax-m2 adapters. Handle architecture-specific RoPE permutations.
5. Modern native kernel closure, with upstream licenses and pinned sources:
   Q1_0, Q2_0, MXFP4, NVFP4, TQ1/TQ2; numerical and route capability gates.
6. Q4_0/Q4_1/Q4_K/Q8_0 load-time repack and rounding report; TurboMind/QPN
   decode, batched GEMM, expert grouping and tensor-core prefill.
7. Remaining types: MMVQ decode, dequant + FP16 GEMM prefill; measured Volta
   fused dequant HMMA for intermediate M; grouped MoE token reuse.

## TP/EP decision

Quantization block alignment is an operator/storage constraint, independent
of model names. K-quant has block K=256: local intermediate K=4352 is aligned;
Flash-Next expert K=640 divided over TP4 gives K=160 and is not aligned.
Do not slice packed bytes at that boundary. Prefer EP with complete experts
when the installed GGUF expert method supports EP; otherwise convert the
unaligned expert tensor at load time to a format with an admitted K boundary.
EP is not considered implemented by setting enable_expert_parallel alone.
Coordinate the final decision with the Flash-Next decode work; its observed
GPU topology differs from the historical full-NVLink baseline.

## References and provenance

- vllm-project/vllm-gguf-plugin main
  `e2b8ad532b8b5ea175100202c30430c1d2b5e6a8` (Apache-2.0): adapter structure
  and GGUFHeadTilingLayout. Its current config parser still reads HF config;
  copying that parser does not implement standalone GGUF metadata parsing.
- Plugin PR 141, open at inspection, head
  `aa09d6522f29325d64d999d7d7c794f79836de07`: kernel integration pins llama.cpp
  `002a12ad25503a93501b2e188c360029830a241a`. Published throughput is external
  evidence, not acceptance data from this campaign.
- ggml-org/llama.cpp reference
  `bed0a856606ee4a24a164066f73d2379447033f5` (MIT). PR 27742 is merged;
  its original head is `eaf93765572e794b8e3754fe45adbe12d381e997`.

Retain the original SPDX headers in adapted plugin layout code. No llama.cpp
CUDA code is included by the first review scope.

## Current evidence / rejected paths

Initial integration base `3a147164833e58df2f80203cd169cbd71cea41e1`.
Owned branch `codex/v100-gguf-loader-20261002-230454`.

- Synthetic actual GGUF files: 17 native metadata/split-validation tests passed.
- Unsloth Qwen3.5-0.8B Q4_K_M revision
  `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`: native metadata recovered 24
  layers, hidden=1024, attention head_dim=256, SSM 16 key/16 value heads,
  inner=2048, and interleaved MRoPE [11,11,10], without HF config.
- Six tokenization/decode samples matched HF token IDs, including Chinese,
  combining accents, multilingual scripts, digits, emoji and tool tags.
  GGUF chat template is authoritative; it differs from current HF template.
- Rejected tokenizer shortcut: treating qwen35 pretokenization as qwen2
  mismatches combining marks. qwen35 requires NFC and the mark-aware regex.
- First TP4 initialization attempt stopped before model loading because the
  new environment lacked torchvision. This is not GPU route/quality evidence.
- CPU llama.cpp oracle built from the pinned reference; raw first-token logits
  and greedy token sequences are retained with the task artifacts.

No 27B/35B performance acceptance or full GGUF support is claimed yet.

## Runtime and artifact hygiene

Models and reference builds are under `/mnt/nvme3/ymzx-data`; the volume is
mounted by its existing fstab UUID. GPU work must hold `/tmp/gpu0-3.lock` for
its entire process lifetime and inspect existing GPU processes first. Use
owned compiler caches. Do not use private kernel DSOs/preloads as PR evidence.

Large raw output is retained outside Git, in the owned worktree's `.artifacts`.

### Repack admission finding: Q4_K affine coefficients

27B UD-Q4_K_M: 212,992 sampled group coefficients, 6,815,744 weights,
stratified over all Q4_K tensors (up to 256 superblocks per tensor). Keeping
codes identical and rounding scale and additive min/bias to FP16 gives maximum
absolute weight error 9.9182e-5, RMSE 9.5210e-6 and relative L2 6.7039e-4.
68.2262% of reconstructed FP16 weights differ from direct GGUF dequantization.
This is a CPU coefficient-rounding study, not GPU/model performance or a
quality gate. Q4_K repack is therefore approximate; require full logits and
quality evidence before making it the default. Avoid representing additive
bias as a rounded zero-point ratio: the existing prepare function computes
-zero*scale, which adds another rounding step and fails for zero scale.
A direct affine scale+bias preparation interface is needed for Q4_1/Q4_K.
+### Primary acceptance checkpoint: Flash-Next GSQ-RCO IQ3_XXS

The requested primary checkpoint is
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://www.modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/files),
IQ3_XXS (not IQ3_S). GSQ-RCO here assigns precision per tensor; it does not
introduce an additional weight rotation.

The first shard is 47,039,860,096 bytes, SHA256
`219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`.
The second is 28,800,138,432 bytes, SHA256
`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.
ModelScope file revisions are respectively
`e3a513b7830d32c0d87b07d54ca650cbc422cc26` and
`26dfa70d5c61d623f472c234462b8205ef3448e2`. Downloads use these immutable
revisions, resume into .part files, and require size/SHA256 validation before
rename.

The real first-shard directory has 1,223 tensors and GGML type counts:
F32=292, F16=1, BF16=484, Q4_K=45, Q5_K=15, Q6_K=71,
IQ2_XXS=18, IQ2_XS=20, IQ2_S=20, IQ3_XXS=12, IQ3_S=82,
IQ4_NL=58, IQ4_XS=67, Q2_0=38. The remaining tensor is the IQ4_NL
PLE table (320,001,536 rows of dimension 160), requiring packed sparse lookup.
GPU vocabulary partitions or host mmap must retain the quantized rows; never
materialize the complete table in FP16.

Actual metadata: qwen4exp, 48 layers, H=2560, 24 Q / 2 KV attention heads,
head_dim=256, 512 experts, top-k=10, expert K=640, 16 GDN key / 48 value
heads, 128-dimensional GDN heads, HC count=4/rank=320. PLE keys include
exact int64 multipliers, offsets and prime vocabulary sizes. GGUF PLE layer
index 1 maps to HF layer ID 2. Q2_0 uses K blocks of 64, so expert down
local K=160 still cannot be byte-sliced under TP4; IQ4_NL K blocks of 32
are aligned. The user requires Flash-Next TP as well as attention TP.
For its 30 Q2_0 routed down projections, loading now converts each expert
from 64-value Q2_0 to two 32-value Q4_1 blocks before TP4 slicing: reuse
scale d, integer code 0-3, and additive min=-d. No values are requantized.
The Q2_0 source bytes total 7,077,888,000; converted bytes total
15,728,640,000. TP4 storage increases by 2,162,688,000 bytes (2.014 GiB)
per rank, avoiding a whole-expert FP16 expansion or an EP requirement.

The native expert loader retains separate gate/up/down formats and validates
every TP block boundary before allocating storage. Three CPU tests cover
exact decoded equality including signed/subnormal/maximal finite scales,
all four K=160 boundaries, independent projection storage and rejection of
unconverted Q2_0. Three GPU FFN tests at M=1/8/64 compare the sum of four
TP partials against the full expert reference. The real two-shard directory
contains 1,224 tensors and all have adapter names. These are local contracts;
the primary-model forward and quality gates are still pending.

### Loading validation and open gates

17 metadata/tokenizer/split tests and 13 dense adapter/TP/mixed-storage
tests pass on CPU. Earlier failures localized missing runtime
dependencies, unaligned K-quant FFN storage, missing hybrid state traits,
and KV-head replication when TP exceeds KV heads. Those have source fixes;
alignment and replication have focused CPU coverage. A later run completed
engine initialization and GPU prefill warmup, then rejected the text-only
class's missing MRoPE position interface. That interface is now implemented
and tested on CPU. Inverse A_log loading preserves its FP32 parameter contract
even with FP16 model dtype.

The corrected TP4 tiny Qwen3.5-0.8B Q4_K_M run completed GPU inference on
2026-10-03 with FP16, eager execution, max length 2048, no MTP, FP16 KV,
Flash-V100 attention and FlashQLA GDN. Three raw prompts generated 64 greedy
tokens each. The English and arithmetic sequences matched llama.cpp for all
64 tokens; the Chinese sequence matched its first 13 tokens and then diverged.
First-token full-vocabulary logits have RMSE 0.1243/0.1408/0.1656, maximum
absolute error 0.6073/0.8044/0.8116 and relative L2 0.02949/0.04647/0.06349.
All three top-1 tokens match. The reference is CPU llama.cpp at
`bed0a856606ee4a24a164066f73d2379447033f5`, using identical token IDs.
These numeric differences and the Chinese divergence remain an open quality
gate; a completed inference smoke does not establish adapter correctness.
The following chat harness failed because Transformers 5 returns a dict
from apply_chat_template by default; it now requests return_dict=False.

These are implementation-localization results, not primary-model quality
or speed acceptance. Hold the shared GPU flock and do not terminate other
tasks. No accepted whole-model throughput is recorded.
