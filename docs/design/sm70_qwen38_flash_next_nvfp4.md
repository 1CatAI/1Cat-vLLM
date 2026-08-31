# Qwen3.8 Flash Next NVFP4 on SM70

## Status and ownership

- Status: source bring-up, pinned-host PLE, native Qwen4Exp MTP, and
  prefix-cache configuration are implemented; focused CPU/configuration gates
  pass and the local ModelScope snapshot is fully verified. The historical
  no-MTP TP4 online-QPN8 candidate reaches 82.274 steady decode tokens/s on the
  8192x512 contract, but online QPN8 requantizes dense checkpoint weights and
  is now experimental opt-in. Its matched 96-request GSM8K screen keeps 96/96
  answers correct in both arms but changes 88/96 token sequences and contains
  a large long-output/repetition outlier. The exact, byte-preserving NVFP4
  ten-route QPN-M1 MoE remains default-on; online QPN8 is default-off. The
  precision-preserving no-MTP control is about 67.6 tokens/s, so reaching 80
  tokens/s without online QPN8 remains an open optimization target. Corrected
  native MTP4 is stable across an 11-request TP4 transition run, and the
  no-thinking HumanEval8 gate reaches
  4.747 mean accepted length, 93.676% draft acceptance, 126.81 warmed pure
  decode tokens/s, and 8/8 standard semantic executions. The V2 M16 plus
  M5/M1 warmup also removes the first-request MTP `fused_moe_kernel` JIT.
- Integration line: public `main`; Qwen3.8 bring-up
  [#345](https://github.com/1CatAI/1Cat-vLLM/pull/345) and initial decode work
  [#361](https://github.com/1CatAI/1Cat-vLLM/pull/361) are merged. The compact
  MTP loader, exact SM70 MTP tile, and V2 warmup are the current follow-up.
- Current correctness-audit base SHA:
  `62ad1e02693f4c857f3b7547cef1860ee54e8053`.
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Model download: `/data/models/RadixArk/Qwen3.8-Flash-Next-NVFP4`.
- Download source: ModelScope `master`, verified against the fixed Hugging Face
  revision above: all 419 file sizes match and all 208 comparable LFS SHA-256
  values match. After download, all 419 local files were independently hashed
  against the ModelScope manifest with zero missing, size-mismatched, or
  SHA-256-mismatched files.
- Upstream references:
  [vLLM PR 53896](https://github.com/vllm-project/vllm/pull/53896) and
  [SGLang PR 36497](https://github.com/sgl-project/sglang/pull/36497).

The upstream PRs are implementation references, not acceptance evidence. Both
were still open when this work started, so imports must be narrowed to the
Qwen4Exp route and validated against this branch.

## Frozen first-pass contract

The first correctness route deliberately excludes speculative decoding.

- Hardware: four NVIDIA V100-SXM2-32GB GPUs (SM70).
- Parallelism: TP4, PP1, no expert parallelism.
- Model mode: `--language-model-only`. The initial SM70 route deliberately
  excludes the vision tower from its memory, quality, and performance gates;
  omitting the flag fails during configuration instead of reaching a private
  Qwen3.5 multimodal API mismatch at model construction.
- Compute dtype: FP16; no BF16 or native FP8/NVFP4 tensor-core assumptions.
- Checkpoint: ModelOpt NVFP4 routed-expert weights, consumed as an SM70
  weight-only W4A16 route. Ignored dense, attention, GDN, shared-expert, GR,
  PLE, and LM-head modules stay in their checkpoint dtypes and execute in
  FP16 where required.
- PLE/N-gram table: allocate and load each TP shard directly in pinned host
  memory. It must never be materialized on a GPU before being moved to the
  host. Gathered rows are transferred asynchronously and converted to FP16.
- Initial KV cache: FP16. FP8 KV cache is a separate, quality-gated follow-up.
- Initial decoding: MTP disabled. MTP may be enabled only after the no-MTP
  route is correct and its emitted-token baseline is recorded.
- Model runner: V2 is the default initial route. Its Qwen4Exp model state keeps
  raw token IDs and builds the PLE context from committed tokens, so rejected
  speculative candidates cannot leak into the next trigram. V1 remains a
  correctness control, not the primary performance route.
- Prefix caching: enabled for the MTP validation route. Hybrid recurrent state
  uses `mamba_cache_mode=align` with chunked prefill. The fixed QSA compressor
  ring is explicitly non-cacheable and is excluded from prefix-hit
  reconciliation; a clean ring block is allocated after a hit. Main QSA KV,
  compressed QSA KV, and aligned GDN/Mamba state remain cacheable.

The native-MTP validation route keeps the same TP4/PP1, FP16 activation,
ModelOpt NVFP4, language-model-only, and FlashAttention-V100 contract. It uses
four speculative tokens, V2, CUDA graphs, prefix caching, a deterministic
greedy prompt, and two identical requests. The second request is both the hot
speed sample and the prefix-cache reuse check. A matched no-MTP run is still
required for exact token-ID quality comparison and incremental verifier cost.

## Architecture facts that affect the port

The text stack has 48 layers: 36 gated-delta-net layers and 12 QSA layers in a
3:1 pattern. Hidden size is 2560. QSA uses 24 query heads, two KV heads,
head-dimension 256, index dimension 128, compression ratio four, and a 2048
token sparse budget. The MoE has 512 routed experts, top-10 routing, a 640-wide
routed expert, and one 640-wide shared expert. General residual connections
use four streams and rank 320.

PLE is a learned trigram embedding, not prompt-ngram speculative decoding. It
uses 16 heads (`ngram_size=3`, eight heads per n-gram order), embedding width
2560, and FP8 E4M3 storage. Native speculative decoding is the separate
one-layer MTP head.

## Memory budget hypothesis

Safetensor payloads total about 125.910 GiB. The sharded PLE payload is about
47.684 GiB, or about 11.921 GiB of pinned host memory per TP rank. Removing PLE
from device residency leaves an idealized 19.556 GiB of checkpoint payload per
GPU before replicated tensors, KV/index caches, CUDA graphs, and workspaces.

This is a planning bound, not a measured peak. Startup must record host RSS,
pinned memory, per-rank device peak, post-load device residency, and whether a
loader creates duplicate staging buffers. A 262144-token context is admitted
only after the measured peak leaves a safe margin on every 32GB GPU.

The SM70 TurboMind repack changes routed-expert FP4 scales from FP8 to FP16.
For TP4, routed experts are estimated at about 15.82 GiB/rank in the source
checkpoint and 17.57 GiB/rank after repack. This puts the idealized final
device weights near 21.3 GiB/rank. Because layers are repacked sequentially,
the estimated transient weight peak is about 22.1 GiB/rank before runtime
buffers, KV/index caches, NCCL, and CUDA graphs. These are storage calculations,
not `torch.cuda.max_memory_allocated` measurements.

The loader marks the PLE parameter as permanently host-resident so generic
quantization post-processing cannot stage the entire 11.921 GiB TP shard on a
GPU. Only its small scale parameter resides on device; lookup reads selected
FP8 rows through a stable UVA view and converts the gathered output to FP16.

With the real SM70 platform alignment and the QSA attention backend selected,
the V2 scheduler block is 784 tokens, the recurrent-state block is 32768
tokens at a 32768-token initial maximum length, and each padded recurrent page
is 802816 bytes. The exact synthetic model layout has one 24-layer uniform QSA
main/compressed group, one 12-layer fixed circular QSA ring group, three
12-layer GDN state groups, and one PLE short-convolution state group. It
allocates 24 physical cache tensors. The aligned pool cost is 10235904 bytes
(9.762 MiB) per shared block per TP rank.

For one request, the resulting cache-pool planning estimates are about 0.448
GiB/rank at 32K (47 shared blocks), 1.649 GiB/rank at 128K (173 blocks), and
3.241 GiB/rank at 262144 tokens (340 blocks). Combining the last figure with
the estimated 21.3 GiB final weights gives about 24.54 GiB/rank before CUDA
graphs, workspaces, NCCL, allocator fragmentation, and loader transients. This
explains why TP4 is plausible, but it is not evidence that the maximum context
will load safely.

## No-MTP TP4 decode performance and quality audit

The retained experimental online-QPN8 performance candidate is
`.artifacts/qwen38_flash_next_nvfp4_20260826/results/target80-qpn-m1-mtp-off-8192x512.json`.
It uses one request, TP4/PP1, FP16 activations, MTP off, V2, CUDA graphs, the
direct QPN-M1 NVFP4 experts, online channel-QPN8 attention/GR projections,
FlashAttention-V100 plus FlashQLA, and pinned-host PLE. Three matched repeats
measure 82.2521, 82.2750, and 82.2956 steady decode tokens/s: mean 82.274268
tokens/s, or 12.154469 ms/token. The repeat range is only 0.043466 tokens/s.
This is 4.904026 tokens/s faster and 0.770399 ms/token lower than the preceding
77.370242-token/s architecture control. The arithmetic gate emits `42` with
the unchanged token hash `93def17b...`, and the Chinese gate emits the expected
statement that water boils at 100 degrees Celsius under standard atmospheric
pressure with the unchanged token hash `11d98c01...`. Peak sampled device
memory is 29,752 MiB on each 32,768-MiB V100. Model load reports 20.86 GiB/rank,
3.73 GiB/rank available for KV cache, and a 0.16-GiB/rank graph footprint.

The graph-node trace is
`.artifacts/qwen38_flash_next_nvfp4_20260826/profiles/qwen38_nvfp4_target80_arch_mtp_off_i8192_o16_graph_nodes.nsys-rep`;
its stable parsed report is the adjacent `*_per_token_steady.{md,csv,json}`
set. Mean wall TPOT is 14.046 ms under Nsight, rank-max GPU service is 13.797
ms, and the residual host/scheduler gap is only 0.248 ms. Each rank replays
about 1,928 graph kernels per token, so both memory traffic and launch count
matter.

| architecture path | rank-average service | important sub-costs |
|---|---:|---|
| GR/HC read, mix, and combine | about 2.46 ms | QPN8 down 1.047 ms, reduce/SiLU 0.250 ms, QPN8 up+gate+mean 0.763 ms, combine+norm 0.400 ms |
| QSA selection and sparse attention | about 1.08 ms | MQA scorer 0.289 ms, persistent top-k 0.214 ms, sparse attention 0.444 ms, split merge 0.110 ms |
| attention input/output projections | about 1.27 ms | split-16 QPN8 0.789 ms, split-12 QPN8 0.480 ms |
| routed MoE expert GEMMs | 1.675 ms | TurboMind W4A16 expert kernels |
| MoE route and reduction | about 1.07 ms | specialized top-k 0.568 ms, activation 0.287 ms, prepare 0.107 ms, and weighted reduce 0.105 ms |
| TP communication | about 0.87 ms | exact 5-KiB four-rank push all-reduce dominates |
| GDN recurrent core and convolution | about 0.41 ms | recurrent update 0.282 ms and convolution update 0.126 ms |
| LM head and sampling | about 0.70 ms | LM-head GEMV dominates |
| PLE host lookup | about 0.02 ms | already hidden behind the first transformer work |

This ranking determines the implementation order. It also prevents repeating
architecture work whose maximum possible gain cannot close the current
0.424868-ms/token gap to the first 80 tokens/s gate.

The accepted QPN-M1 route removes the per-layer routed-input replication and
consumes the existing `(512, 2560, 40)` W13 and `(512, 160, 320)` W2
TurboMind-packed NVFP4 tensors directly. Across ten checkpoint experts, W13
falls from 20.983 to 12.213 microseconds with split-K 8; its maximum stage
error is 3.052e-5. W2 split-K 1 is bitwise equal and falls from 8.239 to 7.541
microseconds. A second screen spans expert IDs 0 through 470 and 16 random
inputs; the final weighted MoE output differs by at most 4.768e-7. Together
with removal of the traced 0.107-ms input-prepare work, the trace projection is
about 12.36 ms/token, or 80.9 tokens/s. The fixed resident-model measurement
above exceeds that projection at 82.274 tokens/s and is the performance
evidence; no MTP or speculative decoding is enabled in this result.

### Decode output-quality A/B (2026-08-28)

The published NVFP4 checkpoint quantizes only the routed experts. Online QPN8
also converts six dense projection roles to row-wise FP8 E4M3 at model load.
A retained real-weight screen reports 2.44% to 2.75% relative L2 error against
the original FP16 projections across mHC, GDN, and QSA. That is useful
numerical diagnostics, but it is not by itself a model-output quality result.

The matched model-level gate uses 32 fixed GSM8K questions with three indexed
sampling seeds per question (96 sequential requests), temperature 0.6,
top-p 0.95, top-k 20, natural EOS, and an 8192-token output limit. Both arms
run source `f4040b8f`, TP4/PP1, V2, no MTP, FP16 activation/KV,
FlashAttention-V100/FlashQLA, and FULL+PIECEWISE graphs; only online QPN8 and
its dependent GDN BA split differ. A boxed-first scorer gives 96/96 for both
arms, with zero invalid, replacement-character, NUL, or non-`stop` outputs.
QPN8 raises mean steady decode from 67.627 to 79.105 tokens/s (+16.97%) and
median steady decode from 67.640 to 79.369 tokens/s (+17.34%).

Only 8/96 token hashes match. The mean absolute output-length delta is 133.5
tokens and the maximum is 4477. One sample remains correct but grows from 1653
tokens and a repeated-4-gram ratio of 0.199 with QPN8 off to 3626 tokens and
0.507 with QPN8 on. A short GSM8K score therefore does not establish that
requantizing the checkpoint's dense BF16 projections preserves general output
quality. Online QPN8 is default-off and requires the explicit experimental
opt-in `VLLM_SM70_QWEN4_EXP_ONLINE_QPN8=1`; accepted no-MTP quality and speed
baselines must report the flag and must not attribute an opt-in QPN8 result to
the precision-preserving default route.

The fixed A/B execution SHA predates the compressed-QSA zeroing repair merged
as PR #373 (`ddd8c0b601`). A compressed QSA scheduler block is logically 784
tokens but stores one 98-row physical page. The old zeroer multiplied the
compression ratio a second time and could clear the wrong range when a block
was reused. Current main derives zeroing from `storage_block_size`, clears
exactly the selected physical page, and is covered by a real V100 page test
and a TP4 1K/4K generation smoke. Both A/B arms intentionally use the same old
binary to isolate QPN8, while this branch includes the QSA repair.

### Upstream correctness sync (2026-08-29)

The correctness audit starts from public `main` at
`62ad1e02693f4c857f3b7547cef1860ee54e8053` in the isolated branch
`codex/v100-qwen38-correctness-upstream-20260828-152418`. It adapts the
request-layout fix from vLLM PR #53896, the Mamba state-grid fixes from vLLM
PRs #53802 and #54076, and the EAGLE cache-drop fix from vLLM PR #48375. The
local adaptations also reject the unvalidated Qwen4Exp V1 runner override,
stop QSA from claiming batch-invariant split-K reductions, and make online
QPN8 experimental opt-in.

The retained source/unit evidence is:

- 124 focused Qwen4Exp, hybrid-cache, prefix-cache, QSA, and QPN policy tests
  passed; one separately executed native QPN operator test also passed.
- Seven V100 packed-GDN tests passed, including an exact FP32-beta state check.
- Changed-file pre-commit passed every applicable hook, including Ruff,
  markdownlint, mypy, SPDX, forbidden-import, configuration, and API checks.
- The downloaded RadixArk safetensors index assigns 296,475 tensors across 206
  files; comparing every file's actual keys found zero extra or missing keys,
  so vLLM PR #54230 does not affect this checkpoint and was not duplicated.
- vLLM PR #51599's async Mamba accepted-count race is already covered by the
  local MRV2 runner-owned snapshots and request-ID remapping; `InputBatch` is
  not an asynchronous D2H destination on this path.
- vLLM PR #53520 fixes prompt-logprob lifetime in the legacy runner. Qwen4Exp
  now rejects that runner, while MRV2 computes prompt logprobs before invoking
  its padded model drafter, so the vulnerable ordering is absent.
- vLLM PR #53122 configures quantized standalone DFlash draft models. Native
  Qwen4Exp MTP already calls `configure_quant_config` with `Qwen4ExpMTP`, so
  importing the DFlash guard would not change this checkpoint's route.
- vLLM issue #53982's narrow circular-table read is avoided in both local
  runners: QSA circular groups skip the generic slot-mapping kernel and build
  ring slots from logical positions in their metadata builder.

Two separate risks remain explicit. 1Cat PR #406 owns the independent
`KVBlockZeroer` asynchronous H2D staging-lifetime fix and should merge before a
high-concurrency prefix-cache acceptance run. Upstream issue #54199 still has
no proven fix for a donor-lifecycle overlap in Mamba prefix-cache precopy; a
bounds-only workaround would trade a crash for silent recurrent-state
corruption and is therefore not accepted without a reproducer and state
oracle.

The checkpoint also carries multimodal `mrope_section` and
`mrope_interleaved` fields in its text config. The initial SM70 route requires
`language_model_only` and removes those fields before constructing text RoPE.
For text-only positions the three MRoPE axes are identical, so this reduces to
the same one-dimensional rotation; warnings emitted while the raw config is
first validated do not indicate an unknown runtime RoPE implementation.

### Upstream methods and SM70 decisions

The [official Qwen repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)
and its
[technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
specify the 3:1 GDN/QSA pattern, four-branch GR, and the layer-2 PLE placement.
The report's relevant inference recommendations are fused GR reads/writes,
folding RMSNorm into the read, low-precision residual-branch storage, and
FlashQLA for GDN. It explicitly reports quality loss when GR is sparsified to
only two branches, so branch pruning is excluded from this port. FP8 branch
storage is not copied blindly either: V100 has no native FP8 arithmetic, so a
conversion-inclusive screen is required before it can replace FP16 residuals.

The official [vLLM Qwen4Exp PR](https://github.com/vllm-project/vllm/pull/53896)
is the structural base for the QSA/GDN/GR model path. The current branch adds
SM70-only projection, cache, attention, and graph routes rather than importing
unrelated upstream platform changes. The
[SGLang day-zero optimization report](https://staging.lmsys.org/blog/2026-08-26-qwen-flash-next)
describes the production HC strategy more precisely: low-M Mix uses split-K
GEMMs with SiLU in the down epilogue and sigmoid/four-stream reduction in the
up epilogue, while Combine splits the hidden dimension to expose more CTAs.
The experimental QPN8 HC route already mirrors those two epilogue fusions and
the SM70 Combine kernel already partitions the hidden dimension; importing the
SM100-only CuTeDSL implementation from
[FlashInfer PR 4266](https://github.com/flashinfer-ai/flashinfer/pull/4266)
would therefore add no V100 kernel. The exact V100 screen also rejects
collapsing the two-QPN8-kernel experiment into one persistent launch. The
[SGLang day-zero HC kernel](https://github.com/sgl-project/sglang/blob/02d38b77db92699e5d4f1a78226bf711e9cc762a/python/sglang/srt/layers/hc_mix_triton.py)
replaces the FP16 down-GEMV, SiLU, up-GEMV, sigmoid, and four-branch mean with
one persistent kernel. A V100 form preserves the existing QPN8 weights and
exact split-K reduction order. All 32 checkpoint-weight seeds and 1,000 graph
replays are bitwise clean, but its cold median is 26.624 microseconds versus
23.552 microseconds for the accepted path. At 97 HC calls per token that would
regress TPOT by about 0.298 ms, so the experiment is removed. Using SGLang's
FP16 weights unchanged would also double the GR weight traffic.
SGLang's fused QSA norm/RoPE/cache-store kernels map to only about 0.063
ms/token in this trace and are therefore a later cleanup, not the 80-token/s
critical path.

[TokenSpeed's Qwen4Exp implementation](https://github.com/lightseekorg/tokenspeed/tree/4cb771487f2a070bc3a39c36b6dc8d959a36a92f)
adds a streaming QSA scorer/top-k that avoids materializing the full FP32 score
matrix, as well as FP8 PLE storage. The streaming design is attractive at long
context, but the exact 8K graph has only 2,048 visible compressed blocks out of
a 32,768-block capacity. TokenSpeed's default geometry would launch 64 split
programs and merge 64 by 512 packed candidates while only four splits contain
work. It must beat the already-screened two-warp scorer plus existing top-k in
an exact cold-cache microbenchmark before being ported. This checkpoint's PLE
is already FP8 E4M3 on pinned host memory, so TokenSpeed's FP8 table route does
not reduce its current 11.92-GiB/rank footprint or materially improve the
measured 0.02-ms PLE latency.

The original [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464) and the
[Flash Linear Attention project](https://github.com/fla-org/flash-linear-attention)
support recurrent decode plus chunked/parallel prefill as the intended
hardware decomposition. This branch already uses FlashQLA for the recurrent
decode state update and keeps FLA-style packed state handling. The current
trace assigns only about 0.41 ms/token to the GDN recurrent core plus short
convolution, versus more than 2 ms to its surrounding GR and projections, so
replacing the GDN recurrence is lower priority than removing those projection
and launch costs. The [Hyper-Connections paper](https://arxiv.org/abs/2409.19606)
also makes clear that the four streams carry distinct learned paths; pruning or
averaging them before their learned gates would change the architecture, not
merely optimize it.

The currently admitted candidates are deliberately narrow:

- QSA MQA scorer: 32 columns and two warps for SM70 M=1. The checkpoint-shape
  cold-cache screen falls from 29.696 to 11.264 microseconds, without changing
  selected block IDs. The projected 12-layer saving is about 0.22-0.33
  ms/token.
- MoE router: an exact M=1, E=512, top-10, FP16, renormalized SM70 kernel. It
  falls from 11.729 to 5.657 microseconds in the source screen and preserves
  NaN, positive-infinity, all-negative-infinity, and tie behavior. The trace
  projection is about 0.53 ms/token. Model-level acceptance also requires the
  `SM70 Qwen3.8 E512/K10 router top-k path enabled` marker.
- GR/HC persistent launch: rejected despite bitwise output and clean replay
  state because it is 3.072 microseconds slower per call in the checkpoint-
  shape cold-cache screen, a projected 0.298-ms/token regression. The
  experimental split-QPN8 down/reduce/up implementation remains unchanged.
- MRv2 greedy sampling: the active V2 runner always materialized globally
  gathered logits and then launched Gumbel sampling, even for the exact
  temperature-zero single-request decode used here. The new SM70-only route
  keeps the same full local LM-head calculation, performs the existing
  TP-local exact argmax/pair gather, and falls back whenever logprobs, grammar,
  penalties, bad words, logit bias, NaN counting, random sampling, prefill, or
  speculative decoding is present. This removes the traced 0.103-ms Gumbel
  kernel and 0.044-ms full-logit all-gather before accounting for the local
  argmax cost.

The QSA scorer, router, and V2 greedy projections expose about 0.90-1.01
ms/token in the historical online-QPN8 trace. That trace is useful for kernel
ranking but no longer proves the precision-preserving 80 tokens/s target; the
next optimization cycle must use a fresh no-QPN8 trace and retain the exact
checkpoint weights. The rejected persistent GR candidate is not included.

## Native MTP4 TP4 validation snapshot

The first complete native-MTP run uses V2, TP4, FP16 activations, ModelOpt
NVFP4 weights, `FLASH_ATTN_V100` for target and draft attention, four draft
tokens, `mamba_cache_mode=align`, prefix caching, chunked prefill, and
FULL+PIECEWISE CUDA graphs. It runs two identical deterministic 1024-token
prompts with 256 forced output tokens each. The artifact is
`.artifacts/qwen4_exp_mtp_tp4_20260827/mtp4_prefix_graph_i1024_o256_r2_hetero_v2.json`.

- Source HEAD is `d9a39ea434` on the Qwen3.8 worktree branch. The measurement
  also sees the worktree's separately owned, uncommitted SM70 kernel changes;
  it is bring-up evidence and must be repeated from a clean, pinned source
  before becoming release-baseline evidence.
- Target plus MTP weights use 23.16 GiB/rank. The aligned MTP attention block
  is 816 tokens with 1.62% recurrent-page padding. Available KV-cache memory
  is 4.77 GiB/rank, or 219,942 tokens. Observed peak device memory is
  32,330 MiB on every 32,768-MiB V100; the run therefore has only 438 MiB of
  peak device headroom at `gpu_memory_utilization=0.90`.
- PLE remains host-resident at 11.92 GiB/rank. The sampled minimum host
  `MemAvailable` is 48.107 GiB and minimum free swap is 247.994 GiB.
- Both repeats emit 256 tokens and are exactly equal token-for-token. The first
  request has 9.750-second TTFT and 53.125 steady decode tokens/s. The repeated
  prefix has 2.667-second TTFT and 52.187 steady decode tokens/s. Mean steady
  decode is 52.656 tokens/s; the first end-to-end request includes prefill and
  JIT and is not a decode baseline.
- Across 256 speculative steps, the MTP head proposes 1,024 draft tokens and
  254 are accepted. Mean acceptance length including the target bonus is
  1.9921875. Draft acceptance is 24.8047%; per-position acceptance is
  54.6875%, 26.5625%, 13.28125%, and 4.6875%.
- A strictly matched no-MTP run is required before claiming target-token
  equality or a verifier-cost ratio. Older 1024x256 artifacts use the distinct
  Qwen3.8-27B checkpoint and are not valid controls for Flash Next.

### MTP4 cycle-cost split

The graph-node trace is
`.artifacts/qwen4_exp_mtp_tp4_20260827/mtp4_cost_nsys_i1024_o32_hot_u085_v2.nsys-rep`.
It changes only `gpu_memory_utilization` from 0.90 to 0.85 to leave room for
Nsight Systems; the resulting KV-pool size is not a new throughput baseline.
The uninstrumented 0.90 run above remains the absolute-speed evidence.

Nsight's PIECEWISE ranges are subgraphs rather than emitted-token boundaries,
so the generic per-token parser must not be used for this trace. Stable cycle
boundaries are instead identified by the first W13 kernel of the unquantized
MTP MoE. The first target TurboMind NVFP4 GEMM after that proposal marks the
start of the next target verification. Fourteen complete steady cycles have
the following wall-time split:

| phase | mean | median | p90 | share of cycle |
|---|---:|---:|---:|---:|
| MTP4 proposal plus scheduler handoff | 13.477 ms | 13.426 ms | 13.699 ms | 31.2% |
| five-token target verify, reject, and state update | 29.714 ms | 29.710 ms | 29.750 ms | 68.8% |
| complete speculative cycle | 43.191 ms | 43.147 ms | 43.416 ms | 100% |

At the approximately 2.00-token natural acceptance length, this corresponds to
about 14.86 ms of target verification and 6.74 ms of proposal/handoff per
emitted token, or 21.60 ms total (46.3 tokens/s from the traced cycle). The
five-position verifier is only about 40% utilized: two emitted positions per
five computed positions. These normalized costs, rather than raw cycle time
alone, are the comparison gates for MTP3.

The proposal phase's rank-average GPU-service sums are led by FP16 GEMM/GEMV
(2.791 ms), other MTP/QSA kernels (2.345 ms), and TP communication (1.337 ms).
The verification phase is led by FP16 GEMM/GEMV (10.357 ms), other target
kernels (5.511 ms), target TurboMind NVFP4 GEMM (3.665 ms), SiLU/gating
(1.910 ms), and TP communication (1.762 ms). GPU-service sums can overlap and
are composition evidence, not additive wall time.

The current route remains fixed at native MTP4. Before changing verifier width,
the acceptance defect must be corrected and remeasured: the first-position
acceptance near 50% shows that the dominant issue occurs before later draft
iterations. Sampling and bookkeeping micro-optimizations come only after a
matched MTP4 acceptance measurement. The 32-token trace prompt still forces
generation after an early EOS and is therefore cost evidence only, not
representative acceptance evidence.

### Natural-prompt baseline and MTP4 acceptance root cause

The completed portion of
`.artifacts/qwen4_exp_mtp_tp4_20260827/mtp4_acceptance_natural6_batch_o256_v4_sm70cpp.log`
is the current natural-prompt MTP4 baseline. Four requests complete before the
fifth prefill triggers an asynchronous illegal-memory access. Immediately
before that failure, mean acceptance length is 2.00, 248 of 992 drafted tokens
are accepted, per-position acceptance is 49.6%, 25.4%, 15.3%, and 9.7%, and
aggregate output throughput is 47.54 tokens/s. These are preliminary
four-request statistics, not a stable six-request result.

The failure is deterministic at the fifth request in both the original
TileLang GDN prefill route and a separately checked SM70 C++ GDN prefill
route. The failing request schedules 88 new tokens and receives block IDs 89
and 90 for the first two cache groups after four prior requests have finished.
An isolated 88-token C++ GDN prefill matches the FLA reference and survives 64
repetitions. Because a known fifth-request failure would discard the complete
benchmark JSON, the first optimization run deliberately keeps the exact four
prompts that already complete in the MTP4 baseline. The 88-token shape versus
prefix-cache/block-lifetime distinction remains a separate minimum-size
stability experiment rather than being mixed into the acceptance comparison.

Commit `70b63a1e45` adds two independently measurable MTP cost optimizations:

- The V2 Eagle path now honors `use_local_argmax_reduction`. Greedy MTP no
  longer gathers a 248,320-column logits tensor on every draft step; it uses
  vocab-shard-local top-1 values and IDs followed by a compact TP reduction.
  SM70 MTP defaults to greedy drafting so the fast path is reachable, while
  DFlash retains probabilistic drafting.
- Qwen4Exp MTP can reuse the step-0 QSA sparse indices on later draft steps.
  The prefill output is compacted to one row per request and steps 1+ skip the
  repeated QSA top-k. This follows the upstream vLLM lifecycle hooks and the
  SGLang Qwen4Exp default, while exposing
  `index_share_for_mtp_iteration=false` as an A/B control.

The acceptance audit found a separate correctness root cause. This checkpoint
stores the BF16 draft routed experts as two fused tensors:
`mtp.layers.0.mlp.experts.gate_up_proj` with shape `(512, 1280, 2560)` and
`mtp.layers.0.mlp.experts.down_proj` with shape `(512, 2560, 640)`. The private
tree's legacy `FusedMoE` recursive mapping recognized only per-expert names
such as `experts.0.gate_proj.weight`. Its child loader consumed the unmatched
generator without reporting those tensors, leaving `w13_weight` and
`w2_weight` at their `torch.empty` initialization. That is about 1.17 GiB of
uninitialized draft parameters per TP rank, or 4.69 GiB across TP4, and directly
explains why first-position acceptance was already abnormally low.

Commit `cf5c44a1aa` ports the upstream fused-checkpoint aliases for Qwen4Exp and
tests gate/up splitting plus complete expert fan-out. Commit `ceb543c055` makes
the draft loader fail closed if either routed-expert parameter is absent, so a
future mapping regression cannot masquerade as a successful model start. The
focused loader gate is 21 passed tests plus Ruff and format checks.

### Corrected MTP4 acceptance, quality, and warmup closure

The standalone drafter now prefers the four compact `model-bf16-*` shards that
contain all 33 required MTP/shared parameters instead of rescanning all 206
target shards. The complete-expert validation remains fail closed, and generic
safetensors/bin/pt patterns remain compatibility fallbacks. In the accepted
11-request transition artifact
`.artifacts/qwen4_exp_mtp_tp4_20260827/mtp4_transition9_v16_storage_ratio_natural4_same7_gpu4567.json`,
target loading reads 206 shards once and MTP reads only four shards / 14.91 GiB
in 5.04 seconds. All requests complete without a new GPU Xid or cross-request
state drift.

That run emits 2,816 tokens with 2.9159 mean accepted length including the
target token, 47.8972% draft-token acceptance, and 72.8972%, 50.2596%,
38.5254%, and 29.9065% position acceptance. The ten warmed requests average
78.98 pure decode tokens/s. Minimum host `MemAvailable` is 47.46 GiB, peak swap
use is 8.03 GiB, and process exit restores roughly 115--120 GiB available with
no vLLM worker left behind.

Matched QSA index-sharing A/B/A did not expose an acceptance defect: sharing
is 0.0359 accepted tokens better overall than recomputing indices. SGLang's
causal-tail refresh candidate is also rejected because it loses 0.0322 accepted
tokens and reduces decode speed. The retained natural-prompt control measures
2.762 mean accepted length and 44.05% draft acceptance, inside the published
NVFP4 range. Conditional acceptance after the first draft is roughly 70--74%,
so the weakest event is the first proposal rather than accumulated MTP-step
drift.

The corrected no-thinking HumanEval8 artifact is
`.artifacts/qwen4_exp_mtp_tp4_20260827/mtp4_transition16_v23_q38_moe_warmup_humaneval8_nothink_gpu0123.json`.
It emits 797 tokens. Global MTP4 acceptance is 4.747 mean length and 93.676%
draft acceptance, with 98.82%, 95.88%, 91.76%, and 88.24% by position.
Excluding the cold first request gives 4.696 mean length, 92.407% draft
acceptance, and 126.81 weighted pure decode tokens/s. The standard semantic
policy preserves each task's imports and signature and passes 8/8 in the
sandboxed executor. Minimum host `MemAvailable` is 48.08 GiB, peak swap use is
7.98 GiB, and each GPU peaks at 32,179 MiB; the post-exit audit finds no
persistent worker or host-memory leak.

The exact TP4-local BF16 draft-MoE shape is
`E=512, N=160, K=2560, topk=10`. An exact-shape SM70 tile of
`BM=2, BN=128, BK=64, G=1`, four warps, and three stages reduces M5 from
1492.500 to 283.853 microseconds (5.26x) and M1 from 425.789 to 272.415
microseconds (1.56x), with bitwise-identical output. The route is bounded by
the exact SM70 TP4 local MoE shape (currently exercised by Qwen3.8), is enabled
by default, and can be rolled back with
`VLLM_SM70_MTP_MOE_TUNED_CONFIG=0`.

V2 needs one additional prefill specialization beyond its faithful M5/M1
decode warmup. The real HumanEval prompt first runs the draft model at M152;
existing M8 and M18 captures do not cover the sorted-assignment,
`M * topk`-divisible-by-16 specialization. A fresh-cache V100 microtest records
two new compilations and 526.97 ms at M152 without M16. Adding only M16 reduces
M152 to zero new compilations and 4.51 ms. The final TP4 integration run logs
KV-zero, M16, and M5/M1 warmup before request monitoring and has no first-
request `fused_moe_kernel` JIT. Its artifact is
`.artifacts/qwen4_exp_mtp_tp4_20260827/mtp4_transition19_v26_q38_v2_m16_moe_warmup_humaneval1_nothink_gpu0123.json`.
It emits the same 173 token IDs as the earlier quality run, keeps 4.943 mean
accepted length / 98.571% draft acceptance, and reaches 138.26 steady decode
tokens/s.

The accepted Nsight control still identifies target verification as the next
MTP cost center: one 43.348-ms speculative round spends 30.040 ms in the
five-token verifier, versus 2.733 ms in the first draft graph and 5.603 ms of
critical GPU service in the remaining draft/sampling passes. Further MTP speed
work should reduce verifier cost without trading away the now-validated
acceptance or code quality. The separate no-MTP final target remains at least
100 tokens/s; its last accepted control is 82.274 tokens/s.

## Acceptance gates

1. Static route: Transformers config, model registry/processor registration,
   QSA/GDN/GR/PLE modules, and ModelOpt NVFP4 mapping load without importing an
   Ampere-only backend. Multimodal execution is outside this first route.
2. Loader route: TP4 expert shards select TurboMind SM70 W4A16; PLE shards are
   born on pinned CPU memory and do not consume persistent device memory.
3. Numerical route: focused operator comparisons against FP32/FP16 references,
   followed by deterministic token-ID and output checks on the full model.
4. Memory route: 32K bring-up first, then 128K and the exact 262144 boundary;
   report controlled OOM separately from corrupted output.
5. Performance route: one request, TP4, PP1, MTP off, FP16 activations, 8192
   input tokens and 512 output tokens. Report TTFT/prefill separately and
   calculate steady pure decode from emitted tokens 33-512. The first recovery
   gate is at least 80 emitted tokens/s (at most 12.5 ms/token); the final target
   remains at least 100 emitted tokens/s (at most 10 ms/token), both with CUDA
   graphs enabled. Record an otherwise identical eager control.
6. MTP follow-up: report accepted length, target passes, emitted tokens/s, and
   output quality separately; do not compare accepted candidates with emitted
   tokens.

## Initial implementation boundary

Reuse the upstream Qwen4Exp Python structure and tests where they match this
tree. Do not import unrelated AMD, SM90, build-system, or broad engine changes.
The first SM70-specific changes are limited to genericizing the existing
TurboMind NVFP4 MoE shape contract, adding the QSA/indexer route, and adding a
pinned-host PLE loader/gather path. Optimize GDN, GR, sparse attention, and MTP
only after profiles identify them as measured decode bottlenecks.

## Source validation snapshot

- The ModelScope download completed successfully at the path above. A full
  post-download verification checked 419 files and about 125.910 GiB of
  safetensor payload: zero files were missing and zero size or SHA-256 values
  differed from the remote manifest.
- The real downloaded `config.json` resolves without remote model code as
  `Qwen4ExpConfig` / `Qwen4ExpTextConfig`: 48 layers, 36 GDN, 12 QSA, 512
  experts, top-10, HC count four/rank 320, and one trigram PLE layer.
- Exact-SM70 configuration construction with FP16, TP4, prefix caching off,
  language-model-only mode, and V2 selects `ModelOptNvFp4Config`, the
  pinned-host PLE default, and the Qwen4Exp PLE/QSA compilation split
  operators. The same real configuration rejects the unvalidated multimodal
  route with an actionable `--language-model-only` error.
- Exact real-checkpoint construction with native MTP4 and prefix caching on
  resolves the target as `Qwen4ExpForConditionalGeneration`, the draft as
  `Qwen4ExpMTP`, target and draft attention as `FLASH_ATTN_V100`, and recurrent
  caching as `align` with 16-token configured blocks and chunked prefill.
  Focused prefix-cache/QSA coordinator and model-config tests pass. The
  non-cacheable QSA compressor ring is skipped during prefix-hit matching,
  matching the current upstream Qwen4Exp contract.
- TP4 loaded all 206 checkpoint shards in 109.27 seconds and then correctly
  rejected an incomplete runtime extension set: `_C` and
  `_C_stable_libtorch` were present, but `_moe_C` was omitted. A complete
  runtime must carry all three. The matching `_moe_C` artifact has SHA-256
  `a14eeb4fa06947e335cf69ee188e23509fc294da61cada786df09888ca5b4469`;
  its graph-safe permute, workspace-size, unpermute schemas, and SM70 support
  probe all pass before the next full-model attempt.
- Full 48-layer meta construction from the real checkpoint config succeeds in
  language-model-only mode. It instantiates QSA, GDN, HC, PLE, and all 512
  experts without materializing weights; the routed experts select
  `ModelOptNvFp4SM70MoEMethod(use_a16=True)` and the PLE table has shape
  `(320001536, 160)` with FP8 E4M3 storage. This constructor probe used TP1;
  TP4 selection and expert geometry are covered separately. Full TP4 target
  plus native-MTP loading now completes with 23.16 GiB/rank of loaded model
  state before cache and graph allocation.
- Focused CPU tests cover PLE shard loading and hashing, `seed=None`, permanent
  host residency during post-load processing, QSA cache grouping, V1 and V2
  n-gram inputs, V2 circular block-table sizing, scheduler-manager conversion,
  official checkpoint weight mappings, and Qwen3.6/Qwen3.8 NVFP4 route
  selection. The current CPU-only focused run is 76 passed and 7 CUDA skips;
  all 55 changed Python files pass Ruff, format, and compileall checks.
- In the pre-final real V100-SXM2-32GB snapshot, 63 focused tests pass. They
  include the Triton V2 slot-mapping kernel with its QSA circular group
  disabled, pinned-host FP8 lookup through a CUDA UVA view, the compressed QSA
  storage-page reshape, V2 committed-token PLE state, and the SM70 ModelOpt
  NVFP4 selection gates. A final V100 rerun is still required after the current
  GPU owners release a device.
- The upstream QSA fused pre-indexer executes on SM70 for both ordinary RoPE
  and MRoPE inputs and matches a PyTorch normalization reference. This also
  exposed and fixed two private-tree API differences: QKV projection is local
  because the branch's `Qwen3NextAttention` has no `_project_qkv_gate`, and its
  `triton_mrope` accepts eight rather than nine arguments.
- Actual SM70 platform alignment produces a 784-token attention block and an
  802816-byte padded recurrent page; the exact synthetic 48-layer cache layout
  validates successfully after that alignment.
- The HC grouped norm/gate/combine kernels pass FP16 reference checks on a real
  V100. A captured pinned-host FP8 embedding probe (228.9 MiB synthetic table,
  16 rows by 160 elements per replay) measured 95.81 microseconds/replay,
  including the input-ID copy. This only demonstrates that the isolated UVA
  lookup can be captured and is not by itself an end-to-end throughput result
  or a measurement of the full 11.921 GiB TP shard.
- The existing general KV-cache utility/manager suites pass 69 tests; one
  unrelated DeepSeek-v4 fixture failure is unchanged from the integration
  base because its `SimpleNamespace` omits `max_in_flight_tokens`.
