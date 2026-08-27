# Qwen3.8 Flash Next NVFP4 on SM70

## Status and ownership

- Status: source bring-up, pinned-host PLE, native Qwen4Exp MTP, and
  prefix-cache configuration are implemented; focused CPU/configuration gates
  pass and the local ModelScope snapshot is fully verified. The current
  accepted no-MTP TP4 candidate reaches 82.274 steady decode tokens/s on the
  8192x512 contract, with exact arithmetic and Chinese quality gates. Its
  direct NVFP4 ten-route QPN-M1 MoE route is now the default for the exact
  SM70/TP4/B1 contract. The preceding 77.370-token/s graph-node trace remains
  the detailed optimization control. Native MTP4 also completes TP4 model
  load, graph
  capture, warmup, and two 1024x256 requests; its separate acceptance and
  cycle-cost evidence remains recorded below.
- Integration line: public `origin/main`, through Draft PR
  [#345](https://github.com/1CatAI/1Cat-vLLM/pull/345).
- Base SHA: `d63e9490f65f9e01f6649053c1ab72922034b931`.
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

## No-MTP TP4 decode control and architecture audit

The current unprofiled control is
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
above exceeds that projection at 82.274 tokens/s and is the acceptance
evidence; no MTP or speculative decoding is enabled in this result.

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
The accepted QPN8 HC route already mirrors those two epilogue fusions and the
SM70 Combine kernel already partitions the hidden dimension; importing the
SM100-only CuTeDSL implementation from
[FlashInfer PR 4266](https://github.com/flashinfer-ai/flashinfer/pull/4266)
would therefore add no V100 kernel. The exact V100 screen also rejects
collapsing the accepted two-QPN8-kernel path into one persistent launch. The
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
  shape cold-cache screen, a projected 0.298-ms/token regression. The accepted
  split-QPN8 down/reduce/up implementation remains unchanged.
- MRv2 greedy sampling: the active V2 runner always materialized globally
  gathered logits and then launched Gumbel sampling, even for the exact
  temperature-zero single-request decode used here. The new SM70-only route
  keeps the same full local LM-head calculation, performs the existing
  TP-local exact argmax/pair gather, and falls back whenever logprobs, grammar,
  penalties, bad words, logit bias, NaN counting, random sampling, prefill, or
  speculative decoding is present. This removes the traced 0.103-ms Gumbel
  kernel and 0.044-ms full-logit all-gather before accounting for the local
  argmax cost.

The QSA scorer, router, and V2 greedy projections together expose about
0.90-1.01 ms/token of gross trace cost, which is enough on paper to cross the
immediate 80 tokens/s gate after allowing for the pair-gather argmax overhead.
A single resident-model run will validate them together; the model will not be
restarted separately for each micro-optimization. The rejected persistent GR
candidate is not included in that run.

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

The next GPU gate is one TP4, prefix-cache-on, four-prompt native MTP4 run with
greedy local argmax and QSA index sharing explicitly enabled. It uses the same
prompts, order, sampling, and output cap as the completed four-request baseline
and records per-request and per-position accepted length, code output quality,
emitted tokens/s, and peak memory. MTP3 is not an optimization candidate for
this gate. A subsequent Nsight trace is warranted only after the corrected
MTP4 acceptance result is stable; it will report proposal time, five-token
target verification time, and cycle cost per emitted token separately.

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
