# SM70 GLM-5.3-Flash NVFP4 Bring-Up

Date: 2026-08-27

## Contract

- Checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4`, revision
  `9e0d74e3cef17f634e84fb8e2223707e02616290`.
- Hardware: eight V100 GPUs, TP4 and PP2. PP is required for weight capacity;
  every PP stage contains one four-GPU TP group.
- Runtime activations and unquantized weights: FP16 on SM70. The checkpoint's
  BF16 tensors are cast during load because V100 has no native BF16 execution.
- Speculative decoding: disabled. The checkpoint contains one NEXTN/MTP layer,
  but MTP results do not count toward this acceptance.
- Speed gate: batch-one text generation, exact 1024-token input and 256-token
  output, steady pure decode reported separately from TTFT and prefill. The
  initial target is at least 70 output tokens/s with CUDA graphs enabled.
- Quality gate: fixed-seed greedy token identity across three same-process
  requests, finite logits, coherent Chinese and English output, and a natural
  official-sampling completion.

## Architecture

The language backbone has 45 layers, hidden size 4096, and a 1,048,576-token
position limit. Attention follows an exact four-layer cycle:

- 34 KDA linear-attention layers at indices `0,1,2,4,5,6,...,44`. KDA uses
  64 heads of dimension 128, three independent convolutional states with a
  kernel width of four, and a recurrent FP32 matrix state. The safe gate has a
  configured lower bound of -5.
- 11 DeepSeek-style sparse MLA layers at indices `3,7,11,...,43`. They use
  NoPE MLA with 64 query heads, 256 NoPE Q/K dimensions, 256 value dimensions,
  a 1536 Q LoRA rank, and a 512 KV LoRA rank.
- Every sparse layer has a 32-head, 128-dimension DSA indexer. It selects 2048
  pools, compresses four tokens into each KPool entry, and always retains the
  incomplete tail pool. The indexer and tail require separate scheduler
  semantics even though the tail co-owns the indexer's backing tensor.

The residual path uses mHC4 with 20 Sinkhorn iterations. Layers 0-2 have dense
12288-wide SwiGLU MLPs. Layers 3-44 have 288 routed experts, top-8 routing, a
2048-wide expert intermediate, one separate shared expert, sigmoid routing,
FP32 router logits, normalized top-k probabilities, and a routed scale of 2.5.

The model is natively multimodal. Its vision tower has 24 layers, hidden size
1024, 16 heads, 448-pixel images, 14-pixel patches, temporal patch size two,
and a 4096-dimensional language projection. Initial SM70 acceptance is for the
text path; image and video serving need their own memory and quality gates.

## Checkpoint Format

The checkpoint contains 120 safetensor shards and is about 181 GiB. It was
produced by ModelOpt 0.45.0. Its `NVFP4` configuration is weight-only:
`input_activations` and `output_activations` are null. Only the routed-expert
gate/up/down matrices are packed as E2M1 FP4 with FP8 E4M3 block scales at
group size 16 and FP32 tensor scales. Attention, KDA, the sparse indexer,
shared experts, routers, dense MLPs, vision, MTP, mHC, embeddings, LM head,
and norms remain BF16.

Under TP4, each routed expert uses the validated local shapes:

- packed gate/up: `[288, 1024, 2048]` bytes after stacking;
- packed down: `[288, 4096, 256]` bytes after stacking;
- global contract: hidden 4096, intermediate 2048, 288 experts, top-8.

The SM70 path combines block and global scales once, repacks directly into the
TurboMind layout, and deletes the checkpoint tensors. It never keeps a full
FP16 expert-weight copy.

## Upstream Disposition

- vLLM PR [#53906](https://github.com/vllm-project/vllm/pull/53906) is the
  primary model implementation. It is still open, needs a rebase, and mixes
  the model with broad scheduler, connector, multimodal, and runner changes.
  This tree ports the model and the required hybrid-cache pieces instead of
  merging the entire conflicting commit.
- SGLang PR [#36507](https://github.com/sgl-project/sglang/pull/36507) confirms
  the same KDA/DSA/KPool/mHC/NEXTN architecture, but its reported validation is
  FP8 on 4x GB300 TP4/EP4 and 8x H100 TP8/EP8. It is useful as a semantic
  reference, not as an SM70 kernel or speed baseline.
- SGLang PR [#36513](https://github.com/sgl-project/sglang/pull/36513) reports
  final-weight Blackwell/Hopper serving recipes and notes that some measured
  BF16 high-throughput runs disabled shared-expert fusion. 1Cat keeps the GLM
  shared expert as a separate dense branch, so routed NVFP4 experts are not
  fused with an incompatible BF16 shared-expert tensor.

## 1Cat Integration

The adaptation is divided into independently testable surfaces:

1. Register `glm5_next`, its Transformers config and processor, multimodal
   wrapper, text backbone, KDA, sparse MLA/indexer, mHC, and NEXTN modules.
2. Carry KDA's bounded gate into the local FLA recurrent/chunked operators and
   preserve the existing SM70 recurrent schedule.
3. Add GLM hybrid-cache grouping: 34 KDA states are balanced into four Mamba
   groups; 11 MLA and 11 compressed-indexer tensors own the physical slots;
   the 11 four-token tail caches co-own their sibling indexer allocations.
4. Virtually split the actual TP4 indexer storage block of 288 entries into
   nine 32-entry kernel pages. Exclude the one-block-per-request tail scratch
   group from prefix-cache hashing and global block-size selection.
5. Carry mHC post/combination state across the PP boundary. The PP payload is
   hidden `[B,4096]`, residual `[B,4,4096]`, post `[B,4,1]` FP32, and
   combination `[B,4,4]` FP32.
6. Admit ModelOpt NVFP4 only on exact SM70 and route the validated GLM expert
   contract to the graph-safe TurboMind kernel. No Marlin or dequantized-FP16
   fallback counts as accepted performance.
7. Encode KPool E4M3 entries in software on exact SM70. V100 cannot execute
   native FP8 conversions, so the writer stores the checkpoint-compatible
   E4M3 byte representation through `uint8` pointers. Sparse-indexer query
   rotation and scoring stay in FP16 and reuse the existing SM70 HMMA path.
8. Store sparse-MLA latent KV in a GLM-specific packed E4M3FN page. Each token
   uses 512 data bytes plus eight UE8M0 power-of-two scales (64 values/group),
   for 520 bytes total. Decode keeps this packed representation resident and
   gathers/dequantizes only the selected latent rows into a fixed-width FP16
   workspace before two Tensor Core GEMMs. The older direct scalar kernel is
   retained only as a reference/test path and is not accepted for B1 decode.
   Use the explicit `fp8_e4m3` cache dtype because the historical generic SM70
   `fp8` alias resolves to E5M2 for other model families.
9. Keep all GLM mHC4/H4096 execution on native SM70 kernels. Small-M fused
   decode follows the DeepSeek-V4 FP32 staging design, but its final Sinkhorn,
   residual mix, and RMSNorm stage is a dedicated single-CTA CUDA kernel for
   exact FP16 SM70. Standalone pre/post and large-M prefill use dedicated
   Triton kernels that avoid TileLang's SM70 BF16-header compilation failure.
   Layer zero also uses the mathematically equivalent broadcast weight,
   avoiding a four-stream expanded GEMV.
10. Fuse KDA's two B1 `128 -> 2048` f/g projections into one exact-shape SM70
    CUDA launch. It preserves FP32 accumulation and FP16 stores, is graph-safe,
    and hard-errors if the native operator is absent for the accepted TP4
    contract instead of silently returning to two cuBLAS launches.

## Validation Status

The 120 checkpoint shards are complete at the pinned revision (about 181 GiB).
The full source extension build completed on the target host. Import checks
confirm the SM70 NVFP4 prepare/dense-stage operators, strided MoE pointer
builder, and MoE permutation operator are registered. Static model/config
parsing, PP intermediate construction, weight-name mapping, KPool E4M3 byte
encoding, KPool tail slot mapping, sparse decode sequence lengths, hybrid
cache layout, and compressed physical-page reshape are covered by focused
tests. The KPool GPU suite passes 28 tests, and the current CPU integration
suite passes 64 tests.

An eight-process Gloo construction smoke using the exact TP4/PP2 rank layout
also passes on `meta`: PP0 owns 23 layers, PP1 owns 22 layers, every rank
selects `ModelOptNvFp4SM70MoEMethod`, and the PP payload carries `hidden_states`,
`residual`, `post`, and `comb`.

Exact eight-V100 dummy-weight requests now pass in eager mode for both FP16 and
packed E4M3FN KV. Runtime logs select the GLM sparse backend, ModelOpt NVFP4
TurboMind MoE, KDA recurrent kernels, KPool/indexer kernels, standalone and
fused SM70 mHC kernels, and direct packed-FP8 sparse MLA. At the same cache
budget, reported capacity increases from 66,355 tokens with FP16 KV to 111,957
tokens with packed E4M3FN KV (about 1.69x). The focused mHC suite passes 17
tests on V100, including M=1 decode and M=17 prefill boundaries.

The real checkpoint now loads and serves on eight V100s with TP4/PP2, packed
E4M3FN KV, no MTP, and full decode CUDA Graph capture. A retained 16-input,
16-output route benchmark improved steady pure decode from 5.35 tok/s on the
initial scalar sparse-attention route to 45.83 tok/s after the packed-KV B1
gather/dequant plus Tensor Core GEMMs. Both runs emitted the identical token
hash `70007811d61c68bb6ec6b4ac5758744f2d4c6b64b51b6b1352f380091c990902`.
This is a short optimization checkpoint, not the 1024/256 acceptance result.

An Nsight Systems graph-node trace reports a 23.76 ms mean replay interval and
95.09% graph-node coverage. The sparse attention hotspot is gone; the remaining
rank-average GPU service is dominated by pipeline send/receive wait (12.46 ms,
including the idle peer stage), dense GEMV/GEMM (4.55 ms), routed NVFP4 MoE
(1.44 ms), and mHC (1.40 ms). The native 128-thread mHC final kernel benchmarks
at about 16.9 us with 48 registers, 48 bytes of shared memory, and no spills.
The fused KDA f/g projection benchmarks at 6.28 us versus 23.02 us for two
cuBLAS calls, a 3.66x local speedup and an estimated 0.57 ms per model token.
Its real-model end-to-end rerun remains pending an uncontended eight-GPU slot.

The exact 1024-input/256-output, three-repeat speed gate and broader Chinese and
English quality gate remain open. Retain all JSON, logs, and profiles under
`/data/minimax-h3/task-cache/glm53-nvfp4-sm70-20260827`; a route-hit smoke or
short quality completion must not be reported as the 70-token/s result.

## DFlash2 Acceptance Qualification (2026-08-31)

The DFlash2 draft is `incoai/GLM-5.3-Flash-DFlash2`, with its release revision
`7d74cdd881ed7e32c31175984a67823127b66cfe` retained as the first comparison
arm. There are two deliberately separate gates. The deterministic implementation
gate is token-weighted mean acceptance length at least `4.85`; this matches the
real-checkpoint 5-shot GSM8K result reported in SGLang PR
[#36755](https://github.com/sgl-project/sglang/pull/36755). Final qualification
uses the release card's official GSM8K contract: temperature `1.0`, top-p
`0.95`, probabilistic draft sampling with standard rejection, Max reasoning,
128 sequential requests, at most 4096 new tokens, and mean per-request
completion tokens divided by verification steps at least `5.78`. A short route
smoke, first-token hit rate, four-token sample, greedy-draft sampling, or merely
passing the `4.85` localization gate is not final acceptance.

SGLang's accepted result is TP8 without pipeline parallelism. Its GLM target
captures completed outputs of layers `5,14,24,33,42` immediately before layers
`6,15,25,34,43`, contracts the four mHC streams, and runs the five-layer
non-causal DFlash2 draft. SGLang only enables capture on the last PP rank, so
it is not evidence for PP support. The TP4/PP2 V100 route therefore transports
the first two auxiliary states explicitly and replicates the TP-sharded target
embedding on PP1 for draft weight sharing.

Historical TP4/PP2 smokes produced target-correct output but only `1.0-1.6`
mean acceptance length. They are rejected evidence. The following causes have
now been excluded independently:

- exact GLM draft attention at sequence lengths 181 and 4096 matches dense and
  paged references with maximum absolute error at most `2.44140625e-4`;
- draft FC, MLP, RMSNorm, grouped convolution, selector weights, and checkpoint
  fingerprints match their reference computations;
- the SM70 mHC suite passes `22` GPU tests (one unrelated case skipped),
  including the layer-zero broadcast path and FP32 staging;
- target KDA/MoE/mHC batch-eight versus batch-one localization is nearly exact;
- a release-checkpoint load audit compared all 18 selector, codebook, FC,
  normalization, convolution, fused QKV, and fused gate-up tensors after the
  required BF16-to-FP16 conversion; every element matched the safetensors
  source exactly;
- experimental small-query metadata, sharded context FC, grouped verification,
  and selector QPN8 are all disabled in the qualification launch.

vLLM PR [#54373](https://github.com/vllm-project/vllm/pull/54373) subsequently
identified a failure with the same signature on GLM-5.3-Flash: copying the
target's RoPE layout into the draft silently collapses acceptance to `1.0`,
while taking the layout from the DFlash checkpoint restores approximately
`5.5`. The target is NoPE and its indexer RoPE does not describe the draft's
Q/K projections; the release draft omits the flag and therefore requires its
trained default, neox `True`. The target-to-draft override has been removed.
PR [#54374](https://github.com/vllm-project/vllm/pull/54374) is also carried so
a sliding-window draft cannot inherit a FlashAttention AOT split schedule for
the wrong geometry. The SM70 backend does not currently use that schedule, but
the guard keeps backend changes from reintroducing the same corruption.

The staged rerun is now complete. A second runtime failure was caused by the
compressed MRV2 indexer exposing 64-token logical pages over 576-token physical
storage while generic metadata treated every logical page as a physical block.
The SM70 route now presents the compressed storage as real 64-token virtual
pages. The retained one-request smoke accepts all seven draft tokens and no
longer indexes outside the cache.

The final quality drift was in target verification rather than the draft. B1
target decode dequantized packed E4M3 KV and used FP16 Tensor Core GEMMs, while
M2-M8 verification used a different online-softmax kernel. A layer trace found
bitwise-equal layer-0 input and KDA state, followed by the first material
difference at sparse-attention layer 3. The direct attention difference was
only about `1.5e-5`, but repeated sparse output projections amplified it enough
to reverse low-margin logits. Commit `494770be21` keeps batched KV gather, QK,
and softmax for M2-M8, then issues each PV with the same FP16 Tensor Core GEMM
arithmetic as B1. The 2048-index-width operator gate is bitwise equal to eight
B1 calls in eager and CUDA Graph replay. A fully strided-batched PV is rejected
because cuBLAS changes its reduction order once enough probabilities are
nonzero.

The retained deterministic audit is
`gsm8k60_hybrid_fp8_gemm_release_eager_tp4pp2_20260901.json`: 60 sequential
GSM8K requests, temperature zero, greedy draft, Max reasoning, 1024 output
tokens, TP4/PP2, target E4M3 KV, FP16 target dense/KDA, and every optional
DFlash fast path disabled. It reports `59/60` accuracy, zero invalid answers,
token-weighted acceptance length `5.5305`, and per-request mean/P50 acceptance
`5.7297/5.7617`. All seven per-position acceptance rates decrease normally
from `0.9123` to `0.4125`. This passes the `4.85` implementation gate and
matches the independent target-only audit's `59/60`; the only wrong question
is also wrong under target-only. Compared with the prior DFlash audit, case 16
improves from an extracted answer of 2 to 230 with no newly regressed question.

The same eager audit averages `35.14 tok/s` steady decode (P50 `35.33`, P90
`39.65`) versus `36.43 tok/s` before the arithmetic correction and target-only
`14.60 tok/s`. The exact verifier is therefore about 3.5% slower than the old
approximate verifier but remains about 2.41x target-only. Logical KV capacity
is 10,406 tokens versus 10,516 before the larger workspace. These are
quality-lane eager numbers, not the final CUDA Graph speed result.

The remaining localization surface is now captured by
`VLLM_DFLASH_DEBUG_TENSOR_DUMP_DIR`. A real request records all five target
auxiliary states, the projected context, draft embeddings, context/query slot
mappings, backbone hidden states, top-16 candidates, unary logits, the full
`7 x 16 x 16` selector lattice, and emitted tokens. PP0 sender and PP1 receiver
also save their raw auxiliary tensors per TP rank. The analyzer
`benchmarks/compare_sm70_dflash2_tensor_dump.py` reports exact sender/receiver
identity, maximum error, cosine, tensor statistics, and the greedy selector
path.

Two qualification-only A/B switches isolate the last architecture-specific
risks without changing production defaults:

- `VLLM_SM70_DFLASH2_BF16_EMULATION=0` compares ordinary FP16 draft execution
  with the default range-preserving BF16-semantics SM70 path;
- `VLLM_GLM53_PP_MHC_MATERIALIZE=1` completes `hc_post` at the PP boundary and
  sends four materialized residual streams instead of deferred `x/residual/`
  `post/comb` state. This matches SGLang's inter-layer representation and
  reduces the base mHC PP payload from approximately five hidden-state widths
  to four.

The focused closure passes 26 SM70 sparse/KDA GPU tests and 170 CPU-side
DFlash, PP, and benchmark tests (21 environment-dependent skips). Case 16
still reaches the 1024-token cap on one valid eager greedy branch, but an
independent target-only replay diverges at the identical token 58 to the same
`simple` branch. This is target greedy sensitivity rather than a DFlash-only
regression.

The final production graph audit is
`gsm8k60_hybrid_graph_fast_nopush_tp4pp2_20260901.json`. It enables CUDA Graph,
grouped verification, fused small-query and grouped metadata, sharded context
projection, verifier fast paths, and local draft argmax. The only rejected
optimization is the SM70 TP4 push all-reduce. Across the same 60 deterministic
GSM8K requests it records `59/60` accuracy, zero invalid answers, and `60/60`
natural stops. No target-only-correct question regresses; case 16 now stops at
443 tokens. Token-weighted acceptance length is `5.6119`, with per-request
mean/P50/P90 `5.6908/5.7854/6.4004`.

Steady decode averages `112.41 tok/s` (P50 `114.16`, P90 `126.32`) and aggregate
output throughput is `77.50 tok/s`, so the no-MTP TP4/PP2 production graph
exceeds the 75-token/s goal while preserving the retained target-only quality
set. The separate 1024-input/256-output three-repeat graph checkpoint averages
`154.57 tok/s`; its repeated low-entropy prompt accepts all eight target tokens,
so it is speed evidence rather than an acceptance-quality estimate.

The matched push-all-reduce graph audit is rejected: accuracy falls to `58/60`,
case 20 first diverges at output token 14 and reaches the 1024-token cap, while
the same case passes three repeated graph runs with the ordinary custom
all-reduce. GLM5 DFlash2 TP4 therefore auto-sets
`VLLM_SM70_TP4_PUSH_ALLREDUCE=0` when the variable is absent. Explicit `1`
remains diagnostic-only and emits a quality warning; other model routes retain
the global default. At that checkpoint, the official 128-request,
temperature-1.0, top-p-0.95, probabilistic-draft `5.78` gate remained open and
had to be reported separately.

### Official release-card closure (2026-09-01)

The no-MTP eight-V100 TP4/PP2 graph route initially passed isolated prompts but
failed after request-slot reuse. The scheduler could start request B while a
sampled-token receive for request A was still in flight; the old packet then
updated B's recycled request state. Dataset item 16 exposed this by changing
the correct `230` answer to `170` only after 16 earlier requests.

The retained implementation ports the V2 PP cadence and deferred slot-ring
design from upstream vLLM PR #42187. Non-last stages consume sampled output one
full pipeline traversal later on a dedicated sibling NCCL communicator and
side stream. Per-slot generation counters reject stale receives after request
free/reuse. For DFlash2, each collective packet contains the eight target
sample slots and seven next-step draft tokens; Triton scatters valid rows and
skips `-1` sentinels without a host gather. Query-position updates stay
optimistic, while sampled and rejected-token updates are applied when the ring
entry matures. The neutral Mamba update also accepts the V2 runner's native
`int32` slot indices through a GPU kernel rather than an indexing fallback.

The deterministic sequential artifact is
`gsm8k60_deterministic_graph_pp24_21_slotring_dtypefix_tp4pp2_20260901.json`.
It uses q7 greedy drafts, E4M3 target KV, CUDA Graph, and the retained `24,21`
partition. Accuracy is `59/60`, zero answers are invalid, and all 60 requests
stop naturally. The only miss is the target baseline's dataset item 12; all 60
extracted answers match the target-only deterministic audit. Item 16 again
returns `230` and its token hash matches the accepted isolated DFlash baseline.
Token-weighted acceptance length is `5.615924`; aggregate output throughput is
`75.9518 tok/s`, and mean steady decode is `109.1768 tok/s`.

The official retained artifact is
`gsm8k128_official_graph_slotring_auto_tp4pp2_20260901.json`. It uses the
release-card `zlab-shuffle42` order, no explicit request seed, temperature
`1.0`, top-p `0.95`, probabilistic q7 draft sampling, Max reasoning, a
4096-token output limit, E4M3 target KV, and CUDA Graph. The runtime itself
auto-selects `VLLM_PP_LAYER_PARTITION=24,21`, proposal temperature scale `0.8`,
and proposal top-p `0.95`; the benchmark clears manual overrides before engine
construction. All 128 requests stop naturally, zero answers are invalid, and
accuracy is `122/128` (`95.3125%`). Mean completion tokens per verification
step are `5.810047`, passing the `5.78` release-card gate. Token-weighted
acceptance length is `5.585307`, passing the `4.85` implementation gate, with
per-position acceptance decreasing normally from `0.921979` to `0.424104`.

The same official run records `77.8477 tok/s` aggregate output throughput and
`112.1869 tok/s` mean steady decode (P50 `111.3379`, P90 `127.3029`, P99
`138.8517`). Mean TPOT is `9.0196 ms`, mean prefill is `89.4121 tok/s`, and FP8
KV capacity is 18,064 logical tokens at 0.90 GPU-memory utilization.

The faster/larger-KV `25,20` alternative is not the quality default. Its
deterministic audit had one 1024-token length stop, and its official accuracy
was `120/128`, below the retained `24,21` result. Explicit mHC boundary
materialization remains rejected: the matched deterministic run fell to
`58/60`, with item 16 ending at `170`. The diagnostic switch remains available
with a quality warning.

For the exact SM70 GLM5 NVFP4, probabilistic q7 DFlash2, TP4/PP2 contract, the
runtime selects the retained `24,21` partition and proposal settings only when
they are absent. Explicit overrides remain untouched and emit a warning. Other
architectures, quantization formats, draft methods, q lengths, GPU
capabilities, and TP/PP topologies do not receive these defaults.

## TP8/PP1 Verifier Latency Qualification (2026-09-02)

The PP-free TP8 route is retained as a verifier lower bound and a focused
kernel-development lane. It uses all eight V100s, E4M3 FP8 target KV, q7
probabilistic DFlash2, CUDA Graph, FP16 target arithmetic, and no MTP. The
production TP4/PP2 quality route above remains a separate topology and must not
be inferred by multiplying or dividing the TP8 result.

The initial quality-qualified hierarchical-push endpoint took `33.2440 ms` per
q8 verification round. Exact fixed-shape cuBLASLt KDA projections reduced it to
`32.7333 ms`; fusing the four GDN metadata groups reduced it to `30.1857 ms`.
The final native q8 mHC post+dot operator reaches `29.8732 ms` on the seed-zero
endpoint and `29.9130 ms` across 74 rounds and three seeds. With all flags
selected by source defaults, the same 74-round workload measures
`29.8868 ms/round` and `172.2715 tok/s` weighted pure decode.

The target top-p pass now uses eight warps only for the exact SM70 GLM5 shape:
batch 8, vocabulary 154,880, top-k disabled, and top-p enabled. A
candidate/control/candidate sandwich measures `29.8475/30.2105/30.0496 ms`
per round. The candidate mean is `29.9486 ms`, `0.2620 ms` (0.87%) below the
four-warp rollback. A same-seed node trace preserves all 128 output tokens,
the token hash, and all 23 verification steps while reducing
`_topk_topp_kernel` from `303.237 us` to `207.963 us` per round. One candidate
trace step contains 4.8 ms of rank-start skew, so graph-node timing remains
diagnostic rather than the endpoint acceptance number.

The accepted mHC arithmetic is intentionally strict. A first native kernel
used explicit `fmaf` operations and differed from the staged TileLang path by
one ULP in seven of 131,072 FP16 residual elements; that changed a generated
token hash and is rejected. Rewriting the source expressions in the same
accumulation order as the staged kernel produces identical FFMA instructions
while restoring bitwise equality for residual, mapped, residual output,
squared sum, dot, and hidden output. The focused V100 test reports five passed
cases, including CUDA Graph replay.

The retained proposal calibration is temperature scale `0.9` with proposal
top-p `0.95`. On the official 128-request contract it records `124/128`
accuracy, zero invalid answers, 128 natural stops, `5.787283` mean completion
tokens per verification step, and `5.573722` token-weighted acceptance. A
second fixed-seed 128-request audit records `122/128`, zero invalid answers,
128 natural stops, `5.753127` per-request acceptance, and `5.602134`
token-weighted acceptance. The `0.8/0.95` alternative is rejected because the
same fixed seed produced one invalid length-capped response.

The complete post-sampler-change audit again records `124/128`, zero invalid
answers, and 128 natural stops. Mean completion tokens per verification step
are `5.784293`, token-weighted acceptance is `5.561909`, and both release gates
pass. Acceptance min/P50/P90/P99/max are unchanged. Mean per-request steady
decode is `190.320 tok/s`, weighted pure decode is `180.008 tok/s`, aggregate
output throughput is `106.575 tok/s`, and mean TPOT is `5.4041 ms`. Dividing
the full audit's total decode time by its 6,873 verification steps gives
`30.8020 ms/round`; that long-output workload is deliberately reported
separately from the short steady-shape `29.9486 ms` result.

The official run averages `187.4022 tok/s` steady decode with P50/P90/P99
`189.8949/218.3878/233.8120 tok/s`; aggregate output throughput is
`101.2123 tok/s`. Mean TPOT is `5.4907 ms`, mean prefill is `72.2387 tok/s`,
and the runtime exposes 34,071 logical KV tokens with 1.7 GiB available per
rank. GPU sampling during steady generation reports 100% utilization on all
eight V100s and approximately 38-39% memory-controller utilization.

The runtime only auto-enables this set for SM70, GLM5 ModelOpt NVFP4, FP16,
probabilistic q7 DFlash2, TP8/PP1, and no DBO. It preserves explicit overrides,
keeps sparse target rejection and MoE QPN W13 disabled, enables the exact
hierarchical push collective, cuBLASLt KDA, grouped expert rows, fused GDN
metadata, and fused q8 mHC, and uses regular `torch.compile`. The matched AOT
compile path is rejected at `30.98 ms/round` despite an exact output prefix.

Faster-looking alternatives remain excluded. The upstream historical
single-pass top-p implementation changes 703 mask values on a random
GLM-shaped oracle, non-default tile sizes change the mask, and an eight-warp
rejection-statistics schedule changes the accepted-token chain. Compact target
rejection is not valid for this workload because the official target uses
`top_k=-1`; enabling top-k 20 would change the target distribution.

### TP8 fused KDA f_b/g_b closure (2026-09-02)

The fixed-shape native KDA projection now also supports the TP8 output width:
`B=1..8`, `N=1024`, and `K=128`. The existing TP4 `N=2048` specialization is
unchanged. On the TP8 q8 operator benchmark, replacing the two FP16 linear
launches with one native kernel reduces CUDA Graph service from `6.218 us` to
`4.264 us` per layer (`1.458x`). The native output is CUDA Graph stable; versus
the retained FP16 linear path, f/g differ in 8/10 of 8,192 elements with
maximum absolute errors `1.526e-5` and `7.63e-6` respectively.

After a full 128-token warmup, a matched ten-seed candidate/control pair
measures `29.6850/29.9173 ms` per verification round. The fusion saves
`0.2323 ms` (`0.78%`); dropping the first three requests still gives
`29.7347/29.9182 ms`, a `0.1834 ms` gain. Short 32-token warmups are not valid
for this comparison because lazy kernel loading produced one-time 31-56 ms
request outliers.

The matched seed-zero node traces preserve the output hash and all 23
verification steps. Across 24 captured replay groups and eight ranks, the
fusion removes 13,056 launches of the CUTLASS FP16 `32x32x64 TN` family and
adds 6,528 native launches: exactly 68 removed and 34 added per rank/round.
The removed launches cost `78.562 ms` TP GPU-sum and the native launches cost
`33.705 ms`, a net `0.2336 ms` per-rank round reduction. This matches the
low-overhead endpoint A/B; the previously suspected `16x16x128` family is not
the f_b/g_b projection.

The full no-request-seed 128-question audit passes the release gates with
`123/128` accuracy (`96.09375%`), zero invalid answers, and 128 natural stops.
Mean completion tokens per verification step are `5.827398`, and
token-weighted acceptance is `5.585305`. Mean per-request steady decode is
`196.012 tok/s`, weighted pure decode is `187.716 tok/s`, aggregate output
throughput is `99.039 tok/s`, and mean TPOT is `5.1581 ms`. Total decode time
divided by all 6,805 verification steps is `29.6639 ms/round`; stochastic
output-length and context differences mean this long audit is quality evidence,
not the matched `0.2323 ms` kernel speed claim.

`VLLM_SM70_GLM53_TP8_FUSED_FG_B` remains globally off. The runtime enables it
only inside the audited SM70, GLM-5.3 ModelOpt NVFP4, FP16, probabilistic q7
DFlash2, TP8/PP1, no-DBO contract and preserves explicit rollback value `0`.
