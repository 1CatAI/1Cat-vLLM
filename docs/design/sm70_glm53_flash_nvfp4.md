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
