# Shared grouped MLA cache qualification

The existing GLM NoPE paged, gather/GEMM and batched GEMM executors now bind
grouped E4M3 and INT8 through the common codec/view. Their page traversal,
selection masks, softmax, reductions and bounded workspace owners remain
shared. Existing FP8 function names remain compatibility aliases; public
codec-neutral names reference the same implementations.

## Representation and allocation

`KVCodec` declares `mla_latent` and `mla_mixed` separately from ordinary,
history and indexer storage. Latent dimensions are grouped by 64; mixed
storage also declares unquantized RoPE. FP8 uses its existing UE8M0 scale
encoding and FP16 scale application. INT8 uses signed bytes, RN FP32 scale
division, nearest-even encoding and FP32 group scales. Ordinary INT8 retains
token/head scales and its planar K/V layout.

`MLAPageLayout` specifies group count, padded scale slots and data/scale bytes.
`KVCacheView.bind_mla` borrows payload, scales and RoPE from a single allocation,
checks representation, dtype, capacity and alignment, and preserves physical
page padding. The `MLAAttentionSpec` stores the provider's layout and uses it
for exact accounting, independent of model names. A layer's effective cache
format is retained when the outer target/draft configuration differs.

The qualified NoPE representation has 512 latent dimensions: 520 bytes per
token with eight UE8M0 scales, or 544 bytes with eight FP32 INT8 scales. The
mixed representation view also describes 448 latent and 64 BF16 RoPE fields;
that view's CPU validation does not qualify DeepSeek execution.

## GPU component evidence

Platform: NVIDIA V100 SXM2, compute capability 7.0; CUDA 12.8.93,
Torch 2.10.0+cu128. The tests run in an isolated source/component stage with
normal installed parent native components and the changed Triton source.
This is explicitly not a final clean whole-wheel installation.

`gpu-mla-latent-r3.log` records **34 passing checks**: 24 generated GPU
writer/attention checks and ten existing GLM regressions (six GPU, four CPU).
Writer cases cover page16/64, owner padding, negative and capacity-invalid
slots, changed-input graph replay, zero rows and independent exact payload/
scale reconstruction. Untouched payload/scale rows and physical padding stay
unchanged. Attention cases cover paged q1/q9, gather/GEMM q1 and batched GEMM
q8, non-contiguous slot order, empty selections and changed-input replay.
The existing MLA `atol=0.004, rtol=0.03` limits are unchanged.

The first GPU run exposed unsupported ellipsis indexing in the new common
Triton encoder. The second retained eight failures in the reference decoder:
its FP64 output limit overflowed an intermediate FP32 clamp. Both are fixed
without changing numerical limits. Existing FP8 component tests pass.

After stacking on the QSA page-view stage, 62 focused CPU checks pass across
MLA representations/specs, ordinary/history/indexer identities and owner-page
mapping. These CPU checks do not substitute for GPU or engine evidence.

## Remaining gates

R05's bounded codec/executor binding is implemented. Representative GLM engine
quality, state lifecycle, normal whole-package initialization and latency are
pending; R05 remains experimental. DeepSeek mixed MLA writers/readers/fusion,
generic Triton MLA and the other inventory components require their own
execution evidence. No global MLA or end-to-end speed claim follows from
this stage.
