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
that view's CPU validation does not qualify DeepSeek execution. The separate
[mixed MLA stage](kv_mla_mixed_20261012.md) records its actual bindings.

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

## Frozen component cost comparison

`benchmark_sm70_mla_codec_cost.py` reproduces the fixed seed/workload and
measurement protocol: 4096 cached tokens, 2048 selected tokens, page64,
16 heads and 512 latent dimensions. Each process runs 30 eager warmups,
2000 graph warmups and nine samples of 1000 graph replays. Two independent
FP8 controls precede a frozen noise envelope; INT8/FP8 then run in the order
INT8, INT8, FP8. Output hashes repeat within each codec. This is component
timing with dynamic group scales, not a calibrated model-quality baseline.

| Case | INT8 / two-control FP8 median | Frozen decision |
| --- | --- | --- |
| Writer q1 | 1.0473 | Failed: one candidate exceeds the 6.1932% control envelope |
| Writer q32 | 1.0381 | Inside the 8.4769% envelope |
| Paged q1 | 0.8144 | Both candidates and the final control satisfy the frozen envelope |
| Paged q9 | 0.8393 | Inconclusive: final FP8 control drifts beyond the 0.02331% envelope |
| Gather/GEMM q1 | 0.9616 | Both candidates and the final control satisfy the frozen envelope |
| Batched GEMM q8 | 0.9844 | Both candidates and the final control satisfy the frozen envelope |

The envelope is the maximum relative deviation of the pre-candidate control
process/sample medians. Each candidate is judged separately; averaging does
not erase the failed writer sample. No limit is enlarged after results.
Receipt SHA256 values for control0/control1/candidate0/candidate1/control2 are
`0843672c450f39732cbae79364dcfe68c7aa99734aea0ed90e755179daf2913a`,
`76d2734b6e6659371ccab31d9a9ccd1536dd469256f01a8fe91041b3aa03d378`,
`3cd2cb1d7846a07e24b8eb5f43d025f73593b2d14db177787480305cfb8f8704`,
`c96cc5f2899c2fafad1b31d61eff465a8901e8d91c45335fa68770c40f5852cf`,
`3a41bc98dce7c453f9e951100d51639e24e1847657e2c4708c449861c00719bb`.
The aggregate latency gate remains open.

## Remaining engine gates

R05's bounded codec/executor binding is implemented. Representative GLM engine
quality, state lifecycle, normal whole-package initialization and latency are
pending; R05 remains experimental. Generic Triton MLA and the other inventory
components require their own execution evidence. No global MLA or end-to-end speed claim follows from
this stage.
