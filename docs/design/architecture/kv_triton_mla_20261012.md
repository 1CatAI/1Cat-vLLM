# Shared concatenated MLA codecs

The existing `TRITON_MLA` provider, native concatenation/prefix gather and
RoPE/cache fusion bind FP16, BF16, E4M3 and INT8 through `KVCacheView`.
Page traversal, split allocation, masks, softmax and partial reduction remain
shared. This stage introduces no model, weight or TP whitelist and does not
replace the grouped or mixed MLA mathematical representations.

## Storage and execution contract

The concatenated representation stores latent and RoPE dimensions together.
FP8 retains its tensor FP32 scale and existing on-load scaling/activation cast.
INT8 uses one dynamically computed FP32 scale per complete token/head row,
nearest-even rounding and the existing symmetric scale floor. Both fields are
quantized here; the mixed MLA codec continues to retain unquantized BF16 RoPE.
A page contains contiguous payload rows followed by FP32 scales in the same
physical owner. Its slot accounting is D+4 bytes, with actual page strides and
padding supplied by the view; ordinary KV remains unchanged.

The common MLA layer previously treated every quantized format as FP8,
reinterpreting INT8 bytes and admitting FP8 query quantization. FP8 operations
now explicitly require FP8, while INT8 keeps its cache representation and
unquantized queries. Existing scaled-MLA DCP rejection remains explicit.

Native writers share the whole-row encoder/reduction. Prefix gather shares
FP8's vector traversal/load/store loop with a compile-time row-scale binding.
RoPE fusion retains the existing rotation and scheduling, then encodes the
completed row through that same encoder. The normal native ABI probe is 2;
ABI 1 can supply unfused writer/gather, and missing fusion capability rejects
pattern replacement before Q/K or cache mutation. Old tensor-scale calls keep
their optional-argument compatibility.

Triton loads E4M3 through the shared software byte decoder on SM70, avoiding
unsupported native FP8 pointer types. INT8 applies its row scale after QK and
inside PV using FP32 accumulation and high/low partials; the softmax denominator
remains unweighted. Invalid page IDs and empty partitions are masked through
the common implementation. Actual page strides replace implicit dense-page
addressing, preserving hybrid-owner padding.

## Bounded evidence

Platform: V100 SXM2/SM70, CUDA 12.8.93, Torch 2.10.0+cu128.
The normal main-package CMake stable component was rebuilt and installed;
its final component SHA256 is
`e343474055e5d05edd4a9803723b3be5d6ad852f41702183b2a1ecb20245ad6c`.
Source/component qualification uses the owned staged source tree. A clean
complete wheel and representative engine are still separate open gates.

The component records 68 distinct passing GPU cases in three bounded runs:

- R3: 48 cases for FP16/E4M3/INT8 native writers and prefix gather,
  actual MLA MHA/grouped decode, fused RoPE writers and changed-input graphs.
- R4: 16 additional cases preserving the explicit BF16 cache configuration.
- R5: four BF16-query cases through the actual grouped MLA implementation.

Cases cover page16/64, D288/320/576 as applicable, H1/16, non-contiguous pages,
physical padding, negative/capacity-invalid slots, empty rows and graph replay.
Native prefix gather retains its pre-existing D320/576 geometry restriction;
a D288 decode case does not establish D288 prefix-gather support. Attention
limits remain `atol=0.004, rtol=0.03`; cache bytes, scales, skipped slots and
padding are checked exactly. References independently reconstruct stored
values and calculate attention in FP64. These cases do not prove all shapes.

The common-entry regression is reproduced on parent `b4b6c47992`: FP16 and
FP8 pass, INT8 fails the cache-dtype assertion. The repaired component passes
71 focused CPU checks, including representation/page accounting, actual
forward argument propagation and missing-ABI rejection before mutation.
Eight existing MHA/GQA tests also pass, covering dense and raw FP8 inputs
with page1/16 through the affected common wrappers. Eight offline release
artifact contract checks pass; they do not inspect the final complete wheel.
An older SM70 FP8 Triton reproducer fails compilation on the parent because
its FP8 pointer is unsupported. Thus the prior format declaration is not
reported as actual FP8 execution evidence on V100.

R2's first GPU attempt retained six passing legacy-format cases and stopped
on the new no-tensor ABI probe's incorrect CUDA-only registration. Registering
the probe through CompositeExplicitAutograd fixes normal dispatch; R3 exercises
it through the real writer/gather/fusion calls. Numerical bounds were not changed.

## Additional valid-partition gap

A subsequent 8192-token single-partition case exposes an INT8 weighted-Half
cast failure outside the initial short-context checks. Its unchanged limits,
shared binary-rescaling repair and finite-value/graph boundary checks are in
[weighted-partial qualification](kv_mla_weighted_tail_20261012.md). The initial
68-case result does not cover that gap.

## Open gates

This supplies R07 component bindings, not a final release or performance claim.
Representative engine startup, common-prefix execution, cold/warm compilation,
state/rollback, offload/transfer, quality and interleaved latency remain open.
The independent P05 late-scale prefill performance failure and the broader
43-item inventory are not closed by these results.
