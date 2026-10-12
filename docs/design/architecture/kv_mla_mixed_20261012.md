# Shared mixed MLA execution

The existing SM70 mixed MLA paged, split-K, QK-D split and gather/BMM
executors bind grouped FP8 and INT8 through the same cache view and compiled
codec functions. Traversal, sink/mask semantics, softmax, partial reduction,
selection order and bounded workspace owners remain shared. Historical FP8
function names remain compatibility aliases.

## Representation and engine allocation

The representation contains 448 G64 latent dimensions and 64 unquantized
BF16 RoPE dimensions. The FP8 layout remains 576 payload bytes and eight
UE8M0 scale slots per token, or 584 total bytes. INT8 uses the same payload
fields with eight FP32 scale slots, or 608 total bytes; only seven scales are
active, and writers clear the padded eighth slot. Ordinary INT8 retains its
token/head scales. Indexer storage remains independently configured G128.

`KVCacheView.bind_mla` supports the existing block-major owner and flattened
gather pages with explicit token capacity. Data, scales and RoPE borrow the
same physical allocation and preserve its page padding. The existing main
and sliding-window MLA specs carry `MLAPageLayout`; allocation, compression
ratio and merging use those bytes, with incompatible layouts rejected.
The SM70 backend supplies its layout by representation geometry. INT8 on an
unbound provider is rejected before cache use; no new model or weight
qualification is introduced.

The common encoder retains each FP8 writer's division or reciprocal order.
INT8 uses RN FP32 division, nearest-even rounding and FP32 group scales.
Readers preserve existing FP8 half-first or FP32-first scaling as appropriate
to each executor. QK-D tiles index scales by latent group; tile32/128 no
longer assumes that tile identity equals a 64-wide scale group.

## Actual component execution

Platform: NVIDIA V100 SXM2/SM70, CUDA 12.8.93, Torch 2.10.0+cu128.
The isolated source/component stage uses normally installed parent native
components and changed Triton source. It does not qualify a clean whole-wheel
installation or representative model engine.

R2 records **88 passing GPU checks**, repeated in R3 after isolating Triton
compiler caches and finalizing the shared decoder and allocation checks:

- Eight exact mixed-cache writer/gather cases, including page64/128,
  owner padding, FP16/BF16 input, negative/capacity-invalid slots, unchanged
  regions, zero rows and changed-input CUDA Graph replay.
- Thirty-two generated algorithm cases: paged q1/q7, split-K q8,
  QK-D split q8 with tile32/64/128, BMM q1/q8, each codec and optional extra
  cache. Cases include non-contiguous selection, invalid indexes, empty rows,
  varying group amplitudes and poisoned unselected RoPE fields.
- Eight existing C4/C128 compression writer cases across mixed MLA and
  independent indexer codecs, plus two qualified QNorm/RoPE fusion cases.
  Graph replay and untouched/padded fields are checked; fusion storage and
  Q outputs retain their explicit byte-exact compatibility contract.
- Twenty-four GLM grouped-codec regressions, twelve existing FP8 BMM
  regressions and two existing QNorm/RoPE regressions.

Attention limits remain `atol=0.004, rtol=0.03`; BMM retains its existing
`atol=0.002, rtol=0.002`. References independently reconstruct stored latent
and RoPE fields and compute the selected attention in FP64. No failed limit
is widened. Fifty focused CPU checks pass, including actual SWA construction,
compressed cache specs, per-layer formats, layout incompatibility and existing
SM70 routes. One policy fixture previously failed on the parent because it
implicitly requested a CUDA device on a CPU host; its explicit CPU device now
tests the unchanged engine policy.

## Open gates

These bindings cover the existing mixed representation's SM70 component
algorithms. Generic Triton MLA is separate R07 work. Representative engine
initialization, quality, compression/indexer selection, request/graph state,
commit/rollback, cache transport, clean installation and latency remain open.
The prior GLM cost comparison retains failed/inconclusive cases in
[latent MLA evidence](kv_mla_latent_20261012.md). This stage makes no combined
MLA or end-to-end performance claim.
