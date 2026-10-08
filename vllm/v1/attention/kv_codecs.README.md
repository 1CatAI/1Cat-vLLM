# KV storage codecs

`kv_codecs.py` owns format identity, aliases, storage dtype and reference
dequantization for FP16, BF16, FP8 E4M3 and FP8 E5M2. `fp8` resolves to E4M3.
Codecs describe KV storage independently of model weight quantization.

Routes admit codec objects. Native SM70 scalar/vector readers and storage
traits live in `flash-attention-v100/kernel/{kv_codec,kv_codec_traits}.cuh`;
they currently implement FP16/E4M3/E5M2. A Python codec alone does not imply a
native reader, allocation layout, scale ABI or support by every route.

The current reference method takes one scalar scale. INT8-G64 requires
group-scale storage and a shared reader/launch context. See the
[INT8-G64 handoff](../../../docs/design/architecture/int8_g64_codec.md)
for proposed layout, required integration seams and route reuse limits.
See [CUDA traits](../../../docs/design/architecture/kv_codec_traits.md)
for existing signatures, compatibility includes and bridge fallback policy.

`int8_g64` is a proposed distinct format, not an alias of the existing
`int8_per_token_head` quantization mode. No INT8-G64 codec or kernel is
implemented by the architecture handoff.
