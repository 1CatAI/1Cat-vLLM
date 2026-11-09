# INT8-G64 KV codec design handoff

Status: proposal, not an implemented or qualified format. This updates PR #1064
against main `18784e027`; former stacked native-codec dependencies are not
assumed merged. Python `KV_CODECS` contains FP16, BF16, E4M3 and E5M2, not
`Int8G64Codec`.

## Reusable contracts and remaining work

Existing codec admission, metadata and executors are reusable extension points.
A descriptor does not enable an operator. G64 still needs defined payload/scale
layout, writer and reader semantics, byte accounting, native bindings, capability
probes and qualified fallbacks. Reference conversion currently accepts a scalar
scale; group scales need an explicit contract. Do not assume a shared CUDA traits
API exists merely because an earlier design branch proposed it.

Keep proposed `int8_g64` distinct from `int8_per_token_head`: one scale per 64
channels differs from one scale per token/head. Separate signed-int8 K/V payloads
and FP16 group scales are a possible design, not an accepted allocation or ABI.
Inline scales, channel tails, padding, zero groups and saturation need explicit
physical strides and rounding rules before numerical qualification.

| Family | Reusable boundary | Required qualification |
| --- | --- | --- |
| Scalar/XQA decode | Shape checks and execution scheduling | Reader, scale addressing, writer, binding, loaded capability and output accuracy |
| Grouped verify / mixed rows | Metadata and grouped admission | Feature-specific row/scale layout, accepted-token semantics and supported shapes |
| FP16 dense / Split-D prefill | Arithmetic after explicit conversion | Group-aware conversion workspace and declared/counted fallback; no inherited direct INT8 support |
| BF16 / specialized features | Applicable common contracts | Independent dtype, layout, numerical and feature admission |

Qualify one family first, then extend existing declarations/bindings while
preserving labels. Test writer/reader boundaries on CPU, then native output,
non-contiguous pages, scale indexing and changed-input graph replay. Static
matrices do not establish native support or speed.

See the [codec contract](../../../vllm/v1/attention/kv_codecs.README.md) and
[architecture overview](README.md). INT8 implementation remains outside E.
