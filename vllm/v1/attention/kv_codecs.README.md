# KV-cache codec contract

`kv_codecs.py` owns Python KV storage identity and reference dequantization.
It does not describe weight quantization, select attention algorithms or allocate
cache/workspace resources. `resolve_kv_codec` resolves aliases;
`canonical_kv_cache_dtype` preserves unknown names for existing callers.

## Storage and numerical semantics

`KV_CODECS` declares FP16 (`auto`, `float16`), BF16, E4M3 (`fp8`, `fp8_e4m3`)
and E5M2. FP8 payload tensors use `uint8` storage reinterpreted as the declared
FP8 element type; reference conversion casts then applies a scalar scale.
Callers supply independent K/V scales. Unquantized reference conversion returns
the original cache.

The descriptor is immutable. Cache allocation, page/stride contracts, persistent
metadata, capture addresses and workspace lifetime belong to the cache manager,
metadata builder and attention resource owners, not the codec.

## Admission, execution and fallback

The [Flash-V100 component](backends/flash_v100/README.md) combines codec identity
with shape restrictions, actual operator availability, native ABI and layout/scale
contracts. Codec membership alone never proves native support. A conversion
bridge describes input storage separately from the downstream FP16 arithmetic.
Unsupported combinations retain explicit rejection or the existing fallback;
do not alias new formats to old ones just because their storage width matches.

Published aliases and historical counters remain compatible. Add formats here,
then extend the relevant existing family binding and admission; do not copy
execution flows. The [INT8-G64 handoff](../../../docs/design/architecture/int8_g64_codec.md)
is a proposal with outstanding writer, scale-layout, accounting and native
binding work, not another member of `KV_CODECS`.

## Minimal validation

From the repository root:

```bash
.venv/bin/python -m pytest -q tests/v1/attention/test_kv_codecs.py \
  tests/v1/attention/test_flash_v100_routes.py
```

These CPU contracts cover aliases, storage/reference conversion and declared
admission. Native formats additionally need writer/reader numerical tests,
boundary layouts and capture/replay on supported hardware. Static coverage does
not substitute for GPU evidence.
