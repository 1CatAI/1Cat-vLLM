# Packed GGUF PLE rows

Qwen4Exp GGUF stores its n-gram embedding as a packed IQ4_NL row table.
The table must remain mmap-backed on the host; decoding the full table would
expand storage from about 28.8 GB to about 102.4 GB. Decode only requested
rows and keep the existing PLE transport, hashing, scheduling and hybrid
placement policy.

The CPU offload loader must admit GGUF and filter tensors before decoding
unrelated projections. The row reader must validate logical row width, type,
indices and target dtype. Resident GPU rows and returned host rows must use
the same dequantization contract. No quantization check should disable hybrid
PLE in model configuration.

This layer depends on the Qwen4Exp adapter in #876. Validation will cover
packed mmap storage identity, selected-row official dequantization, repeated
IDs, invalid IDs, n-gram ordering, transport dtype and graph replay. Model
quality and latency follow in the Flash-Next TP4 integration.

Integration base: `3e42d7c753d9ad094d5eaf2895cb2cd4a22d8108`.

The initial selected-row reader keeps its original NumPy mmap view, gathers
unique requested IDs, applies the official GGUF reader dequantization and
restores repeated-ID order. Five CPU cases check exact FP16/FP32 outputs,
identity of mmap storage, empty/invalid indices, malformed rows and FP16
range rejection without clipping. The filtered loader iterator removes mapped
names before entering the adapter; the PLE worker admits this loader alongside
the existing default and dummy loaders.

## Installed row and transport checks

The normal wheel contains the row reader, GGUF loader filter, PLE registration
and hybrid row gather. All nine CPU reader/module/transport checks pass; all
three V100 GPU cases pass for device-only, host-only and split device/host/disk
placement. GPU outputs match official IQ4_NL dequantization exactly on the
controlled rows. CUDA graph replay with changed IDs and remote rows preserves
ordering and values. CPU-owned table storage retains the original mmap pointer.

The first cascade transport test exposed a NumPy/Torch mask mismatch; it now
passes using the shared Torch mask helper. The test and failed result are
retained. CUDA 12.8, Torch 2.10.0+cu128, V100 SM70, FP16 row outputs; no private
extension or preload is used. These are layer checks, not Flash-Next model
correctness or throughput evidence. The full CPU offload process and model
integration still need validation.

## Cascade capability checks

The configuration capability accepts file-backed GGUF shards in addition to
safetensors. Its embedding-method check admits packed GGUF rows without an
E4M3 storage hint and retains the existing dtype, worker topology, PP layer
layout and explicit-placement guards. Missing checkpoint storage information
still rejects the route with a reason. All 21 source configuration checks
pass from source and the normal combined wheel, including GGUF activation and
unsupported embedding metadata. The actual Flash-Next PLE worker starts and
loads all five required checkpoint entries; two materialized parameters pass
its completeness check. Whole-model generation and remote request handling
remain pending.

Selected first, middle, final and repeated rows from the real 320,001,536-row
IQ4_NL table produce finite values without copying the table. The packed view
shares its checkpoint mmap, and FP16 results equal rounded FP32 values. The
maximum absolute FP16 rounding error on those six selected rows is 1.12057e-5.
This is a storage-boundary check, not a full-table numerical measurement.
