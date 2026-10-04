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

These checks cover the CPU reader only. Hybrid resident-table loading, row
transport and GPU replay integration are not complete yet.
