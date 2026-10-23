# Packed Qwen3.5 GGUF vocabulary tables

Qwen3.5 constructed an ordinary FP16 vocabulary table, so its GGUF adapter
expanded quantized input embeddings before TP loading. Pass the GGUF
quantization configuration to the embedding layer and load the original
GGML bytes instead. `GGUFEmbeddingMethod` gathers the selected local rows,
decodes them to the model dtype, and retains the existing vocabulary masking
and TP reduction.

Floating tables retain their dense method. Tied input/output tables also
retain the dense method so this change does not introduce activation
quantization into their output projection. DFlash and Qwen4Exp adapters retain
their existing table-loading contracts. A DFlash checkpoint without its own
embedding shares the loaded target module through the existing sharing path.

Loading still checks FP16 overflow against official GGUF dequantization in
1024-row chunks. It keeps a packed table and bounded decode temporaries instead
of allocating a complete dense table.

## Storage

The Qwen3.8-27B-GSQ-RCO-IQ3_S checkpoint has an IQ2_S input embedding with
248320 rows and width 5120:

| Storage | Whole table, bytes | TP2 rank, bytes |
| --- | ---: | ---: |
| GGML IQ2_S | 407244800 | 203622400 |
| Expanded FP16 | 2542796800 | 1271398400 |
| Difference | 2135552000 | 1067776000 |

The saving is approximately 0.994 GiB per TP2 rank and 0.497 GiB per TP4 rank,
before the one-byte quantization-type parameter. The decoded FP16 values are
unchanged.

## Validation

The complete wheel contains the Python implementation and the existing shipped
GGUF row decoders. Native artifacts are unchanged from `753ae2bca8`; no external
kernel library or runtime source overlay is required.

- 56 installed adapter/cache checks pass, including packed bytes, floating and
  tied tables, overflow rejection, and unchanged DFlash/Qwen4Exp contracts.
- Four SM70 GPU tests pass for Q8_0/IQ2_S and TP2/TP4 row ownership. Selected
  rows match official dequantization rounded to FP16 exactly. Changed-token
  CUDA Graph replay matches exactly as well.
- A real TP2 model retains the same arithmetic and unit-test natural answers
  with normal EOS. DFlash2 remains enabled and shares the packed embedding.
- The combined cache-page and embedding changes serve real 128K and 261120-token
  retrieval requests, followed by 262080 input plus 64 output tokens. The exact
  262144-token boundary completes with CUDA Graph enabled.

See `gguf_tp2_256k_memory.md` for the complete workload and memory accounting.
