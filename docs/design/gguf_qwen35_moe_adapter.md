# Qwen3.5 MoE GGUF adapter

Select the native adapter for `qwen3_5_moe_text`, including text configurations
nested in a multimodal HF config. Map router, shared-expert and stacked expert
names directly to vLLM parameters. Reuse the dense adapter's GDN inverse
transformations, norm offsets, convolution layout and target/nextn separation.

The stacked expert transport is shared with Qwen4Exp. Gate, up and down keep
independent source type descriptors. Quantized payloads are expert views of the
original bytes, preserving mmap storage; floating expert payloads convert to
the requested dtype with an explicit finite-to-infinite overflow check. TP
boundaries remain the canonical expert bank's responsibility. This adapter
does not substitute expert parallelism for tensor parallelism.

Name rules follow the Apache-2.0 vllm-gguf-plugin Qwen3.5 adapter at
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`. Source attribution is retained in the
adapter. No additional CUDA kernel or precision change is introduced.

## Validation

A normal whole-wheel package in an independent Python 3.12 environment passes
25 adapter checks. Mixed Q4_0/Q8_0/F32 expert projections reproduce official
GGUF dequantization after the declared FP16 conversion, preserve quantized
byte storage, and emit separate descriptors for all three experts. Invalid
expert counts and FP16 overflow are rejected. Existing dense and Flash-Next
adapter checks pass after the common transport extraction.

Four relevant adapter modules match source, wheel and installed contents.
All 210 installed dependencies pass compatibility checking. Core and FA2
fingerprints remain unchanged. These results cover adapter behavior only;
35B full-model numerical distributions, quality and FULL decode graph
performance remain to be measured before promotion.
