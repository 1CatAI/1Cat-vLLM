# Original IQ3_XXS/IQ4_XS gated pairs

Layers 26, 27 and 34 share the IQ3_XXS gate and IQ4_XS up combination at
TP4 N4352/K5120. Their original payload is 20,367,360 bytes per rank and
layer, and together they account for 8.174% of mixed gate/up bytes.

Instantiate the existing two-reader shared-activation kernel for this
orientation. Source-sized IQ3_XXS records, original FP32 scale products,
shared FP32 IQ4 lookup, final FP16 operands, FP32 dot/reduction and gated
epilogue are unchanged. No second decoder or weight representation is
introduced. Model admission remains canonical until actual-weight
numerical checks and matched cold-L2 graph measurements pass.

The focused benchmark adds a prototype selector for these three real
layers. It checks official GGUF FP32 dequantization, runtime-M graph
fallback and fixed-clock ABBA. No end-to-end speed result is claimed.
