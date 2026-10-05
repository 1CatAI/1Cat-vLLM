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

The normal CUDA12.8 SM70 extension and complete wheel build pass. The
new composition uses 64 registers, 33,792 shared bytes, zero stack and
zero local memory. At stable secondary 1530/877MHz and 300W, M8 actual
TP4 shards, 16MiB cold-L2 graph ABBA, layer26 native53.248–54.272us
versus canonical75.264–75.776us; layers27/34 native54.272us versus
canonical75.776us. Original-payload bandwidth is 375.3–382.5GB/s.
Relative L2 versus official FP32 GEMM is 0.000488–0.000523, compared
with canonical0.000587–0.000624. Runtime M512/8/1/5/16/20/32/8 and
graph checks pass; non-M8 outputs remain bitwise canonical. Primary
fixed-clock comparison is pending before model admission is merged.
