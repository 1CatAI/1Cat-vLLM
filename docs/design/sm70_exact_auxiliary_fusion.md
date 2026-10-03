# Exact FP16 auxiliary projection fusion

The SM70 auxiliary fusion combines three dense projections of the same
single-token input in one Triton launch. It keeps the existing FP32 FMA and
reduction order, with FP32 compressor outputs and an FP16 indexer projection.
Projection row counts come from the loaded tensors; the kernel supports common
positive input widths aligned to its 1024-element reduction blocks.

`kernel_config.fused_fp16_aux_gemv` defaults to `true`. Set it to `false` to
retain separate launches. Fusion requires all three projections to have already
selected the existing exact FP16 GEMV route. It does not replace the default
FP13 route or a cuBLAS projection, and it adds no environment variable.

Preparation runs after checkpoint loading and stores a nonpersistent joined
weight buffer. Its additional memory is the total size of the three FP16
weights. Unsupported dtype, layout, device or individual projection routes
retain the original implementation. The startup acceleration report includes
the preparation flag, rejection reason and resident packed-buffer size.
