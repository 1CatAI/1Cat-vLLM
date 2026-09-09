# Shared SM70 projection epilogues

This development branch is stacked on the common prepared execution (#571)
and workflow accounting (#578) branches. No configuration has passed the
campaign's >80 useful TFLOP/s/card and complete official quality gates.

## Measured problem and implementation

The matching four-step FA denoise profile spends 6.890 seconds in miscellaneous
elementwise kernels, including FP32 row-scale restoration, adapter additions
and output casts. Attention, GEMM and communication separately consume
31.439, 16.041 and 8.453 seconds. These are profiled service diagnostics;
the unprofiled audited baseline is 47.091839–47.091855 useful TFLOP/s/card.

`sm70_diffusion.fp16_linear_add` prepares the projection input at the existing
FP16 boundary and keeps GEMM accumulation in FP32. A shared CUDA epilogue
restores row scales with a rounded FP32 multiplication, adds the scaled delta
with the same FP32 fused multiply-add as PyTorch, and stores FP16 or FP32.
It writes only the requested output slice. Delta/output and scale/output
storage overlap is rejected, including differently typed views of one buffer.

The H3 adapter uses this interface when output hardware, precision and layout
support it and adapter slices are disjoint. Overlapping contributions retain
the original FP32 buffer until the final cast. Wheels without the new extension
ABI retain the ordinary path. No quantization label or adapter name controls
dispatch. The first LoRA projection still uses unrotated inputs; row-parallel
increments still join the partial result before the collective.

## Development evidence

Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA toolkit 12.8.93,
V100 SXM2 32GB. All GPU tests use an owned native lease. Evidence root:
`/data/minimax-h3/sm70-general-20260909/`.

- `epilogue-integration-v1.log`: 43 checks pass, including explicit comparison
  with the old unfused adapter path for original/W8A16 bases, FP16/FP32 output,
  prepared/ordinary inputs, negative scales, wide intermediates, untouched
  slices, overlap fallback, unaligned storage and CUDA Graph replay.
- `epilogue-cpu-v1.log`: 94 adapter, workflow and strict acceptance checks pass.
- `epilogue-micro.json`: paired postprocessing-only medians for 34,560 rows,
  output width 5,376: three FP16 QKV slices 7.524352 -> 2.015232 ms; one FP32
  row projection 4.562944 -> 3.094528 ms. Results match bitwise. These timings
  exclude GEMM and do not establish complete-request speedup.
- `epilogue-binaries.json` retains the first tested binary. The strengthened
  alias check is in `epilogue-binaries-v3.json`; validation is recorded separately.

Complete four-step latent/RGB/PCM comparison and one full warmup plus three
unprofiled requests remain required before this change can be promoted.
Independent official reference and full audiovisual review remain required
even if frozen-mainline preservation passes.
