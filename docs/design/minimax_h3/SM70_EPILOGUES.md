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

The final alias guard passed on GPU0 (`epilogue-alias-v3-gpu0.log`). An earlier
GPU4 attempt was refused by an existing lease before starting the test.

## Complete four-step control and measurements

Source `7ea8908d83827dd8d82c34ba6a60b2beaa8057d6`, TP4, LightX2V four-step v1.2,
W8A16, FA, exact residual sharding, no persistent FP16 weight cache, and the
original 1280x736/124-frame internal canvas for the five-second sample:

- `epilogue-quality.json`: final video/audio latents and all pre-encoding RGB
  frames match frozen mainline bitwise; video SSIM 1, spectral cosine 1 and RMS
  ratio 1. This is numerical preservation, not independent official acceptance.
- `epilogue-720p-three-runs/performance.json`: one full warmup (64.373997 seconds
  denoise) followed by three complete requests without profiler or captures.
  Denoise times are 62.245159 / 62.183194 / 62.162648 seconds; CV 0.056387%.
  Every rank reports median **49.905491–49.905510 useful TFLOP/s**. The declared
  >80 gate fails and remains incomplete.
- Complete request times are 84.363265 / 91.992465 / 88.620830 seconds. Peak
  allocation remains 19,501,498,880 bytes per card. Exact source/kernel hashes,
  per-step records and NVML samples are retained beside each run.
- The 62.183194-second median is 5.64% below the audited original FA baseline
  (65.898529 seconds). This measures the **combined** prepared/residual/epilogue
  changes, not an isolated attribution to this CUDA epilogue. A separate matched
  profile is necessary for attribution.

Independent official reference, full audiovisual review, other adapters and
the complete shape/TP matrix remain required. No AUTO selection is qualified.

## Explicit attention query geometry

`attention_query_tile=128` / `--attention-query-tile 128` opts into a 128-query
FlashAttention-V100 CTA. The default remains 64 and retains the previous call
ABI. Both sizes use the same 32x64 warp arithmetic and key-tile selection.
The option applies independently of model weights and adapters; other attention
backends reject this explicit tiling option rather than ignoring it.

The separate prototype retains exact outputs at nine boundary lengths and
the actual 34,551-token Q/K/V capture. Full four-step video/audio latents also
match frozen mainline bitwise (`q128-profile-quality.json`). A matching pair of
full-denoise profiles is retained in `epilogue-profile-breakdown/` and
`q128-epilogue-profile-breakdown/`; profiler timings are not acceptance results.
The public kernel/CLI implementation additionally passes 69 GPU tail, storage,
cross-attention-length and graph checks. Complete native API quality and three
unprofiled measurements of this explicit option are still pending.
