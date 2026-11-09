# H3 FlashInfer: exact weight cache and the 50-second milestone

The newer opt-in FP32 residual-sharding result is recorded in
[FLASHINFER_RESIDUAL.md](FLASHINFER_RESIDUAL.md): 66.863312 seconds with numerical
differences and pending human quality review. The replicated-residual baseline
and its bitwise output checks remain documented below.

The current source completes the unchanged 39-frame, 1344x768, 24-FPS,
seed-42 workload in **70.828264 seconds for 20 actual updates**. It uses TP4
on GPUs 0–3, Comfy INT8 ConvRot, the original prompt and video/audio shifts
12/3. Twenty updates require 21 sigma positions in the existing scheduler.
The below-50-second milestone and the per-card 80-TFLOPS gate remain incomplete.
No frames, steps, model operations or quality requirements were removed.

## Retained implementation

- Q uses an XOR shared-memory layout matching the SM70 A-fragment lane mapping.
  This removes conflicting operand loads while preserving QK/PV arithmetic.
  The Q128/K64 kernel uses 128 registers/thread, no spills and 90,112 bytes
  of shared memory. It still has one resident CTA per SM.
- The existing explicit FP16 weight cache stores physical `[K,N]` while
  retaining logical `[N,K]` and the original ConvRot coordinates. INT8 weights
  and FP32 scales remain intact. Admission accounts for both conversion
  temporaries and the 30-GiB allocation limit; partial preparation is cleaned
  up on failure. Caching remains disabled unless a budget and layer list are
  explicitly supplied.
- Cached projections use native cuBLASLt descriptors from the TurboMind H3
  extension: FP16 inputs, FP32 compute, no split-K reduction and no workspace.
  A bounded cache reuses descriptors per shape, output type and device. The
  measured algorithm 21/tile 24 is used only when the runtime heuristic
  reports it supported; unsupported shapes use the original row-major GEMM.
  Plans validate device, layout, shape and alignment. After dependency
  reconciliation, cached and uncached column-major weights share the same
  dispatch helper; no second cuBLASLt implementation is retained.

The fixed cache list is the four projections `attn.qkv_proj`, `attn.out_proj`,
`mlp.fc1`, `mlp.fc2` in each of the 50 main blocks. It holds 200 weights,
9,633,792,000 bytes per rank within a 10-GiB budget. This is exact weight
caching, with no reuse of denoiser activations or approximate outputs.

## Measured results

One single-update warmup precedes one unprofiled complete short schedule.
The verified TP4 text conditioning is reused to focus development on denoise;
these measurements are not end-to-end generation or the primary three-run
acceptance protocol.

| Same workload | Previous source | Current source |
| --- | ---: | ---: |
| Full denoise seconds | 75.140751 | 70.828264 |
| Seconds per update | 3.757038 | 3.541413 |
| Useful TFLOPS per rank | 46.096872 | 48.903550 |
| Useful FLOPs per rank | 3,463,753,579,661,312 | 3,463,753,579,661,312 |

The video and audio latents are bitwise equal to the preceding source.
Fresh VAE decode takes 6.158398 seconds and produces the same MP4 SHA256:
`02ee9057bfa997ac578d8fdda11acd9770d99b86022207d1674a9dbc65eb50cb`.
All automatic video/audio checks pass. Human five-axis quality review remains
pending; output equality establishes no regression for this fixed test.

Weight staging/cache preparation takes 1.078–1.105 seconds, reported separately
from denoise. The earlier development proxy took 70.826228 seconds but retained
four unused 64-MiB cuBLASLt workspaces per rank. Native descriptors remove them:
DiT Torch peak allocation is 15.650796 GiB and NVML peak is 17.763184 GiB per
card. These are DiT measurements, not whole-pipeline peak-memory acceptance.

| NVML median during the current run | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | ---: | ---: | ---: | ---: |
| GPU utilization (%) | 100 | 100 | 100 | 100 |
| SM clock (MHz) | 1402 | 1500 | 1515 | 1496 |
| Power (W) | 277.686 | 266.494 | 269.139 | 270.527 |
| Maximum temperature (C) | 55 | 60 | 56 | 64 |

Throttle reason masks are 0 or 4; no clock/power settings changed. Utilization
does not establish Tensor Core saturation. The older detailed instruction
capture in [FLASHINFER_ROOT_CAUSE.md](FLASHINFER_ROOT_CAUSE.md) diagnoses shared
operand access, ready-warp shortage and synchronization; its counters are not
claimed as measurements of this new binary.

Isolated alternating controls on GPU 1 separate the changes:

| Operator, median milliseconds | Previous | Candidate |
| --- | ---: | ---: |
| Attention S12323/H14/D128, five pairs | 26.707968 | 25.097216 |
| Attention S73483/H14/D128, three pairs | 930.114563 | 883.246094 |
| QKV M12352/N5376/K5376, FP16 output | 8.150016 | 7.542784 |
| FC1 M12352/N7168/K5376, FP16 output | 10.720256 | 9.936896 |
| Attention output M12352/N5376/K1792, FP32 output | 2.936832 | 2.782208 |
| FC2 M12352/N5376/K3584, FP32 output | 5.549056 | 5.184512 |

All retained operator outputs are bitwise equal. GEMM uses nine alternating
samples per variant after warmup. The short attention pair runs at 1530 MHz;
the long control compares 1530 against 1507 MHz. GEMM clocks vary around
1267–1327 MHz under its sustained power load and are recorded per sample.
Ordinary cuBLAS with transposed weights is slower than the retained cuBLASLt
route on these four shapes, so it was not selected.

**Shape correction:** attention uses 12,323 valid tokens, while physical GEMM
and all-reduce use 12,352 padded rows. Earlier 12,323-row communication and
overlap controls were 0.235% smaller than the actual short payload. The new
GEMM controls use physical rows; the model FLOP numerator still excludes
padding. The earlier communication table has been corrected accordingly.

## Rejected paths and next bottleneck

- Fully register-resident Q, half-register Q, additional P swizzling and the
  initial K128 alias variants spill registers; they were not installed.
- Serializing vector loads removes spills from the one-warp/query K128
  variant (253 registers, 48-KiB shared memory). Its 13-length reference check
  passes, but S12323 takes 63.6672 versus 26.6742 ms. More resident CTAs do not
  compensate for the reduced warp parallelism and extra operand traffic.
- A four-warp/query K64 variant spills at the register cap required for two
  512-thread CTAs. A smaller Q32 version removes spills (123 registers,
  256 threads, 37,120 shared bytes) but takes 51.3116 versus 26.6527 ms at
  identical 1530-MHz clocks. Its 13-length reference check passes. Neither
  path justifies a full video experiment.
- Packing probability writes into FP16 pairs preserves bits, but has no
  measurable short-shape benefit: 25.0798 versus 25.0788 ms at 1530 MHz.
  Additional P swizzling with reduced loop unrolling removes spills (126
  registers) but regresses from 25.0808 to 28.2798 ms, also at 1530 MHz.
  Both pass 13-length reference checks and preserve bits; neither is installed.

The remaining gap is 20.828 seconds, about 29.4% of the current denoise time.
Attention feeding and time spent outside matrix operations both need further
work; graph launch reduction alone cannot meet this gap. No standalone peak,
NVML utilization or extrapolated rate is counted as a milestone result.

## Reproduction and checks

Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA Toolkit 12.8.93,
Transformers 5.15.1, four V100-SXM2-32GB GPUs 0–3, FP16/FP32 mixed computation.
Model revisions remain `42ed227ee7df40d41602854ae760620d6eb651fe` and
`a98869194787969724c7425d95d0ed73ce9202af`. Rebuild the H3 extensions after
updating source; both the CMake target and lazy source builder link cuBLASLt.

For normal generation, use the existing default boat/duck prompt:

```bash
h3_cache_args=(--fp16-weight-cache-gib 10)
for h3_block in {0..49}; do
  for h3_projection in attn.qkv_proj attn.out_proj mlp.fc1 mlp.fc2; do
    h3_cache_args+=(--fp16-cache-layer "blocks.${h3_block}.${h3_projection}")
  done
done
vllm video generate --model "$H3_MODEL_PATH" --partition fl2va \
  --transformer-path "$H3_INT8_PATH" --tensor-parallel-size 4 \
  --attention-backend FLASHINFER_SM70 --int8-weight-layout column --width 1344 --height 768 \
  --num-frames 39 --num-inference-steps 21 --seed 42 \
  "${h3_cache_args[@]}" --output-dir h3-flashinfer-cached
```

Dependency `22d784475e` contributes the shared column-major decode/GEMM helper;
this branch retains its own FlashInfer attention and QK fusion. The duplicate
cached GEMM class was removed. The shared heuristic now explicitly advertises
the same 16-byte operand alignment checked by `run()`, with an offset-view
regression. Unsupported algorithms/layouts fall back to the original GEMM.
The dependency's FlashAttention sources are unchanged by this reconciliation.
The earlier independent cache implementation took 70.819016 seconds; the
reconciled run above preserves the same latents and video. This comparison
checks integration, not an additional performance improvement.

The current native suite passes **105 tests**, excluding the parallel FlashAttention test file and backend case. Before reconciliation, memcheck, racecheck and synccheck each passed 14
targeted tail, query-group, cache-failure, FP32-overflow and graph cases, with
zero errors or hazards. The FlashInfer binary is unchanged; an additional
post-merge memcheck passes seven column-decode, cache-lifetime and 16-byte
alignment cases with zero errors. Commands use the owned environment:

```bash
.venv/bin/python -m pytest tests/video -q \
  --ignore=tests/video/test_h3_flashattn.py -k 'not FLASH_ATTN_V100'
compute-sanitizer --tool memcheck --error-exitcode 1 \
  .venv/bin/python -m pytest -q tests/video/test_h3_numerics.py \
  tests/video/test_h3_weight_cache.py \
  -k 'prefetch_tail or query_groups or cached_gemm_fp32_output or cache_preserves'
```

Repeat the sanitizer command with `racecheck` and `synccheck`. Raw evidence is
under the campaign artifact directory's `feeding-to50/`: `reconcile-manifest.json`,
`reconcile-tests.log`, sanitizer logs, `reconciled-quality-summary.json`,
`reconciled-nvml-summary.json`, operator results and
`outputs/reconciled39-20steps/FLASHINFER_SM70/` (MP4, WAV, latents, screenshots,
NVML curves and exact run configuration). Build products and weights are not
committed. Roll back cached GEMM by omitting the cache flags; roll back the
entire change by using parent `ebec83fbd4526d73b5c8bd8b437104f93c0bd632` and
rebuilding its H3 extensions in a separate worktree.
