# FlashInfer TP4 residual sharding: 66.86-second development result

The optional residual sequence-parallel path completes the unchanged
1344x768, 39-frame, 24-FPS, seed-42 workload in **66.863312 seconds for 20
actual denoise updates** (21 sigma positions, video/audio shifts 12/3).
This is 5.60% less time than the 70.828264-second replicated-residual baseline.
The below-50-second target and the broader >80 useful TFLOPS/card requirement
remain incomplete. Quality review is also incomplete: this option changes
floating-point reduction order and is **disabled by default**.

## Implementation and precision

`--residual-sequence-parallel` is accepted by both `vllm video generate` and
`vllm video serve`. The initial supported configuration is FL2VA, serialized
INT8 ConvRot, TP4 and explicit `FLASHINFER_SM70`. Other configurations fail
before loading. The model additionally rejects unaligned rows, multiple
requests and concurrent Ulysses/sequence-parallel hooks.

Each rank owns one quarter of the main blocks' FP32 residual rows. Normalization,
modulation and gated residual updates operate on those local rows. After the
existing normalization casts to FP16, an all-gather supplies the full sequence
to the column-parallel QKV/FC1 projections. Attention remains the independent
FlashInfer D128 kernel with the same 14 local heads and valid-token metadata.
The attention output and FC2 projections return FP32 partial sums, followed by
FP32 reduce-scatter instead of all-reduce. The final block's FP32 rows are
gathered before the existing padding boundary and output heads.

No FP32 residual or partial sum is compressed to FP16. The FP16 all-gather
occurs at a pre-existing FP16 activation boundary. Original INT8 codes,
channel scales, ConvRot coordinates, FP32 accumulation and effective model
FLOPs are unchanged. Floating-point sums can still differ because NCCL
reduce-scatter and all-reduce use different reduction orders.

For a ring-volume comparison, replacing a full FP32 all-reduce with FP32
reduce-scatter plus FP16 all-gather reduces traffic by 25% per transition.
This is an explanatory estimate, not a measured NCCL-protocol or timing claim.
There is one additional full FP32 gather at the block-stack exit. Actual
physical rows are 12,352; useful attention and GEMM accounting excludes padding
and uses 12,323 valid rows.

## Full short-schedule measurements

Both measurements use GPUs 0–3 exclusively and the same 200-weight exact FP16
cache within a 10-GiB budget. The cache stores 9,633,792,000 bytes/rank. One
single-update warmup precedes one unprofiled full short schedule. Prompt-verified
TP4 text conditioning is reused; these are denoise, not end-to-end results.

| Metric | Replicated residuals | Sharded residuals |
| --- | ---: | ---: |
| Complete synchronized denoise (s) | 70.828264 | 66.863312 |
| Seconds / actual update | 3.541413 | 3.343166 |
| Useful TFLOPS / each rank | 48.903550 | 51.803500 |
| Useful FLOPs / each rank | 3,463,753,579,661,312 | 3,463,753,579,661,312 |
| DiT peak Torch allocation (GiB/card) | 15.650796 | 15.279062 |
| DiT peak NVML used memory (GiB/card) | 17.763184 | 17.433105 |

Fresh VAE decode takes 5.488337 seconds, reported separately. The development
prototype measured 66.892855 seconds and produces exactly the same video/audio
latents and MP4 as the native implementation. These two builds are not combined
into a formal three-run statistic. Whole-pipeline memory acceptance remains open.

All automatic checks pass: 39 complete frames, 1344x768, 24 FPS, finite nonzero
audio with valid duration, no black frames or prolonged static run. Compared
with the previous replicated-residual baseline, video latent relative L2 is
0.026166 (maximum absolute difference 2.215890), and audio latent relative L2
is 0.008113 (maximum absolute difference 0.011405). Full-video SSIM is 0.986179
(minimum frame SSIM 0.983301). Decoded stereo PCM correlation is 0.984292 and
relative L2 is 0.176570; the audio waveform is not unchanged.
Endpoint visual inspection retains one red boat, one yellow duck and the pool
background, with small detail/reflection differences. This is supporting
evidence, **not five-axis human acceptance or proof of temporal/audio quality**.

Native MP4 SHA256:
`5f1e58e555d613ed28e3793d02022b914fc62486190b9bfcf6300648a1974fb2`.

| NVML during native denoise | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | ---: | ---: | ---: | ---: |
| Median utilization (%) | 100 | 100 | 100 | 100 |
| Median SM clock (MHz) | 1387 | 1492 | 1500 | 1485 |
| Median power (W) | 278.967 | 272.149 | 273.940 | 275.126 |
| Maximum temperature (C) | 56 | 60 | 57 | 64 |

Throttle masks are 0 or 4. No power or clock settings were changed. NVML busy
percentage does not establish useful model throughput or Tensor Core saturation.

## Current attention counters and unsuccessful candidates

A fresh Nsight Compute 2022.4.1 capture measures the retained FlashInfer binary
`d6b5240d4f10823f7919bdc038291f296c92b8f945528d7763a6a0a351082ab7`
on GPU 1, S12323/H14/D128. It uses 24 profiler passes and is separate from
unprofiled operator timing and complete-model measurements.

- Tensor pipe active: 34.91% of peak sustained elapsed cycles.
- 50.09% of scheduler cycles have no eligible warp; 1.01 eligible warps per
  scheduler on average, out of four active warps.
- 128 registers/thread, no spills, 90,112 shared bytes, one 512-thread CTA/SM;
  achieved occupancy 24.99%.
- Excess shared wavefronts: 235,879,168 / 1,931,106,352, about 12.2%.
  L1/TEX throughput is 69.65%, DRAM throughput 1.12%, L2 hit rate 97.26%.
- PC sampling records 296,681 MIO-throttle, 261,126 short-scoreboard and
  112,292 barrier samples, out of 1,494,689 total samples. These counts are
  diagnostic and must not be added as exclusive wall time.

The 29.82-ms profiler duration at about 1294 MHz is not the roughly 25.1-ms
unprofiled operator result at 1530 MHz. Earlier detailed counters belonged to
an older binary and remain in the preceding investigation document.

Three new variants assign each warp 32 query rows and 32 output columns,
reusing K/V fragments across two query fragments while retaining FP32
accumulation. Each passes 13 short/tail/long-length reference checks, but
regresses in five alternating S12323 controls at matched 1530-MHz clocks:

| Candidate | Registers/thread | Baseline ms | Candidate ms |
| --- | ---: | ---: | ---: |
| Two query fragments, unroll 1 | 118 | 25.082880 | 25.911299 |
| Two query fragments, unroll 2 | 124 | 25.080830 | 26.317820 |
| Shared maximum/alpha broadcast | 118 | 25.094145 | 27.087872 |

These variants are not installed or credited toward model performance. A
Q128/K128 experiment aliases Q/P and K/V shared storage and reloads Q per tile;
it needs 76 bytes of spill stores and 64 bytes of spill loads at 128 registers,
so no GPU benchmark or full video was run for that variant.

Reducing the aliased variant to Q96/K128 avoids spills (162 registers/thread,
384 threads, 63,232 shared bytes), but takes 41.662464 versus 25.200640 ms in
five alternating controls at 1522–1530 MHz. Its 13-length reference check passes;
it is rejected without a full-video run. Fewer softmax iterations did not offset
reloading Q, staging V and reduced warp parallelism in this implementation.

A further artifact-only control replicates full INT8 FC1/FC2 weights and runs
each MLP on local residual rows, removing its FP16 all-gather and FP32
reduce-scatter. All 100 MLP weights/scales are checked exactly against their
original TP shards before installation. Only the 100 attention projections
remain in the FP16 cache (3,853,516,800 bytes, within the same 10-GiB budget).
Two updates regress from **6.645113 to 7.030977 seconds**, while peak Torch
allocation grows from 16,405,571,072 to 19,280,807,936 bytes/card. Finite outputs
and rank equality pass; latent relative L2 against residual sharding is
0.004782 video and 0.002660 audio. This does not justify full-video testing.
The implementation is not installed: saving collectives did not offset its
larger-weight GEMM/dequantization cost.

This control explicitly corrects valid MLP rows to 3,088 on ranks 0–2 and
3,059 on rank 3. The global useful FLOP total equals the baseline; rank 3's
29 padding rows are excluded. Without this correction, the original counter
would overcount this new experimental distribution. The first attempt kept
an obsolete global-row assertion after removing the gather and stopped during
warmup; its log is retained. `outputs/local-mlp-v2-probe/results.json` contains
the completed control. No production FLOP accounting was changed.

## Two-update trace of the native implementation

Nsight Systems captures only the first two updates of the unchanged 21-position
schedule, after warmup. Rank 0's synchronized NVTX span is 6.692385 seconds:

| Exclusive GPU category | Seconds | Share of span |
| --- | ---: | ---: |
| Independent FlashInfer attention | 2.753340 | 41.14% |
| GEMM | 2.515180 | 37.58% |
| TP communication | 0.841279 | 12.57% |
| Other GPU kernels | 0.407857 | 6.09% |
| ConvRot | 0.122021 | 1.82% |
| Copies | 0.012634 | 0.19% |
| No recorded GPU activity | 0.040074 | 0.60% |

There is no observed compute/communication overlap. This trace supports
prioritizing attention operand feeding and communication; launch-gap reduction
alone cannot close the remaining 16.86-second full-schedule gap. The trace is
diagnostic, not an additional unprofiled acceptance run. Its capture contains
module/step NVTX ranges, all four ranks, `steps-breakdown.json` and CSV.
Rank 0 executes 200 main FP32 reduce-scatter kernels totaling 0.527586 seconds;
all-gather kernels, including AdaLN and boundary gathers, total 0.313676 seconds.
The kernel symbols alone do not establish the selected NCCL protocol.

## Reproduction, evidence and rollback

Use the environment, fixed model revisions, prompt and 200-layer cache command
in [FLASHINFER_TO50.md](FLASHINFER_TO50.md), adding
`--residual-sequence-parallel`. This keeps `--num-frames 39`,
`--num-inference-steps 21`, TP4 and explicit `--attention-backend FLASHINFER_SM70`.
Omit the new flag to restore replicated residuals. No new CUDA binary is
required beyond the retained FlashInfer and TurboMind H3 extensions.

The four-rank regression exercises two consecutive real DiT blocks for valid
lengths 32, 33 and 131, including row padding, modal indices, real TP partial
sums and residual values above FP16 range. It compares against replicated
blocks and verifies that altered padding does not affect valid outputs:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m pytest tests/video/test_h3_residual_parallel.py -q -k gpu
.venv/bin/python -m pytest tests/video -q \
  --ignore=tests/video/test_h3_flashattn.py -k 'not FLASH_ATTN_V100'
```

Only run the GPU commands on an exclusively leased group. Twelve additional
CPU cases cover deployment/CLI and invalid-shape guards. The full affected
suite passes **117 tests, with one TP4 test skipped and one parallel-backend
case deselected**. The skipped TP4 test is run separately under torchrun and
passes on all four ranks for all three shapes. Ruff and diff checks pass.
No CUDA code changed; existing operator sanitizer evidence remains applicable.
Initial harness
failures are retained: a missing prototype timing dictionary was fixed before
its warmup; a module-scoped TP fixture conflicted with the repository's
per-test distributed teardown, and per-case reinitialization hung. The final
test runs its three shapes inside one distributed lifetime.

Raw artifacts live under the campaign's `feeding-reuse/`: `HANDOFF.json`,
`attention-current.ncu-rep`, `native-sequence-quality-summary.json`,
`native-sequence-media-comparison.json`, `native-sequence-nvml-summary.json`,
build/probe logs and `outputs/native-sequence39-20steps/FLASHINFER_SM70/`
(MP4, WAV, latents, screenshots, NVML JSONL and PNG/SVG curves).

The remaining denoise gap is 16.863312 seconds. At the current useful FLOP
count, 50 seconds would equal 69.275072 TFLOPS/card; the separate 80-TFLOPS
requirement needs denoise below 43.296920 seconds. Neither target is met.
