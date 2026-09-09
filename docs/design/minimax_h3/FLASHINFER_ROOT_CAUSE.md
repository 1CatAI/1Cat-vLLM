# H3 FlashInfer: why Tensor Cores are underfed

This investigation uses the independent FlashInfer Q128/K64 D128 route at
`482b6fa8446935fdb60b5419640b54c1cb00bc37`. The retained unprofiled result is
43.862270 useful TFLOPS per TP4 rank over the complete 20-update, 39-frame
1344x768 schedule. The 80-TFLOPS requirement is still incomplete. Measurements
below diagnose that result; operator rates and profiler percentages are not
substituted for full-denoise throughput.

## Critical-path budget

The existing two-update Nsight Systems capture contains no compute/communication
overlap and less than 1% of wall time without recorded GPU activity. Rank 0:

| Work | Seconds | Effective matrix TFLOPS while executing |
| --- | ---: | ---: |
| Noncausal attention | 2.946113 | 36.9473 |
| Linear GEMM | 2.722174 | 87.2554 |
| TP communication | 1.057550 | — |
| Other GPU kernels | 0.932589 | — |
| ConvRot, dequantization, copies, gaps | 0.245456 | — |
| Total synchronized interval | 7.903882 | — |

The matrix numerators are the measured model's per-layer useful-FLOP ledger,
scaled from 20 to two updates, including the refiners. They exclude padding,
ConvRot and dequantization. Ordinary GEMMs are already above 80 TFLOPS during
their own execution; attention and time outside matrix work lower the full
model rate substantially.

An optimistic calculation illustrates the problem for **this short shape**:
retain 2.235595 seconds outside attention/GEMM, but assume every matrix
operation reaches 125 TFLOPS and attention has no internal softmax overhead.
The resulting whole-model rate is only 69.1838 TFLOPS. This is a fixed-overhead
budget estimate from a profiled two-update sample, not a hardware limit for
other shapes or the 243-frame primary workload. Both matrix feeding and
non-matrix/communication time must improve. CUDA Graph launch reduction alone
cannot remove these GPU costs.

## Attention instruction evidence

Nsight Compute 2022.4.1 on GPU 1 profiles the installed binary SHA256
`23eb9d8a2f60ccf674f82bbfc8fe9bcb7af57ee693f1b8dddae3340cc227e238`
at B1/S12323/H14/D128, with the final operator selected by NVTX. One operator
uses 24 replay passes for scheduler, memory and per-PC source counters.

| Counter | Observed |
| --- | ---: |
| Tensor pipe active cycles | 32.27% |
| L1 LSU data-pipe wavefront utilization | 78.99% |
| DRAM throughput utilization | 1.08% |
| L2 hit rate | 97.08% |
| Active warps per scheduler | 4 of 16 |
| Eligible warps per scheduler | 0.90 |
| Scheduler cycles with no eligible warp | 56.41% |
| Registers/thread | 126, no spills |
| Dynamic shared memory/CTA | 92,160 bytes |
| Resident CTAs/SM | 1, limited by both registers and shared memory |
| Excessive shared wavefronts | 704,508,672 of 2,466,484,272 (28.56%) |

PC sampling observes 465,404 MIO-throttle samples, 230,458 short-scoreboard
samples and 130,462 barrier samples out of 1,621,577 total. These are sampled
warp states, not additive wall-time percentages. QK's shared operand loads
dominate the largest MIO-throttle PCs. Four scalar loads in the V transpose
each generate 50,061,312 excessive shared wavefronts; their dependent stores
are the two largest short-scoreboard PCs.

The conclusion is supported by both occupancy and instruction evidence:
there are too few independent ready warps to hide on-chip operand access and
synchronization latency. HBM bandwidth and CPU launch starvation are not the
dominant causes in this capture. Large MIO/scoreboard stalls also explain why
GPU utilization can read 100% while Tensor Core activity is much lower.

## Communication control

A separate probe initializes the same TP4 vLLM groups on GPUs 0–3. Runtime
dispatch logs confirm `pynccl` for FP32 row-projection controls.
CPU-group barriers precede CUDA-event timing; three warmups and five measured
calls are used. All output elements exactly equal the expected four-rank sum.

| Payload | Rank 0 median | Other rank medians | Ring bus bandwidth |
| --- | ---: | ---: | ---: |
| 12323 x 5376 FP32, 264,993,792 bytes | 5.486 ms | 5.417–5.454 ms | 72.46–73.38 GB/s |
| 73483 x 5376 FP32, 1,580,178,432 bytes | 31.905 ms | 31.767–31.898 ms | 74.29–74.62 GB/s |

Subsequent GEMM route instrumentation found that the short model pads 12,323
valid tokens to 12,352 rows for linear layers and all-reduce. Its actual short
payload is 265,617,408 bytes. These standalone controls use valid-token rows,
so the short control is 0.235% smaller; it is not an exact physical-shape
benchmark. The model useful-FLOP ledger intentionally excludes that padding.

Bus bandwidth uses `1.5 * payload_bytes / elapsed_seconds` for a four-rank
ring. It is not model FLOPS. The standalone short payload time is similar to
the model's approximately 5.2 ms per major all-reduce on rank 0. Other model
ranks spend longer in collective kernels while arriving ahead of rank 0;
that extra time cannot all be labeled link transfer.

The topology uses direct NVLink connections among GPUs 0–3, with unequal
single/double-link widths. NCCL initialization reports eight collective
channels and P2P/CUMEM routes. The collected tuning tables describe estimated
algorithm/protocol costs but do not prove a selected per-call protocol.
The generic kernel's `RING_LL` suffix is still insufficient evidence to force
a protocol change. No protocol, power or clock override was introduced.

## Implemented candidates and rejected combinations

The V transpose now exchanges adjacent half pairs within each warp's 8x8 tile.
It uses one 32-bit shared load, a lane shuffle, and one 32-bit shared store per
pair, avoiding the old strided scalar bank conflicts without changing bits.
All 13 tested lengths and the S73483 sampled FP32 reference pass; outputs are
bitwise equal to the preceding Q128/K64 implementation. CUDA 12.8 emits
128 registers/thread with no spills; shared capacity and occupancy remain
unchanged.

| Isolated attention control, GPU 1 | Previous K64 | 8x8 transpose |
| --- | ---: | ---: |
| S12323, median of five alternating pairs | 27.1452 ms | 26.7059 ms |
| S73483, median of three alternating pairs | 943.9109 ms | 928.0246 ms |

All these samples report 1530 MHz SM clocks. The improvement is about 1.6–1.7%,
which establishes a partial fix rather than removal of the dominant QK stalls.

Q/K RMSNorm and RoPE previously executed many PyTorch intermediate operations.
The new Triton path fuses each Q or K pass while preserving the FP32 reduction,
normalized FP16 conversion, separate FP16 rotary products and final FP16
addition/subtraction. It supports strided QKV projections, D128 and both
96/128 rotary widths; other inputs retain the numerical reference route.
The exact S12323/H14 control takes 0.3031 ms for Q and K versus 3.5215 ms for
the reference. The four initial sampled shapes are bitwise equal. Model and
expanded regression validation are recorded separately below when available.

Rejected or unpromoted experiments:

- Reuse P fragments across four PV output fragments: 27.1452 -> 26.7141 ms,
  bitwise equal on the 13-length control. Combining it with the new transpose
  takes 26.8483 ms, worse than either change alone. Do not combine the changes
  on the assumption that isolated gains add together.
- Chunked GEMM/FP32 all-reduce overlap, M12323/N5376 controls:
  two chunks usually regress. Four chunks reduce rank 0 medians from 8.3528
  to 7.8387 ms (K1792), and 10.6076 to 9.4966 ms (K3584), with noticeable
  variability. Maximum numerical difference is 6.10352e-5, relative L2 below
  5e-8. This is only a synthetic projection control; the overlap path is not
  installed or credited as model speed. It needs a steady-state trace and
  real-model evidence before adding stream/collective complexity.
- Two probe launch attempts encountered another task's GPU lease and exited
  before allocating GPU memory. Those attempts have no performance result.

## Complete short-denoise verification

The 39-frame, 1344x768, 24-FPS, seed-42 run retains TP4 on GPUs 0–3,
Comfy INT8 ConvRot, video/audio shifts 12/3 and 20 actual updates from 21
sigma positions. It uses verified cached TP4 text conditioning, one
single-update warmup, and one unprofiled full short schedule.

| Full short-denoise result | Before | After |
| --- | ---: | ---: |
| Denoise seconds | 78.968862 | 75.140751 |
| Seconds per actual update | 3.948443 | 3.757038 |
| Useful TFLOPS on each rank | 43.862270 | 46.096872 |
| Useful FLOPs per rank | 3,463,753,579,661,312 | 3,463,753,579,661,312 |

Throughput increases 5.095%; denoise time falls 4.848%. These are single-run
development measurements, not the primary three-run acceptance. Video and
audio latents are bitwise equal to the preceding K64 run. Fresh decoding
takes 6.572545 seconds, passes all automatic checks, and produces the same
MP4 SHA256 `02ee9057bfa997ac578d8fdda11acd9770d99b86022207d1674a9dbc65eb50cb`.
This preserves that test output; human five-axis acceptance remains pending.

The follow-up two-update trace gives the actual improvement source:

| Exclusive wall category | Before seconds | After seconds |
| --- | ---: | ---: |
| Attention | 2.946113 | 2.899181 |
| GEMM | 2.722174 | 2.728789 |
| TP communication | 1.057550 | 1.061110 |
| Other GPU kernels | 0.932589 | 0.608317 |
| ConvRot | 0.119705 | 0.120093 |
| Dequantization | 0.062780 | 0.062977 |
| Copies | 0.012645 | 0.012464 |
| No recorded GPU activity | 0.050326 | 0.033749 |
| Synchronized interval | 7.903882 | 7.526680 |

The trace records 200 `_qk_norm_rope_kernel` launches and 104 independent
FlashInfer attention launches on rank 0. Auxiliary GPU work falls by 0.324272
seconds; attention falls by 0.046932 seconds. Communication remains material.

All 79 video tests pass, including 14 QK rounding/stride/graph/autograd tests.
The final autograd dispatch guard retains the reference outside inference;
it was added after timing and covered by a full 79-test rerun. The timed
inference path's GPU operations are unchanged. Compute Sanitizer memcheck,
racecheck and synccheck each passed the 24 focused kernel cases before this
host dispatch guard; no hazards or errors were reported.

Final attention binary SHA256 is
`9ee5d37eb9f930f6df92d434efd377cb324f9970f3becec161eca4230cb6d868`.
DiT peak allocated memory is 6.647851 GiB/rank, with NVML monitored peak
8.583496 GiB/rank. Median SM clocks are 1402/1500/1507/1500 MHz; median powers
are 272.884/270.581/273.465/276.414 W. GPU 0 retains the preceding run's
1402-MHz median. No clock or power settings were changed.

## Evidence and next gate

Artifacts are under
`/data/minimax-h3/native-h3-20260908/flashinfer-rootcause/`: detailed NCU report,
raw CSV, SASS counters, `short-shape-budget.json`, operator controls,
`comm-result.json`, NCCL logs, topology output, and `overlap-result.json`.
The preceding complete-denoise evidence is retained in `feeding-round3/`.

Prioritize reducing QK operand traffic and hiding its on-chip dependencies,
then validate whether communication overlap can improve the complete critical
path. Do not replace useful-FLOP acceptance with Tensor Core percentages,
isolated operator rates, or a longer video's extrapolated rate.

The next user-specified milestone is **below 50 seconds** for the same
39-frame/20-update configuration without quality regression. At the fixed
useful-FLOP numerator this corresponds to greater than 69.275072 TFLOPS per
rank. The broader >80-TFLOPS requirement remains open. Reaching 50 seconds
from 75.140751 requires another 33.46% wall-time reduction, so both attention
and non-attention costs must improve.

The dependency branch independently added a QK fusion in `88421d1182` while
this isolated investigation ran. Its common helper/test paths overlap this
change. Reconcile that shared helper before merging PR #564; no concurrent
FlashAttention worktree or branch was modified by this task.
