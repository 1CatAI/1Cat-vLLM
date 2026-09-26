# SM70 quantized batch reuse, 2026-09-26

## Workload and scope

Baseline source: `fcf59f8e9ae50c186333e98e5cf6aae705f320de`.
Qwen3.8-27B-NVFP4 mixed NVFP4/channel-FP8 target, DFlash2 q7 probabilistic
sampling, TP4 on V100-SXM2-32GB, FP16 execution, E4M3 target KV and automatic
draft KV, Flash-V100, prefix caching, MRV2 full/piecewise CUDA graphs.
CUDA 12.8 / Torch 2.10.0+cu128 / Python 3.12.13. Max length 262144,
max sequences 16, max batched tokens 8192, GPU memory utilization 0.8.
The fixed performance workload is 2048 input / 256 output with temperature
0.7, top-p 0.8, top-k 20, seeds starting at 20260923. Performance requests
explicitly ignore EOS; quality requests use natural EOS separately.

This change recovers the qualified M17..32 row-reuse work and extends it to
M9..16 and M33..64 in the native FP4/FP8 paths. It changes neither sampling
nor the scheduler. A complete DFlash q8 verifier has M=8*C; partial verifier
steps remain part of the serving measurements.

## Implementation

- M9..16: share each packed activation fragment between two output tiles,
  reading the already resident compressed weights and scales through small
  format-specific readers. This removes the C2 research prototype's extra
  paired weight layout (approximately 1.87 GiB per TP4 rank). There is no
  new persistent weight allocation. The existing M64 compressed layouts are
  unchanged. The transient activation pack is included in timings.
- M17..32: FP4 reuses decoded weights over four eight-row tiles. FP4 and
  FP8 use packed activations and ordered two-phase reductions. FP8 gate/up
  executes the complete row batch together instead of M16 plus a tail.
  FP4's intermediate FP16 activation rounding and FP8's established FP32
  SiLU contract remain distinct. Real-weight M<=32 outputs are bitwise equal
  to the normal baseline artifact.
- M33..64: FP4 and FP8 share padded activation shared-memory layout and
  a small set of batch tile candidates in the existing TurboMind registry.
  The complete M64 tile candidate requires exact M/N/K tile boundaries;
  masked candidates handle tails. Dense-only feasibility excludes grouped
  MoE, AWQ weight types and larger prefill batches. The normal autotuner
  chooses among accepted and new candidates. No weight expansion is added.

No new environment switches are introduced. Existing SM70 DFlash2 defaults
already enable compressed batch layouts and M64 warmup/tuning. The service
launch explicitly unsets the four manual batch-layout/tuning overrides;
worker logs and environment snapshots confirm automatic settings of 1/64/64/64.

## Native real-weight microbenchmark

These are representative TP-local projections weighted by actual layer count,
including activation packing and gate/up activation. They are not complete
forward latency or decode throughput. Seven timing rounds per projection,
CUDA Graph, seven projection types; all eager/replay outputs agree.

| Concurrency at full q8 | Baseline GEMM ms | Candidate GEMM ms | Latency reduction |
| --- | ---: | ---: | ---: |
| C1 / M8 | 6.712 | 6.724 | -0.2% |
| C2 / M16 | 11.099 | 8.365 | 24.6% |
| C4 / M32 | 19.207 | 12.723 | 33.8% |
| C8 / M64 | 21.289 | 17.125 | 19.6% |

The eager operator measurements, including host launch gaps and activation
allocation, change from 7.062/11.477/19.618/21.599 to
7.076/9.077/13.185/17.433 ms at C1/C2/C4/C8. C1 is unchanged within noise;
C2/C4/C8 reductions are 20.9%/32.8%/19.3%. C2 and C8 have not met the
aspirational 30–50% GEMM latency reduction target.

## Service validation status

The candidate's three uninstrumented rolling repeats are complete. Median
decode capacity at C1/C2/C4/C8 is 243.49/289.22/330.50/379.23 tok/s.
The reference service is being remeasured; these candidate-only values do
not establish a service speedup yet. The controlled-admission windows and
acceptance records are retained separately.

The matched 16-question natural-EOS GSM8K checks both have 14 correct,
naturally stopped responses, the same wrong response (index 12) and the
same 4096-token length cap (index 16). Counting an answer mentioned in
unfinished reasoning would incorrectly score 15/16. The original
four-question smoke remains 4/4. This supports no new regression in this
sample; neither build passes an absolute 16/16 natural-stop gate. Keep the
PR in Draft until the uninstrumented service and acceptance comparisons
have been assessed.

The fresh normal-source services both load 9.57 GiB per rank and have
11.0 GiB available for KV, with capacity 1,009,312 tokens at the same memory
limit. Captured graphs change from 0.61 to 0.62 GiB per rank. Thus removing
the research C2 duplicate weights preserves KV capacity; activation packing
still has a small transient/graph allocation cost. The approximately
1.87 GiB per-rank saving is relative to the rejected prepared-weight
prototype, not a reduction from the mainline service's resident weights.

## Validation

132 focused SM70 GPU tests pass: exact reduction order, M9/15/16/17/24/31/32,
M33/63/64/65 and N tails, FP8 channel/block scales, changed-input eager/graph
replay, and gated/dense projections. Native real-weight oracle checks cover
M8/16/17/24/32/40/48/56/64 at three input amplitudes. M<=32 is bitwise equal
to baseline; M>32 uses the established TurboMind tolerance (rtol 0.02,
atol 0.0003), and eager/replay remains bitwise equal.

The candidate was built from the owned source tree and linked as the normal
`vllm/_C.abi3.so`, with no private sidecar kernels or library overrides.
The final normal extension SHA256 is
`56a195318be0c24cc46682c39f49fc8f93ecc82b21037b01adca8bb6ee3b68b1`.
Both the eager/graph table above and service measurements use this final
normal extension. The initial nine-M screening build was
`8adec32c5b8c4e84195c8953f82e90f8e8121d4c76da16b93af3e013eda0566e`.
The normal reference extension SHA256 is
`7449bff7e4cc50fd2c9e9d243b65199126c4063d3aacd7d17a034422173c1890`.
Final service and GPU tests use the final extension. `readelf` shows only
standard Torch/CUDA/system dependencies. Other runtime native extensions and
FlashQLA were built from the same owned base source.

## Rejected research paths

Preserve the previous negative results: repeated M16/M32 GEMM slicing,
per-step full FP16 weight expansion, prepared C2 duplicate weights,
register caps alone, shared decoded-B staging and the tested Marlin replacement.
A no-copy fast-scale C2 variant was slower than the exact reader and was
not integrated. Screened C8 tile variants were admitted only when the
layer-weighted total improved; isolated projection wins were insufficient.

## Reproduction and retained evidence

Use `benchmarks/benchmark_sm70_batch_gemm_reuse.py` with a single idle V100,
`--model MODEL/model.safetensors`, a normal `--extension` and an output
path. Run baseline with `--write-oracles`, then candidate against the same
`--oracles` directory. Match `--rows` and seed/order. For direct operator
micros, set the FP8/NVFP4 dense tune maxima to 64; the serving configuration
sets these automatically for the tested DFlash2 workload.

Artifact set: `sm70-gemm-expand-20260926`, containing `native-control.json`,
`native-candidate-v1.json`, `native-summary-v1.json`, `final-eager-*.json`,
`native-route-trace.json`, source/build manifests,
clean build logs, final GPU test log, both service logs, rolling request records,
controlled-admission client token events, quality outputs, and context checks.
Raw traces/build products/weights are excluded from Git.
