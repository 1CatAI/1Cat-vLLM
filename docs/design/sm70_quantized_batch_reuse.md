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
ignore EOS; natural-EOS quality is checked separately.

The final implementation extends batch reuse to FP4 and FP8 at M9..32.
A full DFlash q8 verification step has M=8*C, so these are the C2/C4 paths;
partial steps also benefit. Target-model M<=8 and the TurboMind M>32 registry
and tuner are retained. The experimental M64 changes were withdrawn after
serving guards failed. No scheduler, sampling or attention code is changed.

## Implementation and defaults

- M9..16: a common paired-projection kernel shares each packed activation
  fragment between two output tiles. Format-specific readers consume existing
  compressed FP4/FP8 weights and scales. No second persistent weight layout
  is needed. This eliminates the research C2 prototype's approximately
  1.87 GiB per-rank duplicate weights without adding mainline weight memory.
- M17..32: FP4 reuses each decoded weight over four eight-row tiles. FP4 and
  FP8 pack activations and use two physical reduction phases while retaining
  the original logical split order. FP8 gate/up processes all rows in one
  invocation, including M17 tails that previously needed M16 plus another call.
- The established FP4 intermediate FP16 activation rounding and FP8 FP32 SiLU
  contracts remain distinct. Small projections and unsupported geometry keep
  their existing dispatch. The activation pack is transient; the measured
  TP4 projections use at most 160 KiB at C2 and 320 KiB at C4 per invocation.

No new environment switch is introduced. Existing SM70 DFlash2 defaults
already enable compressed batch layouts and M64 warmup/tuning. Service
launches explicitly unset the four manual batch-layout/tuning overrides;
worker audits check the automatic settings of 1/64/64/64 and the normal
source-built extension. This does not enable previously rejected opt-in
experiments such as FP8 prescaled batch GEMM.

## Native real-weight microbenchmark

Seven representative TP-local projections are weighted by actual layer count,
including activation packing and gate/up activation. These are GEMM estimates,
not complete target forward or serving decode. Seven timing rounds per
projection are measured in eager and CUDA Graph execution.

| Full-q8 concurrency | Baseline GEMM ms | Final GEMM ms | Latency reduction |
| --- | ---: | ---: | ---: |
| C1 / M8 | 6.712 | 6.700 | 0.2% |
| C2 / M16 | 11.099 | 8.284 | 25.4% |
| C4 / M32 | 19.207 | 12.582 | 34.5% |

FP8-only C2/C4 GEMM changes from 5.313/8.118 to 4.043/5.611 ms,
reductions of 23.9%/30.9%. FP4-only changes from 5.786/11.089 to
4.241/6.971 ms, reductions of 26.7%/37.1%. C1 is unchanged within noise.
C2 has not reached the aspirational 30–50% GEMM latency reduction.

M64 still uses the original TurboMind implementation. Its microbenchmark
varies with legacy tuning; no M64 optimization gain is credited to this
final patch. Earlier 19% M64 estimates belong to rejected candidates.

## Service validation

Candidate `qpn-fc-default` is completing the paired serving gates using the
normal extension and automatic configuration. It is not yet qualified for
promotion; the GEMM estimates must not be presented as accepted serving gains.

Rolling decode capacity is C times emitted decode tokens divided by summed
per-request decode duration. It excludes each request's TTFT but includes
pauses caused by replacement prefill. The fixed workloads are C1/16,
C2/24, C4/32 and C8/48 requests, with three measured repetitions. ITL is the
distribution of per-request mean ITL, not individual SSE/token gaps.

The no-new-prefill measurement counts returned token IDs between the latest
first token and earliest last token of a concurrently admitted wave. It
includes q1..q8 and host/transport overhead; it is not a GPU-step recorder.
C8 covers all 48 fixed prompts in six waves per repetition. The initial
first-eight-prompt diagnostic is retained separately rather than substituted
for the complete C8 workload.

## Draft projection stability

The initial C1 acceptance regression was isolated with an eager native-artifact
A/B test. The 4,023 shared native GPU kernels had identical instruction streams.
Importing the reference TurboMind plans before candidate initialization restored
the two diagnostic requests exactly. Importing only FP8 plans did not restore
them (71 -> 121 verification rounds for the sensitive request); importing only
FP16 plans restored every returned token and the 71/77/71 round counts.
The sharded draft context projection was independently tuned on each rank,
including split-K 10 on two ranks and 12 on the others. Its different rounding
changes draft scores, acceptance and eventually the seeded generated path.
The recorded failed service candidates therefore do not isolate batch-kernel
arithmetic from this existing startup-tuning variability.

An experimental cuBLAS projection retained the TP4 column partition and
all-gather. It did not import a task LUT or introduce a user switch.
On the actual 1280-by-25600 local `fc.weight`, seven-round CUDA Graph medians
were 142.49/109.08/134.80/242.77 us for tuned TurboMind at M8/M16/M32/M64,
and 116.87/108.26/118.01/189.91 us for cuBLAS. Changed-input eager/Graph
checks passed. The ordinary heuristic was also slower. Omitting its packed
FP16 copy saves 62.5 MiB per rank; service model memory drops 9.57 -> 9.51 GiB.
Six existing projection-contract tests passed, but its three-run C1 service
median was 236.50 tok/s with 54.86% acceptance, versus 243.15 tok/s and
57.04% in the saved reference. The -2.74% speed and -2.18-point acceptance
changes failed the serving guards. The cuBLAS source change was withdrawn.
Disabling generic FP16 autotuning alone also failed (227.05 tok/s, 52.07%).
These are failed diagnostic candidates, not production defaults.

The remaining legacy split-10 tactic passed the first diagnostic C1 guard:
242.89 tok/s and 56.77% acceptance. It is now selected in source for the
existing SM70 TP4 sharded context-FC contract, M1..8/N1280/K25600. The
CTA8x256x64 two-stage kernel and split-10 reduction are fixed; an imported
or previously tuned incompatible plan cannot override them. A small host
cache retains the eight row-count plans. This adds no weight layout, device
workspace, environment switch or private LUT dependency. Other FP16 shapes,
M16+ projections, quantized GEMM tuning and sampling remain unchanged.
The runtime micro confirms the default selector; all eight changed-input
eager/Graph tests pass after seeding an incompatible split-16 cache. Full
service acceptance, performance and quality still decide promotion.

## Numerical and build validation

92 focused GPU tests cover the context projection and FP4/FP8 gate/up projections,
M9/15/16/17/24/31/32, changed-input CUDA Graph replay and exact reduction
order. Representative real-weight M8/M16/M32 results are bitwise equal to
the normal reference extension at three input amplitudes; eager and Graph
outputs also agree. The unchanged M64 path retains its existing tolerance.
A separate artifact audit compares 27 relevant C1 kernel instruction streams
and finds them identical to the baseline; this does not replace C1 service
latency and acceptance checks.

The final normal extension SHA256 is
`75303f12d50f02ebfb1f4f7616e009125851a207041ecda4df9e3ede9ec39d00`.
The reference normal extension SHA256 is
`7449bff7e4cc50fd2c9e9d243b65199126c4063d3aacd7d17a034422173c1890`.
The candidate is built from the owned source tree and loaded as
`vllm/_C.abi3.so`, without a private sidecar or library override. Runtime
extensions and FlashQLA use the same source base. Build artifacts and raw
traces are not committed.

## Rejected candidates and limits

Retain these results so later work does not repeat the same experiments:

- Unrestricted FP4+FP8 M64 tiles reduced estimated GEMM by about 19.6%, but
  full-48 no-prefill acceptance fell 50.51% -> 47.24% (-3.27 points).
  The first-eight-prompt diagnostic also failed (-4.97 points).
- A two-stage tuner preserved the selected legacy split-K boundaries and
  passed 134 GPU checks, including captured tails. Full-48 acceptance still
  fell to 47.86% (-2.65 points), while its decode gain was only 0.25%.
  Cross-startup bitwise identity was not established: independent legacy
  tuning itself can select different reduction partitions.
- Restoring FP4 M64 while retaining the new FP8 M64 tiles passed the full-48
  window gate: 621.91 -> 628.43 tok/s, acceptance 50.51% -> 49.57%.
  It nevertheless failed the C1 rolling guard in two runs: approximately
  226.6 tok/s and 52.22% acceptance versus 243.15 and 57.04%. That run reused
  a compilation cache. The later FP16-only plan ablation above identifies
  draft projection tuning as a confounder; this candidate remains unqualified
  until its own ordinary-service gates are repeated.
- The initial broader candidate had the same 14/16 correct natural stops as
  the reference on its quality pair: one wrong answer and one request
  capped at 4096 tokens. Counting an answer embedded in unfinished reasoning
  would incorrectly report 15/16. The original four-question smoke was 4/4.
- Previous rejected paths include repeated M16/M32 slicing, per-step full
  FP16 weight expansion, prepared C2 duplicate weights, register caps alone,
  shared decoded-B staging, the tested Marlin replacement, and a slower
  no-copy fast-scale C2 variant.

No fresh PRO comparison or full 35B-A3B AWQ/FP8 model-speed claim is made.
The final patch does not change AWQ or grouped MoE operators. It does not
establish the broader objective of surpassing PRO by 5%.

## Reproduction and retained evidence

Use `benchmarks/benchmark_sm70_batch_gemm_reuse.py` with a single idle V100,
`--model MODEL/model.safetensors`, a normal `--extension`, an output path
and `--rows 8 16 32 64`. Run the baseline with `--write-oracles`, then the
candidate against the same `--oracles` directory, seed and row order.
For direct operator micros, set the FP8/NVFP4 dense tune maxima to 64;
the tested serving configuration sets these automatically.

Artifact set: `sm70-gemm-expand-20260926`. The final evidence uses
`qpn-only-eager*.json`, `qpn-only-kernel-tests.log`, `c1-sass-audit.json`,
`service-qpn-only.log`, request-level JSON records, native/build manifests,
controlled-admission token events, natural-EOS outputs and context checks.
Failed candidates retain their own artifact hashes, checkpoint directories
and `candidate`, `split-safe`, and `fp8-m64` result labels.
