# Flash-Next HC boundary on SM70

HC optimization must measure the complete dependency chain: block-output TP
reduction, residual combine, grouped Gemma RMSNorm, down projection and SiLU,
up projection, and gated stream mix. Projection time alone excludes useful
fusion opportunities and cannot establish model-round savings.

The Flash-Next contract is four streams, hidden size 2560, low-rank size 320,
FP16 materialization and FP32 projection accumulation. The verifier uses
M=5 for C1 and M=20 for C4. There are 96 HC boundaries per target round.

## Projection attribution

`benchmark_sm70_hc_waits.py` reads eight real HC pairs and runs four-rank CUDA
graphs. Compute-only probes retain projection, ordered split-K reduction,
activation, and local output work. They exclude remote writes, forwarding,
and remote output collection. These are diagnostic probes, not selectable
model operators.

Hardware: four Tesla V100-SXM2 32GB GPUs, SM70. Direct NVLink edges are
0–1, 0–2, 1–3 and 2–3; diagonal connections require forwarding. Runtime:
Python 3.12, Torch 2.10.0, CUDA 12.8, driver 580.173.02. Control wheel:
`1.5.2.dev1089+g09930f896`. Each point reports the maximum of four rank medians,
five samples, 40 replays/sample and 64 projection pairs/replay. Activations
are synthetic FP16; the eight weight pairs exceed L2 capacity.

| Projection scope | M=5, µs | M=20, µs |
| --- | ---: | ---: |
| Packaged down + up | 22.612 | 31.526 |
| Diagnostic down + up | 21.155 | 31.986 |
| Diagnostic down, exchange enabled | 10.182 | 14.106 |
| Diagnostic down, local compute/output | 9.049 | 12.650 |
| Diagnostic up, exchange enabled | 10.289 | 17.402 |
| Diagnostic up, local compute/output | 8.276 | 12.582 |

Diagnostic full output and injection match the packaged operator bitwise at
both M values on every rank. Adding a runtime probe branch changes compiler
scheduling: the diagnostic and packaged M=5 timings differ. Subtract only
matched diagnostic arms to screen hypotheses; do not interpret subtraction
from the packaged kernel as an exact communication breakdown.

Local compute and reduction dominate M=5: eliminating exchange still leaves
roughly 17.3µs for the two projections. Communication-only optimization cannot
bring the complete boundary below 10.4µs, the 1ms/96-boundary target.

Phase timestamps use `clock64` differences within the same CTA. They are
recorded at the tail of repeated four-rank graph replay to exclude independent
Python launch skew. Do not compare absolute SM clocks across CTAs or GPUs, or
convert cycles to time without recording the clock. The initial independently
launched phase sample included host skew and is unsuitable for exchange
attribution; the graph timing rows above remain valid.

## Complete-boundary fusion screen

A separate research-only DSO screens the supplied persistent HCX algorithm
against the same packaged control. It is not a production model route.
Maximum rank median times include the TP reduction and combine/norm.

| Complete boundary | M=5, µs | M=20, µs |
| --- | ---: | ---: |
| Existing boundary | 32.138 | 58.374 |
| Reassociated norm-factor HCX | 25.906 | 94.268 |
| HCX restoring FP16 norm materialization before MMA | 27.332 | 100.597 |
| Restored rounding, per-CTA lora-word polling | 49.388 | 167.280 |

Restored rounding gives a **0.461ms operator estimate** across 96 boundaries,
not a measured model-round gain. Its M=5 relative L2 errors against the control
are 6.53e-5 for block output and 3.88e-5 for injection. The reassociated variant
changes the FP16 rounding boundary; it is not a lossless algebraic rewrite.

At M=20 the control's 102400-byte reduction exceeds the small-message ring's
25600-byte capability and uses NCCL. The candidate executes three groups of
at most eight rows. All screened chunked variants regress and must retain the
existing M=20 route. Repeated per-CTA polling also regresses M=5 and is rejected.

These screens do not establish natural-output quality, MTP acceptance, or
end-to-end model speed. Accepted candidates must enter the normal CMake and
wheel build, pass replay/numerical tests, and then undergo fixed-contract C1/C4
model comparison.

## Community mechanisms to evaluate

[mKernel](https://github.com/uccl-project/mKernel) uses tile-ready computation,
explicit communication roles, and persistent scheduling. Its default targets
Hopper; TMA, multicast, and warpgroup MMA cannot be copied to V100. The useful
mechanism here is bounded, coalesced progress publication, rather than every
consumer repeatedly polling every word. Its MIT license must accompany any
ported implementation.

[DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) provides HC
prenorm with joint square-sum/GEMM and adaptive split-K.
[vLLM Qwen4Exp](https://github.com/vllm-project/vllm/blob/049507aa76200090f8bc6ca4e48ab0e09f2f0d29/vllm/models/qwen4_exp/nvidia/hyperconnection.py)
and [SGLang Qwen HC](https://github.com/sgl-project/sglang/blob/730f1f3e5be9c781f523f856fc39c18c2ad266d6/python/sglang/kernels/ops/gemm/hc_mix.py)
provide fused projection epilogues and persistent alternatives on newer
architectures. DeepSeek mHC's coefficient dependencies differ from Flash-Next;
its delayed pre-mix cannot be assumed to preserve this graph.

Next decisions are driven by local load/MMA/partial-reduction counters. For
the restored-rounding candidate, separate per-stream partials are unnecessary:
norm factors have already been applied before down MMA. A compact FP32 partial
layout can reduce traffic without moving the FP16 rounding boundary.
