# SM70 mixed NVFP4 DFlash2 complete-round latency

## Acceptance contract

The target is a complete single-request speculative round below 12 ms on
four V100-SXM2-32GB GPUs: target verification, rejection sampling, and DFlash2
draft forward with seven speculative tokens. Freeze a new baseline on the
measurement host before accepting performance deltas. The 21.1 ms observation
from another host is context only.

Use the unsloth Qwen3.8-27B mixed checkpoint: layers 0–55 have NVFP4 MLPs;
GDN QKV/Z/output projections, attention projections, layers 56–63 MLPs, and
the LM head have channel FP8 weights; GDN A/B projections are BF16.
Checkpoint identity must be verified before substituting any existing model.

Launch with maximum model length 262144. Measure 1024- and 8192-token inputs
with thinking disabled and temperature 0.7. Recover the remaining sampling,
output-length, KV, graph, and prompt settings from the original baseline
scripts before freezing the contract. Report complete-round wall time and
emitted tokens per round separately from TTFT, prefill, and GPU kernel service
sums. Use unprofiled runs for acceptance and short Torch profiler captures for
attribution. Preserve acceptance and concurrent-request performance.

Numerical changes require mean KL <= 0.001, p99 KL <= 0.01, maximum KL <= 0.05,
top-1 agreement >= 99%, and maximum logit difference <= 0.5. Sampling changes
must preserve the distribution, including ties, top-p boundaries, and
presence/frequency penalties. Draft or LM-head precision reductions require
separate approval. Accepted routes must select automatically by capability;
new per-optimization environment switches are outside this scope.

## Initial evidence, 2026-10-03

Integration base: `8002bc10709c25db8dc576e4c96ab3e614d6131e`.

All six physical GPU pairs report NV2. A standalone SM-store benchmark tests
all twelve directed pairs with an 81920-byte payload and a system fence,
using 8, 20, and 40 blocks of 256 threads. Each sample times a CUDA Graph with
100 store kernels; three warmup replays precede twenty timed replays. All
payload checks pass. Mean per-node times range from 7.139 to 8.591 us across
pairs and grids. CUDA 12.8 compiled the benchmark for SM70. This is a peer-store
diagnostic, not collective latency, a receiver handshake benchmark, or a
model speedup. Percentiles describe graph-average samples rather than
individual stores.

Reproduce under exclusive ownership of the selected GPUs:

```bash
nvcc -O3 -arch=sm_70 benchmarks/csrc/sm70_peer_store_latency.cu -o peer_store
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 ./peer_store
```

The measurements support trying the existing fully connected push collective
first. No slow edge requiring a butterfly was observed on this host. At this
base, `CustomAllreduce.sm70_tp4_all_reduce_gemma_rms_norm` is explicitly
benchmark-only; its existence does not prove model dispatch.

The original unsloth checkpoint has now been downloaded with ModelScope.
Its `model.safetensors` is exactly 22568192096 bytes and `model_mtp.safetensors`
is 849400392 bytes. The header identifies `lm_head.weight` as F8_E4M3,
shape [248320, 5120], with channel BF16 scales. The existing DFlash2 draft
is retained. The obsolete checkpoint with a replaced BF16 head was removed
with explicit authorization.

A fresh SM70 wheel was built from integration source
`f90026bf77a382559370d54a3661a21fae634b4c`, using a dedicated Python 3.12
environment and compiler caches. The complete artifact and dependency checks
pass. Its SHA256 is
`6d15681e321b21ca6fe4274103bc4cd7c275c96a384c4066d06ad67773d932f3`.
The baseline uses this installed wheel without private libraries or overlays.

`benchmarks/analyze_sm70_dflash2_round_trace.py` uses CUDA Graph correlation
IDs to avoid assigning asynchronously executed kernels by CPU timestamps.
Its sanity check on the retained original-model trace identifies eleven
complete rounds, with nine compact and two dense-reference rounds. All eleven
target and draft graphs match the expected prefix collective followed by two
collectives per layer: 64 target layers and five draft layers. The analyzer
retains per-kernel duration, preceding gap, grid, block, stream, and collective
backend. It leaves layer attribution unassigned when this graph structure
does not match, rather than inventing layer labels.

## Next evidence required

### Hardware and communication follow-up

The measurement host has one Xeon E5-2680 v4 socket (14 cores/28 threads),
one NUMA node, 64 GB nominal system RAM, and four V100-SXM2-32GB GPUs. Software
is Ubuntu 24.04.2, NVIDIA driver 580.173.02, CUDA toolkit 12.8.93, Python
3.12.14, Torch 2.10.0+cu128, NCCL 2.27.5, Triton 3.6.0, and Transformers
5.18.0. Each GPU has six active NVLink links, reported at 25.781 GB/s each;
each peer pair has two links. ECC is enabled.

The original 185 W GPU power caps were explicitly raised to the 300 W hardware
default before the following tests. Preserve 300 W for the forthcoming model
baseline. GPU ownership locks covered both tests, and their processes exited.

| Model-free test, 81920 bytes per call | Mean for 140 calls | Amortized call |
| --- | ---: | ---: |
| NCCL, four processes, captured graph | 4.340 ms | 31.003 us |
| Existing push kernel, 80 blocks, captured graph | 0.913 ms | 6.524 us |
| Existing push kernel, 40 blocks, captured graph | 0.909 ms | 6.495 us |

The push probe directly instantiates the unchanged kernel from this source
base using one host process and four CUDA devices. It validates every output
element on every rank while changing the input markers between graph replays.
It does not validate multi-process IPC setup or model dispatch. The NCCL probe
uses separate processes, checks eager and captured sums, and reports the
critical rank's event interval. These different harnesses are diagnostic
evidence; their difference is not an accepted end-to-end speedup. A single
collective per graph is dominated by rank launch skew and is not used to infer
communication limits.

The approximately 1.6 us needed to serialize 81920 bytes at the reported
two-link rate is only an ideal wire-time floor. Synchronization, protocol,
reduction, and kernel execution are additional costs. The measured push
chain establishes a practical sub-10-us candidate on this topology; it does
not establish the absolute communication lower bound.

## Frozen model baseline, 2026-10-04

The installed profile uses TP4, FP16 transport, E4M3 target KV, automatic
draft KV, Flash-V100 attention, seven probabilistic draft tokens, maximum
length 262144, maximum four sequences, and full/piecewise CUDA Graphs.
Sampling is temperature 0.7, top-p 0.9, checkpoint top-k 20, thinking off,
and 600 output tokens. The retained speed methodology uses `ignore_eos`;
natural stopping and text quality need separate checks before promotion.
All four compute GPUs have a verified 300 W limit. The display GPU is excluded.

The 1024-token fixture is unchanged. The 8192-token fixture takes the prefix
and question suffix of the retained 32768-token input and is checked against
the checkpoint chat template: all four rendered prompts have exactly 8192
tokens. The combined fixture SHA256 is
`d12fad131e8b1807d704c4f974c4749d969367fa667669a500975d9c1c86624f`.

| Unprofiled request | Mean streamed round interval | Emitted tokens per round | TTFT |
| --- | ---: | ---: | ---: |
| 1K, run 1 | 18.486 ms | 2.823 | 354.8 ms |
| 1K, run 2 | 18.382 ms | 3.962 | 354.8 ms |
| 1K, run 3 | 18.407 ms | 2.765 | 354.1 ms |
| 8K, run 1 | 20.198 ms | 3.297 | 2039.3 ms |

The first twenty stream events are excluded. The 1K mean across three
requests is 18.425 ms. These are client-observed complete-round intervals,
not synchronized worker-only graph times. Acceptance varies between requests;
changes need matching seeded comparisons in addition to this speed contract.
Greedy 1K requests measure 16.981–16.989 ms per round.

Each context has four rank traces with sixteen target graphs and fifteen
complete rounds. All rank ledgers match 64 target and five draft layers.
All fifteen rounds include eight compact/seven dense-reference rounds at
1K and eleven compact/four dense-reference rounds at 8K. Excluding the two
edge rounds gives six compact/seven dense and nine compact/four dense,
respectively. These fractions describe the short capture, not all requests.

The logs and kernel ledgers show that all 140 allreduces already use the
SM70 push kernel. Rank-zero collective service is about 1.2 ms per round;
the NCCL microbenchmark improvement must not be counted again as a potential
model gain. Residual/norm remains separate. Target kernel service is about
13.4–13.8 ms, and the five draft layer service sums are about 2.59 ms at 1K
and 2.92 ms at 8K. Profiling overhead makes these diagnostic service sums
different from the unprofiled complete-round interval.

Draft sliding attention launches eight CTAs and takes about 105 us per layer
at 1K and 171 us at 8K. Draft projection and split-K reduction service is
about 334 us per layer. GDN delta-rule service remains about 20 us per layer.
The retained JSON ledgers include every layer's kernel names, duration,
preceding gap, grid, block, and backend on every rank.

The C4 baseline measures 412.40 tok/s at 1K and 387.45 tok/s at 8K in the
common steady decode window. This excludes prefill, each request's first
twenty events, and partial edge intervals, and stops when the first request
finishes.

The first candidate splits the D128 noncausal draft window using the existing
FP32 partial-state machinery. Partitions start at the live window origin so
the captured grid does not grow with historical context. This candidate is
under construction and is not an accepted speedup. Preserve the baseline
artifact and compare exact inputs, graph replay with changing lengths,
model logits, emitted-token behavior, and C4 performance before promotion.

## Candidate decisions and negative evidence

The single-request tie guard incorrectly used request count to choose the
wide probe, while the dense sampler dispatches on logit rows. A q8 request
already meets the eight-row reference contract. The candidate uses that
row count and retains the existing 64-candidate support, vocabulary tie
order, truncation guard, and numerical-boundary fallback. Fifteen focused
GPU unit tests pass against the installed baseline operators: 21/24/63-way
cutoff ties preserve sampled tokens and acceptance counts for the tested
random streams; 64/80-way ties still fall back. This is source unit evidence,
not complete installed-candidate model validation. Head logits are unchanged.

An exact-state GDN geometry probe compares BV8/4/2/1 with one warp and two
two-warp variants, using the production q8/H4/Hv12/K128/V128 shape and strided
QKV rows. Single-request BV8 measures 18.575 us versus 14.118 us for BV2 in a
140-node graph, with bitwise identical output and all stored state snapshots.
At four requests, BV8/4 are about 38.4 us and BV2/1 regress to 41.7/43.2 us.
Two-warp variants both regress and change state bits; reject them. Only the
single-request FP32-state/precomputed-gating contract is a candidate for BV2.
Sixteen focused packed-verifier tests now pass against the source wrapper,
including FP16/FP32 state, q3/q7 draft counts, single/multiple requests and
the runtime bridge. Installed combined-model validation remains open.

NCU confirms the exact shared-layout M8 NVFP4 gate/up and down kernels. The
gate/up grid is 136 blocks of 512 threads; down has 160 blocks. DRAM throughput
is about 63% of the measured peak, with 0.49–0.51 issued warps per scheduler
per cycle. Roughly half the sampled stall cycles wait on L1/TEX scoreboard
dependencies; theoretical occupancy is limited to 50% by registers and shared
memory. This does not establish unpacking ALU as the sole bottleneck. Compare
exact decoding and software weight prefetch before choosing a change. These
are profiler counters with uncontrolled caches/clocks, not accepted speed deltas.

The first root profiler invocation could not truncate user-owned lock files
in the protected temporary directory. Keep ownership locks in a user parent
process while invoking the privileged profiler child; do not alter shared lock
permissions. The corrected counter captures complete. GPU probes wait when
another task owns the shared GPU lock; a busy lock is not a numerical failure.

## Source-complete sliding-window experiment

The isolated sliding-window wheel builds and passes the clean-artifact and
dependency checks. Its SHA256 is
`9fa9cc1a30f611238958532092732612eb0bc60104777802b3cd2bd86830c4ec`.
Five GPU tests pass, including captured replays with live sequence lengths,
permuted physical pages, heterogeneous batches, split boundaries and 262144
tokens. Thirty seeded operator cases have maximum absolute difference
0.00012207 against retained baseline outputs. Full-model KL is still required.

| Diagnostic | Baseline | Sliding split |
| --- | ---: | ---: |
| B1 1K attention, 140-node graph average | 82.56 us | 45.97 us |
| B1 8K attention, 140-node graph average | 154.02 us | 58.23 us |
| B1 1K draft layer service, Torch profile | 2.588 ms | 2.270 ms |
| B1 8K draft layer service, Torch profile | 2.921 ms | 2.350 ms |
| 1K complete interval, three unseeded requests | 18.425 ms | 18.041 ms |
| 8K complete interval, one unseeded request | 20.198 ms | 19.639 ms |
| C4 1K shared steady window | 412.40 tok/s | 430.71 tok/s |
| C4 8K shared steady window | 387.45 tok/s | 374.81 tok/s |

The short captures contain ten compact/five reference rounds at 1K and
thirteen compact/two reference rounds at 8K. Greedy output token sequences
match all three retained 1K fixtures, but some round boundaries change.
Temperature-0.7 outputs and emitted-token counts differ, and the unseeded
8K C4 result declines 3.3%. These observations do not admit the optimization.
The benchmark drivers now accept fixed request seeds; C4 gives each request
a stable distinct seed. Matching control/candidate results and numerical gates
must resolve acceptance and concurrent throughput before promotion.

The existing TP4 pull allreduce plus Gemma norm passes output/residual checks,
but measures 22.009 us against 10.713 us for the actual push plus separate norm
in the same four-process 140-node harness. Do not register that existing pull
fusion for decode. The first harness passed an already incremented Gemma
weight to a helper that increments it; preserve that harness failure separately
from the corrected passing check.

A research-only FP16 M8 projection kernel retains checkpoint values and uses
FP32 local partials. Original-shard microbenchmarks suggest about 0.45 ms across
five draft layers; software prefetch does not improve the chosen geometries.
Small operator error is insufficient: earlier draft GEMM experiments changed
proposal probabilities despite improving local reference error. Require a
fixed-prefix full-head and selector comparison before model timing. Source
integration and native artifact tests are in progress; no private extension
is eligible as a serving dependency or performance result.
