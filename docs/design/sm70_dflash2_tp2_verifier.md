# DFlash2 TP2 verification cost

## Scope and frozen baseline

The TP2 campaign targets approximately 25 ms per complete B1/q8 DFlash2 round
on two rear V100-SXM2-32GB GPUs. A round includes target, logits/sampling,
state handling, context work and draft. TP4 optimization is a separate campaign.

Integration base: `e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`.
Use the QUASAR Qwen3.8-27B NVFP4 checkpoint at
`d8e6fbfa3e3a78899b440222b827430045a05b44` and the DFlash2 checkpoint at
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. The workload uses Python 3.12.13,
Torch 2.10.0+cu128, CUDA 12.8, TP2 on physical GPUs 4 and 7, FP16 activations,
E4M3 target KV, FP16 draft KV and FP32 logits. Both attention backends are
FLASH_ATTN_V100. Keep V2 runner, target/draft CUDA Graphs, context pipeline and
context KV graph enabled. Maximum context is 262144, batch-token budget 4096,
maximum sequences 4 and memory utilization 0.8. Only one request is active.

Sampling remains temperature 1, top-k 20, top-p 0.95, xhigh thinking, natural
EOS and at most 1024 output tokens. The release1k fixture uses seed 20260925
and 1019 input tokens; MBPP28 uses seed 0 and 135 input tokens. Startup and
model preparation are outside decode timing. The original baseline uses
frozen copies of existing native libraries; it is not a rebuild of all main
sources. Retained runtime manifests hash the actual mapped worker libraries.

One startup, one warmup and five measured requests per fixture gave:

| Metric | release1k | MBPP28 |
| --- | ---: | ---: |
| Median request-average complete round, ms | 44.973 | 35.119 |
| Median pure decode, tokens/s | 67.832 | 127.662 |
| Median warm TTFT, ms | 575.387 | 147.247 |
| Accepted drafts per round | 2.063291 | 3.500000 |
| Emitted tokens per round | 3.063291 | 4.500000 |
| Output tokens | 242 | 270 |
| Draft rounds | 79 | 60 |

Outputs repeat within this startup and finish naturally. MBPP28 passes its
three supplied assertions. These are short-context baselines, not a 256K
latency result or the three-startup final acceptance gate.

## Trace and first optimization

Ten steady rounds from both ranks show approximately 35.260 ms of target GPU
service, including 14.963 ms of TurboMind projections and 12.838 ms of scalar
attention. Draft GPU service is 6.638 ms; target head/sampling is 2.274 ms.
The profiled critical-rank round interval is 47.057 ms. Service sums and
profiled wall intervals are diagnostic, not unprofiled performance claims.

TP2's 12 query heads and two KV heads do not enter the existing six-head,
single-KV-head E4M3 grouped route. Its scalar attention uses 1024-token
partitions, FP32 partial output and FP32 partition statistics. The observed
launch is `(8, 12, 256)` CTAs, 256 threads/CTA, 40 registers/thread and
12880 bytes shared memory for the frozen control.

`VLLM_FLASH_V100_TP2_E4M3_SCALAR_FAST=1` selects an experimental specialization
only for q shape `[8,12,256]`, E4M3 KV with two heads, FP32 partial storage,
1024-token partitions and full attention without an anchored window. It is
off by default. Unverified shapes use the original route. A requested matching
route rejects a stale native library instead of silently reporting success.

The specialization constructs normal E4M3 values directly in FP32 bit fields,
retains the original signed zeros, subnormals and NaN payload, and unrolls the
PV loop by eight. Each output still follows the original ascending-token FMA
chain. It retains partition boundaries, score reductions, FP32 intermediate
storage, output rounding, KV scales and the original final reduction kernel.
The native launch counter proves host dispatch, including capture-time calls;
it does not count CUDA Graph replays or model rounds.

## Evidence and promotion status

The initial isolated implementation passes:

- All 256 E4M3 byte encodings, with bitwise equality against the original
  decoder, including both signed zeros and both NaN encodings.
- 33 fixed-operand comparisons across bit conversion alone and PV unroll
  factors four/eight. Final outputs and valid partial output/max/sum bits
  match the frozen native implementation. Lengths include zero, partition
  and page boundaries, 65537 and 262144; live CUDA Graph inputs change between
  replays.
- At length 3297, all variants retain the same FP64-reference error:
  maximum absolute `3.0444386e-5`, p99 absolute `1.4819749e-5`, relative L2
  `2.0301283e-4`.
- CUDA 12.8 Compute Sanitizer memcheck and racecheck on the winning u8
  partition kernel report zero errors and zero hazards, respectively.
- Live same-call shadow comparison on both model ranks: 27072 attention
  calls, 665321472 output elements, zero bit differences and zero nonfinite
  outputs. The original result drives generation. These runs contain
  diagnostic work and are excluded from speed evidence.

For sixteen distinct KV layer working sets, the operator median is 12.712 ms
for the frozen scalar implementation and 3.892 ms for exact bit conversion
with PV unroll eight. Conversion alone and unroll four are approximately
6.335/6.314 ms. These are attention operator results, not complete rounds.
The private u8 implementation is pinned by SHA256
`696545418c6dae261f0bc6a3a530b34464d040de8e404a3069cfd8c2a7762ad3`.
The integrated native build is separately pinned by SHA256
`f916e9e370eeb8d865b4de9d8b64f6e66d8831c0b4458dbe3087141dbadc1d19`;
it passes 14 native tests and supplies the fast partition function in the
paired model comparison below. Its retained build-source snapshot precedes
the final changed-line formatting pass; the manifest hashes the actual
as-built source and library.

The first contemporaneous control reproduces round cost at 44.986/35.075 ms.
Its release1k trajectory has 349 tokens rather than the original startup's
242, while within-startup repetitions match. The candidate was disabled in
this control. This pre-existing startup variation is not an allowed quality
tolerance; cross-startup token/acceptance comparisons must retain this limit.
The first separate-startup candidate measures 35.921/32.706 ms, with
release1k/MBPP28 outputs of 283/297 tokens. Its corresponding control produces
349/270 tokens. The trajectories and acceptance counts differ, so the
approximately 20.15%/6.75% latency reductions are provisional performance
observations, not accepted quality-preserving gains. A within-startup graph
comparison now keeps prefill and projection choices fixed and isolates the
attention change.

The integrated native build passes 14 tests, including exhaustive byte
decoding, 262144-token graph replay, FP64 reference, unsupported-shape fallback
and stale-library rejection.

### Three-startup paired attention result

After warmup, each of three independent services alternates five A/B pairs
per fixture. Between quiescent requests, a diagnostic CUDA driver API helper
changes only the executable graph function for the sixteen scalar partition
nodes on each rank. Node arguments, grid, reducer, buffers, prefill, model
weights and that startup's TurboMind choices stay fixed. The candidate
function comes from the pinned integrated native build. Switching and route
verification happen outside request timing; no profiler or per-round tensor
dump is active. This harness is not a service API change.

| Startup | release1k control / candidate, ms | MBPP28 control / candidate, ms |
| --- | ---: | ---: |
| 3 | 44.808 / 35.823 | 35.051 / 32.694 |
| 4 | 44.897 / 35.797 | 34.953 / 32.678 |
| 5 | 44.872 / 35.930 | 35.318 / 32.861 |
| Median of startup medians | **44.872 / 35.823** | **35.051 / 32.694** |

Each cell is the median of five request-average complete-round costs.
All fifteen pairs per fixture have identical token IDs, acceptance counters
and natural EOS. Cross-startup controls still differ; this experiment isolates
the attention optimization without claiming to fix the existing variation.

Pooled endpoint stream intervals have one interval per round, checked against
the round count. These host-observed intervals include delivery jitter:

| Fixture / mode | Round p50 / p90 / p99, ms | Median TTFT, ms | Median pure decode, tokens/s |
| --- | ---: | ---: | ---: |
| release1k control | 44.772 / 45.159 / 47.196 | 575.877 | 67.947 |
| release1k candidate | 35.789 / 36.063 / 38.027 | 576.364 | 85.219 |
| MBPP28 control | 35.133 / 36.452 / 37.276 | 148.971 | 128.277 |
| MBPP28 candidate | 32.728 / 33.232 / 33.915 | 149.931 | 137.237 |

Acceptance is reported separately from emitted tokens:

| Fixture | Startup | Accepted drafts / round, both modes | Emitted tokens / round, both modes |
| --- | ---: | ---: | ---: |
| release1k | 3 | 2.455446 | 3.455446 |
| release1k | 4 | 2.063291 | 3.063291 |
| release1k | 5 | 2.010638 | 3.010638 |
| MBPP28 | 3, 4 | 3.500000 | 4.500000 |
| MBPP28 | 5 | 3.569231 | 4.569231 |

These paired results admit the attention component for continued experiments.
The approximately 25 ms target, full context sweep and broader quality suite
remain outstanding. The production flag stays off and the PR stays Draft.
Raw reports are `attention-three-start-pair-summary.json`,
`attention-three-start-secondary-metrics.json`, and
`tp2-attention-within-start-{3,4,5}-switch.json` in the campaign results.

### Projection screening and rejected paths

Sixteen real matrices from four adjacent target layers cover all five TP2
physical projection shapes. Inputs are fixed synthetic M8 operands, so these
are operator screens, not live hidden-state or complete-model evidence.
The first QPN2 screen is faster (0.685 versus 0.930 ms per working set) but
increases several FP64-reference error metrics and is rejected for model use.

TurboMind first combines each group scale with the global scale in FP32 and
rounds that effective scale to FP16. Matching this order, the actual selected
split-K count, and its K64 chunk boundaries produces bitwise-identical outputs
on all sixteen tested matrices, with identical FP64-reference errors. Observed
split counts vary across startup tuning, including 14 and 15; the experiment
reads the selected kernel rather than assuming a fixed count. This is not yet
proof that tuning explains the model's cross-startup variation.

The exact prepacked variant measures 0.724 versus 0.891 ms, but duplicates
roughly 5.67 GiB of codes per rank plus scales across the full target. It is
not admitted under the frozen memory/context contract. Reusing the TurboMind
code and effective-scale storage avoids that duplication but is slower:

| Same-working-set comparison | TurboMind, ms | Candidate, ms | Decision |
| --- | ---: | ---: | --- |
| Shared codes, cached loads | 0.891 | 0.940 | Reject |
| Shared codes, streaming loads | 0.896 | 0.921 | Reject |
| Two / four adjacent N tiles | 0.896 | 1.081 / 1.614 | Reject |
| Vector code load plus lane exchange | 0.887 | 0.907 | Reject |
| Effective-scale-only repack, shared codes | 0.883 | 0.978 | Reject |

All these exact variants match the sixteen outputs bit for bit. No slower
variant advances to model testing. The shared layout reader references PR561;
that memory campaign is separate from this attention PR. Expanding the
attention PV unroll from eight to sixteen retains the operator output and
partial-state bits through length 262144 and changes its sixteen-layer median
from 3.888 to 3.652 ms. This small additional gain is not yet a complete-round
result and is not enabled in the published specialization.

### Selective MLP and packed GDN follow-up

Only the 128 MLP projections per rank can retain a fast duplicate layout
without duplicating the full target. Their codes and scales cost 4.482422 GiB
per rank. Explicitly disabling the unused TP4-only QPN8 rerank request avoids
FP16 head packing on TP2; both actual target/draft heads still use the original
FP16 parameters and FP32 dense logits. The shadow startup loads 16.82 GiB/rank,
retains 7.09 GiB of KV and reports capacity for 332993 tokens, above the frozen
262144 maximum context. This is capacity evidence, not long-context latency.

The sixteen-MLP working set across both real TP2 shards improves from 1.203
to 0.963 ms. Four changing-input graph cases per matrix match bitwise.
Memcheck and racecheck pass for every supported split count 1 through 16 plus
32. A complete live shadow covers all 128 MLP projections on each rank:
248068 calls, 22353903616 output elements, zero bit differences and zero
nonfinite outputs. The TurboMind output drives generation. Both fixtures
finish naturally and repeat within that startup; diagnostic timings are
excluded.

One subsequent startup performs five unprofiled A/B pairs per fixture,
switching only marked MLP graph regions between quiescent requests. Both arms
retain the same prefill, attention u8, allocations, selected TurboMind splits
and sampling. Every pair has identical tokens, acceptance and natural EOS:

| Fixture | Control / candidate complete round, ms | Accepted drafts / round | Emitted tokens / round |
| --- | ---: | ---: | ---: |
| release1k | 35.903174 / 35.413639 | 2.063291 | 3.063291 |
| MBPP28 | 32.767878 / 32.293540 | 3.500000 | 4.500000 |

The complete-round benefit is only 0.490/0.474 ms. Do not extrapolate the
approximately 20% projection microbenchmark into a multi-millisecond model
gain. This is one startup, not the final three-startup performance gate.
Raw evidence is `mlp-first-paired-summary.json` and
`tp2-mlp-within-start-2-switch.json`.

The first MLP graph-switch startup stops before its first request because the
diagnostic queries edges of an unrelated graph and receives invalid argument.
The helper now skips unmarked graphs and uses the edge-data-aware driver API,
rejecting non-default dependencies inside a marked region. A bounded gate
covers empty, one-node and two-node unmarked graphs plus twelve real-matrix
switches. The failed startup is retained and contributes no speed evidence.

The packed GDN audit also finds a concrete precision mismatch in its existing
entry: the ordinary speculative path explicitly materializes beta in FP32,
while the packed entry relied on the helper's FP16-input default. On a fixed
TP2 q8 case, the old entry differs from the ordinary FP32-beta path in 5623
output elements and 3142001 state elements, with maximum absolute differences
of 3.8146973e-6 and 1.9565225e-5. The earlier shared-FP16-beta tests therefore
do not admit the actual packed entry.

The entry now explicitly requests FP32 beta. With the same FP32 gating,
the packed subchain matches output and every pool-state bit through all eight
acceptance selectors and changing-input graph replays. Sixteen distinct layer
states measure 0.712 ms for QKV materialization, recurrence and output copy,
versus 0.507 ms for direct packed recurrence. Common convolution and gating
are outside this operator timing. The feature remains disabled pending live
state and full-round validation; this finding is not attributed as the cause
of historical text-quality changes while the packed feature was disabled.
The actual-entry GPU regression passes for both TP2 and TP4 head geometry,
with FP32 state, gaps between pool slots, all eight selectors, two changing
replays per selector and untouched retired rows. Reproduce with
`.venv/bin/python -m pytest --confcutdir=tests/kernels tests/kernels/test_sm70_dflash2_packed_gdn_fp32.py -q`
(two tests passed).

The first live packed-recurrence shadow has no positive coverage and is not
a pass: it assumes state indices `[1,8]`, while the real q8 graph passes a
padded `[8,8]` index buffer and `[8]` selector buffer. The active sequence count
comes from the two-element cumulative-length tensor. Its metadata report is
also written before capture and therefore misses later dispatches. A second
diagnostic correctly slices the active row and observes zero output/state
differences, but its per-layer gate fails: indexing counters by state-pool base
collapses 48 GDN layers into eight shared pool addresses. These diagnostic
failures remain retained; positive coverage must be attributed to actual
layer identity before model admission.

The third shadow attributes calls by the active GDN layer prefix and pool
pointer, and resets its GPU counters after graph capture. The final admission
snapshot covers all 48 layers on each rank: 93792 calls, 2305032192 output
elements and 295044120576 state elements, with zero output/state bit
differences, nonfinite values or unsupported active calls. Original output
and state continue to drive generation. Both natural-stop fixtures complete;
these shadow timings are excluded from performance evidence. The fail-closed
client now requires 48 named, positive-coverage layers per rank.
Evidence: `tp2-gdn-shadow-3-admission.json` and the per-rank shadow reports.

### Updated target and draft attribution

An actual candidate service with selective MLP and exact u8 attention captures
twelve complete rounds; attribution uses the ten interior rounds, on both
ranks. Both warmup and measured requests finish naturally with 242 tokens,
79 rounds and 163 accepted drafts. The profiler wrapper subsequently exits
137 during shutdown. The completed Nsight capture and exported SQLite are
retained, but the job is not recorded as a clean success. No performance
acceptance claim uses this instrumented service.

The same critical-rank wall window closes as follows:

| Diagnostic wall component | Mean, ms |
| --- | ---: |
| Complete round interval | 39.235763 |
| GPU interval union inside that window | 36.186996 |
| Uncovered wall interval | 3.048767 |

Launch-correlated GPU service identifies the next priorities. These service
sums use both ranks and are separate from the wall-clock closure:

| GPU work | Mean service per rank and round, ms |
| --- | ---: |
| Target graph, total | 26.222108 |
| Target QPN2 MLP projections | 8.903925 |
| Remaining target TurboMind projections | 5.881736 |
| Target exact scalar attention | 3.845147 |
| GDN recurrent core | 1.577858 |
| Draft proposal, total | 6.831827 |
| Target head and sampling | 2.270963 |

The QPN2 MLP kernel uses 52 registers/thread with no local-memory allocation
in the native resource dump. Register-cap screens preserve all sixteen real
matrix outputs and their FP64-reference errors, but are slower: matched
0.716544 ms, cap 48 at 0.739584 ms and cap 40 at 0.888576 ms. Both are rejected.
Moving effective-scale conversion to preparation also preserves every tested
output bit but increases the working set from 0.708352 to 0.790528 ms. Reject
it before model work. These are new measured negative results, not evidence
of a changed numerical tolerance. Raw reports are `tp2-mlp-trace.json`,
`tp2-qpn2-register-screen.json` and `tp2-qpn2-effective-scale-screen.json`.

## Reproduction and retained negative results

Build Flash-V100 from this branch with the same CUDA/Torch/compiler flags and
select that module before running the tests. Set `CUDA_VISIBLE_DEVICES` only
to an owned rear GPU, and use private build/compiler caches.

```bash
TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2 .venv/bin/python -m pytest \
  --confcutdir=tests/kernels/attention \
  tests/kernels/attention/test_sm70_tp2_e4m3_scalar_fast.py \
  tests/kernels/attention/test_sm70_e4m3_scalar_fp32.py -q
```

The GPU gate includes an exhaustive decoder comparison, strided output
sentinels, changing page/sequence visibility, graph replay, FP64 reference
and fallback dispatch. The stale-library gate also runs without a GPU.

Task artifacts are retained under campaign identifier
`v100-quasar-dflash2-tp2-25ms-20260908`. They contain baseline contracts,
worker DSO inventories, Nsight data, raw endpoint responses, operator results,
sanitizer logs, source/build hashes and serial GPU queue records. The baseline
campaign identifier is `v100-quasar-dflash2-tp2-baseline-20260908`.

A capped partition-grid experiment passed 66 bitwise cases but did not improve
the sixteen-layer working set: 12.701 ms control, 13.641 ms at cap one and
approximately 12.719 ms at caps two through sixteen. It was rejected before
model testing. Do not repeat that path without new bottleneck evidence.

The first memcheck invocation loaded both experimental u4/u8 DSOs and reported
`CUDA_ERROR_INVALID_HANDLE` in `cuKernelGetFunction` at the second decoder-LUT
launch. The quality gate blocked model work. Running only the winning DSO
passed both sanitizer tools with API error checking retained. The failed
invocation remains recorded rather than counted as a pass.
