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

Revision 1 constructs normal E4M3 values directly in FP32 bit fields. Revision
2 uses an exact FP16 bit expansion followed by FP32 multiplication by 256.
Both retain the original signed zeros, subnormals and NaN payload and unroll
the PV loop by eight. Each output follows the original ascending-token FMA
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
The initial actual-entry GPU regression passes for both TP2 and TP4 head geometry,
with FP32 state, gaps between pool slots, all eight selectors, two changing
replays per selector and untouched retired rows. Reproduce with
`.venv/bin/python -m pytest --confcutdir=tests/kernels tests/kernels/test_sm70_dflash2_packed_gdn_fp32.py -q`
(two initial tests passed; the extended stride suite below contains four).

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

The first paired GDN service is blocked before its first request because no
marked packed regions are captured. Source inspection identifies a layout
gate: the Qwen3.5 QKV view shares its row with Z/b/a and the convolution writes
in place, while the packed verifier requires a contiguous QKV matrix. The
native mixed-QKV kernel already accepts a row stride, but its Python wrapper
copies the input and then passes the logical width as that stride.

The opt-in packed entry now accepts contiguous features with a separate row
stride. Its operator wrapper passes the actual stride, retaining the old
copy fallback for non-unit feature strides. Arithmetic, beta/state precision
and state selectors are unchanged. The extended actual-entry suite passes
four TP2/TP4 contiguous/strided cases, including physical QKV row widths 8256
and 4128, untouched input padding, every acceptance selector, changing graphs,
all state-pool bits and retired rows. A second gate performs 24 graph-arm
switches with padded metadata, the 8256-element QKV row stride and strided
state pools; every output and pool bit matches. CUDA 12.8 memcheck reports
zero errors and racecheck reports zero hazards for that gate. The failed first
paired startup supplies no speed evidence; the corrected route and three
subsequent paired startups are reported below.

### Three-startup paired packed GDN result

The corrected route captures all 48 GDN regions on each rank. The actual q8
QKV view has shape `[8,5120]` and row stride 8256. Three independent startups
each run five alternating A/B pairs per fixture, holding exact attention u8,
selective MLP, prefill, graph buffers and each startup's projection choices
fixed. Only the packed GDN region changes between quiescent requests. Markers
and the inactive arm are disabled before replay; no profiler or tensor dump
runs during these measurements.

| Startup | release1k control / candidate, ms | MBPP28 control / candidate, ms |
| --- | ---: | ---: |
| 2 | 36.487586 / 34.678536 | 33.492413 / 31.710558 |
| 3 | 36.305965 / 34.508011 | 33.299356 / 31.508836 |
| 4 | 36.362439 / 34.528673 | 33.312182 / 31.477274 |
| Median of startup medians | **36.362439 / 34.528673** | **33.312182 / 31.508836** |

Each entry is the median of five request-average complete-round costs. All
15 measured pairs per fixture, and each startup's warmup pair, retain token
IDs, acceptance counters and natural EOS. GDN adds a paired whole-round
benefit of 1.833766/1.803346 ms. The startup controls still vary; this isolates
the GDN change and does not resolve the pre-existing repeatability issue.

| Fixture / mode | Host round p50 / p90 / p99, ms | Median TTFT, ms | Median pure decode, tokens/s |
| --- | ---: | ---: | ---: |
| release1k control | 36.334 / 36.578 / 38.608 | 575.826 | 82.220 |
| release1k candidate | 34.526 / 34.778 / 36.907 | 575.153 | 86.480 |
| MBPP28 control | 33.313 / 33.883 / 34.619 | 147.617 | 144.241 |
| MBPP28 candidate | 31.526 / 32.105 / 32.897 | 147.968 | 152.346 |

Host intervals include endpoint delivery jitter and are checked against the
round count. Accepted drafts and emitted tokens are reported separately in
`gdn-three-start-pair-summary.json` and
`tp2-gdn-within-start-{2,3,4}-switch.json`. Both modes share each startup's
acceptance exactly; MBPP28 accepted drafts per round are 3.845070, 3.569231
and 4.306122, with emitted tokens per round 4.845070, 4.569231 and 5.306122.
The candidate remains above 25 ms on both fixtures. Wider quality/context
gates remain outstanding, so these results do not enable a production default.

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

Further scheduling screens are also rejected. Compiler unroll one/two/eight
measures 0.886784/0.803584/0.732672 ms against 0.711168 ms for the existing
unroll four. Shortening live input/dequant fragments reduces registers from
52 to 48 without local memory, but measures 0.806400 ms against 0.705792 ms.
All sixteen real matrix outputs and FP64-reference errors remain identical.
Improved occupancy potential alone is not a measured speedup.

Changing only weight block placement also loses performance. K-major block
interleaving measures 0.807424 ms and 128-byte N-tile pitch padding 0.728832 ms,
against 0.708608 ms for the existing matched layout. Bounded L2 prefetch eight
groups ahead measures 0.794112 ms against 0.711680 ms. All sixteen tested
matrix outputs and FP64-reference errors remain identical, and padding is
untouched. Reject all three before model testing. These are latency
hypotheses tested by timing; unavailable NCU counters do not establish a
specific stall or cache-bank cause. Reports are
`tp2-qpn2-memory-layout-screen.json` and `tp2-qpn2-prefetch8-screen.json`.

### Existing communication fusion screening

The existing TP2 all-reduce/Gemma RMSNorm fusion is compared with the actual
DFlash2 Triton normalization, using real layer-0 norm weights, FP16 `[8,5120]`
rank inputs and FP32 residuals. Five fixed operand amplitudes run on both
ranks through graphs. Residual bits match, but nonzero cases differ in 1 to 4
normalized FP16 elements. Some FP64 relative-L2 errors also increase slightly;
this is not accepted as a new tolerance.

The fusion is slower on both ranks: 0.021990/0.022349 ms per graph versus
0.018048/0.018278 ms for the existing collective plus DFlash2 norm. These are
operator timings, not whole-round gains. Reject this fusion for the current
TP2 route. The native CUB reduction and current Triton variance reduction do
not share an established arithmetic order. Do not attribute historical text
quality changes to this disabled candidate. Evidence is
`tp2-fused-comm-norm-rank{0,1}.json`.

### Packed-route trace and next projection scope, 2026-09-09

A fresh service uses the actual packed GDN route with exact MLP and attention
u8. Both warmup and measured release1k requests finish naturally with 280
tokens, 96 rounds and 184 accepted drafts. Capture, SQLite export and owned
service shutdown complete with exit zero. Ten interior rounds on both ranks
give a critical-rank interval of 37.306504 ms, GPU union of 35.103202 ms and
uncovered interval of 2.203302 ms. These remain instrumented diagnostics.

| GPU work | Mean service per rank and round, ms |
| --- | ---: |
| Target graph | 25.422060 |
| Target QPN2 MLP | 8.906423 |
| Remaining target TurboMind projections | 5.881097 |
| Target scalar attention | 3.848959 |
| Draft proposal | 6.699200 |
| Target head and sampling | 2.270655 |
| All-phase gather/scatter/copy | 1.012917 |

The packed GDN kernel launches 192 CTAs of 32 threads, with 128 registers per
thread and zero static shared memory. Its mean invocation is 32.665 us. A
separate q8 V-tile screen retains one warp and the original K dimension.
BV16/8/4/2/32 all preserve output and every state bit through eight acceptance
selectors and two changing replays, including strided QKV/state and retired
rows. Sixteen distinct layer-state working sets measure respectively
0.525824/0.455936/0.398592/0.392832/0.793984 ms. BV2 passes memcheck, racecheck
and 24 graph-arm switches with padded metadata. Its subsequent live shadow
covers all 48 GDN layers on each rank: 81120 calls compare 1993605120 output
and 255181455360 state elements with zero bit differences, nonfinite values
or unsupported active calls. The ordinary recurrence supplies the outputs
and states used for generation; diagnostic latency is excluded. Independent
multi-request GDN measurements do not admit this route. Raw evidence is
`tp2-packed-trace.json`,
`tp2-gdn-bv-screen.json`, `gdn-bv2-graph-switch-gate.json` and
`tp2-gdn-bv2-shadow-1-admission.json`.

Three independent startups then each run five BV16/BV2 graph-switch pairs
per fixture, holding the exact MLP projection route, attention u8, prefill
and graph allocations fixed. All fifteen measured pairs and the warmup
pairs retain identical token IDs, acceptance counters and natural EOS.

| Fixture | Startup | BV16 complete round, ms | BV2 complete round, ms |
| --- | ---: | ---: | ---: |
| release1k | 1 | 34.496555 | 34.132393 |
| release1k | 2 | 34.429623 | 34.174315 |
| release1k | 3 | 34.576248 | 34.201821 |
| MBPP28 | 1 | 31.288474 | 30.914687 |
| MBPP28 | 2 | 31.708845 | 31.277764 |
| MBPP28 | 3 | 31.530112 | 31.126483 |

The medians of startup medians improve by 0.322239/0.403629 ms to
34.174315/31.126483 ms. Candidate host-observed round p50/p90/p99 are
34.102/34.415/36.301 ms for release1k and 31.060/31.755/33.735 ms for MBPP28.
Median TTFT is 575.230/148.479 ms and pure decode is 87.902/145.594 tokens/s.
Acceptance and emitted counts are reported separately for each startup in
`gdn-bv2-three-start-pair-summary.json`; cross-startup trajectories still
vary. These are unprofiled paired results, not a 25 ms or broad quality gate.
The source now exposes `VLLM_SM70_DFLASH2_TP2_GDN_BV2`, default off and
dependent on the packed verifier flag. It admits only TP2 q8, H8/HV24,
K/V128, FP16 input/output, FP32 state and precomputed gating, retaining the
original recurrent arithmetic and stage count. Eight actual-entry GPU tests
pass, including the original TP4 BV8 fallback with the new flag requested.
The first test revision incorrectly expected TP4 BV16; every output/state
check passed, and only that launch assertion failed. The corrected fixture
and failed evidence are retained. A subsequent source-integrated model A/B
verifies the constructor flag on all 48 GDN layers per rank and toggles the
source guard. All five pairs and warmup per fixture retain token IDs,
acceptance and natural EOS. Release1k improves from 34.433621 to 34.063787 ms
and MBPP28 from 31.428823 to 31.008482 ms. This confirms the source integration;
it is a separate startup from the preceding three-start cohort.

A separate collective launch-geometry screen keeps the original peer protocol,
rank reduction order and DFlash2 normalization. All tested q7/q8 outputs,
residuals, signed-zero/cancellation inputs and changing graph replays match.
The best q8 median improves by only about 0.33 us per invocation; no model
gain or production setting is established. The first diagnostic stops before
its first replay because the retained graph was not explicitly instantiated;
the corrected run passes and the failed run remains excluded.

Granular real-weight projection timing identifies a smaller next layout
scope: all 64 target output projections and the 16 attention QKV projections.
Together they require 896532480 additional bytes per rank, compared with the
existing MLP layout's 4812963840 bytes. Their real-shard operator gates pass
four changing-input graph cases and canaries; memcheck and racecheck cover
both TP2 shards and all split counts 1 through 16 plus 32. Its first live
shadow startup fails before generation: available KV is 5.14 GiB, below the
5.58 GiB needed for the unchanged 262144 maximum length. Both ranks prepare
208 projections, but there is no live quality or speed result. The static
layout estimate alone did not establish the full runtime memory budget.

The next private candidate instead retains 64 MLP down projections and all
128 non-MLP projections; gate/up stays on TurboMind. This uses 3642163200
layout bytes per rank, less than the previous MLP-only candidate. GDN input
weights/scales retain zero-filled padding from 8240 to 8256 columns, and the
producer output stride is unchanged. Eight real non-MLP shard cases pass
changing-input graphs, full output bits, canaries, memcheck and racecheck
across all supported splits; the MLP down gates are retained. The live shadow
now covers all 192 projections on both ranks: 376904 calls compare
18317493248 output elements with zero bit differences or nonfinite values.
Original outputs drive generation. The observed KV budget is 8870215885 and
8874410189 bytes, about 8.26 GiB per rank, with maximum length 262144 and
memory utilization 0.8 unchanged. The paired speed comparison retains the
same allocation in both arms and switches those 192 projections against
TurboMind; it does not compare separate startups or allocate both complete
layout choices. The source kernels and
layouts remain private experiments. Evidence is
`tp2-balanced-shadow-1-admission.json` and the two rank memory reports.
The first paired startup stops before generation because the diagnostic
counts both the 192-region full graph and prefill piecewise graphs. That
failure is retained. The corrected tool selects exactly one full graph,
checks 192 unique prepared weight pointers and layer prefixes, and leaves
all prefill pieces on the control path. The corrected startup runs five
pairs per fixture with exact tokens, acceptance and natural EOS. Complete
rounds improve from 35.178025 to 33.911462 ms on release1k and 32.107532 to
30.847108 ms on MBPP28 in the first corrected startup. Three independent
startups now pass all fifteen measured pairs and warmup per fixture. The
median of startup medians is 35.163780 to 33.911462 ms for release1k and
32.052971 to 30.844085 ms for MBPP28. Candidate round p50/p90/p99 are
33.878/34.173/36.521 and 30.791/31.315/31.913 ms, respectively. Median warm
TTFT is 578.279/147.689 ms and pure decode is 89.938/147.626 tokens/s.
Both arms retain packed BV16 and attention u8; the control uses TurboMind
for all target projections. This does not establish a paired comparison
with the previous MLP-only layout. Raw per-start acceptance and emitted
counts remain separate in `balanced-three-start-pair-summary.json`.

A subsequent private layout experiment stores one persistent code buffer,
761200640 bytes of extra QPN2 scales and a 44564736-byte shared conversion
workspace per rank. Integer word permutation reconstructs the original
TurboMind prefill layout without changing weights or arithmetic. Live
same-call shadow covers all 256 target projections on each rank: 495818
calls and 35330540544 output elements total, with zero bit differences or
nonfinite values. Original outputs drive generation. The observed KV
budgets are 11804131533/11808325837 bytes, with the same maximum context
and memory utilization. This is quality and memory evidence, not speed.
The first paired startup stops before generation: the V2 full-graph manager
calls its forward function with runtime mode NONE while capturing, so a
FULL-runtime-mode guard misses the route. The revised private harness uses
the existing SM70 decode-graph capture context and retains unique coverage
and split-K checks for all 256 projections. The corrected startup passes all
five paired requests and warmup per fixture: release1k improves from
35.275544 to 33.318339 ms and MBPP28 from 32.274264 to 30.331608 ms.
Candidate warm TTFT is 591.741/166.197 ms. Both arms materialize original
prefill weights into the fixed workspace, so this new prefill cost must
remain visible in TTFT rather than being attributed to decode. Three
independent startups now pass all fifteen measured pairs and warmup per
fixture. The median of startup medians is 35.281057 to 33.318339 ms for
release1k and 32.180410 to 30.175731 ms for MBPP28. Candidate round
p50/p90/p99 are 33.267/33.602/35.439 and 30.195/30.677/31.347 ms;
warm TTFT is 592.478/165.612 ms and pure decode is 90.140/150.135 tokens/s.
Per-start accepted and emitted counts remain separate in
`single-layout-3-start-pair-summary.json`. This does not establish the 25 ms
target.

The full-target QPN2 trace now covers ten interior rounds on both ranks,
with 256 projection calls per rank per round. Profiled critical-rank wall
is 35.615096 ms, GPU union 33.243090 ms and uncovered time 2.372007 ms.
Target service is 23.626618 ms, draft 6.628412 ms, target head/sampling
2.268724 ms and context/output 0.609346 ms. Projection shapes account for:

| Projection | N / K / original split | Calls per round | GPU service, ms |
| --- | --- | ---: | ---: |
| MLP gate/up | 17408 / 5120 / 7 | 64 | 6.091905 |
| MLP down | 5120 / 8704 / 8 | 64 | 2.790530 |
| GDN input | 8256 / 5120 / 9 | 48 | 2.008633 |
| Attention/GDN output | 5120 / 3072 / 9 | 64 | 1.278775 |
| Attention QKV | 7168 / 5120 / 5 | 16 | 0.655133 |

MLP gate/up and down account for approximately 69% of QPN2 service. Draft
dense projections, including its head, cost 5.398522 ms. These are priorities
for further work, not estimates of additive end-to-end savings. Hardware
counters remain unavailable. The first profile attempt fails before
generation because the launcher overrides the requested worker extension.
The corrected capture and request complete with 280 output tokens, 96 rounds
and 184 accepted drafts, matching warmup. Its bounded process cleanup exits
137; the trace is not described as a clean exit-zero benchmark. The retained
report, exported SQLite and interval/route checks admit only diagnostic use.
Evidence is `tp2-single-layout-trace-admission.json`,
`tp2-single-layout-trace.json` and `tp2-single-layout-projection-shapes.json`.

The reproducible private-kernel generator is now checked in as
`benchmarks/kernels/build_sm70_tp2_matched_qpn2.py`. It preserves the tested
CUDA source byte for byte (SHA256
`139ff11214d1fb49062efe1e6d9dc824588e5f2f14439435154d30b43915fc62`),
including the original K64 partition boundaries, effective FP16 scale
rounding, single accumulator chain and ordered partial sum. It generates
an isolated library; it does not install a library or enable a serving route.

The first two conversion sanitizer jobs incorrectly retain a GDN-only
kernel filter. Their zero-error summaries do not establish conversion-kernel
memory or race coverage. Failed admission records and logs remain intact;
the replacement gates explicitly select the conversion/materialization
kernels and retain CUDA API error checking. Both corrected memcheck and
racecheck gates pass twelve real rank/projection cases, conversion in both
directions, original TurboMind M8/M129 consumers, changing graph inputs,
changing layout flags and scratch canaries. The retained native library
SHAs are `7e24b7f014df0060af6ba2e7eb8df88d1839d0cd483e96f1a0990a4954b09657`
for static conversion and
`41c8ad3998f7c826355b5333c17568fa4a617b899b56185debac5c58b7b6f640`
for conversion with dynamic prefill materialization.

An E2M1 register-permutation decoder is also exact on the sixteen-matrix
screen but slower: 0.763648 versus 0.712960 ms. A separate scale-folding probe
initially misreads the half constant `0x5c00` as 64 instead of 256; it changes
1139840 output elements and is rejected before any model use. That diagnostic
failure is retained independently of its corrected probe. Neither decoder
experiment changes repository kernels or production behavior. The corrected
256-factor probe restores all sixteen matrix outputs and FP64-reference
errors, but is slower: 0.774144 versus 0.705024 ms, and is also rejected.

A separate scheduling probe maps each existing logical split to its own
32-thread CTA, then reduces explicit FP32 partials in the original order.
All sixteen real-matrix outputs and FP64-reference errors match, and scratch
canaries remain intact. The working-set median is 0.886016 ms versus
0.711936 ms for the matched QPN2 kernel, so the extra launch/workspace path
is rejected before model use. The result does not establish an instruction
stall diagnosis; hardware performance counters remain unavailable.

An attention address-reuse screen retains the original shared-memory size,
QK/softmax, ascending PV FMA and FP32 partition/reducer. Forty-four cases
across page sizes 1648/3296 preserve output and valid partial/statistic bits
through 262144 tokens, with unchanged FP64-reference errors. It has no
stable working-set gain: 3.921664 versus u8's 3.859456 ms on the actual
1648-token page, and 3.902720 versus 3.944448 ms on page3296. It is not
promoted. Reusing the existing grouped FP32 kernel separately for both TP2
KV heads also fails the strict arithmetic gate: all nine tested cases change
some FP16 outputs and some expand FP64 relative-L2 error. Timing and model
admission are skipped. These results do not establish a text-quality cause.
A separate eight-value PV staging kernel also preserves all forty-four
output/partial/statistic comparisons, but increases the page1648 working
set from 3.858176 to 4.857088 ms and page3296 from 3.941120 to 4.908800 ms.
It is rejected before model use.

An E4M3 decoder probe constructs an exact FP16 bit pattern, converts it to
FP32 and multiplies by 256. It retains signed zeros, reserved-NaN handling,
all original QK/PV arithmetic and FP32 partial/reduction storage. All 256
encodings and forty-four output/partial/statistic cases through 262144
tokens match, with the same FP64-reference errors. The sixteen-layer
working-set median is 3.429376 versus u8's 3.859712 ms on page1648, and
3.465472 versus 3.942912 ms on page3296. The winning-library-only memcheck
and racecheck pass; the earlier two-library memcheck reports the previously
observed `cuKernelGetFunction` invalid-handle error and remains excluded.
Twenty-four alternating graph replacements across six changing sequence
lengths retain output and valid partial/statistic bits. Live model shadow
now covers all sixteen logical attention layers on both ranks: 27040 calls
and 664535040 output elements, with zero bit differences or nonfinite values.
The first coverage check incorrectly equates unique KV base pointers with
logical layers and fails after completing generation. Sixteen layers share
eight KV memory pools, with distinct page-table pointers for the paired
layers. The revised diagnostic attributes each call to the backend layer
name and requires positive coverage for every layer. Original u8 outputs
drive generation. This probe changes the decoder instructions, not the stored
E4M3 KV precision. Its library SHA is
`6dc516f8d629f15b578b0c287257fefa70cf4dc58a50878988642b38129b0cd7`.

Another probe shares KV load/decode between two adjacent query heads in one
CTA while retaining each original scalar head's arithmetic. All forty-four
output/partial/statistic comparisons and FP64-reference errors match, but
the working set regresses from 3.430144 to 4.232704 ms on page1648 and
3.469312 to 4.279040 ms on page3296. It is rejected before sanitizer or
model follow-up. These timings establish a regression, not a measured
hardware-counter explanation of its cause.

The exact FP16-bridge decoder now passes three independent startups with
five alternating pairs after warmup per fixture. Both arms retain balanced
192-projection QPN2, BV16 GDN and original u8 prefill. The median of startup
complete-round medians is 33.895185 to 33.373525 ms on release1k and
30.741620 to 30.641039 ms on MBPP28. All fifteen pairs and warmups match token
IDs, acceptance counters and natural EOS. Candidate round p50/p90/p99 are
33.289/33.683/35.361 and 30.564/31.070/32.148 ms; TTFT is 576.749/148.847 ms
and pure decode 103.063/146.342 tokens/s. Per-start trajectories and accepted
versus emitted counts are retained in `half-bridge-3-start-pair-summary.json`.
This is not a paired comparison with the single-layout campaign.

A K16-major activation-layout candidate also preserves all sixteen real
matrix outputs, FP64-reference errors and three changing graph replays.
The continuous four-layer working set regresses from matched QPN2's
0.712448 to 0.722688 ms, even before input packing is charged. The tested
candidate is rejected before producer or model integration. Its initial
build omits operator registration and fails before GPU comparison; the
corrected build and failure are retained separately. A separate experiment
changes only the executable grid of the unchanged scalar attention kernel.
It preserves outputs and valid partial/statistic bits but saves only about
0.06–0.07 ms across sixteen KV layers. It is not advanced to a model route.

The context/probe scheduling experiment retains the original NumPy top-p
cutoff guard. It copies the FP32 probe to a fixed pinned buffer, records an
event, then launches the original context graph on its original stream.
The CPU guard waits for the probe copy while context work continues. Missing
or unsupported probes flush pending context before state updates. Live
shadow compares hidden states, positions and every projected context K/V
before versus after the target head: 966 context and probe checks per rank
pass. Three independent startups retain all fifteen measured pairs and
warmups per fixture. With balanced 192-projection QPN2 and BV16 fixed,
complete rounds improve from 33.778139 to 33.251763 ms on release1k and
30.696557 to 30.018241 ms on MBPP28. Candidate p50/p90/p99 are
33.176/33.809/35.623 and 30.118/30.981/32.607 ms; TTFT is 579.024/148.915 ms
and pure decode 92.123/149.354 tokens/s. Evidence is
`context-probe-3-start-pair-summary.json`. Composition with the full 256
projection layout, BV2 and the newer decoder requires a separate diagnostic
and unprofiled campaign; these independent gains are not added together.

The combined diagnostic fixes all 256 QPN2 projections and source-integrated
BV2, then switches the exact decoder and context schedule together. Its first
check incorrectly compares the entire sampled-token allocation. All target
hidden states, full local-vocabulary logits, accepted lengths and final
outputs match, but unused sampled-token tails differ. The original sparse
sampler uses `new_empty` and writes only its valid prefix; downstream output
uses `num_sampled`. The failed report is retained without admission.

The corrected diagnostic records each raw differing column and confirms it
is outside the valid prefix. It also fills invalid tails with distinct values
in the two arms before downstream consumers run. All later hidden states,
full local-vocabulary logits, valid sampled tokens, acceptance and natural EOS
still match. The comparison covers 1162 complete-vocabulary rows, represented
by 288547840 FP32 elements across the two rank shards. Each arm poisons
476/139 tail elements per rank on release1k/MBPP28; context comparisons cover
95/50 rounds per rank. This tests tail isolation on these requests and does
not resolve the earlier cross-startup variation. Evidence is
`tp2-combined-shadow-2-admission.json` and `combined-shadow-2-summary.json`.

The first unprofiled combined startup passes all five pairs and warmup per
fixture. With full QPN2/BV2 fixed, adding the private exact decoder and
context overlap changes complete rounds from 33.184675 to 31.896585 ms on
release1k and 30.082788 to 29.146665 ms on MBPP28. Candidate round p50/p90/p99
are 31.863/32.362/34.516 and 29.078/29.677/30.255 ms. Warm TTFT is
593.983/163.962 ms, and pure decode 108.022/153.820 tokens/s. This is one
startup, not the three-startup gate or the approximately 25 ms goal. It is
also not a performance claim for the newly rebuilt native revision 2.

A separate MLP two-accumulator-chain screen preserves weights, scale rounding
and logical K64 splits but changes accumulation order. Both TP2 ranks and
three fixed input amplitudes produce twelve checks of the actual gate/up
and down shapes. Eight checks expand independent FP64-reference error, with
122–604 changed FP16 output elements per case. The arithmetic candidate is
rejected before timing or model integration. See
`qpn2-mlp-two-chain-decision.json`.

Native revision 2 now contains the exact FP16-bridge decoder behind the same
default-off TP2 flag. An explicitly requested matching route rejects revision
1 and older libraries. The isolated build SHA256 is
`9d0fe7186bfe82ccd0b58f0795b9f0b4a70eb7ecf8dc9caf345efb4055752cdf`.
It passes 15 native tests, including both stale-library cases, exhaustive
byte decoding, FP64 reference, changing graphs and unsupported shapes.
Twenty-four replacements of the real kernel function preserve outputs and
valid partial/statistic bits. Memcheck covers 1025, 3297 and 262144 tokens
with zero errors. The 262144-token racecheck reaches the 240-second limit
and is retained as a timeout, not a pass. A bounded racecheck at 1025/3297
tokens completes with zero hazards. Each successful native invocation
records positive fast-path host dispatch counts. The final source includes
a whitespace-only changed-line formatting pass after the build snapshot.
Whole-model admission of this rebuilt library remains separate from the
private decoder's earlier paired performance evidence.

A q8-only QPN2 specialization removes unused row predicates and row offsets
while preserving all dot-product arithmetic. All sixteen real projection
outputs, FP64-reference errors, changing replays and output canaries match;
unsupported row counts are rejected. Compiler register use drops from 52 to
48 per thread, but the working-set median worsens from 0.725504 to 0.776960 ms.
It is rejected before model work. Register count alone is not a performance
result; evidence is `qpn2-static-m8-decision.json`.

### LM-head width and accumulation order

The trace spends approximately 4.133 ms across the target and draft dense
FP32 heads. A bounded probe keeps real TP2 target-head weights and input
rows fixed, then selects 256 vocabulary rows for recomputation. Default
`torch.mm` changes its effective split-K grid from two to sixteen. On M8/M7,
2001/1739 FP32 output elements change, with maximum differences of
6.4820051e-7/6.1839819e-7. The cuBLASLt log requests split 19 for the narrower
matrix; its resulting kernel grid uses sixteen partitions. Do not treat
default narrowed GEMM as bit-exact candidate reranking.

A private probe uses the official cuBLASLt algorithm-selection interface to
retain the complete-head algorithm: ID 21, tile ID 5, stage ID 14, split two,
output-type reduction (scheme 4). It uses the observed Torch workspace limit
of 8519680 bytes. M7 and M8 each pass N64/N256/N1024 with three changing input
amplitudes, with zero FP32 bit differences from the selected complete-head
outputs. The N1024 graph is about 0.032 ms; this excludes candidate selection
and weight gathering and is not a complete-head or model speed result.
Candidate coverage, both actual head inputs, memory safety and full-round
admission remain outstanding. No narrowed LM-head route is enabled.
See `head-cublaslt-probe.json`, `head-lt-plan-probe.json` and the
[CUDA 12.8 cuBLASLt reference](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html).

## Reproduction and retained negative results

Generate the isolated TP2 projection candidate without installing it:

```bash
CUDA_VISIBLE_DEVICES="" TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2 \
  .venv/bin/python benchmarks/kernels/build_sm70_tp2_matched_qpn2.py \
  --output-dir /tmp/tp2-matched-qpn2 --build
```

The manifest records source and library hashes. The tested candidate uses
the observed TurboMind split for each real TP2 q8 projection and one
accumulator chain. Other shapes, split choices or new builds require their
own numerical and model admission.

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
