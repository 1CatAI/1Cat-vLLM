# Flash-Next QSA and router execution structure on SM70

The historical target service totals are 1.63 ms for QSA/indexing and 1.10 ms
for routing. The corresponding 0.101 ms and 0.140 ms estimates account for
ideal operand traffic at 900 GB/s, not complete operator latency. They exclude
selection, synchronization, dependent reductions, and concurrent work. These
service totals come from a diagnostic trace and are not an additive partition
of the unprofiled 17.4018 ms/round baseline.

## Profile and structural candidates

The QSA total contains scoring (0.540 ms), sparse attention (0.638 ms), merge
(0.133 ms), pre-index work (0.105 ms), selection (0.180 ms), and expansion
(0.034 ms). The paged scorer launches a separate query axis: the observed
M5 grid is `(5, 39, 1)`. Five queries from the same request repeatedly load
the same compressed keys and each pads four indexer heads to an MMA tile.

The router total contains projection (0.714 ms) and selection (0.383 ms).
The projection overlaps the shared-expert branch. One representative layer
has a 13.568 us projection overlapping a 16.224 us shared gate/up, followed
by a 7.616 us selector. Isolating that projection gives about 7.2 us; moving
its overlap to another part of the MoE chain does not automatically save
the difference.

The category label does not cover the entire QSA dependency chain. In the
retained device-KV diagnostic (source `17fa23e57`, Torch 2.10.0+cu128), the
37 central rank-0 target graphs contain twelve serial intervals between
QKV projection completion and output projection start. Their median summed
envelope is 2.5638 ms, including 0.1737 ms between kernels. Each interval has
14 kernels on one stream; no overlap is subtracted twice in this accounting.

| Work inside those intervals | Median service ms/round |
| --- | ---: |
| Query/key preparation outside the QSA custom operation | 0.1059 |
| Indexer projection and split-K reduction | 0.2165 |
| Pre-index cache update | 0.1060 |
| Scoring, top-k and expansion | 0.7368 |
| Main KV write | 0.0957 |
| Protected KV resolution and gather | 0.3331 |
| Sparse attention and partition merge | 0.7942 |

Individual medians need not sum exactly to the median envelope. This is a
profiled device-KV chain, not a replacement for the historical unprofiled
17.4018 ms/round baseline. Direct reads can remove the 0.3331 ms placement
dependency for device history; host history still requires its reader. The
paired model comparison must determine the actual round-time reduction.

The same trace also distinguishes overlap from dependency slack. Across 48
MoE layers, router projection and selection total 0.7101 and 0.3802 ms;
1.0512 ms overlaps shared-expert kernels. However, the routed-expert branch
finishes after the shared branch at every join and depends on routing first.
Removing routing while holding every other observed duration fixed therefore
has an optimistic 1.0906 ms critical-path bound. Subtracting the overlap from
router service would incorrectly make this bound almost zero. Conversely,
the bound cannot predict a speedup because changing the overlap changes
contention, as the complete-chain scheduling screens demonstrate.

QKV preparation/cache write and indexer computation have a different
dependency: both consume the input hidden state and join before attention.
Moving them onto independent streams has an optimistic 0.5299 ms bound across
twelve layers, including 0.2821 ms of QKV projection. This estimate includes
the measured gaps around main-query preparation and excludes additional
stream scheduling or contention costs. A complete-chain screen is required
before model integration.

## Shared-key indexer

A single request's two to eight queries occupy the MMA output dimension
together. A CTA loads a stripe of 32 compressed keys; the four warps each
score eight keys against up to eight queries and four heads. This replaces
independent query grids without changing the stored keys, causal positions,
FP16 operands, FP32 scores, final selector, or sparse-attention computation.
The FP32 K-reduction order can differ slightly from the Triton reference.

The normal CMake extension is selected through `sm70_qsa_shared_key` and
reports both its runtime use and fallback reasons. Admission requires SM70,
FP16 Q and indexer cache, H4/D128, M2..8, and one request. Multiple requests,
M20, prefill, and other layouts keep the existing implementation. Dimension
strides are supported; aligned contiguous dimensions use vector loads.
The capability is enabled by default for admitted shapes; explicit rollback
uses the same KernelConfig field.

## Experiments

All entries below are isolated, same-process CUDA-graph ABBA measurements
on a V100-SXM2-32GB. They do not establish an end-to-end model speedup.
Router projections use 48 distinct weight tensors; QSA uses 12 distinct
layer caches with 2,496 compressed-key capacity, H4/D128, and M5.

| Candidate | Control ms | Candidate ms | Result |
| --- | ---: | ---: | --- |
| CUDA shared-key QSA scoring, 12 layers | 0.2210 | 0.1037 | Continue to model gates |
| Triton grouped-query scorer, first version | 0.215 | 2.222 | Reject: register spills |
| Triton grouped-query scorer, simplified | 0.216 | 1.172 | Reject: still slower |
| Router split-K producers and last-producer join, 48 layers | 0.3475 | 1.4590 | Reject: extra work and joins |
| Router scalar weight reuse across five queries, 48 layers | 0.3478 | 0.3937 | Reject: slower than MMA |
| Router repeated top-10 selection, 48 layers | 0.2031 | 0.2025 | Reject: no useful M5 gain |

The CUDA scorer's prototype maximum score difference was below 5.3e-6.
The split-K router was bit-exact. The scalar router had relative L2 error
1.94e-5 and maximum absolute FP16 output difference 0.001953125. Repeated
top-10 preserved IDs and differed by at most 1.2e-7 in normalized weights.
These numerical results alone do not admit a slower implementation.

Delaying shared-expert work until after router selection was also tested
using complete real-weight MoE chains, including both branches and their
join. Eight chained calls took 0.6337/0.6346 ms for IQ3_S,
0.6256/0.6307 ms for IQ3_XXS, and 0.5898/0.5914 ms for IQ2_S
(control/candidate). Outputs were identical. The later contention cancels
the router's isolated gain, so this scheduling change is not included.

### Router dependency-chain follow-up

The installed M5 projection uses 64 single-warp CTAs. Its fully unrolled
SM70 binary contains 640 HMMA step instructions, 160 vector global loads,
32 shuffles and 24 FP32 adds per CTA, with no local spills. Static instruction
counts are not runtime counters, but they explain why a byte-only estimate
misses a long per-CTA dependency chain and a narrow grid.

A separate research candidate assigns 32 expert rows and 160 K elements
to each of 256 producers. The MMA quad dimension handles different expert
rows rather than duplicating one partition's work. Total MMA work remains
unchanged. A second kernel merges FP32 partials, rounds logits to FP16 and
selects the ten experts. There is no global polling or additional launch
relative to projection followed by selection. The producer compiles to
44 registers without spills; the reducer/selector uses 32 registers without
spills. Reassociated FP32 sums require numerical and model checks.

An independent full-K SIMD reference follows the scheduling idea in
[SGLang's router implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/router.py):
expand the K dimension in one parallel load/reduction instead of repeatedly
waiting on a narrow load loop. This increases issued weight traffic across
query rows, so its latency must be measured with rotating weights and the
shared-expert branch present. SM70 does not use its optional dependent-launch
mechanism. Both follow-up candidates were slower in a 48-layer, M5
projection-plus-selection chain. N32/K160 took 0.5437 ms against 0.5005 ms;
full-K SIMD took 0.5538 ms against 0.5139 ms. Neither changed selected expert
IDs in the 48 input cases. Maximum relative L2 projection errors were
3.84e-5 and 3.98e-5, respectively. Both are rejected before model integration.

### Packaged scorer and position types

The updated normal extension passed all 16 GPU cases, including changed graph
inputs, invalid pages, ties, strided dimensions, int32/int64 positions,
truncated score width, the unchanged C4 fallback, short-context selector
replay, and negative-position visibility. The last case preserves Triton's
signed integer division toward zero; mathematical floor division differs
for negative padded positions. This does not explain the positive-context
model differences below.

Position dtype must be recorded with the scorer measurements. The prototype
table above uses int32 positions. The packaged benchmark uses int64, matching
the QSA metadata builder. The unchanged Triton reference compiled to a
4,184-byte stack per thread for the contiguous int64 case, compared with no
stack for int32. Thus the packaged M5 result (1.9302 to 0.1158 ms for 12
scorers) cannot be compared directly with the earlier 0.2210 ms control.
The int64 score/selection/expansion chain measured 2.0876 to 0.2741 ms;
M20, which uses the same fallback in both arms, measured 6.6087 to
6.6065 ms. These are isolated measurements, not model latency savings.
The full-model A/B must establish both the actual dispatch and the benefit
under production page geometry and concurrent work.

The device-KV model run resolves an 816-token scheduler page, so the
compressed indexer page contains 204 keys. Offline compilation of that
geometry has no int64 stack spill (254 registers versus 224 for int32).
The 16-key-page result therefore does not establish a 1.8 ms model gain.
The benchmark now defaults to the observed 204-key pages and 12 pages per
request; the earlier micro geometry is reproducible with `--page-size 16
--pages-per-request 156`. Position type and page geometry are included in
each result.

A diagnostic keeps the original int64 metadata and FP32 arithmetic but
narrows only the nonnegative, bounded loop index to int32. All initial and
changed-input graph outputs are exact. Twelve scorers with actual 204-key
pages take 0.2331 to 0.2241 ms at M5 and 0.6233 to 0.5997 ms at M20.
The M5 native scorer takes 0.1154 ms against its paired 0.2325 ms control.
These are scoring-only measurements. The bounded-index change saves only
0.0090/0.0236 ms at model geometry and is not pursued as a structural gain.

### Initial separate-process model comparison

The same installed wheel was tested with only `sm70_qsa_shared_key` changed.
The workload was Flash-Next IQ3_S, FP16 MTP4, TP4 on four full-NVLink
V100-SXM2-32GB GPUs at 1530 MHz, CUDA 12.8, Torch 2.10.0+cu128, and FULL
CUDA graphs. Target history used device-resident E4M3 with an 8192-token
protected FP16 hot region; draft KV and attention staging were FP16. PLE
used disk-backed rows. Model length was 9216, the prefill budget was 512,
maximum sequences were four, and GPU memory utilization was 0.95.

This machine/configuration's control does not replace the historical
17.4018 ms resident-FP16-KV baseline. The native M5 scorer was reported active
during model graph capture.

| Measurement | Control | Shared-key scorer |
| --- | ---: | ---: |
| C1 before GPU observation, ms/round | 19.5093 | 19.5466 |
| C1 after GPU observation, ms/round | 19.9484 | 19.5555 |
| C1 mean of the two unobserved segments, ms/round | 19.7289 | 19.5511 |
| Emitted tokens per C1 round | 4.8857 | 4.8857 |
| C4, ms/round | 43.1158 | 42.8130 |
| Natural-prompt mean draft acceptance | 44.8576% | 45.9460% |

The apparent 0.1778 ms C1 difference is smaller than the control's own
0.4390 ms before/after drift. It is not an established model speedup.
The 37 central event-observed rounds also show no target improvement:

| Rank-0 GPU event envelope, ms/round | Control | Shared-key scorer |
| --- | ---: | ---: |
| Target | 15.3266 | 15.3816 |
| Target to draft | 0.6186 | 0.6074 |
| Four draft steps | 3.3963 | 3.3726 |
| Draft to next target | 0.2191 | 0.2225 |
| Round | 19.5606 | 19.5842 |

These are same-rank event envelopes, not kernel-service sums or aligned
cross-GPU clocks. Subsequent interleaved and production-dispatch checks are
recorded below.

The fixed C1 probe and all four C4 output streams were identical. The eight
600-token natural continuations differed, with first differences between
tokens 5 and 292; all stopped at the token limit, so this run does not prove
natural EOS termination. Paired bootstrap draft-acceptance difference was
+1.0884 percentage points with 95% interval [-1.4020, +3.8124]. Teacher
forcing at 64 matched conditions gave mean KL 8.6279e-4, maximum KL 0.0069709,
and 63/64 matching top-1 results. Repeating those conditions within the
candidate process was bit-exact.

At this stage the distribution difference was unresolved. All teacher contexts
contain fewer than 512 compressed keys, where the selector returns every causal
key without consulting scores. All four ranks' captured computation graphs
and compiled-subgraph keys also match after removing cache-directory names.
Thus attributing these differences to the scorer's FP32 summation order is
not supported. An actual-input boundary comparison is required before
claiming a numerical cause or admitting the path.

### Distributed router candidate

The next research prototype partitions the 512 router weight rows across
TP4 instead of calculating every row on every rank. Each rank keeps its
local ten largest FP16 logits, exchanges those candidates, then selects and
normalizes the global top ten. The global top ten must lie in the union of
the local top tens, including the original lower-expert-ID tie rule. This
reduces projection weight traffic from about 126 MB to 31.5 MB per rank per
48-layer chain. FP32 partial-sum reassociation is measured separately from
the exact selection rule.

Two execution structures were tested: projection followed by a fused
reduction/selection/peer exchange, and one kernel with 64 projection
producers plus five row consumers connected by readiness tags. The latter
removes the producer/consumer launch boundary. Communication uses tagged
double-buffered words on direct NVLink peers, with bounded polling. Both
are research-only. A same-process four-rank graph ABBA measures the slowest
rank across 48 distinct router layers:

| TP4 router structure | Control ms | Candidate ms |
| --- | ---: | ---: |
| Two launches, including local/global selection and LL exchange | 0.5047 | 0.7232 |
| Producer/consumer kernel, including selection and LL exchange | 0.5051 | 0.8448 |

Both are rejected before model integration. Reducing weight bytes does not
offset the added selection, communication and readiness work in either
complete chain. The individual contributions have not been isolated.

Changing-input graph replay preserves expert IDs and source indices on all
four ranks. Maximum projection relative L2 errors against FP64 matmul are
2.1731e-4 for the reference and 2.2663e-4 for the candidate. Reassociation
can move a logit by 0.001953125 and a normalized weight by 7.1239e-5 relative
to the reference. Reconstructing the global candidate FP16 logits confirms
exact IDs and a maximum 5.9605e-8 error in selection/normalization itself.
The first stricter cross-projection weight comparison failed; this separate
projection/selection accounting explains it without claiming bit identity.

### Joint router and shared-expert scheduling

Two further research kernels combine the existing router projection, exact
expert selection, input Q8 quantization and shared gate/up. Shared down and
routed expert projections remain the installed implementations. The first
schedule assigns different CTAs to the independent work and connects router
producers to selectors with readiness tags. The second lets one router warp
and the original eight shared-expert warps execute concurrently inside a CTA,
with a named barrier excluding the router warp. Both retain each projection's
FP32 summation order.

Eight complete MoE calls on real IQ3_S layer-17 weights take 0.6520 to
0.7009 ms for separate CTA roles and 0.6471 to 0.7394 ms for concurrent
in-CTA roles. Activations are synthetic and the last input hits 47 experts;
these measurements are rejection screens, not a model-wide routing sample.
Both schedules are rejected. Fusing independent branches also joins their
completion before the next routed-expert launch, and the fused binary
requires 166/168 registers per thread. The first 99-CTA schedule exceeds
one resident 80-SM wave; the second reduces the grid to 69 CTAs but remains
slower. Resource declarations and the added dependency are known facts;
their individual timing contributions have not been isolated.

For the initial input and four changed-input graph replays, router logits,
expert/source IDs, Q8 activations/intermediates, and shared outputs are exact.
Routing normalization differs by at most 1.4901e-8; the maximum routed-output
relative L2 difference is 1.1673e-5. Neither candidate reaches model testing.

### Sparse attention concurrency

The existing grouped page4 entry uses `grid(1, 1, num_groups)`, and its
partial kernel fixes `active_splits=1` for sparse attention. Padding M5 to
its eight-query contract therefore gives one active attention CTA. The
[earlier grouped/XQA screen](https://github.com/1CatAI/1Cat-vLLM/pull/398)
rejected this entry at the verifier shape. Its result does not evaluate
shared-query KV reuse with parallel context stripes.

A new research screen retains those sparse masks and the existing grouped
MMA computation, divides the selected KV union across context stripes and
merges FP32 partitions. The first timing excludes union planning and the
protected hot/cold KV reader. This optimistic bound must show enough benefit
to pay those costs before adapting the full chain.

With M5/H6/D256, 816-token pages, 2,051 selected columns and about 82% shared
selected pages, twelve distinct layer caches take 0.5969 to 0.4355 ms,
including the FP32 partition merge. The 0.1613 ms difference excludes CPU
union planning and protected-reader adaptation; it is not a full-chain gain.
Initial and two changed-input graph replays pass a direct FP64 attention
oracle. Maximum relative L2 errors are 2.918e-4 for the control and 2.953e-4
for the candidate; maximum candidate/control absolute difference is 3.052e-5.

A second implementation maps the 30 live query/head rows into four MMA884
warps, replacing the eight-query template. It is slower than the original
striped implementation: the twelve-layer chain takes 0.5712 ms with an
explicit V transpose and 0.5053 ms with row-major PV operands. Both pass the
same numerical checks. The row-major version eliminates the transpose using
the documented [PTX MMA884 operand mapping](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16);
it does not change operand precision. Neither compact implementation is
selected, and no additional tile-parameter sweep is justified by these gains.

### Same-process model localization

Fresh-process QSA comparisons cannot uniquely attribute the recorded teacher
differences to native scoring: all their short teacher contexts select every
compressed key, independent of scores. A related HCX experiment also reports
a same-wheel reference/reference cross-start difference while its same-process
reference recapture is exact. This is new evidence for a more controlled
comparison, not evidence that either QSA candidate preserves model quality.

A diagnostic therefore retains one loaded target and its original draft
graphs, recaptures M5/M20 target graphs and compares original, recaptured
control, scorer-only, direct FP32-probability attention, both changes and
the original graphs again. The direct-attention operator is already shipped
in the installed extension; the diagnostic selects it explicitly instead of
loading a research library. It separately records teacher distributions,
eight natural continuations, acceptance and symmetric C1/C4 graph ablations.
Timing is skipped if either same-process control comparison fails.

The final signed-visibility-fix wheel completed this comparison. Original,
recaptured-control and original-again graphs are identical at all 64 teacher
positions, all eight 600-token natural continuations and all acceptance
counters. Scorer-only has the same exact result. This resolves the scorer's
quality attribution within this process; the earlier cross-start difference
does not establish a scorer error.

Direct attention, alone or with the scorer, has mean teacher KL 0.00052268,
maximum 0.00948088 and top-1 agreement 63/64. All eight natural continuations
diverge. Their paired draft-acceptance difference is -0.0251 percentage points,
with 95% bootstrap interval [-1.7921, +2.0874]. This finds no clear acceptance
drop but does not prove strict equivalence. The writer updates hot copies
from decoded encoded history, so this comparison uses the same stored KV
precision; attention partitioning and probability arithmetic differ.

Each timing entry below is one same-loaded-model ABBA, using the installed
wheel, TP4 V100, device E4M3 history, protected FP16 hot cache, disk PLE and
unchanged FP16 MTP4 draft graphs. C1 uses I8192/O256 and C4 I128/O600. These
controls do not replace the historical resident-FP16-KV 17.4018 ms baseline.

| Target change | C1 control/candidate ms | C4 control/candidate ms |
| --- | ---: | ---: |
| Shared-key scorer | 19.5569 / 18.9428 | 38.8645 / 39.0130 |
| Direct FP32-probability attention | 19.3522 / 18.7608 | 38.9607 / 38.4102 |
| Both | 19.4232 / 18.3879 | 38.7403 / 38.5621 |

C1 outputs are identical in every arm, with 4.8857 tokens/round. C4 outputs
remain exact for scorer-only; direct attention changes the continuation,
with 2.3248/2.3263 emitted tokens per request per round. C1 controls drift
0.6087, 0.7236 and 0.8381 ms between ABBA endpoints. Thus the combined
1.0353 ms mean difference is provisional, not an admitted model saving.
The diagnostic below alternates graphs inside a single request in symmetric
eight-round blocks and records target/draft GPU envelopes to localize drift.
This measured diagnostic remains separate from ordinary acceptance timing.

## Validation and admission

The packaged scorer passes causal masks, invalid pages, ties, sliced
dimensions, both position integer types, changing inputs during graph
replay, and the M20 fallback. The same-loaded-model comparison resolves
the earlier scorer numerical attribution: teacher logits, natural outputs
and acceptance counters are identical to control. The direct-history
reader has a different FP32 probability contract, measured separately.

Admission uses the KernelConfig capability and loaded shape checks. Both
capabilities are enabled by default for admitted shapes. Host history retains
its protected reader, and native device-history attention excludes draft
owners. The model ablation uses ordinary installed production dispatch and
compares C1/C4 ms per round, emitted tokens, teacher distributions, natural
completion and paired acceptance. Timing drift is retained in the result;
isolated service or GPU-event savings are not substituted for round latency.

## External references

[DeepSeek TileKernels top-k](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/moe/topk_gate_kernel.py)
provides a useful stable-tie contract for small expert selection.
[FlashInfer selection](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/topk.cuh)
provides alternative radix selection and synchronization strategies. Their
algorithmic ideas require measurements at this workload's 512 experts and
five query rows; reducing comparison count alone did not improve M5 here.

[Cohere's decode megakernel](https://cohere.com/blog/megakernels) also separates
ready work at tile granularity and uses named worker barriers. Its overlap
depends on the actual model dependencies and worker resource contract. The
joint-router screens above evaluate those constraints on SM70 while keeping
the existing arithmetic; reducing launch count alone was insufficient.

## Interleaved graph attribution and integration

A separate request-internal diagnostic compares eight-round ABBA/BAAB blocks
using one loaded model and each rank's own GPU event clock. Each complete
32-round block contributes sixteen rounds per arm. C1 uses I8192/O600 and
C4 uses I128/O600. All four ranks observe the same scheduling labels.

| Target capability | C1 target GPU savings, ABBA / BAAB ms | C4 target GPU savings, ABBA / BAAB ms |
| --- | ---: | ---: |
| Shared-key scorer | 0.4143 / 0.3587 | 0.0886 / 0.1573 |
| Native device-history reader | 0.4309 / 0.6561 | 1.3345 / 0.1347 |
| Both | 0.7938 / 1.2263 | 0.4263 / 0.3551 |

These rank-0 figures are instrumented GPU envelopes, not unobserved acceptance
ms/round. Other ranks agree, and changes outside target are small. M20 scorer
fallback has no distinct kernel optimization; its approximately 0.1-ms
variation is an estimate of schedule noise. All C1 sequences and the scorer's
C4 sequences match control. The native reader retains FP32 probabilities, so
its C4 and natural continuations may differ; numerical and acceptance results
are recorded above. The historical 17.401789-ms baseline is unchanged.

Production dispatch now calls the packaged native reader and preallocates
both per-split maxima and denominators before capture. Draft attention is
excluded from the new path. The production benchmark changes only capability
state before capturing each target graph; it does not replace attention
functions or load private extensions. It also checks three short natural
completions in addition to the eight-prompt acceptance and teacher forcing.

## Installed production-dispatch validation

The final comparison uses runtime source `eca6e737038e6136b08a787ecd67ab0a2f97ecdb`
and benchmark revision `eb552a12a849934196860c8252550191ab007182`. The normally
installed wheel is `1.5.2.dev1434+geca6e7370.cu128`, with SHA256
`dd24b1c100b484653149081cb26725077ebd4d3b7292b987daefeb0424156861`.
The integrated GPU suite passes 137 cases, including all 16 scorer and 22
direct-history cases, host-history checks and four-rank HCX graph replay.
The benchmark also verifies that draft-owner policy survives target-worker
cache binding and cannot admit the native target reader for the draft.

The same loaded model reproduces all 64 teacher logits, eight natural
outputs and eight acceptance counters exactly after control recapture and
after restoring the original graphs. With both optimizations enabled,
teacher mean/max KL is 0.000451698/0.004927856 and top-1 agrees at 62/64
positions. Natural continuations differ, consistent with the native reader's
FP32 probability contract. The three short completion checks stop normally
and produce coherent Chinese/English answers in every arm.

The two teacher top-1 disagreements exchange the leading two choices: the
control logits are tied in one case and differ by 0.03125 in the other.
The largest absolute-logit-error case retains its top-1, whose probability
changes from 0.998777 to 0.997708. These observations characterize the measured
error; they do not make the changed continuations bit-exact.

Mean draft acceptance is 44.7423% for control and 46.1643% with both changes.
The paired difference is +1.4219 percentage points, with 95% prompt-bootstrap
interval [-0.7449, +3.5926]. No clear decrease is observed; this interval does
not establish equivalence or a positive acceptance change.

Each timing entry is one uninstrumented, same-loaded-model ABBA with two
control and two candidate cohorts. Both arms are warmed first. The workload
remains I8192/O256 for C1 and I128/O600 for C4, with device E4M3 history,
FP16 MTP4, TP4, full graphs and disk PLE.

| Target capability | C1 control/candidate ms | C4 control/candidate ms |
| --- | ---: | ---: |
| Shared-key scorer | 19.8219 / 19.4180 | 39.0585 / 39.2229 |
| Native device-history reader | 19.7791 / 19.0889 | 39.4380 / 38.9911 |
| Both | 19.8134 / 18.7204 | 41.1333 / 39.2898 |

C1 savings are 0.4039, 0.6902 and 1.0930 ms respectively; control endpoint
drifts are -0.0154, +0.0967 and -0.0470 ms. All C1 outputs agree, at 4.885714
tokens/round. Scorer-only C4 outputs and 2.334764 tokens/request/round also
agree; M20 uses the same fallback. Its measured 0.1644-ms increase is retained
as an observed difference, not relabeled as an optimization. Native and
combined C4 continuations differ, at 2.326271 tokens/request/round.

C4 control endpoint drifts are -0.0163, +0.8998 and +4.2775 ms. In particular,
the combined comparison's late slowdown makes its 1.8435-ms apparent mean
saving inconclusive. This motivates a C4-only repeat with symmetric ABBA and
BAAB orders; completed quality and C1 checks are not rerun. Historical
17.401789-ms resident-FP16 performance remains a separate baseline.

### Repeated C4 comparison

The same installed wheel is retested with benchmark revision
`905f27355f3801be65994415fabee6ac0b9bd9b9` and `--repeat-c4-only`. One newly
loaded model captures control and combined target graphs, retains the draft
graphs, warms both arms, then runs twelve I128/O600 cohorts in three symmetric
groups. There are no GPU-event observers or additional quality measurements.
The scorer uses its unchanged M20 fallback in both arms.

| Order | Control ms/round | Combined ms/round | Saved ms/round |
| --- | ---: | ---: | ---: |
| ABBA | 38.7835 | 38.5543 | 0.2292 |
| BAAB | 38.9067 | 37.9435 | 0.9632 |
| ABBA | 38.8167 | 38.5644 | 0.2523 |
| Mean of six cohorts per arm | 38.8356 | 38.3541 | 0.4815 |

All three groups improve. Mean per-stream throughput is 60.1208 to 60.6562
tokens/s, including the measured 2.334764 to 2.326271 tokens/request/round
difference. Outputs are identical across all six repetitions within each arm;
the two arms retain their different continuations. Outer endpoint drift is
0.5014, -0.1967 and 0.4391 ms respectively. Thus the repeat supports a modest
C4 benefit, not the earlier drift-inflated 1.8435-ms estimate.

Five-second GPU samples during the twelve measured cohorts report 1530-MHz
SM and 877-MHz memory clocks on all four cards; the highest sampled GPU
temperature is 47 C. Concurrent host samples show no swap-in or swap-out.
Sampling does not exclude shorter system disturbances. The final production
comparison retains the qualified C1 reduction of 1.0930 ms and this separate
C4 repeat; neither is substituted into the historical 17.401789-ms baseline.
