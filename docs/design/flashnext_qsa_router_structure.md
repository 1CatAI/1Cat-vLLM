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
The switch remains disabled until the full-model gates pass.

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

### Full-model comparison

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
cross-GPU clocks. The native scorer is not enabled by default.

The fixed C1 probe and all four C4 output streams were identical. The eight
600-token natural continuations differed, with first differences between
tokens 5 and 292; all stopped at the token limit, so this run does not prove
natural EOS termination. Paired bootstrap draft-acceptance difference was
+1.0884 percentage points with 95% interval [-1.4020, +3.8124]. Teacher
forcing at 64 matched conditions gave mean KL 8.6279e-4, maximum KL 0.0069709,
and 63/64 matching top-1 results. Repeating those conditions within the
candidate process was bit-exact.

The distribution difference is unresolved. All teacher contexts contain
fewer than 512 compressed keys, where the selector returns every causal
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
Timing is skipped if either same-process control comparison fails. Results
are pending; neither path is admitted by this diagnostic design alone.

## Validation and admission

The packaged scorer must pass causal masks, invalid pages, ties, sliced
dimensions, both position integer types, changing inputs during graph
replay, and the M20 fallback. The benchmark compares both scoring alone
and the score/selection/expansion chain. The final gate compares the same
wheel with only `sm70_qsa_shared_key` changed: C1/C4 ms per round, tokens
per round, target teacher-forcing agreement, natural outputs, and acceptance.
The first paired model run above did not establish a gain or resolve the
distribution difference. No model gain is admitted.

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
