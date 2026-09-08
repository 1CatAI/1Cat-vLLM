# QUASAR + DFlash2 complete-round resource audit, 2026-09-09

The 15-ms goal is not met. The latest unprofiled screen with the actual grouped
attention route is 16.638 ms for release1k and 16.248 ms for MBPP28. Both use
rear GPUs 4–7,
TP4/B1/q8, E4M3 target KV, FP32 logits/state, the frozen model and natural
EOS. One startup pair with five warmups and five measured requests per fixture
does not complete the final performance or quality gates.

## Unprofiled endpoint evidence

All values below summarize the five measured requests; no profiler or tensor
dump is active. Complete-round cost is engine decode time divided by draft
round count, and includes target, sampling, state and draft.

| Fixture / arm | Complete-round mean | Median | p90 | p99 | TTFT median | Pure decode median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| release1k / control | 17.023 ms | 17.017 ms | 17.034 ms | 17.039 ms | 353.351 ms | 175.00 token/s |
| release1k / candidate | 16.757 ms | 16.761 ms | 16.774 ms | 16.780 ms | 353.295 ms | 177.67 token/s |
| mbpp28 / control | 16.627 ms | 16.625 ms | 16.634 ms | 16.638 ms | 126.470 ms | 292.89 token/s |
| mbpp28 / candidate | 16.491 ms | 16.373 ms | 16.741 ms | 16.919 ms | 116.408 ms | 297.39 token/s |

The p90/p99 columns above describe request-average round cost, not individual
GPU rounds. Five requests are insufficient to establish tail reliability.
Candidate MBPP28 retains a 16.939-ms outlier. All post-request telemetry samples
show 1530/877-MHz SM/memory clocks; those samples do not exclude transient
events during requests. The earlier one-warmup pair is retained separately:
release1k 17.002/17.081 ms, MBPP28 16.560/16.381 ms. No requests were discarded.

Measured tokens, natural EOS and acceptance match bytewise across arms:

| Fixture | Output tokens | Rounds | Accepted drafts | Accepted drafts / round | Emitted tokens / round |
| --- | ---: | ---: | ---: | ---: | ---: |
| release1k | 272 | 91 | 181 | 1.989011 | 2.989011 |
| MBPP28 | 634 | 130 | 504 | 3.876923 | 4.876923 |

See `results/v4-sparse-dense-order-warm5-ab.json` and its four hashed input
reports. The primitive passes 54 boundary cases, native memcheck and 192 real
four-rank input comparisons. The complete four-rank fixed-prefix pair now passes: 144 records per arm,
no captured intermediate differences, all native logits byte-equal, TV zero
and no top-p support or top-1 changes. This includes the captured layer 0/1
conv/SSM state and metadata; it is not an all-layer operator oracle. See
`results/v4-sparse-dense-order-audit-comparison.json`.

## Whole-round trace closure

The node trace contains twelve complete four-rank rounds; discard the edge
rounds and analyze rounds 9–18. Select the longest worker interval in each
round, then close that same rank with GPU event union plus uncovered time.

| Same critical rank / round | Mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Worker round interval | 18.651 ms | 18.471 ms | 18.809 ms | 19.922 ms |
| GPU event union | 16.774 ms | 16.639 ms | 16.934 ms | 17.949 ms |
| Time without GPU events | 1.876 ms | 1.859 ms | 2.038 ms | 2.114 ms |

GPU activity covers 89.94% of this instrumented interval. This measures the
presence of GPU work, not achieved SM occupancy, issue rate, Tensor Core use
or HBM efficiency. NCU counters are unavailable. Profiled gaps and collective
waiting are not directly recoverable latency. These values do not replace the
16.761-ms unprofiled endpoint median.

| Phase | Mean GPU service per rank | Mean GPU envelope per rank | Kernel calls / rank / round |
| --- | ---: | ---: | ---: |
| target_graph | 12.295 ms | 12.762 ms | 952 |
| target_head_sampling | 0.543 ms | 0.908 ms | 27 |
| request_state | 0.013 ms | 0.115 ms | 3 |
| draft_propose | 3.556 ms | 3.960 ms | 193 |
| input_metadata | 0.057 ms | 0.107 ms | 14 |
| context_and_output | 0.230 ms | 5.664 ms | 13 |

Context/output work is interleaved with sampling and draft; its envelope
spans those phases. Do not sum phase envelopes or compare independent rank
maxima as a single critical path. Native memcpy/memset events are included in
service, while the call count column counts kernels.

## Large costs and weak launch parallelism

These are observed launch resources. Grid counts constrain work distribution
but do not establish achieved occupancy or a particular stall reason.

| Work | GPU service / rank / round | Observed launch | Implication / next bounded step |
| --- | ---: | --- | --- |
| QPN2 gate/up | 2.963 ms | 136 CTAs, 512 threads, 64 registers, 16 KiB shared | Largest individual family; retain HMMA chains and test only loading/layout ideas supported by real-weight working sets. |
| Published QPN2 row projections | 2.889 ms | 160 CTAs; 256/512 threads; 48 registers | Preserve rank reduction order and epoch lifetime. Earlier bounded overlap and cooperative consumers were slower. |
| Other QPN2 columns | 1.489 ms | 112/129 CTAs, 512 threads, 64 registers | Limited grid alongside finite register residency; cap64 is active. Tile/chain changes need separate error gates. |
| Draft dense projections/reductions | 1.834 ms | Main WMMA kernel uses 32-thread CTAs; common grids have 320 CTAs | Small-row GEMM work is spread over few warps per SM. Earlier arithmetic candidate changed acceptance and remains off. |
| Target grouped attention | 0.951 ms | Original partial: 80 CTAs, 512 threads, 128 registers, 56832-byte shared | Correct the inactive experiment binding, then verify 240-CTA/256-thread candidate in the actual replay. |
| Target normalization/residual | 0.835 ms | Dominant fused Gemma kernel has 8 CTAs, 256 threads | Small grid and many dependent launches. Prior direct residual stores were slower; no new fusion benefit assumed. |
| Draft attention | 0.490 ms | 8 CTAs, 512 threads, 97920-byte shared | At most 8 of 80 SMs receive a CTA per invocation. Investigate output-work partitioning without changing QK/softmax order; changing KV splits is arithmetic. |
| GDN convolution | 0.283 ms | 10 CTAs, 128 threads | Small work per invocation; fusion must retain each token state and rollback boundaries. |
| Target KV write | 0.231 ms | 8 CTAs, 32 threads | Compare direct producer layout only with exact cache/slot checks. |

QPN2 totals 7.342 ms of service. This remains the main performance target;
resource-thin attention and small kernels are complementary opportunities,
not a claim that their service time can all be removed.

## Host gaps and quality-sensitive decisions

The same critical-rank gap closure assigns 0.468 ms per round to gaps between
target-graph nodes (the largest individual such gap is only 0.001344 ms),
0.422 ms inside draft, 0.388 ms inside target sampling, and 0.208 ms between
state handling and draft. Numerous short node gaps cannot be treated as one
large idle segment. The largest sampling gap lies between the probe memcpy
and sparse rejection: usually about 0.27–0.31 ms, with a 0.461-ms sample.

The source copies the 21-candidate probe to CPU and checks top-20 cutoff ties,
ties crossing the nucleus and FP32 CDF proximity before selecting compact or
full-vocabulary rejection. Keep this guard and fallback. Eliminating its wait
requires preserving the decision and dependent RNG/acceptance state; simply
removing the CPU branch is not an admissible optimization.

## Route correction and evidence limits

The compact FP32 collector appears once per target and once per draft round
on every rank (80 calls across the forty analyzed rank-rounds). Its final
native sorter is also present. That proves active target/draft dispatch.

The head-regrouping hook instead patched top-level `flash_attn_v100_cuda`.
The model interface calls `flash_attn_v100.flash_attn_v100_cuda`. Both resolve
to DSO SHA256 `a751fed902279b0de23537c4aad2dc4fee360146d7fce7ef0c4f255a77f48b02`,
but CPU identity checks prove separate module objects and function bindings.
No regrouped capture marker or 240-CTA launch is present. Withdraw the earlier
head-regrouping speed attribution and its model-level candidate quality claim;
keep the raw measurements and isolated operator gates.

`benchmarks/kernels/sm70_grouped_attention_candidate_route.py` now resolves the
actual native object through the interface. It installs only when explicitly
called by an experiment and delegates non-q8/eager calls to the original.
The corrected route is now proven in ten steady rounds across four ranks:
640 grouped partial kernels use 240 CTAs and 256 threads. Their measured
launch footprint is 234 registers/thread and 30464-byte shared memory per CTA.
Thus more CTAs do not by themselves establish better achieved occupancy;
register pressure remains a constraint. Grouped attention service changes
from 0.951279 to 0.909762 ms in the two diagnostic traces. This is not an
unprofiled full-round improvement. The canonical release token IDs and
acceptance remain unchanged in the profiled request. The separate five-warmup
unprofiled pair and four-rank fixed-prefix comparison are now complete.

Raw evidence: `profile/v4-sparse-dense-order-nodes/tp4.{nsys-rep,sqlite}`,
`results/v4-sparse-dense-order-nodes-trace.json`,
`results/v4-sparse-dense-order-resource-trace.json`, and
`results/attention-headsplit-binding-identity.json`. The trace capture and export
completed, but the wrapper then failed its runtime-map ownership-name assertion.
Therefore this trace lacks its own final map manifest; separate unprofiled
four-worker DSO manifests are retained. The corrected-route trace has a separate four-worker/360-library manifest.
Its original client waited on a mismatched ownership-name suffix, so a
corrected client completed the capture against the existing owned service.
Map collection was expanded to verified descendants because Nsight gives
the application a separate process group. After saving the manifest, the
obsolete waiting client was stopped and the wrapper cleaned its service,
exiting 143. The client/capture completed; the wrapper did not exit cleanly.
See `results/attention-bound-profile-harness-recovery.json`,
`results/attention-bound-route-hit.json` and
`results/nsys-v4-attention-bound-nodes-runtime-libraries.json`.

Three independent startup pairs, acceptance non-inferiority and model
long-context gates remain open. No
15-ms result, default promotion, merge or 256K performance claim follows.

## Actual attention quality and unprofiled follow-up

The completed pair keeps five warmups and five measured requests per fixture.
Request-average complete-round medians change 16.797233 -> 16.637915 ms for
release1k and 16.416132 -> 16.248439 ms for MBPP28. Every measured token hash,
natural EOS, accepted-draft count and emitted-token count remains canonical.
This is one startup pair, not the final three-pair gate. The input reports and
their hashes are in `results/v4-attention-bound-warm5-ab.json`.

Both actual-route fixed-prefix jobs exit zero and collect 144 records each.
The comparison finds no captured intermediate differences, all native logits
byte-equal, TV zero, and no support or top-1 changes. The recorded conv/SSM
states and metadata cover layers 0/1, not every layer's operator internals.
Both arms retain mapped-library manifests; candidate capture logs prove the
actual module binding. See `results/v4-attention-bound-audit-comparison.json`.
Completed raw tapes are retired only after lossless archive reconstruction
verifies each of the 144 per-file SHA256 values.

## Two-chunk QPN2 publication screen: rejected

The new private builder partitions the 5120 output columns into two 2560-column
chunks, preserving each output's original dot product and rank reduction.
Each chunk has separate two-epoch storage. Its consumer waits on the local
producer's completion event; the main stream joins both consumers before
dependent work. This does not reuse the rejected pre-producer polling scheme.

Four ranks, sixteen real consecutive-layer projection weights, nine changing
synthetic-input cycles, rank start delays and mixed ordinary-push calls pass
bytewise output comparisons with intact allocation canaries. Seven alternating
working-set measurements give 0.456499 ms for frozen publication, 0.511037 ms
for serial chunks and 0.557527 ms for overlapped chunks. All paired differences
are regressions. Therefore neither candidate gets a model run or a four-chunk
extension; there is no end-to-end speed claim and no default change.

`benchmarks/kernels/build_sm70_qpn2_chunked_candidate.py` and
`benchmarks/kernels/benchmark_sm70_qpn2_chunked.py` reproduce the screen.
The native DSO SHA256 is
`745a2bf88bef7c5bd5284f1f45ebc36575f2cb1a320a5a3a04e6db817e224688`.
The original publisher and communicator remain independently frozen and are
hashed in `results/qpn2-two-chunks-real.json`. Kernel-level race and memory
sanitizer admission is not claimed for this rejected route.

NCU 2022.4.1 exists at `/usr/bin/ncu`, but the driver reports
`RmProfilingAdminOnly: 1` and this task's noninteractive sudo attempt requires
a password. Other campaigns' counters do not establish access for this task;
its occupancy and memory-throughput counter gap remains explicit.

## Context computation behind the target probe: model screen

The explicit benchmark installer defers eligible q8 context preparation until
after the target's 21-candidate probe is copied to preallocated pinned memory.
It records a copy event, submits the original context graph on the original
stream, waits only for the copy, and calls the unchanged CPU cutoff predicate.
Full-vocabulary/structured-output paths flush any pending preparation before
the caller updates request state or proposes drafts. KV stores retain their
acceptance-dependent ordering. There is no additional CUDA compute stream.

The CPU dependency/fallback gate passes 256 predicate inputs, including ties,
and checks missing-guard fallback, unsupported probe layout, prefill and error
cleanup. The actual natural-sampling shadow then checks at least 1280 calls
on each rank: probe bytes, cutoff decisions, staged hidden states and projected
context K/V match. All release/MBPP measured output hashes and acceptance
counts remain canonical. Shadow executes additional reference work and is
not performance evidence. See `results/context-probe-cpu-dispatch.json` and
`results/context-probe-actual-shadow.json`.

The first uninstrumented five-warmup pair measures release1k
17.038700 -> 16.554804 ms and MBPP28 16.767940 -> 16.217768 ms. Its control is
slower than the preceding actual-attention pair; do not attribute that entire
difference to the pipeline. A reversed candidate/control pair is queued to
check startup-order and environment effects. The candidate remains experimental;
this does not clear distribution/state, final performance or long-context gates.

## Draft cuBLAS layout screen: numerical rejection of broad changes

All 400 retained four-rank raw projection controls reproduce bytewise with the
original layout. A column-major weight view changes 300/400 outputs and expands
FP64 reference error in 298 cases. Padding queries to sixteen rows changes
200/400 outputs and expands reference error in 102 cases. Combining both changes
has 300 differences and 298 expanded-error cases. The aggregate working-set
medians 1.458115/1.283830/1.409249/1.355162 ms do not admit these broad routes.

Only `o_proj` with column-major weights and `down_proj` with padded row-major
weights retain byte parity in their respective 100-case subsets. A separate
screen times these shapes with the required input copy included, leaving QKV
and gate/up on the original path. No model route is installed for any layout
candidate. `benchmarks/kernels/benchmark_sm70_draft_f16_layout.py` reproduces the
full numerical screen; `results/draft-f16-layout-real.json` retains every case,
FP64 metric, original snapshot hash and aggregate timing.
