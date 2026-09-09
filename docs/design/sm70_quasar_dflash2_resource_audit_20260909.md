# QUASAR + DFlash2 complete-round resource audit, 2026-09-09

The 15-ms goal is not met. The latest same-startup unprofiled GDN BV2 isolation
measures 16.280 ms for release1k and 15.873 ms for MBPP28. Both use
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
difference to the pipeline. The reversed candidate completed, but its control was interrupted by a host
reboot and produced no endpoint result. It is not a paired comparison. A fresh
post-reboot pair measures release1k 16.779740 -> 16.591267 ms and MBPP28
16.466058 -> 16.106911 ms, with five warmups and five measured requests per
fixture. Every measured token hash and acceptance count remains canonical. The candidate remains experimental;
this does not clear distribution/state, final performance or long-context gates.

## Draft cuBLAS layout screen: numerical rejection of broad changes

All 400 retained four-rank raw projection controls reproduce bytewise with the
original layout. A column-major weight view changes 300/400 outputs and expands
FP64 reference error in 298 cases. Padding queries to sixteen rows changes
200/400 outputs and expands reference error in 102 cases. Combining both changes
has 300 differences and 298 expanded-error cases. The aggregate working-set
medians 1.458115/1.283830/1.409249/1.355162 ms do not admit these broad routes.

Only `o_proj` with column-major weights and `down_proj` with padded row-major
weights retain byte parity in their respective 100-case subsets. The separate
screen includes the required input copy and leaves QKV and gate/up unchanged.
Its complete twenty-projection working set regresses from 1.451684 ms to
1.485681 ms when combining the byte-equal subsets. Either subset alone also
regresses. These exact-layout routes are rejected before model testing. `benchmarks/kernels/benchmark_sm70_draft_f16_layout.py` reproduces the
full numerical screen; `results/draft-f16-layout-real.json` retains every case,
FP64 metric, original snapshot hash and aggregate timing.

## Post-reboot context trace and closure

The machine rebooted at 2026-09-09 02:34:57 UTC. The old lease and unfinished
reverse-control processes were gone. The user-authorized rear-GPU default
service was stopped, and the task lease was restarted with explicit physical
GPU order 4,5,6,7. Frozen candidate/control DSOs were rehashed. The interrupted
job is retained as `zz240-v4-context-probe-warm5-reverse-control.interrupted.json`;
`host-recovery-20260909.json` records ownership recovery.

The new node trace exits cleanly and retains its own four-worker library
manifest. Its ten steady rounds have critical-rank interval mean 19.553882 ms,
p50 18.468254 ms, GPU union mean 17.426705 ms and uncovered mean 2.127177 ms.
Two roughly 24-ms rounds remain in those aggregates: one has 7.417044 ms of
uncovered time and the next has 22.366826 ms of GPU activity, including waits.
Their cause is not assigned to a source change. The trace is diagnostic and
cannot replace the separate unprofiled results above.

GPU correlation identifies exactly one six-kernel context graph after the
672-byte target probe on each of forty rank-rounds. Its mean service/envelope
is 0.076562/0.082549 ms; mean overlap with the remaining host sampling span is
0.080438 ms. That span includes the unchanged CPU guard and rejection launch
handling, so it is not a pure predicate timer. The roughly 11-ms event wait
includes queued target work, not just probe transfer. QPN2 still totals
7.351498 ms per rank-round. No achieved-occupancy or HBM counter claim is made.

Evidence: `results/v4-context-probe-nodes-trace.json`,
`results/v4-context-probe-nodes-resource-trace.json`,
`results/v4-context-probe-overlap-proof.json`, and
`results/nsys-v4-context-probe-nodes-runtime-libraries.json`.

## Strict draft column GEMM: arithmetic gate passes, model trajectory held

The bounded follow-up disables reduced-precision FP16 GEMM reduction only
while selecting each candidate kernel, then restores the process setting.
The original layout still reproduces all 400 retained controls. Strict column
weights change 300 outputs but expand none of the registered FP64 max, p99 or
relative-L2 errors. Strict padded-row weights still expand three cases and
are rejected. Twenty-projection medians are 1.469460 ms for the original and
1.307750 ms for strict column weights. This is a local arithmetic screen.
The switch follows the documented PyTorch 2.10 reduction control; its effect
here is measured, not a diagnosis inferred solely from the documentation.

The explicit `sm70_draft_column_candidate_route.py` installer selects only the
twenty captured q8 query projections and retains original prefill/context
calls. In the first natural model run, release1k changes 272 -> 248 emitted
tokens, with first token difference at zero-based offset 123; MBPP28 changes
634 -> 357, first differing at offset 194. Accepted drafts per round are
1.989011 -> 2.024390 and 3.876923 -> 4.100000, respectively. These changed
trajectories do not establish acceptance non-inferiority or preserved quality.
The apparent 16.369764/15.902543-ms medians are not admitted performance gains.
The arithmetic route remains closed pending causal distribution/acceptance
and broader quality evidence. No score is used to excuse these differences.

See `results/draft-f16-layout-strict-real.json`,
`results/draft-f16-layout-gated-real.json`, and
`results/draft-column-model-screen.json`. The operator benchmark restores the
original reduction property in a `finally` block; no global serving default
is changed. Official reference:
[PyTorch 2.10 numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html).

## Native FlashInfer fragment draft prototype: rejected for speed

A private B1/H8/q8/D128 FP16 paged prototype reuses the project's native
FlashInfer Volta WMMA fragments with the frozen Flash-V100 K176 and online
softmax schedule. It reduces the query tile to sixteen rows and uses one or
four independent output-column partitions, retaining every output's QK, FP16
probability and FP32 PV order. Each version passes 28 changing-length,
permuted-page, tail, graph-replay and output-canary comparisons bytewise
against frozen native attention. This screen uses 16-token pages; it is not
an actual 1648-token model-page or long-context admission.

The first variant has 128 registers and a four-byte spill. A second keeps PV
accumulators in registers across K tiles and uses the actual one-resident-CTA
launch bound. Its four-part kernel has 147 registers, 73088 bytes shared and
zero spill stores/loads. It also passes all 28 comparisons, but still has no
stable speed gain. At 4096 keys, baseline/one-part/four-part medians are
0.382853/0.463544/0.420690 ms. Neither version is installed in a model, and no
sanitizer or model-quality admission is claimed for these rejected routes.

Retained source, flags and library hashes are in
`candidates/draft-fi-q8-p{1,4}{,-register}/manifest.json`; direct gate reports
are `results/draft-fi-q8-gate.json` and
`results/draft-fi-q8-register-gate.json`. Register four-part DSO SHA256:
`76766685df4a15c1ecc60f8dff6d890dd2578b58076d7b88b0070c2b8f9cdd6e`.
This reuse does not claim that an unmodified upstream FlashInfer kernel was
run. The native project route remains a valid porting base; GPU support-list
membership is not used to reject further implementations.

## QPN2 gate/up CTA redistribution: rejected

Another bounded screen replaces each 136-CTA/512-thread fused gate/up with
272 256-thread GEMM CTAs followed by the original native SiLU. Split-K eight,
two accumulator chains, reduction order and FP16 activation boundaries remain
unchanged. Four real consecutive-layer weights and nine changing-input cycles
pass all 36 bytewise comparisons, but seven alternating working-set medians
regress 0.159058 -> 0.171489 ms. No model run or default change follows.
See `results/qpn2-unfused-gated-real.json`.

## GDN value-tile candidate

The current TP4 trace launches 192 one-warp GDN CTAs with 80 registers/thread,
covering twelve value heads with BV=8. A separate exact-shape screen adapts the
value-tiling mechanism to TP4; it does not transfer another TP size's timings.
BV 8/4/2/1 each preserve all output and FP32 state bits for eight acceptance
selectors and two changing graph replays per selector, including strided QKV,
strided state pools, padding canaries and untouched retired slots.
Sixteen distinct state working sets measure 0.381416/0.312884/0.305196/0.308504 ms.
BV2 retains the original K reduction shape and one-warp schedule while exposing
768 CTAs. Focused BV8/BV2 memcheck and racecheck both exit zero.

The explicit `sm70_gdn_value_tile_candidate_route.py` installer is limited to
captured TP4/B1/q8 with twelve value heads and FP32 state. Other shapes/dtypes
retain the original schedule. The live same-input/state shadow now passes all 48 GDN
layers on every rank, with at least 2230 calls per layer. Output/state bits,
finite-value checks and active-slot validity all match. Original outputs drive
shadow generation, and both natural trajectories remain canonical. This is
diagnostic evidence, not a timing result. Evidence is `results/tp4-gdn-bv-screen.json` and
`results/tp4-gdn-bv2-{memcheck,racecheck}.json`. Reusing FP32 Q/K normalization
across value tiles is a separate unadmitted screen and is not combined yet.

The first separate-startup unprofiled GDN pair retains five warmups and five
measurements per fixture. Release1k changes 17.085427 -> 16.333210 ms and MBPP28
16.551539 -> 15.969318 ms, with identical canonical token hashes, natural EOS,
accepted-draft counts and emitted-token counts. Its control is slower than the
prior context pair; the entire gap cannot yet be credited to BV2. The reversed
pair measures control/candidate 16.585019/16.552420 ms for release1k and
16.326226/16.324162 ms for MBPP28. All trajectories remain canonical, but
these 0.032598/0.002064-ms differences do not establish a stable whole-round
gain. Six CPU mocked checks also confirm q8 dispatch, other dtype,
TP size, query width, eager and head-count fallbacks, and restored scope.
See `results/gdn-value-tile-live-shadow-admission.json`,
`results/gdn-value-tile-first-pair.json`, and
`results/gdn-value-tile-cpu-dispatch.json`.

Actual CUDA graph node tracing independently confirms the candidate route:
all recurrent launches use grid `(1,64,12)`, one warp and 55 registers, versus
the frozen `(1,16,12)` and 80 registers. The forty steady rank-rounds each
contain all 48 recurrent calls. Their mean summed service falls from
0.955449 to 0.682117 ms between the retained context and BV2 traces; QPN2
service remains about 7.35 ms. These are profiled observations, not endpoint
speed evidence or measured occupancy. The BV2 trace retains host/rank-wait
outliers: critical-round p50 is 17.876601 ms and mean 18.513127 ms, with
2.048493 ms mean uncovered GPU time. See
`results/v4-gdn-value-tile-nodes-{trace,resource-trace}.json` and the four-worker
runtime-map manifest. Separate-startup variability still needs isolation
before the route is promoted.

The first same-startup diagnostic captures adjacent BV2/BV8 kernels on the
same buffers, disabling one state mutation before the first replay. Dependency
edges identify each pair; both node-enable states are read back after changes.
Twenty-four changing-input/selector switches first pass complete-state, output,
padding and retired-slot checks. The owned model client then switches only
between requests, with five warmups per arm and five interleaved measurements.
Release1k changes 16.493886 -> 16.279546 ms and MBPP28 16.088556 -> 15.872776 ms,
with canonical token IDs, accepted drafts and natural EOS in every request.
This isolates an approximately 0.21-ms whole-round gain while retaining startup,
prefill, allocations and GEMM choices. One such startup is not final admission.
See `results/gdn-value-tile-within-start-1-summary.json` and
`results/gdn-pair-graph-gate.json`. The diagnostic uses CUDA's documented
[individual-node enable behavior](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#individual-node-enable);
disabled nodes retain dependencies and behave as empty nodes.

## Q/K reuse exposes a recursive FP32 rounding boundary

The first private normalization-reuse screen leaves the immediate FP16 output
unchanged but changes 62767 FP32 state elements on its first candidate case.
Timing is skipped. A diagnostic tap is first checked against the frozen
recurrence: both its output and complete state pool remain byte-equal.
On the exact failing input, the standalone and in-recurrence normalized Q/K
also match bytewise. Thus this candidate's first state difference is downstream
of Q/K normalization, not in those normalized operands.

PTX identifies a changed contraction boundary in the local four-element
`h dot k` reduction. The frozen kernel first rounds the product at local K1,
then contracts K0, K2 and K3 through FMA. Loading materialized normalized K
allows the compiler to choose K0 as the initially rounded product. The
subsequent warp reduction has the same shape, but these programs need not
produce identical FP32 state. The output's FP16 rounding initially conceals it.

The corrected private candidate makes the K1 product's rounding explicit with
`mul.rn.f32` and retains the remaining reduction. All 48 checked cases across
BV8, BV2 and corrected reuse now have zero output and complete-state bit
differences. Sixteen-state working-set medians are 0.379136/0.303040/0.291456 ms;
the extra gain over BV2 is only 0.011584 ms and is not a model gain. Further
kernel safety, real-model shadow and whole-round validation remain open.
This diagnosis concerns the new reuse experiment; it does not resolve the
previous unrelated 4.33% repeat-start distribution discrepancy.

Evidence: `results/tp4-gdn-precomputed-qk-screen.json`,
`results/gdn-norm-tap-failure-comparison.json`,
`results/gdn-norm-tap-failure-operands.pt`, and
`results/tp4-gdn-precomputed-qk-fmafix-screen.json`. The corrected derived-source
SHA256 is `ce200148aa03ab9da6692d69e07f5d48f8e58b1f08763aaa88972ab3b0e7acee`.
The artifact root retains original/corrected Triton source, TTGIR and PTX.

## Cooperative MLP publication: native gates pass

A new private candidate executes gate/up and the dependent down projection
inside a 160-CTA cooperative kernel, with one grid barrier between them. It
retains the original dot-product chains, FP16 SiLU boundaries and packet
publisher. The original consumer remains separate and starts after the local
producer finishes; this does not revive the rejected pre-producer polling
scheme. The launcher checks that all 160 CTAs can be resident before launch.
Compilation reports 64 registers, 32768-byte shared storage and no local stack
or spills; runtime confirms two resident CTAs per SM. Four ranks, four real
consecutive-layer weight sets, nine changing-input cycles, skewed ranks and an
additional ordinary push all preserve gate, down and reduced output bits and
buffer canaries. Seven paired working-set trials measure 0.457871 -> 0.451072
ms, about 1.5% locally; both arms drift during the trials, so raw samples are
retained. Four-rank memcheck and racecheck exit zero, with zero reported errors
or hazards. The first private model shadow executes no candidate calls: its
outer Python shape guard is specialized away during dynamic model compilation.
Those results are explicitly excluded. Moving eligibility into the opaque
runtime custom op and asserting all 64 captured prefixes fixes the route.
All four ranks then pass at least 1338 live comparisons in each of 64 layers,
with zero gate/final-output bit differences or nonfinite values. Original
outputs drive generation; both natural fixtures stay canonical.

The separate-startup five-warmup/five-measurement pair measures release1k
16.445409 -> 16.453666 ms and MBPP28 16.129142 -> 15.966142 ms. All token IDs,
acceptance counts and natural EOS match, but the gain is workload-dependent
and only one pair is available. This is not a promoted combination or a
sub-15-ms result. `results/coop-mlp-live-shadow-admission.json` and the
`v4-coop-mlp-warm5-{control,candidate}-speed-*` reports retain the evidence.

Reports are `results/qpn2-coop-mlp-{real,memcheck,racecheck}.json` with sanitizer
logs and manifests alongside them. The private library SHA256 is
`22a91bd9f9e8aa0cc1324b0482c0fc4d6fc695ef7b553935801a935e47194c31`;
`candidates/qpn2-coop-mlp/cooperative-manifest.json` retains the source and flags.

The versioned `build_sm70_qpn2_cooperative_mlp.py` accepts an explicit private
output directory and reproduces the exact validated CUDA source SHA256
`21f5a7448cb71ec3b847f21e41068b64f7eae44db0e5e2ec365b96a2fbd99b65`.
Its companion benchmark keeps real consecutive-layer weights, changing inputs,
rank skew and mixed-protocol epochs, and adds six rejected non-q8 row counts.
Its four-rank rerun passes both changing-input cycles, all six rejected shapes
on every rank and all output/canary comparisons; the generated source matches
the previously built and sanitized DSO. The rerun is recorded in
`results/qpn2-coop-mlp-versioned-gate.json`.
The benchmark is an operator/communication gate, not a model quality score.

## Independent draft query-row partitions: rejected

A further native FlashInfer-fragment experiment partitions the eight query
rows among four or eight CTAs per head, retaining K176, per-row reduction and
FP16 probability boundaries. This differs from the earlier output-column
partitions. Both versions match frozen FP16 output in 40 graph/canary cases at
the actual 1648-token page size, including 1647/1648/1649 and 3295/3296/3297 key
lengths. The candidates' FP32 LSE also matches each other; that check is not an
independent FP32-score reference.
Yet both regress: at 1024 keys, control/four/eight-part medians are
0.074189/0.077496/0.080691 ms; at 4096 keys they are
0.322202/0.369172/0.380150 ms. Increasing the grid from eight to 32/64 CTAs does
not itself improve latency. No serving hook or model admission follows.
`results/draft-fi-query-rows-gate.json` and the two candidate manifests retain
the frozen native DSO hash, generated source, raw samples and compile resources.

The upstream QPN2 source was rechecked at
[`v100-skinny` 5b589c0](https://github.com/dnv2003/v100-skinny/blob/5b589c0dc81223e0ba65bcb3e755874723f8b515/kernels/skinny_kernels.cu)
and the independent
[`ninfer-v100` 8fd0e2e implementation](https://github.com/geoffwatts/ninfer-v100/blob/8fd0e2efdea77bab944991f2394309c07b8baffe/src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh).
Their prepacked quadpair-on-N layout and independent accumulator mechanism
are already represented in this campaign; their weight/KV contracts and
published timings are not imported as this model's performance evidence.

## Direct native m8 draft QK/PV: no whole-round admission

The next native FlashInfer-fragment screen uses Volta's
`mma.sync.aligned.m8n8k4` for the eight actual query rows. It preserves the
original K4 accumulation order, K176 schedule, FP32 softmax and FP16
probability boundary. A separate QK oracle compares original WMMA and native
FP32 scores before softmax in 45 cases. Extending the same mechanism to PV
adds 45 comparisons with nonzero FP32 initial accumulators. All 90 FP32
comparisons are byte-equal; independent FP64 references are also retained.
Twenty complete attention cases cover the actual 1648-token pages and changing
graph inputs, with matching frozen FP16 outputs and intact canaries. Candidate
FP32 LSE matches the parent fragment implementation, not an independent
original-native LSE oracle.

The QK-only implementation reports 180 registers and the combined QK/PV
implementation 138, both with 73088-byte shared memory and zero spills.
At 512/1024/4096 keys, frozen/combined medians are respectively
0.043653/0.042240, 0.074793/0.074117 and 0.322703/0.360151 ms.
The small short-context difference and longer-context regression do not
justify a serving route. No sanitizer or model admission follows. These
results do not establish the memory/issue bottleneck without counters.

Evidence is `results/draft-{qk,qkpv}-m8-gate.json`. Generated source SHA256s
are `b98ee65791dcafe46e0ccb7529feff7a555b9d5ab02798bb70fc2dd54a8bb6f0`
and `1f6d113deb1144838de6f49aefa4328f9f5a45aeacdcde3793d9a94689d575a8`;
their native DSO SHA256s are
`a15460c1078136d84bd9630055fce7095b45b65f7c0b09ca08dea9e9d178d084`
and `be9844d71c36b1ed6e9309fd8faf8388e07fa4ebea32a5d23c3169d9a9c86f9c`.
Operand mapping follows NVIDIA's
[PTX m8n8k4 fragment documentation](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html).

## Positive QPN2 scale decoding: no stable gain

A separate screen checks every actual scale byte in the four-rank,
four-consecutive-layer working set before removing a redundant sign-bit
construction. All scale bytes are below 128, and exhaustive bit mapping of
those 128 codes matches the original. The two FP16 multiplies, HMMA chains,
SiLU and rank reduction remain unchanged. Nine changing-input cycles, skewed
ranks, dependent gate/down projections, mixed epochs and canaries all pass
bytewise comparison.

Seven paired working-set trials nevertheless measure 0.458691/0.459366 ms
for control/candidate, with mixed signs in the paired differences. Keep this
specialization off and do not advance it to a model test without new evidence.
The result is `results/qpn2-positive-scales-real.json`. Column/row source
SHA256s are `ef64e021cd88403acd2dfa676653fa293244aa280330338760e91c8b344198ee`
and `ab2adf4c76298186eed97c684c461ed792c8d5c46c945f4be4e225ba465fe5d6`;
DSO SHA256s are `ed02ddf8baac4d537caaf328d181ff9dca6fedaac15a7453410a37beb58aef9d`
and `be15316b1f8471063803a3f87d4a5aee1c63f9955c5a57c17bb1c7dab019b968`.

## QPN2 layout substitution within the original graph

The preceding packed-input quality hold first diverges during prefill, before
its q8 layout is active. A new diagnostic therefore retains the original
Python/FX path and changes only executable q8 CUDA graph nodes after capture.
Dependency ancestry and buffer addresses pair a norm with its QPN2 consumer.
The replacement norm keeps its original row-major output and FP32 residual,
and additionally writes private `[320,8,16]` storage. Only the paired projection
receives the packed pointer. Original raw graph nodes and edges remain intact;
the caller synchronizes and switches executable parameters between requests.
This uses CUDA's documented
[kernel-node parameter update interface](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html),
not a change to the compilation boundary or prefill implementation.

Four ranks each pass 216 real-weight projection cases across three residual
contracts, three magnitudes and changing inputs. Eighty-one control/candidate/
control replays per rank preserve output/residual bits, logical packed values,
allocation canaries and the raw graph fingerprint. M1/M7 norm-plus-QPN2 graphs
remain unmodified. M9/M32 checks cover norms only: the independently frozen raw
QPN2 entry correctly rejects M greater than eight. The first script mistakenly
used that entry for larger rows and is not recorded as a complete gate pass.
The eight-column/norm working set has only a small local difference: rank 0
medians are 0.294416/0.291616 ms. It excludes row projections, communication and
the model round; no end-to-end saving is inferred.

An initial graph reader mishandles the zero-edge single-node case. A later
parameter-count probe intentionally reaches an invalid API index, producing
4912 memcheck API errors despite passing data comparisons. Neither is a clean
admission. Reading the known kernel signatures removes that probe; the next
memcheck and racecheck each exit zero, with zero errors/hazards. Reports retain
these separate attempts rather than filtering the earlier errors.

The first model startup then stops during draft graph capture: the reader
assumes pointer-array arguments for an unrelated cuBLAS kernel using the packed
launch-parameter convention. No endpoint timing or model quality result is
produced. The reader is narrowed to registered kernels before accessing their
arguments; a separate unmodified-cuBLAS graph check is added. A subsequent
model attempt remains necessary. The scoped installer also checks the target
norm weights/epsilon and verifies packed producers were written on candidate
requests and left untouched on control requests. These checks occur between
requests, not inside the timed round.

The explicit builder `build_sm70_qpn2_dual_norm.py` reproduces generated source
SHA256 `346e063dfaf185650c279394588eac2663e12b886e179a7a6a71a899a6b9f096`
from the pinned norm expressions. `benchmark_sm70_qpn2_graph_layout.py` accepts
the two frozen projection libraries, generated norm module and real-weight
root explicitly; its `sm70_qpn2_graph_{nodes,layout}.py` helpers install no
serving default. The first versioned four-rank rerun passes 72 cases per rank.
The final helper revision adds the cuBLAS fallback check and uses the standard
accelerator synchronization API. Its four-rank rerun passes another 72 cases
per rank, including the untouched cuBLAS graph. The matching private helper's
memcheck/racecheck reruns both exit zero with zero errors/hazards and no invalid
API queries. A second model attempt finds the 128-pair full graph plus temporary
compiler/piecewise graphs, so its broad count assertion fails before serving.
The installer is then scoped directly to the model manager's owned
`FULL / num_tokens=8 / num_reqs=1 / uniform_token_count=8` descriptor.

The third attempt completes the same-startup five-warmup/five-measurement pair.
All four ranks match 128 norm/projection pairs on that exact descriptor. The
between-request sentinel check confirms all candidate producers were written
and control requests leave their packed buffers untouched. Both fixtures keep
canonical tokens, acceptance and natural EOS. Release1k medians are
16.348511/16.328542 ms and MBPP28 16.001308/15.983139 ms. The paired MBPP savings
include two regressions; approximately 0.02 ms is not a substantial or admitted
full-round gain. Keep this candidate off. The original fixed-prefix quality
hold is not cleared by these natural trajectories alone.

Evidence is retained in
`results/qpn2-layout-graph-*`, the corresponding queue records, and
`candidates/qpn2-layout-model-v{1,2}`. The full pair and raw samples are in
`results/qpn2-layout-within-start-summary.json` and
`results/v4-qpn2-layout-within-start-3-switch.json`, with their own four-worker
runtime-library manifest. Final fixed-prefix, acceptance and complete-round
admission remain open; this does not resolve the old repeat-start TV discrepancy
by itself.
