# QUASAR + DFlash2 complete-round resource audit, 2026-09-09

The 15-ms goal is not met. The current unprofiled sparse-selector screen is
16.761 ms for release1k and 16.373 ms for MBPP28. Both use rear GPUs 4–7,
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
The corrected private route has a separate trace queued and unprofiled pair
prepared. It is not promoted.

Raw evidence: `profile/v4-sparse-dense-order-nodes/tp4.{nsys-rep,sqlite}`,
`results/v4-sparse-dense-order-nodes-trace.json`,
`results/v4-sparse-dense-order-resource-trace.json`, and
`results/attention-headsplit-binding-identity.json`. The trace capture and export
completed, but the wrapper then failed its runtime-map ownership-name assertion.
Therefore this trace lacks its own final map manifest; separate unprofiled
four-worker DSO manifests are retained. The next trace job fixes the ownership
name. Do not describe the earlier wrapper job as wholly successful.

Final combination fixed-prefix quality, three independent startup pairs,
acceptance non-inferiority and model long-context gates remain open. No
15-ms result, default promotion, merge or 256K performance claim follows.
