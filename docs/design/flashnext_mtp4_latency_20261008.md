# Flash-Next MTP4 current latency diagnosis

The objective remains C1 at or below 12 ms/round in the acceptance benchmark,
with target correctness, retained acceptance and no C4 regression. This report
measures the current route before selecting another implementation. It does not
claim a new endpoint improvement.

## Workload and provenance

Both recent placement arms and the focused trace use the normal wheel from
`c1ebacb677`, version `1.5.2.dev1190+gc1ebacb67`, native core SHA256
`784d1447f4f5f5593fa77db525e6b40e841366d654202aab5a56beec67398a0c`.
The model is Flash-Next GSQ-RCO IQ3_S with FP16-loaded MTP4, TP4 on four
V100-SXM2-32GB GPUs, CUDA 12.8 and Torch 2.10.0+cu128. Application SM clock is
1290 MHz and memory clock 877 MHz; clocks are not changed. GPU0/3 and GPU1/2
have SYS rather than direct NVLink connections.

The working route has per-vector E4M3 target history in pinned host memory,
FP16 draft history, an 8192-token FP16 hot cache, FP32 recurrent state, 157
allocator blocks, maximum length 9216, prefill batch 512 and four concurrent
requests. Target and draft decode use FULL graphs. C1 uses 8192 input and 256
output tokens; C4 uses 128 input and 600 output tokens per stream. Natural
acceptance uses eight prompts, greedy sampling, EOS enabled and a 600-token
limit. Placement quality and C4 qualification remain unresolved, as recorded
in [the host KV design](flashnext_host_fp8_kv.md).

| History backing | Unprofiled C1 ms/round | Tokens/round | C4 ms/round |
| --- | ---: | ---: | ---: |
| Device E4M3 reference | 17.849 | 4.886 | 42.417 |
| Pinned host E4M3 | 18.320 | 4.886 | 44.565 |

C1 uses the unchanged mean of before/after probes. Host before/after means are
18.053/18.587 ms; the slower tail remains in the reported mean. Its medians
18.041/18.065 ms are diagnostics, not a replacement endpoint. All C1 probe
outputs and all four 600-token C4 outputs match between placement arms.

## Low-overhead round partition

For 32 central steps, partition each rank's target-start to next-target-start
using only that GPU's event clock. Do not subtract event timestamps across
GPUs. The following rank-0 medians describe typical phase sizes; medians are
not additive. Each individual recorded interval closes exactly.

| Current host phase | Median ms | Interpretation |
| --- | ---: | --- |
| Target replay | 13.935 | Verification graph, including GPU dependency waits |
| Target end to draft start | 0.530 | Head, sampling and handoff envelope |
| Four draft steps | 3.396 | Dependent proposal chain, including head/state work |
| Draft end to next target start | 0.112 | Exposed preparation/submission envelope |
| Target start to next target start | 17.976 | Observed typical round |

Target medians across ranks are 13.935–13.938 ms. Draft medians are
3.389–3.414 ms. The device-reference target median is 13.873 ms on rank0;
its round median is 17.897 ms. Thus the larger endpoint mean placement delta
also contains tail events, not only steady target work. One host observed
proposal envelope reaches 28.412 ms, versus its approximately 3.40-ms median.
The device observed arm has a 17.55-ms preparation tail. Their causes are not
established by these envelope events.

The target alone exceeds 12 ms. Retaining approximately 3.40 ms of draft and
0.64 ms of other work leaves about 7.96 ms for the verifier. Reaching 12 ms
therefore requires roughly a 6-ms reduction of its current typical envelope,
or a combination of verifier and draft-chain improvements. Ordinary host
submission cannot supply that budget.

## Current graph-node trace

The focused Nsight Systems 2024.6.2 capture uses CUDA, NVTX and OSRT graph-node
tracing around generation. Its C1 output matches the unprofiled arm. Trace
endpoint is 19.921 ms/round, so trace durations are composition evidence,
not accepted performance. Thirty-seven common TP windows align by target
replay ordinal, with M5 and one-request metadata verified on every worker.
All windows are retained. Mean window is 19.935 ms; target activity envelope
is 14.926 ms. GPU target-entry skew median/p90 is 0.224/0.331 ms. This is
actual GPU entry skew in the perturbed trace, not the larger CPU submission
spread or a claimed unprofiled skew measurement.

Every target graph contains 1007 kernels per rank. Against the prior repaired
983-node graph, the only static call-count differences are 12 protect, 12
gather and 12 new history-write kernels, replacing 12 ordinary cache writes.
This is a cross-version graph-count comparison, not an endpoint ablation.
The former dense-class admission and token-at-a-time expert fallback issues
are no longer the dominant diagnosis. Runtime logs confirm batch FP16 a/b
GEMV, native Q8 expert gate/up and Q8 routed intermediates at M5.

| Target family | Calls/rank/round | Rank-mean service ms |
| --- | ---: | ---: |
| Complete HC boundary | 94 | 3.091 |
| Dense MMA projections, including shared down | 144 | 2.307 |
| Routed gate/up | 48 | 1.819 |
| Routed down/unroute | 48 | 0.933 |
| QSA selected attention | 24 | 1.201 |
| Router projection and top-k | 96 | 1.115 |
| Shared gate/up | 48 | 0.805 |
| GDN | 72 | 0.777 |
| Host cache fill/miss resolution | 12 | 0.387 |
| Host history write | 12 | 0.097 |
| Host cache protection | 12 | 0.051 |
| QSA indexer top-k | 12 | 0.180 |
| Tensor concatenation | 48 | 0.164 |
| Half Fill | 48 | 0.120 |

These service rows overlap. Full target service is 14.816 ms/rank/round;
concurrent kernel overlap is approximately 1.23–1.32 ms/rank. Shared gate/up
runs alongside routed work, so its standalone savings cannot be added to the
round. Fill ownership and padding requirements must be checked before removal.

The rank2 activity partition closes the mean 19.935-ms profiled window:

| Exclusive activity | ms/window |
| --- | ---: |
| Target | 13.735 |
| Draft | 3.609 |
| Outside graphs | 0.676 |
| Copies only | 0.131 |
| No recorded kernel/copy activity | 1.783 |

Rank0 has 2.126 ms without recorded activity; other ranks have 1.783–1.814 ms.
These gaps include graph dependencies and GPU arrival waits, not exclusively
CPU execution. Communication spin inside a kernel remains classified as busy.
Even eliminating every uncovered gap on rank2 cannot redeem the remaining
approximately 6-ms requirement.

Largest rank0 gap edges are HC-to-shared-gate/up 0.183 ms/round and
expert-down-to-concatenation 0.148 ms/round. Concatenation-to-HC adds 0.044 ms.
These identify buffer and launch boundaries but do not independently prove
that all their time is recoverable. GDN a/b GEMV-to-split contributes 0.062 ms.

## Root causes and next implementation order

1. **HC dependency chain.** Ninety-four serialized boundaries average about
   32.9 microseconds of profiled service each. The existing TP sum, normalization,
   down partials, LoRA publication, peer exchange, up and gate-mix form a
   readiness chain. Prior coupled stage measurements locate a particularly
   expensive LoRA-arrival/up-prefetch/second-barrier stage. HC is primarily a
   dependency/synchronization target, not merely a weight-decoding target.
   First screen producer-to-HC readiness and the reduction/exchange schedule
   while preserving FP32 accumulation and the current normalization contract.
   Distinguish a new cross-operator schedule from previously rejected per-warp
   polling, bitmap barriers, early up loads and resident CTA partitioning.
   A 1-ms HC target would require about 10.6 microseconds per boundary and
   supply only about 2 ms of the needed reduction; it is not a measured result.
2. **IQ expert decode and dense projection chains.** Routed work contributes
   2.75 ms of service, dense MMA 2.31 ms. The expert route already quantizes each
   activation once, uses native dp4a, fuses gate/up and SiLU, and consumes Q8
   intermediates directly in down/unroute. The next work must improve the
   existing native decoder/issue schedule, not restore the superseded MMQ or
   FP16-dequant fallback. Retain compressed weights and existing precision;
   prior expanded scalar-LUT and packet-prefetch variants have negative data.
   Use an exact-shape screen before another model restart.
3. **Buffer and QSA cache boundaries.** Producer outputs should eventually
   reach an owned HC payload without a separate concatenation, preserving
   explicit compiler consumers and graph lifetime. The previous hidden-Tensor
   registry path failed correctness and must not return. Host resolution also
   launches 645 small gather CTAs for M5/2051 columns despite mostly hot pages;
   coarse all-hit resolution is a concrete microbenchmark candidate, with
   legacy miss protection retained. The entire write/protect/gather service
   is only 0.535 ms, an upper bound rather than a predicted endpoint saving.
4. **Draft/head and tails.** The proposal chain is about 3.4 ms and sampling
   handoff about 0.53 ms. Draft steps depend on preceding predictions. The
   existing same-engine single-graph experiment regressed and cannot be
   assumed beneficial. Screen its actual head/collective/state boundaries
   only when the trace supports a specific change. Attribute the occasional
   17–25-ms excess preparation/proposal tails separately; do not remove them
   from the acceptance benchmark mean.

No implementation is changed by this diagnosis. For each selected candidate,
run isolated correctness and changed-input graph tests, then quantify its
critical-path ceiling. Accumulate meaningful candidates before same-wheel
C1/C4 endpoint A/B and ablations. Preserve tokens/round, teacher distributions,
natural acceptance and C4; fewer graph nodes or faster microbenchmarks alone
do not qualify progress toward 12 ms.
