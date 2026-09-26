# Flash-Next MTP4 timing reconciliation, 2026-09-26

The accepted **27.3963 ms complete-round baseline remains unchanged**. The
36.5387-ms node-trace round and 29.3167-ms target graph are perturbed diagnostic
measurements, not evidence that the target verifier regressed from #398's
23.409-ms result. A matched capture comparison now measures the perturbation;
a separate run uses the historical V2 phase timer to anchor target latency.

## Matched capture comparison

Use the [frozen baseline contract](sm70_flash_next_mtp4_default_profile.md):
TP4 V100 GPUs 0-3, NVFP4 weights, FP16 activation/KV, FP32 SSM, V2, MTP4,
FULL_AND_PIECEWISE graphs, hybrid pinned-UVA PLE with all rows in host RAM,
32768 capacity, 8192 batch tokens, memory 0.95, prefix caching and C=1.
The same 8192-token prompt and 129-token greedy forced-length output run in
one engine, with capture disabled, enabled, then disabled. Nsight remains
attached throughout; this isolates capture activation, not attachment overhead.

| Capture | Decode / speculative rounds, ms | Rounds | Accepted drafts |
| --- | ---: | ---: | ---: |
| Off before | 27.1598 | 85 | 43 |
| Node capture on | 35.4506 | 85 | 43 |
| Off after | 28.0762 | 85 | 43 |

All three complete token lists and speculative counters are identical, and
the tokens exactly match the original short trace. Off measurements bracket
27.6180 ms on average; capture adds 7.8325 ms, or 28.36%. Compared separately
with either off measurement, the increase is 7.3743-8.2908 ms. The off arms
differ by 3.37%, so this single bracket is not a confidence interval or proof
that every millisecond of the original 9.1424-ms discrepancy is tracing cost.
The original endpoint also had 513 output tokens rather than 129.

The new trace has the same 84 closed intervals per rank and 2612 target
kernels per rank/round. Rank-0 closed-round mean is 35.4946 ms and target
graph mean is 28.2399 ms, versus the original 36.5387/29.3167 ms. These two
traces demonstrate timing variability with the same output trajectory; they
must not be proportionally rescaled into the accepted endpoint baseline.

## Target timing using the historical method

A separate process, without Nsight, uses the existing opt-in
`VLLM_SM70_MTP_PROFILE=1` V2 CUDA-event timing and diagnostic fences. No serving
code or arithmetic changed. Named worker-extension methods reset/read the
existing timing accumulators. The first warmup and prompt-prime requests are
excluded from the following aggregates.

| Measurement | 8192-input fixture, TP0 ms | Same historical 152-input HumanEval/0, TP0 ms |
| --- | ---: | ---: |
| Target forward, CUDA-event interval | 21.1253 | 20.7002 |
| Target sampling, CUDA-event interval | 0.7477 | 0.7505 |
| Target state update, CUDA-event interval | 0.0164 | 0.0171 |
| Sum of those event intervals | 21.8895 | 21.4678 |
| Target verifier synchronized CPU wall | **21.9472** | **21.5247** |
| Four-draft event interval | 4.9831 | 4.9486 |

Across rank means, target wall is **21.9472-22.2415 ms** at 8K and
**21.5247-21.8349 ms** on the historical input. These are rank aggregates,
not means of per-round critical-rank maxima. The 8K aggregate has 85 profiled
calls per rank; the short request has 14 worker calls and 13 reported
speculative rounds. Do not silently equate worker calls with emitted-round
counters in asynchronous generation.

The 8K output and acceptance counters exactly match the paired capture.
The historical request emits the identical 62 token IDs and retains 13 drafts,
50 accepted drafts and 4.8462 acceptance length. The retained #398 result is
23.409-ms TP0 target wall over its last eight steady calls, including forward,
sampling and state update; its 22.543/0.762/0.027-ms values are CUDA-event
intervals, despite the old report's label "GPU service". They are not sums of
kernel durations. That campaign used batch 2048 and memory 0.90, unlike this
8192/0.95 contract. This audit restores measurement scope and output parity;
it is not a fresh historical-source A/B or a new percentage-speedup claim.

The phase timer perturbs scheduling: the fenced 8K endpoint is 29.7579
ms/round. Do not substitute that for 27.3963 ms or add separately collected
phase measurements to a node-trace table to manufacture endpoint closure.

## Original target graph, excluding every draft and sampling

For the original 84 rank-0 windows:

```text
29.316745 ms wall
  = 23.602515 ms summed kernel service
  -  2.247072 ms overlapping service
  +  7.961302 ms without a recorded kernel
```

Kernel busy-time union is 21.355443 ms. Of the remaining time, 0.142238 ms
contains only memcpy/memset and 7.819064 ms has none of these GPU activities.
Busy time includes collective spin waits; it is not a compute-only or normal
target-latency measurement. Excluding the first and last intervals gives
29.224001-ms target wall and 7.958007-ms no-kernel time, so edge intervals do
not explain the discrepancy.

| Target-only family | TP0 mean kernel service, ms |
| --- | ---: |
| Dense GEMM/GEMV and reductions | 9.3119 |
| Other elementwise/copy/index | 3.6022 |
| TP communication, including readiness waits | 2.8790 |
| Direct NVFP4 experts | 2.8771 |
| QSA/attention | 1.8041 |
| HC combine/gating | 1.1655 |
| GDN/convolution | 0.9659 |
| Routing/shared gate | 0.7526 |
| Metadata/state | 0.1298 |
| PLE pinned-UVA lookup | 0.1144 |

These rows overlap. The target's hottest CUTLASS 16x16 WMMA family has
291 calls and 5.6565 ms, whereas the previously published 304 calls and
8.07-8.54 ms include graph-external/draft work. Likewise 12.8330 ms was
whole-round rank-max dense service, not target-only latency.

The original longest target `cudaGraphLaunch` CPU call averages 5.9451 ms;
5.3819 ms of no-kernel time overlaps that call. In the new trace the respective
values are 5.5113 and 5.1601 ms. This locates launch/readiness disruption, but
does not independently attribute every idle gap to CPU work or tracing.
Collective service contains rank-arrival waits and is not transfer bandwidth.

## Reuse the historical optimization evidence

Both traces retain the direct M5 W13 split4 and W2 split1 kernels, 48 calls
each, plus the TP4 push/sum2 collectives. Runtime logs confirm the M5 E512/K10
router and shared GDN split-copy fusion. #398's acceleration is executing.

The earlier campaign already screened ordinary FP16 M5 projection replacement:
its weighted microbenchmark saved only 0.1902 ms (2.77%). Persistent HC and
small-row native QSA alternatives also have retained negative results. A large
service category does not justify repeating those model startups. Reuse the
existing shape/candidate evidence; require a specific new component win before
another endpoint comparison. Use the phase timer to assess target cost and
node traces to locate calls/dependencies, keeping their measurement scopes apart.

## Provenance and failed attempts

Measured source is `3ed85627c811013a49760eed7a9046b0d53a4663`. Its runtime
source equals the frozen 27.3963-ms source tree except for one typing-only
decorator overload argument. The original measured tree was reconstructed from
the recorded commit/patch and compared before running. Native extensions and
all production flags match the frozen baseline. No wheel was built.

Raw evidence is retained under the same owned worktree's `.artifacts/`:

- `paired_trace_source_audit.json`: reconstructed-tree comparison.
- `paired_trace_audit_retry1.json`, `_contract.json`, `.log`, `_trace.nsys-rep`,
  `_trace.sqlite`, `_summary.json`, `_wall.json`, `_target_breakdown.json`:
  complete off/on/off measurements and their node trace.
- `target_phase_audit.json`, `_contract.json`, `.log`, `.exit`, `_summary.json`:
  separate phase-only completion, exit 0, and all worker means/output tokens.
- `stacked_mtp_graph_target_breakdown.json` and `target_graph_original_audit.md`:
  original target-only decomposition.

Two harness failures are retained explicitly. The first startup stopped after
warmup because LLM construction mutated a nested configuration used for JSON
output; passing a deep copy fixes it. The next startup completed off/on/off,
then rejected a callable diagnostic RPC under secure serialization. Its overall
exit is 1 and report completion flag is false; only the completed three-arm
segment is admitted. No phase result is claimed from it. The final phase-only
startup uses the previously exercised named worker interface, without enabling
insecure serialization or recapturing the trace. All owned GPU workers exited.
