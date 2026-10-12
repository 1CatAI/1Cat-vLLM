# Complete DFlash2 round baseline clarification

The historical approximately 14-ms baseline is the **complete speculative
round**, including draft and verification. It must not be relabeled as
only target GPU service or compared with sums of profiled kernel durations.
The prior profiling chat and release PRs establish the following records.

| Evidence | Input tokens | Complete round ms | Emitted tokens / round |
| --- | ---: | ---: | ---: |
| PR #1008 candidate | 1024 | 14.390 | 2.980 |
| PR #1008 candidate | 8192 | 14.842 | 2.948 |
| Later accepted production result | 1024 | 13.456 | 2.983 |
| Later accepted production result | 8192 | 13.860 | 2.924 |

[PR #1008](https://github.com/1CatAI/1Cat-vLLM/pull/1008) separately reports
11.041-ms target timing at 1K; that is a different metric from its complete
14.390-ms round. [Draft #1034](https://github.com/1CatAI/1Cat-vLLM/pull/1034)
explicitly identifies the subsequently admitted 13.456/13.860-ms complete-round
baseline. These agree with the historical local production summary and the
chat titled “Profile Qwen3.8-27B NVFP4 decode”.

The admitted protocol uses V100 TP4, 1530-MHz graphics and 877-MHz memory,
300-W power limits, DFlash2 seven proposals, maximum context capacity 262144,
prefix caching and CUDA Graphs. Its C1 workload has eight prompts per length,
600 output tokens, temperature 0.7, top-p 0.9, top-k 20 and seed 123;
steady streaming intervals exclude the first 20 warm-up intervals.
The output-length bound is a timing fixture, not natural-EOS quality evidence.
The later summary is `production-admission-results/results/summary.json`
with `default_source_routes=true` and `profiler=false`.

The current [4K/32K KV comparison](kv_int8_e2e_20261012.md) uses a different
latency corpus, sampling and timing interval. It divides request decode time
by completed draft rounds; “accepted drafts” excludes the target bonus token.
The historical table reports total emitted tokens per round. Those acceptance
columns cannot be compared directly. The current FP8/INT8 pair shares its
model and settings, but does not replace the historical admission protocol.

A read-only device snapshot on 2026-10-12 showed 185-W limits and 1290-MHz
application graphics clocks. This is a runtime configuration difference to
control in future measurements, not proof of the settings during earlier
paired runs. It does not establish their cause or magnitude of regression.
Keep the current model and weights fixed. Reuse the historical complete-round
protocol and frozen current paired corpus to separate runtime cost, cache
attention cost, accepted drafts and milliseconds per emitted token.
Profiled target/draft/communication service remains diagnostic only.
