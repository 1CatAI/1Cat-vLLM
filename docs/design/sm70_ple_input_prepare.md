# SM70 PLE input preparation

The model-state reference launches thirteen GPU operations to gather n-gram
history, mask early sequence positions, pad requests with EOS, and copy query
boundaries. A single integer Triton operator performs those steps from the
same request mapping and computed-token buffers. It reads the RequestState
pinned UVA token storage directly. Rollback and inactive-request padding retain
the exact reference values.

Admission requires SM70, contiguous CUDA int32 sources, at most four padded
requests, and context length at most eight. KernelConfig.ple_input_prepare
is enabled by default. Unqualified inputs retain the reference and report a
reason in ple_input_preparations. The switch and observed status are excluded
from the compilation hash because these operations run before model graph
submission. No new environment variable is introduced.

## Operator qualification

A complete installed wheel passes sixteen GPU cases and four CPU policy
checks. GPU cases include ordinary and actual pinned UVA token storage,
empty and padded batches, short history, changed request mappings, rollback,
and changed-input CUDA Graph replay. Context and query boundaries match the
model-state reference exactly.

On V100 with CUDA 12.8 and Torch 2.10, the actual pinned-token benchmark gives:

|Batch|Reference GPU us|Fused GPU us|Reference host us|Fused host us|GPU operations|
|---|---:|---:|---:|---:|---|
|M5 / one request|45.855|36.666|252.928|36.224|13 to 1|
|M20 / four requests|47.848|43.884|428.776|49.350|13 to 1|

GPU times are amortized graph replay. Host times are medians of one hundred
synchronized submissions. Host and GPU deltas overlap and must not be added.
At one preparation per target round, the isolated GPU delta is 9.19us at M5
and 3.96us at M20; host submission decreases 216.70us and 379.43us. These are
operator measurements, not whole-model latency claims.

## Model comparison contract

The installed-model benchmark switches only ple_input_prepare within one
loaded model. Both observed and unobserved M5 cohorts use the same greedy
8192-token input and 256-token output limit. M20 cohorts use four requests,
128 input tokens and 600 output tokens per request. The benchmark also checks
eight natural prompts, bounded EOS completions and matched-prefix teacher
logits. CPU submission skew is reported separately from GPU graph entry.

The prior dense/HC target trace has GPU entry skew p50 0.586ms and p90 0.663ms,
with rank 0 last in all 45 stable rounds. The other ranks differ by at most
about 16us at p90. CPU stage observations show rank-0 PLE submission about
120us longer than other ranks; output materialization includes GPU waits.
This change does not reinterpret those waits as removable CPU work.

## Installed-model results

One installed Flash-Next IQ3_S TP4 model, FP16 MTP4, FULL target decode,
FP16 KV and FP32 recurrent state compares both input implementations without
reloading weights. C1 unobserved median is 21.671ms for the reference and
21.551ms for fusion, p90 21.881ms and 21.767ms. Means are 21.692ms and
21.970ms: fusion includes a transient outlier, so the mean is not reported as
a speedup. Both cohorts emit 4.886 tokens per measured round. All four C1
cohorts, including two CPU-observed runs, return identical output tokens.

C4 reference/fused medians are 57.630/50.396ms, means 57.220/50.263ms.
Output tokens match exactly and the candidate has no observed C4 regression.
The reference is the first C4 cohort; its unusually large gap is not assigned
to the input operator alone. These are node-validation measurements and do
not establish a seven-millisecond isolated fusion gain.

CPU submission skew medians are 0.927ms reference and 0.662ms fused.
The fused observer has long submission outliers, p90 18.565ms; reference p90
is 1.399ms. CPU launch times may lead queued GPU work. This does not establish
the 0.1ms GPU-entry goal, and no new GPU-entry speed claim is made.

Two bounded natural completions terminate normally. Sixty-four candidate teacher distributions are retained. Fifty-six match
the previous artifact's exact prefix, forced token and position; mean KL
is 0.000986, maximum 0.010389 and top-1 agreement 55/56. Eight changed-prefix
positions are excluded. Acceptance across eight natural prompts increases
44.802 to 47.170%, paired difference CI [+0.792,+4.439] percentage points.
These compare separately loaded artifacts, rather than isolating integer
input preparation. Future reports save complete teacher prefixes and reject
unrecoverable legacy conditions instead of comparing different histories. The 18.5ms C1 stage target remains open.

The previous graph-node trace separates asynchronous output materialization:
median 24.116ms total contains 24.086ms in cudaEventSynchronize. Work after
that wait has median 30.311us and p90 84.625us. These profiled edge-inclusive
ranges do not support treating the full materialization interval as CPU
serialization overhead.
