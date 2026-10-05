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
