# Flash-Next GGUF graph parity and MTP4 acceptance

## Fixed workload

The paired run uses ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_S GGUF and
RadixArk Qwen3.8-Flash-Next NVFP4. Both use the same FP16 MTP4 draft weights,
TP4 on GPU0–3 (V100-SXM2-32GB), CUDA12.8, Torch2.10.0+cu128, Python3.12,
FP16 activations/KV, FP32 recurrent state and accumulation, FULL graphs,
max length9216, max batch512, four sequences, memory utilization0.95,
prefix caching disabled and metric collection enabled.

Eight natural prompts are shipped in benchmarks/flashnext_acceptance_prompts.json.
Greedy sampling overrides model defaults explicitly; EOS is respected, max
output is600. Both tokenizers produce identical chat input IDs. All eight
requests in both runs produce600 tokens and stop at the length limit; this is
not EOS-termination evidence. Texts are plausible writing, explanation, code
and planning in Chinese and English.

## Paired acceptance

Intervals use20,000 deterministic paired bootstrap resamples with prompt as the
cluster unit. They do not treat4,800 output tokens as independent observations.

| Metric | GGUF mean | NVFP4 mean | Difference | Difference95% interval |
|---|---:|---:|---:|---:|
| Draft acceptance |45.470%|45.204%|+0.266pp|−1.468 to+2.057pp|
| Length, including bonus |2.8188|2.8081|+0.0107|−0.0587 to+0.0823|

The earlier large acceptance decrease is not reproduced on this set. Eight
prompt clusters do not establish equivalence for all workloads.

The individual 95% mean intervals are 34.789–58.168% for GGUF acceptance
and 34.378–57.973% for NVFP4. Mean-length intervals are 2.3916–3.3267 and
2.3751–3.3189 respectively. Prompt-to-prompt variation is much larger than
the paired difference.

## Complete-round probes

Separate synthetic probes respect exact token budgets and ignore EOS to keep
verification geometry fixed. They are performance probes, not quality tests.
C1 uses I8192/O256; C4 uses I128/O600. Steady intervals trim eight eligible
rounds at both ends. Primary reports use an ordinary source-containing wheel,
not graph-node profiling.

| Route | C1 before observer (ms) | C1 observed (ms) | C1 after observer (ms) | C4 mean (ms) |
|---|---:|---:|---:|---:|
| GGUF |24.2522|24.1096|25.4961|49.7277|
| NVFP4 |21.3536|21.4406|22.1936|47.2284|

The last control arms contain outliers. These absolute values are not a matched
gain against the earlier26.07/52.06ms workload, which used different prompts
and limits. Different speculative trajectories also affect emitted-token
throughput; verification-round time alone is not the MTP throughput metric.

## Replay entry observations

The benchmark-only worker extension timestamps the actual CUDAGraph replay
call after manager-side waits, along with input, attention, sampling, PLE and
asynchronous output stages. It adds no GPU fence. CPU stage wall durations can
include waits for preceding GPU work; nested durations are not additive.

| Route | Rounds | Median spread (µs) | p90 (µs) | Latest rank |
|---|---:|---:|---:|---|
| GGUF |38|109.804|206.838|rank3 in31 rounds|
| NVFP4 |81|647.349|769.847|rank0 in79 rounds|

GGUF worker-entry spread has median529.27µs. Attention-metadata preparation
includes about20ms of GPU waiting, ending with only53.78µs inter-rank spread.
Subsequent model-input preparation expands this to100.99µs; actual replay
entry reaches109.80µs. The post-attention tail is255–303µs, including201–230µs
for model input preparation. This motivates moving independent position
preparation before attention metadata while preserving current-stream order.
The same-engine phase comparison uses the ordinary wheel from `173fc0b97a`.
All four C1 arms have identical complete output IDs; both C4 arms have
identical IDs in all streams. Unobserved C1 round means change from24.3621ms
late to23.9364ms early (35 middle intervals,171 tokens). C4 changes from
49.2177 to48.9204ms (232 middle intervals,2171 tokens). This comparison does
not mix different speculative trajectories.

In38 matched middle observed rounds, median actual replay spread changes
from547.455 to202.568µs. The early post-attention tail is45.5–50.9µs across
ranks; the remaining spread already exists at attention preparation exit.
Rank0's position-launch wall is620µs versus246–251µs on other ranks, and only
rank0 handles asynchronous output materialization/serialization. Reordering
hides that position launch but does not meet the100µs objective. Further
diagnosis needs the attention-wait/output boundary. The observed arms have
large scheduling outliers and do not replace unobserved speed measurements.

The fresh graph-node ledger is pending. Nsight2024.6.2 fails during NCCL NVTX
initialization before weights load. Its failed path is retained; a normal
four-rank NCCL initialization succeeds under Nsight2026.2.1. Neither failure
nor graph-node service time supplies an unprofiled performance result.

## Prepared GGUF coverage

All four ranks prepare17 IQ3_XXS gate/up layers for original M1/M5/M20,
20 IQ2_S layers for M1/M5, and10 IQ3_S layers for M1/M5/M20. IQ2_S M20 retains
canonical grouped fallback. Original rows add2.552GiB per rank for IQ3_XXS.

Dense GDN QKV uses Q6_K at local N2560/K2560, Z uses Q4_K at N1536/K2560,
and output uses Q6_K at N2560/K1536. Shared gate/up includes Q4_K/IQ4_XS at
N160/K2560. Existing IQ3_S/IQ2_S native dense candidates do not cover Q6_K.
The IQ4_XS source-layout reader provides preparation/oracle support without
changing runtime dispatch.

HC modules, row GEMV, GDN projection tails, router top-k, shared-expert gates,
QSA batch selection and FlashQLA report preparation/route hits. Graph-node
inspection must confirm the final captured kernel composition.
