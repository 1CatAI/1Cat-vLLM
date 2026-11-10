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

Per-position rates use all drafts as denominator, rather than conditioning on
the previous position being accepted:

| Draft position | GGUF | NVFP4 | Paired difference95% interval |
|---|---:|---:|---:|
|1|70.461%|72.545%|−4.071 to−0.388pp|
|2|48.957%|49.892%|−4.262 to+2.038pp|
|3|36.110%|33.957%|−0.853 to+5.387pp|
|4|26.352%|24.421%|−0.084 to+4.439pp|

The first-position rate is lower on this set, while later-position rates
compensate in the mean length. Overall acceptance parity does not imply that
every position has identical behavior between the two quantized checkpoints.

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

The first graph-node collection follows below; the final same-artifact check is recorded at the end. Nsight2024.6.2 with NVTX tracing fails
during NCCL NVTX initialization. Nsight2026.2.1 passes initialization but
produces no report even for a single-process256-kernel CUDA smoke on this
system. CUDA-only2024.6.2 with fork tracing captures all four NCCL ranks.
Use `--kill=none` so ending collection does not terminate the application.
The CUDA-only ledger strips Nsight's system-ID bits from process identifiers
and selects the largest repeating graph, checking its launch count against
the independent M5 CPU records. A regression using the retained historical
trace reproduces2175 nodes on each rank across59 middle rounds; this is
parser validation, not a new model measurement.

A whole-wheel native template passes CPU checks but fails model startup with
rank1 cuBLAS allocation failure; it is not model-qualified. An owned SM70
native build then passes29 CPU and25 GPU checks, including official expert
dequantization, changed-input graphs and mixed-output bitwise comparisons.
Its ordinary wheel uses source `b8dbe0d0ae715b2b5e12889ecb1115fe8ec3c1f4`, with core
`d9a0d86934a412c1847d20ce4422216d93b720d1f25019568e0d650e3117f6e6`.
The source build retains existing CUDA objects and rebuilds the new CPU
gather registration through the normal CMake/package path. Neither failed
startup nor profiled service time supplies an unprofiled speed result.

An asynchronous-output event experiment reduces waiting-thread CPU from
129.62ms busy waiting to0.066ms blocking in a150ms GPU-wait proxy, but median
host position launches remain67.47 versus66.18µs. It does not explain the
model's rank0 launch discrepancy, so no production event-wait change is made.

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

## CUDA-only target graph ledger

Fresh CUDA-only captures use the same installed SM70 wheel (`b8dbe0d0ae715b2b5e12889ecb1115fe8ec3c1f4`),
TP4 MTP4 and verifier M5. Each rank has1,799 GGUF target nodes and1,527
NVFP4 target nodes, a272-node difference. The historical2,175/1,428 counts
are different snapshots and must not be used as this comparison's control.
The GGUF capture retains12 middle rounds per rank. Its short128-token run
completed generation but failed the optional interval summary before saving
CPU events; its ledger is explicitly GPU-only. NVFP4 retains81 middle rounds,
checked against97 independent M5 CPU replay records before trimming.
The collector now saves CPU records before optional statistics and captures256
output tokens. Profiling service durations do not establish round latency.

| Operation family | GGUF calls | NVFP4 calls |
|---|---:|---:|
|HC|386|386|
|Router|96|96|
|Shared gate|48|48|
|QSA|72|72|
|TP ring|98|98|
|FP16 row GEMV|36|0|
|Tensor copies|41|5|
|GGUF affine bitplane GEMM|131|0|
|GGUF LUT GEMM|87|0|
|Other TurboMind GEMM|41|0|
|FP16 CUTLASS|16|76|
|Raw expert gate/up|47|0|
|Canonical expert down|48|0|
|Expert route/gather|48|0|
|Expert unroute|48|0|
|PLE n-gram|1|1|
|Packed PLE row gather|1|0|
|PLE dequantization|1|0|
|Tensor scatter|1|0|
|Other|552|745|

GGUF raw joint gate/up comprises17 IQ3_XXS,20 IQ2_S and10 IQ3_S calls.
IQ2_S retains its canonical M20 fallback. Neither trace has torch sort or
searchsorted in the target graph. The36 additional copies occur between the
GDN gated normalization and its Q6_K output projection: GGUF head-tiling
restoration currently permutes activations on every forward. Restoration of
canonical weight groups at load time is investigated separately. Six mixed
GDN input projections and shared gate/up projections already write directly
into merged output views; the remaining dense format/tile tuning belongs to
the GGUF operator line.

All common HC/router/QSA/GDN recurrent kernels appear in the GGUF trace.
NVFP4 combines the GDN input projection where GGUF uses its quantized
projections plus the FP16 a/b row GEMV and split. NVFP4 also combines shared
expert operations and has a different expert plan/W13/W2 organization.
Those are operator replacements, not copies removable by deleting an op.

The CUDA-only GGUF API entry spread has median113.427us over12 middle rounds;
NVFP4 has138.440us over81 rounds under profiling. These are not substituted
for the unprofiled early-input A/B result. Each rank's attention preparation
contains two synchronous8-byte host-to-device transfers. A benchmark-only
one-shot dispatcher records blocking host transfers or CPU tensor indices
with source stacks; it adds no fence and is not enabled in production.

## Final combined check

The final pair uses the same ordinary wheel from source `615abb0108`, with
core SHA `d4604bbee4f70346eb23dc8b08dd071a7209de380206a35497c802123d0f7f58`.
It includes prepared GDN output head order and asynchronous metadata index
reuse. The workload above is unchanged. Eight natural requests per route
produce 600 tokens each, reach the length limit and contain plausible text;
this does not establish EOS termination or identical quantized-model outputs.

| Metric | GGUF mean | NVFP4 mean | Paired difference | Difference 95% interval |
|---|---:|---:|---:|---:|
|Draft acceptance|44.246%|45.204%|−0.957pp|−3.347 to +0.929pp|
|Length including bonus|2.7699|2.8081|−0.0383|−0.1339 to +0.0372|

Acceptance mean intervals are 34.165–56.750% and 34.378–57.973%; length
mean intervals are 2.3666–3.2700 and 2.3751–3.3189. The first-position
means are 70.509% and 72.545%, with paired interval −5.095 to +0.645pp.
All four position intervals now span zero. This set does not reproduce a
statistically significant overall decrease, but eight clusters cannot prove
equivalence or exclude a modest decrease.

|Route|C1 before observer(ms)|C1 observed(ms)|C1 after observer(ms)|C4 mean(ms)|
|---|---:|---:|---:|---:|
|GGUF|23.2973|23.2519|23.6183|48.1459|
|NVFP4|20.9488|20.6814|20.5529|46.3875|

GGUF C1 controls have identical complete IDs, including the earlier head-order
check. C4 differs from the earlier head-order run, so its absolute timing is
not a same-trajectory implementation speedup. GGUF C4 emits 2,160 tokens over
233 steady intervals (192.55 tokens/s aggregate); NVFP4 emits 2,099 over 217
(208.52 tokens/s). Whole-round timings include draft and host work; graph
service sums must not replace these measurements.

Each target graph now contains 1,763 GGUF nodes and 1,527 NVFP4 nodes, over
38 and 81 middle rounds on each rank. The 36 head-order copies disappear:
Tensor copy counts are 5 versus 5. The remaining net difference is 236.
HC(386), router(96), shared gate(48), QSA(72), ring(98) and n-gram(1)
counts match. GGUF has 48 expert route/gather and 48 unroute calls, plus
47 raw joint gate/up and 48 canonical down calls; NVFP4 uses 48 plan,
48 W13 and 48 W2-reduce calls. The standalone expert activation exists in
both routes. GGUF shared gate/up still needs standalone SiLU and multiply
(48 each), whereas NVFP4 uses a different shared dense batch organization.
These are computation/fusion differences, not removable copies. Shared-expert
and GDN core fusion work remains with the common Flash-Next decode line.

GGUF projection families contain 131 affine, 87 LUT and 41 other TurboMind
calls. NVFP4 has 60 more CUTLASS calls, 36 combined GDN input calls and a
different shared-projection organization. GGUF retains 36 FP16 a/b row GEMV
and 36 projection splits. No GGUF dense projection is dequantized wholesale
inside this graph. Its sole dequantization follows the packed PLE row gather;
a single scatter/gather then copies rows using an identity index. Direct row
decoding is qualified separately. Neither graph contains sort/searchsorted.

Actual GPU first-node entry spread is median 49.856us, p90 407.928us and
maximum 14.722ms for GGUF; NVFP4 is 41.741us, p90 63.406us and maximum
168.565us. GGUF meets the 100us median target, but its tail does not. CPU
replay submission remains asymmetric: median spread 1,366.582us versus
998.141us for NVFP4. Rank0 is latest in all 38 GGUF CPU rounds and also owns
asynchronous output materialization. CPU draft-proposal medians vary from
roughly 1.6 to 2.8ms across ranks; these durations include queued GPU work
and cannot be called pure CPU service. GPU entry is much closer because
submission overlaps preceding draft execution. No barrier was added.
First-allreduce median services are 4.512/23.040/55.072/50.511us across the
GGUF ranks, rather than a millisecond. CPU asymmetry and the profiled entry
tail remain open; a median alone is not evidence that all rank skew vanished.
Real-model prefix API attribution records zero cudaStreamSynchronize calls
on every rank after the metadata change, matching its independent microcheck.

The merged IQ3_S dense gated-pair route has exact geometry M8/N4352/K5120;
its IQ3_XXS extension uses the same dense specialization. Flash-Next dense
Q6_K/Q4_K/IQ4 projections at hidden width2560 do not match those capabilities.
The raw expert route is separate and hits all17 IQ3_XXS layers. IQ2_S M20
keeps canonical fallback. No unmerged dense decoder or duplicate GEMV is
introduced on this line.

The newly merged floating-shard specialization admits only converted FP16
sources1/30 with M8 and N12 or24/K5120. Flash-Next uses M5/M20 verification
and K2560, so it is not admitted by that specialization. Its ordinary dense
GDN a/b tensors already hit the common FP16 row GEMV (36 calls per target
graph). The merged IQ3_S/IQ4_XS mixed native pair also admits M8/N4352/K5120;
Flash-Next shared gate/up uses Q4_K/IQ4_XS at different geometry. These
capability exclusions are expected. Changing those operator shapes remains
with the GGUF dense operator line, without adding duplicate implementations.
