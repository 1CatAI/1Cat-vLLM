# SM70 memory reuse defaults

Implementation and matched validation for PR671, based on main `949728e891`.

## Defaults and compatibility

`VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT=1` removes duplicate QPN2 codes on
compatible local NVFP4 projection layouts. `VLLM_SM70_NVFP4_QPN2_SHARED_SCALES=1`
also removes persistent TurboMind FP16 scales. Both now default to one;
explicit zero retains the previous storage choice. No new TP-count,
model-name or concurrency allowlist was introduced. Existing layout,
DFlash2 and numerical contracts still apply.

Compact scales require native `nvfp4_qpn2_compact_scales_version_sm70() >= 1`.
An older extension can expose the original compact-scale operator while
still retaining an allocation per captured layer. Such extensions keep
persistent FP16 scales until rebuilt. Shared codes retain their existing
native capability check.

TurboMind fallback restores each layer's original E4M3/global scale product
with FP32 multiplication and FP16 rounding into scratch retained per device,
CUDA stream and matrix size. Graphs keep stable addresses and later calls
reuse storage. Configured capture sizes above 32 are supported. The scratch
lifetime follows TurboMind's existing stream contract; worker graph replays
must remain ordered. QPN2 and large-prefill arithmetic are unchanged.

Q8000 and Q8192 share one maximum-sized score buffer, common host lock and
completion event. Captures use [external event nodes](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html#external-events-and-streams)
for dependencies originating outside the current graph. The tail aliases
the prefix score buffer by default; `PREFIX_TORCH_SERIAL_TAIL=0` restores
concurrent tail allocation when memory permits. Block size, FP32 QK/PV
accumulation and reduction order remain unchanged.

SM70 V2 reserves one measured steady-state activation peak for graph pools
before allocating KV. `VLLM_V2_CUDAGRAPH_MEM_MIB` explicitly overrides it,
including zero. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` disables this
budgeting. V1 retains its previous policy. This is an admission estimate,
not a hard CUDA allocator ceiling or an exact prediction for every model.

## Matched TP4 evidence

V100-SXM2-32GB,185W throughout, CUDA12.8, Torch2.10.0+cu128;
Qwen3.8-27B QUASAR NVFP4, TP4, DFlash2/7, target E4M3 and draft FP16 KV,
max length262144, chunk8192, max sequences32, memory utilization0.85.
Both services use FULL_AND_PIECEWISE CUDA Graph. No wheel builds or
`--enforce-eager` serving benchmarks. Every measured prompt starts after
prefix-cache reset; standard bench uses fixed2048/256 lengths and
quality requests use natural EOS.

| Per-GPU measurement | Main control | Candidate |
| --- | ---: | ---: |
| Model loading, GiB | 10.07 | 6.47 |
| Available KV budget, GiB | 10.57 | 12.68 |
| Logical KV capacity, tokens | 922,965 | 1,107,285 |
| Graph budget estimate, GiB | 0 | 2.21 |
| Actual graph capture allocation, GiB | 1.75 | 1.80 |

Automatic KV sizing reuses freed model memory. Total NVML usage therefore
need not fall by the same amount as weight storage. The graph estimate
leaves approximately0.41 GiB beyond this run's capture allocation.

Q8192/KV262144 operator graph replay uses3,259,010,048 versus2,843,773,952
bytes of persistent workspace:396 MiB saved. Median of five samples of
five graph replays:241.345 versus243.447 ms (+0.871%). Both outputs are finite,
FP32 accumulation is reported by the native capability query, and complete
FP16 output bytes match (SHA256 `f03b24f84e23eb549a002623b721bb28e63dfa5ba6a5115af0447390dbddb38f`).
The second query family reuses scores instead of allocating another
approximately2.2 GiB buffer; fresh-process tests cover alternating and
combined captures with repeated replay.

| vLLM bench concurrency | Main output tok/s | Candidate output tok/s | Main median TTFT, ms | Candidate median TTFT, ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 177.168 | 179.429 | 565.371 | 576.195 |
| 2 | 180.191 | 192.070 | 892.775 | 882.946 |
| 4 | 166.565 | 228.093 | 2139.433 | 2183.926 |
| 8 | 178.040 | 189.718 | 3412.361 | 3401.229 |
| 16 | 196.159 | 231.333 | 5721.327 | 5720.470 |
| 32 | 246.210 | 235.983 | 13379.222 | 13335.549 |

All six cases complete without request failures. These are complete-request
output throughput, not pure decode. Inputs and sampling settings match,
but bench's dataset seed does not set the request generation seed.
C32 candidate repeat214.214 tok/s has acceptance length3.680 versus4.598
in its first run and5.193 in the control. Separate controls with request
generation seed20260922 give:

| C32 diagnostic | Main output tok/s | Candidate output tok/s | Main median TPOT, ms | Candidate median TPOT, ms |
| --- | ---: | ---: | ---: | ---: |
| Repeat1 | 224.516 | 227.854 | 34.314 | 40.548 |
| Repeat2 | 223.094 | 223.841 | 36.243 | 42.429 |

Mean aggregate throughput changes by+0.913%, while per-request latency
increases. Main reaches13 resident requests (mean9.90/10.16); the larger
candidate KV pool reaches15 (mean12.06/12.17). These runs do not isolate
kernel latency from scheduling and acceptance. Within each implementation,
only10/32 repeated generated texts match; fixed request seeds alone do not
establish batch-invariant output. Retain the primary rows above rather than
replacing them with these diagnostics.

| Cold long request | Main TTFT, s | Candidate TTFT, s | Result |
| --- | ---: | ---: | --- |
| 32768 input | 9.050 | 9.044 | Correct, natural EOS |
| 131072 input | 47.604 | 47.840 | Correct, natural EOS |
| 262128 input +16 output | 132.151 | 132.884 | Correct, natural EOS at262144 |
| 256000 input +256 output bench | 123.681 | 124.182 | Both complete |

The256000-input TTFT implies2069.8 versus2061.5 input tok/s (-0.40%).
This is at185W; it must not be compared as a code regression against an
older300W run. The separate full-FP32 75T and 35B migration targets remain open.

## Quality, memory admission and failure ledger

The main MBPP32 control passes 25/32; 31 answers stop naturally and one
(index164/task304) reaches 65536 generated tokens. The candidate passes
24/32 with all 32 answers stopping naturally. The only score flip is task279
(index144): its prompt asks for the nth decagonal number but its entry
function is named `is_num_decagonal`. The candidate returns a membership
predicate from that entry, putting the nth-number formula in a separate
helper. Serial C1 with the same sampling and seed also gives main pass
(21814 output tokens) and candidate fail (14700); both stop naturally.
This is an observed one-question difference, not proof of equivalence
or proof that the memory change causes a general quality regression.
The complete C1/temperature-zero diagnostic gives both implementations
15756 output tokens, natural EOS and a passing answer. The complete
reasoning and final-answer strings match exactly. These diagnostics do not
replace the primary scores or establish global accuracy equivalence; they
do not reproduce a deterministic loss from the memory changes. In particular,
do not label the sampled one-question difference as proven precision loss.

A third process uses the candidate's exact native libraries with shared codes
and shared scales explicitly disabled. It also generates15756 tokens, passes,
and returns identical complete response text. Eight teacher-forced prefixes
(0/16/64/128/256/537/1024/4096 reasoning tokens, including the first sampled
divergence) return identical next tokens and identical API top20 logprob maps
with sharing on/off. These are observed text and probability comparisons;
the chat endpoint did not return the full generated token-ID sequence or
full-vocabulary logits. The tests find no storage-induced numerical loss;
they do not establish the exact proposal/rejection step responsible for the
sampled trajectory difference. Do not rerun sampled sweeps to select a score.

The current native build passes504 ordinary/gated real-projection cases
across all four TP4 shards, six projection types, M=1/8/16/32/33/64/93/128/1024,
and both shared-code and compact-scale comparisons. Outputs match every
FP16 bit in direct warmup and changed-input CUDA Graph replay. Code layouts,
packed E4M3 scales and restored effective FP16 scales also match exactly.
This isolates projection arithmetic; it is not whole-model quality acceptance.

TP1/TP2 inventories use the same model, DFlash2, KV types, chunk8192,
max sequences32, utilization0.85 and CUDA Graph; max length is65536.

| Per-GPU measurement, GiB | TP1 on32GB | TP2 on2x32GB |
| --- | ---: | ---: |
| Model loading | 23.142 | 12.167 |
| Live torch allocation after profile | 25.996 | 15.030 |
| Profile activation peak increase | 2.645 | 2.110 |
| Graph reserve estimate | 2.645 | 2.110 |
| Available KV budget | -6.557 | 6.766 |
| Ready device memory | startup rejected | 24.724 |
| Device memory after32K retrieval | not run | 28.004 |

TP2 returns all four retrieval values correctly and stops naturally. Its
after-request live torch allocation is22.648 GiB, with26.943 GiB reserved;
device usage includes allocator cache and non-torch allocations. TP1 fails
before KV allocation: this configuration still does not fit a32GB device.
The existing warmup-residual accounting includes retained allocator segments,
so the negative budget must not be interpreted as an exact physical deficit.
No physical16GB GPUs are available, and these results do not establish
2x16GB support. Lower memory layouts remain separate work.

Forty GPU tests pass on the final formatted attention binary:24 scale
basis cases, two changed-input compact-scale graph cases, nine attention
stability/replay cases and five worker-budget cases. Nineteen CPU checks
cover defaults, rollback, older native extensions and explicit graph budgets.

The initial shared-event capture failed because it attempted an ordinary
wait on uncaptured work. External event nodes repair it; separate and
combined graph replay tests pass. The scale-memory test now warms the same
capture stream before measuring, excluding TurboMind's own stream scratch.

A binary-section audit accidentally invoked `objcopy --dump-section` without
an output filename on a loaded library. It rewrites the input with
`O_TRUNC`, reproduced on a disposable copy, and interrupted the initial128K
request. Completed C1-C32 and32K results predate the interruption. Fresh
service128K/262144-boundary and256000/256 tests pass; this interrupted run
is not counted as an attention-kernel failure or a completed quality audit.
Subsequent binary inspections use copies only.
