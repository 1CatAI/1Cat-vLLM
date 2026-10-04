# GGUF target and DFlash2 on SM70

Load Qwen3.8-27B GSQ-RCO IQ3_S as the target and the Q8_0 DFlash2 GGUF as
the default draft. Preserve FP16 projection operands, FP32 accumulation and
the existing BF16 range-preserving draft arithmetic. Shared embedding and
LM-head weights remain target-owned. The complete verification round includes
target forward, rejection sampling and draft forward; its single-request goal
is below 12 ms on four fully connected V100s with TP4.

## Loading boundary

The native DFlash metadata adapter recovers five layers, 32/8 attention heads
of dimension 128, a non-causal 2048-token sliding window, an eight-token
trained block, convolution and selector dimensions, and the mask token ID.
GGUF extraction layers are 1-based; convert them to HF indices
`[5, 19, 33, 47, 61]`. Declare BF16 in the config so existing SM70 range
preservation and output scaling remain active under FP16 runtime transport.

Map all 81 checkpoint tensors directly to the draft loader. Retain canonical
quantization for the backbone and context projection. Decode the convolution
projections and selector tables into the dense parameters those modules own.
Keep Qwen3 norm weights unchanged. The SM70 context projection uses TP4
output sharding with compact all-gather for GGUF as well as dense drafts.
Dequantization already allocates a new array; avoid a second dense copy before
dtype conversion, especially for the target embedding table.

Name and format references: llama.cpp `conversion/qwen.py` and
`gguf/tensor_mapping.py` at `bed0a856606ee4a24a164066f73d2379447033f5`
(MIT). No llama.cpp kernel shape tuning is part of this work.

## Checkpoints

| Checkpoint | Bytes | SHA256 |
| --- | --- | --- |
| Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf | 11771546784 | `64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810` |
| Qwen3.8-27B-DFlash2-Q8_0.gguf | 2056414816 | `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e` |

Both complete files pass SHA256 verification. The target ModelScope revision
is `fc62412a020e0c1ee3a8f9a12f10aea2102478f0`; the draft Hugging Face revision
is `2d9571f8ce46e151f61c6499c99dee6079e1d610`. An incomplete HF transfer is
retained as a failed path; the ModelScope mirror has the same draft SHA256.

## Measurement contract

Develop and measure on the same four SM70 V100s, all pairs NVLink connected,
with power limit 300 W. Set PCI bus device ordering and expose GPUs 0–3.
Single-GPU probes hold that GPU's lease. Four-GPU runs hold the aggregate
lease, the shared 0–3 lease and all four individual leases. Preserve other
processes and use separate source, runtime and mutable compiler caches.

Operator probes use actual checkpoint weights and TP4 shapes, M=8 with
M=1–16 coverage, cold L2 and repeated calls inside CUDA graph replay. Report
physical bytes, GB/s, latency delta and error against official FP32 GGUF
dequantization. Layer graphs cover full GDN, attention, draft and head tails.
Use full-model runs at integration boundaries or after at least about 1 ms of
predicted savings. Final runs use maximum length 262144, 1K/8K inputs,
temperature 0.7 and thinking disabled.

The kernel ledger records calls per round, physical bytes, the 750 GB/s
bandwidth floor, measured service and launch grids. Project savings as latency
delta times calls, distinguishing overlap and using the measured profiler
correction rather than treating service sums as wall time. Investigate
prediction errors above 15%. Compare Q8_0 and BF16 drafts with at least eight
seeded prompts and 600 emitted tokens per prompt, reporting mean tokens per
round and confidence intervals.

Numerical changes require teacher-forced mean KL ≤0.001, p99 ≤0.01,
maximum ≤0.05, top-1 agreement ≥99% and maximum logit difference ≤0.5.
Validate rejection samples against the dense reference token by token. Check
fixed prompts against llama.cpp, the fixed quality suite including 128K and
258K needles, and C4 before promotion. No additional precision reduction is
authorized. Source checks pass; GPU loading, graph, numerical and model-speed
results remain pending.

## Installed loading checks and storage budget

The installed normal wheel passes 78 CPU checks on Python 3.12.14,
Torch 2.10.0+cu128 and CUDA 12.8. Nine changed modules match the source,
wheel members and installed files exactly; all sixteen native libraries match
the qualified normal base wheel. The release profile accepts local GGUF draft
files. Packed context projections expose their declared operand dtype to
auxiliary-state conversion without changing the existing runtime transport.
These are package and CPU checks; they do not establish GPU model quality or
throughput.

The verifier capability gate admits both the multimodal wrapper and native
text-only Qwen3.5 architecture with the same dtype, head dimensions, draft
width and scheduling guards. Previously a standalone GGUF text model missed
this gate and disabled the quantized LM head and sharded context projection.
The initial loading probe was stopped before generation; its logs remain a
negative result. Positive and incompatible-head-dimension CPU cases cover the
corrected gate. The expanded policy suite also required adding the existing
local argmax field to its speculative-hash test fixture.

The actual target has 40 mixed gate/up layers and 35 mixed GDN qkv/z layers.
There are 72 distinct checkpoint role/type/TP4-shape combinations across the
target and draft. Predicting storage from the current canonical code and
metadata streams gives:

| Storage | Per-rank bytes |
| --- | --- |
| Target checkpoint body excluding embedding/head | 2659536384 |
| Canonical target projection buffers | 3175956480 |
| Checkpoint head, one read | 178790400 |
| Canonical head, one read | 198656000 |

The canonical body estimate is about 19% larger than checkpoint storage.
Expanded coefficients and index/sign metadata therefore need to be included
in bandwidth accounting. IQ3_S grows from 0.4296875 to 0.5 bytes per weight;
IQ2_XS grows from 0.2890625 to 0.5. These are CPU format predictions for
aligned projections, excluding norms, codebook loads, inputs/outputs and
workspace traffic. Verify them against the loaded GPU buffers before using
them in a bandwidth or complete-round speed claim.

## Sampled reconstruction accuracy

An installed-wheel CPU audit reads the first eight actual rows of each unique
checkpoint quantization type and full shape: 55 cases across both files.
Compare canonical dequantization against `gguf.quants.dequantize` in FP32,
then separately include FP16 weight reconstruction. All samples are finite.
The table reports the worst relative L2 within each type.

| Type | Cases | Canonical relative L2 | FP16 reconstructed relative L2 |
| --- | ---: | ---: | ---: |
| IQ1_M | 1 | 0.0002155 | 0.0003067 |
| IQ2_S | 5 | 0.0002143 | 0.0002898 |
| IQ2_XS | 3 | 0.0002133 | 0.0002908 |
| IQ2_XXS | 3 | 0.0002092 | 0.0002861 |
| IQ3_S | 7 | 0.0002186 | 0.0003044 |
| IQ3_XXS | 7 | 0.0002145 | 0.0003027 |
| IQ4_XS | 7 | 0.0001954 | 0.0002876 |
| Q2_K | 5 | 0.0005346 | 0.0005486 |
| Q4_K | 8 | 0.0006983 | 0.0007226 |
| Q8_0 | 9 | 0 | 0.0002176 |

The largest sampled absolute canonical error is 0.0001231 for IQ2_S.
These results audit the existing coefficient expansion and weight
reconstruction. They exclude MMA accumulation, unsampled rows and model
propagation; the teacher-forcing distribution gate remains required.

## Embedding loading memory

The first full TP4 weight-loading probe exhausted host RAM. The kernel OOM
record identifies rank 0 as the killed process; the four workers held roughly
13–17 GB RSS each. Whole-table IQ2_S dequantization and finite-range checks
produce temporary arrays proportional to the full vocabulary in every worker.

Decode dense vocabulary tables in 1024-row chunks, convert each chunk to the
requested dtype and perform the same finite-range overflow check before
copying into the final global table. The existing TP loader still owns row
sharding. Regression tests check bounded decode batches, exact converted
values and FP16 overflow rejection. Sixteen actual IQ2_S embedding rows,
decoded in three-row chunks, match the official FP32 dequantization followed
by FP16 conversion bit for bit, with maximum difference zero. The rerun of
target loading completed without a host OOM. A loading sample showed about
6.2 GiB anonymous resident memory per worker; the checkpoint mapping is
shared file-backed memory and must not be summed as four private copies.
This sample does not establish the final anonymous memory peak.

## Draft construction and local side files

Pass the explicit model configuration to GGUF model initialization. A
speculative worker carries both target and draft configurations; relying on
the implicit configuration constructed a second target backbone and collided
with existing GDN layer registrations. A regression uses distinct configs.

Local GGUF drafts resolve the optional `mask_embedding.pt` beside the
resolved checkpoint. Missing side files retain the shared target embedding.
Repository-based drafts preserve their existing file lookup. Six regression
cases cover a directory, a direct GGUF path and a resolved GGUF path, with
and without the optional file.

The loading rerun reached the correct DFlash2 constructor and enabled the
TP4 output-sharded 25600-to-5120 context projection and existing
range-preserving arithmetic. It then stopped when the local GGUF filename
was passed to a repository lookup. The fix passes the installed-wheel CPU
suite. Full target and draft loading and engine warmup now complete on all
four ranks, with about 4.99 GiB of model allocation per rank. Startup takes
485 seconds; this is initialization time, not inference latency. A diagnostic
call stopped the probe before generation because safe serialization rejects
function-valued RPC arguments. Use the existing worker-extension interface
and a named RPC for the read-only buffer inventory. Secure message encoding,
method-name compatibility and spawned extension import pass CPU checks.
The named-RPC rerun completes three natural prompts at temperature 0.7 and
seed 123 with thinking disabled: Paris, 4 and a coherent Chinese explanation
of Rayleigh scattering. All stop naturally (2, 2 and 55 output tokens). This
is a basic loading/generation check, not the fixed quality or acceptance gate.
Graph speed and distribution/acceptance/quality gates remain pending.

## Loaded TP4 projection buffers

The read-only worker inventory agrees across all four ranks. Every draft
range-preserving flag observed is enabled (six modules per rank).

| Buffer group | Per-rank bytes |
| --- | ---: |
| Target canonical projection streams | 3174973440 |
| Draft canonical projection streams including context FC | 486604800 |
| TP4 context FC, included in draft streams | 36864000 |
| Ten dense convolution projections | 131072000 |
| Two dense selector tables, stored | 254279680 |
| Selector hidden projection | 2621440 |

The context FC partition has K=25600 and N=1280. Selector tables are read by
selected rows; their complete storage size is not per-round traffic. These
figures describe allocated weight buffers, excluding codebook constants,
activations, outputs, workspaces and other parameters. Calls and physical DRAM
traffic still require the baseline trace and operator counters.

The 8192-token loading probe rejects the grouped verifier capability because
its minimum model length is 32768. Short graph baselines use max length 32768;
the final comparison remains 262144. FP16 activation/KV and FP32 SSM state
remain the same.

## Preliminary graph and operator measurements

The first unprofiled TP4 graph run uses FP16 activation/KV, FP32 SSM,
max length 32768, batch budget 1024, one sequence, memory utilization 0.9,
seven draft tokens, temperature 0.7, seed 123 and no thinking or prefix cache.
Both target and draft graphs capture successfully; the graph pool is about
0.55 GiB per rank.

| Actual input tokens | Output tokens | Draft rounds | Accepted drafts | Mean tokens/round | Mean full round ms | Median full round ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 916 | 128 | 32 | 95 | 3.96875 | 46.9065 | 46.5396 |
| 7242 | 128 | 30 | 98 | 4.26667 | 47.4787 | 47.0882 |

These are engine-core output timestamp intervals, including scheduling and
communication, with prefill excluded. GPU trace attribution and dense-reference
branch counts remain required. Re-encoding repeated decoded text merged BPE
boundaries, so this run does not meet the exact 1024/8192 input contract.
The corrected producer inserts token IDs between the embedded chat prefix and
suffix; CPU checks with the actual tokenizer verify exactly 1024/8192 tokens
and preserve the thinking-disabled assistant suffix.

The following standalone measurements are exploratory: their wrappers used an
incorrect single-GPU lease filename. They checked for an idle device before
launch, but require repetition under the prescribed lease before performance
conclusions are accepted. The wrappers are corrected.

Independent CUDA extensions preserve FP16 weight reconstruction and FP32 MMA
accumulation. Probes use actual checkpoint rank-0 TP4 slices and M=1/2/4/8/16.
The reference is official FP32 GGUF dequantization plus FP32 cuBLAS. Each graph
has twelve repeated calls with a 16 MiB L2 flush before each call; the median
covers seven replays. Timing events use external record nodes. Internal graph
events did not provide timestamps, invalidating the first timer attempt;
those failed measurements are excluded.

| Tensor and local N/K | Canonical M8 us | Best research M8 us | Raw model head M8 us |
| --- | ---: | ---: | ---: |
| IQ3_S gate, 4352/5120 | 37.376 | 48.128 | — |
| IQ3_S down, 5120/4352 | 35.840 | 44.032 | — |
| Q4_K head, 62080/5120 | 273.408 | 295.936 | 620.032 |
| Q8_0 context FC, 1280/25600 | 75.776 | 156.672 | — |

The lattice samples use blk.5.ffn_gate.weight and blk.1.ffn_down.weight.
Their canonical streams are 11141120 bytes each; head streams are 198656000
bytes and context FC streams are 36864000 bytes. Research relative L2 errors
are about 0.00036 for lattice, 0.00072 for head and 0.00030 for FC. These
validate only operator reconstruction/accumulation, not model distribution.
A focused M8 head comparison against official FP32 dequantization and FP32
cuBLAS gives relative L2 0.0007174 for canonical and 0.0161201 for raw, with
maximum absolute errors 0.004103 and 0.064516. These sampled activations do not
replace the model-level KL gate.

The first research kernels lose to the existing canonical paths and are not
promoted. Their intrablock split increases warps, not CTA count; FC has only
40 CTAs. Nsight Compute reached the kernel but reported insufficient GPU counter
permissions. Static cubin resources show no local-memory spills; runtime
occupancy, stalls and physical DRAM traffic are not measured. Canonical head saves about 347 us per M8 call
against the current raw GGUF head. M1 differs: canonical 340.992 us versus raw
243.712 us. Routing must therefore be measured by M range. Footprint/elapsed
bandwidth is useful effective throughput, not measured DRAM traffic. Calls
per complete round and model-level savings remain unverified.

## Graph-linked exact-input baseline

The profiled run uses exactly 1024 input tokens, 64 output tokens, max length
32768 and the same TP4/FP16 KV/FP32 SSM configuration above. Fifteen observed
verification output intervals average 48.8762 ms. GPU target graph submissions
are asynchronous: after the first rounds, CPU submission precedes execution
by one round. CPU NVTX start intervals therefore mix adjacent GPU rounds.
Attribution uses the worker's `cudaGraphLaunch` correlation ID and successive
GPU graph starts, discarding boundary rounds.

All four ranks agree: complete-round GPU intervals average 48.965–48.969 ms,
and the target graph envelope averages 44.390–44.397 ms. Each target graph has
1430 kernel nodes. These are verification rounds, not individual output-token
latencies. A matched unprofiled exact-input run is still required before
calibrating profiler overhead; multiplying by 0.9 remains an initial estimate.

| Rank-0 kernel group | Calls/round | Service ms/round |
| --- | ---: | ---: |
| Small CUTLASS FP16 matrix products, one CTA | 96 | 10.208 |
| IQ3_S dequantization, grid 10880 | 46 | 4.262 |
| FP16 wide XQA attention | 16 | 3.767 |
| CUTLASS matrix products following dequantization | 29 | 2.645 |
| cuBLAS Volta split-K matrix products | 22 | 1.797 |
| TP push reductions, grid 40 | 140 | 1.666 |
| Raw Q4_K vocabulary projection | 2 | 1.195 |

Service times can overlap and are not an additive wall-time decomposition.
For each IQ3_S dequantization of a 4352×5120 matrix, known buffers comprise
11,141,120 packed bytes and 44,564,480 output bytes. At 750 GB/s this is a
74.274 us footprint floor, excluding activation and later GEMM traffic. The
46 calls total 3.417 ms at that floor versus 4.262 ms observed. Other kernel
traffic remains unspecified until its tensor descriptor is established.

Generated AOT code fixes the Python prefill branches during initial tracing,
then reuses them for M=8 with shape guards dropped. This causes 46 IQ3_S and
five affine projections to dequantize full matrices during verification.
Aliased shared scratch also generates large clone/copyback kernels. Runtime
projection dispatch keeps the family and capability bands but checks M inside
an opaque operator; shared scratch is resolved internally from its prepared
workspace. CPU tests verify branch selection and a single dynamic graph across
M=512/8/16. GPU route, accuracy and timing checks remain pending; no speedup is
claimed and the installed model wheel remains unchanged.
