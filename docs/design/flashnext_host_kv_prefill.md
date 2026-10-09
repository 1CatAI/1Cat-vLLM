# Flash-Next host KV prefill on SM70

## Initial TP4 diagnostic

The installed `4967ee9d18` wheel completes a 32768-token request with a
16384-token scheduler chunk, IQ3_S target, FP16 MTP4, FP16 host QSA history
and a 1 GiB KV pool. Hardware is four V100-SXM2-32GB GPUs with NV2 links
between every pair, an 185 W power limit, CUDA 12.8 and Torch 2.10.0+cu128.
The PLE host tier is 1 GiB per rank with mapped-checkpoint spill; graph mode
is FULL with piecewise compilation for larger batches.

Two unprofiled scheduled-to-first-token measurements are 16.9007 and
19.3983 seconds (1938.9 and 1689.2 input tokens/s). Both return first token
147113. Three natural prompts finish at EOS with correct arithmetic,
a reasonable prefill/decode explanation and a valid Python example.
This diagnostic has no C4 qualification: reducing the KV pool obtains a
trace but does not resolve the admission issue at the required concurrency.

The four-rank Torch traces have observed device envelopes of 23.032--23.035
seconds. Rank 0's recorded kernel service includes QSA 5.588 s, MoE 4.312 s,
HC 1.760 s and GDN 1.263 s. Projection kernels nested in opaque GDN/MoE
operators are included in those categories; these are not isolated arithmetic
costs. Recorded device-event coverage leaves a 9.031 s residual. Its largest
7.515 s interval overlaps the QSA indexer's `aten::item` stream synchronization
and a long compiled-graph GPU annotation. Kernel records alone therefore do
not establish that the GPU is idle throughout this interval. A CUDA graph-node
Nsight trace is required before counting this residual as recoverable time.
Several other residuals overlap large `aten::empty` allocations.

## CUDA timeline and memory admission

A subsequent same-process norm-only ABBA on the installed `250a317024`
wheel measures 23.743/22.343 s for the control and 26.428/18.173 s for the
candidate. Variation exceeds the mean difference; these rows do not establish
a norm throughput improvement. All measured first tokens match. Arithmetic
and Python natural completions match and reach EOS; the explanatory completion
differs, so full-output parity and C4 remain unqualified. Both arms have the
same 30.207 GiB whole-model Torch allocation peak despite the isolated norm's
smaller temporary allocation.

CUDA-only Nsight capture observes the profiled candidate's 18.781 s request.
Rank 0's device envelope is 18.776 s, with 14.133 s covered by kernels/copies
and 4.644 s between recorded activities. It contains 26,892 kernels, including
384 recorded graph nodes. Initial four-rank entry skew is 0.744 ms.

| Rank 0 activity family | Recorded service, s |
| --- | ---: |
| QSA named attention/history | 5.140 |
| Grouped expert GEMM | 3.195 |
| cuBLAS and TurboMind projections, role unresolved | 2.389 |
| TP communication | 0.902 |
| Other kernels | 1.147 |
| HC named glue | 0.554 |
| MoE routing/generic | 0.433 |
| Copies/fills | 0.314 |
| Remaining named GDN/norm/PLE | 0.060 |

Name families are not layer attribution or independent critical-path savings.
The other-kernel bucket includes 0.282 s of fused GDN recurrence, 0.207 s of
prefill inverse routing, 0.197 s of host-history staging, 0.133 s of KV block
zeroing and 0.113 s of host-history writing. These useful operations are not
automatically removable overhead.

Intersecting each rank's CUDA allocator API intervals with that rank's device
gaps accounts for 4.377 of rank 0's 4.644 s (94.25%). Ranks 1--3 account for
4.808/5.063, 4.545/4.787 and 4.768/5.016 s respectively (94.94--95.06%).
Long gaps repeatedly follow MoE unroute/add/allreduce and precede HC
combine/norm. The same worker unmaps and releases physical CUDA memory, then
creates, maps and sets access for new memory. This is allocation work on the
launch dependency, not unidentified GPU arithmetic or seconds of entry skew.
Each rank records 16 failed `cuMemCreate` calls returning CUDA error 2
(`CUDA_ERROR_OUT_OF_MEMORY`), followed by cache release and successful new
allocations. The request completes after these allocation retries. No allocator
call stacks were recorded to identify every requesting tensor. The benchmark
records retry/OOM/synchronization counters before and after each request, plus
initialized and post-profile segment summaries with graph pool IDs and largest
inactive blocks. These distinguish capacity pressure from memory retained in a
different stream or private graph pool before choosing a storage change.

The initialized global modular MoE workspace is 880 MiB. With the real MTP TP4
expert geometry (512 experts, top10, K2560, local gate/up width 320) and 16K
rows, the installed allocation code requires 880 MiB for a full-row operation
or 292.5 MiB for 4K-row chunks, a 587.5 MiB reduction. This preserves the full
16K scheduler batch and output; only independent expert rows are chunked. The
existing modular activation chunking controls are default-off. Before selecting
this mode, compare full and chunked FFNs on the real FP16 MTP shard, including
M5/M20 changed-input graph replay. Separate-process model controls are required
because the initialized workspace is locked for graph use. Reducing its shape
in place would not reclaim the original graph-owned allocation. The benchmark
records these registered settings and supports a fixed control alongside the
existing fixed candidate, so storage and grouped-attention changes can be
measured separately in the same wheel. No numerical or model gain is claimed
from the capacity calculation alone.

The immediate throughput milestone is 4000 input tokens/s, or 8.192 s for
32768 tokens with the same 16K scheduler chunk; 6000 remains a later goal.
Even removing all observed gaps and all QSA named attention plus staging would
leave about 8.80 s in this profiled request. Thus allocator relief and grouped
attention are necessary candidates but cannot by themselves establish the
milestone. Grouped expert scheduling and projection attribution must be
revisited after their measured model A/B. The 3.195 s expert service is an
upper bound on that direction, not a promised speedup. No implementation is
promoted from an operator ratio or this subtraction.

## Native grouped staged attention

The staged reader's 256-row causal Triton launches use scalar FP32 math on
SM70. `qsa_host_kv_prefill_grouped` admits the existing Flash-V100 grouped
page4 planner and tensor-core attention after the same history staging. The
planner consumes only canonical four-token groups followed by the compact
causal tail generated by the QSA indexer. Generic arbitrary causal indices
retain the generic sparse attention route. Admission checks SM70, FP16 staged
K/V, G6D256 layout, 2051 selection columns and the native grouped ABI; rejected
layouts log their reason and retain the 256-row fallback. The final fewer
than eight queries also retain the causal fallback. Output gating is applied
after attention. MTP steps that reuse an earlier sparse-index row retain the
causal fallback: their compact tail may describe the preceding position.
A stream-scoped planner workspace is reused across layers;
no second persistent history bank is added.
At 16384 query rows this workspace adds 65.383 MiB per CUDA stream if no
existing capacity is available. Startup reports this estimate separately
from the borrowed history allocation; it is not zero additional workspace.

The benchmark supports an in-process grouped-attention ABBA and an optional
CUDA profiler capture for Nsight graph-node traces. Integration correctness
passes all 32 host-prefill tests under the installed `3c0791f40` wheel.
These cover native route selection, FP16/E4M3 history, early causal tails,
invalid requests/pages and changed-input graph replay. Full-model speed
remains pending; existing synthetic ABI microbenchmarks are not used as
end-to-end speed evidence.

## Large-batch gated norm admission

A TP4 IQ3_S run with FP16 MTP4, 32K input, a 16K scheduler chunk and a
1.375 GiB KV pool failed on the first real prefill. Initialization used
29.13 GiB of active Torch storage per rank. GDN's gated norm then attempted
another 96 MiB FP32 allocation with only 62.75 MiB device memory free.
No throughput is recorded for this failed request.

The opaque GDN forward selects `RMSNormGated.forward_native`. Its exact SM70
decode kernel admits at most 192 N128 rows; prefill falls back to separate
FP32 casts, square/mean, normalization, activation and multiplication tensors.
The new `prefill_rmsnorm_gated` capability reuses the existing FLA one-pass
implementation for contiguous FP16 input, gate and N128 weights with at least
4096 rows, normalization before gating and no group normalization. It retains
FP32 normalization/gating and FP16 output. Small decode dispatch is unchanged.

This route can change FP32 reduction/activation rounding and is checked against
the native FP32 reference; it does not claim bitwise equality. An installed
operator ABBA test on V100-SXM2-32GB uses 196608 N128 rows, FP16 operands,
sigmoid gating, Torch 2.10.0+cu128 and CUDA 12.8. CUDA-event medians are
2.159/2.159 ms for the native decomposition and 0.333/0.340 ms for FLA.
Temporary peaks are 576.75 and 48.75 MiB. Relative L2 difference is
4.613e-6, maximum absolute difference 0.001953125, and 99.9951% of output
values are bitwise equal. These are synthetic operator measurements, not
model throughput. The noisy full-model comparison above is inconclusive. The benchmark
supports an in-process norm-only ABBA comparison and records profiled request
timing separately from its unprofiled rows.

The corrected standalone GPU suite passes all five new norm checks: FP32
reference comparisons, peak allocation and changed-input compiled graph replay.
An existing small-M compiled exact-norm check differs at one of 3072 FP16
elements, index `(13,30)`, with maximum absolute error `7.62939453125e-6`.
Both the unchanged `4967ee9d18` wheel and `250a317024` candidate reproduce
the identical discrepancy. Its exact tolerance is unchanged.

## Persistent weight layouts and CPU lookup review

The baseline's initialized Python-reachable device inventory accounts for
28.501 GiB per rank; native-only allocations are outside this inventory.
Across 48 target expert layers, canonical gate/up codes occupy 4.785 GiB,
their metadata 4.614 GiB, and the down banks 4.944 GiB. Another 6.752 GiB
holds original packed gate/up weights across 47 layers. These original blocks
serve small-M raw/dp4a kernels while canonical banks serve large prefill.
This is avoidable simultaneous storage for two execution routes. The original
expert storage implementation has already measured 11.808 GiB per rank for
the same target expert projections, including TP boundary blocks. The current
21.095 GiB therefore adds 9.286 GiB per rank. Keep these representations in
the weight-storage ledger rather than treating them as required model size.
Removing the duplication still requires matched decode/concurrency checks.

An isolated CPU diagnostic uses the exact tokenizer and repeated technical-text
input hash from the 32K baseline. Its two 16K chunks contain 568 and 544 unique
ngram IDs. On warm reads, the existing ngram implementation takes 5.7--6.5 ms
and full-table GGUF lookup 30.4--33.3 ms per chunk; each materialized FP16
result is 80 MiB. The first read takes 166 ms with 609 major faults. These
measurements exclude resident-tier masking, IPC, copy/publication and model
contention. They cannot attribute the multi-second trace residual to CPU lookup.

Nsight Systems 2024.6.2 with CUDA and NVTX tracing fails during NCCL 2.27.5
extended-NVTX enum registration before model computation. Disabling NCCL NVTX
passes a single-process smoke but does not resolve multiprocess model startup.
CUDA-only tracing passes a four-process PyNccl initialization/allreduce check;
the model diagnostic uses that mode. Torch-specific profiler options are
provided only to the Torch profiler. Failed profiler starts supply no model
speed evidence.

The host QSA reader batches 32 queries in a protected hot-page cache. A 16,384
query prefill therefore executes 512 protection, resolution and attention
groups per owner. Decode-sized groups repeatedly resolve overlapping history,
and a long-context miss reads mapped host bytes again for each query.

For sufficiently large prefill, stage each logical request page once in the
existing shared FP16 miss workspace. The page mapping retains aliases, invalid
pages, sequence bounds and per-query causal limits. E4M3 uses the same software
decode and per-vector FP32 scales as the host reader; FP16 copies values without
rounding. Authoritative history remains in host memory.

Attention processes up to 256 rows together. Full 32-row groups retain the
eight-split profile and accumulation sequence of the host reader. A final
short group stays separate, retaining its original split/warp profile. Selected
counts skip the same compact empty suffix. Decoder and attention arithmetic
are shared with the existing QSA implementation.

## Admission and memory

The optional host-cache policy must already admit SM70, FP16 activations,
D256, one local KV head and DCP1/PCP1. Kernel policy
`qsa_host_kv_prefill` enables the prefill reader within that host-cache policy.
Runtime admission requires at least 512 queries, a 32-row reader workspace,
and logical request pages that fit the fixed FP16 miss allocation. Unadmitted
inputs retain the existing reader; logs report the rejection reason.

The default 32-row, 2051-column miss workspace holds 65,664 FP16 K/V tokens.
A single 32K request with 816-token pages uses 32.672 MiB of that allocation.
No additional persistent history bank is allocated. Remapped page IDs and
selected counts are temporary metadata. Attention partials remain bounded by
256 rows, rather than a full 16K eight-split allocation.

Small-M decode does not select this reader. A four-request 32K history does
not fit the existing scratch and therefore falls back. This change does not
qualify the parent host-cache feature or its placement/precision policy for
default promotion.

## Single-layer results

The source experiment uses one V100-SXM2-32GB from the TP4 host, Torch
2.10.0+cu128 and CUDA 12.8. Histories are FP16; page size is 816, hot capacity
8192, query heads 6, head dimension 256 and selected width 2051. Synthetic
causal block selections are spread with row-dependent shifts. Timings are
same-process eager ABBA, including host dispatch, with two iterations per arm.
These are single-layer results, not model prefill throughput.

| History tokens | Query rows | 32-row reader, ms | Staged reader, ms | Ratio |
| --- | --- | --- | --- | --- |
| 16384 | 512 | 119.04 | 12.64 | 9.4× |
| 16384 | 16384 | 1329.56 | 185.95 | 7.2× |
| 32768 | 512 | 201.36 | 18.86 | 10.7× |
| 32768 | 16384 | 4919.50 | 196.68 | 25.0× |

All four points are bitwise equal, with zero maximum absolute and relative
L2 difference. Group counts fall from 512 to 64 at 16K queries; these are
launch groups rather than traced CUDA kernel counts. A synthetic selection
can have different locality and cache pressure from the trained indexer,
so the ratios cannot be multiplied into a model-level speed claim.

Fifteen source tests pass. They cover official FP16/E4M3 value reconstruction,
physical-page aliases, noncontiguous table row strides, invalid pages, causal
compact tails, short final query groups, rewrites and captured replay. Offline
SM70 compilation succeeds for FP16 and E4M3 staging.

## Negative E4M3 observation

The initial E4M3 ABBA point rejects comparison with the first cold-cache host
output. Isolated repeated cold-cache resolution shows occasional incorrect
fallback values despite correct hot payloads and page mappings. A full
synchronization before resolution does not remove the observation. Writing
all valid mixed-tile values instead of only misses also fails; that prototype
is rejected and the original host reader is retained.

The staged E4M3 history matches official software decode exactly on valid
sequence positions, and its tested output matches resident decoded FP16
attention. This is not a model-level E4M3 qualification or an explanation of
the parent's acceptance difference. No E4M3 speed result is claimed from the
rejected cold-cache comparison.

## Model validation pending

The prefill contract is Flash-Next IQ3_S, FP16 MTP4, TP4 on four V100 32GB,
16K scheduler chunks and 32K input without prefix-cache reuse. Report scheduled
to first token, total request wall and pure decode separately. The immediate
4000 input tokens/s milestone corresponds to 8.192 seconds; 6000 input tokens/s
corresponds to 5.461 seconds for 32768 tokens.

Obtain a matched host-cache baseline, then an in-process prefill-only A/B,
followed by target token/logit checks, natural completion and C1/C4 acceptance
checks. GPU timelines must identify transfer, attention, expert/dense compute,
host launch gaps and the critical TP rank. No 32K model speed claim or default
promotion is established by the single-layer measurements above.

## Full-model admission and storage

The first 32K / 16K-chunk FP16 host-KV run did not reach attention: rank 0
failed allocating the 80 MiB first embedding all-reduce output. Torch had
29.94 GiB allocated, with only 7.5 MiB device free; a CPU PLE helper also owned
a 388 MiB CUDA context. This run is an admission failure, not throughput
or correctness evidence for the staged attention reader.

A prior storage audit identified 14.343 GiB/rank of canonical expert banks
plus 6.752 GiB/rank of original gate/up blocks. The constrained-memory branch
already supplies a single original-block storage policy. The prefill branch
had ported initialization fixes but omitted that policy and the sharded HC
storage policy. This omission explains the much larger expert residency;
it is not an intrinsic memory requirement of the GGUF checkpoint.

The memory follow-up ports context-free attention and GDN capability queries,
CPU/meta GDN norm placement, and TP-only EP communicator admission from the
compact-storage work. SM70 all-gather reuses the initialized PyNccl
communicator, avoiding another lazy NCCL allocation. Text-only fixed-frequency
RoPE uses the engine's position bound; its values inside the admitted range
remain identical. MTP construction leaves placeholders for target-shared IO
rather than allocating temporary duplicate embedding/head parameters.

An explicit `sm70_gguf.embedding_storage="original"` policy preserves packed
token embeddings and dequantizes selected rows per lookup. It retains dense
storage as the default and does not change LM-head dispatch. These changes
must be measured in the same full-model contract before quoting memory or
throughput savings. The benchmark now snapshots its serializable configuration
before engine construction so a mutated runtime config cannot hide failures.

A subsequent packed-embedding launch failed during loading because the adapter
emitted `embed_tokens.qweight_type` while the backbone still constructed a
plain embedding. The backbone constructor now passes the selected quantization
method, and construction-level tests cover both policies. This failure has no
throughput result.

CPU-helper tracing also identified two additional initialization paths: shared
expert auxiliary streams during meta discovery, and compilation of the meta
backbone, which imports GPU providers through fusion passes. The offload helper
now suppresses its unused MoE stream and constructs the discovery-only backbone
with compilation/graphs disabled on a separate config copy. GPU worker policy
and CPU-owned PLE execution configuration remain unchanged.

The isolated helper completed real PLE weight discovery/loading without Torch
CUDA initialization or a NVML allocation. Startup cancellation now also checks
the parent shutdown event while waiting for GPU registration; a loader failure
must not leave the helper blocked indefinitely with retained resources.

## HC peak during a real 16K chunk

The memory fixes completed loading and graph initialization, with 29.45 GiB/rank
of active Torch allocations and about 0.98 GiB device free. The first real 16K
chunk then reached HC gate-mix and failed allocating its 80 MiB output. Torch
allocations had risen to 30.45 GiB; attention still had not executed.

For 16K rows, the four-stream hidden, normalized hidden and full gate are each
320 MiB. `prefill_hc_chunk_size=4096` bounds projection intermediates, and bounds
combine/norm intermediates inside the existing opaque HCX fallback. It retains
full outputs, FP16 materialization and FP32 accumulation. M1/M5/M20 dispatch is
unchanged; zero disables blocking. The scheduling chunk remains 16K.

A same-card real-weight microbenchmark measured:

| 16K rows | Full temporary peak | Blocked temporary peak | Full GPU time | Blocked GPU time |
| --- | ---: | ---: | ---: | ---: |
| Projection | 420.50 MiB | 185.25 MiB | 4.56 ms | 5.45 ms |
| Combine + projection | 1060.50 MiB | 665.25 MiB | 6.32 ms | 8.09 ms |

All returned tensors matched bitwise in this microbenchmark. Blocking is a
capacity fix with a measured operator slowdown; no prefill-throughput gain is
claimed. Model timing and the 32K throughput target remain pending. Both
attention policies are now warmed before the timed ABBA sequence to exclude
first-use compilation from the comparison.

## Prefill compiler boundary

The installed row-blocking candidate still failed at the first HC. The dynamic
prefill compiler intentionally bypasses decode operators, so the first HC
retained two external GEMMs and a full gate-mix allocation. A helper-only
microbenchmark did not exercise that entry point. A separate opaque prefill
projection now resolves actual row counts at runtime; it retains the existing
FP16 dense projection below the row limit. Decode compiler dispatch is unchanged.
The first-HC export and dynamic compiled replay regression cover the production
entry point, including a short row count after a blocked invocation.

The initialized worker inventory identifies 27.6415 GiB of unique reachable
device storage per rank against 29.4512 GiB of live Torch allocation. The
1.8097 GiB difference is unclassified allocation, not proof of a leak. Mapped
host storage is counted separately; the inventory reports allocator block sizes
for allocations not reachable through the inspected Python owners. No complete
32K request or model throughput result is available from the failed attempt.

## PLE initialization residency

After resident GPU row placements are registered, the CPU helper can own no
disk rows. Prefaulting its entire mapped table then duplicates unnecessary RAM
residency and delays startup. The placement-aware prefault check from
`74cfbd380de0dd3695a4507183ab18d7912ba85d` skips tables only after all rows are
served by registered ranks; unregistered placements and actual disk tiers keep
the existing prefault contract. Five targeted CPU checks pass. This is a startup
and host-memory change, not measured prefill throughput.

## Generic MoE workspace over-allocation

The corrected prefill HC entry passed the first projection, then the model
failed on a 96 MiB GDN norm allocation at 30.48 GiB of live Torch memory. The
initialized allocator snapshot identifies a single 1600 MiB persistent block,
matching the generic Triton expert workspace for M=16384, top-k=10, K=2560 and
N=320: two 800 MiB buffers.

The first buffer holds the post-activation expert rows. It need not reserve
`top-k * K` reduced-output elements: the modular allocator already takes the
maximum of activation storage and the separate `[M,K]` output shape. Declaring
activation storage as `[M,top-k,activation_out_dim]` reduces the shared first
buffer to 80 MiB (50 MiB activation, 80 MiB output). The second buffer remains
800 MiB, making the total 880 MiB rather than 1600 MiB. This predicts 720 MiB
less persistent workspace per rank without changing kernels or arithmetic.

Twenty-two targeted checks pass, including real expert geometry, M1/M5 native
projection graph checks, changed routes and inputs, and bitwise agreement of
M33 graph replay with reduced output sharing activation storage. Capacity tests
cover output sharing, chunked allocation and gated/non-gated activations. The
worker inventory now includes the global workspace manager. The next installed-wheel run measured 28.7476 GiB/rank of initialized live
Torch allocation, down from 29.4512 GiB/rank: 720.627 MiB saved. The global
workspace inventory reports exactly 880 MiB. The unclassified remainder is
253.232 MiB, including normal native and cuBLAS workspaces. This run passed HC
and GDN, then failed allocating a 320 MiB PLE convolution output; no complete
32K request or throughput result was produced.

## PLE prefill convolution peak

The previous dilated short-convolution packs a full channels-first history,
materializes a full convolution result, and gathers rows into token order. At
16K tokens and 10240 channels, each FP16 token buffer occupies 320 MiB. A direct
prefill kernel reads token-order inputs and a small initial-state snapshot,
computes the dilated depthwise convolution and SiLU, then commits base history
in a separate kernel. It retains FP32 convolution accumulation and the FP16
rounding boundary before FP32 SiLU. Short decode and speculative rollback
paths remain unchanged.

Admission is SM70, FP16 input/weight, at least 512 query rows, 1–16 requests,
2–8 taps, dilation 1–8, and compatible FP16/FP32 state storage.
`prefill_ple_short_conv=false` selects the existing implementation. The startup
capability report describes these guards and runtime fallback reports its
reason.

On the same V100-32GB (CUDA 12.8, Torch 2.10), an eager ABBA test using
the checkpoint's actual layer-1 PLE convolution weight measured:

| Query rows | Existing temporary peak | Direct temporary peak | Existing median | Direct median |
| --- | ---: | ---: | ---: | ---: |
| 512 | 20.582 MiB | 10.178 MiB | 0.850 ms | 0.327 ms |
| 16384 | 640.945 MiB | 320.178 MiB | 15.715 ms | 1.616 ms |

Both results and state updates matched the existing path bitwise in this
real-weight test. Both also matched an independent FP32 convolution reference
rounded to FP16 before SiLU. Tests additionally cover ragged and empty requests,
NULL state rows, initial history, FP32 state storage, channel tails, strided
inputs, history carried between chunks, and graph replay with changed inputs
and initial-state flags. These are operator measurements, not full-model
prefill throughput or quality results. Full 32K model evidence remains pending.

## Prefill request publication and allocator synchronization

The next installed run initialized at 28.748 GiB/rank but stalled before the
new convolution executed. A native stack shows allocator cache reclamation
(`ExpandableSegment::unmapHandles` → `cudaStreamSynchronize`) while the model
thread retains the GIL. The PLE notification thread has finished waiting for
the input D2H event but is blocked in `PyEval_RestoreThread` before publishing
the request. The GPU consumer waits for PLE output, and the helper polls
without receiving work. This is a synchronization cycle under memory pressure,
not measured slow prefill. The executor eventually times out; retained workers
were removed after verifying their benchmark process identities.

For CPU-owned PLE, the model thread now waits until the
notifier has published the request and returned its D2H event. This wait releases
the GIL and precedes the GPU consumer. It does not wait for the PLE result.
`ple_request_publish_before_wait=false` retains asynchronous submission. Local
pinned decode bypasses the CPU helper and this barrier. Six targeted CPU checks
cover per-batch D2H ownership, CUDA/CPU input publication order and the
asynchronous control. Complete-model verification remains pending.

The opaque convolution also receives its already allocated output directly
for a pure prefill batch. Mixed decode/speculative batches retain their existing
merge and copy path; padded output rows remain zero. The convolution writes
using output strides, avoiding a second full-sized result allocation and copy.
Twenty-one targeted GPU/CPU convolution checks pass, including the opaque
entry, padding, state carry and graph replay.

With the checkpoint's actual weight at M=16384, an eager ABBA comparison of
opaque output handling measured 320.178 to 0.393 MiB of extra temporary
allocation, excluding the fixed input/output buffers. Median operator time
falls from 2.495 to 1.657 ms; outputs and state updates match bitwise. This is
an operator result. The full-model prefill and decode gates remain pending.

## Execution policy and the next memory peak

The installed publication fix reached the direct PLE convolution in a real
32K/16K run, resolving the notification/allocator synchronization cycle.
The request subsequently failed allocating an 80 MiB HC projection result;
initialized allocation remained 28.748 GiB/rank. No 32K throughput was recorded.

A real-weight HC experiment compared 4096 and 2048 row blocks. Temporary peak
fell from 665.250 to 532.688 MiB; median time increased from 7.676 to 9.210 ms.
The combined hidden output matched bitwise. Block output relative L2 difference
was 5.447e-5, maximum absolute difference 0.0078125. Against independent FP32
projections, relative L2 was 6.734e-5 for 4096 and 5.683e-5 for 2048. Both retain
FP16 projection boundaries and FP32 accumulation. Two of 41,943,040 block
values exceeded the initial 2e-3 elementwise tolerance; aggregate and independent
reference errors are reported rather than treating this as bitwise equivalence.
These are capacity/operator measurements, not model correctness evidence.

The full-model retry configured 2048 rows and reported it at startup, but still
allocated 80 MiB: the opaque operator read the construction-only configuration,
which is absent during forward execution, and silently used 4096. This retry
is not an effective 2048 model comparison. ForwardContext now carries the
execution's KernelConfig. HC row blocking, PLE prefill convolution admission,
and GGUF expert prefill row blocking read that policy before any construction
fallback. Runtime route logs include the actual row-block size. The generic
construction accessor and unrelated precision policies remain unchanged.
Regression tests exercise an expired construction scope, conflicting scopes,
and a compiled opaque HC call with the execution policy and no construction
configuration. Another installed-wheel model run is required to qualify capacity,
throughput, output quality and decode behavior.

The next installed wheel passed 39 targeted checks. The model logs confirmed
HC 2048-row and expert 4096-row blocking with construction scope absent.
Initialized allocation was 28.7484 GiB/rank. The first 16K query chunk progressed
through the decoder layers and QSA attention, then failed in the final HC mix:
its no-injection variant was excluded from the combine/projection row loop and
requested a full 320 MiB gate result with only 221–247 MiB device memory free.
This is another bounded-work coverage gap, not a completed 32K throughput run.
The existing HC row loop now also handles the final no-injection variant,
retaining its `None` injection result and the same FP16 projection boundaries.

A real final-HC weight source-only comparison at M=16384 reduces temporary
peak from 1050 to 531.25 MiB; median time changes from 6.004 to 7.616 ms.
The hidden output matches bitwise, and block relative L2 difference is
5.148e-5 with maximum absolute difference 0.010742. Independent FP32 projection
relative L2 is 6.348e-5 for the unblocked path and 5.716e-5 for 2048 rows.
Thirteen of 41,943,040 block elements exceed 2e-3 absolute-plus-relative
comparison tolerance. This is an operator capacity result, not a model gate.
The next installed wheel passes 40 targeted checks.

That model attempt exposes another peak in the compiled HC mix: after the
PLE hidden-state additions, norm materializes all 320 MiB before the already
blocked projection. Its 40 MiB gate allocation fails with 21.5 MiB free on
rank0. Add an opaque prefill norm/projection boundary and normalize each row
block immediately before projecting it. The common norm/projection operators
and their FP16 boundaries are unchanged. Decode compilation retains its
existing route; unsupported layouts and methods retain their own forward path.

With real HC weights at M=16384 and 2048 projection rows, the source-only
ABBA comparison reduces extra temporary peak from 452.688 to 172.688 MiB.
Outputs are bitwise identical; median time is 7.971 versus 8.029 ms. Fifteen
HC checks pass, including export without a standalone full-row norm allocation,
compiled runtime policy after construction scope ends, short tails and the
no-injection final mix. Complete 32K model and decode gates remain pending.

The installed norm/projection wheel passes 42 targeted checks. A real model
run completes 32K/16K prefill for both the old and staged FP16 host readers;
both warmups emit token 147113. Live allocation returns to 28.7484 GiB/rank,
with peak live allocation 30.390 GiB/rank. This establishes capacity for this
single-request workload; it does not establish natural-answer quality or C4.

The subsequent ABBA measurement completes its first request but the runner
fails when reading timestamps: LLM statistics default off, so RequestOutput
metrics is None. No valid scheduled-to-first-token throughput or decode result
was saved. Enable statistics explicitly and save wall time/output/memory before
requiring timestamps so a reporting failure retains useful evidence. Re-run
the matched benchmark with the same installed production wheel; this runner
change does not alter production modules or native kernels.

An optional post-measurement CPU/CUDA trace uses the existing worker callable
RPC. After all unprofiled requests, release unused allocator cache to admit
CUPTI, warm one staged request, then record the next request on all four ranks.
Export separate Chrome traces and label their timing as profiled. Production
modules, weights and dispatch remain unchanged. A normal installed-wheel GPU
smoke confirms that the schedule exports actual CUDA kernel events; a CPU-only
schedule smoke checks warmup/record/export ordering. Full-model trace collection
still requires the next benchmark run.

## First complete 32K model comparison

The matched rerun uses the same normally installed production wheel, four
V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, TP4 and FP16 MTP4. Target and
draft KV are FP16, recurrent state is FP32, the GPU KV/state budget is 1 GiB,
HC blocks are 2048 rows, expert blocks are 4096 rows, and the scheduler chunk
is 16384. Prefix caching is disabled. Both reader policies are warmed before
the unprofiled ABBA requests. Input is 32768 fixed tokens of repeated technical
text; output is one greedy token. This is a throughput workload, not a natural
completion or quality corpus. Observed prefill SM clocks reach 1530 MHz.

| Host reader | Scheduled-to-first-token samples, s | Mean, s | Input tokens/s |
| --- | --- | ---: | ---: |
| Existing 32-row reader | 96.9423, 96.5130 | 96.7277 | 338.8 |
| Staged prefill reader | 13.2925, 13.2781 | 13.2853 | 2466.5 |

The measured ratio is 7.28x. All warmup and timed first-token IDs are 147113.
Live Torch allocation after each request returns to 28.7483 GiB/rank; peak
live allocation is 30.3892 GiB/rank. Near-full allocator reservations retain
unused cached blocks and must be reported separately from live tensor storage.
These repeated requests do not show growing live allocations.

The 6000 tokens/s goal remains unmet: the candidate still needs about 2.43x
throughput improvement. C1/C4 checks and post-measurement GPU tracing follow
this comparison. Those results, natural-output checks and the critical-rank
latency account are required before promotion.

The first C1 check reports 18.471 ms/round on the existing reader at 8K input.
The C4 check then finds no eligible four-request intervals: scheduler logs show
three running requests, one waiting, and 96% GPU KV/state-pool usage. Its
failure terminates that attempt before the candidate decode checks and trace.
This is not a C4 pass or regression measurement. Retain raw cohort records,
output IDs and the admission failure, and allow subsequent diagnostic capture
to proceed. A new matched run increases the GPU KV/state budget by 384 MiB
to 1.375 GiB; its capacity and performance must be measured separately.

An offline audit of seven cached SM70 QSA partial-kernel PTX variants finds
no `mma.sync` or `wmma` instructions; they contain 513 or 1026 static FP32 FMA
instructions. Source `tl.dot` therefore does not establish Tensor Core use
for this route. This identifies a candidate for native prefill attention, not
its share of model latency. The existing source rounds probabilities to FP16
before P.V and accumulates in FP32; preserve those boundaries when evaluating
a native implementation. Attribute the actual route in the model trace before
choosing its implementation or quoting an expected end-to-end gain.

## Compact PLE prefill gate

Increasing the device KV/state pool from 1 GiB to 1.375 GiB fails on a
320 MiB expanded PLE gated-value allocation during the first 16K chunk. The
compiled prefill route now retains FP32 scalar gates, reuses the uniquely owned
projected key for normalized convolution input, and reuses the convolution
output for the residual sum. Eager execution, decode compilation, and diagnostic
snapshots retain their existing path. The capability is enabled by default and
controlled through `KernelConfig.prefill_ple_compact_gate`.

The numerical reference follows actual compiled prefill boundaries: gate and
variance calculations remain FP32, gated values are materialized in FP16 before
convolution normalization, and the residual epilogue accumulates in FP32 before
FP16 storage. A prototype that inserted additional FP16 gate/norm boundaries was
rejected. On 16384 rows with official GGUF norm weights, relative L2 error is
8.79e-6 for normalized convolution input and 4.43e-6 for the final PLE output;
maximum absolute errors are 0.00390625 and 0.00048828125. All values are finite.

A source-only ABBA microbenchmark of the gate/normalization and output phases,
excluding projections and convolution, measures median 3.374 ms versus 2.843 ms.
Standalone temporary allocation changes from 960.25 MiB to 0.25 MiB. The existing
compiled model already reuses some buffers, so this is not a model-level memory
or latency claim. Installed-wheel peak allocation, four-request admission, and
full-model throughput remain to be measured.

The compiled-entry regression explicitly selects the prefill compiler phase and
verifies both custom operators occur in the exported graph. GPU checks cover
strided inputs, FP16/FP32 norm weights, optional residuals, and replay with changed
inputs. Disk/offload fixtures now initialize the required layer name and config;
invalid disk rows must fail in both the native and Python reader.

## Profiling transport

The first full-model profiling attempt failed before recording because callable
RPC transport requires unsafe serialization. The benchmark now uses the built-in
Torch profiler through named worker RPCs, preserving default serialization
policy. Two profiling cycles warm CUPTI before the recorded request; compressed
worker traces are excluded from unprofiled throughput. A normal-wheel smoke
verifies named-RPC serialization, restart, and CUDA events in both trace files.
Three natural chat prompts complete with EOS, including arithmetic, a Chinese
explanation, and Python code. These health checks do not substitute for the
pending model-level logit and C4 acceptance comparisons.

The first installed compact-gate model attempt fails before request timing:
`logger.info_once` inside PLE forward is unsupported by production fullgraph
compilation. The earlier numerical test permitted graph breaks. Logging now
runs inside the opaque operator boundary, and static fallback reporting is
excluded from graph tracing. Two strict fullgraph/export regressions cover both
HC4 admission and HC3 fallback, including residual and eager output comparison.
No model memory or speed claim is taken from that failed initialization.

## Original-block memory policy follow-up

The source comparison with the constrained-memory GGUF implementation identifies
omitted weight-storage policies, rather than an unusually large required expert
footprint. The initialized target expert ledger is:

| Representation | GiB per rank |
| --- | ---: |
| Current canonical expert codes and metadata | 14.343 |
| Current additional original gate/up blocks | 6.752 |
| Current total expert weights | 21.095 |
| Reference single original-block expert banks, including TP margins | 11.808 |
| Avoidable difference | 9.286 |

The reference target weights total 13.058 GiB per rank; adding the approved
FP16 MTP weights brings model weights to about 14.281 GiB. These figures exclude
runtime, CUDA graph, cache and prefill temporaries. They must not be compared
with the current 28.501 GiB initialized runner census as if both were weights.

This change reuses `expert_storage="original"`, `dense_storage="original"`,
sharded HC storage and guarded bank allocations from the constrained-memory
implementation. Defaults retain accelerated canonical execution. Original
expert fallback is a capacity comparison, not a claim of equal decode or
prefill speed. HC's existing LL operators read retained local shards; large
batches gather local projected rows and columns. The current HCX implementation
requires dense checkpoint matrices and reports rejection when only shards
remain. A compact HCX representation needs its own admission evidence.

The next model contract uses memory utilization 0.5, FULL graphs, TP4, MTP4,
32768 input tokens and a 16384-token scheduler chunk. Historical QSA K/V remains
host-backed; PLE retains the existing disk cascade with bounded resident rows.
The benchmark now accepts memory utilization directly and leaves the explicit
KV byte override unset by default. Passing a fixed KV byte override bypasses
the automatic memory-utilization budget and cannot prove this capacity contract.
Actual device peak, graph allocations and allocator retries must be recorded.

A separate GPU staging primitive reconstructs the existing canonical IQ3_XXS,
IQ3_S and IQ2_S packet layouts from original blocks. Its test compares every
prepared byte against the packaged Converter, including signed zeros, small
coefficients and replay after replacing the source bank. All six device cases
pass in the installed `5d33064de9` wheel on V100-SXM2-32GB. Staged model dispatch
now retains original IQ banks and borrows one canonical gate/up working area
for prefill. Small-M raw/dp4a routes and canonical FP16 prefill arithmetic are
unchanged. The installed `81e76e7384` wheel passes all nine staging checks:
the six primitive cases plus three compiled FFN cases at E512/N160/K2560.
Two successive IQ formats reuse the pool, M5/M20 graph replay responds to
changed activations/routes, and M512 matches resident canonical execution
bitwise. Extra peak allocation stays below 48 MiB, rejecting large compiler
clones; the M5/M20 dp4a route leaves poisoned staging storage untouched.
The full-model memory and throughput comparison remains pending.

The next baseline raises the four V100 power limits to 300 W, sets the context
limit to 262144 tokens, and enables prefix caching. The timed workload remains
32768 input tokens with a 16384-token scheduler chunk and utilization 0.5.
Every warmup, timed request and profile starts with a successful idle prefix
cache reset. Completed outputs must report zero cached tokens; a hit or missing
cache evidence rejects the cold-prefill measurement. Five unprofiled repeats
report median, range and coefficient of variation, with a 5% stability bound.
Compiler and filesystem caches remain warm; these measurements describe cold
prefix filling, not cold process startup or cold disk I/O.

Host-prefill staging now slices the request block table to its actual visible
history before testing scratch capacity. Unused columns reserved for the 256K
context limit must not force a 32K request into the scalar fallback. The view
retains original storage and stride, and M5/M20 graph table geometry remains
unchanged. Fifteen focused CPU checks pass for cold-prefix evidence and this
capacity/view policy; device and full-model checks are still required.

The reusable gate/up pool has two disjoint slots. At 512 experts, local width
160 and K2560, code storage is 100 MiB and group metadata is 100 MiB. IQ3 uses
eight-byte group32 metadata and IQ2 uses four-byte group16 metadata, so both
fit the same backing allocation. A CPU storage test checks the exact 200 MiB
total, cross-format aliasing and independent gate/up offsets. Nine Triton
interpreter cases at K256/K512/K2560 agree bytewise with an independent
canonical-layout oracle. Interpreter agreement is not CUDA qualification.

Prefill dispatch must stage once before the internal expert row chunks and
consume the workspace inside the opaque operator. Passing large mutable
scratch views through compilation can introduce clones/copybacks and invalidate
the prepared pointer tables. The intended workspace is private to the operator,
retained by the model, and shared only between serialized forward calls. Device
tests must cover compilation, changed-input replay, and two successive layers
with different IQ formats before model admission.

The current model comparison retains GPU memory utilization 0.5, FULL graphs,
32768 input tokens and the 16384-token scheduler chunk, with FP16 host history
for both target and draft and the existing PLE disk cascade. It explicitly
disables additional MTP, GGUF expert and HC activation row blocking:
`VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0`,
`sm70_gguf.prefill_expert_chunk_size=0`, and `prefill_hc_chunk_size=0`.
No fixed KV-byte override is supplied. Weight-storage repairs are measured
before selecting any capacity-driven blocking; actual device peak and the
automatic cache budget must explain any remaining admission failure.

The existing FP16 MTP MoE activation chunker was tested separately on the actual
TP4 rank-0 expert weights. M5, M20 and M16384 outputs agree bitwise with
unchunked execution. Changed-input graph replay also passes for M5 and M20.
At M16384 the measured workspace falls from 880 MiB to 292.5 MiB with an
internal 4096-row chunk. Event medians, including restoration of the inplace
input, are 132.720 ms and 209.030 ms respectively. The 76.31 ms slowdown
rejects this mode as a prefill speed optimization; the model baseline leaves
it disabled.

Two failed fixtures are retained. The first reused inputs overwritten by the
production inplace output. The second restored graph-test inputs by multiplying
FP16 values by 0.5 and then by 2, which changes subnormal values: 22 M20 input
elements differ by at most 5.96e-8, producing a 9.54e-7 output difference.
Restoring from an untouched input copy resolves both failures without relaxing
the bitwise comparison or altering the production operator.

39 CPU storage/initialization checks pass, including all four Q2_0 TP boundaries,
exact HC shard reconstruction, shared staging storage and converter-free original
IQ bank loading. This is not yet the installed
32K/FULL memory or throughput result. The existing NVFP4 prefix-prefill report
records 5409 tokens/s at 32K with TP4/MTP4; its 8K scheduler chunk, cache layout
and memory budget differ and must be accounted for in a matched comparison.
