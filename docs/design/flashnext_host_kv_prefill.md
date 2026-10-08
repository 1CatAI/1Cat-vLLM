# Flash-Next host KV prefill on SM70

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
to first token, total request wall and pure decode separately. The requested
6000 input tokens/s corresponds to 5.461 seconds for 32768 tokens.

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

A prior storage audit identified 14.345 GiB/rank of canonical expert banks
plus 6.752 GiB/rank of original gate/up blocks. Both serve selected runtime
routes. Removing one without a replacement would alter the established
small-batch or prefill route; this duplicate storage remains unresolved.

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
