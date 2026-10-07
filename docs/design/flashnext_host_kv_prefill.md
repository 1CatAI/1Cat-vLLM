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
