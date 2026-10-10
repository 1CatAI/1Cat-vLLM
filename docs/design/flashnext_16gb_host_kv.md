# Flash-Next IQ3_S on four 16 GiB V100s

## Storage and execution

The compressed checkpoint fits across four devices, but keeping original expert
blocks alongside canonical banks, replicated HC weights, and full-context padded
state tensors can exceed device capacity. This profile keeps original expert
blocks as the authoritative storage and bounds recurrent-state capacity
independently of authoritative host attention history.

- Gate/up use the existing SM70 Q8_1/dp4a lattice decoder for M=1..20, including
  exact M=5 verification and graph-padded M=8.
  IQ4_NL and IQ4_XS use the same four-value lookup as TurboMind. IQ4_XS
  reconstructs each group coefficient with the canonical FP16 rounding boundary.
  Down/unroute reads original
  IQ4_NL or Q2_0 blocks, including TP boundaries inside Q2_0's 64-element blocks.
  FP32 dots and routing accumulation retain the existing FP16 epilogue boundary.
- Uncovered batches retain the shipped native GGUF fallback without a
  persistent second bank of all experts. The original-block MoE route currently
  uses chunked MMVQ; grouped dequantization/BLAS is an eager fallback that needs
  separate prefill qualification. Dense canonical storage remains available to
  preserve qualified small-M projection routes.
- HC matrices are staged on CPU and only their TP4 packs move to the GPU. When
  HCX already prepared a pack, the storage owner reuses it. Larger batches
  recover local checkpoint rows from that same pack for the ordinary collective
  fallback. Target and draft share compressed embedding/head resources.
- Pure TP MoE keeps EP membership metadata without allocating an unused GPU
  device communicator. CPU PLE discovery uses context-free architecture queries.

## Separate block capacities

`qsa_host_kv_state_blocks` reserves a low global block-ID range for recurrent
state. History and compressor blocks use the remaining range. Hashes, reference
counts, copy-on-write, and prefix eviction retain their existing globally unique
IDs; there is no inference-time remapping kernel.

The scheduler admits a request only if both ranges can satisfy their group
allocation demands. Free blocks in the history range cannot mask state
exhaustion. State tensors have a physical block count, so rank-capacity
reconciliation does not shrink their fixed storage alongside logical history.
Prefix checkpoints can occupy idle state blocks and are evicted by the normal
per-range LRU order. Active and speculative states stay protected by reference
counts. The profile requires align-mode state caching and authoritative host KV;
KV connectors and manual logical-block overrides are rejected.

The profile supports one complete 256K request. It does not advertise four
simultaneous complete 256K requests. Reported concurrency is the minimum of the
state and history capacities.

An allocator-only check using the TP4/MTP4 geometry, 36 GDN layers, 13 target/draft
QSA owners, and the replicated PLE state produces 357 logical blocks at 256K.
The device pools need about 532 MiB per rank, including about 301 MiB of recurrent
state. Authoritative FP16 history pools need about 14.45 GiB across the four
workers. These are calculated allocation sizes, not measured process peaks;
hot caches, staging, weights, graph pools and non-Torch allocations are separate.
Startup requires at least 21 GiB of available host memory before loading.

## Disk PLE with bounded caching

`ple_disk_only` retains the mapped compressed table and a sentinel row per rank.
`ple_row_cache_mib` bounds the combined native CPU row-cache payload and keys in
the shared offload process. The cache stores original bytes, so misses and hits
use the same official dequantization. Direct-mapped collisions replace cache
entries without changing requested output order. Allocation is lazy; selected
cold pages are prefetched rather than faulting the entire table into RAM.
The profile uses a 512 MiB total CPU budget and an 8192-token GPU KV hot cache.

## Validation status

CPU regressions cover fixed state capacity at 256K, partition exhaustion and
reuse, shared quota across recurrent groups, historical align-mode asynchronous
state lifetime, exact HC shard recovery, compact expert block boundaries, and
packed-row official dequantization. These checks do not establish GPU speed or
full-model correctness.

GPU test cases cover IQ4_NL/IQ4_XS gate/up, raw down across all TP4 ranks, FP16 and Q8_1
intermediates, M=5/M=20, and changed inputs during CUDA graph replay. Run them
before admitting the constrained-memory route. Then record C1/C4, target outputs,
accepted tokens per round, long prefill, repeated-prefix hits, and tool calls.
Host-KV overhead must be reported separately from the device-KV acceptance
benchmark. The historical 15.53 ms/round measurement is a reference on 32 GiB
V100s and is not a measured result for this profile.

## Startup

Use `examples/deployment/sm70_flashnext_gguf/serve-16gb-256k.sh` with `MODEL`,
`DRAFT`, and `VLLM_PYTHON` pointing to the verified checkpoint and source runtime.
Build native extensions from that source with CUDA 12.8 and SM70, including the
policy ABI in both `_C` and `_moe_C`. Do not reuse private extension binaries.
Apply `examples/deployment/sm70_flashnext_gguf/constraints-cu128.txt` when
installing runtime requirements. It pins XGrammar 0.2.0 for the declared
TVM-FFI 0.1.10 runtime; XGrammar 0.2.8 requires a newer library-loading interface.
The script enables TP4, FP16 MTP4, 256K context, prefix caching, Qwen tool and
reasoning parsing, FULL graphs, prefill chunks of 512, disk PLE, and FP16 host
history. It holds the shared four-GPU and per-GPU locks for the service lifetime.

Collect a storage ledger after target loading, draft sharing, cache allocation,
and graph capture. Distinguish unique weight storage, recurrent state, KV hot
cache, staging, BLAS/workspaces, graph pools, and non-Torch allocations. Capacity
checks must use the actual ledger rather than checkpoint size alone.
