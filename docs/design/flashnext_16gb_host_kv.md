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
  uses chunked MMVQ for unmeasured inputs. At the measured 512-token prefill
  chunk, the capability registry selects the shipped MMQ operator for
  IQ3_XXS/IQ2_S/IQ3_S/IQ4_XS gate/up and IQ4_NL/Q2_0 down. Down counts 5120
  routed input rows. Partial chunks retain MMVQ. Dense canonical storage
  remains available to preserve qualified small-M projection routes.
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
QSA owners, and the replicated PLE state uses 29 state blocks, the minimum
derived for one request with MTP4. With host-backed compressed indexer history, device pools reserve 272.9 MiB
of recurrent state per rank. Keeping indexer history on device instead adds
229.2 MiB per rank. Larger concurrency requires a newly
derived state quota. Authoritative FP16 history pools need about 15.22 GiB
across the four workers. These are calculated allocation sizes, not measured process peaks;
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
intermediates, M=5/M=20, and changed inputs during CUDA graph replay. Independent
fused-Q8 gates (30), raw-down gates (32), host-KV collision/rewrite/graph gates
(20), and storage-accounting gates (13) passed on four NV2-connected V100-SXM2
16 GiB devices with CUDA 12.8 and Torch 2.10.0+cu128. These operator checks do
not establish full-model fit, acceptance, or performance. Record C1/C4, target outputs,
accepted tokens per round, long prefill, repeated-prefix hits, and tool calls.
Host-KV overhead must be reported separately from the device-KV acceptance
benchmark. The historical 15.53 ms/round measurement is a reference on 32 GiB
V100s and is not a measured result for this profile.

### Original-bank prefill qualification

Real TP4 slices from layers 0, 1, 17 and 47 were tested at M=512, top-k=10,
512 experts, hidden size 2560 and local intermediate size 160. Layer 0 and
layer 1 also covered all four TP ranks, including Q2_0 boundary padding.
Each comparison reconstructs official FP32 weights and independently quantizes
the activation to Q8_1; it retains FP16 projection/activation boundaries.
Changed activation, expert IDs and routing probabilities produce exactly the
same output in captured and eager execution. No persistent decoded bank is
allocated. The table measures a complete single-rank expert layer, excluding
TP reduction, other model operators and host-KV work.

| Gate/up and down formats | Chunked MMVQ control | Prepared MMQ path |
| --- | --- | --- |
| IQ3_XXS / IQ4_NL | 81–84 ms | 19–21 ms |
| IQ2_S / Q2_0 | 75–87 ms | 19–21 ms |
| IQ3_S / IQ4_NL | 80–91 ms | 17–19 ms |
| IQ4_XS / IQ4_NL | 87–89 ms | 17–20 ms |

Across the ten layer/rank checks, maximum relative L2 error against the
independent reference was 0.00507 and maximum absolute error was 9.06e-6.
Allocated expert-layer storage and temporaries were 211–341 MiB; allocator
reservations were at most 646 MiB. Decode and partial-prefill capabilities
are unchanged; full-model checks are still required.

Rejected candidates are retained as negative evidence: host-sorted grouped
execution took 330–380 ms per layer. Bounded FP16 down chunks initially failed
the K=160 eligibility predicate; zero-padding to K=192 made them eligible
but took 169–183 ms versus 75–88 ms for original MMVQ. Neither is enabled.

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

### Single-bank floating router and HC lifetime

The capacity profile selects `sm70_router_weight_storage=row_major`. The M5/M10
batched FP16 router reads its existing contiguous checkpoint matrix, while
M1, M20 and prefill retain the original dense fallback. The default `dual`
layout remains unchanged for other profiles. Across 48 real router matrices,
packed and row-major M5/M10 outputs are bitwise equal; changed-input graph
replay is also exact. Removing the second bank saves 120 MiB per rank. Same-GPU
M5 chain timing was close to the packed control; M10 showed greater variability
and a regression in one repeat. This is a capacity option, not a speed claim.

After HC sharding, the generic dense provider's borrowed checkpoint reference
is released. Both the LL and larger-batch schedules consume the shard bank.
A lifetime test proves the former CPU parameter can be collected and the
sharded fallback reconstructs the correct matrix. Full-process host savings
must still be measured after loading.

### Exact MTP storage and prefill peak

The 16 GiB profile enables `sm70_mtp_lossless_storage`. BF16-origin expert
values are first converted by the existing FP16 loading rule. Normal values
have three zero low mantissa bits; subnormal magnitudes and signed zero are
retained in full. A checked 13-bit representation stores 32 values in 52 bytes.
Every input value must be exactly representable, otherwise loading fails.
This saves 225 MiB per rank (975 versus 1200 MiB for draft experts), with no
change to weight values, sequential FP32 accumulation or FP16 epilogues.
The complete projection/SiLU/down/reduction chain is bitwise equal to the
FP16 reference for M1, M5 and M20; original FP16 kernel regressions also pass.
An isolated four-rank real-checkpoint load reports 1,087,571,472 bytes
of registered draft storage per rank, including 1,022,361,600 expert bytes.
All four ranks compare the selected real experts bitwise with BF16-to-FP16
checkpoint values; complete chains and changed-input graphs agree exactly.
Same-process ABBA chain timings are below. M1/M5 controls use the existing
FP16 native kernel, M4/M20 the tuned BM2 Triton fallback. Timings exclude
attention, HC, target work and inter-rank output reduction.

| M | FP16 control, ranks 0–3 | Lossless storage, ranks 0–3 |
| --- | --- | --- |
| 1 | 49–90 µs | 70–95 µs |
| 4 | 285–322 µs | 165–169 µs |
| 5 | 114–148 µs | 193–229 µs |
| 20 | 1010–1118 µs | 595–671 µs |

M1 increases by 5–21 µs per draft step. This is a measured capacity tradeoff,
not an end-to-end speed claim. Whole-model acceptance remains pending.

Original-bank expert reduction now accumulates routing-weighted outputs in
FP32 without materializing FP32 copies of the full routed tensor. At M512,
top-k 10 and hidden size 2560, extra allocated memory is 2.5 MiB versus
100 MiB for the previous expression. This removes 97.5 MiB of profiling peak.

`qsa_host_indexer_history` places compressed keys in mapped pinned memory,
while recurrent state remains on device. GPU writers and score kernels use
the same FP16 values and block layout. Tests at 8K and 256K compare all visible
scores, changed inputs and graph replay bitwise against device storage. The
main attention history still has the 8K GPU hot cache; compressed indexer
history currently uses direct UVA. Its PCIe cost, especially at long context,
is separate from the historical device-KV benchmark.

The fourth full startup loaded target and FP16 draft successfully, then failed
during M512 profiling before cache allocation or graph capture. Its target
registered storage was 14,134,449,834 bytes per rank; the old reduction needed
an additional 50 MiB allocation. This is capacity evidence, not serving or
acceptance evidence. Startup-only CPU allocator reclamation releases unused
checkpoint-conversion pages before allocating pinned histories.

### Allocator reservation accounting

The fifth startup completed target and exact MTP loading at 14.36 GiB per
rank and passed M512 profiling. Its admission budget then reported only
0.02 GiB for KV versus 0.27 GiB required. The warmup residual calculation
used reserved allocator bytes rather than active tensor bytes, charging idle
space in partially occupied segments as persistent warmup tensors. Snapshots
now record both values; the residual uses allocated bytes. Reserved memory
continues to determine non-Torch usage, preserving the CUDA accounting identity.
Regression fixtures retain a real 64 MiB warmup tensor and a 256 MiB simulated
idle pool independently, and cover graph reserve on/off. Real startup logs
report weights, activation peak, live warmup tensors, idle reservation,
non-Torch usage and graph reserve separately. Physical fit still requires
cache allocation, actual graph capture and prefill peak checks.

The source runtime also requires TileLang 0.1.10 from the CUDA requirements.
A missing package caused the FlashQLA prefill warmup import to fail; the source
launcher now checks its presence before loading weights.

### Pinned history allocation

The sixth startup passed GPU KV admission (0.35 GiB available versus 0.27 GiB
required), then hit its 27 GiB host-memory service limit while allocating
history. The CUDA pinned caching allocator rounds individual allocations to
powers of two. Thirteen 295,796,736-byte main history banks and thirteen
18,487,296-byte compressed banks request 3.805 GiB per rank, but independent
allocation reserves 6.906 GiB per rank, or 27.625 GiB across TP4.

Both model runners now place host cache banks in one 256-byte-aligned pinned
pool per worker. Individual views preserve sharing, mapped storage ownership,
zero initialization and graph pointers. The same payload rounds to 4 GiB per
rank, saving 11.625 GiB of host reservations across TP4. This changes storage
allocation, not KV values or block addressing.

A four-rank allocation test using the complete 256K geometry passed layer
boundary checks and changed-input graph replay. Process-group memory peaked
at 18.7 GiB without swap, and host available memory remained 9.64 GiB.
Three targeted pool tests also passed. These tests establish pool capacity
and ownership; whole-model startup and acceptance remain pending.

The original TileLang prefill route also passed a 512-token comparison with
an independent FP32 recurrence: maximum absolute error 1.69e-6, relative L2
4.45e-4, and changed-input graph output/state identical to eager. Its SM70
NVRTC callback needs the CUDA runtime, NVCC and CCCL pip header packages even
when a system CUDA toolkit is installed. Install the two additional header
packages with the CUDA 12.8 constraints and retain the declared Torch runtime:

```bash
python -m pip install -c examples/deployment/sm70_flashnext_gguf/constraints-cu128.txt \
  nvidia-cuda-nvcc-cu12 nvidia-cuda-cccl-cu12
```

The source launcher checks these header packages before loading weights.
