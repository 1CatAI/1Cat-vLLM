# GGUF storage and constrained GPU memory

A checkpoint's size divided by TP size omits replicated parameters, expanded
representations, MTP, CUDA/NCCL state, caches and loading workspaces. PLE's mapped
host tables must be accounted separately from device allocations.

On four 16 GiB V100-SXM2 GPUs, Flash-Next IQ3_S target loading initially used
14.35 GiB of unique device tensor storage per rank. Preparation reached 14.69 GiB
of PyTorch allocations and exhausted device memory before loading MTP. The main
GGUF shard is 51.05 GiB. PLE is supplied by its separate host-resident shard.
CUDA/NCCL and other allocations outside PyTorch reserved storage consumed about
0.95 GiB on ranks without the PLE helper. These are loading measurements, not
inference capacity or performance results.

The checkpoint-loaded per-rank storage ledger was:

| Storage | GiB |
| --- | ---: |
| Original expert gate/up | 6.9305 |
| Expert down, including expanded Q2_0 | 5.3833 |
| Replicated HC matrices | 1.2152 |
| Expanded FP16 token embedding | 0.2960 |
| Original Q6_K LM head | 0.1214 |
| Other tensors | 0.3996 |

The kernel configuration supplies optional storage policies. Defaults retain the
existing accelerated paths:

- `sm70_gguf.expert_storage="original"` keeps original blocks rather than canonical
  expert banks and retained original copies. The packaged native fallback runs
  these banks; this is a memory tradeoff, not an equal-speed claim.
- `sm70_gguf.dense_storage="original"` keeps original quantized projection rows and
  skips canonical representations. Floating projections still become ordinary
  FP16 linear layers. LM heads follow the same storage policy.
- `sm70_gguf.embedding_storage="original"` retains compressed token embedding rows
  and dequantizes only requested rows. IQ4_XS embedding storage falls from 303.125
  MiB to 80.52 MiB per rank without changing integer codes or coefficients.
- `hc_weight_storage="sharded"` retains only the existing lossless TP4 HC packs.
  Admission requires qualified HC LL transport, FP16 HC4/H2560/R320 matrices,
  and no LoRA. Small batches reuse the packaged LL operators. Larger batches
  gather projected local rows and columns; replicated matrices are not rebuilt.
  Full checkpoint matrices are staged on CPU; only the final local packs are
  copied to CUDA. Packing after a full GPU copy left about 973 MiB of allocator
  fragmentation during the target-to-draft loading transition.

Q2_0 down projections split K640 into K160 at TP4. Instead of expanding to Q4_1,
original storage retains the three overlapping K64 blocks per rank and pads the
activation's boundary elements with zeros. Rank 1 and rank 3 have 32 leading
zeros. Physical K is 192; source integer codes and coefficients are unchanged.
Nine expert banks shrink from 125 MiB to 67.5 MiB per rank, saving 517.5 MiB.
The retained boundary blocks slightly duplicate checkpoint bytes across ranks.

Native expert banks allocate their zero safety tail before checkpoint rows are
copied. Preparation reuses the allocation rather than copying a 114--126 MiB
bank at the memory limit. The safety tail belongs to the final allocation, not
each row.

For original dense weights, `sm70_gguf.dequant_workspace_bytes` caps each
fallback dequantization tile (32 MiB by default). Output rows are written into a
single preallocated output; a full FP16 vocabulary matrix is not materialized.

Fixed/default text RoPE caches follow the engine's maximum context instead of
allocating the checkpoint's full 262K range for a 2K engine. Dynamic/YaRN and
multimodal caches retain their existing behavior. Under PP1 MTP, draft embedding
and head constructors use placeholders before sharing the target modules. The
original BF16 draft checkpoint is loaded through the already approved FP16 path.

SM70 all-gather uses the initialized PyNccl communicator for supported CUDA
dtypes on that device. This avoids creating a second ProcessGroupNCCL GPU
communicator on the first large-batch HC or vocabulary gather. Other devices and
unsupported layouts/dtypes retain the previous fallback. Capability discovery
uses platform metadata for FLA, FA2 and shared-expert/GDN dispatch. GDN norm
parameters follow the surrounding CPU/meta/CUDA construction device instead of
forcing a CUDA allocation. A complete real Flash-Next CPU helper meta build
passed with CUDA initialization forbidden, recording zero initialization calls.

## Measured loading and profiling

The test uses four V100-SXM2-16GB GPUs with full NV2 peer connectivity, driver
580.178.04, CUDA 12.8, Torch 2.10.0+cu128, TP4, IQ3_S target, FP16 MTP4,
FP16 attention KV and FP32 recurrent state. The initial capacity probe is eager,
maximum context 2048, token budget 128 and maximum requests 4. PLE packed tables
remain in pinned host memory. This is distinct from the FULL-graph speed baseline.

| Phase | Model tensor storage per rank | Allocator observation |
| --- | ---: | --- |
| Initial target checkpoint loaded | 14.346 GiB | Failed during preparation before MTP |
| Compact target prepared | 13.058 GiB | CPU-staged HC removes about 973 MiB of fragmentation |
| Compact target plus FP16 MTP4 | 14.281 GiB | Weights and startup profiling complete |

The measured compact target decomposition is:

| Storage | GiB per rank |
| --- | ---: |
| Original expert gate/up | 6.9305 |
| Original expert down, including TP boundary blocks | 4.8779 |
| Sharded HC packs | 0.3275 |
| Packed token embedding | 0.0786 |
| Packed LM head | 0.1214 |
| Other tensors | 0.7234 |

Each GPU exposes 15.766 GiB to CUDA. Before the context-free discovery fix,
profiling completed with about 0.444 GiB driver-free on ranks 1--3 and only
0.084 GiB on rank 0. The PLE helper's unnecessary CUDA context consumed another
368 MiB on GPU0. The installed-wheel CPU helper subsequently completed its
actual filtered PLE weight load with CUDA uninitialized and zero NVML allocations,
confirming that the avoidable context has been removed. At utilization 0.93
the earlier calculated KV budget was negative
(-0.08 GiB), so no generation result was produced. Weight-loading success alone
is not a usable inference capacity result. The subsequent installed-wheel capacity run used utilization 0.938. It completed
weight loading, profiling and cache allocation, with 0.40 GiB cache per rank.
The engine reported 3157 cache tokens and estimated 1.54 requests at a 2048-token
maximum length. Peak NVML usage was 16113 MiB on every card. Three short natural
prompts ended normally, including the answer `85` to `17 * 5`.

The C1 fixed-input probe produced 72.84 ms/round, 4.676 emitted tokens/round and
64.19 tokens/s. Its synthetic arithmetic workload accepted 201/220 draft tokens
(91.36%). This is a capacity route that disables extra packed copies and graphs;
it is not comparable to the 17.406 ms FULL-graph speed control or a natural-prompt
acceptance comparison. At C4, requests queued and no fixed-width C4 interval was
observed. The benchmark correctly rejected that cohort. Full 2K-input prefill,
long-context capacity and C4 remain unqualified.

Startup took 533 seconds. A CPU PLE prefault of the entire 26.82 GiB mapped table
accounted for 249 seconds even though every registered rank retained its rows
in pinned/device storage and no disk rows were served. The CPU worker now skips
weight prefault only after placements establish an empty disk tier; unregistered
and genuinely CPU-served table configurations retain prefault. Five focused
CPU checks cover both the no-row and real-disk-row cases. The startup-time effect
of that fix has not yet been measured in a complete model run.

A separate four-card initialization ledger measured 392 MiB outside PyTorch
reserved memory for TP communicator creation and 244 MiB for EP communicator
creation, although EP was disabled. Together with the CUDA context and lazy
libraries, this explains most of the roughly 0.95 GiB non-PyTorch cost. Smaller
standard NCCL buffers are an unqualified candidate for capacity; they must be
measured for both memory and collective latency before adoption.

## Correctness checks

CPU checks cover block retention, exact reconstructed TP boundary weights and
padded projections, byte-identical embedding rows, exact HC shard reconstruction,
MTP I/O placeholders and fixed-RoPE prefix equality. Nine additional discovery
checks enforce imports without CUDA initialization and per-device FA2 selection.

Four-card operator checks cover HC M1/M5/M20/M32, Q2_0 TP4 expert M5/M20 and
packed embeddings. HC mixing relative errors were below 1.6e-6; injection errors
below 1.1e-4. IQ4_XS embedding rows were bitwise equal to the official FP16
reference. Bounded Q6_K head dequantization at chunk boundaries had relative
error 1.01e-6. Twelve all-gather layout/dtype cases per rank and captured graph
replay were byte-identical. The isolated graph fixture releases captured graphs
before destroying communicators.

Native M5 projection fallbacks use the previously approved Q8_1 activation
quantization (observed relative error about 0.5--0.6%); M20/M32 comparisons were
about 1e-5. No additional weight precision reduction is introduced. Short natural outputs
and MTP counters now pass. Matched acceptance comparisons, full-context prefill,
C4 capacity and speed remain validation gates; short eager capacity results do
not qualify FULL-graph decode performance.
