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

CPU tests check original-block retention, exact reconstructed TP boundary
weights and padded projections, byte-identical packed embedding rows, and exact
HC shard reconstruction. Complete MTP4 loading, target output quality, available
KV memory and speed on four 16 GiB GPUs still require device validation. A short
eager capacity probe does not qualify the existing FULL-graph performance path.
