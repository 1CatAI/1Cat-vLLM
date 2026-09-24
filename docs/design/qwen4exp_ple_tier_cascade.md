# Qwen4Exp PLE overflow cascade

## Decision

The Qwen4Exp PLE table is large: 47.684 GiB of FP8 E4M3 rows for
Qwen3.8-Flash-Next, 23.84 GiB per tensor-parallel rank at TP2, all of it on the
first pipeline stage. Before the cascade a pre-Ampere deployment could keep the
rank's shard in device memory with a pinned-host remainder, hold the whole
table in the PLE offload worker, or map it from disk in that worker. A host
with little RAM loses with each of them: the split pins every rank's
remainder, the worker's table needs the full 47.7 GiB of host memory, and the
disk lane reads every row from the checkpoint.

The cascade fills three tiers per rank, fastest first, and places every row
exactly once:

| Tier | Holds | Served by | Budget |
|---|---|---|---|
| 1 device | rows up to the measured device budget | the compute rank, gathered in its graph | measured at load |
| 2 host | pinned host share | the compute rank, through the UVA view | `VLLM_QWEN4EXP_PLE_HOST_GIB` per rank |
| 3 disk | the checkpoint shards, read in place | the PLE offload worker, mmap reads | none, enabled by `VLLM_QWEN4EXP_PLE_DISK=1` |

The table is addressed by hashes, so every row is equally likely to be read and
the split points carry no meaning beyond capacity. The disk tier reads the
checkpoint where it is: nothing is copied or written back.

The feature is off by default. Without its variables the pinned-host split, the
whole-table worker, the disk lane and the hybrid lane are unchanged.

An earlier version also had store cards (GPU memory of a spare card or of the
other pipeline stages). Measured under PP4 it served decode no faster than the
disk tier and saved only 0.3 to 2.5 s of prefill with a cold page cache, at
the cost of an extra startup phase in the GPU worker. It was removed; the code
is archived on the fork branch `archive/ple-store-cardlist`.

## Configuration

| Variable | Meaning |
|---|---|
| `VLLM_QWEN4EXP_PLE_DISK` | `1` reads the rows beyond device and host from the checkpoint and starts the cascade. |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | Pinned host share per rank. With the cascade it is used only for rows the device cannot hold; `0` pins nothing. |
| `VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB` | Device memory the measured budget keeps free (default 8 % of the card, at most 4 GiB). |
| `VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB` | Host memory the host-share check keeps free (default a quarter of the host). |
| `VLLM_PLE_DISK_RELEASE_PAGES` | `1` unmaps the checkpoint pages the worker read after every disk gather (disk lane and cascade). Recommended on hosts with little RAM; off (default) keeps them mapped. |

```bash
VLLM_QWEN4EXP_PLE_HOST_GIB=0 VLLM_QWEN4EXP_PLE_DISK=1 \
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 --pipeline-parallel-size 4 ...
```

Refused before any worker starts: the cascade together with
`VLLM_SM70_QWEN38_HYBRID_PLE` or `VLLM_PLE_DISK_OFFLOAD`, a model without PLE
layers, and an explicit host share that exceeds available host memory minus the
host reserve, divided by the TP size.

## Placement

`plan_ple_placement` in `vllm/models/qwen4_exp/common/ple.py` splits one rank's
rows into `PLEPlacement(vram_rows, host_rows, disk_rows)`:

1. device rows up to the measured budget: utilization budget minus the memory
   already allocated, the KV cache for `max_model_len` and the device reserve,
2. host rows up to the host share,
3. the rest on disk, or a startup error that names the knobs when the disk tier
   is not enabled.

Without the cascade the host share is taken first and the device holds the
rest, as before. A rank needs at least one resident row, because both resident
gathers always run.

## Per step

Compute rank (`Qwen4ExpNGramEmbedding` and `Qwen4ExpPinnedHostEmbedding`):

- gathers the device and host rows as in the pinned-host split; ids beyond the
  resident rows read row 0 there,
- waits for the worker with `ple_offload_wait` inside the CUDA graph,
- dequantizes the worker's raw FP8 bytes with the same op and scale as the
  resident gathers and merges them by rank-local id with `where`,
- masks foreign vocabulary ranges and runs the tensor-parallel all-reduce.

PLE offload worker (`_remote_lookup`):

- computes the n-gram ids once per data-parallel rank,
- fills one output buffer for all tensor-parallel ranks: the ranks' disk
  segments are disjoint in the global id space, so each rank takes only its
  own slots,
- reads the rows with the disk lane's mmap reader (`_gather_mapped_rows`:
  sorted unique ids per shard, `MADV_RANDOM`, thread pool). With
  `VLLM_PLE_DISK_RELEASE_PAGES=1` it then unmaps the pages it read
  (`madvise(MADV_DONTNEED)` on the private, file-backed shard mappings). The
  pages stay in the page cache; unmapped, the kernel can drop them first under
  pressure instead of swapping other processes out. Without the switch they
  stay mapped, which saves the re-mapping on repeated reads on a host with RAM
  to spare. Only shards the loader recorded as file-backed are released, since
  on anonymous memory `MADV_DONTNEED` discards data.

Greedy outputs are identical to the pre-change path because every row is
dequantized by the same kernel arithmetic, wherever it was stored.

## Loading and registration

- Each rank copies only its device and host rows from the checkpoint
  (`copy_ple_embedding_shard_tiers_`), with `copy_` directly: staging through
  `.to()` left one checkpoint shard (0.37 GiB) cached per card.
- The worker keeps the 128 checkpoint shards file-backed, like the disk lane,
  and holds no anonymous copy of the table.
- Each rank registers a `PLERemotePlacement` (vocabulary range, resident rows).
  The worker refuses missing or inconsistent registrations and plans its disk
  segments from them.
- Under pipeline parallelism only ranks that own a `PleOffloadLayer` build a
  connector, and the worker drops an inherited `VLLM_PP_LAYER_PARTITION`, which
  would make `get_pp_indices()` refuse its single-stage world.

## Validation

`nvidia/Qwen3.8-Flash-Next-NVFP4`, TP1 x PP4 (12 layers per stage) on 2x Quadro
RTX 8000 + 2x Tesla V100, PCIe Gen3 x4, 30 GiB host RAM, checkpoint on an NVMe
SSD in a USB enclosure, MTP k=4, `--max-model-len 262144`. Page cache of the
checkpoint evicted before the requests (`posix_fadvise(DONTNEED)`); twelve
different slices of real text (8k to 17k tokens) — word-list prompts share
most n-grams and hit the cache far more often.

| Device | Host | Disk | Prefill, first six texts | Decode | MemAvailable | Swap-out |
|---|---|---|---|---|---|---|
| 18.9 GiB | 0 | 28.8 GiB | 8.9–11.2 s | 36–51 tok/s | 17.7 GiB | 0 |
| 18.9 GiB | 3 GiB | 25.8 GiB | 8.8–11.1 s | 35–45 tok/s | 13.3 GiB | 0 |
| 0 | 0 | 47.7 GiB | 10.0–12.8 s | 37–53 tok/s | 13 GiB | 0 |

For comparison, the whole table in GPU memory (store cards, removed): 7.7 to
12.5 s prefill, same decode. The disk tier costs about 1 s of prefill with a
cold page cache and less as the cache fills; 36,000–118,000 page faults (8k to 17k tokens) per
request go to the SSD.

Unmapping, PP4 with a 12 GiB host share and 16.8 GiB on disk, twelve texts:
worker RssFile 112 → 1,939 MiB and 2.1 GiB of other processes swapped out
before; 105–109 MiB and 96 MiB with it, prefill and decode unchanged.

## Limits

- The offload worker holds memory on each first-stage card (184 and 212 MiB on
  a TP2 rig) outside the utilization budget, as described in #530.
- The disk tier is read on every step. Prefill pays for it with a cold page
  cache; decode reads too few rows per step to notice.
