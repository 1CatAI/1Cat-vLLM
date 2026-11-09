# Qwen4Exp QSA decode context parallelism (DCP2)

Target-only decode context parallelism for the Qwen4Exp QSA checkpoint on
TP4 V100. The 12 target QSA layers shard their main K/V over two DCP ranks.
The selector's compressed keys, the compressor ring, the GDN and PLE states
and the single MTP draft QSA layer stay replicated, so every rank still
scores and selects over the whole sequence and only the sparse attention is
distributed. DCP1 is unchanged; QSA supports DCP1 and DCP2.

## Related changes

- **E4M3 main KV with MTP** is #664. It merges with this change without
  conflicts, and E4M3 KV + MTP3 + DCP2 was validated on a stack that
  contains it (see Validation).
- **Grouped CPU offload** for this hybrid model comes from #617 and #598.
  This change adds the DCP pieces: the grouped CPU pool is partitioned by DCP
  owner spans, the unpartitioned tiering pool accepts groups with different
  block sizes, and grouped tiering accepts decode context parallelism.
- **GDN metadata.** DCP2 has six GDN state groups instead of three (see
  Layout). With #684 and #699, native MTP builds the GDN metadata once per
  step for all groups; without them every group builds its metadata every
  decode step.

## Cache ownership

| Cache | DCP2 |
| --- | --- |
| Target QSA main K/V (12 layers) | sharded, `block_size // 2` slots per rank |
| Selector compressed keys, compressor ring | replicated |
| GDN recurrent states, PLE | replicated |
| MTP draft QSA layer | replicated |

- `KVCacheSpec.dcp_sharded` and `global_block_size()` give each owner its own
  token span. The single-type managers, both coordinators, block-size
  resolution and the offload spec use it instead of a blanket DCP factor. The
  hybrid coordinator accepts DCP when every sharded group is full attention;
  it normalizes the prefix-lookup specs to global tokens and keeps the worker
  specs local, so physical cache strides do not change.
- The v1 and v2 GPU runners build block tables and slot mappings per group:
  rank-local slots for sharded groups, the global table for replicated ones.
  Dummy (profile and capture) batches are marked explicitly, because under DCP
  a PAD slot can also mean that another rank owns the token.
- The QSA metadata builders expand the replicated selector and draft tables
  from the rank-local table span. The replicated draft maps its KV writes and
  slots through its own split kernel block table.

## Layout

The CSA+linear pool sizes a block so that one physical page holds one
recurrent state (817,152 bytes per rank with MTP3). A block spans
`cache_config.block_size` global tokens at every DCP size, and the platform
aligns the block size to kernel block x DCP so each rank's share stays kernel
aligned.

A sharded target layer's page is then half a state page, so the allocator
packs two sharded target layers into one physical page, interleaved one
kernel block at a time (`KVCacheTensor.packed_members`). Each member is a
strided view of the shared tensor; the cache writer and the QSA kernels
address by block stride and need no change. The v2 GPU runner builds the
views, the offload worker registers a packed page once, and the v1 runner and
pipeline parallelism reject packed pages.

| KV dtype | Block (DCP1 and DCP2) | DCP2 target slots per rank | Physical page |
| --- | --- | --- | --- |
| E4M3 | 1,600 | 800 | 819,200 B |
| FP16 | 800 | 400 | 819,200 B |

Seven physical owners per block (DCP1: thirteen) hold the recurrent states,
which gives six GDN state groups instead of three.

## Attention

1. All-gather the queries over the DCP group.
2. Localize the selected token IDs to this rank's slots, compacted to the
   front of a separate buffer (MTP steps reuse the global selection). With
   interleave 1 only the first `(topk // ratio + 1) * (ratio // dcp)` columns
   can hold an owned token (1,026 of 2,051 at DCP2), and only those are passed
   to the kernel.
3. QSA sparse attention returns an FP32 partial output and a base-2 LSE.
4. Combine over the ranks with one all-to-all (the Qwen4Exp default through
   `ParallelConfig.set_dcp_defaults()`; an explicit `dcp_comm_backend` wins),
   round to the output dtype as DCP1 does, and apply the output gate with the
   Triton kernel of DCP1's page4 route.

The QSA metadata builders skip FlashAttention's DCP context lengths, which
QSA never reads.

## Validation

Gates, each against DCP1 on the same code and fresh servers:

- Accuracy: six fixed MTP prompts (tokens, accepted and drafted counts), five
  code repeats, and a 30,030-token prompt (first request, prefix-hit repeat,
  changed suffix) must match exactly. DCP1 is also compared with DCP1 before
  this change.
- Regressions: prefix eviction under pressure and recovery (from CPU where
  offload is enabled, otherwise by recomputation), a mixed 30K prefill with
  decode, and five independent 30K admissions with GPU memory sampling.
- No errors, deadlocks or out-of-memory in any server log.

Results on TP4 V100 with MTP3, CUDA graphs and prefix caching. Capacity is
GPU KV tokens at `max_model_len` 32,768 / 2 sequences / GMU 0.90, and at
262,144 / 4 sequences / GMU 0.96; decode is C1 output tokens per second.

| Configuration | Gate | 32K capacity DCP1 -> DCP2 | 256K capacity DCP1 -> DCP2 | Decode DCP1 / DCP2 |
| --- | --- | --- | --- | --- |
| E4M3 KV, with #664, #598 and GDN sharing | identical | 396,336 -> 493,244 | 775,096 -> 1,178,337 | 77.9 / 75.8 |
| FP16 KV, same stack | identical | 284,341 -> 404,706 | 433,401 -> 718,015 | 72.9 / 70.3 |
| FP16 KV, DCP2 core on main before #699, no offload | identical | 284,341 -> 404,706 | - | 69.9 / 63.8 |
| FP16 KV, DCP2 core + #699 on main, no offload | identical | 284,341 -> 404,706 | - | 72.5 / 70.3 |
| E4M3 KV without MTP, DCP2 core on main | DCP2 only (see below) | - -> 1,066,780 | - | - |

The regressions pass in every configuration with outputs identical to DCP1
(on main without offload, pressure evicts the prompt and recomputation
reproduces its output). DCP2 on main gives bit-identical logprobs to DCP2 on
the full stack, and DCP1 with this change gives bit-identical logprobs to DCP1
on plain main. Without #699 the six GDN groups each build their metadata every
MTP3 decode step, which costs DCP2 8.6% against DCP1; with it the gap is 3.0%.

With E4M3 KV and no MTP, DCP2 serves on main (block 1,568). DCP1 fails there
before serving, independently of this change: main allocates an FP16 XQA
decode workspace that the E4M3 kernel rejects, which #664 fixes.

With FP16 KV, three benchmark prompts continue differently under DCP1 and
DCP2. At the first differing token each arm's choice is the other's
runner-up, with top-2 margins of 0 to 0.16 nats (an exact tie in DCP1 for
one of them). Both continuations answer correctly, and the gate prompts are
unaffected. Per-token logprobs of DCP1 and DCP2 differ by at most 0.05.

## Limits

- DCP2 prefill has no counterpart of DCP1's XQA page4 route, so a repeated
  2K prompt is 16-18% slower to first token.
- Batches the shared GDN metadata path cannot express (prefill, mixed) still
  build the metadata of each of the six GDN groups.
- Packed pages reject pipeline parallelism and the v1 GPU runner.
